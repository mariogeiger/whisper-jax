"""Pure JAX Whisper implementation using Flax NNX."""

import jax
import jax.numpy as jnp
from flax import nnx


class MultiHeadAttention(nnx.Module):
    """Multi-head attention with optional causal masking."""

    def __init__(
        self, embed_dim: int, num_heads: int, is_causal: bool = False, rngs: nnx.Rngs = None
    ):
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.is_causal = is_causal

        self.q_proj = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.k_proj = nnx.Linear(embed_dim, embed_dim, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.out_proj = nnx.Linear(embed_dim, embed_dim, rngs=rngs)

    def __call__(
        self, x: jax.Array, kv: jax.Array | None = None, deterministic: bool = True
    ) -> jax.Array:
        """Apply attention. Use kv for cross-attention, None for self-attention."""
        B, L, _D = x.shape
        if kv is None:
            kv = x

        # Project and reshape to (B, H, L, head_dim)
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(kv).reshape(B, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(kv).reshape(B, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Attention scores
        attn = (q @ k.transpose(0, 1, 3, 2)) / jnp.sqrt(self.head_dim)

        # Causal mask
        if self.is_causal:
            mask = jnp.triu(jnp.ones((L, L)), k=1)
            attn = jnp.where(mask[None, None], -1e10, attn)

        attn = jax.nn.softmax(attn, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(out)


class FeedForward(nnx.Module):
    """Two-layer feed-forward network with GELU."""

    def __init__(self, embed_dim: int, ffn_dim: int, rngs: nnx.Rngs = None):
        self.fc1 = nnx.Linear(embed_dim, ffn_dim, rngs=rngs)
        self.fc2 = nnx.Linear(ffn_dim, embed_dim, rngs=rngs)

    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        return self.fc2(jax.nn.gelu(self.fc1(x), approximate=False))


class EncoderLayer(nnx.Module):
    """Encoder layer: LayerNorm → Self-Attention → Add → LayerNorm → FFN → Add."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, rngs: nnx.Rngs = None):
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, is_causal=False, rngs=rngs)
        self.self_attn_layer_norm = nnx.LayerNorm(embed_dim, epsilon=1e-5, rngs=rngs)
        self.ffn = FeedForward(embed_dim, ffn_dim, rngs=rngs)
        self.final_layer_norm = nnx.LayerNorm(embed_dim, epsilon=1e-5, rngs=rngs)

    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        x = x + self.self_attn(self.self_attn_layer_norm(x), deterministic=deterministic)
        x = x + self.ffn(self.final_layer_norm(x), deterministic=deterministic)
        return x


class DecoderLayer(nnx.Module):
    """Decoder layer: LN → Self-Attn → Add → LN → Cross-Attn → Add → LN → FFN → Add."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, rngs: nnx.Rngs = None):
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, is_causal=True, rngs=rngs)
        self.self_attn_layer_norm = nnx.LayerNorm(embed_dim, epsilon=1e-5, rngs=rngs)
        self.encoder_attn = MultiHeadAttention(embed_dim, num_heads, is_causal=False, rngs=rngs)
        self.encoder_attn_layer_norm = nnx.LayerNorm(embed_dim, epsilon=1e-5, rngs=rngs)
        self.ffn = FeedForward(embed_dim, ffn_dim, rngs=rngs)
        self.final_layer_norm = nnx.LayerNorm(embed_dim, epsilon=1e-5, rngs=rngs)

    def __call__(
        self, x: jax.Array, enc: jax.Array | None = None, deterministic: bool = True
    ) -> jax.Array:
        x = x + self.self_attn(self.self_attn_layer_norm(x), deterministic=deterministic)
        if enc is not None:
            x = x + self.encoder_attn(
                self.encoder_attn_layer_norm(x), kv=enc, deterministic=deterministic
            )
        x = x + self.ffn(self.final_layer_norm(x), deterministic=deterministic)
        return x


class WhisperEncoder(nnx.Module):
    """Encoder: Conv layers → Positional embedding → Transformer layers."""

    def __init__(
        self,
        num_mel_bins: int = 80,
        max_source_positions: int = 1500,
        embed_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        rngs: nnx.Rngs = None,
    ):
        self.conv1 = nnx.Conv(num_mel_bins, embed_dim, kernel_size=(3,), padding=1, rngs=rngs)
        self.conv2 = nnx.Conv(
            embed_dim, embed_dim, kernel_size=(3,), strides=(2,), padding=1, rngs=rngs
        )
        self.embed_positions = nnx.Embed(max_source_positions, embed_dim, rngs=rngs)
        self.layers = nnx.List(
            [EncoderLayer(embed_dim, num_heads, ffn_dim, rngs=rngs) for _ in range(num_layers)]
        )
        self.layer_norm = nnx.LayerNorm(embed_dim, epsilon=1e-5, rngs=rngs)

    def __call__(self, input_features: jax.Array, deterministic: bool = True) -> jax.Array:
        """Encode mel-spectrogram (B, 80, T) → (B, T/2, D)."""
        x = input_features.transpose(0, 2, 1)  # → (B, T, 80)
        x = jax.nn.gelu(self.conv1(x), approximate=False)
        x = jax.nn.gelu(self.conv2(x), approximate=False)
        x = x + self.embed_positions(jnp.arange(x.shape[1])[None, :])
        for layer in self.layers:
            x = layer(x, deterministic=deterministic)
        return self.layer_norm(x)


class WhisperDecoder(nnx.Module):
    """Decoder: Token + Positional embeddings → Transformer layers."""

    def __init__(
        self,
        vocab_size: int = 51865,
        max_target_positions: int = 448,
        embed_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        rngs: nnx.Rngs = None,
    ):
        self.embed_tokens = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
        self.embed_positions = nnx.Embed(max_target_positions, embed_dim, rngs=rngs)
        self.layers = nnx.List(
            [DecoderLayer(embed_dim, num_heads, ffn_dim, rngs=rngs) for _ in range(num_layers)]
        )
        self.layer_norm = nnx.LayerNorm(embed_dim, epsilon=1e-5, rngs=rngs)

    def __call__(
        self,
        input_ids: jax.Array,
        encoder_hidden_states: jax.Array | None = None,
        deterministic: bool = True,
    ) -> jax.Array:
        """Decode token IDs (B, L) → (B, L, D)."""
        x = self.embed_tokens(input_ids) + self.embed_positions(
            jnp.arange(input_ids.shape[1])[None, :]
        )
        for layer in self.layers:
            x = layer(x, enc=encoder_hidden_states, deterministic=deterministic)
        return self.layer_norm(x)


class WhisperModel(nnx.Module):
    """Complete Whisper model: encoder + decoder + language model head."""

    def __init__(
        self,
        num_mel_bins: int = 80,
        vocab_size: int = 51865,
        max_source_positions: int = 1500,
        max_target_positions: int = 448,
        embed_dim: int = 512,
        encoder_layers: int = 6,
        decoder_layers: int = 6,
        encoder_attention_heads: int = 8,
        decoder_attention_heads: int = 8,
        encoder_ffn_dim: int = 2048,
        decoder_ffn_dim: int = 2048,
        rngs: nnx.Rngs = None,
    ):
        self.encoder = WhisperEncoder(
            num_mel_bins,
            max_source_positions,
            embed_dim,
            encoder_layers,
            encoder_attention_heads,
            encoder_ffn_dim,
            rngs,
        )
        self.decoder = WhisperDecoder(
            vocab_size,
            max_target_positions,
            embed_dim,
            decoder_layers,
            decoder_attention_heads,
            decoder_ffn_dim,
            rngs,
        )
        self.lm_head = nnx.Linear(embed_dim, vocab_size, use_bias=False, rngs=rngs)

    def encode(self, input_features: jax.Array, deterministic: bool = True) -> jax.Array:
        return self.encoder(input_features, deterministic)

    def __call__(
        self,
        input_features: jax.Array,
        decoder_input_ids: jax.Array,
        deterministic: bool = True,
    ) -> jax.Array:
        enc = self.encode(input_features, deterministic)
        dec = self.decoder(decoder_input_ids, enc, deterministic)
        return self.lm_head(dec)


def create_whisper_tiny(rngs: nnx.Rngs | None = None) -> WhisperModel:
    """Create Whisper tiny model (39M parameters)."""
    return WhisperModel(
        embed_dim=384,
        encoder_layers=4,
        decoder_layers=4,
        encoder_attention_heads=6,
        decoder_attention_heads=6,
        encoder_ffn_dim=1536,
        decoder_ffn_dim=1536,
        rngs=rngs or nnx.Rngs(0),
    )


def create_whisper_base(rngs: nnx.Rngs | None = None) -> WhisperModel:
    """Create Whisper base model (74M parameters)."""
    return WhisperModel(
        embed_dim=512,
        encoder_layers=6,
        decoder_layers=6,
        encoder_attention_heads=8,
        decoder_attention_heads=8,
        encoder_ffn_dim=2048,
        decoder_ffn_dim=2048,
        rngs=rngs or nnx.Rngs(0),
    )


def create_whisper_small(rngs: nnx.Rngs | None = None) -> WhisperModel:
    """Create Whisper small model (244M parameters)."""
    return WhisperModel(
        embed_dim=768,
        encoder_layers=12,
        decoder_layers=12,
        encoder_attention_heads=12,
        decoder_attention_heads=12,
        encoder_ffn_dim=3072,
        decoder_ffn_dim=3072,
        rngs=rngs or nnx.Rngs(0),
    )

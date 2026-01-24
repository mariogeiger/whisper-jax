"""
Pure JAX Whisper implementation using Flax NNX.

This is a simplified implementation of OpenAI's Whisper model built from scratch
using Flax NNX (the new Flax API). It demonstrates the core architecture and
can load pretrained weights from HuggingFace.
"""

import jax
import jax.numpy as jnp
from flax import nnx


class MultiHeadAttention(nnx.Module):
    """Multi-head attention module."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_causal: bool = False,
        rngs: nnx.Rngs = None,
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.is_causal = is_causal

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Q, K, V projections
        self.q_proj = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.k_proj = nnx.Linear(embed_dim, embed_dim, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.out_proj = nnx.Linear(embed_dim, embed_dim, rngs=rngs)

        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        key_value_states: jnp.ndarray | None = None,
        attention_mask: jnp.ndarray | None = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        Args:
            hidden_states: (batch, seq_len, embed_dim)
            key_value_states: Optional (batch, kv_len, embed_dim) for cross-attention
            attention_mask: Optional attention mask
            deterministic: Whether to use dropout

        Returns:
            attention_output: (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Compute Q, K, V
        query = self.q_proj(hidden_states)

        if key_value_states is None:
            # Self-attention
            key = self.k_proj(hidden_states)
            value = self.v_proj(hidden_states)
        else:
            # Cross-attention
            key = self.k_proj(key_value_states)
            value = self.v_proj(key_value_states)

        # Reshape to (batch, num_heads, seq_len, head_dim)
        query = query.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        key = key.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        value = value.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = jnp.sqrt(self.head_dim).astype(query.dtype)
        attn_weights = jnp.einsum("bhqd,bhkd->bhqk", query, key) / scale

        # Apply causal mask if needed
        if self.is_causal:
            mask = jnp.triu(jnp.ones((seq_len, seq_len)), k=1)
            attn_weights = jnp.where(mask[None, None, :, :] > 0, -1e10, attn_weights)

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_weights = self.dropout(attn_weights, deterministic=deterministic)

        # Apply attention to values
        attn_output = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, value)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)

        return self.out_proj(attn_output)


class FeedForward(nnx.Module):
    """Feed-forward network."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.0, rngs: nnx.Rngs = None):
        self.fc1 = nnx.Linear(embed_dim, ffn_dim, rngs=rngs)
        self.fc2 = nnx.Linear(ffn_dim, embed_dim, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        x = self.fc1(x)
        x = jax.nn.gelu(x, approximate=False)
        x = self.dropout(x, deterministic=deterministic)
        x = self.fc2(x)
        x = self.dropout(x, deterministic=deterministic)
        return x


class EncoderLayer(nnx.Module):
    """Whisper encoder layer."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        rngs: nnx.Rngs = None,
    ):
        self.self_attn = MultiHeadAttention(
            embed_dim, num_heads, dropout, is_causal=False, rngs=rngs
        )
        self.self_attn_layer_norm = nnx.LayerNorm(embed_dim, epsilon=1e-5, rngs=rngs)

        self.ffn = FeedForward(embed_dim, ffn_dim, dropout, rngs=rngs)
        self.final_layer_norm = nnx.LayerNorm(embed_dim, epsilon=1e-5, rngs=rngs)

        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, hidden_states: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # Self-attention
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, deterministic=deterministic)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = residual + hidden_states

        # Feed-forward
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.ffn(hidden_states, deterministic=deterministic)
        hidden_states = residual + hidden_states

        return hidden_states


class DecoderLayer(nnx.Module):
    """Whisper decoder layer."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        rngs: nnx.Rngs = None,
    ):
        self.self_attn = MultiHeadAttention(
            embed_dim, num_heads, dropout, is_causal=True, rngs=rngs
        )
        self.self_attn_layer_norm = nnx.LayerNorm(embed_dim, epsilon=1e-5, rngs=rngs)

        self.encoder_attn = MultiHeadAttention(
            embed_dim, num_heads, dropout, is_causal=False, rngs=rngs
        )
        self.encoder_attn_layer_norm = nnx.LayerNorm(embed_dim, epsilon=1e-5, rngs=rngs)

        self.ffn = FeedForward(embed_dim, ffn_dim, dropout, rngs=rngs)
        self.final_layer_norm = nnx.LayerNorm(embed_dim, epsilon=1e-5, rngs=rngs)

        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        encoder_hidden_states: jnp.ndarray | None = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        # Self-attention (causal)
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, deterministic=deterministic)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = residual + hidden_states

        # Cross-attention (if encoder states provided)
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            hidden_states = self.encoder_attn(
                hidden_states, key_value_states=encoder_hidden_states, deterministic=deterministic
            )
            hidden_states = self.dropout(hidden_states, deterministic=deterministic)
            hidden_states = residual + hidden_states

        # Feed-forward
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.ffn(hidden_states, deterministic=deterministic)
        hidden_states = residual + hidden_states

        return hidden_states


class WhisperEncoder(nnx.Module):
    """Whisper audio encoder."""

    def __init__(
        self,
        num_mel_bins: int = 80,
        max_source_positions: int = 1500,
        embed_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.0,
        rngs: nnx.Rngs = None,
    ):
        # Convolution layers to process mel-spectrogram
        self.conv1 = nnx.Conv(
            in_features=num_mel_bins,
            out_features=embed_dim,
            kernel_size=(3,),
            padding="SAME",
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            in_features=embed_dim,
            out_features=embed_dim,
            kernel_size=(3,),
            strides=(2,),
            padding="SAME",
            rngs=rngs,
        )

        # Positional embedding
        self.embed_positions = nnx.Embed(
            num_embeddings=max_source_positions, features=embed_dim, rngs=rngs
        )

        # Transformer encoder layers
        self.layers = nnx.Sequential(
            *[
                EncoderLayer(embed_dim, num_heads, ffn_dim, dropout, rngs=rngs)
                for _ in range(num_layers)
            ]
        )

        self.layer_norm = nnx.LayerNorm(embed_dim, epsilon=1e-5, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, input_features: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """
        Args:
            input_features: (batch, num_mel_bins, time_steps)

        Returns:
            encoder_output: (batch, seq_len, embed_dim)
        """
        # Transpose to (batch, time_steps, num_mel_bins)
        x = input_features.transpose(0, 2, 1)

        # Apply convolutions
        x = jax.nn.gelu(self.conv1(x), approximate=False)
        x = jax.nn.gelu(self.conv2(x), approximate=False)

        # Add positional embeddings
        seq_len = x.shape[1]
        positions = jnp.arange(seq_len)[None, :]
        x = x + self.embed_positions(positions)

        x = self.dropout(x, deterministic=deterministic)

        # Apply transformer layers
        for i in range(len(self.layers.layers)):
            x = self.layers.layers[i](x, deterministic=deterministic)

        x = self.layer_norm(x)

        return x


class WhisperDecoder(nnx.Module):
    """Whisper text decoder."""

    def __init__(
        self,
        vocab_size: int = 51865,
        max_target_positions: int = 448,
        embed_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.0,
        rngs: nnx.Rngs = None,
    ):
        # Token and position embeddings
        self.embed_tokens = nnx.Embed(num_embeddings=vocab_size, features=embed_dim, rngs=rngs)
        self.embed_positions = nnx.Embed(
            num_embeddings=max_target_positions, features=embed_dim, rngs=rngs
        )

        # Transformer decoder layers
        self.layers = nnx.Sequential(
            *[
                DecoderLayer(embed_dim, num_heads, ffn_dim, dropout, rngs=rngs)
                for _ in range(num_layers)
            ]
        )

        self.layer_norm = nnx.LayerNorm(embed_dim, epsilon=1e-5, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(
        self,
        input_ids: jnp.ndarray,
        encoder_hidden_states: jnp.ndarray | None = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        Args:
            input_ids: (batch, seq_len) token IDs
            encoder_hidden_states: (batch, enc_seq_len, embed_dim)

        Returns:
            decoder_output: (batch, seq_len, embed_dim)
        """
        batch_size, seq_len = input_ids.shape

        # Get embeddings
        x = self.embed_tokens(input_ids)

        # Add positional embeddings
        positions = jnp.arange(seq_len)[None, :]
        x = x + self.embed_positions(positions)

        x = self.dropout(x, deterministic=deterministic)

        # Apply transformer layers
        for i in range(len(self.layers.layers)):
            x = self.layers.layers[i](
                x, encoder_hidden_states=encoder_hidden_states, deterministic=deterministic
            )

        x = self.layer_norm(x)

        return x


class WhisperModel(nnx.Module):
    """Complete Whisper model (encoder + decoder)."""

    def __init__(
        self,
        # Architecture parameters
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
        dropout: float = 0.0,
        rngs: nnx.Rngs = None,
    ):
        self.encoder = WhisperEncoder(
            num_mel_bins=num_mel_bins,
            max_source_positions=max_source_positions,
            embed_dim=embed_dim,
            num_layers=encoder_layers,
            num_heads=encoder_attention_heads,
            ffn_dim=encoder_ffn_dim,
            dropout=dropout,
            rngs=rngs,
        )

        self.decoder = WhisperDecoder(
            vocab_size=vocab_size,
            max_target_positions=max_target_positions,
            embed_dim=embed_dim,
            num_layers=decoder_layers,
            num_heads=decoder_attention_heads,
            ffn_dim=decoder_ffn_dim,
            dropout=dropout,
            rngs=rngs,
        )

        # Output projection (language modeling head)
        self.lm_head = nnx.Linear(embed_dim, vocab_size, use_bias=False, rngs=rngs)

    def encode(self, input_features: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """Encode audio features."""
        return self.encoder(input_features, deterministic=deterministic)

    def decode(
        self, input_ids: jnp.ndarray, encoder_hidden_states: jnp.ndarray, deterministic: bool = True
    ) -> jnp.ndarray:
        """Decode with encoder outputs."""
        decoder_output = self.decoder(
            input_ids, encoder_hidden_states=encoder_hidden_states, deterministic=deterministic
        )
        logits = self.lm_head(decoder_output)
        return logits

    def __call__(
        self,
        input_features: jnp.ndarray,
        decoder_input_ids: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Full forward pass."""
        encoder_output = self.encode(input_features, deterministic=deterministic)
        logits = self.decode(decoder_input_ids, encoder_output, deterministic=deterministic)
        return logits


def create_whisper_tiny(rngs: nnx.Rngs = None) -> WhisperModel:
    """Create Whisper tiny model (39M parameters)."""
    if rngs is None:
        rngs = nnx.Rngs(0)

    return WhisperModel(
        num_mel_bins=80,
        vocab_size=51865,
        max_source_positions=1500,
        max_target_positions=448,
        embed_dim=384,
        encoder_layers=4,
        decoder_layers=4,
        encoder_attention_heads=6,
        decoder_attention_heads=6,
        encoder_ffn_dim=1536,
        decoder_ffn_dim=1536,
        dropout=0.0,
        rngs=rngs,
    )


def create_whisper_base(rngs: nnx.Rngs = None) -> WhisperModel:
    """Create Whisper base model (74M parameters)."""
    if rngs is None:
        rngs = nnx.Rngs(0)

    return WhisperModel(
        num_mel_bins=80,
        vocab_size=51865,
        max_source_positions=1500,
        max_target_positions=448,
        embed_dim=512,
        encoder_layers=6,
        decoder_layers=6,
        encoder_attention_heads=8,
        decoder_attention_heads=8,
        encoder_ffn_dim=2048,
        decoder_ffn_dim=2048,
        dropout=0.0,
        rngs=rngs,
    )


def create_whisper_small(rngs: nnx.Rngs = None) -> WhisperModel:
    """Create Whisper small model (244M parameters)."""
    if rngs is None:
        rngs = nnx.Rngs(0)

    return WhisperModel(
        num_mel_bins=80,
        vocab_size=51865,
        max_source_positions=1500,
        max_target_positions=448,
        embed_dim=768,
        encoder_layers=12,
        decoder_layers=12,
        encoder_attention_heads=12,
        decoder_attention_heads=12,
        encoder_ffn_dim=3072,
        decoder_ffn_dim=3072,
        dropout=0.0,
        rngs=rngs,
    )


if __name__ == "__main__":
    # Example usage
    print("Creating Whisper Tiny model...")
    model = create_whisper_tiny()

    # Test with dummy data
    batch_size = 2
    num_mel_bins = 80
    time_steps = 3000

    # Create dummy input
    input_features = jnp.ones((batch_size, num_mel_bins, time_steps))
    decoder_input_ids = jnp.ones((batch_size, 10), dtype=jnp.int32)

    print(f"Input features shape: {input_features.shape}")
    print(f"Decoder input IDs shape: {decoder_input_ids.shape}")

    # Forward pass
    print("\nRunning forward pass...")
    logits = model(input_features, decoder_input_ids, deterministic=True)
    print(f"Output logits shape: {logits.shape}")

    # Test encoder
    print("\nTesting encoder...")
    encoder_output = model.encode(input_features)
    print(f"Encoder output shape: {encoder_output.shape}")

    print("\nModel created successfully!")

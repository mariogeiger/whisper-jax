"""Pure JAX decoding and alignment functions."""

import base64
import gzip
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from whisper_jax.core.audio import log_mel_spectrogram

# Alignment heads for each model - base85-encoded boolean arrays
_ALIGNMENT_HEADS = {
    "tiny.en": b"ABzY8J1N>@0{>%R00Bk>$p{7v037`oCl~+#00",
    "tiny": b"ABzY8bu8Lr0{>%RKn9Fp%m@SkK7Kt=7ytkO",
    "base.en": b"ABzY8;40c<0{>%RzzG;p*o+Vo09|#PsxSZm00",
    "base": b"ABzY8KQ!870{>%RzyTQH3`Q^yNP!>##QT-<FaQ7m",
    "small.en": b"ABzY8>?_)10{>%RpeA61k&I|OI3I$65C{;;pbCHh0B{qLQ;+}v00",
    "small": b"ABzY8DmU6=0{>%Rpa?J`kvJ6qF(V^F86#Xh7JUGMK}P<N0000",
    "medium.en": b"ABzY8usPae0{>%R7<zz_OvQ{)4kMa0BMw6u5rT}kRKX;$NfYBv00*Hl@qhsU00",
    "medium": b"ABzY8B0Jh+0{>%R7}kK1fFL7w6%<-Pf*t^=N)Qr&0RR9",
    "large-v1": b"ABzY8r9j$a0{>%R7#4sLmoOs{s)o3~84-RPdcFk!JR<kSfC2yj",
    "large-v2": b"ABzY8zd+h!0{>%R7=D0pU<_bnWW*tkYAhobTNnu$jnkEkXqp)j;w1Tzk)UH3X%SZd&fFZ2fC2yj",
    "large-v3": b"ABzY8gWO1E0{>%R7(9S+Kn!D~%ngiGaR?*L!iJG9p-nab0JQ=-{D1-g00",
}

_MODEL_DIMS = {
    "tiny": {"n_text_layer": 4, "n_text_head": 6},
    "tiny.en": {"n_text_layer": 4, "n_text_head": 6},
    "base": {"n_text_layer": 6, "n_text_head": 8},
    "base.en": {"n_text_layer": 6, "n_text_head": 8},
    "small": {"n_text_layer": 12, "n_text_head": 12},
    "small.en": {"n_text_layer": 12, "n_text_head": 12},
    "medium": {"n_text_layer": 24, "n_text_head": 16},
    "medium.en": {"n_text_layer": 24, "n_text_head": 16},
    "large-v1": {"n_text_layer": 32, "n_text_head": 20},
    "large-v2": {"n_text_layer": 32, "n_text_head": 20},
    "large-v3": {"n_text_layer": 32, "n_text_head": 20},
}


@lru_cache(maxsize=16)
def decode_alignment_heads(model_name: str) -> tuple[tuple[int, int], ...]:
    """Decode alignment heads. Returns tuple of (layer, head) pairs."""
    if model_name not in _ALIGNMENT_HEADS:
        dims = _MODEL_DIMS.get(model_name, {"n_text_layer": 4, "n_text_head": 6})
        n_layers, n_heads = dims["n_text_layer"], dims["n_text_head"]
        return tuple(
            (layer, head) for layer in range(n_layers // 2, n_layers) for head in range(n_heads)
        )

    data = _ALIGNMENT_HEADS[model_name]
    dims = _MODEL_DIMS[model_name]
    n_layers, n_heads = dims["n_text_layer"], dims["n_text_head"]

    array = np.frombuffer(gzip.decompress(base64.b85decode(data)), dtype=bool).reshape(
        n_layers, n_heads
    )

    return tuple(
        (layer, head) for layer in range(n_layers) for head in range(n_heads) if array[layer, head]
    )


def get_alignment_mask(model_name: str) -> jnp.ndarray:
    """Get alignment head mask as JAX array for JIT compatibility."""
    dims = _MODEL_DIMS.get(model_name, {"n_text_layer": 4, "n_text_head": 6})
    n_layers, n_heads = dims["n_text_layer"], dims["n_text_head"]

    mask = np.zeros((n_layers, n_heads), dtype=np.float32)
    for layer, head in decode_alignment_heads(model_name):
        mask[layer, head] = 1.0

    return jnp.array(mask)


def create_transcribe_fn(model, max_tokens: int = 100, use_timestamps: bool = False):
    """Create a JIT-compiled transcription function.

    This function creates a highly optimized transcription function that:
    - Uses jax.lax.while_loop for early stopping at EOT
    - Captures model components in closure for optimal JIT performance
    - Uses fixed-size token buffer for consistent shapes

    Args:
        model: WhisperModel instance
        max_tokens: Maximum tokens to generate (compile-time constant)
        use_timestamps: If True, use timestamp mode for seek-based chunking.
            When True, the model outputs timestamp tokens that indicate
            segment boundaries, enabling proper handling of audio chunks.

    Returns:
        JIT-compiled function with signature:
            (audio: jax.Array, lang_token: jax.Array) -> (tokens, num_generated)

    Example:
        transcribe = create_transcribe_fn(model, use_timestamps=True)
        tokens, n = transcribe(audio, jnp.array(50259))  # English
        # tokens may include timestamp tokens when use_timestamps=True
    """
    encoder = model.encoder
    decoder = model.decoder
    lm_head = model.lm_head
    EOT = 50257
    NO_TIMESTAMPS = 50363
    TIMESTAMP_BEGIN = 50364

    # Use <|0.00|> as the initial timestamp when in timestamp mode
    last_prompt_token = TIMESTAMP_BEGIN if use_timestamps else NO_TIMESTAMPS

    @jax.jit
    def transcribe(audio: jax.Array, lang_token: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Transcribe audio to tokens.

        Args:
            audio: Audio samples, shape (samples,), padded to 30s
            lang_token: Language token ID (e.g., 50259 for English)

        Returns:
            (tokens, num_generated): Full token buffer and count of generated tokens
        """
        mel = log_mel_spectrogram(audio)
        enc_out = encoder(mel, deterministic=True)

        prompt = jnp.array([50258, lang_token, 50359, last_prompt_token], dtype=jnp.int32)
        tokens = jnp.zeros(4 + max_tokens, dtype=jnp.int32).at[:4].set(prompt)

        def cond(state):
            tokens, _enc, idx = state
            not_at_max = idx < max_tokens
            last_token = tokens[3 + idx]
            not_eot = last_token != EOT
            return not_at_max & not_eot

        def body(state):
            tokens, enc, idx = state
            dec_out, _ = decoder(tokens[None], enc, deterministic=True)
            logits = lm_head(dec_out)
            next_token = jnp.argmax(logits[0, 3 + idx])
            tokens = tokens.at[4 + idx].set(next_token)
            return (tokens, enc, idx + 1)

        tokens, _, num_gen = jax.lax.while_loop(cond, body, (tokens, enc_out, jnp.array(0)))
        return tokens, num_gen

    return transcribe


def create_alignment_fn(model):
    """Create a JIT-compiled alignment function for the given model.

    This captures the model components in a closure for optimal JIT performance.
    Call this once per model, then reuse the returned function.
    """
    encoder = model.encoder
    decoder = model.decoder
    lm_head = model.lm_head

    @jax.jit
    def compute_alignment_data(
        audio: jax.Array, tokens: jax.Array, alignment_mask: jax.Array
    ) -> tuple[jax.Array, jax.Array, int]:
        """JIT-compiled forward pass + attention processing.

        Returns:
            token_probs: (seq_len, vocab_size)
            attention_matrix: (seq_len, encoder_len) processed for DTW
            num_frames: encoder output length
        """
        mel = log_mel_spectrogram(audio)
        enc_out = encoder(mel, deterministic=True)
        dec_out, cross_attns = decoder(tokens, enc_out, deterministic=True)
        logits = lm_head(dec_out)
        probs = jax.nn.softmax(logits, axis=-1)

        # Stack cross-attention: (num_layers, num_heads, seq_len, enc_len)
        all_attn = jnp.stack([attn[0] for attn in cross_attns], axis=0)

        # Weighted mean using alignment mask
        mask = alignment_mask[:, :, None, None]
        masked_attn = all_attn * mask
        weights = masked_attn.sum(axis=(0, 1)) / (mask.sum() + 1e-8)

        # Softmax per position
        weights = weights - weights.max(axis=-1, keepdims=True)
        weights = jnp.exp(weights)
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-8)

        # Z-score normalization
        mean = weights.mean(axis=-1, keepdims=True)
        std = weights.std(axis=-1, keepdims=True)
        weights = (weights - mean) / (std + 1e-8)

        return probs[0], weights, enc_out.shape[1]

    return compute_alignment_data

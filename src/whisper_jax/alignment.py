"""Word-level timestamp alignment using DTW on cross-attention weights.

This module implements the Dynamic Time Warping (DTW) approach from OpenAI's Whisper
to extract word-level timestamps from cross-attention patterns.

Optimizations:
- JAX JIT-compiled forward pass (encoder + decoder + attention processing)
- Numba JIT-compiled DTW algorithm
"""

import base64
import gzip
import string
from dataclasses import dataclass
from functools import lru_cache

import jax
import jax.numpy as jnp
import numba
import numpy as np
from scipy.ndimage import median_filter

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

SAMPLE_RATE = 16000
HOP_LENGTH = 160
TOKENS_PER_SECOND = SAMPLE_RATE / HOP_LENGTH / 2


@lru_cache(maxsize=16)
def decode_alignment_heads(model_name: str) -> tuple[tuple[int, int], ...]:
    """Decode alignment heads. Returns tuple of (layer, head) pairs."""
    if model_name not in _ALIGNMENT_HEADS:
        dims = _MODEL_DIMS.get(model_name, {"n_text_layer": 4, "n_text_head": 6})
        n_layers, n_heads = dims["n_text_layer"], dims["n_text_head"]
        return tuple(
            (layer, head)
            for layer in range(n_layers // 2, n_layers)
            for head in range(n_heads)
        )

    data = _ALIGNMENT_HEADS[model_name]
    dims = _MODEL_DIMS[model_name]
    n_layers, n_heads = dims["n_text_layer"], dims["n_text_head"]

    array = np.frombuffer(
        gzip.decompress(base64.b85decode(data)), dtype=bool
    ).reshape(n_layers, n_heads)

    return tuple(
        (layer, head)
        for layer in range(n_layers)
        for head in range(n_heads)
        if array[layer, head]
    )


def get_alignment_mask(model_name: str) -> jnp.ndarray:
    """Get alignment head mask as JAX array for JIT compatibility."""
    dims = _MODEL_DIMS.get(model_name, {"n_text_layer": 4, "n_text_head": 6})
    n_layers, n_heads = dims["n_text_layer"], dims["n_text_head"]

    mask = np.zeros((n_layers, n_heads), dtype=np.float32)
    for layer, head in decode_alignment_heads(model_name):
        mask[layer, head] = 1.0

    return jnp.array(mask)


@dataclass
class WordTiming:
    """A word with its timestamp and probability."""

    word: str
    start: float
    end: float
    probability: float


@numba.jit(nopython=True)
def dtw(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Numba-compiled DTW for fast alignment."""
    N, M = x.shape
    cost = np.full((N + 1, M + 1), np.inf, dtype=np.float32)
    trace = np.full((N + 1, M + 1), -1, dtype=np.int32)
    cost[0, 0] = 0

    for j in range(1, M + 1):
        for i in range(1, N + 1):
            c0 = cost[i - 1, j - 1]
            c1 = cost[i - 1, j]
            c2 = cost[i, j - 1]

            if c0 <= c1 and c0 <= c2:
                c, t = c0, 0
            elif c1 <= c2:
                c, t = c1, 1
            else:
                c, t = c2, 2

            cost[i, j] = x[i - 1, j - 1] + c
            trace[i, j] = t

    trace[0, :] = 2
    trace[:, 0] = 1

    i, j = N, M
    path = np.zeros((N + M, 2), dtype=np.int32)
    path_len = 0

    while i > 0 or j > 0:
        path[path_len, 0] = i - 1
        path[path_len, 1] = j - 1
        path_len += 1

        if trace[i, j] == 0:
            i -= 1
            j -= 1
        elif trace[i, j] == 1:
            i -= 1
        else:
            j -= 1

    result = path[:path_len][::-1]
    return result[:, 0].copy(), result[:, 1].copy()


def split_tokens_to_words(
    tokens: list[int], tokenizer
) -> tuple[list[str], list[list[int]]]:
    """Split token sequence into words."""
    if not tokens:
        return [], []

    words = []
    word_tokens = []
    current_tokens = []

    for i, token in enumerate(tokens):
        current_tokens.append(token)
        decoded = tokenizer.decode(current_tokens)

        if i < len(tokens) - 1:
            next_decoded = tokenizer.decode(current_tokens + [tokens[i + 1]])
            if len(next_decoded) > len(decoded):
                extra = next_decoded[len(decoded) :]
                if extra.startswith(" ") or decoded.rstrip() in string.punctuation:
                    if decoded.strip():
                        words.append(decoded)
                        word_tokens.append(current_tokens.copy())
                    current_tokens = []
        else:
            if decoded.strip():
                words.append(decoded)
                word_tokens.append(current_tokens.copy())

    return words, word_tokens


def create_alignment_fn(model):
    """Create a JIT-compiled alignment function for the given model.

    This captures the model components in a closure for optimal JIT performance.
    Call this once per model, then reuse the returned function.
    """
    from .processor import log_mel_spectrogram

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


def find_word_alignments(
    attention_matrix: np.ndarray,
    text_tokens: list[int],
    tokenizer,
    num_frames: int,
    token_probs: np.ndarray | None = None,
    medfilt_width: int = 7,
    prompt_length: int = 4,
) -> list[WordTiming]:
    """Find word-level timestamps from processed attention matrix."""
    if len(text_tokens) == 0:
        return []

    # Apply median filter for smoothing
    matrix = median_filter(attention_matrix[None, :, :], size=(1, 1, medfilt_width))[0]

    # Trim to actual audio frames
    matrix = matrix[:, :num_frames]

    # Extract just text tokens (skip prompt)
    matrix = matrix[prompt_length : prompt_length + len(text_tokens)]

    if matrix.shape[0] == 0:
        return []

    # Run DTW
    text_indices, time_indices = dtw(-matrix.astype(np.float32))

    # Split tokens into words
    words, word_token_lists = split_tokens_to_words(text_tokens, tokenizer)

    if len(words) == 0:
        return []

    # Compute word boundaries
    word_boundaries = np.pad(np.cumsum([len(t) for t in word_token_lists]), (1, 0))

    # Find DTW jumps
    jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
    jump_times = time_indices[jumps] / TOKENS_PER_SECOND

    # Map to times
    start_times = jump_times[np.minimum(word_boundaries[:-1], len(jump_times) - 1)]
    end_times = jump_times[np.minimum(word_boundaries[1:], len(jump_times) - 1)]

    # Word probabilities
    if token_probs is not None:
        word_probs = [
            float(np.mean(token_probs[i:j])) if j > i else 0.0
            for i, j in zip(word_boundaries[:-1], word_boundaries[1:])
        ]
    else:
        word_probs = [1.0] * len(words)

    return [
        WordTiming(word=word, start=float(start), end=float(end), probability=prob)
        for word, start, end, prob in zip(words, start_times, end_times, word_probs)
        if word.strip()
    ]


def get_word_timestamps(
    model,
    tokenizer,
    audio: np.ndarray,
    text_tokens: list[int],
    model_name: str = "tiny",
    medfilt_width: int = 7,
    _alignment_fn=None,
) -> list[WordTiming]:
    """Get word-level timestamps for transcribed text.

    For best performance, create the alignment function once with
    create_alignment_fn(model) and pass it as _alignment_fn.
    """
    if len(text_tokens) == 0:
        return []

    # Get or create alignment function
    if _alignment_fn is None:
        _alignment_fn = create_alignment_fn(model)

    # Get alignment mask
    alignment_mask = get_alignment_mask(model_name)

    # Build token sequence
    prompt = [50258, 50259, 50359, 50363]
    full_tokens = prompt + text_tokens + [50257]
    tokens_jax = jnp.array([full_tokens])

    # Run JIT-compiled forward pass
    probs, attn_matrix, num_frames = _alignment_fn(
        jnp.array(audio), tokens_jax, alignment_mask
    )

    # Extract text token probabilities
    prompt_len = len(prompt)
    token_probs = np.array(
        [float(probs[prompt_len - 1 + i, tok]) for i, tok in enumerate(text_tokens)]
    )

    # Convert attention to numpy
    attn_np = np.array(attn_matrix)

    # Find alignments
    return find_word_alignments(
        attention_matrix=attn_np,
        text_tokens=text_tokens,
        tokenizer=tokenizer,
        num_frames=int(num_frames),
        token_probs=token_probs,
        medfilt_width=medfilt_width,
        prompt_length=prompt_len,
    )

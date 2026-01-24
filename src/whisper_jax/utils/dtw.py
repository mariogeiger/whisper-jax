"""Word-level timestamp alignment using DTW on cross-attention weights.

This module implements the Dynamic Time Warping (DTW) approach from OpenAI's Whisper
to extract word-level timestamps from cross-attention patterns.
"""

import string
from dataclasses import dataclass
from itertools import pairwise

import jax.numpy as jnp
import numba
import numpy as np
from scipy.ndimage import median_filter

from whisper_jax.core.audio import HOP_LENGTH, SAMPLE_RATE
from whisper_jax.core.decode import create_alignment_fn, get_alignment_mask
from whisper_jax.utils.tokenizer import EOT, LANG_TOKENS, NO_TIMESTAMPS, SOT, TRANSCRIBE

TOKENS_PER_SECOND = SAMPLE_RATE / HOP_LENGTH / 2


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


def split_tokens_to_words(tokens: list[int], tokenizer) -> tuple[list[str], list[list[int]]]:
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
            next_decoded = tokenizer.decode([*current_tokens, tokens[i + 1]])
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
            for i, j in pairwise(word_boundaries)
        ]
    else:
        word_probs = [1.0] * len(words)

    return [
        WordTiming(word=word, start=float(start), end=float(end), probability=prob)
        for word, start, end, prob in zip(words, start_times, end_times, word_probs, strict=True)
        if word.strip()
    ]


# Fixed token buffer size to avoid JIT recompilation on varying lengths
# Prompt (4) + max_tokens (200) + EOT (1) = 205
ALIGNMENT_TOKEN_BUFFER = 205


def warmup_dtw() -> None:
    """Warmup Numba JIT compilation for DTW.

    Call this once to front-load the ~0.5s Numba compilation time.
    """
    dtw(np.zeros((10, 100), dtype=np.float32))


def get_word_timestamps(
    model,
    tokenizer,
    audio: np.ndarray,
    text_tokens: list[int],
    model_name: str = "tiny",
    language: str = "en",
    medfilt_width: int = 7,
    _alignment_fn=None,
) -> list[WordTiming]:
    """Get word-level timestamps for transcribed text.

    Args:
        model: WhisperModel instance
        tokenizer: WhisperTokenizer instance
        audio: Audio samples (float32, 16kHz)
        text_tokens: List of token IDs from transcription
        model_name: Model size name for alignment heads
        language: Language code (e.g., "en", "fr") for correct prompt
        medfilt_width: Median filter width for smoothing
        _alignment_fn: Optional pre-created alignment function for speed

    Returns:
        List of WordTiming objects with word-level timestamps

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

    # Build token sequence with correct language token
    lang_token = LANG_TOKENS.get(language, LANG_TOKENS["en"])
    prompt = [SOT, lang_token, TRANSCRIBE, NO_TIMESTAMPS]
    full_tokens = prompt + text_tokens + [EOT]
    actual_len = len(full_tokens)

    # Pad to fixed size to avoid JIT recompilation
    if actual_len < ALIGNMENT_TOKEN_BUFFER:
        full_tokens = full_tokens + [0] * (ALIGNMENT_TOKEN_BUFFER - actual_len)
    tokens_jax = jnp.array([full_tokens])

    # Run JIT-compiled forward pass
    probs, attn_matrix, num_frames = _alignment_fn(audio, tokens_jax, alignment_mask)

    # Extract text token probabilities (only for actual tokens, not padding)
    prompt_len = len(prompt)
    token_probs = np.array(
        [float(probs[prompt_len - 1 + i, tok]) for i, tok in enumerate(text_tokens)]
    )

    # Convert attention to numpy, slice to actual length
    attn_np = np.array(attn_matrix[:actual_len])

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

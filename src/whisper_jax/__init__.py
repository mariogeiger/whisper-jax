"""Whisper JAX - Pure JAX implementation of OpenAI's Whisper using Flax NNX."""

# Primary high-level API
# Core components for advanced users
from whisper_jax.core import (
    SAMPLE_RATE,
    WhisperDecoder,
    WhisperEncoder,
    WhisperModel,
    create_alignment_fn,
    create_transcribe_fn,
    create_whisper_base,
    create_whisper_large,
    create_whisper_medium,
    create_whisper_small,
    create_whisper_tiny,
    log_mel_spectrogram,
    stft,
)
from whisper_jax.pipeline import TranscriptionResult, Whisper

# Utilities for advanced users
from whisper_jax.utils import (
    LANG_TOKENS,
    WhisperTokenizer,
    get_word_timestamps,
    load_audio,
    load_pretrained_weights,
    load_whisper_vocab,
)
from whisper_jax.utils.dtw import WordTiming

__version__ = "0.1.0"

__all__ = [
    # Utils
    "LANG_TOKENS",
    "SAMPLE_RATE",
    "TranscriptionResult",
    # Primary API
    "Whisper",
    "WhisperDecoder",
    "WhisperEncoder",
    # Core - Model
    "WhisperModel",
    "WhisperTokenizer",
    "WordTiming",
    "create_alignment_fn",
    # Core - Functions
    "create_transcribe_fn",
    "create_whisper_base",
    "create_whisper_large",
    "create_whisper_medium",
    "create_whisper_small",
    "create_whisper_tiny",
    "get_word_timestamps",
    "load_audio",
    "load_pretrained_weights",
    "load_whisper_vocab",
    "log_mel_spectrogram",
    "stft",
]

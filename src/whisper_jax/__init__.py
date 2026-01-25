"""Whisper JAX - Pure JAX implementation of OpenAI's Whisper using Flax NNX."""

# Core components (always available)
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
from whisper_jax.utils import (
    LANG_TOKENS,
    WhisperTokenizer,
    load_audio,
    load_pretrained_weights,
    load_whisper_vocab,
)

# Alignment features require optional deps: pip install whisper-jax[alignment]
# Import from whisper_jax.alignment for: WordTiming, get_word_timestamps, etc.

__version__ = "0.1.0"

__all__ = [
    "LANG_TOKENS",
    "SAMPLE_RATE",
    "TranscriptionResult",
    "Whisper",
    "WhisperDecoder",
    "WhisperEncoder",
    "WhisperModel",
    "WhisperTokenizer",
    "create_alignment_fn",
    "create_transcribe_fn",
    "create_whisper_base",
    "create_whisper_large",
    "create_whisper_medium",
    "create_whisper_small",
    "create_whisper_tiny",
    "load_audio",
    "load_pretrained_weights",
    "load_whisper_vocab",
    "log_mel_spectrogram",
    "stft",
]

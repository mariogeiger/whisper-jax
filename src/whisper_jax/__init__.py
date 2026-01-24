"""Whisper JAX - Pure JAX implementation of OpenAI's Whisper using Flax NNX."""

from whisper_jax.model import (
    WhisperModel,
    create_whisper_base,
    create_whisper_small,
    create_whisper_tiny,
)
from whisper_jax.weight_loader import load_pretrained_weights

__version__ = "0.1.0"

__all__ = [
    "WhisperModel",
    "create_whisper_base",
    "create_whisper_small",
    "create_whisper_tiny",
    "load_pretrained_weights",
]

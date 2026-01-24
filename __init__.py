"""
Whisper NNX - Pure JAX implementation of OpenAI's Whisper using Flax NNX.

A clean, from-scratch implementation of the Whisper speech recognition model
built with JAX and the modern Flax NNX API.
"""

__version__ = "0.1.0"

from weight_loader import (
    download_whisper_weights,
    get_whisper_config,
    print_model_info,
)
from whisper_nnx import (
    DecoderLayer,
    EncoderLayer,
    FeedForward,
    # Components
    MultiHeadAttention,
    WhisperDecoder,
    WhisperEncoder,
    # Core model
    WhisperModel,
    create_whisper_base,
    create_whisper_small,
    # Model factories
    create_whisper_tiny,
)

__all__ = [
    # Version
    "__version__",
    # Models
    "WhisperModel",
    "WhisperEncoder",
    "WhisperDecoder",
    # Components
    "MultiHeadAttention",
    "FeedForward",
    "EncoderLayer",
    "DecoderLayer",
    # Factories
    "create_whisper_tiny",
    "create_whisper_base",
    "create_whisper_small",
    # Utilities
    "download_whisper_weights",
    "get_whisper_config",
    "print_model_info",
]

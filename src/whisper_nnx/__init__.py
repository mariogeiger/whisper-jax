"""
Whisper NNX - Pure JAX implementation of OpenAI's Whisper using Flax NNX.

A clean, from-scratch implementation of the Whisper speech recognition model
built with JAX and the modern Flax NNX API.
"""

from whisper_nnx.model import (
    DecoderLayer,
    EncoderLayer,
    FeedForward,
    MultiHeadAttention,
    WhisperDecoder,
    WhisperEncoder,
    WhisperModel,
    create_whisper_base,
    create_whisper_small,
    create_whisper_tiny,
)
from whisper_nnx.weight_loader import (
    get_whisper_config,
    load_pretrained_weights,
    print_model_info,
)

__version__ = "0.1.0"

__all__ = [
    "DecoderLayer",
    "EncoderLayer",
    "FeedForward",
    "MultiHeadAttention",
    "WhisperDecoder",
    "WhisperEncoder",
    "WhisperModel",
    "__version__",
    "create_whisper_base",
    "create_whisper_small",
    "create_whisper_tiny",
    "get_whisper_config",
    "load_pretrained_weights",
    "print_model_info",
]

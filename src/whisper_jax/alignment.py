"""Word-level timestamp alignment (requires: pip install whisper-jax[alignment])."""

# Check dependencies upfront - fail fast with clear message
_missing = []
try:
    import numba  # noqa: F401
except ImportError:
    _missing.append("numba")
try:
    import scipy  # noqa: F401
except ImportError:
    _missing.append("scipy")

if _missing:
    raise ImportError(
        f"Alignment features require: {', '.join(_missing)}. "
        "Install with: pip install whisper-jax[alignment]"
    )

from whisper_jax.utils.dtw import (  # noqa: E402
    WordTiming,
    get_vad_speech_segments,
    get_word_timestamps,
    refine_word_timestamps,
    warmup_dtw,
)

__all__ = [
    "WordTiming",
    "get_vad_speech_segments",
    "get_word_timestamps",
    "refine_word_timestamps",
    "warmup_dtw",
]

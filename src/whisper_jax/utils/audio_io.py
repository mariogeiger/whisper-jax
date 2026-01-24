"""Audio I/O utilities for loading audio files."""

import subprocess
from pathlib import Path

import numpy as np

from whisper_jax.core.audio import SAMPLE_RATE


def load_audio(
    path: str | Path,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Load audio file using ffmpeg and convert to float32.

    Args:
        path: Path to audio file (supports any format ffmpeg can decode)
        sample_rate: Target sample rate (default: 16000 for Whisper)

    Returns:
        Audio waveform as float32 array in range [-1, 1]

    Raises:
        RuntimeError: If ffmpeg fails to load the file
        FileNotFoundError: If the file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    cmd = [
        "ffmpeg",
        "-i",
        str(path),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-loglevel",
        "error",
        "-",
    ]

    result = subprocess.run(cmd, capture_output=True)

    if result.returncode != 0:
        error_msg = result.stderr.decode().strip()
        raise RuntimeError(f"ffmpeg failed to load audio: {error_msg}")

    audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0

    return audio


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to float32 in range [-1, 1].

    Args:
        audio: Audio array (int16 or float32)

    Returns:
        Normalized float32 audio in range [-1, 1]
    """
    audio = audio.astype(np.float32)

    if np.abs(audio).max() > 1.0:
        audio = audio / 32768.0

    return audio

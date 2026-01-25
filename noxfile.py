"""Nox sessions for testing dependency management."""

import nox

nox.options.sessions = ["test_deps"]
nox.options.reuse_existing_virtualenvs = True


@nox.session
def test_deps(session: nox.Session) -> None:
    """Test that all dependency groups install correctly."""
    # Test base installation (should work without optional deps)
    session.install(".")
    session.run("python", "-c", "import whisper_jax; print('Base install OK')")

    # Test weights dependencies
    session.install(".[weights]")
    session.run(
        "python",
        "-c",
        "from huggingface_hub import hf_hub_download; "
        "from safetensors import safe_open; "
        "print('Weights deps OK')",
    )

    # Test alignment dependencies (explicit import from whisper_jax.alignment)
    session.install(".[alignment]")
    session.run(
        "python",
        "-c",
        "import numba; "
        "import scipy; "
        "import webrtcvad; "
        "import soundfile; "
        "from whisper_jax.alignment import get_vad_speech_segments, refine_word_timestamps, WordTiming; "
        "print('Alignment deps OK')",
    )

    # Test demo dependencies
    session.install(".[demo]")
    session.run(
        "python",
        "-c",
        "import fastapi; "
        "import uvicorn; "
        "print('Demo deps OK')",
    )


@nox.session
def test_all(session: nox.Session) -> None:
    """Test that the 'all' extra installs everything."""
    session.install(".[all]")
    session.run(
        "python",
        "-c",
        # Base
        "import whisper_jax; "
        # Weights
        "from huggingface_hub import hf_hub_download; "
        "from safetensors import safe_open; "
        # Alignment (from whisper_jax.alignment)
        "import numba; "
        "import scipy; "
        "import webrtcvad; "
        "import soundfile; "
        "from whisper_jax.alignment import get_vad_speech_segments, refine_word_timestamps, WordTiming; "
        # Demo
        "import fastapi; "
        "import uvicorn; "
        # Dev
        "import pytest; "
        "import ruff; "
        "print('All deps OK')",
    )


@nox.session
def test_clean_audio_imports(session: nox.Session) -> None:
    """Test that clean_audio.py example has all required dependencies."""
    session.install(".[weights,alignment]")
    session.run(
        "python",
        "-c",
        "import argparse; "
        "import subprocess; "
        "from dataclasses import dataclass; "
        "from pathlib import Path; "
        "import numpy as np; "
        "import soundfile as sf; "
        "from whisper_jax import SAMPLE_RATE, Whisper, load_audio; "
        "from whisper_jax.alignment import get_vad_speech_segments; "
        "print('clean_audio.py imports OK')",
    )


@nox.session
def test_alignment_error(session: nox.Session) -> None:
    """Test that alignment import fails clearly without deps."""
    session.install(".")
    # This should fail with a clear ImportError message
    session.run(
        "python",
        "-c",
        "try:\n"
        "    from whisper_jax.alignment import WordTiming\n"
        "    print('ERROR: Should have raised ImportError')\n"
        "    exit(1)\n"
        "except ImportError as e:\n"
        "    assert 'numba' in str(e) or 'scipy' in str(e), f'Bad error: {e}'\n"
        "    print(f'Got expected ImportError: {e}')\n"
        "    print('Alignment error test OK')",
    )

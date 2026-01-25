# Whisper JAX

A clean, simple JAX implementation of OpenAI's Whisper speech recognition model.

Whisper JAX is a pure JAX/Flax NNX implementation that's easy to understand, modify, and use. Whether you're transcribing podcasts, generating subtitles, or building speech applications, you can get started in just a few lines of code.

---

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/mariogeiger/whisper-jax.git
```

For word-level timestamps (requires numba and scipy):

```bash
pip install "whisper-jax[alignment] @ git+https://github.com/mariogeiger/whisper-jax.git"
```

For local development:

```bash
git clone https://github.com/mariogeiger/whisper-jax.git
cd whisper-jax
pip install -e ".[alignment]"
```

---

## Quick Start

```python
from whisper_jax import Whisper

whisper = Whisper.load("tiny")
result = whisper.transcribe("audio.mp3")
print(result.text)
```

That's it! The first call compiles the model with JAX's JIT, then subsequent calls are fast.

---

## Examples

The `examples/` folder contains ready-to-use scripts:

| Example | Description |
|---------|-------------|
| `word_timestamps.py` | Get word-level timing, export SRT/JSON subtitles |
| `clean_audio.py` | Remove filler words ("um", "uh") and silences |
| `demo_server.py` | Real-time browser-based transcription demo |
| `compare_implementations.py` | Verify JAX outputs match PyTorch reference |

```bash
# Generate SRT subtitles
python examples/word_timestamps.py audio.mp3 --format srt > subtitles.srt

# Clean up audio by removing filler words
python examples/clean_audio.py audio.mp3 --lang en

# Run the web demo (requires: pip install ".[demo]")
python examples/demo_server.py
```

---

## Available Models

| Model | Parameters | Usage |
|-------|-----------|-------|
| tiny  | 39M       | `Whisper.load("tiny")` |
| base  | 74M       | `Whisper.load("base")` |
| small | 244M      | `Whisper.load("small")` |
| medium | 769M     | `Whisper.load("medium")` |
| large-v3 | 1.5B   | `Whisper.load("large-v3")` |

---

## Features

- **Simple API**: Just `Whisper.load()` and `transcribe()`
- **Fast**: JAX's JIT compilation for speed
- **Hackable**: Clean, readable code
- **Word Timestamps**: Built-in word-level alignment via DTW
- **Lightweight**: Minimal dependencies
- **Multi-language**: Supports all Whisper languages
- **Long Audio**: Automatic chunking for audio over 30 seconds

---

## Advanced Usage

For lower-level access to the model:

```python
from whisper_jax import (
    WhisperModel,
    create_whisper_tiny,
    load_pretrained_weights,
    log_mel_spectrogram,
)

# Create model manually
model = create_whisper_tiny()
load_pretrained_weights(model, "openai/whisper-tiny")

# Process audio
mel = log_mel_spectrogram(audio_array)
```

---

## Project Structure

```
src/whisper_jax/
    __init__.py      # Public API
    pipeline.py      # High-level Whisper class
    
    core/            # Pure JAX (JIT-compatible)
        model.py     # Neural network layers
        audio.py     # Mel spectrogram
        decode.py    # Transcription/alignment functions
    
    utils/           # Non-JAX helpers
        tokenizer.py # Tokenizer
        weights.py   # Weight loading
        dtw.py       # Word alignment (numba/scipy)
        audio_io.py  # Audio file loading
```

---

## Acknowledgments

- OpenAI for the original [Whisper model](https://github.com/openai/whisper)
- HuggingFace for [hosting pretrained weights](https://huggingface.co/openai)
- Google for [JAX](https://github.com/google/jax) and [Flax](https://github.com/google/flax)

---

## License

MIT License - See LICENSE file for details.

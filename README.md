# Whisper JAX

**A clean, simple JAX implementation of OpenAI's Whisper speech recognition model.**

Pure JAX/Flax NNX implementation that's easy to understand, modify, and use.

---

## Why Whisper JAX?

- **Simple API**: Just `Whisper.load()` and `transcribe()`
- **Fast**: JAX's JIT compilation for speed
- **Hackable**: Clean, readable code
- **Word Timestamps**: Built-in word-level alignment
- **Lightweight**: Minimal dependencies

---

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/mariogeiger/whisper-jax.git
```

With word timestamp support (requires numba and scipy):

```bash
pip install "whisper-jax[alignment] @ git+https://github.com/mariogeiger/whisper-jax.git"
```

Or clone and install locally:

```bash
git clone https://github.com/mariogeiger/whisper-jax.git
cd whisper-jax
pip install -e ".[alignment]"
```

---

## Quick Start

```python
from whisper_jax import Whisper

# Load a model
whisper = Whisper.load("tiny")  # or "base", "small", "medium", "large-v3"

# Transcribe audio
result = whisper.transcribe("audio.mp3")
print(result.text)
```

### Word-Level Timestamps

```python
from whisper_jax import Whisper

whisper = Whisper.load("tiny")
result = whisper.transcribe("audio.mp3", word_timestamps=True)

for word in result.words:
    print(f"[{word.start:.2f}s - {word.end:.2f}s] {word.word}")
```

### Multiple Languages

```python
# Transcribe French audio
result = whisper.transcribe("french_audio.mp3", language="fr")

# See available languages
print(whisper.available_languages)
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

## Examples

Check out the `examples/` folder:

- **`word_timestamps.py`** - Transcribe with word-level timestamps, output SRT/JSON
- **`demo_server.py`** - Real-time browser-based transcription demo
- **`compare_implementations.py`** - Verify outputs match PyTorch reference

```bash
# Transcribe with word timestamps
python examples/word_timestamps.py audio.mp3 --model base

# Generate SRT subtitles
python examples/word_timestamps.py audio.mp3 --format srt > subtitles.srt

# Run the web demo
pip install ".[demo]"
python examples/demo_server.py
```

---

## Advanced Usage

For users who need lower-level access:

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

## Features

- **Pure JAX/Flax NNX**: Modern, functional API
- **Load HuggingFace weights**: Use pretrained OpenAI models
- **Word-level timestamps**: DTW-based alignment on cross-attention
- **Chunking**: Automatic handling of long audio (>30s)
- **Clean architecture**: Separate pure JAX code from utilities

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

## Contributing

Contributions are welcome! The codebase is intentionally kept simple and readable.

---

## License

MIT License - See LICENSE file for details.

---

## Acknowledgments

- OpenAI for the original [Whisper model](https://github.com/openai/whisper)
- HuggingFace for [hosting pretrained weights](https://huggingface.co/openai)
- Google for [JAX](https://github.com/google/jax) and [Flax](https://github.com/google/flax)

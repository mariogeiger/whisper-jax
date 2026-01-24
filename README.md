# Whisper JAX 🎙️

**A clean, simple JAX implementation of OpenAI's Whisper speech recognition model.**

Pure JAX/Flax NNX implementation that's easy to understand, modify, and use. Perfect for research and learning!

---

## Why Whisper JAX?

- ✨ **Simple & Clean**: ~400 lines of readable code
- 🚀 **Fast**: JAX's JIT compilation for speed
- 🔧 **Hackable**: Easy to modify and experiment with
- 📦 **Lightweight**: Minimal dependencies
- ✅ **Verified**: Matches PyTorch reference implementation

---

## Installation

Install directly from GitHub:
```bash
pip install git+https://github.com/mariogeiger/whisper-jax.git
```

Or clone and install locally:
```bash
git clone https://github.com/mariogeiger/whisper-jax.git
cd whisper-jax
pip install -e .
```

For development with all tools:
```bash
pip install -e ".[dev]"
```

---

## Quick Start

### Load and Run Pretrained Whisper

```python
import jax.numpy as jnp
from whisper_jax import create_whisper_tiny, load_pretrained_weights

# Create model and load pretrained weights
model = create_whisper_tiny()
load_pretrained_weights(model, "openai/whisper-tiny")

# Prepare inputs
input_features = jnp.ones((1, 80, 3000))  # Mel spectrogram
decoder_ids = jnp.array([[50258, 50259, 50359, 50363]])  # Token IDs

# Run inference
logits = model(input_features, decoder_ids, deterministic=True)
print(f"Output shape: {logits.shape}")  # (1, 4, 51865)
```

### Available Models

| Model | Parameters | Command |
|-------|-----------|---------|
| Tiny  | 39M       | `create_whisper_tiny()` |
| Base  | 74M       | `create_whisper_base()` |
| Small | 244M      | `create_whisper_small()` |

---

## Examples

Check out the `examples/` folder:

- **`compare_implementations.py`** - Compare with PyTorch reference
- **`example_usage.py`** - Complete usage examples

Run the comparison to verify outputs match PyTorch:
```bash
python examples/compare_implementations.py
```

---

## Features

- **Pure JAX/Flax NNX**: Modern, functional API
- **Load HuggingFace weights**: Use pretrained OpenAI models
- **Clean architecture**: Easy to read and understand
- **Type hints**: Full typing support with `jax.Array`
- **Verified**: Outputs match PyTorch within float32 precision

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

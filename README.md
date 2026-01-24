# Whisper NNX

Pure JAX implementation of OpenAI's Whisper using Flax NNX.

A clean, from-scratch implementation of the Whisper speech recognition model built with JAX and the modern Flax NNX API.

## Installation

```bash
# Basic installation
pip install -e .

# With CUDA support
pip install -e ".[cuda]"

# With HuggingFace weights loading
pip install -e ".[weights]"

# Full development installation
pip install -e ".[all]"
```

## Quick Start

```python
from whisper_nnx import create_whisper_tiny
import jax.numpy as jnp

# Create model
model = create_whisper_tiny()

# Create dummy input (batch, mel_bins, time_steps)
input_features = jnp.ones((1, 80, 3000))
decoder_input_ids = jnp.array([[50258, 50259, 50359]])

# Run inference
logits = model(input_features, decoder_input_ids, deterministic=True)
```

## Available Models

- `create_whisper_tiny()` - 39M parameters
- `create_whisper_base()` - 74M parameters  
- `create_whisper_small()` - 244M parameters

## Loading Pretrained Weights

```python
from whisper_nnx import download_whisper_weights, get_whisper_config

# Download weights from HuggingFace
params, config = download_whisper_weights("openai/whisper-tiny")
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
hatch run test

# Lint and format
hatch run lint
hatch run format

# Type checking
hatch run typecheck
```

## License

MIT

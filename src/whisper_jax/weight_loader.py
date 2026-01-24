"""Weight loading utilities for Whisper JAX models."""

import jax.numpy as jnp
import numpy as np
from flax import nnx


def flatten_state_dict(d, parent_key="", sep="."):
    """Flatten nested dict, converting integer keys to strings."""
    items = []
    for k, v in d.items():
        k_str = str(k) if isinstance(k, int) else k
        new_key = f"{parent_key}{sep}{k_str}" if parent_key else k_str
        if isinstance(v, dict):
            items.extend(flatten_state_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_state_dict(d, sep="."):
    """Unflatten dict, converting string keys back to ints where appropriate."""
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        for part in parts[:-1]:
            part_key = int(part) if part.isdigit() else part
            if part_key not in current:
                current[part_key] = {}
            current = current[part_key]
        last_part_key = int(parts[-1]) if parts[-1].isdigit() else parts[-1]
        current[last_part_key] = value
    return result


def needs_transpose(hf_name: str, hf_param: np.ndarray) -> bool:
    """Check if weight needs transpose: PyTorch (out, in) → JAX (in, out)."""
    return len(hf_param.shape) == 2 and "weight" in hf_name and "embed" not in hf_name


def convert_hf_name_to_nnx(hf_name: str) -> str:
    """Convert HuggingFace parameter name to NNX parameter name."""
    # Direct mappings
    mappings = {
        "encoder.conv1.weight": "encoder.conv1.kernel",
        "encoder.conv2.weight": "encoder.conv2.kernel",
        "encoder.embed_positions.weight": "encoder.embed_positions.embedding",
        "decoder.embed_tokens.weight": "decoder.embed_tokens.embedding",
        "decoder.embed_positions.weight": "decoder.embed_positions.embedding",
        "encoder.layer_norm.weight": "encoder.layer_norm.scale",
        "decoder.layer_norm.weight": "decoder.layer_norm.scale",
    }

    if hf_name in mappings:
        return mappings[hf_name]

    if hf_name.startswith("encoder.layers."):
        return convert_layer_name(hf_name, "encoder")
    if hf_name.startswith("decoder.layers."):
        return convert_layer_name(hf_name, "decoder")

    return hf_name


def convert_layer_name(hf_name: str, module: str) -> str:
    """Convert encoder/decoder layer parameter name."""
    parts = hf_name.split(".")
    layer_idx = parts[2]
    rest = ".".join(parts[3:])

    # Simple replacements
    replacements = {
        ".weight": ".kernel",
        "_layer_norm.kernel": "_layer_norm.scale",
        "fc1.kernel": "ffn.fc1.kernel",
        "fc2.kernel": "ffn.fc2.kernel",
        "fc1.bias": "ffn.fc1.bias",
        "fc2.bias": "ffn.fc2.bias",
    }

    for old, new in replacements.items():
        rest = rest.replace(old, new)

    return f"{module}.layers.{layer_idx}.{rest}"


def load_pretrained_weights(model: nnx.Module, model_name: str = "openai/whisper-tiny") -> int:
    """Load pretrained weights from HuggingFace into NNX model."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    print(f"Loading pretrained weights from {model_name}...")

    # Download and load weights directly from HuggingFace Hub
    weights_path = hf_hub_download(repo_id=model_name, filename="model.safetensors")
    hf_state = {}
    with safe_open(weights_path, framework="numpy") as f:
        for key in f.keys():
            # Remove "model." prefix that safetensors uses
            clean_key = key[6:] if key.startswith("model.") else key
            hf_state[clean_key] = f.get_tensor(key)

    # Get NNX state
    _, nnx_state = nnx.split(model)
    flat_nnx = flatten_state_dict(nnx_state.to_pure_dict(), sep=".")

    print(f"  HuggingFace: {len(hf_state)} params, NNX: {len(flat_nnx)} params")

    # Transfer weights
    transferred = 0
    for hf_name, hf_param in hf_state.items():
        nnx_name = convert_hf_name_to_nnx(hf_name)
        if nnx_name not in flat_nnx:
            continue

        # Transpose linear layers: PyTorch (out, in) → JAX (in, out)
        if needs_transpose(hf_name, hf_param):
            hf_param = hf_param.T

        # Transpose conv layers: PyTorch (out, in, k) → JAX (k, in, out)
        if "conv" in hf_name and "weight" in hf_name:
            hf_param = hf_param.transpose(2, 1, 0)

        # Check shape and update
        if hf_param.shape == flat_nnx[nnx_name].shape:
            flat_nnx[nnx_name] = jnp.array(hf_param)
            transferred += 1

    # Handle tied weights: LM head shares weights with decoder embed tokens
    # In HuggingFace, proj_out uses the same weights as decoder.embed_tokens
    embed_key = "decoder.embed_tokens.embedding"
    lm_head_key = "lm_head.kernel"
    if embed_key in flat_nnx and lm_head_key in flat_nnx:
        # LM head kernel is (embed_dim, vocab_size), embed is (vocab_size, embed_dim)
        flat_nnx[lm_head_key] = flat_nnx[embed_key].T
        transferred += 1

    # Update model
    nnx.update(model, nnx.State(unflatten_state_dict(flat_nnx, sep=".")))

    print(f"  ✓ Transferred {transferred} parameters")
    return transferred

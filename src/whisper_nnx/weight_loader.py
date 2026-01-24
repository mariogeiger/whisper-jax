"""
Weight loading utilities for Whisper NNX models.

This module provides functions to download pretrained weights from HuggingFace Hub
and load them into Flax NNX models.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from flax import nnx


def flatten_state_dict(d, parent_key="", sep="."):
    """Flatten a nested dict, converting integer keys to strings."""
    items = []
    for k, v in d.items():
        # Convert int keys to strings
        k_str = str(k) if isinstance(k, int) else k
        new_key = f"{parent_key}{sep}{k_str}" if parent_key else k_str
        if isinstance(v, dict):
            items.extend(flatten_state_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_state_dict(d, sep="."):
    """Unflatten a dict, converting string keys back to ints where appropriate."""
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        for part in parts[:-1]:
            # Try to convert to int
            try:
                part_key = int(part)
            except ValueError:
                part_key = part

            if part_key not in current:
                current[part_key] = {}
            current = current[part_key]

        # Handle the last part
        last_part = parts[-1]
        try:
            last_part_key = int(last_part)
        except ValueError:
            last_part_key = last_part

        current[last_part_key] = value
    return result


def needs_transpose(hf_name: str, hf_param: np.ndarray) -> bool:
    """Check if weight needs to be transposed when loading from PyTorch."""
    # Embeddings should NOT be transposed
    if "embed" in hf_name or "embedding" in hf_name:
        return False

    # Linear layers need transpose: PyTorch (out, in) -> JAX (in, out)
    if len(hf_param.shape) == 2 and "weight" in hf_name:
        return True

    return False


def convert_hf_name_to_nnx(hf_name: str) -> str:
    """Convert HuggingFace parameter name to NNX parameter name."""
    # Remove "model." prefix if present
    name = hf_name

    # Map component names
    replacements = [
        # Encoder convolutions
        ("encoder.conv1.weight", "encoder.conv1.kernel"),
        ("encoder.conv1.bias", "encoder.conv1.bias"),
        ("encoder.conv2.weight", "encoder.conv2.kernel"),
        ("encoder.conv2.bias", "encoder.conv2.bias"),
        # Embeddings
        ("encoder.embed_positions.weight", "encoder.embed_positions.embedding"),
        ("decoder.embed_tokens.weight", "decoder.embed_tokens.embedding"),
        ("decoder.embed_positions.weight", "decoder.embed_positions.embedding"),
        # Layer norms
        ("encoder.layer_norm.weight", "encoder.layer_norm.scale"),
        ("encoder.layer_norm.bias", "encoder.layer_norm.bias"),
        ("decoder.layer_norm.weight", "decoder.layer_norm.scale"),
        ("decoder.layer_norm.bias", "decoder.layer_norm.bias"),
    ]

    for old, new in replacements:
        if name == old:
            return new

    # Handle encoder layers
    if name.startswith("encoder.layers."):
        return convert_encoder_layer_name(name)

    # Handle decoder layers
    if name.startswith("decoder.layers."):
        return convert_decoder_layer_name(name)

    return name


def convert_encoder_layer_name(hf_name: str) -> str:
    """Convert encoder layer parameter name."""
    # encoder.layers.0.self_attn.q_proj.weight -> encoder.layers.layers.0.self_attn.q_proj.kernel
    parts = hf_name.split(".")
    layer_idx = parts[2]
    rest = ".".join(parts[3:])

    # Map attention and ffn names
    rest = rest.replace("self_attn_layer_norm.weight", "self_attn_layer_norm.scale")
    rest = rest.replace("self_attn_layer_norm.bias", "self_attn_layer_norm.bias")
    rest = rest.replace("final_layer_norm.weight", "final_layer_norm.scale")
    rest = rest.replace("final_layer_norm.bias", "final_layer_norm.bias")

    # Attention projections
    rest = rest.replace("self_attn.k_proj.weight", "self_attn.k_proj.kernel")
    rest = rest.replace("self_attn.v_proj.weight", "self_attn.v_proj.kernel")
    rest = rest.replace("self_attn.v_proj.bias", "self_attn.v_proj.bias")
    rest = rest.replace("self_attn.q_proj.weight", "self_attn.q_proj.kernel")
    rest = rest.replace("self_attn.q_proj.bias", "self_attn.q_proj.bias")
    rest = rest.replace("self_attn.out_proj.weight", "self_attn.out_proj.kernel")
    rest = rest.replace("self_attn.out_proj.bias", "self_attn.out_proj.bias")

    # FFN (fc1/fc2 in HF -> fc1/fc2 in our FFN module)
    rest = rest.replace("fc1.weight", "ffn.fc1.kernel")
    rest = rest.replace("fc1.bias", "ffn.fc1.bias")
    rest = rest.replace("fc2.weight", "ffn.fc2.kernel")
    rest = rest.replace("fc2.bias", "ffn.fc2.bias")

    return f"encoder.layers.layers.{layer_idx}.{rest}"


def convert_decoder_layer_name(hf_name: str) -> str:
    """Convert decoder layer parameter name."""
    parts = hf_name.split(".")
    layer_idx = parts[2]
    rest = ".".join(parts[3:])

    # Map layer norm names
    rest = rest.replace("self_attn_layer_norm.weight", "self_attn_layer_norm.scale")
    rest = rest.replace("self_attn_layer_norm.bias", "self_attn_layer_norm.bias")
    rest = rest.replace("encoder_attn_layer_norm.weight", "encoder_attn_layer_norm.scale")
    rest = rest.replace("encoder_attn_layer_norm.bias", "encoder_attn_layer_norm.bias")
    rest = rest.replace("final_layer_norm.weight", "final_layer_norm.scale")
    rest = rest.replace("final_layer_norm.bias", "final_layer_norm.bias")

    # Self attention projections
    rest = rest.replace("self_attn.k_proj.weight", "self_attn.k_proj.kernel")
    rest = rest.replace("self_attn.v_proj.weight", "self_attn.v_proj.kernel")
    rest = rest.replace("self_attn.v_proj.bias", "self_attn.v_proj.bias")
    rest = rest.replace("self_attn.q_proj.weight", "self_attn.q_proj.kernel")
    rest = rest.replace("self_attn.q_proj.bias", "self_attn.q_proj.bias")
    rest = rest.replace("self_attn.out_proj.weight", "self_attn.out_proj.kernel")
    rest = rest.replace("self_attn.out_proj.bias", "self_attn.out_proj.bias")

    # Cross attention projections
    rest = rest.replace("encoder_attn.k_proj.weight", "encoder_attn.k_proj.kernel")
    rest = rest.replace("encoder_attn.v_proj.weight", "encoder_attn.v_proj.kernel")
    rest = rest.replace("encoder_attn.v_proj.bias", "encoder_attn.v_proj.bias")
    rest = rest.replace("encoder_attn.q_proj.weight", "encoder_attn.q_proj.kernel")
    rest = rest.replace("encoder_attn.q_proj.bias", "encoder_attn.q_proj.bias")
    rest = rest.replace("encoder_attn.out_proj.weight", "encoder_attn.out_proj.kernel")
    rest = rest.replace("encoder_attn.out_proj.bias", "encoder_attn.out_proj.bias")

    # FFN
    rest = rest.replace("fc1.weight", "ffn.fc1.kernel")
    rest = rest.replace("fc1.bias", "ffn.fc1.bias")
    rest = rest.replace("fc2.weight", "ffn.fc2.kernel")
    rest = rest.replace("fc2.bias", "ffn.fc2.bias")

    return f"decoder.layers.layers.{layer_idx}.{rest}"


def build_weight_mapping(hf_state: dict, nnx_flat: dict) -> dict:
    """Build mapping from HuggingFace parameter names to NNX parameter names."""
    mapping = {}

    for hf_name in hf_state.keys():
        nnx_name = convert_hf_name_to_nnx(hf_name)
        if nnx_name in nnx_flat:
            mapping[hf_name] = nnx_name
        else:
            mapping[hf_name] = None

    return mapping


def load_pretrained_weights(model: nnx.Module, model_name: str = "openai/whisper-tiny") -> int:
    """
    Load pretrained weights from HuggingFace into NNX model.

    Args:
        model: NNX Whisper model
        model_name: HuggingFace model identifier (e.g., "openai/whisper-tiny")

    Returns:
        Number of parameters successfully transferred
    """
    from transformers import WhisperModel as HFWhisperModel

    print(f"Loading pretrained weights from {model_name}...")

    # Load HuggingFace PyTorch model
    hf_model = HFWhisperModel.from_pretrained(model_name)
    hf_model.eval()
    hf_state = hf_model.state_dict()

    print(f"  HuggingFace model has {len(hf_state)} parameters")

    # Get NNX state
    _graph_def, nnx_state = nnx.split(model)
    flat_nnx = flatten_state_dict(nnx_state.to_pure_dict(), sep=".")

    print(f"  NNX model has {len(flat_nnx)} parameters")

    # Build mapping and transfer weights
    transferred = 0
    missing_in_nnx = []
    shape_mismatches = []

    weight_map = build_weight_mapping(hf_state, flat_nnx)

    for hf_name, nnx_name in weight_map.items():
        if nnx_name is None:
            missing_in_nnx.append(hf_name)
            continue

        hf_param = hf_state[hf_name].numpy()

        if nnx_name not in flat_nnx:
            missing_in_nnx.append(hf_name)
            continue

        nnx_param = flat_nnx[nnx_name]

        # Handle transpositions for linear layers
        if needs_transpose(hf_name, hf_param):
            hf_param = hf_param.T

        # Handle conv layers (need to rearrange dimensions)
        if "conv" in hf_name.lower() and "weight" in hf_name:
            # PyTorch conv: (out_channels, in_channels, kernel_size)
            # JAX conv: (kernel_size, in_channels, out_channels)
            hf_param = np.transpose(hf_param, (2, 1, 0))

        # Verify shapes match
        if hf_param.shape != nnx_param.shape:
            shape_mismatches.append((hf_name, nnx_name, hf_param.shape, nnx_param.shape))
            continue

        # Update the flat dict
        flat_nnx[nnx_name] = jnp.array(hf_param)
        transferred += 1

    print(f"  Transferred {transferred} parameters successfully")

    if missing_in_nnx:
        print(f"  Warning: {len(missing_in_nnx)} parameters not found in NNX model")

    if shape_mismatches:
        print(f"  Warning: {len(shape_mismatches)} shape mismatches:")
        for hf_n, nnx_n, hf_s, nnx_s in shape_mismatches[:3]:
            print(f"    {hf_n} {hf_s} -> {nnx_n} {nnx_s}")

    # Unflatten and update the model
    unflat_state = unflatten_state_dict(dict(flat_nnx.items()), sep=".")
    new_state = nnx.State(unflat_state)
    nnx.update(model, new_state)

    print("  ✓ Weights loaded successfully")

    return transferred


def get_whisper_config(model_name: str):
    """Get Whisper configuration for a model."""
    from transformers import WhisperConfig as HFWhisperConfig

    config = HFWhisperConfig.from_pretrained(model_name)
    return config


def print_model_info(config) -> None:
    """Print model architecture information."""
    print(f"\n{'=' * 80}")
    print(f"MODEL CONFIGURATION: {config._name_or_path}")
    print(f"{'=' * 80}")
    print("Encoder:")
    print(f"  - Layers: {config.encoder_layers}")
    print(f"  - Attention heads: {config.encoder_attention_heads}")
    print(f"  - FFN dimension: {config.encoder_ffn_dim}")
    print("\nDecoder:")
    print(f"  - Layers: {config.decoder_layers}")
    print(f"  - Attention heads: {config.decoder_attention_heads}")
    print(f"  - FFN dimension: {config.decoder_ffn_dim}")
    print("\nModel:")
    print(f"  - Embedding dimension: {config.d_model}")
    print(f"  - Vocabulary size: {config.vocab_size}")
    print(f"  - Mel bins: {config.num_mel_bins}")
    print(f"  - Max source positions: {config.max_source_positions}")
    print(f"  - Max target positions: {config.max_target_positions}")
    print(f"{'=' * 80}\n")

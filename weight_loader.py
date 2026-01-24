"""
Weight loading utilities for Whisper NNX models.

This module provides functions to download pretrained weights from HuggingFace Hub
and load them into Flax NNX models.
"""

from flax import nnx
from transformers import WhisperConfig


def download_whisper_weights(model_name: str = "openai/whisper-tiny") -> tuple[dict, WhisperConfig]:
    """
    Download pretrained Whisper weights from HuggingFace Hub.

    Args:
        model_name: HuggingFace model identifier (e.g., "openai/whisper-tiny")

    Returns:
        Tuple of (weights_dict, config)
    """
    from transformers import FlaxWhisperForConditionalGeneration

    print(f"Downloading weights for {model_name}...")

    # Load the pretrained Flax model
    try:
        model = FlaxWhisperForConditionalGeneration.from_pretrained(
            model_name,
            from_pt=False,  # Try Flax weights first
        )
    except Exception:
        print("Flax weights not found, converting from PyTorch...")
        model = FlaxWhisperForConditionalGeneration.from_pretrained(
            model_name,
            from_pt=True,  # Convert from PyTorch if Flax not available
        )

    config = model.config

    # Extract parameters
    params = model.params

    print(
        f"Model config: {config.d_model}d, {config.encoder_layers} encoder layers, {config.decoder_layers} decoder layers"
    )

    return params, config


def map_huggingface_to_nnx(hf_params: dict, config: WhisperConfig) -> dict:
    """
    Map HuggingFace Flax parameters to NNX parameter structure.

    Args:
        hf_params: HuggingFace model parameters
        config: Whisper configuration

    Returns:
        Dictionary mapping NNX parameter paths to values
    """
    # Flatten HuggingFace parameters
    from flax.traverse_util import flatten_dict

    flat_hf = flatten_dict(hf_params, sep="/")

    print(f"\nTotal HuggingFace parameters: {len(flat_hf)}")
    print("Sample parameter keys:")
    for key in list(flat_hf.keys())[:10]:
        print(f"  {key}: {flat_hf[key].shape}")

    return flat_hf


def load_weights_into_nnx_model(model: nnx.Module, hf_params: dict, config: WhisperConfig):
    """
    Load HuggingFace weights into NNX model.

    This is a simplified loader that shows the concept. A full implementation
    would need to carefully map all parameter names between the two frameworks.

    Args:
        model: NNX Whisper model
        hf_params: HuggingFace Flax parameters
        config: Whisper configuration
    """
    from flax.traverse_util import flatten_dict

    # Get NNX parameters
    nnx_state = nnx.state(model)
    flat_nnx = flatten_dict(nnx_state, sep="/")

    print(f"\nNNX model parameters: {len(flat_nnx)}")
    print("Sample NNX parameter keys:")
    for key in list(flat_nnx.keys())[:10]:
        if hasattr(flat_nnx[key], "value"):
            print(f"  {key}: {flat_nnx[key].value.shape}")

    print(f"\n{'=' * 80}")
    print("PARAMETER MAPPING GUIDE")
    print(f"{'=' * 80}")
    print("\nHuggingFace uses the following structure:")
    print("  model/encoder/...")
    print("  model/decoder/...")
    print("\nNNX model uses:")
    print("  encoder/...")
    print("  decoder/...")
    print(f"{'=' * 80}\n")

    # This is a conceptual demonstration
    # A full implementation would need detailed parameter name mapping
    print("Note: Full parameter loading requires careful name mapping between frameworks.")
    print("This implementation demonstrates the structure. For production use,")
    print("implement detailed parameter name translation.")


def get_whisper_config(model_name: str) -> WhisperConfig:
    """Get Whisper configuration for a model."""
    from transformers import WhisperConfig

    config = WhisperConfig.from_pretrained(model_name)
    return config


def print_model_info(config: WhisperConfig):
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


if __name__ == "__main__":
    # Example: Download and inspect weights
    model_name = "openai/whisper-tiny"

    print(f"Loading {model_name}...\n")

    # Get configuration
    config = get_whisper_config(model_name)
    print_model_info(config)

    # Download weights
    params, config = download_whisper_weights(model_name)

    # Map to NNX format
    nnx_params = map_huggingface_to_nnx(params, config)

    print("\nWeight download and mapping demonstration complete!")
    print("\nNext steps:")
    print("1. Implement detailed parameter name mapping")
    print("2. Load parameters into NNX model")
    print("3. Verify model outputs match HuggingFace implementation")

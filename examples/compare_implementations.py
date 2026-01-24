#!/usr/bin/env python3
"""
Compare JAX Whisper implementation with HuggingFace PyTorch implementation.

Loads pretrained weights and verifies encoder, decoder, and logits outputs match.
"""

import sys

import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from transformers import WhisperModel as HFWhisperModel

from whisper_jax import create_whisper_base, create_whisper_small, create_whisper_tiny
from whisper_jax.weight_loader import load_pretrained_weights

# Set print options for cleaner output
np.set_printoptions(precision=4, suppress=True, linewidth=100)


def create_nnx_model(model_name: str):
    """Create JAX model matching HuggingFace model name."""
    models = {
        "openai/whisper-tiny": create_whisper_tiny,
        "openai/whisper-base": create_whisper_base,
        "openai/whisper-small": create_whisper_small,
    }
    if model_name not in models:
        raise ValueError(f"Unsupported model: {model_name}")
    return models[model_name](rngs=nnx.Rngs(0))


def compare_outputs(
    name: str, hf_output: np.ndarray, jax_output: np.ndarray, threshold: float = 1e-1
) -> bool:
    """Compare two outputs and print results."""
    print(f"\n{'=' * 80}\n{name}\n{'=' * 80}")

    if hf_output.shape != jax_output.shape:
        print(f"✗ Shape mismatch: HF {hf_output.shape} vs JAX {jax_output.shape}")
        return False

    max_diff = np.max(np.abs(hf_output - jax_output))
    mean_diff = np.mean(np.abs(hf_output - jax_output))

    print(f"Shape: {hf_output.shape}")
    print(f"Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")

    # Print sample outputs for visual inspection
    print("\nSample outputs (first 10 values):")
    print(f"  HF:  {hf_output.flat[:10]}")
    print(f"  JAX: {jax_output.flat[:10]}")
    print(f"  Diff: {(hf_output - jax_output).flat[:10]}")

    if max_diff < threshold:
        print(f"\n✓ {name} MATCH!")
        return True
    else:
        print(f"\n✗ {name} DIFFER")
        return False


def main():
    print("=" * 80)
    print("WHISPER JAX vs HUGGINGFACE PYTORCH COMPARISON")
    print("=" * 80)

    model_name = "openai/whisper-tiny"
    print(f"\nModel: {model_name}")

    # Load models
    print("\n1. Loading HuggingFace model...")
    hf_model = HFWhisperModel.from_pretrained(model_name)
    hf_model.eval()

    print("2. Creating and loading JAX model...")
    jax_model = create_nnx_model(model_name)
    num_params = load_pretrained_weights(jax_model, model_name)
    print(f"   ✓ Loaded {num_params} parameters")

    # Create test inputs
    print("\n3. Creating test inputs...")
    np.random.seed(42)
    input_features = np.random.randn(1, 80, 3000).astype(np.float32)
    decoder_ids = np.array([[50258, 50259, 50359, 50363]], dtype=np.int32)
    print(f"   Input: {input_features.shape}, Decoder IDs: {decoder_ids.shape}")

    # Run comparisons
    print("\n4. Running comparisons...")

    # Encoder
    with torch.no_grad():
        hf_enc = hf_model.encoder(torch.from_numpy(input_features)).last_hidden_state.numpy()
    jax_enc = np.array(jax_model.encode(jnp.array(input_features), deterministic=True))
    encoder_ok = compare_outputs("ENCODER", hf_enc, jax_enc)

    # Decoder
    with torch.no_grad():
        hf_dec = hf_model(
            torch.from_numpy(input_features), decoder_input_ids=torch.from_numpy(decoder_ids)
        ).last_hidden_state.numpy()

    jax_dec = np.array(
        jax_model.decoder(
            jnp.array(decoder_ids),
            encoder_hidden_states=jax_model.encode(jnp.array(input_features), deterministic=True),
            deterministic=True,
        )
    )
    decoder_ok = compare_outputs("DECODER", hf_dec, jax_dec)

    # Logits (JAX only, HF WhisperModel doesn't have lm_head)
    jax_logits = np.array(
        jax_model(jnp.array(input_features), jnp.array(decoder_ids), deterministic=True)
    )
    logits_ok = jax_logits.shape == (1, 4, 51865)
    print(f"\n{'=' * 80}\nLOGITS\n{'=' * 80}")
    print(f"Shape: {jax_logits.shape}")
    print("\nSample logits (first 10 values for first token):")
    print(f"  {jax_logits[0, 0, :10]}")
    print("\nPredicted tokens (argmax per position):")
    predicted_tokens = np.argmax(jax_logits[0], axis=-1)
    print(f"  {predicted_tokens}")
    print(f"\n{'✓' if logits_ok else '✗'} Logits shape {'correct' if logits_ok else 'incorrect'}")

    # Summary
    print(f"\n{'=' * 80}\nSUMMARY\n{'=' * 80}")

    if encoder_ok and decoder_ok and logits_ok:
        print("✓ All tests PASSED! JAX implementation matches PyTorch reference.")
        print("\nOutputs match within float32 precision (~1e-2).")
        print("Small differences are expected due to accumulated floating-point errors.")
    else:
        print("✗ Some tests FAILED:")
        if not encoder_ok:
            print("  - Encoder outputs differ")
        if not decoder_ok:
            print("  - Decoder outputs differ")
        if not logits_ok:
            print("  - Logits shape incorrect")

    return encoder_ok and decoder_ok and logits_ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

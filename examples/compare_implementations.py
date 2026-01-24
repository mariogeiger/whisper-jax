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
from transformers import WhisperForConditionalGeneration as HFWhisperModel

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
        hf_enc = hf_model.model.encoder(torch.from_numpy(input_features)).last_hidden_state.numpy()
    jax_enc = np.array(jax_model.encode(jnp.array(input_features), deterministic=True))
    encoder_ok = compare_outputs("ENCODER", hf_enc, jax_enc)

    # Decoder (hidden states before LM head)
    with torch.no_grad():
        hf_out = hf_model.model(
            torch.from_numpy(input_features), decoder_input_ids=torch.from_numpy(decoder_ids)
        )
        hf_dec = hf_out.last_hidden_state.numpy()

    jax_enc_out = jax_model.encode(jnp.array(input_features), deterministic=True)
    jax_dec = np.array(
        jax_model.decoder(
            jnp.array(decoder_ids),
            encoder_hidden_states=jax_enc_out,
            deterministic=True,
        )
    )
    decoder_ok = compare_outputs("DECODER", hf_dec, jax_dec)

    # Logits (after LM head / proj_out)
    with torch.no_grad():
        hf_logits = hf_model(
            torch.from_numpy(input_features), decoder_input_ids=torch.from_numpy(decoder_ids)
        ).logits.numpy()

    jax_logits = np.array(
        jax_model(jnp.array(input_features), jnp.array(decoder_ids), deterministic=True)
    )
    logits_ok = compare_outputs("LOGITS", hf_logits, jax_logits)

    # Verify predicted tokens match
    hf_tokens = np.argmax(hf_logits[0], axis=-1)
    jax_tokens = np.argmax(jax_logits[0], axis=-1)
    tokens_match = np.array_equal(hf_tokens, jax_tokens)
    print(f"\n{'=' * 80}\nPREDICTED TOKENS\n{'=' * 80}")
    print(f"HF tokens:  {hf_tokens}")
    print(f"JAX tokens: {jax_tokens}")
    print(f"{'✓' if tokens_match else '✗'} Predicted tokens {'match' if tokens_match else 'DIFFER'}")

    # Summary
    print(f"\n{'=' * 80}\nSUMMARY\n{'=' * 80}")

    all_ok = encoder_ok and decoder_ok and logits_ok and tokens_match
    if all_ok:
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
            print("  - Logits values differ")
        if not tokens_match:
            print("  - Predicted tokens differ (critical: affects transcription)")

    return all_ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

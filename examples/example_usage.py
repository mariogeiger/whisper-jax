#!/usr/bin/env python3
"""Example usage of the pure JAX Whisper implementation."""

import jax
import jax.numpy as jnp
from flax import nnx

from whisper_jax import (
    create_whisper_base,
    create_whisper_small,
    create_whisper_tiny,
    load_pretrained_weights,
)


def print_section(title):
    """Print section header."""
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def example_1_create_model():
    """Create a Whisper model from scratch."""
    print_section("EXAMPLE 1: Creating a model from scratch")

    rngs = nnx.Rngs(42)
    model = create_whisper_tiny(rngs=rngs)

    # Create dummy input
    input_features = jax.random.normal(rngs(), (1, 80, 3000))
    decoder_input_ids = jnp.array([[50258, 50259, 50359]])

    # Run inference
    encoder_output = model.encode(input_features, deterministic=True)
    logits = model.decode(decoder_input_ids, encoder_output, deterministic=True)

    print(f"Input: {input_features.shape}")
    print(f"Encoder output: {encoder_output.shape}")
    print(f"Logits: {logits.shape}")
    print(f"Predicted IDs: {jnp.argmax(logits, axis=-1)[0][:10]}")
    print("\n✓ Complete!")


def example_2_load_pretrained():
    """Load pretrained weights."""
    print_section("EXAMPLE 2: Loading pretrained weights")

    model_name = "openai/whisper-tiny"

    # Create and load model
    model = create_whisper_tiny(rngs=nnx.Rngs(42))
    num_params = load_pretrained_weights(model, model_name)
    print(f"✓ Loaded {num_params} parameters")

    # Test
    input_features = jax.random.normal(jax.random.PRNGKey(0), (1, 80, 3000))
    logits = model(input_features, jnp.array([[50258, 50259, 50359]]), deterministic=True)
    print(f"Logits shape: {logits.shape}")
    print("\n✓ Complete!")


def example_3_model_comparison():
    """Compare different model sizes."""
    print_section("EXAMPLE 3: Comparing model sizes")

    models = {
        "tiny": create_whisper_tiny,
        "base": create_whisper_base,
        "small": create_whisper_small,
    }

    rngs = nnx.Rngs(0)
    input_features = jax.random.normal(rngs(), (1, 80, 3000))

    for name, create_fn in models.items():
        model = create_fn(rngs=nnx.Rngs(0))
        params = nnx.state(model, nnx.Param)
        total = sum(
            x.value.size if hasattr(x, "value") else (x.size if hasattr(x, "size") else 0)
            for x in jax.tree.leaves(params)
        )
        encoder_out = model.encode(input_features, deterministic=True)
        print(f"{name.upper()}: {total:,} params, embed_dim={encoder_out.shape[-1]}")

    print("\n✓ Complete!")


def example_4_inference_pipeline():
    """Complete inference pipeline with autoregressive decoding."""
    print_section("EXAMPLE 4: Inference pipeline")

    model = create_whisper_tiny(rngs=nnx.Rngs(42))
    input_features = jax.random.normal(jax.random.PRNGKey(42), (1, 80, 3000))

    print("1. Encoding audio...")
    encoder_output = model.encode(input_features, deterministic=True)
    print(f"   Encoder output: {encoder_output.shape}")

    print("\n2. Autoregressive decoding...")
    generated = jnp.array([[50258, 50259, 50359]])  # Start tokens

    for step in range(10):
        logits = model.decode(generated, encoder_output, deterministic=True)
        next_token = jnp.argmax(logits[:, -1:, :], axis=-1)
        generated = jnp.concatenate([generated, next_token], axis=1)
        print(f"   Step {step + 1}: token {int(next_token[0, 0])}")

    print(f"\n   Generated {len(generated[0])} tokens: {generated[0]}")
    print("\n✓ Complete!")


def example_5_batch_processing():
    """Batch processing multiple inputs."""
    print_section("EXAMPLE 5: Batch processing")

    model = create_whisper_tiny(rngs=nnx.Rngs(42))
    batch_size = 4

    # Batch inputs
    input_features = jax.random.normal(jax.random.PRNGKey(0), (batch_size, 80, 3000))
    decoder_ids = jnp.tile(jnp.array([[50258, 50259, 50359]]), (batch_size, 1))

    # Batch forward
    encoder_out = model.encode(input_features, deterministic=True)
    logits = model.decode(decoder_ids, encoder_out, deterministic=True)

    print(f"Batch size: {batch_size}")
    print(f"Encoder output: {encoder_out.shape}")
    print(f"Logits: {logits.shape}")
    print(f"\n✓ Processed {batch_size} samples in parallel!")


def main():
    """Run all examples."""
    print_section("WHISPER JAX - EXAMPLES")

    examples = [
        example_1_create_model,
        example_2_load_pretrained,
        example_3_model_comparison,
        example_4_inference_pipeline,
        example_5_batch_processing,
    ]

    for i, example_fn in enumerate(examples, 1):
        try:
            example_fn()
        except Exception as e:
            print(f"\n✗ Example {i} failed: {e}")
            import traceback

            traceback.print_exc()

    print_section("ALL EXAMPLES COMPLETE!")


if __name__ == "__main__":
    main()

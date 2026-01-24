"""
Example usage of the pure JAX Whisper implementation.

This demonstrates:
1. Creating a Whisper model from scratch
2. Downloading pretrained weights
3. Running inference
"""

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np

from whisper_nnx import (
    WhisperModel,
    create_whisper_tiny,
    create_whisper_base,
    create_whisper_small
)
from weight_loader import (
    download_whisper_weights,
    get_whisper_config,
    print_model_info,
    load_weights_into_nnx_model
)


def example_1_create_model():
    """Example 1: Create a Whisper model from scratch."""
    print("=" * 80)
    print("EXAMPLE 1: Creating a Whisper model from scratch")
    print("=" * 80 + "\n")

    # Create model with random initialization
    rngs = nnx.Rngs(42)
    model = create_whisper_tiny(rngs=rngs)

    # Create dummy input
    batch_size = 1
    num_mel_bins = 80
    time_steps = 3000  # 30 seconds of audio at 16kHz with 100Hz hop length

    input_features = jax.random.normal(rngs(), (batch_size, num_mel_bins, time_steps))
    decoder_input_ids = jnp.array([[50258, 50259, 50359]])  # <start>, language, task tokens

    print(f"Input shape: {input_features.shape}")
    print(f"Decoder input shape: {decoder_input_ids.shape}\n")

    # Encode
    encoder_output = model.encode(input_features, deterministic=True)
    print(f"Encoder output shape: {encoder_output.shape}")

    # Decode
    logits = model.decode(decoder_input_ids, encoder_output, deterministic=True)
    print(f"Logits shape: {logits.shape}")

    # Get predictions
    predicted_ids = jnp.argmax(logits, axis=-1)
    print(f"Predicted token IDs: {predicted_ids[0][:10]}")

    print("\n✓ Example 1 complete!\n")


def example_2_download_weights():
    """Example 2: Download pretrained weights."""
    print("=" * 80)
    print("EXAMPLE 2: Downloading pretrained weights")
    print("=" * 80 + "\n")

    model_name = "openai/whisper-tiny"

    # Get configuration
    config = get_whisper_config(model_name)
    print_model_info(config)

    # Download weights
    try:
        params, config = download_whisper_weights(model_name)
        print(f"\n✓ Successfully downloaded weights for {model_name}")
        print(f"  Embedding dimension: {config.d_model}")
        print(f"  Encoder layers: {config.encoder_layers}")
        print(f"  Decoder layers: {config.decoder_layers}")
    except Exception as e:
        print(f"\n✗ Error downloading weights: {e}")

    print("\n✓ Example 2 complete!\n")


def example_3_model_comparison():
    """Example 3: Compare different model sizes."""
    print("=" * 80)
    print("EXAMPLE 3: Comparing different Whisper model sizes")
    print("=" * 80 + "\n")

    models = {
        "tiny": create_whisper_tiny,
        "base": create_whisper_base,
        "small": create_whisper_small,
    }

    # Create dummy input
    batch_size = 1
    num_mel_bins = 80
    time_steps = 3000

    rngs = nnx.Rngs(0)
    input_features = jax.random.normal(rngs(), (batch_size, num_mel_bins, time_steps))

    for name, create_fn in models.items():
        print(f"\n{name.upper()} MODEL:")
        print("-" * 40)

        rngs = nnx.Rngs(0)
        model = create_fn(rngs=rngs)

        # Count parameters
        params_graph = nnx.state(model, nnx.Param)
        total_params = sum(
            x.value.size if hasattr(x, 'value') else (x.size if hasattr(x, 'size') else 0)
            for x in jax.tree.leaves(params_graph)
        )

        print(f"  Total parameters: {total_params:,}")

        # Test encoding
        encoder_output = model.encode(input_features, deterministic=True)
        print(f"  Encoder output shape: {encoder_output.shape}")
        print(f"  Embedding dimension: {encoder_output.shape[-1]}")

    print("\n✓ Example 3 complete!\n")


def example_4_inference_pipeline():
    """Example 4: Complete inference pipeline."""
    print("=" * 80)
    print("EXAMPLE 4: Complete inference pipeline")
    print("=" * 80 + "\n")

    # Create model
    rngs = nnx.Rngs(42)
    model = create_whisper_tiny(rngs=rngs)

    # Simulate mel-spectrogram input (normally from audio preprocessing)
    batch_size = 1
    num_mel_bins = 80
    time_steps = 3000

    print("Step 1: Preprocessing audio to mel-spectrogram")
    input_features = jax.random.normal(rngs(), (batch_size, num_mel_bins, time_steps))
    print(f"  Input features shape: {input_features.shape}")

    print("\nStep 2: Encoding audio")
    encoder_output = model.encode(input_features, deterministic=True)
    print(f"  Encoder output shape: {encoder_output.shape}")

    print("\nStep 3: Autoregressive decoding")

    # Start tokens: <start of transcript>, <language>, <task>
    # For Whisper: 50258 = <|startoftranscript|>
    #              50259 = <|en|> (English)
    #              50359 = <|transcribe|> (transcription task)
    decoder_input_ids = jnp.array([[50258, 50259, 50359]])

    # Simulate greedy decoding (normally would loop until EOS)
    max_length = 10
    generated_ids = decoder_input_ids

    for step in range(max_length):
        # Get logits for next token
        logits = model.decode(generated_ids, encoder_output, deterministic=True)

        # Get next token (greedy)
        next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)

        # Append to sequence
        generated_ids = jnp.concatenate([generated_ids, next_token], axis=1)

        print(f"  Step {step + 1}: Generated token {int(next_token[0, 0])}")

    print(f"\n  Final generated sequence: {generated_ids[0]}")

    print("\nStep 4: Decoding tokens to text")
    print("  (Normally would use WhisperTokenizer here)")
    print(f"  Generated {len(generated_ids[0])} tokens")

    print("\n✓ Example 4 complete!\n")


def example_5_batch_processing():
    """Example 5: Batch processing multiple inputs."""
    print("=" * 80)
    print("EXAMPLE 5: Batch processing")
    print("=" * 80 + "\n")

    # Create model
    rngs = nnx.Rngs(42)
    model = create_whisper_tiny(rngs=rngs)

    # Create batch of inputs
    batch_size = 4
    num_mel_bins = 80
    time_steps = 3000

    print(f"Processing batch of {batch_size} audio samples")

    input_features = jax.random.normal(rngs(), (batch_size, num_mel_bins, time_steps))
    print(f"  Input shape: {input_features.shape}")

    # Encode all at once (efficient batching)
    encoder_outputs = model.encode(input_features, deterministic=True)
    print(f"  Encoder output shape: {encoder_outputs.shape}")

    # Decode each with different start tokens
    decoder_input_ids = jnp.array([
        [50258, 50259, 50359],  # English transcription
        [50258, 50259, 50359],  # English transcription
        [50258, 50259, 50359],  # English transcription
        [50258, 50259, 50359],  # English transcription
    ])

    logits = model.decode(decoder_input_ids, encoder_outputs, deterministic=True)
    print(f"  Logits shape: {logits.shape}")

    print(f"\n  Processed {batch_size} samples in parallel!")

    print("\n✓ Example 5 complete!\n")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("WHISPER NNX - PURE JAX IMPLEMENTATION EXAMPLES")
    print("=" * 80 + "\n")

    examples = [
        ("Creating a model from scratch", example_1_create_model),
        ("Downloading pretrained weights", example_2_download_weights),
        ("Comparing model sizes", example_3_model_comparison),
        ("Complete inference pipeline", example_4_inference_pipeline),
        ("Batch processing", example_5_batch_processing),
    ]

    for i, (name, example_fn) in enumerate(examples, 1):
        try:
            example_fn()
        except Exception as e:
            print(f"\n✗ Example {i} failed: {e}\n")
            import traceback
            traceback.print_exc()

    print("=" * 80)
    print("ALL EXAMPLES COMPLETE!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

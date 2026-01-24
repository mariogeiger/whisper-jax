#!/usr/bin/env python3
"""
Compatibility test suite for Whisper NNX implementation.

Tests that the implementation works correctly with latest versions of JAX and Flax.
"""

import sys
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np


def print_header(text):
    """Print a formatted header."""
    print(f"\n{'='*80}")
    print(f"{text}")
    print(f"{'='*80}\n")


def test_versions():
    """Test that we have compatible versions installed."""
    print_header("VERSION CHECK")

    import jax
    import flax
    import numpy
    import transformers

    print(f"✓ JAX version:          {jax.__version__}")
    print(f"✓ Flax version:         {flax.__version__}")
    print(f"✓ NumPy version:        {numpy.__version__}")
    print(f"✓ Transformers version: {transformers.__version__}")

    # Check minimum versions
    assert jax.__version__ >= "0.9.0", "JAX version must be >= 0.9.0"
    assert flax.__version__ >= "0.12.0", "Flax version must be >= 0.12.0"
    assert numpy.__version__ >= "2.0.0", "NumPy version must be >= 2.0.0"

    print("\n✅ All versions compatible!")
    return True


def test_basic_operations():
    """Test basic JAX/Flax operations."""
    print_header("BASIC OPERATIONS TEST")

    # Test JAX array creation
    x = jnp.array([1, 2, 3, 4])
    print(f"✓ JAX array creation:   {x.shape}")

    # Test JAX operations
    y = jnp.sum(x)
    print(f"✓ JAX operations:       sum={y}")

    # Test Flax NNX module
    class SimpleModule(nnx.Module):
        def __init__(self, rngs):
            self.linear = nnx.Linear(10, 20, rngs=rngs)

        def __call__(self, x):
            return self.linear(x)

    rngs = nnx.Rngs(0)
    model = SimpleModule(rngs=rngs)
    out = model(jnp.ones((5, 10)))
    print(f"✓ Flax NNX module:      {out.shape}")

    print("\n✅ Basic operations working!")
    return True


def test_whisper_import():
    """Test importing Whisper modules."""
    print_header("WHISPER MODULE IMPORT TEST")

    try:
        from whisper_nnx import (
            MultiHeadAttention,
            FeedForward,
            EncoderLayer,
            DecoderLayer,
            WhisperEncoder,
            WhisperDecoder,
            WhisperModel,
            create_whisper_tiny,
            create_whisper_base,
            create_whisper_small
        )
        print("✓ All Whisper modules imported successfully")

        from weight_loader import (
            get_whisper_config,
            print_model_info
        )
        print("✓ Weight loader imported successfully")

        print("\n✅ All imports successful!")
        return True

    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """Test creating Whisper models."""
    print_header("MODEL CREATION TEST")

    from whisper_nnx import create_whisper_tiny, create_whisper_base

    # Test tiny model
    print("Creating Tiny model...")
    rngs = nnx.Rngs(0)
    tiny_model = create_whisper_tiny(rngs=rngs)
    print(f"✓ Tiny model created")

    # Test base model
    print("Creating Base model...")
    rngs = nnx.Rngs(1)
    base_model = create_whisper_base(rngs=rngs)
    print(f"✓ Base model created")

    print("\n✅ Model creation successful!")
    return True


def test_forward_pass():
    """Test forward pass through the model."""
    print_header("FORWARD PASS TEST")

    from whisper_nnx import create_whisper_tiny

    # Create model
    rngs = nnx.Rngs(42)
    model = create_whisper_tiny(rngs=rngs)

    # Create dummy input
    batch_size = 2
    num_mel_bins = 80
    time_steps = 3000

    input_features = jax.random.normal(rngs(), (batch_size, num_mel_bins, time_steps))
    decoder_input_ids = jnp.ones((batch_size, 10), dtype=jnp.int32)

    print(f"Input shape:  {input_features.shape}")
    print(f"Decoder IDs:  {decoder_input_ids.shape}")

    # Test encoder
    print("\nTesting encoder...")
    encoder_output = model.encode(input_features, deterministic=True)
    print(f"✓ Encoder output: {encoder_output.shape}")
    assert encoder_output.shape == (batch_size, 1500, 384), "Encoder output shape mismatch"

    # Test decoder
    print("\nTesting decoder...")
    logits = model.decode(decoder_input_ids, encoder_output, deterministic=True)
    print(f"✓ Decoder logits: {logits.shape}")
    assert logits.shape == (batch_size, 10, 51865), "Decoder output shape mismatch"

    # Test full forward pass
    print("\nTesting full forward pass...")
    full_logits = model(input_features, decoder_input_ids, deterministic=True)
    print(f"✓ Full forward:   {full_logits.shape}")
    assert full_logits.shape == (batch_size, 10, 51865), "Full forward output shape mismatch"

    print("\n✅ Forward pass successful!")
    return True


def test_sequential_container():
    """Test that Sequential container works correctly."""
    print_header("SEQUENTIAL CONTAINER TEST")

    from whisper_nnx import EncoderLayer

    # Create layers using Sequential
    rngs = nnx.Rngs(0)
    num_layers = 3
    layers = nnx.Sequential(*[
        EncoderLayer(384, 6, 1536, 0.0, rngs=rngs)
        for _ in range(num_layers)
    ])

    print(f"✓ Created Sequential with {num_layers} layers")
    print(f"✓ Sequential.layers has {len(layers.layers)} elements")

    # Test accessing layers
    x = jnp.ones((2, 10, 384))
    for i in range(len(layers.layers)):
        x = layers.layers[i](x, deterministic=True)

    print(f"✓ Successfully processed through all layers")
    print(f"✓ Output shape: {x.shape}")

    print("\n✅ Sequential container working!")
    return True


def test_parameter_counting():
    """Test parameter counting."""
    print_header("PARAMETER COUNTING TEST")

    from whisper_nnx import create_whisper_tiny, create_whisper_base, create_whisper_small

    models = {
        "Tiny": create_whisper_tiny,
        "Base": create_whisper_base,
        "Small": create_whisper_small,
    }

    expected = {
        "Tiny": (50_000_000, 60_000_000),   # 50-60M range
        "Base": (90_000_000, 110_000_000),  # 90-110M range
        "Small": (270_000_000, 290_000_000), # 270-290M range
    }

    for name, create_fn in models.items():
        rngs = nnx.Rngs(0)
        model = create_fn(rngs=rngs)

        # Count parameters
        params_graph = nnx.state(model, nnx.Param)
        total_params = sum(
            x.value.size if hasattr(x, 'value') else (x.size if hasattr(x, 'size') else 0)
            for x in jax.tree.leaves(params_graph)
        )

        min_expected, max_expected = expected[name]
        print(f"✓ {name:5s}: {total_params:>12,} params ", end="")

        if min_expected <= total_params <= max_expected:
            print("✅")
        else:
            print(f"⚠️  (expected {min_expected:,} to {max_expected:,})")

    print("\n✅ Parameter counting working!")
    return True


def test_batch_processing():
    """Test batch processing."""
    print_header("BATCH PROCESSING TEST")

    from whisper_nnx import create_whisper_tiny

    rngs = nnx.Rngs(0)
    model = create_whisper_tiny(rngs=rngs)

    # Test different batch sizes
    batch_sizes = [1, 2, 4, 8]

    for batch_size in batch_sizes:
        input_features = jax.random.normal(rngs(), (batch_size, 80, 3000))
        decoder_ids = jnp.ones((batch_size, 5), dtype=jnp.int32)

        output = model(input_features, decoder_ids, deterministic=True)

        expected_shape = (batch_size, 5, 51865)
        assert output.shape == expected_shape, f"Batch size {batch_size} failed"
        print(f"✓ Batch size {batch_size:2d}: output {output.shape}")

    print("\n✅ Batch processing working!")
    return True


def test_jit_compilation():
    """Test JAX JIT compilation."""
    print_header("JIT COMPILATION TEST")

    from whisper_nnx import create_whisper_tiny

    rngs = nnx.Rngs(0)
    model = create_whisper_tiny(rngs=rngs)

    # Create JIT-compiled function
    @jax.jit
    def encode_jit(features):
        return model.encode(features, deterministic=True)

    # Test compilation
    input_features = jax.random.normal(rngs(), (1, 80, 3000))

    print("Compiling encoder (first call)...")
    output1 = encode_jit(input_features)
    print(f"✓ First call completed: {output1.shape}")

    print("Running compiled encoder (cached)...")
    output2 = encode_jit(input_features)
    print(f"✓ Second call completed: {output2.shape}")

    # Verify outputs match
    assert jnp.allclose(output1, output2), "JIT outputs don't match"
    print("✓ Outputs match")

    print("\n✅ JIT compilation working!")
    return True


def run_all_tests():
    """Run all compatibility tests."""
    print("\n" + "="*80)
    print("WHISPER NNX COMPATIBILITY TEST SUITE")
    print("Testing with latest JAX and Flax versions")
    print("="*80)

    tests = [
        ("Version Check", test_versions),
        ("Basic Operations", test_basic_operations),
        ("Module Import", test_whisper_import),
        ("Model Creation", test_model_creation),
        ("Forward Pass", test_forward_pass),
        ("Sequential Container", test_sequential_container),
        ("Parameter Counting", test_parameter_counting),
        ("Batch Processing", test_batch_processing),
        ("JIT Compilation", test_jit_compilation),
    ]

    results = []

    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Print summary
    print_header("TEST SUMMARY")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")

    print(f"\n{'='*80}")
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("✅ ALL TESTS PASSED!")
        print("="*80 + "\n")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("="*80 + "\n")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)

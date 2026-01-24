#!/usr/bin/env python3
"""
Real-time speech-to-text demo server using Whisper JAX.

Usage:
    pip install -e ".[demo]"
    python examples/demo_server.py
    # Open http://localhost:8000 in browser
"""

import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from whisper_jax import (
    create_whisper_tiny,
    load_pretrained_weights,
    load_whisper_vocab,
    log_mel_spectrogram,
)

# Configuration
MODEL_NAME = "openai/whisper-tiny"
SAMPLE_RATE = 16000
CHUNK_DURATION = 5.0  # seconds between transcriptions
MIN_AUDIO_LENGTH = 0.5  # minimum audio length to transcribe

# Global model and tokenizer
print("Loading Whisper model...")
model = create_whisper_tiny()
load_pretrained_weights(model, MODEL_NAME)
tokenizer = load_whisper_vocab(MODEL_NAME)
print("Model loaded!")


# Create FastAPI app
app = FastAPI(title="Whisper JAX Demo")

# Serve static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# JIT-compiled transcription function
@partial(jax.jit, static_argnames=["max_tokens"])
def transcribe_jit(
    audio: jax.Array,
    encoder,
    decoder,
    lm_head,
    max_tokens: int = 100,
) -> jax.Array:
    """
    JIT-compiled transcription: mel spectrogram + encoding + decoding.

    Returns token IDs as array.
    """
    print(f"Compiling transcribe_jit with shape {audio.shape} and max_tokens={max_tokens}")

    # Convert audio to mel spectrogram
    input_features = log_mel_spectrogram(audio)

    # Encode audio
    encoder_output = encoder(input_features, deterministic=True)

    # Start tokens: <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
    prompt_tokens = jnp.array([50258, 50259, 50359, 50363])
    prompt_len = 4

    # Pre-allocate fixed-size buffer for tokens (prompt + max_tokens)
    tokens = jnp.zeros((prompt_len + max_tokens,), dtype=jnp.int32)
    tokens = tokens.at[:prompt_len].set(prompt_tokens)

    # Autoregressive decoding with fixed-size carry
    def decode_step(carry, idx):
        toks, enc_out = carry
        # Pass full buffer - causal attention ensures future tokens don't affect current output
        seq = toks[None, :]  # Add batch dimension (1, total_len)

        logits = decoder(seq, enc_out, deterministic=True)
        logits = lm_head(logits)
        # Get logits at current position (prompt_len + idx - 1 is last valid, predict next)
        current_pos = prompt_len + idx - 1
        next_token = jnp.argmax(logits[0, current_pos, :])

        # Write next token to buffer
        new_toks = toks.at[prompt_len + idx].set(next_token)
        return (new_toks, enc_out), next_token

    (tokens, _), generated_tokens = jax.lax.scan(
        decode_step, (tokens, encoder_output), jnp.arange(max_tokens)
    )

    return tokens


def transcribe(audio: np.ndarray) -> str:
    """Transcribe audio using Whisper JAX."""
    if len(audio) < int(SAMPLE_RATE * MIN_AUDIO_LENGTH):
        return ""

    # Normalize audio
    audio = audio.astype(np.float32)
    if np.abs(audio).max() > 1.0:
        audio = audio / 32768.0  # Convert from int16 range

    # Run JIT-compiled transcription
    tokens = transcribe_jit(
        jnp.array(audio),
        model.encoder,
        model.decoder,
        model.lm_head,
        max_tokens=100,
    )

    # Decode tokens to text (skip prompt tokens)
    token_ids = np.array(tokens[4:])  # Skip first 4 prompt tokens

    # Stop at end token if present
    end_idx = np.where(token_ids == 50257)[0]
    if len(end_idx) > 0:
        token_ids = token_ids[: end_idx[0]]

    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return text


@app.get("/")
async def index():
    """Serve the main HTML page."""
    return FileResponse(static_dir / "index.html")


class AudioBuffer:
    """Buffer to accumulate audio chunks for segment-based transcription."""

    def __init__(self):
        self.buffer = np.array([], dtype=np.float32)

    def add_chunk(self, chunk: bytes) -> str | None:
        """Add audio chunk and return transcription if ready."""
        # Convert bytes to float32 array (expecting 16-bit PCM)
        audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        self.buffer = np.concatenate([self.buffer, audio])

        # Transcribe if we have enough audio, then clear buffer
        if len(self.buffer) >= int(SAMPLE_RATE * CHUNK_DURATION):
            start_time = time.perf_counter()
            text = transcribe(self.buffer)
            self.buffer = np.array([], dtype=np.float32)  # Clear after transcription
            elapsed = time.perf_counter() - start_time
            print(f"[segment {CHUNK_DURATION}s] Transcribed in {elapsed:.3f}s: {text}")
            return text if text else None

        return None

    def flush(self) -> str | None:
        """Transcribe remaining audio in buffer."""
        if len(self.buffer) >= int(SAMPLE_RATE * MIN_AUDIO_LENGTH):
            text = transcribe(self.buffer)
            self.buffer = np.array([], dtype=np.float32)
            return text if text else None
        self.buffer = np.array([], dtype=np.float32)
        return None

    def clear(self):
        """Clear the buffer."""
        self.buffer = np.array([], dtype=np.float32)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle real-time audio streaming via WebSocket."""
    await websocket.accept()
    buffer = AudioBuffer()

    try:
        await websocket.send_json({"type": "status", "message": "Connected"})

        while True:
            # Receive audio data
            data = await websocket.receive()

            # Check for disconnect
            if data["type"] == "websocket.disconnect":
                break

            if "bytes" in data:
                # Process audio chunk
                text = buffer.add_chunk(data["bytes"])
                if text:
                    await websocket.send_json({"type": "transcription", "text": text})

            elif "text" in data:
                msg = data["text"]
                if msg == "flush":
                    # Flush remaining audio
                    text = buffer.flush()
                    if text:
                        await websocket.send_json({"type": "transcription", "text": text})
                    await websocket.send_json({"type": "status", "message": "Stopped"})
                elif msg == "clear":
                    buffer.clear()
                    await websocket.send_json({"type": "status", "message": "Cleared"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        print("Client disconnected")


def main():
    """Run the demo server."""
    print("\n" + "=" * 60)
    print("Whisper JAX Demo Server")
    print("=" * 60)
    print("\nOpen http://localhost:8000 in your browser")
    print("Press Ctrl+C to stop\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

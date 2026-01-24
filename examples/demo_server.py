#!/usr/bin/env python3
"""Real-time speech-to-text demo server using Whisper JAX."""

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from whisper_jax import (
    LANG_TOKENS,
    create_whisper_base,
    create_whisper_large,
    create_whisper_medium,
    create_whisper_small,
    create_whisper_tiny,
    load_pretrained_weights,
    load_whisper_vocab,
    log_mel_spectrogram,
)

# Config
SAMPLE_RATE = 16000
CHUNK_DURATION = 5.0
MAX_AUDIO_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
MAX_TOKENS = 100

# Model configs
MODEL_CONFIGS = {
    "tiny": ("openai/whisper-tiny", create_whisper_tiny),
    "base": ("openai/whisper-base", create_whisper_base),
    "small": ("openai/whisper-small", create_whisper_small),
    "medium": ("openai/whisper-medium", create_whisper_medium),
    "large-v3": ("openai/whisper-large-v3", create_whisper_large),
}


class WhisperTranscriber:
    """Encapsulates Whisper model with JIT-compiled transcription.

    JIT compilation in JAX traces the function on first call for each unique
    combination of:
    - Input shapes (we fix this by padding audio to MAX_AUDIO_SAMPLES)
    - Static arguments (max_tokens is static, set at compile time)
    - Closed-over values (model weights are captured in the closure)

    By creating the JIT function as a closure over model components (rather than
    passing them as arguments), we avoid JAX re-tracing the model structure.
    The compiled function is cached and reused for all subsequent calls.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self._transcribe_fn = None  # JIT-compiled function (created per model)
        self._is_ready = False

    def load(self, name: str) -> None:
        """Load a model and prepare JIT-compiled transcription function."""
        if name not in MODEL_CONFIGS:
            name = "tiny"

        model_id, create_fn = MODEL_CONFIGS[name]
        print(f"Loading {name} model...")

        self.model = create_fn()
        load_pretrained_weights(self.model, model_id)
        self.tokenizer = load_whisper_vocab(model_id)
        self.model_name = name
        self._is_ready = False

        # Create JIT function as closure over model components.
        # This captures encoder/decoder/lm_head in the closure rather than
        # passing them as traced arguments, avoiding re-tracing on each call.
        encoder = self.model.encoder
        decoder = self.model.decoder
        lm_head = self.model.lm_head

        @jax.jit
        def _transcribe_jit(audio: jax.Array, lang_token: jax.Array) -> jax.Array:
            """JIT-compiled transcription with fixed shapes."""
            # This print only executes during JAX tracing (compilation), not at runtime
            print("[JIT TRACE] Compiling transcribe function...")
            mel = log_mel_spectrogram(audio)
            encoder_output = encoder(mel, deterministic=True)

            # Prompt: <|startoftranscript|><|lang|><|transcribe|><|notimestamps|>
            prompt = jnp.array([50258, lang_token, 50359, 50363], dtype=jnp.int32)
            tokens = jnp.zeros(4 + MAX_TOKENS, dtype=jnp.int32).at[:4].set(prompt)

            def decode_step(carry, idx):
                toks, enc = carry
                dec_out, _ = decoder(toks[None], enc, deterministic=True)
                logits = lm_head(dec_out)
                next_tok = jnp.argmax(logits[0, 3 + idx])
                return (toks.at[4 + idx].set(next_tok), enc), None

            (tokens, _), _ = jax.lax.scan(
                decode_step, (tokens, encoder_output), jnp.arange(MAX_TOKENS)
            )
            return tokens

        self._transcribe_fn = _transcribe_jit
        print(f"Model {name} loaded!")

    def warmup(self) -> None:
        """Trigger JIT compilation with dummy input.

        JAX compiles on first call, so we run once with zeros to:
        1. Compile the XLA graph (expensive, ~seconds)
        2. Cache it for all future calls with same shapes
        """
        if self._is_ready:
            return

        print("Warming up JIT compilation...")
        t0 = time.perf_counter()

        # Run with fixed-shape dummy input to trigger compilation
        dummy_audio = jnp.zeros(MAX_AUDIO_SAMPLES, dtype=jnp.float32)
        dummy_lang = jnp.array(LANG_TOKENS["en"], dtype=jnp.int32)
        _ = self._transcribe_fn(dummy_audio, dummy_lang).block_until_ready()

        self._is_ready = True
        print(f"JIT warmup complete in {time.perf_counter() - t0:.1f}s")

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def transcribe(self, audio: np.ndarray, lang: str = "en") -> str:
        """Transcribe audio to text.

        Audio is padded/truncated to fixed length to avoid JIT recompilation.
        """
        if len(audio) < SAMPLE_RATE * 0.5:
            return ""

        # Normalize audio
        audio = audio.astype(np.float32)
        if np.abs(audio).max() > 1.0:
            audio /= 32768.0

        # Pad or truncate to fixed length (avoids shape-based recompilation)
        if len(audio) < MAX_AUDIO_SAMPLES:
            audio = np.pad(audio, (0, MAX_AUDIO_SAMPLES - len(audio)))
        else:
            audio = audio[:MAX_AUDIO_SAMPLES]

        lang_token = jnp.array(LANG_TOKENS.get(lang, LANG_TOKENS["en"]), dtype=jnp.int32)
        tokens = np.array(self._transcribe_fn(jnp.array(audio), lang_token))
        tokens = tokens[4:]  # Skip prompt tokens

        # Stop at end-of-transcript token
        eot_positions = np.where(tokens == 50257)[0]
        if len(eot_positions):
            tokens = tokens[: eot_positions[0]]

        return self.tokenizer.decode(tokens, skip_special_tokens=True)


# Global transcriber instance
transcriber = WhisperTranscriber()
transcriber.load("tiny")


# FastAPI app
app = FastAPI(title="Whisper JAX Demo")
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def index():
    return FileResponse(static_dir / "index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    buffer = np.array([], dtype=np.float32)
    lang = "en"

    try:
        # Warmup JIT on first connection if needed
        if not transcriber.is_ready:
            await websocket.send_json({"type": "status", "message": "Warming up JIT..."})
            transcriber.warmup()

        await websocket.send_json({"type": "status", "message": "Ready"})

        while True:
            data = await websocket.receive()
            if data["type"] == "websocket.disconnect":
                break

            if "bytes" in data:
                audio = np.frombuffer(data["bytes"], dtype=np.int16).astype(np.float32) / 32768.0
                buffer = np.concatenate([buffer, audio])

                progress = min(len(buffer) / MAX_AUDIO_SAMPLES, 1.0)
                await websocket.send_json({"type": "progress", "value": progress})

                if len(buffer) >= MAX_AUDIO_SAMPLES:
                    await websocket.send_json({"type": "processing"})
                    t0 = time.perf_counter()
                    text = transcriber.transcribe(buffer, lang)
                    print(f"[{CHUNK_DURATION}s] {time.perf_counter() - t0:.2f}s [{lang}]: {text}")
                    buffer = np.array([], dtype=np.float32)
                    await websocket.send_json({"type": "transcription", "text": text or ""})

            elif "text" in data:
                msg = data["text"]
                if msg.startswith("{"):
                    parsed = json.loads(msg)
                    if parsed.get("type") == "lang":
                        lang = parsed.get("value", "en")
                        print(f"Language set to: {lang}")
                        await websocket.send_json(
                            {"type": "status", "message": f"Language: {lang}"}
                        )
                    elif parsed.get("type") == "model":
                        new_model = parsed.get("value", "tiny")
                        if new_model != transcriber.model_name:
                            await websocket.send_json(
                                {"type": "status", "message": f"Loading {new_model}..."}
                            )
                            transcriber.load(new_model)
                            await websocket.send_json(
                                {"type": "status", "message": "Warming up JIT..."}
                            )
                            transcriber.warmup()
                        await websocket.send_json({"type": "status", "message": "Ready"})
                elif msg == "flush":
                    if len(buffer) >= SAMPLE_RATE * 0.5:
                        await websocket.send_json({"type": "processing"})
                        text = transcriber.transcribe(buffer, lang)
                        await websocket.send_json({"type": "transcription", "text": text or ""})
                    buffer = np.array([], dtype=np.float32)
                    await websocket.send_json({"type": "status", "message": "Ready"})
                elif msg == "clear":
                    buffer = np.array([], dtype=np.float32)
                    await websocket.send_json({"type": "status", "message": "Cleared"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        print("Client disconnected")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Whisper JAX Demo - http://localhost:8000")
    print("=" * 50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)

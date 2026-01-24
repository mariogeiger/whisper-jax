#!/usr/bin/env python3
"""Real-time speech-to-text demo server using Whisper JAX."""

import json
import time
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from whisper_jax import SAMPLE_RATE, Whisper

# Config
CHUNK_DURATION = 5.0
MAX_AUDIO_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)


class WhisperTranscriber:
    """Encapsulates Whisper model for real-time transcription."""

    def __init__(self):
        self.whisper: Whisper | None = None
        self.model_name: str | None = None

    def load(self, name: str) -> None:
        """Load a model by name."""
        if name not in Whisper.available_models():
            name = "tiny"

        print(f"Loading {name} model...")
        self.whisper = Whisper.load(name)
        self.model_name = name
        print(f"Model {name} loaded!")

    def warmup(self) -> None:
        """Trigger JIT compilation with dummy input."""
        if self.whisper is None:
            return

        print("Warming up JIT compilation...")
        t0 = time.perf_counter()
        self.whisper.warmup()
        print(f"JIT warmup complete in {time.perf_counter() - t0:.1f}s")

    @property
    def is_ready(self) -> bool:
        return self.whisper is not None and self.whisper._warmed_up

    def transcribe(
        self, audio: np.ndarray, lang: str = "en", word_timestamps: bool = False
    ) -> dict:
        """Transcribe audio to text with optional word timestamps."""
        if self.whisper is None or len(audio) < SAMPLE_RATE * 0.5:
            return {"text": "", "words": []}

        result = self.whisper.transcribe(audio, language=lang, word_timestamps=word_timestamps)
        response = {"text": result.text}
        if word_timestamps and result.words:
            response["words"] = [
                {"word": w.word.strip(), "start": round(w.start, 2), "end": round(w.end, 2)}
                for w in result.words
            ]
        return response


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


async def do_warmup(websocket: WebSocket) -> None:
    """Warmup the model and notify client."""
    if not transcriber.is_ready:
        await websocket.send_json({"type": "status", "message": "Warming up JIT..."})
        transcriber.warmup()
    await websocket.send_json({"type": "status", "message": "Ready"})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    buffer = np.array([], dtype=np.float32)
    lang = "en"
    timestamps_enabled = False
    initialized = False

    try:
        # Tell client to send preferences, we'll warmup after
        await websocket.send_json({"type": "status", "message": "Waiting for preferences..."})

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
                    result = transcriber.transcribe(buffer, lang, timestamps_enabled)
                    print(
                        f"[{CHUNK_DURATION}s] {time.perf_counter() - t0:.2f}s [{lang}]: "
                        f"{result['text']}"
                    )
                    buffer = np.array([], dtype=np.float32)
                    await websocket.send_json({"type": "transcription", **result})

            elif "text" in data:
                msg = data["text"]
                if msg.startswith("{"):
                    parsed = json.loads(msg)
                    if parsed.get("type") == "lang":
                        lang = parsed.get("value", "en")
                        print(f"Language set to: {lang}")
                        if initialized:
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
                            if initialized:
                                await do_warmup(websocket)
                    elif parsed.get("type") == "timestamps":
                        timestamps_enabled = parsed.get("value", False)
                        print(f"Timestamps: {'on' if timestamps_enabled else 'off'}")
                elif msg == "init":
                    initialized = True
                    await do_warmup(websocket)
                elif msg == "flush":
                    if len(buffer) >= SAMPLE_RATE * 0.5:
                        await websocket.send_json({"type": "processing"})
                        result = transcriber.transcribe(buffer, lang, timestamps_enabled)
                        await websocket.send_json({"type": "transcription", **result})
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

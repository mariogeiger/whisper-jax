#!/usr/bin/env python3
"""Web interface for cleaning audio by removing filler words and silences.

Usage:
    python examples/clean_audio_server.py
    python examples/clean_audio_server.py --model small --port 8001
"""

import argparse
import io
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from whisper_jax import SAMPLE_RATE, Whisper, load_audio

# Filler words by language (from clean_audio.py)
FILLER_WORDS = {
    "en": {
        "um",
        "uh",
        "uhm",
        "uhh",
        "umm",
        "er",
        "err",
        "ah",
        "ahh",
        "hm",
        "hmm",
        "huh",
        "mhm",
        "uh-huh",
        "mm",
        "mmm",
    },
    "fr": {
        "euh",
        "euhh",
        "heu",
        "heuu",
        "bah",
        "ben",
        "hm",
        "hmm",
        "ah",
        "oh",
        "ouais",
    },
    "de": {
        "äh",
        "ähm",
        "öh",
        "öhm",
        "hm",
        "hmm",
        "na",
        "naja",
        "tja",
    },
    "es": {
        "eh",
        "ehh",
        "em",
        "emm",
        "ah",
        "ahh",
        "este",
        "pues",
        "bueno",
        "hm",
        "hmm",
        "mm",
    },
    "it": {
        "eh",
        "ehm",
        "uhm",
        "ah",
        "beh",
        "mah",
        "cioè",
        "allora",
        "hm",
        "hmm",
    },
    "pt": {
        "é",
        "eh",
        "hum",
        "humm",
        "ah",
        "ahh",
        "então",
        "tipo",
        "hm",
        "hmm",
    },
    "nl": {
        "eh",
        "ehm",
        "uh",
        "uhm",
        "hm",
        "hmm",
        "nou",
        "ja",
    },
    "pl": {
        "yyy",
        "eee",
        "eem",
        "hm",
        "hmm",
        "no",
        "znaczy",
    },
    "ru": {
        "э",
        "ээ",
        "эм",
        "ну",
        "вот",
        "это",
        "типа",
    },
    "ja": {
        "えー",
        "えーと",
        "あの",
        "その",
        "まあ",
        "うーん",
    },
    "zh": {
        "嗯",
        "呃",
        "那个",
        "这个",
        "就是",
    },
}


@dataclass
class WordSegment:
    """A word segment with timing and filler status."""

    start: float
    end: float
    text: str
    is_filler: bool = False


def get_filler_words(lang: str) -> set[str]:
    """Get filler words for a language, falling back to English."""
    return FILLER_WORDS.get(lang, FILLER_WORDS["en"])


def is_filler(word: str, lang: str = "en") -> bool:
    """Check if a word is a filler word in the given language."""
    filler_set = get_filler_words(lang)
    return word.lower().strip().rstrip(".,!?") in filler_set


def merge_segments(
    segments: list[WordSegment],
    max_gap: float = 0.3,
) -> list[WordSegment]:
    """Merge adjacent non-filler segments with small gaps."""
    if not segments:
        return []

    # Filter out filler segments (discarded words)
    kept = [s for s in segments if not s.is_filler]
    if not kept:
        return []

    merged = [WordSegment(start=kept[0].start, end=kept[0].end, text=kept[0].text)]
    for seg in kept[1:]:
        gap = seg.start - merged[-1].end
        if gap <= max_gap:
            # Merge with previous
            merged[-1] = WordSegment(
                start=merged[-1].start,
                end=seg.end,
                text=merged[-1].text + " " + seg.text,
            )
        else:
            merged.append(WordSegment(start=seg.start, end=seg.end, text=seg.text))

    return merged


def extract_and_join(
    audio: np.ndarray,
    segments: list[WordSegment],
    silence_duration: float = 0.2,
    crossfade_ms: int = 10,
) -> np.ndarray:
    """Extract segments and join with silence."""
    if not segments:
        return np.array([], dtype=np.float32)

    silence = np.zeros(int(silence_duration * SAMPLE_RATE), dtype=np.float32)
    fade_samples = int(crossfade_ms * SAMPLE_RATE / 1000)

    chunks = []
    for i, seg in enumerate(segments):
        # Add padding around segment
        start = max(0, int((seg.start - 0.02) * SAMPLE_RATE))
        end = min(len(audio), int((seg.end + 0.02) * SAMPLE_RATE))
        chunk = audio[start:end].copy()

        # Apply crossfade
        if len(chunk) > fade_samples * 2:
            chunk[:fade_samples] *= np.linspace(0, 1, fade_samples)
            chunk[-fade_samples:] *= np.linspace(1, 0, fade_samples)

        if i > 0:
            chunks.append(silence)
        chunks.append(chunk)

    return np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)


def compute_waveform_peaks(audio: np.ndarray, num_peaks: int = 1000) -> list[float]:
    """Downsample audio to peaks for waveform visualization."""
    if len(audio) == 0:
        return []

    # Compute absolute values for peak detection
    abs_audio = np.abs(audio)

    # Downsample to num_peaks points
    chunk_size = max(1, len(audio) // num_peaks)
    peaks = []

    for i in range(0, len(audio), chunk_size):
        chunk = abs_audio[i : i + chunk_size]
        if len(chunk) > 0:
            peaks.append(float(np.max(chunk)))

    # Normalize to 0-1 range
    max_peak = max(peaks) if peaks else 1.0
    if max_peak > 0:
        peaks = [p / max_peak for p in peaks]

    return peaks


# Storage for uploaded audio files
audio_storage: dict[str, tuple[np.ndarray, str]] = {}


class WhisperTranscriber:
    """Encapsulates Whisper model for transcription."""

    def __init__(self):
        self.whisper: Whisper | None = None
        self.model_name: str | None = None
        self._warmed_up: bool = False

    def load(self, name: str) -> None:
        """Load a model by name."""
        if name not in Whisper.available_models():
            name = "tiny"

        if name == self.model_name and self.whisper is not None:
            return

        print(f"Loading {name} model...")
        self.whisper = Whisper.load(name)
        self.model_name = name
        self._warmed_up = False
        print(f"Model {name} loaded!")

    def warmup(self) -> None:
        """Trigger JIT compilation with dummy input."""
        if self.whisper is None:
            return

        if not self._warmed_up:
            print("Warming up JIT compilation...")
            self.whisper.warmup(word_timestamps=True)
            self._warmed_up = True
            print("JIT warmup complete!")

    def transcribe(self, audio: np.ndarray, lang: str = "en") -> list[dict]:
        """Transcribe audio and return words with timestamps."""
        if self.whisper is None or len(audio) < SAMPLE_RATE * 0.5:
            return []

        self.warmup()
        result = self.whisper.transcribe(audio, language=lang, word_timestamps=True)

        words = []
        if result.words:
            for w in result.words:
                word_text = w.word.strip()
                words.append(
                    {
                        "word": word_text,
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "is_filler": is_filler(word_text, lang),
                    }
                )

        return words


# Global transcriber instance
transcriber = WhisperTranscriber()

# FastAPI app
app = FastAPI(title="Whisper JAX Audio Cleaner")
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def index():
    return FileResponse(static_dir / "clean_audio.html")


@app.post("/upload")
async def upload_audio(file: Annotated[UploadFile, File()]):
    """Upload audio file and return waveform data."""
    # Read uploaded file
    content = await file.read()

    # Save to temp file for loading
    with tempfile.NamedTemporaryFile(
        suffix=Path(file.filename or "audio").suffix, delete=False
    ) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Load audio
        audio = load_audio(tmp_path)
        duration = len(audio) / SAMPLE_RATE

        # Generate waveform peaks
        peaks = compute_waveform_peaks(audio)

        # Store audio with unique ID
        audio_id = str(uuid.uuid4())
        audio_storage[audio_id] = (audio, file.filename or "audio")

        return {
            "audio_id": audio_id,
            "filename": file.filename,
            "duration": round(duration, 2),
            "peaks": peaks,
        }
    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/transcribe")
async def transcribe_audio(
    audio_id: str = Form(...),
    language: str = Form("en"),
    model: str = Form("tiny"),
):
    """Transcribe uploaded audio and return words with timestamps."""
    if audio_id not in audio_storage:
        return {"error": "Audio not found. Please upload again."}

    audio, _ = audio_storage[audio_id]

    # Load model if needed
    transcriber.load(model)

    # Transcribe
    words = transcriber.transcribe(audio, language)

    return {"words": words}


@app.post("/export")
async def export_audio(
    audio_id: str = Form(...),
    segments: str = Form(...),
    max_gap: float = Form(0.3),
    format: str = Form("wav"),
):
    """Export cleaned audio with only kept segments."""
    import json

    if audio_id not in audio_storage:
        return {"error": "Audio not found. Please upload again."}

    audio, original_filename = audio_storage[audio_id]

    # Parse segments from JSON
    kept_segments = json.loads(segments)

    # Convert to WordSegment objects
    word_segments = [
        WordSegment(start=s["start"], end=s["end"], text=s["word"], is_filler=False)
        for s in kept_segments
    ]

    # Merge segments
    merged = merge_segments(word_segments, max_gap=max_gap)

    # Extract and join audio
    cleaned = extract_and_join(audio, merged, silence_duration=max_gap)

    if len(cleaned) == 0:
        return {"error": "No audio segments to export"}

    # Generate output filename
    base_name = Path(original_filename).stem if original_filename else "audio"
    output_filename = f"{base_name}_cleaned.{format}"

    # Write to buffer
    buffer = io.BytesIO()

    if format == "mp3":
        # Write as WAV first, then convert (requires ffmpeg)
        import subprocess

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            sf.write(tmp_wav.name, cleaned, SAMPLE_RATE)
            tmp_wav_path = tmp_wav.name

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
            tmp_mp3_path = tmp_mp3.name

        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_wav_path, "-b:a", "192k", tmp_mp3_path],
                capture_output=True,
                check=True,
            )
            with open(tmp_mp3_path, "rb") as f:
                buffer.write(f.read())
        finally:
            Path(tmp_wav_path).unlink(missing_ok=True)
            Path(tmp_mp3_path).unlink(missing_ok=True)

        media_type = "audio/mpeg"
    else:
        sf.write(buffer, cleaned, SAMPLE_RATE, format="WAV")
        media_type = "audio/wav"

    buffer.seek(0)

    return Response(
        content=buffer.read(),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{output_filename}"'},
    )


@app.on_event("startup")
async def startup():
    """Load default model on startup."""
    transcriber.load("tiny")


def main():
    parser = argparse.ArgumentParser(description="Audio Cleaner Web Server")
    parser.add_argument("--model", default="tiny", help="Initial model to load")
    parser.add_argument("--port", type=int, default=8000, help="Port to run on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    transcriber.load(args.model)

    print("\n" + "=" * 50)
    print(f"Audio Cleaner - http://localhost:{args.port}")
    print("=" * 50 + "\n")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Clean audio by removing filler words and excessive silences.

Usage:
    python examples/clean_audio.py audio.mp3
    python examples/clean_audio.py audio.mp3 --model small --max-silence 300
    python examples/clean_audio.py audio.mp3 --keep-fillers --lang fr
"""

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from whisper_jax import SAMPLE_RATE, Whisper, load_audio
from whisper_jax.alignment import get_vad_speech_segments

# Filler words (hesitations and verbal tics)
# Note: "like", "so", "well" are only fillers when standalone
FILLER_WORDS = {
    "um", "uh", "uhm", "uhh", "umm", "er", "err", "ah", "ahh",
    "euh", "hm", "hmm", "huh", "mhm", "uh-huh", "mm", "mmm",
}


@dataclass
class SpeechSegment:
    """A segment of speech to keep."""
    start: float
    end: float
    text: str
    is_filler: bool = False


def is_filler(word: str) -> bool:
    """Check if a word is a filler word."""
    return word.lower().strip().rstrip(".,!?") in FILLER_WORDS


def merge_segments(
    segments: list[SpeechSegment],
    max_gap: float = 0.3,
) -> list[SpeechSegment]:
    """Merge adjacent non-filler segments with small gaps."""
    if not segments:
        return []

    # Filter out filler-only segments
    kept = [s for s in segments if not s.is_filler]
    if not kept:
        return []

    merged = [kept[0]]
    for seg in kept[1:]:
        gap = seg.start - merged[-1].end
        if gap <= max_gap:
            # Merge with previous
            merged[-1] = SpeechSegment(
                start=merged[-1].start,
                end=seg.end,
                text=merged[-1].text + " " + seg.text,
            )
        else:
            merged.append(seg)

    return merged


def extract_and_join(
    audio: np.ndarray,
    segments: list[SpeechSegment],
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


def export_audio(audio: np.ndarray, output_path: str, sample_rate: int = SAMPLE_RATE) -> None:
    """Export audio to file (uses ffmpeg for mp3)."""
    temp_wav = "/tmp/cleaned_temp.wav"
    sf.write(temp_wav, audio, sample_rate)

    if output_path.endswith(".mp3"):
        subprocess.run(
            ["ffmpeg", "-y", "-i", temp_wav, "-b:a", "192k", output_path],
            capture_output=True,
            check=True,
        )
        Path(temp_wav).unlink()
    else:
        Path(temp_wav).rename(output_path)


def main():
    parser = argparse.ArgumentParser(description="Clean audio by removing fillers and silences")
    parser.add_argument("audio_file", help="Input audio file")
    parser.add_argument(
        "--model",
        default="small",
        choices=Whisper.available_models(),
        help="Model size (default: small)",
    )
    parser.add_argument("--lang", default="en", help="Language code (default: en)")
    parser.add_argument(
        "--max-silence",
        type=int,
        default=300,
        help="Max silence between segments in ms (default: 300)",
    )
    parser.add_argument(
        "--keep-fillers",
        action="store_true",
        help="Keep filler words (only remove silences)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file (default: {input}_cleaned.mp3)",
    )
    args = parser.parse_args()

    input_path = Path(args.audio_file)
    output_path = args.output or str(
        input_path.with_stem(input_path.stem + "_cleaned").with_suffix(".mp3")
    )

    # Load audio
    print(f"Loading: {input_path}")
    audio = load_audio(str(input_path))
    duration = len(audio) / SAMPLE_RATE
    print(f"  Duration: {duration:.2f}s")

    # Transcribe
    print(f"\nTranscribing with {args.model} model...")
    whisper = Whisper.load(args.model)
    result = whisper.transcribe(str(input_path), language=args.lang, word_timestamps=True)

    if not result.words:
        print("No speech detected.")
        return

    print(f"  Found {len(result.words)} words")
    print(f"  Text: {result.text[:60]}{'...' if len(result.text) > 60 else ''}")

    # Get VAD segments
    print("\nAnalyzing speech segments...")
    vad_segments = get_vad_speech_segments(audio)
    print(f"  VAD found {len(vad_segments)} speech segments")

    # Map words to VAD segments and detect fillers
    segments = []
    for vad_start, vad_end in vad_segments:
        # Find words that fall within this VAD segment
        words_in_segment = []
        for w in result.words:
            word_center = (w.start + w.end) / 2
            if vad_start <= word_center <= vad_end:
                words_in_segment.append(w.word.strip())

        # Check if segment is entirely fillers
        if words_in_segment:
            filler_count = sum(1 for w in words_in_segment if is_filler(w))
            # Only mark as filler if ALL words are fillers
            is_filler_segment = filler_count == len(words_in_segment) and filler_count > 0
            text = " ".join(words_in_segment)
        else:
            is_filler_segment = False
            text = ""

        segments.append(SpeechSegment(
            start=vad_start,
            end=vad_end,
            text=text,
            is_filler=is_filler_segment and not args.keep_fillers,
        ))

    # Report fillers found
    filler_segments = [s for s in segments if s.is_filler]
    if filler_segments and not args.keep_fillers:
        print(f"\nRemoving {len(filler_segments)} filler segment(s):")
        for s in filler_segments:
            print(f"    [{s.start:.2f}s - {s.end:.2f}s] \"{s.text}\"")

    # Merge segments
    max_gap = args.max_silence / 1000
    merged = merge_segments(segments, max_gap=max_gap)
    print(f"\nKept {len(merged)} segment(s) after merging")

    # Extract and export
    print("\nExtracting audio...")
    cleaned = extract_and_join(audio, merged, silence_duration=max_gap)

    if len(cleaned) == 0:
        print("No audio to keep!")
        return

    cleaned_duration = len(cleaned) / SAMPLE_RATE
    saved = duration - cleaned_duration

    print(f"\nExporting: {output_path}")
    export_audio(cleaned, output_path)

    print("\nResults:")
    print(f"  Original:  {duration:.2f}s")
    print(f"  Cleaned:   {cleaned_duration:.2f}s")
    print(f"  Saved:     {saved:.2f}s ({100 * saved / duration:.1f}%)")


if __name__ == "__main__":
    main()

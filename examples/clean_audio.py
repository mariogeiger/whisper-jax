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

# Filler words by language (hesitations and verbal tics)
FILLER_WORDS = {
    "en": {
        "um", "uh", "uhm", "uhh", "umm", "er", "err", "ah", "ahh",
        "hm", "hmm", "huh", "mhm", "uh-huh", "mm", "mmm",
    },
    "fr": {
        "euh", "euhh", "heu", "heuu", "bah", "ben", "hm", "hmm",
        "ah", "oh", "ouais",
    },
    "de": {
        "äh", "ähm", "öh", "öhm", "hm", "hmm", "na", "naja", "tja",
    },
    "es": {
        "eh", "ehh", "em", "emm", "ah", "ahh", "este", "pues", "bueno",
        "hm", "hmm", "mm",
    },
    "it": {
        "eh", "ehm", "uhm", "ah", "beh", "mah", "cioè", "allora",
        "hm", "hmm",
    },
    "pt": {
        "é", "eh", "hum", "humm", "ah", "ahh", "então", "tipo",
        "hm", "hmm",
    },
    "nl": {
        "eh", "ehm", "uh", "uhm", "hm", "hmm", "nou", "ja",
    },
    "pl": {
        "yyy", "eee", "eem", "hm", "hmm", "no", "znaczy",
    },
    "ru": {
        "э", "ээ", "эм", "ну", "вот", "это", "типа",
    },
    "ja": {
        "えー", "えーと", "あの", "その", "まあ", "うーん",
    },
    "zh": {
        "嗯", "呃", "那个", "这个", "就是",
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

    # Filter out filler segments
    kept = [s for s in segments if not s.is_filler]
    if not kept:
        return []

    merged = [kept[0]]
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
            merged.append(seg)

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
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Show timing breakdown",
    )
    args = parser.parse_args()

    input_path = Path(args.audio_file)
    output_path = args.output or str(
        input_path.with_stem(input_path.stem + "_cleaned").with_suffix(".mp3")
    )

    # Show language info
    if args.lang not in FILLER_WORDS:
        print(f"Note: No filler words for '{args.lang}', using English fillers")

    # Load audio
    print(f"Loading: {input_path}")
    audio = load_audio(str(input_path))
    duration = len(audio) / SAMPLE_RATE
    print(f"  Duration: {duration:.2f}s")

    # Transcribe with refined word timestamps
    print(f"\nTranscribing with {args.model} model...")
    whisper = Whisper.load(args.model)
    if args.profile:
        whisper.warmup(word_timestamps=True)
    result = whisper.transcribe(str(input_path), language=args.lang, word_timestamps=True, _profile=args.profile)

    if not result.words:
        print("No speech detected.")
        return

    print(f"  Found {len(result.words)} words")
    print(f"  Text: {result.text[:60]}{'...' if len(result.text) > 60 else ''}")

    # Build word segments directly from refined timestamps
    segments = []
    filler_count = 0

    print("\nWord analysis:")
    for w in result.words:
        word_text = w.word.strip()
        is_filler_word = is_filler(word_text, args.lang) and not args.keep_fillers
        status = "DISCARD" if is_filler_word else "KEEP"
        print(f"  [{w.start:6.2f}s - {w.end:6.2f}s] {status:8} \"{word_text}\"")

        if is_filler_word:
            filler_count += 1

        segments.append(WordSegment(
            start=w.start,
            end=w.end,
            text=word_text,
            is_filler=is_filler_word,
        ))

    if filler_count > 0:
        print(f"\nRemoving {filler_count} filler word(s)")

    # Merge segments
    max_gap = args.max_silence / 1000
    merged = merge_segments(segments, max_gap=max_gap)
    print(f"Kept {len(merged)} segment(s) after merging")

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

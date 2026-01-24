#!/usr/bin/env python3
"""Transcribe audio with word-level timestamps.

Usage:
    python examples/word_timestamps.py audio.mp3
    python examples/word_timestamps.py audio.mp3 --model base --lang fr
    python examples/word_timestamps.py audio.mp3 --format srt > subtitles.srt
    python examples/word_timestamps.py audio.mp3 --format json > timestamps.json
"""

import argparse
import json
import sys
import time

from whisper_jax import Whisper, WordTiming


def format_srt_time(seconds: float) -> str:
    """Format time as SRT timestamp: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def words_to_srt(words: list[WordTiming], words_per_line: int = 7) -> str:
    """Convert word timings to SRT subtitle format."""
    lines = []
    idx = 1

    for i in range(0, len(words), words_per_line):
        group = words[i : i + words_per_line]
        if not group:
            continue

        start = group[0].start
        end = group[-1].end
        text = " ".join(w.word.strip() for w in group)

        lines.append(f"{idx}")
        lines.append(f"{format_srt_time(start)} --> {format_srt_time(end)}")
        lines.append(text)
        lines.append("")
        idx += 1

    return "\n".join(lines)


def words_to_json(words: list[WordTiming]) -> str:
    """Convert word timings to JSON format."""
    output = [
        {
            "word": w.word.strip(),
            "start": round(w.start, 3),
            "end": round(w.end, 3),
            "probability": round(w.probability, 3),
        }
        for w in words
    ]
    return json.dumps(output, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio with word-level timestamps")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument(
        "--model",
        default="small",
        choices=Whisper.available_models(),
        help="Model size (default: small)",
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="Language code (default: en)",
    )
    parser.add_argument(
        "--format",
        default="text",
        choices=["text", "json", "srt"],
        help="Output format (default: text)",
    )
    args = parser.parse_args()

    # Load model
    if args.format == "text":
        print(f"Loading {args.model} model...", file=sys.stderr)
    whisper = Whisper.load(args.model)

    # Transcribe with word timestamps
    if args.format == "text":
        print(f"Transcribing: {args.audio_file}", file=sys.stderr)

    t0 = time.perf_counter()
    result = whisper.transcribe(
        args.audio_file,
        language=args.lang,
        word_timestamps=True,
    )
    elapsed = time.perf_counter() - t0

    if not result.words:
        if args.format == "text":
            print("No speech detected.", file=sys.stderr)
        return

    # Output in requested format
    if args.format == "json":
        print(words_to_json(result.words))
    elif args.format == "srt":
        print(words_to_srt(result.words))
    else:
        # Text format with detailed output
        print(f"\nTranscription ({elapsed:.2f}s):", file=sys.stderr)
        print(f"  {result.text}", file=sys.stderr)

        print(f"\nWord-level timestamps ({len(result.words)} words):", file=sys.stderr)
        print("-" * 60, file=sys.stderr)
        for w in result.words:
            print(
                f"  [{w.start:6.2f}s - {w.end:6.2f}s] {w.word:<20} (p={w.probability:.2f})",
                file=sys.stderr,
            )

        # Also output JSON to stdout for easy piping
        print("\n" + words_to_json(result.words))


if __name__ == "__main__":
    main()

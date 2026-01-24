#!/usr/bin/env python3
"""Transcribe audio with word-level timestamps using DTW alignment.

Uses Dynamic Time Warping (DTW) on cross-attention weights to align
decoded tokens to audio frames, providing accurate word-level timestamps.

Usage:
    python examples/word_timestamps.py audio.mp3
    python examples/word_timestamps.py audio.mp3 --model base --lang fr
    python examples/word_timestamps.py audio.mp3 --format srt > subtitles.srt
    python examples/word_timestamps.py audio.mp3 --format json > timestamps.json
"""

import argparse
import json
import subprocess
import sys
import time

import jax.numpy as jnp
import numpy as np

from whisper_jax import (
    LANG_TOKENS,
    create_alignment_fn,
    create_transcribe_fn,
    create_whisper_base,
    create_whisper_small,
    create_whisper_tiny,
    get_word_timestamps,
    load_pretrained_weights,
    load_whisper_vocab,
)

SAMPLE_RATE = 16000
MAX_AUDIO_SAMPLES = SAMPLE_RATE * 30  # 30 seconds

MODEL_CONFIGS = {
    "tiny": ("openai/whisper-tiny", create_whisper_tiny),
    "base": ("openai/whisper-base", create_whisper_base),
    "small": ("openai/whisper-small", create_whisper_small),
}


def load_audio(path: str) -> np.ndarray:
    """Load audio file using ffmpeg and convert to 16kHz mono float32."""
    cmd = [
        "ffmpeg", "-i", path,
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-ar", str(SAMPLE_RATE), "-ac", "1",
        "-loglevel", "error", "-"
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")
    return np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def extract_text_tokens(tokens: jnp.ndarray, num_generated: int) -> list[int]:
    """Extract text tokens from full token buffer."""
    return [int(t) for t in tokens[4 : 4 + num_generated] if t != 50257 and t < 50257]


def format_srt_time(seconds: float) -> str:
    """Format time as SRT timestamp: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def words_to_srt(words, words_per_line: int = 7) -> str:
    """Convert word timings to SRT subtitle format."""
    lines = []
    idx = 1

    for i in range(0, len(words), words_per_line):
        group = words[i:i + words_per_line]
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


def words_to_json(words) -> str:
    """Convert word timings to JSON format."""
    output = [
        {
            "word": w.word.strip(),
            "start": round(w.start, 3),
            "end": round(w.end, 3),
            "probability": round(w.probability, 3)
        }
        for w in words
    ]
    return json.dumps(output, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio with word-level timestamps using DTW alignment"
    )
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--model", default="tiny", choices=MODEL_CONFIGS.keys(),
                        help="Model size (default: tiny)")
    parser.add_argument("--lang", default="en", choices=LANG_TOKENS.keys(),
                        help="Language code (default: en)")
    parser.add_argument("--format", default="text", choices=["text", "json", "srt"],
                        help="Output format (default: text)")
    args = parser.parse_args()

    # Load model
    model_id, create_fn = MODEL_CONFIGS[args.model]
    if args.format == "text":
        print(f"Loading {args.model} model...", file=sys.stderr)
    model = create_fn()
    load_pretrained_weights(model, model_id)
    tokenizer = load_whisper_vocab(model_id)

    # Create JIT-compiled functions (captures model in closure for speed)
    transcribe_fn = create_transcribe_fn(model, max_tokens=200)
    alignment_fn = create_alignment_fn(model)

    # Load audio
    if args.format == "text":
        print(f"Loading audio: {args.audio_file}", file=sys.stderr)
    audio = load_audio(args.audio_file)
    duration = len(audio) / SAMPLE_RATE

    # Process in 30-second chunks
    all_words = []
    chunk_size = MAX_AUDIO_SAMPLES
    num_chunks = (len(audio) + chunk_size - 1) // chunk_size

    if args.format == "text":
        print(f"Duration: {duration:.1f}s ({num_chunks} chunk(s))", file=sys.stderr)
        print("Processing...", file=sys.stderr)

    t0 = time.perf_counter()

    for chunk_idx in range(num_chunks):
        start_sample = chunk_idx * chunk_size
        end_sample = min(start_sample + chunk_size, len(audio))
        chunk = audio[start_sample:end_sample]
        time_offset = start_sample / SAMPLE_RATE

        # Pad chunk to 30 seconds
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

        # Transcribe using JIT-compiled function
        lang_token = LANG_TOKENS[args.lang]
        tokens, num_gen = transcribe_fn(jnp.array(chunk), jnp.array(lang_token))
        text_tokens = extract_text_tokens(tokens, int(num_gen))

        if not text_tokens:
            continue

        # Get word timestamps (reuse alignment_fn for speed)
        words = get_word_timestamps(
            model=model,
            tokenizer=tokenizer,
            audio=chunk,
            text_tokens=text_tokens,
            model_name=args.model,
            _alignment_fn=alignment_fn,
        )

        # Apply time offset for multi-chunk processing
        for w in words:
            w.start += time_offset
            w.end += time_offset
            all_words.append(w)

    elapsed = time.perf_counter() - t0

    if not all_words:
        if args.format == "text":
            print("No speech detected.", file=sys.stderr)
        return

    # Output in requested format
    if args.format == "json":
        print(words_to_json(all_words))
    elif args.format == "srt":
        print(words_to_srt(all_words))
    else:
        # Text format with detailed output
        full_text = " ".join(w.word.strip() for w in all_words)
        print(f"\nTranscription ({elapsed:.2f}s):", file=sys.stderr)
        print(f"  {full_text}", file=sys.stderr)

        print(f"\nWord-level timestamps ({len(all_words)} words):", file=sys.stderr)
        print("-" * 60, file=sys.stderr)
        for w in all_words:
            print(f"  [{w.start:6.2f}s - {w.end:6.2f}s] {w.word:<20} (p={w.probability:.2f})",
                  file=sys.stderr)

        # Also output JSON to stdout for easy piping
        print("\n" + words_to_json(all_words))


if __name__ == "__main__":
    main()

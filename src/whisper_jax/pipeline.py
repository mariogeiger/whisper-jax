"""High-level Whisper transcription API."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from whisper_jax.core.audio import N_SAMPLES, SAMPLE_RATE
from whisper_jax.core.decode import create_alignment_fn, create_transcribe_fn, get_alignment_mask
from whisper_jax.core.model import (
    WhisperModel,
    create_whisper_base,
    create_whisper_large,
    create_whisper_medium,
    create_whisper_small,
    create_whisper_tiny,
)
from whisper_jax.utils.audio_io import load_audio, normalize_audio
from whisper_jax.utils.tokenizer import (
    EOT,
    FRAMES_PER_SECOND,
    INPUT_STRIDE,
    LANG_TOKENS,
    NO_TIMESTAMPS,
    SOT,
    TIMESTAMP_BEGIN,
    TRANSCRIBE,
    WhisperTokenizer,
    is_timestamp_token,
    load_whisper_vocab,
)
from whisper_jax.utils.weights import load_pretrained_weights

# Hop length in samples (for seek calculations)
HOP_LENGTH = 160

# Fixed token buffer size for alignment (prompt=4 + max_tokens=200 + EOT=1 = 205)
ALIGNMENT_TOKEN_BUFFER = 205

# Model configurations: name -> (huggingface_id, create_fn)
_MODEL_CONFIGS = {
    "tiny": ("openai/whisper-tiny", create_whisper_tiny),
    "tiny.en": ("openai/whisper-tiny.en", create_whisper_tiny),
    "base": ("openai/whisper-base", create_whisper_base),
    "base.en": ("openai/whisper-base.en", create_whisper_base),
    "small": ("openai/whisper-small", create_whisper_small),
    "small.en": ("openai/whisper-small.en", create_whisper_small),
    "medium": ("openai/whisper-medium", create_whisper_medium),
    "medium.en": ("openai/whisper-medium.en", create_whisper_medium),
    "large": ("openai/whisper-large-v3", create_whisper_large),
    "large-v1": ("openai/whisper-large-v1", create_whisper_large),
    "large-v2": ("openai/whisper-large-v2", create_whisper_large),
    "large-v3": ("openai/whisper-large-v3", create_whisper_large),
}


@dataclass
class TranscriptionResult:
    """Result of a transcription."""

    text: str
    words: list | None  # list[WordTiming] when alignment deps installed
    language: str
    duration: float


class Whisper:
    """High-level Whisper transcription interface.

    Example:
        whisper = Whisper.load("tiny")
        result = whisper.transcribe("audio.mp3")
        print(result.text)

        # With word timestamps
        result = whisper.transcribe("audio.mp3", word_timestamps=True)
        for word in result.words:
            print(f"[{word.start:.2f}s] {word.word}")
    """

    def __init__(
        self,
        model: WhisperModel,
        tokenizer: WhisperTokenizer,
        model_name: str,
        max_tokens: int = 200,
    ):
        """Initialize Whisper with a loaded model.

        Use Whisper.load() for easier initialization.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_tokens = max_tokens

        self._transcribe_fn = create_transcribe_fn(
            model, max_tokens=max_tokens, use_timestamps=False
        )
        self._transcribe_ts_fn = create_transcribe_fn(
            model, max_tokens=max_tokens, use_timestamps=True
        )
        self._alignment_fn = create_alignment_fn(model)

    @classmethod
    def load(
        cls,
        model_name: str = "tiny",
        max_tokens: int = 200,
    ) -> Whisper:
        """Load a Whisper model by name.

        Args:
            model_name: Model size ("tiny", "base", "small", "medium", "large-v3")
            max_tokens: Maximum tokens to generate per chunk

        Returns:
            Whisper instance ready for transcription

        Example:
            whisper = Whisper.load("small")
            result = whisper.transcribe("audio.mp3")
        """
        if model_name not in _MODEL_CONFIGS:
            available = ", ".join(sorted(_MODEL_CONFIGS.keys()))
            raise ValueError(f"Unknown model: {model_name}. Available: {available}")

        hf_id, create_fn = _MODEL_CONFIGS[model_name]

        model = create_fn()
        load_pretrained_weights(model, hf_id)
        tokenizer = load_whisper_vocab(hf_id)

        return cls(model, tokenizer, model_name, max_tokens)

    def warmup(self, word_timestamps: bool = False) -> None:
        """Trigger JIT compilation with dummy input.

        This is called automatically on first transcription, but can be
        called explicitly to front-load the compilation time.

        Args:
            word_timestamps: If True, also warmup alignment function
                for word-level timestamps. This adds ~10-15s but makes
                the first word_timestamps=True call much faster.
        """
        self._warmup_transcription()
        self._warmup_transcription_ts()
        if word_timestamps:
            self._warmup_alignment()

    def _warmup_transcription(self) -> None:
        """Warmup no-timestamp transcription JIT (~15s for small model)."""
        dummy_audio = jnp.zeros(N_SAMPLES, dtype=jnp.float32)
        lang_token = jnp.array(LANG_TOKENS["en"], dtype=jnp.int32)
        tokens, _ = self._transcribe_fn(dummy_audio, lang_token)

        # Warmup JAX->Python int conversion (has first-call overhead)
        _ = [int(t) for t in tokens[:5]]

    def _warmup_transcription_ts(self) -> None:
        """Warmup timestamp-enabled transcription JIT."""
        dummy_audio = jnp.zeros(N_SAMPLES, dtype=jnp.float32)
        lang_token = jnp.array(LANG_TOKENS["en"], dtype=jnp.int32)
        tokens, _ = self._transcribe_ts_fn(dummy_audio, lang_token)

        # Warmup JAX->Python int conversion
        _ = [int(t) for t in tokens[:5]]

    def _warmup_alignment(self) -> None:
        """Warmup alignment JIT + Numba DTW (~15s for small model)."""
        # Import on demand - raises clear error if deps missing
        from whisper_jax.alignment import warmup_dtw

        dummy_audio = jnp.zeros(N_SAMPLES, dtype=jnp.float32)
        alignment_mask = get_alignment_mask(self.model_name)

        # Use fixed buffer size to match runtime shape
        dummy_tokens = jnp.zeros((1, ALIGNMENT_TOKEN_BUFFER), dtype=jnp.int32)
        dummy_tokens = dummy_tokens.at[0, :5].set(
            jnp.array([SOT, LANG_TOKENS["en"], TRANSCRIBE, NO_TIMESTAMPS, EOT])
        )
        probs, _attn, _ = self._alignment_fn(dummy_audio, dummy_tokens, alignment_mask)
        probs.block_until_ready()

        # Warmup Numba DTW
        warmup_dtw()

    def transcribe(
        self,
        audio: str | Path | np.ndarray,
        language: str = "en",
        word_timestamps: bool = False,
        _profile: bool = False,
    ) -> TranscriptionResult:
        """Transcribe audio to text using seek-based sliding window.

        This implementation uses timestamp tokens to determine chunk boundaries,
        avoiding word splitting at fixed intervals. The approach mirrors the
        official OpenAI Whisper implementation.

        Args:
            audio: Path to audio file or numpy array (float32, 16kHz)
            language: Language code (e.g., "en", "fr", "de")
            word_timestamps: Whether to compute word-level timestamps (refined via VAD)
            _profile: If True, print timing breakdown

        Returns:
            TranscriptionResult with text, optional words, language, and duration
        """
        timings: dict[str, float] = {}

        def _time(name: str, start: float) -> None:
            if _profile:
                timings[name] = timings.get(name, 0.0) + time.perf_counter() - start

        # Load audio if path
        t0 = time.perf_counter()
        if isinstance(audio, str | Path):
            audio = load_audio(audio)
        else:
            audio = normalize_audio(audio)
        _time("load_audio", t0)

        audio_duration = len(audio) / SAMPLE_RATE
        content_frames = len(audio) // HOP_LENGTH  # Total frames in audio

        # Get language token
        if language not in LANG_TOKENS:
            available = ", ".join(sorted(LANG_TOKENS.keys()))
            raise ValueError(f"Unknown language: {language}. Available: {available}")
        lang_token = jnp.array(LANG_TOKENS[language], dtype=jnp.int32)

        # Seek-based sliding window processing
        all_text_tokens: list[int] = []
        all_words: list = []
        seek = 0  # Current position in frames

        while seek < content_frames:
            # Calculate chunk bounds
            time_offset = seek * HOP_LENGTH / SAMPLE_RATE
            start_sample = seek * HOP_LENGTH
            segment_frames = min(N_SAMPLES // HOP_LENGTH, content_frames - seek)

            # Extract chunk (always N_SAMPLES for model input, pad if needed)
            end_sample = min(start_sample + N_SAMPLES, len(audio))
            chunk = audio[start_sample:end_sample]

            t0 = time.perf_counter()
            if len(chunk) < N_SAMPLES:
                chunk = np.pad(chunk, (0, N_SAMPLES - len(chunk)))
            _time("pad_chunk", t0)

            # Transcribe with timestamps for proper seek handling
            t0 = time.perf_counter()
            tokens, num_gen = self._transcribe_ts_fn(jnp.array(chunk), lang_token)
            tokens.block_until_ready()
            _time("transcribe_fn", t0)

            t0 = time.perf_counter()
            generated_tokens = [int(t) for t in tokens[4 : 4 + int(num_gen)]]
            _time("extract_tokens", t0)

            # Parse segments from timestamp tokens and extract text tokens
            text_tokens, last_timestamp_pos = self._parse_timestamp_tokens(
                generated_tokens, segment_frames
            )

            if text_tokens:
                all_text_tokens.extend(text_tokens)

                # Get word timestamps if requested
                if word_timestamps:
                    from whisper_jax.alignment import get_word_timestamps

                    t0 = time.perf_counter()
                    words = get_word_timestamps(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        audio=chunk,
                        text_tokens=text_tokens,
                        model_name=self.model_name,
                        language=language,
                        _alignment_fn=self._alignment_fn,
                    )
                    _time("word_timestamps", t0)

                    # Apply time offset and collect words
                    for w in words:
                        w.start += time_offset
                        w.end += time_offset
                        all_words.append(w)

            # Determine next seek position
            t0 = time.perf_counter()
            if word_timestamps and all_words:
                # Use word boundary for precise seeking
                last_word_end = all_words[-1].end
                if last_word_end > time_offset:
                    seek = round(last_word_end * FRAMES_PER_SECOND)
                else:
                    seek += segment_frames
            elif last_timestamp_pos is not None:
                # Use timestamp token for seeking (like official Whisper)
                # last_timestamp_pos is in 20ms units, convert to frames
                seek += last_timestamp_pos * INPUT_STRIDE
            else:
                # Fallback: advance by segment size
                seek += segment_frames
            _time("seek_calc", t0)

            # Safety check: ensure we always make progress
            if seek <= (start_sample // HOP_LENGTH):
                seek = (start_sample // HOP_LENGTH) + segment_frames

        # Decode full text
        t0 = time.perf_counter()
        text = self.tokenizer.decode(all_text_tokens)
        _time("decode_text", t0)

        # Refine word timestamps using VAD
        if word_timestamps and all_words:
            from whisper_jax.alignment import refine_word_timestamps

            t0 = time.perf_counter()
            all_words = refine_word_timestamps(audio, all_words)
            _time("refine_timestamps", t0)

        if _profile:
            print("\n=== Profiling Breakdown ===")
            total = sum(timings.values())
            for name, elapsed in sorted(timings.items(), key=lambda x: -x[1]):
                pct = 100 * elapsed / total if total > 0 else 0
                print(f"  {name:20s}: {elapsed:7.3f}s ({pct:5.1f}%)")
            print(f"  {'TOTAL':20s}: {total:7.3f}s")
            print()

        return TranscriptionResult(
            text=text,
            words=all_words if word_timestamps else None,
            language=language,
            duration=audio_duration,
        )

    def _parse_timestamp_tokens(
        self, tokens: list[int], segment_frames: int
    ) -> tuple[list[int], int | None]:
        """Parse generated tokens to extract text and find seek position.

        Handles Whisper's timestamp token format where segments are bounded
        by consecutive timestamp tokens (e.g., <|0.00|>text<|2.50|><|2.50|>text<|5.00|>).

        Args:
            tokens: Generated tokens (may include timestamps and text)
            segment_frames: Number of frames in this segment (for fallback)

        Returns:
            (text_tokens, last_timestamp_pos):
                - text_tokens: List of text tokens (excluding timestamps and EOT)
                - last_timestamp_pos: Position of last timestamp in 20ms units,
                  or None if no valid timestamp found
        """
        text_tokens: list[int] = []
        last_timestamp_pos: int | None = None

        # Find consecutive timestamp pairs and extract text between them
        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token == EOT:
                break

            if is_timestamp_token(token):
                # Record this timestamp position
                last_timestamp_pos = token - TIMESTAMP_BEGIN

                # Check for consecutive timestamps (segment boundary)
                if i + 1 < len(tokens) and is_timestamp_token(tokens[i + 1]):
                    # Two consecutive timestamps mark segment boundary
                    # The second one is the start of the next segment
                    i += 1
                    continue
            elif token < EOT:
                # Regular text token
                text_tokens.append(token)

            i += 1

        return text_tokens, last_timestamp_pos

    def _extract_text_tokens(self, tokens: jnp.ndarray, num_generated: int) -> list[int]:
        """Extract text tokens from full token buffer."""
        # Prompt is at indices 0-3, generated tokens start at index 4
        return [int(t) for t in tokens[4 : 4 + num_generated] if t != EOT and t < EOT]

    @property
    def available_languages(self) -> list[str]:
        """List of supported language codes."""
        return list(LANG_TOKENS.keys())

    @staticmethod
    def available_models() -> list[str]:
        """List of available model names."""
        return list(_MODEL_CONFIGS.keys())

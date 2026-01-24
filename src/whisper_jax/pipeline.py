"""High-level Whisper transcription API."""

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
from whisper_jax.utils.dtw import (
    ALIGNMENT_TOKEN_BUFFER,
    WordTiming,
    get_word_timestamps,
    warmup_dtw,
)
from whisper_jax.utils.tokenizer import (
    EOT,
    LANG_TOKENS,
    NO_TIMESTAMPS,
    SOT,
    TRANSCRIBE,
    WhisperTokenizer,
    load_whisper_vocab,
)
from whisper_jax.utils.weights import load_pretrained_weights

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
    words: list[WordTiming] | None
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

        # Create JIT-compiled functions
        self._transcribe_fn = create_transcribe_fn(model, max_tokens=max_tokens)
        self._alignment_fn = create_alignment_fn(model)

        # Warmup tracking
        self._transcription_ready = False
        self._alignment_ready = False

    @classmethod
    def load(
        cls,
        model_name: str = "tiny",
        max_tokens: int = 200,
    ) -> "Whisper":
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
        if word_timestamps:
            self._warmup_alignment()

    def _warmup_transcription(self) -> None:
        """Warmup transcription JIT (~15s for small model)."""
        if self._transcription_ready:
            return

        dummy_audio = jnp.zeros(N_SAMPLES, dtype=jnp.float32)
        lang_token = jnp.array(LANG_TOKENS["en"], dtype=jnp.int32)
        tokens, _ = self._transcribe_fn(dummy_audio, lang_token)

        # Warmup JAX->Python int conversion (has first-call overhead)
        _ = [int(t) for t in tokens[:5]]

        self._transcription_ready = True

    def _warmup_alignment(self) -> None:
        """Warmup alignment JIT + Numba DTW (~15s for small model)."""
        if self._alignment_ready:
            return

        # Alignment needs transcription warmed up first
        self._warmup_transcription()

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

        self._alignment_ready = True

    def transcribe(
        self,
        audio: str | Path | np.ndarray,
        language: str = "en",
        word_timestamps: bool = False,
        _profile: bool = False,
    ) -> TranscriptionResult:
        """Transcribe audio to text.

        Args:
            audio: Path to audio file or numpy array (float32, 16kHz)
            language: Language code (e.g., "en", "fr", "de")
            word_timestamps: Whether to compute word-level timestamps
            _profile: If True, print timing breakdown

        Returns:
            TranscriptionResult with text, optional words, language, and duration
        """
        timings: dict[str, float] = {}

        def _time(name: str, start: float) -> None:
            if _profile:
                timings[name] = timings.get(name, 0.0) + time.perf_counter() - start

        # Lazy warmup: transcription on first call, alignment on first word_timestamps
        self._warmup_transcription()
        if word_timestamps:
            self._warmup_alignment()

        # Load audio if path
        t0 = time.perf_counter()
        if isinstance(audio, (str, Path)):
            audio = load_audio(audio)
        else:
            audio = normalize_audio(audio)
        _time("load_audio", t0)

        duration = len(audio) / SAMPLE_RATE

        # Get language token
        if language not in LANG_TOKENS:
            available = ", ".join(sorted(LANG_TOKENS.keys()))
            raise ValueError(f"Unknown language: {language}. Available: {available}")
        lang_token = jnp.array(LANG_TOKENS[language], dtype=jnp.int32)

        # Process in chunks
        chunk_size = N_SAMPLES
        all_text_tokens: list[int] = []
        all_words: list[WordTiming] = []

        num_chunks = (len(audio) + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            start_sample = chunk_idx * chunk_size
            end_sample = min(start_sample + chunk_size, len(audio))
            chunk = audio[start_sample:end_sample]
            time_offset = start_sample / SAMPLE_RATE

            # Pad chunk to 30 seconds
            t0 = time.perf_counter()
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            _time("pad_chunk", t0)

            # Transcribe
            t0 = time.perf_counter()
            tokens, num_gen = self._transcribe_fn(jnp.array(chunk), lang_token)
            tokens.block_until_ready()  # Force sync for accurate timing
            _time("transcribe_fn", t0)

            t0 = time.perf_counter()
            text_tokens = self._extract_text_tokens(tokens, int(num_gen))
            _time("extract_tokens", t0)

            if not text_tokens:
                continue

            all_text_tokens.extend(text_tokens)

            # Get word timestamps if requested
            if word_timestamps:
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

                # Apply time offset for multi-chunk processing
                for w in words:
                    w.start += time_offset
                    w.end += time_offset
                    all_words.append(w)

        # Decode full text
        t0 = time.perf_counter()
        text = self.tokenizer.decode(all_text_tokens)
        _time("decode_text", t0)

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
            duration=duration,
        )

    def _extract_text_tokens(self, tokens: jnp.ndarray, num_generated: int) -> list[int]:
        """Extract text tokens from full token buffer."""
        # Prompt is at indices 0-3, generated tokens start at index 4
        return [int(t) for t in tokens[4 : 4 + num_generated] if t != EOT and t < EOT]

    @property
    def _warmed_up(self) -> bool:
        """Backward-compatible property for transcription readiness."""
        return self._transcription_ready

    @property
    def available_languages(self) -> list[str]:
        """List of supported language codes."""
        return list(LANG_TOKENS.keys())

    @staticmethod
    def available_models() -> list[str]:
        """List of available model names."""
        return list(_MODEL_CONFIGS.keys())

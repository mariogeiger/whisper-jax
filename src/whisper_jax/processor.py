"""Pure JAX/NumPy audio processor for Whisper - no transformers dependency."""

import json
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Whisper audio constants
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
CHUNK_LENGTH = 30  # seconds
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000


def hz_to_mel(freq: float, mel_scale: str = "slaney") -> float:
    """Convert frequency in Hz to mel scale."""
    if mel_scale == "slaney":
        # Slaney-style mel scale
        min_log_hz = 1000.0
        min_log_mel = 15.0
        logstep = 27.0 / np.log(6.4)
        if freq >= min_log_hz:
            return min_log_mel + np.log(freq / min_log_hz) * logstep
        return 3.0 * freq / 200.0
    # HTK-style
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def mel_to_hz(mel: float, mel_scale: str = "slaney") -> float:
    """Convert mel scale to frequency in Hz."""
    if mel_scale == "slaney":
        min_log_hz = 1000.0
        min_log_mel = 15.0
        logstep = np.log(6.4) / 27.0
        if mel >= min_log_mel:
            return min_log_hz * np.exp(logstep * (mel - min_log_mel))
        return 200.0 * mel / 3.0
    # HTK-style
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filter_bank(
    num_frequency_bins: int = 201,
    num_mel_filters: int = 80,
    min_frequency: float = 0.0,
    max_frequency: float = 8000.0,
    sampling_rate: int = 16000,
) -> np.ndarray:
    """
    Create mel filterbank matrix (Slaney-style, matching Whisper).

    Returns:
        np.ndarray: (num_frequency_bins, num_mel_filters) filterbank matrix
    """
    # Compute mel frequencies
    min_mel = hz_to_mel(min_frequency)
    max_mel = hz_to_mel(max_frequency)

    # Create mel points evenly spaced in mel scale
    mel_points = np.linspace(min_mel, max_mel, num_mel_filters + 2)
    freq_points = np.array([mel_to_hz(m) for m in mel_points])

    # Convert to FFT bin indices
    fft_freqs = np.linspace(0, sampling_rate / 2, num_frequency_bins)

    # Create filterbank
    filterbank = np.zeros((num_frequency_bins, num_mel_filters))

    for i in range(num_mel_filters):
        left = freq_points[i]
        center = freq_points[i + 1]
        right = freq_points[i + 2]

        # Rising edge
        for j, freq in enumerate(fft_freqs):
            if left <= freq < center:
                filterbank[j, i] = (freq - left) / (center - left)
            elif center <= freq <= right:
                filterbank[j, i] = (right - freq) / (right - center)

        # Slaney-style normalization
        enorm = 2.0 / (freq_points[i + 2] - freq_points[i])
        filterbank[:, i] *= enorm

    return filterbank.astype(np.float32)


# Pre-compute mel filterbank (constant)
MEL_FILTERS = mel_filter_bank(
    num_frequency_bins=N_FFT // 2 + 1,
    num_mel_filters=N_MELS,
    min_frequency=0.0,
    max_frequency=8000.0,
    sampling_rate=SAMPLE_RATE,
)


@partial(jax.jit, static_argnames=["n_fft", "hop_length"])
def stft(audio: jax.Array, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH) -> jax.Array:
    """
    Compute Short-Time Fourier Transform using JAX.

    Args:
        audio: (n_samples,) audio waveform
        n_fft: FFT window size
        hop_length: hop length between frames

    Returns:
        (n_freqs, n_frames) complex STFT output
    """
    # Create Hann window
    window = 0.5 - 0.5 * jnp.cos(2 * jnp.pi * jnp.arange(n_fft) / n_fft)

    # Pad audio for centered STFT
    pad_amount = n_fft // 2
    audio_padded = jnp.pad(audio, (pad_amount, pad_amount), mode="reflect")

    # Number of frames
    n_frames = (len(audio_padded) - n_fft) // hop_length + 1

    # Create frame indices
    frame_starts = jnp.arange(n_frames) * hop_length
    frame_indices = frame_starts[:, None] + jnp.arange(n_fft)

    # Extract frames and apply window
    frames = audio_padded[frame_indices] * window

    # Compute FFT
    stft_result = jnp.fft.rfft(frames, n=n_fft, axis=1)

    return stft_result.T  # (n_freqs, n_frames)


@partial(jax.jit, static_argnames=["n_samples"])
def pad_or_trim(audio: jax.Array, n_samples: int = N_SAMPLES) -> jax.Array:
    """Pad or trim audio to exact length."""
    if audio.shape[0] > n_samples:
        return audio[:n_samples]
    elif audio.shape[0] < n_samples:
        return jnp.pad(audio, (0, n_samples - audio.shape[0]))
    return audio


def log_mel_spectrogram(
    audio: np.ndarray | jax.Array,
    n_samples: int = N_SAMPLES,
) -> jax.Array:
    """
    Compute log-mel spectrogram for Whisper model input.

    Args:
        audio: Audio waveform at 16kHz, values in [-1, 1]
        n_samples: Target number of samples (default: 30 seconds)

    Returns:
        (1, 80, 3000) log-mel spectrogram ready for model input
    """
    audio = jnp.asarray(audio, dtype=jnp.float32)

    # Pad or trim to exact length
    audio = pad_or_trim(audio, n_samples)

    # Compute STFT
    stft_result = stft(audio)

    # Compute power spectrogram
    magnitudes = jnp.abs(stft_result) ** 2

    # Apply mel filterbank
    mel_filters = jnp.asarray(MEL_FILTERS)
    mel_spec = mel_filters.T @ magnitudes

    # Log scale with floor
    log_spec = jnp.log10(jnp.maximum(mel_spec, 1e-10))

    # Whisper-style normalization
    log_spec = jnp.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    # Remove last frame to match Whisper (3000 frames for 30s)
    log_spec = log_spec[:, :-1]

    # Add batch dimension
    return log_spec[None, :, :]


def log_mel_spectrogram_np(audio: np.ndarray) -> np.ndarray:
    """NumPy version for compatibility. Returns numpy array."""
    return np.array(log_mel_spectrogram(audio))


# ============================================================================
# Token decoding
# ============================================================================

# Whisper special tokens
SPECIAL_TOKENS = {
    50256: "<|endoftext|>",
    50257: "<|endoftext|>",
    50258: "<|startoftranscript|>",
    50259: "<|en|>",
    50359: "<|transcribe|>",
    50363: "<|notimestamps|>",
    50364: "<|0.00|>",
}

# Language token IDs for Whisper models
LANG_TOKENS = {
    "en": 50259, "fr": 50265, "de": 50261, "es": 50262,
    "it": 50274, "pt": 50267, "nl": 50271, "pl": 50269,
    "ru": 50263, "zh": 50260, "ja": 50266, "ko": 50264,
}


class WhisperTokenizer:
    """Simple tokenizer for decoding Whisper output tokens."""

    def __init__(self, vocab: dict[int, str]):
        self.vocab = vocab
        self.special_token_ids = set(range(50257, 50365))  # Special tokens range

    def decode(self, token_ids: list[int] | np.ndarray, skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()

        tokens = []
        for tid in token_ids:
            if skip_special_tokens and tid in self.special_token_ids:
                continue
            if tid == 50257:  # End of text
                break
            if tid in self.vocab:
                tokens.append(self.vocab[tid])

        # Join and clean up
        text = "".join(tokens)
        # Whisper uses GPT-2 style byte encoding - decode it
        text = bytearray([self._byte_decoder.get(c, ord(c)) for c in text]).decode(
            "utf-8", errors="replace"
        )
        return text.strip()

    @property
    def _byte_decoder(self) -> dict[str, int]:
        """GPT-2 style byte decoder."""
        if not hasattr(self, "_cached_byte_decoder"):
            # Build byte decoder (reverse of byte encoder)
            bs = (
                list(range(ord("!"), ord("~") + 1))
                + list(range(ord("¡"), ord("¬") + 1))
                + list(range(ord("®"), ord("ÿ") + 1))
            )
            cs = bs[:]
            n = 0
            for b in range(256):
                if b not in bs:
                    bs.append(b)
                    cs.append(256 + n)
                    n += 1
            self._cached_byte_decoder = {chr(c): b for b, c in zip(bs, cs, strict=True)}
        return self._cached_byte_decoder


def load_whisper_vocab(model_name: str = "openai/whisper-tiny") -> WhisperTokenizer:
    """
    Load Whisper vocabulary from HuggingFace Hub.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        WhisperTokenizer instance
    """
    try:
        from huggingface_hub import hf_hub_download

        vocab_path = hf_hub_download(
            repo_id=model_name,
            filename="vocab.json",
        )

        with open(vocab_path, encoding="utf-8") as f:
            vocab_dict = json.load(f)

        # Invert: vocab.json is {token: id}, we need {id: token}
        vocab = {int(v): k for k, v in vocab_dict.items()}

        return WhisperTokenizer(vocab)

    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to download the vocabulary. "
            "Install it with: pip install huggingface_hub"
        ) from e


def load_whisper_vocab_from_file(vocab_path: str | Path) -> WhisperTokenizer:
    """Load vocabulary from a local file."""
    with open(vocab_path, encoding="utf-8") as f:
        vocab_dict = json.load(f)
    vocab = {int(v): k for k, v in vocab_dict.items()}
    return WhisperTokenizer(vocab)


def create_transcribe_fn(model, max_tokens: int = 100):
    """Create a JIT-compiled transcription function.

    This function creates a highly optimized transcription function that:
    - Uses jax.lax.while_loop for early stopping at EOT
    - Captures model components in closure for optimal JIT performance
    - Uses fixed-size token buffer for consistent shapes

    Args:
        model: WhisperModel instance
        max_tokens: Maximum tokens to generate (compile-time constant)

    Returns:
        JIT-compiled function with signature:
            (audio: jax.Array, lang_token: jax.Array) -> (tokens, num_generated)

    Example:
        transcribe = create_transcribe_fn(model)
        tokens, n = transcribe(audio, jnp.array(50259))  # English
        text_tokens = [int(t) for t in tokens[4:4+int(n)] if t < 50257]
        text = tokenizer.decode(text_tokens)
    """
    encoder = model.encoder
    decoder = model.decoder
    lm_head = model.lm_head
    EOT = 50257

    @jax.jit
    def transcribe(audio: jax.Array, lang_token: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Transcribe audio to tokens.

        Args:
            audio: Audio samples, shape (samples,), padded to 30s
            lang_token: Language token ID (e.g., 50259 for English)

        Returns:
            (tokens, num_generated): Full token buffer and count of generated tokens
        """
        mel = log_mel_spectrogram(audio)
        enc_out = encoder(mel, deterministic=True)

        prompt = jnp.array([50258, lang_token, 50359, 50363], dtype=jnp.int32)
        tokens = jnp.zeros(4 + max_tokens, dtype=jnp.int32).at[:4].set(prompt)

        def cond(state):
            tokens, enc, idx = state
            not_at_max = idx < max_tokens
            last_token = tokens[3 + idx]
            not_eot = last_token != EOT
            return not_at_max & not_eot

        def body(state):
            tokens, enc, idx = state
            dec_out, _ = decoder(tokens[None], enc, deterministic=True)
            logits = lm_head(dec_out)
            next_token = jnp.argmax(logits[0, 3 + idx])
            tokens = tokens.at[4 + idx].set(next_token)
            return (tokens, enc, idx + 1)

        tokens, _, num_gen = jax.lax.while_loop(
            cond, body, (tokens, enc_out, jnp.array(0))
        )
        return tokens, num_gen

    return transcribe

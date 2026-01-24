"""Pure JAX audio processing for Whisper."""

from functools import partial

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


def _hz_to_mel(freq: float, mel_scale: str = "slaney") -> float:
    """Convert frequency in Hz to mel scale."""
    if mel_scale == "slaney":
        min_log_hz = 1000.0
        min_log_mel = 15.0
        logstep = 27.0 / np.log(6.4)
        if freq >= min_log_hz:
            return min_log_mel + np.log(freq / min_log_hz) * logstep
        return 3.0 * freq / 200.0
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def _mel_to_hz(mel: float, mel_scale: str = "slaney") -> float:
    """Convert mel scale to frequency in Hz."""
    if mel_scale == "slaney":
        min_log_hz = 1000.0
        min_log_mel = 15.0
        logstep = np.log(6.4) / 27.0
        if mel >= min_log_mel:
            return min_log_hz * np.exp(logstep * (mel - min_log_mel))
        return 200.0 * mel / 3.0
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filter_bank(
    num_frequency_bins: int = 201,
    num_mel_filters: int = 80,
    min_frequency: float = 0.0,
    max_frequency: float = 8000.0,
    sampling_rate: int = 16000,
) -> np.ndarray:
    """Create mel filterbank matrix (Slaney-style, matching Whisper)."""
    min_mel = _hz_to_mel(min_frequency)
    max_mel = _hz_to_mel(max_frequency)

    mel_points = np.linspace(min_mel, max_mel, num_mel_filters + 2)
    freq_points = np.array([_mel_to_hz(m) for m in mel_points])
    fft_freqs = np.linspace(0, sampling_rate / 2, num_frequency_bins)

    filterbank = np.zeros((num_frequency_bins, num_mel_filters))

    for i in range(num_mel_filters):
        left = freq_points[i]
        center = freq_points[i + 1]
        right = freq_points[i + 2]

        for j, freq in enumerate(fft_freqs):
            if left <= freq < center:
                filterbank[j, i] = (freq - left) / (center - left)
            elif center <= freq <= right:
                filterbank[j, i] = (right - freq) / (right - center)

        enorm = 2.0 / (freq_points[i + 2] - freq_points[i])
        filterbank[:, i] *= enorm

    return filterbank.astype(np.float32)


# Pre-compute mel filterbank (constant)
MEL_FILTERS = _mel_filter_bank(
    num_frequency_bins=N_FFT // 2 + 1,
    num_mel_filters=N_MELS,
    min_frequency=0.0,
    max_frequency=8000.0,
    sampling_rate=SAMPLE_RATE,
)


@partial(jax.jit, static_argnames=["n_fft", "hop_length"])
def stft(audio: jax.Array, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH) -> jax.Array:
    """Compute Short-Time Fourier Transform using JAX.

    Args:
        audio: (n_samples,) audio waveform
        n_fft: FFT window size
        hop_length: hop length between frames

    Returns:
        (n_freqs, n_frames) complex STFT output
    """
    window = 0.5 - 0.5 * jnp.cos(2 * jnp.pi * jnp.arange(n_fft) / n_fft)

    pad_amount = n_fft // 2
    audio_padded = jnp.pad(audio, (pad_amount, pad_amount), mode="reflect")

    n_frames = (len(audio_padded) - n_fft) // hop_length + 1

    frame_starts = jnp.arange(n_frames) * hop_length
    frame_indices = frame_starts[:, None] + jnp.arange(n_fft)

    frames = audio_padded[frame_indices] * window

    stft_result = jnp.fft.rfft(frames, n=n_fft, axis=1)

    return stft_result.T


@partial(jax.jit, static_argnames=["n_samples"])
def pad_or_trim(audio: jax.Array, n_samples: int = N_SAMPLES) -> jax.Array:
    """Pad or trim audio to exact length."""
    if audio.shape[0] > n_samples:
        return audio[:n_samples]
    elif audio.shape[0] < n_samples:
        return jnp.pad(audio, (0, n_samples - audio.shape[0]))
    return audio


def log_mel_spectrogram(audio: jax.Array, n_samples: int = N_SAMPLES) -> jax.Array:
    """Compute log-mel spectrogram for Whisper model input.

    Args:
        audio: Audio waveform at 16kHz, values in [-1, 1]
        n_samples: Target number of samples (default: 30 seconds)

    Returns:
        (1, 80, 3000) log-mel spectrogram ready for model input
    """
    audio = pad_or_trim(audio, n_samples)

    stft_result = stft(audio)

    magnitudes = jnp.abs(stft_result) ** 2

    mel_filters = jnp.asarray(MEL_FILTERS)
    mel_spec = mel_filters.T @ magnitudes

    log_spec = jnp.log10(jnp.maximum(mel_spec, 1e-10))

    log_spec = jnp.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    log_spec = log_spec[:, :-1]

    return log_spec[None, :, :]

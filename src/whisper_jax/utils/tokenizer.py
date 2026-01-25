"""Whisper tokenizer utilities."""

import json
from pathlib import Path

import numpy as np

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
    "en": 50259,
    "fr": 50265,
    "de": 50261,
    "es": 50262,
    "it": 50274,
    "pt": 50267,
    "nl": 50271,
    "pl": 50269,
    "ru": 50263,
    "zh": 50260,
    "ja": 50266,
    "ko": 50264,
}

# Token constants
SOT = 50258  # Start of transcript
EOT = 50257  # End of transcript
TRANSCRIBE = 50359  # Transcribe task
NO_TIMESTAMPS = 50363  # No timestamps mode

# Timestamp token constants
TIMESTAMP_BEGIN = 50364  # First timestamp token <|0.00|>
TIMESTAMP_END = 51865  # Last timestamp token (30 seconds)
TIME_PRECISION = 0.02  # 20ms per timestamp token
FRAMES_PER_SECOND = 100  # 16000 / 160 (sample_rate / hop_length)
INPUT_STRIDE = 2  # mel frames per output token


def is_timestamp_token(token_id: int) -> bool:
    """Check if a token is a timestamp token."""
    return TIMESTAMP_BEGIN <= token_id <= TIMESTAMP_END


def timestamp_to_seconds(token_id: int) -> float:
    """Convert a timestamp token to seconds."""
    if not is_timestamp_token(token_id):
        raise ValueError(f"Token {token_id} is not a timestamp token")
    return (token_id - TIMESTAMP_BEGIN) * TIME_PRECISION


def seconds_to_timestamp_token(seconds: float) -> int:
    """Convert seconds to the nearest timestamp token."""
    token = round(seconds / TIME_PRECISION) + TIMESTAMP_BEGIN
    return max(TIMESTAMP_BEGIN, min(token, TIMESTAMP_END))


def extract_timestamp_positions(tokens: list[int]) -> list[tuple[int, float]]:
    """Extract timestamp tokens and their positions in seconds.

    Returns:
        List of (token_index, time_in_seconds) tuples
    """
    return [(i, timestamp_to_seconds(t)) for i, t in enumerate(tokens) if is_timestamp_token(t)]


def find_last_timestamp(tokens: list[int]) -> float | None:
    """Find the last timestamp in a token sequence (in seconds)."""
    for token in reversed(tokens):
        if is_timestamp_token(token):
            return timestamp_to_seconds(token)
    return None


class WhisperTokenizer:
    """Simple tokenizer for decoding Whisper output tokens."""

    def __init__(self, vocab: dict[int, str]):
        self.vocab = vocab
        self.special_token_ids = set(range(50257, 50365))

    def decode(self, token_ids: list[int] | np.ndarray, skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()

        tokens = []
        for tid in token_ids:
            if skip_special_tokens and tid in self.special_token_ids:
                continue
            if tid == EOT:
                break
            if tid in self.vocab:
                tokens.append(self.vocab[tid])

        text = "".join(tokens)
        text = bytearray([self._byte_decoder.get(c, ord(c)) for c in text]).decode(
            "utf-8", errors="replace"
        )
        return text.strip()

    @property
    def _byte_decoder(self) -> dict[str, int]:
        """GPT-2 style byte decoder."""
        if not hasattr(self, "_cached_byte_decoder"):
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
    """Load Whisper vocabulary from HuggingFace Hub.

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

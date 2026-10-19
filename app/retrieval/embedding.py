from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np

from app.text import tokenize


class Embedder(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @property
    @abstractmethod
    def identity(self) -> str: ...

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self.model = _load_sentence_transformer(model_name, device)
        self._dimension = int(self.model.get_sentence_embedding_dimension())

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def identity(self) -> str:
        return f"sentence-transformers:{self.model_name}"

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.asarray(
            self.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )


class HashingEmbedder(Embedder):
    """Dependency-light deterministic fallback for smoke tests, not production semantics."""

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def identity(self) -> str:
        return f"hashing:v1:{self.dimension}"

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = tokenize(text)
            for token in tokens:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                index = value % self.dimension
                matrix[row, index] += 1.0 if value & 1 else -1.0
            norm = np.linalg.norm(matrix[row])
            if norm:
                matrix[row] /= norm
        return matrix


def build_embedder(provider: str, model_name: str, device: str = "cpu") -> Embedder:
    if provider == "hashing":
        return HashingEmbedder()
    if provider == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name, device)
    raise ValueError(f"Unknown embedding provider: {provider}")


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str, device: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install sentence-transformers to use this embedder") from exc
    return SentenceTransformer(model_name, device=device)

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from app.domain import Chunk, SearchResult
from app.exceptions import IndexNotReadyError
from app.retrieval.embedding import Embedder


class VectorStore(ABC):
    embedder: Embedder

    @property
    @abstractmethod
    def ready(self) -> bool: ...

    @property
    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def all_chunks(self) -> list[Chunk]: ...

    @abstractmethod
    def build(self, chunks: list[Chunk]) -> None: ...

    @abstractmethod
    def search(self, query: str, top_k: int) -> list[SearchResult]: ...

    @abstractmethod
    def save(self) -> None: ...

    @abstractmethod
    def load(self) -> bool: ...


class FaissVectorStore(VectorStore):
    INDEX_FILE = "vectors.faiss"
    METADATA_FILE = "chunks.json"
    MANIFEST_FILE = "manifest.json"

    def __init__(self, embedder: Embedder, directory: str | Path) -> None:
        self.embedder = embedder
        self.directory = Path(directory)
        self.index = None
        self.chunks: list[Chunk] = []

    @staticmethod
    def _faiss():
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("Install faiss-cpu to use the vector store") from exc
        return faiss

    @property
    def ready(self) -> bool:
        return self.index is not None and bool(self.chunks)

    @property
    def count(self) -> int:
        return len(self.chunks)

    def all_chunks(self) -> list[Chunk]:
        return list(self.chunks)

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build an index with no chunks")
        vectors = self.embedder.encode([chunk.text for chunk in chunks])
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.shape != (len(chunks), self.embedder.dimension):
            raise ValueError("Embedder returned an unexpected vector shape")
        faiss = self._faiss()
        self.index = faiss.IndexFlatIP(self.embedder.dimension)
        self.index.add(vectors)
        self.chunks = list(chunks)

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        if not self.ready:
            raise IndexNotReadyError("No index is loaded. Upload or index documents first.")
        query_vector = np.ascontiguousarray(self.embedder.encode([query]), dtype=np.float32)
        count = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query_vector, count)
        return [
            SearchResult(chunk=self.chunks[int(index)], score=float(score))
            for score, index in zip(scores[0], indices[0], strict=True)
            if index >= 0
        ]

    def save(self) -> None:
        if not self.ready:
            raise IndexNotReadyError("Cannot save an empty index")
        self.directory.mkdir(parents=True, exist_ok=True)
        # FAISS's Windows file I/O cannot reliably open paths containing non-ASCII text.
        index_bytes = self._faiss().serialize_index(self.index).tobytes()
        (self.directory / self.INDEX_FILE).write_bytes(index_bytes)
        (self.directory / self.METADATA_FILE).write_text(
            json.dumps([chunk.to_dict() for chunk in self.chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self.directory / self.MANIFEST_FILE).write_text(
            json.dumps(
                {
                    "dimension": self.embedder.dimension,
                    "embedder": self.embedder.identity,
                    "count": len(self.chunks),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def load(self) -> bool:
        index_path = self.directory / self.INDEX_FILE
        metadata_path = self.directory / self.METADATA_FILE
        manifest_path = self.directory / self.MANIFEST_FILE
        if not all(path.exists() for path in (index_path, metadata_path, manifest_path)):
            return False
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["dimension"] != self.embedder.dimension:
            raise ValueError("Stored index dimension does not match the configured embedder")
        if manifest.get("embedder") != self.embedder.identity:
            raise ValueError("Stored index was built with a different embedding model")
        index_bytes = np.frombuffer(index_path.read_bytes(), dtype=np.uint8)
        self.index = self._faiss().deserialize_index(index_bytes)
        values = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.chunks = [Chunk.from_dict(value) for value in values]
        if self.index.ntotal != len(self.chunks):
            raise ValueError("Corrupted index: vector and metadata counts differ")
        return True

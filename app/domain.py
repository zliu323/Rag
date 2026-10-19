from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class Document:
    text: str
    source: str
    page: int | None = None


@dataclass(slots=True)
class Chunk:
    id: str
    text: str
    source: str
    page: int | None
    chunk_index: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Chunk:
        return cls(**value)


@dataclass(slots=True)
class SearchResult:
    chunk: Chunk
    score: float
    retrieval_score: float | None = None

    def citation(self) -> str:
        location = f"p.{self.chunk.page}" if self.chunk.page else "source"
        return f"{self.chunk.source} ({location})"


from __future__ import annotations

import math
from collections import Counter, defaultdict

from app.domain import Chunk, SearchResult
from app.text import tokenize


class BM25Index:
    """Small in-process BM25 index over the same chunks as vector search."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, chunk in enumerate(chunks):
            counts = Counter(tokenize(chunk.text))
            self.lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                self.postings[term].append((index, frequency))
        self.average_length = sum(self.lengths) / len(chunks) if chunks else 0.0

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        if not self.chunks or top_k <= 0:
            return []
        scores: dict[int, float] = defaultdict(float)
        total = len(self.chunks)
        for term in set(tokenize(query)):
            matches = self.postings.get(term, [])
            if not matches:
                continue
            idf = math.log1p((total - len(matches) + 0.5) / (len(matches) + 0.5))
            for index, frequency in matches:
                normalized = self.k1 * (
                    1 - self.b + self.b * self.lengths[index] / self.average_length
                )
                scores[index] += idf * frequency * (self.k1 + 1) / (frequency + normalized)
        ranked = sorted(scores, key=lambda index: (-scores[index], index))[:top_k]
        return [SearchResult(self.chunks[index], scores[index]) for index in ranked]


def reciprocal_rank_fusion(
    dense: list[SearchResult], lexical: list[SearchResult], top_k: int, k: int = 60
) -> list[SearchResult]:
    scores: dict[str, float] = defaultdict(float)
    chunks: dict[str, Chunk] = {}
    for ranking in (dense, lexical):
        for rank, result in enumerate(ranking, start=1):
            scores[result.chunk.id] += 1.0 / (k + rank)
            chunks[result.chunk.id] = result.chunk
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:top_k]
    return [SearchResult(chunks[chunk_id], scores[chunk_id]) for chunk_id in ordered]

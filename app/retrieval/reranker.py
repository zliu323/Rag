from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache

from app.domain import SearchResult
from app.text import tokenize


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, results: list[SearchResult], top_n: int) -> list[SearchResult]: ...


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model = _load_cross_encoder(model_name, device)

    def rerank(self, query: str, results: list[SearchResult], top_n: int) -> list[SearchResult]:
        if not results:
            return []
        scores = self.model.predict([(query, item.chunk.text) for item in results])
        reranked = [
            SearchResult(chunk=item.chunk, score=float(score), retrieval_score=item.score)
            for item, score in zip(results, scores, strict=True)
        ]
        return sorted(reranked, key=lambda item: item.score, reverse=True)[:top_n]


class LexicalReranker(Reranker):
    """Offline fallback using query token coverage."""

    def rerank(self, query: str, results: list[SearchResult], top_n: int) -> list[SearchResult]:
        query_tokens = set(tokenize(query))
        reranked = []
        for item in results:
            text_tokens = set(tokenize(item.chunk.text))
            lexical = len(query_tokens & text_tokens) / max(len(query_tokens), 1)
            reranked.append(
                SearchResult(chunk=item.chunk, score=lexical, retrieval_score=item.score)
            )
        return sorted(reranked, key=lambda item: item.score, reverse=True)[:top_n]


@lru_cache(maxsize=4)
def _load_cross_encoder(model_name: str, device: str):
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError("Install sentence-transformers to use the reranker") from exc
    return CrossEncoder(model_name, device=device)

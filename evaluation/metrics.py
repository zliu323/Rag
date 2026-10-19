from __future__ import annotations

import re

from app.domain import SearchResult


def _is_relevant(
    result: SearchResult, relevant_sources: set[str], required_text: list[str]
) -> bool:
    source_match = not relevant_sources or result.chunk.source in relevant_sources
    text = result.chunk.text.lower()
    text_match = not required_text or any(value.lower() in text for value in required_text)
    return source_match and text_match


def retrieval_metrics(
    results: list[SearchResult],
    relevant_sources: list[str],
    required_text: list[str] | None = None,
    k: int = 5,
) -> dict[str, float]:
    top = results[:k]
    relevant = set(relevant_sources)
    requirements = required_text or []
    matches = [_is_relevant(item, relevant, requirements) for item in top]
    hit = float(any(matches))
    reciprocal_rank = next((1.0 / rank for rank, match in enumerate(matches, 1) if match), 0.0)
    retrieved_sources = {
        item.chunk.source for item, match in zip(top, matches, strict=True) if match
    }
    recall = len(retrieved_sources & relevant) / len(relevant) if relevant else hit
    return {f"hit@{k}": hit, f"recall@{k}": recall, "mrr": reciprocal_rank}


def answer_keyword_correctness(answer: str, keywords: list[str]) -> dict[str, float]:
    if not keywords:
        return {"answer_correctness": 0.0, "matched_keywords": 0.0}
    normalized = re.sub(r"\s+", " ", answer.lower())
    matched = sum(1 for keyword in keywords if keyword.lower() in normalized)
    return {
        "answer_correctness": matched / len(keywords),
        "matched_keywords": float(matched),
    }


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}

from __future__ import annotations

import re

from app.domain import SearchResult
from app.text import tokenize

_FRAMING = re.compile(
    r"请帮我查一下|能不能告诉我|根据知识库|帮我查一下|我想知道|请告诉我|请问|"
    r"是什么|是多少|有哪些|什么|多少|哪些|如何|是否|能否"
)
_STOPWORDS = set("的了是在有和与为于及或吗呢啊我你您这那几") | {
    "the",
    "a",
    "an",
    "is",
    "are",
    "of",
    "to",
    "in",
    "and",
    "what",
    "how",
    "does",
    "do",
    "can",
    "please",
}


def context_coverage(question: str, contexts: list[SearchResult]) -> float:
    """Conservative lexical relevance signal, not a semantic answerability guarantee."""
    terms = set(tokenize(_FRAMING.sub("", question))) - _STOPWORDS
    if not terms or not contexts:
        return 0.0
    return max(len(terms & set(tokenize(item.chunk.text))) / len(terms) for item in contexts)


def has_relevant_context(
    question: str, contexts: list[SearchResult], minimum_coverage: float = 0.3
) -> bool:
    return bool(contexts) and (
        minimum_coverage == 0 or context_coverage(question, contexts) >= minimum_coverage
    )

from __future__ import annotations

import re
from abc import ABC, abstractmethod

import httpx

from app.domain import SearchResult
from app.exceptions import ProviderError
from app.generation.prompt import SYSTEM_PROMPT, build_prompt
from app.text import tokenize


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, question: str, contexts: list[SearchResult]) -> str: ...

    def generate_with_metadata(
        self, question: str, contexts: list[SearchResult]
    ) -> tuple[str, dict[str, int]]:
        return self.generate(question, contexts), {}


class ExtractiveProvider(LLMProvider):
    """No-key fallback: selects context sentences with the best query-token overlap."""

    def generate(self, question: str, contexts: list[SearchResult]) -> str:
        query_tokens = set(tokenize(question))
        candidates: list[tuple[float, str, int]] = []
        for citation_id, item in enumerate(contexts, start=1):
            sentences = re.split(r"(?<=[。！？.!?])\s*|\n+", item.chunk.text)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 12:
                    continue
                tokens = set(tokenize(sentence))
                overlap = len(query_tokens & tokens) / max(len(query_tokens), 1)
                candidates.append((overlap, sentence, citation_id))
        selected = sorted(candidates, key=lambda value: value[0], reverse=True)[:3]
        selected = [value for value in selected if value[0] > 0]
        if not selected:
            return "现有知识库内容不足以回答该问题。"
        return " ".join(f"{sentence} [{citation_id}]" for _, sentence, citation_id in selected)


class OpenAICompatibleProvider(LLMProvider):
    """Works with OpenAI-compatible cloud APIs, Ollama, vLLM and LM Studio."""

    def __init__(self, base_url: str, model: str, api_key: str | None, timeout: float = 60) -> None:
        self.endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def generate(self, question: str, contexts: list[SearchResult]) -> str:
        return self.generate_with_metadata(question, contexts)[0]

    def generate_with_metadata(
        self, question: str, contexts: list[SearchResult]
    ) -> tuple[str, dict[str, int]]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(question, contexts)},
            ],
        }
        try:
            response = httpx.post(
                self.endpoint, headers=headers, json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            payload = response.json()
            answer = payload["choices"][0]["message"]["content"].strip()
            usage = payload.get("usage") or {}
            details = (
                {
                    "input": int(usage["prompt_tokens"]),
                    "output": int(usage["completion_tokens"]),
                }
                if "prompt_tokens" in usage and "completion_tokens" in usage
                else {}
            )
            return answer, details
        except (httpx.HTTPError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"LLM provider request failed: {exc}") from exc


def build_llm(provider: str, base_url: str, model: str, api_key: str | None) -> LLMProvider:
    if provider == "extractive":
        return ExtractiveProvider()
    if provider == "openai-compatible":
        return OpenAICompatibleProvider(base_url, model, api_key)
    raise ValueError(f"Unknown LLM provider: {provider}")

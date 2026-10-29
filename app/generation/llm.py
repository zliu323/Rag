from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

import httpx

from app.domain import GenerationResult, SearchResult
from app.exceptions import ProviderError
from app.generation.prompt import SYSTEM_PROMPT, build_prompt
from app.text import tokenize

INSUFFICIENT_CONTEXT_ANSWER = "现有知识库内容不足以回答该问题。"
STRUCTURED_OUTPUT_INSTRUCTION = (
    "Return only a JSON object with exactly two fields: "
    '"decision" ("answered" or "refused") and "answer" (a nonempty string). '
    'Use "refused" whenever the supplied context does not establish the requested facts, '
    "even if it discusses a related topic. For an answer, preserve conditions and cite [n]. "
    "Answer in the same language as the question. Do not wrap JSON in Markdown fences."
)


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, question: str, contexts: list[SearchResult]) -> str: ...

    def generate_with_metadata(
        self, question: str, contexts: list[SearchResult]
    ) -> tuple[str, dict[str, int]]:
        return self.generate(question, contexts), {}

    @abstractmethod
    def generate_result(self, question: str, contexts: list[SearchResult]) -> GenerationResult:
        """Return an explicit decision alongside the answer and usage."""
        ...


class ExtractiveProvider(LLMProvider):
    """No-key fallback: selects context sentences with the best query-token overlap."""

    def generate(self, question: str, contexts: list[SearchResult]) -> str:
        return self.generate_result(question, contexts).answer

    def generate_result(self, question: str, contexts: list[SearchResult]) -> GenerationResult:
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
            return GenerationResult(INSUFFICIENT_CONTEXT_ANSWER, "refused", {})
        return GenerationResult(
            " ".join(f"{sentence} [{citation_id}]" for _, sentence, citation_id in selected),
            "answered",
            {},
        )


class OpenAICompatibleProvider(LLMProvider):
    """Works with OpenAI-compatible cloud APIs, Ollama, vLLM and LM Studio."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None,
        timeout: float = 60,
        extra_payload: dict | None = None,
    ) -> None:
        self.endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.extra_payload = extra_payload or {}

    def generate(self, question: str, contexts: list[SearchResult]) -> str:
        return self.generate_with_metadata(question, contexts)[0]

    def generate_with_metadata(
        self, question: str, contexts: list[SearchResult]
    ) -> tuple[str, dict[str, int]]:
        return self._request(question, contexts)

    def generate_result(self, question: str, contexts: list[SearchResult]) -> GenerationResult:
        content, usage = self._request(question, contexts, structured=True)
        try:
            result = json.loads(content)
            if not isinstance(result, dict) or set(result) != {"decision", "answer"}:
                raise ValueError("Invalid result fields")
            decision, answer = result["decision"], result["answer"]
            if decision not in ("answered", "refused"):
                raise ValueError("Invalid decision")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("Empty answer")
        except (ValueError, TypeError) as exc:
            raise ProviderError("LLM returned an invalid structured answer") from exc
        return GenerationResult(
            INSUFFICIENT_CONTEXT_ANSWER if decision == "refused" else answer.strip(),
            decision,
            usage,
        )

    def _request(
        self, question: str, contexts: list[SearchResult], *, structured: bool = False
    ) -> tuple[str, dict[str, int]]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                    + ("\n\n" + STRUCTURED_OUTPUT_INSTRUCTION if structured else ""),
                },
                {"role": "user", "content": build_prompt(question, contexts)},
            ],
        }
        payload.update(self.extra_payload)
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


def build_llm(
    provider: str,
    base_url: str,
    model: str,
    api_key: str | None,
    *,
    deepseek_base_url: str = "https://api.deepseek.com",
    deepseek_model: str = "deepseek-flash",
    deepseek_api_key: str | None = None,
) -> LLMProvider:
    if provider == "extractive":
        return ExtractiveProvider()
    if provider == "openai-compatible":
        return OpenAICompatibleProvider(base_url, model, api_key)
    if provider == "deepseek":
        if not deepseek_api_key or not deepseek_api_key.strip():
            raise ValueError("RAG_DEEPSEEK_API_KEY is required when RAG_LLM_PROVIDER=deepseek")
        return OpenAICompatibleProvider(
            deepseek_base_url,
            deepseek_model,
            deepseek_api_key.strip(),
            timeout=120,
            extra_payload={"thinking": {"type": "disabled"}},
        )
    raise ValueError(f"Unknown LLM provider: {provider}")

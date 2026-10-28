import pytest

from app.config import Settings
from app.dependencies import build_pipeline
from app.generation.llm import OpenAICompatibleProvider


def test_openai_compatible_returns_provider_token_usage(monkeypatch) -> None:
    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {
                "choices": [{"message": {"content": "答案 [1]"}}],
                "usage": {"prompt_tokens": 25, "completion_tokens": 8, "total_tokens": 33},
            }

    monkeypatch.setattr("app.generation.llm.httpx.post", lambda *args, **kwargs: Response())
    provider = OpenAICompatibleProvider("http://example.test/v1", "test-model", None)
    answer, usage = provider.generate_with_metadata("问题", [])
    assert answer == "答案 [1]"
    assert usage == {"input": 25, "output": 8}


def test_deepseek_pipeline_uses_official_endpoint_and_key(monkeypatch, tmp_path) -> None:
    requests = []

    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {
                "choices": [{"message": {"content": "根据文档回答 [1]"}}],
                "usage": {"prompt_tokens": 30, "completion_tokens": 9},
            }

    def fake_post(url, *, headers, json, timeout):
        requests.append((url, headers, json, timeout))
        return Response()

    monkeypatch.setattr("app.generation.llm.httpx.post", fake_post)
    settings = Settings(
        index_dir=tmp_path / "index",
        embedding_provider="hashing",
        reranker_enabled=False,
        llm_provider="deepseek",
        deepseek_api_key="test-secret",
    )
    pipeline = build_pipeline(settings)
    answer, usage = pipeline.llm.generate_with_metadata("P0 事件多久通知负责人？", [])

    assert answer == "根据文档回答 [1]"
    assert usage == {"input": 30, "output": 9}
    assert requests[0][0] == "https://api.deepseek.com/chat/completions"
    assert requests[0][1]["Authorization"] == "Bearer test-secret"
    assert requests[0][2]["model"] == "deepseek-flash"
    assert requests[0][2]["thinking"] == {"type": "disabled"}
    assert "P0 事件多久通知负责人？" in requests[0][2]["messages"][1]["content"]
    assert requests[0][3] == 120


def test_deepseek_requires_api_key(tmp_path) -> None:
    settings = Settings(
        index_dir=tmp_path / "index",
        embedding_provider="hashing",
        reranker_enabled=False,
        llm_provider="deepseek",
        deepseek_api_key="",
    )
    with pytest.raises(ValueError, match="RAG_DEEPSEEK_API_KEY"):
        build_pipeline(settings)

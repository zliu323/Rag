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

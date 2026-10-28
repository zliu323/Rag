from contextlib import contextmanager

from app.config import ROOT, Settings
from app.dependencies import build_pipeline


class Observation:
    trace_id = "a" * 32

    def __init__(self) -> None:
        self.updates: list[dict] = []

    def update(self, **kwargs) -> None:
        self.updates.append(kwargs)


class Client:
    def __init__(self) -> None:
        self.names: list[str] = []
        self.calls: list[tuple[str, dict]] = []
        self.scores: list[dict] = []

    @contextmanager
    def start_as_current_observation(self, *, name, **kwargs):
        self.names.append(name)
        self.calls.append((name, kwargs))
        yield Observation()

    def create_score(self, **kwargs) -> None:
        self.scores.append(kwargs)


def test_query_trace_and_feedback_without_recording_content(tmp_path) -> None:
    settings = Settings(
        index_dir=tmp_path / "index",
        embedding_provider="hashing",
        reranker_enabled=False,
        llm_provider="extractive",
    )
    pipeline = build_pipeline(settings)
    client = Client()
    pipeline.observability.client = client
    pipeline.index_paths(sorted((ROOT / "sample_data" / "documents").glob("*")), persist=False)

    response = pipeline.query("P0 事件通知时限？", rerank=False)
    assert response["trace_id"] == "a" * 32
    assert {"rag-query", "vector-retrieval", "bm25-retrieval", "answer-generation"} <= set(
        client.names
    )
    assert dict(client.calls)["rag-query"]["input"] is None
    assert dict(client.calls)["answer-generation"]["input"] is None
    assert pipeline.observability.record_feedback(response["trace_id"], True, "有帮助")
    assert client.scores[0]["value"] == 1

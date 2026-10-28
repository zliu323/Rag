from contextlib import asynccontextmanager
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.dependencies import build_pipeline
from app.main import app
from app.rag.graph_workflow import GraphRAGWorkflow


def test_health_index_and_query(tmp_path: Path) -> None:
    settings = Settings(
        index_dir=tmp_path / "index",
        upload_dir=tmp_path / "uploads",
        embedding_provider="hashing",
        reranker_enabled=False,
        llm_provider="extractive",
    )
    settings.upload_dir.mkdir(parents=True)

    @asynccontextmanager
    async def test_lifespan(application):
        application.state.settings = settings
        application.state.pipeline = build_pipeline(settings)
        application.state.graph_workflow = GraphRAGWorkflow(application.state.pipeline)
        yield

    original = app.router.lifespan_context
    app.router.lifespan_context = test_lifespan
    try:
        with TestClient(app, raise_server_exceptions=True) as client:
            health = client.get("/health")
            assert health.status_code == 200
            uploaded = client.post(
                "/documents/upload",
                files={"files": ("extra.txt", "额外的知识库内容。", "text/plain")},
            )
            assert uploaded.status_code == 200
            assert uploaded.json()["count"] == 1
            indexed = client.post(
                "/index", json={"include_samples": True, "include_uploads": True}
            )
            assert indexed.status_code == 200
            response = client.post(
                "/query", json={"question": "年度学习预算是多少？", "rerank": False}
            )
            assert response.status_code == 200
            assert response.json()["citations"]
            assert response.json()["trace_id"] is None
            graph_response = client.post(
                "/query/graph", json={"question": "年度学习预算是多少？", "rerank": False}
            )
            assert graph_response.status_code == 200
            assert graph_response.json()["decision"] == "answered"
            assert graph_response.json()["retrieval_attempts"] == 1
            assert graph_response.json()["citations"]
            feedback = client.post(
                "/feedback",
                json={"trace_id": "a" * 32, "helpful": True},
            )
            assert feedback.status_code == 503
    finally:
        app.router.lifespan_context = original

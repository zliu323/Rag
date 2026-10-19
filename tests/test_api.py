from contextlib import asynccontextmanager
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.dependencies import build_pipeline
from app.main import app


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
    finally:
        app.router.lifespan_context = original

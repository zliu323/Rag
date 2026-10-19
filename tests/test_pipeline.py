from pathlib import Path

from app.config import ROOT, Settings
from app.dependencies import build_pipeline


def test_lightweight_pipeline_end_to_end(tmp_path: Path) -> None:
    settings = Settings(
        index_dir=tmp_path / "index",
        upload_dir=tmp_path / "uploads",
        embedding_provider="hashing",
        reranker_enabled=False,
        llm_provider="extractive",
        chunk_size=350,
        chunk_overlap=60,
    )
    pipeline = build_pipeline(settings)
    stats = pipeline.index_paths(sorted((ROOT / "sample_data" / "documents").glob("*")))
    response = pipeline.query("P0 事件需要在几分钟内通知负责人？", rerank=False)
    assert stats["documents"] == 3
    assert stats["chunks"] >= 3
    assert response["answer"]
    assert response["citations"]
    assert (tmp_path / "index" / "manifest.json").exists()


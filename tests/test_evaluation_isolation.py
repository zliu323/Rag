import json

import pytest

from app.config import ROOT, Settings
from app.dependencies import build_pipeline
from app.retrieval.pgvector_store import PgVectorStore
from app.retrieval.vector_store import FaissVectorStore
from evaluation.evaluate import evaluate, main
from evaluation.experiment import run_experiments


def make_stale_index(directory):
    directory.mkdir()
    (directory / "manifest.json").write_text(
        json.dumps({"dimension": 384, "embedder": "another-model", "count": 1}),
        encoding="utf-8",
    )
    (directory / "vectors.faiss").write_bytes(b"do not load or overwrite this index")
    (directory / "chunks.json").write_text("[]", encoding="utf-8")


@pytest.mark.parametrize("backend", ["faiss", "pgvector"])
def test_evaluation_never_loads_saves_or_contacts_application_store(tmp_path, monkeypatch, backend):
    directory = tmp_path / "application-index"
    make_stale_index(directory)
    before = {path.name: path.read_bytes() for path in directory.iterdir()}

    def forbidden(*args, **kwargs):
        pytest.fail("Evaluation accessed application persistence")

    monkeypatch.setattr(FaissVectorStore, "load", forbidden)
    monkeypatch.setattr(FaissVectorStore, "save", forbidden)
    monkeypatch.setattr(PgVectorStore, "_drivers", forbidden)
    settings = Settings(
        index_dir=directory,
        vector_backend=backend,
        embedding_provider="hashing",
        reranker_enabled=False,
        llm_provider="extractive",
        observability_enabled=True,
        langfuse_public_key=None,
        langfuse_secret_key=None,
    )
    report = evaluate(settings, ROOT / "sample_data" / "evaluation" / "qa.json", 5)
    assert len(report["examples"]) == 4
    assert report["summary"]["hit@5"] == 1
    assert before == {path.name: path.read_bytes() for path in directory.iterdir()}
    assert settings.vector_backend == backend
    assert settings.observability_enabled


def test_application_loading_still_validates_embedding_identity(tmp_path):
    directory = tmp_path / "application-index"
    make_stale_index(directory)
    settings = Settings(
        index_dir=directory,
        embedding_provider="hashing",
        reranker_enabled=False,
        llm_provider="extractive",
    )
    with pytest.raises(ValueError, match="different embedding model"):
        build_pipeline(settings)


def test_lightweight_cli_defaults_to_local_generation_even_with_deepseek_env(
    monkeypatch,
    capsys,
):
    monkeypatch.setenv("RAG_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("RAG_DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("RAG_VECTOR_BACKEND", "pgvector")
    monkeypatch.setattr("sys.argv", ["evaluate", "--lightweight"])
    main()
    report = json.loads(capsys.readouterr().out)
    assert len(report["examples"]) == 4
    assert {row["decision"] for row in report["examples"]} == {"answered"}


def test_parameter_experiments_also_use_isolated_store(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Parameter grid accessed application persistence")

    monkeypatch.setattr(FaissVectorStore, "load", forbidden)
    monkeypatch.setattr(FaissVectorStore, "save", forbidden)
    monkeypatch.setattr(PgVectorStore, "_drivers", forbidden)
    monkeypatch.setenv("RAG_VECTOR_BACKEND", "pgvector")
    reports = run_experiments(lightweight=True)
    assert len(reports) == 16

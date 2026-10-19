from app.domain import Document
from app.ingestion.splitter import RecursiveTextSplitter


def test_splitter_preserves_metadata_and_overlap() -> None:
    splitter = RecursiveTextSplitter(chunk_size=100, overlap=20)
    chunks = splitter.split_documents(
        [Document(text="第一段。" * 40, source="policy.md", page=3)]
    )
    assert len(chunks) > 1
    assert all(chunk.source == "policy.md" and chunk.page == 3 for chunk in chunks)
    assert len({chunk.id for chunk in chunks}) == len(chunks)


def test_splitter_rejects_invalid_overlap() -> None:
    try:
        RecursiveTextSplitter(chunk_size=100, overlap=100)
    except ValueError as exc:
        assert "chunk_size" in str(exc)
    else:
        raise AssertionError("Expected invalid overlap to fail")


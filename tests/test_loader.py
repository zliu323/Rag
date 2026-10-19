from pathlib import Path

import pytest

from app.exceptions import EmptyDocumentError, UnsupportedDocumentError
from app.ingestion.loader import load_document


def test_load_markdown(tmp_path: Path) -> None:
    path = tmp_path / "guide.md"
    path.write_text("# Guide\n\nUseful content", encoding="utf-8")
    documents = load_document(path)
    assert documents[0].source == "guide.md"
    assert "Useful content" in documents[0].text


def test_reject_unsupported_and_empty(tmp_path: Path) -> None:
    unsupported = tmp_path / "data.csv"
    unsupported.write_text("x", encoding="utf-8")
    with pytest.raises(UnsupportedDocumentError):
        load_document(unsupported)
    empty = tmp_path / "empty.txt"
    empty.write_text("  ", encoding="utf-8")
    with pytest.raises(EmptyDocumentError):
        load_document(empty)


def test_load_pdf_with_page_number(tmp_path: Path) -> None:
    import fitz

    path = tmp_path / "policy.pdf"
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Page one policy")
    pdf.save(path)
    pdf.close()

    documents = load_document(path)
    assert documents[0].source == "policy.pdf"
    assert documents[0].page == 1
    assert "Page one policy" in documents[0].text

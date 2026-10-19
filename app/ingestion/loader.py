from __future__ import annotations

from pathlib import Path

from app.domain import Document
from app.exceptions import EmptyDocumentError, UnsupportedDocumentError

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


def load_document(path: str | Path) -> list[Document]:
    file_path = Path(path)
    extension = file_path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedDocumentError(
            f"Unsupported file type '{extension}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    if extension == ".pdf":
        documents = _load_pdf(file_path)
    else:
        documents = _load_text(file_path)
    if not documents or not any(document.text.strip() for document in documents):
        raise EmptyDocumentError(f"No extractable text found in {file_path.name}")
    return documents


def _load_pdf(path: Path) -> list[Document]:
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - dependency error
        raise RuntimeError("PyMuPDF is required to read PDF files") from exc

    pages: list[Document] = []
    with fitz.open(path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append(Document(text=text, source=path.name, page=page_number))
    return pages


def _load_text(path: Path) -> list[Document]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8-sig")
    return [Document(text=text.strip(), source=path.name)]


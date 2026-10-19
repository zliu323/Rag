from __future__ import annotations

import hashlib
import re

from app.domain import Chunk, Document


class RecursiveTextSplitter:
    """Character-aware splitter that prefers paragraph and sentence boundaries."""

    def __init__(self, chunk_size: int = 700, overlap: int = 120) -> None:
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("Require chunk_size > overlap >= 0")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_documents(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        source_counters: dict[str, int] = {}
        for document in documents:
            source_counters.setdefault(document.source, 0)
            for text in self._split_text(document.text):
                index = source_counters[document.source]
                raw_id = f"{document.source}:{document.page}:{index}:{text}"
                chunks.append(
                    Chunk(
                        id=hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:20],
                        text=text,
                        source=document.source,
                        page=document.page,
                        chunk_index=index,
                    )
                )
                source_counters[document.source] += 1
        return chunks

    def _split_text(self, text: str) -> list[str]:
        cleaned = re.sub(r"[ \t]+", " ", text).strip()
        if not cleaned:
            return []
        chunks: list[str] = []
        start = 0
        length = len(cleaned)
        while start < length:
            hard_end = min(start + self.chunk_size, length)
            end = hard_end
            if hard_end < length:
                window = cleaned[start:hard_end]
                candidates = [window.rfind(sep) for sep in ("\n\n", "\n", "。", ". ", "; ", " ")]
                boundary = max(candidates)
                if boundary >= int(self.chunk_size * 0.55):
                    end = start + boundary + 1
            piece = cleaned[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= length:
                break
            next_start = end - self.overlap
            if next_start <= start:
                next_start = end
            start = next_start
        return chunks


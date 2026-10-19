from __future__ import annotations

import logging
from pathlib import Path

from app.domain import SearchResult
from app.generation.llm import LLMProvider
from app.ingestion.loader import load_document
from app.ingestion.splitter import RecursiveTextSplitter
from app.retrieval.reranker import Reranker
from app.retrieval.vector_store import FaissVectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(
        self,
        vector_store: FaissVectorStore,
        splitter: RecursiveTextSplitter,
        llm: LLMProvider,
        reranker: Reranker | None = None,
        top_k: int = 8,
        rerank_top_n: int = 4,
    ) -> None:
        self.vector_store = vector_store
        self.splitter = splitter
        self.llm = llm
        self.reranker = reranker
        self.top_k = top_k
        self.rerank_top_n = rerank_top_n

    def index_paths(self, paths: list[str | Path], persist: bool = True) -> dict[str, int]:
        documents = []
        for path in paths:
            documents.extend(load_document(path))
        chunks = self.splitter.split_documents(documents)
        self.vector_store.build(chunks)
        if persist:
            self.vector_store.save()
        source_count = len({document.source for document in documents})
        logger.info("index_built documents=%s chunks=%s", source_count, len(chunks))
        return {"documents": source_count, "chunks": len(chunks)}

    def retrieve(
        self, question: str, top_k: int | None = None, rerank: bool = True
    ) -> list[SearchResult]:
        results = self.vector_store.search(question, top_k or self.top_k)
        if rerank and self.reranker:
            return self.reranker.rerank(question, results, self.rerank_top_n)
        return results

    def query(self, question: str, top_k: int | None = None, rerank: bool = True) -> dict:
        contexts = self.retrieve(question, top_k=top_k, rerank=rerank)
        answer = self.llm.generate(question, contexts)
        citations = [
            {
                "id": index,
                "source": item.chunk.source,
                "page": item.chunk.page,
                "chunk_id": item.chunk.id,
                "score": round(item.score, 6),
                "retrieval_score": (
                    round(item.retrieval_score, 6) if item.retrieval_score is not None else None
                ),
                "text": item.chunk.text,
            }
            for index, item in enumerate(contexts, start=1)
        ]
        return {"question": question, "answer": answer, "citations": citations}

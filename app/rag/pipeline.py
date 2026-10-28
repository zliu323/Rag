from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from app.domain import Chunk, SearchResult
from app.generation.llm import LLMProvider
from app.generation.prompt import SYSTEM_PROMPT
from app.ingestion.loader import load_document
from app.ingestion.splitter import RecursiveTextSplitter
from app.observability import Observability
from app.retrieval.lexical import BM25Index, reciprocal_rank_fusion
from app.retrieval.reranker import Reranker
from app.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(
        self,
        vector_store: VectorStore,
        splitter: RecursiveTextSplitter,
        llm: LLMProvider,
        reranker: Reranker | None = None,
        top_k: int = 8,
        rerank_top_n: int = 4,
        hybrid_enabled: bool = True,
        observability: Observability | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.splitter = splitter
        self.llm = llm
        self.reranker = reranker
        self.top_k = top_k
        self.rerank_top_n = rerank_top_n
        self.hybrid_enabled = hybrid_enabled
        self.observability = observability
        existing = vector_store.all_chunks() if vector_store.ready else []
        self.lexical = BM25Index(existing) if hybrid_enabled and existing else None
        self.index_version = self._index_version(existing) if existing else None

    @staticmethod
    def _index_version(chunks: list[Chunk]) -> str:
        ids = "\n".join(sorted(chunk.id for chunk in chunks))
        return hashlib.sha256(ids.encode("utf-8")).hexdigest()[:12]

    def index_paths(self, paths: list[str | Path], persist: bool = True) -> dict[str, int]:
        documents = []
        for path in paths:
            documents.extend(load_document(path))
        chunks = self.splitter.split_documents(documents)
        self.vector_store.build(chunks)
        if persist:
            self.vector_store.save()
        self.lexical = BM25Index(chunks) if self.hybrid_enabled else None
        self.index_version = self._index_version(chunks)
        source_count = len({document.source for document in documents})
        logger.info("index_built documents=%s chunks=%s", source_count, len(chunks))
        return {"documents": source_count, "chunks": len(chunks)}

    def retrieve(
        self, question: str, top_k: int | None = None, rerank: bool = True
    ) -> list[SearchResult]:
        limit = top_k or self.top_k
        if self.observability:
            with self.observability.observe("vector-retrieval", kind="retriever") as span:
                dense = self.vector_store.search(question, limit)
                span.update(
                    output={
                        "matches": [
                            {"id": item.chunk.id, "score": item.score} for item in dense
                        ]
                    }
                )
        else:
            dense = self.vector_store.search(question, limit)
        results = dense
        if self.lexical is not None:
            if self.observability:
                with self.observability.observe("bm25-retrieval", kind="retriever") as span:
                    lexical = self.lexical.search(question, limit)
                    span.update(
                        output={
                            "matches": [
                                {"id": item.chunk.id, "score": item.score} for item in lexical
                            ]
                        }
                    )
            else:
                lexical = self.lexical.search(question, limit)
            results = reciprocal_rank_fusion(dense, lexical, limit)
        if rerank and self.reranker:
            if self.observability:
                with self.observability.observe("cross-encoder-rerank", kind="retriever") as span:
                    ranked = self.reranker.rerank(question, results, self.rerank_top_n)
                    span.update(
                        output={
                            "matches": [
                                {"id": item.chunk.id, "score": item.score} for item in ranked
                            ]
                        }
                    )
                    return ranked
            return self.reranker.rerank(question, results, self.rerank_top_n)
        return results

    def query(self, question: str, top_k: int | None = None, rerank: bool = True) -> dict:
        trace_id = None
        if self.observability:
            with self.observability.observe(
                "rag-query",
                input=self.observability.content({"question": question}),
                metadata={
                    "hybrid": self.hybrid_enabled,
                    "backend": type(self.vector_store).__name__,
                    "embedding_model": self.vector_store.embedder.identity,
                    "top_k": top_k or self.top_k,
                    "rerank": rerank,
                    "index_version": self.index_version,
                    "prompt_version": hashlib.sha256(
                        SYSTEM_PROMPT.encode("utf-8")
                    ).hexdigest()[:12],
                },
            ) as trace:
                trace_id = getattr(trace, "trace_id", None)
                contexts = self.retrieve(question, top_k=top_k, rerank=rerank)
                answer = self.answer_from_contexts(question, contexts)
                trace.update(
                    output=self.observability.content(answer),
                    metadata={"citations": len(contexts), "answer_length": len(answer)},
                )
        else:
            contexts = self.retrieve(question, top_k=top_k, rerank=rerank)
            answer = self.answer_from_contexts(question, contexts)
        return self.format_response(question, answer, contexts, trace_id)

    def answer_from_contexts(self, question: str, contexts: list[SearchResult]) -> str:
        if not self.observability:
            return self.llm.generate(question, contexts)
        with self.observability.observe(
            "answer-generation",
            kind="generation",
            model=getattr(self.llm, "model", "extractive"),
            input=self.observability.content(
                {"question": question, "contexts": [item.chunk.text for item in contexts]}
            ),
        ) as generation:
            answer, usage = self.llm.generate_with_metadata(question, contexts)
            generation.update(
                output=self.observability.content(answer),
                usage_details=usage or None,
                metadata={"context_chunk_ids": [item.chunk.id for item in contexts]},
            )
            return answer

    @staticmethod
    def format_response(
        question: str, answer: str, contexts: list[SearchResult], trace_id: str | None
    ) -> dict:
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
        return {
            "question": question,
            "answer": answer,
            "citations": citations,
            "trace_id": trace_id,
        }

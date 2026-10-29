from __future__ import annotations

from app.config import Settings
from app.generation.llm import build_llm
from app.ingestion.splitter import RecursiveTextSplitter
from app.observability import Observability
from app.rag.pipeline import RAGPipeline
from app.retrieval.embedding import build_embedder
from app.retrieval.pgvector_store import PgVectorStore
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_store import FaissVectorStore


def build_pipeline(settings: Settings, *, load_index: bool = True) -> RAGPipeline:
    embedder = build_embedder(
        settings.embedding_provider, settings.embedding_model, settings.embedding_device
    )
    store = (
        PgVectorStore(embedder, settings.postgres_dsn)
        if settings.vector_backend == "pgvector"
        else FaissVectorStore(embedder, settings.index_dir)
    )
    if load_index:
        store.load()
    reranker = (
        CrossEncoderReranker(settings.reranker_model, settings.embedding_device)
        if settings.reranker_enabled
        else None
    )
    llm = build_llm(
        settings.llm_provider,
        settings.llm_base_url,
        settings.llm_model,
        settings.llm_api_key,
        deepseek_base_url=settings.deepseek_base_url,
        deepseek_model=settings.deepseek_model,
        deepseek_api_key=settings.deepseek_api_key,
    )
    return RAGPipeline(
        vector_store=store,
        splitter=RecursiveTextSplitter(settings.chunk_size, settings.chunk_overlap),
        llm=llm,
        reranker=reranker,
        top_k=settings.top_k,
        rerank_top_n=settings.rerank_top_n,
        hybrid_enabled=settings.hybrid_enabled,
        context_min_coverage=settings.context_min_coverage,
        observability=Observability(settings),
    )

from app.config import Settings
from app.dependencies import build_pipeline
from app.rag.pipeline import RAGPipeline


def build_evaluation_pipeline(settings: Settings) -> RAGPipeline:
    # pgvector.build() writes even when persist=False. Always use a fresh in-memory FAISS
    # store and disable application tracing for fixture evaluation.
    isolated = settings.model_copy(
        update={"vector_backend": "faiss", "observability_enabled": False}
    )
    return build_pipeline(isolated, load_index=False)

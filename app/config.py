from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_prefix="RAG_", extra="ignore"
    )

    environment: str = "development"
    log_level: str = "INFO"
    index_dir: Path = ROOT / "data" / "index"
    upload_dir: Path = ROOT / "data" / "uploads"
    vector_backend: Literal["faiss", "pgvector"] = "faiss"
    postgres_dsn: str = "postgresql://rag:rag@localhost:5432/rag"
    hybrid_enabled: bool = True
    observability_enabled: bool = False
    observability_capture_content: bool = False
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"
    embedding_provider: Literal["sentence-transformers", "hashing"] = "sentence-transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_provider: Literal["extractive", "openai-compatible", "deepseek"] = "extractive"
    llm_model: str = "gpt-4.1-mini"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str | None = None
    deepseek_model: str = "deepseek-flash"
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_api_key: str | None = Field(default=None, repr=False)
    chunk_size: int = Field(700, ge=100, le=4000)
    chunk_overlap: int = Field(120, ge=0, le=1000)
    top_k: int = Field(8, ge=1, le=50)
    rerank_top_n: int = Field(4, ge=1, le=20)
    max_upload_mb: int = Field(20, ge=1, le=200)

    @model_validator(mode="after")
    def validate_overlap(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()

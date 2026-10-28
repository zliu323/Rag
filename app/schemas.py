from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
    include_samples: bool = True
    include_uploads: bool = True


class IndexResponse(BaseModel):
    documents: int
    chunks: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    rerank: bool = True


class Citation(BaseModel):
    id: int
    source: str
    page: int | None
    chunk_id: str
    score: float
    retrieval_score: float | None
    text: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    trace_id: str | None = None


class FeedbackRequest(BaseModel):
    trace_id: str = Field(pattern=r"^[0-9a-fA-F]{32}$")
    helpful: bool
    comment: str | None = Field(default=None, max_length=1000)


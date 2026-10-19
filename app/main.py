from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app.config import ROOT, get_settings
from app.dependencies import build_pipeline
from app.exceptions import RAGError
from app.ingestion.loader import SUPPORTED_EXTENSIONS
from app.logging import configure_logging
from app.schemas import IndexRequest, IndexResponse, QueryRequest, QueryResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    logger.info("initializing_pipeline embedding_provider=%s", settings.embedding_provider)
    app.state.settings = settings
    app.state.pipeline = build_pipeline(settings)
    yield


app = FastAPI(
    title="Enterprise Knowledge Base RAG",
    version="1.0.0",
    description="Grounded document Q&A with FAISS retrieval, reranking and citations.",
    lifespan=lifespan,
)


@app.exception_handler(RAGError)
async def rag_error_handler(_: Request, exc: RAGError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def unexpected_error_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled_error")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health")
def health(request: Request) -> dict:
    pipeline = request.app.state.pipeline
    return {
        "status": "ok",
        "index_ready": pipeline.vector_store.ready,
        "indexed_chunks": len(pipeline.vector_store.chunks),
        "embedding_provider": request.app.state.settings.embedding_provider,
        "llm_provider": request.app.state.settings.llm_provider,
    }


@app.post("/documents/upload")
async def upload_documents(
    request: Request, files: Annotated[list[UploadFile], File()]
) -> dict:
    settings = request.app.state.settings
    saved = []
    for upload in files:
        original = Path(upload.filename or "document").name
        extension = Path(original).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise HTTPException(status_code=415, detail=f"Unsupported file type: {extension}")
        destination = settings.upload_dir / f"{uuid.uuid4().hex[:10]}_{original}"
        size = 0
        with destination.open("wb") as output:
            while content := await upload.read(1024 * 1024):
                size += len(content)
                if size > settings.max_upload_mb * 1024 * 1024:
                    output.close()
                    destination.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail=f"{original} exceeds upload limit")
                output.write(content)
        saved.append(destination.name)
    return {"uploaded": saved, "count": len(saved)}


@app.post("/index", response_model=IndexResponse)
def create_index(payload: IndexRequest, request: Request) -> dict:
    settings = request.app.state.settings
    paths: list[Path] = []
    if payload.include_samples:
        sample_dir = ROOT / "sample_data" / "documents"
        paths.extend(
            path
            for path in sample_dir.iterdir()
            if path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    if payload.include_uploads:
        paths.extend(
            path
            for path in settings.upload_dir.iterdir()
            if path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    if not paths:
        raise HTTPException(status_code=400, detail="No supported documents found to index")
    return request.app.state.pipeline.index_paths(paths)


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest, request: Request) -> dict:
    return request.app.state.pipeline.query(payload.question, payload.top_k, payload.rerank)

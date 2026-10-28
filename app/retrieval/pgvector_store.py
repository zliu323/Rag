from __future__ import annotations

import numpy as np

from app.domain import Chunk, SearchResult
from app.exceptions import IndexNotReadyError
from app.retrieval.embedding import Embedder
from app.retrieval.vector_store import VectorStore


class PgVectorStore(VectorStore):
    """Transactional shared vector index. The embedding identity is stored with the index."""

    def __init__(self, embedder: Embedder, dsn: str) -> None:
        self.embedder = embedder
        self.dsn = dsn
        self._count = 0

    @staticmethod
    def _drivers():
        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError as exc:
            raise RuntimeError(
                "Install psycopg[binary] and pgvector for the postgres backend"
            ) from exc
        return psycopg, register_vector

    def _connect(self):
        psycopg, register_vector = self._drivers()
        connection = psycopg.connect(self.dsn)
        try:
            register_vector(connection)
        except Exception:
            connection.close()
            raise
        return connection

    def _ensure_schema(self) -> None:
        psycopg, _ = self._drivers()
        with psycopg.connect(self.dsn) as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            connection.execute(
                f"""CREATE TABLE IF NOT EXISTS rag_chunks (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    source TEXT NOT NULL,
                    page INTEGER,
                    chunk_index INTEGER NOT NULL,
                    embedding vector({self.embedder.dimension}) NOT NULL
                )"""
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS rag_chunks_embedding_hnsw "
                "ON rag_chunks USING hnsw (embedding vector_cosine_ops)"
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS rag_index_manifest (
                    singleton BOOLEAN PRIMARY KEY DEFAULT true CHECK (singleton),
                    embedder TEXT NOT NULL,
                    dimension INTEGER NOT NULL
                )"""
            )

    @property
    def ready(self) -> bool:
        return self.count > 0

    @property
    def count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT count(*) FROM rag_chunks").fetchone()[0])

    def all_chunks(self) -> list[Chunk]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, text, source, page, chunk_index FROM rag_chunks "
                "ORDER BY source, chunk_index"
            ).fetchall()
        return [Chunk(*row) for row in rows]

    def load(self) -> bool:
        self._ensure_schema()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT embedder, dimension FROM rag_index_manifest WHERE singleton = true"
            ).fetchone()
            if row and row != (self.embedder.identity, self.embedder.dimension):
                raise ValueError(
                    "Stored pgvector index uses a different embedding model or dimension"
                )
            self._count = int(connection.execute("SELECT count(*) FROM rag_chunks").fetchone()[0])
        return self._count > 0

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build an index with no chunks")
        vectors = np.asarray(
            self.embedder.encode([chunk.text for chunk in chunks]), dtype=np.float32
        )
        if vectors.shape != (len(chunks), self.embedder.dimension):
            raise ValueError("Embedder returned an unexpected vector shape")
        with self._connect() as connection:
            with connection.transaction():
                connection.execute("DELETE FROM rag_chunks")
                connection.cursor().executemany(
                    """INSERT INTO rag_chunks (id, text, source, page, chunk_index, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)""",
                    [
                        (chunk.id, chunk.text, chunk.source, chunk.page, chunk.chunk_index, vector)
                        for chunk, vector in zip(chunks, vectors, strict=True)
                    ],
                )
                connection.execute(
                    """INSERT INTO rag_index_manifest (singleton, embedder, dimension)
                    VALUES (true, %s, %s)
                    ON CONFLICT (singleton) DO UPDATE SET
                        embedder = EXCLUDED.embedder, dimension = EXCLUDED.dimension""",
                    (self.embedder.identity, self.embedder.dimension),
                )
        self._count = len(chunks)

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        if not self.ready:
            raise IndexNotReadyError("No index is loaded. Upload or index documents first.")
        vector = np.asarray(self.embedder.encode([query])[0], dtype=np.float32)
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT id, text, source, page, chunk_index, 1 - (embedding <=> %s) AS score
                FROM rag_chunks ORDER BY embedding <=> %s LIMIT %s""",
                (vector, vector, top_k),
            ).fetchall()
        return [SearchResult(Chunk(*row[:5]), float(row[5])) for row in rows]

    def save(self) -> None:
        # build() commits the index and manifest together.
        pass

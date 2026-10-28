import os

import pytest

from app.domain import Chunk
from app.retrieval.embedding import HashingEmbedder
from app.retrieval.pgvector_store import PgVectorStore


@pytest.mark.skipif(not os.getenv("RAG_TEST_POSTGRES_DSN"), reason="Requires test pgvector DB")
def test_pgvector_build_reload_and_search() -> None:
    dsn = os.environ["RAG_TEST_POSTGRES_DSN"]
    store = PgVectorStore(HashingEmbedder(), dsn)
    store.load()
    store.build(
        [
            Chunk("policy", "P0 事件需要十五分钟通知负责人", "policy.md", None, 0),
            Chunk("leave", "员工每年有十天年假", "leave.md", None, 0),
        ]
    )
    assert store.count == 2
    assert store.search("P0 事件通知", 1)[0].chunk.id == "policy"
    reloaded = PgVectorStore(HashingEmbedder(), dsn)
    assert reloaded.load()
    assert {item.id for item in reloaded.all_chunks()} == {"policy", "leave"}

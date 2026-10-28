from app.domain import Chunk
from app.retrieval.embedding import HashingEmbedder
from app.retrieval.vector_store import FaissVectorStore


def test_faiss_index_round_trip_in_chinese_directory(tmp_path) -> None:
    directory = tmp_path / "工作" / "索引"
    embedder = HashingEmbedder()
    store = FaissVectorStore(embedder, directory)
    store.build(
        [Chunk(id="one", text="P0 事件应通知负责人", source="制度.txt", page=None, chunk_index=0)]
    )
    store.save()

    loaded = FaissVectorStore(embedder, directory)
    assert loaded.load()
    assert loaded.count == 1
    assert loaded.search("P0 事件通知", 1)[0].chunk.id == "one"

from app.domain import Chunk, SearchResult
from app.retrieval.lexical import BM25Index, reciprocal_rank_fusion


def chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id, text, "guide.md", None, 0)


def test_bm25_finds_exact_identifier() -> None:
    index = BM25Index([chunk("a", "普通员工休假制度"), chunk("b", "工单 P0-3847 需要升级")])
    results = index.search("P0-3847", 2)
    assert results[0].chunk.id == "b"
    assert results[0].score > 0


def test_rrf_rewards_results_found_by_both_methods() -> None:
    a, b, c = (chunk(key, key) for key in "abc")
    dense = [SearchResult(a, 0.9), SearchResult(b, 0.8)]
    lexical = [SearchResult(b, 5.0), SearchResult(c, 4.0)]
    fused = reciprocal_rank_fusion(dense, lexical, 3)
    assert [item.chunk.id for item in fused] == ["b", "a", "c"]
    assert len({item.chunk.id for item in fused}) == 3

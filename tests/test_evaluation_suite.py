from app.domain import Chunk, SearchResult
from evaluation.suite import (
    evidence_metrics,
    load_dataset,
    paired_interval,
    score_answer,
    summarize,
)


def test_extended_dataset_has_traceable_evidence_and_negative_cases():
    examples, paths = load_dataset()
    assert len(examples) == 60
    assert len(paths) == 6
    assert sum(bool(item["evidence"]) for item in examples) == 50
    assert sum(item["category"] == "multi_source" for item in examples) >= 1


def test_evidence_requires_correct_source_and_all_evidence_units():
    results = [SearchResult(Chunk("a", "needle", "a.md", None, 0), 0.9)]
    metrics = evidence_metrics(results, [["a.md", "needle"], ["b.md", "needle"]])
    assert metrics == {"hit": 1.0, "evidence_recall": 0.5, "all_evidence_hit": 0.0, "mrr": 1.0}
    assert evidence_metrics(results, []) == {}


def test_keyword_aliases_citation_bounds_and_refusal():
    score = score_answer("预算5000元 [2]", [["5,000", "5000"]], 1)
    assert score["keyword_coverage_proxy"] == 1.0
    assert score["citation_format_valid"] == 0.0
    assert score_answer("资料未提供此信息", [], 1)["refusal_proxy"] == 1.0
    assert score_answer("答案是42", [], 1)["refusal_proxy"] == 0.0


def test_summary_separates_generation_failures_from_retrieval_and_negatives():
    report = summarize(
        [
            {"id": "a", "hit": 1.0, "evidence_recall": 0.5, "error": "ProviderError"},
            {"id": "b", "refusal_proxy": 1.0, "retrieval_ms": 1.0, "generation_ms": 2.0},
        ]
    )
    assert report["hit"] == 1.0
    assert report["hit_n"] == 1
    assert report["refusal_proxy_n"] == 1
    assert report["errors"] == 1
    assert summarize([])["generation_ms"] is None


def test_pairing_excludes_unpaired_examples():
    comparison = paired_interval(
        [
            {"id": "a", "strategy": "dense", "hit": 0.0},
            {"id": "a", "strategy": "hybrid", "hit": 1.0},
            {"id": "b", "strategy": "hybrid", "hit": 0.0},
        ],
        "hit",
    )
    assert comparison["n"] == 1
    assert comparison["hybrid_minus_dense"] == 1.0

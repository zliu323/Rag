from app.domain import Chunk, SearchResult
from evaluation.metrics import answer_keyword_correctness, retrieval_metrics


def result(source: str, text: str) -> SearchResult:
    return SearchResult(Chunk(source, text, source, None, 0), score=0.5)


def test_retrieval_metrics() -> None:
    results = [result("wrong.md", "other"), result("right.md", "needle here")]
    metrics = retrieval_metrics(results, ["right.md"], ["needle"], k=2)
    assert metrics == {"hit@2": 1.0, "recall@2": 1.0, "mrr": 0.5}


def test_answer_keyword_correctness() -> None:
    metrics = answer_keyword_correctness(
        "Budget is 5,000 and needs approval", ["5,000", "approval"]
    )
    assert metrics["answer_correctness"] == 1.0

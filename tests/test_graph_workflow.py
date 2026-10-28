from app.domain import Chunk, SearchResult
from app.rag.graph_workflow import REFUSAL, GraphRAGWorkflow


def result(text: str) -> SearchResult:
    return SearchResult(Chunk("chunk-1", text, "policy.txt", None, 0), 0.5)


class StubPipeline:
    def __init__(self, first: list[SearchResult], second: list[SearchResult]) -> None:
        self.first = first
        self.second = second
        self.calls: list[str] = []
        self.generated = False
        self.observability = None

    def retrieve(self, question: str, top_k: int | None, rerank: bool) -> list[SearchResult]:
        self.calls.append(question)
        return self.first if len(self.calls) == 1 else self.second

    def answer_from_contexts(self, question: str, contexts: list[SearchResult]) -> str:
        self.generated = True
        return "负责人应在15分钟内收到通知。"

    @staticmethod
    def format_response(question: str, answer: str, contexts: list[SearchResult], trace_id):
        return {
            "question": question,
            "answer": answer,
            "citations": [item.chunk.id for item in contexts],
            "trace_id": trace_id,
        }


def test_graph_rewrites_once_then_answers_with_second_retrieval() -> None:
    pipeline = StubPipeline(
        first=[result("员工年度学习预算为1000元。")],
        second=[result("P0事件需要在15分钟内通知负责人。")],
    )
    response = GraphRAGWorkflow(pipeline).query(
        "请问 P0 事件需要在几分钟内通知负责人？", rerank=False
    )
    assert response["decision"] == "answered"
    assert response["retrieval_attempts"] == 2
    assert response["rewritten_question"] == "P0 事件需要在几分钟内通知负责人"
    assert len(pipeline.calls) == 2
    assert response["citations"] == ["chunk-1"]
    assert pipeline.generated


def test_graph_refuses_after_bounded_retry_without_generation() -> None:
    pipeline = StubPipeline(first=[], second=[])
    response = GraphRAGWorkflow(pipeline).query("请问月球基地如何建设？")
    assert response["decision"] == "refused"
    assert response["answer"] == REFUSAL
    assert response["retrieval_attempts"] == 2
    assert response["citations"] == []
    assert not pipeline.generated


def test_graph_skips_redundant_retry_when_rewrite_is_unchanged() -> None:
    pipeline = StubPipeline(first=[], second=[])
    response = GraphRAGWorkflow(pipeline).query("月球基地如何建设？")
    assert response["decision"] == "refused"
    assert response["retrieval_attempts"] == 1
    assert pipeline.calls == ["月球基地如何建设？"]

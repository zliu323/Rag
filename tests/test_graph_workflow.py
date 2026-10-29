import pytest

from app.domain import Chunk, GenerationResult, SearchResult
from app.rag.graph_workflow import REFUSAL, GraphRAGWorkflow
from app.rag.grounding import has_relevant_context


def result(text: str) -> SearchResult:
    return SearchResult(Chunk("chunk-1", text, "policy.txt", None, 0), 0.5)


class StubPipeline:
    def __init__(
        self,
        first: list[SearchResult],
        second: list[SearchResult],
        answer: str = "负责人应在15分钟内收到通知。",
        decision: str = "answered",
    ) -> None:
        self.first = first
        self.second = second
        self.answer = answer
        self.calls: list[str] = []
        self.generated = False
        self.observability = None
        self.decision = decision

    def retrieve(self, question: str, top_k: int | None, rerank: bool) -> list[SearchResult]:
        self.calls.append(question)
        return self.first if len(self.calls) == 1 else self.second

    def has_sufficient_context(self, question: str, contexts: list[SearchResult]) -> bool:
        return has_relevant_context(question, contexts)

    def answer_result_from_contexts(
        self, question: str, contexts: list[SearchResult]
    ) -> GenerationResult:
        self.generated = True
        return GenerationResult(self.answer, self.decision, {})

    @staticmethod
    def format_response(
        question: str, answer: str, contexts: list[SearchResult], trace_id, decision="answered"
    ):
        return {
            "question": question,
            "answer": answer,
            "citations": [item.chunk.id for item in contexts],
            "trace_id": trace_id,
            "decision": decision,
        }


def test_graph_rewrites_once_then_answers_with_second_retrieval() -> None:
    pipeline = StubPipeline(
        first=[],
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


def test_graph_retries_nonempty_but_unrelated_results_then_refuses() -> None:
    pipeline = StubPipeline(first=[result("员工年度学习预算为1000元。")], second=[])
    response = GraphRAGWorkflow(pipeline).query("请问 P0 事件需要在几分钟内通知负责人？")
    assert response["decision"] == "refused"
    assert response["retrieval_attempts"] == 2
    assert response["citations"] == []
    assert not pipeline.generated


@pytest.mark.parametrize(
    "answer",
    [
        REFUSAL,
        "资料未提供修复时限，无法确定。",
        "The context does not specify a resolution time.",
    ],
)
def test_graph_uses_structured_refusal_regardless_of_wording(answer) -> None:
    pipeline = StubPipeline(
        first=[result("P1工单的首次响应时限为2小时。")],
        second=[],
        answer=answer,
        decision="refused",
    )
    response = GraphRAGWorkflow(pipeline).query("P1工单的修复时限是多少？")
    assert response["decision"] == "refused"
    assert response["citations"] == []
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

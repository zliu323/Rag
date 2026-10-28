from __future__ import annotations

import re
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.domain import SearchResult
from app.rag.pipeline import RAGPipeline
from app.text import tokenize

REFUSAL = "现有知识库内容不足以回答该问题。"
_FILLER = re.compile(r"请帮我查一下|能不能告诉我|根据知识库|帮我查一下|我想知道|请告诉我|请问")
_STOPWORDS = {
    "what", "how", "is", "are", "the", "a", "an", "of", "to", "in", "for", "please",
    "的", "了", "在", "是", "有", "吗", "呢", "我", "你", "这", "个", "该",
}


class GraphState(TypedDict, total=False):
    question: str
    search_query: str
    top_k: int | None
    rerank: bool
    attempts: int
    contexts: list[SearchResult]
    coverage: float
    sufficient: bool
    answer: str
    decision: Literal["answered", "refused"]


def rewrite_query(question: str) -> str:
    """Remove polite framing while preserving the factual question."""
    return _normalize_query(_FILLER.sub("", question))


def _normalize_query(question: str) -> str:
    return re.sub(r"^[\s，,。.!！？?]+|[\s，,。.!！？?]+$", "", question)


def query_terms(question: str) -> set[str]:
    simplified = re.sub(r"什么|多少|如何|怎么|是否", "", rewrite_query(question))
    return set(tokenize(simplified)) - _STOPWORDS


class GraphRAGWorkflow:
    """Bounded LangGraph controller around the existing retrieval and generation pipeline."""

    def __init__(self, pipeline: RAGPipeline, min_coverage: float = 0.4) -> None:
        if not 0 <= min_coverage <= 1:
            raise ValueError("min_coverage must be between 0 and 1")
        self.pipeline = pipeline
        self.min_coverage = min_coverage
        graph = StateGraph(GraphState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("judge_evidence", self._judge_evidence)
        graph.add_node("rewrite", self._rewrite)
        graph.add_node("answer", self._answer)
        graph.add_node("refuse", self._refuse)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "judge_evidence")
        graph.add_conditional_edges(
            "judge_evidence",
            self._route,
            {"answer": "answer", "rewrite": "rewrite", "refuse": "refuse"},
        )
        graph.add_edge("rewrite", "retrieve")
        graph.add_edge("answer", END)
        graph.add_edge("refuse", END)
        self.graph = graph.compile()

    def _retrieve(self, state: GraphState) -> GraphState:
        return {
            "contexts": self.pipeline.retrieve(
                state["search_query"], top_k=state["top_k"], rerank=state["rerank"]
            ),
            "attempts": state["attempts"] + 1,
        }

    def _judge_evidence(self, state: GraphState) -> GraphState:
        terms = query_terms(state["question"])
        matches = [
            len(terms.intersection(tokenize(item.chunk.text)))
            for item in state["contexts"]
        ]
        best = max(matches, default=0)
        coverage = best / len(terms) if terms else 0.0
        sufficient = bool(terms) and best >= min(2, len(terms)) and coverage >= self.min_coverage
        return {"coverage": round(coverage, 4), "sufficient": sufficient}

    @staticmethod
    def _route(state: GraphState) -> Literal["answer", "rewrite", "refuse"]:
        if state["sufficient"]:
            return "answer"
        if state["attempts"] == 1 and rewrite_query(state["question"]) != _normalize_query(
            state["search_query"]
        ):
            return "rewrite"
        return "refuse"

    @staticmethod
    def _rewrite(state: GraphState) -> GraphState:
        return {"search_query": rewrite_query(state["question"])}

    def _answer(self, state: GraphState) -> GraphState:
        return {
            "answer": self.pipeline.answer_from_contexts(state["question"], state["contexts"]),
            "decision": "answered",
        }

    @staticmethod
    def _refuse(_state: GraphState) -> GraphState:
        return {"answer": REFUSAL, "decision": "refused"}

    def query(self, question: str, top_k: int | None = None, rerank: bool = True) -> dict:
        initial: GraphState = {
            "question": question,
            "search_query": question,
            "top_k": top_k,
            "rerank": rerank,
            "attempts": 0,
        }
        trace_id = None
        observability = self.pipeline.observability
        if observability:
            with observability.observe(
                "rag-graph-query",
                input=observability.content({"question": question}),
                metadata={
                    "top_k": top_k or self.pipeline.top_k,
                    "rerank": rerank,
                    "index_version": self.pipeline.index_version,
                    "evidence_min_coverage": self.min_coverage,
                },
            ) as trace:
                trace_id = getattr(trace, "trace_id", None)
                result = self.graph.invoke(initial)
                trace.update(
                    output=observability.content(result["answer"]),
                    metadata={
                        "attempts": result["attempts"],
                        "coverage": result["coverage"],
                        "decision": result["decision"],
                    },
                )
        else:
            result = self.graph.invoke(initial)
        contexts = result["contexts"] if result["decision"] == "answered" else []
        response = self.pipeline.format_response(question, result["answer"], contexts, trace_id)
        response.update(
            decision=result["decision"],
            retrieval_attempts=result["attempts"],
            evidence_coverage=result["coverage"],
            rewritten_question=(result["search_query"] if result["attempts"] > 1 else None),
        )
        return response

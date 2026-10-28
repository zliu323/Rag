from __future__ import annotations

import re
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.domain import SearchResult
from app.generation.llm import INSUFFICIENT_CONTEXT_ANSWER
from app.rag.pipeline import RAGPipeline

REFUSAL = INSUFFICIENT_CONTEXT_ANSWER
_FILLER = re.compile(r"请帮我查一下|能不能告诉我|根据知识库|帮我查一下|我想知道|请告诉我|请问")


class GraphState(TypedDict, total=False):
    question: str
    search_query: str
    top_k: int | None
    rerank: bool
    attempts: int
    contexts: list[SearchResult]
    answer: str
    decision: Literal["answered", "refused"]


def rewrite_query(question: str) -> str:
    """Remove polite framing while preserving the factual question."""
    return _normalize_query(_FILLER.sub("", question))


def _normalize_query(question: str) -> str:
    return re.sub(r"^[\s，,。.!！？?]+|[\s，,。.!！？?]+$", "", question)


class GraphRAGWorkflow:
    """Bounded LangGraph controller around the existing retrieval and generation pipeline."""

    def __init__(self, pipeline: RAGPipeline) -> None:
        self.pipeline = pipeline
        graph = StateGraph(GraphState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("rewrite", self._rewrite)
        graph.add_node("answer", self._answer)
        graph.add_node("refuse", self._refuse)
        graph.add_edge(START, "retrieve")
        graph.add_conditional_edges(
            "retrieve",
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

    @staticmethod
    def _route(state: GraphState) -> Literal["answer", "rewrite", "refuse"]:
        if state["contexts"]:
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
        answer = self.pipeline.answer_from_contexts(state["question"], state["contexts"])
        return {
            "answer": answer,
            "decision": "refused" if answer == REFUSAL else "answered",
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
                },
            ) as trace:
                trace_id = getattr(trace, "trace_id", None)
                result = self.graph.invoke(initial)
                trace.update(
                    output=observability.content(result["answer"]),
                    metadata={
                        "attempts": result["attempts"],
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
            rewritten_question=(result["search_query"] if result["attempts"] > 1 else None),
        )
        return response

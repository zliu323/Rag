from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from dataclasses import replace
from pathlib import Path

from app.config import ROOT
from app.domain import Chunk
from app.generation.llm import ExtractiveProvider
from app.ingestion.loader import load_document
from app.ingestion.splitter import RecursiveTextSplitter
from app.rag.pipeline import RAGPipeline
from app.retrieval.embedding import HashingEmbedder
from app.retrieval.vector_store import FaissVectorStore
from evaluation.metrics import mean_metrics, retrieval_metrics


def percentile(values: list[float], percentage: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def load_chunks(chunk_size: int, overlap: int, multiplier: int) -> list[Chunk]:
    documents = []
    for path in sorted((ROOT / "sample_data" / "documents").glob("*")):
        documents.extend(load_document(path))
    base_chunks = RecursiveTextSplitter(chunk_size, overlap).split_documents(documents)
    return [
        replace(chunk, id=f"{chunk.id}-copy-{copy_index}")
        for copy_index in range(multiplier)
        for chunk in base_chunks
    ]


def run_configuration(
    chunks: list[Chunk],
    examples: list[dict],
    *,
    hybrid: bool,
    repeats: int,
    warmups: int,
    top_k: int,
) -> dict:
    store = FaissVectorStore(HashingEmbedder(), ROOT / "data" / "benchmark-unused")
    index_started = time.perf_counter()
    store.build(chunks)
    index_ms = (time.perf_counter() - index_started) * 1000
    pipeline = RAGPipeline(
        vector_store=store,
        splitter=RecursiveTextSplitter(700, 120),
        llm=ExtractiveProvider(),
        top_k=top_k,
        hybrid_enabled=hybrid,
    )

    questions = [example["question"] for example in examples]
    for index in range(warmups):
        pipeline.retrieve(questions[index % len(questions)], top_k=top_k, rerank=False)

    latencies_ms = []
    started = time.perf_counter()
    for index in range(repeats):
        question = questions[index % len(questions)]
        query_started = time.perf_counter()
        pipeline.retrieve(question, top_k=top_k, rerank=False)
        latencies_ms.append((time.perf_counter() - query_started) * 1000)
    elapsed = time.perf_counter() - started

    metric_rows = []
    for example in examples:
        results = pipeline.retrieve(example["question"], top_k=top_k, rerank=False)
        metric_rows.append(
            retrieval_metrics(
                results,
                example["relevant_sources"],
                example.get("relevant_text_contains"),
                top_k,
            )
        )
    return {
        "name": "hybrid" if hybrid else "dense_only",
        "index_ms": round(index_ms, 3),
        "retrieval": mean_metrics(metric_rows),
        "latency_ms": {
            "mean": round(statistics.fmean(latencies_ms), 3),
            "p50": round(percentile(latencies_ms, 0.50), 3),
            "p95": round(percentile(latencies_ms, 0.95), 3),
            "p99": round(percentile(latencies_ms, 0.99), 3),
        },
        "sequential_qps": round(repeats / elapsed, 3),
    }


def benchmark(multiplier: int, repeats: int, warmups: int, top_k: int) -> dict:
    examples = json.loads(
        (ROOT / "sample_data" / "evaluation" / "qa.json").read_text(encoding="utf-8")
    )
    chunks = load_chunks(700, 120, multiplier)
    return {
        "disclaimer": (
            "Local synthetic-load benchmark using repeated sample chunks; "
            "not a production traffic or business-impact measurement."
        ),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "embedding": "hashing:v1:384",
        },
        "workload": {
            "base_qa_examples": len(examples),
            "corpus_multiplier": multiplier,
            "chunks": len(chunks),
            "measured_queries": repeats,
            "warmup_queries": warmups,
            "top_k": top_k,
        },
        "results": [
            run_configuration(
                chunks,
                examples,
                hybrid=False,
                repeats=repeats,
                warmups=warmups,
                top_k=top_k,
            ),
            run_configuration(
                chunks,
                examples,
                hybrid=True,
                repeats=repeats,
                warmups=warmups,
                top_k=top_k,
            ),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a reproducible local retrieval benchmark")
    parser.add_argument("--multiplier", type=int, default=250)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--warmups", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "benchmark.json")
    args = parser.parse_args()
    if min(args.multiplier, args.repeats, args.top_k) < 1 or args.warmups < 0:
        parser.error("multiplier, repeats and top-k must be positive; warmups cannot be negative")

    report = benchmark(args.multiplier, args.repeats, args.warmups, args.top_k)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

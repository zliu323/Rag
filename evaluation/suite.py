"""Versioned synthetic QA suite; supports offline and real DeepSeek generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from app.config import ROOT, Settings
from app.domain import SearchResult
from app.generation.llm import ExtractiveProvider, build_llm
from app.ingestion.loader import load_document
from app.ingestion.splitter import RecursiveTextSplitter
from app.rag.pipeline import RAGPipeline
from app.retrieval.embedding import HashingEmbedder
from app.retrieval.vector_store import FaissVectorStore
from evaluation.benchmark import percentile

FIXTURES = ROOT / "evaluation" / "fixtures"


def load_dataset() -> tuple[list[dict], list[Path]]:
    examples = json.loads((FIXTURES / "qa_extended.json").read_text(encoding="utf-8"))
    paths = sorted((ROOT / "sample_data" / "documents").glob("*"))
    paths += sorted((FIXTURES / "documents").glob("*"))
    texts = {path.name: path.read_text(encoding="utf-8") for path in paths}
    if len({item["id"] for item in examples}) != len(examples):
        raise ValueError("Duplicate example IDs")
    for item in examples:
        if bool(item["evidence"]) != bool(item["keywords"]):
            raise ValueError(f"Inconsistent answerability: {item['id']}")
        for source, text in item["evidence"]:
            if text not in texts.get(source, ""):
                raise ValueError(f"Evidence missing in source: {item['id']}: {source}")
    return examples, paths


def evidence_metrics(results: list[SearchResult], evidence: list[list[str]]) -> dict:
    """Recall over annotated evidence units, including all sources for multi-hop QA."""
    if not evidence:
        return {}
    ranks = [
        next(
            (
                rank
                for rank, result in enumerate(results, 1)
                if result.chunk.source == source and text in result.chunk.text
            ),
            0,
        )
        for source, text in evidence
    ]
    found = [rank for rank in ranks if rank]
    return {
        "hit": float(bool(found)),
        "evidence_recall": len(found) / len(evidence),
        "all_evidence_hit": float(len(found) == len(evidence)),
        "mrr": 1 / min(found) if found else 0.0,
    }


def score_answer(answer: str, keywords: list[list[str]], context_count: int) -> dict:
    normalized = re.sub(r"\s+", "", answer.lower()).replace(",", "")
    if not keywords:
        # Lexical refusal proxy only; retained answers allow manual review of false matches.
        refusal = bool(
            re.search(
                r"不足|无法|不能确定|不能回答|未(?:提供|提及|说明|规定|明确|给出)|没有.*(?:信息|说明|提及|规定)|不包含|不清楚",
                answer,
            )
        )
        return {"refusal_proxy": float(refusal)}
    coverage = sum(
        any(re.sub(r"\s+", "", value.lower()).replace(",", "") in normalized for value in group)
        for group in keywords
    ) / len(keywords)
    citations = [int(value) for value in re.findall(r"\[(\d+)\]", answer)]
    return {
        "keyword_coverage_proxy": coverage,
        "citation_format_valid": float(
            bool(citations) and all(1 <= number <= context_count for number in citations)
        ),
    }


def summarize(rows: list[dict]) -> dict:
    successful = [row for row in rows if "error" not in row]
    summary = {
        "attempted": len(rows),
        "succeeded": len(successful),
        "errors": len(rows) - len(successful),
    }
    for key in (
        "hit",
        "evidence_recall",
        "all_evidence_hit",
        "mrr",
        "keyword_coverage_proxy",
        "citation_format_valid",
        "refusal_proxy",
    ):
        population = (
            rows if key in {"hit", "evidence_recall", "all_evidence_hit", "mrr"} else successful
        )
        values = [row[key] for row in population if key in row]
        summary[key] = sum(values) / len(values) if values else None
        summary[f"{key}_n"] = len(values)
    for key in ("retrieval_ms", "generation_ms"):
        values = [row[key] for row in successful]
        summary[key] = (
            {"p50": percentile(values, 0.5), "p95": percentile(values, 0.95)} if values else None
        )
    summary["token_usage"] = {
        key: sum(row.get("usage", {}).get(key, 0) for row in successful)
        for key in ("input", "output")
    }
    return summary


def paired_interval(rows: list[dict], metric: str) -> dict:
    pairs: dict[str, dict] = {}
    for row in rows:
        if metric in row and "error" not in row:
            pairs.setdefault(row["id"], {})[row["strategy"]] = row[metric]
    deltas = [
        pair["hybrid"] - pair["dense"]
        for pair in pairs.values()
        if "hybrid" in pair and "dense" in pair
    ]
    if not deltas:
        return {}
    rng = random.Random(42)
    samples = [sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(2000)]
    return {
        "n": len(deltas),
        "hybrid_minus_dense": sum(deltas) / len(deltas),
        "bootstrap_95_interval": [percentile(samples, 0.025), percentile(samples, 0.975)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm", choices=["extractive", "deepseek"], default="extractive")
    parser.add_argument("--strategy", choices=["dense", "hybrid", "both"], default="both")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=220)
    parser.add_argument("--overlap", type=int, default=40)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "suite.json")
    args = parser.parse_args()
    if args.k < 1 or args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("k, workers and limit must be positive")
    examples, paths = load_dataset()
    config = Settings()
    llm = (
        ExtractiveProvider()
        if args.llm == "extractive"
        else build_llm(
            "deepseek",
            config.llm_base_url,
            config.llm_model,
            None,
            deepseek_base_url=config.deepseek_base_url,
            deepseek_model=config.deepseek_model,
            deepseek_api_key=config.deepseek_api_key,
        )
    )
    code_paths = sorted((ROOT / "app").rglob("*.py"))
    code_paths += [ROOT / "evaluation" / "benchmark.py", Path(__file__)]
    hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [*paths, FIXTURES / "qa_extended.json", *code_paths]
    }
    settings = {
        "llm": args.llm,
        "model": getattr(llm, "model", "extractive"),
        "endpoint": getattr(llm, "endpoint", None),
        "embedding": "hashing:v1:384",
        "reranker": False,
        "k": args.k,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "strategy": args.strategy,
        "workers": args.workers,
    }
    fingerprint = hashlib.sha256(
        json.dumps([hashes, settings], sort_keys=True).encode()
    ).hexdigest()
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "fingerprint": fingerprint,
        "provenance": "Human-authored synthetic fixtures, no production users or traffic. "
        "Exploratory test set, not an independently held-out business benchmark.",
        "metric_notes": "Retrieval metrics use answerable QA only; recall counts exact annotated "
        "evidence units. Keyword coverage and refusal are lexical proxies. "
        "Citation checks validate numbering, not factual support. "
        "Bootstrap intervals resample questions; they do not model production traffic.",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "settings": settings,
        "file_sha256": hashes,
        "rows": [],
    }
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        if previous["fingerprint"] != fingerprint:
            raise ValueError("Resume fingerprint mismatch; use a new output file")
        report["rows"] = [row for row in previous["rows"] if "error" not in row]
    done = {(row["strategy"], row["id"]) for row in report["rows"]}
    documents = [doc for path in paths for doc in load_document(path)]
    splitter = RecursiveTextSplitter(args.chunk_size, args.overlap)
    chunks = splitter.split_documents(documents)
    store = FaissVectorStore(HashingEmbedder(), ROOT / "data" / "suite-unused")
    store.build(chunks)  # Isolated in-memory index; never loads or saves the application index.
    report["corpus"] = {"documents": len(paths), "chunks": len(chunks), "qa": len(examples)}
    strategies = ["dense", "hybrid"] if args.strategy == "both" else [args.strategy]
    jobs = []
    for strategy in strategies:
        pipeline = RAGPipeline(store, splitter, llm, hybrid_enabled=strategy == "hybrid")
        for example in examples[: args.limit]:
            if (strategy, example["id"]) in done:
                continue
            started = time.perf_counter()
            contexts = pipeline.retrieve(example["question"], top_k=args.k, rerank=False)
            retrieval_ms = (time.perf_counter() - started) * 1000
            jobs.append((strategy, example, contexts, retrieval_ms))

    def generate(job: tuple) -> dict:
        strategy, example, contexts, retrieval_ms = job
        row = {
            "id": example["id"],
            "category": example["category"],
            "strategy": strategy,
            "question": example["question"],
            "retrieval_ms": retrieval_ms,
            "contexts": [
                {"source": item.chunk.source, "chunk_id": item.chunk.id, "text": item.chunk.text}
                for item in contexts
            ],
            **evidence_metrics(contexts, example["evidence"]),
        }
        started = time.perf_counter()
        try:
            answer, usage = llm.generate_with_metadata(example["question"], contexts)
            row.update(
                answer=answer,
                usage=usage,
                generation_ms=(time.perf_counter() - started) * 1000,
                **score_answer(answer, example["keywords"], len(contexts)),
            )
        except Exception as exc:
            # Do not persist exception strings that could contain credentials/request details.
            row["error"] = type(exc).__name__
        return row

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(generate, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            report["rows"].append(row)
            args.output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                f"{len(report['rows'])}: {row['strategy']} {row['id']} "
                f"{'ERROR ' + row['error'] if 'error' in row else 'OK'}",
                flush=True,
            )
    report["rows"].sort(key=lambda row: (row["strategy"], row["id"]))
    report["summary"] = {
        strategy: summarize([row for row in report["rows"] if row["strategy"] == strategy])
        for strategy in strategies
    }
    report["paired_comparison"] = {
        key: paired_interval(report["rows"], key) for key in ("hit", "evidence_recall", "mrr")
    }
    report["completed_at"] = datetime.now(UTC).isoformat()
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if any("error" in row for row in report["rows"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import itertools
import json
import tempfile
from pathlib import Path

from app.config import ROOT, Settings
from app.dependencies import build_pipeline
from evaluation.metrics import mean_metrics, retrieval_metrics


def run_experiments(lightweight: bool = False) -> list[dict]:
    examples = json.loads(
        (ROOT / "sample_data" / "evaluation" / "qa.json").read_text(encoding="utf-8")
    )
    documents = sorted((ROOT / "sample_data" / "documents").glob("*"))
    grid = itertools.product([350, 700], [60, 120], [3, 5], [False, True])
    reports = []
    with tempfile.TemporaryDirectory() as directory:
        for chunk_size, overlap, top_k, use_reranker in grid:
            if overlap >= chunk_size:
                continue
            settings = Settings(
                index_dir=Path(directory) / f"{chunk_size}-{overlap}-{top_k}-{use_reranker}",
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                top_k=top_k,
                reranker_enabled=use_reranker and not lightweight,
                embedding_provider="hashing" if lightweight else "sentence-transformers",
            )
            pipeline = build_pipeline(settings)
            pipeline.index_paths(documents, persist=False)
            metrics = []
            for example in examples:
                results = pipeline.retrieve(example["question"], top_k=top_k, rerank=use_reranker)
                metrics.append(
                    retrieval_metrics(
                        results,
                        example["relevant_sources"],
                        example.get("relevant_text_contains"),
                        top_k,
                    )
                )
            reports.append(
                {
                    "config": {
                        "chunk_size": chunk_size,
                        "overlap": overlap,
                        "top_k": top_k,
                        "rerank": use_reranker and not lightweight,
                    },
                    "metrics": mean_metrics(metrics),
                }
            )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small RAG parameter grid")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "experiments.json")
    parser.add_argument("--lightweight", action="store_true")
    args = parser.parse_args()
    report = run_experiments(args.lightweight)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(report)} experiment rows to {args.output}")


if __name__ == "__main__":
    main()


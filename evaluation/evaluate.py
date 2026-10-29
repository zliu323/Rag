from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import ROOT, Settings
from evaluation.metrics import answer_keyword_correctness, mean_metrics, retrieval_metrics
from evaluation.runtime import build_evaluation_pipeline


def evaluate(settings: Settings, qa_path: Path, k: int) -> dict:
    pipeline = build_evaluation_pipeline(settings)
    documents = sorted((ROOT / "sample_data" / "documents").glob("*"))
    pipeline.index_paths(documents, persist=False)
    examples = json.loads(qa_path.read_text(encoding="utf-8"))
    rows = []
    for example in examples:
        results = pipeline.retrieve(example["question"], top_k=k, rerank=True)
        retrieval = retrieval_metrics(
            results,
            example["relevant_sources"],
            example.get("relevant_text_contains"),
            k,
        )
        generated = pipeline.answer_result_from_contexts(example["question"], results)
        answer = generated.answer
        correctness = answer_keyword_correctness(answer, example.get("answer_keywords", []))
        rows.append(
            {
                "id": example["id"],
                "question": example["question"],
                "answer": answer,
                "decision": generated.decision,
                **retrieval,
                **correctness,
            }
        )
    numeric_rows = [
        {key: value for key, value in row.items() if isinstance(value, float)} for row in rows
    ]
    return {"summary": mean_metrics(numeric_rows), "examples": rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and fallback answer quality")
    parser.add_argument("--qa", type=Path, default=ROOT / "sample_data" / "evaluation" / "qa.json")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--llm",
        choices=["extractive", "deepseek", "openai-compatible"],
        default="extractive",
        help="Defaults to local generation; external API generation must be explicitly selected",
    )
    parser.add_argument(
        "--lightweight",
        action="store_true",
        help="Use hashing embeddings and no reranker",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {"llm_provider": args.llm}
    if args.lightweight:
        overrides.update(embedding_provider="hashing", reranker_enabled=False)
    report = evaluate(Settings(**overrides), args.qa, args.k)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

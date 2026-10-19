from tempfile import TemporaryDirectory

from app.config import ROOT, Settings
from app.dependencies import build_pipeline


def main() -> None:
    with TemporaryDirectory() as index_dir:
        settings = Settings(
            index_dir=index_dir,
            embedding_provider="hashing",
            reranker_enabled=False,
            llm_provider="extractive",
        )
        pipeline = build_pipeline(settings)
        stats = pipeline.index_paths(sorted((ROOT / "sample_data" / "documents").glob("*")))
        result = pipeline.query("员工每年有多少天带薪年假？", rerank=False)
        assert stats["documents"] >= 2 and result["citations"]
        print(
            {
                "index": stats,
                "answer": result["answer"],
                "citation": result["citations"][0]["source"],
            }
        )


if __name__ == "__main__":
    main()

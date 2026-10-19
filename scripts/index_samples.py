from app.config import ROOT, Settings
from app.dependencies import build_pipeline


def main() -> None:
    pipeline = build_pipeline(Settings())
    paths = sorted((ROOT / "sample_data" / "documents").glob("*"))
    print(pipeline.index_paths(paths))


if __name__ == "__main__":
    main()


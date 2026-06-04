from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence


STAGES: dict[str, list[str]] = {
    "rag": [
        "src.pdf_loader",
        "src.chunking",
        "src.vector_store",
    ],
    "features": [
        "src.price_loader",
        "src.target_builder",
        "src.llm_feature_builder",
        "src.price_feature_builder",
        "src.dataset_builder",
    ],
    "models": [
        "src.model_training",
        "src.model_evaluation",
        "src.backtest",
    ],
}


def _modules_for_stage(stage: str) -> list[str]:
    if stage == "all":
        return STAGES["rag"] + STAGES["features"] + STAGES["models"]
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    return STAGES[stage]


def run_modules(modules: Sequence[str]) -> None:
    """Run pipeline modules in sequence using the current Python interpreter."""
    for module in modules:
        print(f"\n=== Running python -m {module} ===", flush=True)
        subprocess.run([sys.executable, "-m", module], check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the earnings RAG quant pipeline.")
    parser.add_argument(
        "--stage",
        choices=["rag", "features", "models", "all"],
        default="all",
        help="Pipeline section to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_modules(_modules_for_stage(args.stage))


if __name__ == "__main__":
    main()

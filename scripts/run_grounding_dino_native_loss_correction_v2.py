#!/usr/bin/env python
"""Run the corrected GroundingDINO-T architecture adaptation on local 1444."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fair_baselines.training import run_baseline  # noqa: E402


DEFAULT_OUTPUT = ROOT / "experiments" / "baseline_faithful_reproduction" / "20260712_v2" / "corrections"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    args = parser.parse_args()
    output_dir = args.output_root.resolve() / "grounding_dino_native" / f"seed_{args.seed}"
    result = run_baseline(
        model_key="grounding_dino_native",
        output_dir=output_dir,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=1,
        image_size=640,
        learning_rate=args.learning_rate,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

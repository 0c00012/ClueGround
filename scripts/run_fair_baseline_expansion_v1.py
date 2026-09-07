#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fair_baselines.training import run_baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=["grounding_dino", "grounding_dino_native", "lvit", "guide_decoder", "reclmis"],
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--protocol", choices=["multibox_1444", "singlebox_888"], default="multibox_1444")
    args = parser.parse_args()
    output_dir = Path(args.output_root).resolve() / "runs" / args.model / f"seed_{args.seed}"
    result = run_baseline(
        model_key=args.model,
        output_dir=output_dir,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        image_size=args.image_size,
        learning_rate=args.learning_rate,
        **({"data_root": args.data_root.resolve()} if args.data_root is not None else {}),
        protocol=args.protocol,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

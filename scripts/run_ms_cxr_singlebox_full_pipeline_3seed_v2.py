#!/usr/bin/env python
"""Complete strict MS-CXR 888 YOLO--RAD-DINO full-pipeline seed repeats.

The four class-specific YOLO detectors were already trained on the exact
638/87/163 split for seeds 13, 42, and 2026. This runner reuses those immutable
seed-specific detector artifacts, retrains the RAD-DINO shallow query head for
each matching seed, retunes fusion on val only, and evaluates eval163 once.

No Chest ImaGenome/CIG task pretraining artifact is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_ms_cxr_rad_dino_singlebox_retrain_v1 as rad  # noqa: E402
from scripts import run_ms_cxr_singlebox_fair_yolo_dino_fusion_v1 as fusion  # noqa: E402


EXP = PROJECT_ROOT / "experiments" / "ms_cxr_singlebox_full_pipeline_3seed_v2"
YOLO_REPEAT_EXP = PROJECT_ROOT / "experiments" / "ms_cxr_singlebox_merged_yolo_dino_seed_repeats_v1"
YOLO_REPEAT_TRAIN = PROJECT_ROOT / "training" / "ms_cxr_singlebox_merged_yolo_dino_seed_repeats_v1"
YOLO_SEED42_EXP = PROJECT_ROOT / "experiments" / "ms_cxr_singlebox_fair_detector_sweep_v2"
YOLO_SEED42_TRAIN = PROJECT_ROOT / "training" / "ms_cxr_singlebox_fair_detector_sweep_v2"

MODELS = ("yolov8s", "yolov8m", "yolo11s", "yolo11m")
EXPECTED_SPLITS = {"train": 638, "val": 87, "eval": 163}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def configure_rad_paths(seed: int) -> Path:
    root = EXP / f"seed_{seed}" / "rad_dino"
    train = EXP / f"seed_{seed}" / "training" / "rad_dino"
    rad.EXP_NAME = f"ms_cxr_singlebox_full_pipeline_3seed_v2_s{seed}"
    rad.EXP = root
    rad.PRED = root / "predictions"
    rad.MET = root / "metrics"
    rad.CFG = root / "configs"
    rad.LOGS = root / "logs"
    rad.TRAIN = train
    rad.RUNS = train / "runs"
    rad.CKPT = train / "checkpoints"
    rad.REPORT = root / "report"

    fusion.PRED = root / "predictions"
    fusion.MET = root / "metrics"
    fusion.CFG = root / "configs"
    fusion.LOGS = root / "logs"
    fusion.REPORT = root / "report"
    fusion.RAD_CKPT = rad.CKPT
    return root


def yolo_artifacts(seed: int, split: str, model: str) -> tuple[Path, Path]:
    if seed == 42:
        candidate = YOLO_SEED42_EXP / "predictions" / f"{model}_{split}_conf0p001_candidates.csv"
        weight = YOLO_SEED42_TRAIN / "runs" / f"{model}_singlebox_e100_s42" / "weights" / "best.pt"
    else:
        candidate = YOLO_REPEAT_EXP / "predictions" / f"{model}_s{seed}_{split}_conf0p001_candidates.csv"
        weight = YOLO_REPEAT_TRAIN / "runs" / f"{model}_singlebox_e100_s{seed}" / "weights" / "best.pt"
    return candidate, weight


def load_candidates(path: Path) -> Dict[str, List[Dict]]:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"dicom_id", "class_id", "score", "x1", "y1", "x2", "y2", "source_model", "rank"}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Candidate schema missing {sorted(missing)}: {path}")
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for r in df.to_dict("records"):
        grouped[str(r["dicom_id"])].append(
            {
                "class_id": int(r["class_id"]),
                "score": float(r["score"]),
                "box": [float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])],
                "source_model": str(r["source_model"]),
                "rank": int(r["rank"]),
            }
        )
    return grouped


def validate_contract(seeds: Iterable[int]) -> dict:
    counts = {split: len(fusion.row_dicts(split)) for split in EXPECTED_SPLITS}
    if counts != EXPECTED_SPLITS:
        raise RuntimeError(f"Unexpected strict-888 split counts: {counts}")
    artifacts = []
    for seed in seeds:
        for model in MODELS:
            for split in ("val", "eval"):
                candidate, weight = yolo_artifacts(seed, split, model)
                if not candidate.exists() or not weight.exists():
                    raise FileNotFoundError(f"seed={seed} model={model}: {candidate} / {weight}")
                artifacts.append(
                    {
                        "seed": seed,
                        "model": model,
                        "split": split,
                        "candidate": str(candidate),
                        "candidate_sha256": sha256(candidate),
                        "weight": str(weight),
                        "weight_sha256": sha256(weight),
                    }
                )
    report = {
        "status": "PASS",
        "protocol": "singlebox_888",
        "split_counts": counts,
        "seeds": list(seeds),
        "cig_task_pretraining": False,
        "finding_conditioned_yolo_classes": 8,
        "yolo_models": list(MODELS),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    write_json(EXP / "preflight.json", report)
    return report


def rad_args(seed: int, args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        variants=["full_phrase"],
        context_only=False,
        epochs=args.rad_epochs,
        batch_size=args.rad_batch_size,
        lr=1e-3,
        weight_decay=1e-4,
        dropout=0.1,
        hidden=384,
        patience=args.rad_patience,
        center_loss_weight=0.05,
        bootstrap_reps=200,
        seed=seed,
        quick=False,
    )


def run_seed(seed: int, args: argparse.Namespace) -> dict:
    seed_root = EXP / f"seed_{seed}"
    status_path = seed_root / "RUN_STATUS.json"
    if status_path.exists() and not args.force:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "complete" and status.get("full_pipeline_seed") is True:
            return status

    configure_rad_paths(seed)
    rad.set_seed(seed)
    start = time.time()
    write_json(status_path, {"status": "running", "seed": seed, "stage": "rad_dino_head"})
    rad_result = rad.run(rad_args(seed, args))

    device = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    dino_val_df = fusion.load_rad_prediction("full_phrase", "val", device, False)
    dino_eval_df = fusion.load_rad_prediction("full_phrase", "eval", device, False)
    dino_val = fusion.dino_map(dino_val_df)
    dino_eval = fusion.dino_map(dino_eval_df)

    train_rows = fusion.row_dicts("train")
    val_rows = fusion.row_dicts("val")
    eval_rows = fusion.row_dicts("eval")
    priors = fusion.ybase.make_train_priors(train_rows)

    val_sets = []
    eval_sets = []
    yolo_provenance = []
    for model in MODELS:
        val_path, weight = yolo_artifacts(seed, "val", model)
        eval_path, _ = yolo_artifacts(seed, "eval", model)
        val_sets.append(load_candidates(val_path))
        eval_sets.append(load_candidates(eval_path))
        yolo_provenance.append(
            {
                "model": model,
                "seed": seed,
                "weight": str(weight),
                "weight_sha256": sha256(weight),
                "val_candidates": str(val_path),
                "eval_candidates": str(eval_path),
            }
        )

    merged_val = fusion.merge_candidates(*val_sets)
    merged_eval = fusion.merge_candidates(*eval_sets)
    params, rule_grid, rule_per = fusion.yv2.tune(val_rows, merged_val, priors, False)
    out_met = seed_root / "metrics"
    out_pred = seed_root / "predictions"
    out_met.mkdir(parents=True, exist_ok=True)
    out_pred.mkdir(parents=True, exist_ok=True)
    rule_grid.to_csv(out_met / "rule_context_val_grid.csv", index=False)
    rule_per.to_csv(out_met / "rule_context_best_params_by_finding.csv", index=False)

    fusion_by_finding, fusion_grid, fusion_per = fusion.old_fusion.tune_fusion(
        val_rows, merged_val, priors, dino_val, params, False
    )
    fusion_grid.to_csv(out_met / "fusion_val_grid.csv", index=False)
    fusion_per.to_csv(out_met / "fusion_best_weights_by_finding.csv", index=False)
    method = f"ours_merged_yolo_rad_dino_strict888_s{seed}"
    pred = fusion.old_fusion.evaluate_fusion(
        eval_rows, merged_eval, priors, dino_eval, params, fusion_by_finding, method, "eval"
    )
    pred.to_csv(out_pred / "eval_predictions.csv", index=False)
    metric = fusion.metrics(pred, method, "all8")

    status = {
        "status": "complete",
        "protocol": "singlebox_888",
        "seed": seed,
        "full_pipeline_seed": True,
        "yolo_detectors_trained_with_seed": True,
        "yolo_seed_specific_artifacts_reused": True,
        "rad_dino_head_trained_with_seed": True,
        "rad_dino_backbone": "microsoft/rad-dino frozen",
        "cig_task_pretraining": False,
        "selection_split": "val only",
        "training_rows": 638,
        "validation_rows": 87,
        "eval_rows": 163,
        "mean_iou": metric["mean_iou"],
        "hit_0_3": metric["Hit@0.3"],
        "hit_0_5": metric["Hit@0.5"],
        "prediction_path": str(out_pred / "eval_predictions.csv"),
        "rad_result": rad_result,
        "yolo_provenance": yolo_provenance,
        "elapsed_sec": time.time() - start,
    }
    write_json(status_path, status)
    return status


def aggregate(rows: List[dict]) -> dict:
    metrics = ("mean_iou", "hit_0_3", "hit_0_5")
    out = {
        "status": "complete",
        "protocol": "singlebox_888",
        "n_seeds": len(rows),
        "seeds": [int(r["seed"]) for r in rows],
        "n_upstream_seeds": len(rows),
        "seed_scope": "four YOLO detectors and RAD-DINO shallow head repeated per seed",
        "cig_task_pretraining": False,
        "training_rows": 638,
        "validation_rows": 87,
        "eval_rows": 163,
    }
    for metric in metrics:
        values = np.asarray([float(r[metric]) for r in rows], dtype=float)
        out[f"{metric}_mean"] = float(values.mean())
        out[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return out


def run(args: argparse.Namespace) -> None:
    EXP.mkdir(parents=True, exist_ok=True)
    write_json(EXP / "QUEUE_STATUS.json", {"status": "preflight", "seeds": args.seeds})
    validate_contract(args.seeds)
    results = []
    try:
        for index, seed in enumerate(args.seeds, start=1):
            write_json(
                EXP / "QUEUE_STATUS.json",
                {
                    "status": "running",
                    "seed": seed,
                    "seed_index": index,
                    "n_seeds": len(args.seeds),
                    "completed_seeds": [int(r["seed"]) for r in results],
                },
            )
            results.append(run_seed(seed, args))
        aggregate_row = aggregate(results)
        final_dir = EXP / "final_tables"
        final_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).drop(columns=["rad_result", "yolo_provenance"]).to_csv(
            final_dir / "ours_strict888_full_pipeline_seed_results.csv", index=False
        )
        write_json(final_dir / "ours_strict888_full_pipeline_3seed_aggregate.json", aggregate_row)
        write_json(EXP / "FINAL_STATUS.json", aggregate_row)
        write_json(
            EXP / "QUEUE_STATUS.json",
            {"status": "complete", "completed_seeds": args.seeds, "aggregate": aggregate_row},
        )
    except Exception as exc:
        write_json(EXP / "QUEUE_STATUS.json", {"status": "failed", "error": repr(exc)})
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 2026])
    parser.add_argument("--rad-epochs", type=int, default=120)
    parser.add_argument("--rad-patience", type=int, default=20)
    parser.add_argument("--rad-batch-size", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

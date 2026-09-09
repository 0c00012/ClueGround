#!/usr/bin/env python
"""Learned candidate re-ranker for the direct-888 single-region protocol.

Same method as ``run_clueground_canonical_learned_reranker_3seed_v1`` applied
to the direct-888 upstream (train638 / val87 / eval163; four finding-
conditioned YOLO detectors and the phrase-conditioned RAD-DINO head trained
per seed on that split).  Feature set = the 1444 set without the auxiliary
xattn/light-detector estimates (those assets were trained on the 1444 split
and must not touch the 888 partition), i.e. detector evidence, geometry,
RAD-DINO agreement, cross-detector consensus, spatial prior and phrase
context.  Scorer family by 5-fold subject CV on train638; alpha and the
historical single-route calibration on val87 only; eval163 once per seed.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_clueground_canonical_learned_reranker_3seed_v1 as rr  # noqa: E402
from scripts import run_clueground_exact_hybrid_v4_direct_3seed_v1 as exact  # noqa: E402
from scripts import run_ms_cxr_singlebox_fair_yolo_dino_fusion_v1 as single_fusion  # noqa: E402
from scripts import run_ms_cxr_singlebox_full_pipeline_3seed_v2 as single_source  # noqa: E402

SEEDS = (13, 42, 2026)
SPLITS = ("train", "val", "eval")
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "clueground_direct888_learned_reranker_3seed_v1"
ALPHA_SELECTION = "val"


def log(message: str) -> None:
    print(f"[reranker-888 {time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_rows() -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, list[list[float]]]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    labels: dict[str, dict[str, list[list[float]]]] = {}
    for split in SPLITS:
        part = single_fusion.row_dicts(split)
        for row in part:
            row["group_id"] = str(row["task_id"])
        rows[split] = part
        labels[split] = {str(row["task_id"]): [list(map(float, row["gold_bbox_xyxy"]))] for row in part}
    return rows, labels


def load_candidates(seed: int) -> dict[str, dict[str, list[dict[str, Any]]]]:
    out = {}
    for split in SPLITS:
        parts = []
        for model in exact.MODELS:
            candidate_path, _ = single_source.yolo_artifacts(seed, split, model)
            parts.append(single_source.load_candidates(candidate_path))
        out[split] = single_fusion.merge_candidates(*parts)
    return out


def load_dino_boxes(seed: int, split: str) -> dict[str, list[float]]:
    path = single_source.EXP / f"seed_{seed}" / "rad_dino" / "predictions" / f"rad_dino_full_phrase_singlebox_{split}_predictions.csv"
    frame = pd.read_csv(path)
    return {
        str(r.sample_id): [float(r.pred_x1), float(r.pred_y1), float(r.pred_x2), float(r.pred_y2)]
        for r in frame.itertuples()
        if not bool(getattr(r, "bbox_missing", False))
    }


def run_seed(seed: int, output_root: Path) -> dict[str, Any]:
    seed_root = output_root / f"seed_{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    rows, labels = load_rows()
    candidates = load_candidates(seed)
    priors = single_fusion.ybase.make_train_priors(rows["train"])
    estimates = {split: {"dino": load_dino_boxes(seed, split), "xattn": {}, "aux": {}} for split in SPLITS}
    tables = {}
    for split in SPLITS:
        cache = seed_root / f"candidates_{split}.csv"
        if cache.exists():
            tables[split] = pd.read_csv(cache)
        else:
            tables[split] = rr.build_table(split, rows[split], labels[split], candidates[split], priors, estimates[split])
            tables[split].to_csv(cache, index=False)
        log(f"seed {seed}: table {split} rows={len(tables[split])} groups={tables[split].group_id.nunique()}")

    best_name, cv_table, oof = rr.select_scorer(tables["train"], seed)
    cv_table.to_csv(seed_root / "scorer_cv_selection.csv", index=False)
    cols = rr.feature_columns(tables["train"])
    model = rr.scorer_models(seed)[best_name]
    model.fit(tables["train"][cols].to_numpy(np.float32), tables["train"]["target_iou"].to_numpy(float))
    scored = {}
    for split in SPLITS:
        frame = tables[split].copy()
        frame["learned"] = oof[best_name] if split == "train" else np.clip(model.predict(frame[cols].to_numpy(np.float32)), 0.0, 1.0)
        frame.to_csv(seed_root / f"scored_{split}.csv", index=False)
        scored[split] = frame
    val_learned = rr.top1_iou(scored["val"], scored["val"]["learned"].to_numpy())
    val_conf = rr.top1_iou(scored["val"], scored["val"]["conf"].to_numpy())
    rr.write_json(seed_root / "scorer_selection.json", {"seed": seed, "selected_model": best_name, "cv": cv_table.to_dict("records"), "features": cols, "val_top1_iou_learned": val_learned, "val_top1_iou_confidence": val_conf})
    log(f"seed {seed}: scorer={best_name}; val top-1 IoU learned {val_learned:.4f} vs confidence {val_conf:.4f}")

    grid_rows = []
    for alpha in rr.ALPHA_GRID:
        adjusted_val = rr.adjust_with_learned(candidates["val"], scored["val"], alpha)
        val_metric = rr.val_top_candidate_iou(rows["val"], labels["val"], adjusted_val)
        record = {"alpha": alpha, "val_top_candidate_iou": val_metric}
        if ALPHA_SELECTION == "train_oof_val":
            # Train candidates carry out-of-fold learned scores, so the train
            # top-candidate IoU is a legitimate (eval-free) selection signal.
            adjusted_train = rr.adjust_with_learned(candidates["train"], scored["train"], alpha)
            train_metric = rr.val_top_candidate_iou(rows["train"], labels["train"], adjusted_train)
            n_train, n_val = len(rows["train"]), len(rows["val"])
            record["train_oof_top_candidate_iou"] = train_metric
            record["pooled_top_candidate_iou"] = (train_metric * n_train + val_metric * n_val) / (n_train + n_val)
        grid_rows.append(record)
    key = "pooled_top_candidate_iou" if ALPHA_SELECTION == "train_oof_val" else "val_top_candidate_iou"
    grid = pd.DataFrame(grid_rows).sort_values([key, "alpha"], ascending=[False, True])
    alpha = float(grid.iloc[0]["alpha"])
    grid.to_csv(seed_root / "alpha_val_grid.csv", index=False)
    log(f"seed {seed}: alpha={alpha} val top-candidate IoU {grid.iloc[0]['val_top_candidate_iou']:.4f} (alpha 0: {grid[grid.alpha == 0].val_top_candidate_iou.iloc[0]:.4f})")

    context = exact.load_single_context(seed)
    context.candidates = {split: rr.adjust_with_learned(candidates[split], scored[split], alpha) for split in ("val", "eval")}
    result = exact.run_protocol(context, output_root / "learned", quick=False, retune_single_full_val=True, separate_multi_route_params=True, calibration_cache_root=None)
    result.update({"variant": "learned", "seed": seed, "learned_alpha": alpha, "scorer": best_name})
    rr.write_json(seed_root / "RUN_STATUS.json", result)
    row = {"seed": seed, "scorer": best_name, "alpha": alpha}
    for key in ("mean_iou", "Hit@0.3", "Hit@0.5"):
        row[key] = float(result.get(key, float("nan")))
    log(f"seed {seed}: mean IoU {row['mean_iou']:.4f} Hit@0.3 {row['Hit@0.3']:.4f} Hit@0.5 {row['Hit@0.5']:.4f}")
    return row


def main() -> None:
    global ALPHA_SELECTION
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--alpha-grid", nargs="+", type=float, default=list(rr.ALPHA_GRID))
    parser.add_argument("--drop-features", nargs="*", default=["xattn_iou", "aux_iou"])
    parser.add_argument("--alpha-selection", choices=("val", "train_oof_val"), default="val")
    args = parser.parse_args()
    ALPHA_SELECTION = args.alpha_selection
    rr.ALPHA_GRID = tuple(args.alpha_grid)
    rr.FEATURE_EXCLUDE = set(rr.FEATURE_EXCLUDE) | set(args.drop_features)
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        rows.append(run_seed(seed, args.output_root))
        pd.DataFrame(rows).to_csv(args.output_root / "per_seed_metrics.csv", index=False)
    frame = pd.DataFrame(rows)
    status = {"status": "complete", "seeds": args.seeds, "dropped_features": args.drop_features, "alpha_selection": ALPHA_SELECTION}
    for key in ("mean_iou", "Hit@0.3", "Hit@0.5"):
        status[f"{key}_mean"] = float(frame[key].mean())
        status[f"{key}_std"] = float(frame[key].std(ddof=1)) if len(frame) > 1 else 0.0
    rr.write_json(args.output_root / "FINAL_STATUS.json", status)
    log(f"FINAL mean IoU {status['mean_iou_mean']:.4f} +/- {status['mean_iou_std']:.4f}")


if __name__ == "__main__":
    main()

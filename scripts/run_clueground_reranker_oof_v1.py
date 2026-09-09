#!/usr/bin/env python
"""Out-of-fold candidate generation for the learned re-ranker (distribution fix).

The re-ranker is trained on candidates of train813.  In the sealed runs those
candidates come from detectors and a RAD-DINO head that were trained on the
same images, so train-side features (confidence, RAD-DINO agreement) are
optimistically biased relative to val/eval.  This runner removes that bias:

  for each seed and each of K subject-grouped folds of train813
      train the four finding-conditioned YOLO detectors and the RAD-DINO head
      on train minus the fold (model selection on the canonical val124),
      predict candidates and RAD-DINO boxes for the held-out fold
  assemble the out-of-fold train candidate table (priors from train minus fold)
  train / select the scorer on that table (5-fold subject CV as before)
  apply it to the sealed val/eval candidate tables (full-train detectors),
  select alpha on val124, run the unchanged decoder, evaluate eval220 once.

GPU cost is roughly K x the upstream detector training per seed.  Stages are
resumable: completed folds are skipped.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_clueground_canonical_learned_reranker_3seed_v1 as rr  # noqa: E402
from scripts import run_clueground_canonical_neural_agreement_selector_pilot_v1 as canonical  # noqa: E402
from scripts import run_clueground_exact_hybrid_v4_direct_3seed_v1 as exact  # noqa: E402
from scripts import run_clueground_vfm_legacy_fusion_unified_3seed_v1 as legacy  # noqa: E402
from scripts import run_ms_cxr_yolo_rule_context_v1 as ybase  # noqa: E402
from src.three_task_grounding.contracts import get_protocol  # noqa: E402
from src.three_task_grounding.manifests import read_jsonl, write_jsonl  # noqa: E402
from src.three_task_grounding.pipeline import TaskRunConfig, ThreeTaskPipeline  # noqa: E402

SEEDS = (13, 42, 2026)
SPLITS = ("train", "val", "eval")
PROTOCOL_KEY = legacy.PROTOCOL_KEY
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "clueground_reranker_oof_v1"
SEALED_ROOT = PROJECT_ROOT / "experiments" / "clueground_canonical_learned_reranker_noaux_assets_v1"


def log(message: str) -> None:
    print(f"[oof {time.strftime('%H:%M:%S')}] {message}", flush=True)


def fold_assignment(inputs: list[dict[str, Any]], n_folds: int, seed: int) -> dict[str, int]:
    subjects = sorted({str(row["subject_id"]) for row in inputs})
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    subject_fold = {subject: index % n_folds for index, subject in enumerate(subjects)}
    return {str(row["group_id"]): subject_fold[str(row["subject_id"])] for row in inputs}


def write_fold_protocol(root: Path, inputs: dict, labels_rows: dict, fold_groups: set[str]) -> Path:
    """train = canonical train minus fold, val = canonical val, eval = fold."""
    protocol = root / "protocol" / PROTOCOL_KEY
    protocol.mkdir(parents=True, exist_ok=True)
    train_inputs = [r for r in inputs["train"] if str(r["group_id"]) not in fold_groups]
    fold_inputs = [r for r in inputs["train"] if str(r["group_id"]) in fold_groups]
    parts = {"train": train_inputs, "val": inputs["val"], "eval": fold_inputs}
    label_by_gid = {split: {str(r["group_id"]): r for r in labels_rows[split]} for split in ("train", "val")}
    for split, rows in parts.items():
        source_split = "val" if split == "val" else "train"
        write_jsonl(protocol / f"{split}_inputs.jsonl", [{**r, "split": split} for r in rows])
        write_jsonl(protocol / f"{split}_labels.jsonl", [{**label_by_gid[source_split][str(r["group_id"])], "split": split} for r in rows])
    return root / "protocol"


def run_fold(seed: int, fold: int, fold_root: Path, inputs: dict, labels: dict, labels_rows: dict, fold_groups: set[str], args: argparse.Namespace) -> None:
    marker = fold_root / "FOLD_COMPLETE.json"
    if marker.exists():
        log(f"seed {seed} fold {fold}: complete, skip")
        return
    protocol_root = write_fold_protocol(fold_root, inputs, labels_rows, fold_groups)
    config = TaskRunConfig.from_json(legacy.DEFAULT_CONFIG)
    config = replace(
        config,
        yolo_models=tuple(exact.MODELS),
        yolo_epochs=args.yolo_epochs,
        candidate_top_per_detector=30,
        candidate_limit=160,
        pipeline_seed=seed,
    )
    legacy.set_seed(seed)
    stage_marker = fold_root / PROTOCOL_KEY / "yolo_runs" / "STAGE_COMPLETE.json"
    pipeline = ThreeTaskPipeline(get_protocol(PROTOCOL_KEY), protocol_root, fold_root, config, disable_rad_dino=True, force=not stage_marker.exists())
    pipeline.build_yolo_dataset()
    log(f"seed {seed} fold {fold}: training four YOLO on {len(pipeline.inputs['train'])} train groups")
    weights = pipeline.train_yolo()
    legacy.write_json(stage_marker, {"status": "complete", "seed": seed, "fold": fold, "models": {k: str(v) for k, v in weights.items()}})
    pipeline.predict_yolo(weights)

    fold_inputs = read_jsonl(protocol_root / PROTOCOL_KEY / "eval_inputs.jsonl")
    train_inputs = read_jsonl(protocol_root / PROTOCOL_KEY / "train_inputs.jsonl")
    ids = {
        "train": legacy.resolve_source_task_ids_from_inputs(train_inputs, "train"),
        "val": legacy.resolve_source_task_ids_from_inputs(inputs["val"], "val"),
        "eval": legacy.resolve_source_task_ids_from_inputs(fold_inputs, "train"),
    }
    allowed = {split: {t for ts in ids[split].values() for t in ts} for split in SPLITS}
    bundles = {
        "train": legacy.load_rad_bundle("train", allowed["train"]),
        "val": legacy.load_rad_bundle("val", allowed["val"]),
        "eval": legacy.load_rad_bundle("train", allowed["eval"]),
    }
    log(f"seed {seed} fold {fold}: training RAD-DINO head on {len(bundles['train'][0])} rows")
    legacy.train_rad_head(fold_root, seed, bundles, args.rad_epochs, 24, True)
    legacy.write_json(fold_root / "source_ids.json", ids)
    legacy.write_json(marker, {"status": "complete", "seed": seed, "fold": fold, "n_train_groups": len(train_inputs), "n_fold_groups": len(fold_inputs)})


def oof_table(seed: int, seed_root: Path, inputs: dict, labels: dict, assignment: dict[str, int], n_folds: int) -> pd.DataFrame:
    parts = []
    for fold in range(n_folds):
        fold_root = seed_root / f"fold_{fold}"
        fold_groups = {gid for gid, f in assignment.items() if f == fold}
        fold_inputs = [r for r in inputs["train"] if str(r["group_id"]) in fold_groups]
        rest_inputs = [r for r in inputs["train"] if str(r["group_id"]) not in fold_groups]
        rows = legacy.group_rows(fold_inputs, labels["train"], "train")
        rest_rows = legacy.group_rows(rest_inputs, labels["train"], "train")
        candidates = legacy.load_yolo_candidates(fold_root / PROTOCOL_KEY / "yolo_predictions", "eval")
        archive = np.load(fold_root / PROTOCOL_KEY / "rad_dino_legacy" / "predictions_by_split.npz", allow_pickle=True)
        task_boxes = {str(t): np.asarray(b, dtype=np.float32) for t, b in zip(archive["eval_ids"], archive["eval_boxes"])}
        ids = json.loads((fold_root / "source_ids.json").read_text(encoding="utf-8"))["eval"]
        dino = rr.group_boxes_from_task_norm(task_boxes, ids, rows)
        priors = ybase.make_train_priors(legacy.expanded_prior_rows(rest_rows))
        table = rr.build_table("train", rows, labels["train"], candidates, priors, {"dino": dino, "xattn": {}, "aux": {}})
        table["fold"] = fold
        parts.append(table)
        log(f"seed {seed} fold {fold}: OOF table rows={len(table)} groups={table.group_id.nunique()}")
    return pd.concat(parts, ignore_index=True)


def run_seed(seed: int, args: argparse.Namespace) -> dict[str, Any]:
    rr.NO_AUX_ASSETS = True
    rr.FEATURE_EXCLUDE = set(rr.FEATURE_EXCLUDE) | {"xattn_iou", "aux_iou"}
    seed_root = args.output_root / f"seed_{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    inputs, labels, source_ids = legacy.load_protocol(canonical.PROTOCOL_ROOT)
    labels_rows = {split: read_jsonl(canonical.PROTOCOL_ROOT / PROTOCOL_KEY / f"{split}_labels.jsonl") for split in SPLITS}
    assignment = fold_assignment(inputs["train"], args.folds, seed)
    legacy.write_json(seed_root / "fold_assignment.json", assignment)
    for fold in range(args.folds):
        fold_groups = {gid for gid, f in assignment.items() if f == fold}
        run_fold(seed, fold, seed_root / f"fold_{fold}", inputs, labels, labels_rows, fold_groups, args)

    table_path = seed_root / "candidates_train_oof.csv"
    if table_path.exists():
        train_table = pd.read_csv(table_path)
    else:
        train_table = oof_table(seed, seed_root, inputs, labels, assignment, args.folds)
        train_table.to_csv(table_path, index=False)
    sealed = SEALED_ROOT / f"seed_{seed}"
    tables = {"train": train_table, "val": pd.read_csv(sealed / "candidates_val.csv"), "eval": pd.read_csv(sealed / "candidates_eval.csv")}
    for split in ("val", "eval"):
        tables[split] = tables[split][[c for c in tables[split].columns if c in train_table.columns or c in rr.FEATURE_EXCLUDE]]

    best_name, cv_table, oof = rr.select_scorer(tables["train"], seed)
    cv_table.to_csv(seed_root / "scorer_cv_selection.csv", index=False)
    cols = rr.feature_columns(tables["train"])
    cols = [c for c in cols if c != "fold"]
    assert not any("target" in c or "gold" in c for c in cols), cols
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
    train_oof_top1 = rr.top1_iou(scored["train"], scored["train"]["learned"].to_numpy())
    train_conf_top1 = rr.top1_iou(scored["train"], scored["train"]["conf"].to_numpy())
    rr.write_json(seed_root / "scorer_selection.json", {"seed": seed, "selected_model": best_name, "cv": cv_table.to_dict("records"), "features": cols,
                                                         "val_top1_iou_learned": val_learned, "val_top1_iou_confidence": val_conf,
                                                         "train_oof_top1_iou_learned": train_oof_top1, "train_oof_top1_iou_confidence": train_conf_top1})
    log(f"seed {seed}: scorer={best_name}; val top-1 learned {val_learned:.4f} vs conf {val_conf:.4f}; train(OOF cands) learned {train_oof_top1:.4f} vs conf {train_conf_top1:.4f}")

    rows = {split: legacy.group_rows(inputs[split], labels[split], split) for split in SPLITS}
    upstream = canonical.UPSTREAM_ROOT / f"seed_{seed}" / canonical.PROTOCOL
    candidates = {split: legacy.load_yolo_candidates(upstream / "yolo_predictions", split) for split in SPLITS}
    context = exact.load_multi_context(seed, protocol_root=canonical.PROTOCOL_ROOT, multi_source_root=canonical.UPSTREAM_ROOT, canonical_v3=True)
    result = rr.run_variant(seed, "learned", args.output_root, context, candidates, scored, rows, labels, source_ids)
    row = {"seed": seed, "scorer": best_name, "alpha": result["learned_alpha"], "val_selected_coverage": result.get("val_selected_coverage")}
    for key in ("coverage_iou", "exact_union_iou", "set_f1_0_3", "set_f1_0_5", "mean_pred_count"):
        row[key] = float(result.get(key, float("nan")))
    log(f"seed {seed}: OOF re-ranker coverage {row['coverage_iou']:.4f} union {row['exact_union_iou']:.4f} f1@.5 {row['set_f1_0_5']:.4f}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--yolo-epochs", type=int, default=100)
    parser.add_argument("--rad-epochs", type=int, default=120)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        rows.append(run_seed(seed, args))
        pd.DataFrame(rows).to_csv(args.output_root / "per_seed_metrics.csv", index=False)
    frame = pd.DataFrame(rows)
    status = {"status": "complete", "seeds": args.seeds, "folds": args.folds}
    for key in ("coverage_iou", "exact_union_iou", "set_f1_0_3", "set_f1_0_5"):
        status[f"{key}_mean"] = float(frame[key].mean())
        status[f"{key}_std"] = float(frame[key].std(ddof=1)) if len(frame) > 1 else 0.0
    rr.write_json(args.output_root / "FINAL_STATUS.json", status)
    log(f"FINAL coverage {status['coverage_iou_mean']:.4f} +/- {status['coverage_iou_std']:.4f}")


if __name__ == "__main__":
    main()

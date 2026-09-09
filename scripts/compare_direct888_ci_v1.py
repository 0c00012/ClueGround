#!/usr/bin/env python
"""Paired, patient-clustered bootstrap comparison on direct-888 (163 phrases).

Arms (all rescored with the common evaluator, one evaluated box each):
  * clueground_reranker   sealed direct-888 re-ranker run
  * clueground_base       sealed hybrid-v4 (paper 0.5486 path)
  * medrpg, transvg       controlled Linux matrix runs (common_eval predictions)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_repro.evaluator import evaluate_rows, patient_cluster_bootstrap  # noqa: E402

SEEDS = (13, 42, 2026)
METRICS = ("mean_iou_row", "hit_0_3", "hit_0_5")
EVAL_CSV = PROJECT_ROOT / "experiments" / "ms_cxr_medrpg_fair_retrain_final_v1" / "data" / "single_box_full_phrase" / "eval.csv"
MATRIX = Path(r"C:/Users/_idal/Desktop/matrix/runs/singlebox_888")
RERANKER = PROJECT_ROOT / "experiments" / "clueground_direct888_learned_reranker_3seed_v1" / "learned" / "singlebox_888"
BASE = PROJECT_ROOT / "experiments" / "clueground_exact_hybrid_v4_route_specific_3seed_v3" / "singlebox_888"


def norm_text(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", str(text).lower()).split())


def samples() -> tuple[list[dict[str, Any]], dict[tuple[str, str], str]]:
    frame = pd.read_csv(EVAL_CSV)
    out, key_to_id = [], {}
    for r in frame.itertuples():
        out.append({"query_id": str(r.sample_id), "subject_id": str(r.subject_id), "finding": str(r.finding_label),
                    "gold_boxes": [[float(r.bbox_x1), float(r.bbox_y1), float(r.bbox_x2), float(r.bbox_y2)]]})
        key_to_id[(str(r.dicom_id), norm_text(r.phrase_text))] = str(r.sample_id)
    assert len(out) == 163
    return out, key_to_id


def load_jsonl_predictions(path: Path) -> dict[str, list[list[float]]]:
    out = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                r = json.loads(line)
                out[str(r["group_id"])] = [[float(v) for v in b] for b in r["pred_boxes_xyxy"]]
    return out


def load_matrix_predictions(model: str, seed: int, key_to_id: dict) -> dict[str, list[list[float]]]:
    frame = pd.read_csv(MATRIX / model / f"seed_{seed}" / "common_eval" / "predictions.csv")
    out = {}
    for r in frame.itertuples():
        dicom, phrase = str(r.group_id).split("|", 1)
        out[key_to_id[(dicom, norm_text(phrase))]] = json.loads(r.pred_boxes_json)
    assert len(out) == 163, len(out)
    return out


def score(method: str, seed: int, predictions: dict, samp: list[dict[str, Any]]) -> pd.DataFrame:
    summary, detail = evaluate_rows(samp, [{"query_id": k, "pred_boxes": v} for k, v in predictions.items()], force_single_box=True)
    frame = pd.DataFrame(detail)
    frame["method"], frame["seed"] = method, seed
    frame["mean_iou_row"] = frame["top1_iou"]
    frame["hit_0_3"] = (frame["top1_iou"] >= 0.3).astype(float)
    frame["hit_0_5"] = (frame["top1_iou"] >= 0.5).astype(float)
    return frame


def paired(left: pd.DataFrame, right: pd.DataFrame, reps: int, seed: int) -> list[dict[str, Any]]:
    merged = left.merge(right, on="query_id", suffixes=("_a", "_b"))
    clusters = sorted(merged["subject_id_a"].astype(str).unique())
    member = merged["subject_id_a"].astype(str).map({c: i for i, c in enumerate(clusters)}).to_numpy()
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(clusters), size=(reps, len(clusters)))
    rows = []
    for metric in METRICS:
        diff = merged[f"{metric}_a"].to_numpy(float) - merged[f"{metric}_b"].to_numpy(float)
        sums = np.bincount(member, weights=diff, minlength=len(clusters))
        counts = np.bincount(member, minlength=len(clusters)).astype(float)
        boot = sums[choices].sum(axis=1) / counts[choices].sum(axis=1)
        rows.append({"metric": metric, "mean_difference": float(diff.mean()), "ci_95_low": float(np.quantile(boot, 0.025)),
                     "ci_95_high": float(np.quantile(boot, 0.975)), "p_bootstrap_two_sided": float(2 * min((boot <= 0).mean(), (boot >= 0).mean()))})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=RERANKER.parent.parent / "comparison_ci")
    parser.add_argument("--reps", type=int, default=2000)
    args = parser.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    samp, key_to_id = samples()
    details = []
    for seed in SEEDS:
        details.append(score("clueground_reranker", seed, load_jsonl_predictions(RERANKER / f"seed_{seed}" / "eval_predictions.jsonl"), samp))
        details.append(score("clueground_base", seed, load_jsonl_predictions(BASE / f"seed_{seed}" / "eval_predictions.jsonl"), samp))
        for model in ("medrpg", "transvg"):
            details.append(score(model, seed, load_matrix_predictions(model, seed, key_to_id), samp))
    detail = pd.concat(details, ignore_index=True)
    detail.to_csv(args.out_root / "per_row_detail.csv", index=False)
    ci = patient_cluster_bootstrap(detail, identity_columns=["method", "seed"], metric_columns=list(METRICS), cluster_column="subject_id", reps=args.reps)
    ci.to_csv(args.out_root / "per_seed_ci.csv", index=False)
    agg = detail.groupby(["method", "seed"])[list(METRICS)].mean().reset_index().groupby("method")[list(METRICS)].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(args.out_root / "seed_aggregate.csv", index=False)
    paired_rows = []
    for method in ("clueground_reranker", "clueground_base"):
        for ref in ("medrpg", "transvg"):
            for seed in SEEDS:
                left = detail[(detail.method == method) & (detail.seed == seed)]
                right = detail[(detail.method == ref) & (detail.seed == seed)]
                for row in paired(left, right, args.reps, 20260909 + seed):
                    paired_rows.append({"method": method, "vs": ref, "seed": seed, **row})
        for seed in SEEDS:
            left = detail[(detail.method == "clueground_reranker") & (detail.seed == seed)]
            right = detail[(detail.method == "clueground_base") & (detail.seed == seed)]
            for row in paired(left, right, args.reps, 20260909 + seed):
                paired_rows.append({"method": "clueground_reranker", "vs": "clueground_base", "seed": seed, **row})
    paired_frame = pd.DataFrame(paired_rows).drop_duplicates()
    paired_frame.to_csv(args.out_root / "paired_difference_ci.csv", index=False)
    lines = ["# direct-888 paired comparison (163 phrases, common evaluator, subject-cluster bootstrap)", "",
             "| method | mean IoU | Hit@0.3 | Hit@0.5 |", "|---|---:|---:|---:|"]
    for r in agg.itertuples():
        lines.append(f"| {r.method} | {r.mean_iou_row_mean:.4f} +/- {r.mean_iou_row_std:.4f} | {r.hit_0_3_mean:.4f} | {r.hit_0_5_mean:.4f} |")
    lines += ["", "| method | vs | seed | metric | mean diff | CI low | CI high | p |", "|---|---|---:|---|---:|---:|---:|---:|"]
    for r in paired_frame.sort_values(["method", "vs", "metric", "seed"]).itertuples():
        lines.append(f"| {r.method} | {r.vs} | {r.seed} | {r.metric} | {r.mean_difference:+.4f} | {r.ci_95_low:+.4f} | {r.ci_95_high:+.4f} | {r.p_bootstrap_two_sided:.3f} |")
    (args.out_root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

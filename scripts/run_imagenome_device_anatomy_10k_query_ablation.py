#!/usr/bin/env python
"""Add no-query and label-only ablations for the Chest ImaGenome 10k split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_imagenome_device_anatomy_10k_rule_smm_v1 as exp


def build_ablation_query_features(force: bool = False) -> None:
    exp.ensure_dirs()
    for task in ["device", "anatomy"]:
        labels = sorted({r["finding"] for split in exp.SPLITS for r in exp.load_rows(task, split)})
        label_to_i = {label: i for i, label in enumerate(labels)}
        for split in exp.SPLITS:
            rows = exp.load_rows(task, split)
            ids = np.array([r["task_id"] for r in rows], dtype=object)
            no_path = exp.FEAT / f"{task}_no_query_query_{split}.npz"
            label_path = exp.FEAT / f"{task}_label_only_query_{split}.npz"
            if force or not no_path.exists():
                np.savez(
                    no_path,
                    task_ids=ids,
                    features=np.zeros((len(rows), 0), dtype="float32"),
                    query_text=np.array(["<no_query>" for _ in rows], dtype=object),
                )
            if force or not label_path.exists():
                feats = np.zeros((len(rows), len(labels)), dtype="float32")
                for i, r in enumerate(rows):
                    feats[i, label_to_i[r["finding"]]] = 1.0
                np.savez(
                    label_path,
                    task_ids=ids,
                    features=feats,
                    query_text=np.array([f"label={r['finding']}" for r in rows], dtype=object),
                )
    exp.write_text(
        exp.REPORT / "STAGE8_QUERY_ABLATION_FEATURES.md",
        "# Query Ablation Features\n\n"
        "- `no_query`: RAD-DINO patch tokens only; no task/query vector.\n"
        "- `label_only`: one-hot target label only; no parsed rule slots.\n"
        "- These ablations use the same train/val/eval split and the same frozen RAD-DINO features as the rule/SMM comparison.\n",
    )


def train_ablation(args) -> None:
    infos = []
    for task in ["device", "anatomy"]:
        for query_type in ["no_query", "label_only"]:
            infos.append(exp.train_heatmap(task, query_type, args))
    prev = pd.read_csv(exp.MET / "training_runs.csv") if (exp.MET / "training_runs.csv").exists() else pd.DataFrame()
    add = pd.DataFrame(infos)
    pd.concat([prev, add], ignore_index=True).drop_duplicates("method", keep="last").to_csv(exp.MET / "training_runs.csv", index=False)
    exp.write_text(
        exp.REPORT / "STAGE9_QUERY_ABLATION_TRAINING.md",
        "# Query Ablation Training\n\n"
        + add.to_markdown(index=False)
        + "\n",
    )


def write_report() -> None:
    result = exp.evaluate_all()
    summary = pd.read_csv(exp.MET / "summary.csv")
    keep = summary[
        summary["method"].isin(
            [
                "device_no_query_query_heatmap",
                "device_label_only_query_heatmap",
                "device_rule_query_heatmap",
                "device_smm_query_heatmap",
                "anatomy_no_query_query_heatmap",
                "anatomy_label_only_query_heatmap",
                "anatomy_rule_query_heatmap",
                "anatomy_smm_query_heatmap",
            ]
        )
    ].copy()
    rows = []
    for task in ["device", "anatomy"]:
        sdf = keep[keep["subset"].eq(task)].set_index("method")
        rule = float(sdf.loc[f"{task}_rule_query_heatmap", "mean_iou"]) if f"{task}_rule_query_heatmap" in sdf.index else float("nan")
        no = float(sdf.loc[f"{task}_no_query_query_heatmap", "mean_iou"]) if f"{task}_no_query_query_heatmap" in sdf.index else float("nan")
        lab = float(sdf.loc[f"{task}_label_only_query_heatmap", "mean_iou"]) if f"{task}_label_only_query_heatmap" in sdf.index else float("nan")
        rows.append(
            {
                "task": task,
                "no_query_iou": no,
                "label_only_iou": lab,
                "rule_iou": rule,
                "rule_minus_no_query": rule - no,
                "rule_minus_label_only": rule - lab,
            }
        )
    ab = pd.DataFrame(rows)
    ab.to_csv(exp.MET / "query_ablation_comparison.csv", index=False)
    exp.write_text(
        exp.REPORT / "QUERY_ABLATION_RESULT.md",
        "# Query Ablation Result\n\n"
        "This checks whether the rulebase really helps beyond frozen RAD-DINO features alone.\n\n"
        "## Summary\n\n"
        + keep.sort_values(["subset", "mean_iou"], ascending=[True, False]).to_markdown(index=False)
        + "\n\n## Rule Gain\n\n"
        + ab.to_markdown(index=False)
        + "\n",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    p.add_argument("--build-features", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--heatmap-batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=28)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--heatmap-loss-weight", type=float, default=0.2)
    p.add_argument("--vfm-model", default="microsoft/rad-dino")
    args = p.parse_args()
    if not any([args.build_features, args.train, args.evaluate]):
        args.build_features = args.train = args.evaluate = True
    return args


def main() -> None:
    args = parse_args()
    if args.build_features:
        build_ablation_query_features(args.force)
    if args.train:
        train_ablation(args)
    if args.evaluate:
        write_report()
    q = pd.read_csv(exp.MET / "query_ablation_comparison.csv")
    print(f"project_root={exp.PROJECT_ROOT}")
    print(f"report_path={exp.REPORT / 'QUERY_ABLATION_RESULT.md'}")
    print(q.to_string(index=False))


if __name__ == "__main__":
    main()

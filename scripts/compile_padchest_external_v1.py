#!/usr/bin/env python
"""Collect the PadChest-GR zero-shot numbers for the manuscript.

Reads the outputs of ``padchest/run_clueground_padchest_gr_zero_shot.py``
(strict and extended label mappings), adds seed-wise paired differences
(re-ranker minus no-re-ranker, re-ranker minus RAD-DINO-only) with
patient-cluster bootstrap intervals, and writes
``experiments/padchest_gr_external_v1/PADCHEST_NUMBERS.{json,md}``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP = PROJECT_ROOT / "experiments"
METRICS = ("coverage_iou", "exact_union_iou", "set_f1_optimal_0_3", "set_f1_optimal_0_5")
METHODS = ("clueground_reranker", "hybrid_no_reranker", "rad_dino_only")


def paired(left: pd.DataFrame, right: pd.DataFrame, reps: int, seed: int) -> list[dict]:
    merged = left.merge(right, on="query_id", suffixes=("_a", "_b"))
    clusters = sorted(merged["subject_id_a"].astype(str).unique())
    member = merged["subject_id_a"].astype(str).map({c: i for i, c in enumerate(clusters)}).to_numpy()
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(clusters), size=(reps, len(clusters)))
    counts = np.bincount(member, minlength=len(clusters)).astype(float)
    rows = []
    for metric in METRICS:
        diff = merged[f"{metric}_a"].to_numpy(float) - merged[f"{metric}_b"].to_numpy(float)
        sums = np.bincount(member, weights=diff, minlength=len(clusters))
        boot = sums[choices].sum(axis=1) / counts[choices].sum(axis=1)
        rows.append({"metric": metric, "mean_difference": float(diff.mean()), "ci_95_low": float(np.quantile(boot, 0.025)), "ci_95_high": float(np.quantile(boot, 0.975)),
                     "p_bootstrap_two_sided": float(2 * min((boot <= 0).mean(), (boot >= 0).mean()))})
    return rows


def collect(root: Path, mapping: str, reps: int) -> dict:
    agg = pd.read_csv(root / "seed_aggregate.csv")
    per_finding = pd.read_csv(root / "per_finding_aggregate.csv")
    seeds = sorted(int(p.name.split("_")[1]) for p in root.glob("seed_*") if (p / "SEED_RESULT.json").exists())
    out = {"mapping": mapping, "root": str(root), "seeds": seeds, "aggregate": {}, "per_finding": {}, "per_reference_count": {}, "paired": {}, "ci": {}}
    for method in METHODS:
        out["aggregate"][method] = {m: {"mean": float(agg[(agg.method == method) & (agg.metric == m)]["mean"].iloc[0]), "std": float(agg[(agg.method == method) & (agg.metric == m)]["std"].iloc[0])} for m in METRICS}
    for r in per_finding[per_finding.method == "clueground_reranker"].itertuples():
        target = out["per_finding"] if r.group == "per_finding" else out["per_reference_count"]
        target[str(r.name)] = {"n": int(r.n), **{m: float(getattr(r, m)) for m in METRICS}}
    details = pd.concat([pd.read_csv(root / f"seed_{s}" / "per_row_detail.csv") for s in seeds], ignore_index=True)
    out["n_rows"] = int(details[(details.method == "clueground_reranker") & (details.seed == seeds[0])].shape[0])
    for ref in ("hybrid_no_reranker", "rad_dino_only"):
        out["paired"][f"clueground_reranker_minus_{ref}"] = {}
        for s in seeds:
            left = details[(details.method == "clueground_reranker") & (details.seed == s)]
            right = details[(details.method == ref) & (details.seed == s)]
            out["paired"][f"clueground_reranker_minus_{ref}"][str(s)] = paired(left, right, reps, 20260910 + s)
    for s in seeds:
        ci = pd.read_csv(root / f"seed_{s}" / "per_seed_ci.csv")
        out["ci"][str(s)] = {f"{r.method}:{r.metric}": {"point": float(r.point_estimate), "low": float(r.ci_95_low), "high": float(r.ci_95_high)} for r in ci.itertuples()}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-root", type=Path, default=EXP / "padchest_gr_external_v1" / "results_strict")
    parser.add_argument("--extended-root", type=Path, default=EXP / "pgx_ext")
    parser.add_argument("--out-root", type=Path, default=EXP / "padchest_gr_external_v1")
    parser.add_argument("--reps", type=int, default=2000)
    args = parser.parse_args()
    numbers = {"strict": collect(args.strict_root, "strict", args.reps), "extended": collect(args.extended_root, "extended", args.reps)}
    (args.out_root / "PADCHEST_NUMBERS.json").write_text(json.dumps(numbers, indent=2), encoding="utf-8")
    lines = ["# PadChest-GR zero-shot numbers", ""]
    for mapping, block in numbers.items():
        lines.append(f"## {mapping} ({block['n_rows']} rows, seeds {block['seeds']})")
        for method, vals in block["aggregate"].items():
            lines.append(f"- {method}: " + "; ".join(f"{m} {v['mean']:.4f} +/- {v['std']:.4f}" for m, v in vals.items()))
        for key, per_seed in block["paired"].items():
            for s, rows in per_seed.items():
                lines.append(f"- {key} seed {s}: " + "; ".join(f"{r['metric']} {r['mean_difference']:+.4f} [{r['ci_95_low']:+.4f}, {r['ci_95_high']:+.4f}]" for r in rows))
        lines.append("")
    (args.out_root / "PADCHEST_NUMBERS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

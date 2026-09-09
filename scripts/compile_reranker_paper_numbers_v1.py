#!/usr/bin/env python
"""Collect every number the revised manuscript needs from sealed artifacts.

Writes ``experiments/clueground_reranker_paper_numbers_v1/PAPER_NUMBERS.json``
and a Markdown rendering, so the LaTeX edit is a transcription of one audited
file rather than of scattered logs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP = PROJECT_ROOT / "experiments"
OUT = EXP / "clueground_reranker_paper_numbers_v1"


def ms(frame: pd.DataFrame, col: str) -> dict[str, float]:
    return {"mean": float(frame[col].mean()), "std": float(frame[col].std(ddof=1)), "per_seed": [float(v) for v in frame[col]]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    numbers: dict[str, Any] = {}

    # --- 1444 main: sealed no-aux re-ranker (val alpha) ---
    m = pd.read_csv(EXP / "clueground_canonical_learned_reranker_noaux_assets_v1" / "per_seed_metrics.csv")
    numbers["clueground_1444"] = {k: ms(m, c) for k, c in [("coverage", "coverage_iou"), ("union", "exact_union_iou"), ("f1_0_3", "set_f1_0_3"), ("f1_0_5", "set_f1_0_5"), ("count", "mean_pred_count")]}
    numbers["clueground_1444"]["alpha_per_seed"] = [float(v) for v in m["alpha"]]
    numbers["clueground_1444"]["scorer_per_seed"] = [str(v) for v in m["scorer"]]
    numbers["clueground_1444"]["source"] = "experiments/clueground_canonical_learned_reranker_noaux_assets_v1"

    # --- 888 main: sealed re-ranker (val alpha) ---
    m8 = pd.read_csv(EXP / "clueground_direct888_learned_reranker_3seed_v1" / "per_seed_metrics.csv")
    numbers["clueground_888"] = {k: ms(m8, c) for k, c in [("miou", "mean_iou"), ("hit_0_3", "Hit@0.3"), ("hit_0_5", "Hit@0.5")]}
    numbers["clueground_888"]["alpha_per_seed"] = [float(v) for v in m8["alpha"]]
    numbers["clueground_888"]["scorer_per_seed"] = [str(v) for v in m8["scorer"]]
    numbers["clueground_888"]["source"] = "experiments/clueground_direct888_learned_reranker_3seed_v1"

    # --- baselines (unchanged, verified refresh) ---
    b14 = pd.read_csv(EXP / "baseline_table_verified_refresh" / "20260724_v1" / "controlled_multibox_1444.csv")
    b88 = pd.read_csv(EXP / "baseline_table_verified_refresh" / "20260724_v1" / "controlled_singlebox_888.csv")
    numbers["baselines_1444"] = {r.method: {"coverage_mean": float(r.coverage_iou_mean), "coverage_std": float(r.coverage_iou_std), "union_mean": float(r.exact_union_iou_mean), "union_std": float(r.exact_union_iou_std), "f1_0_3": float(r.set_f1_optimal_0_3_mean), "f1_0_5": float(r.set_f1_optimal_0_5_mean)} for r in b14.itertuples()}
    numbers["baselines_888"] = {r.method: {"miou_mean": float(r.mean_iou_mean), "miou_std": float(r.mean_iou_std), "hit_0_3": float(getattr(r, "_10")) if False else float(b88.loc[b88.method == r.method, "Hit@0.3_mean"].iloc[0]), "hit_0_5": float(b88.loc[b88.method == r.method, "Hit@0.5_mean"].iloc[0])} for r in b88.itertuples()}

    # --- ablations (new path) ---
    a14 = pd.read_csv(EXP / "clueground_reranker_ablations_v1" / "aggregate.csv")
    numbers["ablation_1444"] = {f"{r.table}:{r.variant}": {"coverage_mean": float(r.coverage_iou_mean), "coverage_std": float(r.coverage_iou_std), "union_mean": float(r.exact_union_iou_mean), "f1_0_3": float(r.set_f1_optimal_0_3_mean), "f1_0_5": float(r.set_f1_optimal_0_5_mean)} for r in a14.itertuples()}
    a88 = pd.read_csv(EXP / "clueground_reranker_ablations_888_v1" / "aggregate.csv")
    numbers["ablation_888"] = {f"{r.table}:{r.variant}": {"miou_mean": float(r.mean_iou_mean), "miou_std": float(r.mean_iou_std), "hit_0_3": float(getattr(r, "_5")), "hit_0_5": float(getattr(r, "_7"))} for r in a88.itertuples()}
    # base (no re-ranker) rows come from the sealed hybrid runs
    numbers["ablation_1444"]["component:no_reranker"] = numbers["baselines_1444"]["Ours YOLO-RAD-DINO hybrid-v4"]
    numbers["ablation_888"]["component:no_reranker"] = numbers["baselines_888"]["Ours YOLO-RAD-DINO hybrid-v4"]

    # --- robustness rows ---
    for key, folder in [("no_retune", "clueground_canonical_learned_reranker_noaux_noretune_v1"), ("multi_alpha_zero", "clueground_canonical_learned_reranker_noaux_multialpha0_v1"), ("pooled_alpha", "clueground_canonical_learned_reranker_noaux_pooledalpha_v1"), ("aux_features", "clueground_canonical_learned_reranker_3seed_v1")]:
        f = pd.read_csv(EXP / folder / "per_seed_metrics.csv")
        f = f[f.variant == "learned"] if "variant" in f.columns else f
        numbers.setdefault("robustness_1444", {})[key] = {"coverage": ms(f, "coverage_iou"), "f1_0_5": ms(f, "set_f1_0_5")}
    f = pd.read_csv(EXP / "clueground_direct888_learned_reranker_pooledalpha_v1" / "per_seed_metrics.csv")
    numbers["robustness_888"] = {"pooled_alpha": {"miou": ms(f, "mean_iou")}}
    oof = EXP / "clueground_reranker_oof_v1" / "per_seed_metrics.csv"
    if oof.exists():
        f = pd.read_csv(oof)
        numbers["robustness_1444"]["oof_candidates"] = {"coverage": ms(f, "coverage_iou"), "f1_0_5": ms(f, "set_f1_0_5"), "n_seeds": int(len(f))}

    # --- paired CI vs MedGrounder (1444) and vs MedRPG/TransVG (888) ---
    p14 = pd.read_csv(EXP / "clueground_canonical_learned_reranker_noaux_v1" / "comparison_final" / "paired_difference_ci.csv")
    p14 = p14[p14.method == "reranker_noaux_valalpha"]
    numbers["paired_vs_medgrounder_1444"] = {metric: [{"seed": int(r.seed), "diff": float(r.mean_difference), "ci_low": float(r.ci_95_low), "ci_high": float(r.ci_95_high), "p": float(r.p_bootstrap_two_sided)} for r in p14[p14.metric == metric].itertuples()] for metric in p14.metric.unique()}
    ci14 = pd.read_csv(EXP / "clueground_canonical_learned_reranker_noaux_v1" / "comparison_final" / "per_seed_ci.csv")
    numbers["ci_1444"] = {f"{r.method}:{r.seed}:{r.metric}": {"point": float(r.point_estimate), "low": float(r.ci_95_low), "high": float(r.ci_95_high)} for r in ci14.itertuples() if r.method in ("reranker_noaux_valalpha", "medgrounder")}
    p88 = pd.read_csv(EXP / "clueground_direct888_learned_reranker_3seed_v1" / "comparison_ci" / "paired_difference_ci.csv")
    numbers["paired_888"] = [{"method": r.method, "vs": r.vs, "seed": int(r.seed), "metric": r.metric, "diff": float(r.mean_difference), "ci_low": float(r.ci_95_low), "ci_high": float(r.ci_95_high)} for r in p88.itertuples()]

    (OUT / "PAPER_NUMBERS.json").write_text(json.dumps(numbers, indent=2), encoding="utf-8")

    def f4(x: float) -> str:
        return f"{x:.4f}"

    lines = ["# Paper numbers (sealed)", ""]
    c = numbers["clueground_1444"]
    lines.append(f"1444 ClueGround: C-IoU {f4(c['coverage']['mean'])} +/- {f4(c['coverage']['std'])}; U-IoU {f4(c['union']['mean'])} +/- {f4(c['union']['std'])}; F1@.3 {f4(c['f1_0_3']['mean'])}; F1@.5 {f4(c['f1_0_5']['mean'])}; per-seed coverage {c['coverage']['per_seed']}; alpha {c['alpha_per_seed']}; scorer {c['scorer_per_seed']}")
    c = numbers["clueground_888"]
    lines.append(f"888 ClueGround: mIoU {f4(c['miou']['mean'])} +/- {f4(c['miou']['std'])}; Hit@.3 {f4(c['hit_0_3']['mean'])}; Hit@.5 {f4(c['hit_0_5']['mean'])}; per-seed {c['miou']['per_seed']}; alpha {c['alpha_per_seed']}; scorer {c['scorer_per_seed']}")
    lines.append("")
    lines.append("## 1444 ablations (new path)")
    for k, v in numbers["ablation_1444"].items():
        lines.append(f"- {k}: C {f4(v['coverage_mean'])} +/- {f4(v['coverage_std'])}, U {f4(v['union_mean'])}, F1@.3 {f4(v['f1_0_3'])}, F1@.5 {f4(v['f1_0_5'])}")
    lines.append("## 888 ablations (new path)")
    for k, v in numbers["ablation_888"].items():
        lines.append(f"- {k}: mIoU {f4(v['miou_mean'])} +/- {f4(v['miou_std'])}, Hit@.3 {f4(v['hit_0_3'])}, Hit@.5 {f4(v['hit_0_5'])}")
    lines.append("## robustness 1444")
    for k, v in numbers["robustness_1444"].items():
        lines.append(f"- {k}: C {f4(v['coverage']['mean'])} +/- {f4(v['coverage']['std'])}, F1@.5 {f4(v['f1_0_5']['mean'])}")
    lines.append("## paired vs MedGrounder (1444)")
    for metric, rows in numbers["paired_vs_medgrounder_1444"].items():
        lines.append(f"- {metric}: " + "; ".join(f"seed {r['seed']} {r['diff']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]" for r in rows))
    (OUT / "PAPER_NUMBERS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Per-finding and per-reference-count breakdown of the sealed re-ranker results.

No new inference: the row-level scoring files written by the paired-comparison
scripts (common evaluator, sealed predictions) are regrouped by MS-CXR finding
and by the number of reference boxes.  Means are means of per-seed means
(seeds 13/42/2026); ``n`` is the number of evaluation rows per seed.

Outputs ``experiments/clueground_reranker_per_finding_v1/{per_finding_1444,
per_finding_888,per_refcount_1444}.csv`` and ``PER_FINDING.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP = PROJECT_ROOT / "experiments"
OUT = EXP / "clueground_reranker_per_finding_v1"
D1444 = EXP / "clueground_canonical_learned_reranker_noaux_v1" / "comparison_final" / "per_group_detail.csv"
D888 = EXP / "clueground_direct888_learned_reranker_3seed_v1" / "comparison_ci" / "per_row_detail.csv"
METHODS_1444 = {"reranker_noaux_valalpha": "ClueGround", "medgrounder": "MedGrounder"}
METHODS_888 = {"clueground_reranker": "ClueGround", "medrpg": "MedRPG", "transvg": "TransVG"}
FINDINGS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Lung Opacity", "Pleural Effusion", "Pneumonia", "Pneumothorax"]


def breakdown(frame: pd.DataFrame, methods: dict[str, str], key: str, metrics: list[str]) -> pd.DataFrame:
    frame = frame[frame.method.isin(methods)].copy()
    frame["method"] = frame["method"].map(methods)
    per_seed = frame.groupby(["method", "seed", key])[metrics].mean().reset_index()
    n = frame.groupby(["method", "seed", key]).size().rename("n").reset_index()
    per_seed = per_seed.merge(n, on=["method", "seed", key])
    agg = per_seed.groupby(["method", key]).agg({**{m: ["mean", "std"] for m in metrics}, "n": "first"})
    agg.columns = ["_".join(c) if c[1] != "first" else c[0] for c in agg.columns]
    agg = agg.reset_index()
    # overall row
    overall = frame.groupby(["method", "seed"])[metrics].mean().reset_index()
    overall_n = frame.groupby(["method", "seed"]).size().rename("n").reset_index()
    overall = overall.merge(overall_n, on=["method", "seed"]).groupby("method").agg({**{m: ["mean", "std"] for m in metrics}, "n": "first"})
    overall.columns = ["_".join(c) if c[1] != "first" else c[0] for c in overall.columns]
    overall = overall.reset_index()
    overall[key] = "All"
    return pd.concat([agg, overall[agg.columns]], ignore_index=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d14 = pd.read_csv(D1444)
    d88 = pd.read_csv(D888)
    assert set(d14[d14.method == "medgrounder"].query_id) == set(d14[d14.method == "reranker_noaux_valalpha"].query_id)

    f14 = breakdown(d14, METHODS_1444, "finding", ["coverage_iou", "exact_union_iou", "set_f1_optimal_0_5"])
    f14.to_csv(OUT / "per_finding_1444.csv", index=False)

    d14["ref_count"] = d14["n_gt"].map(lambda k: "single (1 box)" if k == 1 else "multi (2+ boxes)")
    r14 = breakdown(d14, METHODS_1444, "ref_count", ["coverage_iou", "exact_union_iou", "set_f1_optimal_0_5"])
    r14.to_csv(OUT / "per_refcount_1444.csv", index=False)

    f88 = breakdown(d88, METHODS_888, "finding", ["mean_iou_row", "hit_0_5"])
    f88.to_csv(OUT / "per_finding_888.csv", index=False)

    def rows(frame: pd.DataFrame, key: str) -> list[dict]:
        return [{k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else (v if isinstance(v, str) else v.item())) for k, v in r.items()} for r in frame.to_dict("records")]

    (OUT / "PER_FINDING.json").write_text(json.dumps({"per_finding_1444": rows(f14, "finding"), "per_refcount_1444": rows(r14, "ref_count"), "per_finding_888": rows(f88, "finding")}, indent=2, default=float), encoding="utf-8")

    pd.set_option("display.width", 200)
    print("== 1444 per finding (coverage / F1@.5)")
    piv = f14.pivot(index="finding", columns="method", values=["n", "coverage_iou_mean", "set_f1_optimal_0_5_mean"]).round(4)
    print(piv.reindex(FINDINGS + ["All"]))
    print("== 1444 per reference count")
    print(r14.pivot(index="ref_count", columns="method", values=["n", "coverage_iou_mean", "set_f1_optimal_0_5_mean"]).round(4))
    print("== 888 per finding (mIoU / Hit@.5)")
    print(f88.pivot(index="finding", columns="method", values=["n", "mean_iou_row_mean", "hit_0_5_mean"]).round(4))


if __name__ == "__main__":
    main()

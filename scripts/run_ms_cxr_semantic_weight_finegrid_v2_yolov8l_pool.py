#!/usr/bin/env python
"""Fine semantic blend search for the YOLOv8l-pool hybrid.

This reuses the v3 YOLO-DINO hybrid inputs and tunes only the convex blend
between hybrid, SigLIP crop-text, and BioMedCLIP crop-text boxes on validation.
Eval boxes are used once after the validation-selected weights are fixed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_ms_cxr_multibox_phrase_grounding_v1 as mb  # noqa: E402
from scripts import run_ms_cxr_semantic_ensemble_gated_hybrid_v2_yolov8m_pool as sem2  # noqa: E402
from scripts import run_ms_cxr_semantic_weight_grid_v3_yolov8l_pool as base  # noqa: E402


EXP_NAME = "ms_cxr_semantic_weight_finegrid_v2_yolov8l_pool"
EXP = PROJECT_ROOT / "experiments" / EXP_NAME
PRED = EXP / "predictions"
MET = EXP / "metrics"
CFG = EXP / "configs"
REPORT = PROJECT_ROOT / "reports" / EXP_NAME


def ensure_dirs() -> None:
    for p in [EXP, PRED, MET, CFG, REPORT]:
        p.mkdir(parents=True, exist_ok=True)


def focused_weights() -> list[tuple[float, float, float]]:
    vals_h = [round(x, 4) for x in np.arange(0.425, 0.5751, 0.0125)]
    vals_s = [round(x, 4) for x in np.arange(0.0, 0.1251, 0.0125)]
    out: list[tuple[float, float, float]] = []
    for wh in vals_h:
        for ws in vals_s:
            wb = round(1.0 - wh - ws, 4)
            if wb < -1e-8:
                continue
            out.append((wh, ws, max(wb, 0.0)))
    return out


def tune() -> tuple[dict[str, Any], pd.DataFrame]:
    groups, hybrid, siglip, biomed, cue = base.build_inputs("val")
    rows: list[dict[str, Any]] = []
    best_key = None
    best = None
    for keep_multi in [True]:
        for min_iou in [0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2]:
            for weights in focused_weights():
                preds, audit = base.predict(groups, hybrid, siglip, biomed, cue, weights, keep_multi, min_iou)
                d = base.detail("val_finegrid", groups, preds)
                s = mb.summarize(d.to_dict("records"), "val_finegrid", "val_all")
                row = {
                    "keep_multi": keep_multi,
                    "min_hybrid_semantic_iou": min_iou,
                    "w_hybrid": weights[0],
                    "w_siglip": weights[1],
                    "w_biomed": weights[2],
                    "n_changed": int((audit["action"] == "blend").sum()),
                    **s,
                }
                rows.append(row)
                key = (float(s["coverage_mean_iou"]), float(s["gt_hit_rate_0_5"]), float(s["set_f1_0_3"]))
                if best_key is None or key > best_key:
                    best_key = key
                    best = row
    grid = pd.DataFrame(rows).sort_values(["coverage_mean_iou", "gt_hit_rate_0_5", "set_f1_0_3"], ascending=False)
    assert best is not None
    return best, grid


def bootstrap(target_detail: pd.DataFrame, refs: pd.DataFrame) -> pd.DataFrame:
    target = "semantic_weight_finegrid_v2_yolov8l_pool"
    methods = [
        "semantic_weight_grid_v3_yolov8l_pool",
        "semantic_weight_grid_v2_yolov8m_pool",
        "hybrid_singlebox_fusion_plus_context_multibox_v4_yolov8l_pool",
        "semantic_ensemble_gated_hybrid_v2_yolov8m_pool",
        "medrpg_rowlevel_full_phrase_s2026",
        "medrpg_rowlevel_full_phrase_s13",
        "medrpg_rowlevel_full_phrase_s42",
    ]
    all_df = pd.concat([target_detail, refs[refs["method"].isin(methods)]], ignore_index=True, sort=False)
    rng = np.random.default_rng(20260705)
    rows = []
    for b in methods:
        piv = all_df[all_df["method"].isin([target, b])].pivot(index="group_id", columns="method", values="coverage_mean_iou")
        if target not in piv.columns or b not in piv.columns:
            continue
        piv = piv.dropna()
        diff = (piv[target] - piv[b]).to_numpy()
        boots = [float(diff[rng.integers(0, len(diff), len(diff))].mean()) for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({
            "method_a": target,
            "method_b": b,
            "n_groups": int(len(diff)),
            "mean_diff": float(diff.mean()),
            "ci95_low": float(lo),
            "ci95_high": float(hi),
            "p_diff_le_0": float((np.asarray(boots) <= 0).mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    best, grid = tune()
    grid.to_csv(MET / "finegrid_val.csv", index=False)
    best_params = {
        "keep_multi": bool(best["keep_multi"]),
        "min_hybrid_semantic_iou": float(best["min_hybrid_semantic_iou"]),
        "w_hybrid": float(best["w_hybrid"]),
        "w_siglip": float(best["w_siglip"]),
        "w_biomed": float(best["w_biomed"]),
        "n_changed_val": int(best["n_changed"]),
    }
    (CFG / "best_finegrid_params.json").write_text(json.dumps(best_params, ensure_ascii=False, indent=2), encoding="utf-8")

    groups, hybrid, siglip, biomed, cue = base.build_inputs("eval")
    preds, audit = base.predict(
        groups,
        hybrid,
        siglip,
        biomed,
        cue,
        (best_params["w_hybrid"], best_params["w_siglip"], best_params["w_biomed"]),
        best_params["keep_multi"],
        best_params["min_hybrid_semantic_iou"],
    )
    method = "semantic_weight_finegrid_v2_yolov8l_pool"
    d = base.detail(method, groups, preds)
    s = base.summary(method, d)
    d.to_csv(PRED / "semantic_weight_finegrid_phrase_group_predictions.csv", index=False)
    audit.to_csv(PRED / "semantic_weight_finegrid_action_audit.csv", index=False)
    s.to_csv(MET / "semantic_weight_finegrid_summary.csv", index=False)

    refs = []
    ref_paths = [
        PROJECT_ROOT / "experiments" / "ms_cxr_semantic_weight_grid_v3_yolov8l_pool" / "predictions" / "semantic_weight_grid_phrase_group_predictions.csv",
        PROJECT_ROOT / "experiments" / "ms_cxr_semantic_weight_grid_v2_yolov8m_pool" / "predictions" / "semantic_weight_grid_phrase_group_predictions.csv",
        PROJECT_ROOT / "experiments" / "ms_cxr_multibox_rule_context_fusion_v4_yolov8l_pool" / "predictions" / "phrase_group_set_predictions.csv",
        PROJECT_ROOT / "experiments" / "ms_cxr_semantic_ensemble_gated_hybrid_v2_yolov8m_pool" / "predictions" / "semantic_ensemble_gated_hybrid_v2_phrase_group_predictions.csv",
    ]
    for p in ref_paths:
        if p.exists():
            refs.append(pd.read_csv(p))
    refs_detail = pd.concat(refs, ignore_index=True, sort=False) if refs else pd.DataFrame()

    ref_summaries = [s]
    summary_paths = [
        PROJECT_ROOT / "experiments" / "ms_cxr_semantic_weight_grid_v3_yolov8l_pool" / "metrics" / "summary_with_references.csv",
        PROJECT_ROOT / "experiments" / "ms_cxr_semantic_weight_grid_v2_yolov8m_pool" / "metrics" / "semantic_weight_grid_summary.csv",
    ]
    for p in summary_paths:
        if p.exists():
            ref_summaries.append(pd.read_csv(p))
    combined = pd.concat(ref_summaries, ignore_index=True, sort=False).drop_duplicates(subset=["method", "subset"], keep="first")
    combined.to_csv(MET / "summary_with_references.csv", index=False)

    boot = bootstrap(d, refs_detail) if not refs_detail.empty else pd.DataFrame()
    boot.to_csv(MET / "bootstrap_vs_references.csv", index=False)

    all_rows = combined[combined["subset"] == "eval_phrase_groups_all"][
        ["method", "n_groups", "coverage_mean_iou", "gt_hit_rate_0_3", "gt_hit_rate_0_5", "set_f1_0_3", "set_f1_0_5"]
    ].sort_values("coverage_mean_iou", ascending=False)
    lines = [
        "# Semantic Weight Finegrid V2",
        "",
        "YOLOv8l까지 포함한 후보 pool 위에서 hybrid, SigLIP, BioMedCLIP 박스를 validation weight로 섞었다.",
        "Eval gold bbox는 weight 선택에 사용하지 않았다.",
        "",
        "## Best params",
        "",
        "```json",
        json.dumps(best_params, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Eval summary",
        "",
        all_rows.to_markdown(index=False),
        "",
        "## Bootstrap",
        "",
        boot.to_markdown(index=False) if not boot.empty else "not available",
        "",
        "## 주의",
        "",
        "- MS-CXR bbox는 phrase-grounding bbox이며 lesion mask가 아니다.",
        "- 이 실험은 validation으로 semantic blend weight만 선택했다.",
    ]
    (REPORT / "README_KO.md").write_text("\n".join(lines), encoding="utf-8")

    own = s[s["subset"] == "eval_phrase_groups_all"].iloc[0]
    print(f"project_root={PROJECT_ROOT}")
    print(f"best_params={json.dumps(best_params, ensure_ascii=False)}")
    print(f"coverage_mean_iou_all={own['coverage_mean_iou']:.6f}")
    print(f"hit03_all={own['gt_hit_rate_0_3']:.6f}")
    print(f"hit05_all={own['gt_hit_rate_0_5']:.6f}")
    print(f"summary_path={MET / 'summary_with_references.csv'}")
    print(f"report_path={REPORT / 'README_KO.md'}")
    print(all_rows.to_string(index=False))


if __name__ == "__main__":
    main()

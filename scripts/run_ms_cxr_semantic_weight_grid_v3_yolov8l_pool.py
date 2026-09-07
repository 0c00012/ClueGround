#!/usr/bin/env python
"""Continuous semantic weight grid for YOLOv8m-pool hybrid.

This tunes a global convex blend of:
  1. v3 YOLO-DINO hybrid box
  2. SigLIP crop-text candidate box
  3. BioMedCLIP crop-text candidate box

Only validation phrase groups choose the weights.  Eval is used once after the
weights are fixed.  Multi-box cue groups keep the v3 hybrid unchanged by
default, because semantic crop-text scorers were weaker on multi-box phrases.
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
from scripts import run_ms_cxr_multibox_rule_context_fusion_v4_yolov8l_pool as mv4  # noqa: E402
from scripts import run_ms_cxr_semantic_ensemble_gated_hybrid_v2_yolov8m_pool as sem2  # noqa: E402
from scripts import run_ms_cxr_yolo_dino_rule_fusion_v1 as yd  # noqa: E402
from scripts import run_ms_cxr_yolo_rule_context_v2 as yv2  # noqa: E402


EXP_NAME = "ms_cxr_semantic_weight_grid_v3_yolov8l_pool"
EXP = PROJECT_ROOT / "experiments" / EXP_NAME
PRED = EXP / "predictions"
MET = EXP / "metrics"
CFG = EXP / "configs"
REPORT = PROJECT_ROOT / "reports" / EXP_NAME

SEM2_EXP = PROJECT_ROOT / "experiments" / "ms_cxr_semantic_ensemble_gated_hybrid_v2_yolov8m_pool"
MULTIBOX_V4 = PROJECT_ROOT / "experiments" / "ms_cxr_multibox_rule_context_fusion_v4_yolov8l_pool"
YOLO_V2_CFG = PROJECT_ROOT / "experiments" / "ms_cxr_yolo_rule_context_v2" / "configs" / "best_params_by_finding.json"


def ensure_dirs() -> None:
    for p in [EXP, PRED, MET, CFG, REPORT]:
        p.mkdir(parents=True, exist_ok=True)


def weighted_box(boxes: list[list[float]], weights: list[float]) -> list[float]:
    w = np.asarray(weights, dtype=float)
    w = w / max(float(w.sum()), 1e-8)
    arr = np.asarray(boxes, dtype=float)
    return (arr * w[:, None]).sum(axis=0).tolist()


def has_multi_cue_map(cue: pd.DataFrame) -> dict[str, bool]:
    return {str(r["group_id"]): bool(r.get("has_multi_cue", False)) for _, r in cue.iterrows()}


def build_inputs(split: str) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    pd.DataFrame,
]:
    groups = sem2.gh.load_groups(split)
    hybrid, cue = build_hybrid_v4(split, groups)
    siglip = sem2.build_siglip_set(split, groups)
    biomed = sem2.build_biomedclip_set(split, groups)
    return groups, hybrid, siglip, biomed, cue


def build_hybrid_v4(split: str, groups: dict[str, dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], pd.DataFrame]:
    train_rows = mv4.load_rows("train")
    priors = yd.base.make_train_priors(train_rows)
    yolo_params = json.loads(YOLO_V2_CFG.read_text(encoding="utf-8"))
    set_params = json.loads((MULTIBOX_V4 / "configs" / "best_set_prediction_params.json").read_text(encoding="utf-8"))
    candidates = yv2.read_candidates(split, ["yolov8n", "yolov8s", "yolov8m", "yolov8l"])
    dino = mv4.load_dino_by_group(split, groups)
    v4_preds, cue_rows = mv4.predict_groups(groups, candidates, priors, yolo_params, dino, set_params)
    cue_by_gid = {r["group_id"]: bool(r.get("has_multi_cue", False)) for r in cue_rows}
    single = sem2.gh.build_yolo_dino_singleton(split, groups)
    hybrid: dict[str, list[dict[str, Any]]] = {}
    for gid in groups:
        hybrid[gid] = v4_preds.get(gid, []) if cue_by_gid.get(gid, False) else single.get(gid, [])
    return hybrid, pd.DataFrame(cue_rows)


def predict(
    groups: dict[str, dict[str, Any]],
    hybrid: dict[str, list[dict[str, Any]]],
    siglip: dict[str, list[dict[str, Any]]],
    biomed: dict[str, list[dict[str, Any]]],
    cue: pd.DataFrame,
    weights: tuple[float, float, float],
    keep_multi: bool,
    min_hybrid_semantic_iou: float,
) -> tuple[dict[str, list[dict[str, Any]]], pd.DataFrame]:
    multi = has_multi_cue_map(cue)
    out: dict[str, list[dict[str, Any]]] = {}
    audit: list[dict[str, Any]] = []
    for gid in groups:
        h = hybrid.get(gid, [])
        s = siglip.get(gid, [])
        b = biomed.get(gid, [])
        pred = h
        action = "keep_hybrid"
        hs = hb = None
        eligible = len(h) == 1 and len(s) == 1 and len(b) == 1
        if eligible and not (keep_multi and multi.get(gid, False)):
            hb0, sb0, bb0 = h[0]["box"], s[0]["box"], b[0]["box"]
            hs = mb.iou_xyxy(hb0, sb0)
            hb = mb.iou_xyxy(hb0, bb0)
            if max(hs, hb) >= min_hybrid_semantic_iou:
                box = weighted_box([hb0, sb0, bb0], list(weights))
                pred = [{
                    "box": box,
                    "score": max(float(h[0].get("score", 0.0)), float(s[0].get("score", 0.0)), float(b[0].get("score", 0.0))),
                    "source": "semantic_weight_grid",
                }]
                action = "blend"
        out[gid] = pred
        audit.append({
            "group_id": gid,
            "eligible": eligible,
            "has_multi_cue": multi.get(gid, False),
            "action": action,
            "iou_hybrid_siglip": hs,
            "iou_hybrid_biomedclip": hb,
            "w_hybrid": weights[0],
            "w_siglip": weights[1],
            "w_biomed": weights[2],
        })
    return out, pd.DataFrame(audit)


def detail(method: str, groups: dict[str, dict[str, Any]], preds: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    return pd.DataFrame(mb.eval_method(method, groups, preds))


def summary(method: str, d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subset, sub in {
        "eval_phrase_groups_all": d,
        "eval_phrase_groups_single_box": d[d["n_gt"] == 1],
        "eval_phrase_groups_multi_box": d[d["n_gt"] > 1],
    }.items():
        rows.append(mb.summarize(sub.to_dict("records"), method, subset))
    return pd.DataFrame(rows)


def weight_grid(step: float = 0.05) -> list[tuple[float, float, float]]:
    vals = [round(i * step, 10) for i in range(int(round(1 / step)) + 1)]
    out = []
    for wh in vals:
        for ws in vals:
            wb = round(1.0 - wh - ws, 10)
            if wb < -1e-8:
                continue
            if wb < 0:
                wb = 0.0
            out.append((wh, ws, wb))
    return out


def tune() -> tuple[dict[str, Any], pd.DataFrame]:
    groups, hybrid, siglip, biomed, cue = build_inputs("val")
    rows = []
    best_key = None
    best = None
    for keep_multi in [True, False]:
        for min_iou in [0.0, 0.1, 0.2, 0.35, 0.5, 0.65]:
            for weights in weight_grid(0.05):
                preds, audit = predict(groups, hybrid, siglip, biomed, cue, weights, keep_multi, min_iou)
                d = detail("val_weight_grid", groups, preds)
                s = mb.summarize(d.to_dict("records"), "val_weight_grid", "val_all")
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
                key = (float(s["coverage_mean_iou"]), float(s["set_f1_0_3"]), float(s["gt_hit_rate_0_5"]))
                if best_key is None or key > best_key:
                    best_key = key
                    best = row
    grid = pd.DataFrame(rows).sort_values(["coverage_mean_iou", "set_f1_0_3", "gt_hit_rate_0_5"], ascending=False)
    assert best is not None
    return best, grid


def bootstrap(target_detail: pd.DataFrame, refs: pd.DataFrame) -> pd.DataFrame:
    target = "semantic_weight_grid_v3_yolov8l_pool"
    methods = [
        "semantic_ensemble_gated_hybrid_v2_yolov8m_pool",
        "hybrid_singlebox_fusion_plus_context_multibox_v4_yolov8l_pool",
        "semantic_weight_grid_v2_yolov8m_pool",
        "semantic_ensemble_conservative_multibox_v1",
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
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, len(diff), len(diff))
            boots.append(float(diff[idx].mean()))
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
    grid.to_csv(MET / "weight_grid_val.csv", index=False)
    best_params = {
        "keep_multi": bool(best["keep_multi"]),
        "min_hybrid_semantic_iou": float(best["min_hybrid_semantic_iou"]),
        "w_hybrid": float(best["w_hybrid"]),
        "w_siglip": float(best["w_siglip"]),
        "w_biomed": float(best["w_biomed"]),
        "n_changed_val": int(best["n_changed"]),
    }
    (CFG / "best_weight_grid_params.json").write_text(json.dumps(best_params, ensure_ascii=False, indent=2), encoding="utf-8")

    groups, hybrid, siglip, biomed, cue = build_inputs("eval")
    preds, audit = predict(
        groups,
        hybrid,
        siglip,
        biomed,
        cue,
        (best_params["w_hybrid"], best_params["w_siglip"], best_params["w_biomed"]),
        best_params["keep_multi"],
        best_params["min_hybrid_semantic_iou"],
    )
    method = "semantic_weight_grid_v3_yolov8l_pool"
    d = detail(method, groups, preds)
    s = summary(method, d)
    d.to_csv(PRED / "semantic_weight_grid_phrase_group_predictions.csv", index=False)
    audit.to_csv(PRED / "semantic_weight_grid_action_audit.csv", index=False)
    s.to_csv(MET / "semantic_weight_grid_summary.csv", index=False)

    sem2_detail = pd.read_csv(SEM2_EXP / "predictions" / "semantic_ensemble_gated_hybrid_v2_phrase_group_predictions.csv")
    v3_detail = pd.read_csv(PROJECT_ROOT / "experiments" / "ms_cxr_multibox_rule_context_fusion_v4_yolov8l_pool" / "predictions" / "phrase_group_set_predictions.csv")
    wg2_detail_path = PROJECT_ROOT / "experiments" / "ms_cxr_semantic_weight_grid_v2_yolov8m_pool" / "predictions" / "semantic_weight_grid_phrase_group_predictions.csv"
    old_cons = PROJECT_ROOT / "experiments" / "ms_cxr_semantic_ensemble_gated_hybrid_v1" / "predictions" / "semantic_ensemble_conservative_multibox_phrase_group_predictions.csv"
    refs_list = [sem2_detail, v3_detail]
    if wg2_detail_path.exists():
        refs_list.append(pd.read_csv(wg2_detail_path))
    if old_cons.exists():
        refs_list.append(pd.read_csv(old_cons))
    refs_detail = pd.concat(refs_list, ignore_index=True, sort=False)
    sem2_summary = pd.read_csv(SEM2_EXP / "metrics" / "summary_with_references.csv")
    v4_summary = pd.read_csv(MULTIBOX_V4 / "metrics" / "phrase_group_set_summary.csv")
    wg2_summary_path = PROJECT_ROOT / "experiments" / "ms_cxr_semantic_weight_grid_v2_yolov8m_pool" / "metrics" / "semantic_weight_grid_summary.csv"
    summary_refs = [sem2_summary, v4_summary]
    if wg2_summary_path.exists():
        summary_refs.append(pd.read_csv(wg2_summary_path))
    combined = pd.concat([s, *summary_refs], ignore_index=True, sort=False).drop_duplicates(subset=["method", "subset"], keep="first")
    combined.to_csv(MET / "summary_with_references.csv", index=False)
    boot = bootstrap(d, refs_detail)
    boot.to_csv(MET / "bootstrap_vs_references.csv", index=False)

    lines = [
        "# Semantic Weight Grid V3",
        "",
        "## 핵심",
        "",
        "v3 YOLO-DINO hybrid, SigLIP, BioMedCLIP 세 박스를 validation에서 정한 전역 가중치로 섞었다.",
        "eval gold bbox는 weight 선택에 사용하지 않았다.",
        "",
        "## Best params",
        "",
        "```json",
        json.dumps(best_params, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Eval summary",
        "",
        combined[combined["subset"] == "eval_phrase_groups_all"][
            ["method", "n_groups", "coverage_mean_iou", "gt_hit_rate_0_3", "gt_hit_rate_0_5", "set_f1_0_3", "set_f1_0_5"]
        ].sort_values("coverage_mean_iou", ascending=False).to_markdown(index=False),
        "",
        "## Bootstrap",
        "",
        boot.to_markdown(index=False),
        "",
        "## 해석 주의",
        "",
        "- weight는 validation split에서만 선택했다.",
        "- MS-CXR bbox는 phrase-grounding bbox이며 lesion mask가 아니다.",
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
    print(combined[combined["subset"] == "eval_phrase_groups_all"][
        ["method", "n_groups", "coverage_mean_iou", "gt_hit_rate_0_3", "gt_hit_rate_0_5", "set_f1_0_3", "set_f1_0_5"]
    ].sort_values("coverage_mean_iou", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()

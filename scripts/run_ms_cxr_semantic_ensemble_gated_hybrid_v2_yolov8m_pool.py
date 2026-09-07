#!/usr/bin/env python
"""Semantic-ensemble gated YOLO-DINO hybrid V2 for MS-CXR phrase groups.

This experiment combines the v3 YOLO-DINO multibox hybrid with two
frozen semantic crop-text scorers:

* SigLIP candidate set
* BioMedCLIP candidate set

The rule is selected on validation phrase groups only.  Eval gold is never used
to choose weights or gates.  Multi-box phrases keep the existing multibox
hybrid by default because both standalone semantic scorers were weaker there.

MS-CXR boxes are phrase-grounding boxes, not pixel-level lesion masks.
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
from scripts import run_ms_cxr_multibox_rule_context_fusion_v3_yolov8m_pool as mv3  # noqa: E402
from scripts import run_ms_cxr_siglip_candidate_fusion_v1 as sf  # noqa: E402
from scripts import run_ms_cxr_siglip_gated_hybrid_v1 as gh  # noqa: E402
from scripts import run_ms_cxr_yolo_dino_rule_fusion_v1 as yd  # noqa: E402
from scripts import run_ms_cxr_yolo_rule_context_v2 as yv2  # noqa: E402


EXP_NAME = "ms_cxr_semantic_ensemble_gated_hybrid_v2_yolov8m_pool"
EXP = PROJECT_ROOT / "experiments" / EXP_NAME
PRED = EXP / "predictions"
MET = EXP / "metrics"
CFG = EXP / "configs"
REPORT = PROJECT_ROOT / "reports" / EXP_NAME

MULTIBOX_V3 = PROJECT_ROOT / "experiments" / "ms_cxr_multibox_rule_context_fusion_v3_yolov8m_pool"
SIGLIP_EXP = PROJECT_ROOT / "experiments" / "ms_cxr_siglip_candidate_fusion_v1"
SIGLIP_GATE = PROJECT_ROOT / "experiments" / "ms_cxr_siglip_gated_hybrid_v1"
BIOMED_EXP = PROJECT_ROOT / "experiments" / "ms_cxr_biomedclip_gated_hybrid_v1"
YOLO_V2_CFG = PROJECT_ROOT / "experiments" / "ms_cxr_yolo_rule_context_v2" / "configs" / "best_params_by_finding.json"


def ensure_dirs() -> None:
    for p in [EXP, PRED, MET, CFG, REPORT]:
        p.mkdir(parents=True, exist_ok=True)


def build_siglip_set(split: str, groups: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return gh.build_siglip_set(split, groups)


def build_hybrid_v3(split: str, groups: dict[str, dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], pd.DataFrame]:
    """Build v3 hybrid predictions using YOLOv8n/v8s/v8m candidates."""
    train_rows = mv3.load_rows("train")
    priors = yd.base.make_train_priors(train_rows)
    yolo_params = json.loads(YOLO_V2_CFG.read_text(encoding="utf-8"))
    set_params = json.loads((MULTIBOX_V3 / "configs" / "best_set_prediction_params.json").read_text(encoding="utf-8"))
    candidates = yv2.read_candidates(split, ["yolov8n", "yolov8s", "yolov8m"])
    dino = mv3.load_dino_by_group(split, groups)
    v3_preds, cue_rows = mv3.predict_groups(groups, candidates, priors, yolo_params, dino, set_params)
    cue_by_gid = {r["group_id"]: bool(r.get("has_multi_cue", False)) for r in cue_rows}
    single = gh.build_yolo_dino_singleton(split, groups)
    hybrid: dict[str, list[dict[str, Any]]] = {}
    for gid in groups:
        hybrid[gid] = v3_preds.get(gid, []) if cue_by_gid.get(gid, False) else single.get(gid, [])
    return hybrid, pd.DataFrame(cue_rows)


def build_biomedclip_set(split: str, groups: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    path = BIOMED_EXP / "predictions" / f"{split}_biomedclip_fusion_scored_candidates.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    params_path = BIOMED_EXP / "configs" / "best_candidate_set_params.json"
    if not params_path.exists():
        raise FileNotFoundError(params_path)
    scored = pd.read_csv(path)
    params = json.loads(params_path.read_text(encoding="utf-8"))
    cands = sf.scored_candidates_by_group(scored, groups, "biomedclip_fusion_score")
    return sf.predict_phrase_sets(groups, cands, params)


def box_iou(a: list[float], b: list[float]) -> float:
    return float(mb.iou_xyxy(a, b))


def weighted_box(boxes: list[list[float]], weights: list[float]) -> list[float]:
    w = np.asarray(weights, dtype=float)
    w = w / max(float(w.sum()), 1e-8)
    arr = np.asarray(boxes, dtype=float)
    return (arr * w[:, None]).sum(axis=0).tolist()


def median_box(boxes: list[list[float]]) -> list[float]:
    return np.median(np.asarray(boxes, dtype=float), axis=0).tolist()


def cue_by_gid(cue: pd.DataFrame) -> dict[str, bool]:
    return {str(r["group_id"]): bool(r.get("has_multi_cue", False)) for _, r in cue.iterrows()}


def combine(
    groups: dict[str, dict[str, Any]],
    hybrid: dict[str, list[dict[str, Any]]],
    siglip: dict[str, list[dict[str, Any]]],
    biomed: dict[str, list[dict[str, Any]]],
    cue: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], pd.DataFrame]:
    multi = cue_by_gid(cue)
    out: dict[str, list[dict[str, Any]]] = {}
    audit: list[dict[str, Any]] = []
    for gid in groups:
        h = hybrid.get(gid, [])
        s = siglip.get(gid, [])
        b = biomed.get(gid, [])
        has_multi = multi.get(gid, False)
        action = "keep_hybrid"
        pred = h
        hs = hb = sb = None
        eligible = len(h) == 1 and len(s) == 1 and len(b) == 1
        if eligible:
            hb0 = h[0]["box"]
            sb0 = s[0]["box"]
            bb0 = b[0]["box"]
            hs = box_iou(hb0, sb0)
            hb = box_iou(hb0, bb0)
            sb = box_iou(sb0, bb0)
            if not (has_multi and bool(params["keep_multi"])):
                use = False
                mode = str(params["gate_mode"])
                if mode == "always":
                    use = True
                elif mode == "semantic_agree":
                    use = sb >= float(params["min_semantic_agree"])
                elif mode == "hybrid_semantic_agree":
                    use = max(hs, hb) >= float(params["min_hybrid_semantic_agree"])
                elif mode == "both":
                    use = (
                        sb >= float(params["min_semantic_agree"])
                        and max(hs, hb) >= float(params["min_hybrid_semantic_agree"])
                    )
                if use:
                    blend = str(params["blend_mode"])
                    if blend == "weighted":
                        box = weighted_box(
                            [hb0, sb0, bb0],
                            [float(params["w_hybrid"]), float(params["w_siglip"]), float(params["w_biomed"])],
                        )
                    elif blend == "median":
                        box = median_box([hb0, sb0, bb0])
                    elif blend == "siglip_biomed_mid":
                        box = weighted_box([sb0, bb0], [0.5, 0.5])
                    elif blend == "hybrid_siglip_mid":
                        box = weighted_box([hb0, sb0], [0.5, 0.5])
                    elif blend == "hybrid_biomed_mid":
                        box = weighted_box([hb0, bb0], [0.5, 0.5])
                    else:
                        raise ValueError(blend)
                    pred = [{"box": box, "score": max(float(h[0].get("score", 0.0)), float(s[0].get("score", 0.0)), float(b[0].get("score", 0.0))), "source": f"semantic_ensemble_{blend}"}]
                    action = f"ensemble_{blend}"
        out[gid] = pred
        audit.append({
            "group_id": gid,
            "has_multi_cue": has_multi,
            "eligible": eligible,
            "action": action,
            "iou_hybrid_siglip": hs,
            "iou_hybrid_biomedclip": hb,
            "iou_siglip_biomedclip": sb,
            "n_hybrid": len(h),
            "n_siglip": len(s),
            "n_biomedclip": len(b),
        })
    return out, pd.DataFrame(audit)


def summarize(method: str, groups: dict[str, dict[str, Any]], preds: dict[str, list[dict[str, Any]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail = pd.DataFrame(mb.eval_method(method, groups, preds))
    rows: list[dict[str, Any]] = []
    subsets = {
        "eval_phrase_groups_all": detail,
        "eval_phrase_groups_single_box": detail[detail["n_gt"] == 1],
        "eval_phrase_groups_multi_box": detail[detail["n_gt"] > 1],
    }
    for subset, sub in subsets.items():
        rows.append(mb.summarize(sub.to_dict("records"), method, subset))
    return detail, pd.DataFrame(rows)


def tune(
    groups: dict[str, dict[str, Any]],
    hybrid: dict[str, list[dict[str, Any]]],
    siglip: dict[str, list[dict[str, Any]]],
    biomed: dict[str, list[dict[str, Any]]],
    cue: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    best_params: dict[str, Any] | None = None
    best_key: tuple[float, float, float] | None = None
    weight_grid = [
        (0.50, 0.25, 0.25),
        (0.40, 0.30, 0.30),
        (0.60, 0.20, 0.20),
        (0.33, 0.33, 0.34),
        (0.25, 0.50, 0.25),
        (0.25, 0.25, 0.50),
        (0.70, 0.15, 0.15),
        (0.45, 0.40, 0.15),
        (0.45, 0.15, 0.40),
    ]
    for keep_multi in [True, False]:
        for gate_mode in ["always", "semantic_agree", "hybrid_semantic_agree", "both"]:
            for min_sem in [0.0, 0.2, 0.35, 0.5, 0.65]:
                for min_hsem in [0.0, 0.2, 0.35, 0.5, 0.65]:
                    for blend_mode in ["weighted", "median", "siglip_biomed_mid", "hybrid_siglip_mid", "hybrid_biomed_mid"]:
                        combos = weight_grid if blend_mode == "weighted" else [(0.0, 0.0, 0.0)]
                        for wh, ws, wb in combos:
                            params = {
                                "keep_multi": keep_multi,
                                "gate_mode": gate_mode,
                                "min_semantic_agree": min_sem,
                                "min_hybrid_semantic_agree": min_hsem,
                                "blend_mode": blend_mode,
                                "w_hybrid": wh,
                                "w_siglip": ws,
                                "w_biomed": wb,
                            }
                            preds, audit = combine(groups, hybrid, siglip, biomed, cue, params)
                            _, summary = summarize("val_semantic_ensemble", groups, preds)
                            all_row = summary[summary["subset"] == "eval_phrase_groups_all"].iloc[0].to_dict()
                            n_changed = int((audit["action"] != "keep_hybrid").sum())
                            row = {**params, **all_row, "n_changed": n_changed}
                            rows.append(row)
                            key = (
                                float(all_row["coverage_mean_iou"]),
                                float(all_row["set_f1_0_3"]),
                                -abs(n_changed - 30),
                            )
                            if best_key is None or key > best_key:
                                best_key = key
                                best_params = params
    assert best_params is not None
    grid = pd.DataFrame(rows).sort_values(["coverage_mean_iou", "set_f1_0_3"], ascending=False)
    return best_params, grid


def bootstrap(detail: pd.DataFrame, ref_detail: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(20260705)
    target = "semantic_ensemble_gated_hybrid_v2_yolov8m_pool"
    methods = [
        "hybrid_singlebox_fusion_plus_context_multibox_v3_yolov8m_pool",
        "semantic_ensemble_conservative_multibox_v1",
        "siglip_gated_hybrid_v1",
        "biomedclip_gated_hybrid_v1",
        "medrpg_rowlevel_full_phrase_s2026",
        "medrpg_rowlevel_full_phrase_s13",
        "medrpg_rowlevel_full_phrase_s42",
    ]
    all_df = pd.concat([detail, ref_detail[ref_detail["method"].isin(methods)]], ignore_index=True, sort=False)
    rows: list[dict[str, Any]] = []
    for b in methods:
        piv = all_df[all_df["method"].isin([target, b])].pivot(index="group_id", columns="method", values="coverage_mean_iou")
        if target not in piv.columns or b not in piv.columns:
            continue
        piv = piv.dropna(subset=[target, b])
        if not len(piv):
            continue
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
            "metric": "coverage_mean_iou",
            "mean_diff": float(diff.mean()),
            "ci95_low": float(lo),
            "ci95_high": float(hi),
            "p_diff_le_0": float((np.asarray(boots) <= 0).mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    val_groups = gh.load_groups("val")
    eval_groups = gh.load_groups("eval")
    val_hybrid, val_cue = build_hybrid_v3("val", val_groups)
    eval_hybrid, eval_cue = build_hybrid_v3("eval", eval_groups)
    val_siglip = build_siglip_set("val", val_groups)
    eval_siglip = build_siglip_set("eval", eval_groups)
    val_biomed = build_biomedclip_set("val", val_groups)
    eval_biomed = build_biomedclip_set("eval", eval_groups)

    best, grid = tune(val_groups, val_hybrid, val_siglip, val_biomed, val_cue)
    grid.to_csv(MET / "ensemble_val_grid.csv", index=False)
    (CFG / "best_ensemble_params.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")

    method_name = "semantic_ensemble_gated_hybrid_v2_yolov8m_pool"
    preds, audit = combine(eval_groups, eval_hybrid, eval_siglip, eval_biomed, eval_cue, best)
    detail, summary = summarize(method_name, eval_groups, preds)
    detail.to_csv(PRED / "semantic_ensemble_gated_hybrid_v2_phrase_group_predictions.csv", index=False)
    audit.to_csv(PRED / "semantic_ensemble_action_audit.csv", index=False)

    ref_detail = pd.read_csv(MULTIBOX_V3 / "predictions" / "phrase_group_set_predictions.csv")
    sig_detail = pd.read_csv(SIGLIP_GATE / "predictions" / "siglip_gated_hybrid_phrase_group_predictions.csv")
    bio_detail = pd.read_csv(BIOMED_EXP / "predictions" / "biomedclip_gated_hybrid_phrase_group_predictions.csv")
    ref_detail = pd.concat([ref_detail, sig_detail, bio_detail], ignore_index=True, sort=False)
    old_cons = PROJECT_ROOT / "experiments" / "ms_cxr_semantic_ensemble_gated_hybrid_v1"
    cons_detail_path = old_cons / "predictions" / "semantic_ensemble_conservative_multibox_phrase_group_predictions.csv"
    if cons_detail_path.exists():
        ref_detail = pd.concat([ref_detail, pd.read_csv(cons_detail_path)], ignore_index=True, sort=False)

    ref_sum = pd.read_csv(MULTIBOX_V3 / "metrics" / "phrase_group_set_summary.csv")
    sig_sum = pd.read_csv(SIGLIP_GATE / "metrics" / "summary_with_references.csv")
    bio_sum = pd.read_csv(BIOMED_EXP / "metrics" / "summary_with_references.csv")
    cons_sum_path = old_cons / "metrics" / "conservative_multibox_summary.csv"
    cons_sum = pd.read_csv(cons_sum_path) if cons_sum_path.exists() else pd.DataFrame()
    keep = [
        "hybrid_singlebox_fusion_plus_context_multibox_v3_yolov8m_pool",
        "hybrid_singlebox_fusion_plus_context_multibox_v2",
        "semantic_ensemble_conservative_multibox_v1",
        "siglip_gated_hybrid_v1",
        "biomedclip_gated_hybrid_v1",
        "rule_context_multibox_yolo_dino_v3_yolov8m_pool",
        "medrpg_rowlevel_full_phrase_s2026",
        "medrpg_rowlevel_full_phrase_s13",
        "medrpg_rowlevel_full_phrase_s42",
    ]
    ref_sum = pd.concat([ref_sum, sig_sum, bio_sum, cons_sum], ignore_index=True, sort=False)
    ref_sum = ref_sum[ref_sum["method"].isin(keep)].drop_duplicates(subset=["method", "subset"], keep="first")
    combined = pd.concat([summary, ref_sum], ignore_index=True, sort=False)
    combined.to_csv(MET / "summary_with_references.csv", index=False)

    boot = bootstrap(detail, ref_detail)
    boot.to_csv(MET / "bootstrap_vs_references.csv", index=False)

    lines = [
        "# Semantic-Ensemble Gated Hybrid V2",
        "",
        "## 결론",
        "",
        "기존 YOLO-DINO hybrid를 기본값으로 두고, SigLIP과 BioMedCLIP 후보를 validation에서 선택한 규칙으로만 섞었다.",
        "두 semantic scorer는 모두 frozen crop-text scorer로만 사용했다.",
        "",
        "## Best ensemble params",
        "",
        "```json",
        json.dumps(best, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Eval summary",
        "",
        combined[combined["subset"] == "eval_phrase_groups_all"][[
            "method", "n_groups", "coverage_mean_iou", "gt_hit_rate_0_3", "gt_hit_rate_0_5", "set_f1_0_3", "set_f1_0_5"
        ]].sort_values("coverage_mean_iou", ascending=False).to_markdown(index=False),
        "",
        "## Action audit",
        "",
        audit.groupby("action").size().reset_index(name="n_groups").to_markdown(index=False),
        "",
        "## Bootstrap",
        "",
        boot.to_markdown(index=False),
        "",
        "## 해석 주의",
        "",
        "- ensemble rule은 validation split에서만 선택했다.",
        "- eval gold bbox는 조합 규칙 선택에 사용하지 않았다.",
        "- MS-CXR bbox는 phrase-grounding bbox이며 lesion mask가 아니다.",
        "- semantic scorer 단독이 아니라 YOLO-DINO 후보 보정기로만 해석해야 한다.",
    ]
    (REPORT / "README_KO.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"project_root={PROJECT_ROOT}")
    print(f"best_params={json.dumps(best, ensure_ascii=False)}")
    print(f"summary_path={MET / 'summary_with_references.csv'}")
    print(f"report_path={REPORT / 'README_KO.md'}")
    print(combined[combined["subset"] == "eval_phrase_groups_all"][[
        "method", "n_groups", "coverage_mean_iou", "gt_hit_rate_0_3", "gt_hit_rate_0_5", "set_f1_0_3", "set_f1_0_5"
    ]].sort_values("coverage_mean_iou", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()

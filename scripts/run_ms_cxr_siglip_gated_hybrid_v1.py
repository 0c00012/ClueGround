#!/usr/bin/env python
"""SigLIP-gated YOLO-DINO hybrid for MS-CXR phrase-group grounding.

This is a conservative fusion experiment:

* Keep the existing hybrid YOLO-DINO multibox v2 as the default prediction.
* Use frozen SigLIP crop-text scores only as a gate/blend for ambiguous
  one-box phrase groups.
* Select the gate/blend rule on the validation split only.
* Apply the selected rule once to the eval phrase groups.

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
from scripts import run_ms_cxr_multibox_rule_context_fusion_v2 as mv2  # noqa: E402
from scripts import run_ms_cxr_siglip_candidate_fusion_v1 as sf  # noqa: E402
from scripts import run_ms_cxr_yolo_dino_rule_fusion_v1 as yd  # noqa: E402
from scripts import run_ms_cxr_yolo_rule_context_v2 as yv2  # noqa: E402


EXP_NAME = "ms_cxr_siglip_gated_hybrid_v1"
EXP = PROJECT_ROOT / "experiments" / EXP_NAME
PRED = EXP / "predictions"
MET = EXP / "metrics"
CFG = EXP / "configs"
REPORT = PROJECT_ROOT / "reports" / EXP_NAME

MULTIBOX_V2 = PROJECT_ROOT / "experiments" / "ms_cxr_multibox_rule_context_fusion_v2"
SIGLIP_EXP = PROJECT_ROOT / "experiments" / "ms_cxr_siglip_candidate_fusion_v1"
FUSION_PRED = PROJECT_ROOT / "experiments" / "ms_cxr_yolo_dino_rule_fusion_v1" / "predictions"
FUSION_CFG = PROJECT_ROOT / "experiments" / "ms_cxr_yolo_dino_rule_fusion_v1" / "configs" / "best_fusion_by_finding.json"
YOLO_V2_CFG = PROJECT_ROOT / "experiments" / "ms_cxr_yolo_rule_context_v2" / "configs" / "best_params_by_finding.json"


def ensure_dirs() -> None:
    for p in [EXP, PRED, MET, CFG, REPORT]:
        p.mkdir(parents=True, exist_ok=True)


def load_groups(split: str) -> dict[str, dict[str, Any]]:
    return mb.make_groups(mv2.load_rows(split))


def singleton_map_from_row_predictions(df: pd.DataFrame, groups: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_task: dict[str, dict[str, Any]] = {}
    for _, r in df.iterrows():
        by_task[str(r["task_id"])] = {
            "box": [float(r["pred_x1"]), float(r["pred_y1"]), float(r["pred_x2"]), float(r["pred_y2"])],
            "score": float(r.get("confidence", 1.0)),
            "source": str(r.get("source", "yolo_dino_rule_fusion_v1")),
        }
    out: dict[str, list[dict[str, Any]]] = {}
    for gid, g in groups.items():
        boxes = []
        for task_id in g["task_ids"]:
            if task_id in by_task:
                boxes = [by_task[task_id]]
                break
        out[gid] = boxes
    return out


def build_yolo_dino_singleton(split: str, groups: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    rows = mv2.load_rows(split)
    priors = yd.base.make_train_priors(mv2.load_rows("train"))
    params_by_finding = json.loads(YOLO_V2_CFG.read_text(encoding="utf-8"))
    fusion_by_finding = json.loads(FUSION_CFG.read_text(encoding="utf-8"))
    candidates = yv2.read_candidates(split, ["yolov8n", "yolov8s"])
    dino_path = FUSION_PRED / f"rad_dino_rule_context_{split}_predictions.csv"
    dino = yd.dino_map(pd.read_csv(dino_path))
    pred = yd.evaluate_fusion(
        rows,
        candidates,
        priors,
        dino,
        params_by_finding,
        fusion_by_finding,
        "yolo_dino_rule_fusion_v1",
        split,
    )
    pred.to_csv(PRED / f"{split}_recomputed_yolo_dino_singleton.csv", index=False)
    return singleton_map_from_row_predictions(pred, groups)


def build_hybrid_v2(split: str, groups: dict[str, dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], pd.DataFrame]:
    train_rows = mv2.load_rows("train")
    priors = yd.base.make_train_priors(train_rows)
    yolo_params = json.loads(YOLO_V2_CFG.read_text(encoding="utf-8"))
    set_params = json.loads((MULTIBOX_V2 / "configs" / "best_set_prediction_params.json").read_text(encoding="utf-8"))
    candidates = yv2.read_candidates(split, ["yolov8n", "yolov8s"])
    dino = mv2.load_dino_by_group(split, groups)
    v2_preds, cue_rows = mv2.predict_groups(groups, candidates, priors, yolo_params, dino, set_params)
    cue_by_gid = {r["group_id"]: bool(r.get("has_multi_cue", False)) for r in cue_rows}
    single = build_yolo_dino_singleton(split, groups)
    hybrid: dict[str, list[dict[str, Any]]] = {}
    for gid in groups:
        hybrid[gid] = v2_preds.get(gid, []) if cue_by_gid.get(gid, False) else single.get(gid, [])
    return hybrid, pd.DataFrame(cue_rows)


def build_siglip_set(split: str, groups: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    suffix = "cxr_claim_m0p15"
    scored_path = SIGLIP_EXP / "predictions" / f"{split}_siglip_scored_candidates_{suffix}.csv"
    if not scored_path.exists():
        raise FileNotFoundError(scored_path)
    scored = pd.read_csv(scored_path)
    row_params = json.loads((SIGLIP_EXP / "configs" / "best_row_fusion_params.json").read_text(encoding="utf-8"))
    set_params = json.loads((SIGLIP_EXP / "configs" / "best_set_params.json").read_text(encoding="utf-8"))
    scored = sf.apply_fusion_score(scored, row_params, "siglip_fusion_score")
    cands = sf.scored_candidates_by_group(scored, groups, "siglip_fusion_score")
    return sf.predict_phrase_sets(groups, cands, set_params)


def box_iou(a: list[float], b: list[float]) -> float:
    return float(mb.iou_xyxy(a, b))


def blend_box(base: list[float], siglip: list[float], alpha: float) -> list[float]:
    return ((1.0 - alpha) * np.asarray(base, dtype=float) + alpha * np.asarray(siglip, dtype=float)).tolist()


def combine_predictions(
    groups: dict[str, dict[str, Any]],
    hybrid: dict[str, list[dict[str, Any]]],
    siglip: dict[str, list[dict[str, Any]]],
    cue: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], pd.DataFrame]:
    cue_by_gid = {str(r["group_id"]): bool(r.get("has_multi_cue", False)) for _, r in cue.iterrows()}
    rows: list[dict[str, Any]] = []
    out: dict[str, list[dict[str, Any]]] = {}
    for gid in groups:
        h = hybrid.get(gid, [])
        s = siglip.get(gid, [])
        use_scope = str(params["scope"])
        has_multi_cue = cue_by_gid.get(gid, False)
        eligible = False
        if use_scope == "no_multi_cue":
            eligible = (not has_multi_cue) and len(h) == 1 and len(s) == 1
        elif use_scope == "all_one_pred":
            eligible = len(h) == 1 and len(s) == 1
        elif use_scope == "all_groups":
            eligible = len(h) == 1 and len(s) == 1

        action = "keep_hybrid"
        agreement = None
        pred = h
        if eligible:
            hb = h[0]["box"]
            sb = s[0]["box"]
            agreement = box_iou(hb, sb)
            agree_min = float(params["agree_min"])
            agree_max = float(params["agree_max"])
            if agree_min <= agreement <= agree_max:
                alpha = float(params["alpha"])
                if alpha <= 0.0:
                    pred = h
                    action = "hybrid_agree"
                elif alpha >= 1.0:
                    pred = [{**s[0], "source": "siglip_gate"}]
                    action = "siglip_override"
                else:
                    pred = [{
                        "box": blend_box(hb, sb, alpha),
                        "score": max(float(h[0].get("score", 0.0)), float(s[0].get("score", 0.0))),
                        "source": f"hybrid_siglip_blend_{alpha:g}",
                    }]
                    action = "blend"
            elif str(params["outside_action"]) == "siglip":
                pred = [{**s[0], "source": "siglip_outside"}]
                action = "siglip_outside"
        out[gid] = pred
        rows.append({
            "group_id": gid,
            "has_multi_cue": has_multi_cue,
            "n_hybrid": len(h),
            "n_siglip": len(s),
            "eligible": eligible,
            "agreement_iou": agreement,
            "action": action,
        })
    return out, pd.DataFrame(rows)


def summarize(method: str, groups: dict[str, dict[str, Any]], preds: dict[str, list[dict[str, Any]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail = pd.DataFrame(mb.eval_method(method, groups, preds))
    rows: list[dict[str, Any]] = []
    for subset, sub in [
        ("eval_phrase_groups_all", detail),
        ("eval_phrase_groups_single_box", detail[detail["n_gt"] == 1]),
        ("eval_phrase_groups_multi_box", detail[detail["n_gt"] > 1]),
    ]:
        rows.append(mb.summarize(sub.to_dict("records"), method, subset))
    return detail, pd.DataFrame(rows)


def tune_gate(
    groups: dict[str, dict[str, Any]],
    hybrid: dict[str, list[dict[str, Any]]],
    siglip: dict[str, list[dict[str, Any]]],
    cue: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    best_params: dict[str, Any] | None = None
    best_key: tuple[float, float, float] | None = None
    for scope in ["no_multi_cue", "all_one_pred"]:
        for agree_min in [0.0, 0.1, 0.25, 0.4, 0.55, 0.7]:
            for agree_max in [0.35, 0.55, 0.75, 1.01]:
                if agree_max < agree_min:
                    continue
                for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
                    for outside_action in ["hybrid", "siglip"]:
                        params = {
                            "scope": scope,
                            "agree_min": agree_min,
                            "agree_max": agree_max,
                            "alpha": alpha,
                            "outside_action": outside_action,
                        }
                        pred, audit = combine_predictions(groups, hybrid, siglip, cue, params)
                        detail, summary = summarize("val_siglip_gated_hybrid", groups, pred)
                        all_row = summary[summary["subset"] == "eval_phrase_groups_all"].iloc[0].to_dict()
                        n_changed = int((audit["action"] != "keep_hybrid").sum())
                        row = {**params, **all_row, "n_changed": n_changed}
                        rows.append(row)
                        key = (
                            float(all_row["coverage_mean_iou"]),
                            float(all_row["set_f1_0_3"]),
                            -abs(n_changed - 10),
                        )
                        if best_key is None or key > best_key:
                            best_key = key
                            best_params = params
    assert best_params is not None
    grid = pd.DataFrame(rows).sort_values(["coverage_mean_iou", "set_f1_0_3"], ascending=False)
    return best_params, grid


def main() -> None:
    ensure_dirs()
    val_groups = load_groups("val")
    eval_groups = load_groups("eval")

    val_hybrid, val_cue = build_hybrid_v2("val", val_groups)
    eval_hybrid, eval_cue = build_hybrid_v2("eval", eval_groups)
    val_siglip = build_siglip_set("val", val_groups)
    eval_siglip = build_siglip_set("eval", eval_groups)

    best_params, grid = tune_gate(val_groups, val_hybrid, val_siglip, val_cue)
    grid.to_csv(MET / "gate_val_grid.csv", index=False)
    (CFG / "best_gate_params.json").write_text(json.dumps(best_params, ensure_ascii=False, indent=2), encoding="utf-8")

    eval_pred, audit = combine_predictions(eval_groups, eval_hybrid, eval_siglip, eval_cue, best_params)
    detail, summary = summarize("siglip_gated_hybrid_v1", eval_groups, eval_pred)
    detail.to_csv(PRED / "siglip_gated_hybrid_phrase_group_predictions.csv", index=False)
    audit.to_csv(PRED / "siglip_gated_hybrid_action_audit.csv", index=False)

    ref = pd.read_csv(MULTIBOX_V2 / "metrics" / "phrase_group_set_summary.csv")
    ref = ref[
        (ref["subset"].isin(["eval_phrase_groups_all", "eval_phrase_groups_single_box", "eval_phrase_groups_multi_box"]))
        & (ref["method"].isin([
            "hybrid_singlebox_fusion_plus_context_multibox_v2",
            "rule_context_multibox_yolo_dino_v2",
            "yolo_dino_rule_fusion_v1",
            "medrpg_rowlevel_full_phrase_s13",
            "medrpg_rowlevel_full_phrase_s42",
            "medrpg_rowlevel_full_phrase_s2026",
        ]))
    ].copy()
    sig = pd.read_csv(SIGLIP_EXP / "metrics" / "phrase_group_summary_with_references.csv")
    sig = sig[
        (sig["method"] == "siglip_candidate_fusion_phrase_set")
        & (sig["subset"].isin(["eval_phrase_groups_all", "eval_phrase_groups_single_box", "eval_phrase_groups_multi_box"]))
    ].copy()
    combined = pd.concat([summary, ref, sig], ignore_index=True, sort=False)
    combined.to_csv(MET / "summary_with_references.csv", index=False)

    boot_path = MET / "bootstrap_vs_references.csv"
    boot_text = "(bootstrap not generated)"
    if boot_path.exists():
        boot = pd.read_csv(boot_path)
        boot_text = boot.to_markdown(index=False)

    lines = [
        "# SigLIP-Gated Hybrid V1",
        "",
        "## 결론",
        "",
        "기존 hybrid YOLO-DINO multibox v2를 기본값으로 유지하고, validation에서 선택한 규칙으로만 SigLIP 후보를 제한적으로 섞었다.",
        "SigLIP은 frozen crop-text scorer이며 detector나 bbox-supervised model로 재학습하지 않았다.",
        "",
        "## Best gate params",
        "",
        "```json",
        json.dumps(best_params, ensure_ascii=False, indent=2),
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
        boot_text,
        "",
        "## 해석 주의",
        "",
        "- gate는 validation split에서만 선택했다.",
        "- eval gold bbox는 gate 선택에 사용하지 않았다.",
        "- MS-CXR bbox는 phrase-grounding bbox이며 lesion mask가 아니다.",
        "- SigLIP은 일반 image-text alignment 모델이라 CXR multi-box phrase에는 약할 수 있다.",
    ]
    (REPORT / "README_KO.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"project_root={PROJECT_ROOT}")
    print(f"best_gate_params={json.dumps(best_params, ensure_ascii=False)}")
    print(f"summary_path={MET / 'summary_with_references.csv'}")
    print(f"report_path={REPORT / 'README_KO.md'}")
    print(combined[combined["subset"] == "eval_phrase_groups_all"][[
        "method", "n_groups", "coverage_mean_iou", "gt_hit_rate_0_3", "gt_hit_rate_0_5", "set_f1_0_3", "set_f1_0_5"
    ]].sort_values("coverage_mean_iou", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()

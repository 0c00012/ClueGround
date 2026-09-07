from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .box_features import area_xyxy, iou_xyxy
from .multibox_cue_parser import DIFFUSE_CUE_TYPES, SIDE_CUE_TYPES


def union_box(boxes: list[list[float]]) -> list[float]:
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def candidate_laterality(cx_norm: float) -> str:
    # Radiology display convention: patient right is image-left.
    if cx_norm < 0.47:
        return "right"
    if cx_norm > 0.53:
        return "left"
    return "center"


def candidate_vertical(cy_norm: float) -> str:
    if cy_norm < 0.24:
        return "apical"
    if cy_norm < 0.42:
        return "upper"
    if cy_norm < 0.68:
        return "mid"
    if cy_norm < 0.84:
        return "lower"
    return "basal"


def target_match_score(row: pd.Series, target: dict[str, str]) -> float:
    lat = candidate_laterality(float(row.get("box_cx_norm", 0.5)))
    vert = candidate_vertical(float(row.get("box_cy_norm", 0.5)))
    t_lat = str(target.get("laterality", "unknown"))
    t_vert = str(target.get("vertical", "unknown"))
    score = 0.0
    if t_lat in {"right", "left"}:
        if lat == t_lat:
            score += 1.0
        elif lat in {"right", "left"} and lat != t_lat:
            score -= 1.25
        else:
            score -= 0.2
    if t_vert in {"apical", "upper"}:
        if vert in {"apical", "upper"}:
            score += 0.75
        elif vert in {"lower", "basal"}:
            score -= 0.75
    elif t_vert in {"lower", "basal"}:
        if vert in {"lower", "basal"}:
            score += 0.75
        elif vert in {"apical", "upper"}:
            score -= 0.75
    elif t_vert == "mid":
        if vert == "mid":
            score += 0.5
    return score


def _row_box(row: pd.Series) -> list[float]:
    return [float(row["pred_x1"]), float(row["pred_y1"]), float(row["pred_x2"]), float(row["pred_y2"])]


def _row_to_pred(row: pd.Series, score_col: str, source_suffix: str = "") -> dict[str, Any]:
    return {
        "box": _row_box(row),
        "score": float(row.get(score_col, 0.0)),
        "source": f"{score_col}{source_suffix}",
        "candidate_id": row.get("candidate_id", ""),
        "candidate_source": row.get("candidate_source", row.get("source_model", "")),
    }


def _distinct_enough(box: list[float], selected: list[dict[str, Any]], nms_iou: float) -> bool:
    return all(iou_xyxy(box, old["box"]) < nms_iou for old in selected)


def _max_k_for_cue(cue: dict[str, Any], params: dict[str, Any], finding: str) -> int:
    if finding == "Cardiomegaly":
        return 1
    cue_type = str(cue.get("multi_cue_type", "none"))
    has_multi = bool(cue.get("has_multi_cue", False))
    if not has_multi:
        return int(params.get("single_max_k", 1))
    k_hint = int(cue.get("k_hint", 2) or 2)
    if cue_type in DIFFUSE_CUE_TYPES:
        return min(k_hint, int(params.get("diffuse_max_k", 3)))
    if cue_type in SIDE_CUE_TYPES:
        return min(k_hint, int(params.get("side_max_k", 2)))
    return min(k_hint, int(params.get("max_k_if_cue", 2)))


def _score_threshold(top_score: float, cue: dict[str, Any], params: dict[str, Any]) -> tuple[float, float]:
    if bool(cue.get("has_multi_cue", False)):
        cue_type = str(cue.get("multi_cue_type", "none"))
        if cue_type in DIFFUSE_CUE_TYPES:
            return float(params.get("diffuse_score_ratio", 0.90)), float(params.get("diffuse_abs_score", -1e9))
        return float(params.get("side_score_ratio", 0.85)), float(params.get("side_abs_score", -1e9))
    return 1.0, -1e9


def _passes_threshold(score: float, top_score: float, cue: dict[str, Any], params: dict[str, Any]) -> bool:
    ratio, abs_thr = _score_threshold(top_score, cue, params)
    if score < abs_thr:
        return False
    if top_score > 0 and score < top_score * ratio:
        return False
    return True


def _hull_expansion_ok(selected: list[dict[str, Any]], box: list[float], cue: dict[str, Any], params: dict[str, Any], row: pd.Series) -> bool:
    if not selected:
        return True
    if not bool(cue.get("has_multi_cue", False)):
        return False
    cue_type = str(cue.get("multi_cue_type", "none"))
    old_hull = union_box([x["box"] for x in selected])
    new_hull = union_box([x["box"] for x in selected] + [box])
    width = float(row.get("image_width", 1.0) or 1.0)
    height = float(row.get("image_height", 1.0) or 1.0)
    image_area = max(width * height, 1.0)
    expansion = max(0.0, area_xyxy(new_hull) - area_xyxy(old_hull)) / image_area
    if cue_type in DIFFUSE_CUE_TYPES:
        limit = float(params.get("diffuse_hull_expansion", 0.55))
    else:
        limit = float(params.get("side_hull_expansion", 0.36))
    return expansion <= limit


def select_cue_aware_set(
    part: pd.DataFrame,
    score_col: str,
    cue: dict[str, Any],
    *,
    params: dict[str, Any],
    variant: str,
) -> list[dict[str, Any]]:
    """Select one or more candidate boxes for one phrase group."""

    if part.empty:
        return []
    part = part.sort_values(score_col, ascending=False).copy()
    finding = str(part.iloc[0].get("finding", ""))
    nms_iou = float(params.get("nms_iou", 0.4))
    max_k = _max_k_for_cue(cue, params, finding)
    top_score = float(part.iloc[0].get(score_col, 0.0))
    selected: list[dict[str, Any]] = []
    use_target = "sideaware" in variant or "conservative" in variant
    use_guard = "union_guard" in variant or "conservative" in variant

    if max_k <= 1 or not bool(cue.get("has_multi_cue", False)):
        row = part.iloc[0]
        pred = _row_to_pred(row, score_col, "::cue_single")
        pred.update({"cue_text": cue.get("cue_text", ""), "multi_cue_type": cue.get("multi_cue_type", "none"), "has_multi_cue": False})
        return [pred]

    if use_target and str(cue.get("multi_cue_type", "none")) in SIDE_CUE_TYPES:
        scan_topn = int(params.get("target_scan_topn", 30))
        target_weight = float(params.get("target_match_weight", 0.18))
        for target in cue.get("target_qs", [])[:max_k]:
            scan = part.head(scan_topn).copy()
            if scan.empty:
                continue
            scan["_target_match"] = scan.apply(lambda r: target_match_score(r, target), axis=1)
            scan["_target_score"] = scan[score_col].astype(float) + target_weight * scan["_target_match"].astype(float)
            for _, row in scan.sort_values("_target_score", ascending=False).iterrows():
                score = float(row.get(score_col, 0.0))
                box = _row_box(row)
                if not _passes_threshold(score, top_score, cue, params):
                    continue
                if not _distinct_enough(box, selected, nms_iou):
                    continue
                if use_guard and not _hull_expansion_ok(selected, box, cue, params, row):
                    continue
                pred = _row_to_pred(row, score_col, "::target")
                pred.update(
                    {
                        "target": json.dumps(target),
                        "target_match": float(target_match_score(row, target)),
                        "cue_text": cue.get("cue_text", ""),
                        "multi_cue_type": cue.get("multi_cue_type", "none"),
                        "has_multi_cue": True,
                    }
                )
                selected.append(pred)
                break

    for _, row in part.iterrows():
        if len(selected) >= max_k:
            break
        score = float(row.get(score_col, 0.0))
        box = _row_box(row)
        if not _passes_threshold(score, top_score, cue, params):
            continue
        if not _distinct_enough(box, selected, nms_iou):
            continue
        if use_guard and not _hull_expansion_ok(selected, box, cue, params, row):
            continue
        pred = _row_to_pred(row, score_col, "::fill")
        pred.update({"cue_text": cue.get("cue_text", ""), "multi_cue_type": cue.get("multi_cue_type", "none"), "has_multi_cue": True})
        selected.append(pred)

    if not selected:
        row = part.iloc[0]
        pred = _row_to_pred(row, score_col, "::fallback")
        pred.update({"cue_text": cue.get("cue_text", ""), "multi_cue_type": cue.get("multi_cue_type", "none"), "has_multi_cue": bool(cue.get("has_multi_cue", False))})
        selected.append(pred)
    return selected[:max_k]


def predict_cue_aware(
    df: pd.DataFrame,
    score_col: str,
    cue_info: dict[str, dict[str, Any]],
    *,
    params: dict[str, Any],
    variant: str,
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for qid, part in df.groupby("query_id", sort=False):
        cue = cue_info.get(str(qid), {"has_multi_cue": False, "multi_cue_type": "none", "k_hint": 1, "target_qs": []})
        out[str(qid)] = select_cue_aware_set(part, score_col, cue, params=params, variant=variant)
    return out


def default_selector_grids() -> dict[str, list[dict[str, Any]]]:
    base = {
        "single_max_k": 1,
        "side_max_k": 2,
        "diffuse_max_k": 3,
        "max_k_if_cue": 2,
        "nms_iou": 0.4,
        "target_scan_topn": 30,
        "target_match_weight": 0.18,
        "side_hull_expansion": 0.36,
        "diffuse_hull_expansion": 0.55,
        "side_abs_score": -1e9,
        "diffuse_abs_score": -1e9,
    }
    ratios_side = [0.75, 0.85, 0.90]
    ratios_diffuse = [0.85, 0.90, 0.95]
    nms_vals = [0.25, 0.4, 0.55]
    grids: dict[str, list[dict[str, Any]]] = {}
    grids["cue_cap_only"] = [
        {**base, "side_score_ratio": rs, "diffuse_score_ratio": rd, "nms_iou": nms}
        for rs in ratios_side
        for rd in ratios_diffuse
        for nms in nms_vals
    ]
    grids["cue_sideaware_selector"] = [
        {**base, "side_score_ratio": rs, "diffuse_score_ratio": rd, "nms_iou": nms, "target_match_weight": tw}
        for rs in ratios_side
        for rd in ratios_diffuse
        for nms in nms_vals
        for tw in [0.12, 0.18, 0.25]
    ]
    grids["cue_sideaware_union_guard"] = [
        {
            **base,
            "side_score_ratio": rs,
            "diffuse_score_ratio": rd,
            "nms_iou": nms,
            "target_match_weight": tw,
            "side_hull_expansion": se,
            "diffuse_hull_expansion": de,
        }
        for rs in ratios_side
        for rd in ratios_diffuse
        for nms in nms_vals
        for tw in [0.12, 0.18]
        for se in [0.28, 0.36, 0.44]
        for de in [0.45, 0.55, 0.65]
    ]
    grids["conservative_cue_rescue_multibox"] = [
        {
            **base,
            "side_score_ratio": rs,
            "diffuse_score_ratio": rd,
            "nms_iou": nms,
            "target_match_weight": tw,
            "side_hull_expansion": se,
            "diffuse_hull_expansion": de,
            "diffuse_max_k": dk,
        }
        for rs in [0.85, 0.90]
        for rd in [0.90, 0.95]
        for nms in [0.35, 0.5]
        for tw in [0.18, 0.25]
        for se in [0.24, 0.32]
        for de in [0.40, 0.50]
        for dk in [2, 3]
    ]
    return grids


def prediction_count_frame(preds: dict[str, list[dict[str, Any]]], name: str) -> pd.DataFrame:
    return pd.DataFrame([{"group_id": gid, f"{name}_pred_count": len(v)} for gid, v in preds.items()])


def add_cue_columns_to_eval(rows: pd.DataFrame, cue_info: dict[str, dict[str, Any]]) -> pd.DataFrame:
    out = rows.copy()
    out["has_multi_cue"] = out["group_id"].map(lambda x: bool(cue_info.get(str(x), {}).get("has_multi_cue", False)))
    out["multi_cue_type"] = out["group_id"].map(lambda x: str(cue_info.get(str(x), {}).get("multi_cue_type", "none")))
    out["k_hint"] = out["group_id"].map(lambda x: int(cue_info.get(str(x), {}).get("k_hint", 1)))
    out["target_qs"] = out["group_id"].map(lambda x: json.dumps(cue_info.get(str(x), {}).get("target_qs", []), ensure_ascii=False))
    return out

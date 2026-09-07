from __future__ import annotations

import itertools
import json
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from scripts import run_ms_cxr_multibox_phrase_grounding_v1 as mb

from .box_features import area_xyxy, iou_xyxy
from .cue_aware_set_selector import candidate_laterality, candidate_vertical, target_match_score, union_box
from .multibox_cue_parser import DIFFUSE_CUE_TYPES, SIDE_CUE_TYPES


def safe_boxes(value: Any) -> list[list[float]]:
    try:
        boxes = json.loads(value) if isinstance(value, str) else value
        return [[float(x) for x in box] for box in boxes]
    except Exception:
        return []


def pred_rows_to_map(rows: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    if rows.empty:
        return out
    pred_col = "pred_boxes_json" if "pred_boxes_json" in rows.columns else "pred_boxes"
    for _, row in rows.iterrows():
        gid = str(row["group_id"])
        out[gid] = [
            {"box": box, "score": float(row.get("score", 1.0) or 1.0), "source": str(row.get("method", "base"))}
            for box in safe_boxes(row.get(pred_col, "[]"))
        ]
    return out


def _box(row: pd.Series) -> list[float]:
    return [float(row["pred_x1"]), float(row["pred_y1"]), float(row["pred_x2"]), float(row["pred_y2"])]


def _candidate(row: pd.Series, score_col: str, source: str = "candidate") -> dict[str, Any]:
    return {
        "box": _box(row),
        "score": float(row.get(score_col, 0.0)),
        "source": source,
        "candidate_id": str(row.get("candidate_id", "")),
        "candidate_source": str(row.get("candidate_source", row.get("source_model", ""))),
    }


def _query_candidates(
    candidates: pd.DataFrame,
    qid: str,
    score_col: str,
    *,
    top_n: int = 30,
) -> list[dict[str, Any]]:
    if candidates.empty:
        return []
    part = candidates[candidates["query_id"].astype(str).eq(str(qid))].copy()
    if part.empty:
        return []
    part = part.sort_values(score_col, ascending=False).head(top_n)
    out = []
    for _, row in part.iterrows():
        cand = _candidate(row, score_col)
        cand["row"] = row
        out.append(cand)
    return out


def _eval_one(gt_boxes: list[list[float]], pred_boxes: list[list[float]]) -> dict[str, float]:
    return mb.eval_set(gt_boxes, pred_boxes)


def balanced_metric(score: dict[str, Any]) -> float:
    return (
        0.45 * float(score.get("union_iou", 0.0))
        + 0.40 * float(score.get("set_f1_0_3", 0.0))
        + 0.15 * float(score.get("coverage_mean_iou", 0.0))
    )


def _max_k(cue: dict[str, Any], finding: str) -> int:
    if finding == "Cardiomegaly":
        return 1
    if not bool(cue.get("has_multi_cue", False)):
        return 1
    cue_type = str(cue.get("multi_cue_type", "none"))
    k_hint = int(float(cue.get("k_hint", 1) or 1))
    if cue_type in DIFFUSE_CUE_TYPES:
        return min(max(k_hint, 2), 3)
    if cue_type in SIDE_CUE_TYPES:
        return min(max(k_hint, 2), 2)
    return min(max(k_hint, 1), 2)


def _candidate_conflicts_with_cue(cand: dict[str, Any], cue: dict[str, Any]) -> bool:
    if cand.get("row") is None:
        return False
    row = cand["row"]
    targets = cue.get("target_qs", []) or []
    if not targets:
        return False
    # For single/unilateral targets, avoid strong opposite-side replacements.
    if bool(cue.get("has_multi_cue", False)):
        return False
    target = targets[0]
    t_lat = str(target.get("laterality", "unknown"))
    t_vert = str(target.get("vertical", "unknown"))
    lat = candidate_laterality(float(row.get("box_cx_norm", 0.5)))
    vert = candidate_vertical(float(row.get("box_cy_norm", 0.5)))
    if t_lat in {"right", "left"} and lat in {"right", "left"} and lat != t_lat:
        return True
    if t_vert in {"apical", "upper"} and vert in {"lower", "basal"}:
        return True
    if t_vert in {"lower", "basal"} and vert in {"apical", "upper"}:
        return True
    return False


def _target_covered(box: list[float], target: dict[str, str], width: float, height: float) -> bool:
    cx = ((box[0] + box[2]) / 2.0) / max(width, 1.0)
    cy = ((box[1] + box[3]) / 2.0) / max(height, 1.0)
    lat = candidate_laterality(cx)
    vert = candidate_vertical(cy)
    t_lat = str(target.get("laterality", "unknown"))
    t_vert = str(target.get("vertical", "unknown"))
    if t_lat in {"right", "left"} and lat != t_lat:
        return False
    if t_vert in {"apical", "upper"} and vert not in {"apical", "upper"}:
        return False
    if t_vert in {"lower", "basal"} and vert not in {"lower", "basal"}:
        return False
    return True


def _base_score_for_box(
    box: list[float],
    cands: list[dict[str, Any]],
    default_score: float,
    *,
    match_iou: float = 0.5,
) -> float:
    best = default_score
    for cand in cands:
        if iou_xyxy(box, cand["box"]) >= match_iou:
            best = max(best, float(cand.get("score", 0.0)))
    return float(best)


def _hull_expansion_ratio(old_boxes: list[list[float]], new_boxes: list[list[float]], width: float, height: float) -> float:
    if not old_boxes or not new_boxes:
        return 1.0
    old = union_box(old_boxes)
    new = union_box(new_boxes)
    old_area = max(area_xyxy(old), 1e-8)
    image_area = max(float(width) * float(height), 1.0)
    raw = max(0.0, area_xyxy(new) - area_xyxy(old)) / image_area
    return max(raw, area_xyxy(new) / old_area)


def _union_guard_ok(
    old_boxes: list[list[float]],
    new_boxes: list[list[float]],
    cue: dict[str, Any],
    width: float,
    height: float,
    params: dict[str, Any],
) -> bool:
    if not old_boxes:
        return True
    cue_type = str(cue.get("multi_cue_type", "none"))
    if cue_type in DIFFUSE_CUE_TYPES:
        limit = float(params.get("diffuse_union_expansion_threshold", 2.2))
    elif bool(cue.get("has_multi_cue", False)):
        limit = float(params.get("multi_union_expansion_threshold", 1.8))
    else:
        limit = float(params.get("single_union_expansion_threshold", 1.25))
    return _hull_expansion_ratio(old_boxes, new_boxes, width, height) <= limit


def _distinct(box: list[float], boxes: list[list[float]], duplicate_iou: float) -> bool:
    return all(iou_xyxy(box, old) < duplicate_iou for old in boxes)


def _score_box_list(boxes: list[list[float]], cands: list[dict[str, Any]], default_score: float) -> list[tuple[list[float], float]]:
    return [(box, _base_score_for_box(box, cands, default_score)) for box in boxes]


def _prune_boxes(
    boxes: list[list[float]],
    cands: list[dict[str, Any]],
    cue: dict[str, Any],
    finding: str,
    params: dict[str, Any],
) -> tuple[list[list[float]], str]:
    max_k = _max_k(cue, finding)
    if len(boxes) <= max_k:
        return boxes, "keep"
    scored = _score_box_list(boxes, cands, float(params.get("base_default_score", 0.45)))
    kept = [box for box, _score in sorted(scored, key=lambda x: x[1], reverse=True)[:max_k]]
    return kept, "prune_extra"


def _replace_boxes(
    boxes: list[list[float]],
    cands: list[dict[str, Any]],
    cue: dict[str, Any],
    width: float,
    height: float,
    params: dict[str, Any],
) -> tuple[list[list[float]], str]:
    if not boxes or not cands:
        return boxes, "keep"
    margin = float(params.get("replace_margin", 0.16))
    ratio = float(params.get("replace_ratio", 1.15))
    abs_thr = float(params.get("replace_abs_threshold", 0.62))
    default = float(params.get("base_default_score", 0.45))
    scan_topn = int(params.get("replace_scan_topn", 20))
    out = list(boxes)
    changed = False
    for idx, base_box in enumerate(list(out)):
        # Estimate the score of the base box only from near-identical
        # candidates.  With the old 0.5 overlap threshold, a replacement
        # candidate often contributed to its own comparator and could never
        # exceed itself by the requested margin.
        base_score = _base_score_for_box(
            base_box,
            cands,
            default,
            match_iou=float(params.get("same_box_iou", 0.82)),
        )
        best: dict[str, Any] | None = None
        best_score = -1e9
        for cand in cands[:scan_topn]:
            box = cand["box"]
            score = float(cand.get("score", 0.0))
            if iou_xyxy(box, base_box) >= float(params.get("same_box_iou", 0.82)):
                continue
            if score < abs_thr:
                continue
            if score < base_score + margin and score < base_score * ratio:
                continue
            if _candidate_conflicts_with_cue(cand, cue):
                continue
            candidate_set = list(out)
            candidate_set[idx] = box
            if not _union_guard_ok(out, candidate_set, cue, width, height, params):
                continue
            if score > best_score:
                best = cand
                best_score = score
        if best is not None:
            out[idx] = best["box"]
            changed = True
    return out, "replace_one" if changed else "keep"


def _add_missing_side(
    boxes: list[list[float]],
    cands: list[dict[str, Any]],
    cue: dict[str, Any],
    finding: str,
    width: float,
    height: float,
    params: dict[str, Any],
) -> tuple[list[list[float]], str]:
    if not bool(cue.get("has_multi_cue", False)) or not cands:
        return boxes, "keep"
    max_k = _max_k(cue, finding)
    if len(boxes) >= max_k:
        return boxes, "keep"
    targets = cue.get("target_qs", []) or []
    if not targets:
        return boxes, "keep"
    top_score = max(float(c.get("score", 0.0)) for c in cands)
    abs_thr = float(params.get("add_abs_threshold", 0.66))
    ratio = float(params.get("add_score_ratio", 0.86))
    dup_iou = float(params.get("duplicate_iou_threshold", 0.45))
    target_min = float(params.get("target_match_min", 0.5))
    out = list(boxes)
    changed = False
    missing = [t for t in targets if not any(_target_covered(b, t, width, height) for b in out)]
    for target in missing:
        if len(out) >= max_k:
            break
        best: dict[str, Any] | None = None
        best_target_score = -1e9
        for cand in cands[: int(params.get("add_scan_topn", 30))]:
            score = float(cand.get("score", 0.0))
            if score < abs_thr or (top_score > 0 and score < top_score * ratio):
                continue
            box = cand["box"]
            if not _distinct(box, out, dup_iou):
                continue
            row = cand.get("row")
            if row is None:
                continue
            t_score = target_match_score(row, target)
            if t_score < target_min:
                continue
            candidate_set = out + [box]
            if not _union_guard_ok(out, candidate_set, cue, width, height, params):
                continue
            combined = score + float(params.get("target_weight", 0.18)) * t_score
            if combined > best_target_score:
                best = cand
                best_target_score = combined
        if best is not None:
            out.append(best["box"])
            changed = True
    return out, "add_missing_side" if changed else "keep"


def select_base_preserving(
    qid: str,
    base_boxes: list[list[float]],
    candidates: pd.DataFrame,
    cue: dict[str, Any],
    group: dict[str, Any],
    params: dict[str, Any],
    *,
    score_col: str,
    variant: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cands = _query_candidates(candidates, qid, score_col, top_n=int(params.get("candidate_topn", 30)))
    finding = str(group.get("finding", ""))
    width, height = float(group.get("image_width", 1.0)), float(group.get("image_height", 1.0))
    boxes = [list(map(float, b)) for b in base_boxes]
    actions: list[str] = []

    if variant in {"prune_overprediction", "base_preserving_full_rule", "base_preserving_full_router_optional"}:
        boxes, act = _prune_boxes(boxes, cands, cue, finding, params)
        if act != "keep":
            actions.append(act)

    if variant in {"keep_or_replace_only", "base_preserving_full_rule", "base_preserving_full_router_optional"}:
        old = boxes
        boxes, act = _replace_boxes(boxes, cands, cue, width, height, params)
        if act != "keep" and boxes != old:
            actions.append(act)

    if variant in {"add_missing_side_only", "base_preserving_full_rule", "base_preserving_full_router_optional"}:
        old = boxes
        boxes, act = _add_missing_side(boxes, cands, cue, finding, width, height, params)
        if act != "keep" and boxes != old:
            actions.append(act)

    if not boxes and cands:
        boxes = [cands[0]["box"]]
        actions.append("fallback_candidate")

    selected_sources: list[str] = []
    replacement_sources: list[str] = []
    for box in boxes:
        if any(iou_xyxy(box, base_box) >= 0.999 for base_box in base_boxes):
            selected_sources.append("base_preserved")
            continue
        matches = [cand for cand in cands if iou_xyxy(box, cand["box"]) >= 0.999]
        source = str(matches[0].get("candidate_source", "candidate")) if matches else "candidate_unmatched"
        selected_sources.append(source)
        replacement_sources.append(source)

    preds = [
        {
            "box": box,
            "score": _base_score_for_box(box, cands, float(params.get("base_default_score", 0.45))),
            "source": f"base_preserving::{'+'.join(actions) if actions else 'keep_base'}",
        }
        for box in boxes
    ]
    debug = {
        "group_id": qid,
        "action_taken": "+".join(actions) if actions else "keep_base",
        "base_pred_count": len(base_boxes),
        "new_pred_count": len(preds),
        "has_multi_cue": bool(cue.get("has_multi_cue", False)),
        "multi_cue_type": str(cue.get("multi_cue_type", "none")),
        "k_hint": int(cue.get("k_hint", 1) or 1),
        "selected_box_sources": "|".join(selected_sources),
        "replacement_candidate_sources": "|".join(replacement_sources),
    }
    return preds, debug


def predict_base_preserving(
    base_preds: dict[str, list[dict[str, Any]]],
    candidates: pd.DataFrame,
    cue_info: dict[str, dict[str, Any]],
    groups: dict[str, dict[str, Any]],
    params: dict[str, Any],
    *,
    score_col: str,
    variant: str,
) -> tuple[dict[str, list[dict[str, Any]]], pd.DataFrame]:
    out: dict[str, list[dict[str, Any]]] = {}
    audit: list[dict[str, Any]] = []
    for qid, group in groups.items():
        base_boxes = [p["box"] for p in base_preds.get(str(qid), [])]
        cue = cue_info.get(str(qid), {"has_multi_cue": False, "multi_cue_type": "none", "k_hint": 1, "target_qs": []})
        preds, debug = select_base_preserving(str(qid), base_boxes, candidates, cue, group, params, score_col=score_col, variant=variant)
        out[str(qid)] = preds
        audit.append(debug)
    return out, pd.DataFrame(audit)


def _best_action_record(
    action_name: str,
    group: dict[str, Any],
    pred_boxes: list[list[float]],
) -> dict[str, Any]:
    score = _eval_one(group["gt_boxes"], pred_boxes)
    return {"oracle_action": action_name, "pred_boxes": pred_boxes, "balanced_score": balanced_metric(score), **score}


def action_oracle_analysis(
    groups: dict[str, dict[str, Any]],
    base_preds: dict[str, list[dict[str, Any]]],
    candidates: pd.DataFrame,
    cue_info: dict[str, dict[str, Any]],
    *,
    score_col: str,
    top_n: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_rows: list[dict[str, Any]] = []
    by_action: dict[str, list[dict[str, Any]]] = {
        "keep_base": [],
        "oracle_replace_only": [],
        "oracle_replace_all_same_count": [],
        "oracle_add_missing_side": [],
        "oracle_prune_extra": [],
        "oracle_base_preserving_full": [],
    }
    for qid, group in groups.items():
        base_boxes = [p["box"] for p in base_preds.get(str(qid), [])]
        cands = _query_candidates(candidates, str(qid), score_col, top_n=top_n)
        cand_boxes = [c["box"] for c in cands]
        cue = cue_info.get(str(qid), {"has_multi_cue": False, "multi_cue_type": "none", "k_hint": 1, "target_qs": []})
        finding = str(group.get("finding", ""))
        max_k = _max_k(cue, finding)

        records: list[dict[str, Any]] = []
        records.append(_best_action_record("keep_base", group, base_boxes))

        replace_candidates = []
        if base_boxes and cand_boxes:
            for idx in range(len(base_boxes)):
                for cand_box in cand_boxes:
                    repl = list(base_boxes)
                    repl[idx] = cand_box
                    replace_candidates.append(_best_action_record("replace_one", group, repl))
        if replace_candidates:
            records.append(max(replace_candidates, key=lambda r: r["balanced_score"]) | {"oracle_action": "oracle_replace_only"})

        same_count_candidates = []
        same_k = max(1, len(base_boxes))
        for combo in itertools.combinations(cand_boxes[:top_n], min(same_k, len(cand_boxes))):
            same_count_candidates.append(_best_action_record("replace_all_same_count", group, [list(b) for b in combo]))
        if same_count_candidates:
            records.append(max(same_count_candidates, key=lambda r: r["balanced_score"]) | {"oracle_action": "oracle_replace_all_same_count"})

        add_candidates = []
        if bool(cue.get("has_multi_cue", False)) and len(base_boxes) < max_k:
            for cand_box in cand_boxes:
                add_candidates.append(_best_action_record("add_missing_side", group, base_boxes + [cand_box]))
        if add_candidates:
            records.append(max(add_candidates, key=lambda r: r["balanced_score"]) | {"oracle_action": "oracle_add_missing_side"})

        prune_candidates = []
        if len(base_boxes) > 1:
            for k in range(1, min(len(base_boxes), max_k) + 1):
                for combo in itertools.combinations(base_boxes, k):
                    prune_candidates.append(_best_action_record("prune_extra", group, [list(b) for b in combo]))
        if prune_candidates:
            records.append(max(prune_candidates, key=lambda r: r["balanced_score"]) | {"oracle_action": "oracle_prune_extra"})

        full_candidates = [r for r in records if r["oracle_action"] != "keep_base"]
        if full_candidates:
            records.append(max(full_candidates, key=lambda r: r["balanced_score"]) | {"oracle_action": "oracle_base_preserving_full"})

        for rec in records:
            rec = dict(rec)
            rec.update(
                {
                    "group_id": qid,
                    "finding": group.get("finding", ""),
                    "phrase": group.get("claim_sentence", ""),
                    "gold_count": len(group.get("gt_boxes", [])),
                    "base_pred_count": len(base_boxes),
                    "has_multi_cue": bool(cue.get("has_multi_cue", False)),
                    "multi_cue_type": str(cue.get("multi_cue_type", "none")),
                    "k_hint": int(cue.get("k_hint", 1) or 1),
                    "pred_count": len(rec.get("pred_boxes", [])),
                    "pred_boxes_json": json.dumps(rec.get("pred_boxes", [])),
                }
            )
            per_rows.append(rec)
            if rec["oracle_action"] in by_action:
                by_action[rec["oracle_action"]].append(rec)

    per = pd.DataFrame(per_rows)
    summaries = []
    for action, rows in by_action.items():
        if not rows:
            continue
        frame = pd.DataFrame(rows)
        summaries.append(
            {
                "oracle_action": action,
                "n_groups": int(len(frame)),
                "coverage_mean_iou": float(frame["coverage_mean_iou"].astype(float).mean()),
                "union_iou": float(frame["union_iou"].astype(float).mean()),
                "set_f1_0_3": float(frame["set_f1_0_3"].astype(float).mean()),
                "set_f1_0_5": float(frame["set_f1_0_5"].astype(float).mean()),
                "balanced_score": float(frame["balanced_score"].astype(float).mean()),
                "mean_pred_count": float(frame["pred_count"].astype(float).mean()),
            }
        )
    return pd.DataFrame(summaries), per


def action_distribution(audit: pd.DataFrame) -> pd.DataFrame:
    if audit.empty:
        return pd.DataFrame()
    counts = Counter(audit["action_taken"].astype(str))
    total = max(sum(counts.values()), 1)
    return pd.DataFrame(
        [{"action_taken": k, "n": int(v), "rate": float(v / total)} for k, v in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]
    )

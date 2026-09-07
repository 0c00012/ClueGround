from __future__ import annotations

import re
from typing import Any

import pandas as pd
from pandas.util import hash_pandas_object

from .box_features import iou_xyxy
from .cue_aware_set_selector import candidate_laterality, target_match_score
from .multibox_cue_parser import SIDE_CUE_TYPES, norm_text


# These cues explicitly describe more than one side, base, lobe, or focus.
# Patchy/diffuse wording alone is intentionally not a hard count instruction.
EXPLICIT_SIDE_CUES = SIDE_CUE_TYPES - {"upper_lobes", "lower_lobes"}
EXPLICIT_REGION_CUES = {"upper_lobes", "lower_lobes", "multifocal", "multifocal_precision"}
SOFT_DISTRIBUTION_CUES = {"diffuse", "widespread", "scattered"}
WEAK_DISTRIBUTION_CUES = {"patchy"}


def classify_cardinality_cue(cue: dict[str, Any], phrase: str | None) -> dict[str, Any]:
    """Turn a text-only cue into a gold-independent cardinality instruction."""

    cue_type = str(cue.get("multi_cue_type", "none"))
    text = norm_text(phrase)
    if not bool(cue.get("has_multi_cue", False)):
        return {
            "cardinality_cue_class": "none",
            "semantic_min_count": 1,
            "hard_count_floor": False,
            "target_mode": "none",
            "cardinality_reason": "no_multi_box_text_cue",
        }
    if cue_type in EXPLICIT_SIDE_CUES:
        return {
            "cardinality_cue_class": "explicit_side",
            "semantic_min_count": 2,
            "hard_count_floor": True,
            "target_mode": "side",
            "cardinality_reason": f"explicit_side_cue:{cue_type}",
        }
    if cue_type in EXPLICIT_REGION_CUES:
        return {
            "cardinality_cue_class": "explicit_region",
            "semantic_min_count": 2,
            "hard_count_floor": True,
            "target_mode": "generic",
            "cardinality_reason": f"explicit_multi_region_cue:{cue_type}",
        }
    if cue_type in SOFT_DISTRIBUTION_CUES:
        unilateral = bool(re.search(r"\b(right|left)\b", text)) and not bool(
            re.search(r"\b(bilateral|bilaterally|both|right\s+and\s+left|left\s+and\s+right)\b", text)
        )
        reason = "unilateral_distribution_can_use_one_box" if unilateral else "distribution_word_is_not_a_hard_box_count"
        return {
            "cardinality_cue_class": "soft_distribution",
            "semantic_min_count": 1,
            "hard_count_floor": False,
            "target_mode": "generic",
            "cardinality_reason": f"{reason}:{cue_type}",
        }
    if cue_type in WEAK_DISTRIBUTION_CUES:
        return {
            "cardinality_cue_class": "weak_distribution",
            "semantic_min_count": 1,
            "hard_count_floor": False,
            "target_mode": "generic",
            "cardinality_reason": f"weak_distribution_word:{cue_type}",
        }
    return {
        "cardinality_cue_class": "other_multi_cue",
        "semantic_min_count": 1,
        "hard_count_floor": False,
        "target_mode": "generic",
        "cardinality_reason": f"unclassified_multi_cue:{cue_type}",
    }


def prepare_candidate_index(
    candidates: pd.DataFrame,
    *,
    score_col: str,
    max_top_n: int = 60,
) -> dict[str, pd.DataFrame]:
    """Build a deterministic, score-sorted candidate index without gold columns."""

    out: dict[str, pd.DataFrame] = {}
    for qid, part in candidates.groupby("query_id", sort=False):
        work = part.copy()
        tie_columns = ["query_id", "pred_x1", "pred_y1", "pred_x2", "pred_y2"]
        work["_stable_tie_hash"] = hash_pandas_object(
            work[tie_columns].astype(str),
            index=False,
        ).astype("uint64")
        work = work.sort_values(
            [score_col, "_stable_tie_hash"],
            ascending=[False, True],
            kind="mergesort",
        ).head(max_top_n)
        out[str(qid)] = work
    return out


def _row_box(row: pd.Series) -> list[float]:
    return [float(row["pred_x1"]), float(row["pred_y1"]), float(row["pred_x2"]), float(row["pred_y2"])]


def _box_laterality(box: list[float], image_width: float) -> str:
    width = max(float(image_width), 1.0)
    return candidate_laterality(((float(box[0]) + float(box[2])) * 0.5) / width)


def _missing_side_targets(
    boxes: list[list[float]],
    cue: dict[str, Any],
    image_width: float,
) -> list[dict[str, str]]:
    covered = {_box_laterality(box, image_width) for box in boxes}
    targets = [
        target
        for target in cue.get("target_qs", []) or []
        if str(target.get("laterality", "unknown")) in {"right", "left"}
        and str(target.get("laterality")) not in covered
    ]
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, str]] = []
    for target in targets:
        key = (str(target.get("laterality", "unknown")), str(target.get("vertical", "unknown")))
        if key not in seen:
            seen.add(key)
            unique.append(target)
    return unique


def _best_addition(
    part: pd.DataFrame,
    boxes: list[list[float]],
    cue: dict[str, Any],
    decision: dict[str, Any],
    group: dict[str, Any],
    params: dict[str, Any],
    *,
    score_col: str,
) -> tuple[pd.Series | None, str]:
    if part.empty:
        return None, "no_candidates"
    scan = part.head(int(params.get("candidate_top_n", 40)))
    nms_iou = float(params.get("nms_iou", 0.35))
    width = float(group.get("image_width", 1.0) or 1.0)
    target_weight = float(params.get("target_weight", 0.18))
    targets = _missing_side_targets(boxes, cue, width) if decision.get("target_mode") == "side" else []

    ranked: list[tuple[float, int, pd.Series]] = []
    for _, row in scan.iterrows():
        box = _row_box(row)
        if any(iou_xyxy(box, old) >= nms_iou for old in boxes):
            continue
        target_score = 0.0
        if targets:
            target_score = max(target_match_score(row, target) for target in targets)
            if target_score < float(params.get("side_target_min", 0.5)):
                continue
        rank_score = float(row.get(score_col, 0.0)) + target_weight * target_score
        ranked.append((rank_score, int(row.get("_stable_tie_hash", 0)), row))
    if ranked:
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return ranked[0][2], "targeted_candidate" if targets else "distinct_candidate"

    # A hard semantic floor should still act when the proposal pool has no
    # candidate matching the parsed side. Fall back only to a distinct box.
    for _, row in scan.iterrows():
        box = _row_box(row)
        if all(iou_xyxy(box, old) < nms_iou for old in boxes):
            return row, "distinct_fallback"
    return None, "no_distinct_candidate"


def apply_unified_adaptive_cardinality(
    base_preds: dict[str, list[dict[str, Any]]],
    candidate_index: dict[str, pd.DataFrame],
    cue_info: dict[str, dict[str, Any]],
    groups: dict[str, dict[str, Any]],
    params: dict[str, Any],
    *,
    score_col: str,
) -> tuple[dict[str, list[dict[str, Any]]], pd.DataFrame]:
    """Apply one cue-conditioned count policy to any MS-CXR protocol view.

    The function never reads gold boxes or gold counts. Existing action/count
    predictions are preserved; only an explicit semantic count floor may add
    a distinct proposal.
    """

    out: dict[str, list[dict[str, Any]]] = {}
    audit_rows: list[dict[str, Any]] = []
    for gid, group in groups.items():
        gid = str(gid)
        base = [{**pred, "box": [float(x) for x in pred["box"]]} for pred in base_preds.get(gid, [])]
        preds = list(base)
        cue = cue_info.get(gid, {"has_multi_cue": False, "multi_cue_type": "none", "k_hint": 1})
        decision = classify_cardinality_cue(cue, str(group.get("claim_sentence", "")))
        semantic_floor = int(decision["semantic_min_count"])
        target_count = max(len(preds), semantic_floor) if bool(decision["hard_count_floor"]) else len(preds)
        target_count = min(target_count, int(params.get("max_pred_count", 3)))
        additions: list[str] = []
        part = candidate_index.get(gid, pd.DataFrame())
        while len(preds) < target_count:
            row, reason = _best_addition(
                part,
                [pred["box"] for pred in preds],
                cue,
                decision,
                group,
                params,
                score_col=score_col,
            )
            if row is None:
                additions.append(reason)
                break
            preds.append(
                {
                    "box": _row_box(row),
                    "score": float(row.get(score_col, 0.0)),
                    "source": f"unified_adaptive_cardinality::{reason}",
                    "candidate_id": str(row.get("candidate_id", "")),
                    "candidate_source": str(row.get("candidate_source", row.get("source_model", ""))),
                }
            )
            additions.append(reason)
        out[gid] = preds
        changed = len(preds) != len(base) or any(
            pred.get("box") != base[idx].get("box") for idx, pred in enumerate(preds[: len(base)])
        )
        audit_rows.append(
            {
                "group_id": gid,
                "has_multi_cue": bool(cue.get("has_multi_cue", False)),
                "multi_cue_type": str(cue.get("multi_cue_type", "none")),
                "k_hint": int(cue.get("k_hint", 1) or 1),
                **decision,
                "base_pred_count": len(base),
                "target_pred_count": target_count,
                "new_pred_count": len(preds),
                "count_floor_satisfied": len(preds) >= target_count,
                "changed": bool(changed),
                "action_taken": "+".join(additions) if additions else "keep_base",
            }
        )
    return out, pd.DataFrame(audit_rows)


def default_policy_grid() -> list[dict[str, Any]]:
    return [
        {
            "nms_iou": nms_iou,
            "target_weight": target_weight,
            "side_target_min": side_target_min,
            "candidate_top_n": top_n,
            "max_pred_count": 3,
        }
        for nms_iou in [0.25, 0.35, 0.45, 0.55]
        for target_weight in [0.0, 0.12, 0.25]
        for side_target_min in [0.5, 0.75]
        # Search depth is fixed rather than tuned: explicit cardinality cues
        # must be given the full proposal pool, even when that costs a set
        # metric on singleton annotations.
        for top_n in [400]
    ]

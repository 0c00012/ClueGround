#!/usr/bin/env python
"""Controlled ClueGround visual-component ablation on direct-888 and 1444.

Every arm reuses the same seed-paired 640-pixel YOLO proposal tables, phrase
groups, rule-context cardinality decoder, NMS, and evaluator.  Only the visual
evidence used to rank a candidate changes.  The RAD-DINO-only arm is the one
exception: it emits the standalone phrase-conditioned RAD-DINO proposal and is
therefore explicitly a one-box diagnostic on 1444.
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

from scripts import clueground_siglip_analysis_common_v1 as common  # noqa: E402
from scripts import run_clueground_exact_hybrid_v4_direct_3seed_v1 as exact  # noqa: E402
from scripts import run_ms_cxr_multibox_rule_context_fusion_v4_yolov8l_pool as hybrid_v4  # noqa: E402
from scripts import run_ms_cxr_singlebox_fair_yolo_dino_fusion_v1 as single_fusion  # noqa: E402


SEEDS = (13, 42, 2026)
ARMS = (
    "yolo_only",
    "rad_dino_only",
    "yolo_siglip",
    "rad_dino_siglip_over_yolo_proposals",
    "yolo_rad_dino_context",
    "full_yolo_rad_dino_siglip_context",
)
OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "clueground_ablation_study_v1" / "architecture"
TARGET_ROOT = PROJECT_ROOT / "experiments" / "clueground_siglip_analysis_v1" / "predictions" / "target_siglip_1444"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def ranks(values: list[float]) -> np.ndarray:
    if not values:
        return np.asarray([], dtype=np.float64)
    return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=np.float64)


def box_iou(a: list[float], b: list[float]) -> float:
    return float(hybrid_v4.base.iou_xyxy(a, b))


def valid_candidate(candidate: dict[str, Any]) -> bool:
    box = np.asarray(candidate["box"], dtype=np.float64)
    return bool(
        box.shape == (4,) and np.isfinite(box).all()
        and box[0] < box[2] and box[1] < box[3]
    )


def dino_box(resources: common.Resources, group: dict[str, Any], split: str) -> list[float] | None:
    value = resources.context.dino[split].get(str(group["group_id"]))
    if value is None:
        return None
    return single_fusion.old_fusion.norm_to_xyxy(
        np.asarray(value, dtype=np.float64),
        float(group["image_width"]),
        float(group["image_height"]),
    )


def position(box: list[float], width: float, height: float) -> tuple[str, str]:
    cx = ((box[0] + box[2]) / 2.0) / max(width, 1.0)
    cy = ((box[1] + box[3]) / 2.0) / max(height, 1.0)
    laterality = "right" if cx < 0.47 else ("left" if cx > 0.53 else "center")
    if cy < 0.25:
        vertical = "apical"
    elif cy < 0.45:
        vertical = "upper"
    elif cy < 0.68:
        vertical = "mid"
    elif cy < 0.85:
        vertical = "lower"
    else:
        vertical = "basal"
    return laterality, vertical


def target_bonus(candidate: dict[str, Any], query: dict[str, str], group: dict[str, Any]) -> float:
    laterality, vertical = position(
        candidate["box"], float(group["image_width"]), float(group["image_height"])
    )
    wanted_side = str(query.get("laterality", "unknown"))
    wanted_vertical = str(query.get("vertical", "unknown"))
    bonus = 0.0
    if wanted_side in {"left", "right"}:
        bonus += 1.0 if laterality == wanted_side else -1.0
    if wanted_vertical in {"apical", "upper"}:
        bonus += 0.5 if vertical in {"apical", "upper"} else -0.25
    elif wanted_vertical in {"lower", "basal"}:
        bonus += 0.5 if vertical in {"lower", "basal"} else -0.25
    elif wanted_vertical in {"mid", "middle"}:
        bonus += 0.35 if vertical == "mid" else -0.15
    return bonus


def target_scores(seed: int, split: str) -> dict[tuple[str, int, tuple[Any, ...]], float]:
    if split not in {"val", "eval"}:
        return {}
    path = TARGET_ROOT / f"seed_{seed}" / f"{split}_target_siglip_scores.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    return {
        (str(row["group_id"]), int(row["target_index"]), common.candidate_key(row)): float(row["siglip_target_raw"])
        for row in frame.to_dict("records")
    }


def visual_scores(
    resources: common.Resources,
    group: dict[str, Any],
    split: str,
    candidates: list[dict[str, Any]],
    arm: str,
    query: dict[str, str],
    target_index: int | None,
    target_lookup: dict[tuple[str, int, tuple[Any, ...]], float],
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    copied = [dict(row) for row in candidates]
    confidence = ranks([float(row["score"]) for row in copied])
    dino = dino_box(resources, group, split)
    agreement = ranks([box_iou(row["box"], dino) if dino is not None else 0.0 for row in copied])
    semantic_raw = []
    for row in copied:
        value = float(row["_row"].get("siglip_raw", 0.0))
        if target_index is not None:
            value = target_lookup.get(
                (str(group["group_id"]), int(target_index), common.candidate_key(row["_row"])),
                value,
            )
        semantic_raw.append(value)
    semantic = ranks(semantic_raw)

    if arm == "yolo_only":
        values = confidence
    elif arm == "yolo_siglip":
        values = semantic
    elif arm == "rad_dino_siglip_over_yolo_proposals":
        values = 0.5 * agreement + 0.5 * semantic
    elif arm in {"yolo_rad_dino_context", "full_yolo_rad_dino_siglip_context"}:
        geometric = common._score_geometric(
            resources, group, query, copied, multi=bool(group["is_multi"]), no_prior=False
        )
        geometric_by_key = {
            common.candidate_key(row["_row"]): float(row["score"])
            for row in geometric
        }
        geometric_rank = ranks([
            geometric_by_key.get(common.candidate_key(row["_row"]), 0.0)
            for row in copied
        ])
        values = geometric_rank
        if arm == "full_yolo_rad_dino_siglip_context":
            values = 0.75 * geometric_rank + 0.25 * semantic
    else:
        raise ValueError(arm)

    for row, value in zip(copied, values):
        row["score"] = float(value)
    return sorted(copied, key=lambda row: (float(row["score"]), -int(row.get("rank", 9999))), reverse=True)


def add_distinct(selected: list[dict[str, Any]], candidate: dict[str, Any], threshold: float) -> bool:
    if any(box_iou(old["box"], candidate["box"]) >= threshold for old in selected):
        return False
    selected.append(candidate)
    return True


def predict_arm(
    resources: common.Resources,
    split: str,
    arm: str,
) -> tuple[dict[str, list[list[float]]], pd.DataFrame]:
    groups = exact.make_groups(resources.context, split)
    frame = resources.candidates[split]
    by_group = {
        str(group_id): part.copy()
        for group_id, part in frame.groupby(frame["group_id"].astype(str), sort=False)
    }
    target_lookup = target_scores(resources.seed, split) if resources.protocol == "1444" else {}
    outputs: dict[str, list[list[float]]] = {}
    audit_rows: list[dict[str, Any]] = []
    for group_id, group in groups.items():
        group = dict(group)
        group["group_id"] = group_id
        group["split"] = split
        base_query = single_fusion.ybase.parse_rule_context(hybrid_v4.group_row(group))
        cue = hybrid_v4.context_cues(str(group["claim_sentence"]), str(group["finding"]), base_query)
        group["is_multi"] = bool(cue["has_multi_cue"])
        direct_dino = dino_box(resources, group, split)
        if arm == "rad_dino_only":
            boxes = [direct_dino] if direct_dino is not None else []
            outputs[group_id] = boxes
            audit_rows.append({
                "protocol": resources.protocol, "seed": resources.seed, "split": split,
                "arm": arm, "group_id": group_id, "n_candidates": 0,
                "n_pred": len(boxes), "multi_cue": bool(cue["has_multi_cue"]),
                "fallback": int(not boxes), "yolo_proposals_used": False,
            })
            continue

        part = by_group.get(group_id, pd.DataFrame())
        candidates = (
            [candidate for candidate in common._candidate_dicts(part) if valid_candidate(candidate)]
            if not part.empty else []
        )
        selected: list[dict[str, Any]] = []
        nms_iou = float(resources.set_params["nms_iou"])
        if not cue["has_multi_cue"]:
            ranked = visual_scores(
                resources, group, split, candidates, arm, base_query, None, target_lookup
            )
            selected = ranked[:1]
        else:
            max_k = max(int(cue["k_hint"]), int(resources.set_params["min_k_if_cue"]))
            max_k = min(max_k, int(resources.set_params["max_k_if_cue"]))
            for target_index, query in enumerate(cue["target_qs"]):
                ranked = visual_scores(
                    resources, group, split, candidates, arm, query, target_index, target_lookup
                )
                ranked = sorted(
                    ranked,
                    key=lambda row: float(row["score"]) + 0.20 * target_bonus(row, query, group),
                    reverse=True,
                )
                for candidate in ranked[: int(resources.set_params["target_scan_topn"])]:
                    if add_distinct(selected, candidate, nms_iou):
                        break
            ranked = visual_scores(
                resources, group, split, candidates, arm, base_query, None, target_lookup
            )
            for candidate in ranked:
                if len(selected) >= max_k:
                    break
                add_distinct(selected, candidate, nms_iou)
            if not selected:
                selected = ranked[:1]

        fallback = False
        if not selected:
            source = resources.upstream.hybrid[split].get(group_id, [])
            boxes = [[float(value) for value in row["box"]] for row in source]
            fallback = True
        elif not cue["has_multi_cue"] and arm in {
            "yolo_rad_dino_context", "full_yolo_rad_dino_siglip_context"
        }:
            boxes = [common._single_final_box(resources, group, selected[0], no_prior=False)]
        else:
            boxes = [[float(value) for value in row["box"]] for row in selected]
        outputs[group_id] = boxes
        audit_rows.append({
            "protocol": resources.protocol, "seed": resources.seed, "split": split,
            "arm": arm, "group_id": group_id, "n_candidates": len(candidates),
            "n_pred": len(boxes), "multi_cue": bool(cue["has_multi_cue"]),
            "fallback": int(fallback), "yolo_proposals_used": True,
        })
    return outputs, pd.DataFrame(audit_rows)


def run() -> None:
    for directory in ("audit", "metrics", "predictions"):
        (OUTPUT_ROOT / directory).mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_ROOT / "METHOD_DEFINITIONS.json", {
        "seeds": list(SEEDS),
        "candidate_resolution": 640,
        "shared": "seed-paired YOLO candidates, raw phrase parser, cardinality/NMS decoder, evaluator",
        "arms": {
            "yolo_only": "YOLO confidence rank; no RAD-DINO or SigLIP evidence",
            "rad_dino_only": "standalone phrase-conditioned RAD-DINO proposal; one-box diagnostic",
            "yolo_siglip": "YOLO proposal coordinates ranked only by frozen SigLIP crop-phrase score",
            "rad_dino_siglip_over_yolo_proposals": "YOLO coordinates used only as proposals; rank is equal RAD-DINO agreement and SigLIP",
            "yolo_rad_dino_context": "full geometric/context branch with SigLIP removed; exact full-minus-SigLIP ablation",
            "full_yolo_rad_dino_siglip_context": "historical geometry/context score plus frozen SigLIP rank; 75/25 fixed blend",
        },
        "eval_selection": "none; all component weights fixed before this run",
    })
    rows: list[dict[str, Any]] = []
    split_contexts = []
    for protocol in ("888", "1444"):
        for seed in SEEDS:
            resources = common.load_resources(protocol, seed)
            split_contexts.append(resources.context)
            for arm in ARMS:
                outputs, audit = predict_arm(resources, "eval", arm)
                summary, detail = common.evaluate(resources, outputs, "eval")
                if int(summary["n"]) != (163 if protocol == "888" else 220):
                    raise RuntimeError(f"Denominator mismatch: {protocol}/{seed}/{arm}")
                root = OUTPUT_ROOT / "predictions" / protocol / f"seed_{seed}"
                common.save_prediction_map(root / f"{arm}.json", outputs)
                audit.to_csv(root / f"{arm}_audit.csv", index=False)
                detail.to_csv(root / f"{arm}_per_group.csv", index=False)
                rows.append({"protocol": protocol, "seed": seed, "arm": arm, **summary})
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(OUTPUT_ROOT / "metrics" / "architecture_per_seed.csv", index=False)
    aggregate = common.aggregate(per_seed.rename(columns={"arm": "method"}))
    aggregate.to_csv(OUTPUT_ROOT / "metrics" / "architecture_aggregate.csv", index=False)
    write_json(OUTPUT_ROOT / "audit" / "SPLIT_OVERLAP_AUDIT.json", exact.split_audit(split_contexts))
    write_json(OUTPUT_ROOT / "FINAL_STATUS.json", {
        "status": "complete",
        "n_expected_runs": 36,
        "n_completed_runs": int(len(per_seed)),
        "seeds": list(SEEDS),
        "denominators": {"888": 163, "1444_groups": 220, "1444_boxes": 280},
        "missing_predictions": int(per_seed["n_missing_predictions"].sum()),
    })
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    run()

#!/usr/bin/env python
"""Repeat the ClueGround context ablation without any SigLIP input.

This runner reuses the original context encoders, scorer/count architecture,
loss, decoder, and seed schedule.  It reads the pre-SigLIP semantic-base YOLO
candidate tables and removes ``siglip_rank`` from the visual feature schema.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import clueground_siglip_analysis_common_v1 as common  # noqa: E402
from scripts import run_clueground_context_ablation_v1 as original  # noqa: E402
from scripts import run_clueground_exact_hybrid_v4_direct_3seed_v1 as exact  # noqa: E402
from scripts import run_ms_cxr_singlebox_fair_yolo_dino_fusion_v1 as single_fusion  # noqa: E402


OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "clueground_context_ablation_no_siglip_v2"
OLD_EMBEDDINGS = (
    PROJECT_ROOT / "experiments" / "clueground_ablation_study_v1" / "context" / "embeddings"
)
FEATURE_NAMES = (
    "cx",
    "cy",
    "w",
    "h",
    "area",
    "confidence",
    "within_source_rank",
    "rad_dino_iou",
    *[f"source_{name}" for name in original.SOURCE_MODELS],
)


def no_siglip_resources(protocol: str, seed: int, **_: Any) -> common.Resources:
    resources = REAL_LOAD_RESOURCES(protocol, seed, require_scores=False)
    cache_root = common.siglip_cache_root(protocol, seed)
    frames: dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "eval"):
        path = cache_root / f"{split}_semantic_base_candidates.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        forbidden = [column for column in frame.columns if "siglip" in column.lower()]
        if forbidden:
            raise RuntimeError(f"SigLIP columns in semantic-base table {path}: {forbidden}")
        frames[split] = frame
    resources.candidates = frames
    return resources


def build_group_data_no_siglip(resources: common.Resources, split: str) -> original.GroupData:
    groups = exact.make_groups(resources.context, split)
    frame = resources.candidates[split]
    forbidden = [column for column in frame.columns if "siglip" in column.lower()]
    if forbidden:
        raise RuntimeError(f"SigLIP features survived in {resources.protocol}/{resources.seed}/{split}")
    by_group = {
        str(group_id): part.copy()
        for group_id, part in frame.groupby(frame["group_id"].astype(str), sort=False)
    }
    group_ids = list(groups)
    max_candidates = 12
    visual = np.zeros((len(group_ids), max_candidates, len(FEATURE_NAMES)), dtype=np.float32)
    quality = np.zeros((len(group_ids), max_candidates), dtype=np.float32)
    mask = np.zeros((len(group_ids), max_candidates), dtype=bool)
    counts = np.zeros(len(group_ids), dtype=np.int64)
    all_boxes: list[list[list[float]]] = []

    for group_index, group_id in enumerate(group_ids):
        group = dict(groups[group_id])
        group["group_id"] = group_id
        part = by_group.get(group_id, pd.DataFrame())
        candidates = (
            [
                candidate
                for candidate in common._candidate_dicts(part)
                if original.valid_candidate(candidate)
            ][:max_candidates]
            if not part.empty
            else []
        )
        boxes = [[float(value) for value in candidate["box"]] for candidate in candidates]
        all_boxes.append(boxes)
        gt_boxes = [[float(value) for value in box] for box in group["gt_boxes"]]
        counts[group_index] = min(max(len(gt_boxes), 1), 4) - 1
        width = float(group["image_width"])
        height = float(group["image_height"])
        dino = resources.context.dino[split].get(group_id)
        dino_xyxy = None
        if dino is not None:
            dino_xyxy = single_fusion.old_fusion.norm_to_xyxy(
                np.asarray(dino), width, height
            )

        for candidate_index, candidate in enumerate(candidates):
            x1, y1, x2, y2 = candidate["box"]
            cx = (x1 + x2) / (2 * width)
            cy = (y1 + y2) / (2 * height)
            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height
            row = candidate["_row"]
            source = str(candidate.get("source_model", ""))
            features = [
                cx,
                cy,
                box_width,
                box_height,
                box_width * box_height,
                float(row.get("confidence", 0.0)),
                1.0 / (1.0 + max(float(row.get("rank", 0.0)), 0.0)),
                original.box_iou(candidate["box"], dino_xyxy) if dino_xyxy is not None else 0.0,
                *[float(source == model) for model in original.SOURCE_MODELS],
            ]
            visual[group_index, candidate_index] = np.asarray(features, dtype=np.float32)
            mask[group_index, candidate_index] = True
            quality[group_index, candidate_index] = max(
                (original.box_iou(candidate["box"], gt) for gt in gt_boxes),
                default=0.0,
            )
    return original.GroupData(group_ids, visual, mask, quality, counts, all_boxes)


def patched_write_json(path: Path, value: Any) -> None:
    payload = dict(value) if isinstance(value, dict) else value
    if isinstance(payload, dict) and path.name == "MODEL_CONFIG.json":
        payload.update(
            {
                "visual_features": list(FEATURE_NAMES),
                "fixed_visual_signals": ["YOLO confidence", "RAD-DINO agreement"],
                "candidate_pool": "seed-paired YOLO-640 top-12 semantic-base pool",
                "siglip_used": False,
                "biomedclip_used": False,
                "difference_from_v1": "siglip_rank removed; pre-SigLIP candidate tables loaded",
            }
        )
    if isinstance(payload, dict) and path.name == "FINAL_STATUS.json":
        payload.update(
            {
                "siglip_used": False,
                "biomedclip_used": False,
                "siglip_feature_count": 0,
            }
        )
    REAL_WRITE_JSON(path, payload)


def write_pre_run_audit() -> None:
    rows = []
    for protocol in ("888", "1444"):
        for seed in original.SEEDS:
            cache_root = common.siglip_cache_root(protocol, seed)
            for split in ("train", "val", "eval"):
                path = cache_root / f"{split}_semantic_base_candidates.csv"
                frame = pd.read_csv(path, nrows=1)
                rows.append(
                    {
                        "protocol": protocol,
                        "seed": seed,
                        "split": split,
                        "source_path": str(path.resolve()),
                        "siglip_columns": sum("siglip" in column.lower() for column in frame.columns),
                    }
                )
    audit = pd.DataFrame(rows)
    (OUTPUT_ROOT / "audit").mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUTPUT_ROOT / "audit" / "NO_SIGLIP_CANDIDATE_SOURCES.csv", index=False)
    if int(audit["siglip_columns"].sum()) != 0:
        raise RuntimeError("SigLIP column detected during pre-run audit")


REAL_LOAD_RESOURCES = common.load_resources
REAL_WRITE_JSON = original.write_json


def main() -> None:
    if OLD_EMBEDDINGS.exists() and not (OUTPUT_ROOT / "embeddings").exists():
        shutil.copytree(OLD_EMBEDDINGS, OUTPUT_ROOT / "embeddings")
    write_pre_run_audit()
    original.OUTPUT_ROOT = OUTPUT_ROOT
    original.FEATURE_NAMES = FEATURE_NAMES
    original.build_group_data = build_group_data_no_siglip
    original.write_json = patched_write_json
    common.load_resources = no_siglip_resources
    try:
        original.run()
    finally:
        common.load_resources = REAL_LOAD_RESOURCES
        original.write_json = REAL_WRITE_JSON


if __name__ == "__main__":
    main()

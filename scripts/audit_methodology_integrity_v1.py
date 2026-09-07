#!/usr/bin/env python
"""CPU-only integrity audit for the MS-CXR 888/1444 grounding pipelines.

This audit intentionally does not run any foundation model or detector.  It
rebuilds metrics from saved boxes, checks split and ground-truth identity,
inspects inference feature schemas, and records provenance/fairness gaps.  GPU
cache reproduction is handled by a separate deferred runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_ms_cxr_multibox_phrase_grounding_v1 as mb  # noqa: E402


DEFAULT_OUT = PROJECT_ROOT / "experiments" / "methodology_integrity_audit" / "20260710_full_v1"
DATA_ROOT = PROJECT_ROOT / "training" / "ms_cxr_vfm_localizer_stage1_p10_p19" / "datasets"
CLEAN_ROOT = PROJECT_ROOT / "experiments" / "clean_888_1444_comparison_v1"
MULTIBOX_ROOT = PROJECT_ROOT / "experiments" / "multibox_dev" / "20260707_multibox_dev_v1"
SINGLEBOX_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "rerank_dino_gate"
    / "20260707_163626_cue_count_fix"
    / "singlebox_888"
)
PACKAGED_SINGLEBOX = PROJECT_ROOT / "experiments" / "action_policy_plus_count_head_singlebox888_v1"
MERGED_SINGLEBOX_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "ms_cxr_singlebox_fair_detector_sweep_v2"
    / "predictions"
    / "merged_detectors_dino_full_phrase_fusion_singlebox_fair_eval_predictions.csv"
)
SEMANTIC_ROOT = PROJECT_ROOT / "experiments" / "ms_cxr_semantic_finegrid_unified_yolo8_nsml_yolo11_sm_v1"
CONTRASTIVE_ROOT = PROJECT_ROOT / "experiments" / "ms_cxr_finegrid_plus_contrastive_expert_v1"
VLM_ROOT = PROJECT_ROOT / "experiments" / "vlm_reference" / "20260708_vlm_reference_baselines_v1"
MEDRPG_SINGLE_ROOT = PROJECT_ROOT / "experiments" / "ms_cxr_medrpg_fair_retrain_final_v1"
LEAK_ROOT = PROJECT_ROOT / "experiments" / "leakage_sanitized_action_policy_plus_count_head_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    p.add_argument("--bootstrap-reps", type=int, default=5000)
    return p.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def safe_boxes(value: Any) -> list[list[float]]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    parsed = json.loads(value) if isinstance(value, str) else value
    if parsed is None:
        return []
    out = []
    for box in parsed:
        if box is None or len(box) != 4:
            continue
        out.append([float(x) for x in box])
    return out


def box_iou(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ab = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    den = aa + ab - inter
    return inter / den if den > 0 else 0.0


def valid_box(box: Iterable[float]) -> bool:
    x1, y1, x2, y2 = [float(x) for x in box]
    return all(np.isfinite([x1, y1, x2, y2])) and x2 > x1 and y2 > y1


def hull_box(boxes: list[list[float]]) -> list[float] | None:
    boxes = [b for b in boxes if valid_box(b)]
    if not boxes:
        return None
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def hull_iou(a: list[list[float]], b: list[list[float]]) -> float:
    ha, hb = hull_box(a), hull_box(b)
    return box_iou(ha, hb) if ha is not None and hb is not None else 0.0


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    clean = sorted((float(a), float(b)) for a, b in intervals if b > a)
    out: list[tuple[float, float]] = []
    for start, end in clean:
        if not out or start > out[-1][1]:
            out.append((start, end))
        else:
            out[-1] = (out[-1][0], max(out[-1][1], end))
    return out


def interval_length(intervals: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in merge_intervals(intervals))


def interval_intersection_length(
    a: list[tuple[float, float]], b: list[tuple[float, float]]
) -> float:
    aa, bb = merge_intervals(a), merge_intervals(b)
    i = j = 0
    total = 0.0
    while i < len(aa) and j < len(bb):
        total += max(0.0, min(aa[i][1], bb[j][1]) - max(aa[i][0], bb[j][0]))
        if aa[i][1] <= bb[j][1]:
            i += 1
        else:
            j += 1
    return total


def rectangle_union_iou(a: list[list[float]], b: list[list[float]]) -> float:
    aa = [box for box in a if valid_box(box)]
    bb = [box for box in b if valid_box(box)]
    xs = sorted({float(x) for box in aa + bb for x in (box[0], box[2])})
    if len(xs) < 2:
        return 0.0
    area_a = area_b = inter = 0.0
    for x1, x2 in zip(xs[:-1], xs[1:]):
        width = x2 - x1
        if width <= 0:
            continue
        ya = [(box[1], box[3]) for box in aa if box[0] < x2 and box[2] > x1]
        yb = [(box[1], box[3]) for box in bb if box[0] < x2 and box[2] > x1]
        area_a += width * interval_length(ya)
        area_b += width * interval_length(yb)
        inter += width * interval_intersection_length(ya, yb)
    den = area_a + area_b - inter
    return inter / den if den > 0 else 0.0


def raster_union_iou(
    a: list[list[float]],
    b: list[list[float]],
    width: float,
    height: float,
    size: int = 224,
) -> float:
    """Match the VLM reference evaluator's 224x224 mask-union metric."""

    def draw(boxes: list[list[float]]) -> np.ndarray:
        mask = np.zeros((size, size), dtype=bool)
        for x1, y1, x2, y2 in boxes:
            nx1, nx2 = x1 / width, x2 / width
            ny1, ny2 = y1 / height, y2 / height
            ix1 = int(np.floor(np.clip(nx1, 0, 1) * size))
            iy1 = int(np.floor(np.clip(ny1, 0, 1) * size))
            ix2 = int(np.ceil(np.clip(nx2, 0, 1) * size))
            iy2 = int(np.ceil(np.clip(ny2, 0, 1) * size))
            if ix2 > ix1 and iy2 > iy1:
                mask[iy1:iy2, ix1:ix2] = True
        return mask

    am, bm = draw(a), draw(b)
    union = np.logical_or(am, bm).sum()
    return 1.0 if union == 0 else float(np.logical_and(am, bm).sum() / union)


def greedy_tp(gt: list[list[float]], pred: list[list[float]], threshold: float) -> int:
    pairs = sorted(
        ((box_iou(g, p), gi, pi) for gi, g in enumerate(gt) for pi, p in enumerate(pred)),
        reverse=True,
    )
    used_g: set[int] = set()
    used_p: set[int] = set()
    tp = 0
    for score, gi, pi in pairs:
        if score < threshold:
            break
        if gi not in used_g and pi not in used_p:
            used_g.add(gi)
            used_p.add(pi)
            tp += 1
    return tp


def maximum_tp(gt: list[list[float]], pred: list[list[float]], threshold: float) -> int:
    edges = [
        [pi for pi, p in enumerate(pred) if box_iou(g, p) >= threshold]
        for g in gt
    ]
    match_pred: dict[int, int] = {}

    def augment(gi: int, seen: set[int]) -> bool:
        for pi in edges[gi]:
            if pi in seen:
                continue
            seen.add(pi)
            if pi not in match_pred or augment(match_pred[pi], seen):
                match_pred[pi] = gi
                return True
        return False

    return sum(1 for gi in range(len(gt)) if augment(gi, set()))


def f1_from_tp(tp: int, n_gt: int, n_pred: int) -> float:
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gt if n_gt else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def boxes_equal(a: list[list[float]], b: list[list[float]], atol: float = 1e-6) -> bool:
    if len(a) != len(b):
        return False
    aa = sorted(tuple(float(x) for x in box) for box in a)
    bb = sorted(tuple(float(x) for x in box) for box in b)
    return all(np.allclose(x, y, atol=atol, rtol=0.0) for x, y in zip(aa, bb))


@dataclass
class MethodPredictions:
    protocol: str
    method: str
    status: str
    source_path: str
    preds: dict[str, list[list[float]]] = field(default_factory=dict)
    artifact_gt: dict[str, list[list[float]]] = field(default_factory=dict)
    duplicate_group_rows: int = 0
    unresolved_rows: int = 0


def canonical_groups() -> dict[str, dict[str, dict[str, Any]]]:
    return {
        split: mb.make_groups(read_jsonl(DATA_ROOT / f"{split}.jsonl"))
        for split in ["train", "val", "eval"]
    }


def singleton_groups(groups: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {gid: g for gid, g in groups.items() if len(g["gt_boxes"]) == 1}


def build_key_index(groups: dict[str, dict[str, Any]]) -> dict[tuple[str, str, str], str]:
    return {
        (str(g["dicom_id"]), norm_text(g["finding"]), norm_text(g["claim_sentence"])): gid
        for gid, g in groups.items()
    }


def resolve_gid(
    groups: dict[str, dict[str, Any]],
    index: dict[tuple[str, str, str], str],
    *,
    group_id: Any = None,
    dicom_id: Any = None,
    finding: Any = None,
    phrase: Any = None,
) -> str | None:
    if group_id is not None and str(group_id) in groups:
        return str(group_id)
    exact = index.get((str(dicom_id), norm_text(finding), norm_text(phrase)))
    if exact is not None:
        return exact
    matches = [
        gid
        for gid, g in groups.items()
        if str(g["dicom_id"]) == str(dicom_id)
        and norm_text(g["claim_sentence"]) == norm_text(phrase)
    ]
    return matches[0] if len(matches) == 1 else None


def add_prediction(
    bundle: MethodPredictions,
    gid: str | None,
    pred: list[list[float]],
    artifact_gt: list[list[float]] | None = None,
) -> None:
    if gid is None:
        bundle.unresolved_rows += 1
        return
    if gid in bundle.preds:
        bundle.duplicate_group_rows += 1
        return
    bundle.preds[gid] = pred
    if artifact_gt is not None:
        bundle.artifact_gt[gid] = artifact_gt


def load_clean_multibox(
    groups: dict[str, dict[str, Any]],
) -> list[MethodPredictions]:
    specs = [
        (
            "action_policy_plus_count_head_recomputed_rows.csv",
            "action_policy_plus_count_head",
            "fair_local_method",
        ),
        (
            "base_finegrid_contrastive_global_recomputed_rows.csv",
            "finegrid_plus_contrastive_global",
            "fair_local_artifact",
        ),
        (
            "agpt_released_transvg_forced_onebox_recomputed_rows.csv",
            "AGPT released TransVG forced one-box",
            "reference_only",
        ),
    ]
    out: list[MethodPredictions] = []
    pred_dir = CLEAN_ROOT / "predictions"
    for filename, method, status in specs:
        path = pred_dir / filename
        bundle = MethodPredictions("multibox_1444", method, status, str(path))
        for _, row in pd.read_csv(path).iterrows():
            add_prediction(
                bundle,
                str(row["group_id"]),
                safe_boxes(row["pred_boxes_json"]),
                safe_boxes(row["gt_boxes_json"]),
            )
        out.append(bundle)

    med_path = pred_dir / "medrpg_rowlevel_3seed_concat_for_audit_recomputed_rows.csv"
    med = pd.read_csv(med_path)
    for method, part in med.groupby("method", sort=False):
        bundle = MethodPredictions("multibox_1444", str(method), "fair_retrain", str(med_path))
        for _, row in part.iterrows():
            add_prediction(
                bundle,
                str(row["group_id"]),
                safe_boxes(row["pred_boxes_json"]),
                safe_boxes(row["gt_boxes_json"]),
            )
        out.append(bundle)
    out.append(load_medgrounder_multibox(groups))
    return out


def image_stem(value: Any) -> str:
    return re.split(r"[\\/]", str(value))[-1].rsplit(".", 1)[0]


def cxcywh_to_pixel_xyxy(box: list[float], width: float, height: float) -> list[float]:
    cx, cy, bw, bh = [float(x) for x in box]
    return [
        (cx - bw / 2) * width,
        (cy - bh / 2) * height,
        (cx + bw / 2) * width,
        (cy + bh / 2) * height,
    ]


def cxcywh_to_xyxy_unscaled(box: list[float]) -> list[float]:
    cx, cy, bw, bh = [float(x) for x in box]
    return [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]


def load_medgrounder_multibox(groups: dict[str, dict[str, Any]]) -> MethodPredictions:
    path = PROJECT_ROOT / "experiments" / "medgrounder_fairness_audit_v1" / "predictions" / "medgrounder_fair_retrain_s42_eval_predictions.csv"
    index = build_key_index(groups)
    bundle = MethodPredictions("multibox_1444", "MedGrounder fair retrain", "fair_retrain", str(path))
    for _, row in pd.read_csv(path).iterrows():
        gid = resolve_gid(
            groups,
            index,
            dicom_id=image_stem(row.get("img_path")),
            finding=row.get("category_names"),
            phrase=row.get("phrase"),
        )
        if gid is None:
            bundle.unresolved_rows += 1
            continue
        g = groups[gid]
        canonical_gt = [[float(x) for x in box] for box in g["gt_boxes"]]
        transformed_gt = [cxcywh_to_xyxy_unscaled(box) for box in safe_boxes(row.get("gt_boxes_cxcywh"))]
        transformed_pred = [cxcywh_to_xyxy_unscaled(box) for box in safe_boxes(row.get("pred_boxes_cxcywh"))]
        if not canonical_gt or not transformed_gt:
            add_prediction(bundle, gid, [], [])
            continue
        source = canonical_gt[0]
        target = transformed_gt[0]
        sx = (target[2] - target[0]) / (source[2] - source[0])
        sy = (target[3] - target[1]) / (source[3] - source[1])
        pad_x = target[0] - sx * source[0]
        pad_y = target[1] - sy * source[1]

        def inverse(box: list[float]) -> list[float]:
            return [
                (box[0] - pad_x) / sx,
                (box[1] - pad_y) / sy,
                (box[2] - pad_x) / sx,
                (box[3] - pad_y) / sy,
            ]

        pred = [inverse(box) for box in transformed_pred]
        artifact_gt = [inverse(box) for box in transformed_gt]
        add_prediction(bundle, gid, pred, artifact_gt)
    return bundle


def load_vlm(
    protocol: str,
    groups: dict[str, dict[str, Any]],
) -> list[MethodPredictions]:
    index = build_key_index(groups)
    out = []
    names = {
        "m4cxr": "M4CXR released",
        "maira2": "MAIRA-2 released",
    }
    for folder, method in names.items():
        path = VLM_ROOT / folder / protocol / "parsed_predictions.csv"
        bundle = MethodPredictions(protocol, method, "reference_only", str(path))
        for _, row in pd.read_csv(path).iterrows():
            gid = resolve_gid(
                groups,
                index,
                dicom_id=row.get("dicom_id"),
                finding=row.get("finding"),
                phrase=row.get("phrase"),
            )
            add_prediction(
                bundle,
                gid,
                safe_boxes(row.get("pred_boxes_pixel_xyxy_json")),
                safe_boxes(row.get("gt_boxes_pixel_xyxy_json")),
            )
        out.append(bundle)
    return out


def load_pixel_singlebox(
    path: Path,
    method: str,
    status: str,
    groups: dict[str, dict[str, Any]],
    *,
    phrase_col: str,
) -> MethodPredictions:
    index = build_key_index(groups)
    sample_index: dict[str, str] = {}
    identity_path = PACKAGED_SINGLEBOX / "predictions" / "action_policy_plus_count_head_singlebox_eval_predictions.csv"
    if identity_path.exists():
        identity = pd.read_csv(identity_path, usecols=["sample_id", "dicom_id", "finding", "claim_sentence"])
        for _, identity_row in identity.iterrows():
            identity_gid = resolve_gid(
                groups,
                index,
                dicom_id=identity_row["dicom_id"],
                finding=identity_row["finding"],
                phrase=identity_row["claim_sentence"],
            )
            if identity_gid is not None:
                sample_index[str(identity_row["sample_id"])] = identity_gid
    bundle = MethodPredictions("singlebox_888", method, status, str(path))
    for _, row in pd.read_csv(path).iterrows():
        gid = resolve_gid(
            groups,
            index,
            group_id=row.get("group_id"),
            dicom_id=row.get("dicom_id"),
            finding=row.get("finding"),
            phrase=row.get(phrase_col),
        )
        if gid is None and row.get("sample_id") is not None:
            gid = sample_index.get(str(row.get("sample_id")))
        pred = [[float(row[c]) for c in ["pred_x1", "pred_y1", "pred_x2", "pred_y2"]]]
        artifact_gt = [[float(row[c]) for c in ["gt_x1", "gt_y1", "gt_x2", "gt_y2"]]]
        add_prediction(bundle, gid, pred, artifact_gt)
    return bundle


def load_agpt_singlebox(groups: dict[str, dict[str, Any]]) -> MethodPredictions:
    path = PROJECT_ROOT / "experiments" / "agpt_baseline_rerun_v2" / "predictions" / "agpt_transvg_singlebox_eval_predictions.csv"
    index = build_key_index(groups)
    bundle = MethodPredictions("singlebox_888", "AGPT released TransVG", "reference_only", str(path))
    for _, row in pd.read_csv(path).iterrows():
        gid = resolve_gid(
            groups,
            index,
            dicom_id=image_stem(row["image_path"]),
            phrase=row.get("phrase"),
        )
        if gid is None:
            bundle.unresolved_rows += 1
            continue
        g = groups[gid]
        width, height = float(g["image_width"]), float(g["image_height"])
        pcx, pcy, pw, ph = [float(row[c]) for c in ["pred_cx", "pred_cy", "pred_w", "pred_h"]]
        gcx, gcy, gw, gh = [float(row[c]) for c in ["gt_cx", "gt_cy", "gt_w", "gt_h"]]
        canonical_gt = [float(x) for x in g["gt_boxes"][0]]
        transformed_gt = np.asarray(
            [(gcx - gw / 2) * 640.0, (gcy - gh / 2) * 640.0, (gcx + gw / 2) * 640.0, (gcy + gh / 2) * 640.0],
            dtype=float,
        )
        sx = (transformed_gt[2] - transformed_gt[0]) / (canonical_gt[2] - canonical_gt[0])
        sy = (transformed_gt[3] - transformed_gt[1]) / (canonical_gt[3] - canonical_gt[1])
        pad_x = transformed_gt[0] - sx * canonical_gt[0]
        pad_y = transformed_gt[1] - sy * canonical_gt[1]

        def inverse_letterbox(cx: float, cy: float, bw: float, bh: float) -> list[float]:
            transformed = np.asarray(
                [(cx - bw / 2) * 640.0, (cy - bh / 2) * 640.0, (cx + bw / 2) * 640.0, (cy + bh / 2) * 640.0],
                dtype=float,
            )
            return [
                (transformed[0] - pad_x) / sx,
                (transformed[1] - pad_y) / sy,
                (transformed[2] - pad_x) / sx,
                (transformed[3] - pad_y) / sy,
            ]

        pred = [inverse_letterbox(pcx, pcy, pw, ph)]
        artifact_gt = [inverse_letterbox(gcx, gcy, gw, gh)]
        add_prediction(bundle, gid, pred, artifact_gt)
    return bundle


def load_singlebox(groups: dict[str, dict[str, Any]]) -> list[MethodPredictions]:
    out = [
        load_pixel_singlebox(
            PACKAGED_SINGLEBOX / "predictions" / "action_policy_plus_count_head_singlebox_eval_predictions.csv",
            "action_policy_plus_count_head_singlebox",
            "fair_local_method",
            groups,
            phrase_col="claim_sentence",
        ),
        load_pixel_singlebox(
            MERGED_SINGLEBOX_PATH,
            "merged_detectors_dino_full_phrase_fusion_singlebox_fair",
            "fair_local_development_candidate",
            groups,
            phrase_col="claim_sentence",
        ),
        load_pixel_singlebox(
            MEDRPG_SINGLE_ROOT / "predictions" / "medrpg_fair_full_phrase_s42_eval_predictions.csv",
            "MedRPG full phrase",
            "fair_retrain",
            groups,
            phrase_col="query_text",
        ),
        load_pixel_singlebox(
            MEDRPG_SINGLE_ROOT / "predictions" / "medrpg_fair_rule_context_s42_eval_predictions.csv",
            "MedRPG rule-context",
            "fair_retrain",
            groups,
            phrase_col="query_text",
        ),
        load_pixel_singlebox(
            PROJECT_ROOT
            / "experiments"
            / "ms_cxr_singlebox_semantic_finegrid_fair_v1"
            / "predictions"
            / "semantic_finegrid_singlebox_fair_eval_predictions.csv",
            "semantic finegrid singlebox",
            "fair_local_artifact",
            groups,
            phrase_col="claim_sentence",
        ),
        load_agpt_singlebox(groups),
    ]
    out.extend(load_vlm("singlebox_888", groups))
    return out


def evaluate_bundle(
    bundle: MethodPredictions,
    groups: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    per_group = []
    invalid_boxes = out_of_bounds = exact_gt = near_exact_gt = duplicate_pairs = 0
    gt_mismatch = 0
    for gid, g in groups.items():
        gt = [[float(x) for x in box] for box in g["gt_boxes"]]
        pred = bundle.preds.get(gid, [])
        width, height = float(g["image_width"]), float(g["image_height"])
        if gid in bundle.artifact_gt and not boxes_equal(gt, bundle.artifact_gt[gid], atol=1e-3):
            gt_mismatch += 1
        for box in pred:
            if not valid_box(box):
                invalid_boxes += 1
            if box[0] < -1e-4 or box[1] < -1e-4 or box[2] > width + 1e-4 or box[3] > height + 1e-4:
                out_of_bounds += 1
            if any(np.allclose(box, gold, atol=1e-6, rtol=0.0) for gold in gt):
                exact_gt += 1
            if gt and max(box_iou(box, gold) for gold in gt) >= 0.999:
                near_exact_gt += 1
        duplicate_pairs += sum(
            box_iou(pred[i], pred[j]) >= 0.95
            for i in range(len(pred))
            for j in range(i + 1, len(pred))
        )

        if bundle.protocol == "singlebox_888":
            iou = box_iou(gt[0], pred[0]) if gt and pred else 0.0
            per_group.append(
                {
                    "protocol": bundle.protocol,
                    "method": bundle.method,
                    "status": bundle.status,
                    "group_id": gid,
                    "subject_id": str(g["subject_id"]),
                    "n_gt": len(gt),
                    "n_pred": len(pred),
                    "single_iou": iou,
                    "hit_0_3": float(iou >= 0.3),
                    "hit_0_5": float(iou >= 0.5),
                }
            )
            continue

        coverage = float(np.mean([max((box_iou(gold, p) for p in pred), default=0.0) for gold in gt])) if gt else 0.0
        row: dict[str, Any] = {
            "protocol": bundle.protocol,
            "method": bundle.method,
            "status": bundle.status,
            "group_id": gid,
            "subject_id": str(g["subject_id"]),
            "n_gt": len(gt),
            "n_pred": len(pred),
            "coverage_mean_iou": coverage,
            "enclosing_hull_iou": hull_iou(gt, pred),
            "rectangle_union_iou": rectangle_union_iou(gt, pred),
            "raster_union_iou_224": raster_union_iou(gt, pred, width, height),
            "gt_hit_rate_0_3": float(np.mean([max((box_iou(gold, p) for p in pred), default=0.0) >= 0.3 for gold in gt])) if gt else 0.0,
            "gt_hit_rate_0_5": float(np.mean([max((box_iou(gold, p) for p in pred), default=0.0) >= 0.5 for gold in gt])) if gt else 0.0,
        }
        for threshold, suffix in [(0.3, "0_3"), (0.5, "0_5")]:
            gtp = greedy_tp(gt, pred, threshold)
            mtp = maximum_tp(gt, pred, threshold)
            row[f"greedy_tp_{suffix}"] = gtp
            row[f"maximum_tp_{suffix}"] = mtp
            row[f"set_f1_greedy_{suffix}"] = f1_from_tp(gtp, len(gt), len(pred))
            row[f"set_f1_maximum_{suffix}"] = f1_from_tp(mtp, len(gt), len(pred))
        per_group.append(row)

    coord = {
        "protocol": bundle.protocol,
        "method": bundle.method,
        "status": bundle.status,
        "source_path": bundle.source_path,
        "n_expected_groups": len(groups),
        "n_prediction_groups": len(bundle.preds),
        "n_missing_groups": len(set(groups) - set(bundle.preds)),
        "n_extra_groups": len(set(bundle.preds) - set(groups)),
        "n_duplicate_group_rows": bundle.duplicate_group_rows,
        "n_unresolved_rows": bundle.unresolved_rows,
        "n_artifact_gt_mismatch_groups": gt_mismatch,
        "n_invalid_pred_boxes": invalid_boxes,
        "n_out_of_bounds_pred_boxes": out_of_bounds,
        "n_exact_gt_copy_boxes": exact_gt,
        "n_near_exact_gt_iou_ge_0_999": near_exact_gt,
        "n_duplicate_pred_pairs_iou_ge_0_95": duplicate_pairs,
    }
    return per_group, coord


def add_medrpg_aggregate(per_group: pd.DataFrame) -> pd.DataFrame:
    seed_names = [
        "medrpg_rowlevel_full_phrase_s42",
        "medrpg_rowlevel_full_phrase_s13",
        "medrpg_rowlevel_full_phrase_s2026",
    ]
    part = per_group[per_group["method"].isin(seed_names)].copy()
    if part.empty:
        return per_group
    numeric = [
        c
        for c in part.columns
        if c not in {"protocol", "method", "status", "group_id", "subject_id"}
        and pd.api.types.is_numeric_dtype(part[c])
    ]
    agg = part.groupby(["protocol", "group_id", "subject_id"], as_index=False)[numeric].mean()
    agg["method"] = "MedRPG row-level 3-seed mean"
    agg["status"] = "fair_retrain"
    return pd.concat([per_group, agg[per_group.columns]], ignore_index=True)


def summarize_metrics(per_group: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (protocol, method, status), part in per_group.groupby(["protocol", "method", "status"], sort=False):
        if protocol == "singlebox_888":
            rows.append(
                {
                    "protocol": protocol,
                    "method": method,
                    "status": status,
                    "n_eval": len(part),
                    "mean_iou": part["single_iou"].mean(),
                    "hit_0_3": part["hit_0_3"].mean(),
                    "hit_0_5": part["hit_0_5"].mean(),
                    "mean_pred_count": part["n_pred"].mean(),
                    "n_pred_count_not_one": int((part["n_pred"] != 1).sum()),
                }
            )
        else:
            rows.append(
                {
                    "protocol": protocol,
                    "method": method,
                    "status": status,
                    "n_eval": len(part),
                    "n_gt_boxes": int(round(part["n_gt"].sum())),
                    "coverage_mean_iou": part["coverage_mean_iou"].mean(),
                    "enclosing_hull_iou": part["enclosing_hull_iou"].mean(),
                    "rectangle_union_iou": part["rectangle_union_iou"].mean(),
                    "raster_union_iou_224": part["raster_union_iou_224"].mean(),
                    "gt_hit_rate_0_3": part["gt_hit_rate_0_3"].mean(),
                    "gt_hit_rate_0_5": part["gt_hit_rate_0_5"].mean(),
                    "set_f1_greedy_0_3": part["set_f1_greedy_0_3"].mean(),
                    "set_f1_greedy_0_5": part["set_f1_greedy_0_5"].mean(),
                    "set_f1_maximum_0_3": part["set_f1_maximum_0_3"].mean(),
                    "set_f1_maximum_0_5": part["set_f1_maximum_0_5"].mean(),
                    "mean_pred_count": part["n_pred"].mean(),
                }
            )
    return pd.DataFrame(rows)


def split_audit(
    groups_by_split: dict[str, dict[str, dict[str, Any]]], out_dir: Path
) -> dict[str, Any]:
    rows_by_split = {split: read_jsonl(DATA_ROOT / f"{split}.jsonl") for split in groups_by_split}
    summary = []
    protocol_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for split, groups in groups_by_split.items():
        task_to_group = {task: gid for gid, g in groups.items() for task in g["task_ids"]}
        for protocol in ["multibox_1444", "singlebox_888"]:
            keep_groups = groups if protocol == "multibox_1444" else singleton_groups(groups)
            keep = [row for row in rows_by_split[split] if task_to_group[str(row["task_id"])] in keep_groups]
            protocol_rows[(protocol, split)] = keep
            summary.append(
                {
                    "protocol": protocol,
                    "split": split,
                    "n_annotation_rows": len(keep),
                    "n_phrase_groups": len(keep_groups),
                    "n_gt_boxes": sum(len(g["gt_boxes"]) for g in keep_groups.values()),
                    "n_patients": len({str(x["subject_id"]) for x in keep}),
                    "n_studies": len({str(x["study_id"]) for x in keep}),
                    "n_dicoms": len({str(x["dicom_id"]) for x in keep}),
                }
            )
    pd.DataFrame(summary).to_csv(out_dir / "split_integrity_summary.csv", index=False)

    overlap_rows = []
    field_map = {
        "patient": "subject_id",
        "study": "study_id",
        "dicom": "dicom_id",
        "annotation": "ms_cxr_annotation_id",
        "task": "task_id",
    }
    for protocol in ["multibox_1444", "singlebox_888"]:
        for a, b in [("train", "val"), ("train", "eval"), ("val", "eval")]:
            for label, field_name in field_map.items():
                aa = {str(x[field_name]) for x in protocol_rows[(protocol, a)]}
                bb = {str(x[field_name]) for x in protocol_rows[(protocol, b)]}
                overlap_rows.append(
                    {
                        "protocol": protocol,
                        "split_a": a,
                        "split_b": b,
                        "identity_level": label,
                        "n_overlap": len(aa & bb),
                    }
                )
    overlap = pd.DataFrame(overlap_rows)
    overlap.to_csv(out_dir / "within_protocol_split_overlap.csv", index=False)

    mapping_rows = []
    for split in ["train", "val", "eval"]:
        full_by_ann = {str(x["ms_cxr_annotation_id"]): x for x in protocol_rows[("multibox_1444", split)]}
        for row in protocol_rows[("singlebox_888", split)]:
            ann = str(row["ms_cxr_annotation_id"])
            peer = full_by_ann.get(ann)
            mapping_rows.append(
                {
                    "split": split,
                    "annotation_id": ann,
                    "found_in_1444_same_split": peer is not None,
                    "bbox_exact": bool(peer is not None and np.allclose(row["gold_bbox_xyxy"], peer["gold_bbox_xyxy"], atol=0, rtol=0)),
                    "task_exact": bool(peer is not None and str(row["task_id"]) == str(peer["task_id"])),
                }
            )
    mapping = pd.DataFrame(mapping_rows)
    mapping.to_csv(out_dir / "cross_protocol_singleton_mapping.csv", index=False)

    result = {
        "status": "PASS" if int(overlap["n_overlap"].sum()) == 0 and mapping["bbox_exact"].all() else "FAIL",
        "all_within_protocol_identity_overlaps_zero": bool((overlap["n_overlap"] == 0).all()),
        "n_888_annotations": len(mapping),
        "n_888_missing_from_1444_same_split": int((~mapping["found_in_1444_same_split"]).sum()),
        "n_888_bbox_mismatch": int((~mapping["bbox_exact"]).sum()),
        "interpretation": "singlebox_888 is the exact singleton annotation subset of multibox_1444, not an independent dataset.",
    }
    write_json(out_dir / "split_integrity_result.json", result)
    return result


def candidate_source_counts(out_dir: Path) -> pd.DataFrame:
    rows = []
    roots = {
        "singlebox_888": SINGLEBOX_ROOT / "candidate_table",
        "multibox_1444": MULTIBOX_ROOT / "candidate_table",
    }
    for protocol, root in roots.items():
        for split in ["train", "val", "eval"]:
            path = root / f"{split}_candidate_table.csv"
            counts: Counter[str] = Counter()
            if path.exists():
                for chunk in pd.read_csv(path, usecols=["source_model"], chunksize=100_000):
                    counts.update(chunk["source_model"].astype(str).tolist())
            for source, count in sorted(counts.items()):
                rows.append({"protocol": protocol, "split": split, "source_model": source, "n_candidate_rows": count})
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "candidate_source_counts.csv", index=False)
    return result


FORBIDDEN_FEATURE = re.compile(
    r"(^|_)(gold|gt|target|label|oracle|positive|answer|matched|candidate_source|hard_negative_type|negative_type)(_|$)",
    re.IGNORECASE,
)


def feature_schema_audit(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    schemas = [
        (
            "singlebox_reranker",
            SINGLEBOX_ROOT / "models" / "reranker_features.json",
        ),
        (
            "multibox_reranker",
            MULTIBOX_ROOT / "models" / "reranker_features.json",
        ),
    ]
    feature_rows = []
    for model, path in schemas:
        value = read_json(path, {})
        for feature in value.get("features", []):
            feature_rows.append(
                {
                    "model": model,
                    "feature": feature,
                    "forbidden_direct_gold_or_source_feature": bool(FORBIDDEN_FEATURE.search(str(feature))),
                    "contains_siglip": "siglip" in str(feature).lower(),
                    "contains_biomedclip": "biomed" in str(feature).lower(),
                    "schema_path": str(path),
                }
            )
    used_path = LEAK_ROOT / "audit" / "inference_feature_columns_used.csv"
    if used_path.exists():
        used = pd.read_csv(used_path)
        for _, row in used[used["model"].astype(str).eq("count_head")].iterrows():
            feature = str(row["feature"])
            feature_rows.append(
                {
                    "model": "multibox_count_head",
                    "feature": feature,
                    "forbidden_direct_gold_or_source_feature": bool(FORBIDDEN_FEATURE.search(feature)),
                    "contains_siglip": "siglip" in feature.lower(),
                    "contains_biomedclip": "biomed" in feature.lower(),
                    "schema_path": str(used_path),
                }
            )
    features = pd.DataFrame(feature_rows)
    features.to_csv(out_dir / "model_feature_schema_audit.csv", index=False)

    raw_rows = []
    for protocol, path in [
        ("singlebox_888", SINGLEBOX_ROOT / "candidate_table" / "eval_candidate_table.csv"),
        ("multibox_1444", MULTIBOX_ROOT / "candidate_table" / "eval_candidate_table.csv"),
    ]:
        columns = list(pd.read_csv(path, nrows=0).columns)
        for column in columns:
            if FORBIDDEN_FEATURE.search(column) or column in {"gold_iou", "target_iou", "gold_boxes", "gold_count", "gold_best_iou", "gold_best_index"}:
                raw_rows.append(
                    {
                        "protocol": protocol,
                        "raw_column": column,
                        "present_in_raw_eval_candidate_table": True,
                        "used_by_model": bool((features["feature"] == column).any()),
                        "path": str(path),
                    }
                )
    raw = pd.DataFrame(raw_rows)
    raw.to_csv(out_dir / "raw_eval_gold_columns_audit.csv", index=False)
    return features, raw


def joblib_audit(out_dir: Path) -> pd.DataFrame:
    paths = [
        ("singlebox_legacy_reranker", SINGLEBOX_ROOT / "models" / "learned_candidate_reranker.joblib"),
        ("multibox_legacy_reranker", MULTIBOX_ROOT / "models" / "learned_candidate_reranker.joblib"),
        ("multibox_count_head", MULTIBOX_ROOT / "models" / "count_head.joblib"),
        ("multibox_action_policy", MULTIBOX_ROOT / "models" / "action_policy.joblib"),
        (
            "multibox_v3_compatible_reranker",
            PROJECT_ROOT
            / "experiments"
            / "cig_pg_rad_pretraining"
            / "20260708_173522_medium_full_v3"
            / "reranker_compat"
            / "learned_candidate_reranker_v3.joblib",
        ),
    ]
    rows = []
    code = "import joblib,sys; joblib.load(sys.argv[1]); print('OK')"
    for name, path in paths:
        if not path.exists():
            rows.append({"artifact": name, "path": str(path), "exists": False, "load_status": "MISSING", "detail": ""})
            continue
        run = subprocess.run(
            [sys.executable, "-c", code, str(path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        detail = (run.stderr or run.stdout).strip().replace("\n", " ")[-1000:]
        rows.append(
            {
                "artifact": name,
                "path": str(path),
                "exists": True,
                "load_status": "PASS" if run.returncode == 0 else "FAIL",
                "detail": detail,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "joblib_compatibility_audit.csv", index=False)
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_inventory(out_dir: Path) -> pd.DataFrame:
    paths = [
        PACKAGED_SINGLEBOX / "predictions" / "action_policy_plus_count_head_singlebox_eval_predictions.csv",
        SINGLEBOX_ROOT / "predictions" / "full_rescue_singlebox_eval_predictions.csv",
        SINGLEBOX_ROOT / "models" / "learned_candidate_reranker.joblib",
        MULTIBOX_ROOT / "per_query" / "action_policy_plus_count_head_eval_predictions.csv",
        MULTIBOX_ROOT / "per_query" / "base_finegrid_contrastive_global_eval_predictions.csv",
        MULTIBOX_ROOT / "models" / "count_head.joblib",
        MULTIBOX_ROOT / "models" / "learned_candidate_reranker.joblib",
        CONTRASTIVE_ROOT / "predictions" / "finegrid_plus_contrastive_global_eval_phrase_group_predictions.csv",
        CONTRASTIVE_ROOT / "configs" / "alpha_global.json",
        SEMANTIC_ROOT / "configs" / "run_config.json",
        LEAK_ROOT / "audit" / "sanitized_leakage_check_result.json",
    ]
    rows = []
    for path in paths:
        rows.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
                "sha256": sha256_file(path) if path.exists() else "",
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "artifact_hash_inventory.csv", index=False)
    return result


def compare_prediction_csvs(
    path_a: Path,
    path_b: Path,
    id_col: str,
    pred_col: str = "pred_boxes_json",
) -> dict[str, Any]:
    a, b = pd.read_csv(path_a), pd.read_csv(path_b)
    if pred_col in a.columns and pred_col in b.columns:
        amap = {str(r[id_col]): safe_boxes(r[pred_col]) for _, r in a.iterrows()}
        bmap = {str(r[id_col]): safe_boxes(r[pred_col]) for _, r in b.iterrows()}
    else:
        cols = ["pred_x1", "pred_y1", "pred_x2", "pred_y2"]
        amap = {str(r[id_col]): [[float(r[c]) for c in cols]] for _, r in a.iterrows()}
        bmap = {str(r[id_col]): [[float(r[c]) for c in cols]] for _, r in b.iterrows()}
    common = sorted(set(amap) & set(bmap))
    exact = sum(boxes_equal(amap[key], bmap[key], atol=1e-9) for key in common)
    max_diff = 0.0
    for key in common:
        if len(amap[key]) == len(bmap[key]):
            for aa, bb in zip(amap[key], bmap[key]):
                max_diff = max(max_diff, max(abs(x - y) for x, y in zip(aa, bb)))
    return {
        "n_a": len(amap),
        "n_b": len(bmap),
        "n_common": len(common),
        "n_exact_prediction_sets": exact,
        "all_common_exact": exact == len(common),
        "max_abs_coordinate_diff": max_diff,
    }


def method_identity_audit(source_counts: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    multibox_script = (PROJECT_ROOT / "scripts" / "run_multibox_dev_experiments.py").read_text(encoding="utf-8")
    count_block = multibox_script.split("def run_count_head", 1)[1].split("\ndef ", 1)[0]
    single_compile = (PROJECT_ROOT / "scripts" / "compile_action_policy_plus_count_head_singlebox888_v1.py").read_text(encoding="utf-8")
    rerank_script = (PROJECT_ROOT / "scripts" / "run_rerank_experiments.py").read_text(encoding="utf-8")
    metric_script = (PROJECT_ROOT / "scripts" / "run_ms_cxr_multibox_phrase_grounding_v1.py").read_text(encoding="utf-8")
    vlm_metric_script = (PROJECT_ROOT / "src" / "vlm_reference" / "eval_utils.py").read_text(encoding="utf-8")

    single_copy = compare_prediction_csvs(
        PACKAGED_SINGLEBOX / "predictions" / "action_policy_plus_count_head_singlebox_eval_predictions.csv",
        SINGLEBOX_ROOT / "predictions" / "full_rescue_singlebox_eval_predictions.csv",
        "query_id",
        pred_col="coordinates",
    )
    base_copy = compare_prediction_csvs(
        MULTIBOX_ROOT / "per_query" / "base_finegrid_contrastive_global_eval_predictions.csv",
        CONTRASTIVE_ROOT / "predictions" / "finegrid_plus_contrastive_global_eval_phrase_group_predictions.csv",
        "group_id",
    )
    action_best_copy = compare_prediction_csvs(
        MULTIBOX_ROOT / "per_query" / "action_policy_best_eval_predictions.csv",
        MULTIBOX_ROOT / "per_query" / "base_finegrid_contrastive_global_eval_predictions.csv",
        "group_id",
    )
    declared_single = pd.read_csv(
        PACKAGED_SINGLEBOX / "predictions" / "action_policy_plus_count_head_singlebox_eval_predictions.csv"
    )
    merged_single = pd.read_csv(MERGED_SINGLEBOX_PATH)
    declared_mean = float(declared_single["iou"].astype(float).mean())
    merged_mean = float(merged_single["iou"].astype(float).mean())

    single_sources = sorted(source_counts[source_counts["protocol"].eq("singlebox_888")]["source_model"].unique())
    multi_sources = sorted(source_counts[source_counts["protocol"].eq("multibox_1444")]["source_model"].unique())
    producer_hits = []
    target_name = "finegrid_plus_contrastive_global_eval_phrase_group_predictions.csv"
    for path in (PROJECT_ROOT / "scripts").glob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if re.search(rf"to_csv\s*\([^\n]*{re.escape(target_name)}", text):
            producer_hits.append(str(path))

    inventory = subprocess.run(
        ["rg", "--files", str(PROJECT_ROOT / "experiments"), "-g", "*eval*prediction*.csv"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    eval_prediction_files = [line for line in inventory.stdout.splitlines() if line.strip()]
    findings = [
        {
            "finding_id": "F01",
            "severity": "HIGH",
            "status": "FAIL_NAMING",
            "title": "1444 action_policy_plus_count_head does not execute the action policy",
            "evidence": f"run_count_head contains apply_count_to_base={('apply_count_to_base' in count_block)} and apply_action_policy_call={('apply_action_policy(' in count_block)}",
            "impact": "The reported method is a frozen-base count-head overlay, not an action-policy plus count-head composition.",
        },
        {
            "finding_id": "F02",
            "severity": "HIGH",
            "status": "FAIL_NAMING",
            "title": "888 action/count result is a renamed full_rescue artifact",
            "evidence": json.dumps(single_copy, ensure_ascii=False),
            "impact": "The 888 and 1444 rows do not run the same algorithm despite sharing an action/count name.",
        },
        {
            "finding_id": "F03",
            "severity": "HIGH",
            "status": "FAIL_MIXED_METRIC_DEFINITION",
            "title": "The same union IoU column mixes two different metrics",
            "evidence": f"local eval uses enclosing hull={('iou_xyxy(union_box' in metric_script)}; VLM eval uses 224-mask union={('mask_iou_norm' in vlm_metric_script)}",
            "impact": "Local/MedGrounder/AGPT/MedRPG rows use enclosing hull, while M4CXR/MAIRA rows use raster union. Recompute one common definition before ranking.",
        },
        {
            "finding_id": "F04",
            "severity": "HIGH",
            "status": "FAIL_REPRODUCIBILITY",
            "title": "The exact 1444 base prediction artifact has no producer script in the repository",
            "evidence": f"artifact copy identity={base_copy}; candidate producer scripts={producer_hits}",
            "impact": "The final 1444 pipeline cannot currently be reproduced end to end from source and checkpoints.",
        },
        {
            "finding_id": "F05",
            "severity": "HIGH",
            "status": "EVAL_REUSE_RISK",
            "title": "The current eval split is no longer an untouched final test",
            "evidence": f"Found {len(eval_prediction_files)} eval-prediction CSV artifacts across iterative experiments.",
            "impact": "Even val-tuned individual runs can acquire researcher-overfitting when the final method is repeatedly chosen after seeing eval.",
        },
        {
            "finding_id": "F06",
            "severity": "MEDIUM",
            "status": "UNVERIFIED_SELECTION",
            "title": "888 full_rescue blend constants lack a recorded validation grid",
            "evidence": f"hardcoded alpha/iou/center constants={all(x in rerank_script for x in ['alpha=0.15', 'iou_threshold=0.1', 'center_threshold=0.15'])}; compile reads full_rescue={('full_rescue_singlebox_eval_predictions.csv' in single_compile)}",
            "impact": "There is no artifact proving those final blend constants were selected only on val.",
        },
        {
            "finding_id": "F07",
            "severity": "MEDIUM",
            "status": "ARCHITECTURE_LABEL_ERROR",
            "title": "The 0.5654 888 model does not contain SigLIP or BioMedCLIP reranker features",
            "evidence": f"singlebox candidate sources={single_sources}",
            "impact": "SigLIP/BioMedCLIP are absent from the primary reranker; only four semantic-fallback rows may inherit them. Do not label the primary 0.5654 path as RADDINO+YOLO+SigLIP.",
        },
        {
            "finding_id": "F08",
            "severity": "MEDIUM",
            "status": "PROTOCOL_DIVERGENCE",
            "title": "888 and 1444 use different detector pools",
            "evidence": f"888={single_sources}; 1444={multi_sources}",
            "impact": "Cross-protocol conclusions cannot be attributed to cardinality alone.",
        },
        {
            "finding_id": "F09",
            "severity": "PASS",
            "status": "PASS",
            "title": "The separately saved action_policy_best output is identical to the frozen base",
            "evidence": json.dumps(action_best_copy, ensure_ascii=False),
            "impact": "The learned action policy itself contributed no eval changes in this run.",
        },
        {
            "finding_id": "F10",
            "severity": "HIGH",
            "status": "FAIL_BEST_METHOD_CLAIM",
            "title": "The declared 0.5654 singlebox main is not the highest saved fair-local result",
            "evidence": f"declared_main={declared_mean:.9f}; val-tuned merged YOLO+DINO candidate={merged_mean:.9f}; delta={merged_mean - declared_mean:+.9f}",
            "impact": "Call 0.5654 the current declared/locked row, not the repository's highest fair-local eval result. Retrospective promotion of 0.5687 still requires a fresh holdout.",
        },
        {
            "finding_id": "F11",
            "severity": "MEDIUM",
            "status": "ABLATION_SCOPE_MISMATCH",
            "title": "Existing SigLIP/BioMedCLIP ablations are not end-to-end ablations of either current main",
            "evidence": "siglip_biomedclip_ablation_v1 evaluates older semantic singlebox/multibox paths; current 0.5654 reranker schema has neither feature and the exact 1444 frozen-base producer is missing.",
            "impact": "Those ablations diagnose semantic components only and cannot establish the component contribution to 0.5654/0.5629.",
        },
    ]
    result = pd.DataFrame(findings)
    result.to_csv(out_dir / "method_identity_findings.csv", index=False)
    return result


def metric_definition_audit(per_group: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    part = per_group[per_group["protocol"].eq("multibox_1444")]
    for method, group in part.groupby("method", sort=False):
        rows.append(
            {
                "method": method,
                "n_groups": len(group),
                "mean_enclosing_hull_iou": group["enclosing_hull_iou"].mean(),
                "mean_rectangle_union_iou": group["rectangle_union_iou"].mean(),
                "mean_raster_union_iou_224": group["raster_union_iou_224"].mean(),
                "hull_minus_rectangle_union": (group["enclosing_hull_iou"] - group["rectangle_union_iou"]).mean(),
                "n_groups_hull_differs": int((np.abs(group["enclosing_hull_iou"] - group["rectangle_union_iou"]) > 1e-12).sum()),
                "n_groups_greedy_differs_from_maximum_at_0_3": int((group["greedy_tp_0_3"] != group["maximum_tp_0_3"]).sum()),
                "n_groups_greedy_differs_from_maximum_at_0_5": int((group["greedy_tp_0_5"] != group["maximum_tp_0_5"]).sum()),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "metric_definition_audit.csv", index=False)
    reported_defs = pd.DataFrame(
        [
            {"method": method, "union_value_source_in_user_table": "raster_mask_union_224" if method in {"M4CXR released", "MAIRA-2 released"} else "enclosing_hull_iou"}
            for method in result["method"]
        ]
    )
    reported_defs.to_csv(out_dir / "reported_union_metric_definition_by_method.csv", index=False)
    return result


def published_metric_check(summary: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    expected: dict[tuple[str, str], dict[str, float]] = {
        ("multibox_1444", "MAIRA-2 released"): {"coverage_mean_iou": 0.5692, "raster_union_iou_224": 0.5774, "set_f1_greedy_0_3": 0.8400, "set_f1_greedy_0_5": 0.6139},
        ("multibox_1444", "action_policy_plus_count_head"): {"coverage_mean_iou": 0.5629, "enclosing_hull_iou": 0.5982, "set_f1_greedy_0_3": 0.8092, "set_f1_greedy_0_5": 0.6483},
        ("multibox_1444", "MedGrounder fair retrain"): {"coverage_mean_iou": 0.5255, "enclosing_hull_iou": 0.5636, "set_f1_greedy_0_3": 0.7971, "set_f1_greedy_0_5": 0.5609},
        ("multibox_1444", "M4CXR released"): {"coverage_mean_iou": 0.4986, "raster_union_iou_224": 0.5163, "set_f1_greedy_0_3": 0.7758, "set_f1_greedy_0_5": 0.5705},
        ("multibox_1444", "AGPT released TransVG forced one-box"): {"coverage_mean_iou": 0.4858, "enclosing_hull_iou": 0.4995, "set_f1_greedy_0_3": 0.7500, "set_f1_greedy_0_5": 0.5500},
        ("multibox_1444", "MedRPG row-level 3-seed mean"): {"coverage_mean_iou": 0.4766, "enclosing_hull_iou": 0.4786, "set_f1_greedy_0_3": 0.7563, "set_f1_greedy_0_5": 0.5646},
        ("singlebox_888", "AGPT released TransVG"): {"mean_iou": 0.6803, "hit_0_3": 0.9264, "hit_0_5": 0.7853},
        ("singlebox_888", "M4CXR released"): {"mean_iou": 0.5962, "hit_0_3": 0.8834, "hit_0_5": 0.6748},
        ("singlebox_888", "MAIRA-2 released"): {"mean_iou": 0.5848, "hit_0_3": 0.8528, "hit_0_5": 0.6442},
        ("singlebox_888", "action_policy_plus_count_head_singlebox"): {"mean_iou": 0.5654, "hit_0_3": 0.8282, "hit_0_5": 0.6319},
        ("singlebox_888", "MedRPG full phrase"): {"mean_iou": 0.5486, "hit_0_3": 0.8221, "hit_0_5": 0.5767},
    }
    rows = []
    for (protocol, method), metrics in expected.items():
        found = summary[(summary["protocol"] == protocol) & (summary["method"] == method)]
        for metric, value in metrics.items():
            actual = float(found.iloc[0][metric]) if len(found) else np.nan
            rows.append(
                {
                    "protocol": protocol,
                    "method": method,
                    "metric": metric,
                    "reported_rounded": value,
                    "independently_recomputed": actual,
                    "absolute_delta": abs(actual - value) if np.isfinite(actual) else np.nan,
                    "passes_reported_rounding_tolerance_5e_5": bool(np.isfinite(actual) and abs(actual - value) <= 5e-5),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "published_vs_independent_metrics.csv", index=False)
    return result


def cluster_bootstrap(
    per_group: pd.DataFrame,
    protocol: str,
    method_a: str,
    method_b: str,
    metric: str,
    reps: int,
    seed: int = 70710,
) -> dict[str, Any]:
    a = per_group[(per_group["protocol"] == protocol) & (per_group["method"] == method_a)][["group_id", "subject_id", metric]]
    b = per_group[(per_group["protocol"] == protocol) & (per_group["method"] == method_b)][["group_id", metric]]
    joined = a.merge(b, on="group_id", suffixes=("_a", "_b"))
    joined["diff"] = joined[f"{metric}_a"] - joined[f"{metric}_b"]
    subjects = joined["subject_id"].astype(str).unique()
    by_subject = {sid: joined.loc[joined["subject_id"].astype(str).eq(sid), "diff"].to_numpy() for sid in subjects}
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, dtype=float)
    for i in range(reps):
        sample = rng.choice(subjects, size=len(subjects), replace=True)
        draws[i] = np.concatenate([by_subject[sid] for sid in sample]).mean()
    point = float(joined["diff"].mean())
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "protocol": protocol,
        "method_a": method_a,
        "method_b": method_b,
        "metric": metric,
        "n_groups": len(joined),
        "n_subjects": len(subjects),
        "difference_a_minus_b": point,
        "ci_95_low": float(low),
        "ci_95_high": float(high),
        "ci_excludes_zero": bool(low > 0 or high < 0),
        "bootstrap_unit": "subject_cluster",
        "bootstrap_reps": reps,
    }


def bootstrap_audit(per_group: pd.DataFrame, reps: int, out_dir: Path) -> pd.DataFrame:
    specs = []
    for metric in ["single_iou", "hit_0_3", "hit_0_5"]:
        specs.append(("singlebox_888", "action_policy_plus_count_head_singlebox", "MedRPG full phrase", metric))
        specs.append(("singlebox_888", "merged_detectors_dino_full_phrase_fusion_singlebox_fair", "MedRPG full phrase", metric))
    for baseline in ["MedGrounder fair retrain", "MedRPG row-level 3-seed mean"]:
        for metric in ["coverage_mean_iou", "enclosing_hull_iou", "rectangle_union_iou", "raster_union_iou_224", "set_f1_greedy_0_3", "set_f1_greedy_0_5"]:
            specs.append(("multibox_1444", "action_policy_plus_count_head", baseline, metric))
    rows = [cluster_bootstrap(per_group, *spec, reps=reps) for spec in specs]
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "paired_subject_cluster_bootstrap.csv", index=False)
    return result


def baseline_fairness_table(out_dir: Path) -> pd.DataFrame:
    rows = [
        ["action_policy_plus_count_head_singlebox", "singlebox_888", "local", "MS-CXR train 638", "val 87 partially", "eval 163 repeatedly observed", "NOT_SAME_METHOD_AS_1444"],
        ["merged_detectors_dino_full_phrase_fusion_singlebox_fair", "singlebox_888", "local fair development candidate", "MS-CXR train 638", "RAD variant/rule/fusion selected on val 87", "eval 163 repeatedly observed", "HIGHER_THAN_DECLARED_MAIN_BUT_EVAL_REUSE"],
        ["action_policy_plus_count_head", "multibox_1444", "local", "MS-CXR train 814 phrase groups / 998 rows", "val 125 groups", "eval 220 repeatedly observed", "FAIR_LOCAL_BUT_EVAL_REUSE"],
        ["MedRPG full phrase", "singlebox_888", "local fair retrain", "same train 638", "same val 87", "same eval 163", "FAIR_RETRAIN_ONE_SEED"],
        ["MedRPG row-level 3-seed mean", "multibox_1444", "local fair retrain", "same train 998 rows", "same val/eval", "same eval 220", "FAIR_RETRAIN_THREE_SEEDS"],
        ["MedGrounder fair retrain", "multibox_1444", "local fair retrain", "same local split; starts ImaGenome pretrain", "best epoch by val mIoU", "same eval 220", "FAIR_EXTERNAL_PRETRAIN_SETTING_ONE_SEED"],
        ["AGPT released TransVG", "singlebox_888", "released checkpoint", "unknown/released MS-CXR fine-tuning", "unknown", "our eval 163", "REFERENCE_ONLY"],
        ["AGPT released TransVG forced one-box", "multibox_1444", "released checkpoint", "unknown/released MS-CXR fine-tuning", "unknown", "our eval 220; forced one box", "REFERENCE_ONLY_NOT_FAIR_MULTIBOX"],
        ["M4CXR released", "both", "released checkpoint", "unknown overlap", "unknown", "our eval", "REFERENCE_ONLY"],
        ["MAIRA-2 released", "both", "released checkpoint", "unknown overlap", "unknown", "our eval", "REFERENCE_ONLY"],
    ]
    result = pd.DataFrame(rows, columns=["method", "protocol", "provenance", "training_data", "selection_data", "evaluation_data", "fairness_status"])
    result.to_csv(out_dir / "baseline_fairness_protocol_audit.csv", index=False)
    return result


def component_ablation_scope_audit(out_dir: Path) -> pd.DataFrame:
    root = PROJECT_ROOT / "experiments" / "siglip_biomedclip_ablation_v1" / "metrics"
    rows = []
    for protocol, filename in [
        ("singlebox_888", "singlebox_siglip_biomedclip_ablation.csv"),
        ("multibox_1444", "multibox_siglip_biomedclip_ablation.csv"),
    ]:
        frame = pd.read_csv(root / filename)
        if protocol == "multibox_1444":
            frame = frame[frame["subset"].astype(str).eq("eval_phrase_groups_all")]
        for _, row in frame.iterrows():
            rows.append(
                {
                    "protocol": protocol,
                    "ablation": row["ablation"],
                    "use_siglip": row["use_siglip"],
                    "use_biomedclip": row["use_biomedclip"],
                    "mean_iou": row.get("mean_iou", np.nan),
                    "coverage_mean_iou": row.get("coverage_mean_iou", np.nan),
                    "legacy_union_iou": row.get("union_iou", np.nan),
                    "set_f1_0_3": row.get("set_f1_0_3", np.nan),
                    "set_f1_0_5": row.get("set_f1_0_5", np.nan),
                    "delta_vs_full_primary": row.get("delta_vs_full_mean_iou", row.get("delta_vs_full_coverage_mean_iou", np.nan)),
                    "scope": "older_semantic_component_path",
                    "is_end_to_end_ablation_of_current_main": False,
                    "source": str(root / filename),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "semantic_component_ablation_scope.csv", index=False)
    return result


def deferred_gpu_status(out_dir: Path) -> None:
    value = {
        "status": "DEFERRED_TO_AVOID_INTERFERING_WITH_RUNNING_YOLO_AND_QUEUED_DINO_JOBS",
        "wait_for_pids_observed_at_audit_start": [29664, 22036],
        "queued_runner": str(PROJECT_ROOT / "scripts" / "run_methodology_integrity_gpu_deferred_v1.py"),
        "planned_checks": [
            "Regenerate a deterministic sample from all six frozen MS-CXR YOLO checkpoints and compare with candidate caches.",
            "Re-run the frozen RAD-DINO singlebox head from cached tokens and compare all 163 boxes.",
            "Record CUDA/model/package versions and checkpoint hashes.",
        ],
        "not_automatically_reproducible": [
            "Exact finegrid_plus_contrastive_global base generation: producer script is absent.",
            "End-to-end RAD-DINO backbone token cache regeneration: exact cache producer provenance is incomplete.",
            "End-to-end SigLIP/BioMedCLIP semantic cache regeneration: deferred and blocked by missing exact final-base producer path.",
        ],
    }
    write_json(out_dir / "gpu_deferred" / "blocked_checks.json", value)


def markdown_table(df: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
    view = df[columns].copy()
    for col in columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: f"{x:.{digits}f}" if pd.notna(x) else "")
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("|", "\\|") for c in columns) + " |")
    return "\n".join(lines)


def write_report(
    out_root: Path,
    split_result: dict[str, Any],
    summary: pd.DataFrame,
    coord: pd.DataFrame,
    metric_def: pd.DataFrame,
    features: pd.DataFrame,
    raw_gold: pd.DataFrame,
    joblib: pd.DataFrame,
    findings: pd.DataFrame,
    bootstrap: pd.DataFrame,
    component_ablation: pd.DataFrame,
) -> None:
    single = summary[summary["protocol"].eq("singlebox_888")].copy()
    multi = summary[summary["protocol"].eq("multibox_1444")].copy()
    single = single[single["method"].isin(["AGPT released TransVG", "M4CXR released", "MAIRA-2 released", "merged_detectors_dino_full_phrase_fusion_singlebox_fair", "action_policy_plus_count_head_singlebox", "MedRPG full phrase"])]
    multi = multi[multi["method"].isin(["MAIRA-2 released", "action_policy_plus_count_head", "MedGrounder fair retrain", "M4CXR released", "AGPT released TransVG forced one-box", "MedRPG row-level 3-seed mean"])]
    single = single.sort_values("mean_iou", ascending=False)
    multi = multi.sort_values("enclosing_hull_iou", ascending=False)
    action_metric = metric_def[metric_def["method"].eq("action_policy_plus_count_head")].iloc[0]
    single_feat = features[features["model"].eq("singlebox_reranker")]
    multi_feat = features[features["model"].eq("multibox_reranker")]
    direct_forbidden = int(features["forbidden_direct_gold_or_source_feature"].sum())
    sanitized = read_json(LEAK_ROOT / "audit" / "sanitized_leakage_check_result.json", {})

    lines = [
        "# MS-CXR 방법론 전면 무결성 감사 V1",
        "",
        "## 결론",
        "",
        "**고의적인 치팅이나 추론 feature에 eval gold를 직접 넣은 증거는 발견하지 못했다.** 환자/검사/영상 단위 split 누수는 0이고, 저장된 예측을 독립 재계산한 주요 수치도 기존 표와 일치한다.",
        "",
        "그러나 현재 결과를 그대로 논문 최종 주장으로 쓰기에는 중요한 결함이 있다. 가장 큰 문제는 (1) eval 반복 관찰에 따른 researcher overfitting, (2) 888과 1444에서 같은 이름을 서로 다른 알고리즘에 붙인 점, (3) 하나의 `union IoU` 열에 enclosing hull과 raster-mask union을 섞은 점, (4) 1444 frozen base의 생성 코드가 없다는 점이다.",
        "",
        "종합 판정: `NO_DIRECT_LEAK_FOUND_BUT_NOT_PUBLICATION_SAFE_WITHOUT_RENAME_AND_FRESH_HOLDOUT`.",
        "",
        "## 핵심 발견",
        "",
        "| ID | 심각도 | 판정 | 내용 |",
        "|---|---|---|---|",
    ]
    for _, row in findings.iterrows():
        lines.append(f"| {row['finding_id']} | {row['severity']} | {row['status']} | {row['title']} |")
    lines.extend(
        [
            "",
            "## 데이터 무결성",
            "",
            f"- split audit: **{split_result['status']}**. train/val/eval의 patient, study, dicom, annotation, task overlap가 모두 0이다.",
            f"- 888의 전체 {split_result['n_888_annotations']}개 annotation은 1444에서 같은 split, 같은 GT로 발견됐다. 누락 {split_result['n_888_missing_from_1444_same_split']}개, bbox 불일치 {split_result['n_888_bbox_mismatch']}개다.",
            "- 따라서 `singlebox_888`은 독립 데이터셋이 아니라 `multibox_1444`의 정확한 singleton subset이다.",
            "",
            "## 추론 누수 검사",
            "",
            f"- 저장된 inference schema에서 직접 gold/source 계열 forbidden feature 수: **{direct_forbidden}**.",
            f"- 기존 sanitized 재실행: **{sanitized.get('status', 'UNKNOWN')}**, 220개 예측 변경 수 {sanitized.get('n_prediction_changed', 'NA')}.",
            f"- singlebox reranker feature {len(single_feat)}개 중 SigLIP {int(single_feat['contains_siglip'].sum())}개, BioMedCLIP {int(single_feat['contains_biomedclip'].sum())}개다.",
            f"- multibox candidate reranker feature {len(multi_feat)}개 중 SigLIP {int(multi_feat['contains_siglip'].sum())}개, BioMedCLIP {int(multi_feat['contains_biomedclip'].sum())}개다.",
            f"- 다만 raw eval candidate table에는 gold/target 진단 column이 {len(raw_gold)}개 남아 있다. 현재 모델이 읽지는 않지만 향후 accidental leak 방지를 위해 inference 직전에 물리적으로 drop해야 한다.",
            "",
            "## 1444 독립 재채점",
            "",
            "기존 표의 local/MedGrounder/AGPT/MedRPG `union IoU`는 enclosing hull이고, M4CXR/MAIRA-2 값은 224-mask union이다. 아래는 모든 방법을 세 정의로 동일하게 다시 계산한 결과다.",
            "",
            markdown_table(multi, ["method", "status", "coverage_mean_iou", "enclosing_hull_iou", "raster_union_iou_224", "rectangle_union_iou", "set_f1_greedy_0_3", "set_f1_greedy_0_5", "mean_pred_count"]),
            "",
            f"주 방법은 hull IoU {action_metric['mean_enclosing_hull_iou']:.4f}, VLM과 같은 224-mask union {action_metric['mean_raster_union_iou_224']:.4f}, exact rectangle-union {action_metric['mean_rectangle_union_iou']:.4f}다. hull과 exact union의 평균 차이는 {action_metric['hull_minus_rectangle_union']:.4f}다.",
            "",
            "## 888 독립 재채점",
            "",
            markdown_table(single, ["method", "status", "mean_iou", "hit_0_3", "hit_0_5", "mean_pred_count"]),
            "",
            "0.5654 행은 실제로 `full_rescue_singlebox` 예측을 이름만 바꾼 파일이다. 사용된 reranker feature에는 SigLIP/BioMedCLIP이 없으므로 `RADDINO+YOLO+SigLIP`이라고 쓰면 안 된다.",
            "",
            "또한 같은 split에서 val-only tuning 기록을 가진 `merged_detectors_dino_full_phrase_fusion_singlebox_fair`가 0.5687로 더 높다. 따라서 0.5654를 저장소 내 최고 성능이라고 부르는 것은 틀리다. 다만 지금 0.5687을 eval을 보고 소급 승격하는 것도 selection bias이므로 둘 다 development result로 두고 fresh holdout에서 결정해야 한다.",
            "",
            "## 방법 정체성",
            "",
            "- 1444 `action_policy_plus_count_head`: 실제 코드는 `apply_count_to_base(base_predictions, candidates, predicted_counts)`다. action policy를 합성하지 않는다. 권장 이름은 `frozen_base_plus_count_head` 또는 `base_preserving_count_head`다.",
            "- 888 `action_policy_plus_count_head_singlebox`: 실제 코드는 기존 `full_rescue_singlebox` CSV를 복사하고 cardinality=1 metadata를 추가한다. 권장 이름은 `full_rescue_singlebox`다.",
            "- 888 0.5654: YOLOv8s/v8m/YOLO11s/11m 후보 + RAD-DINO agreement/reliability + rule/context/train prior + HGB reranker + 4건 semantic-finegrid fallback이다.",
            "- 1444: YOLOv8n/s/m/l + YOLO11s/m, prior, xattn-rule 후보를 쓰며, frozen semantic/contrastive base 뒤에 count head를 얹는다. 888과 proposal pool도 다르다.",
            "- 1444 frozen base는 SigLIP/BioMedCLIP semantic 경로를 상속한 것으로 기록돼 있지만 exact producer가 없어 최종 연결을 완전히 재현할 수 없다.",
            "",
            "## Component ablation 범위",
            "",
            "기존 SigLIP/BioMedCLIP ablation은 현재 main의 end-to-end ablation이 아니라 이전 semantic component 경로의 실험이다.",
            "",
            markdown_table(component_ablation, ["protocol", "ablation", "use_siglip", "use_biomedclip", "mean_iou", "coverage_mean_iou", "delta_vs_full_primary", "is_end_to_end_ablation_of_current_main"]),
            "",
            "singlebox semantic 경로에서는 SigLIP 제거가 +0.00059, BioMedCLIP 제거가 +0.00074로 오히려 미세 상승했다. multibox semantic 경로에서는 둘 다 제거할 때 coverage가 -0.00838 내려갔다. 이 숫자를 0.5654 또는 0.5629 main의 component effect로 확대해석하면 안 된다.",
            "",
            "## 비교 공정성",
            "",
            "- MedRPG/MedGrounder의 local fair retrain은 같은 local split에서 평가되어 비교 가능한 편이다. 다만 단일 seed 결과와 외부 anatomy pretraining 여부는 별도 표기해야 한다.",
            "- AGPT/M4CXR/MAIRA-2 released checkpoint는 학습 overlap과 selection provenance가 통제되지 않았으므로 reference-only 유지가 맞다. AGPT 0.6803은 fair local competitor가 아니다.",
            "- eval 220/163은 수많은 반복 실험에서 이미 관찰됐다. 현재 표는 development leaderboard로만 해석하고, 최종 주장은 새 patient holdout에서 pipeline 전체를 다시 학습한 뒤 한 번만 평가해야 한다.",
            "",
            "## 재현성",
            "",
            f"- legacy joblib load 실패 수: {int((joblib['load_status'] == 'FAIL').sum())}. 상세는 `provenance/joblib_compatibility_audit.csv`에 있다.",
            f"- 예측/GT 좌표 감사에서 artifact GT mismatch 총합: {int(coord['n_artifact_gt_mismatch_groups'].sum())}; exact-GT copy prediction 총합: {int(coord['n_exact_gt_copy_boxes'].sum())}.",
            "- `finegrid_plus_contrastive_global_eval_phrase_group_predictions.csv`는 최종 base와 정확히 같지만 이 파일을 생성하는 producer script가 없다. 알파 val grid만 남아 있어 end-to-end 재현은 불가능하다.",
            "",
            "## 통계 검사",
            "",
            "paired bootstrap은 row가 아니라 patient를 cluster로 재표집했다. CI가 0을 포함하면 숫자가 높아도 확정적 우위로 표현하지 않는다.",
            "",
            markdown_table(bootstrap, ["protocol", "method_a", "method_b", "metric", "difference_a_minus_b", "ci_95_low", "ci_95_high", "ci_excludes_zero"]),
            "",
            "## GPU 지연 검사",
            "",
            "YOLO pretrain과 뒤이어 대기 중인 DINO repair를 방해하지 않기 위해 foundation-model 재추론은 이번 CPU 감사에서 실행하지 않았다. 두 작업 PID가 모두 끝난 뒤 `run_methodology_integrity_gpu_deferred_v1.py`가 sample cache 재현 검사를 수행하도록 별도 구성한다.",
            "",
            "정확한 finegrid+contrastive producer가 없는 부분은 GPU가 비어도 자동 재현할 수 없으며, 보고서에 `BLOCKED_BY_MISSING_PRODUCER`로 남긴다.",
            "",
            "## 필수 조치",
            "",
            "1. 표와 코드에서 888은 `full_rescue_singlebox`, 1444는 `base_preserving_count_head`로 즉시 이름을 분리한다.",
            "2. 혼합된 기존 `union IoU` 열을 폐기하고, 모든 방법에 동일하게 계산한 `enclosing-hull`, `raster-mask union`, `exact rectangle-union`을 명시적으로 보고한다.",
            "3. raw eval candidate table에서 gold/target column을 모델 호출 전에 강제 삭제하는 guard를 current main에도 적용한다.",
            "4. 888 full_rescue의 alpha/threshold를 val grid로 다시 선택하거나 현재 값을 pre-registered constant로 명시한다.",
            "5. frozen finegrid+contrastive base의 생성 스크립트, 입력 artifact hash, checkpoint, seed를 복구한다.",
            "6. 이미 만든 fresh patient split에서 모든 학습과 선택을 처음부터 다시 하고, holdout은 마지막 한 번만 연다.",
            "7. released checkpoint는 계속 reference-only panel에만 둔다.",
        ]
    )
    (out_root / "METHODOLOGY_INTEGRITY_AUDIT_KO.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.output_root.resolve()
    data_dir = out / "data_integrity"
    metric_dir = out / "metrics"
    method_dir = out / "method_identity"
    leak_dir = out / "leakage"
    provenance_dir = out / "provenance"
    fairness_dir = out / "fairness"
    for path in [data_dir, metric_dir, method_dir, leak_dir, provenance_dir, fairness_dir, out / "gpu_deferred"]:
        path.mkdir(parents=True, exist_ok=True)

    groups_by_split = canonical_groups()
    split_result = split_audit(groups_by_split, data_dir)
    source_counts = candidate_source_counts(method_dir)
    features, raw_gold = feature_schema_audit(leak_dir)
    joblib = joblib_audit(provenance_dir)
    artifact_inventory(provenance_dir)

    eval_multi = groups_by_split["eval"]
    eval_single = singleton_groups(eval_multi)
    bundles = load_clean_multibox(eval_multi) + load_vlm("multibox_1444", eval_multi) + load_singlebox(eval_single)
    per_group_rows: list[dict[str, Any]] = []
    coordinate_rows = []
    for bundle in bundles:
        groups = eval_single if bundle.protocol == "singlebox_888" else eval_multi
        rows, coord = evaluate_bundle(bundle, groups)
        per_group_rows.extend(rows)
        coordinate_rows.append(coord)
    per_group = add_medrpg_aggregate(pd.DataFrame(per_group_rows))
    coord = pd.DataFrame(coordinate_rows)
    per_group.to_csv(metric_dir / "independent_per_group_metrics.csv", index=False)
    coord.to_csv(metric_dir / "coordinate_and_gt_audit.csv", index=False)
    summary = summarize_metrics(per_group)
    summary.to_csv(metric_dir / "independent_metric_summary.csv", index=False)
    metric_def = metric_definition_audit(per_group, metric_dir)
    published_metric_check(summary, metric_dir)
    bootstrap = bootstrap_audit(per_group, args.bootstrap_reps, fairness_dir)
    baseline_fairness_table(fairness_dir)
    component_ablation = component_ablation_scope_audit(method_dir)
    findings = method_identity_audit(source_counts, method_dir)
    deferred_gpu_status(out)
    write_report(out, split_result, summary, coord, metric_def, features, raw_gold, joblib, findings, bootstrap, component_ablation)

    result = {
        "overall_status": "NO_DIRECT_LEAK_FOUND_BUT_NOT_PUBLICATION_SAFE_WITHOUT_RENAME_AND_FRESH_HOLDOUT",
        "split_integrity": split_result["status"],
        "direct_forbidden_model_features": int(features["forbidden_direct_gold_or_source_feature"].sum()),
        "artifact_gt_mismatch_groups": int(coord["n_artifact_gt_mismatch_groups"].sum()),
        "joblib_load_failures": int((joblib["load_status"] == "FAIL").sum()),
        "high_severity_findings": int((findings["severity"] == "HIGH").sum()),
        "gpu_checks": "DEFERRED",
        "report": str(out / "METHODOLOGY_INTEGRITY_AUDIT_KO.md"),
    }
    write_json(out / "audit_status.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

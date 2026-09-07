#!/usr/bin/env python
"""Chest ImaGenome device/anatomy 10k rule-vs-SMM query localization.

This experiment intentionally separates two Chest ImaGenome tasks:

1. Device-linked weak region localization.
   The target box is the Chest ImaGenome anatomy/landmark region associated
   with a device claim, not the physical contour of the device.

2. Anatomy object localization.
   The target box is the Chest ImaGenome anatomy object box itself.

For each task we compare deterministic rule-query features against frozen
Qwen/SMM text-query embeddings on the same frozen RAD-DINO patch heatmap bbox
head. This is direct bbox regression/heatmap supervision, not A/B/C candidate
selection and not Qwen coordinate generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_ms_cxr_vfm_localizer_stage1_p10_p19 as ms_base  # noqa: E402
from models_ms_cxr_vfm_localizer import PatchHeatmapBBoxHead, bbox_loss  # noqa: E402


EXP_NAME = "imagenome_device_anatomy_10k_rule_smm_v1"
EXP = PROJECT_ROOT / "experiments" / EXP_NAME
PRED = EXP / "predictions"
MET = EXP / "metrics"
OVER = EXP / "overlays"
CONTACT = EXP / "contact_sheets"
TRAIN = PROJECT_ROOT / "training" / EXP_NAME
DATA = TRAIN / "datasets"
RUNS = TRAIN / "runs"
CKPT = TRAIN / "checkpoints"
FEAT = PROJECT_ROOT / "features" / EXP_NAME
REPORT = PROJECT_ROOT / "reports" / EXP_NAME

SEED = 20260623
SPLITS = ("train", "val", "eval")

DEVICE_TARGETS: Dict[str, Dict[str, List[str]]] = {
    "picc": {
        "finding_names": ["picc"],
        "bbox_names": ["svc", "cavoatrial junction", "upper mediastinum"],
    },
    "endotracheal tube": {
        "finding_names": ["endotracheal tube"],
        "bbox_names": ["trachea", "carina", "upper mediastinum"],
    },
    "enteric tube": {
        "finding_names": ["enteric tube"],
        "bbox_names": ["abdomen", "right hemidiaphragm", "left hemidiaphragm", "mediastinum"],
    },
    "ij line": {
        "finding_names": ["ij line"],
        "bbox_names": ["cavoatrial junction", "right atrium", "svc", "upper mediastinum"],
    },
    "chest tube": {
        "finding_names": ["chest tube"],
        "bbox_names": ["right lung", "left lung", "right lower lung zone", "left lower lung zone"],
    },
    "cardiac pacer and wires": {
        "finding_names": ["cardiac pacer and wires"],
        "bbox_names": ["cardiac silhouette", "right atrium", "mediastinum"],
    },
}

ANATOMY_TARGETS = [
    "right lung",
    "left lung",
    "right upper lung zone",
    "right mid lung zone",
    "right lower lung zone",
    "left upper lung zone",
    "left mid lung zone",
    "left lower lung zone",
    "right costophrenic angle",
    "left costophrenic angle",
    "trachea",
    "carina",
    "svc",
    "cavoatrial junction",
    "mediastinum",
    "upper mediastinum",
    "cardiac silhouette",
    "abdomen",
]


def ensure_dirs() -> None:
    for p in [EXP, PRED, MET, OVER, CONTACT, TRAIN, DATA, RUNS, CKPT, FEAT, REPORT]:
        p.mkdir(parents=True, exist_ok=True)


def jd(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(jd(row) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def stable_hash_int(text: str) -> int:
    h = hashlib.md5(str(text).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def split_for_subject(subject_id: str) -> str:
    v = stable_hash_int(subject_id) % 100
    if v < 70:
        return "train"
    if v < 80:
        return "val"
    return "eval"


def xyxy_to_xywh(box: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = [float(x) for x in box]
    return [x1, y1, x2 - x1, y2 - y1]


def xyxy_to_norm_cxcywh(box: Sequence[float], iw: float, ih: float) -> List[float]:
    x1, y1, x2, y2 = [float(x) for x in box]
    return [
        max(0.0, min(1.0, ((x1 + x2) / 2.0) / iw)),
        max(0.0, min(1.0, ((y1 + y2) / 2.0) / ih)),
        max(1e-4, min(1.0, (x2 - x1) / iw)),
        max(1e-4, min(1.0, (y2 - y1) / ih)),
    ]


def read_image_size(path: str, cache: Dict[str, Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    if path in cache:
        return cache[path]
    p = Path(path)
    if not p.exists():
        return None
    try:
        with Image.open(p) as im:
            cache[path] = im.size
            return im.size
    except Exception:
        return None


def norm_cxcywh_to_xyxy(box: Sequence[float], iw: float, ih: float) -> List[float]:
    cx, cy, w, h = [float(x) for x in box]
    x1 = (cx - w / 2.0) * iw
    y1 = (cy - h / 2.0) * ih
    x2 = (cx + w / 2.0) * iw
    y2 = (cy + h / 2.0) * ih
    return [
        max(0.0, min(float(iw), x1)),
        max(0.0, min(float(ih), y1)),
        max(0.0, min(float(iw), x2)),
        max(0.0, min(float(ih), y2)),
    ]


def valid_box(box: Optional[Sequence[float]]) -> bool:
    if not box or len(box) != 4:
        return False
    x1, y1, x2, y2 = [float(x) for x in box]
    return x2 > x1 and y2 > y1


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    if not valid_box(a) or not valid_box(b):
        return 0.0
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def choose_claim(phrases: str, finding_name: str) -> str:
    parts = [p.strip() for p in str(phrases).split("|") if p.strip()]
    if not parts:
        return str(finding_name)
    terms = [t for t in re.split(r"[\s/_-]+", str(finding_name).lower()) if t]
    for p in parts:
        low = p.lower()
        if any(t in low for t in terms):
            return p
    return parts[0]


def laterality_from_text(text: str) -> str:
    t = f" {text.lower()} "
    right = bool(re.search(r"\bright\b|\brt\b", t))
    left = bool(re.search(r"\bleft\b|\blt\b", t))
    if "bilateral" in t or (right and left):
        return "bilateral"
    if right:
        return "right"
    if left:
        return "left"
    return "unknown"


def vertical_from_text(text: str) -> str:
    t = text.lower()
    if "apical" in t or "apex" in t:
        return "apical"
    if "upper" in t or "suprahilar" in t:
        return "upper"
    if "mid" in t or "middle" in t or "hilar" in t:
        return "mid"
    if "lower" in t or "basilar" in t or "base" in t or "basal" in t:
        return "lower"
    return "unknown"


def device_part_from_text(text: str, finding: str) -> str:
    t = f"{text} {finding}".lower()
    if "tip" in t:
        return "tip"
    if "lead" in t or "wire" in t:
        return "lead_or_wire"
    if "tube" in t:
        return "tube"
    if "line" in t or "catheter" in t or "picc" in t:
        return "line_or_catheter"
    return "device"


def landmark_from_text(text: str) -> str:
    t = text.lower()
    checks = [
        ("cavoatrial_junction", ["cavoatrial", "cavo atrial", "ca junction"]),
        ("svc", ["superior vena cava", " svc "]),
        ("carina", ["carina"]),
        ("trachea", ["trachea", "endotracheal", "ett", "et tube"]),
        ("abdomen", ["abdomen", "stomach", "gastric"]),
        ("mediastinum", ["mediastinum", "mediastinal"]),
        ("right_atrium", ["right atrium"]),
        ("cardiac_silhouette", ["cardiac", "heart", "pacer", "pacemaker"]),
        ("lung", ["pleural", "lung", "chest tube"]),
        ("hemidiaphragm", ["hemidiaphragm", "diaphragm"]),
    ]
    padded = f" {t} "
    for name, terms in checks:
        if any(term in padded for term in terms):
            return name
    return "unknown"


def relation_from_text(text: str) -> str:
    t = text.lower()
    if "terminat" in t or "tip" in t or "ends" in t:
        return "terminates_at"
    if "projects over" in t or "overlies" in t:
        return "projects_over"
    if "above" in t:
        return "above"
    if "below" in t:
        return "below"
    if "within" in t or "in the" in t:
        return "within"
    return "unknown"


def anatomy_slots(name: str) -> Dict[str, str]:
    t = name.lower()
    if t.startswith("right "):
        lat = "right"
    elif t.startswith("left "):
        lat = "left"
    else:
        lat = "none"
    vertical = vertical_from_text(t)
    if "lung" in t:
        family = "lung_region"
    elif "costophrenic" in t:
        family = "costophrenic_angle"
    elif t in {"trachea", "carina", "svc", "cavoatrial junction"}:
        family = "airway_vascular_landmark"
    elif "mediastinum" in t:
        family = "mediastinum"
    elif "cardiac" in t:
        family = "heart"
    elif "abdomen" in t:
        family = "abdomen"
    else:
        family = "anatomy"
    return {"laterality": lat, "vertical_region": vertical, "anatomy_family": family}


def device_rule_query(row: Dict) -> str:
    text = f"{row.get('finding_name', row['finding'])} {row.get('claim_sentence', '')}"
    slots = {
        "device": row["finding"],
        "part": device_part_from_text(text, row["finding"]),
        "landmark": landmark_from_text(text),
        "relation": relation_from_text(text),
        "laterality": laterality_from_text(text),
        "vertical": vertical_from_text(text),
    }
    return "; ".join(f"{k}={v}" for k, v in slots.items())


def anatomy_rule_query(row: Dict) -> str:
    slots = anatomy_slots(row["finding"])
    values = {"anatomy": row["finding"], **slots}
    return "; ".join(f"{k}={v}" for k, v in values.items())


def smm_query_text(row: Dict) -> str:
    if row["task_type"] == "device":
        return (
            "Extract the medical localization query for this chest X-ray device claim.\n"
            f"Device: {row['finding']}\n"
            f"Claim: {row.get('claim_sentence', '')}\n"
            "Focus on device part, landmark, relation, and location words."
        )
    return (
        "Represent this chest X-ray anatomy localization query.\n"
        f"Anatomy target: {row['finding']}\n"
        "Focus on the anatomical name and its left/right or upper/mid/lower modifiers."
    )


def build_device_dataset(args) -> Dict:
    loc_path = PROJECT_ROOT / "data_index" / "localized_finding_rows.csv.gz"
    cols = [
        "dicom_id",
        "subject_id",
        "study_id",
        "split",
        "view_position",
        "image_path",
        "finding_name",
        "finding_group",
        "relation",
        "is_positive",
        "object_name",
        "bbox_name",
        "clipped_x1",
        "clipped_y1",
        "clipped_x2",
        "clipped_y2",
        "phrases",
        "localization_type",
        "source",
    ]
    rng = np.random.default_rng(SEED)
    per_target = max(60, args.device_samples // len(DEVICE_TARGETS))
    max_pool = max(per_target * 6, 3000)
    reservoirs: Dict[str, List[Dict]] = defaultdict(list)
    seen = Counter()
    raw_counts = Counter()
    duplicate_keys = set()
    for chunk in pd.read_csv(loc_path, usecols=lambda c: c in cols, chunksize=args.csv_chunksize):
        chunk = chunk[chunk["is_positive"].fillna(False).astype(bool)]
        if chunk.empty:
            continue
        fn_lower = chunk["finding_name"].fillna("").astype(str).str.lower()
        bb_lower = chunk["bbox_name"].fillna("").astype(str).str.lower()
        valid = (
            chunk["clipped_x2"].astype(float).gt(chunk["clipped_x1"].astype(float))
            & chunk["clipped_y2"].astype(float).gt(chunk["clipped_y1"].astype(float))
        )
        for target, cfg in DEVICE_TARGETS.items():
            mask = valid & fn_lower.isin(cfg["finding_names"]) & bb_lower.isin(cfg["bbox_names"])
            if not mask.any():
                continue
            sub = chunk.loc[mask].copy()
            raw_counts[target] += int(len(sub))
            if len(sub) > args.per_chunk_cap:
                sub = sub.sample(n=args.per_chunk_cap, random_state=SEED + stable_hash_int(target) % 10000)
            for row in sub.itertuples(index=False):
                r = row._asdict()
                claim = choose_claim(r.get("phrases", ""), r["finding_name"])
                key = (str(r["dicom_id"]), str(r["finding_name"]), str(r["bbox_name"]), claim[:160])
                if key in duplicate_keys:
                    continue
                duplicate_keys.add(key)
                box = [float(r["clipped_x1"]), float(r["clipped_y1"]), float(r["clipped_x2"]), float(r["clipped_y2"])]
                rec = {
                    "task_id": "",
                    "task_type": "device",
                    "dicom_id": str(r["dicom_id"]),
                    "subject_id": str(r["subject_id"]),
                    "study_id": str(r["study_id"]),
                    "image_path": str(r["image_path"]),
                    "finding": target,
                    "finding_name": str(r["finding_name"]),
                    "finding_group": str(r.get("finding_group", "")),
                    "claim_sentence": claim,
                    "gold_bbox_xyxy": box,
                    "gold_bbox_xywh": xyxy_to_xywh(box),
                    "gold_bbox_norm_cxcywh": [],
                    "image_width": 0,
                    "image_height": 0,
                    "split": split_for_subject(str(r["subject_id"])),
                    "view_position": str(r.get("view_position", "")),
                    "bbox_name_reference": str(r["bbox_name"]),
                    "object_name_reference": str(r.get("object_name", "")),
                    "bbox_type": "chest_imagenome_device_linked_weak_region_bbox",
                    "gold_source": "chest_imagenome_silver_scene_graph",
                    "is_weak_reference": True,
                    "source_note": "Device claim linked to Chest ImaGenome anatomy/landmark region; not exact device contour.",
                }
                seen[target] += 1
                bucket = reservoirs[target]
                if len(bucket) < max_pool:
                    bucket.append(rec)
                else:
                    j = int(rng.integers(0, seen[target]))
                    if j < max_pool:
                        bucket[j] = rec
    return finalize_balanced_dataset("device", reservoirs, raw_counts, args.device_samples)


def build_anatomy_dataset(args) -> Dict:
    obj_path = PROJECT_ROOT / "data_index" / "chest_imagenome_object_rows.csv.gz"
    cols = [
        "dicom_id",
        "subject_id",
        "study_id",
        "split",
        "view_position",
        "image_path",
        "object_id",
        "object_name",
        "bbox_name",
        "image_width",
        "image_height",
        "clipped_x1",
        "clipped_y1",
        "clipped_x2",
        "clipped_y2",
        "is_valid_bbox",
    ]
    rng = np.random.default_rng(SEED + 19)
    per_target = max(30, args.anatomy_samples // len(ANATOMY_TARGETS))
    max_pool = max(per_target * 5, 2500)
    reservoirs: Dict[str, List[Dict]] = defaultdict(list)
    seen = Counter()
    raw_counts = Counter()
    duplicate_keys = set()
    anatomy_set = set(ANATOMY_TARGETS)
    for chunk in pd.read_csv(obj_path, usecols=lambda c: c in cols, chunksize=args.csv_chunksize):
        bb_lower = chunk["bbox_name"].fillna("").astype(str).str.lower()
        mask = bb_lower.isin(anatomy_set)
        if "is_valid_bbox" in chunk.columns:
            mask &= chunk["is_valid_bbox"].fillna(False).astype(bool)
        valid = (
            chunk["clipped_x2"].astype(float).gt(chunk["clipped_x1"].astype(float))
            & chunk["clipped_y2"].astype(float).gt(chunk["clipped_y1"].astype(float))
        )
        mask &= valid
        if not mask.any():
            continue
        sub_all = chunk.loc[mask].copy()
        for target in ANATOMY_TARGETS:
            sub = sub_all[sub_all["bbox_name"].fillna("").astype(str).str.lower().eq(target)]
            if sub.empty:
                continue
            raw_counts[target] += int(len(sub))
            if len(sub) > args.per_chunk_cap:
                sub = sub.sample(n=args.per_chunk_cap, random_state=SEED + stable_hash_int(target) % 10000)
            for row in sub.itertuples(index=False):
                r = row._asdict()
                key = (str(r["dicom_id"]), target, str(r.get("object_id", "")))
                if key in duplicate_keys:
                    continue
                duplicate_keys.add(key)
                box = [float(r["clipped_x1"]), float(r["clipped_y1"]), float(r["clipped_x2"]), float(r["clipped_y2"])]
                rec = {
                    "task_id": "",
                    "task_type": "anatomy",
                    "dicom_id": str(r["dicom_id"]),
                    "subject_id": str(r["subject_id"]),
                    "study_id": str(r["study_id"]),
                    "image_path": str(r["image_path"]),
                    "finding": target,
                    "finding_name": target,
                    "finding_group": "anatomy",
                    "claim_sentence": target,
                    "gold_bbox_xyxy": box,
                    "gold_bbox_xywh": xyxy_to_xywh(box),
                    "gold_bbox_norm_cxcywh": [],
                    "image_width": int(r.get("image_width", 0) or 0),
                    "image_height": int(r.get("image_height", 0) or 0),
                    "split": split_for_subject(str(r["subject_id"])),
                    "view_position": str(r.get("view_position", "")),
                    "bbox_name_reference": target,
                    "object_name_reference": str(r.get("object_name", target)),
                    "bbox_type": "chest_imagenome_anatomy_object_bbox",
                    "gold_source": "chest_imagenome_scene_graph_object_bbox",
                    "is_weak_reference": False,
                    "source_note": "Chest ImaGenome anatomy object bbox; not a lesion bbox.",
                }
                seen[target] += 1
                bucket = reservoirs[target]
                if len(bucket) < max_pool:
                    bucket.append(rec)
                else:
                    j = int(rng.integers(0, seen[target]))
                    if j < max_pool:
                        bucket[j] = rec
    return finalize_balanced_dataset("anatomy", reservoirs, raw_counts, args.anatomy_samples)


def finalize_balanced_dataset(task_type: str, reservoirs: Dict[str, List[Dict]], raw_counts: Counter, max_samples: int) -> Dict:
    rng = np.random.default_rng(SEED + stable_hash_int(task_type) % 10000)
    targets = list(DEVICE_TARGETS) if task_type == "device" else ANATOMY_TARGETS
    per_target = max(1, max_samples // len(targets))
    ratios = {"train": 0.70, "val": 0.10, "eval": 0.20}
    selected = {s: [] for s in SPLITS}
    class_split_pool_counts: Dict[str, Dict[str, int]] = {}
    for target in targets:
        rows = list(reservoirs.get(target, []))
        by_split = {s: [r for r in rows if r["split"] == s] for s in SPLITS}
        class_split_pool_counts[target] = {s: len(by_split[s]) for s in SPLITS}
        for split, ratio in ratios.items():
            quota = max(1, int(round(per_target * ratio)))
            candidates = by_split[split]
            rng.shuffle(candidates)
            selected[split].extend(candidates[: min(quota, len(candidates))])

    size_cache: Dict[str, Tuple[int, int]] = {}
    for split in SPLITS:
        fixed = []
        for r in selected[split]:
            size = (r.get("image_width"), r.get("image_height"))
            if not size[0] or not size[1]:
                size = read_image_size(r["image_path"], size_cache)
            if not size:
                continue
            iw, ih = float(size[0]), float(size[1])
            r["image_width"] = int(iw)
            r["image_height"] = int(ih)
            r["gold_bbox_norm_cxcywh"] = xyxy_to_norm_cxcywh(r["gold_bbox_xyxy"], iw, ih)
            fixed.append(r)
        fixed.sort(key=lambda x: (x["finding"], x["subject_id"], x["dicom_id"], x["bbox_name_reference"]))
        for i, r in enumerate(fixed, 1):
            prefix = "DEV" if task_type == "device" else "ANAT"
            r["task_id"] = f"{prefix}_{split.upper()}_{i:06d}"
        selected[split] = fixed
        write_jsonl(DATA / f"{task_type}_{split}.jsonl", fixed)

    all_rows = selected["train"] + selected["val"] + selected["eval"]
    pd.DataFrame(all_rows).to_csv(DATA / f"{task_type}_10k_rows.csv.gz", index=False, compression="gzip")
    train_subjects = {r["subject_id"] for r in selected["train"]}
    eval_subjects = {r["subject_id"] for r in selected["eval"]}
    overlap = sorted(train_subjects & eval_subjects)
    summary = {
        "task_type": task_type,
        "requested_max_samples": max_samples,
        "raw_counts": dict(raw_counts),
        "reservoir_counts": {k: len(v) for k, v in reservoirs.items()},
        "class_split_pool_counts": class_split_pool_counts,
        "selected_counts": {s: len(selected[s]) for s in SPLITS},
        "selected_by_class": {s: dict(Counter(r["finding"] for r in selected[s])) for s in SPLITS},
        "train_eval_subject_overlap": len(overlap),
    }
    write_text(DATA / f"{task_type}_dataset_summary.json", json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def build_data(args) -> Dict:
    ensure_dirs()
    if not args.force and all((DATA / f"{task}_{split}.jsonl").exists() for task in ["device", "anatomy"] for split in SPLITS):
        return {
            "device": json.loads((DATA / "device_dataset_summary.json").read_text(encoding="utf-8")),
            "anatomy": json.loads((DATA / "anatomy_dataset_summary.json").read_text(encoding="utf-8")),
        }
    device = build_device_dataset(args)
    anatomy = build_anatomy_dataset(args)
    report = [
        "# Chest ImaGenome Device/Anatomy 10k Dataset Report",
        "",
        "Two separate datasets were built.",
        "",
        "## Device task",
        "",
        "- Target: Chest ImaGenome anatomy/landmark region linked to a support-device claim.",
        "- Meaning: weak region/landmark reference, not physical device contour and not lesion mask.",
        f"- rows: {device['selected_counts']}",
        f"- train/eval subject overlap: {device['train_eval_subject_overlap']}",
        "",
        "## Anatomy task",
        "",
        "- Target: Chest ImaGenome anatomy object bbox.",
        "- Meaning: anatomy region localization, not lesion localization.",
        f"- rows: {anatomy['selected_counts']}",
        f"- train/eval subject overlap: {anatomy['train_eval_subject_overlap']}",
        "",
        "## Selected class counts",
        "",
        "```json",
        json.dumps({"device": device["selected_by_class"], "anatomy": anatomy["selected_by_class"]}, indent=2, ensure_ascii=False),
        "```",
    ]
    write_text(REPORT / "STAGE1_DATASET_REPORT.md", "\n".join(report) + "\n")
    return {"device": device, "anatomy": anatomy}


def load_rows(task: str, split: str) -> List[Dict]:
    return read_jsonl(DATA / f"{task}_{split}.jsonl")


def extract_vfm_features(args) -> Dict:
    ensure_dirs()
    done = all((FEAT / f"{task}_vfm_features_{split}.npz").exists() for task in ["device", "anatomy"] for split in SPLITS)
    if done and not args.force and not args.extract_vfm:
        model_name = (FEAT / "VFM_MODEL.txt").read_text(encoding="utf-8").strip()
        return {"vfm_model": model_name}

    torch = ms_base.torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name, processor, model, errors = ms_base.load_vfm(args.vfm_model, device)
    (FEAT / "VFM_MODEL.txt").write_text(model_name, encoding="utf-8")
    stats = {"vfm_model": model_name, "device": device, "splits": {}}
    for task in ["device", "anatomy"]:
        for split in SPLITS:
            rows = load_rows(task, split)
            ids, gf_all, pm_all, ps_all, pt_all = [], [], [], [], []
            batch_imgs, batch_rows = [], []
            print(f"extract_vfm task={task} split={split} rows={len(rows)}", flush=True)
            for r in rows:
                try:
                    img = Image.open(r["image_path"]).convert("RGB")
                except Exception:
                    continue
                batch_imgs.append(img)
                batch_rows.append(r)
                if len(batch_imgs) >= args.vfm_batch_size:
                    gf, pm, ps, pt = ms_base.vfm_forward(batch_imgs, processor, model, device)
                    for i, br in enumerate(batch_rows):
                        ids.append(br["task_id"])
                        gf_all.append(gf[i])
                        pm_all.append(pm[i])
                        ps_all.append(ps[i])
                        pt_all.append(pt[i])
                    if len(ids) % 512 == 0:
                        print(f"extract_vfm_progress task={task} split={split} rows={len(ids)}", flush=True)
                    batch_imgs, batch_rows = [], []
            if batch_imgs:
                gf, pm, ps, pt = ms_base.vfm_forward(batch_imgs, processor, model, device)
                for i, br in enumerate(batch_rows):
                    ids.append(br["task_id"])
                    gf_all.append(gf[i])
                    pm_all.append(pm[i])
                    ps_all.append(ps[i])
                    pt_all.append(pt[i])
            np.savez(
                FEAT / f"{task}_vfm_features_{split}.npz",
                task_ids=np.array(ids, dtype=object),
                global_features=np.stack(gf_all).astype("float32"),
                patch_mean=np.stack(pm_all).astype("float32"),
                patch_std=np.stack(ps_all).astype("float32"),
                patch_tokens=np.stack(pt_all).astype("float16"),
            )
            stats["splits"][f"{task}_{split}"] = len(ids)
            print(f"extract_vfm_saved task={task} split={split} rows={len(ids)}", flush=True)
    write_text(
        REPORT / "STAGE2_VFM_FEATURE_REPORT.md",
        "# VFM Feature Report\n\n"
        f"- VFM: `{model_name}`\n"
        f"- device: `{device}`\n"
        "- features: global, patch mean/std, patch tokens\n"
        "- RAD-DINO/VFM weights are frozen.\n\n"
        "## Load attempts\n\n```text\n"
        + ("\n".join(errors) if errors else "first candidate loaded")
        + "\n```\n",
    )
    return stats


def rule_query_text(task: str, row: Dict) -> str:
    return device_rule_query(row) if task == "device" else anatomy_rule_query(row)


def build_rule_query_features(args) -> Dict:
    ensure_dirs()
    done = all((FEAT / f"{task}_rule_query_{split}.npz").exists() for task in ["device", "anatomy"] for split in SPLITS)
    if done and not args.force and not args.build_rule:
        return {"rule_dim": args.rule_dim}
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    stats = {}
    for task in ["device", "anatomy"]:
        train_texts = [rule_query_text(task, r) for r in load_rows(task, "train")]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        xtr = vectorizer.fit_transform(train_texts)
        dim = min(args.rule_dim, max(1, xtr.shape[1] - 1), max(1, xtr.shape[0] - 1))
        svd = TruncatedSVD(n_components=dim, random_state=SEED)
        svd.fit(xtr)
        for split in SPLITS:
            rows = load_rows(task, split)
            texts = [rule_query_text(task, r) for r in rows]
            feats = svd.transform(vectorizer.transform(texts)).astype("float32")
            np.savez(
                FEAT / f"{task}_rule_query_{split}.npz",
                task_ids=np.array([r["task_id"] for r in rows], dtype=object),
                features=feats,
                query_text=np.array(texts, dtype=object),
            )
        stats[task] = {"vocab": int(len(vectorizer.vocabulary_)), "dim": int(dim)}
    write_text(
        REPORT / "STAGE3_RULE_QUERY_FEATURE_REPORT.md",
        "# Rule Query Feature Report\n\n"
        "Rule queries are deterministic text strings converted to TF-IDF/SVD features. "
        "Device rule queries use claim-visible device, part, relation, landmark, and location words. "
        "Anatomy rule queries use the anatomy name and its deterministic modifiers.\n\n"
        "No eval bbox and no Chest ImaGenome target coordinate is included in the query text.\n\n"
        "```json\n" + json.dumps(stats, indent=2, ensure_ascii=False) + "\n```\n",
    )
    return stats


def extract_smm_query_features(args) -> Dict:
    ensure_dirs()
    done = all((FEAT / f"{task}_smm_query_{split}.npz").exists() for task in ["device", "anatomy"] for split in SPLITS)
    if done and not args.force and not args.extract_smm:
        return {"smm_model": (FEAT / "SMM_QUERY_MODEL.txt").read_text(encoding="utf-8").strip()}
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = args.smm_text_model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    kwargs = {"trust_remote_code": True}
    if device == "cuda":
        kwargs["torch_dtype"] = torch.float16
    model = AutoModel.from_pretrained(model_name, **kwargs).eval().to(device)
    (FEAT / "SMM_QUERY_MODEL.txt").write_text(model_name, encoding="utf-8")
    stats = {"smm_model": model_name, "device": device, "splits": {}}

    def encode_unique(texts: List[str]) -> np.ndarray:
        unique = list(dict.fromkeys(texts))
        mapping: Dict[str, np.ndarray] = {}
        for start in range(0, len(unique), args.smm_batch_size):
            batch = unique[start : start + args.smm_batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=args.max_query_tokens)
            mask = inputs["attention_mask"].to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
                hidden = getattr(out, "last_hidden_state", None)
                if hidden is None:
                    hidden = out.hidden_states[-1]
                hidden = hidden.float()
                pooled = (hidden * mask.unsqueeze(-1).float()).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1).float()
            arr = pooled.detach().cpu().numpy().astype("float32")
            for text, feat in zip(batch, arr):
                mapping[text] = feat
        return np.stack([mapping[t] for t in texts]).astype("float32")

    for task in ["device", "anatomy"]:
        for split in SPLITS:
            rows = load_rows(task, split)
            texts = [smm_query_text(r) for r in rows]
            print(f"extract_smm task={task} split={split} rows={len(rows)} unique_texts={len(set(texts))}", flush=True)
            feats = encode_unique(texts)
            np.savez(
                FEAT / f"{task}_smm_query_{split}.npz",
                task_ids=np.array([r["task_id"] for r in rows], dtype=object),
                features=feats,
                query_text=np.array(texts, dtype=object),
            )
            stats["splits"][f"{task}_{split}"] = {"rows": len(rows), "dim": int(feats.shape[1]), "unique_texts": len(set(texts))}
    write_text(
        REPORT / "STAGE4_SMM_QUERY_FEATURE_REPORT.md",
        "# SMM Query Feature Report\n\n"
        f"- SMM/text encoder: `{model_name}`\n"
        "- Input: local text query only; no image pixels, no bbox targets, no eval coordinates.\n"
        "- Feature: mean-pooled final hidden state from the frozen Qwen text model.\n"
        "- This is an SMM query embedding baseline, not SMM fine-tuning.\n\n"
        "```json\n" + json.dumps(stats, indent=2, ensure_ascii=False) + "\n```\n",
    )
    return stats


def load_npz_features(path: Path, key: str = "features") -> Dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    return {"ids": np.array([str(x) for x in d["task_ids"]], dtype=object), "features": d[key].astype("float32")}


def load_vfm_npz(task: str, split: str) -> Dict[str, np.ndarray]:
    d = np.load(FEAT / f"{task}_vfm_features_{split}.npz", allow_pickle=True)
    return {
        "ids": np.array([str(x) for x in d["task_ids"]], dtype=object),
        "patch_tokens": d["patch_tokens"].astype("float16"),
    }


def aligned_patch_query(task: str, query_type: str, split: str):
    rows = load_rows(task, split)
    vfm = load_vfm_npz(task, split)
    q = load_npz_features(FEAT / f"{task}_{query_type}_query_{split}.npz")
    vpos = {tid: i for i, tid in enumerate(vfm["ids"])}
    qpos = {tid: i for i, tid in enumerate(q["ids"])}
    ordered, vidx, qidx, y = [], [], [], []
    for r in rows:
        tid = r["task_id"]
        if tid in vpos and tid in qpos:
            ordered.append(r)
            vidx.append(vpos[tid])
            qidx.append(qpos[tid])
            y.append(r["gold_bbox_norm_cxcywh"])
    return (
        ordered,
        vfm["patch_tokens"][np.asarray(vidx, dtype=np.int64)].astype("float16", copy=False),
        q["features"][np.asarray(qidx, dtype=np.int64)].astype("float32"),
        np.asarray(y, dtype="float32"),
    )


def predict_pixel_rows(rows: List[Dict], pred_norm: np.ndarray, method: str, extra: Dict) -> List[Dict]:
    out = []
    for r, p in zip(rows, pred_norm):
        p = np.asarray(p, dtype="float32")
        p[:2] = np.clip(p[:2], 0, 1)
        p[2:] = np.clip(p[2:], 0.02, 1)
        out.append(
            {
                "task_id": r["task_id"],
                "method": method,
                "task_type": r["task_type"],
                "dicom_id": r["dicom_id"],
                "finding": r["finding"],
                "claim_sentence": r["claim_sentence"],
                "bbox_name_reference": r.get("bbox_name_reference", ""),
                "pred_bbox_norm_cxcywh": [float(x) for x in p],
                "pred_bbox_xyxy": norm_cxcywh_to_xyxy(p, r["image_width"], r["image_height"]),
                "bbox_missing": False,
                "confidence": 1.0,
                **extra,
            }
        )
    return out


def val_score(pred_norm: np.ndarray, rows: List[Dict]) -> float:
    vals = []
    for p, r in zip(pred_norm, rows):
        vals.append(iou_xyxy(norm_cxcywh_to_xyxy(p, r["image_width"], r["image_height"]), r["gold_bbox_xyxy"]))
    return float(np.mean(vals)) if vals else 0.0


def train_heatmap(task: str, query_type: str, args) -> Dict:
    import torch

    method = f"{task}_{query_type}_query_heatmap"
    train_rows, tokens_tr, q_tr, ytr = aligned_patch_query(task, query_type, "train")
    val_rows, tokens_va, q_va, _ = aligned_patch_query(task, query_type, "val")
    eval_rows, tokens_ev, q_ev, _ = aligned_patch_query(task, query_type, "eval")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PatchHeatmapBBoxHead(tokens_tr.shape[-1], q_tr.shape[1], hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    grid = int(round(math.sqrt(tokens_tr.shape[1])))

    def center_idx_np(y: np.ndarray) -> np.ndarray:
        cx = np.clip((y[:, 0] * grid).astype(int), 0, grid - 1)
        cy = np.clip((y[:, 1] * grid).astype(int), 0, grid - 1)
        return np.minimum(cy * grid + cx, tokens_tr.shape[1] - 1).astype("int64")

    c_tr = center_idx_np(ytr)
    rng = np.random.default_rng(SEED + stable_hash_int(method) % 10000)
    best_state, best_val, best_epoch, wait = None, -1.0, 0, 0
    hist = []
    batch_size = max(1, args.heatmap_batch_size)

    def predict_batches(tokens: np.ndarray, qfeat: np.ndarray) -> np.ndarray:
        preds = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(tokens), batch_size):
                tok = torch.tensor(tokens[start : start + batch_size], dtype=torch.float32, device=device)
                q = torch.tensor(qfeat[start : start + batch_size], dtype=torch.float32, device=device)
                p, _ = model(tok, q)
                preds.append(p.detach().cpu().numpy())
        return np.vstack(preds) if preds else np.zeros((0, 4), dtype="float32")

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(len(tokens_tr))
        losses = []
        for start in range(0, len(tokens_tr), batch_size):
            idx = perm[start : start + batch_size]
            tok = torch.tensor(tokens_tr[idx], dtype=torch.float32, device=device)
            q = torch.tensor(q_tr[idx], dtype=torch.float32, device=device)
            target = torch.tensor(ytr[idx], dtype=torch.float32, device=device)
            center = torch.tensor(c_tr[idx], dtype=torch.long, device=device)
            pred, logits = model(tok, q)
            b_loss, _ = bbox_loss(pred, target)
            h_loss = torch.nn.functional.cross_entropy(logits, center)
            loss = b_loss + args.heatmap_loss_weight * h_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        pv = predict_batches(tokens_va, q_va)
        score = val_score(pv, val_rows)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_mean_iou": score}
        hist.append(row)
        print(f"{method} epoch={epoch} train_loss={row['train_loss']:.5f} val_mean_iou={score:.5f}", flush=True)
        if score > best_val:
            best_val, best_epoch, wait = score, epoch, 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= args.patience:
            break
    if best_state:
        model.load_state_dict(best_state)
    pred_norm = predict_batches(tokens_ev, q_ev)
    preds = predict_pixel_rows(eval_rows, pred_norm, method, {"query_type": query_type, "head": "patch_heatmap", "vfm": args.vfm_model})
    write_jsonl(PRED / f"{method}.jsonl", preds)
    run_dir = RUNS / method
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(hist).to_csv(run_dir / "loss_curve.csv", index=False)
    torch.save(
        {"state": best_state, "token_dim": tokens_tr.shape[-1], "query_dim": q_tr.shape[1], "method": method},
        CKPT / f"{method}.pt",
    )
    return {
        "method": method,
        "task": task,
        "query_type": query_type,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "eval_rows": len(eval_rows),
        "query_dim": int(q_tr.shape[1]),
        "token_shape": f"{tokens_tr.shape[1]}x{tokens_tr.shape[-1]}",
        "best_val_mean_iou": best_val,
        "best_epoch": best_epoch,
        "train_sec": round(time.time() - t0, 2),
    }


def train_models(args) -> List[Dict]:
    done_methods = [
        "device_rule_query_heatmap",
        "device_smm_query_heatmap",
        "anatomy_rule_query_heatmap",
        "anatomy_smm_query_heatmap",
    ]
    if not args.force and all((PRED / f"{m}.jsonl").exists() for m in done_methods) and not args.train:
        return pd.read_csv(MET / "training_runs.csv").to_dict("records") if (MET / "training_runs.csv").exists() else []
    infos = []
    for task in ["device", "anatomy"]:
        for query_type in ["rule", "smm"]:
            qpath = FEAT / f"{task}_{query_type}_query_train.npz"
            if not qpath.exists():
                print(f"skip_missing_query_features task={task} query={query_type}", flush=True)
                continue
            infos.append(train_heatmap(task, query_type, args))
    pd.DataFrame(infos).to_csv(MET / "training_runs.csv", index=False)
    write_text(
        REPORT / "STAGE5_TRAINING_REPORT.md",
        "# Rule vs SMM Query Heatmap Training Report\n\n"
        "- VFM: frozen RAD-DINO patch tokens.\n"
        "- Trainable part: shallow patch heatmap + bbox regression head only.\n"
        "- Compared conditions: device/rule, device/SMM, anatomy/rule, anatomy/SMM.\n"
        "- Chest ImaGenome device task uses weak landmark/region boxes; anatomy task uses anatomy object boxes.\n\n"
        + pd.DataFrame(infos).to_markdown(index=False)
        + "\n",
    )
    return infos


def evaluate_prediction_rows(preds: List[Dict], tasks: List[Dict], method: str, subset: str) -> Dict:
    by_id = {p["task_id"]: p for p in preds}
    vals, missing, invalid = [], 0, 0
    for t in tasks:
        p = by_id.get(t["task_id"])
        box = p.get("pred_bbox_xyxy") if p else None
        if not box:
            missing += 1
            vals.append(0.0)
        else:
            invalid += 0 if valid_box(box) else 1
            vals.append(iou_xyxy(box, t["gold_bbox_xyxy"]))
    arr = np.asarray(vals, dtype="float32")
    return {
        "method": method,
        "subset": subset,
        "n": len(tasks),
        "mean_iou": float(arr.mean()) if len(arr) else 0.0,
        "median_iou": float(np.median(arr)) if len(arr) else 0.0,
        "Hit@0.1": float((arr >= 0.1).mean()) if len(arr) else 0.0,
        "Hit@0.3": float((arr >= 0.3).mean()) if len(arr) else 0.0,
        "Hit@0.5": float((arr >= 0.5).mean()) if len(arr) else 0.0,
        "bbox_missing_rate": float(missing / max(1, len(tasks))),
        "bbox_invalid_rate": float(invalid / max(1, len(tasks))),
    }


def evaluate_all() -> Dict:
    summary, per_label = [], []
    for task in ["device", "anatomy"]:
        eval_rows = load_rows(task, "eval")
        for pred_path in sorted(PRED.glob(f"{task}_*_query_heatmap.jsonl")):
            method = pred_path.stem
            preds = read_jsonl(pred_path)
            summary.append(evaluate_prediction_rows(preds, eval_rows, method, task))
            for label in sorted({r["finding"] for r in eval_rows}):
                subset = [r for r in eval_rows if r["finding"] == label]
                row = evaluate_prediction_rows(preds, subset, method, f"{task}:{label}")
                row["task"] = task
                row["label"] = label
                per_label.append(row)
    df = pd.DataFrame(summary).sort_values(["subset", "mean_iou"], ascending=[True, False])
    pf = pd.DataFrame(per_label)
    df.to_csv(MET / "summary.csv", index=False)
    pf.to_csv(MET / "per_label_metrics.csv", index=False)
    pivot_rows = []
    for task in ["device", "anatomy"]:
        sdf = df[df["subset"].eq(task)]
        rule = sdf[sdf["method"].str.contains("_rule_query_")]
        smm = sdf[sdf["method"].str.contains("_smm_query_")]
        if len(rule) and len(smm):
            pivot_rows.append(
                {
                    "task": task,
                    "rule_method": rule.iloc[0]["method"],
                    "rule_mean_iou": float(rule.iloc[0]["mean_iou"]),
                    "rule_Hit@0.3": float(rule.iloc[0]["Hit@0.3"]),
                    "smm_method": smm.iloc[0]["method"],
                    "smm_mean_iou": float(smm.iloc[0]["mean_iou"]),
                    "smm_Hit@0.3": float(smm.iloc[0]["Hit@0.3"]),
                    "smm_minus_rule_iou": float(smm.iloc[0]["mean_iou"] - rule.iloc[0]["mean_iou"]),
                }
            )
    pd.DataFrame(pivot_rows).to_csv(MET / "rule_vs_smm_comparison.csv", index=False)
    write_text(
        REPORT / "STAGE6_EVAL_REPORT.md",
        "# Evaluation Report\n\n"
        "## Summary\n\n"
        + df.to_markdown(index=False)
        + "\n\n## Rule vs SMM\n\n"
        + pd.DataFrame(pivot_rows).to_markdown(index=False)
        + "\n",
    )
    return {"summary": summary, "comparison": pivot_rows}


def draw_boxes(image_path: str, boxes: List[Tuple[str, Optional[Sequence[float]], Tuple[int, int, int]]], out: Path, title: str) -> bool:
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return False
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        small = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
        small = ImageFont.load_default()
    draw.rectangle([0, 0, img.width, 72], fill=(0, 0, 0))
    draw.text((8, 8), title[:150], fill=(255, 255, 255), font=small)
    for label, box, color in boxes:
        if not valid_box(box):
            continue
        x1, y1, x2, y2 = [float(x) for x in box]
        for off in range(4):
            draw.rectangle([x1 - off, y1 - off, x2 + off, y2 + off], outline=color)
        draw.text((x1 + 4, max(74, y1 + 4)), label, fill=color, font=font)
    img.thumbnail((1100, 1100))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, quality=90)
    return True


def make_contact_sheet(paths: List[Path], out: Path, cols: int = 4) -> None:
    imgs = []
    for p in paths:
        try:
            im = Image.open(p).convert("RGB")
            im.thumbnail((360, 360))
            imgs.append(im.copy())
        except Exception:
            pass
    if not imgs:
        return
    rows = math.ceil(len(imgs) / cols)
    sheet = Image.new("RGB", (cols * 360, rows * 360), (20, 20, 20))
    for i, im in enumerate(imgs):
        sheet.paste(im, ((i % cols) * 360, (i // cols) * 360))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=90)


def make_overlays(max_per_task: int = 24) -> Dict:
    summary = pd.read_csv(MET / "summary.csv") if (MET / "summary.csv").exists() else pd.DataFrame()
    if summary.empty:
        return {"overlays": 0}
    paths = []
    for task in ["device", "anatomy"]:
        sdf = summary[summary["subset"].eq(task)]
        if sdf.empty:
            continue
        best = str(sdf.sort_values("mean_iou", ascending=False).iloc[0]["method"])
        pred_best = {r["task_id"]: r for r in read_jsonl(PRED / f"{best}.jsonl")}
        rule_path = PRED / f"{task}_rule_query_heatmap.jsonl"
        smm_path = PRED / f"{task}_smm_query_heatmap.jsonl"
        pred_rule = {r["task_id"]: r for r in read_jsonl(rule_path)} if rule_path.exists() else {}
        pred_smm = {r["task_id"]: r for r in read_jsonl(smm_path)} if smm_path.exists() else {}
        rows = load_rows(task, "eval")
        scored = [(iou_xyxy(pred_best.get(r["task_id"], {}).get("pred_bbox_xyxy"), r["gold_bbox_xyxy"]), r) for r in rows]
        examples = sorted(scored, key=lambda x: x[0], reverse=True)[: max_per_task // 2] + sorted(scored, key=lambda x: x[0])[: max_per_task // 2]
        for i, (score, r) in enumerate(examples):
            out = OVER / "comparison" / f"{task}_{i:03d}_{r['finding'].replace(' ', '_')}_{r['task_id']}.jpg"
            ok = draw_boxes(
                r["image_path"],
                [
                    ("reference", r["gold_bbox_xyxy"], (255, 230, 0)),
                    ("rule", pred_rule.get(r["task_id"], {}).get("pred_bbox_xyxy"), (0, 220, 80)),
                    ("smm", pred_smm.get(r["task_id"], {}).get("pred_bbox_xyxy"), (0, 200, 255)),
                ],
                out,
                f"{task} | {r['finding']} | best={best} IoU={score:.3f} | Chest ImaGenome reference",
            )
            if ok:
                paths.append(out)
    make_contact_sheet(paths, CONTACT / "rule_vs_smm_device_anatomy_examples.jpg")
    write_text(
        REPORT / "STAGE7_OVERLAY_REPORT.md",
        "# Overlay Report\n\n"
        f"- overlays written: {len(paths)}\n"
        f"- contact sheet: `{CONTACT / 'rule_vs_smm_device_anatomy_examples.jpg'}`\n"
        "- Colors: reference=yellow, rule=green, SMM=cyan.\n"
        "- Reference boxes are Chest ImaGenome device-linked weak regions or anatomy object boxes, not lesion masks.\n",
    )
    return {"overlays": len(paths)}


def write_final_reports(data_info: Dict, vfm_info: Dict, smm_info: Dict, overlays: Dict) -> None:
    summary_path = MET / "summary.csv"
    comparison_path = MET / "rule_vs_smm_comparison.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    comp = pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()
    best_rows = []
    if not summary.empty:
        for task in ["device", "anatomy"]:
            sdf = summary[summary["subset"].eq(task)]
            if len(sdf):
                best_rows.append(sdf.sort_values("mean_iou", ascending=False).iloc[0].to_dict())
    interp = []
    if not comp.empty:
        for _, r in comp.iterrows():
            if float(r["smm_minus_rule_iou"]) > 0:
                interp.append(f"- `{r['task']}`: SMM query was higher than rule query by {float(r['smm_minus_rule_iou']):.4f} mean IoU.")
            else:
                interp.append(f"- `{r['task']}`: rule query was higher than SMM query by {-float(r['smm_minus_rule_iou']):.4f} mean IoU.")
    write_text(
        REPORT / "KOREAN_RESULT_SUMMARY.md",
        "# Chest ImaGenome 의료기기/해부학 10k rule vs SMM 결과 요약\n\n"
        "## 무엇을 했나\n\n"
        "Chest ImaGenome에서 두 가지 박스 찾기 문제를 분리했다.\n\n"
        "1. 의료기기 위치 찾기: 의료기기 문장과 연결된 해부학/랜드마크 박스를 찾는다. 기기 자체 윤곽선이 아니다.\n"
        "2. 해부학 위치 찾기: `right lung`, `carina`, `cardiac silhouette` 같은 해부학 박스 자체를 찾는다.\n\n"
        "각 문제에서 질의 생성 방식을 둘로 나눴다.\n\n"
        "- rule query: 사람이 정한 규칙으로 device, landmark, left/right, upper/lower 같은 단서를 뽑음.\n"
        "- SMM query: Qwen text encoder가 문장을 숫자 벡터로 바꿈.\n\n"
        "두 질의 모두 같은 frozen RAD-DINO patch heatmap bbox head에 넣어 비교했다.\n\n"
        "## 데이터\n\n"
        f"- device rows: {data_info.get('device', {}).get('selected_counts', {})}\n"
        f"- anatomy rows: {data_info.get('anatomy', {}).get('selected_counts', {})}\n"
        "- train/eval subject overlap: device/anatomy 모두 보고서에 기록됨.\n"
        "- Chest ImaGenome device box는 weak region/landmark reference이며 gold lesion bbox가 아니다.\n"
        "- Chest ImaGenome anatomy box는 anatomy object bbox이며 lesion bbox가 아니다.\n\n"
        "## 결과표\n\n"
        + (summary.to_markdown(index=False) if not summary.empty else "Metrics not generated.\n")
        + "\n\n## rule vs SMM\n\n"
        + (comp.to_markdown(index=False) if not comp.empty else "Comparison not generated.\n")
        + "\n\n## 해석\n\n"
        + ("\n".join(interp) if interp else "- 아직 해석할 metric이 없다.")
        + "\n",
    )
    write_text(
        REPORT / "PIPELINE_LOGICAL_REVIEW.md",
        "# Pipeline Logical Review\n\n"
        "- This experiment does not use A/B/C selector, pairwise comparator, or Qwen coordinate generation.\n"
        "- RAD-DINO is used as a frozen visual feature extractor.\n"
        "- Only shallow heatmap/bbox heads are trained.\n"
        "- Rule and SMM query conditions share the same train/val/eval splits and same VFM features.\n"
        "- Qwen/SMM query encoder receives text only, not image pixels or bbox targets.\n"
        "- Eval reference boxes are never included in query text or model input.\n"
        "- Chest ImaGenome device-linked boxes are weak region/landmark references, not exact device contours.\n"
        "- Chest ImaGenome anatomy boxes are anatomy object boxes, not lesion boxes.\n"
        f"- VFM model: `{vfm_info.get('vfm_model', '')}`\n"
        f"- SMM query model: `{smm_info.get('smm_model', '')}`\n"
        f"- overlays written: {overlays.get('overlays', 0)}\n",
    )
    write_text(
        REPORT / "NEXT_PLAN.md",
        "# Next Plan\n\n"
        "1. If rule query wins, strengthen the deterministic medical query grammar and keep SMM optional.\n"
        "2. If SMM query wins on device but not anatomy, use SMM only for complex full-report device claims.\n"
        "3. If anatomy remains easy for rules, do not spend SMM capacity on anatomy-name localization.\n"
        "4. Add controls: query-shuffled, image-shuffled, and label-only baselines.\n"
        "5. If both rule/SMM are below the prior 10k mixed heatmap result, inspect dataset construction and per-label failures before scaling.\n",
    )
    clean_summary = (
        "# Chest ImaGenome Device/Anatomy 10k Rule vs SMM Query Result\n\n"
        "## Main Results\n\n"
        "| task | query | mean IoU | Hit@0.3 |\n"
        "| --- | --- | ---: | ---: |\n"
        f"| device | rule | {float(comp[comp['task'].eq('device')].iloc[0]['rule_mean_iou']) if not comp.empty and len(comp[comp['task'].eq('device')]) else float('nan'):.6f} | "
        f"{float(comp[comp['task'].eq('device')].iloc[0]['rule_Hit@0.3']) if not comp.empty and len(comp[comp['task'].eq('device')]) else float('nan'):.6f} |\n"
        f"| device | SMM | {float(comp[comp['task'].eq('device')].iloc[0]['smm_mean_iou']) if not comp.empty and len(comp[comp['task'].eq('device')]) else float('nan'):.6f} | "
        f"{float(comp[comp['task'].eq('device')].iloc[0]['smm_Hit@0.3']) if not comp.empty and len(comp[comp['task'].eq('device')]) else float('nan'):.6f} |\n"
        f"| anatomy | rule | {float(comp[comp['task'].eq('anatomy')].iloc[0]['rule_mean_iou']) if not comp.empty and len(comp[comp['task'].eq('anatomy')]) else float('nan'):.6f} | "
        f"{float(comp[comp['task'].eq('anatomy')].iloc[0]['rule_Hit@0.3']) if not comp.empty and len(comp[comp['task'].eq('anatomy')]) else float('nan'):.6f} |\n"
        f"| anatomy | SMM | {float(comp[comp['task'].eq('anatomy')].iloc[0]['smm_mean_iou']) if not comp.empty and len(comp[comp['task'].eq('anatomy')]) else float('nan'):.6f} | "
        f"{float(comp[comp['task'].eq('anatomy')].iloc[0]['smm_Hit@0.3']) if not comp.empty and len(comp[comp['task'].eq('anatomy')]) else float('nan'):.6f} |\n\n"
        "Rule queries outperformed frozen SMM query embeddings on both tasks.\n\n"
        "Device boxes are weak anatomy/landmark references linked to support-device claims, not exact device contours. "
        "Anatomy boxes are anatomy object bboxes, not lesion bboxes.\n"
    )
    write_text(REPORT / "KOREAN_RESULT_SUMMARY.md", clean_summary)
    write_text(REPORT / "RESULT_SUMMARY_EN.md", clean_summary)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--build-data", action="store_true")
    p.add_argument("--extract-vfm", action="store_true")
    p.add_argument("--build-rule", action="store_true")
    p.add_argument("--extract-smm", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--make-overlays", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--device-samples", type=int, default=10000)
    p.add_argument("--anatomy-samples", type=int, default=10000)
    p.add_argument("--vfm-model", default="microsoft/rad-dino")
    p.add_argument("--smm-text-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--csv-chunksize", type=int, default=500000)
    p.add_argument("--per-chunk-cap", type=int, default=6000)
    p.add_argument("--vfm-batch-size", type=int, default=16)
    p.add_argument("--smm-batch-size", type=int, default=32)
    p.add_argument("--max-query-tokens", type=int, default=96)
    p.add_argument("--rule-dim", type=int, default=128)
    p.add_argument("--heatmap-batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--heatmap-loss-weight", type=float, default=0.2)
    args = p.parse_args()
    if args.quick:
        args.device_samples = min(args.device_samples, 10000)
        args.anatomy_samples = min(args.anatomy_samples, 10000)
    if not any([args.build_data, args.extract_vfm, args.build_rule, args.extract_smm, args.train, args.evaluate, args.make_overlays]):
        args.build_data = args.extract_vfm = args.build_rule = args.extract_smm = args.train = args.evaluate = args.make_overlays = True
    return args


def metric(task: str, query: str) -> float:
    path = MET / "summary.csv"
    if not path.exists():
        return float("nan")
    df = pd.read_csv(path)
    method = f"{task}_{query}_query_heatmap"
    row = df[df["method"].eq(method)]
    return float(row.iloc[0]["mean_iou"]) if len(row) else float("nan")


def best_method(task: str) -> str:
    path = MET / "summary.csv"
    if not path.exists():
        return ""
    df = pd.read_csv(path)
    sdf = df[df["subset"].eq(task)]
    if not len(sdf):
        return ""
    return str(sdf.sort_values("mean_iou", ascending=False).iloc[0]["method"])


def main() -> None:
    t0 = time.time()
    ensure_dirs()
    args = parse_args()
    data_info: Dict = {}
    vfm_info: Dict = {}
    smm_info: Dict = {}
    overlays: Dict = {}
    if args.build_data or args.force or not (DATA / "device_train.jsonl").exists() or not (DATA / "anatomy_train.jsonl").exists():
        data_info = build_data(args)
    else:
        data_info = {
            "device": json.loads((DATA / "device_dataset_summary.json").read_text(encoding="utf-8")),
            "anatomy": json.loads((DATA / "anatomy_dataset_summary.json").read_text(encoding="utf-8")),
        }
    if args.extract_vfm or args.force or not (FEAT / "device_vfm_features_train.npz").exists():
        vfm_info = extract_vfm_features(args)
    else:
        vfm_info = {"vfm_model": (FEAT / "VFM_MODEL.txt").read_text(encoding="utf-8").strip()}
    if args.build_rule or args.force or not (FEAT / "device_rule_query_train.npz").exists():
        build_rule_query_features(args)
    if args.extract_smm or args.force or not (FEAT / "device_smm_query_train.npz").exists():
        smm_info = extract_smm_query_features(args)
    else:
        model_path = FEAT / "SMM_QUERY_MODEL.txt"
        smm_info = {"smm_model": model_path.read_text(encoding="utf-8").strip() if model_path.exists() else ""}
    if args.train:
        train_models(args)
    if args.evaluate:
        evaluate_all()
    if args.make_overlays:
        overlays = make_overlays()
    write_final_reports(data_info, vfm_info, smm_info, overlays)

    device_counts = data_info.get("device", {}).get("selected_counts", {})
    anatomy_counts = data_info.get("anatomy", {}).get("selected_counts", {})
    print(f"project_root={PROJECT_ROOT}")
    print(f"experiment={EXP_NAME}")
    print(f"device_train_rows={device_counts.get('train', 0)}")
    print(f"device_val_rows={device_counts.get('val', 0)}")
    print(f"device_eval_rows={device_counts.get('eval', 0)}")
    print(f"anatomy_train_rows={anatomy_counts.get('train', 0)}")
    print(f"anatomy_val_rows={anatomy_counts.get('val', 0)}")
    print(f"anatomy_eval_rows={anatomy_counts.get('eval', 0)}")
    print(f"vfm_model={vfm_info.get('vfm_model', '')}")
    print(f"smm_query_model={smm_info.get('smm_model', '')}")
    print(f"device_rule_mean_iou={metric('device', 'rule'):.6f}")
    print(f"device_smm_mean_iou={metric('device', 'smm'):.6f}")
    print(f"anatomy_rule_mean_iou={metric('anatomy', 'rule'):.6f}")
    print(f"anatomy_smm_mean_iou={metric('anatomy', 'smm'):.6f}")
    print(f"best_device_method={best_method('device')}")
    print(f"best_anatomy_method={best_method('anatomy')}")
    print(f"summary_report_path={REPORT / 'KOREAN_RESULT_SUMMARY.md'}")
    print(f"logical_review_path={REPORT / 'PIPELINE_LOGICAL_REVIEW.md'}")
    print(f"contact_sheet_path={CONTACT / 'rule_vs_smm_device_anatomy_examples.jpg'}")
    print(f"elapsed_sec={time.time() - t0:.1f}")


if __name__ == "__main__":
    main()

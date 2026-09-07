#!/usr/bin/env python
"""MS-CXR VFM localizer Stage 1 on local p10-p19 images.

This experiment trains claim/finding-conditioned bbox heads directly against
MS-CXR phrase-grounding boxes. It deliberately avoids Qwen A/B/C selector
experiments and direct SMM coordinate generation.

No MIMIC images or report-derived data are sent to external APIs. Public model
weights may be downloaded by Hugging Face if not cached locally.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.ensemble import ExtraTreesRegressor

from models_ms_cxr_vfm_localizer import (
    BBoxMLP,
    FeatureScaler,
    PatchHeatmapBBoxHead,
    bbox_loss,
    cxcywh_to_xyxy,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]

EXP_NAME = "ms_cxr_vfm_localizer_stage1_p10_p19"
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

MIMIC_JPG_ROOTS = [
    Path(p)
    for p in (
        os.environ.get("MIMIC_CXR_JPG_ROOT", ""),
        os.environ.get("MS_CXR_DATA_ROOT", ""),
        str(PROJECT_ROOT.parents[0]),
    )
    if p
]
P_PREFIXES = [f"p{i}" for i in range(10, 20)]
MAIN5 = ["Atelectasis", "Consolidation", "Lung Opacity", "Pleural Effusion", "Pneumothorax"]
SECONDARY = ["Cardiomegaly", "Edema", "Pneumonia"]
ALL8 = MAIN5 + SECONDARY
SEED = 20260622


def ensure_dirs() -> None:
    for p in [EXP, PRED, MET, OVER, CONTACT, TRAIN, DATA, RUNS, CKPT, FEAT, REPORT]:
        p.mkdir(parents=True, exist_ok=True)


def jd(obj) -> str:
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o).__name__)

    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=conv)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(jd(row) + "\n")
            n += 1
    return n


def clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def clip_box(box: Sequence[float], iw: float, ih: float) -> List[float]:
    x1, y1, x2, y2 = [float(x) for x in box]
    x1 = max(0.0, min(float(iw), x1))
    x2 = max(0.0, min(float(iw), x2))
    y1 = max(0.0, min(float(ih), y1))
    y2 = max(0.0, min(float(ih), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def valid_box(box: Sequence[float]) -> bool:
    return len(box) == 4 and float(box[2]) > float(box[0]) and float(box[3]) > float(box[1])


def xywh_to_xyxy(x: float, y: float, w: float, h: float) -> List[float]:
    return [float(x), float(y), float(x) + float(w), float(y) + float(h)]


def xyxy_to_xywh(box: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = [float(x) for x in box]
    return [x1, y1, x2 - x1, y2 - y1]


def xyxy_to_norm_cxcywh(box: Sequence[float], iw: float, ih: float) -> List[float]:
    x1, y1, x2, y2 = [float(x) for x in box]
    return [
        clip01(((x1 + x2) / 2) / iw),
        clip01(((y1 + y2) / 2) / ih),
        max(1e-4, min(1.0, (x2 - x1) / iw)),
        max(1e-4, min(1.0, (y2 - y1) / ih)),
    ]


def norm_cxcywh_to_xyxy(box: Sequence[float], iw: float, ih: float) -> List[float]:
    cx, cy, w, h = [float(x) for x in box]
    return clip_box([(cx - w / 2) * iw, (cy - h / 2) * ih, (cx + w / 2) * iw, (cy + h / 2) * ih], iw, ih)


def iou_xyxy(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> float:
    if not a or not b or not valid_box(a) or not valid_box(b):
        return 0.0
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def max_iou(pred: Optional[Sequence[float]], row: Dict) -> float:
    boxes = row.get("gold_group_bboxes_xyxy") or [row.get("gold_bbox_xyxy")]
    return max([iou_xyxy(pred, b) for b in boxes if b], default=0.0)


def find_image_root() -> Path:
    candidates = []
    for root in MIMIC_JPG_ROOTS:
        candidates.extend(
            [
                root / "mimic-cxr-jpg-p10-p14",
                root / "MIMIC-CXR-JPG" / "mimic-cxr-jpg-p10-p14",
                root,
            ]
        )
    for c in candidates:
        if c.exists() and all((c / p).exists() for p in P_PREFIXES):
            return c
    raise FileNotFoundError("Could not find a local MIMIC-CXR-JPG root containing p10-p19.")


def infer_path_parts(path: Path) -> Tuple[str, str, str, str]:
    parts = path.parts
    p_prefix = next((p for p in parts if p in P_PREFIXES), "")
    subject_id = next((p[1:] for p in parts if re.fullmatch(r"p\d{8}", p)), "")
    study_id = next((p[1:] for p in parts if re.fullmatch(r"s\d{8}", p)), "")
    dicom_id = path.stem
    return p_prefix, subject_id, study_id, dicom_id


def refresh_image_index(args) -> Dict:
    ensure_dirs()
    out = DATA / "jpg_file_index_p10_p19.csv.gz"
    if out.exists() and not args.force and not args.refresh_image_index:
        df = pd.read_csv(out)
    else:
        image_root = find_image_root()
        rows = []
        for p in P_PREFIXES:
            prefix_dir = image_root / p
            for jpg in prefix_dir.rglob("*.jpg"):
                pfx, sid, study, dicom = infer_path_parts(jpg)
                rel = str(jpg.relative_to(image_root)).replace("\\", "/")
                rows.append(
                    {
                        "dicom_id": dicom,
                        "subject_id": sid,
                        "study_id": study,
                        "p_prefix": pfx,
                        "jpg_relpath": rel,
                        "image_path": str(jpg),
                    }
                )
        df = pd.DataFrame(rows).sort_values(["p_prefix", "subject_id", "study_id", "dicom_id"])
        df.to_csv(out, index=False, compression="gzip")
    p_counts = df["p_prefix"].value_counts().sort_index().to_dict()
    dup = int(df["dicom_id"].duplicated().sum())
    sample = df.head(10).to_dict("records")
    pil_rows = []
    for r in sample:
        try:
            with Image.open(r["image_path"]) as img:
                pil_rows.append({**r, "pil_open": True, "image_width": img.size[0], "image_height": img.size[1]})
        except Exception as exc:
            pil_rows.append({**r, "pil_open": False, "error": repr(exc)})
    missing_prefix = [p for p in P_PREFIXES if p_counts.get(p, 0) == 0]
    report = [
        "# Stage 1 Image Index Report",
        "",
        f"- image index path: `{out}`",
        f"- total jpg files indexed: {len(df):,}",
        f"- duplicate dicom_id rows: {dup:,}",
        f"- missing p-prefixes: {missing_prefix if missing_prefix else 'none'}",
        "",
        "## Prefix Counts",
        "",
        "| prefix | jpg_count |",
        "|---|---:|",
    ]
    for p in P_PREFIXES:
        report.append(f"| {p} | {int(p_counts.get(p, 0)):,} |")
    report += ["", "## PIL Sample", "", pd.DataFrame(pil_rows).to_markdown(index=False)]
    write_text(REPORT / "STAGE1_IMAGE_INDEX_REPORT.md", "\n".join(report) + "\n")
    return {"jpg_rows": len(df), "p_counts": p_counts, "duplicate_dicom": dup, "image_index": str(out)}


def find_ms_cxr_annotations() -> Path:
    candidates = [
        PROJECT_ROOT / "dataset_audit" / "tables" / "ms_cxr_annotations.csv.gz",
        PROJECT_ROOT / "dataset_audit" / "tables" / "ms_cxr_local_p10_p19_annotations.csv.gz",
        PROJECT_ROOT / "dataset_audit" / "tables" / "ms_cxr_local_p10_p14_annotations.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    search_roots = [
        Path(p)
        for p in (
            os.environ.get("MS_CXR_DATA_ROOT", ""),
            os.environ.get("MIMIC_CXR_JPG_ROOT", ""),
            str(PROJECT_ROOT.parents[0]),
        )
        if p
    ]
    for root in search_roots:
        if not root.exists():
            continue
        for p in root.rglob("*MS_CXR*Local*Alignment*.csv*"):
            return p
    raise FileNotFoundError("MS-CXR annotation CSV not found.")


def rebuild_ms_cxr_match(args) -> Dict:
    ensure_dirs()
    out = DATA / "ms_cxr_p10_p19_annotations.csv.gz"
    if out.exists() and not args.force and not args.rebuild_ms_cxr_match:
        df = pd.read_csv(out)
    else:
        jpg = pd.read_csv(DATA / "jpg_file_index_p10_p19.csv.gz")
        img_by_dicom = jpg.drop_duplicates("dicom_id").set_index("dicom_id")
        src = find_ms_cxr_annotations()
        ms = pd.read_csv(src, compression="infer")
        rows = []
        for idx, r in ms.iterrows():
            dicom = str(r.get("dicom_id", "")).strip()
            if not dicom or dicom not in img_by_dicom.index:
                continue
            img = img_by_dicom.loc[dicom]
            finding = str(r.get("category_name") or r.get("finding") or "").strip()
            if finding not in ALL8:
                continue
            x, y, w, h = [float(r.get(k, 0)) for k in ["x", "y", "w", "h"]]
            iw = int(float(r.get("image_width") or 0))
            ih = int(float(r.get("image_height") or 0))
            if iw <= 0 or ih <= 0:
                try:
                    with Image.open(img["image_path"]) as im:
                        iw, ih = im.size
                except Exception:
                    continue
            box = clip_box(xywh_to_xyxy(x, y, w, h), iw, ih)
            if not valid_box(box):
                continue
            claim = str(r.get("label_text") or r.get("phrase") or r.get("sentence") or finding).strip()
            rows.append(
                {
                    "dicom_id": dicom,
                    "subject_id": str(img.get("subject_id") or r.get("subject_id") or ""),
                    "study_id": str(img.get("study_id") or r.get("study_id") or ""),
                    "image_path": str(img["image_path"]),
                    "jpg_relpath": str(img["jpg_relpath"]),
                    "p_prefix": str(img["p_prefix"]),
                    "finding": finding,
                    "claim_sentence": claim,
                    "phrase": claim,
                    "sentence": claim,
                    "gold_bbox_xyxy": jd(box),
                    "gold_bbox_xywh": jd(xyxy_to_xywh(box)),
                    "gold_bbox_norm_cxcywh": jd(xyxy_to_norm_cxcywh(box, iw, ih)),
                    "image_width": iw,
                    "image_height": ih,
                    "original_split": str(r.get("split", "")),
                    "view_position": str(r.get("ViewPosition") or r.get("view_position") or ""),
                    "ms_cxr_annotation_id": f"mscxr_p1019_{idx:06d}",
                    "bbox_type": "ms_cxr_phrase_grounding_bbox",
                    "gold_source": "MS-CXR",
                }
            )
        df = pd.DataFrame(rows)
        if len(df):
            key = df["dicom_id"] + "|" + df["finding"] + "|" + df["claim_sentence"]
            group_boxes = df.assign(row_key=key).groupby("row_key")["gold_bbox_xyxy"].apply(lambda s: [json.loads(x) for x in s]).to_dict()
            df["row_key"] = key
            df["gold_group_bboxes_xyxy"] = df["row_key"].map(lambda k: jd(group_boxes.get(k, [])))
            df["is_secondary_finding"] = df["finding"].isin(SECONDARY)
            df = df.sort_values(["p_prefix", "subject_id", "study_id", "finding", "ms_cxr_annotation_id"])
        df.to_csv(out, index=False, compression="gzip")
    src_total = len(pd.read_csv(find_ms_cxr_annotations(), usecols=["dicom_id"], compression="infer"))
    p_counts = df["p_prefix"].value_counts().sort_index().to_dict() if len(df) else {}
    cat_counts = df["finding"].value_counts().to_dict() if len(df) else {}
    report = [
        "# Stage 2 MS-CXR Match Report",
        "",
        f"- source MS-CXR annotations: {src_total:,}",
        f"- matched p10-p19 annotations: {len(df):,}",
        f"- unique matched images: {df['dicom_id'].nunique() if len(df) else 0:,}",
        f"- missing/unmatched annotations: {src_total - len(df):,}",
        "- MS-CXR bbox type: phrase-grounding bbox, not pixel-level lesion mask.",
        "- support_devices and lung_lesion are not MS-CXR categories and are excluded.",
        "",
        "## By Prefix",
        "",
        "| prefix | matched_annotations |",
        "|---|---:|",
    ]
    for p in P_PREFIXES:
        report.append(f"| {p} | {int(p_counts.get(p, 0)):,} |")
    report += ["", "## By Finding", "", "| finding | matched_annotations |", "|---|---:|"]
    for k, v in sorted(cat_counts.items()):
        report.append(f"| {k} | {int(v):,} |")
    write_text(REPORT / "STAGE2_MS_CXR_MATCH_REPORT.md", "\n".join(report) + "\n")
    return {"ms_total": src_total, "matched": len(df), "unique_images": int(df["dicom_id"].nunique()) if len(df) else 0}


def parse_json_list(value) -> List:
    if isinstance(value, list):
        return value
    try:
        return json.loads(value)
    except Exception:
        return []


def build_split(args) -> Dict:
    ensure_dirs()
    df = pd.read_csv(DATA / "ms_cxr_p10_p19_annotations.csv.gz")
    rng = random.Random(SEED)
    subjects = sorted(df["subject_id"].astype(str).unique())
    rng.shuffle(subjects)
    n = len(subjects)
    train_sub = set(subjects[: int(n * 0.70)])
    val_sub = set(subjects[int(n * 0.70) : int(n * 0.80)])
    split_map = {}
    for s in subjects:
        split_map[s] = "train" if s in train_sub else "val" if s in val_sub else "eval"
    df["split"] = df["subject_id"].astype(str).map(split_map)
    out_counts = {}
    subjects_by_split = {}
    for split in ["train", "val", "eval"]:
        rows = []
        for idx, r in df[df["split"].eq(split)].reset_index(drop=True).iterrows():
            box = parse_json_list(r["gold_bbox_xyxy"])
            norm = parse_json_list(r["gold_bbox_norm_cxcywh"])
            group_boxes = parse_json_list(r.get("gold_group_bboxes_xyxy", "[]"))
            row = {
                "task_id": f"MSCXR_P1019_{split.upper()}_{idx + 1:05d}",
                "dicom_id": str(r["dicom_id"]),
                "subject_id": str(r["subject_id"]),
                "study_id": str(r["study_id"]),
                "image_path": str(r["image_path"]),
                "finding": str(r["finding"]),
                "claim_sentence": str(r["claim_sentence"]),
                "phrase": str(r["phrase"]),
                "sentence": str(r["sentence"]),
                "gold_bbox_xyxy": box,
                "gold_bbox_xywh": parse_json_list(r["gold_bbox_xywh"]),
                "gold_bbox_norm_cxcywh": norm,
                "gold_group_bboxes_xyxy": group_boxes,
                "image_width": int(r["image_width"]),
                "image_height": int(r["image_height"]),
                "split": split,
                "original_split": str(r.get("original_split", "")),
                "view_position": str(r.get("view_position", "")),
                "p_prefix": str(r["p_prefix"]),
                "ms_cxr_annotation_id": str(r["ms_cxr_annotation_id"]),
                "bbox_type": "ms_cxr_phrase_grounding_bbox",
                "gold_source": "MS-CXR",
            }
            rows.append(row)
        out_counts[split] = write_jsonl(DATA / f"{split}.jsonl", rows)
        subjects_by_split[split] = {r["subject_id"] for r in rows}
    overlap = {
        "train_val": len(subjects_by_split["train"] & subjects_by_split["val"]),
        "train_eval": len(subjects_by_split["train"] & subjects_by_split["eval"]),
        "val_eval": len(subjects_by_split["val"] & subjects_by_split["eval"]),
    }
    clean = []
    for split in ["train", "val", "eval"]:
        clean.extend(read_jsonl(DATA / f"{split}.jsonl"))
    pd.DataFrame(clean).to_csv(DATA / "ms_cxr_p10_p19_split_all.csv", index=False)
    report = [
        "# Stage 3 Split Report",
        "",
        "- Split source: newly matched p10-p19 MS-CXR annotations.",
        "- Split unit: subject_id.",
        "- Target ratio: approximately 70/10/20.",
        "",
        "| split | rows | unique_subjects |",
        "|---|---:|---:|",
    ]
    for split in ["train", "val", "eval"]:
        report.append(f"| {split} | {out_counts[split]:,} | {len(subjects_by_split[split]):,} |")
    report += ["", "Subject overlap:", "", "```json", json.dumps(overlap, indent=2), "```", "", "## Finding Counts"]
    for split in ["train", "val", "eval"]:
        counts = Counter(r["finding"] for r in read_jsonl(DATA / f"{split}.jsonl"))
        report += ["", f"### {split}", "", "| finding | rows |", "|---|---:|"]
        for f in ALL8:
            report.append(f"| {f} | {counts.get(f, 0):,} |")
    write_text(REPORT / "STAGE3_SPLIT_REPORT.md", "\n".join(report) + "\n")
    return {"train": out_counts["train"], "val": out_counts["val"], "eval": out_counts["eval"], "overlap": overlap}


def parse_claim_context(text: str, finding: str, strip_location: bool = False) -> Dict:
    t = f"{text} {finding}".lower()
    if strip_location:
        t = strip_location_words(t)
    if "bilateral" in t or "both" in t or "bibasilar" in t:
        laterality = "bilateral"
    elif re.search(r"\bright\b|\brt\b", t):
        laterality = "right"
    elif re.search(r"\bleft\b|\blt\b", t):
        laterality = "left"
    else:
        laterality = "unknown"
    if "apical" in t or "apex" in t:
        vertical = "apical"
    elif "basal" in t or "base" in t or "bibasilar" in t:
        vertical = "basal"
    elif "upper" in t or "suprahilar" in t:
        vertical = "upper"
    elif "lower" in t or "inferior" in t or "infrahilar" in t or "costophrenic" in t:
        vertical = "lower"
    elif "middle" in t or re.search(r"\bmid\b", t) or "perihilar" in t:
        vertical = "mid"
    elif "diffuse" in t or "throughout" in t:
        vertical = "whole"
    else:
        vertical = "unknown"
    if any(w in t for w in ["trace", "tiny", "minimal"]):
        severity = "trace"
    elif "small" in t:
        severity = "small"
    elif "mild" in t:
        severity = "mild"
    elif "moderate" in t:
        severity = "moderate"
    elif "large" in t or "severe" in t:
        severity = "large"
    else:
        severity = "unknown"
    uncertainty = "possible" if any(w in t for w in ["possible", "possibly", "may ", "could"]) else "definite"
    return {
        "finding_normalized": finding,
        "laterality": laterality,
        "vertical_region": vertical,
        "severity": severity,
        "uncertainty": uncertainty,
    }


def strip_location_words(text: str) -> str:
    words = [
        "left",
        "right",
        "bilateral",
        "both",
        "upper",
        "lower",
        "middle",
        "mid",
        "apical",
        "apex",
        "basal",
        "base",
        "bases",
        "bibasilar",
        "infrahilar",
        "suprahilar",
        "perihilar",
        "costophrenic",
    ]
    out = text
    for w in words:
        out = re.sub(rf"\b{re.escape(w)}\b", "", out, flags=re.I)
    return " ".join(out.split())


def parse_claims() -> Dict[str, int]:
    counts = {}
    for split in ["train", "val", "eval"]:
        rows = []
        no_loc = []
        for r in read_jsonl(DATA / f"{split}.jsonl"):
            ctx = parse_claim_context(r["claim_sentence"], r["finding"], strip_location=False)
            no = parse_claim_context(r["claim_sentence"], r["finding"], strip_location=True)
            rows.append({"task_id": r["task_id"], **ctx, "parser": "rule"})
            no_loc.append({"task_id": r["task_id"], "claim_no_location": strip_location_words(r["claim_sentence"]), **no, "parser": "rule_no_location"})
        counts[split] = write_jsonl(DATA / f"claim_context_{split}.jsonl", rows)
        write_jsonl(DATA / f"claim_context_{split}_no_location.jsonl", no_loc)
    eval_no_loc = []
    for r in read_jsonl(DATA / "eval.jsonl"):
        rr = dict(r)
        rr["claim_sentence_original"] = rr["claim_sentence"]
        rr["claim_sentence"] = strip_location_words(rr["claim_sentence"])
        eval_no_loc.append(rr)
    write_jsonl(DATA / "eval_no_location_claim.jsonl", eval_no_loc)
    write_text(
        REPORT / "STAGE5_TEXT_PRIOR_BASELINE_REPORT.md",
        "# Stage 5 Text/Context Prior Baseline Report\n\n"
        "Claim context was parsed with deterministic rules. No image or eval bbox was provided to the parser.\n",
    )
    return counts


def vfm_candidates(requested: str) -> List[str]:
    if requested and requested.lower() != "auto":
        return [requested]
    return [
        "microsoft/rad-dino",
        "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "facebook/dinov2-small",
        "facebook/dinov2-base",
    ]


def load_vfm(requested: str, device: str):
    from transformers import AutoImageProcessor, AutoModel

    errors = []
    for name in vfm_candidates(requested):
        try:
            processor = AutoImageProcessor.from_pretrained(name, trust_remote_code=True)
            model = AutoModel.from_pretrained(name, trust_remote_code=True)
            model.eval().to(device)
            for p in model.parameters():
                p.requires_grad_(False)
            return name, processor, model, errors
        except Exception as exc:
            errors.append(f"{name}: {exc!r}")
    raise RuntimeError("No VFM could be loaded: " + " | ".join(errors))


@torch.no_grad()
def vfm_forward(images: List[Image.Image], processor, model, device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs)
    hidden = getattr(out, "last_hidden_state", None)
    if hidden is None:
        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            raise RuntimeError("VFM output has neither last_hidden_state nor pooler_output.")
        hidden = pooled.unsqueeze(1)
    hidden = hidden.detach().float()
    if hidden.ndim != 3:
        hidden = hidden.reshape(hidden.shape[0], 1, -1)
    global_feat = hidden[:, 0, :]
    patches = hidden[:, 1:, :] if hidden.shape[1] > 1 else hidden
    patch_mean = patches.mean(dim=1)
    patch_std = patches.std(dim=1) if patches.shape[1] > 1 else torch.zeros_like(patch_mean)
    return (
        global_feat.cpu().numpy().astype("float32"),
        patch_mean.cpu().numpy().astype("float32"),
        patch_std.cpu().numpy().astype("float32"),
        patches.cpu().numpy().astype("float16"),
    )


def extract_features(args) -> Dict:
    ensure_dirs()
    done = all((FEAT / f"vfm_features_{split}.npz").exists() for split in ["train", "val", "eval"])
    if done and not args.force and not args.extract_features:
        model_name = (FEAT / "VFM_MODEL.txt").read_text(encoding="utf-8").strip()
        return {"vfm_model": model_name, "radiology_attempted": True, "radiology_success": "rad-dino" in model_name.lower()}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name, processor, model, errors = load_vfm(args.vfm_model, device)
    (FEAT / "VFM_MODEL.txt").write_text(model_name, encoding="utf-8")
    index_rows = []
    for split in ["train", "val", "eval"]:
        rows = read_jsonl(DATA / f"{split}.jsonl")
        ids, gf_all, pm_all, ps_all, pt_all = [], [], [], [], []
        batch_imgs, batch_rows = [], []
        for r in rows:
            try:
                img = Image.open(r["image_path"]).convert("RGB")
            except Exception:
                continue
            batch_imgs.append(img)
            batch_rows.append(r)
            if len(batch_imgs) >= args.batch_size:
                gf, pm, ps, pt = vfm_forward(batch_imgs, processor, model, device)
                for i, br in enumerate(batch_rows):
                    ids.append(br["task_id"])
                    gf_all.append(gf[i])
                    pm_all.append(pm[i])
                    ps_all.append(ps[i])
                    pt_all.append(pt[i])
                    index_rows.append({"task_id": br["task_id"], "split": split, "model_name": model_name, "feature_type": "global_patch_tokens"})
                batch_imgs, batch_rows = [], []
        if batch_imgs:
            gf, pm, ps, pt = vfm_forward(batch_imgs, processor, model, device)
            for i, br in enumerate(batch_rows):
                ids.append(br["task_id"])
                gf_all.append(gf[i])
                pm_all.append(pm[i])
                ps_all.append(ps[i])
                pt_all.append(pt[i])
                index_rows.append({"task_id": br["task_id"], "split": split, "model_name": model_name, "feature_type": "global_patch_tokens"})
        np.savez_compressed(
            FEAT / f"vfm_features_{split}.npz",
            task_ids=np.array(ids, dtype=object),
            global_features=np.stack(gf_all).astype("float32"),
            patch_mean=np.stack(pm_all).astype("float32"),
            patch_std=np.stack(ps_all).astype("float32"),
            patch_tokens=np.stack(pt_all).astype("float16"),
        )
    pd.DataFrame(index_rows).to_csv(FEAT / "vfm_feature_index.csv", index=False)
    report = [
        "# Stage 7 VFM Feature Report",
        "",
        f"- VFM model used: `{model_name}`",
        f"- device: `{device}`",
        "- encoder status: frozen feature extraction",
        "- features saved: global feature, patch mean, patch std, patch tokens",
        "",
        "## VFM Selection Attempts",
        "",
        "```text",
        "\n".join(errors) if errors else "first candidate loaded",
        "```",
    ]
    write_text(REPORT / "STAGE6_VFM_SELECTION_REPORT.md", "\n".join(report) + "\n")
    write_text(REPORT / "STAGE7_VFM_FEATURE_REPORT.md", "\n".join(report) + "\n")
    return {"vfm_model": model_name, "radiology_attempted": True, "radiology_success": "rad-dino" in model_name.lower()}


def load_features(split: str) -> Dict[str, np.ndarray]:
    d = np.load(FEAT / f"vfm_features_{split}.npz", allow_pickle=True)
    return {
        "ids": np.array([str(x) for x in d["task_ids"]], dtype=object),
        "global": d["global_features"].astype("float32"),
        "patch_mean": d["patch_mean"].astype("float32"),
        "patch_std": d["patch_std"].astype("float32"),
        "patch_tokens": d["patch_tokens"].astype("float16"),
    }


def context_maps() -> Dict[str, List[str]]:
    rows = []
    for split in ["train", "val", "eval"]:
        rows += read_jsonl(DATA / f"claim_context_{split}.jsonl")
    fields = ["finding_normalized", "laterality", "vertical_region", "severity", "uncertainty"]
    return {f: sorted(set(str(r.get(f, "unknown")) for r in rows)) for f in fields}


def encode_context(rows: List[Dict], contexts: List[Dict], maps: Dict[str, List[str]], mode: str) -> np.ndarray:
    by_id = {r["task_id"]: r for r in contexts}
    feats = []
    for row in rows:
        ctx = by_id.get(row["task_id"], {})
        vals = []
        if mode in {"finding", "context"}:
            for v in maps["finding_normalized"]:
                vals.append(1.0 if str(ctx.get("finding_normalized", row["finding"])) == v else 0.0)
        if mode == "context":
            for f in ["laterality", "vertical_region", "severity", "uncertainty"]:
                for v in maps[f]:
                    vals.append(1.0 if str(ctx.get(f, "unknown")) == v else 0.0)
        feats.append(vals)
    return np.asarray(feats, dtype="float32") if feats and feats[0] else np.zeros((len(rows), 0), dtype="float32")


def aligned_data(split: str, feature_mode: str, context_mode: str, maps: Dict[str, List[str]], no_location: bool = False):
    rows = read_jsonl(DATA / f"{split}.jsonl")
    contexts = read_jsonl(DATA / f"claim_context_{split}{'_no_location' if no_location else ''}.jsonl")
    feat = load_features(split)
    pos = {tid: i for i, tid in enumerate(feat["ids"])}
    ordered, idxs, y = [], [], []
    for r in rows:
        if r["task_id"] in pos:
            ordered.append(r)
            idxs.append(pos[r["task_id"]])
            y.append(r["gold_bbox_norm_cxcywh"])
    parts = []
    if feature_mode == "global":
        parts.append(feat["global"][idxs])
    elif feature_mode == "patch":
        parts += [feat["patch_mean"][idxs], feat["patch_std"][idxs]]
    elif feature_mode == "global_patch":
        parts += [feat["global"][idxs], feat["patch_mean"][idxs], feat["patch_std"][idxs]]
    elif feature_mode == "none":
        pass
    else:
        raise ValueError(feature_mode)
    if context_mode != "none":
        ctx = encode_context(ordered, contexts, maps, "finding" if context_mode == "finding" else "context")
        if ctx.shape[1]:
            parts.append(ctx)
    x = np.concatenate(parts, axis=1).astype("float32") if parts else np.zeros((len(ordered), 1), dtype="float32")
    return ordered, x, np.asarray(y, dtype="float32")


def aligned_patch_data(split: str, context_mode: str, maps: Dict[str, List[str]], no_location: bool = False):
    rows = read_jsonl(DATA / f"{split}.jsonl")
    contexts = read_jsonl(DATA / f"claim_context_{split}{'_no_location' if no_location else ''}.jsonl")
    feat = load_features(split)
    pos = {tid: i for i, tid in enumerate(feat["ids"])}
    ordered, idxs, y = [], [], []
    for r in rows:
        if r["task_id"] in pos:
            ordered.append(r)
            idxs.append(pos[r["task_id"]])
            y.append(r["gold_bbox_norm_cxcywh"])
    ctx = encode_context(ordered, contexts, maps, "context" if context_mode == "context" else "finding")
    return ordered, feat["patch_tokens"][idxs].astype("float32"), ctx.astype("float32"), np.asarray(y, dtype="float32")


def pixel_pred_rows(rows: List[Dict], pred_norm: np.ndarray, method: str, extra: Optional[Dict] = None) -> List[Dict]:
    out = []
    extra = extra or {}
    for r, p in zip(rows, pred_norm):
        p = np.asarray(p, dtype="float32")
        p[:2] = np.clip(p[:2], 0, 1)
        p[2:] = np.clip(p[2:], 0.02, 1)
        out.append(
            {
                "task_id": r["task_id"],
                "method": method,
                "dicom_id": r["dicom_id"],
                "finding": r["finding"],
                "claim_sentence": r["claim_sentence"],
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
        vals.append(max_iou(norm_cxcywh_to_xyxy(p, r["image_width"], r["image_height"]), r))
    return float(np.mean(vals)) if vals else 0.0


def train_text_prior() -> Dict:
    train = read_jsonl(DATA / "train.jsonl")
    val = read_jsonl(DATA / "val.jsonl")
    eval_rows = read_jsonl(DATA / "eval.jsonl")
    ctx_train = {r["task_id"]: r for r in read_jsonl(DATA / "claim_context_train.jsonl")}
    ctx_eval = {r["task_id"]: r for r in read_jsonl(DATA / "claim_context_eval.jsonl")}
    buckets = defaultdict(list)
    finding = defaultdict(list)
    all_boxes = []
    for r in train:
        ctx = ctx_train.get(r["task_id"], {})
        key = (r["finding"], ctx.get("laterality", "unknown"), ctx.get("vertical_region", "unknown"))
        buckets[key].append(r["gold_bbox_norm_cxcywh"])
        finding[r["finding"]].append(r["gold_bbox_norm_cxcywh"])
        all_boxes.append(r["gold_bbox_norm_cxcywh"])
    bucket_mean = {k: np.asarray(v, dtype="float32").mean(axis=0) for k, v in buckets.items()}
    finding_mean = {k: np.asarray(v, dtype="float32").mean(axis=0) for k, v in finding.items()}
    global_mean = np.asarray(all_boxes, dtype="float32").mean(axis=0)

    def predict(rows, ctx_by_id, method):
        preds = []
        for r in rows:
            ctx = ctx_by_id.get(r["task_id"], {})
            key = (r["finding"], ctx.get("laterality", "unknown"), ctx.get("vertical_region", "unknown"))
            p = bucket_mean.get(key)
            src = "finding_laterality_vertical_mean"
            if p is None:
                p = finding_mean.get(r["finding"])
                src = "finding_mean"
            if p is None:
                p = global_mean
                src = "global_mean"
            preds.extend(pixel_pred_rows([r], np.asarray([p]), method, {"prior_source": src}))
        return preds

    preds = predict(eval_rows, ctx_eval, "text_prior_baseline")
    write_jsonl(PRED / "text_prior_baseline.jsonl", preds)
    no_ctx_eval = {r["task_id"]: r for r in read_jsonl(DATA / "claim_context_eval_no_location.jsonl")}
    no_preds = predict(eval_rows, no_ctx_eval, "text_prior_no_location")
    write_jsonl(PRED / "text_prior_no_location.jsonl", no_preds)
    return {"method": "text_prior_baseline", "bucket_count": len(bucket_mean), "val_rows": len(val), "pred_rows": preds}


def train_mlp(method: str, feature_mode: str, context_mode: str, maps: Dict[str, List[str]], args) -> Dict:
    train_rows, xtr, ytr = aligned_data("train", feature_mode, context_mode, maps)
    val_rows, xva, yva = aligned_data("val", feature_mode, context_mode, maps)
    eval_rows, xev, _ = aligned_data("eval", feature_mode, context_mode, maps)
    scaler = FeatureScaler.fit(xtr)
    xtr = scaler.transform(xtr)
    xva = scaler.transform(xva)
    xev = scaler.transform(xev)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BBoxMLP(xtr.shape[1], hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    xt = torch.tensor(xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.float32, device=device)
    xv = torch.tensor(xva, dtype=torch.float32, device=device)
    best_state, best_val, best_epoch, wait = None, -1.0, 0, 0
    hist = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(xt), device=device)
        losses = []
        for start in range(0, len(xt), args.train_batch_size):
            idx = perm[start : start + args.train_batch_size]
            pred = model(xt[idx])
            loss, info = bbox_loss(pred, yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            pv = model(xv).detach().cpu().numpy()
        score = val_score(pv, val_rows)
        hist.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_mean_iou": score})
        if score > best_val:
            best_val, best_epoch, wait = score, epoch, 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= args.patience:
            break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pev = model(torch.tensor(xev, dtype=torch.float32, device=device)).detach().cpu().numpy()
    preds = pixel_pred_rows(eval_rows, pev, method, {"feature_mode": feature_mode, "context_mode": context_mode})
    write_jsonl(PRED / f"{method}.jsonl", preds)
    run_dir = RUNS / method
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(hist).to_csv(run_dir / "loss_curve.csv", index=False)
    torch.save({"state": best_state, "scaler": scaler.state_dict(), "input_dim": xtr.shape[1], "method": method}, CKPT / f"{method}.pt")
    return {"method": method, "train_rows": len(train_rows), "val_rows": len(val_rows), "eval_rows": len(eval_rows), "input_dim": xtr.shape[1], "best_val_mean_iou": best_val, "best_epoch": best_epoch, "pred_rows": preds}


def train_heatmap(method: str, maps: Dict[str, List[str]], args) -> Dict:
    train_rows, tokens_tr, ctx_tr, ytr = aligned_patch_data("train", "context", maps)
    val_rows, tokens_va, ctx_va, yva = aligned_patch_data("val", "context", maps)
    eval_rows, tokens_ev, ctx_ev, _ = aligned_patch_data("eval", "context", maps)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PatchHeatmapBBoxHead(tokens_tr.shape[-1], ctx_tr.shape[1], hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tok_tr = torch.tensor(tokens_tr, dtype=torch.float32, device=device)
    ctx_tr_t = torch.tensor(ctx_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(ytr, dtype=torch.float32, device=device)
    tok_va = torch.tensor(tokens_va, dtype=torch.float32, device=device)
    ctx_va_t = torch.tensor(ctx_va, dtype=torch.float32, device=device)
    y_va_t = torch.tensor(yva, dtype=torch.float32, device=device)
    grid = int(round(math.sqrt(tokens_tr.shape[1])))

    def center_idx(y: np.ndarray) -> torch.Tensor:
        cx = np.clip((y[:, 0] * grid).astype(int), 0, grid - 1)
        cy = np.clip((y[:, 1] * grid).astype(int), 0, grid - 1)
        return torch.tensor(cy * grid + cx, dtype=torch.long, device=device).clamp(max=tokens_tr.shape[1] - 1)

    c_tr = center_idx(ytr)
    c_va = center_idx(yva)
    best_state, best_val, best_epoch, wait = None, -1.0, 0, 0
    hist = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(tok_tr), device=device)
        losses = []
        for start in range(0, len(tok_tr), max(1, args.heatmap_batch_size)):
            idx = perm[start : start + args.heatmap_batch_size]
            pred, logits = model(tok_tr[idx], ctx_tr_t[idx])
            b_loss, _ = bbox_loss(pred, y_tr_t[idx])
            h_loss = torch.nn.functional.cross_entropy(logits, c_tr[idx])
            loss = b_loss + args.heatmap_loss_weight * h_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            pv, _ = model(tok_va, ctx_va_t)
        score = val_score(pv.detach().cpu().numpy(), val_rows)
        hist.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_mean_iou": score})
        if score > best_val:
            best_val, best_epoch, wait = score, epoch, 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= args.patience:
            break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    preds_norm = []
    with torch.no_grad():
        for start in range(0, len(tokens_ev), max(1, args.heatmap_batch_size)):
            tok = torch.tensor(tokens_ev[start : start + args.heatmap_batch_size], dtype=torch.float32, device=device)
            ctx = torch.tensor(ctx_ev[start : start + args.heatmap_batch_size], dtype=torch.float32, device=device)
            p, _ = model(tok, ctx)
            preds_norm.append(p.detach().cpu().numpy())
    pred_norm = np.vstack(preds_norm)
    preds = pixel_pred_rows(eval_rows, pred_norm, method, {"feature_mode": "patch_tokens", "context_mode": "context", "head": "patch_heatmap"})
    write_jsonl(PRED / f"{method}.jsonl", preds)
    run_dir = RUNS / method
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(hist).to_csv(run_dir / "loss_curve.csv", index=False)
    torch.save({"state": best_state, "token_dim": tokens_tr.shape[-1], "context_dim": ctx_tr.shape[1], "method": method}, CKPT / f"{method}.pt")
    return {"method": method, "train_rows": len(train_rows), "val_rows": len(val_rows), "eval_rows": len(eval_rows), "input_dim": f"{tokens_tr.shape[1]}x{tokens_tr.shape[-1]}+{ctx_tr.shape[1]}", "best_val_mean_iou": best_val, "best_epoch": best_epoch, "pred_rows": preds}


def train_extra_trees(method: str, feature_mode: str, context_mode: str, maps: Dict[str, List[str]], args) -> Dict:
    train_rows, xtr, ytr = aligned_data("train", feature_mode, context_mode, maps)
    val_rows, xva, _ = aligned_data("val", feature_mode, context_mode, maps)
    eval_rows, xev, _ = aligned_data("eval", feature_mode, context_mode, maps)
    model = ExtraTreesRegressor(n_estimators=args.extra_trees, min_samples_leaf=2, random_state=SEED, n_jobs=-1)
    t0 = time.time()
    model.fit(xtr, ytr)
    pv = np.clip(model.predict(xva).astype("float32"), 0, 1)
    pe = np.clip(model.predict(xev).astype("float32"), 0, 1)
    pe[:, 2:] = np.clip(pe[:, 2:], 0.02, 1)
    preds = pixel_pred_rows(eval_rows, pe, method, {"feature_mode": feature_mode, "context_mode": context_mode, "regressor": "ExtraTreesRegressor"})
    write_jsonl(PRED / f"{method}.jsonl", preds)
    return {"method": method, "train_rows": len(train_rows), "val_rows": len(val_rows), "eval_rows": len(eval_rows), "input_dim": xtr.shape[1], "best_val_mean_iou": val_score(pv, val_rows), "best_epoch": "n/a", "fit_sec": time.time() - t0, "pred_rows": preds}


def train_models(args) -> Dict:
    maps = context_maps()
    infos = [train_text_prior()]
    model_specs = [
        ("context_only_mlp_prior", "none", "context"),
        ("vfm_global_bbox_head", "global", "finding"),
        ("vfm_patch_pool_bbox_head", "patch", "context"),
        ("vfm_claim_context_bbox_head", "global_patch", "context"),
    ]
    for method, fm, cm in model_specs:
        infos.append(train_mlp(method, fm, cm, maps, args))
    infos.append(train_heatmap("vfm_heatmap_bbox_head", maps, args))
    infos.append(train_extra_trees("extra_trees_context_vfm_regressor", "global_patch", "context", maps, args))
    pd.DataFrame([{k: v for k, v in info.items() if k != "pred_rows"} for info in infos]).to_csv(MET / "training_runs.csv", index=False)
    write_text(
        REPORT / "STAGE8_MODEL_ARCHITECTURE_REPORT.md",
        "# Stage 8 Model Architecture Report\n\n"
        "- `text_prior_baseline`: no image, train-set grouped mean bbox by finding/laterality/vertical region.\n"
        "- `context_only_mlp_prior`: parsed claim/finding context only -> bbox MLP.\n"
        "- `vfm_global_bbox_head`: frozen VFM global feature + finding one-hot -> bbox MLP.\n"
        "- `vfm_patch_pool_bbox_head`: frozen VFM patch mean/std + parsed claim context -> bbox MLP.\n"
        "- `vfm_heatmap_bbox_head`: frozen VFM patch tokens + parsed claim context -> patch heatmap + bbox head.\n"
        "- `vfm_claim_context_bbox_head`: frozen VFM global + patch summaries + parsed claim context -> bbox MLP.\n"
        "- `extra_trees_context_vfm_regressor`: small-data non-neural regressor for stability audit.\n\n"
        "This is direct bbox supervision, not candidate A/B/C selection.\n",
    )
    report = [
        "# Stage 9 Training Report",
        "",
        "- Train split only used for optimization.",
        "- Val split used for checkpoint/model selection.",
        "- Eval split used only after training.",
        "- Loss for neural heads: SmoothL1 normalized cxcywh + 1 - GIoU; heatmap head adds center-token CE loss.",
        "- VFM encoder is frozen in this Stage 1 quick run; last-block unfreeze is recorded as not attempted if memory/time constrained.",
        "",
        pd.DataFrame([{k: v for k, v in info.items() if k != "pred_rows"} for info in infos]).to_markdown(index=False),
    ]
    write_text(REPORT / "STAGE9_TRAINING_REPORT.md", "\n".join(report) + "\n")
    return {"infos": infos}


def evaluate_prediction_rows(preds: List[Dict], tasks: List[Dict], method: str, subset: str) -> Dict:
    by_id = {p["task_id"]: p for p in preds}
    vals, missing = [], 0
    invalid = 0
    for t in tasks:
        p = by_id.get(t["task_id"])
        box = p.get("pred_bbox_xyxy") if p else None
        if not box:
            missing += 1
            vals.append(0.0)
        else:
            if not valid_box(box):
                invalid += 1
            vals.append(max_iou(box, t))
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
    eval_rows = read_jsonl(DATA / "eval.jsonl")
    main_rows = [r for r in eval_rows if r["finding"] in MAIN5]
    pred_files = sorted(PRED.glob("*.jsonl"))
    all_summary, main_summary, per_rows = [], [], []
    for pf in pred_files:
        method = pf.stem
        if method.endswith("_no_location"):
            continue
        preds = read_jsonl(pf)
        all_summary.append(evaluate_prediction_rows(preds, eval_rows, method, "all8"))
        main_summary.append(evaluate_prediction_rows(preds, main_rows, method, "main5"))
        for finding in ALL8:
            subset = [r for r in eval_rows if r["finding"] == finding]
            if subset:
                row = evaluate_prediction_rows(preds, subset, method, finding)
                row["finding"] = finding
                per_rows.append(row)
    pd.DataFrame(all_summary).sort_values("mean_iou", ascending=False).to_csv(MET / "summary_all8.csv", index=False)
    pd.DataFrame(main_summary).sort_values("mean_iou", ascending=False).to_csv(MET / "summary_main5.csv", index=False)
    pd.DataFrame(per_rows).to_csv(MET / "per_finding_metrics.csv", index=False)
    write_text(
        REPORT / "STAGE10_EVAL_REPORT.md",
        "# Stage 10 Eval Report\n\n"
        "## All8\n\n"
        + pd.DataFrame(all_summary).sort_values("mean_iou", ascending=False).to_markdown(index=False)
        + "\n\n## Main5\n\n"
        + pd.DataFrame(main_summary).sort_values("mean_iou", ascending=False).to_markdown(index=False)
        + "\n",
    )
    return {"all8": all_summary, "main5": main_summary}


def run_location_hint_ablation() -> None:
    eval_rows = read_jsonl(DATA / "eval.jsonl")
    rows = []
    for method in ["text_prior_baseline", "text_prior_no_location", "vfm_claim_context_bbox_head"]:
        path = PRED / f"{method}.jsonl"
        if path.exists():
            rows.append(evaluate_prediction_rows(read_jsonl(path), eval_rows, method, "all8"))
    # Re-evaluate context VFM with no-location context by retraining is intentionally avoided;
    # the primary check is how much text prior drops when location words are removed.
    pd.DataFrame(rows).to_csv(MET / "location_hint_ablation.csv", index=False)
    write_text(
        REPORT / "STAGE11_LOCATION_HINT_ABLATION_REPORT.md",
        "# Stage 11 Location Hint Ablation Report\n\n"
        "- `eval_no_location_claim.jsonl` was written with left/right/upper/lower/apical/basal-like words removed.\n"
        "- This quick ablation measures the no-image text prior drop when location terms are removed.\n"
        "- VFM visual features are unchanged; no eval gold is used in input.\n\n"
        + pd.DataFrame(rows).to_markdown(index=False)
        + "\n",
    )


def draw_boxes(image_path: str, boxes: List[Tuple[str, Optional[Sequence[float]], Tuple[int, int, int]]], out: Path, title: str) -> bool:
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return False
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
        small = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
        small = ImageFont.load_default()
    draw.rectangle([0, 0, img.width, 64], fill=(0, 0, 0))
    draw.text((8, 8), title[:150], fill=(255, 255, 255), font=small)
    for label, box, color in boxes:
        if not box or not valid_box(box):
            continue
        x1, y1, x2, y2 = [float(x) for x in box]
        for off in range(4):
            draw.rectangle([x1 - off, y1 - off, x2 + off, y2 + off], outline=color)
        draw.text((x1 + 4, max(66, y1 + 4)), label, fill=color, font=font)
    img.thumbnail((1000, 1000))
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


def make_gold_overlays() -> int:
    paths, counts = [], Counter()
    for split in ["train", "val", "eval"]:
        for r in read_jsonl(DATA / f"{split}.jsonl"):
            if counts[r["finding"]] >= 8:
                continue
            counts[r["finding"]] += 1
            out = OVER / "gold_ms_cxr" / f"{r['finding'].replace(' ','_')}_{counts[r['finding']]:02d}_{r['task_id']}.jpg"
            if draw_boxes(
                r["image_path"],
                [("MS-CXR phrase bbox", r["gold_bbox_xyxy"], (255, 230, 0))],
                out,
                f"dataset=MS-CXR | finding={r['finding']} | phrase bbox, not pixel mask",
            ):
                paths.append(out)
    make_contact_sheet(paths, CONTACT / "gold_ms_cxr_contact_sheet.jpg")
    write_text(
        REPORT / "STAGE4_OVERLAY_SANITY_REPORT.md",
        "# Stage 4 Overlay Sanity Report\n\n"
        f"- gold overlays written: {len(paths)}\n"
        f"- contact sheet: `{CONTACT / 'gold_ms_cxr_contact_sheet.jpg'}`\n"
        "- Gold boxes are MS-CXR phrase-grounding bboxes, not masks.\n",
    )
    return len(paths)


def make_model_overlays() -> int:
    eval_rows = read_jsonl(DATA / "eval.jsonl")
    summary = pd.read_csv(MET / "summary_all8.csv")
    best = str(summary.sort_values("mean_iou", ascending=False).iloc[0]["method"])
    pred_best = {r["task_id"]: r for r in read_jsonl(PRED / f"{best}.jsonl")}
    pred_global = {r["task_id"]: r for r in read_jsonl(PRED / "vfm_global_bbox_head.jsonl")} if (PRED / "vfm_global_bbox_head.jsonl").exists() else {}
    pred_heat = {r["task_id"]: r for r in read_jsonl(PRED / "vfm_heatmap_bbox_head.jsonl")} if (PRED / "vfm_heatmap_bbox_head.jsonl").exists() else {}
    scored = [(max_iou(pred_best.get(r["task_id"], {}).get("pred_bbox_xyxy"), r), r) for r in eval_rows]
    examples = sorted(scored, key=lambda x: x[0], reverse=True)[:16] + sorted(scored, key=lambda x: x[0])[:16]
    paths = []
    for i, (score, r) in enumerate(examples):
        out = OVER / "comparison" / f"{i:03d}_{r['finding'].replace(' ','_')}_{r['task_id']}.jpg"
        ok = draw_boxes(
            r["image_path"],
            [
                ("gold", r["gold_bbox_xyxy"], (255, 230, 0)),
                ("vfm_global", pred_global.get(r["task_id"], {}).get("pred_bbox_xyxy"), (0, 120, 255)),
                ("vfm_heatmap", pred_heat.get(r["task_id"], {}).get("pred_bbox_xyxy"), (0, 220, 80)),
                ("best", pred_best.get(r["task_id"], {}).get("pred_bbox_xyxy"), (0, 255, 255)),
            ],
            out,
            f"{r['finding']} | best={best} IoU={score:.3f} | gold=MS-CXR phrase bbox",
        )
        if ok:
            paths.append(out)
    make_contact_sheet(paths[:32], CONTACT / "vfm_localizer_success_failure_cases.jpg")
    write_text(
        REPORT / "STAGE13_OVERLAY_REVIEW_REPORT.md",
        "# Stage 13 Overlay Review Report\n\n"
        f"- comparison overlays written: {len(paths)}\n"
        "- Colors: gold=yellow, vfm_global=blue, vfm_heatmap=green, best=cyan.\n"
        "- Gold boxes are drawn only for analysis overlays.\n",
    )
    return len(paths)


def write_weak_pretrain_report() -> None:
    write_text(
        REPORT / "STAGE12_CHEST_IMAGENOME_WEAK_PRETRAIN_REPORT.md",
        "# Stage 12 Chest ImaGenome Weak Pretrain Report\n\n"
        "Not run in this Stage 1 quick pass. Chest ImaGenome remains weak region/device evidence, not gold lesion bbox. "
        "A later run can pretrain on weak Chest ImaGenome boxes and fine-tune on MS-CXR phrase-grounding boxes.\n",
    )


def write_logic_and_paper(vfm_info: Dict, overlay_count: int) -> Dict:
    train, val, ev = [read_jsonl(DATA / f"{s}.jsonl") for s in ["train", "val", "eval"]]
    overlap = {
        "train_val": len({r["subject_id"] for r in train} & {r["subject_id"] for r in val}),
        "train_eval": len({r["subject_id"] for r in train} & {r["subject_id"] for r in ev}),
        "val_eval": len({r["subject_id"] for r in val} & {r["subject_id"] for r in ev}),
    }
    all8 = pd.read_csv(MET / "summary_all8.csv")
    main5 = pd.read_csv(MET / "summary_main5.csv")
    best_all8 = all8.sort_values("mean_iou", ascending=False).iloc[0]
    best_main5 = main5.sort_values("mean_iou", ascending=False).iloc[0]
    logic = [
        "# Pipeline Logical Review",
        "",
        "- p10-p19 image index was rebuilt by scanning local JPG files.",
        "- MS-CXR matching was rebuilt against the p10-p19 image index.",
        "- MS-CXR bbox is phrase-grounding bbox, not pixel-level lesion mask.",
        "- Chest ImaGenome was not used as gold bbox.",
        "- A/B/C selector, pairwise comparator, and Qwen selector LoRA are not part of this main method.",
        "- Eval gold bbox was used only for metrics and analysis overlays.",
        "- Val split selected checkpoints; eval split was final measurement.",
        "- Existing p10-p14 results are not directly ranked against the new p10-p19 split.",
        "",
        "Subject overlap:",
        "",
        "```json",
        json.dumps(overlap, indent=2),
        "```",
    ]
    write_text(REPORT / "PIPELINE_LOGICAL_REVIEW.md", "\n".join(logic) + "\n")
    interpretation = []
    text_iou = float(all8[all8["method"].eq("text_prior_baseline")]["mean_iou"].iloc[0]) if "text_prior_baseline" in set(all8["method"]) else float("nan")
    global_iou = float(all8[all8["method"].eq("vfm_global_bbox_head")]["mean_iou"].iloc[0]) if "vfm_global_bbox_head" in set(all8["method"]) else float("nan")
    heat_iou = float(all8[all8["method"].eq("vfm_heatmap_bbox_head")]["mean_iou"].iloc[0]) if "vfm_heatmap_bbox_head" in set(all8["method"]) else float("nan")
    context_iou = float(all8[all8["method"].eq("vfm_claim_context_bbox_head")]["mean_iou"].iloc[0]) if "vfm_claim_context_bbox_head" in set(all8["method"]) else float("nan")
    if not math.isnan(global_iou) and not math.isnan(text_iou):
        interpretation.append("VFM global head beat text prior." if global_iou > text_iou else "Text prior remained stronger than VFM global head.")
    if not math.isnan(heat_iou) and not math.isnan(global_iou):
        interpretation.append("Patch heatmap head beat global head." if heat_iou > global_iou else "Patch heatmap head did not beat global head.")
    if not math.isnan(context_iou) and not math.isnan(global_iou):
        interpretation.append("Claim context improved over global VFM." if context_iou > global_iou else "Claim context did not clearly improve over global VFM.")
    paper = [
        "# MS-CXR VFM Localizer Stage 1 P10-P19 Summary",
        "",
        "- Dataset: local p10-p19 MS-CXR matched subset.",
        "- Target: MS-CXR phrase-grounding bbox, not pixel-level lesion mask.",
        "- VFM localizer was trained with direct bbox supervision.",
        "- Selector/A/B/C and Qwen candidate-selection experiments were excluded from the main method.",
        "- Chest ImaGenome is weak evidence only and was not used as gold bbox.",
        f"- VFM used: `{vfm_info.get('vfm_model')}`.",
        "",
        "## All8 Results",
        "",
        all8.to_markdown(index=False),
        "",
        "## Main5 Results",
        "",
        main5.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "\n".join(f"- {x}" for x in interpretation) if interpretation else "- No interpretation available.",
        "",
        f"- overlays written: {overlay_count}",
    ]
    write_text(REPORT / "EXPERIMENT_SUMMARY_FOR_PAPER.md", "\n".join(paper) + "\n")
    write_text(
        REPORT / "FAILURE_ANALYSIS.md",
        "# Failure Analysis\n\n"
        + ("\n".join(f"- {x}" for x in interpretation) if interpretation else "- Pending.")
        + "\n\nLikely failure modes: small MS-CXR train size, broad phrase boxes, diffuse findings, and frozen VFM features not aligned to CXR lesion grounding.\n",
    )
    write_text(
        REPORT / "NEXT_PLAN_AFTER_STAGE1_VFM_LOCALIZER.md",
        "# Next Plan After Stage 1 VFM Localizer\n\n"
        "1. If text prior is strong, report location-word bias clearly and emphasize no-location ablation.\n"
        "2. Try a radiology-specific VFM if DINOv2 fallback was used.\n"
        "3. Add weak Chest ImaGenome pretrain only as weak supervision, then MS-CXR fine-tune.\n"
        "4. Consider unfreezing the final VFM block or using a lightweight adapter if frozen heads underperform.\n"
        "5. Keep Qwen as claim parser/explanation module later, not the main bbox generator.\n",
    )
    return {"best_all8": best_all8.to_dict(), "best_main5": best_main5.to_dict()}


def run_pipeline(args) -> Dict:
    ensure_dirs()
    info = {}
    if args.refresh_image_index:
        info.update(refresh_image_index(args))
    elif not (DATA / "jpg_file_index_p10_p19.csv.gz").exists():
        info.update(refresh_image_index(args))
    else:
        df = pd.read_csv(DATA / "jpg_file_index_p10_p19.csv.gz")
        info.update({"jpg_rows": len(df), "p_counts": df["p_prefix"].value_counts().sort_index().to_dict()})
    if args.rebuild_ms_cxr_match or not (DATA / "ms_cxr_p10_p19_annotations.csv.gz").exists():
        info.update(rebuild_ms_cxr_match(args))
    else:
        df = pd.read_csv(DATA / "ms_cxr_p10_p19_annotations.csv.gz")
        info.update({"matched": len(df), "ms_total": 1448})
    if args.build_split or not (DATA / "train.jsonl").exists():
        info.update(build_split(args))
    else:
        info.update({s: len(read_jsonl(DATA / f"{s}.jsonl")) for s in ["train", "val", "eval"]})
    if args.make_overlays:
        make_gold_overlays()
    if args.train_baselines or not (DATA / "claim_context_train.jsonl").exists():
        parse_claims()
    if args.extract_features or not (FEAT / "vfm_features_train.npz").exists():
        vfm_info = extract_features(args)
    else:
        model = (FEAT / "VFM_MODEL.txt").read_text(encoding="utf-8").strip() if (FEAT / "VFM_MODEL.txt").exists() else "cached"
        vfm_info = {"vfm_model": model, "radiology_attempted": True, "radiology_success": "rad-dino" in model.lower()}
    if args.train_vfm_localizer or args.train_baselines or not (PRED / "vfm_global_bbox_head.jsonl").exists():
        train_models(args)
    if args.eval:
        evaluate_all()
        run_location_hint_ablation()
    if args.weak_pretrain_smoke:
        write_weak_pretrain_report()
    else:
        write_weak_pretrain_report()
    overlay_count = 0
    if args.make_overlays:
        overlay_count += make_model_overlays()
    final = write_logic_and_paper(vfm_info, overlay_count)
    info.update(vfm_info)
    info.update(final)
    return info


def metric(method: str, path: Path = MET / "summary_all8.csv") -> float:
    if not path.exists():
        return float("nan")
    df = pd.read_csv(path)
    row = df[df["method"].eq(method)]
    return float(row.iloc[0]["mean_iou"]) if len(row) else float("nan")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--refresh-image-index", action="store_true")
    p.add_argument("--rebuild-ms-cxr-match", action="store_true")
    p.add_argument("--build-split", action="store_true")
    p.add_argument("--extract-features", action="store_true")
    p.add_argument("--train-baselines", action="store_true")
    p.add_argument("--train-vfm-localizer", action="store_true")
    p.add_argument("--train-heatmap-head", action="store_true")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--make-overlays", action="store_true")
    p.add_argument("--main5", action="store_true")
    p.add_argument("--all8", action="store_true")
    p.add_argument("--weak-pretrain-smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--vfm-model", default="auto")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--train-batch-size", type=int, default=64)
    p.add_argument("--heatmap-batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--extra-trees", type=int, default=500)
    p.add_argument("--heatmap-loss-weight", type=float, default=0.2)
    args = p.parse_args()
    if not any(
        [
            args.refresh_image_index,
            args.rebuild_ms_cxr_match,
            args.build_split,
            args.extract_features,
            args.train_baselines,
            args.train_vfm_localizer,
            args.train_heatmap_head,
            args.eval,
            args.make_overlays,
            args.weak_pretrain_smoke,
        ]
    ):
        args.refresh_image_index = True
        args.rebuild_ms_cxr_match = True
        args.build_split = True
        args.extract_features = True
        args.train_baselines = True
        args.train_vfm_localizer = True
        args.train_heatmap_head = True
        args.eval = True
        args.make_overlays = True
    if args.quick:
        args.epochs = min(args.epochs, 60)
        args.patience = min(args.patience, 10)
    return args


def main() -> None:
    args = parse_args()
    t0 = time.time()
    info = run_pipeline(args)
    all8 = MET / "summary_all8.csv"
    main5 = MET / "summary_main5.csv"
    print(f"project_root={PROJECT_ROOT}")
    print(f"p10_p19_jpg_files_indexed={info.get('jpg_rows', 0)}")
    print(f"ms_cxr_total_annotations={info.get('ms_total', 0)}")
    print(f"ms_cxr_matched_annotations_p10_p19={info.get('matched', 0)}")
    print(f"train_rows={info.get('train', 0)}")
    print(f"val_rows={info.get('val', 0)}")
    print(f"eval_rows={info.get('eval', 0)}")
    print(f"vfm_model_used={info.get('vfm_model', '')}")
    print(f"radiology_vfm_attempted={info.get('radiology_attempted', True)}")
    print(f"radiology_vfm_success={info.get('radiology_success', False)}")
    print(f"text_prior_mean_iou_all8={metric('text_prior_baseline'):.6f}")
    print(f"context_only_mean_iou_all8={metric('context_only_mlp_prior'):.6f}")
    print(f"vfm_global_mean_iou_all8={metric('vfm_global_bbox_head'):.6f}")
    print(f"vfm_patch_mean_iou_all8={metric('vfm_patch_pool_bbox_head'):.6f}")
    print(f"vfm_heatmap_mean_iou_all8={metric('vfm_heatmap_bbox_head'):.6f}")
    print(f"vfm_context_mean_iou_all8={metric('vfm_claim_context_bbox_head'):.6f}")
    print(f"vfm_global_mean_iou_main5={metric('vfm_global_bbox_head', main5):.6f}")
    print(f"vfm_patch_mean_iou_main5={metric('vfm_patch_pool_bbox_head', main5):.6f}")
    print(f"vfm_heatmap_mean_iou_main5={metric('vfm_heatmap_bbox_head', main5):.6f}")
    print(f"vfm_context_mean_iou_main5={metric('vfm_claim_context_bbox_head', main5):.6f}")
    ba = info.get("best_all8", {})
    bm = info.get("best_main5", {})
    print(f"best_method_all8={ba.get('method', '')}")
    print(f"best_method_main5={bm.get('method', '')}")
    print(f"location_hint_ablation_report_path={REPORT / 'STAGE11_LOCATION_HINT_ABLATION_REPORT.md'}")
    print(f"paper_summary_path={REPORT / 'EXPERIMENT_SUMMARY_FOR_PAPER.md'}")
    print(f"failure_analysis_path={REPORT / 'FAILURE_ANALYSIS.md'}")
    print(f"logical_review_path={REPORT / 'PIPELINE_LOGICAL_REVIEW.md'}")
    print(f"elapsed_sec={time.time() - t0:.1f}")


if __name__ == "__main__":
    main()

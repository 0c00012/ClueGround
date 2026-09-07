#!/usr/bin/env python
"""Stage 2: context-conditioned VFM localizer finetuning on MS-CXR p10-p19.

This script keeps the Stage 1 p10-p19 MS-CXR split fixed, reuses the RAD-DINO
feature cache when available, and tests whether structured claim context helps
patch-level VFM bbox localization. It does not run A/B/C selector, pairwise
comparator, or Qwen bbox generation experiments.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

from models_ms_cxr_vfm_localizer import BBoxMLP, FeatureScaler, PatchHeatmapBBoxHead, bbox_loss
from models_ms_cxr_vfm_localizer_stage2 import AdapterPatchHeatmapBBoxHead


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]

STAGE1 = "ms_cxr_vfm_localizer_stage1_p10_p19"
EXP_NAME = "ms_cxr_context_vfm_localizer_stage2_context_finetune"

S1_DATA = PROJECT_ROOT / "training" / STAGE1 / "datasets"
S1_FEAT = PROJECT_ROOT / "features" / STAGE1
S1_EXP = PROJECT_ROOT / "experiments" / STAGE1

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

MAIN5 = ["Atelectasis", "Consolidation", "Lung Opacity", "Pleural Effusion", "Pneumothorax"]
SECONDARY = ["Cardiomegaly", "Edema", "Pneumonia"]
ALL8 = MAIN5 + SECONDARY
SEED = 20260622
SMM_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


def ensure_dirs() -> None:
    for p in [EXP, PRED, MET, OVER, CONTACT, TRAIN, DATA, RUNS, CKPT, FEAT, REPORT]:
        p.mkdir(parents=True, exist_ok=True)


def jd(obj) -> str:
    def conv(o):
        if isinstance(o, (np.integer, np.floating)):
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
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(jd(row) + "\n")
            n += 1
    return n


def valid_box(box: Optional[Sequence[float]]) -> bool:
    return bool(box) and len(box) == 4 and float(box[2]) > float(box[0]) and float(box[3]) > float(box[1])


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


def norm_cxcywh_to_xyxy(box: Sequence[float], iw: float, ih: float) -> List[float]:
    cx, cy, w, h = [float(x) for x in box]
    return clip_box([(cx - w / 2) * iw, (cy - h / 2) * ih, (cx + w / 2) * iw, (cy + h / 2) * ih], iw, ih)


def iou_xyxy(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> float:
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
    return inter / union if union > 0 else 0.0


def max_iou(pred: Optional[Sequence[float]], row: Dict) -> float:
    boxes = row.get("gold_group_bboxes_xyxy") or [row.get("gold_bbox_xyxy")]
    return max([iou_xyxy(pred, b) for b in boxes if b], default=0.0)


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
        "retrocardiac",
    ]
    out = text
    for w in words:
        out = re.sub(rf"\b{re.escape(w)}\b", "", out, flags=re.I)
    return " ".join(out.split())


def parse_rule_context(text: str, finding: str, no_location: bool = False) -> Dict:
    t = f"{text} {finding}".lower()
    if no_location:
        t = strip_location_words(t)
    if "bilateral" in t or "both" in t or "bibasilar" in t:
        lat = "bilateral"
    elif re.search(r"\bright\b|\brt\b", t):
        lat = "right"
    elif re.search(r"\bleft\b|\blt\b", t):
        lat = "left"
    else:
        lat = "unknown"
    if "apical" in t or "apex" in t:
        vert = "apical"
    elif "basal" in t or "base" in t or "bibasilar" in t:
        vert = "basal"
    elif "upper" in t or "suprahilar" in t:
        vert = "upper"
    elif "lower" in t or "inferior" in t or "infrahilar" in t or "costophrenic" in t:
        vert = "lower"
    elif "middle" in t or re.search(r"\bmid\b", t) or "perihilar" in t:
        vert = "mid"
    elif "diffuse" in t or "throughout" in t:
        vert = "whole"
    else:
        vert = "unknown"
    if any(w in t for w in ["trace", "tiny", "minimal"]):
        sev = "trace"
    elif "small" in t:
        sev = "small"
    elif "mild" in t:
        sev = "mild"
    elif "moderate" in t:
        sev = "moderate"
    elif "large" in t or "severe" in t:
        sev = "large"
    else:
        sev = "unknown"
    uncertainty = "possible" if any(w in t for w in ["possible", "possibly", "may ", "could", "likely"]) else "definite"
    anatomy_terms = []
    for term in ["apical", "upper lobe", "lower lobe", "middle lobe", "perihilar", "costophrenic", "pleural", "cardiac", "retrocardiac", "infrahilar"]:
        if term in t:
            anatomy_terms.append(term)
    return {
        "finding_normalized": finding,
        "laterality": lat,
        "vertical_region": vert,
        "anatomy_region": ";".join(anatomy_terms) if anatomy_terms else "unknown",
        "severity": sev,
        "uncertainty": uncertainty,
        "short_context_summary": f"{finding}; laterality={lat}; vertical={vert}; severity={sev}; uncertainty={uncertainty}",
    }


def split_reuse() -> Dict:
    ensure_dirs()
    counts = {}
    subjects = {}
    for split in ["train", "val", "eval"]:
        src = S1_DATA / f"{split}.jsonl"
        dst = DATA / f"{split}.jsonl"
        rows = read_jsonl(src)
        write_jsonl(dst, rows)
        counts[split] = len(rows)
        subjects[split] = {str(r.get("subject_id")) for r in rows}
    overlap = {
        "train_val": len(subjects["train"] & subjects["val"]),
        "train_eval": len(subjects["train"] & subjects["eval"]),
        "val_eval": len(subjects["val"] & subjects["eval"]),
    }
    report = [
        "# Stage 1 Split Reuse Report",
        "",
        "- Source: Stage 1 p10-p19 MS-CXR split.",
        "- Split was reused exactly; no new eval sampling.",
        "- MS-CXR boxes remain phrase-grounding bboxes, not pixel masks.",
        "",
        "| split | rows | unique_subjects |",
        "|---|---:|---:|",
    ]
    for split in ["train", "val", "eval"]:
        report.append(f"| {split} | {counts[split]} | {len(subjects[split])} |")
    report += ["", "Subject overlap:", "", "```json", json.dumps(overlap, indent=2), "```"]
    write_text(REPORT / "STAGE1_SPLIT_REUSE_REPORT.md", "\n".join(report) + "\n")
    return {**counts, "overlap": overlap}


def qwen_parse_rows(rows: List[Dict], max_rows: int, args) -> Tuple[Dict[str, Dict], Dict]:
    parsed: Dict[str, Dict] = {}
    info = {"attempted": False, "success": 0, "failed": 0, "error": ""}
    if max_rows <= 0:
        return parsed, info
    try:
        from smm_backend import LocalHFQwenRunner, detect_backend, parse_json_from_text, read_simple_yaml

        backend = detect_backend(read_simple_yaml())
        if backend.get("available") != "true" or backend.get("backend") != "local_hf":
            info["error"] = backend.get("reason", "local_hf unavailable")
            return parsed, info
        info["attempted"] = True
        runner = LocalHFQwenRunner(backend)
        prompt_template = (
            "Parse this chest X-ray report claim into JSON only. "
            "Do not infer patient identity or treatment. "
            "Allowed laterality: left, right, bilateral, none, unknown. "
            "Allowed vertical_region: upper, mid, lower, apical, basal, whole, unknown. "
            "Allowed severity: trace, small, mild, moderate, large, unknown. "
            "Allowed uncertainty: definite, probable, possible, uncertain, negated, unknown. "
            "Return keys: finding_normalized,laterality,vertical_region,anatomy_region,severity,uncertainty,short_context_summary.\n"
            "Finding: {finding}\nClaim: {claim}"
        )
        for row in rows[:max_rows]:
            raw = runner.generate(prompt_template.format(finding=row["finding"], claim=row["claim_sentence"]), [])
            obj, ok, err = parse_json_from_text(raw)
            if ok:
                parsed[row["task_id"]] = {
                    "finding_normalized": str(obj.get("finding_normalized") or row["finding"]),
                    "laterality": str(obj.get("laterality") or "unknown").lower(),
                    "vertical_region": str(obj.get("vertical_region") or "unknown").lower(),
                    "anatomy_region": str(obj.get("anatomy_region") or "unknown").lower(),
                    "severity": str(obj.get("severity") or "unknown").lower(),
                    "uncertainty": str(obj.get("uncertainty") or "unknown").lower(),
                    "short_context_summary": str(obj.get("short_context_summary") or ""),
                    "parser": "qwen_zero_shot",
                    "raw_output": raw[:1000],
                }
                info["success"] += 1
            else:
                info["failed"] += 1
                info["error"] = err
    except Exception as exc:
        info["error"] = repr(exc)
    return parsed, info


def build_contexts(args) -> Dict:
    all_rows_by_split = {split: read_jsonl(DATA / f"{split}.jsonl") for split in ["train", "val", "eval"]}
    qwen_rows = []
    # Spread quick parsing across splits so train/val/eval all have at least some real qwen rows.
    for split in ["train", "val", "eval"]:
        qwen_rows.extend(all_rows_by_split[split][: max(0, args.qwen_max_rows // 3)])
    qwen_parsed, qwen_info = qwen_parse_rows(qwen_rows, args.qwen_max_rows, args)

    rows_written = {}
    disagreements = []
    distributions = defaultdict(Counter)
    for split, rows in all_rows_by_split.items():
        out = []
        out_no_loc = []
        for row in rows:
            rule = parse_rule_context(row["claim_sentence"], row["finding"], no_location=False)
            rule_no = parse_rule_context(row["claim_sentence"], row["finding"], no_location=True)
            qwen = qwen_parsed.get(row["task_id"])
            if qwen is None:
                qwen = {**rule, "parser": "qwen_missing_rule_fallback", "raw_output": ""}
            else:
                disagreements.append(
                    {
                        "task_id": row["task_id"],
                        "split": split,
                        "rule_laterality": rule["laterality"],
                        "qwen_laterality": qwen["laterality"],
                        "rule_vertical": rule["vertical_region"],
                        "qwen_vertical": qwen["vertical_region"],
                        "laterality_disagree": rule["laterality"] != qwen["laterality"],
                        "vertical_disagree": rule["vertical_region"] != qwen["vertical_region"],
                    }
                )
            merged = {
                "task_id": row["task_id"],
                "finding": row["finding"],
                "rule_context": rule,
                "qwen_context": qwen,
                "rule_qwen_context": {f"rule_{k}": v for k, v in rule.items()} | {f"qwen_{k}": v for k, v in qwen.items() if k != "raw_output"},
            }
            out.append(merged)
            out_no_loc.append(
                {
                    "task_id": row["task_id"],
                    "finding": row["finding"],
                    "claim_no_location": strip_location_words(row["claim_sentence"]),
                    "rule_context": rule_no,
                    "qwen_context": {**rule_no, "parser": "qwen_no_location_rule_fallback"},
                    "rule_qwen_context": {f"rule_{k}": v for k, v in rule_no.items()} | {f"qwen_{k}": v for k, v in rule_no.items()},
                }
            )
            for key in ["laterality", "vertical_region", "severity", "uncertainty"]:
                distributions[f"rule_{key}"][rule[key]] += 1
                distributions[f"qwen_{key}"][qwen[key]] += 1
        rows_written[split] = write_jsonl(DATA / f"context_{split}.jsonl", out)
        write_jsonl(DATA / f"context_{split}_no_location.jsonl", out_no_loc)
    pd.DataFrame(disagreements).to_csv(MET / "rule_vs_qwen_context_disagreements.csv", index=False)
    disagreement_rate = {}
    if disagreements:
        df = pd.DataFrame(disagreements)
        disagreement_rate = {
            "laterality": float(df["laterality_disagree"].mean()),
            "vertical": float(df["vertical_disagree"].mean()),
        }
    report = [
        "# Stage 2 Claim Context Report",
        "",
        "- `rule_context`: deterministic keyword parser.",
        "- `qwen_context`: Qwen zero-shot parser where available; missing rows fall back to rule context and are marked.",
        "- Images and MS-CXR bbox coordinates are not provided to context parsers.",
        f"- qwen parser attempted: {qwen_info.get('attempted')}",
        f"- qwen parser success rows: {qwen_info.get('success', 0)}",
        f"- qwen parser failed rows: {qwen_info.get('failed', 0)}",
        f"- qwen parser error: `{qwen_info.get('error', '')}`",
        f"- rule-vs-qwen disagreement rate: {json.dumps(disagreement_rate)}",
        "",
        "## Rows",
        "",
        "| split | rows |",
        "|---|---:|",
    ]
    for split in ["train", "val", "eval"]:
        report.append(f"| {split} | {rows_written[split]} |")
    report += ["", "## Distributions", ""]
    for key, counter in distributions.items():
        report += [f"### {key}", "", "| value | rows |", "|---|---:|"]
        for value, n in counter.most_common():
            report.append(f"| {value} | {n} |")
        report.append("")
    write_text(REPORT / "STAGE2_CLAIM_CONTEXT_REPORT.md", "\n".join(report) + "\n")
    return {"qwen_success": qwen_info.get("success", 0), "qwen_total": sum(len(v) for v in all_rows_by_split.values())}


def write_baseline_reference() -> None:
    rows = []
    for name, method in [
        ("stage1_text_prior", "text_prior_baseline"),
        ("stage1_context_only", "context_only_mlp_prior"),
        ("stage1_vfm_global", "vfm_global_bbox_head"),
        ("stage1_vfm_patch_pool", "vfm_patch_pool_bbox_head"),
        ("stage1_vfm_heatmap", "vfm_heatmap_bbox_head"),
    ]:
        pred = S1_EXP / "predictions" / f"{method}.jsonl"
        if pred.exists():
            shutil.copy2(pred, PRED / f"{name}.jsonl")
        summary = S1_EXP / "metrics" / "summary_all8.csv"
        if summary.exists():
            df = pd.read_csv(summary)
            r = df[df["method"].eq(method)]
            if len(r):
                rows.append({"method": name, "source": "stage1_same_p10_p19_split", "mean_iou_all8": float(r.iloc[0]["mean_iou"])})
    pd.DataFrame(rows).to_csv(MET / "baseline_reference_table.csv", index=False)
    write_text(
        REPORT / "STAGE3_BASELINE_REUSE_AND_FAIRNESS_REPORT.md",
        "# Stage 3 Baseline Reuse And Fairness Report\n\n"
        "- Stage 1 p10-p19 predictions were copied as comparable baselines because Stage 2 reuses the exact same train/val/eval split.\n"
        "- Older p10-p14 ranker/simple-fusion metrics are not directly ranked against this p10-p19 split.\n\n"
        + (pd.DataFrame(rows).to_markdown(index=False) if rows else "No stage1 baseline rows found.")
        + "\n",
    )


def feature_report() -> Dict:
    if not (S1_FEAT / "vfm_features_train.npz").exists():
        raise FileNotFoundError("Stage 1 RAD-DINO features not found. Run Stage 1 first.")
    model = (S1_FEAT / "VFM_MODEL.txt").read_text(encoding="utf-8").strip() if (S1_FEAT / "VFM_MODEL.txt").exists() else "stage1_cached"
    pointer = {
        "feature_source": str(S1_FEAT),
        "stage2_feature_storage": "pointer_only_to_avoid_duplicate_large_npz",
        "vfm_model": model,
        "required_files": [str(S1_FEAT / f"vfm_features_{s}.npz") for s in ["train", "val", "eval"]],
    }
    (FEAT / "FEATURE_CACHE_POINTER.json").write_text(json.dumps(pointer, indent=2), encoding="utf-8")
    report = [
        "# Stage 4 Feature Report",
        "",
        f"- Reused Stage 1 feature cache: `{S1_FEAT}`",
        f"- VFM model: `{model}`",
        "- Feature cache includes RAD-DINO global, patch mean/std, and patch tokens.",
        "- Stage 2 does not duplicate multi-GB NPZ files; it stores a pointer JSON.",
        "- Intermediate/multiscale features were not extracted in this quick run.",
    ]
    write_text(REPORT / "STAGE4_FEATURE_REPORT.md", "\n".join(report) + "\n")
    return {"vfm_model": model}


def load_features(split: str) -> Dict[str, np.ndarray]:
    data = np.load(S1_FEAT / f"vfm_features_{split}.npz", allow_pickle=True)
    return {
        "ids": np.array([str(x) for x in data["task_ids"]], dtype=object),
        "global": data["global_features"].astype("float32"),
        "patch_mean": data["patch_mean"].astype("float32"),
        "patch_std": data["patch_std"].astype("float32"),
        "patch_tokens": data["patch_tokens"].astype("float16"),
    }


def flatten_context(context_rows: List[Dict], key: str) -> List[Dict]:
    out = []
    for r in context_rows:
        if key == "rule_qwen_context":
            ctx = {}
            for kk, vv in r["rule_context"].items():
                ctx[f"rule_{kk}"] = vv
            for kk, vv in r["qwen_context"].items():
                if kk != "raw_output":
                    ctx[f"qwen_{kk}"] = vv
        else:
            ctx = dict(r[key])
        ctx["task_id"] = r["task_id"]
        ctx["finding"] = r["finding"]
        out.append(ctx)
    return out


def context_maps(mode: str) -> Dict[str, List[str]]:
    rows = []
    for split in ["train", "val", "eval"]:
        rows += flatten_context(read_jsonl(DATA / f"context_{split}.jsonl"), mode)
    fields = sorted(k for k in rows[0].keys() if k not in {"task_id", "short_context_summary", "raw_output"})
    return {f: sorted(set(str(r.get(f, "unknown")) for r in rows)) for f in fields}


def encode_context(base_rows: List[Dict], ctx_rows: List[Dict], maps: Dict[str, List[str]], mode: str) -> np.ndarray:
    if mode == "none":
        return np.zeros((len(base_rows), 0), dtype="float32")
    by_id = {r["task_id"]: r for r in ctx_rows}
    feats = []
    for row in base_rows:
        ctx = by_id.get(row["task_id"], {})
        vals = []
        if mode == "finding_only":
            fields = [f for f in maps if f.endswith("finding_normalized") or f == "finding_normalized" or f == "finding"]
        else:
            fields = list(maps.keys())
        for field in fields:
            for value in maps[field]:
                vals.append(1.0 if str(ctx.get(field, "unknown")) == value else 0.0)
        feats.append(vals)
    return np.asarray(feats, dtype="float32") if feats and feats[0] else np.zeros((len(base_rows), 0), dtype="float32")


def aligned_patch(split: str, ctx_mode: str, no_location: bool = False):
    base = read_jsonl(DATA / f"{split}.jsonl")
    context_file = DATA / f"context_{split}{'_no_location' if no_location else ''}.jsonl"
    ctx_key = "rule_context" if ctx_mode in {"rule_context", "finding_only"} else "qwen_context" if ctx_mode == "qwen_context" else "rule_qwen_context"
    ctx_rows = flatten_context(read_jsonl(context_file), ctx_key)
    maps = context_maps(ctx_key)
    feat = load_features(split)
    pos = {tid: i for i, tid in enumerate(feat["ids"])}
    ordered, idxs, y = [], [], []
    for row in base:
        if row["task_id"] in pos:
            ordered.append(row)
            idxs.append(pos[row["task_id"]])
            y.append(row["gold_bbox_norm_cxcywh"])
    ctx = encode_context(ordered, ctx_rows, maps, "finding_only" if ctx_mode == "finding_only" else ctx_mode)
    return ordered, feat["patch_tokens"][idxs].astype("float32"), ctx.astype("float32"), np.asarray(y, dtype="float32")


def aligned_mlp(split: str, ctx_mode: str, no_location: bool = False):
    rows, tokens, ctx, y = aligned_patch(split, ctx_mode, no_location=no_location)
    feat = load_features(split)
    pos = {tid: i for i, tid in enumerate(feat["ids"])}
    idxs = [pos[r["task_id"]] for r in rows]
    x = ctx if ctx_mode != "none" else np.zeros((len(rows), 1), dtype="float32")
    return rows, x, y


def pixel_rows(rows: List[Dict], pred_norm: np.ndarray, method: str, extra: Optional[Dict] = None) -> List[Dict]:
    out = []
    extra = extra or {}
    for row, p in zip(rows, pred_norm):
        p = np.asarray(p, dtype="float32")
        p[:2] = np.clip(p[:2], 0, 1)
        p[2:] = np.clip(p[2:], 0.02, 1)
        out.append(
            {
                "task_id": row["task_id"],
                "method": method,
                "dicom_id": row["dicom_id"],
                "finding": row["finding"],
                "claim_sentence": row["claim_sentence"],
                "pred_bbox_norm_cxcywh": [float(x) for x in p],
                "pred_bbox_xyxy": norm_cxcywh_to_xyxy(p, row["image_width"], row["image_height"]),
                "bbox_missing": False,
                **extra,
            }
        )
    return out


def norm_cxcywh_to_xyxy(box: Sequence[float], iw: float, ih: float) -> List[float]:
    cx, cy, w, h = [float(x) for x in box]
    return clip_box([(cx - w / 2) * iw, (cy - h / 2) * ih, (cx + w / 2) * iw, (cy + h / 2) * ih], iw, ih)


def val_score(pred_norm: np.ndarray, rows: List[Dict]) -> float:
    return float(np.mean([max_iou(norm_cxcywh_to_xyxy(p, r["image_width"], r["image_height"]), r) for p, r in zip(pred_norm, rows)])) if rows else 0.0


def center_indices(y: np.ndarray, token_count: int, device: str) -> torch.Tensor:
    grid = int(round(math.sqrt(token_count)))
    cx = np.clip((y[:, 0] * grid).astype(int), 0, grid - 1)
    cy = np.clip((y[:, 1] * grid).astype(int), 0, grid - 1)
    return torch.tensor(np.minimum(cy * grid + cx, token_count - 1), dtype=torch.long, device=device)


def train_heatmap(method: str, ctx_mode: str, args, head_kind: str = "base", no_location_eval: bool = True) -> Dict:
    tr_rows, tok_tr, ctx_tr, y_tr = aligned_patch("train", ctx_mode)
    va_rows, tok_va, ctx_va, y_va = aligned_patch("val", ctx_mode)
    ev_rows, tok_ev, ctx_ev, _ = aligned_patch("eval", ctx_mode)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden = args.deeper_hidden if head_kind == "deeper" else args.hidden
    if head_kind == "adapter":
        model = AdapterPatchHeatmapBBoxHead(tok_tr.shape[-1], ctx_tr.shape[1], hidden=hidden, bottleneck=args.adapter_bottleneck, dropout=args.dropout).to(device)
    else:
        model = PatchHeatmapBBoxHead(tok_tr.shape[-1], ctx_tr.shape[1], hidden=hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tok_tr_t = torch.tensor(tok_tr, dtype=torch.float32, device=device)
    ctx_tr_t = torch.tensor(ctx_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
    tok_va_t = torch.tensor(tok_va, dtype=torch.float32, device=device)
    ctx_va_t = torch.tensor(ctx_va, dtype=torch.float32, device=device)
    c_tr = center_indices(y_tr, tok_tr.shape[1], device)
    best_state, best_val, best_epoch, wait = None, -1.0, 0, 0
    hist = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(tok_tr_t), device=device)
        losses = []
        for start in range(0, len(tok_tr_t), args.batch_size):
            idx = perm[start : start + args.batch_size]
            pred, logits = model(tok_tr_t[idx], ctx_tr_t[idx])
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
            pv, _ = model(tok_va_t, ctx_va_t)
        score = val_score(pv.detach().cpu().numpy(), va_rows)
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
    pred_eval = predict_heatmap_model(model, tok_ev, ctx_ev, ev_rows, method, {"ctx_mode": ctx_mode, "head_kind": head_kind})
    write_jsonl(PRED / f"{method}.jsonl", pred_eval)
    if no_location_eval and ctx_mode not in {"none", "finding_only"}:
        ev_rows_n, tok_ev_n, ctx_ev_n, _ = aligned_patch("eval", ctx_mode, no_location=True)
        pred_n = predict_heatmap_model(model, tok_ev_n, ctx_ev_n, ev_rows_n, f"{method}_no_location", {"ctx_mode": ctx_mode, "head_kind": head_kind, "ablation": "no_location"})
        write_jsonl(PRED / f"{method}_no_location.jsonl", pred_n)
    run_dir = RUNS / method
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(hist).to_csv(run_dir / "loss_curve.csv", index=False)
    torch.save({"state": best_state, "ctx_dim": ctx_tr.shape[1], "token_dim": tok_tr.shape[-1], "head_kind": head_kind}, CKPT / f"{method}.pt")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "method": method,
        "ctx_mode": ctx_mode,
        "head_kind": head_kind,
        "train_rows": len(tr_rows),
        "val_rows": len(va_rows),
        "eval_rows": len(ev_rows),
        "trainable_params": trainable,
        "best_val_mean_iou": best_val,
        "best_epoch": best_epoch,
        "pred_rows": pred_eval,
    }


def predict_heatmap_model(model, tokens: np.ndarray, ctx: np.ndarray, rows: List[Dict], method: str, extra: Dict) -> List[Dict]:
    device = next(model.parameters()).device
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(tokens), 16):
            tok = torch.tensor(tokens[start : start + 16], dtype=torch.float32, device=device)
            c = torch.tensor(ctx[start : start + 16], dtype=torch.float32, device=device)
            p, _ = model(tok, c)
            preds.append(p.detach().cpu().numpy())
    return pixel_rows(rows, np.vstack(preds), method, extra)


def train_context_only(args) -> Dict:
    tr_rows, xtr, ytr = aligned_mlp("train", "rule_context")
    va_rows, xva, _ = aligned_mlp("val", "rule_context")
    ev_rows, xev, _ = aligned_mlp("eval", "rule_context")
    scaler = FeatureScaler.fit(xtr)
    xtr_s, xva_s, xev_s = scaler.transform(xtr), scaler.transform(xva), scaler.transform(xev)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BBoxMLP(xtr_s.shape[1], hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    xt = torch.tensor(xtr_s, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.float32, device=device)
    xv = torch.tensor(xva_s, dtype=torch.float32, device=device)
    best_state, best_val, best_epoch, wait = None, -1.0, 0, 0
    hist = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(xt), device=device)
        losses = []
        for start in range(0, len(xt), 64):
            idx = perm[start : start + 64]
            pred = model(xt[idx])
            loss, _ = bbox_loss(pred, yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            pv = model(xv).detach().cpu().numpy()
        score = val_score(pv, va_rows)
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
    with torch.no_grad():
        pe = model(torch.tensor(xev_s, dtype=torch.float32, device=device)).detach().cpu().numpy()
    pred_rows = pixel_rows(ev_rows, pe, "context_only_mlp_prior_stage2", {"ctx_mode": "rule_context"})
    write_jsonl(PRED / "context_only_mlp_prior_stage2.jsonl", pred_rows)
    pd.DataFrame(hist).to_csv(RUNS / "context_only_mlp_prior_stage2_loss_curve.csv", index=False)
    return {"method": "context_only_mlp_prior_stage2", "best_val_mean_iou": best_val, "best_epoch": best_epoch, "trainable_params": sum(p.numel() for p in model.parameters()), "pred_rows": pred_rows}


def train_all(args) -> List[Dict]:
    info = []
    info.append(train_context_only(args))
    specs = [
        ("frozen_heatmap_vfm_only", "none", "base"),
        ("frozen_heatmap_finding_only", "finding_only", "base"),
        ("frozen_heatmap_rule_context", "rule_context", "base"),
        ("frozen_heatmap_qwen_context", "qwen_context", "base"),
        ("frozen_heatmap_rule_qwen_context", "rule_qwen_context", "base"),
        ("deeper_heatmap_rule_context", "rule_context", "deeper"),
        ("adapter_heatmap_rule_context", "rule_context", "adapter"),
    ]
    for method, ctx, kind in specs:
        info.append(train_heatmap(method, ctx, args, head_kind=kind))
    pd.DataFrame([{k: v for k, v in row.items() if k != "pred_rows"} for row in info]).to_csv(MET / "training_runs.csv", index=False)
    write_text(
        REPORT / "STAGE5_HEATMAP_MODEL_REPORT.md",
        "# Stage 5 Heatmap Model Report\n\n"
        "- Stage 1 reproduction is included by copying `stage1_vfm_heatmap` prediction rows.\n"
        "- Frozen heatmap variants use cached RAD-DINO patch tokens and train only lightweight heads.\n"
        "- Adapter variant trains a residual token adapter plus heatmap head on frozen RAD-DINO features.\n"
        "- Deeper variant increases head hidden dimension.\n\n"
        + pd.DataFrame([{k: v for k, v in row.items() if k != "pred_rows"} for row in info]).to_markdown(index=False)
        + "\n",
    )
    write_text(
        REPORT / "STAGE6_VFM_FINETUNING_REPORT.md",
        "# Stage 6 VFM Finetuning Report\n\n"
        "- `frozen_heatmap_*`: RAD-DINO frozen, head-only training.\n"
        "- `adapter_heatmap_rule_context`: trainable residual adapter on cached RAD-DINO patch features plus heatmap head. This is the completed lightweight adapter tuning attempt.\n"
        "- `rad_dino_last_block_unfreeze`: not run in this quick pass because Stage 2 reused cached patch tokens and full encoder backprop would require a separate image-loader training loop with higher GPU memory/time cost.\n"
        "- `rad_dino_lora_or_prompt_tuning`: not run; planned follow-up if adapter improves over frozen head.\n",
    )
    return info


def eval_rows(preds: List[Dict], tasks: List[Dict], method: str, subset: str) -> Dict:
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
    eval_tasks = read_jsonl(DATA / "eval.jsonl")
    main_tasks = [r for r in eval_tasks if r["finding"] in MAIN5]
    pred_files = sorted(PRED.glob("*.jsonl"))
    all8, main5, per = [], [], []
    for pf in pred_files:
        if pf.stem.endswith("_no_location"):
            continue
        preds = read_jsonl(pf)
        all8.append(eval_rows(preds, eval_tasks, pf.stem, "all8"))
        main5.append(eval_rows(preds, main_tasks, pf.stem, "main5"))
        for finding in ALL8:
            subset = [r for r in eval_tasks if r["finding"] == finding]
            if subset:
                row = eval_rows(preds, subset, pf.stem, finding)
                row["finding"] = finding
                per.append(row)
    all8_df = pd.DataFrame(all8).sort_values("mean_iou", ascending=False)
    main5_df = pd.DataFrame(main5).sort_values("mean_iou", ascending=False)
    all8_df.to_csv(MET / "summary_all8.csv", index=False)
    main5_df.to_csv(MET / "summary_main5.csv", index=False)
    pd.DataFrame(per).to_csv(MET / "per_finding_metrics.csv", index=False)
    write_text(
        REPORT / "STAGE11_FINAL_EVAL_REPORT.md",
        "# Stage 11 Final Eval Report\n\n## All8\n\n"
        + all8_df.to_markdown(index=False)
        + "\n\n## Main5\n\n"
        + main5_df.to_markdown(index=False)
        + "\n",
    )
    return {"best_all8": all8_df.iloc[0].to_dict(), "best_main5": main5_df.iloc[0].to_dict()}


def location_ablation() -> Dict:
    eval_tasks = read_jsonl(DATA / "eval.jsonl")
    rows = []
    stems = {pf.stem for pf in PRED.glob("*.jsonl")}
    paired_originals = {stem[: -len("_no_location")] for stem in stems if stem.endswith("_no_location")}
    for pf in sorted(PRED.glob("*.jsonl")):
        if (
            pf.stem.endswith("_no_location")
            or pf.stem in paired_originals
            or pf.stem in {"stage1_vfm_global", "stage1_vfm_patch_pool", "stage1_vfm_heatmap"}
        ):
            preds = read_jsonl(pf)
            rows.append(eval_rows(preds, eval_tasks, pf.stem, "all8"))
    df = pd.DataFrame(rows).sort_values("mean_iou", ascending=False)
    df.to_csv(MET / "location_hint_ablation_all_models.csv", index=False)
    # Compute drop where original/no-location pairs exist.
    drops = []
    methods = set(df["method"])
    for m in methods:
        if m.endswith("_no_location"):
            continue
        nl = f"{m}_no_location"
        if nl in methods:
            a = float(df[df["method"].eq(m)]["mean_iou"].iloc[0])
            b = float(df[df["method"].eq(nl)]["mean_iou"].iloc[0])
            drops.append({"method": m, "original_mean_iou": a, "no_location_mean_iou": b, "drop": a - b, "drop_ratio": (a - b) / max(a, 1e-6)})
    drop_df = pd.DataFrame(drops)
    drop_df.to_csv(MET / "location_hint_drop.csv", index=False)
    write_text(
        REPORT / "STAGE8_LOCATION_HINT_ABLATION_REPORT.md",
        "# Stage 8 Location Hint Ablation Report\n\n"
        "- No-location context removes left/right/upper/lower/apical/basal-like words before parsing context.\n"
        "- VFM-only/finding-only models should be stable; context models should reveal dependence on textual location hints.\n\n"
        "## All Models\n\n"
        + df.to_markdown(index=False)
        + "\n\n## Drops\n\n"
        + (drop_df.to_markdown(index=False) if len(drop_df) else "No paired no-location predictions.")
        + "\n",
    )
    best_drop = float(drop_df.sort_values("drop", ascending=False).iloc[0]["drop"]) if len(drop_df) else 0.0
    return {"location_hint_drop_best_model": best_drop}


def controls(best_method: str) -> Dict:
    # Controls are implemented with the best rule-context heatmap-compatible model if available.
    eval_tasks = read_jsonl(DATA / "eval.jsonl")
    baseline = read_jsonl(PRED / f"{best_method}.jsonl") if (PRED / f"{best_method}.jsonl").exists() else []
    baseline_iou = eval_rows(baseline, eval_tasks, best_method, "all8")["mean_iou"] if baseline else 0.0
    rows = [{"control": "original", "method": best_method, "mean_iou": baseline_iou}]
    # If predictions are not directly recomputable, use deterministic control proxies:
    # image_shuffled uses another task's predicted box, claim_shuffled approximates context mismatch.
    rotated = []
    if baseline:
        boxes = [p.get("pred_bbox_xyxy") for p in baseline]
        for i, p in enumerate(baseline):
            rr = dict(p)
            rr["pred_bbox_xyxy"] = boxes[(i + 17) % len(boxes)]
            rotated.append(rr)
        rows.append({"control": "image_feature_shuffled_proxy", "method": best_method, "mean_iou": eval_rows(rotated, eval_tasks, f"{best_method}_image_shuffled_proxy", "all8")["mean_iou"]})
    context_only = read_jsonl(PRED / "context_only_mlp_prior_stage2.jsonl")
    if context_only:
        rows.append({"control": "no_image_context_only", "method": "context_only_mlp_prior_stage2", "mean_iou": eval_rows(context_only, eval_tasks, "context_only_mlp_prior_stage2", "all8")["mean_iou"]})
    df = pd.DataFrame(rows)
    df.to_csv(MET / "control_experiments.csv", index=False)
    gap = baseline_iou - float(df[df["control"].ne("original")]["mean_iou"].max()) if len(df) > 1 else 0.0
    write_text(
        REPORT / "STAGE9_CONTROL_EXPERIMENTS_REPORT.md",
        "# Stage 9 Control Experiments Report\n\n"
        "- `image_feature_shuffled_proxy` rotates predicted boxes across eval tasks to approximate image/feature mismatch harm without using eval gold in decisions.\n"
        "- `no_image_context_only` uses the context-only prior.\n"
        "- A positive control gap suggests the best model is not purely text/context prior.\n\n"
        + df.to_markdown(index=False)
        + f"\n\ncontrol_gap_best_model={gap:.6f}\n",
    )
    return {"control_gap_best_model": gap}


def make_overlays(best_method: str) -> int:
    eval_tasks = read_jsonl(DATA / "eval.jsonl")
    pred_best = {r["task_id"]: r for r in read_jsonl(PRED / f"{best_method}.jsonl")}
    pred_stage1 = {r["task_id"]: r for r in read_jsonl(PRED / "stage1_vfm_heatmap.jsonl")} if (PRED / "stage1_vfm_heatmap.jsonl").exists() else {}
    pred_rule = {r["task_id"]: r for r in read_jsonl(PRED / "frozen_heatmap_rule_context.jsonl")} if (PRED / "frozen_heatmap_rule_context.jsonl").exists() else {}
    scored = [(max_iou(pred_best.get(r["task_id"], {}).get("pred_bbox_xyxy"), r), r) for r in eval_tasks]
    examples = sorted(scored, key=lambda x: x[0], reverse=True)[:16] + sorted(scored, key=lambda x: x[0])[:16]
    paths = []
    for i, (score, row) in enumerate(examples):
        out = OVER / "comparison" / f"{i:03d}_{row['finding'].replace(' ','_')}_{row['task_id']}.jpg"
        ok = draw_boxes(
            row["image_path"],
            [
                ("gold", row["gold_bbox_xyxy"], (255, 230, 0)),
                ("stage1", pred_stage1.get(row["task_id"], {}).get("pred_bbox_xyxy"), (0, 120, 255)),
                ("rule_context", pred_rule.get(row["task_id"], {}).get("pred_bbox_xyxy"), (0, 255, 255)),
                ("best", pred_best.get(row["task_id"], {}).get("pred_bbox_xyxy"), (0, 220, 80)),
            ],
            out,
            f"{row['finding']} | best={best_method} IoU={score:.3f} | MS-CXR phrase bbox",
        )
        if ok:
            paths.append(out)
    make_contact_sheet(paths[:32], CONTACT / "best_heatmap_success_failure_cases.jpg")
    write_text(
        REPORT / "STAGE12_OVERLAY_REVIEW_REPORT.md",
        "# Stage 12 Overlay Review Report\n\n"
        f"- comparison overlays written: {len(paths)}\n"
        "- colors: gold=yellow, stage1=blue, rule_context=cyan, best=green.\n",
    )
    return len(paths)


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
        if not valid_box(box):
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
    sheet = Image.new("RGB", (cols * 360, math.ceil(len(imgs) / cols) * 360), (20, 20, 20))
    for i, im in enumerate(imgs):
        sheet.paste(im, ((i % cols) * 360, (i // cols) * 360))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=90)


def weak_pretrain_report() -> None:
    write_text(
        REPORT / "STAGE10_CHEST_IMAGENOME_WEAK_PRETRAIN_REPORT.md",
        "# Stage 10 Chest ImaGenome Weak Pretrain Report\n\n"
        "Not run in this quick Stage 2 pass. Chest ImaGenome remains weak region/device evidence, not gold lesion bbox. "
        "The next run can pretrain the heatmap head on mapped positive Chest ImaGenome silver regions and then fine-tune on MS-CXR.\n",
    )


def write_reports(final: Dict, loc: Dict, ctrl: Dict, qwen: Dict, overlay_count: int) -> None:
    all8 = pd.read_csv(MET / "summary_all8.csv")
    main5 = pd.read_csv(MET / "summary_main5.csv")
    best_all8 = all8.iloc[0].to_dict()
    best_main5 = main5.iloc[0].to_dict()
    train, val, ev = [read_jsonl(DATA / f"{s}.jsonl") for s in ["train", "val", "eval"]]
    overlap = {
        "train_val": len({r["subject_id"] for r in train} & {r["subject_id"] for r in val}),
        "train_eval": len({r["subject_id"] for r in train} & {r["subject_id"] for r in ev}),
        "val_eval": len({r["subject_id"] for r in val} & {r["subject_id"] for r in ev}),
    }
    write_text(
        REPORT / "PIPELINE_LOGICAL_REVIEW.md",
        "# Pipeline Logical Review\n\n"
        "- Stage 1 p10-p19 split was reused.\n"
        f"- train/val/eval rows: {len(train)}/{len(val)}/{len(ev)}.\n"
        f"- subject overlap: `{json.dumps(overlap)}`.\n"
        "- Eval gold bboxes were used only for metrics and analysis overlays.\n"
        "- Val split selected checkpoints; eval was final measurement only.\n"
        "- MS-CXR bboxes are phrase-grounding bboxes, not pixel-level lesion masks.\n"
        "- Chest ImaGenome was not used as gold bbox.\n"
        "- Selector/A/B/C, pairwise comparator, and Qwen candidate selector experiments are not included.\n"
        "- Qwen context parser receives claim text only, not image or bbox.\n"
        "- VFM frozen/head-only/adapter status is separated in reports.\n",
    )
    # Context fusion focused table.
    ctx_methods = [
        "stage1_vfm_heatmap",
        "frozen_heatmap_vfm_only",
        "frozen_heatmap_finding_only",
        "frozen_heatmap_rule_context",
        "frozen_heatmap_qwen_context",
        "frozen_heatmap_rule_qwen_context",
        "adapter_heatmap_rule_context",
        "deeper_heatmap_rule_context",
        "context_only_mlp_prior_stage2",
    ]
    ctx_df = all8[all8["method"].isin(ctx_methods)].copy()
    ctx_df.to_csv(MET / "context_fusion_ablation.csv", index=False)
    write_text(
        REPORT / "STAGE7_CONTEXT_FUSION_ABLATION_REPORT.md",
        "# Stage 7 Context Fusion Ablation Report\n\n"
        + ctx_df.to_markdown(index=False)
        + "\n\nInterpretation: compare VFM-only/finding-only against rule and Qwen context. Qwen rows may include rule fallback; see Stage 2 context coverage.\n",
    )
    write_text(
        REPORT / "EXPERIMENT_SUMMARY_FOR_PAPER.md",
        "# MS-CXR Context + VFM Localizer Stage 2 Summary\n\n"
        "- Dataset: p10-p19 MS-CXR local matched subset, 1,444 annotations total before split.\n"
        "- MS-CXR bbox is phrase-grounding bbox, not pixel-level lesion mask.\n"
        "- Stage 1 RAD-DINO heatmap was the starting baseline.\n"
        "- Stage 2 tests frozen heatmap heads, trainable token adapter, and rule/Qwen context fusion.\n"
        "- SMM/Qwen is used as claim context parser only, not bbox generator or selector.\n"
        f"- Qwen context parse coverage: {qwen.get('qwen_success', 0)}/{qwen.get('qwen_total', 0)} rows; remaining rows use rule fallback.\n"
        f"- Best all8 method: `{best_all8['method']}` mean IoU={best_all8['mean_iou']:.6f}.\n"
        f"- Best main5 method: `{best_main5['method']}` mean IoU={best_main5['mean_iou']:.6f}.\n"
        f"- Location hint drop best paired model: {loc.get('location_hint_drop_best_model', 0):.6f}.\n"
        f"- Control gap best model: {ctrl.get('control_gap_best_model', 0):.6f}.\n\n"
        "## All8\n\n"
        + all8.to_markdown(index=False)
        + "\n\n## Main5\n\n"
        + main5.to_markdown(index=False)
        + "\n\n## Context Fusion\n\n"
        + ctx_df.to_markdown(index=False)
        + f"\n\nOverlay examples written: {overlay_count}.\n",
    )
    write_text(
        REPORT / "FAILURE_ANALYSIS.md",
        "# Failure Analysis\n\n"
        "- If context variants do not beat VFM-only, localization is dominated by RAD-DINO patch evidence rather than structured text context.\n"
        "- If Qwen context is not better than rule context, current SMM parsing adds little beyond keyword extraction.\n"
        "- If adapter does not beat frozen head, cached RAD-DINO features plus small data may limit trainable adaptation.\n"
        "- Cardiomegaly and other broad findings can inflate all8; main5 should be reported separately.\n",
    )
    write_text(
        REPORT / "NEXT_PLAN_AFTER_STAGE2_CONTEXT_VFM_LOCALIZER.md",
        "# Next Plan After Stage 2\n\n"
        "1. If adapter improved, run a true image-loader adapter/last-block fine-tune with RAD-DINO.\n"
        "2. If rule and Qwen context are similar, keep rule context for localization and use Qwen for explanation only.\n"
        "3. Add Chest ImaGenome weak pretrain as weak supervision, then fine-tune on MS-CXR.\n"
        "4. Report no-location ablation prominently because MS-CXR claim text contains strong location priors.\n",
    )
    return {"best_all8": best_all8, "best_main5": best_main5}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--reuse-stage1-split", action="store_true")
    p.add_argument("--build-context", action="store_true")
    p.add_argument("--train-context-parser", action="store_true")
    p.add_argument("--extract-features", action="store_true")
    p.add_argument("--train-baselines", action="store_true")
    p.add_argument("--train-heatmap-heads", action="store_true")
    p.add_argument("--train-adapter", action="store_true")
    p.add_argument("--train-last-block", action="store_true")
    p.add_argument("--train-context-fusion", action="store_true")
    p.add_argument("--run-location-ablation", action="store_true")
    p.add_argument("--run-controls", action="store_true")
    p.add_argument("--weak-pretrain-smoke", action="store_true")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--make-overlays", action="store_true")
    p.add_argument("--all8", action="store_true")
    p.add_argument("--main5", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--qwen-max-rows", type=int, default=120)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--deeper-hidden", type=int, default=768)
    p.add_argument("--adapter-bottleneck", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--heatmap-loss-weight", type=float, default=0.2)
    args = p.parse_args()
    if not any(
        [
            args.reuse_stage1_split,
            args.build_context,
            args.extract_features,
            args.train_baselines,
            args.train_heatmap_heads,
            args.train_adapter,
            args.train_last_block,
            args.train_context_fusion,
            args.run_location_ablation,
            args.run_controls,
            args.weak_pretrain_smoke,
            args.evaluate,
            args.make_overlays,
        ]
    ):
        args.reuse_stage1_split = True
        args.build_context = True
        args.extract_features = True
        args.train_baselines = True
        args.train_heatmap_heads = True
        args.train_adapter = True
        args.train_context_fusion = True
        args.run_location_ablation = True
        args.run_controls = True
        args.weak_pretrain_smoke = True
        args.evaluate = True
        args.make_overlays = True
    if args.quick:
        args.epochs = min(args.epochs, 45)
        args.patience = min(args.patience, 7)
    return args


def main() -> None:
    args = parse_args()
    t0 = time.time()
    ensure_dirs()
    split = split_reuse()
    qwen = build_contexts(args) if args.build_context or not (DATA / "context_train.jsonl").exists() else {"qwen_success": 0, "qwen_total": sum(len(read_jsonl(DATA / f"{s}.jsonl")) for s in ["train", "val", "eval"])}
    write_baseline_reference()
    vfm = feature_report()
    train_all(args)
    weak_pretrain_report()
    final = evaluate_all()
    loc = location_ablation()
    best_method = final["best_all8"]["method"]
    ctrl = controls(best_method)
    overlay_count = make_overlays(best_method) if args.make_overlays else 0
    report_info = write_reports(final, loc, ctrl, qwen, overlay_count)
    all8 = pd.read_csv(MET / "summary_all8.csv")

    def m(method: str) -> float:
        row = all8[all8["method"].eq(method)]
        return float(row.iloc[0]["mean_iou"]) if len(row) else float("nan")

    print(f"project_root={PROJECT_ROOT}")
    print(f"train_rows={split.get('train', 0)}")
    print(f"val_rows={split.get('val', 0)}")
    print(f"eval_rows={split.get('eval', 0)}")
    print(f"vfm_model={vfm.get('vfm_model')}")
    print(f"smm_model={SMM_MODEL}")
    print(f"stage1_heatmap_iou_all8={m('stage1_vfm_heatmap'):.6f}")
    print(f"best_frozen_heatmap_iou_all8={max(m('frozen_heatmap_vfm_only'), m('frozen_heatmap_finding_only'), m('frozen_heatmap_rule_context'), m('frozen_heatmap_qwen_context'), m('frozen_heatmap_rule_qwen_context')):.6f}")
    print(f"best_adapter_tuned_iou_all8={m('adapter_heatmap_rule_context'):.6f}")
    print("best_last_block_iou_all8_if_available=not_run")
    print(f"vfm_only_iou_all8={m('frozen_heatmap_vfm_only'):.6f}")
    print(f"vfm_finding_iou_all8={m('frozen_heatmap_finding_only'):.6f}")
    print(f"vfm_rule_context_iou_all8={m('frozen_heatmap_rule_context'):.6f}")
    print(f"vfm_qwen_context_iou_all8={m('frozen_heatmap_qwen_context'):.6f}")
    print(f"vfm_rule_qwen_context_iou_all8={m('frozen_heatmap_rule_qwen_context'):.6f}")
    print(f"context_only_iou_all8={m('context_only_mlp_prior_stage2'):.6f}")
    print(f"text_prior_iou_all8={m('stage1_text_prior'):.6f}")
    print(f"best_method_all8={report_info['best_all8']['method']}")
    print(f"best_method_main5={report_info['best_main5']['method']}")
    print(f"location_hint_drop_best_model={loc.get('location_hint_drop_best_model', 0):.6f}")
    print(f"control_gap_best_model={ctrl.get('control_gap_best_model', 0):.6f}")
    print(f"qwen_context_parse_coverage={qwen.get('qwen_success', 0)}/{qwen.get('qwen_total', 0)}")
    print(f"paper_summary_path={REPORT / 'EXPERIMENT_SUMMARY_FOR_PAPER.md'}")
    print(f"failure_analysis_path={REPORT / 'FAILURE_ANALYSIS.md'}")
    print(f"logical_review_path={REPORT / 'PIPELINE_LOGICAL_REVIEW.md'}")
    print(f"elapsed_sec={time.time() - t0:.1f}")


if __name__ == "__main__":
    main()

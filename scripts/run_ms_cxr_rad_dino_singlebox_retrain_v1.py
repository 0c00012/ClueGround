#!/usr/bin/env python
"""Retrain the RAD-DINO query head on the MS-CXR single-box split.

This is a fairness check for the MedRPG comparison. Earlier RAD-DINO numbers
on the MedRPG single-box eval set were reaggregated from a model trained on the
larger row-level p10-p19 split. This script trains the same lightweight frozen
RAD-DINO patch head on the exact single-box train/val/eval manifests used for
the fair MedRPG retrain.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from models_ms_cxr_vfm_localizer import BBoxMLP, PatchHeatmapBBoxHead, bbox_loss


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]

EXP_NAME = "ms_cxr_rad_dino_singlebox_retrain_v1"
MEDRPG_DATA = PROJECT_ROOT / "experiments" / "ms_cxr_medrpg_baseline_v1" / "data"
MEDRPG_FINAL = PROJECT_ROOT / "experiments" / "ms_cxr_medrpg_fair_retrain_final_v1"
STAGE1_DATA = PROJECT_ROOT / "training" / "ms_cxr_vfm_localizer_stage1_p10_p19" / "datasets"
STAGE1_FEATURES = PROJECT_ROOT / "features" / "ms_cxr_vfm_localizer_stage1_p10_p19"

EXP = PROJECT_ROOT / "experiments" / EXP_NAME
PRED = EXP / "predictions"
MET = EXP / "metrics"
CFG = EXP / "configs"
LOGS = EXP / "logs"
TRAIN = PROJECT_ROOT / "training" / EXP_NAME
RUNS = TRAIN / "runs"
CKPT = TRAIN / "checkpoints"
REPORT = PROJECT_ROOT / "reports" / EXP_NAME

ALL_FINDINGS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Lung Opacity",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
]
MAIN5 = ["Atelectasis", "Consolidation", "Lung Opacity", "Pleural Effusion", "Pneumothorax"]

VARIANT_DIRS = {
    "full_phrase": MEDRPG_DATA / "medrpg_our_p10p19_single_box_full_phrase",
    "label_only": MEDRPG_DATA / "medrpg_our_p10p19_single_box_label_only",
    "rule_context": MEDRPG_DATA / "medrpg_our_p10p19_single_box_rule_context",
}


def ensure_dirs() -> None:
    for path in [EXP, PRED, MET, CFG, LOGS, TRAIN, RUNS, CKPT, REPORT]:
        path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def valid_box(box: Optional[Sequence[float]]) -> bool:
    return bool(box) and len(box) == 4 and float(box[2]) > float(box[0]) and float(box[3]) > float(box[1])


def clip_box(box: Sequence[float], iw: float, ih: float) -> List[float]:
    x1, y1, x2, y2 = [float(x) for x in box]
    x1 = max(0.0, min(iw, x1))
    x2 = max(0.0, min(iw, x2))
    y1 = max(0.0, min(ih, y1))
    y2 = max(0.0, min(ih, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def norm_cxcywh_to_xyxy(box: Sequence[float], iw: float, ih: float) -> List[float]:
    cx, cy, w, h = [float(x) for x in box]
    return clip_box([(cx - w / 2) * iw, (cy - h / 2) * ih, (cx + w / 2) * iw, (cy + h / 2) * ih], iw, ih)


def xyxy_to_norm_cxcywh(row: pd.Series) -> List[float]:
    iw = float(row["image_width"])
    ih = float(row["image_height"])
    x1, y1, x2, y2 = [float(row[k]) for k in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]]
    return [
        ((x1 + x2) / 2.0) / iw,
        ((y1 + y2) / 2.0) / ih,
        max(1e-6, (x2 - x1) / iw),
        max(1e-6, (y2 - y1) / ih),
    ]


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
    return float(inter / union) if union > 0 else 0.0


def metrics_from_predictions(preds: pd.DataFrame, method: str, subset: str = "all8") -> Dict:
    if subset == "main5":
        df = preds[preds["finding_label"].isin(MAIN5)].copy()
    else:
        df = preds.copy()
    if len(df) == 0:
        return {"method": method, "subset": subset, "n": 0}
    ious = df["iou"].astype(float).to_numpy()
    return {
        "method": method,
        "subset": subset,
        "n": int(len(df)),
        "mean_iou": float(np.mean(ious)),
        "median_iou": float(np.median(ious)),
        "Hit@0.1": float(np.mean(ious >= 0.1)),
        "Hit@0.3": float(np.mean(ious >= 0.3)),
        "Hit@0.5": float(np.mean(ious >= 0.5)),
        "bbox_missing_rate": float(df["bbox_missing"].astype(bool).mean()) if "bbox_missing" in df else 0.0,
        "bbox_invalid_rate": float(df["bbox_invalid"].astype(bool).mean()) if "bbox_invalid" in df else 0.0,
    }


def parse_location(text: str) -> Dict[str, str]:
    t = f" {str(text).lower()} "
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
    elif "upper" in t:
        vertical = "upper"
    elif "middle" in t or re.search(r"\bmid\b", t):
        vertical = "mid"
    elif "lower" in t or "inferior" in t:
        vertical = "lower"
    elif "basal" in t or "base" in t or "bibasilar" in t:
        vertical = "basal"
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
    uncertainty = "possible" if any(w in t for w in ["possible", "possibly", "may ", "could", "likely"]) else "definite"
    return {"laterality": laterality, "vertical": vertical, "severity": severity, "uncertainty": uncertainty}


def one_hot(value: str, choices: Sequence[str]) -> List[float]:
    return [1.0 if str(value) == c else 0.0 for c in choices]


def token_hash_features(text: str, dim: int = 64) -> np.ndarray:
    vec = np.zeros(dim, dtype="float32")
    for token in re.findall(r"[a-z0-9]+", str(text).lower()):
        # Stable small hash without relying on Python's randomized hash().
        h = 2166136261
        for ch in token:
            h = (h ^ ord(ch)) * 16777619
            h &= 0xFFFFFFFF
        vec[h % dim] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def encode_query(df: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "none":
        return np.zeros((len(df), 0), dtype="float32")
    laterality_choices = ["left", "right", "bilateral", "unknown"]
    vertical_choices = ["apical", "upper", "mid", "lower", "basal", "whole", "unknown"]
    severity_choices = ["trace", "small", "mild", "moderate", "large", "unknown"]
    uncertainty_choices = ["definite", "possible"]
    feats = []
    for _, row in df.iterrows():
        text = row.get("phrase_text", "")
        loc = parse_location(text)
        vals: List[float] = []
        vals += one_hot(row.get("finding_label", ""), ALL_FINDINGS)
        vals += one_hot(loc["laterality"], laterality_choices)
        vals += one_hot(loc["vertical"], vertical_choices)
        vals += one_hot(loc["severity"], severity_choices)
        vals += one_hot(loc["uncertainty"], uncertainty_choices)
        vals += token_hash_features(text, 64).tolist()
        feats.append(vals)
    return np.asarray(feats, dtype="float32")


def load_stage_mapping(split: str) -> Dict[str, str]:
    rows = read_jsonl(STAGE1_DATA / f"{split}.jsonl")
    mapping = {}
    for row in rows:
        ann = str(row.get("ms_cxr_annotation_id", "")).lower()
        if ann:
            mapping[ann] = row["task_id"]
    return mapping


def load_feature_npz(split: str) -> Dict:
    path = STAGE1_FEATURES / f"vfm_features_{split}.npz"
    z = np.load(path, allow_pickle=True)
    return {
        "task_ids": [str(x) for x in z["task_ids"]],
        "patch_tokens": z["patch_tokens"],
    }


def load_variant_rows(query_variant: str, split: str) -> pd.DataFrame:
    path = VARIANT_DIRS[query_variant] / f"{split}.csv"
    df = pd.read_csv(path)
    df["source_annotation_id_norm"] = df["source_annotation_id"].astype(str).str.lower()
    df["target_norm"] = df.apply(xyxy_to_norm_cxcywh, axis=1)
    return df


def align_split(query_variant: str, split: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    df = load_variant_rows(query_variant, split)
    ann_to_task = load_stage_mapping(split)
    feat = load_feature_npz(split)
    pos = {tid: i for i, tid in enumerate(feat["task_ids"])}
    idxs = []
    keep = []
    missing = []
    for i, row in df.iterrows():
        task_id = ann_to_task.get(row["source_annotation_id_norm"])
        if task_id is not None and task_id in pos:
            idxs.append(pos[task_id])
            keep.append(i)
        else:
            missing.append(row["sample_id"])
    if missing:
        raise RuntimeError(f"Missing {len(missing)} feature rows for {query_variant}/{split}; first={missing[:3]}")
    aligned = df.loc[keep].reset_index(drop=True)
    tokens = feat["patch_tokens"][idxs].astype("float32")
    ctx = encode_query(aligned, query_variant).astype("float32")
    y = np.asarray(aligned["target_norm"].tolist(), dtype="float32")
    return aligned, tokens, ctx, y


def center_indices(y: np.ndarray, token_count: int, device: str) -> torch.Tensor:
    grid = int(round(math.sqrt(token_count)))
    cx = np.clip((y[:, 0] * grid).astype(int), 0, grid - 1)
    cy = np.clip((y[:, 1] * grid).astype(int), 0, grid - 1)
    return torch.tensor(np.minimum(cy * grid + cx, token_count - 1), dtype=torch.long, device=device)


def val_score(pred_norm: np.ndarray, rows: pd.DataFrame) -> float:
    scores = []
    for pred, (_, row) in zip(pred_norm, rows.iterrows()):
        pxy = norm_cxcywh_to_xyxy(pred, float(row["image_width"]), float(row["image_height"]))
        gxy = [float(row[k]) for k in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]]
        scores.append(iou_xyxy(pxy, gxy))
    return float(np.mean(scores)) if scores else 0.0


def predict_heatmap(model: PatchHeatmapBBoxHead, tokens: np.ndarray, ctx: np.ndarray, rows: pd.DataFrame, method: str) -> pd.DataFrame:
    device = next(model.parameters()).device
    model.eval()
    batches = []
    with torch.no_grad():
        for start in range(0, len(tokens), 16):
            tok = torch.tensor(tokens[start : start + 16], dtype=torch.float32, device=device)
            c = torch.tensor(ctx[start : start + 16], dtype=torch.float32, device=device)
            pred, _ = model(tok, c)
            batches.append(pred.detach().cpu().numpy())
    pred_norm = np.vstack(batches)
    out = []
    for pred, (_, row) in zip(pred_norm, rows.iterrows()):
        pred = np.asarray(pred, dtype="float32")
        pred[:2] = np.clip(pred[:2], 0.0, 1.0)
        pred[2:] = np.clip(pred[2:], 0.02, 1.0)
        pxy = norm_cxcywh_to_xyxy(pred, float(row["image_width"]), float(row["image_height"]))
        gxy = [float(row[k]) for k in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]]
        out.append(
            {
                "sample_id": row["sample_id"],
                "subject_id": row["subject_id"],
                "study_id": row["study_id"],
                "dicom_id": row["dicom_id"],
                "image_path": row["image_path"],
                "finding_label": row["finding_label"],
                "query_text": row.get("phrase_text", ""),
                "method": method,
                "gt_x1": gxy[0],
                "gt_y1": gxy[1],
                "gt_x2": gxy[2],
                "gt_y2": gxy[3],
                "pred_x1": pxy[0],
                "pred_y1": pxy[1],
                "pred_x2": pxy[2],
                "pred_y2": pxy[3],
                "pred_cx": float(pred[0]),
                "pred_cy": float(pred[1]),
                "pred_w": float(pred[2]),
                "pred_h": float(pred[3]),
                "iou": iou_xyxy(pxy, gxy),
                "hit_0_1": iou_xyxy(pxy, gxy) >= 0.1,
                "hit_0_3": iou_xyxy(pxy, gxy) >= 0.3,
                "hit_0_5": iou_xyxy(pxy, gxy) >= 0.5,
                "bbox_missing": False,
                "bbox_invalid": not valid_box(pxy),
            }
        )
    return pd.DataFrame(out)


def train_heatmap(query_variant: str, args) -> Dict:
    method = f"rad_dino_{query_variant}_singlebox"
    tr_rows, tok_tr, ctx_tr, y_tr = align_split(query_variant, "train")
    va_rows, tok_va, ctx_va, y_va = align_split(query_variant, "val")
    ev_rows, tok_ev, ctx_ev, _ = align_split(query_variant, "eval")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PatchHeatmapBBoxHead(tok_tr.shape[-1], ctx_tr.shape[1], hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    tok_tr_t = torch.tensor(tok_tr, dtype=torch.float32, device=device)
    ctx_tr_t = torch.tensor(ctx_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
    tok_va_t = torch.tensor(tok_va, dtype=torch.float32, device=device)
    ctx_va_t = torch.tensor(ctx_va, dtype=torch.float32, device=device)
    c_tr = center_indices(y_tr, tok_tr.shape[1], device)

    best_state = None
    best_val = -1.0
    best_epoch = 0
    wait = 0
    hist = []
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(tok_tr_t), device=device)
        losses = []
        for start in range(0, len(tok_tr_t), args.batch_size):
            idx = perm[start : start + args.batch_size]
            pred, logits = model(tok_tr_t[idx], ctx_tr_t[idx])
            b_loss, parts = bbox_loss(pred, y_tr_t[idx])
            c_loss = torch.nn.functional.cross_entropy(logits, c_tr[idx])
            loss = b_loss + args.center_loss_weight * c_loss
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
            best_val = score
            best_epoch = epoch
            wait = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    pred_df = predict_heatmap(model, tok_ev, ctx_ev, ev_rows, method)
    pred_df.to_csv(PRED / f"{method}_eval_predictions.csv", index=False)
    run_dir = RUNS / method
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(hist).to_csv(run_dir / "loss_curve.csv", index=False)
    torch.save(
        {
            "state": best_state,
            "query_variant": query_variant,
            "ctx_dim": int(ctx_tr.shape[1]),
            "token_dim": int(tok_tr.shape[-1]),
            "best_val_mean_iou": best_val,
            "best_epoch": best_epoch,
        },
        CKPT / f"{method}.pt",
    )
    summary = metrics_from_predictions(pred_df, method, "all8")
    summary_main5 = metrics_from_predictions(pred_df, method, "main5")
    return {
        **summary,
        "main5_mean_iou": summary_main5.get("mean_iou"),
        "main5_Hit@0.3": summary_main5.get("Hit@0.3"),
        "query_variant": query_variant,
        "train_rows": int(len(tr_rows)),
        "val_rows": int(len(va_rows)),
        "eval_rows": int(len(ev_rows)),
        "ctx_dim": int(ctx_tr.shape[1]),
        "token_dim": int(tok_tr.shape[-1]),
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "best_val_mean_iou": float(best_val),
        "best_epoch": int(best_epoch),
        "train_time_sec": float(time.time() - start_time),
    }


def train_context_only(args) -> Dict:
    method = "context_only_prior_singlebox"
    tr_rows, _, x_tr, y_tr = align_split("rule_context", "train")
    va_rows, _, x_va, _ = align_split("rule_context", "val")
    ev_rows, _, x_ev, _ = align_split("rule_context", "eval")
    mean = x_tr.mean(axis=0, keepdims=True)
    std = x_tr.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x_tr = (x_tr - mean) / std
    x_va = (x_va - mean) / std
    x_ev = (x_ev - mean) / std
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BBoxMLP(x_tr.shape[1], hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    xt = torch.tensor(x_tr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=device)
    xv = torch.tensor(x_va, dtype=torch.float32, device=device)
    best_state = None
    best_val = -1.0
    best_epoch = 0
    wait = 0
    hist = []
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(xt), device=device)
        losses = []
        for start in range(0, len(xt), args.batch_size):
            idx = perm[start : start + args.batch_size]
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
            best_val = score
            best_epoch = epoch
            wait = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        pred_norm = model(torch.tensor(x_ev, dtype=torch.float32, device=device)).detach().cpu().numpy()
    out = []
    for pred, (_, row) in zip(pred_norm, ev_rows.iterrows()):
        pred = np.asarray(pred, dtype="float32")
        pxy = norm_cxcywh_to_xyxy(pred, float(row["image_width"]), float(row["image_height"]))
        gxy = [float(row[k]) for k in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]]
        out.append(
            {
                "sample_id": row["sample_id"],
                "subject_id": row["subject_id"],
                "study_id": row["study_id"],
                "dicom_id": row["dicom_id"],
                "image_path": row["image_path"],
                "finding_label": row["finding_label"],
                "query_text": row.get("phrase_text", ""),
                "method": method,
                "gt_x1": gxy[0],
                "gt_y1": gxy[1],
                "gt_x2": gxy[2],
                "gt_y2": gxy[3],
                "pred_x1": pxy[0],
                "pred_y1": pxy[1],
                "pred_x2": pxy[2],
                "pred_y2": pxy[3],
                "iou": iou_xyxy(pxy, gxy),
                "hit_0_1": iou_xyxy(pxy, gxy) >= 0.1,
                "hit_0_3": iou_xyxy(pxy, gxy) >= 0.3,
                "hit_0_5": iou_xyxy(pxy, gxy) >= 0.5,
                "bbox_missing": False,
                "bbox_invalid": not valid_box(pxy),
            }
        )
    pred_df = pd.DataFrame(out)
    pred_df.to_csv(PRED / f"{method}_eval_predictions.csv", index=False)
    run_dir = RUNS / method
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(hist).to_csv(run_dir / "loss_curve.csv", index=False)
    torch.save({"state": best_state, "mean": mean, "std": std, "ctx_dim": int(x_tr.shape[1])}, CKPT / f"{method}.pt")
    summary = metrics_from_predictions(pred_df, method, "all8")
    summary_main5 = metrics_from_predictions(pred_df, method, "main5")
    return {
        **summary,
        "main5_mean_iou": summary_main5.get("mean_iou"),
        "main5_Hit@0.3": summary_main5.get("Hit@0.3"),
        "query_variant": "rule_context",
        "train_rows": int(len(tr_rows)),
        "val_rows": int(len(va_rows)),
        "eval_rows": int(len(ev_rows)),
        "ctx_dim": int(x_tr.shape[1]),
        "token_dim": 0,
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "best_val_mean_iou": float(best_val),
        "best_epoch": int(best_epoch),
        "train_time_sec": float(time.time() - start_time),
    }


def load_medrpg_summary() -> pd.DataFrame:
    path = MEDRPG_FINAL / "metrics" / "medrpg_query_ablation_summary.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_historical_summary() -> pd.DataFrame:
    path = MEDRPG_FINAL / "metrics" / "historical_reaggregated_comparison.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, name: str, reps: int = 1000, seed: int = 42) -> Dict:
    merged = a[["sample_id", "iou", "hit_0_3", "hit_0_5"]].merge(
        b[["sample_id", "iou", "hit_0_3", "hit_0_5"]], on="sample_id", suffixes=("_a", "_b")
    )
    rng = np.random.default_rng(seed)
    n = len(merged)
    diffs_iou, diffs_h03, diffs_h05 = [], [], []
    arr = merged.to_numpy()
    cols = {c: i for i, c in enumerate(merged.columns)}
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        sample = arr[idx]
        diffs_iou.append(float(sample[:, cols["iou_a"]].astype(float).mean() - sample[:, cols["iou_b"]].astype(float).mean()))
        diffs_h03.append(float(sample[:, cols["hit_0_3_a"]].astype(float).mean() - sample[:, cols["hit_0_3_b"]].astype(float).mean()))
        diffs_h05.append(float(sample[:, cols["hit_0_5_a"]].astype(float).mean() - sample[:, cols["hit_0_5_b"]].astype(float).mean()))
    def ci(vals):
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
    return {
        "comparison": name,
        "n": int(n),
        "mean_iou_diff": float(np.mean(merged["iou_a"] - merged["iou_b"])),
        "mean_iou_ci_low": ci(diffs_iou)[0],
        "mean_iou_ci_high": ci(diffs_iou)[1],
        "hit_0_3_diff": float(np.mean(merged["hit_0_3_a"].astype(float) - merged["hit_0_3_b"].astype(float))),
        "hit_0_3_ci_low": ci(diffs_h03)[0],
        "hit_0_3_ci_high": ci(diffs_h03)[1],
        "hit_0_5_diff": float(np.mean(merged["hit_0_5_a"].astype(float) - merged["hit_0_5_b"].astype(float))),
        "hit_0_5_ci_low": ci(diffs_h05)[0],
        "hit_0_5_ci_high": ci(diffs_h05)[1],
    }


def build_reports(summary: pd.DataFrame, bootstrap: pd.DataFrame) -> None:
    medrpg = load_medrpg_summary()
    hist = load_historical_summary()
    report = [
        "# MS-CXR RAD-DINO Single-Box Retrain V1",
        "",
        "## Purpose",
        "",
        "This run checks whether the earlier RAD-DINO same-eval result was unfairly low because it was reaggregated from a model trained on a larger row-level MS-CXR split rather than the MedRPG single-box train split.",
        "",
        "## Data",
        "",
        "- Dataset: MS-CXR p10-p19 single-box phrase-level subset.",
        "- Train/val/eval rows: 638 / 87 / 163.",
        "- Bboxes: MS-CXR phrase-grounding boxes, not pixel-level lesion masks.",
        "- VFM: microsoft/rad-dino cached patch tokens from Stage 1.",
        "- Encoder: frozen. Trained module: shallow query-conditioned heatmap/bbox head.",
        "",
        "## New RAD-DINO Single-Box Results",
        "",
        summary.to_markdown(index=False),
        "",
    ]
    if len(medrpg):
        report += ["## MedRPG Fair Retrain Reference", "", medrpg.to_markdown(index=False), ""]
    if len(hist):
        report += ["## Historical RAD-DINO Reaggregated Reference", "", hist.to_markdown(index=False), ""]
    if len(bootstrap):
        report += ["## Paired Bootstrap", "", bootstrap.to_markdown(index=False), ""]
    report += [
        "## Interpretation Rules",
        "",
        "- If single-box retrained RAD-DINO is close to or above historical 0.464, the old number was not necessarily unfairly low.",
        "- If it rises substantially, the old comparison under-estimated our method because training data did not match the MedRPG single-box setting.",
        "- Released MedRPG checkpoint results remain reference-only because of official split overlap contamination.",
        "- Fair MedRPG retrain and RAD-DINO single-box retrain are the relevant controlled comparison.",
        "",
    ]
    write_text(REPORT / "README_KO.md", "\n".join(report) + "\n")


def run(args) -> Dict:
    ensure_dirs()
    set_seed(args.seed)
    configs = {
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "patience": args.patience,
        "hidden": args.hidden,
        "center_loss_weight": args.center_loss_weight,
        "stage1_feature_cache": str(STAGE1_FEATURES),
        "single_box_data": {k: str(v) for k, v in VARIANT_DIRS.items()},
    }
    write_text(CFG / "run_config.json", json.dumps(configs, indent=2))

    rows = []
    if args.context_only:
        rows.append(train_context_only(args))
    for variant in args.variants:
        rows.append(train_heatmap(variant, args))
    summary = pd.DataFrame(rows)
    summary.to_csv(MET / "ours_singlebox_retrain_summary.csv", index=False)
    summary[summary["subset"].eq("all8")].to_csv(MET / "summary_all8.csv", index=False)

    main5_rows = []
    for variant in args.variants:
        method = f"rad_dino_{variant}_singlebox"
        pred_path = PRED / f"{method}_eval_predictions.csv"
        if pred_path.exists():
            main5_rows.append(metrics_from_predictions(pd.read_csv(pred_path), method, "main5"))
    if (PRED / "context_only_prior_singlebox_eval_predictions.csv").exists():
        main5_rows.append(metrics_from_predictions(pd.read_csv(PRED / "context_only_prior_singlebox_eval_predictions.csv"), "context_only_prior_singlebox", "main5"))
    pd.DataFrame(main5_rows).to_csv(MET / "summary_main5.csv", index=False)

    boot_rows = []
    pred_label = PRED / "rad_dino_label_only_singlebox_eval_predictions.csv"
    pred_rule = PRED / "rad_dino_rule_context_singlebox_eval_predictions.csv"
    if pred_label.exists() and pred_rule.exists():
        boot_rows.append(paired_bootstrap(pd.read_csv(pred_rule), pd.read_csv(pred_label), "rad_dino_rule_context_singlebox - rad_dino_label_only_singlebox", args.bootstrap_reps, args.seed))
    med_pred_dir = MEDRPG_FINAL / "runs"
    med_rule = med_pred_dir / "rule_s42" / "eval_predictions.csv"
    med_label = med_pred_dir / "label_s42" / "eval_predictions.csv"
    med_full = med_pred_dir / "full_s42" / "eval_predictions.csv"
    if pred_rule.exists() and med_rule.exists():
        boot_rows.append(paired_bootstrap(pd.read_csv(pred_rule), pd.read_csv(med_rule), "rad_dino_rule_context_singlebox - medrpg_rule_context_s42", args.bootstrap_reps, args.seed))
    if pred_label.exists() and med_label.exists():
        boot_rows.append(paired_bootstrap(pd.read_csv(pred_label), pd.read_csv(med_label), "rad_dino_label_only_singlebox - medrpg_label_only_s42", args.bootstrap_reps, args.seed))
    if pred_rule.exists() and med_full.exists():
        boot_rows.append(paired_bootstrap(pd.read_csv(pred_rule), pd.read_csv(med_full), "rad_dino_rule_context_singlebox - medrpg_full_phrase_s42", args.bootstrap_reps, args.seed))
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(MET / "bootstrap_ci.csv", index=False)

    build_reports(summary, boot)
    return {
        "project_root": str(PROJECT_ROOT),
        "summary_path": str(MET / "ours_singlebox_retrain_summary.csv"),
        "report_path": str(REPORT / "README_KO.md"),
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=["label_only", "rule_context"])
    parser.add_argument("--context-only", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--center-loss-weight", type=float, default=0.05)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.epochs = min(args.epochs, 40)
        args.patience = min(args.patience, 10)
        args.bootstrap_reps = min(args.bootstrap_reps, 300)
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, indent=2))

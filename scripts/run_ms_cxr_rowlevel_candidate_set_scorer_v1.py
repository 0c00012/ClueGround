#!/usr/bin/env python
"""Row-level MS-CXR candidate-set scorer for the 1444 protocol.

This is the 1444-row counterpart of run_ms_cxr_candidate_set_scorer_v1.py.
It uses the subject-safe p10-p19 split train 998 / val 166 / eval 280.

The detector pool is deliberately restricted to models trained on the same
row-level split: YOLOv8n_full and YOLOv8s_full.  It does not reuse the
single-box 888 four-detector pool.

Eval gold is used only for final metric computation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_ms_cxr_rad_dino_transvg_context_head_v1 as xattn  # noqa: E402
from scripts import run_ms_cxr_yolo_detector_v1 as yd  # noqa: E402
from scripts import run_ms_cxr_yolo_rule_context_v1 as ybase  # noqa: E402
from scripts import run_ms_cxr_yolo_rule_context_v2 as yv2  # noqa: E402
from scripts import run_ms_cxr_yolo_dino_rule_fusion_v1 as fusion_ref  # noqa: E402


EXP_NAME = "ms_cxr_rowlevel_candidate_set_scorer_v1"
EXP = PROJECT_ROOT / "experiments" / EXP_NAME
PRED = EXP / "predictions"
MET = EXP / "metrics"
CFG = EXP / "configs"
REPORT = PROJECT_ROOT / "reports" / EXP_NAME
TRAIN = PROJECT_ROOT / "training" / EXP_NAME
RUNS = TRAIN / "runs"

YOLO_RUNS = PROJECT_ROOT / "training" / "ms_cxr_yolo_detector_v1" / "runs"
XATTN_CKPT = PROJECT_ROOT / "training" / "ms_cxr_rad_dino_transvg_context_head_v1" / "checkpoints"
XATTN_PRED = PROJECT_ROOT / "experiments" / "ms_cxr_rad_dino_transvg_context_head_v1" / "predictions"
MEDRPG_ROW = PROJECT_ROOT / "experiments" / "ms_cxr_medrpg_rowlevel_fair_retrain_v1"
FUSION_PRED = PROJECT_ROOT / "experiments" / "ms_cxr_yolo_dino_rule_fusion_v1" / "predictions"

SOURCE_MODELS = {
    "yolov8n": YOLO_RUNS / "yolov8n_full" / "weights" / "best.pt",
    "yolov8s": YOLO_RUNS / "yolov8s_full" / "weights" / "best.pt",
}
MAIN5 = set(yd.MAIN5)


def ensure_dirs() -> None:
    for path in [EXP, PRED, MET, CFG, REPORT, TRAIN, RUNS]:
        path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def source_rows(split: str) -> list[dict[str, Any]]:
    rows = yd.source_rows(split)
    out = []
    for row in rows:
        r = dict(row)
        r["sample_id"] = str(row["task_id"])
        out.append(r)
    return out


def norm_to_xyxy(box: Sequence[float], iw: float, ih: float) -> list[float]:
    return ybase.norm_to_xyxy(box, iw, ih)


def xyxy_to_norm(box: Sequence[float], iw: float, ih: float) -> np.ndarray:
    return ybase.xyxy_to_norm(box, iw, ih)


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    return ybase.iou_xyxy(a, b)


def iou_norm(a: Sequence[float], b: Sequence[float] | None) -> float:
    if b is None:
        return 0.0
    return ybase.iou_norm(a, b)


def sanitize_norm(box: Sequence[float]) -> np.ndarray:
    return ybase.sanitize_norm(box)


def blend_norm(a: Sequence[float], b: Sequence[float], weight_a: float) -> np.ndarray:
    return ybase.blend_norm(a, b, weight_a)


def xattn_prediction_path(split: str) -> Path:
    return XATTN_PRED / f"row1444_rule_plus_full_{split}_predictions.jsonl"


def ensure_xattn_predictions(splits: Iterable[str], batch_size: int) -> None:
    ckpt_path = XATTN_CKPT / "row1444_rule_plus_full_s42_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    missing = [split for split in splits if not xattn_prediction_path(split).exists()]
    if not missing:
        return
    ckpt = torch.load(ckpt_path, map_location=xattn.DEVICE)
    model = xattn.RadDinoContextCrossAttentionHead(
        token_dim=ckpt["token_dim"],
        query_dim=ckpt["query_dim"],
        hidden=256,
        num_query_tokens=4,
        num_heads=4,
        dropout=0.1,
    ).to(xattn.DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for split in missing:
        ds = xattn.MSContextDataset("row1444", "rule_plus_full", split)
        print(f"[xattn-row] predict {split} n={len(ds)}", flush=True)
        xattn.eval_model(model, ds, batch_size, xattn_prediction_path(split))


def load_xattn_map(split: str) -> dict[str, np.ndarray]:
    path = xattn_prediction_path(split)
    out: dict[str, np.ndarray] = {}
    for row in read_jsonl(path):
        box = np.asarray(row["pred_bbox_norm_cxcywh"], dtype="float32")
        out[str(row["sample_id"])] = box
        out[str(row["task_id"])] = box
    return out


def lookup_box(mapping: dict[str, np.ndarray], task_id: Any, sample_id: Any) -> np.ndarray | None:
    box = mapping.get(str(task_id))
    if box is not None:
        return box
    return mapping.get(str(sample_id))


def predict_candidates(weights: Path, rows: list[dict[str, Any]], split: str, tag: str, args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    out_csv = PRED / f"{tag}_{split}_conf{str(args.pred_conf).replace('.', 'p')}_candidates.csv"
    if out_csv.exists() and not args.force_predict:
        df = pd.read_csv(out_csv)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for _, r in df.iterrows():
            grouped[str(r["dicom_id"])].append(
                {
                    "class_id": int(r["class_id"]),
                    "score": float(r["score"]),
                    "box": [float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])],
                    "source_model": str(r["source_model"]),
                    "rank": int(r["rank"]),
                }
            )
        return grouped

    from ultralytics import YOLO

    image_by_dicom = {str(row["dicom_id"]): row["image_path"] for row in rows}
    items = sorted(image_by_dicom.items())
    model = YOLO(str(weights))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    out_rows: list[dict[str, Any]] = []
    chunk = max(1, int(args.predict_batch))
    for start in range(0, len(items), chunk):
        batch = items[start : start + chunk]
        results = model.predict(
            source=[p for _, p in batch],
            imgsz=args.imgsz,
            conf=args.pred_conf,
            device=args.device,
            verbose=False,
            stream=False,
            max_det=300,
            batch=chunk,
        )
        for (dicom_id, image_path), res in zip(batch, results):
            preds = []
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                cls = res.boxes.cls.cpu().numpy().astype(int)
                scores = res.boxes.conf.cpu().numpy()
                for box, c, s in zip(xyxy, cls, scores):
                    preds.append({"class_id": int(c), "score": float(s), "box": [float(x) for x in box]})
            preds.sort(key=lambda x: x["score"], reverse=True)
            rank_by_class: dict[int, int] = defaultdict(int)
            for pred in preds:
                cls = int(pred["class_id"])
                rank = rank_by_class[cls]
                rank_by_class[cls] += 1
                cand = {**pred, "source_model": tag, "rank": rank}
                grouped[dicom_id].append(cand)
                out_rows.append(
                    {
                        "dicom_id": dicom_id,
                        "image_path": image_path,
                        "class_id": cls,
                        "score": cand["score"],
                        "x1": cand["box"][0],
                        "y1": cand["box"][1],
                        "x2": cand["box"][2],
                        "y2": cand["box"][3],
                        "source_model": tag,
                        "rank": rank,
                    }
                )
    pd.DataFrame(out_rows).to_csv(out_csv, index=False)
    return grouped


def merge_candidates(*groups: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        for dicom, cands in group.items():
            out[dicom].extend(cands)
    return out


def candidate_groups(split: str, rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    groups = []
    for tag, weights in SOURCE_MODELS.items():
        if not weights.exists():
            raise FileNotFoundError(weights)
        print(f"[row-candidates] {split} {tag}", flush=True)
        groups.append(predict_candidates(weights, rows, split, tag, args))
    return merge_candidates(*groups)


def row_prior(row: dict[str, Any], priors: dict[str, dict]) -> tuple[dict[str, str], np.ndarray]:
    return yv2.target_prior_for_row(row, priors, {"side_mode": "radiology_right", "prior_train_weight": 0.75})


def class_one_hot(class_id: int) -> list[float]:
    return [1.0 if int(class_id) == i else 0.0 for i in range(len(yd.CLASS_NAMES))]


def source_one_hot(source: str) -> list[float]:
    return [1.0 if source == s else 0.0 for s in ["yolov8n", "yolov8s", "xattn_rule_plus", "prior"]]


def variant_one_hot(variant: str) -> list[float]:
    return [1.0 if variant == v else 0.0 for v in ["raw", "xattn085", "prior085", "xattn_prior", "xattn_only", "prior_only"]]


def context_one_hot(q: dict[str, str]) -> list[float]:
    lat = ["left", "right", "bilateral", "none", "unknown"]
    vert = ["apical", "upper", "mid", "lower", "basal", "whole", "unknown"]
    return [1.0 if q.get("laterality", "unknown") == x else 0.0 for x in lat] + [
        1.0 if q.get("vertical", "unknown") == x else 0.0 for x in vert
    ]


def box_features(box: Sequence[float]) -> list[float]:
    cx, cy, w, h = [float(x) for x in box]
    area = max(1e-6, w * h)
    aspect = w / max(1e-6, h)
    return [cx, cy, w, h, area, math.log(area), aspect, math.log(max(1e-6, aspect))]


def make_feature(row: pd.Series) -> list[float]:
    return (
        class_one_hot(int(row["class_id"]))
        + source_one_hot(str(row["source_model"]))
        + variant_one_hot(str(row["variant"]))
        + [float(row.get(x, 0.0)) for x in ["lat_left", "lat_right", "lat_bilateral", "lat_none", "lat_unknown"]]
        + [float(row.get(x, 0.0)) for x in ["v_apical", "v_upper", "v_mid", "v_lower", "v_basal", "v_whole", "v_unknown"]]
        + [
            float(row["confidence"]),
            math.log1p(20.0 * max(0.0, float(row["confidence"]))),
            float(row["rank_bonus"]),
            float(row["rank_norm"]),
            float(row["n_candidates_norm"]),
            float(row["prior_iou"]),
            float(row["xattn_iou"]),
            float(row["region_score"]),
            float(row["consensus_max"]),
            float(row["consensus_mean"]),
        ]
        + box_features([row["pred_cx"], row["pred_cy"], row["pred_w"], row["pred_h"]])
    )


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.asarray([make_feature(r) for _, r in df.iterrows()], dtype="float32")


def add_candidate_row(
    out: list[dict[str, Any]],
    row: dict[str, Any],
    q: dict[str, str],
    box_norm: Sequence[float],
    source: str,
    variant: str,
    conf: float,
    rank: int,
    n_cands: int,
    prior: np.ndarray,
    xbox: np.ndarray | None,
    raw_norms: list[np.ndarray],
) -> None:
    box = sanitize_norm(box_norm)
    iw, ih = float(row["image_width"]), float(row["image_height"])
    pred = norm_to_xyxy(box, iw, ih)
    gt = ybase.clip_box(row["gold_bbox_xyxy"], iw, ih)
    consensus = [iou_norm(box, b) for b in raw_norms] if raw_norms else [0.0]
    ctx = context_one_hot(q)
    rec = {
        "split": row["split"],
        "task_id": row["task_id"],
        "sample_id": row["sample_id"],
        "dicom_id": row["dicom_id"],
        "subject_id": row.get("subject_id", ""),
        "study_id": row.get("study_id", ""),
        "image_path": row["image_path"],
        "finding": row["finding"],
        "class_id": yd.CLASS_TO_ID[row["finding"]],
        "claim_sentence": row.get("claim_sentence", ""),
        "gt_x1": gt[0],
        "gt_y1": gt[1],
        "gt_x2": gt[2],
        "gt_y2": gt[3],
        "pred_x1": pred[0],
        "pred_y1": pred[1],
        "pred_x2": pred[2],
        "pred_y2": pred[3],
        "pred_cx": float(box[0]),
        "pred_cy": float(box[1]),
        "pred_w": float(box[2]),
        "pred_h": float(box[3]),
        "image_width": iw,
        "image_height": ih,
        "confidence": float(conf),
        "source_model": source,
        "candidate_rank": int(rank),
        "rank_bonus": 1.0 / (1.0 + rank) if rank < 999 else 0.0,
        "rank_norm": min(rank, 999) / 999.0,
        "n_candidates": int(n_cands),
        "n_candidates_norm": min(n_cands, 100) / 100.0,
        "variant": variant,
        "prior_iou": iou_norm(box, prior),
        "xattn_iou": iou_norm(box, xbox),
        "region_score": yv2.center_region_score_v2(box, q, "radiology_right"),
        "consensus_max": float(np.max(consensus)),
        "consensus_mean": float(np.mean(consensus)),
        "target_iou": iou_xyxy(pred, gt),
        "candidate_id": f"{row['task_id']}::{source}::{variant}::{rank}",
    }
    for name, value in zip(["lat_left", "lat_right", "lat_bilateral", "lat_none", "lat_unknown"], ctx[:5]):
        rec[name] = value
    for name, value in zip(["v_apical", "v_upper", "v_mid", "v_lower", "v_basal", "v_whole", "v_unknown"], ctx[5:]):
        rec[name] = value
    out.append(rec)


def build_candidate_table(split: str, rows: list[dict[str, Any]], candidates: dict[str, list[dict[str, Any]]], priors: dict[str, dict], args: argparse.Namespace) -> pd.DataFrame:
    out_pkl = PRED / f"{split}_row_candidates.pkl"
    out_csv = PRED / f"{split}_row_candidates_light.csv"
    if out_pkl.exists() and not args.rebuild_tables:
        return pd.read_pickle(out_pkl)
    xmap = load_xattn_map(split)
    out: list[dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        row["split"] = split
        q, prior = row_prior(row, priors)
        xbox = lookup_box(xmap, row["task_id"], row["sample_id"])
        class_id = yd.CLASS_TO_ID[row["finding"]]
        iw, ih = float(row["image_width"]), float(row["image_height"])
        same = [
            c
            for c in candidates.get(str(row["dicom_id"]), [])
            if int(c["class_id"]) == class_id and int(c.get("rank", 9999)) < args.max_rank
        ]
        raw_norms = [xyxy_to_norm(c["box"], iw, ih) for c in same]
        n_cands = len(same)
        for cand, yolo_box in zip(same, raw_norms):
            rank = int(cand.get("rank", 999))
            conf = float(cand.get("score", 0.0))
            source = str(cand.get("source_model", ""))
            add_candidate_row(out, row, q, yolo_box, source, "raw", conf, rank, n_cands, prior, xbox, raw_norms)
            add_candidate_row(out, row, q, blend_norm(yolo_box, prior, 0.85), source, "prior085", conf, rank, n_cands, prior, xbox, raw_norms)
            if xbox is not None:
                add_candidate_row(out, row, q, blend_norm(yolo_box, xbox, 0.85), source, "xattn085", conf, rank, n_cands, prior, xbox, raw_norms)
                add_candidate_row(out, row, q, blend_norm(blend_norm(yolo_box, xbox, 0.85), prior, 0.9), source, "xattn_prior", conf, rank, n_cands, prior, xbox, raw_norms)
        add_candidate_row(out, row, q, prior, "prior", "prior_only", 0.0, 999, n_cands, prior, xbox, raw_norms)
        if xbox is not None:
            add_candidate_row(out, row, q, xbox, "xattn_rule_plus", "xattn_only", 0.0, 999, n_cands, prior, xbox, raw_norms)
    df = pd.DataFrame(out)
    df.to_pickle(out_pkl)
    df.to_csv(out_csv, index=False)
    return df


def train_models(train_df: pd.DataFrame, val_df: pd.DataFrame, quick: bool) -> tuple[str, dict[str, object], pd.DataFrame]:
    x_train = feature_matrix(train_df)
    y_train = train_df["target_iou"].to_numpy(float)
    x_val = feature_matrix(val_df)
    models: dict[str, object] = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=2.0)),
        "hgb": HistGradientBoostingRegressor(max_iter=140 if quick else 260, learning_rate=0.04, max_leaf_nodes=31, l2_regularization=0.04, random_state=42),
        "rf": RandomForestRegressor(n_estimators=140 if quick else 280, max_depth=13, min_samples_leaf=3, n_jobs=-1, random_state=42),
        "extra": ExtraTreesRegressor(n_estimators=180 if quick else 360, max_depth=None, min_samples_leaf=2, n_jobs=-1, random_state=42),
    }
    rows = []
    fitted = {}
    for name, model in models.items():
        print(f"[row-scorer] train {name}", flush=True)
        model.fit(x_train, y_train)
        fitted[name] = model
        score = np.clip(model.predict(x_val), 0.0, 1.0)
        tmp = val_df.copy()
        tmp["score_head"] = score
        pred = choose_by_score(tmp, "score_head", f"{name}_selector_val")
        row = metrics(pred, f"{name}_selector", "all8")
        row["val_rmse"] = float(mean_squared_error(val_df["target_iou"].to_numpy(float), score) ** 0.5)
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("mean_iou", ascending=False)
    best = str(summary.iloc[0]["method"]).replace("_selector", "")
    return best, fitted, summary


def apply_model(df: pd.DataFrame, model: object) -> pd.DataFrame:
    out = df.copy()
    out["score_head"] = np.clip(model.predict(feature_matrix(df)), 0.0, 1.0)
    return out


def prediction_from_candidates(chosen: pd.DataFrame, method: str) -> pd.DataFrame:
    out = chosen.copy()
    out["iou"] = out["target_iou"].astype(float)
    out["hit_0_1"] = out["iou"] >= 0.1
    out["hit_0_3"] = out["iou"] >= 0.3
    out["hit_0_5"] = out["iou"] >= 0.5
    out["bbox_missing"] = False
    out["bbox_invalid"] = (out["pred_x2"] <= out["pred_x1"]) | (out["pred_y2"] <= out["pred_y1"])
    out["method"] = method
    return out


def choose_by_score(df: pd.DataFrame, score_col: str, method: str) -> pd.DataFrame:
    idx = df.groupby("task_id")[score_col].idxmax()
    return prediction_from_candidates(df.loc[idx].copy(), method)


def weighted_box_fusion(scored_df: pd.DataFrame, top_k: int, temp: float, method: str) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for _, sub in scored_df.groupby("task_id"):
        top = sub.sort_values("score_head", ascending=False).head(top_k).copy()
        scores = top["score_head"].to_numpy(float)
        if temp <= 0:
            weights = np.zeros_like(scores)
            weights[0] = 1.0
        else:
            z = (scores - scores.max()) / temp
            weights = np.exp(z)
            weights = weights / max(1e-8, weights.sum())
        boxes = top[["pred_cx", "pred_cy", "pred_w", "pred_h"]].to_numpy(float)
        fused = sanitize_norm((boxes * weights[:, None]).sum(axis=0))
        r = top.iloc[0].copy()
        pred = norm_to_xyxy(fused, float(r["image_width"]), float(r["image_height"]))
        gt = [float(r["gt_x1"]), float(r["gt_y1"]), float(r["gt_x2"]), float(r["gt_y2"])]
        r["pred_cx"], r["pred_cy"], r["pred_w"], r["pred_h"] = fused.tolist()
        r["pred_x1"], r["pred_y1"], r["pred_x2"], r["pred_y2"] = pred
        r["target_iou"] = iou_xyxy(pred, gt)
        r["variant"] = f"wbf_top{top_k}_t{temp}"
        rows.append(r)
    return prediction_from_candidates(pd.DataFrame(rows), method)


def metrics(df: pd.DataFrame, method: str, subset: str) -> dict[str, Any]:
    sub = df.copy()
    if subset == "main5":
        sub = sub[sub["finding"].isin(MAIN5)]
    ious = sub["iou"].astype(float).to_numpy()
    return {
        "method": method,
        "subset": subset,
        "n": int(len(sub)),
        "mean_iou": float(ious.mean()) if len(ious) else 0.0,
        "median_iou": float(np.median(ious)) if len(ious) else 0.0,
        "Hit@0.1": float((ious >= 0.1).mean()) if len(ious) else 0.0,
        "Hit@0.3": float((ious >= 0.3).mean()) if len(ious) else 0.0,
        "Hit@0.5": float((ious >= 0.5).mean()) if len(ious) else 0.0,
        "bbox_missing_rate": float(df.get("bbox_missing", pd.Series([False] * len(df))).astype(bool).mean()) if len(df) else 0.0,
        "bbox_invalid_rate": float(df.get("bbox_invalid", pd.Series([False] * len(df))).astype(bool).mean()) if len(df) else 0.0,
    }


def reference_predictions() -> dict[str, pd.DataFrame]:
    refs = {}
    paths = {
        "yolo_dino_rule_fusion_v1": FUSION_PRED / "yolo_dino_rule_fusion_eval_predictions.csv",
        "rad_dino_rule_context": FUSION_PRED / "rad_dino_rule_context_eval_predictions.csv",
        "medrpg_rowlevel_fair_s42": MEDRPG_ROW / "runs" / "row_full_s42" / "eval_predictions.csv",
        "medrpg_rowlevel_fair_s13": MEDRPG_ROW / "runs" / "row_full_s13" / "eval_predictions.csv",
        "medrpg_rowlevel_fair_s2026": MEDRPG_ROW / "runs" / "row_full_s2026" / "eval_predictions.csv",
    }
    for name, path in paths.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "finding" not in df.columns:
            # MedRPG row predictions usually use sample_id for the task id.
            rows = pd.DataFrame(source_rows("eval"))[["task_id", "sample_id", "finding"]]
            if "task_id" in df.columns:
                df = df.merge(rows[["task_id", "finding"]], on="task_id", how="left")
            elif "sample_id" in df.columns:
                df = df.merge(rows[["sample_id", "finding"]], on="sample_id", how="left")
        refs[name] = df
    return refs


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, label: str, reps: int, seed: int) -> dict[str, Any]:
    key = "task_id" if "task_id" in a.columns and "task_id" in b.columns else "sample_id"
    cols = [key, "iou", "hit_0_3", "hit_0_5"]
    if "subject_id" in a.columns:
        cols.append("subject_id")
    aa = a[cols].rename(columns={"iou": "iou_a", "hit_0_3": "h03_a", "hit_0_5": "h05_a"})
    bb = b[[key, "iou", "hit_0_3", "hit_0_5"]].rename(columns={"iou": "iou_b", "hit_0_3": "h03_b", "hit_0_5": "h05_b"})
    m = aa.merge(bb, on=key, how="inner")
    if "subject_id" in m.columns:
        units = m["subject_id"].astype(str).to_numpy()
    else:
        units = m[key].astype(str).to_numpy()
    uniq = np.unique(units)
    rng = np.random.default_rng(seed)
    diffs_iou: list[float] = []
    diffs_h03: list[float] = []
    diffs_h05: list[float] = []
    for _ in range(reps):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        mask = np.isin(units, sampled)
        sub = m[mask]
        if len(sub) == 0:
            continue
        diffs_iou.append(float((sub["iou_a"].astype(float) - sub["iou_b"].astype(float)).mean()))
        diffs_h03.append(float((sub["h03_a"].astype(float) - sub["h03_b"].astype(float)).mean()))
        diffs_h05.append(float((sub["h05_a"].astype(float) - sub["h05_b"].astype(float)).mean()))
    def ci(vals: list[float]) -> tuple[float, float]:
        if not vals:
            return 0.0, 0.0
        return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
    iou_low, iou_high = ci(diffs_iou)
    h03_low, h03_high = ci(diffs_h03)
    h05_low, h05_high = ci(diffs_h05)
    mean_iou_diff = float((m["iou_a"].astype(float) - m["iou_b"].astype(float)).mean()) if len(m) else 0.0
    return {
        "comparison": label,
        "n": int(len(m)),
        "mean_iou_diff": mean_iou_diff,
        "mean_iou_ci_low": iou_low,
        "mean_iou_ci_high": iou_high,
        "mean_iou_prob_gt_0": float(np.mean(np.asarray(diffs_iou) > 0.0)) if diffs_iou else 0.0,
        "hit_0_3_diff": float((m["h03_a"].astype(float) - m["h03_b"].astype(float)).mean()) if len(m) else 0.0,
        "hit_0_3_ci_low": h03_low,
        "hit_0_3_ci_high": h03_high,
        "hit_0_5_diff": float((m["h05_a"].astype(float) - m["h05_b"].astype(float)).mean()) if len(m) else 0.0,
        "hit_0_5_ci_low": h05_low,
        "hit_0_5_ci_high": h05_high,
        "bootstrap_unit": "subject_id_cluster" if "subject_id" in m.columns else f"{key}_row",
        "bootstrap_reps": reps,
    }


def run(args: argparse.Namespace) -> None:
    ensure_dirs()
    write_text(CFG / "run_config.json", json.dumps(vars(args), indent=2, ensure_ascii=False))
    ensure_xattn_predictions(["train", "val", "eval"], args.batch_size)

    train_rows = source_rows("train")
    val_rows = source_rows("val")
    eval_rows = source_rows("eval")
    priors = ybase.make_train_priors(train_rows)

    train_c = candidate_groups("train", train_rows, args)
    val_c = candidate_groups("val", val_rows, args)
    eval_c = candidate_groups("eval", eval_rows, args)

    train_df = build_candidate_table("train", train_rows, train_c, priors, args)
    val_df = build_candidate_table("val", val_rows, val_c, priors, args)
    eval_df = build_candidate_table("eval", eval_rows, eval_c, priors, args)

    oracle_rows = []
    predictions: dict[str, pd.DataFrame] = {}
    for split, df in [("val", val_df), ("eval", eval_df)]:
        pred = choose_by_score(df, "target_iou", f"candidate_pool_oracle_{split}")
        predictions[f"candidate_pool_oracle_{split}"] = pred
        for subset in ["all8", "main5"]:
            oracle_rows.append(metrics(pred, f"candidate_pool_oracle_{split}", subset) | {"split": split})
    pd.DataFrame(oracle_rows).to_csv(MET / "candidate_pool_oracle_summary.csv", index=False)

    best_name, fitted, val_summary = train_models(train_df, val_df, args.quick)
    val_summary.to_csv(MET / "candidate_scorer_val_model_selection.csv", index=False)
    best_model = fitted[best_name]
    joblib.dump(best_model, RUNS / f"best_row_candidate_scorer_{best_name}.joblib")

    # Persist all three scored splits.  Downstream multibox training consumes
    # the train table too; falling back to the unscored light table silently
    # turns score_head into an all-missing feature for that split.
    train_scored = apply_model(train_df, best_model)
    val_scored = apply_model(val_df, best_model)
    eval_scored = apply_model(eval_df, best_model)
    train_scored.to_csv(PRED / "train_scored_candidates.csv", index=False)
    val_scored.to_csv(PRED / "val_scored_candidates.csv", index=False)
    eval_scored.to_csv(PRED / "eval_scored_candidates.csv", index=False)

    selector_eval = choose_by_score(eval_scored, "score_head", "row_candidate_set_scorer_selector")
    selector_eval.to_csv(PRED / "row_candidate_set_scorer_selector_eval_predictions.csv", index=False)
    predictions["row_candidate_set_scorer_selector"] = selector_eval

    wbf_rows = []
    for top_k in [2, 3, 5, 8]:
        for temp in [0.0, 0.03, 0.06, 0.1, 0.2, 0.5]:
            val_pred = weighted_box_fusion(val_scored, top_k, temp, f"row_candidate_wbf_k{top_k}_t{temp}")
            m = metrics(val_pred, f"row_candidate_wbf_k{top_k}_t{temp}", "all8")
            m["top_k"] = top_k
            m["temp"] = temp
            wbf_rows.append(m)
    wbf_grid = pd.DataFrame(wbf_rows).sort_values("mean_iou", ascending=False)
    wbf_grid.to_csv(MET / "candidate_wbf_val_grid.csv", index=False)
    best_wbf = wbf_grid.iloc[0]
    wbf_eval = weighted_box_fusion(eval_scored, int(best_wbf["top_k"]), float(best_wbf["temp"]), "row_candidate_weighted_box_fusion")
    wbf_eval.to_csv(PRED / "row_candidate_weighted_box_fusion_eval_predictions.csv", index=False)
    predictions["row_candidate_weighted_box_fusion"] = wbf_eval

    refs = reference_predictions()
    predictions.update(refs)
    predictions["candidate_pool_oracle"] = predictions["candidate_pool_oracle_eval"]

    summary_rows = []
    for name, pred in predictions.items():
        if name.endswith("_val"):
            continue
        for subset in ["all8", "main5"]:
            summary_rows.append(metrics(pred, name, subset))
    summary = pd.DataFrame(summary_rows).sort_values(["subset", "mean_iou"], ascending=[True, False])
    summary.to_csv(MET / "final_summary.csv", index=False)

    boot_rows = []
    for method in ["row_candidate_set_scorer_selector", "row_candidate_weighted_box_fusion"]:
        for other in ["yolo_dino_rule_fusion_v1", "rad_dino_rule_context", "medrpg_rowlevel_fair_s42"]:
            if method in predictions and other in predictions:
                boot_rows.append(paired_bootstrap(predictions[method], predictions[other], f"{method} - {other}", args.bootstrap_reps, args.seed))
    pd.DataFrame(boot_rows).to_csv(MET / "bootstrap_ci.csv", index=False)

    report = [
        "# MS-CXR Row-level Candidate-Set Scorer V1",
        "",
        "## 한 줄 결론",
        "",
        "p10-p19 row-level 1444 protocol에서 YOLOv8n/v8s full detector 후보와 RAD-DINO context-cross box를 후보별 feature로 묶어 scorer를 학습했다.",
        "",
        "## 공정성",
        "",
        "- Dataset: train 998 / val 166 / eval 280.",
        "- Detector pool: YOLOv8n_full + YOLOv8s_full only.",
        "- Candidate scorer training: train only.",
        "- Model/WBF selection: val only.",
        "- Eval gold: final metrics/bootstrap only.",
        "",
        f"- Best scorer: `{best_name}`.",
        f"- Best WBF: top_k={int(best_wbf['top_k'])}, temp={float(best_wbf['temp'])}.",
        "",
        "## Final Eval Summary",
        "",
        summary[summary["subset"].eq("all8")].to_markdown(index=False),
        "",
        "## Candidate Pool Oracle",
        "",
        pd.read_csv(MET / "candidate_pool_oracle_summary.csv").to_markdown(index=False),
        "",
        "## Bootstrap",
        "",
        pd.read_csv(MET / "bootstrap_ci.csv").to_markdown(index=False),
    ]
    write_text(REPORT / "README_KO.md", "\n".join(report) + "\n")

    print("project_root", PROJECT_ROOT)
    print("best_row_candidate_scorer", best_name)
    print("best_wbf_top_k", int(best_wbf["top_k"]))
    print("best_wbf_temp", float(best_wbf["temp"]))
    print("summary_path", MET / "final_summary.csv")
    print("report_path", REPORT / "README_KO.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--pred-conf", type=float, default=0.001)
    parser.add_argument("--predict-batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-rank", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force-predict", action="store_true")
    parser.add_argument("--rebuild-tables", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    t0 = time.time()
    run(parse_args())
    print("elapsed_sec", round(time.time() - t0, 2))

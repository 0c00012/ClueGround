#!/usr/bin/env python
"""YOLO + rule-context postprocessing for MS-CXR p10-p19 boxes.

This experiment keeps the trained YOLO detector fixed and asks whether the
same structured query idea used by the RAD-DINO head can improve YOLO outputs.

YOLO itself does not receive text. The rule-context is used after YOLO
prediction to rerank same-class candidate boxes and optionally blend them with
train-split location priors. MS-CXR boxes are phrase-grounding boxes, not pixel
lesion masks.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]

EXP_NAME = "ms_cxr_yolo_rule_context_v1"
EXP = PROJECT_ROOT / "experiments" / EXP_NAME
PRED = EXP / "predictions"
MET = EXP / "metrics"
OVER = EXP / "overlays"
CONTACT = EXP / "contact_sheets"
REPORT = PROJECT_ROOT / "reports" / EXP_NAME
CFG = EXP / "configs"

STAGE1_DATA = PROJECT_ROOT / "training" / "ms_cxr_vfm_localizer_stage1_p10_p19" / "datasets"
YOLO_RUNS = PROJECT_ROOT / "training" / "ms_cxr_yolo_detector_v1" / "runs"
YOLO_MET = PROJECT_ROOT / "experiments" / "ms_cxr_yolo_detector_v1" / "metrics"
RAD_DINO_MET = PROJECT_ROOT / "experiments" / "ms_cxr_rad_dino_detector_v1" / "metrics"
SINGLE_BOX = PROJECT_ROOT / "experiments" / "ms_cxr_medrpg_baseline_v1" / "data" / "medrpg_our_p10p19_single_box_full_phrase"

CLASS_NAMES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Lung Opacity",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
MAIN5 = {"Atelectasis", "Consolidation", "Lung Opacity", "Pleural Effusion", "Pneumothorax"}


def ensure_dirs() -> None:
    for p in [EXP, PRED, MET, OVER, CONTACT, REPORT, CFG]:
        p.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def source_rows(split: str) -> List[Dict]:
    path = STAGE1_DATA / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    return read_jsonl(path)


def valid_box(box: Optional[Sequence[float]]) -> bool:
    if box is None or len(box) != 4:
        return False
    x1, y1, x2, y2 = [float(x) for x in box]
    return x2 > x1 and y2 > y1


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


def iou_xyxy(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> float:
    if not valid_box(a) or not valid_box(b):
        return 0.0
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + bb - inter
    return float(inter / union) if union > 0 else 0.0


def xyxy_to_norm(box: Sequence[float], iw: float, ih: float) -> np.ndarray:
    x1, y1, x2, y2 = clip_box(box, iw, ih)
    return np.asarray(
        [
            ((x1 + x2) / 2.0) / iw,
            ((y1 + y2) / 2.0) / ih,
            max(1e-6, (x2 - x1) / iw),
            max(1e-6, (y2 - y1) / ih),
        ],
        dtype="float32",
    )


def norm_to_xyxy(box: Sequence[float], iw: float, ih: float) -> List[float]:
    cx, cy, w, h = [float(x) for x in box]
    return clip_box([(cx - w / 2) * iw, (cy - h / 2) * ih, (cx + w / 2) * iw, (cy + h / 2) * ih], iw, ih)


def sanitize_norm(box: Sequence[float]) -> np.ndarray:
    b = np.asarray(box, dtype="float32").copy()
    b[:2] = np.clip(b[:2], 0.0, 1.0)
    b[2:] = np.clip(b[2:], 0.02, 1.0)
    return b


def iou_norm(a: Sequence[float], b: Sequence[float]) -> float:
    return iou_xyxy(norm_to_xyxy(a, 1.0, 1.0), norm_to_xyxy(b, 1.0, 1.0))


def parse_rule_context(row: Dict) -> Dict[str, str]:
    text = " ".join(str(row.get(k, "")) for k in ["claim_sentence", "phrase", "sentence", "finding"]).lower()
    if "bilateral" in text or "both" in text or "bibasilar" in text:
        lat = "bilateral"
    elif re.search(r"\bright\b|\brt\b", text):
        lat = "right"
    elif re.search(r"\bleft\b|\blt\b", text):
        lat = "left"
    else:
        lat = "unknown"

    if re.search(r"\b(apical|apex|apices)\b", text):
        vert = "apical"
    elif "upper" in text or "suprahilar" in text:
        vert = "upper"
    elif "middle" in text or re.search(r"\bmid\b", text) or "perihilar" in text:
        vert = "mid"
    elif "lower" in text or "inferior" in text or "infrahilar" in text or "costophrenic" in text or "retrocardiac" in text:
        vert = "lower"
    elif "basal" in text or "base" in text or "bibasilar" in text:
        vert = "basal"
    elif "diffuse" in text or "throughout" in text or "bilateral" in text:
        vert = "whole"
    else:
        vert = "unknown"

    finding = str(row.get("finding", ""))
    return {"finding": finding, "laterality": lat, "vertical": vert}


def location_region(q: Dict[str, str]) -> Tuple[float, float, float, float]:
    # CXR display convention in these local images: patient right is usually image-left.
    lat = q.get("laterality", "unknown")
    vert = q.get("vertical", "unknown")
    finding = q.get("finding", "")
    if lat == "right":
        x1, x2 = 0.03, 0.53
    elif lat == "left":
        x1, x2 = 0.47, 0.97
    elif lat == "bilateral":
        x1, x2 = 0.06, 0.94
    else:
        x1, x2 = 0.05, 0.95

    if vert in {"apical", "upper"}:
        y1, y2 = 0.04, 0.42
    elif vert == "mid":
        y1, y2 = 0.22, 0.68
    elif vert in {"lower", "basal"}:
        y1, y2 = 0.42, 0.92
    elif vert == "whole":
        y1, y2 = 0.08, 0.92
    else:
        y1, y2 = 0.06, 0.92

    if finding == "Cardiomegaly":
        x1, x2, y1, y2 = 0.22, 0.80, 0.36, 0.88
    return x1, y1, x2, y2


def template_box(q: Dict[str, str]) -> np.ndarray:
    x1, y1, x2, y2 = location_region(q)
    finding = q.get("finding", "")
    if finding == "Pneumothorax":
        w, h = min(0.34, x2 - x1), min(0.28, y2 - y1)
    elif finding == "Pleural Effusion":
        w, h = min(0.42, x2 - x1), min(0.30, y2 - y1)
        y1 = max(y1, 0.52)
        y2 = max(y2, 0.90)
    elif finding == "Cardiomegaly":
        w, h = 0.54, 0.34
    elif finding in {"Lung Opacity", "Consolidation", "Pneumonia", "Atelectasis", "Edema"}:
        w, h = min(0.46, x2 - x1), min(0.40, y2 - y1)
    else:
        w, h = min(0.40, x2 - x1), min(0.36, y2 - y1)
    return sanitize_norm([(x1 + x2) / 2.0, (y1 + y2) / 2.0, w, h])


def center_region_score(box_norm: Sequence[float], q: Dict[str, str]) -> float:
    cx, cy = float(box_norm[0]), float(box_norm[1])
    x1, y1, x2, y2 = location_region(q)
    inside = 1.0 if x1 <= cx <= x2 and y1 <= cy <= y2 else 0.0
    tx = min(max(cx, x1), x2)
    ty = min(max(cy, y1), y2)
    dx = (cx - tx) / max(1e-6, x2 - x1)
    dy = (cy - ty) / max(1e-6, y2 - y1)
    dist_penalty = min(1.0, math.sqrt(dx * dx + dy * dy))
    return float(inside + (1.0 - dist_penalty))


def make_train_priors(train_rows: List[Dict]) -> Dict[str, Dict[Tuple, np.ndarray]]:
    buckets: Dict[Tuple[str, Tuple], List[np.ndarray]] = defaultdict(list)
    for row in train_rows:
        q = parse_rule_context(row)
        box = xyxy_to_norm(row["gold_bbox_xyxy"], row["image_width"], row["image_height"])
        finding, lat, vert = q["finding"], q["laterality"], q["vertical"]
        buckets[("global", ("all",))].append(box)
        buckets[("finding", (finding,))].append(box)
        buckets[("finding_lat", (finding, lat))].append(box)
        buckets[("finding_vert", (finding, vert))].append(box)
        buckets[("finding_lat_vert", (finding, lat, vert))].append(box)
    priors: Dict[str, Dict[Tuple, np.ndarray]] = defaultdict(dict)
    for (name, key), boxes in buckets.items():
        priors[name][key] = sanitize_norm(np.median(np.vstack(boxes), axis=0))
    return priors


def lookup_prior(priors: Dict[str, Dict[Tuple, np.ndarray]], q: Dict[str, str]) -> np.ndarray:
    finding, lat, vert = q["finding"], q["laterality"], q["vertical"]
    ordered = [
        ("finding_lat_vert", (finding, lat, vert)),
        ("finding_lat", (finding, lat)),
        ("finding_vert", (finding, vert)),
        ("finding", (finding,)),
        ("global", ("all",)),
    ]
    for name, key in ordered:
        if key in priors.get(name, {}):
            return priors[name][key]
    return sanitize_norm([0.5, 0.5, 0.5, 0.5])


def blend_norm(a: Sequence[float], b: Sequence[float], weight_a: float) -> np.ndarray:
    return sanitize_norm(np.asarray(a, dtype="float32") * weight_a + np.asarray(b, dtype="float32") * (1.0 - weight_a))


def predict_candidates(
    weights: Path,
    rows: List[Dict],
    imgsz: int,
    conf: float,
    device: str,
    tag: str,
    augment: bool = False,
) -> Dict[str, List[Dict]]:
    cache = PRED / f"{tag}_conf{str(conf).replace('.', 'p')}_all_candidates.csv"
    if cache.exists():
        df = pd.read_csv(cache)
        grouped: Dict[str, List[Dict]] = defaultdict(list)
        for _, r in df.iterrows():
            grouped[str(r["dicom_id"])].append(
                {
                    "class_id": int(r["class_id"]),
                    "score": float(r["score"]),
                    "box": [float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])],
                }
            )
        return grouped

    from ultralytics import YOLO

    image_by_dicom = {}
    for row in rows:
        image_by_dicom[str(row["dicom_id"])] = row["image_path"]
    items = sorted(image_by_dicom.items())
    model = YOLO(str(weights))
    out_rows = []
    grouped = defaultdict(list)
    chunk_size = 96
    for start in range(0, len(items), chunk_size):
        chunk = items[start : start + chunk_size]
        results = model.predict(
            source=[p for _, p in chunk],
            imgsz=imgsz,
            conf=conf,
            device=device,
            verbose=False,
            stream=False,
            max_det=300,
            augment=augment,
            batch=8,
        )
        for (dicom_id, image_path), res in zip(chunk, results):
            if res.boxes is None or len(res.boxes) == 0:
                continue
            xyxy = res.boxes.xyxy.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            scores = res.boxes.conf.cpu().numpy()
            for box, c, s in zip(xyxy, cls, scores):
                cand = {"class_id": int(c), "score": float(s), "box": [float(x) for x in box]}
                grouped[dicom_id].append(cand)
                out_rows.append(
                    {
                        "dicom_id": dicom_id,
                        "image_path": image_path,
                        "class_id": int(c),
                        "score": float(s),
                        "x1": cand["box"][0],
                        "y1": cand["box"][1],
                        "x2": cand["box"][2],
                        "y2": cand["box"][3],
                    }
                )
    pd.DataFrame(out_rows).to_csv(cache, index=False)
    return grouped


def choose_candidate(
    row: Dict,
    candidates_by_dicom: Dict[str, List[Dict]],
    priors: Dict[str, Dict[Tuple, np.ndarray]],
    params: Dict,
) -> Tuple[List[float], float, bool, Dict]:
    iw, ih = float(row["image_width"]), float(row["image_height"])
    q = parse_rule_context(row)
    prior = lookup_prior(priors, q)
    template = template_box(q)
    target_prior = blend_norm(prior, template, float(params.get("prior_train_weight", 0.75)))
    class_id = CLASS_TO_ID[row["finding"]]
    cands = [c for c in candidates_by_dicom.get(str(row["dicom_id"]), []) if int(c["class_id"]) == class_id and float(c["score"]) >= float(params["conf"])]
    if not cands:
        if params.get("fallback") == "prior":
            return norm_to_xyxy(target_prior, iw, ih), 0.0, False, {"source": "prior_fallback", **q}
        return [0.0, 0.0, 0.0, 0.0], 0.0, True, {"source": "missing", **q}

    scored = []
    for cand in cands:
        box_norm = xyxy_to_norm(cand["box"], iw, ih)
        conf_score = math.log(max(1e-6, float(cand["score"])))
        region = center_region_score(box_norm, q)
        prior_iou = iou_norm(box_norm, target_prior)
        area_penalty = abs(math.log(max(1e-6, float(box_norm[2] * box_norm[3])) / max(1e-6, float(target_prior[2] * target_prior[3]))))
        total = (
            float(params["w_conf"]) * conf_score
            + float(params["w_region"]) * region
            + float(params["w_prior"]) * prior_iou
            - float(params["w_area"]) * area_penalty
        )
        scored.append((total, cand, box_norm, region, prior_iou, area_penalty))
    scored.sort(key=lambda x: x[0], reverse=True)
    total, cand, box_norm, region, prior_iou, area_penalty = scored[0]
    blend_weight = float(params.get("blend_yolo_weight", 1.0))
    final_norm = blend_norm(box_norm, target_prior, blend_weight)
    return norm_to_xyxy(final_norm, iw, ih), float(cand["score"]), False, {
        "source": "yolo_candidate",
        "n_candidates": len(cands),
        "rerank_score": total,
        "region_score": region,
        "prior_iou": prior_iou,
        "area_penalty": area_penalty,
        **q,
    }


def evaluate_rows(rows: List[Dict], candidates: Dict[str, List[Dict]], priors: Dict[str, Dict[Tuple, np.ndarray]], params: Dict, method: str, split: str) -> pd.DataFrame:
    out = []
    for row in rows:
        pred, conf, missing, info = choose_candidate(row, candidates, priors, params)
        gt = clip_box(row["gold_bbox_xyxy"], row["image_width"], row["image_height"])
        iou = 0.0 if missing else iou_xyxy(pred, gt)
        out.append(
            {
                "task_id": row["task_id"],
                "sample_id": row["task_id"],
                "split": split,
                "dicom_id": row["dicom_id"],
                "subject_id": row.get("subject_id", ""),
                "study_id": row.get("study_id", ""),
                "image_path": row["image_path"],
                "finding": row["finding"],
                "class_id": CLASS_TO_ID[row["finding"]],
                "claim_sentence": row.get("claim_sentence", row.get("phrase", "")),
                "gt_x1": gt[0],
                "gt_y1": gt[1],
                "gt_x2": gt[2],
                "gt_y2": gt[3],
                "pred_x1": pred[0],
                "pred_y1": pred[1],
                "pred_x2": pred[2],
                "pred_y2": pred[3],
                "confidence": conf,
                "iou": iou,
                "hit_0_1": iou >= 0.1,
                "hit_0_3": iou >= 0.3,
                "hit_0_5": iou >= 0.5,
                "bbox_missing": bool(missing),
                "bbox_invalid": (not missing) and (pred[2] <= pred[0] or pred[3] <= pred[1]),
                "method": method,
                "bbox_type": "ms_cxr_phrase_grounding_bbox",
                **info,
            }
        )
    return pd.DataFrame(out)


def metrics_from_predictions(df: pd.DataFrame, method: str, subset: str) -> Dict:
    sub = df.copy()
    if subset == "main5":
        sub = sub[sub["finding"].isin(MAIN5)]
    if len(sub) == 0:
        return {"method": method, "subset": subset, "n": 0}
    ious = sub["iou"].astype(float).to_numpy()
    return {
        "method": method,
        "subset": subset,
        "n": int(len(sub)),
        "mean_iou": float(np.mean(ious)),
        "median_iou": float(np.median(ious)),
        "Hit@0.1": float(np.mean(ious >= 0.1)),
        "Hit@0.3": float(np.mean(ious >= 0.3)),
        "Hit@0.5": float(np.mean(ious >= 0.5)),
        "bbox_missing_rate": float(sub["bbox_missing"].astype(bool).mean()),
        "bbox_invalid_rate": float(sub["bbox_invalid"].astype(bool).mean()),
    }


def tune_params(val_rows: List[Dict], candidates: Dict[str, List[Dict]], priors: Dict[str, Dict[Tuple, np.ndarray]], quick: bool) -> Tuple[Dict, pd.DataFrame]:
    grid = []
    confs = [0.001, 0.01, 0.03, 0.05]
    w_conf = [0.25, 0.5, 1.0]
    w_region = [0.0, 0.5, 1.0, 2.0]
    w_prior = [0.0, 0.5, 1.0, 2.0]
    w_area = [0.0, 0.1, 0.25]
    blend = [1.0, 0.85, 0.70, 0.55]
    if quick:
        confs = [0.001, 0.01]
        w_conf = [0.5, 1.0]
        w_region = [0.0, 1.0]
        w_prior = [0.0, 1.0]
        w_area = [0.0, 0.1]
        blend = [1.0, 0.75]
    for c in confs:
        for wc in w_conf:
            for wr in w_region:
                for wp in w_prior:
                    for wa in w_area:
                        for by in blend:
                            grid.append(
                                {
                                    "conf": c,
                                    "w_conf": wc,
                                    "w_region": wr,
                                    "w_prior": wp,
                                    "w_area": wa,
                                    "blend_yolo_weight": by,
                                    "fallback": "none",
                                    "prior_train_weight": 0.75,
                                }
                            )
    rows = []
    best_params: Optional[Dict] = None
    best = -1.0
    for idx, params in enumerate(grid):
        pred = evaluate_rows(val_rows, candidates, priors, params, "val_grid", "val")
        m = metrics_from_predictions(pred, "val_grid", "all8")
        m.update(params)
        m["grid_index"] = idx
        rows.append(m)
        if float(m["mean_iou"]) > best:
            best = float(m["mean_iou"])
            best_params = dict(params)
    assert best_params is not None
    grid_df = pd.DataFrame(rows).sort_values("mean_iou", ascending=False)
    return best_params, grid_df


def build_singlebox_rows(eval_rows: List[Dict]) -> List[Dict]:
    csv_path = SINGLE_BOX / "eval.csv"
    if not csv_path.exists():
        return []
    sb = pd.read_csv(csv_path)
    out = []
    for _, r in sb.iterrows():
        matches = [
            row
            for row in eval_rows
            if str(row["dicom_id"]) == str(r["dicom_id"])
            and str(row["finding"]) == str(r["finding_label"])
            and np.isclose(float(row["gold_bbox_xyxy"][0]), float(r["bbox_x1"]))
            and np.isclose(float(row["gold_bbox_xyxy"][1]), float(r["bbox_y1"]))
            and np.isclose(float(row["gold_bbox_xyxy"][2]), float(r["bbox_x2"]))
            and np.isclose(float(row["gold_bbox_xyxy"][3]), float(r["bbox_y2"]))
        ]
        if matches:
            row = dict(matches[0])
            row["task_id"] = str(r["sample_id"])
            out.append(row)
    return out


def load_existing_baselines() -> pd.DataFrame:
    frames = []
    for path in [
        YOLO_MET / "final_same_split_yolo_vs_query_baselines.csv",
        RAD_DINO_MET / "final_same_split_rad_dino_detector_vs_baselines.csv",
    ]:
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=["method", "subset"], keep="first")


def make_overlays(pred: pd.DataFrame, n: int = 20) -> None:
    samples = pred.sort_values("iou", ascending=False).head(n // 2)
    hard = pred.sort_values("iou", ascending=True).head(n - len(samples))
    rows = pd.concat([samples, hard], ignore_index=True)
    paths = []
    for i, row in rows.iterrows():
        im = Image.open(row["image_path"]).convert("RGB")
        draw = ImageDraw.Draw(im)
        gt = [row["gt_x1"], row["gt_y1"], row["gt_x2"], row["gt_y2"]]
        pr = [row["pred_x1"], row["pred_y1"], row["pred_x2"], row["pred_y2"]]
        width = max(3, int(min(im.size) / 450))
        draw.rectangle(gt, outline="yellow", width=width)
        if valid_box(pr):
            draw.rectangle(pr, outline="lime", width=width)
        text = f"{row['finding']} IoU={float(row['iou']):.3f} lat={row.get('laterality','')} vert={row.get('vertical','')}"
        draw.rectangle([0, 0, min(im.width, 1000), 52], fill=(0, 0, 0))
        draw.text((8, 8), text, fill="white")
        im.thumbnail((1200, 1200))
        out = OVER / "yolo_rule_context_examples" / f"example_{i:03d}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        im.save(out, quality=92)
        paths.append(out)
    make_contact_sheet(paths, CONTACT / "yolo_rule_context_eval_examples.jpg")


def make_contact_sheet(paths: Sequence[Path], out: Path, cols: int = 5) -> None:
    if not paths:
        return
    thumbs = []
    for path in paths:
        im = Image.open(path).convert("RGB")
        im.thumbnail((360, 360))
        canvas = Image.new("RGB", (360, 360), "black")
        canvas.paste(im, ((360 - im.width) // 2, (360 - im.height) // 2))
        thumbs.append(canvas)
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * 360, rows * 360), "white")
    for idx, im in enumerate(thumbs):
        sheet.paste(im, ((idx % cols) * 360, (idx // cols) * 360))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=92)


def write_report(model_tag: str, best_params: Dict, summary: pd.DataFrame, comparison: pd.DataFrame, grid: pd.DataFrame) -> None:
    lines = [
        "# MS-CXR YOLO Rule-Context v1",
        "",
        "## 한 줄 결론",
        "YOLO detector를 새로 바꾸지 않고, YOLO 후보 박스를 rule-context로 재점수화/보정했다. 이 실험은 YOLO가 문장을 직접 읽는 것이 아니라, YOLO가 만든 후보 위에 의학적 위치 문맥을 후처리로 얹는 방식이다.",
        "",
        "## 방식",
        "- YOLO 입력: CXR image only",
        "- Rule-context 입력: MS-CXR claim_sentence에서 finding, laterality, vertical region을 파싱",
        "- 사용 방식: 같은 class YOLO 후보 중 위치 문맥과 train-split prior에 맞는 후보를 선택하고, val에서 고른 비율로 bbox를 보정",
        "- eval gold는 tuning에 사용하지 않음",
        "",
        "## Val-selected parameters",
        "```json",
        json.dumps(best_params, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Summary",
        summary.to_markdown(index=False),
        "",
    ]
    if len(comparison):
        lines += ["## Same split comparison", comparison.to_markdown(index=False), ""]
    lines += [
        "## 해석 주의",
        "- 이 방법은 YOLO 내부가 text-conditioned detector가 된 것은 아니다.",
        "- MS-CXR bbox는 phrase-grounding bbox이지 pixel-level lesion mask가 아니다.",
        "- rule-context가 올라가면 YOLO 후보 생성과 의학적 위치 후처리가 상보적이라는 뜻이다.",
        "- 성능이 안 올라가면 YOLO 후보 pool 자체가 phrase-level 위치를 충분히 포함하지 못했거나, 후보 선택보다 detector head 학습이 병목이라는 뜻이다.",
        "",
        "## Top val grid rows",
        grid.head(10).to_markdown(index=False),
        "",
    ]
    text = "\n".join(lines) + "\n"
    (REPORT / "README_KO.md").write_text(text, encoding="utf-8")
    (REPORT / "FINAL_RESULT_KO.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-tag", default="yolov8n")
    p.add_argument("--weights", default="")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--pred-conf", type=float, default=0.001)
    p.add_argument("--augment", action="store_true", help="Use YOLO test-time augmentation while exporting candidates.")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--make-overlays", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    weights = Path(args.weights) if args.weights else YOLO_RUNS / f"{args.model_tag}_full" / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(weights)

    train_rows = source_rows("train")
    val_rows = source_rows("val")
    eval_rows = source_rows("eval")
    priors = make_train_priors(train_rows)
    val_candidates = predict_candidates(
        weights,
        val_rows,
        args.imgsz,
        args.pred_conf,
        args.device,
        f"{args.model_tag}_val",
        augment=args.augment,
    )
    eval_candidates = predict_candidates(
        weights,
        eval_rows,
        args.imgsz,
        args.pred_conf,
        args.device,
        f"{args.model_tag}_eval",
        augment=args.augment,
    )

    best_params, grid = tune_params(val_rows, val_candidates, priors, quick=args.quick)
    grid.to_csv(MET / f"{args.model_tag}_val_rule_context_grid_search.csv", index=False)
    (CFG / f"{args.model_tag}_best_params.json").write_text(json.dumps(best_params, ensure_ascii=False, indent=2), encoding="utf-8")

    pred = evaluate_rows(eval_rows, eval_candidates, priors, best_params, f"{args.model_tag}_rule_context_postprocess", "eval")
    pred_path = PRED / f"{args.model_tag}_rule_context_eval_row_predictions.csv"
    pred.to_csv(pred_path, index=False)
    summary_rows = [
        metrics_from_predictions(pred, f"{args.model_tag}_rule_context_postprocess", "all8"),
        metrics_from_predictions(pred, f"{args.model_tag}_rule_context_postprocess", "main5"),
    ]

    single_rows = build_singlebox_rows(eval_rows)
    if single_rows:
        sb_pred = evaluate_rows(single_rows, eval_candidates, priors, best_params, f"{args.model_tag}_rule_context_postprocess_singlebox", "eval")
        sb_pred.to_csv(PRED / f"{args.model_tag}_rule_context_singlebox_eval_predictions.csv", index=False)
        summary_rows.append(metrics_from_predictions(sb_pred, f"{args.model_tag}_rule_context_postprocess_singlebox", "singlebox"))

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(MET / f"{args.model_tag}_rule_context_summary.csv", index=False)
    per = pred.groupby("finding")["iou"].agg(["count", "mean", "median"]).reset_index()
    per["Hit@0.3"] = pred.groupby("finding")["hit_0_3"].mean().values
    per["Hit@0.5"] = pred.groupby("finding")["hit_0_5"].mean().values
    per.to_csv(MET / f"{args.model_tag}_rule_context_per_finding.csv", index=False)

    base = load_existing_baselines()
    comparison = pd.concat([base, summary], ignore_index=True) if len(base) else summary.copy()
    comparison.to_csv(MET / f"{args.model_tag}_same_split_yolo_rule_context_comparison.csv", index=False)

    if args.make_overlays:
        make_overlays(pred)

    write_report(args.model_tag, best_params, summary, comparison, grid)
    print(f"project_root={PROJECT_ROOT}")
    print(f"model_tag={args.model_tag}")
    print(f"eval_rows={len(eval_rows)}")
    print(f"rule_context_mean_iou_all8={summary.iloc[0]['mean_iou']:.6f}")
    print(f"rule_context_hit03_all8={summary.iloc[0]['Hit@0.3']:.6f}")
    print(f"rule_context_mean_iou_main5={summary.iloc[1]['mean_iou']:.6f}")
    print(f"summary_path={MET / f'{args.model_tag}_rule_context_summary.csv'}")
    print(f"comparison_path={MET / f'{args.model_tag}_same_split_yolo_rule_context_comparison.csv'}")
    print(f"report_path={REPORT / 'README_KO.md'}")


if __name__ == "__main__":
    main()

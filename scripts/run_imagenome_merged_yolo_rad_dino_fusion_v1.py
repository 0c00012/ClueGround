#!/usr/bin/env python
"""Chest ImaGenome YOLO + RAD-DINO fusion experiments.

This extends the MS-CXR "merged detector candidates + RAD-DINO query head"
idea to three Chest ImaGenome localization tasks:

1. finding-region weak/reference boxes
2. device-linked weak landmark/region boxes
3. anatomy object boxes

The detector is trained only on each task's train split.  Fusion weights are
tuned on the corresponding validation split only and then fixed for eval.
Chest ImaGenome finding/device boxes are weak region/reference boxes, not
pixel-level lesion masks and not exact device contours.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from models_ms_cxr_vfm_localizer import PatchHeatmapBBoxHead  # noqa: E402
import run_ms_cxr_vfm_localizer_stage1_p10_p19 as ms_base  # noqa: E402
import run_chest_imagenome_finding_region_rule_ablation_v1 as fr_base  # noqa: E402
import run_imagenome_device_anatomy_10k_rule_smm_v1 as da_base  # noqa: E402
import run_imagenome_device_anatomy_10k_query_ablation as da_ablation  # noqa: E402


EXP_NAME = "imagenome_merged_yolo_rad_dino_fusion_v1"
EXP = PROJECT_ROOT / "experiments" / EXP_NAME
PRED = EXP / "predictions"
MET = EXP / "metrics"
CFG = EXP / "configs"
LOGS = EXP / "logs"
REPORT = PROJECT_ROOT / "reports" / EXP_NAME
TRAIN = PROJECT_ROOT / "training" / EXP_NAME
DATASET = TRAIN / "yolo_dataset"
RUNS = TRAIN / "runs"

FINDING_DATA = PROJECT_ROOT / "training" / "chest_imagenome_finding_region_rule_ablation_v1" / "datasets"
FINDING_PRED = PROJECT_ROOT / "experiments" / "chest_imagenome_finding_region_rule_ablation_v1" / "predictions"
FINDING_MET = PROJECT_ROOT / "experiments" / "chest_imagenome_finding_region_rule_ablation_v1" / "metrics"
FINDING_CKPT = PROJECT_ROOT / "training" / "chest_imagenome_finding_region_rule_ablation_v1" / "checkpoints"

DA_DATA = PROJECT_ROOT / "training" / "imagenome_device_anatomy_10k_rule_smm_v1" / "datasets"
DA_PRED = PROJECT_ROOT / "experiments" / "imagenome_device_anatomy_10k_rule_smm_v1" / "predictions"
DA_MET = PROJECT_ROOT / "experiments" / "imagenome_device_anatomy_10k_rule_smm_v1" / "metrics"
DA_CKPT = PROJECT_ROOT / "training" / "imagenome_device_anatomy_10k_rule_smm_v1" / "checkpoints"

SPLITS = ("train", "val", "eval")
MAIN_TASKS = ("finding_region", "device", "anatomy")


@dataclass
class TaskConfig:
    name: str
    rows: Dict[str, Path]
    label_col: str
    rad_rule_method: str
    rad_label_method: Optional[str]
    checkpoint_rule: Path
    baseline_metrics: Path
    bbox_note: str


TASKS: Dict[str, TaskConfig] = {
    "finding_region": TaskConfig(
        name="finding_region",
        rows={s: FINDING_DATA / f"{s}.jsonl" for s in SPLITS},
        label_col="finding",
        rad_rule_method="finding_region_rule_context_heatmap",
        rad_label_method="finding_region_label_only_heatmap",
        checkpoint_rule=FINDING_CKPT / "finding_region_rule_context_heatmap.pt",
        baseline_metrics=FINDING_MET / "summary.csv",
        bbox_note="Chest ImaGenome weak finding-region/reference bbox; not lesion mask.",
    ),
    "device": TaskConfig(
        name="device",
        rows={s: DA_DATA / f"device_{s}.jsonl" for s in SPLITS},
        label_col="finding",
        rad_rule_method="device_rule_query_heatmap",
        rad_label_method="device_label_only_query_heatmap",
        checkpoint_rule=DA_CKPT / "device_rule_query_heatmap.pt",
        baseline_metrics=DA_MET / "summary.csv",
        bbox_note="Chest ImaGenome device-linked weak landmark/region bbox; not exact device contour.",
    ),
    "anatomy": TaskConfig(
        name="anatomy",
        rows={s: DA_DATA / f"anatomy_{s}.jsonl" for s in SPLITS},
        label_col="finding",
        rad_rule_method="anatomy_rule_query_heatmap",
        rad_label_method="anatomy_label_only_query_heatmap",
        checkpoint_rule=DA_CKPT / "anatomy_rule_query_heatmap.pt",
        baseline_metrics=DA_MET / "summary.csv",
        bbox_note="Chest ImaGenome anatomy object bbox; not lesion bbox.",
    ),
}


def ensure_dirs() -> None:
    for p in [EXP, PRED, MET, CFG, LOGS, REPORT, TRAIN, DATASET, RUNS]:
        p.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def task_rows(task: str, split: str) -> List[Dict]:
    return read_jsonl(TASKS[task].rows[split])


def class_names(task: str) -> List[str]:
    labels = []
    seen = set()
    for split in SPLITS:
        for r in task_rows(task, split):
            label = str(r[TASKS[task].label_col])
            if label not in seen:
                seen.add(label)
                labels.append(label)
    return sorted(labels)


def valid_box(box: Optional[Sequence[float]]) -> bool:
    if not box or len(box) != 4:
        return False
    x1, y1, x2, y2 = [float(x) for x in box]
    return x2 > x1 and y2 > y1


def clip_box(box: Sequence[float], iw: float, ih: float) -> List[float]:
    x1, y1, x2, y2 = [float(x) for x in box]
    return [max(0.0, min(iw, x1)), max(0.0, min(ih, y1)), max(0.0, min(iw, x2)), max(0.0, min(ih, y2))]


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


def xyxy_to_norm(box: Sequence[float], iw: float, ih: float) -> np.ndarray:
    x1, y1, x2, y2 = clip_box(box, iw, ih)
    return sanitize_norm([(x1 + x2) / (2 * iw), (y1 + y2) / (2 * ih), (x2 - x1) / iw, (y2 - y1) / ih])


def norm_to_xyxy(box: Sequence[float], iw: float, ih: float) -> List[float]:
    cx, cy, w, h = [float(x) for x in box]
    return clip_box([(cx - w / 2) * iw, (cy - h / 2) * ih, (cx + w / 2) * iw, (cy + h / 2) * ih], iw, ih)


def sanitize_norm(box: Sequence[float]) -> np.ndarray:
    arr = np.asarray(box, dtype="float32")
    arr[:2] = np.clip(arr[:2], 0.0, 1.0)
    arr[2:] = np.clip(arr[2:], 0.02, 1.0)
    return arr


def iou_norm(a: Sequence[float], b: Sequence[float]) -> float:
    return iou_xyxy(norm_to_xyxy(a, 1.0, 1.0), norm_to_xyxy(b, 1.0, 1.0))


def blend_norm(a: Sequence[float], b: Sequence[float], weight_a: float) -> np.ndarray:
    return sanitize_norm(float(weight_a) * np.asarray(a, dtype="float32") + (1.0 - float(weight_a)) * np.asarray(b, dtype="float32"))


def xyxy_to_yolo(box: Sequence[float], iw: float, ih: float) -> Optional[List[float]]:
    if not valid_box(box):
        return None
    cx, cy, w, h = xyxy_to_norm(box, iw, ih)
    if w <= 0 or h <= 0:
        return None
    return [float(cx), float(cy), float(w), float(h)]


def copy_or_link(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    if mode in {"symlink", "auto"}:
        try:
            dst.symlink_to(src)
            return "symlink"
        except Exception:
            if mode == "symlink":
                raise
    if mode in {"hardlink", "auto"}:
        try:
            dst.hardlink_to(src)
            return "hardlink"
        except Exception:
            if mode == "hardlink":
                raise
    shutil.copy2(src, dst)
    return "copy"


def build_yolo_dataset(task: str, link_mode: str, force: bool) -> pd.DataFrame:
    names = class_names(task)
    label_to_id = {name: i for i, name in enumerate(names)}
    task_dir = DATASET / task
    manifest_path = task_dir / "manifest.csv"
    if manifest_path.exists() and not force:
        return pd.read_csv(manifest_path)
    if force and task_dir.exists():
        shutil.rmtree(task_dir)
    records = []
    for split in SPLITS:
        rows = task_rows(task, split)
        image_dir = task_dir / "images" / split
        label_dir = task_dir / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        grouped: Dict[str, List[Dict]] = defaultdict(list)
        for row in rows:
            grouped[str(row["image_path"])].append(row)
        for image_path, image_rows in grouped.items():
            src = Path(image_path)
            if not src.exists():
                for row in image_rows:
                    records.append({"task": task, "split": split, "task_id": row["task_id"], "status": "missing_image", "image_path": image_path})
                continue
            dst = image_dir / f"{src.stem}{src.suffix.lower()}"
            link_status = copy_or_link(src, dst, link_mode)
            label_path = label_dir / f"{dst.stem}.txt"
            lines = []
            for row in image_rows:
                yolo = xyxy_to_yolo(row["gold_bbox_xyxy"], row["image_width"], row["image_height"])
                if yolo is None:
                    records.append({"task": task, "split": split, "task_id": row["task_id"], "status": "invalid_box", "image_path": image_path})
                    continue
                cls = label_to_id[str(row[TASKS[task].label_col])]
                lines.append(f"{cls} {yolo[0]:.8f} {yolo[1]:.8f} {yolo[2]:.8f} {yolo[3]:.8f}")
                records.append(
                    {
                        "task": task,
                        "split": split,
                        "task_id": row["task_id"],
                        "dicom_id": row["dicom_id"],
                        "subject_id": row["subject_id"],
                        "study_id": row["study_id"],
                        "image_path": str(dst),
                        "source_image_path": image_path,
                        "label_path": str(label_path),
                        "finding": row[TASKS[task].label_col],
                        "class_id": cls,
                        "gold_x1": row["gold_bbox_xyxy"][0],
                        "gold_y1": row["gold_bbox_xyxy"][1],
                        "gold_x2": row["gold_bbox_xyxy"][2],
                        "gold_y2": row["gold_bbox_xyxy"][3],
                        "image_width": row["image_width"],
                        "image_height": row["image_height"],
                        "claim_sentence": row.get("claim_sentence", ""),
                        "bbox_type": row.get("bbox_type", ""),
                        "status": "ok",
                        "link_status": link_status,
                    }
                )
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    yaml = [f"path: {task_dir.as_posix()}", "train: images/train", "val: images/val", "test: images/eval", "names:"]
    yaml += [f"  {i}: {name}" for i, name in enumerate(names)]
    (task_dir / f"{task}_yolo.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    manifest = pd.DataFrame(records)
    manifest.to_csv(manifest_path, index=False)
    manifest[manifest["status"].eq("ok")].groupby("split").agg(rows=("task_id", "count"), images=("image_path", "nunique")).reset_index().to_csv(
        MET / f"{task}_yolo_dataset_summary.csv", index=False
    )
    pd.DataFrame({"class_id": list(label_to_id.values()), "class_name": list(label_to_id.keys())}).to_csv(CFG / f"{task}_class_mapping.csv", index=False)
    return manifest


def model_tag(model_name: str) -> str:
    return Path(model_name).stem.replace("-", "_")


def train_yolo(task: str, model_name: str, args: argparse.Namespace) -> Dict:
    from ultralytics import YOLO

    tag = model_tag(model_name)
    run_name = f"{task}_{tag}_e{args.epochs}_s{args.seed}"
    run_dir = RUNS / task / run_name
    best = run_dir / "weights" / "best.pt"
    if best.exists() and not args.force_train:
        return {"task": task, "model": model_name, "tag": tag, "status": "cached", "weights": str(best), "time_sec": 0.0}
    start = time.time()
    model = YOLO(model_name)
    model.train(
        data=str(DATASET / task / f"{task}_yolo.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=str(RUNS / task),
        name=run_name,
        exist_ok=True,
        device=args.device,
        pretrained=True,
        seed=args.seed,
        patience=max(10, min(40, args.epochs // 2)),
    )
    if not best.exists():
        raise FileNotFoundError(best)
    return {"task": task, "model": model_name, "tag": tag, "status": "ok", "weights": str(best), "time_sec": round(time.time() - start, 2)}


def read_candidate_csv(path: Path) -> Dict[str, List[Dict]]:
    df = pd.read_csv(path)
    grouped: Dict[str, List[Dict]] = defaultdict(list)
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


def predict_candidates(task: str, weights: Path, tag: str, split: str, args: argparse.Namespace) -> Dict[str, List[Dict]]:
    out_csv = PRED / f"{task}_{tag}_{split}_conf{str(args.pred_conf).replace('.', 'p')}_candidates.csv"
    if out_csv.exists() and not args.force_predict:
        return read_candidate_csv(out_csv)
    from ultralytics import YOLO

    image_by_dicom = {}
    for row in task_rows(task, split):
        image_by_dicom[str(row["dicom_id"])] = row["image_path"]
    items = sorted(image_by_dicom.items())
    model = YOLO(str(weights))
    out_rows = []
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for start in range(0, len(items), max(1, args.predict_batch)):
        batch = items[start : start + args.predict_batch]
        results = model.predict(
            source=[p for _, p in batch],
            imgsz=args.imgsz,
            conf=args.pred_conf,
            device=args.device,
            verbose=False,
            stream=False,
            max_det=args.max_det,
            batch=max(1, args.predict_batch),
        )
        for (dicom_id, image_path), res in zip(batch, results):
            preds = []
            if res.boxes is not None and len(res.boxes):
                xyxy = res.boxes.xyxy.cpu().numpy()
                cls = res.boxes.cls.cpu().numpy().astype(int)
                scores = res.boxes.conf.cpu().numpy()
                for box, c, s in zip(xyxy, cls, scores):
                    preds.append({"class_id": int(c), "score": float(s), "box": [float(x) for x in box]})
            preds.sort(key=lambda x: x["score"], reverse=True)
            for rank, pred in enumerate(preds):
                cand = {**pred, "source_model": tag, "rank": rank}
                grouped[dicom_id].append(cand)
                out_rows.append(
                    {
                        "task": task,
                        "split": split,
                        "dicom_id": dicom_id,
                        "image_path": image_path,
                        "class_id": cand["class_id"],
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


def merge_candidates(*groups: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = defaultdict(list)
    for group in groups:
        for dicom, rows in group.items():
            out[dicom].extend(rows)
    for dicom in out:
        out[dicom].sort(key=lambda r: float(r["score"]), reverse=True)
    return out


def load_existing_eval_rad_jsonl(task: str, method: str) -> Optional[pd.DataFrame]:
    if task == "finding_region":
        path = FINDING_PRED / f"{method}.jsonl"
    else:
        path = DA_PRED / f"{method}.jsonl"
    if not path.exists():
        return None
    rows = read_jsonl(path)
    return pd.DataFrame(rows)


def infer_head_from_checkpoint(path: Path) -> PatchHeatmapBBoxHead:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["state"]
    token_dim = int(state["token_proj.weight"].shape[1])
    hidden = int(state["token_proj.weight"].shape[0])
    if "context_proj.weight" in state:
        query_dim = int(state["context_proj.weight"].shape[1])
    else:
        query_dim = int(ckpt.get("query_dim", 0))
    model = PatchHeatmapBBoxHead(token_dim, query_dim, hidden=hidden, dropout=0.1)
    model.load_state_dict(state, strict=True)
    return model


def ensure_device_anatomy_rule_features() -> None:
    if (da_base.FEAT / "device_rule_query_val.npz").exists() and (da_base.FEAT / "anatomy_rule_query_val.npz").exists():
        return
    class Args:
        force = False
        build_rule = True
        rule_dim = 128
    da_base.ensure_dirs()
    da_base.build_rule_query_features(Args())


def query_features(task: str, split: str) -> Tuple[List[Dict], np.ndarray]:
    if task == "finding_region":
        fr_base.ensure_dirs()
        maps = fr_base.build_context_maps()
        rows = fr_base.load_rows(split)
        return rows, fr_base.encode_rule_context(rows, split, maps).astype("float32")
    ensure_device_anatomy_rule_features()
    rows = da_base.load_rows(task, split)
    q = np.load(da_base.FEAT / f"{task}_rule_query_{split}.npz", allow_pickle=True)
    ids = np.array([str(x) for x in q["task_ids"]], dtype=object)
    pos = {tid: i for i, tid in enumerate(ids)}
    ordered, idx = [], []
    for row in rows:
        tid = str(row["task_id"])
        if tid in pos:
            ordered.append(row)
            idx.append(pos[tid])
    return ordered, q["features"][np.asarray(idx, dtype=np.int64)].astype("float32")


class RadPredictor:
    def __init__(self, model_name: str, device: str):
        self.device = "cuda" if torch.cuda.is_available() and device != "cpu" else "cpu"
        self.model_name, self.processor, self.vfm, self.errors = ms_base.load_vfm(model_name, self.device)

    @torch.no_grad()
    def patch_tokens(self, images: List[Image.Image]) -> torch.Tensor:
        _, _, _, tokens = ms_base.vfm_forward(images, self.processor, self.vfm, self.device)
        return torch.tensor(tokens, dtype=torch.float32, device=self.device)


def rad_predict(task: str, split: str, args: argparse.Namespace, rad: RadPredictor) -> pd.DataFrame:
    cfg = TASKS[task]
    out_csv = PRED / f"{task}_rad_dino_rule_{split}_predictions.csv"
    if out_csv.exists() and not args.force_rad:
        return pd.read_csv(out_csv)
    if split == "eval":
        existing = load_existing_eval_rad_jsonl(task, cfg.rad_rule_method)
        if existing is not None and not args.force_rad:
            rows = task_rows(task, "eval")
            by_id = {r["task_id"]: r for r in rows}
            out = []
            for _, p in existing.iterrows():
                row = by_id.get(str(p["task_id"]))
                if not row:
                    continue
                box = p["pred_bbox_xyxy"]
                if isinstance(box, str):
                    box = json.loads(box)
                gt = row["gold_bbox_xyxy"]
                iou = iou_xyxy(box, gt)
                out.append(pred_record(task, row, box, iou, "rad_dino_rule"))
            df = pd.DataFrame(out)
            df.to_csv(out_csv, index=False)
            return df

    rows, q = query_features(task, split)
    model = infer_head_from_checkpoint(cfg.checkpoint_rule).to(rad.device).eval()
    out = []
    bs = max(1, args.rad_batch)
    for start in range(0, len(rows), bs):
        batch = rows[start : start + bs]
        images = [Image.open(r["image_path"]).convert("RGB") for r in batch]
        tokens = rad.patch_tokens(images)
        q_t = torch.tensor(q[start : start + len(batch)], dtype=torch.float32, device=rad.device)
        with torch.no_grad():
            pred_norm, _ = model(tokens, q_t)
        pred_np = pred_norm.detach().cpu().numpy()
        for row, p in zip(batch, pred_np):
            p = sanitize_norm(p)
            box = norm_to_xyxy(p, row["image_width"], row["image_height"])
            out.append(pred_record(task, row, box, iou_xyxy(box, row["gold_bbox_xyxy"]), "rad_dino_rule"))
    df = pd.DataFrame(out)
    df.to_csv(out_csv, index=False)
    return df


def pred_record(task: str, row: Dict, box: Sequence[float], iou: float, method: str, extra: Optional[Dict] = None) -> Dict:
    rec = {
        "task": task,
        "task_id": row["task_id"],
        "sample_id": row["task_id"],
        "split": row.get("split", ""),
        "dicom_id": row["dicom_id"],
        "subject_id": row["subject_id"],
        "study_id": row["study_id"],
        "image_path": row["image_path"],
        "finding": row[TASKS[task].label_col],
        "claim_sentence": row.get("claim_sentence", ""),
        "bbox_name_reference": row.get("bbox_name_reference", ""),
        "gt_x1": row["gold_bbox_xyxy"][0],
        "gt_y1": row["gold_bbox_xyxy"][1],
        "gt_x2": row["gold_bbox_xyxy"][2],
        "gt_y2": row["gold_bbox_xyxy"][3],
        "pred_x1": float(box[0]),
        "pred_y1": float(box[1]),
        "pred_x2": float(box[2]),
        "pred_y2": float(box[3]),
        "iou": float(iou),
        "hit_0_1": float(iou) >= 0.1,
        "hit_0_3": float(iou) >= 0.3,
        "hit_0_5": float(iou) >= 0.5,
        "bbox_missing": False,
        "bbox_invalid": not valid_box(box),
        "method": method,
        "bbox_note": TASKS[task].bbox_note,
    }
    if extra:
        rec.update(extra)
    return rec


def row_class_id(row: Dict, label_to_id: Dict[str, int], task: str) -> int:
    return label_to_id[str(row[TASKS[task].label_col])]


def evaluate_yolo_class_only(task: str, rows: List[Dict], candidates: Dict[str, List[Dict]], label_to_id: Dict[str, int], method: str) -> pd.DataFrame:
    out = []
    for row in rows:
        cls = row_class_id(row, label_to_id, task)
        cands = [c for c in candidates.get(str(row["dicom_id"]), []) if int(c["class_id"]) == cls]
        cands.sort(key=lambda c: float(c["score"]), reverse=True)
        if not cands:
            box = [0.0, 0.0, 0.0, 0.0]
            out.append(pred_record(task, row, box, 0.0, method, {"bbox_missing": True, "source_model": "", "confidence": 0.0}))
            continue
        cand = cands[0]
        box = clip_box(cand["box"], row["image_width"], row["image_height"])
        out.append(
            pred_record(
                task,
                row,
                box,
                iou_xyxy(box, row["gold_bbox_xyxy"]),
                method,
                {"source_model": cand["source_model"], "confidence": cand["score"], "candidate_rank": cand["rank"]},
            )
        )
    return pd.DataFrame(out)


def choose_fusion(
    task: str,
    row: Dict,
    candidates: Dict[str, List[Dict]],
    label_to_id: Dict[str, int],
    rad_map: Dict[str, np.ndarray],
    cfg: Dict,
) -> Tuple[List[float], Dict]:
    iw, ih = float(row["image_width"]), float(row["image_height"])
    cls = row_class_id(row, label_to_id, task)
    rad_norm = rad_map.get(str(row["task_id"]))
    if rad_norm is None:
        rad_norm = xyxy_to_norm(row["gold_bbox_xyxy"], iw, ih) * 0 + np.asarray([0.5, 0.5, 0.5, 0.5], dtype="float32")
    cands = [
        c
        for c in candidates.get(str(row["dicom_id"]), [])
        if int(c["class_id"]) == cls and float(c["score"]) >= float(cfg["conf"]) and int(c.get("rank", 9999)) < int(cfg["max_rank"])
    ]
    if not cands:
        return norm_to_xyxy(rad_norm, iw, ih), {"source": "rad_fallback", "confidence": 0.0, "dino_iou": 1.0, "candidate_rank": -1}
    scored = []
    for c in cands:
        b_norm = xyxy_to_norm(c["box"], iw, ih)
        conf_score = math.log1p(20.0 * max(0.0, float(c["score"])))
        dino_i = iou_norm(b_norm, rad_norm)
        rank_bonus = 1.0 / (1.0 + float(c.get("rank", 0)))
        area = max(1e-6, float(b_norm[2] * b_norm[3]))
        rad_area = max(1e-6, float(rad_norm[2] * rad_norm[3]))
        area_penalty = abs(math.log(area / rad_area))
        total = (
            float(cfg["w_conf"]) * conf_score
            + float(cfg["w_dino"]) * dino_i
            + float(cfg["w_rank"]) * rank_bonus
            - float(cfg["w_area"]) * area_penalty
        )
        scored.append((total, c, b_norm, dino_i, area_penalty, rank_bonus))
    scored.sort(key=lambda x: x[0], reverse=True)
    total, cand, b_norm, dino_i, area_penalty, rank_bonus = scored[0]
    final_norm = blend_norm(b_norm, rad_norm, float(cfg["blend_yolo"]))
    return norm_to_xyxy(final_norm, iw, ih), {
        "source": "yolo_rad_fusion",
        "source_model": cand.get("source_model", ""),
        "confidence": cand["score"],
        "candidate_rank": cand.get("rank", -1),
        "n_candidates": len(cands),
        "fusion_score": total,
        "dino_iou": dino_i,
        "area_penalty": area_penalty,
        "rank_bonus": rank_bonus,
    }


def evaluate_fusion(
    task: str,
    rows: List[Dict],
    candidates: Dict[str, List[Dict]],
    label_to_id: Dict[str, int],
    rad_pred: pd.DataFrame,
    cfg: Dict,
    method: str,
) -> pd.DataFrame:
    rad_map = {}
    for _, r in rad_pred.iterrows():
        rad_map[str(r["task_id"])] = xyxy_to_norm([r["pred_x1"], r["pred_y1"], r["pred_x2"], r["pred_y2"]], float(r.get("image_width", 1) or 1), float(r.get("image_height", 1) or 1)) if "image_width" in r else None
    # Rebuild with row dimensions, because prediction CSV may not carry them.
    if any(v is None for v in rad_map.values()):
        rad_by_id = {str(r["task_id"]): r for _, r in rad_pred.iterrows()}
        rad_map = {}
        for row in rows:
            rp = rad_by_id.get(str(row["task_id"]))
            if rp is not None:
                rad_map[str(row["task_id"])] = xyxy_to_norm([rp["pred_x1"], rp["pred_y1"], rp["pred_x2"], rp["pred_y2"]], row["image_width"], row["image_height"])
    out = []
    for row in rows:
        box, info = choose_fusion(task, row, candidates, label_to_id, rad_map, cfg)
        out.append(pred_record(task, row, box, iou_xyxy(box, row["gold_bbox_xyxy"]), method, info))
    return pd.DataFrame(out)


def fusion_grid(quick: bool) -> List[Dict]:
    if quick:
        confs = [0.001, 0.01]
        max_ranks = [30, 80]
        w_dinos = [0.0, 1.0, 2.0]
        blends = [1.0, 0.8, 0.65]
    else:
        confs = [0.001, 0.005, 0.01, 0.03]
        max_ranks = [20, 50, 100]
        w_dinos = [0.0, 0.5, 1.0, 2.0, 4.0]
        blends = [1.0, 0.9, 0.8, 0.65, 0.5]
    rows = []
    for conf in confs:
        for max_rank in max_ranks:
            for w_conf in [0.75, 1.0]:
                for w_dino in w_dinos:
                    for w_rank in [0.0, 0.15]:
                        for w_area in [0.0, 0.15]:
                            for blend in blends:
                                rows.append(
                                    {
                                        "conf": conf,
                                        "max_rank": max_rank,
                                        "w_conf": w_conf,
                                        "w_dino": w_dino,
                                        "w_rank": w_rank,
                                        "w_area": w_area,
                                        "blend_yolo": blend,
                                    }
                                )
    return rows


def metric(df: pd.DataFrame, method: str, subset: str) -> Dict:
    arr = df["iou"].astype(float).to_numpy()
    return {
        "method": method,
        "subset": subset,
        "n": int(len(df)),
        "mean_iou": float(arr.mean()) if len(arr) else 0.0,
        "median_iou": float(np.median(arr)) if len(arr) else 0.0,
        "Hit@0.1": float((arr >= 0.1).mean()) if len(arr) else 0.0,
        "Hit@0.3": float((arr >= 0.3).mean()) if len(arr) else 0.0,
        "Hit@0.5": float((arr >= 0.5).mean()) if len(arr) else 0.0,
        "bbox_missing_rate": float(df.get("bbox_missing", pd.Series([False] * len(df))).astype(bool).mean()) if len(df) else 0.0,
        "bbox_invalid_rate": float(df.get("bbox_invalid", pd.Series([False] * len(df))).astype(bool).mean()) if len(df) else 0.0,
    }


def tune_fusion(task: str, val_rows: List[Dict], val_candidates: Dict[str, List[Dict]], label_to_id: Dict[str, int], rad_val: pd.DataFrame, quick: bool) -> Tuple[Dict, pd.DataFrame]:
    grid_rows = []
    best_cfg, best_score = None, -1.0
    for idx, cfg in enumerate(fusion_grid(quick)):
        pred = evaluate_fusion(task, val_rows, val_candidates, label_to_id, rad_val, cfg, f"{task}_fusion_val_grid")
        m = metric(pred, f"{task}_fusion_val_grid", task)
        row = {**m, **cfg, "grid_index": idx}
        grid_rows.append(row)
        if row["mean_iou"] > best_score:
            best_score = row["mean_iou"]
            best_cfg = dict(cfg) | {"grid_index": idx}
    assert best_cfg is not None
    grid = pd.DataFrame(grid_rows).sort_values("mean_iou", ascending=False)
    return best_cfg, grid


def candidate_oracle(task: str, rows: List[Dict], candidates: Dict[str, List[Dict]], label_to_id: Dict[str, int]) -> pd.DataFrame:
    out = []
    for row in rows:
        cls = row_class_id(row, label_to_id, task)
        cands = [c for c in candidates.get(str(row["dicom_id"]), []) if int(c["class_id"]) == cls]
        best = None
        best_iou = -1.0
        for c in cands:
            box = clip_box(c["box"], row["image_width"], row["image_height"])
            val = iou_xyxy(box, row["gold_bbox_xyxy"])
            if val > best_iou:
                best_iou, best = val, c
        if best is None:
            out.append(pred_record(task, row, [0, 0, 0, 0], 0.0, f"{task}_candidate_oracle", {"bbox_missing": True}))
        else:
            box = clip_box(best["box"], row["image_width"], row["image_height"])
            out.append(pred_record(task, row, box, best_iou, f"{task}_candidate_oracle", {"source_model": best["source_model"], "candidate_rank": best["rank"]}))
    return pd.DataFrame(out)


def baseline_summary_for_task(task: str) -> List[Dict]:
    cfg = TASKS[task]
    out = []
    if not cfg.baseline_metrics.exists():
        return out
    df = pd.read_csv(cfg.baseline_metrics)
    if task == "finding_region":
        subset = "finding_region_all"
        methods = [cfg.rad_label_method, cfg.rad_rule_method]
    else:
        subset = task
        methods = [
            f"{task}_no_query_query_heatmap",
            f"{task}_label_only_query_heatmap",
            f"{task}_rule_query_heatmap",
            f"{task}_smm_query_heatmap",
        ]
    for method in methods:
        row = df[df["method"].eq(method) & df["subset"].eq(subset)]
        if not len(row):
            continue
        rr = row.iloc[0].to_dict()
        rr["task"] = task
        rr["source"] = "existing_rad_dino_baseline"
        out.append(rr)
    return out


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, label: str, reps: int, seed: int) -> Dict:
    aa = a[["sample_id", "subject_id", "iou"]].rename(columns={"iou": "a"})
    bb = b[["sample_id", "iou"]].rename(columns={"iou": "b"})
    df = aa.merge(bb, on="sample_id", how="inner")
    rng = np.random.default_rng(seed)
    groups = df.groupby(df["subject_id"].astype(str)).indices
    units = np.array(list(groups.keys()))
    diffs, h03, h05 = [], [], []
    for _ in range(reps):
        sampled = rng.choice(units, size=len(units), replace=True)
        idx = []
        for u in sampled:
            idx.extend(groups[u])
        sub = df.iloc[idx]
        diffs.append(float((sub["a"] - sub["b"]).mean()))
        h03.append(float(((sub["a"] >= 0.3).astype(float) - (sub["b"] >= 0.3).astype(float)).mean()))
        h05.append(float(((sub["a"] >= 0.5).astype(float) - (sub["b"] >= 0.5).astype(float)).mean()))
    return {
        "comparison": label,
        "n": int(len(df)),
        "mean_iou_diff": float(np.mean(diffs)),
        "mean_iou_ci_low": float(np.quantile(diffs, 0.025)),
        "mean_iou_ci_high": float(np.quantile(diffs, 0.975)),
        "mean_iou_prob_gt_0": float(np.mean(np.asarray(diffs) > 0)),
        "hit_0_3_diff": float(np.mean(h03)),
        "hit_0_3_ci_low": float(np.quantile(h03, 0.025)),
        "hit_0_3_ci_high": float(np.quantile(h03, 0.975)),
        "hit_0_5_diff": float(np.mean(h05)),
        "hit_0_5_ci_low": float(np.quantile(h05, 0.025)),
        "hit_0_5_ci_high": float(np.quantile(h05, 0.975)),
        "bootstrap_unit": "subject_id_cluster",
        "bootstrap_reps": int(reps),
    }


def run_task(task: str, args: argparse.Namespace, rad: Optional[RadPredictor]) -> Tuple[List[Dict], List[Dict]]:
    print(f"[task] {task}", flush=True)
    manifest = build_yolo_dataset(task, args.link_mode, args.force_data)
    ok = manifest[manifest["status"].eq("ok")]
    link_counts = ok.groupby("link_status").size().to_dict() if "link_status" in ok else {}
    names = class_names(task)
    label_to_id = {name: i for i, name in enumerate(names)}
    train_rows, val_rows, eval_rows = task_rows(task, "train"), task_rows(task, "val"), task_rows(task, "eval")
    train_status = []
    candidate_sets_val = []
    candidate_sets_eval = []
    for model_name in args.models:
        rec = train_yolo(task, model_name, args)
        train_status.append(rec)
        if rec["status"] in {"ok", "cached"}:
            tag = rec["tag"]
            weights = Path(rec["weights"])
            candidate_sets_val.append(predict_candidates(task, weights, tag, "val", args))
            candidate_sets_eval.append(predict_candidates(task, weights, tag, "eval", args))
    if not candidate_sets_eval:
        raise RuntimeError(f"No detector candidates for task={task}")
    val_candidates = merge_candidates(*candidate_sets_val)
    eval_candidates = merge_candidates(*candidate_sets_eval)

    rad_val = rad_predict(task, "val", args, rad) if rad is not None else pd.DataFrame()
    rad_eval = rad_predict(task, "eval", args, rad) if rad is not None else pd.DataFrame()
    best_cfg, grid = tune_fusion(task, val_rows, val_candidates, label_to_id, rad_val, args.quick)
    grid.to_csv(MET / f"{task}_fusion_val_grid.csv", index=False)
    write_text(CFG / f"{task}_best_fusion.json", json.dumps(best_cfg, indent=2, ensure_ascii=False))

    yolo_eval = evaluate_yolo_class_only(task, eval_rows, eval_candidates, label_to_id, f"{task}_merged_yolo_class_only")
    fusion_eval = evaluate_fusion(task, eval_rows, eval_candidates, label_to_id, rad_eval, best_cfg, f"{task}_merged_yolo_rad_dino_rule_fusion")
    oracle_eval = candidate_oracle(task, eval_rows, eval_candidates, label_to_id)
    yolo_eval.to_csv(PRED / f"{task}_merged_yolo_class_only_eval_predictions.csv", index=False)
    fusion_eval.to_csv(PRED / f"{task}_merged_yolo_rad_dino_rule_fusion_eval_predictions.csv", index=False)
    oracle_eval.to_csv(PRED / f"{task}_candidate_oracle_eval_predictions.csv", index=False)
    rad_eval.to_csv(PRED / f"{task}_rad_dino_rule_eval_predictions.csv", index=False)

    summary = []
    summary.extend(baseline_summary_for_task(task))
    for df, method in [
        (yolo_eval, f"{task}_merged_yolo_class_only"),
        (rad_eval, f"{task}_rad_dino_rule_recomputed"),
        (fusion_eval, f"{task}_merged_yolo_rad_dino_rule_fusion"),
        (oracle_eval, f"{task}_candidate_oracle_upper_bound"),
    ]:
        row = metric(df, method, task)
        row["task"] = task
        row["source"] = "new_experiment"
        summary.append(row)
    boot = [
        paired_bootstrap(fusion_eval, rad_eval, f"{task}_fusion - rad_dino_rule", args.bootstrap_reps, args.seed),
        paired_bootstrap(fusion_eval, yolo_eval, f"{task}_fusion - merged_yolo_class_only", args.bootstrap_reps, args.seed),
    ]
    train_df = pd.DataFrame(train_status)
    train_df["task"] = task
    train_df["link_counts"] = json.dumps(link_counts, ensure_ascii=False)
    train_df.to_csv(MET / f"{task}_detector_train_status.csv", index=False)
    return summary, boot


def write_report(summary: pd.DataFrame, boot: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# Chest ImaGenome Merged YOLO + RAD-DINO Fusion V1",
        "",
        "## 한 줄 결론",
        "",
        "MS-CXR에서 썼던 merged detector 후보 + RAD-DINO rule-query box fusion을 Chest ImaGenome finding-region, device, anatomy task에 적용했다.",
        "",
        "## 실험 조건",
        "",
        f"- YOLO models: `{', '.join(args.models)}`",
        f"- epochs: `{args.epochs}`",
        f"- image size: `{args.imgsz}`",
        "- detector train: 각 task train split만 사용",
        "- fusion tuning: 각 task val split만 사용",
        "- final eval: 각 task eval split만 사용",
        "- RAD-DINO: 기존 rule-query heatmap head checkpoint 사용",
        "- Chest ImaGenome finding/device bbox는 weak region/reference bbox다. 병변 mask나 device contour로 쓰지 않는다.",
        "",
        "## Summary",
        "",
        summary.sort_values(["task", "mean_iou"], ascending=[True, False]).to_markdown(index=False),
        "",
        "## Bootstrap",
        "",
        boot.to_markdown(index=False),
        "",
        "## 해석",
        "",
        "- `candidate_oracle_upper_bound`는 실제 방법이 아니라 detector 후보 풀 상한선이다.",
        "- fusion이 RAD-DINO rule보다 낮으면, detector 후보가 해당 Chest ImaGenome target semantics와 잘 맞지 않거나 val fusion이 일반화되지 않은 것이다.",
        "- anatomy는 target 이름 자체에 위치 정보가 강하게 들어 있어 rule-context gain이 작을 수 있다.",
    ]
    write_text(REPORT / "README_KO.md", "\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="finding_region,device,anatomy")
    p.add_argument("--models", default="yolov8s.pt")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pred-conf", type=float, default=0.001)
    p.add_argument("--predict-batch", type=int, default=4)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--rad-batch", type=int, default=16)
    p.add_argument("--vfm-model", default="microsoft/rad-dino")
    p.add_argument("--link-mode", choices=["auto", "copy", "symlink", "hardlink"], default="auto")
    p.add_argument("--bootstrap-reps", type=int, default=500)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--force-data", action="store_true")
    p.add_argument("--force-train", action="store_true")
    p.add_argument("--force-predict", action="store_true")
    p.add_argument("--force-rad", action="store_true")
    args = p.parse_args()
    args.tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.quick:
        args.epochs = min(args.epochs, 15)
        args.bootstrap_reps = min(args.bootstrap_reps, 200)
    return args


def main() -> None:
    t0 = time.time()
    ensure_dirs()
    args = parse_args()
    write_text(CFG / "run_config.json", json.dumps(vars(args), indent=2, ensure_ascii=False))
    bad = [t for t in args.tasks if t not in TASKS]
    if bad:
        raise ValueError(f"Unknown tasks: {bad}")
    rad = RadPredictor(args.vfm_model, args.device)
    all_summary: List[Dict] = []
    all_boot: List[Dict] = []
    for task in args.tasks:
        s, b = run_task(task, args, rad)
        all_summary.extend(s)
        all_boot.extend(b)
    summary = pd.DataFrame(all_summary)
    boot = pd.DataFrame(all_boot)
    summary.to_csv(MET / "summary.csv", index=False)
    boot.to_csv(MET / "bootstrap_ci.csv", index=False)
    write_report(summary, boot, args)

    print(f"project_root={PROJECT_ROOT}")
    print(f"experiment={EXP_NAME}")
    print(f"tasks={','.join(args.tasks)}")
    print(f"models={','.join(args.models)}")
    for task in args.tasks:
        sub = summary[summary["task"].eq(task)].sort_values("mean_iou", ascending=False)
        if len(sub):
            top = sub.iloc[0]
            fusion = summary[(summary["task"].eq(task)) & (summary["method"].eq(f"{task}_merged_yolo_rad_dino_rule_fusion"))]
            print(f"{task}_best_method={top['method']}")
            print(f"{task}_best_mean_iou={float(top['mean_iou']):.6f}")
            if len(fusion):
                print(f"{task}_fusion_mean_iou={float(fusion.iloc[0]['mean_iou']):.6f}")
    print(f"summary_path={MET / 'summary.csv'}")
    print(f"report_path={REPORT / 'README_KO.md'}")
    print(f"elapsed_sec={time.time() - t0:.1f}")


if __name__ == "__main__":
    main()

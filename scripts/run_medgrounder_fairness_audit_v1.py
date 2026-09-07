#!/usr/bin/env python
"""MedGrounder fairness audit utilities for local MS-CXR p10-p19.

This script covers three paper-defense checks:
1. Audit whether an official MedGrounder split file is available locally.
2. Export MedGrounder predictions and compute our phrase-group metrics.
3. Fair-retrain MedGrounder on our local p10-p19 split starting from the
   Chest ImaGenome pretrain checkpoint, never from MS-CXR fine-tuned weights.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_repro.evaluator import exact_union_iou, hull_iou  # noqa: E402

MEDGROUNDER_ROOT = PROJECT_ROOT / "third_party" / "MedGrounder"
BASE_EXP = PROJECT_ROOT / "experiments" / "medgrounder_fairness_audit_v1"
BASE_REPORT = PROJECT_ROOT / "reports" / "medgrounder_fairness_audit_v1"
LOCAL_CONFIG = PROJECT_ROOT / "experiments" / "medgrounder_gmpg_baseline_v1" / "configs" / "conf_local_mscxr_p10p19.yaml"
GMPG_CSV = PROJECT_ROOT / "experiments" / "medgrounder_gmpg_baseline_v1" / "data" / "ms_cxr_p10p19_gmpg_medgrounder.csv"
PHRASE_AUDIT = PROJECT_ROOT / "experiments" / "medgrounder_gmpg_baseline_v1" / "metadata" / "ms_cxr_phrase_group_audit.csv"


def add_medgrounder_to_path() -> None:
    sys.path.insert(0, str(MEDGROUNDER_ROOT))


def load_cfg(config_path: Path = LOCAL_CONFIG):
    os.chdir(MEDGROUNDER_ROOT)
    base_cfg = OmegaConf.load(MEDGROUNDER_ROOT / "conf" / "base.yaml")
    datasets_cfg = OmegaConf.load(MEDGROUNDER_ROOT / "conf" / "datasets.yaml")
    run_cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.merge(datasets_cfg, base_cfg, run_cfg)
    cfg.datasets = ["gmpg_mscxr"]
    cfg.num_workers = 0
    return cfg


def ensure_dirs() -> None:
    for p in [
        BASE_EXP / "metrics",
        BASE_EXP / "predictions",
        BASE_EXP / "logs",
        BASE_EXP / "checkpoints",
        BASE_EXP / "metadata",
        BASE_EXP / "configs",
        BASE_REPORT,
    ]:
        p.mkdir(parents=True, exist_ok=True)


def cxcywh_to_xyxy_np(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    out = np.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], axis=1)
    return np.clip(out, 0.0, 1.0)


def iou_xyxy_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(br - tl, 0.0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def mask_iou_xyxy_np(pred: np.ndarray, gt: np.ndarray, size: int = 224) -> float:
    def draw(boxes: np.ndarray) -> np.ndarray:
        mask = np.zeros((size, size), dtype=bool)
        for x1, y1, x2, y2 in np.clip(boxes, 0.0, 1.0):
            ix1 = int(np.floor(x1 * size))
            iy1 = int(np.floor(y1 * size))
            ix2 = int(np.ceil(x2 * size))
            iy2 = int(np.ceil(y2 * size))
            ix1, ix2 = np.clip([ix1, ix2], 0, size)
            iy1, iy2 = np.clip([iy1, iy2], 0, size)
            if ix2 > ix1 and iy2 > iy1:
                mask[iy1:iy2, ix1:ix2] = True
        return mask

    pm = draw(pred)
    gm = draw(gt)
    union = np.logical_or(pm, gm).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pm, gm).sum() / union)


def greedy_set_f1(pred: np.ndarray, gt: np.ndarray, thr: float) -> tuple[float, float, float]:
    if len(pred) == 0 and len(gt) == 0:
        return 1.0, 1.0, 1.0
    if len(pred) == 0 or len(gt) == 0:
        return 0.0, 0.0, 0.0
    mat = iou_xyxy_np(pred, gt)
    pairs: list[tuple[float, int, int]] = []
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] >= thr:
                pairs.append((float(mat[i, j]), i, j))
    pairs.sort(reverse=True)
    used_p: set[int] = set()
    used_g: set[int] = set()
    tp = 0
    for _, i, j in pairs:
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            tp += 1
    precision = tp / len(pred) if len(pred) else 0.0
    recall = tp / len(gt) if len(gt) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def build_model_and_load(cfg, checkpoint_path: Path | None, device: torch.device):
    add_medgrounder_to_path()
    from model.medgrounder import build_medgrounder

    model, criterion, postprocessor = build_medgrounder(cfg)
    model.to(device)
    if checkpoint_path is not None and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
    return model, criterion, postprocessor


def get_loaders(cfg):
    add_medgrounder_to_path()
    from dataloaders.dataset_builder import get_dataloaders

    return get_dataloaders(cfg)


def run_wbf_if_needed(cfg, detections):
    add_medgrounder_to_path()
    from utils.box_utils import run_wbf

    if cfg.get("test_post_processing", False) and cfg.get("test_post_processing_params", {}).get("run_wbf", False):
        params = cfg.test_post_processing_params
        return run_wbf(
            detections,
            iou_threshold=params.get("wbf_iou_threshold", 0.1),
            skip_box_threshold=params.get("skip_box_thr", 0.0),
        )
    return detections


def evaluate_model(cfg, model, postprocessor, dataloader, device: torch.device, save_predictions: Path | None = None) -> dict[str, float]:
    add_medgrounder_to_path()
    from gmpg_metrics import MedGrounderMetrics

    model.eval()
    metrics = MedGrounderMetrics(pred_box_format="xywh", gt_box_format="xywh").to(device)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device)
            phrases = batch["phrases"]
            targets = [{k: v.to(device) for k, v in t.items()} for t in batch["targets"]]
            batch["targets"] = targets
            outputs = model(images, phrases)
            outputs = postprocessor(outputs)
            detections = run_wbf_if_needed(cfg, outputs["detections"])

            # MedGrounderMetrics converts boxes from cxcywh to xyxy in-place.
            # Keep a clean export copy before metric mutation so our downstream
            # phrase-group recomputation does not mix coordinate conventions.
            export_detections = [
                {
                    "boxes": det["boxes"].detach().cpu().clone(),
                    "scores": det.get("scores", torch.ones(len(det["boxes"]), device=det["boxes"].device)).detach().cpu().clone(),
                }
                for det in detections
            ]
            export_gt_boxes = [t["boxes"].detach().cpu().clone() for t in targets]
            metrics(detections, batch)

            if save_predictions is not None:
                for i, det in enumerate(export_detections):
                    pred_boxes = det["boxes"].numpy().tolist()
                    pred_scores = det["scores"].numpy().tolist()
                    gt_boxes = export_gt_boxes[i].numpy().tolist()
                    rows.append(
                        {
                            "group_id": batch["ids"][i],
                            "phrase": phrases[i],
                            "img_path": batch["img_paths"][i],
                            "category_names": "|".join(map(str, batch["category_names"][i])),
                            "n_pred": len(pred_boxes),
                            "n_gt": len(gt_boxes),
                            "pred_boxes_cxcywh": json.dumps(pred_boxes),
                            "pred_scores": json.dumps(pred_scores),
                            "gt_boxes_cxcywh": json.dumps(gt_boxes),
                            "box_format": "normalized_cxcywh",
                        }
                    )

    out = metrics.compute_metrics()
    out_dict = {k: float(v.item() if hasattr(v, "item") else v) for k, v in out.items()}
    if save_predictions is not None:
        save_predictions.parent.mkdir(parents=True, exist_ok=True)
        with save_predictions.open("w", newline="", encoding="utf-8") as f:
            fieldnames = list(rows[0].keys()) if rows else []
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return out_dict


def compute_our_phrase_metrics_from_predictions(pred_csv: Path, out_csv: Path, label: str) -> dict[str, Any]:
    import pandas as pd

    df = pd.read_csv(pred_csv)
    per_rows = []
    for _, r in df.iterrows():
        pred_c = np.array(json.loads(r["pred_boxes_cxcywh"]), dtype=np.float64).reshape(-1, 4)
        pred_scores = np.array(json.loads(r.get("pred_scores", "[]")), dtype=np.float64).reshape(-1)
        if len(pred_scores) == len(pred_c) and len(pred_c):
            order = np.argsort(-pred_scores, kind="stable")
            pred_c = pred_c[order]
        gt_c = np.array(json.loads(r["gt_boxes_cxcywh"]), dtype=np.float64).reshape(-1, 4)
        pred = cxcywh_to_xyxy_np(pred_c)
        gt = cxcywh_to_xyxy_np(gt_c)
        mat = iou_xyxy_np(pred, gt)
        gt_best = mat.max(axis=0) if mat.size else np.zeros(len(gt))
        pred_best = mat.max(axis=1) if mat.size else np.zeros(len(pred))
        top1_iou = float(mat[0].max()) if len(pred) and len(gt) else 0.0
        p03, r03, f03 = greedy_set_f1(pred, gt, 0.3)
        p05, r05, f05 = greedy_set_f1(pred, gt, 0.5)
        pred_list = pred.tolist()
        gt_list = gt.tolist()
        per_rows.append(
            {
                "method": label,
                "group_id": r["group_id"],
                "n_pred": len(pred),
                "n_gt_boxes": len(gt),
                "top1_iou": top1_iou,
                "coverage_mean_iou": float(gt_best.mean()) if len(gt_best) else 0.0,
                "enclosing_hull_iou": hull_iou(pred_list, gt_list),
                "exact_union_iou": exact_union_iou(pred_list, gt_list),
                "raster_union_iou_224": mask_iou_xyxy_np(pred, gt),
                "all_gt_hit_0_3": float((gt_best >= 0.3).mean()) if len(gt_best) else 0.0,
                "all_gt_hit_0_5": float((gt_best >= 0.5).mean()) if len(gt_best) else 0.0,
                "any_gt_hit_0_3": float((gt_best >= 0.3).any()) if len(gt_best) else 0.0,
                "any_gt_hit_0_5": float((gt_best >= 0.5).any()) if len(gt_best) else 0.0,
                "pred_precision_hit_0_3": float((pred_best >= 0.3).mean()) if len(pred_best) else 0.0,
                "pred_precision_hit_0_5": float((pred_best >= 0.5).mean()) if len(pred_best) else 0.0,
                "set_precision_0_3": p03,
                "set_recall_0_3": r03,
                "set_f1_0_3": f03,
                "set_precision_0_5": p05,
                "set_recall_0_5": r05,
                "set_f1_0_5": f05,
            }
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_rows).to_csv(out_csv, index=False)
    arr = pd.DataFrame(per_rows)
    summary = {
        "method": label,
        "n_groups": int(len(arr)),
        "n_gt_boxes": int(arr["n_gt_boxes"].sum()),
        "top1_mean_iou": float(arr["top1_iou"].mean()),
        "top1_hit_0_3": float((arr["top1_iou"] >= 0.3).mean()),
        "top1_hit_0_5": float((arr["top1_iou"] >= 0.5).mean()),
        "top1_mean_n_pred": float((arr["n_pred"] > 0).mean()),
        "coverage_mean_iou": float(arr["coverage_mean_iou"].mean()),
        "enclosing_hull_iou": float(arr["enclosing_hull_iou"].mean()),
        "exact_union_iou": float(arr["exact_union_iou"].mean()),
        "raster_union_iou_224": float(arr["raster_union_iou_224"].mean()),
        "union_iou": float(arr["raster_union_iou_224"].mean()),
        "all_gt_hit_0_3": float(arr["all_gt_hit_0_3"].mean()),
        "all_gt_hit_0_5": float(arr["all_gt_hit_0_5"].mean()),
        "any_gt_hit_0_3": float(arr["any_gt_hit_0_3"].mean()),
        "any_gt_hit_0_5": float(arr["any_gt_hit_0_5"].mean()),
        "set_f1_0_3": float(arr["set_f1_0_3"].mean()),
        "set_f1_0_5": float(arr["set_f1_0_5"].mean()),
        "mean_n_pred": float(arr["n_pred"].mean()),
    }
    return summary


def export_predictions(args) -> None:
    ensure_dirs()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = load_cfg(Path(getattr(args, "config_path", LOCAL_CONFIG)))
    _, _, test_loaders = get_loaders(cfg)
    test_loader = test_loaders["gmpg_mscxr"]
    checkpoints = {
        "released_finetune_ms": MEDGROUNDER_ROOT / "model_weight" / "medgrounder_finetune_ms.pth",
        "zero_shot_pretrain_imagenome": MEDGROUNDER_ROOT / "model_weight" / "medgrounder_pretrain_imagenome.pth",
    }
    summaries = []
    for name, ckpt in checkpoints.items():
        print(f"[export] loading/evaluating {name}: {ckpt}", flush=True)
        model, _, postprocessor = build_model_and_load(cfg, ckpt, device)
        pred_path = BASE_EXP / "predictions" / f"medgrounder_{name}_phrase_group_predictions.csv"
        native = evaluate_model(cfg, model, postprocessor, test_loader, device, save_predictions=pred_path)
        metric_path = BASE_EXP / "metrics" / f"medgrounder_{name}_our_phrase_group_metrics_per_group.csv"
        ours = compute_our_phrase_metrics_from_predictions(pred_path, metric_path, f"medgrounder_{name}")
        print(f"[export] {name} miou_all={native.get('miou_all'):.4f} our_union={ours.get('union_iou'):.4f}", flush=True)
        summaries.append({"checkpoint": str(ckpt), **native, **{f"our_{k}": v for k, v in ours.items() if k != "method"}})
    with (BASE_EXP / "metrics" / "medgrounder_exported_prediction_metric_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)


def audit_official_overlap() -> None:
    ensure_dirs()
    candidates: list[Path] = []
    roots = [
        MEDGROUNDER_ROOT,
        PROJECT_ROOT / "experiments" / "medgrounder_gmpg_baseline_v1",
        PROJECT_ROOT / "reports" / "medgrounder_gmpg_baseline_v1",
    ]
    patterns = ["*gmpg*", "*MedGrounder*", "*mscxr*", "*MS_CXR*", "*MS-CXR*", "*split*"]
    for root in roots:
        if not root.exists():
            continue
        for pat in patterns:
            try:
                for p in root.rglob(pat):
                    if p.is_file() and p.suffix.lower() in {".csv", ".json", ".pth", ".pkl", ".yaml"}:
                        candidates.append(p)
            except Exception:
                pass
    downloads = Path(r"C:/Users/_idal/Downloads")
    if downloads.exists():
        for p in downloads.glob("*"):
            if p.is_file() and any(k in p.name.lower() for k in ["gmpg", "medgrounder", "mscxr", "ms_cxr", "ms-cxr", "split", "drive-download"]):
                candidates.append(p)
    official_like = []
    for p in sorted(set(candidates)):
        name = p.name.lower()
        text = str(p).lower()
        if any(k in text for k in ["official", "split", "train", "val", "test", "gmpg", "medgrounder"]):
            official_like.append(p)

    out_csv = BASE_EXP / "metadata" / "official_split_artifact_search.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "size", "usable_for_official_overlap", "note"])
        writer.writeheader()
        for p in official_like:
            usable = False
            note = "candidate file, not confirmed official split"
            if p == GMPG_CSV or "medgrounder_gmpg_baseline_v1" in str(p):
                note = "local converted p10-p19 split, not MedGrounder official split"
            if "model_weight" in str(p):
                note = "checkpoint weight, not split file"
            writer.writerow({"path": str(p), "size": p.stat().st_size, "usable_for_official_overlap": usable, "note": note})

    report = BASE_REPORT / "OFFICIAL_SPLIT_OVERLAP_AUDIT_KO.md"
    lines = [
        "# MedGrounder official split overlap audit",
        "",
        "## 결론",
        "",
        "현재 로컬 MedGrounder repo, Downloads, 프로젝트 산출물 안에서는 MedGrounder official MS-CXR train/val/test split artifact를 찾지 못했다.",
        "다운로드된 Google Drive zip도 checkpoint weight만 포함하고 있었고 official split CSV/PTH는 없었다.",
        "",
        "따라서 `medgrounder_finetune_ms.pth`가 우리 p10-p19 eval을 봤는지 직접 overlap count로 확정할 수 없다.",
        "이 경우 논문에서는 해당 released checkpoint 결과를 main fair result가 아니라 `released-checkpoint reference, official overlap not audited`로만 써야 한다.",
        "",
        f"- search inventory: `{out_csv}`",
        f"- local converted CSV: `{GMPG_CSV}`",
        "- local converted CSV는 우리가 만든 p10-p19 split이라 official overlap 감사의 기준 파일이 아니다.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    clean_lines = [
        "# MedGrounder official split overlap audit",
        "",
        "## 결론",
        "",
        "현재 로컬 MedGrounder repo, Downloads, 프로젝트 산출물 안에서는 MedGrounder official MS-CXR train/val/test split artifact를 찾지 못했다.",
        "다운로드된 Google Drive zip에는 checkpoint weight만 있었고 official split CSV/PTH는 없었다.",
        "",
        "따라서 `medgrounder_finetune_ms.pth`가 우리 p10-p19 eval을 학습 중에 봤는지 직접 overlap count로 확정할 수 없다.",
        "이 경우 논문에서는 해당 released checkpoint 결과를 main fair result가 아니라 `released-checkpoint reference, official overlap not audited`로만 써야 한다.",
        "",
        f"- search inventory: `{out_csv}`",
        f"- local converted CSV: `{GMPG_CSV}`",
        "- local converted CSV는 우리가 만든 p10-p19 split이므로 official overlap 감사 기준 파일이 아니다.",
    ]
    report.write_text("\n".join(clean_lines) + "\n", encoding="utf-8")


def train_fair(args) -> None:
    ensure_dirs()
    add_medgrounder_to_path()
    from dataloaders.dataset_builder import get_dataloaders

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = load_cfg(Path(getattr(args, "config_path", LOCAL_CONFIG)))
    cfg.batch_size = args.batch_size
    cfg.num_epochs = args.epochs
    cfg.optimiser_kwargs.lr = args.lr
    train_loader, val_loader, test_loaders = get_dataloaders(cfg)
    test_loader = test_loaders["gmpg_mscxr"]

    init_ckpt = MEDGROUNDER_ROOT / "model_weight" / "medgrounder_pretrain_imagenome.pth"
    model, criterion, postprocessor = build_model_and_load(cfg, init_ckpt, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=float(cfg.optimiser_kwargs.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    run_dir = BASE_EXP / "checkpoints" / f"fair_retrain_pretrain_imagenome_s{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_metric = -1.0
    best_epoch = -1
    log_rows = []
    torch.manual_seed(args.seed)

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        n_batches = 0
        for batch in train_loader:
            images = batch["images"].to(device)
            phrases = batch["phrases"]
            targets = [{k: v.to(device) for k, v in t.items()} for t in batch["targets"]]
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                outputs = model(images, phrases)
                loss_dict = criterion(outputs, targets)
                loss = sum(loss_dict[k] * criterion.weight_dict.get(k, 1.0) for k in loss_dict if k in criterion.weight_dict)
            scaler.scale(loss).backward()
            if args.clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach().cpu())
            n_batches += 1

        val_metrics = evaluate_model(cfg, model, postprocessor, val_loader, device)
        val_miou = val_metrics["miou_all"]
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(n_batches, 1),
            "val_miou_all": val_miou,
            "val_precision_f1eq1_overall@0_5": val_metrics["precision_f1eq1_overall@0_5"],
            "elapsed_sec": time.time() - t0,
        }
        log_rows.append(row)
        print(json.dumps(row), flush=True)
        if val_miou > best_metric:
            best_metric = val_miou
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_miou_all": val_miou, "cfg": OmegaConf.to_container(cfg)}, run_dir / "best.pth")
        torch.save({"model": model.state_dict(), "epoch": epoch, "val_miou_all": val_miou}, run_dir / "last.pth")

    with (run_dir / "train_log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    best_ckpt = run_dir / "best.pth"
    model, _, postprocessor = build_model_and_load(cfg, best_ckpt, device)
    pred_path = BASE_EXP / "predictions" / f"medgrounder_fair_retrain_s{args.seed}_eval_predictions.csv"
    test_native = evaluate_model(cfg, model, postprocessor, test_loader, device, save_predictions=pred_path)
    per_group_path = BASE_EXP / "metrics" / f"medgrounder_fair_retrain_s{args.seed}_our_phrase_group_metrics_per_group.csv"
    ours_metrics = compute_our_phrase_metrics_from_predictions(pred_path, per_group_path, f"medgrounder_fair_retrain_s{args.seed}")
    summary = {
        "method": "medgrounder_fair_retrain_from_imagenome_pretrain",
        "seed": args.seed,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_miou_all": best_metric,
        **test_native,
        **{f"our_{k}": v for k, v in ours_metrics.items() if k != "method"},
        "init_checkpoint": str(init_ckpt),
        "forbidden_checkpoint": str(MEDGROUNDER_ROOT / "model_weight" / "medgrounder_finetune_ms.pth"),
    }
    with (BASE_EXP / "metrics" / f"medgrounder_fair_retrain_s{args.seed}_eval_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def write_summary_report() -> None:
    ensure_dirs()
    lines = ["# MedGrounder fairness audit v1", ""]
    zero = BASE_EXP / "metrics" / "evaluation_results_pretrain_imagenome_zero_shot.json"
    rel = PROJECT_ROOT / "experiments" / "medgrounder_gmpg_baseline_v1" / "metrics" / "evaluation_results_gmpg_mscxr_20260703_003755.json"
    if rel.exists():
        r = json.loads(rel.read_text())
        lines += ["## Released finetune_ms reference", "", f"- miou_all: {r['miou_all']:.4f}", f"- miou_single: {r['miou_single']:.4f}", f"- miou_multi: {r['miou_multi']:.4f}", ""]
    if zero.exists():
        z = json.loads(zero.read_text())
        lines += ["## Chest ImaGenome pretrain zero-shot", "", f"- miou_all: {z['miou_all']:.4f}", f"- miou_single: {z['miou_single']:.4f}", f"- miou_multi: {z['miou_multi']:.4f}", ""]
    fair_files = sorted((BASE_EXP / "metrics").glob("medgrounder_fair_retrain_s*_eval_summary.json"))
    if fair_files:
        lines += ["## Fair retrain", ""]
        for p in fair_files:
            s = json.loads(p.read_text())
            lines.append(f"- seed {s['seed']}: miou_all={s['miou_all']:.4f}, our_union_iou={s['our_union_iou']:.4f}, best_epoch={s['best_epoch']}")
        lines.append("")
    lines += [
        "## 해석 원칙",
        "",
        "- `medgrounder_finetune_ms.pth`는 MS-CXR fine-tuned released checkpoint라 main fair result로 쓰지 않는다.",
        "- `medgrounder_pretrain_imagenome.pth` zero-shot은 MS-CXR fine-tuning 없이 보는 더 안전한 reference다.",
        "- fair retrain은 `pretrain_imagenome`에서 시작하고 `finetune_ms`는 금지 checkpoint로 둔다.",
        "- MedGrounder native `miou_all`은 phrase-group mask/union metric이며 row-level IoU와 직접 비교하지 않는다.",
    ]
    (BASE_REPORT / "README_KO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    clean_tail = [
        "## 해석 원칙",
        "",
        "- `medgrounder_finetune_ms.pth`는 MS-CXR fine-tuned released checkpoint라서 main fair result로 쓰지 않는다.",
        "- `medgrounder_pretrain_imagenome.pth` zero-shot은 MS-CXR fine-tuning 없이 보는 안전한 reference다.",
        "- fair retrain은 `pretrain_imagenome`에서 시작하고 `finetune_ms`는 금지 checkpoint로 둔다.",
        "- MedGrounder native `miou_all`은 phrase-group mask/union metric이며 row-level IoU와 직접 비교하지 않는다.",
    ]
    cleaned = []
    for line in lines:
        if line.startswith("## ?"):
            continue
        if line.startswith("- `medgrounder_") or line.startswith("- fair retrain") or line.startswith("- MedGrounder native"):
            continue
        cleaned.append(line)
    cleaned += clean_tail
    (BASE_REPORT / "README_KO.md").write_text("\n".join(cleaned) + "\n", encoding="utf-8")


def main() -> None:
    global BASE_EXP, BASE_REPORT, LOCAL_CONFIG
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-official-overlap", action="store_true")
    parser.add_argument("--export-predictions", action="store_true")
    parser.add_argument("--train-fair", action="store_true")
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--config-path", default=str(LOCAL_CONFIG))
    parser.add_argument("--base-exp", default=str(BASE_EXP))
    parser.add_argument("--base-report", default=str(BASE_REPORT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--clip-norm", type=float, default=0.1)
    args = parser.parse_args()
    BASE_EXP = Path(args.base_exp).resolve()
    BASE_REPORT = Path(args.base_report).resolve()
    LOCAL_CONFIG = Path(args.config_path).resolve()
    ensure_dirs()
    if args.audit_official_overlap:
        audit_official_overlap()
    if args.export_predictions:
        export_predictions(args)
    if args.train_fair:
        train_fair(args)
    if args.write_report:
        write_summary_report()


if __name__ == "__main__":
    main()

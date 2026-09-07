#!/usr/bin/env python
"""Audit and smoke-train an MDETR-style baseline on the local MS-CXR split.

This script is intentionally conservative.  The AGPT repository provides
released MS-CXR-finetuned MDETR/TransVG weights, but those are reference-only
for our paper unless official overlap is audited.  The `smoke-train` mode below
does not load those released AGPT weights by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGPT_ROOT = PROJECT_ROOT / "third_party" / "AGPT"
EXP_ROOT = PROJECT_ROOT / "experiments" / "ms_cxr_mdetr_fair_baseline_v1"
REPORT_ROOT = PROJECT_ROOT / "reports" / "ms_cxr_mdetr_fair_baseline_v1"
SINGLE_BOX_ROOT = PROJECT_ROOT / "experiments" / "agpt_baseline_v1" / "data" / "single_box_p10p19"


def configure_runtime_paths(cli: argparse.Namespace) -> None:
    """Route a controlled run to isolated data and output directories."""

    global EXP_ROOT, REPORT_ROOT, SINGLE_BOX_ROOT
    if getattr(cli, "output_root", ""):
        EXP_ROOT = Path(cli.output_root).resolve()
        REPORT_ROOT = EXP_ROOT / "report"
    if getattr(cli, "dataset_root", ""):
        SINGLE_BOX_ROOT = Path(cli.dataset_root).resolve()


def add_agpt_to_path() -> None:
    if str(AGPT_ROOT) not in sys.path:
        sys.path.insert(0, str(AGPT_ROOT))


def ensure_dirs() -> None:
    for path in [
        EXP_ROOT / "metadata",
        EXP_ROOT / "metrics",
        EXP_ROOT / "logs",
        EXP_ROOT / "checkpoints",
        REPORT_ROOT,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def file_size_mb(path: Path) -> float | None:
    return round(path.stat().st_size / (1024 * 1024), 2) if path.exists() else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_audit() -> None:
    ensure_dirs()
    rows: list[dict[str, Any]] = []
    candidate_files = [
        AGPT_ROOT / "README.md",
        AGPT_ROOT / "demo.py",
        AGPT_ROOT / "Args.py",
        AGPT_ROOT / "dataloader" / "mscxr_dataloader.py",
        AGPT_ROOT / "models" / "mdetr" / "mdetr.py",
        AGPT_ROOT / "models" / "mdetr" / "transformer.py",
        AGPT_ROOT / "model_weight" / "mdetr.pth",
        AGPT_ROOT / "model_weight" / "transvg.pth",
        SINGLE_BOX_ROOT / "dataset_info.json",
        SINGLE_BOX_ROOT / "split_root" / "MS_CXR" / "MS_CXR_train.pth",
        SINGLE_BOX_ROOT / "split_root" / "MS_CXR" / "MS_CXR_val.pth",
        SINGLE_BOX_ROOT / "split_root" / "MS_CXR" / "MS_CXR_test.pth",
    ]
    for path in candidate_files:
        rows.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "size_mb": file_size_mb(path),
                "role": infer_role(path),
            }
        )

    status_rows = [
        {
            "item": "AGPT repository",
            "status": "available" if AGPT_ROOT.exists() else "missing",
            "notes": "Local repo contains MDETR/TransVG model code and demo-oriented released checkpoint loading.",
        },
        {
            "item": "AGPT MDETR released weight",
            "status": "available" if (AGPT_ROOT / "model_weight" / "mdetr.pth").exists() else "missing",
            "notes": "MS-CXR-finetuned released checkpoint; reference-only, not a fair initialization.",
        },
        {
            "item": "AGPT full training entrypoint",
            "status": "not_found",
            "notes": "No train.py was found. Model, dataloader, optimizer, and loss components exist, so a wrapper is needed.",
        },
        {
            "item": "Strict single-box split",
            "status": "available" if (SINGLE_BOX_ROOT / "dataset_info.json").exists() else "missing",
            "notes": "MS-CXR p10-p19 single-box protocol: train 638 / val 87 / test 163.",
        },
        {
            "item": "Fair MDETR retrain",
            "status": "possible_with_wrapper",
            "notes": "Use our split, do not load mdetr.pth/transvg.pth. Start with box-only smoke training, then implement official token/contrastive losses if stable.",
        },
    ]
    write_csv(EXP_ROOT / "metadata" / "local_mdetr_agpt_file_audit.csv", rows)
    write_csv(EXP_ROOT / "metadata" / "mdetr_fairness_status.csv", status_rows)
    print(json.dumps({"audit_files": len(rows), "status_rows": len(status_rows)}, indent=2))


def infer_role(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".pth") and "model_weight" in str(path):
        return "released_checkpoint_reference_only"
    if name.endswith(".pth"):
        return "dataset_split"
    if name == "mdetr.py":
        return "model_and_criterion_components"
    if name == "transformer.py":
        return "text_image_fusion_transformer"
    if name == "mscxr_dataloader.py":
        return "official_style_agpt_dataset_loader"
    if name == "dataset_info.json":
        return "local_single_box_split_manifest"
    return "code_or_documentation"


class SimpleMSCXRDataset(Dataset):
    """Read AGPT-style MS_CXR_*.pth without loading an unused ClinicalBERT model."""

    def __init__(self, args: Any, split: str) -> None:
        add_agpt_to_path()
        from dataloader.mscxr_dataloader import get_transforms
        from utils.box_utils import xyxy2xywh

        split_name = "test" if split == "eval" else split
        self.args = args
        self.split = split_name
        self.annotations = torch.load(Path(args.anno_dir) / f"MS_CXR_{split_name}.pth", map_location="cpu")
        self.img_dir = Path(args.img_dir)
        transform_split = "train" if split_name == "train" and not getattr(args, "disable_train_aug", True) else "val"
        self.transform = get_transforms(args, transform_split)
        self.xyxy2xywh = xyxy2xywh

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        _, _, category_id, img_rel_path, bbox_xywh, height, width, phrase = self.annotations[idx]
        img_path = self.img_dir / img_rel_path
        image = np.array(Image.open(img_path).convert("RGB"))
        bbox = np.array(bbox_xywh, dtype=np.float32)
        bbox_xyxy = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], dtype=np.float32)
        transformed = self.transform(image=image, bboxes=[bbox_xyxy], class_labels=[category_id])
        image_tensor = transformed["image"]
        transformed_bbox = torch.tensor(transformed["bboxes"], dtype=torch.float32)
        bbox_norm_cxcywh = self.xyxy2xywh(transformed_bbox) / torch.tensor(
            [self.args.imsize, self.args.imsize, self.args.imsize, self.args.imsize],
            dtype=torch.float32,
        )
        return {
            "image": image_tensor,
            "bbox": bbox_norm_cxcywh,
            "phrase": phrase,
            "img_path": str(img_path),
            "category_id": int(category_id),
        }


def simple_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    add_agpt_to_path()
    from utils.misc import NestedTensor

    images = torch.stack([item["image"] for item in batch], dim=0)
    masks = torch.zeros_like(images[:, 0, :, :], dtype=torch.bool)
    return {
        "images": NestedTensor(images, masks),
        "bbox": torch.stack([item["bbox"] for item in batch], dim=0),
        "phrases": [item["phrase"] for item in batch],
        "img_paths": [item["img_path"] for item in batch],
        "category_ids": [item["category_id"] for item in batch],
    }


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def pairwise_iou_diag(pred_cxcywh: torch.Tensor, tgt_cxcywh: torch.Tensor) -> torch.Tensor:
    pred = cxcywh_to_xyxy(pred_cxcywh).clamp(0, 1)
    tgt = cxcywh_to_xyxy(tgt_cxcywh).clamp(0, 1)
    x1 = torch.maximum(pred[:, 0], tgt[:, 0])
    y1 = torch.maximum(pred[:, 1], tgt[:, 1])
    x2 = torch.minimum(pred[:, 2], tgt[:, 2])
    y2 = torch.minimum(pred[:, 3], tgt[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_p = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (pred[:, 3] - pred[:, 1]).clamp(min=0)
    area_t = (tgt[:, 2] - tgt[:, 0]).clamp(min=0) * (tgt[:, 3] - tgt[:, 1]).clamp(min=0)
    return inter / (area_p + area_t - inter + 1e-8)


def box_loss(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    add_agpt_to_path()
    from utils.box_utils import generalized_box_iou

    pred = pred.squeeze(1).clamp(0, 1)
    target = target.squeeze(1).clamp(0, 1)
    l1 = F.smooth_l1_loss(pred, target)
    giou_mat = generalized_box_iou(cxcywh_to_xyxy(pred), cxcywh_to_xyxy(target))
    giou_loss = 1 - torch.diag(giou_mat).mean()
    loss = l1 + giou_loss
    return loss, torch.diag(giou_mat).detach()


@dataclass
class EvalSummary:
    split: str
    n: int
    mean_iou: float
    median_iou: float
    hit_0_1: float
    hit_0_3: float
    hit_0_5: float


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
    prediction_path: Path | None = None,
) -> EvalSummary:
    model.eval()
    all_ious: list[float] = []
    pred_rows: list[dict[str, Any]] = []
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = batch["images"].to(device)
        targets = batch["bbox"].to(device)
        outputs = model(images, batch["phrases"])
        pred = outputs["pred_boxes"]
        ious = pairwise_iou_diag(pred.squeeze(1), targets.squeeze(1))
        all_ious.extend([float(x) for x in ious.detach().cpu().tolist()])
        if prediction_path is not None:
            pred_cpu = pred.squeeze(1).detach().cpu().clamp(0, 1).numpy()
            tgt_cpu = targets.squeeze(1).detach().cpu().clamp(0, 1).numpy()
            iou_cpu = ious.detach().cpu().numpy()
            for i, phrase in enumerate(batch["phrases"]):
                pred_rows.append(
                    {
                        "row_index": len(pred_rows),
                        "img_path": batch["img_paths"][i],
                        "phrase": phrase,
                        "category_id": batch["category_ids"][i],
                        "pred_cx": float(pred_cpu[i, 0]),
                        "pred_cy": float(pred_cpu[i, 1]),
                        "pred_w": float(pred_cpu[i, 2]),
                        "pred_h": float(pred_cpu[i, 3]),
                        "gt_cx": float(tgt_cpu[i, 0]),
                        "gt_cy": float(tgt_cpu[i, 1]),
                        "gt_w": float(tgt_cpu[i, 2]),
                        "gt_h": float(tgt_cpu[i, 3]),
                        "iou": float(iou_cpu[i]),
                        "hit_0_1": int(iou_cpu[i] >= 0.1),
                        "hit_0_3": int(iou_cpu[i] >= 0.3),
                        "hit_0_5": int(iou_cpu[i] >= 0.5),
                    }
                )
    arr = np.asarray(all_ious, dtype=np.float32)
    if prediction_path is not None:
        write_csv(prediction_path, pred_rows)
    if arr.size == 0:
        return EvalSummary("eval", 0, math.nan, math.nan, math.nan, math.nan, math.nan)
    return EvalSummary(
        split="eval",
        n=int(arr.size),
        mean_iou=float(arr.mean()),
        median_iou=float(np.median(arr)),
        hit_0_1=float((arr >= 0.1).mean()),
        hit_0_3=float((arr >= 0.3).mean()),
        hit_0_5=float((arr >= 0.5).mean()),
    )


def configure_agpt_args(cli: argparse.Namespace) -> Any:
    add_agpt_to_path()
    from Args import build_args

    dataset_info = json.loads((SINGLE_BOX_ROOT / "dataset_info.json").read_text(encoding="utf-8"))
    args = build_args("mdetr", "ms_cxr")
    args.img_dir = dataset_info["image_root"]
    args.anno_dir = str(SINGLE_BOX_ROOT / "split_root" / "MS_CXR")
    args.device = cli.device
    args.batch_size = cli.batch_size
    args.num_workers = cli.num_workers
    args.imsize = cli.image_size
    args.epochs = cli.epochs
    args.lr = cli.lr
    args.lr_backbone = cli.lr_backbone
    args.text_encoder_lr = cli.text_encoder_lr
    args.freeze_text_encoder = cli.freeze_text_encoder
    args.num_queries = 1
    args.aux_loss = False
    args.contrastive_align_loss = False
    args.contrastive_loss = False
    args.disable_train_aug = cli.disable_train_aug
    return args


def smoke_train(cli: argparse.Namespace) -> None:
    ensure_dirs()
    add_agpt_to_path()
    from models.mdetr import build_mdetr_model

    args = configure_agpt_args(cli)
    device = torch.device(cli.device if torch.cuda.is_available() and cli.device.startswith("cuda") else "cpu")
    args.device = str(device)
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)

    train_ds = SimpleMSCXRDataset(args, "train")
    val_ds = SimpleMSCXRDataset(args, "val")
    train_loader = DataLoader(
        train_ds,
        batch_size=cli.batch_size,
        shuffle=True,
        num_workers=cli.num_workers,
        collate_fn=simple_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cli.batch_size,
        shuffle=False,
        num_workers=cli.num_workers,
        collate_fn=simple_collate,
    )

    model = build_mdetr_model(args)
    if cli.init_released_reference:
        checkpoint = torch.load(AGPT_ROOT / "model_weight" / "mdetr.pth", map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        init_note = "released_ms_cxr_checkpoint_reference_only"
    else:
        init_note = "no_agpt_released_checkpoint_random_head_with_pretrained_backbone_text_encoder"
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cli.lr, weight_decay=cli.weight_decay)
    log_rows: list[dict[str, Any]] = []
    best_val = -1.0
    best_path = EXP_ROOT / "checkpoints" / f"mdetr_smoke_s{cli.seed}_best.pt"
    start = time.time()
    error_text = ""
    try:
        for epoch in range(cli.epochs):
            model.train()
            epoch_losses: list[float] = []
            for batch_idx, batch in enumerate(train_loader):
                if cli.max_train_batches is not None and batch_idx >= cli.max_train_batches:
                    break
                images = batch["images"].to(device)
                targets = batch["bbox"].to(device)
                outputs = model(images, batch["phrases"])
                loss, _ = box_loss(outputs["pred_boxes"], targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cli.clip_max_norm)
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
            val_summary = evaluate(model, val_loader, device, max_batches=cli.max_val_batches)
            mean_loss = float(np.mean(epoch_losses)) if epoch_losses else math.nan
            row = {
                "epoch": epoch + 1,
                "train_loss": mean_loss,
                "val_mean_iou": val_summary.mean_iou,
                "val_hit_0_3": val_summary.hit_0_3,
                "val_n": val_summary.n,
                "init_note": init_note,
            }
            log_rows.append(row)
            if val_summary.mean_iou == val_summary.mean_iou and val_summary.mean_iou > best_val:
                best_val = val_summary.mean_iou
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch + 1,
                        "best_val_mean_iou": best_val,
                        "args": vars(cli),
                        "init_note": init_note,
                    },
                    best_path,
                )
    except Exception:
        error_text = traceback.format_exc()
        (EXP_ROOT / "logs" / f"mdetr_smoke_s{cli.seed}_error.log").write_text(error_text, encoding="utf-8")
    elapsed = time.time() - start
    write_csv(EXP_ROOT / "metrics" / f"mdetr_smoke_train_s{cli.seed}_log.csv", log_rows)
    status = {
        "seed": cli.seed,
        "epochs_requested": cli.epochs,
        "completed_epochs": len(log_rows),
        "best_val_mean_iou": best_val,
        "checkpoint_path": str(best_path) if best_path.exists() else "",
        "elapsed_sec": elapsed,
        "init_note": init_note,
        "error": bool(error_text),
    }
    (EXP_ROOT / "metadata" / f"mdetr_smoke_train_s{cli.seed}_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(status, indent=2, ensure_ascii=False))
    if error_text:
        raise RuntimeError("MDETR smoke training failed; see log file.")


def full_train_box_only(cli: argparse.Namespace) -> None:
    ensure_dirs()
    add_agpt_to_path()
    from models.mdetr import build_mdetr_model

    args = configure_agpt_args(cli)
    device = torch.device(cli.device if torch.cuda.is_available() and cli.device.startswith("cuda") else "cpu")
    args.device = str(device)
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)

    train_ds = SimpleMSCXRDataset(args, "train")
    val_ds = SimpleMSCXRDataset(args, "val")
    eval_ds = SimpleMSCXRDataset(args, "eval")
    train_loader = DataLoader(train_ds, batch_size=cli.batch_size, shuffle=True, num_workers=cli.num_workers, collate_fn=simple_collate)
    val_loader = DataLoader(val_ds, batch_size=cli.batch_size, shuffle=False, num_workers=cli.num_workers, collate_fn=simple_collate)
    eval_loader = DataLoader(eval_ds, batch_size=cli.batch_size, shuffle=False, num_workers=cli.num_workers, collate_fn=simple_collate)

    model = build_mdetr_model(args)
    init_note = "no_agpt_released_checkpoint_random_head_with_pretrained_backbone_text_encoder"
    if cli.init_released_reference:
        raise ValueError("Released AGPT checkpoint is forbidden for fair full training.")
    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cli.lr, weight_decay=cli.weight_decay)
    log_rows: list[dict[str, Any]] = []
    best_val = -1.0
    best_epoch = -1
    run_name = cli.run_name or f"mdetr_box_only_s{cli.seed}"
    best_path = EXP_ROOT / "checkpoints" / f"{run_name}_best.pt"
    last_path = EXP_ROOT / "checkpoints" / f"{run_name}_last.pt"
    start = time.time()

    for epoch in range(cli.epochs):
        model.train()
        epoch_losses: list[float] = []
        epoch_ious: list[float] = []
        for batch_idx, batch in enumerate(train_loader):
            if cli.max_train_batches is not None and batch_idx >= cli.max_train_batches:
                break
            images = batch["images"].to(device)
            targets = batch["bbox"].to(device)
            outputs = model(images, batch["phrases"])
            loss, giou_diag = box_loss(outputs["pred_boxes"], targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cli.clip_max_norm)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            with torch.no_grad():
                ious = pairwise_iou_diag(outputs["pred_boxes"].detach().squeeze(1), targets.squeeze(1))
            epoch_ious.extend([float(x) for x in ious.detach().cpu().tolist()])

        val_summary = evaluate(model, val_loader, device, max_batches=cli.max_val_batches)
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else math.nan
        train_mean_iou = float(np.mean(epoch_ious)) if epoch_ious else math.nan
        row = {
            "epoch": epoch + 1,
            "train_loss": mean_loss,
            "train_mean_iou_batch": train_mean_iou,
            "val_mean_iou": val_summary.mean_iou,
            "val_median_iou": val_summary.median_iou,
            "val_hit_0_1": val_summary.hit_0_1,
            "val_hit_0_3": val_summary.hit_0_3,
            "val_hit_0_5": val_summary.hit_0_5,
            "val_n": val_summary.n,
            "init_note": init_note,
        }
        log_rows.append(row)
        write_csv(EXP_ROOT / "metrics" / f"{run_name}_train_log.csv", log_rows)
        if val_summary.mean_iou == val_summary.mean_iou and val_summary.mean_iou > best_val:
            best_val = val_summary.mean_iou
            best_epoch = epoch + 1
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": best_epoch,
                    "best_val_mean_iou": best_val,
                    "args": vars(cli),
                    "init_note": init_note,
                },
                best_path,
            )
        torch.save({"model": model.state_dict(), "epoch": epoch + 1, "args": vars(cli)}, last_path)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model"], strict=True)
    pred_path = EXP_ROOT / "predictions" / f"{run_name}_eval_predictions.csv"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    eval_summary = evaluate(model, eval_loader, device, prediction_path=pred_path)
    elapsed = time.time() - start
    summary = {
        "method": "mdetr_box_only_fair_retrain",
        "run_name": run_name,
        "seed": cli.seed,
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "eval_rows": len(eval_ds),
        "epochs": cli.epochs,
        "batch_size": cli.batch_size,
        "lr": cli.lr,
        "image_size": cli.image_size,
        "best_epoch": best_epoch,
        "best_val_mean_iou": best_val,
        "eval_mean_iou": eval_summary.mean_iou,
        "eval_median_iou": eval_summary.median_iou,
        "eval_hit_0_1": eval_summary.hit_0_1,
        "eval_hit_0_3": eval_summary.hit_0_3,
        "eval_hit_0_5": eval_summary.hit_0_5,
        "eval_n": eval_summary.n,
        "trainable_params": trainable,
        "total_params": total,
        "elapsed_sec": elapsed,
        "released_agpt_weight_used": False,
        "checkpoint_path": str(best_path),
        "prediction_path": str(pred_path),
        "note": "Box-only MDETR-style fair retrain; not official full MDETR token-alignment reproduction.",
    }
    write_csv(EXP_ROOT / "metrics" / f"{run_name}_eval_summary.csv", [summary])
    (EXP_ROOT / "metadata" / f"{run_name}_status.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def build_token_positive_map(phrases: list[str], tokenizer: Any, num_classes: int, device: torch.device) -> torch.Tensor:
    tokenized = tokenizer.batch_encode_plus(phrases, padding="longest", return_tensors="pt")
    attention = tokenized.attention_mask
    positive_map = torch.zeros((len(phrases), num_classes + 1), dtype=torch.float32)
    for i in range(len(phrases)):
        length = int(attention[i].sum().item())
        # RoBERTa usually has <s> at 0 and </s> at length-1. Supervise all phrase tokens.
        token_ids = list(range(1, max(1, length - 1)))
        token_ids = [idx for idx in token_ids if idx < num_classes]
        if not token_ids:
            token_ids = [0]
        weight = 1.0 / len(token_ids)
        for idx in token_ids:
            positive_map[i, idx] = weight
    return positive_map.to(device)


def build_mdetr_targets(
    batch: dict[str, Any],
    device: torch.device,
    include_tokens_positive: bool = False,
) -> list[dict[str, Any]]:
    boxes = batch["bbox"].to(device).squeeze(1).clamp(0, 1)
    targets: list[dict[str, Any]] = []
    for i in range(boxes.shape[0]):
        target: dict[str, Any] = {
            "boxes": boxes[i : i + 1],
            "labels": torch.zeros(1, dtype=torch.long, device=device),
        }
        if include_tokens_positive:
            phrase = batch["phrases"][i]
            target["tokens_positive"] = [[(0, len(phrase))]]
        targets.append(target)
    return targets


def full_train_token_box(cli: argparse.Namespace) -> None:
    ensure_dirs()
    add_agpt_to_path()
    from models.mdetr.mdetr import build

    args = configure_agpt_args(cli)
    args.contrastive_align_loss = cli.contrastive_align
    args.contrastive_loss = False
    args.aux_loss = False
    device = torch.device(cli.device if torch.cuda.is_available() and cli.device.startswith("cuda") else "cpu")
    args.device = str(device)
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)

    train_ds = SimpleMSCXRDataset(args, "train")
    val_ds = SimpleMSCXRDataset(args, "val")
    eval_ds = SimpleMSCXRDataset(args, "eval")
    train_loader = DataLoader(train_ds, batch_size=cli.batch_size, shuffle=True, num_workers=cli.num_workers, collate_fn=simple_collate)
    val_loader = DataLoader(val_ds, batch_size=cli.batch_size, shuffle=False, num_workers=cli.num_workers, collate_fn=simple_collate)
    eval_loader = DataLoader(eval_ds, batch_size=cli.batch_size, shuffle=False, num_workers=cli.num_workers, collate_fn=simple_collate)

    if cli.init_released_reference:
        raise ValueError("Released AGPT checkpoint is forbidden for fair full training.")
    model, criterion, _, _, weight_dict = build(args)
    model.to(device)
    criterion.to(device)
    tokenizer = model.transformer.tokenizer
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cli.lr, weight_decay=cli.weight_decay)

    run_name = cli.run_name or f"mdetr_token_box_s{cli.seed}"
    best_path = EXP_ROOT / "checkpoints" / f"{run_name}_best.pt"
    last_path = EXP_ROOT / "checkpoints" / f"{run_name}_last.pt"
    log_rows: list[dict[str, Any]] = []
    best_val = -1.0
    best_epoch = -1
    start = time.time()

    for epoch in range(cli.epochs):
        model.train()
        criterion.train()
        epoch_losses: list[float] = []
        epoch_ious: list[float] = []
        for batch_idx, batch in enumerate(train_loader):
            if cli.max_train_batches is not None and batch_idx >= cli.max_train_batches:
                break
            images = batch["images"].to(device)
            targets = build_mdetr_targets(batch, device, include_tokens_positive=cli.contrastive_align)
            positive_map = build_token_positive_map(batch["phrases"], tokenizer, args.num_classes, device)
            outputs = model(images, batch["phrases"])
            loss_dict = criterion(outputs, targets, positive_map)
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cli.clip_max_norm)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            with torch.no_grad():
                target_boxes = torch.stack([t["boxes"][0] for t in targets], dim=0)
                ious = pairwise_iou_diag(outputs["pred_boxes"].detach().squeeze(1), target_boxes)
            epoch_ious.extend([float(x) for x in ious.detach().cpu().tolist()])

        val_summary = evaluate(model, val_loader, device, max_batches=cli.max_val_batches)
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(epoch_losses)) if epoch_losses else math.nan,
            "train_mean_iou_batch": float(np.mean(epoch_ious)) if epoch_ious else math.nan,
            "val_mean_iou": val_summary.mean_iou,
            "val_median_iou": val_summary.median_iou,
            "val_hit_0_1": val_summary.hit_0_1,
            "val_hit_0_3": val_summary.hit_0_3,
            "val_hit_0_5": val_summary.hit_0_5,
            "val_n": val_summary.n,
            "init_note": "no_agpt_released_checkpoint_token_box_loss" + ("_with_contrastive_align" if cli.contrastive_align else ""),
        }
        log_rows.append(row)
        write_csv(EXP_ROOT / "metrics" / f"{run_name}_train_log.csv", log_rows)
        if val_summary.mean_iou == val_summary.mean_iou and val_summary.mean_iou > best_val:
            best_val = val_summary.mean_iou
            best_epoch = epoch + 1
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": best_epoch,
                    "best_val_mean_iou": best_val,
                    "args": vars(cli),
                    "weight_dict": {k: float(v) if isinstance(v, (int, float)) else v for k, v in weight_dict.items()},
                },
                best_path,
            )
        torch.save({"model": model.state_dict(), "epoch": epoch + 1, "args": vars(cli)}, last_path)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model"], strict=True)
    pred_path = EXP_ROOT / "predictions" / f"{run_name}_eval_predictions.csv"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    eval_summary = evaluate(model, eval_loader, device, prediction_path=pred_path)
    elapsed = time.time() - start
    summary = {
        "method": "mdetr_token_box_contrastive_fair_retrain" if cli.contrastive_align else "mdetr_token_box_fair_retrain",
        "run_name": run_name,
        "seed": cli.seed,
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "eval_rows": len(eval_ds),
        "epochs": cli.epochs,
        "batch_size": cli.batch_size,
        "lr": cli.lr,
        "image_size": cli.image_size,
        "best_epoch": best_epoch,
        "best_val_mean_iou": best_val,
        "eval_mean_iou": eval_summary.mean_iou,
        "eval_median_iou": eval_summary.median_iou,
        "eval_hit_0_1": eval_summary.hit_0_1,
        "eval_hit_0_3": eval_summary.hit_0_3,
        "eval_hit_0_5": eval_summary.hit_0_5,
        "eval_n": eval_summary.n,
        "trainable_params": trainable,
        "total_params": total,
        "elapsed_sec": elapsed,
        "released_agpt_weight_used": False,
        "checkpoint_path": str(best_path),
        "prediction_path": str(pred_path),
        "note": (
            "MDETR-style fair retrain with soft token classification, bbox/GIoU, and contrastive alignment losses."
            if cli.contrastive_align
            else "MDETR-style fair retrain with soft token classification plus bbox/GIoU losses; contrastive alignment not enabled."
        ),
    }
    write_csv(EXP_ROOT / "metrics" / f"{run_name}_eval_summary.csv", [summary])
    (EXP_ROOT / "metadata" / f"{run_name}_status.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("audit")
    smoke = sub.add_parser("smoke-train")
    smoke.add_argument("--epochs", type=int, default=1)
    smoke.add_argument("--batch-size", type=int, default=1)
    smoke.add_argument("--image-size", type=int, default=640)
    smoke.add_argument("--num-workers", type=int, default=0)
    smoke.add_argument("--device", default="cuda:0")
    smoke.add_argument("--seed", type=int, default=42)
    smoke.add_argument("--dataset-root", default="")
    smoke.add_argument("--output-root", default="")
    smoke.add_argument("--lr", type=float, default=1e-5)
    smoke.add_argument("--lr-backbone", type=float, default=0.0)
    smoke.add_argument("--text-encoder-lr", type=float, default=0.0)
    smoke.add_argument("--weight-decay", type=float, default=1e-4)
    smoke.add_argument("--clip-max-norm", type=float, default=0.1)
    smoke.add_argument("--max-train-batches", type=int, default=2)
    smoke.add_argument("--max-val-batches", type=int, default=2)
    smoke.add_argument(
        "--freeze-text-encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze RoBERTa text encoder for the first smoke run.",
    )
    smoke.add_argument(
        "--init-released-reference",
        action="store_true",
        help="Load AGPT MS-CXR-finetuned mdetr.pth. This is reference-only and not fair.",
    )
    smoke.add_argument(
        "--enable-train-aug",
        dest="disable_train_aug",
        action="store_false",
        default=True,
        help="Use AGPT train augmentation. Default is disabled because the local transform can crop before padding.",
    )
    full = sub.add_parser("train-box-only")
    full.add_argument("--epochs", type=int, default=20)
    full.add_argument("--batch-size", type=int, default=1)
    full.add_argument("--image-size", type=int, default=640)
    full.add_argument("--num-workers", type=int, default=0)
    full.add_argument("--device", default="cuda:0")
    full.add_argument("--seed", type=int, default=42)
    full.add_argument("--dataset-root", default="")
    full.add_argument("--output-root", default="")
    full.add_argument("--lr", type=float, default=1e-5)
    full.add_argument("--lr-backbone", type=float, default=0.0)
    full.add_argument("--text-encoder-lr", type=float, default=0.0)
    full.add_argument("--weight-decay", type=float, default=1e-4)
    full.add_argument("--clip-max-norm", type=float, default=0.1)
    full.add_argument("--max-train-batches", type=int, default=None)
    full.add_argument("--max-val-batches", type=int, default=None)
    full.add_argument("--run-name", default="")
    full.add_argument("--freeze-text-encoder", action=argparse.BooleanOptionalAction, default=True)
    full.add_argument("--init-released-reference", action="store_true")
    full.add_argument("--enable-train-aug", dest="disable_train_aug", action="store_false", default=True)
    token = sub.add_parser("train-token-box")
    token.add_argument("--epochs", type=int, default=20)
    token.add_argument("--batch-size", type=int, default=1)
    token.add_argument("--image-size", type=int, default=640)
    token.add_argument("--num-workers", type=int, default=0)
    token.add_argument("--device", default="cuda:0")
    token.add_argument("--seed", type=int, default=42)
    token.add_argument("--dataset-root", default="")
    token.add_argument("--output-root", default="")
    token.add_argument("--lr", type=float, default=1e-5)
    token.add_argument("--lr-backbone", type=float, default=0.0)
    token.add_argument("--text-encoder-lr", type=float, default=0.0)
    token.add_argument("--weight-decay", type=float, default=1e-4)
    token.add_argument("--clip-max-norm", type=float, default=0.1)
    token.add_argument("--max-train-batches", type=int, default=None)
    token.add_argument("--max-val-batches", type=int, default=None)
    token.add_argument("--run-name", default="")
    token.add_argument("--freeze-text-encoder", action=argparse.BooleanOptionalAction, default=True)
    token.add_argument("--init-released-reference", action="store_true")
    token.add_argument("--enable-train-aug", dest="disable_train_aug", action="store_false", default=True)
    token.add_argument("--contrastive-align", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_runtime_paths(args)
    if args.mode == "audit":
        run_audit()
    elif args.mode == "smoke-train":
        smoke_train(args)
    elif args.mode == "train-box-only":
        full_train_box_only(args)
    elif args.mode == "train-token-box":
        full_train_token_box(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()

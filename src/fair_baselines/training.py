from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from .metrics import iou_xyxy, summarize_records, tune_mask_decoder
from .models import GroundingDinoAdapter, LViTAdapter, RecLMISAdapter, SegmentationAdapter, build_adapter
from .mscxr_data import (
    DEFAULT_DATA_ROOT,
    GroundingSample,
    MaskGroundingDataset,
    audit_splits,
    collate_mask_batch,
    load_all_splits,
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def soft_dice_bce(probability: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    # These adapters expose post-sigmoid probabilities. BCE on probabilities is
    # deliberately kept in FP32 because PyTorch rejects it inside autocast.
    with torch.autocast(device_type=probability.device.type, enabled=False):
        probability = probability.float()
        finite = torch.isfinite(probability)
        if not bool(finite.all()):
            nonfinite_count = int((~finite).sum().item())
            raise RuntimeError(f"Non-finite segmentation probabilities: {nonfinite_count}")
        probability = probability.clamp(1e-5, 1.0 - 1e-5)
        target = target.float()
        bce = F.binary_cross_entropy(probability, target)
        dims = tuple(range(1, probability.ndim))
        intersection = (probability * target).sum(dims)
        denominator = probability.sum(dims) + target.sum(dims)
        dice = ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
        loss = bce + (1.0 - dice)
    return loss, {"bce": float(bce.detach()), "soft_dice": float(dice.detach())}


@torch.no_grad()
def evaluate_mask_loss(
    adapter: SegmentationAdapter,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
) -> dict[str, float]:
    adapter.eval()
    losses: list[float] = []
    mask_ious: list[float] = []
    for batch in loader:
        target = batch["masks"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            probability, auxiliary = adapter.forward_batch(batch, device)
            base_loss, _ = soft_dice_bce(probability, target)
            if isinstance(adapter, RecLMISAdapter):
                loss = adapter.loss_weight["loss_criterion"] * base_loss + adapter.auxiliary_loss(auxiliary)
            else:
                loss = base_loss
        losses.append(float(loss.detach()))
        pred = probability.float() >= 0.5
        truth = target >= 0.5
        intersection = torch.logical_and(pred, truth).flatten(1).sum(1).float()
        union = torch.logical_or(pred, truth).flatten(1).sum(1).float()
        mask_ious.extend(torch.where(union > 0, intersection / union, torch.zeros_like(union)).cpu().tolist())
    return {"loss": float(np.mean(losses)), "mask_iou": float(np.mean(mask_ious))}


def smoke_overfit_segmentation(
    adapter: SegmentationAdapter,
    samples: list[GroundingSample],
    device: torch.device,
    image_size: int,
    amp: bool,
    steps: int = 10,
) -> dict[str, Any]:
    subset = samples[: min(4, len(samples))]
    loader = DataLoader(
        MaskGroundingDataset(subset, image_size=image_size),
        batch_size=min(2, len(subset)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_mask_batch,
    )
    if isinstance(adapter, LViTAdapter):
        adapter.precompute_phrases([sample.phrase for sample in subset], device)
    batch = next(iter(loader))
    target = batch["masks"].to(device)
    optimizer = torch.optim.AdamW((p for p in adapter.parameters() if p.requires_grad), lr=3e-4, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    losses: list[float] = []
    adapter.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            probability, auxiliary = adapter.forward_batch(batch, device)
            base_loss, _ = soft_dice_bce(probability, target)
            if isinstance(adapter, RecLMISAdapter):
                loss = adapter.loss_weight["loss_criterion"] * base_loss + adapter.auxiliary_loss(auxiliary)
            else:
                loss = base_loss
        if not torch.isfinite(loss):
            return {"pass": False, "reason": "non-finite loss", "losses": losses}
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for p in adapter.parameters() if p.requires_grad], 1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))
    improved = min(losses[-3:]) < max(losses[:3]) if len(losses) >= 6 else losses[-1] < losses[0]
    return {
        "pass": bool(improved and all(math.isfinite(value) for value in losses)),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "best_loss": min(losses),
        "steps": steps,
        "losses": losses,
    }


@torch.no_grad()
def collect_mask_probabilities(
    adapter: SegmentationAdapter,
    samples: list[GroundingSample],
    device: torch.device,
    image_size: int,
    batch_size: int,
    amp: bool,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    loader = DataLoader(
        MaskGroundingDataset(samples, image_size=image_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        pin_memory=device.type == "cuda",
        collate_fn=collate_mask_batch,
    )
    adapter.eval()
    probabilities: list[np.ndarray] = []
    geometries: list[dict[str, Any]] = []
    for batch in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            output, _ = adapter.forward_batch(batch, device)
        probabilities.extend(output.float().cpu().numpy()[:, 0])
        geometries.extend(batch["geometries"])
    return probabilities, geometries


def records_from_masks(
    probabilities: list[np.ndarray],
    samples: list[GroundingSample],
    geometries: list[dict[str, Any]],
    decoder: dict[str, Any],
) -> list[dict[str, Any]]:
    from .metrics import decode_probability_mask

    records = []
    for probability, sample, geometry in zip(probabilities, samples, geometries):
        boxes, scores = decode_probability_mask(
            probability,
            geometry,
            float(decoder["threshold"]),
            float(decoder["min_area_ratio"]),
            int(decoder["max_components"]),
        )
        records.append({"sample": sample, "pred_boxes": boxes, "scores": scores})
    return records


def save_prediction_records(path: Path, records: list[dict[str, Any]]) -> None:
    rows = []
    for record in records:
        sample: GroundingSample = record["sample"]
        rows.append(
            {
                "group_id": sample.group_id,
                "dicom_id": sample.dicom_id,
                "subject_id": sample.subject_id,
                "study_id": sample.study_id,
                "finding": sample.finding,
                "phrase": sample.phrase,
                "gold_count": sample.gold_count,
                "pred_count": len(record["pred_boxes"]),
                "pred_boxes_xyxy": json.dumps(record["pred_boxes"]),
                "pred_scores": json.dumps(record.get("scores", [])),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def train_segmentation_baseline(
    adapter: SegmentationAdapter,
    splits: dict[str, list[GroundingSample]],
    output_dir: Path,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    image_size: int,
    learning_rate: float,
    protocol: str,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # RecLMIS' official optimizer path is FP32. Its reconstruction/contrastive
    # branches become non-finite after a few epochs under FP16 autocast.
    amp = device.type == "cuda" and not isinstance(adapter, RecLMISAdapter)
    adapter.to(device)
    train_val_phrases = [
        sample.phrase for split in ("train", "val") for sample in splits[split]
    ]
    if isinstance(adapter, LViTAdapter):
        adapter.precompute_phrases(train_val_phrases, device)

    smoke = smoke_overfit_segmentation(adapter, splits["train"], device, image_size, amp)
    write_json(output_dir / "smoke_overfit.json", smoke)
    if not smoke["pass"]:
        raise RuntimeError(f"{adapter.model_name} smoke overfit did not improve")

    # Rebuild after smoke so the full run starts from the declared public initialization.
    del adapter
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model_key = output_dir.parent.name
    adapter = build_adapter(model_key)
    if not isinstance(adapter, SegmentationAdapter):
        raise TypeError(model_key)
    adapter.to(device)
    if isinstance(adapter, LViTAdapter):
        adapter.precompute_phrases(train_val_phrases, device)

    train_loader = DataLoader(
        MaskGroundingDataset(splits["train"], image_size=image_size),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        pin_memory=amp,
        collate_fn=collate_mask_batch,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        MaskGroundingDataset(splits["val"], image_size=image_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        pin_memory=amp,
        collate_fn=collate_mask_batch,
    )
    trainable = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=learning_rate * 0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best_metric = -1.0
    best_epoch = 0
    stale = 0
    log_rows: list[dict[str, Any]] = []
    checkpoint = output_dir / "checkpoint" / "best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for epoch in range(1, epochs + 1):
        adapter.train()
        epoch_losses: list[float] = []
        for batch in train_loader:
            target = batch["masks"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                probability, auxiliary = adapter.forward_batch(batch, device)
                base_loss, _ = soft_dice_bce(probability, target)
                if isinstance(adapter, RecLMISAdapter):
                    loss = adapter.loss_weight["loss_criterion"] * base_loss + adapter.auxiliary_loss(auxiliary)
                else:
                    loss = base_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(float(loss.detach()))
        val = evaluate_mask_loss(adapter, val_loader, device, amp)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "val_loss": val["loss"],
            "val_mask_iou": val["mask_iou"],
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_sec": time.time() - started,
        }
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(output_dir / "train_log.csv", index=False)
        if val["mask_iou"] > best_metric + 1e-6:
            best_metric = val["mask_iou"]
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model": adapter.checkpoint_state(),
                    "model_name": adapter.model_name,
                    "seed": seed,
                    "epoch": epoch,
                    "val_mask_iou": best_metric,
                },
                checkpoint,
            )
        else:
            stale += 1
        scheduler.step()
        if epoch >= 5 and stale >= patience:
            break

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    adapter.core.load_state_dict(state)
    val_prob, val_geometry = collect_mask_probabilities(
        adapter, splits["val"], device, image_size, batch_size, amp
    )
    decoder, grid = tune_mask_decoder(
        val_prob,
        splits["val"],
        val_geometry,
        singlebox_only=protocol == "singlebox_888",
    )
    pd.DataFrame(grid).to_csv(output_dir / "val_decoder_grid.csv", index=False)
    write_json(output_dir / "selected_decoder.json", decoder)

    if isinstance(adapter, LViTAdapter):
        # Eval text is encoded only after checkpoint and decoder selection are fixed.
        adapter.precompute_phrases([sample.phrase for sample in splits["eval"]], device)
    eval_prob, eval_geometry = collect_mask_probabilities(
        adapter, splits["eval"], device, image_size, batch_size, amp
    )
    eval_records = records_from_masks(eval_prob, splits["eval"], eval_geometry, decoder)
    save_prediction_records(output_dir / "predictions" / "eval_predictions.csv", eval_records)
    single, multi = summarize_records(eval_records)
    expected_groups = 163 if protocol == "singlebox_888" else 220
    expected_boxes = 163 if protocol == "singlebox_888" else 280
    if int(single["n"]) != 163 or int(multi["n_groups"]) != expected_groups or int(multi["n_gt_boxes"]) != expected_boxes:
        raise RuntimeError(f"Protocol count mismatch: single={single}, multi={multi}")
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([single]).to_csv(output_dir / "metrics" / "singlebox_888.csv", index=False)
    if protocol == "multibox_1444":
        pd.DataFrame([multi]).to_csv(output_dir / "metrics" / "multibox_1444.csv", index=False)
    return {
        "model_name": adapter.model_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_mask_iou": best_metric,
        "decoder": decoder,
        "singlebox_888": single,
        "multibox_1444": multi if protocol == "multibox_1444" else None,
        "protocol": protocol,
        "elapsed_sec": time.time() - started,
        "checkpoint": str(checkpoint),
    }


def _normalized_cxcywh(sample: GroundingSample) -> torch.Tensor:
    rows = []
    for x1, y1, x2, y2 in sample.gt_boxes:
        rows.append(
            [
                ((x1 + x2) / 2.0) / sample.image_width,
                ((y1 + y2) / 2.0) / sample.image_height,
                (x2 - x1) / sample.image_width,
                (y2 - y1) / sample.image_height,
            ]
        )
    return torch.tensor(rows, dtype=torch.float32)


def _dino_prompt(phrase: str) -> str:
    return " ".join(phrase.replace(".", " ").split()).strip() + "."


def _dino_training_input(
    adapter: GroundingDinoAdapter, sample: GroundingSample, device: torch.device
) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
    with Image.open(sample.image_path) as source:
        image = source.convert("RGB")
    # Grounding DINO treats periods as class delimiters. Each medical phrase is
    # one target class, even when the source sentence contains punctuation.
    inputs = adapter.processor(images=[image], text=[_dino_prompt(sample.phrase)], return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    labels = [
        {
            "class_labels": torch.zeros(sample.gold_count, dtype=torch.long, device=device),
            "boxes": _normalized_cxcywh(sample).to(device),
        }
    ]
    return inputs, labels


def _dino_compatibility_loss(adapter: GroundingDinoAdapter, output: Any) -> torch.Tensor:
    if adapter.native_loss:
        if output.loss is None:
            raise RuntimeError("Grounding DINO did not return its model-native loss")
        return output.loss
    losses = output.loss_dict
    config = adapter.core.config
    total = 2.0 * losses["loss_ce"]
    total = total + float(config.bbox_loss_coefficient) * losses["loss_bbox"]
    total = total + float(config.giou_loss_coefficient) * losses["loss_giou"]
    if "loss_bbox_enc" in losses:
        total = total + float(config.bbox_loss_coefficient) * losses["loss_bbox_enc"]
    if "loss_giou_enc" in losses:
        total = total + float(config.giou_loss_coefficient) * losses["loss_giou_enc"]
    return total


@torch.no_grad()
def _dino_val_loss(
    adapter: GroundingDinoAdapter,
    samples: list[GroundingSample],
    device: torch.device,
    amp: bool,
    amp_dtype: torch.dtype = torch.float16,
) -> float:
    adapter.eval()
    losses = []
    for sample in samples:
        inputs, labels = _dino_training_input(adapter, sample, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
            output = adapter.core(**inputs, labels=labels)
            loss = _dino_compatibility_loss(adapter, output)
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


@torch.no_grad()
def collect_dino_candidates(
    adapter: GroundingDinoAdapter, samples: list[GroundingSample], device: torch.device, amp: bool
) -> list[list[dict[str, Any]]]:
    adapter.eval()
    all_candidates: list[list[dict[str, Any]]] = []
    for sample in samples:
        with Image.open(sample.image_path) as source:
            image = source.convert("RGB")
        inputs = adapter.processor(images=[image], text=[_dino_prompt(sample.phrase)], return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16 if adapter.native_loss else torch.float16,
            enabled=amp,
        ):
            outputs = adapter.core(**inputs)
        result = adapter.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=0.03,
            text_threshold=0.03,
            target_sizes=[(sample.image_height, sample.image_width)],
        )[0]
        candidates = [
            {"box": [float(value) for value in box], "score": float(score)}
            for box, score in zip(result["boxes"].detach().cpu().tolist(), result["scores"].detach().cpu().tolist())
        ]
        candidates.sort(key=lambda row: row["score"], reverse=True)
        all_candidates.append(candidates)
    return all_candidates


def _nms_candidates(candidates: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda row: row["score"], reverse=True):
        if all(iou_xyxy(candidate["box"], old["box"]) < threshold for old in kept):
            kept.append(candidate)
    return kept


def tune_dino_decoder(
    candidates: list[list[dict[str, Any]]],
    samples: list[GroundingSample],
    singlebox_only: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grid = []
    max_box_values = (1,) if singlebox_only else (1, 2, 3)
    for score_threshold in (0.05, 0.1, 0.2, 0.3, 0.4):
        for nms_threshold in (0.3, 0.5, 0.7):
            for max_boxes in max_box_values:
                records = []
                for rows, sample in zip(candidates, samples):
                    selected = [row for row in rows if row["score"] >= score_threshold]
                    selected = _nms_candidates(selected, nms_threshold)[:max_boxes]
                    if not selected and rows:
                        selected = rows[:1]
                    records.append(
                        {
                            "sample": sample,
                            "pred_boxes": [row["box"] for row in selected],
                            "scores": [row["score"] for row in selected],
                        }
                    )
                single, metrics = summarize_records(records)
                selection_score = (
                    float(single["mean_iou"])
                    if singlebox_only
                    else float(
                        0.25 * metrics["coverage_mean_iou"]
                        + 0.25 * metrics["exact_union_iou"]
                        + 0.25 * metrics["set_f1_0_3"]
                        + 0.25 * metrics["set_f1_0_5"]
                    )
                )
                grid.append(
                    {
                        "score_threshold": score_threshold,
                        "nms_threshold": nms_threshold,
                        "max_boxes": max_boxes,
                        "selection_score": selection_score,
                        **metrics,
                    }
                )
    best = max(grid, key=lambda row: (row["selection_score"], -row["max_boxes"]))
    return {
        "score_threshold": float(best["score_threshold"]),
        "nms_threshold": float(best["nms_threshold"]),
        "max_boxes": int(best["max_boxes"]),
        "selection_score": float(best["selection_score"]),
    }, grid


def dino_records(
    candidates: list[list[dict[str, Any]]], samples: list[GroundingSample], decoder: dict[str, Any]
) -> list[dict[str, Any]]:
    records = []
    for rows, sample in zip(candidates, samples):
        selected = [row for row in rows if row["score"] >= decoder["score_threshold"]]
        selected = _nms_candidates(selected, decoder["nms_threshold"])[: decoder["max_boxes"]]
        if not selected and rows:
            selected = rows[:1]
        records.append(
            {
                "sample": sample,
                "pred_boxes": [row["box"] for row in selected],
                "scores": [row["score"] for row in selected],
            }
        )
    return records


def train_grounding_dino(
    adapter: GroundingDinoAdapter,
    splits: dict[str, list[GroundingSample]],
    output_dir: Path,
    seed: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    protocol: str,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Grounding DINO fair retrain requires CUDA")
    # BF16 keeps the model-native encoder focal loss finite on RTX 4090 while
    # preserving most of the memory benefit of mixed precision.
    amp = True
    amp_dtype = torch.bfloat16 if adapter.native_loss else torch.float16
    adapter.to(device)
    trainable = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    checkpoint = output_dir / "checkpoint" / "best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    logs = []
    started = time.time()
    for epoch in range(1, epochs + 1):
        adapter.train()
        order = list(range(len(splits["train"])))
        random.Random(seed + epoch).shuffle(order)
        train_losses = []
        optimizer.zero_grad(set_to_none=True)
        accumulation = 4
        for step, index in enumerate(order, start=1):
            inputs, labels = _dino_training_input(adapter, splits["train"][index], device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                output = adapter.core(**inputs, labels=labels)
                raw_loss = _dino_compatibility_loss(adapter, output)
                window_start = ((step - 1) // accumulation) * accumulation
                window_size = min(accumulation, len(order) - window_start)
                loss = raw_loss / window_size
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite Grounding DINO loss at epoch {epoch}")
            scaler.scale(loss).backward()
            train_losses.append(float(raw_loss.detach()))
            if step % accumulation == 0 or step == len(order):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        val_loss = _dino_val_loss(adapter, splits["val"], device, amp, amp_dtype)
        logs.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "val_loss": val_loss,
                "elapsed_sec": time.time() - started,
            }
        )
        pd.DataFrame(logs).to_csv(output_dir / "train_log.csv", index=False)
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model": adapter.core.state_dict(),
                    "model_name": adapter.model_name,
                    "seed": seed,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                checkpoint,
            )
        else:
            stale += 1
        if epoch >= 3 and stale >= patience:
            break
    adapter.core.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    val_candidates = collect_dino_candidates(adapter, splits["val"], device, amp)
    decoder, grid = tune_dino_decoder(
        val_candidates,
        splits["val"],
        singlebox_only=protocol == "singlebox_888",
    )
    pd.DataFrame(grid).to_csv(output_dir / "val_decoder_grid.csv", index=False)
    write_json(output_dir / "selected_decoder.json", decoder)
    eval_candidates = collect_dino_candidates(adapter, splits["eval"], device, amp)
    records = dino_records(eval_candidates, splits["eval"], decoder)
    save_prediction_records(output_dir / "predictions" / "eval_predictions.csv", records)
    single, multi = summarize_records(records)
    expected_groups = 163 if protocol == "singlebox_888" else 220
    expected_boxes = 163 if protocol == "singlebox_888" else 280
    if int(single["n"]) != 163 or int(multi["n_groups"]) != expected_groups or int(multi["n_gt_boxes"]) != expected_boxes:
        raise RuntimeError(f"Protocol count mismatch: single={single}, multi={multi}")
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([single]).to_csv(output_dir / "metrics" / "singlebox_888.csv", index=False)
    if protocol == "multibox_1444":
        pd.DataFrame([multi]).to_csv(output_dir / "metrics" / "multibox_1444.csv", index=False)
    return {
        "model_name": adapter.model_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "decoder": decoder,
        "singlebox_888": single,
        "multibox_1444": multi if protocol == "multibox_1444" else None,
        "protocol": protocol,
        "elapsed_sec": time.time() - started,
        "checkpoint": str(checkpoint),
    }


def run_baseline(
    model_key: str,
    output_dir: Path,
    seed: int = 42,
    epochs: int = 20,
    patience: int = 5,
    batch_size: int = 2,
    image_size: int = 224,
    learning_rate: float = 3e-4,
    data_root: Path = DEFAULT_DATA_ROOT,
    protocol: str = "multibox_1444",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "FINAL_STATUS.json"
    write_json(status_path, {"state": "running", "model": model_key, "seed": seed, "started_at": time.time()})
    set_seed(seed)
    try:
        splits = load_all_splits(data_root)
        split_audit = audit_splits(splits, data_root, protocol=protocol)
        write_json(output_dir / "split_audit.json", split_audit)
        if not split_audit["pass"]:
            raise RuntimeError(f"Split audit failed: {split_audit}")
        adapter = build_adapter(model_key)
        trainable = sum(parameter.numel() for parameter in adapter.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in adapter.parameters())
        config = {
            "model_key": model_key,
            "model_name": adapter.model_name,
            "seed": seed,
            "requested_epochs": epochs,
            "actual_max_epochs": (
                epochs
                if isinstance(adapter, GroundingDinoAdapter) and adapter.native_loss
                else min(epochs, 10)
                if isinstance(adapter, GroundingDinoAdapter)
                else epochs
            ),
            "patience": patience,
            "batch_size": 1 if isinstance(adapter, GroundingDinoAdapter) else batch_size,
            "image_size": image_size,
            "learning_rate": learning_rate,
            "mixed_precision": not isinstance(adapter, RecLMISAdapter),
            "mixed_precision_dtype": (
                "bfloat16"
                if isinstance(adapter, GroundingDinoAdapter) and adapter.native_loss
                else "float16"
                if not isinstance(adapter, RecLMISAdapter)
                else "float32"
            ),
            "precision_note": (
                "FP32, matching official RecLMIS training; AMP is numerically unstable"
                if isinstance(adapter, RecLMISAdapter)
                else "CUDA AMP when available"
            ),
            "trainable_parameters": trainable,
            "total_parameters": total,
            "provenance": adapter.provenance,
            "protocol": protocol,
            "protocol_note": (
                "trained on singleton-only 638/87 and evaluated once on singleton eval163; exactly one box"
                if protocol == "singlebox_888"
                else "one model trained on 1444 train phrase groups and decoded identically for 888/1444"
            ),
            "eval_selection_forbidden": True,
        }
        write_json(output_dir / "config.json", config)
        if isinstance(adapter, GroundingDinoAdapter):
            result = train_grounding_dino(
                adapter,
                splits,
                output_dir,
                seed=seed,
                epochs=epochs if adapter.native_loss else min(epochs, 10),
                patience=patience if adapter.native_loss else min(patience, 3),
                learning_rate=min(learning_rate, 2e-5),
                protocol=protocol,
            )
        else:
            result = train_segmentation_baseline(
                adapter,
                splits,
                output_dir,
                seed=seed,
                epochs=epochs,
                patience=patience,
                batch_size=batch_size,
                image_size=image_size,
                learning_rate=learning_rate,
                protocol=protocol,
            )
        final = {"state": "completed", "completed_at": time.time(), **result}
        write_json(status_path, final)
        return final
    except BaseException as exc:
        write_json(
            status_path,
            {
                "state": "failed",
                "failed_at": time.time(),
                "model": model_key,
                "seed": seed,
                "error": repr(exc),
            },
        )
        raise

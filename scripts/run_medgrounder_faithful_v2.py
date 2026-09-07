#!/usr/bin/env python
"""Faithful MedGrounder fine-tuning with the paper's available recipe."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from transformers import get_linear_schedule_with_warmup


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_medgrounder_fairness_audit_v1 as medgrounder_utils  # noqa: E402
from src.baseline_repro.hf_pins import enable_offline_hf, pin_record  # noqa: E402
from src.baseline_repro.letterbox import (  # noqa: E402
    canonical_xyxy_to_letterboxed_normalized_xyxy,
)
from src.baseline_repro.protocols import sha256_file, write_json  # noqa: E402


MEDGROUNDER_ROOT = PROJECT_ROOT / "third_party" / "MedGrounder"
DEFAULT_INIT_CHECKPOINT = (
    MEDGROUNDER_ROOT / "model_weight" / "medgrounder_pretrain_imagenome.pth"
)
FORBIDDEN_MS_CXR_CHECKPOINTS = (
    MEDGROUNDER_ROOT / "model_weight" / "medgrounder_finetune_ms.pth",
    MEDGROUNDER_ROOT / "model_weight" / "medgrounder_finetune_mspc.pth",
)
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "baseline_faithful_reproduction" / "20260712_v2"
OFFICIAL_PROTOCOL = DEFAULT_OUTPUT / "protocols" / "official_mscxr"
LOCAL_CONFIG = (
    PROJECT_ROOT
    / "experiments"
    / "medgrounder_gmpg_baseline_v1"
    / "configs"
    / "conf_local_mscxr_p10p19.yaml"
)
MSCXR_ALLOWED_CATEGORIES = {
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Lung Opacity",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
    "No Finding",
}


def git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        choices=["official", "local_singlebox", "local_multibox"],
        default="official",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--effective-batch-size", type=int, default=32)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protocol-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--limit-train-samples", type=int, default=0)
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_INIT_CHECKPOINT)
    parser.add_argument("--dev-only", action="store_true")
    parser.add_argument("--dev-annotation-file", type=Path)
    parser.add_argument(
        "--selection-metric",
        choices=["faithful_strict", "protocol_primary"],
        default="faithful_strict",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_config(args: argparse.Namespace):
    os.chdir(MEDGROUNDER_ROOT)
    base = OmegaConf.load(MEDGROUNDER_ROOT / "conf" / "base.yaml")
    datasets = OmegaConf.load(MEDGROUNDER_ROOT / "conf" / "datasets.yaml")
    if args.protocol == "official":
        run = OmegaConf.create(
            {
                "datasets": ["gmpg_mscxr"],
                "gmpg_mscxr_args": {
                    "mscxr_annotation_file": str(
                        args.protocol_root
                        / "protocols"
                        / "official_mscxr"
                        / "medgrounder"
                        / "official_mscxr.csv"
                    ),
                    "mscxr_image_root": str(MEDGROUNDER_ROOT.parent / "MedRPG" / "ln_data" / "MS_CXR"),
                },
            }
        )
    elif args.protocol == "local_singlebox":
        run = OmegaConf.create(
            {
                "datasets": ["gmpg_mscxr"],
                "gmpg_mscxr_args": {
                    "mscxr_annotation_file": str(
                        args.protocol_root
                        / "protocols"
                        / "local_singlebox_888"
                        / "medgrounder_local_singlebox.csv"
                    ),
                    # The exported CSV stores absolute paths.
                    "mscxr_image_root": str(PROJECT_ROOT),
                },
            }
        )
    else:
        run = OmegaConf.create(
            {
                "datasets": ["gmpg_mscxr"],
                "gmpg_mscxr_args": {
                    "mscxr_annotation_file": str(
                        args.protocol_root
                        / "protocols"
                        / "local_multibox_1444"
                        / "medgrounder_local_multibox.csv"
                    ),
                    # The canonical export stores absolute paths.
                    "mscxr_image_root": str(PROJECT_ROOT),
                },
            }
        )
    if args.dev_only:
        if args.dev_annotation_file is None:
            raise ValueError("--dev-only requires --dev-annotation-file")
        run.gmpg_mscxr_args.mscxr_annotation_file = str(args.dev_annotation_file)
    cfg = OmegaConf.merge(datasets, base, run)
    cfg.datasets = ["gmpg_mscxr"]
    cfg.batch_size = args.batch_size
    cfg.num_epochs = args.epochs
    cfg.num_workers = 0
    cfg.grounding_threshold = 0.8
    cfg.post_processing = True
    cfg.test_post_processing = True
    cfg.test_post_processing_params.run_wbf = True
    cfg.test_post_processing_params.run_nms = False
    cfg.test_post_processing_params.wbf_iou_threshold = 0.1
    cfg.test_post_processing_params.skip_box_thr = 0.0
    cfg.optimiser_kwargs.lr = 1e-5
    cfg.optimiser_kwargs.weight_decay = 1e-4
    cfg.lr_backbone = 1e-5
    cfg.text_encoder_lr = 5e-5
    cfg.schedule = "linear_with_warmup"
    cfg.fraction_warmup_steps = 0.01
    cfg.clip_max_norm = 0.1
    cfg.limit_train_samples = args.limit_train_samples or None
    return cfg


def parameter_groups(model: torch.nn.Module, cfg: Any) -> tuple[list[dict[str, Any]], dict[str, int]]:
    groups: dict[str, list[torch.nn.Parameter]] = {"main": [], "backbone": [], "text": []}
    counts = {"main": 0, "backbone": 0, "text": 0}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "text_encoder" in name or "tokenizer" in name:
            key = "text"
        elif "backbone" in name:
            key = "backbone"
        else:
            key = "main"
        groups[key].append(parameter)
        counts[key] += parameter.numel()
    payload = [
        {"params": groups["main"], "lr": float(cfg.optimiser_kwargs.lr), "name": "main"},
        {"params": groups["backbone"], "lr": float(cfg.lr_backbone), "name": "backbone"},
        {"params": groups["text"], "lr": float(cfg.text_encoder_lr), "name": "text"},
    ]
    return [group for group in payload if group["params"]], counts


def atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp_{os.getpid()}{path.suffix}")
    torch.save(payload, temporary, _use_new_zipfile_serialization=False)
    torch.load(temporary, map_location="cpu", weights_only=False)
    os.replace(temporary, path)


def audit_dataset_cardinality(
    cfg: Any,
    train_loader: Any,
    val_loader: Any,
    test_loader: Any,
    output: Path,
) -> dict[str, Any]:
    annotation = Path(cfg.gmpg_mscxr_args.mscxr_annotation_file)
    frame = pd.read_csv(annotation)
    required_columns = {
        "split",
        "label_text",
        "path",
        "category_name",
        "x",
        "y",
        "w",
        "h",
        "image_width",
        "image_height",
    }
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        payload = {
            "status": "fail",
            "annotation_file": str(annotation),
            "missing_required_columns": missing_columns,
        }
        write_json(output, payload)
        raise RuntimeError(f"MedGrounder annotation columns missing: {missing_columns}")

    image_root = str(cfg.gmpg_mscxr_args.mscxr_image_root)
    labels = frame["label_text"].astype("string")
    paths = frame["path"].astype("string")
    geometry_key = ["split", "label_text", "path", "x", "y", "w", "h"]
    invalid_geometry = (
        (frame["x"] < 0)
        | (frame["y"] < 0)
        | (frame["w"] <= 0)
        | (frame["h"] <= 0)
        | (frame["x"] + frame["w"] > frame["image_width"] + 1e-5)
        | (frame["y"] + frame["h"] > frame["image_height"] + 1e-5)
    )
    unique_paths = frame["path"].dropna().astype(str).drop_duplicates()
    missing_images = [
        path
        for path in unique_paths
        if not Path(os.path.join(image_root, path)).exists()
    ]
    group_key = ["split", "label_text", "path"]
    all_group_sizes = frame.groupby(group_key, dropna=False).size()
    mixed_category_groups = int(
        (
            frame.groupby(group_key, dropna=False)["category_name"].nunique(dropna=False)
            > 1
        ).sum()
    )
    validation = {
        "null_label_rows": int(frame["label_text"].isna().sum()),
        "blank_label_rows": int(labels.str.strip().eq("").fillna(False).sum()),
        "null_path_rows": int(frame["path"].isna().sum()),
        "blank_path_rows": int(paths.str.strip().eq("").fillna(False).sum()),
        "unexpected_splits": sorted(set(frame["split"].dropna().astype(str)) - {"train", "val", "test"}),
        "unknown_categories": sorted(set(frame["category_name"].dropna().astype(str)) - MSCXR_ALLOWED_CATEGORIES),
        "invalid_geometry_rows": int(invalid_geometry.sum()),
        "duplicate_geometry_rows": int(frame.duplicated(geometry_key).sum()),
        "mixed_category_phrase_groups": mixed_category_groups,
        "max_boxes_per_phrase_group": int(all_group_sizes.max()) if len(all_group_sizes) else 0,
        "groups_exceeding_num_queries": int((all_group_sizes > int(cfg.num_queries)).sum()),
        "missing_image_count": int(len(missing_images)),
        "missing_image_examples": missing_images[:10],
    }
    violations = {
        key: value
        for key, value in validation.items()
        if (
            (isinstance(value, int) and value != 0 and key != "max_boxes_per_phrase_group")
            or (isinstance(value, list) and bool(value))
        )
    }
    expected: dict[str, dict[str, int]] = {}
    actual = {
        "train": len(train_loader.dataset),
        "val": len(val_loader.dataset),
        "test": len(test_loader.dataset),
    }
    for split in ("train", "val", "test"):
        part = frame[frame["split"].eq(split)]
        group_sizes = part.groupby(["label_text", "path"], dropna=False).size()
        expected[split] = {
            "candidate_rows": int(len(part)),
            "phrase_groups": int(len(group_sizes)),
            "multibox_phrase_groups": int((group_sizes > 1).sum()),
            "boxes": int(group_sizes.sum()),
        }
    expected_dataset_lengths = {
        split: expected[split]["phrase_groups"] for split in ("train", "val", "test")
    }
    if cfg.limit_train_samples is not None:
        expected_dataset_lengths["train"] = min(
            expected_dataset_lengths["train"], int(cfg.limit_train_samples)
        )
    mismatches = {
        split: {"expected": expected_dataset_lengths[split], "actual": int(actual[split])}
        for split in actual
        if expected_dataset_lengths[split] != int(actual[split])
    }
    payload = {
        "status": "pass" if not mismatches and not violations else "fail",
        "annotation_file": str(annotation),
        "annotation_sha256": sha256_file(annotation),
        "expected": expected,
        "expected_dataset_lengths": expected_dataset_lengths,
        "actual_dataset_lengths": actual,
        "mismatches": mismatches,
        "validation": validation,
        "violations": violations,
    }
    write_json(output, payload)
    if mismatches or violations:
        raise RuntimeError(
            f"MedGrounder dataset audit failed: mismatches={mismatches}, violations={violations}"
        )
    return payload


def audit_prediction_gt_contract(
    annotation_file: Path, prediction_file: Path, output: Path, model_size: int = 640
) -> dict[str, Any]:
    source = pd.read_csv(annotation_file)
    source = source[source["split"].eq("test")].dropna(subset=["label_text", "path"]).copy()
    source["label_text"] = source["label_text"].astype(str).str.strip()
    source = source[source["label_text"].ne("")].reset_index(drop=True)
    grouped = source.groupby(["label_text", "path"]).indices
    phrase_keys = list(grouped.keys())
    predictions = pd.read_csv(prediction_file)
    observed_ids = predictions["group_id"].astype(int).tolist()
    expected_ids = list(range(len(phrase_keys)))
    mismatches = []
    for row in predictions.itertuples(index=False):
        group_id = int(row.group_id)
        if group_id < 0 or group_id >= len(phrase_keys):
            mismatches.append({"group_id": group_id, "reason": "out_of_range"})
            continue
        phrase, rel_path = phrase_keys[group_id]
        rows = source.iloc[grouped[(phrase, rel_path)]]
        expected_xyxy = np.asarray(
            [
                canonical_xyxy_to_letterboxed_normalized_xyxy(
                    [raw.x, raw.y, raw.x + raw.w, raw.y + raw.h],
                    raw.image_width,
                    raw.image_height,
                    model_size,
                )
                for raw in rows.itertuples(index=False)
            ],
            dtype=np.float64,
        )
        expected_cxcywh = np.stack(
            [
                (expected_xyxy[:, 0] + expected_xyxy[:, 2]) / 2,
                (expected_xyxy[:, 1] + expected_xyxy[:, 3]) / 2,
                expected_xyxy[:, 2] - expected_xyxy[:, 0],
                expected_xyxy[:, 3] - expected_xyxy[:, 1],
            ],
            axis=1,
        )
        observed = np.asarray(json.loads(row.gt_boxes_cxcywh), dtype=np.float64).reshape(-1, 4)
        expected_sorted = expected_cxcywh[np.lexsort(expected_cxcywh.T[::-1])]
        observed_sorted = observed[np.lexsort(observed.T[::-1])] if len(observed) else observed
        reasons = []
        if str(row.phrase).strip() != phrase:
            reasons.append("phrase")
        if not str(row.img_path).replace("\\", "/").endswith(str(rel_path).replace("\\", "/")):
            reasons.append("path")
        if expected_sorted.shape != observed_sorted.shape or not np.allclose(
            expected_sorted, observed_sorted, atol=1e-5, rtol=0.0
        ):
            reasons.append("gt_boxes")
        if reasons:
            mismatches.append({"group_id": group_id, "reason": "|".join(reasons)})
    payload = {
        "status": (
            "PASS"
            if observed_ids == expected_ids and not mismatches
            else "FAIL"
        ),
        "annotation_file": str(annotation_file),
        "prediction_file": str(prediction_file),
        "n_expected_groups": len(expected_ids),
        "n_prediction_rows": len(predictions),
        "duplicate_prediction_ids": len(observed_ids) - len(set(observed_ids)),
        "ordered_ids_match": observed_ids == expected_ids,
        "mismatches": mismatches[:100],
    }
    write_json(output, payload)
    if payload["status"] != "PASS":
        raise RuntimeError(f"MedGrounder prediction/GT contract failed: {payload}")
    return payload


def train(args: argparse.Namespace) -> dict[str, Any]:
    root = args.output_root.resolve()
    args.output_root = root
    args.protocol_root = args.protocol_root.resolve()
    args.init_checkpoint = args.init_checkpoint.resolve()
    if args.dev_annotation_file is not None:
        args.dev_annotation_file = args.dev_annotation_file.resolve()
    run_dir = root / "runs" / "medgrounder" / args.protocol / f"seed_{args.seed}"
    status_path = run_dir / "run_status.json"
    if status_path.exists() and not args.force:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "complete":
            return status | {"status": "skipped_existing"}
    for directory in ("checkpoint", "logs", "predictions", "metrics"):
        (run_dir / directory).mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    enable_offline_hf()
    text_pin = pin_record(
        "thomas-sounack/BioClinical-ModernBERT-base", require_main_ref=True
    )
    write_json(run_dir / "hf_text_encoder_pin.json", text_pin)
    cfg = build_config(args)
    if str(cfg.text_encoder_type) != text_pin["repo_id"]:
        raise RuntimeError(
            f"Unexpected MedGrounder text encoder: {cfg.text_encoder_type} != {text_pin['repo_id']}"
        )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    accumulation_steps = max(1, math.ceil(args.effective_batch_size / args.batch_size))

    medgrounder_utils.add_medgrounder_to_path()
    from dataloaders.dataset_builder import get_dataloaders

    train_loader, val_loader, test_loaders = get_dataloaders(cfg)
    test_loader = test_loaders["gmpg_mscxr"]
    cardinality_audit = audit_dataset_cardinality(
        cfg,
        train_loader,
        val_loader,
        test_loader,
        run_dir / "dataset_cardinality_audit.json",
    )
    init_checkpoint = args.init_checkpoint
    forbidden_checkpoints = tuple(path.resolve() for path in FORBIDDEN_MS_CXR_CHECKPOINTS)
    if not init_checkpoint.is_file():
        raise FileNotFoundError(f"Initialization checkpoint not found: {init_checkpoint}")
    if init_checkpoint in forbidden_checkpoints:
        raise RuntimeError(
            "MS-CXR or MS-CXR+PadChest fine-tuned checkpoints are forbidden for local evaluation: "
            f"{init_checkpoint}"
        )
    model, criterion, postprocessor = medgrounder_utils.build_model_and_load(cfg, init_checkpoint, device)
    param_groups, parameter_counts = parameter_groups(model, cfg)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=float(cfg.optimiser_kwargs.weight_decay))
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    total_steps = max(1, optimizer_steps_per_epoch * args.epochs)
    warmup_steps = max(1, int(total_steps * float(cfg.fraction_warmup_steps)))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    config_payload = {
        "method": "MedGrounder",
        "protocol": args.protocol,
        "seed": args.seed,
        "hf_text_encoder_pin": text_pin,
        "text_encoder_pin": text_pin,
        "epochs": args.epochs,
        "physical_batch_size": args.batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "effective_batch_size": args.batch_size * accumulation_steps,
        "gradient_accumulation_semantics": (
            "dynamic final-window averaging; approximates paper batch 32 because "
            "set losses are normalized inside each physical microbatch"
        ),
        "amp": use_amp,
        "amp_dtype": "bfloat16" if use_amp else "float32",
        "optimizer": "AdamW",
        "learning_rates": {"main": 1e-5, "backbone": 1e-5, "text": 5e-5},
        "parameter_counts": parameter_counts,
        "weight_decay": 1e-4,
        "scheduler": "linear_with_warmup",
        "warmup_steps": warmup_steps,
        "total_optimizer_steps": total_steps,
        "clip_max_norm": 0.1,
        "grounding_threshold": 0.8,
        "postprocessing": "WBF iou=0.1, skip_box_thr=0.0",
        "checkpoint_selection": args.selection_metric,
        "initialization": str(init_checkpoint),
        "initialization_sha256": sha256_file(init_checkpoint),
        "forbidden_checkpoints": [str(path) for path in forbidden_checkpoints],
        "released_finetune_checkpoint_loaded": False,
        "dev_only": bool(args.dev_only),
        "eval_accessed": False,
        "source_commit": git_head(MEDGROUNDER_ROOT),
        "compatibility_wrapper": str(Path(__file__).resolve()),
        "compatibility_wrapper_sha256": sha256_file(Path(__file__).resolve()),
        "evaluation_wrapper": str(Path(medgrounder_utils.__file__).resolve()),
        "evaluation_wrapper_sha256": sha256_file(Path(medgrounder_utils.__file__).resolve()),
        "annotation_sha256": sha256_file(Path(cfg.gmpg_mscxr_args.mscxr_annotation_file)),
        "split_file_sha256": {
            "annotation": sha256_file(Path(cfg.gmpg_mscxr_args.mscxr_annotation_file))
        },
        "trainer_status": "paper-and-released-config recipe reconstruction; official repository omits its trainer",
        "source_paper_discrepancy": (
            "released dataset.py broadcasts scalar mean=0.485/std=0.229, while the paper "
            "describes ImageNet normalization; this run follows released source/checkpoint semantics"
        ),
        "dataset_cardinality_audit": cardinality_audit,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    write_json(run_dir / "config.json", config_payload)

    started = time.time()
    best_key: tuple[float, ...] | None = None
    best_row: dict[str, Any] | None = None
    best_epoch = -1
    logs = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        model.train()
        epoch_started = time.time()
        loss_sum = 0.0
        seen_batches = 0
        optimizer_steps = 0
        for batch_index, batch in enumerate(train_loader):
            images = batch["images"].to(device)
            phrases = batch["phrases"]
            targets = [{key: value.to(device, non_blocking=True) for key, value in target.items()} for target in batch["targets"]]
            context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_amp
                else nullcontext()
            )
            with context:
                outputs = model(images, phrases)
                loss_dict = criterion(outputs, targets)
                loss = sum(
                    loss_dict[key] * criterion.weight_dict.get(key, 1.0)
                    for key in loss_dict
                    if key in criterion.weight_dict
                )
                window_start = (batch_index // accumulation_steps) * accumulation_steps
                window_size = min(accumulation_steps, len(train_loader) - window_start)
                scaled_loss = loss / window_size
            scaled_loss.backward()
            loss_sum += float(loss.detach().cpu())
            seen_batches += 1
            should_step = (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(train_loader)
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.clip_max_norm))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

        val_prediction_path = run_dir / "logs" / "val_latest_predictions.csv"
        val_group_path = run_dir / "logs" / "val_latest_per_group.csv"
        val_metrics = medgrounder_utils.evaluate_model(
            cfg,
            model,
            postprocessor,
            val_loader,
            device,
            save_predictions=val_prediction_path,
        )
        val_common = medgrounder_utils.compute_our_phrase_metrics_from_predictions(
            val_prediction_path,
            val_group_path,
            f"medgrounder_{args.protocol}_s{args.seed}_val",
        )
        strict_metric = float(val_metrics.get("precision_f1eq1_overall@0_5", 0.0))
        val_miou = float(val_metrics.get("miou_all", 0.0))
        if args.selection_metric == "protocol_primary":
            if args.protocol == "local_singlebox":
                key = (
                    float(val_common["top1_mean_iou"]),
                    float(val_common["top1_hit_0_5"]),
                )
            else:
                key = (
                    float(val_common["coverage_mean_iou"]),
                    float(val_common["exact_union_iou"]),
                    float(val_common["set_f1_0_5"]),
                )
        else:
            key = (strict_metric, val_miou)
        row = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / max(seen_batches, 1),
            "optimizer_steps": optimizer_steps,
            "learning_rate_main": optimizer.param_groups[0]["lr"],
            "val_precision_f1eq1_overall@0_5": strict_metric,
            "val_miou_all": val_miou,
            "val_common_top1_mean_iou": float(val_common["top1_mean_iou"]),
            "val_common_coverage_mean_iou": float(val_common["coverage_mean_iou"]),
            "val_common_exact_union_iou": float(val_common["exact_union_iou"]),
            "val_common_set_f1_0_3": float(val_common["set_f1_0_3"]),
            "val_common_set_f1_0_5": float(val_common["set_f1_0_5"]),
            "val_common_mean_n_pred": float(val_common["mean_n_pred"]),
            "elapsed_sec": time.time() - epoch_started,
        }
        logs.append(row)
        pd.DataFrame(logs).to_csv(run_dir / "logs" / "train_log.csv", index=False)
        print(json.dumps(row), flush=True)
        if best_key is None or key > best_key:
            best_key = key
            best_row = dict(row)
            best_epoch = epoch + 1
            atomic_save(
                {
                    "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                    "epoch": best_epoch,
                    "val_strict_metric": strict_metric,
                    "val_miou_all": val_miou,
                    "seed": args.seed,
                    "protocol": args.protocol,
                },
                run_dir / "checkpoint" / "best.pth",
            )
            shutil.copyfile(
                val_prediction_path,
                run_dir / "metrics" / "best_val_predictions.csv",
            )
            shutil.copyfile(
                val_group_path,
                run_dir / "metrics" / "best_val_per_group.csv",
            )

    best_checkpoint = run_dir / "checkpoint" / "best.pth"
    if args.dev_only:
        summary = {
            "status": "development_complete",
            "method": "MedGrounder",
            "protocol": args.protocol,
            "seed": args.seed,
            "best_epoch": best_epoch,
            "best_validation_key": list(best_key) if best_key else None,
            "best_validation_metrics": best_row,
            "selection_metric": args.selection_metric,
            "elapsed_sec": time.time() - started,
            "checkpoint": str(best_checkpoint),
            "checkpoint_sha256": sha256_file(best_checkpoint),
            "initialization": str(init_checkpoint),
            "initialization_sha256": sha256_file(init_checkpoint),
            "eval_accessed": False,
        }
        pd.DataFrame([summary]).to_csv(
            run_dir / "metrics" / "development_summary.csv", index=False
        )
        write_json(status_path, summary)
        return summary

    model, _, postprocessor = medgrounder_utils.build_model_and_load(cfg, best_checkpoint, device)
    prediction_path = run_dir / "predictions" / "test_predictions.csv"
    native_metrics = medgrounder_utils.evaluate_model(
        cfg,
        model,
        postprocessor,
        test_loader,
        device,
        save_predictions=prediction_path,
    )
    prediction_contract = audit_prediction_gt_contract(
        Path(cfg.gmpg_mscxr_args.mscxr_annotation_file),
        prediction_path,
        run_dir / "metrics" / "prediction_gt_contract.json",
        int(cfg.imsize),
    )
    common_metrics = medgrounder_utils.compute_our_phrase_metrics_from_predictions(
        prediction_path,
        run_dir / "metrics" / "per_group_metrics.csv",
        f"medgrounder_{args.protocol}_s{args.seed}",
    )
    summary = {
        "status": "complete",
        "method": "MedGrounder",
        "protocol": args.protocol,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_precision_f1eq1_overall@0_5": (
            best_row["val_precision_f1eq1_overall@0_5"] if best_row else None
        ),
        "best_val_miou_all": best_row["val_miou_all"] if best_row else None,
        "best_validation_key": list(best_key) if best_key else None,
        "best_validation_metrics": best_row,
        "elapsed_sec": time.time() - started,
        "checkpoint": str(best_checkpoint),
        "checkpoint_sha256": sha256_file(best_checkpoint),
        "prediction_gt_contract": prediction_contract["status"],
        "initialization": str(init_checkpoint),
        "initialization_sha256": sha256_file(init_checkpoint),
        "eval_accessed": True,
        **{f"native_{key}": value for key, value in native_metrics.items()},
        **{f"common_{key}": value for key, value in common_metrics.items() if key != "method"},
    }
    pd.DataFrame([summary]).to_csv(run_dir / "metrics" / "metrics_summary.csv", index=False)
    write_json(status_path, summary)
    return summary


def main() -> None:
    args = parse_args()
    result = train(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

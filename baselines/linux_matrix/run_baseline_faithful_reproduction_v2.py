#!/usr/bin/env python
"""Run one reproducible baseline job from the v2 protocol registry."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_repro.protocols import sha256_file, write_json  # noqa: E402
from src.baseline_repro.hf_pins import pin_record  # noqa: E402


DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "baseline_faithful_reproduction" / "20260712_v2"
PYTHON = PROJECT_ROOT / ".venv_smm" / "Scripts" / "python.exe"
MEDRPG = PROJECT_ROOT / "third_party" / "MedRPG"
TRANSVG = PROJECT_ROOT / "third_party" / "TransVG"


def git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def split_hashes(split_root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(split_root)).replace("\\", "/"): sha256_file(path)
        for path in sorted(split_root.rglob("*.pth"))
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["medrpg", "transvg", "agpt_release", "m4cxr", "maira2", "vicca"],
        required=True,
    )
    parser.add_argument("--protocol", choices=["local", "official"], default="official")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="")
    return parser.parse_args()


def run_process(command: list[str], log_path: Path, cwd: Path = PROJECT_ROOT) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as handle:
        handle.write("$ " + subprocess.list2cmdline(command) + "\n\n")
        handle.flush()
        process = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return int(process.returncode)


def protocol_paths(root: Path, protocol: str) -> dict[str, Path]:
    official = root / "protocols" / "official_mscxr"
    if protocol == "official":
        return {
            "medrpg_split": official / "medrpg_split_root",
            "transvg_split": official / "transvg_split_root",
            "agpt_split": official / "agpt_split_root" / "MS_CXR",
            "image_root": MEDRPG / "ln_data" / "MS_CXR",
            "vlm_manifest": official / "official_mscxr_vlm.jsonl",
        }
    return {
        "medrpg_split": PROJECT_ROOT
        / "experiments"
        / "ms_cxr_medrpg_fair_retrain_final_v1"
        / "data"
        / "single_box_full_phrase"
        / "split_root",
        "transvg_split": PROJECT_ROOT / "experiments" / "transvg_mscxr_strict_singlebox_fair_v1" / "data",
        "agpt_split": PROJECT_ROOT
        / "experiments"
        / "agpt_baseline_v1"
        / "data"
        / "single_box_p10p19"
        / "split_root"
        / "MS_CXR",
        "image_root": MEDRPG / "ln_data" / "MS_CXR",
        "vlm_manifest": Path(),
    }


def xyxy_iou(left: list[float], right: list[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def score_medrpg_bbox_save(split_root: Path, bbox_save: Path, output_dir: Path) -> dict[str, Any]:
    source_rows = torch.load(split_root / "MS_CXR" / "MS_CXR_test.pth", map_location="cpu", weights_only=False)
    predictions = torch.load(bbox_save, map_location="cpu", weights_only=False)
    expected_ids = {int(source[0]) for source in source_rows}
    prediction_ids = {int(key) for key in predictions}
    duplicate_expected_ids = len(source_rows) - len(expected_ids)
    missing_ids = sorted(expected_ids - prediction_ids)
    unexpected_ids = sorted(prediction_ids - expected_ids)
    id_audit = {
        "status": (
            "PASS"
            if not missing_ids and not unexpected_ids and duplicate_expected_ids == 0
            else "FAIL"
        ),
        "n_source_rows": len(source_rows),
        "n_expected_ids": len(expected_ids),
        "n_prediction_ids": len(prediction_ids),
        "duplicate_expected_ids": duplicate_expected_ids,
        "missing_ids": missing_ids,
        "unexpected_ids": unexpected_ids,
    }
    write_json(output_dir / "prediction_id_contract.json", id_audit)
    if id_audit["status"] != "PASS":
        raise RuntimeError(f"MedRPG prediction anno_id contract failed: {id_audit}")
    rows = []
    for index, source in enumerate(source_rows):
        anno_id, image_id, category_id, image_path, bbox_xywh, width, height, phrase = source
        x, y, box_width, box_height = [float(value) for value in bbox_xywh]
        gold = [x, y, x + box_width, y + box_height]
        payload = predictions[int(anno_id)]
        pred = [float(value) for value in payload.get("pbox", [0.0, 0.0, 0.0, 0.0])]
        iou = xyxy_iou(pred, gold)
        rows.append(
            {
                "row_index": index,
                "anno_id": int(anno_id),
                "image_id": int(image_id),
                "category_id": int(category_id),
                "image_path": str(image_path),
                "phrase": str(phrase),
                "gt_box_xyxy": json.dumps(gold),
                "pred_box_xyxy": json.dumps(pred),
                "iou": iou,
                "hit_0_3": int(iou >= 0.3),
                "hit_0_5": int(iou >= 0.5),
                "prediction_missing": 0,
                "source_width": int(width),
                "source_height": int(height),
            }
        )
    frame = pd.DataFrame(rows)
    pred_path = output_dir / "predictions.csv"
    frame.to_csv(pred_path, index=False)
    summary = {
        "n": int(len(frame)),
        "mean_iou": float(frame["iou"].mean()),
        "median_iou": float(frame["iou"].median()),
        "Hit@0.3": float(frame["hit_0_3"].mean()),
        "Hit@0.5": float(frame["hit_0_5"].mean()),
        "n_missing_predictions": int(frame["prediction_missing"].sum()),
        "prediction_id_contract": "PASS",
        "prediction_file": str(pred_path),
    }
    pd.DataFrame([summary]).to_csv(output_dir / "metrics_summary.csv", index=False)
    write_json(output_dir / "metrics_summary.json", summary)
    return summary


def run_medrpg(args: argparse.Namespace, root: Path, paths: dict[str, Path]) -> dict[str, Any]:
    epochs = args.epochs or 90
    batch_size = args.batch_size or 8
    run_dir = root / "runs" / "medrpg" / args.protocol / f"seed_{args.seed}"
    metrics_path = run_dir / "metrics_summary.json"
    if metrics_path.exists() and not args.force:
        return json.loads(metrics_path.read_text(encoding="utf-8")) | {"status": "skipped_existing"}
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON),
        str(PROJECT_ROOT / "scripts" / "run_medrpg_train_compat_wrapper_v2.py"),
        "--repo-root",
        str(MEDRPG),
        "--split-root",
        str(paths["medrpg_split"]),
        "--data-root",
        str(MEDRPG / "ln_data"),
        "--output-dir",
        str(run_dir / "checkpoint"),
        "--resume",
        str(MEDRPG / "pretrained" / "TransVG_R50_unc.pth"),
        "--epochs",
        str(epochs),
        "--lr-drop",
        "60",
        "--batch-size",
        str(batch_size),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--num-workers",
        "0",
    ]
    initialization = MEDRPG / "pretrained" / "TransVG_R50_unc.pth"
    wrapper = PROJECT_ROOT / "scripts" / "run_medrpg_train_compat_wrapper_v2.py"
    eval_wrapper = PROJECT_ROOT / "scripts" / "run_medrpg_eval_compat_wrapper.py"
    config = {
        "method": "MedRPG",
        "protocol": args.protocol,
        "seed": args.seed,
        "epochs": epochs,
        "lr_drop": 60,
        "batch_size": batch_size,
        "augmentation": ["crop", "scale", "translate"],
        "checkpoint_selection": "validation mean IoU",
        "split_root": str(paths["medrpg_split"]),
        "initialization": str(initialization),
        "initialization_sha256": sha256_file(initialization),
        "source_commit": git_head(MEDRPG),
        "compatibility_wrapper": str(wrapper),
        "compatibility_wrapper_sha256": sha256_file(wrapper),
        "evaluation_wrapper": str(eval_wrapper),
        "evaluation_wrapper_sha256": sha256_file(eval_wrapper),
        "text_encoder_pin": pin_record("bert-base-uncased", require_main_ref=True),
        "split_file_sha256": split_hashes(paths["medrpg_split"]),
        "command": command,
    }
    write_json(run_dir / "config.json", config)
    started = time.time()
    returncode = run_process(command, run_dir / "train.log")
    checkpoint = run_dir / "checkpoint" / "best.pth"
    if returncode != 0 or not checkpoint.exists():
        result = {"status": "failed", "returncode": returncode, "checkpoint_exists": checkpoint.exists()}
        write_json(run_dir / "run_status.json", result)
        return result
    bbox_save = run_dir / "bbox_save.pth"
    eval_command = [
        str(PYTHON),
        str(PROJECT_ROOT / "scripts" / "run_medrpg_eval_compat_wrapper.py"),
        "--repo-root",
        str(MEDRPG),
        "--split-root",
        str(paths["medrpg_split"]),
        "--data-root",
        str(MEDRPG / "ln_data"),
        "--eval-model",
        str(checkpoint),
        "--output-dir",
        str(run_dir / "native_eval"),
        "--eval-set",
        "test",
        "--batch-size",
        str(batch_size),
        "--device",
        args.device,
        "--save-copy",
        str(bbox_save),
    ]
    eval_returncode = run_process(eval_command, run_dir / "eval.log")
    if eval_returncode != 0 or not bbox_save.exists():
        result = {"status": "eval_failed", "returncode": eval_returncode, "checkpoint": str(checkpoint)}
        write_json(run_dir / "run_status.json", result)
        return result
    summary = score_medrpg_bbox_save(paths["medrpg_split"], bbox_save, run_dir)
    result = {
        "status": "complete",
        "elapsed_sec": time.time() - started,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        **summary,
    }
    write_json(run_dir / "run_status.json", result)
    return result


def run_transvg(args: argparse.Namespace, root: Path, paths: dict[str, Path]) -> dict[str, Any]:
    epochs = args.epochs or 90
    batch_size = args.batch_size or 8
    run_dir = root / "runs" / "transvg" / args.protocol / f"seed_{args.seed}"
    metrics_path = run_dir / "eval" / "transvg_best_checkpoint_test_summary.json"
    common_metrics_path = (
        run_dir / "eval_common_val_miou" / "transvg_best_miou_checkpoint_test_summary.json"
    )
    if metrics_path.exists() and common_metrics_path.exists() and not args.force:
        native = json.loads(metrics_path.read_text(encoding="utf-8"))
        common = json.loads(common_metrics_path.read_text(encoding="utf-8"))
        return native | {
            "status": "skipped_existing",
            **{f"common_val_miou_{key}": value for key, value in common.items()},
        }
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON),
        str(PROJECT_ROOT / "scripts" / "run_transvg_train_compat_wrapper_v2.py"),
        "--repo-root",
        str(TRANSVG),
        "--split-root",
        str(paths["transvg_split"]),
        "--data-root",
        str(root / "protocols" / "dummy_data_root"),
        "--output-dir",
        str(run_dir / "checkpoint"),
        "--resume",
        str(MEDRPG / "pretrained" / "TransVG_R50_unc.pth"),
        "--epochs",
        str(epochs),
        "--lr-drop",
        "60",
        "--batch-size",
        str(batch_size),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--num-workers",
        "0",
        "--max-query-len",
        "20",
    ]
    initialization = MEDRPG / "pretrained" / "TransVG_R50_unc.pth"
    wrapper = PROJECT_ROOT / "scripts" / "run_transvg_train_compat_wrapper_v2.py"
    eval_wrapper = PROJECT_ROOT / "scripts" / "evaluate_transvg_mscxr_strict_singlebox_v1.py"
    write_json(
        run_dir / "config.json",
        {
            "method": "TransVG",
            "protocol": args.protocol,
            "seed": args.seed,
            "epochs": epochs,
            "lr_drop": 60,
            "batch_size": batch_size,
            "effective_batch_size": batch_size,
            "distributed_world_size": 1,
            "published_mscxr_recipe_available": False,
            "fidelity_class": "controlled MS-CXR architecture adaptation, not a TransVG paper-result reproduction",
            "augmentation": ["crop", "scale", "translate"],
            "checkpoint_selection": (
                "dual: validation Acc@0.5 (official train.py) and validation mean IoU "
                "(common-selection sensitivity arm)"
            ),
            "split_root": str(paths["transvg_split"]),
            "initialization": str(initialization),
            "initialization_sha256": sha256_file(initialization),
            "source_commit": git_head(TRANSVG),
            "compatibility_wrapper": str(wrapper),
            "compatibility_wrapper_sha256": sha256_file(wrapper),
            "evaluation_wrapper": str(eval_wrapper),
            "evaluation_wrapper_sha256": sha256_file(eval_wrapper),
            "text_encoder_pin": pin_record("bert-base-uncased", require_main_ref=True),
            "split_file_sha256": split_hashes(paths["transvg_split"]),
            "command": command,
        },
    )
    started = time.time()
    returncode = run_process(command, run_dir / "train.log")
    checkpoint = run_dir / "checkpoint" / "best_checkpoint.pth"
    common_checkpoint = run_dir / "checkpoint" / "best_miou_checkpoint.pth"
    if returncode != 0 or not checkpoint.exists() or not common_checkpoint.exists():
        result = {
            "status": "failed",
            "returncode": returncode,
            "checkpoint_exists": checkpoint.exists(),
            "common_val_miou_checkpoint_exists": common_checkpoint.exists(),
        }
        write_json(run_dir / "run_status.json", result)
        return result

    def evaluate_checkpoint(selected_checkpoint: Path, output_dir: Path, log_path: Path) -> int:
        eval_command = [
            str(PYTHON),
            str(PROJECT_ROOT / "scripts" / "evaluate_transvg_mscxr_strict_singlebox_v1.py"),
            "--transvg-root",
            str(TRANSVG),
            "--split-root",
            str(paths["transvg_split"]),
            "--data-root",
            str(root / "protocols" / "dummy_data_root"),
            "--checkpoint",
            str(selected_checkpoint),
            "--output-dir",
            str(output_dir),
            "--eval-set",
            "test",
            "--batch-size",
            str(batch_size),
            "--max-query-len",
            "20",
            "--device",
            args.device,
            "--seed",
            str(args.seed),
        ]
        return run_process(eval_command, log_path)

    eval_returncode = evaluate_checkpoint(checkpoint, run_dir / "eval", run_dir / "eval.log")
    common_eval_returncode = evaluate_checkpoint(
        common_checkpoint,
        run_dir / "eval_common_val_miou",
        run_dir / "eval_common_val_miou.log",
    )
    if (
        eval_returncode != 0
        or common_eval_returncode != 0
        or not metrics_path.exists()
        or not common_metrics_path.exists()
    ):
        result = {
            "status": "eval_failed",
            "returncode": eval_returncode,
            "common_val_miou_returncode": common_eval_returncode,
            "checkpoint": str(checkpoint),
            "common_val_miou_checkpoint": str(common_checkpoint),
        }
        write_json(run_dir / "run_status.json", result)
        return result
    summary = json.loads(metrics_path.read_text(encoding="utf-8"))
    common_summary = json.loads(common_metrics_path.read_text(encoding="utf-8"))
    comparison_rows = [
        {"selection": "official_val_hit_0_5", **summary},
        {"selection": "common_val_mean_iou", **common_summary},
    ]
    pd.DataFrame(comparison_rows).to_csv(run_dir / "checkpoint_selection_sensitivity.csv", index=False)
    result = {
        "status": "complete",
        "elapsed_sec": time.time() - started,
        "checkpoint_sha256": sha256_file(checkpoint),
        "common_val_miou_checkpoint": str(common_checkpoint),
        "common_val_miou_checkpoint_sha256": sha256_file(common_checkpoint),
        **summary,
        **{f"common_val_miou_{key}": value for key, value in common_summary.items()},
    }
    write_json(run_dir / "run_status.json", result)
    return result


def run_agpt_release(args: argparse.Namespace, root: Path, paths: dict[str, Path]) -> dict[str, Any]:
    run_dir = root / "released_official" / "agpt"
    models = args.model or "both"
    command = [
        str(PYTHON),
        str(PROJECT_ROOT / "scripts" / "run_agpt_singlebox_eval_v1.py"),
        "--model",
        models,
        "--image-root",
        str(paths["image_root"]),
        "--split-root",
        str(paths["agpt_split"]),
        "--out-dir",
        str(run_dir),
        "--report-dir",
        str(run_dir),
        "--batch-size",
        str(args.batch_size or 2),
    ]
    returncode = run_process(command, run_dir / "run.log")
    result = {"status": "complete" if returncode == 0 else "failed", "returncode": returncode, "command": command}
    write_json(run_dir / "run_status.json", result)
    return result


def run_vlm(args: argparse.Namespace, root: Path, paths: dict[str, Path]) -> dict[str, Any]:
    if args.protocol != "official":
        raise ValueError("Released VLM reruns in this runner are restricted to the official manifest")
    run_dir = root / "released_official" / "vlm"
    command = [
        str(PYTHON),
        str(PROJECT_ROOT / "scripts" / "run_vlm_reference_baselines.py"),
        "--model",
        args.method,
        "--protocol",
        "official_singlebox_890",
        "--split",
        "test",
        "--output-root",
        str(run_dir),
        "--dataset-manifest",
        str(paths["vlm_manifest"]),
        "--dataset-label",
        "official_mscxr_test167",
        "--max-new-tokens",
        "150",
    ]
    if args.force:
        command.append("--force")
    returncode = run_process(command, run_dir / f"{args.method}.log")
    result = {"status": "complete" if returncode == 0 else "failed", "returncode": returncode, "command": command}
    write_json(run_dir / f"{args.method}_run_status.json", result)
    return result


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    paths = protocol_paths(root, args.protocol)
    missing = [str(path) for key, path in paths.items() if key != "vlm_manifest" and not path.exists()]
    if args.protocol == "official" and not paths["vlm_manifest"].exists():
        missing.append(str(paths["vlm_manifest"]))
    if missing:
        raise FileNotFoundError(f"Run prepare_baseline_faithful_reproduction_v2.py first: {missing}")
    if args.method == "medrpg":
        result = run_medrpg(args, root, paths)
    elif args.method == "transvg":
        result = run_transvg(args, root, paths)
    elif args.method == "agpt_release":
        result = run_agpt_release(args, root, paths)
    else:
        result = run_vlm(args, root, paths)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if result.get("status") in {"failed", "eval_failed"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Re-evaluate the historical YOLO--RAD-DINO hybrid-v4 on direct 888/1444.

This runner deliberately contains no learned candidate ranker and no crop-text
semantic scorer.  It reuses the independently trained seed-specific YOLO and
RAD-DINO artifacts, restores the validation-selected single-box fusion, and
applies the historical raw-phrase switch:

* no multi-region cue -> finding-specific YOLO--RAD-DINO single-box fusion;
* multi-region cue -> target-decomposed multibox-v4 set selection.

The same prediction function is used for direct singlebox_888 and direct
multibox_1444.  Each protocol uses only its own train split for priors, its own
validation split for parameter selection, and its own eval split for metrics.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "CLUEGROUND_VFM_METHOD_PACKAGE_20260713"
for path in (PROJECT_ROOT, PACKAGE_ROOT, PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.baseline_repro.evaluator import evaluate_rows  # noqa: E402
from src.three_task_grounding.contracts import get_protocol  # noqa: E402
from src.three_task_grounding.manifests import write_jsonl  # noqa: E402
from src.three_task_grounding.metrics import summarize_protocol  # noqa: E402
from scripts import run_clueground_vfm_legacy_fusion_unified_3seed_v1 as multi_source  # noqa: E402
from scripts import run_ms_cxr_multibox_rule_context_fusion_v4_yolov8l_pool as hybrid_v4  # noqa: E402
from scripts import run_ms_cxr_singlebox_fair_yolo_dino_fusion_v1 as single_fusion  # noqa: E402
from scripts import run_ms_cxr_singlebox_full_pipeline_3seed_v2 as single_source  # noqa: E402


SEEDS = (13, 42, 2026)
MODELS = ("yolov8s", "yolov8m", "yolo11s", "yolo11m")
PROTOCOL_ROOT = (
    PROJECT_ROOT
    / "training"
    / "three_task_clueground_vfm_finding_conditioned_v2"
    / "protocols"
    / "task_isolated"
)
MULTI_SOURCE_ROOT = PROJECT_ROOT / "experiments" / "clueground_vfm_legacy_fusion_unified_3seed_v1"
SINGLE_SOURCE_ROOT = PROJECT_ROOT / "experiments" / "ms_cxr_singlebox_full_pipeline_3seed_v2"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "clueground_exact_hybrid_v4_direct_3seed_v1"


@dataclass
class ProtocolContext:
    protocol: str
    seed: int
    rows: dict[str, list[dict[str, Any]]]
    inputs: dict[str, list[dict[str, Any]]]
    labels: dict[str, dict[str, list[list[float]]]]
    candidates: dict[str, dict[str, list[dict[str, Any]]]]
    dino: dict[str, dict[str, np.ndarray]]
    priors: dict[str, Any]
    yolo_params: dict[str, dict[str, Any]]
    fusion_params: dict[str, dict[str, Any]]
    provenance: dict[str, Any]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def restore_single_params(seed: int) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    metric_root = SINGLE_SOURCE_ROOT / f"seed_{seed}" / "metrics"
    rule_per = pd.read_csv(metric_root / "rule_context_best_params_by_finding.csv")
    fusion_per = pd.read_csv(metric_root / "fusion_best_weights_by_finding.csv")
    if rule_per.empty or fusion_per.empty:
        raise RuntimeError(f"Missing saved validation selection for singlebox seed {seed}")

    rule_grid = single_fusion.yv2.build_grid(False)
    global_rule_index = int(rule_per.iloc[0]["global_grid_index"])
    yolo_params = {"__global__": dict(rule_grid[global_rule_index])}
    per_rule = {str(row.finding): int(row.best_grid_index) for row in rule_per.itertuples()}
    for finding in single_fusion.CLASS_NAMES:
        yolo_params[finding] = dict(rule_grid[per_rule.get(finding, global_rule_index)])

    fusion_grid = single_fusion.old_fusion.fusion_grid(False)
    global_fusion_index = int(fusion_per.iloc[0]["global_grid_index"])
    fusion_params = {"__global__": dict(fusion_grid[global_fusion_index])}
    per_fusion = {str(row.finding): int(row.best_grid_index) for row in fusion_per.itertuples()}
    for finding in single_fusion.CLASS_NAMES:
        fusion_params[finding] = dict(fusion_grid[per_fusion.get(finding, global_fusion_index)])
    return yolo_params, fusion_params


def load_single_candidates(seed: int) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], list[dict[str, Any]]]:
    output: dict[str, dict[str, list[dict[str, Any]]]] = {}
    provenance = []
    for split in ("val", "eval"):
        parts = []
        for model in MODELS:
            candidate_path, weight_path = single_source.yolo_artifacts(seed, split, model)
            parts.append(single_source.load_candidates(candidate_path))
            provenance.append(
                {
                    "split": split,
                    "model": model,
                    "candidate_path": str(candidate_path),
                    "candidate_sha256": sha256(candidate_path),
                    "weight_path": str(weight_path),
                    "weight_sha256": sha256(weight_path),
                }
            )
        output[split] = single_fusion.merge_candidates(*parts)
    return output, provenance


def load_single_context(seed: int) -> ProtocolContext:
    rows = {split: single_fusion.row_dicts(split) for split in ("train", "val", "eval")}
    labels = {
        split: {str(row["task_id"]): [list(map(float, row["gold_bbox_xyxy"]))] for row in part}
        for split, part in rows.items()
    }
    inputs: dict[str, list[dict[str, Any]]] = {}
    for split, part in rows.items():
        inputs[split] = []
        for row in part:
            gid = str(row["task_id"])
            row["group_id"] = gid
            row["gold_boxes_xyxy"] = labels[split][gid]
            inputs[split].append(
                {
                    "protocol_key": "singlebox_888",
                    "group_id": gid,
                    "subject_id": str(row["subject_id"]),
                    "study_id": str(row["study_id"]),
                    "dicom_id": str(row["dicom_id"]),
                    "image_path": str(row["image_path"]),
                    "image_width": int(row["image_width"]),
                    "image_height": int(row["image_height"]),
                    "finding": str(row["finding"]),
                    "query_text": str(row["claim_sentence"]),
                }
            )

    candidates, candidate_provenance = load_single_candidates(seed)
    single_source.configure_rad_paths(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dino = {
        split: single_fusion.dino_map(single_fusion.load_rad_prediction("full_phrase", split, device, False))
        for split in ("val", "eval")
    }
    priors = single_fusion.ybase.make_train_priors(rows["train"])
    yolo_params, fusion_params = restore_single_params(seed)
    rad_checkpoint = (
        SINGLE_SOURCE_ROOT
        / f"seed_{seed}"
        / "training"
        / "rad_dino"
        / "checkpoints"
        / "rad_dino_full_phrase_singlebox.pt"
    )
    return ProtocolContext(
        "singlebox_888",
        seed,
        rows,
        inputs,
        labels,
        candidates,
        dino,
        priors,
        yolo_params,
        fusion_params,
        {
            "source_run": str(SINGLE_SOURCE_ROOT / f"seed_{seed}"),
            "candidate_artifacts": candidate_provenance,
            "rad_checkpoint": str(rad_checkpoint),
            "rad_checkpoint_sha256": sha256(rad_checkpoint),
        },
    )


def validate_canonical_1444(
    inputs: dict[str, list[dict[str, Any]]],
    labels: dict[str, dict[str, list[list[float]]]],
) -> dict[str, dict[str, int]]:
    expected_groups = {"train": 813, "val": 124, "eval": 220}
    expected_boxes = {"train": 996, "val": 164, "eval": 280}
    groups = {split: len(rows) for split, rows in inputs.items()}
    boxes = {
        split: sum(len(labels[split][str(row["group_id"])]) for row in rows)
        for split, rows in inputs.items()
    }
    if groups != expected_groups or boxes != expected_boxes:
        raise RuntimeError(
            "Canonical 1444 membership mismatch: "
            f"groups={groups}, boxes={boxes}, expected_groups={expected_groups}, expected_boxes={expected_boxes}"
        )
    return {"groups": groups, "gt_boxes": boxes}


def load_multi_dino(
    seed: int,
    source_ids: dict[str, dict[str, list[str]]],
    multi_source_root: Path,
) -> dict[str, dict[str, np.ndarray]]:
    path = multi_source_root / f"seed_{seed}" / "mscxr_multibox_1444" / "rad_dino_legacy" / "predictions_by_split.npz"
    archive = np.load(path, allow_pickle=True)
    output: dict[str, dict[str, np.ndarray]] = {}
    for split in ("val", "eval"):
        task_map = {
            str(task_id): np.asarray(box, dtype=np.float32)
            for task_id, box in zip(archive[f"{split}_ids"], archive[f"{split}_boxes"])
        }
        output[split] = multi_source.group_dino_map(task_map, source_ids[split])
    return output


def load_multi_context(
    seed: int,
    protocol_root: Path = PROTOCOL_ROOT,
    multi_source_root: Path = MULTI_SOURCE_ROOT,
    canonical_v3: bool = False,
) -> ProtocolContext:
    inputs, labels, source_ids = multi_source.load_protocol(protocol_root)
    membership = validate_canonical_1444(inputs, labels) if canonical_v3 else None
    rows = {split: multi_source.group_rows(inputs[split], labels[split], split) for split in inputs}
    root = multi_source_root / f"seed_{seed}" / "mscxr_multibox_1444"
    candidates = {
        split: multi_source.load_yolo_candidates(root / "yolo_predictions", split)
        for split in ("val", "eval")
    }
    dino = load_multi_dino(seed, source_ids, multi_source_root)
    priors = single_fusion.ybase.make_train_priors(multi_source.expanded_prior_rows(rows["train"]))
    fusion_root = root / "legacy_fusion"
    yolo_params = json.loads((fusion_root / "yolo_params.json").read_text(encoding="utf-8"))
    fusion_params = json.loads((fusion_root / "fusion_params.json").read_text(encoding="utf-8"))
    yolo_weights = sorted((root / "yolo_runs").glob("*/weights/best.pt"))
    rad_archive = root / "rad_dino_legacy" / "predictions_by_split.npz"
    return ProtocolContext(
        "multibox_1444",
        seed,
        rows,
        inputs,
        labels,
        candidates,
        dino,
        priors,
        yolo_params,
        fusion_params,
        {
            "source_run": str(multi_source_root / f"seed_{seed}"),
            "protocol_root": str(protocol_root),
            "canonical_v3": bool(canonical_v3),
            "canonical_membership": membership,
            "yolo_weights": [
                {"path": str(path), "sha256": sha256(path)} for path in yolo_weights
            ],
            "rad_prediction_archive": str(rad_archive),
            "rad_prediction_archive_sha256": sha256(rad_archive),
        },
    )


def make_groups(context: ProtocolContext, split: str) -> dict[str, dict[str, Any]]:
    groups = {}
    for row in context.rows[split]:
        gid = str(row["group_id"])
        groups[gid] = {
            "group_id": gid,
            "task_ids": [gid],
            "dicom_id": str(row["dicom_id"]),
            "subject_id": str(row.get("subject_id", "")),
            "study_id": str(row.get("study_id", "")),
            "image_path": str(row["image_path"]),
            "finding": str(row["finding"]),
            "class_id": int(single_fusion.CLASS_TO_ID[str(row["finding"])]),
            "claim_sentence": str(row["claim_sentence"]),
            "image_width": int(row["image_width"]),
            "image_height": int(row["image_height"]),
            "split": split,
            "gt_boxes": context.labels[split][gid],
        }
    return groups


def expand_validation_rows(
    context: ProtocolContext,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Represent every validation annotation box as one calibration row."""

    rows = []
    dino_by_row: dict[str, np.ndarray] = {}
    for source in context.rows["val"]:
        gid = str(source["group_id"])
        for index, box in enumerate(context.labels["val"][gid]):
            task_id = f"{gid}::calibration_box_{index}"
            row = {
                **source,
                "task_id": task_id,
                "sample_id": task_id,
                "gold_bbox_xyxy": [float(value) for value in box],
            }
            rows.append(row)
            if gid in context.dino["val"]:
                dino_by_row[task_id] = context.dino["val"][gid]
    return rows, dino_by_row


def retune_single_route_full_validation(
    context: ProtocolContext,
    run_root: Path,
    quick: bool,
) -> int:
    calibration_rows, calibration_dino = expand_validation_rows(context)
    yolo_params, yolo_grid, yolo_per_finding = single_fusion.yv2.tune(
        calibration_rows,
        context.candidates["val"],
        context.priors,
        quick,
    )
    fusion_params, fusion_grid, fusion_per_finding = single_fusion.old_fusion.tune_fusion(
        calibration_rows,
        context.candidates["val"],
        context.priors,
        calibration_dino,
        yolo_params,
        quick,
    )
    context.yolo_params = yolo_params
    context.fusion_params = fusion_params
    calibration_root = run_root / "single_route_fullval_calibration"
    calibration_root.mkdir(parents=True, exist_ok=True)
    yolo_grid.to_csv(calibration_root / "yolo_rule_val_grid.csv", index=False)
    yolo_per_finding.to_csv(calibration_root / "yolo_rule_by_finding.csv", index=False)
    fusion_grid.to_csv(calibration_root / "fusion_val_grid.csv", index=False)
    fusion_per_finding.to_csv(calibration_root / "fusion_by_finding.csv", index=False)
    write_json(calibration_root / "yolo_params.json", yolo_params)
    write_json(calibration_root / "fusion_params.json", fusion_params)
    write_json(
        calibration_root / "calibration_contract.json",
        {
            "protocol": context.protocol,
            "seed": context.seed,
            "selection_split": "val only",
            "n_validation_phrase_groups": len(context.rows["val"]),
            "n_validation_box_rows": len(calibration_rows),
            "uses_eval_labels": False,
            "uses_gold_count_at_inference": False,
        },
    )
    return len(calibration_rows)


def load_cached_single_route_calibration(
    context: ProtocolContext,
    cache_root: Path,
    run_root: Path,
) -> int:
    calibration_root = (
        cache_root
        / context.protocol
        / f"seed_{context.seed}"
        / "single_route_fullval_calibration"
    )
    contract = json.loads((calibration_root / "calibration_contract.json").read_text(encoding="utf-8"))
    context.yolo_params = json.loads((calibration_root / "yolo_params.json").read_text(encoding="utf-8"))
    context.fusion_params = json.loads((calibration_root / "fusion_params.json").read_text(encoding="utf-8"))
    destination = run_root / "single_route_fullval_calibration"
    destination.mkdir(parents=True, exist_ok=True)
    for source in calibration_root.iterdir():
        if source.is_file():
            shutil.copy2(source, destination / source.name)
    contract["reused_from"] = str(calibration_root)
    write_json(destination / "calibration_contract.json", contract)
    return int(contract["n_validation_box_rows"])


def single_route(context: ProtocolContext, split: str) -> dict[str, list[dict[str, Any]]]:
    frame = single_fusion.old_fusion.evaluate_fusion(
        context.rows[split],
        context.candidates[split],
        context.priors,
        context.dino[split],
        context.yolo_params,
        context.fusion_params,
        "historical_single_route",
        split,
    )
    result = {}
    for row in frame.itertuples():
        result[str(row.task_id)] = [
            {
                "box": [float(row.pred_x1), float(row.pred_y1), float(row.pred_x2), float(row.pred_y2)],
                "score": float(row.confidence),
                "source_model": str(getattr(row, "source", "legacy_fusion")),
            }
        ]
    return result


def run_hybrid(
    context: ProtocolContext,
    split: str,
    set_params: dict[str, Any],
    multi_route_yolo_params: dict[str, dict[str, Any]],
) -> tuple[dict[str, list[list[float]]], pd.DataFrame]:
    groups = make_groups(context, split)
    multi_predictions, cue_rows = hybrid_v4.predict_groups(
        groups,
        context.candidates[split],
        context.priors,
        multi_route_yolo_params,
        context.dino[split],
        set_params,
    )
    singleton_predictions = single_route(context, split)
    cue_by_gid = {str(row["group_id"]): bool(row.get("has_multi_cue", False)) for row in cue_rows}
    outputs: dict[str, list[list[float]]] = {}
    audit = []
    cue_info = {str(row["group_id"]): row for row in cue_rows}
    for gid in groups:
        use_multi = cue_by_gid.get(gid, False)
        source = multi_predictions.get(gid, []) if use_multi else singleton_predictions.get(gid, [])
        boxes = [[float(value) for value in candidate["box"]] for candidate in source]
        outputs[gid] = boxes
        info = cue_info.get(gid, {})
        audit.append(
            {
                "group_id": gid,
                "split": split,
                "finding": groups[gid]["finding"],
                "phrase": groups[gid]["claim_sentence"],
                "route": "multibox_v4" if use_multi else "singlebox_yolo_dino_fusion",
                "has_multi_cue": use_multi,
                "cue_text": info.get("cue_text", "single_or_unspecified"),
                "k_hint": int(info.get("k_hint", 1)),
                "n_gt": len(groups[gid]["gt_boxes"]),
                "n_pred": len(boxes),
            }
        )
    return outputs, pd.DataFrame(audit)


def evaluator_payload(context: ProtocolContext, split: str, outputs: dict[str, list[list[float]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    samples = []
    predictions = []
    by_gid = {str(row["group_id"]): row for row in context.inputs[split]}
    for gid, gold in context.labels[split].items():
        source = by_gid[gid]
        samples.append(
            {
                "query_id": gid,
                "subject_id": str(source.get("subject_id", "")),
                "finding": str(source.get("finding", "")),
                "gold_boxes": gold,
            }
        )
        predictions.append({"query_id": gid, "pred_boxes": outputs.get(gid, [])})
    return samples, predictions


def evaluate_context(context: ProtocolContext, outputs: dict[str, list[list[float]]]) -> tuple[dict[str, Any], pd.DataFrame]:
    samples, predictions = evaluator_payload(context, "eval", outputs)
    common, detail = evaluate_rows(samples, predictions, force_single_box=context.protocol == "singlebox_888")
    if context.protocol == "multibox_1444":
        native, _ = summarize_protocol(
            get_protocol("mscxr_multibox_1444"),
            context.inputs["eval"],
            context.labels["eval"],
            outputs,
        )
        common.update(
            {
                "coverage_mean_iou": float(native["coverage_mean_iou"]),
                "exact_union_iou": float(native["exact_union_iou"]),
                "hull_union_iou_diagnostic": float(native["hull_union_iou_diagnostic"]),
                "set_f1_0_3": float(native["set_f1_0_3"]),
                "set_f1_0_5": float(native["set_f1_0_5"]),
                "mean_pred_count": float(native["mean_pred_count"]),
            }
        )
    return common, pd.DataFrame(detail)


def split_audit(contexts: Iterable[ProtocolContext]) -> dict[str, Any]:
    result = {}
    for context in contexts:
        identities = {
            field: {
                split: {str(row[field]) for row in context.inputs[split]}
                for split in ("train", "val", "eval")
            }
            for field in ("subject_id", "study_id", "dicom_id", "group_id")
        }
        result[context.protocol] = {
            f"{left}_{right}_{field}_overlap": len(identities[field][left] & identities[field][right])
            for field in identities
            for left, right in (("train", "val"), ("train", "eval"), ("val", "eval"))
        }
    result["status"] = "PASS" if all(
        value == 0
        for protocol, rows in result.items()
        if protocol != "status"
        for value in rows.values()
    ) else "FAIL"
    return result


def aggregate(per_seed: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    metric_columns = [
        column
        for column in (
            "mean_iou",
            "Hit@0.3",
            "Hit@0.5",
            "coverage_mean_iou",
            "exact_union_iou",
            "hull_union_iou_diagnostic",
            "set_f1_0_3",
            "set_f1_0_5",
            "set_f1_optimal_0_3",
            "set_f1_optimal_0_5",
            "mean_pred_count",
            "raw_mean_pred_count",
        )
        if column in per_seed.columns
    ]
    rows = []
    for protocol, part in per_seed.groupby("protocol"):
        row: dict[str, Any] = {"protocol": protocol, "n_seeds": len(part)}
        for metric in metric_columns:
            values = pd.to_numeric(part[metric], errors="coerce").dropna()
            if len(values):
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std(ddof=0))
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "aggregate_metrics.csv", index=False)
    return frame


def run_protocol(
    context: ProtocolContext,
    output_root: Path,
    quick: bool,
    retune_single_full_val: bool,
    separate_multi_route_params: bool,
    calibration_cache_root: Path | None,
) -> dict[str, Any]:
    run_root = output_root / context.protocol / f"seed_{context.seed}"
    restored_yolo_params = copy.deepcopy(context.yolo_params)
    n_single_calibration_rows = len(context.rows["val"])
    if retune_single_full_val:
        if calibration_cache_root is not None:
            n_single_calibration_rows = load_cached_single_route_calibration(
                context,
                calibration_cache_root,
                run_root,
            )
        else:
            n_single_calibration_rows = retune_single_route_full_validation(context, run_root, quick)
    multi_route_yolo_params = (
        restored_yolo_params
        if retune_single_full_val and separate_multi_route_params
        else context.yolo_params
    )
    val_groups = make_groups(context, "val")
    set_params, val_grid = hybrid_v4.tune_set_params(
        val_groups,
        context.candidates["val"],
        context.priors,
        multi_route_yolo_params,
        context.dino["val"],
        quick,
    )
    run_root.mkdir(parents=True, exist_ok=True)
    val_grid.to_csv(run_root / "multibox_v4_val_grid.csv", index=False)
    write_json(run_root / "selected_set_params.json", set_params)

    outputs, cue_audit = run_hybrid(
        context,
        "eval",
        set_params,
        multi_route_yolo_params,
    )
    cue_audit.to_csv(run_root / "eval_route_audit.csv", index=False)
    write_jsonl(
        run_root / "eval_predictions.jsonl",
        [
            {"protocol": context.protocol, "group_id": gid, "pred_boxes_xyxy": boxes}
            for gid, boxes in sorted(outputs.items())
        ],
    )
    metrics, detail = evaluate_context(context, outputs)
    metrics["raw_mean_pred_count"] = float(np.mean([len(boxes) for boxes in outputs.values()]))
    metrics["n_raw_multi_box_predictions"] = int(sum(len(boxes) > 1 for boxes in outputs.values()))
    detail.to_csv(run_root / "eval_detail_common_evaluator.csv", index=False)
    result = {
        "status": "complete",
        "method": (
            "historical_yolo_rad_dino_hybrid_v4_fullval_calibrated"
            if retune_single_full_val
            else "historical_yolo_rad_dino_hybrid_v4"
        ),
        "protocol": context.protocol,
        "seed": context.seed,
        "upstream_seed_scope": "four YOLO detectors and phrase-conditioned RAD-DINO head",
        "query_contract": "finding category + raw phrase",
        "route_contract": "raw phrase cue only; never gold count",
        "single_route": "finding-specific YOLO calibration + RAD-DINO agreement + prior + coordinate blend",
        "multi_route": "target decomposition + NMS/diversity + validation-selected set thresholds",
        "candidate_ranker": "none; historical hand-scored fusion",
        "hgb_used": False,
        "siglip_used": False,
        "biomedclip_used": False,
        "cig_task_pretraining": False,
        "selection_split": "protocol-specific validation only",
        "single_route_calibration": "all validation annotation rows" if retune_single_full_val else "restored saved parameters",
        "n_single_route_calibration_rows": n_single_calibration_rows,
        "calibration_cache_root": str(calibration_cache_root) if calibration_cache_root else None,
        "separate_route_parameterization": bool(separate_multi_route_params),
        "multi_route_yolo_params": "restored multibox-v4 inputs" if separate_multi_route_params else "shared with single route",
        "set_params": set_params,
        "n_eval": len(outputs),
        "n_multi_route": int((cue_audit["route"] == "multibox_v4").sum()),
        "n_single_route": int((cue_audit["route"] == "singlebox_yolo_dino_fusion").sum()),
        "provenance": context.provenance,
        **metrics,
    }
    write_json(run_root / "RUN_STATUS.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--protocols", nargs="+", choices=("singlebox_888", "multibox_1444"), default=("singlebox_888", "multibox_1444"))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--protocol-root", type=Path, default=PROTOCOL_ROOT)
    parser.add_argument("--multi-source-root", type=Path, default=MULTI_SOURCE_ROOT)
    parser.add_argument(
        "--canonical-v3",
        action="store_true",
        help="Require canonical 1444 membership: 813/124/220 groups and 996/164/280 GT boxes.",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--retune-single-full-val", action="store_true")
    parser.add_argument("--separate-multi-route-params", action="store_true")
    parser.add_argument("--calibration-cache-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    contexts = []
    results = []
    for protocol in args.protocols:
        for seed in args.seeds:
            context = (
                load_single_context(seed)
                if protocol == "singlebox_888"
                else load_multi_context(
                    seed,
                    args.protocol_root,
                    args.multi_source_root,
                    args.canonical_v3,
                )
            )
            contexts.append(context)
            results.append(
                run_protocol(
                    context,
                    args.output_root,
                    args.quick,
                    args.retune_single_full_val,
                    args.separate_multi_route_params,
                    args.calibration_cache_root,
                )
            )

    per_seed = pd.DataFrame(results)
    scalar_columns = [column for column in per_seed.columns if not per_seed[column].map(lambda value: isinstance(value, (dict, list))).any()]
    per_seed[scalar_columns].to_csv(args.output_root / "per_seed_metrics.csv", index=False)
    aggregate_frame = aggregate(per_seed[scalar_columns], args.output_root)
    audit = split_audit(contexts)
    write_json(args.output_root / "split_overlap_audit.json", audit)
    final = {
        "status": "PASS" if audit["status"] == "PASS" and len(results) == len(args.protocols) * len(args.seeds) else "FAIL",
        "method": (
            "historical_yolo_rad_dino_hybrid_v4_fullval_calibrated"
            if args.retune_single_full_val
            else "historical_yolo_rad_dino_hybrid_v4"
        ),
        "protocols": list(args.protocols),
        "seeds": list(args.seeds),
        "canonical_v3": bool(args.canonical_v3),
        "protocol_root": str(args.protocol_root),
        "multi_source_root": str(args.multi_source_root),
        "same_method_code_for_both_protocols": True,
        "hgb_used": False,
        "semantic_crop_text_scorers_used": False,
        "cig_task_pretraining": False,
        "single_route_full_validation_calibration": bool(args.retune_single_full_val),
        "separate_route_parameterization": bool(args.separate_multi_route_params),
        "aggregate_metrics": json.loads(aggregate_frame.to_json(orient="records")),
        "split_overlap_audit": audit,
    }
    write_json(args.output_root / "FINAL_STATUS.json", final)
    print(json.dumps(final, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

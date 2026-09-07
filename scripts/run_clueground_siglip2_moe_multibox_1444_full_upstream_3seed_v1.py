#!/usr/bin/env python
"""Evaluate YOLOs + RAD-DINO + frozen SigLIP2 on multibox-1444.

The same two-expert finding-conditioned MoE used by the direct-888 runner is
applied here. Multi-region phrases keep the hybrid-v4 set decoder; singleton
phrases may be blended by the hybrid/SigLIP2 gate.

No BioMedCLIP and no original SigLIP checkpoint are used.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import clueground_direct_upstream_helpers_v1 as upstream_helpers  # noqa: E402
from scripts import clueground_siglip2_moe_common_v1 as common  # noqa: E402
from scripts import run_clueground_exact_hybrid_v4_direct_3seed_v1 as exact  # noqa: E402
from scripts import run_clueground_vfm_legacy_fusion_unified_3seed_v1 as multi_source  # noqa: E402
from scripts import audit_methodology_integrity_v1 as integrity  # noqa: E402
from scripts import clueground_siglip2_gate_v1 as gate  # noqa: E402

SEEDS = (13, 42, 2026)
DEFAULT_HYBRID_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "clueground_exact_hybrid_v4_route_specific_3seed_v3"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "clueground_siglip2_moe_multibox_1444_full_upstream_3seed_v1"
)


@dataclass
class MultiSeedUpstream:
    context: exact.ProtocolContext
    hybrid: dict[str, dict[str, list[dict[str, Any]]]]
    cue: dict[str, pd.DataFrame]


@dataclass
class SigLIP2Expert:
    predictions: dict[str, dict[str, list[dict[str, Any]]]]
    provenance: dict[str, Any]



def add_train_artifacts(context: exact.ProtocolContext) -> None:
    root = (
        exact.MULTI_SOURCE_ROOT
        / f"seed_{context.seed}"
        / "mscxr_multibox_1444"
    )
    context.candidates["train"] = multi_source.load_yolo_candidates(
        root / "yolo_predictions", "train"
    )
    _inputs, _labels, source_ids = multi_source.load_protocol(exact.PROTOCOL_ROOT)
    archive = np.load(
        root / "rad_dino_legacy" / "predictions_by_split.npz",
        allow_pickle=True,
    )
    task_map = {
        str(task_id): np.asarray(box, dtype=np.float32)
        for task_id, box in zip(archive["train_ids"], archive["train_boxes"])
    }
    context.dino["train"] = multi_source.group_dino_map(
        task_map, source_ids["train"]
    )


def load_raw_1024_candidates(root: Path, seed: int, split: str) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Minimal proposal-path adapter for the established 1444 SigLIP2 runner."""
    path = root / f"seed_{seed}" / "raw" / f"{split}_candidates.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing raw 1024 candidate file: {path}")
    frame = pd.read_csv(path)
    required = {"dicom_id", "class_id", "score", "x1", "y1", "x2", "y2", "source_model", "rank"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"Invalid raw 1024 schema {path}: missing {missing}")
    if frame["source_model"].astype(str).str.contains("wbf", case=False, na=False).any():
        raise RuntimeError(f"WBF candidate detected in raw-only input: {path}")
    candidates: dict[str, list[dict[str, Any]]] = {}
    for row in frame.to_dict("records"):
        candidates.setdefault(str(row["dicom_id"]), []).append({
            "box": [float(row[key]) for key in ("x1", "y1", "x2", "y2")],
            "score": float(row["score"]),
            "class_id": int(row["class_id"]),
            "rank": int(row["rank"]),
            "source_model": str(row["source_model"]),
        })
    return candidates, {
        "seed": seed, "split": split, "model": "four_model_raw_1024_ensemble",
        "candidate_path": str(path), "candidate_sha256": common.sha256_file(path),
        "candidate_mode": "raw", "imgsz": 1024,
    }


def load_hybrid_upstream(seed: int, args: argparse.Namespace) -> MultiSeedUpstream:
    context = exact.load_multi_context(seed)
    add_train_artifacts(context)
    if args.raw_yolo_candidate_root is not None:
        records = []
        for split in ("train", "val", "eval"):
            candidates, record = load_raw_1024_candidates(
                args.raw_yolo_candidate_root, seed, split
            )
            context.candidates[split] = candidates
            records.append(record)
        context.provenance["candidate_artifacts"] = records
    restored_multi_yolo = copy.deepcopy(context.yolo_params)
    run_root = args.hybrid_root / "multibox_1444" / f"seed_{seed}"
    calibration_root = run_root / "single_route_fullval_calibration"
    required = [
        calibration_root / "yolo_params.json",
        calibration_root / "fusion_params.json",
        run_root / "selected_set_params.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing hybrid-v4 validation artifacts: {missing}")
    context.yolo_params = json.loads(
        (calibration_root / "yolo_params.json").read_text(encoding="utf-8")
    )
    context.fusion_params = json.loads(
        (calibration_root / "fusion_params.json").read_text(encoding="utf-8")
    )
    set_params = json.loads(
        (run_root / "selected_set_params.json").read_text(encoding="utf-8")
    )

    hybrid_maps: dict[str, dict[str, list[dict[str, Any]]]] = {}
    cue_maps: dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "eval"):
        outputs, cue = exact.run_hybrid(
            context,
            split,
            set_params,
            restored_multi_yolo,
        )
        hybrid_maps[split] = {
            str(group_id): [
                {
                    "box": [float(value) for value in box],
                    "score": float(max(0.0, 1.0 - index * 1e-6)),
                    "source": f"multibox1444_hybrid_v4_seed_{seed}",
                }
                for index, box in enumerate(boxes)
            ]
            for group_id, boxes in outputs.items()
        }
        cue_maps[split] = cue.copy()

    context.provenance["hybrid_artifact_root"] = str(run_root)
    context.provenance["selected_set_params"] = set_params
    return MultiSeedUpstream(context=context, hybrid=hybrid_maps, cue=cue_maps)


def make_bundle(
    upstream: MultiSeedUpstream,
    semantic: SigLIP2Expert,
    split: str,
) -> gate.ExpertBundle:
    groups = exact.make_groups(upstream.context, split)
    expected = set(groups)
    for name, mapping in (
        ("hybrid", upstream.hybrid[split]),
        ("siglip2", semantic.predictions[split]),
    ):
        missing = expected.difference(mapping)
        if missing:
            raise RuntimeError(
                f"{name}/{split} missing {len(missing)} groups; first={sorted(missing)[:3]}"
            )
    return gate.ExpertBundle(
        groups=groups,
        hybrid=upstream.hybrid[split],
        siglip2=semantic.predictions[split],
        cue=upstream.cue[split],
    )


def cue_lookup(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {
        str(row["group_id"]): {
            "has_multi_cue": bool(row.get("has_multi_cue", False)),
            "k_hint": int(row.get("k_hint", 1)),
        }
        for _, row in frame.iterrows()
    }


def predict_sets(
    fused: pd.DataFrame,
    groups: dict[str, dict[str, Any]],
    cue_frame: pd.DataFrame,
    params: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    candidates = common.candidate_map(fused, groups)
    cues = cue_lookup(cue_frame)
    output: dict[str, list[dict[str, Any]]] = {}
    for group_id in groups:
        cue = cues.get(group_id, {"has_multi_cue": False, "k_hint": 1})
        output[group_id] = common.decode_phrase_conditioned(
            candidates.get(group_id, []),
            groups[group_id]["claim_sentence"],
            cue,
            params,
            iou_fn=gate.mb.iou_xyxy,
        )
    return output


def prediction_boxes(
    predictions: dict[str, list[dict[str, Any]]],
) -> dict[str, list[list[float]]]:
    return {
        str(group_id): [
            [float(value) for value in row["box"]] for row in rows
        ]
        for group_id, rows in predictions.items()
    }


def save_predictions(
    path: Path,
    predictions: dict[str, list[dict[str, Any]]],
) -> None:
    boxes = prediction_boxes(predictions)
    rows = [
        {
            "group_id": group_id,
            "pred_boxes_json": json.dumps(values),
            "n_pred": len(values),
        }
        for group_id, values in boxes.items()
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def summarize_frame(
    frame: pd.DataFrame,
    method: str,
    subset: str,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "method": method,
        "subset": subset,
        "n_groups": int(len(frame)),
        "n_gt_boxes": int(frame["n_gt"].sum()),
        "coverage_mean_iou": float(frame["coverage_mean_iou"].mean()),
        "enclosing_hull_iou": float(frame["enclosing_hull_iou"].mean()),
        "exact_rectangle_union_iou": float(frame["rectangle_union_iou"].mean()),
        "raster_union_iou_224": float(frame["raster_union_iou_224"].mean()),
        "set_f1_0_3": float(frame["set_f1_greedy_0_3"].mean()),
        "set_f1_0_5": float(frame["set_f1_greedy_0_5"].mean()),
        "mean_pred_count": float(frame["n_pred"].mean()),
        "exact_count_rate": float((frame["n_pred"] == frame["n_gt"]).mean()),
        "overprediction_rate": float((frame["n_pred"] > frame["n_gt"]).mean()),
        "underprediction_rate": float((frame["n_pred"] < frame["n_gt"]).mean()),
    }
    if len(frame) and (frame["n_gt"] == 1).all():
        summary.update(
            {
                "top1_mean_iou": float(frame["top1_iou"].mean()),
                "top1_hit_0_3": float((frame["top1_iou"] >= 0.3).mean()),
                "top1_hit_0_5": float((frame["top1_iou"] >= 0.5).mean()),
            }
        )
    return summary


def evaluate_map(
    method: str,
    groups: dict[str, dict[str, Any]],
    predictions: dict[str, list[dict[str, Any]]],
    source: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    boxes = prediction_boxes(predictions)
    bundle = integrity.MethodPredictions(
        protocol="unified_variable_cardinality",
        method=method,
        status="controlled_mscxr_only_retrain",
        source_path=source,
        preds=boxes,
    )
    rows, _ = integrity.evaluate_bundle(bundle, groups)
    frame = pd.DataFrame(rows)
    frame["top1_iou"] = [
        integrity.box_iou(groups[group_id]["gt_boxes"][0], boxes.get(group_id, [])[0])
        if groups[group_id]["gt_boxes"] and boxes.get(group_id, [])
        else 0.0
        for group_id in frame["group_id"].astype(str)
    ]
    subsets = {
        "all_220": frame,
        "singleton_163": frame[frame["n_gt"].astype(int).eq(1)],
        "multibox_57": frame[frame["n_gt"].astype(int).gt(1)],
    }
    summaries = [
        summarize_frame(part, method, subset)
        for subset, part in subsets.items()
    ]
    return frame, summaries


def evaluate(
    method: str,
    groups: dict[str, dict[str, Any]],
    predictions: dict[str, list[dict[str, Any]]],
    root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=True)
    save_predictions(root / "predictions.csv", predictions)
    detail, summaries = evaluate_map(
        method,
        groups,
        predictions,
        str(root / "predictions.csv"),
    )
    detail.to_csv(root / "per_group.csv", index=False)
    summary = next(row for row in summaries if row["subset"] == "all_220")
    common.write_json(root / "summary.json", summary)
    return detail, summary


def tune_candidate_fusion_params(
    val_scored: pd.DataFrame,
    val_groups: dict[str, dict[str, Any]],
    val_cue: pd.DataFrame,
    output_root: Path,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Tune SigLIP2 candidate scoring on singleton validation groups only."""

    cues = cue_lookup(val_cue)
    eligible = {
        group_id: group
        for group_id, group in val_groups.items()
        if len(group["gt_boxes"]) == 1
        and not cues.get(group_id, {}).get("has_multi_cue", False)
    }
    if not eligible:
        raise RuntimeError("No singleton validation groups for SigLIP2 fusion tuning")

    rows: list[dict[str, Any]] = []
    for w_siglip2_z in (-0.10, -0.05, 0.0, 0.03, 0.06, 0.10, 0.15, 0.22):
        for w_siglip2_rank in (0.0, 0.03, 0.06, 0.10):
            for w_prior in (0.0, 0.05, 0.10):
                for w_xattn in (0.0, 0.05, 0.10):
                    params = {
                        "w_head": 1.0,
                        "w_siglip2_z": w_siglip2_z,
                        "w_siglip2_rank": w_siglip2_rank,
                        "w_prior": w_prior,
                        "w_xattn": w_xattn,
                        "w_conf": 0.03,
                    }
                    fused = common.apply_candidate_fusion_score(val_scored, params)
                    candidates = common.candidate_map(fused, val_groups)
                    ious = []
                    for group_id, group in eligible.items():
                        rows_for_group = candidates.get(group_id, [])
                        iou = (
                            0.0
                            if not rows_for_group
                            else float(
                                gate.mb.iou_xyxy(
                                    rows_for_group[0]["box"], group["gt_boxes"][0]
                                )
                            )
                        )
                        ious.append(iou)
                    values = np.asarray(ious, dtype=np.float64)
                    rows.append(
                        {
                            **params,
                            "n_singleton_val": int(len(values)),
                            "mean_iou": float(values.mean()),
                            "Hit@0.3": float((values >= 0.3).mean()),
                            "Hit@0.5": float((values >= 0.5).mean()),
                        }
                    )
    grid = pd.DataFrame(rows).sort_values(
        ["mean_iou", "Hit@0.5", "Hit@0.3"], ascending=False
    )
    output_root.mkdir(parents=True, exist_ok=True)
    grid.to_csv(output_root / "siglip2_candidate_fusion_val_grid.csv", index=False)
    keys = (
        "w_head",
        "w_siglip2_z",
        "w_siglip2_rank",
        "w_prior",
        "w_xattn",
        "w_conf",
    )
    return {key: float(grid.iloc[0][key]) for key in keys}, grid


def tune_set_params(
    val_fused: pd.DataFrame,
    val_groups: dict[str, dict[str, Any]],
    val_cue: pd.DataFrame,
    output_root: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Tune only the multibox decoder after candidate-fusion weights are frozen."""

    rows: list[dict[str, Any]] = []
    for nms_iou in (0.35, 0.50, 0.65):
        for score_ratio in (0.0, 0.55, 0.70, 0.85):
            for min_k in (1, 2):
                for max_k in (2, 3):
                    if max_k < min_k:
                        continue
                    params = {
                        "nms_iou": nms_iou,
                        "score_ratio": score_ratio,
                        "min_k_if_cue": min_k,
                        "max_k_if_cue": max_k,
                    }
                    predictions = predict_sets(
                        val_fused, val_groups, val_cue, params
                    )
                    _detail, summaries = evaluate_map(
                        "val_siglip2_grid",
                        val_groups,
                        predictions,
                        "validation_grid_in_memory",
                    )
                    summary = next(
                        row for row in summaries if row["subset"] == "all_220"
                    )
                    rows.append(
                        {
                            **params,
                            "coverage_mean_iou": float(summary["coverage_mean_iou"]),
                            "exact_union_iou": float(
                                summary["exact_rectangle_union_iou"]
                            ),
                            "set_f1_0_3": float(summary["set_f1_0_3"]),
                            "set_f1_0_5": float(summary["set_f1_0_5"]),
                            "mean_pred_count": float(summary["mean_pred_count"]),
                        }
                    )
    grid = pd.DataFrame(rows).sort_values(
        ["set_f1_0_3", "coverage_mean_iou", "exact_union_iou", "set_f1_0_5"],
        ascending=False,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    grid.to_csv(output_root / "siglip2_set_decoder_val_grid.csv", index=False)
    keys = ("nms_iou", "score_ratio", "min_k_if_cue", "max_k_if_cue")
    best = {key: float(grid.iloc[0][key]) for key in keys}
    best["min_k_if_cue"] = int(best["min_k_if_cue"])
    best["max_k_if_cue"] = int(best["max_k_if_cue"])
    return best, grid


def prepare_siglip2_expert(
    upstream: MultiSeedUpstream,
    source_seed: int,
    args: argparse.Namespace,
) -> SigLIP2Expert:
    root = args.output_root / "siglip2_experts" / f"source_seed_{source_seed}"
    root.mkdir(parents=True, exist_ok=True)
    contract_base = {
        "protocol": "multibox_1444",
        "semantic_source_seed": source_seed,
        "model_id": args.siglip2_model_id,
        "revision": args.siglip2_revision,
        "prompt_mode": args.prompt_mode,
        "crop_margin": float(args.crop_margin),
        "max_candidates_per_task": int(args.max_candidates_per_task),
        "siglip2_batch_size": int(args.siglip2_batch_size),
        "siglip2_dtype": args.siglip2_dtype,
        "scoring_recipe": "siglip2_fixres_pairwise_diagonal_v2",
        "candidate_fusion_recipe": "legacy_siglip_grid_with_siglip2_scores_v1",
        "set_decoder_recipe": "common_decode_phrase_conditioned_v1",
        "selection_split": "multibox-1444 validation only",
        "experts": common.EXPERTS,
        "biomedclip_used": False,
        "original_siglip_used": False,
    }
    contract_path = root / "SIGLIP2_CACHE_CONTRACT.json"
    status_path = root / "SIGLIP2_EXPERT_STATUS.json"

    groups = {
        split: exact.make_groups(upstream.context, split)
        for split in ("train", "val", "eval")
    }
    base_frames: dict[str, pd.DataFrame] = {}
    base_paths: dict[str, Path] = {}
    for split in ("train", "val", "eval"):
        base_path = root / f"{split}_semantic_base_candidates.csv"
        if base_path.exists() and not args.force_semantic_candidates:
            base = pd.read_csv(base_path)
        else:
            base = upstream_helpers.candidate_table_for_split(
                upstream, split, args.max_candidates_per_task
            )
            base.to_csv(base_path, index=False)
        base_frames[split] = base
        base_paths[split] = base_path

    contract = {
        **contract_base,
        "candidate_sha256_by_split": {
            split: common.sha256_file(path) for split, path in base_paths.items()
        },
    }
    if contract_path.exists() and not (
        args.force_siglip2 or args.force_semantic_candidates
    ):
        observed = json.loads(contract_path.read_text(encoding="utf-8"))
        if observed != contract:
            raise RuntimeError(
                "SigLIP2 cache contract or candidate pool changed. Re-run with "
                "--force-siglip2 --force-semantic-candidates."
            )

    scored: dict[str, pd.DataFrame] = {}
    missing_score_frames: dict[str, pd.DataFrame] = {}
    score_paths: dict[str, Path] = {}
    model_provenance: dict[str, Any] | None = None
    if status_path.exists() and not args.force_siglip2:
        previous_status = json.loads(status_path.read_text(encoding="utf-8"))
        model_provenance = previous_status.get("model_provenance")
    for split in ("train", "val", "eval"):
        scored_path = root / f"{split}_siglip2_scored_candidates.csv"
        score_paths[split] = scored_path
        if scored_path.exists() and not args.force_siglip2:
            scored[split] = pd.read_csv(scored_path)
        else:
            missing_score_frames[split] = base_frames[split]

    if missing_score_frames:
        newly_scored, model_provenance = common.score_siglip2_frames(
            missing_score_frames,
            model_id=args.siglip2_model_id,
            revision=args.siglip2_revision,
            prompt_mode=args.prompt_mode,
            margin=args.crop_margin,
            batch_size=args.siglip2_batch_size,
            device_name=args.device,
            dtype_name=args.siglip2_dtype,
        )
        for split, frame in newly_scored.items():
            scored[split] = frame
            frame.to_csv(score_paths[split], index=False)

    fusion_params, _ = tune_candidate_fusion_params(
        scored["val"], groups["val"], upstream.cue["val"], root
    )
    val_fused = common.apply_candidate_fusion_score(
        scored["val"], fusion_params
    )
    set_params, _ = tune_set_params(
        val_fused, groups["val"], upstream.cue["val"], root
    )
    predictions: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for split in ("train", "val", "eval"):
        fused = common.apply_candidate_fusion_score(
            scored[split], fusion_params
        )
        fused.to_csv(root / f"{split}_siglip2_fusion_candidates.csv", index=False)
        predictions[split] = predict_sets(
            fused, groups[split], upstream.cue[split], set_params
        )

    provenance = {
        **contract,
        "selected_candidate_fusion_params": fusion_params,
        "selected_set_decoder_params": set_params,
        "model_provenance": model_provenance,
    }
    common.write_json(contract_path, contract)
    common.write_json(status_path, provenance)
    return SigLIP2Expert(predictions=predictions, provenance=provenance)


def run_seed(
    seed: int,
    upstream: MultiSeedUpstream,
    semantic: SigLIP2Expert,
    args: argparse.Namespace,
) -> dict[str, Any]:
    upstream_helpers.set_seed(seed)
    bundles = {
        split: make_bundle(upstream, semantic, split)
        for split in ("train", "val", "eval")
    }
    seed_root = args.output_root / f"seed_{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)

    model, params, _, _ = gate.train_gate(
        f"finding_siglip2_moe_multibox1444_s{seed}",
        bundles["train"],
        bundles["val"],
        common.EXPERTS,
        seed=seed,
        device=args.device,
        output_root=seed_root,
    )
    val_gate, val_gate_audit = gate.predict_gate(
        model, bundles["val"], common.EXPERTS, device=args.device
    )
    eval_gate, eval_gate_audit = gate.predict_gate(
        model, bundles["eval"], common.EXPERTS, device=args.device
    )
    (seed_root / "val").mkdir(parents=True, exist_ok=True)
    (seed_root / "eval").mkdir(parents=True, exist_ok=True)
    val_gate_audit.to_csv(seed_root / "val" / "gate_prediction_audit.csv", index=False)
    eval_gate_audit.to_csv(seed_root / "eval" / "gate_prediction_audit.csv", index=False)

    _, val_base = evaluate(
        f"base_s{seed}",
        bundles["val"].groups,
        bundles["val"].hybrid,
        seed_root / "val" / "base",
    )
    _, val_gate_summary = evaluate(
        f"gate_s{seed}",
        bundles["val"].groups,
        val_gate,
        seed_root / "val" / "gate",
    )
    use_gate = (
        float(val_gate_summary["coverage_mean_iou"]),
        float(val_gate_summary["exact_rectangle_union_iou"]),
        float(val_gate_summary["set_f1_0_5"]),
    ) >= (
        float(val_base["coverage_mean_iou"]),
        float(val_base["exact_rectangle_union_iou"]),
        float(val_base["set_f1_0_5"]),
    )
    selected_name = "gate" if use_gate else "base"
    selected = eval_gate if use_gate else bundles["eval"].hybrid
    _, eval_summary = evaluate(
        f"selected_{selected_name}_s{seed}",
        bundles["eval"].groups,
        selected,
        seed_root / "eval" / selected_name,
    )

    mutated = copy.deepcopy(bundles["eval"])
    for group in mutated.groups.values():
        group["gt_boxes"] = [[0.0, 0.0, 1.0, 1.0]]
    mutated_predictions, _ = gate.predict_gate(
        model, mutated, common.EXPERTS, device=args.device
    )
    gold_independent = gate.prediction_maps_equal(eval_gate, mutated_predictions)

    result = {
        "status": "complete",
        "protocol": "multibox_1444",
        "method": "YOLOs + RAD-DINO + frozen SigLIP2 finding-conditioned MoE",
        "seed": seed,
        "selected_variant": selected_name,
        "selection_split": "multibox-1444 validation only",
        "gate_best_epoch": int(params["epoch"]),
        "train_gate_rows": int(params["train_rows"]),
        "val_gate_rows": int(params["val_rows"]),
        "val_base_coverage": float(val_base["coverage_mean_iou"]),
        "val_gate_coverage": float(val_gate_summary["coverage_mean_iou"]),
        "coverage_mean_iou": float(eval_summary["coverage_mean_iou"]),
        "exact_union_iou": float(eval_summary["exact_rectangle_union_iou"]),
        "hull_union_iou": float(eval_summary["enclosing_hull_iou"]),
        "set_f1_0_3": float(eval_summary["set_f1_0_3"]),
        "set_f1_0_5": float(eval_summary["set_f1_0_5"]),
        "mean_pred_count": float(eval_summary["mean_pred_count"]),
        "gold_mutation_independence": bool(gold_independent),
        "biomedclip_used": False,
        "original_siglip_used": False,
        "siglip2_used": True,
        "upstream_provenance": upstream.context.provenance,
        "siglip2_provenance": semantic.provenance,
    }
    common.write_json(seed_root / "RUN_STATUS.json", result)
    return result


def split_audit(upstreams: Iterable[MultiSeedUpstream]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for upstream in upstreams:
        context = upstream.context
        subjects = {
            split: {str(row["subject_id"]) for row in context.inputs[split]}
            for split in ("train", "val", "eval")
        }
        result[f"seed_{context.seed}"] = {
            "counts": {split: len(context.rows[split]) for split in ("train", "val", "eval")},
            "train_val_subject_overlap": len(subjects["train"] & subjects["val"]),
            "train_eval_subject_overlap": len(subjects["train"] & subjects["eval"]),
            "val_eval_subject_overlap": len(subjects["val"] & subjects["eval"]),
        }
    expected = {"train": 814, "val": 125, "eval": 220}
    passed = all(
        row["counts"] == expected
        and row["train_val_subject_overlap"] == 0
        and row["train_eval_subject_overlap"] == 0
        and row["val_eval_subject_overlap"] == 0
        for row in result.values()
    )
    result["status"] = "PASS" if passed else "FAIL"
    return result


def aggregate(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "complete",
        "method": "YOLOs + RAD-DINO + frozen SigLIP2 finding-conditioned MoE",
        "protocol": "multibox_1444",
        "experts": common.EXPERTS,
        "n_upstream_seeds": int(len(frame)),
        "n_gate_seeds": int(len(frame)),
        "seeds": [int(value) for value in frame["seed"].tolist()],
        "uses_prediction_ensemble": False,
        "uses_eval_for_selection": False,
        "biomedclip_used": False,
        "original_siglip_used": False,
        "siglip2_used": True,
        "selected_variants": frame["selected_variant"].tolist(),
        "gold_mutation_independence_pass": bool(
            frame["gold_mutation_independence"].all()
        ),
    }
    for metric in (
        "coverage_mean_iou", "exact_union_iou", "hull_union_iou",
        "set_f1_0_3", "set_f1_0_5", "mean_pred_count",
    ):
        values = frame[metric].astype(float).to_numpy()
        result[f"{metric}_mean"] = float(values.mean())
        result[f"{metric}_std"] = (
            float(values.std(ddof=1)) if len(values) > 1 else 0.0
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--hybrid-root", type=Path, default=DEFAULT_HYBRID_ROOT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument(
        "--semantic-source-mode",
        choices=("paired", "fixed"),
        default="paired",
        help="paired is the paper-facing full-upstream setting",
    )
    parser.add_argument("--semantic-source-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-candidates-per-task", type=int, default=12)
    parser.add_argument(
        "--raw-yolo-candidate-root",
        type=Path,
        default=None,
        help="1024 raw candidate root (seed_<n>/raw/*.csv); WBF rows are rejected.",
    )
    parser.add_argument("--siglip2-batch-size", type=int, default=16)
    parser.add_argument("--siglip2-model-id", default=common.SIGLIP2_MODEL_ID)
    parser.add_argument("--siglip2-revision", default=None)
    parser.add_argument(
        "--siglip2-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("claim", "cxr_claim", "finding_claim", "region_prompt"),
        default="cxr_claim",
    )
    parser.add_argument("--crop-margin", type=float, default=0.15)
    parser.add_argument("--force-siglip2", action="store_true")
    parser.add_argument("--force-semantic-candidates", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    common.write_json(
        args.output_root / "RUN_CONFIG.json",
        {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "method_contract": "YOLOs + RAD-DINO + SigLIP2 only; no BioMedCLIP/CLIP expert",
        },
    )

    upstream_cache: dict[int, MultiSeedUpstream] = {}
    semantic_cache: dict[int, SigLIP2Expert] = {}
    results: list[dict[str, Any]] = []
    required = set(args.seeds)
    if args.semantic_source_mode == "fixed":
        required.add(args.semantic_source_seed)
    for seed in sorted(required):
        upstream_cache[seed] = load_hybrid_upstream(seed, args)

    for seed in args.seeds:
        semantic_seed = (
            seed if args.semantic_source_mode == "paired" else args.semantic_source_seed
        )
        if semantic_seed not in semantic_cache:
            semantic_cache[semantic_seed] = prepare_siglip2_expert(
                upstream_cache[semantic_seed], semantic_seed, args
            )
        results.append(
            run_seed(
                seed,
                upstream_cache[seed],
                semantic_cache[semantic_seed],
                args,
            )
        )

    frame = pd.DataFrame(results)
    scalar_columns = [
        column
        for column in frame.columns
        if not frame[column].map(lambda value: isinstance(value, (dict, list))).any()
    ]
    frame[scalar_columns].to_csv(args.output_root / "per_seed_metrics.csv", index=False)
    final = aggregate(frame)
    final["semantic_source_mode"] = args.semantic_source_mode
    final["semantic_source_seed"] = (
        args.semantic_source_seed if args.semantic_source_mode == "fixed" else None
    )
    final["seed_pairing"] = [
        {
            "hybrid_seed": int(seed),
            "gate_seed": int(seed),
            "siglip2_candidate_seed": int(
                seed if args.semantic_source_mode == "paired" else args.semantic_source_seed
            ),
        }
        for seed in args.seeds
    ]
    final["split_overlap_audit"] = split_audit(
        upstream_cache[seed] for seed in args.seeds
    )
    final["status"] = (
        "complete"
        if final["split_overlap_audit"]["status"] == "PASS"
        and final["gold_mutation_independence_pass"]
        else "failed_audit"
    )
    common.write_json(args.output_root / "FINAL_STATUS.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

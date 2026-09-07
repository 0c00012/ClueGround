#!/usr/bin/env python
"""Run a one-semantic-expert ablation of the verified 1444 semantic MoE.

This is an exact clone of the verified full-upstream three-seed semantic MoE:
it keeps the same hybrid route, frozen semantic tables, splits, calibration,
and validation-only selection. The only intervention is retaining either
SigLIP or BioMedCLIP alongside the hybrid expert.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import audit_unified_finding_annotation_dependency_v2 as finding_audit  # noqa: E402
from scripts import run_clueground_exact_hybrid_v4_direct_3seed_v1 as exact  # noqa: E402
from scripts import run_clueground_moe_transplant_diagnostic_3seed_v1 as transplant  # noqa: E402
from scripts import run_clueground_vfm_legacy_fusion_unified_3seed_v1 as multi_source  # noqa: E402
from scripts import run_final_methodology_verification_v1 as verify  # noqa: E402
from scripts import run_ms_cxr_trainable_moe_gate_v1 as moe  # noqa: E402


SEEDS = (13, 42, 2026)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "experiments"
    / "clueground_finding_moe_full_upstream_siglip_only_3seed_v1"
)
BIOMED_OUTPUT = (
    PROJECT_ROOT
    / "experiments"
    / "clueground_finding_moe_full_upstream_biomed_only_3seed_v1"
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def finding_gate_feature(
    bundle: moe.ExpertBundle,
    group_id: str,
    experts: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    result = finding_audit._legacy_gate_feature(
        bundle,
        group_id,
        experts,
        finding_audit.FINDINGS,
    )
    if result is None:
        return None
    feature, boxes = result
    return feature, boxes, np.zeros(len(experts), dtype=np.float32)


@torch.no_grad()
def predict_gate(
    model: moe.MoEGate,
    bundle: moe.ExpertBundle,
    experts: list[str],
) -> dict[str, list[dict[str, Any]]]:
    """Predict with a gate whose feature dimension follows the expert subset."""
    model.eval().to("cpu")
    multi = moe.has_multi_cue(bundle.cue)
    output: dict[str, list[dict[str, Any]]] = {}
    for group_id, group in bundle.groups.items():
        prediction = bundle.hybrid.get(group_id, [])
        result = finding_gate_feature(bundle, group_id, experts)
        if result is not None and not multi.get(group_id, False):
            values, boxes, _ = result
            weights = torch.softmax(model(torch.from_numpy(values[None, :]).float()), dim=-1)
            weights_array = weights.detach().cpu().numpy()[0]
            box_norm = (boxes * weights_array[:, None]).sum(axis=0)
            box_norm[:2] = np.clip(box_norm[:2], 0.0, 1.0)
            box_norm[2:] = np.clip(box_norm[2:], 1e-4, 1.0)
            prediction = [{
                "box": moe.norm_to_xyxy(
                    box_norm, float(group["image_width"]), float(group["image_height"])
                ),
                "score": float(weights_array.max()),
                "source": f"{experts[1]}_only_gate",
            }]
        output[group_id] = prediction
    return output


def add_train_artifacts(context: exact.ProtocolContext) -> None:
    root = (
        exact.MULTI_SOURCE_ROOT
        / f"seed_{context.seed}"
        / "mscxr_multibox_1444"
    )
    context.candidates["train"] = multi_source.load_yolo_candidates(
        root / "yolo_predictions",
        "train",
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
        task_map,
        source_ids["train"],
    )


def load_seed_hybrid(
    seed: int,
) -> tuple[exact.ProtocolContext, dict[str, dict[str, list[list[float]]]]]:
    context = exact.load_multi_context(seed)
    add_train_artifacts(context)
    restored_multi_yolo = copy.deepcopy(context.yolo_params)
    run_root = transplant.BASE_ROOT / "multibox_1444" / f"seed_{seed}"
    calibration = run_root / "single_route_fullval_calibration"
    context.yolo_params = json.loads(
        (calibration / "yolo_params.json").read_text(encoding="utf-8")
    )
    context.fusion_params = json.loads(
        (calibration / "fusion_params.json").read_text(encoding="utf-8")
    )
    set_params = json.loads(
        (run_root / "selected_set_params.json").read_text(encoding="utf-8")
    )
    outputs = {}
    for split in ("train", "val", "eval"):
        outputs[split], _ = exact.run_hybrid(
            context,
            split,
            set_params,
            restored_multi_yolo,
        )
    return context, outputs


def evaluate(
    method: str,
    bundle: moe.ExpertBundle,
    predictions: dict[str, list[dict[str, Any]]],
    root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    verify.save_predictions(root / "predictions.csv", predictions)
    detail, summaries = verify.evaluate_map(
        method,
        bundle.groups,
        predictions,
        str(root / "predictions.csv"),
    )
    detail.to_csv(root / "per_group.csv", index=False)
    return detail, next(row for row in summaries if row["subset"] == "all_220")


def maps_equal(
    left: dict[str, list[dict[str, Any]]],
    right: dict[str, list[dict[str, Any]]],
) -> bool:
    return verify.maps_equal(left, right)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--semantic-expert",
        choices=("siglip", "biomed"),
        default="siglip",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    output_root = args.output_root or (
        DEFAULT_OUTPUT if args.semantic_expert == "siglip" else BIOMED_OUTPUT
    )
    output_root.mkdir(parents=True, exist_ok=True)

    original_feature_builder = moe.gate_feature_for_group
    moe.gate_feature_for_group = finding_gate_feature
    rows: list[dict[str, Any]] = []
    try:
        for seed in SEEDS:
            context, outputs = load_seed_hybrid(seed)
            bundles = {
                split: moe.build_bundle(
                    split,
                    args.device,
                    "disabled_for_mscxr_only_retrain",
                    include_pretrain_expert=False,
                )
                for split in ("train", "val", "eval")
            }
            for split, bundle in bundles.items():
                bundle.hybrid = transplant.remap_hybrid(
                    context,
                    split,
                    outputs[split],
                    bundle,
                )

            seed_root = output_root / f"seed_{seed}"
            moe.CKPT = seed_root / "checkpoints"
            moe.LOG = seed_root / "logs"
            moe.MET = seed_root / "training_metrics"
            moe.PRED = seed_root / "predictions"
            for path in (moe.CKPT, moe.LOG, moe.MET, moe.PRED):
                path.mkdir(parents=True, exist_ok=True)

            # Exact ablation: retain one frozen semantic expert.
            experts = ["hybrid", args.semantic_expert]
            model, params, _, _ = moe.train_gate(
                f"finding_moe_{args.semantic_expert}_only_s{seed}",
                bundles["train"],
                bundles["val"],
                experts,
                hardneg=False,
                seed=seed,
                device=args.device,
            )
            val_gate, val_audit = predict_gate(model, bundles["val"], experts), None
            eval_gate = predict_gate(model, bundles["eval"], experts)

            _, val_base_summary = evaluate(
                f"base_s{seed}",
                bundles["val"],
                bundles["val"].hybrid,
                seed_root / "val" / "base",
            )
            _, val_gate_summary = evaluate(
                f"gate_s{seed}",
                bundles["val"],
                val_gate,
                seed_root / "val" / "gate",
            )
            use_gate = (
                float(val_gate_summary["coverage_mean_iou"]),
                float(val_gate_summary["exact_rectangle_union_iou"]),
                float(val_gate_summary["set_f1_0_5"]),
            ) >= (
                float(val_base_summary["coverage_mean_iou"]),
                float(val_base_summary["exact_rectangle_union_iou"]),
                float(val_base_summary["set_f1_0_5"]),
            )
            selected = eval_gate if use_gate else bundles["eval"].hybrid
            selected_name = "gate" if use_gate else "base"
            _, eval_summary = evaluate(
                f"selected_{selected_name}_s{seed}",
                bundles["eval"],
                selected,
                seed_root / "eval" / selected_name,
            )

            mutated = copy.deepcopy(bundles["eval"])
            mutated.groups = verify.mutate_gold(mutated.groups)
            mutated_predictions = predict_gate(model, mutated, experts)
            gold_independent = maps_equal(eval_gate, mutated_predictions)
            rows.append(
                {
                    "seed": seed,
                    "selected_variant": selected_name,
                    "selection_split": "val only",
                    "gate_best_epoch": int(params["epoch"]),
                    "train_gate_rows": int(params["train_rows"]),
                    "val_gate_rows": int(params["val_rows"]),
                    "val_base_coverage": float(val_base_summary["coverage_mean_iou"]),
                    "val_gate_coverage": float(val_gate_summary["coverage_mean_iou"]),
                    "coverage_mean_iou": float(eval_summary["coverage_mean_iou"]),
                    "exact_union_iou": float(eval_summary["exact_rectangle_union_iou"]),
                    "set_f1_0_3": float(eval_summary["set_f1_0_3"]),
                    "set_f1_0_5": float(eval_summary["set_f1_0_5"]),
                    "mean_pred_count": float(eval_summary["mean_pred_count"]),
                    "gold_mutation_independence": gold_independent,
                }
            )
    finally:
        moe.gate_feature_for_group = original_feature_builder

    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "per_seed_metrics.csv", index=False)
    metrics = [
        "coverage_mean_iou",
        "exact_union_iou",
        "set_f1_0_3",
        "set_f1_0_5",
        "mean_pred_count",
    ]
    aggregate: dict[str, Any] = {
        "status": "complete",
        "method": (
            "finding-conditioned YOLO-RAD-DINO + "
            f"{'SigLIP' if args.semantic_expert == 'siglip' else 'BioMedCLIP'} MoE"
        ),
        "classification": "MS-CXR task-supervised semantic extension",
        "query_contract": "finding category + raw phrase",
        "n_upstream_seeds": 3,
        "n_gate_seeds": 3,
        "seed_pairing": "13/13, 42/42, 2026/2026",
        "gate_retrained_for_each_upstream_seed": True,
        "selection_split": "1444 validation only",
        "uses_eval_for_selection": False,
        "semantic_expert": args.semantic_expert,
        "siglip_used": args.semantic_expert == "siglip",
        "biomedclip_used": args.semantic_expert == "biomed",
        "forbidden_gold_features": [],
        "gold_mutation_independence_pass": bool(frame["gold_mutation_independence"].all()),
        "selected_variants": frame["selected_variant"].tolist(),
    }
    for metric in metrics:
        aggregate[f"{metric}_mean"] = float(frame[metric].mean())
        aggregate[f"{metric}_std"] = float(frame[metric].std(ddof=1))
    write_json(output_root / "FINAL_STATUS.json", aggregate)


if __name__ == "__main__":
    main()

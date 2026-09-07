#!/usr/bin/env python
"""Transplant the leak-fixed semantic MoE onto three YOLO--RAD-DINO seeds.

This is deliberately a diagnostic, not a full-pipeline reproduction.  The
YOLO/RAD-DINO hybrid varies over seeds 13/42/2026, while the MoE checkpoints
were trained once on the historical upstream expert bundle.  A paired seed is
used for each upstream run, and whether to apply the gate is decided on the
corresponding 1444 validation split only.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_clueground_exact_hybrid_v4_direct_3seed_v1 as exact  # noqa: E402
from scripts import run_final_methodology_verification_v1 as verify  # noqa: E402
from scripts import run_ms_cxr_trainable_moe_gate_v1 as moe  # noqa: E402
from scripts import audit_unified_finding_annotation_dependency_v2 as finding_audit  # noqa: E402


SEEDS = (13, 42, 2026)
BASE_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "clueground_exact_hybrid_v4_route_specific_3seed_v3"
)
GATE_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "final_methodology_verification"
    / "20260712_v1"
    / "moe_clean"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "experiments"
    / "clueground_moe_transplant_diagnostic_3seed_v1"
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_text(value: Any) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value).lower()).split())


def canonical_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["dicom_id"]),
        normalize_text(row["finding"]),
        normalize_text(row["claim_sentence"]),
    )


def load_current_outputs(
    seed: int,
) -> tuple[exact.ProtocolContext, dict[str, dict[str, list[list[float]]]]]:
    context = exact.load_multi_context(seed)
    restored_multi_yolo = copy.deepcopy(context.yolo_params)
    run_root = BASE_ROOT / "multibox_1444" / f"seed_{seed}"
    calibration_root = run_root / "single_route_fullval_calibration"
    context.yolo_params = json.loads(
        (calibration_root / "yolo_params.json").read_text(encoding="utf-8")
    )
    context.fusion_params = json.loads(
        (calibration_root / "fusion_params.json").read_text(encoding="utf-8")
    )
    set_params = json.loads(
        (run_root / "selected_set_params.json").read_text(encoding="utf-8")
    )
    outputs: dict[str, dict[str, list[list[float]]]] = {}
    for split in ("val", "eval"):
        outputs[split], _ = exact.run_hybrid(
            context,
            split,
            set_params,
            restored_multi_yolo,
        )
    return context, outputs


def remap_hybrid(
    context: exact.ProtocolContext,
    split: str,
    outputs: dict[str, list[list[float]]],
    bundle: moe.ExpertBundle,
) -> dict[str, list[dict[str, Any]]]:
    current_to_key = {
        str(row["group_id"]): canonical_key(row)
        for row in context.rows[split]
    }
    key_to_old = {
        canonical_key(group): gid
        for gid, group in bundle.groups.items()
    }
    if set(current_to_key.values()) != set(key_to_old):
        raise RuntimeError(f"Canonical group mapping mismatch for {split}")
    remapped: dict[str, list[dict[str, Any]]] = {}
    for current_gid, boxes in outputs.items():
        old_gid = key_to_old[current_to_key[current_gid]]
        remapped[old_gid] = [
            {
                "box": [float(value) for value in box],
                "score": float(max(0.0, 1.0 - 1e-6 * index)),
                "source": f"current_hybrid_seed_{context.seed}",
            }
            for index, box in enumerate(boxes)
        ]
    if set(remapped) != set(bundle.groups):
        raise RuntimeError(f"Prediction coverage mismatch for {split}")
    return remapped


def load_gate(seed: int, device: str) -> moe.MoEGate:
    path = GATE_ROOT / "checkpoints" / f"hybrid_siglip_biomedclip_s{seed}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = moe.MoEGate(int(payload["in_dim"]), int(payload["n_experts"]))
    model.load_state_dict(payload["state"])
    return model.to(device)


def predict_legacy_finding_gate(
    model: moe.MoEGate,
    bundle: moe.ExpertBundle,
) -> tuple[dict[str, list[dict[str, Any]]], pd.DataFrame]:
    predictions = finding_audit._legacy_predict(
        model,
        bundle,
        finding_audit.FINDINGS,
    )
    multi = moe.has_multi_cue(bundle.cue)
    rows = []
    for gid in bundle.groups:
        base = bundle.hybrid.get(gid, [])
        pred = predictions.get(gid, [])
        changed = len(base) != len(pred)
        if not changed and base:
            changed = bool(
                np.max(
                    np.abs(
                        np.asarray([item["box"] for item in base], dtype=float)
                        - np.asarray([item["box"] for item in pred], dtype=float)
                    )
                )
                > 1e-8
            )
        rows.append(
            {
                "group_id": gid,
                "has_multi_cue": bool(multi.get(gid, False)),
                "action": "finding_conditioned_moe_blend" if changed else "keep_hybrid",
            }
        )
    return predictions, pd.DataFrame(rows)


def summary_map(
    method: str,
    bundle: moe.ExpertBundle,
    predictions: dict[str, list[dict[str, Any]]],
    output_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    verify.save_predictions(output_path, predictions)
    detail, summaries = verify.evaluate_map(
        method,
        bundle.groups,
        predictions,
        str(output_path),
    )
    all_row = next(row for row in summaries if row["subset"] == "all_220")
    return detail, all_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        context, current = load_current_outputs(seed)
        bundles = {
            split: moe.build_bundle(
                split,
                args.device,
                "disabled_for_mscxr_only_diagnostic",
                include_pretrain_expert=False,
            )
            for split in ("val", "eval")
        }
        for split, bundle in bundles.items():
            bundle.hybrid = remap_hybrid(context, split, current[split], bundle)

        model = load_gate(seed, args.device)
        experts = ["hybrid", "siglip", "biomed"]
        predictions: dict[str, dict[str, list[dict[str, Any]]]] = {}
        summaries: dict[str, dict[str, dict[str, Any]]] = {}
        for split, bundle in bundles.items():
            gate_pred, audit = predict_legacy_finding_gate(model, bundle)
            predictions[split] = {"base": bundle.hybrid, "gate": gate_pred}
            summaries[split] = {}
            for variant in ("base", "gate"):
                root = args.output_root / f"seed_{seed}" / split / variant
                detail, summary = summary_map(
                    f"{variant}_s{seed}",
                    bundle,
                    predictions[split][variant],
                    root / "predictions.csv",
                )
                detail.to_csv(root / "per_group.csv", index=False)
                summaries[split][variant] = summary
            audit.to_csv(
                args.output_root / f"seed_{seed}" / split / "gate_action_audit.csv",
                index=False,
            )

        val_base = summaries["val"]["base"]
        val_gate = summaries["val"]["gate"]
        use_gate = (
            float(val_gate["coverage_mean_iou"]),
            float(val_gate["exact_rectangle_union_iou"]),
            float(val_gate["set_f1_0_5"]),
        ) >= (
            float(val_base["coverage_mean_iou"]),
            float(val_base["exact_rectangle_union_iou"]),
            float(val_base["set_f1_0_5"]),
        )
        selected = "gate" if use_gate else "base"
        eval_summary = summaries["eval"][selected]
        rows.append(
            {
                "seed": seed,
                "paired_gate_seed": seed,
                "selection_split": "val only",
                "selected_variant": selected,
                "val_base_coverage": float(val_base["coverage_mean_iou"]),
                "val_gate_coverage": float(val_gate["coverage_mean_iou"]),
                "eval_coverage_mean_iou": float(eval_summary["coverage_mean_iou"]),
                "eval_exact_union_iou": float(eval_summary["exact_rectangle_union_iou"]),
                "eval_set_f1_0_3": float(eval_summary["set_f1_0_3"]),
                "eval_set_f1_0_5": float(eval_summary["set_f1_0_5"]),
                "eval_mean_pred_count": float(eval_summary["mean_pred_count"]),
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_root / "per_seed_metrics.csv", index=False)
    metrics = [
        "eval_coverage_mean_iou",
        "eval_exact_union_iou",
        "eval_set_f1_0_3",
        "eval_set_f1_0_5",
        "eval_mean_pred_count",
    ]
    aggregate = {
        "status": "complete",
        "classification": "transplant diagnostic; not full-pipeline gate retraining",
        "n_upstream_seeds": 3,
        "n_gate_training_upstream_seeds": 1,
        "n_gate_random_seeds": 3,
        "query_contract": "finding category + raw phrase",
        "gate_input_dim": 47,
        "forbidden_gold_features": [],
        "selection_split": "1444 validation only",
        "uses_eval_for_selection": False,
        "selected_variants": frame["selected_variant"].tolist(),
    }
    for metric in metrics:
        aggregate[f"{metric}_mean"] = float(frame[metric].mean())
        aggregate[f"{metric}_std"] = float(frame[metric].std(ddof=1))
    write_json(args.output_root / "FINAL_STATUS.json", aggregate)


if __name__ == "__main__":
    main()

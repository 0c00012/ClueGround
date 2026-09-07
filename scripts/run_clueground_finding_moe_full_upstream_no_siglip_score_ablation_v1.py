#!/usr/bin/env python
"""Remove SigLIP from the historical 1444 Hybrid+SigLIP MoE.

Everything except the SigLIP signal is inherited from the 0.5360 runner:
the hybrid predictions, legacy semantic candidate rows, non-SigLIP score
terms, finding-conditioned gate, and multi-cue bypass.  SigLIP columns are
dropped before semantic candidate ranking so accidental use fails closed.
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

from scripts import run_clueground_finding_moe_full_upstream_siglip_only_3seed_v1 as source  # noqa: E402
from scripts import run_clueground_moe_transplant_diagnostic_3seed_v1 as transplant  # noqa: E402
from scripts import run_final_methodology_verification_v1 as verify  # noqa: E402
from scripts import run_ms_cxr_siglip_candidate_fusion_v1 as semantic  # noqa: E402
from scripts import run_ms_cxr_trainable_moe_gate_v1 as moe  # noqa: E402


SEEDS = (13, 42, 2026)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "experiments"
    / "clueground_finding_moe_full_upstream_no_siglip_score_ablation_v1"
)
ORIGINAL_RESULT = (
    PROJECT_ROOT
    / "experiments"
    / "clueground_finding_moe_full_upstream_siglip_only_3seed_v1"
    / "FINAL_STATUS.json"
)
SIGLIP_PREDICTIONS = (
    PROJECT_ROOT
    / "experiments"
    / "ms_cxr_siglip_candidate_fusion_v1"
    / "predictions"
)
SET_PARAMS = (
    PROJECT_ROOT
    / "experiments"
    / "ms_cxr_siglip_candidate_fusion_v1"
    / "configs"
    / "best_set_params.json"
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.zeros(len(frame), dtype=np.float64)
    return np.nan_to_num(
        pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def semantic_set_without_siglip(
    split: str,
    groups: dict[str, dict[str, Any]],
    output_root: Path,
) -> dict[str, list[dict[str, Any]]]:
    path = SIGLIP_PREDICTIONS / f"{split}_siglip_scored_candidates_cxr_claim_m0p15.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    removed = [column for column in frame.columns if column.lower().startswith("siglip")]
    frame = frame.drop(columns=removed)

    score_head_present = "score_head" in frame.columns
    frame["no_siglip_score"] = (
        numeric(frame, "score_head")
        + 0.05 * numeric(frame, "prior_iou")
        + 0.03 * numeric(frame, "confidence")
    )
    if any(column.lower().startswith("siglip") for column in frame.columns):
        raise RuntimeError("SigLIP feature survived the ablation drop")

    params = json.loads(SET_PARAMS.read_text(encoding="utf-8"))
    candidates = semantic.scored_candidates_by_group(frame, groups, "no_siglip_score")
    predictions = semantic.predict_phrase_sets(groups, candidates, params)

    audit_dir = output_root / "semantic_candidate_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(audit_dir / f"{split}_no_siglip_scored_candidates.csv", index=False)
    write_json(
        audit_dir / f"{split}_summary.json",
        {
            "split": split,
            "source_path": str(path),
            "n_rows": int(len(frame)),
            "n_tasks": int(frame["task_id"].astype(str).nunique()),
            "n_phrase_groups": int(len(groups)),
            "removed_siglip_columns": removed,
            "siglip_feature_count_after_drop": 0,
            "score_head_present": bool(score_head_present),
            "score_formula": "score_head_if_present_else_0 + 0.05*prior_iou + 0.03*confidence",
            "set_params": params,
        },
    )
    return predictions


def build_bundle_without_siglip(
    split: str,
    output_root: Path,
) -> moe.ExpertBundle:
    groups = moe.sem2.gh.load_groups(split)
    hybrid, cue = moe.fine_base.build_hybrid_v4(split, groups)
    no_siglip = semantic_set_without_siglip(split, groups, output_root)
    return moe.ExpertBundle(groups, hybrid, no_siglip, {}, {}, cue)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    original_feature_builder = moe.gate_feature_for_group
    moe.gate_feature_for_group = source.finding_gate_feature
    rows: list[dict[str, Any]] = []
    try:
        for seed in SEEDS:
            context, outputs = source.load_seed_hybrid(seed)
            bundles = {
                split: build_bundle_without_siglip(split, output_root)
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

            # The field name remains "siglip" only because ExpertBundle has a
            # fixed slot; its contents were produced after dropping all SigLIP columns.
            experts = ["hybrid", "siglip"]
            model, params, _, _ = moe.train_gate(
                f"finding_moe_no_siglip_s{seed}",
                bundles["train"],
                bundles["val"],
                experts,
                hardneg=False,
                seed=seed,
                device=args.device,
            )
            val_gate = source.predict_gate(model, bundles["val"], experts)
            eval_gate = source.predict_gate(model, bundles["eval"], experts)

            _, val_base = source.evaluate(
                f"base_s{seed}", bundles["val"], bundles["val"].hybrid, seed_root / "val" / "base"
            )
            _, val_gate_summary = source.evaluate(
                f"no_siglip_gate_s{seed}", bundles["val"], val_gate, seed_root / "val" / "gate"
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
            selected = eval_gate if use_gate else bundles["eval"].hybrid
            selected_name = "gate" if use_gate else "base"
            _, eval_summary = source.evaluate(
                f"selected_{selected_name}_s{seed}",
                bundles["eval"],
                selected,
                seed_root / "eval" / selected_name,
            )

            mutated = copy.deepcopy(bundles["eval"])
            mutated.groups = verify.mutate_gold(mutated.groups)
            mutated_predictions = source.predict_gate(model, mutated, experts)
            gold_independent = source.maps_equal(eval_gate, mutated_predictions)
            rows.append(
                {
                    "seed": seed,
                    "selected_variant": selected_name,
                    "gate_best_epoch": int(params["epoch"]),
                    "train_gate_rows": int(params["train_rows"]),
                    "val_gate_rows": int(params["val_rows"]),
                    "val_base_coverage": float(val_base["coverage_mean_iou"]),
                    "val_gate_coverage": float(val_gate_summary["coverage_mean_iou"]),
                    "coverage_mean_iou": float(eval_summary["coverage_mean_iou"]),
                    "exact_union_iou": float(eval_summary["exact_rectangle_union_iou"]),
                    "set_f1_0_3": float(eval_summary["set_f1_0_3"]),
                    "set_f1_0_5": float(eval_summary["set_f1_0_5"]),
                    "mean_pred_count": float(eval_summary["mean_pred_count"]),
                    "gold_mutation_independence": bool(gold_independent),
                }
            )
    finally:
        moe.gate_feature_for_group = original_feature_builder

    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "per_seed_metrics.csv", index=False)
    metrics = ("coverage_mean_iou", "exact_union_iou", "set_f1_0_3", "set_f1_0_5", "mean_pred_count")
    status: dict[str, Any] = {
        "status": "complete",
        "method": "finding-conditioned YOLO-RAD-DINO + legacy semantic scorer without SigLIP",
        "ablation_of": "1444 Hybrid+SigLIP MoE Coverage 0.5360",
        "siglip_used": False,
        "biomedclip_used": False,
        "hybrid_unchanged": True,
        "gate_architecture_unchanged": True,
        "multi_cue_bypass_unchanged": True,
        "set_params_unchanged": True,
        "semantic_score_formula": "score_head_if_present_else_0 + 0.05*prior_iou + 0.03*confidence",
        "selected_variants": frame["selected_variant"].tolist(),
        "gold_mutation_independence_pass": bool(frame["gold_mutation_independence"].all()),
    }
    for metric in metrics:
        status[f"{metric}_mean"] = float(frame[metric].mean())
        status[f"{metric}_std"] = float(frame[metric].std(ddof=1))

    original = json.loads(ORIGINAL_RESULT.read_text(encoding="utf-8"))
    comparison = {
        "original_hybrid_plus_siglip": {
            metric: float(original[f"{metric}_mean"])
            for metric in metrics
        },
        "no_siglip": {metric: float(status[f"{metric}_mean"]) for metric in metrics},
        "delta_no_siglip_minus_original": {
            metric: float(status[f"{metric}_mean"] - original[f"{metric}_mean"])
            for metric in metrics
        },
    }
    write_json(output_root / "FINAL_STATUS.json", status)
    write_json(output_root / "DELTA_VS_0P5360.json", comparison)


if __name__ == "__main__":
    main()

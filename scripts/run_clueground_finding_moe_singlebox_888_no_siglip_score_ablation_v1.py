#!/usr/bin/env python
"""Run the 1444 no-SigLIP semantic-score method on direct-888.

The YOLO--RAD-DINO hybrid, direct-888 split, finding-conditioned gate, and
multi-cue bypass are inherited from the paired 888 experiment.  The semantic
candidate branch drops every SigLIP-derived column and uses the exact score
formula from the 1444 no-SigLIP ablation.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_clueground_finding_moe_singlebox_888_full_upstream_3seed_v1 as direct  # noqa: E402
from scripts import run_clueground_siglip_only_singlebox_888_paired_3seed_v1 as paired  # noqa: E402
from scripts import run_ms_cxr_siglip_candidate_fusion_v1 as candidate_set  # noqa: E402


SEEDS = (13, 42, 2026)
PAIRED_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "clueground_finding_moe_singlebox_888_yolo640_full_paired_3seed_v1"
)
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "clueground_finding_moe_singlebox_888_no_siglip_score_ablation_v1"
)
SET_PARAMS = {
    "nms_iou": 0.35,
    "score_ratio": 0.0,
    "min_k_if_cue": 1,
    "max_k_if_cue": 2,
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise RuntimeError(f"Required no-SigLIP feature is missing: {column}")
    return np.nan_to_num(
        pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def load_args() -> SimpleNamespace:
    config = json.loads((PAIRED_ROOT / "RUN_CONFIG.json").read_text(encoding="utf-8"))
    config["hybrid_root"] = Path(config["hybrid_root"])
    config["output_root"] = OUTPUT_ROOT
    config["raw_yolo_candidate_root"] = (
        Path(config["raw_yolo_candidate_root"])
        if config.get("raw_yolo_candidate_root")
        else None
    )
    return SimpleNamespace(**config)


def load_no_siglip_expert(
    upstream: direct.SeedUpstream,
    seed: int,
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    source_root = PAIRED_ROOT / "semantic_experts" / f"source_seed_{seed}"
    audit_root = OUTPUT_ROOT / "semantic_candidate_audit" / f"source_seed_{seed}"
    audit_root.mkdir(parents=True, exist_ok=True)
    predictions: dict[str, dict[str, list[dict[str, Any]]]] = {}
    split_audits: list[dict[str, Any]] = []

    for split in ("train", "val", "eval"):
        source_path = source_root / f"{split}_siglip_fusion_scored_candidates.csv"
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        frame = pd.read_csv(source_path)
        removed = [
            column
            for column in frame.columns
            if column.lower().startswith("siglip")
            or column.lower() == "siglip_fusion_score"
        ]
        frame = frame.drop(columns=removed)
        frame["no_siglip_score"] = (
            numeric(frame, "score_head")
            + 0.05 * numeric(frame, "prior_iou")
            + 0.03 * numeric(frame, "confidence")
        )
        surviving = [
            column
            for column in frame.columns
            if column.lower().startswith("siglip_")
            or column.lower() == "siglip_fusion_score"
        ]
        if surviving:
            raise RuntimeError(f"SigLIP columns survived: {surviving}")

        groups = direct.exact.make_groups(upstream.context, split)
        candidates = candidate_set.scored_candidates_by_group(
            frame,
            groups,
            "no_siglip_score",
        )
        predictions[split] = candidate_set.predict_phrase_sets(
            groups,
            candidates,
            SET_PARAMS,
        )
        if set(predictions[split]) != set(groups):
            raise RuntimeError(f"Prediction ID mismatch for seed={seed}/{split}")

        output_path = audit_root / f"{split}_no_siglip_scored_candidates.csv"
        frame.to_csv(output_path, index=False)
        split_audits.append(
            {
                "split": split,
                "source_path": str(source_path.resolve()),
                "output_path": str(output_path.resolve()),
                "n_rows": int(len(frame)),
                "n_groups": int(len(groups)),
                "removed_siglip_columns": removed,
                "surviving_siglip_columns": surviving,
                "score_head_present": "score_head" in frame.columns,
            }
        )

    provenance = {
        "semantic_expert": "legacy candidate scorer without SigLIP",
        "visual_models": "YOLO family + RAD-DINO only",
        "score_formula": "score_head + 0.05*prior_iou + 0.03*confidence",
        "set_params": SET_PARAMS,
        "siglip_used": False,
        "biomedclip_used": False,
        "splits": split_audits,
    }
    write_json(audit_root / "SUMMARY.json", provenance)
    return predictions, provenance


def make_bundle(
    upstream: direct.SeedUpstream,
    no_siglip: dict[str, dict[str, list[dict[str, Any]]]],
    split: str,
) -> direct.moe.ExpertBundle:
    groups = direct.exact.make_groups(upstream.context, split)
    # ExpertBundle has a fixed field named siglip.  This field contains the
    # no-SigLIP predictions built above; no SigLIP score enters the model.
    return direct.moe.ExpertBundle(
        groups=groups,
        hybrid=upstream.hybrid[split],
        siglip=no_siglip[split],
        biomed={},
        pretrain={},
        cue=upstream.cue[split],
    )


@torch.no_grad()
def predict_gate(
    model: direct.moe.MoEGate,
    bundle: direct.moe.ExpertBundle,
) -> dict[str, list[dict[str, Any]]]:
    experts = ["hybrid", "siglip"]
    model.eval().to("cpu")
    multi = direct.moe.has_multi_cue(bundle.cue)
    output: dict[str, list[dict[str, Any]]] = {}
    for group_id, group in bundle.groups.items():
        prediction = bundle.hybrid.get(group_id, [])
        result = direct.finding_gate_feature(bundle, group_id, experts)
        if result is not None and not multi.get(group_id, False):
            values, boxes, _ = result
            weights = torch.softmax(
                model(torch.from_numpy(values[None, :]).float()), dim=-1
            ).detach().cpu().numpy()[0]
            box_norm = (boxes * weights[:, None]).sum(axis=0)
            box_norm[:2] = np.clip(box_norm[:2], 0.0, 1.0)
            box_norm[2:] = np.clip(box_norm[2:], 1e-4, 1.0)
            prediction = [
                {
                    "box": direct.moe.norm_to_xyxy(
                        box_norm,
                        float(group["image_width"]),
                        float(group["image_height"]),
                    ),
                    "score": float(weights.max()),
                    "source": "hybrid_no_siglip_gate",
                }
            ]
        output[group_id] = prediction
    return output


def save_variant(
    root: Path,
    name: str,
    bundle: direct.moe.ExpertBundle,
    predictions: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    summary = direct.save_evaluation(root, name, bundle, predictions)
    multi = direct.moe.has_multi_cue(bundle.cue)
    rows = []
    for group_id in bundle.groups:
        rows.append(
            {
                "group_id": group_id,
                "has_multi_cue": bool(multi.get(group_id, False)),
                "changed_vs_hybrid": not direct.prediction_maps_equal(
                    {group_id: bundle.hybrid.get(group_id, [])},
                    {group_id: predictions.get(group_id, [])},
                ),
            }
        )
    pd.DataFrame(rows).to_csv(root / "action_audit.csv", index=False)
    return summary


def run_seed(seed: int, args: SimpleNamespace) -> dict[str, Any]:
    direct.set_seed(seed)
    upstream = direct.load_hybrid_upstream(seed, args)
    no_siglip, provenance = load_no_siglip_expert(upstream, seed)
    bundles = {
        split: make_bundle(upstream, no_siglip, split)
        for split in ("train", "val", "eval")
    }

    seed_root = OUTPUT_ROOT / f"seed_{seed}"
    direct.moe.CKPT = seed_root / "checkpoints"
    direct.moe.LOG = seed_root / "logs"
    direct.moe.MET = seed_root / "training_metrics"
    direct.moe.PRED = seed_root / "predictions"
    for path in (direct.moe.CKPT, direct.moe.LOG, direct.moe.MET, direct.moe.PRED):
        path.mkdir(parents=True, exist_ok=True)

    experts = ["hybrid", "siglip"]
    model, params, _, _ = direct.moe.train_gate(
        f"finding_moe_direct888_no_siglip_s{seed}",
        bundles["train"],
        bundles["val"],
        experts,
        hardneg=False,
        seed=seed,
        device=args.device,
    )
    val_gate = predict_gate(model, bundles["val"])
    eval_gate = predict_gate(model, bundles["eval"])

    val_base = direct.save_evaluation(
        seed_root / "val" / "base",
        f"base_s{seed}",
        bundles["val"],
        bundles["val"].hybrid,
    )
    val_no_siglip = direct.save_evaluation(
        seed_root / "val" / "no_siglip_gate",
        f"no_siglip_gate_s{seed}",
        bundles["val"],
        val_gate,
    )
    use_gate = (
        float(val_no_siglip["mean_iou"]),
        float(val_no_siglip["Hit@0.5"]),
        float(val_no_siglip["Hit@0.3"]),
    ) >= (
        float(val_base["mean_iou"]),
        float(val_base["Hit@0.5"]),
        float(val_base["Hit@0.3"]),
    )
    selected_name = "no_siglip_gate" if use_gate else "base"
    selected = eval_gate if use_gate else bundles["eval"].hybrid

    eval_base = save_variant(
        seed_root / "eval" / "base",
        f"base_s{seed}",
        bundles["eval"],
        bundles["eval"].hybrid,
    )
    eval_no_siglip = save_variant(
        seed_root / "eval" / "no_siglip_gate",
        f"no_siglip_gate_s{seed}",
        bundles["eval"],
        eval_gate,
    )
    eval_selected = direct.save_evaluation(
        seed_root / "eval" / "selected",
        f"selected_{selected_name}_s{seed}",
        bundles["eval"],
        selected,
    )

    mutated = copy.deepcopy(bundles["eval"])
    for group in mutated.groups.values():
        group["gt_boxes"] = [[0.0, 0.0, 1.0, 1.0]]
    gold_independent = direct.prediction_maps_equal(eval_gate, predict_gate(model, mutated))
    row = {
        "seed": seed,
        "selected_variant": selected_name,
        "selection_split": "direct-888 validation only",
        "gate_best_epoch": int(params["epoch"]),
        "train_gate_rows": int(params["train_rows"]),
        "val_gate_rows": int(params["val_rows"]),
        "val_base_mean_iou": float(val_base["mean_iou"]),
        "val_no_siglip_mean_iou": float(val_no_siglip["mean_iou"]),
        "base_mean_iou": float(eval_base["mean_iou"]),
        "base_hit_0_3": float(eval_base["Hit@0.3"]),
        "base_hit_0_5": float(eval_base["Hit@0.5"]),
        "no_siglip_mean_iou": float(eval_no_siglip["mean_iou"]),
        "no_siglip_hit_0_3": float(eval_no_siglip["Hit@0.3"]),
        "no_siglip_hit_0_5": float(eval_no_siglip["Hit@0.5"]),
        "selected_mean_iou": float(eval_selected["mean_iou"]),
        "selected_hit_0_3": float(eval_selected["Hit@0.3"]),
        "selected_hit_0_5": float(eval_selected["Hit@0.5"]),
        "gold_mutation_independence": bool(gold_independent),
    }
    write_json(seed_root / "RUN_STATUS.json", {**row, "semantic": provenance})
    return row


def aggregate(frame: pd.DataFrame) -> dict[str, Any]:
    status: dict[str, Any] = {
        "status": "complete",
        "method": "ClueGround YOLO-RAD-DINO + no-SigLIP semantic-score gate",
        "protocol": "direct-888 train638/val87/eval163",
        "siglip_used": False,
        "biomedclip_used": False,
        "score_formula": "score_head + 0.05*prior_iou + 0.03*confidence",
        "set_params": SET_PARAMS,
        "selected_variants": frame["selected_variant"].tolist(),
        "gold_mutation_independence_pass": bool(frame["gold_mutation_independence"].all()),
    }
    for prefix in ("base", "no_siglip", "selected"):
        for metric in ("mean_iou", "hit_0_3", "hit_0_5"):
            values = frame[f"{prefix}_{metric}"].astype(float).to_numpy()
            status[f"{prefix}_{metric}_mean"] = float(values.mean())
            status[f"{prefix}_{metric}_std"] = float(values.std(ddof=1))
    status["delta_no_siglip_vs_base_mean_iou"] = (
        status["no_siglip_mean_iou_mean"] - status["base_mean_iou_mean"]
    )
    status["delta_selected_vs_base_mean_iou"] = (
        status["selected_mean_iou_mean"] - status["base_mean_iou_mean"]
    )
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args_cli = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    args = load_args()
    args.device = args_cli.device

    original_feature_builder = direct.moe.gate_feature_for_group
    direct.moe.gate_feature_for_group = direct.finding_gate_feature
    try:
        rows = [run_seed(seed, args) for seed in SEEDS]
    finally:
        direct.moe.gate_feature_for_group = original_feature_builder

    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_ROOT / "per_seed_metrics.csv", index=False)
    split = direct.split_audit(direct.load_hybrid_upstream(seed, args) for seed in SEEDS)
    status = aggregate(frame)
    status["split_overlap_audit"] = split
    if split["status"] != "PASS" or not status["gold_mutation_independence_pass"]:
        status["status"] = "failed_audit"
    write_json(OUTPUT_ROOT / "split_overlap_audit.json", split)
    write_json(OUTPUT_ROOT / "FINAL_STATUS.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

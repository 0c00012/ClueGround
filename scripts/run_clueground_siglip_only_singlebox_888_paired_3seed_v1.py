#!/usr/bin/env python
"""Evaluate direct-888 ClueGround with a seed-paired frozen SigLIP expert.

This reuses the already generated YOLO/RAD-DINO upstream predictions and
SigLIP-scored candidate tables from the controlled paired experiment.  The
only learned downstream component is a two-expert gate over ``hybrid`` and
``siglip``.  BioMedCLIP is never included in the gate or its feature vector.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
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
from scripts import run_ms_cxr_siglip_candidate_fusion_v1 as semantic  # noqa: E402


SEEDS = (13, 42, 2026)
PAIRED_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "clueground_finding_moe_singlebox_888_yolo640_full_paired_3seed_v1"
)
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "clueground_siglip_only_singlebox_888_full_paired_3seed_v1"
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_upstream_args() -> SimpleNamespace:
    config = json.loads((PAIRED_ROOT / "RUN_CONFIG.json").read_text(encoding="utf-8"))
    config["hybrid_root"] = Path(config["hybrid_root"])
    config["output_root"] = OUTPUT_ROOT
    config["raw_yolo_candidate_root"] = (
        Path(config["raw_yolo_candidate_root"])
        if config.get("raw_yolo_candidate_root")
        else None
    )
    return SimpleNamespace(**config)


def load_siglip_expert(
    upstream: direct.SeedUpstream,
    seed: int,
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    root = PAIRED_ROOT / "semantic_experts" / f"source_seed_{seed}"
    status_path = root / "SEMANTIC_EXPERT_STATUS.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    params = status["siglip"]
    score_column = str(params["score_column"])
    set_params = dict(params["set_params"])
    predictions: dict[str, dict[str, list[dict[str, Any]]]] = {}
    cache_rows = []
    for split in ("train", "val", "eval"):
        path = root / f"{split}_siglip_fusion_scored_candidates.csv"
        frame = pd.read_csv(path)
        groups = direct.exact.make_groups(upstream.context, split)
        missing = set(groups).difference(frame["task_id"].astype(str).unique())
        if missing:
            raise RuntimeError(
                f"SigLIP cache seed={seed} split={split} misses {len(missing)} groups"
            )
        candidates = semantic.scored_candidates_by_group(frame, groups, score_column)
        predictions[split] = semantic.predict_phrase_sets(groups, candidates, set_params)
        if set(predictions[split]) != set(groups):
            raise RuntimeError(f"SigLIP prediction IDs differ for seed={seed}/{split}")
        cache_rows.append(
            {
                "seed": seed,
                "split": split,
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "n_candidate_rows": int(len(frame)),
                "n_groups": int(len(groups)),
            }
        )
    provenance = {
        "semantic_expert": "SigLIP only",
        "semantic_source_seed": seed,
        "model_id": status["siglip_model_id"],
        "prompt_mode": status["prompt_mode"],
        "crop_margin": status["crop_margin"],
        "score_column": score_column,
        "set_params": set_params,
        "cache_rows": cache_rows,
    }
    return predictions, provenance


def make_bundle(
    upstream: direct.SeedUpstream,
    siglip: dict[str, dict[str, list[dict[str, Any]]]],
    split: str,
) -> direct.moe.ExpertBundle:
    groups = direct.exact.make_groups(upstream.context, split)
    return direct.moe.ExpertBundle(
        groups=groups,
        hybrid=upstream.hybrid[split],
        siglip=siglip[split],
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
                    "source": "hybrid_siglip_gate",
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
    rows = []
    multi = direct.moe.has_multi_cue(bundle.cue)
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
    siglip, semantic_provenance = load_siglip_expert(upstream, seed)
    bundles = {
        split: make_bundle(upstream, siglip, split)
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
        f"finding_moe_direct888_siglip_only_s{seed}",
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
        seed_root / "val" / "base", f"base_s{seed}", bundles["val"], bundles["val"].hybrid
    )
    val_siglip = direct.save_evaluation(
        seed_root / "val" / "siglip_gate", f"siglip_gate_s{seed}", bundles["val"], val_gate
    )
    use_gate = (
        float(val_siglip["mean_iou"]),
        float(val_siglip["Hit@0.5"]),
        float(val_siglip["Hit@0.3"]),
    ) >= (
        float(val_base["mean_iou"]),
        float(val_base["Hit@0.5"]),
        float(val_base["Hit@0.3"]),
    )
    selected_name = "siglip_gate" if use_gate else "base"
    selected = eval_gate if use_gate else bundles["eval"].hybrid

    eval_base = save_variant(
        seed_root / "eval" / "base", f"base_s{seed}", bundles["eval"], bundles["eval"].hybrid
    )
    eval_siglip = save_variant(
        seed_root / "eval" / "siglip_gate", f"siglip_gate_s{seed}", bundles["eval"], eval_gate
    )
    eval_selected = direct.save_evaluation(
        seed_root / "eval" / "selected", f"selected_{selected_name}_s{seed}", bundles["eval"], selected
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
        "val_siglip_mean_iou": float(val_siglip["mean_iou"]),
        "base_mean_iou": float(eval_base["mean_iou"]),
        "base_hit_0_3": float(eval_base["Hit@0.3"]),
        "base_hit_0_5": float(eval_base["Hit@0.5"]),
        "siglip_mean_iou": float(eval_siglip["mean_iou"]),
        "siglip_hit_0_3": float(eval_siglip["Hit@0.3"]),
        "siglip_hit_0_5": float(eval_siglip["Hit@0.5"]),
        "selected_mean_iou": float(eval_selected["mean_iou"]),
        "selected_hit_0_3": float(eval_selected["Hit@0.3"]),
        "selected_hit_0_5": float(eval_selected["Hit@0.5"]),
        "gold_mutation_independence": bool(gold_independent),
    }
    write_json(seed_root / "RUN_STATUS.json", {**row, "semantic": semantic_provenance})
    return row


def aggregate(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "complete",
        "method": "ClueGround hybrid-v4 + frozen SigLIP-only gate",
        "protocol": "direct-888 train638/val87/eval163",
        "seeds": list(SEEDS),
        "seed_pairing": "13/13, 42/42, 2026/2026",
        "semantic_experts": ["siglip"],
        "biomedclip_used": False,
        "selection_split": "direct-888 validation only",
        "selected_variants": frame["selected_variant"].tolist(),
        "gold_mutation_independence_pass": bool(frame["gold_mutation_independence"].all()),
    }
    for prefix in ("base", "siglip", "selected"):
        for metric in ("mean_iou", "hit_0_3", "hit_0_5"):
            values = frame[f"{prefix}_{metric}"].astype(float).to_numpy()
            result[f"{prefix}_{metric}_mean"] = float(values.mean())
            result[f"{prefix}_{metric}_std"] = float(values.std(ddof=1))
    result["delta_siglip_vs_base_mean_iou"] = (
        result["siglip_mean_iou_mean"] - result["base_mean_iou_mean"]
    )
    result["delta_selected_vs_base_mean_iou"] = (
        result["selected_mean_iou_mean"] - result["base_mean_iou_mean"]
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parsed = parser.parse_args()
    if tuple(parsed.seeds) != SEEDS:
        raise ValueError(f"This controlled run requires seeds {SEEDS}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    args = load_upstream_args()
    write_json(
        OUTPUT_ROOT / "RUN_CONFIG.json",
        {
            "protocol": "direct-888 train638/val87/eval163",
            "seeds": list(SEEDS),
            "experts": ["hybrid", "siglip"],
            "biomedclip_used": False,
            "semantic_cache_mode": "paired",
            "semantic_cache_root": str((PAIRED_ROOT / "semantic_experts").resolve()),
            "hybrid_root": str(args.hybrid_root.resolve()),
            "selection_split": "validation only",
        },
    )

    original_feature_builder = direct.moe.gate_feature_for_group
    direct.moe.gate_feature_for_group = direct.finding_gate_feature
    try:
        rows = [run_seed(seed, args) for seed in SEEDS]
    finally:
        direct.moe.gate_feature_for_group = original_feature_builder

    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_ROOT / "per_seed_metrics.csv", index=False)
    split = direct.split_audit(direct.load_hybrid_upstream(seed, args) for seed in SEEDS)
    final = aggregate(frame)
    final["split_overlap_audit"] = split
    final["status"] = (
        "PASS"
        if split["status"] == "PASS" and final["gold_mutation_independence_pass"]
        else "FAIL"
    )
    write_json(OUTPUT_ROOT / "split_overlap_audit.json", split)
    write_json(OUTPUT_ROOT / "FINAL_STATUS.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Measure hidden dependence of the local unified method on eval finding labels."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_ms_cxr_trainable_moe_gate_v1 as moe  # noqa: E402
from src.baseline_repro.evaluator import iou_matrix  # noqa: E402


DEFAULT_BASELINE_ROOT = ROOT / "experiments" / "baseline_faithful_reproduction" / "20260712_v2"
DEFAULT_METHOD_ROOT = ROOT / "experiments" / "final_methodology_verification" / "20260712_v1"
FINDINGS = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Lung Opacity",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
)
OFFICIAL_CATEGORY_ORDER = (
    "Cardiomegaly",
    "Lung Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
)


def _boxes(predictions: dict[str, list[dict[str, Any]]]) -> dict[str, list[list[float]]]:
    return {
        str(group_id): [
            [float(value) for value in prediction["box"]] for prediction in group_predictions
        ]
        for group_id, group_predictions in predictions.items()
    }


def _first_iou(predictions: list[list[float]], gold: list[list[float]]) -> float:
    if not predictions or not gold:
        return 0.0
    return float(iou_matrix([predictions[0]], [gold[0]])[0, 0])


def _legacy_gate_feature(
    bundle: moe.ExpertBundle,
    group_id: str,
    experts: list[str],
    finding_order: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray] | None:
    group = bundle.groups[group_id]
    expert_rows = moe.expert_list(bundle, experts, group_id)
    if not all(len(rows) == 1 for rows in expert_rows):
        return None
    width = float(group["image_width"])
    height = float(group["image_height"])
    boxes = np.stack(
        [moe.xyxy_to_norm(rows[0]["box"], width, height) for rows in expert_rows]
    ).astype(np.float32)
    scores = np.asarray(
        [float(rows[0].get("score", 0.0)) for rows in expert_rows], dtype=np.float32
    )
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    scores_scaled = np.asarray([math.tanh(float(score)) for score in scores], dtype=np.float32)
    pairwise = []
    for left in range(len(experts)):
        for right in range(left + 1, len(experts)):
            pairwise.append(
                moe.mb.iou_xyxy(expert_rows[left][0]["box"], expert_rows[right][0]["box"])
            )
    laterality, vertical, multi_text = moe.parse_context(group)
    context = moe.one_hot(str(group["finding"]), list(finding_order))
    context += moe.one_hot(laterality, moe.LATS)
    context += moe.one_hot(vertical, moe.VERTS)
    context += [float(multi_text)]
    stats = [
        float(np.mean(boxes[:, 0])),
        float(np.std(boxes[:, 0])),
        float(np.mean(boxes[:, 1])),
        float(np.std(boxes[:, 1])),
        float(np.mean(boxes[:, 2] * boxes[:, 3])),
        float(np.std(boxes[:, 2] * boxes[:, 3])),
        float(max(pairwise) if pairwise else 0.0),
        float(np.mean(pairwise) if pairwise else 0.0),
    ]
    feature = np.asarray(
        list(boxes.reshape(-1)) + list(scores_scaled) + pairwise + context + stats,
        dtype=np.float32,
    )
    return feature, boxes


@torch.no_grad()
def _legacy_predict(
    model: moe.MoEGate,
    bundle: moe.ExpertBundle,
    finding_order: tuple[str, ...],
) -> dict[str, list[dict[str, Any]]]:
    experts = ["hybrid", "siglip", "biomed"]
    model.eval().to("cpu")
    multi = moe.has_multi_cue(bundle.cue)
    output: dict[str, list[dict[str, Any]]] = {}
    for group_id, group in bundle.groups.items():
        prediction = bundle.hybrid.get(group_id, [])
        feature = _legacy_gate_feature(bundle, group_id, experts, finding_order)
        if feature is not None and not multi.get(group_id, False):
            values, boxes = feature
            weights = torch.softmax(model(torch.from_numpy(values[None, :]).float()), dim=-1)
            weights_array = weights.detach().cpu().numpy()[0]
            box_norm = (boxes * weights_array[:, None]).sum(axis=0)
            box_norm[:2] = np.clip(box_norm[:2], 0.0, 1.0)
            box_norm[2:] = np.clip(box_norm[2:], 1e-4, 1.0)
            box = moe.norm_to_xyxy(
                box_norm, float(group["image_width"]), float(group["image_height"])
            )
            prediction = [
                {"box": box, "score": float(weights_array.max()), "source": "legacy_gate"}
            ]
        output[group_id] = prediction
    return output


def _saved_prediction_map(path: Path) -> dict[str, list[list[float]]]:
    frame = pd.read_csv(path)
    return {
        str(row.group_id): [
            [float(value) for value in box] for box in json.loads(str(row.pred_boxes_json))
        ]
        for row in frame.itertuples(index=False)
    }


def _prediction_difference(
    left: dict[str, list[list[float]]], right: dict[str, list[list[float]]]
) -> tuple[int, float]:
    changed = 0
    maximum = 0.0
    for group_id in sorted(set(left) | set(right)):
        a = left.get(group_id, [])
        b = right.get(group_id, [])
        if len(a) != len(b):
            changed += 1
            continue
        local = max(
            (
                float(np.max(np.abs(np.asarray(box_a) - np.asarray(box_b))))
                for box_a, box_b in zip(a, b)
            ),
            default=0.0,
        )
        maximum = max(maximum, local)
        changed += int(local > 1e-8)
    return changed, maximum


def _summary(
    groups: dict[str, dict[str, Any]],
    original: dict[str, list[list[float]]],
    mutated: dict[str, list[list[float]]],
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = []
    for group_id, group in groups.items():
        left = original.get(str(group_id), [])
        right = mutated.get(str(group_id), [])
        same_count = len(left) == len(right)
        max_difference = 0.0
        if same_count and left:
            max_difference = max(
                float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))
                for a, b in zip(left, right)
            )
        changed = not same_count or max_difference > 1e-8
        rows.append(
            {
                "group_id": str(group_id),
                "phrase": str(group.get("claim_sentence", "")),
                "original_finding": str(group.get("finding", "")),
                "mutated_finding": FINDINGS[(FINDINGS.index(str(group["finding"])) + 1) % len(FINDINGS)],
                "n_pred_original": len(left),
                "n_pred_mutated": len(right),
                "max_coordinate_abs_difference": max_difference,
                "prediction_changed": changed,
                "original_top1_iou": _first_iou(left, group.get("gt_boxes", [])),
                "mutated_top1_iou": _first_iou(right, group.get("gt_boxes", [])),
            }
        )
    frame = pd.DataFrame(rows)
    summary = {
        "status": "FAIL_PHRASE_ONLY_CONTRACT" if bool(frame["prediction_changed"].any()) else "PASS",
        "n_groups": int(len(frame)),
        "n_changed_groups": int(frame["prediction_changed"].sum()),
        "changed_fraction": float(frame["prediction_changed"].mean()),
        "max_coordinate_abs_difference": float(frame["max_coordinate_abs_difference"].max()),
        "original_top1_mean_iou": float(frame["original_top1_iou"].mean()),
        "mutated_top1_mean_iou": float(frame["mutated_top1_iou"].mean()),
        "delta_mutated_minus_original": float(
            frame["mutated_top1_iou"].mean() - frame["original_top1_iou"].mean()
        ),
        "interpretation": (
            "The raw phrase is unchanged. Only the dataset finding/category annotation is rotated. "
            "Any prediction change proves that the artifact is annotation-category-assisted and cannot "
            "enter a phrase-only ranking."
        ),
    }
    return summary, frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--method-root", type=Path, default=DEFAULT_METHOD_ROOT)
    args = parser.parse_args()

    output = args.baseline_root.resolve() / "fidelity" / "ours_local_unified_finding_dependency"
    output.mkdir(parents=True, exist_ok=True)
    method_root = args.method_root.resolve()
    checkpoint = (
        method_root
        / "moe_clean"
        / "checkpoints"
        / "hybrid_siglip_biomedclip_s2026.pt"
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = moe.MoEGate(int(payload["in_dim"]), int(payload["n_experts"]))
    model.load_state_dict(payload["state"], strict=True)

    original_bundle = moe.build_bundle(
        "eval",
        "cpu",
        "disabled_for_mscxr_only_verification",
        include_pretrain_expert=False,
    )
    saved_path = (
        method_root
        / "moe_clean"
        / "predictions"
        / "hybrid_siglip_biomedclip_s2026.csv"
    )
    saved = _saved_prediction_map(saved_path)
    observed_order = tuple(
        dict.fromkeys(str(group["finding"]) for group in original_bundle.groups.values())
    )
    candidate_orders = [FINDINGS, OFFICIAL_CATEGORY_ORDER, observed_order]
    reconstructions = []
    for finding_order in candidate_orders:
        candidate = _legacy_predict(model, original_bundle, finding_order)
        changed, maximum = _prediction_difference(saved, _boxes(candidate))
        reconstructions.append((changed, maximum, finding_order, candidate))
    changed, maximum, finding_order, original_predictions = min(
        reconstructions, key=lambda item: (item[0], item[1])
    )
    count_mismatches = sum(
        len(saved.get(group_id, [])) != len(_boxes(original_predictions).get(group_id, []))
        for group_id in set(saved) | set(_boxes(original_predictions))
    )
    if count_mismatches or maximum > 1e-3:
        raise RuntimeError(
            "Could not reconstruct the 47-feature checkpoint against its saved predictions: "
            f"changed={changed}, count_mismatches={count_mismatches}, max_diff={maximum}"
        )

    mutated_groups = copy.deepcopy(original_bundle.groups)
    for group in mutated_groups.values():
        finding = str(group["finding"])
        group["finding"] = FINDINGS[(FINDINGS.index(finding) + 1) % len(FINDINGS)]
    mutated_hybrid, mutated_cue = moe.fine_base.build_hybrid_v4("eval", mutated_groups)
    mutated_bundle = moe.ExpertBundle(
        groups=mutated_groups,
        hybrid=mutated_hybrid,
        siglip=moe.semantic_set_from_scored("eval", mutated_groups, "siglip"),
        biomed=moe.semantic_set_from_scored("eval", mutated_groups, "biomed"),
        pretrain={},
        cue=mutated_cue,
    )
    mutated_predictions = _legacy_predict(model, mutated_bundle, finding_order)

    summary, details = _summary(
        original_bundle.groups,
        _boxes(original_predictions),
        _boxes(mutated_predictions),
    )
    summary["checkpoint_input_dim"] = int(payload["in_dim"])
    summary["current_phrase_only_input_dim"] = 39
    summary["reconstructed_finding_order"] = list(finding_order)
    summary["saved_prediction_reconstruction_status"] = "PASS"
    summary["saved_prediction_reconstruction_max_abs_difference"] = maximum
    details.to_csv(output / "per_group_finding_mutation.csv", index=False)
    (output / "finding_annotation_dependency.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

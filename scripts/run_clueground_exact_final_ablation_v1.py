#!/usr/bin/env python
"""Exact component and RAD-DINO query ablations for the final ClueGround rows.

The positive controls are immutable stored predictions:
  * direct-888: 0.5486 mean IoU
  * legacy-1444: 0.5373 coverage IoU

For query ablations, the trained 91-D RAD-DINO head is kept fixed and one
input block is masked at inference.  YOLO candidates, validation-selected
calibration, the no-SigLIP semantic scorer/gate, coordinate blending, and the
rule-context decoder are inherited from the final method.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import clueground_siglip_analysis_common_v1 as common  # noqa: E402
from scripts import run_clueground_exact_hybrid_v4_direct_3seed_v1 as exact  # noqa: E402
from scripts import run_clueground_finding_moe_full_upstream_no_siglip_score_ablation_v1 as final1444  # noqa: E402
from scripts import run_clueground_finding_moe_full_upstream_siglip_only_3seed_v1 as source1444  # noqa: E402
from scripts import run_clueground_finding_moe_singlebox_888_full_upstream_3seed_v1 as direct888  # noqa: E402
from scripts import run_clueground_finding_moe_singlebox_888_no_siglip_score_ablation_v1 as final888  # noqa: E402
from scripts import run_clueground_moe_transplant_diagnostic_3seed_v1 as transplant  # noqa: E402
from scripts import run_clueground_no_siglip_component_ablation_v2 as old_component  # noqa: E402
from scripts import run_clueground_vfm_legacy_fusion_unified_3seed_v1 as multi_source  # noqa: E402
from scripts import run_ms_cxr_rad_dino_singlebox_retrain_v1 as rad_single  # noqa: E402
from scripts.models_ms_cxr_vfm_localizer import PatchHeatmapBBoxHead  # noqa: E402


SEEDS = (13, 42, 2026)
COMPONENT_ARMS = ("full", "yolo_only", "rad_dino_only")
QUERY_VARIANTS = (
    "full_query",
    "finding_only",
    "without_f",
    "without_l",
    "without_v",
    "without_z",
    "without_u",
    "without_m_lexical",
    "without_vq",
)
DEFAULT_OUTPUT = ROOT / "experiments" / "clueground_exact_final_ablation_v1"

# Exact 91-D query used by the final checkpoints.  There is no independent m block.
QUERY_SLICES = {
    "f": (0, 8),
    "l": (8, 12),
    "v": (12, 19),
    "z": (19, 25),
    "u": (25, 27),
    "vq": (27, 91),
}
MULTI_TERMS = re.compile(
    r"\b(bilateral|both|bibasal|bibasilar|multifocal|multilobar|multiple|scattered)\b",
    flags=re.IGNORECASE,
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phrase_from_row(row: Any) -> str:
    if isinstance(row, dict):
        return str(row.get("phrase_text", row.get("claim_sentence", "")))
    if hasattr(row, "get"):
        return str(row.get("phrase_text", row.get("claim_sentence", "")))
    return ""


def mask_query(query: np.ndarray, rows: list[dict[str, Any]] | pd.DataFrame, variant: str) -> np.ndarray:
    output = np.asarray(query, dtype=np.float32).copy()
    if output.ndim != 2 or output.shape[1] != 91:
        raise RuntimeError(f"Expected exact 91-D query, got {output.shape}")
    if variant == "full_query":
        return output
    if variant == "finding_only":
        output[:, QUERY_SLICES["l"][0] :] = 0.0
        return output
    if variant == "without_m_lexical":
        iterable = [row for _, row in rows.iterrows()] if isinstance(rows, pd.DataFrame) else rows
        for index, row in enumerate(iterable):
            cleaned = MULTI_TERMS.sub(" ", phrase_from_row(row))
            output[index, QUERY_SLICES["vq"][0] : QUERY_SLICES["vq"][1]] = (
                rad_single.token_hash_features(cleaned, 64)
            )
        return output
    if not variant.startswith("without_"):
        raise ValueError(variant)
    component = variant.removeprefix("without_")
    start, end = QUERY_SLICES[component]
    output[:, start:end] = 0.0
    return output


def load_rad_model(checkpoint: Path, device: torch.device) -> PatchHeatmapBBoxHead:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["state"]
    token_dim = int(state["token_proj.weight"].shape[1])
    query_dim = int(state["context_proj.weight"].shape[1])
    hidden = int(state["token_proj.weight"].shape[0])
    if query_dim != 91:
        raise RuntimeError(f"Final checkpoint is not 91-D: {checkpoint} -> {query_dim}")
    model = PatchHeatmapBBoxHead(token_dim, query_dim, hidden=hidden, dropout=0.1).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_rad(
    model: PatchHeatmapBBoxHead,
    tokens: np.ndarray,
    queries: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    parts = []
    for start in range(0, len(tokens), batch_size):
        box, _ = model(
            torch.from_numpy(tokens[start : start + batch_size]).float().to(device),
            torch.from_numpy(queries[start : start + batch_size]).float().to(device),
        )
        parts.append(box.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(parts, axis=0)


def single_dino_variant(
    seed: int,
    variant: str,
    device: torch.device,
    batch_size: int,
) -> dict[str, dict[str, np.ndarray]]:
    from scripts import run_ms_cxr_singlebox_full_pipeline_3seed_v2 as single_source  # noqa: E402

    single_source.configure_rad_paths(seed)
    checkpoint = (
        exact.SINGLE_SOURCE_ROOT
        / f"seed_{seed}"
        / "training"
        / "rad_dino"
        / "checkpoints"
        / "rad_dino_full_phrase_singlebox.pt"
    )
    model = load_rad_model(checkpoint, device)
    output: dict[str, dict[str, np.ndarray]] = {}
    for split in ("val", "eval"):
        frame, tokens, query, _targets = rad_single.align_split("full_phrase", split)
        masked = mask_query(query, frame, variant)
        boxes = predict_rad(model, tokens, masked, device, batch_size)
        output[split] = {
            str(task_id): box
            for task_id, box in zip(frame["sample_id"].astype(str).tolist(), boxes)
        }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def multi_dino_variant(
    seed: int,
    variant: str,
    device: torch.device,
    batch_size: int,
) -> dict[str, dict[str, np.ndarray]]:
    _inputs, _labels, source_ids = multi_source.load_protocol(exact.PROTOCOL_ROOT)
    root = exact.MULTI_SOURCE_ROOT / f"seed_{seed}" / "mscxr_multibox_1444"
    checkpoint = root / "rad_dino_legacy" / "best.pt"
    model = load_rad_model(checkpoint, device)
    output: dict[str, dict[str, np.ndarray]] = {}
    for split in ("val", "eval"):
        allowed = {task_id for ids in source_ids[split].values() for task_id in ids}
        rows, tokens, query, _targets = multi_source.load_rad_bundle(split, allowed)
        masked = mask_query(query, rows, variant)
        boxes = predict_rad(model, tokens, masked, device, batch_size)
        task_map = {
            str(row["task_id"]): box
            for row, box in zip(rows, boxes)
        }
        output[split] = multi_source.group_dino_map(task_map, source_ids[split])
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def single_hybrid_with_dino(
    seed: int,
    dino: dict[str, dict[str, np.ndarray]],
    args888: SimpleNamespace,
) -> tuple[exact.ProtocolContext, dict[str, list[list[float]]]]:
    context = exact.load_single_context(seed)
    restored_multi_yolo = copy.deepcopy(context.yolo_params)
    run_root = args888.hybrid_root / "singlebox_888" / f"seed_{seed}"
    calibration = run_root / "single_route_fullval_calibration"
    context.yolo_params = json.loads((calibration / "yolo_params.json").read_text(encoding="utf-8"))
    context.fusion_params = json.loads((calibration / "fusion_params.json").read_text(encoding="utf-8"))
    set_params = json.loads((run_root / "selected_set_params.json").read_text(encoding="utf-8"))
    context.dino.update(dino)
    outputs, _ = exact.run_hybrid(context, "eval", set_params, restored_multi_yolo)
    return context, outputs


def load_gate(seed: int) -> torch.nn.Module:
    checkpoint = (
        final1444.DEFAULT_OUTPUT
        / f"seed_{seed}"
        / "checkpoints"
        / f"finding_moe_no_siglip_s{seed}.pt"
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = final1444.moe.MoEGate(int(payload["in_dim"]), int(payload["n_experts"]))
    model.load_state_dict(payload["state"])
    model.eval()
    return model


def multi_context_and_hybrid(
    seed: int,
    dino: dict[str, dict[str, np.ndarray]] | None,
    *,
    disable_dino: bool = False,
) -> tuple[exact.ProtocolContext, dict[str, dict[str, list[list[float]]]]]:
    context = exact.load_multi_context(seed)
    source1444.add_train_artifacts(context)
    restored_multi_yolo = copy.deepcopy(context.yolo_params)
    run_root = transplant.BASE_ROOT / "multibox_1444" / f"seed_{seed}"
    calibration = run_root / "single_route_fullval_calibration"
    context.yolo_params = json.loads((calibration / "yolo_params.json").read_text(encoding="utf-8"))
    context.fusion_params = json.loads((calibration / "fusion_params.json").read_text(encoding="utf-8"))
    set_params = json.loads((run_root / "selected_set_params.json").read_text(encoding="utf-8"))
    if disable_dino:
        context.dino = {split: {} for split in ("train", "val", "eval")}
    elif dino is not None:
        context.dino.update(dino)
    outputs: dict[str, dict[str, list[list[float]]]] = {}
    for split in ("val", "eval"):
        outputs[split], _ = exact.run_hybrid(
            context,
            split,
            set_params,
            restored_multi_yolo,
        )
    return context, outputs


def final_1444_from_hybrid(
    seed: int,
    context: exact.ProtocolContext,
    hybrid_outputs: dict[str, dict[str, list[list[float]]]],
    output_root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], final1444.moe.ExpertBundle]:
    bundle = final1444.build_bundle_without_siglip("eval", output_root)
    bundle.hybrid = transplant.remap_hybrid(context, "eval", hybrid_outputs["eval"], bundle)
    model = load_gate(seed)
    old_builder = final1444.moe.gate_feature_for_group
    final1444.moe.gate_feature_for_group = source1444.finding_gate_feature
    try:
        prediction = source1444.predict_gate(model, bundle, ["hybrid", "siglip"])
    finally:
        final1444.moe.gate_feature_for_group = old_builder
    return prediction, bundle


def dino_only_outputs(
    context: exact.ProtocolContext,
    split: str,
) -> dict[str, list[list[float]]]:
    outputs: dict[str, list[list[float]]] = {}
    for row in context.rows[split]:
        group_id = str(row["group_id"])
        dino = context.dino[split].get(group_id)
        if dino is None:
            outputs[group_id] = []
            continue
        outputs[group_id] = [
            direct888.norm_cxcywh_to_xyxy(
                np.asarray(dino, dtype=np.float32),
                float(row["image_width"]),
                float(row["image_height"]),
            )
        ]
    return outputs


def evaluate_888(outputs: dict[str, list[list[float]]], seed: int) -> dict[str, Any]:
    resources = common.load_resources("888", seed)
    summary, _detail = common.evaluate(resources, outputs, "eval")
    return summary


def evaluate_1444(
    outputs: dict[str, list[dict[str, Any]]],
    bundle: final1444.moe.ExpertBundle,
    root: Path,
    name: str,
) -> dict[str, Any]:
    _detail, summary = source1444.evaluate(name, bundle, outputs, root)
    return summary


def component_runs(
    seed: int,
    args888: SimpleNamespace,
    output_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # Immutable positive controls.
    stored888, path888 = old_component.load_full_outputs("888", seed)
    metric888 = evaluate_888(stored888, seed)
    rows.append({"table": "component", "protocol": "888", "seed": seed, "variant": "full", **metric888})

    context888 = exact.load_single_context(seed)
    yolo_context, yolo888 = single_hybrid_with_dino(seed, {"val": {}, "eval": {}}, args888)
    rows.append({"table": "component", "protocol": "888", "seed": seed, "variant": "yolo_only", **evaluate_888(yolo888, seed)})
    dino888 = dino_only_outputs(context888, "eval")
    rows.append({"table": "component", "protocol": "888", "seed": seed, "variant": "rad_dino_only", **evaluate_888(dino888, seed)})

    stored1444, path1444 = old_component.load_full_outputs("1444", seed)
    context_full, hybrid_full = source1444.load_seed_hybrid(seed)
    bundle_full = final1444.build_bundle_without_siglip("eval", output_root / "fixed_semantic")
    remapped_stored = old_component.remap_legacy_1444_ids(
        stored1444,
        {gid: {**group, "claim_sentence": group["claim_sentence"]} for gid, group in bundle_full.groups.items()},
    ) if set(stored1444) != set(bundle_full.groups) else stored1444
    full_pred = {
        gid: [{"box": box, "score": 1.0, "source": "stored_final"} for box in boxes]
        for gid, boxes in remapped_stored.items()
    }
    full_metric = evaluate_1444(full_pred, bundle_full, output_root / "component" / "1444" / f"seed_{seed}" / "full", "full")
    rows.append({"table": "component", "protocol": "1444", "seed": seed, "variant": "full", **full_metric})

    yolo_context1444, yolo_hybrid = multi_context_and_hybrid(seed, None, disable_dino=True)
    yolo_bundle = final1444.build_bundle_without_siglip("eval", output_root / "fixed_semantic")
    yolo_remapped = transplant.remap_hybrid(yolo_context1444, "eval", yolo_hybrid["eval"], yolo_bundle)
    yolo_metric = evaluate_1444(yolo_remapped, yolo_bundle, output_root / "component" / "1444" / f"seed_{seed}" / "yolo_only", "yolo_only")
    rows.append({"table": "component", "protocol": "1444", "seed": seed, "variant": "yolo_only", **yolo_metric})

    dino_raw = dino_only_outputs(context_full, "eval")
    dino_bundle = final1444.build_bundle_without_siglip("eval", output_root / "fixed_semantic")
    dino_remapped = transplant.remap_hybrid(context_full, "eval", dino_raw, dino_bundle)
    dino_metric = evaluate_1444(dino_remapped, dino_bundle, output_root / "component" / "1444" / f"seed_{seed}" / "rad_dino_only", "rad_dino_only")
    rows.append({"table": "component", "protocol": "1444", "seed": seed, "variant": "rad_dino_only", **dino_metric})

    write_json(
        output_root / "provenance" / f"component_seed_{seed}.json",
        {"stored_888": str(path888), "stored_1444": str(path1444)},
    )
    return rows


def query_runs(
    seed: int,
    args888: SimpleNamespace,
    output_root: Path,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stored888, _ = old_component.load_full_outputs("888", seed)
    stored1444, _ = old_component.load_full_outputs("1444", seed)

    for variant in QUERY_VARIANTS:
        if variant == "full_query":
            outputs888 = stored888
        else:
            dino888 = single_dino_variant(seed, variant, device, batch_size)
            _context888, outputs888 = single_hybrid_with_dino(seed, dino888, args888)
        metric888 = evaluate_888(outputs888, seed)
        rows.append({"table": "query", "protocol": "888", "seed": seed, "variant": variant, **metric888})

        if variant == "full_query":
            bundle = final1444.build_bundle_without_siglip("eval", output_root / "fixed_semantic")
            remapped = old_component.remap_legacy_1444_ids(
                stored1444,
                {gid: {**group, "claim_sentence": group["claim_sentence"]} for gid, group in bundle.groups.items()},
            ) if set(stored1444) != set(bundle.groups) else stored1444
            outputs1444 = {
                gid: [{"box": box, "score": 1.0, "source": "stored_final"} for box in boxes]
                for gid, boxes in remapped.items()
            }
        else:
            dino1444 = multi_dino_variant(seed, variant, device, batch_size)
            context1444, hybrid1444 = multi_context_and_hybrid(seed, dino1444)
            outputs1444, bundle = final_1444_from_hybrid(
                seed,
                context1444,
                hybrid1444,
                output_root / "fixed_semantic",
            )
        metric1444 = evaluate_1444(
            outputs1444,
            bundle,
            output_root / "query" / "1444" / f"seed_{seed}" / variant,
            variant,
        )
        rows.append({"table": "query", "protocol": "1444", "seed": seed, "variant": variant, **metric1444})
    return rows


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "mean_iou", "hit_0_3", "hit_0_5", "coverage_mean_iou",
        "exact_rectangle_union_iou", "set_f1_0_3", "set_f1_0_5", "mean_pred_count",
    ]
    rows = []
    for keys, part in frame.groupby(["table", "protocol", "variant"], sort=False):
        row = {"table": keys[0], "protocol": keys[1], "variant": keys[2], "n_seeds": len(part)}
        for metric in metrics:
            if metric not in part or part[metric].dropna().empty:
                continue
            values = pd.to_numeric(part[metric], errors="coerce").dropna().to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--skip-components", action="store_true")
    parser.add_argument("--skip-query", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    args888 = final888.load_args()
    args888.device = args.device

    write_json(
        output_root / "METHOD_SPEC.json",
        {
            "status": "frozen_before_run",
            "seeds": list(SEEDS),
            "positive_controls": {"888_mean_iou": 0.5486315252772557, "1444_coverage": 0.5372594933248681},
            "query_dim": 91,
            "query_slices": QUERY_SLICES,
            "independent_m_block": False,
            "without_m_definition": "remove multiplicity lexemes from the existing 64-D raw-phrase hash only",
            "query_intervention": "inference-time masking; checkpoint is fixed",
            "fixed_downstream": ["YOLO candidates", "calibration", "no-SigLIP scorer", "MoE gate", "coordinate blend", "rule-context decoder"],
        },
    )

    rows: list[dict[str, Any]] = []
    old_builder = final1444.moe.gate_feature_for_group
    final1444.moe.gate_feature_for_group = source1444.finding_gate_feature
    try:
        for seed in SEEDS:
            if not args.skip_components:
                rows.extend(component_runs(seed, args888, output_root))
            if not args.skip_query:
                rows.extend(query_runs(seed, args888, output_root, device, args.batch_size))
            pd.DataFrame(rows).to_csv(output_root / "per_seed_metrics.csv", index=False)
    finally:
        final1444.moe.gate_feature_for_group = old_builder

    frame = pd.DataFrame(rows)
    aggregate_frame = aggregate(frame)
    aggregate_frame.to_csv(output_root / "aggregate_metrics.csv", index=False)
    full888 = aggregate_frame.loc[
        (aggregate_frame["table"] == "component")
        & (aggregate_frame["protocol"] == "888")
        & (aggregate_frame["variant"] == "full"),
        "mean_iou_mean",
    ]
    full1444 = aggregate_frame.loc[
        (aggregate_frame["table"] == "component")
        & (aggregate_frame["protocol"] == "1444")
        & (aggregate_frame["variant"] == "full"),
        "coverage_mean_iou_mean",
    ]
    controls_pass = (
        len(full888) == 1
        and len(full1444) == 1
        and np.isclose(float(full888.iloc[0]), 0.5486315252772557, atol=1e-12, rtol=0.0)
        and np.isclose(float(full1444.iloc[0]), 0.5372594933248681, atol=1e-12, rtol=0.0)
    )
    status = {
        "status": "complete" if controls_pass else "failed_positive_control",
        "positive_controls_pass": bool(controls_pass),
        "n_runs": int(len(frame)),
        "expected_runs": int(
            len(SEEDS)
            * (
                (0 if args.skip_components else len(COMPONENT_ARMS))
                + (0 if args.skip_query else len(QUERY_VARIANTS))
            )
            * 2
        ),
        "output_root": str(output_root),
    }
    write_json(output_root / "FINAL_STATUS.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

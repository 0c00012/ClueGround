#!/usr/bin/env python
"""Paper-aligned YOLO/RAD-DINO ablation for the no-SigLIP ClueGround main.

The component arms reuse the seed-paired 640-pixel resources and the existing
rule-context decoder.  The full arm is loaded from the exact no-SigLIP main
artifacts, then independently re-evaluated with the same common evaluator.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import clueground_siglip_analysis_common_v1 as common  # noqa: E402
from scripts import run_clueground_architecture_ablation_v1 as architecture  # noqa: E402
from scripts import run_clueground_exact_hybrid_v4_direct_3seed_v1 as exact  # noqa: E402


SEEDS = (13, 42, 2026)
ARMS = ("yolo_only", "rad_dino_only", "yolo_rad_dino")
OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "clueground_no_siglip_component_ablation_v2"
MAIN_888_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "clueground_finding_moe_singlebox_888_no_siglip_score_ablation_v1"
)
MAIN_1444_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "clueground_finding_moe_full_upstream_no_siglip_score_ablation_v1"
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


def parse_boxes(value: Any) -> list[list[float]]:
    boxes = json.loads(str(value)) if isinstance(value, str) else value
    output: list[list[float]] = []
    for box in boxes or []:
        values = [float(item) for item in box]
        if len(values) != 4 or not np.isfinite(values).all():
            raise RuntimeError(f"Invalid prediction box: {box}")
        if values[0] >= values[2] or values[1] >= values[3]:
            raise RuntimeError(f"Degenerate prediction box: {box}")
        output.append(values)
    return output


def load_full_outputs(protocol: str, seed: int) -> tuple[dict[str, list[list[float]]], Path]:
    if protocol == "888":
        path = MAIN_888_ROOT / f"seed_{seed}" / "eval" / "selected" / "per_group.csv"
    else:
        path = MAIN_1444_ROOT / f"seed_{seed}" / "eval" / "gate" / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    outputs = {
        str(row["group_id"]): parse_boxes(row["pred_boxes_json"])
        for row in frame.to_dict("records")
    }
    if len(outputs) != len(frame):
        raise RuntimeError(f"Duplicate group ID in {path}")
    return outputs, path


def normalized_text(value: Any) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).split())


def remap_legacy_1444_ids(
    outputs: dict[str, list[list[float]]],
    groups: dict[str, dict[str, Any]],
) -> dict[str, list[list[float]]]:
    """Map legacy dicom|finding|phrase IDs to canonical task IDs."""
    lookup: dict[tuple[str, str, str], str] = {}
    for group_id, group in groups.items():
        key = (
            normalized_text(group["dicom_id"]),
            normalized_text(group["finding"]),
            normalized_text(group["claim_sentence"]),
        )
        if key in lookup:
            raise RuntimeError(f"Ambiguous canonical 1444 key: {key}")
        lookup[key] = group_id

    remapped: dict[str, list[list[float]]] = {}
    for legacy_id, boxes in outputs.items():
        parts = str(legacy_id).split("|", 2)
        if len(parts) != 3:
            raise RuntimeError(f"Malformed legacy 1444 group ID: {legacy_id}")
        key = tuple(normalized_text(part) for part in parts)
        group_id = lookup.get(key)
        if group_id is None:
            raise RuntimeError(f"No canonical 1444 match for: {legacy_id}")
        if group_id in remapped:
            raise RuntimeError(f"Duplicate remapped 1444 group ID: {group_id}")
        remapped[group_id] = boxes
    return remapped


def expected_main_metric(protocol: str, seed: int) -> float:
    path = (MAIN_888_ROOT if protocol == "888" else MAIN_1444_ROOT) / "per_seed_metrics.csv"
    frame = pd.read_csv(path)
    row = frame.loc[frame["seed"].astype(int) == seed]
    if len(row) != 1:
        raise RuntimeError(f"Missing main metric for {protocol}/seed={seed}")
    column = "selected_mean_iou" if protocol == "888" else "coverage_mean_iou"
    return float(row.iloc[0][column])


def aggregate(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_columns = {
        "888": ("mean_iou", "hit_0_3", "hit_0_5", "mean_pred_count"),
        "1444": (
            "coverage_iou",
            "exact_union_iou",
            "set_f1_0_3",
            "set_f1_0_5",
            "mean_pred_count",
        ),
    }
    for (protocol, arm), part in per_seed.groupby(["protocol", "arm"], sort=False):
        row: dict[str, Any] = {
            "protocol": protocol,
            "arm": arm,
            "n_seeds": int(len(part)),
            "seeds": "/".join(str(value) for value in part["seed"].astype(int)),
        }
        for metric in metric_columns[str(protocol)]:
            values = pd.to_numeric(part[metric], errors="raise").to_numpy(dtype=np.float64)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def run() -> None:
    for directory in ("audit", "metrics", "predictions"):
        (OUTPUT_ROOT / directory).mkdir(parents=True, exist_ok=True)

    method_definitions = {
        "status": "frozen_before_run",
        "seeds": list(SEEDS),
        "siglip_used": False,
        "biomedclip_used": False,
        "shared": [
            "direct-888 eval163 or multibox-1444 eval220",
            "seed-paired 640-pixel upstream resources",
            "raw phrase rule-context parser",
            "same candidate NMS/cardinality decoder for candidate-based arms",
            "same common evaluator",
        ],
        "arms": {
            "yolo_only": (
                "YOLO candidate coordinates ranked by detector confidence; rule-context "
                "multibox decoder retained; RAD-DINO proposal, agreement, and coordinate blend disabled"
            ),
            "rad_dino_only": (
                "standalone phrase-conditioned RAD-DINO proposal; no YOLO proposal is read; "
                "one-box diagnostic on multibox-1444"
            ),
            "yolo_rad_dino": (
                "exact stored no-SigLIP ClueGround main prediction; YOLO, RAD-DINO, "
                "finding/rule context, scorer/gate, and multibox decoder retained"
            ),
        },
        "selection": "No eval-based arm or parameter selection; exact full-main artifacts are immutable.",
    }
    write_json(OUTPUT_ROOT / "METHOD_DEFINITIONS.json", method_definitions)

    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    split_contexts = []
    for protocol in ("888", "1444"):
        for seed in SEEDS:
            resources = common.load_resources(protocol, seed)
            split_contexts.append(resources.context)
            eval_groups = exact.make_groups(resources.context, "eval")
            expected_ids = set(eval_groups)
            for arm in ARMS:
                if arm == "yolo_rad_dino":
                    outputs, source_path = load_full_outputs(protocol, seed)
                    if protocol == "1444":
                        outputs = remap_legacy_1444_ids(outputs, eval_groups)
                    audit = pd.DataFrame(
                        {
                            "protocol": protocol,
                            "seed": seed,
                            "split": "eval",
                            "arm": arm,
                            "group_id": list(outputs),
                            "n_pred": [len(outputs[group_id]) for group_id in outputs],
                            "source": "exact_no_siglip_main_artifact",
                        }
                    )
                    provenance.append(
                        {
                            "protocol": protocol,
                            "seed": seed,
                            "arm": arm,
                            "path": str(source_path.resolve()),
                            "sha256": sha256(source_path),
                        }
                    )
                else:
                    outputs, audit = architecture.predict_arm(resources, "eval", arm)
                    source_path = None

                if set(outputs) != expected_ids:
                    missing = sorted(expected_ids - set(outputs))
                    extra = sorted(set(outputs) - expected_ids)
                    raise RuntimeError(
                        f"Prediction ID mismatch {protocol}/{seed}/{arm}: "
                        f"missing={len(missing)} extra={len(extra)}"
                    )
                summary, detail = common.evaluate(resources, outputs, "eval")
                expected_n = 163 if protocol == "888" else 220
                if int(summary["n"]) != expected_n or int(summary["n_missing_predictions"]) != 0:
                    raise RuntimeError(f"Denominator failure: {protocol}/{seed}/{arm}: {summary}")
                if arm == "yolo_rad_dino":
                    expected_metric = expected_main_metric(protocol, seed)
                    observed_metric = float(
                        summary["mean_iou"] if protocol == "888" else summary["coverage_iou"]
                    )
                    if not np.isclose(observed_metric, expected_metric, atol=1e-12, rtol=0.0):
                        raise RuntimeError(
                            f"Full-main reproduction mismatch {protocol}/{seed}: "
                            f"expected={expected_metric} observed={observed_metric}"
                        )

                root = OUTPUT_ROOT / "predictions" / protocol / f"seed_{seed}"
                common.save_prediction_map(root / f"{arm}.json", outputs)
                audit.to_csv(root / f"{arm}_audit.csv", index=False)
                detail.to_csv(root / f"{arm}_per_group.csv", index=False)
                rows.append({"protocol": protocol, "seed": seed, "arm": arm, **summary})

    per_seed = pd.DataFrame(rows)
    aggregate_frame = aggregate(per_seed)
    per_seed.to_csv(OUTPUT_ROOT / "metrics" / "component_per_seed.csv", index=False)
    aggregate_frame.to_csv(OUTPUT_ROOT / "metrics" / "component_aggregate.csv", index=False)
    pd.DataFrame(provenance).to_csv(OUTPUT_ROOT / "audit" / "FULL_MAIN_PROVENANCE.csv", index=False)
    split_audit = exact.split_audit(split_contexts)
    write_json(OUTPUT_ROOT / "audit" / "SPLIT_OVERLAP_AUDIT.json", split_audit)

    full_888 = aggregate_frame.loc[
        (aggregate_frame["protocol"] == "888") & (aggregate_frame["arm"] == "yolo_rad_dino")
    ].iloc[0]
    full_1444 = aggregate_frame.loc[
        (aggregate_frame["protocol"] == "1444") & (aggregate_frame["arm"] == "yolo_rad_dino")
    ].iloc[0]
    status = {
        "status": "complete" if split_audit.get("status") == "PASS" else "failed_audit",
        "n_expected_runs": 18,
        "n_completed_runs": int(len(per_seed)),
        "seeds": list(SEEDS),
        "siglip_used": False,
        "biomedclip_used": False,
        "denominators": {"888": 163, "1444_groups": 220, "1444_boxes": 280},
        "missing_predictions": int(per_seed["n_missing_predictions"].sum()),
        "full_main_reproduced": {
            "888_mean_iou": float(full_888["mean_iou_mean"]),
            "1444_coverage_iou": float(full_1444["coverage_iou_mean"]),
        },
    }
    write_json(OUTPUT_ROOT / "FINAL_STATUS.json", status)
    print(aggregate_frame.to_string(index=False))


if __name__ == "__main__":
    run()

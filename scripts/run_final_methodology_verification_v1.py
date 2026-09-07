#!/usr/bin/env python
"""Final leakage, reproducibility, and component audit for the MS-CXR method.

The run is intentionally limited to MS-CXR-only model fitting. It retrains the
three-expert YOLO/RAD-DINO + SigLIP + BioMedCLIP gate after removing annotated
GT cardinality from its inference features, evaluates matched component
ablations, and verifies prediction invariance to randomized eval gold.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import audit_methodology_integrity_v1 as integrity  # noqa: E402
from scripts import run_ms_cxr_semantic_weight_finegrid_v2_yolov8l_pool as fine  # noqa: E402
from scripts import run_ms_cxr_trainable_moe_gate_v1 as moe  # noqa: E402
from scripts import run_rerank_experiments as rerank  # noqa: E402
from src.rerank.base_preserving_multibox_rescue import pred_rows_to_map  # noqa: E402
from src.rerank.cardinality_head import (  # noqa: E402
    CountHeadBundle,
    apply_count_to_base,
    predict_counts,
)
from src.rerank.multibox_cue_parser import cue_info_from_groups  # noqa: E402


DEFAULT_OUT = (
    PROJECT_ROOT
    / "experiments"
    / "final_methodology_verification"
    / "20260712_v1"
)
UNIFIED_SOURCE = (
    PROJECT_ROOT
    / "experiments"
    / "unified_cardinality_reexperiment"
    / "20260710_before_power_v1"
    / "retrained_count_head"
)
COMPONENT_SOURCE = (
    PROJECT_ROOT
    / "experiments"
    / "ms_cxr_multibox_rule_context_fusion_v4_yolov8l_pool"
    / "predictions"
    / "phrase_group_set_predictions.csv"
)
SEMANTIC_SOURCE = (
    PROJECT_ROOT
    / "experiments"
    / "ms_cxr_semantic_weight_finegrid_v2_yolov8l_pool"
    / "predictions"
    / "semantic_weight_finegrid_phrase_group_predictions.csv"
)
FORBIDDEN = re.compile(
    r"(^|_)(gold|gt|target|label|oracle|positive|answer|matched|candidate_source|hard_negative_type|negative_type)(_|$)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", default="13,42,2026")
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--reuse-clean-moe", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )


def json_safe(value: Any) -> Any:
    """Convert numpy scalars and non-finite floats to strict JSON values."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_boxes(preds: dict[str, list[dict[str, Any]]]) -> dict[str, list[list[float]]]:
    return {
        str(gid): [[float(value) for value in pred["box"]] for pred in values]
        for gid, values in preds.items()
    }


def maps_equal(
    left: dict[str, list[dict[str, Any]]],
    right: dict[str, list[dict[str, Any]]],
    *,
    atol: float = 1e-8,
) -> tuple[bool, int, float]:
    a = prediction_boxes(left)
    b = prediction_boxes(right)
    changed = 0
    max_diff = 0.0
    for gid in sorted(set(a) | set(b)):
        av = a.get(gid, [])
        bv = b.get(gid, [])
        if len(av) != len(bv):
            changed += 1
            continue
        same = True
        for box_a, box_b in zip(av, bv):
            diff = float(np.max(np.abs(np.asarray(box_a) - np.asarray(box_b))))
            max_diff = max(max_diff, diff)
            same = same and diff <= atol
        if not same:
            changed += 1
    return changed == 0, changed, max_diff


def mutate_gold(groups: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = copy.deepcopy(groups)
    for index, group in enumerate(out.values()):
        width = float(group["image_width"])
        height = float(group["image_height"])
        count = 3 if index % 2 == 0 else 1
        group["gt_boxes"] = [
            [
                0.01 * width * (offset + 1),
                0.01 * height * (offset + 1),
                0.02 * width * (offset + 1),
                0.02 * height * (offset + 1),
            ]
            for offset in range(count)
        ]
    return out


def rotate_finding_annotations(groups: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = copy.deepcopy(groups)
    findings = sorted({str(group["finding"]) for group in out.values()})
    rotated = {finding: findings[(index + 1) % len(findings)] for index, finding in enumerate(findings)}
    for group in out.values():
        group["finding"] = rotated[str(group["finding"])]
    return out


def method_bundle(
    method: str,
    groups: dict[str, dict[str, Any]],
    preds: dict[str, list[dict[str, Any]]],
    source: str,
) -> integrity.MethodPredictions:
    return integrity.MethodPredictions(
        protocol="unified_variable_cardinality",
        method=method,
        status="controlled_mscxr_only_retrain",
        source_path=source,
        preds=prediction_boxes(preds),
    )


def summarize_frame(frame: pd.DataFrame, method: str, subset: str) -> dict[str, Any]:
    out: dict[str, Any] = {
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
    if (frame["n_gt"] == 1).all():
        out.update(
            {
                "top1_mean_iou": float(frame["top1_iou"].mean()),
                "top1_hit_0_3": float((frame["top1_iou"] >= 0.3).mean()),
                "top1_hit_0_5": float((frame["top1_iou"] >= 0.5).mean()),
            }
        )
    return out


def evaluate_map(
    method: str,
    groups: dict[str, dict[str, Any]],
    preds: dict[str, list[dict[str, Any]]],
    source: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    bundle = method_bundle(method, groups, preds, source)
    rows, _ = integrity.evaluate_bundle(bundle, groups)
    frame = pd.DataFrame(rows)
    frame["top1_iou"] = [
        integrity.box_iou(groups[gid]["gt_boxes"][0], bundle.preds.get(gid, [])[0])
        if groups[gid]["gt_boxes"] and bundle.preds.get(gid, [])
        else 0.0
        for gid in frame["group_id"].astype(str)
    ]
    subsets = {
        "all_220": frame,
        "singleton_163": frame[frame["n_gt"].astype(int).eq(1)],
        "multibox_57": frame[frame["n_gt"].astype(int).gt(1)],
    }
    summaries = [summarize_frame(part, method, subset) for subset, part in subsets.items()]
    return frame, summaries


def save_predictions(path: Path, preds: dict[str, list[dict[str, Any]]]) -> None:
    rows = [
        {
            "group_id": gid,
            "pred_boxes_json": json.dumps(boxes),
            "n_pred": len(boxes),
        }
        for gid, boxes in prediction_boxes(preds).items()
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def cluster_bootstrap(
    left: pd.DataFrame,
    right: pd.DataFrame,
    comparison: str,
    reps: int,
) -> pd.DataFrame:
    metrics = [
        "coverage_mean_iou",
        "enclosing_hull_iou",
        "rectangle_union_iou",
        "raster_union_iou_224",
        "set_f1_greedy_0_3",
        "set_f1_greedy_0_5",
    ]
    joined = left[["group_id", "subject_id", *metrics]].merge(
        right[["group_id", *metrics]], on="group_id", suffixes=("_left", "_right"), validate="one_to_one"
    )
    subjects = sorted(joined["subject_id"].astype(str).unique())
    by_subject = {subject: joined[joined["subject_id"].astype(str).eq(subject)] for subject in subjects}
    rng = np.random.default_rng(20260712)
    rows = []
    for metric in metrics:
        observed = float((joined[f"{metric}_left"] - joined[f"{metric}_right"]).mean())
        samples = np.empty(reps, dtype=float)
        for index in range(reps):
            chosen = rng.choice(subjects, size=len(subjects), replace=True)
            values = [
                by_subject[str(subject)][f"{metric}_left"].to_numpy(dtype=float)
                - by_subject[str(subject)][f"{metric}_right"].to_numpy(dtype=float)
                for subject in chosen
            ]
            samples[index] = float(np.concatenate(values).mean())
        rows.append(
            {
                "comparison": comparison,
                "metric": metric,
                "mean_difference": observed,
                "ci_95_low": float(np.quantile(samples, 0.025)),
                "ci_95_high": float(np.quantile(samples, 0.975)),
                "ci_excludes_zero": bool(np.quantile(samples, 0.025) > 0 or np.quantile(samples, 0.975) < 0),
                "bootstrap_reps": reps,
            }
        )
    return pd.DataFrame(rows)


def run_clean_moe(
    out: Path,
    seeds: list[int],
    device: str,
    bootstrap_reps: int,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    root = out / "moe_clean"
    for path in [root, root / "checkpoints", root / "logs", root / "metrics", root / "predictions", root / "audit"]:
        path.mkdir(parents=True, exist_ok=True)
    moe.CKPT = root / "checkpoints"
    moe.LOG = root / "logs"
    moe.MET = root / "metrics"
    moe.PRED = root / "predictions"

    pretrain_name = "disabled_for_mscxr_only_verification"
    train_bundle = moe.build_bundle("train", device, pretrain_name, include_pretrain_expert=False)
    val_bundle = moe.build_bundle("val", device, pretrain_name, include_pretrain_expert=False)
    eval_bundle = moe.build_bundle("eval", device, pretrain_name, include_pretrain_expert=False)

    arms: list[tuple[str, list[str], int | None]] = [
        ("hybrid_only", ["hybrid"], None),
        ("hybrid_plus_siglip_s2026", ["hybrid", "siglip"], 2026),
        ("hybrid_plus_biomedclip_s2026", ["hybrid", "biomed"], 2026),
    ]
    arms.extend((f"hybrid_siglip_biomedclip_s{seed}", ["hybrid", "siglip", "biomed"], seed) for seed in seeds)
    arms.append(("hybrid_siglip_biomedclip_s2026_replay", ["hybrid", "siglip", "biomed"], 2026))

    all_summaries: list[dict[str, Any]] = []
    all_frames: dict[str, pd.DataFrame] = {}
    all_preds: dict[str, dict[str, list[dict[str, Any]]]] = {}
    params_rows: list[dict[str, Any]] = []
    models: dict[str, torch.nn.Module] = {}
    for name, experts, seed in arms:
        if seed is None:
            preds = eval_bundle.hybrid
            audit = pd.DataFrame({"group_id": list(eval_bundle.groups), "action": "hybrid_only"})
        else:
            model, params, _, _ = moe.train_gate(
                name,
                train_bundle,
                val_bundle,
                experts,
                hardneg=False,
                seed=seed,
                device=device,
            )
            preds, audit = moe.predict_gate(name, model, eval_bundle, experts, device=device, keep_multi=True)
            models[name] = model
            params_rows.append(params)
        save_predictions(root / "predictions" / f"{name}.csv", preds)
        audit.to_csv(root / "audit" / f"{name}_actions.csv", index=False)
        frame, summaries = evaluate_map(name, eval_bundle.groups, preds, str(root / "predictions" / f"{name}.csv"))
        frame.to_csv(root / "metrics" / f"{name}_per_group.csv", index=False)
        all_summaries.extend(summaries)
        all_frames[name] = frame
        all_preds[name] = preds

    summary = pd.DataFrame(all_summaries)
    summary.to_csv(root / "metrics" / "matched_arm_metrics.csv", index=False)
    pd.DataFrame(params_rows).to_csv(root / "metrics" / "training_selection_summary.csv", index=False)

    full_names = [f"hybrid_siglip_biomedclip_s{seed}" for seed in seeds]
    seed_rows = summary[summary["method"].isin(full_names)].copy()
    numeric = [
        "coverage_mean_iou",
        "enclosing_hull_iou",
        "exact_rectangle_union_iou",
        "raster_union_iou_224",
        "set_f1_0_3",
        "set_f1_0_5",
        "mean_pred_count",
        "top1_mean_iou",
        "top1_hit_0_3",
        "top1_hit_0_5",
    ]
    aggregate_rows = []
    for subset, part in seed_rows.groupby("subset"):
        row: dict[str, Any] = {"method": "hybrid_siglip_biomedclip_3seed", "subset": subset, "n_seeds": len(part)}
        for column in numeric:
            if column in part and part[column].notna().any():
                row[f"{column}_mean"] = float(part[column].mean())
                row[f"{column}_std"] = float(part[column].std(ddof=1)) if len(part) > 1 else 0.0
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(root / "metrics" / "three_seed_aggregate.csv", index=False)

    replay_ok, replay_changed, replay_diff = maps_equal(
        all_preds["hybrid_siglip_biomedclip_s2026"],
        all_preds["hybrid_siglip_biomedclip_s2026_replay"],
    )
    fake_bundle = copy.deepcopy(eval_bundle)
    fake_bundle.groups = mutate_gold(eval_bundle.groups)
    model_2026 = models["hybrid_siglip_biomedclip_s2026"]
    fake_preds, _ = moe.predict_gate(
        "gold_mutation_probe",
        model_2026,
        fake_bundle,
        ["hybrid", "siglip", "biomed"],
        device=device,
        keep_multi=True,
    )
    gold_ok, gold_changed, gold_diff = maps_equal(all_preds["hybrid_siglip_biomedclip_s2026"], fake_preds)

    bootstrap_parts = []
    full_frame = all_frames["hybrid_siglip_biomedclip_s2026"]
    for other in ["hybrid_only", "hybrid_plus_siglip_s2026", "hybrid_plus_biomedclip_s2026"]:
        bootstrap_parts.append(
            cluster_bootstrap(full_frame, all_frames[other], f"full_s2026_minus_{other}", bootstrap_reps)
        )
    bootstrap = pd.concat(bootstrap_parts, ignore_index=True)
    bootstrap.to_csv(root / "metrics" / "component_paired_bootstrap.csv", index=False)

    status = {
        "state": "complete",
        "supervision": "MS-CXR task supervision only; generic foundation initialization allowed",
        "external_imagenome_expert_loaded": False,
        "seeds": seeds,
        "deterministic_replay": {
            "status": "PASS" if replay_ok else "FAIL",
            "changed_groups": replay_changed,
            "max_coordinate_difference": replay_diff,
        },
        "gold_mutation_independence": {
            "status": "PASS" if gold_ok else "FAIL",
            "changed_groups": gold_changed,
            "max_coordinate_difference": gold_diff,
        },
        "aggregate": aggregate.to_dict("records"),
    }
    write_json(root / "FINAL_STATUS.json", status)
    return status, all_preds["hybrid_siglip_biomedclip_s2026"]


def load_saved_clean_moe(out: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    root = out / "moe_clean"
    status = json.loads((root / "FINAL_STATUS.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(root / "predictions" / "hybrid_siglip_biomedclip_s2026.csv")
    preds = {
        str(row["group_id"]): [
            {"box": [float(value) for value in box], "score": 1.0, "source": "saved_clean_moe"}
            for box in integrity.safe_boxes(row["pred_boxes_json"])
        ]
        for _, row in frame.iterrows()
    }
    return status, preds


def run_finding_annotation_independence(out: Path, device: str) -> dict[str, Any]:
    root = out / "moe_clean"
    bundle = moe.build_bundle(
        "eval",
        device,
        "disabled_for_mscxr_only_verification",
        include_pretrain_expert=False,
    )
    payload = torch.load(
        root / "checkpoints" / "hybrid_siglip_biomedclip_s2026.pt",
        map_location="cpu",
        weights_only=False,
    )
    experts = [str(value) for value in payload["params"]["experts"]]
    model = moe.MoEGate(int(payload["in_dim"]), int(payload["n_experts"]))
    model.load_state_dict(payload["state"], strict=True)
    model.to(device)
    original, _ = moe.predict_gate(
        "finding_original", model, bundle, experts, device=device, keep_multi=True
    )

    mutated = copy.deepcopy(bundle)
    mutated.groups = rotate_finding_annotations(bundle.groups)
    mutated.hybrid, mutated.cue = moe.fine_base.build_hybrid_v4("eval", mutated.groups)
    changed, _ = moe.predict_gate(
        "finding_rotated", model, mutated, experts, device=device, keep_multi=True
    )
    same, n_changed, max_diff = maps_equal(original, changed)
    status = {
        "status": "PASS" if same else "FAIL_PHRASE_ONLY_CONTRACT",
        "n_groups": len(bundle.groups),
        "n_changed_groups": n_changed,
        "max_coordinate_difference": max_diff,
        "checkpoint_input_dim": int(payload["in_dim"]),
        "probe": "raw phrase and image fixed; only dataset finding/category annotation rotated before rebuilding upstream hybrid expert",
        "interpretation": (
            "Any changed prediction proves upstream annotation-category dependence even when the gate itself has no finding one-hot."
        ),
    }
    write_json(out / "source_leak_guard" / "finding_annotation_independence.json", status)
    return status


def run_semantic_gold_independence(out: Path) -> dict[str, Any]:
    params = json.loads(
        (
            PROJECT_ROOT
            / "experiments"
            / "ms_cxr_semantic_weight_finegrid_v2_yolov8l_pool"
            / "configs"
            / "best_finegrid_params.json"
        ).read_text(encoding="utf-8")
    )
    groups, hybrid, siglip, biomed, cue = fine.base.build_inputs("eval")
    original, _ = fine.base.predict(
        groups,
        hybrid,
        siglip,
        biomed,
        cue,
        (float(params["w_hybrid"]), float(params["w_siglip"]), float(params["w_biomed"])),
        bool(params["keep_multi"]),
        float(params["min_hybrid_semantic_iou"]),
    )
    fake_groups = mutate_gold(groups)
    fake_hybrid, fake_cue = fine.base.build_hybrid_v4("eval", fake_groups)
    fake_siglip = fine.base.sem2.build_siglip_set("eval", fake_groups)
    fake_biomed = fine.base.sem2.build_biomedclip_set("eval", fake_groups)
    mutated, _ = fine.base.predict(
        fake_groups,
        fake_hybrid,
        fake_siglip,
        fake_biomed,
        fake_cue,
        (float(params["w_hybrid"]), float(params["w_siglip"]), float(params["w_biomed"])),
        bool(params["keep_multi"]),
        float(params["min_hybrid_semantic_iou"]),
    )
    same, changed, max_diff = maps_equal(original, mutated)
    status = {
        "status": "PASS" if same else "FAIL",
        "changed_groups": changed,
        "max_coordinate_difference": max_diff,
        "probe": "all eval GT boxes and cardinalities replaced before rebuilding hybrid/SigLIP/BioMedCLIP inputs",
    }
    write_json(out / "source_leak_guard" / "semantic_finegrid_gold_independence.json", status)
    return status


def score_unified_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    weights = json.loads((UNIFIED_SOURCE / "configs" / "best_gate_weights.json").read_text(encoding="utf-8"))
    scaler_path = UNIFIED_SOURCE / "configs" / "gate_train_scaler.json"
    scaler = json.loads(scaler_path.read_text(encoding="utf-8")) if scaler_path.exists() else None
    scored = rerank.add_gate_score(candidates, weights, scaler)
    payload = joblib.load(UNIFIED_SOURCE / "models" / "learned_candidate_reranker.joblib")
    features = [str(value) for value in payload["feature_columns"]]
    scored["reranker_score"] = payload["model"].predict(scored.reindex(columns=features, fill_value=0.0))
    scored["full_score"] = 0.55 * scored["reranker_score"].astype(float) + 0.45 * scored["score_gate"].astype(float)
    return scored


def run_count_head_gold_independence(out: Path) -> dict[str, Any]:
    groups = integrity.canonical_groups()["eval"]
    fake_groups = mutate_gold(groups)
    raw = pd.read_csv(UNIFIED_SOURCE / "candidate_table" / "eval_candidate_table.csv")
    forbidden_columns = [
        column
        for column in raw.columns
        if FORBIDDEN.search(str(column)) or str(column).lower() in {"source_model", "candidate_source"}
    ]
    stripped = raw.drop(columns=forbidden_columns, errors="ignore")
    scored_raw = score_unified_candidates(raw)
    scored_stripped = score_unified_candidates(stripped)

    payload = joblib.load(UNIFIED_SOURCE / "models" / "count_head.joblib")
    bundle = CountHeadBundle(payload["model"], [str(value) for value in payload["feature_columns"]])
    base_rows = pd.read_csv(UNIFIED_SOURCE / "per_query" / "base_finegrid_contrastive_global_eval_predictions.csv")
    base_preds = pred_rows_to_map(base_rows)
    cues = cue_info_from_groups(groups)
    fake_cues = cue_info_from_groups(fake_groups)
    original_counts = predict_counts(bundle, groups, base_preds, scored_raw, cues, score_col="full_score")
    mutated_counts = predict_counts(bundle, fake_groups, base_preds, scored_stripped, fake_cues, score_col="full_score")
    count_columns = [column for column in original_counts.columns if column != "group_id"]
    left = original_counts.sort_values("group_id").reset_index(drop=True)
    right = mutated_counts.sort_values("group_id").reset_index(drop=True)
    count_same = left["group_id"].equals(right["group_id"]) and np.allclose(
        left[count_columns].to_numpy(dtype=float), right[count_columns].to_numpy(dtype=float), atol=1e-12
    )
    original_preds, _ = apply_count_to_base(base_preds, scored_raw, original_counts, score_col="full_score")
    mutated_preds, _ = apply_count_to_base(base_preds, scored_stripped, mutated_counts, score_col="full_score")
    pred_same, changed, max_diff = maps_equal(original_preds, mutated_preds)
    forbidden_features = [feature for feature in bundle.feature_columns if FORBIDDEN.search(feature)]
    reranker_payload = joblib.load(UNIFIED_SOURCE / "models" / "learned_candidate_reranker.joblib")
    forbidden_features += [
        str(feature) for feature in reranker_payload["feature_columns"] if FORBIDDEN.search(str(feature))
    ]
    status = {
        "status": "PASS" if count_same and pred_same and not forbidden_features else "FAIL",
        "raw_columns_physically_dropped": sorted(forbidden_columns),
        "n_raw_columns_physically_dropped": len(forbidden_columns),
        "forbidden_model_features": sorted(set(forbidden_features)),
        "count_outputs_identical": bool(count_same),
        "predictions_identical": bool(pred_same),
        "changed_groups": changed,
        "max_coordinate_difference": max_diff,
        "note": "This checks the compatible retrained tabular heads; the frozen external base itself is unchanged.",
    }
    write_json(out / "source_leak_guard" / "count_head_gold_independence.json", status)
    return status


def frame_to_prediction_map(frame: pd.DataFrame, method: str) -> dict[str, list[dict[str, Any]]]:
    part = frame[frame["method"].astype(str).eq(method)] if "method" in frame else frame
    return {
        str(row["group_id"]): [
            {"box": [float(value) for value in box], "score": 1.0, "source": method}
            for box in integrity.safe_boxes(row["pred_boxes_json"])
        ]
        for _, row in part.iterrows()
    }


def run_component_accounting(
    out: Path,
    clean_full_preds: dict[str, list[dict[str, Any]]],
) -> pd.DataFrame:
    groups = integrity.canonical_groups()["eval"]
    component_frame = pd.read_csv(COMPONENT_SOURCE)
    specs = [
        ("YOLO-only proposal set", "yolov8n_yolov8s_yolov8m_yolov8l_detection_set_val_tuned", COMPONENT_SOURCE),
        ("RAD-DINO rule-context only", "rad_dino_rule_context", COMPONENT_SOURCE),
        ("YOLO + RAD-DINO hybrid", "hybrid_singlebox_fusion_plus_context_multibox_v4_yolov8l_pool", COMPONENT_SOURCE),
        ("YOLO + RAD-DINO + SigLIP + BioMedCLIP semantic finegrid", "semantic_weight_finegrid_v2_yolov8l_pool", SEMANTIC_SOURCE),
    ]
    rows = []
    for display, method, path in specs:
        frame = component_frame if path == COMPONENT_SOURCE else pd.read_csv(path)
        preds = frame_to_prediction_map(frame, method)
        _, summaries = evaluate_map(display, groups, preds, str(path))
        rows.extend(summaries)
    _, clean_summaries = evaluate_map(
        "leak-fixed trainable 3-expert gate s2026", groups, clean_full_preds, "fresh verification run"
    )
    rows.extend(clean_summaries)
    result = pd.DataFrame(rows)
    path = out / "components" / "stagewise_component_metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    return result


def runtime_audit(out: Path) -> dict[str, Any]:
    paths = {
        "compatible_reranker": UNIFIED_SOURCE / "models" / "learned_candidate_reranker.joblib",
        "compatible_count_head": UNIFIED_SOURCE / "models" / "count_head.joblib",
        "legacy_singlebox_reranker": (
            PROJECT_ROOT
            / "experiments"
            / "rerank_dino_gate"
            / "20260707_163626_cue_count_fix"
            / "singlebox_888"
            / "models"
            / "learned_candidate_reranker.joblib"
        ),
        "legacy_multibox_reranker": (
            PROJECT_ROOT
            / "experiments"
            / "multibox_dev"
            / "20260707_multibox_dev_v1"
            / "models"
            / "learned_candidate_reranker.joblib"
        ),
    }
    artifacts = []
    for name, path in paths.items():
        status = "MISSING"
        detail = ""
        if path.exists():
            try:
                joblib.load(path)
                status = "PASS"
            except Exception as exc:  # noqa: BLE001
                status = "FAIL"
                detail = f"{type(exc).__name__}: {exc}"
        artifacts.append(
            {
                "artifact": name,
                "path": str(path),
                "exists": path.exists(),
                "load_status": status,
                "detail": detail,
                "sha256": sha256_file(path) if path.exists() else "",
            }
        )
    frame = pd.DataFrame(artifacts)
    target = out / "runtime" / "joblib_compatibility.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    status = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "compatible_final_heads_load": bool(
            frame[frame["artifact"].isin(["compatible_reranker", "compatible_count_head"])]["load_status"].eq("PASS").all()
        ),
        "legacy_failures": int(
            frame[frame["artifact"].str.startswith("legacy_")]["load_status"].eq("FAIL").sum()
        ),
    }
    write_json(out / "runtime" / "environment_status.json", status)
    return status


def main() -> None:
    args = parse_args()
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in str(args.seeds).split(",") if value.strip()]
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.reuse_clean_moe:
        moe_status, clean_full_preds = load_saved_clean_moe(out)
    else:
        moe_status, clean_full_preds = run_clean_moe(out, seeds, device, args.bootstrap_reps)
    finding_status = run_finding_annotation_independence(out, device)
    semantic_status = run_semantic_gold_independence(out)
    try:
        count_status = run_count_head_gold_independence(out)
    except Exception as exc:  # noqa: BLE001 - legacy pickle incompatibility is an audited blocker
        count_status = {
            "status": "BLOCKED_RUNTIME_INCOMPATIBLE",
            "error": f"{type(exc).__name__}: {exc}",
            "training_or_predictions_changed": False,
        }
        write_json(out / "source_leak_guard" / "count_head_gold_independence.json", count_status)
    components = run_component_accounting(out, clean_full_preds)
    runtime = runtime_audit(out)

    final = {
        "state": "complete",
        "direct_split_overlap": "see integrity/split audit",
        "clean_moe": moe_status,
        "finding_annotation_independence": finding_status,
        "phrase_only_comparable": finding_status["status"] == "PASS",
        "semantic_finegrid_gold_independence": semantic_status,
        "count_head_gold_independence": count_status,
        "runtime": runtime,
        "all_dynamic_gold_independence_pass": bool(
            moe_status["gold_mutation_independence"]["status"] == "PASS"
            and semantic_status["status"] == "PASS"
            and count_status["status"] == "PASS"
        ),
        "component_rows": int(len(components)),
        "full_cig_pretraining_run": False,
    }
    write_json(out / "RUNTIME_VERIFICATION_STATUS.json", final)
    print(json.dumps(json_safe(final), ensure_ascii=False, indent=2, default=str, allow_nan=False))


if __name__ == "__main__":
    main()

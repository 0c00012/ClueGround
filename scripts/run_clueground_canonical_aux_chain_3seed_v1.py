#!/usr/bin/env python
"""Per-seed regeneration of the ClueGround 1444 auxiliary candidate chain.

The paper's historical 1444 value (Coverage 0.5373) combined a seed-specific
YOLO--RAD-DINO hybrid with an auxiliary candidate set that was a *fixed*
seed-42 asset: yolov8n/yolov8s "full" detectors, a RAD-DINO cross-attention
context head, a row-level candidate scorer (``score_head``), a fixed
candidate table, and a finding-conditioned MoE gate.

This runner rebuilds that auxiliary chain independently for each seed so that
the three reported runs are genuinely independent, gives the train split the
same ``score_head`` schema as val/eval (out-of-fold), and evaluates on the
canonical 813/124/220 phrase groups with the canonical per-seed hybrid.

Two modes:

* ``legacy_assets_check``: rebuild the candidate table and scorer from the
  archived seed-42 detector/xattn assets, then run the gate stage for seeds
  13/42/2026 exactly as the historical runner did.  The result must reproduce
  the archived per-seed metrics; this validates the new code path.
* ``canonical``: train yolov8n/yolov8s and the xattn head per seed, build the
  candidate table and scorer per seed (train OOF), tune set parameters on the
  canonical validation groups, and run the gate with the canonical hybrid.

Nothing outside ``--output-root`` is written.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_clueground_exact_hybrid_v4_direct_3seed_v1 as exact  # noqa: E402
from scripts import run_clueground_finding_moe_full_upstream_siglip_only_3seed_v1 as source  # noqa: E402
from scripts import run_clueground_moe_transplant_diagnostic_3seed_v1 as transplant  # noqa: E402
from scripts import run_clueground_vfm_legacy_fusion_unified_3seed_v1 as multi_source  # noqa: E402
from scripts import run_final_methodology_verification_v1 as verify  # noqa: E402
from scripts import run_ms_cxr_multibox_phrase_grounding_v1 as mb  # noqa: E402
from scripts import run_ms_cxr_rad_dino_transvg_context_head_v1 as xattn  # noqa: E402
from scripts import run_ms_cxr_rowlevel_candidate_set_scorer_v1 as rs  # noqa: E402
from scripts import run_ms_cxr_siglip_candidate_fusion_v1 as sf  # noqa: E402
from scripts import run_ms_cxr_trainable_moe_gate_v1 as moe  # noqa: E402
from scripts import run_ms_cxr_yolo_detector_v1 as yd  # noqa: E402


SEEDS = (13, 42, 2026)
SPLITS = ("train", "val", "eval")
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "clueground_canonical_aux_chain_v1"

CANONICAL_PROTOCOL_ROOT = (
    PROJECT_ROOT
    / "training"
    / "three_task_clueground_vfm_finding_conditioned_canonical_v3"
    / "protocols"
    / "task_isolated"
)
CANONICAL_UPSTREAM_ROOT = PROJECT_ROOT / "experiments" / "clueground_canonical_v3_hybrid_then_moe_3seed_v1"
CANONICAL_BASE_ROOTS = {
    13: PROJECT_ROOT / "experiments" / "clueground_canonical_exact_hybrid_v4_fullval_baseline_s13_v1",
    42: PROJECT_ROOT / "experiments" / "clueground_canonical_exact_hybrid_v4_fullval_baseline_remaining_42_2026_v1",
    2026: PROJECT_ROOT / "experiments" / "clueground_canonical_exact_hybrid_v4_fullval_baseline_remaining_42_2026_v1",
}
LEGACY_PROTOCOL_ROOT = exact.PROTOCOL_ROOT
LEGACY_UPSTREAM_ROOT = exact.MULTI_SOURCE_ROOT
LEGACY_BASE_ROOT = transplant.BASE_ROOT

LEGACY_DETECTOR_RUNS = PROJECT_ROOT / "training" / "ms_cxr_yolo_detector_v1" / "runs"
LEGACY_XATTN_CKPT = PROJECT_ROOT / "training" / "ms_cxr_rad_dino_transvg_context_head_v1" / "checkpoints" / "row1444_rule_plus_full_s42_best.pt"
LEGACY_XATTN_PRED = PROJECT_ROOT / "experiments" / "ms_cxr_rad_dino_transvg_context_head_v1" / "predictions"
LEGACY_SCORER_PRED = PROJECT_ROOT / "experiments" / "ms_cxr_rowlevel_candidate_set_scorer_v1" / "predictions"
LEGACY_SEMANTIC_PRED = PROJECT_ROOT / "experiments" / "ms_cxr_siglip_candidate_fusion_v1" / "predictions"
LEGACY_SET_PARAMS = PROJECT_ROOT / "experiments" / "ms_cxr_siglip_candidate_fusion_v1" / "configs" / "best_set_params.json"
LEGACY_RESULT = PROJECT_ROOT / "experiments" / "clueground_finding_moe_full_upstream_no_siglip_score_ablation_v1" / "per_seed_metrics.csv"

DETECTORS = {"yolov8n": PROJECT_ROOT / "yolov8n.pt", "yolov8s": PROJECT_ROOT / "yolov8s.pt"}
MAX_CANDIDATES_PER_TASK = 12
SCORE_FORMULA = "score_head + 0.05*prior_iou + 0.03*confidence"


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------


def log(message: str) -> None:
    print(f"[aux-chain {time.strftime('%H:%M:%S')}] {message}", flush=True)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.zeros(len(frame), dtype=np.float64)
    return np.nan_to_num(
        pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def require(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


# ----------------------------------------------------------------------------
# stage 1: light detectors per seed
# ----------------------------------------------------------------------------


def train_detector(seed: int, tag: str, runs_dir: Path, device: str, epochs: int) -> Path:
    """Mirror ``yd.run_train`` with a caller-controlled seed and output root."""
    from ultralytics import YOLO

    name = f"{tag}_full_s{seed}"
    best = runs_dir / name / "weights" / "best.pt"
    if best.exists():
        log(f"seed {seed}: detector {tag} exists, skip")
        return best
    log(f"seed {seed}: train detector {tag} ({epochs} epochs)")
    model = YOLO(str(DETECTORS[tag]))
    model.train(
        data=str(yd.DATASET / "ms_cxr_yolo.yaml"),
        epochs=epochs,
        imgsz=640,
        batch=16,
        workers=0,
        project=str(runs_dir),
        name=name,
        exist_ok=True,
        device=device,
        pretrained=True,
        seed=seed,
        patience=30,
    )
    if not best.exists():
        raise FileNotFoundError(best)
    return best


# ----------------------------------------------------------------------------
# stage 2: RAD-DINO cross-attention context head per seed
# ----------------------------------------------------------------------------


def xattn_args(seed: int) -> SimpleNamespace:
    ckpt = torch.load(LEGACY_XATTN_CKPT, map_location="cpu")
    args = dict(ckpt["args"])
    args["seed"] = int(seed)
    return SimpleNamespace(**args)


def point_xattn(root: Path) -> None:
    xattn.CKPT = root / "checkpoints"
    xattn.MET = root / "metrics"
    xattn.REPORT = root / "reports"
    xattn.PRED = root / "predictions"
    for path in (xattn.CKPT, xattn.MET, xattn.REPORT, xattn.PRED):
        path.mkdir(parents=True, exist_ok=True)


def train_xattn(seed: int, root: Path) -> Path:
    point_xattn(root)
    ckpt_path = xattn.CKPT / f"row1444_rule_plus_full_s{seed}_best.pt"
    args = xattn_args(seed)
    if not ckpt_path.exists():
        log(f"seed {seed}: train xattn head ({args.epochs} epochs)")
        stats = xattn.train_context_head(args, "row1444", "rule_plus_full")
        write_json(root / "train_stats.json", stats)
    predictions_missing = [
        split for split in SPLITS if not (xattn.PRED / f"row1444_rule_plus_full_{split}_predictions.jsonl").exists()
    ]
    if predictions_missing:
        ckpt = torch.load(ckpt_path, map_location=xattn.DEVICE)
        model = xattn.RadDinoContextCrossAttentionHead(
            token_dim=ckpt["token_dim"],
            query_dim=ckpt["query_dim"],
            hidden=args.hidden,
            num_query_tokens=args.num_query_tokens,
            num_heads=args.num_heads,
            dropout=args.dropout,
        ).to(xattn.DEVICE)
        model.load_state_dict(ckpt["model"])
        model.eval()
        for split in predictions_missing:
            dataset = xattn.MSContextDataset("row1444", "rule_plus_full", split)
            log(f"seed {seed}: xattn predict {split} n={len(dataset)}")
            xattn.eval_model(model, dataset, args.batch_size, xattn.PRED / f"row1444_rule_plus_full_{split}_predictions.jsonl")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return ckpt_path


# ----------------------------------------------------------------------------
# stage 3: candidate table and row-level scorer per seed
# ----------------------------------------------------------------------------


def point_scorer(root: Path, detectors: dict[str, Path], xattn_pred: Path) -> None:
    rs.EXP = root
    rs.PRED = root / "predictions"
    rs.MET = root / "metrics"
    rs.CFG = root / "configs"
    rs.REPORT = root / "reports"
    rs.TRAIN = root / "training"
    rs.RUNS = rs.TRAIN / "runs"
    rs.SOURCE_MODELS = dict(detectors)
    rs.XATTN_PRED = xattn_pred
    rs.ensure_dirs()


def scorer_args(seed: int, device: str) -> SimpleNamespace:
    return SimpleNamespace(
        imgsz=640,
        pred_conf=0.001,
        predict_batch=8,
        device=device,
        max_rank=40,
        batch_size=32,
        quick=False,
        force_predict=False,
        rebuild_tables=False,
        seed=seed,
    )


def scorer_models(seed: int, quick: bool = False) -> dict[str, Any]:
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=2.0)),
        "hgb": HistGradientBoostingRegressor(
            max_iter=140 if quick else 260,
            learning_rate=0.04,
            max_leaf_nodes=31,
            l2_regularization=0.04,
            random_state=seed,
        ),
        "rf": RandomForestRegressor(
            n_estimators=140 if quick else 280,
            max_depth=13,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=seed,
        ),
        "extra": ExtraTreesRegressor(
            n_estimators=180 if quick else 360,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        ),
    }


def select_scorer(train_df: pd.DataFrame, val_df: pd.DataFrame, seed: int) -> tuple[str, dict[str, Any], pd.DataFrame]:
    """Same four-model, validation-selected procedure as the legacy scorer."""
    from sklearn.metrics import mean_squared_error

    x_train = rs.feature_matrix(train_df)
    y_train = train_df["target_iou"].to_numpy(float)
    x_val = rs.feature_matrix(val_df)
    fitted: dict[str, Any] = {}
    rows = []
    for name, model in scorer_models(seed).items():
        log(f"scorer fit {name}")
        model.fit(x_train, y_train)
        fitted[name] = model
        score = np.clip(model.predict(x_val), 0.0, 1.0)
        tmp = val_df.copy()
        tmp["score_head"] = score
        pred = rs.choose_by_score(tmp, "score_head", f"{name}_selector_val")
        row = rs.metrics(pred, f"{name}_selector", "all8")
        row["val_rmse"] = float(mean_squared_error(val_df["target_iou"].to_numpy(float), score) ** 0.5)
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("mean_iou", ascending=False)
    best = str(summary.iloc[0]["method"]).replace("_selector", "")
    return best, fitted, summary


def oof_score_head(train_df: pd.DataFrame, best_name: str, seed: int, n_splits: int = 5) -> np.ndarray:
    """Out-of-fold train scores so train/val/eval share the score_head schema."""
    from sklearn.base import clone
    from sklearn.model_selection import GroupKFold

    x_train = rs.feature_matrix(train_df)
    y_train = train_df["target_iou"].to_numpy(float)
    groups = train_df["subject_id"].astype(str).to_numpy()
    out = np.zeros(len(train_df), dtype=np.float64)
    template = scorer_models(seed)[best_name]
    for fold, (fit_idx, held_idx) in enumerate(GroupKFold(n_splits=n_splits).split(x_train, y_train, groups)):
        model = clone(template)
        model.fit(x_train[fit_idx], y_train[fit_idx])
        out[held_idx] = np.clip(model.predict(x_train[held_idx]), 0.0, 1.0)
        log(f"oof fold {fold + 1}/{n_splits} rows={len(held_idx)}")
    return out


def build_scored_tables(
    seed: int,
    root: Path,
    detectors: dict[str, Path],
    xattn_pred: Path,
    device: str,
    train_score_mode: str,
    legacy_cache: bool,
) -> dict[str, pd.DataFrame]:
    point_scorer(root, detectors, xattn_pred)
    if legacy_cache:
        # Reuse the archived YOLO candidate CSVs and feature pickles so the
        # rebuilt tables are bit-identical to the historical run.
        for path in LEGACY_SCORER_PRED.glob("*_conf0p001_candidates.csv"):
            shutil.copy2(path, rs.PRED / path.name)
        # The archived pickles predate the current pandas; rebuild them from
        # the identical CSV dump so build_candidate_table reuses them.
        for split in SPLITS:
            light = LEGACY_SCORER_PRED / f"{split}_row_candidates_light.csv"
            target = rs.PRED / f"{split}_row_candidates.pkl"
            if not target.exists():
                pd.read_csv(light).to_pickle(target)
    done = {split: rs.PRED / f"{split}_scored_candidates.csv" for split in SPLITS}
    if all(path.exists() for path in done.values()) and (rs.CFG / "scorer_selection.json").exists():
        log(f"seed {seed}: scored tables exist, skip")
        return {split: pd.read_csv(path) for split, path in done.items()}

    args = scorer_args(seed, device)
    rows = {split: rs.source_rows(split) for split in SPLITS}
    priors = rs.ybase.make_train_priors(rows["train"])
    tables: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        candidates = rs.candidate_groups(split, rows[split], args)
        tables[split] = rs.build_candidate_table(split, rows[split], candidates, priors, args)
        log(f"seed {seed}: candidate table {split} rows={len(tables[split])}")

    best_name, fitted, summary = select_scorer(tables["train"], tables["val"], seed)
    summary.to_csv(rs.MET / "candidate_scorer_val_model_selection.csv", index=False)
    best_model = fitted[best_name]
    import joblib

    joblib.dump(best_model, rs.RUNS / f"best_row_candidate_scorer_{best_name}.joblib")

    scored = {
        "val": rs.apply_model(tables["val"], best_model),
        "eval": rs.apply_model(tables["eval"], best_model),
    }
    train_table = tables["train"].copy()
    if train_score_mode == "oof":
        train_table["score_head"] = oof_score_head(train_table, best_name, seed)
    elif train_score_mode == "insample":
        train_table = rs.apply_model(train_table, best_model)
    elif train_score_mode == "none":
        pass
    else:
        raise ValueError(train_score_mode)
    scored["train"] = train_table
    for split, frame in scored.items():
        frame.to_csv(done[split], index=False)
    write_json(
        rs.CFG / "scorer_selection.json",
        {
            "seed": seed,
            "selected_model": best_name,
            "train_score_mode": train_score_mode,
            "candidate_sources": {tag: str(path) for tag, path in detectors.items()},
            "candidate_source_sha256": {tag: sha256(path) for tag, path in detectors.items()},
            "xattn_predictions": str(xattn_pred),
            "rows": {split: int(len(frame)) for split, frame in scored.items()},
        },
    )
    return scored


# ----------------------------------------------------------------------------
# stage 4: no-SigLIP auxiliary table and set parameters
# ----------------------------------------------------------------------------


def aux_table(frame: pd.DataFrame) -> pd.DataFrame:
    if "score_head" not in frame.columns:
        frame = frame.copy()
        frame["score_head"] = 0.0
    out = sf.select_top_candidates(frame, MAX_CANDIDATES_PER_TASK)
    out = out.drop(columns=[c for c in out.columns if c.lower().startswith("siglip")])
    out["no_siglip_score"] = numeric(out, "score_head") + 0.05 * numeric(out, "prior_iou") + 0.03 * numeric(out, "confidence")
    return out


def aux_predictions(
    frame: pd.DataFrame,
    groups: dict[str, dict[str, Any]],
    params: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    candidates = sf.scored_candidates_by_group(frame, groups, "no_siglip_score")
    return sf.predict_phrase_sets(groups, candidates, params)


# ----------------------------------------------------------------------------
# groups and hybrid context
# ----------------------------------------------------------------------------


def canonical_groups(protocol_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    inputs, labels, source_ids = multi_source.load_protocol(protocol_root)
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for split in SPLITS:
        groups: dict[str, dict[str, Any]] = {}
        for src in inputs[split]:
            gid = str(src["group_id"])
            groups[gid] = {
                "group_id": gid,
                "split": split,
                "dicom_id": str(src["dicom_id"]),
                "subject_id": str(src["subject_id"]),
                "study_id": str(src["study_id"]),
                "image_path": str(src["image_path"]),
                "finding": str(src["finding"]),
                "class_id": mb.CLASS_MAP[str(src["finding"])],
                "claim_sentence": multi_source.phrase_only(str(src["query_text"])),
                "image_width": int(src["image_width"]),
                "image_height": int(src["image_height"]),
                "task_ids": list(source_ids[split][gid]),
                "annotation_ids": [],
                "gt_boxes": [[float(x) for x in box] for box in labels[split][gid]],
            }
        out[split] = groups
    return out


def legacy_groups() -> dict[str, dict[str, dict[str, Any]]]:
    return {split: moe.sem2.gh.load_groups(split) for split in SPLITS}


def load_seed_hybrid(
    seed: int,
    protocol_root: Path,
    upstream_root: Path,
    base_root: Path,
    canonical_v3: bool,
) -> tuple[exact.ProtocolContext, dict[str, dict[str, list[list[float]]]]]:
    """``source.load_seed_hybrid`` with explicit roots instead of module defaults."""
    exact.PROTOCOL_ROOT = protocol_root
    exact.MULTI_SOURCE_ROOT = upstream_root
    context = exact.load_multi_context(seed, protocol_root, upstream_root, canonical_v3)
    source.add_train_artifacts(context)
    restored_multi_yolo = copy.deepcopy(context.yolo_params)
    run_root = base_root / "multibox_1444" / f"seed_{seed}"
    calibration = run_root / "single_route_fullval_calibration"
    context.yolo_params = json.loads((calibration / "yolo_params.json").read_text(encoding="utf-8"))
    context.fusion_params = json.loads((calibration / "fusion_params.json").read_text(encoding="utf-8"))
    set_params = json.loads((run_root / "selected_set_params.json").read_text(encoding="utf-8"))
    outputs = {}
    for split in SPLITS:
        outputs[split], _ = exact.run_hybrid(context, split, set_params, restored_multi_yolo)
    return context, outputs


# ----------------------------------------------------------------------------
# stage 5: gate
# ----------------------------------------------------------------------------


def run_gate(
    seed: int,
    seed_root: Path,
    groups_by_split: dict[str, dict[str, dict[str, Any]]],
    tables: dict[str, pd.DataFrame],
    set_params: dict[str, Any],
    context: exact.ProtocolContext,
    outputs: dict[str, dict[str, list[list[float]]]],
    device: str,
) -> dict[str, Any]:
    bundles: dict[str, moe.ExpertBundle] = {}
    for split in SPLITS:
        groups = groups_by_split[split]
        hybrid, cue = moe.fine_base.build_hybrid_v4(split, groups)
        aux = aux_predictions(tables[split], groups, set_params)
        bundle = moe.ExpertBundle(groups, hybrid, aux, {}, {}, cue)
        bundle.hybrid = transplant.remap_hybrid(context, split, outputs[split], bundle)
        bundles[split] = bundle

    gate_root = seed_root / "gate"
    moe.CKPT = gate_root / "checkpoints"
    moe.LOG = gate_root / "logs"
    moe.MET = gate_root / "training_metrics"
    moe.PRED = gate_root / "predictions"
    for path in (moe.CKPT, moe.LOG, moe.MET, moe.PRED):
        path.mkdir(parents=True, exist_ok=True)

    original_feature_builder = moe.gate_feature_for_group
    moe.gate_feature_for_group = source.finding_gate_feature
    try:
        # The slot name "siglip" is a fixed ExpertBundle field; it carries the
        # no-SigLIP auxiliary set.
        experts = ["hybrid", "siglip"]
        model, params, _, _ = moe.train_gate(
            f"finding_moe_no_siglip_s{seed}",
            bundles["train"],
            bundles["val"],
            experts,
            hardneg=False,
            seed=seed,
            device=device,
        )
        val_gate = source.predict_gate(model, bundles["val"], experts)
        eval_gate = source.predict_gate(model, bundles["eval"], experts)
        _, val_base = source.evaluate(f"base_s{seed}", bundles["val"], bundles["val"].hybrid, gate_root / "val" / "base")
        _, val_gate_summary = source.evaluate(f"no_siglip_gate_s{seed}", bundles["val"], val_gate, gate_root / "val" / "gate")
        _, val_aux = source.evaluate(f"aux_only_s{seed}", bundles["val"], bundles["val"].siglip, gate_root / "val" / "aux_only")
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
            f"selected_{selected_name}_s{seed}", bundles["eval"], selected, gate_root / "eval" / selected_name
        )
        # Diagnostics only: both arms on eval, never used for selection.
        _, eval_base = source.evaluate(f"base_s{seed}", bundles["eval"], bundles["eval"].hybrid, gate_root / "eval" / "base_diagnostic")
        _, eval_gate_summary = source.evaluate(f"gate_s{seed}", bundles["eval"], eval_gate, gate_root / "eval" / "gate_diagnostic")

        mutated = copy.deepcopy(bundles["eval"])
        mutated.groups = verify.mutate_gold(mutated.groups)
        mutated_predictions = source.predict_gate(model, mutated, experts)
        gold_independent = source.maps_equal(eval_gate, mutated_predictions)
    finally:
        moe.gate_feature_for_group = original_feature_builder

    return {
        "seed": seed,
        "selected_variant": selected_name,
        "gate_best_epoch": int(params["epoch"]),
        "train_gate_rows": int(params["train_rows"]),
        "val_gate_rows": int(params["val_rows"]),
        "val_base_coverage": float(val_base["coverage_mean_iou"]),
        "val_gate_coverage": float(val_gate_summary["coverage_mean_iou"]),
        "val_aux_only_coverage": float(val_aux["coverage_mean_iou"]),
        "coverage_mean_iou": float(eval_summary["coverage_mean_iou"]),
        "exact_union_iou": float(eval_summary["exact_rectangle_union_iou"]),
        "set_f1_0_3": float(eval_summary["set_f1_0_3"]),
        "set_f1_0_5": float(eval_summary["set_f1_0_5"]),
        "mean_pred_count": float(eval_summary["mean_pred_count"]),
        "eval_base_coverage_diagnostic": float(eval_base["coverage_mean_iou"]),
        "eval_gate_coverage_diagnostic": float(eval_gate_summary["coverage_mean_iou"]),
        "n_eval_groups": int(len(bundles["eval"].groups)),
        "n_eval_gt_boxes": int(sum(len(g["gt_boxes"]) for g in bundles["eval"].groups.values())),
        "gold_mutation_independence": bool(gold_independent),
    }


# ----------------------------------------------------------------------------
# modes
# ----------------------------------------------------------------------------


def preflight(mode: str, seeds: list[int]) -> None:
    needed = [yd.DATASET / "ms_cxr_yolo.yaml", LEGACY_XATTN_CKPT, LEGACY_SET_PARAMS, LEGACY_RESULT]
    needed += list(DETECTORS.values())
    if mode == "legacy_assets_check":
        needed += [LEGACY_DETECTOR_RUNS / f"{tag}_full" / "weights" / "best.pt" for tag in DETECTORS]
        needed += [LEGACY_XATTN_PRED / f"row1444_rule_plus_full_{split}_predictions.jsonl" for split in SPLITS]
        needed += [LEGACY_SEMANTIC_PRED / "train_siglip_scored_candidates_cxr_claim_m0p15.csv"]
        needed += [LEGACY_SCORER_PRED / f"{split}_row_candidates.pkl" for split in SPLITS]
        for seed in SEEDS:
            needed += [LEGACY_BASE_ROOT / "multibox_1444" / f"seed_{seed}" / "selected_set_params.json"]
    else:
        needed += [CANONICAL_PROTOCOL_ROOT / "mscxr_multibox_1444" / f"{split}_inputs.jsonl" for split in SPLITS]
        for seed in seeds:
            upstream = CANONICAL_UPSTREAM_ROOT / f"seed_{seed}" / "mscxr_multibox_1444"
            needed += [upstream / "rad_dino_legacy" / "predictions_by_split.npz", upstream / "legacy_fusion" / "yolo_params.json"]
            needed += [upstream / "yolo_predictions" / f"yolov8s_{split}.csv" for split in SPLITS]
            base = CANONICAL_BASE_ROOTS[seed] / "multibox_1444" / f"seed_{seed}"
            needed += [
                base / "selected_set_params.json",
                base / "single_route_fullval_calibration" / "yolo_params.json",
                base / "single_route_fullval_calibration" / "fusion_params.json",
            ]
    require(needed)


def aggregate(rows: list[dict[str, Any]], output_root: Path, extra: dict[str, Any]) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "per_seed_metrics.csv", index=False)
    metrics = ("coverage_mean_iou", "exact_union_iou", "set_f1_0_3", "set_f1_0_5", "mean_pred_count")
    status: dict[str, Any] = {
        "status": "complete",
        "seeds": frame["seed"].tolist(),
        "selected_variants": frame["selected_variant"].tolist(),
        "gold_mutation_independence_pass": bool(frame["gold_mutation_independence"].all()),
        "semantic_score_formula": SCORE_FORMULA,
        **extra,
    }
    for metric in metrics:
        status[f"{metric}_mean"] = float(frame[metric].mean())
        status[f"{metric}_std"] = float(frame[metric].std(ddof=1)) if len(frame) > 1 else 0.0
    write_json(output_root / "FINAL_STATUS.json", status)
    log(f"FINAL coverage {status['coverage_mean_iou_mean']:.6f} +/- {status['coverage_mean_iou_std']:.6f}")


def run_legacy_assets_check(output_root: Path, device: str, yolo_device: str) -> None:
    """Rebuild the seed-42 auxiliary table with the new code, then reproduce."""
    root = output_root / "legacy_assets_check"
    root.mkdir(parents=True, exist_ok=True)
    detectors = {tag: LEGACY_DETECTOR_RUNS / f"{tag}_full" / "weights" / "best.pt" for tag in DETECTORS}
    scored = build_scored_tables(
        42, root / "aux_seed42" / "scorer", detectors, LEGACY_XATTN_PRED, yolo_device, "none", legacy_cache=True
    )
    selection = json.loads((root / "aux_seed42" / "scorer" / "configs" / "scorer_selection.json").read_text(encoding="utf-8"))

    # Unit check 1: score_head must match the archived scorer output.
    checks: dict[str, Any] = {"selected_model": selection["selected_model"]}
    for split in ("val", "eval"):
        legacy = pd.read_csv(LEGACY_SCORER_PRED / f"{split}_scored_candidates.csv")
        merged = legacy[["candidate_id", "score_head"]].merge(
            scored[split][["candidate_id", "score_head"]], on="candidate_id", suffixes=("_legacy", "_new")
        )
        diff = float(np.abs(merged["score_head_legacy"] - merged["score_head_new"]).max()) if len(merged) else float("nan")
        checks[f"{split}_rows_legacy"] = int(len(legacy))
        checks[f"{split}_rows_new"] = int(len(scored[split]))
        checks[f"{split}_matched"] = int(len(merged))
        checks[f"{split}_max_abs_score_head_diff"] = diff
        log(f"check {split}: matched={len(merged)}/{len(legacy)} max|dscore_head|={diff:.3e}")

    # Unit check 2: the top-12 auxiliary table must match the archived one.
    tables: dict[str, pd.DataFrame] = {}
    table_root = root / "aux_seed42" / "aux_table"
    table_root.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        legacy = pd.read_csv(LEGACY_SEMANTIC_PRED / f"{split}_siglip_scored_candidates_cxr_claim_m0p15.csv")
        legacy = legacy.drop(columns=[c for c in legacy.columns if c.lower().startswith("siglip")])
        if split == "train":
            # The archived train table has no score_head; mirror it exactly.
            table = legacy.copy()
            table["no_siglip_score"] = numeric(table, "score_head") + 0.05 * numeric(table, "prior_iou") + 0.03 * numeric(table, "confidence")
        else:
            table = aux_table(scored[split])
            same_ids = set(table["candidate_id"]) == set(legacy["candidate_id"])
            checks[f"{split}_top12_candidate_ids_equal"] = bool(same_ids)
            log(f"check {split}: top-12 candidate ids equal={same_ids}")
        table.to_csv(table_root / f"{split}_no_siglip_candidates.csv", index=False)
        tables[split] = table
    set_params = json.loads(LEGACY_SET_PARAMS.read_text(encoding="utf-8"))
    write_json(table_root / "best_set_params.json", set_params)

    groups_by_split = legacy_groups()
    rows = []
    for seed in SEEDS:
        log(f"seed {seed}: legacy hybrid + gate")
        context, outputs = load_seed_hybrid(seed, LEGACY_PROTOCOL_ROOT, LEGACY_UPSTREAM_ROOT, LEGACY_BASE_ROOT, False)
        row = run_gate(seed, root / f"seed_{seed}", groups_by_split, tables, set_params, context, outputs, device)
        write_json(root / f"seed_{seed}" / "RUN_STATUS.json", row)
        rows.append(row)
        log(f"seed {seed}: {row['selected_variant']} coverage={row['coverage_mean_iou']:.6f}")

    legacy_rows = pd.read_csv(LEGACY_RESULT)
    frame = pd.DataFrame(rows)
    compare = {}
    for column in ("coverage_mean_iou", "exact_union_iou", "set_f1_0_3", "set_f1_0_5", "gate_best_epoch"):
        compare[column] = float(np.abs(frame[column].to_numpy(float) - legacy_rows[column].to_numpy(float)).max())
    checks["per_seed_max_abs_diff_vs_legacy"] = compare
    checks["reproduced"] = bool(max(compare.values()) < 1e-9)
    write_json(root / "CHECKS.json", checks)
    aggregate(rows, root, {"mode": "legacy_assets_check", "checks": checks})
    log(f"legacy reproduction: {checks['reproduced']} ({compare})")


def run_canonical(output_root: Path, seeds: list[int], device: str, yolo_device: str, epochs: int, train_score_mode: str) -> None:
    root = output_root / "canonical"
    root.mkdir(parents=True, exist_ok=True)
    groups_by_split = canonical_groups(CANONICAL_PROTOCOL_ROOT)
    write_json(
        root / "GROUP_COUNTS.json",
        {
            split: {"groups": len(groups), "gt_boxes": sum(len(g["gt_boxes"]) for g in groups.values())}
            for split, groups in groups_by_split.items()
        },
    )
    rows = []
    for seed in seeds:
        seed_root = root / f"seed_{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        started = time.time()
        detectors = {tag: train_detector(seed, tag, seed_root / "detectors", yolo_device, epochs) for tag in DETECTORS}
        train_xattn(seed, seed_root / "xattn")
        scored = build_scored_tables(
            seed, seed_root / "scorer", detectors, seed_root / "xattn" / "predictions", yolo_device, train_score_mode, legacy_cache=False
        )
        tables = {split: aux_table(frame) for split, frame in scored.items()}
        table_root = seed_root / "aux_table"
        table_root.mkdir(parents=True, exist_ok=True)
        for split, table in tables.items():
            table.to_csv(table_root / f"{split}_no_siglip_candidates.csv", index=False)
        schema = {split: sorted(table.columns) for split, table in tables.items()}
        if not (schema["train"] == schema["val"] == schema["eval"]):
            raise RuntimeError("candidate table schema differs across splits")
        set_params, grid = sf.tune_set_params(groups_by_split["val"], tables["val"], "no_siglip_score")
        grid.to_csv(table_root / "set_param_val_grid.csv", index=False)
        write_json(table_root / "best_set_params.json", set_params)
        log(f"seed {seed}: set params {set_params}")

        context, outputs = load_seed_hybrid(
            seed, CANONICAL_PROTOCOL_ROOT, CANONICAL_UPSTREAM_ROOT, CANONICAL_BASE_ROOTS[seed], True
        )
        row = run_gate(seed, seed_root, groups_by_split, tables, set_params, context, outputs, device)
        row["set_params"] = set_params
        row["elapsed_sec"] = round(time.time() - started, 1)
        row["provenance"] = {
            "detectors": {tag: {"path": str(path), "sha256": sha256(path)} for tag, path in detectors.items()},
            "xattn_checkpoint": str(seed_root / "xattn" / "checkpoints" / f"row1444_rule_plus_full_s{seed}_best.pt"),
            "scorer_selection": json.loads((seed_root / "scorer" / "configs" / "scorer_selection.json").read_text(encoding="utf-8")),
            "hybrid_upstream": str(CANONICAL_UPSTREAM_ROOT / f"seed_{seed}"),
            "hybrid_calibration": str(CANONICAL_BASE_ROOTS[seed] / "multibox_1444" / f"seed_{seed}"),
            "protocol_root": str(CANONICAL_PROTOCOL_ROOT),
            "canonical_membership": context.provenance.get("canonical_membership") if hasattr(context, "provenance") else None,
        }
        write_json(seed_root / "RUN_STATUS.json", row)
        rows.append(row)
        log(f"seed {seed}: {row['selected_variant']} coverage={row['coverage_mean_iou']:.6f} ({row['elapsed_sec']} s)")
        pd.DataFrame(rows).drop(columns=["provenance", "set_params"]).to_csv(root / "per_seed_metrics_partial.csv", index=False)

    aggregate(
        [{k: v for k, v in row.items() if k not in ("provenance", "set_params")} for row in rows],
        root,
        {
            "mode": "canonical",
            "protocol": "canonical 813/124/220 phrase groups, 996/164/280 GT boxes",
            "per_seed_regenerated": ["yolov8n", "yolov8s", "xattn head", "row-level scorer", "candidate table", "set params", "gate"],
            "train_score_mode": train_score_mode,
            "siglip_used": False,
            "biomedclip_used": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("legacy_assets_check", "canonical"), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--yolo-device", default="0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train-score-mode", choices=("oof", "insample", "none"), default="oof")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    preflight(args.mode, args.seeds)
    write_json(output_root / f"RUN_ARGS_{args.mode}.json", vars(args))
    if args.mode == "legacy_assets_check":
        run_legacy_assets_check(output_root, args.device, args.yolo_device)
    else:
        run_canonical(output_root, args.seeds, args.device, args.yolo_device, args.epochs, args.train_score_mode)


if __name__ == "__main__":
    main()

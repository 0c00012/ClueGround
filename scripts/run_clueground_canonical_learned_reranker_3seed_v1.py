#!/usr/bin/env python
"""Learned candidate re-ranker plugged into the canonical ClueGround decoder.

Diagnosis (2026-09-09): the four-YOLO candidate pool already contains a box
with high IoU for almost every GT region (per-GT oracle 0.74 single / 0.66
multi), but the hand-weighted confidence ranking selects a wrong candidate
twice as often as MedGrounder does.  This runner trains, per seed, a tabular
candidate scorer on canonical train813 (candidate-vs-GT IoU regression) and
uses it to re-rank the YOLO candidates before the unchanged validation-
calibrated hybrid fusion and rule-context decoder (``exact.run_protocol``).

Per-candidate features combine the detector evidence (confidence, rank,
source), geometry, agreement with three independent phrase-conditioned
estimates (RAD-DINO box, cross-attention context head box, light-detector
auxiliary top box), cross-detector consensus, the training-derived spatial
prior, and the deterministic phrase context.  No eval field is used anywhere;
scorer family is chosen by 5-fold subject-grouped CV on train, the re-rank
strength alpha and all fusion/decoder parameters on val124 only.

Variants per seed:
  * ``learned``            re-ranked candidates -> unchanged decoder
  * ``learned_heat``       + the sealed per-seed RAD-DINO candidate heat
  * ``learned_heat_slot``  + explicit anatomic-slot phrase parser
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import functools

import numpy as np
import pandas as pd
import torch

# Candidate tables are re-read from CSV by the ablation/OOF runners.  The default
# pandas float parser is not round-trip exact, and ExtraTrees thresholds are
# sensitive to 1-ulp changes, so force the round-trip parser process-wide.
pd.read_csv = functools.partial(pd.read_csv, float_precision="round_trip")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_clueground_canonical_explicit_anatomic_slot_parser_pilot_v1 as slots  # noqa: E402
from scripts import run_clueground_canonical_heat_slot_parser_combo_3seed_v1 as combo  # noqa: E402
from scripts import run_clueground_canonical_heatmap_candidate_rank_pilot_v1 as heat  # noqa: E402
from scripts import run_clueground_canonical_neural_agreement_selector_pilot_v1 as canonical  # noqa: E402
from scripts import run_clueground_exact_hybrid_v4_direct_3seed_v1 as exact  # noqa: E402
from scripts import run_clueground_vfm_legacy_fusion_unified_3seed_v1 as legacy  # noqa: E402
from scripts import run_ms_cxr_multibox_rule_context_fusion_v2 as mv2  # noqa: E402
from scripts import run_ms_cxr_multibox_rule_context_fusion_v4_yolov8l_pool as hybrid_v4  # noqa: E402
from scripts import run_ms_cxr_yolo_detector_v1 as yd  # noqa: E402
from scripts import run_ms_cxr_yolo_rule_context_v1 as ybase  # noqa: E402
from scripts import run_ms_cxr_yolo_rule_context_v2 as yv2  # noqa: E402
from scripts.models_ms_cxr_vfm_localizer import PatchHeatmapBBoxHead  # noqa: E402

SEEDS = (13, 42, 2026)
SPLITS = ("train", "val", "eval")
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "clueground_canonical_learned_reranker_3seed_v1"
AUX_ROOT = PROJECT_ROOT / "experiments" / "clueground_canonical_aux_chain_v1" / "canonical"
SOURCES = ("yolov8s", "yolov8m", "yolo11s", "yolo11m")
LATS = ["left", "right", "bilateral", "none", "unknown"]
VERTS = ["apical", "upper", "mid", "lower", "basal", "whole", "unknown"]
MAX_RANK = 40
ALPHA_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
PRIOR_PARAMS = {"side_mode": "radiology_right", "prior_train_weight": 0.75}
ALPHA_SELECTION = "val"
NO_AUX_ASSETS = False
MULTI_ALPHA = "same"
NO_RETUNE = False
AUDIT_GOLD_MUTATION = False


def log(message: str) -> None:
    print(f"[reranker {time.strftime('%H:%M:%S')}] {message}", flush=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def iou_xyxy(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def one_hot(value: str, options: list[str]) -> list[float]:
    return [1.0 if value == option else 0.0 for option in options]


# ----------------------------------------------------------------------------
# independent phrase-conditioned estimates per group (pixel xyxy)
# ----------------------------------------------------------------------------


def group_boxes_from_task_norm(
    task_boxes: dict[str, np.ndarray], source_ids: dict[str, list[str]], rows: list[dict[str, Any]]
) -> dict[str, list[float]]:
    out = {}
    for row in rows:
        gid = str(row["group_id"])
        available = [task_boxes[t] for t in source_ids[gid] if t in task_boxes]
        if not available:
            continue
        norm = np.mean(np.stack(available), axis=0)
        out[gid] = ybase.norm_to_xyxy(norm, float(row["image_width"]), float(row["image_height"]))
    return out


def load_dino_boxes(seed: int, split: str, source_ids: dict, rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    archive = np.load(
        canonical.UPSTREAM_ROOT / f"seed_{seed}" / canonical.PROTOCOL / "rad_dino_legacy" / "predictions_by_split.npz",
        allow_pickle=True,
    )
    task_boxes = {str(t): np.asarray(b, dtype=np.float32) for t, b in zip(archive[f"{split}_ids"], archive[f"{split}_boxes"])}
    return group_boxes_from_task_norm(task_boxes, source_ids, rows)


def load_xattn_boxes(seed: int, split: str, source_ids: dict, rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    path = AUX_ROOT / f"seed_{seed}" / "xattn" / "predictions" / f"row1444_rule_plus_full_{split}_predictions.jsonl"
    task_boxes = {str(r["task_id"]): np.asarray(r["pred_bbox_norm_cxcywh"], dtype=np.float32) for r in read_jsonl(path)}
    return group_boxes_from_task_norm(task_boxes, source_ids, rows)


def load_aux_top_boxes(seed: int, split: str, source_ids: dict, rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    frame = pd.read_csv(AUX_ROOT / f"seed_{seed}" / "aux_table" / f"{split}_no_siglip_candidates.csv")
    best: dict[str, tuple[float, list[float]]] = {}
    for r in frame.itertuples():
        task = str(r.task_id)
        score = float(r.no_siglip_score)
        if task not in best or score > best[task][0]:
            best[task] = (score, [float(r.pred_x1), float(r.pred_y1), float(r.pred_x2), float(r.pred_y2)])
    out = {}
    for row in rows:
        gid = str(row["group_id"])
        hits = [best[t] for t in source_ids[gid] if t in best]
        if hits:
            out[gid] = max(hits, key=lambda x: x[0])[1]
    return out


# ----------------------------------------------------------------------------
# candidate feature table
# ----------------------------------------------------------------------------


def candidate_key(candidate: dict[str, Any]) -> str:
    return f"{candidate['source_model']}::{int(candidate['rank'])}"


def build_table(
    split: str,
    rows: list[dict[str, Any]],
    labels: dict[str, list[list[float]]],
    candidates: dict[str, list[dict[str, Any]]],
    priors: dict[str, Any],
    estimates: dict[str, dict[str, list[float]]],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        gid = str(row["group_id"])
        class_id = int(yd.CLASS_TO_ID[str(row["finding"])])
        iw, ih = float(row["image_width"]), float(row["image_height"])
        pool = [
            c for c in candidates.get(str(row["dicom_id"]), [])
            if int(c["class_id"]) == class_id and int(c.get("rank", 9999)) < MAX_RANK
        ]
        if not pool:
            continue
        q = ybase.parse_rule_context(row)
        cue = mv2.context_cues(str(row["claim_sentence"]), str(row["finding"]), q)
        _, prior = yv2.target_prior_for_row(row, priors, PRIOR_PARAMS)
        prior_area = max(1e-6, float(prior[2] * prior[3]))
        gold = labels[gid]
        dino = estimates["dino"].get(gid)
        xattn = estimates["xattn"].get(gid)
        aux = estimates["aux"].get(gid)
        n_pool = len(pool)
        by_source = defaultdict(list)
        for c in pool:
            by_source[str(c["source_model"])].append(c["box"])
        for c in pool:
            box = [float(v) for v in c["box"]]
            box_norm = ybase.xyxy_to_norm(box, iw, ih)
            area = max(1e-6, float(box_norm[2] * box_norm[3]))
            others = [iou_xyxy(box, other) for src, boxes in by_source.items() if src != str(c["source_model"]) for other in boxes]
            same_rank0 = [iou_xyxy(box, boxes[0]) for src, boxes in by_source.items() if src != str(c["source_model"]) and boxes]
            record = {
                "split": split,
                "group_id": gid,
                "subject_id": str(row["subject_id"]),
                "dicom_id": str(row["dicom_id"]),
                "finding": str(row["finding"]),
                "candidate_key": candidate_key(c),
                "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                "cx": float(box_norm[0]), "cy": float(box_norm[1]), "w": float(box_norm[2]), "h": float(box_norm[3]),
                "area": area, "log_area": math.log(area), "aspect": float(box_norm[2] / max(1e-6, box_norm[3])),
                "conf": float(c["score"]), "log_conf": math.log1p(20.0 * max(0.0, float(c["score"]))),
                "rank": float(c["rank"]), "rank_bonus": 1.0 / (1.0 + float(c["rank"])),
                "n_pool": float(n_pool),
                "dino_iou": iou_xyxy(box, dino) if dino else 0.0,
                "xattn_iou": iou_xyxy(box, xattn) if xattn else 0.0,
                "aux_iou": iou_xyxy(box, aux) if aux else 0.0,
                "consensus_max": max(others) if others else 0.0,
                "consensus_mean": float(np.mean(others)) if others else 0.0,
                "consensus_top_mean": float(np.mean(same_rank0)) if same_rank0 else 0.0,
                "prior_iou": float(ybase.iou_norm(box_norm, prior)),
                "area_penalty": abs(math.log(area / prior_area)),
                "region_score": float(yv2.center_region_score_v2(box_norm, q, "radiology_right")),
                "has_multi_cue": float(bool(cue["has_multi_cue"])),
                "k_hint": float(cue.get("k_hint", 1)),
                "target_iou": max(iou_xyxy(box, g) for g in gold),
            }
            for name, value in zip([f"src_{s}" for s in SOURCES], one_hot(str(c["source_model"]), list(SOURCES))):
                record[name] = value
            for name, value in zip([f"lat_{v}" for v in LATS], one_hot(q.get("laterality", "unknown"), LATS)):
                record[name] = value
            for name, value in zip([f"vert_{v}" for v in VERTS], one_hot(q.get("vertical", "unknown"), VERTS)):
                record[name] = value
            for name, value in zip([f"find_{k}" for k in yd.CLASS_NAMES], one_hot(str(row["finding"]), list(yd.CLASS_NAMES))):
                record[name] = value
            records.append(record)
    return pd.DataFrame(records)


FEATURE_EXCLUDE = {"split", "group_id", "subject_id", "dicom_id", "finding", "candidate_key", "x1", "y1", "x2", "y2", "target_iou"}


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in FEATURE_EXCLUDE]


# ----------------------------------------------------------------------------
# scorer family selection by grouped CV on train
# ----------------------------------------------------------------------------


def scorer_models(seed: int) -> dict[str, Any]:
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=2.0)),
        "hgb": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=31, l2_regularization=0.05, random_state=seed),
        "rf": RandomForestRegressor(n_estimators=300, max_depth=14, min_samples_leaf=3, n_jobs=-1, random_state=seed),
        "extra": ExtraTreesRegressor(n_estimators=400, max_depth=None, min_samples_leaf=2, n_jobs=-1, random_state=seed),
    }


def top1_iou(frame: pd.DataFrame, score: np.ndarray) -> float:
    tmp = frame[["group_id", "target_iou"]].copy()
    tmp["score"] = score
    idx = tmp.groupby("group_id")["score"].idxmax()
    return float(tmp.loc[idx, "target_iou"].mean())


def select_scorer(train: pd.DataFrame, seed: int, n_splits: int = 5) -> tuple[str, pd.DataFrame, dict[str, np.ndarray]]:
    from sklearn.base import clone
    from sklearn.model_selection import GroupKFold

    cols = feature_columns(train)
    x = train[cols].to_numpy(np.float32)
    y = train["target_iou"].to_numpy(float)
    groups = train["subject_id"].astype(str).to_numpy()
    folds = list(GroupKFold(n_splits=n_splits).split(x, y, groups))
    rows = []
    oof_by_model: dict[str, np.ndarray] = {}
    for name, template in scorer_models(seed).items():
        oof = np.zeros(len(train), dtype=np.float64)
        for fit_idx, held_idx in folds:
            model = clone(template)
            model.fit(x[fit_idx], y[fit_idx])
            oof[held_idx] = np.clip(model.predict(x[held_idx]), 0.0, 1.0)
        oof_by_model[name] = oof
        rows.append({"model": name, "cv_top1_iou": top1_iou(train, oof), "cv_rmse": float(np.sqrt(np.mean((oof - y) ** 2)))})
        log(f"cv {name}: top1 IoU {rows[-1]['cv_top1_iou']:.4f}")
    table = pd.DataFrame(rows).sort_values("cv_top1_iou", ascending=False)
    baseline = top1_iou(train, train["conf"].to_numpy(float))
    table["cv_top1_iou_confidence_baseline"] = baseline
    return str(table.iloc[0]["model"]), table, oof_by_model


# ----------------------------------------------------------------------------
# plug-in: adjust YOLO candidate scores with the learned score
# ----------------------------------------------------------------------------


def adjust_with_learned(
    candidates: dict[str, list[dict[str, Any]]],
    scored: pd.DataFrame,
    alpha: float,
    score_col: str = "learned",
    skip_keys: set[tuple[str, int]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    lookup: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    class_by_finding = {name: int(idx) for name, idx in yd.CLASS_TO_ID.items()}
    for r in scored.itertuples():
        lookup[(str(r.dicom_id), class_by_finding[str(r.finding)], str(r.candidate_key))].append(float(getattr(r, score_col)))
    output = copy.deepcopy(candidates)
    for dicom, values in output.items():
        by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for candidate in values:
            by_class[int(candidate["class_id"])].append(candidate)
        for class_id, subset in by_class.items():
            if skip_keys and (str(dicom), int(class_id)) in skip_keys:
                continue
            learned = np.asarray(
                [np.mean(lookup.get((str(dicom), class_id, candidate_key(c)), [np.nan])) for c in subset], dtype=np.float64
            )
            if np.all(np.isnan(learned)):
                continue
            fill = float(np.nanmin(learned))
            learned = np.where(np.isnan(learned), fill, learned)
            z = (learned - learned.mean()) / max(float(learned.std()), 1e-6)
            for candidate, value, zed in zip(subset, learned, z):
                raw = min(1.0 - 1e-5, max(1e-5, float(candidate.get("raw_yolo_score", candidate["score"]))))
                base_logit = math.log(raw / (1.0 - raw))
                extra = float(candidate.get("_heat_term", 0.0))
                candidate["raw_yolo_score"] = raw
                candidate["learned_score"] = float(value)
                candidate["learned_z"] = float(zed)
                candidate["score"] = float(1.0 / (1.0 + math.exp(-(base_logit + extra + alpha * float(zed)))))
    return output


def multi_cue_keys(rows: list[dict[str, Any]]) -> set[tuple[str, int]]:
    keys = set()
    for row in rows:
        q = ybase.parse_rule_context(row)
        cue = mv2.context_cues(str(row["claim_sentence"]), str(row["finding"]), q)
        if cue["has_multi_cue"]:
            keys.add((str(row["dicom_id"]), int(yd.CLASS_TO_ID[str(row["finding"])])))
    return keys


def val_top_candidate_iou(rows: list[dict[str, Any]], labels: dict, candidates: dict[str, list[dict[str, Any]]]) -> float:
    values = []
    for row in rows:
        class_id = int(yd.CLASS_TO_ID[str(row["finding"])])
        pool = [c for c in candidates.get(str(row["dicom_id"]), []) if int(c["class_id"]) == class_id and int(c.get("rank", 9999)) < MAX_RANK]
        if not pool:
            values.append(0.0)
            continue
        best = max(pool, key=lambda c: float(c["score"]))
        values.append(max(iou_xyxy([float(v) for v in best["box"]], g) for g in labels[str(row["group_id"])]))
    return float(np.mean(values))


def apply_heat_terms(seed: int, rows: dict, labels: dict, source_ids: dict, candidates: dict) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Return candidates whose scores carry the sealed heat adjustment and a
    stored ``_heat_term`` so the learned re-rank can be added on top."""
    source_root = combo.HEAT_SOURCES[seed]
    alpha_heat = float(json.loads((source_root / "RUN_STATUS.json").read_text(encoding="utf-8"))["selected_heat_alpha"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    allowed = {split: {str(t) for ts in source_ids[split].values() for t in ts} for split in ("val", "eval")}
    bundles = {split: legacy.load_rad_bundle(split, allowed[split]) for split in ("val", "eval")}
    data = {split: heat.build_group_data(rows[split], labels[split], source_ids[split], bundles[split], candidates[split]) for split in ("val", "eval")}
    payload = torch.load(source_root / "best.pt", map_location="cpu", weights_only=False)
    model = PatchHeatmapBBoxHead(data["val"][0].shape[-1], data["val"][1].shape[-1], hidden=384, dropout=0.1).to(device)
    model.load_state_dict(payload["state"])
    heat_scores = {split: heat.heat_by_group(model, data[split], 12, device, "mean", 0.25) for split in ("val", "eval")}
    adjusted = {split: heat.adjust_candidates(rows[split], heat_scores[split], candidates[split], alpha_heat) for split in ("val", "eval")}
    for split in adjusted:
        for values in adjusted[split].values():
            for candidate in values:
                if "dino_candidate_heat_z" in candidate:
                    candidate["_heat_term"] = alpha_heat * float(candidate["dino_candidate_heat_z"])
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return adjusted


def run_variant(
    seed: int,
    variant: str,
    output_root: Path,
    context: exact.ProtocolContext,
    base_candidates: dict[str, dict[str, list[dict[str, Any]]]],
    scored: dict[str, pd.DataFrame],
    rows: dict,
    labels: dict,
    source_ids: dict,
) -> dict[str, Any]:
    candidates = {split: copy.deepcopy(base_candidates[split]) for split in ("val", "eval")}
    if variant in ("learned_heat", "learned_heat_slot"):
        candidates = apply_heat_terms(seed, rows, labels, source_ids, candidates)
    skip = {split: (multi_cue_keys(rows[split]) if MULTI_ALPHA == "zero" else None) for split in SPLITS}
    grid_rows = []
    for alpha in ALPHA_GRID:
        adjusted_val = adjust_with_learned(candidates["val"], scored["val"], alpha, skip_keys=skip["val"])
        val_metric = val_top_candidate_iou(rows["val"], labels["val"], adjusted_val)
        record = {"alpha": alpha, "val_top_candidate_iou": val_metric}
        if ALPHA_SELECTION == "train_oof_val":
            # Train candidates carry out-of-fold learned scores; pooling the
            # train and val top-candidate IoU gives an eval-free, larger-sample
            # selection signal than val124 alone.
            adjusted_train = adjust_with_learned(base_candidates["train"], scored["train"], alpha, skip_keys=skip["train"])
            train_metric = val_top_candidate_iou(rows["train"], labels["train"], adjusted_train)
            n_train, n_val = len(rows["train"]), len(rows["val"])
            record["train_oof_top_candidate_iou"] = train_metric
            record["pooled_top_candidate_iou"] = (train_metric * n_train + val_metric * n_val) / (n_train + n_val)
        grid_rows.append(record)
    key = "pooled_top_candidate_iou" if ALPHA_SELECTION == "train_oof_val" else "val_top_candidate_iou"
    grid = pd.DataFrame(grid_rows).sort_values([key, "alpha"], ascending=[False, True])
    alpha = float(grid.iloc[0]["alpha"])
    variant_root = output_root / variant
    variant_root.mkdir(parents=True, exist_ok=True)
    grid.to_csv(variant_root / f"alpha_val_grid_s{seed}.csv", index=False)
    log(f"seed {seed} {variant}: alpha={alpha} val top-candidate IoU {grid.iloc[0]['val_top_candidate_iou']:.4f} (alpha 0: {grid[grid.alpha == 0].val_top_candidate_iou.iloc[0]:.4f})")
    run_context = copy.copy(context)
    run_context.candidates = {split: adjust_with_learned(candidates[split], scored[split], alpha, skip_keys=skip[split]) for split in ("val", "eval")}
    original = hybrid_v4.context_cues
    if variant == "learned_heat_slot":
        hybrid_v4.context_cues = slots.explicit_slot_context(original)
    try:
        result = exact.run_protocol(run_context, variant_root, quick=False, retune_single_full_val=not NO_RETUNE, separate_multi_route_params=True, calibration_cache_root=None)
    finally:
        hybrid_v4.context_cues = original
    run_root = variant_root / "multibox_1444" / f"seed_{seed}"
    try:
        grid_table = pd.read_csv(run_root / "multibox_v4_val_grid.csv")
        set_params = json.loads((run_root / "selected_set_params.json").read_text(encoding="utf-8"))
        mask = np.ones(len(grid_table), dtype=bool)
        for key, value in set_params.items():
            if key in grid_table.columns:
                mask &= np.isclose(grid_table[key].astype(float), float(value))
        picked = grid_table[mask]
        result["val_selected_coverage"] = float(picked["val_all_coverage_mean_iou"].iloc[0]) if len(picked) else float("nan")
        result["val_selected_set_f1_0_3"] = float(picked["val_all_set_f1_0_3"].iloc[0]) if len(picked) else float("nan")
    except Exception as exc:  # diagnostics only
        result["val_selected_coverage"] = float("nan")
        result["val_grid_error"] = str(exc)
    result.update({"variant": variant, "seed": seed, "learned_alpha": alpha, "multi_alpha": MULTI_ALPHA, "retune_on_val": not NO_RETUNE, "no_aux_assets": NO_AUX_ASSETS})
    write_json(variant_root / f"RUN_STATUS_s{seed}.json", result)
    return result


def run_seed(seed: int, output_root: Path, variants: list[str]) -> list[dict[str, Any]]:
    seed_root = output_root / f"seed_{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    inputs, labels, source_ids = legacy.load_protocol(canonical.PROTOCOL_ROOT)
    rows = {split: legacy.group_rows(inputs[split], labels[split], split) for split in SPLITS}
    upstream = canonical.UPSTREAM_ROOT / f"seed_{seed}" / canonical.PROTOCOL
    candidates = {split: legacy.load_yolo_candidates(upstream / "yolo_predictions", split) for split in SPLITS}
    priors = ybase.make_train_priors(legacy.expanded_prior_rows(rows["train"]))
    estimates = {
        split: {
            "dino": load_dino_boxes(seed, split, source_ids[split], rows[split]),
            "xattn": {} if NO_AUX_ASSETS else load_xattn_boxes(seed, split, source_ids[split], rows[split]),
            "aux": {} if NO_AUX_ASSETS else load_aux_top_boxes(seed, split, source_ids[split], rows[split]),
        }
        for split in SPLITS
    }
    tables = {}
    for split in SPLITS:
        cache = seed_root / f"candidates_{split}.csv"
        if cache.exists():
            tables[split] = pd.read_csv(cache)
        else:
            tables[split] = build_table(split, rows[split], labels[split], candidates[split], priors, estimates[split])
            tables[split].to_csv(cache, index=False)
        log(f"seed {seed}: table {split} rows={len(tables[split])} groups={tables[split].group_id.nunique()}")

    best_name, cv_table, oof = select_scorer(tables["train"], seed)
    cv_table.to_csv(seed_root / "scorer_cv_selection.csv", index=False)
    cols = feature_columns(tables["train"])
    assert not any("target" in c or "gold" in c for c in cols), cols
    model = scorer_models(seed)[best_name]
    model.fit(tables["train"][cols].to_numpy(np.float32), tables["train"]["target_iou"].to_numpy(float))
    scored = {}
    for split in SPLITS:
        frame = tables[split].copy()
        frame["learned"] = oof[best_name] if split == "train" else np.clip(model.predict(frame[cols].to_numpy(np.float32)), 0.0, 1.0)
        frame.to_csv(seed_root / f"scored_{split}.csv", index=False)
        scored[split] = frame
    write_json(
        seed_root / "scorer_selection.json",
        {"seed": seed, "selected_model": best_name, "cv": cv_table.to_dict("records"), "n_features": len(cols), "features": cols,
         "val_top1_iou_learned": top1_iou(scored["val"], scored["val"]["learned"].to_numpy()),
         "val_top1_iou_confidence": top1_iou(scored["val"], scored["val"]["conf"].to_numpy())},
    )
    if AUDIT_GOLD_MUTATION:
        # Shift every eval gold box; candidate features and learned scores must not move.
        mutated = {}
        for row in rows["eval"]:
            gid = str(row["group_id"])
            w, h = float(row["image_width"]), float(row["image_height"])
            mutated[gid] = [[min(w, b[0] + 0.15 * w), min(h, b[1] + 0.15 * h), min(w, b[2] + 0.15 * w), min(h, b[3] + 0.15 * h)] for b in labels["eval"][gid]]
        table_mut = build_table("eval", rows["eval"], mutated, candidates["eval"], priors, estimates["eval"])
        same_keys = (table_mut["candidate_key"].tolist() == tables["eval"]["candidate_key"].tolist()) and (table_mut["group_id"].tolist() == tables["eval"]["group_id"].tolist())
        pred_mut = np.clip(model.predict(table_mut[cols].to_numpy(np.float32)), 0.0, 1.0)
        max_diff = float(np.abs(pred_mut - scored["eval"]["learned"].to_numpy()).max()) if same_keys else float("nan")
        target_moved = float(np.abs(table_mut["target_iou"].to_numpy() - tables["eval"]["target_iou"].to_numpy()).mean())
        audit = {"same_candidate_rows": bool(same_keys), "max_abs_learned_score_diff_after_gold_mutation": max_diff, "mean_abs_target_iou_change": target_moved, "pass": bool(same_keys and max_diff == 0.0 and target_moved > 0.0)}
        write_json(seed_root / "GOLD_MUTATION_AUDIT.json", audit)
        log(f"seed {seed}: gold-mutation audit pass={audit['pass']} (score diff {max_diff:.2e}, target moved {target_moved:.3f})")
    log(f"seed {seed}: scorer={best_name}; val top-1 IoU learned {top1_iou(scored['val'], scored['val']['learned'].to_numpy()):.4f} vs confidence {top1_iou(scored['val'], scored['val']['conf'].to_numpy()):.4f}")

    context = exact.load_multi_context(seed, protocol_root=canonical.PROTOCOL_ROOT, multi_source_root=canonical.UPSTREAM_ROOT, canonical_v3=True)
    base_candidates = {split: candidates[split] for split in SPLITS}
    results = []
    for variant in variants:
        result = run_variant(seed, variant, output_root, context, base_candidates, scored, rows, labels, source_ids)
        row = {"seed": seed, "variant": variant, "scorer": best_name, "alpha": result["learned_alpha"], "val_selected_coverage": result.get("val_selected_coverage")}
        for key in ("coverage_iou", "exact_union_iou", "set_f1_0_3", "set_f1_0_5", "mean_pred_count"):
            row[key] = float(result.get(key, float("nan")))
        results.append(row)
        log(f"seed {seed} {variant}: coverage {row['coverage_iou']:.4f} union {row['exact_union_iou']:.4f} f1@.5 {row['set_f1_0_5']:.4f}")
    return results


def main() -> None:
    global ALPHA_GRID, FEATURE_EXCLUDE, ALPHA_SELECTION, NO_AUX_ASSETS, MULTI_ALPHA, NO_RETUNE, AUDIT_GOLD_MUTATION
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--variants", nargs="+", default=["learned", "learned_heat", "learned_heat_slot"])
    parser.add_argument("--alpha-grid", nargs="+", type=float, default=list(ALPHA_GRID))
    parser.add_argument("--drop-features", nargs="*", default=[])
    parser.add_argument("--alpha-selection", choices=("val", "train_oof_val"), default="val")
    parser.add_argument("--no-aux-assets", action="store_true", help="do not load xattn/light-detector assets at all (features dropped)")
    parser.add_argument("--multi-alpha", choices=("same", "zero"), default="same", help="zero = re-rank only single-route candidates")
    parser.add_argument("--no-retune", action="store_true", help="keep the upstream validation calibration instead of re-tuning fusion on val")
    parser.add_argument("--audit-gold-mutation", action="store_true")
    args = parser.parse_args()
    ALPHA_SELECTION = args.alpha_selection
    NO_AUX_ASSETS = bool(args.no_aux_assets)
    MULTI_ALPHA = args.multi_alpha
    NO_RETUNE = bool(args.no_retune)
    AUDIT_GOLD_MUTATION = bool(args.audit_gold_mutation)
    if NO_AUX_ASSETS:
        args.drop_features = sorted(set(args.drop_features) | {"xattn_iou", "aux_iou"})
    ALPHA_GRID = tuple(args.alpha_grid)
    FEATURE_EXCLUDE = set(FEATURE_EXCLUDE) | set(args.drop_features)
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        rows.extend(run_seed(seed, args.output_root, args.variants))
        pd.DataFrame(rows).to_csv(args.output_root / "per_seed_metrics_partial.csv", index=False)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_root / "per_seed_metrics.csv", index=False)
    agg = frame.groupby("variant")[["coverage_iou", "exact_union_iou", "set_f1_0_3", "set_f1_0_5", "mean_pred_count"]].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg.reset_index().to_csv(args.output_root / "aggregate_metrics.csv", index=False)
    write_json(args.output_root / "FINAL_STATUS.json", {"status": "complete", "seeds": args.seeds, "variants": args.variants, "alpha_selection": ALPHA_SELECTION, "dropped_features": sorted(set(FEATURE_EXCLUDE)), "no_aux_assets": NO_AUX_ASSETS, "multi_alpha": MULTI_ALPHA, "retune_on_val": not NO_RETUNE, "aggregate": agg.reset_index().to_dict("records")})
    log("FINAL\n" + agg.round(4).to_string())


if __name__ == "__main__":
    main()

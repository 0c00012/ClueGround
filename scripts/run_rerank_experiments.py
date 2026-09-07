#!/usr/bin/env python
"""Reliability-gated candidate reranking for MS-CXR grounding.

This runner keeps the two protocols separate:

* singlebox_888: one query instance -> one bbox, MedRPG-compatible.
* multibox_1444: one query instance -> one or more boxes, MedGrounder-compatible.

Eval gold is used only for final metric/oracle/failure reporting.  Train is
used for the learned reranker; val is used for score and threshold selection.
Chest ImaGenome is intentionally not used here.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_ms_cxr_multibox_phrase_grounding_v1 as mb  # noqa: E402
from scripts import run_ms_cxr_semantic_finegrid_unified_yolo_pool_v1 as semantic_fg  # noqa: E402
from src.rerank.agreement_blend import choose_or_blend  # noqa: E402
from src.rerank.base_preserving_multibox_rescue import (  # noqa: E402
    action_distribution,
    action_oracle_analysis,
    pred_rows_to_map,
    predict_base_preserving,
)
from src.rerank.box_features import iou_xyxy  # noqa: E402
from src.rerank.candidate_table import (  # noqa: E402
    CONTRASTIVE_FINEGRID,
    SINGLEBOX_SEM,
    STAGE1_DATA,
    UNIFIED_FINEGRID,
    build_multibox_candidate_table,
    build_singlebox_candidate_table,
)
from src.rerank.cue_aware_set_selector import (  # noqa: E402
    add_cue_columns_to_eval,
    default_selector_grids,
    predict_cue_aware,
)
from src.rerank.learned_reranker import add_predictions, fit_reranker  # noqa: E402
from src.rerank.metrics import candidates_to_singlebox_predictions, oracle_singlebox, singlebox_metrics  # noqa: E402
from src.rerank.multibox_cue_parser import (  # noqa: E402
    cue_audit_frame,
    cue_gold_count_distribution,
    cue_info_from_groups,
    cue_type_counts,
)
from src.rerank.topk_policy import by_query_topk  # noqa: E402


DEFAULT_SINGLEBOX_TAGS = ["yolov8s", "yolov8m", "yolo11s", "yolo11m"]
GATE_COLUMNS = {
    "confidence": "confidence",
    "rule": "rule_context_score",
    "prior": "train_prior_score",
    "dino": "dino_reliable_agreement",
    "siglip": "siglip_z",
    "biomed": "biomedclip_z",
    "contrastive": "contrastive_score",
    "head": "score_head",
}


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dirs(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def minmax_norm(s: pd.Series) -> pd.Series:
    vals = s.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-8:
        return pd.Series(np.zeros(len(vals)), index=s.index)
    return (vals - lo) / (hi - lo)


def fit_gate_scaler(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    if "train" not in tables:
        raise ValueError("Gate scaler requires a train split")
    features: dict[str, dict[str, Any]] = {}
    for key, column in GATE_COLUMNS.items():
        if not all(column in frame.columns for frame in tables.values()):
            continue
        values = tables["train"][column].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        features[key] = {"column": column, "min": float(values.min()), "max": float(values.max())}
    return {"fit_split": "train", "shared_schema_splits": sorted(tables), "features": features}


def sanitize_gate_weights(weights: dict[str, float], scaler: dict[str, Any] | None) -> dict[str, float]:
    if scaler is None:
        return dict(weights)
    allowed = set(scaler.get("features", {}))
    return {key: (float(value) if key in allowed else 0.0) for key, value in weights.items()}


def add_gate_score(df: pd.DataFrame, weights: dict[str, float], scaler: dict[str, Any] | None = None) -> pd.DataFrame:
    out = df.copy()
    weights = sanitize_gate_weights(weights, scaler)
    score = np.zeros(len(out), dtype=float)
    for key, col in GATE_COLUMNS.items():
        if col in out.columns and float(weights.get(key, 0.0)) != 0:
            if scaler is None:
                normalized = minmax_norm(out[col]).to_numpy()
            else:
                spec = scaler.get("features", {}).get(key)
                if spec is None:
                    continue
                values = out[col].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
                lo, hi = float(spec["min"]), float(spec["max"])
                normalized = np.zeros(len(values), dtype=float) if hi - lo < 1e-8 else np.clip((values - lo) / (hi - lo), 0.0, 1.0)
            score += float(weights[key]) * normalized
    out["score_gate"] = score
    return out


def tune_gate_singlebox(train: pd.DataFrame, val: pd.DataFrame, scaler: dict[str, Any] | None = None) -> dict[str, float]:
    scaler = scaler or fit_gate_scaler({"train": train, "val": val})
    grids = []
    for dino_w in [0.0, 0.05, 0.1, 0.2]:
        for prior_w in [0.05, 0.1, 0.2]:
            for conf_w in [0.2, 0.4, 0.6]:
                grids.append({"confidence": conf_w, "rule": 0.12, "prior": prior_w, "dino": dino_w, "head": 0.25})
    best, best_metric = grids[0], -1.0
    for raw_weights in grids:
        w = sanitize_gate_weights(raw_weights, scaler)
        pred = candidates_to_singlebox_predictions(add_gate_score(val, w, scaler), "gate_val", "score_gate")
        metric = singlebox_metrics(pred, "gate_val")
        key = metric["mean_iou"] + 0.05 * metric["Hit@0.5"]
        if key > best_metric:
            best, best_metric = w, key
    return best


def tune_gate_multibox(
    val: pd.DataFrame,
    val_groups: dict[str, dict[str, Any]],
    cue_info: dict[str, dict[str, Any]] | None = None,
    gate_scaler: dict[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, Any], pd.DataFrame]:
    weights_grid = [
        {"confidence": 0.3, "rule": 0.15, "prior": 0.10, "dino": d, "head": 0.25}
        for d in [0.0, 0.05, 0.1, 0.2]
    ]
    if cue_info is None:
        params_grid = [
            {"max_k": k, "score_ratio": ratio, "nms_iou": nms}
            for k in [1, 2, 3]
            for ratio in [0.0, 0.25, 0.5, 0.75]
            for nms in [0.2, 0.4, 0.6]
        ]
    else:
        params_grid = [
            {
                "count_policy": "cue_count",
                "single_max_k": 1,
                "min_k_if_cue": min_k,
                "max_k_if_cue": max_k,
                "single_score_ratio": 1.0,
                "multi_score_ratio": ratio,
                "nms_iou": nms,
            }
            for min_k in [2]
            for max_k in [2, 3]
            for ratio in [0.0, 0.25, 0.5, 0.75]
            for nms in [0.2, 0.4, 0.6]
        ]
    rows = []
    best_w, best_p, best_key = weights_grid[0], params_grid[0], -1.0
    for raw_weights in weights_grid:
        w = sanitize_gate_weights(raw_weights, gate_scaler)
        scored = add_gate_score(val, w, gate_scaler)
        for p in params_grid:
            pred = multibox_predict(scored, "score_gate", p, cue_info=cue_info)
            eval_rows = pd.DataFrame(mb.eval_method("gate_val", val_groups, pred))
            summ = mb.summarize(eval_rows.to_dict("records"), "gate_val", "val_phrase_groups_all")
            row = {**w, **p, **summ}
            rows.append(row)
            if cue_info is None:
                key = float(summ["coverage_mean_iou"]) + 0.05 * float(summ["set_f1_0_3"])
            else:
                key = (
                    float(summ["coverage_mean_iou"])
                    + 0.18 * float(summ["union_iou"])
                    + 0.18 * float(summ["set_f1_0_3"])
                    + 0.05 * float(summ["set_f1_0_5"])
                    - 0.02 * float(summ["pred_count_abs_error"])
                )
            if key > best_key:
                best_key, best_w, best_p = key, w, p
    return best_w, best_p, pd.DataFrame(rows)


def load_or_build_tables(protocol_dir: Path, protocol: str, args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for split in ["train", "val", "eval"]:
        path = protocol_dir / "candidate_table" / f"{split}_candidate_table.csv"
        if path.exists() and not args.force_rebuild:
            out[split] = pd.read_csv(path)
            continue
        if protocol == "singlebox_888":
            out[split] = build_singlebox_candidate_table(split, detector_tags=args.singlebox_tags.split(","), allow_cache_fill=args.allow_cache_fill)
        else:
            out[split] = build_multibox_candidate_table(split)
        path.parent.mkdir(parents=True, exist_ok=True)
        out[split].to_csv(path, index=False)
    return out


def add_topk_oracles(protocol_dir: Path, tables: dict[str, pd.DataFrame], protocol: str) -> pd.DataFrame:
    rows = []
    for k in [5, 10, 15, 25]:
        for split in ["val", "eval"]:
            topk = by_query_topk(tables[split], max_total=k)
            if protocol == "singlebox_888":
                pred = oracle_singlebox(topk, f"topK_oracle_k{k}_{split}")
                rows.append(singlebox_metrics(pred, f"topK_oracle_k{k}_{split}") | {"split": split, "k": k, "oracle": True})
            else:
                # Candidate-level upper bound: best candidate per gold row proxy.
                vals = topk.groupby("query_id")["gold_best_iou"].max().to_numpy()
                rows.append(
                    {
                        "method": f"topK_candidate_oracle_k{k}_{split}",
                        "split": split,
                        "k": k,
                        "oracle": True,
                        "n": int(len(vals)),
                        "mean_iou": float(np.mean(vals)) if len(vals) else 0.0,
                        "Hit@0.3": float(np.mean(vals >= 0.3)) if len(vals) else 0.0,
                        "Hit@0.5": float(np.mean(vals >= 0.5)) if len(vals) else 0.0,
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(protocol_dir / "topk_oracle_summary.csv", index=False)
    return df


def fit_and_score_reranker(protocol_dir: Path, tables: dict[str, pd.DataFrame], target_col: str, args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    bundle = fit_reranker(tables["train"], target_col, model_name=args.reranker_model)
    model_dir = protocol_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    bundle.save(model_dir / "learned_candidate_reranker.joblib")
    write_json(model_dir / "reranker_features.json", {"features": bundle.feature_columns, "target": target_col, "model": bundle.model_name})
    return {split: add_predictions(bundle, df, "reranker_score") for split, df in tables.items()}


def singlebox_baseline_summary() -> pd.DataFrame:
    path = SINGLEBOX_SEM / "metrics" / "semantic_finegrid_singlebox_fair_summary.csv"
    df = pd.read_csv(path)
    keep = [
        "ours_semantic_finegrid_singlebox_fair",
        "YOLO-DINO fusion singlebox fair",
        "MedRPG fair full phrase s42",
        "MDETR token-box contrastive fair retrain s42",
    ]
    return df[df["method"].isin(keep)].copy()


def run_singlebox(protocol_dir: Path, args: argparse.Namespace) -> pd.DataFrame:
    ensure_dirs([protocol_dir / "metrics", protocol_dir / "predictions", protocol_dir / "configs", protocol_dir / "candidate_table"])
    tables = load_or_build_tables(protocol_dir, "singlebox_888", args)
    add_topk_oracles(protocol_dir / "metrics", tables, "singlebox_888")
    gate_scaler = fit_gate_scaler(tables)
    gate_w = tune_gate_singlebox(tables["train"], tables["val"], gate_scaler)
    write_json(protocol_dir / "configs" / "best_gate_weights.json", gate_w)
    write_json(protocol_dir / "configs" / "gate_train_scaler.json", gate_scaler)
    gated = {split: add_gate_score(df, gate_w, gate_scaler) for split, df in tables.items()}
    reranked = fit_and_score_reranker(protocol_dir, gated, "gold_iou", args)

    val_scores = []
    for name, col in [("dino_reliability_gated_rule_score", "score_gate"), ("learned_reranker_only", "reranker_score")]:
        pred = candidates_to_singlebox_predictions(reranked["val"], name, col)
        val_scores.append(singlebox_metrics(pred, name))
    pd.DataFrame(val_scores).to_csv(protocol_dir / "metrics" / "val_model_selection.csv", index=False)

    eval_preds = []
    for name, col in [
        ("dino_reliability_gated_rule_score", "score_gate"),
        ("learned_reranker_only", "reranker_score"),
    ]:
        pred = candidates_to_singlebox_predictions(reranked["eval"], name, col)
        pred = fill_singlebox_missing_with_baseline(pred, name)
        pred.to_csv(protocol_dir / "predictions" / f"{name}_eval_predictions.csv", index=False)
        eval_preds.append(pred)

    full = agreement_singlebox(reranked["eval"], protocol_dir)
    full = fill_singlebox_missing_with_baseline(full, "full_rescue_singlebox")
    full.to_csv(protocol_dir / "predictions" / "full_rescue_singlebox_eval_predictions.csv", index=False)
    eval_preds.append(full)

    summary = []
    for pred in eval_preds:
        summary.append(singlebox_metrics(pred, str(pred.iloc[0]["method"])))
    base = singlebox_baseline_summary()
    base_rows = []
    for _, r in base.iterrows():
        base_rows.append(
            {
                "method": r["method"],
                "n": int(r["n"]),
                "mean_iou": float(r["mean_iou"]),
                "median_iou": float(r["median_iou"]),
                "Hit@0.1": float(r["Hit@0.1"]),
                "Hit@0.3": float(r["Hit@0.3"]),
                "Hit@0.5": float(r["Hit@0.5"]),
                "bbox_missing_rate": float(r["bbox_missing_rate"]),
                "bbox_invalid_rate": float(r["bbox_invalid_rate"]),
                "reference": True,
            }
        )
    summary_df = pd.DataFrame(summary + base_rows).sort_values("mean_iou", ascending=False)
    summary_df.to_csv(protocol_dir / "metrics" / "metrics_summary.csv", index=False)
    failure_singlebox(protocol_dir, full)
    return summary_df


def fill_singlebox_missing_with_baseline(pred: pd.DataFrame, method: str) -> pd.DataFrame:
    """Fill no-candidate query instances with existing semantic finegrid output.

    This preserves the strict single-box protocol size.  The fallback does not
    use eval labels for tuning; it only avoids silently dropping hard examples.
    """

    base_path = SINGLEBOX_SEM / "predictions" / "semantic_finegrid_singlebox_fair_eval_predictions.csv"
    if not base_path.exists():
        return pred
    base = pd.read_csv(base_path)
    present = set(pred["sample_id"].astype(str)) if "sample_id" in pred.columns else set()
    missing = base[~base["sample_id"].astype(str).isin(present)].copy()
    if missing.empty:
        return pred
    rows = []
    for _, r in missing.iterrows():
        rows.append(
            {
                "method": method,
                "query_id": str(r["sample_id"]),
                "task_id": str(r["task_id"]),
                "sample_id": str(r["sample_id"]),
                "split": r.get("split", "eval"),
                "dicom_id": r.get("dicom_id", ""),
                "subject_id": r.get("subject_id", ""),
                "study_id": r.get("study_id", ""),
                "image_path": r.get("image_path", ""),
                "finding": r.get("finding", ""),
                "claim_sentence": r.get("claim_sentence", ""),
                "gt_x1": r["gt_x1"],
                "gt_y1": r["gt_y1"],
                "gt_x2": r["gt_x2"],
                "gt_y2": r["gt_y2"],
                "pred_x1": r["pred_x1"],
                "pred_y1": r["pred_y1"],
                "pred_x2": r["pred_x2"],
                "pred_y2": r["pred_y2"],
                "score": r.get("semantic_score", 0.0),
                "candidate_id": "fallback_existing_semantic_finegrid",
                "candidate_source": "fallback_existing_semantic_finegrid",
                "iou": r["iou"],
                "hit_0_1": bool(r["hit_0_1"]),
                "hit_0_3": bool(r["hit_0_3"]),
                "hit_0_5": bool(r["hit_0_5"]),
                "bbox_missing": False,
                "bbox_invalid": False,
                "fallback": True,
            }
        )
    return pd.concat([pred, pd.DataFrame(rows)], ignore_index=True, sort=False)


def agreement_singlebox(df: pd.DataFrame, protocol_dir: Path) -> pd.DataFrame:
    rows = []
    for qid, part in df.groupby("query_id", sort=False):
        gate = part.sort_values("score_gate", ascending=False).iloc[0]
        rer = part.sort_values("reranker_score", ascending=False).iloc[0]
        box_a = [gate["pred_x1"], gate["pred_y1"], gate["pred_x2"], gate["pred_y2"]]
        box_b = [rer["pred_x1"], rer["pred_y1"], rer["pred_x2"], rer["pred_y2"]]
        final, action = choose_or_blend(
            box_a,
            float(gate["score_gate"]),
            box_b,
            float(rer["reranker_score"]),
            alpha=0.15,
            iou_threshold=0.1,
            center_threshold=0.15,
            image_width=float(gate.get("image_width", 1.0)),
            image_height=float(gate.get("image_height", 1.0)),
        )
        gt = [gate["gt_x1"], gate["gt_y1"], gate["gt_x2"], gate["gt_y2"]]
        iou = iou_xyxy(final, gt)
        rows.append(
            {
                "method": "full_rescue_singlebox",
                "query_id": qid,
                "task_id": gate.get("task_id", qid),
                "sample_id": gate.get("sample_id", qid),
                "split": gate.get("split", ""),
                "dicom_id": gate.get("dicom_id", ""),
                "subject_id": gate.get("subject_id", ""),
                "study_id": gate.get("study_id", ""),
                "image_path": gate.get("image_path", ""),
                "finding": gate.get("finding", ""),
                "claim_sentence": gate.get("claim_sentence", ""),
                "gt_x1": gt[0],
                "gt_y1": gt[1],
                "gt_x2": gt[2],
                "gt_y2": gt[3],
                "pred_x1": final[0],
                "pred_y1": final[1],
                "pred_x2": final[2],
                "pred_y2": final[3],
                "action": action,
                "gate_candidate_id": gate.get("candidate_id", ""),
                "reranker_candidate_id": rer.get("candidate_id", ""),
                "iou": iou,
                "hit_0_1": iou >= 0.1,
                "hit_0_3": iou >= 0.3,
                "hit_0_5": iou >= 0.5,
                "bbox_missing": False,
                "bbox_invalid": False,
            }
        )
    return pd.DataFrame(rows)


def multibox_groups(split: str) -> dict[str, dict[str, Any]]:
    return mb.make_groups(read_jsonl(STAGE1_DATA / f"{split}.jsonl"))


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def infer_multibox_cue(claim: str, finding: str = "") -> dict[str, Any]:
    text = re.sub(r"\s+", " ", str(claim or "").lower()).strip()
    cues: list[str] = []
    k_hint = 1
    bilateral = bool(
        re.search(r"\b(bilateral|bilaterally|both|bibasilar|bibasal|pneumothoraces)\b", text)
        or re.search(r"\bright\s+(greater|more)\s+than\s+left\b", text)
        or re.search(r"\bleft\s+(greater|more)\s+than\s+right\b", text)
        or re.search(r"\bright\s+and\s+left\b|\bleft\s+and\s+right\b", text)
        or re.search(r"\bupper\s+lobes\b|\blower\s+lobes\b|\bbases\b", text)
    )
    plural_bilateral_like = bool(re.search(r"\b(effusions|opacities|consolidations|infiltrates)\b", text))
    if bilateral or (plural_bilateral_like and re.search(r"\bsmall\s+pleural\s+effusions\b|\bpleural\s+effusions\b", text)):
        cues.append("bilateral_or_plural")
        k_hint = max(k_hint, 2)
    diffuse = bool(re.search(r"\b(multifocal|multisegmental|multilobar|multiple|scattered|diffuse|widespread|extensive|several)\b", text))
    if diffuse:
        cues.append("diffuse_or_multifocal")
        k_hint = max(k_hint, 3 if re.search(r"\b(diffuse|widespread|extensive|multifocal)\b", text) else 2)
    if str(finding) in {"Edema", "Pneumonia"} and re.search(r"\b(perihilar|pulmonary edema|septal|interstitial)\b", text):
        cues.append("finding_distribution_prior")
        k_hint = max(k_hint, 2)
    return {
        "cue_text": ";".join(cues) if cues else "single_or_unspecified",
        "has_multi_cue": bool(cues),
        "k_hint": min(max(int(k_hint), 1), 4),
    }


def load_multibox_cue_info(split: str, groups: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    cue_path = UNIFIED_FINEGRID / "predictions" / f"{split}_cue_rows.csv"
    cue_info: dict[str, dict[str, Any]] = {}
    if cue_path.exists():
        cue_df = pd.read_csv(cue_path)
        for _, r in cue_df.iterrows():
            gid = str(r["group_id"])
            cue_info[gid] = {
                "cue_text": str(r.get("cue_text", "single_or_unspecified")),
                "has_multi_cue": _boolish(r.get("has_multi_cue", False)),
                "k_hint": int(float(r.get("k_hint", 1) or 1)),
                "cue_source": "semantic_finegrid_cue_rows",
            }
    for gid, g in groups.items():
        if gid not in cue_info:
            inferred = infer_multibox_cue(str(g.get("claim_sentence", "")), str(g.get("finding", "")))
            inferred["cue_source"] = "local_inference"
            cue_info[gid] = inferred
    return cue_info


def cue_limited_params(qid: str, params: dict[str, Any], cue_info: dict[str, dict[str, Any]] | None) -> tuple[int, float, dict[str, Any]]:
    if cue_info is None or str(params.get("count_policy", "")) != "cue_count":
        return int(params.get("max_k", 2)), float(params.get("score_ratio", 0.5)), {}
    cue = cue_info.get(str(qid), {"has_multi_cue": False, "k_hint": 1, "cue_text": "single_or_unspecified"})
    if _boolish(cue.get("has_multi_cue", False)):
        k_hint = int(float(cue.get("k_hint", 1) or 1))
        max_k = max(k_hint, int(params.get("min_k_if_cue", 2)))
        max_k = min(max_k, int(params.get("max_k_if_cue", 3)))
        ratio = float(params.get("multi_score_ratio", params.get("score_ratio", 0.5)))
    else:
        max_k = int(params.get("single_max_k", 1))
        ratio = float(params.get("single_score_ratio", 1.0))
    return max_k, ratio, cue


def multibox_predict(
    df: pd.DataFrame,
    score_col: str,
    params: dict[str, Any],
    cue_info: dict[str, dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    nms_iou = float(params.get("nms_iou", 0.4))
    out: dict[str, list[dict[str, Any]]] = {}
    for qid, part in df.groupby("query_id", sort=False):
        max_k, ratio, cue = cue_limited_params(str(qid), params, cue_info)
        part = part.sort_values(score_col, ascending=False)
        if part.empty:
            out[str(qid)] = []
            continue
        top_score = float(part.iloc[0][score_col])
        selected = []
        for _, r in part.iterrows():
            if len(selected) >= max_k:
                break
            if top_score > 0 and float(r[score_col]) < top_score * ratio:
                continue
            box = [float(r["pred_x1"]), float(r["pred_y1"]), float(r["pred_x2"]), float(r["pred_y2"])]
            if all(iou_xyxy(box, old["box"]) < nms_iou for old in selected):
                selected.append(
                    {
                        "box": box,
                        "score": float(r[score_col]),
                        "source": score_col,
                        "cue_text": cue.get("cue_text", "") if cue else "",
                        "has_multi_cue": bool(cue.get("has_multi_cue", False)) if cue else False,
                    }
                )
        if not selected:
            r = part.iloc[0]
            selected = [{"box": [r["pred_x1"], r["pred_y1"], r["pred_x2"], r["pred_y2"]], "score": float(r[score_col]), "source": score_col}]
        out[str(qid)] = selected
    return out


def summarize_multibox(method: str, groups: dict[str, dict[str, Any]], preds: dict[str, list[dict[str, Any]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = pd.DataFrame(mb.eval_method(method, groups, preds))
    subsets = {
        "eval_phrase_groups_all": list(groups.keys()),
        "eval_phrase_groups_single_box": [gid for gid, g in groups.items() if len(g["gt_boxes"]) == 1],
        "eval_phrase_groups_multi_box": [gid for gid, g in groups.items() if len(g["gt_boxes"]) > 1],
    }
    summary = []
    for subset, gids in subsets.items():
        part = rows[rows["group_id"].isin(gids)]
        summary.append(mb.summarize(part.to_dict("records"), method, subset))
    return rows, pd.DataFrame(summary)


def multibox_objective(rows: pd.DataFrame, summ: dict[str, Any], objective: str) -> float:
    over = float((rows["n_pred"].astype(float) > rows["n_gt"].astype(float)).mean()) if len(rows) else 0.0
    if objective == "f1":
        return float(summ["set_f1_0_3"]) + 0.08 * float(summ["union_iou"]) + 0.04 * float(summ["coverage_mean_iou"]) - 0.02 * over
    if objective == "union":
        return float(summ["union_iou"]) + 0.10 * float(summ["set_f1_0_3"]) + 0.05 * float(summ["coverage_mean_iou"]) - 0.04 * over
    return (
        0.45 * float(summ["union_iou"])
        + 0.40 * float(summ["set_f1_0_3"])
        + 0.15 * float(summ["coverage_mean_iou"])
        - 0.05 * over
    )


def tune_cue_selector(
    val_df: pd.DataFrame,
    val_groups: dict[str, dict[str, Any]],
    cue_info: dict[str, dict[str, Any]],
    *,
    score_col: str,
    variant: str,
    objective: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    grids = default_selector_grids()
    base_variant = variant
    if variant in {"cue_sideaware_f1_selected", "cue_sideaware_union_selected", "cue_sideaware_balanced_selected"}:
        base_variant = "cue_sideaware_union_guard"
    params_list = grids.get(base_variant, grids["cue_sideaware_union_guard"])
    rows: list[dict[str, Any]] = []
    best_params = params_list[0]
    best_score = -1e9
    for idx, params in enumerate(params_list):
        preds = predict_cue_aware(val_df, score_col, cue_info, params=params, variant=base_variant)
        eval_rows = pd.DataFrame(mb.eval_method(f"{variant}_val", val_groups, preds))
        summ = mb.summarize(eval_rows.to_dict("records"), f"{variant}_val", "val_phrase_groups_all")
        key = multibox_objective(eval_rows, summ, objective)
        rows.append({"grid_index": idx, "selector_variant": variant, "objective": objective, "objective_value": key, **params, **summ})
        if key > best_score:
            best_score = key
            best_params = dict(params)
    return best_params, pd.DataFrame(rows)


def selector_param_keys() -> list[str]:
    keys: set[str] = set()
    for params_list in default_selector_grids().values():
        for params in params_list:
            keys.update(params.keys())
    return sorted(keys)


def select_params_from_grid(grid: pd.DataFrame, objective: str) -> dict[str, Any]:
    if grid.empty:
        return {}
    scored = grid.copy()
    over = scored.get("overprediction_rate", pd.Series(0.0, index=scored.index)).astype(float)
    if "overprediction_rate" not in scored.columns and {"n_pred", "n_gt_boxes", "n_groups"}.issubset(scored.columns):
        over = pd.Series(0.0, index=scored.index)
    if objective == "f1":
        key = scored["set_f1_0_3"].astype(float) + 0.08 * scored["union_iou"].astype(float) + 0.04 * scored["coverage_mean_iou"].astype(float) - 0.02 * over
    elif objective == "union":
        key = scored["union_iou"].astype(float) + 0.10 * scored["set_f1_0_3"].astype(float) + 0.05 * scored["coverage_mean_iou"].astype(float) - 0.04 * over
    else:
        key = (
            0.45 * scored["union_iou"].astype(float)
            + 0.40 * scored["set_f1_0_3"].astype(float)
            + 0.15 * scored["coverage_mean_iou"].astype(float)
            - 0.05 * over
        )
    row = scored.loc[key.idxmax()]
    params: dict[str, Any] = {}
    for k in selector_param_keys():
        if k not in row or pd.isna(row[k]):
            continue
        val = row[k]
        if isinstance(val, (np.integer, int)):
            params[k] = int(val)
        elif isinstance(val, (np.floating, float)):
            as_float = float(val)
            params[k] = int(as_float) if abs(as_float - round(as_float)) < 1e-9 and k.endswith("_k") else as_float
        else:
            params[k] = val
    return params


def count_diagnostics(rows: pd.DataFrame, method: str) -> dict[str, Any]:
    if rows.empty:
        return {"method": method}
    out: dict[str, Any] = {
        "method": method,
        "n_groups": int(len(rows)),
        "mean_pred_count": float(rows["n_pred"].astype(float).mean()),
        "overprediction_rate": float((rows["n_pred"].astype(float) > rows["n_gt"].astype(float)).mean()),
        "underprediction_rate": float((rows["n_pred"].astype(float) < rows["n_gt"].astype(float)).mean()),
    }
    for gc in [1, 2, 3]:
        part = rows[rows["n_gt"].astype(int).eq(gc)]
        if len(part):
            out[f"gold_count_{gc}_n"] = int(len(part))
            out[f"gold_count_{gc}_mean_pred_count"] = float(part["n_pred"].astype(float).mean())
            out[f"gold_count_{gc}_union_iou"] = float(part["union_iou"].astype(float).mean())
            out[f"gold_count_{gc}_set_f1_0_3"] = float(part["set_f1_0_3"].astype(float).mean())
    return out


def subgroup_diagnostics(rows: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if rows.empty or group_col not in rows.columns:
        return pd.DataFrame()
    out = []
    for (method, key), part in rows.groupby(["method", group_col], dropna=False):
        out.append(
            {
                "method": method,
                group_col: key,
                "n_groups": int(len(part)),
                "mean_pred_count": float(part["n_pred"].astype(float).mean()),
                "overprediction_rate": float((part["n_pred"].astype(float) > part["n_gt"].astype(float)).mean()),
                "underprediction_rate": float((part["n_pred"].astype(float) < part["n_gt"].astype(float)).mean()),
                "coverage_mean_iou": float(part["coverage_mean_iou"].astype(float).mean()),
                "union_iou": float(part["union_iou"].astype(float).mean()),
                "set_f1_0_3": float(part["set_f1_0_3"].astype(float).mean()),
                "set_f1_0_5": float(part["set_f1_0_5"].astype(float).mean()),
            }
        )
    return pd.DataFrame(out)


def load_existing_finegrid_rows() -> pd.DataFrame:
    path = CONTRASTIVE_FINEGRID / "predictions" / "finegrid_plus_contrastive_global_eval_phrase_group_predictions.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _semantic_args() -> argparse.Namespace:
    cfg_path = UNIFIED_FINEGRID / "configs" / "run_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    return argparse.Namespace(**cfg["args"])


def _semantic_tags() -> list[str]:
    cfg_path = UNIFIED_FINEGRID / "configs" / "run_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    return list(cfg.get("candidate_tags") or semantic_fg.parse_tags(cfg["args"].get("tag_preset", "yolo8_nsml_yolo11_sm")))


def _semantic_scored_sets(split: str, groups: dict[str, dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    args = _semantic_args()
    pred_dir = UNIFIED_FINEGRID / "predictions"
    cfg_dir = UNIFIED_FINEGRID / "configs"
    sig_path = pred_dir / f"{split}_siglip_scored_candidates_{args.prompt_mode}_m{str(args.margin).replace('.', 'p')}.csv"
    bio_path = pred_dir / f"{split}_biomedclip_scored_candidates_{args.prompt_mode}_m{str(args.margin).replace('.', 'p')}.csv"
    if not sig_path.exists() or not bio_path.exists():
        raise FileNotFoundError(f"Missing cached semantic scored candidates for split={split}: {sig_path}, {bio_path}")
    sig = pd.read_csv(sig_path)
    bio = pd.read_csv(bio_path)
    sig_row_params = json.loads((cfg_dir / "best_siglip_row_params.json").read_text(encoding="utf-8"))
    sig_set_params = json.loads((cfg_dir / "best_siglip_set_params.json").read_text(encoding="utf-8"))
    bio_row_params = json.loads((cfg_dir / "best_biomedclip_row_params.json").read_text(encoding="utf-8"))
    bio_set_params = json.loads((cfg_dir / "best_biomedclip_set_params.json").read_text(encoding="utf-8"))
    sig = semantic_fg.sf.apply_fusion_score(sig, sig_row_params, "siglip_fusion_score")
    bio = semantic_fg.sf.apply_fusion_score(bio, bio_row_params, "biomedclip_fusion_score")
    sig_set = semantic_fg.sf.predict_phrase_sets(groups, semantic_fg.sf.scored_candidates_by_group(sig, groups, "siglip_fusion_score"), sig_set_params)
    bio_set = semantic_fg.sf.predict_phrase_sets(groups, semantic_fg.sf.scored_candidates_by_group(bio, groups, "biomedclip_fusion_score"), bio_set_params)
    return sig_set, bio_set


def build_semantic_finegrid_proxy_rows(split: str, groups: dict[str, dict[str, Any]], protocol_dir: Path) -> pd.DataFrame:
    """Rebuild cached semantic finegrid rows for val-time policy selection.

    The exact finegrid+contrastive-global eval rows are loaded from the frozen
    previous experiment.  Its val per-query boxes were not persisted, so val
    tuning uses the closest saved base pipeline: semantic finegrid with the
    frozen validation-selected parameters.  This keeps eval/test labels out of
    policy selection.
    """

    cache = protocol_dir / "base_cache" / f"{split}_semantic_finegrid_proxy_rows.csv"
    if cache.exists():
        return pd.read_csv(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tags = _semantic_tags()
    fusion = json.loads((UNIFIED_FINEGRID / "configs" / "best_singleton_fusion_by_finding.json").read_text(encoding="utf-8"))
    hybrid, cue = semantic_fg.build_hybrid(split, groups, tags, fusion, cache.parent)
    sig_set, bio_set = _semantic_scored_sets(split, groups)
    best = json.loads((UNIFIED_FINEGRID / "configs" / "best_finegrid_params.json").read_text(encoding="utf-8"))
    weights = (float(best["w_hybrid"]), float(best["w_siglip"]), float(best["w_biomed"]))
    preds, audit = semantic_fg.predict_finegrid(
        groups,
        hybrid,
        sig_set,
        bio_set,
        cue,
        weights,
        float(best["min_hybrid_semantic_iou"]),
        bool(best["keep_multi"]),
    )
    rows = pd.DataFrame(mb.eval_method("base_semantic_finegrid_proxy", groups, preds))
    rows.to_csv(cache, index=False)
    audit.to_csv(protocol_dir / "base_cache" / f"{split}_semantic_finegrid_proxy_audit.csv", index=False)
    return rows


def load_base_rows(split: str, groups: dict[str, dict[str, Any]], protocol_dir: Path, *, exact_eval: bool) -> pd.DataFrame:
    if split == "eval" and exact_eval:
        rows = load_existing_finegrid_rows()
        if not rows.empty:
            return rows.copy()
    return build_semantic_finegrid_proxy_rows(split, groups, protocol_dir)


def base_preserving_objective(rows: pd.DataFrame, summ: dict[str, Any], audit: pd.DataFrame) -> float:
    over = float((rows["n_pred"].astype(float) > rows["n_gt"].astype(float)).mean()) if len(rows) else 0.0
    changed = float((~audit["action_taken"].astype(str).eq("keep_base")).mean()) if len(audit) else 0.0
    return (
        0.45 * float(summ["union_iou"])
        + 0.40 * float(summ["set_f1_0_3"])
        + 0.15 * float(summ["coverage_mean_iou"])
        - 0.05 * over
        - 0.03 * changed
    )


def base_preserving_param_grid(variant: str) -> list[dict[str, Any]]:
    common = {
        "candidate_topn": 30,
        "base_default_score": 0.45,
        "same_box_iou": 0.82,
        "replace_scan_topn": 20,
        "add_scan_topn": 30,
        "target_weight": 0.18,
        "target_match_min": 0.5,
    }
    if variant == "keep_or_replace_only":
        return [
            {
                **common,
                "replace_margin": margin,
                "replace_ratio": ratio,
                "replace_abs_threshold": abs_thr,
                "single_union_expansion_threshold": hull,
            }
            for margin in [0.12, 0.16, 0.20]
            for ratio in [1.10, 1.20]
            for abs_thr in [0.55, 0.65]
            for hull in [1.15, 1.30]
        ]
    if variant == "add_missing_side_only":
        return [
            {
                **common,
                "add_abs_threshold": abs_thr,
                "add_score_ratio": ratio,
                "duplicate_iou_threshold": dup,
                "multi_union_expansion_threshold": hull,
                "diffuse_union_expansion_threshold": 2.2,
            }
            for abs_thr in [0.58, 0.66, 0.74]
            for ratio in [0.80, 0.88, 0.94]
            for dup in [0.35, 0.50]
            for hull in [1.5, 1.9]
        ]
    if variant == "prune_overprediction":
        return [{**common}]
    return [
        {
            **common,
            "replace_margin": margin,
            "replace_ratio": ratio,
            "replace_abs_threshold": repl_abs,
            "add_abs_threshold": add_abs,
            "add_score_ratio": add_ratio,
            "duplicate_iou_threshold": 0.45,
            "single_union_expansion_threshold": 1.20,
            "multi_union_expansion_threshold": 1.7,
            "diffuse_union_expansion_threshold": 2.2,
        }
        for margin in [0.14, 0.18]
        for ratio in [1.15, 1.25]
        for repl_abs in [0.58, 0.68]
        for add_abs in [0.62, 0.72]
        for add_ratio in [0.84, 0.92]
    ]


def tune_base_preserving_policy(
    val_base_rows: pd.DataFrame,
    val_df: pd.DataFrame,
    val_groups: dict[str, dict[str, Any]],
    val_cue_info: dict[str, dict[str, Any]],
    *,
    score_col: str,
    variant: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    base_preds = pred_rows_to_map(val_base_rows)
    rows = []
    best_params: dict[str, Any] = {}
    best_key = -1e9
    for idx, params in enumerate(base_preserving_param_grid(variant)):
        preds, audit = predict_base_preserving(base_preds, val_df, val_cue_info, val_groups, params, score_col=score_col, variant=variant)
        eval_rows = pd.DataFrame(mb.eval_method(f"{variant}_val", val_groups, preds))
        summ = mb.summarize(eval_rows.to_dict("records"), f"{variant}_val", "val_phrase_groups_all")
        key = base_preserving_objective(eval_rows, summ, audit)
        rows.append({"grid_index": idx, "variant": variant, "objective": key, **params, **summ, "changed_query_rate": float((~audit["action_taken"].astype(str).eq("keep_base")).mean())})
        if key > best_key:
            best_key = key
            best_params = dict(params)
    return best_params, pd.DataFrame(rows)


def add_base_preserving_deltas(base_rows: pd.DataFrame, new_rows: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    keep_cols = ["group_id", "finding", "claim_sentence", "n_gt", "n_pred", "coverage_mean_iou", "union_iou", "set_f1_0_3", "set_f1_0_5"]
    base = base_rows[keep_cols].rename(
        columns={
            "n_pred": "base_pred_count",
            "coverage_mean_iou": "base_coverage_iou",
            "union_iou": "base_union_iou",
            "set_f1_0_3": "base_setf1_03",
            "set_f1_0_5": "base_setf1_05",
        }
    )
    new = new_rows[keep_cols].rename(
        columns={
            "n_pred": "new_pred_count",
            "coverage_mean_iou": "new_coverage_iou",
            "union_iou": "new_union_iou",
            "set_f1_0_3": "new_setf1_03",
            "set_f1_0_5": "new_setf1_05",
        }
    )
    out = base.merge(new, on=["group_id", "finding", "claim_sentence", "n_gt"], how="left")
    out = out.merge(audit[["group_id", "action_taken", "has_multi_cue", "multi_cue_type", "k_hint"]], on="group_id", how="left")
    out["delta_union_iou"] = out["new_union_iou"].astype(float) - out["base_union_iou"].astype(float)
    out["delta_setf1_03"] = out["new_setf1_03"].astype(float) - out["base_setf1_03"].astype(float)
    out["delta_coverage_iou"] = out["new_coverage_iou"].astype(float) - out["base_coverage_iou"].astype(float)
    out["changed"] = ~out["action_taken"].fillna("keep_base").astype(str).eq("keep_base")
    return out


def write_cue_audit_outputs(
    protocol_dir: Path,
    eval_groups: dict[str, dict[str, Any]],
    cue_info: dict[str, dict[str, Any]],
    old_rows: pd.DataFrame,
    new_rows: pd.DataFrame,
) -> None:
    audit = cue_audit_frame(eval_groups, cue_info)
    old_small = old_rows[
        ["group_id", "n_pred", "union_iou", "set_f1_0_3", "coverage_mean_iou"]
    ].rename(
        columns={
            "n_pred": "old_full_rescue_pred_count",
            "union_iou": "old_union_iou",
            "set_f1_0_3": "old_setf1_03",
            "coverage_mean_iou": "old_coverage_mean_iou",
        }
    )
    new_small = new_rows[
        ["group_id", "n_pred", "union_iou", "set_f1_0_3", "coverage_mean_iou"]
    ].rename(
        columns={
            "n_pred": "new_cue_selector_pred_count",
            "union_iou": "new_union_iou",
            "set_f1_0_3": "new_setf1_03",
            "coverage_mean_iou": "new_coverage_mean_iou",
        }
    )
    audit = audit.merge(old_small, on="group_id", how="left").merge(new_small, on="group_id", how="left")
    audit.to_csv(protocol_dir / "cue_audit.csv", index=False)
    cue_type_counts(audit).to_csv(protocol_dir / "cue_type_counts.csv", index=False)
    cue_gold_count_distribution(audit).to_csv(protocol_dir / "cue_confusion_with_gold_count.csv", index=False)
    examples = {}
    for cue_type, part in audit.groupby("multi_cue_type", dropna=False):
        examples[str(cue_type)] = part.head(20)[
            ["group_id", "phrase", "finding", "gold_count", "k_hint", "target_qs", "cue_debug_string"]
        ].to_dict("records")
    (protocol_dir / "examples_by_cue_type.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_boxes_json(value: Any) -> list[list[float]]:
    try:
        boxes = json.loads(value) if isinstance(value, str) else value
        return [[float(x) for x in b] for b in boxes]
    except Exception:
        return []


def make_multibox_contact_sheets(
    protocol_dir: Path,
    eval_groups: dict[str, dict[str, Any]],
    cue_audit: pd.DataFrame,
    baseline_rows: pd.DataFrame,
    old_rows: pd.DataFrame,
    new_rows: pd.DataFrame,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    out_dir = protocol_dir / "contact_sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()

    def rows_to_pred_map(rows: pd.DataFrame) -> dict[str, list[list[float]]]:
        if rows.empty:
            return {}
        return {str(r["group_id"]): _parse_boxes_json(r.get("pred_boxes_json", "[]")) for _, r in rows.iterrows()}

    baseline_map = rows_to_pred_map(baseline_rows)
    old_map = rows_to_pred_map(old_rows)
    new_map = rows_to_pred_map(new_rows)

    merged = cue_audit.copy()
    merged["union_delta"] = merged.get("new_union_iou", 0).astype(float) - merged.get("old_union_iou", 0).astype(float)
    merged["setf1_delta"] = merged.get("new_setf1_03", 0).astype(float) - merged.get("old_setf1_03", 0).astype(float)
    cases = {
        "no_cue_overprediction_fixed.jpg": merged[
            (~merged["has_multi_cue"].astype(bool))
            & (merged["old_full_rescue_pred_count"].fillna(0).astype(float) > 1)
            & (merged["new_cue_selector_pred_count"].fillna(0).astype(float) <= 1)
        ].sort_values("union_delta", ascending=False),
        "bilateral_targetq_success.jpg": merged[
            merged["multi_cue_type"].astype(str).isin(["bilateral", "both_sides", "right_and_left", "left_and_right", "bibasilar", "both_bases"])
        ].sort_values("setf1_delta", ascending=False),
        "cue_parser_failure_cases.jpg": merged[
            (merged["has_multi_cue"].astype(bool) & merged["gold_count"].astype(int).eq(1))
            | ((~merged["has_multi_cue"].astype(bool)) & merged["gold_count"].astype(int).gt(1))
        ],
        "union_guard_rescue_cases.jpg": merged.sort_values("union_delta", ascending=False),
        "conservative_selector_hurt_cases.jpg": merged.sort_values("union_delta", ascending=True),
    }

    def draw_case(gid: str) -> Image.Image | None:
        g = eval_groups.get(str(gid))
        if not g:
            return None
        path = Path(g["image_path"])
        if not path.exists():
            return None
        img = Image.open(path).convert("RGB")
        img.thumbnail((420, 420))
        sx = img.width / float(g["image_width"])
        sy = img.height / float(g["image_height"])
        draw = ImageDraw.Draw(img)

        def box_scaled(b: list[float]) -> list[float]:
            return [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]

        for b in g["gt_boxes"]:
            draw.rectangle(box_scaled(b), outline=(255, 220, 0), width=3)
        for b in baseline_map.get(str(gid), []):
            draw.rectangle(box_scaled(b), outline=(60, 140, 255), width=2)
        for b in old_map.get(str(gid), []):
            draw.rectangle(box_scaled(b), outline=(255, 60, 60), width=2)
        for b in new_map.get(str(gid), []):
            draw.rectangle(box_scaled(b), outline=(0, 230, 120), width=3)
        cue = cue_audit[cue_audit["group_id"].astype(str).eq(str(gid))]
        title = str(g["claim_sentence"])[:70]
        if len(cue):
            r = cue.iloc[0]
            title = f"{str(r['multi_cue_type'])} k={r['k_hint']} {title}"
        canvas = Image.new("RGB", (img.width, img.height + 56), (20, 20, 20))
        canvas.paste(img, (0, 0))
        d2 = ImageDraw.Draw(canvas)
        d2.text((4, img.height + 4), title, fill=(255, 255, 255), font=font)
        d2.text((4, img.height + 22), "gold yellow | baseline blue | old full red | new cue green", fill=(255, 255, 255), font=font)
        return canvas

    for filename, frame in cases.items():
        panels: list[Image.Image] = []
        for gid in frame["group_id"].astype(str).head(16):
            panel = draw_case(gid)
            if panel is not None:
                panels.append(panel)
        if not panels:
            continue
        w = max(p.width for p in panels)
        h = max(p.height for p in panels)
        cols = 4
        rows = int(math.ceil(len(panels) / cols))
        sheet = Image.new("RGB", (cols * w, rows * h), (30, 30, 30))
        for idx, p in enumerate(panels):
            sheet.paste(p, ((idx % cols) * w, (idx // cols) * h))
        sheet.save(out_dir / filename, quality=92)


def run_multibox(protocol_dir: Path, args: argparse.Namespace) -> pd.DataFrame:
    ensure_dirs([protocol_dir / "metrics", protocol_dir / "predictions", protocol_dir / "configs", protocol_dir / "candidate_table", protocol_dir / "contact_sheets"])
    tables = load_or_build_tables(protocol_dir, "multibox_1444", args)
    add_topk_oracles(protocol_dir / "metrics", tables, "multibox_1444")
    val_groups = multibox_groups("val")
    eval_groups = multibox_groups("eval")
    val_cue_info = cue_info_from_groups(val_groups)
    eval_cue_info = cue_info_from_groups(eval_groups)
    cue_audit_frame(val_groups, val_cue_info).to_csv(protocol_dir / "metrics" / "val_cue_count_info.csv", index=False)
    cue_audit_frame(eval_groups, eval_cue_info).to_csv(protocol_dir / "metrics" / "eval_cue_count_info.csv", index=False)
    gate_scaler = fit_gate_scaler(tables)
    gate_w, gate_params, gate_grid = tune_gate_multibox(tables["val"], val_groups, gate_scaler=gate_scaler)
    write_json(protocol_dir / "configs" / "best_gate_weights.json", gate_w)
    write_json(protocol_dir / "configs" / "gate_train_scaler.json", gate_scaler)
    write_json(protocol_dir / "configs" / "best_multibox_selection_params.json", gate_params)
    gate_grid.to_csv(protocol_dir / "metrics" / "gate_val_grid.csv", index=False)
    gated = {split: add_gate_score(df, gate_w, gate_scaler) for split, df in tables.items()}
    reranked = fit_and_score_reranker(protocol_dir, gated, "gold_best_iou", args)

    summaries = []
    pred_frames = []
    eval_rows_for_diagnostics = []
    for name, col in [
        ("dino_reliability_gated_rule_score_multibox", "score_gate"),
        ("learned_reranker_topK_threshold_multibox", "reranker_score"),
    ]:
        preds = multibox_predict(reranked["eval"], col, gate_params)
        rows, summ = summarize_multibox(name, eval_groups, preds)
        rows.to_csv(protocol_dir / "predictions" / f"{name}_eval_phrase_group_predictions.csv", index=False)
        summaries.append(summ)
        pred_frames.append(rows)
        eval_rows_for_diagnostics.append(add_cue_columns_to_eval(rows, eval_cue_info))

    full_preds = multibox_predict(reranked["eval"].assign(full_score=0.55 * reranked["eval"]["reranker_score"] + 0.45 * reranked["eval"]["score_gate"]), "full_score", gate_params)
    full_rows, full_summary = summarize_multibox("full_rescue_multibox", eval_groups, full_preds)
    full_rows.to_csv(protocol_dir / "predictions" / "full_rescue_multibox_eval_phrase_group_predictions.csv", index=False)
    summaries.append(full_summary)
    pred_frames.append(full_rows)
    eval_rows_for_diagnostics.append(add_cue_columns_to_eval(full_rows, eval_cue_info))

    full_val_df = reranked["val"].assign(full_score=0.55 * reranked["val"]["reranker_score"] + 0.45 * reranked["val"]["score_gate"])
    full_eval_df = reranked["eval"].assign(full_score=0.55 * reranked["eval"]["reranker_score"] + 0.45 * reranked["eval"]["score_gate"])
    cue_variant_specs = [
        ("cue_cap_only", "balanced", True),
        ("cue_sideaware_selector", "balanced", True),
        ("cue_sideaware_union_guard", "balanced", True),
        ("cue_sideaware_f1_selected", "f1", False),
        ("cue_sideaware_union_selected", "union", False),
        ("cue_sideaware_balanced_selected", "balanced", False),
        ("conservative_cue_rescue_multibox", "balanced", True),
    ]
    selector_grids = []
    cue_variant_eval_rows: dict[str, pd.DataFrame] = {}
    guard_grid_cache: pd.DataFrame | None = None
    guard_params_cache: dict[str, Any] | None = None
    for name, objective, needs_tune in cue_variant_specs:
        if needs_tune:
            best_params, grid = tune_cue_selector(
                full_val_df,
                val_groups,
                val_cue_info,
                score_col="full_score",
                variant=name,
                objective=objective,
            )
            if name == "cue_sideaware_union_guard":
                guard_grid_cache = grid.copy()
                guard_params_cache = dict(best_params)
            selector_grids.append(grid)
        else:
            if guard_grid_cache is None:
                best_params, grid = tune_cue_selector(
                    full_val_df,
                    val_groups,
                    val_cue_info,
                    score_col="full_score",
                    variant="cue_sideaware_union_guard",
                    objective="balanced",
                )
                guard_grid_cache = grid.copy()
                guard_params_cache = dict(best_params)
                selector_grids.append(grid)
            best_params = select_params_from_grid(guard_grid_cache, objective)
            if not best_params and guard_params_cache is not None:
                best_params = dict(guard_params_cache)
        write_json(protocol_dir / "configs" / f"best_{name}_params.json", best_params)
        base_variant = "cue_sideaware_union_guard" if name in {"cue_sideaware_f1_selected", "cue_sideaware_union_selected", "cue_sideaware_balanced_selected"} else name
        preds = predict_cue_aware(full_eval_df, "full_score", eval_cue_info, params=best_params, variant=base_variant)
        rows, summ = summarize_multibox(name, eval_groups, preds)
        rows = add_cue_columns_to_eval(rows, eval_cue_info)
        rows.to_csv(protocol_dir / "predictions" / f"{name}_eval_phrase_group_predictions.csv", index=False)
        summaries.append(summ)
        pred_frames.append(rows)
        eval_rows_for_diagnostics.append(rows)
        cue_variant_eval_rows[name] = rows

    if selector_grids:
        pd.concat(selector_grids, ignore_index=True, sort=False).to_csv(protocol_dir / "metrics" / "cue_selector_val_grid.csv", index=False)

    # Base-preserving rescue: keep finegrid+contrastive as the default and
    # only apply conservative replace/add/prune actions when val-selected
    # rules are confident enough.  Singlebox_888 is intentionally untouched.
    val_base_rows = load_base_rows("val", val_groups, protocol_dir, exact_eval=False)
    eval_base_rows = load_base_rows("eval", eval_groups, protocol_dir, exact_eval=True)
    eval_base_rows.to_csv(protocol_dir / "predictions" / "base_finegrid_contrastive_eval_phrase_group_predictions.csv", index=False)
    oracle_dir = protocol_dir / "action_oracle"
    oracle_dir.mkdir(parents=True, exist_ok=True)
    oracle_summary, oracle_per_query = action_oracle_analysis(
        eval_groups,
        pred_rows_to_map(eval_base_rows),
        full_eval_df,
        eval_cue_info,
        score_col="full_score",
        top_n=20,
    )
    oracle_summary.to_csv(oracle_dir / "action_oracle_summary.csv", index=False)
    oracle_per_query.to_csv(oracle_dir / "action_oracle_per_query.csv", index=False)

    base_preserving_grids: list[pd.DataFrame] = []
    base_preserving_eval_rows: dict[str, pd.DataFrame] = {}
    base_preserving_specs = [
        "keep_or_replace_only",
        "add_missing_side_only",
        "prune_overprediction",
        "base_preserving_full_rule",
    ]
    val_base_preds = pred_rows_to_map(val_base_rows)
    eval_base_preds = pred_rows_to_map(eval_base_rows)
    for name in base_preserving_specs:
        best_params, grid = tune_base_preserving_policy(
            val_base_rows,
            full_val_df,
            val_groups,
            val_cue_info,
            score_col="full_score",
            variant=name,
        )
        base_preserving_grids.append(grid)
        write_json(protocol_dir / "configs" / f"best_{name}_params.json", best_params)
        val_preds, val_audit = predict_base_preserving(
            val_base_preds,
            full_val_df,
            val_cue_info,
            val_groups,
            best_params,
            score_col="full_score",
            variant=name,
        )
        pd.DataFrame(mb.eval_method(name + "_val_selected", val_groups, val_preds)).to_csv(
            protocol_dir / "predictions" / f"{name}_val_proxy_phrase_group_predictions.csv",
            index=False,
        )
        val_audit.to_csv(protocol_dir / "predictions" / f"{name}_val_proxy_action_audit.csv", index=False)
        eval_preds, eval_audit = predict_base_preserving(
            eval_base_preds,
            full_eval_df,
            eval_cue_info,
            eval_groups,
            best_params,
            score_col="full_score",
            variant=name,
        )
        rows, summ = summarize_multibox(name, eval_groups, eval_preds)
        rows = add_cue_columns_to_eval(rows, eval_cue_info)
        rows.to_csv(protocol_dir / "predictions" / f"{name}_eval_phrase_group_predictions.csv", index=False)
        eval_audit.to_csv(protocol_dir / "predictions" / f"{name}_eval_action_audit.csv", index=False)
        add_base_preserving_deltas(eval_base_rows, rows, eval_audit).to_csv(
            protocol_dir / "predictions" / f"{name}_per_query_delta.csv",
            index=False,
        )
        action_distribution(eval_audit).to_csv(protocol_dir / "metrics" / f"{name}_action_distribution.csv", index=False)
        summaries.append(summ)
        pred_frames.append(rows)
        eval_rows_for_diagnostics.append(rows)
        base_preserving_eval_rows[name] = rows
    if base_preserving_grids:
        pd.concat(base_preserving_grids, ignore_index=True, sort=False).to_csv(protocol_dir / "metrics" / "base_preserving_val_grid.csv", index=False)

    refs = []
    for p in [
        UNIFIED_FINEGRID / "metrics" / "summary_with_references.csv",
        CONTRASTIVE_FINEGRID / "metrics" / "summary_with_references.csv",
    ]:
        if p.exists():
            ref = pd.read_csv(p)
            ref = ref[ref["subset"].eq("eval_phrase_groups_all")].copy()
            refs.append(ref)
    ref_df = pd.concat(refs, ignore_index=True) if refs else pd.DataFrame()
    summary_df = pd.concat(summaries + ([ref_df] if len(ref_df) else []), ignore_index=True, sort=False)
    summary_df.to_csv(protocol_dir / "metrics" / "metrics_summary.csv", index=False)

    all_eval_rows = pd.concat(eval_rows_for_diagnostics, ignore_index=True, sort=False) if eval_rows_for_diagnostics else pd.DataFrame()
    if len(all_eval_rows):
        diag = pd.DataFrame([count_diagnostics(part, str(method)) for method, part in all_eval_rows.groupby("method", sort=False)])
        diag.to_csv(protocol_dir / "metrics" / "cue_variant_count_diagnostics.csv", index=False)
        subgroup_diagnostics(all_eval_rows, "multi_cue_type").to_csv(protocol_dir / "metrics" / "cue_type_metrics.csv", index=False)
        subgroup_diagnostics(all_eval_rows, "finding").to_csv(protocol_dir / "metrics" / "finding_cue_selector_metrics.csv", index=False)
        all_eval_rows.assign(gold_count=all_eval_rows["n_gt"]).pipe(subgroup_diagnostics, "gold_count").to_csv(
            protocol_dir / "metrics" / "gold_count_metrics.csv",
            index=False,
        )

    main_new_rows = base_preserving_eval_rows.get("base_preserving_full_rule")
    if main_new_rows is None:
        main_new_rows = cue_variant_eval_rows.get("conservative_cue_rescue_multibox")
    if main_new_rows is None and cue_variant_eval_rows:
        main_new_rows = list(cue_variant_eval_rows.values())[-1]
    if main_new_rows is not None:
        write_cue_audit_outputs(protocol_dir, eval_groups, eval_cue_info, full_rows, main_new_rows)
        cue_audit = pd.read_csv(protocol_dir / "cue_audit.csv")
        make_multibox_contact_sheets(protocol_dir, eval_groups, cue_audit, load_existing_finegrid_rows(), full_rows, main_new_rows)
        failure_multibox(protocol_dir, main_new_rows)
    return summary_df


def failure_singlebox(protocol_dir: Path, new_pred: pd.DataFrame) -> None:
    base_path = SINGLEBOX_SEM / "predictions" / "semantic_finegrid_singlebox_fair_eval_predictions.csv"
    if not base_path.exists():
        return
    base = pd.read_csv(base_path)[["sample_id", "iou"]].rename(columns={"iou": "baseline_iou"})
    merged = new_pred.merge(base, on="sample_id", how="left")
    merged["new_iou"] = merged["iou"]
    merged["category"] = "unchanged_or_mixed"
    merged.loc[merged["new_iou"] >= 0.5, "category"] = "success_final_iou_ge_0.5"
    merged.loc[(merged["baseline_iou"] < 0.5) & (merged["new_iou"] >= 0.5), "category"] = "reranker_rescued"
    merged.loc[(merged["baseline_iou"] >= 0.5) & (merged["new_iou"] < 0.5), "category"] = "reranker_hurt"
    merged.to_csv(protocol_dir / "failure_analysis_before_after.csv", index=False)
    merged[merged["category"].eq("reranker_rescued")].to_csv(protocol_dir / "rescue_cases.csv", index=False)
    merged[merged["category"].eq("reranker_hurt")].to_csv(protocol_dir / "hurt_cases.csv", index=False)


def failure_multibox(protocol_dir: Path, new_rows: pd.DataFrame) -> None:
    base_path = CONTRASTIVE_FINEGRID / "predictions" / "finegrid_plus_contrastive_global_eval_phrase_group_predictions.csv"
    if not base_path.exists():
        return
    base = pd.read_csv(base_path)[["group_id", "coverage_mean_iou", "set_f1_0_3"]].rename(
        columns={"coverage_mean_iou": "baseline_coverage_mean_iou", "set_f1_0_3": "baseline_set_f1_0_3"}
    )
    merged = new_rows.merge(base, on="group_id", how="left")
    merged["category"] = "unchanged_or_mixed"
    merged.loc[(merged["baseline_coverage_mean_iou"] < 0.5) & (merged["coverage_mean_iou"] >= 0.5), "category"] = "reranker_rescued_multibox"
    merged.loc[(merged["baseline_coverage_mean_iou"] >= 0.5) & (merged["coverage_mean_iou"] < 0.5), "category"] = "reranker_hurt_multibox"
    merged.to_csv(protocol_dir / "failure_analysis_before_after.csv", index=False)
    merged[merged["category"].eq("reranker_rescued_multibox")].to_csv(protocol_dir / "rescue_cases.csv", index=False)
    merged[merged["category"].eq("reranker_hurt_multibox")].to_csv(protocol_dir / "hurt_cases.csv", index=False)


def write_readme(root: Path, single_summary: pd.DataFrame | None, multi_summary: pd.DataFrame | None, args: argparse.Namespace) -> None:
    lines = [
        "# Rerank Dino Gate Experiment",
        "",
        "## Protocol separation",
        "",
        "- `singlebox_888`: one query instance, one output box. Use only for MedRPG-compatible interpretation.",
        "- `multibox_1444`: phrase group can have one or more output boxes. Use only for MedGrounder-compatible interpretation.",
        "- Chest ImaGenome was not used in this experiment.",
        "- Train split fits learned reranker; val split selects gate/threshold; eval split is final reporting only.",
        "",
        "## Candidate policy",
        "",
        "- Top-K candidates are preserved from multiple cues before scoring.",
        "- RAD-DINO agreement is a reliability-gated boost, not a hard penalty.",
        "- Agreement-conditioned blending averages boxes only when candidate positions are already close.",
        "",
    ]
    if single_summary is not None:
        lines += ["## Singlebox 888 summary", "", single_summary.to_markdown(index=False), ""]
    if multi_summary is not None:
        lines += ["## Multibox 1444 summary", "", multi_summary.to_markdown(index=False), ""]
    lines += [
        "## Notes",
        "",
        "- Top-K oracle rows are upper bounds and must not be reported as method performance.",
        "- Singlebox and multibox numbers must not be mixed in one ranking table.",
    ]
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="")
    parser.add_argument("--protocol", choices=["both", "singlebox_888", "multibox_1444"], default="both")
    parser.add_argument("--singlebox-tags", default=",".join(DEFAULT_SINGLEBOX_TAGS))
    parser.add_argument("--allow-cache-fill", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--reranker-model", choices=["hgb", "rf"], default="hgb")
    args = parser.parse_args()

    root = Path(args.output_root) if args.output_root else PROJECT_ROOT / "experiments" / "rerank_dino_gate" / now_stamp()
    ensure_dirs([root])
    single_summary = None
    multi_summary = None
    if args.protocol in {"both", "singlebox_888"}:
        single_summary = run_singlebox(root / "singlebox_888", args)
    if args.protocol in {"both", "multibox_1444"}:
        multi_summary = run_multibox(root / "multibox_1444", args)
    write_readme(root, single_summary, multi_summary, args)

    print("project_root", PROJECT_ROOT)
    print("output_root", root)
    if single_summary is not None:
        print("singlebox_summary", root / "singlebox_888" / "metrics" / "metrics_summary.csv")
        print(single_summary.head(8).to_string(index=False))
    if multi_summary is not None:
        print("multibox_summary", root / "multibox_1444" / "metrics" / "metrics_summary.csv")
        cols = [c for c in ["method", "subset", "n_groups", "coverage_mean_iou", "union_iou", "set_f1_0_3", "set_f1_0_5"] if c in multi_summary.columns]
        print(multi_summary[cols].head(12).to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""MS-CXR YOLO rule-context v2 reranker.

This is a follow-up to run_ms_cxr_yolo_rule_context_v1.py.  It keeps YOLO
weights fixed and tries to extract more value from the existing detector
outputs without using eval gold for tuning:

* combine YOLOv8n and YOLOv8s low-confidence candidates;
* tune scoring weights on the validation split only;
* optionally choose separate scoring weights per finding from the same val grid;
* keep the method as a post-processing/query-control experiment, not a new
  end-to-end phrase grounding model.

MS-CXR boxes are phrase-grounding boxes, not pixel-level lesion masks.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_ms_cxr_yolo_rule_context_v1 as base  # noqa: E402


EXP_NAME = "ms_cxr_yolo_rule_context_v2"
EXP = PROJECT_ROOT / "experiments" / EXP_NAME
PRED = EXP / "predictions"
MET = EXP / "metrics"
CFG = EXP / "configs"
REPORT = PROJECT_ROOT / "reports" / EXP_NAME

V1_PRED = PROJECT_ROOT / "experiments" / "ms_cxr_yolo_rule_context_v1" / "predictions"
V1_MET = PROJECT_ROOT / "experiments" / "ms_cxr_yolo_rule_context_v1" / "metrics"
YOLO_MET = PROJECT_ROOT / "experiments" / "ms_cxr_yolo_detector_v1" / "metrics"
STAGE2_MET = PROJECT_ROOT / "experiments" / "ms_cxr_context_vfm_localizer_stage2_context_finetune" / "metrics"


def ensure_dirs() -> None:
    for p in [EXP, PRED, MET, CFG, REPORT]:
        p.mkdir(parents=True, exist_ok=True)


def read_candidates(split: str, model_tags: Sequence[str]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for tag in model_tags:
        path = V1_PRED / f"{tag}_{split}_conf0p001_all_candidates.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        df = df.sort_values(["dicom_id", "class_id", "score"], ascending=[True, True, False])
        rank_by_key: Dict[Tuple[str, int], int] = defaultdict(int)
        for _, r in df.iterrows():
            dicom = str(r["dicom_id"])
            cls = int(r["class_id"])
            key = (dicom, cls)
            rank = rank_by_key[key]
            rank_by_key[key] += 1
            grouped[dicom].append(
                {
                    "class_id": cls,
                    "score": float(r["score"]),
                    "box": [float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])],
                    "source_model": tag,
                    "rank": rank,
                }
            )
    return grouped


def location_region_v2(q: Dict[str, str], side_mode: str) -> Tuple[float, float, float, float]:
    lat = q.get("laterality", "unknown")
    vert = q.get("vertical", "unknown")
    finding = q.get("finding", "")

    if side_mode == "image_right":
        right_region = (0.47, 0.97)
        left_region = (0.03, 0.53)
    else:
        right_region = (0.03, 0.53)
        left_region = (0.47, 0.97)

    if lat == "right":
        x1, x2 = right_region
    elif lat == "left":
        x1, x2 = left_region
    elif lat == "bilateral":
        x1, x2 = 0.04, 0.96
    else:
        x1, x2 = 0.05, 0.95

    if vert in {"apical", "upper"}:
        y1, y2 = 0.03, 0.40
    elif vert == "mid":
        y1, y2 = 0.20, 0.70
    elif vert in {"lower", "basal"}:
        y1, y2 = 0.42, 0.94
    elif vert == "whole":
        y1, y2 = 0.06, 0.94
    else:
        y1, y2 = 0.05, 0.94

    if finding == "Cardiomegaly":
        x1, x2, y1, y2 = 0.20, 0.82, 0.34, 0.90
    elif finding == "Edema" and q.get("laterality") == "unknown":
        x1, x2, y1, y2 = 0.08, 0.92, 0.18, 0.86
    return x1, y1, x2, y2


def center_region_score_v2(box_norm: Sequence[float], q: Dict[str, str], side_mode: str) -> float:
    cx, cy = float(box_norm[0]), float(box_norm[1])
    x1, y1, x2, y2 = location_region_v2(q, side_mode)
    tx = min(max(cx, x1), x2)
    ty = min(max(cy, y1), y2)
    dx = (cx - tx) / max(1e-6, x2 - x1)
    dy = (cy - ty) / max(1e-6, y2 - y1)
    inside = 1.0 if x1 <= cx <= x2 and y1 <= cy <= y2 else 0.0
    dist = math.sqrt(dx * dx + dy * dy)
    return float(inside + math.exp(-3.0 * dist))


def template_box_v2(q: Dict[str, str], side_mode: str) -> np.ndarray:
    x1, y1, x2, y2 = location_region_v2(q, side_mode)
    finding = q.get("finding", "")
    if finding == "Pneumothorax":
        w, h = min(0.34, x2 - x1), min(0.26, y2 - y1)
    elif finding == "Pleural Effusion":
        w, h = min(0.42, x2 - x1), min(0.32, y2 - y1)
        cy = max((y1 + y2) / 2.0, 0.72)
        return base.sanitize_norm([(x1 + x2) / 2.0, cy, w, h])
    elif finding == "Cardiomegaly":
        w, h = 0.58, 0.36
    elif finding == "Edema":
        w, h = min(0.74, x2 - x1), min(0.54, y2 - y1)
    elif finding in {"Lung Opacity", "Consolidation", "Pneumonia", "Atelectasis"}:
        w, h = min(0.48, x2 - x1), min(0.42, y2 - y1)
    else:
        w, h = min(0.42, x2 - x1), min(0.36, y2 - y1)
    return base.sanitize_norm([(x1 + x2) / 2.0, (y1 + y2) / 2.0, w, h])


def target_prior_for_row(row: Dict, priors: Dict, params: Dict) -> Tuple[Dict[str, str], np.ndarray]:
    q = base.parse_rule_context(row)
    train_prior = base.lookup_prior(priors, q)
    templ = template_box_v2(q, str(params.get("side_mode", "radiology_right")))
    return q, base.blend_norm(train_prior, templ, float(params.get("prior_train_weight", 0.75)))


def choose_candidate_v2(row: Dict, candidates_by_dicom: Dict[str, List[Dict]], priors: Dict, params: Dict) -> Tuple[List[float], float, bool, Dict]:
    iw, ih = float(row["image_width"]), float(row["image_height"])
    q, target_prior = target_prior_for_row(row, priors, params)
    class_id = base.CLASS_TO_ID[row["finding"]]
    max_rank = int(params.get("max_rank", 30))
    allowed_models = set(str(params.get("model_mode", "both")).split("+"))
    # "both" originally meant the old yolov8n+yolov8s pool.  The detector
    # sweep also uses yolov8m/yolo11*, so treat "both" as "all sources in the
    # provided candidate pool"; otherwise newer detector candidates are
    # silently filtered out and the evaluator falls back to priors.
    allow_all_models = bool({"both", "all", "any"} & allowed_models)
    query_side = str(row.get("query_laterality", "unknown"))
    allowed_laterality = {
        "right": {"right", "central"},
        "left": {"left", "central"},
        "bilateral": {"right", "left", "central"},
    }.get(query_side, {"right", "left", "central", ""})
    cands = [
        c
        for c in candidates_by_dicom.get(str(row["dicom_id"]), [])
        if int(c["class_id"]) == class_id
        and (not str(c.get("candidate_laterality_class", "")) or str(c.get("candidate_laterality_class", "")) in allowed_laterality)
        and float(c["score"]) >= float(params["conf"])
        and int(c.get("rank", 9999)) < max_rank
        and (allow_all_models or str(c.get("source_model", "")) in allowed_models)
    ]
    if not cands:
        if params.get("fallback") == "prior":
            return base.norm_to_xyxy(target_prior, iw, ih), 0.0, False, {"source": "prior_fallback", **q}
        return [0.0, 0.0, 0.0, 0.0], 0.0, True, {"source": "missing", **q}

    scored = []
    side_mode = str(params.get("side_mode", "radiology_right"))
    for cand in cands:
        box_norm = base.xyxy_to_norm(cand["box"], iw, ih)
        conf_score = math.log1p(20.0 * max(0.0, float(cand["score"])))
        region = center_region_score_v2(box_norm, q, side_mode)
        prior_iou = base.iou_norm(box_norm, target_prior)
        rank_bonus = 1.0 / (1.0 + float(cand.get("rank", 0)))
        area = max(1e-6, float(box_norm[2] * box_norm[3]))
        prior_area = max(1e-6, float(target_prior[2] * target_prior[3]))
        area_penalty = abs(math.log(area / prior_area))
        source_bias = float(params.get("w_yolov8s_bias", 0.0)) if cand.get("source_model") == "yolov8s" else 0.0
        total = (
            float(params["w_conf"]) * conf_score
            + float(params["w_region"]) * region
            + float(params["w_prior"]) * prior_iou
            + float(params["w_rank"]) * rank_bonus
            + source_bias
            - float(params["w_area"]) * area_penalty
        )
        scored.append((total, cand, box_norm, region, prior_iou, rank_bonus, area_penalty))
    scored.sort(key=lambda x: x[0], reverse=True)
    total, cand, box_norm, region, prior_iou, rank_bonus, area_penalty = scored[0]
    blend_weight = float(params.get("blend_yolo_weight", 1.0))
    final_norm = base.blend_norm(box_norm, target_prior, blend_weight)
    return base.norm_to_xyxy(final_norm, iw, ih), float(cand["score"]), False, {
        "source": "yolo_candidate_v2",
        "source_model": cand.get("source_model", ""),
        "candidate_rank": int(cand.get("rank", -1)),
        "n_candidates": len(cands),
        "rerank_score": total,
        "region_score": region,
        "prior_iou": prior_iou,
        "rank_bonus": rank_bonus,
        "area_penalty": area_penalty,
        **q,
    }


def evaluate_rows_v2(rows: List[Dict], candidates: Dict[str, List[Dict]], priors: Dict, params_by_finding: Dict[str, Dict], method: str, split: str) -> pd.DataFrame:
    out = []
    for row in rows:
        params = params_by_finding.get(str(row["finding"]), params_by_finding["__global__"])
        pred, conf, missing, info = choose_candidate_v2(row, candidates, priors, params)
        gt = base.clip_box(row["gold_bbox_xyxy"], row["image_width"], row["image_height"])
        iou = 0.0 if missing else base.iou_xyxy(pred, gt)
        out.append(
            {
                "task_id": row["task_id"],
                "sample_id": row["task_id"],
                "split": split,
                "dicom_id": row["dicom_id"],
                "subject_id": row.get("subject_id", ""),
                "study_id": row.get("study_id", ""),
                "image_path": row["image_path"],
                "finding": row["finding"],
                "class_id": base.CLASS_TO_ID[row["finding"]],
                "claim_sentence": row.get("claim_sentence", row.get("phrase", "")),
                "gt_x1": gt[0],
                "gt_y1": gt[1],
                "gt_x2": gt[2],
                "gt_y2": gt[3],
                "pred_x1": pred[0],
                "pred_y1": pred[1],
                "pred_x2": pred[2],
                "pred_y2": pred[3],
                "confidence": conf,
                "iou": iou,
                "hit_0_1": iou >= 0.1,
                "hit_0_3": iou >= 0.3,
                "hit_0_5": iou >= 0.5,
                "bbox_missing": bool(missing),
                "bbox_invalid": (not missing) and (pred[2] <= pred[0] or pred[3] <= pred[1]),
                "method": method,
                "bbox_type": "ms_cxr_phrase_grounding_bbox",
                **info,
            }
        )
    return pd.DataFrame(out)


def metrics(df: pd.DataFrame, method: str, subset: str) -> Dict:
    return base.metrics_from_predictions(df, method, subset)


def build_grid(quick: bool) -> List[Dict]:
    confs = [0.001, 0.003, 0.01]
    w_conf = [0.6, 1.0]
    w_region = [0.5, 1.5, 2.5]
    w_prior = [0.5, 1.5, 2.5]
    w_rank = [0.0, 0.3]
    w_area = [0.0, 0.15]
    blend = [1.0, 0.85, 0.7]
    model_modes = ["yolov8n", "both"]
    side_modes = ["radiology_right", "image_right"]
    max_ranks = [20, 80]
    prior_weights = [0.65, 0.85]
    if quick:
        confs = [0.001, 0.01]
        w_conf = [0.8]
        w_region = [0.5, 1.5]
        w_prior = [0.5, 1.5]
        w_rank = [0.0, 0.3]
        w_area = [0.0, 0.1]
        blend = [1.0, 0.8]
        model_modes = ["yolov8n", "both"]
        side_modes = ["radiology_right"]
        max_ranks = [30]
        prior_weights = [0.75]
    grid = []
    for conf in confs:
        for wc in w_conf:
            for wr in w_region:
                for wp in w_prior:
                    for wk in w_rank:
                        for wa in w_area:
                            for by in blend:
                                for mm in model_modes:
                                    for sm in side_modes:
                                        for mr in max_ranks:
                                            for ptw in prior_weights:
                                                grid.append(
                                                    {
                                                        "conf": conf,
                                                        "w_conf": wc,
                                                        "w_region": wr,
                                                        "w_prior": wp,
                                                        "w_rank": wk,
                                                        "w_area": wa,
                                                        "blend_yolo_weight": by,
                                                        "model_mode": mm,
                                                        "side_mode": sm,
                                                        "max_rank": mr,
                                                        "prior_train_weight": ptw,
                                                        "w_yolov8s_bias": 0.0,
                                                        "fallback": "prior",
                                                    }
                                                )
    return grid


def tune(val_rows: List[Dict], candidates: Dict[str, List[Dict]], priors: Dict, quick: bool) -> Tuple[Dict[str, Dict], pd.DataFrame, pd.DataFrame]:
    grid = build_grid(quick)
    grid_rows = []
    pred_cache: Dict[int, pd.DataFrame] = {}
    best_global_idx = -1
    best_global = -1.0
    for idx, params in enumerate(grid):
        pred = evaluate_rows_v2(val_rows, candidates, priors, {"__global__": params}, "val_grid", "val")
        pred_cache[idx] = pred[["task_id", "finding", "iou", "hit_0_3", "hit_0_5"]].copy()
        m = metrics(pred, "val_grid", "all8")
        m.update(params)
        m["grid_index"] = idx
        grid_rows.append(m)
        if float(m["mean_iou"]) > best_global:
            best_global = float(m["mean_iou"])
            best_global_idx = idx
    grid_df = pd.DataFrame(grid_rows).sort_values("mean_iou", ascending=False)

    params_by_finding = {"__global__": dict(grid[best_global_idx])}
    per_rows = []
    val_df = pd.concat(pred_cache.values(), keys=pred_cache.keys(), names=["grid_index", "row"]).reset_index(level=0)
    for finding in base.CLASS_NAMES:
        sub = val_df[val_df["finding"] == finding]
        if len(sub) < 5:
            params_by_finding[finding] = dict(grid[best_global_idx])
            continue
        by_grid = sub.groupby("grid_index")["iou"].mean().sort_values(ascending=False)
        best_idx = int(by_grid.index[0])
        params_by_finding[finding] = dict(grid[best_idx])
        per_rows.append(
            {
                "finding": finding,
                "val_rows": int(len(sub[sub["grid_index"] == best_idx])),
                "best_grid_index": best_idx,
                "val_mean_iou": float(by_grid.iloc[0]),
                "global_grid_index": best_global_idx,
            }
        )
    per_df = pd.DataFrame(per_rows)
    return params_by_finding, grid_df, per_df


def add_singlebox_metrics(eval_rows: List[Dict], pred: pd.DataFrame, method: str) -> Optional[Dict]:
    single_rows = base.build_singlebox_rows(eval_rows)
    if not single_rows:
        return None
    task_ids = {r["task_id"] for r in single_rows}
    sub = pred[pred["task_id"].isin(task_ids)].copy()
    if len(sub) == 0:
        return None
    return metrics(sub, method, "singlebox")


def comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    frames = [summary]
    for path in [
        YOLO_MET / "final_same_split_yolo_vs_query_baselines.csv",
        V1_MET / "yolov8n_same_split_yolo_rule_context_comparison.csv",
        STAGE2_MET / "summary_all8.csv",
    ]:
        if path.exists():
            try:
                df = pd.read_csv(path)
                frames.append(df)
            except Exception:
                pass
    return pd.concat(frames, ignore_index=True, sort=False)


def write_report(params_by_finding: Dict[str, Dict], summary: pd.DataFrame, grid: pd.DataFrame, per_find: pd.DataFrame, comparison: pd.DataFrame) -> None:
    lines = [
        "# MS-CXR YOLO Rule-Context v2",
        "",
        "## One-line conclusion",
        "",
        "YOLO detector weights are fixed.  This experiment uses validation-only tuning to rerank YOLOv8n/YOLOv8s low-confidence candidates with structured rule-context priors.  MS-CXR boxes are phrase-grounding boxes, not pixel-level lesion masks.",
        "",
        "## Output summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Per-finding val-selected params",
        "",
        per_find.to_markdown(index=False) if len(per_find) else "(no per-finding params)",
        "",
        "## Top validation grid rows",
        "",
        grid.head(15).to_markdown(index=False),
        "",
        "## Comparison table",
        "",
        comparison.to_markdown(index=False),
        "",
        "## Global params",
        "",
        "```json",
        json.dumps(params_by_finding.get("__global__", {}), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Interpretation guardrails",
        "",
        "- The detector is not text-conditioned internally; rule-context is used as post-processing.",
        "- All scoring weights are selected on val only.",
        "- This is a performance/ablation extension, not an end-to-end phrase grounding model.",
        "- Do not call MS-CXR boxes lesion masks.",
    ]
    (REPORT / "README_KO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    train_rows = base.source_rows("train")
    val_rows = base.source_rows("val")
    eval_rows = base.source_rows("eval")
    priors = base.make_train_priors(train_rows)
    val_candidates = read_candidates("val", ["yolov8n", "yolov8s"])
    eval_candidates = read_candidates("eval", ["yolov8n", "yolov8s"])

    params_by_finding, grid, per_find = tune(val_rows, val_candidates, priors, quick=args.quick)
    grid.to_csv(MET / "val_grid_search.csv", index=False)
    per_find.to_csv(MET / "per_finding_val_params.csv", index=False)
    (CFG / "best_params_by_finding.json").write_text(json.dumps(params_by_finding, ensure_ascii=False, indent=2), encoding="utf-8")

    method = "yolo_rule_context_v2_per_finding"
    pred = evaluate_rows_v2(eval_rows, eval_candidates, priors, params_by_finding, method, "eval")
    pred.to_csv(PRED / "yolo_rule_context_v2_eval_predictions.csv", index=False)
    summary_rows = [metrics(pred, method, "all8"), metrics(pred, method, "main5")]
    single = add_singlebox_metrics(eval_rows, pred, method)
    if single:
        summary_rows.append(single)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(MET / "yolo_rule_context_v2_summary.csv", index=False)

    per = pred.groupby("finding")["iou"].agg(["count", "mean", "median"]).reset_index()
    per["Hit@0.3"] = pred.groupby("finding")["hit_0_3"].mean().values
    per["Hit@0.5"] = pred.groupby("finding")["hit_0_5"].mean().values
    per.to_csv(MET / "yolo_rule_context_v2_per_finding.csv", index=False)

    comp = comparison_table(summary)
    comp.to_csv(MET / "comparison_with_existing_baselines.csv", index=False)
    write_report(params_by_finding, summary, grid, per_find, comp)

    print(f"project_root={PROJECT_ROOT}")
    print(f"train_rows={len(train_rows)}")
    print(f"val_rows={len(val_rows)}")
    print(f"eval_rows={len(eval_rows)}")
    print(f"best_val_mean_iou={float(grid.iloc[0]['mean_iou']):.6f}")
    print(f"eval_mean_iou_all8={float(summary.iloc[0]['mean_iou']):.6f}")
    print(f"eval_hit03_all8={float(summary.iloc[0]['Hit@0.3']):.6f}")
    print(f"eval_mean_iou_main5={float(summary.iloc[1]['mean_iou']):.6f}")
    print(f"summary_path={MET / 'yolo_rule_context_v2_summary.csv'}")
    print(f"report_path={REPORT / 'README_KO.md'}")


if __name__ == "__main__":
    main()

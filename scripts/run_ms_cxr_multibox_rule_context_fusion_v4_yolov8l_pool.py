#!/usr/bin/env python
"""MS-CXR rule-context multi-box set prediction v3 with a larger YOLO pool.

This experiment fixes a logical weakness in the earlier single-box fusion
line.  Some MS-CXR phrases map to more than one phrase-grounding box, e.g.
"bilateral pneumothoraces" or "bibasilar opacities".  A method that always
returns one box is structurally unable to cover both boxes.

The method below does not use eval gold to decide how many boxes to emit.
It predicts a set size from the claim text only, then selects diverse same-
class YOLO candidates using rule-context and optional RAD-DINO agreement.

V3 keeps the V2 protocol and changes only the detector candidate pool.  By
default it evaluates yolov8n + yolov8s + yolov8m candidates, with all set-size
and fusion parameters still selected on validation groups only.

MS-CXR boxes are phrase-grounding boxes, not pixel-level lesion masks.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_ms_cxr_multibox_phrase_grounding_v1 as mb  # noqa: E402
from scripts import run_ms_cxr_yolo_rule_context_v1 as base  # noqa: E402
from scripts import run_ms_cxr_yolo_rule_context_v2 as yv2  # noqa: E402


EXP_NAME = "ms_cxr_multibox_rule_context_fusion_v4_yolov8l_pool"
EXP = PROJECT_ROOT / "experiments" / EXP_NAME
PRED = EXP / "predictions"
MET = EXP / "metrics"
CFG = EXP / "configs"
REPORT = PROJECT_ROOT / "reports" / EXP_NAME
CONTACT = EXP / "contact_sheets"

STAGE1_DATA = PROJECT_ROOT / "training" / "ms_cxr_vfm_localizer_stage1_p10_p19" / "datasets"
FUSION_PRED = PROJECT_ROOT / "experiments" / "ms_cxr_yolo_dino_rule_fusion_v1" / "predictions"
V2_PHRASE_PRED = PROJECT_ROOT / "experiments" / "ms_cxr_multibox_rule_context_fusion_v2" / "predictions" / "phrase_group_set_predictions.csv"
YOLO_V2_PRED = PROJECT_ROOT / "experiments" / "ms_cxr_yolo_rule_context_v2" / "predictions"
MEDRPG_ROW_PRED = PROJECT_ROOT / "experiments" / "ms_cxr_medrpg_rowlevel_fair_retrain_v1" / "predictions"
YOLO_V2_CFG = PROJECT_ROOT / "experiments" / "ms_cxr_yolo_rule_context_v2" / "configs" / "best_params_by_finding.json"
FUSION_CFG = PROJECT_ROOT / "experiments" / "ms_cxr_yolo_dino_rule_fusion_v1" / "configs" / "best_fusion_by_finding.json"


def ensure_dirs() -> None:
    for p in [EXP, PRED, MET, CFG, REPORT, CONTACT]:
        p.mkdir(parents=True, exist_ok=True)


def load_rows(split: str) -> list[dict[str, Any]]:
    return mb.load_jsonl(STAGE1_DATA / f"{split}.jsonl")


def norm_claim(text: str) -> str:
    return mb.norm_text(text)


def group_row(g: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": g["task_ids"][0],
        "dicom_id": g["dicom_id"],
        "subject_id": g.get("subject_id", ""),
        "study_id": g.get("study_id", ""),
        "image_path": g["image_path"],
        "finding": g["finding"],
        "claim_sentence": g["claim_sentence"],
        "image_width": g["image_width"],
        "image_height": g["image_height"],
        "split": g["split"],
    }


def with_q(base_q: dict[str, str], **updates: str) -> dict[str, str]:
    q = dict(base_q)
    for k, v in updates.items():
        if v:
            q[k] = v
    return q


def context_cues(claim: str, finding: str, base_q: dict[str, str]) -> dict[str, Any]:
    """Infer possible multi-box targets from claim text only.

    The output is intentionally conservative.  Explicit bilateral/both/bibasal
    cues create side-specific targets.  Diffuse/multifocal/scattered cues only
    raise the diverse top-k limit.
    """

    text = norm_claim(claim)
    cues: list[str] = []
    target_qs: list[dict[str, str]] = []
    k_hint = 1

    bilateral = bool(
        re.search(
            r"\b(bilateral|bilaterally|both|bibasilar|bibasal|pneumothoraces)\b",
            text,
        )
        or re.search(r"\bright\s+(greater|more)\s+than\s+left\b", text)
        or re.search(r"\bleft\s+(greater|more)\s+than\s+right\b", text)
        or re.search(r"\bright\s+and\s+left\b|\bleft\s+and\s+right\b", text)
        or re.search(r"\bupper\s+lobes\b|\blower\s+lobes\b|\bbases\b", text)
    )
    plural_bilateral_like = bool(re.search(r"\b(effusions|opacities|consolidations|infiltrates)\b", text))
    if bilateral or (plural_bilateral_like and re.search(r"\bsmall\s+pleural\s+effusions\b|\bpleural\s+effusions\b", text)):
        cues.append("bilateral_or_plural")
        vert = base_q.get("vertical", "unknown")
        if "bibasilar" in text or "bibasal" in text or "bases" in text:
            vert = "basal"
        target_qs.extend([
            with_q(base_q, laterality="right", vertical=vert),
            with_q(base_q, laterality="left", vertical=vert),
        ])
        k_hint = max(k_hint, 2)

    # Same-side upper/lower phrases can be annotated as one long phrase box in
    # MS-CXR.  Treat them as multi-box only when the sentence also contains a
    # clear cross-side cue.  This avoids turning "right middle and lower lobe"
    # into two boxes when the annotation is one covering box.
    cross_side = bool(re.search(r"\b(right|left)\b.*\b(left|right)\b", text))
    for lat_word in ["right", "left"]:
        if cross_side and re.search(rf"\b{lat_word}\s+(upper|apical)\s+and\s+(lower|basilar|basal)\b", text):
            cues.append(f"{lat_word}_upper_lower")
            target_qs.extend([
                with_q(base_q, laterality=lat_word, vertical="upper"),
                with_q(base_q, laterality=lat_word, vertical="lower"),
            ])
            k_hint = max(k_hint, 2)
        if cross_side and re.search(rf"\b{lat_word}\s+(middle|mid)\s+and\s+(lower|basilar|basal)\b", text):
            cues.append(f"{lat_word}_mid_lower")
            target_qs.extend([
                with_q(base_q, laterality=lat_word, vertical="mid"),
                with_q(base_q, laterality=lat_word, vertical="lower"),
            ])
            k_hint = max(k_hint, 2)

    if cross_side and re.search(r"\b(upper|apical)\s+and\s+(lower|basilar|basal)\b", text) and not target_qs:
        cues.append("upper_lower")
        target_qs.extend([
            with_q(base_q, vertical="upper"),
            with_q(base_q, vertical="lower"),
        ])
        k_hint = max(k_hint, 2)

    # "Patchy" alone often still maps to one MS-CXR phrase box.  Keep stronger
    # distribution words, but do not let patchy by itself trigger extra boxes.
    diffuse = bool(re.search(r"\b(multifocal|multisegmental|multilobar|multiple|scattered|diffuse|widespread|extensive|several)\b", text))
    if diffuse:
        cues.append("diffuse_or_multifocal")
        k_hint = max(k_hint, 3 if re.search(r"\b(diffuse|widespread|extensive|multifocal)\b", text) else 2)

    if finding in {"Edema", "Pneumonia"} and re.search(r"\b(perihilar|pulmonary edema|septal|interstitial)\b", text):
        cues.append("finding_distribution_prior")
        k_hint = max(k_hint, 2)

    # Keep duplicated target specs from over-counting.
    unique: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for q in target_qs:
        key = (q.get("finding", ""), q.get("laterality", ""), q.get("vertical", ""))
        if key not in seen:
            unique.append(q)
            seen.add(key)

    return {
        "cue_text": ";".join(cues) if cues else "single_or_unspecified",
        "target_qs": unique,
        "k_hint": min(max(k_hint, len(unique), 1), 4),
        "has_multi_cue": bool(cues),
    }


def target_prior(q: dict[str, str], priors: dict[str, dict[tuple, np.ndarray]], params: dict[str, Any]) -> np.ndarray:
    train_prior = base.lookup_prior(priors, q)
    templ = yv2.template_box_v2(q, str(params.get("side_mode", "radiology_right")))
    return base.blend_norm(train_prior, templ, float(params.get("prior_train_weight", 0.75)))


def load_dino_by_group(split: str, groups: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
    path = FUSION_PRED / f"rad_dino_rule_context_{split}_predictions.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    by_task: dict[str, np.ndarray] = {}
    for _, r in df.iterrows():
        if {"pred_cx", "pred_cy", "pred_w", "pred_h"}.issubset(df.columns):
            by_task[str(r["task_id"])] = base.sanitize_norm([r["pred_cx"], r["pred_cy"], r["pred_w"], r["pred_h"]])
        else:
            iw = float(r.get("image_width", 1.0))
            ih = float(r.get("image_height", 1.0))
            by_task[str(r["task_id"])] = base.xyxy_to_norm([r["pred_x1"], r["pred_y1"], r["pred_x2"], r["pred_y2"]], iw, ih)
    out: dict[str, np.ndarray] = {}
    for gid, g in groups.items():
        for task_id in g["task_ids"]:
            if task_id in by_task:
                out[gid] = by_task[task_id]
                break
    return out


def score_candidates(
    g: dict[str, Any],
    q: dict[str, str],
    cands: list[dict[str, Any]],
    priors: dict[str, dict[tuple, np.ndarray]],
    params: dict[str, Any],
    dino_norm: np.ndarray | None,
    dino_weight: float,
) -> list[dict[str, Any]]:
    iw, ih = float(g["image_width"]), float(g["image_height"])
    prior = target_prior(q, priors, params)
    side_mode = str(params.get("side_mode", "radiology_right"))
    scored: list[dict[str, Any]] = []
    for cand in cands:
        box_norm = base.xyxy_to_norm(cand["box"], iw, ih)
        conf_score = math.log1p(20.0 * max(0.0, float(cand["score"])))
        region = yv2.center_region_score_v2(box_norm, q, side_mode)
        prior_iou = base.iou_norm(box_norm, prior)
        rank_bonus = 1.0 / (1.0 + float(cand.get("rank", 0)))
        area = max(1e-6, float(box_norm[2] * box_norm[3]))
        prior_area = max(1e-6, float(prior[2] * prior[3]))
        area_penalty = abs(math.log(area / prior_area))
        dino_iou = base.iou_norm(box_norm, dino_norm) if dino_norm is not None else 0.0
        total = (
            float(params.get("w_conf", 0.8)) * conf_score
            + float(params.get("w_region", 0.5)) * region
            + float(params.get("w_prior", 0.5)) * prior_iou
            + float(params.get("w_rank", 0.0)) * rank_bonus
            + float(dino_weight) * dino_iou
            - float(params.get("w_area", 0.0)) * area_penalty
        )
        scored.append({
            **cand,
            "score": total,
            "raw_conf": float(cand["score"]),
            "box_norm": box_norm.tolist(),
            "query_laterality": q.get("laterality", ""),
            "query_vertical": q.get("vertical", ""),
            "region_score": region,
            "prior_iou": prior_iou,
            "dino_iou": dino_iou,
            "rank_bonus": rank_bonus,
            "area_penalty": area_penalty,
        })
    return sorted(scored, key=lambda x: float(x["score"]), reverse=True)


def add_if_distinct(selected: list[dict[str, Any]], cand: dict[str, Any], nms_iou: float) -> bool:
    if all(mb.iou_xyxy(cand["box"], old["box"]) < nms_iou for old in selected):
        selected.append(cand)
        return True
    return False


def predict_group_set(
    g: dict[str, Any],
    candidates_by_dicom: dict[str, list[dict[str, Any]]],
    priors: dict[str, dict[tuple, np.ndarray]],
    yolo_params_by_finding: dict[str, dict[str, Any]],
    dino_norm: np.ndarray | None,
    grid_params: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = group_row(g)
    base_q = base.parse_rule_context(row)
    cue = context_cues(g["claim_sentence"], g["finding"], base_q)
    params = dict(yolo_params_by_finding.get(g["finding"], yolo_params_by_finding.get("__global__", {})))
    class_id = base.CLASS_TO_ID[g["finding"]]
    max_rank = int(grid_params.get("max_rank", params.get("max_rank", 30)))
    conf = float(grid_params.get("conf", params.get("conf", 0.001)))
    raw_cands = [
        c for c in candidates_by_dicom.get(str(g["dicom_id"]), [])
        if int(c["class_id"]) == class_id and float(c["score"]) >= conf and int(c.get("rank", 9999)) < max_rank
    ]
    if not raw_cands:
        return [], {**cue, "prediction_rule": "no_candidate", "available_candidates": 0}

    nms_iou = float(grid_params["nms_iou"])
    dino_weight = float(grid_params["dino_weight"])
    top_ratio = float(grid_params["extra_score_ratio"])
    max_k = 1
    if cue["has_multi_cue"]:
        max_k = max(int(cue["k_hint"]), int(grid_params["min_k_if_cue"]))
        max_k = min(max_k, int(grid_params["max_k_if_cue"]))

    selected: list[dict[str, Any]] = []
    # Explicit target specs first: left/right, upper/lower, etc.
    for q in cue["target_qs"]:
        scored = score_candidates(g, q, raw_cands, priors, params, dino_norm, dino_weight)
        for cand in scored[: int(grid_params["target_scan_topn"])]:
            if add_if_distinct(selected, cand, nms_iou):
                break

    # Then fill remaining slots with diverse candidates under the original query.
    scored_base = score_candidates(g, base_q, raw_cands, priors, params, dino_norm, dino_weight)
    top_score = float(scored_base[0]["score"]) if scored_base else 0.0
    min_extra = top_score * top_ratio if top_score > 0 else -1e9
    for cand in scored_base:
        if len(selected) >= max_k:
            break
        if len(selected) > 0 and float(cand["score"]) < min_extra:
            continue
        add_if_distinct(selected, cand, nms_iou)

    if not selected and scored_base:
        selected = [scored_base[0]]

    return selected[:max_k], {
        **cue,
        "prediction_rule": "cue_set" if cue["has_multi_cue"] else "single",
        "available_candidates": len(raw_cands),
        "requested_k": max_k,
    }


def predict_groups(
    groups: dict[str, dict[str, Any]],
    candidates_by_dicom: dict[str, list[dict[str, Any]]],
    priors: dict[str, dict[tuple, np.ndarray]],
    yolo_params_by_finding: dict[str, dict[str, Any]],
    dino_by_gid: dict[str, np.ndarray],
    grid_params: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    preds: dict[str, list[dict[str, Any]]] = {}
    audit: list[dict[str, Any]] = []
    for gid, g in groups.items():
        boxes, info = predict_group_set(
            g,
            candidates_by_dicom,
            priors,
            yolo_params_by_finding,
            dino_by_gid.get(gid),
            grid_params,
        )
        preds[gid] = boxes
        audit.append({
            "group_id": gid,
            "split": g["split"],
            "finding": g["finding"],
            "claim_sentence": g["claim_sentence"],
            "n_gt": len(g["gt_boxes"]),
            "n_pred": len(boxes),
            **{k: (len(v) if isinstance(v, list) else v) for k, v in info.items() if k != "target_qs"},
        })
    return preds, audit


def eval_predictions(
    method: str,
    groups: dict[str, dict[str, Any]],
    pred_by_gid: dict[str, list[dict[str, Any]]],
) -> pd.DataFrame:
    return pd.DataFrame(mb.eval_method(method, groups, pred_by_gid))


def load_singleton_methods(groups: dict[str, dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    methods: dict[str, dict[str, list[dict[str, Any]]]] = {}
    paths = {
        "rad_dino_rule_context": FUSION_PRED / "rad_dino_rule_context_eval_predictions.csv",
        "yolo_dino_rule_fusion_v1": FUSION_PRED / "yolo_dino_rule_fusion_eval_predictions.csv",
        "yolo_rule_context_v2_singleton": YOLO_V2_PRED / "yolo_rule_context_v2_eval_predictions.csv",
        "medrpg_rowlevel_full_phrase_s42": MEDRPG_ROW_PRED / "medrpg_row_level_full_phrase_s42_eval_predictions.csv",
        "medrpg_rowlevel_full_phrase_s13": MEDRPG_ROW_PRED / "medrpg_row_level_full_phrase_s13_eval_predictions.csv",
        "medrpg_rowlevel_full_phrase_s2026": MEDRPG_ROW_PRED / "medrpg_row_level_full_phrase_s2026_eval_predictions.csv",
    }
    for name, path in paths.items():
        if path.exists():
            methods[name] = mb.load_row_prediction_csv(path, name, groups)
    return methods


def detection_set_baseline(
    groups: dict[str, dict[str, Any]],
    candidates_by_dicom: dict[str, list[dict[str, Any]]],
    params: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for gid, g in groups.items():
        raw = [
            c for c in candidates_by_dicom.get(str(g["dicom_id"]), [])
            if int(c["class_id"]) == int(g["class_id"]) and float(c["score"]) >= float(params["score_thr"])
        ]
        out[gid] = mb.nms(raw, float(params["nms_iou"]))[: int(params["max_k"])]
    return out


def summarize_all(eval_rows: pd.DataFrame, groups: dict[str, dict[str, Any]], method: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subset, gids in {
        "eval_phrase_groups_all": list(groups.keys()),
        "eval_phrase_groups_main5": [gid for gid, g in groups.items() if g["finding"] in mb.MAIN5],
        "eval_phrase_groups_single_box": [gid for gid, g in groups.items() if len(g["gt_boxes"]) == 1],
        "eval_phrase_groups_multi_box": [gid for gid, g in groups.items() if len(g["gt_boxes"]) > 1],
    }.items():
        sub = eval_rows[eval_rows["group_id"].isin(gids)]
        if len(sub):
            rows.append(mb.summarize(sub.to_dict("records"), method, subset))
    return rows


def tune_set_params(
    val_groups: dict[str, dict[str, Any]],
    val_candidates: dict[str, list[dict[str, Any]]],
    priors: dict[str, dict[tuple, np.ndarray]],
    yolo_params: dict[str, dict[str, Any]],
    dino_val: dict[str, np.ndarray],
    quick: bool,
) -> tuple[dict[str, Any], pd.DataFrame]:
    grid = []
    for dino_weight in ([0.0, 0.5] if quick else [0.0, 0.25, 0.5, 1.0]):
        for nms_iou in ([0.45, 0.6] if quick else [0.35, 0.5, 0.65, 0.8]):
            for max_k_if_cue in ([2, 3] if quick else [2, 3, 4]):
                for extra_score_ratio in ([0.0, 0.65] if quick else [0.0, 0.5, 0.65, 0.8]):
                    grid.append({
                        "conf": 0.001,
                        "max_rank": 80,
                        "nms_iou": nms_iou,
                        "dino_weight": dino_weight,
                        "min_k_if_cue": 2,
                        "max_k_if_cue": max_k_if_cue,
                        "extra_score_ratio": extra_score_ratio,
                        "target_scan_topn": 30,
                    })
    rows: list[dict[str, Any]] = []
    best_params: dict[str, Any] | None = None
    best_key: tuple[float, float, float, float] | None = None
    for i, params in enumerate(grid):
        pred, _ = predict_groups(val_groups, val_candidates, priors, yolo_params, dino_val, params)
        scores = eval_predictions("val_candidate", val_groups, pred)
        all_s = mb.summarize(scores.to_dict("records"), "val_candidate", "val_all")
        multi = scores[scores["n_gt"] > 1]
        multi_s = mb.summarize(multi.to_dict("records"), "val_candidate", "val_multi") if len(multi) else {}
        row = {
            "grid_index": i,
            **params,
            "val_all_coverage_mean_iou": all_s["coverage_mean_iou"],
            "val_all_set_f1_0_3": all_s["set_f1_0_3"],
            "val_all_pred_count_abs_error": all_s["pred_count_abs_error"],
            "val_multi_coverage_mean_iou": multi_s.get("coverage_mean_iou", 0.0),
            "val_multi_set_f1_0_3": multi_s.get("set_f1_0_3", 0.0),
            "val_multi_all_gt_hit_0_3": multi_s.get("all_gt_hit_0_3", 0.0),
        }
        rows.append(row)
        key = (
            float(all_s["set_f1_0_3"]),
            float(all_s["coverage_mean_iou"]),
            float(multi_s.get("set_f1_0_3", 0.0)),
            -float(all_s["pred_count_abs_error"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_params = dict(params)
    assert best_params is not None
    return best_params, pd.DataFrame(rows).sort_values(["val_all_set_f1_0_3", "val_all_coverage_mean_iou"], ascending=False)


def bootstrap_diff(rows: pd.DataFrame, method_a: str, method_b: str, metric: str, reps: int = 1000) -> dict[str, Any]:
    piv = rows.pivot(index="group_id", columns="method", values=metric).dropna(subset=[method_a, method_b])
    ids = list(piv.index)
    if not ids:
        return {"comparison": f"{method_a} - {method_b}", "metric": metric, "n": 0}
    observed = float((piv[method_a] - piv[method_b]).mean())
    rng = random.Random(2026)
    diffs = []
    for _ in range(reps):
        sample = [rng.choice(ids) for _ in ids]
        diffs.append(float((piv.loc[sample, method_a].to_numpy() - piv.loc[sample, method_b].to_numpy()).mean()))
    diffs.sort()
    return {
        "comparison": f"{method_a} - {method_b}",
        "metric": metric,
        "n": len(ids),
        "observed_diff": observed,
        "ci95_low": diffs[int(0.025 * reps)],
        "ci95_high": diffs[min(reps - 1, int(0.975 * reps))],
        "prob_gt_0": sum(d > 0 for d in diffs) / len(diffs),
        "bootstrap_unit": "phrase_group",
        "bootstrap_reps": reps,
    }


def make_contact_sheet(groups: dict[str, dict[str, Any]], pred_rows: pd.DataFrame) -> None:
    method = "rule_context_multibox_yolo_dino_v4_yolov8l_pool"
    base_method = "yolo_dino_rule_fusion_v1"
    if method not in set(pred_rows["method"]):
        return
    piv = pred_rows.pivot(index="group_id", columns="method", values="coverage_mean_iou")
    if method not in piv:
        return
    if base_method in piv:
        order = (piv[method] - piv[base_method]).sort_values(ascending=False).index.tolist()
    else:
        order = piv[method].sort_values(ascending=False).index.tolist()
    selected = [gid for gid in order if gid in groups and len(groups[gid]["gt_boxes"]) > 1][:15]
    if not selected:
        return
    font = ImageFont.load_default()
    panels: list[Image.Image] = []
    colors = {
        "gold": (255, 220, 0),
        method: (0, 230, 90),
        base_method: (80, 140, 255),
    }
    for gid in selected:
        g = groups[gid]
        try:
            img = Image.open(g["image_path"]).convert("RGB")
        except Exception:
            continue
        w, h = img.size
        scale = 640 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)))
        draw = ImageDraw.Draw(img)
        sx, sy = img.size[0] / w, img.size[1] / h

        def rect(box: list[float], color: tuple[int, int, int], width: int = 3) -> None:
            b = [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]
            for k in range(width):
                draw.rectangle([b[0] - k, b[1] - k, b[2] + k, b[3] + k], outline=color)

        for box in g["gt_boxes"]:
            rect(box, colors["gold"], 3)
        for m in [base_method, method]:
            row = pred_rows[(pred_rows["group_id"] == gid) & (pred_rows["method"] == m)]
            if row.empty:
                continue
            for box in json.loads(row.iloc[0]["pred_boxes_json"]):
                rect(box, colors[m], 2)
        title = f"{g['finding']} | gt={len(g['gt_boxes'])} | {g['claim_sentence'][:72]}"
        draw.rectangle([0, 0, img.size[0], 36], fill=(0, 0, 0))
        draw.text((4, 4), title, fill=(255, 255, 255), font=font)
        panels.append(img)
    if not panels:
        return
    cell_w = max(p.width for p in panels)
    cell_h = max(p.height for p in panels)
    cols = 3
    rows = math.ceil(len(panels) / cols)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (25, 25, 25))
    for idx, panel in enumerate(panels):
        sheet.paste(panel, ((idx % cols) * cell_w, (idx // cols) * cell_h))
    sheet.save(CONTACT / "multibox_rule_context_v2_examples.jpg", quality=94)


def write_report(
    group_audit: pd.DataFrame,
    cue_audit: pd.DataFrame,
    summary: pd.DataFrame,
    per_finding: pd.DataFrame,
    bootstrap: pd.DataFrame,
    best_params: dict[str, Any],
) -> None:
    best = summary[summary["subset"] == "eval_phrase_groups_all"].sort_values("set_f1_0_3", ascending=False).iloc[0]
    lines = [
        "# MS-CXR Multi-box Rule-context Fusion V3",
        "",
        "## 한 줄 결론",
        "",
        (
            f"eval 280 row를 phrase group으로 묶어 평가했다. 전체 eval phrase group에서 "
            f"`{best['method']}`가 set F1@0.3 {best['set_f1_0_3']:.4f}, "
            f"coverage mean IoU {best['coverage_mean_iou']:.4f}를 기록했다."
        ),
        "",
        "## 왜 이 평가가 필요한가",
        "",
        "- 기존 row-level 평가는 같은 이미지/같은 phrase에 bbox가 여러 개 붙은 경우를 제대로 표현하지 못한다.",
        "- `bilateral`, `bibasilar`, `multifocal`, `diffuse`, `upper and lower` 같은 문맥은 한 질의가 여러 위치를 가리킬 수 있다.",
        "- 따라서 이 실험은 row 하나당 bbox 하나가 아니라, phrase group 하나당 bbox set을 예측한다.",
        "",
        "## 논리 안전장치",
        "",
        "- 예측 박스 개수는 claim text와 val-tuned threshold로만 결정했다.",
        "- eval gold의 bbox 개수는 예측 개수 결정에 쓰지 않았다.",
        "- 전체 phrase group 평가에서는 false positive가 set precision/F1로 벌점 처리된다.",
        "- multi-box subset은 보조 분석이며, 전체 성능 주장은 full phrase-group metric을 우선한다.",
        "- MS-CXR bbox는 phrase-grounding bbox이며 lesion pixel mask가 아니다.",
        "",
        "## 데이터 구조",
        "",
        group_audit.groupby(["split", "is_multi_box"]).size().reset_index(name="n_groups").to_markdown(index=False),
        "",
        "## 문맥 cue 분포",
        "",
        cue_audit.groupby(["split", "cue_text"]).size().reset_index(name="n_groups").sort_values(["split", "n_groups"], ascending=[True, False]).to_markdown(index=False),
        "",
        "## 전체 결과",
        "",
        summary.to_markdown(index=False),
        "",
        "## Finding별 결과",
        "",
        per_finding.to_markdown(index=False),
        "",
        "## Bootstrap",
        "",
        bootstrap.to_markdown(index=False) if len(bootstrap) else "(bootstrap 없음)",
        "",
        "## Val에서 선택한 v3 파라미터",
        "",
        "```json",
        json.dumps(best_params, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 해석",
        "",
        "- 이 실험이 성공하려면 단순 coverage만 높아서는 안 되고 set F1도 같이 올라야 한다.",
        "- 추가 박스를 찍는 방식은 bilateral/multifocal 같은 문맥에는 논리적으로 맞지만, single-box phrase에 남발되면 오히려 나쁜 방법이다.",
        "- MedRPG와 비교할 때도 row IoU가 아니라 같은 phrase-group set metric으로 재집계해야 한다.",
    ]
    (REPORT / "README_KO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report_clean(
    group_audit: pd.DataFrame,
    cue_audit: pd.DataFrame,
    summary: pd.DataFrame,
    per_finding: pd.DataFrame,
    bootstrap: pd.DataFrame,
    best_params: dict[str, Any],
) -> None:
    all_rows = summary[summary["subset"] == "eval_phrase_groups_all"].copy()
    best_cov = all_rows.sort_values("coverage_mean_iou", ascending=False).iloc[0]
    best_f1 = all_rows.sort_values("set_f1_0_3", ascending=False).iloc[0]
    lines = [
        "# MS-CXR Multi-box Rule-context Fusion V3",
        "",
        "## 한 줄 결론",
        "",
        (
            f"eval 280 row를 220개 phrase group으로 묶어 평가했다. "
            f"coverage mean IoU 기준 최고는 `{best_cov['method']}` "
            f"{best_cov['coverage_mean_iou']:.4f}이고, "
            f"set F1@0.3 기준 최고는 `{best_f1['method']}` "
            f"{best_f1['set_f1_0_3']:.4f}이다."
        ),
        "",
        "## 왜 이 평가가 필요한가",
        "",
        "- 기존 row-level 평가는 같은 이미지와 같은 phrase에 bbox가 여러 개 붙는 경우를 제대로 표현하지 못한다.",
        "- `bilateral`, `bibasilar`, `multifocal`, `diffuse` 같은 문맥은 한 질의가 여러 위치를 가리킬 수 있다.",
        "- 이 실험은 row 하나당 bbox 하나가 아니라, phrase group 하나당 bbox set을 예측한다.",
        "",
        "## 논리 안전장치",
        "",
        "- 예측 박스 개수는 claim text와 validation에서 고른 threshold로만 결정했다.",
        "- eval gold의 bbox 개수는 예측 개수 결정에 쓰지 않았다.",
        "- 추가 박스는 set precision/F1에서 false positive로 벌점 처리된다.",
        "- multi-box subset은 보조 분석이고, 전체 주장은 full phrase-group metric을 우선한다.",
        "- MS-CXR bbox는 phrase-grounding bbox이며 lesion pixel mask가 아니다.",
        "",
        "## 데이터 구조",
        "",
        group_audit.groupby(["split", "is_multi_box"]).size().reset_index(name="n_groups").to_markdown(index=False),
        "",
        "## 문맥 cue 분포",
        "",
        cue_audit.groupby(["split", "cue_text"]).size().reset_index(name="n_groups").sort_values(["split", "n_groups"], ascending=[True, False]).to_markdown(index=False),
        "",
        "## 전체 결과",
        "",
        summary.to_markdown(index=False),
        "",
        "## Finding별 결과",
        "",
        per_finding.to_markdown(index=False),
        "",
        "## Bootstrap",
        "",
        bootstrap.to_markdown(index=False) if len(bootstrap) else "(bootstrap 없음)",
        "",
        "## Validation에서 선택한 파라미터",
        "",
        "```json",
        json.dumps(best_params, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 해석",
        "",
        "- 좋은 multi-box 방법은 coverage만 올리면 안 되고 set precision/F1도 같이 유지해야 한다.",
        "- 추가 박스는 bilateral/multifocal 문맥에서는 논리적으로 맞지만 single-box phrase에 남발되면 손해다.",
        "- MedRPG와도 같은 phrase-group set metric으로 재집계해야 이 분석이 공정하다.",
    ]
    (REPORT / "README_KO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_tags(text: str) -> list[str]:
    tags = [t.strip() for t in text.split(",") if t.strip()]
    if not tags:
        raise ValueError("At least one --candidate-tags value is required.")
    return tags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--candidate-tags", default="yolov8n,yolov8s,yolov8m,yolov8l")
    args = parser.parse_args()
    candidate_tags = parse_tags(args.candidate_tags)

    ensure_dirs()
    train_rows = load_rows("train")
    val_rows = load_rows("val")
    eval_rows = load_rows("eval")
    priors = base.make_train_priors(train_rows)

    val_groups_all = mb.make_groups(val_rows)
    eval_groups_all = mb.make_groups(eval_rows)
    group_audit_rows: list[dict[str, Any]] = []
    for split, groups in [("val", val_groups_all), ("eval", eval_groups_all)]:
        for gid, g in groups.items():
            group_audit_rows.append({
                "group_id": gid,
                "split": split,
                "finding": g["finding"],
                "claim_sentence": g["claim_sentence"],
                "n_boxes": len(g["gt_boxes"]),
                "is_multi_box": len(g["gt_boxes"]) > 1,
            })
    group_audit = pd.DataFrame(group_audit_rows)
    group_audit.to_csv(MET / "phrase_group_audit.csv", index=False)

    yolo_params = json.loads(YOLO_V2_CFG.read_text(encoding="utf-8"))
    val_candidates = yv2.read_candidates("val", candidate_tags)
    eval_candidates = yv2.read_candidates("eval", candidate_tags)
    dino_val = load_dino_by_group("val", val_groups_all)
    dino_eval = load_dino_by_group("eval", eval_groups_all)

    best_params, val_grid = tune_set_params(val_groups_all, val_candidates, priors, yolo_params, dino_val, args.quick)
    val_grid.to_csv(MET / "set_prediction_val_grid.csv", index=False)
    (CFG / "best_set_prediction_params.json").write_text(json.dumps(best_params, ensure_ascii=False, indent=2), encoding="utf-8")

    v2_preds, cue_rows = predict_groups(eval_groups_all, eval_candidates, priors, yolo_params, dino_eval, best_params)
    cue_audit = pd.DataFrame(cue_rows)
    cue_audit.to_csv(MET / "context_cue_audit_eval.csv", index=False)

    singleton_methods = load_singleton_methods(eval_groups_all)
    method_preds: dict[str, dict[str, list[dict[str, Any]]]] = {
        "rule_context_multibox_yolo_dino_v4_yolov8l_pool": v2_preds,
    }
    method_preds.update(singleton_methods)

    # The logically clean final policy: keep the strongest single-box fusion
    # when the claim does not ask for multiple locations, and switch to the
    # multi-box set predictor only when text cues imply multiple regions.
    if "yolo_dino_rule_fusion_v1" in singleton_methods:
        cue_by_gid = {r["group_id"]: bool(r.get("has_multi_cue", False)) for r in cue_rows}
        single = singleton_methods["yolo_dino_rule_fusion_v1"]
        hybrid: dict[str, list[dict[str, Any]]] = {}
        for gid in eval_groups_all:
            hybrid[gid] = v2_preds.get(gid, []) if cue_by_gid.get(gid, False) else single.get(gid, [])
        method_preds["hybrid_singlebox_fusion_plus_context_multibox_v4_yolov8l_pool"] = hybrid

    # Detector-set references: tuned with val phrase groups, not eval.
    det_grid = []
    for score_thr in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
        for max_k in [1, 2, 3, 4]:
            for nms_iou in [0.35, 0.5, 0.65]:
                p = {"score_thr": score_thr, "max_k": max_k, "nms_iou": nms_iou}
                pred = detection_set_baseline(val_groups_all, val_candidates, p)
                scores = eval_predictions("det_val", val_groups_all, pred)
                summ = mb.summarize(scores.to_dict("records"), "det_val", "val_all")
                det_grid.append({**p, **summ})
    det_grid_df = pd.DataFrame(det_grid).sort_values(["set_f1_0_3", "coverage_mean_iou"], ascending=False)
    det_grid_df.to_csv(MET / "detector_set_val_grid.csv", index=False)
    det_params = {k: det_grid_df.iloc[0][k].item() for k in ["score_thr", "max_k", "nms_iou"]}
    det_method = "_".join(candidate_tags) + "_detection_set_val_tuned"
    method_preds[det_method] = detection_set_baseline(eval_groups_all, eval_candidates, det_params)

    all_eval_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for method, preds in method_preds.items():
        scored = eval_predictions(method, eval_groups_all, preds)
        all_eval_rows.append(scored)
        summary_rows.extend(summarize_all(scored, eval_groups_all, method))
    if V2_PHRASE_PRED.exists():
        ref = pd.read_csv(V2_PHRASE_PRED)
        ref = ref[
            ref["method"].isin(
                [
                    "rule_context_multibox_yolo_dino_v2",
                    "hybrid_singlebox_fusion_plus_context_multibox_v2",
                ]
            )
        ].copy()
        if len(ref):
            all_eval_rows.append(ref)
            for method in sorted(ref["method"].unique()):
                summary_rows.extend(summarize_all(ref[ref["method"] == method], eval_groups_all, method))
    eval_detail = pd.concat(all_eval_rows, ignore_index=True)
    eval_detail.to_csv(PRED / "phrase_group_set_predictions.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(MET / "phrase_group_set_summary.csv", index=False)

    per_rows: list[dict[str, Any]] = []
    for method in sorted(eval_detail["method"].unique()):
        sub_m = eval_detail[eval_detail["method"] == method]
        for finding in sorted(sub_m["finding"].unique()):
            sub = sub_m[sub_m["finding"] == finding]
            row = mb.summarize(sub.to_dict("records"), method, "eval_phrase_groups_by_finding")
            row["finding"] = finding
            per_rows.append(row)
    per_finding = pd.DataFrame(per_rows)
    per_finding.to_csv(MET / "phrase_group_set_per_finding.csv", index=False)

    boot_rows = []
    comparisons = [
        ("hybrid_singlebox_fusion_plus_context_multibox_v4_yolov8l_pool", "hybrid_singlebox_fusion_plus_context_multibox_v2"),
        ("hybrid_singlebox_fusion_plus_context_multibox_v4_yolov8l_pool", "yolo_dino_rule_fusion_v1"),
        ("hybrid_singlebox_fusion_plus_context_multibox_v4_yolov8l_pool", "medrpg_rowlevel_full_phrase_s42"),
        ("hybrid_singlebox_fusion_plus_context_multibox_v4_yolov8l_pool", "medrpg_rowlevel_full_phrase_s13"),
        ("hybrid_singlebox_fusion_plus_context_multibox_v4_yolov8l_pool", "medrpg_rowlevel_full_phrase_s2026"),
        ("rule_context_multibox_yolo_dino_v4_yolov8l_pool", "rule_context_multibox_yolo_dino_v2"),
        ("rule_context_multibox_yolo_dino_v4_yolov8l_pool", "yolo_dino_rule_fusion_v1"),
        ("rule_context_multibox_yolo_dino_v4_yolov8l_pool", det_method),
        ("rule_context_multibox_yolo_dino_v4_yolov8l_pool", "medrpg_rowlevel_full_phrase_s42"),
        ("yolo_dino_rule_fusion_v1", "medrpg_rowlevel_full_phrase_s42"),
    ]
    for a, b in comparisons:
        if a in set(eval_detail["method"]) and b in set(eval_detail["method"]):
            for metric in ["coverage_mean_iou", "set_f1_0_3", "gt_hit_rate_0_3"]:
                boot_rows.append(bootstrap_diff(eval_detail, a, b, metric, reps=1000 if not args.quick else 300))
    bootstrap = pd.DataFrame(boot_rows)
    bootstrap.to_csv(MET / "phrase_group_set_bootstrap_ci.csv", index=False)

    make_contact_sheet(eval_groups_all, eval_detail)
    write_report_clean(group_audit, cue_audit, summary, per_finding, bootstrap, best_params)

    print(f"project_root={PROJECT_ROOT}")
    print(f"eval_rows={len(eval_rows)}")
    print(f"eval_phrase_groups={len(eval_groups_all)}")
    print(f"eval_single_box_groups={int((group_audit[(group_audit.split == 'eval')]['n_boxes'] == 1).sum())}")
    print(f"eval_multi_box_groups={int((group_audit[(group_audit.split == 'eval')]['n_boxes'] > 1).sum())}")
    own = summary[(summary.method == "rule_context_multibox_yolo_dino_v4_yolov8l_pool") & (summary.subset == "eval_phrase_groups_all")].iloc[0]
    print(f"rule_context_multibox_set_f1_0_3_all={own['set_f1_0_3']:.6f}")
    print(f"rule_context_multibox_coverage_iou_all={own['coverage_mean_iou']:.6f}")
    hybrid_row = summary[
        (summary.method == "hybrid_singlebox_fusion_plus_context_multibox_v4_yolov8l_pool")
        & (summary.subset == "eval_phrase_groups_all")
    ]
    if len(hybrid_row):
        hybrid = hybrid_row.iloc[0]
        print(f"hybrid_set_f1_0_3_all={hybrid['set_f1_0_3']:.6f}")
        print(f"hybrid_coverage_iou_all={hybrid['coverage_mean_iou']:.6f}")
    print(f"summary_path={MET / 'phrase_group_set_summary.csv'}")
    print(f"report_path={REPORT / 'README_KO.md'}")


if __name__ == "__main__":
    main()

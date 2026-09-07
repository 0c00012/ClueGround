#!/usr/bin/env python
"""Learned numeric candidate scorer for YOLO-DINO MS-CXR fusion.

This trains a small sklearn regressor to score YOLO candidate boxes from
non-gold features.  The target during training is candidate IoU to the
phrase-grounding boxes.  Model family and selection thresholds are chosen on
validation phrase groups only; eval is used once afterward.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_ms_cxr_multibox_phrase_grounding_v1 as mb  # noqa: E402
from scripts import run_ms_cxr_multibox_rule_context_fusion_v4_yolov8l_pool as mv4  # noqa: E402
from scripts import run_ms_cxr_semantic_ensemble_gated_hybrid_v2_yolov8m_pool as sem2  # noqa: E402
from scripts import run_ms_cxr_yolo_dino_rule_fusion_v1 as yd  # noqa: E402
from scripts import run_ms_cxr_yolo_rule_context_v1 as base  # noqa: E402
from scripts import run_ms_cxr_yolo_rule_context_v2 as yv2  # noqa: E402


EXP_NAME = "ms_cxr_learned_candidate_scorer_v1"
EXP = PROJECT_ROOT / "experiments" / EXP_NAME
PRED = EXP / "predictions"
MET = EXP / "metrics"
CFG = EXP / "configs"
REPORT = PROJECT_ROOT / "reports" / EXP_NAME

YOLO_V2_CFG = PROJECT_ROOT / "experiments" / "ms_cxr_yolo_rule_context_v2" / "configs" / "best_params_by_finding.json"
SOURCE_FINE = PROJECT_ROOT / "experiments" / "ms_cxr_validation_selected_method_ensemble_v1" / "predictions"

ALL_TAGS = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolo11s", "yolo11m", "yolov8m_tta", "yolov8l_tta"]
FINDINGS = list(base.CLASS_TO_ID.keys())
LATS = ["right", "left", "bilateral", "none", "unknown"]
VERTS = ["apical", "upper", "mid", "lower", "basal", "whole", "unknown"]


def ensure_dirs() -> None:
    for p in [EXP, PRED, MET, CFG, REPORT]:
        p.mkdir(parents=True, exist_ok=True)


def available_tags() -> list[str]:
    tags = []
    for t in ALL_TAGS:
        if (yv2.V1_PRED / f"{t}_train_conf0p001_all_candidates.csv").exists():
            tags.append(t)
    return tags


def load_candidates(split: str, tags: list[str]) -> dict[str, list[dict[str, Any]]]:
    return yv2.read_candidates(split, tags)


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


def one_hot(value: str, vocab: list[str]) -> list[float]:
    return [1.0 if value == v else 0.0 for v in vocab]


def features_for_candidate(
    g: dict[str, Any],
    cand: dict[str, Any],
    q: dict[str, str],
    target_prior: np.ndarray,
    dino_norm: np.ndarray | None,
    yolo_params: dict[str, Any],
) -> list[float]:
    iw = float(g["image_width"])
    ih = float(g["image_height"])
    box_norm = base.xyxy_to_norm(cand["box"], iw, ih)
    cx, cy, bw, bh = [float(x) for x in box_norm]
    area = max(1e-6, bw * bh)
    aspect = bw / max(bh, 1e-6)
    prior_area = max(1e-6, float(target_prior[2] * target_prior[3]))
    side_mode = str(yolo_params.get("side_mode", "radiology_right"))
    region = yv2.center_region_score_v2(box_norm, q, side_mode)
    prior_iou = base.iou_norm(box_norm, target_prior)
    dino_iou = base.iou_norm(box_norm, dino_norm) if dino_norm is not None else 0.0
    conf = max(0.0, float(cand["score"]))
    rank = float(cand.get("rank", 99))
    source = str(cand.get("source_model", ""))
    feat = [
        math.log1p(20.0 * conf),
        conf,
        1.0 / (1.0 + rank),
        rank / 50.0,
        cx,
        cy,
        bw,
        bh,
        area,
        math.log(area / prior_area),
        aspect,
        region,
        prior_iou,
        dino_iou,
        abs(cx - float(target_prior[0])),
        abs(cy - float(target_prior[1])),
        abs(math.log(max(bw, 1e-6) / max(float(target_prior[2]), 1e-6))),
        abs(math.log(max(bh, 1e-6) / max(float(target_prior[3]), 1e-6))),
    ]
    feat += one_hot(g["finding"], FINDINGS)
    feat += one_hot(q.get("laterality", "unknown"), LATS)
    feat += one_hot(q.get("vertical", "unknown"), VERTS)
    feat += one_hot(source, ALL_TAGS)
    return feat


def group_candidates(
    g: dict[str, Any],
    candidates: dict[str, list[dict[str, Any]]],
    priors: dict[str, dict[tuple, np.ndarray]],
    yolo_params_by_finding: dict[str, dict[str, Any]],
    dino_by_group: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, str], np.ndarray]:
    q = base.parse_rule_context(group_row(g))
    params = yolo_params_by_finding.get(str(g["finding"]), yolo_params_by_finding["__global__"])
    target_prior = mv4.target_prior(q, priors, params)
    class_id = base.CLASS_TO_ID[g["finding"]]
    rows = []
    for cand in candidates.get(str(g["dicom_id"]), []):
        if int(cand["class_id"]) != class_id:
            continue
        if int(cand.get("rank", 999)) >= 50:
            continue
        rows.append({
            "candidate": cand,
            "features": features_for_candidate(g, cand, q, target_prior, dino_by_group.get(g["group_id"]), params),
        })
    return rows, q, target_prior


def make_training_table(
    split: str,
    tags: list[str],
    priors: dict[str, dict[tuple, np.ndarray]],
    yolo_params: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    groups = sem2.gh.load_groups(split)
    candidates = load_candidates(split, tags)
    dino = mv4.load_dino_by_group(split, groups)
    X: list[list[float]] = []
    y: list[float] = []
    rows = []
    for gid, g in groups.items():
        cand_rows, _, _ = group_candidates(g, candidates, priors, yolo_params, dino)
        for idx, item in enumerate(cand_rows):
            box = item["candidate"]["box"]
            target = max([mb.iou_xyxy(box, gt) for gt in g["gt_boxes"]] or [0.0])
            X.append(item["features"])
            y.append(float(target))
            rows.append({
                "split": split,
                "group_id": gid,
                "finding": g["finding"],
                "candidate_idx": idx,
                "source_model": item["candidate"].get("source_model", ""),
                "score": item["candidate"].get("score", 0.0),
                "target_iou": target,
            })
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), pd.DataFrame(rows)


def fallback_preds(split: str) -> dict[str, list[dict[str, Any]]]:
    path = SOURCE_FINE / f"candidate_method_{split}_details.csv"
    df = pd.read_csv(path)
    df = df[df["method"] == "semantic_fine_v2"]
    out = {}
    for _, r in df.iterrows():
        boxes = json.loads(r["pred_boxes_json"])
        out[str(r["group_id"])] = [{"box": [float(x) for x in b], "score": 0.0, "source": "semantic_fine_fallback"} for b in boxes]
    return out


def predict_groups(
    split: str,
    tags: list[str],
    model: Any,
    priors: dict[str, dict[tuple, np.ndarray]],
    yolo_params: dict[str, dict[str, Any]],
    score_threshold: float,
    diversity_iou: float,
    max_k: int,
    fallback: bool,
) -> tuple[dict[str, list[dict[str, Any]]], pd.DataFrame]:
    groups = sem2.gh.load_groups(split)
    candidates = load_candidates(split, tags)
    dino = mv4.load_dino_by_group(split, groups)
    fallback_by_gid = fallback_preds(split) if fallback else {}
    preds: dict[str, list[dict[str, Any]]] = {}
    audit = []
    for gid, g in groups.items():
        cand_rows, q, _ = group_candidates(g, candidates, priors, yolo_params, dino)
        cues = mv4.context_cues(str(g["claim_sentence"]), str(g["finding"]), q)
        k = min(int(cues.get("k_hint", 1)), max_k)
        if not bool(cues.get("has_multi_cue", False)):
            k = 1
        scored = []
        for item in cand_rows:
            pred_score = float(model.predict(np.asarray([item["features"]], dtype=np.float32))[0])
            scored.append((pred_score, item["candidate"]))
        selected = []
        for pred_score, cand in sorted(scored, key=lambda x: x[0], reverse=True):
            if pred_score < score_threshold:
                continue
            if any(mb.iou_xyxy(cand["box"], s["box"]) > diversity_iou for s in selected):
                continue
            selected.append({
                "box": [float(x) for x in cand["box"]],
                "score": pred_score,
                "source": f"learned_scorer:{cand.get('source_model', '')}",
            })
            if len(selected) >= k:
                break
        used_fallback = False
        if not selected and fallback:
            selected = fallback_by_gid.get(gid, [])
            used_fallback = True
        preds[gid] = selected
        audit.append({
            "group_id": gid,
            "finding": g["finding"],
            "claim_sentence": g["claim_sentence"],
            "has_multi_cue": bool(cues.get("has_multi_cue", False)),
            "k": k,
            "n_candidates": len(cand_rows),
            "n_selected": len(selected),
            "used_fallback": used_fallback,
            "top_score": max([x[0] for x in scored], default=np.nan),
        })
    return preds, pd.DataFrame(audit)


def detail(method: str, split: str, preds: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    groups = sem2.gh.load_groups(split)
    return pd.DataFrame(mb.eval_method(method, groups, preds))


def summarize(method: str, d: pd.DataFrame) -> dict[str, Any]:
    return mb.summarize(d.to_dict("records"), method, "eval_phrase_groups_all")


def make_models() -> dict[str, Any]:
    return {
        "extra_trees": ExtraTreesRegressor(n_estimators=500, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "random_forest": RandomForestRegressor(n_estimators=350, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "hist_gbdt": HistGradientBoostingRegressor(max_iter=220, learning_rate=0.045, max_leaf_nodes=15, l2_regularization=0.02, random_state=42),
    }


def main() -> None:
    ensure_dirs()
    tags = available_tags()
    train_rows = mv4.load_rows("train")
    priors = yd.base.make_train_priors(train_rows)
    yolo_params = json.loads(YOLO_V2_CFG.read_text(encoding="utf-8"))
    X_train, y_train, train_cands = make_training_table("train", tags, priors, yolo_params)
    train_cands.to_csv(PRED / "train_candidate_training_rows.csv", index=False)

    models = make_models()
    search_rows = []
    best_key = None
    best: dict[str, Any] | None = None
    for model_name, model in models.items():
        weights = 1.0 + 5.0 * y_train
        try:
            model.fit(X_train, y_train, sample_weight=weights)
        except TypeError:
            model.fit(X_train, y_train)
        for score_threshold in [-0.05, 0.0, 0.05, 0.1, 0.15, 0.2]:
            for diversity_iou in [0.35, 0.5, 0.65, 0.8]:
                for max_k in [1, 2, 3, 4]:
                    for fallback in [True, False]:
                        preds, audit = predict_groups("val", tags, model, priors, yolo_params, score_threshold, diversity_iou, max_k, fallback)
                        d = detail("val_learned_scorer", "val", preds)
                        s = summarize("val_learned_scorer", d)
                        row = {
                            "model_name": model_name,
                            "score_threshold": score_threshold,
                            "diversity_iou": diversity_iou,
                            "max_k": max_k,
                            "fallback": fallback,
                            "mean_selected": float(audit["n_selected"].mean()),
                            "fallback_rate": float(audit["used_fallback"].mean()),
                            **s,
                        }
                        search_rows.append(row)
                        key = (float(s["coverage_mean_iou"]), float(s["gt_hit_rate_0_5"]), float(s["set_f1_0_3"]))
                        if best_key is None or key > best_key:
                            best_key = key
                            best = row
    assert best is not None
    search = pd.DataFrame(search_rows).sort_values(["coverage_mean_iou", "gt_hit_rate_0_5", "set_f1_0_3"], ascending=False)
    search.to_csv(MET / "learned_scorer_val_search.csv", index=False)

    best_model = make_models()[str(best["model_name"])]
    try:
        best_model.fit(X_train, y_train, sample_weight=1.0 + 5.0 * y_train)
    except TypeError:
        best_model.fit(X_train, y_train)
    preds, audit = predict_groups(
        "eval",
        tags,
        best_model,
        priors,
        yolo_params,
        float(best["score_threshold"]),
        float(best["diversity_iou"]),
        int(best["max_k"]),
        bool(best["fallback"]),
    )
    method = "learned_candidate_scorer_v1"
    d_eval = detail(method, "eval", preds)
    s_eval = pd.DataFrame([
        mb.summarize(d_eval.to_dict("records"), method, "eval_phrase_groups_all"),
        mb.summarize(d_eval[d_eval["n_gt"] == 1].to_dict("records"), method, "eval_phrase_groups_single_box"),
        mb.summarize(d_eval[d_eval["n_gt"] > 1].to_dict("records"), method, "eval_phrase_groups_multi_box"),
    ])
    d_eval.to_csv(PRED / "learned_scorer_phrase_group_predictions.csv", index=False)
    audit.to_csv(PRED / "learned_scorer_eval_audit.csv", index=False)
    s_eval.to_csv(MET / "learned_scorer_summary.csv", index=False)

    ref = pd.read_csv(PROJECT_ROOT / "experiments" / "ms_cxr_semantic_weight_finegrid_v2_yolov8l_pool" / "metrics" / "semantic_weight_finegrid_summary.csv")
    combined = pd.concat([s_eval, ref], ignore_index=True, sort=False)
    combined.to_csv(MET / "summary_with_reference.csv", index=False)
    best_params = {
        "candidate_tags": tags,
        "model_name": str(best["model_name"]),
        "score_threshold": float(best["score_threshold"]),
        "diversity_iou": float(best["diversity_iou"]),
        "max_k": int(best["max_k"]),
        "fallback": bool(best["fallback"]),
        "val_coverage_mean_iou": float(best["coverage_mean_iou"]),
        "val_hit05": float(best["gt_hit_rate_0_5"]),
    }
    (CFG / "best_learned_scorer_params.json").write_text(json.dumps(best_params, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Learned Candidate Scorer V1",
        "",
        "YOLO 후보별 numeric feature로 후보 IoU를 예측하는 sklearn scorer를 train split에서 학습했다.",
        "모델 종류, threshold, diversity, max_k, fallback 여부는 validation에서만 선택했다.",
        "",
        "## Best params",
        "",
        "```json",
        json.dumps(best_params, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Eval summary",
        "",
        combined[combined["subset"] == "eval_phrase_groups_all"][["method", "n_groups", "coverage_mean_iou", "gt_hit_rate_0_3", "gt_hit_rate_0_5", "set_f1_0_3", "set_f1_0_5"]].to_markdown(index=False),
        "",
        "## 주의",
        "",
        "- eval gold는 scorer 학습, threshold 선택, candidate 선택에 사용하지 않았다.",
        "- MS-CXR bbox는 phrase-grounding bbox이며 lesion mask가 아니다.",
    ]
    (REPORT / "README_KO.md").write_text("\n".join(lines), encoding="utf-8")

    own = s_eval[s_eval["subset"] == "eval_phrase_groups_all"].iloc[0]
    print(f"project_root={PROJECT_ROOT}")
    print(f"best_params={json.dumps(best_params, ensure_ascii=False)}")
    print(f"coverage_mean_iou_all={own['coverage_mean_iou']:.6f}")
    print(f"hit03_all={own['gt_hit_rate_0_3']:.6f}")
    print(f"hit05_all={own['gt_hit_rate_0_5']:.6f}")
    print(f"summary_path={MET / 'summary_with_reference.csv'}")
    print(f"report_path={REPORT / 'README_KO.md'}")
    print(combined[combined["subset"] == "eval_phrase_groups_all"][["method", "n_groups", "coverage_mean_iou", "gt_hit_rate_0_3", "gt_hit_rate_0_5", "set_f1_0_3", "set_f1_0_5"]].to_string(index=False))


if __name__ == "__main__":
    main()

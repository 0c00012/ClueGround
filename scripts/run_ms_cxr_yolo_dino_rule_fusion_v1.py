#!/usr/bin/env python
"""YOLO-DINO rule-context fusion for MS-CXR p10-p19.

The experiment keeps both trained components fixed:

* YOLOv8n/v8s produce detection proposals.
* The Stage2 frozen RAD-DINO rule-context heatmap head predicts a
  claim-conditioned coarse box.

Validation split only is used to tune how much the YOLO proposal scorer should
trust RAD-DINO agreement.  Eval gold is used only for final metrics.

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
import torch
from PIL import Image, ImageFile

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from models_ms_cxr_vfm_localizer import PatchHeatmapBBoxHead  # noqa: E402
from scripts import run_ms_cxr_context_vfm_localizer_stage2_context_finetune as stage2  # noqa: E402
from scripts import run_ms_cxr_yolo_rule_context_v1 as base  # noqa: E402
from scripts import run_ms_cxr_yolo_rule_context_v2 as yv2  # noqa: E402


ImageFile.LOAD_TRUNCATED_IMAGES = True

EXP_NAME = "ms_cxr_yolo_dino_rule_fusion_v1"
EXP = PROJECT_ROOT / "experiments" / EXP_NAME
PRED = EXP / "predictions"
MET = EXP / "metrics"
CFG = EXP / "configs"
REPORT = PROJECT_ROOT / "reports" / EXP_NAME

STAGE2_DATA = PROJECT_ROOT / "training" / "ms_cxr_context_vfm_localizer_stage2_context_finetune" / "datasets"
STAGE2_CKPT = PROJECT_ROOT / "training" / "ms_cxr_context_vfm_localizer_stage2_context_finetune" / "checkpoints" / "frozen_heatmap_rule_context.pt"
YOLO_V2_CFG = PROJECT_ROOT / "experiments" / "ms_cxr_yolo_rule_context_v2" / "configs" / "best_params_by_finding.json"
YOLO_V2_MET = PROJECT_ROOT / "experiments" / "ms_cxr_yolo_rule_context_v2" / "metrics"
STAGE2_MET = PROJECT_ROOT / "experiments" / "ms_cxr_context_vfm_localizer_stage2_context_finetune" / "metrics"
MEDRPG_MET = PROJECT_ROOT / "experiments" / "ms_cxr_medrpg_fair_retrain_final_v1" / "metrics"


def ensure_dirs() -> None:
    for p in [EXP, PRED, MET, CFG, REPORT]:
        p.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def source_rows(split: str) -> List[Dict]:
    return read_jsonl(STAGE2_DATA / f"{split}.jsonl")


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_stage2_head(device: str) -> PatchHeatmapBBoxHead:
    ckpt = torch.load(STAGE2_CKPT, map_location="cpu")
    state = ckpt["state"]
    token_dim = int(state["token_proj.weight"].shape[1])
    hidden = int(state["token_proj.weight"].shape[0])
    if "context_proj.weight" in state:
        ctx_dim = int(state["context_proj.weight"].shape[1])
    else:
        ctx_dim = int(ckpt.get("ctx_dim", 0))
    model = PatchHeatmapBBoxHead(token_dim=token_dim, context_dim=ctx_dim, hidden=hidden, dropout=0.1)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model


def context_for_rows(split: str, rows: List[Dict]) -> np.ndarray:
    context_file = STAGE2_DATA / f"context_{split}.jsonl"
    if not context_file.exists():
        raise FileNotFoundError(context_file)
    ctx_rows = stage2.flatten_context(read_jsonl(context_file), "rule_context")
    maps = stage2.context_maps("rule_context")
    return stage2.encode_context(rows, ctx_rows, maps, "rule_context").astype("float32")


@torch.no_grad()
def extract_patch_tokens(images: List[Image.Image], processor, vfm, device: str) -> torch.Tensor:
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = vfm(**inputs)
    hidden = getattr(out, "last_hidden_state", None)
    if hidden is None:
        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            raise RuntimeError("VFM output has neither last_hidden_state nor pooler_output.")
        hidden = pooled.unsqueeze(1)
    hidden = hidden.float()
    if hidden.ndim != 3:
        hidden = hidden.reshape(hidden.shape[0], 1, -1)
    return hidden[:, 1:, :] if hidden.shape[1] > 1 else hidden


def norm_to_xyxy(box: Sequence[float], iw: float, ih: float) -> List[float]:
    return base.norm_to_xyxy(box, iw, ih)


@torch.no_grad()
def build_dino_predictions(split: str, args: argparse.Namespace) -> pd.DataFrame:
    out_csv = PRED / f"rad_dino_rule_context_{split}_predictions.csv"
    out_jsonl = PRED / f"rad_dino_rule_context_{split}_predictions.jsonl"
    if out_csv.exists() and not args.force_dino:
        return pd.read_csv(out_csv)

    from transformers import AutoImageProcessor, AutoModel

    rows = source_rows(split)
    ctx = context_for_rows(split, rows)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    processor = AutoImageProcessor.from_pretrained(args.vfm_model, trust_remote_code=True)
    vfm = AutoModel.from_pretrained(args.vfm_model, trust_remote_code=True).eval().to(device)
    for p in vfm.parameters():
        p.requires_grad_(False)
    head = load_stage2_head(device)

    out_rows: List[Dict] = []
    json_rows: List[Dict] = []
    bs = int(args.batch_size)
    for start in range(0, len(rows), bs):
        batch = rows[start : start + bs]
        images = []
        for r in batch:
            im = Image.open(r["image_path"]).convert("RGB")
            images.append(im)
        tokens = extract_patch_tokens(images, processor, vfm, device)
        ctx_t = torch.tensor(ctx[start : start + len(batch)], dtype=torch.float32, device=device)
        pred_norm, _ = head(tokens, ctx_t)
        pred_np = pred_norm.detach().cpu().numpy()
        for r, p in zip(batch, pred_np):
            p = np.asarray(p, dtype="float32")
            p[:2] = np.clip(p[:2], 0.0, 1.0)
            p[2:] = np.clip(p[2:], 0.02, 1.0)
            pred_xy = norm_to_xyxy(p, r["image_width"], r["image_height"])
            gt = base.clip_box(r["gold_bbox_xyxy"], r["image_width"], r["image_height"])
            iou = base.iou_xyxy(pred_xy, gt)
            rec = {
                "task_id": r["task_id"],
                "dicom_id": r["dicom_id"],
                "split": split,
                "finding": r["finding"],
                "claim_sentence": r.get("claim_sentence", ""),
                "pred_x1": pred_xy[0],
                "pred_y1": pred_xy[1],
                "pred_x2": pred_xy[2],
                "pred_y2": pred_xy[3],
                "pred_cx": float(p[0]),
                "pred_cy": float(p[1]),
                "pred_w": float(p[2]),
                "pred_h": float(p[3]),
                "gt_x1": gt[0],
                "gt_y1": gt[1],
                "gt_x2": gt[2],
                "gt_y2": gt[3],
                "iou": iou,
                "hit_0_3": iou >= 0.3,
                "hit_0_5": iou >= 0.5,
            }
            out_rows.append(rec)
            json_rows.append(
                {
                    "task_id": r["task_id"],
                    "method": "rad_dino_rule_context_recomputed",
                    "dicom_id": r["dicom_id"],
                    "finding": r["finding"],
                    "claim_sentence": r.get("claim_sentence", ""),
                    "pred_bbox_norm_cxcywh": [float(x) for x in p],
                    "pred_bbox_xyxy": pred_xy,
                    "bbox_missing": False,
                    "ctx_mode": "rule_context",
                    "head_kind": "base",
                }
            )
    df = pd.DataFrame(out_rows)
    df.to_csv(out_csv, index=False)
    write_jsonl(out_jsonl, json_rows)
    return df


def dino_map(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    out = {}
    for _, r in df.iterrows():
        out[str(r["task_id"])] = np.asarray([float(r["pred_cx"]), float(r["pred_cy"]), float(r["pred_w"]), float(r["pred_h"])], dtype="float32")
    return out


def load_yolo_v2_params() -> Dict[str, Dict]:
    if not YOLO_V2_CFG.exists():
        raise FileNotFoundError(YOLO_V2_CFG)
    return json.loads(YOLO_V2_CFG.read_text(encoding="utf-8"))


def choose_fusion_candidate(
    row: Dict,
    candidates_by_dicom: Dict[str, List[Dict]],
    priors: Dict,
    dino_by_task: Dict[str, np.ndarray],
    params: Dict,
    fusion: Dict,
) -> Tuple[List[float], float, bool, Dict]:
    iw, ih = float(row["image_width"]), float(row["image_height"])
    q, target_prior = yv2.target_prior_for_row(row, priors, params)
    dino_norm = dino_by_task.get(str(row["task_id"]))
    class_id = base.CLASS_TO_ID[row["finding"]]
    max_rank = int(params.get("max_rank", 30))
    allowed_models = set(str(params.get("model_mode", "both")).split("+"))
    # "both" was hard-coded for the old yolov8n+yolov8s experiment.  In the
    # detector sweep it must mean all candidate sources currently supplied;
    # otherwise yolov8m/yolo11* candidates are dropped and fusion degenerates
    # to the RAD-DINO fallback.
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
        fallback = str(fusion.get("fallback", "dino"))
        if fallback == "dino" and dino_norm is not None:
            return norm_to_xyxy(dino_norm, iw, ih), 0.0, False, {"source": "dino_fallback", **q}
        return norm_to_xyxy(target_prior, iw, ih), 0.0, False, {"source": "prior_fallback", **q}

    side_mode = str(params.get("side_mode", "radiology_right"))
    scored = []
    for cand in cands:
        box_norm = base.xyxy_to_norm(cand["box"], iw, ih)
        conf_score = math.log1p(20.0 * max(0.0, float(cand["score"])))
        region = yv2.center_region_score_v2(box_norm, q, side_mode)
        prior_iou = base.iou_norm(box_norm, target_prior)
        rank_bonus = 1.0 / (1.0 + float(cand.get("rank", 0)))
        area = max(1e-6, float(box_norm[2] * box_norm[3]))
        prior_area = max(1e-6, float(target_prior[2] * target_prior[3]))
        area_penalty = abs(math.log(area / prior_area))
        source_bias = float(params.get("w_yolov8s_bias", 0.0)) if cand.get("source_model") == "yolov8s" else 0.0
        dino_iou = base.iou_norm(box_norm, dino_norm) if dino_norm is not None else 0.0
        total = (
            float(params["w_conf"]) * conf_score
            + float(params["w_region"]) * region
            + float(params["w_prior"]) * prior_iou
            + float(params["w_rank"]) * rank_bonus
            + source_bias
            + float(fusion.get("w_dino", 0.0)) * dino_iou
            - float(params["w_area"]) * area_penalty
        )
        scored.append((total, cand, box_norm, region, prior_iou, rank_bonus, area_penalty, dino_iou))
    scored.sort(key=lambda x: x[0], reverse=True)
    total, cand, box_norm, region, prior_iou, rank_bonus, area_penalty, dino_iou = scored[0]
    pre_norm = base.blend_norm(box_norm, target_prior, float(params.get("blend_yolo_weight", 1.0)))
    if dino_norm is not None:
        final_norm = base.blend_norm(pre_norm, dino_norm, float(fusion.get("yolo_dino_blend", 1.0)))
    else:
        final_norm = pre_norm
    return norm_to_xyxy(final_norm, iw, ih), float(cand["score"]), False, {
        "source": "yolo_dino_fusion",
        "source_model": cand.get("source_model", ""),
        "candidate_rank": int(cand.get("rank", -1)),
        "n_candidates": len(cands),
        "rerank_score": total,
        "region_score": region,
        "prior_iou": prior_iou,
        "rank_bonus": rank_bonus,
        "area_penalty": area_penalty,
        "dino_iou": dino_iou,
        **q,
    }


def evaluate_fusion(
    rows: List[Dict],
    candidates: Dict[str, List[Dict]],
    priors: Dict,
    dino_by_task: Dict[str, np.ndarray],
    params_by_finding: Dict[str, Dict],
    fusion_by_finding: Dict[str, Dict],
    method: str,
    split: str,
) -> pd.DataFrame:
    out = []
    for row in rows:
        params = params_by_finding.get(str(row["finding"]), params_by_finding["__global__"])
        fusion = fusion_by_finding.get(str(row["finding"]), fusion_by_finding["__global__"])
        pred, conf, missing, info = choose_fusion_candidate(row, candidates, priors, dino_by_task, params, fusion)
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
                "claim_sentence": row.get("claim_sentence", ""),
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


def fusion_grid(quick: bool) -> List[Dict]:
    if quick:
        w_dino = [0.0, 0.5, 1.0, 2.0]
        blends = [1.0, 0.85, 0.7]
    else:
        w_dino = [0.0, 0.25, 0.5, 1.0, 1.5, 2.5, 4.0]
        blends = [1.0, 0.9, 0.8, 0.65, 0.5]
    return [{"w_dino": w, "yolo_dino_blend": b, "fallback": "dino"} for w in w_dino for b in blends]


def tune_fusion(
    val_rows: List[Dict],
    val_candidates: Dict[str, List[Dict]],
    priors: Dict,
    dino_val: Dict[str, np.ndarray],
    params_by_finding: Dict[str, Dict],
    quick: bool,
) -> Tuple[Dict[str, Dict], pd.DataFrame, pd.DataFrame]:
    grid = fusion_grid(quick)
    rows = []
    pred_cache: Dict[int, pd.DataFrame] = {}
    best_idx, best = 0, -1.0
    for idx, fusion in enumerate(grid):
        pred = evaluate_fusion(val_rows, val_candidates, priors, dino_val, params_by_finding, {"__global__": fusion}, "val_fusion_grid", "val")
        pred_cache[idx] = pred[["task_id", "finding", "iou", "hit_0_3", "hit_0_5"]].copy()
        m = metrics(pred, "val_fusion_grid", "all8")
        m.update(fusion)
        m["grid_index"] = idx
        rows.append(m)
        if float(m["mean_iou"]) > best:
            best = float(m["mean_iou"])
            best_idx = idx
    grid_df = pd.DataFrame(rows).sort_values("mean_iou", ascending=False)
    fusion_by_finding = {"__global__": dict(grid[best_idx])}

    val_df = pd.concat(pred_cache.values(), keys=pred_cache.keys(), names=["grid_index", "row"]).reset_index(level=0)
    per_rows = []
    for finding in base.CLASS_NAMES:
        sub = val_df[val_df["finding"] == finding]
        if len(sub) < 5:
            fusion_by_finding[finding] = dict(grid[best_idx])
            continue
        by_grid = sub.groupby("grid_index")["iou"].mean().sort_values(ascending=False)
        f_idx = int(by_grid.index[0])
        fusion_by_finding[finding] = dict(grid[f_idx])
        per_rows.append({"finding": finding, "best_grid_index": f_idx, "val_mean_iou": float(by_grid.iloc[0]), "global_grid_index": best_idx})
    return fusion_by_finding, grid_df, pd.DataFrame(per_rows)


def summarize_dino(df: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = []
    for subset in ["all8", "main5"]:
        sub = df.copy()
        if subset == "main5":
            sub = sub[sub["finding"].isin(base.MAIN5)]
        ious = sub["iou"].astype(float).to_numpy()
        rows.append(
            {
                "method": "rad_dino_rule_context_recomputed",
                "split": split,
                "subset": subset,
                "n": int(len(sub)),
                "mean_iou": float(np.mean(ious)) if len(ious) else 0.0,
                "median_iou": float(np.median(ious)) if len(ious) else 0.0,
                "Hit@0.1": float(np.mean(ious >= 0.1)) if len(ious) else 0.0,
                "Hit@0.3": float(np.mean(ious >= 0.3)) if len(ious) else 0.0,
                "Hit@0.5": float(np.mean(ious >= 0.5)) if len(ious) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def comparison(summary: pd.DataFrame, dino_eval_summary: pd.DataFrame) -> pd.DataFrame:
    frames = [summary, dino_eval_summary[dino_eval_summary["split"] == "eval"].drop(columns=["split"], errors="ignore")]
    for path in [
        YOLO_V2_MET / "yolo_rule_context_v2_summary.csv",
        YOLO_V2_MET / "yolo_rule_context_v2_global_summary.csv",
        STAGE2_MET / "summary_all8.csv",
        STAGE2_MET / "summary_main5.csv",
        MEDRPG_MET / "main_fair_comparison.csv",
    ]:
        if path.exists():
            try:
                frames.append(pd.read_csv(path))
            except Exception:
                pass
    return pd.concat(frames, ignore_index=True, sort=False)


def write_report(summary: pd.DataFrame, dino_summary: pd.DataFrame, grid: pd.DataFrame, per: pd.DataFrame, comp: pd.DataFrame) -> None:
    lines = [
        "# MS-CXR YOLO-DINO Rule Fusion v1",
        "",
        "## One-line conclusion",
        "",
        "This experiment combines fixed YOLO detector proposals with fixed RAD-DINO rule-context localization. Fusion weights are selected on validation only; eval gold is used only for final metrics.",
        "",
        "## Fusion summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Recomputed RAD-DINO standalone",
        "",
        dino_summary.to_markdown(index=False),
        "",
        "## Top val fusion grid",
        "",
        grid.head(15).to_markdown(index=False),
        "",
        "## Per-finding fusion choices",
        "",
        per.to_markdown(index=False) if len(per) else "(none)",
        "",
        "## Comparison",
        "",
        comp.head(30).to_markdown(index=False),
        "",
        "## Guardrails",
        "",
        "- MS-CXR boxes are phrase-grounding boxes, not lesion masks.",
        "- YOLO and RAD-DINO weights are fixed in this fusion experiment.",
        "- Fusion uses validation-selected post-processing; it is not an end-to-end phrase grounding model.",
    ]
    (REPORT / "README_KO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--force-dino", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--vfm-model", default="microsoft/rad-dino")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    train_rows = source_rows("train")
    val_rows = source_rows("val")
    eval_rows = source_rows("eval")
    priors = base.make_train_priors(train_rows)
    params_by_finding = load_yolo_v2_params()
    val_candidates = yv2.read_candidates("val", ["yolov8n", "yolov8s"])
    eval_candidates = yv2.read_candidates("eval", ["yolov8n", "yolov8s"])

    dino_val_df = build_dino_predictions("val", args)
    dino_eval_df = build_dino_predictions("eval", args)
    dino_val = dino_map(dino_val_df)
    dino_eval = dino_map(dino_eval_df)
    dino_summary = pd.concat([summarize_dino(dino_val_df, "val"), summarize_dino(dino_eval_df, "eval")], ignore_index=True)
    dino_summary.to_csv(MET / "rad_dino_recomputed_summary.csv", index=False)

    fusion_by_finding, grid, per = tune_fusion(val_rows, val_candidates, priors, dino_val, params_by_finding, quick=args.quick)
    grid.to_csv(MET / "fusion_val_grid_search.csv", index=False)
    per.to_csv(MET / "fusion_per_finding_val_choices.csv", index=False)
    (CFG / "best_fusion_by_finding.json").write_text(json.dumps(fusion_by_finding, ensure_ascii=False, indent=2), encoding="utf-8")

    method = "yolo_dino_rule_fusion_v1"
    pred = evaluate_fusion(eval_rows, eval_candidates, priors, dino_eval, params_by_finding, fusion_by_finding, method, "eval")
    pred.to_csv(PRED / "yolo_dino_rule_fusion_eval_predictions.csv", index=False)
    summary_rows = [metrics(pred, method, "all8"), metrics(pred, method, "main5")]
    single = yv2.add_singlebox_metrics(eval_rows, pred, method)
    if single:
        summary_rows.append(single)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(MET / "yolo_dino_rule_fusion_summary.csv", index=False)

    per_eval = pred.groupby("finding")["iou"].agg(["count", "mean", "median"]).reset_index()
    per_eval["Hit@0.3"] = pred.groupby("finding")["hit_0_3"].mean().values
    per_eval["Hit@0.5"] = pred.groupby("finding")["hit_0_5"].mean().values
    per_eval.to_csv(MET / "yolo_dino_rule_fusion_per_finding.csv", index=False)

    comp = comparison(summary, dino_summary)
    comp.to_csv(MET / "comparison_with_existing_baselines.csv", index=False)
    write_report(summary, dino_summary, grid, per, comp)

    print(f"project_root={PROJECT_ROOT}")
    print(f"train_rows={len(train_rows)}")
    print(f"val_rows={len(val_rows)}")
    print(f"eval_rows={len(eval_rows)}")
    print(f"dino_val_iou_all8={float(dino_summary[(dino_summary['split']=='val') & (dino_summary['subset']=='all8')].iloc[0]['mean_iou']):.6f}")
    print(f"dino_eval_iou_all8={float(dino_summary[(dino_summary['split']=='eval') & (dino_summary['subset']=='all8')].iloc[0]['mean_iou']):.6f}")
    print(f"fusion_val_best_iou={float(grid.iloc[0]['mean_iou']):.6f}")
    print(f"fusion_eval_iou_all8={float(summary.iloc[0]['mean_iou']):.6f}")
    print(f"fusion_eval_hit03_all8={float(summary.iloc[0]['Hit@0.3']):.6f}")
    print(f"fusion_eval_iou_main5={float(summary.iloc[1]['mean_iou']):.6f}")
    print(f"summary_path={MET / 'yolo_dino_rule_fusion_summary.csv'}")
    print(f"report_path={REPORT / 'README_KO.md'}")


if __name__ == "__main__":
    main()

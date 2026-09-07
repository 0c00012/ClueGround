#!/usr/bin/env python
"""Semantic finegrid over an explicit unified YOLO candidate pool.

This fixes the old partial-pool problem:

* row candidate scorer sees the configured detector tags;
* SigLIP/BioMedCLIP crop scorers are rescored from that same candidate table;
* singleton YOLO-DINO fusion and multi-box rule-context fusion both receive the
  same detector pool.

The default pool is YOLOv8n/s/m/l trained on the p10-p19 row-level split.
YOLO11 tags can be supplied only if matching row-level candidate CSVs exist.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_ms_cxr_biomedclip_gated_hybrid_v1 as bm  # noqa: E402
from scripts import run_ms_cxr_multibox_phrase_grounding_v1 as mb  # noqa: E402
from scripts import run_ms_cxr_multibox_rule_context_fusion_v4_yolov8l_pool as mv4  # noqa: E402
from scripts import run_ms_cxr_semantic_weight_grid_v3_yolov8l_pool as gridbase  # noqa: E402
from scripts import run_ms_cxr_siglip_candidate_fusion_v1 as sf  # noqa: E402
from scripts import run_ms_cxr_siglip_gated_hybrid_v1 as gh  # noqa: E402
from scripts import run_ms_cxr_yolo_dino_rule_fusion_v1 as yd  # noqa: E402
from scripts import run_ms_cxr_yolo_rule_context_v2 as yv2  # noqa: E402


TAG_PRESETS = {
    "yolo8_nsml": ["yolov8n", "yolov8s", "yolov8m", "yolov8l"],
    "yolo8_nsml_yolo11_sm": ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolo11s", "yolo11m"],
}

YOLO_V2_CFG = PROJECT_ROOT / "experiments" / "ms_cxr_yolo_rule_context_v2" / "configs" / "best_params_by_finding.json"
FUSION_PRED = PROJECT_ROOT / "experiments" / "ms_cxr_yolo_dino_rule_fusion_v1" / "predictions"


def parse_tags(text: str) -> list[str]:
    if text in TAG_PRESETS:
        return list(TAG_PRESETS[text])
    return [x.strip() for x in text.split(",") if x.strip()]


def ensure_dirs(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def weighted_box(boxes: list[list[float]], weights: list[float]) -> list[float]:
    w = np.asarray(weights, dtype=float)
    w = w / max(float(w.sum()), 1e-8)
    return (np.asarray(boxes, dtype=float) * w[:, None]).sum(axis=0).tolist()


def singleton_map_from_row_predictions(df: pd.DataFrame, groups: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_task: dict[str, dict[str, Any]] = {}
    for _, r in df.iterrows():
        by_task[str(r["task_id"])] = {
            "box": [float(r["pred_x1"]), float(r["pred_y1"]), float(r["pred_x2"]), float(r["pred_y2"])],
            "score": float(r.get("confidence", 1.0)),
            "source": str(r.get("source_model", "yolo_dino_unified_pool")),
        }
    out: dict[str, list[dict[str, Any]]] = {}
    for gid, g in groups.items():
        boxes = []
        for task_id in g["task_ids"]:
            if str(task_id) in by_task:
                boxes = [by_task[str(task_id)]]
                break
        out[gid] = boxes
    return out


def tune_singleton_fusion(tags: list[str], quick: bool, out_cfg: Path) -> dict[str, dict[str, Any]]:
    val_rows = mv4.load_rows("val")
    train_rows = mv4.load_rows("train")
    priors = yd.base.make_train_priors(train_rows)
    params_by_finding = json.loads(YOLO_V2_CFG.read_text(encoding="utf-8"))
    val_candidates = yv2.read_candidates("val", tags)
    dino_val = yd.dino_map(pd.read_csv(FUSION_PRED / "rad_dino_rule_context_val_predictions.csv"))
    fusion_by_finding, grid, per = yd.tune_fusion(val_rows, val_candidates, priors, dino_val, params_by_finding, quick)
    out_cfg.mkdir(parents=True, exist_ok=True)
    (out_cfg / "best_singleton_fusion_by_finding.json").write_text(json.dumps(fusion_by_finding, ensure_ascii=False, indent=2), encoding="utf-8")
    grid.to_csv(out_cfg / "singleton_fusion_val_grid.csv", index=False)
    per.to_csv(out_cfg / "singleton_fusion_per_finding_val.csv", index=False)
    return fusion_by_finding


def build_singleton(
    split: str,
    groups: dict[str, dict[str, Any]],
    tags: list[str],
    fusion_by_finding: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    rows = mv4.load_rows(split)
    priors = yd.base.make_train_priors(mv4.load_rows("train"))
    params_by_finding = json.loads(YOLO_V2_CFG.read_text(encoding="utf-8"))
    candidates = yv2.read_candidates(split, tags)
    dino = yd.dino_map(pd.read_csv(FUSION_PRED / f"rad_dino_rule_context_{split}_predictions.csv"))
    pred = yd.evaluate_fusion(rows, candidates, priors, dino, params_by_finding, fusion_by_finding, "yolo_dino_unified_pool_singleton", split)
    return singleton_map_from_row_predictions(pred, groups)


def build_hybrid(
    split: str,
    groups: dict[str, dict[str, Any]],
    tags: list[str],
    fusion_by_finding: dict[str, dict[str, Any]],
    out_pred: Path,
) -> tuple[dict[str, list[dict[str, Any]]], pd.DataFrame]:
    train_rows = mv4.load_rows("train")
    priors = yd.base.make_train_priors(train_rows)
    yolo_params = json.loads(YOLO_V2_CFG.read_text(encoding="utf-8"))
    set_params = json.loads((gridbase.MULTIBOX_V4 / "configs" / "best_set_prediction_params.json").read_text(encoding="utf-8"))
    candidates = yv2.read_candidates(split, tags)
    dino = mv4.load_dino_by_group(split, groups)
    set_preds, cue_rows = mv4.predict_groups(groups, candidates, priors, yolo_params, dino, set_params)
    cue_by_gid = {r["group_id"]: bool(r.get("has_multi_cue", False)) for r in cue_rows}
    single = build_singleton(split, groups, tags, fusion_by_finding)
    hybrid: dict[str, list[dict[str, Any]]] = {}
    for gid in groups:
        hybrid[gid] = set_preds.get(gid, []) if cue_by_gid.get(gid, False) else single.get(gid, [])
    pd.DataFrame(cue_rows).to_csv(out_pred / f"{split}_cue_rows.csv", index=False)
    return hybrid, pd.DataFrame(cue_rows)


def load_row_candidates(row_scorer_exp: Path, split: str, max_candidates: int) -> pd.DataFrame:
    path = row_scorer_exp / "predictions" / f"{split}_scored_candidates.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    return sf.select_top_candidates(df, max_candidates)


def load_or_score_siglip(
    split: str,
    row_scorer_exp: Path,
    pred_dir: Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    out = pred_dir / f"{split}_siglip_scored_candidates_{args.prompt_mode}_m{str(args.margin).replace('.', 'p')}.csv"
    if out.exists() and not args.force_semantic:
        return pd.read_csv(out)
    df = load_row_candidates(row_scorer_exp, split, int(args.max_candidates_per_task))
    scored = sf.score_siglip(df, args.siglip_model_id, args.prompt_mode, float(args.margin), int(args.batch_size))
    scored.to_csv(out, index=False)
    return scored


def load_or_score_biomed(
    split: str,
    row_scorer_exp: Path,
    pred_dir: Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    out = pred_dir / f"{split}_biomedclip_scored_candidates_{args.prompt_mode}_m{str(args.margin).replace('.', 'p')}.csv"
    if out.exists() and not args.force_semantic:
        return pd.read_csv(out)
    df = load_row_candidates(row_scorer_exp, split, int(args.max_candidates_per_task))
    scored = bm.score_biomedclip(df, args.biomed_model_id, args.prompt_mode, float(args.margin), int(args.batch_size))
    scored.to_csv(out, index=False)
    return scored


def build_semantic_sets(
    val_groups: dict[str, dict[str, Any]],
    eval_groups: dict[str, dict[str, Any]],
    row_scorer_exp: Path,
    exp_pred: Path,
    exp_met: Path,
    exp_cfg: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    val_sig = load_or_score_siglip("val", row_scorer_exp, exp_pred, args)
    eval_sig = load_or_score_siglip("eval", row_scorer_exp, exp_pred, args)
    sig_row_params, sig_grid = sf.tune_row_params(val_sig)
    sig_grid.to_csv(exp_met / "siglip_row_fusion_val_grid.csv", index=False)
    (exp_cfg / "best_siglip_row_params.json").write_text(json.dumps(sig_row_params, ensure_ascii=False, indent=2), encoding="utf-8")
    val_sig = sf.apply_fusion_score(val_sig, sig_row_params, "siglip_fusion_score")
    eval_sig = sf.apply_fusion_score(eval_sig, sig_row_params, "siglip_fusion_score")
    sig_set_params, sig_set_grid = sf.tune_set_params(val_groups, val_sig, "siglip_fusion_score")
    sig_set_grid.to_csv(exp_met / "siglip_set_val_grid.csv", index=False)
    (exp_cfg / "best_siglip_set_params.json").write_text(json.dumps(sig_set_params, ensure_ascii=False, indent=2), encoding="utf-8")
    val_sig_set = sf.predict_phrase_sets(val_groups, sf.scored_candidates_by_group(val_sig, val_groups, "siglip_fusion_score"), sig_set_params)
    eval_sig_set = sf.predict_phrase_sets(eval_groups, sf.scored_candidates_by_group(eval_sig, eval_groups, "siglip_fusion_score"), sig_set_params)

    val_bio = load_or_score_biomed("val", row_scorer_exp, exp_pred, args)
    eval_bio = load_or_score_biomed("eval", row_scorer_exp, exp_pred, args)
    bio_row_params, bio_grid = sf.tune_row_params(val_bio)
    bio_grid.to_csv(exp_met / "biomedclip_row_fusion_val_grid.csv", index=False)
    (exp_cfg / "best_biomedclip_row_params.json").write_text(json.dumps(bio_row_params, ensure_ascii=False, indent=2), encoding="utf-8")
    val_bio = sf.apply_fusion_score(val_bio, bio_row_params, "biomedclip_fusion_score")
    eval_bio = sf.apply_fusion_score(eval_bio, bio_row_params, "biomedclip_fusion_score")
    bio_set_params, bio_set_grid = sf.tune_set_params(val_groups, val_bio, "biomedclip_fusion_score")
    bio_set_grid.to_csv(exp_met / "biomedclip_set_val_grid.csv", index=False)
    (exp_cfg / "best_biomedclip_set_params.json").write_text(json.dumps(bio_set_params, ensure_ascii=False, indent=2), encoding="utf-8")
    val_bio_set = sf.predict_phrase_sets(val_groups, sf.scored_candidates_by_group(val_bio, val_groups, "biomedclip_fusion_score"), bio_set_params)
    eval_bio_set = sf.predict_phrase_sets(eval_groups, sf.scored_candidates_by_group(eval_bio, eval_groups, "biomedclip_fusion_score"), bio_set_params)
    return val_sig_set, eval_sig_set, val_bio_set, eval_bio_set


def has_multi_cue(cue: pd.DataFrame) -> dict[str, bool]:
    return {str(r["group_id"]): bool(r.get("has_multi_cue", False)) for _, r in cue.iterrows()}


def predict_finegrid(
    groups: dict[str, dict[str, Any]],
    hybrid: dict[str, list[dict[str, Any]]],
    siglip: dict[str, list[dict[str, Any]]],
    biomed: dict[str, list[dict[str, Any]]],
    cue: pd.DataFrame,
    weights: tuple[float, float, float],
    min_hybrid_semantic_iou: float,
    keep_multi: bool,
) -> tuple[dict[str, list[dict[str, Any]]], pd.DataFrame]:
    multi = has_multi_cue(cue)
    out: dict[str, list[dict[str, Any]]] = {}
    audit: list[dict[str, Any]] = []
    for gid in groups:
        h = hybrid.get(gid, [])
        s = siglip.get(gid, [])
        b = biomed.get(gid, [])
        pred = h
        action = "keep_hybrid"
        hs = hb = None
        eligible = len(h) == 1 and len(s) == 1 and len(b) == 1
        if eligible and not (keep_multi and multi.get(gid, False)):
            hb0, sb0, bb0 = h[0]["box"], s[0]["box"], b[0]["box"]
            hs = mb.iou_xyxy(hb0, sb0)
            hb = mb.iou_xyxy(hb0, bb0)
            if max(hs, hb) >= float(min_hybrid_semantic_iou):
                pred = [{
                    "box": weighted_box([hb0, sb0, bb0], list(weights)),
                    "score": max(float(h[0].get("score", 0.0)), float(s[0].get("score", 0.0)), float(b[0].get("score", 0.0))),
                    "source": "unified_pool_semantic_finegrid",
                }]
                action = "blend"
        out[gid] = pred
        audit.append({
            "group_id": gid,
            "eligible": eligible,
            "has_multi_cue": multi.get(gid, False),
            "action": action,
            "iou_hybrid_siglip": hs,
            "iou_hybrid_biomedclip": hb,
            "w_hybrid": weights[0],
            "w_siglip": weights[1],
            "w_biomed": weights[2],
        })
    return out, pd.DataFrame(audit)


def detail(method: str, groups: dict[str, dict[str, Any]], preds: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    return pd.DataFrame(mb.eval_method(method, groups, preds))


def summary(method: str, d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subset, sub in {
        "eval_phrase_groups_all": d,
        "eval_phrase_groups_single_box": d[d["n_gt"] == 1],
        "eval_phrase_groups_multi_box": d[d["n_gt"] > 1],
    }.items():
        rows.append(mb.summarize(sub.to_dict("records"), method, subset))
    return pd.DataFrame(rows)


def fine_weights(step: float) -> list[tuple[float, float, float]]:
    vals = [round(i * step, 10) for i in range(int(round(1 / step)) + 1)]
    out = []
    for wh in vals:
        for ws in vals:
            wb = round(1.0 - wh - ws, 10)
            if wb < -1e-8:
                continue
            out.append((wh, ws, max(wb, 0.0)))
    return out


def tune_finegrid(
    groups: dict[str, dict[str, Any]],
    hybrid: dict[str, list[dict[str, Any]]],
    siglip: dict[str, list[dict[str, Any]]],
    biomed: dict[str, list[dict[str, Any]]],
    cue: pd.DataFrame,
    quick: bool,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    best_key = None
    best = None
    min_ious = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5] if not quick else [0.0, 0.1, 0.35]
    for keep_multi in [True, False]:
        for min_iou in min_ious:
            for weights in fine_weights(0.05 if not quick else 0.1):
                preds, audit = predict_finegrid(groups, hybrid, siglip, biomed, cue, weights, min_iou, keep_multi)
                d = detail("val_unified_pool_finegrid", groups, preds)
                s = mb.summarize(d.to_dict("records"), "val_unified_pool_finegrid", "val_all")
                row = {
                    "keep_multi": keep_multi,
                    "min_hybrid_semantic_iou": min_iou,
                    "w_hybrid": weights[0],
                    "w_siglip": weights[1],
                    "w_biomed": weights[2],
                    "n_changed": int((audit["action"] == "blend").sum()),
                    **s,
                }
                rows.append(row)
                key = (float(s["coverage_mean_iou"]), float(s["gt_hit_rate_0_5"]), float(s["set_f1_0_3"]))
                if best_key is None or key > best_key:
                    best_key = key
                    best = row
    assert best is not None
    return {
        "keep_multi": bool(best["keep_multi"]),
        "min_hybrid_semantic_iou": float(best["min_hybrid_semantic_iou"]),
        "w_hybrid": float(best["w_hybrid"]),
        "w_siglip": float(best["w_siglip"]),
        "w_biomed": float(best["w_biomed"]),
        "n_changed_val": int(best["n_changed"]),
    }, pd.DataFrame(rows).sort_values(["coverage_mean_iou", "gt_hit_rate_0_5", "set_f1_0_3"], ascending=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag-preset", default="yolo8_nsml", choices=sorted(TAG_PRESETS))
    parser.add_argument("--candidate-tags", default="")
    parser.add_argument("--row-scorer-exp", default="")
    parser.add_argument("--exp-name", default="")
    parser.add_argument("--siglip-model-id", default="google/siglip-base-patch16-224")
    parser.add_argument("--biomed-model-id", default="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    parser.add_argument("--prompt-mode", default="cxr_claim", choices=["claim", "cxr_claim", "finding_claim", "region_prompt"])
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-candidates-per-task", type=int, default=12)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force-semantic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tags = parse_tags(args.candidate_tags) if args.candidate_tags else parse_tags(args.tag_preset)
    exp_name = args.exp_name or f"ms_cxr_semantic_finegrid_unified_{args.tag_preset}_v1"
    exp = PROJECT_ROOT / "experiments" / exp_name
    pred_dir = exp / "predictions"
    met_dir = exp / "metrics"
    cfg_dir = exp / "configs"
    report_dir = PROJECT_ROOT / "reports" / exp_name
    ensure_dirs([exp, pred_dir, met_dir, cfg_dir, report_dir])
    row_scorer = PROJECT_ROOT / "experiments" / (args.row_scorer_exp or f"ms_cxr_rowlevel_candidate_set_scorer_{args.tag_preset}_v1")
    if not (row_scorer / "predictions" / "eval_scored_candidates.csv").exists():
        raise FileNotFoundError(row_scorer / "predictions" / "eval_scored_candidates.csv")

    (cfg_dir / "run_config.json").write_text(
        json.dumps({"args": vars(args), "candidate_tags": tags, "row_scorer_exp": str(row_scorer)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    val_groups = gh.load_groups("val")
    eval_groups = gh.load_groups("eval")
    fusion_by_finding = tune_singleton_fusion(tags, bool(args.quick), cfg_dir)
    val_hybrid, val_cue = build_hybrid("val", val_groups, tags, fusion_by_finding, pred_dir)
    eval_hybrid, eval_cue = build_hybrid("eval", eval_groups, tags, fusion_by_finding, pred_dir)
    val_sig, eval_sig, val_bio, eval_bio = build_semantic_sets(val_groups, eval_groups, row_scorer, pred_dir, met_dir, cfg_dir, args)

    best_params, grid = tune_finegrid(val_groups, val_hybrid, val_sig, val_bio, val_cue, bool(args.quick))
    grid.to_csv(met_dir / "finegrid_val.csv", index=False)
    (cfg_dir / "best_finegrid_params.json").write_text(json.dumps(best_params, ensure_ascii=False, indent=2), encoding="utf-8")

    preds, audit = predict_finegrid(
        eval_groups,
        eval_hybrid,
        eval_sig,
        eval_bio,
        eval_cue,
        (best_params["w_hybrid"], best_params["w_siglip"], best_params["w_biomed"]),
        best_params["min_hybrid_semantic_iou"],
        best_params["keep_multi"],
    )
    method = f"semantic_finegrid_unified_{args.tag_preset}"
    d = detail(method, eval_groups, preds)
    s = summary(method, d)
    d.to_csv(pred_dir / "phrase_group_predictions.csv", index=False)
    audit.to_csv(pred_dir / "action_audit.csv", index=False)
    s.to_csv(met_dir / "summary.csv", index=False)

    refs = []
    for path in [
        PROJECT_ROOT / "experiments" / "ms_cxr_finegrid_plus_contrastive_expert_v1" / "metrics" / "summary_with_references.csv",
        PROJECT_ROOT / "experiments" / "ms_cxr_semantic_weight_finegrid_v2_yolov8l_pool" / "metrics" / "semantic_weight_finegrid_summary.csv",
        PROJECT_ROOT / "experiments" / "ms_cxr_medrpg_rowlevel_fair_retrain_v1" / "metrics" / "medrpg_rowlevel_3seed_phrase_group_summary.csv",
    ]:
        if path.exists():
            refs.append(pd.read_csv(path))
    combined = pd.concat([s] + refs, ignore_index=True, sort=False) if refs else s.copy()
    combined.to_csv(met_dir / "summary_with_references.csv", index=False)

    all_row = s[s["subset"] == "eval_phrase_groups_all"].iloc[0]
    lines = [
        "# MS-CXR Semantic Finegrid Unified YOLO Pool V1",
        "",
        "## Conclusion",
        "",
        f"- candidate tags: `{tags}`",
        f"- row scorer: `{row_scorer}`",
        f"- eval phrase groups: {int(all_row['n_groups'])}",
        f"- coverage mean IoU: {float(all_row['coverage_mean_iou']):.6f}",
        f"- union IoU: {float(all_row['union_iou']):.6f}",
        f"- Hit@0.3: {float(all_row['gt_hit_rate_0_3']):.6f}",
        f"- Hit@0.5: {float(all_row['gt_hit_rate_0_5']):.6f}",
        f"- SetF1@0.3: {float(all_row['set_f1_0_3']):.6f}",
        f"- SetF1@0.5: {float(all_row['set_f1_0_5']):.6f}",
        "",
        "## Best Params",
        "",
        "```json",
        json.dumps(best_params, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Action Audit",
        "",
        audit.groupby("action").size().reset_index(name="n_groups").to_markdown(index=False),
        "",
        "## Notes",
        "",
        "- MS-CXR boxes are phrase-grounding boxes, not lesion masks.",
        "- Finegrid weights are selected on val only and applied once to eval.",
        "- This run fixes the detector-pool visibility issue by rescoring semantic crops from the same row scorer table.",
    ]
    (report_dir / "README_KO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"project_root={PROJECT_ROOT}")
    print(f"method={method}")
    print(f"candidate_tags={','.join(tags)}")
    print(f"row_scorer={row_scorer}")
    print(f"coverage_mean_iou_all={float(all_row['coverage_mean_iou']):.6f}")
    print(f"union_iou_all={float(all_row['union_iou']):.6f}")
    print(f"hit03_all={float(all_row['gt_hit_rate_0_3']):.6f}")
    print(f"hit05_all={float(all_row['gt_hit_rate_0_5']):.6f}")
    print(f"setf1_03={float(all_row['set_f1_0_3']):.6f}")
    print(f"setf1_05={float(all_row['set_f1_0_5']):.6f}")
    print(f"summary_path={met_dir / 'summary_with_references.csv'}")
    print(f"report_path={report_dir / 'README_KO.md'}")


if __name__ == "__main__":
    main()

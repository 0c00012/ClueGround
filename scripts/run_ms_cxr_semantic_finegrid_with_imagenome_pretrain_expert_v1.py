#!/usr/bin/env python
"""Add an ImaGenome-pretrained candidate expert to YOLO-DINO semantic finegrid.

This is the experiment the final hybrid line actually needs:

  semantic finegrid = YOLO-DINO hybrid + SigLIP + BioMedCLIP
  this run          = YOLO-DINO hybrid + SigLIP + BioMedCLIP
                      + ImaGenome-pretrained candidate scorer expert

The external expert is generated from the already trained broad
MIMIC/Chest-ImaGenome candidate scorer.  Fusion weights are selected only on
MS-CXR validation phrase groups and then applied once to MS-CXR eval groups.

MS-CXR boxes are phrase-grounding boxes, not lesion masks.  Chest ImaGenome
boxes used to train the candidate expert are weak/reference region boxes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_ms_cxr_multibox_phrase_grounding_v1 as mb  # noqa: E402
from scripts import run_ms_cxr_semantic_weight_finegrid_v2_yolov8l_pool as fine  # noqa: E402
from scripts import run_mscxr_imagenome_pretrained_candidate_scorer_v1 as scorer_v1  # noqa: E402
from scripts import run_mimic_imagenome_candidate_pretrain_to_mscxr_v2 as scorer_v2  # noqa: E402


EXP_NAME = "ms_cxr_semantic_finegrid_with_imagenome_pretrain_expert_v1"
EXP = PROJECT_ROOT / "experiments" / EXP_NAME
PRED = EXP / "predictions"
MET = EXP / "metrics"
CFG = EXP / "configs"
REPORT = PROJECT_ROOT / "reports" / EXP_NAME

SCORER_EXP = PROJECT_ROOT / "experiments" / "mimic_imagenome_candidate_pretrain_to_mscxr_v2"
SCORER_CKPT = PROJECT_ROOT / "training" / "mimic_imagenome_candidate_pretrain_to_mscxr_v2" / "checkpoints"
BASE_FINE = PROJECT_ROOT / "experiments" / "ms_cxr_semantic_weight_finegrid_v2_yolov8l_pool"

EXPERT_METHODS = [
    "mimic_imagenome_pretrain_then_mscxr_finetune",
    "mimic_imagenome_pretrain_mscxr_replay_finetune",
    "balanced_mimic_imagenome_plus_mscxr_pooled",
]


def ensure_dirs() -> None:
    for p in [EXP, PRED, MET, CFG, REPORT]:
        p.mkdir(parents=True, exist_ok=True)


def weighted_box(boxes: list[list[float]], weights: list[float]) -> list[float]:
    w = np.asarray(weights, dtype=float)
    w = w / max(float(w.sum()), 1e-8)
    arr = np.asarray(boxes, dtype=float)
    return (arr * w[:, None]).sum(axis=0).tolist()


def load_scorer_model(method: str, dim: int, device: str) -> scorer_v1.CandidateMLP:
    model = scorer_v1.CandidateMLP(dim, hidden=224, dropout=0.12)
    state = torch.load(SCORER_CKPT / f"{method}.pt", map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def selected_params_for(method: str) -> dict[str, Any]:
    selected = pd.read_csv(SCORER_EXP / "configs" / "selected_params.csv")
    row = selected[selected["method"].eq(method)]
    if row.empty:
        raise RuntimeError(f"No selected params for {method}")
    r = row.iloc[0]
    return {
        "threshold": float(r["threshold"]),
        "diversity_iou": float(r["diversity_iou"]),
        "max_k": int(r["max_k"]),
        "fallback": bool(r["fallback"]),
    }


def build_pretrain_expert(split: str, method: str, device: str) -> tuple[dict[str, list[dict[str, Any]]], pd.DataFrame, pd.DataFrame]:
    spec = scorer_v1.FeatureSpec(labels=scorer_v1.label_vocab(), sources=scorer_v1.source_vocab())
    X, _y, meta = scorer_v1.build_ms_table(split, spec, include_target=False)
    dim = int(X.shape[1])
    model = load_scorer_model(method, dim, device)
    meta = meta.copy()
    score_col = f"score_{method}"
    meta[score_col] = scorer_v1.predict_scores(model, X, device)
    params = selected_params_for(method)
    fallback = scorer_v1.fallback_preds(split)
    preds, audit = scorer_v1.select_predictions(
        meta,
        score_col,
        params["threshold"],
        params["diversity_iou"],
        params["max_k"],
        params["fallback"],
        fallback,
    )
    detail = scorer_v1.eval_method(f"{method}_{split}_expert", split, preds)
    meta.to_csv(PRED / f"{method}_{split}_candidate_scores.csv", index=False)
    detail.to_csv(PRED / f"{method}_{split}_expert_phrase_group_predictions.csv", index=False)
    audit.to_csv(PRED / f"{method}_{split}_expert_audit.csv", index=False)
    return preds, detail, audit


def focused_weights4() -> list[tuple[float, float, float, float]]:
    """Focused 4-way grid around the old finegrid optimum.

    Old best was roughly hybrid=0.5, SigLIP=0.0625, BioMedCLIP=0.4375.
    We let the pretrain expert take up to 0.30 mass and re-normalize the rest.
    """

    out: list[tuple[float, float, float, float]] = []
    pre_vals = [round(x, 4) for x in np.arange(0.0, 0.3001, 0.025)]
    h_vals = [round(x, 4) for x in np.arange(0.35, 0.6251, 0.025)]
    s_vals = [round(x, 4) for x in np.arange(0.0, 0.1251, 0.025)]
    for wp in pre_vals:
        rest = 1.0 - wp
        if rest < 0:
            continue
        for wh in h_vals:
            for ws in s_vals:
                wb = round(rest - wh - ws, 4)
                if wb < -1e-8:
                    continue
                out.append((wh, ws, max(wb, 0.0), wp))
    # Always include old best with pretrain=0.
    out.append((0.5, 0.0625, 0.4375, 0.0))
    return sorted(set(out))


def has_multi_cue_map(cue: pd.DataFrame) -> dict[str, bool]:
    return {str(r["group_id"]): bool(r.get("has_multi_cue", False)) for _, r in cue.iterrows()}


def predict4(
    groups: dict[str, dict[str, Any]],
    hybrid: dict[str, list[dict[str, Any]]],
    siglip: dict[str, list[dict[str, Any]]],
    biomed: dict[str, list[dict[str, Any]]],
    pretrain: dict[str, list[dict[str, Any]]],
    cue: pd.DataFrame,
    weights: tuple[float, float, float, float],
    keep_multi: bool,
    min_agreement_iou: float,
) -> tuple[dict[str, list[dict[str, Any]]], pd.DataFrame]:
    multi = has_multi_cue_map(cue)
    out: dict[str, list[dict[str, Any]]] = {}
    audit: list[dict[str, Any]] = []
    wh, ws, wb, wp = weights
    for gid in groups:
        h = hybrid.get(gid, [])
        s = siglip.get(gid, [])
        b = biomed.get(gid, [])
        p = pretrain.get(gid, [])
        pred = h
        action = "keep_hybrid"
        agreement = None
        eligible = len(h) == 1 and len(s) == 1 and len(b) == 1 and len(p) == 1
        if eligible and not (keep_multi and multi.get(gid, False)):
            boxes = [h[0]["box"], s[0]["box"], b[0]["box"], p[0]["box"]]
            agreement = max(
                mb.iou_xyxy(h[0]["box"], s[0]["box"]),
                mb.iou_xyxy(h[0]["box"], b[0]["box"]),
                mb.iou_xyxy(h[0]["box"], p[0]["box"]),
            )
            if agreement >= min_agreement_iou:
                box = weighted_box(boxes, [wh, ws, wb, wp])
                pred = [{
                    "box": box,
                    "score": max(float(x[0].get("score", 0.0)) for x in [h, s, b, p]),
                    "source": "semantic_finegrid_plus_imagenome_pretrain_expert",
                }]
                action = "blend4"
        out[gid] = pred
        audit.append({
            "group_id": gid,
            "eligible": eligible,
            "has_multi_cue": multi.get(gid, False),
            "action": action,
            "agreement_iou": agreement,
            "w_hybrid": wh,
            "w_siglip": ws,
            "w_biomed": wb,
            "w_pretrain_expert": wp,
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


def tune_for_expert(expert_method: str, device: str) -> tuple[dict[str, Any], pd.DataFrame]:
    groups, hybrid, siglip, biomed, cue = fine.base.build_inputs("val")
    pretrain, _detail_pre, _audit_pre = build_pretrain_expert("val", expert_method, device)
    rows: list[dict[str, Any]] = []
    best_key = None
    best = None
    for keep_multi in [True]:
        for min_iou in [0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2]:
            for weights in focused_weights4():
                preds, audit = predict4(groups, hybrid, siglip, biomed, pretrain, cue, weights, keep_multi, min_iou)
                d = detail("val_finegrid_pretrain_expert", groups, preds)
                s = mb.summarize(d.to_dict("records"), "val_finegrid_pretrain_expert", "val_all")
                row = {
                    "expert_method": expert_method,
                    "keep_multi": keep_multi,
                    "min_agreement_iou": min_iou,
                    "w_hybrid": weights[0],
                    "w_siglip": weights[1],
                    "w_biomed": weights[2],
                    "w_pretrain_expert": weights[3],
                    "n_changed": int((audit["action"] == "blend4").sum()),
                    **s,
                }
                rows.append(row)
                key = (float(s["coverage_mean_iou"]), float(s["gt_hit_rate_0_5"]), float(s["set_f1_0_3"]))
                if best_key is None or key > best_key:
                    best_key = key
                    best = row
    assert best is not None
    return best, pd.DataFrame(rows).sort_values(["coverage_mean_iou", "gt_hit_rate_0_5", "set_f1_0_3"], ascending=False)


def bootstrap(target_detail: pd.DataFrame, refs: pd.DataFrame, target: str) -> pd.DataFrame:
    methods = [
        "semantic_weight_finegrid_v2_yolov8l_pool",
        "mimic_imagenome_pretrain_then_mscxr_finetune",
        "mimic_imagenome_pretrain_mscxr_replay_finetune",
        "hybrid_singlebox_fusion_plus_context_multibox_v4_yolov8l_pool",
        "medrpg_rowlevel_full_phrase_s42",
        "medrpg_rowlevel_full_phrase_s13",
        "medrpg_rowlevel_full_phrase_s2026",
    ]
    all_df = pd.concat([target_detail, refs[refs["method"].isin(methods)]], ignore_index=True, sort=False)
    rng = np.random.default_rng(20260706)
    rows = []
    for b in methods:
        piv = all_df[all_df["method"].isin([target, b])].pivot(index="group_id", columns="method", values="coverage_mean_iou")
        if target not in piv.columns or b not in piv.columns:
            continue
        piv = piv.dropna()
        diff = (piv[target] - piv[b]).to_numpy()
        boots = [float(diff[rng.integers(0, len(diff), len(diff))].mean()) for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({
            "method_a": target,
            "method_b": b,
            "n_groups": int(len(diff)),
            "mean_diff": float(diff.mean()),
            "ci95_low": float(lo),
            "ci95_high": float(hi),
            "p_diff_le_0": float((np.asarray(boots) <= 0).mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_grid = []
    best_rows = []
    eval_details = []
    eval_summaries = []
    eval_audits = []

    for expert in EXPERT_METHODS:
        best, grid = tune_for_expert(expert, device)
        all_grid.append(grid)
        best_rows.append(best)

        params = {
            "expert_method": expert,
            "keep_multi": bool(best["keep_multi"]),
            "min_agreement_iou": float(best["min_agreement_iou"]),
            "w_hybrid": float(best["w_hybrid"]),
            "w_siglip": float(best["w_siglip"]),
            "w_biomed": float(best["w_biomed"]),
            "w_pretrain_expert": float(best["w_pretrain_expert"]),
            "n_changed_val": int(best["n_changed"]),
        }
        groups, hybrid, siglip, biomed, cue = fine.base.build_inputs("eval")
        pretrain, _detail_pre, _audit_pre = build_pretrain_expert("eval", expert, device)
        preds, audit = predict4(
            groups,
            hybrid,
            siglip,
            biomed,
            pretrain,
            cue,
            (params["w_hybrid"], params["w_siglip"], params["w_biomed"], params["w_pretrain_expert"]),
            params["keep_multi"],
            params["min_agreement_iou"],
        )
        method = f"semantic_finegrid_plus_{expert}"
        d = detail(method, groups, preds)
        s = summary(method, d)
        d.to_csv(PRED / f"{method}_phrase_group_predictions.csv", index=False)
        audit.to_csv(PRED / f"{method}_action_audit.csv", index=False)
        eval_details.append(d)
        eval_summaries.append(s)
        eval_audits.append(audit.assign(method=method))
        (CFG / f"{method}_params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")

    pd.concat(all_grid, ignore_index=True, sort=False).to_csv(MET / "val_finegrid_with_pretrain_expert_grid.csv", index=False)
    pd.DataFrame(best_rows).to_csv(CFG / "best_params_by_expert.csv", index=False)
    pd.concat(eval_audits, ignore_index=True, sort=False).to_csv(PRED / "all_action_audits.csv", index=False)

    fine_ref = pd.read_csv(BASE_FINE / "predictions" / "semantic_weight_finegrid_phrase_group_predictions.csv")
    candidate_refs = []
    for name in EXPERT_METHODS:
        p = SCORER_EXP / "predictions" / f"{name}_eval_phrase_group_predictions.csv"
        if p.exists():
            candidate_refs.append(pd.read_csv(p))
    v4 = PROJECT_ROOT / "experiments" / "ms_cxr_multibox_rule_context_fusion_v4_yolov8l_pool" / "predictions" / "phrase_group_set_predictions.csv"
    med_refs = PROJECT_ROOT / "experiments" / "ms_cxr_medrpg_rowlevel_fair_retrain_v1" / "predictions" / "medrpg_rowlevel_phrase_group_predictions.csv"
    refs = [fine_ref, *candidate_refs]
    if v4.exists():
        refs.append(pd.read_csv(v4))
    if med_refs.exists():
        refs.append(pd.read_csv(med_refs))
    refs_detail = pd.concat(refs, ignore_index=True, sort=False)
    combined_summary = pd.concat(eval_summaries, ignore_index=True, sort=False)

    ref_summary_paths = [
        BASE_FINE / "metrics" / "semantic_weight_finegrid_summary.csv",
        SCORER_EXP / "metrics" / "summary_with_reference.csv",
    ]
    for p in ref_summary_paths:
        if p.exists():
            combined_summary = pd.concat([combined_summary, pd.read_csv(p)], ignore_index=True, sort=False)
    combined_summary = combined_summary.drop_duplicates(subset=["method", "subset"], keep="first")
    combined_summary.to_csv(MET / "summary_with_references.csv", index=False)

    boots = []
    for d in eval_details:
        target = str(d["method"].iloc[0])
        boots.append(bootstrap(d, refs_detail, target))
    boot_df = pd.concat(boots, ignore_index=True, sort=False) if boots else pd.DataFrame()
    boot_df.to_csv(MET / "bootstrap_vs_references.csv", index=False)

    all_rows = combined_summary[combined_summary["subset"].eq("eval_phrase_groups_all")][
        ["method", "n_groups", "coverage_mean_iou", "union_iou", "gt_hit_rate_0_3", "gt_hit_rate_0_5", "set_f1_0_3", "set_f1_0_5"]
    ].sort_values("coverage_mean_iou", ascending=False)

    report = [
        "# Semantic Finegrid + ImaGenome Pretrained Expert v1",
        "",
        "## 한 줄 결론",
        "",
        "기존 YOLO-DINO semantic finegrid/fusion에 Chest ImaGenome-pretrained candidate scorer expert를 네 번째 box expert로 추가했다. "
        "Fusion weight는 MS-CXR validation에서만 선택했고, eval gold는 weight 선택에 사용하지 않았다.",
        "",
        "## Eval summary",
        "",
        all_rows.to_markdown(index=False),
        "",
        "## Bootstrap",
        "",
        boot_df.to_markdown(index=False) if not boot_df.empty else "not available",
        "",
        "## 공정성",
        "",
        "- MS-CXR bbox는 phrase-grounding bbox이지 lesion mask가 아니다.",
        "- Chest ImaGenome pretraining boxes는 weak/reference region supervision이다.",
        "- pretrain expert 자체는 기존 `mimic_imagenome_candidate_pretrain_to_mscxr_v2` checkpoint를 사용했다.",
        "- 4-way fusion weights는 val phrase groups에서만 선택했다.",
    ]
    (REPORT / "README_KO.md").write_text("\n".join(report), encoding="utf-8")

    print(f"project_root={PROJECT_ROOT}")
    print(f"experiment={EXP_NAME}")
    print(f"summary_path={MET / 'summary_with_references.csv'}")
    print(f"bootstrap_path={MET / 'bootstrap_vs_references.csv'}")
    print(f"report_path={REPORT / 'README_KO.md'}")
    print(all_rows.to_string(index=False))


if __name__ == "__main__":
    main()

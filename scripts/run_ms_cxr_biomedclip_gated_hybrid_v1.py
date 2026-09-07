#!/usr/bin/env python
"""BioMedCLIP-gated YOLO-DINO hybrid for MS-CXR phrase grounding.

BioMedCLIP is used only as a frozen crop-text semantic scorer.  It does not
replace the detector and it is not fine-tuned on MS-CXR.  The experiment mirrors
the SigLIP-gated hybrid:

1. Score YOLO/RAD-DINO candidate crops against the claim text with BioMedCLIP.
2. Tune candidate fusion and set parameters on val only.
3. Keep the existing YOLO-DINO multibox hybrid as the default prediction.
4. On val, select a conservative gate/blend rule for BioMedCLIP correction.
5. Apply the selected rule once to eval.

MS-CXR boxes are phrase-grounding boxes, not pixel-level lesion masks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import open_clip
from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_ms_cxr_multibox_phrase_grounding_v1 as mb  # noqa: E402
from scripts import run_ms_cxr_siglip_candidate_fusion_v1 as sf  # noqa: E402
from scripts import run_ms_cxr_siglip_gated_hybrid_v1 as gh  # noqa: E402


EXP_NAME = "ms_cxr_biomedclip_gated_hybrid_v1"
EXP = PROJECT_ROOT / "experiments" / EXP_NAME
PRED = EXP / "predictions"
MET = EXP / "metrics"
CFG = EXP / "configs"
REPORT = PROJECT_ROOT / "reports" / EXP_NAME
BIOMEDCLIP_REVISIONS = {
    "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224": (
        "9f341de24bfb00180f1b847274256e9b65a3a32e"
    ),
}
BIOMEDCLIP_REQUIRED_FILES = (
    "open_clip_config.json",
    "open_clip_pytorch_model.bin",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)

ROW_SCORER = PROJECT_ROOT / "experiments" / "ms_cxr_rowlevel_candidate_set_scorer_v1"
MULTIBOX_V2 = PROJECT_ROOT / "experiments" / "ms_cxr_multibox_rule_context_fusion_v2"
SIGLIP_GATE = PROJECT_ROOT / "experiments" / "ms_cxr_siglip_gated_hybrid_v1"


def ensure_dirs() -> None:
    for p in [EXP, PRED, MET, CFG, REPORT]:
        p.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def score_biomedclip(
    df: pd.DataFrame,
    model_id: str,
    prompt_mode: str,
    margin: float,
    batch_size: int,
) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    revision = BIOMEDCLIP_REVISIONS.get(model_id)
    if revision is None:
        raise ValueError(f"Unpinned BioMedCLIP model_id: {model_id}")
    snapshot = snapshot_download(
        repo_id=model_id,
        revision=revision,
        allow_patterns=list(BIOMEDCLIP_REQUIRED_FILES),
        local_files_only=True,
    )
    model_reference = f"local-dir:{snapshot}"
    model, _, preprocess = open_clip.create_model_and_transforms(model_reference)
    tokenizer = open_clip.get_tokenizer(model_reference)
    model = model.to(device).eval()
    rows = df.copy()
    scores: list[float] = []
    for start in range(0, len(rows), batch_size):
        sub = rows.iloc[start:start + batch_size]
        images = [preprocess(sf.crop_candidate(r, margin)) for _, r in sub.iterrows()]
        texts = [sf.make_prompt(r["claim_sentence"], r["finding"], prompt_mode) for _, r in sub.iterrows()]
        image_tensor = torch.stack(images).to(device)
        text_tensor = tokenizer(texts).to(device)
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        logits = image_features @ text_features.T
        if hasattr(model, "logit_scale"):
            logits = logits * model.logit_scale.exp()
        vals = torch.diag(logits)
        scores.extend(vals.detach().float().cpu().numpy().tolist())
        print(f"[biomedclip] {start + len(sub)}/{len(rows)}", flush=True)

    rows["biomedclip_raw"] = scores
    rows["biomedclip_z"] = 0.0
    rows["biomedclip_rank"] = 0.0
    group_column = next(
        (column for column in ("task_id", "group_id", "query_id") if column in rows.columns),
        None,
    )
    if group_column is None:
        raise ValueError("BioMedCLIP scoring requires task_id, group_id, or query_id")
    for _, idx in rows.groupby(group_column).groups.items():
        vals = rows.loc[idx, "biomedclip_raw"].astype(float).to_numpy()
        mu = vals.mean()
        sd = vals.std() if vals.std() > 1e-6 else 1.0
        rows.loc[idx, "biomedclip_z"] = (vals - mu) / sd
        ranks = pd.Series(vals).rank(method="average", ascending=True).to_numpy()
        rows.loc[idx, "biomedclip_rank"] = (ranks - 1) / max(1, len(vals) - 1)
    rows["biomedclip_sigmoid"] = 1.0 / (
        1.0 + np.exp(-rows["biomedclip_raw"].astype(float).clip(-50, 50))
    )
    # Standalone legacy callers historically consumed the generic SigLIP names.
    # Preserve that compatibility only when a real SigLIP score is not already present.
    if "siglip_raw" not in rows.columns:
        rows["siglip_raw"] = rows["biomedclip_raw"]
        rows["siglip_z"] = rows["biomedclip_z"]
        rows["siglip_rank"] = rows["biomedclip_rank"]
        rows["siglip_sigmoid"] = rows["biomedclip_sigmoid"]
    return rows


def load_or_score(split: str, args: argparse.Namespace) -> pd.DataFrame:
    suffix = f"{args.prompt_mode}_m{str(args.margin).replace('.', 'p')}"
    out_path = PRED / f"{split}_biomedclip_scored_candidates_{suffix}.csv"
    if out_path.exists() and not args.force:
        return pd.read_csv(out_path)
    src = ROW_SCORER / "predictions" / f"{split}_scored_candidates.csv"
    if not src.exists():
        raise FileNotFoundError(src)
    df = pd.read_csv(src)
    df = sf.select_top_candidates(df, args.max_candidates_per_task)
    scored = score_biomedclip(df, args.model_id, args.prompt_mode, args.margin, args.batch_size)
    scored.to_csv(out_path, index=False)
    return scored


def rename_method(df: pd.DataFrame, old: str, new: str) -> pd.DataFrame:
    out = df.copy()
    out["method"] = out["method"].replace(old, new)
    return out


def bootstrap_vs_refs(detail: pd.DataFrame, refs: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(20260705)
    rows: list[dict[str, Any]] = []
    methods = [
        "hybrid_singlebox_fusion_plus_context_multibox_v2",
        "siglip_gated_hybrid_v1",
        "medrpg_rowlevel_full_phrase_s2026",
        "medrpg_rowlevel_full_phrase_s13",
        "medrpg_rowlevel_full_phrase_s42",
        "rule_context_multibox_yolo_dino_v2",
    ]
    all_df = pd.concat([detail, refs[refs["method"].isin(methods)]], ignore_index=True, sort=False)
    for b in methods:
        piv = all_df[all_df["method"].isin(["biomedclip_gated_hybrid_v1", b])].pivot(
            index="group_id", columns="method", values="coverage_mean_iou"
        )
        if "biomedclip_gated_hybrid_v1" not in piv.columns or b not in piv.columns:
            continue
        piv = piv.dropna(subset=["biomedclip_gated_hybrid_v1", b])
        if not len(piv):
            continue
        diff = (piv["biomedclip_gated_hybrid_v1"] - piv[b]).to_numpy()
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, len(diff), len(diff))
            boots.append(float(diff[idx].mean()))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({
            "method_a": "biomedclip_gated_hybrid_v1",
            "method_b": b,
            "n_groups": int(len(diff)),
            "metric": "coverage_mean_iou",
            "mean_diff": float(diff.mean()),
            "ci95_low": float(lo),
            "ci95_high": float(hi),
            "p_diff_le_0": float((np.asarray(boots) <= 0).mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    parser.add_argument("--prompt-mode", default="cxr_claim", choices=["claim", "cxr_claim", "finding_claim", "region_prompt"])
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-candidates-per-task", type=int, default=12)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    if args.quick:
        args.max_candidates_per_task = min(args.max_candidates_per_task, 6)
        args.batch_size = min(args.batch_size, 24)
    (CFG / "run_config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    val_df = load_or_score("val", args)
    eval_df = load_or_score("eval", args)

    best_row_params, row_grid = sf.tune_row_params(val_df)
    row_grid.to_csv(MET / "candidate_row_fusion_val_grid.csv", index=False)
    (CFG / "best_candidate_row_params.json").write_text(json.dumps(best_row_params, ensure_ascii=False, indent=2), encoding="utf-8")

    val_scored = sf.apply_fusion_score(val_df, best_row_params, "biomedclip_fusion_score")
    eval_scored = sf.apply_fusion_score(eval_df, best_row_params, "biomedclip_fusion_score")
    val_scored.to_csv(PRED / "val_biomedclip_fusion_scored_candidates.csv", index=False)
    eval_scored.to_csv(PRED / "eval_biomedclip_fusion_scored_candidates.csv", index=False)

    row_pred = sf.row_prediction(eval_scored, "biomedclip_fusion_score", "biomedclip_candidate_fusion_row")
    head_pred = sf.row_prediction(eval_scored, "score_head", "score_head_row")
    only_pred = sf.row_prediction(eval_scored, "siglip_rank", "biomedclip_only_row")
    row_pred.to_csv(PRED / "biomedclip_candidate_fusion_row_eval_predictions.csv", index=False)
    only_pred.to_csv(PRED / "biomedclip_only_row_eval_predictions.csv", index=False)
    row_summary = pd.DataFrame([
        sf.row_metrics(row_pred, "biomedclip_candidate_fusion_row"),
        sf.row_metrics(head_pred, "score_head_row"),
        sf.row_metrics(only_pred, "biomedclip_only_row"),
    ])
    row_summary.to_csv(MET / "rowlevel_eval_summary.csv", index=False)

    val_groups = gh.load_groups("val")
    eval_groups = gh.load_groups("eval")
    val_hybrid, val_cue = gh.build_hybrid_v2("val", val_groups)
    eval_hybrid, eval_cue = gh.build_hybrid_v2("eval", eval_groups)

    best_set_params, set_grid = sf.tune_set_params(val_groups, val_scored, "biomedclip_fusion_score")
    set_grid.to_csv(MET / "candidate_set_val_grid.csv", index=False)
    (CFG / "best_candidate_set_params.json").write_text(json.dumps(best_set_params, ensure_ascii=False, indent=2), encoding="utf-8")

    val_cands = sf.scored_candidates_by_group(val_scored, val_groups, "biomedclip_fusion_score")
    eval_cands = sf.scored_candidates_by_group(eval_scored, eval_groups, "biomedclip_fusion_score")
    val_biomed_set = sf.predict_phrase_sets(val_groups, val_cands, best_set_params)
    eval_biomed_set = sf.predict_phrase_sets(eval_groups, eval_cands, best_set_params)

    best_gate_params, gate_grid = gh.tune_gate(val_groups, val_hybrid, val_biomed_set, val_cue)
    gate_grid.to_csv(MET / "gate_val_grid.csv", index=False)
    (CFG / "best_gate_params.json").write_text(json.dumps(best_gate_params, ensure_ascii=False, indent=2), encoding="utf-8")

    gated_preds, audit = gh.combine_predictions(eval_groups, eval_hybrid, eval_biomed_set, eval_cue, best_gate_params)
    gated_detail, gated_summary = gh.summarize("biomedclip_gated_hybrid_v1", eval_groups, gated_preds)
    gated_detail.to_csv(PRED / "biomedclip_gated_hybrid_phrase_group_predictions.csv", index=False)
    audit.to_csv(PRED / "biomedclip_gated_hybrid_action_audit.csv", index=False)

    cand_detail = pd.DataFrame(mb.eval_method("biomedclip_candidate_fusion_phrase_set", eval_groups, eval_biomed_set))
    cand_summary = pd.DataFrame([
        sf.summarize_phrase("biomedclip_candidate_fusion_phrase_set", eval_groups, eval_biomed_set, "all"),
        sf.summarize_phrase("biomedclip_candidate_fusion_phrase_set", eval_groups, eval_biomed_set, "single"),
        sf.summarize_phrase("biomedclip_candidate_fusion_phrase_set", eval_groups, eval_biomed_set, "multi"),
    ])
    cand_detail.to_csv(PRED / "biomedclip_candidate_fusion_phrase_group_predictions.csv", index=False)

    ref = pd.read_csv(MULTIBOX_V2 / "predictions" / "phrase_group_set_predictions.csv")
    ref_sum = pd.read_csv(MULTIBOX_V2 / "metrics" / "phrase_group_set_summary.csv")
    sig_gate_sum = pd.read_csv(SIGLIP_GATE / "metrics" / "summary_with_references.csv")
    sig_gate_sum = sig_gate_sum[sig_gate_sum["method"] == "siglip_gated_hybrid_v1"]
    combined = pd.concat([gated_summary, cand_summary, ref_sum, sig_gate_sum], ignore_index=True, sort=False)
    combined.to_csv(MET / "summary_with_references.csv", index=False)

    boot = bootstrap_vs_refs(gated_detail, ref)
    boot.to_csv(MET / "bootstrap_vs_references.csv", index=False)

    lines = [
        "# BioMedCLIP-Gated Hybrid V1",
        "",
        "## 결론",
        "",
        "BioMedCLIP을 detector로 쓰지 않고, YOLO/RAD-DINO 후보 crop과 claim text의 의미 유사도 점수기로 사용했다.",
        "기존 YOLO-DINO multibox hybrid를 기본값으로 두고, validation에서 선택한 gate/blend 규칙만 eval에 고정 적용했다.",
        "",
        "## 설정",
        "",
        f"- model: `{args.model_id}`",
        f"- prompt_mode: `{args.prompt_mode}`",
        f"- crop margin: `{args.margin}`",
        f"- max candidates per task: `{args.max_candidates_per_task}`",
        "",
        "## Row-level eval",
        "",
        row_summary.to_markdown(index=False),
        "",
        "## Phrase-group eval",
        "",
        combined[combined["subset"] == "eval_phrase_groups_all"][[
            "method", "n_groups", "coverage_mean_iou", "gt_hit_rate_0_3", "gt_hit_rate_0_5", "set_f1_0_3", "set_f1_0_5"
        ]].sort_values("coverage_mean_iou", ascending=False).to_markdown(index=False),
        "",
        "## Best gate params",
        "",
        "```json",
        json.dumps(best_gate_params, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Action audit",
        "",
        audit.groupby("action").size().reset_index(name="n_groups").to_markdown(index=False),
        "",
        "## Bootstrap",
        "",
        boot.to_markdown(index=False),
        "",
        "## 해석 주의",
        "",
        "- BioMedCLIP gate는 validation split에서만 선택했다.",
        "- eval gold bbox는 gate 선택에 사용하지 않았다.",
        "- MS-CXR bbox는 phrase-grounding bbox이며 lesion mask가 아니다.",
        "- BioMedCLIP은 PubMed figure-caption 기반 biomedical VLM이라, CXR 전용 localization 모델은 아니다.",
    ]
    (REPORT / "README_KO.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"project_root={PROJECT_ROOT}")
    print(f"model_id={args.model_id}")
    print(f"best_gate_params={json.dumps(best_gate_params, ensure_ascii=False)}")
    print(f"summary_path={MET / 'summary_with_references.csv'}")
    print(f"report_path={REPORT / 'README_KO.md'}")
    print(combined[combined["subset"] == "eval_phrase_groups_all"][[
        "method", "n_groups", "coverage_mean_iou", "gt_hit_rate_0_3", "gt_hit_rate_0_5", "set_f1_0_3", "set_f1_0_5"
    ]].sort_values("coverage_mean_iou", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()

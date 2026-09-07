"""SigLIP candidate semantic scoring for MS-CXR YOLO-DINO fusion.

SigLIP is used only as a crop-query semantic scorer:
YOLO/RAD-DINO candidate boxes are already generated; SigLIP scores whether
each candidate crop matches the claim text.  Val split selects fusion weights,
then eval is scored once.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_ms_cxr_multibox_phrase_grounding_v1 as mb  # noqa: E402
from scripts import run_ms_cxr_multibox_rule_context_fusion_v2 as mv2  # noqa: E402
from scripts import run_ms_cxr_yolo_rule_context_v1 as ybase  # noqa: E402

EXP_NAME = "ms_cxr_siglip_candidate_fusion_v1"
EXP = PROJECT_ROOT / "experiments" / EXP_NAME
PRED = EXP / "predictions"
MET = EXP / "metrics"
CFG = EXP / "configs"
REPORT = PROJECT_ROOT / "reports" / EXP_NAME

ROW_SCORER = PROJECT_ROOT / "experiments" / "ms_cxr_rowlevel_candidate_set_scorer_v1"
MULTIBOX_V2 = PROJECT_ROOT / "experiments" / "ms_cxr_multibox_rule_context_fusion_v2"
STAGE1_DATA = PROJECT_ROOT / "training" / "ms_cxr_vfm_localizer_stage1_p10_p19" / "datasets"
SIGLIP_REVISIONS = {
    "google/siglip-base-patch16-224": "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed",
}


def resolve_siglip_source(model_id: str, revision: str) -> str:
    """Use an already downloaded, pinned checkpoint before querying the Hub.

    This keeps a frozen experiment reproducible and avoids an unrelated expired
    Hub OAuth session preventing an otherwise fully local run.  An explicit
    SIGLIP_LOCAL_SNAPSHOT takes precedence; the historical local cache is the
    compatibility fallback for this Windows project.
    """
    candidates = [
        os.environ.get("SIGLIP_LOCAL_SNAPSHOT", ""),
        str(
            Path("C:/Users/_idal/PycharmProjects/XAI/cache/huggingface/hub")
            / f"models--{model_id.replace('/', '--')}"
            / "snapshots"
            / revision
        ),
    ]
    for candidate in candidates:
        path = Path(candidate) if candidate else None
        if path and (path / "config.json").is_file() and (path / "model.safetensors").is_file():
            print(f"[siglip] using pinned local snapshot: {path}", flush=True)
            return str(path)
    return model_id


def ensure_dirs() -> None:
    for p in [PRED, MET, CFG, REPORT]:
        p.mkdir(parents=True, exist_ok=True)


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def source_rows(split: str) -> list[dict[str, Any]]:
    return load_jsonl(STAGE1_DATA / f"{split}.jsonl")


def make_prompt(claim: str, finding: str, mode: str) -> str:
    claim = str(claim).strip()
    finding = str(finding).strip()
    if mode == "claim":
        return claim
    if mode == "cxr_claim":
        return f"chest x-ray showing {claim}"
    if mode == "finding_claim":
        return f"{finding}: {claim}"
    if mode == "region_prompt":
        return f"a radiographic region corresponding to {claim}"
    raise ValueError(mode)


def select_top_candidates(df: pd.DataFrame, max_per_task: int) -> pd.DataFrame:
    base = df.copy()
    base["_base_sort"] = (
        base["score_head"].astype(float).fillna(0.0)
        + 0.04 * base["confidence"].astype(float).fillna(0.0)
        + 0.04 * base["prior_iou"].astype(float).fillna(0.0)
        + 0.04 * base["xattn_iou"].astype(float).fillna(0.0)
    )
    kept = []
    for _, sub in base.groupby("task_id", sort=False):
        kept.append(sub.sort_values("_base_sort", ascending=False).head(max_per_task))
    return pd.concat(kept, ignore_index=True).drop(columns=["_base_sort"])


def crop_candidate(row: pd.Series, margin: float) -> Image.Image:
    path = str(row["image_path"])
    img = Image.open(path).convert("RGB")
    w, h = img.size
    x1 = safe_float(row["pred_x1"])
    y1 = safe_float(row["pred_y1"])
    x2 = safe_float(row["pred_x2"])
    y2 = safe_float(row["pred_y2"])
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    x1 -= bw * margin
    x2 += bw * margin
    y1 -= bh * margin
    y2 += bh * margin
    box = (
        int(max(0, min(w - 1, x1))),
        int(max(0, min(h - 1, y1))),
        int(max(1, min(w, x2))),
        int(max(1, min(h, y2))),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        return img.resize((224, 224))
    return img.crop(box)


@torch.no_grad()
def score_siglip(
    df: pd.DataFrame,
    model_id: str,
    prompt_mode: str,
    margin: float,
    batch_size: int,
) -> pd.DataFrame:
    # Keep the heavyweight optional dependency lazy so cached-score workflows
    # and CLI help can run without importing Transformers at module import time.
    from transformers import AutoModel, AutoProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    revision = SIGLIP_REVISIONS.get(model_id)
    if revision is None:
        raise ValueError(f"Unpinned SigLIP model_id: {model_id}")
    source = resolve_siglip_source(model_id, revision)
    load_kwargs = {} if source != model_id else {"revision": revision}
    processor = AutoProcessor.from_pretrained(source, **load_kwargs)
    model = AutoModel.from_pretrained(source, **load_kwargs).to(device).eval()
    rows = df.copy()
    scores: list[float] = []
    for start in range(0, len(rows), batch_size):
        sub = rows.iloc[start:start + batch_size]
        images = [crop_candidate(r, margin) for _, r in sub.iterrows()]
        texts = [make_prompt(r["claim_sentence"], r["finding"], prompt_mode) for _, r in sub.iterrows()]
        inputs = processor(text=texts, images=images, padding="max_length", return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model(**inputs)
        logits = out.logits_per_image
        if logits.ndim == 2 and logits.shape[0] == logits.shape[1]:
            vals = torch.diag(logits)
        else:
            vals = logits.reshape(-1)[: len(sub)]
        scores.extend(vals.detach().float().cpu().numpy().tolist())
        print(f"[siglip] {start + len(sub)}/{len(rows)}", flush=True)
    rows["siglip_raw"] = scores
    rows["siglip_z"] = 0.0
    rows["siglip_rank"] = 0.0
    group_column = next(
        (column for column in ("task_id", "group_id", "query_id") if column in rows.columns),
        None,
    )
    if group_column is None:
        raise ValueError("SigLIP scoring requires task_id, group_id, or query_id")
    for _, idx in rows.groupby(group_column).groups.items():
        vals = rows.loc[idx, "siglip_raw"].astype(float).to_numpy()
        mu = vals.mean()
        sd = vals.std() if vals.std() > 1e-6 else 1.0
        rows.loc[idx, "siglip_z"] = (vals - mu) / sd
        ranks = pd.Series(vals).rank(method="average", ascending=True).to_numpy()
        rows.loc[idx, "siglip_rank"] = (ranks - 1) / max(1, len(vals) - 1)
    rows["siglip_sigmoid"] = 1.0 / (1.0 + np.exp(-rows["siglip_raw"].astype(float).clip(-50, 50)))
    return rows


def load_or_score(split: str, args: argparse.Namespace) -> pd.DataFrame:
    out_path = PRED / f"{split}_siglip_scored_candidates_{args.prompt_mode}_m{str(args.margin).replace('.', 'p')}.csv"
    if out_path.exists() and not args.force:
        return pd.read_csv(out_path)
    src = ROW_SCORER / "predictions" / f"{split}_scored_candidates.csv"
    if not src.exists():
        raise FileNotFoundError(src)
    df = pd.read_csv(src)
    df = select_top_candidates(df, args.max_candidates_per_task)
    scored = score_siglip(df, args.model_id, args.prompt_mode, args.margin, args.batch_size)
    scored.to_csv(out_path, index=False)
    return scored


def norm_score(df: pd.DataFrame, col: str) -> np.ndarray:
    vals = df[col].astype(float).to_numpy()
    return np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)


def apply_fusion_score(df: pd.DataFrame, params: dict[str, float], out_col: str) -> pd.DataFrame:
    out = df.copy()
    out[out_col] = (
        float(params["w_head"]) * norm_score(out, "score_head")
        + float(params["w_siglip_z"]) * norm_score(out, "siglip_z")
        + float(params["w_siglip_rank"]) * norm_score(out, "siglip_rank")
        + float(params["w_prior"]) * norm_score(out, "prior_iou")
        + float(params["w_xattn"]) * norm_score(out, "xattn_iou")
        + float(params["w_conf"]) * norm_score(out, "confidence")
    )
    return out


def row_prediction(scored: pd.DataFrame, score_col: str, method: str) -> pd.DataFrame:
    idx = scored.groupby("task_id")[score_col].idxmax()
    chosen = scored.loc[idx].copy()
    rows = []
    for _, r in chosen.iterrows():
        gt = [safe_float(r["gt_x1"]), safe_float(r["gt_y1"]), safe_float(r["gt_x2"]), safe_float(r["gt_y2"])]
        pred = [safe_float(r["pred_x1"]), safe_float(r["pred_y1"]), safe_float(r["pred_x2"]), safe_float(r["pred_y2"])]
        iou = mb.iou_xyxy(gt, pred)
        rows.append({
            "method": method,
            "task_id": r["task_id"],
            "sample_id": r.get("sample_id", r["task_id"]),
            "dicom_id": r["dicom_id"],
            "subject_id": r.get("subject_id", ""),
            "finding": r["finding"],
            "pred_x1": pred[0],
            "pred_y1": pred[1],
            "pred_x2": pred[2],
            "pred_y2": pred[3],
            "iou": iou,
            "hit_0_3": float(iou >= 0.3),
            "hit_0_5": float(iou >= 0.5),
        })
    return pd.DataFrame(rows)


def row_metrics(pred: pd.DataFrame, method: str) -> dict[str, Any]:
    ious = pred["iou"].astype(float).to_numpy()
    return {
        "method": method,
        "n": int(len(pred)),
        "mean_iou": float(ious.mean()) if len(ious) else 0.0,
        "median_iou": float(np.median(ious)) if len(ious) else 0.0,
        "Hit@0.3": float((ious >= 0.3).mean()) if len(ious) else 0.0,
        "Hit@0.5": float((ious >= 0.5).mean()) if len(ious) else 0.0,
    }


def tune_row_params(val_df: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    grid = []
    for w_siglip_z in [-0.10, -0.05, 0.0, 0.03, 0.06, 0.10, 0.15, 0.22]:
        for w_siglip_rank in [0.0, 0.03, 0.06, 0.10]:
            for w_prior in [0.0, 0.05, 0.10]:
                for w_xattn in [0.0, 0.05, 0.10]:
                    params = {
                        "w_head": 1.0,
                        "w_siglip_z": w_siglip_z,
                        "w_siglip_rank": w_siglip_rank,
                        "w_prior": w_prior,
                        "w_xattn": w_xattn,
                        "w_conf": 0.03,
                    }
                    scored = apply_fusion_score(val_df, params, "fusion_score")
                    pred = row_prediction(scored, "fusion_score", "val")
                    m = row_metrics(pred, "val")
                    grid.append({**params, **m})
    g = pd.DataFrame(grid).sort_values(["mean_iou", "Hit@0.5", "Hit@0.3"], ascending=False)
    best = {k: float(g.iloc[0][k]) for k in ["w_head", "w_siglip_z", "w_siglip_rank", "w_prior", "w_xattn", "w_conf"]}
    return best, g


def group_base_query(group: dict[str, Any]) -> dict[str, str]:
    return ybase.parse_rule_context(mv2.group_row(group))


def scored_candidates_by_group(scored: pd.DataFrame, groups: dict[str, dict[str, Any]], score_col: str) -> dict[str, list[dict[str, Any]]]:
    task_to_group = {tid: gid for gid, g in groups.items() for tid in g["task_ids"]}
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for _, r in scored.iterrows():
        gid = task_to_group.get(str(r.get("task_id", "")))
        if gid is None:
            continue
        box = [safe_float(r["pred_x1"]), safe_float(r["pred_y1"]), safe_float(r["pred_x2"]), safe_float(r["pred_y2"])]
        out[gid].append({"box": box, "score": safe_float(r[score_col]), "source": "siglip_candidate_fusion"})
    return out


def nms_select(cands: list[dict[str, Any]], max_k: int, nms_iou: float, score_ratio: float) -> list[dict[str, Any]]:
    if not cands:
        return []
    ordered = sorted(cands, key=lambda x: float(x["score"]), reverse=True)
    top = float(ordered[0]["score"])
    min_score = top * score_ratio if top > 0 else -math.inf
    selected: list[dict[str, Any]] = []
    for c in ordered:
        if len(selected) >= max_k:
            break
        if selected and float(c["score"]) < min_score:
            continue
        if all(mb.iou_xyxy(c["box"], old["box"]) < nms_iou for old in selected):
            selected.append(c)
    return selected or [ordered[0]]


def predict_phrase_sets(groups: dict[str, dict[str, Any]], cand_by_gid: dict[str, list[dict[str, Any]]], params: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for gid, g in groups.items():
        cue = mv2.context_cues(str(g["claim_sentence"]), str(g["finding"]), group_base_query(g))
        max_k = 1
        if cue["has_multi_cue"]:
            max_k = min(max(int(cue["k_hint"]), int(params["min_k_if_cue"])), int(params["max_k_if_cue"]))
        out[gid] = nms_select(cand_by_gid.get(gid, []), max_k, float(params["nms_iou"]), float(params["score_ratio"]))
    return out


def summarize_phrase(method: str, groups: dict[str, dict[str, Any]], preds: dict[str, list[dict[str, Any]]], subset: str) -> dict[str, Any]:
    detail = pd.DataFrame(mb.eval_method(method, groups, preds))
    if subset == "multi":
        detail = detail[detail["n_gt"] > 1]
        subset_name = "eval_phrase_groups_multi_box"
    elif subset == "single":
        detail = detail[detail["n_gt"] == 1]
        subset_name = "eval_phrase_groups_single_box"
    else:
        subset_name = "eval_phrase_groups_all"
    return mb.summarize(detail.to_dict("records"), method, subset_name)


def tune_set_params(val_groups: dict[str, dict[str, Any]], val_scored: pd.DataFrame, score_col: str) -> tuple[dict[str, Any], pd.DataFrame]:
    cands = scored_candidates_by_group(val_scored, val_groups, score_col)
    rows = []
    best = None
    best_key = None
    for nms_iou in [0.35, 0.5, 0.65]:
        for score_ratio in [0.0, 0.55, 0.7, 0.85]:
            for min_k in [1, 2]:
                for max_k in [2, 3]:
                    if max_k < min_k:
                        continue
                    params = {"nms_iou": nms_iou, "score_ratio": score_ratio, "min_k_if_cue": min_k, "max_k_if_cue": max_k}
                    preds = predict_phrase_sets(val_groups, cands, params)
                    row = summarize_phrase("val_siglip", val_groups, preds, "all")
                    rows.append({**params, **row})
                    key = (float(row["set_f1_0_3"]), float(row["coverage_mean_iou"]))
                    if best_key is None or key > best_key:
                        best_key = key
                        best = params
    assert best is not None
    return best, pd.DataFrame(rows).sort_values(["set_f1_0_3", "coverage_mean_iou"], ascending=False)


def load_existing_reference_summary() -> pd.DataFrame:
    path = MULTIBOX_V2 / "metrics" / "phrase_group_set_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df[df["subset"] == "eval_phrase_groups_all"].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="google/siglip-base-patch16-224")
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

    best_row_params, row_grid = tune_row_params(val_df)
    row_grid.to_csv(MET / "row_fusion_val_grid.csv", index=False)
    (CFG / "best_row_fusion_params.json").write_text(json.dumps(best_row_params, ensure_ascii=False, indent=2), encoding="utf-8")

    eval_scored = apply_fusion_score(eval_df, best_row_params, "siglip_fusion_score")
    val_scored = apply_fusion_score(val_df, best_row_params, "siglip_fusion_score")
    eval_scored.to_csv(PRED / "eval_siglip_fusion_scored_candidates.csv", index=False)

    siglip_pred = row_prediction(eval_scored, "siglip_fusion_score", "siglip_candidate_fusion_row")
    head_pred = row_prediction(eval_scored, "score_head", "score_head_row")
    siglip_only_pred = row_prediction(eval_scored, "siglip_rank", "siglip_only_row")
    for p in [siglip_pred, head_pred, siglip_only_pred]:
        p.to_csv(PRED / f"{p.iloc[0]['method']}_eval_predictions.csv", index=False)
    row_summary = pd.DataFrame([
        row_metrics(siglip_pred, "siglip_candidate_fusion_row"),
        row_metrics(head_pred, "score_head_row"),
        row_metrics(siglip_only_pred, "siglip_only_row"),
    ])
    row_summary.to_csv(MET / "rowlevel_eval_summary.csv", index=False)

    val_groups = mb.make_groups(source_rows("val"))
    eval_groups = mb.make_groups(source_rows("eval"))
    best_set_params, set_grid = tune_set_params(val_groups, val_scored, "siglip_fusion_score")
    set_grid.to_csv(MET / "set_param_val_grid.csv", index=False)
    (CFG / "best_set_params.json").write_text(json.dumps(best_set_params, ensure_ascii=False, indent=2), encoding="utf-8")
    cands = scored_candidates_by_group(eval_scored, eval_groups, "siglip_fusion_score")
    preds = predict_phrase_sets(eval_groups, cands, best_set_params)
    detail = pd.DataFrame(mb.eval_method("siglip_candidate_fusion_phrase_set", eval_groups, preds))
    detail.to_csv(PRED / "siglip_candidate_fusion_phrase_group_predictions.csv", index=False)
    phrase_rows = [
        summarize_phrase("siglip_candidate_fusion_phrase_set", eval_groups, preds, "all"),
        summarize_phrase("siglip_candidate_fusion_phrase_set", eval_groups, preds, "single"),
        summarize_phrase("siglip_candidate_fusion_phrase_set", eval_groups, preds, "multi"),
    ]
    ref = load_existing_reference_summary()
    ref_keep = ref[ref["method"].isin([
        "hybrid_singlebox_fusion_plus_context_multibox_v2",
        "rule_context_multibox_yolo_dino_v2",
        "yolo_dino_rule_fusion_v1",
        "medrpg_rowlevel_full_phrase_s13",
        "medrpg_rowlevel_full_phrase_s42",
        "medrpg_rowlevel_full_phrase_s2026",
    ])]
    phrase_summary = pd.concat([pd.DataFrame(phrase_rows), ref_keep], ignore_index=True, sort=False)
    phrase_summary.to_csv(MET / "phrase_group_summary_with_references.csv", index=False)

    lines = [
        "# SigLIP Candidate Fusion v1",
        "",
        "## 한 줄 결론",
        "",
        "SigLIP을 bbox detector로 쓰지 않고, YOLO/RAD-DINO 후보 crop과 claim text의 의미 유사도 점수기로 사용했다.",
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
        phrase_summary[phrase_summary["subset"] == "eval_phrase_groups_all"][[
            "method", "n_groups", "coverage_mean_iou", "gt_hit_rate_0_3", "gt_hit_rate_0_5", "set_f1_0_3", "set_f1_0_5"
        ]].sort_values("coverage_mean_iou", ascending=False).to_markdown(index=False),
        "",
        "## 해석 주의",
        "",
        "- SigLIP은 일반 image-text 모델이라 CXR 병변 crop 의미 정렬이 약할 수 있다.",
        "- val에서 fusion weight를 고르고 eval에는 고정 적용했다.",
        "- released 외부 모델 학습 데이터와 무관하게, 여기서는 SigLIP을 frozen semantic scorer로만 썼다.",
    ]
    (REPORT / "README_KO.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"project_root={PROJECT_ROOT}")
    print(f"model_id={args.model_id}")
    print(f"row_summary_path={MET / 'rowlevel_eval_summary.csv'}")
    print(f"phrase_summary_path={MET / 'phrase_group_summary_with_references.csv'}")
    print(f"report_path={REPORT / 'README_KO.md'}")
    print(row_summary.to_string(index=False))


if __name__ == "__main__":
    main()

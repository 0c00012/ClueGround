#!/usr/bin/env python
"""Broad MIMIC/Chest ImaGenome pretraining for MS-CXR candidate grounding.

This is the larger-pool version of the previous ImaGenome candidate-scorer
experiment.  The point is not to claim Chest ImaGenome boxes are lesion gold.
They are weak/reference region supervision used to teach a candidate scorer how
structured CXR context relates to candidate boxes before MS-CXR fine-tuning.

Protocol:
  1. Use Chest ImaGenome finding-region and device-linked weak/reference boxes
     from train/val/eval splits as external pretraining source.
  2. Remove any Chest ImaGenome rows whose subject/dicom overlaps MS-CXR
     validation or evaluation rows, so the MS-CXR holdout is not seen through
     external weak supervision.
  3. Train candidate scoring heads on candidate-level features:
       YOLO candidate geometry/confidence/source
       + structured rule context
       + label/domain/source one-hot features
       + train-set geometry priors.
  4. Fine-tune on MS-CXR train candidate rows only.
  5. Select threshold/diversity/max-k/fallback using MS-CXR val only.
  6. Evaluate once on MS-CXR eval phrase groups.

MS-CXR boxes are phrase-grounding bboxes, not lesion masks.
Chest ImaGenome boxes are weak/reference region boxes, not gold lesion boxes.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_mscxr_imagenome_pretrained_candidate_scorer_v1 as v1  # noqa: E402
from scripts import run_ms_cxr_multibox_phrase_grounding_v1 as mb  # noqa: E402


EXP_NAME = "mimic_imagenome_candidate_pretrain_to_mscxr_v2"
EXP = PROJECT_ROOT / "experiments" / EXP_NAME
PRED = EXP / "predictions"
MET = EXP / "metrics"
CFG = EXP / "configs"
REPORT = PROJECT_ROOT / "reports" / EXP_NAME
CKPT = PROJECT_ROOT / "training" / EXP_NAME / "checkpoints"

IMG_PRETRAIN_TASKS = ["finding_region", "device"]
IMG_PRETRAIN_SPLITS = ["train", "val", "eval"]
IMG_TAGS = ["yolov8n", "yolov8s"]


def ensure_dirs() -> None:
    for p in [EXP, PRED, MET, CFG, REPORT, CKPT]:
        p.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mscxr_holdout_ids() -> dict[str, set[str]]:
    """Return MS-CXR val/eval ids excluded from external pretraining."""

    holdout_subjects: set[str] = set()
    holdout_studies: set[str] = set()
    holdout_dicoms: set[str] = set()
    for split in ["val", "eval"]:
        for row in v1.mv4.load_rows(split):
            holdout_subjects.add(str(row.get("subject_id", "")))
            holdout_studies.add(str(row.get("study_id", "")))
            holdout_dicoms.add(str(row.get("dicom_id", "")))
    return {
        "subject_id": {x for x in holdout_subjects if x},
        "study_id": {x for x in holdout_studies if x},
        "dicom_id": {x for x in holdout_dicoms if x},
    }


def is_holdout_overlap(row: dict[str, Any], heldout: dict[str, set[str]]) -> bool:
    return (
        str(row.get("subject_id", "")) in heldout["subject_id"]
        or str(row.get("study_id", "")) in heldout["study_id"]
        or str(row.get("dicom_id", "")) in heldout["dicom_id"]
    )


def imagenome_label_maps(task: str) -> tuple[list[str], dict[str, int]]:
    names = v1.img.class_names(task)
    return names, {name: i for i, name in enumerate(names)}


def build_broad_imagenome_table(
    spec: v1.FeatureSpec,
    heldout: dict[str, set[str]],
    max_rank: int = 50,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    X: list[list[float]] = []
    y: list[float] = []
    meta: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for task in IMG_PRETRAIN_TASKS:
        labels, label_to_id = imagenome_label_maps(task)
        priors = v1.class_priors(v1.img.task_rows(task, "train"), v1.img.TASKS[task].label_col)
        for split in IMG_PRETRAIN_SPLITS:
            rows_all = v1.img.task_rows(task, split)
            rows = [r for r in rows_all if not is_holdout_overlap(r, heldout)]
            excluded = len(rows_all) - len(rows)
            paths = [
                v1.IMAGENOME_EXP / "predictions" / f"{task}_{tag}_{split}_conf0p001_candidates.csv"
                for tag in IMG_TAGS
            ]
            candidates = v1.merge_candidate_groups(paths)
            n_rows_with_candidates = 0
            domain = f"imagenome_{task}"
            for r in rows:
                label = str(r[v1.img.TASKS[task].label_col])
                if label not in label_to_id:
                    continue
                cls = label_to_id[label]
                prior = priors.get(label, np.asarray(r["gold_bbox_norm_cxcywh"], dtype=np.float32))
                gt = [float(x) for x in r["gold_bbox_xyxy"]]
                cand_rows = candidates.get(str(r["dicom_id"]), [])
                kept_for_row = 0
                for cand in cand_rows:
                    if int(cand["class_id"]) != cls or int(cand.get("rank", 999)) >= max_rank:
                        continue
                    target = mb.iou_xyxy(cand["box"], gt)
                    X.append(v1.feature_vector(
                        domain=domain,
                        label=label,
                        text=str(r.get("claim_sentence", label)),
                        cand=cand,
                        image_width=float(r["image_width"]),
                        image_height=float(r["image_height"]),
                        class_id=cls,
                        n_classes=len(labels),
                        prior=prior,
                        spec=spec,
                    ))
                    y.append(float(target))
                    meta.append({
                        "domain": domain,
                        "source_split": split,
                        "task_id": r["task_id"],
                        "dicom_id": r["dicom_id"],
                        "subject_id": r.get("subject_id", ""),
                        "study_id": r.get("study_id", ""),
                        "finding": label,
                        "candidate_source": cand.get("source_model", ""),
                        "candidate_rank": cand.get("rank", -1),
                        "target_iou": target,
                    })
                    kept_for_row += 1
                if kept_for_row:
                    n_rows_with_candidates += 1
            audit_rows.append({
                "task": task,
                "source_split": split,
                "rows_total": len(rows_all),
                "rows_excluded_mscxr_val_eval_overlap": excluded,
                "rows_after_exclusion": len(rows),
                "rows_with_candidates": n_rows_with_candidates,
                "candidate_feature_rows_added": sum(1 for m in meta if m["domain"] == domain and m["source_split"] == split),
            })

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        pd.DataFrame(meta),
        pd.DataFrame(audit_rows),
    )


def stratified_external_sample(
    X_img: np.ndarray,
    y_img: np.ndarray,
    meta: pd.DataFrame,
    target_n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample external rows while preserving useful high-IoU examples."""

    if len(X_img) <= target_n:
        return X_img, y_img
    rng = np.random.default_rng(seed)
    buckets: list[np.ndarray] = []
    for domain in sorted(meta["domain"].dropna().unique()):
        idx_domain = np.where(meta["domain"].to_numpy() == domain)[0]
        y_d = y_img[idx_domain]
        hard_pos = idx_domain[y_d >= 0.5]
        mid_pos = idx_domain[(y_d >= 0.3) & (y_d < 0.5)]
        low = idx_domain[y_d < 0.3]
        quota = max(1, target_n // max(1, len(meta["domain"].dropna().unique())))
        picks: list[np.ndarray] = []
        for arr, frac in [(hard_pos, 0.35), (mid_pos, 0.30), (low, 0.35)]:
            n = min(len(arr), max(1, int(round(quota * frac))))
            if n:
                picks.append(rng.choice(arr, size=n, replace=False))
        if picks:
            buckets.append(np.concatenate(picks))
    if buckets:
        idx = np.unique(np.concatenate(buckets))
    else:
        idx = rng.choice(np.arange(len(X_img)), size=min(target_n, len(X_img)), replace=False)
    if len(idx) < target_n:
        rest = np.setdiff1d(np.arange(len(X_img)), idx)
        extra = rng.choice(rest, size=min(target_n - len(idx), len(rest)), replace=False)
        idx = np.concatenate([idx, extra])
    if len(idx) > target_n:
        idx = rng.choice(idx, size=target_n, replace=False)
    return X_img[idx], y_img[idx]


def train_variants(
    X_img: np.ndarray,
    y_img: np.ndarray,
    img_meta: pd.DataFrame,
    X_ms: np.ndarray,
    y_ms: np.ndarray,
    dim: int,
    device: str,
) -> dict[str, tuple[v1.CandidateMLP, list[dict[str, float]]]]:
    variants: dict[str, tuple[v1.CandidateMLP, list[dict[str, float]]]] = {}

    ms_only = v1.CandidateMLP(dim, hidden=224, dropout=0.12)
    logs = v1.train_mlp(ms_only, X_ms, y_ms, epochs=14, lr=1.8e-3, batch_size=2048, device=device, seed=42)
    variants["mscxr_only_candidate_scorer"] = (ms_only, logs)

    pre = v1.CandidateMLP(dim, hidden=224, dropout=0.12)
    logs_pre = v1.train_mlp(pre, X_img, y_img, epochs=10, lr=1.8e-3, batch_size=4096, device=device, seed=43)
    variants["mimic_imagenome_pretrain_only"] = (pre, logs_pre)

    ft = v1.CandidateMLP(dim, hidden=224, dropout=0.12)
    ft.load_state_dict(pre.state_dict())
    logs_ft = logs_pre + [{"epoch": -1, "loss": -1.0}]
    logs_ft += v1.train_mlp(ft, X_ms, y_ms, epochs=12, lr=6e-4, batch_size=2048, device=device, seed=44)
    variants["mimic_imagenome_pretrain_then_mscxr_finetune"] = (ft, logs_ft)

    X_img_bal, y_img_bal = stratified_external_sample(X_img, y_img, img_meta, target_n=max(len(X_ms) * 3, len(X_ms)), seed=45)
    pooled = v1.CandidateMLP(dim, hidden=224, dropout=0.12)
    X_pool = np.concatenate([X_img_bal, X_ms], axis=0)
    y_pool = np.concatenate([y_img_bal, y_ms], axis=0)
    logs_pool = v1.train_mlp(pooled, X_pool, y_pool, epochs=14, lr=1.2e-3, batch_size=4096, device=device, seed=45)
    variants["balanced_mimic_imagenome_plus_mscxr_pooled"] = (pooled, logs_pool)

    # Replay keeps external positives alive during MS-CXR fine-tuning without
    # letting external weak boxes dominate the final domain.
    X_img_replay, y_img_replay = stratified_external_sample(X_img, y_img, img_meta, target_n=max(len(X_ms) // 2, 1), seed=46)
    replay = v1.CandidateMLP(dim, hidden=224, dropout=0.12)
    replay.load_state_dict(pre.state_dict())
    X_replay_pool = np.concatenate([X_ms, X_img_replay], axis=0)
    y_replay_pool = np.concatenate([y_ms, y_img_replay], axis=0)
    logs_replay = logs_pre + [{"epoch": -2, "loss": -1.0}]
    logs_replay += v1.train_mlp(replay, X_replay_pool, y_replay_pool, epochs=12, lr=7e-4, batch_size=2048, device=device, seed=46)
    variants["mimic_imagenome_pretrain_mscxr_replay_finetune"] = (replay, logs_replay)

    return variants


def bootstrap_many(details: list[pd.DataFrame], ref_detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for detail in details:
        b = v1.bootstrap(detail, ref_detail)
        if not b.empty:
            rows.append(b)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def write_report(
    *,
    X_img: np.ndarray,
    X_ms: np.ndarray,
    X_val: np.ndarray,
    X_eval: np.ndarray,
    img_audit: pd.DataFrame,
    summary_df: pd.DataFrame,
    selected: pd.DataFrame,
    boot_df: pd.DataFrame,
    heldout: dict[str, set[str]],
) -> None:
    all_rows = summary_df[summary_df["subset"].eq("eval_phrase_groups_all")][
        ["method", "n_groups", "coverage_mean_iou", "gt_hit_rate_0_3", "gt_hit_rate_0_5", "set_f1_0_3", "set_f1_0_5"]
    ].sort_values("coverage_mean_iou", ascending=False)
    lines = [
        "# MIMIC/ImaGenome Candidate Pretrain to MS-CXR V2",
        "",
        "## 한 줄 결론",
        "",
        "Chest ImaGenome finding-region/device weak/reference boxes를 넓게 사용해서 후보 scorer를 사전학습하고 MS-CXR에서 fine-tune/eval했다.",
        "MS-CXR val/eval subject, study, dicom overlap은 외부 pretrain source에서 제외했다.",
        "",
        "## 데이터",
        "",
        f"- external ImaGenome candidate rows: `{len(X_img)}`",
        f"- MS-CXR train candidate rows: `{len(X_ms)}`",
        f"- MS-CXR val candidate rows: `{len(X_val)}`",
        f"- MS-CXR eval candidate rows: `{len(X_eval)}`",
        f"- heldout MS-CXR val/eval subjects excluded from external pretrain: `{len(heldout['subject_id'])}`",
        f"- heldout MS-CXR val/eval dicoms excluded from external pretrain: `{len(heldout['dicom_id'])}`",
        "- MS-CXR bbox: phrase-grounding bbox",
        "- Chest ImaGenome bbox: weak/reference region bbox, not lesion gold",
        "",
        "## External source audit",
        "",
        img_audit.to_markdown(index=False),
        "",
        "## Eval summary",
        "",
        all_rows.to_markdown(index=False),
        "",
        "## Selected val parameters",
        "",
        selected.to_markdown(index=False),
        "",
        "## Bootstrap vs semantic finegrid reference",
        "",
        boot_df.to_markdown(index=False) if not boot_df.empty else "not available",
        "",
        "## 해석 규칙",
        "",
        "- 이 실험은 Chest ImaGenome을 gold lesion bbox로 사용하지 않는다.",
        "- 외부 pretraining은 weak/reference region supervision이다.",
        "- MS-CXR eval gold는 학습, threshold 선택, fallback 선택에 쓰지 않았다.",
        "- 성능이 오르면 broad pretraining이 후보 선택에 도움을 준다는 근거다.",
        "- 성능이 안 오르면 현재 feature/scorer 형태에서는 넓은 weak pretraining만으로 부족하다는 negative ablation이다.",
    ]
    (REPORT / "README_KO.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Reuse the v1 feature schema, but write it into the new experiment.
    spec = v1.FeatureSpec(labels=v1.label_vocab(), sources=v1.source_vocab())
    (CFG / "feature_spec.json").write_text(json.dumps({
        "labels": spec.labels,
        "sources": spec.sources,
        "domains": v1.DOMAINS,
        "external_tasks": IMG_PRETRAIN_TASKS,
        "external_splits": IMG_PRETRAIN_SPLITS,
        "heldout_policy": "exclude MS-CXR val/eval subject_id, study_id, dicom_id from external pretraining",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    heldout = mscxr_holdout_ids()
    X_img, y_img, img_meta, img_audit = build_broad_imagenome_table(spec, heldout)
    X_ms, y_ms, ms_meta_train = v1.build_ms_table("train", spec, include_target=True)
    X_val, _, ms_meta_val = v1.build_ms_table("val", spec, include_target=False)
    X_eval, _, ms_meta_eval = v1.build_ms_table("eval", spec, include_target=False)

    img_meta.to_csv(PRED / "external_imagenome_pretrain_candidate_rows.csv", index=False)
    img_audit.to_csv(MET / "external_source_overlap_audit.csv", index=False)
    ms_meta_train.to_csv(PRED / "mscxr_train_candidate_rows.csv", index=False)
    ms_meta_val.to_csv(PRED / "mscxr_val_candidate_rows.csv", index=False)
    ms_meta_eval.to_csv(PRED / "mscxr_eval_candidate_rows.csv", index=False)

    dim = int(X_ms.shape[1])
    variants = train_variants(X_img, y_img, img_meta, X_ms, y_ms, dim, device)
    fallback_val = v1.fallback_preds("val")
    fallback_eval = v1.fallback_preds("eval")

    all_eval_details: list[pd.DataFrame] = []
    all_summaries: list[pd.DataFrame] = []
    all_search: list[pd.DataFrame] = []
    configs: list[dict[str, Any]] = []

    for name, (model, logs) in variants.items():
        pd.DataFrame(logs).to_csv(MET / f"{name}_train_log.csv", index=False)
        torch.save(model.state_dict(), CKPT / f"{name}.pt")

        ms_meta_val[f"score_{name}"] = v1.predict_scores(model, X_val, device)
        ms_meta_eval[f"score_{name}"] = v1.predict_scores(model, X_eval, device)

        best, search = v1.tune_on_val(name, ms_meta_val, fallback_val)
        all_search.append(search)

        preds, audit = v1.select_predictions(
            ms_meta_eval,
            f"score_{name}",
            float(best["threshold"]),
            float(best["diversity_iou"]),
            int(best["max_k"]),
            bool(best["fallback"]),
            fallback_eval,
        )
        detail = v1.eval_method(name, "eval", preds)
        summ = v1.summary(name, detail)
        detail.to_csv(PRED / f"{name}_eval_phrase_group_predictions.csv", index=False)
        audit.to_csv(PRED / f"{name}_eval_audit.csv", index=False)
        all_eval_details.append(detail)
        all_summaries.append(summ)
        configs.append({
            "method": name,
            "threshold": float(best["threshold"]),
            "diversity_iou": float(best["diversity_iou"]),
            "max_k": int(best["max_k"]),
            "fallback": bool(best["fallback"]),
            "fallback_rate": float(best.get("fallback_rate", np.nan)),
            "mean_selected_val": float(best.get("mean_selected", np.nan)),
            "val_coverage_mean_iou": float(best["coverage_mean_iou"]),
            "val_hit05": float(best["gt_hit_rate_0_5"]),
            "device": device,
        })

    pd.concat(all_search, ignore_index=True, sort=False).to_csv(MET / "val_search_all_methods.csv", index=False)
    selected = pd.DataFrame(configs)
    selected.to_csv(CFG / "selected_params.csv", index=False)

    summary_df = pd.concat(all_summaries, ignore_index=True, sort=False)
    ref = pd.read_csv(v1.MS_FINE / "metrics" / "semantic_weight_finegrid_summary.csv")
    combined = pd.concat([summary_df, ref], ignore_index=True, sort=False)
    combined.to_csv(MET / "summary_with_reference.csv", index=False)

    ref_detail = pd.read_csv(v1.MS_FINE / "predictions" / "semantic_weight_finegrid_phrase_group_predictions.csv")
    boot_df = bootstrap_many(all_eval_details, ref_detail)
    boot_df.to_csv(MET / "bootstrap_vs_semantic_finegrid.csv", index=False)

    write_report(
        X_img=X_img,
        X_ms=X_ms,
        X_val=X_val,
        X_eval=X_eval,
        img_audit=img_audit,
        summary_df=combined,
        selected=selected,
        boot_df=boot_df,
        heldout=heldout,
    )

    all_rows = combined[combined["subset"].eq("eval_phrase_groups_all")][
        ["method", "n_groups", "coverage_mean_iou", "gt_hit_rate_0_3", "gt_hit_rate_0_5", "set_f1_0_3", "set_f1_0_5"]
    ].sort_values("coverage_mean_iou", ascending=False)

    print(f"project_root={PROJECT_ROOT}")
    print(f"experiment={EXP_NAME}")
    print(f"external_imagenome_candidate_rows={len(X_img)}")
    print(f"mscxr_train_candidate_rows={len(X_ms)}")
    print(f"mscxr_val_candidate_rows={len(X_val)}")
    print(f"mscxr_eval_candidate_rows={len(X_eval)}")
    print(f"heldout_subjects_excluded={len(heldout['subject_id'])}")
    print(f"summary_path={MET / 'summary_with_reference.csv'}")
    print(f"report_path={REPORT / 'README_KO.md'}")
    print(all_rows.to_string(index=False))


if __name__ == "__main__":
    main()

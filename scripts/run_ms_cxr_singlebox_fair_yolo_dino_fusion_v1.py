#!/usr/bin/env python
"""Fair single-box YOLO + RAD-DINO fusion for MS-CXR.

This runner retrains YOLO on the exact MedRPG fair single-box split
(638/87/163), reuses the already retrained single-box RAD-DINO heads, tunes
fusion only on the single-box validation set, and evaluates on the same
single-box eval set used by the fair MedRPG retrain.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_ms_cxr_rad_dino_singlebox_retrain_v1 as rad  # noqa: E402
from scripts import run_ms_cxr_yolo_detector_v1 as yd  # noqa: E402
from scripts import run_ms_cxr_yolo_dino_rule_fusion_v1 as old_fusion  # noqa: E402
from scripts import run_ms_cxr_yolo_rule_context_v1 as ybase  # noqa: E402
from scripts import run_ms_cxr_yolo_rule_context_v2 as yv2  # noqa: E402
from models_ms_cxr_vfm_localizer import PatchHeatmapBBoxHead  # noqa: E402


EXP_NAME = "ms_cxr_singlebox_fair_yolo_dino_fusion_v1"
EXP = PROJECT_ROOT / "experiments" / EXP_NAME
PRED = EXP / "predictions"
MET = EXP / "metrics"
CFG = EXP / "configs"
LOGS = EXP / "logs"
REPORT = PROJECT_ROOT / "reports" / EXP_NAME
TRAIN = PROJECT_ROOT / "training" / EXP_NAME
DATASET = TRAIN / "yolo_dataset"
RUNS = TRAIN / "runs"

MEDRPG_FINAL = PROJECT_ROOT / "experiments" / "ms_cxr_medrpg_fair_retrain_final_v1"
SINGLE_BOX_DATA = MEDRPG_FINAL / "data" / "single_box_full_phrase"
RAD_EXP = PROJECT_ROOT / "experiments" / "ms_cxr_rad_dino_singlebox_retrain_v1"
RAD_CKPT = PROJECT_ROOT / "training" / "ms_cxr_rad_dino_singlebox_retrain_v1" / "checkpoints"

CLASS_NAMES = yd.CLASS_NAMES
CLASS_TO_ID = yd.CLASS_TO_ID
MAIN5 = yd.MAIN5


def ensure_dirs() -> None:
    for path in [EXP, PRED, MET, CFG, LOGS, REPORT, TRAIN, DATASET, RUNS]:
        path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_singlebox(split: str) -> pd.DataFrame:
    path = SINGLE_BOX_DATA / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["split"] = split
    return df


def row_dicts(split: str) -> List[Dict]:
    rows = []
    df = read_singlebox(split)
    for _, r in df.iterrows():
        rows.append(
            {
                "task_id": str(r["sample_id"]),
                "sample_id": str(r["sample_id"]),
                "dicom_id": str(r["dicom_id"]),
                "subject_id": str(r["subject_id"]),
                "study_id": str(r["study_id"]),
                "image_path": str(r["image_path"]),
                "finding": str(r["finding_label"]),
                "claim_sentence": str(r["phrase_text"]),
                "gold_bbox_xyxy": [float(r["bbox_x1"]), float(r["bbox_y1"]), float(r["bbox_x2"]), float(r["bbox_y2"])],
                "image_width": int(r["image_width"]),
                "image_height": int(r["image_height"]),
                "bbox_type": "ms_cxr_phrase_grounding_bbox",
                "split": split,
            }
        )
    return rows


def copy_or_link(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    if mode in {"symlink", "auto"}:
        try:
            dst.symlink_to(src)
            return "symlink"
        except Exception:
            if mode == "symlink":
                raise
    if mode in {"hardlink", "auto"}:
        try:
            dst.hardlink_to(src)
            return "hardlink"
        except Exception:
            if mode == "hardlink":
                raise
    shutil.copy2(src, dst)
    return "copy"


def build_yolo_dataset(link_mode: str, force: bool) -> pd.DataFrame:
    records = []
    if force and DATASET.exists():
        for sub in ["images", "labels"]:
            target = DATASET / sub
            if target.exists():
                shutil.rmtree(target)
    for split in ["train", "val", "eval"]:
        rows = row_dicts(split)
        image_dir = DATASET / "images" / split
        label_dir = DATASET / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        grouped: Dict[str, List[Dict]] = defaultdict(list)
        for row in rows:
            grouped[row["image_path"]].append(row)
        for image_path, image_rows in grouped.items():
            src = Path(image_path)
            if not src.exists():
                for row in image_rows:
                    records.append({"split": split, "sample_id": row["sample_id"], "status": "missing_image", "image_path": image_path})
                continue
            dst = image_dir / f"{src.stem}{src.suffix.lower()}"
            link_status = copy_or_link(src, dst, link_mode)
            label_path = label_dir / f"{dst.stem}.txt"
            lines = []
            for row in image_rows:
                yolo = yd.xyxy_to_yolo(row["gold_bbox_xyxy"], row["image_width"], row["image_height"])
                if yolo is None:
                    records.append({"split": split, "sample_id": row["sample_id"], "status": "invalid_box", "image_path": image_path})
                    continue
                cls = CLASS_TO_ID[row["finding"]]
                lines.append(f"{cls} {yolo[0]:.8f} {yolo[1]:.8f} {yolo[2]:.8f} {yolo[3]:.8f}")
                records.append(
                    {
                        "split": split,
                        "task_id": row["task_id"],
                        "sample_id": row["sample_id"],
                        "dicom_id": row["dicom_id"],
                        "subject_id": row["subject_id"],
                        "study_id": row["study_id"],
                        "image_path": str(dst),
                        "source_image_path": image_path,
                        "label_path": str(label_path),
                        "finding": row["finding"],
                        "class_id": cls,
                        "gold_x1": row["gold_bbox_xyxy"][0],
                        "gold_y1": row["gold_bbox_xyxy"][1],
                        "gold_x2": row["gold_bbox_xyxy"][2],
                        "gold_y2": row["gold_bbox_xyxy"][3],
                        "image_width": row["image_width"],
                        "image_height": row["image_height"],
                        "claim_sentence": row["claim_sentence"],
                        "bbox_type": row["bbox_type"],
                        "status": "ok",
                        "link_status": link_status,
                    }
                )
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    yaml_text = "\n".join(
        [
            f"path: {DATASET.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/eval",
            "names:",
            *[f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES)],
            "",
        ]
    )
    (DATASET / "ms_cxr_singlebox_yolo.yaml").write_text(yaml_text, encoding="utf-8")
    manifest = pd.DataFrame(records)
    manifest.to_csv(DATASET / "singlebox_yolo_manifest.csv", index=False)
    summary = manifest[manifest["status"].eq("ok")].groupby("split").agg(rows=("sample_id", "count"), images=("image_path", "nunique")).reset_index()
    summary.to_csv(MET / "singlebox_yolo_dataset_summary.csv", index=False)
    return manifest


def train_yolo(
    model_name: str,
    model_tag: str,
    epochs: int,
    imgsz: int,
    batch: int,
    workers: int,
    device: str,
    force: bool,
    seed: int = 42,
) -> Path:
    from ultralytics import YOLO

    run_name = f"{model_tag}_singlebox_e{epochs}_s{seed}"
    best = RUNS / run_name / "weights" / "best.pt"
    if best.exists() and not force:
        return best
    model = YOLO(model_name)
    start = time.time()
    model.train(
        data=str(DATASET / "ms_cxr_singlebox_yolo.yaml"),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        project=str(RUNS),
        name=run_name,
        exist_ok=True,
        device=device,
        pretrained=True,
        seed=seed,
        patience=max(20, min(50, epochs // 2)),
    )
    elapsed = time.time() - start
    if not best.exists():
        raise FileNotFoundError(best)
    write_text(LOGS / f"{run_name}_train_summary.json", json.dumps({"weights": str(best), "time_sec": elapsed}, indent=2))
    return best


def predict_candidates(weights: Path, rows: List[Dict], split: str, model_tag: str, imgsz: int, conf: float, device: str, force: bool) -> Dict[str, List[Dict]]:
    out_csv = PRED / f"{model_tag}_{split}_conf{str(conf).replace('.', 'p')}_candidates.csv"
    if out_csv.exists() and not force:
        df = pd.read_csv(out_csv)
        grouped: Dict[str, List[Dict]] = defaultdict(list)
        for _, r in df.iterrows():
            grouped[str(r["dicom_id"])].append(
                {
                    "class_id": int(r["class_id"]),
                    "score": float(r["score"]),
                    "box": [float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])],
                    "source_model": str(r["source_model"]),
                    "rank": int(r["rank"]),
                }
            )
        return grouped

    from ultralytics import YOLO

    image_by_dicom = {}
    for row in rows:
        image_by_dicom[str(row["dicom_id"])] = row["image_path"]
    items = sorted(image_by_dicom.items())
    model = YOLO(str(weights))
    results = model.predict(
        source=[p for _, p in items],
        imgsz=imgsz,
        conf=conf,
        device=device,
        verbose=False,
        stream=False,
        max_det=300,
    )
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    out_rows = []
    for (dicom_id, image_path), res in zip(items, results):
        preds = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            scores = res.boxes.conf.cpu().numpy()
            for box, c, s in zip(xyxy, cls, scores):
                preds.append({"class_id": int(c), "score": float(s), "box": [float(x) for x in box]})
        preds.sort(key=lambda x: x["score"], reverse=True)
        for rank, pred in enumerate(preds):
            cand = {**pred, "source_model": model_tag, "rank": rank}
            grouped[dicom_id].append(cand)
            out_rows.append(
                {
                    "dicom_id": dicom_id,
                    "image_path": image_path,
                    "class_id": cand["class_id"],
                    "score": cand["score"],
                    "x1": cand["box"][0],
                    "y1": cand["box"][1],
                    "x2": cand["box"][2],
                    "y2": cand["box"][3],
                    "source_model": model_tag,
                    "rank": rank,
                }
            )
    pd.DataFrame(out_rows).to_csv(out_csv, index=False)
    return grouped


def merge_candidates(*candidate_sets: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    merged: Dict[str, List[Dict]] = defaultdict(list)
    for cands in candidate_sets:
        for dicom, rows in cands.items():
            merged[dicom].extend(rows)
    for dicom in list(merged):
        merged[dicom] = sorted(merged[dicom], key=lambda r: float(r["score"]), reverse=True)
    return merged


def evaluate_class_only(rows: List[Dict], candidates: Dict[str, List[Dict]], method: str, split: str) -> pd.DataFrame:
    out = []
    for row in rows:
        class_id = CLASS_TO_ID[row["finding"]]
        cands = [c for c in candidates.get(str(row["dicom_id"]), []) if int(c["class_id"]) == class_id]
        if cands:
            cand = max(cands, key=lambda x: float(x["score"]))
            pred = cand["box"]
            conf = cand["score"]
            missing = False
            source_model = cand.get("source_model", "")
        else:
            pred = [0.0, 0.0, 0.0, 0.0]
            conf = 0.0
            missing = True
            source_model = ""
        gt = ybase.clip_box(row["gold_bbox_xyxy"], row["image_width"], row["image_height"])
        iou = 0.0 if missing else ybase.iou_xyxy(pred, gt)
        out.append(
            {
                "task_id": row["task_id"],
                "sample_id": row["sample_id"],
                "split": split,
                "dicom_id": row["dicom_id"],
                "subject_id": row["subject_id"],
                "study_id": row["study_id"],
                "image_path": row["image_path"],
                "finding": row["finding"],
                "class_id": class_id,
                "claim_sentence": row["claim_sentence"],
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
                "bbox_missing": missing,
                "bbox_invalid": (not missing) and (pred[2] <= pred[0] or pred[3] <= pred[1]),
                "method": method,
                "source_model": source_model,
                "bbox_type": "ms_cxr_phrase_grounding_bbox",
            }
        )
    return pd.DataFrame(out)


def metrics(df: pd.DataFrame, method: str, subset: str = "all8") -> Dict:
    sub = df.copy()
    finding_col = "finding" if "finding" in sub.columns else "finding_label"
    if subset == "main5":
        sub = sub[sub[finding_col].isin(MAIN5)]
    if len(sub) == 0:
        return {"method": method, "subset": subset, "n": 0}
    ious = sub["iou"].astype(float).to_numpy()
    return {
        "method": method,
        "subset": subset,
        "n": int(len(sub)),
        "mean_iou": float(np.mean(ious)),
        "median_iou": float(np.median(ious)),
        "Hit@0.1": float(np.mean(ious >= 0.1)),
        "Hit@0.3": float(np.mean(ious >= 0.3)),
        "Hit@0.5": float(np.mean(ious >= 0.5)),
        "bbox_missing_rate": float(sub["bbox_missing"].astype(bool).mean()) if "bbox_missing" in sub else 0.0,
        "bbox_invalid_rate": float(sub["bbox_invalid"].astype(bool).mean()) if "bbox_invalid" in sub else 0.0,
    }


def load_rad_prediction(variant: str, split: str, device: str, force: bool) -> pd.DataFrame:
    out = PRED / f"rad_dino_{variant}_singlebox_{split}_predictions.csv"
    if out.exists() and not force:
        return pd.read_csv(out)
    ckpt_path = RAD_CKPT / f"rad_dino_{variant}_singlebox.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    rows, tokens, ctx, _ = rad.align_split(variant, split)
    model = PatchHeatmapBBoxHead(int(ckpt["token_dim"]), int(ckpt["ctx_dim"]), hidden=384, dropout=0.1).to(device)
    model.load_state_dict(ckpt["state"])
    pred = rad.predict_heatmap(model, tokens, ctx, rows, f"rad_dino_{variant}_singlebox")
    pred["task_id"] = pred["sample_id"].astype(str)
    pred.to_csv(out, index=False)
    return pred


def dino_map(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    return {
        str(r["sample_id"]): np.asarray([float(r["pred_cx"]), float(r["pred_cy"]), float(r["pred_w"]), float(r["pred_h"])], dtype="float32")
        for _, r in df.iterrows()
    }


def select_rad_variant_by_val(variants: Sequence[str], device: str, force: bool) -> Tuple[str, pd.DataFrame]:
    rows = []
    for variant in variants:
        pred = load_rad_prediction(variant, "val", device, force)
        rows.append(metrics(pred.rename(columns={"finding_label": "finding"}), f"rad_dino_{variant}_singlebox", "all8") | {"variant": variant})
    df = pd.DataFrame(rows).sort_values("mean_iou", ascending=False)
    df.to_csv(MET / "rad_dino_singlebox_val_variant_selection.csv", index=False)
    return str(df.iloc[0]["variant"]), df


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, label: str, reps: int, seed: int) -> Dict:
    aa = a[["sample_id", "subject_id", "iou"]].rename(columns={"iou": "a"})
    bb = b[["sample_id", "iou"]].rename(columns={"iou": "b"})
    df = aa.merge(bb, on="sample_id", how="inner")
    rng = np.random.default_rng(seed)
    groups = df.groupby(df["subject_id"].astype(str)).indices
    subjects = np.array(list(groups.keys()))
    diffs = []
    h03 = []
    h05 = []
    for _ in range(reps):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        idx = []
        for s in sampled:
            idx.extend(groups[s])
        sub = df.iloc[idx]
        diffs.append(float((sub["a"] - sub["b"]).mean()))
        h03.append(float(((sub["a"] >= 0.3).astype(float) - (sub["b"] >= 0.3).astype(float)).mean()))
        h05.append(float(((sub["a"] >= 0.5).astype(float) - (sub["b"] >= 0.5).astype(float)).mean()))
    return {
        "comparison": label,
        "n": int(len(df)),
        "mean_iou_diff": float((df["a"] - df["b"]).mean()),
        "mean_iou_ci_low": float(np.percentile(diffs, 2.5)),
        "mean_iou_ci_high": float(np.percentile(diffs, 97.5)),
        "mean_iou_prob_gt_0": float(np.mean(np.asarray(diffs) > 0)),
        "hit_0_3_diff": float(((df["a"] >= 0.3).astype(float) - (df["b"] >= 0.3).astype(float)).mean()),
        "hit_0_3_ci_low": float(np.percentile(h03, 2.5)),
        "hit_0_3_ci_high": float(np.percentile(h03, 97.5)),
        "hit_0_5_diff": float(((df["a"] >= 0.5).astype(float) - (df["b"] >= 0.5).astype(float)).mean()),
        "hit_0_5_ci_low": float(np.percentile(h05, 2.5)),
        "hit_0_5_ci_high": float(np.percentile(h05, 97.5)),
        "bootstrap_unit": "subject_id_cluster",
        "bootstrap_reps": reps,
    }


def load_medrpg_predictions() -> Dict[str, pd.DataFrame]:
    manifest = read_singlebox("eval")[["sample_id", "finding_label"]].copy()
    paths = {
        "medrpg_fair_full_phrase_s42": MEDRPG_FINAL / "predictions" / "medrpg_fair_full_phrase_s42_eval_predictions.csv",
        "medrpg_fair_rule_context_s42": MEDRPG_FINAL / "predictions" / "medrpg_fair_rule_context_s42_eval_predictions.csv",
        "medrpg_fair_label_only_s42": MEDRPG_FINAL / "predictions" / "medrpg_fair_label_only_s42_eval_predictions.csv",
    }
    out = {}
    for name, path in paths.items():
        df = pd.read_csv(path)
        df["method"] = name
        if "finding" not in df.columns:
            df = df.merge(manifest, on="sample_id", how="left")
            df = df.rename(columns={"finding_label": "finding"})
        out[name] = df
    return out


def write_report(summary: pd.DataFrame, boot: pd.DataFrame, rad_val: pd.DataFrame, yolo_models: Sequence[str]) -> None:
    best = summary.iloc[0]
    med = summary[summary["method"].eq("medrpg_fair_full_phrase_s42")].iloc[0]
    lines = [
        "# MS-CXR Single-box Fair YOLO-DINO Fusion V1",
        "",
        "## 한 줄 결론",
        "",
        (
            f"YOLO와 RAD-DINO를 모두 single-box train 638 기준으로 맞춘 뒤 val 87에서 fusion을 튜닝했다. "
            f"최고 방법은 `{best['method']}`이고 eval 163 mean IoU는 {best['mean_iou']:.4f}이다. "
            f"MedRPG fair full-phrase는 {med['mean_iou']:.4f}이다."
        ),
        "",
        "## 공정 조건",
        "",
        "- train/val/eval: MS-CXR p10-p19 single-box phrase-level 638 / 87 / 163.",
        "- YOLO: single-box train 638만으로 재학습, COCO pretrained YOLO weight에서 시작.",
        "- RAD-DINO: 기존 `ms_cxr_rad_dino_singlebox_retrain_v1` checkpoint 사용. 이 checkpoint도 single-box train 638로 학습된 shallow head다.",
        "- Fusion: single-box val 87에서만 parameter/weight 선택.",
        "- Eval gold는 최종 평가와 bootstrap에만 사용.",
        "- MS-CXR bbox는 phrase-grounding bbox이며 lesion pixel mask가 아니다.",
        "",
        "## 사용 YOLO 모델",
        "",
        ", ".join(yolo_models),
        "",
        "## RAD-DINO val variant 선택",
        "",
        rad_val.to_markdown(index=False),
        "",
        "## 최종 비교",
        "",
        summary.to_markdown(index=False),
        "",
        "## Bootstrap",
        "",
        boot.to_markdown(index=False),
        "",
        "## 해석 주의",
        "",
        "- 이 비교는 같은 train/val/eval split을 사용한 fair single-box 비교다.",
        "- MedRPG released checkpoint 0.777은 official train/val overlap 때문에 여기에 포함하지 않는다.",
        "- YOLO-DINO fusion이 높더라도 full MS-CXR row-level 전체 SOTA라고 쓰면 안 된다. 이 결과는 p10-p19 single-box 163개 subset 기준이다.",
    ]
    write_text(REPORT / "README_KO.md", "\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> None:
    ensure_dirs()
    write_text(CFG / "run_config.json", json.dumps(vars(args), indent=2, ensure_ascii=False))

    manifest_path = DATASET / "singlebox_yolo_manifest.csv"
    if args.build_data or not manifest_path.exists():
        manifest = build_yolo_dataset(args.link_mode, args.force_data)
    else:
        manifest = pd.read_csv(manifest_path)

    train_rows = row_dicts("train")
    val_rows = row_dicts("val")
    eval_rows = row_dicts("eval")
    if (len(train_rows), len(val_rows), len(eval_rows)) != (638, 87, 163):
        raise RuntimeError(f"Unexpected split sizes: {len(train_rows)}, {len(val_rows)}, {len(eval_rows)}")

    weights_by_tag: Dict[str, Path] = {}
    for model in args.models:
        tag = Path(model).stem.replace(".pt", "")
        if args.train_yolo:
            weights_by_tag[tag] = train_yolo(
                model,
                tag,
                args.epochs,
                args.imgsz,
                args.batch,
                args.workers,
                args.device,
                args.force_train,
                args.seed,
            )
        else:
            candidate = RUNS / f"{tag}_singlebox_e{args.epochs}_s{args.seed}" / "weights" / "best.pt"
            if not candidate.exists():
                raise FileNotFoundError(candidate)
            weights_by_tag[tag] = candidate

    val_candidate_sets = []
    eval_candidate_sets = []
    summary_rows = []
    pred_for_boot: Dict[str, pd.DataFrame] = {}
    for tag, weights in weights_by_tag.items():
        vc = predict_candidates(weights, val_rows, "val", tag, args.imgsz, args.pred_conf, args.device, args.force_predict)
        ec = predict_candidates(weights, eval_rows, "eval", tag, args.imgsz, args.pred_conf, args.device, args.force_predict)
        val_candidate_sets.append(vc)
        eval_candidate_sets.append(ec)
        pred = evaluate_class_only(eval_rows, ec, f"{tag}_singlebox_class_only_fair", "eval")
        pred.to_csv(PRED / f"{tag}_singlebox_class_only_eval_predictions.csv", index=False)
        pred_for_boot[f"{tag}_singlebox_class_only_fair"] = pred
        for subset in ["all8", "main5"]:
            row = metrics(pred, f"{tag}_singlebox_class_only_fair", subset)
            row.update({"training_data": "single_box_train_638", "query_variant": "label_only", "model_family": "YOLO detector"})
            summary_rows.append(row)

    val_candidates = merge_candidates(*val_candidate_sets)
    eval_candidates = merge_candidates(*eval_candidate_sets)
    priors = ybase.make_train_priors(train_rows)
    params_by_finding, yolo_grid, yolo_per = yv2.tune(val_rows, val_candidates, priors, args.quick)
    yolo_grid.to_csv(MET / "yolo_rule_context_val_grid.csv", index=False)
    yolo_per.to_csv(MET / "yolo_rule_context_best_params_by_finding.csv", index=False)
    write_text(CFG / "yolo_rule_context_best_params_by_finding.json", json.dumps(params_by_finding, indent=2, ensure_ascii=False))

    yolo_rule_eval = yv2.evaluate_rows_v2(eval_rows, eval_candidates, priors, params_by_finding, "yolo_rule_context_singlebox_fair", "eval")
    yolo_rule_eval.to_csv(PRED / "yolo_rule_context_singlebox_fair_eval_predictions.csv", index=False)
    pred_for_boot["yolo_rule_context_singlebox_fair"] = yolo_rule_eval
    for subset in ["all8", "main5"]:
        row = metrics(yolo_rule_eval, "yolo_rule_context_singlebox_fair", subset)
        row.update({"training_data": "single_box_train_638", "query_variant": "rule_context", "model_family": "YOLO detector + rule rerank"})
        summary_rows.append(row)

    device = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    rad_variant, rad_val = select_rad_variant_by_val(args.rad_variants, device, args.force_rad_predict)
    rad_val_eval = load_rad_prediction(rad_variant, "eval", device, args.force_rad_predict)
    rad_val_eval = rad_val_eval.rename(columns={"finding_label": "finding"})
    pred_for_boot[f"rad_dino_{rad_variant}_singlebox"] = rad_val_eval
    for subset in ["all8", "main5"]:
        row = metrics(rad_val_eval, f"rad_dino_{rad_variant}_singlebox", subset)
        row.update({"training_data": "single_box_train_638", "query_variant": rad_variant, "model_family": "RAD-DINO query head"})
        summary_rows.append(row)

    dino_val = dino_map(load_rad_prediction(rad_variant, "val", device, args.force_rad_predict))
    dino_eval = dino_map(load_rad_prediction(rad_variant, "eval", device, args.force_rad_predict))
    fusion_by_finding, fusion_grid, fusion_per = old_fusion.tune_fusion(
        val_rows, val_candidates, priors, dino_val, params_by_finding, args.quick
    )
    fusion_grid.to_csv(MET / "fusion_val_grid.csv", index=False)
    fusion_per.to_csv(MET / "fusion_best_weights_by_finding.csv", index=False)
    write_text(CFG / "fusion_best_weights_by_finding.json", json.dumps(fusion_by_finding, indent=2, ensure_ascii=False))
    fusion_eval = old_fusion.evaluate_fusion(
        eval_rows,
        eval_candidates,
        priors,
        dino_eval,
        params_by_finding,
        fusion_by_finding,
        f"yolo_dino_{rad_variant}_fusion_singlebox_fair",
        "eval",
    )
    fusion_eval.to_csv(PRED / f"yolo_dino_{rad_variant}_fusion_singlebox_fair_eval_predictions.csv", index=False)
    pred_for_boot[f"yolo_dino_{rad_variant}_fusion_singlebox_fair"] = fusion_eval
    for subset in ["all8", "main5"]:
        row = metrics(fusion_eval, f"yolo_dino_{rad_variant}_fusion_singlebox_fair", subset)
        row.update({"training_data": "single_box_train_638", "query_variant": f"yolo_rule_plus_rad_{rad_variant}", "model_family": "YOLO-DINO fusion"})
        summary_rows.append(row)

    med_preds = load_medrpg_predictions()
    for name, df in med_preds.items():
        pred_for_boot[name] = df
        for subset in ["all8", "main5"]:
            row = metrics(df, name, subset)
            q = "full_phrase" if "full" in name else "rule_context" if "rule" in name else "label_only"
            row.update({"training_data": "single_box_train_638", "query_variant": q, "model_family": "MedRPG"})
            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values(["subset", "mean_iou"], ascending=[True, False])
    summary.to_csv(MET / "main_fair_singlebox_comparison.csv", index=False)

    boot_rows = []
    fusion_name = f"yolo_dino_{rad_variant}_fusion_singlebox_fair"
    for other in ["medrpg_fair_full_phrase_s42", "medrpg_fair_rule_context_s42", "medrpg_fair_label_only_s42", "yolo_rule_context_singlebox_fair", f"rad_dino_{rad_variant}_singlebox"]:
        if fusion_name in pred_for_boot and other in pred_for_boot:
            boot_rows.append(paired_bootstrap(pred_for_boot[fusion_name], pred_for_boot[other], f"{fusion_name} - {other}", args.bootstrap_reps, args.seed))
    if "yolo_rule_context_singlebox_fair" in pred_for_boot and "medrpg_fair_full_phrase_s42" in pred_for_boot:
        boot_rows.append(paired_bootstrap(pred_for_boot["yolo_rule_context_singlebox_fair"], pred_for_boot["medrpg_fair_full_phrase_s42"], "yolo_rule_context_singlebox_fair - medrpg_fair_full_phrase_s42", args.bootstrap_reps, args.seed))
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(MET / "main_fair_singlebox_bootstrap_ci.csv", index=False)

    write_report(summary[summary["subset"].eq("all8")], boot, rad_val, list(weights_by_tag.keys()))

    print("project_root", PROJECT_ROOT)
    print("train_rows", len(train_rows))
    print("val_rows", len(val_rows))
    print("eval_rows", len(eval_rows))
    print("yolo_models", ",".join(weights_by_tag.keys()))
    print("selected_rad_variant", rad_variant)
    all8 = summary[summary["subset"].eq("all8")]
    for _, row in all8.head(10).iterrows():
        print(f"{row['method']} mean_iou={row['mean_iou']:.6f} hit03={row['Hit@0.3']:.6f} hit05={row['Hit@0.5']:.6f}")
    print("summary_path", MET / "main_fair_singlebox_comparison.csv")
    print("bootstrap_path", MET / "main_fair_singlebox_bootstrap_ci.csv")
    print("report_path", REPORT / "README_KO.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-data", action="store_true")
    parser.add_argument("--train-yolo", action="store_true")
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-predict", action="store_true")
    parser.add_argument("--force-rad-predict", action="store_true")
    parser.add_argument("--models", nargs="+", default=["yolov8n.pt", "yolov8s.pt"])
    parser.add_argument("--rad-variants", nargs="+", default=["full_phrase", "rule_context", "label_only"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--link-mode", choices=["auto", "copy", "symlink", "hardlink"], default="auto")
    parser.add_argument("--pred-conf", type=float, default=0.001)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.epochs = min(args.epochs, 20)
        args.bootstrap_reps = min(args.bootstrap_reps, 1000)
    return args


if __name__ == "__main__":
    run(parse_args())

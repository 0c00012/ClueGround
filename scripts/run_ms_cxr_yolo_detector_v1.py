#!/usr/bin/env python
"""Pure YOLO detector baseline for MS-CXR p10-p19 boxes.

This baseline intentionally ignores claim text, rule context, SMM outputs, and
RAD-DINO query conditioning. It trains a class-only object detector from MS-CXR
phrase-grounding bboxes. MS-CXR boxes are not pixel-level lesion masks.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]

EXP_NAME = "ms_cxr_yolo_detector_v1"
EXP = PROJECT_ROOT / "experiments" / EXP_NAME
PRED = EXP / "predictions"
MET = EXP / "metrics"
OVERLAYS = EXP / "overlays"
CONTACT = EXP / "contact_sheets"
TRAIN_ROOT = PROJECT_ROOT / "training" / EXP_NAME
DATASET = TRAIN_ROOT / "yolo_dataset"
RUNS = TRAIN_ROOT / "runs"
REPORT = PROJECT_ROOT / "reports" / EXP_NAME

STAGE1_DATA = PROJECT_ROOT / "training" / "ms_cxr_vfm_localizer_stage1_p10_p19" / "datasets"
SINGLE_BOX = PROJECT_ROOT / "experiments" / "ms_cxr_medrpg_baseline_v1" / "data" / "medrpg_our_p10p19_single_box_full_phrase"

CLASS_NAMES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Lung Opacity",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
MAIN5 = {"Atelectasis", "Consolidation", "Lung Opacity", "Pleural Effusion", "Pneumothorax"}


def ensure_dirs() -> None:
    for path in [EXP, PRED, MET, OVERLAYS, CONTACT, TRAIN_ROOT, DATASET, RUNS, REPORT]:
        path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def source_rows(split: str) -> List[Dict]:
    path = STAGE1_DATA / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    return read_jsonl(path)


def safe_stem(image_path: str) -> str:
    return Path(image_path).stem


def xyxy_to_yolo(box: Sequence[float], iw: float, ih: float) -> Optional[Tuple[float, float, float, float]]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(iw, x1))
    x2 = max(0.0, min(iw, x2))
    y1 = max(0.0, min(ih, y1))
    y2 = max(0.0, min(ih, y2))
    if x2 <= x1 or y2 <= y1 or iw <= 0 or ih <= 0:
        return None
    cx = ((x1 + x2) / 2.0) / iw
    cy = ((y1 + y2) / 2.0) / ih
    w = (x2 - x1) / iw
    h = (y2 - y1) / ih
    return (
        float(np.clip(cx, 0.0, 1.0)),
        float(np.clip(cy, 0.0, 1.0)),
        float(np.clip(w, 1e-6, 1.0)),
        float(np.clip(h, 1e-6, 1.0)),
    )


def copy_or_link_image(src: Path, dst: Path, mode: str) -> str:
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


def build_yolo_dataset(link_mode: str = "copy", force: bool = False) -> pd.DataFrame:
    DATASET.mkdir(parents=True, exist_ok=True)
    records = []
    for split in ["train", "val", "eval"]:
        rows = source_rows(split)
        split_img_dir = DATASET / "images" / split
        split_lab_dir = DATASET / "labels" / split
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lab_dir.mkdir(parents=True, exist_ok=True)

        grouped: Dict[str, List[Dict]] = {}
        for row in rows:
            grouped.setdefault(row["image_path"], []).append(row)

        if force:
            for old in split_lab_dir.glob("*.txt"):
                old.unlink()

        for image_path, image_rows in grouped.items():
            src = Path(image_path)
            if not src.exists():
                records.append({"split": split, "image_path": image_path, "status": "missing_image", "n_boxes": len(image_rows)})
                continue
            dst = split_img_dir / f"{safe_stem(image_path)}{src.suffix.lower()}"
            link_status = copy_or_link_image(src, dst, link_mode)
            label_path = split_lab_dir / f"{dst.stem}.txt"
            lines = []
            for row in image_rows:
                cls = CLASS_TO_ID[row["finding"]]
                yolo = xyxy_to_yolo(row["gold_bbox_xyxy"], row["image_width"], row["image_height"])
                if yolo is None:
                    records.append({"split": split, "image_path": image_path, "status": "invalid_box", "task_id": row["task_id"]})
                    continue
                lines.append(f"{cls} {yolo[0]:.8f} {yolo[1]:.8f} {yolo[2]:.8f} {yolo[3]:.8f}")
                records.append(
                    {
                        "split": split,
                        "task_id": row["task_id"],
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
                        "bbox_type": "ms_cxr_phrase_grounding_bbox",
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
    (DATASET / "ms_cxr_yolo.yaml").write_text(yaml_text, encoding="utf-8")
    manifest = pd.DataFrame(records)
    manifest.to_csv(DATASET / "ms_cxr_yolo_manifest.csv", index=False)
    return manifest


def make_sanity_overlays(manifest: pd.DataFrame, n: int = 20) -> None:
    ok = manifest[manifest["status"] == "ok"].copy()
    samples = ok.groupby("image_path").head(1).sample(min(n, ok["image_path"].nunique()), random_state=42)
    out_paths = []
    colors = ["red", "yellow", "cyan", "lime", "magenta", "orange", "deepskyblue", "white"]
    for i, image_path in enumerate(samples["image_path"].tolist()):
        sub = ok[ok["image_path"] == image_path]
        im = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(im)
        for _, row in sub.iterrows():
            color = colors[int(row["class_id"]) % len(colors)]
            box = [float(row["gold_x1"]), float(row["gold_y1"]), float(row["gold_x2"]), float(row["gold_y2"])]
            draw.rectangle(box, outline=color, width=max(3, int(min(im.size) / 400)))
            draw.text((box[0] + 4, max(0, box[1] - 18)), str(row["finding"]), fill=color)
        out = OVERLAYS / "sanity_yolo_labels" / f"sanity_{i:03d}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        im.thumbnail((1200, 1200))
        im.save(out, quality=92)
        out_paths.append(out)
    make_contact_sheet(out_paths, CONTACT / "yolo_label_sanity_contact_sheet.jpg")


def make_contact_sheet(paths: Sequence[Path], out: Path, cols: int = 5) -> None:
    if not paths:
        return
    thumbs = []
    for path in paths:
        im = Image.open(path).convert("RGB")
        im.thumbnail((360, 360))
        canvas = Image.new("RGB", (360, 360), "black")
        canvas.paste(im, ((360 - im.width) // 2, (360 - im.height) // 2))
        thumbs.append(canvas)
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * 360, rows * 360), "white")
    for idx, im in enumerate(thumbs):
        sheet.paste(im, ((idx % cols) * 360, (idx // cols) * 360))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=92)


def summarize_dataset(manifest: pd.DataFrame) -> Dict:
    ok = manifest[manifest["status"] == "ok"].copy()
    rows = []
    for split in ["train", "val", "eval"]:
        sub = ok[ok["split"] == split]
        rows.append(
            {
                "split": split,
                "rows": int(len(sub)),
                "images": int(sub["image_path"].nunique()),
                "labels": int(len(sub)),
            }
        )
    pd.DataFrame(rows).to_csv(MET / "dataset_split_summary.csv", index=False)
    dist = ok.groupby(["split", "finding"]).size().reset_index(name="n")
    dist.to_csv(MET / "dataset_class_distribution.csv", index=False)
    return {"split_summary": rows, "class_distribution": dist.to_dict("records")}


def run_train(model_name: str, epochs: int, imgsz: int, batch: int, workers: int, name: str, device: str) -> Path:
    from ultralytics import YOLO

    model = YOLO(model_name)
    model.train(
        data=str(DATASET / "ms_cxr_yolo.yaml"),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        project=str(RUNS),
        name=name,
        exist_ok=True,
        device=device,
        pretrained=True,
        seed=42,
        patience=30,
    )
    best = RUNS / name / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(best)
    return best


def run_yolo_val(weights: Path, split: str, imgsz: int, batch: int, device: str, name: str) -> Optional[Path]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    mode = "val" if split == "val" else "test"
    model.val(
        data=str(DATASET / "ms_cxr_yolo.yaml"),
        split=mode,
        imgsz=imgsz,
        batch=batch,
        project=str(RUNS),
        name=name,
        exist_ok=True,
        device=device,
        save_json=False,
    )
    return RUNS / name


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def predict_images(weights: Path, image_paths: List[str], imgsz: int, conf: float, device: str):
    from ultralytics import YOLO

    model = YOLO(str(weights))
    return model.predict(source=image_paths, imgsz=imgsz, conf=conf, device=device, verbose=False, stream=False)


def evaluate_rows(weights: Path, manifest: pd.DataFrame, split: str, imgsz: int, conf: float, device: str, out_name: str) -> pd.DataFrame:
    rows = manifest[(manifest["split"] == split) & (manifest["status"] == "ok")].copy()
    image_paths = rows["image_path"].drop_duplicates().tolist()
    results = predict_images(weights, image_paths, imgsz, conf, device)
    pred_by_image: Dict[str, List[Dict]] = {}
    for image_path, res in zip(image_paths, results):
        # Ultralytics may report list inputs as image0.jpg/image1.jpg, so keep
        # the manifest path by position rather than trusting res.path.
        path = str(Path(image_path))
        preds = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            scores = res.boxes.conf.cpu().numpy()
            for box, c, s in zip(xyxy, cls, scores):
                preds.append({"class_id": int(c), "score": float(s), "box": [float(x) for x in box]})
        pred_by_image[path] = preds

    out_rows = []
    for _, row in rows.iterrows():
        candidates = [p for p in pred_by_image.get(str(Path(row["image_path"])), []) if p["class_id"] == int(row["class_id"])]
        if candidates:
            chosen = sorted(candidates, key=lambda p: p["score"], reverse=True)[0]
            pred_box = chosen["box"]
            score = chosen["score"]
            missing = False
        else:
            pred_box = [0.0, 0.0, 0.0, 0.0]
            score = 0.0
            missing = True
        gt = [row["gold_x1"], row["gold_y1"], row["gold_x2"], row["gold_y2"]]
        iou = 0.0 if missing else iou_xyxy(pred_box, gt)
        out_rows.append(
            {
                "task_id": row["task_id"],
                "sample_id": row.get("sample_id", row["task_id"]),
                "split": split,
                "image_path": row["image_path"],
                "finding": row["finding"],
                "class_id": int(row["class_id"]),
                "gt_x1": gt[0],
                "gt_y1": gt[1],
                "gt_x2": gt[2],
                "gt_y2": gt[3],
                "pred_x1": pred_box[0],
                "pred_y1": pred_box[1],
                "pred_x2": pred_box[2],
                "pred_y2": pred_box[3],
                "confidence": score,
                "iou": iou,
                "hit_0_1": iou >= 0.1,
                "hit_0_3": iou >= 0.3,
                "hit_0_5": iou >= 0.5,
                "bbox_missing": missing,
                "bbox_invalid": (not missing) and (pred_box[2] <= pred_box[0] or pred_box[3] <= pred_box[1]),
                "bbox_type": "ms_cxr_phrase_grounding_bbox",
            }
        )
    pred_df = pd.DataFrame(out_rows)
    pred_df.to_csv(PRED / f"{out_name}_{split}_row_predictions.csv", index=False)
    return pred_df


def metrics_from_predictions(df: pd.DataFrame, method: str, subset: str) -> Dict:
    sub = df.copy()
    if subset == "main5":
        sub = sub[sub["finding"].isin(MAIN5)]
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
        "bbox_missing_rate": float(sub["bbox_missing"].astype(bool).mean()),
        "bbox_invalid_rate": float(sub["bbox_invalid"].astype(bool).mean()),
    }


def sweep_confidence(weights: Path, manifest: pd.DataFrame, imgsz: int, device: str, model_tag: str) -> float:
    rows = []
    best_conf = 0.001
    best_iou = -1.0
    for conf in [0.001, 0.01, 0.03, 0.05, 0.10, 0.20, 0.30]:
        pred = evaluate_rows(weights, manifest, "val", imgsz, conf, device, f"{model_tag}_conf{str(conf).replace('.', 'p')}")
        m = metrics_from_predictions(pred, f"{model_tag}_conf{conf}", "all8")
        m["conf"] = conf
        rows.append(m)
        if m["mean_iou"] > best_iou:
            best_iou = m["mean_iou"]
            best_conf = conf
    pd.DataFrame(rows).to_csv(MET / f"{model_tag}_confidence_sweep_val.csv", index=False)
    return best_conf


def build_single_box_manifest(base_manifest: pd.DataFrame) -> pd.DataFrame:
    csv_path = SINGLE_BOX / "eval.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    sb = pd.read_csv(csv_path)
    base = base_manifest[base_manifest["status"] == "ok"].copy()
    by_ann = base.set_index("task_id")
    rows = []
    # Match by source annotation id when possible; fallback to bbox/image/class.
    for _, row in sb.iterrows():
        match = base[
            (base["source_image_path"].astype(str).str.contains(str(row["dicom_id"]), regex=False))
            & (base["finding"] == row["finding_label"])
            & (np.isclose(base["gold_x1"], row["bbox_x1"]))
            & (np.isclose(base["gold_y1"], row["bbox_y1"]))
            & (np.isclose(base["gold_x2"], row["bbox_x2"]))
            & (np.isclose(base["gold_y2"], row["bbox_y2"]))
            & (base["split"] == "eval")
        ]
        if len(match):
            r = match.iloc[0].to_dict()
            r["sample_id"] = row["sample_id"]
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(DATASET / "single_box_eval_manifest.csv", index=False)
    return out


def write_report(dataset_summary: Dict, train_rows: List[Dict], eval_rows: List[Dict], best_conf: Optional[float]) -> None:
    lines = [
        "# MS-CXR YOLO Detector v1",
        "",
        "## 한 줄 결론",
        "문맥 없이 MS-CXR bbox만으로 YOLO class-only detector baseline을 학습/평가하는 실험이다.",
        "",
        "## 데이터",
        "- MS-CXR p10-p19 row-level split 사용",
        "- MS-CXR bbox는 phrase-grounding bbox이며 pixel-level lesion mask가 아니다.",
        "- 클래스 수: 8",
        "",
        "## Split summary",
        pd.DataFrame(dataset_summary["split_summary"]).to_markdown(index=False),
        "",
    ]
    if train_rows:
        lines += ["## Training runs", pd.DataFrame(train_rows).to_markdown(index=False), ""]
    if eval_rows:
        lines += ["## Eval summary", pd.DataFrame(eval_rows).to_markdown(index=False), ""]
    if best_conf is not None:
        lines += [f"## Val-selected confidence", f"- selected confidence: `{best_conf}`", ""]
    lines += [
        "## 해석 주의",
        "- YOLO는 class-only detector라 claim/location phrase를 보지 않는다.",
        "- 같은 이미지에 같은 class가 여러 bbox로 존재하면 phrase-level row 매칭에는 불리할 수 있다.",
        "- MedRPG single-box 비교는 별도 참고 비교로만 해석한다.",
    ]
    (REPORT / "README_KO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(dataset_summary: Dict, train_rows: List[Dict], eval_rows: List[Dict], best_conf: Optional[float]) -> None:
    """Write a UTF-8 Korean report. This definition intentionally overrides the
    earlier garbled report text left from an encoding-damaged draft."""
    lines = [
        "# MS-CXR YOLO Detector v1",
        "",
        "## 한 줄 결론",
        "문맥, rule-context, SMM, RAD-DINO query 없이 MS-CXR bbox만으로 YOLO class-only detector baseline을 학습/평가한 실험이다.",
        "",
        "## 데이터",
        "- MS-CXR p10-p19 row-level split 사용",
        "- MS-CXR bbox는 phrase-grounding bbox이며 pixel-level lesion mask가 아니다.",
        "- 클래스 수: 8",
        "",
        "## Split summary",
        pd.DataFrame(dataset_summary["split_summary"]).to_markdown(index=False),
        "",
    ]
    if train_rows:
        lines += ["## Training runs", pd.DataFrame(train_rows).to_markdown(index=False), ""]
    if eval_rows:
        lines += ["## Eval summary", pd.DataFrame(eval_rows).to_markdown(index=False), ""]
    if best_conf is not None:
        lines += ["## Val-selected confidence", f"- selected confidence: `{best_conf}`", ""]
    lines += [
        "## 해석 주의",
        "- YOLO는 class-only detector라 claim/location phrase를 보지 않는다.",
        "- 같은 이미지에 같은 class가 여러 bbox로 존재하면 phrase-level row 매칭에는 불리하다.",
        "- MedRPG single-box 비교는 별도 참고 비교로만 해석한다.",
    ]
    (REPORT / "README_KO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-data", action="store_true")
    parser.add_argument("--make-overlays", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--model-tag", default="yolov8n")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--smoke-epochs", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--link-mode", choices=["copy", "symlink", "hardlink", "auto"], default="copy")
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument("--weights", default="")
    parser.add_argument("--skip-sweep", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    manifest_path = DATASET / "ms_cxr_yolo_manifest.csv"
    if args.build_data or not manifest_path.exists():
        manifest = build_yolo_dataset(args.link_mode, args.force_data)
    else:
        manifest = pd.read_csv(manifest_path)
    dataset_summary = summarize_dataset(manifest)
    if args.make_overlays:
        make_sanity_overlays(manifest, n=20)

    train_rows = []
    eval_rows = []
    best_conf = None
    weights: Optional[Path] = Path(args.weights) if args.weights else None

    if args.smoke:
        start = time.time()
        smoke_best = run_train(args.model, args.smoke_epochs, args.imgsz, args.batch, args.workers, f"{args.model_tag}_smoke", args.device)
        train_rows.append({"run": f"{args.model_tag}_smoke", "epochs": args.smoke_epochs, "weights": str(smoke_best), "time_sec": time.time() - start})

    if args.train:
        start = time.time()
        weights = run_train(args.model, args.epochs, args.imgsz, args.batch, args.workers, f"{args.model_tag}_full", args.device)
        train_rows.append({"run": f"{args.model_tag}_full", "epochs": args.epochs, "weights": str(weights), "time_sec": time.time() - start})

    if args.eval:
        if weights is None:
            candidate = RUNS / f"{args.model_tag}_full" / "weights" / "best.pt"
            if not candidate.exists():
                candidate = RUNS / f"{args.model_tag}_smoke" / "weights" / "best.pt"
            weights = candidate
        if not weights.exists():
            raise FileNotFoundError(weights)
        run_yolo_val(weights, "eval", args.imgsz, args.batch, args.device, f"{args.model_tag}_eval_yolo_metrics")
        best_conf = 0.001 if args.skip_sweep else sweep_confidence(weights, manifest, args.imgsz, args.device, args.model_tag)
        eval_pred = evaluate_rows(weights, manifest, "eval", args.imgsz, best_conf, args.device, args.model_tag)
        eval_rows.append(metrics_from_predictions(eval_pred, f"{args.model_tag}_class_only_detector", "all8"))
        eval_rows.append(metrics_from_predictions(eval_pred, f"{args.model_tag}_class_only_detector", "main5"))
        pd.DataFrame(eval_rows).to_csv(MET / f"{args.model_tag}_row_level_iou_summary.csv", index=False)
        per_finding = eval_pred.groupby("finding")["iou"].agg(["count", "mean", "median"]).reset_index()
        per_finding["Hit@0.3"] = eval_pred.groupby("finding")["hit_0_3"].mean().values
        per_finding["Hit@0.5"] = eval_pred.groupby("finding")["hit_0_5"].mean().values
        per_finding.to_csv(MET / f"{args.model_tag}_per_finding_iou.csv", index=False)

        single = build_single_box_manifest(manifest)
        if len(single):
            sb_pred = evaluate_rows(weights, single, "eval", args.imgsz, best_conf, args.device, f"{args.model_tag}_singlebox")
            pd.DataFrame([metrics_from_predictions(sb_pred, f"{args.model_tag}_singlebox_class_only_detector", "all8")]).to_csv(
                MET / f"{args.model_tag}_singlebox_iou_summary.csv", index=False
            )

    write_report(dataset_summary, train_rows, eval_rows, best_conf)
    print("dataset_yaml", DATASET / "ms_cxr_yolo.yaml")
    print("manifest", manifest_path)
    print("report", REPORT / "README_KO.md")


if __name__ == "__main__":
    main()

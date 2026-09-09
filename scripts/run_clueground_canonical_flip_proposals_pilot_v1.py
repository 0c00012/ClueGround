#!/usr/bin/env python
"""Seed-13 proposal-diversity pilot using horizontally flipped YOLO inference.

This is deliberately not WBF, coordinate averaging, a new detector, or a
selection-grid search.  It adds unflipped coordinates from the same four
finding-conditioned YOLO checkpoints as extra candidates, then applies the
already frozen validation-selected YOLO--RAD-DINO fusion and decoder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import run_clueground_vfm_legacy_fusion_unified_3seed_v1 as hybrid
from src.three_task_grounding.contracts import get_protocol
from src.three_task_grounding.metrics import singleton_projection, summarize_protocol


PROTOCOL = "mscxr_multibox_1444"
PROTOCOL_ROOT = (
    PROJECT_ROOT
    / "training"
    / "three_task_clueground_vfm_finding_conditioned_canonical_v3"
    / "protocols"
    / "task_isolated"
)
UPSTREAM_ROOT = PROJECT_ROOT / "experiments" / "clueground_canonical_v3_hybrid_then_moe_3seed_v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "clueground_canonical_flip_proposals_pilot_s13_v1"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--protocol-root", type=Path, default=PROTOCOL_ROOT)
    parser.add_argument("--upstream-root", type=Path, default=UPSTREAM_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def predict_flipped(
    inputs: list[dict[str, Any]], weights: dict[str, Path], device: str
) -> dict[str, list[dict[str, Any]]]:
    from ultralytics import YOLO

    unique = {str(row["dicom_id"]): row for row in inputs}
    rows: dict[str, list[dict[str, Any]]] = {key: [] for key in unique}
    for tag, checkpoint in sorted(weights.items()):
        model = YOLO(str(checkpoint))
        for dicom_id, item in sorted(unique.items()):
            image = cv2.imread(str(item["image_path"]), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(item["image_path"])
            flipped = np.ascontiguousarray(np.fliplr(image))
            result = model.predict(
                source=flipped,
                imgsz=640,
                conf=0.001,
                max_det=100,
                device=device,
                half=torch.cuda.is_available(),
                verbose=False,
            )[0]
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            scores = result.boxes.conf.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            order = np.argsort(-scores, kind="stable")
            width = float(image.shape[1])
            for rank, index in enumerate(order):
                x1, y1, x2, y2 = [float(v) for v in boxes[index]]
                rows[dicom_id].append(
                    {
                        "box": [width - x2, y1, width - x1, y2],
                        "score": float(scores[index]),
                        "class_id": int(classes[index]),
                        "rank": int(rank),
                        "source_model": f"{tag}_flip",
                    }
                )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def predict_side_crops(
    inputs: list[dict[str, Any]], weights: dict[str, Path], device: str, model_tags: tuple[str, ...] = ("yolov8m",)
) -> dict[str, list[dict[str, Any]]]:
    """Infer fixed overlapping left/right image crops as independent proposals.

    Crops are deliberately proposal augmentation, not a coordinate merge.  The
    original 640-pixel detector sees a denser local field, then each resulting
    box is mapped back to the untouched full-image coordinate system.
    """
    from ultralytics import YOLO

    unique = {str(row["dicom_id"]): row for row in inputs}
    rows: dict[str, list[dict[str, Any]]] = {key: [] for key in unique}
    for tag in model_tags:
        checkpoint = weights[tag]
        model = YOLO(str(checkpoint))
        for dicom_id, item in sorted(unique.items()):
            image = cv2.imread(str(item["image_path"]), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(item["image_path"])
            height, width = image.shape[:2]
            crops = (("leftcrop", 0, int(round(width * .60))), ("rightcrop", int(round(width * .40)), width))
            for crop_name, start_x, end_x in crops:
                crop = np.ascontiguousarray(image[:, start_x:end_x])
                result = model.predict(source=crop, imgsz=640, conf=0.001, max_det=100, device=device, half=torch.cuda.is_available(), verbose=False)[0]
                if result.boxes is None:
                    continue
                boxes = result.boxes.xyxy.detach().cpu().numpy(); scores = result.boxes.conf.detach().cpu().numpy(); classes = result.boxes.cls.detach().cpu().numpy().astype(int)
                for rank, index in enumerate(np.argsort(-scores, kind="stable")):
                    x1, y1, x2, y2 = [float(value) for value in boxes[index]]
                    rows[dicom_id].append({"box": [start_x + x1, y1, start_x + x2, y2], "score": float(scores[index]), "class_id": int(classes[index]), "rank": int(rank), "source_model": f"{tag}_{crop_name}"})
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def main() -> None:
    args = parse_args()
    if not args.execute:
        write_json(args.output_root / "RUN_PLAN.json", {"status": "PLAN_ONLY", "seed": args.seed})
        return

    inputs, labels, source_ids = hybrid.load_protocol(args.protocol_root)
    expected = {"train": 813, "val": 124, "eval": 220}
    observed = {split: len(inputs[split]) for split in inputs}
    if observed != expected:
        raise RuntimeError(f"Canonical group contract failed: {observed}")
    seed_root = args.upstream_root / f"seed_{args.seed}" / PROTOCOL
    weights = {
        path.parents[1].name.split("_mscxr_", 1)[0]: path
        for path in seed_root.glob("yolo_runs/*/weights/best.pt")
    }
    if len(weights) != 4:
        raise RuntimeError(f"Expected four frozen YOLO checkpoints, found {sorted(weights)}")
    rows = {split: hybrid.group_rows(inputs[split], labels[split], split) for split in inputs}
    base_candidates = {
        split: hybrid.load_yolo_candidates(seed_root / "yolo_predictions", split)
        for split in inputs
    }
    archive = np.load(seed_root / "rad_dino_legacy" / "predictions_by_split.npz", allow_pickle=True)
    dino: dict[str, dict[str, np.ndarray]] = {}
    for split in inputs:
        task_map = {
            str(task_id): np.asarray(box, dtype=np.float32)
            for task_id, box in zip(archive[f"{split}_ids"], archive[f"{split}_boxes"])
        }
        dino[split] = hybrid.group_dino_map(task_map, source_ids[split])
    fusion_root = seed_root / "legacy_fusion"
    params = json.loads((fusion_root / "yolo_params.json").read_text(encoding="utf-8"))
    fusion = json.loads((fusion_root / "fusion_params.json").read_text(encoding="utf-8"))
    decoder = json.loads((fusion_root / "decoder_params.json").read_text(encoding="utf-8"))
    priors = hybrid.old_base.ybase.make_train_priors(hybrid.expanded_prior_rows(rows["train"]))

    flip = {split: predict_flipped(inputs[split], weights, args.device) for split in ("val", "eval")}
    merged = {split: {key: list(value) for key, value in base_candidates[split].items()} for split in inputs}
    for split in ("val", "eval"):
        for dicom_id, candidates in flip[split].items():
            merged[split].setdefault(dicom_id, []).extend(candidates)

    records = []
    for split in ("val", "eval"):
        prediction, audit = hybrid.decode_predictions(
            rows[split], merged[split], priors, dino[split], params, fusion, decoder
        )
        summary, detail = summarize_protocol(get_protocol(PROTOCOL), inputs[split], labels[split], prediction)
        if split == "eval":
            summary.update(singleton_projection(inputs[split], labels[split], prediction))
        audit.to_csv(args.output_root / f"{split}_cardinality_audit.csv", index=False)
        pd.DataFrame(detail).to_csv(args.output_root / f"{split}_detail.csv", index=False)
        records.append({"split": split, **summary})
        write_json(args.output_root / f"{split}_summary.json", summary)
        with (args.output_root / f"{split}_predictions.jsonl").open("w", encoding="utf-8") as handle:
            for gid, boxes in sorted(prediction.items()):
                handle.write(json.dumps({"group_id": gid, "pred_boxes_xyxy": boxes}) + "\n")
    pd.DataFrame(records).to_csv(args.output_root / "summary.csv", index=False)
    eval_summary = next(row for row in records if row["split"] == "eval")
    status = {
        "status": "complete",
        "seed": args.seed,
        "method": "four frozen finding-conditioned YOLO models + horizontally flipped proposal augmentation + frozen RAD-DINO/fusion/decoder",
        "not_used": ["WBF", "coordinate averaging", "new foundation", "YOLO1024", "new YOLO weights", "selection grid"],
        "base_candidate_counts": {split: int(sum(len(v) for v in base_candidates[split].values())) for split in ("val", "eval")},
        "flip_candidate_counts": {split: int(sum(len(v) for v in flip[split].values())) for split in ("val", "eval")},
        "seed13_gate_pass": float(eval_summary["coverage_mean_iou"]) >= 0.5,
        **eval_summary,
    }
    write_json(args.output_root / "RUN_STATUS.json", status)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_repro.controlled_direct_matrix import (
    DEFAULT_OUTPUT,
    MODELS,
    PROTOCOLS,
    expected_artifact,
    write_json,
)
from src.baseline_repro.evaluator import evaluate_rows


def clip_box(box: list[float], width: float, height: float) -> list[float]:
    x1, y1, x2, y2 = box
    return [
        max(0.0, min(width, x1)),
        max(0.0, min(height, y1)),
        max(0.0, min(width, x2)),
        max(0.0, min(height, y2)),
    ]


def cxcywh_norm_to_xyxy(row: Any, width: float, height: float, prefix: str) -> list[float]:
    cx = float(getattr(row, f"{prefix}_cx"))
    cy = float(getattr(row, f"{prefix}_cy"))
    w = float(getattr(row, f"{prefix}_w"))
    h = float(getattr(row, f"{prefix}_h"))
    return clip_box([(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height], width, height)


def prediction_map(output_root: Path, model: str, protocol: str, seed: int, canonical: pd.DataFrame) -> dict[str, list[list[float]]]:
    path = expected_artifact(output_root, model, protocol, seed)
    if not path.is_file():
        raise FileNotFoundError(path)
    output: dict[str, list[list[float]]] = {}
    if model == "transvg":
        frame = pd.read_csv(path)
        if len(frame) != len(canonical):
            raise RuntimeError(f"TransVG denominator mismatch: {len(frame)} != {len(canonical)}")
        for source, pred in zip(canonical.itertuples(), frame.itertuples()):
            if Path(str(pred.image_path)).stem != str(source.dicom_id) or str(pred.phrase).strip() != str(source.phrase).strip():
                raise RuntimeError("TransVG prediction order/identity mismatch")
            box = [
                float(pred.pred_norm_x1) * source.image_width,
                float(pred.pred_norm_y1) * source.image_height,
                float(pred.pred_norm_x2) * source.image_width,
                float(pred.pred_norm_y2) * source.image_height,
            ]
            output[str(source.group_id)] = [clip_box(box, source.image_width, source.image_height)]
    elif model == "medrpg":
        saved = torch.load(path, map_location="cpu", weights_only=False)
        available_ids = {int(key) for key in saved}
        expected_ids = set(range(1, len(canonical) + 1))
        if available_ids != expected_ids:
            missing = sorted(expected_ids - available_ids)
            extra = sorted(available_ids - expected_ids)
            raise RuntimeError(
                f"MedRPG prediction ID mismatch: missing={missing[:10]} extra={extra[:10]}"
            )
        for index, source in enumerate(canonical.itertuples(), start=1):
            obj = saved.get(index)
            raw = [] if obj is None else [float(value) for value in obj.get("pbox", [])[:4]]
            if len(raw) != 4:
                output[str(source.group_id)] = []
                continue
            box = [raw[0] / 640 * source.image_width, raw[1] / 640 * source.image_height, raw[2] / 640 * source.image_width, raw[3] / 640 * source.image_height]
            output[str(source.group_id)] = [clip_box(box, source.image_width, source.image_height)]
    elif model == "reclmis":
        frame = pd.read_csv(path)
        for row in frame.itertuples():
            output[str(row.group_id)] = json.loads(row.pred_boxes_xyxy)
    elif model == "mdetr_style":
        frame = pd.read_csv(path)
        if len(frame) != len(canonical):
            raise RuntimeError(f"MDETR denominator mismatch: {len(frame)} != {len(canonical)}")
        for source, pred in zip(canonical.itertuples(), frame.itertuples()):
            if Path(str(pred.img_path)).stem != str(source.dicom_id) or str(pred.phrase).strip() != str(source.phrase).strip():
                raise RuntimeError("MDETR prediction order/identity mismatch")
            output[str(source.group_id)] = [cxcywh_norm_to_xyxy(pred, source.image_width, source.image_height, "pred")]
    else:
        raise ValueError(model)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--protocol", choices=PROTOCOLS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    canonical = pd.read_csv(output_root / "data" / args.protocol / "eval_canonical.csv")
    predictions = prediction_map(output_root, args.model, args.protocol, args.seed, canonical)
    samples = []
    prediction_rows = []
    common_rows = []
    for row in canonical.itertuples():
        group_id = str(row.group_id)
        gold = json.loads(row.gold_boxes_json)
        boxes = predictions.get(group_id, [])
        samples.append({"query_id": group_id, "subject_id": str(row.subject_id), "finding": row.finding, "gold_boxes": gold})
        prediction_rows.append({"query_id": group_id, "pred_boxes": boxes})
        common_rows.append({"group_id": group_id, "pred_boxes_json": json.dumps(boxes), "n_pred": len(boxes)})
    summary, detail = evaluate_rows(samples, prediction_rows, force_single_box=args.protocol == "singlebox_888")
    if args.model == "transvg" and args.protocol == "singlebox_888":
        native_mean = float(pd.read_csv(expected_artifact(output_root, args.model, args.protocol, args.seed))["iou"].mean())
        if abs(native_mean - float(summary["mean_iou"])) > 1e-6:
            raise RuntimeError(
                f"TransVG native/common metric mismatch: native={native_mean} common={summary['mean_iou']}"
            )
    expected = 163 if args.protocol == "singlebox_888" else 220
    summary.update(
        {
            "model": args.model,
            "protocol": args.protocol,
            "seed": args.seed,
            "seed_scope": "full model training",
            "training_protocol": f"direct-{args.protocol}",
            "expected_denominator": expected,
            "prediction_ids": len(predictions),
            "denominator_pass": len(canonical) == expected and len(predictions) == expected and summary["n"] == expected,
            "native_artifact": str(expected_artifact(output_root, args.model, args.protocol, args.seed)),
        }
    )
    run = output_root / "runs" / args.protocol / args.model / f"seed_{args.seed}" / "common_eval"
    run.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(common_rows).to_csv(run / "predictions.csv", index=False)
    pd.DataFrame(detail).to_csv(run / "per_query_metrics.csv", index=False)
    pd.DataFrame([summary]).to_csv(run / "metrics.csv", index=False)
    write_json(run / "RUN_STATUS.json", summary)
    if not summary["denominator_pass"]:
        raise RuntimeError(f"Denominator contract failed: {summary}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

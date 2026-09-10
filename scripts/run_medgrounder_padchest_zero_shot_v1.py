#!/usr/bin/env python
"""Zero-shot MedGrounder on the PadChest-GR external protocol (no retraining).

Applies the sealed MS-CXR-1444 MedGrounder checkpoints
(``experiments/baseline_faithful_reproduction/20260712_v2/runs/medgrounder/local_multibox/seed_S``)
to the PadChest-GR protocol written by the external-evaluation bundle, using
the MedGrounder data loader and post-processing exactly as in the sealed
evaluation (640 letterbox, grounding threshold 0.8, WBF iou 0.1).  Predictions
are mapped back from the centred-square letterbox frame to pixels and scored
with the common ClueGround evaluator (multi-box aware, plus patient-cluster
bootstrap), so the numbers are directly comparable with the ClueGround
zero-shot rows.

Run with the MedGrounder virtual environment (``.venv_smm``).
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_medgrounder_faithful_v2 as trainer  # noqa: E402
from scripts import run_medgrounder_fairness_audit_v1 as medgrounder_utils  # noqa: E402
from src.baseline_repro.evaluator import evaluate_rows, patient_cluster_bootstrap  # noqa: E402
from src.baseline_repro.hf_pins import enable_offline_hf, pin_record  # noqa: E402
from src.baseline_repro.protocols import sha256_file, write_json  # noqa: E402

SEEDS = (13, 42, 2026)
METRICS = ("coverage_iou", "exact_union_iou", "set_f1_optimal_0_3", "set_f1_optimal_0_5")
CHECKPOINT_ROOT = PROJECT_ROOT / "experiments" / "baseline_faithful_reproduction" / "20260712_v2" / "runs" / "medgrounder" / "local_multibox"
DEFAULT_PROTOCOL = PROJECT_ROOT / "experiments" / "padchest_gr_external_v1" / "protocol_strict"


def norm_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def write_annotation(flat: pd.DataFrame, path: Path) -> pd.DataFrame:
    """MedGrounder CSV: one row per gold box; the loader groups by (label_text, path)."""
    rows = []
    for r in flat.itertuples():
        for k, box in enumerate(json.loads(r.gold_boxes_json)):
            x1, y1, x2, y2 = [float(v) for v in box]
            rows.append({"sample_id": f"{r.group_id}::{k}", "group_id": r.group_id, "split": "test", "label_text": norm_text(r.phrase), "path": str(r.image_path),
                         "category_name": r.finding, "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1, "image_width": int(r.image_width), "image_height": int(r.image_height),
                         "subject_id": r.subject_id, "study_id": r.study_id, "dicom_id": r.dicom_id, "ms_cxr_annotation_id": "", "bbox_type": "padchest_gr_sentence_box", "dataset_variant": "padchest_gr_external"})
    frame = pd.DataFrame(rows)
    # the loader builds train/val/test loaders; give train and val one placeholder row each (never used for learning here)
    placeholder = frame.iloc[[0]].copy()
    for split in ("train", "val"):
        p = placeholder.copy()
        p["split"], p["sample_id"] = split, f"placeholder_{split}"
        frame = pd.concat([frame, p], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def letterbox_norm_cxcywh_to_pixels(box: list[float], width: float, height: float) -> list[float]:
    cx, cy, bw, bh = [float(v) for v in box]
    side = max(width, height)
    ox, oy = (side - width) / 2.0, (side - height) / 2.0
    return [max(0.0, (cx - bw / 2) * side - ox), max(0.0, (cy - bh / 2) * side - oy), min(width, (cx + bw / 2) * side - ox), min(height, (cy + bh / 2) * side - oy)]


def run_seed(seed: int, protocol_root: Path, out_root: Path, annotation: Path, flat: pd.DataFrame, device: torch.device, batch_size: int) -> dict[str, Any]:
    seed_root = out_root / f"seed_{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    checkpoint = CHECKPOINT_ROOT / f"seed_{seed}" / "checkpoint" / "best.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    trainer.set_seed(seed)
    config_args = argparse.Namespace(protocol="local_multibox", protocol_root=PROJECT_ROOT / "experiments" / "baseline_faithful_reproduction" / "20260712_v2", batch_size=batch_size, epochs=15, limit_train_samples=0, dev_only=True, dev_annotation_file=annotation.resolve())
    cfg = trainer.build_config(config_args)
    medgrounder_utils.add_medgrounder_to_path()
    from dataloaders.dataset_builder import get_dataloaders

    _train, _val, test_loaders = get_dataloaders(cfg)
    test_loader = test_loaders["gmpg_mscxr"]
    model, _, postprocessor = medgrounder_utils.build_model_and_load(cfg, checkpoint, device)
    prediction_path = seed_root / "test_predictions.csv"
    native = medgrounder_utils.evaluate_model(cfg, model, postprocessor, test_loader, device, save_predictions=prediction_path)
    del model
    torch.cuda.empty_cache()

    # map predictions back to protocol groups (same sentence + image -> same prediction)
    preds = pd.read_csv(prediction_path)
    by_key: dict[tuple[str, str], list[list[float]]] = {}
    for r in preds.itertuples():
        by_key[(Path(str(r.img_path)).stem, norm_text(r.phrase))] = json.loads(r.pred_boxes_cxcywh)
    samples, predictions, out_rows, missing = [], [], [], 0
    for r in flat.itertuples():
        key = (str(r.dicom_id), norm_text(r.phrase))
        raw = by_key.get(key)
        if raw is None:
            missing += 1
            boxes = []
        else:
            boxes = [letterbox_norm_cxcywh_to_pixels(b, float(r.image_width), float(r.image_height)) for b in raw]
        samples.append({"query_id": str(r.group_id), "subject_id": str(r.subject_id), "finding": str(r.finding), "gold_boxes": json.loads(r.gold_boxes_json)})
        predictions.append({"query_id": str(r.group_id), "pred_boxes": boxes})
        out_rows.append({"group_id": r.group_id, "matrix_group_id": r.matrix_group_id, "pred_boxes_json": json.dumps(boxes), "n_pred": len(boxes)})
    pd.DataFrame(out_rows).to_csv(seed_root / "predictions_pixels.csv", index=False)
    summary, detail = evaluate_rows(samples, predictions, force_single_box=False)
    detail = pd.DataFrame(detail)
    detail["method"], detail["seed"] = "medgrounder", seed
    detail.to_csv(seed_root / "per_row_detail.csv", index=False)
    ci = patient_cluster_bootstrap(detail, identity_columns=["method", "seed"], metric_columns=list(METRICS), cluster_column="subject_id", reps=2000)
    ci.to_csv(seed_root / "per_seed_ci.csv", index=False)
    detail["ref_count"] = np.where(detail["n_gt"].astype(int) == 1, "single (1 box)", "multi (2+ boxes)")
    result = {
        "seed": seed, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint), "n_rows": len(flat), "missing_predictions": missing,
        "native_metrics": native, **{m: float(summary[m]) for m in METRICS}, "mean_pred_count": float(detail["n_pred"].mean()),
        "per_finding": {str(f): {"n": int(len(p)), **{m: float(p[m].mean()) for m in METRICS}} for f, p in detail.groupby("finding")},
        "per_reference_count": {str(k): {"n": int(len(p)), **{m: float(p[m].mean()) for m in METRICS}} for k, p in detail.groupby("ref_count")},
    }
    write_json(seed_root / "SEED_RESULT.json", result)
    print(f"[medgrounder] seed {seed}: coverage {result['coverage_iou']:.4f} union {result['exact_union_iou']:.4f} f1@.3 {result['set_f1_optimal_0_3']:.4f} f1@.5 {result['set_f1_optimal_0_5']:.4f} (missing {missing})", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-root", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    # trainer.build_config() chdirs into third_party/MedGrounder; keep every path absolute
    args.protocol_root = args.protocol_root.resolve()
    args.out_root = args.out_root.resolve()
    args.out_root.mkdir(parents=True, exist_ok=True)
    enable_offline_hf()
    pin_record("thomas-sounack/BioClinical-ModernBERT-base", require_main_ref=True)
    flat = pd.read_csv(args.protocol_root / "eval_padchest_gr.csv")
    annotation_path = args.out_root / "medgrounder_padchest_annotation.csv"
    annotation = write_annotation(flat, annotation_path)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    per_seed = []
    for seed in args.seeds:
        path = args.out_root / f"seed_{seed}" / "SEED_RESULT.json"
        per_seed.append(json.loads(path.read_text(encoding="utf-8")) if path.exists() else run_seed(seed, args.protocol_root, args.out_root, annotation_path, flat, device, args.batch_size))
    agg = {m: {"mean": float(np.mean([r[m] for r in per_seed])), "std": float(np.std([r[m] for r in per_seed], ddof=1)) if len(per_seed) > 1 else float("nan"), "per_seed": [r[m] for r in per_seed]} for m in METRICS}
    findings = sorted({f for r in per_seed for f in r["per_finding"]})
    per_finding = {f: {"n": per_seed[0]["per_finding"][f]["n"], **{m: float(np.mean([r["per_finding"][f][m] for r in per_seed])) for m in METRICS}} for f in findings}
    per_ref = {k: {"n": per_seed[0]["per_reference_count"][k]["n"], **{m: float(np.mean([r["per_reference_count"][k][m] for r in per_seed])) for m in METRICS}} for k in per_seed[0]["per_reference_count"]}
    write_json(args.out_root / "SUMMARY.json", {"method": "medgrounder_zero_shot", "protocol_root": str(args.protocol_root), "n_rows": per_seed[0]["n_rows"], "seeds": args.seeds, "aggregate": agg, "per_finding": per_finding, "per_reference_count": per_ref,
                                                "annotation_rows": int(len(annotation)), "note": "sealed MS-CXR-1444 MedGrounder checkpoints applied without retraining; MedGrounder loader (640 letterbox), threshold 0.8, WBF 0.1; scored with the common evaluator in pixel space"})
    print("MedGrounder zero-shot:", {m: f"{v['mean']:.4f} +/- {v['std']:.4f}" for m, v in agg.items()})


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Seed-gated neural selector for the fixed finding-conditioned proposal pool.

The four YOLO checkpoints, frozen RAD-DINO head, validation decoder, and box
coordinates are untouched.  A small MLP only orders existing fused candidates
using detector confidence, YOLO--DINO agreement, geometry, query context, and
the permitted finding category.  It is an alternative to the rejected HGB
ranker, not a new foundation model or coordinate blender.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import run_clueground_vfm_legacy_fusion_unified_3seed_v1 as hybrid
from scripts.run_clueground_canonical_flip_proposals_pilot_v1 import predict_side_crops
from src.three_task_grounding.contracts import get_protocol
from src.three_task_grounding.metrics import singleton_projection, summarize_protocol


PROTOCOL = "mscxr_multibox_1444"
PROTOCOL_ROOT = (
    PROJECT_ROOT / "training" / "three_task_clueground_vfm_finding_conditioned_canonical_v3" / "protocols" / "task_isolated"
)
UPSTREAM_ROOT = PROJECT_ROOT / "experiments" / "clueground_canonical_v3_hybrid_then_moe_3seed_v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "clueground_canonical_neural_agreement_selector_pilot_s13_v1"
FINDINGS = ("Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Lung Opacity", "Pleural Effusion", "Pneumonia", "Pneumothorax")
SOURCES = ("yolov8s", "yolov8m", "yolo11s", "yolo11m")


class Selector(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 96), nn.LayerNorm(96), nn.GELU(), nn.Dropout(0.1), nn.Linear(96, 48), nn.GELU(), nn.Linear(48, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--protocol-root", type=Path, default=PROTOCOL_ROOT)
    p.add_argument("--upstream-root", type=Path, default=UPSTREAM_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--side-crop-yolov8m", action="store_true", help="Append independent left/right YOLOv8m crop proposals; no box merge.")
    p.add_argument("--extra-yolo-predictions", type=Path, default=None, help="Append a separately trained candidate-only YOLO directory for an upstream proposal pilot.")
    p.add_argument("--execute", action="store_true")
    return p.parse_args()


def feature(row: dict[str, Any], candidate: dict[str, Any], dino_box: np.ndarray | None) -> np.ndarray:
    box = np.asarray(candidate["box_norm"], dtype=np.float32)
    cx, cy, bw, bh = [float(x) for x in box]
    area = max(bw * bh, 1e-6)
    dino = hybrid.old_fusion.base.iou_norm(box, dino_box) if dino_box is not None else 0.0
    source = str(candidate.get("source_model", ""))
    source_base = source.replace("_flip", "")
    source_hot = [float(source_base == item) for item in SOURCES]
    finding_hot = [float(str(row["finding"]) == item) for item in FINDINGS]
    lat = str(row.get("query_laterality", "unknown"))
    lat_hot = [float(lat == item) for item in ("right", "left", "bilateral", "unknown")]
    return np.asarray([
        math.tanh(float(candidate["score"]) / 5.0), dino, cx, cy, bw, bh, area,
        float(candidate.get("rank", 0)) / 100.0,
        *lat_hot, *source_hot, *finding_hot,
    ], dtype=np.float32)


def make_context(args: argparse.Namespace) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, list[list[float]]]], dict[str, list[dict[str, Any]]], dict[str, dict[str, list[dict[str, Any]]]], dict[str, dict[str, np.ndarray]], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    inputs, labels, source_ids = hybrid.load_protocol(args.protocol_root)
    if {split: len(inputs[split]) for split in inputs} != {"train": 813, "val": 124, "eval": 220}:
        raise RuntimeError("Canonical 1444 group contract failed")
    root = args.upstream_root / f"seed_{args.seed}" / PROTOCOL
    rows = {split: hybrid.group_rows(inputs[split], labels[split], split) for split in inputs}
    candidates = {split: hybrid.load_yolo_candidates(root / "yolo_predictions", split) for split in inputs}
    archive = np.load(root / "rad_dino_legacy" / "predictions_by_split.npz", allow_pickle=True)
    dino = {}
    for split in inputs:
        task_map = {str(i): np.asarray(b, dtype=np.float32) for i, b in zip(archive[f"{split}_ids"], archive[f"{split}_boxes"])}
        dino[split] = hybrid.group_dino_map(task_map, source_ids[split])
    fusion_root = root / "legacy_fusion"
    params = json.loads((fusion_root / "yolo_params.json").read_text(encoding="utf-8"))
    fusion = json.loads((fusion_root / "fusion_params.json").read_text(encoding="utf-8"))
    decoder = json.loads((fusion_root / "decoder_params.json").read_text(encoding="utf-8"))
    priors = hybrid.old_base.ybase.make_train_priors(hybrid.expanded_prior_rows(rows["train"]))
    return inputs, labels, rows, candidates, dino, priors, params, fusion, decoder


def candidate_rows(rows: list[dict[str, Any]], labels: dict[str, list[list[float]]], candidates: dict[str, list[dict[str, Any]]], dino: dict[str, np.ndarray], priors: dict[str, Any], params: dict[str, Any], fusion: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []; targets: list[float] = []
    for row in rows:
        gid = str(row["group_id"]); p = params.get(row["finding"], params["__global__"]); f = fusion.get(row["finding"], fusion["__global__"])
        for candidate in hybrid.score_legacy_candidates(row, candidates, priors, dino.get(gid), p, f):
            features.append(feature(row, candidate, dino.get(gid)))
            targets.append(max(hybrid.iou_xyxy(candidate["box"], gold) for gold in labels[gid]))
    return np.stack(features), np.asarray(targets, dtype=np.float32)


def main() -> None:
    args = parse_args()
    if not args.execute:
        write_json(args.output_root / "RUN_PLAN.json", {"status": "PLAN_ONLY", "seed": args.seed}); return
    set_seed(args.seed)
    inputs, labels, rows, candidates, dino, priors, params, fusion, decoder = make_context(args)
    crop_counts: dict[str, int] = {}
    if args.side_crop_yolov8m:
        upstream = args.upstream_root / f"seed_{args.seed}" / PROTOCOL
        weights = {
            path.parents[1].name.split("_mscxr_", 1)[0]: path
            for path in upstream.glob("yolo_runs/*/weights/best.pt")
        }
        if "yolov8m" not in weights:
            raise RuntimeError(f"Missing frozen YOLOv8m checkpoint: {sorted(weights)}")
        for split in ("train", "val", "eval"):
            crop = predict_side_crops(inputs[split], weights, "0" if torch.cuda.is_available() else "cpu")
            crop_counts[split] = int(sum(len(value) for value in crop.values()))
            for dicom_id, values in crop.items():
                candidates[split].setdefault(dicom_id, []).extend(values)
    extra_counts: dict[str, int] = {}
    if args.extra_yolo_predictions is not None:
        if not args.extra_yolo_predictions.exists():
            raise FileNotFoundError(args.extra_yolo_predictions)
        for split in ("train", "val", "eval"):
            extra = hybrid.load_yolo_candidates(args.extra_yolo_predictions, split)
            extra_counts[split] = int(sum(len(value) for value in extra.values()))
            if not extra_counts[split]:
                raise RuntimeError(f"No extra candidate rows for {split}: {args.extra_yolo_predictions}")
            for dicom_id, values in extra.items():
                candidates[split].setdefault(dicom_id, []).extend(values)
    x_train, y_train = candidate_rows(rows["train"], labels["train"], candidates["train"], dino["train"], priors, params, fusion)
    mean, std = x_train.mean(0), x_train.std(0); std = np.maximum(std, 1e-5)
    x_train = (x_train - mean) / std
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Selector(x_train.shape[1]).to(device); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)), batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    original_ranker = hybrid.score_legacy_candidates
    best_state: dict[str, torch.Tensor] | None = None; best_val = -1.0; stale = 0; log = []

    def learned_ranker(row: dict[str, Any], pool: dict[str, list[dict[str, Any]]], prior: dict[str, Any], dino_box: np.ndarray | None, p: dict[str, Any], f: dict[str, Any]) -> list[dict[str, Any]]:
        ranked = original_ranker(row, pool, prior, dino_box, p, f)
        if not ranked: return ranked
        feats = np.stack([feature(row, item, dino_box) for item in ranked]); feats = (feats - mean) / std
        with torch.no_grad(): scores = model(torch.from_numpy(feats).float().to(device)).detach().cpu().numpy()
        for item, score in zip(ranked, scores): item["score"] = float(score)
        return sorted(ranked, key=lambda item: (-item["score"], item["source_model"], item["rank"]))

    try:
        for epoch in range(1, args.epochs + 1):
            model.train(); losses = []
            for xb, yb in loader:
                pred = model(xb.float().to(device)); loss = torch.nn.functional.smooth_l1_loss(pred, yb.float().to(device))
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); losses.append(float(loss.detach().cpu()))
            model.eval(); hybrid.score_legacy_candidates = learned_ranker
            val_pred, _ = hybrid.decode_predictions(rows["val"], candidates["val"], priors, dino["val"], params, fusion, decoder)
            val_summary, _ = summarize_protocol(get_protocol(PROTOCOL), inputs["val"], labels["val"], val_pred)
            val_score = float(val_summary["coverage_mean_iou"])
            log.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_coverage": val_score, "val_f1_0_5": float(val_summary["set_f1_0_5"])})
            if val_score > best_val:
                best_val = val_score; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; stale = 0
            else: stale += 1
            if stale >= 15: break
    finally:
        hybrid.score_legacy_candidates = original_ranker
    if best_state is None: raise RuntimeError("No selector checkpoint")
    model.load_state_dict(best_state); model.eval(); hybrid.score_legacy_candidates = learned_ranker
    try:
        pred, audit = hybrid.decode_predictions(rows["eval"], candidates["eval"], priors, dino["eval"], params, fusion, decoder)
        summary, detail = summarize_protocol(get_protocol(PROTOCOL), inputs["eval"], labels["eval"], pred); summary.update(singleton_projection(inputs["eval"], labels["eval"], pred))
    finally:
        hybrid.score_legacy_candidates = original_ranker
    args.output_root.mkdir(parents=True, exist_ok=True); pd.DataFrame(log).to_csv(args.output_root / "train_log.csv", index=False); audit.to_csv(args.output_root / "eval_cardinality_audit.csv", index=False); pd.DataFrame(detail).to_csv(args.output_root / "eval_detail.csv", index=False)
    torch.save({"state": best_state, "mean": mean, "std": std, "feature_dim": int(x_train.shape[1]), "selection": "canonical val124 coverage only"}, args.output_root / "best.pt")
    write_json(args.output_root / "RUN_STATUS.json", {"status": "complete", "seed": args.seed, "method": "frozen four-YOLO/RAD-DINO proposals + neural agreement selector" + (" + independent YOLOv8m left/right crop proposals" if args.side_crop_yolov8m else "") + (" + class-balanced YOLO proposal source" if args.extra_yolo_predictions is not None else ""), "selection_split": "val124 only", "not_used": ["HGB", "WBF", "coordinate averaging", "new foundation", "YOLO1024", "large finding grid"], "n_train_candidates": int(len(y_train)), "side_crop_candidate_counts": crop_counts, "extra_yolo_candidate_counts": extra_counts, "best_val_coverage": best_val, "seed_gate_pass": float(summary["coverage_mean_iou"]) >= .5, **summary})


if __name__ == "__main__": main()

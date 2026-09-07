#!/usr/bin/env python
"""Chest ImaGenome finding-region label-only vs rule-context ablation.

This experiment reuses the existing 10k Chest ImaGenome weak-region split and
RAD-DINO feature cache, but evaluates only the finding-region subset:

- pneumothorax
- pleural effusion
- lung opacity
- atelectasis
- consolidation

The comparison is intentionally narrow and fair:

1. label_only_heatmap:
   frozen RAD-DINO patch tokens + target finding label only.

2. rule_context_heatmap:
   frozen RAD-DINO patch tokens + target finding label + rule-parsed claim
   context, such as laterality, vertical region, anatomy region, severity, and
   uncertainty.

Chest ImaGenome boxes are weak region/reference boxes, not pixel-level lesion
masks and not gold lesion contours.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from models_ms_cxr_vfm_localizer import PatchHeatmapBBoxHead, bbox_loss  # noqa: E402


EXP_NAME = "chest_imagenome_finding_region_rule_ablation_v1"
BASE_EXP_NAME = "chest_imagenome_10k_vfm_localizer_v1"
BASE_DATA = PROJECT_ROOT / "training" / BASE_EXP_NAME / "datasets"
BASE_FEAT = PROJECT_ROOT / "features" / BASE_EXP_NAME
BASE_PRED = PROJECT_ROOT / "experiments" / BASE_EXP_NAME / "predictions"

EXP = PROJECT_ROOT / "experiments" / EXP_NAME
PRED = EXP / "predictions"
MET = EXP / "metrics"
TRAIN = PROJECT_ROOT / "training" / EXP_NAME
DATA = TRAIN / "datasets"
RUNS = TRAIN / "runs"
CKPT = TRAIN / "checkpoints"
REPORT = PROJECT_ROOT / "reports" / EXP_NAME

SPLITS = ("train", "val", "eval")
SEED = 20260623
FINDING_TARGETS = [
    "atelectasis",
    "consolidation",
    "lung opacity",
    "pleural effusion",
    "pneumothorax",
]
CONTEXT_FIELDS = [
    "finding_normalized",
    "laterality",
    "vertical_region",
    "anatomy_region",
    "severity",
    "uncertainty",
]


def ensure_dirs() -> None:
    for p in [EXP, PRED, MET, TRAIN, DATA, RUNS, CKPT, REPORT]:
        p.mkdir(parents=True, exist_ok=True)


def jd(row: Dict) -> str:
    return json.dumps(row, ensure_ascii=False, separators=(",", ":"))


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(jd(row) + "\n")
            n += 1
    return n


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def valid_box(box: Optional[Sequence[float]]) -> bool:
    if not box or len(box) != 4:
        return False
    x1, y1, x2, y2 = [float(x) for x in box]
    return x2 > x1 and y2 > y1


def iou_xyxy(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> float:
    if not valid_box(a) or not valid_box(b):
        return 0.0
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def norm_cxcywh_to_xyxy(box: Sequence[float], iw: float, ih: float) -> List[float]:
    cx, cy, w, h = [float(x) for x in box]
    x1 = (cx - w / 2.0) * iw
    y1 = (cy - h / 2.0) * ih
    x2 = (cx + w / 2.0) * iw
    y2 = (cy + h / 2.0) * ih
    return [
        max(0.0, min(float(iw), x1)),
        max(0.0, min(float(ih), y1)),
        max(0.0, min(float(iw), x2)),
        max(0.0, min(float(ih), y2)),
    ]


def build_data(args: argparse.Namespace) -> Dict:
    ensure_dirs()
    if not args.force and all((DATA / f"{split}.jsonl").exists() for split in SPLITS):
        return {split: len(read_jsonl(DATA / f"{split}.jsonl")) for split in SPLITS}

    info: Dict[str, int] = {}
    for split in SPLITS:
        rows = read_jsonl(BASE_DATA / f"{split}.jsonl")
        kept = [
            r
            for r in rows
            if str(r.get("task_family")) == "finding"
            and str(r.get("finding")) in set(FINDING_TARGETS)
            and valid_box(r.get("gold_bbox_xyxy"))
        ]
        write_jsonl(DATA / f"{split}.jsonl", kept)
        info[split] = len(kept)
        pd.DataFrame(kept).to_csv(DATA / f"{split}.csv.gz", index=False, compression="gzip")

    train_subj = {r["subject_id"] for r in read_jsonl(DATA / "train.jsonl")}
    eval_subj = {r["subject_id"] for r in read_jsonl(DATA / "eval.jsonl")}
    summary = {
        "experiment": EXP_NAME,
        "base_experiment": BASE_EXP_NAME,
        "targets": FINDING_TARGETS,
        "rows": info,
        "selected_by_label": {
            split: dict(Counter(r["finding"] for r in read_jsonl(DATA / f"{split}.jsonl"))) for split in SPLITS
        },
        "train_eval_subject_overlap": len(train_subj & eval_subj),
        "bbox_meaning": "Chest ImaGenome weak region/reference bbox; not pixel-level lesion mask.",
    }
    write_text(DATA / "dataset_summary.json", json.dumps(summary, indent=2, ensure_ascii=False))
    write_text(
        REPORT / "STAGE1_FINDING_REGION_DATASET_REPORT.md",
        "# Finding-region Dataset Report\n\n"
        + json.dumps(summary, indent=2, ensure_ascii=False)
        + "\n",
    )
    return info


def load_rows(split: str) -> List[Dict]:
    return read_jsonl(DATA / f"{split}.jsonl")


def load_contexts(split: str) -> List[Dict]:
    return read_jsonl(BASE_DATA / f"claim_context_{split}.jsonl")


def build_context_maps() -> Dict[str, List[str]]:
    rows = load_rows("train")
    ctx_by_id = {r["task_id"]: r for r in load_contexts("train")}
    maps: Dict[str, List[str]] = {}
    for field in CONTEXT_FIELDS:
        vals = sorted({str(ctx_by_id.get(r["task_id"], {}).get(field, "unknown")) for r in rows})
        if "unknown" not in vals:
            vals.append("unknown")
        maps[field] = vals
    return maps


def encode_label_only(rows: List[Dict]) -> np.ndarray:
    label_to_i = {label: i for i, label in enumerate(FINDING_TARGETS)}
    feats = np.zeros((len(rows), len(FINDING_TARGETS)), dtype="float32")
    for i, row in enumerate(rows):
        label = str(row.get("finding"))
        if label in label_to_i:
            feats[i, label_to_i[label]] = 1.0
    return feats


def encode_rule_context(rows: List[Dict], split: str, maps: Dict[str, List[str]]) -> np.ndarray:
    ctx_by_id = {r["task_id"]: r for r in load_contexts(split)}
    feats: List[List[float]] = []
    for row in rows:
        ctx = ctx_by_id.get(row["task_id"], {})
        vals: List[float] = []
        for field in CONTEXT_FIELDS:
            raw = str(ctx.get(field, "unknown"))
            allowed = maps[field]
            value = raw if raw in allowed else "unknown"
            vals.extend(1.0 if value == v else 0.0 for v in allowed)
        feats.append(vals)
    return np.asarray(feats, dtype="float32")


def load_selected_patch_tokens(split: str, rows: List[Dict]) -> np.ndarray:
    """Load base feature file and keep only the finding-region rows.

    The base feature cache is large, so this function is intentionally called
    once per split and returns only the selected task rows.
    """
    data = np.load(BASE_FEAT / f"vfm_features_{split}.npz", allow_pickle=True)
    ids = np.array([str(x) for x in data["task_ids"]], dtype=object)
    pos = {tid: i for i, tid in enumerate(ids)}
    idx = np.asarray([pos[r["task_id"]] for r in rows if r["task_id"] in pos], dtype=np.int64)
    tokens = data["patch_tokens"][idx].astype("float16", copy=False)
    data.close()
    return tokens


def aligned(split: str, mode: str, maps: Dict[str, List[str]]) -> Tuple[List[Dict], np.ndarray, np.ndarray, np.ndarray]:
    rows_all = load_rows(split)
    data = np.load(BASE_FEAT / f"vfm_features_{split}.npz", allow_pickle=True)
    ids = np.array([str(x) for x in data["task_ids"]], dtype=object)
    pos = {tid: i for i, tid in enumerate(ids)}
    rows = [r for r in rows_all if r["task_id"] in pos]
    idx = np.asarray([pos[r["task_id"]] for r in rows], dtype=np.int64)
    tokens = data["patch_tokens"][idx].astype("float16", copy=False)
    data.close()
    if mode == "label_only":
        query = encode_label_only(rows)
    elif mode == "rule_context":
        query = encode_rule_context(rows, split, maps)
    else:
        raise ValueError(f"unknown mode: {mode}")
    y = np.asarray([r["gold_bbox_norm_cxcywh"] for r in rows], dtype="float32")
    return rows, tokens, query, y


def val_score(pred_norm: np.ndarray, rows: List[Dict]) -> float:
    vals = []
    for pred, row in zip(pred_norm, rows):
        box = norm_cxcywh_to_xyxy(pred, row["image_width"], row["image_height"])
        vals.append(iou_xyxy(box, row["gold_bbox_xyxy"]))
    return float(np.mean(vals)) if vals else 0.0


def pixel_pred_rows(rows: List[Dict], pred_norm: np.ndarray, method: str, mode: str) -> List[Dict]:
    out = []
    for row, pred in zip(rows, pred_norm):
        pred = np.asarray(pred, dtype="float32")
        pred[:2] = np.clip(pred[:2], 0, 1)
        pred[2:] = np.clip(pred[2:], 0.02, 1)
        out.append(
            {
                "task_id": row["task_id"],
                "method": method,
                "query_mode": mode,
                "dicom_id": row["dicom_id"],
                "subject_id": row["subject_id"],
                "finding": row["finding"],
                "claim_sentence": row.get("claim_sentence", ""),
                "bbox_name_reference": row.get("bbox_name_reference", ""),
                "pred_bbox_norm_cxcywh": [float(x) for x in pred],
                "pred_bbox_xyxy": norm_cxcywh_to_xyxy(pred, row["image_width"], row["image_height"]),
                "bbox_missing": False,
                "gold_source": row.get("gold_source", "chest_imagenome_silver_scene_graph"),
                "source_note": "Prediction for Chest ImaGenome weak region/reference bbox, not lesion mask.",
            }
        )
    return out


def train_mode(mode: str, args: argparse.Namespace, maps: Dict[str, List[str]]) -> Dict:
    import torch

    method = f"finding_region_{mode}_heatmap"
    pred_path = PRED / f"{method}.jsonl"
    if pred_path.exists() and not args.force and not args.train:
        return {"method": method, "skipped": True}

    train_rows, tok_tr, q_tr, y_tr = aligned("train", mode, maps)
    val_rows, tok_val, q_val, _ = aligned("val", mode, maps)
    eval_rows, tok_eval, q_eval, _ = aligned("eval", mode, maps)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PatchHeatmapBBoxHead(tok_tr.shape[-1], q_tr.shape[1], hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    grid = int(round(math.sqrt(tok_tr.shape[1])))

    def center_idx_np(y: np.ndarray) -> np.ndarray:
        cx = np.clip((y[:, 0] * grid).astype(int), 0, grid - 1)
        cy = np.clip((y[:, 1] * grid).astype(int), 0, grid - 1)
        return np.minimum(cy * grid + cx, tok_tr.shape[1] - 1).astype("int64")

    centers = center_idx_np(y_tr)
    rng = np.random.default_rng(SEED + (0 if mode == "label_only" else 17))
    best_state, best_val, best_epoch, wait = None, -1.0, 0, 0
    hist = []
    batch_size = max(1, args.batch_size)

    def predict(tokens: np.ndarray, query: np.ndarray) -> np.ndarray:
        model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(tokens), batch_size):
                tok = torch.tensor(tokens[start : start + batch_size], dtype=torch.float32, device=device)
                q = torch.tensor(query[start : start + batch_size], dtype=torch.float32, device=device)
                pred, _ = model(tok, q)
                preds.append(pred.detach().cpu().numpy())
        return np.vstack(preds) if preds else np.zeros((0, 4), dtype="float32")

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(len(tok_tr))
        losses = []
        for start in range(0, len(tok_tr), batch_size):
            idx = perm[start : start + batch_size]
            tok = torch.tensor(tok_tr[idx], dtype=torch.float32, device=device)
            q = torch.tensor(q_tr[idx], dtype=torch.float32, device=device)
            target = torch.tensor(y_tr[idx], dtype=torch.float32, device=device)
            center = torch.tensor(centers[idx], dtype=torch.long, device=device)
            pred, logits = model(tok, q)
            b_loss, _ = bbox_loss(pred, target)
            h_loss = torch.nn.functional.cross_entropy(logits, center)
            loss = b_loss + args.heatmap_loss_weight * h_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        pred_val = predict(tok_val, q_val)
        score = val_score(pred_val, val_rows)
        hist.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_mean_iou": score})
        print(f"{method} epoch={epoch} train_loss={float(np.mean(losses)):.5f} val_mean_iou={score:.5f}", flush=True)
        if score > best_val:
            best_state, best_val, best_epoch, wait = {k: v.detach().cpu() for k, v in model.state_dict().items()}, score, epoch, 0
        else:
            wait += 1
        if wait >= args.patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    pred_eval = predict(tok_eval, q_eval)
    preds = pixel_pred_rows(eval_rows, pred_eval, method, mode)
    write_jsonl(pred_path, preds)
    run_dir = RUNS / method
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(hist).to_csv(run_dir / "loss_curve.csv", index=False)
    torch.save(
        {
            "state": best_state,
            "method": method,
            "token_dim": int(tok_tr.shape[-1]),
            "query_dim": int(q_tr.shape[1]),
            "best_val_mean_iou": float(best_val),
        },
        CKPT / f"{method}.pt",
    )
    return {
        "method": method,
        "mode": mode,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "eval_rows": len(eval_rows),
        "query_dim": int(q_tr.shape[1]),
        "best_val_mean_iou": float(best_val),
        "best_epoch": int(best_epoch),
        "train_sec": round(time.time() - t0, 2),
    }


def evaluate_predictions(preds: List[Dict], rows: List[Dict], method: str, subset: str) -> Dict:
    by_id = {p["task_id"]: p for p in preds}
    vals, missing, invalid = [], 0, 0
    for row in rows:
        pred = by_id.get(row["task_id"])
        box = pred.get("pred_bbox_xyxy") if pred else None
        if not box:
            missing += 1
            vals.append(0.0)
        else:
            invalid += 0 if valid_box(box) else 1
            vals.append(iou_xyxy(box, row["gold_bbox_xyxy"]))
    arr = np.asarray(vals, dtype="float32")
    return {
        "method": method,
        "subset": subset,
        "n": len(rows),
        "mean_iou": float(arr.mean()) if len(arr) else 0.0,
        "median_iou": float(np.median(arr)) if len(arr) else 0.0,
        "Hit@0.1": float((arr >= 0.1).mean()) if len(arr) else 0.0,
        "Hit@0.3": float((arr >= 0.3).mean()) if len(arr) else 0.0,
        "Hit@0.5": float((arr >= 0.5).mean()) if len(arr) else 0.0,
        "bbox_missing_rate": float(missing / max(1, len(rows))),
        "bbox_invalid_rate": float(invalid / max(1, len(rows))),
    }


def evaluate_all() -> Dict:
    eval_rows = load_rows("eval")
    summary, per_label = [], []
    for pred_path in sorted(PRED.glob("finding_region_*_heatmap.jsonl")):
        method = pred_path.stem
        preds = read_jsonl(pred_path)
        summary.append(evaluate_predictions(preds, eval_rows, method, "finding_region_all"))
        for label in FINDING_TARGETS:
            rows = [r for r in eval_rows if r["finding"] == label]
            out = evaluate_predictions(preds, rows, method, label)
            out["label"] = label
            per_label.append(out)

    summary_df = pd.DataFrame(summary).sort_values("mean_iou", ascending=False)
    per_label_df = pd.DataFrame(per_label).sort_values(["label", "method"])
    summary_df.to_csv(MET / "summary.csv", index=False)
    per_label_df.to_csv(MET / "per_label_metrics.csv", index=False)

    comp = {}
    if set(summary_df["method"]) >= {"finding_region_label_only_heatmap", "finding_region_rule_context_heatmap"}:
        lo = summary_df[summary_df["method"].eq("finding_region_label_only_heatmap")].iloc[0]
        ru = summary_df[summary_df["method"].eq("finding_region_rule_context_heatmap")].iloc[0]
        comp = {
            "label_only_mean_iou": float(lo["mean_iou"]),
            "label_only_Hit@0.3": float(lo["Hit@0.3"]),
            "rule_context_mean_iou": float(ru["mean_iou"]),
            "rule_context_Hit@0.3": float(ru["Hit@0.3"]),
            "rule_minus_label_only_iou": float(ru["mean_iou"] - lo["mean_iou"]),
            "rule_minus_label_only_Hit@0.3": float(ru["Hit@0.3"] - lo["Hit@0.3"]),
        }
    pd.DataFrame([comp]).to_csv(MET / "label_only_vs_rule_context.csv", index=False)

    report = [
        "# Chest ImaGenome Finding-region Rule Ablation Result",
        "",
        "## Meaning",
        "",
        "This compares the same frozen RAD-DINO patch heatmap bbox head on the same Chest ImaGenome finding-region eval set.",
        "",
        "- label_only: target finding name only.",
        "- rule_context: target finding name plus rule-parsed claim context.",
        "- target boxes: Chest ImaGenome weak region/reference boxes, not pixel-level lesion masks.",
        "",
        "## Summary",
        "",
        summary_df.to_markdown(index=False),
        "",
        "## Label-only vs rule context",
        "",
        pd.DataFrame([comp]).to_markdown(index=False),
        "",
        "## Per finding",
        "",
        per_label_df.to_markdown(index=False),
    ]
    write_text(REPORT / "RESULT_SUMMARY.md", "\n".join(report) + "\n")
    return {"summary": summary_df.to_dict("records"), "comparison": comp}


def write_logical_review(runs: List[Dict]) -> None:
    train_subj = {r["subject_id"] for r in load_rows("train")}
    eval_subj = {r["subject_id"] for r in load_rows("eval")}
    text = [
        "# Pipeline Logical Review",
        "",
        "- Base split reused from `chest_imagenome_10k_vfm_localizer_v1`.",
        f"- train/eval subject overlap: {len(train_subj & eval_subj)}",
        "- Both methods use the same eval rows, same RAD-DINO feature cache, same heatmap head, same optimizer, same epoch limit, and same early stopping rule.",
        "- label_only query contains the target finding name only.",
        "- rule_context query contains finding plus rule-parsed claim context from claim text only.",
        "- `bbox_name_reference` and target coordinates are not encoded as model inputs.",
        "- Chest ImaGenome boxes are weak region/reference boxes, not gold lesion masks.",
        "",
        "## Training runs",
        "",
        pd.DataFrame(runs).to_markdown(index=False) if runs else "No training run metadata.",
    ]
    write_text(REPORT / "PIPELINE_LOGICAL_REVIEW.md", "\n".join(text) + "\n")


def run(args: argparse.Namespace) -> Dict:
    ensure_dirs()
    data_info = build_data(args)
    maps = build_context_maps()
    runs = []
    if args.train or not all((PRED / f"finding_region_{mode}_heatmap.jsonl").exists() for mode in ["label_only", "rule_context"]):
        for mode in ["label_only", "rule_context"]:
            runs.append(train_mode(mode, args, maps))
        pd.DataFrame(runs).to_csv(MET / "training_runs.csv", index=False)
    elif (MET / "training_runs.csv").exists():
        runs = pd.read_csv(MET / "training_runs.csv").to_dict("records")
    result = evaluate_all()
    write_logical_review(runs)
    return {"data": data_info, "result": result, "runs": runs}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--heatmap-loss-weight", type=float, default=0.2)
    args = p.parse_args()
    if not args.train and not args.evaluate:
        args.train = True
        args.evaluate = True
    if args.debug:
        args.epochs = min(args.epochs, 3)
        args.patience = min(args.patience, 2)
    return args


def main() -> None:
    t0 = time.time()
    args = parse_args()
    info = run(args)
    summary_path = MET / "summary.csv"
    comp_path = MET / "label_only_vs_rule_context.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    comp = pd.read_csv(comp_path).iloc[0].to_dict() if comp_path.exists() and len(pd.read_csv(comp_path)) else {}
    best = summary.sort_values("mean_iou", ascending=False).iloc[0].to_dict() if len(summary) else {}
    print(f"project_root={PROJECT_ROOT}")
    print(f"experiment={EXP_NAME}")
    print(f"base_experiment={BASE_EXP_NAME}")
    print(f"train_rows={info['data'].get('train', 0)}")
    print(f"val_rows={info['data'].get('val', 0)}")
    print(f"eval_rows={info['data'].get('eval', 0)}")
    print(f"label_only_mean_iou={comp.get('label_only_mean_iou', float('nan')):.6f}")
    print(f"label_only_hit03={comp.get('label_only_Hit@0.3', float('nan')):.6f}")
    print(f"rule_context_mean_iou={comp.get('rule_context_mean_iou', float('nan')):.6f}")
    print(f"rule_context_hit03={comp.get('rule_context_Hit@0.3', float('nan')):.6f}")
    print(f"rule_minus_label_only_iou={comp.get('rule_minus_label_only_iou', float('nan')):.6f}")
    print(f"best_method={best.get('method', '')}")
    print(f"summary_path={REPORT / 'RESULT_SUMMARY.md'}")
    print(f"logical_review_path={REPORT / 'PIPELINE_LOGICAL_REVIEW.md'}")
    print(f"elapsed_sec={time.time() - t0:.1f}")


if __name__ == "__main__":
    main()

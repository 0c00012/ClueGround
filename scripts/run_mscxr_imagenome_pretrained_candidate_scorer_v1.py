#!/usr/bin/env python
"""ImaGenome-pretrained candidate scorer for MS-CXR YOLO-DINO fusion.

This experiment tests whether a candidate scoring head benefits from weak
Chest ImaGenome region/device supervision before fine-tuning on MS-CXR.

Protocol:
  1. Build candidate-level training rows from Chest ImaGenome finding-region
     and device tasks.  The target is candidate IoU to the weak/reference box.
  2. Build candidate-level rows from MS-CXR phrase groups.  The target is max
     candidate IoU to the phrase-grounding boxes.
  3. Train small MLP scorers:
       - MS-CXR only
       - ImaGenome pretrain only
       - ImaGenome pretrain, then MS-CXR fine-tune
  4. Select threshold/diversity/max-k/fallback on MS-CXR validation groups.
  5. Evaluate once on MS-CXR eval phrase groups.

MS-CXR boxes are phrase-grounding boxes.  Chest ImaGenome boxes are weak
region/reference boxes, not lesion masks and not exact device contours.
"""

from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_imagenome_merged_yolo_rad_dino_fusion_v1 as img  # noqa: E402
from scripts import run_ms_cxr_learned_candidate_scorer_v1 as ms_slow  # noqa: E402
from scripts import run_ms_cxr_multibox_phrase_grounding_v1 as mb  # noqa: E402
from scripts import run_ms_cxr_multibox_rule_context_fusion_v4_yolov8l_pool as mv4  # noqa: E402
from scripts import run_ms_cxr_semantic_ensemble_gated_hybrid_v2_yolov8m_pool as sem2  # noqa: E402
from scripts import run_ms_cxr_yolo_dino_rule_fusion_v1 as yd  # noqa: E402
from scripts import run_ms_cxr_yolo_rule_context_v1 as ms_base  # noqa: E402
from scripts import run_ms_cxr_yolo_rule_context_v2 as yv2  # noqa: E402


EXP_NAME = "mscxr_imagenome_pretrained_candidate_scorer_v1"
EXP = PROJECT_ROOT / "experiments" / EXP_NAME
PRED = EXP / "predictions"
MET = EXP / "metrics"
CFG = EXP / "configs"
REPORT = PROJECT_ROOT / "reports" / EXP_NAME
CKPT = PROJECT_ROOT / "training" / EXP_NAME / "checkpoints"

IMAGENOME_EXP = PROJECT_ROOT / "experiments" / "imagenome_merged_yolo_rad_dino_fusion_v1"
MS_FINE = PROJECT_ROOT / "experiments" / "ms_cxr_semantic_weight_finegrid_v2_yolov8l_pool"
MS_FALLBACK = PROJECT_ROOT / "experiments" / "ms_cxr_validation_selected_method_ensemble_v1" / "predictions"
MS_YOLO_PARAMS = PROJECT_ROOT / "experiments" / "ms_cxr_yolo_rule_context_v2" / "configs" / "best_params_by_finding.json"

IMG_TASKS = ["finding_region", "device"]
IMG_TAGS = ["yolov8n", "yolov8s"]
MS_TAGS = ["yolov8n", "yolov8s", "yolov8m", "yolov8l"]
DOMAINS = ["ms_cxr", "imagenome_finding_region", "imagenome_device"]
LATS = ["right", "left", "bilateral", "none", "unknown"]
VERTS = ["apical", "upper", "mid", "lower", "basal", "whole", "unknown"]


def ensure_dirs() -> None:
    for p in [EXP, PRED, MET, CFG, REPORT, CKPT]:
        p.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def one_hot(value: str, vocab: list[str]) -> list[float]:
    return [1.0 if value == v else 0.0 for v in vocab]


def valid_box(box: list[float]) -> bool:
    return len(box) == 4 and float(box[2]) > float(box[0]) and float(box[3]) > float(box[1])


def xyxy_to_norm(box: list[float], iw: float, ih: float) -> np.ndarray:
    x1, y1, x2, y2 = [float(x) for x in box]
    return np.asarray([
        ((x1 + x2) / 2.0) / iw,
        ((y1 + y2) / 2.0) / ih,
        max((x2 - x1) / iw, 1e-6),
        max((y2 - y1) / ih, 1e-6),
    ], dtype=np.float32)


def norm_to_xyxy(v: np.ndarray, iw: float, ih: float) -> list[float]:
    cx, cy, w, h = [float(x) for x in v]
    return [
        max(0.0, (cx - w / 2.0) * iw),
        max(0.0, (cy - h / 2.0) * ih),
        min(iw, (cx + w / 2.0) * iw),
        min(ih, (cy + h / 2.0) * ih),
    ]


def iou_norm(a: np.ndarray, b: np.ndarray) -> float:
    return mb.iou_xyxy(norm_to_xyxy(a, 1.0, 1.0), norm_to_xyxy(b, 1.0, 1.0))


def parse_text_context(text: str) -> tuple[str, str]:
    lower = str(text).lower()
    if "bilateral" in lower or "both" in lower or "bibasilar" in lower or "bibasal" in lower:
        lat = "bilateral"
    elif "right" in lower:
        lat = "right"
    elif "left" in lower:
        lat = "left"
    elif "no " in lower or "without" in lower:
        lat = "none"
    else:
        lat = "unknown"

    if "apical" in lower or "apex" in lower:
        vert = "apical"
    elif "upper" in lower or "superior" in lower:
        vert = "upper"
    elif "middle" in lower or "mid" in lower:
        vert = "mid"
    elif "lower" in lower or "inferior" in lower:
        vert = "lower"
    elif "basilar" in lower or "bibasilar" in lower or "basal" in lower or "base" in lower:
        vert = "basal"
    elif "diffuse" in lower or "widespread" in lower:
        vert = "whole"
    else:
        vert = "unknown"
    return lat, vert


def label_vocab() -> list[str]:
    labels: set[str] = set()
    for split in ["train", "val", "eval"]:
        for g in sem2.gh.load_groups(split).values():
            labels.add(str(g["finding"]))
    for task in IMG_TASKS:
        for split in ["train", "val", "eval"]:
            for r in img.task_rows(task, split):
                labels.add(str(r[img.TASKS[task].label_col]))
    return sorted(labels)


def source_vocab() -> list[str]:
    return sorted(set(IMG_TAGS + MS_TAGS))


def class_priors(rows: list[dict[str, Any]], label_col: str) -> dict[str, np.ndarray]:
    buckets: dict[str, list[np.ndarray]] = {}
    for r in rows:
        label = str(r[label_col])
        b = xyxy_to_norm([float(x) for x in r["gold_bbox_xyxy"]], float(r["image_width"]), float(r["image_height"]))
        buckets.setdefault(label, []).append(b)
    return {k: np.median(np.stack(v), axis=0).astype(np.float32) for k, v in buckets.items() if v}


def parse_candidate_csv(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for _, r in df.iterrows():
        grouped.setdefault(str(r["dicom_id"]), []).append({
            "class_id": int(r["class_id"]),
            "score": float(r["score"]),
            "box": [float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])],
            "source_model": str(r.get("source_model", "")),
            "rank": int(r.get("rank", 0)),
        })
    return grouped


def merge_candidate_groups(paths: Iterable[Path]) -> dict[str, list[dict[str, Any]]]:
    merged: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        if not path.exists():
            continue
        for dicom, rows in parse_candidate_csv(path).items():
            merged.setdefault(dicom, []).extend(rows)
    return merged


@dataclass
class FeatureSpec:
    labels: list[str]
    sources: list[str]


def feature_vector(
    *,
    domain: str,
    label: str,
    text: str,
    cand: dict[str, Any],
    image_width: float,
    image_height: float,
    class_id: int,
    n_classes: int,
    prior: np.ndarray,
    spec: FeatureSpec,
) -> list[float]:
    box_norm = xyxy_to_norm(cand["box"], image_width, image_height)
    cx, cy, bw, bh = [float(x) for x in box_norm]
    area = max(1e-6, bw * bh)
    prior_area = max(1e-6, float(prior[2] * prior[3]))
    lat, vert = parse_text_context(text)
    conf = max(0.0, float(cand["score"]))
    rank = float(cand.get("rank", 99))
    prior_i = iou_norm(box_norm, prior)
    feat = [
        math.log1p(20.0 * conf),
        conf,
        1.0 / (1.0 + rank),
        min(rank, 50.0) / 50.0,
        cx,
        cy,
        bw,
        bh,
        area,
        bw / max(bh, 1e-6),
        abs(cx - float(prior[0])),
        abs(cy - float(prior[1])),
        abs(math.log(max(bw, 1e-6) / max(float(prior[2]), 1e-6))),
        abs(math.log(max(bh, 1e-6) / max(float(prior[3]), 1e-6))),
        math.log(area / prior_area),
        prior_i,
        float(class_id) / max(float(n_classes - 1), 1.0),
    ]
    feat += one_hot(domain, DOMAINS)
    feat += one_hot(label, spec.labels)
    feat += one_hot(str(cand.get("source_model", "")), spec.sources)
    feat += one_hot(lat, LATS)
    feat += one_hot(vert, VERTS)
    return feat


def imagenome_label_maps(task: str) -> tuple[list[str], dict[str, int]]:
    names = img.class_names(task)
    return names, {name: i for i, name in enumerate(names)}


def build_imagenome_table(split: str, spec: FeatureSpec, max_rank: int = 40) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    X: list[list[float]] = []
    y: list[float] = []
    meta: list[dict[str, Any]] = []
    for task in IMG_TASKS:
        rows = img.task_rows(task, split)
        labels, label_to_id = imagenome_label_maps(task)
        priors = class_priors(img.task_rows(task, "train"), img.TASKS[task].label_col)
        paths = [
            IMAGENOME_EXP / "predictions" / f"{task}_{tag}_{split}_conf0p001_candidates.csv"
            for tag in IMG_TAGS
        ]
        candidates = merge_candidate_groups(paths)
        domain = f"imagenome_{task}"
        for r in rows:
            label = str(r[img.TASKS[task].label_col])
            cls = label_to_id[label]
            prior = priors.get(label, np.asarray(r["gold_bbox_norm_cxcywh"], dtype=np.float32))
            gt = [float(x) for x in r["gold_bbox_xyxy"]]
            for cand in candidates.get(str(r["dicom_id"]), []):
                if int(cand["class_id"]) != cls or int(cand.get("rank", 999)) >= max_rank:
                    continue
                target = mb.iou_xyxy(cand["box"], gt)
                X.append(feature_vector(
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
                    "split": split,
                    "task_id": r["task_id"],
                    "dicom_id": r["dicom_id"],
                    "finding": label,
                    "candidate_source": cand.get("source_model", ""),
                    "candidate_rank": cand.get("rank", -1),
                    "target_iou": target,
                })
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), pd.DataFrame(meta)


def ms_group_priors() -> dict[str, np.ndarray]:
    rows = mv4.load_rows("train")
    buckets: dict[str, list[np.ndarray]] = {}
    for r in rows:
        label = str(r["finding"])
        b = xyxy_to_norm([float(x) for x in r["gold_bbox_xyxy"]], float(r["image_width"]), float(r["image_height"]))
        buckets.setdefault(label, []).append(b)
    return {k: np.median(np.stack(v), axis=0).astype(np.float32) for k, v in buckets.items() if v}


def build_ms_table(split: str, spec: FeatureSpec, include_target: bool, max_rank: int = 50) -> tuple[np.ndarray, np.ndarray | None, pd.DataFrame]:
    groups = sem2.gh.load_groups(split)
    paths = [
        yv2.V1_PRED / f"{tag}_{split}_conf0p001_all_candidates.csv"
        for tag in MS_TAGS
    ]
    candidates = merge_candidate_groups(paths)
    priors = ms_group_priors()
    X: list[list[float]] = []
    y: list[float] = []
    meta: list[dict[str, Any]] = []
    for gid, g in groups.items():
        label = str(g["finding"])
        cls = int(g["class_id"])
        prior = priors.get(label, np.asarray([0.5, 0.5, 0.5, 0.5], dtype=np.float32))
        cues = mv4.context_cues(str(g["claim_sentence"]), label, {"finding": label, "laterality": "unknown", "vertical": "unknown"})
        for cand in candidates.get(str(g["dicom_id"]), []):
            if int(cand["class_id"]) != cls or int(cand.get("rank", 999)) >= max_rank:
                continue
            target = max([mb.iou_xyxy(cand["box"], gt) for gt in g["gt_boxes"]] or [0.0])
            X.append(feature_vector(
                domain="ms_cxr",
                label=label,
                text=str(g["claim_sentence"]),
                cand=cand,
                image_width=float(g["image_width"]),
                image_height=float(g["image_height"]),
                class_id=cls,
                n_classes=8,
                prior=prior,
                spec=spec,
            ))
            if include_target:
                y.append(float(target))
            meta.append({
                "domain": "ms_cxr",
                "split": split,
                "group_id": gid,
                "dicom_id": g["dicom_id"],
                "finding": label,
                "claim_sentence": g["claim_sentence"],
                "candidate_source": cand.get("source_model", ""),
                "candidate_rank": cand.get("rank", -1),
                "box_json": json.dumps([float(x) for x in cand["box"]]),
                "has_multi_cue": bool(cues.get("has_multi_cue", False)),
                "k_hint": int(cues.get("k_hint", 1)),
                "target_iou": target if include_target else np.nan,
            })
    return np.asarray(X, dtype=np.float32), (np.asarray(y, dtype=np.float32) if include_target else None), pd.DataFrame(meta)


class CandidateMLP(nn.Module):
    def __init__(self, dim: int, hidden: int = 192, dropout: float = 0.12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_mlp(
    model: CandidateMLP,
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    seed: int,
) -> list[dict[str, float]]:
    set_seed(seed)
    model.to(device)
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=gen, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss(reduction="none")
    logs = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            weights = 1.0 + 4.0 * yb
            loss = (loss_fn(pred, yb) * weights).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * len(xb)
            n += len(xb)
        logs.append({"epoch": epoch, "loss": total / max(n, 1)})
    return logs


def predict_scores(model: CandidateMLP, X: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    model.to(device)
    out = []
    with torch.no_grad():
        for start in range(0, len(X), 8192):
            xb = torch.from_numpy(X[start : start + 8192]).float().to(device)
            out.append(model(xb).detach().cpu().numpy())
    return np.concatenate(out) if out else np.asarray([], dtype=np.float32)


def parse_box(s: str) -> list[float]:
    return [float(x) for x in json.loads(s)]


def fallback_preds(split: str) -> dict[str, list[dict[str, Any]]]:
    path = MS_FALLBACK / f"candidate_method_{split}_details.csv"
    df = pd.read_csv(path)
    df = df[df["method"] == "semantic_fine_v2"]
    out = {}
    for _, r in df.iterrows():
        boxes = json.loads(r["pred_boxes_json"])
        out[str(r["group_id"])] = [{"box": [float(x) for x in b], "score": 0.0, "source": "semantic_fine_fallback"} for b in boxes]
    return out


def select_predictions(
    table: pd.DataFrame,
    score_col: str,
    threshold: float,
    diversity_iou: float,
    max_k: int,
    fallback: bool,
    fallback_by_gid: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], pd.DataFrame]:
    preds: dict[str, list[dict[str, Any]]] = {}
    audit = []
    for gid, sub in table.groupby("group_id"):
        first = sub.iloc[0]
        k = min(int(first["k_hint"]), max_k)
        if not bool(first["has_multi_cue"]):
            k = 1
        selected = []
        for _, r in sub.sort_values(score_col, ascending=False).iterrows():
            score = float(r[score_col])
            if score < threshold:
                continue
            box = parse_box(str(r["box_json"]))
            if any(mb.iou_xyxy(box, s["box"]) > diversity_iou for s in selected):
                continue
            selected.append({"box": box, "score": score, "source": f"imagenome_pretrained_scorer:{r['candidate_source']}"})
            if len(selected) >= k:
                break
        used_fallback = False
        if not selected and fallback:
            selected = fallback_by_gid.get(str(gid), [])
            used_fallback = True
        preds[str(gid)] = selected
        audit.append({
            "group_id": gid,
            "finding": first["finding"],
            "has_multi_cue": bool(first["has_multi_cue"]),
            "k": k,
            "n_candidates": len(sub),
            "n_selected": len(selected),
            "used_fallback": used_fallback,
            "top_score": float(sub[score_col].max()),
        })
    return preds, pd.DataFrame(audit)


def eval_method(method: str, split: str, preds: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    groups = sem2.gh.load_groups(split)
    return pd.DataFrame(mb.eval_method(method, groups, preds))


def summary(method: str, detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subset, sub in {
        "eval_phrase_groups_all": detail,
        "eval_phrase_groups_single_box": detail[detail["n_gt"] == 1],
        "eval_phrase_groups_multi_box": detail[detail["n_gt"] > 1],
    }.items():
        rows.append(mb.summarize(sub.to_dict("records"), method, subset))
    return pd.DataFrame(rows)


def tune_on_val(method: str, val_table: pd.DataFrame, fallback_val: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = []
    best_key = None
    best = None
    for threshold in [-0.05, 0.0, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2]:
        for diversity_iou in [0.35, 0.5, 0.65, 0.8]:
            for max_k in [1, 2, 3, 4]:
                for fallback in [True, False]:
                    preds, audit = select_predictions(val_table, f"score_{method}", threshold, diversity_iou, max_k, fallback, fallback_val)
                    d = eval_method("val_tune", "val", preds)
                    s = mb.summarize(d.to_dict("records"), "val_tune", "val_all")
                    row = {
                        "method": method,
                        "threshold": threshold,
                        "diversity_iou": diversity_iou,
                        "max_k": max_k,
                        "fallback": fallback,
                        "fallback_rate": float(audit["used_fallback"].mean()),
                        "mean_selected": float(audit["n_selected"].mean()),
                        **s,
                    }
                    rows.append(row)
                    key = (float(s["coverage_mean_iou"]), float(s["gt_hit_rate_0_5"]), float(s["set_f1_0_3"]))
                    if best_key is None or key > best_key:
                        best_key = key
                        best = row
    assert best is not None
    return best, pd.DataFrame(rows).sort_values(["coverage_mean_iou", "gt_hit_rate_0_5", "set_f1_0_3"], ascending=False)


def train_variants(X_img: np.ndarray, y_img: np.ndarray, X_ms: np.ndarray, y_ms: np.ndarray, dim: int, device: str) -> dict[str, tuple[CandidateMLP, list[dict[str, float]]]]:
    variants: dict[str, tuple[CandidateMLP, list[dict[str, float]]]] = {}

    ms_only = CandidateMLP(dim)
    logs = train_mlp(ms_only, X_ms, y_ms, epochs=12, lr=2e-3, batch_size=2048, device=device, seed=42)
    variants["ms_only_mlp"] = (ms_only, logs)

    pre = CandidateMLP(dim)
    logs_pre = train_mlp(pre, X_img, y_img, epochs=8, lr=2e-3, batch_size=4096, device=device, seed=42)
    variants["imagenome_pretrain_only"] = (pre, logs_pre)

    ft = CandidateMLP(dim)
    ft.load_state_dict(pre.state_dict())
    logs_ft = logs_pre + [{"epoch": -1, "loss": -1.0}]
    logs_ft += train_mlp(ft, X_ms, y_ms, epochs=8, lr=8e-4, batch_size=2048, device=device, seed=43)
    variants["imagenome_pretrain_then_mscxr_finetune"] = (ft, logs_ft)

    pooled = CandidateMLP(dim)
    X_pool = np.concatenate([X_img, X_ms], axis=0)
    y_pool = np.concatenate([y_img, y_ms], axis=0)
    logs_pool = train_mlp(pooled, X_pool, y_pool, epochs=10, lr=1.5e-3, batch_size=4096, device=device, seed=44)
    variants["imagenome_mscxr_pooled_mlp"] = (pooled, logs_pool)

    return variants


def bootstrap(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    piv = pd.concat([a, b], ignore_index=True).pivot(index="group_id", columns="method", values="coverage_mean_iou").dropna()
    methods = list(piv.columns)
    if len(methods) != 2:
        return pd.DataFrame()
    diff = (piv[methods[0]] - piv[methods[1]]).to_numpy()
    rng = np.random.default_rng(20260706)
    boots = [float(diff[rng.integers(0, len(diff), len(diff))].mean()) for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return pd.DataFrame([{
        "method_a": methods[0],
        "method_b": methods[1],
        "n_groups": int(len(diff)),
        "mean_diff": float(diff.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "p_diff_le_0": float((np.asarray(boots) <= 0).mean()),
    }])


def main() -> None:
    ensure_dirs()
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = FeatureSpec(labels=label_vocab(), sources=source_vocab())
    (CFG / "feature_spec.json").write_text(json.dumps({"labels": spec.labels, "sources": spec.sources, "domains": DOMAINS}, ensure_ascii=False, indent=2), encoding="utf-8")

    X_img, y_img, img_meta = build_imagenome_table("train", spec)
    X_ms, y_ms, ms_meta_train = build_ms_table("train", spec, include_target=True)
    X_val, _, ms_meta_val = build_ms_table("val", spec, include_target=False)
    X_eval, _, ms_meta_eval = build_ms_table("eval", spec, include_target=False)

    img_meta.to_csv(PRED / "imagenome_pretrain_candidate_rows.csv", index=False)
    ms_meta_train.to_csv(PRED / "mscxr_train_candidate_rows.csv", index=False)
    ms_meta_val.to_csv(PRED / "mscxr_val_candidate_rows.csv", index=False)
    ms_meta_eval.to_csv(PRED / "mscxr_eval_candidate_rows.csv", index=False)

    dim = X_ms.shape[1]
    variants = train_variants(X_img, y_img, X_ms, y_ms, dim, device)
    fallback_val = fallback_preds("val")
    fallback_eval = fallback_preds("eval")

    all_eval_details = []
    all_summaries = []
    all_search = []
    configs = []
    for name, (model, logs) in variants.items():
        pd.DataFrame(logs).to_csv(MET / f"{name}_train_log.csv", index=False)
        torch.save(model.state_dict(), CKPT / f"{name}.pt")
        ms_meta_val[f"score_{name}"] = predict_scores(model, X_val, device)
        ms_meta_eval[f"score_{name}"] = predict_scores(model, X_eval, device)
        best, search = tune_on_val(name, ms_meta_val, fallback_val)
        all_search.append(search)
        preds, audit = select_predictions(
            ms_meta_eval,
            f"score_{name}",
            float(best["threshold"]),
            float(best["diversity_iou"]),
            int(best["max_k"]),
            bool(best["fallback"]),
            fallback_eval,
        )
        method = name
        detail = eval_method(method, "eval", preds)
        summ = summary(method, detail)
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
            "val_coverage_mean_iou": float(best["coverage_mean_iou"]),
            "val_hit05": float(best["gt_hit_rate_0_5"]),
            "device": device,
        })

    pd.concat(all_search, ignore_index=True, sort=False).to_csv(MET / "val_search_all_methods.csv", index=False)
    summary_df = pd.concat(all_summaries, ignore_index=True, sort=False)
    ref = pd.read_csv(MS_FINE / "metrics" / "semantic_weight_finegrid_summary.csv")
    combined = pd.concat([summary_df, ref], ignore_index=True, sort=False)
    combined.to_csv(MET / "summary_with_reference.csv", index=False)
    pd.DataFrame(configs).to_csv(CFG / "selected_params.csv", index=False)

    best_method = combined[combined["subset"].eq("eval_phrase_groups_all")].sort_values("coverage_mean_iou", ascending=False).iloc[0]["method"]
    ref_detail = pd.read_csv(MS_FINE / "predictions" / "semantic_weight_finegrid_phrase_group_predictions.csv")
    boot_rows = []
    for detail in all_eval_details:
        boot = bootstrap(detail, ref_detail)
        if not boot.empty:
            boot_rows.append(boot)
    boot_df = pd.concat(boot_rows, ignore_index=True, sort=False) if boot_rows else pd.DataFrame()
    boot_df.to_csv(MET / "bootstrap_vs_semantic_finegrid.csv", index=False)

    all_rows = combined[combined["subset"].eq("eval_phrase_groups_all")][
        ["method", "n_groups", "coverage_mean_iou", "gt_hit_rate_0_3", "gt_hit_rate_0_5", "set_f1_0_3", "set_f1_0_5"]
    ].sort_values("coverage_mean_iou", ascending=False)
    lines = [
        "# MS-CXR ImaGenome-pretrained Candidate Scorer V1",
        "",
        "## 한 줄 결론",
        "",
        "Chest ImaGenome finding-region/device 후보 IoU supervision으로 candidate scorer를 먼저 학습한 뒤 MS-CXR로 fine-tune했다.",
        "",
        "## Data",
        "",
        f"- ImaGenome candidate rows: `{len(X_img)}` from finding-region/device train splits",
        f"- MS-CXR train candidate rows: `{len(X_ms)}`",
        f"- MS-CXR val candidate rows: `{len(X_val)}`",
        f"- MS-CXR eval candidate rows: `{len(X_eval)}`",
        "- Chest ImaGenome boxes are weak/reference region boxes, not lesion masks.",
        "- MS-CXR boxes are phrase-grounding boxes.",
        "",
        "## Eval summary",
        "",
        all_rows.to_markdown(index=False),
        "",
        "## Selected params",
        "",
        pd.DataFrame(configs).to_markdown(index=False),
        "",
        "## Bootstrap vs current semantic finegrid",
        "",
        boot_df.to_markdown(index=False) if not boot_df.empty else "not available",
        "",
        "## 공정성",
        "",
        "- scorer architecture and MS-CXR val selection protocol are shared across variants.",
        "- eval gold is not used for scorer training, threshold selection, or model selection.",
        "- ImaGenome pretraining uses weak region/device reference boxes only and is reported as weak pretraining.",
    ]
    lines = [
        "# MS-CXR ImaGenome-pretrained Candidate Scorer V1",
        "",
        "## Result",
        "",
        "This experiment pretrains a candidate scorer with Chest ImaGenome finding-region/device weak box supervision, then fine-tunes or pools it with MS-CXR candidate supervision.",
        "",
        "## Data",
        "",
        f"- ImaGenome candidate rows: `{len(X_img)}` from finding-region/device train splits",
        f"- MS-CXR train candidate rows: `{len(X_ms)}`",
        f"- MS-CXR val candidate rows: `{len(X_val)}`",
        f"- MS-CXR eval candidate rows: `{len(X_eval)}`",
        "- Chest ImaGenome boxes are weak/reference region boxes, not lesion masks.",
        "- MS-CXR boxes are phrase-grounding boxes.",
        "",
        "## Eval Summary",
        "",
        all_rows.to_markdown(index=False),
        "",
        "## Selected Params",
        "",
        pd.DataFrame(configs).to_markdown(index=False),
        "",
        "## Bootstrap vs Current Semantic Finegrid",
        "",
        boot_df.to_markdown(index=False) if not boot_df.empty else "not available",
        "",
        "## Fairness Notes",
        "",
        "- The scorer architecture and MS-CXR validation-selection protocol are shared across variants.",
        "- Eval gold is not used for scorer training, threshold selection, or model selection.",
        "- ImaGenome pretraining uses weak region/device reference boxes only and is reported as weak pretraining.",
        "- Anatomy boxes are intentionally excluded from the pretraining source because the anatomy-name task is too direct and did not show a meaningful rule-context gain.",
    ]
    (REPORT / "README_KO.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"project_root={PROJECT_ROOT}")
    print(f"imagenome_candidate_rows={len(X_img)}")
    print(f"mscxr_train_candidate_rows={len(X_ms)}")
    print(f"mscxr_eval_candidate_rows={len(X_eval)}")
    print(f"best_method={best_method}")
    print(f"summary_path={MET / 'summary_with_reference.csv'}")
    print(f"report_path={REPORT / 'README_KO.md'}")
    print(all_rows.to_string(index=False))


if __name__ == "__main__":
    main()

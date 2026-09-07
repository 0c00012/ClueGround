#!/usr/bin/env python
"""Trainable MoE gate and contrastive candidate scorer for MS-CXR grounding.

This experiment is a direct follow-up to semantic finegrid.  The frozen
experts stay fixed:

  - YOLO-DINO hybrid set prediction
  - SigLIP crop-text candidate expert
  - BioMedCLIP crop-text candidate expert
  - ImaGenome-pretrained candidate scorer expert

The new part is a small trainable gate that learns sample-specific expert
weights from MS-CXR train only.  It is evaluated on the held-out p10-p19 eval
phrase groups.  A separate candidate-query scorer is also trained with hard
negative context flips and pairwise ranking loss.

MS-CXR boxes are phrase-grounding boxes, not lesion masks.  Chest ImaGenome
boxes used by the fixed external expert are weak/reference boxes.
"""

from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_ms_cxr_multibox_phrase_grounding_v1 as mb  # noqa: E402
from scripts import run_ms_cxr_semantic_ensemble_gated_hybrid_v2_yolov8m_pool as sem2  # noqa: E402
from scripts import run_ms_cxr_semantic_finegrid_with_imagenome_pretrain_expert_v1 as imgexp  # noqa: E402
from scripts import run_ms_cxr_semantic_weight_grid_v3_yolov8l_pool as fine_base  # noqa: E402
from scripts import run_ms_cxr_siglip_candidate_fusion_v1 as sf  # noqa: E402
from scripts import run_mscxr_imagenome_pretrained_candidate_scorer_v1 as cand_v1  # noqa: E402


EXP_NAME = "ms_cxr_trainable_moe_gate_v1"
EXP = PROJECT_ROOT / "experiments" / EXP_NAME
PRED = EXP / "predictions"
MET = EXP / "metrics"
CFG = EXP / "configs"
LOG = EXP / "logs"
REPORT = PROJECT_ROOT / "reports" / EXP_NAME
CKPT = PROJECT_ROOT / "training" / EXP_NAME / "checkpoints"

SIGLIP_EXP = PROJECT_ROOT / "experiments" / "ms_cxr_siglip_candidate_fusion_v1"
BIOMED_EXP = PROJECT_ROOT / "experiments" / "ms_cxr_biomedclip_gated_hybrid_v1"
BASE_FINE = PROJECT_ROOT / "experiments" / "ms_cxr_semantic_weight_finegrid_v2_yolov8l_pool"
IMG_FINE = PROJECT_ROOT / "experiments" / "ms_cxr_semantic_finegrid_with_imagenome_pretrain_expert_v1"

LATS = ["right", "left", "bilateral", "none", "unknown"]
VERTS = ["apical", "upper", "mid", "lower", "basal", "whole", "unknown"]


def ensure_dirs() -> None:
    for p in [EXP, PRED, MET, CFG, LOG, REPORT, CKPT]:
        p.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def one_hot(value: str, vocab: list[str]) -> list[float]:
    return [1.0 if value == v else 0.0 for v in vocab]


def xyxy_to_norm(box: list[float], iw: float, ih: float) -> np.ndarray:
    x1, y1, x2, y2 = [float(x) for x in box]
    return np.asarray(
        [
            ((x1 + x2) / 2.0) / max(iw, 1.0),
            ((y1 + y2) / 2.0) / max(ih, 1.0),
            max((x2 - x1) / max(iw, 1.0), 1e-5),
            max((y2 - y1) / max(ih, 1.0), 1e-5),
        ],
        dtype=np.float32,
    )


def norm_to_xyxy(v: np.ndarray, iw: float, ih: float) -> list[float]:
    cx, cy, w, h = [float(x) for x in v]
    return [
        max(0.0, (cx - w / 2.0) * iw),
        max(0.0, (cy - h / 2.0) * ih),
        min(iw, (cx + w / 2.0) * iw),
        min(ih, (cy + h / 2.0) * ih),
    ]


def torch_cxcywh_to_xyxy(v: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = v.unbind(-1)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return torch.stack([x1, y1, x2, y2], dim=-1)


def torch_iou_cxcywh(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax = torch_cxcywh_to_xyxy(a)
    bx = torch_cxcywh_to_xyxy(b)
    ix1 = torch.maximum(ax[:, 0], bx[:, 0])
    iy1 = torch.maximum(ax[:, 1], bx[:, 1])
    ix2 = torch.minimum(ax[:, 2], bx[:, 2])
    iy2 = torch.minimum(ax[:, 3], bx[:, 3])
    inter = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)
    area_a = torch.clamp(ax[:, 2] - ax[:, 0], min=0) * torch.clamp(ax[:, 3] - ax[:, 1], min=0)
    area_b = torch.clamp(bx[:, 2] - bx[:, 0], min=0) * torch.clamp(bx[:, 3] - bx[:, 1], min=0)
    return inter / torch.clamp(area_a + area_b - inter, min=1e-6)


def parse_context(group: dict[str, Any]) -> tuple[str, str, bool]:
    text = str(group.get("claim_sentence", "")).lower()
    if any(x in text for x in ["bilateral", "both", "bibasilar", "bibasal"]):
        lat = "bilateral"
    elif "right" in text:
        lat = "right"
    elif "left" in text:
        lat = "left"
    elif any(x in text for x in ["without", " no "]):
        lat = "none"
    else:
        lat = "unknown"

    if any(x in text for x in ["apical", "apex"]):
        vert = "apical"
    elif any(x in text for x in ["upper", "superior"]):
        vert = "upper"
    elif any(x in text for x in ["middle", " mid "]):
        vert = "mid"
    elif any(x in text for x in ["lower", "inferior"]):
        vert = "lower"
    elif any(x in text for x in ["basilar", "bibasilar", "basal", "base"]):
        vert = "basal"
    elif any(x in text for x in ["diffuse", "widespread", "bilateral"]):
        vert = "whole"
    else:
        vert = "unknown"
    has_multi = any(x in text for x in ["bilateral", "both", "bibasilar", "multifocal", "multilobar"])
    return lat, vert, has_multi


def semantic_set_from_scored(split: str, groups: dict[str, dict[str, Any]], kind: str) -> dict[str, list[dict[str, Any]]]:
    if kind == "siglip":
        scored_path = SIGLIP_EXP / "predictions" / f"{split}_siglip_fusion_scored_candidates.csv"
        raw_path = SIGLIP_EXP / "predictions" / f"{split}_siglip_scored_candidates_cxr_claim_m0p15.csv"
        params_path = SIGLIP_EXP / "configs" / "best_set_params.json"
        score_preference = ["siglip_fusion_score", "score_head", "siglip_sigmoid", "siglip_rank", "confidence"]
    elif kind == "biomed":
        scored_path = BIOMED_EXP / "predictions" / f"{split}_biomedclip_fusion_scored_candidates.csv"
        raw_path = BIOMED_EXP / "predictions" / f"{split}_biomedclip_scored_candidates_cxr_claim_m0p15.csv"
        params_path = BIOMED_EXP / "configs" / "best_candidate_set_params.json"
        score_preference = ["biomedclip_fusion_score", "score_head", "biomedclip_sigmoid", "biomedclip_rank", "confidence"]
    else:
        raise ValueError(kind)

    path = scored_path if scored_path.exists() else raw_path
    if not path.exists():
        raise FileNotFoundError(path)
    scored = pd.read_csv(path)
    score_col = next((c for c in score_preference if c in scored.columns), None)
    if score_col is None:
        raise RuntimeError(f"No usable score column in {path}")
    params = json.loads(params_path.read_text(encoding="utf-8"))
    cands = sf.scored_candidates_by_group(scored, groups, score_col)
    out = sf.predict_phrase_sets(groups, cands, params)
    return out


def load_pretrain_expert(split: str, method: str, device: str) -> dict[str, list[dict[str, Any]]]:
    spec = cand_v1.FeatureSpec(labels=cand_v1.label_vocab(), sources=cand_v1.source_vocab())
    X, _y, meta = cand_v1.build_ms_table(split, spec, include_target=False)
    dim = int(X.shape[1])
    model = imgexp.load_scorer_model(method, dim, device)
    score_col = f"score_{method}"
    meta = meta.copy()
    meta[score_col] = cand_v1.predict_scores(model, X, device)
    params = imgexp.selected_params_for(method)
    preds, audit = cand_v1.select_predictions(
        meta,
        score_col,
        params["threshold"],
        params["diversity_iou"],
        params["max_k"],
        False,
        {},
    )
    meta.to_csv(PRED / f"{method}_{split}_candidate_scores.csv", index=False)
    audit.to_csv(PRED / f"{method}_{split}_selection_audit.csv", index=False)
    return preds


@dataclass
class ExpertBundle:
    groups: dict[str, dict[str, Any]]
    hybrid: dict[str, list[dict[str, Any]]]
    siglip: dict[str, list[dict[str, Any]]]
    biomed: dict[str, list[dict[str, Any]]]
    pretrain: dict[str, list[dict[str, Any]]]
    cue: pd.DataFrame


def build_bundle(
    split: str,
    device: str,
    pretrain_method: str,
    *,
    include_pretrain_expert: bool = True,
) -> ExpertBundle:
    groups = sem2.gh.load_groups(split)
    hybrid, cue = fine_base.build_hybrid_v4(split, groups)
    siglip = semantic_set_from_scored(split, groups, "siglip")
    biomed = semantic_set_from_scored(split, groups, "biomed")
    pretrain = load_pretrain_expert(split, pretrain_method, device) if include_pretrain_expert else {}
    return ExpertBundle(groups, hybrid, siglip, biomed, pretrain, cue)


def has_multi_cue(cue: pd.DataFrame) -> dict[str, bool]:
    return {str(r["group_id"]): bool(r.get("has_multi_cue", False)) for _, r in cue.iterrows()}


def expert_list(bundle: ExpertBundle, experts: list[str], gid: str) -> list[list[dict[str, Any]]]:
    maps = {
        "hybrid": bundle.hybrid,
        "siglip": bundle.siglip,
        "biomed": bundle.biomed,
        "imagenome": bundle.pretrain,
    }
    return [maps[e].get(gid, []) for e in experts]


def gate_feature_for_group(bundle: ExpertBundle, gid: str, experts: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    g = bundle.groups[gid]
    ex = expert_list(bundle, experts, gid)
    if not all(len(x) == 1 for x in ex):
        return None
    iw = float(g["image_width"])
    ih = float(g["image_height"])
    boxes = np.stack([xyxy_to_norm(x[0]["box"], iw, ih) for x in ex]).astype(np.float32)
    scores = np.asarray([float(x[0].get("score", 0.0)) for x in ex], dtype=np.float32)
    # Compress arbitrary scorer ranges.
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    scores_scaled = np.asarray([math.tanh(float(s)) for s in scores], dtype=np.float32)
    pairwise = []
    for i in range(len(experts)):
        for j in range(i + 1, len(experts)):
            pairwise.append(mb.iou_xyxy(ex[i][0]["box"], ex[j][0]["box"]))
    lat, vert, multi_text = parse_context(g)
    # The dataset finding/category label is not part of a phrase-grounding
    # inference request.  Context features must be derived from the raw phrase.
    context = []
    context += one_hot(lat, LATS)
    context += one_hot(vert, VERTS)
    # Query text may imply multiple regions, but annotated GT cardinality is a
    # training/evaluation label and must never be an inference feature.
    context += [float(multi_text)]
    stats = [
        float(np.mean(boxes[:, 0])),
        float(np.std(boxes[:, 0])),
        float(np.mean(boxes[:, 1])),
        float(np.std(boxes[:, 1])),
        float(np.mean(boxes[:, 2] * boxes[:, 3])),
        float(np.std(boxes[:, 2] * boxes[:, 3])),
        float(max(pairwise) if pairwise else 0.0),
        float(np.mean(pairwise) if pairwise else 0.0),
    ]
    feat = np.asarray(list(boxes.reshape(-1)) + list(scores_scaled) + pairwise + context + stats, dtype=np.float32)
    return feat, boxes, scores_scaled


def flip_context_features(x: np.ndarray, n_experts: int) -> np.ndarray:
    """Flip laterality/vertical one-hot bits in the fixed feature layout."""

    y = x.copy()
    # Layout: boxes 4E, scores E, pairwise E*(E-1)/2, lat 5,
    # vert 7, text flag 1, stats 8.
    start = 4 * n_experts + n_experts + (n_experts * (n_experts - 1)) // 2
    lat_start = start
    vert_start = lat_start + len(LATS)
    lat = y[lat_start : lat_start + len(LATS)].copy()
    vert = y[vert_start : vert_start + len(VERTS)].copy()
    if lat.sum() > 0:
        idx = int(lat.argmax())
        if LATS[idx] == "right":
            y[lat_start + LATS.index("right")] = 0.0
            y[lat_start + LATS.index("left")] = 1.0
        elif LATS[idx] == "left":
            y[lat_start + LATS.index("left")] = 0.0
            y[lat_start + LATS.index("right")] = 1.0
        elif LATS[idx] == "bilateral":
            y[lat_start + LATS.index("bilateral")] = 0.0
            y[lat_start + LATS.index("right")] = 1.0
    if vert.sum() > 0:
        idx = int(vert.argmax())
        pairs = {"upper": "lower", "lower": "upper", "apical": "basal", "basal": "apical"}
        src = VERTS[idx]
        dst = pairs.get(src)
        if dst:
            y[vert_start + VERTS.index(src)] = 0.0
            y[vert_start + VERTS.index(dst)] = 1.0
    return y


def build_gate_table(bundle: ExpertBundle, experts: list[str], include_multi: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    X: list[np.ndarray] = []
    B: list[np.ndarray] = []
    Y: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    multi = has_multi_cue(bundle.cue)
    for gid, g in bundle.groups.items():
        if multi.get(gid, False):
            continue
        if not include_multi and len(g["gt_boxes"]) != 1:
            continue
        got = gate_feature_for_group(bundle, gid, experts)
        if got is None:
            continue
        feat, boxes, _scores = got
        iw = float(g["image_width"])
        ih = float(g["image_height"])
        target_box = g["gt_boxes"][0] if len(g["gt_boxes"]) == 1 else mb.union_box(g["gt_boxes"])
        target = xyxy_to_norm(target_box, iw, ih)
        ex_maps = expert_list(bundle, experts, gid)
        expert_ious = np.asarray([mb.iou_xyxy(x[0]["box"], target_box) for x in ex_maps], dtype=np.float32)
        X.append(feat)
        B.append(boxes)
        Y.append(np.concatenate([target, expert_ious], axis=0))
        meta.append({
            "group_id": gid,
            "finding": g["finding"],
            "n_gt": len(g["gt_boxes"]),
            "best_expert": experts[int(expert_ious.argmax())],
            "best_expert_iou": float(expert_ious.max()),
            **{f"iou_{e}": float(v) for e, v in zip(experts, expert_ious)},
        })
    return np.stack(X), np.stack(B), np.stack(Y), pd.DataFrame(meta)


def build_gate_input_audit(bundle: ExpertBundle, experts: list[str]) -> pd.DataFrame:
    """Audit inference eligibility without reading GT boxes or cardinality."""

    multi = has_multi_cue(bundle.cue)
    rows = []
    for gid in bundle.groups:
        rows.append(
            {
                "group_id": gid,
                "eligible": gate_feature_for_group(bundle, gid, experts) is not None,
                "has_multi_cue_from_raw_phrase": bool(multi.get(gid, False)),
            }
        )
    return pd.DataFrame(rows)


class MoEGate(nn.Module):
    def __init__(self, in_dim: int, n_experts: int, hidden: int = 128, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_gate(
    name: str,
    train_bundle: ExpertBundle,
    val_bundle: ExpertBundle,
    experts: list[str],
    *,
    hardneg: bool,
    seed: int,
    device: str,
) -> tuple[MoEGate, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    set_seed(seed)
    X, B, Y, meta = build_gate_table(train_bundle, experts)
    Xv, Bv, Yv, metav = build_gate_table(val_bundle, experts)
    if len(X) == 0 or len(Xv) == 0:
        raise RuntimeError(f"No train/val rows for {name}")
    X_neg = np.stack([flip_context_features(x, len(experts)) for x in X]).astype(np.float32)
    ds_tensors = [
        torch.from_numpy(X).float(),
        torch.from_numpy(B).float(),
        torch.from_numpy(Y).float(),
        torch.from_numpy(X_neg).float(),
    ]
    loader = DataLoader(TensorDataset(*ds_tensors), batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
    model = MoEGate(X.shape[1], len(experts)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    logs: list[dict[str, Any]] = []
    best_key: tuple[float, float] | None = None
    best_state = None
    best_params: dict[str, Any] = {}
    x_val_t = torch.from_numpy(Xv).float().to(device)
    b_val_t = torch.from_numpy(Bv).float().to(device)
    y_val_t = torch.from_numpy(Yv).float().to(device)
    for epoch in range(1, 121):
        model.train()
        total = 0.0
        n = 0
        for xb, bb, yy, xneg in loader:
            xb = xb.to(device)
            bb = bb.to(device)
            yy = yy.to(device)
            xneg = xneg.to(device)
            target_box = yy[:, :4]
            expert_iou = yy[:, 4:]
            logits = model(xb)
            weights = torch.softmax(logits, dim=-1)
            pred_box = torch.sum(weights.unsqueeze(-1) * bb, dim=1).clamp(0.0, 1.0)
            soft_target = torch.softmax(expert_iou / 0.08, dim=-1)
            box_l1 = torch.nn.functional.smooth_l1_loss(pred_box, target_box)
            box_iou = torch_iou_cxcywh(pred_box, target_box)
            iou_loss = (1.0 - box_iou).mean()
            ce = torch.nn.functional.kl_div(torch.log_softmax(logits, dim=-1), soft_target, reduction="batchmean")
            rank_terms = []
            for i in range(len(experts)):
                for j in range(len(experts)):
                    if i == j:
                        continue
                    diff = expert_iou[:, i] - expert_iou[:, j]
                    mask = diff > 0.05
                    if mask.any():
                        rank_terms.append((torch.relu(0.12 - (logits[:, i] - logits[:, j])) * diff.clamp(min=0)).masked_select(mask).mean())
            rank_loss = torch.stack(rank_terms).mean() if rank_terms else torch.tensor(0.0, device=device)
            neg_loss = torch.tensor(0.0, device=device)
            if hardneg:
                neg_logits = model(xneg)
                neg_target = torch.softmax((1.0 - expert_iou) / 0.12, dim=-1)
                neg_loss = torch.nn.functional.kl_div(torch.log_softmax(neg_logits, dim=-1), neg_target, reduction="batchmean")
            loss = box_l1 + iou_loss + 0.45 * ce + 0.25 * rank_loss + (0.15 * neg_loss if hardneg else 0.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach().cpu()) * len(xb)
            n += len(xb)
        model.eval()
        with torch.no_grad():
            logits = model(x_val_t)
            weights = torch.softmax(logits, dim=-1)
            pred_box = torch.sum(weights.unsqueeze(-1) * b_val_t, dim=1).clamp(0.0, 1.0)
            val_iou = torch_iou_cxcywh(pred_box, y_val_t[:, :4]).detach().cpu().numpy()
            val_mean = float(val_iou.mean())
            val_hit05 = float((val_iou >= 0.5).mean())
        logs.append({"epoch": epoch, "loss": total / max(n, 1), "val_eligible_mean_iou": val_mean, "val_eligible_hit05": val_hit05})
        key = (val_mean, val_hit05)
        if best_key is None or key > best_key:
            best_key = key
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_params = {
                "name": name,
                "experts": experts,
                "hardneg": hardneg,
                "seed": seed,
                "epoch": epoch,
                "val_eligible_mean_iou": val_mean,
                "val_eligible_hit05": val_hit05,
                "train_rows": int(len(X)),
                "val_rows": int(len(Xv)),
            }
    assert best_state is not None
    model.load_state_dict(best_state)
    torch.save({"state": best_state, "params": best_params, "in_dim": int(X.shape[1]), "n_experts": len(experts)}, CKPT / f"{name}.pt")
    pd.DataFrame(logs).to_csv(LOG / f"{name}_train_log.csv", index=False)
    meta.to_csv(MET / f"{name}_train_gate_rows.csv", index=False)
    metav.to_csv(MET / f"{name}_val_gate_rows.csv", index=False)
    return model, best_params, pd.DataFrame(logs), meta


@torch.no_grad()
def predict_gate(
    method: str,
    model: MoEGate,
    bundle: ExpertBundle,
    experts: list[str],
    *,
    device: str,
    keep_multi: bool = True,
) -> tuple[dict[str, list[dict[str, Any]]], pd.DataFrame]:
    model.eval().to(device)
    multi = has_multi_cue(bundle.cue)
    out: dict[str, list[dict[str, Any]]] = {}
    audit: list[dict[str, Any]] = []
    for gid, g in bundle.groups.items():
        h = bundle.hybrid.get(gid, [])
        pred = h
        action = "keep_hybrid"
        weights_out: list[float] | None = None
        eligible = gate_feature_for_group(bundle, gid, experts) is not None
        if eligible and not (keep_multi and multi.get(gid, False)):
            feat, boxes_norm, _scores = gate_feature_for_group(bundle, gid, experts)  # type: ignore[misc]
            xb = torch.from_numpy(feat[None, :]).float().to(device)
            logits = model(xb)
            weights = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
            box_norm = (boxes_norm * weights[:, None]).sum(axis=0)
            box_norm[:2] = np.clip(box_norm[:2], 0.0, 1.0)
            box_norm[2:] = np.clip(box_norm[2:], 1e-4, 1.0)
            box = norm_to_xyxy(box_norm, float(g["image_width"]), float(g["image_height"]))
            pred = [{"box": box, "score": float(weights.max()), "source": method}]
            action = "moe_gate_blend"
            weights_out = [float(x) for x in weights]
        out[gid] = pred
        row = {
            "group_id": gid,
            "eligible": bool(eligible),
            "has_multi_cue": bool(multi.get(gid, False)),
            "action": action,
        }
        if weights_out is not None:
            for e, w in zip(experts, weights_out):
                row[f"w_{e}"] = w
        audit.append(row)
    return out, pd.DataFrame(audit)


def detail_and_summary(method: str, bundle: ExpertBundle, preds: dict[str, list[dict[str, Any]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail = pd.DataFrame(mb.eval_method(method, bundle.groups, preds))
    rows = []
    for subset, sub in {
        "eval_phrase_groups_all": detail,
        "eval_phrase_groups_single_box": detail[detail["n_gt"] == 1],
        "eval_phrase_groups_multi_box": detail[detail["n_gt"] > 1],
    }.items():
        rows.append(mb.summarize(sub.to_dict("records"), method, subset))
    return detail, pd.DataFrame(rows)


class ContrastiveCandidateMLP(nn.Module):
    def __init__(self, dim: int, hidden: int = 256, dropout: float = 0.12):
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
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def augment_hard_negative_features(X: np.ndarray, y: np.ndarray, spec: cand_v1.FeatureSpec) -> tuple[np.ndarray, np.ndarray]:
    # Candidate feature layout comes from cand_v1.feature_vector.
    base_len = 17 + len(cand_v1.DOMAINS) + len(spec.labels) + len(spec.sources)
    lat_start = base_len
    vert_start = lat_start + len(cand_v1.LATS)
    neg_rows = []
    neg_y = []
    for row, target in zip(X, y):
        nr = row.copy()
        lat = nr[lat_start : lat_start + len(cand_v1.LATS)]
        vert = nr[vert_start : vert_start + len(cand_v1.VERTS)]
        if lat.sum() > 0:
            idx = int(lat.argmax())
            if cand_v1.LATS[idx] == "right":
                nr[lat_start + cand_v1.LATS.index("right")] = 0
                nr[lat_start + cand_v1.LATS.index("left")] = 1
            elif cand_v1.LATS[idx] == "left":
                nr[lat_start + cand_v1.LATS.index("left")] = 0
                nr[lat_start + cand_v1.LATS.index("right")] = 1
        if vert.sum() > 0:
            idx = int(vert.argmax())
            pairs = {"upper": "lower", "lower": "upper", "apical": "basal", "basal": "apical"}
            src = cand_v1.VERTS[idx]
            dst = pairs.get(src)
            if dst:
                nr[vert_start + cand_v1.VERTS.index(src)] = 0
                nr[vert_start + cand_v1.VERTS.index(dst)] = 1
        neg_rows.append(nr)
        neg_y.append(float(target) * 0.20)
    return np.asarray(neg_rows, dtype=np.float32), np.asarray(neg_y, dtype=np.float32)


def train_contrastive_candidate_scorer(device: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = cand_v1.FeatureSpec(labels=cand_v1.label_vocab(), sources=cand_v1.source_vocab())
    X_train, y_train, meta_train = cand_v1.build_ms_table("train", spec, include_target=True)
    X_val, y_val, meta_val = cand_v1.build_ms_table("val", spec, include_target=True)
    X_neg, y_neg = augment_hard_negative_features(X_train, y_train, spec)
    X_aug = np.concatenate([X_train, X_neg], axis=0).astype(np.float32)
    y_aug = np.concatenate([y_train, y_neg], axis=0).astype(np.float32)
    set_seed(2026)
    model = ContrastiveCandidateMLP(X_aug.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-4)
    ds = TensorDataset(torch.from_numpy(X_aug).float(), torch.from_numpy(y_aug).float())
    loader = DataLoader(ds, batch_size=2048, shuffle=True, generator=torch.Generator().manual_seed(2026), num_workers=0)
    logs = []
    best_state = None
    best_key = None
    xv = torch.from_numpy(X_val).float().to(device)
    yv = torch.from_numpy(y_val).float().to(device)
    for epoch in range(1, 35):
        model.train()
        total = 0.0
        n = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = torch.sigmoid(model(xb))
            loss_point = (torch.nn.functional.smooth_l1_loss(pred, yb, reduction="none") * (1.0 + 4.0 * yb)).mean()
            # In-batch contrastive/ranking proxy: high-IoU candidates should score
            # above low-IoU candidates.  This is not eval gold leakage; it is train
            # supervision inside the train split.
            hi = yb >= 0.5
            lo = yb <= 0.15
            loss_rank = torch.tensor(0.0, device=device)
            if hi.any() and lo.any():
                pos = pred[hi]
                neg = pred[lo]
                loss_rank = torch.relu(0.25 - (pos[:, None] - neg[None, :])).mean()
            loss = loss_point + 0.35 * loss_rank
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach().cpu()) * len(xb)
            n += len(xb)
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(xv))
            val_loss = torch.nn.functional.smooth_l1_loss(pv, yv).item()
            corr = float(np.corrcoef(pv.detach().cpu().numpy(), y_val)[0, 1]) if len(y_val) > 2 else 0.0
        logs.append({"epoch": epoch, "loss": total / max(n, 1), "val_smooth_l1": val_loss, "val_score_iou_corr": corr})
        key = (-val_loss, corr)
        if best_key is None or key > best_key:
            best_key = key
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    torch.save({"state": best_state, "dim": int(X_aug.shape[1])}, CKPT / "candidate_query_contrastive_scorer.pt")
    pd.DataFrame(logs).to_csv(LOG / "candidate_query_contrastive_train_log.csv", index=False)

    selected_rows = []
    summaries = []
    detail_frames = []
    for split in ["val", "eval"]:
        X, _y, meta = cand_v1.build_ms_table(split, spec, include_target=False)
        with torch.no_grad():
            scores = torch.sigmoid(model(torch.from_numpy(X).float().to(device))).detach().cpu().numpy()
        table = meta.copy()
        table["score_candidate_query_contrastive"] = scores
        fallback = {}
        best = None
        grid_rows = []
        if split == "val":
            thresholds = [-0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
            divs = [0.35, 0.5, 0.65]
            maxks = [1, 2, 3, 4]
        else:
            params = json.loads((CFG / "candidate_query_contrastive_best_params.json").read_text(encoding="utf-8"))
            thresholds = [params["threshold"]]
            divs = [params["diversity_iou"]]
            maxks = [params["max_k"]]
        for thr in thresholds:
            for div in divs:
                for max_k in maxks:
                    preds, audit = cand_v1.select_predictions(table, "score_candidate_query_contrastive", float(thr), float(div), int(max_k), False, fallback)
                    d = cand_v1.eval_method(f"candidate_query_contrastive_{split}", split, preds)
                    s = cand_v1.summary(f"candidate_query_contrastive_{split}", d)
                    allrow = s[s["subset"] == "eval_phrase_groups_all"].iloc[0].to_dict()
                    row = {"split": split, "threshold": thr, "diversity_iou": div, "max_k": max_k, **allrow, "mean_selected": float(audit["n_selected"].mean())}
                    grid_rows.append(row)
                    key = (float(allrow["coverage_mean_iou"]), float(allrow["gt_hit_rate_0_5"]), float(allrow["set_f1_0_3"]))
                    if best is None or key > best[0]:
                        best = (key, row, preds, d, audit)
        assert best is not None
        if split == "val":
            params = {"threshold": float(best[1]["threshold"]), "diversity_iou": float(best[1]["diversity_iou"]), "max_k": int(best[1]["max_k"])}
            (CFG / "candidate_query_contrastive_best_params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
            pd.DataFrame(grid_rows).to_csv(MET / "candidate_query_contrastive_val_grid.csv", index=False)
        else:
            detail = best[3]
            audit = best[4]
            detail["method"] = "candidate_query_contrastive_scorer"
            audit.to_csv(PRED / "candidate_query_contrastive_eval_audit.csv", index=False)
            detail.to_csv(PRED / "candidate_query_contrastive_eval_phrase_group_predictions.csv", index=False)
            detail_frames.append(detail)
            summaries.append(cand_v1.summary("candidate_query_contrastive_scorer", detail))
        selected_rows.append(meta_train.head(0))
    return (pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(), pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame())


def bootstrap(target_details: pd.DataFrame, refs: pd.DataFrame, target_methods: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(20260706)
    rows = []
    all_df = pd.concat([target_details, refs], ignore_index=True, sort=False)
    ref_methods = [
        "semantic_weight_finegrid_v2_yolov8l_pool",
        "semantic_finegrid_plus_mimic_imagenome_pretrain_then_mscxr_finetune",
        "medrpg_rowlevel_full_phrase_s42",
        "medrpg_rowlevel_full_phrase_s13",
        "medrpg_rowlevel_full_phrase_s2026",
    ]
    for a in target_methods:
        for b in ref_methods:
            piv = all_df[all_df["method"].isin([a, b])].pivot_table(index="group_id", columns="method", values="coverage_mean_iou", aggfunc="first").dropna()
            if a not in piv.columns or b not in piv.columns or len(piv) < 5:
                continue
            diff = (piv[a] - piv[b]).to_numpy()
            boots = [float(diff[rng.integers(0, len(diff), len(diff))].mean()) for _ in range(2000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append({
                "method_a": a,
                "method_b": b,
                "n_groups": int(len(diff)),
                "mean_diff": float(diff.mean()),
                "ci95_low": float(lo),
                "ci95_high": float(hi),
                "p_diff_le_0": float((np.asarray(boots) <= 0).mean()),
            })
    return pd.DataFrame(rows)


def load_reference_details() -> pd.DataFrame:
    paths = [
        BASE_FINE / "predictions" / "semantic_weight_finegrid_phrase_group_predictions.csv",
        IMG_FINE / "predictions" / "semantic_finegrid_plus_mimic_imagenome_pretrain_then_mscxr_finetune_phrase_group_predictions.csv",
        PROJECT_ROOT / "experiments" / "ms_cxr_medrpg_rowlevel_fair_retrain_v1" / "predictions" / "medrpg_row_level_full_phrase_s42_eval_phrase_group_predictions.csv",
        PROJECT_ROOT / "experiments" / "ms_cxr_medrpg_rowlevel_fair_retrain_v1" / "predictions" / "medrpg_row_level_full_phrase_s13_eval_phrase_group_predictions.csv",
        PROJECT_ROOT / "experiments" / "ms_cxr_medrpg_rowlevel_fair_retrain_v1" / "predictions" / "medrpg_row_level_full_phrase_s2026_eval_phrase_group_predictions.csv",
    ]
    frames = []
    for p in paths:
        if p.exists():
            frames.append(pd.read_csv(p))
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def main() -> None:
    ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pretrain_method = "mimic_imagenome_pretrain_then_mscxr_finetune"
    config = {
        "device": device,
        "pretrain_method": pretrain_method,
        "note": "Eval gold is not used for tuning. Gate checkpoints use train rows and val selection only.",
    }
    (CFG / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    train_bundle = build_bundle("train", device, pretrain_method)
    val_bundle = build_bundle("val", device, pretrain_method)
    eval_bundle = build_bundle("eval", device, pretrain_method)

    bundle_audit = []
    for split, bundle in [("train", train_bundle), ("val", val_bundle), ("eval", eval_bundle)]:
        for experts in [["hybrid", "siglip", "biomed"], ["hybrid", "siglip", "biomed", "imagenome"]]:
            if split == "eval":
                meta = build_gate_input_audit(bundle, experts)
                eligible_rows = int(meta["eligible"].sum())
            else:
                X, _B, _Y, meta = build_gate_table(bundle, experts)
                eligible_rows = int(len(X))
            bundle_audit.append({"split": split, "experts": "+".join(experts), "eligible_rows": eligible_rows, "groups": int(len(bundle.groups))})
            meta.to_csv(MET / f"{split}_{'_'.join(experts)}_eligible_gate_rows.csv", index=False)
    pd.DataFrame(bundle_audit).to_csv(MET / "expert_bundle_audit.csv", index=False)

    runs = [
        ("trainable_moe_gate_3expert_ms_only", ["hybrid", "siglip", "biomed"], False),
        ("trainable_moe_gate_4expert_imagenome_expert", ["hybrid", "siglip", "biomed", "imagenome"], False),
        ("trainable_moe_gate_4expert_hardneg_rank", ["hybrid", "siglip", "biomed", "imagenome"], True),
    ]
    all_details = []
    all_summaries = []
    best_params = []
    for name, experts, hardneg in runs:
        model, params, _logs, _meta = train_gate(name, train_bundle, val_bundle, experts, hardneg=hardneg, seed=2026, device=device)
        preds, audit = predict_gate(name, model, eval_bundle, experts, device=device, keep_multi=True)
        d, s = detail_and_summary(name, eval_bundle, preds)
        d.to_csv(PRED / f"{name}_eval_phrase_group_predictions.csv", index=False)
        audit.to_csv(PRED / f"{name}_eval_action_audit.csv", index=False)
        s.to_csv(MET / f"{name}_summary.csv", index=False)
        all_details.append(d)
        all_summaries.append(s)
        best_params.append(params)

    cand_summary, cand_detail = train_contrastive_candidate_scorer(device)
    if not cand_summary.empty:
        all_summaries.append(cand_summary)
    if not cand_detail.empty:
        all_details.append(cand_detail)

    ref_summaries = []
    for p in [
        BASE_FINE / "metrics" / "semantic_weight_finegrid_summary.csv",
        IMG_FINE / "metrics" / "summary_with_references.csv",
    ]:
        if p.exists():
            ref_summaries.append(pd.read_csv(p))
    combined = pd.concat(all_summaries + ref_summaries, ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset=["method", "subset"], keep="first")
    combined.to_csv(MET / "summary_with_references.csv", index=False)
    target_details = pd.concat(all_details, ignore_index=True, sort=False)
    target_details.to_csv(PRED / "all_new_method_eval_phrase_group_predictions.csv", index=False)
    refs = load_reference_details()
    boot = bootstrap(
        target_details,
        refs,
        [x[0] for x in runs] + (["candidate_query_contrastive_scorer"] if not cand_detail.empty else []),
    )
    boot.to_csv(MET / "bootstrap_vs_references.csv", index=False)
    (CFG / "best_gate_params.json").write_text(json.dumps(best_params, ensure_ascii=False, indent=2), encoding="utf-8")

    all_rows = combined[combined["subset"] == "eval_phrase_groups_all"][
        ["method", "n_groups", "coverage_mean_iou", "union_iou", "gt_hit_rate_0_3", "gt_hit_rate_0_5", "set_f1_0_3", "set_f1_0_5"]
    ].sort_values("coverage_mean_iou", ascending=False)
    lines = [
        "# Trainable MoE Gate V1",
        "",
        "## 한 줄 결론",
        "",
        "손으로 고른 semantic finegrid를 학습형 MoE gate, hard-negative/ranking loss, candidate-query contrastive scorer로 바꿔 검증했다.",
        "",
        "## 방법",
        "",
        "- fixed experts: YOLO-DINO hybrid, SigLIP, BioMedCLIP, ImaGenome-pretrained candidate scorer.",
        "- trainable MoE gate: MS-CXR train eligible phrase groups에서 expert별 가중치를 학습하고 val eligible IoU로 checkpoint를 선택했다.",
        "- hard negative: right/left, upper/lower, apical/basal context one-hot을 뒤집은 synthetic wrong-query feature를 사용했다.",
        "- candidate-query contrastive scorer: 후보 feature와 rule-context feature를 같이 보고 좋은 후보가 나쁜 후보보다 높은 score를 갖도록 학습했다.",
        "",
        "## Expert bundle audit",
        "",
        pd.DataFrame(bundle_audit).to_markdown(index=False),
        "",
        "## Eval summary",
        "",
        all_rows.to_markdown(index=False),
        "",
        "## Bootstrap",
        "",
        boot.to_markdown(index=False) if not boot.empty else "not available",
        "",
        "## 해석 주의",
        "",
        "- MS-CXR eval gold는 gate 학습, threshold 선택, checkpoint 선택에 사용하지 않았다.",
        "- Chest ImaGenome은 fixed candidate expert의 weak/reference supervision으로만 들어갔다.",
        "- 여기서 'ImaGenome expert'는 gate 자체를 external task에서 직접 pretrain했다는 뜻이 아니라, ImaGenome-pretrained candidate scorer를 MoE expert로 넣었다는 뜻이다.",
    ]
    (REPORT / "README_KO.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"project_root={PROJECT_ROOT}")
    print(f"device={device}")
    print(f"summary_path={MET / 'summary_with_references.csv'}")
    print(f"bootstrap_path={MET / 'bootstrap_vs_references.csv'}")
    print(f"report_path={REPORT / 'README_KO.md'}")
    print(all_rows.to_string(index=False))


if __name__ == "__main__":
    main()

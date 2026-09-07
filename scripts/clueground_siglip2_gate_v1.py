#!/usr/bin/env python
"""Small finding-conditioned two-expert gate for ClueGround SigLIP2 runs.

This file is intentionally independent of the legacy SigLIP/BioMedCLIP MoE
module. It only knows about the experts explicitly passed by the new runners.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts import run_ms_cxr_multibox_phrase_grounding_v1 as mb


FINDINGS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Lung Opacity",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
]
LATERALITIES = ["right", "left", "bilateral", "none", "unknown"]
VERTICALS = ["apical", "upper", "mid", "lower", "basal", "whole", "unknown"]


@dataclass
class ExpertBundle:
    groups: dict[str, dict[str, Any]]
    hybrid: dict[str, list[dict[str, Any]]]
    siglip2: dict[str, list[dict[str, Any]]]
    cue: pd.DataFrame


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def one_hot(value: str, vocabulary: list[str]) -> list[float]:
    return [float(value == item) for item in vocabulary]


def xyxy_to_norm(box: list[float], width: float, height: float) -> np.ndarray:
    x1, y1, x2, y2 = [float(value) for value in box]
    return np.asarray(
        [
            ((x1 + x2) / 2.0) / max(width, 1.0),
            ((y1 + y2) / 2.0) / max(height, 1.0),
            max(1e-4, x2 - x1) / max(width, 1.0),
            max(1e-4, y2 - y1) / max(height, 1.0),
        ],
        dtype=np.float32,
    )


def norm_to_xyxy(value: np.ndarray, width: float, height: float) -> list[float]:
    cx, cy, bw, bh = [float(item) for item in value]
    return [
        max(0.0, (cx - bw / 2.0) * width),
        max(0.0, (cy - bh / 2.0) * height),
        min(width, (cx + bw / 2.0) * width),
        min(height, (cy + bh / 2.0) * height),
    ]


def torch_cxcywh_to_xyxy(value: torch.Tensor) -> torch.Tensor:
    cx, cy, width, height = value.unbind(dim=-1)
    return torch.stack(
        [cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0],
        dim=-1,
    )


def torch_iou_cxcywh(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    a = torch_cxcywh_to_xyxy(left)
    b = torch_cxcywh_to_xyxy(right)
    top_left = torch.maximum(a[..., :2], b[..., :2])
    bottom_right = torch.minimum(a[..., 2:], b[..., 2:])
    intersection = (bottom_right - top_left).clamp(min=0.0)
    intersection_area = intersection[..., 0] * intersection[..., 1]
    area_a = (a[..., 2] - a[..., 0]).clamp(min=0.0) * (
        a[..., 3] - a[..., 1]
    ).clamp(min=0.0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0.0) * (
        b[..., 3] - b[..., 1]
    ).clamp(min=0.0)
    union = area_a + area_b - intersection_area
    return intersection_area / union.clamp(min=1e-7)


def parse_context(group: dict[str, Any]) -> tuple[str, str, bool]:
    text = " " + re.sub(
        r"[^a-z0-9]+", " ", str(group.get("claim_sentence", "")).lower()
    ) + " "
    has_right = any(token in text for token in [" right ", "right-sided", "right sided"])
    has_left = any(token in text for token in [" left ", "left-sided", "left sided"])
    has_bilateral = any(
        token in text
        for token in [" bilateral ", " both ", " bibasilar ", " biapical "]
    )
    if has_bilateral or (has_right and has_left):
        laterality = "bilateral"
    elif has_right:
        laterality = "right"
    elif has_left:
        laterality = "left"
    elif any(token in text for token in [" heart ", " cardiac ", " cardiomegaly "]):
        laterality = "none"
    else:
        laterality = "unknown"

    if any(token in text for token in [" apical ", " apex "]):
        vertical = "apical"
    elif any(token in text for token in [" upper ", " superior "]):
        vertical = "upper"
    elif any(token in text for token in [" middle ", " mid "]):
        vertical = "mid"
    elif any(token in text for token in [" lower ", " inferior "]):
        vertical = "lower"
    elif any(token in text for token in [" basilar ", " bibasilar ", " basal ", " base "]):
        vertical = "basal"
    elif any(token in text for token in [" diffuse ", " widespread ", " bilateral "]):
        vertical = "whole"
    else:
        vertical = "unknown"

    has_multi = any(
        token in text
        for token in [
            " bilateral ",
            " both ",
            " bibasilar ",
            " biapical ",
            " multifocal ",
            " multilobar ",
            " multisegmental ",
        ]
    )
    return laterality, vertical, has_multi


def has_multi_cue(cue: pd.DataFrame) -> dict[str, bool]:
    return {
        str(row["group_id"]): bool(row.get("has_multi_cue", False))
        for _, row in cue.iterrows()
    }


def expert_list(
    bundle: ExpertBundle,
    experts: list[str],
    group_id: str,
) -> list[list[dict[str, Any]]]:
    mappings = {
        "hybrid": bundle.hybrid,
        "siglip2": bundle.siglip2,
    }
    unknown = [name for name in experts if name not in mappings]
    if unknown:
        raise ValueError(f"Unknown experts: {unknown}")
    return [mappings[name].get(group_id, []) for name in experts]


def gate_feature_for_group(
    bundle: ExpertBundle,
    group_id: str,
    experts: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    group = bundle.groups[group_id]
    rows = expert_list(bundle, experts, group_id)
    if not all(len(items) == 1 for items in rows):
        return None
    width = float(group["image_width"])
    height = float(group["image_height"])
    boxes = np.stack(
        [xyxy_to_norm(items[0]["box"], width, height) for items in rows]
    ).astype(np.float32)
    scores = np.asarray(
        [float(items[0].get("score", 0.0)) for items in rows], dtype=np.float32
    )
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    scores_scaled = np.asarray(
        [math.tanh(float(score)) for score in scores], dtype=np.float32
    )
    pairwise: list[float] = []
    for left in range(len(experts)):
        for right in range(left + 1, len(experts)):
            pairwise.append(
                float(mb.iou_xyxy(rows[left][0]["box"], rows[right][0]["box"]))
            )
    laterality, vertical, multi_text = parse_context(group)
    context: list[float] = []
    context += one_hot(str(group["finding"]), FINDINGS)
    context += one_hot(laterality, LATERALITIES)
    context += one_hot(vertical, VERTICALS)
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
    feature = np.asarray(
        list(boxes.reshape(-1))
        + list(scores_scaled)
        + pairwise
        + context
        + stats,
        dtype=np.float32,
    )
    return feature, boxes, scores_scaled


def build_gate_table(
    bundle: ExpertBundle,
    experts: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    features: list[np.ndarray] = []
    boxes: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    multi = has_multi_cue(bundle.cue)
    for group_id, group in bundle.groups.items():
        if multi.get(group_id, False) or len(group["gt_boxes"]) != 1:
            continue
        item = gate_feature_for_group(bundle, group_id, experts)
        if item is None:
            continue
        feature, expert_boxes, _ = item
        target_box = group["gt_boxes"][0]
        target = xyxy_to_norm(
            target_box,
            float(group["image_width"]),
            float(group["image_height"]),
        )
        expert_rows = expert_list(bundle, experts, group_id)
        expert_ious = np.asarray(
            [float(mb.iou_xyxy(rows[0]["box"], target_box)) for rows in expert_rows],
            dtype=np.float32,
        )
        features.append(feature)
        boxes.append(expert_boxes)
        targets.append(np.concatenate([target, expert_ious], axis=0))
        metadata.append(
            {
                "group_id": group_id,
                "finding": group["finding"],
                "best_expert": experts[int(expert_ious.argmax())],
                "best_expert_iou": float(expert_ious.max()),
                **{
                    f"iou_{expert}": float(value)
                    for expert, value in zip(experts, expert_ious)
                },
            }
        )
    if not features:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0, 0, 4), dtype=np.float32),
            np.empty((0, 0), dtype=np.float32),
            pd.DataFrame(metadata),
        )
    return (
        np.stack(features),
        np.stack(boxes),
        np.stack(targets),
        pd.DataFrame(metadata),
    )


class MoEGate(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_experts: int,
        hidden: int = 128,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_experts),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


def train_gate(
    name: str,
    train_bundle: ExpertBundle,
    val_bundle: ExpertBundle,
    experts: list[str],
    *,
    seed: int,
    device: str,
    output_root: Path,
    epochs: int = 120,
) -> tuple[MoEGate, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    set_seed(seed)
    train_x, train_boxes, train_y, train_meta = build_gate_table(
        train_bundle, experts
    )
    val_x, val_boxes, val_y, val_meta = build_gate_table(val_bundle, experts)
    if len(train_x) == 0 or len(val_x) == 0:
        raise RuntimeError(f"No train/validation gate rows for {name}")

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_x).float(),
            torch.from_numpy(train_boxes).float(),
            torch.from_numpy(train_y).float(),
        ),
        batch_size=128,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    model = MoEGate(train_x.shape[1], len(experts)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.5e-3, weight_decay=1e-4
    )
    val_x_tensor = torch.from_numpy(val_x).float().to(device)
    val_boxes_tensor = torch.from_numpy(val_boxes).float().to(device)
    val_y_tensor = torch.from_numpy(val_y).float().to(device)

    logs: list[dict[str, Any]] = []
    best_key: tuple[float, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_params: dict[str, Any] = {}
    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for features, expert_boxes, target in loader:
            features = features.to(device)
            expert_boxes = expert_boxes.to(device)
            target = target.to(device)
            target_box = target[:, :4]
            expert_iou = target[:, 4:]
            logits = model(features)
            weights = torch.softmax(logits, dim=-1)
            predicted_box = torch.sum(
                weights.unsqueeze(-1) * expert_boxes, dim=1
            ).clamp(0.0, 1.0)
            soft_target = torch.softmax(expert_iou / 0.08, dim=-1)
            box_l1 = torch.nn.functional.smooth_l1_loss(
                predicted_box, target_box
            )
            iou_loss = (1.0 - torch_iou_cxcywh(predicted_box, target_box)).mean()
            distribution_loss = torch.nn.functional.kl_div(
                torch.log_softmax(logits, dim=-1),
                soft_target,
                reduction="batchmean",
            )
            rank_terms = []
            for left in range(len(experts)):
                for right in range(len(experts)):
                    if left == right:
                        continue
                    difference = expert_iou[:, left] - expert_iou[:, right]
                    mask = difference > 0.05
                    if mask.any():
                        rank_terms.append(
                            (
                                torch.relu(
                                    0.12 - (logits[:, left] - logits[:, right])
                                )
                                * difference.clamp(min=0)
                            )
                            .masked_select(mask)
                            .mean()
                        )
            rank_loss = (
                torch.stack(rank_terms).mean()
                if rank_terms
                else torch.tensor(0.0, device=device)
            )
            loss = box_l1 + iou_loss + 0.45 * distribution_loss + 0.25 * rank_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(features)
            total_rows += len(features)

        model.eval()
        with torch.no_grad():
            logits = model(val_x_tensor)
            weights = torch.softmax(logits, dim=-1)
            predicted_box = torch.sum(
                weights.unsqueeze(-1) * val_boxes_tensor, dim=1
            ).clamp(0.0, 1.0)
            validation_iou = torch_iou_cxcywh(
                predicted_box, val_y_tensor[:, :4]
            ).detach().cpu().numpy()
            validation_mean = float(validation_iou.mean())
            validation_hit_05 = float((validation_iou >= 0.5).mean())
        logs.append(
            {
                "epoch": epoch,
                "loss": total_loss / max(total_rows, 1),
                "val_eligible_mean_iou": validation_mean,
                "val_eligible_hit05": validation_hit_05,
            }
        )
        key = (validation_mean, validation_hit_05)
        if best_key is None or key > best_key:
            best_key = key
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_params = {
                "name": name,
                "experts": experts,
                "seed": seed,
                "epoch": epoch,
                "val_eligible_mean_iou": validation_mean,
                "val_eligible_hit05": validation_hit_05,
                "train_rows": int(len(train_x)),
                "val_rows": int(len(val_x)),
            }

    if best_state is None:
        raise RuntimeError("Gate training failed to produce a checkpoint")
    model.load_state_dict(best_state)
    checkpoint_root = output_root / "checkpoints"
    log_root = output_root / "logs"
    metric_root = output_root / "training_metrics"
    for path in (checkpoint_root, log_root, metric_root):
        path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state": best_state,
            "params": best_params,
            "in_dim": int(train_x.shape[1]),
            "n_experts": len(experts),
        },
        checkpoint_root / f"{name}.pt",
    )
    pd.DataFrame(logs).to_csv(log_root / f"{name}_train_log.csv", index=False)
    train_meta.to_csv(metric_root / f"{name}_train_gate_rows.csv", index=False)
    val_meta.to_csv(metric_root / f"{name}_val_gate_rows.csv", index=False)
    return model, best_params, pd.DataFrame(logs), train_meta


@torch.inference_mode()
def predict_gate(
    model: MoEGate,
    bundle: ExpertBundle,
    experts: list[str],
    *,
    device: str,
    keep_multi: bool = True,
) -> tuple[dict[str, list[dict[str, Any]]], pd.DataFrame]:
    model.eval().to(device)
    multi = has_multi_cue(bundle.cue)
    output: dict[str, list[dict[str, Any]]] = {}
    audit: list[dict[str, Any]] = []
    for group_id, group in bundle.groups.items():
        prediction = bundle.hybrid.get(group_id, [])
        action = "keep_hybrid"
        feature = gate_feature_for_group(bundle, group_id, experts)
        eligible = feature is not None
        weights_output: list[float] | None = None
        if feature is not None and not (keep_multi and multi.get(group_id, False)):
            values, boxes, _ = feature
            logits = model(torch.from_numpy(values[None, :]).float().to(device))
            weights = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
            box_norm = (boxes * weights[:, None]).sum(axis=0)
            box_norm[:2] = np.clip(box_norm[:2], 0.0, 1.0)
            box_norm[2:] = np.clip(box_norm[2:], 1e-4, 1.0)
            box = norm_to_xyxy(
                box_norm,
                float(group["image_width"]),
                float(group["image_height"]),
            )
            prediction = [
                {
                    "box": box,
                    "score": float(weights.max()),
                    "source": "finding_conditioned_siglip2_gate",
                }
            ]
            action = "siglip2_gate_blend"
            weights_output = [float(value) for value in weights]
        output[group_id] = prediction
        row: dict[str, Any] = {
            "group_id": group_id,
            "eligible": bool(eligible),
            "has_multi_cue": bool(multi.get(group_id, False)),
            "action": action,
        }
        if weights_output is not None:
            for expert, weight in zip(experts, weights_output):
                row[f"w_{expert}"] = weight
        audit.append(row)
    return output, pd.DataFrame(audit)


def prediction_maps_equal(
    left: dict[str, list[dict[str, Any]]],
    right: dict[str, list[dict[str, Any]]],
    tolerance: float = 1e-8,
) -> bool:
    if set(left) != set(right):
        return False
    for group_id in left:
        left_rows = left[group_id]
        right_rows = right[group_id]
        if len(left_rows) != len(right_rows):
            return False
        for left_row, right_row in zip(left_rows, right_rows):
            maximum = np.max(
                np.abs(
                    np.asarray(left_row["box"], dtype=float)
                    - np.asarray(right_row["box"], dtype=float)
                )
            )
            if maximum > tolerance:
                return False
    return True


def save_status(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

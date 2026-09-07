#!/usr/bin/env python
"""Context-only ablation over frozen ClueGround visual candidates.

The candidate pool, RAD-DINO agreement, SigLIP score, candidate feature
schema, scorer, count head, loss, decoder, and evaluator are identical.  Only
the phrase representation is changed: raw hashing, frozen SBERT, frozen Qwen,
or deterministic ClueGround context fields.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import HashingVectorizer
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import clueground_siglip_analysis_common_v1 as common  # noqa: E402
from scripts import run_clueground_exact_hybrid_v4_direct_3seed_v1 as exact  # noqa: E402
from scripts import run_ms_cxr_multibox_rule_context_fusion_v4_yolov8l_pool as hybrid_v4  # noqa: E402
from scripts import run_ms_cxr_singlebox_fair_yolo_dino_fusion_v1 as single_fusion  # noqa: E402


SEEDS = (13, 42, 2026)
VARIANTS = ("raw_phrase_hash", "sbert", "qwen", "deterministic_context")
OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "clueground_ablation_study_v1" / "context"
HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
SBERT_ID = "sentence-transformers/all-MiniLM-L6-v2"
QWEN_ROOT = HF_HOME / "hub" / "models--Qwen--Qwen2.5-0.5B-Instruct" / "snapshots"
SOURCE_MODELS = ("yolov8s", "yolov8m", "yolo11s", "yolo11m")
FEATURE_NAMES = (
    "cx", "cy", "w", "h", "area", "confidence", "within_source_rank",
    "rad_dino_iou", "siglip_rank", *[f"source_{name}" for name in SOURCE_MODELS],
)


@dataclass
class GroupData:
    group_ids: list[str]
    visual: np.ndarray
    mask: np.ndarray
    quality: np.ndarray
    count: np.ndarray
    boxes: list[list[list[float]]]


class ContextCandidateScorer(nn.Module):
    def __init__(self, visual_dim: int, context_dim: int) -> None:
        super().__init__()
        self.visual = nn.Sequential(nn.Linear(visual_dim, 64), nn.GELU(), nn.Dropout(0.15))
        self.context = nn.Sequential(nn.Linear(context_dim, 64), nn.GELU(), nn.Dropout(0.15))
        self.score = nn.Sequential(
            nn.Linear(64 * 3, 96), nn.GELU(), nn.Dropout(0.20), nn.Linear(96, 1)
        )
        self.count = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.20), nn.Linear(64, 4)
        )

    def forward(
        self, visual: torch.Tensor, context: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v = self.visual(visual)
        c = self.context(context)[:, None, :].expand(-1, visual.shape[1], -1)
        scores = self.score(torch.cat([v, c, v * c], dim=-1)).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e4)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(v.dtype)
        pooled = (v * mask[..., None]).sum(dim=1) / denom
        counts = self.count(torch.cat([pooled, c[:, 0, :]], dim=-1))
        return scores, counts


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def stable_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def box_iou(a: list[float], b: list[float]) -> float:
    return float(hybrid_v4.base.iou_xyxy(a, b))


def valid_candidate(candidate: dict[str, Any]) -> bool:
    box = np.asarray(candidate["box"], dtype=np.float64)
    return bool(
        box.shape == (4,) and np.isfinite(box).all()
        and box[0] < box[2] and box[1] < box[3]
    )


def phrase_text(group: dict[str, Any]) -> str:
    return f"finding: {group['finding']}; phrase: {group['claim_sentence']}"


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    weights = attention_mask[..., None].to(last_hidden.dtype)
    return (last_hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


@torch.no_grad()
def transformer_embeddings(texts: list[str], variant: str, device: torch.device) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer

    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    if variant == "sbert":
        source: str | Path = SBERT_ID
        kwargs = {"token": False}
        max_length, batch_size = 128, 64
    elif variant == "qwen":
        snapshots = sorted(path for path in QWEN_ROOT.glob("*") if path.is_dir())
        if not snapshots:
            raise FileNotFoundError(QWEN_ROOT)
        source = snapshots[0]
        # Qwen2.5 declares sliding-window attention, which is not implemented
        # by the installed SDPA path. Eager attention avoids an ambiguous
        # approximation in the frozen embedding ablation.
        kwargs = {"local_files_only": True, "attn_implementation": "eager"}
        max_length, batch_size = 128, 24
    else:
        raise ValueError(variant)
    tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
    model = AutoModel.from_pretrained(source, **kwargs).to(device).eval()
    chunks = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start:start + batch_size], padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        hidden = model(**encoded).last_hidden_state
        pooled = F.normalize(mean_pool(hidden.float(), encoded["attention_mask"]), dim=-1)
        chunks.append(pooled.cpu().numpy().astype(np.float32))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(chunks, axis=0)


def deterministic_embeddings(groups: list[dict[str, Any]]) -> np.ndarray:
    findings = sorted(single_fusion.CLASS_TO_ID, key=single_fusion.CLASS_TO_ID.get)
    lateralities = ("unknown", "left", "right", "bilateral")
    verticals = ("unknown", "apical", "upper", "mid", "lower", "basal", "whole")
    rows = []
    for group in groups:
        row = hybrid_v4.group_row(group)
        query = single_fusion.ybase.parse_rule_context(row)
        cue = hybrid_v4.context_cues(str(group["claim_sentence"]), str(group["finding"]), query)
        vector = np.zeros(len(findings) + len(lateralities) + len(verticals) + 8, dtype=np.float32)
        vector[findings.index(str(group["finding"]))] = 1.0
        offset = len(findings)
        laterality = str(query.get("laterality", "unknown"))
        if laterality not in lateralities:
            laterality = "unknown"
        vector[offset + lateralities.index(laterality)] = 1.0
        offset += len(lateralities)
        vertical = str(query.get("vertical", "unknown"))
        if vertical == "middle":
            vertical = "mid"
        if vertical not in verticals:
            vertical = "unknown"
        vector[offset + verticals.index(vertical)] = 1.0
        offset += len(verticals)
        text = str(group["claim_sentence"]).lower()
        vector[offset:] = np.asarray([
            float(bool(cue["has_multi_cue"])),
            float(cue["k_hint"] == 2),
            float(cue["k_hint"] >= 3),
            float(bool(re.search(r"\b(bilateral|both|bibasal|bibasilar)\b", text))),
            float(bool(re.search(r"\b(multifocal|multiple|scattered|diffuse)\b", text))),
            float(bool(re.search(r"\b(small|tiny|minimal|mild)\b", text))),
            float(bool(re.search(r"\b(large|severe|extensive|marked)\b", text))),
            min(len(text.split()), 32) / 32.0,
        ], dtype=np.float32)
        rows.append(vector)
    return np.stack(rows)


def context_embeddings(
    protocol: str,
    split: str,
    variant: str,
    groups: dict[str, dict[str, Any]],
    device: torch.device,
) -> tuple[list[str], np.ndarray]:
    root = OUTPUT_ROOT / "embeddings" / protocol
    cache_variant = "qwen_eager" if variant == "qwen" else variant
    path = root / f"{split}_{cache_variant}.npz"
    ids = list(groups)
    if path.exists():
        archive = np.load(path, allow_pickle=True)
        saved_ids = [str(value) for value in archive["group_ids"]]
        if saved_ids != ids:
            raise RuntimeError(f"Context cache ID mismatch: {path}")
        return ids, archive["embeddings"].astype(np.float32)
    ordered = [groups[group_id] for group_id in ids]
    texts = [phrase_text(group) for group in ordered]
    if variant == "raw_phrase_hash":
        encoder = HashingVectorizer(
            n_features=256, alternate_sign=True, norm="l2",
            lowercase=True, ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b",
        )
        values = encoder.transform(texts).toarray().astype(np.float32)
    elif variant == "deterministic_context":
        values = deterministic_embeddings(ordered)
    else:
        values = transformer_embeddings(texts, variant, device)
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, group_ids=np.asarray(ids, dtype=object), embeddings=values)
    return ids, values


def build_group_data(resources: common.Resources, split: str) -> GroupData:
    groups = exact.make_groups(resources.context, split)
    frame = resources.candidates[split]
    by_group = {
        str(group_id): part.copy()
        for group_id, part in frame.groupby(frame["group_id"].astype(str), sort=False)
    }
    group_ids = list(groups)
    max_candidates = 12
    visual = np.zeros((len(group_ids), max_candidates, len(FEATURE_NAMES)), dtype=np.float32)
    quality = np.zeros((len(group_ids), max_candidates), dtype=np.float32)
    mask = np.zeros((len(group_ids), max_candidates), dtype=bool)
    counts = np.zeros(len(group_ids), dtype=np.int64)
    all_boxes: list[list[list[float]]] = []
    for group_index, group_id in enumerate(group_ids):
        group = dict(groups[group_id])
        group["group_id"] = group_id
        part = by_group.get(group_id, pd.DataFrame())
        candidates = (
            [candidate for candidate in common._candidate_dicts(part) if valid_candidate(candidate)][:max_candidates]
            if not part.empty else []
        )
        boxes = [[float(value) for value in row["box"]] for row in candidates]
        all_boxes.append(boxes)
        gt_boxes = [[float(value) for value in box] for box in group["gt_boxes"]]
        counts[group_index] = min(max(len(gt_boxes), 1), 4) - 1
        width, height = float(group["image_width"]), float(group["image_height"])
        dino = resources.context.dino[split].get(group_id)
        dino_xyxy = None
        if dino is not None:
            dino_xyxy = single_fusion.old_fusion.norm_to_xyxy(np.asarray(dino), width, height)
        semantic_values = [float(row["_row"].get("siglip_raw", 0.0)) for row in candidates]
        semantic_ranks = pd.Series(semantic_values).rank(method="average", pct=True).to_numpy() if candidates else []
        for candidate_index, (candidate, semantic_rank) in enumerate(zip(candidates, semantic_ranks)):
            box = candidate["box"]
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height)
            bw, bh = (x2 - x1) / width, (y2 - y1) / height
            row = candidate["_row"]
            source = str(candidate.get("source_model", ""))
            features = [
                cx, cy, bw, bh, bw * bh,
                float(row.get("confidence", 0.0)),
                1.0 / (1.0 + max(float(row.get("rank", 0.0)), 0.0)),
                box_iou(box, dino_xyxy) if dino_xyxy is not None else 0.0,
                float(semantic_rank),
                *[float(source == model) for model in SOURCE_MODELS],
            ]
            visual[group_index, candidate_index] = np.asarray(features, dtype=np.float32)
            mask[group_index, candidate_index] = True
            quality[group_index, candidate_index] = max(
                (box_iou(box, gt) for gt in gt_boxes), default=0.0
            )
    return GroupData(group_ids, visual, mask, quality, counts, all_boxes)


def standardize(train: GroupData, *others: GroupData) -> tuple[np.ndarray, np.ndarray]:
    values = train.visual[train.mask]
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    for data in (train, *others):
        data.visual[:] = (data.visual - mean[None, None, :]) / std[None, None, :]
        data.visual[~data.mask] = 0.0
    return mean, std


def loss_value(
    scores: torch.Tensor,
    count_logits: torch.Tensor,
    quality: torch.Tensor,
    count: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    regression = F.smooth_l1_loss(scores[mask], quality[mask])
    true_diff = quality[:, :, None] - quality[:, None, :]
    pred_diff = scores[:, :, None] - scores[:, None, :]
    pair_mask = mask[:, :, None] & mask[:, None, :] & (true_diff.abs() >= 0.10)
    pair_mask &= true_diff > 0
    ranking = F.relu(0.10 - pred_diff[pair_mask]).mean() if pair_mask.any() else regression.new_tensor(0.0)
    count_loss = F.cross_entropy(count_logits, count)
    return regression + 0.35 * ranking + 0.25 * count_loss


def tensors(data: GroupData, context: np.ndarray) -> tuple[torch.Tensor, ...]:
    return (
        torch.from_numpy(data.visual), torch.from_numpy(context),
        torch.from_numpy(data.mask), torch.from_numpy(data.quality),
        torch.from_numpy(data.count),
    )


def train_model(
    seed: int,
    train: GroupData,
    val: GroupData,
    train_context: np.ndarray,
    val_context: np.ndarray,
    device: torch.device,
    checkpoint: Path,
) -> tuple[ContextCandidateScorer, pd.DataFrame]:
    stable_seed(seed)
    model = ContextCandidateScorer(train.visual.shape[-1], train_context.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_tensors = tensors(train, train_context)
    val_tensors = tuple(value.to(device) for value in tensors(val, val_context))
    loader = DataLoader(TensorDataset(*train_tensors), batch_size=32, shuffle=True, num_workers=0)
    best_loss = float("inf")
    best_state = None
    patience = 0
    log_rows = []
    for epoch in range(1, 81):
        model.train()
        batch_losses = []
        for visual, context, mask, quality, count in loader:
            visual, context = visual.to(device), context.to(device)
            mask, quality, count = mask.to(device), quality.to(device), count.to(device)
            optimizer.zero_grad(set_to_none=True)
            scores, count_logits = model(visual, context, mask)
            loss = loss_value(scores, count_logits, quality, count, mask)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_scores, val_counts = model(val_tensors[0], val_tensors[1], val_tensors[2])
            val_loss = float(loss_value(val_scores, val_counts, val_tensors[3], val_tensors[4], val_tensors[2]).cpu())
        log_rows.append({"epoch": epoch, "train_loss": float(np.mean(batch_losses)), "val_loss": val_loss})
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= 12:
            break
    if best_state is None:
        raise RuntimeError("No checkpoint selected")
    model.load_state_dict(best_state)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "seed": seed, "best_val_loss": best_loss}, checkpoint)
    return model, pd.DataFrame(log_rows)


@torch.no_grad()
def predict(
    model: ContextCandidateScorer,
    data: GroupData,
    context: np.ndarray,
    resources: common.Resources,
    split: str,
    device: torch.device,
) -> tuple[dict[str, list[list[float]]], pd.DataFrame]:
    model.eval()
    visual, context_tensor, mask, _, _ = tensors(data, context)
    scores, counts = model(visual.to(device), context_tensor.to(device), mask.to(device))
    scores = scores.cpu().numpy()
    predicted_count = counts.argmax(dim=-1).cpu().numpy() + 1
    outputs: dict[str, list[list[float]]] = {}
    audit = []
    nms_iou = float(resources.set_params["nms_iou"])
    fallback_map = common.source_outputs(resources, split)
    for index, group_id in enumerate(data.group_ids):
        boxes = data.boxes[index]
        order = np.argsort(-scores[index, :len(boxes)], kind="stable") if boxes else []
        selected: list[list[float]] = []
        for candidate_index in order:
            box = boxes[int(candidate_index)]
            if all(box_iou(box, old) < nms_iou for old in selected):
                selected.append(box)
            if len(selected) >= int(predicted_count[index]):
                break
        fallback = False
        if not selected:
            selected = fallback_map.get(group_id, [])
            fallback = True
        outputs[group_id] = selected
        audit.append({
            "group_id": group_id, "predicted_count": int(predicted_count[index]),
            "n_candidates": len(boxes), "n_pred": len(selected), "fallback": int(fallback),
        })
    return outputs, pd.DataFrame(audit)


def run() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for directory in ("audit", "embeddings", "checkpoints", "logs", "metrics", "predictions"):
        (OUTPUT_ROOT / directory).mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_ROOT / "MODEL_CONFIG.json", {
        "variants": list(VARIANTS), "seeds": list(SEEDS),
        "visual_features": list(FEATURE_NAMES),
        "candidate_pool": "same seed-paired YOLO-640 top-12 pool for every context arm",
        "fixed_visual_signals": ["YOLO confidence", "RAD-DINO agreement", "SigLIP rank"],
        "network": "64-d visual adapter + 64-d context adapter + identical 2-layer score/count heads",
        "optimizer": "AdamW lr=1e-3 weight_decay=1e-4; max80; patience12",
        "forbidden_features": ["GT coordinates", "GT count", "target IoU", "eval finding metadata beyond declared query finding"],
        "selection": "validation loss only",
    })
    per_seed_rows = []
    all_contexts = []
    for protocol in ("888", "1444"):
        reference = common.load_resources(protocol, 13)
        train_groups = exact.make_groups(reference.context, "train")
        val_groups = exact.make_groups(reference.context, "val")
        all_contexts.append(reference.context)
        for variant in VARIANTS:
            train_ids, train_context = context_embeddings(protocol, "train", variant, train_groups, device)
            val_ids, val_context = context_embeddings(protocol, "val", variant, val_groups, device)
            if train_ids != list(train_groups) or val_ids != list(val_groups):
                raise RuntimeError("Context ID ordering mismatch")
            for seed in SEEDS:
                resources = common.load_resources(protocol, seed)
                train_data = build_group_data(resources, "train")
                val_data = build_group_data(resources, "val")
                if train_data.group_ids != train_ids or val_data.group_ids != val_ids:
                    raise RuntimeError(f"Split mismatch: {protocol}/{variant}/{seed}")
                mean, std = standardize(train_data, val_data)
                root = OUTPUT_ROOT / protocol / variant / f"seed_{seed}"
                checkpoint = OUTPUT_ROOT / "checkpoints" / protocol / variant / f"seed_{seed}.pt"
                model, log = train_model(
                    seed, train_data, val_data, train_context, val_context, device, checkpoint
                )
                log.to_csv(OUTPUT_ROOT / "logs" / f"{protocol}_{variant}_seed_{seed}.csv", index=False)
                # Eval is opened only after this seed's checkpoint is fixed.
                eval_groups = exact.make_groups(resources.context, "eval")
                eval_ids, eval_context = context_embeddings(protocol, "eval", variant, eval_groups, device)
                eval_data = build_group_data(resources, "eval")
                if eval_data.group_ids != eval_ids:
                    raise RuntimeError(f"Eval ID mismatch: {protocol}/{variant}/{seed}")
                eval_data.visual[:] = (eval_data.visual - mean[None, None, :]) / std[None, None, :]
                eval_data.visual[~eval_data.mask] = 0.0
                outputs, audit = predict(model, eval_data, eval_context, resources, "eval", device)
                summary, detail = common.evaluate(resources, outputs, "eval")
                expected = 163 if protocol == "888" else 220
                if int(summary["n"]) != expected or int(summary["n_missing_predictions"]) != 0:
                    raise RuntimeError(f"Denominator failure: {protocol}/{variant}/{seed}")
                root.mkdir(parents=True, exist_ok=True)
                common.save_prediction_map(root / "predictions.json", outputs)
                audit.to_csv(root / "prediction_audit.csv", index=False)
                detail.to_csv(root / "per_group.csv", index=False)
                write_json(root / "RUN_STATUS.json", {
                    "status": "complete", "protocol": protocol, "variant": variant,
                    "seed": seed, "checkpoint": str(checkpoint),
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "selection_split": "validation only", "eval_used_for_selection": False,
                })
                per_seed_rows.append({
                    "protocol": protocol, "method": variant, "seed": seed, **summary,
                })
    per_seed = pd.DataFrame(per_seed_rows)
    per_seed.to_csv(OUTPUT_ROOT / "metrics" / "context_per_seed.csv", index=False)
    aggregate = common.aggregate(per_seed)
    aggregate.to_csv(OUTPUT_ROOT / "metrics" / "context_aggregate.csv", index=False)
    write_json(OUTPUT_ROOT / "audit" / "SPLIT_OVERLAP_AUDIT.json", exact.split_audit(all_contexts))
    write_json(OUTPUT_ROOT / "FINAL_STATUS.json", {
        "status": "complete", "n_expected_runs": 24, "n_completed_runs": int(len(per_seed)),
        "seeds": list(SEEDS), "variants": list(VARIANTS),
        "same_visual_features": True, "same_model_and_loss": True,
        "missing_predictions": int(per_seed["n_missing_predictions"].sum()),
    })
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    run()

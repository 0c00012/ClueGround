#!/usr/bin/env python
"""Shared utilities for ClueGround YOLO--RAD-DINO + frozen SigLIP2 MoE.

SigLIP2 never generates a bounding box. It scores crops from the existing
YOLO/RAD-DINO candidate pool. A two-expert gate then combines:

1. the existing YOLO--RAD-DINO hybrid prediction;
2. the SigLIP2-selected candidate prediction.

BioMedCLIP and the original SigLIP checkpoint are intentionally not used.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image


SIGLIP2_MODEL_ID = "google/siglip2-base-patch16-224"
EXPERTS = ["hybrid", "siglip2"]


def resolve_siglip2_source(model_id: str) -> str:
    """Prefer the already downloaded frozen SigLIP2 checkpoint on this host."""
    candidates = [
        os.environ.get("SIGLIP2_LOCAL_SNAPSHOT", ""),
        "C:/Users/_idal/PycharmProjects/XAI/cache/huggingface/hub/"
        "models--google--siglip2-base-patch16-224/snapshots/"
        "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
    ]
    for candidate in candidates:
        path = Path(candidate) if candidate else None
        if path and (path / "config.json").is_file() and (path / "model.safetensors").is_file():
            print(f"[siglip2] using pinned local snapshot: {path}", flush=True)
            return str(path)
    return model_id


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def make_prompt(claim: str, finding: str, mode: str) -> str:
    claim = " ".join(str(claim).strip().split())
    finding = " ".join(str(finding).strip().split())
    if mode == "claim":
        return claim
    if mode == "cxr_claim":
        return f"chest x-ray showing {claim}"
    if mode == "finding_claim":
        return f"{finding}: {claim}"
    if mode == "region_prompt":
        return f"a radiographic region corresponding to {claim}"
    raise ValueError(f"Unknown prompt mode: {mode}")


def crop_candidate(row: pd.Series, margin: float) -> Image.Image:
    image_path = Path(str(row["image_path"]))
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    x1 = safe_float(row["pred_x1"])
    y1 = safe_float(row["pred_y1"])
    x2 = safe_float(row["pred_x2"])
    y2 = safe_float(row["pred_y2"])
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    x1 -= box_width * margin
    x2 += box_width * margin
    y1 -= box_height * margin
    y2 += box_height * margin
    crop_box = (
        int(max(0, min(width - 1, x1))),
        int(max(0, min(height - 1, y1))),
        int(max(1, min(width, x2))),
        int(max(1, min(height, y2))),
    )
    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        return image
    crop = image.crop(crop_box)
    # Transformers cannot infer an image channel layout for 1-pixel-wide or
    # 1-pixel-tall crops. Degenerate detector boxes carry no useful local
    # appearance signal, so score the original image instead.
    if crop.width < 2 or crop.height < 2:
        return image
    return crop


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype | None:
    normalized = str(name).lower()
    if normalized == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    choices = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in choices:
        raise ValueError(f"Unsupported SigLIP2 dtype: {name}")
    dtype = choices[normalized]
    if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    return dtype


def _load_siglip2(
    model_id: str,
    revision: str | None,
    device: torch.device,
    dtype_name: str,
) -> tuple[Any, Any, str | None]:
    try:
        from transformers import AutoModel, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Transformers is missing. Install a current release with SigLIP2 support."
        ) from exc

    load_kwargs: dict[str, Any] = {}
    if revision:
        load_kwargs["revision"] = revision
    dtype = _resolve_dtype(dtype_name, device)
    if dtype is not None:
        load_kwargs["torch_dtype"] = dtype
    if device.type == "cuda":
        load_kwargs["attn_implementation"] = "sdpa"

    source = resolve_siglip2_source(model_id)
    source_revision = None if source != model_id else revision
    if source != model_id:
        load_kwargs.pop("revision", None)
    try:
        processor = AutoProcessor.from_pretrained(source, revision=source_revision)
        try:
            model = AutoModel.from_pretrained(source, **load_kwargs)
        except (TypeError, ValueError):
            # Some Transformers versions reject attn_implementation for this
            # checkpoint. Retry without it rather than silently changing model.
            load_kwargs.pop("attn_implementation", None)
            model = AutoModel.from_pretrained(source, **load_kwargs)
    except Exception as exc:
        raise RuntimeError(
            "Could not load SigLIP2. Upgrade Transformers and verify access to "
            f"{model_id}. Original error: {exc}"
        ) from exc

    model = model.to(device).eval()
    resolved_revision = getattr(getattr(model, "config", None), "_commit_hash", None)
    return processor, model, resolved_revision


@torch.inference_mode()
def _score_siglip2_frame_with_loaded_model(
    frame: pd.DataFrame,
    *,
    processor: Any,
    model: torch.nn.Module,
    device: torch.device,
    prompt_mode: str,
    margin: float,
    batch_size: int,
    progress_label: str,
) -> pd.DataFrame:
    if frame.empty:
        raise ValueError(f"Cannot score an empty candidate table: {progress_label}")

    output = frame.copy().reset_index(drop=True)
    scores: list[float] = []
    effective_batch = max(1, int(batch_size))
    for start in range(0, len(output), effective_batch):
        sub = output.iloc[start : start + effective_batch]
        images = [crop_candidate(row, margin) for _, row in sub.iterrows()]
        texts = [
            make_prompt(row["claim_sentence"], row["finding"], prompt_mode)
            for _, row in sub.iterrows()
        ]
        inputs = processor(
            text=texts,
            images=images,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        model_dtype = next(model.parameters()).dtype
        moved: dict[str, Any] = {}
        for key, value in inputs.items():
            if not isinstance(value, torch.Tensor):
                moved[key] = value
            elif key == "pixel_values" and value.is_floating_point():
                moved[key] = value.to(device=device, dtype=model_dtype)
            else:
                moved[key] = value.to(device)
        result = model(**moved)
        logits = result.logits_per_image
        if logits.ndim == 2 and logits.shape[0] == logits.shape[1]:
            paired = torch.diag(logits)
        elif logits.ndim == 2 and logits.shape[1] == 1:
            paired = logits[:, 0]
        else:
            paired = logits.reshape(-1)[: len(sub)]
        scores.extend(paired.detach().float().cpu().tolist())
        print(
            f"[siglip2:{progress_label}] {start + len(sub)}/{len(output)}",
            flush=True,
        )

    output["siglip2_raw"] = np.asarray(scores, dtype=np.float64)
    output["siglip2_sigmoid"] = 1.0 / (
        1.0 + np.exp(-output["siglip2_raw"].clip(-50.0, 50.0))
    )
    group_column = next(
        (name for name in ("task_id", "group_id", "query_id") if name in output.columns),
        None,
    )
    if group_column is None:
        raise ValueError("Candidate table needs task_id, group_id, or query_id")

    output["siglip2_z"] = 0.0
    output["siglip2_rank"] = 0.0
    output["head_rank"] = 0.0
    for _, indices in output.groupby(group_column, sort=False).groups.items():
        idx = list(indices)
        raw = output.loc[idx, "siglip2_raw"].astype(float).to_numpy()
        mean = float(raw.mean())
        std = float(raw.std())
        output.loc[idx, "siglip2_z"] = (raw - mean) / (std if std > 1e-6 else 1.0)
        if len(raw) == 1:
            output.loc[idx, "siglip2_rank"] = 1.0
        else:
            rank = pd.Series(raw).rank(method="average", ascending=True).to_numpy()
            output.loc[idx, "siglip2_rank"] = (rank - 1.0) / (len(raw) - 1.0)

        head = output.loc[idx, "score_head"].astype(float).fillna(0.0).to_numpy()
        if len(head) == 1:
            output.loc[idx, "head_rank"] = 1.0
        else:
            rank = pd.Series(head).rank(method="average", ascending=True).to_numpy()
            output.loc[idx, "head_rank"] = (rank - 1.0) / (len(head) - 1.0)
    return output


@torch.inference_mode()
def score_siglip2_frames(
    frames: dict[str, pd.DataFrame],
    *,
    model_id: str = SIGLIP2_MODEL_ID,
    revision: str | None = None,
    prompt_mode: str = "cxr_claim",
    margin: float = 0.15,
    batch_size: int = 16,
    device_name: str | None = None,
    dtype_name: str = "auto",
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Score multiple split candidate tables while loading SigLIP2 only once."""

    if not frames:
        raise ValueError("No candidate tables were supplied for SigLIP2 scoring")
    device = torch.device(
        device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    processor, model, resolved_revision = _load_siglip2(
        model_id, revision, device, dtype_name
    )
    try:
        outputs = {
            label: _score_siglip2_frame_with_loaded_model(
                frame,
                processor=processor,
                model=model,
                device=device,
                prompt_mode=prompt_mode,
                margin=margin,
                batch_size=batch_size,
                progress_label=label,
            )
            for label, frame in frames.items()
        }
        provenance = {
            "model_id": model_id,
            "requested_revision": revision,
            "resolved_revision": resolved_revision,
            "prompt_mode": prompt_mode,
            "crop_margin": float(margin),
            "batch_size": int(batch_size),
            "device": str(device),
            "dtype": str(next(model.parameters()).dtype),
            "text_padding": "max_length",
            "text_max_length": 64,
            "frozen": True,
        }
        return outputs, provenance
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@torch.inference_mode()
def score_siglip2(
    frame: pd.DataFrame,
    *,
    model_id: str = SIGLIP2_MODEL_ID,
    revision: str | None = None,
    prompt_mode: str = "cxr_claim",
    margin: float = 0.15,
    batch_size: int = 16,
    device_name: str | None = None,
    dtype_name: str = "auto",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Backward-compatible one-table wrapper around :func:`score_siglip2_frames`."""

    outputs, provenance = score_siglip2_frames(
        {"single": frame},
        model_id=model_id,
        revision=revision,
        prompt_mode=prompt_mode,
        margin=margin,
        batch_size=batch_size,
        device_name=device_name,
        dtype_name=dtype_name,
    )
    return outputs["single"], provenance


def normalized_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.zeros(len(frame), dtype=np.float64)
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def apply_candidate_fusion_score(
    frame: pd.DataFrame,
    params: dict[str, float],
) -> pd.DataFrame:
    """Apply the original SigLIP candidate-fusion recipe with SigLIP2 scores.

    This mirrors the previous semantic expert: the upstream candidate score stays
    the anchor, while SigLIP2 z/rank scores and existing location/agreement terms
    provide validation-selected corrections.
    """

    output = frame.copy()
    output["siglip2_fusion_score"] = (
        float(params.get("w_head", 1.0)) * normalized_column(output, "score_head")
        + float(params.get("w_siglip2_z", 0.0))
        * normalized_column(output, "siglip2_z")
        + float(params.get("w_siglip2_rank", 0.0))
        * normalized_column(output, "siglip2_rank")
        + float(params.get("w_prior", 0.0)) * normalized_column(output, "prior_iou")
        + float(params.get("w_xattn", 0.0)) * normalized_column(output, "xattn_iou")
        + float(params.get("w_conf", 0.0)) * normalized_column(output, "confidence")
    )
    return output


def apply_fusion_score(frame: pd.DataFrame, head_weight: float) -> pd.DataFrame:
    """Compatibility wrapper for earlier two-rank patch versions."""

    if not 0.0 <= float(head_weight) <= 1.0:
        raise ValueError("head_weight must be in [0, 1]")
    output = frame.copy()
    output["siglip2_fusion_score"] = (
        float(head_weight) * output["head_rank"].astype(float)
        + (1.0 - float(head_weight)) * output["siglip2_rank"].astype(float)
    )
    return output


def candidate_map(
    frame: pd.DataFrame,
    groups: dict[str, dict[str, Any]],
    score_column: str = "siglip2_fusion_score",
) -> dict[str, list[dict[str, Any]]]:
    task_to_group: dict[str, str] = {}
    for group_id, group in groups.items():
        task_ids = group.get("task_ids", [group_id])
        for task_id in task_ids:
            task_to_group[str(task_id)] = str(group_id)
        task_to_group[str(group_id)] = str(group_id)

    output: dict[str, list[dict[str, Any]]] = {str(gid): [] for gid in groups}
    for row in frame.itertuples(index=False):
        raw_id = str(getattr(row, "task_id", getattr(row, "group_id", "")))
        group_id = task_to_group.get(raw_id, raw_id if raw_id in groups else None)
        if group_id is None:
            continue
        box = [
            safe_float(getattr(row, "pred_x1")),
            safe_float(getattr(row, "pred_y1")),
            safe_float(getattr(row, "pred_x2")),
            safe_float(getattr(row, "pred_y2")),
        ]
        output[group_id].append(
            {
                "box": box,
                "score": safe_float(getattr(row, score_column)),
                "source": "siglip2_candidate_fusion",
                "source_model": str(getattr(row, "source_model", "")),
            }
        )
    for rows in output.values():
        rows.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return output


def nms_select(
    candidates: Iterable[dict[str, Any]],
    *,
    max_k: int,
    nms_iou: float,
    score_ratio: float,
    iou_fn: Any,
) -> list[dict[str, Any]]:
    ordered = sorted(
        (dict(item) for item in candidates),
        key=lambda item: float(item.get("score", 0.0)),
        reverse=True,
    )
    if not ordered:
        return []
    top_score = float(ordered[0].get("score", 0.0))
    threshold = top_score * float(score_ratio) if top_score > 0 else -math.inf
    selected: list[dict[str, Any]] = []
    for candidate in ordered:
        if len(selected) >= int(max_k):
            break
        if selected and float(candidate.get("score", 0.0)) < threshold:
            continue
        if all(
            float(iou_fn(candidate["box"], previous["box"])) < float(nms_iou)
            for previous in selected
        ):
            selected.append(candidate)
    return selected or [ordered[0]]


def decode_phrase_conditioned(
    ranked_candidates: Iterable[dict[str, Any]],
    raw_phrase: str,
    parsed_context: dict[str, Any],
    decoder_config: dict[str, Any],
    *,
    iou_fn: Any,
) -> list[dict[str, Any]]:
    """Protocol-agnostic phrase-conditioned candidate-set decoder.

    Both 888 and 1444 call this exact function.  Cardinality is determined by
    phrase context (with a raw-text fallback), never by protocol name or gold
    annotations.  The 888 evaluator may subsequently score only the first
    ranked box; it does not alter the decoder output.
    """
    phrase = str(raw_phrase).lower()
    has_multi = bool(parsed_context.get("has_multi_cue", False))
    if not has_multi:
        has_multi = any(token in phrase for token in (
            "bilateral", "both", "bibasilar", "multifocal", "multilobar", "multiple", "two ",
        ))
    k_hint = max(1, int(parsed_context.get("k_hint", 1)))
    max_k = 1
    if has_multi:
        max_k = min(
            max(k_hint, int(decoder_config["min_k_if_cue"])),
            int(decoder_config["max_k_if_cue"]),
        )
    return nms_select(
        ranked_candidates,
        max_k=max_k,
        nms_iou=float(decoder_config["nms_iou"]),
        score_ratio=float(decoder_config["score_ratio"]),
        iou_fn=iou_fn,
    )


@torch.inference_mode()
def predict_finding_gate(
    model: torch.nn.Module,
    bundle: Any,
    *,
    experts: list[str],
    feature_builder: Any,
    moe_module: Any,
) -> dict[str, list[dict[str, Any]]]:
    """Generic finding-conditioned gate prediction for any expert count."""

    model_device = next(model.parameters()).device
    model.eval()
    multi = moe_module.has_multi_cue(bundle.cue)
    output: dict[str, list[dict[str, Any]]] = {}
    for group_id, group in bundle.groups.items():
        prediction = bundle.hybrid.get(group_id, [])
        feature = feature_builder(bundle, group_id, experts)
        if feature is not None and not multi.get(group_id, False):
            values, boxes, _ = feature
            tensor = torch.from_numpy(values[None, :]).float().to(model_device)
            weights = torch.softmax(model(tensor), dim=-1)[0].detach().cpu().numpy()
            box_norm = (boxes * weights[:, None]).sum(axis=0)
            box_norm[:2] = np.clip(box_norm[:2], 0.0, 1.0)
            box_norm[2:] = np.clip(box_norm[2:], 1e-4, 1.0)
            box = moe_module.norm_to_xyxy(
                box_norm,
                float(group["image_width"]),
                float(group["image_height"]),
            )
            prediction = [
                {
                    "box": box,
                    "score": float(weights.max()),
                    "source": "finding_conditioned_siglip2_gate",
                    "expert_weights": {
                        name: float(weight) for name, weight in zip(experts, weights)
                    },
                }
            ]
        output[str(group_id)] = prediction
    return output

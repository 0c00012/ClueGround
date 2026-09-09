from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np

from .contracts import ProtocolSpec


MSCXR_FINDINGS = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Lung Opacity",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
)


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def stable_hash(value: str, length: int = 20) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:length]


def parse_location(text: str) -> dict[str, str]:
    value = f" {normalize_text(text)} "
    if re.search(r"\b(bilateral|bilaterally|both|bibasilar|bibasal)\b", value):
        laterality = "bilateral"
    elif re.search(r"\b(right|rt)\b", value):
        laterality = "right"
    elif re.search(r"\b(left|lt)\b", value):
        laterality = "left"
    else:
        laterality = "unknown"

    if re.search(r"\b(apical|apex)\b", value):
        vertical = "apical"
    elif re.search(r"\b(upper|superior)\b", value):
        vertical = "upper"
    elif re.search(r"\b(mid|middle|hilar)\b", value):
        vertical = "mid"
    elif re.search(r"\b(lower|inferior)\b", value):
        vertical = "lower"
    elif re.search(r"\b(basal|base|basilar|bibasilar|bibasal)\b", value):
        vertical = "basal"
    elif re.search(r"\b(diffuse|throughout|widespread)\b", value):
        vertical = "whole"
    else:
        vertical = "unknown"
    return {"laterality": laterality, "vertical": vertical}


def build_model_query(spec: ProtocolSpec, row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Build the model-visible query and record exactly which fields supplied it."""

    if spec.query_mode == "raw_phrase":
        phrase = row.get("phrase") or row.get("claim_sentence") or row.get("sentence") or ""
        finding = str(row.get("finding") or "").strip()
        if finding not in MSCXR_FINDINGS:
            raise ValueError(f"Unknown MS-CXR query finding {finding!r}")
        query = normalize_text(f"finding {finding}; phrase {phrase}")
        lineage = {
            "mode": spec.query_mode,
            "source_fields": ["finding query", "phrase/raw claim"],
            "target_reference_used": False,
            "note": "The finding names what to localize; it is query metadata, not a spatial target.",
        }
    elif spec.query_mode == "anatomy_name":
        # The anatomy label is the public query. Do not silently fall back to a
        # target-reference column even if a malformed source row omits it.
        anatomy = row.get("finding") or row.get("finding_name") or ""
        query = normalize_text(anatomy)
        lineage = {
            "mode": spec.query_mode,
            "source_fields": ["anatomy task query"],
            "target_reference_used": False,
            "note": "The anatomy name defines the query itself; it is not a hidden target feature.",
        }
    elif spec.query_mode == "device_claim":
        device = normalize_text(row.get("finding") or row.get("finding_name"))
        claim = normalize_text(row.get("claim_sentence") or row.get("phrase") or row.get("sentence"))
        query = normalize_text(f"device {device}; claim {claim}")
        lineage = {
            "mode": spec.query_mode,
            "source_fields": ["device task query", "raw report claim"],
            "target_reference_used": False,
            "explicitly_excluded_fields": ["bbox_name_reference", "object_name_reference"],
        }
    else:
        raise ValueError(spec.query_mode)
    if not query:
        raise ValueError(f"Empty query for {spec.key}: {row.get('task_id', '<unknown>')}")
    return query, lineage


def token_hash_features(text: str, dim: int = 64) -> np.ndarray:
    values = np.zeros(dim, dtype=np.float32)
    for token in re.findall(r"[a-z0-9]+", normalize_text(text)):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        values[int.from_bytes(digest[:4], "little") % dim] += 1.0
    norm = float(np.linalg.norm(values))
    return values / norm if norm > 0 else values


def encode_query(text: str, task_key: str, dim: int = 64) -> np.ndarray:
    location = parse_location(text)
    laterality_vocab = ("right", "left", "bilateral", "unknown")
    vertical_vocab = ("apical", "upper", "mid", "lower", "basal", "whole", "unknown")
    task_vocab = ("mscxr_multibox_1444", "imagenome_anatomy_10k", "imagenome_device_weak_10k")
    categorical = [float(location["laterality"] == value) for value in laterality_vocab]
    categorical += [float(location["vertical"] == value) for value in vertical_vocab]
    categorical += [float(task_key == value) for value in task_vocab]
    return np.concatenate([np.asarray(categorical, dtype=np.float32), token_hash_features(text, dim)])


def context_score(query_text: str, box: list[float], image_width: float, image_height: float) -> float:
    location = parse_location(query_text)
    center_x = (float(box[0]) + float(box[2])) * 0.5 / max(float(image_width), 1.0)
    center_y = (float(box[1]) + float(box[3])) * 0.5 / max(float(image_height), 1.0)
    score = 0.5
    if location["laterality"] == "right":
        score += 0.25 if center_x < 0.5 else -0.25
    elif location["laterality"] == "left":
        score += 0.25 if center_x >= 0.5 else -0.25
    expected_y = {"apical": 0.18, "upper": 0.30, "mid": 0.52, "lower": 0.72, "basal": 0.84}.get(
        location["vertical"]
    )
    if expected_y is not None:
        score += 0.25 * (1.0 - min(abs(center_y - expected_y) / 0.5, 1.0))
    return float(np.clip(score, 0.0, 1.0))

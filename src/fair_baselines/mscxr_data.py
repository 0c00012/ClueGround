from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = (
    PROJECT_ROOT / "training" / "ms_cxr_vfm_localizer_stage1_p10_p19" / "datasets"
)


def _norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _box_key(box: Iterable[float]) -> tuple[float, float, float, float]:
    values = tuple(round(float(value), 4) for value in box)
    if len(values) != 4:
        raise ValueError(f"Expected four box coordinates, got {values!r}")
    return values


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class GroundingSample:
    group_id: str
    split: str
    image_path: str
    dicom_id: str
    study_id: str
    subject_id: str
    finding: str
    phrase: str
    image_width: int
    image_height: int
    gt_boxes: tuple[tuple[float, float, float, float], ...]

    @property
    def gold_count(self) -> int:
        return len(self.gt_boxes)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def group_id_for_row(row: dict[str, Any]) -> str:
    phrase = row.get("phrase") or row.get("claim_sentence") or row.get("sentence") or ""
    return f"{row['dicom_id']}|{_norm_text(phrase)}"


def load_grouped_split(split: str, data_root: Path = DEFAULT_DATA_ROOT) -> list[GroundingSample]:
    path = data_root / f"{split}.jsonl"
    rows = read_jsonl(path)
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        group_id = group_id_for_row(row)
        phrase = row.get("phrase") or row.get("claim_sentence") or row.get("sentence") or ""
        raw_boxes = row.get("gold_group_bboxes_xyxy") or [row["gold_bbox_xyxy"]]
        box_map = {_box_key(box): _box_key(box) for box in raw_boxes}
        if group_id not in groups:
            groups[group_id] = {"row": row, "phrase": str(phrase), "boxes": box_map}
        else:
            groups[group_id]["boxes"].update(box_map)

    samples: list[GroundingSample] = []
    for group_id, payload in sorted(groups.items()):
        row = payload["row"]
        boxes = tuple(sorted(payload["boxes"].values()))
        sample = GroundingSample(
            group_id=group_id,
            split=split,
            image_path=str(row["image_path"]),
            dicom_id=str(row["dicom_id"]),
            study_id=str(row["study_id"]),
            subject_id=str(row["subject_id"]),
            finding=str(row.get("finding", "")),
            phrase=str(payload["phrase"]),
            image_width=int(row["image_width"]),
            image_height=int(row["image_height"]),
            gt_boxes=boxes,
        )
        if not Path(sample.image_path).is_file():
            raise FileNotFoundError(sample.image_path)
        samples.append(sample)
    return samples


def load_all_splits(data_root: Path = DEFAULT_DATA_ROOT) -> dict[str, list[GroundingSample]]:
    return {split: load_grouped_split(split, data_root) for split in ("train", "val", "eval")}


def audit_splits(
    splits: dict[str, list[GroundingSample]],
    data_root: Path = DEFAULT_DATA_ROOT,
    protocol: str = "multibox_1444",
) -> dict[str, Any]:
    fields = ("group_id", "subject_id", "study_id", "dicom_id")
    overlap: dict[str, dict[str, int]] = {}
    for left, right in (("train", "val"), ("train", "eval"), ("val", "eval")):
        overlap[f"{left}_{right}"] = {
            field: len({getattr(x, field) for x in splits[left]} & {getattr(x, field) for x in splits[right]})
            for field in fields
        }
    counts = {
        split: {
            "phrase_groups": len(rows),
            "gt_boxes": sum(row.gold_count for row in rows),
            "singletons": sum(row.gold_count == 1 for row in rows),
            "patients": len({row.subject_id for row in rows}),
            "studies": len({row.study_id for row in rows}),
            "dicoms": len({row.dicom_id for row in rows}),
        }
        for split, rows in splits.items()
    }
    if protocol == "multibox_1444":
        expected = {
            "train": {"phrase_groups": 813, "gt_boxes": 996, "singletons": 638},
            "val": {"phrase_groups": 124, "gt_boxes": 164, "singletons": 87},
            "eval": {"phrase_groups": 220, "gt_boxes": 280, "singletons": 163},
        }
    elif protocol == "singlebox_888":
        expected = {
            "train": {"phrase_groups": 638, "gt_boxes": 638, "singletons": 638},
            "val": {"phrase_groups": 87, "gt_boxes": 87, "singletons": 87},
            "eval": {"phrase_groups": 163, "gt_boxes": 163, "singletons": 163},
        }
    else:
        raise ValueError(f"Unsupported protocol: {protocol}")
    count_pass = all(
        counts[split][key] == value
        for split, values in expected.items()
        for key, value in values.items()
    )
    overlap_pass = all(value == 0 for pair in overlap.values() for value in pair.values())
    return {
        "counts": counts,
        "expected_counts": expected,
        "count_pass": count_pass,
        "cross_split_overlap": overlap,
        "overlap_pass": overlap_pass,
        "pass": count_pass and overlap_pass,
        "protocol": protocol,
        "source_hashes": {
            split: file_sha256(data_root / f"{split}.jsonl") for split in ("train", "val", "eval")
        },
        "training_policy": "train groups only; val for checkpoint and decoder selection; eval for one final pass",
        "group_key": "dicom_id + normalized phrase, matching the canonical multibox evaluator",
        "known_duplicate_semantics": (
            "One train phrase and one val phrase have identical boxes under both Consolidation and Edema; "
            "they are collapsed to one phrase group and exact duplicate rectangles are removed."
        ),
    }


def letterbox_geometry(width: int, height: int, image_size: int) -> dict[str, float | int]:
    scale = min(image_size / float(width), image_size / float(height))
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    pad_x = (image_size - resized_w) // 2
    pad_y = (image_size - resized_h) // 2
    return {
        "scale": scale,
        "resized_w": resized_w,
        "resized_h": resized_h,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "image_size": image_size,
        "orig_w": width,
        "orig_h": height,
    }


def load_letterboxed(sample: GroundingSample, image_size: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    with Image.open(sample.image_path) as source:
        image = source.convert("L")
        geometry = letterbox_geometry(image.width, image.height, image_size)
        resized = image.resize(
            (int(geometry["resized_w"]), int(geometry["resized_h"])),
            resample=Image.Resampling.BILINEAR,
        )
    canvas = np.zeros((image_size, image_size), dtype=np.float32)
    pad_x = int(geometry["pad_x"])
    pad_y = int(geometry["pad_y"])
    resized_array = np.asarray(resized, dtype=np.float32) / 255.0
    canvas[pad_y : pad_y + resized_array.shape[0], pad_x : pad_x + resized_array.shape[1]] = resized_array

    mask = np.zeros((image_size, image_size), dtype=np.float32)
    scale = float(geometry["scale"])
    for box in sample.gt_boxes:
        x1 = int(np.floor(box[0] * scale + pad_x))
        y1 = int(np.floor(box[1] * scale + pad_y))
        x2 = int(np.ceil(box[2] * scale + pad_x))
        y2 = int(np.ceil(box[3] * scale + pad_y))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image_size, x2), min(image_size, y2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1.0
    image_tensor = torch.from_numpy(np.repeat(canvas[None, :, :], 3, axis=0))
    mask_tensor = torch.from_numpy(mask[None, :, :])
    return image_tensor, mask_tensor, geometry


class MaskGroundingDataset(Dataset):
    def __init__(self, samples: list[GroundingSample], image_size: int = 224) -> None:
        self.samples = samples
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image, mask, geometry = load_letterboxed(sample, self.image_size)
        return {
            "image": image,
            "mask": mask,
            "phrase": sample.phrase,
            "sample": sample,
            "geometry": geometry,
        }


def collate_mask_batch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": torch.stack([row["image"] for row in rows]),
        "masks": torch.stack([row["mask"] for row in rows]),
        "phrases": [row["phrase"] for row in rows],
        "samples": [row["sample"] for row in rows],
        "geometries": [row["geometry"] for row in rows],
    }

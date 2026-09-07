from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_PTH_ROOT = PROJECT_ROOT / "third_party" / "MedRPG" / "data" / "MS_CXR"
OFFICIAL_IMAGE_ROOT = PROJECT_ROOT / "third_party" / "MedRPG" / "ln_data" / "MS_CXR"

CATEGORY_ID_TO_NAME = {
    # Exact order used by MedRPG's released MS-CXR preprocessing.
    1: "Cardiomegaly",
    2: "Lung Opacity",
    3: "Edema",
    4: "Consolidation",
    5: "Pneumonia",
    6: "Atelectasis",
    7: "Pneumothorax",
    8: "Pleural Effusion",
}


def normalize_phrase(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).lower())).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


@dataclass(frozen=True)
class OfficialSample:
    official_sample_id: str
    split: str
    row_index: int
    anno_id: int
    image_id: int
    category_id: int
    category_name: str
    relative_image_path: str
    image_path: str
    dicom_id: str
    subject_id: str
    study_id: str
    phrase: str
    phrase_norm: str
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    image_width: int
    image_height: int
    source_width: int
    source_height: int

    @property
    def query_key(self) -> str:
        return f"{self.dicom_id}|{self.phrase_norm}"

    @property
    def bbox_xywh(self) -> list[float]:
        return [
            self.bbox_x1,
            self.bbox_y1,
            self.bbox_x2 - self.bbox_x1,
            self.bbox_y2 - self.bbox_y1,
        ]

    @property
    def bbox_xyxy(self) -> list[float]:
        return [self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2]


def _ids_from_relative_path(relative_path: str) -> tuple[str, str, str]:
    path = Path(str(relative_path).replace("\\", "/"))
    dicom_id = path.stem
    # The first pXX component is the MIMIC shard, not the patient identifier.
    subject = next((part[1:] for part in path.parts if re.fullmatch(r"p\d{8}", part)), "")
    study = next((part[1:] for part in path.parts if re.fullmatch(r"s\d{8}", part)), "")
    return subject, study, dicom_id


def load_official_mscxr(
    pth_root: Path = OFFICIAL_PTH_ROOT,
    image_root: Path = OFFICIAL_IMAGE_ROOT,
) -> list[OfficialSample]:
    samples: list[OfficialSample] = []
    for split in ("train", "val", "test"):
        path = pth_root / f"MS_CXR_{split}.pth"
        rows = torch.load(path, map_location="cpu", weights_only=False)
        for index, row in enumerate(rows):
            if len(row) != 8:
                raise ValueError(f"Unexpected official MS-CXR row in {path}: {row!r}")
            anno_id, image_id, category_id, rel_path, bbox_xywh, source_w, source_h, phrase = row
            x, y, width, height = [float(value) for value in bbox_xywh]
            subject_id, study_id, dicom_id = _ids_from_relative_path(str(rel_path))
            category_id = int(category_id)
            samples.append(
                OfficialSample(
                    official_sample_id=f"official_{split}_{index + 1:04d}",
                    split=split,
                    row_index=index,
                    anno_id=int(anno_id),
                    image_id=int(image_id),
                    category_id=category_id,
                    category_name=CATEGORY_ID_TO_NAME[category_id],
                    relative_image_path=str(rel_path).replace("\\", "/"),
                    image_path=str((image_root / str(rel_path)).resolve()),
                    dicom_id=dicom_id,
                    subject_id=subject_id,
                    study_id=study_id,
                    phrase=str(phrase),
                    phrase_norm=normalize_phrase(str(phrase)),
                    bbox_x1=x,
                    bbox_y1=y,
                    bbox_x2=x + width,
                    bbox_y2=y + height,
                    image_width=640,
                    image_height=640,
                    source_width=int(source_w),
                    source_height=int(source_h),
                )
            )
    return samples


def sample_frame(samples: Iterable[OfficialSample]) -> pd.DataFrame:
    rows = []
    for sample in samples:
        row = asdict(sample)
        row["query_key"] = sample.query_key
        row["gold_boxes_json"] = json.dumps([sample.bbox_xyxy])
        rows.append(row)
    return pd.DataFrame(rows)


def split_audit(samples: Iterable[OfficialSample]) -> dict[str, Any]:
    frame = sample_frame(samples)
    counts = frame.groupby("split").size().astype(int).to_dict()
    overlaps: list[dict[str, Any]] = []
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        a = frame[frame["split"].eq(left)]
        b = frame[frame["split"].eq(right)]
        for column in ("subject_id", "study_id", "dicom_id", "query_key", "anno_id"):
            overlap = sorted(set(a[column].astype(str)) & set(b[column].astype(str)))
            overlaps.append(
                {
                    "left": left,
                    "right": right,
                    "identity": column,
                    "n_overlap": len(overlap),
                    "examples": overlap[:5],
                }
            )
    missing_images = frame.loc[~frame["image_path"].map(lambda value: Path(value).exists())]
    return {
        "counts": counts,
        "n_total": int(len(frame)),
        "n_unique_query_keys": int(frame["query_key"].nunique()),
        "n_missing_images": int(len(missing_images)),
        "missing_images": missing_images["image_path"].head(20).tolist(),
        "overlaps": overlaps,
        "split_leak_pass": all(row["n_overlap"] == 0 for row in overlaps),
    }


def export_official_protocol(samples: list[OfficialSample], output_root: Path) -> dict[str, str]:
    output_root.mkdir(parents=True, exist_ok=True)
    frame = sample_frame(samples)
    manifest = output_root / "official_mscxr_638_85_167.csv"
    frame.to_csv(manifest, index=False)

    jsonl = output_root / "official_mscxr_vlm.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(
                json.dumps(
                    {
                        "protocol": "official_singlebox_890",
                        "split": sample.split,
                        "query_id": sample.official_sample_id,
                        "image_id": sample.dicom_id,
                        "dicom_id": sample.dicom_id,
                        "study_id": sample.study_id,
                        "subject_id": sample.subject_id,
                        "image_path": sample.image_path,
                        "phrase": sample.phrase,
                        "finding": sample.category_name,
                        "gold_boxes": [sample.bbox_xyxy],
                        "image_width": 640,
                        "image_height": 640,
                        "coordinate_frame": "pixel_xyxy_medrpg_640",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    medrpg_root = output_root / "medrpg_split_root" / "MS_CXR"
    medrpg_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        shutil.copy2(OFFICIAL_PTH_ROOT / f"MS_CXR_{split}.pth", medrpg_root / f"MS_CXR_{split}.pth")

    transvg_root = output_root / "transvg_split_root" / "flickr"
    transvg_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        rows = [
            (sample.image_path, sample.bbox_xyxy, sample.phrase)
            for sample in samples
            if sample.split == split
        ]
        torch.save(rows, transvg_root / f"flickr_{split}.pth")

    medgrounder_rows = []
    for sample in samples:
        medgrounder_rows.append(
            {
                "sample_id": sample.official_sample_id,
                "group_id": sample.query_key,
                "split": sample.split,
                "label_text": sample.phrase,
                "path": sample.relative_image_path,
                "category_name": sample.category_name,
                "x": sample.bbox_x1,
                "y": sample.bbox_y1,
                "w": sample.bbox_x2 - sample.bbox_x1,
                "h": sample.bbox_y2 - sample.bbox_y1,
                "image_width": 640,
                "image_height": 640,
                "subject_id": sample.subject_id,
                "study_id": sample.study_id,
                "dicom_id": sample.dicom_id,
                "ms_cxr_annotation_id": sample.anno_id,
                "bbox_type": "official_mscxr_phrase_grounding_bbox",
                "dataset_variant": "official_638_85_167",
            }
        )
    medgrounder_csv = output_root / "medgrounder" / "official_mscxr.csv"
    medgrounder_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(medgrounder_rows).to_csv(medgrounder_csv, index=False)

    agpt_root = output_root / "agpt_split_root" / "MS_CXR"
    agpt_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        rows = [
            (
                sample.subject_id,
                sample.study_id,
                sample.category_id,
                sample.relative_image_path,
                sample.bbox_xywh,
                640,
                640,
                sample.phrase,
            )
            for sample in samples
            if sample.split == split
        ]
        torch.save(rows, agpt_root / f"MS_CXR_{split}.pth")

    audit_path = output_root / "official_split_audit.json"
    write_json(audit_path, split_audit(samples))
    return {
        "manifest_csv": str(manifest),
        "vlm_jsonl": str(jsonl),
        "medrpg_split_root": str(medrpg_root.parent),
        "transvg_split_root": str(transvg_root.parent),
        "medgrounder_csv": str(medgrounder_csv),
        "agpt_split_root": str(agpt_root),
        "image_root": str(OFFICIAL_IMAGE_ROOT),
        "audit": str(audit_path),
    }

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image

from src.fair_baselines.mscxr_data import GroundingSample, audit_splits, load_all_splits


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "experiments" / "controlled_direct_baseline_matrix" / "20260718_v1"
DEFAULT_DATA = ROOT / "training" / "ms_cxr_vfm_localizer_stage1_p10_p19" / "datasets"
SEEDS = (13, 42, 2026)
PROTOCOLS = ("singlebox_888", "multibox_1444")
MODELS = ("transvg", "medrpg", "reclmis", "mdetr_style")
FINDING_TO_ID = {
    "Cardiomegaly": 1,
    "Lung Opacity": 2,
    "Edema": 3,
    "Consolidation": 4,
    "Pneumonia": 5,
    "Atelectasis": 6,
    "Pneumothorax": 7,
    "Pleural Effusion": 8,
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_splits(protocol: str, data_root: Path = DEFAULT_DATA) -> dict[str, list[GroundingSample]]:
    if protocol not in PROTOCOLS:
        raise ValueError(protocol)
    source = load_all_splits(data_root)
    if protocol == "singlebox_888":
        return {split: [sample for sample in rows if sample.gold_count == 1] for split, rows in source.items()}
    return source


def enclosing_box(sample: GroundingSample) -> list[float]:
    boxes = sample.gt_boxes
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def medrpg_relative_image(sample: GroundingSample) -> str:
    match = re.search(r"[\\/](p\d{2})[\\/](p\d+)[\\/](s\d+)[\\/]([^\\/]+)$", sample.image_path)
    if match is None:
        raise ValueError(f"Cannot derive MedRPG image path: {sample.image_path}")
    return "/".join(("files", *match.groups()))


def canonical_row(sample: GroundingSample, target: list[float]) -> dict[str, Any]:
    return {
        "group_id": sample.group_id,
        "subject_id": sample.subject_id,
        "study_id": sample.study_id,
        "dicom_id": sample.dicom_id,
        "image_path": sample.image_path,
        "image_width": sample.image_width,
        "image_height": sample.image_height,
        "finding": sample.finding,
        "phrase": sample.phrase,
        "gold_boxes_json": json.dumps(sample.gt_boxes),
        "training_target_xyxy_json": json.dumps(target),
        "gold_count": sample.gold_count,
    }


def transvg_input_box(sample: GroundingSample, target: list[float]) -> list[float]:
    """Scale canonical coordinates to the pixels of the actual TransVG input."""
    with Image.open(sample.image_path) as image:
        input_width, input_height = image.size
    x1, y1, x2, y2 = target
    return [
        x1 / sample.image_width * input_width,
        y1 / sample.image_height * input_height,
        x2 / sample.image_width * input_width,
        y2 / sample.image_height * input_height,
    ]


def build_protocol_data(output_root: Path, protocol: str) -> dict[str, Any]:
    splits = protocol_splits(protocol)
    audit = audit_splits(splits, DEFAULT_DATA, protocol=protocol)
    if not audit["pass"]:
        raise RuntimeError(f"Split audit failed for {protocol}: {audit}")

    protocol_root = output_root / "data" / protocol
    transvg_root = protocol_root / "transvg" / "flickr"
    agpt_root = protocol_root / "agpt_medrpg"
    agpt_split = agpt_root / "split_root" / "MS_CXR"
    transvg_root.mkdir(parents=True, exist_ok=True)
    agpt_split.mkdir(parents=True, exist_ok=True)
    summaries = []
    for split, samples in splits.items():
        target_rows = [(sample, enclosing_box(sample)) for sample in samples]
        pd.DataFrame([canonical_row(sample, target) for sample, target in target_rows]).to_csv(
            protocol_root / f"{split}_canonical.csv", index=False
        )
        transvg_rows = [
            (sample.image_path, transvg_input_box(sample, target), sample.phrase)
            for sample, target in target_rows
        ]
        transvg_name = "test" if split == "eval" else split
        torch.save(transvg_rows, transvg_root / f"flickr_{transvg_name}.pth")

        medrpg_rows = []
        for annotation_id, (sample, target) in enumerate(target_rows, start=1):
            x1, y1, x2, y2 = target
            bbox_640 = (
                int(round(x1 / sample.image_width * 640)),
                int(round(y1 / sample.image_height * 640)),
                max(1, int(round((x2 - x1) / sample.image_width * 640))),
                max(1, int(round((y2 - y1) / sample.image_height * 640))),
            )
            medrpg_rows.append(
                (
                    annotation_id,
                    int(sample.study_id),
                    FINDING_TO_ID[sample.finding],
                    medrpg_relative_image(sample),
                    bbox_640,
                    sample.image_width,
                    sample.image_height,
                    sample.phrase,
                )
            )
        medrpg_name = "test" if split == "eval" else split
        torch.save(medrpg_rows, agpt_split / f"MS_CXR_{medrpg_name}.pth")
        summaries.append(
            {
                "split": split,
                "phrase_groups": len(samples),
                "gt_boxes": sum(sample.gold_count for sample in samples),
                "training_targets": len(target_rows),
            }
        )

    write_json(
        agpt_root / "dataset_info.json",
        {
            "protocol": protocol,
            "image_root": str(ROOT / "third_party" / "MedRPG" / "ln_data" / "MS_CXR"),
            "split_root": str(agpt_root / "split_root"),
            "target_policy": "singleton box" if protocol == "singlebox_888" else "phrase-group enclosing box",
            "eval_policy": "one prediction per phrase group; common set evaluator uses all original GT boxes",
        },
    )
    write_json(protocol_root / "split_audit.json", audit)
    pd.DataFrame(summaries).to_csv(protocol_root / "split_summary.csv", index=False)
    return {"protocol": protocol, "audit": audit, "splits": summaries}


def expected_artifact(output_root: Path, model: str, protocol: str, seed: int) -> Path:
    run = output_root / "runs" / protocol / model / f"seed_{seed}"
    if model == "transvg":
        return run / "eval" / "transvg_best_miou_checkpoint_test_predictions.csv"
    if model == "medrpg":
        return run / "eval" / "bbox_save.pth"
    if model == "reclmis":
        return run / "fair" / "runs" / "reclmis" / f"seed_{seed}" / "predictions" / "eval_predictions.csv"
    if model == "mdetr_style":
        return run / "mdetr" / "predictions" / f"mdetr_token_box_{protocol}_s{seed}_eval_predictions.csv"
    raise ValueError(model)

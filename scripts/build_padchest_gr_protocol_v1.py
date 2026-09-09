#!/usr/bin/env python
"""Build the PadChest-GR external-evaluation protocol for ClueGround.

Reads the official PadChest-GR release (``grounded_reports_20240819.json``,
``master_table.csv.zip`` and the split image archive ``PadChest_GR.zip.001..``)
and writes, for the *test* split, one evaluation row per grounded sentence
whose PadChest-GR label maps onto one of the eight MS-CXR finding classes.

Outputs (under ``--out-root``):

* ``images/<ImageID>.jpg``: 8-bit RGB copies of the 16-bit PadChest PNGs
  (per-image min-max scaling), which is what the MS-CXR-trained detectors and
  RAD-DINO expect.
* ``protocol_<mapping>/padchest_gr_external/eval_inputs.jsonl`` and
  ``eval_labels.jsonl`` in the ClueGround protocol format (pixel xyxy gold).
* ``protocol_<mapping>/eval_padchest_gr.csv``: the same rows in the flat
  schema used by the baseline wrappers (``gold_boxes_json`` in pixels).
* ``protocol_<mapping>/PROTOCOL_SUMMARY.json``: counts per finding and per
  reference-box count, plus everything that was skipped and why.

The image archive is read directly from the split zip parts (no need to
concatenate or extract the 37 GB archive); pass ``--images-dir`` instead if
the PNGs are already extracted.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
PROTOCOL_KEY = "padchest_gr_external"
JPEG_QUALITY = 95


class SpanningReader(io.RawIOBase):
    """Seekable read-only view over ``PadChest_GR.zip.001 .. .037``."""

    def __init__(self, parts: list[Path]) -> None:
        self.parts = parts
        self.sizes = [p.stat().st_size for p in parts]
        self.offsets = [0]
        for size in self.sizes:
            self.offsets.append(self.offsets[-1] + size)
        self.handles = [p.open("rb") for p in parts]
        self.pos = 0

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.offsets[-1]}[whence]
        self.pos = base + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:  # type: ignore[override]
        want, got = len(buffer), 0
        total = self.offsets[-1]
        while got < want and self.pos < total:
            index = max(i for i in range(len(self.parts)) if self.offsets[i] <= self.pos)
            handle = self.handles[index]
            handle.seek(self.pos - self.offsets[index])
            chunk = handle.read(min(want - got, self.offsets[index + 1] - self.pos))
            if not chunk:
                break
            buffer[got : got + len(chunk)] = chunk
            got += len(chunk)
            self.pos += len(chunk)
        return got

    def close(self) -> None:
        for handle in self.handles:
            handle.close()
        super().close()


class ImageSource:
    def __init__(self, dataset_root: Path, images_dir: Path | None) -> None:
        self.images_dir = images_dir
        self.zip: zipfile.ZipFile | None = None
        self.members: dict[str, str] = {}
        if images_dir is None:
            parts = sorted((dataset_root / "Padchest_GR_files").glob("PadChest_GR.zip.*"))
            if not parts:
                raise FileNotFoundError(f"no PadChest_GR.zip.* parts under {dataset_root / 'Padchest_GR_files'}")
            self.zip = zipfile.ZipFile(io.BufferedReader(SpanningReader(parts), buffer_size=1 << 20))
            self.members = {os.path.basename(name): name for name in self.zip.namelist() if name.lower().endswith(".png")}

    def open(self, image_id: str) -> Image.Image:
        if self.images_dir is not None:
            candidates = [self.images_dir / image_id, *self.images_dir.rglob(image_id)]
            for path in candidates:
                if path.is_file():
                    return Image.open(path)
            raise FileNotFoundError(image_id)
        assert self.zip is not None
        return Image.open(io.BytesIO(self.zip.read(self.members[image_id])))


def to_uint8_rgb(image: Image.Image) -> Image.Image:
    array = np.asarray(image)
    if array.ndim == 3:
        return image.convert("RGB")
    array = array.astype(np.float32)
    low, high = float(array.min()), float(array.max())
    scaled = np.zeros_like(array, dtype=np.uint8) if high <= low else ((array - low) / (high - low) * 255.0).round().astype(np.uint8)
    return Image.fromarray(scaled, mode="L").convert("RGB")


def load_master(dataset_root: Path) -> pd.DataFrame:
    with zipfile.ZipFile(dataset_root / "master_table.csv.zip") as archive:
        name = [n for n in archive.namelist() if n.endswith(".csv")][0]
        return pd.read_csv(archive.open(name))


def build(args: argparse.Namespace) -> None:
    label_map = json.loads((HERE / "padchest_gr_label_map.json").read_text(encoding="utf-8"))
    mappings = {"strict": dict(label_map["strict"]), "extended": {**label_map["strict"], **label_map["extended_extra"]}}
    selected = ["strict", "extended"] if args.mapping == "both" else [args.mapping]

    master = load_master(args.dataset_root)
    test = master[master.split == args.split]
    image_meta = {str(r.ImageID): {"patient": str(r.PatientID), "study": str(r.StudyID)} for r in test.itertuples()}
    reports = json.loads((args.dataset_root / "grounded_reports_20240819.json").read_text(encoding="utf-8"))
    reports = [r for r in reports if str(r["ImageID"]) in image_meta]
    print(f"{args.split} studies with grounded reports: {len(reports)}")

    source = ImageSource(args.dataset_root, args.images_dir)
    image_dir = args.out_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    size_cache: dict[str, tuple[int, int]] = {}

    def image_path_and_size(image_id: str) -> tuple[Path, int, int]:
        target = image_dir / (Path(image_id).stem + ".jpg")
        if image_id not in size_cache:
            if not target.exists():
                with source.open(image_id) as raw:
                    rgb = to_uint8_rgb(raw)
                rgb.save(target, quality=JPEG_QUALITY)
            with Image.open(target) as img:
                size_cache[image_id] = img.size
        width, height = size_cache[image_id]
        return target, width, height

    for mapping_name in selected:
        mapping = mappings[mapping_name]
        proto_root = args.out_root / f"protocol_{mapping_name}"
        proto_dir = proto_root / PROTOCOL_KEY
        proto_dir.mkdir(parents=True, exist_ok=True)
        inputs, labels, flat = [], [], []
        skipped: Counter = Counter()
        per_image_index: dict[str, int] = defaultdict(int)
        for report in reports:
            image_id = str(report["ImageID"])
            meta = image_meta[image_id]
            for finding in report.get("findings", []):
                boxes = finding.get("boxes") or []
                if not boxes:
                    skipped["no_boxes"] += 1
                    continue
                padchest_labels = [str(l) for l in (finding.get("labels") or [])]
                mapped = sorted({mapping[l] for l in padchest_labels if l in mapping})
                if not mapped:
                    skipped["label_not_in_mapping"] += 1
                    continue
                if len(mapped) > 1:
                    skipped["ambiguous_multi_finding_sentence"] += 1
                    continue
                sentence = " ".join(str(finding.get("sentence_en", "")).split())
                if not sentence:
                    skipped["empty_sentence"] += 1
                    continue
                if args.limit and len(inputs) >= args.limit:
                    break
                path, width, height = image_path_and_size(image_id)
                gold = [[float(b[0]) * width, float(b[1]) * height, float(b[2]) * width, float(b[3]) * height] for b in boxes]
                index = per_image_index[image_id]
                per_image_index[image_id] += 1
                gid = f"{PROTOCOL_KEY}:{Path(image_id).stem}:{index:02d}"
                dicom_id = Path(image_id).stem
                inputs.append(
                    {
                        "dicom_id": dicom_id,
                        "finding": mapped[0],
                        "group_id": gid,
                        "image_height": int(height),
                        "image_path": str(path.resolve()),
                        "image_width": int(width),
                        "output_mode": "variable_set",
                        "protocol_key": PROTOCOL_KEY,
                        "query_lineage": {"mode": "finding_category_plus_raw_phrase", "source_fields": ["PadChest-GR label (mapped)", "sentence_en"], "target_reference_used": False},
                        "query_text": sentence,
                        "source_dataset": "PadChest-GR",
                        "split": "eval",
                        "study_id": meta["study"],
                        "subject_id": meta["patient"],
                        "padchest_labels": padchest_labels,
                        "padchest_mapped_from": [l for l in padchest_labels if l in mapping],
                        "label_mapping": mapping_name,
                    }
                )
                labels.append({"gold_boxes_xyxy": gold, "gold_count": len(gold), "group_id": gid, "protocol_key": PROTOCOL_KEY, "split": "eval", "target_semantics": "PadChest-GR sentence-level boxes (primary annotation)"})
                flat.append(
                    {
                        "group_id": gid,
                        "matrix_group_id": f"{dicom_id}|{sentence}",
                        "subject_id": meta["patient"],
                        "study_id": meta["study"],
                        "dicom_id": dicom_id,
                        "image_path": str(path.resolve()),
                        "image_width": int(width),
                        "image_height": int(height),
                        "finding": mapped[0],
                        "phrase": sentence,
                        "gold_boxes_json": json.dumps(gold),
                        "gold_count": len(gold),
                        "padchest_labels": "; ".join(padchest_labels),
                    }
                )
        with (proto_dir / "eval_inputs.jsonl").open("w", encoding="utf-8") as handle:
            for row in inputs:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        with (proto_dir / "eval_labels.jsonl").open("w", encoding="utf-8") as handle:
            for row in labels:
                handle.write(json.dumps(row) + "\n")
        pd.DataFrame(flat).to_csv(proto_root / "eval_padchest_gr.csv", index=False)
        by_finding = Counter(r["finding"] for r in inputs)
        by_count = Counter(l["gold_count"] for l in labels)
        summary = {
            "dataset": "PadChest-GR (grounded_reports_20240819.json)",
            "split": args.split,
            "label_mapping": mapping_name,
            "n_rows": len(inputs),
            "n_images": len({r["dicom_id"] for r in inputs}),
            "n_patients": len({r["subject_id"] for r in inputs}),
            "rows_per_finding": dict(sorted(by_finding.items())),
            "rows_per_gold_count": {str(k): v for k, v in sorted(by_count.items())},
            "skipped": dict(skipped),
            "image_conversion": "16-bit PNG -> per-image min-max scaled 8-bit RGB JPEG (quality 95)",
            "box_convention": "PadChest-GR normalized [x1, y1, x2, y2] scaled to pixels of the converted image",
            "mapping": mapping,
        }
        (proto_root / "PROTOCOL_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({k: summary[k] for k in ("label_mapping", "n_rows", "n_images", "n_patients", "rows_per_finding", "rows_per_gold_count", "skipped")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, required=True, help="folder holding grounded_reports_20240819.json, master_table.csv.zip, Padchest_GR_files/")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, default=None, help="already-extracted PNG folder (optional; default reads the split zip directly)")
    parser.add_argument("--mapping", choices=["strict", "extended", "both"], default="both")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=0, help="debug: stop after N rows")
    build(parser.parse_args())


if __name__ == "__main__":
    main()

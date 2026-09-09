from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Literal

import pandas as pd

from .contracts import PROTOCOLS, SPLITS, ProtocolSpec
from .guards import assert_no_identity_overlap
from .queries import build_model_query, normalize_text, stable_hash


SplitPolicy = Literal["task_isolated", "global_patient_disjoint"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _box_key(box: list[float]) -> tuple[float, float, float, float]:
    values = tuple(round(float(value), 4) for value in box)
    if len(values) != 4 or values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError(f"Invalid xyxy box: {box}")
    return values


def _source_group_key(spec: ProtocolSpec, row: dict[str, Any], query_text: str) -> str:
    if spec.key == "mscxr_multibox_1444":
        return f"{spec.key}|{row['dicom_id']}|{normalize_text(query_text)}"
    # Anatomy/device protocols are row-level one-box evaluations by definition.
    return f"{spec.key}|{row['task_id']}"


def _canonicalize_split(spec: ProtocolSpec, split: str) -> list[dict[str, Any]]:
    source_rows = read_jsonl(spec.source_path(split))
    groups: dict[str, dict[str, Any]] = {}
    for row in source_rows:
        query_text, lineage = build_model_query(spec, row)
        source_group_key = _source_group_key(spec, row, query_text)
        group_id = f"{spec.key}:{stable_hash(source_group_key)}"
        source_boxes = row.get("gold_group_bboxes_xyxy") if spec.key == "mscxr_multibox_1444" else None
        source_boxes = source_boxes or [row["gold_bbox_xyxy"]]
        boxes = {_box_key(box): list(_box_key(box)) for box in source_boxes}
        if group_id not in groups:
            groups[group_id] = {
                "protocol_key": spec.key,
                "group_id": group_id,
                "split": split,
                "source_dataset": spec.source_dataset,
                "image_path": str(row["image_path"]),
                "dicom_id": str(row["dicom_id"]),
                "study_id": str(row["study_id"]),
                "subject_id": str(row["subject_id"]),
                "image_width": int(row["image_width"]),
                "image_height": int(row["image_height"]),
                "query_text": query_text,
                "finding": str(row.get("finding", "")),
                "query_lineage": lineage,
                "target_semantics": spec.target_semantics,
                "output_mode": spec.output_mode,
                "boxes": boxes,
                "source_task_ids": {str(row["task_id"])},
                "debug": {
                    "source_finding": str(row.get("finding", "")),
                    "bbox_name_reference": str(row.get("bbox_name_reference", "")),
                    "object_name_reference": str(row.get("object_name_reference", "")),
                    "bbox_type": str(row.get("bbox_type", "")),
                    "gold_source": str(row.get("gold_source", "")),
                    "is_weak_reference": bool(row.get("is_weak_reference", False)),
                },
            }
        else:
            groups[group_id]["boxes"].update(boxes)
            groups[group_id]["source_task_ids"].add(str(row["task_id"]))

    output: list[dict[str, Any]] = []
    for group_id in sorted(groups):
        group = groups[group_id]
        group["gold_boxes_xyxy"] = sorted(group.pop("boxes").values())
        group["gold_count"] = len(group["gold_boxes_xyxy"])
        group["source_task_ids"] = sorted(group["source_task_ids"])
        output.append(group)
    return output


def _apply_global_patient_policy(
    rows_by_protocol: dict[str, dict[str, list[dict[str, Any]]]],
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    eval_subjects = {
        row["subject_id"]
        for splits in rows_by_protocol.values()
        for row in splits["eval"]
    }
    raw_val_subjects = {
        row["subject_id"]
        for splits in rows_by_protocol.values()
        for row in splits["val"]
    }
    val_subjects = raw_val_subjects - eval_subjects
    filtered: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    drops: list[dict[str, Any]] = []
    for protocol_key, splits in rows_by_protocol.items():
        for split, rows in splits.items():
            if split == "train":
                kept = [row for row in rows if row["subject_id"] not in eval_subjects | val_subjects]
            elif split == "val":
                kept = [row for row in rows if row["subject_id"] not in eval_subjects]
            else:
                kept = list(rows)
            filtered[protocol_key][split] = kept
            drops.append(
                {
                    "protocol_key": protocol_key,
                    "split": split,
                    "before": len(rows),
                    "after": len(kept),
                    "dropped": len(rows) - len(kept),
                }
            )
    return dict(filtered), {
        "eval_subject_union": len(eval_subjects),
        "val_subject_union_after_eval_precedence": len(val_subjects),
        "drops": drops,
        "policy": "eval > val > train precedence; conflicting train/val rows are dropped, never reassigned",
    }


def _input_row(group: dict[str, Any]) -> dict[str, Any]:
    return {
        key: group[key]
        for key in (
            "protocol_key",
            "group_id",
            "split",
            "source_dataset",
            "image_path",
            "dicom_id",
            "study_id",
            "subject_id",
            "image_width",
            "image_height",
            "query_text",
            "finding",
            "query_lineage",
            "target_semantics",
            "output_mode",
        )
    }


def _label_row(group: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol_key": group["protocol_key"],
        "group_id": group["group_id"],
        "split": group["split"],
        "gold_boxes_xyxy": group["gold_boxes_xyxy"],
        "gold_count": group["gold_count"],
        "source_task_ids": group["source_task_ids"],
        "target_semantics": group["target_semantics"],
    }


def build_protocol_bundle(
    output_root: Path,
    *,
    split_policy: SplitPolicy = "task_isolated",
    check_images: bool = True,
    protocol_keys: Iterable[str] | None = None,
) -> dict[str, Any]:
    if split_policy not in {"task_isolated", "global_patient_disjoint"}:
        raise ValueError(split_policy)
    selected_keys = list(protocol_keys) if protocol_keys is not None else list(PROTOCOLS)
    unknown = sorted(set(selected_keys) - set(PROTOCOLS))
    if unknown:
        raise KeyError(f"Unknown protocols: {unknown}")
    if not selected_keys:
        raise ValueError("At least one protocol must be selected")
    selected_protocols = {key: PROTOCOLS[key] for key in selected_keys}
    rows_by_protocol = {
        key: {split: _canonicalize_split(spec, split) for split in SPLITS}
        for key, spec in selected_protocols.items()
    }
    policy_audit: dict[str, Any] = {"policy": split_policy, "drops": []}
    if split_policy == "global_patient_disjoint":
        rows_by_protocol, policy_audit = _apply_global_patient_policy(rows_by_protocol)

    output_root.mkdir(parents=True, exist_ok=True)
    protocol_audits: dict[str, Any] = {}
    combined_inputs: list[dict[str, Any]] = []
    combined_labels: list[dict[str, Any]] = []
    combined_debug: list[dict[str, Any]] = []
    for key, spec in selected_protocols.items():
        protocol_root = output_root / key
        protocol_root.mkdir(parents=True, exist_ok=True)
        split_frames: dict[str, pd.DataFrame] = {}
        count_rows: list[dict[str, Any]] = []
        for split in SPLITS:
            groups = rows_by_protocol[key][split]
            inputs = [_input_row(group) for group in groups]
            labels = [_label_row(group) for group in groups]
            debug = [
                {
                    "protocol_key": key,
                    "group_id": group["group_id"],
                    "split": split,
                    **group["debug"],
                }
                for group in groups
            ]
            write_jsonl(protocol_root / f"{split}_inputs.jsonl", inputs)
            write_jsonl(protocol_root / f"{split}_labels.jsonl", labels)
            write_jsonl(protocol_root / f"{split}_source_debug.jsonl", debug)
            split_frames[split] = pd.DataFrame(inputs)
            combined_inputs.extend(inputs)
            combined_labels.extend(labels)
            combined_debug.extend(debug)
            missing_images = sum(not Path(row["image_path"]).is_file() for row in inputs) if check_images else None
            count_rows.append(
                {
                    "split": split,
                    "groups": len(groups),
                    "gold_boxes": sum(int(label["gold_count"]) for label in labels),
                    "subjects": len({row["subject_id"] for row in inputs}),
                    "studies": len({row["study_id"] for row in inputs}),
                    "dicoms": len({row["dicom_id"] for row in inputs}),
                    "missing_images": missing_images,
                }
            )
        pd.DataFrame(count_rows).to_csv(protocol_root / "counts.csv", index=False)
        overlap = [
            assert_no_identity_overlap(split_frames[left], split_frames[right], left_name=left, right_name=right)
            for left, right in (("train", "val"), ("train", "eval"), ("val", "eval"))
        ]
        expected_counts_enforced = split_policy == "task_isolated"
        expected_pass = all(
            len(rows_by_protocol[key][split]) == spec.expected_groups[split]
            for split in SPLITS
        ) if expected_counts_enforced else True
        audit = {
            "protocol": spec.to_dict(),
            "split_policy": split_policy,
            "counts": count_rows,
            "expected_counts_enforced": expected_counts_enforced,
            "expected_counts_pass": expected_pass,
            "within_protocol_overlap": overlap,
            "source_hashes": {split: file_sha256(spec.source_path(split)) for split in SPLITS},
            "status": "PASS" if expected_pass and all(row["status"] == "PASS" for row in overlap) else "FAIL",
        }
        (protocol_root / "protocol_audit.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        protocol_audits[key] = audit

    write_jsonl(output_root / "all_inputs.jsonl", combined_inputs)
    write_jsonl(output_root / "all_labels.jsonl", combined_labels)
    write_jsonl(output_root / "all_source_debug.jsonl", combined_debug)

    cross_task = []
    for left_split, right_split in (("train", "val"), ("train", "eval"), ("val", "eval")):
        for left_key, left_splits in rows_by_protocol.items():
            left_subjects = {row["subject_id"] for row in left_splits[left_split]}
            for right_key, right_splits in rows_by_protocol.items():
                right_subjects = {row["subject_id"] for row in right_splits[right_split]}
                cross_task.append(
                    {
                        "left_protocol": left_key,
                        "left_split": left_split,
                        "right_protocol": right_key,
                        "right_split": right_split,
                        "patient_overlap": len(left_subjects & right_subjects),
                        "allowed": split_policy == "task_isolated" and left_key != right_key,
                        "interpretation": (
                            "reported but not shared because task-specific models are isolated"
                            if split_policy == "task_isolated" and left_key != right_key
                            else "must be zero"
                        ),
                    }
                )
    cross_task_frame = pd.DataFrame(cross_task)
    cross_task_frame.to_csv(output_root / "cross_task_patient_overlap.csv", index=False)
    pooled_overlap_pass = split_policy == "task_isolated" or bool((cross_task_frame["patient_overlap"] == 0).all())
    summary = {
        "status": "PASS" if all(audit["status"] == "PASS" for audit in protocol_audits.values()) and pooled_overlap_pass else "FAIL",
        "split_policy": split_policy,
        "protocols": selected_keys,
        "policy_audit": policy_audit,
        "pooled_overlap_pass": pooled_overlap_pass,
        "main_claim_guard": (
            "Only mscxr_multibox_1444/task_isolated is eligible for the CIG-free MS-CXR task-supervision claim. "
            "ImaGenome target-task runs and pooled runs are separate experiments."
        ),
    }
    (output_root / "bundle_audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import pandas as pd


FORBIDDEN_FEATURE_PATTERN = re.compile(
    r"(^|_)(target|iou|label|positive|weak_positive|gold|gt|oracle|source|answer|matched)(_|$)",
    re.IGNORECASE,
)

BASE_RANKER_FEATURES = (
    "confidence",
    "rank_norm",
    "dino_agreement",
    "cross_model_agreement",
    "context_score",
    "box_cx_norm",
    "box_cy_norm",
    "box_w_norm",
    "box_h_norm",
    "box_area_norm",
)

RAW_ONLY_COLUMNS = {
    "candidate_source",
    "source_model",
    "bbox_name_reference",
    "object_name_reference",
    "target_iou",
    "gold_boxes_xyxy",
    "gold_count",
    "source_task_ids",
}


def validate_feature_columns(columns: Iterable[str]) -> list[str]:
    values = [str(column) for column in columns]
    forbidden = [column for column in values if FORBIDDEN_FEATURE_PATTERN.search(column)]
    if forbidden:
        raise ValueError(f"Forbidden model features: {forbidden}")
    return values


def validate_base_ranker_schema(columns: Iterable[str]) -> list[str]:
    values = validate_feature_columns(columns)
    expected = list(BASE_RANKER_FEATURES)
    if values != expected:
        raise ValueError(f"Base-ranker feature schema mismatch: expected {expected}, got {values}")
    return values


def inference_candidate_view(frame: pd.DataFrame) -> pd.DataFrame:
    drop = [column for column in frame.columns if column in RAW_ONLY_COLUMNS or FORBIDDEN_FEATURE_PATTERN.search(column)]
    return frame.drop(columns=drop, errors="ignore").copy()


def assert_inference_table(frame: pd.DataFrame, *, allow_anatomy_query: bool = True) -> dict[str, Any]:
    forbidden = [column for column in frame.columns if FORBIDDEN_FEATURE_PATTERN.search(column)]
    raw_only = [column for column in frame.columns if column in RAW_ONLY_COLUMNS]
    annotation_columns = [column for column in ("category_name", "bbox_name_reference", "object_name_reference") if column in frame]
    status = "PASS" if not forbidden and not raw_only and not annotation_columns else "FAIL"
    report = {
        "status": status,
        "forbidden_columns": forbidden,
        "raw_only_columns": raw_only,
        "annotation_columns": annotation_columns,
        "anatomy_query_allowed": allow_anatomy_query,
        "note": "query_text and the explicit MS-CXR finding query are allowed; hidden target-reference columns are not",
    }
    if status != "PASS":
        raise RuntimeError(f"Inference-table contract failed: {report}")
    return report


def assert_no_identity_overlap(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_name: str,
    right_name: str,
) -> dict[str, Any]:
    fields = ("subject_id", "study_id", "dicom_id", "group_id")
    overlap = {
        field: len(set(left[field].astype(str)) & set(right[field].astype(str)))
        for field in fields
        if field in left and field in right
    }
    report = {
        "left": left_name,
        "right": right_name,
        "overlap": overlap,
        "status": "PASS" if all(value == 0 for value in overlap.values()) else "FAIL",
    }
    return report

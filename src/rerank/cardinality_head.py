from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from .action_policy import query_level_features
from .box_features import iou_xyxy


@dataclass
class CountHeadBundle:
    model: object
    feature_columns: list[str]

    def save(self, path) -> None:
        joblib.dump({"model": self.model, "feature_columns": self.feature_columns}, path)


def fit_count_head(
    groups: dict[str, dict[str, Any]],
    base_preds: dict[str, list[dict[str, Any]]],
    candidates: pd.DataFrame,
    cue_info: dict[str, dict[str, Any]],
    *,
    score_col: str,
) -> CountHeadBundle:
    feat = query_level_features(groups, base_preds, candidates, cue_info, score_col=score_col)
    y_rows = [{"group_id": str(gid), "count_label": min(max(len(g.get("gt_boxes", [])), 1), 3)} for gid, g in groups.items()]
    data = feat.merge(pd.DataFrame(y_rows), on="group_id", how="inner")
    feature_columns = [c for c in data.columns if c not in {"group_id", "count_label"}]
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=2026,
            n_jobs=-1,
        ),
    )
    model.fit(data[feature_columns], data["count_label"].astype(int))
    return CountHeadBundle(model=model, feature_columns=feature_columns)


def predict_counts(
    bundle: CountHeadBundle,
    groups: dict[str, dict[str, Any]],
    base_preds: dict[str, list[dict[str, Any]]],
    candidates: pd.DataFrame,
    cue_info: dict[str, dict[str, Any]],
    *,
    score_col: str,
) -> pd.DataFrame:
    feat = query_level_features(groups, base_preds, candidates, cue_info, score_col=score_col)
    x = feat.reindex(columns=bundle.feature_columns, fill_value=0.0)
    pred = bundle.model.predict(x).astype(int)
    out = feat[["group_id"]].copy()
    out["predicted_count"] = np.clip(pred, 1, 3)
    probs = bundle.model.predict_proba(x)
    for idx, cls in enumerate(bundle.model[-1].classes_):
        out[f"count_prob_{int(cls)}"] = probs[:, idx]
    out["count_confidence"] = probs.max(axis=1)
    return out


def _row_to_pred(row: pd.Series, score_col: str, source: str) -> dict[str, Any]:
    return {
        "box": [float(row["pred_x1"]), float(row["pred_y1"]), float(row["pred_x2"]), float(row["pred_y2"])],
        "score": float(row.get(score_col, 0.0)),
        "source": source,
    }


def select_by_predicted_count(
    candidates: pd.DataFrame,
    count_outputs: pd.DataFrame,
    *,
    score_col: str,
    nms_iou: float = 0.45,
    score_ratio: float = 0.72,
) -> tuple[dict[str, list[dict[str, Any]]], pd.DataFrame]:
    count_map = dict(zip(count_outputs["group_id"].astype(str), count_outputs["predicted_count"].astype(int)))
    out: dict[str, list[dict[str, Any]]] = {}
    audit: list[dict[str, Any]] = []
    for gid, part in candidates.groupby("query_id", sort=False):
        gid = str(gid)
        k = int(count_map.get(gid, 1))
        part = part.sort_values(score_col, ascending=False)
        selected: list[dict[str, Any]] = []
        top_score = float(part.iloc[0].get(score_col, 0.0)) if len(part) else 0.0
        for _, row in part.iterrows():
            if len(selected) >= k:
                break
            if top_score > 0 and float(row.get(score_col, 0.0)) < top_score * score_ratio:
                continue
            box = [float(row["pred_x1"]), float(row["pred_y1"]), float(row["pred_x2"]), float(row["pred_y2"])]
            if all(iou_xyxy(box, old["box"]) < nms_iou for old in selected):
                selected.append(_row_to_pred(row, score_col, "count_head"))
        if not selected and len(part):
            selected = [_row_to_pred(part.iloc[0], score_col, "count_head_fallback")]
        out[gid] = selected
        audit.append({"group_id": gid, "predicted_count": k, "new_pred_count": len(selected), "action_taken": "count_select"})
    return out, pd.DataFrame(audit)


def apply_count_to_base(
    base_preds: dict[str, list[dict[str, Any]]],
    candidates: pd.DataFrame,
    count_outputs: pd.DataFrame,
    *,
    score_col: str,
    nms_iou: float = 0.45,
) -> tuple[dict[str, list[dict[str, Any]]], pd.DataFrame]:
    cand_preds, cand_audit = select_by_predicted_count(candidates, count_outputs, score_col=score_col, nms_iou=nms_iou)
    count_map = dict(zip(count_outputs["group_id"].astype(str), count_outputs["predicted_count"].astype(int)))
    out: dict[str, list[dict[str, Any]]] = {}
    audit: list[dict[str, Any]] = []
    for gid, k in count_map.items():
        base = list(base_preds.get(gid, []))
        if len(base) == int(k):
            preds = base
            act = "keep_base_count_match"
        elif len(base) > int(k):
            preds = base[: int(k)]
            act = "prune_to_count"
        else:
            preds = list(base)
            for cand in cand_preds.get(gid, []):
                if len(preds) >= int(k):
                    break
                if all(iou_xyxy(cand["box"], old["box"]) < nms_iou for old in preds):
                    preds.append(cand)
            act = "add_to_count" if len(preds) > len(base) else "keep_base_no_add"
        out[gid] = [{**p, "source": f"count_head::{act}"} for p in preds]
        audit.append({"group_id": gid, "predicted_count": int(k), "base_pred_count": len(base), "new_pred_count": len(preds), "action_taken": act})
    # Preserve any groups that did not appear in the candidate table.
    for gid, preds in base_preds.items():
        if gid not in out:
            out[gid] = preds
            audit.append({"group_id": gid, "predicted_count": len(preds), "base_pred_count": len(preds), "new_pred_count": len(preds), "action_taken": "keep_base_missing_candidate"})
    return out, pd.DataFrame(audit)

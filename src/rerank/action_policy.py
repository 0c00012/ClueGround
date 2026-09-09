from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from .base_preserving_multibox_rescue import predict_base_preserving


ACTION_VARIANT = {
    "keep_base": None,
    "replace": "keep_or_replace_only",
    "add": "add_missing_side_only",
    "prune": "prune_overprediction",
}


DEFAULT_ACTION_PARAMS: dict[str, Any] = {
    "candidate_topn": 30,
    "base_default_score": 0.45,
    "same_box_iou": 0.82,
    "replace_scan_topn": 20,
    "replace_margin": 0.18,
    "replace_ratio": 1.20,
    "replace_abs_threshold": 0.64,
    "single_union_expansion_threshold": 1.20,
    "add_scan_topn": 30,
    "add_abs_threshold": 0.68,
    "add_score_ratio": 0.88,
    "duplicate_iou_threshold": 0.45,
    "multi_union_expansion_threshold": 1.70,
    "diffuse_union_expansion_threshold": 2.20,
    "target_weight": 0.18,
    "target_match_min": 0.50,
}


@dataclass
class ActionPolicyBundle:
    model: object
    feature_columns: list[str]
    class_order: list[str]

    def save(self, path) -> None:
        joblib.dump(
            {
                "model": self.model,
                "feature_columns": self.feature_columns,
                "class_order": self.class_order,
            },
            path,
        )


def _entropy(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 0.0
    vals = vals - vals.min()
    if float(vals.sum()) <= 1e-8:
        return 0.0
    p = vals / vals.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


def query_level_features(
    groups: dict[str, dict[str, Any]],
    base_preds: dict[str, list[dict[str, Any]]],
    candidates: pd.DataFrame,
    cue_info: dict[str, dict[str, Any]],
    *,
    score_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = {str(k): v for k, v in candidates.groupby("query_id", sort=False)}
    for gid, group in groups.items():
        gid = str(gid)
        part = grouped.get(gid, pd.DataFrame())
        scores = part[score_col].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).sort_values(ascending=False).to_numpy() if not part.empty else np.zeros(0)
        top = float(scores[0]) if len(scores) else 0.0
        second = float(scores[1]) if len(scores) > 1 else 0.0
        cue = cue_info.get(gid, {})
        preds = base_preds.get(gid, [])
        row: dict[str, Any] = {
            "group_id": gid,
            "finding": str(group.get("finding", "")),
            "base_pred_count": len(preds),
            "has_multi_cue": float(bool(cue.get("has_multi_cue", False))),
            "k_hint": float(cue.get("k_hint", 1) or 1),
            "candidate_count": int(len(part)),
            "top_score": top,
            "second_score": second,
            "score_gap": top - second,
            "score_ratio_2_to_1": second / top if top > 1e-8 else 0.0,
            "score_entropy": _entropy(scores[:30]),
            "top_dino_reliability": float(part["dino_reliability"].max()) if "dino_reliability" in part and len(part) else 0.0,
            "top_dino_agreement": float(part["dino_reliable_agreement"].max()) if "dino_reliable_agreement" in part and len(part) else 0.0,
            "top_rule_context_score": float(part["rule_context_score"].max()) if "rule_context_score" in part and len(part) else 0.0,
            "top_prior_score": float(part["train_prior_score"].max()) if "train_prior_score" in part and len(part) else 0.0,
            "top_contrastive_score": float(part["contrastive_score"].max()) if "contrastive_score" in part and len(part) else 0.0,
            "multi_cue_type": str(cue.get("multi_cue_type", "none")),
        }
        rows.append(row)
    frame = pd.DataFrame(rows)
    return pd.get_dummies(frame, columns=["finding", "multi_cue_type"], dummy_na=False)


def fit_action_policy(features: pd.DataFrame, labels: pd.DataFrame) -> ActionPolicyBundle:
    data = features.merge(labels[["group_id", "policy_label"]], on="group_id", how="inner")
    if data.empty:
        raise RuntimeError("No action-policy training rows were available")
    feature_columns = [c for c in data.columns if c not in {"group_id", "policy_label"}]
    x = data[feature_columns]
    y = data["policy_label"].astype(str)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=320,
            min_samples_leaf=6,
            class_weight="balanced_subsample",
            random_state=2026,
            n_jobs=-1,
        ),
    )
    model.fit(x, y)
    return ActionPolicyBundle(model=model, feature_columns=feature_columns, class_order=sorted(y.unique().tolist()))


def predict_action_labels(bundle: ActionPolicyBundle, features: pd.DataFrame) -> pd.DataFrame:
    x = features.reindex(columns=bundle.feature_columns, fill_value=0.0)
    pred = bundle.model.predict(x)
    out = features[["group_id"]].copy()
    out["predicted_action"] = pred.astype(str)
    if hasattr(bundle.model[-1], "predict_proba"):
        probs = bundle.model.predict_proba(x)
        cls = [str(c) for c in bundle.model[-1].classes_]
        for idx, name in enumerate(cls):
            out[f"action_prob_{name}"] = probs[:, idx]
        out["predicted_action_confidence"] = probs.max(axis=1)
    else:
        out["predicted_action_confidence"] = 1.0
    return out


def apply_action_policy(
    action_outputs: pd.DataFrame,
    base_preds: dict[str, list[dict[str, Any]]],
    candidates: pd.DataFrame,
    cue_info: dict[str, dict[str, Any]],
    groups: dict[str, dict[str, Any]],
    *,
    score_col: str,
    params: dict[str, Any] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], pd.DataFrame]:
    params = {**DEFAULT_ACTION_PARAMS, **(params or {})}
    action_map = dict(zip(action_outputs["group_id"].astype(str), action_outputs["predicted_action"].astype(str)))
    pred_by_variant: dict[str, dict[str, list[dict[str, Any]]]] = {}
    audit_by_variant: dict[str, pd.DataFrame] = {}
    for action, variant in ACTION_VARIANT.items():
        if variant is None:
            continue
        pred_by_variant[action], audit_by_variant[action] = predict_base_preserving(
            base_preds,
            candidates,
            cue_info,
            groups,
            params,
            score_col=score_col,
            variant=variant,
        )

    out: dict[str, list[dict[str, Any]]] = {}
    audit_rows: list[dict[str, Any]] = []
    for gid, group in groups.items():
        gid = str(gid)
        action = action_map.get(gid, "keep_base")
        if action not in ACTION_VARIANT:
            action = "keep_base"
        if action == "keep_base":
            preds = base_preds.get(gid, [])
            action_taken = "keep_base"
        else:
            preds = pred_by_variant.get(action, {}).get(gid, base_preds.get(gid, []))
            audit_frame = audit_by_variant.get(action, pd.DataFrame())
            row = audit_frame[audit_frame["group_id"].astype(str).eq(gid)]
            action_taken = str(row.iloc[0]["action_taken"]) if len(row) else action
            if action_taken == "keep_base":
                preds = base_preds.get(gid, [])
        out[gid] = [
            {**p, "source": f"action_policy::{action_taken}"}
            for p in preds
        ]
        audit_rows.append(
            {
                "group_id": gid,
                "predicted_action": action,
                "action_taken": action_taken,
                "base_pred_count": len(base_preds.get(gid, [])),
                "new_pred_count": len(out[gid]),
                "finding": group.get("finding", ""),
            }
        )
    return out, pd.DataFrame(audit_rows)

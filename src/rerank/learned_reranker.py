from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASE_FEATURES = [
    "confidence",
    "candidate_rank",
    "rank_bonus",
    "rank_norm",
    "n_candidates_norm",
    "score_head",
    "region_score",
    "prior_iou",
    "xattn_iou",
    "consensus_max",
    "consensus_mean",
    "dino_reliability",
    "dino_rule_conflict",
    "dino_reliable_agreement",
    "candidate_in_dino",
    "dino_in_candidate",
    "center_distance_to_dino",
    "siglip_z",
    "siglip_rank",
    "siglip_sigmoid",
    "biomedclip_z",
    "biomedclip_rank",
    "biomedclip_sigmoid",
    "contrastive_score",
    "cig_roi_score",
    "box_cx_norm",
    "box_cy_norm",
    "box_w_norm",
    "box_h_norm",
    "box_area_norm",
    "box_aspect",
    "lat_left",
    "lat_right",
    "lat_bilateral",
    "lat_none",
    "lat_unknown",
    "v_apical",
    "v_upper",
    "v_mid",
    "v_lower",
    "v_basal",
    "v_whole",
    "v_unknown",
]


@dataclass
class RerankerBundle:
    model: object
    feature_columns: list[str]
    model_name: str

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        x = df.reindex(columns=self.feature_columns, fill_value=0.0)
        return np.asarray(self.model.predict(x), dtype=float)

    def save(self, path) -> None:
        joblib.dump({"model": self.model, "feature_columns": self.feature_columns, "model_name": self.model_name}, path)


def available_features(df: pd.DataFrame, candidates: Iterable[str] = BASE_FEATURES) -> list[str]:
    return [c for c in candidates if c in df.columns]


def fit_reranker(train_df: pd.DataFrame, target_col: str, model_name: str = "hgb") -> RerankerBundle:
    features = available_features(train_df)
    if not features:
        raise RuntimeError("No usable reranker features were found")
    x = train_df.reindex(columns=features, fill_value=0.0)
    y = train_df[target_col].astype(float).clip(0.0, 1.0).to_numpy()
    if model_name == "rf":
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(n_estimators=240, min_samples_leaf=3, random_state=2026, n_jobs=-1),
        )
    else:
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            HistGradientBoostingRegressor(max_iter=260, learning_rate=0.045, l2_regularization=0.05, random_state=2026),
        )
    model.fit(x, y)
    return RerankerBundle(model=model, feature_columns=features, model_name=model_name)


def add_predictions(bundle: RerankerBundle, df: pd.DataFrame, out_col: str = "reranker_score") -> pd.DataFrame:
    out = df.copy()
    out[out_col] = bundle.predict(out)
    return out

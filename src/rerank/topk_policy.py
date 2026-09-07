from __future__ import annotations

import pandas as pd

from .box_features import iou_xyxy


def _take_top(df: pd.DataFrame, score_col: str, k: int) -> set:
    if score_col not in df.columns or k <= 0:
        return set()
    return set(df.sort_values(score_col, ascending=False).head(k).index)


def preserve_topk(
    df: pd.DataFrame,
    *,
    max_total: int = 25,
    k_hybrid: int = 5,
    k_yolo: int = 5,
    k_rule: int = 3,
    k_prior: int = 3,
    k_dino: int = 3,
    k_siglip: int = 3,
    k_biomedclip: int = 3,
    k_contrastive: int = 3,
    diversity_iou: float = 0.92,
) -> pd.DataFrame:
    """Keep a diverse candidate subset without looking at gold labels."""

    if df.empty:
        return df.copy()
    idx: set = set()
    idx |= _take_top(df, "existing_hybrid_score", k_hybrid)
    idx |= _take_top(df, "score_head", k_hybrid)
    idx |= _take_top(df, "confidence", k_yolo)
    idx |= _take_top(df, "rule_context_score", k_rule)
    idx |= _take_top(df, "region_score", k_rule)
    idx |= _take_top(df, "train_prior_score", k_prior)
    idx |= _take_top(df, "prior_iou", k_prior)
    idx |= _take_top(df, "dino_reliable_agreement", k_dino)
    idx |= _take_top(df, "xattn_iou", k_dino)
    idx |= _take_top(df, "siglip_z", k_siglip)
    idx |= _take_top(df, "biomedclip_z", k_biomedclip)
    idx |= _take_top(df, "contrastive_score", k_contrastive)
    idx |= _take_top(df, "score_gate", k_hybrid)

    pool = df.loc[list(idx)].copy() if idx else df.copy()
    score_col = "score_gate" if "score_gate" in pool.columns else ("score_head" if "score_head" in pool.columns else "confidence")
    pool = pool.sort_values(score_col, ascending=False)
    kept = []
    for _, row in pool.iterrows():
        box = [row["pred_x1"], row["pred_y1"], row["pred_x2"], row["pred_y2"]]
        if all(iou_xyxy(box, [old["pred_x1"], old["pred_y1"], old["pred_x2"], old["pred_y2"]]) < diversity_iou for old in kept):
            kept.append(row.to_dict())
        if len(kept) >= max_total:
            break
    if not kept:
        kept = [pool.iloc[0].to_dict()]
    return pd.DataFrame(kept)


def by_query_topk(df: pd.DataFrame, query_col: str = "query_id", **kwargs) -> pd.DataFrame:
    rows = []
    for _, part in df.groupby(query_col, sort=False):
        rows.append(preserve_topk(part, **kwargs))
    return pd.concat(rows, ignore_index=True) if rows else df.head(0).copy()


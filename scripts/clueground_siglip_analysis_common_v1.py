#!/usr/bin/env python
"""Common utilities for the controlled ClueGround SigLIP analysis.

The module never changes detector, RAD-DINO, or legacy experiment artifacts.
It evaluates alternative candidate ordering policies over the frozen, seed-paired
top-12 candidate tables produced by the controlled SigLIP experiment.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "clueground_siglip_analysis_v1"
HYBRID_ROOT = PROJECT_ROOT / "experiments" / "clueground_exact_hybrid_v4_route_specific_3seed_v3"
SIGLIP_1444_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "clueground_finding_moe_multibox_1444_yolo640_siglip_only_full_paired_3seed_v1"
)
SIGLIP_888_SOURCE_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "clueground_finding_moe_singlebox_888_full_upstream_3seed_v1"
)
SEEDS = (13, 42, 2026)

for path in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import run_clueground_exact_hybrid_v4_direct_3seed_v1 as exact  # noqa: E402
from scripts import run_clueground_finding_moe_singlebox_888_full_upstream_3seed_v1 as direct_sem  # noqa: E402
from scripts import run_clueground_siglip2_moe_multibox_1444_full_upstream_3seed_v1 as multi_upstream  # noqa: E402
from scripts import run_ms_cxr_multibox_rule_context_fusion_v4_yolov8l_pool as hybrid_v4  # noqa: E402
from scripts import run_ms_cxr_singlebox_fair_yolo_dino_fusion_v1 as single_fusion  # noqa: E402
from src.baseline_repro.evaluator import exact_union_iou, iou_matrix, maximum_set_f1  # noqa: E402


@dataclass
class Resources:
    protocol: str
    seed: int
    upstream: Any
    multi_yolo_params: dict[str, dict[str, Any]]
    set_params: dict[str, Any]
    candidates: dict[str, pd.DataFrame]

    @property
    def context(self) -> Any:
        return self.upstream.context


def ensure_dirs() -> None:
    for name in (
        "audit", "configs", "scripts", "predictions", "metrics", "figures",
        "tables", "qualitative", "logs", "report",
    ):
        (OUTPUT_ROOT / name).mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_path(protocol: str, seed: int, method: str, split: str) -> Path:
    return OUTPUT_ROOT / "predictions" / protocol / f"seed_{seed}" / f"{split}_{method}.json"


def save_prediction_map(path: Path, outputs: dict[str, list[list[float]]]) -> None:
    write_json(path, {str(k): [[float(x) for x in b] for b in v] for k, v in outputs.items()})


def load_prediction_map(path: Path) -> dict[str, list[list[float]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): [[float(x) for x in b] for b in v] for k, v in raw.items()}


def _upstream_args() -> argparse.Namespace:
    return argparse.Namespace(
        hybrid_root=HYBRID_ROOT,
        raw_yolo_candidate_root=None,
        max_candidates_per_task=12,
        generate_missing_candidates=False,
        image_size=640,
        detector_confidence=0.001,
        yolo_device="0",
        device="cuda",
        force_rad_predictions=False,
    )


def siglip_cache_root(protocol: str, seed: int) -> Path:
    if protocol == "1444":
        return SIGLIP_1444_ROOT / "semantic_experts" / f"source_seed_{seed}"
    source = SIGLIP_888_SOURCE_ROOT / "semantic_experts" / f"source_seed_{seed}"
    if all((source / f"{split}_siglip_scored_candidates.csv").exists() for split in ("train", "val", "eval")):
        return source
    return OUTPUT_ROOT / "predictions" / "siglip_cache_888" / f"source_seed_{seed}"


def load_resources(protocol: str, seed: int, *, require_scores: bool = True) -> Resources:
    args = _upstream_args()
    if protocol == "888":
        original = exact.load_single_context(seed)
        multi_params = copy.deepcopy(original.yolo_params)
        upstream = direct_sem.load_hybrid_upstream(seed, args)
    elif protocol == "1444":
        original = exact.load_multi_context(seed, canonical_v3=False)
        multi_params = copy.deepcopy(original.yolo_params)
        upstream = multi_upstream.load_hybrid_upstream(seed, args)
    else:
        raise ValueError(protocol)
    set_params = copy.deepcopy(upstream.context.provenance["selected_set_params"])
    cache = siglip_cache_root(protocol, seed)
    frames: dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "eval"):
        scored = cache / f"{split}_siglip_scored_candidates.csv"
        base = cache / f"{split}_semantic_base_candidates.csv"
        path = scored if scored.exists() else base
        if require_scores and not scored.exists():
            raise FileNotFoundError(scored)
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if require_scores and frame["siglip_raw"].isna().any():
            raise RuntimeError(f"Missing SigLIP scores: {scored}")
        frames[split] = frame
    return Resources(protocol, seed, upstream, multi_params, set_params, frames)


def candidate_key(row: Any) -> tuple[Any, ...]:
    get = row.get if hasattr(row, "get") else lambda k, d=None: getattr(row, k, d)
    return (
        round(float(get("pred_x1")), 4), round(float(get("pred_y1")), 4),
        round(float(get("pred_x2")), 4), round(float(get("pred_y2")), 4),
        str(get("source_model", "")), int(float(get("rank", -1))),
    )


def row_box(row: Any) -> list[float]:
    get = row.get if hasattr(row, "get") else lambda k, d=None: getattr(row, k, d)
    return [float(get("pred_x1")), float(get("pred_y1")), float(get("pred_x2")), float(get("pred_y2"))]


def _normalise(values: np.ndarray, mode: str) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "raw":
        return values
    if mode == "minmax":
        span = float(values.max() - values.min()) if len(values) else 0.0
        return (values - values.min()) / span if span > 1e-12 else np.zeros_like(values)
    if mode == "zscore":
        sd = float(values.std())
        return (values - values.mean()) / sd if sd > 1e-12 else np.zeros_like(values)
    if mode == "rank":
        if len(values) <= 1:
            return np.ones_like(values)
        ranks = pd.Series(values).rank(method="average", ascending=True).to_numpy(dtype=float)
        return (ranks - 1.0) / float(len(values) - 1)
    raise ValueError(mode)


def _candidate_dicts(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for row in frame.to_dict("records"):
        if str(row.get("source_model", "")) == "rad_dino":
            continue
        rows.append({
            "box": row_box(row),
            "score": float(row.get("confidence", 0.0)),
            "class_id": int(single_fusion.CLASS_TO_ID[str(row["finding"])]),
            "source_model": str(row.get("source_model", "")),
            "rank": int(float(row.get("rank", -1))),
            "_row": row,
        })
    return rows


def _semantic_values(scored: list[dict[str, Any]], mode: str) -> dict[tuple[Any, ...], float]:
    vals = np.asarray([float(c["_row"].get("siglip_raw", 0.0)) for c in scored], dtype=float)
    norm = _normalise(vals, mode)
    return {candidate_key(c["_row"]): float(v) for c, v in zip(scored, norm)}


def _rank_values(scored: list[dict[str, Any]], key: str) -> dict[tuple[Any, ...], float]:
    vals = np.asarray([float(c.get(key, 0.0)) for c in scored], dtype=float)
    norm = _normalise(vals, "rank")
    return {candidate_key(c["_row"]): float(v) for c, v in zip(scored, norm)}


def _modify_scores(
    scored: list[dict[str, Any]],
    *,
    mode: str,
    norm: str = "rank",
    weight: float = 0.0,
    alpha: float = 0.5,
    bottom_fraction: float = 0.0,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    if not scored:
        return []
    semantic = _semantic_values(scored, norm)
    rows = [dict(c) for c in scored]
    if mode == "siglip_only":
        for c in rows:
            c["score"] = semantic[candidate_key(c["_row"])]
    elif mode == "score_fusion":
        for c in rows:
            c["score"] = float(c["score"]) + float(weight) * semantic[candidate_key(c["_row"])]
    elif mode == "rank_interpolation":
        geometric = _rank_values(rows, "score")
        for c in rows:
            key = candidate_key(c["_row"])
            c["score"] = float(alpha) * geometric[key] + (1.0 - float(alpha)) * semantic[key]
    elif mode in {"hard_filter", "topk"}:
        rows.sort(key=lambda c: semantic[candidate_key(c["_row"])], reverse=True)
        if mode == "topk" and top_k is not None:
            rows = rows[: max(1, int(top_k))]
        elif mode == "hard_filter" and bottom_fraction > 0:
            keep = max(1, int(math.ceil(len(rows) * (1.0 - float(bottom_fraction)))))
            rows = rows[:keep]
    elif mode != "geometric":
        raise ValueError(mode)
    return sorted(rows, key=lambda c: (float(c["score"]), -int(c.get("rank", 9999))), reverse=True)


def _params(resources: Resources, finding: str, *, multi: bool, no_prior: bool) -> dict[str, Any]:
    source = resources.multi_yolo_params if multi else resources.context.yolo_params
    params = copy.deepcopy(source.get(finding, source.get("__global__", {})))
    if no_prior:
        params["w_prior"] = 0.0
        params["w_area"] = 0.0
        params["blend_yolo_weight"] = 1.0
    return params


def _single_final_box(resources: Resources, group: dict[str, Any], candidate: dict[str, Any], no_prior: bool) -> list[float]:
    params = _params(resources, str(group["finding"]), multi=False, no_prior=no_prior)
    fusion = resources.context.fusion_params.get(str(group["finding"]), resources.context.fusion_params.get("__global__", {}))
    row = hybrid_v4.group_row(group)
    query, target_prior = single_fusion.yv2.target_prior_for_row(row, resources.context.priors, params)
    iw, ih = float(group["image_width"]), float(group["image_height"])
    box_norm = single_fusion.ybase.xyxy_to_norm(candidate["box"], iw, ih)
    pre = single_fusion.ybase.blend_norm(box_norm, target_prior, float(params.get("blend_yolo_weight", 1.0)))
    dino = resources.context.dino[group["split"]].get(str(group["group_id"]))
    final = single_fusion.ybase.blend_norm(pre, dino, float(fusion.get("yolo_dino_blend", 1.0))) if dino is not None else pre
    return single_fusion.old_fusion.norm_to_xyxy(final, iw, ih)


def _score_geometric(
    resources: Resources,
    group: dict[str, Any],
    query: dict[str, str],
    rows: list[dict[str, Any]],
    *,
    multi: bool,
    no_prior: bool,
) -> list[dict[str, Any]]:
    params = _params(resources, str(group["finding"]), multi=multi, no_prior=no_prior)
    dino = resources.context.dino[group["split"]].get(str(group["group_id"]))
    weight = float(resources.set_params.get("dino_weight", 0.0))
    return hybrid_v4.score_candidates(group, query, rows, resources.context.priors, params, dino, weight)


def predict_from_candidates(
    resources: Resources,
    split: str,
    *,
    method: str,
    norm: str = "rank",
    weight: float = 0.0,
    alpha: float = 0.5,
    bottom_fraction: float = 0.0,
    top_k: int | None = None,
    no_prior: bool = False,
    force_one_box: bool = False,
    target_scores: pd.DataFrame | None = None,
) -> tuple[dict[str, list[list[float]]], pd.DataFrame]:
    groups = exact.make_groups(resources.context, split)
    frame = resources.candidates[split]
    by_group = {str(gid): part.copy() for gid, part in frame.groupby(frame["group_id"].astype(str), sort=False)}
    target_lookup: dict[tuple[str, int, tuple[Any, ...]], float] = {}
    if target_scores is not None and not target_scores.empty:
        for row in target_scores.to_dict("records"):
            target_lookup[(str(row["group_id"]), int(row["target_index"]), candidate_key(row))] = float(row["siglip_target_raw"])
    outputs: dict[str, list[list[float]]] = {}
    audits: list[dict[str, Any]] = []
    for gid, group in groups.items():
        group["group_id"] = gid
        group["split"] = split
        part = by_group.get(gid, pd.DataFrame())
        raw = _candidate_dicts(part) if not part.empty else []
        base_q = single_fusion.ybase.parse_rule_context(hybrid_v4.group_row(group))
        cue = hybrid_v4.context_cues(str(group["claim_sentence"]), str(group["finding"]), base_q)
        is_multi = bool(cue["has_multi_cue"])
        selected: list[dict[str, Any]] = []

        if method == "yolo_confidence":
            ranked = sorted(raw, key=lambda c: float(c["score"]), reverse=True)
            selected = ranked[:1]
        elif method == "siglip_only":
            selected = _modify_scores(raw, mode="siglip_only", norm=norm)[:1]
        elif not is_multi:
            geometric = _score_geometric(resources, group, base_q, raw, multi=False, no_prior=no_prior)
            ranked = _modify_scores(
                geometric, mode=method, norm=norm, weight=weight, alpha=alpha,
                bottom_fraction=bottom_fraction, top_k=top_k,
            )
            selected = ranked[:1]
        else:
            params = resources.set_params
            nms_iou = float(params["nms_iou"])
            max_k = max(int(cue["k_hint"]), int(params["min_k_if_cue"]))
            max_k = min(max_k, int(params["max_k_if_cue"]))
            for target_index, query in enumerate(cue["target_qs"]):
                geometric = _score_geometric(resources, group, query, raw, multi=True, no_prior=no_prior)
                if target_lookup:
                    conditioned = []
                    for candidate in geometric:
                        copied = dict(candidate)
                        source_row = dict(candidate["_row"])
                        value = target_lookup.get((gid, target_index, candidate_key(source_row)))
                        if value is not None:
                            source_row["siglip_raw"] = value
                        copied["_row"] = source_row
                        conditioned.append(copied)
                    geometric = conditioned
                ranked = _modify_scores(
                    geometric, mode=method, norm=norm, weight=weight, alpha=alpha,
                    bottom_fraction=bottom_fraction, top_k=top_k,
                )
                for candidate in ranked[: int(params["target_scan_topn"])]:
                    if hybrid_v4.add_if_distinct(selected, candidate, nms_iou):
                        break
            geometric = _score_geometric(resources, group, base_q, raw, multi=True, no_prior=no_prior)
            ranked = _modify_scores(
                geometric, mode=method, norm=norm, weight=weight, alpha=alpha,
                bottom_fraction=bottom_fraction, top_k=top_k,
            )
            top_score = float(ranked[0]["score"]) if ranked else 0.0
            min_extra = top_score * float(params["extra_score_ratio"]) if top_score > 0 else -1e9
            for candidate in ranked:
                if len(selected) >= max_k:
                    break
                if selected and float(candidate["score"]) < min_extra:
                    continue
                hybrid_v4.add_if_distinct(selected, candidate, nms_iou)
            if not selected and ranked:
                selected = ranked[:1]

        if not selected:
            fallback = resources.upstream.hybrid[split].get(gid, [])
            boxes = [[float(x) for x in row["box"]] for row in fallback]
            fallback_used = True
        elif is_multi or method in {"yolo_confidence", "siglip_only"}:
            boxes = [[float(x) for x in row["box"]] for row in selected]
            fallback_used = False
        else:
            boxes = [_single_final_box(resources, group, selected[0], no_prior)]
            fallback_used = False
        if force_one_box:
            boxes = boxes[:1]
        outputs[gid] = boxes
        audits.append({
            "protocol": resources.protocol,
            "seed": resources.seed,
            "split": split,
            "method": method,
            "group_id": gid,
            "finding": group["finding"],
            "phrase": group["claim_sentence"],
            "route": "explicit_multi" if is_multi else "single_region",
            "n_candidates": len(raw),
            "n_pred": len(boxes),
            "fallback_used": int(fallback_used),
            "target_conditioned_siglip": int(bool(target_lookup) and is_multi),
        })
    return outputs, pd.DataFrame(audits)


def source_outputs(resources: Resources, split: str) -> dict[str, list[list[float]]]:
    return {
        str(gid): [[float(x) for x in row["box"]] for row in rows]
        for gid, rows in resources.upstream.hybrid[split].items()
    }


def evaluate(resources: Resources, outputs: dict[str, list[list[float]]], split: str) -> tuple[dict[str, Any], pd.DataFrame]:
    context = resources.context
    by_input = {str(row["group_id"]): row for row in context.inputs[split]}
    detail: list[dict[str, Any]] = []
    for gid, gold_raw in context.labels[split].items():
        pred = [[float(x) for x in b] for b in outputs.get(str(gid), [])]
        if resources.protocol == "888":
            pred = pred[:1]
        gold = [[float(x) for x in b] for b in gold_raw]
        matrix = iou_matrix(pred, gold)
        top1 = float(matrix[0, 0]) if resources.protocol == "888" and matrix.size else 0.0
        coverage = float(matrix.max(axis=0).mean()) if len(pred) and len(gold) else 0.0
        source = by_input[str(gid)]
        detail.append({
            "group_id": str(gid), "subject_id": str(source.get("subject_id", "")),
            "study_id": str(source.get("study_id", "")), "dicom_id": str(source.get("dicom_id", "")),
            "finding": str(source.get("finding", "")),
            "phrase": str(source.get("claim_sentence") or source.get("query_text") or ""),
            "n_gt": len(gold), "n_pred": len(pred), "top1_iou": top1,
            "coverage_iou": coverage, "exact_union_iou": exact_union_iou(pred, gold),
            "set_f1_0_3": maximum_set_f1(pred, gold, 0.3),
            "set_f1_0_5": maximum_set_f1(pred, gold, 0.5),
            "missing_prediction": int(str(gid) not in outputs),
        })
    frame = pd.DataFrame(detail)
    summary: dict[str, Any] = {
        "n": len(frame), "n_gt_boxes": int(frame.n_gt.sum()),
        "n_missing_predictions": int(frame.missing_prediction.sum()),
        "mean_pred_count": float(frame.n_pred.mean()),
    }
    if resources.protocol == "888":
        summary.update({
            "mean_iou": float(frame.top1_iou.mean()),
            "hit_0_3": float((frame.top1_iou >= 0.3).mean()),
            "hit_0_5": float((frame.top1_iou >= 0.5).mean()),
        })
    else:
        summary.update({
            "coverage_iou": float(frame.coverage_iou.mean()),
            "exact_union_iou": float(frame.exact_union_iou.mean()),
            "set_f1_0_3": float(frame.set_f1_0_3.mean()),
            "set_f1_0_5": float(frame.set_f1_0_5.mean()),
        })
    return summary, frame


def primary_metric(protocol: str) -> str:
    return "mean_iou" if protocol == "888" else "coverage_iou"


def aggregate(per_seed: pd.DataFrame) -> pd.DataFrame:
    id_cols = [c for c in ("protocol", "method", "calibration") if c in per_seed.columns]
    numeric = [
        c for c in ("mean_iou", "hit_0_3", "hit_0_5", "coverage_iou", "exact_union_iou", "set_f1_0_3", "set_f1_0_5", "mean_pred_count")
        if c in per_seed.columns
    ]
    rows = []
    for keys, part in per_seed.groupby(id_cols, dropna=False, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(id_cols, keys))
        row["n_seeds"] = int(len(part))
        row["seeds"] = "/".join(str(int(v)) for v in part.seed.tolist())
        for metric in numeric:
            values = pd.to_numeric(part[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values.dropna()) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)

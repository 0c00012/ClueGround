#!/usr/bin/env python
"""Component and RAD-DINO query ablations for the direct-888 re-ranker path.

Mirror of ``run_clueground_reranker_ablations_v1`` for the single-region
protocol.  Positive control reproduces the sealed direct-888 re-ranker mean
IoU exactly with all downstream parameters frozen; query variants mask the
91-D RAD-DINO query at inference (checkpoint fixed) and change ``dino_iou``
and the fusion input only; ``yolo_only`` retrains the scorer without
``dino_iou`` and re-selects alpha/calibration on val87; ``rad_dino_only``
evaluates the RAD-DINO box alone.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_clueground_canonical_learned_reranker_3seed_v1 as rr  # noqa: E402
from scripts import run_clueground_direct888_learned_reranker_3seed_v1 as r888  # noqa: E402
from scripts import run_clueground_exact_final_ablation_v1 as fa  # noqa: E402
from scripts import run_clueground_exact_hybrid_v4_direct_3seed_v1 as exact  # noqa: E402
from scripts import run_ms_cxr_multibox_rule_context_fusion_v4_yolov8l_pool as hybrid_v4  # noqa: E402
from scripts import run_ms_cxr_singlebox_fair_yolo_dino_fusion_v1 as single_fusion  # noqa: E402

SEEDS = (13, 42, 2026)
SPLITS = ("train", "val", "eval")
DEFAULT_FULL = PROJECT_ROOT / "experiments" / "clueground_direct888_learned_reranker_3seed_v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "clueground_reranker_ablations_888_v1"
QUERY_VARIANTS = tuple(v for v in fa.QUERY_VARIANTS if v != "full_query")
METRICS = ("mean_iou", "Hit@0.3", "Hit@0.5")


def log(message: str) -> None:
    print(f"[ablation-888 {time.strftime('%H:%M:%S')}] {message}", flush=True)


def metric_row(result: dict[str, Any]) -> dict[str, float]:
    return {m: float(result.get(m, float("nan"))) for m in METRICS}


class FixedSetParams:
    def __init__(self, params: dict[str, Any], grid: pd.DataFrame) -> None:
        self.params, self.grid, self.original = params, grid, hybrid_v4.tune_set_params

    def __enter__(self) -> None:
        hybrid_v4.tune_set_params = lambda *args, **kwargs: (dict(self.params), self.grid.copy())

    def __exit__(self, *exc: Any) -> None:
        hybrid_v4.tune_set_params = self.original


def norm_to_pixels(norm_by_task: dict[str, np.ndarray], rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    out = {}
    for row in rows:
        tid = str(row["task_id"])
        if tid in norm_by_task:
            out[tid] = single_fusion.ybase.norm_to_xyxy(np.asarray(norm_by_task[tid], dtype=np.float32), float(row["image_width"]), float(row["image_height"]))
    return out


def run_seed(seed: int, full_root: Path, output_root: Path, device: torch.device) -> list[dict[str, Any]]:
    rr.FEATURE_EXCLUDE = set(rr.FEATURE_EXCLUDE) | {"xattn_iou", "aux_iou"}
    seed_root = full_root / f"seed_{seed}"
    status = json.loads((seed_root / "RUN_STATUS.json").read_text(encoding="utf-8"))
    alpha = float(status["learned_alpha"])
    sealed = float(status["mean_iou"])
    selection = json.loads((seed_root / "scorer_selection.json").read_text(encoding="utf-8"))
    cols = list(selection["features"])
    run_root = full_root / "learned" / "singlebox_888" / f"seed_{seed}"
    set_params = json.loads((run_root / "selected_set_params.json").read_text(encoding="utf-8"))
    set_grid = pd.read_csv(run_root / "multibox_v4_val_grid.csv")

    rows, labels = r888.load_rows()
    candidates = r888.load_candidates(seed)
    priors = single_fusion.ybase.make_train_priors(rows["train"])
    tables = {split: pd.read_csv(seed_root / f"candidates_{split}.csv") for split in SPLITS}
    sealed_eval = pd.read_csv(seed_root / "scored_eval.csv")
    model = rr.scorer_models(seed)[selection["selected_model"]]
    model.fit(tables["train"][cols].to_numpy(np.float32), tables["train"]["target_iou"].to_numpy(float))
    check = np.clip(model.predict(tables["eval"][cols].to_numpy(np.float32)), 0.0, 1.0)
    max_diff = float(np.abs(check - sealed_eval["learned"].to_numpy()).max())
    if max_diff > 1e-9:
        raise RuntimeError(f"seed {seed}: scorer retrain mismatch {max_diff}")
    log(f"seed {seed}: scorer reproduced (max diff {max_diff:.1e}); alpha={alpha}; sealed mean IoU {sealed:.6f}")

    context = exact.load_single_context(seed)

    def score_split(split: str, dino_px: dict[str, list[float]] | None) -> pd.DataFrame:
        if dino_px is None:
            frame = tables[split].copy()
        else:
            frame = rr.build_table(split, rows[split], labels[split], candidates[split], priors, {"dino": dino_px, "xattn": {}, "aux": {}})
        frame["learned"] = np.clip(model.predict(frame[cols].to_numpy(np.float32)), 0.0, 1.0)
        return frame

    def run_fixed(name: str, dino_norm: dict[str, dict[str, np.ndarray]] | None) -> dict[str, Any]:
        ctx = copy.copy(context)
        ctx.dino = dict(context.dino)
        scored = {}
        for split in ("val", "eval"):
            if dino_norm is not None:
                ctx.dino[split] = dino_norm[split]
                scored[split] = score_split(split, norm_to_pixels(dino_norm[split], rows[split]))
            else:
                scored[split] = score_split(split, None)
        ctx.candidates = {split: rr.adjust_with_learned(candidates[split], scored[split], alpha) for split in ("val", "eval")}
        with FixedSetParams(set_params, set_grid):
            return exact.run_protocol(ctx, output_root / name, quick=False, retune_single_full_val=True, separate_multi_route_params=True, calibration_cache_root=full_root / "learned")

    results: list[dict[str, Any]] = []
    control = run_fixed("full_query", None)
    passed = abs(float(control["mean_iou"]) - sealed) < 1e-9
    log(f"seed {seed}: positive control mean IoU {control['mean_iou']:.6f} sealed {sealed:.6f} pass={passed}")
    if not passed:
        raise RuntimeError("positive control failed")
    results.append({"table": "query", "variant": "full_query", "seed": seed, "positive_control_pass": passed, **metric_row(control)})

    for variant in QUERY_VARIANTS:
        dino_norm = fa.single_dino_variant(seed, variant, device, 64)
        result = run_fixed(variant, dino_norm)
        results.append({"table": "query", "variant": variant, "seed": seed, **metric_row(result)})
        log(f"seed {seed} query {variant}: mean IoU {result['mean_iou']:.4f}")

    outputs = fa.dino_only_outputs(context, "eval")
    summary, _ = exact.evaluate_context(context, outputs)
    results.append({"table": "component", "variant": "rad_dino_only", "seed": seed, **metric_row(summary)})
    log(f"seed {seed} component rad_dino_only: mean IoU {summary['mean_iou']:.4f}")

    cols_nd = [c for c in cols if c != "dino_iou"]
    model_nd = rr.scorer_models(seed)[selection["selected_model"]]
    model_nd.fit(tables["train"][cols_nd].to_numpy(np.float32), tables["train"]["target_iou"].to_numpy(float))
    scored_nd = {}
    for split in ("val", "eval"):
        frame = tables[split].copy()
        frame["learned"] = np.clip(model_nd.predict(frame[cols_nd].to_numpy(np.float32)), 0.0, 1.0)
        scored_nd[split] = frame
    grid = []
    for a in rr.ALPHA_GRID:
        adjusted = rr.adjust_with_learned(candidates["val"], scored_nd["val"], a)
        grid.append({"alpha": a, "val_top_candidate_iou": rr.val_top_candidate_iou(rows["val"], labels["val"], adjusted)})
    grid = pd.DataFrame(grid).sort_values(["val_top_candidate_iou", "alpha"], ascending=[False, True])
    alpha_nd = float(grid.iloc[0]["alpha"])
    ctx = copy.copy(context)
    ctx.dino = {split: {} for split in SPLITS}
    ctx.candidates = {split: rr.adjust_with_learned(candidates[split], scored_nd[split], alpha_nd) for split in ("val", "eval")}
    result = exact.run_protocol(ctx, output_root / "yolo_only", quick=False, retune_single_full_val=True, separate_multi_route_params=True, calibration_cache_root=None)
    results.append({"table": "component", "variant": "yolo_only", "seed": seed, "alpha": alpha_nd, **metric_row(result)})
    log(f"seed {seed} component yolo_only: alpha={alpha_nd} mean IoU {result['mean_iou']:.4f}")
    results.append({"table": "component", "variant": "full", "seed": seed, "alpha": alpha, **metric_row(control)})
    rr.write_json(output_root / f"seed_{seed}_rows.json", results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-root", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        rows.extend(run_seed(seed, args.full_root, args.output_root, device))
        pd.DataFrame(rows).to_csv(args.output_root / "per_seed_rows_partial.csv", index=False)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_root / "per_seed_rows.csv", index=False)
    agg = frame.groupby(["table", "variant"], sort=False)[list(METRICS)].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(args.output_root / "aggregate.csv", index=False)
    rr.write_json(args.output_root / "FINAL_STATUS.json", {"status": "complete", "seeds": args.seeds, "positive_control_pass": bool(frame[frame.variant == "full_query"]["positive_control_pass"].all())})
    log("FINAL\n" + agg.round(4).to_string())


if __name__ == "__main__":
    main()

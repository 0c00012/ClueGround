#!/usr/bin/env python
"""Paired, patient-clustered bootstrap comparison on canonical MS-CXR-1444.

Methods compared on the same 220 evaluation phrase groups, all rescored with
the common evaluator (``src/baseline_repro/evaluator.py``):

* ``clueground_canonical``: validation-selected output of the per-seed
  auxiliary-chain rebuild (``run_clueground_canonical_aux_chain_3seed_v1``).
* ``clueground_hybrid_base``: the pure canonical hybrid-v4 arm of the same run
  (diagnostic; never selected on eval).
* ``clueground_gate_always``: the auxiliary-gate arm applied on every seed
  regardless of validation selection (fixed configuration; ablation row).
* ``clueground_legacy_paper``: the archived paper path (legacy group ids
  mapped by dicom/finding/phrase), for reference.
* ``medgrounder``: the locally retrained MedGrounder (Chest ImaGenome
  initialisation) from ``baseline_faithful_reproduction/20260712_v2``.

For every seed: per-method 95% patient-cluster bootstrap CIs, and paired
difference CIs (ClueGround minus MedGrounder) resampling subjects.  Across
seeds: mean +/- sample std, and the mean of per-seed paired differences.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_repro.evaluator import evaluate_rows, patient_cluster_bootstrap  # noqa: E402

SEEDS = (13, 42, 2026)
METRICS = ("coverage_iou", "exact_union_iou", "set_f1_optimal_0_3", "set_f1_optimal_0_5")
CANONICAL_PROTOCOL = (
    PROJECT_ROOT
    / "training"
    / "three_task_clueground_vfm_finding_conditioned_canonical_v3"
    / "protocols"
    / "task_isolated"
    / "mscxr_multibox_1444"
)
MEDGROUNDER_ROOT = PROJECT_ROOT / "experiments" / "baseline_faithful_reproduction" / "20260712_v2" / "runs" / "medgrounder" / "local_multibox"
LEGACY_PAPER_ROOT = PROJECT_ROOT / "experiments" / "clueground_finding_moe_full_upstream_no_siglip_score_ablation_v1"
LEGACY_STAGE1_EVAL = PROJECT_ROOT / "training" / "ms_cxr_vfm_localizer_stage1_p10_p19" / "datasets" / "eval.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def norm_text(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", str(text).lower()).split())


def phrase_only(query_text: str) -> str:
    marker = "; phrase "
    return str(query_text).split(marker, 1)[1] if marker in str(query_text) else str(query_text)


def key_of(dicom_id: str, finding: str, phrase: str) -> tuple[str, str, str]:
    return str(dicom_id), norm_text(finding), norm_text(phrase)


def cxcywh_to_xyxy(box: list[float]) -> list[float]:
    cx, cy, w, h = [float(v) for v in box]
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


# ----------------------------------------------------------------------------
# canonical evaluation samples (gold boxes, subject ids) keyed by group id
# ----------------------------------------------------------------------------


def canonical_samples() -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], str]]:
    inputs = read_jsonl(CANONICAL_PROTOCOL / "eval_inputs.jsonl")
    labels = {str(r["group_id"]): r["gold_boxes_xyxy"] for r in read_jsonl(CANONICAL_PROTOCOL / "eval_labels.jsonl")}
    samples = []
    key_to_gid: dict[tuple[str, str, str], str] = {}
    for src in inputs:
        gid = str(src["group_id"])
        samples.append(
            {
                "query_id": gid,
                "subject_id": str(src["subject_id"]),
                "finding": str(src["finding"]),
                "gold_boxes": [[float(v) for v in box] for box in labels[gid]],
                "image_width": float(src["image_width"]),
                "image_height": float(src["image_height"]),
            }
        )
        key = key_of(src["dicom_id"], src["finding"], phrase_only(src["query_text"]))
        if key in key_to_gid:
            raise RuntimeError(f"duplicate canonical key {key}")
        key_to_gid[key] = gid
    if len(samples) != 220:
        raise RuntimeError(f"expected 220 canonical eval groups, got {len(samples)}")
    return samples, key_to_gid


# ----------------------------------------------------------------------------
# prediction loaders -> {group_id: [xyxy boxes]}
# ----------------------------------------------------------------------------


def load_clueground_predictions(path: Path) -> dict[str, list[list[float]]]:
    if path.suffix == ".jsonl":
        return {str(r["group_id"]): [[float(v) for v in b] for b in r["pred_boxes_xyxy"]] for r in read_jsonl(path)}
    frame = pd.read_csv(path)
    return {str(r.group_id): json.loads(r.pred_boxes_json) for r in frame.itertuples()}


GOLD_MISMATCH: dict[int, dict[str, Any]] = {}


def _iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def load_medgrounder(seed: int, key_to_gid: dict, samples_by_gid: dict) -> tuple[list[dict[str, Any]], dict[str, list[list[float]]]]:
    """MedGrounder predictions and gold are normalised cxcywh in the model's
    own frame (Table 1 scored them exactly so).  Keep that frame; report how
    far its gold deviates from the canonical pixel gold for transparency."""
    frame = pd.read_csv(MEDGROUNDER_ROOT / f"seed_{seed}" / "predictions" / "test_predictions.csv")
    samples: list[dict[str, Any]] = []
    predictions: dict[str, list[list[float]]] = {}
    mismatched = 0
    max_dev = 0.0
    for r in frame.itertuples():
        dicom = Path(str(r.img_path)).stem
        key = key_of(dicom, str(r.category_names), str(r.phrase))
        gid = key_to_gid.get(key)
        if gid is None:
            raise RuntimeError(f"MedGrounder row not matched to canonical group: {key}")
        sample = samples_by_gid[gid]
        w, h = sample["image_width"], sample["image_height"]
        gold_norm = [cxcywh_to_xyxy(b) for b in json.loads(r.gt_boxes_cxcywh)]
        pred_norm = [cxcywh_to_xyxy(b) for b in json.loads(r.pred_boxes_cxcywh)]
        # MedGrounder normalises boxes on a centred square letterbox canvas.
        # Map its gold back to canonical pixels and check agreement by IoU;
        # IoU is invariant to this similarity transform, so scoring in the
        # model's own frame is equivalent to scoring in canonical pixels.
        L = max(w, h)
        ox, oy = (L - w) / 2, (L - h) / 2
        gold_px = [[b[0] * L - ox, b[1] * L - oy, b[2] * L - ox, b[3] * L - oy] for b in gold_norm]
        canonical_gold = sample["gold_boxes"]
        if len(gold_px) != len(canonical_gold):
            raise RuntimeError(f"gold count mismatch for {gid}")
        dev = min(max(_iou(g, t) for t in gold_px) for g in canonical_gold)
        if dev < 0.99:
            mismatched += 1
        max_dev = min(max_dev, dev) if max_dev else dev
        samples.append({"query_id": gid, "subject_id": sample["subject_id"], "finding": sample["finding"], "gold_boxes": gold_norm})
        predictions[gid] = pred_norm
    if len(predictions) != 220:
        raise RuntimeError(f"MedGrounder seed {seed}: {len(predictions)} groups matched")
    GOLD_MISMATCH[seed] = {"gold_boxes_iou_below_0.99_after_letterbox_inverse": mismatched, "min_gold_iou": round(max_dev, 4)}
    return samples, predictions


def load_legacy_paper_predictions(seed: int, key_to_gid: dict) -> dict[str, list[list[float]]]:
    """Archived paper path: legacy group ids -> canonical ids via (dicom, finding, phrase)."""
    rows = read_jsonl(LEGACY_STAGE1_EVAL)
    legacy_groups: dict[str, tuple[str, str, str]] = {}
    for row in rows:
        gid = f"{row['dicom_id']}|{norm_text(row['finding'])}|{norm_text(row.get('claim_sentence') or row.get('phrase'))}"
        legacy_groups.setdefault(gid, key_of(row["dicom_id"], row["finding"], row.get("claim_sentence") or row.get("phrase")))
    frame = pd.read_csv(LEGACY_PAPER_ROOT / f"seed_{seed}" / "eval" / "gate" / "predictions.csv")
    out: dict[str, list[list[float]]] = {}
    unmatched = 0
    for r in frame.itertuples():
        gid = str(r.group_id)
        # Legacy prediction group ids follow make_groups' group_key; recover the
        # key from the STAGE1 rows by scanning for a matching id.
        key = _legacy_gid_key(gid, rows)
        target = key_to_gid.get(key) if key else None
        if target is None:
            unmatched += 1
            continue
        out[target] = json.loads(r.pred_boxes_json)
    if unmatched or len(out) != 220:
        raise RuntimeError(f"legacy paper seed {seed}: matched {len(out)}, unmatched {unmatched}")
    return out


_LEGACY_KEY_CACHE: dict[str, tuple[str, str, str]] = {}


def _legacy_gid_key(gid: str, rows: list[dict[str, Any]]) -> tuple[str, str, str] | None:
    if not _LEGACY_KEY_CACHE:
        from scripts import run_ms_cxr_multibox_phrase_grounding_v1 as mb  # noqa: E402

        for row in rows:
            row = dict(row)
            row.setdefault("split", "eval")
            _LEGACY_KEY_CACHE[str(mb.group_key(row))] = key_of(
                row["dicom_id"], row["finding"], row.get("claim_sentence") or row.get("phrase")
            )
    return _LEGACY_KEY_CACHE.get(gid)


# ----------------------------------------------------------------------------
# scoring and bootstrap
# ----------------------------------------------------------------------------


def score(method: str, seed: int, predictions: dict[str, list[list[float]]], samples: list[dict[str, Any]]) -> pd.DataFrame:
    pred_rows = [{"query_id": gid, "pred_boxes": boxes} for gid, boxes in predictions.items()]
    summary, detail = evaluate_rows(samples, pred_rows, force_single_box=False)
    if summary["n_missing_predictions"]:
        raise RuntimeError(f"{method} seed {seed}: missing predictions {summary['n_missing_predictions']}")
    frame = pd.DataFrame(detail)
    frame["method"] = method
    frame["seed"] = seed
    frame["coverage_iou"] = frame["coverage_iou"]
    return frame


def paired_difference_ci(
    left: pd.DataFrame, right: pd.DataFrame, metrics: tuple[str, ...], reps: int, seed: int
) -> list[dict[str, Any]]:
    merged = left.merge(right, on="query_id", suffixes=("_a", "_b"))
    assert len(merged) == 220, len(merged)
    clusters = sorted(merged["subject_id_a"].astype(str).unique())
    cluster_index = {c: i for i, c in enumerate(clusters)}
    member = merged["subject_id_a"].astype(str).map(cluster_index).to_numpy()
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(clusters), size=(reps, len(clusters)))
    rows = []
    for metric in metrics:
        diff = merged[f"{metric}_a"].to_numpy(float) - merged[f"{metric}_b"].to_numpy(float)
        sums = np.bincount(member, weights=diff, minlength=len(clusters))
        counts = np.bincount(member, minlength=len(clusters)).astype(float)
        samples = sums[choices].sum(axis=1) / counts[choices].sum(axis=1)
        rows.append(
            {
                "metric": metric,
                "mean_difference": float(diff.mean()),
                "ci_95_low": float(np.quantile(samples, 0.025)),
                "ci_95_high": float(np.quantile(samples, 0.975)),
                "p_bootstrap_two_sided": float(2 * min((samples <= 0).mean(), (samples >= 0).mean())),
                "n_groups": int(len(merged)),
                "n_subjects": int(len(clusters)),
                "groups_a_better": int((diff > 0).sum()),
                "groups_b_better": int((diff < 0).sum()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=None, help="…/clueground_canonical_aux_chain_v1/canonical (aux-chain arms; optional)")
    parser.add_argument("--extra", nargs="*", default=[], help="name=path pattern with {seed}; jsonl (group_id, pred_boxes_xyxy) or csv (group_id, pred_boxes_json)")
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--skip-legacy", action="store_true")
    args = parser.parse_args()
    out_root = args.out_root or (args.run_root / "comparison_vs_medgrounder")
    out_root.mkdir(parents=True, exist_ok=True)

    samples, key_to_gid = canonical_samples()
    samples_by_gid = {s["query_id"]: s for s in samples}

    details: list[pd.DataFrame] = []
    selected_variants: dict[int, str] = {}
    extras = []
    for item in args.extra:
        name, pattern = item.split("=", 1)
        extras.append((name, pattern))
    for seed in args.seeds:
        if args.run_root is not None:
            status = json.loads((args.run_root / f"seed_{seed}" / "RUN_STATUS.json").read_text(encoding="utf-8"))
            variant = status["selected_variant"]
            selected_variants[seed] = variant
            eval_root = args.run_root / f"seed_{seed}" / "gate" / "eval"
            details.append(score("clueground_canonical", seed, load_clueground_predictions(eval_root / variant / "predictions.csv"), samples))
            details.append(score("clueground_hybrid_base", seed, load_clueground_predictions(eval_root / "base_diagnostic" / "predictions.csv"), samples))
            details.append(score("clueground_gate_always", seed, load_clueground_predictions(eval_root / "gate_diagnostic" / "predictions.csv"), samples))
        for name, pattern in extras:
            details.append(score(name, seed, load_clueground_predictions(Path(pattern.format(seed=seed))), samples))
        mg_samples, mg_predictions = load_medgrounder(seed, key_to_gid, samples_by_gid)
        details.append(score("medgrounder", seed, mg_predictions, mg_samples))
        if not args.skip_legacy:
            details.append(score("clueground_legacy_paper", seed, load_legacy_paper_predictions(seed, key_to_gid), samples))
    detail = pd.concat(details, ignore_index=True)
    detail.to_csv(out_root / "per_group_detail.csv", index=False)

    # Per-method, per-seed point estimates and patient-cluster CIs.
    ci = patient_cluster_bootstrap(
        detail,
        identity_columns=["method", "seed"],
        metric_columns=list(METRICS),
        cluster_column="subject_id",
        reps=args.reps,
    )
    ci.to_csv(out_root / "per_seed_ci.csv", index=False)

    # Seed aggregate.
    per_seed = detail.groupby(["method", "seed"])[list(METRICS)].mean().reset_index()
    agg = per_seed.groupby("method")[list(METRICS)].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(out_root / "seed_aggregate.csv", index=False)

    # Paired differences vs MedGrounder, per seed and averaged.
    paired_rows = []
    for method in [m for m in detail["method"].unique() if m != "medgrounder"]:
        for seed in args.seeds:
            left = detail[(detail.method == method) & (detail.seed == seed)]
            right = detail[(detail.method == "medgrounder") & (detail.seed == seed)]
            for row in paired_difference_ci(left, right, METRICS, args.reps, seed=20260908 + seed):
                paired_rows.append({"method": method, "vs": "medgrounder", "seed": seed, **row})
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(out_root / "paired_difference_ci.csv", index=False)

    # Markdown report.
    lines = ["# ClueGround canonical rebuild vs MedGrounder (canonical 1444, common evaluator)", ""]
    lines.append(f"Seeds: {args.seeds}; selected variants: {selected_variants}; bootstrap reps: {args.reps}; cluster: subject_id")
    lines.append(f"MedGrounder scored in its own normalised frame (as in Table 1); gold consistency with canonical pixel gold after inverse letterbox: {GOLD_MISMATCH}")
    lines.append("")
    lines.append("## Seed mean +/- sample std")
    lines.append("")
    lines.append("| method | Coverage | Exact union | SetF1@0.3 | SetF1@0.5 |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in agg.itertuples():
        cells = [f"{getattr(r, m + '_mean'):.4f} +/- {getattr(r, m + '_std'):.4f}" for m in METRICS]
        lines.append(f"| {r.method} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Per-seed 95% patient-cluster bootstrap CI")
    lines.append("")
    lines.append("| method | seed | metric | point | CI low | CI high |")
    lines.append("|---|---:|---|---:|---:|---:|")
    for r in ci.sort_values(["metric", "method", "seed"]).itertuples():
        lines.append(f"| {r.method} | {r.seed} | {r.metric} | {r.point_estimate:.4f} | {r.ci_95_low:.4f} | {r.ci_95_high:.4f} |")
    lines.append("")
    lines.append("## Paired difference vs MedGrounder (same 220 groups, subject-cluster bootstrap)")
    lines.append("")
    lines.append("| method | seed | metric | mean diff | CI low | CI high | p (two-sided) | groups better/worse |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|")
    for r in paired.sort_values(["method", "metric", "seed"]).itertuples():
        lines.append(
            f"| {r.method} | {r.seed} | {r.metric} | {r.mean_difference:+.4f} | {r.ci_95_low:+.4f} | {r.ci_95_high:+.4f} | {r.p_bootstrap_two_sided:.3f} | {r.groups_a_better}/{r.groups_b_better} |"
        )
    lines.append("")
    lines.append("## Mean of per-seed paired differences")
    lines.append("")
    lines.append("| method | metric | mean diff over seeds | min | max | seeds with CI excluding 0 |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for (method, metric), part in paired.groupby(["method", "metric"]):
        excl = int(((part.ci_95_low > 0) | (part.ci_95_high < 0)).sum())
        lines.append(f"| {method} | {metric} | {part.mean_difference.mean():+.4f} | {part.mean_difference.min():+.4f} | {part.mean_difference.max():+.4f} | {excl}/{len(part)} |")
    (out_root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

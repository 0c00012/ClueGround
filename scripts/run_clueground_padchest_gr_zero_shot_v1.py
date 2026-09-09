#!/usr/bin/env python
"""Zero-shot external evaluation of ClueGround (learned re-ranker) on PadChest-GR.

Nothing is trained or tuned here.  For each seed the sealed MS-CXR-1444
assets are applied unchanged to the PadChest-GR protocol written by
``build_padchest_gr_protocol.py``:

1. four finding-conditioned YOLO detectors -> candidate boxes,
2. frozen RAD-DINO patch tokens + sealed localization head -> phrase-conditioned box,
3. sealed re-ranker: scorer refit deterministically from the sealed training
   table (verified against the sealed MS-CXR eval scores), alpha from the
   sealed run, logit adjustment of the detector confidence,
4. unchanged decoder: sealed single-route calibration, sealed multi-route set
   parameters, rule-context routing,
5. common evaluator (Coverage IoU, Exact Union IoU, SetF1@0.3/0.5), per-finding
   and per-reference-count breakdown, patient-cluster bootstrap CIs.

Two diagnostics run alongside: the same pipeline without the re-ranker
(alpha = 0, the preliminary hand-weighted score) and the RAD-DINO box alone.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
for path in (BUNDLE_ROOT, BUNDLE_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import run_clueground_canonical_learned_reranker_3seed_v1 as rr  # noqa: E402
from scripts import run_clueground_exact_hybrid_v4_direct_3seed_v1 as exact  # noqa: E402
from scripts import run_clueground_vfm_legacy_fusion_unified_3seed_v1 as legacy  # noqa: E402
from scripts import run_ms_cxr_multibox_rule_context_fusion_v4_yolov8l_pool as hybrid_v4  # noqa: E402
from scripts import run_ms_cxr_rad_dino_singlebox_retrain_v1 as rad_single  # noqa: E402
from scripts import run_ms_cxr_yolo_detector_v1 as yd  # noqa: E402
from scripts import run_ms_cxr_yolo_rule_context_v1 as ybase  # noqa: E402
from scripts.models_ms_cxr_vfm_localizer import PatchHeatmapBBoxHead  # noqa: E402
from src.baseline_repro.evaluator import patient_cluster_bootstrap  # noqa: E402
from src.three_task_grounding.manifests import read_jsonl, write_jsonl  # noqa: E402

RAD_DINO_MODEL_ID = "microsoft/rad-dino"
RAD_DINO_REVISION = "110cbc18d5133582e320b43d53bf5c44e410c936"
MODELS = ("yolov8s", "yolov8m", "yolo11s", "yolo11m")
SEEDS = (13, 42, 2026)
METRICS = ("coverage_iou", "exact_union_iou", "set_f1_optimal_0_3", "set_f1_optimal_0_5")
FINDINGS = list(yd.CLASS_NAMES)


def log(message: str) -> None:
    print(f"[padchest {time.strftime('%H:%M:%S')}] {message}", flush=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


# ----------------------------------------------------------------------------
# stage 1: YOLO candidates
# ----------------------------------------------------------------------------


def predict_yolo(seed_assets: Path, rows: list[dict[str, Any]], out_dir: Path, device: str, batch: int, force: bool) -> None:
    from ultralytics import YOLO

    out_dir.mkdir(parents=True, exist_ok=True)
    unique = sorted({str(r["dicom_id"]): str(r["image_path"]) for r in rows}.items())
    torch_device = torch.device("cuda" if torch.cuda.is_available() and device != "cpu" else "cpu")
    for tag in MODELS:
        output = out_dir / f"{tag}_eval.csv"
        if output.exists() and not force:
            continue
        model = YOLO(str(seed_assets / "yolo" / f"{tag}_best.pt"))
        names = [model.names[i] for i in range(len(model.names))]
        if names != FINDINGS:
            raise RuntimeError(f"{tag}: detector class order {names} != {FINDINGS}")
        records: list[dict[str, Any]] = []
        start, active = 0, max(1, batch)
        while start < len(unique):
            part = unique[start : start + active]
            try:
                results = model.predict(source=[p for _, p in part], imgsz=640, conf=0.001, max_det=100, batch=len(part), half=torch_device.type == "cuda", device=device, verbose=False)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and active > 1:
                    active = max(1, active // 2)
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                raise
            for (dicom_id, image_path), result in zip(part, results):
                if result.boxes is None:
                    continue
                boxes = result.boxes.xyxy.detach().cpu().numpy()
                scores = result.boxes.conf.detach().cpu().numpy()
                classes = result.boxes.cls.detach().cpu().numpy().astype(int)
                order = np.argsort(-scores, kind="stable")
                for rank, index in enumerate(order):
                    records.append({"dicom_id": dicom_id, "image_path": image_path, "source_model": tag, "rank": rank, "confidence": float(scores[index]), "class_id": int(classes[index]), "class_name": names[int(classes[index])],
                                    "pred_x1": float(boxes[index, 0]), "pred_y1": float(boxes[index, 1]), "pred_x2": float(boxes[index, 2]), "pred_y2": float(boxes[index, 3])})
            start += len(part)
        pd.DataFrame(records, columns=["dicom_id", "image_path", "source_model", "rank", "confidence", "class_id", "class_name", "pred_x1", "pred_y1", "pred_x2", "pred_y2"]).to_csv(output, index=False)
        log(f"{tag}: {len(records)} candidate rows for {len(unique)} images")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ----------------------------------------------------------------------------
# stage 2: RAD-DINO tokens + sealed head
# ----------------------------------------------------------------------------


def extract_rad_tokens(rows: list[dict[str, Any]], cache: Path, device: str, batch: int, force: bool, local_model: str | None) -> tuple[list[str], np.ndarray]:
    if cache.exists() and not force:
        archive = np.load(cache, allow_pickle=True)
        return [str(x) for x in archive["dicom_ids"]], archive["patch_tokens"]
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    source = local_model or RAD_DINO_MODEL_ID
    kwargs = {} if local_model else {"revision": RAD_DINO_REVISION}
    processor = AutoImageProcessor.from_pretrained(source, trust_remote_code=True, **kwargs)
    model = AutoModel.from_pretrained(source, trust_remote_code=True, **kwargs)
    torch_device = torch.device("cuda" if torch.cuda.is_available() and device != "cpu" else "cpu")
    model = model.to(torch_device).eval()
    unique = sorted({str(r["dicom_id"]): str(r["image_path"]) for r in rows}.items())
    ids, tokens = [], []
    with torch.no_grad():
        for start in range(0, len(unique), batch):
            part = unique[start : start + batch]
            images = []
            for _, path in part:
                with Image.open(path) as img:
                    images.append(img.convert("RGB"))
            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to(torch_device) for k, v in inputs.items()}
            hidden = model(**inputs).last_hidden_state.detach().float()
            patches = hidden[:, 1:, :] if hidden.shape[1] > 1 else hidden  # identical to the stage-1 cache (CLS dropped)
            tokens.extend(patches.cpu().numpy().astype(np.float16))
            ids.extend(d for d, _ in part)
    array = np.stack(tokens)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, dicom_ids=np.asarray(ids, dtype=object), patch_tokens=array)
    log(f"RAD-DINO tokens {array.shape} cached")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ids, array


def predict_rad_head(seed_assets: Path, seed: int, rows: list[dict[str, Any]], dicom_ids: list[str], tokens: np.ndarray, device: str) -> dict[str, np.ndarray]:
    torch_device = torch.device("cuda" if torch.cuda.is_available() and device != "cpu" else "cpu")
    index = {d: i for i, d in enumerate(dicom_ids)}
    frame = pd.DataFrame({"finding_label": [str(r["finding"]) for r in rows], "phrase_text": [str(r["claim_sentence"]) for r in rows]})
    queries = rad_single.encode_query(frame, "full_phrase").astype(np.float32)
    payload = torch.load(seed_assets / "rad_dino_head_best.pt", map_location="cpu", weights_only=False)
    model = PatchHeatmapBBoxHead(tokens.shape[-1], queries.shape[-1], hidden=384, dropout=0.1).to(torch_device)
    model.load_state_dict(payload["state"])
    outputs: dict[str, np.ndarray] = {}
    batch = 16
    with torch.no_grad():
        model.eval()
        for start in range(0, len(rows), batch):
            part = rows[start : start + batch]
            tok = np.stack([tokens[index[str(r["dicom_id"])]] for r in part]).astype(np.float32)
            pred, _ = model(torch.from_numpy(tok).to(torch_device), torch.from_numpy(queries[start : start + batch]).to(torch_device))
            for row, box in zip(part, pred.detach().cpu().numpy()):
                outputs[str(row["group_id"])] = box.astype(np.float32)
    return outputs


# ----------------------------------------------------------------------------
# stage 3: sealed re-ranker
# ----------------------------------------------------------------------------


def refit_sealed_scorer(seed_assets: Path, seed: int, tolerance: float) -> tuple[Any, list[str], dict[str, Any]]:
    train = pd.read_csv(seed_assets / "candidates_train.csv")  # rr forces a round-trip float parser
    selection = json.loads((seed_assets / "scorer_selection.json").read_text(encoding="utf-8"))
    alpha_info = json.loads((seed_assets / "learned_alpha.json").read_text(encoding="utf-8"))
    cols = rr.feature_columns(train)
    if cols != list(selection["features"]):
        raise RuntimeError(f"feature columns differ from the sealed run: {cols} vs {selection['features']}")
    model = rr.scorer_models(seed)[str(selection["selected_model"])]
    model.fit(train[cols].to_numpy(np.float32), train["target_iou"].to_numpy(float))
    sealed_eval = pd.read_csv(seed_assets / "candidates_eval.csv")
    sealed_scores = pd.read_csv(seed_assets / "scored_eval.csv")["learned"].to_numpy(float)
    refit = np.clip(model.predict(sealed_eval[cols].to_numpy(np.float32)), 0.0, 1.0)
    max_diff = float(np.abs(refit - sealed_scores).max())
    control = {"scorer": selection["selected_model"], "alpha": float(alpha_info["learned_alpha"]), "n_train_rows": int(len(train)), "max_abs_diff_vs_sealed_mscxr_eval_scores": max_diff, "pass": max_diff <= tolerance}
    log(f"seed {seed}: scorer {selection['selected_model']} refit; max |diff| vs sealed MS-CXR eval scores = {max_diff:.2e} ({'PASS' if control['pass'] else 'FAIL'})")
    return model, cols, control


class FixedSetParams:
    def __init__(self, params: dict[str, Any], grid: pd.DataFrame) -> None:
        self.params, self.grid, self.original = params, grid, hybrid_v4.tune_set_params

    def __enter__(self) -> None:
        hybrid_v4.tune_set_params = lambda *args, **kwargs: (dict(self.params), self.grid.copy())

    def __exit__(self, *exc: Any) -> None:
        hybrid_v4.tune_set_params = self.original


# ----------------------------------------------------------------------------
# evaluation helpers
# ----------------------------------------------------------------------------


def breakdown(detail: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"per_finding": {}, "per_reference_count": {}}
    for finding, part in detail.groupby("finding"):
        out["per_finding"][str(finding)] = {"n": int(len(part)), **{m: float(part[m].mean()) for m in METRICS}}
    detail = detail.copy()
    detail["ref_count"] = np.where(detail["n_gt"].astype(int) == 1, "single (1 box)", "multi (2+ boxes)")
    for key, part in detail.groupby("ref_count"):
        out["per_reference_count"][str(key)] = {"n": int(len(part)), **{m: float(part[m].mean()) for m in METRICS}}
    return out


def run_seed(seed: int, args: argparse.Namespace, inputs: list[dict[str, Any]], labels: dict[str, list[list[float]]], rows: list[dict[str, Any]], priors: dict[str, Any]) -> dict[str, Any]:
    seed_assets = args.assets / f"seed_{seed}"
    seed_root = args.out_root / f"seed_{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)

    predict_yolo(seed_assets, rows, seed_root / "yolo_predictions", args.device, args.yolo_batch, args.force)
    candidates = legacy.load_yolo_candidates(seed_root / "yolo_predictions", "eval")

    dicom_ids, tokens = extract_rad_tokens(rows, args.out_root / "rad_dino_patch_tokens.npz", args.device, args.rad_batch, args.force, args.rad_dino_local)
    dino_norm = predict_rad_head(seed_assets, seed, rows, dicom_ids, tokens, args.device)
    dino_xyxy = {gid: ybase.norm_to_xyxy(box, float(r["image_width"]), float(r["image_height"])) for r in rows for gid in [str(r["group_id"])] for box in [dino_norm[gid]]}
    write_jsonl(seed_root / "rad_dino_boxes.jsonl", [{"group_id": gid, "pred_bbox_norm_cxcywh": [float(v) for v in dino_norm[gid]], "pred_xyxy": dino_xyxy[gid]} for gid in dino_norm])

    table = rr.build_table("eval", rows, labels, candidates, priors, {"dino": dino_xyxy, "xattn": {}, "aux": {}})
    table.to_csv(seed_root / "candidates_eval.csv", index=False)
    model, cols, control = refit_sealed_scorer(seed_assets, seed, args.scorer_tolerance)
    if not control["pass"] and not args.allow_scorer_mismatch:
        raise RuntimeError("sealed scorer could not be reproduced; check scikit-learn version (1.6.1) or pass --allow-scorer-mismatch")
    scored = table.copy()
    scored["learned"] = np.clip(model.predict(scored[cols].to_numpy(np.float32)), 0.0, 1.0)
    scored.to_csv(seed_root / "scored_eval.csv", index=False)
    alpha = float(control["alpha"])
    adjusted = rr.adjust_with_learned(candidates, scored, alpha)

    yolo_params = json.loads((seed_assets / "legacy_fusion" / "yolo_params.json").read_text(encoding="utf-8"))
    fusion_params = json.loads((seed_assets / "legacy_fusion" / "fusion_params.json").read_text(encoding="utf-8"))
    set_params = json.loads((seed_assets / "selected_set_params.json").read_text(encoding="utf-8"))
    set_grid = pd.read_csv(seed_assets / "multibox_v4_val_grid.csv")

    def make_context(cands: dict[str, list[dict[str, Any]]]) -> exact.ProtocolContext:
        return exact.ProtocolContext(
            "multibox_1444", seed,
            {"train": [], "val": rows, "eval": rows},
            {"val": inputs, "eval": inputs},
            {"val": labels, "eval": labels},
            {"val": copy.deepcopy(cands), "eval": copy.deepcopy(cands)},
            {"val": dino_norm, "eval": dino_norm},
            priors, copy.deepcopy(yolo_params), copy.deepcopy(fusion_params),
            {"external_dataset": "PadChest-GR", "protocol": str(args.protocol_root), "assets": str(seed_assets), "frozen": True, "label_mapping": args.mapping},
        )

    results: dict[str, Any] = {"seed": seed, "alpha": alpha, "scorer_control": control, "n_rows": len(rows)}
    details: list[pd.DataFrame] = []
    for variant, cands in (("clueground_reranker", adjusted), ("hybrid_no_reranker", candidates)):
        ctx = make_context(cands)
        with FixedSetParams(set_params, set_grid):
            result = exact.run_protocol(ctx, seed_root / variant, quick=False, retune_single_full_val=True, separate_multi_route_params=True, calibration_cache_root=args.assets / "calibration")
        detail = pd.read_csv(seed_root / variant / "multibox_1444" / f"seed_{seed}" / "eval_detail_common_evaluator.csv")
        detail["method"], detail["seed"] = variant, seed
        details.append(detail)
        results[variant] = {m: float(result[m]) for m in METRICS} | {"mean_pred_count": float(result.get("mean_pred_count", float("nan"))), "n_multi_route": int(result.get("n_multi_route", 0)), **breakdown(detail)}
        log(f"seed {seed} {variant}: coverage {result['coverage_iou']:.4f} union {result['exact_union_iou']:.4f} f1@.3 {result['set_f1_optimal_0_3']:.4f} f1@.5 {result['set_f1_optimal_0_5']:.4f}")

    ctx = make_context(candidates)
    dino_only = {gid: [dino_xyxy[gid]] for gid in dino_xyxy}
    summary, detail = exact.evaluate_context(ctx, dino_only)
    detail["method"], detail["seed"] = "rad_dino_only", seed
    details.append(detail)
    results["rad_dino_only"] = {m: float(summary[m]) for m in METRICS} | breakdown(detail)
    log(f"seed {seed} rad_dino_only: coverage {summary['coverage_iou']:.4f}")

    all_detail = pd.concat(details, ignore_index=True)
    all_detail.to_csv(seed_root / "per_row_detail.csv", index=False)
    ci = patient_cluster_bootstrap(all_detail, identity_columns=["method", "seed"], metric_columns=list(METRICS), cluster_column="subject_id", reps=args.bootstrap_reps)
    ci.to_csv(seed_root / "per_seed_ci.csv", index=False)
    results["ci"] = ci.to_dict("records")
    write_json(seed_root / "SEED_RESULT.json", results)
    return results


def aggregate(args: argparse.Namespace, per_seed: list[dict[str, Any]]) -> None:
    rows = []
    for variant in ("clueground_reranker", "hybrid_no_reranker", "rad_dino_only"):
        for m in METRICS:
            values = [r[variant][m] for r in per_seed]
            rows.append({"method": variant, "metric": m, "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"), "per_seed": values})
    agg = pd.DataFrame(rows)
    agg.to_csv(args.out_root / "seed_aggregate.csv", index=False)
    finding_rows = []
    for variant in ("clueground_reranker", "hybrid_no_reranker", "rad_dino_only"):
        for key in ("per_finding", "per_reference_count"):
            names = sorted({k for r in per_seed for k in r[variant][key]})
            for name in names:
                parts = [r[variant][key][name] for r in per_seed if name in r[variant][key]]
                finding_rows.append({"method": variant, "group": key, "name": name, "n": parts[0]["n"], **{m: float(np.mean([p[m] for p in parts])) for m in METRICS}})
    pd.DataFrame(finding_rows).to_csv(args.out_root / "per_finding_aggregate.csv", index=False)
    lines = [f"# ClueGround zero-shot on PadChest-GR ({args.mapping} mapping, {per_seed[0]['n_rows']} rows, seeds {[r['seed'] for r in per_seed]})", "",
             "| method | C-IoU | U-IoU | F1@.3 | F1@.5 |", "|---|---:|---:|---:|---:|"]
    for variant in ("clueground_reranker", "hybrid_no_reranker", "rad_dino_only"):
        cells = []
        for m in METRICS:
            r = agg[(agg.method == variant) & (agg.metric == m)].iloc[0]
            cells.append(f"{r['mean']:.4f} +/- {r['std']:.4f}")
        lines.append(f"| {variant} | " + " | ".join(cells) + " |")
    lines += ["", "## per finding / reference count (clueground_reranker, three-seed mean)", "", "| group | name | n | C-IoU | U-IoU | F1@.3 | F1@.5 |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in finding_rows:
        if r["method"] == "clueground_reranker":
            lines.append(f"| {r['group']} | {r['name']} | {r['n']} | " + " | ".join(f"{r[m]:.4f}" for m in METRICS) + " |")
    lines += ["", "scorer positive controls: " + "; ".join(f"seed {r['seed']} max|diff|={r['scorer_control']['max_abs_diff_vs_sealed_mscxr_eval_scores']:.1e} {'PASS' if r['scorer_control']['pass'] else 'FAIL'}" for r in per_seed)]
    (args.out_root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol-root", type=Path, required=True, help="…/protocol_strict or …/protocol_extended written by build_padchest_gr_protocol.py")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=BUNDLE_ROOT / "assets")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--device", default="0", help="ultralytics device string ('0', 'cpu')")
    parser.add_argument("--yolo-batch", type=int, default=8)
    parser.add_argument("--rad-batch", type=int, default=8)
    parser.add_argument("--rad-dino-local", default=None, help="local directory of microsoft/rad-dino if the workstation is offline")
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--scorer-tolerance", type=float, default=1e-6)
    parser.add_argument("--allow-scorer-mismatch", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="debug: evaluate only the first N rows")
    parser.add_argument("--force", action="store_true", help="recompute YOLO/RAD-DINO caches")
    args = parser.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    rr.NO_AUX_ASSETS = True
    rr.FEATURE_EXCLUDE = set(rr.FEATURE_EXCLUDE) | {"xattn_iou", "aux_iou"}

    proto = args.protocol_root / "padchest_gr_external"
    inputs = read_jsonl(proto / "eval_inputs.jsonl")
    label_rows = read_jsonl(proto / "eval_labels.jsonl")
    if args.limit:
        inputs = inputs[: args.limit]
        keep = {str(r["group_id"]) for r in inputs}
        label_rows = [r for r in label_rows if str(r["group_id"]) in keep]
    labels = {str(r["group_id"]): [[float(v) for v in b] for b in r["gold_boxes_xyxy"]] for r in label_rows}
    rows = legacy.group_rows(inputs, labels, "eval")
    args.mapping = str(inputs[0].get("label_mapping", "?")) if inputs else "?"
    log(f"protocol {proto}: {len(rows)} rows, {len({r['dicom_id'] for r in rows})} images, mapping={args.mapping}")

    prior_rows = json.loads((args.assets / "mscxr_train_prior_rows.json").read_text(encoding="utf-8"))
    priors = ybase.make_train_priors(prior_rows)

    per_seed = []
    for seed in args.seeds:
        result_path = args.out_root / f"seed_{seed}" / "SEED_RESULT.json"
        if result_path.exists() and not args.force:
            per_seed.append(json.loads(result_path.read_text(encoding="utf-8")))
            log(f"seed {seed}: reusing {result_path}")
            continue
        per_seed.append(run_seed(seed, args, inputs, labels, rows, priors))
    aggregate(args, per_seed)
    write_json(args.out_root / "FINAL_STATUS.json", {"status": "complete", "protocol_root": str(args.protocol_root), "mapping": args.mapping, "seeds": args.seeds, "n_rows": len(rows)})


if __name__ == "__main__":
    main()

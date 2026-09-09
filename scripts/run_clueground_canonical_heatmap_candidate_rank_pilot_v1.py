#!/usr/bin/env python
"""Canonical 1444 seed-13 pilot: train RAD-DINO heatmaps to rank fixed YOLO boxes.

No detector, candidate coordinate, external model, or decoder is added.  The
existing phrase-conditioned PatchHeatmapBBoxHead is warm-started from the
seed-specific MS-CXR head.  Its patch logits are fine-tuned so the mean heat
inside a fixed finding-conditioned YOLO candidate ranks candidates by train
IoU.  One global heat-score mixing weight is selected on val124 only.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import run_clueground_exact_hybrid_v4_direct_3seed_v1 as exact  # noqa: E402
from scripts import run_clueground_canonical_neural_agreement_selector_pilot_v1 as canonical  # noqa: E402
from scripts import run_clueground_vfm_legacy_fusion_unified_3seed_v1 as legacy  # noqa: E402
from scripts import run_ms_cxr_rad_dino_singlebox_retrain_v1 as rad_single  # noqa: E402
from scripts.models_ms_cxr_vfm_localizer import PatchHeatmapBBoxHead  # noqa: E402


DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "clueground_canonical_heatmap_candidate_rank_pilot_s13_v1"
MAX_CANDIDATES = 24


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--alpha-grid",
        type=float,
        nargs="+",
        default=(0.0, 0.25, 0.5, 0.75, 1.0),
        help="Validation-only heat-score mixing weights.",
    )
    parser.add_argument("--roi-pool", choices=("mean", "top_fraction"), default="mean")
    parser.add_argument("--roi-pool-fraction", type=float, default=0.25)
    parser.add_argument(
        "--bilateral-enumeration-query",
        action="store_true",
        help="Encode phrases containing both left and right as bilateral for RAD-DINO.",
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def candidate_key(candidate: dict[str, Any]) -> str:
    box = candidate["box"]
    return f"{candidate.get('source_model','')}|{candidate.get('rank',-1)}|" + ",".join(f"{float(value):.3f}" for value in box)


def box_norm(box: list[float], width: float, height: float) -> np.ndarray:
    return np.asarray([box[0] / width, box[1] / height, box[2] / width, box[3] / height], dtype=np.float32).clip(0.0, 1.0)


def candidate_mask(boxes: np.ndarray, token_count: int) -> np.ndarray:
    side = int(round(token_count ** 0.5))
    if side * side != token_count:
        raise RuntimeError(f"Expected square patch grid, got {token_count} tokens")
    ys, xs = np.meshgrid((np.arange(side, dtype=np.float32) + 0.5) / side, (np.arange(side, dtype=np.float32) + 0.5) / side, indexing="ij")
    xs, ys = xs.reshape(-1), ys.reshape(-1)
    mask = (xs[None] >= boxes[:, 0:1]) & (xs[None] <= boxes[:, 2:3]) & (ys[None] >= boxes[:, 1:2]) & (ys[None] <= boxes[:, 3:4])
    # Small boxes can fall between patch centers.  Assign their center patch.
    for index in np.where(~mask.any(axis=1))[0]:
        cx, cy = (boxes[index, 0] + boxes[index, 2]) * 0.5, (boxes[index, 1] + boxes[index, 3]) * 0.5
        col, row = min(side - 1, max(0, int(cx * side))), min(side - 1, max(0, int(cy * side)))
        mask[index, row * side + col] = True
    return mask


def query_features(bundle_rows: list[dict[str, Any]], bilateral_enumeration: bool) -> np.ndarray:
    if not bilateral_enumeration:
        return np.asarray(
            [rad_single.encode_query(pd.DataFrame(bundle_rows), "full_phrase")], dtype=np.float32
        )[0]

    frame = pd.DataFrame(
        {
            "finding_label": [str(row["finding"]) for row in bundle_rows],
            "phrase_text": [str(row["claim_sentence"]) for row in bundle_rows],
        }
    )
    original = rad_single.parse_location

    def parse(text: str) -> dict[str, str]:
        parsed = dict(original(text))
        value = str(text).lower()
        if re.search(r"\bleft\b", value) and re.search(r"\bright\b", value):
            parsed["laterality"] = "bilateral"
        return parsed

    rad_single.parse_location = parse
    try:
        return rad_single.encode_query(frame, "full_phrase").astype(np.float32)
    finally:
        rad_single.parse_location = original


def build_group_data(
    rows: list[dict[str, Any]],
    labels: dict[str, list[list[float]]],
    source_ids: dict[str, list[str]],
    bundle: tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray],
    candidates: dict[str, list[dict[str, Any]],],
    bilateral_enumeration: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[list[dict[str, Any]]], list[str]]:
    bundle_rows, all_tokens, all_queries, _targets = bundle
    if bilateral_enumeration:
        all_queries = query_features(bundle_rows, bilateral_enumeration=True)
    token_by_task = {str(row["task_id"]): np.asarray(token, dtype=np.float32) for row, token in zip(bundle_rows, all_tokens)}
    query_by_task = {str(row["task_id"]): np.asarray(query, dtype=np.float32) for row, query in zip(bundle_rows, all_queries)}
    tokens, queries, masks, quality, valid, metadata, group_ids = [], [], [], [], [], [], []
    for row in rows:
        gid = str(row["group_id"])
        task_id = next((str(value) for value in source_ids[gid] if str(value) in token_by_task), None)
        if task_id is None:
            raise RuntimeError(f"Missing RAD-DINO task for {gid}")
        width, height = float(row["image_width"]), float(row["image_height"])
        class_id = legacy.old_base.CLASS_TO_ID[str(row["finding"])]
        subset = [candidate for candidate in candidates.get(str(row["dicom_id"]), []) if int(candidate["class_id"]) == class_id]
        subset = sorted(subset, key=lambda candidate: (-float(candidate["score"]), str(candidate["source_model"]), int(candidate["rank"])))[:MAX_CANDIDATES]
        boxes = np.asarray([box_norm(candidate["box"], width, height) for candidate in subset], dtype=np.float32) if subset else np.zeros((0, 4), dtype=np.float32)
        values = np.asarray([max(legacy.old_fusion.base.iou_xyxy(candidate["box"], gold) for gold in labels[gid]) for candidate in subset], dtype=np.float32) if subset else np.zeros(0, dtype=np.float32)
        tokens.append(token_by_task[task_id]); queries.append(query_by_task[task_id]); masks.append(candidate_mask(boxes, all_tokens.shape[1]) if len(boxes) else np.zeros((0, all_tokens.shape[1]), dtype=bool)); quality.append(values); valid.append(np.ones(len(subset), dtype=bool)); metadata.append(subset); group_ids.append(gid)
    width = max((len(values) for values in quality), default=1)
    patch_count = all_tokens.shape[1]
    out_mask = np.zeros((len(rows), width, patch_count), dtype=bool)
    out_quality = np.zeros((len(rows), width), dtype=np.float32)
    out_valid = np.zeros((len(rows), width), dtype=bool)
    for index, (m, q, v) in enumerate(zip(masks, quality, valid)):
        out_mask[index, :len(q)] = m; out_quality[index, :len(q)] = q; out_valid[index, :len(q)] = v
    return np.stack(tokens), np.stack(queries), out_mask, out_quality, out_valid, metadata, group_ids


def roi_scores(
    logits: torch.Tensor,
    mask: torch.Tensor,
    valid: torch.Tensor,
    pool: str = "mean",
    top_fraction: float = 0.25,
) -> torch.Tensor:
    # [batch, candidates, patches] -> pooled patch heat inside each candidate box.
    expanded = logits.unsqueeze(1).expand(-1, mask.shape[1], -1)
    if pool == "mean":
        values = expanded.masked_fill(~mask, 0.0).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
    else:
        ordered, _ = expanded.masked_fill(~mask, float("-inf")).sort(dim=-1, descending=True)
        top_count = torch.ceil(mask.sum(dim=-1).clamp(min=1).float() * top_fraction).long().clamp(min=1)
        ranks = torch.arange(mask.shape[-1], device=logits.device)[None, None, :]
        take = ranks < top_count[:, :, None]
        values = ordered.masked_fill(~take, 0.0).sum(dim=-1) / top_count.float()
    return values.masked_fill(~valid, -1e4)


def train_head(train_data: tuple, val_data: tuple, upstream: Path, args: argparse.Namespace, device: torch.device) -> tuple[PatchHeatmapBBoxHead, pd.DataFrame, float]:
    train_tokens, train_queries, train_masks, train_quality, train_valid, _metadata, _ids = train_data
    val_tokens, val_queries, val_masks, val_quality, val_valid, _metadata, _ids = val_data
    payload = torch.load(upstream / "rad_dino_legacy" / "best.pt", map_location="cpu", weights_only=False)
    model = PatchHeatmapBBoxHead(train_tokens.shape[-1], train_queries.shape[-1], hidden=384, dropout=0.1).to(device)
    model.load_state_dict(payload["state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_tokens), torch.from_numpy(train_queries), torch.from_numpy(train_masks), torch.from_numpy(train_quality), torch.from_numpy(train_valid)),
        batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed), num_workers=0,
    )
    val = tuple(torch.from_numpy(value).to(device) for value in (val_tokens, val_queries, val_masks, val_quality, val_valid))
    best, stale, best_state, history = -1.0, 0, None, []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for tokens, queries, masks, quality, valid in loader:
            tokens, queries, masks, quality, valid = tokens.float().to(device), queries.float().to(device), masks.bool().to(device), quality.float().to(device), valid.bool().to(device)
            _bbox, logits = model(tokens, queries)
            scores = roi_scores(logits, masks, valid, args.roi_pool, args.roi_pool_fraction)
            target = torch.softmax(quality.masked_fill(~valid, -1e4) / 0.12, dim=1)
            loss = torch.nn.functional.kl_div(torch.log_softmax(scores, dim=1), target, reduction="batchmean")
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            _bbox, logits = model(val[0].float(), val[1].float())
            selected = roi_scores(logits, val[2].bool(), val[4].bool(), args.roi_pool, args.roi_pool_fraction).argmax(dim=1)
            score = float(val[3][torch.arange(len(selected), device=device), selected].mean().cpu())
        history.append({"epoch": epoch, "train_candidate_rank_loss": float(np.mean(losses)), "val_selected_candidate_iou": score})
        if score > best + 1e-8:
            best, stale = score, 0; best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 15:
            break
    if best_state is None:
        raise RuntimeError("No candidate-heat checkpoint")
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), best


@torch.no_grad()
def heat_by_group(
    model: PatchHeatmapBBoxHead,
    data: tuple,
    batch_size: int,
    device: torch.device,
    pool: str,
    top_fraction: float,
) -> dict[str, dict[str, float]]:
    tokens, queries, masks, _quality, valid, metadata, group_ids = data
    model.eval(); output: dict[str, dict[str, float]] = {}
    for start in range(0, len(tokens), batch_size):
        _bbox, logits = model(torch.from_numpy(tokens[start:start + batch_size]).float().to(device), torch.from_numpy(queries[start:start + batch_size]).float().to(device))
        score = roi_scores(
            logits,
            torch.from_numpy(masks[start:start + batch_size]).bool().to(device),
            torch.from_numpy(valid[start:start + batch_size]).bool().to(device),
            pool,
            top_fraction,
        ).cpu().numpy()
        for offset, gid in enumerate(group_ids[start:start + batch_size]):
            output[gid] = {candidate_key(candidate): float(score[offset, index]) for index, candidate in enumerate(metadata[start + offset])}
    return output


def adjust_candidates(rows: list[dict[str, Any]], heat: dict[str, dict[str, float]], candidates: dict[str, list[dict[str, Any]]], alpha: float) -> dict[str, list[dict[str, Any]]]:
    # The rare duplicate dicom/finding phrase uses an average phrase heat score;
    # no category, target IoU, or eval field enters inference.
    collected: dict[tuple[str, int, str], list[float]] = {}
    for row in rows:
        gid, dicom, class_id = str(row["group_id"]), str(row["dicom_id"]), int(legacy.old_base.CLASS_TO_ID[str(row["finding"])])
        for key, value in heat[gid].items():
            collected.setdefault((dicom, class_id, key), []).append(value)
    output = copy.deepcopy(candidates)
    for dicom, values in output.items():
        by_class: dict[int, list[dict[str, Any]]] = {}
        for candidate in values:
            by_class.setdefault(int(candidate["class_id"]), []).append(candidate)
        for class_id, subset in by_class.items():
            heat_values = np.asarray([np.mean(collected.get((str(dicom), class_id, candidate_key(candidate)), [0.0])) for candidate in subset], dtype=np.float32)
            z = (heat_values - heat_values.mean()) / max(float(heat_values.std()), 1e-6)
            for candidate, heat_z in zip(subset, z):
                raw = min(1.0 - 1e-5, max(1e-5, float(candidate["score"])))
                candidate["raw_yolo_score"] = raw
                candidate["dino_candidate_heat_z"] = float(heat_z)
                candidate["score"] = float(1.0 / (1.0 + np.exp(-(np.log(raw / (1.0 - raw)) + alpha * float(heat_z)))))
    return output


def alpha_selection(rows: list[dict[str, Any]], labels: dict[str, list[list[float]]], heat: dict[str, dict[str, float]], candidates: dict[str, list[dict[str, Any]]], alpha_grid: list[float]) -> tuple[float, pd.DataFrame]:
    records = []
    for alpha in alpha_grid:
        adjusted = adjust_candidates(rows, heat, candidates, alpha)
        values = []
        for row in rows:
            class_id = int(legacy.old_base.CLASS_TO_ID[str(row["finding"])])
            subset = [candidate for candidate in adjusted[str(row["dicom_id"])] if int(candidate["class_id"]) == class_id]
            if subset:
                selected = max(subset, key=lambda candidate: float(candidate["score"]))
                values.append(max(legacy.old_fusion.base.iou_xyxy(selected["box"], gold) for gold in labels[str(row["group_id"])]))
            else:
                values.append(0.0)
        records.append({"alpha": alpha, "val_raw_score_top_candidate_iou": float(np.mean(values))})
    table = pd.DataFrame(records).sort_values(["val_raw_score_top_candidate_iou", "alpha"], ascending=[False, True])
    return float(table.iloc[0].alpha), table


def main() -> None:
    args = parse_args()
    if not args.execute:
        write_json(args.output_root / "RUN_PLAN.json", {"status": "PLAN_ONLY", "seed": args.seed, "method": "candidate-supervised RAD-DINO heatmap ranking"}); return
    set_seed(args.seed); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs, labels, source_ids = legacy.load_protocol(canonical.PROTOCOL_ROOT)
    if {split: len(inputs[split]) for split in inputs} != {"train": 813, "val": 124, "eval": 220}:
        raise RuntimeError("Canonical 1444 group contract failed")
    if {split: sum(len(boxes) for boxes in labels[split].values()) for split in labels} != {"train": 996, "val": 164, "eval": 280}:
        raise RuntimeError("Canonical 1444 box contract failed")
    rows = {split: legacy.group_rows(inputs[split], labels[split], split) for split in inputs}
    upstream = canonical.UPSTREAM_ROOT / f"seed_{args.seed}" / canonical.PROTOCOL
    candidates = {split: legacy.load_yolo_candidates(upstream / "yolo_predictions", split) for split in ("train", "val", "eval")}
    allowed = {split: {str(task_id) for task_ids in source_ids[split].values() for task_id in task_ids} for split in ("train", "val", "eval")}
    bundles = {split: legacy.load_rad_bundle(split, allowed[split]) for split in ("train", "val", "eval")}
    data = {
        split: build_group_data(
            rows[split],
            labels[split],
            source_ids[split],
            bundles[split],
            candidates[split],
            args.bilateral_enumeration_query,
        )
        for split in ("train", "val", "eval")
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    model, history, best_val = train_head(data["train"], data["val"], upstream, args, device)
    history.to_csv(args.output_root / "train_log.csv", index=False)
    torch.save({"state": model.state_dict(), "head": "PatchHeatmapBBoxHead", "purpose": "candidate heat ranking", "seed": args.seed, "best_val_selected_candidate_iou": best_val}, args.output_root / "best.pt")
    heat = {
        split: heat_by_group(
            model,
            data[split],
            args.batch_size,
            device,
            args.roi_pool,
            args.roi_pool_fraction,
        )
        for split in ("val", "eval")
    }
    alpha, alpha_table = alpha_selection(
        rows["val"], labels["val"], heat["val"], candidates["val"], args.alpha_grid
    )
    alpha_table.to_csv(args.output_root / "heat_alpha_val_selection.csv", index=False)
    context = exact.load_multi_context(args.seed, protocol_root=canonical.PROTOCOL_ROOT, multi_source_root=canonical.UPSTREAM_ROOT, canonical_v3=True)
    context.candidates = {"val": adjust_candidates(rows["val"], heat["val"], candidates["val"], alpha), "eval": adjust_candidates(rows["eval"], heat["eval"], candidates["eval"], alpha)}
    run = exact.run_protocol(context, args.output_root, quick=False, retune_single_full_val=True, separate_multi_route_params=True, calibration_cache_root=None)
    run.update({
        "method": "existing YOLO-RAD-DINO hybrid with candidate-supervised RAD-DINO heat ranking",
        "new_component": "fixed candidate ROI heat ranking; coordinates and detector pools unchanged",
        "unchanged": ["four existing 640 finding-conditioned YOLO proposal generators", "RAD-DINO backbone", "candidate boxes", "fusion family", "raw-phrase cardinality decoder", "no WBF", "no CIG", "no HGB", "no new foundation"],
        "selection": "candidate-head checkpoint and alpha use val124 only; final fusion/decoder also use val124 only",
        "best_val_selected_candidate_iou": best_val,
        "selected_heat_alpha": alpha,
        "heat_alpha_grid": list(map(float, args.alpha_grid)),
        "roi_pool": args.roi_pool,
        "roi_pool_fraction": float(args.roi_pool_fraction),
        "bilateral_enumeration_query": bool(args.bilateral_enumeration_query),
        "seed_gate_pass": float(run["coverage_mean_iou"]) >= 0.5,
    })
    write_json(args.output_root / "RUN_STATUS.json", run)


if __name__ == "__main__":
    main()

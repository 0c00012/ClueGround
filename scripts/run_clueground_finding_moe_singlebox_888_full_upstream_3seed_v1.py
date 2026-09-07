#!/usr/bin/env python
"""Run the 1444 finding-conditioned semantic MoE method on direct-888.

This runner ports ``run_clueground_finding_moe_full_upstream_3seed_v1.py`` to
MS-CXR direct-888 (train638 / val87 / eval163) without using eval labels for
model or parameter selection.

Method contract
---------------
1. Reuse the seed-specific direct-888 YOLO--RAD-DINO hybrid-v4 upstream.
2. Build frozen SigLIP and BioMedCLIP crop-text experts from direct-888
   candidates. Their candidate-fusion and set parameters are selected on the
   direct-888 validation split only.
3. Train the same finding-conditioned three-expert MoE gate on direct-888 train
   and checkpoint-select it on direct-888 validation.
4. Select base-vs-gate on validation only, then evaluate the selected variant on
   eval163 with exactly one top-ranked box.

The default ``--semantic-source-mode fixed`` mirrors the historical 1444 arm:
semantic expert predictions are built once from a fixed candidate pool
(default seed 42) and reused across gate seeds, while the hybrid upstream and
MoE gate are repeated for seeds 13/42/2026. ``paired`` is provided only as an
additional diagnostic in which semantic candidate pools also follow each seed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import audit_unified_finding_annotation_dependency_v2 as finding_audit  # noqa: E402
from scripts import run_clueground_exact_hybrid_v4_direct_3seed_v1 as exact  # noqa: E402
from scripts import run_ms_cxr_multibox_rule_context_fusion_v4_yolov8l_pool as hybrid_v4  # noqa: E402
from scripts import run_ms_cxr_singlebox_fair_yolo_dino_fusion_v1 as single_fusion  # noqa: E402
from scripts import run_ms_cxr_singlebox_full_pipeline_3seed_v2 as single_source  # noqa: E402
from scripts import run_ms_cxr_trainable_moe_gate_v1 as moe  # noqa: E402


SEEDS = (13, 42, 2026)
DEFAULT_HYBRID_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "clueground_exact_hybrid_v4_route_specific_3seed_v3"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "clueground_finding_moe_singlebox_888_full_upstream_3seed_v1"
)
SIGLIP_MODEL_ID = "google/siglip-base-patch16-224"
BIOMEDCLIP_MODEL_ID = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


@dataclass
class SeedUpstream:
    context: exact.ProtocolContext
    hybrid: dict[str, dict[str, list[dict[str, Any]]]]
    cue: dict[str, pd.DataFrame]


@dataclass
class SemanticExperts:
    siglip: dict[str, dict[str, list[dict[str, Any]]]]
    biomed: dict[str, dict[str, list[dict[str, Any]]]]
    provenance: dict[str, Any]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def norm_cxcywh_to_xyxy(box: np.ndarray, width: float, height: float) -> list[float]:
    cx, cy, bw, bh = [float(value) for value in box]
    return [
        max(0.0, (cx - bw / 2.0) * width),
        max(0.0, (cy - bh / 2.0) * height),
        min(width, (cx + bw / 2.0) * width),
        min(height, (cy + bh / 2.0) * height),
    ]



def generate_candidate_csv(
    *,
    weights: Path,
    rows: list[dict[str, Any]],
    output_path: Path,
    model_tag: str,
    image_size: int,
    confidence: float,
    device: str,
) -> None:
    """Run detector inference only when the train candidate CSV is absent."""

    if not weights.exists():
        raise FileNotFoundError(weights)
    from ultralytics import YOLO

    image_by_dicom = {str(row["dicom_id"]): str(row["image_path"]) for row in rows}
    items = sorted(image_by_dicom.items())
    missing = [path for _, path in items if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images; first={missing[:3]}")

    model = YOLO(str(weights))
    results = model.predict(
        source=[path for _, path in items],
        imgsz=image_size,
        conf=confidence,
        device=device,
        verbose=False,
        stream=False,
        max_det=300,
    )
    records: list[dict[str, Any]] = []
    for (dicom_id, image_path), result in zip(items, results):
        predictions: list[dict[str, Any]] = []
        if result.boxes is not None and len(result.boxes) > 0:
            coordinates = result.boxes.xyxy.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            scores = result.boxes.conf.detach().cpu().numpy()
            for box, class_id, score in zip(coordinates, classes, scores):
                predictions.append(
                    {
                        "class_id": int(class_id),
                        "score": float(score),
                        "box": [float(value) for value in box],
                    }
                )
        predictions.sort(key=lambda item: float(item["score"]), reverse=True)
        for rank, prediction in enumerate(predictions):
            x1, y1, x2, y2 = prediction["box"]
            records.append(
                {
                    "dicom_id": dicom_id,
                    "image_path": image_path,
                    "class_id": prediction["class_id"],
                    "score": prediction["score"],
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "source_model": model_tag,
                    "rank": rank,
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "dicom_id", "image_path", "class_id", "score", "x1", "y1",
        "x2", "y2", "source_model", "rank",
    ]
    pd.DataFrame(records, columns=columns).to_csv(output_path, index=False)


def load_candidates_for_split(
    context: exact.ProtocolContext,
    seed: int,
    split: str,
    args: argparse.Namespace,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    # Optional experiment-local adapter: all established fusion, semantic,
    # decoder, and evaluator settings stay unchanged; only detector proposals
    # are replaced by a precomputed raw-1024 candidate table.
    if args.raw_yolo_candidate_root is not None:
        candidate_path = args.raw_yolo_candidate_root / f"seed_{seed}" / "raw" / f"{split}_candidates.csv"
        if not candidate_path.exists():
            raise FileNotFoundError(f"Missing raw 1024 candidate file: {candidate_path}")
        frame = pd.read_csv(candidate_path)
        required = {"dicom_id", "class_id", "score", "x1", "y1", "x2", "y2", "source_model", "rank"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise RuntimeError(f"Invalid raw 1024 candidate schema {candidate_path}: missing {missing}")
        if frame["source_model"].astype(str).str.contains("wbf", case=False, na=False).any():
            raise RuntimeError(f"WBF candidate row detected in raw-only input: {candidate_path}")
        candidates: dict[str, list[dict[str, Any]]] = {}
        for row in frame.to_dict("records"):
            candidates.setdefault(str(row["dicom_id"]), []).append({
                "box": [float(row[k]) for k in ("x1", "y1", "x2", "y2")],
                "score": float(row["score"]),
                "class_id": int(row["class_id"]),
                "rank": int(row["rank"]),
                "source_model": str(row["source_model"]),
            })
        return candidates, [{
            "seed": seed, "split": split, "model": "four_model_raw_1024_ensemble",
            "candidate_path": str(candidate_path), "candidate_sha256": sha256(candidate_path),
            "candidate_mode": "raw", "imgsz": 1024,
        }]

    parts = []
    provenance = []
    for model in exact.MODELS:
        candidate_path, weight_path = single_source.yolo_artifacts(seed, split, model)
        if not candidate_path.exists():
            if not args.generate_missing_candidates:
                raise FileNotFoundError(
                    f"Missing {candidate_path}. Re-run with --generate-missing-candidates."
                )
            generate_candidate_csv(
                weights=weight_path,
                rows=context.rows[split],
                output_path=candidate_path,
                model_tag=model,
                image_size=args.image_size,
                confidence=args.detector_confidence,
                device=args.yolo_device,
            )
        parts.append(single_source.load_candidates(candidate_path))
        provenance.append(
            {
                "seed": seed,
                "split": split,
                "model": model,
                "candidate_path": str(candidate_path),
                "candidate_sha256": sha256(candidate_path),
                "weight_path": str(weight_path),
                "weight_sha256": sha256(weight_path),
            }
        )
    return single_fusion.merge_candidates(*parts), provenance


def load_single_context_with_train(seed: int, args: argparse.Namespace) -> exact.ProtocolContext:
    context = exact.load_single_context(seed)
    # The legacy context preloads val/eval. Override all splits so validation
    # and evaluation cannot accidentally retain 640 proposals.
    for split in ("train", "val", "eval"):
        candidates, provenance = load_candidates_for_split(context, seed, split, args)
        context.candidates[split] = candidates
        context.provenance.setdefault("candidate_artifacts", []).extend(provenance)

    single_source.configure_rad_paths(seed)
    dino_train_frame = single_fusion.load_rad_prediction(
        "full_phrase",
        "train",
        args.device,
        args.force_rad_predictions,
    )
    context.dino["train"] = single_fusion.dino_map(dino_train_frame)
    return context


def load_hybrid_upstream(
    seed: int,
    args: argparse.Namespace,
) -> SeedUpstream:
    context = load_single_context_with_train(seed, args)
    restored_multi_yolo = copy.deepcopy(context.yolo_params)
    run_root = args.hybrid_root / "singlebox_888" / f"seed_{seed}"
    calibration_root = run_root / "single_route_fullval_calibration"
    required = [
        calibration_root / "yolo_params.json",
        calibration_root / "fusion_params.json",
        run_root / "selected_set_params.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing hybrid-v4 validation artifacts: {missing}")
    context.yolo_params = json.loads(
        (calibration_root / "yolo_params.json").read_text(encoding="utf-8")
    )
    context.fusion_params = json.loads(
        (calibration_root / "fusion_params.json").read_text(encoding="utf-8")
    )
    set_params = json.loads(
        (run_root / "selected_set_params.json").read_text(encoding="utf-8")
    )

    hybrid_maps: dict[str, dict[str, list[dict[str, Any]]]] = {}
    cue_maps: dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "eval"):
        outputs, cue = exact.run_hybrid(
            context,
            split,
            set_params,
            restored_multi_yolo,
        )
        hybrid_maps[split] = {
            str(group_id): [
                {
                    "box": [float(value) for value in box],
                    "score": float(max(0.0, 1.0 - index * 1e-6)),
                    "source": f"direct888_hybrid_v4_seed_{seed}",
                }
                for index, box in enumerate(boxes)
            ]
            for group_id, boxes in outputs.items()
        }
        cue_maps[split] = cue.copy()

    context.provenance["hybrid_artifact_root"] = str(run_root)
    context.provenance["selected_set_params"] = set_params
    return SeedUpstream(context=context, hybrid=hybrid_maps, cue=cue_maps)


def _candidate_table_for_split(
    upstream: SeedUpstream,
    split: str,
    max_candidates_per_task: int,
) -> pd.DataFrame:
    context = upstream.context
    groups = exact.make_groups(context, split)
    set_params = context.provenance["selected_set_params"]
    records: list[dict[str, Any]] = []
    for group_id, group in groups.items():
        finding = str(group["finding"])
        params = dict(
            context.yolo_params.get(
                finding,
                context.yolo_params.get("__global__", {}),
            )
        )
        query = single_fusion.ybase.parse_rule_context(hybrid_v4.group_row(group))
        class_id = single_fusion.CLASS_TO_ID[finding]
        raw = [
            dict(candidate)
            for candidate in context.candidates[split].get(str(group["dicom_id"]), [])
            if int(candidate["class_id"]) == class_id
        ]
        dino_norm = context.dino[split].get(str(group_id))
        if dino_norm is not None:
            raw.append(
                {
                    "class_id": class_id,
                    "score": 0.0,
                    "box": norm_cxcywh_to_xyxy(
                        dino_norm,
                        float(group["image_width"]),
                        float(group["image_height"]),
                    ),
                    "source_model": "rad_dino",
                    "rank": 0,
                }
            )
        scored = hybrid_v4.score_candidates(
            group,
            query,
            raw,
            context.priors,
            params,
            dino_norm,
            float(set_params.get("dino_weight", 0.0)),
        )
        # Preserve a RAD-DINO candidate even when many YOLO candidates rank
        # above it. The semantic experts are defined over a YOLO/RAD-DINO pool.
        dino_scored = [row for row in scored if str(row.get("source_model", "")) == "rad_dino"]
        yolo_scored = [row for row in scored if str(row.get("source_model", "")) != "rad_dino"]
        keep_yolo = max_candidates_per_task - (1 if dino_scored else 0)
        scored = yolo_scored[:max(0, keep_yolo)] + dino_scored[:1]
        scored.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
        gold = group["gt_boxes"][0]
        for rank, candidate in enumerate(scored):
            x1, y1, x2, y2 = [float(value) for value in candidate["box"]]
            records.append(
                {
                    "split": split,
                    "task_id": str(group_id),
                    "group_id": str(group_id),
                    "dicom_id": str(group["dicom_id"]),
                    "subject_id": str(group.get("subject_id", "")),
                    "study_id": str(group.get("study_id", "")),
                    "image_path": str(group["image_path"]),
                    "image_width": int(group["image_width"]),
                    "image_height": int(group["image_height"]),
                    "finding": finding,
                    "claim_sentence": str(group["claim_sentence"]),
                    "pred_x1": x1,
                    "pred_y1": y1,
                    "pred_x2": x2,
                    "pred_y2": y2,
                    # Only validation candidates carry gold coordinates,
                    # because validation alone selects semantic fusion weights.
                    # Eval semantic-score artifacts therefore contain no gold box.
                    "gt_x1": float(gold[0]) if split == "val" else np.nan,
                    "gt_y1": float(gold[1]) if split == "val" else np.nan,
                    "gt_x2": float(gold[2]) if split == "val" else np.nan,
                    "gt_y2": float(gold[3]) if split == "val" else np.nan,
                    "score_head": float(candidate.get("score", 0.0)),
                    "confidence": float(candidate.get("raw_conf", candidate.get("score", 0.0))),
                    "prior_iou": float(candidate.get("prior_iou", 0.0)),
                    "xattn_iou": float(candidate.get("dino_iou", 0.0)),
                    "source_model": str(candidate.get("source_model", "")),
                    "rank": int(candidate.get("rank", rank)),
                    "semantic_pool_rank": rank,
                }
            )
    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError(f"No semantic candidate rows for {split}")
    return frame


def _load_or_score_siglip(
    base: pd.DataFrame,
    path: Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if path.exists() and not args.force_semantic:
        return pd.read_csv(path)
    from scripts import run_ms_cxr_siglip_candidate_fusion_v1 as siglip

    scored = siglip.score_siglip(
        base,
        args.siglip_model_id,
        args.prompt_mode,
        args.crop_margin,
        args.semantic_batch_size,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(path, index=False)
    return scored


def _load_or_score_biomed(
    base: pd.DataFrame,
    path: Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if path.exists() and not args.force_semantic:
        return pd.read_csv(path)
    from scripts import run_ms_cxr_biomedclip_gated_hybrid_v1 as biomed

    scored = biomed.score_biomedclip(
        base,
        args.biomedclip_model_id,
        args.prompt_mode,
        args.crop_margin,
        args.semantic_batch_size,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(path, index=False)
    return scored


def _semantic_prediction_map(
    scored_by_split: dict[str, pd.DataFrame],
    groups_by_split: dict[str, dict[str, dict[str, Any]]],
    score_name: str,
    output_root: Path,
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    from scripts import run_ms_cxr_siglip_candidate_fusion_v1 as semantic

    # BioMedCLIP's original helper reuses the generic siglip_z/rank field names
    # for the common fusion grid. Restore those aliases when loading an older
    # cache that contains only biomedclip_* columns.
    if score_name == "biomedclip":
        for frame in scored_by_split.values():
            for suffix in ("raw", "z", "rank", "sigmoid"):
                generic = f"siglip_{suffix}"
                specific = f"biomedclip_{suffix}"
                if generic not in frame.columns and specific in frame.columns:
                    frame[generic] = frame[specific]

    row_params, row_grid = semantic.tune_row_params(scored_by_split["val"])
    row_grid.to_csv(output_root / f"{score_name}_row_fusion_val_grid.csv", index=False)
    write_json(output_root / f"{score_name}_row_fusion_params.json", row_params)

    fusion_column = f"{score_name}_fusion_score"
    fused = {
        split: semantic.apply_fusion_score(frame, row_params, fusion_column)
        for split, frame in scored_by_split.items()
    }
    for split, frame in fused.items():
        frame.to_csv(output_root / f"{split}_{score_name}_fusion_scored_candidates.csv", index=False)

    set_params, set_grid = semantic.tune_set_params(
        groups_by_split["val"],
        fused["val"],
        fusion_column,
    )
    set_grid.to_csv(output_root / f"{score_name}_set_val_grid.csv", index=False)
    write_json(output_root / f"{score_name}_set_params.json", set_params)

    predictions: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for split in ("train", "val", "eval"):
        candidates = semantic.scored_candidates_by_group(
            fused[split],
            groups_by_split[split],
            fusion_column,
        )
        predictions[split] = semantic.predict_phrase_sets(
            groups_by_split[split],
            candidates,
            set_params,
        )
    return predictions, {
        "row_fusion_params": row_params,
        "set_params": set_params,
        "score_column": fusion_column,
    }


def prepare_semantic_experts(
    upstream: SeedUpstream,
    source_seed: int,
    args: argparse.Namespace,
) -> SemanticExperts:
    root = args.output_root / "semantic_experts" / f"source_seed_{source_seed}"
    root.mkdir(parents=True, exist_ok=True)
    protocol = str(getattr(args, "semantic_protocol", "singlebox_888"))
    selection_split = str(
        getattr(args, "semantic_selection_split", "direct-888 validation only")
    )
    expected_contract = {
        "protocol": protocol,
        "semantic_source_seed": source_seed,
        "hybrid_root": str(args.hybrid_root.resolve()),
        "siglip_model_id": args.siglip_model_id,
        "biomedclip_model_id": args.biomedclip_model_id,
        "prompt_mode": args.prompt_mode,
        "crop_margin": float(args.crop_margin),
        "max_candidates_per_task": int(args.max_candidates_per_task),
        "selection_split": "val only",
    }
    contract_path = root / "SEMANTIC_CACHE_CONTRACT.json"
    if contract_path.exists() and not (args.force_semantic or args.force_semantic_candidates):
        observed_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if observed_contract != expected_contract:
            raise RuntimeError(
                "Semantic cache contract changed. Use --force-semantic and "
                "--force-semantic-candidates to rebuild it. "
                f"observed={observed_contract}, expected={expected_contract}"
            )
    groups = {
        split: exact.make_groups(upstream.context, split)
        for split in ("train", "val", "eval")
    }
    base_frames: dict[str, pd.DataFrame] = {}
    siglip_frames: dict[str, pd.DataFrame] = {}
    biomed_frames: dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "eval"):
        base_path = root / f"{split}_semantic_base_candidates.csv"
        if base_path.exists() and not args.force_semantic_candidates:
            base = pd.read_csv(base_path)
        else:
            base = _candidate_table_for_split(
                upstream,
                split,
                args.max_candidates_per_task,
            )
            base.to_csv(base_path, index=False)
        base_frames[split] = base
        siglip_frames[split] = _load_or_score_siglip(
            base.copy(),
            root / f"{split}_siglip_scored_candidates.csv",
            args,
        )
        biomed_frames[split] = _load_or_score_biomed(
            base.copy(),
            root / f"{split}_biomedclip_scored_candidates.csv",
            args,
        )

    siglip_map, siglip_params = _semantic_prediction_map(
        siglip_frames,
        groups,
        "siglip",
        root,
    )
    biomed_map, biomed_params = _semantic_prediction_map(
        biomed_frames,
        groups,
        "biomedclip",
        root,
    )
    provenance = {
        "semantic_source_seed": source_seed,
        "selection_split": selection_split,
        "siglip_model_id": args.siglip_model_id,
        "biomedclip_model_id": args.biomedclip_model_id,
        "prompt_mode": args.prompt_mode,
        "crop_margin": args.crop_margin,
        "max_candidates_per_task": args.max_candidates_per_task,
        "siglip": siglip_params,
        "biomedclip": biomed_params,
    }
    write_json(contract_path, expected_contract)
    write_json(root / "SEMANTIC_EXPERT_STATUS.json", provenance)
    return SemanticExperts(siglip=siglip_map, biomed=biomed_map, provenance=provenance)


def make_bundle(
    upstream: SeedUpstream,
    semantic: SemanticExperts,
    split: str,
) -> moe.ExpertBundle:
    groups = exact.make_groups(upstream.context, split)
    expected = set(groups)
    for name, mapping in (
        ("hybrid", upstream.hybrid[split]),
        ("siglip", semantic.siglip[split]),
        ("biomed", semantic.biomed[split]),
    ):
        missing = expected.difference(mapping)
        if missing:
            raise RuntimeError(f"{name}/{split} missing {len(missing)} groups; first={sorted(missing)[:3]}")
    return moe.ExpertBundle(
        groups=groups,
        hybrid=upstream.hybrid[split],
        siglip=semantic.siglip[split],
        biomed=semantic.biomed[split],
        pretrain={},
        cue=upstream.cue[split],
    )


def finding_gate_feature(
    bundle: moe.ExpertBundle,
    group_id: str,
    experts: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    result = finding_audit._legacy_gate_feature(
        bundle,
        group_id,
        experts,
        finding_audit.FINDINGS,
    )
    if result is None:
        return None
    feature, boxes = result
    return feature, boxes, np.zeros(len(experts), dtype=np.float32)


def prediction_maps_equal(
    left: dict[str, list[dict[str, Any]]],
    right: dict[str, list[dict[str, Any]]],
    tolerance: float = 1e-8,
) -> bool:
    if set(left) != set(right):
        return False
    for group_id in left:
        a = left[group_id]
        b = right[group_id]
        if len(a) != len(b):
            return False
        for row_a, row_b in zip(a, b):
            if np.max(
                np.abs(
                    np.asarray(row_a["box"], dtype=float)
                    - np.asarray(row_b["box"], dtype=float)
                )
            ) > tolerance:
                return False
    return True


def direct888_detail(
    method: str,
    bundle: moe.ExpertBundle,
    predictions: dict[str, list[dict[str, Any]]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for group_id, group in bundle.groups.items():
        expert_rows = predictions.get(group_id, [])
        pred_box = expert_rows[0]["box"] if expert_rows else None
        gold_box = group["gt_boxes"][0]
        iou = 0.0 if pred_box is None else float(moe.mb.iou_xyxy(pred_box, gold_box))
        rows.append(
            {
                "method": method,
                "group_id": group_id,
                "subject_id": str(group.get("subject_id", "")),
                "finding": str(group["finding"]),
                "phrase": str(group["claim_sentence"]),
                "n_raw_predictions": len(expert_rows),
                "pred_boxes_json": json.dumps(
                    [[float(value) for value in row["box"]] for row in expert_rows],
                    ensure_ascii=False,
                ),
                "top1_iou": iou,
                "hit_0_3": float(iou >= 0.3),
                "hit_0_5": float(iou >= 0.5),
            }
        )
    detail = pd.DataFrame(rows)
    summary = {
        "method": method,
        "n": int(len(detail)),
        "mean_iou": float(detail["top1_iou"].mean()),
        "Hit@0.3": float(detail["hit_0_3"].mean()),
        "Hit@0.5": float(detail["hit_0_5"].mean()),
        "mean_raw_pred_count": float(detail["n_raw_predictions"].mean()),
    }
    return detail, summary


def save_evaluation(
    root: Path,
    method: str,
    bundle: moe.ExpertBundle,
    predictions: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    detail, summary = direct888_detail(method, bundle, predictions)
    detail.to_csv(root / "per_group.csv", index=False)
    write_json(root / "summary.json", summary)
    return summary


def gate_action_audit(
    bundle: moe.ExpertBundle,
    gated: dict[str, list[dict[str, Any]]],
) -> pd.DataFrame:
    multi = moe.has_multi_cue(bundle.cue)
    rows = []
    for group_id in bundle.groups:
        base = bundle.hybrid.get(group_id, [])
        pred = gated.get(group_id, [])
        changed = not prediction_maps_equal(
            {group_id: base},
            {group_id: pred},
        )
        rows.append(
            {
                "group_id": group_id,
                "finding": bundle.groups[group_id]["finding"],
                "phrase": bundle.groups[group_id]["claim_sentence"],
                "has_multi_cue": bool(multi.get(group_id, False)),
                "eligible": finding_gate_feature(bundle, group_id, ["hybrid", "siglip", "biomed"]) is not None,
                "action": "finding_conditioned_moe_blend" if changed else "keep_hybrid",
            }
        )
    return pd.DataFrame(rows)


def run_gate_seed(
    seed: int,
    upstream: SeedUpstream,
    semantic: SemanticExperts,
    args: argparse.Namespace,
) -> dict[str, Any]:
    set_seed(seed)
    bundles = {
        split: make_bundle(upstream, semantic, split)
        for split in ("train", "val", "eval")
    }
    seed_root = args.output_root / f"seed_{seed}"
    moe.CKPT = seed_root / "checkpoints"
    moe.LOG = seed_root / "logs"
    moe.MET = seed_root / "training_metrics"
    moe.PRED = seed_root / "predictions"
    for path in (moe.CKPT, moe.LOG, moe.MET, moe.PRED):
        path.mkdir(parents=True, exist_ok=True)

    experts = ["hybrid", "siglip", "biomed"]
    model, params, _, _ = moe.train_gate(
        f"finding_moe_direct888_s{seed}",
        bundles["train"],
        bundles["val"],
        experts,
        hardneg=False,
        seed=seed,
        device=args.device,
    )
    val_gate = finding_audit._legacy_predict(
        model,
        bundles["val"],
        finding_audit.FINDINGS,
    )
    eval_gate = finding_audit._legacy_predict(
        model,
        bundles["eval"],
        finding_audit.FINDINGS,
    )

    val_base_summary = save_evaluation(
        seed_root / "val" / "base",
        f"base_s{seed}",
        bundles["val"],
        bundles["val"].hybrid,
    )
    val_gate_summary = save_evaluation(
        seed_root / "val" / "gate",
        f"gate_s{seed}",
        bundles["val"],
        val_gate,
    )
    gate_key = (
        float(val_gate_summary["mean_iou"]),
        float(val_gate_summary["Hit@0.5"]),
        float(val_gate_summary["Hit@0.3"]),
    )
    base_key = (
        float(val_base_summary["mean_iou"]),
        float(val_base_summary["Hit@0.5"]),
        float(val_base_summary["Hit@0.3"]),
    )
    use_gate = gate_key >= base_key
    selected = eval_gate if use_gate else bundles["eval"].hybrid
    selected_name = "gate" if use_gate else "base"
    eval_summary = save_evaluation(
        seed_root / "eval" / selected_name,
        f"selected_{selected_name}_s{seed}",
        bundles["eval"],
        selected,
    )
    gate_action_audit(bundles["eval"], eval_gate).to_csv(
        seed_root / "eval" / "gate_action_audit.csv",
        index=False,
    )

    mutated = copy.deepcopy(bundles["eval"])
    for group in mutated.groups.values():
        group["gt_boxes"] = [[0.0, 0.0, 1.0, 1.0]]
    mutated_predictions = finding_audit._legacy_predict(
        model,
        mutated,
        finding_audit.FINDINGS,
    )
    gold_independent = prediction_maps_equal(eval_gate, mutated_predictions)

    result = {
        "status": "complete",
        "protocol": "singlebox_888",
        "seed": seed,
        "selected_variant": selected_name,
        "selection_split": "direct-888 validation only",
        "gate_best_epoch": int(params["epoch"]),
        "train_gate_rows": int(params["train_rows"]),
        "val_gate_rows": int(params["val_rows"]),
        "val_base_mean_iou": float(val_base_summary["mean_iou"]),
        "val_gate_mean_iou": float(val_gate_summary["mean_iou"]),
        "mean_iou": float(eval_summary["mean_iou"]),
        "Hit@0.3": float(eval_summary["Hit@0.3"]),
        "Hit@0.5": float(eval_summary["Hit@0.5"]),
        "mean_raw_pred_count": float(eval_summary["mean_raw_pred_count"]),
        "gold_mutation_independence": bool(gold_independent),
        "upstream_provenance": upstream.context.provenance,
        "semantic_provenance": semantic.provenance,
    }
    write_json(seed_root / "RUN_STATUS.json", result)
    return result


def split_audit(upstreams: Iterable[SeedUpstream]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for upstream in upstreams:
        context = upstream.context
        subjects = {
            split: {str(row["subject_id"]) for row in context.inputs[split]}
            for split in ("train", "val", "eval")
        }
        result[f"seed_{context.seed}"] = {
            "counts": {split: len(context.rows[split]) for split in ("train", "val", "eval")},
            "train_val_subject_overlap": len(subjects["train"] & subjects["val"]),
            "train_eval_subject_overlap": len(subjects["train"] & subjects["eval"]),
            "val_eval_subject_overlap": len(subjects["val"] & subjects["eval"]),
        }
    expected = {"train": 638, "val": 87, "eval": 163}
    passed = all(
        row["counts"] == expected
        and row["train_val_subject_overlap"] == 0
        and row["train_eval_subject_overlap"] == 0
        and row["val_eval_subject_overlap"] == 0
        for row in result.values()
    )
    result["status"] = "PASS" if passed else "FAIL"
    return result


def aggregate(frame: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {
        "status": "complete",
        "method": "finding-conditioned YOLO-RAD-DINO + frozen SigLIP + frozen BioMedCLIP MoE",
        "protocol": "singlebox_888",
        "n_upstream_seeds": int(len(frame)),
        "n_gate_seeds": int(len(frame)),
        "seeds": [int(value) for value in frame["seed"].tolist()],
        "selection_split": "direct-888 validation only",
        "uses_eval_for_selection": False,
        "top_ranked_box_only": True,
        "selected_variants": frame["selected_variant"].tolist(),
        "gold_mutation_independence_pass": bool(frame["gold_mutation_independence"].all()),
    }
    for metric in ("mean_iou", "Hit@0.3", "Hit@0.5", "mean_raw_pred_count"):
        values = frame[metric].astype(float).to_numpy()
        output[f"{metric}_mean"] = float(values.mean())
        output[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--hybrid-root", type=Path, default=DEFAULT_HYBRID_ROOT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument(
        "--semantic-source-mode",
        choices=("fixed", "paired"),
        default="fixed",
        help="fixed mirrors the 1444 arm; paired is a seed-matched diagnostic",
    )
    parser.add_argument("--semantic-source-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--yolo-device", default="0")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--detector-confidence", type=float, default=0.001)
    parser.add_argument("--generate-missing-candidates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-rad-predictions", action="store_true")
    parser.add_argument("--max-candidates-per-task", type=int, default=12)
    parser.add_argument("--semantic-batch-size", type=int, default=32)
    parser.add_argument(
        "--raw-yolo-candidate-root",
        type=Path,
        default=None,
        help="Root containing seed_<n>/raw/{train,val,eval}_candidates.csv; WBF rows are rejected.",
    )
    parser.add_argument("--prompt-mode", choices=("claim", "cxr_claim", "finding_claim", "region_prompt"), default="cxr_claim")
    parser.add_argument("--crop-margin", type=float, default=0.15)
    parser.add_argument("--siglip-model-id", default=SIGLIP_MODEL_ID)
    parser.add_argument("--biomedclip-model-id", default=BIOMEDCLIP_MODEL_ID)
    parser.add_argument("--force-semantic", action="store_true")
    parser.add_argument("--force-semantic-candidates", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_root / "RUN_CONFIG.json",
        {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "method_contract": "1444 finding-conditioned semantic MoE ported to direct-888",
        },
    )

    original_feature_builder = moe.gate_feature_for_group
    moe.gate_feature_for_group = finding_gate_feature
    upstream_cache: dict[int, SeedUpstream] = {}
    semantic_cache: dict[int, SemanticExperts] = {}
    results: list[dict[str, Any]] = []
    try:
        required_upstream_seeds = set(args.seeds)
        if args.semantic_source_mode == "fixed":
            required_upstream_seeds.add(args.semantic_source_seed)
        for seed in sorted(required_upstream_seeds):
            upstream_cache[seed] = load_hybrid_upstream(seed, args)

        for seed in args.seeds:
            semantic_seed = args.semantic_source_seed if args.semantic_source_mode == "fixed" else seed
            if semantic_seed not in semantic_cache:
                semantic_cache[semantic_seed] = prepare_semantic_experts(
                    upstream_cache[semantic_seed],
                    semantic_seed,
                    args,
                )
            results.append(
                run_gate_seed(
                    seed,
                    upstream_cache[seed],
                    semantic_cache[semantic_seed],
                    args,
                )
            )
    finally:
        moe.gate_feature_for_group = original_feature_builder

    frame = pd.DataFrame(results)
    scalar_columns = [
        column
        for column in frame.columns
        if not frame[column].map(lambda value: isinstance(value, (dict, list))).any()
    ]
    frame[scalar_columns].to_csv(args.output_root / "per_seed_metrics.csv", index=False)
    final = aggregate(frame)
    final["semantic_source_mode"] = args.semantic_source_mode
    final["semantic_source_seed"] = (
        args.semantic_source_seed if args.semantic_source_mode == "fixed" else None
    )
    final["seed_pairing"] = [
        {
            "hybrid_seed": int(seed),
            "gate_seed": int(seed),
            "semantic_candidate_seed": int(
                args.semantic_source_seed if args.semantic_source_mode == "fixed" else seed
            ),
        }
        for seed in args.seeds
    ]
    audit = split_audit(upstream_cache[seed] for seed in args.seeds)
    final["split_overlap_audit"] = audit
    final["status"] = (
        "PASS"
        if audit["status"] == "PASS"
        and final["gold_mutation_independence_pass"]
        and len(results) == len(args.seeds)
        else "FAIL"
    )
    write_json(args.output_root / "split_overlap_audit.json", audit)
    write_json(args.output_root / "FINAL_STATUS.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

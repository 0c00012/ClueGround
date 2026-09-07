#!/usr/bin/env python
"""Direct-888 upstream helpers shared by the SigLIP2 experiment runner."""

from __future__ import annotations

import copy
import gc
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

from scripts import run_clueground_exact_hybrid_v4_direct_3seed_v1 as exact  # noqa: E402
from scripts import run_ms_cxr_multibox_rule_context_fusion_v4_yolov8l_pool as hybrid_v4  # noqa: E402
from scripts import run_ms_cxr_singlebox_fair_yolo_dino_fusion_v1 as single_fusion  # noqa: E402
from scripts import run_ms_cxr_singlebox_full_pipeline_3seed_v2 as single_source  # noqa: E402
from scripts import clueground_siglip2_gate_v1 as gate  # noqa: E402


DEFAULT_HYBRID_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "clueground_exact_hybrid_v4_route_specific_3seed_v3"
)


@dataclass
class SeedUpstream:
    context: exact.ProtocolContext
    hybrid: dict[str, dict[str, list[dict[str, Any]]]]
    cue: dict[str, pd.DataFrame]


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
    if not weights.exists():
        raise FileNotFoundError(weights)
    from ultralytics import YOLO

    image_by_dicom = {str(row["dicom_id"]): str(row["image_path"]) for row in rows}
    items = sorted(image_by_dicom.items())
    missing = [path for _, path in items if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images; first={missing[:3]}")

    model = YOLO(str(weights))
    # IMPORTANT: stream=True prevents Ultralytics from retaining the full
    # train-split result list and its associated tensors in memory.  With a
    # Python list of hundreds of image paths, stream=False can create a very
    # large inference batch / retained result set and OOM before SigLIP2 runs.
    results = model.predict(
        source=[path for _, path in items],
        imgsz=image_size,
        conf=confidence,
        device=device,
        verbose=False,
        stream=True,
        batch=1,
        half=(str(device).lower() not in {"cpu", "mps"} and torch.cuda.is_available()),
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

    # Release each YOLO model before the next detector is loaded.
    del results
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_candidates_for_split(
    context: exact.ProtocolContext,
    seed: int,
    split: str,
    args: Any,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    if getattr(args, "raw_yolo_candidate_root", None) is not None:
        candidate_path = (
            Path(args.raw_yolo_candidate_root)
            / f"seed_{seed}"
            / "raw"
            / f"{split}_candidates.csv"
        )
        if not candidate_path.exists():
            raise FileNotFoundError(f"Missing raw 1024 candidate file: {candidate_path}")
        frame = pd.read_csv(candidate_path)
        required = {"dicom_id", "class_id", "score", "x1", "y1", "x2", "y2", "source_model", "rank"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise RuntimeError(f"Invalid raw 1024 schema {candidate_path}: missing {missing}")
        if frame["source_model"].astype(str).str.contains("wbf", case=False, na=False).any():
            raise RuntimeError(f"WBF candidate detected in raw-only input: {candidate_path}")
        candidates: dict[str, list[dict[str, Any]]] = {}
        for row in frame.to_dict("records"):
            candidates.setdefault(str(row["dicom_id"]), []).append({
                "box": [float(row[key]) for key in ("x1", "y1", "x2", "y2")],
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


def load_single_context_with_train(seed: int, args: Any) -> exact.ProtocolContext:
    context = exact.load_single_context(seed)
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


def load_hybrid_upstream(seed: int, args: Any) -> SeedUpstream:
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


def candidate_table_for_split(
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
        dino_scored = [row for row in scored if str(row.get("source_model", "")) == "rad_dino"]
        yolo_scored = [row for row in scored if str(row.get("source_model", "")) != "rad_dino"]
        keep_yolo = max_candidates_per_task - (1 if dino_scored else 0)
        scored = yolo_scored[: max(0, keep_yolo)] + dino_scored[:1]
        scored.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
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
    bundle: gate.ExpertBundle,
    predictions: dict[str, list[dict[str, Any]]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for group_id, group in bundle.groups.items():
        expert_rows = predictions.get(group_id, [])
        pred_box = expert_rows[0]["box"] if expert_rows else None
        gold_box = group["gt_boxes"][0]
        iou = 0.0 if pred_box is None else float(gate.mb.iou_xyxy(pred_box, gold_box))
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
    bundle: gate.ExpertBundle,
    predictions: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    detail, summary = direct888_detail(method, bundle, predictions)
    detail.to_csv(root / "per_group.csv", index=False)
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


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

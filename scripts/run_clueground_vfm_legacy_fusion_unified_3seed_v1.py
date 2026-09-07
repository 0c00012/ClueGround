#!/usr/bin/env python
"""Run the original finding-conditioned YOLO--RAD-DINO fusion as one set model.

This is the minimal 888/1444 unification requested for ClueGround-VFM:

* keep the eight-class MS-CXR YOLOv8s/m + YOLO11s/m proposal ensemble;
* keep the original phrase/rule-context RAD-DINO shallow bbox head;
* keep the original validation-tuned YOLO--DINO candidate scoring equation;
* change only the final decoder so explicit multi-region phrases may emit
  additional distinct candidates;
* generate one eval prediction artifact and derive both the 1444 set metrics
  and the singleton-163 top-1 metrics from it.

All trainable components are repeated for seeds 13, 42, and 2026.  The
RAD-DINO backbone and public YOLO initializations remain frozen provenance,
while the YOLO detectors and RAD-DINO shallow head are trained per seed.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "CLUEGROUND_VFM_METHOD_PACKAGE_20260713"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, PACKAGE_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Import the packaged, finding-conditioned ``src`` tree before legacy scripts.
# Several legacy scripts import the repository-level ``src`` package; loading
# them first would silently select the old class-agnostic pipeline.
from src.three_task_grounding.contracts import get_protocol  # noqa: E402
from src.three_task_grounding.manifests import read_jsonl, write_jsonl  # noqa: E402
from src.three_task_grounding.metrics import singleton_projection, summarize_protocol  # noqa: E402
from src.three_task_grounding.pipeline import TaskRunConfig, ThreeTaskPipeline  # noqa: E402
from src.rerank.cue_aware_set_selector import candidate_laterality, target_match_score  # noqa: E402
from src.rerank.multibox_cue_parser import (  # noqa: E402
    cue_info_from_groups,
    cue_info_from_groups_precision,
)
from src.rerank.unified_adaptive_cardinality import classify_cardinality_cue  # noqa: E402
from scripts import run_ms_cxr_rad_dino_singlebox_retrain_v1 as rad_single  # noqa: E402
from scripts import run_ms_cxr_singlebox_fair_yolo_dino_fusion_v1 as old_base  # noqa: E402
from scripts import run_ms_cxr_yolo_dino_rule_fusion_v1 as old_fusion  # noqa: E402
from scripts.models_ms_cxr_vfm_localizer import PatchHeatmapBBoxHead, bbox_loss  # noqa: E402


PROTOCOL_KEY = "mscxr_multibox_1444"
DEFAULT_PROTOCOL_ROOT = (
    PROJECT_ROOT
    / "training"
    / "three_task_clueground_vfm_finding_conditioned_v2"
    / "protocols"
    / "task_isolated"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "clueground_vfm_legacy_fusion_unified_3seed_v1"
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "clueground_vfm_strict_local888_1444_v1.json"
STAGE1_ROWS = PROJECT_ROOT / "training" / "ms_cxr_vfm_localizer_stage1_p10_p19" / "datasets"
STAGE1_FEATURES = PROJECT_ROOT / "features" / "ms_cxr_vfm_localizer_stage1_p10_p19"
SEEDS = (13, 42, 2026)
YOLO_MODELS = ("yolov8s.pt", "yolov8m.pt", "yolo11s.pt", "yolo11m.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-root", type=Path, default=DEFAULT_PROTOCOL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument(
        "--yolo-models",
        nargs="+",
        default=list(YOLO_MODELS),
        help="Proposal models for an isolated upstream-pool experiment. The default is the canonical four-detector recipe.",
    )
    parser.add_argument("--yolo-epochs", type=int, default=100)
    parser.add_argument("--rad-epochs", type=int, default=120)
    parser.add_argument("--rad-batch", type=int, default=24)
    parser.add_argument(
        "--precision-multibox-cues",
        action="store_true",
        help="Use high-precision phrase cues only for hard multi-box floors.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--canonical-v3",
        action="store_true",
        help="Require the canonical 813/124/220 finding-conditioned bundle and filter RAD-DINO rows to its exact source-task membership.",
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def phrase_only(query_text: str) -> str:
    marker = "; phrase "
    return str(query_text).split(marker, 1)[1] if marker in str(query_text) else str(query_text)


def _input_query_key(dicom_id: str, finding: str, phrase: str) -> tuple[str, str, str]:
    normalized = " ".join(str(phrase).lower().split())
    return str(dicom_id), str(finding), normalized


def resolve_source_task_ids_from_inputs(inputs: list[dict[str, Any]], split: str) -> dict[str, list[str]]:
    """Resolve cache rows from image/finding/raw phrase, never protocol labels.

    Multi-box protocol labels retain source task IDs for provenance, but they
    are annotation-derived and must not participate in inference.  The stage-1
    cache is instead indexed using fields present in each public model input.
    Duplicate task rows share the same image and phrase and therefore the same
    frozen patch tokens and phrase encoding.
    """
    index: dict[tuple[str, str, str], list[str]] = {}
    for row in read_jsonl(STAGE1_ROWS / f"{split}.jsonl"):
        key = _input_query_key(row["dicom_id"], row["finding"], row["claim_sentence"])
        index.setdefault(key, []).append(str(row["task_id"]))
    resolved: dict[str, list[str]] = {}
    missing: list[str] = []
    for row in inputs:
        key = _input_query_key(row["dicom_id"], row["finding"], phrase_only(row["query_text"]))
        task_ids = sorted(index.get(key, []))
        if not task_ids:
            missing.append(str(row["group_id"]))
        else:
            resolved[str(row["group_id"])] = task_ids
    if missing:
        raise RuntimeError(f"Input-only RAD-DINO cache resolution failed for {split}: {missing[:5]} (n={len(missing)})")
    return resolved


def load_protocol(protocol_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = protocol_root / PROTOCOL_KEY
    inputs: dict[str, list[dict[str, Any]]] = {}
    labels: dict[str, dict[str, list[list[float]]]] = {}
    source_ids: dict[str, dict[str, list[str]]] = {}
    for split in ("train", "val", "eval"):
        inputs[split] = read_jsonl(root / f"{split}_inputs.jsonl")
        label_rows = read_jsonl(root / f"{split}_labels.jsonl")
        labels[split] = {
            str(row["group_id"]): [[float(x) for x in box] for box in row["gold_boxes_xyxy"]]
            for row in label_rows
        }
        source_ids[split] = resolve_source_task_ids_from_inputs(inputs[split], split)
    return inputs, labels, source_ids


def group_rows(inputs: list[dict[str, Any]], labels: dict[str, list[list[float]]], split: str) -> list[dict[str, Any]]:
    rows = []
    for source in inputs:
        gid = str(source["group_id"])
        boxes = labels[gid]
        rows.append(
            {
                "task_id": gid,
                "sample_id": gid,
                "group_id": gid,
                "dicom_id": str(source["dicom_id"]),
                "subject_id": str(source["subject_id"]),
                "study_id": str(source["study_id"]),
                "image_path": str(source["image_path"]),
                "finding": str(source["finding"]),
                "claim_sentence": phrase_only(str(source["query_text"])),
                "gold_bbox_xyxy": list(boxes[0]),
                "gold_boxes_xyxy": boxes,
                "image_width": int(source["image_width"]),
                "image_height": int(source["image_height"]),
                "split": split,
            }
        )
    return rows


def expanded_prior_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded = []
    for row in rows:
        for index, box in enumerate(row["gold_boxes_xyxy"]):
            expanded.append({**row, "task_id": f"{row['group_id']}::{index}", "gold_bbox_xyxy": list(box)})
    return expanded


def validate_contract(
    inputs: dict[str, list[dict[str, Any]]],
    labels: dict[str, dict[str, list[list[float]]]],
    *,
    canonical_v3: bool,
) -> dict[str, Any]:
    expected = {"train": 813, "val": 124, "eval": 220} if canonical_v3 else {"train": 814, "val": 125, "eval": 220}
    counts = {split: len(rows) for split, rows in inputs.items()}
    singleton = sum(len(labels["eval"][str(row["group_id"])]) == 1 for row in inputs["eval"])
    findings = sorted({str(row["finding"]) for split in inputs.values() for row in split})
    passed = counts == expected and singleton == 163 and findings == list(old_base.CLASS_NAMES)
    return {
        "status": "PASS" if passed else "FAIL",
        "group_counts": counts,
        "eval_singleton_groups": singleton,
        "finding_classes": findings,
        "query_contract": "finding category + raw phrase",
        "prediction_contract": "one variable-cardinality artifact; 1444 full set plus singleton-163 top-1",
        "canonical_v3": canonical_v3,
    }


def load_rad_bundle(
    split: str,
    allowed_task_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    rows = read_jsonl(STAGE1_ROWS / f"{split}.jsonl")
    cache = np.load(STAGE1_FEATURES / f"vfm_features_{split}.npz", allow_pickle=True)
    cache_ids = [str(x) for x in cache["task_ids"]]
    row_by_id = {str(row["task_id"]): row for row in rows}
    selected_ids = [task_id for task_id in cache_ids if allowed_task_ids is None or task_id in allowed_task_ids]
    if allowed_task_ids is not None and set(selected_ids) != allowed_task_ids:
        missing = sorted(allowed_task_ids - set(selected_ids))
        raise RuntimeError(f"RAD-DINO cache misses canonical source task IDs for {split}: {missing[:5]}")
    ordered = [row_by_id[task_id] for task_id in selected_ids]
    frame = pd.DataFrame(
        {
            "finding_label": [str(row["finding"]) for row in ordered],
            "phrase_text": [str(row["claim_sentence"]) for row in ordered],
        }
    )
    queries = rad_single.encode_query(frame, "full_phrase").astype(np.float32)
    targets = np.asarray([row["gold_bbox_norm_cxcywh"] for row in ordered], dtype=np.float32)
    index = [position for position, task_id in enumerate(cache_ids) if allowed_task_ids is None or task_id in allowed_task_ids]
    tokens = cache["patch_tokens"][index]
    return ordered, tokens, queries, targets


def center_indices(target: torch.Tensor, token_count: int) -> torch.Tensor:
    side = int(round(math.sqrt(token_count)))
    if side * side != token_count:
        return torch.zeros(len(target), dtype=torch.long, device=target.device)
    x = torch.clamp((target[:, 0] * side).long(), 0, side - 1)
    y = torch.clamp((target[:, 1] * side).long(), 0, side - 1)
    return y * side + x


def predict_rad(model: torch.nn.Module, tokens: np.ndarray, queries: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(tokens), batch_size):
            pred, _ = model(
                torch.from_numpy(tokens[start : start + batch_size]).float().to(device),
                torch.from_numpy(queries[start : start + batch_size]).float().to(device),
            )
            outputs.append(pred.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def mean_row_iou(predictions: np.ndarray, targets: np.ndarray) -> float:
    values = []
    for pred, target in zip(predictions, targets):
        values.append(old_fusion.base.iou_norm(pred, target))
    return float(np.mean(values)) if values else 0.0


def train_rad_head(
    seed_root: Path,
    seed: int,
    bundles: dict[str, tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]],
    epochs: int,
    batch_size: int,
    force: bool,
) -> dict[str, dict[str, np.ndarray]]:
    rad_root = seed_root / PROTOCOL_KEY / "rad_dino_legacy"
    checkpoint = rad_root / "best.pt"
    predictions_path = rad_root / "predictions_by_split.npz"
    rad_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_rows, train_tokens, train_queries, train_targets = bundles["train"]
    val_rows, val_tokens, val_queries, val_targets = bundles["val"]
    set_seed(seed)
    model = PatchHeatmapBBoxHead(train_tokens.shape[-1], train_queries.shape[-1], hidden=384, dropout=0.1).to(device)
    if checkpoint.exists() and not force:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["state"])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(train_tokens), torch.from_numpy(train_queries), torch.from_numpy(train_targets)),
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            num_workers=0,
        )
        best_score = -1.0
        best_state = None
        patience = 0
        logs = []
        for epoch in range(1, epochs + 1):
            model.train()
            losses = []
            for token, query, target in loader:
                token = token.float().to(device)
                query = query.float().to(device)
                target = target.float().to(device)
                pred, logits = model(token, query)
                regression, _ = bbox_loss(pred, target)
                loss = regression + 0.05 * torch.nn.functional.cross_entropy(logits, center_indices(target, token.shape[1]))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            score = mean_row_iou(predict_rad(model, val_tokens, val_queries, batch_size, device), val_targets)
            logs.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_row_mean_iou": score})
            if score > best_score:
                best_score = score
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            pd.DataFrame(logs).to_csv(rad_root / "train_log.csv", index=False)
            if patience >= 20:
                break
        if best_state is None:
            raise RuntimeError("RAD-DINO shallow head produced no checkpoint")
        torch.save(
            {
                "state": best_state,
                "seed": seed,
                "query_variant": "finding + full phrase rule-context hash",
                "backbone": "microsoft/rad-dino frozen",
                "selection": "row-level val mean IoU",
            },
            checkpoint,
        )
        model.load_state_dict(best_state)

    output: dict[str, dict[str, np.ndarray]] = {}
    npz_payload: dict[str, np.ndarray] = {}
    for split, (rows, tokens, queries, _targets) in bundles.items():
        pred = predict_rad(model, tokens, queries, batch_size, device)
        ids = np.asarray([str(row["task_id"]) for row in rows], dtype=object)
        output[split] = {str(task_id): box.astype(np.float32) for task_id, box in zip(ids, pred)}
        npz_payload[f"{split}_ids"] = ids
        npz_payload[f"{split}_boxes"] = pred.astype(np.float32)
    np.savez_compressed(predictions_path, **npz_payload)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def group_dino_map(
    task_predictions: dict[str, np.ndarray],
    group_source_ids: dict[str, list[str]],
) -> dict[str, np.ndarray]:
    result = {}
    for group_id, task_ids in group_source_ids.items():
        available = [task_predictions[task_id] for task_id in task_ids if task_id in task_predictions]
        if available:
            result[group_id] = np.mean(np.stack(available), axis=0).astype(np.float32)
    return result


def load_yolo_candidates(path: Path, split: str) -> dict[str, list[dict[str, Any]]]:
    candidates: dict[str, list[dict[str, Any]]] = {}
    for csv_path in sorted(path.glob(f"*_{split}.csv")):
        frame = pd.read_csv(csv_path)
        for row in frame.itertuples():
            candidates.setdefault(str(row.dicom_id), []).append(
                {
                    "box": [float(row.pred_x1), float(row.pred_y1), float(row.pred_x2), float(row.pred_y2)],
                    "score": float(row.confidence),
                    "class_id": int(row.class_id),
                    "rank": int(row.rank),
                    "source_model": str(row.source_model),
                    # Laterality-specialized proposal pilots add this optional
                    # column. Established four-detector CSVs remain unchanged.
                    "candidate_laterality_class": str(getattr(row, "candidate_laterality_class", "")),
                }
            )
    return candidates


def singleton_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if len(row["gold_boxes_xyxy"]) == 1]


def score_legacy_candidates(
    row: dict[str, Any],
    candidates_by_dicom: dict[str, list[dict[str, Any]]],
    priors: dict[str, Any],
    dino_norm: np.ndarray | None,
    params: dict[str, Any],
    fusion: dict[str, Any],
) -> list[dict[str, Any]]:
    iw, ih = float(row["image_width"]), float(row["image_height"])
    query, target_prior = old_base.yv2.target_prior_for_row(row, priors, params)
    class_id = old_fusion.base.CLASS_TO_ID[row["finding"]]
    max_rank = int(params.get("max_rank", 30))
    allowed = set(str(params.get("model_mode", "both")).split("+"))
    allow_all = bool({"both", "all", "any"} & allowed)
    query_side = str(row.get("query_laterality", "unknown"))
    allowed_laterality = {
        "right": {"right", "central"},
        "left": {"left", "central"},
        "bilateral": {"right", "left", "central"},
    }.get(query_side, {"right", "left", "central", ""})
    source = [
        candidate
        for candidate in candidates_by_dicom.get(str(row["dicom_id"]), [])
        if int(candidate["class_id"]) == class_id
        and (not str(candidate.get("candidate_laterality_class", "")) or str(candidate.get("candidate_laterality_class", "")) in allowed_laterality)
        and float(candidate["score"]) >= float(params["conf"])
        and int(candidate.get("rank", 9999)) < max_rank
        and (allow_all or str(candidate.get("source_model", "")) in allowed)
    ]
    scored = []
    for candidate in source:
        box_norm = old_fusion.base.xyxy_to_norm(candidate["box"], iw, ih)
        confidence = math.log1p(20.0 * max(0.0, float(candidate["score"])))
        region = old_base.yv2.center_region_score_v2(box_norm, query, str(params.get("side_mode", "radiology_right")))
        prior_iou = old_fusion.base.iou_norm(box_norm, target_prior)
        rank_bonus = 1.0 / (1.0 + float(candidate.get("rank", 0)))
        area = max(1e-6, float(box_norm[2] * box_norm[3]))
        prior_area = max(1e-6, float(target_prior[2] * target_prior[3]))
        area_penalty = abs(math.log(area / prior_area))
        source_bias = float(params.get("w_yolov8s_bias", 0.0)) if candidate.get("source_model") == "yolov8s" else 0.0
        dino_iou = old_fusion.base.iou_norm(box_norm, dino_norm) if dino_norm is not None else 0.0
        # Optional consensus is zero for all historical runs. A candidate is
        # only rewarded when a different YOLO source supports the same box.
        agreement = max(
            (
                old_fusion.base.iou_xyxy(candidate["box"], other["box"])
                for other in source
                if str(other.get("source_model", "")) != str(candidate.get("source_model", ""))
            ),
            default=0.0,
        )
        score = (
            float(params["w_conf"]) * confidence
            + float(params["w_region"]) * region
            + float(params["w_prior"]) * prior_iou
            + float(params["w_rank"]) * rank_bonus
            + source_bias
            + float(fusion.get("w_dino", 0.0)) * dino_iou
            + float(params.get("w_cross_yolo_agreement", 0.0)) * agreement
            - float(params["w_area"]) * area_penalty
        )
        pre_norm = old_fusion.base.blend_norm(box_norm, target_prior, float(params.get("blend_yolo_weight", 1.0)))
        final_norm = old_fusion.base.blend_norm(pre_norm, dino_norm, float(fusion.get("yolo_dino_blend", 1.0))) if dino_norm is not None else pre_norm
        scored.append(
            {
                "box": old_fusion.norm_to_xyxy(final_norm, iw, ih),
                "raw_box": list(candidate["box"]),
                "score": float(score),
                "source_model": str(candidate.get("source_model", "")),
                "rank": int(candidate.get("rank", -1)),
                "box_norm": box_norm,
                "cross_yolo_agreement": float(agreement),
            }
        )
    return sorted(scored, key=lambda item: (-item["score"], item["source_model"], item["rank"]))


def iou_xyxy(a: Iterable[float], b: Iterable[float]) -> float:
    return old_fusion.base.iou_xyxy(list(a), list(b))


def decode_predictions(
    rows: list[dict[str, Any]],
    candidates: dict[str, list[dict[str, Any]]],
    priors: dict[str, Any],
    dino: dict[str, np.ndarray],
    params_by_finding: dict[str, dict[str, Any]],
    fusion_by_finding: dict[str, dict[str, Any]],
    decoder: dict[str, float],
    *,
    precision_multibox_cues: bool = False,
) -> tuple[dict[str, list[list[float]]], pd.DataFrame]:
    groups = {str(row["group_id"]): row for row in rows}
    cues = (
        cue_info_from_groups_precision(groups)
        if precision_multibox_cues
        else cue_info_from_groups(groups)
    )
    outputs: dict[str, list[list[float]]] = {}
    audits = []
    for row in rows:
        gid = str(row["group_id"])
        params = params_by_finding.get(row["finding"], params_by_finding["__global__"])
        fusion = fusion_by_finding.get(row["finding"], fusion_by_finding["__global__"])
        dino_box = dino.get(gid)
        ranked = score_legacy_candidates(row, candidates, priors, dino_box, params, fusion)
        if ranked:
            selected = [ranked[0]["box"]]
            first_source = ranked[0]["source_model"]
        else:
            prediction, _conf, _missing, info = old_fusion.choose_fusion_candidate(
                row, candidates, priors, dino, params, fusion
            )
            selected = [prediction]
            first_source = str(info.get("source", "fallback"))

        cue = cues.get(gid, {"has_multi_cue": False, "multi_cue_type": "none", "k_hint": 1})
        decision = classify_cardinality_cue(cue, row["claim_sentence"])
        target_count = 1
        if bool(decision.get("hard_count_floor", False)):
            target_count = min(max(2, int(cue.get("k_hint", 2))), int(decoder["max_pred_count"]))
        additions = []
        while len(selected) < target_count:
            covered = {candidate_laterality(((box[0] + box[2]) * 0.5) / max(float(row["image_width"]), 1.0)) for box in selected}
            missing_targets = [
                target for target in cue.get("target_qs", [])
                if target.get("laterality") in {"right", "left"} and target.get("laterality") not in covered
            ]
            choices = []
            for candidate in ranked[1:]:
                if any(iou_xyxy(candidate["box"], old) >= float(decoder["nms_iou"]) for old in selected):
                    continue
                bonus = 0.0
                if missing_targets:
                    probe = pd.Series(
                        {
                            "box_cx_norm": (candidate["raw_box"][0] + candidate["raw_box"][2]) * 0.5 / float(row["image_width"]),
                            "box_cy_norm": (candidate["raw_box"][1] + candidate["raw_box"][3]) * 0.5 / float(row["image_height"]),
                        }
                    )
                    bonus = max(target_match_score(probe, target) for target in missing_targets)
                choices.append((candidate["score"] + float(decoder["side_bonus"]) * bonus, candidate))
            if not choices:
                additions.append("no_distinct_candidate")
                break
            choices.sort(key=lambda item: (-item[0], item[1]["source_model"], item[1]["rank"]))
            selected.append(choices[0][1]["box"])
            additions.append(choices[0][1]["source_model"])
        outputs[gid] = [[float(x) for x in box] for box in selected]
        audits.append(
            {
                "group_id": gid,
                "finding": row["finding"],
                "phrase": row["claim_sentence"],
                "multi_cue_type": cue.get("multi_cue_type", "none"),
                "hard_count_floor": bool(decision.get("hard_count_floor", False)),
                "target_count": target_count,
                "pred_count": len(selected),
                "first_source": first_source,
                "additions": ";".join(additions),
            }
        )
    return outputs, pd.DataFrame(audits)


def tune_decoder(
    spec: Any,
    inputs: list[dict[str, Any]],
    labels: dict[str, list[list[float]]],
    rows: list[dict[str, Any]],
    candidates: dict[str, list[dict[str, Any]]],
    priors: dict[str, Any],
    dino: dict[str, np.ndarray],
    params: dict[str, dict[str, Any]],
    fusion: dict[str, dict[str, Any]],
    output_path: Path,
    *,
    precision_multibox_cues: bool = False,
) -> dict[str, float]:
    records = []
    for nms_iou in (0.25, 0.35, 0.45, 0.55):
        for side_bonus in (0.0, 0.15, 0.30):
            decoder = {"nms_iou": nms_iou, "side_bonus": side_bonus, "max_pred_count": 3.0}
            predictions, _ = decode_predictions(
                rows,
                candidates,
                priors,
                dino,
                params,
                fusion,
                decoder,
                precision_multibox_cues=precision_multibox_cues,
            )
            summary, _detail = summarize_protocol(spec, inputs, labels, predictions)
            score = float(np.mean([summary["coverage_mean_iou"], summary["exact_union_iou"], summary["set_f1_0_3"], summary["set_f1_0_5"]]))
            records.append({**decoder, "selection_score": score, **summary})
    grid = pd.DataFrame(records).sort_values(["selection_score", "set_f1_0_5", "coverage_mean_iou"], ascending=False)
    grid.to_csv(output_path, index=False)
    best = grid.iloc[0]
    return {"nms_iou": float(best.nms_iou), "side_bonus": float(best.side_bonus), "max_pred_count": int(best.max_pred_count)}


def run_seed(
    args: argparse.Namespace,
    seed: int,
    inputs: dict[str, list[dict[str, Any]]],
    labels: dict[str, dict[str, list[list[float]]]],
    source_ids: dict[str, dict[str, list[str]]],
    rad_bundles: dict[str, tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    seed_output = args.output_root / f"seed_{seed}"
    set_seed(seed)
    config = TaskRunConfig.from_json(args.config)
    config = replace(
        config,
        yolo_models=tuple(args.yolo_models),
        yolo_epochs=1 if args.quick else args.yolo_epochs,
        candidate_top_per_detector=30,
        candidate_limit=160,
        pipeline_seed=seed,
    )
    stage_marker = seed_output / PROTOCOL_KEY / "yolo_runs" / "STAGE_COMPLETE.json"
    pipeline = ThreeTaskPipeline(
        get_protocol(PROTOCOL_KEY),
        args.protocol_root,
        seed_output,
        config,
        disable_rad_dino=True,
        force=bool(args.force or not stage_marker.exists()),
    )
    if tuple(pipeline._yolo_class_names()) != tuple(old_base.CLASS_NAMES):
        raise RuntimeError(
            "Refusing to run: imported YOLO pipeline is not the eight-class finding-conditioned implementation"
        )
    pipeline.build_yolo_dataset()
    supervision_contract = json.loads(
        (pipeline.paths.yolo_dataset / "supervision_contract.json").read_text(encoding="utf-8")
    )
    if supervision_contract.get("class_agnostic") is not False or supervision_contract.get("n_classes") != 8:
        raise RuntimeError(f"Finding-conditioned YOLO contract failed: {supervision_contract}")
    weights = pipeline.train_yolo()
    write_json(
        stage_marker,
        {
            "status": "complete",
            "seed": seed,
            "epochs_requested": config.yolo_epochs,
            "models": {name: str(path) for name, path in weights.items()},
            "finding_conditioned": True,
            "n_classes": 8,
        },
    )
    pipeline.predict_yolo(weights)

    task_dino = train_rad_head(
        seed_output,
        seed,
        rad_bundles,
        2 if args.quick else args.rad_epochs,
        args.rad_batch,
        args.force,
    )
    rows = {split: group_rows(inputs[split], labels[split], split) for split in ("train", "val", "eval")}
    dino = {split: group_dino_map(task_dino[split], source_ids[split]) for split in rows}
    yolo_path = seed_output / PROTOCOL_KEY / "yolo_predictions"
    candidates = {split: load_yolo_candidates(yolo_path, split) for split in rows}
    priors = old_base.ybase.make_train_priors(expanded_prior_rows(rows["train"]))
    val_singletons = singleton_rows(rows["val"])
    params, yolo_grid, yolo_per = old_base.yv2.tune(val_singletons, candidates["val"], priors, args.quick)
    fusion, fusion_grid, fusion_per = old_fusion.tune_fusion(
        val_singletons, candidates["val"], priors, dino["val"], params, args.quick
    )
    fusion_root = seed_output / PROTOCOL_KEY / "legacy_fusion"
    fusion_root.mkdir(parents=True, exist_ok=True)
    yolo_grid.to_csv(fusion_root / "yolo_rule_val_grid.csv", index=False)
    yolo_per.to_csv(fusion_root / "yolo_rule_by_finding.csv", index=False)
    fusion_grid.to_csv(fusion_root / "fusion_val_grid.csv", index=False)
    fusion_per.to_csv(fusion_root / "fusion_by_finding.csv", index=False)
    write_json(fusion_root / "yolo_params.json", params)
    write_json(fusion_root / "fusion_params.json", fusion)

    spec = get_protocol(PROTOCOL_KEY)
    decoder = tune_decoder(
        spec,
        inputs["val"],
        labels["val"],
        rows["val"],
        candidates["val"],
        priors,
        dino["val"],
        params,
        fusion,
        fusion_root / "decoder_val_grid.csv",
        precision_multibox_cues=args.precision_multibox_cues,
    )
    write_json(fusion_root / "decoder_params.json", decoder)
    predictions, audit = decode_predictions(
        rows["eval"],
        candidates["eval"],
        priors,
        dino["eval"],
        params,
        fusion,
        decoder,
        precision_multibox_cues=args.precision_multibox_cues,
    )
    prediction_path = seed_output / PROTOCOL_KEY / "predictions" / "eval_predictions_unified.jsonl"
    write_jsonl(
        prediction_path,
        [
            {"protocol_key": PROTOCOL_KEY, "group_id": gid, "pred_boxes_xyxy": boxes}
            for gid, boxes in sorted(predictions.items())
        ],
    )
    audit.to_csv(fusion_root / "eval_cardinality_audit.csv", index=False)
    summary, detail = summarize_protocol(spec, inputs["eval"], labels["eval"], predictions)
    summary.update(singleton_projection(inputs["eval"], labels["eval"], predictions))
    detail_path = seed_output / PROTOCOL_KEY / "metrics" / "eval_detail.csv"
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(detail).to_csv(detail_path, index=False)
    result = {
        "status": "complete",
        "method": "ClueGround-VFM legacy fusion unified variable-cardinality",
        "seed": seed,
        "full_pipeline_seed": True,
        "yolo_models": list(args.yolo_models),
        "yolo_finding_classes": list(old_base.CLASS_NAMES),
        "rad_dino_backbone": "microsoft/rad-dino frozen",
        "rad_dino_head_trained_per_seed": True,
        "candidate_ranker": "original validation-tuned YOLO-rule-RAD-DINO fusion equation",
        "decoder": (
            "high-precision raw-phrase multi-region cue; same ranked candidate pool"
            if args.precision_multibox_cues
            else "raw-phrase explicit multi-region cue; same ranked candidate pool"
        ),
        "prediction_path": str(prediction_path),
        "same_prediction_for_888_and_1444": True,
        "selection_split": "val only",
        "decoder_params": decoder,
        **summary,
    }
    write_json(seed_output / "RUN_STATUS.json", result)
    return result


def aggregate_results(output_root: Path, results: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(results)
    final = output_root / "final_tables"
    final.mkdir(parents=True, exist_ok=True)
    frame.to_csv(final / "ours_unified_seed_results.csv", index=False)
    metric_columns = [
        "coverage_mean_iou",
        "exact_union_iou",
        "hull_union_iou_diagnostic",
        "set_f1_0_3",
        "set_f1_0_5",
        "mean_pred_count",
        "singlebox_888_mean_iou",
        "singlebox_888_hit_0_3",
        "singlebox_888_hit_0_5",
    ]
    aggregate: dict[str, Any] = {
        "status": "complete" if len(results) == 3 else "partial",
        "method": "ClueGround-VFM legacy fusion unified variable-cardinality",
        "n_full_pipeline_seeds": len(results),
        "seeds": [int(row["seed"]) for row in results],
        "same_prediction_for_888_and_1444": True,
    }
    for column in metric_columns:
        values = frame[column].astype(float).to_numpy()
        aggregate[f"{column}_mean"] = float(values.mean())
        aggregate[f"{column}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    write_json(final / "ours_unified_3seed_aggregate.json", aggregate)
    pd.DataFrame([aggregate]).to_csv(final / "ours_unified_3seed_baseline_row.csv", index=False)
    write_json(output_root / "FINAL_STATUS.json", aggregate)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "status": "PLAN_ONLY" if not args.execute else "running",
        "method": "original finding-conditioned YOLO--RAD-DINO fusion plus variable-cardinality decoder",
        "seeds": args.seeds,
        "full_pipeline_seed_scope": "four YOLO detectors and RAD-DINO shallow head retrained per seed",
        "unchanged": [
            "eight MS-CXR finding classes",
            f"YOLO proposal pool: {', '.join(args.yolo_models)}",
            "phrase/rule-context RAD-DINO shallow head",
            "validation-tuned YOLO--DINO fusion equation",
        ],
        "only_behavioral_extension": "explicit multi-region cues may add distinct lower-ranked boxes",
        "evaluation": "one prediction artifact -> 1444 full set and singleton-163 top-1",
    }
    write_json(args.output_root / "RUN_PLAN.json", plan)
    if not args.execute:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return

    inputs, labels, source_ids = load_protocol(args.protocol_root)
    contract = validate_contract(inputs, labels, canonical_v3=args.canonical_v3)
    write_json(args.output_root / "METHOD_CONTRACT.json", contract)
    if contract["status"] != "PASS":
        raise RuntimeError(f"Method contract failed: {contract}")

    rad_bundles = {
        split: load_rad_bundle(
            split,
            {task_id for task_ids in source_ids[split].values() for task_id in task_ids}
            if args.canonical_v3
            else None,
        )
        for split in ("train", "val", "eval")
    }
    results = []
    for index, seed in enumerate(args.seeds, start=1):
        write_json(
            args.output_root / "QUEUE_STATUS.json",
            {"status": "running", "seed": seed, "seed_index": index, "n_seeds": len(args.seeds), "completed_seeds": [row["seed"] for row in results]},
        )
        started = time.time()
        try:
            result = run_seed(args, seed, inputs, labels, source_ids, rad_bundles)
            result["elapsed_seconds"] = time.time() - started
            results.append(result)
        except Exception as exc:
            write_json(
                args.output_root / "QUEUE_STATUS.json",
                {"status": "failed", "seed": seed, "completed_seeds": [row["seed"] for row in results], "error": repr(exc)},
            )
            raise
    aggregate_results(args.output_root, results)
    write_json(args.output_root / "QUEUE_STATUS.json", {"status": "complete", "completed_seeds": [row["seed"] for row in results]})
    print(json.dumps(json.loads((args.output_root / "FINAL_STATUS.json").read_text(encoding="utf-8")), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

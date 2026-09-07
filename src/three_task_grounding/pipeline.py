from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.ensemble import HistGradientBoostingRegressor
from torch.utils.data import DataLoader, TensorDataset

from scripts.models_ms_cxr_vfm_localizer import PatchHeatmapBBoxHead, bbox_loss
from scripts import run_ms_cxr_biomedclip_gated_hybrid_v1 as biomed
from scripts import run_ms_cxr_siglip_candidate_fusion_v1 as siglip
from scripts import run_ms_cxr_trainable_moe_gate_v1 as moe
from src.fair_baselines.metrics import hull_box, iou_xyxy
from src.rerank.multibox_cue_parser import cue_info_from_groups
from src.rerank.unified_adaptive_cardinality import (
    apply_unified_adaptive_cardinality,
    default_policy_grid,
    prepare_candidate_index,
)

from .contracts import SPLITS, ProtocolSpec
from .guards import BASE_RANKER_FEATURES, SEMANTIC_EXPERT_FEATURES, validate_ranker_schema
from .manifests import read_jsonl, write_jsonl
from .metrics import singleton_projection, summarize_protocol
from .queries import context_score, encode_query, stable_hash


RAD_DINO_MODEL_ID = "microsoft/rad-dino"
RAD_DINO_REVISION = "110cbc18d5133582e320b43d53bf5c44e410c936"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TaskRunConfig:
    yolo_models: tuple[str, ...]
    yolo_epochs: int
    yolo_batch: int
    yolo_infer_batch: int
    image_size: int
    rad_dino_epochs: int
    rad_dino_batch: int
    rad_dino_extract_batch: int
    candidate_top_per_detector: int
    candidate_limit: int
    gate_seeds: tuple[int, ...]
    pipeline_seed: int
    device: str
    image_root: Path
    use_semantic_experts: bool

    @classmethod
    def from_json(cls, path: Path) -> "TaskRunConfig":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            yolo_models=tuple(payload["yolo_models"]),
            yolo_epochs=int(payload["yolo_epochs"]),
            yolo_batch=int(payload["yolo_batch"]),
            yolo_infer_batch=int(payload["yolo_infer_batch"]),
            image_size=int(payload.get("image_size", 640)),
            rad_dino_epochs=int(payload["rad_dino_epochs"]),
            rad_dino_batch=int(payload["rad_dino_batch"]),
            rad_dino_extract_batch=int(payload["rad_dino_extract_batch"]),
            candidate_top_per_detector=int(payload.get("candidate_top_per_detector", 3)),
            candidate_limit=int(payload.get("candidate_limit", 20)),
            gate_seeds=tuple(int(value) for value in payload["gate_seeds"]),
            pipeline_seed=int(payload["pipeline_seed"]),
            device=str(payload.get("device", "0")),
            image_root=Path(payload["image_root"]),
            use_semantic_experts=bool(payload.get("use_semantic_experts", True)),
        )


@dataclass(frozen=True)
class TaskPaths:
    root: Path
    protocol_root: Path
    yolo_dataset: Path
    yolo_runs: Path
    yolo_predictions: Path
    rad_dino: Path
    candidates: Path
    semantic: Path
    ranker: Path
    gate: Path
    predictions: Path
    metrics: Path

    @classmethod
    def create(cls, output_root: Path, protocol_root: Path, protocol_key: str) -> "TaskPaths":
        root = output_root / protocol_key
        result = cls(
            root=root,
            protocol_root=protocol_root / protocol_key,
            yolo_dataset=root / "yolo_dataset",
            yolo_runs=root / "yolo_runs",
            yolo_predictions=root / "yolo_predictions",
            rad_dino=root / "rad_dino",
            candidates=root / "candidates",
            semantic=root / "semantic",
            ranker=root / "ranker",
            gate=root / "gate",
            predictions=root / "predictions",
            metrics=root / "metrics",
        )
        for path in result.__dict__.values():
            Path(path).mkdir(parents=True, exist_ok=True)
        return result


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _xyxy_to_yolo(box: Iterable[float], width: float, height: float) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    return (
        ((x1 + x2) * 0.5) / width,
        ((y1 + y2) * 0.5) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def _norm_cxcywh_to_xyxy(box: Iterable[float], width: float, height: float) -> list[float]:
    cx, cy, bw, bh = [float(value) for value in box]
    return [
        max(0.0, (cx - bw * 0.5) * width),
        max(0.0, (cy - bh * 0.5) * height),
        min(width, (cx + bw * 0.5) * width),
        min(height, (cy + bh * 0.5) * height),
    ]


def _target_cxcywh(boxes: list[list[float]], width: float, height: float) -> list[float]:
    box = hull_box(boxes)
    return list(_xyxy_to_yolo(box, width, height))


class ThreeTaskPipeline:
    """One task-specific ClueGround-VFM instantiation.

    The class never pools supervision across tasks. A pooled diagnostic must be
    built with the separately audited global_patient_disjoint protocol bundle.
    """

    def __init__(
        self,
        spec: ProtocolSpec,
        protocol_root: Path,
        output_root: Path,
        config: TaskRunConfig,
        *,
        disable_rad_dino: bool,
        force: bool,
    ) -> None:
        self.spec = spec
        self.config = config
        self.disable_rad_dino = disable_rad_dino
        self.force = force
        self.paths = TaskPaths.create(output_root, protocol_root, spec.key)
        self.torch_device = torch.device(
            "cuda" if torch.cuda.is_available() and config.device.lower() != "cpu" else "cpu"
        )
        set_seed(config.pipeline_seed)
        self.inputs = {split: read_jsonl(self.paths.protocol_root / f"{split}_inputs.jsonl") for split in SPLITS}
        self.labels = {
            split: {
                str(row["group_id"]): [list(map(float, box)) for box in row["gold_boxes_xyxy"]]
                for row in read_jsonl(self.paths.protocol_root / f"{split}_labels.jsonl")
            }
            for split in SPLITS
        }
        self._validate_protocol_inputs()

    def _validate_protocol_inputs(self) -> None:
        for split in SPLITS:
            input_ids = {str(row["group_id"]) for row in self.inputs[split]}
            label_ids = set(self.labels[split])
            if input_ids != label_ids:
                raise RuntimeError(f"Input/label ID mismatch for {self.spec.key}/{split}")
        _write_json(
            self.paths.root / "run_contract.json",
            {
                "protocol": self.spec.to_dict(),
                "task_isolated": True,
                "disable_rad_dino": self.disable_rad_dino,
                "semantic_experts_enabled": self.config.use_semantic_experts,
                "pipeline_seed": self.config.pipeline_seed,
                "gate_seeds": list(self.config.gate_seeds),
                "seed_scope": (
                    "one fixed upstream pipeline; gate seeds only"
                    if self.config.use_semantic_experts
                    else "one deterministic complete pipeline; no gate-only repeats"
                ),
                "mscxr_cig_free_eligibility": self.spec.key == "mscxr_multibox_1444",
                "imagenome_supervision_transfer_to_mscxr": False,
                "model_provenance": {
                    "yolo_initializations": [
                        {
                            "model": model_name,
                            "local_path": str((PROJECT_ROOT / model_name).resolve()),
                            "exists": (PROJECT_ROOT / model_name).is_file(),
                            "sha256": (
                                _file_sha256(PROJECT_ROOT / model_name)
                                if (PROJECT_ROOT / model_name).is_file()
                                else ""
                            ),
                        }
                        for model_name in self.config.yolo_models
                    ],
                    "rad_dino": {"model_id": RAD_DINO_MODEL_ID, "revision": RAD_DINO_REVISION},
                    "siglip": {
                        "enabled": self.config.use_semantic_experts,
                        "model_id": "google/siglip-base-patch16-224",
                        "revision": siglip.SIGLIP_REVISIONS["google/siglip-base-patch16-224"],
                    },
                    "biomedclip": {
                        "enabled": self.config.use_semantic_experts,
                        "model_id": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                        "revision": biomed.BIOMEDCLIP_REVISIONS[
                            "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
                        ],
                    },
                },
            },
        )

    def build_yolo_dataset(self) -> None:
        dataset = self.paths.yolo_dataset
        if self.force and dataset.exists():
            shutil.rmtree(dataset)
            dataset.mkdir(parents=True, exist_ok=True)
        image_link = dataset / "images"
        if not image_link.exists():
            if os.name != "nt":
                image_link.symlink_to(self.config.image_root, target_is_directory=True)
            else:
                subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(image_link), str(self.config.image_root)],
                    check=True,
                    capture_output=True,
                    text=True,
                )

        manifest_rows: list[dict[str, Any]] = []
        for split in ("train", "val"):
            by_image: dict[str, dict[str, Any]] = {}
            for row in self.inputs[split]:
                image_path = Path(row["image_path"])
                try:
                    relative = image_path.relative_to(self.config.image_root)
                except ValueError as exc:
                    raise RuntimeError(f"Image is outside configured image_root: {image_path}") from exc
                item = by_image.setdefault(
                    str(image_path),
                    {
                        "relative": relative,
                        "width": int(row["image_width"]),
                        "height": int(row["image_height"]),
                        "boxes": {},
                        "group_ids": [],
                    },
                )
                item["group_ids"].append(str(row["group_id"]))
                for box in self.labels[split][str(row["group_id"])]:
                    key = tuple(round(float(value), 4) for value in box)
                    item["boxes"][key] = box

            image_list: list[str] = []
            for source_path, item in sorted(by_image.items()):
                relative = Path(item["relative"])
                linked_image = image_link / relative
                label_path = dataset / "labels" / relative.with_suffix(".txt")
                label_path.parent.mkdir(parents=True, exist_ok=True)
                lines = []
                for box in item["boxes"].values():
                    cx, cy, bw, bh = _xyxy_to_yolo(box, item["width"], item["height"])
                    lines.append(f"0 {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")
                label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                # Keep the path through yolo_dataset/images. Resolving the
                # junction would point back to E:/... and Ultralytics would no
                # longer discover the mirrored yolo_dataset/labels path.
                image_list.append(str(linked_image.absolute()))
                manifest_rows.append(
                    {
                        "split": split,
                        "source_image_path": source_path,
                        "linked_image_path": str(linked_image),
                        "label_path": str(label_path),
                        "n_boxes": len(lines),
                        "group_ids": "|".join(item["group_ids"]),
                    }
                )
            (dataset / f"{split}.txt").write_text("\n".join(image_list) + "\n", encoding="utf-8")

        yaml_lines = [
            f"path: {dataset.as_posix()}",
            "train: train.txt",
            "val: val.txt",
            "names:",
            "  0: medical_region",
            "",
        ]
        (dataset / "task.yaml").write_text("\n".join(yaml_lines), encoding="utf-8")
        pd.DataFrame(manifest_rows).to_csv(dataset / "dataset_manifest.csv", index=False)
        _write_json(
            dataset / "supervision_contract.json",
            {
                "status": "PASS",
                "class_agnostic": True,
                "n_classes": 1,
                "eval_labels_materialized": False,
                "task_isolated": True,
                "query_or_category_used_by_yolo": False,
            },
        )

    def train_yolo(self) -> dict[str, Path]:
        from ultralytics import YOLO

        weights: dict[str, Path] = {}
        for model_name in self.config.yolo_models:
            tag = Path(model_name).stem
            run_name = f"{tag}_{self.spec.key}_classagnostic_e{self.config.yolo_epochs}"
            best = self.paths.yolo_runs / run_name / "weights" / "best.pt"
            if not best.exists() or self.force:
                initialization = PROJECT_ROOT / model_name
                model = YOLO(str(initialization) if initialization.is_file() else model_name)
                model.train(
                    data=str(self.paths.yolo_dataset / "task.yaml"),
                    epochs=self.config.yolo_epochs,
                    imgsz=self.config.image_size,
                    batch=self.config.yolo_batch,
                    workers=0,
                    project=str(self.paths.yolo_runs),
                    name=run_name,
                    exist_ok=True,
                    device=self.config.device,
                    pretrained=True,
                    seed=self.config.pipeline_seed,
                    patience=max(20, min(50, self.config.yolo_epochs // 2)),
                )
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if not best.is_file():
                raise FileNotFoundError(best)
            weights[tag] = best
        _write_json(self.paths.yolo_runs / "weights_manifest.json", {key: str(value) for key, value in weights.items()})
        return weights

    def predict_yolo(self, weights: dict[str, Path]) -> None:
        from ultralytics import YOLO

        audit: list[dict[str, Any]] = []
        for split in SPLITS:
            unique = {str(row["dicom_id"]): str(row["image_path"]) for row in self.inputs[split]}
            items = sorted(unique.items())
            for tag, checkpoint in weights.items():
                output = self.paths.yolo_predictions / f"{tag}_{split}.csv"
                if output.exists() and not self.force:
                    continue
                model = YOLO(str(checkpoint))
                rows: list[dict[str, Any]] = []
                start = 0
                active_batch = max(1, self.config.yolo_infer_batch)
                while start < len(items):
                    batch_items = items[start : start + active_batch]
                    try:
                        results = model.predict(
                            source=[path for _, path in batch_items],
                            imgsz=self.config.image_size,
                            conf=0.001,
                            max_det=100,
                            batch=len(batch_items),
                            half=self.torch_device.type == "cuda",
                            device=self.config.device,
                            verbose=False,
                        )
                    except RuntimeError as exc:
                        is_oom = isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()
                        audit.append(
                            {
                                "model": tag,
                                "split": split,
                                "start": start,
                                "batch": len(batch_items),
                                "status": "oom" if is_oom else "runtime_error",
                            }
                        )
                        if not is_oom or active_batch == 1:
                            raise
                        active_batch = max(1, active_batch // 2)
                        gc.collect()
                        torch.cuda.empty_cache()
                        continue
                    for (dicom_id, image_path), result in zip(batch_items, results):
                        if result.boxes is None:
                            continue
                        boxes = result.boxes.xyxy.detach().cpu().numpy()
                        scores = result.boxes.conf.detach().cpu().numpy()
                        order = np.argsort(-scores, kind="stable")
                        for rank, index in enumerate(order):
                            rows.append(
                                {
                                    "dicom_id": dicom_id,
                                    "image_path": image_path,
                                    "source_model": tag,
                                    "rank": rank,
                                    "confidence": float(scores[index]),
                                    "pred_x1": float(boxes[index, 0]),
                                    "pred_y1": float(boxes[index, 1]),
                                    "pred_x2": float(boxes[index, 2]),
                                    "pred_y2": float(boxes[index, 3]),
                                }
                            )
                    audit.append(
                        {"model": tag, "split": split, "start": start, "batch": len(batch_items), "status": "complete"}
                    )
                    start += len(batch_items)
                pd.DataFrame(rows).to_csv(output, index=False)
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        pd.DataFrame(audit).to_csv(self.paths.yolo_predictions / "adaptive_batch_audit.csv", index=False)

    def _extract_rad_tokens(self, split: str) -> Path:
        output = self.paths.rad_dino / f"patch_tokens_{split}.npz"
        if output.exists() and not self.force:
            return output
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(
            RAD_DINO_MODEL_ID,
            revision=RAD_DINO_REVISION,
            trust_remote_code=True,
        )
        model = AutoModel.from_pretrained(
            RAD_DINO_MODEL_ID,
            revision=RAD_DINO_REVISION,
            trust_remote_code=True,
        ).to(self.torch_device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        group_ids: list[str] = []
        tokens: list[np.ndarray] = []
        rows = self.inputs[split]
        for start in range(0, len(rows), self.config.rad_dino_extract_batch):
            part = rows[start : start + self.config.rad_dino_extract_batch]
            images = []
            for row in part:
                with Image.open(row["image_path"]) as source:
                    images.append(source.convert("RGB"))
            batch = processor(images=images, return_tensors="pt")
            batch = {key: value.to(self.torch_device) for key, value in batch.items()}
            with torch.no_grad():
                result = model(**batch)
            hidden = result.last_hidden_state.detach().float()
            patch = hidden[:, 1:, :] if hidden.shape[1] > 1 else hidden
            tokens.extend(patch.cpu().numpy().astype(np.float16))
            group_ids.extend(str(row["group_id"]) for row in part)
        np.savez(output, group_ids=np.asarray(group_ids, dtype=object), patch_tokens=np.stack(tokens))
        _write_json(
            self.paths.rad_dino / f"patch_tokens_{split}_provenance.json",
            {
                "model_id": RAD_DINO_MODEL_ID,
                "revision": RAD_DINO_REVISION,
                "frozen": True,
                "n_groups": len(group_ids),
            },
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return output

    def _rad_arrays(self, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        cache = np.load(self._extract_rad_tokens(split), allow_pickle=True)
        index = {str(group_id): idx for idx, group_id in enumerate(cache["group_ids"])}
        rows = [row for row in self.inputs[split] if str(row["group_id"]) in index]
        # NPZ members are decompressed on access. Materialize once rather than
        # decompressing the multi-GB patch-token member once per query.
        patch_tokens = cache["patch_tokens"]
        tokens = np.stack([patch_tokens[index[str(row["group_id"])]] for row in rows])
        queries = np.stack([encode_query(row["query_text"], self.spec.key) for row in rows]).astype(np.float32)
        targets = np.asarray(
            [
                _target_cxcywh(
                    self.labels[split][str(row["group_id"])],
                    float(row["image_width"]),
                    float(row["image_height"]),
                )
                for row in rows
            ],
            dtype=np.float32,
        )
        return tokens, queries, targets, rows

    @staticmethod
    def _center_indices(target: torch.Tensor, token_count: int) -> torch.Tensor:
        side = int(round(math.sqrt(token_count)))
        if side * side != token_count:
            return torch.zeros(len(target), dtype=torch.long, device=target.device)
        x = torch.clamp((target[:, 0] * side).long(), 0, side - 1)
        y = torch.clamp((target[:, 1] * side).long(), 0, side - 1)
        return y * side + x

    def train_rad_dino_head(self) -> None:
        if self.disable_rad_dino:
            _write_json(self.paths.rad_dino / "status.json", {"status": "disabled"})
            return
        train_tokens, train_query, train_targets, _ = self._rad_arrays("train")
        val_tokens, val_query, val_targets, val_rows = self._rad_arrays("val")
        model = PatchHeatmapBBoxHead(
            train_tokens.shape[-1], train_query.shape[-1], hidden=384, dropout=0.1
        ).to(self.torch_device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(train_tokens),
                torch.from_numpy(train_query),
                torch.from_numpy(train_targets),
            ),
            batch_size=self.config.rad_dino_batch,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.config.pipeline_seed),
            num_workers=0,
        )
        best_score = -1.0
        best_state = None
        patience = 0
        logs: list[dict[str, Any]] = []
        for epoch in range(1, self.config.rad_dino_epochs + 1):
            model.train()
            losses = []
            for token, query, target in loader:
                token = token.float().to(self.torch_device)
                query = query.float().to(self.torch_device)
                target = target.float().to(self.torch_device)
                prediction, logits = model(token, query)
                regression, _ = bbox_loss(prediction, target)
                center = self._center_indices(target, token.shape[1])
                loss = regression + 0.05 * torch.nn.functional.cross_entropy(logits, center)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            val_predictions = self._predict_rad_arrays(model, val_tokens, val_query)
            val_iou = [
                iou_xyxy(
                    _norm_cxcywh_to_xyxy(pred, row["image_width"], row["image_height"]),
                    hull_box(self.labels["val"][str(row["group_id"])]),
                )
                for pred, row in zip(val_predictions, val_rows)
            ]
            score = float(np.mean(val_iou))
            logs.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_mean_iou": score})
            if score > best_score:
                best_score = score
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if patience >= 20:
                break
        if best_state is None:
            raise RuntimeError("RAD-DINO head did not produce a checkpoint")
        model.load_state_dict(best_state)
        torch.save(
            {
                "state": best_state,
                "token_dim": int(train_tokens.shape[-1]),
                "query_dim": int(train_query.shape[-1]),
                "protocol_key": self.spec.key,
                "rad_dino_revision": RAD_DINO_REVISION,
            },
            self.paths.rad_dino / "best.pt",
        )
        pd.DataFrame(logs).to_csv(self.paths.rad_dino / "train_log.csv", index=False)
        for split in SPLITS:
            tokens, queries, _targets, rows = self._rad_arrays(split)
            predictions = self._predict_rad_arrays(model, tokens, queries)
            output_rows = []
            for prediction, row in zip(predictions, rows):
                box = _norm_cxcywh_to_xyxy(prediction, row["image_width"], row["image_height"])
                output_rows.append(
                    {
                        "group_id": row["group_id"],
                        "dicom_id": row["dicom_id"],
                        "pred_x1": box[0],
                        "pred_y1": box[1],
                        "pred_x2": box[2],
                        "pred_y2": box[3],
                    }
                )
            pd.DataFrame(output_rows).to_csv(self.paths.rad_dino / f"predictions_{split}.csv", index=False)

    def _predict_rad_arrays(self, model: torch.nn.Module, tokens: np.ndarray, queries: np.ndarray) -> np.ndarray:
        model.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(tokens), self.config.rad_dino_batch):
                prediction, _ = model(
                    torch.from_numpy(tokens[start : start + self.config.rad_dino_batch]).float().to(self.torch_device),
                    torch.from_numpy(queries[start : start + self.config.rad_dino_batch]).float().to(self.torch_device),
                )
                outputs.append(prediction.cpu().numpy())
        return np.concatenate(outputs) if outputs else np.empty((0, 4), dtype=np.float32)

    def build_candidates(self) -> None:
        for split in SPLITS:
            output = self.paths.candidates / f"{split}.csv"
            if output.exists() and not self.force:
                continue
            detector_paths = sorted(self.paths.yolo_predictions.glob(f"*_{split}.csv"))
            detector_frames = [pd.read_csv(path) for path in detector_paths if path.stat().st_size > 0]
            detector = pd.concat(detector_frames, ignore_index=True) if detector_frames else pd.DataFrame()
            by_dicom = {str(key): part for key, part in detector.groupby("dicom_id")} if len(detector) else {}
            rad_path = self.paths.rad_dino / f"predictions_{split}.csv"
            rad_frame = pd.read_csv(rad_path) if not self.disable_rad_dino and rad_path.exists() else pd.DataFrame()
            rad_map = {
                str(row.group_id): [row.pred_x1, row.pred_y1, row.pred_x2, row.pred_y2]
                for row in rad_frame.itertuples()
            }
            output_rows: list[dict[str, Any]] = []
            for source in self.inputs[split]:
                group_id = str(source["group_id"])
                raw_candidates: list[dict[str, Any]] = []
                detector_rows = by_dicom.get(str(source["dicom_id"]), pd.DataFrame())
                if len(detector_rows):
                    for model_name, model_rows in detector_rows.groupby("source_model", sort=True):
                        for row in model_rows.sort_values(["rank", "confidence"], ascending=[True, False]).head(
                            self.config.candidate_top_per_detector
                        ).itertuples():
                            raw_candidates.append(
                                {
                                    "source_model": str(model_name),
                                    "rank": int(row.rank),
                                    "confidence": float(row.confidence),
                                    "box": [row.pred_x1, row.pred_y1, row.pred_x2, row.pred_y2],
                                }
                            )
                dino_box = rad_map.get(group_id)
                if dino_box is not None:
                    raw_candidates.append(
                        # The RAD bbox head has no calibrated detector confidence.
                        # Keep the missing-confidence value neutral instead of
                        # presenting it to the ranker as a perfect YOLO score.
                        {"source_model": "rad_dino", "rank": 0, "confidence": 0.0, "box": dino_box}
                    )
                if not raw_candidates:
                    raw_candidates.append(
                        {
                            "source_model": "fallback_full_image",
                            "rank": 0,
                            "confidence": 0.0,
                            "box": [0.0, 0.0, float(source["image_width"]), float(source["image_height"])],
                        }
                    )

                unique: list[dict[str, Any]] = []
                ordered = sorted(
                    raw_candidates,
                    key=lambda row: (
                        -float(row["confidence"]),
                        stable_hash(json.dumps(row["box"], separators=(",", ":"))),
                    ),
                )
                for candidate in ordered:
                    if any(iou_xyxy(candidate["box"], old["box"]) > 0.97 for old in unique):
                        continue
                    unique.append(candidate)
                unique = unique[: self.config.candidate_limit]
                for candidate in unique:
                    box = [float(value) for value in candidate["box"]]
                    width = max(0.0, box[2] - box[0])
                    height = max(0.0, box[3] - box[1])
                    other_agreements = [
                        iou_xyxy(box, other["box"])
                        for other in unique
                        if other is not candidate and other["source_model"] != candidate["source_model"]
                    ]
                    output_rows.append(
                        {
                            "protocol_key": self.spec.key,
                            "group_id": group_id,
                            "query_id": group_id,
                            "split": split,
                            "dicom_id": source["dicom_id"],
                            "subject_id": source["subject_id"],
                            "image_path": source["image_path"],
                            "image_width": source["image_width"],
                            "image_height": source["image_height"],
                            "claim_sentence": source["query_text"],
                            "query_text": source["query_text"],
                            # The semantic scorers accept a finding column, but strict runs leave it blank.
                            "finding": "",
                            "source_model": candidate["source_model"],
                            "rank": candidate["rank"],
                            "rank_norm": 1.0 / (1.0 + float(candidate["rank"])),
                            "confidence": candidate["confidence"],
                            "pred_x1": box[0],
                            "pred_y1": box[1],
                            "pred_x2": box[2],
                            "pred_y2": box[3],
                            "box_cx_norm": (box[0] + box[2]) * 0.5 / float(source["image_width"]),
                            "box_cy_norm": (box[1] + box[3]) * 0.5 / float(source["image_height"]),
                            "box_w_norm": width / float(source["image_width"]),
                            "box_h_norm": height / float(source["image_height"]),
                            "box_area_norm": width * height / (
                                float(source["image_width"]) * float(source["image_height"])
                            ),
                            "dino_agreement": iou_xyxy(box, dino_box) if dino_box is not None else 0.0,
                            "cross_model_agreement": float(np.mean(other_agreements)) if other_agreements else 0.0,
                            "context_score": context_score(
                                source["query_text"], box, source["image_width"], source["image_height"]
                            ),
                            "candidate_id": stable_hash(
                                f"{group_id}|{candidate['source_model']}|{candidate['rank']}|{box}", 24
                            ),
                        }
                    )
            pd.DataFrame(output_rows).to_csv(output, index=False)

    def score_semantics(self) -> None:
        if not self.config.use_semantic_experts:
            for split in SPLITS:
                source = self.paths.candidates / f"{split}.csv"
                output = self.paths.semantic / f"{split}.csv"
                if output.exists() and not self.force:
                    continue
                pd.read_csv(source).to_csv(output, index=False)
            _write_json(
                self.paths.semantic / "status.json",
                {
                    "status": "disabled_for_main_method",
                    "siglip_enabled": False,
                    "biomedclip_enabled": False,
                    "replacement": "YOLO/RAD-DINO HGB ranker plus raw-phrase rule context",
                },
            )
            return
        for split in SPLITS:
            output = self.paths.semantic / f"{split}.csv"
            if output.exists() and not self.force:
                continue
            candidates = pd.read_csv(self.paths.candidates / f"{split}.csv")
            scored = siglip.score_siglip(
                candidates,
                "google/siglip-base-patch16-224",
                "cxr_claim",
                0.15,
                32,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            scored = biomed.score_biomedclip(
                scored,
                "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                "cxr_claim",
                0.15,
                32,
            )
            scored.to_csv(output, index=False)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        _write_json(
            self.paths.semantic / "status.json",
            {
                "status": "complete",
                "siglip_enabled": True,
                "biomedclip_enabled": True,
                "query_input": "raw phrase only",
                "finding_annotation_used": False,
                "candidate_features": list(SEMANTIC_EXPERT_FEATURES),
            },
        )

    @staticmethod
    def _target_iou_for_candidates(
        frame: pd.DataFrame,
        labels: dict[str, list[list[float]]],
    ) -> np.ndarray:
        values = []
        for row in frame.itertuples():
            box = [row.pred_x1, row.pred_y1, row.pred_x2, row.pred_y2]
            gold = labels[str(row.group_id)]
            values.append(max((iou_xyxy(box, target) for target in gold), default=0.0))
        return np.asarray(values, dtype=np.float32)

    def fit_ranker(self) -> HistGradientBoostingRegressor:
        requested_features = list(BASE_RANKER_FEATURES)
        if self.config.use_semantic_experts:
            requested_features.extend(SEMANTIC_EXPERT_FEATURES)
        features = validate_ranker_schema(
            requested_features,
            include_semantic=self.config.use_semantic_experts,
        )
        inference_audit = []
        for split in ("val", "eval"):
            inference = pd.read_csv(self.paths.semantic / f"{split}.csv")
            forbidden_labels = [
                column
                for column in inference.columns
                if column in {"target_iou", "gold_boxes_xyxy", "gold_count", "bbox_name_reference", "object_name_reference"}
            ]
            finding_nonempty = int(
                inference.get("finding", pd.Series(dtype=str)).fillna("").astype(str).str.strip().ne("").sum()
            )
            inference_audit.append(
                {
                    "split": split,
                    "forbidden_label_columns": forbidden_labels,
                    "finding_annotation_nonempty_rows": finding_nonempty,
                    "model_feature_columns": features,
                    "source_model_present_as_debug_only": "source_model" in inference.columns,
                    "pass": not forbidden_labels and finding_nonempty == 0,
                }
            )
        if not all(row["pass"] for row in inference_audit):
            raise RuntimeError(f"Inference contract failed: {inference_audit}")
        _write_json(
            self.paths.ranker / "inference_contract_audit.json",
            {"status": "PASS", "rows": inference_audit},
        )
        train = pd.read_csv(self.paths.semantic / "train.csv")
        train["target_iou"] = self._target_iou_for_candidates(train, self.labels["train"])
        train.to_csv(self.paths.ranker / "ranker_training_table.csv", index=False)
        if self.config.use_semantic_experts:
            val = pd.read_csv(self.paths.semantic / "val.csv")
            val_target = self._target_iou_for_candidates(val, self.labels["val"])
            selection_rows: list[dict[str, Any]] = []
            best_key: tuple[float, float, float] | None = None
            best_model: HistGradientBoostingRegressor | None = None
            for max_leaf_nodes in (3, 7, 15):
                for min_samples_leaf in (20, 50, 100):
                    for l2_regularization in (0.1, 1.0, 5.0):
                        for max_iter in (80, 160):
                            candidate_model = HistGradientBoostingRegressor(
                                max_iter=max_iter,
                                learning_rate=0.05,
                                max_leaf_nodes=max_leaf_nodes,
                                min_samples_leaf=min_samples_leaf,
                                l2_regularization=l2_regularization,
                                random_state=self.config.pipeline_seed,
                            )
                            candidate_model.fit(
                                train[features].fillna(0.0),
                                train["target_iou"].astype(float),
                            )
                            scored_val = val[["group_id", "candidate_id"]].copy()
                            scored_val["score"] = candidate_model.predict(val[features].fillna(0.0))
                            scored_val["target_iou_selection_only"] = val_target
                            winners = (
                                scored_val.sort_values(
                                    ["group_id", "score", "candidate_id"],
                                    ascending=[True, False, True],
                                    kind="stable",
                                )
                                .groupby("group_id", sort=False)
                                .head(1)
                            )
                            mean_iou = float(winners["target_iou_selection_only"].mean())
                            hit_05 = float((winners["target_iou_selection_only"] >= 0.5).mean())
                            train_pred = candidate_model.predict(train[features].fillna(0.0))
                            train_mse = float(np.mean((train_pred - train["target_iou"].to_numpy()) ** 2))
                            selection_rows.append(
                                {
                                    "max_leaf_nodes": max_leaf_nodes,
                                    "min_samples_leaf": min_samples_leaf,
                                    "l2_regularization": l2_regularization,
                                    "max_iter": max_iter,
                                    "val_top1_mean_iou": mean_iou,
                                    "val_top1_hit_0_5": hit_05,
                                    "train_candidate_mse": train_mse,
                                }
                            )
                            key = (mean_iou, hit_05, -train_mse)
                            if best_key is None or key > best_key:
                                best_key = key
                                best_model = candidate_model
            if best_model is None:
                raise RuntimeError("Semantic HGB validation grid produced no model")
            model = best_model
            grid = pd.DataFrame(selection_rows).sort_values(
                ["val_top1_mean_iou", "val_top1_hit_0_5", "train_candidate_mse"],
                ascending=[False, False, True],
            )
            grid.to_csv(self.paths.ranker / "hgb_validation_grid.csv", index=False)
            _write_json(
                self.paths.ranker / "hgb_selection.json",
                {
                    "selection_split": "val only",
                    "selection_metric": "top1 mean IoU, then Hit@0.5",
                    "selected": grid.iloc[0].to_dict(),
                    "n_candidates": len(selection_rows),
                },
            )
        else:
            model = HistGradientBoostingRegressor(
                max_iter=160,
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=1e-3,
                random_state=self.config.pipeline_seed,
            )
            model.fit(train[features].fillna(0.0), train["target_iou"].astype(float))
        joblib.dump(
            {"model": model, "feature_columns": features, "random_state": self.config.pipeline_seed},
            self.paths.ranker / "base_candidate_ranker.joblib",
        )
        _write_json(
            self.paths.ranker / "feature_schema.json",
            {
                "model_feature_columns": features,
                "training_label": "target_iou",
                "training_label_present_only_in": "ranker_training_table.csv",
                "inference_tables_contain_training_label": False,
                "source_model_used_as_feature": False,
                "query_text_used_by": (
                    "context parser only"
                    if not self.config.use_semantic_experts
                    else "semantic experts and context parser"
                ),
            },
        )
        for split in SPLITS:
            frame = pd.read_csv(self.paths.semantic / f"{split}.csv")
            frame["base_score"] = model.predict(frame[features].fillna(0.0))
            if "target_iou" in frame.columns:
                raise RuntimeError(f"Target leaked into semantic inference table: {split}")
            frame.to_csv(self.paths.ranker / f"scored_{split}.csv", index=False)
        return model

    def _groups(self, split: str) -> dict[str, dict[str, Any]]:
        groups = {}
        for row in self.inputs[split]:
            group_id = str(row["group_id"])
            groups[group_id] = {
                "group_id": group_id,
                "subject_id": row["subject_id"],
                "finding": "",
                "claim_sentence": row["query_text"],
                "image_width": int(row["image_width"]),
                "image_height": int(row["image_height"]),
                "gt_boxes": self.labels[split][group_id],
            }
        return groups

    def _expert_bundle(self, split: str) -> moe.ExpertBundle:
        if not self.config.use_semantic_experts:
            raise RuntimeError("Semantic expert bundle requested for the no-semantic main method")
        groups = self._groups(split)
        scored = pd.read_csv(self.paths.ranker / f"scored_{split}.csv")

        def expert_map(score_column: str, source_name: str) -> dict[str, list[dict[str, Any]]]:
            mapping: dict[str, list[dict[str, Any]]] = {}
            for group_id, part in scored.groupby("group_id", sort=False):
                row = part.sort_values([score_column, "candidate_id"], ascending=[False, True]).iloc[0]
                mapping[str(group_id)] = [
                    {
                        "box": [float(row.pred_x1), float(row.pred_y1), float(row.pred_x2), float(row.pred_y2)],
                        "score": float(row[score_column]),
                        "source": source_name,
                    }
                ]
            return mapping

        hybrid = expert_map("base_score", "base_candidate_ranker")
        siglip_map = expert_map("siglip_rank", "siglip")
        biomed_map = expert_map("biomedclip_rank", "biomedclip")
        cue_info = cue_info_from_groups(groups) if self.spec.output_mode == "variable_set" else {}
        cue = pd.DataFrame(
            [
                {
                    "group_id": group_id,
                    "has_multi_cue": bool(cue_info.get(group_id, {}).get("has_multi_cue", False)),
                }
                for group_id in groups
            ]
        )
        return moe.ExpertBundle(groups, hybrid, siglip_map, biomed_map, {}, cue)

    @staticmethod
    def _boxes_only(predictions: dict[str, list[dict[str, Any]]]) -> dict[str, list[list[float]]]:
        return {
            str(group_id): [[float(value) for value in pred["box"]] for pred in rows]
            for group_id, rows in predictions.items()
        }

    def _select_cardinality_policy(
        self,
        val_groups: dict[str, dict[str, Any]],
        val_predictions: dict[str, list[dict[str, Any]]],
        seed: int,
    ) -> dict[str, Any]:
        candidates = pd.read_csv(self.paths.ranker / "scored_val.csv")
        candidate_index = prepare_candidate_index(candidates, score_col="base_score", max_top_n=400)
        cue_info = cue_info_from_groups(val_groups)
        rows = []
        for params in default_policy_grid():
            predictions, _ = apply_unified_adaptive_cardinality(
                val_predictions,
                candidate_index,
                cue_info,
                val_groups,
                params,
                score_col="base_score",
            )
            summary, _ = summarize_protocol(
                self.spec,
                self.inputs["val"],
                self.labels["val"],
                self._boxes_only(predictions),
            )
            selection_score = float(
                np.mean(
                    [
                        summary["coverage_mean_iou"],
                        summary["exact_union_iou"],
                        summary["set_f1_0_3"],
                        summary["set_f1_0_5"],
                    ]
                )
            )
            rows.append({**params, "selection_score": selection_score, **summary})
        grid = pd.DataFrame(rows).sort_values(
            ["selection_score", "set_f1_0_5", "coverage_mean_iou"], ascending=False
        )
        grid.to_csv(self.paths.gate / f"cardinality_val_grid_s{seed}.csv", index=False)
        best = grid.iloc[0]
        params = {
            key: int(best[key]) if key in {"candidate_top_n", "max_pred_count"} else float(best[key])
            for key in ("nms_iou", "target_weight", "side_target_min", "candidate_top_n", "max_pred_count")
        }
        _write_json(self.paths.gate / f"cardinality_params_s{seed}.json", params)
        return params

    def _base_ranker_predictions(
        self,
        split: str,
    ) -> tuple[dict[str, list[dict[str, Any]]], pd.DataFrame]:
        scored = pd.read_csv(self.paths.ranker / f"scored_{split}.csv")
        predictions: dict[str, list[dict[str, Any]]] = {}
        audit_rows: list[dict[str, Any]] = []
        for group_id, part in scored.groupby("group_id", sort=False):
            row = part.sort_values(
                ["base_score", "candidate_id"],
                ascending=[False, True],
                kind="stable",
            ).iloc[0]
            box = [float(row.pred_x1), float(row.pred_y1), float(row.pred_x2), float(row.pred_y2)]
            predictions[str(group_id)] = [
                {
                    "box": box,
                    "score": float(row.base_score),
                    "source": "yolo_rad_dino_hgb",
                }
            ]
            audit_rows.append(
                {
                    "group_id": str(group_id),
                    "candidate_id": str(row.candidate_id),
                    "candidate_source_debug": str(row.source_model),
                    "base_score": float(row.base_score),
                    "dino_agreement": float(row.dino_agreement),
                    "cross_model_agreement": float(row.cross_model_agreement),
                    "context_score": float(row.context_score),
                }
            )
        return predictions, pd.DataFrame(audit_rows)

    def _decode_main_without_semantic_experts(self) -> None:
        seed = self.config.pipeline_seed
        val_groups = self._groups("val")
        eval_groups = self._groups("eval")
        val_predictions, val_audit = self._base_ranker_predictions("val")
        eval_predictions, eval_audit = self._base_ranker_predictions("eval")
        val_audit.to_csv(self.paths.gate / "val_base_ranker_audit.csv", index=False)
        eval_audit.to_csv(self.paths.gate / "eval_base_ranker_audit.csv", index=False)

        cardinality_params: dict[str, Any] = {}
        if self.spec.output_mode == "variable_set":
            cardinality_params = self._select_cardinality_policy(
                val_groups,
                val_predictions,
                seed,
            )
            eval_candidates = pd.read_csv(self.paths.ranker / "scored_eval.csv")
            eval_index = prepare_candidate_index(
                eval_candidates,
                score_col="base_score",
                max_top_n=400,
            )
            eval_predictions, cardinality_audit = apply_unified_adaptive_cardinality(
                eval_predictions,
                eval_index,
                cue_info_from_groups(eval_groups),
                eval_groups,
                cardinality_params,
                score_col="base_score",
            )
            cardinality_audit.to_csv(
                self.paths.gate / "cardinality_eval_audit_main.csv",
                index=False,
            )

        boxes = self._boxes_only(eval_predictions)
        prediction_rows = [
            {
                "protocol_key": self.spec.key,
                "group_id": group_id,
                "pred_boxes_xyxy": pred_boxes,
            }
            for group_id, pred_boxes in sorted(boxes.items())
        ]
        prediction_path = self.paths.predictions / "eval_predictions_main.jsonl"
        write_jsonl(prediction_path, prediction_rows)
        summary, detail = summarize_protocol(
            self.spec,
            self.inputs["eval"],
            self.labels["eval"],
            boxes,
        )
        if self.spec.key == "mscxr_multibox_1444":
            summary.update(singleton_projection(self.inputs["eval"], self.labels["eval"], boxes))
        pd.DataFrame(detail).to_csv(self.paths.metrics / "eval_detail_main.csv", index=False)
        result = {
            "status": "complete",
            "method": "ClueGround-VFM YOLO-RAD-DINO rule-context strict local",
            "protocol_key": self.spec.key,
            "n_upstream_seeds": 1,
            "n_gate_seeds": 0,
            "pipeline_seed": seed,
            "seed_scope": "one deterministic complete pipeline; no gate-only repeats",
            "task_isolated": True,
            "rad_dino_enabled": not self.disable_rad_dino,
            "semantic_experts_enabled": False,
            "siglip_enabled": False,
            "biomedclip_enabled": False,
            "annotation_category_as_inference_input": False,
            "cig_anatomy_box_pretraining": False,
            "model_components": [
                "YOLOv8s",
                "YOLOv8m",
                "YOLO11s",
                "YOLO11m",
                "frozen RAD-DINO plus MS-CXR bbox head",
                "HGB candidate ranker",
                "raw-phrase rule-context cardinality decoder",
            ],
            "selection_split": "val only",
            "eval_prediction_generated_once": True,
            "eval_prediction_path": str(prediction_path),
            "cardinality_params": cardinality_params,
            "target_semantics": self.spec.target_semantics,
            **summary,
        }
        _write_json(self.paths.metrics / "aggregate.json", result)
        _write_json(self.paths.root / "RUN_STATUS.json", result)

    def train_gate_and_decode(self) -> None:
        # Deterministic and cheap compared with the vision stages. Refit here
        # so a partially written run cannot reuse a model without its matching
        # scored inference tables.
        self.fit_ranker()
        if not self.config.use_semantic_experts:
            self._decode_main_without_semantic_experts()
            return
        train_bundle = self._expert_bundle("train")
        val_bundle = self._expert_bundle("val")
        eval_bundle = self._expert_bundle("eval")
        moe.CKPT = self.paths.gate / "checkpoints"
        moe.LOG = self.paths.gate / "logs"
        moe.MET = self.paths.gate / "metrics"
        moe.PRED = self.paths.gate / "predictions"
        for path in (moe.CKPT, moe.LOG, moe.MET, moe.PRED):
            path.mkdir(parents=True, exist_ok=True)
        experts = ["hybrid", "siglip", "biomed"]
        seed_summaries = []
        for seed in self.config.gate_seeds:
            name = f"{self.spec.key}_hybrid_siglip_biomed_s{seed}"
            model, params, _, _ = moe.train_gate(
                name,
                train_bundle,
                val_bundle,
                experts,
                hardneg=False,
                seed=seed,
                device=str(self.torch_device),
            )
            val_predictions, val_audit = moe.predict_gate(
                name,
                model,
                val_bundle,
                experts,
                device=str(self.torch_device),
                keep_multi=False,
            )
            eval_predictions, eval_audit = moe.predict_gate(
                name,
                model,
                eval_bundle,
                experts,
                device=str(self.torch_device),
                keep_multi=False,
            )
            if self.spec.output_mode == "variable_set":
                cardinality_params = self._select_cardinality_policy(
                    val_bundle.groups,
                    val_predictions,
                    seed,
                )
                eval_candidates = pd.read_csv(self.paths.ranker / "scored_eval.csv")
                eval_index = prepare_candidate_index(eval_candidates, score_col="base_score", max_top_n=400)
                eval_predictions, cardinality_audit = apply_unified_adaptive_cardinality(
                    eval_predictions,
                    eval_index,
                    cue_info_from_groups(eval_bundle.groups),
                    eval_bundle.groups,
                    cardinality_params,
                    score_col="base_score",
                )
                cardinality_audit.to_csv(self.paths.gate / f"cardinality_eval_audit_s{seed}.csv", index=False)
            boxes = self._boxes_only(eval_predictions)
            prediction_rows = [
                {"protocol_key": self.spec.key, "group_id": group_id, "pred_boxes_xyxy": pred_boxes}
                for group_id, pred_boxes in sorted(boxes.items())
            ]
            write_jsonl(self.paths.predictions / f"eval_predictions_s{seed}.jsonl", prediction_rows)
            val_audit.to_csv(self.paths.gate / f"val_gate_audit_s{seed}.csv", index=False)
            eval_audit.to_csv(self.paths.gate / f"eval_gate_audit_s{seed}.csv", index=False)
            summary, detail = summarize_protocol(self.spec, self.inputs["eval"], self.labels["eval"], boxes)
            if self.spec.key == "mscxr_multibox_1444":
                summary.update(singleton_projection(self.inputs["eval"], self.labels["eval"], boxes))
            pd.DataFrame(detail).to_csv(self.paths.metrics / f"eval_detail_s{seed}.csv", index=False)
            seed_summaries.append({"gate_seed": seed, **params, **summary})
        result = pd.DataFrame(seed_summaries)
        result.to_csv(self.paths.metrics / "eval_by_gate_seed.csv", index=False)
        metric_columns = [column for column in self.spec.primary_metrics if column in result]
        aggregate = {
            "protocol_key": self.spec.key,
            "n_upstream_seeds": 1,
            "n_gate_seeds": len(self.config.gate_seeds),
            "pipeline_seed": self.config.pipeline_seed,
            "seed_scope": "one fixed upstream pipeline; gate seeds only",
            "task_isolated": True,
            "rad_dino_enabled": not self.disable_rad_dino,
            "target_semantics": self.spec.target_semantics,
        }
        for column in metric_columns:
            aggregate[column] = float(result[column].mean())
            aggregate[f"{column}_std_gate_only"] = float(result[column].std(ddof=1)) if len(result) > 1 else 0.0
        _write_json(self.paths.metrics / "aggregate.json", aggregate)
        _write_json(self.paths.root / "RUN_STATUS.json", {"status": "complete", **aggregate})

    def run(self, stages: Iterable[str]) -> None:
        requested = [stage.strip() for stage in stages if stage.strip()]
        if "yolo_data" in requested:
            self.build_yolo_dataset()
        weights_path = self.paths.yolo_runs / "weights_manifest.json"
        if "yolo" in requested:
            weights = self.train_yolo()
            self.predict_yolo(weights)
        elif weights_path.exists():
            weights = {key: Path(value) for key, value in json.loads(weights_path.read_text(encoding="utf-8")).items()}
        else:
            weights = {}
        if "rad_dino" in requested:
            self.train_rad_dino_head()
        if "candidates" in requested:
            if not list(self.paths.yolo_predictions.glob("*_train.csv")) and weights:
                self.predict_yolo(weights)
            self.build_candidates()
        if "semantic" in requested:
            self.score_semantics()
        if "ranker_gate" in requested:
            self.train_gate_and_decode()


def planned_stages() -> tuple[str, ...]:
    return ("yolo_data", "yolo", "rad_dino", "candidates", "semantic", "ranker_gate")

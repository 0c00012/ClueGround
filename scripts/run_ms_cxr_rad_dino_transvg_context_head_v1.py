#!/usr/bin/env python
"""RAD-DINO + TransVG-style context head experiment.

This script keeps the existing hybrid YOLO+DINO direction, but strengthens the
RAD-DINO context branch. It uses frozen RAD-DINO patch tokens and trains only a
small query-to-patch cross-attention bbox head.

MS-CXR boxes are phrase-grounding boxes, not pixel-level lesion masks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models_ms_cxr_vfm_localizer import bbox_loss as base_bbox_loss


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
EXP_NAME = "ms_cxr_rad_dino_transvg_context_head_v1"
EXP = PROJECT_ROOT / "experiments" / EXP_NAME
TRAINING = PROJECT_ROOT / "training" / EXP_NAME
REPORT = PROJECT_ROOT / "reports" / EXP_NAME
DATA = TRAINING / "datasets"
RUNS = TRAINING / "runs"
CKPT = TRAINING / "checkpoints"
PRED = EXP / "predictions"
MET = EXP / "metrics"

STAGE1_SPLIT = PROJECT_ROOT / "training" / "ms_cxr_vfm_localizer_stage1_p10_p19" / "datasets"
MS_FEATURES = PROJECT_ROOT / "features" / "ms_cxr_vfm_localizer_stage1_p10_p19"
SINGLEBOX_ROOT = PROJECT_ROOT / "experiments" / "ms_cxr_medrpg_fair_retrain_final_v1" / "data" / "single_box_full_phrase"
FAIR_888 = PROJECT_ROOT / "experiments" / "fair_retrain_ms_cxr_888_1444_comparison_v1" / "metrics" / "fair_888_singlebox_main.csv"
FAIR_1444_GROUP = PROJECT_ROOT / "experiments" / "fair_retrain_ms_cxr_888_1444_comparison_v1" / "metrics" / "fair_1444_phrase_group_main.csv"
AGPT_ROOT = PROJECT_ROOT / "third_party" / "AGPT"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FINDINGS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Lung Opacity",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def ensure_dirs() -> None:
    for root in [EXP, TRAINING, REPORT, DATA, RUNS, CKPT, PRED, MET]:
        root.mkdir(parents=True, exist_ok=True)
    for sub in ["metrics", "metadata", "configs", "logs", "predictions", "contact_sheets"]:
        (EXP / sub).mkdir(parents=True, exist_ok=True)


def norm_text(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip().lower()


def as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def extract_laterality(text: str) -> str:
    t = norm_text(text)
    if re.search(r"\bbilateral|bibasilar|both\b", t):
        return "bilateral"
    if re.search(r"\bright\b", t):
        return "right"
    if re.search(r"\bleft\b", t):
        return "left"
    return "unknown"


def extract_vertical(text: str) -> str:
    t = norm_text(text)
    if re.search(r"\bapical|apex\b", t):
        return "apical"
    if re.search(r"\bbasal|base|basilar|lower\b", t):
        return "lower"
    if re.search(r"\bupper\b", t):
        return "upper"
    if re.search(r"\bmid|middle|lingular\b", t):
        return "mid"
    return "unknown"


def extract_uncertainty(text: str) -> str:
    t = norm_text(text)
    if re.search(r"\bno |without|absent|resolved|removed\b", t):
        return "negated"
    if re.search(r"\bpossible|possibly|may|might|question|suggest|likely\b", t):
        return "uncertain"
    return "present"


def make_rule_context_text(row: dict[str, Any]) -> str:
    finding = str(row.get("finding") or row.get("finding_label") or "")
    claim = str(row.get("claim_sentence") or row.get("phrase") or row.get("sentence") or row.get("phrase_text") or finding)
    parts = [finding]
    lat = extract_laterality(claim)
    vert = extract_vertical(claim)
    unc = extract_uncertainty(claim)
    if lat != "unknown":
        parts.append(lat)
    if vert != "unknown":
        parts.append(vert)
    if unc not in {"present", "unknown"}:
        parts.append(unc)
    return " ".join([p for p in parts if p]).strip()


def query_text_for_row(row: dict[str, Any], query_variant: str) -> str:
    if query_variant == "label_only":
        return str(row.get("finding") or row.get("finding_label") or "")
    if query_variant == "full_phrase":
        return str(row.get("claim_sentence") or row.get("phrase") or row.get("sentence") or row.get("phrase_text") or row.get("finding") or "")
    if query_variant == "rule_context":
        return str(row.get("rule_context_text") or make_rule_context_text(row))
    if query_variant == "rule_plus_full":
        rule = str(row.get("rule_context_text") or make_rule_context_text(row))
        phrase = str(row.get("claim_sentence") or row.get("phrase") or row.get("sentence") or row.get("phrase_text") or "")
        return f"{rule}; phrase {phrase}"
    raise ValueError(f"unknown query_variant={query_variant}")


def xyxy_to_norm_cxcywh(x1: float, y1: float, x2: float, y2: float, width: float, height: float) -> list[float]:
    width = max(float(width), 1.0)
    height = max(float(height), 1.0)
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = max(1.0, x2 - x1) / width
    bh = max(1.0, y2 - y1) / height
    return [float(np.clip(cx, 0, 1)), float(np.clip(cy, 0, 1)), float(np.clip(bw, 1e-4, 1)), float(np.clip(bh, 1e-4, 1))]


def cxcywh_to_xyxy_np(box: np.ndarray) -> np.ndarray:
    cx, cy, w, h = box
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype="float32")


def iou_xyxy_np(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(inter / max(area_a + area_b - inter, 1e-6))


def split_summary() -> dict[str, Any]:
    train = read_jsonl(STAGE1_SPLIT / "train.jsonl")
    val = read_jsonl(STAGE1_SPLIT / "val.jsonl")
    eval_rows = read_jsonl(STAGE1_SPLIT / "eval.jsonl")
    train_subjects = {str(r.get("subject_id")) for r in train}
    eval_subjects = {str(r.get("subject_id")) for r in eval_rows}
    return {
        "train_rows": len(train),
        "val_rows": len(val),
        "eval_rows": len(eval_rows),
        "train_eval_subject_overlap": len(train_subjects & eval_subjects),
        "split_source": str(STAGE1_SPLIT),
    }


def artifact_inventory() -> list[dict[str, Any]]:
    paths = [
        ("stage1_train_jsonl", STAGE1_SPLIT / "train.jsonl", "required"),
        ("stage1_val_jsonl", STAGE1_SPLIT / "val.jsonl", "required"),
        ("stage1_eval_jsonl", STAGE1_SPLIT / "eval.jsonl", "required"),
        ("row_level_rad_dino_features", MS_FEATURES / "vfm_features_train.npz", "required_feature_cache"),
        ("singlebox_train_csv", SINGLEBOX_ROOT / "train.csv", "strict_888_dataset"),
        ("singlebox_val_csv", SINGLEBOX_ROOT / "val.csv", "strict_888_dataset"),
        ("singlebox_eval_csv", SINGLEBOX_ROOT / "eval.csv", "strict_888_dataset"),
        ("fair_888_singlebox_table", FAIR_888, "comparison_reference"),
        ("fair_1444_phrase_group_table", FAIR_1444_GROUP, "comparison_reference"),
        ("agpt_repo", AGPT_ROOT, "transvg_reference_code"),
        ("agpt_transvg_weight", AGPT_ROOT / "model_weight" / "transvg.pth", "reference_only_weight"),
        ("agpt_mdetr_weight", AGPT_ROOT / "model_weight" / "mdetr.pth", "reference_only_weight"),
        ("transvg_model_code", AGPT_ROOT / "models" / "transvg" / "trans_vg_ca.py", "architecture_reference"),
        ("transvg_attention_code", AGPT_ROOT / "models" / "transvg" / "MHA.py", "architecture_reference"),
    ]
    rows = []
    for name, path, role in paths:
        rows.append(
            {
                "artifact": name,
                "path": str(path),
                "exists": path.exists(),
                "role": role,
                "size_mb": round(path.stat().st_size / (1024 * 1024), 2) if path.exists() and path.is_file() else "",
            }
        )
    return rows


def make_plan(split: dict[str, Any], quick: bool) -> dict[str, Any]:
    return {
        "experiment_name": EXP_NAME,
        "plain_korean_method_name": "RAD-DINO 맥락교차 head",
        "main_method_kept": "hybrid YOLO+DINO",
        "purpose": "Use a TransVG-style cross-attention module to strengthen the RAD-DINO context branch, then fuse it with YOLO proposals.",
        "quick_mode": quick,
        "split": split,
        "stages": [
            {"stage": 0, "name": "audit_and_contract", "status": "complete", "description": "Verify splits, artifacts, and fairness constraints."},
            {"stage": 1, "name": "rad_dino_patch_cache_reuse", "status": "implemented", "description": "Reuse row-level RAD-DINO patch tokens."},
            {"stage": 2, "name": "transvg_style_context_head", "status": "implemented", "description": "Train query-to-patch cross-attention bbox head."},
            {"stage": 3, "name": "candidate_aware_yolo_fusion", "status": "pending", "description": "Use the new branch as a context scorer over YOLO candidate boxes."},
            {"stage": 4, "name": "multibox_set_extension", "status": "pending", "description": "For bilateral/multifocal phrases, score and output multiple boxes."},
            {"stage": 5, "name": "fair_eval", "status": "pending", "description": "Evaluate 888 strict, 1444 row-level, and 1444 phrase-group."},
        ],
        "fairness_rules": [
            "Do not use AGPT/MAIRA released checkpoint outputs as training labels unless explicitly marked teacher-distillation and train split only.",
            "Do not evaluate on train/val as main result.",
            "Use val only for fusion/grid/checkpoint selection.",
            "Report 888 single-box and 1444 full phrase-group separately.",
            "MS-CXR boxes are phrase-grounding boxes, not lesion pixel masks.",
        ],
    }


def write_stage0_report(split: dict[str, Any], inventory: list[dict[str, Any]], plan: dict[str, Any]) -> None:
    fair888 = read_csv(FAIR_888)
    fair1444 = read_csv(FAIR_1444_GROUP)

    def rows_to_md(rows: list[dict[str, str]], score_key: str) -> str:
        lines = ["| method | score | status |", "| --- | ---: | --- |"]
        for r in rows[:8]:
            score = r.get(score_key, r.get("mean_iou", ""))
            lines.append(f"| {r.get('method', '')} | {score} | {r.get('fair_main_usable', '')} |")
        return "\n".join(lines)

    inv_md = "\n".join([f"- {r['artifact']}: {'OK' if r['exists'] else 'MISSING'} ({r['role']})" for r in inventory])
    report = f"""# RAD-DINO 맥락교차 head v1

## 한 줄 결론

이 실험은 기존 메인 방법론인 **hybrid YOLO+DINO**를 버리는 것이 아니라, RAD-DINO 쪽 문맥 branch를 TransVG-style cross-attention으로 강화하는 방향이다.

## 왜 하는가

기존 hybrid YOLO+DINO는 후보 생성과 fusion은 강했지만, RAD-DINO branch의 문맥 이해는 rule-context를 얕게 붙인 형태였다. 그래서 좋은 후보가 pool 안에 있어도 최종 선택이 불안정했다.

```text
claim / rule-context / full phrase
  -> query embedding
CXR image
  -> RAD-DINO patch tokens
query embedding x RAD-DINO patch tokens
  -> TransVG-style cross-attention
  -> context-aware heatmap / bbox / candidate score
YOLO proposals + context-aware RAD-DINO score
  -> final one-box or multi-box output
```

## 현재 split 확인

- train rows: {split['train_rows']}
- val rows: {split['val_rows']}
- eval rows: {split['eval_rows']}
- train/eval subject overlap: {split['train_eval_subject_overlap']}

## 입력 artifact 확인

{inv_md}

## 기존 fair baseline 참고

### 888 single-box strict

{rows_to_md(fair888, 'mean_iou')}

### 1444 full phrase-group

{rows_to_md(fair1444, 'union_iou')}

## 주의

- TransVG 자체를 baseline으로 베끼는 것이 아니라, RAD-DINO context branch 안의 cross-attention 모듈로 사용한다.
- AGPT/MAIRA released output은 reference-only다. train teacher로 쓰려면 train split에서만 distillation이라고 명시해야 한다.
- 논문 표현은 `RAD-DINO 맥락교차 head` 또는 `TransVG-style context head` 정도로 제한한다.
"""
    write_text(REPORT / "README_KO.md", report)


def stage0_audit(quick: bool) -> dict[str, Any]:
    ensure_dirs()
    split = split_summary()
    inventory = artifact_inventory()
    plan = make_plan(split, quick)
    write_csv(EXP / "metadata" / "input_artifact_inventory.csv", inventory)
    write_text(EXP / "configs" / "experiment_plan.json", json.dumps(plan, indent=2, ensure_ascii=False))
    write_stage0_report(split, inventory, plan)
    return {"split": split, "inventory": inventory}


def feature_check() -> dict[str, Any]:
    ensure_dirs()
    rows: list[dict[str, Any]] = []
    ok = True
    for split in ["train", "val", "eval"]:
        fpath = MS_FEATURES / f"vfm_features_{split}.npz"
        if not fpath.exists():
            ok = False
            rows.append({"split": split, "exists": False})
            continue
        z = np.load(fpath, allow_pickle=True)
        split_rows = read_jsonl(STAGE1_SPLIT / f"{split}.jsonl")
        task_ids = [str(x) for x in z["task_ids"]]
        json_ids = [str(r["task_id"]) for r in split_rows]
        rows.append(
            {
                "split": split,
                "exists": True,
                "json_rows": len(split_rows),
                "feature_rows": len(task_ids),
                "task_id_order_match": task_ids == json_ids,
                "global_shape": "x".join(map(str, z["global_features"].shape)),
                "patch_shape": "x".join(map(str, z["patch_tokens"].shape)),
                "patch_dtype": str(z["patch_tokens"].dtype),
            }
        )
    write_csv(EXP / "metadata" / "rad_dino_patch_feature_schema.csv", rows)
    write_text(
        REPORT / "STAGE1_FEATURE_SCHEMA_REPORT.md",
        "# Stage1 RAD-DINO Feature Schema Report\n\n"
        + pd.DataFrame(rows).to_markdown(index=False)
        + "\n\nRAD-DINO feature cache is used as frozen patch-token input. No eval gold is used during training or selection.\n",
    )
    return {"ok": ok, "rows": rows}


def build_protocol_rows(protocol: str, query_variant: str, split: str) -> list[dict[str, Any]]:
    base_rows = read_jsonl(STAGE1_SPLIT / f"{split}.jsonl")
    by_ann = {str(r.get("ms_cxr_annotation_id")): r for r in base_rows}
    out: list[dict[str, Any]] = []
    if protocol == "row1444":
        for r in base_rows:
            row = dict(r)
            row["protocol"] = protocol
            row["sample_id"] = row["task_id"]
            row["query_variant"] = query_variant
            row["query_text"] = query_text_for_row(row, query_variant)
            out.append(row)
        return out

    if protocol != "singlebox888":
        raise ValueError(f"unknown protocol={protocol}")

    csv_rows = read_csv(SINGLEBOX_ROOT / f"{split}.csv")
    missing = 0
    for cr in csv_rows:
        ann_id = str(cr.get("source_annotation_id"))
        base = by_ann.get(ann_id)
        if base is None:
            missing += 1
            continue
        row = dict(base)
        row["protocol"] = protocol
        row["sample_id"] = cr["sample_id"]
        row["group_id"] = cr.get("group_id", "")
        row["n_boxes_in_group"] = int(as_float(cr.get("n_boxes_in_group"), 1))
        row["finding"] = cr.get("finding_label") or row.get("finding")
        row["claim_sentence"] = cr.get("phrase_text") or cr.get("full_phrase_text") or row.get("claim_sentence")
        row["phrase"] = row["claim_sentence"]
        row["rule_context_text"] = cr.get("rule_context_text") or make_rule_context_text(row)
        x1, y1, x2, y2 = [as_float(cr.get(k)) for k in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]]
        iw, ih = as_float(cr.get("image_width"), row.get("image_width", 1)), as_float(cr.get("image_height"), row.get("image_height", 1))
        row["gold_bbox_xyxy"] = [x1, y1, x2, y2]
        row["gold_bbox_norm_cxcywh"] = xyxy_to_norm_cxcywh(x1, y1, x2, y2, iw, ih)
        row["image_width"] = iw
        row["image_height"] = ih
        row["query_variant"] = query_variant
        row["query_text"] = query_text_for_row(row, query_variant)
        out.append(row)
    if missing:
        print(f"warning_missing_singlebox_matches split={split} missing={missing}", flush=True)
    return out


def build_datasets(protocol: str, query_variant: str) -> dict[str, Any]:
    ensure_dirs()
    stats = {}
    for split in ["train", "val", "eval"]:
        rows = build_protocol_rows(protocol, query_variant, split)
        write_jsonl(DATA / f"{protocol}_{query_variant}_{split}.jsonl", rows)
        stats[split] = {
            "rows": len(rows),
            "unique_subjects": len({str(r.get("subject_id")) for r in rows}),
            "unique_images": len({str(r.get("dicom_id")) for r in rows}),
        }
    train_subjects = {str(r.get("subject_id")) for r in read_jsonl(DATA / f"{protocol}_{query_variant}_train.jsonl")}
    eval_subjects = {str(r.get("subject_id")) for r in read_jsonl(DATA / f"{protocol}_{query_variant}_eval.jsonl")}
    stats["train_eval_subject_overlap"] = len(train_subjects & eval_subjects)
    write_text(
        REPORT / f"STAGE2_DATASET_{protocol}_{query_variant}_REPORT.md",
        f"# Dataset Report: {protocol} / {query_variant}\n\n"
        f"```json\n{json.dumps(stats, indent=2, ensure_ascii=False)}\n```\n\n"
        "MS-CXR boxes are phrase-grounding boxes, not lesion masks.\n",
    )
    return stats


def build_query_features(protocol: str, query_variant: str, query_dim: int, max_features: int) -> dict[str, Any]:
    rows_by_split = {s: read_jsonl(DATA / f"{protocol}_{query_variant}_{s}.jsonl") for s in ["train", "val", "eval"]}
    train_texts = [r["query_text"] for r in rows_by_split["train"]]
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=1, max_features=max_features)
    x_train = vectorizer.fit_transform(train_texts)
    use_svd = x_train.shape[1] > query_dim and x_train.shape[0] > 2
    svd = None
    if use_svd:
        n_components = min(query_dim, x_train.shape[1] - 1, x_train.shape[0] - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=13)
        svd.fit(x_train)
    for split, rows in rows_by_split.items():
        sparse = vectorizer.transform([r["query_text"] for r in rows])
        if svd is not None:
            x = svd.transform(sparse).astype("float32")
        else:
            x = sparse.toarray().astype("float32")
        if x.shape[1] < query_dim:
            x = np.concatenate([x, np.zeros((x.shape[0], query_dim - x.shape[1]), dtype="float32")], axis=1)
        elif x.shape[1] > query_dim:
            x = x[:, :query_dim]
        np.savez_compressed(
            DATA / f"{protocol}_{query_variant}_query_features_{split}.npz",
            task_ids=np.array([r["task_id"] for r in rows], dtype=object),
            sample_ids=np.array([r.get("sample_id", r["task_id"]) for r in rows], dtype=object),
            features=x.astype("float32"),
        )
    with (DATA / f"{protocol}_{query_variant}_query_vectorizer.pkl").open("wb") as f:
        pickle.dump({"vectorizer": vectorizer, "svd": svd, "query_dim": query_dim}, f)
    stats = {
        "protocol": protocol,
        "query_variant": query_variant,
        "train_rows": len(rows_by_split["train"]),
        "vocab_size": len(vectorizer.vocabulary_),
        "query_dim": query_dim,
        "svd_used": svd is not None,
    }
    write_text(
        REPORT / f"STAGE2_QUERY_FEATURE_{protocol}_{query_variant}_REPORT.md",
        f"# Query Feature Report: {protocol} / {query_variant}\n\n```json\n{json.dumps(stats, indent=2, ensure_ascii=False)}\n```\n",
    )
    return stats


class MSContextDataset(Dataset):
    def __init__(self, protocol: str, query_variant: str, split: str):
        self.protocol = protocol
        self.query_variant = query_variant
        self.split = split
        self.rows = read_jsonl(DATA / f"{protocol}_{query_variant}_{split}.jsonl")
        feat = np.load(MS_FEATURES / f"vfm_features_{split}.npz", allow_pickle=True)
        q = np.load(DATA / f"{protocol}_{query_variant}_query_features_{split}.npz", allow_pickle=True)
        self.patch_tokens = feat["patch_tokens"]
        self.feature_index = {str(t): i for i, t in enumerate(feat["task_ids"])}
        self.query_features = q["features"]
        self.query_index = {str(t): i for i, t in enumerate(q["task_ids"])}
        self.rows = [r for r in self.rows if str(r["task_id"]) in self.feature_index and str(r["task_id"]) in self.query_index]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        fidx = self.feature_index[str(row["task_id"])]
        qidx = self.query_index[str(row["task_id"])]
        tokens = torch.tensor(self.patch_tokens[fidx], dtype=torch.float32)
        query = torch.tensor(self.query_features[qidx], dtype=torch.float32)
        target = torch.tensor(row["gold_bbox_norm_cxcywh"], dtype=torch.float32)
        grid = int(round(math.sqrt(tokens.shape[0])))
        cx, cy = float(target[0]), float(target[1])
        gx = min(grid - 1, max(0, int(round(cx * (grid - 1)))))
        gy = min(grid - 1, max(0, int(round(cy * (grid - 1)))))
        center_index = torch.tensor(gy * grid + gx, dtype=torch.long)
        return tokens, query, target, center_index, idx


class RadDinoContextCrossAttentionHead(nn.Module):
    """Small TransVG-style query-to-patch head on frozen RAD-DINO tokens."""

    def __init__(
        self,
        token_dim: int = 768,
        query_dim: int = 128,
        hidden: int = 256,
        num_query_tokens: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_patches: int = 1369,
    ):
        super().__init__()
        self.num_query_tokens = num_query_tokens
        self.token_norm = nn.LayerNorm(token_dim)
        self.token_proj = nn.Linear(token_dim, hidden)
        self.query_proj = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.Linear(query_dim, hidden * num_query_tokens),
            nn.GELU(),
        )
        self.pos = nn.Parameter(torch.zeros(1, max_patches, hidden))
        nn.init.normal_(self.pos, std=0.02)
        self.cross_attn = nn.MultiheadAttention(hidden, num_heads, dropout=dropout, batch_first=True)
        self.q_norm = nn.LayerNorm(hidden)
        self.query_to_patch = nn.Linear(hidden, hidden)
        self.patch_score = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        coords = self._make_grid_coords(max_patches)
        self.register_buffer("grid_coords", coords, persistent=False)
        self.bbox = nn.Sequential(
            nn.Linear(token_dim + hidden + query_dim + 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 4),
            nn.Sigmoid(),
        )

    @staticmethod
    def _make_grid_coords(max_patches: int) -> torch.Tensor:
        grid = int(round(math.sqrt(max_patches)))
        ys, xs = torch.meshgrid(torch.linspace(0, 1, grid), torch.linspace(0, 1, grid), indexing="ij")
        return torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1)

    def forward(self, tokens: torch.Tensor, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, n_patches, _ = tokens.shape
        patch_h = self.token_proj(self.token_norm(tokens)) + self.pos[:, :n_patches, :]
        query_h = self.query_proj(query).view(bsz, self.num_query_tokens, -1)
        q_out, _attn = self.cross_attn(query_h, patch_h, patch_h, need_weights=False)
        q_summary = self.q_norm(q_out.mean(dim=1))
        conditioned = patch_h + self.query_to_patch(q_summary).unsqueeze(1)
        patch_logits = self.patch_score(conditioned).squeeze(-1)
        patch_weights = torch.softmax(patch_logits, dim=1)
        pooled = torch.bmm(patch_weights.unsqueeze(1), tokens).squeeze(1)
        coords = self.grid_coords[:n_patches].to(tokens.device, tokens.dtype)
        soft_xy = torch.matmul(patch_weights, coords)
        out = self.bbox(torch.cat([pooled, q_summary, query, soft_xy], dim=1))
        box = torch.cat([out[:, :2], out[:, 2:].clamp(min=0.02, max=1.0)], dim=1)
        return box, patch_logits


def model_loss(pred: torch.Tensor, target: torch.Tensor, patch_logits: torch.Tensor, center_index: torch.Tensor, center_weight: float) -> torch.Tensor:
    b_loss, _ = base_bbox_loss(pred, target)
    center_loss = nn.functional.cross_entropy(patch_logits, center_index)
    return b_loss + center_weight * center_loss


def summarize_predictions(df: pd.DataFrame, method: str, protocol: str, query_variant: str) -> dict[str, Any]:
    arr = df["iou"].astype(float).to_numpy()
    return {
        "method": method,
        "protocol": protocol,
        "query_variant": query_variant,
        "n": int(len(arr)),
        "mean_iou": float(arr.mean()) if len(arr) else 0.0,
        "median_iou": float(np.median(arr)) if len(arr) else 0.0,
        "Hit@0.1": float((arr >= 0.1).mean()) if len(arr) else 0.0,
        "Hit@0.3": float((arr >= 0.3).mean()) if len(arr) else 0.0,
        "Hit@0.5": float((arr >= 0.5).mean()) if len(arr) else 0.0,
        "bbox_missing_rate": 0.0,
        "bbox_invalid_rate": 0.0,
    }


@torch.no_grad()
def eval_model(model: nn.Module, ds: MSContextDataset, batch_size: int, save_path: Path | None = None) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    preds: list[dict[str, Any]] = []
    for tokens, query, target, center_index, idxs in loader:
        tokens = tokens.to(DEVICE, non_blocking=True)
        query = query.to(DEVICE, non_blocking=True)
        pred, logits = model(tokens, query)
        pred_np = pred.detach().cpu().numpy()
        for j, box in enumerate(pred_np):
            row = ds.rows[int(idxs[j])]
            box = np.clip(box, 0, 1)
            gold = np.array(row["gold_bbox_norm_cxcywh"], dtype="float32")
            iou = iou_xyxy_np(cxcywh_to_xyxy_np(box), cxcywh_to_xyxy_np(gold))
            iw, ih = float(row["image_width"]), float(row["image_height"])
            x1, y1, x2, y2 = cxcywh_to_xyxy_np(box)
            preds.append(
                {
                    "sample_id": row.get("sample_id", row["task_id"]),
                    "task_id": row["task_id"],
                    "ms_cxr_annotation_id": row.get("ms_cxr_annotation_id"),
                    "protocol": ds.protocol,
                    "query_variant": ds.query_variant,
                    "finding": row.get("finding"),
                    "claim_sentence": row.get("claim_sentence"),
                    "query_text": row.get("query_text"),
                    "gold_bbox_norm_cxcywh": [float(x) for x in gold],
                    "pred_bbox_norm_cxcywh": [float(x) for x in box],
                    "gold_bbox_xyxy": row.get("gold_bbox_xyxy"),
                    "pred_bbox_xyxy": [float(x1 * iw), float(y1 * ih), float(x2 * iw), float(y2 * ih)],
                    "image_width": iw,
                    "image_height": ih,
                    "iou": iou,
                    "hit_0_1": int(iou >= 0.1),
                    "hit_0_3": int(iou >= 0.3),
                    "hit_0_5": int(iou >= 0.5),
                }
            )
    if save_path is not None:
        write_jsonl(save_path, preds)
    return pd.DataFrame(preds), preds


def train_context_head(args: argparse.Namespace, protocol: str, query_variant: str) -> dict[str, Any]:
    train_ds = MSContextDataset(protocol, query_variant, "train")
    val_ds = MSContextDataset(protocol, query_variant, "val")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"empty dataset for {protocol}/{query_variant}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    token_dim = int(train_ds[0][0].shape[-1])
    query_dim = int(train_ds[0][1].shape[-1])
    model = RadDinoContextCrossAttentionHead(
        token_dim=token_dim,
        query_dim=query_dim,
        hidden=args.hidden,
        num_query_tokens=args.num_query_tokens,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=(DEVICE == "cuda"))
    best = -1.0
    run_name = f"{protocol}_{query_variant}_s{args.seed}"
    ckpt_path = CKPT / f"{run_name}_best.pt"
    curve: list[dict[str, Any]] = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for tokens, query, target, center_idx, _ in loader:
            tokens = tokens.to(DEVICE, non_blocking=True)
            query = query.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)
            center_idx = center_idx.to(DEVICE, non_blocking=True)
            pred, logits = model(tokens, query)
            loss = model_loss(pred, target, logits, center_idx, args.center_loss_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_df, _ = eval_model(model, val_ds, args.batch_size)
        val_summary = summarize_predictions(val_df, "rad_dino_context_cross_attention", protocol, query_variant)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "val_mean_iou": val_summary["mean_iou"],
            "val_hit_0_3": val_summary["Hit@0.3"],
            "val_hit_0_5": val_summary["Hit@0.5"],
        }
        curve.append(row)
        print(row, flush=True)
        if val_summary["mean_iou"] > best:
            best = float(val_summary["mean_iou"])
            torch.save(
                {
                    "model": model.state_dict(),
                    "token_dim": token_dim,
                    "query_dim": query_dim,
                    "args": vars(args),
                    "protocol": protocol,
                    "query_variant": query_variant,
                    "best_epoch": epoch,
                    "best_val_mean_iou": best,
                },
                ckpt_path,
            )
    pd.DataFrame(curve).to_csv(MET / f"{run_name}_training_curve.csv", index=False)
    stats = {
        "protocol": protocol,
        "query_variant": query_variant,
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "epochs": args.epochs,
        "best_val_mean_iou": best,
        "checkpoint": str(ckpt_path),
        "elapsed_sec": round(time.time() - t0, 2),
        "device": DEVICE,
    }
    write_text(
        REPORT / f"STAGE3_TRAINING_{protocol}_{query_variant}_REPORT.md",
        f"# Training Report: {protocol} / {query_variant}\n\n"
        f"```json\n{json.dumps(stats, indent=2, ensure_ascii=False)}\n```\n\n"
        "RAD-DINO is frozen; only the query-to-patch cross-attention head is trained.\n",
    )
    return stats


def evaluate_context_head(args: argparse.Namespace, protocol: str, query_variant: str) -> dict[str, Any]:
    ds = MSContextDataset(protocol, query_variant, "eval")
    ckpt_path = CKPT / f"{protocol}_{query_variant}_s{args.seed}_best.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = RadDinoContextCrossAttentionHead(
        token_dim=ckpt["token_dim"],
        query_dim=ckpt["query_dim"],
        hidden=args.hidden,
        num_query_tokens=args.num_query_tokens,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    df, _ = eval_model(model, ds, args.batch_size, PRED / f"{protocol}_{query_variant}_eval_predictions.jsonl")
    summary = summarize_predictions(df, "rad_dino_context_cross_attention", protocol, query_variant)
    pd.DataFrame([summary]).to_csv(MET / f"{protocol}_{query_variant}_eval_summary.csv", index=False)
    per_finding_rows = []
    for finding, sub in df.groupby("finding", dropna=False):
        per_finding_rows.append(summarize_predictions(sub, "rad_dino_context_cross_attention", protocol, query_variant) | {"finding": finding})
    pd.DataFrame(per_finding_rows).to_csv(MET / f"{protocol}_{query_variant}_per_finding_metrics.csv", index=False)
    write_text(
        REPORT / f"STAGE4_EVAL_{protocol}_{query_variant}_REPORT.md",
        f"# Eval Report: {protocol} / {query_variant}\n\n"
        + pd.DataFrame([summary]).to_markdown(index=False)
        + "\n\nMS-CXR boxes are phrase-grounding boxes, not lesion masks.\n",
    )
    return summary


def update_combined_summary(results: list[dict[str, Any]]) -> None:
    if not results:
        return
    out = MET / "context_head_combined_summary.csv"
    existing = pd.read_csv(out).to_dict("records") if out.exists() else []
    key = lambda r: (r.get("protocol"), r.get("query_variant"), r.get("method"))
    merged = {key(r): r for r in existing}
    for r in results:
        merged[key(r)] = r
    rows = list(merged.values())
    rows.sort(key=lambda r: float(r.get("mean_iou", 0)), reverse=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    write_text(
        REPORT / "CONTEXT_HEAD_CURRENT_RESULT_KO.md",
        "# RAD-DINO 맥락교차 head 현재 결과\n\n"
        + pd.DataFrame(rows).to_markdown(index=False)
        + "\n\n## 해석\n\n"
        "- 이 표는 새 cross-attention head 자체의 결과다.\n"
        "- YOLO 후보 fusion은 아직 붙이지 않았다.\n"
        "- 888 single-box에서 기존 RAD-DINO full phrase head 0.4536을 넘는지가 1차 gate다.\n"
        "- 888 single-box에서 기존 hybrid YOLO+DINO 0.5063을 넘거나 가까워져야 Stage3 fusion 확장이 의미 있다.\n",
    )


def load_cached_yolo_candidates(split: str) -> dict[str, list[dict[str, Any]]]:
    """Load the fair single-box YOLOv8n candidate pool from the previous run."""
    cand_path = PROJECT_ROOT / "experiments" / "ms_cxr_singlebox_fair_yolo_dino_fusion_v1" / "predictions" / f"yolov8n_{split}_conf0p001_candidates.csv"
    if not cand_path.exists():
        raise FileNotFoundError(cand_path)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in read_csv(cand_path):
        grouped.setdefault(str(row["dicom_id"]), []).append(
            {
                "class_id": int(as_float(row["class_id"])),
                "score": as_float(row["score"]),
                "box": [as_float(row["x1"]), as_float(row["y1"]), as_float(row["x2"]), as_float(row["y2"])],
                "source_model": row.get("source_model", "yolov8n"),
                "rank": int(as_float(row.get("rank"), 9999)),
            }
        )
    for key in list(grouped):
        grouped[key] = sorted(grouped[key], key=lambda r: float(r["score"]), reverse=True)
    return grouped


def load_context_head_dino_map(protocol: str, query_variant: str, split: str) -> dict[str, np.ndarray]:
    pred_path = PRED / f"{protocol}_{query_variant}_eval_predictions.jsonl"
    if split == "val":
        # Validation predictions are generated on demand from the best checkpoint.
        pred_path = PRED / f"{protocol}_{query_variant}_val_predictions.jsonl"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    out: dict[str, np.ndarray] = {}
    for row in read_jsonl(pred_path):
        key = str(row.get("sample_id") or row.get("task_id"))
        out[key] = np.asarray(row["pred_bbox_norm_cxcywh"], dtype="float32")
    return out


def save_val_predictions_for_fusion(args: argparse.Namespace, protocol: str, query_variant: str) -> None:
    """Fusion tuning needs val predictions; normal eval only writes eval predictions."""
    val_ds = MSContextDataset(protocol, query_variant, "val")
    ckpt_path = CKPT / f"{protocol}_{query_variant}_s{args.seed}_best.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = RadDinoContextCrossAttentionHead(
        token_dim=ckpt["token_dim"],
        query_dim=ckpt["query_dim"],
        hidden=args.hidden,
        num_query_tokens=args.num_query_tokens,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    eval_model(model, val_ds, args.batch_size, PRED / f"{protocol}_{query_variant}_val_predictions.jsonl")


def run_yolo_context_fusion(args: argparse.Namespace, protocol: str, query_variant: str) -> dict[str, Any]:
    if protocol != "singlebox888":
        raise ValueError("Stage3 fusion currently supports singlebox888 only.")
    from scripts import run_ms_cxr_singlebox_fair_yolo_dino_fusion_v1 as sf
    from scripts import run_ms_cxr_yolo_dino_rule_fusion_v1 as old_fusion
    from scripts import run_ms_cxr_yolo_rule_context_v1 as ybase
    from scripts import run_ms_cxr_yolo_rule_context_v2 as yv2

    save_val_predictions_for_fusion(args, protocol, query_variant)
    train_rows = sf.row_dicts("train")
    val_rows = sf.row_dicts("val")
    eval_rows = sf.row_dicts("eval")
    priors = ybase.make_train_priors(train_rows)
    val_candidates = load_cached_yolo_candidates("val")
    eval_candidates = load_cached_yolo_candidates("eval")
    dino_val = load_context_head_dino_map(protocol, query_variant, "val")
    dino_eval = load_context_head_dino_map(protocol, query_variant, "eval")

    params_by_finding, yolo_grid, yolo_per = yv2.tune(val_rows, val_candidates, priors, quick=True)
    fusion_by_finding, fusion_grid, fusion_per = old_fusion.tune_fusion(
        val_rows,
        val_candidates,
        priors,
        dino_val,
        params_by_finding,
        quick=True,
    )
    method = f"yolo_plus_rad_context_cross_attention_{query_variant}"
    fusion_eval = old_fusion.evaluate_fusion(
        eval_rows,
        eval_candidates,
        priors,
        dino_eval,
        params_by_finding,
        fusion_by_finding,
        method,
        "eval",
    )
    fusion_eval.to_csv(PRED / f"{protocol}_{query_variant}_yolo_context_fusion_eval_predictions.csv", index=False)
    yolo_grid.to_csv(MET / f"{protocol}_{query_variant}_stage3_yolo_rule_val_grid.csv", index=False)
    yolo_per.to_csv(MET / f"{protocol}_{query_variant}_stage3_yolo_rule_per_finding.csv", index=False)
    fusion_grid.to_csv(MET / f"{protocol}_{query_variant}_stage3_fusion_val_grid.csv", index=False)
    fusion_per.to_csv(MET / f"{protocol}_{query_variant}_stage3_fusion_per_finding.csv", index=False)
    write_text(EXP / "configs" / f"{protocol}_{query_variant}_stage3_fusion_params.json", json.dumps(fusion_by_finding, indent=2, ensure_ascii=False))

    rows = []
    for subset in ["all8", "main5"]:
        m = sf.metrics(fusion_eval, method, subset)
        m.update({"protocol": protocol, "query_variant": query_variant, "model_family": "YOLO + RAD-DINO context cross-attention fusion"})
        rows.append(m)
    out_path = MET / f"{protocol}_{query_variant}_stage3_fusion_eval_summary.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    write_text(
        REPORT / f"STAGE5_FUSION_{protocol}_{query_variant}_REPORT.md",
        f"# Stage5 YOLO + RAD-DINO Context Fusion: {protocol} / {query_variant}\n\n"
        + pd.DataFrame(rows).to_markdown(index=False)
        + "\n\n"
        "YOLO candidates are the cached fair single-box YOLOv8n candidate pool. Fusion weights are selected on val only.\n",
    )
    return rows[0]


def parse_query_variants(raw: str) -> list[str]:
    out = [x.strip() for x in raw.split(",") if x.strip()]
    valid = {"label_only", "rule_context", "full_phrase", "rule_plus_full"}
    bad = [x for x in out if x not in valid]
    if bad:
        raise ValueError(f"invalid query variants: {bad}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--stage0", action="store_true")
    parser.add_argument("--stage1-feature-check", action="store_true")
    parser.add_argument("--stage2-smoke", action="store_true")
    parser.add_argument("--build-data", action="store_true")
    parser.add_argument("--build-query-features", action="store_true")
    parser.add_argument("--train-context-head", action="store_true")
    parser.add_argument("--evaluate-context-head", action="store_true")
    parser.add_argument("--stage3-fusion", action="store_true")
    parser.add_argument("--protocol", choices=["singlebox888", "row1444"], default="singlebox888")
    parser.add_argument("--query-variants", default="rule_context,full_phrase,label_only")
    parser.add_argument("--query-dim", type=int, default=128)
    parser.add_argument("--max-query-features", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--num-query-tokens", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--center-loss-weight", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    if args.quick and not any([args.stage0, args.stage1_feature_check, args.stage2_smoke, args.build_data, args.build_query_features, args.train_context_head, args.evaluate_context_head, args.stage3_fusion]):
        args.stage0 = True
        args.stage1_feature_check = True
        args.stage2_smoke = True
        args.epochs = min(args.epochs, 3)
        args.batch_size = min(args.batch_size, 8)
    if args.stage2_smoke:
        args.build_data = True
        args.build_query_features = True
        args.train_context_head = True
        args.evaluate_context_head = True

    if args.stage0:
        audit = stage0_audit(args.quick)
        print("stage0_report", REPORT / "README_KO.md")
        print("train_rows", audit["split"]["train_rows"])
        print("val_rows", audit["split"]["val_rows"])
        print("eval_rows", audit["split"]["eval_rows"])
    if args.stage1_feature_check:
        fc = feature_check()
        print("feature_check_ok", fc["ok"])
        print("feature_schema_report", REPORT / "STAGE1_FEATURE_SCHEMA_REPORT.md")

    variants = parse_query_variants(args.query_variants)
    eval_results: list[dict[str, Any]] = []
    should_run_model_stages = any([args.build_data, args.build_query_features, args.train_context_head, args.evaluate_context_head, args.stage3_fusion])
    if should_run_model_stages:
        for query_variant in variants:
            if args.build_data or args.force or not (DATA / f"{args.protocol}_{query_variant}_train.jsonl").exists():
                stats = build_datasets(args.protocol, query_variant)
                print("dataset_built", args.protocol, query_variant, stats)
            if args.build_query_features or args.force or not (DATA / f"{args.protocol}_{query_variant}_query_features_train.npz").exists():
                qstats = build_query_features(args.protocol, query_variant, args.query_dim, args.max_query_features)
                print("query_features_built", qstats)
            if args.train_context_head:
                tstats = train_context_head(args, args.protocol, query_variant)
                print("train_complete", tstats)
            if args.evaluate_context_head:
                estats = evaluate_context_head(args, args.protocol, query_variant)
                eval_results.append(estats)
                print("eval_complete", estats)
            if args.stage3_fusion:
                fstats = run_yolo_context_fusion(args, args.protocol, query_variant)
                print("stage3_fusion_complete", fstats)

    update_combined_summary(eval_results)
    if eval_results:
        best = max(eval_results, key=lambda r: float(r["mean_iou"]))
        print("best_context_head_method", best["query_variant"])
        print("best_context_head_mean_iou", best["mean_iou"])
    print("project_root", PROJECT_ROOT)
    print("combined_summary", MET / "context_head_combined_summary.csv")
    print("current_result_report", REPORT / "CONTEXT_HEAD_CURRENT_RESULT_KO.md")


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = PROJECT_ROOT / "experiments" / "ms_cxr_multibox_phrase_grounding_v1"
REPORT_DIR = PROJECT_ROOT / "reports" / "ms_cxr_multibox_phrase_grounding_v1"
PRED_DIR = EXP_DIR / "predictions"
METRIC_DIR = EXP_DIR / "metrics"
META_DIR = EXP_DIR / "metadata"
CONTACT_DIR = EXP_DIR / "contact_sheets"

STAGE1_DATA = PROJECT_ROOT / "training" / "ms_cxr_vfm_localizer_stage1_p10_p19" / "datasets"

CLASS_MAP = {
    "Atelectasis": 0,
    "Cardiomegaly": 1,
    "Consolidation": 2,
    "Edema": 3,
    "Lung Opacity": 4,
    "Pleural Effusion": 5,
    "Pneumonia": 6,
    "Pneumothorax": 7,
}
MAIN5 = {"Atelectasis", "Consolidation", "Lung Opacity", "Pleural Effusion", "Pneumothorax"}


def ensure_dirs() -> None:
    for p in [EXP_DIR, REPORT_DIR, PRED_DIR, METRIC_DIR, META_DIR, CONTACT_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def norm_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def group_key(row: dict[str, Any]) -> str:
    phrase = row.get("phrase") or row.get("claim_sentence") or row.get("sentence") or ""
    return f"{row['dicom_id']}|{row['finding']}|{norm_text(phrase)}"


def box_area(b: list[float] | tuple[float, float, float, float]) -> float:
    return max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))


def iou_xyxy(a: list[float] | tuple[float, float, float, float], b: list[float] | tuple[float, float, float, float]) -> float:
    ix1 = max(float(a[0]), float(b[0]))
    iy1 = max(float(a[1]), float(b[1]))
    ix2 = min(float(a[2]), float(b[2]))
    iy2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    denom = box_area(a) + box_area(b) - inter
    return 0.0 if denom <= 0 else inter / denom


def union_box(boxes: list[list[float]]) -> list[float]:
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def nms(boxes: list[dict[str, Any]], iou_thr: float = 0.5) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for cand in sorted(boxes, key=lambda x: float(x.get("score", x.get("confidence", 0.0))), reverse=True):
        b = cand["box"]
        if all(iou_xyxy(b, k["box"]) < iou_thr for k in kept):
            kept.append(cand)
    return kept


def dedup_boxes(boxes: list[dict[str, Any]], iou_thr: float = 0.95) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cand in sorted(boxes, key=lambda x: float(x.get("score", x.get("confidence", 1.0))), reverse=True):
        if all(iou_xyxy(cand["box"], old["box"]) < iou_thr for old in out):
            out.append(cand)
    return out


def eval_set(gt_boxes: list[list[float]], pred_boxes: list[list[float]]) -> dict[str, float]:
    n_gt = len(gt_boxes)
    n_pred = len(pred_boxes)
    if n_gt == 0:
        return {
            "coverage_mean_iou": 0.0,
            "union_iou": 0.0,
            "gt_hit_rate_0_3": 0.0,
            "gt_hit_rate_0_5": 0.0,
            "all_gt_hit_0_3": 0.0,
            "all_gt_hit_0_5": 0.0,
            "any_gt_hit_0_3": 0.0,
            "any_gt_hit_0_5": 0.0,
            "set_precision_0_3": 0.0,
            "set_recall_0_3": 0.0,
            "set_f1_0_3": 0.0,
            "set_precision_0_5": 0.0,
            "set_recall_0_5": 0.0,
            "set_f1_0_5": 0.0,
            "n_gt": 0,
            "n_pred": n_pred,
            "pred_count_abs_error": abs(n_pred - n_gt),
        }
    if n_pred == 0:
        return {
            "coverage_mean_iou": 0.0,
            "union_iou": 0.0,
            "gt_hit_rate_0_3": 0.0,
            "gt_hit_rate_0_5": 0.0,
            "all_gt_hit_0_3": 0.0,
            "all_gt_hit_0_5": 0.0,
            "any_gt_hit_0_3": 0.0,
            "any_gt_hit_0_5": 0.0,
            "set_precision_0_3": 0.0,
            "set_recall_0_3": 0.0,
            "set_f1_0_3": 0.0,
            "set_precision_0_5": 0.0,
            "set_recall_0_5": 0.0,
            "set_f1_0_5": 0.0,
            "n_gt": n_gt,
            "n_pred": 0,
            "pred_count_abs_error": n_gt,
        }

    gt_best = [max(iou_xyxy(g, p) for p in pred_boxes) for g in gt_boxes]

    def greedy_tp(thr: float) -> int:
        pairs: list[tuple[float, int, int]] = []
        for i, g in enumerate(gt_boxes):
            for j, p in enumerate(pred_boxes):
                val = iou_xyxy(g, p)
                if val >= thr:
                    pairs.append((val, i, j))
        pairs.sort(reverse=True)
        used_g: set[int] = set()
        used_p: set[int] = set()
        tp = 0
        for _, i, j in pairs:
            if i not in used_g and j not in used_p:
                used_g.add(i)
                used_p.add(j)
                tp += 1
        return tp

    out: dict[str, float] = {
        "coverage_mean_iou": sum(gt_best) / n_gt,
        "union_iou": iou_xyxy(union_box(gt_boxes), union_box(pred_boxes)),
        "gt_hit_rate_0_3": sum(x >= 0.3 for x in gt_best) / n_gt,
        "gt_hit_rate_0_5": sum(x >= 0.5 for x in gt_best) / n_gt,
        "all_gt_hit_0_3": float(all(x >= 0.3 for x in gt_best)),
        "all_gt_hit_0_5": float(all(x >= 0.5 for x in gt_best)),
        "any_gt_hit_0_3": float(any(x >= 0.3 for x in gt_best)),
        "any_gt_hit_0_5": float(any(x >= 0.5 for x in gt_best)),
        "n_gt": n_gt,
        "n_pred": n_pred,
        "pred_count_abs_error": abs(n_pred - n_gt),
    }
    for thr_name, thr in [("0_3", 0.3), ("0_5", 0.5)]:
        tp = greedy_tp(thr)
        precision = tp / n_pred if n_pred else 0.0
        recall = tp / n_gt if n_gt else 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        out[f"set_precision_{thr_name}"] = precision
        out[f"set_recall_{thr_name}"] = recall
        out[f"set_f1_{thr_name}"] = f1
    return out


def make_groups(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = group_key(row)
        if key not in groups:
            groups[key] = {
                "group_id": key,
                "split": row["split"],
                "dicom_id": row["dicom_id"],
                "subject_id": row.get("subject_id", ""),
                "study_id": row.get("study_id", ""),
                "image_path": row["image_path"],
                "finding": row["finding"],
                "class_id": CLASS_MAP[row["finding"]],
                "claim_sentence": row.get("claim_sentence") or row.get("phrase") or "",
                "image_width": row["image_width"],
                "image_height": row["image_height"],
                "task_ids": [],
                "annotation_ids": [],
                "gt_boxes": [],
            }
        groups[key]["task_ids"].append(row["task_id"])
        groups[key]["annotation_ids"].append(row.get("ms_cxr_annotation_id", ""))
        groups[key]["gt_boxes"].append([float(x) for x in row["gold_bbox_xyxy"]])
    return groups


def load_row_prediction_csv(path: Path, method_name: str, groups: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    task_to_group = {task_id: gid for gid, g in groups.items() for task_id in g["task_ids"]}
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        task_id = str(r.get("task_id", r.get("sample_id", "")))
        gid = task_to_group.get(task_id)
        if not gid:
            continue
        if bool(r.get("bbox_missing", False)):
            continue
        box = [float(r["pred_x1"]), float(r["pred_y1"]), float(r["pred_x2"]), float(r["pred_y2"])]
        out[gid].append({"box": box, "score": float(r.get("confidence", 1.0)), "source": method_name})
    return {gid: dedup_boxes(v) for gid, v in out.items()}


def load_row_prediction_jsonl(path: Path, method_name: str, groups: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    task_to_group = {task_id: gid for gid, g in groups.items() for task_id in g["task_ids"]}
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in load_jsonl(path):
        task_id = str(r.get("task_id", ""))
        gid = task_to_group.get(task_id)
        if not gid or bool(r.get("bbox_missing", False)):
            continue
        box = r.get("pred_bbox_xyxy")
        if not box:
            continue
        out[gid].append({"box": [float(x) for x in box], "score": float(r.get("confidence", 1.0)), "source": method_name})
    return {gid: dedup_boxes(v) for gid, v in out.items()}


def load_yolo_candidates(path: Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    out: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        return out
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        key = (str(r["dicom_id"]), int(r["class_id"]))
        out[key].append({
            "box": [float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])],
            "score": float(r["score"]),
            "source": path.stem,
        })
    return out


def detection_preds_for_groups(
    groups: dict[str, dict[str, Any]],
    candidates: dict[tuple[str, int], list[dict[str, Any]]],
    params_by_finding: dict[str, dict[str, float]],
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for gid, g in groups.items():
        params = params_by_finding.get(g["finding"], params_by_finding.get("__global__", {"score_thr": 0.01, "max_k": 2}))
        score_thr = float(params["score_thr"])
        max_k = int(params["max_k"])
        raw = [
            c for c in candidates.get((g["dicom_id"], int(g["class_id"])), [])
            if float(c["score"]) >= score_thr
        ]
        kept = nms(raw, iou_thr=0.5)[:max_k]
        out[gid] = kept
    return out


def summarize(group_scores: list[dict[str, Any]], method: str, subset_name: str) -> dict[str, Any]:
    cols = [
        "coverage_mean_iou",
        "union_iou",
        "gt_hit_rate_0_3",
        "gt_hit_rate_0_5",
        "all_gt_hit_0_3",
        "all_gt_hit_0_5",
        "any_gt_hit_0_3",
        "any_gt_hit_0_5",
        "set_precision_0_3",
        "set_recall_0_3",
        "set_f1_0_3",
        "set_precision_0_5",
        "set_recall_0_5",
        "set_f1_0_5",
        "n_pred",
        "pred_count_abs_error",
    ]
    out: dict[str, Any] = {"method": method, "subset": subset_name, "n_groups": len(group_scores)}
    for c in cols:
        vals = [float(x[c]) for x in group_scores]
        out[c] = sum(vals) / len(vals) if vals else 0.0
    out["n_gt_boxes"] = int(sum(int(x["n_gt"]) for x in group_scores))
    return out


def eval_method(
    method: str,
    groups: dict[str, dict[str, Any]],
    pred_by_gid: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for gid, g in groups.items():
        preds = pred_by_gid.get(gid, [])
        score = eval_set(g["gt_boxes"], [p["box"] for p in preds])
        rows.append({
            "method": method,
            "group_id": gid,
            "dicom_id": g["dicom_id"],
            "subject_id": g["subject_id"],
            "study_id": g["study_id"],
            "finding": g["finding"],
            "claim_sentence": g["claim_sentence"],
            "n_gt": len(g["gt_boxes"]),
            "n_pred": len(preds),
            "gt_boxes_json": json.dumps(g["gt_boxes"]),
            "pred_boxes_json": json.dumps([p["box"] for p in preds]),
            **score,
        })
    return rows


def tune_detection_params(
    val_groups: dict[str, dict[str, Any]],
    candidates: dict[tuple[str, int], list[dict[str, Any]]],
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    score_grid = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
    k_grid = [1, 2, 3, 4, 5]
    rows: list[dict[str, Any]] = []

    def run_subset(name: str, subset: dict[str, dict[str, Any]]) -> dict[str, float]:
        best: dict[str, float] | None = None
        for thr in score_grid:
            for max_k in k_grid:
                preds = detection_preds_for_groups(subset, candidates, {"__global__": {"score_thr": thr, "max_k": max_k}})
                scores = []
                for gid, g in subset.items():
                    p = [x["box"] for x in preds.get(gid, [])]
                    s = eval_set(g["gt_boxes"], p)
                    scores.append(s)
                summ = summarize(scores, "candidate_detection_set", name)
                row = {"finding_for_tuning": name, "score_thr": thr, "max_k": max_k, **summ}
                rows.append(row)
                key = (
                    summ["set_f1_0_3"],
                    summ["coverage_mean_iou"],
                    -abs(summ["n_pred"] - 2.0),
                )
                if best is None or key > best["_key"]:
                    best = {"score_thr": thr, "max_k": max_k, "_key": key}
        assert best is not None
        return best

    params = {"__global__": run_subset("__global__", val_groups)}
    for finding in sorted({g["finding"] for g in val_groups.values()}):
        subset = {gid: g for gid, g in val_groups.items() if g["finding"] == finding}
        if len(subset) >= 3:
            params[finding] = run_subset(finding, subset)
    for p in params.values():
        p.pop("_key", None)
    return params, rows


def candidate_oracle(
    groups: dict[str, dict[str, Any]],
    candidates: dict[tuple[str, int], list[dict[str, Any]]],
    max_k: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for gid, g in groups.items():
        cands = nms(candidates.get((g["dicom_id"], int(g["class_id"])), []), iou_thr=0.5)
        selected: list[dict[str, Any]] = []
        used: set[int] = set()
        for gt in g["gt_boxes"]:
            best_i = None
            best_v = -1.0
            for idx, cand in enumerate(cands):
                if idx in used:
                    continue
                val = iou_xyxy(gt, cand["box"])
                if val > best_v:
                    best_v = val
                    best_i = idx
            if best_i is not None:
                used.add(best_i)
                selected.append(cands[best_i])
        out[gid] = selected[:max_k]
    return out


def bootstrap_diff(rows: pd.DataFrame, method_a: str, method_b: str, metric: str, reps: int = 2000) -> dict[str, Any]:
    piv = rows.pivot(index="group_id", columns="method", values=metric).dropna(subset=[method_a, method_b])
    ids = list(piv.index)
    observed = float((piv[method_a] - piv[method_b]).mean()) if ids else 0.0
    rng = random.Random(2026)
    diffs = []
    for _ in range(reps):
        sample = [rng.choice(ids) for _ in ids]
        diffs.append(float((piv.loc[sample, method_a].to_numpy() - piv.loc[sample, method_b].to_numpy()).mean()))
    diffs.sort()
    if not diffs:
        return {"comparison": f"{method_a} - {method_b}", "metric": metric, "n": 0}
    return {
        "comparison": f"{method_a} - {method_b}",
        "metric": metric,
        "n": len(ids),
        "observed_diff": observed,
        "ci95_low": diffs[int(0.025 * reps)],
        "ci95_high": diffs[min(reps - 1, int(0.975 * reps))],
        "prob_gt_0": sum(d > 0 for d in diffs) / len(diffs),
        "bootstrap_unit": "multi_box_group",
        "bootstrap_reps": reps,
    }


def draw_overlay(groups: dict[str, dict[str, Any]], pred_rows: pd.DataFrame) -> None:
    methods = [
        "rad_dino_rule_context",
        "yolo_dino_rule_fusion_v1",
        "yolov8s_detection_set",
    ]
    available = [m for m in methods if m in set(pred_rows["method"])]
    if not available:
        return
    # Prefer cases where fusion/detection improves over RAD-DINO.
    piv = pred_rows.pivot(index="group_id", columns="method", values="coverage_mean_iou")
    if "rad_dino_rule_context" in piv and "yolo_dino_rule_fusion_v1" in piv:
        order = (piv["yolo_dino_rule_fusion_v1"] - piv["rad_dino_rule_context"]).sort_values(ascending=False).index.tolist()
    else:
        order = list(groups.keys())
    selected = [gid for gid in order if gid in groups][:12]
    panels: list[Image.Image] = []
    font = ImageFont.load_default()
    color_map = {
        "gold": (255, 220, 0),
        "rad_dino_rule_context": (70, 130, 255),
        "yolo_dino_rule_fusion_v1": (0, 220, 80),
        "yolov8s_detection_set": (0, 220, 220),
    }
    for gid in selected:
        g = groups[gid]
        try:
            img = Image.open(g["image_path"]).convert("RGB")
        except Exception:
            continue
        w, h = img.size
        scale = 520 / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size)
        draw = ImageDraw.Draw(img)
        sx, sy = new_size[0] / w, new_size[1] / h

        def rect(box: list[float], color: tuple[int, int, int], width: int = 3) -> None:
            b = [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]
            for k in range(width):
                draw.rectangle([b[0] - k, b[1] - k, b[2] + k, b[3] + k], outline=color)

        for box in g["gt_boxes"]:
            rect(box, color_map["gold"], 3)
        title = f"{g['finding']} | n_gt={len(g['gt_boxes'])}"
        draw.rectangle([0, 0, new_size[0], 38], fill=(0, 0, 0))
        draw.text((4, 4), title[:82], fill=(255, 255, 255), font=font)
        y = 18
        for method in available:
            row = pred_rows[(pred_rows["group_id"] == gid) & (pred_rows["method"] == method)]
            if row.empty:
                continue
            boxes = json.loads(row.iloc[0]["pred_boxes_json"])
            for box in boxes:
                rect(box, color_map[method], 2)
            draw.text((4, y), f"{method}: cov={row.iloc[0]['coverage_mean_iou']:.3f}", fill=color_map[method], font=font)
            y += 10
        panels.append(img)

    if not panels:
        return
    cell_w = max(p.width for p in panels)
    cell_h = max(p.height for p in panels)
    cols = 3
    rows = math.ceil(len(panels) / cols)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (30, 30, 30))
    for idx, p in enumerate(panels):
        x = (idx % cols) * cell_w
        y = (idx // cols) * cell_h
        sheet.paste(p, (x, y))
    sheet.save(CONTACT_DIR / "ms_cxr_multibox_examples.jpg", quality=92)


def write_report(
    audit_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    per_finding_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    val_grid_df: pd.DataFrame,
) -> None:
    actual_summary = summary_df[~summary_df["method"].str.contains("oracle", case=False, na=False)]
    oracle_summary = summary_df[summary_df["method"].str.contains("oracle", case=False, na=False)]
    best = actual_summary.sort_values("coverage_mean_iou", ascending=False).iloc[0]
    best_oracle = oracle_summary.sort_values("coverage_mean_iou", ascending=False).iloc[0] if len(oracle_summary) else None
    n_eval_groups = int((audit_df["split"] == "eval").sum())
    n_eval_multi = int(((audit_df["split"] == "eval") & (audit_df["n_boxes"] > 1)).sum())
    n_eval_rows_multi = int(audit_df[(audit_df["split"] == "eval") & (audit_df["n_boxes"] > 1)]["n_boxes"].sum())

    lines = []
    lines.append("# MS-CXR Multi-box Phrase Grounding V1\n")
    lines.append("## 한 줄 결론\n")
    lines.append(
        f"MS-CXR eval에서 phrase 하나에 bbox가 2개 이상 붙은 multi-box group은 {n_eval_multi}개이고, "
        f"관련 bbox row는 {n_eval_rows_multi}개다. 이 subset에서는 `{best['method']}`가 "
        f"실제 방법 중 coverage mean IoU {best['coverage_mean_iou']:.4f}로 가장 높았다. "
        + (
            f"참고로 후보 pool oracle 상한은 `{best_oracle['method']}` {best_oracle['coverage_mean_iou']:.4f}다.\n"
            if best_oracle is not None else "\n"
        )
    )
    lines.append("## 왜 이 실험을 했나\n")
    lines.append(
        "기존 MedRPG 공정 비교는 single-box subset, 즉 phrase 하나에 bbox 하나만 있는 163개 eval group에서 했다. "
        "하지만 MS-CXR에는 bilateral, multifocal, diffuse 표현처럼 phrase 하나에 여러 bbox가 연결되는 경우가 있다. "
        "이 경우 단일 bbox phrase grounding보다 detector/fusion 구조가 더 자연스러울 수 있어서 별도 평가했다.\n"
    )
    lines.append("## 평가 정의\n")
    lines.append("- MS-CXR bbox는 phrase-grounding bbox이며 pixel-level lesion mask가 아니다.\n")
    lines.append("- group key는 `dicom_id + finding + normalized claim_sentence`로 만들었다.\n")
    lines.append("- multi-box group은 같은 group key 안에 MS-CXR bbox가 2개 이상 있는 경우다.\n")
    lines.append("- `coverage_mean_iou`는 각 정답 bbox가 예측 bbox set 중 가장 잘 맞는 박스와 얻은 IoU의 평균이다.\n")
    lines.append("- `all_gt_hit_0_3`은 한 group 안의 모든 정답 bbox가 IoU 0.3 이상으로 덮인 비율이다.\n")
    lines.append("- YOLO detection set의 threshold와 top-k는 val multi-box group에서만 선택하고 eval에 고정했다.\n")
    lines.append("- MedRPG fair retrain은 single-box train/eval용 산출물이므로 이 multi-box eval에는 직접 결과가 없다.\n")
    lines.append("\n## Split 감사\n")
    lines.append(audit_df.groupby(["split", "is_multi_box"]).size().reset_index(name="n_groups").to_markdown(index=False))
    lines.append("\n\n## 전체 결과\n")
    show_cols = [
        "method", "n_groups", "n_gt_boxes", "coverage_mean_iou", "union_iou",
        "gt_hit_rate_0_3", "all_gt_hit_0_3", "set_f1_0_3", "n_pred", "pred_count_abs_error"
    ]
    lines.append(summary_df[show_cols].to_markdown(index=False))
    lines.append("\n\n## Finding별 결과\n")
    pf_cols = ["finding", "method", "n_groups", "coverage_mean_iou", "gt_hit_rate_0_3", "all_gt_hit_0_3", "set_f1_0_3"]
    lines.append(per_finding_df[pf_cols].sort_values(["finding", "coverage_mean_iou"], ascending=[True, False]).to_markdown(index=False))
    lines.append("\n\n## Bootstrap\n")
    if len(bootstrap_df):
        lines.append(bootstrap_df.to_markdown(index=False))
    else:
        lines.append("비교 가능한 bootstrap 결과가 없다.")
    lines.append("\n\n## Val tuning\n")
    lines.append(
        "YOLO detection set은 eval을 보지 않고 val multi-box group에서 score threshold와 top-k를 골랐다. "
        f"전체 grid row 수는 {len(val_grid_df)}개다.\n"
    )
    lines.append("## 해석\n")
    lines.append(
        "- 이 실험은 MedRPG가 약하다고 바로 주장하는 실험이 아니다. MedRPG fair model은 multi-box eval prediction이 없으므로 직접 비교하지 않았다.\n"
    )
    lines.append(
        "- 다만 single-box만 보던 비교가 우리 방법의 장점을 충분히 보여주지 못할 수 있다는 점은 확인했다. "
        "multi-box에서는 정답 여러 개를 덮는 coverage 지표가 필요하다.\n"
    )
    lines.append(
        "- YOLO detection set은 여러 박스를 낼 수 있지만 false positive도 늘 수 있다. 그래서 coverage와 set F1을 같이 봐야 한다.\n"
    )
    lines.append(
        "- 논문에는 `multi-box phrase grounding subset에서 detector/fusion 구조를 별도 분석했다`고 쓸 수 있다. "
        "`MedRPG보다 이겼다`고 쓰려면 MedRPG도 같은 multi-box setting으로 재학습/평가해야 한다.\n"
    )
    lines.append("\n## 주요 산출물\n")
    lines.append(f"- metrics: `{METRIC_DIR / 'multibox_eval_summary.csv'}`\n")
    lines.append(f"- predictions: `{PRED_DIR / 'multibox_group_predictions.csv'}`\n")
    lines.append(f"- contact sheet: `{CONTACT_DIR / 'ms_cxr_multibox_examples.jpg'}`\n")
    (REPORT_DIR / "README_KO.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    train_rows = load_jsonl(STAGE1_DATA / "train.jsonl")
    val_rows = load_jsonl(STAGE1_DATA / "val.jsonl")
    eval_rows = load_jsonl(STAGE1_DATA / "eval.jsonl")
    all_rows = train_rows + val_rows + eval_rows
    all_groups = make_groups(all_rows)
    val_groups_all = make_groups(val_rows)
    eval_groups_all = make_groups(eval_rows)
    val_groups = {gid: g for gid, g in val_groups_all.items() if len(g["gt_boxes"]) > 1}
    eval_groups = {gid: g for gid, g in eval_groups_all.items() if len(g["gt_boxes"]) > 1}

    audit_rows = []
    for g in all_groups.values():
        audit_rows.append({
            "group_id": g["group_id"],
            "split": g["split"],
            "dicom_id": g["dicom_id"],
            "subject_id": g["subject_id"],
            "study_id": g["study_id"],
            "finding": g["finding"],
            "claim_sentence": g["claim_sentence"],
            "n_boxes": len(g["gt_boxes"]),
            "is_multi_box": len(g["gt_boxes"]) > 1,
            "task_ids": ";".join(g["task_ids"]),
            "annotation_ids": ";".join(g["annotation_ids"]),
        })
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(META_DIR / "ms_cxr_multibox_group_audit.csv", index=False)

    pred_sets: dict[str, dict[str, list[dict[str, Any]]]] = {}
    pred_sets["rad_dino_rule_context"] = load_row_prediction_jsonl(
        PROJECT_ROOT / "experiments" / "ms_cxr_context_vfm_localizer_stage2_context_finetune" / "predictions" / "frozen_heatmap_rule_context.jsonl",
        "rad_dino_rule_context",
        eval_groups,
    )
    pred_sets["yolo_rule_context_v2"] = load_row_prediction_csv(
        PROJECT_ROOT / "experiments" / "ms_cxr_yolo_rule_context_v2" / "predictions" / "yolo_rule_context_v2_eval_predictions.csv",
        "yolo_rule_context_v2",
        eval_groups,
    )
    pred_sets["yolo_dino_rule_fusion_v1"] = load_row_prediction_csv(
        PROJECT_ROOT / "experiments" / "ms_cxr_yolo_dino_rule_fusion_v1" / "predictions" / "yolo_dino_rule_fusion_eval_predictions.csv",
        "yolo_dino_rule_fusion_v1",
        eval_groups,
    )

    val_candidate_paths = {
        "yolov8n_detection_set": PROJECT_ROOT / "experiments" / "ms_cxr_yolo_rule_context_v1" / "predictions" / "yolov8n_val_conf0p001_all_candidates.csv",
        "yolov8s_detection_set": PROJECT_ROOT / "experiments" / "ms_cxr_yolo_rule_context_v1" / "predictions" / "yolov8s_val_conf0p001_all_candidates.csv",
    }
    eval_candidate_paths = {
        "yolov8n_detection_set": PROJECT_ROOT / "experiments" / "ms_cxr_yolo_rule_context_v1" / "predictions" / "yolov8n_eval_conf0p001_all_candidates.csv",
        "yolov8s_detection_set": PROJECT_ROOT / "experiments" / "ms_cxr_yolo_rule_context_v1" / "predictions" / "yolov8s_eval_conf0p001_all_candidates.csv",
    }
    val_grid_rows = []
    params_out = {}
    for method, val_path in val_candidate_paths.items():
        val_cands = load_yolo_candidates(val_path)
        params, grid_rows = tune_detection_params(val_groups, val_cands)
        params_out[method] = params
        for r in grid_rows:
            r["method"] = method
        val_grid_rows.extend(grid_rows)
        eval_cands = load_yolo_candidates(eval_candidate_paths[method])
        pred_sets[method] = detection_preds_for_groups(eval_groups, eval_cands, params)
        pred_sets[f"{method}_candidate_pool_oracle"] = candidate_oracle(eval_groups, eval_cands)

    (EXP_DIR / "configs").mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "configs" / "detection_set_val_tuned_params.json").write_text(
        json.dumps(params_out, indent=2), encoding="utf-8"
    )
    val_grid_df = pd.DataFrame(val_grid_rows)
    val_grid_df.to_csv(METRIC_DIR / "multibox_detection_val_grid.csv", index=False)

    pred_rows: list[dict[str, Any]] = []
    for method, preds in pred_sets.items():
        pred_rows.extend(eval_method(method, eval_groups, preds))
    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(PRED_DIR / "multibox_group_predictions.csv", index=False)

    summary_rows = []
    for method in sorted(pred_df["method"].unique()):
        rows = pred_df[pred_df["method"] == method].to_dict("records")
        summary_rows.append(summarize(rows, method, "eval_multibox_all8"))
        main_rows = [r for r in rows if r["finding"] in MAIN5]
        if main_rows:
            summary_rows.append(summarize(main_rows, method, "eval_multibox_main5"))
    summary_df = pd.DataFrame(summary_rows).sort_values(["subset", "coverage_mean_iou"], ascending=[True, False])
    summary_df.to_csv(METRIC_DIR / "multibox_eval_summary.csv", index=False)

    per_finding_rows = []
    for (finding, method), part in pred_df.groupby(["finding", "method"]):
        per_finding_rows.append(summarize(part.to_dict("records"), method, f"eval_multibox_{finding}") | {"finding": finding})
    per_finding_df = pd.DataFrame(per_finding_rows)
    per_finding_df.to_csv(METRIC_DIR / "multibox_per_finding.csv", index=False)

    bootstrap_rows = []
    comparisons = [
        ("yolo_dino_rule_fusion_v1", "rad_dino_rule_context"),
        ("yolo_dino_rule_fusion_v1", "yolo_rule_context_v2"),
        ("yolov8s_detection_set", "rad_dino_rule_context"),
        ("yolov8s_detection_set", "yolo_dino_rule_fusion_v1"),
        ("yolov8s_detection_set_candidate_pool_oracle", "yolov8s_detection_set"),
    ]
    for a, b in comparisons:
        if a in set(pred_df["method"]) and b in set(pred_df["method"]):
            for metric in ["coverage_mean_iou", "all_gt_hit_0_3", "set_f1_0_3"]:
                bootstrap_rows.append(bootstrap_diff(pred_df, a, b, metric))
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    bootstrap_df.to_csv(METRIC_DIR / "multibox_bootstrap_ci.csv", index=False)

    draw_overlay(eval_groups, pred_df)
    write_report(audit_df, summary_df[summary_df["subset"] == "eval_multibox_all8"], per_finding_df, bootstrap_df, val_grid_df)

    final = {
        "project_root": str(PROJECT_ROOT),
        "eval_groups_total": len(eval_groups_all),
        "eval_multibox_groups": len(eval_groups),
        "eval_multibox_gt_boxes": sum(len(g["gt_boxes"]) for g in eval_groups.values()),
        "best_actual_method_by_coverage": str(
            summary_df[
                (summary_df["subset"] == "eval_multibox_all8")
                & (~summary_df["method"].str.contains("oracle", case=False, na=False))
            ].sort_values("coverage_mean_iou", ascending=False).iloc[0]["method"]
        ),
        "best_actual_coverage_mean_iou": float(
            summary_df[
                (summary_df["subset"] == "eval_multibox_all8")
                & (~summary_df["method"].str.contains("oracle", case=False, na=False))
            ].sort_values("coverage_mean_iou", ascending=False).iloc[0]["coverage_mean_iou"]
        ),
        "best_oracle_coverage_mean_iou": float(
            summary_df[
                (summary_df["subset"] == "eval_multibox_all8")
                & (summary_df["method"].str.contains("oracle", case=False, na=False))
            ].sort_values("coverage_mean_iou", ascending=False).iloc[0]["coverage_mean_iou"]
        ),
        "summary_path": str(METRIC_DIR / "multibox_eval_summary.csv"),
        "report_path": str(REPORT_DIR / "README_KO.md"),
    }
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()

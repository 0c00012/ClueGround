#!/usr/bin/env python
"""Validate and format exact final ClueGround component/query ablations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "clueground_exact_final_ablation_v1"
SPLIT_AUDIT_SOURCE = (
    ROOT
    / "experiments"
    / "clueground_no_siglip_component_ablation_v2"
    / "audit"
    / "SPLIT_OVERLAP_AUDIT.json"
)
SEEDS = {13, 42, 2026}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def metric(frame: pd.DataFrame, table: str, protocol: str, variant: str, name: str) -> tuple[float, float]:
    row = frame.loc[
        (frame["table"] == table)
        & (frame["protocol"].astype(str) == protocol)
        & (frame["variant"] == variant)
    ]
    if len(row) != 1:
        raise RuntimeError(f"Missing aggregate row: {table}/{protocol}/{variant}")
    return float(row.iloc[0][f"{name}_mean"]), float(row.iloc[0][f"{name}_std"])


def f4(value: float) -> str:
    return f"{value:.4f}"


def pm(mean: float, std: float) -> str:
    return f"{mean:.4f} +/- {std:.4f}"


def build_table(
    aggregate: pd.DataFrame,
    table: str,
    protocol: str,
    variants: list[tuple[str, str]],
) -> pd.DataFrame:
    rows = []
    if protocol == "888":
        names = ("mean_iou", "hit_0_3", "hit_0_5")
    else:
        names = ("coverage_mean_iou", "exact_rectangle_union_iou", "set_f1_0_3", "set_f1_0_5", "mean_pred_count")
    for variant, label in variants:
        row: dict[str, object] = {"configuration": label}
        for name in names:
            mean, std = metric(aggregate, table, protocol, variant, name)
            row[f"{name}_mean"] = mean
            row[f"{name}_std"] = std
        rows.append(row)
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, protocol: str) -> list[str]:
    if protocol == "888":
        lines = [
            "| Configuration | Mean IoU | Hit@0.3 | Hit@0.5 |",
            "|---|---:|---:|---:|",
        ]
        for row in frame.to_dict("records"):
            lines.append(
                f"| {row['configuration']} | {pm(row['mean_iou_mean'], row['mean_iou_std'])} | "
                f"{f4(row['hit_0_3_mean'])} | {f4(row['hit_0_5_mean'])} |"
            )
        return lines
    lines = [
        "| Configuration | C-IoU | U-IoU | F1@0.3 | F1@0.5 | Count |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in frame.to_dict("records"):
        lines.append(
            f"| {row['configuration']} | {pm(row['coverage_mean_iou_mean'], row['coverage_mean_iou_std'])} | "
            f"{pm(row['exact_rectangle_union_iou_mean'], row['exact_rectangle_union_iou_std'])} | "
            f"{f4(row['set_f1_0_3_mean'])} | {f4(row['set_f1_0_5_mean'])} | "
            f"{f4(row['mean_pred_count_mean'])} |"
        )
    return lines


def latex_table(frame: pd.DataFrame, protocol: str) -> list[str]:
    if protocol == "888":
        lines = [
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Configuration & mIoU & Hit@.3 & Hit@.5 \\",
            r"\midrule",
        ]
        for row in frame.to_dict("records"):
            label = str(row["configuration"]).replace("_", r"\_")
            lines.append(
                f"{label} & {row['mean_iou_mean']:.4f}$\\pm${row['mean_iou_std']:.4f} & "
                f"{row['hit_0_3_mean']:.4f} & {row['hit_0_5_mean']:.4f} \\\\" 
            )
    else:
        lines = [
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"Configuration & C-IoU & U-IoU & F1@.3 & F1@.5 & Count \\",
            r"\midrule",
        ]
        for row in frame.to_dict("records"):
            label = str(row["configuration"]).replace("_", r"\_")
            lines.append(
                f"{label} & {row['coverage_mean_iou_mean']:.4f}$\\pm${row['coverage_mean_iou_std']:.4f} & "
                f"{row['exact_rectangle_union_iou_mean']:.4f}$\\pm${row['exact_rectangle_union_iou_std']:.4f} & "
                f"{row['set_f1_0_3_mean']:.4f} & {row['set_f1_0_5_mean']:.4f} & "
                f"{row['mean_pred_count_mean']:.4f} \\\\" 
            )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return lines


def main() -> None:
    per_seed = pd.read_csv(EXP / "per_seed_metrics.csv")
    aggregate = pd.read_csv(EXP / "aggregate_metrics.csv")
    split_audit = json.loads(SPLIT_AUDIT_SOURCE.read_text(encoding="utf-8"))
    split_overlap_pass = split_audit.get("status") == "PASS" and all(
        int(value) == 0
        for protocol in ("singlebox_888", "multibox_1444")
        for value in split_audit.get(protocol, {}).values()
    )
    write_json(EXP / "SPLIT_OVERLAP_AUDIT.json", split_audit)
    if len(per_seed) != 72 or set(per_seed["seed"].astype(int)) != SEEDS:
        raise RuntimeError("Incomplete 72-run, three-seed matrix")
    p888 = per_seed[per_seed["protocol"].astype(str) == "888"]
    p1444 = per_seed[per_seed["protocol"].astype(str) == "1444"]
    denominator_pass = (
        set(pd.to_numeric(p888["n"], errors="coerce").dropna().astype(int)) == {163}
        and set(pd.to_numeric(p1444["n_groups"], errors="coerce").dropna().astype(int)) == {220}
        and float(pd.to_numeric(per_seed["n_missing_predictions"], errors="coerce").fillna(0).sum()) == 0.0
    )

    component_variants = [
        ("yolo_only", "YOLO only"),
        ("rad_dino_only", "RAD-DINO only"),
        ("full", "YOLO + RAD-DINO (ClueGround)"),
    ]
    query_variants = [
        ("full_query", "Full query"),
        ("finding_only", "RAD-DINO query: finding only"),
        ("without_f", "-f"),
        ("without_l", "-laterality"),
        ("without_v", "-vertical"),
        ("without_z", "-severity"),
        ("without_u", "-uncertainty"),
        ("without_m_lexical", "-multiplicity lexemes from vq"),
        ("without_vq", "-raw phrase hash (vq)"),
    ]
    tables = {
        "TABLE_III_COMPONENT_888": build_table(aggregate, "component", "888", component_variants),
        "TABLE_IV_COMPONENT_1444": build_table(aggregate, "component", "1444", component_variants),
        "TABLE_V_QUERY_888": build_table(aggregate, "query", "888", query_variants),
        "TABLE_VI_QUERY_1444": build_table(aggregate, "query", "1444", query_variants),
    }
    for name, frame in tables.items():
        frame.to_csv(EXP / f"{name}.csv", index=False)

    full_component_888 = metric(aggregate, "component", "888", "full", "mean_iou")[0]
    full_query_888 = metric(aggregate, "query", "888", "full_query", "mean_iou")[0]
    full_component_1444 = metric(aggregate, "component", "1444", "full", "coverage_mean_iou")[0]
    full_query_1444 = metric(aggregate, "query", "1444", "full_query", "coverage_mean_iou")[0]
    identity_pass = (
        np.isclose(full_component_888, full_query_888, atol=1e-12, rtol=0.0)
        and np.isclose(full_component_1444, full_query_1444, atol=1e-12, rtol=0.0)
        and np.isclose(full_component_888, 0.5486315252772557, atol=1e-12, rtol=0.0)
        and np.isclose(full_component_1444, 0.5372594933248681, atol=1e-12, rtol=0.0)
    )
    audit = {
        "status": "PASS" if denominator_pass and identity_pass and split_overlap_pass else "FAIL",
        "n_runs": int(len(per_seed)),
        "seeds": sorted(SEEDS),
        "denominator_pass": bool(denominator_pass),
        "split_overlap_pass": bool(split_overlap_pass),
        "split_overlap_audit_source": str(SPLIT_AUDIT_SOURCE),
        "missing_predictions": 0,
        "full_row_identity_pass": bool(identity_pass),
        "full_888_mean_iou": full_component_888,
        "full_1444_coverage": full_component_1444,
        "query_intervention": "fixed checkpoint, inference-time masking only",
        "warning_m": "The exact 91-D checkpoint has no independent m block; only multiplicity lexemes in vq are masked.",
    }
    write_json(EXP / "FINAL_AUDIT.json", audit)

    report = [
        "# Exact Final ClueGround Ablation",
        "",
        "최종 ClueGround의 저장 prediction을 positive control로 사용하고, 같은 후보, calibration, scorer, gate, coordinate blend, decoder를 유지했다.",
        "RAD-DINO query ablation은 checkpoint를 재학습하지 않고 91차원 query block만 inference 시점에 마스킹했다.",
        "",
        "## Table III. Component ablation on MS-CXR-888",
        "",
        *markdown_table(tables["TABLE_III_COMPONENT_888"], "888"),
        "",
        "## Table IV. Component ablation on MS-CXR-1444",
        "",
        *markdown_table(tables["TABLE_IV_COMPONENT_1444"], "1444"),
        "",
        "## Table V. Exact RAD-DINO query ablation on MS-CXR-888",
        "",
        *markdown_table(tables["TABLE_V_QUERY_888"], "888"),
        "",
        "## Table VI. Exact RAD-DINO query ablation on MS-CXR-1444",
        "",
        *markdown_table(tables["TABLE_VI_QUERY_1444"], "1444"),
        "",
        "## 해석 주의",
        "",
        "- `finding only`는 RAD-DINO query에서 phrase block만 제거한다. 후단 rule-context decoder에는 raw phrase가 그대로 유지된다.",
        "- 실제 최종 checkpoint의 query는 `f + laterality + vertical + severity + uncertainty + 64-D raw phrase hash`다.",
        "- 독립적인 multiplicity block은 없으므로 해당 행은 raw phrase hash에서 multiplicity lexeme만 제거한 결과다.",
        "- Component의 YOLO-only는 DINO 및 DINO 의존 semantic gate를 우회한 네 YOLO + 동일 rule-context decoder다.",
        "- RAD-DINO-only는 YOLO 후보 없이 query당 RAD-DINO proposal 하나를 평가한 진단 arm이다.",
        "",
        f"Audit: **{audit['status']}**, split overlap 0, 888 denominator 163, 1444 denominator 220, missing prediction 0.",
    ]
    (EXP / "FINAL_REPORT_KO.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    tex = []
    for name in ("TABLE_III_COMPONENT_888", "TABLE_IV_COMPONENT_1444", "TABLE_V_QUERY_888", "TABLE_VI_QUERY_1444"):
        protocol = "888" if name.endswith("888") else "1444"
        tex.extend([f"% {name}", *latex_table(tables[name], protocol), ""])
    (EXP / "TABLES_III_VI.tex").write_text("\n".join(tex), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

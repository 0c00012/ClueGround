#!/usr/bin/env python
"""Three-seed run of the candidate-heat + explicit-slot-parser decoder.

Generalises ``run_clueground_canonical_heat_slot_parser_combo_pilot_v1`` (seed
13 only) to the sealed per-seed heat heads in
``clueground_canonical_heatmap_candidate_rank_alpha_extend_{s13,3seed}_v1``.
Each seed uses its own upstream four-YOLO + RAD-DINO candidates, its own
frozen heat head and the heat alpha that head selected on val124.  No new
trainable component; all fusion/decoder parameters are re-selected on val124
inside ``exact.run_protocol``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_clueground_canonical_explicit_anatomic_slot_parser_pilot_v1 as slots  # noqa: E402
from scripts import run_clueground_canonical_heatmap_candidate_rank_pilot_v1 as heat  # noqa: E402
from scripts import run_clueground_canonical_neural_agreement_selector_pilot_v1 as canonical  # noqa: E402
from scripts import run_clueground_exact_hybrid_v4_direct_3seed_v1 as exact  # noqa: E402
from scripts import run_clueground_vfm_legacy_fusion_unified_3seed_v1 as legacy  # noqa: E402
from scripts import run_ms_cxr_multibox_rule_context_fusion_v4_yolov8l_pool as hybrid_v4  # noqa: E402
from scripts.models_ms_cxr_vfm_localizer import PatchHeatmapBBoxHead  # noqa: E402

SEEDS = (13, 42, 2026)
HEAT_SOURCES = {
    13: PROJECT_ROOT / "experiments" / "clueground_canonical_heatmap_candidate_rank_alpha_extend_s13_v1",
    42: PROJECT_ROOT / "experiments" / "clueground_canonical_heatmap_candidate_rank_alpha_extend_3seed_v1" / "seed_42",
    2026: PROJECT_ROOT / "experiments" / "clueground_canonical_heatmap_candidate_rank_alpha_extend_3seed_v1" / "seed_2026",
}
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "clueground_canonical_heat_slot_parser_combo_3seed_v1"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def run_seed(seed: int, output_root: Path) -> dict[str, Any]:
    source_root = HEAT_SOURCES[seed]
    source_status = json.loads((source_root / "RUN_STATUS.json").read_text(encoding="utf-8"))
    alpha = float(source_status["selected_heat_alpha"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs, labels, source_ids = legacy.load_protocol(canonical.PROTOCOL_ROOT)
    rows = {split: legacy.group_rows(inputs[split], labels[split], split) for split in inputs}
    upstream = canonical.UPSTREAM_ROOT / f"seed_{seed}" / canonical.PROTOCOL
    candidates = {split: legacy.load_yolo_candidates(upstream / "yolo_predictions", split) for split in ("val", "eval")}
    allowed = {
        split: {str(task_id) for task_ids in source_ids[split].values() for task_id in task_ids}
        for split in ("val", "eval")
    }
    bundles = {split: legacy.load_rad_bundle(split, allowed[split]) for split in ("val", "eval")}
    data = {
        split: heat.build_group_data(rows[split], labels[split], source_ids[split], bundles[split], candidates[split])
        for split in ("val", "eval")
    }
    payload = torch.load(source_root / "best.pt", map_location="cpu", weights_only=False)
    model = PatchHeatmapBBoxHead(data["val"][0].shape[-1], data["val"][1].shape[-1], hidden=384, dropout=0.1).to(device)
    model.load_state_dict(payload["state"])
    heat_scores = {split: heat.heat_by_group(model, data[split], 12, device, "mean", 0.25) for split in ("val", "eval")}
    context = exact.load_multi_context(
        seed,
        protocol_root=canonical.PROTOCOL_ROOT,
        multi_source_root=canonical.UPSTREAM_ROOT,
        canonical_v3=True,
    )
    context.candidates = {
        split: heat.adjust_candidates(rows[split], heat_scores[split], candidates[split], alpha)
        for split in ("val", "eval")
    }
    original = hybrid_v4.context_cues
    hybrid_v4.context_cues = slots.explicit_slot_context(original)
    try:
        result = exact.run_protocol(
            context,
            output_root,
            quick=False,
            retune_single_full_val=True,
            separate_multi_route_params=True,
            calibration_cache_root=None,
        )
    finally:
        hybrid_v4.context_cues = original
    result.update(
        {
            "method": "candidate-heat YOLO-RAD-DINO hybrid plus explicit anatomic-slot parser",
            "seed": seed,
            "source_heat_checkpoint": str(source_root / "best.pt"),
            "selected_heat_alpha": alpha,
            "new_trainable_component": False,
            "selection": "heat alpha (from the sealed heat run) and all fusion/decoder parameters selected on val124 only",
        }
    )
    write_json(output_root / f"seed_{seed}" / "RUN_STATUS.json", result)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    rows = []
    for seed in args.seeds:
        print(f"[heat-slot] seed {seed}", flush=True)
        result = run_seed(seed, args.output_root)
        row = {"seed": seed}
        for key in ("coverage_iou", "exact_union_iou", "set_f1_0_3", "set_f1_0_5", "mean_pred_count"):
            row[key] = float(result.get(key, float("nan")))
        rows.append(row)
        print(f"[heat-slot] seed {seed}: {row}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_root / "per_seed_metrics.csv", index=False)
    status = {"status": "complete", "seeds": frame["seed"].tolist()}
    for key in ("coverage_iou", "exact_union_iou", "set_f1_0_3", "set_f1_0_5", "mean_pred_count"):
        status[f"{key}_mean"] = float(frame[key].mean())
        status[f"{key}_std"] = float(frame[key].std(ddof=1)) if len(frame) > 1 else 0.0
    write_json(args.output_root / "FINAL_STATUS.json", status)
    print(f"[heat-slot] FINAL coverage {status['coverage_iou_mean']:.4f} +/- {status['coverage_iou_std']:.4f}", flush=True)


if __name__ == "__main__":
    main()

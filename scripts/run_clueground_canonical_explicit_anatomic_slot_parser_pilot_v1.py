#!/usr/bin/env python
"""Canonical seed-13 pilot for explicit same-phrase anatomic slots.

This is a parser-only change.  It preserves the four finding-conditioned YOLO
proposal models, frozen RAD-DINO head, historical fusion, and decoder.  The
extension recognizes two or three *fully stated* side-plus-vertical regions,
such as ``right upper lobe, left upper lobe and right lower lung``.  It does
not infer regions from severity, diffuse wording, or gold-box count.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_clueground_canonical_neural_agreement_selector_pilot_v1 as canonical
from scripts import run_clueground_exact_hybrid_v4_direct_3seed_v1 as exact
from scripts import run_ms_cxr_multibox_rule_context_fusion_v4_yolov8l_pool as hybrid_v4


DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "clueground_canonical_explicit_anatomic_slot_parser_pilot_s13_v1"
_SLOT = re.compile(
    r"\b(right|left)\s+(upper|apical|middle|mid|lower|basal|basilar)\s+"
    r"(?:lobe|lung|base|hemithorax)\b",
    flags=re.IGNORECASE,
)
_VERTICAL = {"apical": "upper", "mid": "middle", "basal": "lower", "basilar": "lower"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def explicit_slot_context(original: Any):
    def context(claim: str, finding: str, base_q: dict[str, str]) -> dict[str, Any]:
        cue = dict(original(claim, finding, base_q))
        slots: list[tuple[str, str]] = []
        for side, vertical in _SLOT.findall(str(claim)):
            slot = (side.lower(), _VERTICAL.get(vertical.lower(), vertical.lower()))
            if slot not in slots:
                slots.append(slot)

        # Four enumerated lobes frequently map to two broad annotations in
        # MS-CXR.  Only replace the existing route for exactly two or three
        # explicitly written slots; no implicit lobe completion is allowed.
        if len(slots) in (2, 3):
            cue["has_multi_cue"] = True
            cue["k_hint"] = len(slots)
            cue["target_qs"] = [
                hybrid_v4.with_q(base_q, laterality=side, vertical=vertical)
                for side, vertical in slots
            ]
            cue["cue_text"] = "explicit_anatomic_slots"
            cue["cue_policy"] = "explicit_side_vertical_slots"
        else:
            cue["cue_policy"] = "legacy"
        return cue

    return context


def main() -> None:
    args = parse_args()
    if not args.execute:
        write_json(args.output_root / "RUN_PLAN.json", {
            "status": "PLAN_ONLY",
            "method": "explicit side-plus-vertical raw-phrase parser extension",
            "seed": args.seed,
        })
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    original = hybrid_v4.context_cues
    hybrid_v4.context_cues = explicit_slot_context(original)
    try:
        context = exact.load_multi_context(
            args.seed,
            protocol_root=canonical.PROTOCOL_ROOT,
            multi_source_root=canonical.UPSTREAM_ROOT,
            canonical_v3=True,
        )
        result = exact.run_protocol(
            context,
            args.output_root,
            quick=False,
            retune_single_full_val=True,
            separate_multi_route_params=True,
            calibration_cache_root=None,
        )
    finally:
        hybrid_v4.context_cues = original

    result.update({
        "method": "existing YOLO-RAD-DINO hybrid with explicit anatomic-slot parser",
        "parser_change": "two or three complete left/right plus vertical anatomic slots become target-specific decoder queries",
        "unchanged": [
            "four finding-conditioned YOLO proposal generators",
            "RAD-DINO backbone and head",
            "candidate boxes",
            "historical hand-scored fusion",
            "validation-only fusion and decoder selection",
            "no HGB", "no WBF", "no CIG", "no new foundation", "no learned ranker",
        ],
        "selection": "val124 only; eval220 is used only for final metrics",
    })
    write_json(args.output_root / "RUN_STATUS.json", result)


if __name__ == "__main__":
    main()

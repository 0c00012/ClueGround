#!/usr/bin/env python
"""Actually import the main ClueGround entry modules (catches missing relative imports).

``audit_release_structure.py`` only resolves absolute ``scripts.*``/``src.*``
imports statically; this executes the module top levels of the re-ranker
path so that missing submodules (for example relative imports inside
``src/three_task_grounding``) fail loudly.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

MODULES = (
    "src.baseline_repro.evaluator",
    "src.three_task_grounding.pipeline",
    "src.three_task_grounding.manifests",
    "scripts.run_clueground_vfm_legacy_fusion_unified_3seed_v1",
    "scripts.run_clueground_exact_hybrid_v4_direct_3seed_v1",
    "scripts.run_clueground_canonical_learned_reranker_3seed_v1",
    "scripts.run_clueground_direct888_learned_reranker_3seed_v1",
    "scripts.run_clueground_reranker_ablations_v1",
    "scripts.run_clueground_reranker_oof_v1",
    "scripts.compare_clueground_vs_medgrounder_ci_v1",
    "scripts.compile_reranker_paper_numbers_v1",
    "scripts.compile_reranker_per_finding_v1",
)


def main() -> None:
    failures = []
    for name in MODULES:
        try:
            importlib.import_module(name)
            print("ok  ", name)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            print("FAIL", name, exc)
    if failures:
        raise SystemExit("import smoke failed:\n" + "\n".join(failures))
    print(f"PASS: imported {len(MODULES)} modules")


if __name__ == "__main__":
    main()

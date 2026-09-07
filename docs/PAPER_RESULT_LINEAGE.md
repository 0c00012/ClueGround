# Paper result lineage

The machine-readable row-level map is `docs/PAPER_RESULT_LINEAGE.csv`. Aggregate manuscript checks are under `results/`.

## Main ClueGround results

| Paper result | Data split | Seeds and scope | Main script | Aggregate artifact |
|---|---|---|---|---|
| 888 Mean IoU 0.5486 +/- 0.0107 | train638 / val87 / eval163 | full upstream 13/42/2026 | `scripts/run_clueground_finding_moe_singlebox_888_no_siglip_score_ablation_v1.py` | `results/main/controlled_singlebox_888.csv` |
| 1444 Coverage 0.5373 +/- 0.0035 | legacy train814 / val125 / eval220 | fixed legacy candidate table plus downstream seeds 13/42/2026 | `scripts/run_clueground_finding_moe_full_upstream_no_siglip_score_ablation_v1.py` | `results/main/clueground_1444_historical_per_seed.csv` and `clueground_1444_historical_status.json` |

The 1444 result is reproducible from the archived local artifact chain but is not equivalent to independently retraining every upstream component for all three seeds. This distinction is part of the release record.

## Table mapping

- Main local comparisons: `results/main/controlled_singlebox_888.csv`, `results/main/controlled_multibox_1444.csv`
- Component ablation: `results/ablations/component_888.csv`, `results/ablations/component_1444.csv`
- RAD-DINO query ablation: `results/ablations/query_888.csv`, `results/ablations/query_1444.csv`
- End-to-end context comparison: `results/ablations/context_aggregate.csv`

The current submitted PDF was checked separately from the older handoff PDF. Their SHA-256 values differ, so the current PDF, not the handoff copy, is the manuscript source of truth for table numbering and wording.

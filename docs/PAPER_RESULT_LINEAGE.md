# Paper result lineage

The machine-readable row-level map is `docs/PAPER_RESULT_LINEAGE.csv`. Rows whose `paper_table` ends in `(revised)` back the revised manuscript (learned re-ranker, 2026-09-09); rows marked `SUPERSEDED_PRELIMINARY_PATH` back the submitted manuscript and are kept for the record. Aggregate manuscript checks are under `results/`.

## Main ClueGround results (revised manuscript)

| Paper result | Data split | Seeds and scope | Main script | Aggregate artifact |
|---|---|---|---|---|
| 888 Mean IoU 0.5539 +/- 0.0172 | train638 / val87 / eval163 | full upstream 13/42/2026; re-ranker trained on train638, `alpha` on val87 | `scripts/run_clueground_direct888_learned_reranker_3seed_v1.py` | `results/main/clueground_888_reranker_per_seed.csv` |
| 1444 Coverage 0.5419 +/- 0.0034, Exact Union 0.5433 +/- 0.0050, SetF1@0.3 0.7873, SetF1@0.5 0.6246 | canonical train813 / val124 / eval220 | full upstream 13/42/2026, no shared assets; re-ranker trained on train813, `alpha` on val124 | `scripts/run_clueground_canonical_learned_reranker_3seed_v1.py --variants learned --no-aux-assets --alpha-selection val` | `results/main/clueground_1444_reranker_per_seed.csv` |

Both results are independent three-seed runs of every trained component. `results/main/reranker_paper_numbers.json` is the single sheet from which the revised manuscript's numbers were transcribed.

## Superseded results (submitted manuscript)

| Paper result | Data split | Seeds and scope | Main script | Aggregate artifact |
|---|---|---|---|---|
| 888 Mean IoU 0.5486 +/- 0.0107 | train638 / val87 / eval163 | full upstream 13/42/2026 | `scripts/run_clueground_finding_moe_singlebox_888_no_siglip_score_ablation_v1.py` | `results/main/controlled_singlebox_888.csv` |
| 1444 Coverage 0.5373 +/- 0.0035 | legacy train814 / val125 / eval220 | fixed legacy candidate table plus downstream seeds 13/42/2026 | `scripts/run_clueground_finding_moe_full_upstream_no_siglip_score_ablation_v1.py` | `results/main/clueground_1444_historical_per_seed.csv` and `clueground_1444_historical_status.json` |

The historical 1444 result is reproducible from the archived local artifact chain but is not equivalent to independently retraining every upstream component for all three seeds. See `docs/KNOWN_LIMITATIONS.md`.

## Table mapping (revised manuscript)

- Main local comparison (Table 1, Supplementary Table S1): `results/main/controlled_singlebox_888.csv`, `results/main/controlled_multibox_1444.csv` (baselines, unchanged) plus the re-ranker rows above
- Component ablation (Table 2): `component` rows of `results/ablations/reranker_component_query_888.csv` and `results/ablations/reranker_component_query_1444.csv`; the "no re-ranker" row is the `Ours YOLO-RAD-DINO hybrid-v4` row of the controlled tables
- RAD-DINO query ablation (Tables 3 and 4): `query` rows of the same two files
- End-to-end context comparison (Table 5): `results/ablations/context_aggregate.csv` (unchanged)
- Results by finding category and by number of reference regions (Supplementary Tables S4 and S5): `results/per_finding/*.csv`, regrouped from the sealed row-level scoring by `scripts/compile_reranker_per_finding_v1.py` (no new inference)
- External zero-shot evaluation on PadChest-GR (main-text Table 6; label mapping in Supplementary Table S6, results by finding in Supplementary Table S7): `results/external/` (`PADCHEST_NUMBERS.json` is the transcription source; protocol counts in `padchest_gr_*_PROTOCOL_SUMMARY.json`; runners `scripts/build_padchest_gr_protocol_v1.py`, `scripts/run_clueground_padchest_gr_zero_shot_v1.py`, `scripts/run_medgrounder_padchest_zero_shot_v1.py`)
- Not reported in the manuscript but kept as records: paired patient-cluster bootstrap comparisons (`results/comparison/`), re-ranker robustness configurations and the out-of-fold training-proposal run (`results/robustness/`)

The current submitted PDF was checked separately from the older handoff PDF. Their SHA-256 values differ, so the current PDF, not the handoff copy, is the manuscript source of truth for table numbering and wording.

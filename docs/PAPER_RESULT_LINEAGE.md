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

## Table mapping (submitted manuscript, 2026-10-13: "ClueGround: Context-Guided YOLO-RAD-DINO Fusion for Multi-Region Chest X-Ray Phrase Grounding")

Main text
- Table 1 (main local comparison) and Supplementary Table S1 (full threshold metrics): `results/main/controlled_singlebox_888.csv`, `results/main/controlled_multibox_1444.csv` (baselines, unchanged) plus the re-ranker rows above (`results/main/clueground_888_reranker_per_seed.csv`, `results/main/clueground_1444_reranker_per_seed.csv`)
- Table 2 (component ablation): `component` rows of `results/ablations/reranker_component_query_888.csv` and `results/ablations/reranker_component_query_1444.csv`; the "no re-ranker" row is the `Ours YOLO-RAD-DINO hybrid-v4` row of the controlled tables
- Tables 3 and 4 (RAD-DINO query ablation, 888 and 1444): `query` rows of the same two files
- Table 5 (zero-shot results on the PadChest-GR test split, strict and extended mappings): `results/external/PADCHEST_NUMBERS.json` / `.md` (transcription source), MedGrounder zero-shot `results/external/medgrounder_zero_shot_*`; protocol counts in `results/external/padchest_gr_*_PROTOCOL_SUMMARY.json`; runners `scripts/build_padchest_gr_protocol_v1.py`, `scripts/run_clueground_padchest_gr_zero_shot_v1.py`, `scripts/run_medgrounder_padchest_zero_shot_v1.py`
- Figure 2 (MS-CXR-1444 cases) and Figure 3 (PadChest-GR cases): drawn from the sealed per-example predictions, which are not redistributed; the case-level metrics printed in the panels follow `src/baseline_repro/evaluator.py`

Supplementary Information
- Table S1: see Table 1 above
- Tables S2 and S3 (released-checkpoint references on MS-CXR-888 and MS-CXR-1444): rows VIII and IX of `docs/PAPER_RESULT_LINEAGE.csv`; released models are not bundled (see `docs/BASELINES.md`)
- Tables S4 and S5 (results by finding category and by number of reference regions): `results/per_finding/*.csv`, regrouped from the sealed row-level scoring by `scripts/compile_reranker_per_finding_v1.py` (no new inference)
- Table S6 (PadChest-GR label mapping): `scripts/padchest_gr_label_map.json`; Table S7 (PadChest-GR results by finding category): `results/external/`
- Table S8 (deterministic context extraction examples): rule outputs of the parser in `src/` (no numeric results)
- Table S9 (end-to-end context-representation comparison): `results/ablations/context_aggregate.csv` (unchanged)
- Figure S1 (data flow and finding-category distribution): split sizes in `docs/SPLIT_FINGERPRINTS.csv` (direct-888 638/87/163; canonical-1444 813/124/220 groups with 996/164/280 boxes)
- Not reported in the manuscript but kept as records: paired patient-cluster bootstrap comparisons (`results/comparison/`), re-ranker robustness configurations and the out-of-fold training-proposal run (`results/robustness/`)

The current submitted PDF was checked separately from the older handoff PDF. Their SHA-256 values differ, so the current PDF, not the handoff copy, is the manuscript source of truth for table numbering and wording.

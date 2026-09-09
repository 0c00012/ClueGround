# Reproducibility

## 1. Environment

Create a Python 3.10/3.11 environment, install a CUDA-compatible PyTorch build, and install `requirements.txt`. Configure `.env.example` values in the shell.

## 2. Data preparation

Place credentialed MS-CXR v1.1.0 annotations and MIMIC-CXR-JPG images under the configured roots. Reconstructed split files must match `docs/SPLIT_FINGERPRINTS.csv` before comparing results.

## 3. Reported ClueGround runs (learned re-ranker, revised manuscript)

The re-ranker consumes the per-seed candidate tables of the YOLO + RAD-DINO hybrid stage. Generate them first:

```bash
python scripts/run_clueground_exact_hybrid_v4_direct_3seed_v1.py --device cuda
```

MS-CXR-1444 main configuration (no auxiliary assets, `alpha` on validation):

```bash
python scripts/run_clueground_canonical_learned_reranker_3seed_v1.py --variants learned --no-aux-assets --alpha-selection val
```

Direct-888:

```bash
python scripts/run_clueground_direct888_learned_reranker_3seed_v1.py --alpha-selection val
```

Component and RAD-DINO query ablations (downstream frozen to the sealed runs; a positive control must reproduce the sealed numbers exactly):

```bash
python scripts/run_clueground_reranker_ablations_v1.py
python scripts/run_clueground_reranker_ablations_888_v1.py
```

Robustness rows of Supplementary Table S5:

```bash
python scripts/run_clueground_canonical_learned_reranker_3seed_v1.py --variants learned --no-aux-assets --no-retune
python scripts/run_clueground_canonical_learned_reranker_3seed_v1.py --variants learned --no-aux-assets --multi-alpha zero
python scripts/run_clueground_canonical_learned_reranker_3seed_v1.py --variants learned --no-aux-assets --alpha-selection train_oof_val
python scripts/run_clueground_canonical_learned_reranker_3seed_v1.py --variants learned
python scripts/run_clueground_reranker_oof_v1.py --folds 3 --yolo-epochs 100 --rad-epochs 120
```

Paired, patient-cluster bootstrap comparisons and the manuscript number sheet:

```bash
python scripts/compare_clueground_vs_medgrounder_ci_v1.py --extra reranker_noaux_valalpha=experiments/clueground_canonical_learned_reranker_noaux_assets_v1/learned/multibox_1444/seed_{seed}/eval_predictions.jsonl
python scripts/compare_direct888_ci_v1.py
python scripts/compile_reranker_paper_numbers_v1.py
python scripts/compile_reranker_per_finding_v1.py
```

End-to-end context comparison (unchanged from the submitted manuscript):

```bash
python scripts/run_clueground_context_ablation_no_siglip_v2.py
```

Preliminary-path runs of the submitted manuscript are kept for the record: `scripts/run_clueground_finding_moe_singlebox_888_no_siglip_score_ablation_v1.py`, `scripts/run_clueground_finding_moe_full_upstream_no_siglip_score_ablation_v1.py`, `scripts/run_clueground_exact_final_ablation_v1.py`, and the per-seed rebuild of that chain `scripts/run_clueground_canonical_aux_chain_3seed_v1.py`.

The research scripts use project-relative discovery for generated upstream artifacts. Checkpoint and candidate caches are not part of this repository, so reproducing the exact numbers requires regenerating them from the credentialed data. Tree-ensemble scorers are reproducible only when candidate tables are read with a round-trip float parser; the re-ranker module enforces this.

## 4. Seeds and aggregation

The standard seed list is `13, 42, 2026`. Aggregate means use the arithmetic mean; reported standard deviations are sample standard deviations (`ddof=1`). Prediction-level seed ensembling is not used for the paper results. Every seed has its own YOLO detectors, RAD-DINO head, re-ranker, `alpha`, and fusion calibration; no asset is shared across seeds.

## 5. Integrity checks

```bash
python tools/verify_release.py
python tools/audit_release_structure.py
python tools/import_smoke.py
python -m compileall -q scripts src baselines tools
```

`tools/import_smoke.py` actually imports the re-ranker entry modules (the static audit does not follow relative imports inside `src/`). `src/three_task_grounding`, `src/rerank/{multibox_cue_parser,unified_adaptive_cardinality}.py` and `src/fair_baselines/metrics.py` are the method-package versions that the sealed runs resolved first on `sys.path`; they differ from the older repository copies and must not be replaced.

The published checksum manifest verifies the code release itself, not protected datasets or model weights. The re-ranker runners additionally support `--audit-gold-mutation`, which perturbs evaluation labels and asserts that predictions do not change.

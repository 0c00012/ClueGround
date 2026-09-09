# ClueGround

Research code and aggregate evaluation artifacts for **Context-Guided YOLO-RAD-DINO Fusion for Single- and Multi-Region Chest X-Ray Phrase Grounding**.

ClueGround combines four finding-conditioned YOLO proposal generators with a frozen RAD-DINO encoder, a lightweight phrase-conditioned localization head, a learned candidate re-ranker trained on the training split, validation-calibrated candidate fusion, and a deterministic context-guided decoder that can return one or more boxes.

## Reported results (revised manuscript, learned re-ranker)

| Evaluation set | Metric | ClueGround |
|---|---|---:|
| MS-CXR-888, 163 single-region phrases | Mean IoU | 0.5539 +/- 0.0172 |
| MS-CXR-888 | Hit@0.3 / Hit@0.5 | 0.7996 / 0.6033 |
| MS-CXR-1444, 220 phrase groups / 280 boxes | Coverage IoU | 0.5419 +/- 0.0034 |
| MS-CXR-1444 | Exact union IoU | 0.5433 +/- 0.0050 |
| MS-CXR-1444 | SetF1@0.3 / SetF1@0.5 | 0.7873 / 0.6246 |

Both results are independent three-seed runs (13, 42, 2026) of every trained component on the canonical splits (direct-888: 638/87/163 phrases; MS-CXR-1444: 813/124/220 phrase groups). ClueGround exceeds the locally retrained MedGrounder on Coverage IoU, Exact Union IoU and SetF1@0.5 for every seed, but the seed-wise paired 95% patient-cluster bootstrap intervals include zero and MedGrounder remains higher on SetF1@0.3; MedRPG and TransVG remain higher on MS-CXR-888. See [Known limitations](docs/KNOWN_LIMITATIONS.md), [result lineage](docs/PAPER_RESULT_LINEAGE.md), and `results/comparison/`.

The submitted manuscript's values (888 mean IoU 0.5486 +/- 0.0107; 1444 Coverage 0.5373 +/- 0.0035 on the historical 814/125 split with a fixed legacy candidate table) are retained under `results/main/` for the record and are superseded.

## Repository contents

- `scripts/`: ClueGround training, inference, re-ranking, fusion, ablation, and comparison entry points.
- `src/`: shared evaluators, metrics, model components, and data utilities.
- `baselines/`: controlled local baseline wrappers and common evaluation code.
- `results/`: aggregate and per-seed non-identifying result tables used for manuscript checks (`main/`, `ablations/`, `robustness/`, `comparison/`).
- `docs/`: method, data access, result lineage, baseline fidelity, and reproduction notes.
- `checksums/`: release file hashes.

The repository intentionally excludes MS-CXR/MIMIC-CXR images and annotations, patient-level identifiers, model checkpoints, patch-token caches, and per-example predictions.

## Setup

Python 3.10 or 3.11 and a CUDA-capable PyTorch installation are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Windows, activate with `.venv\\Scripts\\activate`. Install the PyTorch build matching the local CUDA driver before installing the remaining requirements.

Copy `.env.example` to `.env` or export the listed variables. The scripts expect a credentialed local copy of MS-CXR and MIMIC-CXR-JPG; no protected data are downloaded by this repository.

## Main entry points

- MS-CXR-1444 ClueGround (learned re-ranker): `scripts/run_clueground_canonical_learned_reranker_3seed_v1.py`
- Direct-888 ClueGround (learned re-ranker): `scripts/run_clueground_direct888_learned_reranker_3seed_v1.py`
- Upstream YOLO-RAD-DINO hybrid and fusion stage: `scripts/run_clueground_exact_hybrid_v4_direct_3seed_v1.py`
- Component/query ablations of the re-ranker path: `scripts/run_clueground_reranker_ablations_v1.py`, `scripts/run_clueground_reranker_ablations_888_v1.py`
- Out-of-fold training-proposal robustness check: `scripts/run_clueground_reranker_oof_v1.py`
- Paired, patient-cluster bootstrap comparisons: `scripts/compare_clueground_vs_medgrounder_ci_v1.py`, `scripts/compare_direct888_ci_v1.py`
- Manuscript number sheet: `scripts/compile_reranker_paper_numbers_v1.py`
- End-to-end context comparison: `scripts/run_clueground_context_ablation_no_siglip_v2.py`
- Common evaluator: `src/baseline_repro/evaluator.py`

Run `python tools/verify_release.py` before using a tagged release. Detailed commands and expected prerequisites are in [Reproducibility](docs/REPRODUCIBILITY.md).

## Data availability

MS-CXR v1.1.0 and the linked MIMIC-CXR data must be obtained independently from PhysioNet. Access is credentialed and requires acceptance of the applicable data use agreement. This repository distributes only code, split fingerprints, and aggregate metrics.

## Code availability statement

The source code, evaluation scripts, aggregate results, and reproducibility documentation supporting this study are available in this repository. The MS-CXR and MIMIC-CXR data are not redistributed and must be obtained independently through PhysioNet under credentialed access and the applicable data use agreement.

## License

The original code in this repository is released under the MIT License. Third-party model code and pretrained weights remain subject to their respective licenses.

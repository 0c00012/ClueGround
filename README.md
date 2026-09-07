# ClueGround

Research code and aggregate evaluation artifacts for **Context-Guided YOLO-RAD-DINO Fusion for Single- and Multi-Region Chest X-Ray Phrase Grounding**.

ClueGround combines four finding-conditioned YOLO proposal generators with a frozen RAD-DINO encoder, a lightweight phrase-conditioned localization head, validation-calibrated candidate fusion, and a deterministic context-guided decoder that can return one or more boxes.

## Reported results

| Evaluation set | Metric | ClueGround |
|---|---|---:|
| MS-CXR-888, 163 single-region phrases | Mean IoU | 0.5486 +/- 0.0107 |
| MS-CXR-888 | Hit@0.3 / Hit@0.5 | 0.7730 / 0.6258 |
| MS-CXR-1444, 220 phrase groups / 280 boxes | Coverage IoU | 0.5373 +/- 0.0035 |
| MS-CXR-1444 | Exact union IoU | 0.5373 +/- 0.0035 |
| MS-CXR-1444 | SetF1@0.3 / SetF1@0.5 | 0.7727 / 0.6148 |

The direct-888 result uses independent upstream seeds 13, 42, and 2026. The reported 1444 result uses the historical 814/125 train/validation split and a fixed legacy candidate table with a finding-conditioned downstream gate evaluated across seeds 13, 42, and 2026. It must not be described as a fully regenerated upstream three-seed result. See [Known limitations](docs/KNOWN_LIMITATIONS.md) and [result lineage](docs/PAPER_RESULT_LINEAGE.md).

## Repository contents

- `scripts/`: ClueGround training, inference, fusion, and ablation entry points.
- `src/`: shared evaluators, metrics, model components, and data utilities.
- `baselines/`: controlled local baseline wrappers and common evaluation code.
- `results/`: aggregate and per-seed non-identifying result tables used for manuscript checks.
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

- Direct-888 ClueGround: `scripts/run_clueground_finding_moe_singlebox_888_no_siglip_score_ablation_v1.py`
- MS-CXR-1444 ClueGround: `scripts/run_clueground_finding_moe_full_upstream_no_siglip_score_ablation_v1.py`
- Shared YOLO-RAD-DINO hybrid: `scripts/run_clueground_exact_hybrid_v4_direct_3seed_v1.py`
- Final component/query ablations: `scripts/run_clueground_exact_final_ablation_v1.py`
- End-to-end context comparison: `scripts/run_clueground_context_ablation_no_siglip_v2.py`
- Common evaluator: `src/baseline_repro/evaluator.py`

Run `python tools/verify_release.py` before using a tagged release. Detailed commands and expected prerequisites are in [Reproducibility](docs/REPRODUCIBILITY.md).

## Data availability

MS-CXR v1.1.0 and the linked MIMIC-CXR data must be obtained independently from PhysioNet. Access is credentialed and requires acceptance of the applicable data use agreement. This repository distributes only code, split fingerprints, and aggregate metrics.

## Code availability statement

The source code, evaluation scripts, aggregate results, and reproducibility documentation supporting this study are available in this repository. The MS-CXR and MIMIC-CXR data are not redistributed and must be obtained independently through PhysioNet under credentialed access and the applicable data use agreement.

## License

The original code in this repository is released under the MIT License. Third-party model code and pretrained weights remain subject to their respective licenses.

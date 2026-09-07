# Reproducibility

## 1. Environment

Create a Python 3.10/3.11 environment, install a CUDA-compatible PyTorch build, and install `requirements.txt`. Configure `.env.example` values in the shell.

## 2. Data preparation

Place credentialed MS-CXR v1.1.0 annotations and MIMIC-CXR-JPG images under the configured roots. Reconstructed split files must match `docs/SPLIT_FINGERPRINTS.csv` before comparing results.

## 3. Reported ClueGround runs

Direct-888:

```bash
python scripts/run_clueground_finding_moe_singlebox_888_no_siglip_score_ablation_v1.py --device cuda
```

MS-CXR-1444 historical no-SigLIP run:

```bash
python scripts/run_clueground_finding_moe_full_upstream_no_siglip_score_ablation_v1.py --device cuda
```

Final component and query ablations:

```bash
python scripts/run_clueground_exact_final_ablation_v1.py --device cuda
python scripts/compile_clueground_exact_final_ablation_v1.py
```

End-to-end context comparison:

```bash
python scripts/run_clueground_context_ablation_no_siglip_v2.py
```

The research scripts use project-relative discovery for generated upstream artifacts. Checkpoint and candidate caches are not part of this repository, so reproducing the exact numbers requires regenerating them from the credentialed data. Do not compare a newly generated canonical 813/124 run directly to the historical 814/125 result without labeling the split difference.

## 4. Seeds and aggregation

The standard seed list is `13, 42, 2026`. Aggregate means use the arithmetic mean; reported standard deviations are sample standard deviations (`ddof=1`). Prediction-level seed ensembling is not used for the paper results.

## 5. Integrity checks

```bash
python tools/verify_release.py
python -m compileall -q scripts src baselines tools
```

The published checksum manifest verifies the code release itself, not protected datasets or model weights.

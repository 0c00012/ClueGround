# Baseline implementation notes

All locally trained comparisons use seeds 13, 42, and 2026 and the common evaluator in `src/baseline_repro/evaluator.py`. Released-model references are not bundled.

| Baseline | Public wrapper | Fidelity note |
|---|---|---|
| MedRPG | `baselines/linux_matrix/run_medrpg_train_compat_wrapper_v2.py` | Official-code compatibility wrapper, 90-epoch local retraining |
| TransVG | `baselines/linux_matrix/run_transvg_train_compat_wrapper_v2.py` | Official architecture/code with local MS-CXR wrapper, 90 epochs |
| MedGrounder | `scripts/run_medgrounder_faithful_v2.py` | Recipe-faithful adaptation initialized from Chest ImaGenome localization pretraining |
| Grounding DINO-T | `scripts/run_grounding_dino_native_loss_correction_v2.py` | Local architecture adaptation with native detection loss |
| GuideDecoder / LViT | `scripts/run_fair_baseline_expansion_v1.py` | Segmentation-to-box local adaptations |
| MDETR-style | `scripts/run_ms_cxr_mdetr_fair_baseline_v1.py` | Local MDETR-inspired model, not an official MDETR reproduction |

The multiregion evaluation keeps inherently one-box methods as one-box predictors. It does not add a ClueGround-style count decoder to those models.

# Known limitations and disclosure

Updated 2026-09-09 for the learned candidate re-ranker (the method reported in the revised manuscript).

## Learned re-ranker: what is and is not controlled

The re-ranker is a gradient-boosted regression model trained on the proposals of the training split to predict each proposal's IoU with its best-matching reference box. Its output adjusts the detector confidence in the logit domain with a strength `alpha`. The following facts should be read together with the reported numbers.

1. **Training-side distribution shift.** The training-split proposals are produced by YOLO detectors that were trained on the same images, so detector confidence is more informative on the training split than on unseen images. The scorer is therefore fitted on optimistically biased confidence features. `alpha` is selected on the validation split (unseen by the detectors), which limits but does not remove this bias. A pooled `alpha` rule (train out-of-fold + validation) was pre-registered as an alternative and performed worse (`results/robustness/reranker_1444_pooled_alpha_per_seed.csv`); it selects a larger `alpha` for exactly this reason. Both rules are reported. The principled remedy, generating training proposals with detectors trained on disjoint folds (`scripts/run_clueground_reranker_oof_v1.py`), is reported as a robustness check when its run completes; it is not the main configuration.
2. **Decision timing.** The validation-only `alpha` rule was the original rule. The pooled rule was proposed mid-study, evaluated once on the test split, and found to be worse; the main configuration remained the validation-only rule. This ordering is disclosed because the choice between the two rules was made after both test results were visible. All other main-configuration choices (feature set, scorer family, grids, fusion calibration) were fixed on training or validation data before the test split was evaluated for that configuration.
3. **Number of test evaluations.** Several configurations were evaluated on the test split (feature set with and without auxiliary-estimate features, `alpha` grids, multi-route `alpha`, fusion re-tuning, pooled `alpha`). Each is listed in the manuscript's Supplementary Table S5 and in `results/robustness/`. The main configuration is the simplest one (no auxiliary assets, validation `alpha`, validation-re-tuned fusion), not the best-scoring one.
4. **Statistical strength.** ClueGround exceeds the locally retrained MedGrounder on Coverage IoU, Exact Union IoU and SetF1@0.5 for every seed, but the seed-wise paired 95% patient-cluster bootstrap intervals include zero (`results/comparison/paired_difference_ci_1444.csv`). MedGrounder remains higher on SetF1@0.3. The improvement is concentrated on single-region phrases; multi-region groups (57 of 220 per seed) remain below MedGrounder, and per finding ClueGround is lower for atelectasis, edema, and pneumothorax (`results/per_finding/`).
5. **Direct-888.** MedRPG and TransVG remain higher on the single-region protocol; paired intervals are in `results/comparison/paired_difference_ci_888.csv`.
6. **Integrity checks.** The re-ranker never receives evaluation labels: the scorer is fitted on training rows only, the model family is chosen by subject-grouped cross-validation on the training split, `alpha` and all fusion parameters come from the validation split, and a gold-mutation audit (perturbing evaluation labels must leave predictions unchanged) passes for all three seeds. Positive controls in the ablation runners reproduce the sealed numbers exactly with all downstream parameters frozen.

## External zero-shot evaluation (PadChest-GR)

The PadChest-GR numbers (`results/external/`) apply the sealed MS-CXR models without any retraining or tuning to the PadChest-GR test split, restricted to sentences whose label maps onto an MS-CXR finding (strict mapping 194 sentences, dominated by cardiomegaly 99; extended mapping adds 85 opacity-type sentences as lung opacity). PadChest-GR has no edema or pneumonia sentences, consolidation and pneumothorax have four sentences each, and no baseline was retrained on this set, so the values describe robustness of ClueGround (and the transfer of the re-ranker gain), not a ranking against other methods. Atelectasis and opacity-type sentences transfer poorly. The 16-bit PNGs were min-max scaled to 8 bit per image; PadChest-GR sentences are English translations of Spanish reports and the deterministic parser was not adapted to them.

## Historical MS-CXR-1444 number (0.5373 +/- 0.0035)

The submitted manuscript's value was produced on the historical `814/125/220` split with a fixed legacy candidate table (auxiliary detectors, context head and row-level scorer built once with seed 42 and shared by all three seeds), a no-SigLIP score configuration and a finding-conditioned downstream gate. It was a three-run downstream aggregate but not a full upstream three-seed reconstruction, and the training candidate table lacked the `score_head` field. A per-seed rebuild of that chain on the canonical split gives 0.5245 +/- 0.0106 (validation-selected) and does not exceed MedGrounder. The value is retained in `results/main/clueground_1444_historical_*` for the record and is superseded by the re-ranker path in the revised manuscript.

## Current manuscript versus handoff PDF

The current manuscript PDF differs from the PDF archived in the August handoff. Table numbering and prose should therefore be checked against the current manuscript, while the handoff remains the source for code/artifact provenance.

## Dataset restrictions

Protected images, annotations, identifier-level split manifests, per-example predictions, and checkpoints are omitted. Exact reruns require independent PhysioNet access and regeneration of upstream artifacts.

## Baseline scope

Several baselines are controlled local adaptations rather than exact official MS-CXR recipes. `docs/BASELINES.md` labels these cases explicitly. Baseline numbers were not recomputed for the revision; the re-ranker is compared against the same sealed baseline predictions as the submitted manuscript.

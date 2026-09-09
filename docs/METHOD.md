# Method summary

ClueGround receives a frontal chest radiograph, a finding category, and the raw grounding phrase.

1. Four finding-conditioned YOLO detectors (`YOLOv8s`, `YOLOv8m`, `YOLO11s`, and `YOLO11m`) produce candidate boxes.
2. A frozen RAD-DINO encoder produces patch tokens. A lightweight trained head combines those tokens with the phrase representation and predicts an auxiliary box.
3. A learned candidate re-ranker scores every YOLO proposal. Each proposal is described by tabular features: detector confidence and rank, source model, normalized geometry, IoU with the RAD-DINO box, cross-detector consensus (maximum and mean IoU with the other detectors' proposals), IoU and area ratio against the training-derived spatial prior, the rule-based context-alignment score, and one-hot finding/laterality/vertical-location/multi-region cues. A gradient-boosted regression ensemble trained on the training split predicts the proposal's IoU with its best-matching reference box; the standardized prediction is added to the confidence logit with a strength `alpha` selected on the validation split. Selected coordinates may then be conservatively blended with the RAD-DINO box using weights selected on the validation split (unchanged fusion stage).
4. The deterministic phrase parser extracts laterality, vertical position, and explicit multiplicity cues. For phrases indicating separated regions, the decoder adds spatially diverse candidates from the re-ranked pool; otherwise it preserves the top-ranked single-region route.
5. The same ranked prediction set is produced in both evaluations. MS-CXR-888 evaluates the top-ranked box; MS-CXR-1444 evaluates the complete set.

The paper's selected method does not use SigLIP or BioMedCLIP scores and does not load the seed-42 auxiliary candidate assets of the preliminary version. Historical semantic scripts are retained only to document the ablations that led to their exclusion.

Main code:

- `scripts/run_clueground_canonical_learned_reranker_3seed_v1.py` (MS-CXR-1444 re-ranker: feature table, scorer selection, `alpha`, fusion, decoder)
- `scripts/run_clueground_direct888_learned_reranker_3seed_v1.py` (MS-CXR-888 re-ranker; imports the module above)
- `scripts/run_clueground_exact_hybrid_v4_direct_3seed_v1.py` (upstream YOLO + RAD-DINO hybrid and fusion stage)
- `scripts/run_ms_cxr_singlebox_fair_yolo_dino_fusion_v1.py`
- `scripts/run_ms_cxr_multibox_rule_context_fusion_v4_yolov8l_pool.py`
- `src/baseline_repro/evaluator.py`

# Method summary

ClueGround receives a frontal chest radiograph, a finding category, and the raw grounding phrase.

1. Four finding-conditioned YOLO detectors (`YOLOv8s`, `YOLOv8m`, `YOLO11s`, and `YOLO11m`) produce candidate boxes.
2. A frozen RAD-DINO encoder produces patch tokens. A lightweight trained head combines those tokens with the phrase representation and predicts an auxiliary box.
3. The fusion stage ranks YOLO candidates using detector confidence, source/rank information, spatial context, and agreement with the RAD-DINO box. Selected coordinates may be conservatively blended using weights selected on the validation split.
4. The deterministic phrase parser extracts laterality, vertical position, and explicit multiplicity cues. For phrases indicating separated regions, the decoder adds spatially diverse candidates; otherwise it preserves the top-ranked single-region route.
5. The same ranked prediction set is produced in both evaluations. MS-CXR-888 evaluates the top-ranked box; MS-CXR-1444 evaluates the complete set.

The paper's selected method does not use SigLIP or BioMedCLIP scores. Historical semantic scripts are retained only to document the ablations that led to their exclusion.

Main code:

- `scripts/run_clueground_exact_hybrid_v4_direct_3seed_v1.py`
- `scripts/run_ms_cxr_singlebox_fair_yolo_dino_fusion_v1.py`
- `scripts/run_ms_cxr_multibox_rule_context_fusion_v4_yolov8l_pool.py`
- `src/baseline_repro/evaluator.py`

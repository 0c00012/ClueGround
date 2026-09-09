# ClueGround canonical rebuild vs MedGrounder (canonical 1444, common evaluator)

Seeds: [13, 42, 2026]; selected variants: {13: 'base', 42: 'gate', 2026: 'base'}; bootstrap reps: 2000; cluster: subject_id
MedGrounder scored in its own normalised frame (as in Table 1); gold consistency with canonical pixel gold after inverse letterbox: {13: {'gold_boxes_iou_below_0.99_after_letterbox_inverse': 33, 'min_gold_iou': 0.9646}, 42: {'gold_boxes_iou_below_0.99_after_letterbox_inverse': 33, 'min_gold_iou': 0.9646}, 2026: {'gold_boxes_iou_below_0.99_after_letterbox_inverse': 33, 'min_gold_iou': 0.9646}}

## Seed mean +/- sample std

| method | Coverage | Exact union | SetF1@0.3 | SetF1@0.5 |
|---|---:|---:|---:|---:|
| clueground_canonical | 0.5245 +/- 0.0106 | 0.5240 +/- 0.0099 | 0.7672 +/- 0.0284 | 0.5998 +/- 0.0077 |
| clueground_gate_always | 0.5335 +/- 0.0069 | 0.5327 +/- 0.0076 | 0.7859 +/- 0.0210 | 0.5993 +/- 0.0081 |
| clueground_hybrid_base | 0.5210 +/- 0.0046 | 0.5206 +/- 0.0041 | 0.7627 +/- 0.0212 | 0.6003 +/- 0.0072 |
| clueground_legacy_paper | 0.5373 +/- 0.0035 | 0.5373 +/- 0.0035 | 0.7727 +/- 0.0170 | 0.6148 +/- 0.0149 |
| medgrounder | 0.5337 +/- 0.0036 | 0.5301 +/- 0.0044 | 0.8007 +/- 0.0113 | 0.6069 +/- 0.0147 |
| reranker_aux_valalpha | 0.5408 +/- 0.0090 | 0.5442 +/- 0.0093 | 0.7890 +/- 0.0292 | 0.6322 +/- 0.0236 |
| reranker_noaux_pooledalpha | 0.5282 +/- 0.0147 | 0.5322 +/- 0.0137 | 0.7719 +/- 0.0148 | 0.6118 +/- 0.0172 |
| reranker_noaux_valalpha | 0.5419 +/- 0.0034 | 0.5433 +/- 0.0050 | 0.7873 +/- 0.0226 | 0.6246 +/- 0.0139 |

## Per-seed 95% patient-cluster bootstrap CI

| method | seed | metric | point | CI low | CI high |
|---|---:|---|---:|---:|---:|
| clueground_canonical | 13 | coverage_iou | 0.5180 | 0.4803 | 0.5554 |
| clueground_canonical | 42 | coverage_iou | 0.5367 | 0.5021 | 0.5729 |
| clueground_canonical | 2026 | coverage_iou | 0.5187 | 0.4761 | 0.5617 |
| clueground_gate_always | 13 | coverage_iou | 0.5383 | 0.4945 | 0.5764 |
| clueground_gate_always | 42 | coverage_iou | 0.5367 | 0.5030 | 0.5722 |
| clueground_gate_always | 2026 | coverage_iou | 0.5257 | 0.4864 | 0.5668 |
| clueground_hybrid_base | 13 | coverage_iou | 0.5180 | 0.4797 | 0.5564 |
| clueground_hybrid_base | 42 | coverage_iou | 0.5264 | 0.4914 | 0.5636 |
| clueground_hybrid_base | 2026 | coverage_iou | 0.5187 | 0.4769 | 0.5622 |
| clueground_legacy_paper | 13 | coverage_iou | 0.5412 | 0.5047 | 0.5789 |
| clueground_legacy_paper | 42 | coverage_iou | 0.5362 | 0.5025 | 0.5733 |
| clueground_legacy_paper | 2026 | coverage_iou | 0.5344 | 0.4965 | 0.5691 |
| medgrounder | 13 | coverage_iou | 0.5367 | 0.5032 | 0.5725 |
| medgrounder | 42 | coverage_iou | 0.5345 | 0.4970 | 0.5681 |
| medgrounder | 2026 | coverage_iou | 0.5297 | 0.4934 | 0.5656 |
| reranker_aux_valalpha | 13 | coverage_iou | 0.5420 | 0.5073 | 0.5791 |
| reranker_aux_valalpha | 42 | coverage_iou | 0.5492 | 0.5141 | 0.5870 |
| reranker_aux_valalpha | 2026 | coverage_iou | 0.5312 | 0.4966 | 0.5704 |
| reranker_noaux_pooledalpha | 13 | coverage_iou | 0.5119 | 0.4717 | 0.5519 |
| reranker_noaux_pooledalpha | 42 | coverage_iou | 0.5321 | 0.4946 | 0.5682 |
| reranker_noaux_pooledalpha | 2026 | coverage_iou | 0.5405 | 0.5048 | 0.5762 |
| reranker_noaux_valalpha | 13 | coverage_iou | 0.5457 | 0.5098 | 0.5819 |
| reranker_noaux_valalpha | 42 | coverage_iou | 0.5393 | 0.5026 | 0.5754 |
| reranker_noaux_valalpha | 2026 | coverage_iou | 0.5408 | 0.5054 | 0.5767 |
| clueground_canonical | 13 | exact_union_iou | 0.5189 | 0.4823 | 0.5548 |
| clueground_canonical | 42 | exact_union_iou | 0.5355 | 0.5011 | 0.5712 |
| clueground_canonical | 2026 | exact_union_iou | 0.5177 | 0.4759 | 0.5600 |
| clueground_gate_always | 13 | exact_union_iou | 0.5385 | 0.4959 | 0.5765 |
| clueground_gate_always | 42 | exact_union_iou | 0.5355 | 0.5020 | 0.5711 |
| clueground_gate_always | 2026 | exact_union_iou | 0.5241 | 0.4851 | 0.5657 |
| clueground_hybrid_base | 13 | exact_union_iou | 0.5189 | 0.4816 | 0.5569 |
| clueground_hybrid_base | 42 | exact_union_iou | 0.5253 | 0.4899 | 0.5626 |
| clueground_hybrid_base | 2026 | exact_union_iou | 0.5177 | 0.4756 | 0.5615 |
| clueground_legacy_paper | 13 | exact_union_iou | 0.5412 | 0.5051 | 0.5782 |
| clueground_legacy_paper | 42 | exact_union_iou | 0.5360 | 0.5014 | 0.5716 |
| clueground_legacy_paper | 2026 | exact_union_iou | 0.5346 | 0.4986 | 0.5689 |
| medgrounder | 13 | exact_union_iou | 0.5349 | 0.5007 | 0.5707 |
| medgrounder | 42 | exact_union_iou | 0.5293 | 0.4917 | 0.5635 |
| medgrounder | 2026 | exact_union_iou | 0.5262 | 0.4896 | 0.5631 |
| reranker_aux_valalpha | 13 | exact_union_iou | 0.5455 | 0.5119 | 0.5812 |
| reranker_aux_valalpha | 42 | exact_union_iou | 0.5528 | 0.5179 | 0.5900 |
| reranker_aux_valalpha | 2026 | exact_union_iou | 0.5343 | 0.5006 | 0.5730 |
| reranker_noaux_pooledalpha | 13 | exact_union_iou | 0.5168 | 0.4777 | 0.5563 |
| reranker_noaux_pooledalpha | 42 | exact_union_iou | 0.5366 | 0.4998 | 0.5726 |
| reranker_noaux_pooledalpha | 2026 | exact_union_iou | 0.5433 | 0.5081 | 0.5785 |
| reranker_noaux_valalpha | 13 | exact_union_iou | 0.5491 | 0.5138 | 0.5844 |
| reranker_noaux_valalpha | 42 | exact_union_iou | 0.5408 | 0.5044 | 0.5764 |
| reranker_noaux_valalpha | 2026 | exact_union_iou | 0.5401 | 0.5061 | 0.5760 |
| clueground_canonical | 13 | set_f1_optimal_0_3 | 0.7617 | 0.7113 | 0.8123 |
| clueground_canonical | 42 | set_f1_optimal_0_3 | 0.7980 | 0.7495 | 0.8441 |
| clueground_canonical | 2026 | set_f1_optimal_0_3 | 0.7420 | 0.6828 | 0.7969 |
| clueground_gate_always | 13 | set_f1_optimal_0_3 | 0.7980 | 0.7380 | 0.8540 |
| clueground_gate_always | 42 | set_f1_optimal_0_3 | 0.7980 | 0.7503 | 0.8450 |
| clueground_gate_always | 2026 | set_f1_optimal_0_3 | 0.7617 | 0.7034 | 0.8223 |
| clueground_hybrid_base | 13 | set_f1_optimal_0_3 | 0.7617 | 0.7095 | 0.8119 |
| clueground_hybrid_base | 42 | set_f1_optimal_0_3 | 0.7844 | 0.7347 | 0.8318 |
| clueground_hybrid_base | 2026 | set_f1_optimal_0_3 | 0.7420 | 0.6796 | 0.8008 |
| clueground_legacy_paper | 13 | set_f1_optimal_0_3 | 0.7886 | 0.7364 | 0.8381 |
| clueground_legacy_paper | 42 | set_f1_optimal_0_3 | 0.7548 | 0.6995 | 0.8111 |
| clueground_legacy_paper | 2026 | set_f1_optimal_0_3 | 0.7745 | 0.7263 | 0.8214 |
| medgrounder | 13 | set_f1_optimal_0_3 | 0.8130 | 0.7560 | 0.8698 |
| medgrounder | 42 | set_f1_optimal_0_3 | 0.7908 | 0.7229 | 0.8513 |
| medgrounder | 2026 | set_f1_optimal_0_3 | 0.7983 | 0.7357 | 0.8575 |
| reranker_aux_valalpha | 13 | set_f1_optimal_0_3 | 0.7912 | 0.7439 | 0.8385 |
| reranker_aux_valalpha | 42 | set_f1_optimal_0_3 | 0.8170 | 0.7698 | 0.8623 |
| reranker_aux_valalpha | 2026 | set_f1_optimal_0_3 | 0.7588 | 0.7102 | 0.8120 |
| reranker_noaux_pooledalpha | 13 | set_f1_optimal_0_3 | 0.7548 | 0.6969 | 0.8154 |
| reranker_noaux_pooledalpha | 42 | set_f1_optimal_0_3 | 0.7817 | 0.7329 | 0.8318 |
| reranker_noaux_pooledalpha | 2026 | set_f1_optimal_0_3 | 0.7792 | 0.7255 | 0.8296 |
| reranker_noaux_valalpha | 13 | set_f1_optimal_0_3 | 0.8071 | 0.7569 | 0.8545 |
| reranker_noaux_valalpha | 42 | set_f1_optimal_0_3 | 0.7920 | 0.7417 | 0.8424 |
| reranker_noaux_valalpha | 2026 | set_f1_optimal_0_3 | 0.7627 | 0.7106 | 0.8172 |
| clueground_canonical | 13 | set_f1_optimal_0_5 | 0.6086 | 0.5443 | 0.6720 |
| clueground_canonical | 42 | set_f1_optimal_0_5 | 0.5950 | 0.5312 | 0.6606 |
| clueground_canonical | 2026 | set_f1_optimal_0_5 | 0.5958 | 0.5186 | 0.6690 |
| clueground_gate_always | 13 | set_f1_optimal_0_5 | 0.6086 | 0.5297 | 0.6777 |
| clueground_gate_always | 42 | set_f1_optimal_0_5 | 0.5950 | 0.5308 | 0.6596 |
| clueground_gate_always | 2026 | set_f1_optimal_0_5 | 0.5942 | 0.5216 | 0.6705 |
| clueground_hybrid_base | 13 | set_f1_optimal_0_5 | 0.6086 | 0.5452 | 0.6742 |
| clueground_hybrid_base | 42 | set_f1_optimal_0_5 | 0.5965 | 0.5322 | 0.6656 |
| clueground_hybrid_base | 2026 | set_f1_optimal_0_5 | 0.5958 | 0.5224 | 0.6725 |
| clueground_legacy_paper | 13 | set_f1_optimal_0_5 | 0.6106 | 0.5467 | 0.6749 |
| clueground_legacy_paper | 42 | set_f1_optimal_0_5 | 0.6314 | 0.5701 | 0.6952 |
| clueground_legacy_paper | 2026 | set_f1_optimal_0_5 | 0.6026 | 0.5322 | 0.6685 |
| medgrounder | 13 | set_f1_optimal_0_5 | 0.6238 | 0.5557 | 0.6906 |
| medgrounder | 42 | set_f1_optimal_0_5 | 0.5988 | 0.5260 | 0.6658 |
| medgrounder | 2026 | set_f1_optimal_0_5 | 0.5980 | 0.5302 | 0.6662 |
| reranker_aux_valalpha | 13 | set_f1_optimal_0_5 | 0.6435 | 0.5836 | 0.7042 |
| reranker_aux_valalpha | 42 | set_f1_optimal_0_5 | 0.6480 | 0.5799 | 0.7157 |
| reranker_aux_valalpha | 2026 | set_f1_optimal_0_5 | 0.6050 | 0.5439 | 0.6749 |
| reranker_noaux_pooledalpha | 13 | set_f1_optimal_0_5 | 0.5942 | 0.5288 | 0.6637 |
| reranker_noaux_pooledalpha | 42 | set_f1_optimal_0_5 | 0.6286 | 0.5564 | 0.6976 |
| reranker_noaux_pooledalpha | 2026 | set_f1_optimal_0_5 | 0.6126 | 0.5463 | 0.6781 |
| reranker_noaux_valalpha | 13 | set_f1_optimal_0_5 | 0.6352 | 0.5645 | 0.7039 |
| reranker_noaux_valalpha | 42 | set_f1_optimal_0_5 | 0.6298 | 0.5612 | 0.6950 |
| reranker_noaux_valalpha | 2026 | set_f1_optimal_0_5 | 0.6089 | 0.5412 | 0.6770 |

## Paired difference vs MedGrounder (same 220 groups, subject-cluster bootstrap)

| method | seed | metric | mean diff | CI low | CI high | p (two-sided) | groups better/worse |
|---|---:|---|---:|---:|---:|---:|---:|
| clueground_canonical | 13 | coverage_iou | -0.0188 | -0.0443 | +0.0059 | 0.137 | 119/98 |
| clueground_canonical | 42 | coverage_iou | +0.0021 | -0.0215 | +0.0249 | 0.863 | 115/103 |
| clueground_canonical | 2026 | coverage_iou | -0.0110 | -0.0412 | +0.0175 | 0.512 | 113/106 |
| clueground_canonical | 13 | exact_union_iou | -0.0160 | -0.0403 | +0.0088 | 0.211 | 118/99 |
| clueground_canonical | 42 | exact_union_iou | +0.0062 | -0.0178 | +0.0280 | 0.620 | 116/102 |
| clueground_canonical | 2026 | exact_union_iou | -0.0085 | -0.0379 | +0.0195 | 0.599 | 114/105 |
| clueground_canonical | 13 | set_f1_optimal_0_3 | -0.0514 | -0.1038 | +0.0010 | 0.057 | 16/30 |
| clueground_canonical | 42 | set_f1_optimal_0_3 | +0.0073 | -0.0465 | +0.0597 | 0.831 | 26/26 |
| clueground_canonical | 2026 | set_f1_optimal_0_3 | -0.0564 | -0.1117 | +0.0036 | 0.059 | 19/37 |
| clueground_canonical | 13 | set_f1_optimal_0_5 | -0.0152 | -0.0663 | +0.0342 | 0.551 | 26/29 |
| clueground_canonical | 42 | set_f1_optimal_0_5 | -0.0038 | -0.0596 | +0.0480 | 0.905 | 32/32 |
| clueground_canonical | 2026 | set_f1_optimal_0_5 | -0.0023 | -0.0653 | +0.0660 | 0.941 | 34/35 |
| clueground_gate_always | 13 | coverage_iou | +0.0016 | -0.0212 | +0.0246 | 0.879 | 124/93 |
| clueground_gate_always | 42 | coverage_iou | +0.0021 | -0.0215 | +0.0249 | 0.863 | 115/103 |
| clueground_gate_always | 2026 | coverage_iou | -0.0041 | -0.0293 | +0.0194 | 0.807 | 105/113 |
| clueground_gate_always | 13 | exact_union_iou | +0.0036 | -0.0192 | +0.0276 | 0.764 | 123/94 |
| clueground_gate_always | 42 | exact_union_iou | +0.0062 | -0.0178 | +0.0280 | 0.620 | 116/102 |
| clueground_gate_always | 2026 | exact_union_iou | -0.0021 | -0.0269 | +0.0211 | 0.933 | 104/114 |
| clueground_gate_always | 13 | set_f1_optimal_0_3 | -0.0150 | -0.0662 | +0.0361 | 0.587 | 22/28 |
| clueground_gate_always | 42 | set_f1_optimal_0_3 | +0.0073 | -0.0465 | +0.0597 | 0.831 | 26/26 |
| clueground_gate_always | 2026 | set_f1_optimal_0_3 | -0.0367 | -0.0909 | +0.0172 | 0.186 | 22/33 |
| clueground_gate_always | 13 | set_f1_optimal_0_5 | -0.0152 | -0.0691 | +0.0388 | 0.595 | 27/29 |
| clueground_gate_always | 42 | set_f1_optimal_0_5 | -0.0038 | -0.0596 | +0.0480 | 0.905 | 32/32 |
| clueground_gate_always | 2026 | set_f1_optimal_0_5 | -0.0038 | -0.0606 | +0.0553 | 0.869 | 32/33 |
| clueground_hybrid_base | 13 | coverage_iou | -0.0188 | -0.0443 | +0.0059 | 0.137 | 119/98 |
| clueground_hybrid_base | 42 | coverage_iou | -0.0082 | -0.0324 | +0.0147 | 0.486 | 109/109 |
| clueground_hybrid_base | 2026 | coverage_iou | -0.0110 | -0.0412 | +0.0175 | 0.512 | 113/106 |
| clueground_hybrid_base | 13 | exact_union_iou | -0.0160 | -0.0403 | +0.0088 | 0.211 | 118/99 |
| clueground_hybrid_base | 42 | exact_union_iou | -0.0040 | -0.0288 | +0.0192 | 0.701 | 111/107 |
| clueground_hybrid_base | 2026 | exact_union_iou | -0.0085 | -0.0379 | +0.0195 | 0.599 | 114/105 |
| clueground_hybrid_base | 13 | set_f1_optimal_0_3 | -0.0514 | -0.1038 | +0.0010 | 0.057 | 16/30 |
| clueground_hybrid_base | 42 | set_f1_optimal_0_3 | -0.0064 | -0.0692 | +0.0538 | 0.771 | 26/30 |
| clueground_hybrid_base | 2026 | set_f1_optimal_0_3 | -0.0564 | -0.1117 | +0.0036 | 0.059 | 19/37 |
| clueground_hybrid_base | 13 | set_f1_optimal_0_5 | -0.0152 | -0.0663 | +0.0342 | 0.551 | 26/29 |
| clueground_hybrid_base | 42 | set_f1_optimal_0_5 | -0.0023 | -0.0561 | +0.0479 | 0.937 | 32/31 |
| clueground_hybrid_base | 2026 | set_f1_optimal_0_5 | -0.0023 | -0.0653 | +0.0660 | 0.941 | 34/35 |
| clueground_legacy_paper | 13 | coverage_iou | +0.0044 | -0.0197 | +0.0273 | 0.699 | 112/107 |
| clueground_legacy_paper | 42 | coverage_iou | +0.0017 | -0.0240 | +0.0269 | 0.887 | 121/97 |
| clueground_legacy_paper | 2026 | coverage_iou | +0.0046 | -0.0206 | +0.0288 | 0.690 | 108/110 |
| clueground_legacy_paper | 13 | exact_union_iou | +0.0063 | -0.0167 | +0.0303 | 0.602 | 115/104 |
| clueground_legacy_paper | 42 | exact_union_iou | +0.0067 | -0.0192 | +0.0315 | 0.643 | 120/98 |
| clueground_legacy_paper | 2026 | exact_union_iou | +0.0084 | -0.0160 | +0.0320 | 0.497 | 109/109 |
| clueground_legacy_paper | 13 | set_f1_optimal_0_3 | -0.0244 | -0.0830 | +0.0288 | 0.379 | 21/30 |
| clueground_legacy_paper | 42 | set_f1_optimal_0_3 | -0.0359 | -0.1002 | +0.0295 | 0.256 | 23/36 |
| clueground_legacy_paper | 2026 | set_f1_optimal_0_3 | -0.0238 | -0.0795 | +0.0294 | 0.398 | 21/34 |
| clueground_legacy_paper | 13 | set_f1_optimal_0_5 | -0.0132 | -0.0684 | +0.0442 | 0.649 | 30/33 |
| clueground_legacy_paper | 42 | set_f1_optimal_0_5 | +0.0326 | -0.0244 | +0.0904 | 0.267 | 34/27 |
| clueground_legacy_paper | 2026 | set_f1_optimal_0_5 | +0.0045 | -0.0558 | +0.0631 | 0.882 | 30/31 |
| reranker_aux_valalpha | 13 | coverage_iou | +0.0053 | -0.0169 | +0.0266 | 0.629 | 123/94 |
| reranker_aux_valalpha | 42 | coverage_iou | +0.0147 | -0.0062 | +0.0360 | 0.169 | 120/97 |
| reranker_aux_valalpha | 2026 | coverage_iou | +0.0015 | -0.0246 | +0.0252 | 0.893 | 112/107 |
| reranker_aux_valalpha | 13 | exact_union_iou | +0.0106 | -0.0108 | +0.0316 | 0.334 | 124/93 |
| reranker_aux_valalpha | 42 | exact_union_iou | +0.0235 | +0.0025 | +0.0444 | 0.027 | 121/96 |
| reranker_aux_valalpha | 2026 | exact_union_iou | +0.0081 | -0.0165 | +0.0317 | 0.471 | 117/102 |
| reranker_aux_valalpha | 13 | set_f1_optimal_0_3 | -0.0218 | -0.0690 | +0.0269 | 0.369 | 16/25 |
| reranker_aux_valalpha | 42 | set_f1_optimal_0_3 | +0.0262 | -0.0313 | +0.0801 | 0.392 | 29/24 |
| reranker_aux_valalpha | 2026 | set_f1_optimal_0_3 | -0.0395 | -0.0890 | +0.0092 | 0.105 | 17/34 |
| reranker_aux_valalpha | 13 | set_f1_optimal_0_5 | +0.0197 | -0.0323 | +0.0731 | 0.463 | 31/24 |
| reranker_aux_valalpha | 42 | set_f1_optimal_0_5 | +0.0492 | -0.0066 | +0.1048 | 0.094 | 40/26 |
| reranker_aux_valalpha | 2026 | set_f1_optimal_0_5 | +0.0070 | -0.0518 | +0.0674 | 0.853 | 33/31 |
| reranker_noaux_pooledalpha | 13 | coverage_iou | -0.0248 | -0.0476 | -0.0013 | 0.036 | 108/108 |
| reranker_noaux_pooledalpha | 42 | coverage_iou | -0.0024 | -0.0259 | +0.0201 | 0.823 | 109/109 |
| reranker_noaux_pooledalpha | 2026 | coverage_iou | +0.0108 | -0.0128 | +0.0345 | 0.348 | 121/98 |
| reranker_noaux_pooledalpha | 13 | exact_union_iou | -0.0181 | -0.0414 | +0.0056 | 0.133 | 112/104 |
| reranker_noaux_pooledalpha | 42 | exact_union_iou | +0.0073 | -0.0172 | +0.0299 | 0.557 | 110/108 |
| reranker_noaux_pooledalpha | 2026 | exact_union_iou | +0.0171 | -0.0063 | +0.0404 | 0.142 | 124/95 |
| reranker_noaux_pooledalpha | 13 | set_f1_optimal_0_3 | -0.0582 | -0.1072 | -0.0085 | 0.022 | 16/33 |
| reranker_noaux_pooledalpha | 42 | set_f1_optimal_0_3 | -0.0091 | -0.0693 | +0.0474 | 0.704 | 26/29 |
| reranker_noaux_pooledalpha | 2026 | set_f1_optimal_0_3 | -0.0191 | -0.0688 | +0.0295 | 0.447 | 20/29 |
| reranker_noaux_pooledalpha | 13 | set_f1_optimal_0_5 | -0.0295 | -0.0863 | +0.0292 | 0.345 | 30/36 |
| reranker_noaux_pooledalpha | 42 | set_f1_optimal_0_5 | +0.0298 | -0.0223 | +0.0854 | 0.273 | 36/28 |
| reranker_noaux_pooledalpha | 2026 | set_f1_optimal_0_5 | +0.0145 | -0.0457 | +0.0771 | 0.652 | 38/34 |
| reranker_noaux_valalpha | 13 | coverage_iou | +0.0090 | -0.0126 | +0.0309 | 0.401 | 126/91 |
| reranker_noaux_valalpha | 42 | coverage_iou | +0.0048 | -0.0203 | +0.0300 | 0.739 | 117/101 |
| reranker_noaux_valalpha | 2026 | coverage_iou | +0.0110 | -0.0137 | +0.0352 | 0.396 | 123/96 |
| reranker_noaux_valalpha | 13 | exact_union_iou | +0.0142 | -0.0072 | +0.0367 | 0.197 | 126/91 |
| reranker_noaux_valalpha | 42 | exact_union_iou | +0.0115 | -0.0141 | +0.0374 | 0.401 | 117/101 |
| reranker_noaux_valalpha | 2026 | exact_union_iou | +0.0139 | -0.0105 | +0.0379 | 0.274 | 123/96 |
| reranker_noaux_valalpha | 13 | set_f1_optimal_0_3 | -0.0059 | -0.0531 | +0.0429 | 0.826 | 20/25 |
| reranker_noaux_valalpha | 42 | set_f1_optimal_0_3 | +0.0012 | -0.0612 | +0.0582 | 0.991 | 28/30 |
| reranker_noaux_valalpha | 2026 | set_f1_optimal_0_3 | -0.0356 | -0.0899 | +0.0155 | 0.183 | 19/33 |
| reranker_noaux_valalpha | 13 | set_f1_optimal_0_5 | +0.0114 | -0.0413 | +0.0619 | 0.659 | 33/27 |
| reranker_noaux_valalpha | 42 | set_f1_optimal_0_5 | +0.0311 | -0.0290 | +0.0915 | 0.315 | 39/29 |
| reranker_noaux_valalpha | 2026 | set_f1_optimal_0_5 | +0.0109 | -0.0467 | +0.0746 | 0.740 | 35/32 |

## Mean of per-seed paired differences

| method | metric | mean diff over seeds | min | max | seeds with CI excluding 0 |
|---|---|---:|---:|---:|---:|
| clueground_canonical | coverage_iou | -0.0092 | -0.0188 | +0.0021 | 0/3 |
| clueground_canonical | exact_union_iou | -0.0061 | -0.0160 | +0.0062 | 0/3 |
| clueground_canonical | set_f1_optimal_0_3 | -0.0335 | -0.0564 | +0.0073 | 0/3 |
| clueground_canonical | set_f1_optimal_0_5 | -0.0071 | -0.0152 | -0.0023 | 0/3 |
| clueground_gate_always | coverage_iou | -0.0001 | -0.0041 | +0.0021 | 0/3 |
| clueground_gate_always | exact_union_iou | +0.0026 | -0.0021 | +0.0062 | 0/3 |
| clueground_gate_always | set_f1_optimal_0_3 | -0.0148 | -0.0367 | +0.0073 | 0/3 |
| clueground_gate_always | set_f1_optimal_0_5 | -0.0076 | -0.0152 | -0.0038 | 0/3 |
| clueground_hybrid_base | coverage_iou | -0.0126 | -0.0188 | -0.0082 | 0/3 |
| clueground_hybrid_base | exact_union_iou | -0.0095 | -0.0160 | -0.0040 | 0/3 |
| clueground_hybrid_base | set_f1_optimal_0_3 | -0.0380 | -0.0564 | -0.0064 | 0/3 |
| clueground_hybrid_base | set_f1_optimal_0_5 | -0.0066 | -0.0152 | -0.0023 | 0/3 |
| clueground_legacy_paper | coverage_iou | +0.0036 | +0.0017 | +0.0046 | 0/3 |
| clueground_legacy_paper | exact_union_iou | +0.0071 | +0.0063 | +0.0084 | 0/3 |
| clueground_legacy_paper | set_f1_optimal_0_3 | -0.0280 | -0.0359 | -0.0238 | 0/3 |
| clueground_legacy_paper | set_f1_optimal_0_5 | +0.0080 | -0.0132 | +0.0326 | 0/3 |
| reranker_aux_valalpha | coverage_iou | +0.0072 | +0.0015 | +0.0147 | 0/3 |
| reranker_aux_valalpha | exact_union_iou | +0.0141 | +0.0081 | +0.0235 | 1/3 |
| reranker_aux_valalpha | set_f1_optimal_0_3 | -0.0117 | -0.0395 | +0.0262 | 0/3 |
| reranker_aux_valalpha | set_f1_optimal_0_5 | +0.0253 | +0.0070 | +0.0492 | 0/3 |
| reranker_noaux_pooledalpha | coverage_iou | -0.0055 | -0.0248 | +0.0108 | 1/3 |
| reranker_noaux_pooledalpha | exact_union_iou | +0.0021 | -0.0181 | +0.0171 | 0/3 |
| reranker_noaux_pooledalpha | set_f1_optimal_0_3 | -0.0288 | -0.0582 | -0.0091 | 1/3 |
| reranker_noaux_pooledalpha | set_f1_optimal_0_5 | +0.0049 | -0.0295 | +0.0298 | 0/3 |
| reranker_noaux_valalpha | coverage_iou | +0.0083 | +0.0048 | +0.0110 | 0/3 |
| reranker_noaux_valalpha | exact_union_iou | +0.0132 | +0.0115 | +0.0142 | 0/3 |
| reranker_noaux_valalpha | set_f1_optimal_0_3 | -0.0134 | -0.0356 | +0.0012 | 0/3 |
| reranker_noaux_valalpha | set_f1_optimal_0_5 | +0.0178 | +0.0109 | +0.0311 | 0/3 |

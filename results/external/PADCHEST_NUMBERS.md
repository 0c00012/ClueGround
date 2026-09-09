# PadChest-GR zero-shot numbers

## strict (194 rows, seeds [13, 42, 2026])
- clueground_reranker: coverage_iou 0.5356 +/- 0.0056; exact_union_iou 0.5385 +/- 0.0054; set_f1_optimal_0_3 0.7735 +/- 0.0117; set_f1_optimal_0_5 0.6380 +/- 0.0226
- hybrid_no_reranker: coverage_iou 0.4895 +/- 0.0139; exact_union_iou 0.4910 +/- 0.0145; set_f1_optimal_0_3 0.7002 +/- 0.0382; set_f1_optimal_0_5 0.5828 +/- 0.0169
- rad_dino_only: coverage_iou 0.4419 +/- 0.0167; exact_union_iou 0.4532 +/- 0.0161; set_f1_optimal_0_3 0.6850 +/- 0.0308; set_f1_optimal_0_5 0.4863 +/- 0.0394
- clueground_reranker_minus_hybrid_no_reranker seed 13: coverage_iou +0.0477 [+0.0247, +0.0727]; exact_union_iou +0.0480 [+0.0251, +0.0731]; set_f1_optimal_0_3 +0.0567 [+0.0176, +0.0980]; set_f1_optimal_0_5 +0.0687 [+0.0253, +0.1117]
- clueground_reranker_minus_hybrid_no_reranker seed 42: coverage_iou +0.0292 [+0.0082, +0.0532]; exact_union_iou +0.0314 [+0.0100, +0.0555]; set_f1_optimal_0_3 +0.0533 [+0.0071, +0.0979]; set_f1_optimal_0_5 +0.0361 [+0.0034, +0.0704]
- clueground_reranker_minus_hybrid_no_reranker seed 2026: coverage_iou +0.0616 [+0.0390, +0.0867]; exact_union_iou +0.0631 [+0.0405, +0.0882]; set_f1_optimal_0_3 +0.1100 [+0.0646, +0.1608]; set_f1_optimal_0_5 +0.0610 [+0.0283, +0.0996]
- clueground_reranker_minus_rad_dino_only seed 13: coverage_iou +0.1187 [+0.0890, +0.1482]; exact_union_iou +0.1097 [+0.0808, +0.1385]; set_f1_optimal_0_3 +0.1357 [+0.0753, +0.1958]; set_f1_optimal_0_5 +0.2208 [+0.1511, +0.2938]
- clueground_reranker_minus_rad_dino_only seed 42: coverage_iou +0.0775 [+0.0469, +0.1074]; exact_union_iou +0.0706 [+0.0404, +0.0999]; set_f1_optimal_0_3 +0.0550 [+0.0035, +0.1094]; set_f1_optimal_0_5 +0.1306 [+0.0555, +0.1964]
- clueground_reranker_minus_rad_dino_only seed 2026: coverage_iou +0.0851 [+0.0584, +0.1143]; exact_union_iou +0.0757 [+0.0487, +0.1045]; set_f1_optimal_0_3 +0.0747 [+0.0183, +0.1289]; set_f1_optimal_0_5 +0.1040 [+0.0431, +0.1702]

## extended (278 rows, seeds [13, 42, 2026])
- clueground_reranker: coverage_iou 0.4680 +/- 0.0031; exact_union_iou 0.4701 +/- 0.0039; set_f1_optimal_0_3 0.6986 +/- 0.0126; set_f1_optimal_0_5 0.5145 +/- 0.0069
- hybrid_no_reranker: coverage_iou 0.4299 +/- 0.0185; exact_union_iou 0.4301 +/- 0.0184; set_f1_optimal_0_3 0.6255 +/- 0.0305; set_f1_optimal_0_5 0.4682 +/- 0.0275
- rad_dino_only: coverage_iou 0.3853 +/- 0.0161; exact_union_iou 0.4051 +/- 0.0152; set_f1_optimal_0_3 0.5943 +/- 0.0256; set_f1_optimal_0_5 0.3773 +/- 0.0381
- clueground_reranker_minus_hybrid_no_reranker seed 13: coverage_iou +0.0485 [+0.0288, +0.0709]; exact_union_iou +0.0490 [+0.0295, +0.0711]; set_f1_optimal_0_3 +0.0881 [+0.0493, +0.1305]; set_f1_optimal_0_5 +0.0692 [+0.0325, +0.1106]
- clueground_reranker_minus_hybrid_no_reranker seed 42: coverage_iou +0.0195 [+0.0014, +0.0376]; exact_union_iou +0.0228 [+0.0050, +0.0404]; set_f1_optimal_0_3 +0.0492 [+0.0099, +0.0860]; set_f1_optimal_0_5 +0.0198 [-0.0119, +0.0525]
- clueground_reranker_minus_hybrid_no_reranker seed 2026: coverage_iou +0.0464 [+0.0279, +0.0655]; exact_union_iou +0.0482 [+0.0296, +0.0677]; set_f1_optimal_0_3 +0.0820 [+0.0447, +0.1212]; set_f1_optimal_0_5 +0.0501 [+0.0183, +0.0828]
- clueground_reranker_minus_rad_dino_only seed 13: coverage_iou +0.1018 [+0.0772, +0.1275]; exact_union_iou +0.0822 [+0.0587, +0.1068]; set_f1_optimal_0_3 +0.1362 [+0.0812, +0.1919]; set_f1_optimal_0_5 +0.1843 [+0.1273, +0.2448]
- clueground_reranker_minus_rad_dino_only seed 42: coverage_iou +0.0738 [+0.0488, +0.0987]; exact_union_iou +0.0586 [+0.0348, +0.0832]; set_f1_optimal_0_3 +0.0954 [+0.0450, +0.1474]; set_f1_optimal_0_5 +0.1213 [+0.0581, +0.1814]
- clueground_reranker_minus_rad_dino_only seed 2026: coverage_iou +0.0725 [+0.0494, +0.0952]; exact_union_iou +0.0542 [+0.0316, +0.0762]; set_f1_optimal_0_3 +0.0811 [+0.0283, +0.1335]; set_f1_optimal_0_5 +0.1061 [+0.0524, +0.1575]


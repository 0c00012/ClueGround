# direct-888 paired comparison (163 phrases, common evaluator, subject-cluster bootstrap)

| method | mean IoU | Hit@0.3 | Hit@0.5 |
|---|---:|---:|---:|
| clueground_base | 0.5486 +/- 0.0107 | 0.7730 | 0.6258 |
| clueground_reranker | 0.5539 +/- 0.0172 | 0.7996 | 0.6033 |
| medrpg | 0.5698 +/- 0.0069 | 0.8078 | 0.6442 |
| transvg | 0.5660 +/- 0.0078 | 0.8262 | 0.6483 |

| method | vs | seed | metric | mean diff | CI low | CI high | p |
|---|---|---:|---|---:|---:|---:|---:|
| clueground_base | medrpg | 13 | hit_0_3 | -0.0184 | -0.0855 | +0.0506 | 0.626 |
| clueground_base | medrpg | 42 | hit_0_3 | -0.0123 | -0.0783 | +0.0510 | 0.806 |
| clueground_base | medrpg | 2026 | hit_0_3 | -0.0736 | -0.1429 | -0.0062 | 0.045 |
| clueground_base | medrpg | 13 | hit_0_5 | -0.0123 | -0.0897 | +0.0651 | 0.788 |
| clueground_base | medrpg | 42 | hit_0_5 | -0.0307 | -0.1006 | +0.0420 | 0.452 |
| clueground_base | medrpg | 2026 | hit_0_5 | -0.0123 | -0.0886 | +0.0647 | 0.856 |
| clueground_base | medrpg | 13 | mean_iou_row | -0.0081 | -0.0421 | +0.0261 | 0.629 |
| clueground_base | medrpg | 42 | mean_iou_row | -0.0143 | -0.0510 | +0.0191 | 0.455 |
| clueground_base | medrpg | 2026 | mean_iou_row | -0.0412 | -0.0735 | -0.0084 | 0.011 |
| clueground_base | transvg | 13 | hit_0_3 | -0.0245 | -0.0800 | +0.0331 | 0.463 |
| clueground_base | transvg | 42 | hit_0_3 | -0.0613 | -0.1202 | +0.0000 | 0.063 |
| clueground_base | transvg | 2026 | hit_0_3 | -0.0736 | -0.1437 | -0.0062 | 0.043 |
| clueground_base | transvg | 13 | hit_0_5 | -0.0184 | -0.0915 | +0.0541 | 0.659 |
| clueground_base | transvg | 42 | hit_0_5 | -0.0552 | -0.1227 | +0.0135 | 0.148 |
| clueground_base | transvg | 2026 | hit_0_5 | +0.0061 | -0.0659 | +0.0854 | 0.872 |
| clueground_base | transvg | 13 | mean_iou_row | -0.0076 | -0.0373 | +0.0235 | 0.610 |
| clueground_base | transvg | 42 | mean_iou_row | -0.0220 | -0.0521 | +0.0079 | 0.155 |
| clueground_base | transvg | 2026 | mean_iou_row | -0.0226 | -0.0577 | +0.0152 | 0.244 |
| clueground_reranker | clueground_base | 13 | hit_0_3 | +0.0307 | -0.0122 | +0.0764 | 0.221 |
| clueground_reranker | clueground_base | 42 | hit_0_3 | -0.0061 | -0.0537 | +0.0374 | 0.919 |
| clueground_reranker | clueground_base | 2026 | hit_0_3 | +0.0552 | +0.0061 | +0.1081 | 0.045 |
| clueground_reranker | clueground_base | 13 | hit_0_5 | -0.0184 | -0.0629 | +0.0250 | 0.508 |
| clueground_reranker | clueground_base | 42 | hit_0_5 | -0.0307 | -0.0745 | +0.0132 | 0.233 |
| clueground_reranker | clueground_base | 2026 | hit_0_5 | -0.0184 | -0.0667 | +0.0248 | 0.526 |
| clueground_reranker | clueground_base | 13 | mean_iou_row | +0.0063 | -0.0137 | +0.0264 | 0.529 |
| clueground_reranker | clueground_base | 42 | mean_iou_row | -0.0185 | -0.0389 | +0.0041 | 0.096 |
| clueground_reranker | clueground_base | 2026 | mean_iou_row | +0.0279 | +0.0051 | +0.0506 | 0.025 |
| clueground_reranker | medrpg | 13 | hit_0_3 | +0.0123 | -0.0523 | +0.0800 | 0.778 |
| clueground_reranker | medrpg | 42 | hit_0_3 | -0.0184 | -0.0824 | +0.0476 | 0.656 |
| clueground_reranker | medrpg | 2026 | hit_0_3 | -0.0184 | -0.0714 | +0.0307 | 0.573 |
| clueground_reranker | medrpg | 13 | hit_0_5 | -0.0307 | -0.1098 | +0.0497 | 0.499 |
| clueground_reranker | medrpg | 42 | hit_0_5 | -0.0613 | -0.1338 | +0.0127 | 0.142 |
| clueground_reranker | medrpg | 2026 | hit_0_5 | -0.0307 | -0.1006 | +0.0397 | 0.450 |
| clueground_reranker | medrpg | 13 | mean_iou_row | -0.0018 | -0.0349 | +0.0323 | 0.881 |
| clueground_reranker | medrpg | 42 | mean_iou_row | -0.0328 | -0.0664 | +0.0015 | 0.058 |
| clueground_reranker | medrpg | 2026 | mean_iou_row | -0.0133 | -0.0403 | +0.0139 | 0.355 |
| clueground_reranker | transvg | 13 | hit_0_3 | +0.0061 | -0.0536 | +0.0667 | 0.895 |
| clueground_reranker | transvg | 42 | hit_0_3 | -0.0675 | -0.1278 | -0.0065 | 0.033 |
| clueground_reranker | transvg | 2026 | hit_0_3 | -0.0184 | -0.0765 | +0.0375 | 0.602 |
| clueground_reranker | transvg | 13 | hit_0_5 | -0.0368 | -0.1131 | +0.0377 | 0.392 |
| clueground_reranker | transvg | 42 | hit_0_5 | -0.0859 | -0.1512 | -0.0140 | 0.019 |
| clueground_reranker | transvg | 2026 | hit_0_5 | -0.0123 | -0.0765 | +0.0588 | 0.837 |
| clueground_reranker | transvg | 13 | mean_iou_row | -0.0013 | -0.0303 | +0.0296 | 0.920 |
| clueground_reranker | transvg | 42 | mean_iou_row | -0.0405 | -0.0687 | -0.0112 | 0.003 |
| clueground_reranker | transvg | 2026 | mean_iou_row | +0.0054 | -0.0246 | +0.0367 | 0.683 |

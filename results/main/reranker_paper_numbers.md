# Paper numbers (sealed)

1444 ClueGround: C-IoU 0.5419 +/- 0.0034; U-IoU 0.5433 +/- 0.0050; F1@.3 0.7873; F1@.5 0.6246; per-seed coverage [0.5457197029594641, 0.5392932499395547, 0.5407830459868764]; alpha [3.0, 2.0, 4.0]; scorer ['hgb', 'hgb', 'hgb']
888 ClueGround: mIoU 0.5539 +/- 0.0172; Hit@.3 0.7996; Hit@.5 0.6033; per-seed [0.5631343239102995, 0.5339642592337108, 0.5644802150002637]; alpha [1.5, 6.0, 4.0]; scorer ['extra', 'hgb', 'hgb']

## 1444 ablations (new path)
- query:full_query: C 0.5419 +/- 0.0034, U 0.5433, F1@.3 0.7873, F1@.5 0.6246
- query:finding_only: C 0.4959 +/- 0.0187, U 0.4971, F1@.3 0.7037, F1@.5 0.5663
- query:without_f: C 0.5330 +/- 0.0066, U 0.5353, F1@.3 0.7734, F1@.5 0.6166
- query:without_l: C 0.5262 +/- 0.0049, U 0.5286, F1@.3 0.7642, F1@.5 0.6067
- query:without_v: C 0.5388 +/- 0.0026, U 0.5405, F1@.3 0.7822, F1@.5 0.6187
- query:without_z: C 0.5393 +/- 0.0016, U 0.5405, F1@.3 0.7865, F1@.5 0.6272
- query:without_u: C 0.5414 +/- 0.0004, U 0.5426, F1@.3 0.7883, F1@.5 0.6216
- query:without_m_lexical: C 0.5414 +/- 0.0035, U 0.5436, F1@.3 0.7878, F1@.5 0.6236
- query:without_vq: C 0.5336 +/- 0.0048, U 0.5343, F1@.3 0.7719, F1@.5 0.6133
- component:rad_dino_only: C 0.4066 +/- 0.0075, U 0.4484, F1@.3 0.6376, F1@.5 0.3783
- component:yolo_only: C 0.5229 +/- 0.0149, U 0.5232, F1@.3 0.7625, F1@.5 0.6148
- component:full: C 0.5419 +/- 0.0034, U 0.5433, F1@.3 0.7873, F1@.5 0.6246
- component:no_reranker: C 0.5210 +/- 0.0046, U 0.5206, F1@.3 0.7627, F1@.5 0.6003
## 888 ablations (new path)
- query:full_query: mIoU 0.5539 +/- 0.0172, Hit@.3 0.7996, Hit@.5 0.6033
- query:finding_only: mIoU 0.4907 +/- 0.0301, Hit@.3 0.6994, Hit@.5 0.5460
- query:without_f: mIoU 0.5392 +/- 0.0113, Hit@.3 0.7832, Hit@.5 0.5971
- query:without_l: mIoU 0.5401 +/- 0.0205, Hit@.3 0.7751, Hit@.5 0.5971
- query:without_v: mIoU 0.5471 +/- 0.0123, Hit@.3 0.7894, Hit@.5 0.6012
- query:without_z: mIoU 0.5510 +/- 0.0145, Hit@.3 0.7955, Hit@.5 0.6012
- query:without_u: mIoU 0.5532 +/- 0.0168, Hit@.3 0.7996, Hit@.5 0.6012
- query:without_m_lexical: mIoU 0.5539 +/- 0.0172, Hit@.3 0.7996, Hit@.5 0.6033
- query:without_vq: mIoU 0.5415 +/- 0.0098, Hit@.3 0.7751, Hit@.5 0.5951
- component:rad_dino_only: mIoU 0.4717 +/- 0.0016, Hit@.3 0.7382, Hit@.5 0.4867
- component:yolo_only: mIoU 0.5391 +/- 0.0063, Hit@.3 0.7587, Hit@.5 0.6217
- component:full: mIoU 0.5539 +/- 0.0172, Hit@.3 0.7996, Hit@.5 0.6033
- component:no_reranker: mIoU 0.5486 +/- 0.0107, Hit@.3 0.7730, Hit@.5 0.6258
## robustness 1444
- no_retune: C 0.5338 +/- 0.0039, F1@.5 0.6108
- multi_alpha_zero: C 0.5388 +/- 0.0037, F1@.5 0.6240
- pooled_alpha: C 0.5282 +/- 0.0147, F1@.5 0.6118
- aux_features: C 0.5408 +/- 0.0090, F1@.5 0.6322
## paired vs MedGrounder (1444)
- coverage_iou: seed 13 +0.0090 [-0.0126, +0.0309]; seed 42 +0.0048 [-0.0203, +0.0300]; seed 2026 +0.0110 [-0.0137, +0.0352]
- exact_union_iou: seed 13 +0.0142 [-0.0072, +0.0367]; seed 42 +0.0115 [-0.0141, +0.0374]; seed 2026 +0.0139 [-0.0105, +0.0379]
- set_f1_optimal_0_3: seed 13 -0.0059 [-0.0531, +0.0429]; seed 42 +0.0012 [-0.0612, +0.0582]; seed 2026 -0.0356 [-0.0899, +0.0155]
- set_f1_optimal_0_5: seed 13 +0.0114 [-0.0413, +0.0619]; seed 42 +0.0311 [-0.0290, +0.0915]; seed 2026 +0.0109 [-0.0467, +0.0746]

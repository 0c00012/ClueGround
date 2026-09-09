# ClueGround zero-shot on PadChest-GR (extended mapping, 278 rows, seeds [13, 42, 2026])

| method | C-IoU | U-IoU | F1@.3 | F1@.5 |
|---|---:|---:|---:|---:|
| clueground_reranker | 0.4680 +/- 0.0031 | 0.4701 +/- 0.0039 | 0.6986 +/- 0.0126 | 0.5145 +/- 0.0069 |
| hybrid_no_reranker | 0.4299 +/- 0.0185 | 0.4301 +/- 0.0184 | 0.6255 +/- 0.0305 | 0.4682 +/- 0.0275 |
| rad_dino_only | 0.3853 +/- 0.0161 | 0.4051 +/- 0.0152 | 0.5943 +/- 0.0256 | 0.3773 +/- 0.0381 |

## per finding / reference count (clueground_reranker, three-seed mean)

| group | name | n | C-IoU | U-IoU | F1@.3 | F1@.5 |
|---|---|---:|---:|---:|---:|---:|
| per_finding | Atelectasis | 45 | 0.2581 | 0.2602 | 0.4123 | 0.1704 |
| per_finding | Cardiomegaly | 99 | 0.7322 | 0.7330 | 0.9899 | 0.9529 |
| per_finding | Consolidation | 4 | 0.1827 | 0.1827 | 0.2500 | 0.0833 |
| per_finding | Lung Opacity | 85 | 0.3083 | 0.3083 | 0.5193 | 0.2267 |
| per_finding | Pleural Effusion | 41 | 0.4215 | 0.4278 | 0.7358 | 0.4661 |
| per_finding | Pneumothorax | 4 | 0.4488 | 0.4817 | 0.5833 | 0.5833 |
| per_reference_count | multi (2+ boxes) | 43 | 0.3164 | 0.3404 | 0.5318 | 0.2904 |
| per_reference_count | single (1 box) | 235 | 0.4958 | 0.4938 | 0.7291 | 0.5556 |

scorer positive controls: seed 13 max|diff|=0.0e+00 PASS; seed 42 max|diff|=0.0e+00 PASS; seed 2026 max|diff|=0.0e+00 PASS

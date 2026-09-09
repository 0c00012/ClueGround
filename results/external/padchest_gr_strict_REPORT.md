# ClueGround zero-shot on PadChest-GR (strict mapping, 194 rows, seeds [13, 42, 2026])

| method | C-IoU | U-IoU | F1@.3 | F1@.5 |
|---|---:|---:|---:|---:|
| clueground_reranker | 0.5356 +/- 0.0056 | 0.5385 +/- 0.0054 | 0.7735 +/- 0.0117 | 0.6380 +/- 0.0226 |
| hybrid_no_reranker | 0.4895 +/- 0.0139 | 0.4910 +/- 0.0145 | 0.7002 +/- 0.0382 | 0.5828 +/- 0.0169 |
| rad_dino_only | 0.4419 +/- 0.0167 | 0.4532 +/- 0.0161 | 0.6850 +/- 0.0308 | 0.4863 +/- 0.0394 |

## per finding / reference count (clueground_reranker, three-seed mean)

| group | name | n | C-IoU | U-IoU | F1@.3 | F1@.5 |
|---|---|---:|---:|---:|---:|---:|
| per_finding | Atelectasis | 46 | 0.2525 | 0.2545 | 0.4034 | 0.1667 |
| per_finding | Cardiomegaly | 99 | 0.7322 | 0.7330 | 0.9899 | 0.9529 |
| per_finding | Consolidation | 4 | 0.1827 | 0.1827 | 0.2500 | 0.0833 |
| per_finding | Pleural Effusion | 41 | 0.4215 | 0.4278 | 0.7358 | 0.4661 |
| per_finding | Pneumothorax | 4 | 0.4488 | 0.4817 | 0.5833 | 0.5833 |
| per_reference_count | multi (2+ boxes) | 19 | 0.3363 | 0.3657 | 0.5643 | 0.3392 |
| per_reference_count | single (1 box) | 175 | 0.5573 | 0.5573 | 0.7962 | 0.6705 |

scorer positive controls: seed 13 max|diff|=0.0e+00 PASS; seed 42 max|diff|=0.0e+00 PASS; seed 2026 max|diff|=0.0e+00 PASS

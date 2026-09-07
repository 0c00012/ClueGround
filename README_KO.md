# ClueGround

이 저장소는 **Context-Guided YOLO-RAD-DINO Fusion for Single- and Multi-Region Chest X-Ray Phrase Grounding** 논문의 연구 코드와 비식별 집계 결과를 공개하기 위한 배포본입니다.

ClueGround는 finding-conditioned YOLO 네 개의 후보, frozen RAD-DINO, phrase-conditioned localization head, validation 기반 후보 결합, rule-context 다중 박스 decoder를 사용합니다.

## 주 결과

| 평가 세트 | 지표 | ClueGround |
|---|---|---:|
| MS-CXR-888, 163개 single-region phrase | Mean IoU | 0.5486 +/- 0.0107 |
| MS-CXR-888 | Hit@0.3 / Hit@0.5 | 0.7730 / 0.6258 |
| MS-CXR-1444, 220 phrase groups / 280 boxes | Coverage IoU | 0.5373 +/- 0.0035 |
| MS-CXR-1444 | Exact union IoU | 0.5373 +/- 0.0035 |
| MS-CXR-1444 | SetF1@0.3 / SetF1@0.5 | 0.7727 / 0.6148 |

888 결과는 upstream 전체를 seeds 13/42/2026으로 독립 실행한 평균입니다. 논문의 1444 `0.5373`은 legacy 814/125 train/validation split, 고정 legacy candidate table, finding-conditioned downstream gate를 사용합니다. 따라서 1444 결과를 upstream 전체가 seed별로 재생성된 full-pipeline 3-seed 결과라고 표현하면 안 됩니다. 자세한 내용은 [알려진 제한](docs/KNOWN_LIMITATIONS.md)과 [결과 계보](docs/PAPER_RESULT_LINEAGE.md)를 보십시오.

MS-CXR/MIMIC-CXR 원본, 환자 식별자, checkpoint, patch-token cache, per-example prediction은 데이터 이용약관과 배포 크기 때문에 포함하지 않았습니다. 공개본에는 코드, split fingerprint, 집계 지표만 들어 있습니다.

설치와 실행은 [영문 README](README.md)와 [재현 문서](docs/REPRODUCIBILITY.md)에 정리했습니다.

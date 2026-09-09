# ClueGround

이 저장소는 **Context-Guided YOLO-RAD-DINO Fusion for Single- and Multi-Region Chest X-Ray Phrase Grounding** 논문의 연구 코드와 비식별 집계 결과를 공개하기 위한 배포본입니다.

ClueGround는 finding-conditioned YOLO 네 개의 후보, frozen RAD-DINO, phrase-conditioned localization head, 학습 split에서 훈련한 학습형 후보 re-ranker, validation 기반 후보 결합, rule-context 다중 박스 decoder를 사용합니다.

## 주 결과 (수정 원고, 학습형 re-ranker)

| 평가 세트 | 지표 | ClueGround |
|---|---|---:|
| MS-CXR-888, 163개 single-region phrase | Mean IoU | 0.5539 +/- 0.0172 |
| MS-CXR-888 | Hit@0.3 / Hit@0.5 | 0.7996 / 0.6033 |
| MS-CXR-1444, 220 phrase groups / 280 boxes | Coverage IoU | 0.5419 +/- 0.0034 |
| MS-CXR-1444 | Exact union IoU | 0.5433 +/- 0.0050 |
| MS-CXR-1444 | SetF1@0.3 / SetF1@0.5 | 0.7873 / 0.6246 |

두 결과 모두 canonical split(direct-888: 638/87/163 phrase, MS-CXR-1444: 813/124/220 phrase group)에서 학습되는 모든 구성 요소를 seeds 13/42/2026으로 독립 실행한 평균입니다. ClueGround는 세 seed 모두에서 로컬 재학습 MedGrounder보다 Coverage IoU, Exact Union IoU, SetF1@0.5가 높지만, seed별 paired 95% patient-cluster bootstrap 구간은 0을 포함하고 SetF1@0.3은 MedGrounder가 더 높습니다. MS-CXR-888에서는 MedRPG와 TransVG가 여전히 더 높습니다. 자세한 내용은 [알려진 제한](docs/KNOWN_LIMITATIONS.md), [결과 계보](docs/PAPER_RESULT_LINEAGE.md), `results/comparison/`을 보십시오.

제출 원고의 값(888 mean IoU 0.5486 +/- 0.0107, 1444 Coverage 0.5373 +/- 0.0035; legacy 814/125 split과 고정 legacy candidate table 사용)은 기록을 위해 `results/main/`에 남겨 두었으며 위 결과로 대체되었습니다.

MS-CXR/MIMIC-CXR 원본, 환자 식별자, checkpoint, patch-token cache, per-example prediction은 데이터 이용약관과 배포 크기 때문에 포함하지 않았습니다. 공개본에는 코드, split fingerprint, 집계 지표만 들어 있습니다.

설치와 실행은 [영문 README](README.md)와 [재현 문서](docs/REPRODUCIBILITY.md)에 정리했습니다.

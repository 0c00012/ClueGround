# Manuscript and release audit

Audit dates: 2026-09-07 (submitted manuscript), 2026-09-09 (revision with the learned re-ranker).

## Source documents

| Item | SHA-256 | Role |
|---|---|---|
| Current manuscript `Context_Guided_YOLO__RAD_DINO_Fusion__4_.pdf` | `EF6A401C1D5D6103DA391C0ADA1D7CEFBC6E495D54601A06084E746464160DC9` | Submitted wording and table numbering |
| August handoff `current_paper.pdf` | `6D42F5ADF6012C3E0BB43C44751A460F6237F7BDC7666D5CA00E0F246A3A671B` | Historical code/artifact audit source |

The two PDFs are not identical. The current manuscript is the source of truth for prose and table numbering; the handoff remains the source for artifact provenance.

## Verification summary (2026-09-07)

- The August handoff matched 241/241 checked manuscript metric cells and resolved 424/424 recorded code, data, and artifact paths.
- Seeds 13, 42, and 2026 were confirmed for the reported local aggregate tables.
- The internal subject/study/DICOM overlap audits passed with zero train/evaluation overlap.
- The current manuscript had no adequate `Code availability` section and described MS-CXR too loosely as publicly available.
- This repository supplies the missing public code location and a credentialed-access data statement.

## Revision audit (2026-09-09)

The revised manuscript replaces the hand-weighted candidate score with a learned candidate re-ranker and reports canonical-split, fully per-seed results. Every number in the revised text and tables was transcribed programmatically from `results/main/reranker_paper_numbers.json`; the LaTeX edit script asserts each replaced sentence exists exactly once.

- Table 1 / Supplementary Table S1 ClueGround rows, Table 2 (with a new "no re-ranker" row), Tables 3 and 4, abstract, results, discussion and conclusion numbers: match `results/main/reranker_paper_numbers.json` and `results/ablations/reranker_component_query_*.csv`.
- Baseline rows are unchanged from the submitted manuscript (`results/main/controlled_*.csv`); no baseline was recomputed.
- Table 5 (end-to-end context ablation) is unchanged; it uses a separately trained downstream path that the re-ranker does not touch.
- Figure 2 was regenerated from the seed-42 re-ranker predictions with the same selection rule and unchanged baseline predictions; two of the four submitted cases remained, two were replaced (`scripts/build_reranker_qualitative_figure_v1.py` in the research tree; it depends on the paper-drafts figure library and is not part of this release).
- New Supplementary Tables S4 (paired differences vs MedGrounder) and S5 (re-ranker robustness) come from `results/comparison/paired_difference_ci_1444.csv` and `results/robustness/`.
- Methods now describe the re-ranker features, scorer selection, `alpha` selection and the training-side distribution shift; the "multiplicity $m$" query block and "count head" wording that did not correspond to code were corrected.
- Data and Code availability statements below were inserted.

## Required manuscript statements

**Data availability**

The MS-CXR dataset (version 1.1.0) analyzed in this study is available through PhysioNet under credentialed access (https://doi.org/10.13026/9g2z-jg61). Access requires completion of the PhysioNet credentialing process and acceptance of the applicable data use agreement. No protected clinical images or annotations are redistributed with the source code.

**Code availability**

The source code, evaluation scripts, aggregate results, and reproducibility documentation supporting this study are available at https://github.com/kimsi1854/ClueGround. The MS-CXR and MIMIC-CXR data are not redistributed and must be obtained independently through PhysioNet under credentialed access and the applicable data use agreement.

## Submission caveat

The historical 1444 value `0.5373 +/- 0.0035` no longer appears in the revised manuscript. If it is ever reinstated, the Methods or Results must state that it comes from the historical 814/125 split and a fixed legacy candidate table with downstream three-seed gating. The disclosures required for the re-ranker path are listed in `docs/KNOWN_LIMITATIONS.md`.

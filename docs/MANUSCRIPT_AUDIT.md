# Manuscript and release audit

Audit date: 2026-09-07

## Source documents

| Item | SHA-256 | Role |
|---|---|---|
| Current manuscript `Context_Guided_YOLO__RAD_DINO_Fusion__4_.pdf` | `EF6A401C1D5D6103DA391C0ADA1D7CEFBC6E495D54601A06084E746464160DC9` | Current wording and table numbering |
| August handoff `current_paper.pdf` | `6D42F5ADF6012C3E0BB43C44751A460F6237F7BDC7666D5CA00E0F246A3A671B` | Historical code/artifact audit source |

The two PDFs are not identical. The current manuscript is the source of truth for prose and table numbering; the handoff remains the source for artifact provenance.

## Verification summary

- The August handoff matched 241/241 checked manuscript metric cells and resolved 424/424 recorded code, data, and artifact paths.
- Seeds 13, 42, and 2026 were confirmed for the reported local aggregate tables.
- The internal subject/study/DICOM overlap audits passed with zero train/evaluation overlap.
- The current manuscript has no adequate `Code availability` section and describes MS-CXR too loosely as publicly available.
- This repository supplies the missing public code location and a credentialed-access data statement.

## Required manuscript statements

**Data availability**

The MS-CXR dataset (version 1.1.0) analyzed in this study is available through PhysioNet under credentialed access (https://doi.org/10.13026/9g2z-jg61). Access requires completion of the PhysioNet credentialing process and acceptance of the applicable data use agreement. No protected clinical images or annotations are redistributed with the source code.

**Code availability**

The source code, evaluation scripts, aggregate results, and reproducibility documentation supporting this study are available at https://github.com/kimsi1854/ClueGround. The MS-CXR and MIMIC-CXR data are not redistributed and must be obtained independently through PhysioNet under credentialed access and the applicable data use agreement.

## Submission caveat

If the manuscript keeps the 1444 ClueGround value of `0.5373 +/- 0.0035`, the Methods or Results must accurately state that it comes from the historical 814/125 split and fixed legacy candidate table with downstream three-seed gating. Calling it a fully regenerated upstream three-seed result would conflict with the artifact record.

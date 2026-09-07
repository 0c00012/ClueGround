# Data access

This release does not redistribute protected clinical data.

## Required datasets

- MS-CXR version 1.1.0
- MIMIC-CXR-JPG images linked by the MS-CXR annotations

Both resources must be obtained from PhysioNet by an appropriately credentialed user under the applicable data use agreement. Set `MS_CXR_DATA_ROOT` and `MIMIC_CXR_JPG_ROOT` after arranging the files locally.

## Public split information

The repository publishes only row/group counts and SHA-256 fingerprints in `docs/SPLIT_FINGERPRINTS.csv`. It does not publish subject IDs, study IDs, DICOM IDs, raw phrases, bounding boxes, or image paths.

The internal overlap audit used subject, study, and DICOM identifiers and found zero overlap between train and evaluation partitions for the reported local comparisons. That identifier-level audit cannot be redistributed here.

## Suggested manuscript wording

**Data availability.** The MS-CXR dataset (version 1.1.0) analyzed in this study is available through PhysioNet (https://doi.org/10.13026/9g2z-jg61). Access is credentialed and requires acceptance of the applicable data use agreement. No protected clinical images or annotations are redistributed with the source code.

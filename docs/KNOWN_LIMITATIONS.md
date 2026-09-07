# Known limitations and disclosure

## MS-CXR-1444 main number

The paper's ClueGround value of `0.5373 +/- 0.0035` was produced on the historical `814/125/220` split. It uses a fixed legacy candidate table, a no-SigLIP score configuration, and a finding-conditioned downstream gate across seeds 13, 42, and 2026.

This result is a three-run downstream aggregate, but it is not a full upstream three-seed reconstruction. The training candidate table also lacks the `score_head` field available in the validation/evaluation candidate tables. These facts should be disclosed if the value remains in the manuscript.

## Current manuscript versus handoff PDF

The current manuscript PDF differs from the PDF archived in the August handoff. Table numbering and prose should therefore be checked against the current manuscript, while the handoff remains the source for code/artifact provenance.

## Dataset restrictions

Protected images, annotations, identifier-level split manifests, per-example predictions, and checkpoints are omitted. Exact reruns require independent PhysioNet access and regeneration of upstream artifacts.

## Baseline scope

Several baselines are controlled local adaptations rather than exact official MS-CXR recipes. `docs/BASELINES.md` labels these cases explicitly.

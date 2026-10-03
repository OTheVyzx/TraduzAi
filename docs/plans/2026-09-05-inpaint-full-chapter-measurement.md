# Full Chapter24 mask/inpaint measurement

Extend the approved isolated experiment to all 18 canonical pages. Measure a real outer clock for each configuration, including recomputation of evidence/style/layout capture before mask creation and the complete replay. OCR, translation, typesetting, repair-after-render and terminal publication remain excluded. Never present this as full pipeline E2E.

Control: prepare all pages sequentially, then original mask/inpaint sequentially. Candidate: prepare fresh inputs in up to four isolated page subprocesses, then ROI residual verification with four CPU workers and one GPU engine. Preparation is recomputed, not reused from the control. This adds explicit page-parallel preparation to the already-approved parallel tests; verify frozen input equivalence and output decisions before accepting a gain.

Preserve sources, dirty production files and every prior run. New full-chapter harness only. Each page capture gets a 300-second timeout; overall profiler 7200 seconds. Record partial progress, child errors, discovered/prepared/skipped page counts, fixture count, model calls and exact per-owner comparisons. Page 18 with no historical translation attempts is accounted for, not turned into an invalid empty translation contract.

Profiler outer wall includes imports and all subprocess startup/I/O; inner parts are preparation wall, replay wall and model/mask/residual accumulated times. Mandatory capture uses compressed fixture snapshots and intentional stop-before-render; disclose that instrumentation overhead and non-production orchestration. Compare current measured totals, not historical 191.53 seconds or a seven-region extrapolation. All failures remain in the denominator/report.

# Owner three probes Implementation Plan

> Execute locally, without agents, commits or production integration.

**Goal:** Test the three user-approved hypotheses against the frozen 18-page owner fixture.
**Architecture:** Existing 3 GPU / 6 CPU pool harness. Experimental wrappers only, canonical line margins before recognition on source and owner, per-page exact detector memoization, and cProfile around CPU layout. Preserve downstream classification and compare complete serialized outputs.
**Tech Stack:** Python, NumPy, Paddle 2.x, cProfile, existing resource profiler.

## Tasks

1. Create `artifacts/e2e/container_parallel_20260905/test_three_probes.py`; fail first for missing canonicalization and detector memo helpers. Check identical content with different pure-white margins, blank input, mutation isolation, real fallback and exception handling.
2. Implement `three_probes.py`: canonicalize only line crops with removable exact-white margins; retain 8 px white margin and do not resize. Shared policy in source and owner. Cache whole batches, not independent lines. Implement exact detector result reuse within the same page, seeded by prior source capture. Keep non-identical input physical; record source versus owner hits.
3. Create `three_probe_runner.py` with fresh named directories and original 3 GPU/6 CPU topology. Profile CPU review/layout using cProfile. No change to production entry points or final proof.
4. Run pytest to green. Execute canonical source capture, unmodified control, canonical owner, detect-once owner, and CPU-layout profile sequentially through profiler with 600-second timeout. Preserve every failed run.
5. Compare all 18 full serialized payloads against `owner_parallel_w1`, record first semantic differences, errors, physical/cache counters, setup/stage/external timing. cProfile overhead disqualifies the layout-profile run as a speed benchmark; exclusive and cumulative times must stay separate.
6. Produce a report with measured scope, timing and quality limitations. Do not promote any variant merely from exit 0 or from a graph-only match.

No implicit commit/worktree operation: existing dirty checkout and old artifacts are preserved. Canonical line margins are a bounded first test, not full page-to-crop geometry projection. Detection-once is restricted to exact input pixels/parameters; it is not removal of detection from arbitrary multiline balloons.

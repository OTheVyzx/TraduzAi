# Optimized E2E timing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run Chapter 24 with the previously approved experimental optimizations and a verifiable timeline, without presenting fixture replays as E2E.

**Architecture:** Preserve current sources and historical runs. Build experimental adapters and timing utilities in a new directory, referencing approved candidates explicitly. Keep canonical QA/performance/resource artifacts; supplementary timing is diagnostic only, never proof or export authority.

**Tech Stack:** Python 3.12, contextvars, monotonic nanosecond clocks, per-process JSONL journals, pytest, existing external resource profiler.

---

## Checkpoint 1: timing foundation

Files: `artifacts/e2e/optimized_e2e_20260906/span_timing.py`, `test_span_timing.py`.

1. Write failing deterministic tests for nested intervals, concurrent intervals, idle gaps, exception propagation, and partial journals.
2. Implement thread/context-local spans with process/thread identifiers and parent IDs; preserve exceptions and record only exception class, not user text.
3. Calculate interval union rather than summing concurrent tasks. Report leaf-local wall attribution, overlapping labels, uninstrumented gaps, and accumulated invocation time separately. A parent residual is not magically identified as a specific function.
4. Run focused tests. Measure instrumentation overhead before accepting real performance numbers.

## Checkpoint 2: live integration inventory and adapters

Files: new `optimization_manifest.json`, experimental launcher and adapters in the same directory. Do not overwrite `container_parallel_20260905` candidates or historical outputs.

1. Inventory each approved candidate and its verified configuration, source hash, fixture assumptions and live integration status. Keep C/D OFF and superseded compressed component cache OFF.
2. Adapt parallel detector/CPU searches, discovery/OCR workers and immutable hash reuse to live page data, preserving ordering and owner contracts. No cached historical OCR observations.
3. Connect Google batches to the existing typed translation attempts/bindings, retaining rejects and validation. Google-only; no Ollama. Never use saved translations as live results.
4. Connect font bounds/render/normalization reuse, hoisting, masks/neighbors/colors/samples, persistent preparation workers, Sweep A/Bbox B/Binary E and measured cleanup workers. Respect immutable input keys and process-local mutable wrappers.
5. Test each boundary and combined output equivalence. A benchmark worker calling capture-only code is not a live adapter. Do not start a purported all-optimized run while required adapters remain unavailable.

## Checkpoint 3: timing coverage

Files: new explicit hook registry and registry tests in the experiment directory; reuse `_PipelineTiming` output as canonical aggregate.

Instrument startup/imports/model construction and warmup; archive extraction/decode/framing; detector and auxiliary searches; discovery/filtering/dedupe; OCR preparation/detection/recognition/recovery/evidence; translation preparation/network/validation/binding; owner glifs/protection/style/fonts/geometry; mask build; inpaint queue/transfer/engine/residual validation; final render/layout; composition; terminal OCR/proof; QA/gate; hashing/serialization/image writes; pool startup/IPC/queue waits/finalization.

Record page/owner identity without text/image payloads. Record cache hits/misses and actual workers. Mark uncalled hooks, missing boundaries and unknown residuals explicitly. CPU wall envelopes around CUDA calls are not kernel time; use existing synchronizations or optional CUDA events without introducing synchronization in the timed path. External GPU memory is global approximate, RSS is process-tree sum. Do not claim exact critical path without task-dependency edges.

## Checkpoint 4: real run and handoff

Use the known Chapter24 CBZ SHA256 `89A47EC64E3E1A230326B8C6BB1C82E80B50BCCAA4FA8A4A1C8D1D281BEA1F2E`; 18 pages; new output; external profiler timeout7200; with_warnings/enforce/shadow. Freeze sources/config during measurement, no benchmark competition.

Require explicit optimization activation counters, complete event, exit code, final performance status, 18 final page artifacts and persisted export gate. Technical completion is separate from export approval. Report interruption/failure as time-to-interruption/failure only. Report a reconciled timeline and all unmeasured residuals, not fictitious per-function precision.

No implicit commit, clean, stash, reset, training or production promotion. First checkpoint does not mean integrations or real measurements are complete.

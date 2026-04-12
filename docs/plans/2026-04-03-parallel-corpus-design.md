# Parallel Corpus And Example Operation Design

**Goal:** Turn the uploaded EN/PT-BR chapter corpus into a reusable internal dataset that supports future translation memory, style benchmarking, and quality calibration without binding the app runtime to a specific work.

**Architecture:** The first phase creates a dataset layer, not direct runtime heuristics. A corpus builder scans the paired chapter archives, normalizes chapter metadata, pairs EN/PT-BR chapters by chapter number, and emits stable JSON artifacts. A profile builder then derives benchmark summaries from the PT-BR side and alignment scaffolding from the EN/PT-BR side, preparing later training and runtime integration.

**Tech Stack:** Python 3.12, zipfile/pathlib/json, existing local example corpus in `exemplos/`, Rust/React unchanged in this first phase.

---

## 1. Scope

- Pair the EN and PT-BR chapter archives.
- Build a canonical corpus manifest.
- Extract reusable profile artifacts from the corpus.
- Keep the artifacts generic so other works can be added later.
- Do not yet hard-bind the runtime pipeline to this one work.

## 2. Artifacts

The first phase should output:

- `manifest.json`
  - one entry per paired chapter
  - chapter number
  - EN file path
  - PT-BR file path
  - page counts
  - PT-BR source group label
- `quality_profile.json`
  - benchmark stats derived from PT-BR references
  - chapter range coverage
  - page count totals
  - publisher/source distribution
- `alignment_profile.json`
  - parallel chapter coverage
  - per-chapter page count deltas
  - readiness flags for later segment/text alignment

## 3. Runtime Positioning

- These artifacts are internal dataset inputs.
- They are not yet injected blindly into normal runtime translation.
- Later phases can consume them to:
  - build translation memory
  - calibrate style
  - validate output against benchmarks

## 4. Constraints

- The corpus must remain decoupled from app behavior by default.
- File paths should remain local and explicit.
- Chapter pairing must be resilient to different scan group file names.
- The system must be extensible to multiple works later.

## 5. First Implementation Wave

1. Add corpus scanning and chapter pairing.
2. Add artifact generation.
3. Add tests for pairing and summary generation.
4. Add a script entrypoint to rebuild the corpus artifacts on demand.
5. Update `context.md` with the exact outputs and intent.

# Pipeline Half Runtime Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to execute this plan.

## Goal

Reduce the measured chapter runtime from `D:\TraduzAi\AAAAAAA\traduzido` from `227.3s` to about `110-115s` without hiding quality regressions.

This is not an OCR-only plan. The measured bottleneck is:

| Stage | Current time |
| --- | ---: |
| Inpaint | `121.3921s` |
| OCR | `69.4488s` |
| Typeset | `7.4284s` |
| Translate | `1.3133s` |
| Review layout | `1.0613s` |

The target is to remove roughly `112s`. The realistic path is:

1. Avoid work that should not happen: decorative/credit/timer noise and unchanged text should not go through translate, inpaint, and typeset.
2. Make most simple balloon cleaning cheap: use the existing fast-fill paths safely before LaMA/inpainting.
3. Reduce PaddleOCR calls in strip mode: run OCR on larger page/macro-band windows and remap results back to bands, first in shadow mode.

## Baseline Evidence

Use this artifact as the initial benchmark:

- `D:\TraduzAi\AAAAAAA\traduzido\project.json`
- `D:\TraduzAi\AAAAAAA\traduzido\qa_report.json`
- `D:\TraduzAi\debug\analysis_aaaaaaa_traduzido\pages_sheet.jpg`
- `D:\TraduzAi\debug\analysis_aaaaaaa_traduzido\suspect_crops.jpg`

Observed facts:

- `27` output pages.
- `114` text layers.
- `154` strip bands.
- `translated_regions = 114`.
- `qa_flags = 0`, but the visual/output review found real issues, so QA cannot be the only gate.
- `fast_white_balloon_count = 0` and `fast_local_balloon_count = 0`, even though fast-fill code exists in `pipeline/inpainter/__init__.py`.
- `ocr_full_page_mapped = 153` and `ocr_crop_fallback_attempts = 0`, so the current OCR path is already avoiding crop-by-crop fallback, but still runs once per strip band.
- Top inpaint outlier: band `0`, y `125-363`, `inpaint = 23.1512s`, `total = 26.0336s`, one detected text/balloon.

## Acceptance Targets

- End-to-end runtime for the same chapter or the same source reconstructed from `originals/` is `<= 115s`.
- `strip_perf_summary.durations_sec.inpaint <= 55s`.
- `strip_perf_summary.durations_sec.ocr <= 40s` for the first accepted optimization pass; stretch target `<= 30s`.
- `fast_white_balloon_count + fast_local_balloon_count > 0` on this chapter, ideally covering most white/simple balloons.
- QA must no longer report a fully clean run when obvious OCR policy issues remain, such as decorative credit/logo text, timers, or untranslated SFX/foreign fragments.
- No regression in metadata contracts:
  - `project.json` remains schema-compatible.
  - `text_layers.texts` and `ocr_result._vision_blocks` stay aligned.
  - `inpaint_blocks` do not include skipped/decorative text that was not actually cleaned.
- Visual spot-checks must pass for the known risky areas from `suspect_crops.jpg`, especially page 1 record-label/timer text.

## Architecture

The optimization should sit inside the existing strip pipeline, not create a second pipeline:

- `pipeline/strip/run.py` remains the chapter/band orchestrator.
- `pipeline/strip/process_bands.py` remains the stage-level contract: OCR -> review/layout -> translate -> inpaint -> typeset.
- `pipeline/vision_stack/runtime.py` remains the OCR stage implementation and text/no-text policy gate.
- `pipeline/inpainter/__init__.py` remains the band inpaint entry point and should decide cheap fill vs real inpaint.
- `pipeline/ocr/postprocess.py` and related reviewer modules should own OCR text policy/classification, not ad hoc checks in the renderer.

The order matters:

1. Add repeatable measurement and output comparison first.
2. Fix avoidable false-positive work.
3. Enable/repair cheap inpaint paths.
4. Only then attempt larger OCR batching/macro-band remapping.

## Risk Verdict

Cutting runtime roughly in half is plausible, but the risk is medium-high. The unsafe version of this plan would skip too much OCR/inpaint and silently leave untranslated or dirty text. The safe version must compare artifacts, metadata, and targeted crops after every optimization.

The highest-risk item is macro-band OCR remapping. It can save a lot of time, but it can also misassign text to the wrong strip band/page. It should ship only after a shadow-mode comparison proves that it preserves the current OCR output on representative chapters.

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Skipping real dialogue as decorative/credit text | Medium | High | Require conservative classifiers, add fixture tests, and compare dropped text lists against baseline. |
| Fast white/local fill erases art or leaves visible boxes | Medium | High | Gate by balloon/background confidence, inspect crop sheets, and keep feature flag fallback to LaMA. |
| Fast fill does not trigger because `balloon_bbox`/`tipo`/confidence metadata is missing | High | Medium | Add debug counters for rejection reasons before changing thresholds. |
| Macro-band OCR maps text to wrong band/page | Medium | High | Implement shadow mode first: old OCR remains source of truth while macro OCR diffs are logged. |
| PaddleOCR quality drops when using larger OCR windows or resized input | Medium | High | Track per-text diff, missing text count, confidence distribution, and known page 1/page 014 cases. |
| `_vision_blocks` and `inpaint_blocks` drift from actual inpainted regions | Medium | High | Add metadata assertions and compare counts to real cleaned/rendered slices. |
| QA still says `clean` while visual/OCR policy issues remain | High | Medium | Add QA warnings for decorative/timer/noise candidates and unchanged suspicious text. |
| Benchmark is polluted by cache, warmup, or translation network variance | Medium | Medium | Run controlled benchmark twice: cold-ish and warm. Focus on per-stage totals, not only wall clock. |
| Parallelism breaks PaddleOCR/LaMA thread safety or GPU memory | Medium | High | Do not use parallelism as the first optimization. Add it only behind a flag after avoid/fast-fill work. |
| Current dirty worktree hides regressions | High | Medium | Execute in a dedicated branch/worktree and never revert unrelated user edits. |

## Implementation Tasks

### Task 0 - Lock the Baseline and Harness

Files:

- `pipeline/scripts/analyze_pipeline_output.py` or a new focused script under `pipeline/scripts/`
- `pipeline/tests/test_strip_perf_summary.py`
- `D:\TraduzAi\debug\perf_baselines\aaaaaaa_2026-05-08.json`

Steps:

1. Add a script that reads an output folder and emits:
   - total pages/texts
   - `tempo_processamento_seg`
   - `strip_perf_summary.durations_sec`
   - top OCR/inpaint bands
   - counts for `fast_white_balloon_count`, `fast_local_balloon_count`, `remaining_inpaint_blocks`, skips, QA flags
   - suspicious text categories: timer-like OCR, all-caps logo/credit strings, unchanged SFX/foreign fragments
2. Save the current `AAAAAAA\traduzido` metrics as a baseline JSON.
3. Add a test for summary parsing so future perf work does not depend on manual JSON inspection.

Validation:

```powershell
cd D:\TraduzAi
pipeline\venv\Scripts\python.exe pipeline\scripts\analyze_pipeline_output.py D:\TraduzAi\AAAAAAA\traduzido --write-baseline D:\TraduzAi\debug\perf_baselines\aaaaaaa_2026-05-08.json
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_strip_perf_summary.py -q
```

### Task 1 - Stop Known False-Positive Work Early

Files:

- `pipeline/vision_stack/runtime.py`
- `pipeline/ocr/postprocess.py`
- `pipeline/strip/process_bands.py`
- `pipeline/tests/test_vision_stack_runtime.py`
- `pipeline/tests/test_ocr_postprocess.py`

Steps:

1. Extend the existing scanlation/cover editorial gates in `runtime.run_ocr_stage()`.
2. Add a counter for `cover_editorial_skipped` to `process_band()` and `_summarize_band_perf()`.
3. Add conservative text policy tags after OCR:
   - `decorative_credit`
   - `timer_noise`
   - `sfx_or_sound`
   - `proper_name_or_title`
   - `foreign_phrase_candidate`
4. Only auto-skip high-confidence decorative/timer/credit noise. SFX/proper-name/foreign candidates should be warnings unless already marked `skip_processing`.
5. Add fixtures based on the page 1 false positives:
   - `Side Stereo SFB2BT...`
   - `MOONAGE DAYDRCAM STARMAN`
   - `00:00:05` becoming `Oo:oo:os`

Expected saving:

- Direct: small to medium.
- Important outlier: likely removes the `23s` band-0 inpaint case if it is decorative/cover text.
- Quality improvement: QA stops calling the chapter fully clean when it is not.

Validation:

```powershell
cd D:\TraduzAi
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_vision_stack_runtime.py pipeline\tests\test_ocr_postprocess.py -q -x
```

### Task 2 - Per-Text Skip Before Inpaint and Typeset

Files:

- `pipeline/strip/process_bands.py`
- `pipeline/inpainter/__init__.py`
- `pipeline/typesetter/` relevant renderer entry point
- `pipeline/tests/test_strip_process_bands.py`
- `pipeline/tests/test_vision_stack_inpainter.py`

Steps:

1. Keep the current whole-band skips, but add a per-text renderable list for mixed bands.
2. Texts with `skip_processing=True` or high-confidence unchanged/decorative policy should remain in metadata as needed, but must not create inpaint work or rendered overlay.
3. Make `_vision_blocks` / `inpaint_blocks` reflect only regions that were actually cleaned.
4. Add perf counters:
   - `per_text_skip_count`
   - `per_text_skip_inpaint_block_count`
   - `renderable_text_count`
5. Verify that editor text layers still preserve enough metadata for manual correction.

Expected saving:

- Medium, depends on how many mixed bands contain names/SFX/decorative fragments.
- Also prevents unnecessary visual damage from unchanged text.

Validation:

```powershell
cd D:\TraduzAi
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_strip_process_bands.py pipeline\tests\test_vision_stack_inpainter.py -q -x
```

### Task 3 - Repair and Safely Enable Fast Inpaint

Files:

- `pipeline/inpainter/__init__.py`
- `pipeline/vision_stack/runtime.py`
- `pipeline/strip/process_bands.py`
- `pipeline/tests/test_vision_stack_inpainter.py`
- `pipeline/tests/test_inpainting_profile.py`
- `pipeline/tests/test_mask_builder.py`

Steps:

1. Add rejection-reason counters to `_apply_fast_white_balloon_fill()` and `_apply_fast_local_balloon_fill()`:
   - disabled flag
   - missing `balloon_bbox`
   - low confidence
   - unsupported `tipo`
   - no white/flat fill mask
   - no covered `_vision_blocks`
2. Run the current artifact/source once with:
   - `TRADUZAI_STRIP_FAST_WHITE_INPAINT=1`
   - `TRADUZAI_STRIP_FAST_LOCAL_INPAINT=1`
   - keep narration disabled first: `TRADUZAI_STRIP_FAST_WHITE_NARRATION=0`
3. If counts remain zero, fix metadata conditions before lowering thresholds.
4. Promote safe defaults only after crop comparison passes:
   - white speech/thought balloons can default on
   - local solid fill can default on only for flat panels
   - narration remains opt-in until it has enough fixtures
5. Keep LaMA as fallback for textured/art-background regions.

Expected saving:

- Main inpaint target: `121s -> <= 55s`.
- If many balloons are white/simple, this is the biggest low-risk runtime win.

Validation:

```powershell
cd D:\TraduzAi
$env:TRADUZAI_STRIP_FAST_WHITE_INPAINT="1"
$env:TRADUZAI_STRIP_FAST_LOCAL_INPAINT="1"
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_vision_stack_inpainter.py pipeline\tests\test_inpainting_profile.py pipeline\tests\test_mask_builder.py -q -x
```

Then run a full chapter benchmark and compare:

- `fast_white_balloon_count`
- `fast_local_balloon_count`
- `remaining_inpaint_blocks`
- `durations_sec.inpaint`
- crop sheet around page 1 and several normal speech balloons

### Task 4 - OCR Macro-Band Shadow Mode

Files:

- `pipeline/strip/run.py`
- `pipeline/strip/process_bands.py`
- `pipeline/vision_stack/runtime.py`
- `pipeline/vision_stack/ocr.py`
- `pipeline/tests/test_strip_macro_ocr.py`
- `pipeline/tests/test_primary_ocr_routing.py`

Steps:

1. Add a feature flag: `TRADUZAI_STRIP_MACRO_OCR_SHADOW=1`.
2. Group strip bands by source page or bounded vertical windows.
3. Run PaddleOCR once per macro window using the same `recognize_blocks_from_page()` logic.
4. Remap OCR line/block results back into each band by y-intersection.
5. In shadow mode:
   - keep old per-band OCR as the output source
   - log macro-vs-band diffs into `_ocr_stats`
   - record missing/extra/changed text counts and confidence deltas
6. Add synthetic tests for y-coordinate remapping and boundary cases where text overlaps two bands/pages.

Expected saving after activation:

- OCR target: `69s -> <= 40s` initially.
- Stretch target: `69s -> <= 30s` if macro windows reduce OCR calls from `153` to near page count.

Validation:

```powershell
cd D:\TraduzAi
$env:TRADUZAI_STRIP_MACRO_OCR_SHADOW="1"
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_strip_macro_ocr.py pipeline\tests\test_primary_ocr_routing.py -q -x
```

Run a full benchmark and require:

- near-zero missing dialogue texts
- no page assignment drift in `project.json`
- known problem crops still reviewed manually

### Task 5 - Enable Macro OCR Behind a Flag

Files:

- `pipeline/strip/run.py`
- `pipeline/vision_stack/runtime.py`
- `pipeline/tests/test_strip_macro_ocr.py`

Steps:

1. Add activation flag: `TRADUZAI_STRIP_MACRO_OCR=1`.
2. Use macro OCR output only when shadow diff is below strict thresholds:
   - no high-confidence text missing
   - text count delta within a small configured bound
   - no line assigned outside its page/band bounds
3. Fall back to per-band OCR automatically if the macro result fails validation.
4. Keep the old per-band path available for quick rollback.

Expected saving:

- This is the second major runtime win after fast inpaint.

Validation:

```powershell
cd D:\TraduzAi
$env:TRADUZAI_STRIP_MACRO_OCR="1"
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_strip_macro_ocr.py pipeline\tests\test_vision_stack_ocr.py pipeline\tests\test_primary_ocr_routing.py -q -x
```

### Task 6 - Full Benchmark and Quality Gate

Files:

- `pipeline/main.py`
- `pipeline/scripts/analyze_pipeline_output.py`
- `debug/perf_baselines/`
- generated output folder under `D:\TraduzAi\debug\perf_runs\`

Steps:

1. Reconstruct the run input:
   - prefer the same source/config from the app work folder if available
   - otherwise use `D:\TraduzAi\AAAAAAA\traduzido\originals` as the source folder for a controlled rerun
2. Run baseline and optimized outputs into separate folders.
3. Generate:
   - perf JSON
   - crop sheet for suspicious text
   - page contact sheet
   - text diff by page
   - metadata diff for `_vision_blocks`, `inpaint_blocks`, and `text_layers`
4. Accept only if runtime and visual/metadata gates both pass.

Suggested command shape:

```powershell
cd D:\TraduzAi
pipeline\venv\Scripts\python.exe pipeline\main.py <config-json>
pipeline\venv\Scripts\python.exe pipeline\scripts\analyze_pipeline_output.py <output-dir> --compare D:\TraduzAi\debug\perf_baselines\aaaaaaa_2026-05-08.json
```

### Task 7 - Optional Parallelism Only After Safe Savings

Files:

- `pipeline/strip/run.py`
- `pipeline/strip/process_bands.py`

Steps:

1. Do not start here.
2. If inpaint+macro OCR still miss the target, evaluate a single-worker pipeline:
   - OCR next macro window while inpaint handles previous one
   - no concurrent PaddleOCR sessions
   - no concurrent LaMA sessions unless verified safe
3. Keep this behind an opt-in flag.

Expected saving:

- Small to medium.
- Higher operational risk than avoid/fast-fill/macro OCR.

## Execution Order

1. Task 0: measurement harness.
2. Task 1: false-positive/decorative skip.
3. Task 2: per-text skip before inpaint/typeset.
4. Task 3: fast inpaint repair and activation.
5. Full benchmark checkpoint. If runtime is already near `115s`, stop.
6. Task 4: macro OCR shadow mode.
7. Task 5: macro OCR activation.
8. Task 6: final benchmark and quality gate.
9. Task 7 only if still above target.

## Recommended First Milestone

The first milestone should be small and measurable:

- Add baseline script.
- Fix/record `cover_editorial_skipped`.
- Add fast-fill rejection counters.
- Run the same chapter once with fast-fill flags enabled.

This tells us whether the inpaint target is reachable without touching risky OCR remapping.

## Commands for the First Milestone

```powershell
cd D:\TraduzAi
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_vision_stack_runtime.py pipeline\tests\test_vision_stack_inpainter.py -q -x
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_vision_stack_ocr.py pipeline\tests\test_primary_ocr_routing.py pipeline\tests\test_ocr_reviewer.py pipeline\tests\test_contextual_reviewer.py -q -x
```

For final hygiene after code changes:

```powershell
cd D:\TraduzAi
git diff --check
```

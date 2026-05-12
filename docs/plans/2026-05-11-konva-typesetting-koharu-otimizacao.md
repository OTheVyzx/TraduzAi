# Konva Typesetting + Koharu Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make editor, preview, and export use one visual rendering contract while keeping Koharu CJK OCR/detect and reducing chapter time safely.

**Architecture:** Keep Python responsible for detect, OCR, inpaint, translation, and project generation. Move final typesetting for preview/export toward the same Konva-based renderer used by the editor, behind a feature flag first, while optimizing Koharu worker latency in measured steps.

**Tech Stack:** React, Konva, Tauri commands, Python pipeline, Koharu worker, PaddleOCR-VL, existing `project.json` text layer schema.

---

## Safety Rules

- Do not remove the Python typesetter until Konva export is visually verified on English and CJK chapters.
- Keep the existing Python render as fallback behind a flag.
- Every performance claim must come from fresh timing output, not estimates.
- Do not weaken Koharu CJK coverage to gain speed. Missing text is worse than a slower run.

## Target Order

1. Instrument real worker batch timings.
2. Make the worker persistent.
3. Add `ocrOnly` with `knownTextBBoxes`.
4. Add cheap empty-ROI filtering.
5. Add dynamic `maxNewTokens`.
6. Add Konva export/preview renderer behind a feature flag.
7. Switch Preview/export to Konva only after visual parity is proven.

---

## Task 1: Worker Batch Timing Instrumentation

**Files:**
- Modify: `vision-worker/src/main.rs`
- Modify: `pipeline/vision_stack/runtime.py`
- Modify: `pipeline/strip/run.py`
- Test: `pipeline/tests/test_strip_run.py`

**Steps:**
1. Add per-batch timing fields for worker startup wait, queue wait, model inference, JSON parse, and postprocess.
2. Thread those timings into `_ocr_stats` and `strip_perf_summary`.
3. Add a focused test that asserts timing keys are preserved.
4. Run:
   - `python -m pytest tests\test_strip_run.py -q -k "koharu_precompute"`
   - `cargo check` in `vision-worker` if local Rust toolchain is available.

**Expected Result:** A chapter run tells exactly why Koharu took 120s vs 70s.

---

## Task 2: Persistent Koharu Worker

**Files:**
- Modify: `src-tauri/src/commands/pipeline.rs`
- Modify: `vision-worker/src/main.rs`
- Modify: `pipeline/main.py`
- Modify: `pipeline/vision_stack/runtime.py`
- Test: `pipeline/tests/test_main_strip_config.py`

**Steps:**
1. Start the worker during app startup or first pipeline setup, not after the first OCR batch.
2. Keep a heartbeat/status endpoint or lightweight ping.
3. Reuse the same process for all chapter batches.
4. Fall back to current startup path if the persistent worker is unavailable.
5. Add a test/fixture proving config still works without the persistent worker.

**Expected Result:** Cold start is paid before the heavy chapter stage, reducing visible pipeline time without removing Koharu.

---

## Task 3: `ocrOnly` With Known Text BBoxes

**Files:**
- Modify: `vision-worker/src/main.rs`
- Modify: `pipeline/vision_stack/runtime.py`
- Modify: `pipeline/strip/run.py`
- Test: `pipeline/tests/test_strip_run.py`
- Test: `pipeline/tests/test_vision_stack_runtime.py`

**Steps:**
1. Add request mode: `ocrOnly`.
2. Send existing detector/Koharu text bboxes as `knownTextBBoxes`.
3. In this mode, skip detector work and OCR only those crops.
4. Preserve full detect+OCR mode as fallback.
5. Add tests proving bboxes are sent and results map back to page/band coordinates.

**Expected Result:** Koharu remains in the pipeline, but avoids repeated detection where we already know the text regions.

---

## Task 4: Cheap Empty-ROI Filter

**Files:**
- Modify: `pipeline/strip/run.py`
- Modify: `pipeline/vision_stack/runtime.py`
- Test: `pipeline/tests/test_strip_run.py`

**Steps:**
1. Before sending ROI to Koharu, run a cheap image check for text-like dark/bright components.
2. Skip ROIs that are only blank panel space, title padding, or pure background.
3. Do not skip if the ROI intersects a known balloon bbox with possible CJK text.
4. Record skip counts in telemetry.

**Expected Result:** Fewer Koharu jobs, without repeating the old bug of losing CJK text.

---

## Task 5: Dynamic `maxNewTokens`

**Files:**
- Modify: `vision-worker/src/main.rs`
- Modify: `pipeline/vision_stack/runtime.py`
- Test: `pipeline/tests/test_vision_stack_runtime.py`

**Steps:**
1. Estimate token budget from ROI size and expected text count.
2. Use lower `maxNewTokens` for small single-balloon OCR.
3. Keep high budget for large connected balloons or dense narration.
4. Log chosen token budget in worker timing metadata.

**Expected Result:** Small OCR calls finish faster while large/complex regions keep quality.

---

## Task 6: Konva Renderer Adapter

**Files:**
- Create: `src/lib/konvaExportRenderer.ts`
- Modify: `src/components/editor/stage/EditorTextLayer.tsx`
- Modify: `src/lib/stores/editorStore.ts`
- Test: `src/lib/__tests__/konvaExportRenderer.test.ts`

**Steps:**
1. Extract the editor text-layer drawing rules into reusable functions.
2. Build a renderer adapter that can draw a page from `project.json` data without editor interaction.
3. Keep exact same font, line height, alignment, rotation, color, shadow, contour, and uppercase behavior.
4. Add tests for line wrapping and style normalization.

**Expected Result:** There is one typesetting logic path shared by editor and export renderer.

---

## Task 7: Headless Konva Preview Render

**Files:**
- Modify: `src/lib/tauri.ts`
- Modify: `src-tauri/src/commands/project.rs`
- Modify: `src/pages/Preview.tsx`
- Modify: `src/pages/previewImage.ts`
- Test: `src/pages/previewImage.test.ts`
- Test: `src/lib/stores/__tests__/editorRenderPreviewCache.test.ts`

**Steps:**
1. Add feature flag `TRADUZAI_KONVA_RENDER_PREVIEW=1`.
2. Render final preview from the Konva adapter into a bitmap.
3. Use existing Python render if Konva render fails.
4. Ensure Preview and Editor consume the same materialized text layer state.
5. Remove the need for manual “Salvar+Render” when there are no pending edits.

**Expected Result:** The preview visually matches the editor on current-page render.

---

## Task 8: Konva Export Path

**Files:**
- Modify: `src-tauri/src/commands/project.rs`
- Modify: `src/lib/tauri.ts`
- Modify: `src/pages/Preview.tsx`
- Test: `src/lib/__tests__/tauriRenderPreview.test.ts`

**Steps:**
1. Add export mode that renders every page through the Konva adapter.
2. Keep Python export as fallback.
3. Block export only for real unsaved editor changes, not missing/stale preview cache.
4. Save generated JPGs to the same export pipeline used by CBZ/ZIP.

**Expected Result:** Exported pages match the editor and do not depend on a second Python typesetting result.

---

## Task 9: Visual Parity Gate

**Files:**
- Create: `pipeline/tools/compare_render_outputs.py`
- Create: `pipeline/tests/test_compare_konva_python_render.py`
- Use outputs:
  - `N:\TraduzAI\TraduzAi\DDDDDDDDDDDDDDDDDDDD\traduzido`
  - `N:\TraduzAI\TraduzAi\CCCCCCCCCCCC\traduzido`
  - `N:\TraduzAI\TraduzAi\exemplos\exemploko\환생천마`

**Steps:**
1. Generate side-by-side contact sheets for Python render, editor/Konva render, and original.
2. Check pages with connected balloons, all-caps English, Korean dialogue, and white-panel narration.
3. Track visual failures as page numbers and layer ids.
4. Do not switch defaults until the reviewed set is clean or better than Python.

**Expected Result:** Konva becomes default only when it demonstrably improves preview/export parity.

---

## Task 10: Full Chapter Benchmark

**Files:**
- Modify: `debug/run_example_candidate_benchmark.py`
- Output: `debug/performance_gates/`

**Steps:**
1. Run baseline with current Python render.
2. Run optimized Koharu with Python render.
3. Run optimized Koharu with Konva preview/export render.
4. Compare:
   - total chapter time
   - Koharu precompute time
   - inpaint time
   - render/export time
   - QA flags
   - visual contact sheet

**Expected Result:** We know whether Konva improves total time or mainly improves consistency.

---

## Rollout Recommendation

Default sequence:

1. Ship timing instrumentation first.
2. Ship persistent worker if timings confirm cold/start wait.
3. Ship `ocrOnly` and empty-ROI filter behind flags.
4. Ship dynamic token budget behind flag.
5. Add Konva preview/export behind flag.
6. After visual parity, make Konva default for preview/export.
7. Keep Python typesetter as fallback for at least one release cycle.

## Risk Notes

- Konva will not fix missed OCR by itself.
- Konva will reduce renderer divergence, but inpaint still needs to remove original text correctly.
- Full headless export may need font-loading synchronization; this must be tested before trusting CBZ output.
- The biggest current time cost is still Koharu/OCR and inpaint, not typesetting. Konva is primarily a consistency improvement, with export/preview speed as a secondary gain.

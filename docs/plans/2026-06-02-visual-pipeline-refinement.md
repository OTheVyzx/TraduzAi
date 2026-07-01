# Visual Pipeline Refinement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refine the TraduzAI visual pipeline so detect/OCR/mask/inpaint/typesetting stop creating residues, wrong-region inpaint, false text captures, tiny text, and EasyOCR fallback regressions.

**Architecture:** Keep TraduzAI as the owner of the automatic pipeline. Do not vendor or call the external repositories directly as production defaults; import only the useful ideas that are immediately actionable: NotAnotherBubbleCleaner-style component filtering inside real BubbleMask and manga-cleaner-style ROI scheduling with snap-to-8 reflected padding. Bubble-Detector-YOLOv4 training/download/inference is deferred to a future plan because the bubble-specific weights are not available in the local checkout; keep Koharu renderer work separate from detect/mask/inpaint and preserve existing QA/export contracts.

**Tech Stack:** Python 3.12, PaddleOCR, OpenCV/NumPy, current TraduzAI inpainter/typesetter/QA modules, optional existing Rust renderer bridge, React/Tauri only for status surfaces if needed.

---

## Ground Rules

- Work in `N:\TraduzAI` unless explicitly moving to `N:\TraduzAI\.worktrees\bubble-mask-cleaner-v2`.
- Preserve dirty working tree changes. Do not reset, checkout, or revert unrelated files.
- Do not version large `DEBUGM/runs`, `data/logs`, generated workspaces, or model binaries.
- Never fall back to EasyOCR. Paddle failure must fail closed with a clear error/event.
- Do not make Bubble-Detector-YOLOv4, NotAnotherBubbleCleaner, or manga-cleaner production dependencies.
- Keep all external-repo-derived logic as owned TraduzAI code with provenance/debug fields.
- Do not implement, download, train, or wire Bubble-Detector-YOLOv4 in this plan. Treat it as future work only.
- Validate one chapter at a time before broad visual sweeps.

---

## Phase 0: Baseline And Target Checkout

### Task 0.1: Confirm Dirty State And Target Branch

**Files:**
- Read-only: repository state

**Steps:**
1. Run:
   ```powershell
   git status --short
   git branch --show-current
   ```
2. Save the output in notes for the execution session.
3. Confirm whether implementation happens in `N:\TraduzAI` or by porting from `N:\TraduzAI\.worktrees\bubble-mask-cleaner-v2`.

**Expected:** No edits yet. Existing dirty files are documented.

### Task 0.2: Inventory Existing Partial Work

**Files:**
- Read: `N:\TraduzAI\.worktrees\bubble-mask-cleaner-v2\pipeline\vision_stack\external_bubble_detector.py`
- Read: `N:\TraduzAI\.worktrees\bubble-mask-cleaner-v2\pipeline\inpainter\notanother_adapter.py`
- Read: `N:\TraduzAI\.worktrees\bubble-mask-cleaner-v2\pipeline\inpainter\manga_cleaner_adapter.py`
- Read: `N:\TraduzAI\.worktrees\bubble-mask-cleaner-v2\pipeline\inpainter\mask_builder.py`
- Read: `N:\TraduzAI\.worktrees\bubble-mask-cleaner-v2\pipeline\vision_stack\runtime.py`

**Steps:**
1. Compare those files against the active checkout.
2. Mark which code is safe to port as owned logic.
3. Do not copy external repo code wholesale.

**Expected:** A small port list, not a broad sync.

---

## Phase 1: OCR Contract And No EasyOCR

### Task 1.1: Add Fail-Closed OCR Tests

**Files:**
- Modify: `pipeline/tests/test_vision_stack_ocr.py`
- Modify or create: `pipeline/tests/test_ocr_no_easyocr.py`
- Modify or create: `pipeline/tests/test_ocr_language_filter.py`

**Tests To Add:**
1. Paddle unavailable must not call `_load_easyocr`.
2. Legacy OCR path must not import or instantiate EasyOCR.
3. When source language is English, Korean-only SFX/art fragments are filtered from translation candidates unless they are geometrically tied to a real English OCR block/bubble route.
4. Paddle full-page OCR remains enabled and maps line polygons/rotation to blocks.

**Run:**
```powershell
N:\TraduzAI\pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_vision_stack_ocr.py pipeline/tests/test_ocr_no_easyocr.py pipeline/tests/test_ocr_language_filter.py -q
```

**Expected First Run:** New tests fail because EasyOCR fallback still exists or language filtering is incomplete.

### Task 1.2: Remove EasyOCR Fallback From Active Paths

**Files:**
- Modify: `pipeline/vision_stack/ocr.py`
- Modify: `pipeline/ocr/detector.py`
- Modify: `pipeline/ocr_legacy/detector.py`
- Modify: `pipeline/ocr_legacy/recognizer_paddle.py`
- Modify: `pipeline/vision_stack/runtime.py`

**Implementation:**
- Replace EasyOCR fallback with an explicit `OcrBackendUnavailable` or existing pipeline error object.
- Keep Paddle retries/upscale/crop repair.
- If Paddle cannot initialize, emit failure evidence and block export through the existing gate; do not silently switch OCR engine.
- Remove `choose_primary_ocr_engine(...)->easyocr` behavior.
- Keep any EasyOCR import only if required by old tests? Prefer removing runtime imports entirely.

**Run:**
```powershell
N:\TraduzAI\pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_vision_stack_ocr.py pipeline/tests/test_ocr_no_easyocr.py pipeline/tests/test_vision_stack_runtime.py -q
```

**Expected:** OCR tests pass. No log should say `usando EasyOCR`.

### Task 1.3: English-Source CJK/SFX Suppression

**Files:**
- Modify: `pipeline/vision_stack/runtime.py`
- Modify: `pipeline/ocr/postprocess.py`
- Modify: `pipeline/ocr/ocr_normalizer.py`
- Modify: `pipeline/ocr/semantic_reviewer.py`

**Implementation:**
- Add a small predicate like `is_foreign_script_artifact_for_source(text, source_lang, geometry)`.
- For `source_lang=en`, Korean-only or CJK-only fragments should not become translation/render blocks unless:
  - the OCR text also has Latin text nearby in the same block, or
  - it is inside a confirmed dialogue BubbleMask with route evidence, or
  - user explicitly enables SFX/art translation.
- Do not delete evidence; mark as `ignored_artifact_candidate`/`non_source_script_artifact`.

**Run:**
```powershell
N:\TraduzAI\pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_ocr_language_filter.py pipeline/tests/test_ocr_postprocess.py -q
```

**Expected:** Korean SFX in English chapters does not create random Portuguese text, and English dialogue still passes.

---

## Phase 2: BubbleMask And Text Mask Refinement

### Task 2.1: Add NotAnother-Style Component Mask Tests

**Files:**
- Modify or create: `pipeline/tests/test_notanother_component_mask.py`
- Modify: `pipeline/tests/test_inpaint_mask_geometry.py`
- Modify: `pipeline/tests/test_mask_builder.py`

**Tests To Add:**
1. Given a real BubbleMask and OCR support mask, only glyph components inside the eroded interior are accepted.
2. Components touching the balloon outline are shrunk or rejected, preventing eaten borders.
3. Components outside BubbleMask are rejected even if inside `balloon_bbox`.
4. Empty/invalid BubbleMask does not fall back to filling the bbox.
5. `mask_density_high` must be debug evidence only, not a blocking decision.

**Run:**
```powershell
N:\TraduzAI\pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_notanother_component_mask.py pipeline/tests/test_inpaint_mask_geometry.py pipeline/tests/test_mask_builder.py -q
```

**Expected First Run:** Tests fail until owned component filtering is added/ported.

### Task 2.2: Implement Owned Component Mask Builder

**Files:**
- Create or port: `pipeline/inpainter/notanother_adapter.py`
- Modify: `pipeline/inpainter/mask_builder.py`

**Implementation:**
- Build a binary glyph/text mask from image + BubbleMask + OCR support mask.
- Erode BubbleMask interior by 1-3 px before accepting components.
- Threshold dark glyph candidates adaptively inside the support region.
- Use connected components:
  - reject too-small components;
  - reject centroid outside safe BubbleMask;
  - reject low overlap with safe BubbleMask;
  - reject low overlap with OCR support mask;
  - fill holes only after acceptance;
  - clip final mask back to safe BubbleMask and support mask.
- Return debug metrics: component counts, threshold, rejected reasons, mask pixels.

**Important:** This is not a direct import of NotAnotherBubbleCleaner. It is TraduzAI-owned code implementing the same safe geometric idea.

**Run:**
```powershell
N:\TraduzAI\pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_notanother_component_mask.py pipeline/tests/test_mask_builder.py -q
```

**Expected:** Tests pass.

### Task 2.3: Remove BBox As Strong Mask Source

**Files:**
- Modify: `pipeline/inpainter/mask_builder.py`
- Modify: `pipeline/vision_stack/runtime.py`
- Modify: `pipeline/qa/render_geometry.py`
- Modify: `pipeline/qa/export_gate.py`

**Implementation:**
- Keep `balloon_bbox` only as weak ROI/debug evidence.
- Do not generate inpaint mask from full bbox.
- If real BubbleMask is missing, use OCR glyph mask + conservative expansion only.
- If mask cannot be built safely, mark review/block with explicit evidence; do not erase bbox area.
- Remove decisions based on `content_class`, `tipo`, `balloon_type`, `skip_processing`, or `preserve_original`.

**Run:**
```powershell
N:\TraduzAI\pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_inpaint_mask_geometry.py pipeline/tests/test_export_gate.py pipeline/tests/test_render_geometry.py -q
```

**Expected:** No bbox-fill path remains as automatic default.

---

## Phase 3: Manga-Cleaner-Style ROI Scheduling

### Task 3.1: Add ROI Scheduler Tests

**Files:**
- Modify: `pipeline/tests/test_inpaint_region_strategy.py`
- Modify or create: `pipeline/tests/test_manga_cleaner_roi_strategy.py`

**Tests To Add:**
1. Mask components produce independent ROIs centered on components.
2. ROI dimensions are padded to multiples of 8.
3. ROI uses reflect padding before model call and crops back to original size.
4. Inpaint pasteback only changes pixels covered by the mask.
5. Overlapping ROIs do not repeatedly damage already processed regions.

**Run:**
```powershell
N:\TraduzAI\pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_inpaint_region_strategy.py pipeline/tests/test_manga_cleaner_roi_strategy.py -q
```

**Expected First Run:** Fails where current strategy still uses broader tiles/pasteback.

### Task 3.2: Implement Component ROI Strategy

**Files:**
- Modify: `pipeline/inpainter/region_strategy.py`
- Modify: `pipeline/inpainter/lama_onnx.py`
- Modify: `pipeline/vision_stack/inpainter.py`

**Implementation:**
- Add strategy `component_roi_snap8`.
- Split masks by connected components.
- For each component:
  - compute center;
  - choose ROI with configurable margin;
  - snap model input size to multiple of 8;
  - reflect-pad image and mask;
  - run AOT/LaMa only on ROI;
  - paste back only masked pixels.
- Default to this strategy for text masks.
- Keep global/large ROI only for masks above a high area threshold with explicit debug evidence.

**Run:**
```powershell
N:\TraduzAI\pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_inpaint_region_strategy.py pipeline/tests/test_vision_stack_inpainter.py -q
```

**Expected:** ROI tests pass; no whole-art distortion for small SFX/text fragments.

---

## Phase 4: Safe Fill And Inpaint Decisions

### Task 4.1: Add Safe Fill Tests

**Files:**
- Modify: `pipeline/tests/test_inpaint_mask_geometry.py`
- Modify: `pipeline/tests/test_inpaint_region_strategy.py`

**Tests To Add:**
1. Fast/white fill is allowed only in low-variance white BubbleMask interiors.
2. Fast fill never modifies outline pixels.
3. Dark/colored/textured backgrounds route to AOT/LaMa.
4. If safe conditions fail, result uses real inpaint, not bbox cleanup.

**Run:**
```powershell
N:\TraduzAI\pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_inpaint_mask_geometry.py pipeline/tests/test_inpaint_region_strategy.py -q
```

### Task 4.2: Implement Conservative Safe Fill

**Files:**
- Modify: `pipeline/inpainter/mask_builder.py`
- Modify: `pipeline/inpainter/region_strategy.py`
- Modify: `pipeline/vision_stack/runtime.py`

**Implementation:**
- Safe fill condition:
  - has real BubbleMask ID;
  - final text mask is inside eroded BubbleMask;
  - sampled background inside bubble has low variance and high brightness;
  - only final glyph mask pixels are modified.
- Otherwise route to AOT/LaMa.
- Keep `mask_density_high` as debug-only.

**Run:**
```powershell
N:\TraduzAI\pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_inpaint_mask_geometry.py pipeline/tests/test_vision_stack_runtime.py -q
```

---

## Phase 5: External Repo Boundaries And Deferred Bubble-Detector

### Task 5.1: Document Bubble-Detector-YOLOv4 As Deferred

**Files:**
- Modify: `docs/plans/2026-06-02-visual-pipeline-refinement.md`
- Create or modify only if a code reference already exists: `pipeline/tests/test_external_bubble_detector.py`

**Implementation:**
- Do not add Bubble-Detector-YOLOv4 as an active detector.
- Do not download Google Drive weights in this plan.
- Do not create a runtime dependency on `N:\TraduzAI\Bubble-Detector-YOLOv4`.
- If active checkout already has an adapter from prior work, add/keep a guard test proving:
  - missing bubble-specific weights produce no detections;
  - adapter is disabled by default;
  - bboxes are never converted into erase masks.
- Move actual Bubble-Detector training/conversion/evaluation to a separate future plan.

**Run:**
```powershell
N:\TraduzAI\pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_external_bubble_detector.py -q
```

**Expected:** If this test file exists, it confirms Bubble-Detector is inactive/no-op without usable weights. If no adapter exists in the active checkout, skip this test and do not create runtime code.

### Task 5.2: Keep Manga-Cleaner Direct Adapter Out Of Runtime

**Files:**
- Modify or create only if already referenced by active code: `pipeline/tests/test_manga_cleaner_adapter.py`

**Implementation:**
- Do not call the local manga-cleaner GUI.
- Do not create a runtime dependency on PySide, pywin32, or manga-cleaner models.
- Main pipeline must use owned ROI code from Phase 3, not a subprocess.
- If an old subprocess adapter exists, keep it disabled and prove it is unavailable unless explicitly configured.

**Run:**
```powershell
N:\TraduzAI\pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_manga_cleaner_adapter.py -q
```

**Expected:** Direct manga-cleaner integration stays out of the automatic path.

---

## Phase 6: Renderer/Typesetting Refinement

### Task 6.1: Add Tiny Text And Fit Regression Tests

**Files:**
- Modify: `pipeline/tests/test_typesetting_layout.py`
- Modify: `pipeline/tests/test_typesetting_renderer.py`
- Modify: `pipeline/tests/test_render_geometry.py`

**Tests To Add:**
1. Long Portuguese text in a large bubble must not shrink below minimum legible size unless export blocks.
2. Burst/radial balloons must use a safe inner region, not the full spiky outline.
3. Connected/overlapping balloons preserve separate text placement when distinct BubbleMask IDs exist.
4. Page-space text outside the bubble fails QA instead of silently exporting.

**Run:**
```powershell
N:\TraduzAI\pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_typesetting_layout.py pipeline/tests/test_typesetting_renderer.py pipeline/tests/test_render_geometry.py -q
```

### Task 6.2: Refine Current Renderer Before Full Koharu Swap

**Files:**
- Modify: `pipeline/typesetter/renderer.py`
- Modify: `pipeline/layout/balloon_layout.py`
- Modify: `pipeline/qa/render_geometry.py`

**Implementation:**
- Keep `Comic Neue Bold` or configured comic font as default for dialogue unless a deliberate style rule says otherwise.
- Increase minimum legible font floor for large balloons.
- Improve line breaking for PT-BR:
  - no orphan tiny words when avoidable;
  - preserve short semantic groups like `Hosu 24 anos` where possible;
  - prefer balanced line heights over extreme compression.
- If text cannot fit legibly, set QA block with evidence instead of rendering tiny.
- Do not complete full Koharu migration here; keep that behind the existing renderer migration plan and feature flag.

**Run:**
```powershell
N:\TraduzAI\pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_typesetting_layout.py pipeline/tests/test_typesetting_renderer.py pipeline/tests/test_render_geometry.py -q
```

---

## Phase 7: QA And Export Gate

### Task 7.1: QA Must Catch Visual Failures

**Files:**
- Modify: `pipeline/qa/export_gate.py`
- Modify: `pipeline/qa/render_geometry.py`
- Modify: `pipeline/main.py`
- Modify: `src-tauri/src/commands/pipeline.rs` only if event shape changed
- Modify: `src/pages/Preview.tsx` only if UI status changed

**Tests To Add Or Update:**
- `pipeline/tests/test_export_gate.py`
- `pipeline/tests/test_main_emit.py`
- Rust command tests if event payload changed.

**Rules:**
- Export blocks when:
  - original text remains visible after render/inpaint;
  - render is missing;
  - text is outside safe region;
  - fit is below minimum legible;
  - inpaint residual is confirmed.
- Export does not block only because of:
  - low confidence;
  - `mask_density_high`;
  - `ocr_truncated_or_joined` before repair attempt.

**Run:**
```powershell
N:\TraduzAI\pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_export_gate.py pipeline/tests/test_main_emit.py pipeline/tests/test_render_geometry.py -q
```

---

## Phase 8: Visual Validation, One Chapter At A Time

### Task 8.1: One Second Chapter 1

**Input:** The same One Second chapter 1 fixture/path used in prior runs.

**Run With Debug:**
```powershell
N:\TraduzAI\pipeline\venv\Scripts\python.exe pipeline\main.py <config-one-second-ch1.json>
```

**Inspect:**
- translated pages;
- debug overlays;
- manifest/export gate;
- pages with previous failures: UI/search, `LET'S GO!!`, small bubbles, connected balloons, p22/p23/p37/p42/p45/p53.

**Expected:**
- No EasyOCR log.
- No random CJK/SFX translation for English-source chapter.
- Inpaint does not eat balloon borders.
- Text is legible and inside safe regions.
- Export blocks only on real unresolved visual failures.

### Task 8.2: One Second Chapter 2

Repeat Task 8.1. Do not start broad corpus validation until ch1/ch2 are acceptable.

### Task 8.3: Monster / Other English Chapters

Run one English chapter at a time from `D:\Mihon pra pc\downloads\mangas` with `(EN)` in the title.

**Inspect:**
- SFX false captures;
- Korean/CJK remnants;
- white-fill artifacts;
- missing upper text in connected balloons;
- bottom text residues.

**Expected:** Fixes improve at least one chapter without regressing the previous accepted chapter.

---

## Phase 9: Final Test Suite

Run focused Python tests:
```powershell
N:\TraduzAI\pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_vision_stack_ocr.py `
  pipeline/tests/test_ocr_no_easyocr.py `
  pipeline/tests/test_ocr_language_filter.py `
  pipeline/tests/test_notanother_component_mask.py `
  pipeline/tests/test_inpaint_mask_geometry.py `
  pipeline/tests/test_inpaint_region_strategy.py `
  pipeline/tests/test_vision_stack_inpainter.py `
  pipeline/tests/test_mask_builder.py `
  pipeline/tests/test_typesetting_layout.py `
  pipeline/tests/test_typesetting_renderer.py `
  pipeline/tests/test_render_geometry.py `
  pipeline/tests/test_export_gate.py `
  pipeline/tests/test_main_emit.py -q
```

Run frontend/Rust only if touched:
```powershell
npx tsc --noEmit
cd src-tauri
cargo test
```

---

## Agent Split For Execution

- **Agent OCR:** Phase 1 only. Owns Paddle/no-EasyOCR, language filtering, OCR tests.
- **Agent Mask:** Phase 2 only. Owns component mask builder and bbox removal from mask decisions.
- **Agent Inpaint:** Phase 3 and Phase 4. Owns ROI scheduler, snap-to-8, safe fill routing.
- **Agent Repo Boundary:** Phase 5. Owns deferred Bubble-Detector notes and no-op/disabled tests only if existing code references those adapters.
- **Agent Renderer/QA:** Phase 6 and Phase 7. Owns legibility, safe text boxes, export gate.
- **Main Integrator:** Reviews all patches, prevents conflicting edits, runs visual chapters one at a time.

Agents must not edit overlapping files without coordination. If two agents need `pipeline/vision_stack/runtime.py`, one owns the change and the other returns a patch note only.

---

## Completion Criteria

- No runtime path falls back to EasyOCR.
- For English-source chapters, Korean/CJK art fragments do not become random translated text.
- Inpaint masks are glyph/text-only and clipped by real BubbleMask when available.
- BBox is not used as a strong erase/fill mask.
- Fast/white fill cannot eat balloon outlines.
- ROI inpaint does not distort unrelated artwork.
- Typesetting does not render unreadably tiny text as success.
- `export_gate=BLOCK` cannot be reported as approved/successful.
- One Second ch1/ch2 visual debug is reviewed after implementation, not before.

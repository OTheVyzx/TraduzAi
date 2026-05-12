# Editor Style, Bitmap History, Instant Brush, and Layout Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix editor regressions so default text has no effects, font remains editable, bitmap tools are instant and undoable, automatic text color is legible, and rendered text fits inside the balloon without clipping.

**Architecture:** Unify text style policy across TypeScript, Rust schema normalization, and Python renderer. Treat bitmap edits like normal editor commands: optimistic canvas preview first, async persistence second, with before/after snapshots for undo/redo. Keep typesetting deterministic by improving bbox/font fit scoring instead of adding model-dependent behavior.

**Tech Stack:** React 19 + Zustand + Konva, Vitest, Playwright, Tauri v2 Rust commands, Python typesetter renderer.

---

## Current Findings

- `src/lib/stores/editorStore.ts` creates local text with black fill and no effects, but `src-tauri/src/commands/project_schema.rs::default_text_style()` still defaults to white fill, black outline, and `contorno_px: 2`.
- `src/lib/tauri.ts::hydrateTextLayer()` strips `contorno`, but preserves an old `contorno_px` if it exists. This can leave UI state inconsistent.
- `pipeline/typesetter/renderer.py::_canonical_render_style()` currently forces `fonte = ComicNeue-Bold.ttf`, `bold = True`, and disables some effects. That makes the font feel locked during render/export.
- `pipeline/typesetter/renderer.py::ensure_legible_plan()` automatically adds outline when contrast is low. This conflicts with the desired default of no outline/glow/shadow.
- `src/lib/editorHistory.ts` already has a `bitmap-stroke` command, but `src/lib/stores/editorStore.ts::applyWorkingBitmapRegion()` is currently a no-op, so Ctrl+Z cannot visually undo brush/recovery strokes.
- `src/components/editor/stage/useEditorStageController.ts::finishPaintStroke()` only does optimistic preview for `repairBrush`; normal brush and eraser wait for `applyBitmapStroke()`, so they feel slower.
- `src/lib/stores/editorStore.ts::applyBitmapStroke()` persists the bitmap but does not record a `bitmap-stroke` command with real before/after bytes.
- `pipeline/typesetter/renderer.py` already computes `safe_text_box`, `_fits_in_box()`, `render_bbox`, and `TEXT_CLIPPED` QA, but some paths still use an over-small `layout_bbox` or too-large font, producing clipped text even when OCR/translation are correct.

---

## Non-Negotiable Behavior

- New/default text style:
  - font: `ComicNeue-Bold.ttf`
  - fill: `#000000`
  - no outline: `contorno: ""`, `contorno_px: 0`
  - no glow: `glow: false`, `glow_px: 0`, `glow_cor: ""`
  - no shadow: `sombra: false`, `sombra_offset: [0, 0]`, `sombra_cor: ""`
  - font/color/effects must remain user-editable.
- Loading old projects must normalize missing/default style safely, but must preserve explicit user choices.
- Brush, eraser, and recovery brush must preview in the same frame perceptibly.
- Ctrl+Z/Ctrl+Y must work for text edits, bbox edits, create/delete text, brush, eraser, recovery brush, and lasso/mask edits.
- Export can still use internal bitmap layers, but the editor UX must not require toggling technical layers.
- Automatic contrast must not add outline/glow/shadow by default. For white balloons, black fill is the safe default.

---

## Task 1: Add a Single Text Style Policy

**Files:**
- Modify: `src/lib/editorTextStylePolicy.ts`
- Modify: `src/lib/tauri.ts`
- Modify: `src/lib/stores/editorStore.ts`
- Modify: `src-tauri/src/commands/project_schema.rs`
- Test: `src/lib/__tests__/editorTextStylePolicy.test.ts`
- Test: `src/lib/tauri.test.ts`
- Test: `src/lib/stores/__tests__/editorStoreHistory.test.ts`
- Test: add Rust unit test in `src-tauri/src/commands/project_schema.rs`

**Steps:**

1. Write failing Vitest cases:
   - missing style hydrates to Comic Neue Bold, black fill, no outline/glow/shadow.
   - explicit font like `Newrotic.ttf` is preserved.
   - explicit `contorno_px: 4`, `glow: true`, or `sombra: true` is preserved only when explicitly present.
   - legacy white text + outline from schema defaults is normalized only when it is a default-looking legacy style, not when user explicitly chose it.

2. Update `canonicalizeTextStyle()` to accept an options object:
   - `mode: "default" | "hydrate" | "preserve-explicit"`
   - default missing fields to the no-effects contract.
   - avoid forcing `fonte` if an explicit font exists.
   - avoid forcing `bold` if explicit `bold: false` exists.

3. Update `hydrateTextLayer()`:
   - stop hard-setting only `contorno: ""`; use the centralized policy.
   - normalize `contorno_px` to `0` when `contorno` is empty or legacy-default.
   - keep explicit user effects intact.

4. Update `defaultTextStyle()` in `editorStore.ts` to call or mirror the same canonical policy.

5. Update `default_text_style()` in `project_schema.rs`:
   - `fonte: "ComicNeue-Bold.ttf"`
   - `cor: "#000000"`
   - `contorno: ""`
   - `contorno_px: 0`
   - `bold: true`
   - all effects disabled.

6. Add Rust regression test that normalizing a text layer with no style gets the no-effects style.

7. Run:
   - `npm run test -- src/lib/__tests__/editorTextStylePolicy.test.ts src/lib/tauri.test.ts`
   - `cargo test default_text_style`
   - `npm run check`

---

## Task 2: Stop Renderer From Locking Font and Auto-Adding Effects

**Files:**
- Modify: `pipeline/typesetter/renderer.py`
- Test: add or update `pipeline/tests/test_typesetting_renderer.py`

**Steps:**

1. Write failing Python tests:
   - renderer preserves explicit `estilo["fonte"]`.
   - renderer preserves explicit `bold: false`.
   - default style has no outline/glow/shadow.
   - `ensure_legible_plan()` on white background changes white/missing text to black but does not add outline.

2. Replace `_canonical_render_style()` with a non-locking version:
   - only fill missing `fonte` with `ComicNeue-Bold.ttf`.
   - only fill missing `bold` with `True`.
   - set missing effects to disabled.
   - do not overwrite explicit font, color, bold, italic, outline, glow, or shadow.

3. Update `merge_group_style()` defaults:
   - default `contorno` to `""`, not `"#000000"`.
   - default `contorno_px` to `0`, not `2`.
   - default `cor` to `#000000`, not `#FFFFFF`.

4. Update `build_text_plan()` defaults:
   - `text_color` default `#000000`.
   - `outline_color` default `""`.
   - `outline_px` default `0`.

5. Update `ensure_legible_plan()`:
   - if background is light, choose dark fill.
   - if background is dark, choose light fill only when no explicit color is present or current color is unreadable.
   - do not add outline unless the user already has `outline_px > 0` or an explicit outline color.
   - when outline is explicit but low contrast, adjust outline color, not outline existence.

6. Run:
   - `python -m pytest pipeline/tests/test_typesetting_renderer.py -q`
   - If local pytest is unavailable, run the targeted renderer functions through a minimal Python script and record the limitation.

---

## Task 3: Make Bitmap Undo/Redo Real

**Files:**
- Modify: `src/lib/editorHistory.ts`
- Modify: `src/lib/stores/editorStore.ts`
- Modify: `src/components/editor/toolbar/UndoRedoControls.tsx`
- Test: `src/lib/__tests__/editorHistory.test.ts`
- Test: `src/lib/stores/__tests__/editorBitmapTools.test.ts`
- E2E: `e2e/editor-rebuild.spec.ts`

**Steps:**

1. Write failing unit tests:
   - `bitmap-stroke` undo applies `before` bytes.
   - redo applies `after` bytes.
   - missing bitmap cache returns a clear validation failure.
   - brush, eraser, recovery, and mask/lasso each record one undo command after mouseup.

2. Extend `bitmap-stroke` command metadata:
   - add `layerKey: "brush" | "mask" | "inpaint"`
   - keep `bbox`.
   - store bytes in `bitmapCache` by `commandId`.

3. Implement `applyWorkingBitmapRegion()` in `editorStore.ts`:
   - patch only the dirty bbox in the matching preview bitmap/image layer.
   - bump the right `bitmapLayerVersions` key.
   - update `currentPage.image_layers[layerKey].path` only when needed.
   - mark preview stale.

4. Add helper functions in the store/controller:
   - capture before bytes for dirty bbox before optimistic stroke.
   - capture after bytes after applying preview.
   - insert `bitmapCache.set(commandId, { before, after, byteLength, pageKey })`.
   - call `recordEditorCommand({ type: "bitmap-stroke", layerKey, bbox })`.

5. Update undo/redo labels:
   - brush: `Pincel`
   - eraser: `Borracha`
   - recovery: `Recuperacao`
   - mask/lasso: `Mascara`

6. Run:
   - `npm run test -- src/lib/__tests__/editorHistory.test.ts src/lib/stores/__tests__/editorBitmapTools.test.ts`
   - `npm run check`

---

## Task 4: Make Brush and Eraser Preview Instant

**Files:**
- Modify: `src/components/editor/stage/useEditorStageController.ts`
- Modify: `src/components/editor/stage/EditorStage.tsx`
- Create or modify: `src/components/editor/stage/bitmapStrokePreview.ts`
- Modify: `src/lib/stores/editorStore.ts`
- Test: `src/lib/stores/__tests__/editorBitmapTools.test.ts`
- E2E: `e2e/editor-rebuild.spec.ts`

**Steps:**

1. Write failing Playwright test:
   - draw with brush and assert a visible overlay appears before backend persistence resolves.
   - draw with eraser and assert the overlay updates immediately.
   - press Ctrl+Z and assert the preview returns to the prior pixels.

2. Create `bitmapStrokePreview.ts`:
   - use one offscreen canvas per page/layer.
   - apply stroke into local RGBA canvas for `brush`.
   - apply erase using `destination-out` for `brush`.
   - apply grayscale mask preview for `mask`.
   - reuse the existing recovery patch logic for `inpaint`, but route it through the same preview abstraction.
   - return `{ beforeBytes, afterBytes, dirty_bbox, objectUrlOrImage }`.

3. Update `finishPaintStroke()`:
   - do optimistic preview for `brush`, `eraser`, and `repairBrush`.
   - do not `await applyBitmapStroke()` for UI feedback.
   - persist in background with one serialized queue per page/layer.
   - on persistence success, swap preview to persisted image path/version without flicker.
   - on failure, show a small error and keep undoable local state until retry or reload.

4. Update `applyBitmapStroke()`:
   - accept optional `previewCommandId` or `bitmapSnapshot`.
   - record history after optimistic preview, not after backend roundtrip.
   - do not clear selected text or hide text overlays.

5. Run:
   - `npm run test -- src/lib/stores/__tests__/editorBitmapTools.test.ts`
   - `npm run test:e2e -- e2e/editor-rebuild.spec.ts --grep "brush|borracha|recuperacao" --workers=1`

---

## Task 5: Fix White Text on White Balloons

**Files:**
- Modify: `pipeline/vision_stack/runtime.py`
- Modify: `pipeline/project_writer.py` if it materializes style defaults there.
- Modify: `pipeline/typesetter/renderer.py`
- Test: `pipeline/tests/test_typesetting_renderer.py`
- Test: add/update `pipeline/tests/test_vision_stack_runtime.py`

**Steps:**

1. Write failing tests:
   - speech balloon with white/bright background gets `cor: "#000000"` when no explicit color exists.
   - narration/dark background can use white fill when needed.
   - explicit user/editor color is preserved.
   - no automatic outline is added for white balloons.

2. Locate the source of `estilo["cor"] = "#FFFFFF"` in `pipeline/vision_stack/runtime.py` and gate it:
   - only use white for dark textured/narration contexts.
   - for normal white speech balloons, use black.
   - store an optional marker like `style_source: "auto"` only if useful for distinguishing auto vs explicit later.

3. Ensure renderer contrast repair uses the same policy:
   - default black on light background.
   - default white on dark background only when fill is auto/missing.
   - no automatic outline unless explicit.

4. Run:
   - `python -m pytest pipeline/tests/test_vision_stack_runtime.py pipeline/tests/test_typesetting_renderer.py -q`

---

## Task 6: Improve Text Box and Font Fit So Text Does Not Clip

**Files:**
- Modify: `pipeline/typesetter/renderer.py`
- Modify: `pipeline/layout/balloon_layout.py` only if bbox source selection is wrong.
- Modify: `pipeline/vision_stack/runtime.py` only if `layout_bbox` is written too small.
- Test: `pipeline/tests/test_typesetting_renderer.py`
- E2E/fixture: update `e2e/editor-rebuild.spec.ts` only for editor-visible bbox regressions.

**Steps:**

1. Write failing renderer tests:
   - translated text longer than OCR text fits by shrinking font before clipping.
   - `render_bbox` stays inside `safe_text_box` with a small margin.
   - if `layout_bbox` is too small but `balloon_bbox` is larger and reliable, renderer uses the balloon capacity.
   - connected balloons still split by lobe and do not regress.

2. Add a bbox capacity resolver:
   - prefer reliable `balloon_bbox`/subregion for available area.
   - use `source_bbox`/`text_pixel_bbox` only as anchor, not as hard capacity, unless explicitly locked.
   - add minimum padding based on font size.

3. Tighten `_fits_in_box()`:
   - measure actual ink bbox with the selected font.
   - include outline/shadow only when explicit.
   - remove positive height tolerance for final accept; tolerance can be used for candidate search but final candidate must fit.

4. Update candidate scoring:
   - heavily penalize any candidate where `block_bbox` exceeds `safe_text_box`.
   - prefer a slightly smaller font over clipping.
   - keep readable minimums: speech >= 12, narration >= 12, lobe >= 11 if needed.

5. After rendering, run `_validate_render_fit()`:
   - if `TEXT_CLIPPED`, retry with a smaller size or expanded safe box if balloon capacity allows.
   - record QA flag only if retry fails.

6. Run:
   - `python -m pytest pipeline/tests/test_typesetting_renderer.py -q`
   - Run one known chapter/page fixture if available and compare `qa_flags` plus `render_bbox`.

---

## Task 7: End-to-End Regression Matrix

**Files:**
- Test-only changes as needed:
  - `src/lib/stores/__tests__/editorBitmapTools.test.ts`
  - `src/lib/__tests__/editorHistory.test.ts`
  - `src/lib/__tests__/editorTextStylePolicy.test.ts`
  - `e2e/editor-rebuild.spec.ts`
  - `pipeline/tests/test_typesetting_renderer.py`
  - `pipeline/tests/test_vision_stack_runtime.py`

**Acceptance Tests:**

1. Frontend/unit:
   - `npm run check`
   - `npm run test -- src/lib/__tests__/editorTextStylePolicy.test.ts src/lib/__tests__/editorHistory.test.ts src/lib/stores/__tests__/editorBitmapTools.test.ts src/lib/tauri.test.ts`

2. Rust:
   - `cargo check`
   - `cargo test default_text_style`
   - existing bitmap tests around brush/recovery should still pass.

3. Python:
   - `python -m pytest pipeline/tests/test_typesetting_renderer.py pipeline/tests/test_vision_stack_runtime.py -q`

4. Playwright:
   - `npm run test:e2e -- e2e/editor-rebuild.spec.ts --grep "Konva usa fundo|brush|borracha|recuperacao" --workers=1`

**Manual Acceptance:**

- Create a new text box: it appears black, Comic Neue Bold, no outline, no glow, no shadow.
- Change font manually: selected font persists in editor and render/export.
- Paint with brush: visible immediately, Ctrl+Z removes it, Ctrl+Y restores it.
- Use eraser: visible immediately, Ctrl+Z restores erased paint.
- Use recovery brush: visible immediately, text stays visible, Ctrl+Z restores previous inpaint pixels.
- Process a white speech balloon: text is black, not white.
- Process a long translation: font shrinks or box capacity expands; text does not get cut inside the balloon.

---

## Implementation Order

1. Style policy first, because it removes the visible regression and prevents more white/outline defaults from being written.
2. Renderer font/effect policy second, because otherwise export can still contradict editor behavior.
3. Bitmap undo/redo third, because instant preview needs the same snapshot model.
4. Instant brush/eraser fourth, using the history snapshots.
5. White text and layout fit fifth/sixth, with Python tests and one real page validation.
6. Run full targeted validation and update this plan with any test limitation.

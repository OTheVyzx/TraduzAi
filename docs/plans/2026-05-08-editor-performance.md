# Editor Performance Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the editor feel near-instant for page navigation, text selection/dragging, brush strokes, layer toggles, and final-preview state changes, without changing the public project format.

**Architecture:** Keep Konva as the main interaction surface, but move transient pointer work out of React state, reduce Zustand fan-out, cache decoded image resources, and defer heavy preview/render work to cancellable background queues. Bitmap edits remain persisted through Tauri/Rust, with smaller region-based updates where the current full-image path is too expensive.

**Tech Stack:** React 19, TypeScript, Zustand, React-Konva/Konva, Tauri v2 IPC, Rust image crate, Playwright, Vitest.

---

## Current Findings

- `src/components/editor/stage/useEditorStageController.ts` updates React state on pointer movement for `paintStroke`, `blockDraft`, `maskInProgress`, and `cursorPoint`; this can re-render the full Stage during rapid input.
- `src/components/editor/stage/EditorStage.tsx` maps every text layer on render and mixes static image layers, live stroke previews, selection, hover, and text transforms in the same render path.
- `useObjectUrl()` in `useEditorStageController.ts` reloads bytes and creates object URLs per path/version; there is no shared path+version cache or decoded image reuse.
- `src/components/editor/stage/recoveryComposite.ts` performs full-canvas pixel composition on the main thread when the recovery layer is visible. After the recovery brush fix, this should become an inspection-only path, not the normal brush path.
- `src/lib/stores/editorStore.ts` stores broad objects such as `pendingEdits`, `currentPage`, `renderPreviewCacheByPageKey`, and bitmap versions; several consumers subscribe to large slices and re-render even when only one layer changed.
- Bitmap strokes are committed through Tauri on mouseup, but the current Rust functions decode/write whole bitmap layers. Recovery now also updates the inpaint image, so region bounds matter for speed.
- The local Koharu reference (`D:\koharu\ui\hooks\useTextBlocks.ts`) keeps document data in query cache and UI selection separately. TraduzAi can borrow the separation idea without replacing the current project schema.

## Acceptance Targets

- Pointer move during select/brush/eraser stays under 16 ms p95 on a 2k page.
- Visible brush preview follows the cursor with no dropped interaction for normal strokes.
- Brush mouseup commit shows the updated layer in under 100 ms for small strokes and under 250 ms for large strokes.
- Text selection and drag start feel immediate; only the moved/selected text node re-renders during transform.
- Page switch with cached images completes in under 150 ms after data is loaded.
- No automatic final-preview render or export check blocks typing, dragging, or painting.
- Playwright editor smoke and focused store tests stay green.

## Task 1: Add Editor Performance Instrumentation

**Files:**
- Modify: `src/components/editor/stage/useEditorStageController.ts`
- Modify: `src/components/editor/stage/EditorStage.tsx`
- Create: `src/lib/editorPerformance.ts`
- Test: `src/lib/__tests__/editorPerformance.test.ts`

**Steps:**
1. Add a tiny dev-only performance helper that records marks for `pointermove`, `paint-preview`, `bitmap-commit`, `image-load`, `page-switch`, and `render-preview-schedule`.
2. Add unit tests for rolling p95 calculation and disabled production behavior.
3. Wire marks around Stage mouse move, brush commit start/end, and image load start/end.
4. In E2E mode, expose counters through `data-testid="editor-perf-state"` so Playwright can assert no pathological regressions.
5. Run `npm run test -- src/lib/__tests__/editorPerformance.test.ts` and `npm run check`.

## Task 2: Split Store Selectors And Reduce Fan-Out

**Files:**
- Modify: `src/lib/stores/editorStore.ts`
- Modify: `src/components/editor/stage/useEditorStageController.ts`
- Modify: `src/components/editor/stage/EditorStage.tsx`
- Test: `src/lib/stores/__tests__/editorStoreHistory.test.ts`

**Steps:**
1. Replace broad selectors that return whole objects with scalar selectors or shallow selectors.
2. Add memoized helpers keyed by `pageKey`, `layerId`, and `version` for text-layer materialization.
3. Keep selection, hover, cursor, and bitmap version state independent so a hover change does not rebuild all layers.
4. Add tests that changing one pending text layer does not change unrelated layer references.
5. Run focused store tests and `npm run check`.

## Task 3: Move Live Pointer State Out Of React Render Path

**Files:**
- Modify: `src/components/editor/stage/useEditorStageController.ts`
- Modify: `src/components/editor/stage/EditorStage.tsx`
- Create: `src/components/editor/stage/useImperativeStrokePreview.ts`

**Steps:**
1. Store live stroke points in refs while dragging.
2. Draw the live stroke through an imperative Konva `Line` ref or a dedicated transient canvas layer.
3. Commit a compact stroke array to the store only on mouseup.
4. Keep `blockDraft` and `maskInProgress` in refs during pointer movement, with throttled visual updates through `requestAnimationFrame`.
5. Verify brush, eraser, recovery brush, block creation, and freehand lasso behavior manually and through the editor E2E smoke.

## Task 4: Optimize Konva Layer Rendering

**Files:**
- Modify: `src/components/editor/stage/EditorTextLayer.tsx`
- Modify: `src/components/editor/stage/EditorStage.tsx`
- Modify: `src/components/editor/stage/EditorTransformer.tsx`

**Steps:**
1. Wrap text-layer nodes with `React.memo` and a custom equality check over geometry, style, selection, hover, visibility, and lock state.
2. Put static bitmap/background layers on non-listening layers where possible.
3. Keep interactive text nodes in a separate layer from bitmap overlays and live preview strokes.
4. Avoid calling `paintStroke.flatMap()` on every render by precomputing preview points in the imperative stroke hook.
5. Verify text selection, hover, undo/redo, and transform still work.

## Task 5: Cache Image Bytes And Decoded Images

**Files:**
- Modify: `src/components/editor/stage/useEditorStageController.ts`
- Create: `src/lib/editorImageCache.ts`
- Test: `src/lib/__tests__/editorImageCache.test.ts`

**Steps:**
1. Implement a small LRU cache keyed by normalized path plus version.
2. Reuse object URLs and decoded image metadata while a page stays active or adjacent.
3. Revoke object URLs only when evicted, not on every render.
4. Prefer `createImageBitmap` when available, with `HTMLImageElement` fallback for Konva compatibility.
5. Add tests for cache hit, version bust, eviction, and revoke behavior.

## Task 6: Make Preview Scheduling Non-Blocking

**Files:**
- Modify: `src/lib/stores/editorStore.ts`
- Modify: `src/pages/Preview.tsx`
- Modify: `src/components/editor/EditorToolbar.tsx` or the current render-preview trigger file

**Steps:**
1. Keep stale-preview marking cheap and synchronous.
2. Move render-preview requests into a cancellable queue with backpressure: one active job, latest page wins, cancelled stale jobs do not update UI.
3. Do not auto-render while dragging, typing, painting, or transforming.
4. Keep export allowed per the latest product request, but make the UI clear when the preview is from an older render.
5. Add tests for queue cancellation and latest-result-wins behavior.

## Task 7: Region-Based Bitmap Persistence

**Files:**
- Modify: `src-tauri/src/commands/project.rs`
- Modify: `src/lib/stores/editorStore.ts`
- Test: `src-tauri/src/commands/project.rs` Rust unit tests

**Steps:**
1. Compute a stroke bounding box in TypeScript and send it with bitmap update configs.
2. In Rust, crop processing to the stroke bounding box plus brush radius where the image crate allows it.
3. For recovery brush, copy original pixels into inpaint only inside the dirty rectangle.
4. Keep full-image fallback for malformed dimensions or old callers.
5. Run `cargo test` for the bitmap helpers and `cargo check`.

## Task 8: Add Responsiveness Regression Tests

**Files:**
- Modify: `e2e/editor-rebuild.spec.ts`
- Create: `e2e/editor-performance.spec.ts`

**Steps:**
1. Add a Playwright scenario for rapid text drag and assert the final bbox changes without console errors.
2. Add a brush scenario that performs a long stroke and asserts the commit returns within the configured budget in E2E mode.
3. Add a page-switch scenario that checks cached image reuse through the perf state.
4. Keep the tests budget-based but tolerant enough for CI and Windows dev variance.
5. Run `npx playwright test e2e/editor-performance.spec.ts --project=chromium`.

## Task 9: Final Verification

**Files:**
- No code changes unless a verification failure identifies a concrete fix.

**Steps:**
1. Run `npm run check`.
2. Run `npm run test -- src/lib/stores/__tests__/editorStoreHistory.test.ts src/lib/stores/__tests__/editorRenderPreviewCache.test.ts`.
3. Run `npx playwright test e2e/editor-rebuild.spec.ts --project=chromium`.
4. Run `cargo check`.
5. Run `git diff --check`.
6. Record measured before/after numbers in this plan or a follow-up status file.

## Rollout Notes

- Do not replace the editor architecture or project schema.
- Avoid new visual panels unless guarded by a dev/E2E flag.
- Keep user-visible text in Brazilian Portuguese.
- Make each task independently shippable; stop and write a status note if a phase cannot be completed in one session.

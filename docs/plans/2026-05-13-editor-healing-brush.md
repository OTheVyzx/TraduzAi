# Editor Healing Brush Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an editor-only healing brush that runs a new local inpaint on the painted region and updates only the current page in a reversible, non-destructive editor flow.

**Architecture:** Reuse the existing Konva Stage brush flow, `reinpaintBrush` tool mode, `update_reinpaint_region`, and `reinpaint_page` regional IPC instead of adding a parallel editor path. The first version should generate a stroke mask in image coordinates, send only the dirty ROI to the backend, refresh the `inpaint` bitmap layer, mark preview/export stale, and record a bitmap history command for undo/redo. The original/base image must stay untouched.

**Tech Stack:** React 19, TypeScript, Zustand editor store, Konva/react-konva, Tauri/Rust commands, Python sidecar inpaint path, Vitest, Playwright, Cargo.

---

## Execution Status - 2026-05-13

Implemented in the current dirty checkout without cleaning or reverting unrelated files.

- `reinpaintBrush` now behaves as the editor "Pincel corretor" and runs a fresh regional inpaint from a full-page painted mask.
- The frontend keeps `reinpaintPage(...) => Promise<string>` compatible and uses explicit `writeHealingMask(...)` / `healInpaintRegion(...)` bindings for the healing brush.
- The store action queues healing strokes, marks `isHealingBrushApplying` before awaited backend work, updates the `inpaint` layer, marks preview stale, and records undoable before/after inpaint path snapshots.
- Tauri validates mask containment under `editor_cache/healing_masks`, snapshots before/after inpaint as PNG, updates only the current page `image_layers.inpaint.path`, and returns a regional result.
- Python regional reinpaint now accepts the external full-page mask and crops it to the ROI before passing it to the existing inpaint mask builder.

Validation run:

- `npx tsc --noEmit` - passed.
- `npx vitest run src/lib/stores/__tests__/editorBitmapTools.test.ts src/lib/__tests__/healingBrushMask.test.ts` - passed, 11 tests.
- `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_editor_healing_brush.py -q` - passed, 2 tests.
- `cargo check` in `src-tauri` - passed.
- `cargo fmt --check` still reports unrelated pre-existing formatting drift in `src-tauri/src/commands/project.rs` outside the healing brush sections.

## Scope

- Editor only.
- No automatic pipeline cleanup in this phase.
- No full-page reinpaint triggered by the brush.
- No rewrite of the existing brush, lasso, recovery, or preview systems.
- Keep `base`/original immutable; only the editor `inpaint` layer changes.

## Existing Contracts To Preserve

- Repo root: `N:\TraduzAI\TraduzAi`.
- Current Stage flow: `src/components/editor/stage/EditorStage.tsx`.
- Pointer/paint controller: `src/components/editor/stage/useEditorStageController.ts`.
- Editor state and history: `src/lib/stores/editorStore.ts`, `src/lib/editorHistory.ts`.
- Frontend IPC binding: `src/lib/tauri.ts`.
- Tauri bitmap commands: `src-tauri/src/commands/project.rs`.
- Tauri regional inpaint command: `src-tauri/src/commands/pipeline.rs`.
- Python sidecar must keep JSON-line output contract.
- Existing dirty worktree must be preserved. Do not run `git add .`; stage only files touched for this feature.

## Design Decision

Use the existing `reinpaintBrush` UI as the user-facing "Pincel corretor", but change its persistence behavior from "copy cached previous inpaint pixels into the current inpaint layer" to "run regional inpaint for the painted ROI".

Reason: the user expects a new inpaint. Copying from cache is useful for recovery, but it is not the Photoshop-like spot healing behavior. A true healing brush needs a fresh inpaint over the mask and ROI.

## Reviewed Adjustments

- Keep the existing `reinpaintPage(...) => Promise<string>` binding compatible for current callers. Add a separate `healInpaintRegion(...)` wrapper/result for the editor brush instead of changing old return semantics.
- Set the healing busy/queue state before any awaited work. Do not allow two strokes to pass validation before the first one marks the operation active.
- Use a concrete MVP history policy: capture full inpaint layer data before and after the IPC when possible; optimize to ROI bytes later.
- The temporary mask contract is fixed: full-page PNG, black background, white opaque stroke, image-space coordinates.
- Tauri path containment for `mask_path` is mandatory before the UI is connected. The mask must resolve under `<project_dir>/editor_cache/healing_masks/`.
- Regional inpaint padding is explicit: `padding = max(32, brushSize * 2)`, clamped to the page bounds.

## Data Flow

1. User selects `Pincel corretor`.
2. User paints on the active Konva Stage.
3. `useEditorStageController.ts` records stroke points in image coordinates.
4. On mouseup, controller computes `dirty_bbox` with padding.
5. Frontend creates or persists a temporary mask for the painted stroke.
6. Store calls a new explicit action such as `healPaintedRegion(...)`.
7. Tauri receives `project_path`, `page_index`, `bbox`, `mask_path`.
8. Tauri calls existing `reinpaint_page_with_region(...)`.
9. Python updates only the page's `inpaint` asset.
10. Store reloads current page, bumps `inpaint` version, marks render preview stale, and records undo data.

## Risk Controls

- Gate the new behavior behind `reinpaintBrush` only; leave `brush`, `mask`, `eraser`, and `repairBrush` unchanged.
- Keep an explicit busy flag such as `isHealingBrushApplying` or reuse `isReinpainting` carefully so repeated strokes cannot race.
- Queue brush applications per page. Do not allow two healing strokes to write the same `inpaint` file at the same time.
- Use ROI padding large enough for inpaint context, but clamp to image bounds.
- If regional inpaint fails, keep the previous bitmap visible and show a non-fatal editor error.
- Invalidate preview/export after every successful healing stroke.
- Undo must restore the prior `inpaint` bytes for the affected bbox, not the full project state.

---

## Task 1: Baseline Audit And Safety Snapshot

**Files:**
- Read: `src/components/editor/stage/useEditorStageController.ts`
- Read: `src/lib/stores/editorStore.ts`
- Read: `src/lib/tauri.ts`
- Read: `src-tauri/src/commands/project.rs`
- Read: `src-tauri/src/commands/pipeline.rs`
- Read: `pipeline/main.py`

**Step 1: Record current git state**

Run:

```powershell
git status --short --branch
```

Expected: dirty worktree is visible. Save the output in the implementation notes. Do not clean, reset, or revert unrelated changes.

**Step 2: Confirm existing regional inpaint arguments**

Run:

```powershell
Select-String -Path src-tauri\src\commands\pipeline.rs,src-tauri\src\commands\project.rs,pipeline\main.py -Pattern "reinpaint_page|mask_path|region_bbox|bbox|--reinpaint-page" -Context 2,4
```

Expected: `reinpaint_page_with_region` can receive region config and pass bbox/mask data to the sidecar, or the missing piece is identified precisely.

**Step 3: Confirm current brush behavior**

Run:

```powershell
Select-String -Path src\components\editor\stage\useEditorStageController.ts,src\lib\stores\editorStore.ts -Pattern "reinpaintBrush|updateReinpaintRegion|applyBitmapStroke|bitmap-stroke" -Context 2,4
```

Expected: existing `reinpaintBrush` currently persists through `applyBitmapStroke(... layerKey: "reinpaint")` and `update_reinpaint_region`.

**Step 4: Stop if assumptions are false**

If `reinpaint_page_with_region` cannot receive a mask or bbox, do not improvise in the frontend. Add the missing Tauri/Python contract first in Task 3.

---

## Task 2: Add Focused Tests For Editor Store Contract

**Files:**
- Modify or create: `src/lib/stores/__tests__/editorStoreHealingBrush.test.ts`
- Modify: `src/lib/tauri.ts` only if mocks need a new function

**Step 1: Write failing test for successful healing**

Test behavior:

- set current page with an `inpaint` layer;
- mock a regional healing IPC result;
- call the new store action with `bbox` and `maskPath`;
- assert `inpaint` bitmap version increments;
- assert render preview is stale;
- assert busy flag returns to false.

Suggested test shape:

```ts
it("applies a regional healing brush stroke to the inpaint layer", async () => {
  const store = useEditorStore.getState();
  await store.healPaintedRegion({
    bbox: [10, 20, 80, 90],
    maskPath: "editor/masks/heal-1.png",
  });

  expect(useEditorStore.getState().isHealingBrushApplying).toBe(false);
  expect(useEditorStore.getState().bitmapLayerVersions.inpaint).toBeGreaterThan(0);
});
```

**Step 2: Write failing test for failure handling**

Test behavior:

- mocked IPC rejects;
- busy flag resets;
- previous `inpaint` path/version is not overwritten;
- `pageActionError` or a dedicated healing error is set.

**Step 3: Run tests and verify failure**

Run:

```powershell
npx vitest run src/lib/stores/__tests__/editorStoreHealingBrush.test.ts
```

Expected: FAIL because `healPaintedRegion` and/or `isHealingBrushApplying` do not exist yet.

---

## Task 3: Add Explicit Frontend IPC For Regional Healing

**Files:**
- Modify: `src/lib/tauri.ts`
- Modify if needed: `src/lib/tauriMock.ts` or existing mock block inside `src/lib/tauri.ts`

**Step 1: Add a typed function**

Add a frontend binding:

```ts
export type RegionalInpaintResult = {
  page_index: number;
  inpaint_path: string;
  bbox: [number, number, number, number];
};

export async function healInpaintRegion(config: {
  project_path: string;
  page_index: number;
  bbox: [number, number, number, number];
  mask_path: string;
}): Promise<RegionalInpaintResult> {
  if (isE2E()) return tauriMock.healInpaintRegion(config);
  const inpaint_path = await reinpaintPage({
    project_path: config.project_path,
    page_index: config.page_index,
    bbox: config.bbox,
    mask_path: config.mask_path,
  });
  return { page_index: config.page_index, inpaint_path, bbox: config.bbox };
}
```

Use the existing `reinpaintPage` command if its TypeScript type already supports `bbox` and `mask_path`; otherwise extend the argument type without changing the command name or its `Promise<string>` return for existing callers.

**Step 2: Add E2E mock**

The mock should return a plausible inpaint path, not mutate unrelated state.

Expected result: Playwright can exercise the UI without spawning the Python sidecar.

**Step 3: Run type check**

Run:

```powershell
npx tsc --noEmit
```

Expected: PASS for TypeScript binding changes.

---

## Task 4: Add Store Action For Healing Brush

**Files:**
- Modify: `src/lib/stores/editorStore.ts`
- Test: `src/lib/stores/__tests__/editorStoreHealingBrush.test.ts`

**Step 1: Add state and action types**

Add:

```ts
isHealingBrushApplying: boolean;
healingBrushError: string | null;
healPaintedRegion: (payload: {
  bbox: Bbox;
  maskPath: string;
}) => Promise<void>;
```

**Step 2: Implement guardrails**

The action must:

- return early when no `projectPath` or no `currentPage`;
- enqueue or refuse if `isHealingBrushApplying`, `isReinpainting`, or `activePageAction` is already busy;
- mark/enqueue the operation before any awaited work;
- call `commitEdits()` before image mutation;
- set `isHealingBrushApplying: true`;
- call `healInpaintRegion`;
- reload current page or patch the `inpaint` path;
- call `bumpBitmapLayerVersion("inpaint")`;
- call `markRenderPreviewStale(currentPageKey())`;
- set `viewMode: "inpainted"`;
- clear busy flag in `finally`;
- preserve previous visible bitmap on error.

**Step 3: Use existing history model**

Before calling IPC, capture the previous inpaint region bytes if the existing bitmap cache API supports it. If the current cache model only supports data URLs, capture a full inpaint data URL for MVP and leave a TODO to optimize to ROI bytes later.

Do not skip undo/redo entirely. If ROI history is too risky in this pass, keep full inpaint snapshot for correctness first.

**Step 4: Run focused tests**

Run:

```powershell
npx vitest run src/lib/stores/__tests__/editorStoreHealingBrush.test.ts
```

Expected: PASS.

---

## Task 5: Generate A Real Stroke Mask From The Stage

**Files:**
- Modify: `src/components/editor/stage/useEditorStageController.ts`
- Create if helpful: `src/components/editor/stage/healingBrushMask.ts`
- Test if helper created: `src/components/editor/stage/__tests__/healingBrushMask.test.ts`

**Step 1: Extract mask creation helper**

Create a helper that accepts:

```ts
{
  width: number;
  height: number;
  stroke: [number, number][];
  brushSize: number;
  dirtyBBox: [number, number, number, number];
}
```

It should return a PNG data URL or bytes for the painted mask. Use white pixels on transparent/black background, matching backend mask expectations.
For MVP the mask is full-page with a black background and a white opaque stroke.

**Step 2: Persist temporary mask safely**

Preferred path: add a Tauri command to write a temporary healing mask under the current project editor/cache folder and return its path.

Fallback path: reuse existing `writeMaskFromPng` only if it can write a temporary mask without replacing the user's persistent `mask` layer. Do not overwrite the visible `mask` layer for healing brush.

**Step 3: Add helper tests**

Assert:

- empty stroke returns null/error;
- bbox is clamped;
- mask dimensions match page dimensions or agreed ROI dimensions;
- mask contains non-empty white pixels inside stroke.

**Step 4: Run tests**

Run:

```powershell
npx vitest run src/components/editor/stage/__tests__/healingBrushMask.test.ts
```

Expected: PASS.

---

## Task 6: Wire `reinpaintBrush` To New Healing Action

**Files:**
- Modify: `src/components/editor/stage/useEditorStageController.ts`
- Modify: `src/pages/Editor.tsx`
- Modify: `src/components/editor/stage/EditorPaintCursor.tsx` only if label/color needs cleanup

**Step 1: Rename visible UI copy**

Use plain UI text:

- `Pincel corretor`
- tooltip: `Corrigir area pintada`
- busy label: `Corrigindo`

Avoid technical text like `reinpaint`, `ROI`, or `cache` in the UI.

**Step 2: Replace persistence branch**

In the `strokeToolMode === "reinpaintBrush"` branch:

- keep the immediate visual stroke overlay;
- remove cache-copy persistence as the final behavior;
- calculate `dirty_bbox` with `padding = max(32, brushSize * 2)`, clamped to image bounds;
- create/persist the temporary healing mask;
- call `healPaintedRegion({ bbox: dirty_bbox, maskPath })`;
- clear preview patch after success/failure;
- keep queueing so strokes run sequentially.

**Step 3: Keep recovery brush unchanged**

Do not alter the `repairBrush` branch. It has a different purpose: restoring from original/history, not generating new inpaint.

**Step 4: Manual smoke**

Run the app, open an existing project, select `Pincel corretor`, paint a small area, and confirm:

- the app does not navigate away;
- the page remains responsive;
- only current page refreshes;
- translated preview becomes stale;
- undo restores previous inpaint.

---

## Task 7: Add Or Confirm Tauri Temporary Mask Command

**Files:**
- Modify if needed: `src-tauri/src/commands/project.rs`
- Modify if needed: `src-tauri/src/lib.rs`
- Modify if needed: `src-tauri/capabilities/default.json`
- Modify if needed: `src-tauri/gen/schemas/capabilities.json`
- Modify: `src/lib/tauri.ts`

**Step 1: Prefer project-local cache path**

Command shape:

```rust
#[derive(Debug, Deserialize)]
pub struct WriteHealingMaskConfig {
    pub project_path: String,
    pub page_index: usize,
    pub png_data: String,
    pub bbox: Option<[u32; 4]>,
}
```

Return absolute path to a temp PNG such as:

```text
<project_dir>/editor_cache/healing_masks/page-0001/<uuid>.png
```

**Step 2: Do not modify `project.json` for temporary masks**

The temporary mask is an input artifact for the command, not a user-visible layer. The successful inpaint mutation updates the normal inpaint layer/project metadata.

**Step 3: Add Rust unit-level coverage where practical**

If existing Rust command tests are not available, keep this validation to `cargo check` plus a manual command path smoke.

**Step 4: Run Rust check**

Run:

```powershell
cargo check
```

from:

```powershell
cd src-tauri
```

Expected: PASS.

---

## Task 8: Verify Python Regional Inpaint Honors The Mask

**Files:**
- Modify only if required: `pipeline/main.py`
- Modify only if required: `pipeline/vision_stack/runtime.py`
- Modify only if required: `pipeline/inpainter/`
- Test: `pipeline/tests/test_editor_healing_brush.py`

**Step 1: Write Python regression test**

Create a synthetic page with:

- white background;
- a small dark stain;
- a mask over only the stain;
- bbox tightly around the stain.

Run regional reinpaint and assert:

- pixels outside bbox are unchanged;
- masked area changes;
- output inpaint path exists;
- project metadata remains valid.

**Step 2: Run focused Python test**

Run:

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_editor_healing_brush.py -q
```

Expected: FAIL first if the current sidecar does not honor editor masks, then PASS after the minimal fix.

**Step 3: Keep algorithm conservative**

Do not change global inpaint heuristics unless the test proves the regional path is wrong. The brush should pass a precise mask; it should not weaken automatic pipeline quality.

---

## Task 9: Add Playwright Coverage For Editor UI

**Files:**
- Modify: `e2e/editor-rebuild.spec.ts`

**Step 1: Add scenario**

Test flow:

- open editor fixture;
- choose `Pincel corretor`;
- drag a short stroke on the Stage;
- wait for mocked healing command;
- assert `editor-view-inpainted` or Stage `data-base-kind="inpaint"`;
- assert no full page navigation;
- assert undo button can restore previous state.

**Step 2: Run focused E2E**

Run:

```powershell
npx playwright test e2e/editor-rebuild.spec.ts --grep "pincel corretor"
```

Expected: PASS.

---

## Task 10: Final Validation Bundle

Run these from `N:\TraduzAI\TraduzAi`:

```powershell
npx tsc --noEmit
npx vitest run src/lib/stores/__tests__/editorStoreHealingBrush.test.ts src/components/editor/stage/__tests__/healingBrushMask.test.ts
npx playwright test e2e/editor-rebuild.spec.ts --grep "pincel corretor"
```

Run these from `N:\TraduzAI\TraduzAi\src-tauri`:

```powershell
cargo check
```

Run this from `N:\TraduzAI\TraduzAi`:

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_editor_healing_brush.py pipeline/tests/test_vision_stack_inpainter.py pipeline/tests/test_mask_builder.py -q
```

Expected:

- TypeScript passes.
- Focused Vitest passes.
- Focused Playwright passes.
- Cargo passes.
- Python focused inpaint tests pass.
- Manual editor smoke produces a visible corrected area and undo works.

---

## Commit Plan

Because this checkout is already dirty, use narrow staging only:

```powershell
git status --short
git diff -- docs/plans/2026-05-13-editor-healing-brush.md
git diff -- src/lib/tauri.ts src/lib/stores/editorStore.ts src/components/editor/stage/useEditorStageController.ts src/pages/Editor.tsx
git diff -- src-tauri/src/commands/project.rs src-tauri/src/commands/pipeline.rs src-tauri/src/lib.rs
git diff -- pipeline/main.py pipeline/vision_stack/runtime.py pipeline/inpainter
```

Stage only files changed for this feature:

```powershell
git add docs/plans/2026-05-13-editor-healing-brush.md
git add src/lib/tauri.ts src/lib/stores/editorStore.ts
git add src/components/editor/stage/useEditorStageController.ts src/pages/Editor.tsx
git add src-tauri/src/commands/project.rs src-tauri/src/lib.rs src-tauri/capabilities/default.json src-tauri/gen/schemas/capabilities.json
git add pipeline/main.py pipeline/vision_stack/runtime.py pipeline/tests/test_editor_healing_brush.py
git add e2e/editor-rebuild.spec.ts
git commit -m "feat: add editor healing brush"
```

If any listed file contains unrelated user edits, split the patch manually and stage only the feature hunks.

---

## Rollback Plan

If the healing brush causes instability:

1. Hide only the `Pincel corretor` UI entry.
2. Leave backend commands in place if they are covered and harmless.
3. Keep `repairBrush`, `brush`, `mask`, and `eraser` unchanged.
4. Restore `reinpaintBrush` behavior only if the new action is the source of the bug.
5. Do not revert unrelated dirty files.

## Definition Of Done

- `Pincel corretor` runs a new regional inpaint on mouseup.
- Original/base image is untouched.
- Only active page is changed.
- Preview/export state becomes stale after correction.
- Undo/redo restores the inpaint state.
- Brush actions are queued and cannot corrupt the inpaint layer through concurrent writes.
- Focused TS, Rust, Python, and Playwright checks pass or failures are documented with exact blocker text.

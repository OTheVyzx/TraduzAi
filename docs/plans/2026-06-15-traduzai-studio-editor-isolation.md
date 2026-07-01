# TraduzAI Studio Editor Isolation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform TraduzAI Studio into a separate app with its own isolated editor runtime, using the current normal-app editor as the initial copied base.

**Architecture:** The current editor is copied into `studio/src/current-editor/` and then made self-contained. The normal app remains unchanged. Studio owns its Tauri commands, project store, backend adapter, styles, assets, tests, and package config; after the migration, `studio/` must not import `../src` or depend on `"traduzai": "file:.."`.

**Tech Stack:** Tauri v2, React 19, TypeScript, Vite, Tailwind CSS, Zustand, Konva/react-konva, Vitest, Playwright.

---

## Non-Negotiable Rules

- Do not refactor the normal app to serve Studio.
- Do not keep `../../../src`, `../../../../src`, `../src`, `src/editor-shared`, or package dependency `"traduzai": "file:.."` in the final Studio runtime.
- Keep the normal app behavior unchanged; only run normal-app tests as canaries.
- The Studio editable model remains `paginas[] + image_layers + text_layers`, with aliases such as `textos`, `traduzido`, `translated`, `estilo`, `style`, `arquivo_original`, and `arquivo_traduzido` regenerated at boundaries.
- `v12` remains an import/projection format, not the Studio editor source of truth.
- Pipeline actions that are not implemented in Studio must fail or show clear "not connected" status; they must not silently call the normal app Tauri backend.

## Current Findings To Preserve

- `studio/package.json` currently depends on `"traduzai": "file:.."`.
- `studio/vite.config.ts` currently shims `../src/lib/editorBackend.ts` and allows access to the repo parent.
- `studio/tailwind.config.js` currently imports `../tailwind.config.js` and scans `../src`.
- `studio/src/main.tsx` currently imports `../../src/styles/globals.css`.
- `studio/src/editor/StudioSharedEditor.tsx` currently imports from `../../../src/editor-shared`.
- `studio/src/shims/currentEditorBackend.ts` currently imports from `../../../src/lib/editorBackend`.
- `studio/src/export/psd.ts` currently imports `../../../src/lib/imageSource`.
- `studio/src/editor/__tests__/layeredBitmapCanvas.test.ts` currently imports `../../../../src/editor-shared`.

---

## Task 1: Baseline And Failure Ledger

**Files:**
- Read: `package.json`
- Read: `studio/package.json`
- Read: `studio/src-tauri/Cargo.toml`
- Create: `docs/plans/2026-06-15-studio-editor-isolation-baseline.md`

**Step 1: Capture dirty worktree**

Run:

```powershell
git status --short
```

Expected: dirty tree is allowed. Record only files relevant to Studio/editor isolation.

**Step 2: Run Studio tests**

Run:

```powershell
npm --prefix studio test
```

Expected: PASS, or document exact pre-existing failures in the baseline note.

**Step 3: Run Studio Rust tests**

Run:

```powershell
cargo test --manifest-path studio/src-tauri/Cargo.toml
```

Expected: PASS, or document exact pre-existing failures.

**Step 4: Run normal editor canaries**

Run:

```powershell
npm run test -- src/lib/__tests__/editorHistory.test.ts src/lib/stores/__tests__/editorStoreHistory.test.ts src/lib/__tests__/editorOps.test.ts
```

Expected: PASS, or document exact pre-existing failures.

**Step 5: Write baseline note**

Create `docs/plans/2026-06-15-studio-editor-isolation-baseline.md` with commands, result, and pre-existing failures.

**Step 6: Commit**

Only commit if the user requested commits.

---

## Task 2: Add A Local Studio Editor Facade

**Files:**
- Create: `studio/src/current-editor/index.ts`
- Modify: `studio/src/editor/StudioSharedEditor.tsx`

**Step 1: Create temporary facade**

Create `studio/src/current-editor/index.ts`:

```ts
export {
  EditorStage,
  LayersPanel,
  PageThumbnails,
  ToolSidebar,
  UndoRedoControls,
  ZoomControls,
  useAppStore,
  useEditorStore,
} from "../../../src/editor-shared";

export type { Project, TextLayerStyle } from "../../../src/editor-shared";
```

This is intentionally temporary. Its job is to concentrate root-app imports in one place before copying the runtime.

**Step 2: Point StudioSharedEditor to the facade**

In `studio/src/editor/StudioSharedEditor.tsx`, replace:

```ts
} from "../../../src/editor-shared";
```

with:

```ts
} from "../current-editor";
```

**Step 3: Verify root imports are concentrated**

Run:

```powershell
rg '../../../src|../../../../src|../src|src/editor-shared' studio/src
```

Expected: most editor-root imports should now be in `studio/src/current-editor/index.ts`, plus known non-editor offenders such as `main.tsx`, `psd.ts`, and tests.

**Step 4: Validate**

Run:

```powershell
npm --prefix studio test
npm --prefix studio run build
```

Expected: PASS.

---

## Task 3: Copy Pure Editor Utilities First

**Files:**
- Create: `studio/src/current-editor/bitmap/layeredBitmapCanvas.ts`
- Create: `studio/src/current-editor/lib/editorTextStylePolicy.ts`
- Create: `studio/src/current-editor/lib/editorTextStylePresets.ts`
- Create: `studio/src/current-editor/lib/editorScene.ts`
- Create: `studio/src/current-editor/lib/editorOps.ts`
- Create: `studio/src/current-editor/lib/editorStroke.ts`
- Create: `studio/src/current-editor/lib/lassoSelection.ts`
- Modify: `studio/src/current-editor/index.ts`
- Modify: `studio/src/editor/__tests__/layeredBitmapCanvas.test.ts`

**Step 1: Copy files**

Copy from:

```text
src/editor-shared/bitmap/layeredBitmapCanvas.ts
src/lib/editorTextStylePolicy.ts
src/lib/editorTextStylePresets.ts
src/lib/editorScene.ts
src/lib/editorOps.ts
src/lib/editorStroke.ts
src/lib/lassoSelection.ts
```

to:

```text
studio/src/current-editor/bitmap/layeredBitmapCanvas.ts
studio/src/current-editor/lib/editorTextStylePolicy.ts
studio/src/current-editor/lib/editorTextStylePresets.ts
studio/src/current-editor/lib/editorScene.ts
studio/src/current-editor/lib/editorOps.ts
studio/src/current-editor/lib/editorStroke.ts
studio/src/current-editor/lib/lassoSelection.ts
```

**Step 2: Update local imports**

Inside the copied files, replace imports from root-app paths with local `studio/src/current-editor/*` paths.

**Step 3: Export copied bitmap utility**

In `studio/src/current-editor/index.ts`, export:

```ts
export { LayeredBitmapCanvas, bitmapStrokePasses } from "./bitmap/layeredBitmapCanvas";
```

**Step 4: Update test import**

In `studio/src/editor/__tests__/layeredBitmapCanvas.test.ts`, replace:

```ts
from "../../../../src/editor-shared";
```

with:

```ts
from "../../current-editor";
```

**Step 5: Validate**

Run:

```powershell
npm --prefix studio test -- layeredBitmapCanvas
npm --prefix studio test
```

Expected: PASS.

---

## Task 4: Establish Local Studio Editor Backend Contract

**Files:**
- Create: `studio/src/current-editor/lib/editorBackend.ts`
- Modify: `studio/src/backend/editorBackendCompat.ts`
- Modify: `studio/src/shims/currentEditorBackend.ts` or delete it later
- Test: `studio/src/backend/__tests__/editorBackendCompat.test.ts`

**Step 1: Copy the editor backend interface only**

Copy the `EditorBackendApi` contract from `src/lib/editorBackend.ts`, but do not copy the normal app default Tauri backend fallback.

`studio/src/current-editor/lib/editorBackend.ts` must expose:

```ts
export interface EditorBackendApi {
  // match only methods used by copied studio editorStore
}

let configuredBackend: EditorBackendApi | null = null;

export function configureEditorBackend(backend: EditorBackendApi) {
  configuredBackend = backend;
}

export function getEditorBackend(): EditorBackendApi {
  if (!configuredBackend) {
    throw new Error("Studio editor backend was not configured");
  }
  return configuredBackend;
}
```

**Step 2: Point compat adapter to the local contract**

Make `createLegacyEditorBackendAdapter(getStudioEditorBackend())` satisfy the local `EditorBackendApi`.

**Step 3: Add failure test**

In `studio/src/backend/__tests__/editorBackendCompat.test.ts`, add a test that proves no root Tauri editor backend is reachable.

Expected behavior:

```ts
expect(() => getEditorBackend()).toThrow("Studio editor backend was not configured");
```

before Studio boot configures it.

**Step 4: Validate**

Run:

```powershell
npm --prefix studio test -- editorBackendCompat
```

Expected: PASS.

---

## Task 5: Copy And Slim Stores Into Studio

**Files:**
- Create: `studio/src/current-editor/stores/appStore.ts`
- Create: `studio/src/current-editor/stores/editorStore.ts`
- Create: `studio/src/current-editor/lib/editorHistory.ts`
- Create: `studio/src/current-editor/lib/imageSource.ts`
- Create: `studio/src/current-editor/lib/workContext.ts`
- Modify: `studio/src/current-editor/index.ts`
- Test: `studio/src/current-editor/stores/__tests__/editorStoreHistory.test.ts`

**Step 1: Copy editor history and store files**

Copy:

```text
src/lib/stores/appStore.ts
src/lib/stores/editorStore.ts
src/lib/editorHistory.ts
src/lib/imageSource.ts
src/lib/workContext.ts
```

into:

```text
studio/src/current-editor/stores/appStore.ts
studio/src/current-editor/stores/editorStore.ts
studio/src/current-editor/lib/editorHistory.ts
studio/src/current-editor/lib/imageSource.ts
studio/src/current-editor/lib/workContext.ts
```

**Step 2: Remove unrelated app concerns from local appStore**

In `studio/src/current-editor/stores/appStore.ts`, keep only what the editor needs:

- `Project`
- `PageData`
- `TextEntry`
- `TextLayerStyle`
- `ImageLayerKey`
- `ProcessRegionOverlay`
- project state
- pipeline message/progress state if used by the toolbar
- `setProject`
- `updateProject`

Remove credits, setup, account, batch, onboarding, and non-editor app concerns.

**Step 3: Point editorStore imports to local files**

In `studio/src/current-editor/stores/editorStore.ts`, replace imports from:

```text
../editorBackend
../imageSource
../editorHistory
../editorOps
../workContext
```

with local current-editor paths.

**Step 4: Export local stores**

In `studio/src/current-editor/index.ts`, replace store reexports with:

```ts
export { useAppStore } from "./stores/appStore";
export { useEditorStore } from "./stores/editorStore";
export type { Project, TextLayerStyle } from "./stores/appStore";
```

**Step 5: Add history test port**

Copy the smallest useful history test from `src/lib/stores/__tests__/editorStoreHistory.test.ts` into:

```text
studio/src/current-editor/stores/__tests__/editorStoreHistory.test.ts
```

Adjust imports to `../editorStore` and `../appStore`.

**Step 6: Validate**

Run:

```powershell
npm --prefix studio test -- editorStoreHistory
npm --prefix studio test
```

Expected: PASS.

---

## Task 6: Copy Toolbar And Panel UI

**Files:**
- Create: `studio/src/current-editor/components/editor/PageThumbnails.tsx`
- Create: `studio/src/current-editor/components/editor/LayersPanel.tsx`
- Create: `studio/src/current-editor/components/editor/LayerItem.tsx`
- Create: `studio/src/current-editor/components/editor/LassoContextMenu.tsx`
- Create: `studio/src/current-editor/components/editor/EditorFontPicker.tsx`
- Create: `studio/src/current-editor/components/editor/toolbar/*.tsx`
- Modify: `studio/src/current-editor/index.ts`

**Step 1: Copy files**

Copy these folders/files:

```text
src/components/editor/PageThumbnails.tsx
src/components/editor/LayersPanel.tsx
src/components/editor/LayerItem.tsx
src/components/editor/LassoContextMenu.tsx
src/components/editor/EditorFontPicker.tsx
src/components/editor/toolbar/
```

to:

```text
studio/src/current-editor/components/editor/
```

**Step 2: Update imports**

All imports must point to local current-editor stores/libs, not `src/lib`.

**Step 3: Update facade exports**

Export:

```ts
export { LayersPanel } from "./components/editor/LayersPanel";
export { PageThumbnails } from "./components/editor/PageThumbnails";
export { ToolSidebar } from "./components/editor/toolbar/ToolSidebar";
export { UndoRedoControls } from "./components/editor/toolbar/UndoRedoControls";
export { ZoomControls } from "./components/editor/toolbar/ZoomControls";
export { TypesettingBar } from "./components/editor/toolbar/TypesettingBar";
```

**Step 4: Validate**

Run:

```powershell
npm --prefix studio run build
```

Expected: PASS.

---

## Task 7: Copy Stage Runtime

**Files:**
- Create: `studio/src/current-editor/components/editor/stage/*`
- Modify: `studio/src/current-editor/index.ts`
- Test: copied focused stage tests where available

**Step 1: Copy stage directory**

Copy:

```text
src/components/editor/stage/
```

to:

```text
studio/src/current-editor/components/editor/stage/
```

**Step 2: Update imports**

Replace root imports with local current-editor paths.

Pay special attention to:

- `useEditorStageController.ts`
- `useEditorBitmapDrawing.ts`
- `EditorBitmapOverlay.tsx`
- `renderModeUtils.ts`
- `textLayerStyleUtils.ts`
- `healingBrushMask.ts`
- `recoveryComposite.ts`

**Step 3: Export local stage**

In `studio/src/current-editor/index.ts`:

```ts
export { EditorStage } from "./components/editor/stage/EditorStage";
```

**Step 4: Validate**

Run:

```powershell
npm --prefix studio run build
npm --prefix studio test
```

Expected: PASS.

---

## Task 8: Copy Full Editor Chrome Into Studio

**Files:**
- Create: `studio/src/current-editor/Editor.tsx`
- Modify: `studio/src/editor/StudioSharedEditor.tsx`
- Modify: `studio/src/App.tsx` if needed

**Step 1: Copy normal editor**

Copy:

```text
src/pages/Editor.tsx
```

to:

```text
studio/src/current-editor/Editor.tsx
```

**Step 2: Make it Studio-owned**

In `studio/src/current-editor/Editor.tsx`:

- Remove `useNavigate` dependency if Studio does not need router navigation.
- Keep `onBack` and `emptyBackLabel`.
- Import all components/stores/libs from `./components`, `./stores`, and `./lib`.
- Keep the same header, view toolbar, `TypesettingBar`, pipeline error banner, `ToolSidebar`, `EditorStage`, `LayersPanel`, and shortcuts.

**Step 3: Replace StudioSharedEditor chrome**

In `studio/src/editor/StudioSharedEditor.tsx`, remove:

- custom `StudioEditorChrome`
- `.studio-titlebar`
- `.studio-editor-toolbar`
- hardcoded `English (en)`
- static eye button
- manual page action list

Render:

```tsx
<Editor
  onBack={() => {
    // optional Studio appbar/project browser action
  }}
  emptyBackLabel="Abrir projeto Studio"
/>
```

**Step 4: Keep Studio-only app actions outside the editor**

If `Abrir`, `Salvar como`, and `PSD` must remain visible, put them in a narrow Studio appbar outside the editor. Do not fork the editor toolbar for them.

**Step 5: Validate**

Run:

```powershell
npm --prefix studio run build
npm --prefix studio test
```

Expected: PASS.

---

## Task 9: Localize Styles, Tailwind, Fonts, And Assets

**Files:**
- Create: `studio/src/styles/globals.css`
- Modify: `studio/src/main.tsx`
- Modify: `studio/tailwind.config.js`
- Modify: `studio/vite.config.ts`
- Create/copy: `studio/public/fonts/*`

**Step 1: Copy global CSS**

Copy:

```text
src/styles/globals.css
```

to:

```text
studio/src/styles/globals.css
```

**Step 2: Update Studio main import**

In `studio/src/main.tsx`, replace:

```ts
import "../../src/styles/globals.css";
```

with:

```ts
import "./styles/globals.css";
```

**Step 3: Make Tailwind config autonomous**

Replace `studio/tailwind.config.js` with a local copy of the root config. Its `content` must be:

```js
content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
```

No import from `../tailwind.config.js`.

**Step 4: Localize fonts**

Copy required editor fonts into `studio/public/fonts/`:

```text
public/fonts/ComicNeue-Regular.ttf
public/fonts/ComicNeue-Bold.ttf
public/fonts/Newrotic.ttf
public/fonts/KOMIKAX_.ttf
public/fonts/CCDaveGibbonsLower W00 Regular.ttf
```

Update `studio/src/current-editor/lib/fonts.ts` and `fontCatalog.ts` to resolve from Studio-local public assets.

**Step 5: Validate**

Run:

```powershell
npm --prefix studio run build
```

Expected: PASS.

---

## Task 10: Remove Package And Vite Coupling To Root App

**Files:**
- Modify: `studio/package.json`
- Modify: `studio/package-lock.json`
- Modify: `studio/vite.config.ts`
- Delete: `studio/src/shims/currentEditorBackend.ts` if unused

**Step 1: Remove root package dependency**

In `studio/package.json`, remove:

```json
"traduzai": "file:.."
```

Run:

```powershell
npm --prefix studio install
```

Expected: `studio/package-lock.json` no longer references the root package dependency.

**Step 2: Simplify Vite config**

Remove from `studio/vite.config.ts`:

- `studio-current-editor-backend-shim` plugin
- `currentEditorBackendShim`
- `currentEditorStorePath`
- `currentEditorBackendPath`
- `fs.allow` for `..`
- `publicDir: resolve(__dirname, "..", "public")`

Keep:

```ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 1430,
  },
});
```

**Step 3: Check root imports**

Run:

```powershell
rg '../../../src|../../../../src|../src|src/editor-shared|from "traduzai"|from ''traduzai''' studio
```

Expected: no runtime matches. Test fixtures may mention strings like `"traduzai"` as project format names; those are allowed.

**Step 4: Validate**

Run:

```powershell
npm --prefix studio test
npm --prefix studio run build
```

Expected: PASS.

---

## Task 11: Harden Project Compatibility Tests

**Files:**
- Modify: `studio/src/project/__tests__/adapters.test.ts`
- Modify: `studio/src/backend/__tests__/editorBackendCompat.test.ts`
- Modify: `studio/src/store/__tests__/projectStore.test.ts`
- Possibly modify: `studio/src/project/adapters.ts`

**Step 1: Add round-trip fixtures**

Add tests for imports of:

- `traduzai_v1`
- `traduzai_v2`
- `studio_project`
- `v12_analysis_project`

Each test must assert preservation/regeneration of:

- `paginas`
- `image_layers`
- `text_layers`
- `textos`
- `traduzido`
- `translated`
- `estilo`
- `style`
- `arquivo_original`
- `arquivo_traduzido`

**Step 2: Add backend adapter tests**

Cover:

- create text layer
- patch text layer
- delete text layer
- set text visibility
- set image visibility
- update `brush`
- update `mask`
- update `inpaint`
- update `rendered`

**Step 3: Add image fallback parity test**

Create or extend a helper so Studio chooses final images in this order:

```text
arquivo_traduzido
rendered_path
translated_path
image_layers.rendered
image_layers.inpaint
image_layers.base
arquivo_original
```

Assert it with a focused test.

**Step 4: Validate**

Run:

```powershell
npm --prefix studio test -- adapters editorBackendCompat projectStore
```

Expected: PASS.

---

## Task 12: Add Studio Visual Smoke

**Files:**
- Create: `studio/playwright.config.ts`
- Create: `studio/e2e/studio-editor-smoke.spec.ts`
- Modify: `studio/package.json`

**Step 1: Add script**

In `studio/package.json`:

```json
"test:e2e": "playwright test -c playwright.config.ts"
```

Add dev dependency if needed:

```json
"@playwright/test": "^1.59.1"
```

**Step 2: Add smoke test**

Create `studio/e2e/studio-editor-smoke.spec.ts` with coverage for:

- app opens at `http://127.0.0.1:1430`
- stage/canvas visible
- top header matches normal editor controls
- `Original`, `Limpa`, `Camadas` visible
- `TypesettingBar` appears after selecting a text layer
- source language select is not hardcoded
- zoom controls work
- layer panel visible

**Step 3: Run local server and test**

Run in one terminal:

```powershell
npm --prefix studio run dev
```

Run in another:

```powershell
npm --prefix studio run test:e2e -- --grep @studio-smoke
```

Expected: PASS, with screenshot artifact if Playwright is configured to capture on failure.

---

## Task 13: Desktop Studio Validation

**Files:**
- Read/modify only if needed: `studio/src-tauri/src/main.rs`
- Read/modify only if needed: `studio/src-tauri/capabilities/default.json`

**Step 1: Validate frontend build**

Run:

```powershell
npm --prefix studio run build
```

Expected: PASS.

**Step 2: Validate Tauri commands**

Run:

```powershell
cargo test --manifest-path studio/src-tauri/Cargo.toml
```

Expected: PASS.

**Step 3: Launch desktop Studio**

Run:

```powershell
npm --prefix studio run tauri:dev
```

Expected:

- app opens as TraduzAI Studio
- can load project
- can save project
- can write bitmap layer
- can export PSD
- does not require normal app Tauri commands

---

## Task 14: Final Isolation Gate

**Files:**
- Modify any remaining Studio imports/configs found by gates.

**Step 1: Verify Studio no longer imports root app**

Run:

```powershell
rg '../../../src|../../../../src|../src|src/editor-shared' studio
```

Expected: no matches except this plan document or baseline notes.

**Step 2: Verify package isolation**

Run:

```powershell
rg '"traduzai": "file:\.\."' studio/package.json studio/package-lock.json
```

Expected: no matches.

**Step 3: Verify Vite/Tailwind isolation**

Run:

```powershell
rg '../tailwind.config|fs:\s*\{|allow:\s*\[.*\.\.|publicDir:.*\.\.' studio/vite.config.ts studio/tailwind.config.js
```

Expected: no matches.

**Step 4: Run Studio full checks**

Run:

```powershell
npm --prefix studio test
npm --prefix studio run build
cargo test --manifest-path studio/src-tauri/Cargo.toml
```

Expected: PASS.

**Step 5: Run normal app canaries**

Run:

```powershell
npm run build
npm run test -- src/lib/__tests__/editorHistory.test.ts src/lib/stores/__tests__/editorStoreHistory.test.ts src/lib/__tests__/editorOps.test.ts
```

Expected: PASS. If failures are unrelated pre-existing failures, document them.

---

## Implementation Notes

- The first implementation pass should prefer copying complete files into `studio/src/current-editor/`, then trimming imports. Do not hand-rewrite the editor UI from scratch.
- `src/lib/tauri.ts` should not be copied wholesale unless absolutely necessary; it carries unrelated app commands, E2E mocks, credits/settings, and root Tauri assumptions.
- The Studio local backend should use `studio/src/backend/editorBackend.ts`, `studio/src/backend/editorBackendCompat.ts`, `studio/src/backend/memoryBackend.ts`, and `studio/src/backend/tauriBackend.ts`.
- Pipeline buttons can remain present for UI parity, but their Studio backend behavior must be explicit if not implemented.
- Keep each task small enough that `npm --prefix studio run build` can identify import mistakes early.

## Done Criteria

- Studio shows the same editor surface as the normal app: header, view modes, language selector, pipeline actions, zoom, `TypesettingBar`, side toolbar, thumbnails, stage, and layers panel.
- Studio editor code lives under `studio/src/current-editor/` or another Studio-local path.
- Studio has no runtime import from root `src/`.
- Studio has no package dependency on root `"traduzai": "file:.."`.
- Studio Vite config does not allow parent repo reads for app runtime.
- Studio Tailwind config does not scan root app source.
- Normal app remains unchanged in behavior and passes editor canaries.

# TraduzAI Studio Foundation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the separate TraduzAI Studio foundation without rewriting the current editor from scratch.

**Architecture:** Studio keeps the TraduzAI `project.json` editing model as the source of truth in phase 1. Existing editor contracts are reused through adapters, while Koharu GPL code is only ported behind explicit boundaries after compatibility tests pass.

**Tech Stack:** Tauri v2, React, TypeScript, Zustand, Konva/react-konva, HTML Canvas 2D for bitmap tools, Rust for local filesystem/export work.

---

### Task 1: Project Contract And Adapters

**Files:**
- Create: `studio/src/project/studioProject.ts`
- Create: `studio/src/project/adapters.ts`
- Create: `studio/schemas/studio_project.schema.json`
- Test: `studio/src/project/__tests__/adapters.test.ts`

**Steps:**
1. Define Studio types for `paginas`, `image_layers`, and `text_layers`.
2. Implement v1/v2/v12 import adapters.
3. Implement v2 compatibility export aliases.
4. Preserve unknown metadata fields.
5. Test alias regeneration and final image fallback order.

### Task 2: Separate Studio App Shell

**Files:**
- Create: `studio/package.json`
- Create: `studio/vite.config.ts`
- Create: `studio/vitest.config.ts`
- Create: `studio/tsconfig.json`
- Create: `studio/index.html`
- Create: `studio/src/main.tsx`
- Create: `studio/src/App.tsx`
- Create: `studio/src/styles.css`

**Steps:**
1. Scaffold a minimal React/Vite app under `studio/`.
2. Keep dependencies aligned with the current TraduzAI editor stack.
3. Add scripts for dev, build, and tests.
4. Verify TypeScript and adapter tests.

### Task 3: Reuse Policy

**Files:**
- Create: `studio/README.md`
- Create: `studio/THIRD_PARTY_POLICY.md`

**Steps:**
1. Document that Studio is separate from the current app.
2. Document allowed reuse from TraduzAI and GPL Koharu.
3. Document blocked source-available/no-license repos.

### Task 4: Local Backend And Project Store

**Files:**
- Create: `studio/src/backend/editorBackend.ts`
- Create: `studio/src/backend/memoryBackend.ts`
- Create: `studio/src/store/projectStore.ts`
- Test: `studio/src/backend/__tests__/memoryBackend.test.ts`
- Test: `studio/src/store/__tests__/projectStore.test.ts`
- Modify: `studio/src/App.tsx`
- Modify: `studio/src/styles.css`

**Completed:** Added a local backend contract, memory-backed implementation, project store, and basic import summary screen. This is the bridge before mounting reused editor components.

### Task 5: Current Editor Backend Compatibility Adapter

**Files:**
- Create: `studio/src/backend/editorBackendCompat.ts`
- Test: `studio/src/backend/__tests__/editorBackendCompat.test.ts`

**Completed:** Added a compatibility adapter that exposes the current `EditorBackendApi` method names over `StudioEditorBackend`. Text patching, layer visibility, bitmap brush/mask updates, recovery/reinpaint normalization to `image_layers.inpaint`, healing mask writes, local heal results, preview rendering, and pipeline action stubs are covered by tests.

### Task 6: First Studio Editor Surface

**Files:**
- Create: `studio/src/editor/StudioEditor.tsx`
- Create: `studio/src/editor/pageGeometry.ts`
- Test: `studio/src/editor/__tests__/pageGeometry.test.ts`
- Modify: `studio/src/App.tsx`
- Modify: `studio/src/styles.css`
- Modify: `studio/src/store/projectStore.ts`
- Test: `studio/src/store/__tests__/projectStore.test.ts`

**Completed:** Added a first usable Studio editing surface with page rail, central scan canvas, text boxes, image/text layer visibility controls, and a text inspector. Text edits and visibility changes persist through `createLegacyEditorBackendAdapter`, keeping the Studio route connected to the current editor backend contract.

### Task 7: Mount Current Editor UI In Studio

**Files:**
- Create: `studio/src/editor/StudioSharedEditor.tsx`
- Create: `studio/src/vite-env.d.ts`
- Create: `studio/tailwind.config.js`
- Create: `studio/postcss.config.js`
- Modify: `studio/src/App.tsx`
- Modify: `studio/src/main.tsx`
- Modify: `studio/vite.config.ts`
- Modify: `studio/src/backend/editorBackendCompat.ts`

**Completed:** Mounted the current editor UI in Studio by importing `PageThumbnails`, `ToolSidebar`, `EditorStage`, `LayersPanel`, `ZoomControls`, and `UndoRedoControls` directly from the existing app. Studio now injects a normalized project into the current `useAppStore`, configures `src/lib/editorBackend.ts` with the Studio compatibility adapter, loads the current `editorStore`, and reuses the app's Tailwind/CSS/font assets.

### Task 8: Studio Editor Backend Shim

**Files:**
- Create: `studio/src/shims/currentEditorBackend.ts`
- Modify: `studio/vite.config.ts`
- Modify: `studio/src/editor/StudioSharedEditor.tsx`

**Completed:** Added a Studio-only shim for the current editor backend configuration so the Studio bundle no longer includes the desktop `tauriEditorBackend` fallback chunk. The existing app remains unchanged; Studio redirects the current editor store's `../editorBackend` import through Vite at build/dev time.

### Task 9: Fullscreen Current Editor Workspace

**Files:**
- Modify: `studio/src/App.tsx`
- Modify: `studio/src/editor/StudioSharedEditor.tsx`
- Modify: `studio/src/styles.css`

**Completed:** Changed Studio to open directly into the current editor-style workspace instead of the foundation summary screen. The Studio shell now uses an app bar, project title bar, view mode toolbar, page actions, thumbnails, vertical tool rail, shared `EditorStage`, and the existing right-side layers/text panel.

### Task 10: Shared Editor Boundary And Lazy Loading

**Files:**
- Create: `src/editor-shared/index.ts`
- Modify: `studio/src/editor/StudioSharedEditor.tsx`
- Modify: `studio/src/App.tsx`
- Modify: `studio/tailwind.config.js`

**Completed:** Added a stable shared editor entrypoint for Studio imports and switched `StudioSharedEditor` to import editor UI/stores through that boundary instead of deep component paths. Studio now lazy-loads the full editor workspace with `React.lazy`, moving the heavy editor bundle into a separate `StudioSharedEditor` chunk and removing the large initial chunk warning.

### Task 11: Project JSON Round-Trip Compatibility

**Files:**
- Modify: `studio/src/project/adapters.ts`
- Modify: `studio/src/project/__tests__/adapters.test.ts`

**Completed:** Added desktop-style and site-style `project.json` round-trip tests. The Studio adapters now preserve image layer metadata, synchronize `arquivo_original`/`original_path`, `arquivo_traduzido`/`rendered_path`/`translated_path`, preserve `image_layers.inpaint.path` through `inpaint_path`/`arquivo_final`, keep `text_layers` and `textos` coherent, preserve legacy `texto`, and use the editor/site bbox priority (`render_bbox`, `layout_bbox`, `bbox`, `source_bbox`, `balloon_bbox`).

### Task 12: Layered Canvas Bitmap Foundation

**Files:**
- Create: `src/editor-shared/bitmap/layeredBitmapCanvas.ts`
- Test: `studio/src/editor/__tests__/layeredBitmapCanvas.test.ts`
- Modify: `src/editor-shared/index.ts`
- Modify: `src/components/editor/stage/useEditorStageController.ts`

**Completed:** Added a shared `LayeredBitmapCanvas` utility for the bitmap layers used by scan editing (`base`, `inpaint`, `brush`, `mask`, `rendered`). The current editor controller now routes brush/mask working canvases through this layered Canvas 2D foundation while keeping the existing Konva stage, text editing, bitmap overlay rendering, persistence queues, and backend contract intact. Tests cover layer ordering, soft/hard brush passes, eraser composition, opacity-aware visible compositing, and exported layer data URLs.

### Task 13: Canvas Paint Stroke Preview

**Files:**
- Modify: `src/components/editor/stage/EditorStage.tsx`

**Completed:** Replaced the active Konva `<Line>` paint preview with a Canvas 2D overlay positioned over the existing stage. The overlay follows the same brush colors, opacity, hardness passes, and active lasso clipping, while persistence, undo/redo, bitmap queues, text editing, selection, and existing Konva stage behavior remain unchanged.

### Task 14: Bitmap Layer Canvas Composite

**Files:**
- Modify: `src/components/editor/stage/EditorBitmapOverlay.tsx`
- Modify: `src/components/editor/stage/EditorStage.tsx`

**Completed:** Replaced separate bitmap overlay rendering for `mask` and `brush` with a single Canvas 2D composition step backed by `LayeredBitmapCanvas`. The stage still keeps the existing Konva layer structure for compatibility, but bitmap display now uses one composited Canvas source with mask tinting, brush opacity, placeholder filtering, and dimension mismatch warnings.

### Task 15: Site PSD Export Baseline

**Files:**
- Create: `server/projects/psd_export.py`
- Modify: `server/projects/export_api.py`
- Modify: `server/tests/test_project_exports.py`

**Completed:** Replaced the site PSD placeholder with a real PSD writer baseline that exports page-level raster layers from `project.json`: original, inpaint, hidden mask/brush utility layers, and transparent text layer placeholders. This keeps the first PSD authority inside TraduzAI code and matches the app's layer ordering direction without copying external repository code. Tests now assert the PSD signature, header fields, layer/mask section, and non-empty layer count.

### Task 16: Studio PSD Page Export Action

**Files:**
- Create: `studio/src/export/psd.ts`
- Test: `studio/src/export/__tests__/psd.test.ts`
- Modify: `studio/src/editor/StudioSharedEditor.tsx`

**Completed:** Added a Studio-side PSD writer and a titlebar `PSD` action for exporting the current page without depending on Tauri filesystem APIs. The Studio writer mirrors the site PSD baseline with original, inpaint, hidden utility bitmap layers, and transparent text placeholders. Unit tests validate PSD header fields and a real layer/mask section.

### Task 17: Studio Desktop Tauri Shell

**Files:**
- Create: `studio/src-tauri/Cargo.toml`
- Create: `studio/src-tauri/tauri.conf.json`
- Create: `studio/src-tauri/src/main.rs`
- Create: `studio/src-tauri/capabilities/default.json`
- Create: `studio/src-tauri/.gitignore`
- Modify: `studio/package.json`
- Create: `studio/src/backend/tauriBackend.ts`
- Modify: `studio/src/store/projectStore.ts`
- Modify: `studio/README.md`

**Completed:** Added a separate Tauri v2 desktop shell for TraduzAI Studio with its own app identifier, window config, capabilities, icons, and Rust commands for local `project.json` load/save plus bitmap layer writes. Studio now uses a hybrid backend: browser/dev paths keep the memory backend, while real project paths inside the Tauri runtime use Rust commands through `invoke`.

### Task 18: Desktop Open And Save Project Actions

**Files:**
- Create: `studio/src/backend/projectDialog.ts`
- Modify: `studio/package.json`
- Modify: `studio/package-lock.json`
- Modify: `studio/src/store/projectStore.ts`
- Modify: `studio/src/editor/StudioSharedEditor.tsx`

**Completed:** Added desktop-only project dialogs for opening an existing `project.json` and saving the current Studio project as a chosen `project.json`. The titlebar now exposes `Abrir` and `Salvar como`; they are enabled only inside the Tauri runtime and stay disabled in browser/dev mode. `Salvar como` writes the compatibility JSON through the existing Studio backend and updates `source_path`/`output_path` to the selected file.

### Task 19: Desktop Dialog QA And Custom Project Filenames

**Files:**
- Modify: `studio/src-tauri/src/main.rs`
- Modify: `studio/src/backend/tauriBackend.ts`
- Modify: `studio/src/store/projectStore.ts`
- Modify: `studio/src/export/psd.ts`

**Completed:** Tested the real Tauri desktop window through Computer Use with a copied project fixture. `Abrir` now tolerates UTF-8 BOM project files. `Salvar como` now accepts custom `.json` filenames instead of creating `<name>.json/project.json`, reloads through the backend after saving so UI asset paths stay resolved, and the PSD export path handles custom project filenames. The exported PSD was opened successfully in Adobe Photoshop 2026.

### Task 20: Studio PSD Editable Text Metadata

**Files:**
- Modify: `studio/src/export/psd.ts`
- Modify: `studio/src/export/__tests__/psd.test.ts`

**Completed:** Added `TySh` editable text metadata to Studio PSD text layers by porting the existing TraduzAI PSD writer shape into the TypeScript Studio exporter. Text layers now write `luni`, `TySh`, versioned text/warp descriptors, bounds, transform matrix, and `EngineData` with font, size, color, and paragraph justification. Tests assert `TySh`, `EngineData`, text descriptor keys, and font resources are present in generated PSD bytes.

### Next Tasks

1. Add full-project PSD batch export once the page-level desktop flow is stable.
2. Run batch PSD export against the full project set after choosing the fixture/source list; single-page TySh validation now passes in Photoshop 2026 normal with the text layer recognized as editable.

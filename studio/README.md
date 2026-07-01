# TraduzAI Studio

TraduzAI Studio is a separate GPL-3.0 desktop editor for scan post-production.

The current TraduzAI app remains unchanged. Studio starts from the existing
TraduzAI editor contracts and project files, then adds a lighter editing surface
focused on scans.

## Running

- Browser/dev shell: `npm --prefix studio run dev`
- Desktop app shell: `npm --prefix studio run tauri:dev`

## Initial Scope

- Canonical editable model: `paginas[]`, `image_layers`, `text_layers`.
- Compatibility aliases: `textos`, `traduzido`, `estilo`, `arquivo_original`,
  `arquivo_traduzido`.
- Compatibility adapter: `createLegacyEditorBackendAdapter` maps the Studio
  backend to the current editor backend method names.
- First editor surface: page rail, scan canvas, text boxes, layer visibility,
  and text inspector backed by the compatibility adapter.
- Current editor UI reuse: Studio mounts the existing `EditorStage`,
  `PageThumbnails`, `LayersPanel`, `ToolSidebar`, `ZoomControls`, and
  `UndoRedoControls` through a project/app-store shim.
- Studio backend shim: the current editor store is redirected to a Studio-only
  backend configuration module, avoiding the desktop Tauri fallback in Studio
  builds.
- Fullscreen workspace: Studio opens directly into the current editor-style UI
  instead of a foundation summary screen.
- Shared editor boundary: Studio imports the current editor UI through
  `src/editor-shared` and lazy-loads the heavy workspace chunk.
- Import targets: TraduzAI v1/v2 projects and v12 analysis projects.
- Round-trip compatibility: adapters preserve app/site `project.json` aliases,
  image layer metadata, text aliases, bbox priority, QA/context metadata, and
  inpaint/rendered path fallbacks.
- Layered Canvas bitmap foundation: shared Canvas 2D layers now back the
  brush/mask working surfaces while the reused editor keeps Konva for text,
  selection, and existing stage behavior.
- Canvas paint preview: the active brush stroke preview renders through a
  Canvas 2D overlay with lasso clipping and brush hardness instead of a Konva
  line node.
- Bitmap layer composite: visible `mask` and `brush` layers are merged through
  `LayeredBitmapCanvas` into one Canvas source before the reused stage displays
  them.
- Export targets: site/app compatible `project.json`, ZIP/CBZ/JPG bundles, and
  PSD. The first PSD baseline is implemented in TraduzAI server code and writes
  real raster layer sections instead of a placeholder header.
- Studio PSD action: the editor titlebar can export the current page as PSD
  from the standalone Studio runtime, without depending on Tauri filesystem
  APIs.
- Desktop shell: Studio has its own Tauri v2 app wrapper with Rust commands for
  local `project.json` load/save and bitmap layer writes. Browser mode keeps the
  memory backend for fast development.

## Reuse Policy

- Reuse current TraduzAI editor modules and contracts first.
- Port GPL-3.0 Koharu code only if Studio remains GPL-3.0.
- Do not copy code, assets, models, or UI from source-available or no-license
  manga cleaner repositories.

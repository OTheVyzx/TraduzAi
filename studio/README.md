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
- Post-translation entry: the home screen opens an existing TraduzAI project;
  it no longer creates a sample project that suggests a second pipeline.
- Explicit editor mode: the shared editor keeps the central TraduzAI behavior
  by default while `mode="studio"` removes source-language, detect, OCR,
  translate, block-cleaning, and regional automatic-process surfaces.
- Editorial toolset: Studio currently exposes selection, text boxes, brush,
  eraser, and lasso/mask. Selections support add/subtract regions, feather,
  expansion/contraction, an explicit target layer, and serializable layer-mask
  descriptors. Automatic pipeline repair/reinpaint tools remain hidden.
- Native retouch contract: clone, healing, and patch are serializable scene
  commands that create a masked generated layer above the raster target while
  preserving the source layer and the transactional undo/redo history.
- FLUX generative fill: a Studio-only panel sends the selected local crop,
  black/white mask, and optional prompt to a configured local adapter. It
  returns 2 to 4 variants as independent generated layers; the source raster is
  never overwritten, preview/accept/reject are undoable, and pixels outside the
  selection are forced transparent before the result enters the scene.
- Local FLUX runtime: the desktop bridge starts an exact executable with JSON
  arguments and communicates over persistent JSONL stdin/stdout without a
  shell. Normal jobs reuse the resident model; cancel kills the worker to free
  GPU/RAM, and concurrent generations are rejected. Partial assets are removed
  before scene commit. The bundled Python adapter uses
  `diffusers.FluxFillPipeline`; model downloads are disabled by default and no
  image is uploaded. See
  [`flux_adapter/README.md`](flux_adapter/README.md) for setup and model-license
  requirements.
- Studio layers presentation: the shared panel uses professional raster names
  and hides per-text OCR/translate/clean actions without changing the central
  app panel.
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
- Chapter productivity: the `Capítulo` panel copies/applies text styles,
  previews and applies whole-chapter find/replace, exposes the QA review queue,
  and provides field-patch undo/redo without reverting later unrelated edits.
  Locked text layers are excluded from batch style/replace.
- Autosave and recovery: Studio activates the shared editor's incremental
  autosave every three seconds, flushes before page navigation, writes
  `project.json` transactionally through a per-project mutation queue, and keeps
  up to five atomic recovery snapshots namespaced by the selected JSON under
  `.traduzai-studio/recovery/` beside the project. Recovery is an identity-checked
  modal decision before editing resumes.
- Export parity: PSD text layers remain editable while carrying the same Konva
  raster preview used by the canvas/PNG path; automated pixel tests compare the
  independently composed canvas, PNG round-trip, and embedded PSD composite.
  Long-page slices keep editable text metadata in only one part when a text box
  crosses the 2000 px boundary.
- Desktop shell: Studio has its own Tauri v2 app wrapper with Rust commands for
  local `project.json` load/save and bitmap layer writes. Browser mode keeps the
  memory backend for fast development; generative fill therefore reports the
  local provider as unavailable in a browser-only session.

## Reuse Policy

- Reuse current TraduzAI editor modules and contracts first.
- Port GPL-3.0 Koharu code only if Studio remains GPL-3.0.
- Do not copy code, assets, models, or UI from source-available or no-license
  manga cleaner repositories.

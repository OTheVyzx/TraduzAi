# TraduzAI Studio Library, Workspaces, and Work Tracking Design

**Date:** 2026-07-21  
**Status:** Approved  
**Scope:** Standalone TraduzAI Studio only

## 1. Product decision

TraduzAI Studio becomes the master desktop workspace for organizing and manually finishing manga, manhwa, and manhua chapters. Its start screen follows the project-library interaction model of DaVinci Resolve without copying DaVinci assets or source code:

- the left column represents works;
- the main area represents chapters of the selected work;
- chapter cards open the existing shared editor;
- the editor exposes switchable workspaces, beginning with `Traducao` and `Edicao`;
- external services update publication metadata and chapter availability, but never download chapter images automatically;
- FLUX remains outside this delivery and is addressed only after the core editing workflows are complete.

The Studio remains separate from the automatic translation pipeline. A chapter may come from TraduzAI Central or be created manually from a folder, ZIP, or CBZ of source images.

## 2. Goals

1. Replace the current flat recent-project list with a durable `work -> chapters` library.
2. Let a user add, rename, remove, relink, and inspect a work without moving its chapter files.
3. Let a user create a manual chapter from source images or attach an existing TraduzAI `project.json`.
4. Provide a focused manual-translation workspace that shares the same chapter document with the editing workspace.
5. Track publication status and remote chapter availability using provider adapters.
6. Preserve the open and reimportable TraduzAI project contract.
7. Keep all image editing and project data local; only public metadata requests leave the machine.

## 3. Non-goals

- Running the automatic TraduzAI translation pipeline inside Studio.
- Automatically downloading chapter images from external services.
- Treating a remote chapter record as a local editable chapter.
- Replacing the current editor canvas or scene model in this delivery.
- Adding FLUX, generative fill, collaboration, or cloud synchronization.
- Copying code or UI assets from DaVinci Resolve or Manwha-Studio.

## 4. Chosen architecture

### 4.1 Catalog instead of managed copies

Studio owns a versioned catalog stored in the Tauri application-data directory as `studio-library.json`. The catalog references chapter `project.json` files in their existing locations. It does not copy or relocate Central projects.

This keeps the library fast and expressive while preserving compatibility with existing chapters. All writes must be atomic and recoverable from a last-known-good backup.

```ts
interface StudioLibraryCatalog {
  schemaVersion: 1;
  works: StudioLibraryWork[];
  preferences: {
    lastWorkId?: string;
    homeView: "grid" | "list";
    sort: "recent" | "title" | "chapter" | "status";
    syncOnLaunch: boolean;
  };
}

interface StudioLibraryWork {
  id: string;
  title: string;
  aliases: string[];
  coverPath?: string;
  sourceLanguage: string;
  targetLanguage: string;
  publicationStatus: PublicationStatus;
  publicationStatusOverride?: PublicationStatus;
  externalSources: ExternalWorkSource[];
  chapters: StudioLibraryChapter[];
  createdAt: string;
  updatedAt: string;
  lastOpenedAt?: string;
}

type PublicationStatus =
  | "releasing"
  | "hiatus"
  | "completed"
  | "cancelled"
  | "not_yet_released"
  | "unknown";

interface StudioLibraryChapter {
  id: string;
  label: string;
  sortKey?: string;
  title?: string;
  projectPath: string;
  thumbnailPath?: string;
  origin: "traduzai_central" | "manual";
  workflowStatus: "not_started" | "translating" | "review" | "editing" | "completed";
  pageCount: number;
  createdAt: string;
  updatedAt: string;
  lastOpenedAt?: string;
}
```

Publication state and local workflow state are deliberately separate. A completed series may still contain locally unfinished chapters.

### 4.2 Compatibility boundary

The catalog is the source of truth for library organization. Each chapter `project.json` remains the source of truth for pages, bitmap layers, text layers, and the Studio scene.

Optional compatibility-safe metadata may be added to `project.json`, including:

- `library_work_id`;
- `chapter_workflow_status`;
- per-text translation/review status;
- external source identifiers.

Existing aliases such as `text_layers`/`textos` and `translated`/`traduzido` must continue to round-trip.

## 5. Start screen

### 5.1 Layout

The start screen uses the existing Studio dark theme and density while adopting the reference hierarchy:

- **Left library:** `Todas`, `Atualizacoes`, `Recentes`, then individual works.
- **Main toolbar:** selected work title, search, status/source filters, sorting, grid/list toggle, refresh, and sync state.
- **Chapter area:** thumbnail cards or compact rows for the selected work.
- **Bottom-left action:** `Adicionar obra`.
- **Bottom actions:** `Importar projeto`, `Adicionar capitulo`, and `Abrir`.

### 5.2 Work cards and chapter cards

A work entry shows cover, title, publication status, latest local chapter, latest remote chapter, and pending update count.

A chapter card shows:

- first-page thumbnail;
- chapter label and optional title;
- origin (`TraduzAI Central` or `Manual`);
- local workflow status;
- page count and last edit time;
- missing-path warning when relinking is required.

Chapter labels remain strings so values such as `12.5`, `Prologo`, `Extra`, and `Especial` are preserved. A separate normalized sort key supports ordering without changing the displayed label.

### 5.3 Add work

`Adicionar obra` opens a focused dialog with two routes:

1. search AniList and link a result;
2. create a fully manual work.

The user may edit the title, aliases, cover, source language, target language, and preferred tracking source before saving.

### 5.4 Add chapter

`Adicionar capitulo` accepts:

- a folder containing images;
- ZIP or CBZ containing images;
- an existing TraduzAI `project.json`.

Image imports create a standard manual Studio chapter with base image layers, empty text layers, and a compatible `project.json`. Archive extraction must reuse the repository's established traversal, extension, size, and entry-count safety rules.

Existing projects are attached by reference. Duplicate paths and duplicate work/chapter pairs require confirmation instead of silent replacement.

## 6. Editor workspaces

The chapter editor gains a workspace selector in the upper-right corner:

- `Traducao`
- `Edicao`

Both workspaces share the same loaded project, current page, selected text layer, undoable content operations, and autosave lifecycle. Switching workspaces must not reload the project or discard selection.

The application remembers the last workspace per chapter and offers a clear return-to-library action.

### 6.1 Translation workspace

The translation workspace is optimized for keyboard-first manual work.

**Left panel**

- page thumbnails;
- progress per page;
- filters for pending, translated, review, and approved entries.

**Center canvas**

- original page with selectable text boxes;
- draw-to-create a new text box;
- synchronized selection with the translation queue;
- `Original`, `Limpa`, and `Traduzida` view modes.

**Right panel**

- editable original text;
- editable target translation;
- semantic type (`fala`, `pensamento`, `narracao`, `sistema`, or `sfx`);
- notes/context;
- status (`pending`, `translated`, `review`, or `approved`);
- secondary tabs for glossary and work context.

Saving a translation updates both `translated` and `traduzido` through the existing compatibility backend. Confirming an item advances to the next visible queue item. Keyboard actions cover save, previous, next, mark for review, and approve.

The first delivery contains no automatic translation and does not require OCR. A future `Sugerir traducao` action may be added as an explicit assistive command without changing the default manual workflow.

### 6.2 Editing workspace

`Edicao` remains the current shared Studio editor: page canvas, text styling, scene/layers, selection, masks, retouch, chapter tools, and export. It immediately reflects translations made in the translation workspace.

## 7. Publication and chapter tracking

### 7.1 Provider strategy

Use a provider interface from the start, but ship two providers first:

1. **AniList:** work identity, aliases, cover, country, canonical publication status, known total chapters, and metadata update time.
2. **MangaDex:** publication status plus released chapter feed and publication timestamps for the configured source language.

MangaUpdates may be added later behind the same interface. The current Central lookups provide reusable request and parsing patterns, but Studio tracking must live in the standalone Studio Tauri runtime and must not invoke the automatic pipeline.

```ts
interface ExternalWorkSource {
  provider: "anilist" | "mangadex" | "mangaupdates";
  remoteId: string;
  url: string;
  enabled: boolean;
  preferredForStatus?: boolean;
  preferredForChapters?: boolean;
  lastSyncAt?: string;
  lastError?: string;
}
```

### 7.2 Sync behavior

- `Verificar atualizacoes` always requests a fresh sync.
- Optional launch sync uses a cache TTL and never blocks opening the library.
- Cached metadata keeps the library functional offline.
- Requests run through Rust/Tauri; the React frontend does not call providers directly.
- Rate-limit responses use provider-aware backoff and expose a friendly retry time.
- Covers are cached locally with attribution metadata.

### 7.3 Resolution rules

- A manual status override wins until explicitly cleared.
- Otherwise the preferred status provider wins; AniList is the default.
- Chapter availability comes from the preferred chapter provider; MangaDex is the default.
- Provider disagreements remain visible in the work details instead of being silently collapsed.
- Remote and local chapters are compared using normalized labels, language, and explicit aliases, not only floating-point numbers.
- A remote record never becomes a local editable chapter until the user imports source images or attaches a project.

### 7.4 Updates view

The `Atualizacoes` library section groups:

- newly detected chapters;
- publication-status changes;
- relink-required local chapters;
- provider conflicts or stale synchronization.

Each item offers `Abrir obra`, `Adicionar capitulo`, `Ignorar`, or `Ver fonte`. There is no automatic content download.

## 8. Error handling and recovery

- Invalid catalog JSON falls back to the last-known-good backup and reports recovery.
- Missing chapter paths keep their catalog entries and offer `Relocalizar`, never automatic deletion.
- Failed provider requests retain cached data and show a non-blocking stale marker.
- Ambiguous search results require explicit user selection.
- Archive imports reject unsafe paths, unsupported entries, excessive file counts, and size-limit violations before extraction.
- Partial manual chapter creation cleans only its own temporary directory and does not modify existing projects.
- Closing a dirty chapter requires save/discard/cancel handling before returning to the library.

## 9. Testing strategy

### Catalog and migration

- empty catalog creation;
- atomic save and backup recovery;
- schema migration;
- duplicate work/path handling;
- missing-path preservation and relinking;
- chapter label normalization and ordering.

### Provider synchronization

- AniList and MangaDex response parsing from fixtures;
- status mapping;
- remote/local chapter comparison;
- cache TTL and offline fallback;
- rate-limit/backoff behavior;
- disagreement and manual-override resolution.

### Manual chapter creation

- folder, ZIP, and CBZ imports;
- safe extraction rejection cases;
- deterministic page ordering;
- compatible project round-trip.

### UI and workflow

- work selection filters chapter cards;
- search, sort, status, and source filters;
- workspace switch preserves page and selection;
- translation save updates the shared editor model;
- keyboard next/previous and review statuses;
- return-to-library and dirty-state confirmation;
- recovery and relink surfaces.

### Desktop verification

- focused Vitest suites;
- Studio production build;
- Studio Tauri `cargo test` and `cargo check`;
- packaged-app smoke test with a real Central chapter and a manually created chapter;
- visual comparison against the approved DaVinci-inspired reference at the same viewport.

## 10. Delivery sequence

1. Catalog schema, persistence, migration, and Rust commands.
2. DaVinci-inspired library home with work/chapter organization.
3. Add/relink work and attach existing projects.
4. Create manual chapters from folder, ZIP, and CBZ.
5. Workspace shell and reliable return-to-library behavior.
6. Manual translation workspace and per-text review status.
7. AniList provider and publication status.
8. MangaDex provider, chapter comparison, and updates view.
9. End-to-end desktop hardening, accessibility, and performance.

FLUX remains explicitly after this sequence.

## 11. Reference-repository decision

The public `vathanatork/Manwha-Studio` repository contains its marketing/documentation website, not the desktop editor implementation. It has no root open-source license and its published terms reserve the software, source code, design, graphics, and UI. It may inform product-level ideas such as workspace switching, autosave, and chapter organization, but no code, artwork, or detailed UI implementation will be copied.

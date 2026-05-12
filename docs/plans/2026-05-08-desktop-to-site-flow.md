# Desktop To Site Flow Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Levar o fluxo principal do app desktop para o site local: configurar projeto, processar capítulo, abrir preview, editar páginas e exportar em ZIP completo, CBZ, JPG e PSD.

**Architecture:** O site continua em `site/` como workspace separado e conversa com o backend FastAPI em `server/`. O backend mantém jobs, arquivos, sessão do projeto e endpoints web; o worker local roda o pipeline real e materializa um workspace com `project.json`, `originals/`, `images/`, `translated/` e `layers/`. O editor web deve reutilizar os contratos do editor desktop, mas por adaptadores HTTP em vez de Tauri IPC.

**Tech Stack:** React 19 + TypeScript + Vite em `site/`, FastAPI + SQLite em `server/`, Python pipeline em `pipeline/main.py`, export ZIP/CBZ/JPG no backend e export PSD via CLI Rust reaproveitando `src-tauri/src/commands/project.rs` / `src-tauri/src/export/psd`.

---

## Current Desktop Setup Verification

Verified on 2026-05-08 against the current dirty worktree:

- `src/pages/Setup.tsx` now treats `Configurar projeto` as a context-first flow, not just a generic setup form.
- The first visible block is `Obra e contexto` (`data-testid="work-context-summary"`), with:
  - optional work title;
  - favorite works;
  - context search;
  - context status;
  - suggestion count;
  - accepted glossary count;
  - risk badge.
- Glossary review was promoted into its own primary card (`data-testid="glossary-suggestions"`), with accept, edit and ignore actions.
- The old bulk "apply high confidence candidates" flow was removed. The current behavior expects explicit user review of terms.
- Advanced setup is now a collapsible panel (`data-testid="setup-advanced-panel"`) containing:
  - online sources (`internet-context-panel`);
  - custom preset creation (`custom-preset-panel`);
  - work memory import/export;
  - full glossary editor tabs.
- Preset selection remains primary (`data-testid="project-preset-panel"`), but preset details and custom preset creation live inside advanced.
- `handleStart()` now calls `ensureWorkContext()` first and opens warning modals for missing work/context/glossary before navigating to `/processing`.
- The E2E suite already covers this flow in `e2e/editor-rebuild.spec.ts` with tests for setup warnings, suggested glossary review, central glossary review, presets, and work memory.
- `src/lib/stores/appStore.ts` now includes `recovery` in `ImageLayerKey`; the web workspace/editor/export plan must carry `layers/recovery/` as a first-class layer.
- `npm run check` passed after this verification.

Implication for the site: do not build a separate simplistic wizard. Port this desktop setup contract to the web so both products share the same concepts and test vocabulary.

---

## Current Desktop Editor Verification

Verified on 2026-05-08 against the current dirty worktree:

- `src/lib/stores/editorStore.ts` now treats `recovery` as a mutable bitmap layer and includes it in the render-preview fingerprint.
- The desktop editor has a `repairBrush` path: painting recovery writes `layers/recovery/NNN.png`, restores original pixels into the inpaint layer, keeps recovery hidden by default, and marks preview/export stale.
- Brush layers are now RGBA, not just grayscale masks. `updateBrushRegion` accepts color, opacity, hardness and `dirty_bbox`.
- The lasso flow is first-class:
  - `src/lib/lassoSelection.ts` defines the selection shape and rasterizes it to PNG;
  - `src/components/editor/stage/LassoSelectionOverlay.tsx` renders the selection;
  - `src/components/editor/LassoContextMenu.tsx` can run detect/OCR/translate/inpaint on the selected region or apply the selection to mask.
- Page actions can run regionally by `bbox` or external mask:
  - `src/lib/tauri.ts` exposes `runPageActionWithOptionalMask({ bbox, mask_path })`;
  - `src-tauri/src/commands/pipeline.rs` appends `--region-bbox` and `--external-mask`;
  - `pipeline/main.py` filters detect/OCR/translate/inpaint to intersecting blocks/layers.
- `pipeline/recovery.py` can bake recovery pixels into rendered output, and `pipeline/tests/test_recovery_bake.py` covers it.
- Text style policy is now centralized in `src/lib/editorTextStylePolicy.ts`: canonical font, bold on, italic/shadow/glow/outline disabled while preserving color and size.
- `LayersPanel` separates primary bitmap layers from technical layers (`mask`, `recovery`) and exposes opacity only for the selected layer.
- The desktop E2E smoke now expects Lasso to be visible.

Implication for the site: the web editor cannot stop at text movement. Its API and workspace must carry region tools, lasso, recovery, RGBA brush metadata, and canonical text-style normalization from the beginning, even if some controls ship behind a narrow UI.

---

## Current Verification Gaps

Checked on 2026-05-08 after the editor changes:

- PASS: `npm run check`.
- PASS: `npm run test -- editorTextStylePolicy`.
- PASS: from `pipeline/`, `python -m pytest tests/test_page_action_region.py tests/test_recovery_bake.py -q`.
- FAIL: `npm run test:e2e -- e2e/editor-rebuild.spec.ts --grep "@smoke|@manual-flow|@setup"`.

Observed failures to fix before using desktop behavior as a site migration source:

- `@smoke`: `getByText("Pintura", { exact: true })` is now ambiguous because the selected-layer opacity header and layer row both render `Pintura`.
- `@manual-flow`: clicking `Traducao Manual` lands on `/setup`; the previous contract expected manual flow to prepare the fixture and open `/editor`.
- `@setup`: current setup screen does not expose `data-testid="setup-start-button"` where the tests expect it; the visible button is `Iniciar projeto manual`.
- `@setup`: current setup screen does not expose `data-testid="setup-advanced-panel"`; advanced sections appear flattened into the page.

Implication for execution: first repair or intentionally rewrite these desktop E2E contracts. Do not make the site copy missing test IDs or accidental flattened setup structure unless that is the intended product decision.

Resolved during execution on 2026-05-08:

- PASS: `npm run check`.
- PASS: `npm run test -- editorTextStylePolicy`.
- PASS: from `pipeline/`, `python -m pytest tests/test_page_action_region.py tests/test_recovery_bake.py -q`.
- PASS: `npm run test:e2e -- e2e/editor-rebuild.spec.ts --grep "@smoke|@manual-flow|@setup|@presets|@work-memory"` with 7 tests.
- Manual flow now reaches `/editor` through the processing path in E2E.
- Setup restored the semantic anchors required by the site migration contract.
- Glossary suggestion review has stable E2E IDs for the focused setup flow.

---

## Decisions

- O fluxo web deve usar `127.0.0.1` de ponta a ponta para evitar quebra de cookie/SSE entre `localhost` e `127.0.0.1`.
- O site não deve depender de Tauri. Toda ação de projeto/editor/export precisa ter uma rota HTTP equivalente.
- O formato canônico continua sendo `project.json` v2, com `paginas`, `image_layers`, `text_layers`, `visible`, `locked`, `order` e aliases legados.
- As camadas de imagem canônicas para o site devem incluir `base`, `mask`, `inpaint`, `brush`, `recovery` e `rendered`.
- O setup do site deve espelhar o desktop atual: obra/contexto primeiro, sugestões de glossário revisáveis, presets, painel avançado e memória da obra.
- Preview fiel deve ser não destrutivo: renderiza uma página temporária sem salvar automaticamente no `project.json`.
- Export só deve liberar quando o projeto/páginas estiverem em estado seguro. Se o preview estiver pendente ou com erro, mostrar bloqueio ou exigir modo debug.
- PSD por página já existe no desktop via Rust; para o site, criar uma CLI reutilizável em vez de reimplementar PSD em Python.

Additional editor decisions from 2026-05-08:

- `brush` deve preservar pixels RGBA e aceitar cor/opacidade/hardness; `mask` e `recovery` continuam camadas tecnicas.
- Acoes de pagina no site devem aceitar `bbox` ou mascara externa para rodar detect/OCR/translate/inpaint por regiao.
- Recovery deve ter dois contratos: camada editavel (`layers/recovery/`) e imagem final com recovery aplicado quando exportar formatos achatados.
- O editor web deve reutilizar a politica de estilo canonica do desktop antes de salvar ou renderizar texto.

---

## Target UX Flow

1. Landing pública em `/`.
2. Painel em `/dashboard`.
3. `Novo capítulo` abre `/projects/new`.
4. Usuário escolhe modo:
   - Automático: OCR + tradução + inpaint + typeset.
   - Manual: prepara projeto e vai direto para editor.
   - Lote: múltiplos capítulos em fila.
5. Usuário configura projeto:
   - upload ZIP/CBZ/imagens;
   - nome da obra;
   - busca de contexto da obra;
   - revisão explícita de sugestões de glossário;
   - capítulo;
   - idioma origem/destino;
   - preset visual;
   - preset customizado quando necessário;
   - memória da obra;
   - contexto/glossário avançado;
   - qualidade;
   - opções de export.
6. Usuário confirma e cria job.
7. `/jobs/:id` mostra processamento com eventos.
8. Ao concluir, `Abrir preview` leva para `/projects/:id/preview`.
9. Preview mostra páginas, original/traduzido, status QA, botões editar/exportar.
10. `Editar` abre `/projects/:id/editor?page=N`.
11. Editor permite mover texto, alterar estilo, camada, máscara, brush, OCR/tradução/inpaint por página/bloco.
12. Editor tambem precisa cobrir:
   - lasso com menu contextual;
   - aplicar lasso como mascara;
   - detect/OCR/translate/inpaint regional;
   - brush RGBA;
   - repair brush/recovery;
   - politica canonica de estilo de texto.
13. Export permite:
   - ZIP completo: projeto reimportável.
   - CBZ: páginas traduzidas achatadas.
   - JPG/ZIP: imagens finais.
   - PSD: página atual e depois lote.

---

## Phase 0: Inventory And Contracts

### Task 0.1: Confirm existing desktop contracts

**Files:**
- Read: `src/lib/tauri.ts`
- Read: `src/lib/stores/editorStore.ts`
- Read: `src-tauri/src/commands/project.rs`
- Read: `pipeline/main.py`
- Read: `server/jobs/api.py`
- Read: `server/workers/api.py`

**Steps:**
1. Confirm current Tauri functions used by desktop editor:
   - `loadEditorPage`
   - `saveProjectJson`
   - `renderPreviewPage`
   - `createEditorTextLayer`
   - `patchEditorTextLayer`
   - `deleteEditorTextLayer`
   - `setEditorLayerVisibility`
   - `updateMaskRegion`
   - `updateBrushRegion`
   - `updateRecoveryRegion` or equivalent recovery-layer writer if the desktop command uses a different name
   - `writeMaskFromPng`
   - `runPageActionWithOptionalMask` with `bbox` / `mask_path`
   - `retypesetPage`
   - `detectPage`
   - `ocrPage`
   - `translatePage`
   - `reinpaintPage`
   - `exportProject`
   - `exportPagePsd`
2. Confirm pipeline config shape:
   - `source_path`
   - `work_dir`
   - `models_dir`
   - `obra`
   - `capitulo`
   - `idioma_origem`
   - `idioma_destino`
   - `mode`
   - `export_mode`
   - page-action region args: `--region-bbox x1,y1,x2,y2` and `--external-mask <path>`
3. Confirm server job lifecycle:
   - upload;
   - enqueue;
   - worker claim;
   - worker events;
   - artifacts;
   - complete/fail/cancel.
4. Write a short contract note in `docs/plans/2026-05-08-desktop-to-site-flow-status.md`.

**Validation:**
- Run `npm run build`.
- Run `cd site; npm run build`.
- Run server tests if available: `python -m pytest server/tests -q`.

### Task 0.2: Repair desktop editor/setup regression gates

**Files:**
- Modify: `e2e/editor-rebuild.spec.ts`
- Modify: `src/pages/Home.tsx`
- Modify: `src/pages/Setup.tsx`
- Modify: `src/components/editor/LayersPanel.tsx`
- Reference: `src/lib/e2e/tauriMock.ts`
- Reference: `src/lib/e2e/fixtureProject.ts`

**Contract:**
- Manual flow remains a preparation path: after selecting a source, the app should prepare the project and land in `/editor`, not stop indefinitely on `/setup`.
- Setup can be visually reorganized, but the tested semantic anchors must remain stable while the site migration depends on them:
  - `project-name-input`
  - `work-context-summary`
  - `project-preset-panel`
  - `setup-start-button`
  - `setup-advanced-panel` or a deliberate replacement documented in this plan
  - `work-context-warning-modal`
- Bitmap layer rows need stable automation selectors because visible labels can now repeat in selected-layer summaries and rows.

**Step 1: Fix the bitmap-layer selector contract**

Add stable test IDs in `src/components/editor/LayersPanel.tsx`:

```tsx
<SortableImageLayerRow
  key={key}
  layerKey={key}
  data-testid={`bitmap-layer-row-${key}`}
  ...
/>
```

If passing `data-testid` through the component is awkward, add it to the row root:

```tsx
<div data-testid={`bitmap-layer-row-${layerKey}`} ...>
```

Then update the smoke assertion in `e2e/editor-rebuild.spec.ts`:

```ts
await expect(page.getByTestId("bitmap-layer-row-brush")).toBeVisible();
```

Expected: no strict-mode ambiguity from the duplicated visible text `Pintura`.

**Step 2: Restore the manual-flow product contract**

In `src/pages/Home.tsx`, split manual from auto after `validateImport(path)`:

```ts
if (mode === "manual") {
  setProject({
    ...baseProject,
    status: "processing",
    mode: "manual",
  });
  navigate("/processing");
  return;
}
```

The exact implementation should reuse the existing app store helpers and E2E mock path. The important behavior is:

```ts
await page.getByRole("button", { name: /Tradu..o Manual/i }).click();
await expect(page).toHaveURL(/\/editor$/);
await expect(page.getByTestId("editor-stage")).toBeVisible();
```

Do not route manual users through context/glossary warnings before the editor; that belongs to auto translation setup.

**Step 3: Restore setup primary action test ID**

In `src/pages/Setup.tsx`, add `data-testid="setup-start-button"` to the primary bottom action:

```tsx
<button
  data-testid="setup-start-button"
  onClick={() => {
    void handleStart();
  }}
  ...
>
  {project.mode === "manual" ? "Iniciar projeto manual" : "Traduzir"}
</button>
```

Expected: setup warning tests can click the same semantic action regardless of the visible label.

**Step 4: Decide and encode the advanced setup contract**

Preferred: restore a real collapsible wrapper for advanced setup with `data-testid="setup-advanced-panel"` and move these sections under it:

- `internet-context-panel`
- `custom-preset` controls
- `work-memory-panel`
- `glossary-editor`

Minimum acceptable fallback: if the product intentionally keeps setup flattened, update `e2e/editor-rebuild.spec.ts` and this plan to replace `setup-advanced-panel` with explicit panel-level IDs. Do not leave tests and product contract disagreeing.

Preferred wrapper:

```tsx
<details data-testid="setup-advanced-panel" open={advancedOpen} onToggle={...}>
  <summary>Avancado</summary>
  ...
</details>
```

Expected: `@setup`, `@presets` and `@work-memory` can target one advanced section consistently.

**Step 5: Re-run focused desktop gates**

```powershell
npm run check
npm run test -- editorTextStylePolicy
cd pipeline
python -m pytest tests/test_page_action_region.py tests/test_recovery_bake.py -q
cd ..
npm run test:e2e -- e2e/editor-rebuild.spec.ts --grep "@smoke|@manual-flow|@setup|@presets|@work-memory"
```

**Expected:**
- TypeScript passes.
- Text style policy test passes.
- Region/recovery Python tests pass.
- Focused desktop editor/setup Playwright slice passes before site migration begins.

---

## Phase 1: Project Configuration On The Site

### Task 1.1: Port desktop `Configurar projeto` contract to the site

**Files:**
- Modify: `site/src/App.tsx`
- Modify: `site/src/styles.css`
- Create: `site/src/projectConfig.ts`
- Create: `site/src/projectSetupApi.ts`
- Reference: `src/pages/Setup.tsx`
- Reference: `e2e/editor-rebuild.spec.ts`

**Implementation:**
- Add route `/projects/new`.
- Keep `/novo` as redirect to `/projects/new` for compatibility.
- Build the web setup in this order:
  - Entrada: upload ZIP/CBZ/images and mode selection.
  - Obra e contexto: same concept as `data-testid="work-context-summary"` in desktop.
  - Glossário sugerido: same concept as `data-testid="glossary-suggestions"`.
  - Preset: same concept as `data-testid="project-preset-panel"`.
  - Avançado: same concept as `data-testid="setup-advanced-panel"`.
  - Revisão final.
- Preserve the desktop copy style: short PT-BR labels, low jargon, clear warning modals.
- The site should use the same conceptual states as desktop:
  - missing work;
  - context found;
  - suggestions available;
  - accepted glossary count;
  - risk level;
  - user chose to continue without context.
- Store form state in a typed object:

```ts
export type WebProjectMode = "auto" | "manual" | "batch";
export type WebProjectQuality = "rapida" | "normal" | "alta";

export interface WebProjectConfig {
  mode: WebProjectMode;
  obra: string;
  capitulo: string;
  idioma_origem: string;
  idioma_destino: string;
  preset_id: string;
  preset?: unknown;
  qualidade: WebProjectQuality;
  export_mode: "clean" | "with_warnings" | "debug";
  contexto: {
    sinopse: string;
    genero: string[];
    personagens: string[];
    termos: string[];
    faccoes: string[];
    aliases: Record<string, string[]>;
    glossario: Record<string, string>;
    memoria_lexical: Record<string, string>;
    internet_context?: {
      internet_context_loaded?: boolean;
      rejected_glossary_candidates?: string[];
      glossary_candidates?: Array<{
        kind: string;
        source: string;
        target: string;
        confidence: number;
        status?: "pending" | "reviewed" | "rejected";
      }>;
    };
  };
  work_context?: {
    selected: boolean;
    work_id: string;
    title: string;
    context_loaded: boolean;
    internet_context_loaded: boolean;
    glossary_loaded: boolean;
    glossary_entries_count: number;
    risk_level: "high" | "medium" | "low";
    user_ignored_warning?: boolean;
  };
}
```

**Validation:**
- User can go from dashboard to `/projects/new`.
- Empty required fields show PT-BR validation.
- Manual mode clearly says it prepara o projeto e abre o editor.
- User can search a work and see suggestions before starting.
- User can accept, edit and ignore glossary suggestions.
- User can open advanced setup and create a custom preset.
- User can continue without context only after warning.
- Reuse comparable test ids where practical:
  - `work-context-summary`
  - `glossary-suggestions`
  - `project-preset-panel`
  - `setup-advanced-panel`
  - `setup-start-button`
  - `work-context-continue-without-context`

### Task 1.2: Add setup support endpoints

**Files:**
- Create: `server/projects/setup_api.py`
- Modify: `server/app.py`
- Create: `server/tests/test_project_setup_api.py`

**Endpoints:**
- `GET /api/setup/languages`
- `GET /api/setup/presets`
- `POST /api/setup/work-search`
- `POST /api/setup/work-context`
- `GET /api/setup/glossary/{work_id}`
- `POST /api/setup/glossary/{work_id}/entries`
- `DELETE /api/setup/glossary/{work_id}/entries/{entry_id}`
- `POST /api/setup/local-memory/export`
- `POST /api/setup/local-memory/import`

**Implementation:**
- Mirror the current Tauri calls used by desktop setup:
  - `loadSupportedLanguages`
  - `searchWork`
  - `enrichWorkContext`
  - `loadOrCreateWorkContext`
  - `loadGlossary`
  - `upsertGlossaryEntry`
  - `removeGlossaryEntry`
  - `exportLocalMemory`
  - `importLocalMemory`
- Do not send images/pages to online context lookup.
- Persist accepted/rejected glossary entries by `work_id`, not only inside a transient job form.

**Validation:**
- Work search returns candidates.
- Enriched context returns synopsis, characters, terms and glossary candidates.
- Accepted and rejected glossary entries persist across reload.
- Local memory export/import round-trips.

### Task 1.3: Persist config with job creation

**Files:**
- Modify: `server/models.py`
- Modify: `server/jobs/api.py`
- Modify: `server/tests/test_storage_contract.py`
- Modify: `server/tests/test_worker_api.py`
- Modify: `site/src/App.tsx`

**Implementation:**
- Add `config_json` or equivalent metadata to `Job`.
- Extend `POST /api/jobs` to accept:
  - file;
  - mode;
  - obra;
  - capitulo;
  - idioma_origem / `src_lang`;
  - idioma_destino / `dst_lang`;
  - `project_config` JSON string.
- Store the full setup payload, including:
  - selected preset;
  - work context summary;
  - accepted glossary;
  - rejected glossary candidates;
  - memory summary;
  - user ignored context warning flag.
- Return the job id and route to `/jobs/:id`.
- Ensure worker claim receives the config.

**Validation:**
- Server test creates a job and asserts config is stored.
- Worker claim test asserts config is returned.
- Site creates a real queued job with the wizard payload.
- Manual mode job keeps enough metadata to materialize a project and open editor directly after preparation.

---

## Phase 2: Worker Materializes A Web Project Workspace

### Task 2.1: Standardize worker output artifact kinds

**Files:**
- Modify: `server/workers/api.py`
- Modify: `server/jobs/api.py`
- Modify: `server/jobs/artifacts.py`
- Modify: `worker/` runner file if present
- Create/modify tests under `server/tests/`

**Artifact kinds:**
- `project_json`
- `translated_image`
- `original_image`
- `inpaint_image`
- `layer_mask`
- `layer_brush`
- `layer_recovery`
- `preview_image`
- `bundle_zip`
- `pipeline_log`
- `export_manifest`

**Implementation:**
- Keep existing artifact upload route but validate known kinds.
- Make job detail return artifact `kind`, `filename`, `size`, `id`, and a stable download URL.
- Treat `layer_recovery` as first-class, not as a generic debug artifact, because `ImageLayerKey` now includes `recovery`.

**Validation:**
- Completed job exposes `project_json` and translated images.
- Recovery layer artifacts are accepted, listed and downloadable.
- `/api/artifacts/:id` downloads each artifact.

### Task 2.2: Materialize project workspace on server

**Files:**
- Create: `server/projects/api.py`
- Create: `server/projects/workspace.py`
- Modify: `server/app.py`
- Create: `server/tests/test_project_workspace.py`

**Implementation:**
- Add `POST /api/jobs/{job_id}/materialize-project`.
- Build server-side workspace:

```text
data/projects/{job_id}/
  project.json
  originals/
  images/
  translated/
  layers/
    mask/
    brush/
    recovery/
    text-preview/
```

- Extract from `bundle_zip` when available.
- Otherwise reconstruct from individual artifacts.
- Save a `workspace_state.json` with:
  - job id;
  - source artifact ids;
  - materialized timestamp;
  - dirty flag.

**Validation:**
- Given a fixture bundle, endpoint creates a workspace and returns page count.
- Reject if job is not completed.
- Reject if artifact is missing.

---

## Phase 3: Preview Page

### Task 3.1: Add preview API

**Files:**
- Modify: `server/projects/api.py`
- Modify: `server/projects/workspace.py`
- Create: `server/tests/test_project_preview_api.py`

**Endpoints:**
- `GET /api/projects/{job_id}`
- `GET /api/projects/{job_id}/pages/{page_index}`
- `GET /api/projects/{job_id}/assets/{asset_path}`
- `POST /api/projects/{job_id}/pages/{page_index}/render-preview`

**Implementation:**
- `GET /api/projects/{job_id}` returns normalized `project.json`.
- Asset endpoint only serves files inside the project workspace.
- Page payload must expose image layer URLs for `base`, `mask`, `inpaint`, `brush`, `recovery` and `rendered` when present.
- Preview render endpoint calls the same pipeline contract as desktop:
  - writes override page JSON to temp;
  - calls `pipeline/main.py --render-preview-page <project> <page> <override> <output>`;
  - returns preview asset URL.

**Validation:**
- Path traversal is rejected.
- Preview render creates one image under `render-cache/preview`.
- It does not mutate `project.json`.

### Task 3.2: Build `/projects/:id/preview`

**Files:**
- Modify: `site/src/App.tsx`
- Modify: `site/src/styles.css`
- Create: `site/src/projectApi.ts`

**UI:**
- Header with obra, capítulo, total pages.
- Left rail with thumbnails.
- Main comparison:
  - Original;
  - Traduzido;
  - Preview fiel when available.
- Layer selector includes the same canonical image layers as the desktop store, including `recovery`.
- Status:
  - pronto;
  - precisa render;
  - erro;
  - export bloqueado.
- Actions:
  - Editar página;
  - Renderizar preview;
  - Exportar.

**Validation:**
- Completed job opens preview.
- Clicking thumbnail changes page.
- Render preview updates visible image.
- Export button disabled when preview state is not `fresh`.

---

## Phase 4: Web Editor Adapter

### Task 4.1: Create HTTP adapter matching Tauri editor API

**Files:**
- Create: `site/src/editor/editorApi.ts`
- Create: `server/projects/editor_api.py`
- Modify: `server/app.py`
- Create: `server/tests/test_editor_api.py`

**Adapter methods:**

```ts
export interface WebEditorApi {
  loadEditorPage(projectId: string, pageIndex: number): Promise<EditorPagePayload>;
  saveProjectJson(projectId: string, projectJson: unknown): Promise<void>;
  renderPreviewPage(projectId: string, pageIndex: number, page: PageData): Promise<string>;
  patchTextLayer(projectId: string, pageIndex: number, layerId: string, patch: unknown): Promise<TextEntry>;
  createTextLayer(projectId: string, pageIndex: number, payload: unknown): Promise<TextEntry>;
  deleteTextLayer(projectId: string, pageIndex: number, layerId: string): Promise<void>;
  setLayerVisibility(projectId: string, payload: unknown): Promise<void>;
  updateMaskRegion(projectId: string, pageIndex: number, payload: unknown): Promise<void>;
  updateBrushRegion(projectId: string, pageIndex: number, payload: unknown): Promise<void>;
  updateRecoveryRegion(projectId: string, pageIndex: number, payload: unknown): Promise<void>;
  writeMaskFromPng(projectId: string, pageIndex: number, pngData: string, op: "replace" | "add" | "subtract"): Promise<string>;
  runPageAction(projectId: string, pageIndex: number, action: "detect" | "ocr" | "translate" | "inpaint", region?: PageActionRegion): Promise<PageActionResult>;
}
```

**Implementation:**
- Keep response shapes compatible with `src/lib/tauri.ts`.
- Use the server workspace as project root.
- Reuse the same normalization rules:
  - hydrate relative asset paths to HTTP asset URLs on the client;
  - save relative paths back into `project.json`.
- Preserve `base`, `mask`, `inpaint`, `brush`, `recovery` and `rendered` as explicit layer keys through load, save and visibility changes.
- Match desktop bitmap payloads:
  - mask/recovery writes accept strokes, `dirty_bbox`, clear and erase;
  - brush writes accept strokes, `dirty_bbox`, clear, erase, color, opacity and hardness;
  - lasso writes can send a raster PNG to `mask`;
  - page actions can send `bbox` or `mask_path`.
- Add shared site types for:
  - `LassoSelection`;
  - `PageActionRegion`;
  - `BitmapLayerUpdatePayload`;
  - `PageActionResult.changed_assets`.

**Validation:**
- Load editor page returns page data with absolute asset URLs.
- Text patch updates `project.json`.
- Mask, brush and recovery writes update the expected files under `layers/`.
- Regional action by bbox only changes intersecting text/block data.
- Lasso PNG write creates/updates `layers/mask/NNN.png`.
- Invalid layer id returns 404.

### Task 4.2: Port editor screen to site

**Files:**
- Add deps in `site/package.json`: `konva`, `react-konva`, `zustand`, `framer-motion` if needed.
- Create: `site/src/editor/`
- Create: `site/src/pages/WebEditor.tsx` if pages are split, or integrate in `App.tsx`.
- Modify: `site/src/App.tsx`
- Modify: `site/src/styles.css`

**Implementation:**
- Start with the same user-facing controls as desktop:
  - thumbnails;
  - layer list;
  - primary bitmap layers separated from technical layers;
  - text box editing;
  - move/resize;
  - style controls;
  - brush, mask, lasso and repair brush tools;
  - undo/redo;
  - preview/render status.
- Do not bring every advanced desktop feature in the first pass.
- MVP editor scope:
  - view original/inpaint/recovery/rendered;
  - select text;
  - drag/resize text bbox;
  - edit translated text;
  - change font size, color, align through the canonical text-style policy;
  - draw brush with RGBA color/opacity;
  - draw mask;
  - draw recovery/repair brush;
  - create lasso selection and apply it to mask;
  - save;
  - render preview.
- Reuse or mirror these desktop modules instead of inventing parallel behavior:
  - `src/lib/editorTextStylePolicy.ts`;
  - `src/lib/lassoSelection.ts`;
  - `src/components/editor/stage/LassoSelectionOverlay.tsx`;
  - `src/components/editor/LassoContextMenu.tsx`;
  - `src/components/editor/stage/recoveryComposite.ts`.

**Validation:**
- Open `/projects/:id/editor?page=0`.
- Drag text layer and save.
- Reload page and confirm position persists.
- Draw lasso and apply it as mask.
- Run regional OCR from lasso and confirm only intersecting layers change.
- Paint recovery and confirm preview/export becomes stale.
- Render preview reflects change.

### Task 4.3: Add page/block actions

**Files:**
- Modify: `server/projects/editor_api.py`
- Modify: `site/src/editor/editorApi.ts`
- Modify: `site/src/editor/` UI
- Create/modify: `server/tests/test_editor_actions.py`

**Actions:**
- Detectar página.
- OCR página.
- Traduzir página.
- Reinpaint página.
- Processar bloco: OCR/traduzir.
- Detectar/OCR/traduzir/inpaint por lasso/bbox.

**Implementation:**
- Server calls existing `pipeline/main.py` commands:
  - `--detect-page`
  - `--ocr-page`
  - `--translate-page`
  - `--reinpaint-page`
  - `--process-block`
- For regional actions, append one of:
  - `--region-bbox x1,y1,x2,y2`;
  - `--external-mask <absolute-safe-mask-path>`.
- Never trust a browser-provided arbitrary path. If the client sends a mask, store it inside the project workspace first and pass only that safe path to the pipeline.
- Preserve non-intersecting text layers, inpaint blocks and translations exactly as the desktop regional path does.
- Return changed assets and require client refresh.

**Validation:**
- Each action returns a status payload.
- Bbox action keeps non-intersecting layers unchanged.
- External-mask action rejects paths outside the workspace.
- Changed page reloads updated `project.json`.

---

## Phase 5: Export

### Task 5.1: Add ZIP, CBZ and JPG export endpoints

**Files:**
- Create: `server/projects/export_api.py`
- Modify: `server/app.py`
- Create: `server/tests/test_project_exports.py`

**Endpoints:**
- `POST /api/projects/{job_id}/exports/zip-full`
- `POST /api/projects/{job_id}/exports/cbz`
- `POST /api/projects/{job_id}/exports/jpg-zip`

**Implementation:**
- `zip-full` includes:
  - `project.json`;
  - `translated/`;
  - `originals/`;
  - `images/`;
  - `layers/`, including `mask/`, `brush/`, `recovery/` and `text-preview/` when present;
  - `export_manifest.json`;
  - QA logs when present.
- `cbz` includes only final translated pages at archive root, sorted naturally.
- `jpg-zip` includes translated pages in `translated/`.
- Before writing flattened formats (`cbz`, `jpg-zip`), apply recovery consistently:
  - if the desktop pipeline already baked recovery into `translated/`, verify by manifest/state and do not double-apply;
  - otherwise compose rendered + original + `layers/recovery/NNN.png` using the same semantics as `pipeline/recovery.py`.
- Use `export_mode`:
  - `clean`: block high/critical issues;
  - `with_warnings`: allow warning/high if policy permits;
  - `debug`: allow but mark manifest.

**Validation:**
- CBZ output has `001.jpg`, `002.jpg`, no folders.
- ZIP full has `project.json` and preserves `layers/recovery/`.
- JPG zip has translated files.
- Recovery mask changes are visible in exported CBZ/JPG but remain editable in ZIP full.

### Task 5.2: Add PSD export through Rust CLI

**Files:**
- Create: `src-tauri/src/bin/traduzai_export.rs`
- Modify: `src-tauri/Cargo.toml`
- Create: `server/projects/psd_export.py`
- Modify: `server/projects/export_api.py`
- Create: `server/tests/test_psd_export_contract.py`

**Implementation:**
- Add CLI command:

```powershell
cargo run --bin traduzai_export -- psd-page --project D:\...\project.json --page 0 --out D:\...\page-001.psd
```

- CLI should call the same PSD export logic used by desktop.
- PSD output should include the same layer semantics as desktop, including recovered image content when present in `project.json`.
- PSD should keep editable layer intent where possible: text layers remain text, brush/recovery are image layers, and flattened preview reflects recovery-applied output.
- Server endpoint:
  - `POST /api/projects/{job_id}/exports/psd-page`
  - accepts `page_index`;
  - writes output under `data/projects/{job_id}/exports/`;
  - registers artifact.

**Validation:**
- Rust test confirms CLI writes PSD for fixture project.
- Server test can mock CLI success and returns artifact metadata.
- Manual smoke downloads `.psd`.

### Task 5.3: Build export UI

**Files:**
- Modify: `site/src/App.tsx`
- Modify: `site/src/styles.css`
- Modify: `site/src/projectApi.ts`

**UI:**
- Export modal from preview/editor.
- Formats:
  - ZIP completo;
  - CBZ;
  - JPG;
  - PSD página atual;
  - PSD todas as páginas later.
- Mode selector:
  - Seguro;
  - Com avisos;
  - Debug.
- Show blocked reason clearly in PT-BR.

**Validation:**
- User exports CBZ from preview.
- User exports PSD current page from editor.
- Blocked export shows reason and does not start download.

---

## Phase 6: Project Settings And Reimport

### Task 6.1: Add project settings page

**Files:**
- Modify: `site/src/App.tsx`
- Modify: `site/src/styles.css`
- Modify: `server/projects/api.py`
- Create: `server/tests/test_project_settings.py`

**Route:**
- `/projects/:id/settings`

**Fields:**
- Obra e contexto using the same setup summary shape.
- Accepted glossary, ignored/rejected suggestions and advanced context.
- Local work memory import/export.
- Obra.
- Capítulo.
- Idiomas.
- Preset.
- Qualidade.
- Glossário.
- Contexto.
- Export mode padrão.

**Validation:**
- Save updates `project.json`.
- Accepted/rejected glossary changes persist and are visible when returning to setup/settings.
- Preview/editor show updated title and chapter.

### Task 6.2: Add open/import existing project

**Files:**
- Modify: `site/src/App.tsx`
- Modify: `server/projects/api.py`
- Create: `server/tests/test_project_import.py`

**Implementation:**
- Upload `zip_full` from desktop export.
- Server extracts, validates `project.json`, creates a project record/workspace.
- Redirects to preview.

**Validation:**
- Export desktop ZIP full.
- Import ZIP full in site.
- Preview loads same page count and assets.

---

## Phase 7: QA And Release Gates

### Task 7.1: Add focused backend tests

**Commands:**

```powershell
python -m pytest server/tests/test_project_workspace.py server/tests/test_project_preview_api.py server/tests/test_project_exports.py -q
cd pipeline
python -m pytest tests/test_page_action_region.py tests/test_recovery_bake.py -q
cd ..
```

**Expected:**
- All tests pass.

### Task 7.2: Add focused frontend checks

**Commands:**

```powershell
cd site
npm run build
```

**Expected:**
- TypeScript and Vite build pass.

### Task 7.3: Add browser smoke

**Flow:**
- Open `http://127.0.0.1:5174/dashboard`.
- Create project.
- Upload fixture.
- Finish job with mock or fixture worker.
- Open preview.
- Render preview.
- Open editor.
- Move one text layer.
- Save.
- Export CBZ.

**Expected:**
- No console errors.
- Export artifact downloads.

### Task 7.4: Add desktop regression checks

**Commands:**

```powershell
npm run check
npm run test -- editorTextStylePolicy
npm run test -- editor
npx playwright test e2e/editor-rebuild.spec.ts --project=chromium --grep "@smoke|@setup|@manual-flow"
cd src-tauri
cargo check
cargo test export_cbz_flattens_translated_pages
cargo test export_page_psd_supports_webp_original
cargo test history_brush_restores_only_masked_pixels_from_base_snapshot
cargo test rgba_brush_keeps_previous_stroke_color_after_color_change
```

**Expected:**
- Existing desktop editor/export flow remains intact.
- Lasso is visible in the smoke test.
- Recovery and RGBA brush regression tests remain green.

---

## Implementation Order

1. Phase 0.1: contracts.
2. Phase 0.2: repair focused desktop editor/setup regression gates.
3. Phase 1: port verified desktop setup contract and job config.
4. Phase 2: worker/server project workspace.
5. Phase 3: preview API and preview page.
6. Phase 5.1: CBZ/JPG/ZIP exports.
7. Phase 4.1: editor HTTP adapter.
8. Phase 4.2: editor surface with text, layers, lasso, brush and recovery.
9. Phase 4.3: regional page/block actions.
10. Phase 5.2: PSD CLI/export.
11. Phase 6: settings and reimport.
12. Phase 7: QA gates.

This order ships useful value early: users can configure, process, preview and export before the full editor is complete.

---

## Risk Register

- **PSD:** desktop PSD lives in Rust/Tauri code. Do not rewrite in JS. Extract a CLI or shared Rust path.
- **Preview freshness:** export must not trust stale preview. Treat any non-`fresh` state as needing render/review.
- **Asset paths:** browser needs HTTP URLs, while `project.json` should keep relative paths. Keep conversion at API/client boundary.
- **Large chapters:** do not load all full images at once in preview/editor; thumbnails and current page only.
- **Manual mode:** manual processing should prepare project then open editor, not stop on setup.
- **Recovery double-apply:** flattened export must not apply recovery twice if `translated/` already contains recovered pixels.
- **Regional actions:** bbox/mask actions must preserve non-intersecting layers and blocks, otherwise a small lasso could destroy page context.
- **Brush storage:** web brush must stay RGBA-compatible with desktop; do not regress it to grayscale alpha.
- **Style policy drift:** site and desktop must share the canonical text-style policy, or editor output and PSD/export will disagree.
- **Host mismatch:** keep `127.0.0.1` everywhere for dev.
- **Legal/safety:** marketing and fixtures should use synthetic images, not real protected pages.

---

## Definition Of Done

- User can create/configure a project from the site.
- Worker produces a server-materialized project workspace.
- User can preview original/rendered/fresh preview.
- User can edit at least text layers in web editor.
- User can use lasso for mask and regional page actions.
- User can paint brush and recovery layers without breaking desktop-compatible `project.json`.
- User can export ZIP completo, CBZ, JPG ZIP.
- User can export PSD for current page.
- User can reimport ZIP completo.
- Desktop app flow still passes focused editor/export checks.

---

## Execution Status - 2026-05-08

Implemented in this execution pass:

- Site project wizard at `/projects/new`, with `/novo` compatibility redirect.
- Web setup contract for work context, glossary suggestions, presets, advanced options, final review and job config persistence.
- Setup support API under `/api/setup/*`.
- Job config persistence in `jobs.config_json`; worker claim now returns `project_config`.
- Standard artifact kinds for project/workspace/editor/export flow.
- Server project workspace materialization under `data/saas/storage/projects/{job_id}`.
- Preview API and `/projects/:id/preview`.
- HTTP editor adapter under `/api/projects/{id}/editor/*`.
- Initial web editor at `/projects/:id/editor`, including text selection, direct text drag, text patching, bitmap layer writes for mask/brush/recovery and page-action contract.
- ZIP completo, CBZ, JPG ZIP and current-page PSD export endpoints.
- Project settings page and ZIP completo reimport endpoint.
- Worker artifact collection for original, translated, inpaint, mask, brush and recovery outputs.
- Focused backend/worker/site tests for setup, workspace, preview, editor, exports and import.

Validated:

```powershell
python -m pytest server/tests -q
python -m pytest worker/tests -q
cd site; npm run build
cd ..; npm run check
npm run test -- editorTextStylePolicy
cd pipeline; python -m pytest tests/test_page_action_region.py tests/test_recovery_bake.py -q
cd ..; npm run test:e2e -- e2e/editor-rebuild.spec.ts --grep "@smoke|@manual-flow|@setup|@presets|@work-memory"
cd src-tauri; cargo check
```

Browser smoke:

- `http://127.0.0.1:8787/api/health` returned `ok`.
- `http://127.0.0.1:5174` returned HTTP 200.
- Playwright launched installed Chrome, logged in with the local admin account, opened `/projects/new?profile=manual`, and confirmed `setup-start-button` and `setup-advanced-panel` are visible.

Known remaining hardening:

- PSD endpoint currently writes a server-side PSD contract artifact; extracting the desktop Rust PSD exporter into a reusable CLI remains the next hardening step.
- Page actions are wired through the HTTP contract and persistence path; running the heavy pipeline command from the server should be enabled behind a safe execution flag/worker handoff before production use.
- The web editor now supports direct text drag and bitmap layer writes, but full desktop parity for lasso drawing/paint ergonomics still needs a dedicated canvas pass.

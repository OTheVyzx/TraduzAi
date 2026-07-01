# TraduzAI Studio Undo Para Todas Acoes Visuais Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Tornar undo/redo confiavel para toda acao editavel do Studio que altera texto, layout, camadas, mascara, brush, inpaint, rendered ou qualquer imagem derivada.

**Architecture:** O editor deve ser command-driven: mutadores visuais nao podem alterar `currentPage`, `project.json`, `image_layers` ou assets diretamente sem registrar um `EditorCommand`. A fase 1 fecha bypasses com comandos em memoria e `page-snapshot`; a fase 2 adiciona snapshots persistentes em disco para bitmap undo, mantendo `project.json` como autoridade do estado atual.

**Tech Stack:** React 19, Zustand, TypeScript, Vitest, Tauri v2/Rust, comandos IPC em `src/lib/tauri.ts` e `src-tauri/src/commands/project.rs`.

---

## Invariantes

- `project.json` continua sendo a fonte de verdade para o estado atual: `paginas[]`, `text_layers`, `image_layers` e aliases existentes.
- Historico nao deve virar contrato de exportacao. Snapshots de undo ficam em cache interno, por exemplo `editor_cache/bitmap_undo/...`.
- Toda acao visual deve cair em uma destas categorias:
  - comando pequeno em `EditorCommand`, como texto, estilo, bbox e propriedades de layer;
  - `bitmap-stroke` ou `bitmap-asset-replace` para alteracoes regionais de imagem;
  - `page-snapshot` para acoes de backend/pipeline que recarregam pagina inteira.
- `undoEditor()` e `redoEditor()` devem atualizar `currentPage`, `pendingEdits`, `pendingStructuralEdits`, `image_layers`, cache-bust de bitmap e preview stale de forma consistente.
- O primeiro corte pode manter historico runtime-only, mas bitmap undo nao deve depender de `bitmapCache` no corte final de Studio.

## Auditoria Atual

### Ja undoavel

- Texto traduzido via `useTextEditSession` como `edit-traduzido`.
- Estilo via `updatePendingEstilo` como `edit-estilo`.
- Transformacao de texto via `commitTextTransform`, incluindo batch de bbox + rotacao.
- Criar/deletar camada de texto.
- Visibilidade/lock de camada de texto.
- Brush/mask/eraser/recovery iniciados pelo stage, usando `bitmap-stroke` em `bitmapCache`.
- Healing brush na pagina ativa, usando paths before/after.
- `runProcessRegionFromSelection`, usando `page-snapshot`.

### Bypasses que precisam ser fechados

- `src/components/editor/PropertyEditor.tsx`: edicoes de traducao e bbox numerico usam `updatePendingEdit` sem historico.
- `src/lib/stores/editorStore.ts`: `applyLassoSelectionToMask`.
- `src/lib/stores/editorStore.ts`: `clearMask`.
- `src/lib/stores/editorStore.ts`: `setImageLayerOpacity`, `setImageLayerLocked`, `reorderImageLayers`.
- `src/lib/stores/editorStore.ts`: `toggleImageLayerVisibility`.
- `src/lib/stores/editorStore.ts`: `runMaskedAction` e `runMaskedActionFromLasso`.
- `src/lib/stores/editorStore.ts`: `retypesetCurrentPage`, `reinpaintCurrentPage`, `reProcessBlock`, `disconnectBlock`, `detectInPage`, `ocrInPage`, `translateInPage`.
- `src/lib/stores/editorStore.ts`: `applyBitmapStroke` depende do caller criar comando; isso deve virar wrapper obrigatorio ou comando interno.
- `src/lib/stores/editorStore.ts`: `healPaintedRegion` so registra historico se a pagina ainda esta ativa.

---

## Phase 1: Contrato de Historico e Testes Vermelhos

### Task 1: Declarar comandos faltantes no tipo `EditorCommand`

**Files:**
- Modify: `src/lib/editorHistory.ts`
- Test: `src/lib/__tests__/editorHistory.test.ts`

**Step 1: Write failing tests**

Adicionar testes para:
- `toggle-image-layer-visibility` reverte `image_layers[layerKey].visible`.
- `toggle-image-layer-lock` reverte `image_layers[layerKey].locked`.
- `edit-image-layer-props` reverte `opacity`.
- `reorder-image-layers` reverte `order`.
- `bitmap-asset-replace` troca `image_layers[layerKey].path` entre `beforePath` e `afterPath`.

**Step 2: Run failing tests**

Run:

```powershell
npm test -- src/lib/__tests__/editorHistory.test.ts
```

Expected: falha de tipo/implementacao para os novos comandos.

**Step 3: Implement minimal command support**

Adicionar variantes em `NonBatchCommand`:

```ts
| {
    type: "toggle-image-layer-visibility";
    layerKey: ImageLayerKey;
    before: boolean;
    after: boolean;
  }
| {
    type: "toggle-image-layer-lock";
    layerKey: ImageLayerKey;
    before: boolean;
    after: boolean;
  }
| {
    type: "edit-image-layer-props";
    layerKey: ImageLayerKey;
    before: { opacity?: number };
    after: { opacity?: number };
    touchedKeys: ("opacity")[];
  }
| { type: "reorder-image-layers"; before: ImageLayerKey[]; after: ImageLayerKey[] }
| {
    type: "bitmap-asset-replace";
    layerKey: "brush" | "mask" | "inpaint" | "rendered" | "recovery";
    beforePath: string | null;
    afterPath: string | null;
    bbox?: Bbox;
  }
```

Adicionar metodos em `WorkingStateDraft` para image layers:

```ts
setWorkingImageLayerVisibility(pageKey, layerKey, visible): void
setWorkingImageLayerLocked(pageKey, layerKey, locked): void
setWorkingImageLayerProps(pageKey, layerKey, patch): void
reorderWorkingImageLayers(pageKey, orderedKeys): void
setWorkingImageLayerPath(pageKey, layerKey, path): void
```

**Step 4: Verify**

Run:

```powershell
npm test -- src/lib/__tests__/editorHistory.test.ts
```

Expected: PASS.

### Task 2: Implementar working-state de image layers no store

**Files:**
- Modify: `src/lib/stores/editorStore.ts`
- Test: `src/lib/stores/__tests__/editorStoreHistory.test.ts`

**Step 1: Write failing tests**

Adicionar testes para:
- `toggleImageLayerVisibility("mask")` registra comando e undo/redo reverte `visible`.
- `setImageLayerLocked("brush", true)` registra comando e undo/redo reverte `locked`.
- `setImageLayerOpacity("inpaint", 0.5)` registra comando e undo/redo reverte `opacity`.
- `reorderImageLayers([...])` registra comando e undo/redo reverte `order`.

**Step 2: Run failing tests**

Run:

```powershell
npm test -- src/lib/stores/__tests__/editorStoreHistory.test.ts
```

Expected: FAIL porque os mutadores ainda bypassam historico.

**Step 3: Implement**

- Adicionar implementacoes dos novos metodos `setWorkingImageLayer*`.
- Converter `toggleImageLayerVisibility`, `setImageLayerLocked`, `setImageLayerOpacity`, `reorderImageLayers` para chamar `executeEditorCommand` ou `recordEditorCommand`.
- Manter persistencia atual via backend somente onde ela ja existia, mas garantir que o estado local e o historico sejam a primeira autoridade de UX.

**Step 4: Verify**

Run:

```powershell
npm test -- src/lib/stores/__tests__/editorStoreHistory.test.ts
```

Expected: PASS.

### Task 3: Fechar edicoes do `PropertyEditor`

**Files:**
- Modify: `src/components/editor/PropertyEditor.tsx`
- Modify if needed: `src/lib/stores/editorStore.ts`
- Test: `src/lib/stores/__tests__/editorStoreHistory.test.ts`

**Step 1: Write failing tests**

Adicionar cobertura store-level para:
- editar traducao por API equivalente ao painel de propriedades gera `edit-traduzido`;
- editar bbox numerico gera `edit-bbox` ou batch de transform;
- undo volta ao texto/bbox base e redo reaplica.

**Step 2: Implement**

- Substituir caminhos de `updatePendingEdit` que alteram traducao/bbox por APIs historizadas:
  - texto: usar sessao/commit explicito ou novo `commitTextEdit(layerId, before, after)`;
  - bbox: usar `commitTextTransform` ou novo `commitTextBbox`.

**Step 3: Verify**

Run:

```powershell
npm test -- src/lib/stores/__tests__/editorStoreHistory.test.ts
```

Expected: PASS.

---

## Phase 2: Mascara, Lasso e Bitmap Runtime

### Task 4: Tornar `clearMask` undoavel

**Files:**
- Modify: `src/lib/stores/editorStore.ts`
- Test: `src/lib/stores/__tests__/editorBitmapTools.test.ts`

**Step 1: Write failing test**

Teste:
- pagina tem `image_layers.mask.path = "mask-before.png"`;
- `clearMask()` chama backend mock e retorna `"mask-blank.png"`;
- comando `bitmap-asset-replace` ou `bitmap-stroke` entra na pilha;
- undo restaura `"mask-before.png"`;
- redo aplica `"mask-blank.png"`.

**Step 2: Implement**

- Capturar path anterior antes de chamar `writeMaskFromPng`.
- Atualizar `currentPage.image_layers.mask.path` com path retornado.
- Registrar comando historizado.
- Bump `mask` e marcar preview stale em undo/redo.

**Step 3: Verify**

Run:

```powershell
npm test -- src/lib/stores/__tests__/editorBitmapTools.test.ts
```

Expected: PASS.

### Task 5: Tornar `applyLassoSelectionToMask` undoavel

**Files:**
- Modify: `src/lib/stores/editorStore.ts`
- Test: `src/lib/stores/__tests__/editorBitmapTools.test.ts`

**Step 1: Write failing test**

Teste:
- lasso ativo escreve mascara por `writeMaskFromPng`;
- depois da acao, historico contem comando de mask;
- undo restaura mask anterior e limpa/atualiza selecao de forma previsivel;
- redo reaplica mask nova.

**Step 2: Implement**

- Usar mesmo helper de `clearMask` para registrar replace de mask.
- Garantir que `activeLassoSelection` e `maskInProgress` nao fiquem incoerentes apos undo.

**Step 3: Verify**

Run:

```powershell
npm test -- src/lib/stores/__tests__/editorBitmapTools.test.ts
```

Expected: PASS.

### Task 6: Remover contrato implicito de `applyBitmapStroke`

**Files:**
- Modify: `src/lib/stores/editorStore.ts`
- Modify: `src/components/editor/stage/useEditorStageController.ts`
- Test: `src/lib/stores/__tests__/editorBitmapTools.test.ts`

**Step 1: Write failing tests**

Adicionar teste que chama `applyBitmapStroke` diretamente e exige uma decisao explicita:
- ou retorna erro/nao-op se nao houver snapshot/command context;
- ou cria comando internamente.

**Step 2: Implement**

Escolha recomendada:
- criar wrapper `executeBitmapStrokeCommand(payload)` para uso publico;
- manter `applyBitmapStroke` como persistencia interna;
- atualizar callers do stage para usar o wrapper onde fizer sentido;
- documentar no tipo que `applyBitmapStroke` nao deve ser chamado por UI direta.

**Step 3: Verify**

Run:

```powershell
npm test -- src/lib/stores/__tests__/editorBitmapTools.test.ts
```

Expected: PASS.

---

## Phase 3: Page Snapshots Para Acoes de Backend/Pipeline

### Task 7: Criar helper `executePageSnapshotAction`

**Files:**
- Modify: `src/lib/stores/editorStore.ts`
- Test: `src/lib/stores/__tests__/editorBitmapTools.test.ts`

**Step 1: Write failing test**

Teste helper indiretamente:
- mock backend altera `text_layers` e `image_layers.inpaint.path`;
- acao captura `before`;
- `loadCurrentPage()` carrega `after`;
- `page-snapshot` entra na pilha;
- undo/redo troca toda a pagina.

**Step 2: Implement**

Adicionar helper interno:

```ts
async function executePageSnapshotAction(label: string, run: () => Promise<void>): Promise<void>
```

O helper deve:
- clonar `beforePage`;
- executar `flushAutoSave`/`commitEdits` quando necessario;
- executar `run`;
- carregar ou materializar `afterPage`;
- registrar `page-snapshot` se houver diferenca;
- bump de bitmap layers alteradas;
- marcar render preview stale/fresh conforme acao.

**Step 3: Verify**

Run:

```powershell
npm test -- src/lib/stores/__tests__/editorBitmapTools.test.ts
```

Expected: PASS.

### Task 8: Converter page actions

**Files:**
- Modify: `src/lib/stores/editorStore.ts`
- Test: `src/lib/stores/__tests__/editorBitmapTools.test.ts`

**Step 1: Write failing tests**

Adicionar testes para:
- `runMaskedAction("inpaint")`;
- `runMaskedActionFromLasso("inpaint")`;
- `detectInPage()`;
- `ocrInPage()`;
- `translateInPage()`.

Cada teste deve provar `do -> undo -> redo`.

**Step 2: Implement**

Converter cada funcao para `executePageSnapshotAction` com labels em PT-BR:
- `Detectar caixas`;
- `Refazer OCR`;
- `Traduzir pagina`;
- `Limpar fundo`;
- `Processar selecao`.

**Step 3: Verify**

Run:

```powershell
npm test -- src/lib/stores/__tests__/editorBitmapTools.test.ts
```

Expected: PASS.

### Task 9: Converter retypeset, reinpaint e process block

**Files:**
- Modify: `src/lib/stores/editorStore.ts`
- Test: `src/lib/stores/__tests__/editorBitmapTools.test.ts`

**Step 1: Write failing tests**

Adicionar testes para:
- `retypesetCurrentPage()`;
- `reinpaintCurrentPage()`;
- `reProcessBlock("ocr")`;
- `reProcessBlock("translate")`;
- `reProcessBlock("inpaint")`;
- `disconnectBlock()`.

**Step 2: Implement**

Envolver todos no helper de snapshot. Para `disconnectBlock`, se possivel preferir comando especifico de layout; se o backend continua sendo autoridade, usar `page-snapshot`.

**Step 3: Verify**

Run:

```powershell
npm test -- src/lib/stores/__tests__/editorBitmapTools.test.ts
```

Expected: PASS.

---

## Phase 4: Snapshot Persistente de Bitmap

### Task 10: Adicionar contrato IPC de bitmap undo snapshot

**Files:**
- Modify: `src/lib/editorBackend.ts`
- Modify: `src/lib/editorBackends/tauriEditorBackend.ts`
- Modify: `src/lib/tauri.ts`
- Modify: `src-tauri/src/commands/project.rs`
- Modify: `src-tauri/src/lib.rs`
- Test: `src-tauri/src/commands/project.rs`
- Test: `src/lib/__tests__/tauriPageCommands.test.ts` or relevant tauri contract test

**Step 1: Write failing Rust tests**

Adicionar testes para:
- criar snapshot before/after dentro de `editor_cache/bitmap_undo/page-0001/<commandId>/`;
- restore rejeita `../` e path absoluto fora do projeto;
- restore atualiza `image_layers[layerKey].path` sem mexer em aliases de exportacao indevidos;
- delete snapshots remove arquivos do command.

**Step 2: Add TS contract tests**

Garantir wrappers:

```ts
createBitmapUndoSnapshot(config)
restoreBitmapUndoSnapshot(config)
deleteBitmapUndoSnapshots(config)
```

**Step 3: Implement Rust commands**

Reutilizar padroes existentes:
- `resolve_project_file`;
- `edit_project_value`;
- `ensure_bitmap_layer_path`;
- helpers de snapshot usados por healing brush;
- validacao canonical para garantir snapshot dentro de `editor_cache/bitmap_undo`.

**Step 4: Wire IPC**

Registrar commands em `src-tauri/src/lib.rs`, `src/lib/tauri.ts`, `src/lib/editorBackend.ts` e adapter Tauri.

**Step 5: Verify**

Run:

```powershell
cd src-tauri
cargo test commands::project
cd ..
npm test -- src/lib/__tests__/tauriPageCommands.test.ts
```

Expected: PASS.

### Task 11: Migrar `bitmap-stroke` para snapshots persistentes

**Files:**
- Modify: `src/lib/editorHistory.ts`
- Modify: `src/lib/stores/editorStore.ts`
- Modify: `src/components/editor/stage/useEditorStageController.ts`
- Test: `src/lib/__tests__/editorHistory.test.ts`
- Test: `src/lib/stores/__tests__/editorBitmapTools.test.ts`

**Step 1: Write failing tests**

Testes:
- `bitmap-stroke` com `snapshot.beforePath/afterPath` funciona sem `bitmapCache`;
- undo chama restore before;
- redo chama restore after;
- pruning chama delete snapshots;
- fallback antigo de `bitmapCache` ainda passa durante migracao.

**Step 2: Implement**

- Extender comando `bitmap-stroke` com `snapshot?: { beforePath; afterPath }`.
- Mudar apply/revert para suportar restore async via store-level command execution ou separar sync state update e async restore.
- Se `editorHistory.ts` precisar continuar puro/sync, mover restore persistente para wrappers `undoEditor`/`redoEditor` no store.

**Step 3: Verify**

Run:

```powershell
npm test -- src/lib/__tests__/editorHistory.test.ts src/lib/stores/__tests__/editorBitmapTools.test.ts
```

Expected: PASS.

### Task 12: Cleanup de snapshots e limites

**Files:**
- Modify: `src/lib/editorHistory.ts`
- Modify: `src/lib/stores/editorStore.ts`
- Modify: `src-tauri/src/commands/project.rs`
- Test: `src/lib/__tests__/editorHistory.test.ts`
- Test: `src/lib/stores/__tests__/editorBitmapTools.test.ts`

**Step 1: Write tests**

Testar:
- novo comando apos undo descarta redo tail e apaga snapshots descartados;
- limite por pagina remove comandos antigos e seus snapshots;
- clear history remove snapshots daquela pagina.

**Step 2: Implement**

- Propagar `disposeCommand` para callback async ou fila de cleanup no store.
- Nao bloquear undo/redo se cleanup falhar; logar erro e manter UX.

**Step 3: Verify**

Run:

```powershell
npm test -- src/lib/__tests__/editorHistory.test.ts src/lib/stores/__tests__/editorBitmapTools.test.ts
```

Expected: PASS.

---

## Phase 5: Atalhos e UI de Confianca

### Task 13: Extrair e testar shortcuts

**Files:**
- Create: `src/lib/editorShortcuts.ts`
- Modify: `src/pages/Editor.tsx`
- Test: `src/lib/__tests__/editorShortcuts.test.ts`

**Step 1: Write tests**

Testar:
- Ctrl+Z chama undo fora de input;
- Ctrl+Y chama redo fora de input;
- Ctrl+Shift+Z chama redo;
- dentro de textarea/input/contenteditable nao intercepta undo nativo;
- Delete/Backspace so exclui layer fora de input.

**Step 2: Implement**

Extrair pure function do handler global para `editorShortcuts.ts`.

**Step 3: Verify**

Run:

```powershell
npm test -- src/lib/__tests__/editorShortcuts.test.ts
```

Expected: PASS.

### Task 14: Tornar labels do undo testaveis

**Files:**
- Create: `src/lib/editorHistoryLabels.ts`
- Modify: `src/components/editor/toolbar/UndoRedoControls.tsx`
- Test: `src/lib/__tests__/editorHistoryLabels.test.ts`

**Step 1: Write tests**

Testar labels para todos os comandos novos e antigos.

**Step 2: Implement**

Mover `getTopAction` e `labelForAction` para util puro.

**Step 3: Verify**

Run:

```powershell
npm test -- src/lib/__tests__/editorHistoryLabels.test.ts
```

Expected: PASS.

---

## Phase 6: Validacao Final

### Task 15: Rodar suite focada

Run:

```powershell
npm test -- src/lib/__tests__/editorHistory.test.ts src/lib/stores/__tests__/editorStoreHistory.test.ts src/lib/stores/__tests__/editorBitmapTools.test.ts src/lib/__tests__/editorShortcuts.test.ts src/lib/__tests__/editorHistoryLabels.test.ts
```

Expected: PASS.

### Task 16: Rodar typecheck

Run:

```powershell
npm run check
```

Expected: PASS.

### Task 17: Rodar Rust focado

Run:

```powershell
cd src-tauri
cargo test commands::project
```

Expected: PASS.

### Task 18: Smoke manual

Run:

```powershell
npm run tauri dev
```

Manual:
- abrir projeto com pagina traduzida;
- editar texto e Ctrl+Z/Ctrl+Y;
- mover texto e Ctrl+Z/Ctrl+Y;
- ocultar image layer e Ctrl+Z/Ctrl+Y;
- pintar brush/mask e Ctrl+Z/Ctrl+Y;
- limpar mask e Ctrl+Z/Ctrl+Y;
- rodar inpaint/retypeset e Ctrl+Z/Ctrl+Y;
- trocar pagina e confirmar que nao ha perda silenciosa de pendencias.

Expected: toda acao visual editavel volta e reaplica sem quebrar preview, layer panel ou project.json.

---

## Execucao com Agentes

Quando for implementar, usar workers com write sets separados:

- Worker A: `editorHistory.ts`, labels e testes puros.
- Worker B: `editorStore.ts` para image layer props e PropertyEditor.
- Worker C: bitmap/mask/lasso em `editorStore.ts` e `useEditorStageController.ts`.
- Worker D: page actions com `page-snapshot`.
- Worker E: Rust/IPC de snapshot persistente.
- Worker F: shortcuts/UI tests.

Integracao deve acontecer por fases, nao todos ao mesmo tempo, porque `editorStore.ts` sera arquivo de conflito. Antes de cada fase, rodar os testes focados existentes para fixar baseline.

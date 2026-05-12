# Editor Sem Camadas Tecnicas Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remover a experiencia de camadas bitmap/tecnicas do editor, mantendo `image_layers` apenas como detalhe interno de composicao/export, e garantir que o pincel de recuperacao nunca esconda os textos.

**Architecture:** O Stage volta a ter um modelo simples: uma imagem base editavel (`inpaint -> base -> original`), overlays de ferramentas quando necessario, e camadas de texto sempre renderizadas no modo de edicao traduzida. `image_layers` continua no `project.json` para pipeline, cache, PSD/export e compatibilidade, mas deixa de controlar selecao/visibilidade no editor diario. O painel lateral vira painel de textos, nao painel de camadas bitmap.

**Tech Stack:** React 19, TypeScript, Zustand, React Konva, Tauri IPC, Rust image commands, Vitest, Playwright.

---

## Diagnostico

Hoje o bug acontece porque:

- `useEditorStageController.ts` chama `selectImageLayer("inpaint")` quando a ferramenta e `repairBrush`.
- `editorStore.applyBitmapStroke()` tambem seta `selectedImageLayerKey` para `"inpaint"` depois que o backend persiste a recuperacao.
- `renderModeUtils.isBitmapInspectionLayer("inpaint")` retorna `true`.
- `useEditorStageController.ts` calcula `translatedEditing = viewMode === "translated" && !faithfulPreview && !bitmapInspection`.
- Resultado: ao aplicar recuperacao, o editor entra em modo de inspecao bitmap e deixa de desenhar `EditorTextLayer`. Clicar no olhinho da camada tecnica muda estado suficiente para sair desse caminho, por isso os textos voltam.

Esse comportamento nao deve existir no editor normal. A camada `recovery` e uma acao historica/tecnica, nao uma camada que o usuario precisa mostrar/ocultar.

---

## Decisoes

- Remover selecao de bitmap como conceito de UX principal.
- Nao usar `selectedImageLayerKey` para decidir se textos aparecem.
- Nao expor `mask` e `recovery` no painel lateral do editor normal.
- Nao exigir clique em olho, lock, opacidade ou ordem para ferramentas tecnicas.
- Manter `image_layers` no schema e backend para export, render, PSD e compatibilidade.
- Se ainda for necessario inspecionar bitmap, criar depois um modo debug separado, fora do fluxo principal. Nao implementar agora.

---

## Task 1: Teste de regressao para textos visiveis apos recovery

**Files:**
- Modify: `D:\TraduzAi\e2e\editor-rebuild.spec.ts`
- Optionally Modify: `D:\TraduzAi\src\lib\e2e\tauriMock.ts`

**Step 1: Criar teste Playwright que reproduz o bug**

Adicionar um teste que:

1. Abre o editor com projeto fixture.
2. Confirma que ha texto visivel no Stage ou no estado e2e `editor-stage-state`.
3. Seleciona `repairBrush` pelo atalho `R` ou botao da toolbar.
4. Faz um stroke no Stage.
5. Aguarda o stroke terminar.
6. Confirma que as camadas de texto continuam presentes/renderizaveis.
7. Confirma que `data-base-kind` nao forca um modo onde texto some.

Exemplo de intencao:

```ts
test("recovery brush keeps text layers visible without toggling technical layers", async ({ page }) => {
  await openFixtureEditor(page);
  const stage = page.getByTestId("editor-stage");
  await expect(stage).toBeVisible();

  const state = page.getByTestId("editor-stage-state");
  await expect(state).toHaveAttribute(/data-layers/, /.+/);

  await page.keyboard.press("r");
  const box = await stage.boundingBox();
  expect(box).not.toBeNull();
  await page.mouse.move(box!.x + box!.width / 2, box!.y + box!.height / 2);
  await page.mouse.down();
  await page.mouse.move(box!.x + box!.width / 2 + 24, box!.y + box!.height / 2 + 24);
  await page.mouse.up();

  await expect(state).toHaveAttribute(/data-layers/, /.+/);
  await expect(stage).not.toHaveAttribute("data-base-kind", "recovery");
});
```

**Step 2: Rodar o teste e confirmar falha atual**

Run:

```bash
npm run test:e2e -- e2e/editor-rebuild.spec.ts
```

Expected before implementation: o teste falha porque `selectedImageLayerKey` vira `inpaint` e `translatedEditing` fica falso.

---

## Task 2: Corrigir a causa imediata no Stage

**Files:**
- Modify: `D:\TraduzAi\src\components\editor\stage\useEditorStageController.ts`
- Modify: `D:\TraduzAi\src\components\editor\stage\renderModeUtils.ts`
- Modify: `D:\TraduzAi\src\components\editor\stage\EditorStage.tsx`

**Step 1: Parar de selecionar bitmap durante ferramentas**

Em `useEditorStageController.ts`, remover chamadas de selecao bitmap durante ferramentas tecnicas:

```ts
// Antes
selectImageLayer(toolMode === "brush" ? "brush" : toolMode === "repairBrush" ? "inpaint" : null);

// Depois
selectImageLayer(null);
```

Para mascara/lasso:

```ts
// Antes
selectImageLayer("mask");

// Depois
selectImageLayer(null);
```

Para commit de lasso, remover:

```ts
selectImageLayer("mask");
```

**Step 2: Nao deixar selecao bitmap desligar textos**

Em `useEditorStageController.ts`, substituir:

```ts
const bitmapInspection = isBitmapInspectionLayer(selectedImageLayerKey);
const translatedEditing = viewMode === "translated" && !faithfulPreview && !bitmapInspection;
```

por:

```ts
const bitmapInspection = false;
const translatedEditing = viewMode === "translated" && !faithfulPreview;
```

Ou remover `bitmapInspection` completamente se `EditorStage.tsx` nao precisar mais do badge.

**Step 3: Desacoplar imagem exibida de `selectedImageLayerKey`**

Em `useEditorStageController.ts`, alterar:

```ts
const displayImagePath = useMemo(
  () => displayImagePathForMode(currentPage, viewMode, renderPreviewState, selectedImageLayerKey),
  [currentPage, renderPreviewState, selectedImageLayerKey, viewMode],
);
```

para:

```ts
const displayImagePath = useMemo(
  () => displayImagePathForMode(currentPage, viewMode, renderPreviewState),
  [currentPage, renderPreviewState, viewMode],
);
```

Em `renderModeUtils.ts`, remover o parametro `selectedImageLayerKey` de `displayImagePathForMode` ou manter opcional apenas para futuro debug, sem uso no editor normal.

**Step 4: Remover badge de inspecao bitmap do Stage**

Em `EditorStage.tsx`, remover o bloco:

```tsx
{bitmapInspection && (
  <div>Bitmap: {selectedImageLayerKey}</div>
)}
```

Tambem remover `selectedImageLayerKey` e `bitmapInspection` do destructuring se ficarem sem uso.

**Step 5: Rodar validacao**

Run:

```bash
npm run check
npm run test:e2e -- e2e/editor-rebuild.spec.ts
```

Expected: TypeScript passa e o teste de recovery mantem textos visiveis.

---

## Task 3: Corrigir store para strokes bitmap nao mudarem selecao visual

**Files:**
- Modify: `D:\TraduzAi\src\lib\stores\editorStore.ts`
- Test: `D:\TraduzAi\src\lib\stores\__tests__\editorStoreHistory.test.ts` ou novo `D:\TraduzAi\src\lib\stores\__tests__\editorBitmapTools.test.ts`

**Step 1: Criar teste unitario de selecao**

Adicionar teste que valida o contrato:

- Aplicar `repairBrush` nao seta `selectedImageLayerKey`.
- Aplicar `brush` nao seta `selectedImageLayerKey`.
- Aplicar `mask` nao seta `selectedImageLayerKey`.
- Se havia `selectedLayerId` de texto, o stroke pode limpar a selecao ativa se necessario, mas nao deve entrar em modo bitmap.

**Step 2: Alterar `applyBitmapStroke`**

Em `editorStore.ts`, no final de `applyBitmapStroke`, substituir:

```ts
set({
  currentPage: updatedPage,
  selectedImageLayerKey: visibleLayerKey,
  selectedLayerId: null,
  lastRetypesetTime: Date.now(),
});
```

por:

```ts
set({
  currentPage: updatedPage,
  selectedImageLayerKey: null,
  selectedLayerId: null,
  lastRetypesetTime: Date.now(),
});
```

Manter:

```ts
get().bumpBitmapLayerVersion(visibleLayerKey as MutableBitmapLayerKey);
```

porque o cache da imagem ainda precisa atualizar.

**Step 3: Alterar `toggleImageLayerVisibility` temporariamente**

Enquanto o painel antigo ainda existir, mudar:

```ts
set({ currentPage: updatedPage, selectedImageLayerKey: layerKey });
```

para:

```ts
set({ currentPage: updatedPage, selectedImageLayerKey: null });
```

Isso impede que clicar em olho entre em inspecao bitmap.

**Step 4: Rodar testes**

Run:

```bash
npm run check
npm run test -- src/lib/stores/__tests__/editorBitmapTools.test.ts
```

Expected: selecao bitmap nao e ativada por ferramentas.

---

## Task 4: Substituir painel de camadas por painel de textos

**Files:**
- Modify: `D:\TraduzAi\src\components\editor\LayersPanel.tsx`
- Keep: `D:\TraduzAi\src\components\editor\LayerItem.tsx`
- Modify if needed: `D:\TraduzAi\src\pages\Editor.tsx`

**Step 1: Reduzir `LayersPanel` para texto**

Remover de `LayersPanel.tsx`:

- `DndContext`
- `SortableContext`
- `SortableImageLayerRow`
- `DEFAULT_IMAGE_LAYER_ORDER`
- `IMAGE_LAYER_LABELS`
- `TECHNICAL_LAYERS`
- `showTechnical`
- Controles de opacidade bitmap
- Olho/lock/reorder de bitmap
- Secao "Bitmap"
- Secao "Tecnicas"

Manter:

- Busca por texto
- Contagem de texto
- Botao salvar alteracoes
- Botao excluir camada de texto
- Lista de `LayerItem`

O painel pode continuar se chamando `LayersPanel` internamente por menor diffs, mas o titulo visivel deve virar `Textos` ou `Camadas de texto`.

Estrutura desejada:

```tsx
export function LayersPanel() {
  const currentPage = useEditorStore((s) => s.currentPage);
  const selectedLayerId = useEditorStore((s) => s.selectedLayerId);
  const pendingEdits = useEditorStore((s) => s.pendingEdits);
  const deleteSelectedLayer = useEditorStore((s) => s.deleteSelectedLayer);
  const commitEdits = useEditorStore((s) => s.commitEdits);
  const [query, setQuery] = useState("");

  const textLayers = currentPage?.text_layers ?? [];
  const hasPendingEdits = Object.keys(pendingEdits).length > 0;
  const filteredTextLayers = useMemo(...);

  return (
    <div className="flex h-full w-[340px] flex-col border-l border-border bg-bg-primary">
      ...
      <span>Textos</span>
      ...
      {filteredTextLayers.map((entry, index) => (
        <LayerItem key={entry.id} entry={entry} index={index + 1} />
      ))}
    </div>
  );
}
```

**Step 2: Remover imports mortos**

Garantir que `LayersPanel.tsx` nao importe mais:

- `@dnd-kit/*`
- `ImageIcon`
- `GripVertical`
- `Lock`
- `LockOpen`
- `Eye`
- `EyeOff`
- `ImageLayerKey`
- `toggleImageLayerVisibility`
- `selectImageLayer`
- `setImageLayerOpacity`
- `setImageLayerLocked`
- `reorderImageLayers`

**Step 3: Rodar validacao visual basica**

Run:

```bash
npm run check
```

Expected: painel lateral continua aparecendo, mas sem camadas bitmap/tecnicas.

---

## Task 5: Limpar contratos de selecao bitmap do editor normal

**Files:**
- Modify: `D:\TraduzAi\src\lib\stores\editorStore.ts`
- Modify: `D:\TraduzAi\src\components\editor\stage\renderModeUtils.ts`
- Modify: `D:\TraduzAi\src\components\editor\stage\useEditorStageController.ts`
- Modify: `D:\TraduzAi\src\components\editor\stage\EditorStage.tsx`

**Step 1: Manter `selectedImageLayerKey` apenas se ainda houver uso real**

Depois das Tasks 2-4, rodar:

```bash
rg -n "selectedImageLayerKey|selectImageLayer|isBitmapInspectionLayer" src
```

Se sobrar apenas estado morto, remover do `EditorState`:

```ts
selectedImageLayerKey: ImageLayerKey | null;
selectImageLayer: (key: ImageLayerKey | null) => void;
```

E remover todos os `selectedImageLayerKey: null`.

Se a remocao for grande demais para este ciclo, manter o estado, mas garantir que nenhum fluxo visivel dependa dele.

**Step 2: Remover `isBitmapInspectionLayer` se ficar morto**

Em `renderModeUtils.ts`, remover:

```ts
export function isBitmapInspectionLayer(...)
```

se nao houver chamadas.

**Step 3: Garantir view modes simples**

Contrato final de `displayImagePathForMode`:

- `original` -> `base/original`
- `inpainted` -> `inpaint -> base -> original`
- `translated` com preview fresh -> preview fiel
- `translated` sem preview fresh -> `inpaint -> base -> original`

Nenhuma selecao de camada deve alterar esse resultado.

---

## Task 6: Preservar export/render com `image_layers`

**Files:**
- Read/Verify: `D:\TraduzAi\src-tauri\src\export\psd\*.rs`
- Read/Verify: `D:\TraduzAi\src-tauri\src\commands\project_schema.rs`
- Read/Verify: `D:\TraduzAi\pipeline\project_writer.py`
- Read/Verify: `D:\TraduzAi\src\lib\psd.ts`

**Step 1: Confirmar consumidores de export**

Rodar:

```bash
rg -n "image_layers|brush|mask|inpaint|rendered|recovery" src-tauri pipeline src/lib/psd.ts
```

**Step 2: Garantir que nenhuma mudanca remove schema**

Nao remover:

- `image_layers.base`
- `image_layers.inpaint`
- `image_layers.brush`
- `image_layers.mask`
- `image_layers.rendered`
- `image_layers.recovery` se ja existir em projetos antigos

**Step 3: Export deve continuar usando camadas internas**

Se o export PSD usa `image_layers`, manter.

Se a UI precisa de comando "exportar com camadas", ele deve usar o estado interno, nao uma configuracao visivel no editor.

---

## Task 7: Ajustar eraser e recovery para alvo por ferramenta, nao por camada selecionada

**Files:**
- Modify: `D:\TraduzAi\src\lib\stores\editorStore.ts`
- Modify: `D:\TraduzAi\src\pages\Editor.tsx`

**Step 1: Simplificar inferencia do eraser**

Hoje o eraser tenta inferir alvo de:

```ts
eraserTarget -> selectedImageLayerKey -> lastPaintedLayer
```

Remover `selectedImageLayerKey` da inferencia:

```ts
if (explicit) {
  layerKey = explicit;
} else {
  layerKey = get().lastPaintedLayer;
}
```

**Step 2: Definir targets explicitos por UI simples**

Na barra, manter apenas:

- `Pintura`
- `Mascara`

Nao oferecer `Recovery` como alvo de borracha no editor normal. Recovery deve ser acao sobre `inpaint`, nao camada que o usuario apaga.

**Step 3: Garantir recovery independente**

`repairBrush` sempre chama `updateRecoveryRegion`, independentemente de layer selecionada.

---

## Task 8: Testes finais e aceite

**Commands:**

```bash
npm run check
npm run test -- src/lib/__tests__/editorTextStylePolicy.test.ts
npm run test -- src/lib/stores/__tests__/editorBitmapTools.test.ts
npm run test:e2e -- e2e/editor-rebuild.spec.ts
cargo check
cargo test history_brush_restores_only_masked_pixels_from_base_snapshot
```

**Acceptance Criteria:**

- Aplicar pincel de recuperacao mostra o efeito instantaneamente.
- Textos nao somem depois do stroke.
- Usuario nao precisa clicar no olho de `Tecnicas > Recuperacao`.
- Painel lateral nao mostra camadas bitmap/tecnicas no editor normal.
- Ferramentas continuam funcionando:
  - brush pinta.
  - eraser apaga pintura/mascara conforme alvo simples.
  - repairBrush restaura pixels originais no inpaint.
  - mask/lasso segue funcionando.
- Export/render continuam lendo `image_layers` internamente.
- Nenhum projeto antigo quebra por falta de `recovery` ou `mask`.

---

## Ordem Recomendada de Execucao

1. Task 1 para travar o bug em teste.
2. Task 2 e Task 3 para corrigir o sumico de textos imediatamente.
3. Task 4 para remover a UI de camadas bitmap/tecnicas.
4. Task 5 e Task 7 para limpar acoplamentos restantes.
5. Task 6 para conferir export.
6. Task 8 como gate final.

## Nota de escopo

Nao fazer migracao pesada de `project.json`. O objetivo e remover a feature do editor, nao do formato interno. O schema continua guardando camadas porque pipeline/export precisam delas.

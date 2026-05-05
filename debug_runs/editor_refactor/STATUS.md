# TraduzAi Editor Refactor — Status

Última atualização: 2026-05-05

## Fases concluídas

### ✅ Fase 0 — Diagnóstico Pipeline
**Concluída em:** 2026-05-05  
**Commit:** (incluso no commit "Fases 0–3")

**O que foi feito:**
- Captura de stderr do sidecar Python (era `Stdio::null()` → `Stdio::piped()`)
- Helper `spawn_with_stderr_capture` + `format_pipeline_error` em `pipeline.rs`
- Logs `[EditorAction] start/success/error` no Rust (project.rs + pipeline.rs)
- `log_editor_action()` no Python (`pipeline/main.py`)
- Dispatcher unificado em Python para as 4 ações de página com try/except + logging
- Correção crítica: `changed_assets` sempre retornava `[ProjectJson]` → corrigido por action type
- Frontend `runMaskedAction` envolto em try/catch com `pageActionError` surfaceado como banner

**Arquivos modificados:**
- `src-tauri/src/commands/pipeline.rs`
- `src-tauri/src/commands/project.rs`
- `pipeline/main.py`
- `src/lib/stores/editorStore.ts`
- `src/pages/Editor.tsx`

---

### ✅ Fase 1 — Limpeza Visual
**Concluída em:** 2026-05-05  
**Commit:** (incluso no commit "Fases 0–3")

**O que foi feito:**
- Zoom extraído para `src/components/editor/toolbar/ZoomControls.tsx` (novo componente)
- Zoom montado na toolbar superior (topo, à direita das ações pipeline)
- `ZoomControls` removido do `EditorStage.tsx`
- Filename labels removidos do `LayersPanel.tsx`
- Botões Salvar/Descartar removidos da UI principal (funções preservadas)
- Botões Preview/Render removidos (atalhos `Ctrl+Shift+R` e `Ctrl+Shift+P` adicionados)

**Arquivos modificados:**
- `src/components/editor/stage/EditorStage.tsx`
- `src/components/editor/LayersPanel.tsx`
- `src/pages/Editor.tsx`

**Arquivos novos:**
- `src/components/editor/toolbar/ZoomControls.tsx`

---

### ✅ Fase 2 — Fontes Bundle
**Concluída em:** 2026-05-05  
**Commit:** (incluso no commit "Fases 0–3")

**O que foi feito:**
- `public/fonts/` criado com os 5 arquivos de fonte (servidos pelo Vite/Tauri)
- `src/lib/fonts.ts` criado com `FONT_REGISTRY`, `preloadEditorFonts()`, `resolveLegacyFontFamily()`
- `@font-face` com `font-weight`/`font-style` explícitos adicionados em `globals.css`
- `fontFamilyFromStyle()` atualizado para usar `resolveLegacyFontFamily()`
- `preloadEditorFonts()` chamado no mount do Editor (garante redraw correto do Konva)

**Arquivos modificados:**
- `src/styles/globals.css`
- `src/components/editor/stage/textLayerStyleUtils.ts`
- `src/pages/Editor.tsx`

**Arquivos novos:**
- `src/lib/fonts.ts`
- `public/fonts/` (5 arquivos .ttf)

**Pendente (Fase 2B/2C):**
- Importação manual de fontes via dialog Tauri (`import_font_to_project` Rust command)
- Fontes do sistema via `queryLocalFonts()` + fallback Rust `list_system_fonts`

---

### ✅ Fase 3 — Auto-save Híbrido
**Concluída em:** 2026-05-05  
**Commit:** (incluso no commit "Fases 0–3")

**O que foi feito:**
- Estado auto-save adicionado ao `editorStore.ts`: `dirty`, `lastSavedAt`, `autoSaveStatus`, `saveVersion`, `saveInFlightVersion`, `lastSaveError`
- Métodos: `markDirty()`, `commitEditsPatchOnly()`, `runAutoSave()`, `flushAutoSave()`, `pauseAutoSave()`, `resumeAutoSave()`
- Subscriber Zustand detecta mudanças em `pendingEdits`/`pendingStructuralEdits` → chama `markDirty()` automaticamente
- `setCurrentPage` faz `flushAutoSave()` antes de trocar de página
- `setInterval(3000)` no Editor com flush em `beforeunload`, `pagehide`, `visibilitychange`
- `AutoSaveIndicator.tsx` com 5 estados visuais

**Arquivos modificados:**
- `src/lib/stores/editorStore.ts`
- `src/pages/Editor.tsx`

**Arquivos novos:**
- `src/components/editor/toolbar/AutoSaveIndicator.tsx`

---

---

### ✅ Fase 4 — Reorganização da UI
**Concluída em:** 2026-05-05  
**Commit:** (incluso no commit "Fase 4")

**O que foi feito:**
- `TypesettingBar.tsx` criado — barra horizontal contextual com Fonte/Tamanho/Cor/Alinhamento/B/I + popovers Contorno/Sombra/Brilho. Aparece abaixo da toolbar quando texto está selecionado.
- `ToolSidebar.tsx` criado — sidebar vertical 44px com Selecionar/Novo Bloco/Brush/Borracha/Máscara + atalhos de teclado visíveis. Posicionada entre PageThumbnails e canvas.
- PROPRIEDADES removida do `LayersPanel.tsx` (import `PropertyEditor` eliminado)
- `Editor.tsx`: horizontal TOOL_MODES segmented control removido; ToolSidebar + TypesettingBar montados
- `npm run check` ✓ · `npm run test` 50/50 ✓

**Arquivos modificados:**
- `src/components/editor/LayersPanel.tsx`
- `src/pages/Editor.tsx`

**Arquivos novos:**
- `src/components/editor/toolbar/TypesettingBar.tsx`
- `src/components/editor/toolbar/ToolSidebar.tsx`

---

---

### ✅ Fase 5 — FloatingTextEditor
**Concluída em:** 2026-05-05

**O que foi feito:**
- `FloatingTextEditor.tsx` criado — painel flutuante com Original (readonly) + Tradução (editável) + botão Restaurar + fechar (Esc/Ctrl+Enter/X)
- Posicionamento dinâmico: topo-direito do bbox → à direita; se não couber → à esquerda; senão → abaixo
- Clamp na viewport com margem de 12px
- Coordenadas via `imageToContainer()` usando stageScale + panOffset + containerSize
- `containerSize` adicionado ao return do `useEditorStageController`
- `FloatingTextEditor` montado dentro de `EditorStage.tsx` com acesso ao controller
- `npm run check` ✓ · `npm run test` 50/50 ✓

**Arquivos modificados:**
- `src/components/editor/stage/EditorStage.tsx`
- `src/components/editor/stage/useEditorStageController.ts`

**Arquivos novos:**
- `src/components/editor/stage/FloatingTextEditor.tsx`

---

## Fases pendentes

| Fase | Descrição | Status |
|------|-----------|--------|
| 6 | Live Preview + Auto Fidelity Render + RenderStatusBadge | 🔲 Próxima |
| 7 | Brush Photoshop (layer `paint`, cor/opacidade/dureza) | 🔲 Pendente |
| 8 | Máscara Lasso (freehand + poligonal + ops add/subtract/replace) | 🔲 Pendente |
| 9 | Borracha inteligente (alvo paint/mask) | 🔲 Pendente |
| 10 | BITMAP como Layers MVP (drag reorder, opacity, lock, thumbnails) | 🔲 Pendente |
| 11 | Undo/Redo | 🔲 Pendente |
| 12 | E2E final + relatório | 🔲 Pendente |

---

## Onde parei

**Próximo passo:** Iniciar **Fase 4 — Reorganização da UI**.

### Fase 4 — plano de execução
1. **TypesettingBar** (`src/components/editor/toolbar/TypesettingBar.tsx`)
   - Migrar Estilo+Efeitos do `PropertyEditor.tsx` para barra horizontal
   - Sub-componentes: `FontSelect`, `SizeStepper`, `ColorChip`, `AlignToggle`, `EffectPopover`
   - Montar em `Editor.tsx` condicionalmente (só quando layer de texto selecionada)

2. **ToolSidebar** (`src/components/editor/toolbar/ToolSidebar.tsx`)
   - Substituir segmented horizontal de TOOL_MODES por sidebar vertical ~48px
   - Ferramentas: Selecionar(V), Mover(H), Texto(T), Brush(B), Borracha(E), Máscara(L)
   - Posicionado entre PageThumbnails e Canvas

3. **Remover PROPRIEDADES** do `LayersPanel.tsx`
   - Seção "PROPRIEDADES" com textos/estilos vai para TypesettingBar e FloatingTextEditor

4. Verificar com `npm run check`

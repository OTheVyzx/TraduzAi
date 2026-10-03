# TraduzAI Studio Master Editor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transformar o Studio no editor mestre pós-tradução para scanlation, começando pela separação completa das superfícies de pipeline e pela base do documento profissional em camadas.

**Architecture:** O editor compartilhado recebe um modo explícito e resolve capacidades sem alterar o comportamento padrão do TraduzAI Central. Em seguida, um `studio_scene` aditivo passa a representar camadas livres e continua projetando `image_layers` e `text_layers` nas fronteiras de compatibilidade.

**Tech Stack:** React 19, TypeScript, Zustand, Konva, Canvas 2D, Tauri v2, Vitest, ag-psd.

---

## Progresso de execução

- Tarefas 1–4 concluídas no primeiro corte editorial.
- Tarefa 5 concluída no segundo corte estrutural.
- Tarefa 6 concluída no terceiro corte de camadas profissionais.
- Tarefa 7 concluída no quarto corte de estilos compartilhados.
- Tarefa 8 concluída no sexto corte de seleções, máscaras por camada e retoque não destrutivo.
- Tarefa 9 concluída no sétimo corte de preenchimento generativo local e não destrutivo.
- Tarefa 10 permanece como roadmap subsequente.

### Task 1: Contrato de modo do editor

**Files:**
- Create: `src/components/editor/editorMode.ts`
- Create: `studio/src/editor/__tests__/editorMode.test.ts`
- Modify: `src/pages/Editor.tsx`
- Modify: `studio/src/editor/StudioSharedEditor.tsx`

**Step 1: Write the failing test**

Testar que o modo `studio` desliga ações de pipeline, idioma de origem, ações automáticas por bloco e ferramentas automáticas, preservando seleção, texto, brush, borracha e máscara.

**Step 2: Run test to verify it fails**

Run: `npm --prefix studio test -- src/editor/__tests__/editorMode.test.ts`

Expected: FAIL porque `editorMode.ts` ainda não existe.

**Step 3: Write minimal implementation**

Criar `resolveEditorCapabilities()` e `isEditorToolVisible()` com `traduzai` como comportamento padrão e `studio` como perfil editorial.

**Step 4: Run test to verify it passes**

Run: `npm --prefix studio test -- src/editor/__tests__/editorMode.test.ts`

Expected: PASS.

### Task 2: Entrada do Studio orientada a projeto traduzido

**Files:**
- Create: `studio/src/__tests__/StudioHome.test.ts`
- Modify: `studio/src/App.tsx`

**Step 1: Write the failing test**

Renderizar `StudioHome` estaticamente e exigir “Abrir projeto TraduzAI”, descrição de pós-tradução e ausência de “Novo projeto”.

**Step 2: Run test to verify it fails**

Run: `npm --prefix studio test -- src/__tests__/StudioHome.test.ts`

Expected: FAIL com a copy atual “Novo projeto”.

**Step 3: Write minimal implementation**

Remover o projeto de demonstração da ação principal, abrir o diálogo real e atualizar a copy para pós-tradução.

**Step 4: Run test to verify it passes**

Run: `npm --prefix studio test -- src/__tests__/StudioHome.test.ts`

Expected: PASS.

### Task 3: Remover superfícies automáticas do workspace Studio

**Files:**
- Modify: `src/pages/Editor.tsx`
- Modify: `src/components/editor/toolbar/ToolSidebar.tsx`
- Modify: `src/components/editor/LayersPanel.tsx`
- Modify: `src/components/editor/LayerItem.tsx`
- Create: `studio/src/editor/__tests__/studioEditorSurface.test.ts`

**Step 1: Write the failing tests**

Renderizar as superfícies puras/isoláveis e provar que o perfil Studio não oferece `process`, `repairBrush`, `reinpaintBrush`, OCR, Traduzir ou Limpar.

**Step 2: Run tests to verify they fail**

Run: `npm --prefix studio test -- src/editor/__tests__/studioEditorSurface.test.ts`

Expected: FAIL porque os componentes ainda não aceitam o modo Studio.

**Step 3: Write minimal implementation**

Propagar `mode` de `Editor` para `ToolSidebar`, `LayersPanel` e `LayerItem`; condicionar idioma, sidebar automática, banner e atalhos de ferramentas escondidas.

**Step 4: Run tests to verify they pass**

Run: `npm --prefix studio test -- src/editor/__tests__/studioEditorSurface.test.ts`

Expected: PASS.

### Task 4: Validação do primeiro corte

**Files:**
- Modify: `studio/README.md`

**Step 1:** Atualizar o escopo documentado do Studio.

**Step 2:** Run `npm --prefix studio test`.

**Step 3:** Run `npm --prefix studio run build`.

**Step 4:** Inspecionar `git diff -- studio src/pages/Editor.tsx src/components/editor docs/plans` e confirmar que mudanças externas não foram tocadas.

### Task 5: `studio_scene` aditivo e round-trip

**Status:** concluída em 2026-07-13.

**Files:**
- Modify: `studio/src/project/studioProject.ts`
- Modify: `studio/src/project/adapters.ts`
- Modify: `studio/schemas/studio_project.schema.json`
- Modify: `studio/src/project/__tests__/adapters.test.ts`

Adicionar nós raster/text/group/mask/generated/adjustment/fill, derivar uma cena para projetos legados e preservar projeção compatível.

O contrato implementado mantém `image_layers` e `text_layers` como fonte compatível nesta fase. O adaptador deriva nós somente para conteúdo existente, preserva hierarquia/metadados livres e reconcilia visibilidade, bloqueio e opacidade nas fronteiras de persistência. Nós legados projetados também são removidos quando a camada de origem deixa de existir.

### Task 6: Árvore profissional de camadas

**Status:** concluída em 2026-07-13.

**Files:**
- Create: `studio/src/editor/layers/StudioLayersTree.tsx`
- Create: `studio/src/store/studioSceneStore.ts`
- Modify: `studio/src/editor/StudioSharedEditor.tsx`
- Create: `studio/src/editor/layers/studioScenePersistence.ts`
- Create: `src/components/editor/EditorLayersPanelSlot.tsx`
- Modify: `src/pages/Editor.tsx`

Implementar seleção, reordenação, grupos, visibilidade, bloqueio, opacidade e blend com comandos transacionais.

O painel profissional substitui somente a lateral do modo Studio. As operações usam snapshots `before/after`, confirmam o histórico apenas depois da persistência e restauram a cena inteira em caso de falha. Grupos projetam visibilidade, bloqueio e opacidade efetivos para as camadas compatíveis sem destruir o estado intrínseco dos filhos. O TraduzAI Central continua usando o `LayersPanel` existente pelo slot padrão.

### Task 7: Motor de estilos

**Status:** concluída em 2026-07-13.

**Files:**
- Create: `src/lib/editorTextStyleResolver.ts`
- Create: `studio/src/styles/styleModel.ts`
- Create: `studio/src/styles/styleResolver.ts`
- Modify: `src/components/editor/stage/EditorTextLayer.tsx`
- Modify: `src/components/editor/stage/konvaTextStyleRenderer.tsx`
- Modify: `src/lib/konvaExportRenderer.ts`
- Modify: `studio/src/export/psd.ts`

Unificar tipografia, fills, múltiplos strokes e efeitos para canvas e exportação.

O contrato `studio_style` é aditivo e mantém os campos legados como fallback. Um resolvedor puro compartilhado alimenta a edição Konva, a renderização Konva de exportação e o PSD sem criar dependência do Central sobre o pacote Studio. Fills empilhados, strokes múltiplos, sombras e glow são passes não destrutivos; no PSD, tipografia e efeitos são gravados como propriedades editáveis de texto/camada. O round-trip preserva o contrato profissional dentro de `style`/`estilo`.

### Task 8: Seleções e retoque não destrutivo

**Status:** concluída em 2026-07-13.

**Files:**
- Modify: `src/components/editor/stage/useEditorStageController.ts`
- Create: `studio/src/editor/selection/selectionModel.ts`
- Create: `studio/src/editor/retouch/retouchCommands.ts`

Adicionar feather, expandir/contrair, camada-alvo, clone/healing/patch e máscaras por camada.

A seleção passou a aceitar regiões compostas de adição/subtração, feather, expansão ou contração e camada-alvo. O rasterizador aplica os modificadores ao alfa final e o controlador usa a máscara composta para limitar pinceladas. No Studio, a seleção pode ser persistida como nó `mask` ligado à camada escolhida e participa do mesmo histórico transacional da árvore profissional.

Clone, correção e remendo usam comandos serializáveis com amostragem, configurações e estado de renderização. Cada comando cria uma camada `generated` acima do alvo, com máscara própria e referência ao resultado raster, sem sobrescrever a camada de origem. O executor local grava os pixels em `layers/generated/<pagina>/<id>.png`; o compositor do Studio consome camadas raster/geradas e `mask_ids` no canvas, enquanto o PSD preserva opacidade, blend e máscara editável. Esse contrato também é a fronteira preparada para o preenchimento generativo da Tarefa 9.

### Task 9: FLUX Generative Fill

**Status:** concluída em 2026-07-13.

**Files:**
- Create: `studio/src/ai/fluxContract.ts`
- Create: `studio/src/ai/fluxStore.ts`
- Create: `studio/src/editor/generative/GenerativeFillPanel.tsx`
- Modify: `studio/src-tauri/src/main.rs`

Implementar seleção → prompt → variantes em camadas geradas, com metadados e invariância fora da máscara.

O preenchimento FLUX foi integrado exclusivamente ao Studio como ferramenta de
edição. O recorte, a máscara e o prompt saem do contexto exato da página e da
cena; trocas de página e cancelamentos invalidam resultados tardios. O contrato
aceita de duas a quatro variantes, guarda modelo, provedor, seed, prompt,
dimensões e camada de origem e materializa cada alternativa como camada
`generated` com máscara editável. A camada raster de origem nunca é substituída.

A ponte Tauri executa um adaptador local configurável e persistente, sem shell,
com JSONL por `stdin`/`stdout`, limites de tamanho e rejeição de resultados HTTP.
O modelo permanece residente entre jobs normais; as gerações são serializadas e
o cancelamento encerra o processo para liberar GPU/RAM. Resultados parciais são
limpos antes do commit da cena, e o estado das variantes é recuperável da cena
após Undo, sem depender do job transitório da interface.
O adaptador Python para `diffusers.FluxFillPipeline` usa somente modelo local ou
cacheado por padrão, não envia imagens e exige opt-in explícito para permitir
download de pesos. A saída é novamente recortada pela máscara no Studio, de
modo que a camada gerada não contribua com pixels fora da seleção.

### Task 10: Paridade e produtividade de capítulo

**Files:**
- Modify: `studio/src/export/psd.ts`
- Create: `studio/src/qa/exportParity.test.ts`
- Create: `studio/src/editor/batch/chapterCommands.ts`

Adicionar copiar/aplicar estilo, busca/substituição, fila de revisão, autosave/recuperação e testes de paridade canvas/PNG/PSD.

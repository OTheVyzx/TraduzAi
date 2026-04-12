# Editor Ergonomics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** melhorar a ergonomia da aba de editor e corrigir o bug de arrasto da janela em áreas interativas.

**Architecture:** a implementação mantém a mesma arquitetura React + Zustand atual, mas reorganiza o editor em torno de uma viewport mais previsível e painéis mais informativos. O bug de drag da janela é resolvido isolando regiões de drag do Tauri e endurecendo superfícies interativas.

**Tech Stack:** React 19, TypeScript, Zustand, Tauri v2, lucide-react, Tailwind CSS

---

### Task 1: Corrigir regiões de drag da janela

**Files:**
- Modify: `src/pages/Editor.tsx`
- Modify: `src/components/editor/PageThumbnails.tsx`
- Modify: `src/components/ui/Layout.tsx`

**Step 1: Write the failing test**

Documentar em smoke manual que cliques no editor arrastam a janela por causa de `data-tauri-drag-region` em áreas interativas.

**Step 2: Run test to verify it fails**

Run: abrir o editor e tentar clicar/arrastar bbox, thumbnails e sliders.
Expected: algumas interações arrastam a janela.

**Step 3: Write minimal implementation**

- remover `data-tauri-drag-region` do container raiz do editor
- remover `data-tauri-drag-region` do navegador de páginas
- manter drag apenas em chrome real do app

**Step 4: Run test to verify it passes**

Run: repetir o smoke manual do editor.
Expected: nenhuma dessas interações arrasta a janela.

### Task 2: Melhorar o estado e a viewport do canvas

**Files:**
- Modify: `src/lib/stores/editorStore.ts`
- Modify: `src/components/editor/EditorCanvas.tsx`
- Modify: `src/pages/Editor.tsx`

**Step 1: Write the failing test**

Definir smoke manual para:
- usar `Ctrl+wheel` para zoom
- usar `Space + drag` para pan
- usar `Ajustar` e `100%`

**Step 2: Run test to verify it fails**

Run: abrir o editor e tentar executar esses fluxos.
Expected: não há modelo claro e faltam ações rápidas.

**Step 3: Write minimal implementation**

- adicionar ações de viewport no store
- adicionar toolbar do canvas
- separar pan, scroll e zoom com gestos previsíveis

**Step 4: Run test to verify it passes**

Run: smoke manual do canvas.
Expected: viewport previsível e fácil de recuperar.

### Task 3: Melhorar navegação entre páginas

**Files:**
- Modify: `src/components/editor/PageThumbnails.tsx`
- Modify: `src/pages/Editor.tsx`
- Modify: `src/lib/stores/editorStore.ts`

**Step 1: Write the failing test**

Definir smoke manual para:
- trocar várias páginas seguidas
- manter visível a thumbnail ativa
- usar navegação com teclado

**Step 2: Run test to verify it fails**

Run: navegar em capítulo com várias páginas.
Expected: contexto visual e foco da página atual são fracos.

**Step 3: Write minimal implementation**

- auto-scroll para thumbnail ativa
- indicadores mais claros de página atual
- atalhos e botões mais visíveis

**Step 4: Run test to verify it passes**

Run: repetir navegação.
Expected: navegação mais rápida e estável.

### Task 4: Reestruturar lista de blocos e propriedades

**Files:**
- Modify: `src/components/editor/LayersPanel.tsx`
- Modify: `src/components/editor/LayerItem.tsx`
- Modify: `src/components/editor/PropertyEditor.tsx`

**Step 1: Write the failing test**

Definir smoke manual para:
- localizar bloco rapidamente
- ver quais blocos estão editados
- editar tradução e estilo sem perder contexto

**Step 2: Run test to verify it fails**

Run: editar vários blocos numa página.
Expected: fluxo lento e pouca sinalização visual.

**Step 3: Write minimal implementation**

- enriquecer item de bloco com índice, preview e estado
- reorganizar propriedades para priorizar tradução e bbox
- manter aplicar/descartar sempre acessível

**Step 4: Run test to verify it passes**

Run: repetir edição de vários blocos.
Expected: menos cliques e melhor legibilidade.

### Task 5: Verificação final

**Files:**
- Modify: `context.md`

**Step 1: Run verification**

Run: `npx tsc --noEmit`
Expected: sem erros de TypeScript

**Step 2: Run smoke manual**

Run:
- abrir editor
- selecionar bloco
- editar texto
- mover bbox
- alternar páginas
- usar zoom/pan
- executar `reinpaint` e `retypeset`

Expected: fluxo estável e sem arrasto indevido da janela

**Step 3: Document**

- registrar a rodada em `context.md`

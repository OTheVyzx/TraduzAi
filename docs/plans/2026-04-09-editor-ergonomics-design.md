# Editor Ergonomics Design

**Objetivo:** aproximar a fluidez da aba de editor da experiência do Koharu sem trocar o visual do MangáTL, corrigindo também o bug de arrasto da janela em áreas interativas.

## Escopo

Esta primeira leva cobre ergonomia, não novas ferramentas avançadas de edição. O foco é:

- corrigir `data-tauri-drag-region` em áreas interativas
- tornar o canvas mais previsível para zoom, pan e seleção
- melhorar a navegação entre páginas
- tornar a lista de blocos e o painel de propriedades mais rápidos de usar

## Problemas atuais

### Drag da janela

O editor usa `data-tauri-drag-region` em áreas grandes demais, incluindo superfícies interativas. Isso faz cliques e drags virarem arrasto da janela em vez de interação com o editor.

### Canvas

O canvas atual mistura scroll e pan sem um modo claro de navegação. Falta um conjunto mínimo de ações de viewport como `fit`, `reset`, `100%` e pan por gesto explícito.

### Navegação e contexto

O navegador de páginas funciona, mas não ajuda o usuário a manter contexto. Falta foco automático na página atual e ações rápidas mais visíveis.

### Lista de blocos e propriedades

A lista de camadas mostra apenas o mínimo e o editor de propriedades é funcional, mas lento para ciclos repetidos de revisão.

## Abordagem recomendada

### 1. Isolar o drag da janela

- remover `data-tauri-drag-region` da raiz do editor e de áreas como thumbnails
- manter drag apenas em chrome real
- garantir que canvas, botões, sliders, inputs, overlays e painéis nunca participem do drag da janela

### 2. Transformar o editor em um workbench estável

- preservar a estrutura em três áreas: páginas, canvas, painel lateral
- adicionar uma toolbar do canvas com ações de viewport
- tornar a navegação por teclado e mouse mais consistente

### 3. Melhorar edição sem mudar o visual

- manter tema, cores e tipografia do MangáTL
- aumentar densidade de informação útil em blocos e propriedades
- expor estado de edição pendente de forma mais clara

## Componentes afetados

- `src/pages/Editor.tsx`
- `src/components/editor/EditorCanvas.tsx`
- `src/components/editor/TextOverlay.tsx`
- `src/components/editor/PageThumbnails.tsx`
- `src/components/editor/LayersPanel.tsx`
- `src/components/editor/LayerItem.tsx`
- `src/components/editor/PropertyEditor.tsx`
- `src/lib/stores/editorStore.ts`

## Comportamento esperado

### Janela

- clicar e arrastar texto, bbox, slider, thumbnails e formulários nunca arrasta a janela

### Canvas

- `Ctrl+wheel` faz zoom
- wheel normal faz scroll do canvas
- `Space + drag` faz pan
- botões `Ajustar`, `100%` e `Centralizar` restauram a viewport rapidamente

### Navegação

- trocar de página mantém o editor previsível
- a thumbnail da página atual fica visível automaticamente

### Blocos e propriedades

- lista mostra índice, tipo, preview, visibilidade e estado editado
- painel de propriedades prioriza tradução, estilo e bbox
- ações de aplicar/descartar ficam visíveis e acessíveis

## Riscos

- alterar pan/zoom pode conflitar com seleção de bbox se os eventos não forem bem separados
- mudanças em estado do editor podem quebrar `retypeset` e `reinpaint` se não forem validadas com smoke real

## Validação

- `npx tsc --noEmit`
- smoke manual do editor:
  - selecionar bloco
  - editar tradução
  - mover e redimensionar bbox
  - navegar entre páginas
  - usar zoom/pan
  - executar `reinpaint` e `retypeset`

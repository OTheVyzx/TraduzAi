# No-Reflow Connected Balloons Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remover a quebra automatica de linhas do renderer e substituir por ajuste de alinhamento, tamanho e posicao, com heuristicas mais fortes para separar texto em baloes conectados.

**Architecture:** O renderer deixa de usar wrapping por largura e passa a medir apenas linhas explicitas. O agrupamento de baloes conectados passa a usar sinais geometricos dos bboxes OCR para decidir quando criar dois blocos filhos e como posiciona-los.

**Tech Stack:** Python 3.12, unittest, PIL, numpy, renderer do pipeline.

---

### Task 1: Cobrir o novo contrato de linhas preservadas

**Files:**
- Modify: `pipeline/tests/test_typesetting_renderer.py`
- Modify: `pipeline/tests/test_typesetting_layout.py`

**Step 1: Write the failing test**

Adicionar testes para:
- nao criar novas linhas sem `\n` explicito;
- inferir alinhamento `left/right/center` pela posicao OCR;
- separar em dois filhos quando a geometria indicar balao conectado diagonal/diferenca de altura.

**Step 2: Run test to verify it fails**

Run: `python -m pytest pipeline/tests/test_typesetting_renderer.py pipeline/tests/test_typesetting_layout.py -q`

**Step 3: Write minimal implementation**

Implementar apenas o necessario no renderer para satisfazer os testes.

**Step 4: Run test to verify it passes**

Run: `python -m pytest pipeline/tests/test_typesetting_renderer.py pipeline/tests/test_typesetting_layout.py -q`

### Task 2: Remover reflow automatico do renderer

**Files:**
- Modify: `pipeline/typesetter/renderer.py`

**Step 1: Write the failing test**

Cobrir `_resolve_text_layout` e `_fits_in_box` com texto sem `\n` que antes quebrava em varias linhas.

**Step 2: Run test to verify it fails**

Run: `python -m pytest pipeline/tests/test_typesetting_layout.py -q`

**Step 3: Write minimal implementation**

- criar helper para preservar apenas linhas explicitas;
- trocar `wrap_text(...)` por esse helper nos pontos de medicao/layout;
- manter ajuste de fonte, altura e posicao.

**Step 4: Run test to verify it passes**

Run: `python -m pytest pipeline/tests/test_typesetting_layout.py -q`

### Task 3: Reforcar heuristicas de balao conectado

**Files:**
- Modify: `pipeline/typesetter/renderer.py`
- Test: `pipeline/tests/test_typesetting_layout.py`

**Step 1: Write the failing test**

Cobrir heuristicas para:
- bloco abaixo + direita;
- distancia minima entre grupos;
- diferenca de altura entre grupos.

**Step 2: Run test to verify it fails**

Run: `python -m pytest pipeline/tests/test_typesetting_layout.py -q`

**Step 3: Write minimal implementation**

- adicionar score/heuristica geometrica ao agrupamento de subregions;
- preferir dois filhos quando a composicao indicar dois lobos distintos;
- nao reintroduzir split semantico quando a geometria nao justificar.

**Step 4: Run test to verify it passes**

Run: `python -m pytest pipeline/tests/test_typesetting_layout.py -q`

### Task 4: Verificacao final

**Files:**
- Modify: `pipeline/typesetter/renderer.py`
- Test: `pipeline/tests/test_typesetting_renderer.py`
- Test: `pipeline/tests/test_typesetting_layout.py`

**Step 1: Run targeted verification**

Run: `python -m pytest pipeline/tests/test_typesetting_renderer.py pipeline/tests/test_typesetting_layout.py -q`

**Step 2: Run renderer smoke verification**

Run: `python -m pytest pipeline/tests/test_typesetting_renderer.py::TypesettingRendererTests -q`

**Step 3: Review for regression risk**

Confirmar que:
- textos simples nao quebram em linhas novas;
- `\n` explicito continua funcionando;
- baloes conectados ainda produzem dois lobos quando esperado.

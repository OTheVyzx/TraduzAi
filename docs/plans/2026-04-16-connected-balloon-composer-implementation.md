# Connected Balloon Composer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** elevar o rendering de baloes duplos conectados para um nivel proximo de scan profissional, com compositor dedicado, score visual e fallback seguro.

**Architecture:** promover grupos conectados a blocos proprios no renderer, gerar candidatos de composicao por lobo e escolher o melhor via dry-run tipografico. O layout continua detectando subregioes, mas passa a expor orientacao e ordem do grupo para o compositor.

**Tech Stack:** Python 3.12, Pillow, NumPy, OpenCV, unittest

---

### Task 1: Cobrir o agrupamento conectado com testes

**Files:**
- Modify: `pipeline/tests/test_typesetting_layout.py`

**Step 1: Write the failing tests**

- adicionar teste que prove que `build_render_blocks()` nao deve transformar grupo conectado `1:1` em dois blocos independentes;
- adicionar teste que prove que o bloco conectado preserva `connected_children` ordenados e as `balloon_subregions`;
- adicionar teste que prove que o compositor prefere composicao com menos linhas orfas e melhor ocupacao.

**Step 2: Run test to verify it fails**

Run: `python -m unittest pipeline.tests.test_typesetting_layout.TypesettingLayoutTests`
Expected: FAIL por inexistencia do novo fluxo conectado.

### Task 2: Cobrir metadados de orientacao no layout

**Files:**
- Modify: `pipeline/tests/test_layout_analysis.py`
- Modify: `pipeline/layout/balloon_layout.py`

**Step 1: Write the failing tests**

- adicionar teste para orientacao `left-right` em baloes conectados horizontais;
- adicionar teste para orientacao `top-bottom` em baloes verticais;
- adicionar teste garantindo que a ordem de leitura das subregioes seja estavel.

**Step 2: Run test to verify it fails**

Run: `python -m unittest pipeline.tests.test_layout_analysis.EnforceMinLobeSizeTests`
Expected: FAIL por falta dos metadados/ordenacao.

### Task 3: Implementar o compositor dedicado

**Files:**
- Modify: `pipeline/typesetter/renderer.py`

**Step 1: Write minimal implementation**

- introduzir helpers para:
- ordenar subregioes por leitura;
- promover grupos `1:1` a bloco conectado unico;
- gerar candidatos de split quando necessario;
- resolver um candidato conectado e medir score visual;
- selecionar o melhor candidato com fallback.

**Step 2: Run focused tests**

Run: `python -m unittest pipeline.tests.test_typesetting_layout.TypesettingLayoutTests`
Expected: PASS

### Task 4: Enriquecer metadados do layout conectado

**Files:**
- Modify: `pipeline/layout/balloon_layout.py`

**Step 1: Write minimal implementation**

- anexar orientacao e ordem estavel ao plano de subregioes conectado;
- manter compatibilidade com os campos antigos para nao quebrar o pipeline.

**Step 2: Run focused tests**

Run: `python -m unittest pipeline.tests.test_layout_analysis`
Expected: PASS

### Task 5: Verificacao final

**Files:**
- Modify: `docs/plans/2026-04-16-connected-balloon-composer-design.md`

**Step 1: Run verification**

Run: `$env:PYTHONPATH='D:\\TraduzAi\\pipeline'; python -m unittest pipeline.tests.test_layout_analysis pipeline.tests.test_typesetting_layout`
Expected: todos os testes relevantes passam.

**Step 2: Inspect outputs**

Run: validar visualmente os casos reais do capitulo problematico no fluxo conectado.
Expected: fonte maior, melhor ocupacao do lobo, menos linhas orfas e composicao mais proxima de scan humana.

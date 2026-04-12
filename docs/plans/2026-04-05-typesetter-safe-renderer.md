# Typesetter Safe Renderer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrar o renderer do typesetter para um caminho seguro no Windows, sem depender de `PIL.ImageFont.getbbox()` nem de `ImageDraw.text()` para as fontes reais do projeto.

**Architecture:** Reaproveitar a mesma estratégia segura usada no `font_detector`: medir e rasterizar texto via `matplotlib.textpath.TextPath`, convertendo o resultado em máscaras/overlays com `numpy` e `OpenCV`. Fazer em duas etapas: primeiro texto sólido + contorno + sombra; depois glow e gradiente.

**Tech Stack:** Python 3.12, matplotlib, NumPy, OpenCV, Pillow, unittest

---

### Task 1: Cobrir a etapa mínima com testes

**Files:**
- Modify: `pipeline/tests/test_typesetting_layout.py`
- Create: `pipeline/tests/test_typesetting_renderer.py`

**Step 1: Write the failing tests**

- Adicionar um teste que valide a medição segura de largura/altura para `CCDaveGibbonsLower W00 Regular.ttf`.
- Adicionar um teste que renderize um bloco simples em balão branco sem cair e produza pixels alterados no `Image`.

**Step 2: Run test to verify it fails**

Run: `venv\Scripts\python.exe -m unittest discover -s tests -p "test_typesetting_renderer.py" -v`
Expected: FAIL por helper seguro ausente.

### Task 2: Implementar etapa 1 do renderer seguro

**Files:**
- Modify: `pipeline/typesetter/renderer.py`

**Step 1: Write minimal implementation**

- Criar helpers seguros de:
- resolver caminho da fonte
- medir caixa do texto por `TextPath`
- rasterizar texto preenchido em máscara
- aplicar sombra e contorno básicos sem `ImageDraw.text()`
- Fazer `wrap_text`, `get_line_height`, `measure_text_width` e `render_text_block` usarem o caminho seguro nos casos das fontes do projeto.

**Step 2: Run focused tests**

Run: `venv\Scripts\python.exe -m unittest discover -s tests -p "test_typesetting_renderer.py" -v`
Expected: PASS

### Task 3: Validar caso real da `012__001`

**Files:**
- Modify: `context.md`

**Step 1: Real-image validation**

- Rodar `run_detect_ocr(...)`, `run_inpaint_pages(...)` e `run_typesetting(...)` na `testes/012__001.jpg`.
- Confirmar que a imagem final é gerada sem `access violation`.

### Task 4: Etapa 2

**Files:**
- Modify: `pipeline/typesetter/renderer.py`
- Modify: `pipeline/tests/test_typesetting_renderer.py`

**Step 1: Expand**

- Migrar glow e gradiente para o mesmo pipeline seguro.
- Revalidar a imagem real.

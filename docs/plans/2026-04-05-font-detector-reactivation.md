# Font Detector Reactivation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reativar o font detector no fluxo padrão do `vision_stack` sem reintroduzir o crash em Windows ao renderizar amostras das fontes.

**Architecture:** Substituir apenas a etapa de render das amostras do `FontDetector` por uma rasterização baseada em `matplotlib.textpath.TextPath`, preservando o modelo YuzuMarker e a lógica de similaridade existente. Depois, religar a detecção no caminho padrão de `run_detect_ocr(...)` e cobrir a mudança com testes focados.

**Tech Stack:** Python 3.12, matplotlib, NumPy, OpenCV, unittest

---

### Task 1: Cobrir o comportamento esperado com testes

**Files:**
- Create: `pipeline/tests/test_font_detector.py`
- Modify: `pipeline/tests/test_vision_stack_runtime.py`

**Step 1: Write the failing test**

- Adicionar um teste para uma nova função helper de rasterização que deve gerar uma imagem RGB válida e com pixels escuros para uma fonte real do projeto.
- Adicionar um teste no `vision_stack` garantindo que `run_detect_ocr(...)` volta a chamar `build_page_result(...)` com `enable_font_detection=True`.

**Step 2: Run test to verify it fails**

Run: `venv\Scripts\python.exe -m unittest discover -s tests -p "test_font_detector.py" -v`
Expected: FAIL por função/helper ausente.

Run: `venv\Scripts\python.exe -m unittest discover -s tests -p "test_vision_stack_runtime.py" -k font_detector -v`
Expected: FAIL porque `_run_detect_ocr_on_image(...)` ainda passa `enable_font_detection=False`.

### Task 2: Implementar a troca do renderer e religar o fluxo

**Files:**
- Modify: `pipeline/typesetter/font_detector.py`
- Modify: `pipeline/vision_stack/runtime.py`

**Step 1: Write minimal implementation**

- Criar helper de rasterização via `TextPath`/`FontProperties`.
- Fazer `_render_font_sample(...)` usar o helper novo, sem depender de `PIL.ImageDraw.text(...)`.
- Reativar o font detector no fluxo padrão de `run_detect_ocr(...)`.

**Step 2: Run targeted tests**

Run: `venv\Scripts\python.exe -m unittest discover -s tests -p "test_font_detector.py" -v`
Expected: PASS

Run: `venv\Scripts\python.exe -m unittest discover -s tests -p "test_vision_stack_runtime.py" -k font_detector -v`
Expected: PASS

### Task 3: Validar regressão e caso real

**Files:**
- Modify: `context.md`

**Step 1: Run broader verification**

Run: `venv\Scripts\python.exe -m unittest discover -s tests -p "test_vision_stack_runtime.py" -v`
Expected: PASS

**Step 2: Real-image validation**

Run um script real em `testes/012__001.jpg` para confirmar:
- o pipeline não cai
- as caixas detectadas recebem `estilo["fonte"]`
- os valores detectados fazem sentido visualmente

**Step 3: Document**

- Registrar a reativação e a validação em `context.md`.

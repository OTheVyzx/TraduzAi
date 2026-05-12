# Typesetting Estilo e Contraste Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fazer o processamento automatico concluir capitulos com texto legivel e padronizado: ComicNeue-Bold.ttf, sem contorno, sem sombra, sem brilho, e cor escolhida por contraste contra o fundo.

**Architecture:** A normalizacao pertence ao pipeline automatico, antes do texto virar `text_layers`/`project.json`. O renderer deve preservar estilos manuais quando renderiza preview/export a partir do editor; para isso, camadas automaticas recebem metadata explicita de origem de estilo e camadas editadas continuam livres.

**Tech Stack:** Python pipeline/typesetter/OCR, React/TypeScript editor hydration, Rust project schema, pytest, Vitest, cargo.

---

## Senior Review Notes

O plano anterior estava na direcao certa, mas tinha tres riscos:

- **Risco 1: normalizar dentro do renderer sem contrato.** `pipeline/main.py::_run_render_preview_page()` renderiza a pagina materializada do editor via `_page_text_layers_for_renderer()`. Se `_render_single_text_block()` limpar fonte/cor/efeitos sem distinguir origem, o preview/export do editor vai apagar alteracoes manuais.
- **Risco 2: corrigir tarde demais.** O bug nasce antes do renderer: `pipeline/vision_stack/runtime.py` ainda define `Newrotic.ttf` e `#FFFFFF` no ramo nao branco. O automatico precisa sair correto no `project.json`, nao apenas renderizar correto.
- **Risco 3: sensor de fundo usando area errada.** Usar somente o bbox do texto pode amostrar pixels das letras originais. O sensor deve preferir `balloon_bbox`/`layout_bbox`/`safe_text_box` quando disponivel e filtrar pixels de tinta/borda.

Contrato revisado:

- Pipeline automatico novo: sempre salva `style_origin: "auto"` e estilo canonico.
- Editor: ao modificar qualquer campo de estilo, a camada passa a `style_origin: "editor"` ou equivalente.
- Renderer: aplica policy automatica somente quando `style_origin != "editor"` **e** o render nao for preview materializado do editor com estilos manuais.
- Frontend/Rust: defaults continuam configuraveis; nada fica lockado.

---

## Task 1: Criar Politica Python de Estilo Automatico

**Files:**
- Create: `pipeline/typesetter/style_policy.py`
- Test: `pipeline/tests/test_typesetting_style_policy.py`

**Step 1: Write failing tests**

Criar testes:

```python
from pipeline.typesetter.style_policy import (
    CANONICAL_AUTO_FONT,
    normalize_auto_typesetting_style,
)


def test_auto_style_removes_effects_font_and_bad_white_on_light_background():
    style = normalize_auto_typesetting_style(
        {
            "fonte": "Newrotic.ttf",
            "cor": "#FFFFFF",
            "contorno": "#000000",
            "contorno_px": 3,
            "glow": True,
            "glow_cor": "#ffffff",
            "glow_px": 8,
            "sombra": True,
            "sombra_cor": "#111111",
            "sombra_offset": [3, 3],
            "tamanho": 34,
            "alinhamento": "left",
        },
        background_rgb=(245, 245, 245),
    )

    assert style["fonte"] == CANONICAL_AUTO_FONT
    assert style["cor"] == "#000000"
    assert style["contorno"] == ""
    assert style["contorno_px"] == 0
    assert style["glow"] is False
    assert style["glow_cor"] == ""
    assert style["glow_px"] == 0
    assert style["sombra"] is False
    assert style["sombra_cor"] == ""
    assert style["sombra_offset"] == [0, 0]
    assert style["tamanho"] == 34
    assert style["alinhamento"] == "left"


def test_auto_style_uses_white_only_when_dark_background_needs_it():
    style = normalize_auto_typesetting_style({"cor": "#000000"}, background_rgb=(18, 18, 24))

    assert style["fonte"] == "ComicNeue-Bold.ttf"
    assert style["cor"] == "#FFFFFF"
    assert style["contorno_px"] == 0
    assert style["glow"] is False
    assert style["sombra"] is False
```

**Step 2: Run test to verify it fails**

```bash
cd pipeline
python -m pytest tests/test_typesetting_style_policy.py -q
```

Expected: FAIL porque `style_policy.py` ainda nao existe.

**Step 3: Implement minimal policy**

Implementar em `pipeline/typesetter/style_policy.py`:

```python
from __future__ import annotations

CANONICAL_AUTO_FONT = "ComicNeue-Bold.ttf"


def relative_luminance(rgb: tuple[int, int, int]) -> float:
    def channel(value: int) -> float:
        value = max(0, min(255, int(value))) / 255.0
        return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(v) for v in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def auto_text_color_for_background(background_rgb: tuple[int, int, int]) -> str:
    # Conservador: manga/manhwa tem muito balao claro; em duvida, preto.
    return "#000000" if relative_luminance(background_rgb) >= 0.42 else "#FFFFFF"


def normalize_auto_typesetting_style(style: dict | None, background_rgb: tuple[int, int, int]) -> dict:
    normalized = dict(style or {})
    normalized["fonte"] = CANONICAL_AUTO_FONT
    normalized["cor"] = auto_text_color_for_background(background_rgb)
    normalized["cor_gradiente"] = []
    normalized["contorno"] = ""
    normalized["contorno_px"] = 0
    normalized["glow"] = False
    normalized["glow_cor"] = ""
    normalized["glow_px"] = 0
    normalized["sombra"] = False
    normalized["sombra_cor"] = ""
    normalized["sombra_offset"] = [0, 0]
    normalized["bold"] = True
    normalized.setdefault("italico", False)
    normalized.setdefault("rotacao", 0)
    normalized.setdefault("alinhamento", "center")
    normalized.setdefault("force_upper", False)
    return normalized
```

**Step 4: Run test to verify it passes**

```bash
cd pipeline
python -m pytest tests/test_typesetting_style_policy.py -q
```

Expected: PASS.

---

## Task 2: Sensor de Fundo Robusto

**Files:**
- Modify: `pipeline/typesetter/style_policy.py`
- Test: `pipeline/tests/test_typesetting_style_policy.py`

**Step 1: Write failing tests**

Adicionar:

```python
import numpy as np
from pipeline.typesetter.style_policy import sample_text_background_rgb


def test_background_sensor_prefers_inner_balloon_region():
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    image[20:100, 20:100] = [245, 245, 245]
    image[20:100, 20:23] = [0, 0, 0]
    image[55:60, 35:85] = [0, 0, 0]

    assert sample_text_background_rgb(image, [20, 20, 100, 100]) == (245, 245, 245)


def test_background_sensor_handles_dark_panel():
    image = np.full((100, 100, 3), [20, 24, 30], dtype=np.uint8)

    rgb = sample_text_background_rgb(image, [10, 10, 90, 90])

    assert rgb[0] < 40
    assert rgb[1] < 40
    assert rgb[2] < 50
```

**Step 2: Implement sensor**

Implementar `sample_text_background_rgb(image_rgb, bbox)`:

- Aceitar `np.ndarray` RGB.
- Clamp de bbox aos limites.
- Usar margem interna de `max(2px, 8% do menor lado)`.
- Calcular luminancia de cada pixel.
- Se existir amostra suficiente, descartar os 10% mais escuros e 5% mais claros antes da mediana, para reduzir impacto de letra/borda/brilho.
- Retornar mediana RGB como `tuple[int, int, int]`.
- Fallback: `(255, 255, 255)`.

**Step 3: Run tests**

```bash
cd pipeline
python -m pytest tests/test_typesetting_style_policy.py -q
```

Expected: PASS.

---

## Task 3: Marcar Origem de Estilo em Camadas

**Files:**
- Modify: `pipeline/main.py`
- Modify: `src/lib/stores/appStore.ts`
- Modify: `src/lib/tauri.ts`
- Modify: `src/lib/stores/editorStore.ts`
- Test: `pipeline/tests/test_main_emit.py`
- Test: `src/lib/__tests__/tauriHydration.test.ts`
- Test: `src/lib/stores/__tests__/editorStoreHistory.test.ts`

**Step 1: Write tests**

Python:

```python
def test_build_text_layer_marks_auto_style_origin():
    layer = build_text_layer(...)
    assert layer["style_origin"] == "auto"
    assert layer["estilo"]["fonte"] == "ComicNeue-Bold.ttf"
```

TypeScript:

```ts
expect(hydrated.text_layers[0].style_origin).toBe("auto");
```

Editor style edit:

```ts
useEditorStore.getState().updatePendingEstilo(layerId, { fonte: "Newrotic.ttf" });
expect(useEditorStore.getState().pendingEdits[layerId].style_origin).toBe("editor");
```

**Step 2: Update schema/types**

- Add optional `style_origin?: "auto" | "editor" | "legacy"` to `TextEntry`.
- In `hydrateTextLayer()`, preserve `layer.style_origin ?? "legacy"`.
- In `build_text_layer()`, set `style_origin: "auto"`.
- In `_normalize_text_layer_for_renderer()`, preserve `raw_layer.get("style_origin", "legacy")`.
- In `editorStore.updatePendingEstilo()`, include `style_origin: "editor"` in pending edit/materialized layer.

**Step 3: Run tests**

```bash
npm run test -- src/lib/__tests__/tauriHydration.test.ts src/lib/stores/__tests__/editorStoreHistory.test.ts
cd pipeline
python -m pytest tests/test_main_emit.py -q
```

Expected: PASS.

---

## Task 4: Normalizar Estilo na Origem do Pipeline Automatico

**Files:**
- Modify: `pipeline/vision_stack/runtime.py`
- Test: `pipeline/tests/test_vision_stack_runtime.py`

**Step 1: Write failing test**

O teste deve cobrir o ramo que hoje gera `Newrotic.ttf`/`#FFFFFF`:

```python
def test_runtime_auto_style_never_outputs_newrotic_or_effects_on_white_balloon(...):
    result = ...
    style = result["estilo"]
    assert style["fonte"] == "ComicNeue-Bold.ttf"
    assert style["cor"] == "#000000"
    assert style["contorno"] == ""
    assert style["contorno_px"] == 0
    assert style["glow"] is False
    assert style["sombra"] is False
```

**Step 2: Replace hardcoded style branch**

No trecho de `pipeline/vision_stack/runtime.py` que hoje faz:

```python
if use_base_white_font:
    ...
else:
    estilo["fonte"] = "Newrotic.ttf"
    estilo["cor"] = "#FFFFFF"
```

Trocar por:

```python
from typesetter.style_policy import normalize_auto_typesetting_style, sample_text_background_rgb

style_bbox = (
    ocr_text.get("balloon_bbox")
    or ocr_text.get("layout_bbox")
    or bbox
)
background_rgb = sample_text_background_rgb(image_rgb, style_bbox)
estilo = normalize_auto_typesetting_style(estilo, background_rgb=background_rgb)
estilo["force_upper"] = True
```

**Important:** remover `use_base_white_font` apenas se ele nao for usado para outra decisao. Se ainda for util para metadata/layout, manter a variavel, mas nao deixar ela escolher fonte/cor.

**Step 3: Run focused tests**

```bash
cd pipeline
python -m pytest tests/test_vision_stack_runtime.py -q
```

Expected: PASS.

---

## Task 5: Aplicar Policy em `build_text_layer()` Como Defesa

**Files:**
- Modify: `pipeline/main.py`
- Test: `pipeline/tests/test_main_emit.py`

**Step 1: Write failing test**

Criar teste direto em `build_text_layer()` com `ocr_text["estilo"]` poluido:

```python
def test_build_text_layer_sanitizes_auto_style_before_project_json():
    layer = build_text_layer(
        page_number=1,
        layer_index=0,
        ocr_text={
            "bbox": [0, 0, 100, 100],
            "layout_bbox": [0, 0, 100, 100],
            "text": "HELLO",
            "estilo": {
                "fonte": "Newrotic.ttf",
                "cor": "#FFFFFF",
                "contorno": "#000000",
                "contorno_px": 2,
                "glow": True,
                "sombra": True,
            },
            "background_rgb": [250, 250, 250],
        },
        translated="OLA",
        corpus_visual_benchmark={},
        corpus_textual_benchmark={},
    )

    assert layer["style_origin"] == "auto"
    assert layer["estilo"]["fonte"] == "ComicNeue-Bold.ttf"
    assert layer["estilo"]["cor"] == "#000000"
    assert layer["estilo"]["contorno_px"] == 0
    assert layer["estilo"]["glow"] is False
    assert layer["estilo"]["sombra"] is False
```

**Step 2: Implement defense**

Em `build_text_layer()`:

- Ler `ocr_text.get("background_rgb")` se existir.
- Fallback conservador para `(255, 255, 255)`.
- Aplicar `normalize_auto_typesetting_style(_merge_style(...), background_rgb)`.
- Salvar `style_origin: "auto"`.

**Why here:** esse e o ponto onde `project.json` e `text_layers` nascem. Corrigir aqui garante editor, preview e export herdando default limpo.

**Step 3: Run tests**

```bash
cd pipeline
python -m pytest tests/test_main_emit.py -q
```

Expected: PASS.

---

## Task 6: Renderer Preserva Manual e Só Sanitiza Auto

**Files:**
- Modify: `pipeline/typesetter/renderer.py`
- Test: `pipeline/tests/test_typesetting_renderer.py`

**Step 1: Write two failing tests**

Auto:

```python
def test_renderer_sanitizes_auto_style_as_last_defense():
    text_data = {
        "style_origin": "auto",
        "translated": "OLA",
        "estilo": {"fonte": "Newrotic.ttf", "cor": "#FFFFFF", "contorno_px": 2, "glow": True, "sombra": True},
        ...
    }
    _render_single_text_block(img, text_data, plan)
    assert text_data["estilo"]["fonte"] == "ComicNeue-Bold.ttf"
    assert text_data["estilo"]["cor"] == "#000000"
    assert text_data["estilo"]["contorno_px"] == 0
```

Editor:

```python
def test_renderer_preserves_editor_style():
    text_data = {
        "style_origin": "editor",
        "translated": "OLA",
        "estilo": {"fonte": "Newrotic.ttf", "cor": "#FFFFFF", "contorno": "#000000", "contorno_px": 2, "glow": True, "glow_px": 3},
        ...
    }
    _render_single_text_block(img, text_data, plan)
    assert text_data["estilo"]["fonte"] == "Newrotic.ttf"
    assert text_data["estilo"]["contorno_px"] == 2
    assert text_data["estilo"]["glow"] is True
```

**Step 2: Implement guarded normalization**

Add helper in renderer:

```python
def _should_apply_auto_style_policy(text_data: dict) -> bool:
    return text_data.get("style_origin") in (None, "auto", "legacy_auto")
```

Apply before `plan_text_block()` consumes style, ideally in the caller that builds the plan, not after plan creation. If the only safe insertion point is `_render_single_text_block()`, recompute or update plan fields that depend on style (`font_name`, `text_color`, `outline_color`, `outline_px`, `glow`, `sombra`) after normalization.

**Important:** Do not normalize when `style_origin == "editor"`.

**Step 3: Run renderer tests**

```bash
cd pipeline
python -m pytest tests/test_typesetting_renderer.py -q
```

Expected: PASS.

---

## Task 7: Frontend Editor Continua Configuravel

**Files:**
- Modify: `src/lib/editorTextStylePolicy.ts`
- Modify: `src/lib/tauri.ts`
- Modify: `src/lib/stores/editorStore.ts`
- Test: `src/lib/__tests__/editorTextStylePolicy.test.ts`
- Test: `src/lib/__tests__/tauriHydration.test.ts`
- Test: `src/lib/stores/__tests__/editorStoreHistory.test.ts`

**Step 1: Tests**

Confirmar que explicit/manual stays manual:

```ts
expect(canonicalizeTextStyle({ fonte: "Arial.ttf" }, { mode: "preserve-explicit" }).fonte).toBe("Arial.ttf");
expect(canonicalizeTextStyle({ contorno: "#000000", contorno_px: 2 }, { mode: "preserve-explicit" }).contorno_px).toBe(2);
expect(canonicalizeTextStyle({ glow: true, glow_px: 4 }, { mode: "preserve-explicit" }).glow).toBe(true);
```

Confirmar que `updatePendingEstilo()` marca:

```ts
expect(pendingEdits[layerId].style_origin).toBe("editor");
```

**Step 2: Implementation**

- `hydrateTextLayer()` preserva `style_origin`.
- `mergePendingEdit()` deve preservar `style_origin` quando pendente.
- `updatePendingEstilo()` seta `style_origin: "editor"`.
- `commitEdits()` inclui `style_origin` no patch/full save quando existir.

**Step 3: Run tests**

```bash
npm run test -- src/lib/__tests__/editorTextStylePolicy.test.ts src/lib/__tests__/tauriHydration.test.ts src/lib/stores/__tests__/editorStoreHistory.test.ts
npm run check
```

Expected: PASS.

---

## Task 8: Rust Schema Mantem Defaults, Sem Lock

**Files:**
- Modify if needed: `src-tauri/src/commands/project_schema.rs`

**Checks:**

- `default_text_style()` continua ComicNeue-Bold.ttf, preto, sem efeitos.
- `patch_text_layer()` aceita salvar estilo manual com contorno/glow/sombra.
- Se `style_origin` for adicionado ao schema, ele deve ser opcional e preservado.

**Commands:**

```bash
cd src-tauri
cargo test default_text_style_has_no_visual_effects
cargo check
```

Expected: PASS.

---

## Task 9: Acceptance / QA

**Commands:**

```bash
npm run check
npm run test -- src/lib/__tests__/editorTextStylePolicy.test.ts src/lib/__tests__/tauriHydration.test.ts src/lib/stores/__tests__/editorStoreHistory.test.ts
cd pipeline
python -m pytest tests/test_typesetting_style_policy.py tests/test_main_emit.py tests/test_typesetting_renderer.py -q
cd ..
cd src-tauri
cargo check
```

Optional if environment allows:

```bash
npm run test:e2e -- e2e/editor-rebuild.spec.ts --grep "editor|typesetting" --workers=1
```

**Acceptance criteria:**

- Capitulo automatico novo nao tem `Newrotic.ttf` em `text_layers[*].estilo.fonte`.
- Capitulo automatico novo nao tem `contorno_px > 0`, `glow == true` ou `sombra == true`.
- Texto sobre balao claro fica preto.
- Texto sobre fundo escuro pode ficar branco, mas sem contorno/sombra/brilho.
- Ao abrir editor, o usuario consegue trocar fonte/cor/contorno/sombra/brilho.
- Depois de salvar e renderizar no editor, estilos manuais nao sao revertidos pela policy automatica.
- Preview/export final usa estilos manuais quando `style_origin == "editor"`.

---

## Non-Goals

- Nao remover fontes alternativas do app/editor.
- Nao bloquear controles de estilo.
- Nao fazer migracao pesada de projetos antigos.
- Nao usar CSS como correcao de contraste.
- Nao aplicar policy automatica cegamente em `render_preview_page`.

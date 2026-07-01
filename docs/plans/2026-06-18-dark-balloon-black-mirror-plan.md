# Dark Balloon Black Mirror Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fazer baloes pretos, retangulos pretos e texto claro sem balao seguirem o mesmo nivel de confiabilidade dos baloes brancos, usando OCR/detect negativo apenas como evidencia auxiliar.

**Architecture:** Criar um contrato explicito `dark_balloon` / `dark_rect` espelhando a geometria dos caminhos brancos, mas com polaridade invertida: fundo escuro, texto claro, fill escuro amostrado da imagem normal e render claro/glow. O passe negativo roda como `negative_evidence_pass`, fica em payload shadow, e so promove candidatos depois de gates de fundo escuro, deduplicacao e validacao de mascara canonica.

**Tech Stack:** Python 3.12, OpenCV, PIL, PaddleOCR, pipeline strip/debug atual do TraduzAI.

---

## Baseline obrigatoria

Antes de implementar, ler:

- `.codex-tmp/dark_balloon_black_mirror_baseline_20260618.md`
- `N:\TraduzAI\.codex-tmp\negative_roundtrip_test_20260618\page002_negative_pipeline_inverted_back_positive_crops.jpg`
- `N:\TraduzAI\.codex-tmp\negative_page_test_20260618\negative_page002_crops_sheet.jpg`

Sucesso nao e "logs passaram". Sucesso exige olhar visualmente os bands e a pagina translated.

## Arquivos donos

- Detect:
  - `pipeline/strip/detect_balloons.py`
  - `pipeline/vision_stack/runtime.py`
  - `pipeline/ocr/detector.py`
- Fusion/strip orchestration:
  - `pipeline/strip/process_bands.py`
  - `pipeline/strip/run.py`
- Layout:
  - `pipeline/layout/balloon_layout.py`
- Mask:
  - `pipeline/inpainter/mask_builder.py`
- Inpaint/fill:
  - `pipeline/inpainter/__init__.py`
  - `pipeline/inpainter/lama_onnx.py`
- Typesetting:
  - `pipeline/typesetter/style_policy.py`
  - `pipeline/typesetter/renderer.py`
- Debug:
  - `pipeline/debug_tools/recorder.py`
  - `pipeline/debug_tools/masks.py`

## Contratos que nao podem quebrar

Preservar e manter em coordenada coerente:

- `bbox`
- `source_bbox`
- `text_pixel_bbox`
- `line_polygons`
- `balloon_bbox`
- `layout_bbox`
- `layout_profile`
- `block_profile`
- `background_rgb`
- `route_action`
- `content_class`
- `mask_evidence`
- `qa_flags`
- `qa_metrics`
- `bubble_mask_source`
- `bubble_mask_bbox`
- `bubble_inner_bbox`
- `bubble_mask_shape`
- `bubble_mask_ellipse`
- `_vision_blocks`

Nao depender somente de `balloon_type`, porque `pipeline/project_writer.py` neutraliza campos de decisao removida para texto normal.

---

## Task 1: Tests for dark geometry parity

**Files:**
- Modify: `pipeline/tests/test_strip_detect.py`
- Modify or create focused test in: `pipeline/tests/test_mask_builder.py`

**Step 1: Add failing detect tests**

Criar fixtures sinteticas para:

- Balao oval preto com texto branco.
- Retangulo preto com texto branco.
- Texto branco sem balao em fundo preto.

Os testes devem verificar que a evidencia escura nao e classificada como balao branco e produz perfil dark:

```python
assert candidate["dark_light_text_evidence"]["useful"] is True
assert candidate["background_polarity"] == "dark"
```

**Step 2: Add failing mask tests**

Validar que `build_inpaint_mask()` retorna:

- `bubble_mask_source == "image_dark_bubble_mask"` para balao oval preto.
- `bubble_mask_source in {"image_dark_panel_mask", "derived_card_panel_mask"}` para retangulo preto.
- `mask_evidence.fast_fill_allowed is True` apenas quando a mascara nao invade borda/arte.

**Step 3: Run focused tests**

Run:

```powershell
cd N:\TraduzAI\pipeline
..\pipeline\venv\Scripts\python.exe -m pytest tests\test_strip_detect.py tests\test_mask_builder.py -q -k "dark_bubble or dark_panel or negative"
```

Expected: novos testes falham antes da implementacao.

---

## Task 2: Add negative evidence pass as shadow payload

**Files:**
- Modify: `pipeline/vision_stack/runtime.py`
- Modify: `pipeline/ocr/detector.py`
- Modify: `pipeline/strip/process_bands.py`
- Test: `pipeline/tests/test_vision_stack_runtime.py`
- Test: `pipeline/tests/test_strip_process_bands.py`

**Step 1: Add a helper contract**

Criar payload shadow:

```python
{
    "texts": [...],
    "blocks": [...],
    "source": "negative_detect_ocr",
    "image_transform": "inverted_luma",
    "eligible_for_promotion": False,
}
```

Esse payload deve ficar em `_negative_evidence`, nunca direto em `texts` ou `_vision_blocks`.

**Step 2: Run detect/OCR on negative image only as evidence**

No caminho de band/pagina, criar imagem auxiliar invertida:

```python
negative_rgb = 255 - image_rgb
```

Rodar detect/OCR em modo shadow. Nao passar `negative_rgb` para inpaint, fill, render ou copyback.

**Step 3: Add mutation guard tests**

Tests:

- `test_negative_pass_does_not_mutate_page_texts_until_promoted`
- `test_negative_pass_does_not_mutate_vision_blocks_until_promoted`

**Step 4: Run focused tests**

```powershell
cd N:\TraduzAI\pipeline
..\pipeline\venv\Scripts\python.exe -m pytest tests\test_vision_stack_runtime.py tests\test_strip_process_bands.py -q -k "negative_pass or negative_evidence"
```

Expected: pass depois da implementacao.

---

## Task 3: Fuse negative evidence into canonical dark candidates

**Files:**
- Modify: `pipeline/strip/process_bands.py`
- Modify: `pipeline/layout/balloon_layout.py`
- Test: `pipeline/tests/test_strip_process_bands.py`
- Test: `pipeline/tests/test_typesetting_renderer.py`

**Step 1: Add fusion helper**

Adicionar funcao com comportamento equivalente:

```python
def fuse_negative_dark_bubble_candidates(normal_page, negative_evidence, image_rgb):
    ...
```

Promover candidato so se:

- bbox intersecta ou complementa balao/painel escuro suspeito;
- amostragem na imagem normal confirma fundo escuro;
- OCR tem texto util e confianca minima;
- nao e SFX/arte/scanlator/noise;
- nao duplica candidato normal melhor;
- mascara final sera construida por `build_inpaint_mask()`.

**Step 2: Dedupe policy**

Duplicado se:

- IoU >= 0.65, ou
- overlap/min-area >= 0.55, ou
- centro dentro de 8-16 px e texto similar >= 0.85.

Se duplicado, anexar em `qa_metrics.negative_evidence` em vez de criar nova camada.

**Step 3: Promotion payload**

Candidato promovido deve conter:

```python
ocr_source = "negative_detect_ocr_promoted"
qa_flags += ["negative_pass_promoted", "dark_bubble_negative_evidence"]
layout_profile = "dark_bubble"  # ou "dark_rect"
block_profile = "dark_bubble"   # ou "dark_panel"
```

**Step 4: Tests**

Adicionar:

- `test_negative_dark_bubble_candidate_promotes_when_normal_ocr_misses_white_text`
- `test_negative_candidate_is_attached_as_evidence_when_duplicate_of_normal`
- `test_negative_candidate_rejected_for_light_balloon`
- `test_negative_candidate_rejected_for_sfx_or_suppressed_route`

---

## Task 4: Mirror white balloon geometry for dark balloons

**Files:**
- Modify: `pipeline/strip/detect_balloons.py`
- Modify: `pipeline/inpainter/mask_builder.py`
- Modify: `pipeline/layout/balloon_layout.py`
- Test: `pipeline/tests/test_strip_detect.py`
- Test: `pipeline/tests/test_mask_builder.py`

**Step 1: Implement dark polarity analogs**

Espelhar a logica dos brancos:

- branco: fundo claro + texto escuro;
- preto: fundo escuro + texto claro.

Criar ou consolidar helpers para:

- `dark_balloon` oval;
- `dark_rect`;
- texto claro sem balao em fundo escuro.

**Step 2: Preserve exact shape**

Para balao preto oval:

- usar forma real/ellipse;
- manter cauda quando detectada;
- nao cair em retangulo se o balao e oval.

Para retangulo preto/texto sem balao:

- usar retangulo com padding consistente;
- nao invadir divisoria do painel;
- nao pegar glow como fundo.

**Step 3: Tests**

Validar pixels:

- mascara cobre o interior do balao preto;
- mascara nao cobre borda/glow;
- texto claro gera glyph mask correta;
- texto sem balao vira retangulo dark controlado.

---

## Task 5: Dark fill/inpaint must sample normal image

**Files:**
- Modify: `pipeline/inpainter/__init__.py`
- Modify: `pipeline/inpainter/lama_onnx.py` only if needed
- Test: `pipeline/tests/test_vision_stack_inpainter.py`

**Step 1: Enforce normal-image fill source**

Para `dark_balloon`, samplear o interior real na imagem normal:

- excluir glyph mask dilatada;
- excluir glow/contorno;
- usar pixels de baixa luminancia;
- limitar fill a preto/cor interna real.

**Step 2: Never use negative image color**

Adicionar guard test:

```python
assert fill_color_luma < 60
assert not fill_color_is_glow_blue
assert not fill_color_is_inverted_white
```

**Step 3: Tests**

Adicionar:

- `test_dark_balloon_fill_samples_inner_black_from_normal_image`
- `test_dark_balloon_fill_does_not_use_negative_white`
- `test_dark_balloon_fill_does_not_sample_blue_glow`

---

## Task 6: Dark typesetting parity

**Files:**
- Modify: `pipeline/typesetter/style_policy.py`
- Modify: `pipeline/typesetter/renderer.py`
- Test: `pipeline/tests/test_typesetting_renderer.py`
- Test: `pipeline/tests/test_typesetting_style_policy.py`

**Step 1: Enforce dark style from dark profile**

Para `layout_profile == "dark_bubble"` ou `dark_rect`:

- texto branco/claro;
- glow claro/azulado quando evidenciado;
- fonte e peso herdados de estilo semelhante no capitulo;
- safe box dentro da ellipse/lobo.

**Step 2: Connected lobes**

Cada lobo conectado deve ter:

- `connected_lobe_bbox` em coordenada correta;
- `safe_text_box` proprio;
- texto proprio, nao payload combinado;
- render dentro do lobo correto.

**Step 3: Tests**

Adicionar:

- `test_dark_connected_lobes_keep_distinct_render_boxes`
- `test_dark_connected_lobe_does_not_mix_page_and_band_coordinates`
- `test_dark_bubble_style_uses_light_text_and_preserves_glow`

---

## Task 7: Debug artifacts and visual validation

**Files:**
- Modify: `pipeline/debug_tools/recorder.py`
- Modify: `pipeline/debug_tools/masks.py`
- Modify: `pipeline/strip/process_bands.py`

**Step 1: Write debug artifacts**

Criar:

- `debug/e2e/03_ocr/{band_id}/negative_raw_blocks.jsonl`
- `debug/e2e/03_ocr/{band_id}/negative_fusion_decision.json`
- `debug/e2e/03_ocr/{band_id}/negative_overlay.png`
- `debug/e2e/06_mask_segmentation/{band_id}/negative_promoted_mask_inputs.json`

**Step 2: Final validation commands**

Rodar capitulo 2 completo:

```powershell
cd N:\TraduzAI\pipeline
..\pipeline\venv\Scripts\python.exe main.py ..\.codex-tmp\god_of_death_ch2_dark_balloon_black_mirror_config.json
```

Depois gerar crops visuais dos bands:

- `page_002_band_003`
- `page_002_band_004`
- `page_002_band_005`
- `page_002_band_007`
- `page_002_band_011`
- `page_002_band_021`
- `page_002_band_022`
- `page_002_band_023`

**Step 3: Regression validation**

Rodar capitulo 1 ou pelo menos os bands brancos historicos:

- `page_002_band_002`
- `page_002_band_014`
- `page_003_band_035`

## Definition of Done

- Balões pretos em fundo preto usam mascara no formato real.
- Balões pretos conectados renderizam cada texto no proprio lobo.
- `I am called 'System?'` deixa de sumir quando OCR negativo encontra evidencia.
- `1,000 points..` e outros textos claros em balão preto entram sem retangulo branco.
- Inpaint/fill de dark balloon usa imagem normal, nao negativo.
- Nenhum texto promovido de negativo renderiza sem passar por fusion + mask canonical.
- Capitulo 1 nao regride em baloes brancos, retangulos brancos e texto sem balao.
- Resultado final passa por verificacao visual dos crops e da pagina translated.


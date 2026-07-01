# God of Death Visual Mask OCR Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Corrigir os erros visuais vistos em `god_of_death_ch1_expanded_canvas_full_20260605`: fragmentos OCR renderizados, balões com máscara falsa/parcial, inpaint com máscara encolhida, texto CJK/gibberish indevido e cards/glow sem máscara/estilo correto.

**Architecture:** A correção deve preservar o pipeline atual, mas reforçar contratos entre OCR, detecção de balão, máscara de ação, inpaint e renderer. A ordem é: filtrar/mesclar OCR antes de traduzir, derivar máscara de balão somente ancorada em texto válido, impedir máscaras que encolhem ou pegam arte, e usar anchors/estilo adequados no renderer.

**Tech Stack:** Python 3.12, OpenCV, Pillow, pytest, pipeline/debug artifacts em `DEBUGM/runs`, renderer FT2Font/matplotlib.

---

## Baseline

Run alvo para comparação:

`N:\TraduzAI\DEBUGM\runs\god_of_death_ch1_expanded_canvas_full_20260605`

Casos obrigatórios:

- `page_002_band_005`: `THE SA` fragmentado em cima de `PLEASE, FOR THE CHILD'S SAKE`.
- `page_002_band_008`: máscara de balão parcial no oval grande.
- `page_003_band_035`: máscara pegando prédio/janelas e texto fora do balão.
- `page_004_band_055`: máscara ampla demais e texto sem `render_bbox`.
- `page_004_band_052`: letreiro coreano OCRado como latino/gibberish e nota inferior mal tratada.
- `page_006_band_102`: `expanded_mask` menor que `raw_mask`.
- `page_006_band_106`: card preto/glow sem máscara de painel e sem estilo visual.

## Files

- Modify: `N:\TraduzAI\pipeline\ocr\postprocess.py`
- Modify: `N:\TraduzAI\pipeline\ocr\ocr_normalizer.py`
- Modify: `N:\TraduzAI\pipeline\vision_stack\ocr.py`
- Modify: `N:\TraduzAI\pipeline\vision_stack\cjk_segmentation_mask.py`
- Modify: `N:\TraduzAI\pipeline\inpainter\mask_builder.py`
- Modify: `N:\TraduzAI\pipeline\debug_tools\masks.py`
- Modify: `N:\TraduzAI\pipeline\main.py`
- Modify: `N:\TraduzAI\pipeline\typesetter\renderer.py`
- Modify: `N:\TraduzAI\pipeline\typesetter\style_extractor.py`
- Test: `N:\TraduzAI\pipeline\tests\test_ocr_postprocess.py`
- Test: `N:\TraduzAI\pipeline\tests\test_ocr_language_filter.py`
- Test: `N:\TraduzAI\pipeline\tests\test_mask_builder.py`
- Test: `N:\TraduzAI\pipeline\tests\test_inpaint_mask_geometry.py`
- Test: `N:\TraduzAI\pipeline\tests\test_typesetting_renderer.py`
- Test: `N:\TraduzAI\pipeline\tests\test_style_extractor.py`

---

### Task 1: OCR Fragment Merge and Reject Art Fragments

**Problem:** `page_002_band_005` cria `ocr_003 = THE SA`, marcado como `ocr_art_fragment_suspected`, mas ainda chega ao render. O texto correto já existe no mesmo balão como `THE CHILD'S SAKE.`

**Files:**
- Modify: `N:\TraduzAI\pipeline\ocr\postprocess.py`
- Modify: `N:\TraduzAI\pipeline\main.py`
- Test: `N:\TraduzAI\pipeline\tests\test_ocr_postprocess.py`

- [ ] **Step 1: Add failing test for same-balloon fragment suppression**

Create a focused fixture with three OCR items:

```python
def test_art_fragment_inside_same_phrase_is_not_rendered():
    texts = [
        {"id": "ocr_001", "text": "PLEASE, FOR", "bbox": [25, 4357, 667, 4629], "confidence": 0.91},
        {"id": "ocr_002", "text": "THE CHILD'S SAKE.", "bbox": [501, 4559, 661, 4666], "confidence": 0.94},
        {"id": "ocr_003", "text": "THE SA", "bbox": [320, 4620, 562, 4870], "confidence": 0.42, "qa_flags": ["ocr_art_fragment_suspected"]},
    ]
    out = postprocess_ocr_fragments(texts, page_language="en")
    by_id = {item["id"]: item for item in out}
    assert by_id["ocr_003"]["skip_processing"] is True
    assert "suppressed_duplicate_phrase_fragment" in by_id["ocr_003"]["qa_flags"]
    assert by_id["ocr_002"]["text"] == "THE CHILD'S SAKE."
```

- [ ] **Step 2: Run the failing test**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; python -m pytest N:\TraduzAI\pipeline\tests\test_ocr_postprocess.py::test_art_fragment_inside_same_phrase_is_not_rendered -q
```

Expected: FAIL because the suppression does not exist yet.

- [ ] **Step 3: Implement minimal suppression**

Add a postprocess step before translation/render hydration:

- If text has `ocr_art_fragment_suspected`, `render_on_art_suspected`, or very low phrase value.
- And its normalized text is substring/partial overlap of another nearby OCR phrase in the same band.
- Then set:
  - `skip_processing=True`
  - `preserve_original=False`
  - `route="suppress"`
  - add `suppressed_duplicate_phrase_fragment`
  - do not create render layer.

- [ ] **Step 4: Verify**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; python -m pytest N:\TraduzAI\pipeline\tests\test_ocr_postprocess.py -q
```

Expected: PASS.

---

### Task 2: White Balloon Mask Must Be Anchored Around Valid Text

**Problem:** `page_002_band_008`, `page_003_band_035`, `page_004_band_055` derive masks from white/art regions that are not the real balloon. This creates wrong safe boxes and wrong inpaint.

**Files:**
- Modify: `N:\TraduzAI\pipeline\inpainter\mask_builder.py`
- Modify: `N:\TraduzAI\pipeline\debug_tools\masks.py`
- Test: `N:\TraduzAI\pipeline\tests\test_mask_builder.py`

- [ ] **Step 1: Add failing test for text-anchored balloon mask**

```python
def test_derived_balloon_mask_must_enclose_text_anchor_not_art_region():
    text_bbox = [359, 8155, 708, 8305]
    wrong_component = [348, 761, 793, 1032]
    candidate = choose_balloon_component(
        text_bbox=text_bbox,
        candidates=[wrong_component],
        page_size=(800, 9000),
        band_origin_y=7952,
    )
    assert candidate is None
```

- [ ] **Step 2: Run the failing test**

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; python -m pytest N:\TraduzAI\pipeline\tests\test_mask_builder.py::test_derived_balloon_mask_must_enclose_text_anchor_not_art_region -q
```

- [ ] **Step 3: Implement the anchor gate**

For any `derived_white_crop`, `outline_seeded_contour`, or `component` bubble mask:

- Require text center inside the candidate mask or within a small tolerance.
- Require candidate area to surround the text bbox by at least 8 px on all sides when possible.
- Reject candidate if most of the component lies outside the band-local text support area.
- If rejected, set `bubble_mask_source="rejected_unanchored_component"` and do not use it for safe box or inpaint limit.

- [ ] **Step 4: Update debug output**

`04_balloon_mask.png` must show only accepted real/anchored mask. If fallback is bbox, write:

- `04_balloon_mask_fallback_bbox.png`
- `mask_decision.used_real_bubble_mask=false`
- `mask_decision.rejection_reason="unanchored_component"`

- [ ] **Step 5: Verify**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; python -m pytest N:\TraduzAI\pipeline\tests\test_mask_builder.py N:\TraduzAI\pipeline\tests\test_inpaint_mask_geometry.py -q
```

---

### Task 3: Full Interior Recovery for White-on-White Balloons

**Problem:** em balões sobre fundo branco, o pipeline enxerga só parte do interior ou pega espaços brancos externos. Isso aparece em `page_002_band_008` e nos balões brancos do capítulo.

**Files:**
- Modify: `N:\TraduzAI\pipeline\inpainter\mask_builder.py`
- Test: `N:\TraduzAI\pipeline\tests\test_mask_builder.py`

- [ ] **Step 1: Add failing test for contour-bounded interior fill**

```python
def test_white_on_white_balloon_uses_outline_to_fill_full_interior():
    mask = derive_balloon_interior_from_outline(
        outline_points=[(272, 8092), (793, 8092), (793, 8365), (272, 8365)],
        text_bbox=[359, 8155, 708, 8305],
        page_size=(800, 9000),
    )
    assert mask_contains_bbox(mask, [361, 8127, 703, 8330])
    assert mask_area_ratio(mask, [272, 8092, 793, 8365]) > 0.55
```

- [ ] **Step 2: Implement contour-first recovery**

When white component is ambiguous:

- Search edges/outline around text with expanded ROI.
- Prefer closed contour that surrounds the text bbox.
- Fill the contour interior.
- Trim to page/band boundaries.
- Reject if no closed/semi-closed contour surrounds text.

- [ ] **Step 3: Verify against the real band**

Run a small debug script or focused pipeline path against:

`N:\TraduzAI\DEBUGM\runs\god_of_death_ch1_expanded_canvas_full_20260605\debug\e2e\06_mask_segmentation\page_002_band_008`

Expected:

- `04_balloon_mask.png` covers the whole oval interior.
- `09_final_inpaint_mask.png` remains glyph/text-only, not full balloon.

---

### Task 4: Prevent Expanded Text Mask From Shrinking

**Problem:** `page_006_band_102` has `raw_mask_pixels=4435` and `expanded_mask_pixels=2379`; expansion is smaller than raw because the limit mask clips too early.

**Files:**
- Modify: `N:\TraduzAI\pipeline\inpainter\mask_builder.py`
- Test: `N:\TraduzAI\pipeline\tests\test_inpaint_mask_geometry.py`

- [ ] **Step 1: Add failing test**

```python
def test_expanded_text_mask_never_loses_raw_text_pixels_inside_limit():
    raw = make_mask([(453, 103, 561, 168)], size=(800, 300))
    limit = make_mask([(450, 102, 565, 169)], size=(800, 300))
    expanded = expand_text_mask_for_inpaint(raw, limit_mask=limit, radius=2)
    assert count_pixels(expanded & raw) == count_pixels(raw)
    assert count_pixels(expanded) >= count_pixels(raw)
```

- [ ] **Step 2: Implement monotonic expansion**

Change order:

1. Start with raw glyph mask.
2. Dilate by small elliptical kernel.
3. Intersect only newly-added pixels with limit mask.
4. Always OR original raw mask back in.

Invariant:

`final_expanded_mask = raw_mask | ((dilate(raw_mask) - raw_mask) & limit_mask)`

- [ ] **Step 3: Verify**

Expected for `page_006_band_102`:

- `expanded_mask_pixels >= raw_mask_pixels`
- no visible white text residue on blue card.

---

### Task 5: CJK/Gibberish Sign Guard and Note Handling

**Problem:** `page_004_band_052` turns Korean sign into Latin gibberish (`3 TI2]2H`) and treats `TEXT: DARLING KARAOKE` as a tiny render layer.

**Files:**
- Modify: `N:\TraduzAI\pipeline\vision_stack\ocr.py`
- Modify: `N:\TraduzAI\pipeline\vision_stack\cjk_segmentation_mask.py`
- Modify: `N:\TraduzAI\pipeline\ocr\ocr_normalizer.py`
- Test: `N:\TraduzAI\pipeline\tests\test_ocr_language_filter.py`

- [ ] **Step 1: Add failing test for visual CJK despite Latin OCR**

```python
def test_latin_gibberish_over_cjk_visual_mask_is_suppressed_for_english_source():
    item = {
        "id": "ocr_001",
        "text": "3 TI2]2H",
        "bbox": [237, 2890, 635, 3059],
        "confidence": 0.51,
        "visual_script_hint": "cjk",
    }
    out = apply_language_guards([item], source_language="en")
    assert out[0]["skip_processing"] is True
    assert "visual_cjk_suppressed" in out[0]["qa_flags"]
```

- [ ] **Step 2: Implement visual script hint**

Use CJK segmentation/mask evidence before trusting Latin OCR output:

- If source language is English.
- If visual CJK mask overlaps OCR bbox above threshold.
- If recognized text has high gibberish ratio or mixed digits/symbols.
- Suppress from translation/render/inpaint unless explicitly marked as translatable sign.

- [ ] **Step 3: Treat translator notes separately**

For text beginning with `TEXT:`, `T/N:`, `SFX:`, or scanlation-note patterns:

- Do not use balloon mask.
- Route as marginal note/text layer.
- Keep render anchored to original note bbox.
- Do not inflate note into a panel/card mask.

---

### Task 6: Card/Panel Text Mask and Style Extraction

**Problem:** `page_006_band_106` has a black/glow title card. The current mask sees only text fragments, misses glow, and renders with wrong style.

**Files:**
- Modify: `N:\TraduzAI\pipeline\typesetter\style_extractor.py`
- Modify: `N:\TraduzAI\pipeline\typesetter\renderer.py`
- Modify: `N:\TraduzAI\pipeline\inpainter\mask_builder.py`
- Test: `N:\TraduzAI\pipeline\tests\test_style_extractor.py`
- Test: `N:\TraduzAI\pipeline\tests\test_typesetting_renderer.py`

- [ ] **Step 1: Add failing style extraction test**

```python
def test_glow_card_text_extracts_light_fill_and_glow_context():
    style = infer_text_style_from_region(
        image_region=fixture("black_glow_card_the_devil_knight.png"),
        text_mask=fixture("black_glow_card_text_mask.png"),
    )
    assert style["fill_color_luma"] > 180
    assert style["background_luma"] < 80
    assert style["has_glow_context"] is True
```

- [ ] **Step 2: Add card mask rule**

If background is dark/colored and text has glow:

- Do not derive `bubble_mask` from white crop.
- Use `text_mask + glow_halo_mask` for inpaint.
- Use surrounding panel/card rectangle only as style context and layout context.
- Keep final inpaint mask limited to original text/glow, not the whole card.

- [ ] **Step 3: Render with extracted style**

For card/glow text:

- Prefer white/light fill.
- Preserve approximate font weight/condensed feel where available.
- Add subtle glow/outline if original has halo.
- Avoid Comic Neue default unless style extraction fails.

---

### Task 7: Render Even When Review Is Needed, But Use Safe Anchors

**Problem:** alguns layers ficam sem texto porque `safe_text_box` ou `render_bbox` falham. O usuário quer que renderize mesmo quando o render não for seguro.

**Files:**
- Modify: `N:\TraduzAI\pipeline\typesetter\renderer.py`
- Modify: `N:\TraduzAI\pipeline\main.py`
- Test: `N:\TraduzAI\pipeline\tests\test_typesetting_renderer.py`

- [ ] **Step 1: Add failing test**

```python
def test_review_required_layer_still_renders_inside_best_available_anchor():
    layer = {
        "text": "BEM... NÃO É COMO SE ISSO NÃO FOSSE AGRADÁVEL...",
        "safe_text_box": None,
        "render_bbox": None,
        "balloon_bbox": [462, 5012, 721, 5173],
        "source_bbox": [501, 5042, 682, 5143],
        "qa_flags": ["missing_render_bbox"],
    }
    planned = plan_fallback_render_box(layer)
    assert planned["render_bbox"] is not None
    assert bbox_center_inside(planned["render_bbox"], [462, 5012, 721, 5173])
    assert "rendered_with_review_fallback" in planned["qa_flags"]
```

- [ ] **Step 2: Implement fallback anchor priority**

Render target priority:

1. Valid `safe_text_box`.
2. Valid `bubble_inner_mask` bbox.
3. Valid `bubble_mask` bbox with inner padding.
4. Valid `balloon_bbox` with conservative inner padding.
5. Original `source_bbox` as last resort.

Always render text, but add QA flag:

`rendered_with_review_fallback`

Do not mark as success if fallback has low confidence; export gate can remain blocked.

---

### Task 8: Focused Visual Rerun and Acceptance

**Files:**
- No code changes unless previous tasks fail.

- [ ] **Step 1: Run focused tests**

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; python -m pytest `
  N:\TraduzAI\pipeline\tests\test_ocr_postprocess.py `
  N:\TraduzAI\pipeline\tests\test_ocr_language_filter.py `
  N:\TraduzAI\pipeline\tests\test_mask_builder.py `
  N:\TraduzAI\pipeline\tests\test_inpaint_mask_geometry.py `
  N:\TraduzAI\pipeline\tests\test_style_extractor.py `
  N:\TraduzAI\pipeline\tests\test_typesetting_renderer.py `
  -q
```

- [ ] **Step 2: Rerun one chapter with debug**

Run the same God of Death chapter with a new run id:

`god_of_death_ch1_visual_mask_ocr_fix_YYYYMMDD`

- [ ] **Step 3: Generate comparison sheets**

Generate sheets for:

- `page_002_band_005`
- `page_002_band_008`
- `page_003_band_035`
- `page_004_band_055`
- `page_004_band_052`
- `page_006_band_102`
- `page_006_band_106`

Each sheet must include:

- original crop
- translated crop
- `01_glyph_mask.png`
- `03_detected_text_mask.png`
- `04_balloon_mask.png`
- `05_balloon_inner_mask.png`
- `09_final_inpaint_mask.png`
- `03_inpaint_mask_overlay.jpg`
- `06_band_after_inpaint.jpg`

- [ ] **Step 4: Acceptance criteria**

The fix is acceptable only if:

- `THE SA` is not rendered as a separate text.
- `page_002_band_008` masks the real oval interior and removes text residue.
- `page_003_band_035` no longer uses building/window components as text mask.
- `page_004_band_055` renders text even when review is needed and does not mask across the panel.
- Korean sign in `page_004_band_052` is not translated/rendered as Latin gibberish.
- `page_006_band_102` has `expanded_mask_pixels >= raw_mask_pixels`.
- `page_006_band_106` masks text+glow and renders with light/glow/card-appropriate style.
- Export gate may remain `BLOCK` for real visual issues, but not because of missing render from these seven fixed cases.

---

## Self-Review

- Spec coverage: each of the seven reported visual failures has a dedicated task or acceptance check.
- Conflict check: the plan does not require replacing the whole detector or renderer; it tightens existing contracts.
- Regression guard: OCR suppression only applies to duplicate/art fragments or visual CJK/gibberish evidence, not normal English dialogue.
- Inpaint guard: text mask remains glyph/glow-only; balloon mask is used for limit/anchor, not for painting the whole balloon.
- Renderer guard: fallback render is visible but flagged, so the app does not silently claim success.

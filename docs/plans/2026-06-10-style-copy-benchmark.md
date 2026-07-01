# Style Copy Benchmark Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a repeatable benchmark that proves TraduzAI can copy text style from the original page into the final render, including font, fill color, gradient, outline, glow, shadow, and conservative no-effect cases.

**Architecture:** Use a synthetic style atlas with known ground truth plus a real-chapter regression subset from `18_regressed_items_ch3`. The benchmark validates three contracts: extraction evidence, project-layer style propagation, and final renderer pixels.

**Tech Stack:** Python 3.12, OpenCV, PIL/Pillow, pytest, existing `pipeline/typesetter/style_extractor.py`, `pipeline/main.py`, `pipeline/typesetter/style_policy.py`, and `pipeline/typesetter/renderer.py`.

---

### Task 1: Synthetic Style Atlas Fixture

**Files:**
- Create: `pipeline/tests/fixtures/style_copy_atlas/generate_style_copy_atlas.py`
- Create: `pipeline/tests/fixtures/style_copy_atlas/style_copy_atlas.png`
- Create: `pipeline/tests/fixtures/style_copy_atlas/style_copy_manifest.json`

**Cases:**
- `plain_balloon_black`: ComicNeue-Bold, black fill, no effects.
- `plain_balloon_white_on_dark`: ComicNeue-Bold, white fill, no effects.
- `white_fill_black_outline`: KOMIKAX, white fill, black outline, no glow.
- `black_fill_white_outline`: KOMIKAX, black fill, white outline, no glow.
- `dark_blue_vertical_gradient`: ComicNeue-Bold, dark blue vertical gradient, no outline.
- `impact_gradient_outline`: KOMIKAX, colored gradient, white outline.
- `cyan_card_white_glow`: KOMIKAX, white/cyan fill, cyan card background, glow.
- `black_text_pink_glow`: KOMIKAX, black fill, pink glow.
- `gray_shadow_offset`: ComicNeue-Bold, black fill, gray shadow, offset `[4, 4]`.
- `thick_impact_font`: KOMIKAX or LuckiestGuy, thick no-effect SFX.
- `thin_no_effect_ui_text`: ComicNeue-Bold, thin black UI-like text, no contour/glow/shadow.

**Expected manifest shape:**
```json
{
  "cases": [
    {
      "id": "white_fill_black_outline",
      "bbox": [40, 120, 620, 210],
      "expected": {
        "font_family": "impact",
        "font_candidates": ["KOMIKAX_.ttf", "LuckiestGuy-Regular.ttf"],
        "text_color": "#FFFFFF",
        "stroke_color": "#000000",
        "stroke_width_px_min": 2,
        "gradient": false,
        "glow": false,
        "shadow": false
      }
    }
  ]
}
```

**Validation command:**
```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'
python pipeline\tests\fixtures\style_copy_atlas\generate_style_copy_atlas.py
```

---

### Task 2: Extraction Contract Tests

**Files:**
- Create: `pipeline/tests/test_style_copy_benchmark.py`
- Modify only if needed: `pipeline/typesetter/style_extractor.py`

**Tests:**
- Every atlas case calls `extract_text_style_evidence(crop)`.
- Assert expected positives, such as `gradient=True` for gradient cases.
- Assert expected negatives, such as `glow=False` for solid outline.
- Assert no curve detection.
- Assert thick cases use `KOMIKAX_.ttf` or `LuckiestGuy-Regular.ttf`.
- Assert plain balloons default to `ComicNeue-Bold.ttf`.

**Command:**
```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'
python -m pytest pipeline\tests\test_style_copy_benchmark.py -q
```

---

### Task 3: Project-Layer Propagation Tests

**Files:**
- Modify: `pipeline/tests/test_main_emit.py`
- Modify if needed: `pipeline/main.py`
- Modify if needed: `pipeline/typesetter/style_policy.py`

**Tests:**
- Given synthetic evidence with gradient, `_style_from_evidence()` must set `estilo.cor_gradiente`.
- Given glow evidence, generated layer must include `glow`, `glow_cor`, `glow_px`.
- Given outline evidence, generated layer must include `contorno`, `contorno_px`.
- `normalize_auto_typesetting_style()` must preserve source-detected fields when `style_confidence >= 0.70`.
- Low-confidence source style must fall back to conservative defaults.

**Command:**
```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'
python -m pytest pipeline\tests\test_main_emit.py pipeline\tests\test_typesetting_style_policy.py -k "style or gradient or glow or outline" -q
```

---

### Task 4: Renderer Pixel Tests

**Files:**
- Modify: `pipeline/tests/test_typesetting_renderer.py`
- Modify if needed: `pipeline/typesetter/renderer.py`

**Tests:**
- Render gradient text and assert top/bottom glyph pixels differ in expected direction.
- Render outline text and assert ring pixels match outline color while fill pixels match fill color.
- Render glow text and assert pixels outside glyph mask contain glow color with blur falloff.
- Render no-effect plain text and assert no extra glow/shadow/outline pixels outside glyph mask.
- Render thick-font style and assert selected font path is not ComicNeue when `fonte=KOMIKAX_.ttf` or `LuckiestGuy-Regular.ttf`.

**Command:**
```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'
python -m pytest pipeline\tests\test_typesetting_renderer.py -k "gradient or glow or outline or style" -q
```

---

### Task 5: Real Chapter Regression Harness

**Files:**
- Create: `pipeline/debug_tools/run_style_copy_regression.py`
- Output: `N:\TraduzAI\DEBUGM\runs\style_copy_regression_ch3_*`

**Real input:**
- Source run: `N:\TraduzAI\DEBUGM\runs\2026-05-23_mihon_matrix20_v33\18_regressed_items_ch3`
- Pages: `1, 2, 5, 6, 7, 8`

**Pipeline sequence per page:**
```powershell
python pipeline\main.py --detect-page <test_project.json> <page_index>
python pipeline\main.py --reinpaint-page <test_project.json> <page_index>
python pipeline\main.py --retypeset <test_project.json> <page_index>
```

**Why skip standalone `--ocr-page`:**
Standalone regional OCR currently fails with:
`PaddleOCR indisponivel; OCR regional nao pode usar EasyOCR.`
The `--detect-page` path already runs detect + OCR for this regression flow.

**Assertions:**
- p05 impact text has black outline/white fill or equivalent expected contour style.
- p06/p07/p08 marked gradient remain gradient in `project.json` and render output.
- p01/p02 cyan card text becomes glow, not cyan contour.
- `curva` remains false.
- Generate:
  - `visual_report/index.html`
  - `style_render_crops.jpg`
  - `rendered_pages_contact_sheet.jpg`

---

### Task 6: Scoring and Failure Report

**Files:**
- Create: `pipeline/debug_tools/style_copy_score.py`

**Metrics:**
- `extraction_pass_rate`: expected fields vs extracted evidence.
- `propagation_pass_rate`: evidence fields vs `project.json` style fields.
- `render_pass_rate`: output pixels vs expected style signature.
- `false_positive_count`: glow/shadow/outline/gradient where manifest says no effect.
- `real_chapter_findings`: manually labeled real cases pass/fail.

**Output example:**
```json
{
  "synthetic": {
    "cases": 11,
    "passed": 10,
    "failed": ["cyan_card_white_glow"]
  },
  "real_chapter": {
    "checked_layers": 18,
    "passed": 13,
    "failed": [
      {"page": 5, "id": "sfx_visual_004", "reason": "outline inverted after SFX routing"}
    ]
  }
}
```

---

### Task 7: Iterative Fix Loop

**Order:**
1. Make synthetic extraction tests pass.
2. Make propagation tests pass.
3. Make renderer pixel tests pass.
4. Run real chapter regression.
5. Fix real regressions only when they do not break synthetic contracts.
6. Re-run all focused tests and produce final visual report.

**Final verification commands:**
```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'
python -m pytest pipeline\tests\test_style_copy_benchmark.py pipeline\tests\test_style_extractor.py pipeline\tests\test_typesetting_style_policy.py pipeline\tests\test_main_emit.py -k "style or gradient or glow or outline" -q
python pipeline\debug_tools\run_style_copy_regression.py --source-run 'N:\TraduzAI\DEBUGM\runs\2026-05-23_mihon_matrix20_v33\18_regressed_items_ch3'
```

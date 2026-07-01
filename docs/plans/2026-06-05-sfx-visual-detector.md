# Manhwa SFX Visual Detector Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Detect stylized manhwa SFX regions visually before OCR, create SFX candidate layers automatically, and feed the existing SFX translation/inpaint/render engine without destroying character art.

**Architecture:** Add a conservative `vision_stack` detector that proposes SFX bboxes from visual strokes, croma, component shape, and panel context. The detector does not translate or inpaint directly; it emits candidate regions with masks/evidence, then the existing `ocr.text_router`, `sfx.candidate`, `sfx.mask`, `sfx.inpaint_gate`, LaMa ROI inpaint, and SFX renderer own the downstream behavior. The first version must prefer review over false positives.

**Tech Stack:** Python 3.12, OpenCV, NumPy, existing TraduzAI pipeline sidecar, pytest, optional local benchmark images in `data/sfx_benchmarks/manhwa/`.

---

## Current State

The current SFX engine works after a layer exists:

- `pipeline/ocr/text_router.py` routes Hangul SFX text to `translate_sfx_inpaint_render`.
- `pipeline/sfx/candidate.py` adapts Hangul onomatopoeia to PT-BR.
- `pipeline/sfx/mask.py` builds conservative glyph masks, including color-chroma fallback for colored SFX.
- `pipeline/sfx/inpaint_gate.py` blocks unsafe inpaint.
- `pipeline/sfx/renderer.py` renders adapted SFX.

The missing part is before OCR: stylized SFX is not detected as text on real pages. On the supplied images, OCR found only `HYAAH!!` on one page and found zero text on the other, while manual SFX bboxes worked.

## Acceptance Criteria

- Supplied screenshots produce SFX candidate regions without manual bbox input.
- Candidate bboxes include enough metadata for review: `content_class=sfx`, `route_action=translate_sfx_inpaint_render` when Hangul is known, or `route_action=review_required` when script/text is unknown.
- Detector never treats normal speech balloon text as SFX.
- Detector emits `mask_evidence` and debug artifacts.
- Unsafe dense/overlapping SFX remains review-only.
- Focused SFX suite passes.

---

## Task 1: Add SFX Visual Candidate Detector

**Files:**
- Create: `pipeline/vision_stack/sfx_detector.py`
- Test: `pipeline/tests/test_sfx_visual_detector.py`

**Step 1: Write failing tests for colored and white stylized SFX**

Create synthetic images that mimic:

- red Hangul-like SFX over speed lines;
- white/blue SFX with outline and glow;
- normal black dialogue text inside white balloon.

Test expected behavior:

```python
def test_detects_colored_sfx_over_speed_lines():
    image = make_red_sfx_page()
    candidates = detect_sfx_candidates(image)
    assert len(candidates) >= 1
    assert candidates[0]["content_class"] == "sfx"
    assert candidates[0]["detector"] == "sfx_visual"
    assert candidates[0]["confidence"] >= 0.45
```

```python
def test_does_not_mark_dialogue_balloon_text_as_sfx():
    image = make_dialogue_balloon_page()
    candidates = detect_sfx_candidates(image)
    assert candidates == []
```

**Step 2: Run tests to verify failure**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_sfx_visual_detector.py -q
```

Expected: fail because `vision_stack.sfx_detector` does not exist.

**Step 3: Implement detector primitives**

In `pipeline/vision_stack/sfx_detector.py`, implement:

- `detect_sfx_candidates(image_rgb: np.ndarray, *, existing_texts=None, existing_blocks=None) -> list[dict]`
- `_candidate_masks(image_rgb)`:
  - croma mask: high saturation/chroma, not page-white;
  - contrast mask: local dark/light strokes;
  - outline/glow mask for white glyphs with blue/gray outline;
- `_component_candidates(mask, image_shape)` using `cv2.connectedComponentsWithStats`.
- `_score_candidate(crop, component, context)` with conservative scoring.

Initial candidate payload:

```python
{
    "id": "sfx_visual_001",
    "bbox": [x1, y1, x2, y2],
    "text_pixel_bbox": [x1, y1, x2, y2],
    "content_class": "sfx",
    "tipo": "sfx",
    "detector": "sfx_visual",
    "confidence": score,
    "script": "unknown",
    "route_action": "review_required",
    "translate_policy": "review",
    "render_policy": "review_required",
    "qa_flags": ["sfx_visual_candidate"],
    "sfx": {
        "source_text": "",
        "adapted_text": "",
        "visual_detector": "sfx_visual",
        "visual_confidence": score,
        "inpaint_allowed": False
    }
}
```

**Step 4: Add dedupe and suppression**

Suppress candidates that:

- overlap existing OCR text by more than 60%;
- are inside obvious speech balloon boxes unless very stylized;
- are too large relative to page area;
- are near scanlation credit/footer bands;
- touch most crop borders.

**Step 5: Run detector tests**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_sfx_visual_detector.py -q
```

Expected: pass.

**Step 6: Commit**

```powershell
git add pipeline\vision_stack\sfx_detector.py pipeline\tests\test_sfx_visual_detector.py
git commit -m "feat: detect visual manhwa sfx candidates"
```

---

## Task 2: Integrate Detector into OCR Runtime Output

**Files:**
- Modify: `pipeline/vision_stack/runtime.py`
- Modify: `pipeline/main.py`
- Test: `pipeline/tests/test_vision_stack_runtime.py`
- Test: `pipeline/tests/test_main_emit.py`

**Step 1: Write failing runtime integration test**

Create a fake OCR result with no text and a synthetic SFX image. Assert runtime/page enrichment appends `_sfx_visual_candidates` or text candidates.

Expected candidate fields:

- `content_class=sfx`
- `detector=sfx_visual`
- `route_action=review_required` for unknown text
- `qa_flags` includes `sfx_visual_candidate`

**Step 2: Run focused tests**

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_vision_stack_runtime.py pipeline\tests\test_main_emit.py -q
```

Expected: fail before integration.

**Step 3: Add runtime hook**

After normal detect/OCR completes, call:

```python
from vision_stack.sfx_detector import detect_sfx_candidates

sfx_candidates = detect_sfx_candidates(
    image_rgb,
    existing_texts=ocr_page.get("texts") or [],
    existing_blocks=ocr_page.get("_vision_blocks") or [],
)
ocr_page.setdefault("_sfx_visual_candidates", []).extend(sfx_candidates)
```

Do not blindly append to `texts` in runtime if that would alter existing OCR contracts. Prefer `_sfx_visual_candidates` first.

**Step 4: Convert candidates to text layers in `main.py`**

In the detect-page path, merge `_sfx_visual_candidates` after OCR texts. For unknown text candidates, create review layers. For candidates with later recognized Hangul, route through normal SFX path.

Candidate conversion should preserve:

- `bbox`
- `text_pixel_bbox`
- `content_class`
- `detector`
- `qa_flags`
- `sfx`

**Step 5: Run integration tests**

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_vision_stack_runtime.py pipeline\tests\test_main_emit.py -q
```

Expected: pass.

**Step 6: Commit**

```powershell
git add pipeline\vision_stack\runtime.py pipeline\main.py pipeline\tests\test_vision_stack_runtime.py pipeline\tests\test_main_emit.py
git commit -m "feat: emit visual sfx candidates from ocr runtime"
```

---

## Task 3: Add Regional OCR/Script Probe for Visual SFX Candidates

**Files:**
- Create: `pipeline/sfx/script_probe.py`
- Modify: `pipeline/main.py`
- Test: `pipeline/tests/test_sfx_script_probe.py`

**Step 1: Write tests for script classification**

Use synthetic records and optional OCR text:

- Hangul text -> route `translate_sfx_inpaint_render`.
- Japanese/kana or unreadable visual SFX -> `review_required`.
- Latin shout inside balloon -> not SFX unless visual candidate says SFX.

**Step 2: Implement `probe_sfx_candidate_script`**

Function:

```python
def probe_sfx_candidate_script(candidate: dict, recognized_text: str = "") -> dict:
    ...
```

Behavior:

- If recognized text contains Hangul, call `route_text(..., tipo="sfx")`.
- If recognized text is empty, keep `review_required`.
- If kana/CJK non-Hangul, keep `review_required` with `qa_flags=["sfx_script_unknown"]`.
- Never pass non-Hangul SFX to Hangul adapter automatically.

**Step 3: Optional regional OCR**

In `main.py`, only if cheap and available, run `run_ocr_on_block()` on candidate bbox. If OCR fails or returns empty, keep review candidate. Do not block page detection.

**Step 4: Run tests**

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_sfx_script_probe.py pipeline\tests\test_main_emit.py -q
```

Expected: pass.

**Step 5: Commit**

```powershell
git add pipeline\sfx\script_probe.py pipeline\main.py pipeline\tests\test_sfx_script_probe.py pipeline\tests\test_main_emit.py
git commit -m "feat: probe script for visual sfx candidates"
```

---

## Task 4: Connect Visual Candidates to Mask/Gate Debug Evidence

**Files:**
- Modify: `pipeline/inpainter/mask_builder.py`
- Modify: `pipeline/qa/render_geometry.py`
- Modify: `pipeline/tools/render_vision_debug_sheet.py`
- Test: `pipeline/tests/test_mask_builder.py`
- Test: `pipeline/tests/test_render_geometry.py`
- Test: `pipeline/tests/test_render_vision_debug_sheet.py`

**Step 1: Write tests for visual SFX candidates**

Assert a visual SFX candidate with bbox and no OCR text:

- calls `build_sfx_glyph_mask`;
- receives `mask_evidence.kind=sfx_glyph_mask` or review reject reason;
- produces debug-sheet category `sfx`.

**Step 2: Preserve review-only behavior**

If candidate has:

```python
"route_action": "review_required"
```

then mask/debug may run, but automatic inpaint/render must not run.

**Step 3: Promote QA flags**

Map detector/gate problems:

- `sfx_visual_candidate`
- `sfx_script_unknown`
- `sfx_mask_density_high`
- `sfx_inpaint_damaged_art_risk`

to review-only QA, not OCR rerun.

**Step 4: Run tests**

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_mask_builder.py pipeline\tests\test_render_geometry.py pipeline\tests\test_render_vision_debug_sheet.py -q
```

Expected: pass.

**Step 5: Commit**

```powershell
git add pipeline\inpainter\mask_builder.py pipeline\qa\render_geometry.py pipeline\tools\render_vision_debug_sheet.py pipeline\tests\test_mask_builder.py pipeline\tests\test_render_geometry.py pipeline\tests\test_render_vision_debug_sheet.py
git commit -m "feat: add debug evidence for visual sfx candidates"
```

---

## Task 5: Upgrade Local SFX Benchmark to Measure Detection

**Files:**
- Modify: `pipeline/tools/analyze_cjk_quality_run.py`
- Create: `pipeline/tests/test_sfx_benchmark_detection.py`

**Step 1: Add manifest format**

For local untracked files in `data/sfx_benchmarks/manhwa/`, allow sidecar JSON:

```json
{
  "expected_sfx": [
    {
      "label": "red_top",
      "bbox": [55, 35, 175, 190],
      "script": "hangul",
      "auto_inpaint": false
    }
  ]
}
```

**Step 2: Benchmark detector metrics**

When `--sfx-benchmark` is used and folder exists:

- load images;
- run `detect_sfx_candidates`;
- compare candidate bboxes against manifest using IoU;
- output:
  - `detected_count`
  - `expected_count`
  - `matched_count`
  - `missed_count`
  - `false_positive_count`
  - `mean_iou`
  - per-item records.

**Step 3: Test no-manifest fallback**

Existing behavior should remain:

- if folder missing -> `SKIP`;
- if images exist but no manifests -> `PASS` with `unlabeled`.

**Step 4: Run tests**

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_analyze_cjk_quality_run.py pipeline\tests\test_sfx_benchmark_detection.py -q
```

Expected: pass.

**Step 5: Commit**

```powershell
git add pipeline\tools\analyze_cjk_quality_run.py pipeline\tests\test_sfx_benchmark_detection.py
git commit -m "feat: measure sfx visual detection benchmark"
```

---

## Task 6: Add Manifests for the Two Supplied Images Locally

**Files:**
- Do not commit copyrighted images.
- Create local, untracked:
  - `data/sfx_benchmarks/manhwa/Captura de tela 2026-06-05 164742.json`
  - `data/sfx_benchmarks/manhwa/Captura de tela 2026-06-05 164751.json`

**Step 1: Add expected bboxes from manual probe**

Use the manually tested bboxes:

For `164742`:

- `[55, 35, 175, 190]`
- `[190, 220, 285, 355]`
- `[145, 415, 250, 555]`
- `[390, 345, 510, 515]`

For `164751`:

- `[55, 35, 235, 225]`
- `[295, 130, 475, 315]`
- `[25, 435, 170, 620]`
- `[275, 515, 455, 835]`

**Step 2: Run benchmark**

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; pipeline\venv\Scripts\python.exe -m pipeline.tools.analyze_cjk_quality_run --sfx-benchmark data\sfx_benchmarks\manhwa
```

Expected:

- `expected_count: 8`
- `matched_count >= 5` initially
- false positives documented.

**Step 3: Do not commit images/manifests unless user explicitly approves**

These are copyrighted screenshots unless proven otherwise.

---

## Task 7: Full Pipeline Smoke Test with Visual SFX Candidates

**Files:**
- Modify: `pipeline/tests/test_main_emit.py`
- Modify: `pipeline/tests/test_export_gate.py`
- Test fixture generated in test only.

**Step 1: Create synthetic end-to-end page**

Generate page with:

- normal speech balloon;
- red SFX over speed lines;
- white/blue SFX over action art.

**Step 2: Expected final project behavior**

- normal balloon remains dialogue.
- visual SFX candidate becomes text layer with `content_class=sfx`.
- unknown script candidate is review-only.
- Hangul-recognized candidate uses `translate_sfx_inpaint_render`.
- export gate returns `REVIEW` if unsafe SFX remains.

**Step 3: Run E2E tests**

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_main_emit.py pipeline\tests\test_export_gate.py -q
```

Expected: pass.

**Step 4: Commit**

```powershell
git add pipeline\tests\test_main_emit.py pipeline\tests\test_export_gate.py
git commit -m "test: add e2e visual sfx detector contract"
```

---

## Task 8: Documentation and Operator UX

**Files:**
- Modify: `docs/pipeline.md`
- Modify: `src/components/editor/toolbar/RenderStatusBadge.tsx` only if needed
- Test: existing frontend tests if UI changes

**Step 1: Document detector status**

Add section:

- visual SFX detector is conservative;
- unknown script goes to review;
- automatic inpaint only when mask/gate pass;
- benchmark folder and manifest format.

**Step 2: Optional UI copy**

If visual candidates are review-only, make badge title explicit:

```text
SFX visual detectado: revisar antes de exportar
```

**Step 3: Run docs-adjacent tests**

If UI changes:

```powershell
npm test -- --run src/lib/stores/__tests__/editorRenderPreviewCache.test.ts
```

**Step 4: Commit**

```powershell
git add docs\pipeline.md src\components\editor\toolbar\RenderStatusBadge.tsx
git commit -m "docs: explain visual sfx detector workflow"
```

---

## Final Verification

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_sfx_hangul_adapter.py pipeline\tests\test_sfx_candidate.py pipeline\tests\test_sfx_mask.py pipeline\tests\test_sfx_inpaint_gate.py pipeline\tests\test_sfx_style.py pipeline\tests\test_sfx_renderer.py pipeline\tests\test_sfx_visual_contract.py pipeline\tests\test_sfx_visual_detector.py pipeline\tests\test_sfx_script_probe.py pipeline\tests\test_main_emit.py pipeline\tests\test_export_gate.py -q
```

Expected: pass.

Run benchmark:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; pipeline\venv\Scripts\python.exe -m pipeline.tools.analyze_cjk_quality_run --sfx-benchmark data\sfx_benchmarks\manhwa
```

Expected on supplied images after manifests:

- detector finds most expected SFX bboxes;
- unsafe dense cases remain `review_required`;
- no speech balloon false positives.

---

## Known Non-Goals for This Plan

- No ML training.
- No paid API.
- No automatic Japanese/kana SFX translation.
- No unconditional inpaint of dense/character-overlapping SFX.
- No committed copyrighted manhwa images.

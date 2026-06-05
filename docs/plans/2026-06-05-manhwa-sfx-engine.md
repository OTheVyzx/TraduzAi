# Manhwa SFX Engine Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a dedicated Hangul/manhwa SFX engine that detects, adapts, removes, inpaints, and re-renders sound effects without damaging page art.

**Architecture:** Add a separate `manhwa_sfx` route parallel to the normal dialogue route. The engine should classify Hangul SFX, adapt the SFX into natural PT-BR lettering, build glyph-level masks, run conservative ROI inpaint only when QA allows it, and render translated SFX with source-derived style metadata. Unsafe cases must remain reviewable rather than silently destructive.

**Tech Stack:** Python 3.12, OpenCV, NumPy, PaddleOCR/vision OCR adapters, existing TraduzAI LaMa/AOT inpainters, existing typesetter renderer, pytest, optional local datasets from Manga109/COO or curated manhwa samples.

---

## Design Constraints

- Target manhwa SFX first: Hangul, large stylized letters, rotated/warped placement, glow, outline, motion effects, and text over art.
- Do not treat this as generic CJK OCR. Japanese/Chinese SFX can be added later behind the same interface.
- Never use rectangular inpaint for SFX. Use glyph/component masks only.
- Inpaint is opt-in per SFX candidate after QA. If the SFX overlaps face, hair, weapon, body, energy effect, or high-detail background, mark it `review_required`.
- Translation must adapt onomatopoeia, not literal words. Examples: `쿵` -> `TUM`, `쾅` -> `BOOM`, `철컥` -> `CLAC`, `우우웅` -> `VUUUM`.
- Preserve the existing project schema shape: final output still writes `paginas[]`, `text_layers`, `image_layers`, and QA metadata.

## New Route Contract

Add a route action:

```python
"translate_sfx_inpaint_render"
```

Add SFX-specific metadata to affected text layers:

```json
{
  "content_class": "sfx",
  "script": "hangul",
  "route_action": "translate_sfx_inpaint_render",
  "sfx": {
    "source_text": "쿵",
    "adapted_text": "TUM",
    "confidence": 0.82,
    "translation_mode": "onomatopoeia_adaptation",
    "style_confidence": 0.76,
    "inpaint_allowed": true,
    "qa_flags": []
  }
}
```

Unsafe example:

```json
{
  "content_class": "sfx",
  "route_action": "review_required",
  "sfx": {
    "source_text": "파앗",
    "adapted_text": "FWISH",
    "inpaint_allowed": false,
    "qa_flags": ["sfx_overlaps_character_art", "complex_background"]
  }
}
```

## Task 1: Add SFX Route Vocabulary

**Files:**
- Modify: `pipeline/ocr/text_router.py`
- Modify: `pipeline/inpainter/region_strategy.py`
- Test: `pipeline/tests/test_text_router_debug.py`
- Test: `pipeline/tests/test_inpaint_region_strategy.py`

**Step 1: Write failing route-action tests**

Add tests proving:

```python
def test_hangul_sfx_routes_to_sfx_engine():
    result = route_text("쿵", tipo="sfx")
    assert result["content_class"] == "sfx"
    assert result["route_action"] == "translate_sfx_inpaint_render"
    assert result["translate_policy"] == "adapt_sfx"
    assert result["render_policy"] == "sfx_style"
```

```python
def test_sfx_inpaint_still_blocked_without_explicit_allow():
    plan = plan_inpaint({"background_type": "sfx_text", "bbox": [10, 10, 100, 80]}, "mask.png")
    assert plan["run"] is False
    assert "sfx_preserved" in plan["qa_flags"]
```

**Step 2: Run tests and verify failure**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_text_router_debug.py pipeline\tests\test_inpaint_region_strategy.py -q
```

Expected: fail because the new route action does not exist.

**Step 3: Implement minimal route constants**

In `pipeline/ocr/text_router.py`:

- Add `translate_sfx_inpaint_render` to `ROUTE_ACTIONS`.
- Add it to translation/render/inpaint helper sets.
- Add a Hangul SFX branch before normal text fallback.

Minimal classifier:

```python
HANGUL_RE = re.compile(r"[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]")

def _looks_like_hangul_sfx(value: str, tipo: str = "") -> bool:
    if not HANGUL_RE.search(value):
        return False
    if str(tipo or "").strip().lower() in {"sfx", "sound_effect", "sound"}:
        return True
    compact = re.sub(r"\s+", "", value)
    return 1 <= len(compact) <= 10
```

**Step 4: Keep inpaint safety default**

In `pipeline/inpainter/region_strategy.py`, keep `sfx_text` blocked unless `allow_sfx=True`, but return route metadata that later QA can inspect.

**Step 5: Run tests**

Expected: route tests pass and existing region strategy safety stays intact.

**Step 6: Commit**

```powershell
git add pipeline\ocr\text_router.py pipeline\inpainter\region_strategy.py pipeline\tests\test_text_router_debug.py pipeline\tests\test_inpaint_region_strategy.py
git commit -m "feat: add manhwa sfx route contract"
```

## Task 2: Create Hangul SFX Adapter

**Files:**
- Create: `pipeline/sfx/__init__.py`
- Create: `pipeline/sfx/hangul_adapter.py`
- Create: `pipeline/sfx/lexicon_ko_pt.json`
- Test: `pipeline/tests/test_sfx_hangul_adapter.py`

**Step 1: Write failing adapter tests**

Test known mappings and fallback behavior:

```python
def test_adapts_common_hangul_sfx_to_ptbr():
    assert adapt_hangul_sfx("쿵").adapted_text == "TUM"
    assert adapt_hangul_sfx("쾅").adapted_text == "BOOM"
    assert adapt_hangul_sfx("철컥").adapted_text == "CLAC"
```

```python
def test_unknown_hangul_sfx_requires_review():
    result = adapt_hangul_sfx("힝그르")
    assert result.review_required is True
    assert "unknown_sfx" in result.qa_flags
```

**Step 2: Add lexicon**

Start small and explicit:

```json
{
  "쿵": {"pt": "TUM", "kind": "impact", "confidence": 0.9},
  "쾅": {"pt": "BOOM", "kind": "explosion_impact", "confidence": 0.9},
  "탕": {"pt": "BANG", "kind": "shot", "confidence": 0.82},
  "철컥": {"pt": "CLAC", "kind": "mechanical_click", "confidence": 0.88},
  "우우웅": {"pt": "VUUUM", "kind": "hum", "confidence": 0.84},
  "파앗": {"pt": "FWISH", "kind": "burst_motion", "confidence": 0.76}
}
```

**Step 3: Implement dataclass and adapter**

Return:

```python
@dataclass(frozen=True)
class SfxAdaptation:
    source_text: str
    adapted_text: str
    confidence: float
    kind: str
    review_required: bool
    qa_flags: list[str]
```

**Step 4: Run tests**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_sfx_hangul_adapter.py -q
```

Expected: pass.

**Step 5: Commit**

```powershell
git add pipeline\sfx pipeline\tests\test_sfx_hangul_adapter.py
git commit -m "feat: adapt hangul sfx to ptbr"
```

## Task 3: Build SFX Candidate Enrichment

**Files:**
- Create: `pipeline/sfx/candidate.py`
- Modify: `pipeline/main.py`
- Test: `pipeline/tests/test_main_emit.py`
- Test: `pipeline/tests/test_sfx_candidate.py`

**Step 1: Write failing unit tests**

Test that a routed Hangul SFX layer receives:

- `content_class=sfx`
- `script=hangul`
- `sfx.source_text`
- `sfx.adapted_text`
- `sfx.confidence`
- `sfx.translation_mode=onomatopoeia_adaptation`

**Step 2: Implement enrichment helper**

Create:

```python
def enrich_sfx_candidate(layer: dict) -> dict:
    ...
```

Rules:

- Only run when `route_action == "translate_sfx_inpaint_render"` or `content_class == "sfx"`.
- Preserve original OCR fields.
- Do not overwrite user-edited translated text.
- Set `route_action="review_required"` when adaptation confidence is below threshold.

**Step 3: Wire after OCR routing and before translation**

In `pipeline/main.py`, call enrichment before normal translation batching so SFX does not go through dialogue translation.

**Step 4: Run focused tests**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_sfx_candidate.py pipeline\tests\test_main_emit.py -q
```

Expected: pass.

**Step 5: Commit**

```powershell
git add pipeline\sfx\candidate.py pipeline\main.py pipeline\tests\test_sfx_candidate.py pipeline\tests\test_main_emit.py
git commit -m "feat: enrich manhwa sfx candidates"
```

## Task 4: Add Glyph-Level SFX Mask Builder

**Files:**
- Create: `pipeline/sfx/mask.py`
- Modify: `pipeline/vision_stack/cjk_segmentation_mask.py`
- Modify: `pipeline/inpainter/mask_builder.py`
- Test: `pipeline/tests/test_sfx_mask.py`
- Test: `pipeline/tests/test_cjk_segmentation_mask.py`

**Step 1: Write failing mask tests**

Use synthetic Hangul-like blocks:

```python
def test_sfx_mask_does_not_fill_bbox():
    mask = build_sfx_glyph_mask(image, layer)
    bbox_area = ...
    mask_area = int(np.count_nonzero(mask))
    assert mask_area < bbox_area * 0.45
```

```python
def test_sfx_mask_preserves_component_shape():
    mask = build_sfx_glyph_mask(image, layer)
    assert connected_component_count(mask) >= 2
```

**Step 2: Implement mask builder**

`build_sfx_glyph_mask(image_rgb, layer)` should combine:

- Existing CJK segmentation mask when available.
- Local contrast/stroke extraction from the SFX crop.
- Connected component filtering.
- Expansion via `expand_cjk_glyph_mask_for_inpaint`.

Reject masks when:

- density is too high;
- mask touches most of the crop border;
- mask area is near full bbox;
- no Hangul evidence exists.

**Step 3: Emit mask evidence**

Add `mask_evidence.kind = "sfx_glyph_mask"` and include:

- `raw_mask_pixels`
- `expanded_mask_pixels`
- `bbox_fill_ratio`
- `component_count`
- `reject_reason`

**Step 4: Run focused tests**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_sfx_mask.py pipeline\tests\test_cjk_segmentation_mask.py pipeline\tests\test_mask_builder.py -q
```

Expected: pass.

**Step 5: Commit**

```powershell
git add pipeline\sfx\mask.py pipeline\vision_stack\cjk_segmentation_mask.py pipeline\inpainter\mask_builder.py pipeline\tests\test_sfx_mask.py pipeline\tests\test_cjk_segmentation_mask.py pipeline\tests\test_mask_builder.py
git commit -m "feat: build conservative sfx glyph masks"
```

## Task 5: Add SFX Inpaint QA Gate

**Files:**
- Create: `pipeline/sfx/inpaint_gate.py`
- Modify: `pipeline/inpainter/region_strategy.py`
- Modify: `pipeline/qa/export_gate.py`
- Test: `pipeline/tests/test_sfx_inpaint_gate.py`
- Test: `pipeline/tests/test_export_gate.py`

**Step 1: Write failing QA tests**

Cover:

- safe sparse glyph mask over simple background -> allow;
- high-density mask -> block;
- character/art overlap flag -> block;
- missing mask evidence -> block;
- explicit manual override -> allow but mark `manual_override`.

**Step 2: Implement gate**

Return:

```python
{
  "allow_inpaint": bool,
  "strategy": "lama_component_roi" | "telea_fast" | "review_required",
  "qa_flags": [...],
  "reason": "..."
}
```

**Step 3: Integrate with `plan_inpaint`**

Allow `sfx_text` only when:

- `allow_sfx=True`;
- `mask_evidence.kind == "sfx_glyph_mask"`;
- `allow_inpaint=True`.

**Step 4: Export gate**

If a layer has `route_action=translate_sfx_inpaint_render` but `sfx.inpaint_allowed=false`, final output should be `REVIEW`, not `PASS`.

**Step 5: Run tests**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_sfx_inpaint_gate.py pipeline\tests\test_export_gate.py pipeline\tests\test_inpaint_region_strategy.py -q
```

Expected: pass.

**Step 6: Commit**

```powershell
git add pipeline\sfx\inpaint_gate.py pipeline\inpainter\region_strategy.py pipeline\qa\export_gate.py pipeline\tests\test_sfx_inpaint_gate.py pipeline\tests\test_export_gate.py
git commit -m "feat: gate sfx inpaint by visual safety"
```

## Task 6: Run Component-ROI Inpaint for SFX

**Files:**
- Modify: `pipeline/inpainter/lama_onnx.py`
- Modify: `pipeline/inpainter/__init__.py`
- Test: `pipeline/tests/test_manga_cleaner_roi_strategy.py`
- Test: `pipeline/tests/test_vision_stack_inpainter.py`
- Test: `pipeline/tests/test_sfx_inpaint_gate.py`

**Step 1: Write failing integration test**

Assert that SFX masks use component ROI jobs and never full bbox inpaint.

**Step 2: Add strategy mapping**

Map safe SFX to:

```python
"lama_component_roi"
```

Use existing `build_lama_component_rois` and merge each ROI back into the page.

**Step 3: Add debug outputs**

For each SFX ROI, write under debug:

```text
08_inpaint/<band_id>/sfx_<id>_mask.png
08_inpaint/<band_id>/sfx_<id>_before.png
08_inpaint/<band_id>/sfx_<id>_after.png
08_inpaint/<band_id>/sfx_<id>_diff.png
```

**Step 4: Run tests**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_vision_stack_inpainter.py pipeline\tests\test_manga_cleaner_roi_strategy.py pipeline\tests\test_sfx_inpaint_gate.py -q
```

Expected: pass.

**Step 5: Commit**

```powershell
git add pipeline\inpainter\lama_onnx.py pipeline\inpainter\__init__.py pipeline\tests\test_vision_stack_inpainter.py pipeline\tests\test_manga_cleaner_roi_strategy.py pipeline\tests\test_sfx_inpaint_gate.py
git commit -m "feat: inpaint sfx with component rois"
```

## Task 7: Extract Manhwa SFX Style

**Files:**
- Create: `pipeline/sfx/style.py`
- Modify: `pipeline/typesetter/style_extractor.py`
- Test: `pipeline/tests/test_sfx_style.py`
- Test: `pipeline/tests/test_style_extractor.py`

**Step 1: Write failing style tests**

Use synthetic crops for:

- white fill + black stroke;
- black fill + white glow;
- rotated SFX;
- colored fill;
- thick outline.

Assert:

- fill color;
- outline color;
- outline width;
- rotation;
- approximate scale;
- style confidence.

**Step 2: Implement style extractor**

Create:

```python
@dataclass(frozen=True)
class SfxStyle:
    fill_color: str
    stroke_color: str
    stroke_width_px: int
    glow_color: str
    glow_width_px: int
    rotation_deg: float
    scale_x: float
    scale_y: float
    confidence: float
    qa_flags: list[str]
```

**Step 3: Reuse existing `TextStyleEvidence`**

Do not duplicate color/stroke logic. Wrap it and add SFX-specific geometry:

- rotation from polygons or PCA over mask pixels;
- aspect ratio;
- bounding shape;
- confidence.

**Step 4: Run tests**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_sfx_style.py pipeline\tests\test_style_extractor.py -q
```

Expected: pass.

**Step 5: Commit**

```powershell
git add pipeline\sfx\style.py pipeline\typesetter\style_extractor.py pipeline\tests\test_sfx_style.py pipeline\tests\test_style_extractor.py
git commit -m "feat: extract manhwa sfx style"
```

## Task 8: Render Translated SFX

**Files:**
- Create: `pipeline/sfx/renderer.py`
- Modify: `pipeline/typesetter/renderer.py`
- Modify: `fonts/font-map.json`
- Test: `pipeline/tests/test_sfx_renderer.py`
- Test: `pipeline/tests/test_typesetting_renderer.py`

**Step 1: Add SFX font presets**

Add local font mapping keys:

```json
{
  "sfx_impact": "KOMIKAX_.ttf",
  "sfx_motion": "Newrotic.ttf",
  "sfx_mechanical": "CCDaveGibbonsLower W00 Regular.ttf"
}
```

If these are not good enough visually, add fonts in a later asset task. Do not block engine contracts on perfect font selection.

**Step 2: Write failing render tests**

Assert renderer:

- uses `sfx.adapted_text`;
- preserves rotation;
- applies stroke/glow;
- fits inside source SFX bbox or safe expanded bbox;
- does not render when `review_required`.

**Step 3: Implement SFX renderer adapter**

`render_sfx_layer(page_rgb, layer)` should:

- read `sfx.adapted_text`;
- choose font preset by SFX kind;
- use extracted style fields;
- render on transparent overlay;
- rotate/scale overlay;
- composite onto inpainted page.

**Step 4: Hook into existing renderer**

In `pipeline/typesetter/renderer.py`, branch only for:

```python
content_class == "sfx"
```

All normal dialogue continues through existing renderer.

**Step 5: Run tests**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_sfx_renderer.py pipeline\tests\test_typesetting_renderer.py -q
```

Expected: pass.

**Step 6: Commit**

```powershell
git add pipeline\sfx\renderer.py pipeline\typesetter\renderer.py fonts\font-map.json pipeline\tests\test_sfx_renderer.py pipeline\tests\test_typesetting_renderer.py
git commit -m "feat: render adapted manhwa sfx"
```

## Task 9: Add Visual QA and Debug Sheets

**Files:**
- Modify: `pipeline/qa/render_geometry.py`
- Modify: `pipeline/qa/page_quality.py`
- Modify: `pipeline/tools/export_visual_review_sheet.py`
- Test: `pipeline/tests/test_render_geometry.py`
- Test: `pipeline/tests/test_visual_text_leak.py`

**Step 1: Write QA tests**

Add flags:

- `sfx_render_missing`
- `sfx_render_outside_source_region`
- `sfx_inpaint_damaged_art_risk`
- `sfx_translation_unknown`
- `sfx_style_low_confidence`

**Step 2: Add debug sheet panels**

For each SFX candidate show:

- original crop;
- glyph mask;
- inpaint result;
- translated render overlay;
- final crop;
- QA flags.

**Step 3: Run tests**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_render_geometry.py pipeline\tests\test_visual_text_leak.py -q
```

Expected: pass.

**Step 4: Commit**

```powershell
git add pipeline\qa\render_geometry.py pipeline\qa\page_quality.py pipeline\tools\export_visual_review_sheet.py pipeline\tests\test_render_geometry.py pipeline\tests\test_visual_text_leak.py
git commit -m "feat: add sfx visual qa"
```

## Task 10: Add Corpus-Based Validation

**Files:**
- Create: `pipeline/tests/fixtures/sfx_manhwa/README.md`
- Create: `pipeline/tests/test_sfx_visual_contract.py`
- Modify: `pipeline/tools/analyze_cjk_quality_run.py`

**Step 1: Add fixture policy**

Use only samples that are:

- user-owned;
- public-domain/open-license;
- synthetic;
- or internal benchmark crops not committed if copyright is uncertain.

Do not commit copyrighted manhwa pages unless permission is clear.

**Step 2: Add synthetic fixture tests**

Test engine contracts without real pages:

- Hangul text crop -> SFX route;
- mask does not fill bbox;
- inpaint gate blocks complex art simulation;
- renderer emits overlay.

**Step 3: Add optional local visual benchmark**

Support a local folder:

```text
data/sfx_benchmarks/manhwa/
```

The benchmark should output JSON, not require committed images.

**Step 4: Run benchmark command**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; pipeline\venv\Scripts\python.exe -m pipeline.tools.analyze_cjk_quality_run --sfx-benchmark data\sfx_benchmarks\manhwa
```

Expected: creates a report when local data exists and exits cleanly when it does not.

**Step 5: Commit**

```powershell
git add pipeline\tests\fixtures\sfx_manhwa\README.md pipeline\tests\test_sfx_visual_contract.py pipeline\tools\analyze_cjk_quality_run.py
git commit -m "test: add manhwa sfx visual contracts"
```

## Task 11: Surface SFX Review in Editor

**Files:**
- Modify: `src/lib/projectSchema.ts`
- Modify: `src/lib/stores/editorStore.ts`
- Modify: `src/pages/Editor.tsx`
- Modify: `src/components/editor/toolbar/RenderStatusBadge.tsx`
- Test: `src/lib/__tests__/editorRenderModeUtils.test.ts`
- Test: `src/lib/stores/__tests__/editorRenderPreviewCache.test.ts`

**Step 1: Write frontend tests**

Assert the editor can show:

- SFX layer;
- adapted text;
- QA flags;
- review status;
- no crash when `sfx` metadata is missing on old projects.

**Step 2: Extend TypeScript schema**

Add optional `sfx` metadata to text layers. Keep it optional for backward compatibility.

**Step 3: Add UI status only**

Do not build a full SFX editor yet. Show status and flags so users know why a candidate was reviewed/blocked.

**Step 4: Run tests**

Run:

```powershell
npm test -- --run src/lib/__tests__/editorRenderModeUtils.test.ts src/lib/stores/__tests__/editorRenderPreviewCache.test.ts
```

Expected: pass.

**Step 5: Commit**

```powershell
git add src\lib\projectSchema.ts src\lib\stores\editorStore.ts src\pages\Editor.tsx src\components\editor\toolbar\RenderStatusBadge.tsx src\lib\__tests__\editorRenderModeUtils.test.ts src\lib\stores\__tests__\editorRenderPreviewCache.test.ts
git commit -m "feat: expose sfx review metadata in editor"
```

## Task 12: End-to-End Gate

**Files:**
- Modify: `pipeline/tests/test_main_emit.py`
- Modify: `pipeline/tests/test_export_gate.py`
- Modify: `docs/pipeline.md`

**Step 1: Add E2E contract test**

Create a small synthetic page containing:

- a normal dialogue balloon;
- one safe Hangul SFX;
- one unsafe Hangul SFX over complex art.

Expected:

- dialogue uses normal route;
- safe SFX uses `translate_sfx_inpaint_render`;
- unsafe SFX is `review_required`;
- final export gate is `REVIEW` when unsafe SFX remains.

**Step 2: Run focused E2E tests**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_main_emit.py pipeline\tests\test_export_gate.py -q
```

Expected: pass.

**Step 3: Update docs**

In `docs/pipeline.md`, document:

- SFX route;
- QA behavior;
- fallback policy;
- local benchmark path.

**Step 4: Run full focused pipeline suite**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_sfx_hangul_adapter.py pipeline\tests\test_sfx_candidate.py pipeline\tests\test_sfx_mask.py pipeline\tests\test_sfx_inpaint_gate.py pipeline\tests\test_sfx_style.py pipeline\tests\test_sfx_renderer.py pipeline\tests\test_main_emit.py pipeline\tests\test_export_gate.py -q
```

Expected: pass.

**Step 5: Commit**

```powershell
git add pipeline\tests\test_main_emit.py pipeline\tests\test_export_gate.py docs\pipeline.md
git commit -m "test: validate manhwa sfx pipeline end to end"
```

## External Reuse Policy

- Koharu: use as architecture/reference and possible renderer/PSD concepts only under GPL-compatible policy.
- manga-image-translator: benchmark ideas for OCR/inpaint/render, avoid direct port until license and dependency cost are reviewed.
- BallonsTranslator: useful for review UX and manual correction workflow, but treat as GPL reference.
- MangaInpainting: use as conceptual reference for line/screentone-preserving inpaint; do not depend on old runtime directly.
- Manga109/COO/MangaSeg: use for validation/training research, not as runtime dependency.

## Success Criteria

- Hangul SFX is not routed as normal dialogue.
- Common Korean SFX maps to natural PT-BR equivalents.
- SFX masks are glyph/component-level, not bbox-level.
- Inpaint is blocked unless QA allows it.
- Safe SFX can be removed and re-rendered without changing unrelated art pixels beyond a strict threshold.
- Unsafe SFX appears in the editor/export gate as review required.
- Existing dialogue OCR, inpaint, render, and export tests remain passing.


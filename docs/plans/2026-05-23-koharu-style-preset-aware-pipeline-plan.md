# Koharu-Style Preset-Aware Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Copy Koharu's artifact/evidence behavior into TraduzAI while keeping detection selected only by the existing manga/manhwa engine presets.

**Architecture:** `resolve_engine_preset()` remains the single source of truth for detector choice. The Koharu-style changes live after detection: validated text evidence, bubble masks, safe inpaint, sampled solid fill, OCR reconciliation, and typeset placement based on validated artifacts rather than broad OCR bboxes.

**Tech Stack:** Python 3.12, OpenCV, NumPy, PaddleOCR, existing `pipeline/vision_stack`, `pipeline/inpainter`, `pipeline/layout`, `pipeline/typesetter`, `pipeline/strip`, pytest, `DEBUGM` visual runs.

---

## Non-Negotiable Rule

Do not hard-code Koharu's detector globally.

Detection stays preset-driven:

- `manga` uses whatever `pipeline/vision_stack/engine_presets.py` declares for manga.
- `manga_ocr_guided` uses the same preset system with OCR-guided mask strategy.
- `manhwa_manhua` uses the manhwa/manhua preset.
- `manhwa_manhua_ocr_guided` uses the manhwa/manhua OCR-guided preset.
- `default` remains legacy/default.

Koharu behavior is copied as a pipeline contract:

```text
preset detector -> text candidates -> segment evidence -> bubble mask -> OCR reconciliation -> translation -> inpaint -> typeset
```

The detector produces candidates. It does not alone authorize inpaint or final render placement.

---

## Task 1: Lock The Preset Detector Contract

**Files:**
- Modify: `pipeline/vision_stack/engine_presets.py`
- Modify: `pipeline/vision_stack/runtime.py`
- Test: `pipeline/tests/test_engine_presets.py`
- Test: `pipeline/tests/test_vision_stack_runtime.py`

**Step 1: Write failing tests**

Add tests that assert the detector choice comes from the preset and is not replaced by a Koharu default:

```python
def test_manga_preset_keeps_declared_detector():
    preset = resolve_engine_preset({"engine_preset_id": "manga"})
    assert preset.detector == COMIC_TEXT_BUBBLE_DETECTOR

def test_manhwa_preset_keeps_declared_detector():
    preset = resolve_engine_preset({"engine_preset_id": "manhwa_manhua"})
    assert preset.detector == COMIC_TEXT_BUBBLE_DETECTOR
```

Add a runtime test for `_detector_model_for_preset(...)`:

```python
def test_runtime_detector_model_is_preset_adapter_not_koharu_override():
    preset = resolve_engine_preset({"engine_preset_id": "manhwa_manhua"})
    assert _detector_model_for_preset(preset) in {"comic-text-detector", "anime-text-yolo-n"}
```

Run:

```powershell
cd N:\TraduzAI\pipeline
python -m pytest tests\test_engine_presets.py tests\test_vision_stack_runtime.py -q
```

Expected: PASS after any required test fixture updates. If a new assertion fails, fix the adapter, not the preset itself.

**Step 2: Make detector adapter explicit**

In `pipeline/vision_stack/runtime.py`, make `_detector_model_for_preset(...)` explicit:

- accepts `EnginePreset`;
- maps supported preset detector IDs to actual local model loader IDs;
- records the original preset detector separately from the local loader model.

Required metadata on page result:

```python
page_result["_engine_preset"]["detector"] = preset.detector
page_result["_engine_preset"]["detector_loader"] = actual_loader_id
```

**Step 3: Run tests**

```powershell
cd N:\TraduzAI\pipeline
python -m pytest tests\test_engine_presets.py tests\test_vision_stack_runtime.py -q
```

Expected: PASS.

---

## Task 2: Normalize Detector Output Into Candidate Artifacts

**Files:**
- Modify: `pipeline/vision_stack/runtime.py`
- Modify: `pipeline/vision_stack/detector.py` only if current block shape lacks required fields
- Test: `pipeline/tests/test_vision_stack_detector.py`
- Test: `pipeline/tests/test_vision_stack_runtime.py`

**Step 1: Write failing test**

For a detected block, assert that the downstream page result has a stable candidate contract:

```python
block = page_result["_vision_blocks"][0]
assert block["detector_preset_id"] == "manhwa_manhua"
assert block["detector_engine_id"]
assert block["detector_loader"]
assert block["candidate_kind"] in {"text_box", "bubble_text_box", "layout_text"}
assert block["validated_by_segment_mask"] is False
```

Run:

```powershell
cd N:\TraduzAI\pipeline
python -m pytest tests\test_vision_stack_runtime.py::test_detect_blocks_include_preset_candidate_metadata -q
```

Expected: FAIL before implementation.

**Step 2: Add metadata attachment helper**

Add a helper in `pipeline/vision_stack/runtime.py`:

```python
def _attach_detector_candidate_metadata(block: dict, *, preset: EnginePreset, detector_loader: str) -> dict:
    item = dict(block)
    item["detector_preset_id"] = preset.id
    item["detector_engine_id"] = preset.detector
    item["detector_loader"] = detector_loader
    item.setdefault("candidate_kind", "text_box")
    item.setdefault("validated_by_segment_mask", False)
    return item
```

Call this right after detection, before OCR/layout/inpaint receive `_vision_blocks`.

**Step 3: Preserve compatibility**

Do not remove existing fields:

- `bbox`
- `confidence`
- `text_pixel_bbox`
- `line_polygons`
- `balloon_bbox`
- `balloon_type`
- `block_profile`

Only add metadata.

**Step 4: Run tests**

```powershell
cd N:\TraduzAI\pipeline
python -m pytest tests\test_vision_stack_detector.py tests\test_vision_stack_runtime.py -q
```

Expected: PASS.

---

## Task 3: Make Segmentation The Text Evidence Gate

**Files:**
- Modify: `pipeline/inpainter/mask_builder.py`
- Modify: `pipeline/vision_stack/runtime.py`
- Test: `pipeline/tests/test_inpaint_mask_geometry.py`
- Test: `pipeline/tests/test_mask_builder.py`

**Step 1: Write failing tests**

Add tests for:

1. Broad OCR/detect bbox with text only in top source box.
2. Lower source box contains face/art strokes.
3. Only the top source box becomes validated.
4. If no segment/glyph evidence exists, no inpaint mask is emitted.

Expected metadata:

```python
block["_validated_text_source_bboxes"] == [[408, 32, 1199, 332]]
block["validated_by_segment_mask"] is True
block["_rejected_text_source_bboxes"] == [[312, 219, 1326, 883]]
```

Run:

```powershell
cd N:\TraduzAI\pipeline
python -m pytest tests\test_inpaint_mask_geometry.py::test_segment_evidence_validates_source_boxes_without_art -q
```

Expected: FAIL before implementation.

**Step 2: Implement shared validation metadata**

In `build_raw_text_mask_from_image(...)`, record:

- `_validated_text_source_bboxes`
- `_rejected_text_source_bboxes`
- `_raw_text_evidence_bbox`
- `_raw_text_evidence_pixels`
- `validated_by_segment_mask`

Use current raw text evidence logic. Do not use bbox alone.

**Step 3: Update runtime fallback rule**

In `vision_blocks_to_mask(...)`, preserve this rule:

```python
if raw_text_evidence_rejected(block):
    continue
```

Never fallback to broad bbox after segment evidence rejected a candidate.

**Step 4: Run tests**

```powershell
cd N:\TraduzAI\pipeline
python -m pytest tests\test_inpaint_mask_geometry.py tests\test_mask_builder.py -q
```

Expected: PASS.

---

## Task 4: OCR Reconciliation Uses Preset Detect Plus Segment Evidence

**Files:**
- Modify: `pipeline/vision_stack/runtime.py`
- Modify: `pipeline/ocr/postprocess.py` if token cleanup is needed
- Test: `pipeline/tests/test_vision_stack_ocr.py`
- Test: `pipeline/tests/test_ocr_geometry_dedupe.py`

**Step 1: Write failing test**

Create a page where full-page OCR returns one huge merged block but detect/segment evidence has two separated source boxes.

Expected:

- OCR output is split or clamped to validated source boxes.
- Broad merged OCR text cannot become a single translation/typeset item over face/art.
- Original detect preset metadata remains attached.

Example assertion:

```python
assert all(text.get("_validated_text_source_bboxes") for text in page_result["texts"])
assert not any(text["bbox"][3] > face_top for text in page_result["texts"] if text["tipo"] == "narracao")
```

Run:

```powershell
cd N:\TraduzAI\pipeline
python -m pytest tests\test_vision_stack_ocr.py::test_full_page_ocr_clamps_to_validated_source_boxes -q
```

Expected: FAIL before implementation.

**Step 2: Add reconciliation helper**

In `runtime.py`, after OCR mapping and before translation/layout enrichment:

```python
def _reconcile_ocr_with_validated_sources(page_result: dict) -> dict:
    ...
```

Rules:

- If `text` has `_validated_text_source_bboxes`, use their union as `layout_bbox` and `text_pixel_bbox` unless explicit line polygons are better.
- If a text block has multiple validated source boxes separated by a large gap, split into multiple text items when possible.
- If OCR text is too long for the validated region, add QA flag `ocr_overmerged_validated_sources`.
- Do not drop text just because it is long; mark it and constrain downstream.

**Step 3: Preserve full-page OCR speed**

Do not disable `TRADUZAI_PADDLE_FULL_PAGE`.

The full-page OCR remains allowed, but the geometry authority becomes:

```text
preset detect candidates + segment evidence > full-page OCR bbox
```

**Step 4: Run tests**

```powershell
cd N:\TraduzAI\pipeline
python -m pytest tests\test_vision_stack_ocr.py tests\test_ocr_geometry_dedupe.py -q
```

Expected: PASS.

---

## Task 5: Inpaint Uses Koharu-Style Evidence But Preset Strategy

**Files:**
- Modify: `pipeline/inpainter/__init__.py`
- Modify: `pipeline/vision_stack/runtime.py`
- Test: `pipeline/tests/test_vision_stack_inpainter.py`
- Test: `pipeline/tests/test_inpaint_mask_geometry.py`

**Step 1: Write failing tests**

Test that mask strategy remains preset-driven:

```python
assert page_result["_engine_preset"]["mask_strategy"] == "roi_segmentation_assisted"
```

But the actual erase area must be segment evidence:

```python
assert face_zone_changed_pixels == 0
assert top_text_zone_changed_pixels > 0
```

Run:

```powershell
cd N:\TraduzAI\pipeline
python -m pytest tests\test_vision_stack_inpainter.py::VisionStackInpainterTests::test_inpaint_uses_preset_strategy_and_validated_text_evidence -q
```

Expected: FAIL if metadata/behavior is missing.

**Step 2: Make fast solid fill consume validated sources**

Continue from `docs/plans/2026-05-23-koharu-evidence-solid-fill-typeset-plan.md`:

- replace pure white assumptions with sampled solid color;
- only sample inside bubble/validated source region;
- reject texture/gradient/high variance;
- pass remaining mask to real inpaint.

**Step 3: Record preset plus evidence in debug**

In `debug_inpaint/.../metadata.json`, include:

- `engine_preset_id`
- `mask_strategy`
- `detector_engine_id`
- `validated_text_source_bboxes`
- `rejected_text_source_bboxes`
- `fast_solid_fill_color`
- `fast_solid_fill_reject_reason`

**Step 4: Run tests**

```powershell
cd N:\TraduzAI\pipeline
python -m pytest tests\test_vision_stack_inpainter.py tests\test_inpaint_mask_geometry.py -q
```

Expected: PASS.

---

## Task 6: Typeset Uses Validated Source/Bubble Area

**Files:**
- Modify: `pipeline/layout/balloon_layout.py`
- Modify: `pipeline/typesetter/renderer.py`
- Test: `pipeline/tests/test_typesetting_layout.py`
- Test: `pipeline/tests/test_typesetting_renderer.py`

**Step 1: Write failing test**

Use the Monster Actor page 002 pattern:

```python
block = {
    "bbox": [312, 32, 1326, 883],
    "_validated_text_source_bboxes": [[408, 32, 1199, 332]],
    "tipo": "narracao",
    "balloon_type": "white",
}
```

Expected:

- render target comes from `_validated_text_source_bboxes`;
- render bbox does not overlap lower art/face;
- if text does not fit, emit QA instead of expanding over art.

Run:

```powershell
cd N:\TraduzAI\pipeline
python -m pytest tests\test_typesetting_layout.py::test_typeset_target_uses_validated_source_when_ocr_bbox_is_broad -q
```

Expected: FAIL before implementation.

**Step 2: Add target priority**

Target priority:

1. explicit user/manual edited box;
2. connected lobe/subregion box;
3. `_validated_text_source_bboxes`;
4. `layout_safe_bbox`;
5. `bubble_bbox`;
6. fallback bbox.

Do not let fallback bbox override validated source boxes for broad no-line OCR.

**Step 3: Add containment QA**

In `renderer.py`, compute:

- `render_validated_containment`
- `render_balloon_containment`

If validated containment is low, reject candidate and retry smaller. If still impossible, flag:

- `render_outside_validated_text_source`
- `validated_source_too_small_for_translation`

**Step 4: Run tests**

```powershell
cd N:\TraduzAI\pipeline
python -m pytest tests\test_typesetting_layout.py tests\test_typesetting_renderer.py -q
```

Expected: PASS.

---

## Task 7: Bubble Mask By ID For Safe Areas

**Files:**
- Modify: `pipeline/strip/detect_balloons.py`
- Modify: `pipeline/layout/balloon_layout.py`
- Modify: `pipeline/typesetter/renderer.py`
- Test: `pipeline/tests/test_strip_balloon_bbox_propagation.py`
- Test: `pipeline/tests/test_layout_analysis.py`

**Step 1: Write failing test**

Create two overlapping/connected balloons. Expected:

- each lobe or bubble region has a stable `bubble_id`;
- typeset safe area uses matching `bubble_id`, not whole page or broad bbox;
- connected balloon split remains semantic.

Run:

```powershell
cd N:\TraduzAI\pipeline
python -m pytest tests\test_layout_analysis.py::test_bubble_id_mask_drives_connected_balloon_safe_area -q
```

Expected: FAIL if bubble ID mask is not propagated.

**Step 2: Add bubble ID metadata**

Propagate:

- `bubble_id`
- `bubble_mask_bbox`
- `bubble_inner_bbox`
- `connected_lobe_bboxes`
- `connected_lobe_ids`

Keep old `balloon_bbox` for compatibility.

**Step 3: Use bubble ID for layout/render checks**

Renderer should prefer bubble/lobe safe mask when available. Bbox remains fallback.

**Step 4: Run tests**

```powershell
cd N:\TraduzAI\pipeline
python -m pytest tests\test_strip_balloon_bbox_propagation.py tests\test_layout_analysis.py -q
```

Expected: PASS.

---

## Task 8: Pipeline Stage Trace Like Koharu Needs/Produces

**Files:**
- Modify: `pipeline/main.py`
- Modify: `pipeline/vision_stack/runtime.py`
- Modify: `pipeline/strip/run.py`
- Test: `pipeline/tests/test_debug_report.py`
- Test: `pipeline/tests/test_render_plan_trace_integrity.py`

**Step 1: Write failing test**

Assert every page exports a stage trace:

```python
trace = page_result["_pipeline_artifacts"]
assert trace["TextBoxes"]["producer"] == page_result["_engine_preset"]["detector"]
assert trace["SegmentMask"]["producer"] == page_result["_engine_preset"]["segmenter"]
assert trace["BubbleMask"]["producer"] == page_result["_engine_preset"]["bubble_segmenter"]
assert trace["OcrText"]["producer"] == page_result["_engine_preset"]["ocr"]
```

Run:

```powershell
cd N:\TraduzAI\pipeline
python -m pytest tests\test_render_plan_trace_integrity.py::test_pipeline_artifacts_record_preset_producers -q
```

Expected: FAIL before implementation.

**Step 2: Add artifact trace**

Attach:

```python
page_result["_pipeline_artifacts"] = {
    "TextBoxes": {"producer": preset.detector, "status": "ok"},
    "SegmentMask": {"producer": preset.segmenter, "status": "ok"},
    "BubbleMask": {"producer": preset.bubble_segmenter, "status": "ok"},
    "OcrText": {"producer": preset.ocr, "status": "ok"},
    "Inpainted": {"producer": preset.inpainter, "status": "ok"},
    "FinalRender": {"producer": "traduzai-typesetter", "status": "ok"},
}
```

Mark skipped/failed stages explicitly.

**Step 3: Export to debug**

Write the artifact trace to:

- `debug/e2e/00_run/pipeline_artifacts.json`
- `project.json` page profile or internal debug field.

**Step 4: Run tests**

```powershell
cd N:\TraduzAI\pipeline
python -m pytest tests\test_debug_report.py tests\test_render_plan_trace_integrity.py -q
```

Expected: PASS.

---

## Task 9: Real Run Validation Matrix

**Files:**
- No source edit unless validation exposes a bug.

**Step 1: Validate Monster Actor Ch75**

Run the same chapter that exposed the face-mask issue.

Required checks:

- `page_002_band_005`: face-zone changed pixels = 0.
- translated page 002: no huge OCR translation rendered over face/art.
- render plan target source = `validated_text_source`.
- detect metadata still says preset detector, not Koharu override.

**Step 2: Validate previous problem chapters**

Run:

- `C:\Users\PICHAU\Downloads\Articuno (comick)_Ch. 61 OFFICIAL TRANSLATION`
- `C:\Users\PICHAU\Downloads\Chapter 1`
- `C:\Users\PICHAU\Downloads\Chapter 39`
- `D:\Mihon pra pc\downloads\mangas\Manhwatop (EN)\The God of Death\Chapter 2.cbz`
- selected folders from `D:\Mihon pra pc\downloads\mangas`

Required checks:

- black boxes do not get white fill;
- colored solid boxes use sampled color or fall back safely;
- connected balloons do not collapse into a wrong center;
- manga/manhwa presets remain visible in output metadata;
- OCR overmerged blocks are clamped/split before typeset.

**Step 3: Compare timing**

Use latest reference:

- Monster Actor v5 total: about `128.98s`.
- `strip_process_bands_total`: about `73.20s`.

Accept small overhead for traceability and visual correctness. Reject broad slowdown caused only by debug or duplicate model runs.

---

## Final Verification Commands

```powershell
cd N:\TraduzAI
python -m py_compile pipeline\vision_stack\engine_presets.py pipeline\vision_stack\runtime.py pipeline\vision_stack\detector.py pipeline\inpainter\mask_builder.py pipeline\inpainter\__init__.py pipeline\layout\balloon_layout.py pipeline\typesetter\renderer.py

cd N:\TraduzAI\pipeline
python -m pytest tests\test_engine_presets.py tests\test_vision_stack_runtime.py tests\test_vision_stack_detector.py -q
python -m pytest tests\test_inpaint_mask_geometry.py tests\test_vision_stack_inpainter.py -q
python -m pytest tests\test_vision_stack_ocr.py tests\test_ocr_geometry_dedupe.py -q
python -m pytest tests\test_typesetting_layout.py tests\test_typesetting_renderer.py tests\test_layout_analysis.py -q
python -m pytest tests\test_debug_report.py tests\test_render_plan_trace_integrity.py -q
```

Expected:

- detector selection remains preset-driven;
- evidence validation prevents bbox-only inpaint;
- OCR broad blocks cannot become broad typeset blocks;
- fast solid fill samples only stable solid regions;
- typeset respects validated text source and bubble/lobe masks;
- debug artifacts explain producer, preset, evidence, target, and fallback decisions.

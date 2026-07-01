# Dark Points And Connected Layout Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restore the missing `1,000 points..` dark bubble in full chapter 2 runs and make dark connected bubbles use the same lobe-positioning behavior as the working white connected bubbles from chapter 1.

**Architecture:** Treat negative-image detection as evidence, not as a separate final pipeline. The `1,000 points..` case must become a normal text entity in the full-page run, then use the normal inpaint/render path on the original image. Dark connected bubbles must keep the same lobe metadata contract used by white connected bubbles unless the dark mask is explicitly unsafe.

**Tech Stack:** Python 3.12, OpenCV, PaddleOCR, existing TraduzAI vision/runtime/layout/typesetter pipeline.

---

## Reference Evidence

- Current failing full run: `N:\TraduzAI\DEBUGM\runs\god_of_death_ch2_dark_connected_bothlobes_full3_20260619`
- Old run where `1,000 points..` was detected: `N:\TraduzAI\DEBUGM\runs\god_of_death_ch2_page002_negative_full_20260618`
- Old `points` block in that run:
  - `band`: `page_001_band_011`
  - `original`: `1,000 points..`
  - `translated`: `1.000 pontos..`
  - `bbox`: `[107, 5384, 269, 5424]`
  - `balloon_bbox`: `[69, 5350, 301, 5442]`
  - `bubble_mask_bbox`: `[24, 5287, 346, 5505]`
  - `bubble_mask_source`: `image_white_bubble_mask`
  - `layout_safe_reason`: `visual_rect_inner`
- Current failing full run page 2 sequence:
  - `page_002_band_007`: `Quest Completion...`
  - `page_002_band_008`: `The quest reward is...`
  - no renderable `1,000 points..`
  - `page_002_band_010`: `SHUT UP!`
- Working white connected reference: `N:\TraduzAI\DEBUGM\runs\god_of_death_ch1_dark_typeset_spacing_regression_20260619\debug\e2e\06_mask_segmentation\page_003_band_046`
- Current dark connected reference: `page_002_band_011` and `page_002_band_023` in the full3 cap 2 run.

## Task 1: Add A Regression Fixture For The Missing `1,000 points..`

**Files:**
- Modify: `pipeline/tests/test_vision_stack_runtime.py`
- Modify if needed: `pipeline/tests/test_vision_stack_detector.py`

**Step 1: Write the failing test**

Create a focused test using a crop from chapter 2 page 2 around `y=5200..5600`, covering the `1,000 points..` bubble. The test must assert that the runtime emits a renderable text block with:

```python
assert "1,000 points" in normalized_original
assert text["route_action"] == "translate_inpaint_render"
assert text["bubble_mask_source"] in {"image_dark_bubble_mask", "image_white_bubble_mask"}
assert text["bubble_mask_bbox"]
assert text["mask_evidence"]["raw_mask_pixels"] > 0
```

**Step 2: Run the test and confirm failure**

Run:

```powershell
cd pipeline
..\pipeline\venv\Scripts\python.exe -m pytest tests\test_vision_stack_runtime.py -k "points" -x -vv
```

Expected before fix: the text block is missing in the full normal-image path.

## Task 2: Preserve Safe Split Coverage Around Sparse Dark Bubbles

**Files:**
- Modify: `pipeline/strip/process_bands.py` or the actual band splitter owner found by `rg "band_y_top|process_bands|safe split"`
- Modify: `pipeline/vision_stack/runtime.py`

**Step 1: Identify the owning split function**

Run:

```powershell
rg -n "band_y_top|safe split|split.*band|process_bands|overlap" pipeline -g "*.py"
```

Confirm why the full run creates `page_002_band_007`, `page_002_band_008`, then skips the `1,000 points..` region as a text-producing band.

**Step 2: Add a failing split contract test**

The test must assert that a sparse dark bubble between two detected text bands either:

- becomes its own band, or
- is included in an overlapping neighbor band with enough vertical margin for OCR.

**Step 3: Implement minimal split preservation**

Add a dark bubble candidate pass before final band pruning:

- Detect dark oval/card bubbles using the existing `image_dark_bubble_mask`/negative evidence logic.
- When a candidate has a high-confidence bubble body and visible light glyph pixels, force a band interval around the candidate.
- Do not force bands for scanlation credits or page furniture.
- Keep this as evidence-driven, not a global fallback.

**Step 4: Verify the `points` band exists**

Run a page-only debug for chapter 2 page 2 and assert a band/text entry exists for `1,000 points..`.

## Task 3: Promote Negative OCR Evidence Into The Normal Page Contract

**Files:**
- Modify: `pipeline/vision_stack/runtime.py`
- Test: `pipeline/tests/test_vision_stack_runtime.py`

**Step 1: Write a failing test from old negative run behavior**

Use the old run as the expected contract:

```python
assert text["original"] == "1,000 points.."
assert text["translated"] == "1.000 pontos.."
assert text["layout_safe_reason"] == "visual_rect_inner"
```

The normal full-page path should produce the same semantic entity, even if the evidence came from negative OCR.

**Step 2: Implement evidence merge**

When negative-image OCR finds a text in a dark bubble:

- map bbox back to original page coordinates;
- attach `dark_bubble_negative_evidence`;
- keep the original-image `bubble_mask_source` as `image_dark_bubble_mask`;
- keep the text render/inpaint on the original image;
- reject only if it overlaps an already accepted text with same content and same lobe.

**Step 3: Verify no duplicate text**

Run:

```powershell
cd pipeline
..\pipeline\venv\Scripts\python.exe -m pytest tests\test_vision_stack_runtime.py -k "dark_bubble or points or negative" -x -vv
```

Expected: `1,000 points..` appears once.

## Task 4: Make Dark Connected Bubbles Use The White Connected Lobe Contract

**Files:**
- Modify: `pipeline/typesetter/renderer.py`
- Modify if needed: `pipeline/layout/balloon_layout.py`
- Test: `pipeline/tests/test_typesetting_renderer.py`
- Test: `pipeline/tests/test_typesetting_layout.py`

**Step 1: Add a parity test**

Build two text fixtures:

- white connected pair, equivalent to cap 1 `page_003_band_046`;
- dark connected pair, equivalent to cap 2 `page_002_band_011`.

Assert both preserve:

```python
assert text["connected_lobe_bboxes"]
assert text["connected_position_bboxes"]
assert text["layout_group_size"] >= 2
assert "connected_layout_disabled_dark_panel_visual_mask" not in text["qa_flags"]
```

Only allow disabling connected layout when the dark mask is unsafe, not merely because `bubble_mask_source == "image_dark_bubble_mask"`.

**Step 2: Replace the broad dark-layout disable**

Current owner:

```python
pipeline/typesetter/renderer.py
_should_disable_connected_layout_for_dark_panel_visual_mask()
```

Change policy:

- Do not disable connected layout for `image_dark_bubble_mask` when:
  - `dark_bubble_connected_lobes_promoted` is present;
  - there are at least two `connected_lobe_bboxes`;
  - lobe boxes are distinct and have sane area;
  - each text anchor overlaps one lobe strongly.
- Still disable when the mask is rectangular/panel-like, low confidence, or has no lobe separation.

**Step 3: Preserve the cap 1 behavior**

The cap 1 white connected behavior is two independent texts positioned in their own lobes. The dark path should mirror that:

- each lobe text keeps its own `safe_text_box`;
- no semantic text from one lobe leaks into the other;
- no forced combined paragraph split unless the source text is truly one OCR block spanning both lobes.

## Task 5: Fix Dark Connected Text Payload Contamination

**Files:**
- Modify: `pipeline/vision_stack/runtime.py`
- Modify if needed: `pipeline/layout/balloon_layout.py`
- Test: `pipeline/tests/test_vision_stack_runtime.py`

**Step 1: Add a failing test for `page_002_band_023`**

Use the observed wrong payload:

- left lobe incorrectly contains: `You were loyal ... You the king`
- right lobe contains: `You were the king of being a pushover...`

Expected:

- left lobe should contain only the left source sentence;
- right lobe should contain only the right source sentence;
- no suffix/prefix from adjacent lobe.

**Step 2: Filter OCR line polygons and source text by lobe**

When a dark connected lobe has sibling lobe metadata:

- clip line polygons to the current lobe before OCR text assembly;
- remove text lines whose centers fall inside a sibling lobe;
- reject combined OCR text if it crosses the seam and a sibling block exists.

**Step 3: Verify project payload**

After a page-only run, inspect `project.json` and assert:

```powershell
rg -n "You the king|tarefa simples|1,000 points|1.000 pontos" <run>\project.json
```

Expected:

- no `You the king` contamination in left lobe;
- no mistranslated "tarefa simples" if the English phrase is corrected or retranslated;
- `1,000 points..` present.

## Task 6: Visual Validation Runs

**Files:**
- No code files; create temporary debug sheets under `.codex-tmp`.

**Step 1: Run chapter 2 page 2 only**

Use a focused config that runs the real `Chapter 2.cbz` page 2.

Validate visually:

- `1,000 points..` becomes `1.000 pontos..`;
- mask covers only the glyphs/text area inside its dark bubble;
- the bubble remains black with glow preserved;
- `I am called 'System'` is positioned like the source lobe;
- `page_002_band_023` has both lobes clean and no text contamination.

**Step 2: Run full chapter 2**

Run the full cap 2 config and inspect:

- `translated\002.jpg`;
- dark-bubble contact sheet;
- `debug\e2e\06_mask_segmentation\page_002_band_007..011`;
- `debug\e2e\06_mask_segmentation\page_002_band_023`.

**Step 3: Run chapter 1 regression**

Run the relevant cap 1 page/bands that previously worked:

- `page_003_band_046`
- any white connected balloon cases already used in tests.

Expected:

- white connected behavior unchanged;
- no rectangular mask regression;
- no changed positioning in cap 1 connected lobes.

## Task 7: Acceptance Criteria

The fix is complete only when all are true:

- `1,000 points..` appears in the full cap 2 `project.json`.
- The translated output visually shows `1.000 pontos..` in the correct dark bubble.
- Dark connected bubbles no longer carry `connected_layout_disabled_dark_panel_visual_mask` when they have reliable lobe metadata.
- The cap 2 connected dark lobes position text like the cap 1 connected white lobes: one text per lobe, centered in the lobe safe area.
- `page_002_band_023` has no left/right payload contamination.
- Cap 1 connected white balloons remain visually unchanged.
- Focused tests pass:

```powershell
cd pipeline
..\pipeline\venv\Scripts\python.exe -m pytest tests\test_vision_stack_runtime.py tests\test_vision_stack_detector.py tests\test_typesetting_layout.py tests\test_typesetting_renderer.py -k "points or dark_bubble or connected" -x -vv
```


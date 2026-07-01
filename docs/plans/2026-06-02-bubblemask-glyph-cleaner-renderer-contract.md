# BubbleMask Glyph Cleaner Renderer Contract Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make BubbleMask the real spatial contract for inpaint and renderer sizing while keeping glyph/text masks as the actual pixels to erase.

**Architecture:** The pipeline will separate three concepts: BubbleMask is the real balloon boundary, GlyphMask is the text cleanup target, and RendererFit uses the eroded BubbleMask interior for layout. Inpaint will never erase the whole balloon by default; it will erase accepted text components clipped to a safe BubbleMask interior, with a conservative white-balloon fill mode only when the balloon background is uniform.

**Tech Stack:** Python 3.12, OpenCV, NumPy, pytest, existing `pipeline/inpainter/notanother_adapter.py`, existing `pipeline/strip/process_bands.py`, existing `pipeline/typesetter/renderer.py`.

---

### Task 1: Baseline and Dirty Tree Safety

**Files:**
- Inspect: `pipeline/strip/process_bands.py`
- Inspect: `pipeline/inpainter/mask_builder.py`
- Inspect: `pipeline/inpainter/__init__.py`
- Inspect: `pipeline/inpainter/notanother_adapter.py`
- Inspect: `pipeline/typesetter/renderer.py`

**Step 1: Check status**

Run:

```powershell
git status --short
```

Expected: working tree may be dirty. Record unrelated files and do not revert them.

**Step 2: Run current focused tests**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_mask_builder.py pipeline\tests\test_inpaint_mask_geometry.py pipeline\tests\test_strip_process_bands.py -q
```

Expected: establish baseline. If failing, identify whether failures are from current local edits before proceeding.

**Step 3: Inspect current mask contract**

Run:

```powershell
rg -n "bubble_mask|bubble_mask_bbox|text_pixel_bbox|line_polygons|notanother|safe_text_box|render_bbox" pipeline\strip pipeline\inpainter pipeline\typesetter -g "*.py"
```

Expected: confirm the exact call chain before edits.

---

### Task 2: Define Explicit Mask Contract Helpers

**Files:**
- Modify: `pipeline/inpainter/mask_builder.py`
- Test: `pipeline/tests/test_mask_builder.py`

**Step 1: Write failing tests**

Add tests for:

```python
def test_final_text_cleanup_mask_clips_expanded_glyphs_to_eroded_bubble():
    # broad glyph expansion must not touch outline pixels
    # expected: final mask inside eroded BubbleMask only
    ...

def test_bubble_mask_is_not_used_as_full_erase_mask_by_default():
    # BubbleMask has large interior, glyph mask is small
    # expected: final cleanup pixels stay near glyphs, not whole balloon
    ...
```

**Step 2: Run tests to verify failure**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_mask_builder.py::test_final_text_cleanup_mask_clips_expanded_glyphs_to_eroded_bubble pipeline\tests\test_mask_builder.py::test_bubble_mask_is_not_used_as_full_erase_mask_by_default -q
```

Expected: FAIL because the explicit helper contract does not exist or current behavior is too broad.

**Step 3: Implement minimal helpers**

Add or refactor helpers in `pipeline/inpainter/mask_builder.py`:

```python
def safe_bubble_interior_mask(bubble_mask: np.ndarray, erode_px: int = 2) -> np.ndarray:
    """Return a binary BubbleMask interior that excludes outline/tail border pixels."""
    ...

def build_bubble_limited_glyph_mask(
    glyph_mask: np.ndarray,
    bubble_mask: np.ndarray,
    *,
    glyph_expand_px: int = 2,
    bubble_erode_px: int = 2,
) -> np.ndarray:
    """Expand glyphs, then clip them to the safe BubbleMask interior."""
    ...
```

Rules:
- Never return the whole BubbleMask unless explicitly requested by a later white-fill strategy.
- If BubbleMask is missing or empty, return empty mask and reason metadata, not a bbox substitute.
- Use ellipse kernels for small expansion/erosion.
- When `text_pixel_bbox` or accepted component glyph evidence exists, it wins over broad `line_polygons`; do not union a broad OCR line bbox back into the cleanup limit.
- A broad `line_polygons` bbox may be used as a search/support hint only after component filtering confirms glyph pixels inside the real BubbleMask.

**Step 4: Run focused tests**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_mask_builder.py -q
```

Expected: PASS.

---

### Task 3: Strengthen NotAnother-Style Component Filtering

**Files:**
- Modify: `pipeline/inpainter/notanother_adapter.py`
- Modify: `pipeline/inpainter/mask_builder.py`
- Test: `pipeline/tests/test_mask_builder.py`

**Step 1: Write failing tests**

Add tests for:

```python
def test_notanother_filter_rejects_components_outside_bubble():
    ...

def test_notanother_filter_rejects_components_without_ocr_support():
    ...

def test_notanother_filter_fills_glyph_holes_but_preserves_balloon_outline():
    ...
```

**Step 2: Run tests to verify failure**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_mask_builder.py -k "notanother or glyph_holes" -q
```

Expected: at least one FAIL if current thresholds/outline handling are insufficient.

**Step 3: Implement component acceptance contract**

Use `build_notanother_text_mask()` as the primary text cleanup refinement when all are available:
- `image_rgb`
- real `bubble_mask`
- OCR support mask from `text_pixel_bbox`, `line_polygons`, or source bbox

Rules:
- Reject component if centroid is outside eroded BubbleMask.
- Reject component if overlap with OCR support is too low.
- Reject tiny components unless they connect to accepted text.
- Fill holes only inside accepted glyph components.
- Final mask must be clipped to `safe_bubble_interior_mask()`.

**Step 4: Run focused tests**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_mask_builder.py pipeline\tests\test_inpaint_mask_geometry.py -q
```

Expected: PASS.

---

### Task 4: Add Conservative White-Balloon Interior Fill Mode

**Files:**
- Modify: `pipeline/inpainter/mask_builder.py`
- Modify: `pipeline/inpainter/__init__.py`
- Test: `pipeline/tests/test_inpaint_mask_geometry.py`

**Step 1: Write failing tests**

Add tests for:

```python
def test_uniform_white_balloon_can_fill_safe_interior_without_touching_outline():
    ...

def test_textured_or_nonuniform_balloon_does_not_use_full_interior_fill():
    ...
```

**Step 2: Run tests to verify failure**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_inpaint_mask_geometry.py -k "white_balloon or nonuniform" -q
```

Expected: FAIL until the strategy is implemented.

**Step 3: Implement strategy**

Add a helper that computes:
- background median inside `safe_bubble_interior_mask`
- color/stddev uniformity
- text mask coverage ratio
- whether the bubble is white/near-white

Rules:
- If uniform white balloon: allow safe interior cleanup/fill, but only inside eroded BubbleMask.
- If non-uniform, transparent, textured, or image-art area: use glyph-only mask.
- Never use raw `bubble_mask_bbox` to fill.
- Conservative interior fill requires a real BubbleMask (`bubble_mask_source="real"` or persisted mask layer/value). `bbox_fallback` must never enable full/interior fill.
- Record debug metadata:
  - `bubble_cleanup_strategy`
  - `bubble_uniformity_std`
  - `bubble_safe_interior_pixels`
  - `glyph_cleanup_pixels`

**Step 4: Run focused tests**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_inpaint_mask_geometry.py pipeline\tests\test_mask_builder.py -q
```

Expected: PASS.

---

### Task 5: Enforce BubbleMask as Renderer Layout Boundary

**Files:**
- Modify: `pipeline/typesetter/renderer.py`
- Test: `pipeline/tests/test_renderer_bubble_fit.py`

**Step 1: Write failing tests**

Create or extend renderer tests:

```python
def test_safe_text_box_is_derived_from_eroded_bubble_mask_when_available():
    ...

def test_renderer_uses_mask_area_not_tight_ocr_bbox_for_font_size():
    ...

def test_render_bbox_must_fit_inside_bubble_mask_interior():
    ...
```

**Step 2: Run tests to verify failure**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_renderer_bubble_fit.py -q
```

Expected: FAIL if renderer still prioritizes tight OCR bbox or rectangular `bubble_mask_bbox`.

**Step 3: Implement renderer boundary selection**

Update renderer target selection:
- Prefer eroded real `bubble_mask` interior loaded from `bubble_mask_path`/`bubble_mask_layer_path` + `bubble_mask_value`, or from a `bubble_inner_bbox` explicitly derived from that real mask.
- Derive `safe_text_box` from the largest usable interior component, not from rectangular `bubble_mask_bbox` alone.
- Use OCR bbox only as text anchor/center hint, not as the size limit.
- Keep `rotation_deg`.
- Validate final `render_bbox` against safe mask/bbox.

Rules:
- Do not use full page bbox.
- Do not use `bubble_mask_bbox` if real mask exists.
- Do not treat `bubble_mask_bbox` as proof of real mask. It is a bbox envelope only unless paired with real mask source/layer evidence.
- If mask exists but is degenerate, mark review/debug instead of shrinking text to unreadable size.

**Step 4: Run renderer tests**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_renderer_bubble_fit.py -q
```

Expected: PASS.

---

### Task 6: Ensure Strip Pipeline Carries Real BubbleMask to Inpaint and Renderer

**Files:**
- Modify: `pipeline/strip/process_bands.py`
- Test: `pipeline/tests/test_strip_process_bands.py`

**Step 1: Write failing tests**

Add tests for:

```python
def test_strip_text_block_keeps_real_bubble_mask_not_only_bbox():
    ...

def test_connected_balloon_blocks_keep_separate_bubble_ids_when_masks_are_separate():
    ...

def test_missing_bubble_mask_is_reported_not_silently_marked_ok():
    ...
```

**Step 2: Run tests to verify failure**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_strip_process_bands.py -k "bubble_mask or connected_balloon" -q
```

Expected: FAIL if current strip path only propagates bbox or overwrites real mask.

**Step 3: Implement propagation**

Rules:
- Every text block with a detected balloon must carry:
  - `bubble_mask`
  - `bubble_mask_bbox`
  - `bubble_id`
  - `bubble_mask_source`
  - `bubble_mask_value` after page-level persistence when available
  - `bubble_mask_layer_path` or `bubble_mask_path` after page-level persistence when available
- If only bbox exists, explicitly set `bubble_mask_source="bbox_fallback"` and do not mark it as real.
- Never let bbox fallback pretend to be real BubbleMask.
- If a block has a balloon bbox but real mask derivation fails, keep the bbox for preview/layout fallback but add metadata/QA so export/debug can see it was not a real BubbleMask success.

**Step 4: Run strip tests**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_strip_process_bands.py -q
```

Expected: PASS.

---

### Task 7: Add QA Flags for Mask Contract Violations

**Files:**
- Modify: `pipeline/vision_stack/runtime.py`
- Modify: `pipeline/inpainter/mask_builder.py`
- Test: `pipeline/tests/test_export_gate.py`
- Test: `pipeline/tests/test_vision_stack_runtime.py`

**Step 1: Write failing tests**

Add tests for:

```python
def test_bbox_fallback_bubble_mask_does_not_count_as_real_mask_success():
    ...

def test_render_uses_real_bubble_mask_or_reports_review_reason():
    ...

def test_mask_contract_violation_blocks_export_when_original_text_remains():
    ...
```

**Step 2: Run tests to verify failure**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_export_gate.py pipeline\tests\test_vision_stack_runtime.py -k "bubble_mask or mask_contract" -q
```

Expected: FAIL until QA recognizes the new contract.

**Step 3: Implement QA metadata**

Add clear debug/QA reasons:
- `missing_real_bubble_mask`
- `bbox_fallback_bubble_mask`
- `glyph_mask_outside_bubble`
- `render_outside_bubble_mask`
- `bubble_mask_degenerate`

Rules:
- `mask_density_high` must not block alone.
- Real visual failures still block export.
- Bbox fallback can be allowed for preview, but not reported as real BubbleMask success.

**Step 4: Run QA tests**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_export_gate.py pipeline\tests\test_vision_stack_runtime.py -q
```

Expected: PASS.

---

### Task 8: Visual Debug Outputs for Contract Inspection

**Files:**
- Modify: `pipeline/vision_stack/runtime.py`
- Modify: `pipeline/inpainter/mask_builder.py`

**Step 1: Add debug layers**

For debug runs, save overlays for:
- raw BubbleMask
- eroded BubbleMask interior
- OCR support mask
- accepted glyph cleanup mask
- final inpaint mask
- renderer safe text box
- render bbox

**Step 2: Run one page debug**

Run the existing one-page command used for One Second page with debug enabled.

Expected: debug folder contains separate images that make it obvious whether failure is from detect, mask, inpaint, or renderer.

**Step 3: Inspect target cases**

Check:
- `LET'S GO!!`
- `What! Then...`
- `SIM, NÃO FUNCIONA`
- `PROCURAR`
- connected balloons

Expected: final inpaint mask is inside eroded BubbleMask and text render box is inside safe text area.

---

### Task 9: Integration Validation on One Chapter

**Files:**
- No code changes unless failures point to a previous task.

**Step 1: Run focused unit tests**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_mask_builder.py pipeline\tests\test_inpaint_mask_geometry.py pipeline\tests\test_strip_process_bands.py pipeline\tests\test_renderer_bubble_fit.py pipeline\tests\test_export_gate.py -q
```

Expected: PASS.

**Step 2: Run One Second chapter 1 with debug**

Run the established full chapter command with debug output.

Expected:
- no eaten balloon outline
- no full-bubble erase unless uniform white mode explicitly selected
- less original text residue
- `safe_text_box` follows BubbleMask interior
- renderer text not tiny due to OCR bbox

**Step 3: Inspect outputs**

Inspect translated pages and debug overlays:
- page with jagged speech bubble
- page with UI/search panel
- page with connected balloons
- page with residual text in lower text strokes

**Step 4: Do not auto-correct visual style without user review**

Show the pages/debug outputs to the user and wait for pointed feedback.

---

### Task 10: Regression Guard Against SFX and Art Damage

**Files:**
- Modify: `pipeline/tests/test_mask_builder.py`
- Modify: `pipeline/tests/test_inpaint_mask_geometry.py`

**Step 1: Add tests**

Add tests for:

```python
def test_sfx_outside_bubble_is_not_cleaned_by_bubble_pipeline():
    ...

def test_art_text_without_real_bubble_mask_uses_review_or_glyph_only_path():
    ...

def test_korean_sfx_does_not_expand_cleanup_from_nearby_dialogue_bubble():
    ...
```

**Step 2: Run tests**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_mask_builder.py pipeline\tests\test_inpaint_mask_geometry.py -k "sfx or art_text or korean" -q
```

Expected: PASS.

**Step 3: Final focused validation**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_mask_builder.py pipeline\tests\test_inpaint_mask_geometry.py pipeline\tests\test_strip_process_bands.py pipeline\tests\test_renderer_bubble_fit.py pipeline\tests\test_export_gate.py pipeline\tests\test_vision_stack_runtime.py -q
```

Expected: PASS.

---

## Acceptance Criteria

- Inpaint does not erase using the whole BubbleMask by default.
- Final cleanup mask is `expanded glyph/text mask ∩ eroded real BubbleMask`.
- Uniform white balloon fill is allowed only when background uniformity passes.
- Bubble outline and tail are preserved.
- `safe_text_box` is derived from real BubbleMask interior when available.
- Renderer font size is not based on tight OCR bbox when BubbleMask exists.
- Bbox fallback is explicit and not treated as real BubbleMask success.
- Debug overlays show raw BubbleMask, safe interior, OCR support, glyph cleanup, final mask, safe text box, and render bbox.
- One Second chapter 1 visual run improves residual text without new outline damage.

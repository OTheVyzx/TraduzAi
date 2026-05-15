# Irregular Text Inpaint Mask Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make inpaint use an irregular text-shaped mask expanded by a small fixed radius, avoiding rectangular/white patches on translucent white balloons.

**Architecture:** Prefer real OCR line polygons and dark text pixels inside the OCR text area. Clip the final mask to the detected white balloon interior when available, with a small erosion to protect the black outline. Keep oval/octagon/bbox behavior only as fallback when no reliable text-shaped mask can be built.

**Tech Stack:** Python, OpenCV, NumPy, existing TraduzAi inpaint/runtime tests.

---

### Task 1: Add Text-Shaped Mask Helper

**Files:**
- Modify: `pipeline/inpainter/mask_builder.py`
- Test: `pipeline/tests/test_inpaint_mask_geometry.py`

**Step 1: Write failing tests**

Add tests for:
- `line_polygons` stay the first choice.
- Dark text pixels inside `text_pixel_bbox` create an irregular mask.
- The mask is dilated by exactly the configured radius, default 5 px.
- The corners of the source bbox stay empty when they do not contain text pixels.

**Step 2: Implement minimal helper**

Add a helper like:

```python
def text_pixel_mask_from_image(
    block: dict,
    image_rgb: np.ndarray,
    image_shape: tuple[int, ...],
    expand_px: int = 5,
) -> np.ndarray | None:
    ...
```

Rules:
- Use `line_polygons` first if present.
- Otherwise inspect only `text_pixel_bbox` or `bbox`.
- Detect likely text pixels by local contrast, not global black-only threshold.
- Dilate the resulting mask with an elliptical kernel of `expand_px`.
- Return `None` if the mask is too sparse or too huge.

**Step 3: Run focused tests**

Run from `N:\TraduzAI\TraduzAi\pipeline`:

```powershell
.\venv\Scripts\python.exe -m pytest .\tests\test_inpaint_mask_geometry.py -q
```

Expected: new tests pass.

---

### Task 2: Clip Mask To Balloon Interior

**Files:**
- Modify: `pipeline/inpainter/mask_builder.py`
- Modify: `pipeline/vision_stack/runtime.py`
- Test: `pipeline/tests/test_vision_stack_runtime.py`

**Step 1: Write failing tests**

Add tests for:
- White balloon interior clips the expanded text mask.
- Balloon outline pixels are not included.
- Translucent/gradient white balloon does not trigger solid white fill.

**Step 2: Integrate into `build_inpaint_mask()`**

Update `build_inpaint_mask()` order:

```text
line_polygons or image-derived text pixel mask
-> dilate 5 px
-> intersect with eroded balloon interior if available
-> fallback octagon/oval bbox only if no text mask exists
```

Important:
- Do not change saved `bbox`.
- Do not expand to full `balloon_bbox`.
- Keep `skip_processing` untouched.

**Step 3: Protect translucent balloons**

In the white-balloon heuristics, classify a balloon as translucent/textured if its interior has visible variance or gradient. For that case:
- skip fast white fill;
- skip metadata solid fill;
- require real inpaint/local reconstruction inside the irregular mask.

**Step 4: Run runtime tests**

```powershell
.\venv\Scripts\python.exe -m pytest .\tests\test_vision_stack_runtime.py .\tests\test_vision_stack_inpainter.py -q
```

Expected: pass.

---

### Task 3: Keep It Fast And Debuggable

**Files:**
- Modify: `pipeline/inpainter/__init__.py`
- Modify: `pipeline/vision_stack/runtime.py`
- Test: `pipeline/tests/test_strip_run.py`

**Step 1: Reuse the helper in strip fallback**

Replace local bbox-only masks with the shared helper when `image_rgb` is available. If image data is not available, keep octagon fallback.

**Step 2: Add debug outputs**

In existing debug folder, save:
- raw irregular text mask;
- expanded 5 px mask;
- clipped-to-balloon mask;
- overlay after clipping.

**Step 3: Run the compact regression set**

```powershell
.\venv\Scripts\python.exe -m pytest .\tests\test_inpaint_mask_geometry.py .\tests\test_mask_builder.py .\tests\test_vision_stack_inpainter.py .\tests\test_vision_stack_runtime.py .\tests\test_strip_run.py -q
```

Expected: pass.

---

### Expected Runtime Impact

Very low. OpenCV thresholding, connected components, and 5 px dilation over the small OCR bbox should add milliseconds per text region. The expensive part remains OCR and real inpaint.

### Recommendation

Implement Tasks 1 and 2 first. Only run a full chapter after the compact tests pass. The visual acceptance target is: no rectangular/white patch on translucent balloons, no eaten black outline, and fewer old-text dots.

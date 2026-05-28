# Koharu Evidence Solid Fill And Typeset Clamp Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make TraduzAI use Koharu-style validated text evidence across inpaint and typeset, and replace hard-coded white/black fast fill with safe sampled solid-color fill.

**Architecture:** Bboxes remain search/limit geometry, not erase/render truth. A reusable evidence layer records which source boxes actually contain glyph pixels; inpaint uses that evidence for safe solid fill, and typeset uses the same evidence to avoid rendering over broad OCR blocks that include art/face regions.

**Tech Stack:** Python 3.12, OpenCV, NumPy, existing strip pipeline, `pipeline/inpainter`, `pipeline/vision_stack/runtime.py`, `pipeline/layout`, `pipeline/typesetter`, pytest, real `DEBUGM` chapter runs.

---

## Rules For This Plan

- Do not make a broad bbox become an erase mask.
- Do not make a broad bbox become a typeset target when a validated source bbox exists.
- Fast fill is allowed only when all gates pass: text evidence exists, fill region is solid, and sampled color is stable.
- Textured, gradient, translucent, patterned, noisy, or art-heavy regions must fall back to real inpaint.
- Every new heuristic must emit debug metadata so future visual bugs are traceable.

---

## Task 1: Add A Shared Validated Evidence Contract

**Files:**
- Modify: `pipeline/inpainter/mask_builder.py`
- Modify: `pipeline/vision_stack/runtime.py`
- Test: `pipeline/tests/test_inpaint_mask_geometry.py`

**Step 1: Write failing tests**

Add tests for a no-line broad OCR block with two source boxes:
- top source box contains real glyph evidence;
- lower source box contains face/art-like components;
- result records only the top source box as validated.

Expected fields on the block:

```python
block["_validated_text_source_bboxes"] == [[408, 32, 1199, 332]]
block["_raw_text_evidence_bbox"] == [approx top glyph bbox]
block["qa_metrics"]["raw_text_evidence"]["accepted_source_box_count"] == 1
```

Run:

```powershell
cd N:\TraduzAI\pipeline
python -m pytest tests\test_inpaint_mask_geometry.py::test_raw_text_mask_records_validated_source_boxes -q
```

Expected: FAIL because metadata does not exist yet.

**Step 2: Implement minimal metadata**

In `build_raw_text_mask_from_image(...)`, whenever a source candidate passes `_reject_raw_text_mask_without_koharu_evidence(...)`, append that candidate bbox to `_validated_text_source_bboxes`.

Also record:
- `_raw_text_evidence_bbox`
- `_raw_text_evidence_pixels`
- `qa_metrics.raw_text_evidence.accepted_source_box_count`
- `qa_metrics.raw_text_evidence.rejected_source_box_count`

Keep this metadata internal and defensive. Do not require downstream callers to have it.

**Step 3: Preserve rejection behavior**

In `vision_blocks_to_mask(...)`, keep the current rule:

```python
if raw_text_mask_rejected:
    continue
```

Do not re-enable bbox fallback after evidence rejection.

**Step 4: Run tests**

```powershell
cd N:\TraduzAI\pipeline
python -m pytest tests\test_inpaint_mask_geometry.py -q
```

Expected: PASS.

---

## Task 2: Implement Safe Solid-Color Sampling

**Files:**
- Modify: `pipeline/inpainter/__init__.py`
- Test: `pipeline/tests/test_vision_stack_inpainter.py`

**Step 1: Write failing tests**

Add tests for:

1. Off-white balloon: fast fill samples off-white, not pure white.
2. Gray/blue solid box: fast fill samples local solid color.
3. Textured background: sampler rejects and leaves block for real inpaint.
4. Gradient/translucent region: sampler rejects.
5. Black solid panel: sampler returns black/dark fill and does not create white artifacts.

Run one targeted test first:

```powershell
cd N:\TraduzAI\pipeline
python -m pytest tests\test_vision_stack_inpainter.py::VisionStackInpainterTests::test_fast_solid_fill_samples_off_white_background -q
```

Expected: FAIL because helper does not exist.

**Step 2: Add helper**

Add a helper near `_fill_mask_solid(...)`:

```python
def _sample_solid_fill_color(
    image_rgb: np.ndarray,
    *,
    source_mask: np.ndarray,
    text_mask: np.ndarray,
    limit_mask: np.ndarray,
) -> tuple[tuple[int, int, int] | None, dict]:
    ...
```

Sampling rules:
- sample only inside `limit_mask`;
- exclude `text_mask` expanded by 8-12 px;
- exclude dark outline/edge pixels when possible;
- require enough pixels, for example `sample_count >= 64`;
- compute median RGB;
- compute per-channel std or MAD;
- accept only when local variance is low enough;
- reject if strong gradient or high saturation variance is detected.

Return:

```python
((r, g, b), {"reason": "solid_color", "std": [...], "sample_count": n})
```

or:

```python
(None, {"reason": "textured_or_gradient", ...})
```

**Step 3: Replace hard-coded white where safe**

In `_apply_fast_white_balloon_fill(...)`, rename behavior internally to solid fill without changing public env names yet.

Use:

```python
color, color_stats = _sample_solid_fill_color(...)
if color is None:
    _reject(color_stats["reason"])
    continue
filled_from_original = _fill_mask_solid(band_rgb, text_fill_mask, color=color)
```

Keep white as fallback only for confirmed white balloons where the sampler cannot find enough safe pixels but the old white gate already passed.

**Step 4: Add telemetry**

Record on `ocr_page`:

- `_strip_fast_solid_fill_count`
- `_strip_fast_solid_fill_sampled_colors`
- `_strip_fast_solid_fill_rejection_reasons`
- `_strip_fast_solid_fill_variance_stats`

Keep existing `_strip_used_fast_white_fill` for backward compatibility until UI/debug readers are migrated.

**Step 5: Run tests**

```powershell
cd N:\TraduzAI\pipeline
python -m pytest tests\test_vision_stack_inpainter.py -q
```

Expected: PASS.

---

## Task 3: Use Validated Evidence To Clamp Typeset Target

**Files:**
- Modify: `pipeline/layout/balloon_layout.py`
- Modify: `pipeline/typesetter/renderer.py`
- Possibly modify: `pipeline/strip/process_bands.py`
- Test: `pipeline/tests/test_typesetting_layout.py`
- Test: `pipeline/tests/test_typesetting_renderer.py`

**Step 1: Write failing regression test**

Use a broad no-line narration block shaped like the Monster Actor page:

```python
block = {
    "bbox": [312, 32, 1326, 883],
    "text_pixel_bbox": [400, 46, 1200, 883],
    "_validated_text_source_bboxes": [[408, 32, 1199, 332]],
    "tipo": "narracao",
    "balloon_type": "white",
    "block_profile": "white_balloon",
}
```

Expected:
- target/safe box should be derived from `[408, 32, 1199, 332]`, not from `[312, 32, 1326, 883]`;
- render bbox must not overlap lower face/art zone;
- if translation does not fit, reduce font/wrap inside that validated region or flag QA; never expand back to face/art bbox.

Run:

```powershell
cd N:\TraduzAI\pipeline
python -m pytest tests\test_typesetting_layout.py::test_broad_narration_uses_validated_source_bbox_for_target -q
```

Expected: FAIL before implementation.

**Step 2: Add layout source priority**

In layout code, define target priority for broad no-line blocks:

1. `connected_position_bboxes`, if connected-balloon logic explicitly owns it;
2. `_validated_text_source_bboxes`, when present and not empty;
3. `layout_safe_bbox`;
4. `balloon_bbox`;
5. `text_pixel_bbox`/`bbox`.

Do not apply this clamp to regular speech bubbles with valid line polygons.

**Step 3: Add render safety gate**

In `renderer.py`, before final render:

- if block has `_validated_text_source_bboxes`;
- and `render_bbox` extends outside validated source region by more than a small margin;
- mark candidate as invalid and retry smaller font/wrap;
- if still impossible, emit QA flag instead of rendering over art.

Suggested QA flags:

- `render_outside_validated_text_source`
- `validated_source_too_small_for_translation`

**Step 4: Preserve connected-balloon behavior**

For connected lobes, do not blindly clamp to one source bbox. Only use validated source bboxes if they map to individual lobes or if the existing connected split cannot produce a safe target.

Run:

```powershell
cd N:\TraduzAI\pipeline
python -m pytest tests\test_typesetting_layout.py tests\test_typesetting_renderer.py -q
```

Expected: PASS.

---

## Task 4: Debug Artifacts And Traceability

**Files:**
- Modify: `pipeline/debug_tools/masks.py` if active in current debug path
- Modify: `pipeline/inpainter/__init__.py`
- Modify: `pipeline/typesetter/renderer.py`
- Test: focused existing debug tests if present

**Step 1: Extend inpaint debug metadata**

Add to `debug_inpaint/.../metadata.json`:

- `validated_text_source_bboxes`
- `raw_text_evidence_bbox`
- `fast_solid_fill_color`
- `fast_solid_fill_sample_count`
- `fast_solid_fill_reject_reason`

**Step 2: Extend E2E mask decision**

Add to `debug/e2e/06_mask_segmentation/.../mask_decision.json`:

- accepted/rejected source boxes;
- reason for each rejected source box when available;
- evidence bbox.

**Step 3: Extend render plan**

Add to `render_plan_final.jsonl`:

- `validated_text_source_bboxes`
- `target_source`: one of `connected_lobe`, `validated_text_source`, `layout_safe_bbox`, `balloon_bbox`, `fallback_bbox`;
- `render_validated_containment`.

**Step 4: Run debug tests**

```powershell
cd N:\TraduzAI\pipeline
python -m pytest tests\test_mask_chain_debug.py tests\test_render_plan_trace_integrity.py -q
```

Expected: PASS, or skip unavailable tests with note if this checkout lacks them.

---

## Task 5: Real Chapter Validation Matrix

**Files:**
- No source edits unless validation exposes regressions.

**Step 1: Run Monster Actor Ch75**

Use the same source config as the current regression:

```powershell
cd N:\TraduzAI
$env:TRADUZAI_SKIP_LOCAL_VENV_REEXEC='1'
$env:TRADUZAI_STRIP_PARALLEL_INPAINT_THREADS='3'
$env:TRADUZAI_STRIP_FAST_WHITE_INPAINT='1'
$env:TRADUZAI_STRIP_FAST_WHITE_NARRATION='1'
$env:TRADUZAI_STRIP_FAST_DARK_PANEL_FILL='1'
$env:TRADUZAI_PAGE_CLEANUP_RERENDER='0'
$env:TRADUZAI_PADDLE_FULL_PAGE='1'
$env:TRADUZAI_STRIP_DETECT_FULL_PAGE='1'
$env:TRADUZAI_GOOGLE_PARALLEL_CHUNKS='1'
$env:TRADUZAI_GOOGLE_TRANSLATE_WORKERS='3'
python pipeline\main.py <new-run>\runner_config.json
```

Validate:

- `page_002_band_005` face zone changed pixels = 0;
- translated page 002 no longer renders the huge top text over face/art;
- `debug_inpaint` shows only validated upper source box changed;
- `render_plan_final.jsonl` says target source is `validated_text_source`.

**Step 2: Run previous problem chapters**

Run at least:

- `C:\Users\PICHAU\Downloads\Articuno (comick)_Ch. 61 OFFICIAL TRANSLATION`
- `C:\Users\PICHAU\Downloads\Chapter 1`
- `C:\Users\PICHAU\Downloads\Chapter 39`
- `D:\Mihon pra pc\downloads\mangas\Manhwatop (EN)\The God of Death\Chapter 2.cbz`
- one or two chapters from `D:\Mihon pra pc\downloads\mangas` with black/colored boxes if available.

Validate:

- no face/art fast fill;
- no white fill over colored solid panel;
- black panels keep black/dark background;
- text render stays inside validated source/balloon interior;
- no connected-balloon regression.

**Step 3: Timing comparison**

Compare against latest v5 timing:

- total: about `128.98s`;
- `strip_process_bands_total`: about `73.20s`;
- `sync_inpaint_images`: about `9.94s`;
- `page_cleanup_rerender`: about `19.55s`.

Accept small overhead if visual quality improves. Reject a broad slowdown if it only comes from debug/logging.

---

## Task 6: Default/Rollout Decision

**Files:**
- Modify only after validation: runtime defaults/profile files where current global defaults live.
- Test: `pipeline/tests/test_runtime_profiles.py`

**Step 1: Keep feature behind existing fast flags during validation**

Do not silently enable sampled solid fill for all runs until the real matrix passes.

**Step 2: Promote after matrix passes**

If validation passes:

- keep `TRADUZAI_STRIP_FAST_WHITE_INPAINT` as compatibility alias;
- internally report it as solid fill;
- document that fast fill now supports sampled solid backgrounds, not only white.

**Step 3: Runtime profile test**

Run:

```powershell
cd N:\TraduzAI\pipeline
python -m pytest tests\test_runtime_profiles.py tests\test_engine_presets.py -q
```

Expected: PASS.

---

## Final Verification Commands

```powershell
cd N:\TraduzAI
python -m py_compile pipeline\inpainter\__init__.py pipeline\inpainter\mask_builder.py pipeline\vision_stack\runtime.py pipeline\layout\balloon_layout.py pipeline\typesetter\renderer.py

cd N:\TraduzAI\pipeline
python -m pytest tests\test_inpaint_mask_geometry.py tests\test_vision_stack_inpainter.py -q
python -m pytest tests\test_typesetting_layout.py tests\test_typesetting_renderer.py -q
python -m pytest tests\test_mask_chain_debug.py tests\test_render_plan_trace_integrity.py -q
```

Expected:

- all focused tests pass;
- Monster Actor page 002 no face-zone changes from inpaint;
- translated page 002 no huge overmerged text over face;
- colored/black solid boxes use local sampled color or fall back safely;
- debug artifacts explain every fast fill and every clamp decision.

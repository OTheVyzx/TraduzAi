# Koharu Conflict Remediation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Finish the parts of `2026-05-23-koharu-style-preset-aware-pipeline-plan.md` that were only partially implemented or are conflicting with the automatic app pipeline.

**Architecture:** Keep detector choice preset-driven, but make post-detection cleanup deterministic: segment evidence authorizes masks, sampled solid fill is the only default fast fill, legacy white/black fast paths are opt-in, and debug/QA metadata must show which rule actually touched each block. Avoid changing OCR/detect model selection while correcting inpaint, runtime defaults, and render safety.

**Tech Stack:** Python 3.12, OpenCV, NumPy, pytest, TraduzAI `pipeline/`, Tauri runtime profile config.

---

## Current Audit Baseline

The following items are confirmed in the live checkout:

- Preset metadata is implemented in `N:/TraduzAI/pipeline/vision_stack/runtime.py`.
- Segment evidence metadata is implemented in `N:/TraduzAI/pipeline/inpainter/mask_builder.py`.
- OCR reconciliation clamps broad OCR to validated source boxes, but does not split multiple validated sources.
- Typeset uses `_validated_text_source_bboxes` and records render containment.
- Pipeline artifacts are exported, but `Inpainted` and `FinalRender` remain mostly `pending`.
- `fast_solid` is not an independent stage. It is still embedded in `_apply_fast_white_balloon_fill`.
- `TRADUZAI_STRIP_FAST_WHITE_INPAINT=1` and `TRADUZAI_STRIP_FAST_DARK_PANEL_FILL=1` are forced by `runtime_profiles.py`, so the app path still enables legacy white/dark behavior.
- `_apply_connected_white_geometry_fill` still writes pure white with `result[fill_mask > 0] = 255`.
- No stable `bubble_id` / `bubble_mask_bbox` / `bubble_inner_bbox` propagation exists yet.

---

## Task 1: Lock Desired Runtime Defaults In Tests

**Files:**
- Modify: `N:/TraduzAI/pipeline/tests/test_runtime_profiles.py`
- Modify: `N:/TraduzAI/pipeline/runtime_profiles.py`

**Step 1: Write failing tests**

Change the balanced/performance expectations to the target default:

```python
def test_runtime_profile_defaults_to_solid_fill_only_for_fast_fill():
    decision = resolve_runtime_profile({})
    assert decision.env_defaults["TRADUZAI_STRIP_FAST_SOLID_INPAINT"] == "1"
    assert decision.env_defaults["TRADUZAI_STRIP_FAST_WHITE_INPAINT"] == "0"
    assert decision.env_defaults["TRADUZAI_STRIP_FAST_WHITE_NARRATION"] == "0"
    assert decision.env_defaults["TRADUZAI_STRIP_FAST_DARK_PANEL_FILL"] == "0"
    assert "TRADUZAI_STRIP_FAST_LOCAL_INPAINT" not in decision.env_defaults
```

Also keep the already desired defaults:

```python
assert decision.env_defaults["TRADUZAI_STRIP_PARALLEL_INPAINT_THREADS"] == "3"
assert decision.env_defaults["TRADUZAI_PAGE_CLEANUP_RERENDER"] == "0"
assert decision.env_defaults["TRADUZAI_PADDLE_FULL_PAGE"] == "1"
assert decision.env_defaults["TRADUZAI_STRIP_DETECT_FULL_PAGE"] == "1"
assert decision.env_defaults["TRADUZAI_GOOGLE_PARALLEL_CHUNKS"] == "1"
assert decision.env_defaults["TRADUZAI_GOOGLE_TRANSLATE_WORKERS"] == "3"
```

**Step 2: Run test to verify it fails**

Run:

```powershell
cd N:/TraduzAI
$env:PYTHONPATH='N:/TraduzAI;N:/TraduzAI/pipeline'
N:/TraduzAI/pipeline/venv/Scripts/python.exe -m pytest N:/TraduzAI/pipeline/tests/test_runtime_profiles.py -q
```

Expected: FAIL because current `runtime_profiles.py` still sets fast white/dark to `1`.

**Step 3: Implement minimal runtime default change**

In `N:/TraduzAI/pipeline/runtime_profiles.py`, change `automatic_pipeline_env` to:

```python
automatic_pipeline_env = {
    "TRADUZAI_STRIP_SCHEDULER_EXECUTOR": "overlap_context_release",
    "TRADUZAI_STRIP_PARALLEL_INPAINT_THREADS": "3",
    "TRADUZAI_STRIP_FAST_SOLID_INPAINT": "1",
    "TRADUZAI_STRIP_FAST_WHITE_INPAINT": "0",
    "TRADUZAI_STRIP_FAST_WHITE_NARRATION": "0",
    "TRADUZAI_STRIP_FAST_DARK_PANEL_FILL": "0",
    "TRADUZAI_PAGE_CLEANUP_RERENDER": "0",
    "TRADUZAI_PADDLE_FULL_PAGE": "1",
    "TRADUZAI_STRIP_DETECT_FULL_PAGE": "1",
    "TRADUZAI_GOOGLE_PARALLEL_CHUNKS": "1",
    "TRADUZAI_GOOGLE_TRANSLATE_WORKERS": "3",
}
```

**Step 4: Re-run tests**

Expected: `test_runtime_profiles.py` passes.

---

## Task 2: Extract Independent Fast Solid Fill Stage

**Files:**
- Modify: `N:/TraduzAI/pipeline/inpainter/__init__.py`
- Modify: `N:/TraduzAI/pipeline/tests/test_vision_stack_inpainter.py`
- Modify if needed: `N:/TraduzAI/pipeline/tests/test_mask_builder.py`

**Step 1: Write failing tests**

Add tests for the new public internal function:

```python
def test_fast_solid_fill_runs_when_fast_white_is_disabled():
    with patch.dict(
        "os.environ",
        {
            "TRADUZAI_STRIP_FAST_SOLID_INPAINT": "1",
            "TRADUZAI_STRIP_FAST_WHITE_INPAINT": "0",
            "TRADUZAI_STRIP_FAST_DARK_PANEL_FILL": "0",
        },
        clear=False,
    ):
        result, remaining, stats = _apply_fast_solid_balloon_fill(image, page, vision_blocks)
    assert stats["solid_balloon_count"] == 1
    assert page["_strip_used_fast_solid_fill"] is True
```

Add three sample cases:

- white balloon fills with sampled white;
- black/dark solid panel fills with sampled black;
- light blue balloon fills with sampled light blue, not white.

Add one rejection case:

- gradient/textured/translucent background is rejected and leaves the block for real inpaint.

**Step 2: Run test to verify it fails**

Run:

```powershell
cd N:/TraduzAI/pipeline
N:/TraduzAI/pipeline/venv/Scripts/python.exe -m pytest tests/test_vision_stack_inpainter.py -k "fast_solid" -q
```

Expected: FAIL because `_apply_fast_solid_balloon_fill` does not exist.

**Step 3: Add feature flag**

In `N:/TraduzAI/pipeline/inpainter/__init__.py`, add:

```python
def _fast_solid_balloon_fill_enabled() -> bool:
    flag = os.getenv("TRADUZAI_STRIP_FAST_SOLID_INPAINT", "1").strip().lower()
    return flag in {"1", "true", "yes", "on"}
```

**Step 4: Extract the generic sampled-fill logic**

Create:

```python
def _apply_fast_solid_balloon_fill(
    band_rgb: np.ndarray,
    ocr_page: dict,
    vision_blocks: list[dict],
) -> tuple[np.ndarray, list[dict], dict]:
    ...
```

Implementation requirements:

- Use text geometry or validated source evidence to build `text_fill_mask`.
- Clamp fill to balloon/safe region when available.
- Use `_sample_solid_fill_color_for_mask`.
- Never fallback to `(255, 255, 255)` when sampling fails.
- Reject high variance, gradient, translucent, textured, or uncertain regions.
- Remove covered `vision_blocks` only when changed pixels cover the intended text block.
- Record:

```python
ocr_page["_strip_used_fast_solid_fill"] = True
ocr_page["_strip_fast_solid_balloon_count"] = count
ocr_page["_strip_fast_solid_fill_samples"] = samples
ocr_page["_strip_fast_solid_rejection_reasons"] = rejection_reasons
```

**Step 5: Keep `_apply_fast_white_balloon_fill` legacy-only**

Do not delete it yet. It remains available only when `TRADUZAI_STRIP_FAST_WHITE_INPAINT=1`.

**Step 6: Re-run focused tests**

Expected: solid tests pass without enabling fast white or fast dark.

---

## Task 3: Reorder `apply_inpaint` Around Solid Fill

**Files:**
- Modify: `N:/TraduzAI/pipeline/inpainter/__init__.py`
- Test: `N:/TraduzAI/pipeline/tests/test_vision_stack_inpainter.py`

**Step 1: Write failing flow test**

Assert the default flow calls solid first and does not call legacy white/dark when their env flags are `0`:

```python
with patch("inpainter._apply_fast_solid_balloon_fill", return_value=(working, [], {})) as solid, \
     patch("inpainter._apply_fast_white_balloon_fill") as white, \
     patch("inpainter._apply_fast_dark_panel_text_fill") as dark:
    apply_inpaint(image, page)

solid.assert_called_once()
white.assert_not_called()
dark.assert_not_called()
```

**Step 2: Update flow**

Change the order around `apply_inpaint`:

```text
solid fast fill
connected geometry sampled fill
legacy fast white only if explicitly enabled
legacy fast dark only if explicitly enabled
fast local only if explicitly enabled
real inpaint for remaining blocks
post cleanup if enabled and safe
```

**Step 3: Add metadata reset**

Initialize:

```python
ocr_page["_strip_used_fast_solid_fill"] = False
ocr_page["_strip_used_fast_white_fill"] = False
ocr_page["_strip_used_fast_dark_fill"] = False
ocr_page["_strip_used_fast_local_fill"] = False
```

**Step 4: Re-run flow tests**

Expected: default path uses only solid + real inpaint fallback.

---

## Task 4: Replace Hardcoded White In Connected Geometry Fill

**Files:**
- Modify: `N:/TraduzAI/pipeline/inpainter/__init__.py`
- Modify: `N:/TraduzAI/pipeline/tests/test_vision_stack_inpainter.py`

**Step 1: Write failing tests**

Add a connected balloon with light blue flat fill:

```python
result, remaining, stats = _apply_connected_solid_geometry_fill(image, page, vision_blocks)
sample = result[text_y, text_x].tolist()
assert abs(sample[2] - original_blue[2]) < 8
assert sample != [255, 255, 255]
```

Add a white connected balloon test proving outline is preserved.

**Step 2: Run test to verify it fails**

Expected: FAIL because current `_apply_connected_white_geometry_fill` writes pure white.

**Step 3: Implement sampled connected fill**

Either rename or wrap:

```python
def _apply_connected_solid_geometry_fill(...):
    fill_color, metadata = _sample_solid_fill_color_for_mask(...)
    if not metadata.get("accepted"):
        return band_rgb, vision_blocks, {"connected_solid_count": 0, ...}
    result[fill_mask > 0] = fill_color
```

Rules:

- Use connected geometry mask only for text pixels, not outline.
- Reject if sampled region is not solid.
- Keep legacy `_apply_connected_white_geometry_fill` only behind `TRADUZAI_STRIP_FAST_WHITE_INPAINT=1`.

**Step 4: Re-run tests**

Expected: connected light-colored balloon no longer gets white patches.

---

## Task 5: Retire Fast Dark As Default And Migrate Its Tests

**Files:**
- Modify: `N:/TraduzAI/pipeline/inpainter/__init__.py`
- Modify: `N:/TraduzAI/pipeline/tests/test_vision_stack_inpainter.py`

**Step 1: Split test intent**

Replace default fast-dark expectations with two categories:

1. Default path:

```python
with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_DARK_PANEL_FILL": "0"}, clear=False):
    ...
assert page.get("_strip_used_dark_panel_fill") is not True
assert remaining
```

2. Solid path:

```python
with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_SOLID_INPAINT": "1"}, clear=False):
    result, remaining, stats = _apply_fast_solid_balloon_fill(...)
assert stats["solid_black_count"] == 1
```

**Step 2: Keep explicit legacy coverage**

If keeping `_apply_fast_dark_panel_text_fill`, add one opt-in test:

```python
with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_DARK_PANEL_FILL": "1"}, clear=False):
    ...
```

If it still fails, mark it as legacy broken and do not let it block the default solid migration.

**Step 3: Re-run tests**

Run:

```powershell
cd N:/TraduzAI/pipeline
N:/TraduzAI/pipeline/venv/Scripts/python.exe -m pytest tests/test_vision_stack_inpainter.py -k "fast_dark or fast_solid" -q
```

Expected: default behavior tests pass; legacy opt-in is either fixed or explicitly isolated.

---

## Task 6: Update Strip Metrics For Fast Solid

**Files:**
- Modify: `N:/TraduzAI/pipeline/strip/process_bands.py`
- Modify: `N:/TraduzAI/pipeline/strip/run.py`
- Modify: `N:/TraduzAI/pipeline/tests/test_strip_process_bands.py`
- Modify: `N:/TraduzAI/pipeline/tests/test_strip_run.py`
- Modify: `N:/TraduzAI/pipeline/tests/test_strip_run_metadata.py`

**Step 1: Write failing metrics tests**

Assert band and chapter summaries include:

```python
fast_solid_balloon_count
fast_solid_band_count
fast_solid_rejection_reasons
fast_solid_fill_samples
fast_solid_white_count
fast_solid_black_count
fast_solid_colored_count
```

**Step 2: Aggregate metadata**

In `process_bands.py`, copy `_strip_fast_solid_*` from translated page to band perf.

In `strip/run.py`, summarize solid counts beside existing white/local counts.

**Step 3: Keep legacy metrics**

Do not delete `fast_white_*` and `fast_dark_*` metrics. They must still be visible when explicitly enabled.

**Step 4: Re-run strip tests**

Expected: metrics distinguish solid from legacy white/dark.

---

## Task 7: Complete Pipeline Artifact Status

**Files:**
- Modify: `N:/TraduzAI/pipeline/vision_stack/runtime.py`
- Modify: `N:/TraduzAI/pipeline/strip/process_bands.py`
- Modify: `N:/TraduzAI/pipeline/strip/run.py`
- Modify: `N:/TraduzAI/pipeline/typesetter/renderer.py` only if final render status is easiest there
- Test: `N:/TraduzAI/pipeline/tests/test_strip_run_metadata.py`
- Test: `N:/TraduzAI/pipeline/tests/test_render_plan_trace_integrity.py`

**Step 1: Write failing artifact test**

Assert no completed page leaves the final statuses as `pending`:

```python
assert artifacts["Inpainted"]["status"] in {"ok", "skipped", "failed"}
assert artifacts["FinalRender"]["status"] in {"ok", "skipped", "failed"}
```

**Step 2: Mark stages explicitly**

After inpaint:

```python
artifacts["Inpainted"] = {
    "producer": preset.inpainter,
    "status": "ok" if changed_or_no_blocks else "skipped",
    "strategy": "fast_solid" or "real_inpaint" or "mixed",
}
```

After typeset:

```python
artifacts["FinalRender"] = {
    "producer": "traduzai-typesetter",
    "status": "ok",
    "rendered_text_count": n,
}
```

**Step 3: Export unchanged debug path**

Keep writing:

```text
debug/e2e/00_run/pipeline_artifacts.json
```

**Step 4: Re-run debug/trace tests**

Expected: trace is complete and not misleading.

---

## Task 8: Finish OCR Split For Multiple Validated Sources

**Files:**
- Modify: `N:/TraduzAI/pipeline/vision_stack/runtime.py`
- Test: `N:/TraduzAI/pipeline/tests/test_vision_stack_runtime.py`
- Test if present: `N:/TraduzAI/pipeline/tests/test_vision_stack_ocr.py`

**Step 1: Write failing split test**

Create one broad OCR text and two validated source boxes far apart. Expected:

- either split into two text items when source text strings can be mapped;
- or keep one item but force `_render_target_source="validated_text_source"` and set `ocr_multiple_validated_sources`.

**Step 2: Implement conservative split**

Start conservative:

- If source strings/line polygons identify per-box text, split.
- If not, do not invent text split. Keep one item, clamp geometry, and flag QA.

**Step 3: Re-run runtime tests**

Expected: no broad OCR block can render over face/art after validated source evidence exists.

---

## Task 9: Add Bubble ID / Bubble Mask Safe Areas

**Files:**
- Modify: `N:/TraduzAI/pipeline/strip/detect_balloons.py`
- Modify: `N:/TraduzAI/pipeline/layout/balloon_layout.py`
- Modify: `N:/TraduzAI/pipeline/typesetter/renderer.py`
- Test: `N:/TraduzAI/pipeline/tests/test_layout_analysis.py`
- Test: `N:/TraduzAI/pipeline/tests/test_strip_balloon_bbox_propagation.py`
- Test: `N:/TraduzAI/pipeline/tests/test_typesetting_renderer.py`

**Step 1: Write failing propagation test**

Expected metadata:

```python
text["bubble_id"]
text["bubble_mask_bbox"]
text["bubble_inner_bbox"]
text["connected_lobe_ids"]
```

**Step 2: Generate stable IDs**

Use deterministic IDs per page/band:

```text
page_003_band_012_bubble_001
```

Do not use random UUIDs.

**Step 3: Propagate through layout**

Carry bubble fields from detector/strip to layout to renderer.

**Step 4: Use bubble mask for safe area**

Renderer priority becomes:

1. manual edited box;
2. connected lobe/bubble ID safe area;
3. validated text source;
4. layout safe bbox;
5. balloon bbox;
6. fallback bbox.

**Step 5: Re-run layout/render tests**

Expected: connected balloons and overlapping balloons use the matching bubble/lobe safe area, not a broad page bbox.

---

## Task 10: Real Pipeline Validation Matrix

**Files:**
- No source edit unless validation fails.
- Read outputs under `N:/TraduzAI/DEBUGM/runs/`.

**Step 1: Run focused unit suite**

```powershell
cd N:/TraduzAI
$env:PYTHONPATH='N:/TraduzAI;N:/TraduzAI/pipeline'
N:/TraduzAI/pipeline/venv/Scripts/python.exe -m py_compile `
  N:/TraduzAI/pipeline/runtime_profiles.py `
  N:/TraduzAI/pipeline/inpainter/__init__.py `
  N:/TraduzAI/pipeline/inpainter/mask_builder.py `
  N:/TraduzAI/pipeline/vision_stack/runtime.py `
  N:/TraduzAI/pipeline/layout/balloon_layout.py `
  N:/TraduzAI/pipeline/typesetter/renderer.py `
  N:/TraduzAI/pipeline/strip/process_bands.py `
  N:/TraduzAI/pipeline/strip/run.py

N:/TraduzAI/pipeline/venv/Scripts/python.exe -m pytest `
  N:/TraduzAI/pipeline/tests/test_runtime_profiles.py `
  N:/TraduzAI/pipeline/tests/test_inpaint_mask_geometry.py `
  N:/TraduzAI/pipeline/tests/test_mask_builder.py `
  N:/TraduzAI/pipeline/tests/test_vision_stack_inpainter.py `
  N:/TraduzAI/pipeline/tests/test_typesetting_layout.py `
  N:/TraduzAI/pipeline/tests/test_typesetting_renderer.py `
  N:/TraduzAI/pipeline/tests/test_strip_run_metadata.py `
  -x
```

**Step 2: Run app-shaped normal pipeline**

Use normal `pipeline/main.py` config flow with runtime profile applied. Validate at least:

- `C:/Users/PICHAU/Downloads/Articuno (comick)_Ch. 61 OFFICIAL TRANSLATION`
- Monster Actor Ch75 source used in the previous debug
- `C:/Users/PICHAU/Downloads/Chapter 1`
- one chapter from `D:/Mihon pra pc/downloads/mangas/Manhwatop (EN)/1 Second`

**Step 3: Inspect visual cases**

Required visual checks:

- blue/light-colored balloons have no white patch;
- black panels do not retain English and do not get black bars;
- connected balloons keep lobe-local text placement;
- face/art false positives are not inpainted;
- text is not rendered over border/outline;
- textured balloons go to real inpaint or safe sampled solid only when truly flat.

**Step 4: Inspect QA and metrics**

Expected:

- `qa_report.json`: `critical_issue_count=0`;
- no new `text_residual_after_inpaint` critical;
- `fast_solid_balloon_count > 0` on flat balloons;
- legacy `fast_white_balloon_count == 0` by default;
- legacy `fast_dark_panel_fill_count == 0` by default;
- `pipeline_artifacts.json` has final statuses not `pending`.

**Execution status 2026-05-24**

- Task 8: completed with focused runtime coverage for split/clamp of multiple validated sources.
- Task 9: completed with bubble/lobe metadata propagation and renderer safe-area coverage.
- Task 10: completed through the requested validation boundary. Focused suite passed with `631 passed, 20 skipped`.
- Real validation root: `N:/TraduzAI/DEBUGM/runs/2026-05-24_task10_real_validation_20260524_174315`.
- Real validation matrix after rerun:
  - `articuno_ch61_sample12`: QA `PASS`, critical `0`.
  - `chapter1_full`: QA `PASS`, critical `0`.
  - `monster_actor_ch75_sample8`: QA `PASS`, critical `0`.
  - `one_second_ch1_sample8`: QA `PASS`, critical `0`.
- The `chapter1_full` rerun fixed the former `page_002_band_019` false-positive blocker: final `render_bbox` is contained, stale render geometry flags are removed, and clean inpaint with only tiny cleanup-limit noise no longer repropagates a mask critical.
- Stop point honored: no execution beyond Task 10.

---

## Final Done Criteria

This plan is complete only when:

- The app path no longer forces legacy fast white or fast dark by default.
- Sampled solid fill works with fast white disabled.
- No default path writes hardcoded white or black into a balloon/panel.
- Connected geometry fill uses sampled color or falls back to real inpaint.
- Fast dark is legacy opt-in or replaced by solid fill behavior.
- Pipeline trace says exactly which producer/strategy touched each stage.
- Unit tests pass for the touched contracts.
- At least two real chapters produce QA PASS with no critical visual issues.

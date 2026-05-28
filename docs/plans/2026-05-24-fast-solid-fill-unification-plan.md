# Fast Solid Fill Unification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make sampled solid fill the only default fast-fill strategy, so white, black, and flat colored balloons are cleaned with the sampled local background color instead of separate hardcoded white/black paths.

**Architecture:** Split the current solid-color sampler out of the fast-white path and make it a first-class `_apply_fast_solid_balloon_fill` stage. Legacy fast-white and fast-dark paths become explicit opt-in fallbacks only, while textured, gradient, translucent, or unsafe regions continue to go to real inpaint. Metrics must report solid fill separately so failures are traceable.

**Tech Stack:** Python 3.12, OpenCV/NumPy, `pipeline/inpainter`, `pipeline/strip`, pytest.

---

## Current Finding

Today the useful solid-color behavior is nested inside `N:/TraduzAI/pipeline/inpainter/__init__.py:_apply_fast_white_balloon_fill`.

That means disabling `TRADUZAI_STRIP_FAST_WHITE_INPAINT` also disables the solid sampler. The dedicated dark-panel fast fill is separate, but current tests and debug evidence show it can fail its own `not_solid_dark_panel` gate, so relying on it as the black fill path is brittle.

The target behavior is:

- `fast_solid`: enabled by default.
- `fast_white`: disabled by default, legacy opt-in only.
- `fast_dark_panel`: disabled by default, legacy opt-in only.
- `connected_white_geometry`: either migrated to sampled solid fill or gated as legacy white fill, so it no longer hardcodes white as a default cleanup strategy.
- real inpaint remains the fallback for textures, gradients, translucent UI, faces/art, and uncertain masks.

---

## Task 1: Add Independent Fast Solid Feature Flag

**Files:**
- Modify: `N:/TraduzAI/pipeline/inpainter/__init__.py`
- Test: `N:/TraduzAI/pipeline/tests/test_vision_stack_inpainter.py`

**Step 1: Write failing tests**

Add tests that assert:

- default config enables solid fill;
- default config disables legacy fast-white;
- default config disables legacy fast-dark;
- setting `TRADUZAI_STRIP_FAST_SOLID_INPAINT=0` disables only solid fill.

**Step 2: Run the focused tests**

Run:

```powershell
N:/TraduzAI/pipeline/venv/Scripts/python.exe -m pytest N:/TraduzAI/pipeline/tests/test_vision_stack_inpainter.py -k "fast_solid or fast_white or fast_dark" -v
```

Expected before implementation: at least one test fails because solid has no independent flag or white/dark defaults do not match the new contract.

**Step 3: Implement the feature flag**

Add:

```python
def _fast_solid_balloon_fill_enabled() -> bool:
    flag = os.getenv("TRADUZAI_STRIP_FAST_SOLID_INPAINT", "1").strip().lower()
    return flag in {"1", "true", "yes", "on"}
```

Change defaults:

```python
def _fast_white_balloon_fill_enabled() -> bool:
    flag = os.getenv("TRADUZAI_STRIP_FAST_WHITE_INPAINT", "0").strip().lower()
    return flag in {"1", "true", "yes", "on"}

def _fast_dark_panel_fill_enabled() -> bool:
    flag = os.getenv("TRADUZAI_STRIP_FAST_DARK_PANEL_FILL", "0").strip().lower()
    return flag in {"1", "true", "yes", "on"}
```

**Step 4: Re-run tests**

Expected after implementation: focused flag tests pass.

---

## Task 2: Extract `_apply_fast_solid_balloon_fill`

**Files:**
- Modify: `N:/TraduzAI/pipeline/inpainter/__init__.py`
- Test: `N:/TraduzAI/pipeline/tests/test_vision_stack_inpainter.py`

**Step 1: Write failing behavior tests**

Add tests for four cases:

- white balloon text is filled with white from sampling;
- black/dark solid panel text is filled with black from sampling;
- flat colored balloon is filled with its sampled color, not white;
- textured/gradient background is rejected and leaves the block for real inpaint.

**Step 2: Extract shared logic**

Move the solid sampling section currently inside `_apply_fast_white_balloon_fill` into a new function:

```python
def _apply_fast_solid_balloon_fill(
    band_rgb: np.ndarray,
    ocr_page: dict,
    vision_blocks: list[dict],
) -> tuple[np.ndarray, list[dict], dict]:
    ...
```

This function must:

- require a valid text mask from line polygons, text pixels, or validated text evidence;
- clamp fill to the text geometry and safe balloon/text limit;
- call `_sample_solid_fill_color_for_mask`;
- reject non-solid samples instead of falling back to white;
- remove covered `vision_blocks`;
- record per-text metadata with sampled RGB, luma/chroma/variance, and rejection reason.

**Step 3: Keep legacy white behavior narrow**

Leave `_apply_fast_white_balloon_fill` as a legacy explicit fallback only. It should no longer own the generic solid sampler.

**Step 4: Re-run tests**

Run:

```powershell
N:/TraduzAI/pipeline/venv/Scripts/python.exe -m pytest N:/TraduzAI/pipeline/tests/test_vision_stack_inpainter.py -k "fast_solid" -v
```

Expected: white, black, and colored flat fills pass; textured/gradient rejection passes.

---

## Task 3: Reorder Inpaint Flow

**Files:**
- Modify: `N:/TraduzAI/pipeline/inpainter/__init__.py`
- Test: `N:/TraduzAI/pipeline/tests/test_vision_stack_inpainter.py`
- Test: `N:/TraduzAI/pipeline/tests/test_strip_process_bands.py`

**Step 1: Write a flow test**

Add a test proving the default order is:

1. sampled solid fill;
2. connected geometry fill only if migrated to solid-safe behavior;
3. explicit legacy white/dark only when env-enabled;
4. local fast fill if explicitly enabled;
5. real inpaint for remaining blocks.

**Step 2: Update `apply_inpaint`**

Change the section around `apply_inpaint` so default processing calls `_apply_fast_solid_balloon_fill` before any legacy fast paths.

Set metadata separately:

```python
ocr_page["_strip_used_fast_solid_fill"] = False
ocr_page["_strip_used_fast_white_fill"] = False
ocr_page["_strip_used_fast_dark_fill"] = False
```

**Step 3: Migrate or gate connected white geometry**

Audit `_apply_connected_white_geometry_fill`:

- if it can sample color safely, route its fill color through `_sample_solid_fill_color_for_mask`;
- if it still hardcodes white, gate it behind `TRADUZAI_STRIP_FAST_WHITE_INPAINT=1`.

Default must not apply hardcoded white fill.

**Step 4: Re-run focused tests**

Expected: no default path uses hardcoded white or hardcoded black.

---

## Task 4: Update Metrics And Debug Trace

**Files:**
- Modify: `N:/TraduzAI/pipeline/inpainter/__init__.py`
- Modify: `N:/TraduzAI/pipeline/strip/process_bands.py`
- Modify: `N:/TraduzAI/pipeline/strip/run.py`
- Test: `N:/TraduzAI/pipeline/tests/test_strip_process_bands.py`
- Test: `N:/TraduzAI/pipeline/tests/test_strip_run.py`

**Step 1: Add metadata tests**

Assert the pipeline reports:

- `fast_solid_balloon_count`;
- `fast_solid_band_count`;
- `fast_solid_fill_samples`;
- `fast_solid_rejection_reasons`;
- `used_fast_solid_fill`;
- optional `fast_solid_white_count`, `fast_solid_black_count`, `fast_solid_colored_count`.

**Step 2: Update inpainter metadata**

Record:

```python
ocr_page["_strip_fast_solid_balloon_count"] = ...
ocr_page["_strip_fast_solid_rejection_reasons"] = ...
ocr_page["_strip_fast_solid_fill_samples"] = ...
ocr_page["_strip_used_fast_solid_fill"] = ...
```

**Step 3: Update strip aggregation**

Add solid metrics beside the existing white/local metrics in `process_bands.py` and `run.py`.

**Step 4: Re-run strip tests**

Run:

```powershell
N:/TraduzAI/pipeline/venv/Scripts/python.exe -m pytest N:/TraduzAI/pipeline/tests/test_strip_process_bands.py N:/TraduzAI/pipeline/tests/test_strip_run.py -k "fast_solid or inpaint" -v
```

Expected: solid metrics appear in band and chapter summaries.

---

## Task 5: Validate On Known Failure Pages

**Files:**
- No source edits unless validation finds a regression.
- Read output/debug under `N:/TraduzAI/DEBUGM/runs/`.

**Step 1: Run focused unit tests**

```powershell
N:/TraduzAI/pipeline/venv/Scripts/python.exe -m pytest N:/TraduzAI/pipeline/tests/test_vision_stack_inpainter.py N:/TraduzAI/pipeline/tests/test_mask_builder.py -x
```

**Step 2: Run a small real chapter validation**

Use the same normal app-shaped pipeline command pattern already used in this session, with inherited debug flags cleared, against:

- `C:/Users/PICHAU/Downloads/Articuno (comick)_Ch. 61 OFFICIAL TRANSLATION`
- `N:/TraduzAI/DEBUGM/runs/2026-05-22_mihon_matrix20_v26/13_monster_actor_ch75` source chapter if available, or the original Monster Actor chapter path used for that run.

**Step 3: Inspect visual cases**

Check these specifically:

- light blue speech balloon should be light blue after cleanup, not white patches;
- black/dark panel should not keep English text and should not get black bars;
- textured or translucent panels should go to real inpaint, not solid fill;
- face/art false positives must not be filled;
- connected balloons must not be split or recentered incorrectly by inpaint metadata.

**Step 4: Compare QA and metadata**

Expected:

- QA remains `PASS`;
- `critical_issue_count=0`;
- solid counts are non-zero on flat balloons;
- legacy white/dark counts are zero by default;
- real inpaint still handles uncertain backgrounds.

---

## Task 6: Runtime Defaults And Rollback

**Files:**
- Modify only if needed: `N:/TraduzAI/pipeline/runtime_profiles.py`
- Modify only if needed: `N:/TraduzAI/pipeline/main.py`
- Modify only if needed: `N:/TraduzAI/src-tauri/src/commands/pipeline.rs`
- Test: `N:/TraduzAI/pipeline/tests/test_runtime_profiles.py`
- Test: `N:/TraduzAI/pipeline/tests/test_main_emit.py`

**Step 1: Verify app path does not override env incorrectly**

Confirm the app path does not force `TRADUZAI_STRIP_FAST_WHITE_INPAINT=1` or `TRADUZAI_STRIP_FAST_DARK_PANEL_FILL=1`.

**Step 2: Set safe defaults**

Default expected environment:

```text
TRADUZAI_STRIP_FAST_SOLID_INPAINT=1
TRADUZAI_STRIP_FAST_WHITE_INPAINT=0
TRADUZAI_STRIP_FAST_DARK_PANEL_FILL=0
TRADUZAI_STRIP_FAST_LOCAL_INPAINT=0
```

**Step 3: Keep rollback switches**

If a regression appears, rollback should be possible without code revert:

```powershell
$env:TRADUZAI_STRIP_FAST_SOLID_INPAINT='0'
$env:TRADUZAI_STRIP_FAST_WHITE_INPAINT='1'
$env:TRADUZAI_STRIP_FAST_DARK_PANEL_FILL='1'
```

**Step 4: Re-run runtime tests**

Expected: runtime presets preserve the new defaults unless explicitly overridden.

---

## Final Verification

Run:

```powershell
N:/TraduzAI/pipeline/venv/Scripts/python.exe -m py_compile N:/TraduzAI/pipeline/inpainter/__init__.py N:/TraduzAI/pipeline/strip/process_bands.py N:/TraduzAI/pipeline/strip/run.py
N:/TraduzAI/pipeline/venv/Scripts/python.exe -m pytest N:/TraduzAI/pipeline/tests/test_vision_stack_inpainter.py N:/TraduzAI/pipeline/tests/test_mask_builder.py N:/TraduzAI/pipeline/tests/test_strip_process_bands.py N:/TraduzAI/pipeline/tests/test_strip_run.py -x
```

Then run at least one real chapter through the normal app pipeline and inspect the rendered output plus `qa_report.json`.

Do not commit automatically in this dirty checkout. If executed in a clean branch or worktree, commit task-by-task after each green verification.

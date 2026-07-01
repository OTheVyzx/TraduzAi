# Final Rerender QA Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the final `translated/*.jpg` rerender the visual source of truth, refresh debug band crops from it, and run QA after that final image exists.

**Architecture:** Keep the useful final project rerender path, because it often fixes band-level sizing/positioning. Move/guarantee debug crop refresh and visual QA after `rerender_final_project_images`, so `10_copyback_reassemble/final_bands` and QA reports describe the actual delivered pixels.

**Tech Stack:** Python 3.12, OpenCV/Pillow image diff and crop logic, existing `pipeline/main.py` project rerender path, existing `pipeline/tests/test_main_emit.py` tests.

---

### Task 1: Document the Current Contract Failure

**Files:**
- Modify: `pipeline/tests/test_main_emit.py`
- Reference: `pipeline/main.py`

**Step 1: Write a failing test for stale final band crops**

Add a test that creates:
- a fake `translated/001.jpg`;
- a fake `debug/e2e/10_copyback_reassemble/final_band_crops.jsonl`;
- an old `final_bands/page_001_band_000.jpg`;
- then simulates a final rerender changing `translated/001.jpg`.

Expected assertion: after the final rerender path completes, `final_bands/page_001_band_000.jpg` must equal the crop from the final `translated/001.jpg`.

**Step 2: Run the test**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_main_emit.py -k "final_band_crops" -x
```

Expected: FAIL, proving the debug crop can be stale if refresh happens before final rerender.

---

### Task 2: Move Final Crop Refresh After Final Rerender

**Files:**
- Modify: `pipeline/main.py`

**Step 1: Find the current refresh call**

Current owner:
- `_refresh_debug_final_band_crops_from_translated`
- `sync_final_page_space_typeset`
- `rerender_final_project_images`
- `late_render_contract_repair`

**Step 2: Keep the helper, change the call order**

Do not remove `_refresh_debug_final_band_crops_from_translated`.

Ensure it runs after:

```python
_rerender_final_project_images_from_metadata(...)
_rerender_final_project_images_after_contract(...)
```

and after `late_render_contract_repair` if that path rerenders pages.

**Step 3: Record a stronger audit**

Extend the refresh audit with:

```python
{
    "source": "translated_after_final_project_rerender",
    "after_final_project_image_rerender": True,
    "after_late_render_contract_repair": True,
}
```

**Step 4: Run the focused test**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_main_emit.py -k "final_band_crops" -x
```

Expected: PASS.

---

### Task 3: Add Post-Rerender Visual QA

**Files:**
- Modify: `pipeline/main.py`
- Test: `pipeline/tests/test_main_emit.py`

**Step 1: Add a QA helper**

Add a helper near `_refresh_debug_final_band_crops_from_translated`:

```python
def _qa_translated_final_crops_against_layers(recorder, project_data: dict, work_dir: Path) -> dict:
    ...
```

It should:
- read `debug/e2e/10_copyback_reassemble/final_band_crops.jsonl`;
- crop from final `translated/*.jpg`;
- compare with refreshed `final_bands/*.jpg`;
- inspect project `text_layers` for the same trace ids;
- report suspicious final render conditions.

**Step 2: QA checks**

Minimum checks:
- `translated_crop_matches_final_band`: pixel diff must be near zero after refresh.
- `render_bbox_inside_crop`: final `render_bbox` must intersect the crop.
- `render_bbox_inside_safe_or_balloon`: rendered text should not strongly exceed `safe_text_box` or `balloon_bbox`.
- `center_drift_px`: translated text center should not drift far from original `text_pixel_bbox` center unless explicitly allowed.
- `tiny_text_ratio`: rendered text should not be too small relative to source text or balloon.
- `clipped_text_flag`: if `TEXT_CLIPPED`/`TEXT_OVERFLOW` remains, mark the crop as review/fail.

**Step 3: Persist artifacts**

Write:

```text
debug/e2e/11_qa_export_gate/final_rerender_visual_qa.json
debug/e2e/11_qa_export_gate/final_rerender_visual_qa.jsonl
```

Each row should include:

```json
{
  "band_id": "page_002_band_007",
  "translated_output_page": "001.jpg",
  "trace_ids": ["ocr_001@page_002_band_007"],
  "status": "pass|warn|fail",
  "flags": [],
  "metrics": {}
}
```

**Step 4: Run after final crop refresh**

Call the QA helper only after the final refresh from `translated`.

---

### Task 4: Add Dark Bubble Specific QA

**Files:**
- Modify: `pipeline/main.py`
- Test: `pipeline/tests/test_main_emit.py`

**Step 1: Detect dark bubble layers**

Use layer metadata:

```python
bubble_mask_source in {"image_dark_bubble_mask", "image_dark_panel_mask"}
layout_profile in {"dark_bubble", "dark_panel"}
qa_flags contains "dark_panel_style_grouped" or "dark_oval_safe_height_expanded"
```

**Step 2: Check known failure classes**

For dark bubbles, add metrics:
- `dark_text_tiny_ratio`
- `dark_text_center_drift`
- `dark_connected_lobe_overlap`
- `dark_text_outside_balloon_ratio`
- `dark_original_residual_score`

**Step 3: Connected lobe rule**

For connected dark bubbles, validate each trace id/lobe independently. Do not accept one lobe looking correct if the sibling lobe is tiny, missing, or shifted.

**Step 4: Focused tests**

Add synthetic tests for:
- single dark oval with normal text;
- short dark oval text like `1.000 pontos`;
- connected dark two-lobe text;
- dark rectangular UI panel.

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_main_emit.py -k "final_rerender_visual_qa or dark" -x
```

---

### Task 5: Restore the Noclamp Visual Policy Separately

**Files:**
- Modify: `pipeline/typesetter/renderer.py`
- Test: `pipeline/tests/test_typesetting_renderer.py`

**Step 1: Revert only aggressive dark sizing behavior**

Keep:
- source text center preservation;
- final project rerender;
- dark bubble detection and lobe split.

Disable or gate:
- `dark_visual_effect_capacity_inset` when it shrinks otherwise valid text;
- `dark_auto_font_search_cap` for normal dark bubble text;
- short-text shrink behavior that makes `1.000 pontos` tiny.

**Step 2: Keep size decision with renderer**

The rule should be:
- position follows original text center;
- size is decided by renderer fit;
- QA rejects if it overflows or becomes tiny.

**Step 3: Tests**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_typesetting_renderer.py -k "dark or lobe or center" -x
```

Expected: dark text remains centered without forced miniaturization.

---

### Task 6: Full Visual Validation

**Files:**
- No code unless failures appear.
- Artifacts under `N:\TraduzAI\DEBUGM\runs\...`

**Step 1: Run cap 2**

Run the same cap 2 command/profile used for the latest dark-bubble runs, with a new run name such as:

```text
god_of_death_ch2_noclamp_finalqa_YYYYMMDD
```

**Step 2: Inspect required bands visually**

Compare:
- `page_002_band_007`
- `page_002_band_013`
- `page_002_band_017`
- `page_002_band_018`
- known connected dark lobes on pages 3-6
- dark rectangular panels from page 2

Use the final truth:

```text
translated/*.jpg
debug/e2e/10_copyback_reassemble/final_bands/*.jpg
debug/e2e/11_qa_export_gate/final_rerender_visual_qa.json
```

**Step 3: Confirm debug parity**

For every inspected band:
- `final_bands` must match the crop from `translated`;
- if not, the run fails regardless of text quality.

**Step 4: Regression check cap 1**

Run the cap 1 smoke set or full chapter if time permits.

Target cases:
- white connected bubbles;
- white ovals;
- T/N text-only boxes;
- rotated text.

Expected: no regression from restoring the dark-bubble noclamp policy.

---

### Task 7: Completion Criteria

The work is complete only when:

- `translated` is the declared visual source of truth.
- `final_bands` are refreshed after every final rerender.
- QA runs after final rerender and writes a report.
- The QA catches stale debug crops.
- The QA catches tiny dark text.
- The QA catches overflow/cut dark text.
- The QA catches connected-lobe drift.
- Cap 2 visual output is acceptable in the known dark-bubble cases.
- Cap 1 is not regressed.


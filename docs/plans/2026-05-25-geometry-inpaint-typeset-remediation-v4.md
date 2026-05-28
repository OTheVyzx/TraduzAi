# Geometry Inpaint Typeset Remediation V4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the Chapter 1 visual regressions where text is rendered outside the correct balloon, final translated pages diverge from debug intermediates, fast solid fill leaves English residue, and QA/export gate allows broken output.

**Architecture:** Keep the existing pipeline structure. Fix the coordinate contract in `pipeline/strip/run.py`, extend the existing coordinate auditor in `pipeline/debug_tools/bbox.py`, add final-render safety checks in `pipeline/main.py` and `pipeline/typesetter/renderer.py`, and make `fast_solid_fill` verified before it can suppress real inpaint. Do not create a new geometry package for this fix.

**Tech Stack:** Python 3.12, OpenCV, PaddleOCR debug artifacts, TraduzAI strip pipeline, typesetter renderer, inpaint fast paths, `project.json`, E2E debug/QA export gate.

---

## Problem Summary

Run used as evidence:

```text
N:/TraduzAI/DEBUGM/runs/2026-05-24_task10_real_validation_20260524_174315/chapter1_full
```

The final images in `translated/*.jpg` are wrong even when some intermediate debug artifacts look acceptable. The confirmed root problem is mixed coordinate space:

- `text_pixel_bbox`, `balloon_bbox`, `target_bbox`: page/final coordinates.
- `bubble_inner_bbox`, `bubble_mask_bbox`, `safe_text_box`, `render_bbox`: still band-local in several final layers.

The worst evidence is `page_002_band_005`:

```text
target/balloon around y ~= 5606
bubble_inner/safe/render around y ~= 230
```

The renderer uses `bubble_inner_bbox` as the safe area, so it draws the translated text near y=246 instead of y=5600. The page then gets rewritten by final page-space typeset.

There is also an independent inpaint issue:

```text
used_fast_solid_fill = true
used_real_inpaint = false
raw_mask_pixels = 0
expanded_mask_pixels = 0
post_cleanup_skipped_reason = fast_solid_fill
```

This lets residual English text survive in balloons such as `I'M STARVING`, `SOME.`, blue title boxes, and black panels.

## Required Design Decisions

1. Do not add a generic early-return in `_shift_text_geometry_y` / `_shift_text_geometry_xy` based only on `_coordinate_space == "page"`.

   Reason: the real flow performs at least two required shifts:

   - band-local to strip-global in `pipeline/strip/run.py`.
   - strip-global to final output page-local in `pipeline/strip/run.py`.

   A generic early-return after the first shift would skip the second shift and leave strip-global coordinates inside page-local output.

2. Coordinate conversion must be explicit by caller context, not guessed by a field alone.

3. Final page-space rerender must never overwrite a translated page if the page texts contain confirmed mixed coordinate boxes.

4. Fast solid fill must be quality-verified before removing an inpaint block.

5. QA must block confirmed visual blockers, not merely report warnings.

## File Map

### Existing files to modify

- `pipeline/strip/run.py`
  - Owns `_shift_text_geometry_y`, `_shift_text_geometry_xy`, final output page assembly, and page-space rerender paths.

- `pipeline/debug_tools/bbox.py`
  - Owns `BBOX_KEYS`, `layout_block_records`, `audit_bbox_coordinate_space`, and coordinate audit output.

- `pipeline/typesetter/renderer.py`
  - Owns safe area selection, `bubble_inner_bbox` priority, render plan generation, and copied render debug fields.

- `pipeline/main.py`
  - Owns sync of output pages, `sync_final_page_space_typeset`, project build, debug export gate artifacts, and QA propagation.

- `pipeline/inpainter/__init__.py`
  - Owns `_apply_fast_solid_balloon_fill`, `_block_is_covered_by_fast_fill`, metadata, residual and fallback behavior.

- `pipeline/qa/translation_qa.py`
  - Owns `FLAG_SEVERITY`.

- `pipeline/qa/export_gate.py`
  - Owns export blocking behavior and linked issue details.

- `docs/debug/e2e_pipeline_debug_guide.md`
  - Add the new diagnostic interpretation after code validation.

### Existing tests to extend

- `pipeline/tests/test_derived_bbox_coordinate_audit.py`
- `pipeline/tests/test_strip_balloon_bbox_propagation.py`
- `pipeline/tests/test_typesetting_renderer.py`
- `pipeline/tests/test_main_emit.py`
- `pipeline/tests/test_vision_stack_inpainter.py`
- `pipeline/tests/test_qa_flag_propagation_v2.py`
- `pipeline/tests/test_export_gate.py`

### New test files allowed

- `pipeline/tests/test_page_space_rerender_guard.py`
- `pipeline/tests/test_final_geometry_contract.py`

Only create new files if the existing tests become too broad or unreadable.

---

## Task 1: Reproduce the full coordinate flow bug in tests

**Files:**

- Modify: `pipeline/tests/test_strip_balloon_bbox_propagation.py`
- Modify: `pipeline/tests/test_derived_bbox_coordinate_audit.py`
- Modify later: `pipeline/strip/run.py`
- Modify later: `pipeline/debug_tools/bbox.py`

- [ ] **Step 1: Add a failing test for band-local to strip-global shift**

Append this test to `pipeline/tests/test_strip_balloon_bbox_propagation.py`:

```python
def test_shift_text_geometry_y_shifts_bubble_fields_from_band_to_strip():
    from strip.run import _shift_text_geometry_y

    text = {
        "id": "ocr_002",
        "band_id": "page_002_band_005",
        "bbox": [25, 16, 667, 325],
        "text_pixel_bbox": [498, 235, 656, 320],
        "balloon_bbox": [466, 186, 696, 357],
        "bubble_mask_bbox": [501, 218, 661, 325],
        "bubble_inner_bbox": [513, 230, 649, 313],
        "safe_text_box": [525, 242, 637, 301],
        "render_bbox": [542, 246, 620, 296],
        "line_polygons": [
            [[498, 232], [658, 234], [658, 258], [498, 256]],
        ],
    }

    shifted = _shift_text_geometry_y(text, 5420)

    assert shifted["bbox"] == [25, 5436, 667, 5745]
    assert shifted["text_pixel_bbox"] == [498, 5655, 656, 5740]
    assert shifted["balloon_bbox"] == [466, 5606, 696, 5777]
    assert shifted["bubble_mask_bbox"] == [501, 5638, 661, 5745]
    assert shifted["bubble_inner_bbox"] == [513, 5650, 649, 5733]
    assert shifted["safe_text_box"] == [525, 5662, 637, 5721]
    assert shifted["render_bbox"] == [542, 5666, 620, 5716]
    assert shifted["line_polygons"][0][0] == [498, 5652]
```

- [ ] **Step 2: Add a failing test for strip-global to final-page-local shift**

Append this test to `pipeline/tests/test_strip_balloon_bbox_propagation.py`:

```python
def test_shift_text_geometry_y_shifts_bubble_fields_from_strip_to_output_page():
    from strip.run import _shift_text_geometry_y

    strip_global = {
        "id": "ocr_002",
        "band_id": "page_002_band_005",
        "bbox": [25, 5436, 667, 5745],
        "text_pixel_bbox": [498, 5655, 656, 5740],
        "balloon_bbox": [466, 5606, 696, 5777],
        "bubble_mask_bbox": [501, 5638, 661, 5745],
        "bubble_inner_bbox": [513, 5650, 649, 5733],
        "safe_text_box": [525, 5662, 637, 5721],
        "render_bbox": [542, 5666, 620, 5716],
        "line_polygons": [
            [[498, 5652], [658, 5654], [658, 5678], [498, 5676]],
        ],
    }

    output_page_local = _shift_text_geometry_y(strip_global, 0)

    assert output_page_local["bubble_inner_bbox"] == [513, 5650, 649, 5733]
    assert output_page_local["safe_text_box"] == [525, 5662, 637, 5721]
    assert output_page_local["render_bbox"] == [542, 5666, 620, 5716]
```

This specific test uses `delta_y=0` because the referenced output page starts at y=0. Add another test in Step 3 for a non-zero output page.

- [ ] **Step 3: Add a non-zero output page remap test**

Append:

```python
def test_shift_text_geometry_y_does_not_skip_required_second_shift():
    from strip.run import _shift_text_geometry_y

    strip_global = {
        "id": "ocr_001",
        "band_id": "page_006_band_104",
        "bbox": [453, 81927, 645, 81991],
        "text_pixel_bbox": [451, 81939, 643, 81999],
        "balloon_bbox": [453, 81927, 645, 81991],
        "bubble_mask_bbox": [453, 81927, 645, 81991],
        "bubble_inner_bbox": [465, 81939, 633, 81979],
        "safe_text_box": [474, 81943, 625, 81975],
        "render_bbox": [503, 81947, 594, 81971],
        "_coordinate_space": "strip",
    }

    final_page_local = _shift_text_geometry_y(strip_global, -69050)

    assert final_page_local["bbox"] == [453, 12877, 645, 12941]
    assert final_page_local["bubble_inner_bbox"] == [465, 12889, 633, 12929]
    assert final_page_local["safe_text_box"] == [474, 12893, 625, 12925]
    assert final_page_local["render_bbox"] == [503, 12897, 594, 12921]
```

This test explicitly prevents the dangerous `_coordinate_space == "page"` early-return pattern. If a future implementation skips the second shift, this test fails.

- [ ] **Step 4: Run the focused tests and confirm failure**

Run:

```powershell
cd N:\TraduzAI\pipeline
.\venv\Scripts\python.exe -m pytest tests\test_strip_balloon_bbox_propagation.py -q
```

Expected before implementation:

```text
FAILED ... bubble_inner_bbox ...
FAILED ... bubble_mask_bbox ...
```

Do not continue until the test fails for the expected reason.

---

## Task 2: Fix geometry shifting without unsafe idempotency

**Files:**

- Modify: `pipeline/strip/run.py:187`
- Modify: `pipeline/strip/run.py:1686`
- Test: `pipeline/tests/test_strip_balloon_bbox_propagation.py`

- [ ] **Step 1: Add bubble fields to `_shift_text_geometry_y`**

In `pipeline/strip/run.py`, update the bbox key tuple inside `_shift_text_geometry_y` to include:

```python
        "bubble_mask_bbox",
        "bubble_inner_bbox",
        "balloon_inner_bbox",
```

The tuple should contain these keys in the same loop that already shifts `bbox`, `balloon_bbox`, `safe_text_box`, and `render_bbox`.

Forbidden pattern: do not add any generic `_coordinate_space == "page"` early return to `_shift_text_geometry_y` or `_shift_text_geometry_xy`. Those helpers are pure transforms and must still support the legitimate two-stage remap: band-local to strip-global, then strip-global to output-page-local.

- [ ] **Step 2: Add bubble fields to `_shift_text_geometry_xy`**

In `pipeline/strip/run.py`, update the bbox key tuple inside `_shift_text_geometry_xy` to include:

```python
        "bubble_mask_bbox",
        "bubble_inner_bbox",
        "balloon_inner_bbox",
```

Again, do not add a generic early-return based on coordinate-space fields.

- [ ] **Step 3: Optionally add trace metadata without changing shift behavior**

If trace metadata is needed, add only non-control metadata:

```python
if delta_y:
    shifted["_last_geometry_shift_y"] = int(delta_y)
```

For `_shift_text_geometry_xy`:

```python
if dx or dy:
    shifted["_last_geometry_shift_xy"] = [int(dx), int(dy)]
```

These fields must not control future shifts.

- [ ] **Step 4: Run the focused tests**

Run:

```powershell
cd N:\TraduzAI\pipeline
.\venv\Scripts\python.exe -m pytest tests\test_strip_balloon_bbox_propagation.py -q
```

Expected after implementation:

```text
passed
```

---

## Task 3: Extend coordinate audit to catch bubble/safe/render mismatches

**Files:**

- Modify: `pipeline/debug_tools/bbox.py:7`
- Modify: `pipeline/tests/test_derived_bbox_coordinate_audit.py`

- [ ] **Step 1: Add a failing audit test for bubble-local boxes**

Append this to `pipeline/tests/test_derived_bbox_coordinate_audit.py`:

```python
def test_audit_flags_bubble_and_safe_boxes_that_remain_band_local():
    from debug_tools.bbox import audit_bbox_coordinate_space, layout_block_records

    page = {
        "page_id": "page_001",
        "height": 13832,
        "width": 800,
        "texts": [
            {
                "id": "ocr_002",
                "band_id": "page_002_band_005",
                "band_y_top": 5420,
                "band_height": 895,
                "bbox": [25, 5436, 667, 5745],
                "text_pixel_bbox": [498, 5655, 656, 5740],
                "balloon_bbox": [466, 5606, 696, 5777],
                "bubble_mask_bbox": [501, 218, 661, 325],
                "bubble_inner_bbox": [513, 230, 649, 313],
                "safe_text_box": [525, 242, 637, 301],
                "render_bbox": [542, 246, 620, 296],
            }
        ],
    }

    records = layout_block_records([page])
    audit = audit_bbox_coordinate_space(records)

    assert audit["summary"]["all_consistent"] is False
    assert audit["summary"]["by_key"]["bubble_mask_bbox"]["mismatch"] == 1
    assert audit["summary"]["by_key"]["bubble_inner_bbox"]["mismatch"] == 1
    assert audit["summary"]["by_key"]["safe_text_box"]["mismatch"] == 1
    assert audit["summary"]["by_key"]["render_bbox"]["mismatch"] == 1
    assert any(
        finding["blocker"] == "derived_bbox_coordinate_mismatch"
        and finding["severity"] == "critical"
        for finding in audit["findings"]
    )
```

- [ ] **Step 2: Add bubble fields to `BBOX_KEYS`**

In `pipeline/debug_tools/bbox.py`, extend `BBOX_KEYS`:

```python
BBOX_KEYS = (
    "source_bbox",
    "bbox",
    "text_pixel_bbox",
    "balloon_bbox",
    "bubble_mask_bbox",
    "bubble_inner_bbox",
    "balloon_inner_bbox",
    "layout_bbox",
    "render_bbox",
    "safe_text_box",
    "_debug_safe_text_box",
    "layout_safe_bbox",
    "position_bbox",
    "capacity_bbox",
    "target_bbox",
    "connected_position_bboxes",
)
```

- [ ] **Step 3: Add a specific audit flag helper**

If the existing audit only returns findings but does not provide a direct flag list, add this helper near `audit_bbox_coordinate_space`:

```python
def coordinate_audit_flags(audit: dict[str, Any]) -> list[str]:
    findings = list(audit.get("findings") or [])
    flags: list[str] = []
    if any(item.get("blocker") == "derived_bbox_coordinate_mismatch" for item in findings):
        flags.append("layout_bbox_coordinate_mismatch")
    if any(str(item.get("key") or "") in {"bubble_inner_bbox", "safe_text_box", "render_bbox"} for item in findings):
        flags.append("page_space_rerender_mixed_coordinates")
    return sorted(set(flags))
```

Use `Any` from the existing imports. If `Any` is already imported, do not duplicate.

- [ ] **Step 4: Add a test for `coordinate_audit_flags`**

Append:

```python
def test_coordinate_audit_flags_promote_page_space_mismatch():
    from debug_tools.bbox import audit_bbox_coordinate_space, coordinate_audit_flags, layout_block_records

    records = layout_block_records([
        {
            "height": 13832,
            "width": 800,
            "texts": [
                {
                    "id": "ocr_002",
                    "band_id": "page_002_band_005",
                    "band_y_top": 5420,
                    "band_height": 895,
                    "bbox": [25, 5436, 667, 5745],
                    "balloon_bbox": [466, 5606, 696, 5777],
                    "bubble_inner_bbox": [513, 230, 649, 313],
                    "safe_text_box": [525, 242, 637, 301],
                }
            ],
        }
    ])

    flags = coordinate_audit_flags(audit_bbox_coordinate_space(records))

    assert "layout_bbox_coordinate_mismatch" in flags
    assert "page_space_rerender_mixed_coordinates" in flags
```

- [ ] **Step 5: Run audit tests**

Run:

```powershell
cd N:\TraduzAI\pipeline
.\venv\Scripts\python.exe -m pytest tests\test_derived_bbox_coordinate_audit.py -q
```

Expected:

```text
passed
```

---

## Task 4: Guard final page-space rerender from mixed coordinates

**Files:**

- Modify: `pipeline/main.py:1808`
- Modify: `pipeline/qa/translation_qa.py`
- Test: `pipeline/tests/test_page_space_rerender_guard.py`
- Test: `pipeline/tests/test_export_gate.py`

- [ ] **Step 1: Add critical QA flags**

In `pipeline/qa/translation_qa.py`, add these entries to `FLAG_SEVERITY`:

```python
    "page_space_rerender_mixed_coordinates": "critical",
    "render_bbox_far_from_target_bbox": "critical",
    "bubble_inner_bbox_coordinate_mismatch": "critical",
```

Keep existing `layout_bbox_coordinate_mismatch` unchanged.

- [ ] **Step 2: Create a helper in `pipeline/main.py`**

Add this helper near other debug/QA helpers in `pipeline/main.py`:

```python
def _page_text_coordinate_audit_flags(page_texts: list[dict], *, height: int, width: int) -> list[str]:
    try:
        from debug_tools.bbox import audit_bbox_coordinate_space, coordinate_audit_flags, layout_block_records
    except Exception:
        return []
    page = {"height": int(height), "width": int(width), "texts": page_texts}
    try:
        audit = audit_bbox_coordinate_space(layout_block_records([page]))
        return coordinate_audit_flags(audit)
    except Exception:
        return []
```

- [ ] **Step 3: Create a flag append helper in `pipeline/main.py`**

Add:

```python
def _append_page_text_flags(page_texts: list[dict], flags: list[str]) -> None:
    if not flags:
        return
    for text in page_texts or []:
        if not isinstance(text, dict) or text.get("skip_processing"):
            continue
        existing = {str(flag) for flag in text.get("qa_flags") or [] if flag}
        existing.update(str(flag) for flag in flags if flag)
        text["qa_flags"] = sorted(existing)
```

- [ ] **Step 4: Guard `sync_final_page_space_typeset`**

In `pipeline/main.py`, inside the loop under `with pipeline_timing.measure("sync_final_page_space_typeset")`, before `render_band_image`, add:

```python
                audit_flags = _page_text_coordinate_audit_flags(
                    page_texts,
                    height=int(clean_rgb.shape[0]),
                    width=int(clean_rgb.shape[1]),
                )
                if audit_flags:
                    _append_page_text_flags(page_texts, audit_flags)
                    strip_chapter_telemetry["main_final_page_space_rerender_blocked_count"] = (
                        int(strip_chapter_telemetry.get("main_final_page_space_rerender_blocked_count") or 0) + 1
                    )
                    continue
```

This is not silent: it marks critical flags on page texts, avoids overwriting the page with a bad final render, and lets export gate block later.

- [ ] **Step 5: Add test for the guard helper**

Create `pipeline/tests/test_page_space_rerender_guard.py`:

```python
from __future__ import annotations

import main


def test_page_text_coordinate_audit_flags_detects_local_safe_box():
    texts = [
        {
            "id": "ocr_002",
            "band_id": "page_002_band_005",
            "band_y_top": 5420,
            "band_height": 895,
            "bbox": [25, 5436, 667, 5745],
            "text_pixel_bbox": [498, 5655, 656, 5740],
            "balloon_bbox": [466, 5606, 696, 5777],
            "bubble_inner_bbox": [513, 230, 649, 313],
            "safe_text_box": [525, 242, 637, 301],
            "render_bbox": [542, 246, 620, 296],
            "translated": "POR FAVOR, PELO BEM DA CRIANCA.",
        }
    ]

    flags = main._page_text_coordinate_audit_flags(texts, height=13832, width=800)

    assert "layout_bbox_coordinate_mismatch" in flags
    assert "page_space_rerender_mixed_coordinates" in flags


def test_append_page_text_flags_marks_all_processable_texts():
    texts = [{"id": "ocr_1", "qa_flags": ["TEXT_OVERFLOW"]}, {"id": "ocr_2", "skip_processing": True}]

    main._append_page_text_flags(texts, ["page_space_rerender_mixed_coordinates"])

    assert texts[0]["qa_flags"] == ["TEXT_OVERFLOW", "page_space_rerender_mixed_coordinates"]
    assert "qa_flags" not in texts[1]
```

- [ ] **Step 6: Add export gate test for the new critical flag**

Append to `pipeline/tests/test_export_gate.py`:

```python
def test_export_gate_blocks_page_space_rerender_mixed_coordinates():
    from qa.export_gate import evaluate_export_gate

    project = {
        "paginas": [
            {
                "numero": 1,
                "text_layers": [
                    {
                        "id": "ocr_002",
                        "translated": "POR FAVOR, PELO BEM DA CRIANCA.",
                        "qa_flags": ["page_space_rerender_mixed_coordinates"],
                    }
                ],
            }
        ]
    }

    gate = evaluate_export_gate(project)

    assert gate["status"] == "BLOCK"
    assert gate["critical_issue_count"] == 1
    assert gate["issues"][0]["severity"] == "critical"
```

- [ ] **Step 7: Run tests**

Run:

```powershell
cd N:\TraduzAI\pipeline
.\venv\Scripts\python.exe -m pytest tests\test_page_space_rerender_guard.py tests\test_export_gate.py -q
```

Expected:

```text
passed
```

---

## Task 5: Make renderer reject stale safe boxes

**Files:**

- Modify: `pipeline/typesetter/renderer.py:5090`
- Modify: `pipeline/tests/test_typesetting_renderer.py`

- [ ] **Step 1: Add a renderer unit test**

Append to `pipeline/tests/test_typesetting_renderer.py` near existing safe area tests:

```python
def test_plan_text_layout_rejects_safe_box_that_does_not_intersect_target():
    from typesetter.renderer import plan_text_layout

    text = {
        "id": "ocr_002",
        "translated": "POR FAVOR, PELO BEM DA CRIANCA.",
        "bbox": [466, 5606, 696, 5777],
        "balloon_bbox": [466, 5606, 696, 5777],
        "bubble_inner_bbox": [513, 230, 649, 313],
        "safe_text_box": [525, 242, 637, 301],
        "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 26, "cor": "#000000"},
    }

    plan = plan_text_layout(text, [466, 5606, 696, 5777])

    assert plan["layout_safe_bbox"][1] >= 5600
    assert "safe_text_box_recomputed" in text.get("qa_flags", [])
```

If the local API signature differs, adapt the test to the existing tests around `plan_text_layout`; do not invent a new public API.

- [ ] **Step 2: Add the high severity flag**

In `pipeline/qa/translation_qa.py`, add:

```python
    "safe_text_box_recomputed": "high",
```

- [ ] **Step 3: Add a local helper in `renderer.py`**

Near the bbox helpers, add:

```python
def _append_render_qa_flag(text_data: dict, flag: str) -> None:
    flags = {str(item) for item in text_data.get("qa_flags") or [] if item}
    flags.add(str(flag))
    text_data["qa_flags"] = sorted(flags)
```

If an equivalent helper already exists, reuse it.

- [ ] **Step 4: Reject non-intersecting candidate safe bbox**

In `pipeline/typesetter/renderer.py`, in the loop that checks:

```python
for safe_key, safe_reason in (
    ("_visual_rect_inner_bbox", "visual_rect_inner"),
    ("bubble_inner_bbox", "bubble_inner_bbox"),
    ("layout_safe_bbox", ...),
    ("balloon_inner_bbox", "balloon_inner_bbox"),
):
```

after:

```python
candidate_safe_bbox = _layout_bbox(text_data.get(safe_key))
```

add:

```python
        if candidate_safe_bbox is not None and target_bbox is not None:
            if _bbox_intersection_area(candidate_safe_bbox, target_bbox) <= 0:
                render_debug = text_data.setdefault("_render_debug", {})
                render_debug["rejected_safe_box"] = {
                    "key": safe_key,
                    "value": list(candidate_safe_bbox),
                    "target_bbox": list(target_bbox),
                    "reason": "safe_box_outside_target_bbox",
                }
                _append_render_qa_flag(text_data, "safe_text_box_recomputed")
                continue
```

Then keep the existing assignment to `explicit_layout_safe_bbox`.

- [ ] **Step 5: Run renderer tests**

Run:

```powershell
cd N:\TraduzAI\pipeline
.\venv\Scripts\python.exe -m pytest tests\test_typesetting_renderer.py -q
```

Expected:

```text
passed
```

---

## Task 6: Propagate post-final coordinate audit into project and debug gate

**Files:**

- Modify: `pipeline/main.py`
- Modify: `pipeline/tests/test_qa_flag_propagation_v2.py`
- Modify: `pipeline/tests/test_main_emit.py`

- [ ] **Step 1: Add a final project geometry audit helper**

In `pipeline/main.py`, add:

```python
def _apply_final_project_coordinate_audit(project_data: dict) -> dict:
    try:
        from debug_tools.bbox import audit_bbox_coordinate_space, coordinate_audit_flags, layout_block_records
    except Exception:
        return {"applied": False, "flags_added": 0}

    pages = project_data.get("paginas") or []
    flags_added = 0
    for page in pages:
        texts = page.get("text_layers") or page.get("textos") or []
        height = int(page.get("height") or page.get("altura") or 0)
        width = int(page.get("width") or page.get("largura") or 0)
        audit_page = {"height": height, "width": width, "texts": texts}
        try:
            flags = coordinate_audit_flags(audit_bbox_coordinate_space(layout_block_records([audit_page])))
        except Exception:
            flags = []
        if flags:
            before = sum(len(text.get("qa_flags") or []) for text in texts if isinstance(text, dict))
            _append_page_text_flags(texts, flags)
            after = sum(len(text.get("qa_flags") or []) for text in texts if isinstance(text, dict))
            flags_added += max(0, after - before)
    return {"applied": True, "flags_added": flags_added}
```

- [ ] **Step 2: Call final project audit before export gate**

In `pipeline/main.py`, immediately before:

```python
from qa.export_gate import evaluate_export_gate
```

call:

```python
    final_coordinate_audit = _apply_final_project_coordinate_audit(project_data)
    project_data.setdefault("qa", {}).setdefault("summary", {})["final_coordinate_audit"] = final_coordinate_audit
```

This ensures `evaluate_export_gate(project_data)` sees any critical coordinate flags.

- [ ] **Step 3: Add propagation test**

Append to `pipeline/tests/test_qa_flag_propagation_v2.py`:

```python
def test_final_project_coordinate_audit_marks_mixed_page_texts():
    import main

    project = {
        "paginas": [
            {
                "numero": 1,
                "height": 13832,
                "width": 800,
                "text_layers": [
                    {
                        "id": "ocr_002",
                        "band_id": "page_002_band_005",
                        "band_y_top": 5420,
                        "band_height": 895,
                        "bbox": [25, 5436, 667, 5745],
                        "balloon_bbox": [466, 5606, 696, 5777],
                        "bubble_inner_bbox": [513, 230, 649, 313],
                        "safe_text_box": [525, 242, 637, 301],
                        "translated": "POR FAVOR, PELO BEM DA CRIANCA.",
                    }
                ],
            }
        ],
        "qa": {"summary": {}},
    }

    audit = main._apply_final_project_coordinate_audit(project)

    flags = project["paginas"][0]["text_layers"][0]["qa_flags"]
    assert audit["applied"] is True
    assert audit["flags_added"] > 0
    assert "page_space_rerender_mixed_coordinates" in flags
    assert "layout_bbox_coordinate_mismatch" in flags
```

- [ ] **Step 4: Run propagation tests**

Run:

```powershell
cd N:\TraduzAI\pipeline
.\venv\Scripts\python.exe -m pytest tests\test_qa_flag_propagation_v2.py tests\test_main_emit.py -q
```

Expected:

```text
passed
```

---

## Task 7: Make fast solid fill verified before it can skip real inpaint

**Files:**

- Modify: `pipeline/inpainter/__init__.py:669`
- Modify: `pipeline/inpainter/__init__.py:1056`
- Modify: `pipeline/qa/translation_qa.py`
- Modify: `pipeline/tests/test_vision_stack_inpainter.py`

- [ ] **Step 1: Add QA flags**

In `pipeline/qa/translation_qa.py`, add:

```python
    "fast_fill_unverified_residual": "critical",
    "fast_fill_insufficient_coverage": "critical",
    "text_residual_after_inpaint_confirmed": "critical",
    "text_residual_after_inpaint_suspected": "high",
    "low_inpaint_coverage": "high",
```

- [ ] **Step 2: Add a helper for text-id based verification metadata**

In `pipeline/inpainter/__init__.py`, near `_apply_fast_solid_balloon_fill`, add:

```python
def _fast_solid_verified_text_ids(ocr_page: dict) -> set[str]:
    verified: set[str] = set()
    for sample in ocr_page.get("_strip_fast_solid_fill_samples") or []:
        if not isinstance(sample, dict):
            continue
        if sample.get("fast_fill_verified") is not True:
            continue
        text_id = str(sample.get("text_id") or "").strip()
        if text_id:
            verified.add(text_id)
    return verified
```

- [ ] **Step 3: Add residual score helper**

In `pipeline/inpainter/__init__.py`, add:

```python
def _fast_fill_residual_edge_ratio(
    image_rgb: np.ndarray,
    text_bbox: list[int],
    text_fill_mask: np.ndarray,
) -> float:
    bbox = _normalize_bbox(text_bbox, image_rgb.shape[1], image_rgb.shape[0])
    if bbox is None:
        return 1.0
    x1, y1, x2, y2 = bbox
    roi = image_rgb[y1:y2, x1:x2]
    if roi.size == 0:
        return 1.0
    mask_roi = text_fill_mask[y1:y2, x1:x2] if text_fill_mask.shape[:2] == image_rgb.shape[:2] else None
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    if isinstance(mask_roi, np.ndarray) and mask_roi.size:
        dilated = cv2.dilate((mask_roi > 0).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
        region = dilated > 0
    else:
        region = np.ones(edges.shape, dtype=bool)
    region_pixels = int(np.count_nonzero(region))
    if region_pixels <= 0:
        return 1.0
    return float(np.count_nonzero((edges > 0) & region) / float(region_pixels))
```

This is a cheap residual signal, not the only quality gate. The coverage gate in Step 4 also matters.

- [ ] **Step 4: Add coverage and residual verification in `_apply_fast_solid_balloon_fill`**

Inside `_apply_fast_solid_balloon_fill`, after `changed_in_text` is computed and before accepting the fill sample, add:

```python
        text_bbox_area = max(1, (int(text_bbox[2]) - int(text_bbox[0])) * (int(text_bbox[3]) - int(text_bbox[1])))
        changed_coverage = changed_in_text / float(text_bbox_area)
        min_coverage = float(os.environ.get("TRADUZAI_FAST_FILL_MIN_COVERAGE", "0.18"))
        if changed_coverage < min_coverage:
            _append_inpaint_decision_flag(ocr_page, "fast_fill_insufficient_coverage")
            _reject("fast_fill_insufficient_coverage")
            continue

        residual_ratio = _fast_fill_residual_edge_ratio(filled_from_original, text_bbox, text_fill_mask)
        max_residual = float(os.environ.get("TRADUZAI_FAST_FILL_MAX_RESIDUAL_EDGE_RATIO", "0.08"))
        if residual_ratio > max_residual:
            _append_inpaint_decision_flag(ocr_page, "text_residual_after_inpaint_suspected")
            _reject("text_residual_after_inpaint_suspected")
            continue

        metadata["fast_fill_verified"] = True
        metadata["fast_fill_text_bbox_coverage"] = round(changed_coverage, 4)
        metadata["fast_fill_residual_edge_ratio"] = round(residual_ratio, 4)
```

Do not make the first threshold too aggressive. The initial value must avoid turning every small text into real inpaint.

- [ ] **Step 5: Prevent block removal unless verified**

Before building `remaining_blocks`, compute:

```python
    verified_text_ids = {
        str(sample.get("text_id") or "").strip()
        for sample in fill_samples
        if isinstance(sample, dict) and sample.get("fast_fill_verified") is True
    }
```

Change the `remaining_blocks` list comprehension into a loop:

```python
    remaining_blocks = []
    for block in vision_blocks:
        block_id = str(block.get("id") or block.get("text_id") or block.get("trace_id") or "").strip()
        if block_id and block_id not in verified_text_ids:
            remaining_blocks.append(block)
            continue
        if _block_is_covered_by_fast_fill(block, filled_bboxes, width, height, filled_mask):
            continue
        remaining_blocks.append(block)
```

If block ids use `trace_id` while samples use `text_id`, compare also by the suffix before `@`:

```python
        block_short_id = block_id.split("@", 1)[0]
        if block_id and block_id not in verified_text_ids and block_short_id not in verified_text_ids:
            remaining_blocks.append(block)
            continue
```

- [ ] **Step 6: Add tests for verified fast solid**

Append to `pipeline/tests/test_vision_stack_inpainter.py`:

```python
def test_fast_solid_fill_records_verified_sample_for_clean_white_balloon(monkeypatch):
    import numpy as np
    from inpainter import _apply_fast_solid_balloon_fill

    image = np.full((120, 240, 3), 255, dtype=np.uint8)
    image[50:60, 80:150] = 0
    page = {
        "texts": [
            {
                "id": "ocr_001",
                "tipo": "fala",
                "text_pixel_bbox": [80, 50, 150, 60],
                "bbox": [70, 40, 170, 80],
                "balloon_bbox": [60, 30, 180, 90],
                "line_polygons": [[[80, 50], [150, 50], [150, 60], [80, 60]]],
                "layout_profile": "white_balloon",
            }
        ]
    }
    blocks = [dict(page["texts"][0])]
    monkeypatch.setenv("TRADUZAI_STRIP_FAST_SOLID_INPAINT", "1")

    result, remaining, stats = _apply_fast_solid_balloon_fill(image, page, blocks)

    assert stats["solid_balloon_count"] == 1
    assert remaining == []
    assert page["_strip_fast_solid_fill_samples"][0]["fast_fill_verified"] is True
    assert np.any(result != image)
```

- [ ] **Step 7: Add test for low coverage fallback**

Append:

```python
def test_fast_solid_fill_keeps_block_when_coverage_is_too_low(monkeypatch):
    import numpy as np
    from inpainter import _apply_fast_solid_balloon_fill

    image = np.full((160, 300, 3), 255, dtype=np.uint8)
    image[80:82, 140:145] = 0
    page = {
        "texts": [
            {
                "id": "ocr_001",
                "tipo": "fala",
                "text_pixel_bbox": [60, 50, 240, 120],
                "bbox": [50, 40, 250, 130],
                "balloon_bbox": [40, 30, 260, 140],
                "line_polygons": [[[140, 80], [145, 80], [145, 82], [140, 82]]],
                "layout_profile": "white_balloon",
            }
        ]
    }
    blocks = [dict(page["texts"][0])]
    monkeypatch.setenv("TRADUZAI_STRIP_FAST_SOLID_INPAINT", "1")
    monkeypatch.setenv("TRADUZAI_FAST_FILL_MIN_COVERAGE", "0.50")

    _, remaining, stats = _apply_fast_solid_balloon_fill(image, page, blocks)

    assert stats["solid_balloon_count"] == 0
    assert len(remaining) == 1
    assert "fast_fill_insufficient_coverage" in page.get("_strip_inpaint_decision_flags", []) or page.get("_strip_fast_solid_rejection_reasons")
```

If the internal flag list name differs, assert the rejection reason in `_strip_fast_solid_rejection_reasons`.

- [ ] **Step 8: Run inpaint tests**

Run:

```powershell
cd N:\TraduzAI\pipeline
.\venv\Scripts\python.exe -m pytest tests\test_vision_stack_inpainter.py -q
```

Expected:

```text
passed
```

---

## Task 8: Make export/debug artifacts reflect visual blockers

**Files:**

- Modify: `pipeline/qa/export_gate.py`
- Modify: `pipeline/main.py`
- Modify: `pipeline/tests/test_export_gate.py`
- Modify: `pipeline/tests/test_main_emit.py`

- [ ] **Step 1: Confirm export gate blocks critical flags**

The export gate already blocks `severity == "critical"`. Keep this behavior. Do not make every high issue block.

- [ ] **Step 2: Add linked artifacts for coordinate blockers**

In `pipeline/qa/export_gate.py`, when creating `base_issue`, ensure these fields are present:

```python
                "bbox": layer.get("bbox") or layer.get("layout_bbox") or layer.get("source_bbox"),
                "source_bbox": layer.get("source_bbox") or layer.get("bbox"),
                "balloon_bbox": layer.get("balloon_bbox"),
                "safe_text_box": layer.get("safe_text_box") or layer.get("_debug_safe_text_box"),
                "render_bbox": layer.get("render_bbox"),
                "trace_id": trace_id,
                "band_id": band_id,
```

If already present, do not duplicate.

For critical coordinate flags, add:

```python
                        "linked_artifacts": [
                            "05_layout_geometry/bbox_coordinate_audit.json",
                            "09_typeset/render_plan_final.jsonl",
                        ],
```

- [ ] **Step 3: Add test for linked artifacts**

Append to `pipeline/tests/test_export_gate.py`:

```python
def test_export_gate_coordinate_blocker_links_debug_artifacts():
    from qa.export_gate import evaluate_export_gate

    project = {
        "paginas": [
            {
                "numero": 1,
                "text_layers": [
                    {
                        "id": "ocr_002",
                        "band_id": "page_002_band_005",
                        "trace_id": "ocr_002@page_002_band_005",
                        "translated": "POR FAVOR, PELO BEM DA CRIANCA.",
                        "qa_flags": ["page_space_rerender_mixed_coordinates"],
                        "bbox": [542, 246, 620, 296],
                        "balloon_bbox": [466, 5606, 696, 5777],
                        "safe_text_box": [525, 242, 637, 301],
                        "render_bbox": [542, 246, 620, 296],
                    }
                ],
            }
        ]
    }

    gate = evaluate_export_gate(project)
    issue = gate["issues"][0]

    assert gate["status"] == "BLOCK"
    assert "05_layout_geometry/bbox_coordinate_audit.json" in issue.get("linked_artifacts", [])
    assert issue["band_id"] == "page_002_band_005"
    assert issue["trace_id"] == "ocr_002@page_002_band_005"
```

- [ ] **Step 4: Run export gate tests**

Run:

```powershell
cd N:\TraduzAI\pipeline
.\venv\Scripts\python.exe -m pytest tests\test_export_gate.py tests\test_main_emit.py -q
```

Expected:

```text
passed
```

---

## Task 9: Add minimum trace integrity for grouped render blocks

**Files:**

- Modify: `pipeline/typesetter/renderer.py`
- Modify: `pipeline/main.py`
- Modify: `pipeline/tests/test_render_plan_trace_integrity.py`

This task does not change editor schema yet. It adds enough trace data to debug grouped/connected text without introducing a new `render_group` schema.

- [ ] **Step 1: Add test for grouped source trace ids**

Append to `pipeline/tests/test_render_plan_trace_integrity.py`:

```python
def test_render_plan_for_grouped_text_keeps_source_trace_ids():
    from typesetter.renderer import build_render_blocks

    page = {
        "texts": [
            {
                "id": "ocr_001",
                "trace_id": "ocr_001@page_003_band_035",
                "translated": "ESTOU MORRENDO DE FOME",
                "bbox": [343, 571, 540, 680],
                "balloon_bbox": [343, 571, 540, 680],
                "line_polygons": [[[343, 571], [540, 571], [540, 600], [343, 600]]],
            },
            {
                "id": "ocr_003",
                "trace_id": "ocr_003@page_003_band_035",
                "translated": "HOJE?",
                "bbox": [389, 644, 496, 669],
                "balloon_bbox": [343, 571, 540, 680],
                "line_polygons": [[[389, 644], [496, 644], [496, 669], [389, 669]]],
            },
        ]
    }

    blocks = build_render_blocks(page)

    joined = [block for block in blocks if "source_trace_ids" in block]
    assert joined
    assert any("ocr_001@page_003_band_035" in block["source_trace_ids"] for block in joined)
    assert any("ocr_003@page_003_band_035" in block["source_trace_ids"] for block in joined)
```

Adapt to the exact existing `build_render_blocks` signature if needed.

- [ ] **Step 2: Preserve `source_trace_ids` when consolidating blocks**

In `pipeline/typesetter/renderer.py`, wherever consolidated render blocks are created, ensure:

```python
block["source_trace_ids"] = sorted(
    {
        str(item.get("trace_id") or "")
        for item in source_texts
        if str(item.get("trace_id") or "")
    }
)
block["source_text_ids"] = sorted(
    {
        str(item.get("id") or item.get("text_id") or "")
        for item in source_texts
        if str(item.get("id") or item.get("text_id") or "")
    }
)
```

If the renderer already creates `source_trace_ids`, only add tests and fix missing propagation.

- [ ] **Step 3: Ensure render plan final writes trace ids**

In `pipeline/main.py`, where `render_plan_final.jsonl` is rebuilt from project/render data, ensure each row keeps:

```python
"source_trace_ids": layer.get("source_trace_ids") or [],
"source_text_ids": layer.get("source_text_ids") or [],
```

- [ ] **Step 4: Run trace tests**

Run:

```powershell
cd N:\TraduzAI\pipeline
.\venv\Scripts\python.exe -m pytest tests\test_render_plan_trace_integrity.py -q
```

Expected:

```text
passed
```

---

## Task 10: Focused integration test on saved Chapter 1 debug artifact

**Files:**

- Create: `pipeline/tests/test_final_geometry_contract.py`
- Uses artifact path only if present:
  - `N:/TraduzAI/DEBUGM/runs/2026-05-24_task10_real_validation_20260524_174315/chapter1_full/project.json`

- [ ] **Step 1: Create optional regression test**

Create `pipeline/tests/test_final_geometry_contract.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest


ARTIFACT_PROJECT = Path(
    r"N:/TraduzAI/DEBUGM/runs/2026-05-24_task10_real_validation_20260524_174315/chapter1_full/project.json"
)


def _bbox(value):
    if not isinstance(value, list) or len(value) < 4:
        return None
    return [int(v) for v in value[:4]]


@pytest.mark.skipif(not ARTIFACT_PROJECT.exists(), reason="local debug artifact not present")
def test_known_bad_chapter1_project_contains_mixed_coordinate_evidence():
    project = json.loads(ARTIFACT_PROJECT.read_text(encoding="utf-8"))
    offenders = []
    for page in project.get("paginas") or []:
        for text in page.get("text_layers") or []:
            if text.get("skip_processing"):
                continue
            target = _bbox(text.get("balloon_bbox") or text.get("bbox"))
            safe = _bbox(text.get("safe_text_box") or text.get("bubble_inner_bbox"))
            if not target or not safe:
                continue
            if target[1] > 1000 and safe[1] < 900:
                offenders.append((text.get("trace_id"), target, safe))
    assert offenders, "fixture should remain useful as known-bad coordinate evidence"
```

This test does not assert the current pipeline output; it preserves the old bad artifact as evidence when available.

- [ ] **Step 2: Run optional regression test**

Run:

```powershell
cd N:\TraduzAI\pipeline
.\venv\Scripts\python.exe -m pytest tests\test_final_geometry_contract.py -q
```

Expected:

```text
passed
```

or:

```text
skipped
```

if the artifact is not present.

---

## Task 11: Run focused test suite

**Files:**

- No code changes.

- [ ] **Step 1: Run PR-1/PR-2 tests**

Run:

```powershell
cd N:\TraduzAI\pipeline
.\venv\Scripts\python.exe -m pytest `
  tests\test_strip_balloon_bbox_propagation.py `
  tests\test_derived_bbox_coordinate_audit.py `
  tests\test_page_space_rerender_guard.py `
  tests\test_typesetting_renderer.py `
  tests\test_export_gate.py `
  -q
```

Expected:

```text
passed
```

- [ ] **Step 2: Run inpaint/QA tests**

Run:

```powershell
cd N:\TraduzAI\pipeline
.\venv\Scripts\python.exe -m pytest `
  tests\test_vision_stack_inpainter.py `
  tests\test_qa_flag_propagation_v2.py `
  tests\test_render_plan_trace_integrity.py `
  tests\test_final_geometry_contract.py `
  -q
```

Expected:

```text
passed
```

- [ ] **Step 3: Run smoke suite with fail-fast**

Run:

```powershell
cd N:\TraduzAI\pipeline
.\venv\Scripts\python.exe -m pytest tests -q -x
```

Expected:

```text
passed
```

If this is too slow during implementation, run it after the E2E Chapter 1 validation.

---

## Task 12: Run Chapter 1 E2E debug validation

**Files:**

- No source change.
- Output directory:
  - `N:/TraduzAI/DEBUGM/runs/2026-05-25_chapter1_geometry_inpaint_typeset_v4`

- [ ] **Step 1: Create a runner config**

Create a config JSON under the new run folder, based on the previous debug run:

```json
{
  "source_path": "C:/Users/PICHAU/Downloads/Chapter 1",
  "work_dir": "N:/TraduzAI/DEBUGM/runs/2026-05-25_chapter1_geometry_inpaint_typeset_v4/chapter1_full",
  "models_dir": "N:/TraduzAI/pipeline/models",
  "logs_dir": "N:/TraduzAI/DEBUGM/runs/2026-05-25_chapter1_geometry_inpaint_typeset_v4/chapter1_full/logs",
  "job_id": "chapter1_geometry_inpaint_typeset_v4",
  "run_id": "chapter1_geometry_inpaint_typeset_v4",
  "obra": "chapter1_geometry_inpaint_typeset_v4",
  "capitulo": 1,
  "idioma_origem": "en",
  "idioma_destino": "pt-BR",
  "mode": "auto",
  "debug": true,
  "strict": false,
  "export_mode": "debug",
  "skip_inpaint": false,
  "skip_ocr": false,
  "strip_target_pages": 6,
  "engine_preset_id": "default",
  "runtime_profile": "balanced"
}
```

Use PowerShell to write it:

```powershell
$run = "N:\TraduzAI\DEBUGM\runs\2026-05-25_chapter1_geometry_inpaint_typeset_v4\chapter1_full"
New-Item -ItemType Directory -Force -Path $run | Out-Null
@'
{
  "source_path": "C:/Users/PICHAU/Downloads/Chapter 1",
  "work_dir": "N:/TraduzAI/DEBUGM/runs/2026-05-25_chapter1_geometry_inpaint_typeset_v4/chapter1_full",
  "models_dir": "N:/TraduzAI/pipeline/models",
  "logs_dir": "N:/TraduzAI/DEBUGM/runs/2026-05-25_chapter1_geometry_inpaint_typeset_v4/chapter1_full/logs",
  "job_id": "chapter1_geometry_inpaint_typeset_v4",
  "run_id": "chapter1_geometry_inpaint_typeset_v4",
  "obra": "chapter1_geometry_inpaint_typeset_v4",
  "capitulo": 1,
  "idioma_origem": "en",
  "idioma_destino": "pt-BR",
  "mode": "auto",
  "debug": true,
  "strict": false,
  "export_mode": "debug",
  "skip_inpaint": false,
  "skip_ocr": false,
  "strip_target_pages": 6,
  "engine_preset_id": "default",
  "runtime_profile": "balanced"
}
'@ | Set-Content -LiteralPath "$run\runner_config.json" -Encoding UTF8
```

- [ ] **Step 2: Run the pipeline**

Run:

```powershell
cd N:\TraduzAI
$env:TRADUZAI_STRIP_FAST_SOLID_INPAINT="1"
$env:TRADUZAI_STRIP_FAST_WHITE_INPAINT="0"
$env:TRADUZAI_STRIP_FAST_DARK_PANEL_FILL="0"
$env:TRADUZAI_PAGE_CLEANUP_RERENDER="0"
$env:TRADUZAI_PADDLE_FULL_PAGE="1"
$env:TRADUZAI_STRIP_DETECT_FULL_PAGE="1"
.\pipeline\venv\Scripts\python.exe .\pipeline\main.py "N:\TraduzAI\DEBUGM\runs\2026-05-25_chapter1_geometry_inpaint_typeset_v4\chapter1_full\runner_config.json"
```

Expected:

```text
exit code 0
```

If export gate becomes `BLOCK`, inspect whether the block is real. A real block is acceptable in debug if visual output still has confirmed issues; do not weaken the gate just to get PASS.

- [ ] **Step 3: Check coordinate audit**

Run:

```powershell
$audit = Get-Content "N:\TraduzAI\DEBUGM\runs\2026-05-25_chapter1_geometry_inpaint_typeset_v4\chapter1_full\debug\e2e\05_layout_geometry\bbox_coordinate_audit.json" | ConvertFrom-Json
$audit.summary | Format-List
```

Expected:

```text
all_consistent : True
```

If not true, inspect `findings`.

- [ ] **Step 4: Check export gate**

Run:

```powershell
$gate = Get-Content "N:\TraduzAI\DEBUGM\runs\2026-05-25_chapter1_geometry_inpaint_typeset_v4\chapter1_full\debug\e2e\11_qa_export_gate\export_gate.json" | ConvertFrom-Json
$gate.status
$gate.critical_issue_count
```

Expected:

```text
PASS
0
```

If `BLOCK`, inspect:

```powershell
Get-Content "N:\TraduzAI\DEBUGM\runs\2026-05-25_chapter1_geometry_inpaint_typeset_v4\chapter1_full\debug\e2e\11_qa_export_gate\visual_blockers.jsonl"
```

- [ ] **Step 5: Generate a final contact sheet**

Use a small Python helper:

```powershell
cd N:\TraduzAI
@'
from pathlib import Path
from PIL import Image, ImageDraw

root = Path(r"N:/TraduzAI/DEBUGM/runs/2026-05-25_chapter1_geometry_inpaint_typeset_v4/chapter1_full")
imgs = sorted((root / "translated").glob("*.jpg"))
thumbs = []
for path in imgs:
    im = Image.open(path).convert("RGB")
    im.thumbnail((260, 900))
    canvas = Image.new("RGB", (280, im.height + 30), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), path.name, fill=(0, 0, 0))
    canvas.paste(im, (10, 28))
    thumbs.append(canvas)
w = sum(t.width for t in thumbs)
h = max(t.height for t in thumbs)
sheet = Image.new("RGB", (w, h), "white")
x = 0
for t in thumbs:
    sheet.paste(t, (x, 0))
    x += t.width
out = root / "debug" / "translated_contact_sheet_v4.jpg"
out.parent.mkdir(parents=True, exist_ok=True)
sheet.save(out, quality=92)
print(out)
'@ | .\pipeline\venv\Scripts\python.exe -
```

Expected:

```text
N:\TraduzAI\DEBUGM\runs\2026-05-25_chapter1_geometry_inpaint_typeset_v4\chapter1_full\debug\translated_contact_sheet_v4.jpg
```

- [ ] **Step 6: Manual visual checks**

Open and inspect:

```text
N:/TraduzAI/DEBUGM/runs/2026-05-25_chapter1_geometry_inpaint_typeset_v4/chapter1_full/translated/001.jpg
N:/TraduzAI/DEBUGM/runs/2026-05-25_chapter1_geometry_inpaint_typeset_v4/chapter1_full/translated/002.jpg
N:/TraduzAI/DEBUGM/runs/2026-05-25_chapter1_geometry_inpaint_typeset_v4/chapter1_full/translated/003.jpg
N:/TraduzAI/DEBUGM/runs/2026-05-25_chapter1_geometry_inpaint_typeset_v4/chapter1_full/translated/006.jpg
```

Must pass:

```text
[ ] "POR FAVOR, PELO BEM DA CRIANCA." is in the child balloon, not at page top.
[ ] "ESTOU MORRENDO..." has no visible English residue.
[ ] Connected balloon has no visible "SOME." residue.
[ ] Blue/black title boxes do not keep English/source text under PT text.
[ ] No text is rendered over character art due to local safe box.
```

---

## Task 13: Run short regression matrix

**Files:**

- No source change.
- Output root:
  - `N:/TraduzAI/DEBUGM/runs/2026-05-25_geometry_inpaint_typeset_v4_matrix`

- [ ] **Step 1: Run Articuno Chapter 61**

Source:

```text
C:/Users/PICHAU/Downloads/Articuno (comick)_Ch. 61 OFFICIAL TRANSLATION
```

Check:

```text
[ ] spiked white narration balloons keep black spikes/outline.
[ ] rectangular narration boxes preserve border.
[ ] connected/overlapping balloons keep text inside each lobe/balloon.
```

- [ ] **Step 2: Run Chapter 39**

Source:

```text
C:/Users/PICHAU/Downloads/Chapter 39
```

Check:

```text
[ ] rotated/inclined text still renders with intended angle.
[ ] signs/papers do not get text positioned from wrong coordinate space.
```

- [ ] **Step 3: Run God of Death Chapter 2**

Source:

```text
D:/Mihon pra pc/downloads/mangas/Manhwatop (EN)/The God of Death/Chapter 2.cbz
```

Check:

```text
[ ] black/dark panels do not keep English under PT text.
[ ] glow title boxes still preserve glow/background.
```

- [ ] **Step 4: Run first two chapters of 1 Second**

Source root:

```text
D:/Mihon pra pc/downloads/mangas/Manhwatop (EN)/1 Second
```

Check:

```text
[ ] white/colored balloons use sampled solid fill, not hardcoded white.
[ ] no character face is masked as text.
[ ] top/bottom page filters only affect intended first/last-page regions.
```

- [ ] **Step 5: Summarize matrix**

Create:

```text
N:/TraduzAI/DEBUGM/runs/2026-05-25_geometry_inpaint_typeset_v4_matrix/summary.md
```

Include:

```markdown
# Geometry Inpaint Typeset V4 Matrix Summary

| Run | Exit | Export Gate | Critical Issues | Manual Visual Result | Notes |
|---|---:|---|---:|---|---|
| chapter1 | 0 | PASS/BLOCK | N | pass/fail | ... |
| articuno_ch61 | 0 | PASS/BLOCK | N | pass/fail | ... |
| chapter39 | 0 | PASS/BLOCK | N | pass/fail | ... |
| god_of_death_ch2 | 0 | PASS/BLOCK | N | pass/fail | ... |
| one_second_ch1 | 0 | PASS/BLOCK | N | pass/fail | ... |
| one_second_ch2 | 0 | PASS/BLOCK | N | pass/fail | ... |
```

---

## Task 14: Update debug documentation

**Files:**

- Modify: `docs/debug/e2e_pipeline_debug_guide.md`

- [ ] **Step 1: Add coordinate mismatch section**

Add:

```markdown
## Diagnosing Mixed Coordinate Space

If final `translated/*.jpg` shows text at the top of the page or over character art while the target balloon is empty, inspect:

- `debug/e2e/05_layout_geometry/bbox_coordinate_audit.json`
- `debug/e2e/09_typeset/render_plan_final.jsonl`
- `project.json`

The following fields must be in the same page coordinate space after reassembly:

- `bbox`
- `text_pixel_bbox`
- `balloon_bbox`
- `bubble_mask_bbox`
- `bubble_inner_bbox`
- `safe_text_box`
- `render_bbox`
- `_debug_safe_text_box`
- `_render_debug.*bbox`

Confirmed blockers:

- `layout_bbox_coordinate_mismatch`
- `page_space_rerender_mixed_coordinates`
- `bubble_inner_bbox_coordinate_mismatch`
```

- [ ] **Step 2: Add fast solid verification section**

Add:

```markdown
## Diagnosing Fast Solid Fill Residue

If `debug_inpaint/<band>/metadata.json` shows:

- `used_fast_solid_fill: true`
- `used_real_inpaint: false`
- `raw_mask_pixels: 0`
- `expanded_mask_pixels: 0`

then fast fill must also show verified metadata:

- `fast_fill_verified: true`
- `fast_fill_text_bbox_coverage`
- `fast_fill_residual_edge_ratio`

If verification fails, the block must remain eligible for real inpaint and export gate must surface a visual blocker.
```

- [ ] **Step 3: Documentation check**

Run:

```powershell
Select-String -LiteralPath N:\TraduzAI\docs\debug\e2e_pipeline_debug_guide.md -Pattern "Mixed Coordinate Space","Fast Solid Fill Residue"
```

Expected:

```text
Both sections found.
```

---

## Task 15: Final verification checklist

**Files:**

- No source change.

- [ ] **Step 1: Check forbidden implementation pattern is absent**

Run:

```powershell
Select-String -LiteralPath N:\TraduzAI\pipeline\strip\run.py -Pattern '_coordinate_space.*==.*page'
```

Expected:

```text
No generic early-return in _shift_text_geometry_y/_shift_text_geometry_xy.
```

If there is a match, inspect it. It must not skip required shifts.

- [ ] **Step 2: Check new bbox keys are in the right places**

Run:

```powershell
Select-String -LiteralPath N:\TraduzAI\pipeline\strip\run.py -Pattern 'bubble_inner_bbox','bubble_mask_bbox','balloon_inner_bbox'
Select-String -LiteralPath N:\TraduzAI\pipeline\debug_tools\bbox.py -Pattern 'bubble_inner_bbox','bubble_mask_bbox','balloon_inner_bbox'
```

Expected:

```text
All three fields appear in shift helpers and BBOX_KEYS.
```

- [ ] **Step 3: Check export gate critical behavior**

Run:

```powershell
cd N:\TraduzAI\pipeline
.\venv\Scripts\python.exe -m pytest tests\test_export_gate.py::test_export_gate_blocks_page_space_rerender_mixed_coordinates -q
```

Expected:

```text
passed
```

- [ ] **Step 4: Check final E2E artifacts**

For the Chapter 1 validation run:

```text
[ ] bbox_coordinate_audit.json has all_consistent true, or export gate is BLOCK with real visual blockers.
[ ] visual_blockers.jsonl is empty only when manual visual review confirms no blocker.
[ ] qa_report.json does not claim clean PASS when critical blockers exist.
[ ] translated contact sheet visually matches expected balloon placement.
```

---

## Implementation Order

1. Task 1: failing tests for geometry flow.
2. Task 2: shift `bubble_*` fields without unsafe idempotency.
3. Task 3: coordinate audit catches `bubble_*` and stale safe/render boxes.
4. Task 4: final rerender guard marks critical blockers and avoids overwriting bad pages.
5. Task 5: renderer rejects stale safe areas.
6. Task 6: final project audit before export gate.
7. Task 7: verified fast solid fill and fallback to real inpaint.
8. Task 8: export/debug artifacts show real blockers.
9. Task 9: minimum render trace integrity for grouped/connected text.
10. Task 10-13: focused tests, Chapter 1 E2E, regression matrix.
11. Task 14-15: docs and final verification.

## Done Criteria

The work is complete only when all of these are true:

```text
[ ] No generic _coordinate_space == "page" early-return exists in geometry shift helpers.
[ ] bubble_mask_bbox, bubble_inner_bbox, balloon_inner_bbox are shifted by y and xy helpers.
[ ] bbox_coordinate_audit catches old Chapter 1 mixed-coordinate pattern.
[ ] final page-space rerender refuses to overwrite pages with confirmed mixed coordinates.
[ ] export gate blocks page_space_rerender_mixed_coordinates.
[ ] renderer ignores stale safe_text_box/bubble_inner_bbox outside target balloon.
[ ] fast_solid_fill records fast_fill_verified before removing a block from real inpaint.
[ ] residual fast-fill failures keep blocks eligible for real inpaint or block export.
[ ] Chapter 1 E2E no longer has text rendered at y-local over art.
[ ] Chapter 1 E2E no longer has obvious English residue in the targeted balloons.
[ ] short regression matrix has contact sheets and a written summary.
```

## Known Risks And Mitigations

1. Risk: adding `bubble_*` shift changes tests that accidentally assumed local coordinates after final reassembly.
   - Mitigation: update those tests to name coordinate space explicitly. Local band tests should run before reassembly; final page tests should use page-local coordinates.

2. Risk: renderer safe-box rejection may be too aggressive for cropped page-cleanup renders.
   - Mitigation: test page-cleanup crop paths. The check is intersection with `target_bbox`, not a strict distance threshold.

3. Risk: fast solid verification increases runtime.
   - Mitigation: use cheap coverage and edge checks first. Real inpaint only runs when fast fill is not verified.

4. Risk: export gate blocks more debug runs.
   - Mitigation: this is intended for confirmed visual blockers. Do not downgrade `page_space_rerender_mixed_coordinates`.

5. Risk: render group trace work becomes schema migration.
   - Mitigation: Task 9 only preserves existing `source_trace_ids`/`source_text_ids` in debug/project data. Full editor schema changes are explicitly out of this v4 plan.

## Self Review

1. Spec coverage: the plan covers geometry shift, coordinate audit, final rerender guard, renderer safe-box validation, verified fast solid fill, QA/export gate blockers, render traceability, Chapter 1 E2E, matrix validation, and debug documentation.

2. Forbidden idempotency check: the plan explicitly rejects generic `_coordinate_space == "page"` early-return behavior in shift helpers. The implementation must prove the second shift still runs with `test_shift_text_geometry_y_does_not_skip_required_second_shift`.

3. Audit/gate coverage: the plan requires `bubble_mask_bbox`, `bubble_inner_bbox`, and `balloon_inner_bbox` in the shift helpers and `BBOX_KEYS`, then runs a post-final/project audit before export gate.

4. Visual quality coverage: fast solid fill cannot suppress real inpaint unless coverage and residual checks mark the fill as verified; otherwise blocks remain eligible for real inpaint or become export blockers.

5. Validation coverage: the plan does not stop at unit tests; it requires Chapter 1 debug validation, contact sheet review, and a short multi-work regression matrix before calling the work complete.

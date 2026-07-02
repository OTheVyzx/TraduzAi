# Dark Bubble Style Layout Inpaint SFX Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make dark/black speech bubbles behave like the already-good white bubble path: consistent visual style across similar bubbles, text fitted inside its own balloon/lobe, glyph-only inpaint, and SFX preserved instead of translated as dialogue.

**Architecture:** Split the correction into four contracts: SFX preservation before OCR/text routing, glyph-only inpaint masks, grouped dark-bubble style consistency, and final render QA that refuses clipped/overflowing text. The renderer may increase or decrease font size, but the accepted render must remain inside the resolved safe area of the specific bubble/lobe.

**Tech Stack:** Python 3.12, OpenCV/Numpy mask logic, matplotlib FT2Font renderer, existing TraduzAI debug artifacts under `DEBUGM/runs`, pytest, visual crop/contact sheets generated in `.codex-tmp`.

---

## Checkpoint fix12 freeze - 2026-07-02

Status: resolved and frozen under tag `checkpoint-god-of-death-ch2-fix12`.

Validation source:
- main checkpoint: `6ee11a8ba6501b9f9d47434210f6748332ea333f`
- fix12 OCR checkpoint commit: `894f2ad7301724df71303de2a2f05b8fbe5a9c1b`
- finalguard after rerender commit: `36801e80`
- typeset contract bbox commit: `d6ba5d5140ebd5e9b21694a7cf494bf4358686ee`
- run: `N:\TraduzAI_fix12_freeze_main\DEBUGM\runs\god_of_death_ch2_fix12_freeze_finalguard_typeset_contract_20260702`
- sheet: `N:\TraduzAI_fix12_freeze_main\.codex-tmp\fix12_freeze_validation\fix12_freeze_finalguard_typeset_contract_sheet.jpg`

Audits:
- `translated_page_band_consistency_audit.json`: `passed=true`, `rows_failed=0`
- `baseline_vs_candidate_visual_regression_audit.json`: `passed=true`, `rows_failed=0`
- `final_band_crops_refresh.json`: `final_guard_ran_after_final_project_image_rerender=true`, `final_output_source=clean_final_bands_after_all_rerenders`, `clean_band_source_used=105`, `translated_crop_fallback_used=0`

Resolved issue:
- `page_004_band_056` residual English (`There can't be two kings in an underworld!`) is resolved by the fix12 checkpoint and remains covered by the finalguard/typeset freeze validation.
- `page_004_band_055` companion composite-lobe residual is also covered by the same validation.

Freeze note:
- Do not change the approved guards in follow-up work unless a new task explicitly opens a separate branch from this checkpoint.
- New visual issues should use this tag as baseline.

---

### Task 1: Baseline Visual Evidence Pack

**Files:**
- Create: `.codex-tmp/dark_bubble_goal_cases.json`
- Create: `.codex-tmp/dark_bubble_goal_baseline_sheet.jpg`
- Read: `N:\TraduzAI\DEBUGM\runs\god_of_death_ch2_source_size_rerender_20260625\project.json`
- Read: `N:\TraduzAI\DEBUGM\runs\god_of_death_ch2_source_size_rerender_20260625\translated\*.jpg`
- Read: `N:\TraduzAI\DEBUGM\runs\god_of_death_ch2_source_size_rerender_20260625\debug\e2e\09_typeset\render_plan_candidates.jsonl`

**Step 1: Define fixed regression cases**

Create a JSON manifest with at least these cases:

```json
[
  {"id": "dark_connected_lobes_system", "page": 2, "bands": ["page_002_band_013"], "expectation": "each lobe centered independently, same style"},
  {"id": "dark_single_criteria", "page": 2, "bands": ["page_002_band_007"], "expectation": "text fits inside oval, not tiny, no overflow"},
  {"id": "dark_single_reward", "page": 2, "bands": ["page_002_band_008"], "expectation": "text uses bubble area without leaving it"},
  {"id": "dark_points", "page": 2, "bands": ["page_002_band_018"], "expectation": "short text remains readable and centered"},
  {"id": "dark_subspace_retention", "page": 5, "bands": ["page_005_band_078"], "expectation": "left/right lobes independent, no black rectangle fill"},
  {"id": "dark_rect_move", "page": 5, "bands": ["page_005_band_095"], "expectation": "rect panel stays rectangular, inpaint only text"},
  {"id": "sfx_vertical_blue", "page": 3, "bands": ["page_003_band_029"], "expectation": "SFX preserved, not translated/inpainted as dialogue"},
  {"id": "white_balloon_guard", "page": 5, "bands": ["page_005_band_071"], "expectation": "white bubble uses white bubble rules only"}
]
```

**Step 2: Generate baseline sheet**

Run a small Python script that crops the translated pages and final band/debug images for each manifest item into one sheet.

Expected: a sheet showing the current failures: inconsistent style, overflow, tiny text, black rectangle inpaint, and SFX processed as text.

**Step 3: Record measurable baseline**

For every case, extract from `render_plan_candidates.jsonl`:

```text
band_id, text_id, selected, target_bbox, safe_text_box, render_bbox, font_size, qa_flags, style_source, style_group_id
```

Expected: baseline file proves which selected render path created each visible problem.

---

### Task 2: SFX Preservation Gate

**Files:**
- Modify: `pipeline/vision_stack/runtime.py`
- Modify: `pipeline/project_writer.py`
- Test: `pipeline/tests/test_vision_stack_runtime.py`
- Test: `pipeline/tests/test_sfx_inpaint_gate.py`
- Test: `pipeline/tests/test_sfx_visual_detector.py`

**Step 1: Write failing SFX preservation test**

Add a test with a tall, blue/cyan stylized SFX candidate near a white dialogue bubble. The normal OCR block overlaps the SFX candidate.

Expected behavior:

```python
kept, skipped = _drop_normal_ocr_blocks_overlapping_sfx_candidates(image, [ocr_block], [sfx_candidate])
assert kept == []
assert skipped[0]["reason"] == "english_sfx_pre_ocr_skip"
```

**Step 2: Verify current failure**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_vision_stack_runtime.py::TestVisionStackRuntime::test_pre_ocr_sfx_skip_preserves_stylized_vertical_sfx -q
```

Expected: FAIL if the current gate lets the SFX become normal text.

**Step 3: Re-enable the intended SFX preservation path**

Remove the dead `if False and ...` guards only where they disable intended SFX/noise preservation, and keep confidence/shape checks intact. Do not make raw low-confidence SFX auto-translatable.

Required contract:

```python
if preserve_cjk_sfx and should_preserve_cjk_sfx_candidate(...):
    content_class = "sfx"
    route_action = "review_required" or "preserve_original"
    translate_policy = "review"
    render_policy = "preserve_original"
    sfx["inpaint_allowed"] = False
```

**Step 4: Protect project writer semantics**

Ensure `project_writer._neutralize_removed_decision_fields()` does not convert preserved SFX into `translate_sfx_inpaint_render` unless the SFX was explicitly promoted for adaptation.

Expected: preserved SFX remains excluded from normal translate/inpaint/render.

**Step 5: Run SFX tests**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_sfx_visual_detector.py pipeline/tests/test_sfx_inpaint_gate.py pipeline/tests/test_vision_stack_runtime.py -k "sfx or SFX" -q
```

Expected: PASS, or unrelated dirty-checkout failures documented separately.

---

### Task 3: Glyph-Only Inpaint Contract

**Files:**
- Modify: `pipeline/inpainter/__init__.py`
- Modify: `pipeline/inpainter/mask_builder.py`
- Test: `pipeline/tests/test_inpaint_mask_geometry.py`
- Test: `pipeline/tests/test_mask_builder.py`
- Test: `pipeline/tests/test_vision_stack_inpainter.py`

**Step 1: Write failing glyph-only tests**

Add one dark bubble test and one dark rectangular panel test.

Expected behavior:

```python
mask = build_inpaint_mask(block, image.shape, image_rgb=image)
assert mask_pixels_are_near_original_text(mask, original_text_mask)
assert not mask_fills_safe_box(mask, safe_text_box)
assert not mask_fills_bubble_bbox(mask, bubble_mask_bbox)
```

**Step 2: Verify current failure**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_inpaint_mask_geometry.py::test_dark_bubble_inpaint_uses_expanded_glyph_mask_not_safe_box -q
```

Expected: FAIL where a dark/panel path can still use bbox/safe/bubble area.

**Step 3: Add a single inpaint permission function**

Implement one helper, for example:

```python
def _text_only_inpaint_mask_for_visual_text(image_rgb, text, current_mask):
    visual = _dark_bubble_light_text_visual_mask(image_rgb, text)
    if visual is None:
        visual = build_raw_text_mask_from_image(dict(text), image_rgb, image_rgb.shape)
    if visual is None or not np.any(visual):
        return None
    return expand_text_mask(visual.astype(np.uint8), expand_px=_visual_text_expand_px(text))
```

It may clip to the lobe only as a boundary, but must not create pixels because of lobe/safe/bbox area.

**Step 4: Block area fallback for visual text**

For `image_dark_bubble_mask`, `image_dark_panel_mask`, and `derived_card_panel_mask`, prohibit fallback masks that are created from:

```text
safe_text_box
layout_safe_bbox
bubble_mask_bbox
target_bbox
derived panel bbox
```

These boxes may be used only to clip an already-existing text/glyph mask.

**Step 5: Save debug evidence**

For every inpainted text, debug metadata must show:

```json
{
  "inpaint_mask_contract": "expanded_text_mask",
  "source_pixels": 1234,
  "expanded_pixels": 4321,
  "area_fallback_used": false
}
```

**Step 6: Run inpaint tests**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_mask_builder.py pipeline/tests/test_inpaint_mask_geometry.py pipeline/tests/test_vision_stack_inpainter.py -k "dark or glyph or mask or sfx" -q
```

Expected: dark text residual is removed without black/brown rectangles or full safe-box fills.

---

### Task 4: Similar Dark Bubble Style Group Contract

**Files:**
- Modify: `pipeline/main.py`
- Modify: `pipeline/typesetter/renderer.py`
- Modify: `pipeline/typesetter/style_policy.py`
- Test: `pipeline/tests/test_typesetting_style_policy.py`
- Test: `pipeline/tests/test_style_copy_score.py`
- Test: `pipeline/tests/test_typesetting_renderer.py`

**Step 1: Write failing style-group test**

Create layers representing visually similar dark blue glow bubbles with different detected styles.

Expected:

```python
summary = _apply_dark_panel_style_groups(project)
assert summary["layers"] >= 2
assert all(layer["style_group_id"] == first_group for layer in dark_bubble_layers)
assert all(layer["style"]["fonte"] == expected_font for layer in dark_bubble_layers)
assert all(layer["style"]["glow"] is True for layer in dark_bubble_layers)
```

**Step 2: Verify current inconsistency**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_typesetting_style_policy.py::test_dark_bubble_similar_visuals_share_style_group -q
```

Expected: FAIL if renderer fallback later changes style differently.

**Step 3: Strengthen visual signature**

Group by:

```text
shape: dark_bubble/card/white
panel fill bucket: dark/mid_dark
text fill bucket: white/cyan/warm
glow bucket: cyan/white/warm/unknown
source class: bubble vs status panel
```

Do not group white balloons with dark balloons.

**Step 4: Freeze group style against later fallback**

When `style_source == "dark_panel_visual_style_group"`, renderer fallback functions may adjust fit/layout but must not replace:

```text
fonte, cor, contorno, contorno_px, glow, glow_cor, glow_px, force_upper
```

**Step 5: Preserve source-detected style if stronger**

If one group member has `source_detected` confidence >= threshold, use it as representative. If none does, use the best sampled dark-panel effect colors.

**Step 6: Run style tests**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_typesetting_style_policy.py pipeline/tests/test_style_copy_score.py -q
```

Expected: grouped dark bubbles keep consistent style; SFX raw candidates still do not trigger style-copy unless promoted/confident.

---

### Task 5: Bubble/Lobe Fit Contract With Flexible Font Size

**Files:**
- Modify: `pipeline/typesetter/renderer.py`
- Modify: `pipeline/layout/balloon_layout.py`
- Test: `pipeline/tests/test_typesetting_renderer.py`
- Test: `pipeline/tests/test_typesetting_layout.py`
- Test: `pipeline/tests/test_balloon_layout_shared_regions.py`

**Step 1: Write failing fit tests**

Add tests for:

```text
dark oval long text can grow/shrink but render_bbox remains inside safe_text_box
short dark text may increase size if safe area supports it
font reduction is limited before wrapping is retried
connected dark lobes render each text centered in its own lobe
white balloon behavior remains unchanged
```

**Step 2: Define hard acceptance**

A candidate can be selected only when:

```python
render_bbox inside safe_text_box with small tolerance
render_bbox inside own lobe/bubble visual boundary
font_size >= min_legible_for_page_scale
no TEXT_CLIPPED qa flag
no fit_below_minimum_legible flag
```

**Step 3: Implement flexible size preference**

Use original OCR/source text size as a reference, not a cap:

```text
preferred_size = source_font_size_px
allowed_range = preferred_size * 0.75 .. preferred_size * 1.25
outside this range only if needed to avoid overflow, and mark qa_metrics
```

The renderer may increase or decrease font size, but never accepts overflow.

**Step 4: Retry layout before overshrinking**

Candidate order:

```text
same style + original center
same style + alternative wrap
same style + reduced size within limit
same style + increased safe capacity within lobe only
last resort smaller size with QA flag, not exported silently
```

**Step 5: Lobe center rule**

For connected bubbles:

```text
source_center = center(text_pixel_bbox/source_bbox of that lobe)
translated_render_center must equal source_center when possible
if it would overflow, clamp inside that lobe only
never use sibling lobe space
```

**Step 6: White bubble guard**

Add a regression test proving that `image_white_bubble_mask` and `white_balloon` do not receive dark-style capacity expansion or dark glow fallback after final classification.

**Step 7: Run layout tests**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_typesetting_renderer.py pipeline/tests/test_typesetting_layout.py pipeline/tests/test_balloon_layout_shared_regions.py -k "dark or lobe or connected or fit or white" -q
```

Expected: selected candidates fit inside their own bubble/lobe; sizes look normal because the renderer can grow or shrink while respecting bounds.

---

### Task 6: Final Rerender Parity Guard

**Files:**
- Modify: `pipeline/main.py`
- Modify: `pipeline/typesetter/renderer.py`
- Test: `pipeline/tests/test_page_space_rerender_guard.py`
- Test: `pipeline/tests/test_render_plan_trace_integrity.py`

**Step 1: Write failing rerender parity test**

Use a project layer that is correct in band-local render but wrong after page-space rerender.

Expected:

```python
assert final_project_rerender_preserves_contract(layer)
assert rerender_debug["selected"]["safe_text_box"] == layer["safe_text_box"]
assert rerender_debug["selected"]["style_group_id"] == layer["style_group_id"]
```

**Step 2: Preserve contracts into page-space render**

When copying band result into final page-space, preserve:

```text
style_group_id
style_source
target_bbox
safe_text_box
layout_safe_bbox
text_pixel_bbox/source center
lobe_id/connected_lobe metadata
inpaint_mask_contract
```

**Step 3: Reject bad final-page candidates**

The final `translated/*.jpg` renderer must not select a candidate that violates the same fit/style/inpaint contracts accepted in band debug.

**Step 4: Run rerender tests**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_page_space_rerender_guard.py pipeline/tests/test_render_plan_trace_integrity.py -q
```

Expected: final translated output matches the accepted visual contract, not a different hidden page-space decision.

---

### Task 7: Focused Chapter 2 Visual Run

**Files:**
- Create: `.codex-tmp/god_of_death_ch2_dark_contract_fix_config.json`
- Output: `DEBUGM/runs/god_of_death_ch2_dark_contract_fix_YYYYMMDD`
- Create: `.codex-tmp/dark_contract_fix_visual_sheet.jpg`

**Step 1: Run chapter 2**

Run:

```powershell
cd pipeline
python main.py N:\TraduzAI\.codex-tmp\god_of_death_ch2_dark_contract_fix_config.json
```

Expected: run completes with debug artifacts and translated pages.

**Step 2: Generate visual QA sheet**

Create one sheet with before/after crops for all baseline cases:

```text
original crop
mask overlay
inpaint overlay
final band
translated page crop
render bbox/safe bbox overlay
```

**Step 3: Manual visual acceptance**

The run is not accepted unless I visually confirm:

```text
similar dark bubbles use the same style
text is inside each bubble/lobe
font is not tiny unless original was tiny
no black/brown safe-box inpaint blocks
no English residual behind Portuguese
SFX remains preserved
white bubbles are not affected by dark rules
```

---

### Task 8: Chapter 1 Regression Check

**Files:**
- Create: `.codex-tmp/god_of_death_ch1_dark_contract_regression_config.json`
- Output: `DEBUGM/runs/god_of_death_ch1_dark_contract_regression_YYYYMMDD`
- Create: `.codex-tmp/dark_contract_ch1_regression_sheet.jpg`

**Step 1: Run chapter 1 smoke**

Run the same pipeline on chapter 1 with debug enabled.

Expected: white balloons and T/N text keep the already-good behavior.

**Step 2: Compare known Chapter 1 cases**

Check:

```text
page_002_band_002
page_002_band_014
page_003_band_035
page_003_band_046
page_004_band_054
page_006_band_107
```

Expected:

```text
white balloon mask does not cover outline
T/N text uses text-rect behavior
connected white lobes still center each text per lobe
rotated text position is unchanged
no dark bubble style rules applied to white balloons
```

---

### Task 9: Full Acceptance Gate

**Files:**
- Create: `.codex-tmp/dark_contract_acceptance_report.md`

**Step 1: Run focused tests**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest ^
  pipeline/tests/test_vision_stack_runtime.py ^
  pipeline/tests/test_sfx_inpaint_gate.py ^
  pipeline/tests/test_mask_builder.py ^
  pipeline/tests/test_inpaint_mask_geometry.py ^
  pipeline/tests/test_typesetting_style_policy.py ^
  pipeline/tests/test_typesetting_renderer.py ^
  pipeline/tests/test_typesetting_layout.py ^
  pipeline/tests/test_page_space_rerender_guard.py ^
  -q
```

**Step 2: Record unresolved failures**

If unrelated dirty-checkout tests fail, record:

```text
test name
failure reason
why it is unrelated or related
whether visual goal is blocked
```

**Step 3: Final visual pass**

Open the visual sheets and translated pages, then mark every manifest case:

```text
PASS: visually corrected
FAIL: still wrong, with exact artifact path and reason
```

**Step 4: Do not claim complete until visual pass is clean**

The goal is complete only if:

```text
chapter 2 target cases pass visually
chapter 1 regression cases pass visually
translated pages match final debug render decisions
tests for SFX, inpaint, style grouping, and fit pass or have documented unrelated failures
```


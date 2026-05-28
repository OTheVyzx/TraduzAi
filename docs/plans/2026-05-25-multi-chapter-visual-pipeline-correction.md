# Multi Chapter Visual Pipeline Correction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the recurring visual pipeline failures found across Chapter 1, Articuno ch61, Chapter 39, God of Death ch2, One Second ch1, and One Second ch2: missing OCR, untranslated text, rectangular/partial inpaint, unsafe fast fill, tiny/off-position typeset, connected-balloon mistakes, and QA/export accepting blocked output.

**Architecture:** Keep the current staged pipeline. Strengthen the contracts between OCR, content routing, translation, mask building, inpaint, layout/typeset, and QA/export instead of adding a second pipeline. Every fix must be traceable in `decision_trace.jsonl`, `project.json`, `qa_report.json`, and visual debug artifacts.

**Tech Stack:** Python 3.12, OpenCV/NumPy, PaddleOCR/vision stack OCR, comic text detector, TraduzAI strip pipeline, matplotlib/FT2Font typesetting, pytest, debug E2E artifacts.

---

## Evidence Runs

Use these runs as regression evidence:

- `N:/TraduzAI/DEBUGM/runs/2026-05-25_global_text_ocr_fixes_other_chapters_v4i/articuno_ch61`
- `N:/TraduzAI/DEBUGM/runs/2026-05-25_global_text_ocr_fixes_other_chapters_v4i/chapter39`
- `N:/TraduzAI/DEBUGM/runs/2026-05-25_global_text_ocr_fixes_other_chapters_v4i/god_of_death_ch2`
- `N:/TraduzAI/DEBUGM/runs/2026-05-25_global_text_ocr_fixes_other_chapters_v4i/one_second_ch1`
- `N:/TraduzAI/DEBUGM/runs/2026-05-25_global_text_ocr_fixes_other_chapters_v4i/one_second_ch2`
- `N:/TraduzAI/DEBUGM/runs/2026-05-25_chapter1_geometry_inpaint_typeset_v4d_skip_debug_trace/chapter1_full`

Do not treat a pipeline exit code or `complete` stdout event as visual success. The acceptance source is the QA/export gate plus visual review artifacts.

---

## Root Cause Map

1. **Valid text is dropped before translation.**
   - Examples: `His heart...!`, `wrong`, `know?`, `Hospital's`, short counts like `ONE/TWO/FOUR`.
   - Cause: low-confidence visual-noise filters and text-router rules drop short but legible dialogue/SFX inside balloons.

2. **Proper names are not protected.**
   - Examples: `Hosu`, `Wonho`, `Ajussi/Ahjussi`.
   - Cause: glossary/name-lock is empty or applied too late; OCR noise plus lowercase normalization lets Google mistranslate names.

3. **Masks are often not character masks.**
   - Examples: rectangular residue, half-balloon inpaint, black/glow balloon residue, text not removed.
   - Cause: `line_polygons` and bbox fallback replace glyph masks; many blocks have `raw_mask_pixels=0` but still use fast fill.

4. **Fast solid fill is too eager.**
   - Examples: translucent balloons filled as opaque patches, black/glow balloons with white residue, colored panels sampled from text area.
   - Cause: fill color is sampled from unsafe text/bbox area or runs without verified raw glyph coverage.

5. **Typeset safe area collapses.**
   - Examples: 4x2 or 5x3 `safe_text_box`, font 6, text outside balloon, text in one line and too small.
   - Cause: target/anchor boxes are reused as capacity boxes; connected lobe selection can shrink the render area to a tiny box.

6. **Connected balloons are not a first-class layout object.**
   - Examples: text goes to wrong lobe, only half balloon is used, English remains in other lobe.
   - Cause: split subregions exist in debug, but translation/render/inpaint are not consistently assigned per lobe.

7. **Rotated/inclined text has no reliable contract.**
   - Examples: tilted paper text, vertical/90-degree text, side text rendered horizontal or in the wrong place.
   - Cause: OCR angle metadata is not normalized into a typeset transform and mask geometry.

8. **QA catches issues but output still looks usable.**
   - Examples: `export_gate=BLOCK` while stdout emits `complete`, translated pages are written, UI/user sees bad output.
   - Cause: export status and final artifacts are not gated hard enough.

---

## Acceptance Criteria

The implementation is not complete until all of these are true:

- No critical QA issue in the validation matrix.
- `TEXT_CLIPPED`, `TEXT_OVERFLOW`, `render_outside_balloon`, `mask_outside_balloon`, `weak_text_residual_after_inpaint`, and `fast_fill_insufficient_coverage` are blockers unless explicitly waived in debug mode.
- No `safe_text_box` below the minimum readable capacity for a non-skip translated layer.
- No default fast fill with `raw_mask_pixels=0` unless the block is a verified solid-color rectangle/sign and records the reason.
- No translated non-skip layer without `render_bbox`.
- No English dialogue remains in a balloon when OCR saw it and the block is not intentionally preserved.
- Proper names are preserved by name-lock and do not become semantic mistranslations.
- Debug artifacts link each QA issue to page, band, original crop, mask overlay, inpaint decision, render plan, and translated page.

---

## Task 0: Freeze The Regression Matrix

**Files:**
- Create: `N:/TraduzAI/pipeline/tests/fixtures/visual_regressions/README.md`
- Create: `N:/TraduzAI/pipeline/tests/test_visual_regression_manifest.py`
- Modify: `N:/TraduzAI/docs/debug/e2e_pipeline_debug_guide.md`

**Step 1: Write the manifest test**

Create a test that asserts every evidence run listed above has:

- `qa_report.json`
- `project.json`
- `decision_trace.jsonl`
- `debug/e2e`
- `debug_inpaint`
- `translated`

**Step 2: Run the failing test**

```powershell
N:/TraduzAI/pipeline/venv/Scripts/python.exe -m pytest N:/TraduzAI/pipeline/tests/test_visual_regression_manifest.py -v
```

Expected: fail until the manifest helper exists.

**Step 3: Implement the manifest helper**

Keep it read-only. It should not copy large images into git. It should store run paths and issue labels only.

**Step 4: Document the matrix**

Update the debug guide with the exact validation chapters and issue classes.

---

## Task 1: Add OCR Drop/Truncation Regression Tests

**Files:**
- Modify: `N:/TraduzAI/pipeline/ocr/contextual_reviewer.py`
- Modify: `N:/TraduzAI/pipeline/ocr/postprocess.py`
- Modify: `N:/TraduzAI/pipeline/ocr/ocr_normalizer.py`
- Modify: `N:/TraduzAI/pipeline/vision_stack/ocr.py`
- Test: `N:/TraduzAI/pipeline/tests/test_ocr_reviewer.py`
- Test: `N:/TraduzAI/pipeline/tests/test_ocr_postprocess.py`
- Test: `N:/TraduzAI/pipeline/tests/test_vision_stack_ocr.py`

**Step 1: Add failing tests for short valid text**

Cover these cases:

- `His heart...!` inside a balloon must not be dropped as visual noise.
- `What happened?` must stay a dialogue block.
- `ONE!`, `TWO!`, `THREE!`, `FOUR!` must be routed consistently as count/SFX, not random noise.
- `Hospital's` must not be dropped only because confidence is slightly below threshold if it is inside a balloon.

**Step 2: Add failing tests for truncated OCR**

Inputs:

```python
"Why?!What's"
"But... how did you"
"HES HAD AHEART ATTACK!"
"WEDO"
"ittous"
"lyingil"
```

Expected:

- flag `ocr_truncated_or_joined`
- preserve original block for review/re-OCR
- do not silently send broken text to translation as final dialogue

**Step 3: Implement context-aware retention**

Change the reviewer so low confidence is not enough to drop text when:

- the block is inside a detected balloon;
- the shape is a speech/narration box;
- the text is short but alphabetic/punctuated like dialogue;
- the same band has no better duplicate.

**Step 4: Run tests**

```powershell
N:/TraduzAI/pipeline/venv/Scripts/python.exe -m pytest N:/TraduzAI/pipeline/tests/test_ocr_reviewer.py N:/TraduzAI/pipeline/tests/test_ocr_postprocess.py N:/TraduzAI/pipeline/tests/test_vision_stack_ocr.py -v
```

---

## Task 2: Make Content Routing Explicit

**Files:**
- Modify: `N:/TraduzAI/pipeline/ocr/text_router.py`
- Modify: `N:/TraduzAI/pipeline/ocr/contextual_reviewer.py`
- Modify: `N:/TraduzAI/pipeline/qa/translation_qa.py`
- Modify: `N:/TraduzAI/pipeline/utils/decision_log.py`
- Test: `N:/TraduzAI/pipeline/tests/test_text_router_debug.py`
- Test: `N:/TraduzAI/pipeline/tests/test_special_content_router_v2.py`
- Test: `N:/TraduzAI/pipeline/tests/test_translation_qa.py`

**Step 1: Add routing classes**

The router must produce one of:

- `dialogue`
- `narration`
- `sfx_translatable`
- `sfx_preserve`
- `signage`
- `title_card`
- `scanlation_credit`
- `logo_or_watermark`
- `ocr_noise`
- `review_required`

**Step 2: Add regression cases**

Cases:

- English dialogue inside normal balloons -> `dialogue`.
- Korean/Hangul SFX -> `sfx_preserve` by default.
- Latin SFX/counts like `ONE!` -> `sfx_translatable` or `sfx_preserve`, but consistent.
- `READ AT HIVETOON.COM`, staff credits, emails -> `scanlation_credit` or `logo_or_watermark`.
- Title names/cover logos -> preserve, no inpaint unless user enables title translation.

**Step 3: Require decision trace**

Every routed block must write:

- `content_class`
- `route_action`
- `skip_reason` if skipped
- `preserve_reason` if preserved
- `qa_flag` if review is needed

---

## Task 3: Add Name Lock Before Translation

**Files:**
- Modify: `N:/TraduzAI/pipeline/translator/term_protection.py`
- Modify: `N:/TraduzAI/pipeline/translator/translate.py`
- Modify: `N:/TraduzAI/pipeline/context/entity_detector.py`
- Modify: `N:/TraduzAI/pipeline/ocr/ocr_normalizer.py`
- Test: `N:/TraduzAI/pipeline/tests/test_translate_context.py`
- Test: `N:/TraduzAI/pipeline/tests/test_translation_qa.py`
- Test: `N:/TraduzAI/pipeline/tests/test_ocr_normalizer.py`

**Step 1: Write failing tests**

Cases:

- `Hosu...?` remains `Hosu...?` or a configured PT-BR equivalent, never lowercase corrupted.
- `Wonho` does not become `maravilhoso`.
- `Ajussi/Ahjussi` normalizes to the chosen PT-BR style.

**Step 2: Implement placeholders before Google Translate**

Protect names before batching:

```text
__TRADUZAI_NAME_0__
__TRADUZAI_NAME_1__
```

Restore after translation and record `term_protection.applied=true`.

**Step 3: Add OCR-derived entity candidates**

Use capitalization, repeated names, speech context, and glossary file if present. Do not lowercase entity candidates before protection.

---

## Task 4: Require Real Mask Evidence Before Fast Fill

**Files:**
- Modify: `N:/TraduzAI/pipeline/inpainter/mask_builder.py`
- Modify: `N:/TraduzAI/pipeline/inpainter/mask_validator.py`
- Modify: `N:/TraduzAI/pipeline/vision_stack/text_mask_evidence.py`
- Modify: `N:/TraduzAI/pipeline/inpainter/__init__.py`
- Test: `N:/TraduzAI/pipeline/tests/test_mask_builder.py`
- Test: `N:/TraduzAI/pipeline/tests/test_mask_validator.py`
- Test: `N:/TraduzAI/pipeline/tests/test_text_mask_evidence.py`
- Test: `N:/TraduzAI/pipeline/tests/test_vision_stack_inpainter.py`

**Step 1: Add failing tests for raw mask absence**

For a dialogue block:

```python
raw_mask_pixels = 0
expanded_mask_pixels = 0
used_fast_solid_fill = True
```

Expected: reject fast fill and mark `mask_missing_glyph_evidence`.

**Step 2: Add glyph mask requirements**

A valid default inpaint mask must come from at least one:

- OCR text pixel evidence;
- character/glyph segmentation;
- validated CJK segmentation mask;
- high-confidence line polygon clipped to text pixels.

Plain rectangular bbox is not enough for dialogue text.

**Step 3: Add OCR-on-mask validation**

If the post-mask crop still OCRs the same source text, mark `mask_failed_to_cover_source_text` and retry/QA-block.

---

## Task 5: Make Fast Solid Fill Safe And Local

**Files:**
- Modify: `N:/TraduzAI/pipeline/inpainter/__init__.py`
- Modify: `N:/TraduzAI/pipeline/inpainter/region_strategy.py`
- Modify: `N:/TraduzAI/pipeline/inpainter/fill_normalization.py`
- Test: `N:/TraduzAI/pipeline/tests/test_inpaint_region_strategy.py`
- Test: `N:/TraduzAI/pipeline/tests/test_vision_stack_inpainter.py`
- Test: `N:/TraduzAI/pipeline/tests/test_inpainting_profile.py`

**Step 1: Write failing tests for color sampling**

Cases:

- pale blue balloon samples pale blue from inside balloon outside text mask;
- black glow balloon samples black from interior, not white/glow border;
- translucent/gradient balloon rejects fast fill and uses real inpaint;
- white balloon still works with sampled solid fill.

**Step 2: Change sampling source**

Sample from:

- inside `balloon_inner_bbox` or validated balloon mask;
- outside expanded text mask;
- away from border/glow by erosion;
- never from pixels under the source text.

**Step 3: Reject unsafe fills**

Reject fast solid if:

- local variance is high;
- alpha/translucency/gradient estimate is high;
- sampled color differs too much across lobes;
- text mask coverage is too low.

---

## Task 6: Add Inpaint Residual Retry

**Files:**
- Modify: `N:/TraduzAI/pipeline/inpainter/residual_cleanup.py`
- Modify: `N:/TraduzAI/pipeline/qa/inpaint_residual.py`
- Modify: `N:/TraduzAI/pipeline/inpainter/__init__.py`
- Test: `N:/TraduzAI/pipeline/tests/test_inpaint_debug_residual.py`
- Test: `N:/TraduzAI/pipeline/tests/test_visual_text_leak.py`

**Step 1: Add failing residual tests**

Use synthetic dark, white, blue, and translucent balloons with leftover English pixels.

Expected:

- residual text triggers retry;
- retry expands glyph mask, not full rectangle;
- residual after retry becomes QA blocker.

**Step 2: Implement retry policy**

Order:

1. glyph mask expansion by 1-2 px;
2. OCR-on-mask check;
3. real inpaint if fast fill failed;
4. QA block if still leaking.

---

## Task 7: Fix Safe Text Box And Font Fit

**Files:**
- Modify: `N:/TraduzAI/pipeline/layout/safe_area.py`
- Modify: `N:/TraduzAI/pipeline/layout/balloon_layout.py`
- Modify: `N:/TraduzAI/pipeline/layout/simple_text_geometry.py`
- Modify: `N:/TraduzAI/pipeline/typesetter/text_fit_guard.py`
- Modify: `N:/TraduzAI/pipeline/typesetter/renderer.py`
- Test: `N:/TraduzAI/pipeline/tests/test_typesetting_layout.py`
- Test: `N:/TraduzAI/pipeline/tests/test_typesetting_renderer.py`
- Test: `N:/TraduzAI/pipeline/tests/test_typesetting_fit_qa.py`
- Test: `N:/TraduzAI/pipeline/tests/test_simple_text_geometry.py`

**Step 1: Add failing tests for degenerate safe boxes**

Cases from One Second ch2:

- `safe_text_box` 4x2 for `What happened?`
- `safe_text_box` 5x3 for `Hosu?`
- large balloon but tiny text in center

Expected:

- reject safe box;
- recompute from full balloon interior;
- if recompute fails, block render.

**Step 2: Separate anchor from capacity**

Introduce explicit fields:

- `position_anchor_bbox`
- `capacity_bbox`
- `safe_text_box`

The anchor can be small. The capacity cannot be tiny for normal dialogue.

**Step 3: Implement wrap-before-shrink**

Before using font <= 9:

- try more line breaks;
- use available balloon width;
- preserve line spacing;
- reject unreadable fallback.

**Step 4: Add hard QA flags**

Font <= 9, clipped text, overflow, and outside-balloon render become blockers unless the block is a deliberately tiny annotation.

---

## Task 8: Make Connected Balloons First-Class

**Files:**
- Modify: `N:/TraduzAI/pipeline/layout/connected_balloon_splitter.py`
- Modify: `N:/TraduzAI/pipeline/layout/balloon_layout.py`
- Modify: `N:/TraduzAI/pipeline/typesetter/renderer.py`
- Modify: `N:/TraduzAI/pipeline/inpainter/mask_builder.py`
- Test: `N:/TraduzAI/pipeline/tests/test_layout_analysis.py`
- Test: `N:/TraduzAI/pipeline/tests/test_typesetting_layout.py`
- Test: `N:/TraduzAI/pipeline/tests/test_inpaint_mask_geometry.py`

**Step 1: Add failing tests**

Cases:

- left/right connected balloon with two English blocks;
- top/bottom connected balloon;
- one lobe translated and the other still English;
- text assigned to bridge instead of lobe.

**Step 2: Create lobe assignments**

Each OCR block in a connected balloon must have:

- `connected_balloon_id`
- `lobe_id`
- `lobe_bbox`
- `lobe_safe_text_box`
- `source_text_bbox`

**Step 3: Render per lobe**

Render each translated block inside its lobe. Use original source text position as anchor only within that lobe, never as the capacity area.

**Step 4: Inpaint per lobe/glyph**

The mask must remove glyphs from every lobe that had source text. If any lobe has untranslated English, QA blocks.

---

## Task 9: Support Rotated And Inclined Text

**Files:**
- Modify: `N:/TraduzAI/pipeline/vision_stack/ocr.py`
- Modify: `N:/TraduzAI/pipeline/ocr/postprocess.py`
- Modify: `N:/TraduzAI/pipeline/typesetter/renderer.py`
- Modify: `N:/TraduzAI/pipeline/typesetter/style_policy.py`
- Modify: `N:/TraduzAI/pipeline/inpainter/mask_builder.py`
- Test: `N:/TraduzAI/pipeline/tests/test_vision_stack_ocr.py`
- Test: `N:/TraduzAI/pipeline/tests/test_typesetting_renderer.py`
- Test: `N:/TraduzAI/pipeline/tests/test_inpaint_mask_geometry.py`

**Step 1: Add failing angle tests**

Cases:

- 90-degree side text;
- tilted paper text;
- rotated UI card title;
- vertical text that should be preserved as SFX/signage.

**Step 2: Normalize OCR angle metadata**

Store:

- `text_angle_degrees`
- `text_orientation`
- `rotated_bbox`
- `rotated_polygon`

**Step 3: Pass angle to typeset**

Renderer must draw translated text with the same angle when the block is routed as rotated dialogue/signage.

**Step 4: Mask rotated glyph area**

Mask builder must expand rotated polygons, not axis-aligned rectangles, for tilted text.

---

## Task 10: Enforce Final Render Contract

**Files:**
- Modify: `N:/TraduzAI/pipeline/main.py`
- Modify: `N:/TraduzAI/pipeline/project_writer.py`
- Modify: `N:/TraduzAI/pipeline/qa/export_gate.py`
- Modify: `N:/TraduzAI/pipeline/qa/render_geometry.py`
- Modify: `N:/TraduzAI/pipeline/qa/translation_qa.py`
- Test: `N:/TraduzAI/pipeline/tests/test_final_geometry_contract.py`
- Test: `N:/TraduzAI/pipeline/tests/test_export_gate.py`
- Test: `N:/TraduzAI/pipeline/tests/test_export_gate_debug_consistency.py`
- Test: `N:/TraduzAI/pipeline/tests/test_main_emit.py`

**Step 1: Add failing tests**

Assert blockers for:

- non-skip translated layer without `render_bbox`;
- `render_outside_balloon`;
- `TEXT_CLIPPED`;
- `TEXT_OVERFLOW`;
- unresolved English dialogue in final output;
- `export_gate=BLOCK` with stdout `complete` as success.

**Step 2: Change completion status**

If export gate blocks, emit a distinct status:

```text
completed_blocked
```

or mark the final event with:

```json
{"status":"blocked","export_gate":"BLOCK"}
```

Do not let the UI interpret this as visually approved output.

**Step 3: Require render artifacts**

Every non-skip translated text layer must have:

- `render_bbox`
- `safe_text_box`
- `font_size`
- `fit_status`
- `qa_flags`

Missing fields block export.

---

## Task 11: Improve Debug Traceability

**Files:**
- Modify: `N:/TraduzAI/pipeline/debug_tools/masks.py`
- Modify: `N:/TraduzAI/pipeline/tools/export_visual_review_sheet.py`
- Modify: `N:/TraduzAI/pipeline/debug_visual_artifact.py`
- Modify: `N:/TraduzAI/pipeline/utils/decision_log.py`
- Test: `N:/TraduzAI/pipeline/tests/test_debug_report.py`
- Test: `N:/TraduzAI/pipeline/tests/test_render_plan_trace_integrity.py`
- Test: `N:/TraduzAI/pipeline/tests/test_export_visual_review_sheet.py`

**Step 1: Add issue artifact links**

Each QA issue must link to:

- translated page;
- original page;
- band original;
- mask overlay;
- inpaint decision;
- render plan entry;
- contact sheet crop.

**Step 2: Add visual contact sheets by issue type**

Produce sheets for:

- untranslated text;
- inpaint residual;
- render outside balloon;
- clipped/too-small text;
- connected balloon split;
- fast fill rejected/used.

---

## Task 12: Validate On The Full Chapter Matrix

**Files:**
- Modify only if needed: `N:/TraduzAI/pipeline/tests/test_visual_regression_manifest.py`
- Output: `N:/TraduzAI/DEBUGM/runs/<new_validation_run>/`

**Step 1: Run focused unit tests**

```powershell
N:/TraduzAI/pipeline/venv/Scripts/python.exe -m pytest `
  N:/TraduzAI/pipeline/tests/test_ocr_reviewer.py `
  N:/TraduzAI/pipeline/tests/test_ocr_postprocess.py `
  N:/TraduzAI/pipeline/tests/test_translation_qa.py `
  N:/TraduzAI/pipeline/tests/test_mask_builder.py `
  N:/TraduzAI/pipeline/tests/test_mask_validator.py `
  N:/TraduzAI/pipeline/tests/test_vision_stack_inpainter.py `
  N:/TraduzAI/pipeline/tests/test_typesetting_renderer.py `
  N:/TraduzAI/pipeline/tests/test_typesetting_layout.py `
  N:/TraduzAI/pipeline/tests/test_export_gate.py `
  N:/TraduzAI/pipeline/tests/test_final_geometry_contract.py `
  -v
```

**Step 2: Run chapter validation**

Run the same automatic pipeline/debug path used for the evidence matrix on:

- Articuno ch61
- Chapter 39
- God of Death ch2
- One Second ch1
- One Second ch2
- Chapter 1

**Step 3: Inspect QA summaries**

For each chapter, check:

- `qa_report.json`
- `debug/e2e/11_qa_export_gate`
- `debug/e2e/12_contact_sheets`
- `debug_inpaint`
- `translated`

**Step 4: Produce final report**

The report must list:

- fixed issue classes;
- remaining issue classes;
- before/after timing;
- before/after QA flags;
- visual contact sheets;
- exact chapters/pages still needing manual review.

---

## Task 13: App Pipeline Integration Check

**Files:**
- Modify: `N:/TraduzAI/src-tauri/src/commands/pipeline.rs`
- Modify: `N:/TraduzAI/src/lib/tauri.ts`
- Modify: `N:/TraduzAI/src/lib/stores/appStore.ts`
- Test: `N:/TraduzAI/pipeline/tests/test_main_emit.py`
- Test if available: frontend/Tauri pipeline event tests

**Step 1: Verify the app uses the same defaults**

Confirm the desktop app path uses:

- full-page detect/OCR default;
- inpaint parallel default;
- fast solid safety rules;
- cleanup rerender disabled unless explicitly enabled;
- strict export gate semantics.

**Step 2: Make blocked output visible**

If `completed_blocked` or `export_gate=BLOCK`, the UI must show that the chapter needs review and must not label it as normal success.

---

## Implementation Order

1. Task 0: freeze the evidence matrix.
2. Task 10: make QA/export status truthful first, so broken outputs stop looking successful.
3. Task 7: fix safe text boxes and readable font fallback.
4. Task 4: require real mask evidence.
5. Task 5: make fast solid fill safe.
6. Task 6: add residual retry.
7. Task 1: improve OCR retention/truncation.
8. Task 2: explicit content routing.
9. Task 3: name-lock translation.
10. Task 8: connected balloons.
11. Task 9: rotated/inclined text.
12. Task 11: traceability/contact sheets.
13. Task 12 and 13: validation matrix and app integration.

This order is deliberate: first prevent false success, then fix the highest-impact visual failures, then improve semantic and edge-case quality.

---

## Commit Strategy

Use small commits:

1. `test: add visual regression manifest`
2. `fix: enforce blocked export status`
3. `fix: reject degenerate typeset safe boxes`
4. `fix: require glyph evidence before fast fill`
5. `fix: make solid fill sampling safe`
6. `fix: retry residual text inpaint`
7. `fix: retain short valid OCR text`
8. `fix: route sfx credits and dialogue explicitly`
9. `fix: protect names during translation`
10. `fix: assign connected balloon lobes`
11. `fix: preserve rotated text geometry`
12. `test: validate visual chapter matrix`

Do not stage generated debug outputs unless the project already tracks a small fixture intentionally.

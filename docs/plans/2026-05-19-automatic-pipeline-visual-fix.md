# Automatic Pipeline Visual Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the automatic pipeline produce visually usable chapter output, with real E2E traceability from OCR/detect through mask, inpaint, typeset, `project.json`, QA, and exported images.

**Architecture:** Treat the visual output as the source of truth, not just the JSON metrics. First lock the debug contract so `render_plan_final.jsonl` and `project.json` describe the exact same rendered geometry, then fix the stages that currently fight each other: page cleanup, balloon geometry, mask/inpaint, content routing, and typesetting placement. Every fix must leave a visual artifact and an identity trail keyed by `trace_id`.

**Tech Stack:** Python 3.12, OpenCV, NumPy, current `pipeline/main.py` sidecar, `pipeline/strip`, `pipeline/vision_stack`, `pipeline/layout`, `pipeline/inpainter`, `pipeline/typesetter`, `pipeline/qa`, and `tools/analyze_e2e_debug.py`.

---

## Current Evidence

Fresh run analyzed:

`N:\TraduzAI\DEBUGM\runs\2026-05-19_chapter1_e2e_debug_codex_2026-05-18_215312`

Human-inspectable review pack generated during analysis:

- `N:\TraduzAI\DEBUGM\runs\2026-05-19_chapter1_e2e_debug_codex_2026-05-18_215312\codex_visual_review\page_001_original_A_B_C_D.jpg`
- `N:\TraduzAI\DEBUGM\runs\2026-05-19_chapter1_e2e_debug_codex_2026-05-18_215312\codex_visual_review\page_002_original_A_B_C_D.jpg`
- `N:\TraduzAI\DEBUGM\runs\2026-05-19_chapter1_e2e_debug_codex_2026-05-18_215312\codex_visual_review\page_006_original_A_B_C_D.jpg`
- `N:\TraduzAI\DEBUGM\runs\2026-05-19_chapter1_e2e_debug_codex_2026-05-18_215312\codex_visual_review\qa_blocker_crops_original_A_B_C_D.jpg`

Run summary:

| Run | Exit | Gate | QA issues | Visual blockers | render_on_art | bbox_overreach | missing_balloon | render_plan/project mismatch |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A_baseline_debug | 0 | BLOCK | 76 | 27 | 2 | 3 | 18 | 1 |
| B_skip_inpaint | 0 | BLOCK | 2 | 2 | 2 | 0 | 0 | 55 |
| C_fast_fill | 0 | BLOCK | 31 | 13 | 2 | 3 | 18 | 1 |
| D_strict_export_gate | 2 | BLOCK | 76 | 27 | 2 | 3 | 18 | 1 |

Strict debug audit status:

- Passed: trace IDs, page/band trace consistency, QA traceability, flag propagation, contact sheets, skip-inpaint consistency.
- Failed: `render_plan_final_matches_project` in all four runs.

Performance evidence:

| Run | cleanup_inpaint | cleanup_typeset | cleanup_total | cleanup_skipped |
| --- | ---: | ---: | ---: | --- |
| A_baseline_debug | 228.5055s | 14.5921s | 293.9953s | false |
| B_skip_inpaint | 0.0s | 0.0s | 0.0s | true |
| C_fast_fill | 142.3841s | 8.6925s | 183.0375s | false |
| D_strict_export_gate | 147.649s | 9.2514s | 189.424s | false |

Visual findings from the generated sheets:

- `B_skip_inpaint` is useful as a control run: it proves skip is fast, but the final image overlays PT-BR text on top of the original EN text. It must never be treated as a quality pass unless the user explicitly wants an uncleaned preview.
- A/C/D remove more source text, but still have bad rendered text on art, source text left behind, and misplaced/cut text. Examples: `QUANDO EU VOLTAR AO TRABALHO...` appears on the face/art area, `GIVE ME THE MONEY...` remains in English, and `ANCE, VOCE` is a broken fragment.
- `C_fast_fill` is not yet a safe speed path. It reduces issue count from 76 to 31, but still has 13 visual blockers and visible doubled text in several crops.
- The debug system is now good enough to find real visual blockers, but still not reliable enough to certify a run while `render_plan_final.jsonl` and `project.json` disagree.

---

## Success Criteria

The fix is done only when a fresh run on the same Chapter 1 fixture satisfies all of these:

- `tools/analyze_e2e_debug.py <run-root> --write-report --strict-debug-audit` exits `0`.
- `render_plan_project_mismatch_count == 0` and `render_plan_project_field_mismatch_count == 0` for A/B/C/D.
- Production-quality runs have `visual_blocker_count == 0`.
- `balloon_bbox_missing_count == 0` for renderable dialogue/narration/sign text.
- No final translated page has PT-BR text drawn on top of remaining EN source text.
- No renderable text has `render_on_art_suspected`, `bbox_overreach_critical`, `mask_outside_balloon_critical`, or `text_residual_after_inpaint`.
- `C_fast_fill` is either visually equivalent to A/D within the accepted QA thresholds or remains opt-in only.
- `B_skip_inpaint` stays traceable and geometrically coherent, but its output is labeled/control-only because visual cleanup is intentionally skipped.
- The review pack includes page-level comparison sheets and crop-level blocker sheets for the final run.

---

## Non-Goals

- Do not replace the whole OCR/detector stack before fixing geometry, copyback, and QA contracts.
- Do not add paid/cloud image services; the image pipeline remains local.
- Do not hide blockers by weakening `export_gate`.
- Do not make scanlation credits, URLs, title art, and decorative SFX render as normal dialogue.
- Do not create a second automatic pipeline path. Keep one pipeline with guarded runtime modes.

---

## Root Cause Map

### 1. Render/project contract is still inconsistent

Files:

- `pipeline/typesetter/renderer.py`
- `pipeline/strip/run.py`
- `pipeline/strip/process_bands.py`
- `tools/analyze_e2e_debug.py`
- `pipeline/tests/test_render_plan_trace_integrity.py`
- `pipeline/tests/test_typesetting_renderer.py`
- `pipeline/tests/test_analyze_e2e_debug.py`

Evidence:

- A/C/D each have one mismatch: `ocr_001@page_003_band_042` differs only in `balloon_bbox`. `render_plan_final` stores `[343, 11293, 538, 11317]`; `project.json` stores `[0, 10726, 800, 11410]`.
- B has 55 mismatches and 164 field mismatches. Its `render_plan_final` Y coordinates are shifted into strip/global space, while `project.json` stays page-local. Example: `page_002_band_020` has render plan y around `15125`, project y around `1293`.

Interpretation:

- The final render plan currently mixes coordinate contracts across full-page rerender, band render, and skip cleanup paths.
- The analyzer is doing its job: strict should fail until this is fixed.

### 2. Page cleanup/rerender is visually risky and expensive

Files:

- `pipeline/strip/run.py`
- `pipeline/vision_stack/runtime.py`
- `pipeline/typesetter/renderer.py`
- `pipeline/tests/test_strip_run.py`
- `pipeline/tests/test_page_cleanup_breakdown.py`
- `pipeline/tests/test_vision_stack_runtime.py`

Evidence:

- A spends ~294s in cleanup/rerender, with ~228s in cleanup inpaint.
- C/D still spend ~183-189s after fast fill.
- B skips cleanup correctly and drops this stage to 0s.
- A/C/D have the same `balloon_bbox_missing_count=18`, and their final pages show misplaced text and residual English even after expensive cleanup.

Interpretation:

- Cleanup is doing too much after the core pipeline has already made geometry decisions.
- Full page cleanup/rerender can re-enter typesetting with a different coordinate context, causing mismatches and visual drift.

### 3. Balloon geometry is not trustworthy enough

Files:

- `pipeline/layout/balloon_layout.py`
- `pipeline/vision_stack/runtime.py`
- `pipeline/debug_tools/bbox.py`
- `pipeline/typesetter/renderer.py`
- `pipeline/tests/test_typesetting_layout.py`
- `pipeline/tests/test_ocr_geometry_dedupe.py`

Evidence:

- A/C/D emit 18 `balloon_bbox_missing_audit` rows.
- Top missing bands: `page_002_band_019` with 10 rows, `page_001_band_000` with 4 rows, `page_003_band_047` with 3 rows, `page_006_band_105` with 1 row.
- Some audit rows have confusing identity context, e.g. a trace from `page_002_band_013` recorded under `page_001_band_000`.

Interpretation:

- A renderable block can still reach typesetting without a valid balloon.
- The fallback `bbox_as_balloon_bbox` prevents crashes but creates bad masks and bad text placement.

### 4. Mask/inpaint is overbroad in some places and too weak in others

Files:

- `pipeline/inpainter/mask_builder.py`
- `pipeline/debug_tools/masks.py`
- `pipeline/inpainter/__init__.py`
- `pipeline/vision_stack/runtime.py`
- `pipeline/tests/test_mask_builder.py`
- `pipeline/tests/test_mask_chain_debug.py`
- `pipeline/tests/test_vision_stack_inpainter.py`

Evidence:

- A/D: 42 `mask_density_high`, 22 `mask_outside_balloon_critical`, 2 `text_residual_after_inpaint`.
- C: 14 `mask_density_high`, 6 `mask_outside_balloon_critical`, 4 `text_residual_after_inpaint`.
- Problem crops show masks/placement crossing balloon boundaries and residual English in final output.

Interpretation:

- Fast fill reduces some mask issues, but does not establish a safe visual invariant.
- The mask builder still trusts bad geometry too much. It flags the issue, but the pipeline has already produced a bad page.

### 5. Typesetting can render onto art or produce unusable fragments

Files:

- `pipeline/typesetter/renderer.py`
- `pipeline/typesetter/fit_qa.py`
- `pipeline/qa/render_geometry.py`
- `pipeline/qa/translation_qa.py`
- `pipeline/tests/test_typesetting_renderer.py`
- `pipeline/tests/test_typesetting_fit_qa.py`
- `pipeline/tests/test_render_geometry.py`

Evidence:

- `render_on_art_suspected=2` in every run, including B. That means it is not purely an inpaint problem.
- Visual examples include `POR FAVOR, PELO BEM DA CRIANCA.` rendered across art and `QUANDO EU VOLTAR AO TRABALHO...` rendered outside the intended bubble area.
- The broken fragment `ANCE, VOCE` appears as its own large rendered text.

Interpretation:

- Placement and text routing need to reject unsafe targets before drawing.
- Render QA is currently late enough to block export, but not early enough to choose a better layout or skip a bad render.

### 6. Content routing/OCR still lets non-dialogue or broken text enter render

Files:

- `pipeline/vision_stack/runtime.py`
- `pipeline/ocr/postprocess.py`
- `pipeline/ocr/ocr_normalizer.py`
- `pipeline/translator/translate.py`
- `pipeline/runtime_profiles.py`
- `pipeline/tests/test_vision_stack_runtime.py`
- `pipeline/tests/test_ocr_normalizer.py`
- `pipeline/tests/test_translate_context.py`

Evidence:

- Content classes in the run: `noise=18`, `dialogue=45`, `narration=24`, `tn_note=1`, `sign=1`, `url_watermark=1`.
- Cover/credit/title material is mostly skipped correctly, but title/source art remains in final output by design; that must be explicit in QA instead of confused with dialogue.
- Broken fragments like `ANCE, VOCE` suggest OCR splitting or source text fragment cleanup is incomplete.

Interpretation:

- Routing must distinguish: dialogue to translate/render; narration to translate/render if safe; signs to translate/render only if source removal is possible; SFX/title/credit/url/watermark to preserve or review, not silently treat as dialogue.

---

## Phase 0: Freeze the Visual Fixture and Make Review Reproducible

**Files:**

- Create: `tools/build_e2e_visual_review.py`
- Modify: `tools/analyze_e2e_debug.py`
- Create: `pipeline/tests/test_e2e_visual_review_tool.py`
- Reference artifacts: current run under `DEBUGM\runs\2026-05-19_chapter1_e2e_debug_codex_2026-05-18_215312`

**Step 1: Add a reusable visual review tool**

Implement a small OpenCV-based tool that takes an A/B/C/D run root and emits:

- `visual_review/page_<n>_original_A_B_C_D.jpg`
- `visual_review/qa_blocker_crops_original_A_B_C_D.jpg`
- `visual_review/visual_review_summary.json`

Required crop overlays:

- red: `bbox`
- orange: `source_bbox`
- green: `balloon_bbox`
- blue/purple: `render_bbox`

**Step 2: Add synthetic tests**

Test with tiny generated images and fake `qa_issues.jsonl`; do not commit protected chapter images as fixtures.

Run:

```powershell
N:\TraduzAI\pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_e2e_visual_review_tool.py -q
```

Expected: pass, and the temporary output folder contains the expected sheet names.

**Step 3: Connect analyzer report to visual review**

Extend `tools/analyze_e2e_debug.py --write-report` to mention the visual review folder if it exists, and to fail strict only on machine invariants, not on missing optional review sheets unless `--require-visual-review` is passed.

---

## Phase 1: Fix Render Plan and Project Geometry Before Touching Quality Logic

**Files:**

- Modify: `pipeline/typesetter/renderer.py`
- Modify: `pipeline/strip/run.py`
- Modify: `pipeline/strip/process_bands.py`
- Modify: `tools/analyze_e2e_debug.py`
- Test: `pipeline/tests/test_render_plan_trace_integrity.py`
- Test: `pipeline/tests/test_typesetting_renderer.py`
- Test: `pipeline/tests/test_analyze_e2e_debug.py`

**Step 1: Write failing tests for B skip coordinate drift**

Create a test where:

- input OCR page declares `_coordinate_space="page"`
- `band_y_top` is non-zero
- `skip_page_cleanup_rerender=True`
- render plan and project both remain page-local

Expected failure before fix:

- render plan has y shifted by `band_y_top`
- project layer does not

**Step 2: Fix `_shift_render_plan_to_page` contract**

Current rule is right in spirit but incomplete:

- if `coordinate_space == "page"`, never add `band_y_top`
- if `coordinate_space == "band"`, add `band_y_top` exactly once
- candidates/skipped diagnostics can stay band-local, but final plan must match `project.json`

Make the source of coordinate truth explicit in the payload:

```python
payload["coordinate_space_before_final_shift"] = coordinate_space
payload["final_shift_applied_y"] = 0 or band_y_top
```

**Step 3: Fix project copyback of final render geometry**

After final rerender, update the same layer fields that strict compares:

- `page_id`
- `band_id`
- `balloon_bbox`
- `safe_text_box`
- `_debug_safe_text_box`
- `render_bbox`
- `coordinate_space`

The final `project.json` and `09_typeset/render_plan_final.jsonl` must use the same values for the same `trace_id`.

**Step 4: Fix the one A/C/D balloon mismatch**

Investigate `ocr_001@page_003_band_042` and decide which `balloon_bbox` is canonical:

- if `[0, 10726, 800, 11410]` is the real balloon, render plan must keep that and put safe text inside it;
- if `[343, 11293, 538, 11317]` is only the text mask/ink area, it must not replace `balloon_bbox`.

Add a regression test where `safe_text_box` is small but `balloon_bbox` is broad. Expected: final render plan keeps broad `balloon_bbox` and small `safe_text_box`.

**Step 5: Verify**

Run:

```powershell
N:\TraduzAI\pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_render_plan_trace_integrity.py pipeline\tests\test_typesetting_renderer.py pipeline\tests\test_analyze_e2e_debug.py -q
```

Expected: pass.

Fresh debug expectation:

- `render_plan_project_mismatch_count == 0` for A/B/C/D
- `render_plan_project_field_mismatch_count == 0` for A/B/C/D

---

## Phase 2: Make Page Cleanup Safe and Bounded

**Files:**

- Modify: `pipeline/strip/run.py`
- Modify: `pipeline/vision_stack/runtime.py`
- Modify: `pipeline/typesetter/renderer.py`
- Test: `pipeline/tests/test_strip_run.py`
- Test: `pipeline/tests/test_page_cleanup_breakdown.py`
- Test: `pipeline/tests/test_vision_stack_runtime.py`

**Step 1: Lock skip behavior**

Add or strengthen a test proving that `--skip-inpaint` implies:

- no band inpaint
- no page cleanup/rerender
- no `debug_inpaint`
- `page_cleanup_breakdown.cleanup_skipped == true`
- final render plan/project geometry still match

**Step 2: Change cleanup from "rerender page freely" to "repair only safe targets"**

Before `_cleanup_page_inpaint_and_rerender` modifies pixels, filter candidates:

- must have valid `trace_id`
- must have valid `balloon_bbox`
- must have valid `source_bbox`
- must not have `skip_processing`
- must not have URL/watermark/credit/SFX/title flags
- must not have geometry flags already indicating unsafe mask

If a candidate fails, do not cleanup it. Add `qa_flags=["cleanup_skipped_geometry_untrusted"]`.

**Step 3: Prevent cleanup from changing geometry fields**

Cleanup may change pixels, but it must not change:

- `balloon_bbox`
- `safe_text_box`
- `render_bbox`
- `page_id`
- `band_id`
- `trace_id`

If cleanup rerenders text, it must update both render plan and project through the same copyback function from Phase 1.

**Step 4: Add time budgets**

Add performance guardrails:

- record per-page cleanup durations
- record number of cleanup candidates and skipped candidates
- if cleanup exceeds a page threshold, emit `cleanup_perf_hotspot` warning with the top candidate IDs

Do not make the threshold block export yet; use it as diagnostic.

**Step 5: Verify**

Run:

```powershell
N:\TraduzAI\pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_strip_run.py pipeline\tests\test_page_cleanup_breakdown.py pipeline\tests\test_vision_stack_runtime.py -q
```

Expected: pass.

Fresh debug expectation:

- B remains cleanup skipped with 0s cleanup.
- A/C/D cleanup no longer introduces render/project mismatch.
- Visual blocker count does not increase after cleanup.

---

## Phase 3: Fix Balloon Geometry and Missing-Balloon Flow

**Files:**

- Modify: `pipeline/layout/balloon_layout.py`
- Modify: `pipeline/vision_stack/runtime.py`
- Modify: `pipeline/debug_tools/bbox.py`
- Modify: `pipeline/typesetter/renderer.py`
- Test: `pipeline/tests/test_typesetting_layout.py`
- Test: `pipeline/tests/test_ocr_geometry_dedupe.py`
- Test: `pipeline/tests/test_render_plan_trace_integrity.py`

**Step 1: Add failing tests for renderable text without balloon**

Cases:

- dialogue with OCR text bbox but no balloon candidate
- narration/sign with no safe region
- cover/credit/noise with no balloon but `skip_processing=true`

Expected:

- renderable dialogue/narration cannot proceed silently with `bbox_as_balloon_bbox`
- skipped noise can proceed without a balloon if it is never rendered

**Step 2: Promote missing balloon to explicit routing decision**

For renderable text:

- no `balloon_bbox` -> `qa_flags += ["balloon_bbox_missing"]`
- no render attempt unless a safe fallback is proven
- export gate blocks if a renderable layer has `balloon_bbox_missing`

For skipped/noise text:

- keep traceability
- do not count as `balloon_bbox_missing_count` if not rendered

**Step 3: Fix page_id/band_id confusion in missing audit**

`balloon_bbox_missing_audit.jsonl` must derive `page_id` and `band_id` from the same source as `trace_id`. If `trace_id = ocr_001@page_002_band_013`, the audit row cannot claim `page_001_band_000`.

**Step 4: Improve connected/multi-lobe balloon safety**

In `pipeline/layout/balloon_layout.py`, keep broad `balloon_bbox` as the container and use `connected_position_bboxes`/`safe_text_box` for text placement. Do not replace the balloon with a narrow text mask just because one lobe is small.

**Step 5: Verify**

Run:

```powershell
N:\TraduzAI\pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_typesetting_layout.py pipeline\tests\test_ocr_geometry_dedupe.py pipeline\tests\test_render_plan_trace_integrity.py -q
```

Fresh debug expectation:

- `balloon_bbox_missing_count == 0` for A/C/D, or remaining rows are only skipped/noise and excluded from renderable count.
- No `bbox_as_balloon_bbox` fallback for rendered dialogue.

---

## Phase 4: Make Mask/Inpaint Conservative by Default

**Files:**

- Modify: `pipeline/inpainter/mask_builder.py`
- Modify: `pipeline/debug_tools/masks.py`
- Modify: `pipeline/inpainter/__init__.py`
- Modify: `pipeline/vision_stack/runtime.py`
- Test: `pipeline/tests/test_mask_builder.py`
- Test: `pipeline/tests/test_mask_chain_debug.py`
- Test: `pipeline/tests/test_vision_stack_inpainter.py`

**Step 1: Add regression tests from observed blocker types**

Synthetic cases:

- text mask larger than balloon interior -> clip and flag
- text mask density too high -> reject fast fill and route to real inpaint/manual review
- source text remains after fast fill -> emit `text_residual_after_inpaint`
- protected art adjacent to speech bubble -> mask must not leak outside balloon

**Step 2: Reject unsafe geometry before building a destructive mask**

If any of these are true, do not run destructive inpaint:

- missing `balloon_bbox`
- `bbox_overreach_critical`
- `mask_outside_balloon_critical`
- `render_on_art_suspected`
- URL/watermark/credit/SFX/title content class

Instead mark the block for review with a traceable flag and preserve pixels.

**Step 3: Separate final mask roles**

Keep distinct masks in debug:

- source text mask
- balloon interior mask
- protection/art mask
- final inpaint mask
- residual check mask

The final inpaint mask must equal:

```text
(source text mask intersect balloon interior) minus protection/art mask
```

**Step 4: Make fast fill a candidate, not a default**

`C_fast_fill` can be enabled only when:

- mask density is below threshold
- source residual check passes
- text/balloon geometry is trusted
- visual blocker count does not increase versus baseline

If any condition fails, fallback to the quality path or emit a blocking flag.

**Step 5: Verify**

Run:

```powershell
N:\TraduzAI\pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_mask_builder.py pipeline\tests\test_mask_chain_debug.py pipeline\tests\test_vision_stack_inpainter.py -q
```

Fresh debug expectation:

- A/D: `mask_outside_balloon_critical == 0`
- C: no `text_residual_after_inpaint` introduced by fast fill
- Visual crop sheet shows no source text underneath PT-BR text

---

## Phase 5: Harden Typesetting Placement and Fit

**Files:**

- Modify: `pipeline/typesetter/renderer.py`
- Modify: `pipeline/typesetter/fit_qa.py`
- Modify: `pipeline/qa/render_geometry.py`
- Modify: `pipeline/qa/translation_qa.py`
- Test: `pipeline/tests/test_typesetting_renderer.py`
- Test: `pipeline/tests/test_typesetting_fit_qa.py`
- Test: `pipeline/tests/test_render_geometry.py`
- Test: `pipeline/tests/regression/test_text_clipping_page001.py`

**Step 1: Add tests for the visual blockers**

Use synthetic images to test:

- dark art background under render bbox -> `render_on_art_suspected`
- safe text box outside/near edge of balloon -> reject candidate
- long PT-BR string in small balloon -> reduce font, rewrap, then flag `text_overflow` if still unsafe
- fragment text such as `ANCE, VOCE` -> route to OCR review unless context can repair it

**Step 2: Make render candidate selection QA-aware**

Candidate scoring must reject candidates that:

- place ink outside `safe_text_box`
- place ink on dark/non-white art for white balloon text
- overlap protected art or another active text bbox
- shrink below readable minimum font size

**Step 3: Keep text inside the right target**

Define strict hierarchy:

1. `connected_position_bboxes` when connected-balloon metadata exists.
2. `safe_text_box` inside `balloon_bbox`.
3. fallback from `balloon_bbox` only if it is trusted.
4. no render if only OCR text bbox exists.

**Step 4: Add final render QA before pixels are committed**

Before drawing into the final page:

- run `check_render_inside_balloon`
- run `check_render_background`
- run fit/clipping checks
- if critical flags appear, do not draw the bad text over art; keep the original/cleaned image and block export with traceable QA.

**Step 5: Verify**

Run:

```powershell
N:\TraduzAI\pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_typesetting_renderer.py pipeline\tests\test_typesetting_fit_qa.py pipeline\tests\test_render_geometry.py pipeline\tests\regression\test_text_clipping_page001.py -q
```

Fresh debug expectation:

- `render_on_art_count == 0`
- no `TEXT_CLIPPED`/`TEXT_OVERFLOW` critical leakage in final accepted render
- visual crop sheet no longer shows text drawn across art.

---

## Phase 6: Fix Content Routing, OCR Fragments, and Translation Render Eligibility

**Files:**

- Modify: `pipeline/vision_stack/runtime.py`
- Modify: `pipeline/ocr/postprocess.py`
- Modify: `pipeline/ocr/ocr_normalizer.py`
- Modify: `pipeline/translator/translate.py`
- Modify: `pipeline/runtime_profiles.py`
- Test: `pipeline/tests/test_vision_stack_runtime.py`
- Test: `pipeline/tests/test_ocr_normalizer.py`
- Test: `pipeline/tests/test_translate_context.py`
- Test: `pipeline/tests/test_runtime_profiles.py`

**Step 1: Define render eligibility explicitly**

Add or centralize a function that returns:

```python
{
    "renderable": bool,
    "cleanup_allowed": bool,
    "translate_allowed": bool,
    "reason": "dialogue|narration|sign|sfx|credit|url|noise|geometry_untrusted"
}
```

Rules:

- dialogue/narration: translate and render only with trusted geometry
- sign: translate/render only if source removal can be safe
- SFX/title/credits/url/watermark: preserve or review, not normal dialogue
- OCR fragments: repair/merge or review, not render as standalone dialogue

**Step 2: Add tests for broken fragments**

Use `ANCE, VOCE` style partial fragments and run-on splits. Expected:

- if context repair can reconstruct the source, merge before translation
- otherwise mark `ocr_fragment_needs_review` and block rendering

**Step 3: Ensure skipped content never enters inpaint/render**

Any layer with `skip_processing=true` or `content_class in {"noise", "url_watermark", "credit"}` must not:

- enter mask destruction
- enter final typeset
- be counted as a missing-balloon render blocker

**Step 4: Verify**

Run:

```powershell
N:\TraduzAI\pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_vision_stack_runtime.py pipeline\tests\test_ocr_normalizer.py pipeline\tests\test_translate_context.py pipeline\tests\test_runtime_profiles.py -q
```

Fresh debug expectation:

- `cover_noise_rendered_as_dialogue_count == 0`
- `sign_rendered_as_narration_count == 0`
- broken fragments are not rendered as large standalone text.

---

## Phase 7: Upgrade Export Gate to a Real Visual Quality Gate

**Files:**

- Modify: `pipeline/qa/export_gate.py`
- Modify: `pipeline/qa/translation_qa.py`
- Modify: `tools/analyze_e2e_debug.py`
- Modify: `pipeline/debug_tools/report.py`
- Test: `pipeline/tests/test_export_gate.py`
- Test: `pipeline/tests/test_export_gate_debug_consistency.py`
- Test: `pipeline/tests/test_qa_flag_propagation_v2.py`
- Test: `pipeline/tests/test_analyze_e2e_debug.py`

**Step 1: Separate machine trace invariants from visual quality invariants**

Strict debug audit should keep checking traceability:

- render plan/project match
- trace IDs
- QA issue traceability
- contact sheets

Visual quality audit should check:

- `visual_blocker_count == 0`
- no source residual in renderable balloons
- no render on art
- no destructive mask outside balloon

**Step 2: Add crop paths to visual blockers**

Each `visual_blockers.jsonl` row should include direct linked artifacts:

- render plan row
- layout row
- mask decision row
- inpaint decision row
- crop image path from visual review pack

**Step 3: Make PASS meaningful**

`export_gate.status == PASS` should mean the exported image is visually acceptable under current automated checks. Do not allow a pass when:

- source text remains under translated text
- text is rendered onto art
- render/project geometry diverges
- renderable text lacks trusted balloon geometry

**Step 4: Verify**

Run:

```powershell
N:\TraduzAI\pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_export_gate.py pipeline\tests\test_export_gate_debug_consistency.py pipeline\tests\test_qa_flag_propagation_v2.py pipeline\tests\test_analyze_e2e_debug.py -q
```

---

## Phase 8: Re-run Full Chapter Debug and Decide Default Runtime Mode

**Files:**

- No production file changes unless Phase 8 reveals a missed issue.
- Use: `docs/debug/e2e_pipeline_debug_guide.md`
- Use: `tools/analyze_e2e_debug.py`
- Use: `tools/build_e2e_visual_review.py`

**Step 1: Run A baseline**

```powershell
$root = "N:\TraduzAI\DEBUGM\runs\2026-05-19_chapter1_visual_fix_validation_<timestamp>"
N:\TraduzAI\pipeline\venv\Scripts\python.exe N:\TraduzAI\pipeline\main.py --input "C:\Users\PICHAU\Downloads\Chapter 1" --work "Chapter 1" --source-lang en --target pt-BR --mode real --debug --export-mode with_warnings --output "$root\A_baseline_debug"
```

**Step 2: Run B skip-inpaint control**

```powershell
N:\TraduzAI\pipeline\venv\Scripts\python.exe N:\TraduzAI\pipeline\main.py --input "C:\Users\PICHAU\Downloads\Chapter 1" --work "Chapter 1" --source-lang en --target pt-BR --mode real --debug --skip-inpaint --export-mode with_warnings --output "$root\B_skip_inpaint"
```

**Step 3: Run C fast fill**

```powershell
$env:TRADUZAI_STRIP_FAST_WHITE_INPAINT = "1"
$env:TRADUZAI_STRIP_FAST_LOCAL_INPAINT = "1"
N:\TraduzAI\pipeline\venv\Scripts\python.exe N:\TraduzAI\pipeline\main.py --input "C:\Users\PICHAU\Downloads\Chapter 1" --work "Chapter 1" --source-lang en --target pt-BR --mode real --debug --export-mode with_warnings --output "$root\C_fast_fill"
Remove-Item Env:\TRADUZAI_STRIP_FAST_WHITE_INPAINT -ErrorAction SilentlyContinue
Remove-Item Env:\TRADUZAI_STRIP_FAST_LOCAL_INPAINT -ErrorAction SilentlyContinue
```

**Step 4: Run D strict**

```powershell
N:\TraduzAI\pipeline\venv\Scripts\python.exe N:\TraduzAI\pipeline\main.py --input "C:\Users\PICHAU\Downloads\Chapter 1" --work "Chapter 1" --source-lang en --target pt-BR --mode real --debug --strict --output "$root\D_strict_export_gate"
```

Expected:

- D exits `0` only when export gate passes.
- If D exits `2`, the blocker crop must explain a real visual issue.

**Step 5: Analyze and generate visual review**

```powershell
N:\TraduzAI\pipeline\venv\Scripts\python.exe N:\TraduzAI\tools\analyze_e2e_debug.py "$root" --write-report --strict-debug-audit
N:\TraduzAI\pipeline\venv\Scripts\python.exe N:\TraduzAI\tools\build_e2e_visual_review.py "$root"
```

**Step 6: Decide default**

- If A/D are clean and C is visually equivalent with lower runtime, enable fast fill for safe white-balloon cases only.
- If C has any unique visual blocker, keep C opt-in and ship the quality path first.
- B remains a debug/control mode, not a quality mode.

---

## Phase 9: Final Regression Bundle

Run the focused bundle first:

```powershell
N:\TraduzAI\pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_render_plan_trace_integrity.py pipeline\tests\test_typesetting_renderer.py pipeline\tests\test_analyze_e2e_debug.py pipeline\tests\test_strip_run.py pipeline\tests\test_page_cleanup_breakdown.py pipeline\tests\test_mask_builder.py pipeline\tests\test_mask_chain_debug.py pipeline\tests\test_vision_stack_inpainter.py pipeline\tests\test_render_geometry.py pipeline\tests\test_export_gate.py pipeline\tests\test_qa_flag_propagation_v2.py -q
```

Then run the broader pipeline suite if the focused bundle is green:

```powershell
N:\TraduzAI\pipeline\venv\Scripts\python.exe -m pytest pipeline\tests -q
```

Also run diff hygiene:

```powershell
git diff --check -- pipeline tools docs
```

Final acceptance requires:

- all focused tests pass
- full debug A/B/C/D run completed
- analyzer strict audit passes
- visual review pack generated
- final report lists no unexplained visual blockers

---

## Recommended Execution Order

1. Phase 0: visual review tool.
2. Phase 1: render plan/project contract.
3. Phase 2: cleanup safety.
4. Phase 3: missing balloon and geometry trust.
5. Phase 4: mask/inpaint conservatism.
6. Phase 5: typesetting placement.
7. Phase 6: content routing/OCR fragments.
8. Phase 7: visual export gate.
9. Phase 8/9: full chapter validation and regression bundle.

This order matters. If geometry is not coherent first, later visual fixes can look correct on the image while `project.json`, editor import, and QA point to different boxes.

---

## Parallelization Notes

These can run in parallel after Phase 1 is complete:

- Mask/inpaint worker: Phase 4.
- Typesetting worker: Phase 5.
- OCR/content routing worker: Phase 6.
- QA/reporting worker: Phase 7.

Do not parallelize Phase 1 with downstream fixes. It defines the canonical geometry contract used by every other phase.

---

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Fixing QA counts but not images | Every phase must emit visual review crops and compare final pages. |
| Fast fill hides source text poorly | Keep fast fill candidate-only until residual checks and crop review pass. |
| Cleanup repairs one page and breaks geometry | Cleanup must preserve identity/geometry or write through the same final copyback path. |
| Missing balloon fallback silently draws bad text | Renderable text without trusted balloon becomes a blocking QA issue, not a fallback render. |
| Tests depend on protected manga images | Use synthetic unit tests; Chapter 1 remains a local manual E2E fixture. |
| Export gate becomes too strict for credits/title art | Route credits/title/url/SFX as preserve/review classes so dialogue quality gates do not punish intentional preservation. |

---

## Definition of Done

- `docs/debug/e2e_pipeline_debug_guide.md` reflects the final run contract.
- `tools/analyze_e2e_debug.py` reports strict debug audit green.
- The final A/B/C/D report includes the visual review pack.
- `D_strict_export_gate` exits `0` on the corrected run.
- The final translated images no longer show the current failure classes: doubled source+translation, text over art, missing balloon render, overbroad mask, residual source text under PT-BR, and broken OCR fragments rendered as dialogue.

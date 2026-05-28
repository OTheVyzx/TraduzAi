# Detect/Mask/OCR/Typeset/QA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the regression chain where bad OCR merges, destructive geometry normalization, weak visual QA propagation, and overly broad masks produce poor translated bands.

**Architecture:** Treat the bad chapter as a pipeline contract failure, not a renderer-only bug. Preserve geometry from OCR through layout, project.json, mask building, render QA, and export gate; make aggressive gates diagnostic first and only promote them after measured incidence drops.

**Tech Stack:** Python 3.12 pipeline, pytest, OpenCV masks, PIL/matplotlib typesetting, local project.json schema.

**Status 2026-05-17:** Tasks 1-7 implemented and validated. The full rerun exposed one sign-layout overflow, which was fixed by preserving `content_class` through the renderer/project path and avoiding tiny OCR-pixel capacity locks when a broader `balloon_bbox` is available. Promotion of warning layout flags to blocking export gates remains deferred to a separate stabilization change.

---

## Corrections To Claude Plan

- PR9 is not implemented immediately. It remains a post-rerun stabilization gate because promoting high layout flags to P0 before PR0-PR8 metrics can block too many current chapters.
- PR0 needs a pure helper returning merge-veto reason/details so tests can assert decisions without depending on decision-log side effects. The original `union_area > sum_area * 2` threshold does not catch band_017 or band_043 because their bboxes overlap; use dominance/asymmetry and partial containment metrics instead.
- PR1 must preserve connected metadata in both `build_text_layer()` and `_normalize_text_layer_for_renderer()`, then avoid destructive sanitize in `_sync_page_legacy_aliases()`.
- PR2 must keep `TRADUZAI_SIMPLE_LAYOUT_ONLY=1` as rollback and remove the dead early returns in both `enrich_page_layout()` and `build_render_blocks()`.
- PR3 must propagate QA flags from copied render blocks back to `ocr_page["texts"]` by stable textual `id`, not Python object identity. `_source_text_id` currently stores `id(text)` and is not persistent enough for project.json.
- PR4 must separate critical export blockers from warning-only layout review flags and expose `needs_review`.
- PR5-PR6 must add mask diagnostics without making `build_mask_regions()` depend on nonexistent `balloon_id`; use `balloon_bbox` IoU and special-class veto in both merge passes.
- PR8 should extend `pipeline/ocr/postprocess.py` instead of creating a new classifier module.
- PR7 must not enable fast fill for `balanced` yet. Current tests document balanced as behavior-compatible; enable only for `performance` or keep opt-in until visual rerun confirms safety.

## Task 1: OCR Merge Veto

**Files:**
- Modify: `pipeline/vision_stack/runtime.py`
- Test: `pipeline/tests/test_vision_stack_runtime.py`

- [x] Add helper `_ocr_cluster_merge_veto_reason(texts, region_bbox) -> tuple[str, dict] | None`.
- [x] Veto two-text clusters when one bbox dominates the other while they only partially overlap, matching band_017/band_043 where `union/sum` is close to 1 because the boxes overlap.
- [x] Keep a separate white-pair gap/alignment veto for genuinely separated white balloons.
- [x] Run the veto helper before the `len(texts) >= 3` auto-merge path and from pair-subset selection.
- [x] Use the helper inside `_should_merge_ocr_cluster()` and `_merge_ocr_clusters()` decision logging.
- [x] Add tests for band_017 and band_043 bboxes plus one close stacked positive case.
- [x] Run `pipeline/venv/Scripts/python.exe -m pytest -q pipeline/tests/test_vision_stack_runtime.py -k "should_merge_ocr_cluster"`.

## Task 2: Geometry-Preserving Normalize

**Files:**
- Modify: `pipeline/layout/simple_text_geometry.py`
- Modify: `pipeline/layout/balloon_layout.py`
- Modify: `pipeline/typesetter/renderer.py`
- Modify: `pipeline/main.py`
- Test: `pipeline/tests/test_simple_text_geometry.py`
- Test: `pipeline/tests/test_main_emit.py`

- [x] Add `normalize_text_geometry()` that fills missing bbox aliases but preserves connected metadata.
- [x] Rename destructive path to `sanitize_for_simple_text_only()` and keep `sanitize_simple_text_geometry` as compatibility alias.
- [x] Replace destructive sanitize at layout/render/project aliases with the preserving normalize, except the intentional virtual-lobe split path.
- [x] Preserve connected metadata and `layout_group_size` in `build_text_layer()`, `_normalize_text_layer_for_renderer()`, `_sync_page_legacy_aliases()`, and the project-json writer path.
- [x] Update tests that currently assert connected metadata is stripped by default.
- [x] Run focused simple geometry, main emit, and typesetting layout tests.

## Task 3: Reactivate Full Layout/Renderer Paths

**Files:**
- Modify: `pipeline/layout/balloon_layout.py`
- Modify: `pipeline/typesetter/renderer.py`
- Test: `pipeline/tests/test_layout_analysis.py`
- Test: `pipeline/tests/test_typesetting_layout.py`

- [x] Split the simple layout early-return path behind `TRADUZAI_SIMPLE_LAYOUT_ONLY`.
- [x] Make full layout the default path.
- [x] Change unconditional connected metadata resets in full layout to initialization that preserves existing values until new detection replaces them.
- [x] Make `build_render_blocks()` run the existing full connected code after the initial skip/split preparation.
- [x] Remove the legacy skip list only if reactivated tests pass.
- [x] Run focused connected layout/render tests.

## Task 4: Render QA Propagation And Export Gate

**Files:**
- Modify: `pipeline/typesetter/renderer.py`
- Modify: `pipeline/qa/export_gate.py`
- Test: `pipeline/tests/test_typesetting_renderer.py`
- Test: `pipeline/tests/test_export_gate.py`

- [x] In `render_band_image()`, map `qa_flags` from blocks back to source `ocr_page["texts"]`.
- [x] Ensure OCR text entries have stable textual IDs before strip translation/typeset metadata copy-back.
- [x] Add warning-only export issues for `TEXT_CLIPPED`, `TEXT_OVERFLOW`, `bbox_overreach`, `mask_density_high`, `mask_outside_balloon`, and `balloon_bbox_collapsed_to_text`.
- [x] Keep only explicit critical flags as blockers.
- [x] Add `needs_review` and `review_issue_count` to `evaluate_export_gate()`.
- [x] Run focused renderer/export tests.

## Task 5: Mask Overreach, Balloon Guard, Cluster IoU, Fast Fill

**Files:**
- Modify: `pipeline/inpainter/mask_builder.py`
- Modify: `pipeline/inpainter/__init__.py`
- Modify: `pipeline/runtime_profiles.py`
- Test: `pipeline/tests/test_mask_builder.py`
- Test: `pipeline/tests/test_runtime_profiles.py`

- [x] Add `bbox_overreach_ratio()` using broad bbox versus line/text-pixel bbox.
- [x] Add `qa_flags` diagnostics for overreach, mask density, and outside-balloon pixels.
- [x] Prevent broad raw text search for high-overreach blocks.
- [x] Merge mask regions by `balloon_bbox` IoU when available and veto special classes.
- [x] Keep balanced behavior-compatible; enable fast white/local fills only for `performance` profile env defaults, keep eco disabled.
- [x] Reject both fast white and fast local fill when overreach or outside-balloon critical flags are present.
- [x] Run focused mask/profile tests.

## Task 6: Content Router

**Files:**
- Modify: `pipeline/ocr/postprocess.py`
- Modify: `pipeline/vision_stack/runtime.py`
- Modify: `pipeline/translator/translate.py`
- Test: `pipeline/tests/test_content_classifier.py`
- Test: `pipeline/tests/test_translate_context.py`

- [x] Add `ContentClass`, `classify_content()`, and `split_sfx_inline()` in `ocr/postprocess.py`.
- [x] Classify OCR records in the main OCR finalization path.
- [x] Mark watermark, scanlator credit, TN note, and noise as `skip_processing`.
- [x] Split inline `SFX:` from dialogue text without creating a new module.
- [x] Skip translation for non-dialogue content classes that should remain original.
- [x] Run focused classifier and translation tests.

## Task 7: Regression Rerun And Deferred Stabilization

**Files:**
- Modify: `pipeline/typesetter/renderer.py`
- Modify: `pipeline/main.py`

- [x] Re-run the chapter debug fixture when runtime cost is acceptable.
- [x] Compare `qa_report.json`, `decision_trace.jsonl`, project qa flags, and `pipeline.log` against the Claude v2 failure cases.
- [x] Fix the remaining sign overflow found during the rerun by preserving `content_class` through `build_text_layer()`, `_normalize_text_layer_for_renderer()`, and `_sync_page_legacy_aliases()`.
- [x] Verify the final rerun at `.codex-tmp/chapter1_task7_rerun_20260517_final`: 89 text layers, 0 missing `balloon_bbox`, 5 `dominant_partial_overlap` vetoes, and no `TEXT_CLIPPED`/`TEXT_OVERFLOW` log warnings.
- [x] Only after incidence is below 5%, promote selected HIGH layout flags into P0 in a separate change. Current implementation keeps warning/review flags non-blocking and defers promotion.

## Final Verification

- [x] Focused contract suite: `38 passed`.
- [x] Layout/render/main suite: `216 passed, 10 skipped, 3 deselected, 2 subtests passed`.
- [x] Python compile for touched pipeline modules passed.
- [x] Full `pipeline/tests` run after fixes: `1076 passed, 11 skipped, 11 subtests passed`, with 10 remaining failures classified as out-of-scope/preexisting.
- [x] Full suite excluding those known failures: `1076 passed, 10 skipped, 11 deselected, 11 subtests passed`.
- [x] `git diff --check` passed; only Windows LF/CRLF warnings were emitted.

Known out-of-scope broad-suite failures:

- Missing local fixture images: `002__002.jpg`, `009__001.jpg`, `010__001.jpg`, `012__001.jpg`.
- Legacy Koharu assertions still expecting `comic-text-detector-seg` instead of current `manga-text-segmentation-2025`.
- Legacy CJK fallback test expects `koharu_cjk_fallback` on the quick-skip branch even when the current resolved preset does not enter the Koharu HTTP route.

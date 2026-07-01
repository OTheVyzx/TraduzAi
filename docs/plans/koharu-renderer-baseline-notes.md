# Koharu Renderer Baseline Notes

## Scope

Baseline for the first additive Rust renderer bridge slice. The default renderer remains Python unless `TRADUZAI_RENDERER_BACKEND=koharu_rust`.

## Current Entry Points

- `pipeline/main.py` automatic export uses `typesetter.renderer.run_typesetting()`.
- `pipeline/main.py::render_page_image()` calls `typesetter.renderer._typeset_single_page()`.
- `--render-preview-page` calls the same `_typeset_single_page()` path.
- `--retypeset` reuses `render_page_image()` and `_typeset_single_page()`.

## Renderer Metadata Contract

Python remains the layout and QA authority for this slice. The bridge must not replace:

- `render_bbox`
- `safe_text_box`
- `_debug_safe_text_box`
- `qa_flags`
- `qa_metrics`
- `_render_debug`
- `_render_debug_candidates`
- `_render_debug_skipped`
- `render_plan_final.jsonl`

## Fixture Sources

The mandatory real visual cases are tracked by path, not copied into git:

- `N:\TraduzAI\DEBUGM\runs\2026-05-27_232400_one_second_debug_final_contract_after_qa_metrics_fix\one_second_ch1`
- visual sheet: `N:\TraduzAI\DEBUGM\runs\2026-05-27_232400_one_second_debug_final_contract_after_qa_metrics_fix\_visual_review\one_second_ch1_pages_022_023_037_042_045_053.jpg`
- render plan: `N:\TraduzAI\DEBUGM\runs\2026-05-27_232400_one_second_debug_final_contract_after_qa_metrics_fix\one_second_ch1\debug\e2e\09_typeset\render_plan_final.jsonl`

Mandatory pages:

- p22
- p23
- p37
- p42
- p45
- p53

## Renderer Invariants

- One render block represents one real text/bubble region.
- Rust backend input uses `safe_text_box` only for placement.
- `balloon_bbox` remains debug/compatibility metadata and is not a placement fallback.
- Legacy fields must not decide renderer placement: `tipo`, `kind`, `content_class`, `balloon_type`, `skip_processing`, `preserve_original`.

## First Bridge Slice

- `pipeline/typesetter/backend_contract.py` builds the stable JSON request.
- `src-tauri/renderer-bridge` renders a minimal RGBA PNG from that request.
- `pipeline/typesetter/rust_backend.py` invokes the bridge only when `TRADUZAI_RENDERER_BACKEND=koharu_rust`.
- Python still computes layout, fitting, metadata, and QA.

## Parity Harness

- `pipeline/scripts/compare_render_backends.py` compares Python and `koharu_rust` on renderer golden fixtures.
- Output is written outside tracked fixtures, for example `debug/renderer-backend-parity/`.
- Per case artifacts:
  - `input.png`
  - `python.png`
  - `koharu_rust.png`
  - `diff.png`
- Summary artifacts:
  - `report.json`
  - `contact_sheet.png`
- The checked-in fixture set starts with a synthetic `simple_balloon` case. Real p22/p23/p37/p42/p45/p53 remain referenced by path until we intentionally extract small renderer-only JSON fixtures instead of versioning chapter outputs.

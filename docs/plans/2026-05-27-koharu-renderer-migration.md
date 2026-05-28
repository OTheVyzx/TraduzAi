# Koharu Renderer Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the TraduzAi Python text rasterization path with a Koharu-style Rust renderer while preserving TraduzAi layout contracts, editor preview behavior, QA metadata, and automatic pipeline output.

**Architecture:** Introduce a new Rust renderer bridge as an additive path first, using Koharu's renderer concepts (`tiny-skia`, `skrifa`, `harfrust`, `fontdb`, `fontdue`) behind a stable JSON/IPC contract. Keep Python `pipeline/typesetter/renderer.py` as the layout/QA authority during the first migration, then move fit/layout responsibilities incrementally only after parity images and QA artifacts prove the Rust output is compatible.

**Tech Stack:** Rust, Tauri commands, Python sidecar, JSON lines, `tiny-skia`, `skrifa`, `harfrust`, `fontdb`, `fontdue`, existing `pipeline/typesetter/renderer.py`, existing `project.json` schema.

---

## Non-Negotiable Contracts

- Do not remove `pipeline/typesetter/renderer.py` until the Rust renderer passes visual parity on real chapters.
- Preserve `render_bbox`, `safe_text_box`, `qa_flags`, `_render_debug`, `render_plan_final.jsonl`, and editor preview output semantics.
- Preserve current automatic pipeline entry points: `pipeline/main.py config.json`, `--retypeset`, and `--render-preview-page`.
- Keep a feature flag fallback: `TRADUZAI_RENDERER_BACKEND=python|koharu_rust`, defaulting to `python` until validation is complete.
- Do not vendor the whole Koharu app. Port only the renderer crate concepts and required code, with attribution in docs if code is copied.
- Treat Koharu's renderer as the raster/layout engine, not as a replacement for detection, OCR, translation, inpaint, or project schema.
- Renderer input invariant: one render block must represent one real text/balloon region. OCR or bubble merges must be repaired or split before the block reaches the renderer.
- The primary geometry source for Rust rendering is the real `safe_text_box` derived from BubbleMask/bubble-safe area. `balloon_bbox` is allowed only as debug/compatibility metadata and must never decide render placement, fitting, masking, or fallback.
- The renderer contract must not make decisions from text classification fields such as legacy type/tipo, `kind`, `content_class`, `balloon_type`, `skip_processing`, or `preserve_original`. These may remain upstream pipeline metadata, but they must not be part of Rust renderer input examples or placement logic.

## Migration Strategy

Phase 1 creates a standalone Rust renderer executable/library and proves it can draw text sprites from simple JSON.

Phase 2 plugs it into Python as an optional raster backend while Python still computes layout, fitting, QA, and metadata.

Phase 3 ports selected Koharu layout features: shaping, fallback fonts, bubble mask collision, supersampling, stroke/bold/italic.

Phase 4 enables the Rust path in editor preview and retypeset behind a flag.

Phase 5 runs real chapter parity, then flips defaults only after measurable quality and stability wins.

---

### Task 0: Lock Real Input Fixtures and Renderer Invariants

**Files:**
- Read: real project/page artifacts for pages p22, p23, p37, p42, p45, and p53
- Read: `N:\TraduzAI\pipeline\typesetter\renderer.py`
- Read: `N:\TraduzAI\pipeline\layout\balloon_layout.py`
- Create/Modify later during execution: baseline fixture files under `N:\TraduzAI\pipeline\tests\fixtures\renderer_golden\`

**Step 1: Collect required real fixtures**

Before any baseline or Rust work, collect real renderer inputs and expected outputs for:

```text
p22
p23
p37
p42
p45
p53
```

These pages are mandatory because they cover the failure modes this migration must not hide behind synthetic examples.

**Step 2: Assert the renderer input invariant**

Document and test the invariant before creating Rust contract payloads:

```text
one render block = one real text/balloon region
```

If OCR output or bubble detection merges multiple visual regions into one block, repair/split that data before rendering. The renderer must not infer separate balloons from text class, legacy type fields, or fallback bounding boxes.

**Step 3: Confirm geometry source order**

The renderer-facing source of truth is:

```text
BubbleMask / real safe_text_box
```

`balloon_bbox` may be carried in debug artifacts for compatibility and visual inspection only. It must not be used as the render box fallback, mask selector, fit authority, or placement decision.

**Step 4: Gate Task 1**

Task 1 cannot start until the p22/p23/p37/p42/p45/p53 fixtures exist or the execution notes explicitly record why a page artifact is missing and what equivalent real page replaces it.

---

### Task 1: Baseline Current Renderer Behavior

**Files:**
- Read: `N:\TraduzAI\pipeline\typesetter\renderer.py`
- Read: `N:\TraduzAI\pipeline\tests\test_typesetting_renderer.py`
- Read: `N:\TraduzAI\pipeline\tests\test_typesetting_layout.py`
- Create: `N:\TraduzAI\docs\plans\koharu-renderer-baseline-notes.md`

**Step 1: Capture renderer entry points**

Document the current call graph:

```text
pipeline/main.py
  run_pipeline() -> typesetter.renderer.run_typesetting()
  render_page_image() -> typesetter.renderer._typeset_single_page()
  _run_render_preview_page() -> typesetter.renderer._typeset_single_page()
  retypeset path -> typesetter.renderer._typeset_single_page()
```

**Step 2: Capture required metadata fields**

List every field that `render_text_block()` mutates or emits:

```text
render_bbox
safe_text_box
font_size
qa_flags
qa_metrics
_render_debug
_render_trace_id
render_plan_* debug artifacts
```

**Step 3: Create golden fixtures**

Pick fixtures including the mandatory real pages from Task 0 plus focused cases:

- one simple white speech balloon
- one long PT-BR translated balloon
- one connected balloon
- one narration box
- one rotated/sign text
- one style override from editor
- one Google/custom font case
- one known previous overflow/tiny-font regression

Save their input JSON and expected rendered PNGs under:

```text
N:\TraduzAI\pipeline\tests\fixtures\renderer_golden\
```

**Step 4: Run current tests**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_typesetting_renderer.py pipeline\tests\test_typesetting_layout.py -x
```

Expected: PASS before migration work begins.

**Step 5: Commit**

```powershell
git add docs/plans/koharu-renderer-baseline-notes.md pipeline/tests/fixtures/renderer_golden
git commit -m "test: capture baseline renderer fixtures"
```

---

### Task 2: Add Renderer Backend Contract

**Files:**
- Create: `N:\TraduzAI\pipeline\typesetter\backend_contract.py`
- Modify: `N:\TraduzAI\pipeline\typesetter\renderer.py`
- Test: `N:\TraduzAI\pipeline\tests\test_typesetting_backend_contract.py`

**Step 1: Write failing contract tests**

Test that the backend payload carries only stable renderer data:

```python
import pytest

from typesetter.backend_contract import build_rust_render_request


def test_build_rust_render_request_preserves_geometry_and_style():
    img_size = (800, 1200)
    text_data = {
        "id": "txt-1",
        "translated": "OLA MUNDO",
        "safe_text_box": [110, 210, 290, 350],
        "style": {"fontFamily": "Arial", "fontSize": 32, "color": "#111111"},
    }

    request = build_rust_render_request(img_size, text_data)

    assert request["image_width"] == 800
    assert request["image_height"] == 1200
    assert request["blocks"][0]["text"] == "OLA MUNDO"
    assert request["blocks"][0]["box"] == [110, 210, 290, 350]
    assert request["blocks"][0]["style"]["font_family"] == "Arial"


def test_build_rust_render_request_rejects_balloon_bbox_fallback():
    img_size = (800, 1200)
    text_data = {
        "id": "txt-1",
        "translated": "OLA MUNDO",
        "balloon_bbox": [100, 200, 300, 360],
        "style": {"fontFamily": "Arial"},
    }

    with pytest.raises(ValueError, match="safe_text_box"):
        build_rust_render_request(img_size, text_data)
```

**Step 2: Run test to verify it fails**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_typesetting_backend_contract.py -v
```

Expected: FAIL because `typesetter.backend_contract` does not exist.

**Step 3: Implement minimal contract**

Create `backend_contract.py` with:

```python
from __future__ import annotations

from typing import Any


def _bbox4(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in value]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def build_rust_render_request(img_size: tuple[int, int], text_data: dict[str, Any]) -> dict[str, Any]:
    width, height = img_size
    box = _bbox4(text_data.get("safe_text_box"))
    if box is None:
        raise ValueError("text_data has no valid safe_text_box")

    style = text_data.get("style") if isinstance(text_data.get("style"), dict) else {}
    return {
        "image_width": int(width),
        "image_height": int(height),
        "blocks": [
            {
                "id": str(text_data.get("id") or text_data.get("text_id") or ""),
                "text": str(text_data.get("translated") or ""),
                "box": box,
                "style": {
                    "font_family": style.get("fontFamily") or style.get("font_family") or text_data.get("font_family"),
                    "font_size": style.get("fontSize") or style.get("font_size") or text_data.get("font_size"),
                    "color": style.get("color") or text_data.get("color") or "#000000",
                    "stroke_color": style.get("strokeColor") or style.get("stroke_color"),
                    "stroke_width": style.get("strokeWidth") or style.get("stroke_width") or 0,
                    "bold": bool(style.get("bold") or text_data.get("bold")),
                    "italic": bool(style.get("italic") or text_data.get("italic")),
                    "align": style.get("textAlign") or style.get("align") or "center",
                },
            }
        ],
    }
```

`render_bbox`, `target_bbox`, `bbox`, and `balloon_bbox` may remain available to Python-side metadata/debug code, but this backend request builder must not use them as placement fallback. If `safe_text_box` is missing or invalid, fail before calling Rust and repair the upstream block.

**Step 4: Run test to verify it passes**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_typesetting_backend_contract.py -v
```

Expected: PASS.

**Step 5: Commit**

```powershell
git add pipeline/typesetter/backend_contract.py pipeline/tests/test_typesetting_backend_contract.py
git commit -m "feat: add renderer backend contract"
```

---

### Task 3: Create Rust Renderer Crate

**Files:**
- Modify: `N:\TraduzAI\Cargo.toml` or create workspace entry if the root Rust workspace is elsewhere
- Create: `N:\TraduzAI\src-tauri\renderer-bridge\Cargo.toml`
- Create: `N:\TraduzAI\src-tauri\renderer-bridge\src\main.rs`
- Create: `N:\TraduzAI\src-tauri\renderer-bridge\src\lib.rs`
- Test: `N:\TraduzAI\src-tauri\renderer-bridge\tests\render_contract.rs`

**Step 1: Write failing Rust contract test**

The test should feed JSON and assert a PNG exists and has non-transparent pixels.

```rust
#[test]
fn renders_simple_text_request() {
    let request = traduzai_renderer_bridge::RenderRequest {
        image_width: 320,
        image_height: 240,
        blocks: vec![traduzai_renderer_bridge::RenderBlock {
            id: "a".into(),
            text: "OLA".into(),
            bbox: [40, 40, 220, 120],
            style: Default::default(),
        }],
    };

    let image = traduzai_renderer_bridge::render_to_rgba(&request).unwrap();
    assert_eq!(image.width(), 320);
    assert_eq!(image.height(), 240);
    assert!(image.pixels().any(|p| p.0[3] > 0));
}
```

**Step 2: Run test to verify it fails**

Run:

```powershell
cd src-tauri\renderer-bridge
cargo test renders_simple_text_request
```

Expected: FAIL because the crate does not exist or the API is missing.

**Step 3: Implement minimal Rust renderer**

Implement only:

- serde JSON request/response structs
- `FontBook`
- `TextLayout`
- `TinySkiaRenderer`
- one font fallback path
- RGBA output

Use Koharu code as reference from:

```text
N:\TraduzAI\koharu\koharu-renderer\src\font.rs
N:\TraduzAI\koharu\koharu-renderer\src\layout.rs
N:\TraduzAI\koharu\koharu-renderer\src\shape.rs
N:\TraduzAI\koharu\koharu-renderer\src\renderer.rs
N:\TraduzAI\koharu\koharu-renderer\src\types.rs
```

**Step 4: Add CLI mode**

CLI contract:

```powershell
renderer-bridge.exe --request request.json --output output.png
```

It must return:

- exit code `0` on success
- JSON status on stdout
- human-readable errors on stderr

**Step 5: Run test to verify it passes**

Run:

```powershell
cd src-tauri\renderer-bridge
cargo test
```

Expected: PASS.

**Step 6: Commit**

```powershell
git add src-tauri/renderer-bridge Cargo.toml Cargo.lock
git commit -m "feat: add rust renderer bridge"
```

---

### Task 4: Add Python Optional Backend Invocation

**Files:**
- Modify: `N:\TraduzAI\pipeline\typesetter\renderer.py`
- Modify: `N:\TraduzAI\pipeline\typesetter\backend_contract.py`
- Create: `N:\TraduzAI\pipeline\typesetter\rust_backend.py`
- Test: `N:\TraduzAI\pipeline\tests\test_typesetting_rust_backend.py`

**Step 1: Write failing backend selection tests**

```python
import os

from typesetter.rust_backend import rust_renderer_enabled


def test_rust_renderer_disabled_by_default(monkeypatch):
    monkeypatch.delenv("TRADUZAI_RENDERER_BACKEND", raising=False)
    assert rust_renderer_enabled() is False


def test_rust_renderer_enabled_by_env(monkeypatch):
    monkeypatch.setenv("TRADUZAI_RENDERER_BACKEND", "koharu_rust")
    assert rust_renderer_enabled() is True
```

**Step 2: Run test to verify it fails**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_typesetting_rust_backend.py -v
```

Expected: FAIL because `rust_backend.py` does not exist.

**Step 3: Implement `rust_backend.py`**

Responsibilities:

- locate `renderer-bridge.exe`
- write request JSON to a temp directory
- call the executable with `subprocess.run`
- load output PNG as `PIL.Image`
- return the rendered RGBA sprite or page
- raise a typed exception on failure

**Step 4: Wire without changing default**

In `render_text_block()`:

- keep current Python rendering as default
- if `TRADUZAI_RENDERER_BACKEND=koharu_rust`, build the request and call Rust
- if Rust fails, log warning and fall back to Python unless `TRADUZAI_RENDERER_STRICT=1`

**Step 5: Run focused tests**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_typesetting_rust_backend.py pipeline\tests\test_typesetting_renderer.py -x
```

Expected: PASS with default Python backend.

**Step 6: Commit**

```powershell
git add pipeline/typesetter/rust_backend.py pipeline/typesetter/backend_contract.py pipeline/typesetter/renderer.py pipeline/tests/test_typesetting_rust_backend.py
git commit -m "feat: add optional rust renderer backend"
```

---

### Task 5: Preserve Render Metadata

**Files:**
- Modify: `N:\TraduzAI\pipeline\typesetter\renderer.py`
- Modify: `N:\TraduzAI\pipeline\typesetter\rust_backend.py`
- Test: `N:\TraduzAI\pipeline\tests\test_typesetting_renderer.py`

**Step 1: Add tests for metadata parity**

Add tests asserting the Rust backend path still populates:

```python
assert text_data["render_bbox"]
assert text_data["safe_text_box"]
assert "_render_debug" in text_data
assert text_data["qa_metrics"]["render_fit"]
```

Mock the Rust executable response so this test does not depend on a compiled binary.

**Step 2: Run test to verify it fails**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_typesetting_renderer.py -k "rust and metadata" -v
```

Expected: FAIL until metadata copying is implemented.

**Step 3: Implement metadata adapter**

Keep metadata generation in Python:

- Python computes plan and fit.
- Rust renders pixels from the selected plan.
- Python writes `render_bbox`, `safe_text_box`, `_render_debug`, and QA flags exactly as before.

**Step 4: Run focused tests**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_typesetting_renderer.py -x
```

Expected: PASS.

**Step 5: Commit**

```powershell
git add pipeline/typesetter/renderer.py pipeline/typesetter/rust_backend.py pipeline/tests/test_typesetting_renderer.py
git commit -m "fix: preserve render metadata with rust backend"
```

---

### Task 6: Port Font Resolution and Fallbacks

**Files:**
- Modify: `N:\TraduzAI\src-tauri\renderer-bridge\src\font.rs`
- Modify: `N:\TraduzAI\src-tauri\renderer-bridge\src\lib.rs`
- Modify: `N:\TraduzAI\pipeline\typesetter\backend_contract.py`
- Test: `N:\TraduzAI\src-tauri\renderer-bridge\tests\font_fallback.rs`
- Test: `N:\TraduzAI\pipeline\tests\test_typesetting_backend_contract.py`

**Step 1: Add failing tests**

Cover:

- project font by family or PostScript name
- fallback to Arial or ComicNeue equivalent
- missing glyph fallback for symbols/accented PT-BR
- Google font cached file path if available

**Step 2: Implement minimal font resolver**

Use `fontdb` for system fonts and explicit font files from:

```text
N:\TraduzAI\fonts\
```

Do not implement network Google Fonts here. Only consume already cached/local font files.

**Step 3: Run tests**

Run:

```powershell
cd src-tauri\renderer-bridge
cargo test font
cd N:\TraduzAI
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_typesetting_backend_contract.py -v
```

Expected: PASS.

**Step 4: Commit**

```powershell
git add src-tauri/renderer-bridge pipeline/typesetter/backend_contract.py pipeline/tests/test_typesetting_backend_contract.py
git commit -m "feat: port renderer font resolution"
```

---

### Task 7: Add Bubble Mask Collision Path

**Files:**
- Modify: `N:\TraduzAI\src-tauri\renderer-bridge\src\layout.rs`
- Modify: `N:\TraduzAI\src-tauri\renderer-bridge\src\lib.rs`
- Modify: `N:\TraduzAI\pipeline\typesetter\backend_contract.py`
- Test: `N:\TraduzAI\src-tauri\renderer-bridge\tests\bubble_mask.rs`

**Step 1: Add failing mask tests**

Use a synthetic grayscale mask:

- `0` outside balloon
- `1` inside balloon
- text sprite must not overlap outside the `1` area after fit

**Step 2: Port `BubbleIndex` concept**

Port only the required idea from Koharu:

```text
koharu-renderer/src/text/latin.rs -> BubbleIndex + layout box lookup
koharu-app/src/renderer.rs -> fit_rendered_with_mask_collision
```

**Step 3: Integrate optional mask in request**

Extend JSON request:

```json
{
  "bubble_mask_path": "optional/path/to/mask.png",
  "blocks": [
    { "bubble_id": 1 }
  ]
}
```

**Step 4: Run tests**

Run:

```powershell
cd src-tauri\renderer-bridge
cargo test bubble_mask
```

Expected: PASS.

**Step 5: Commit**

```powershell
git add src-tauri/renderer-bridge pipeline/typesetter/backend_contract.py
git commit -m "feat: add bubble-aware rust text fitting"
```

---

### Task 8: Editor Preview Integration

**Files:**
- Modify: `N:\TraduzAI\pipeline\main.py`
- Modify: `N:\TraduzAI\src-tauri\src\commands\pipeline.rs`
- Modify: `N:\TraduzAI\src\components\editor\toolbar\RenderStatusBadge.tsx`
- Test: `N:\TraduzAI\pipeline\tests\test_main_emit.py`

**Step 1: Add tests for preview backend flag**

Test that `--render-preview-page` forwards/uses the renderer backend without committing `project.json`.

**Step 2: Add status visibility**

Expose the backend in preview status/debug:

```text
renderer_backend: python
renderer_backend: koharu_rust
```

**Step 3: Keep output path identical**

The editor must still receive the same `render_preview_path` contract from Rust/Tauri.

**Step 4: Run tests**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_main_emit.py -k "render_preview" -x
npm run test -- --run
```

Expected: PASS.

**Step 5: Commit**

```powershell
git add pipeline/main.py src-tauri/src/commands/pipeline.rs src/components/editor/toolbar/RenderStatusBadge.tsx pipeline/tests/test_main_emit.py
git commit -m "feat: expose renderer backend in editor preview"
```

---

### Task 9: Visual Parity Harness

**Files:**
- Create: `N:\TraduzAI\pipeline\scripts\compare_render_backends.py`
- Create: `N:\TraduzAI\pipeline\tests\test_renderer_backend_parity.py`
- Modify: `N:\TraduzAI\docs\plans\koharu-renderer-baseline-notes.md`

**Step 1: Add script**

Script behavior:

```powershell
pipeline\venv\Scripts\python.exe pipeline\scripts\compare_render_backends.py --fixture-dir pipeline\tests\fixtures\renderer_golden --out-dir debug\renderer-backend-parity
```

It must generate:

```text
debug/renderer-backend-parity/<case>/python.png
debug/renderer-backend-parity/<case>/koharu_rust.png
debug/renderer-backend-parity/<case>/diff.png
debug/renderer-backend-parity/report.json
debug/renderer-backend-parity/contact_sheet.png
```

**Step 2: Add metrics**

Report:

- pixel diff percentage
- alpha coverage diff
- text bbox diff
- overflow flags
- font size chosen
- whether fallback happened

**Step 3: Run parity**

Run:

```powershell
pipeline\venv\Scripts\python.exe pipeline\scripts\compare_render_backends.py --fixture-dir pipeline\tests\fixtures\renderer_golden --out-dir debug\renderer-backend-parity
```

Expected: report and contact sheet are created. Do not require exact pixel equality.

**Step 4: Commit**

```powershell
git add pipeline/scripts/compare_render_backends.py pipeline/tests/test_renderer_backend_parity.py docs/plans/koharu-renderer-baseline-notes.md
git commit -m "test: add renderer backend visual parity harness"
```

---

### Task 10: Real Chapter Gate

**Files:**
- Modify: `N:\TraduzAI\pipeline\scripts\compare_render_backends.py`
- Create: `N:\TraduzAI\docs\plans\koharu-renderer-rollout-report.md`

**Step 1: Select real projects**

Use at least:

- one manga page set with dense dialogue
- one manhwa/strip chapter
- one page with connected balloons
- one page with signs/narration

**Step 2: Run Python backend**

Run automatic pipeline normally:

```powershell
pipeline\venv\Scripts\python.exe pipeline\main.py config.json
```

**Step 3: Run Rust backend**

Run:

```powershell
$env:TRADUZAI_RENDERER_BACKEND='koharu_rust'
pipeline\venv\Scripts\python.exe pipeline\main.py config.json
Remove-Item Env:\TRADUZAI_RENDERER_BACKEND
```

**Step 4: Inspect outputs**

Compare:

- final rendered images
- `project.json`
- `qa_report.json`
- `debug/e2e/09_typeset/render_plan_final.jsonl`
- contact sheets

**Step 5: Define pass/fail criteria**

Pass only if:

- no increase in `TEXT_OVERFLOW`
- no increase in `fit_below_minimum_legible`
- no missing `render_bbox`
- no editor preview mismatch
- visual review accepts at least 90% of sampled pages
- render time does not regress by more than 20%

**Step 6: Commit report**

```powershell
git add docs/plans/koharu-renderer-rollout-report.md
git commit -m "docs: record rust renderer rollout evidence"
```

---

### Task 11: Flip Default Behind App Setting

**Files:**
- Modify: `N:\TraduzAI\pipeline\typesetter\rust_backend.py`
- Modify: `N:\TraduzAI\pipeline\typesetter\renderer.py`
- Modify: `N:\TraduzAI\src\lib\stores\appStore.ts`
- Modify: `N:\TraduzAI\src\lib\tauri.ts`
- Modify: `N:\TraduzAI\src-tauri\src\commands\pipeline.rs`
- Test: relevant Python, Rust, and UI tests

**Step 1: Add setting**

Add app setting:

```text
rendererBackend = "python" | "koharu_rust"
```

Default remains `python` until Task 10 passes.

**Step 2: Pass setting to pipeline**

Ensure automatic pipeline, retypeset, and render preview all pass the backend selection consistently.

**Step 3: Add visible diagnostic only where useful**

Use debug/status surfaces, not noisy user-facing UI.

**Step 4: Run verification**

Run:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_typesetting_renderer.py pipeline\tests\test_typesetting_layout.py pipeline\tests\test_main_emit.py -x
cargo test --manifest-path src-tauri/Cargo.toml
npm run test -- --run
```

Expected: PASS.

**Step 5: Commit**

```powershell
git add pipeline/typesetter src src-tauri
git commit -m "feat: add selectable renderer backend"
```

---

### Task 12: Remove Python Rasterization Only After Stability Window

**Files:**
- Modify: `N:\TraduzAI\pipeline\typesetter\renderer.py`
- Modify: `N:\TraduzAI\pipeline\tests\test_typesetting_renderer.py`
- Modify: `N:\TraduzAI\docs\plans\koharu-renderer-rollout-report.md`

**Step 1: Wait for evidence**

Do not start this task until:

- Task 10 passes on real chapters.
- The editor has been used with Rust preview/retypeset without regressions.
- Fallback logs show no frequent Rust failures.

**Step 2: Move Python rasterizer to legacy path**

Keep the old renderer available as:

```text
TRADUZAI_RENDERER_BACKEND=python_legacy
```

for one release cycle.

**Step 3: Delete only duplicated dead code**

Do not delete layout/QA code that still owns:

- connected balloon grouping
- safe-area decisions
- render debug artifacts
- project schema normalization

**Step 4: Final verification**

Run full target suite:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_typesetting_layout.py pipeline\tests\test_typesetting_renderer.py pipeline\tests\test_layout_analysis.py pipeline\tests\test_main_emit.py -x
cargo test --manifest-path src-tauri/Cargo.toml
npm run test -- --run
```

Expected: PASS.

**Step 5: Commit**

```powershell
git add pipeline/typesetter pipeline/tests docs/plans/koharu-renderer-rollout-report.md
git commit -m "refactor: retire legacy python text rasterizer"
```

---

## Recommended First Execution Slice

Do not start by porting all of Koharu. Start with this slice:

1. Task 0: lock real p22/p23/p37/p42/p45/p53 fixtures and renderer input invariants.
2. Task 1: baseline fixtures and current behavior notes.
3. Task 2: stable JSON backend contract.
4. Task 3: Rust renderer bridge that renders one simple block.
5. Task 4: optional Python invocation with default fallback.
6. Task 5: metadata parity.

This gives a working vertical path without risking the current automatic pipeline.

## Validation Commands

Use these repeatedly during the migration:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_typesetting_layout.py pipeline\tests\test_typesetting_renderer.py pipeline\tests\test_layout_analysis.py -x
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_main_emit.py -k "render" -x
cargo test --manifest-path src-tauri/Cargo.toml
npm run test -- --run
```

For visual evidence:

```powershell
pipeline\venv\Scripts\python.exe pipeline\scripts\compare_render_backends.py --fixture-dir pipeline\tests\fixtures\renderer_golden --out-dir debug\renderer-backend-parity
```

## Rollback Plan

- Keep `TRADUZAI_RENDERER_BACKEND=python` as the default until rollout evidence passes.
- Any Rust backend exception falls back to Python unless `TRADUZAI_RENDERER_STRICT=1`.
- If editor preview diverges, disable Rust only for `--render-preview-page` first.
- If automatic pipeline diverges, disable Rust globally and keep the parity harness for debugging.

## Open Decisions

- Whether to copy Koharu renderer source directly or implement a smaller TraduzAi-specific crate inspired by it.
- Whether the Rust bridge should live under `src-tauri/renderer-bridge` or as a top-level workspace crate.
- Whether packaging should ship `renderer-bridge.exe` as a sidecar binary or link it into Tauri commands directly.
- Whether Google Fonts should be handled only by the app layer or by the Rust bridge.
- Whether mask-based fitting should consume existing TraduzAi mask artifacts or new renderer-specific masks.

## Execution Handoff

Plan complete and saved to `docs/plans/2026-05-27-koharu-renderer-migration.md`.

Recommended execution mode: implement Tasks 0-5 first in a dedicated branch/worktree, stop for visual review, then continue with Tasks 6-12 only after evidence.

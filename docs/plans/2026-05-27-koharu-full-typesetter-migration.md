# Koharu Full Typesetter Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the TraduzAi Python typesetting/rendering stack with a Rust `typesetter_v2` based on Koharu's renderer architecture, including font resolution, shaping, layout, fitting, rasterization, sprite composition, and render QA outputs.

**Architecture:** Build a new Rust renderer/typesetter executable and library that owns final text layout and pixels. Python becomes a temporary adapter that converts current pipeline/editor page data into the v2 contract and consumes v2 output until the Rust path is wired directly into Tauri and the old Python raster/layout path can be retired.

**Tech Stack:** Rust, Tauri v2, Python adapter, JSON contracts, `tiny-skia`, `skrifa`, `harfrust`, `fontdb`, `fontdue`, `image`, existing TraduzAi `project.json`, existing editor preview/retypeset commands, Koharu renderer source as reference.

---

## Definition Of "Full Migration"

This migration is not just swapping PIL/FT2Font drawing for Rust drawing. The Rust v2 stack must own:

- font discovery and fallback
- text shaping, including bidi and mixed-script text
- line breaking, wrapping, alignment, and hyphenation where applicable
- font-size fitting
- bubble-mask-aware fitting
- rotated/sign/narration rendering policy
- stroke, bold, italic, fill color, and basic effects
- per-text sprite generation
- final page composition
- output geometry and QA metrics consumed by `project.json`, `qa_report.json`, and editor preview

The Python v1 typesetter remains available only as a rollback path until v2 passes real chapter gates.

## Key Risk

The current `pipeline/typesetter/renderer.py` contains years of TraduzAi-specific heuristics, QA flags, and editor contracts. A full migration must preserve the externally visible behavior even if the internal implementation changes completely.

## New Runtime Switch

Use a new high-level switch:

```text
TRADUZAI_TYPESETTER=v1_python
TRADUZAI_TYPESETTER=v2_koharu
```

Default remains `v1_python` until real chapter validation passes. Do not use `v2_koharu` as default during development.

---

### Task 1: Freeze V1 Behavior With Golden Fixtures

**Files:**
- Read: `N:\TraduzAI\pipeline\typesetter\renderer.py`
- Read: `N:\TraduzAI\pipeline\main.py`
- Read: `N:\TraduzAI\src-tauri\src\commands\pipeline.rs`
- Create: `N:\TraduzAI\pipeline\tests\fixtures\typesetter_v2_golden\README.md`
- Create: `N:\TraduzAI\pipeline\scripts\capture_typesetter_golden.py`
- Test: `N:\TraduzAI\pipeline\tests\test_typesetter_golden_capture.py`

**Step 1: Write failing test for golden capture CLI**

```python
from pathlib import Path

from pipeline.scripts.capture_typesetter_golden import build_case_manifest


def test_build_case_manifest_contains_required_outputs(tmp_path):
    case_dir = tmp_path / "case_simple"
    case_dir.mkdir()
    manifest = build_case_manifest(case_dir, page_index=0)

    assert manifest["page_index"] == 0
    assert manifest["inputs"]["page_json"].endswith("page.json")
    assert manifest["outputs"]["rendered_png"].endswith("rendered.png")
    assert manifest["outputs"]["project_page_json"].endswith("project_page.json")
```

**Step 2: Run test to verify it fails**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_typesetter_golden_capture.py -v
```

Expected: FAIL because capture script does not exist.

**Step 3: Implement golden capture script**

The script must capture, per page:

```text
source image
inpainted image
optional mask image
page.json input
current v1 rendered.png
current text_layers after render
render_plan_final.jsonl slice
qa flags slice
```

**Step 4: Capture minimum fixture set**

Create fixtures for:

- simple speech balloon
- long PT-BR dialogue
- connected balloon
- narration box
- sign/rotated text
- editor manual style override
- Google/custom font
- prior overflow regression
- strip/manhwa vertical page
- noisy OCR duplicate case

**Step 5: Run capture on current checkout**

```powershell
pipeline\venv\Scripts\python.exe pipeline\scripts\capture_typesetter_golden.py --out pipeline\tests\fixtures\typesetter_v2_golden
```

Expected: fixture folders and manifest are created.

**Step 6: Commit**

```powershell
git add pipeline/scripts/capture_typesetter_golden.py pipeline/tests/test_typesetter_golden_capture.py pipeline/tests/fixtures/typesetter_v2_golden
git commit -m "test: freeze v1 typesetter golden fixtures"
```

---

### Task 2: Define The V2 Typesetter Contract

**Files:**
- Create: `N:\TraduzAI\pipeline\typesetter_v2\contract.py`
- Create: `N:\TraduzAI\pipeline\typesetter_v2\__init__.py`
- Test: `N:\TraduzAI\pipeline\tests\test_typesetter_v2_contract.py`

**Step 1: Write failing contract tests**

```python
from typesetter_v2.contract import page_to_v2_request


def test_page_to_v2_request_preserves_editor_layer_contract():
    page = {
        "image_path": "images/001.png",
        "text_layers": [
            {
                "id": "txt-1",
                "translated": "OLA",
                "layout_bbox": [10, 20, 110, 80],
                "render_bbox": [12, 22, 108, 78],
                "safe_text_box": [14, 24, 106, 76],
                "style": {"fontFamily": "Arial", "fontSize": 24, "color": "#000000"},
                "visible": True,
            }
        ],
    }

    request = page_to_v2_request(page, page_index=0, image_size=(800, 1200))

    assert request["schema_version"] == 1
    assert request["page_index"] == 0
    assert request["blocks"][0]["id"] == "txt-1"
    assert request["blocks"][0]["text"] == "OLA"
    assert request["blocks"][0]["boxes"]["safe_text_box"] == [14, 24, 106, 76]
```

**Step 2: Run test to verify it fails**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_typesetter_v2_contract.py -v
```

Expected: FAIL because module does not exist.

**Step 3: Implement request schema**

V2 request must include:

```json
{
  "schema_version": 1,
  "page_index": 0,
  "image": {
    "base_path": "...",
    "width": 800,
    "height": 1200
  },
  "masks": {
    "bubble_mask_path": null,
    "brush_mask_path": null
  },
  "defaults": {
    "font_family": "Comic Neue",
    "target_language": "pt-BR"
  },
  "blocks": [
    {
      "id": "txt-1",
      "text": "OLA",
      "source_text": "HELLO",
      "content_class": "dialogue",
      "boxes": {
        "layout_bbox": [10, 20, 110, 80],
        "render_bbox": [12, 22, 108, 78],
        "safe_text_box": [14, 24, 106, 76],
        "balloon_bbox": [8, 18, 112, 82]
      },
      "style": {
        "font_family": "Arial",
        "font_size": 24,
        "color": "#000000",
        "stroke_color": null,
        "stroke_width": 0,
        "bold": false,
        "italic": false,
        "align": "center",
        "rotation_deg": 0
      },
      "policy": {
        "lock_layout_box": false,
        "preserve_manual_style": true,
        "source_direction": null,
        "rendered_direction": null
      }
    }
  ]
}
```

**Step 4: Implement response schema parser**

V2 response must include:

```json
{
  "schema_version": 1,
  "page_index": 0,
  "rendered_page_path": "...",
  "blocks": [
    {
      "id": "txt-1",
      "sprite_path": "...",
      "render_bbox": [12, 22, 108, 78],
      "safe_text_box": [14, 24, 106, 76],
      "font_size": 23,
      "rendered_direction": "horizontal",
      "qa_metrics": {},
      "qa_flags": [],
      "debug": {}
    }
  ]
}
```

**Step 5: Commit**

```powershell
git add pipeline/typesetter_v2 pipeline/tests/test_typesetter_v2_contract.py
git commit -m "feat: define typesetter v2 contract"
```

---

### Task 3: Create Rust `traduzai-typesetter-v2` Crate

**Files:**
- Modify: `N:\TraduzAI\src-tauri\Cargo.toml`
- Create: `N:\TraduzAI\src-tauri\traduzai-typesetter-v2\Cargo.toml`
- Create: `N:\TraduzAI\src-tauri\traduzai-typesetter-v2\src\lib.rs`
- Create: `N:\TraduzAI\src-tauri\traduzai-typesetter-v2\src\main.rs`
- Create: `N:\TraduzAI\src-tauri\traduzai-typesetter-v2\src\contract.rs`
- Test: `N:\TraduzAI\src-tauri\traduzai-typesetter-v2\tests\contract.rs`

**Step 1: Write failing Rust contract test**

```rust
#[test]
fn parses_minimal_request() {
    let json = r##"{
      "schema_version": 1,
      "page_index": 0,
      "image": {"base_path": "base.png", "width": 320, "height": 240},
      "masks": {},
      "defaults": {"font_family": "Arial", "target_language": "pt-BR"},
      "blocks": [{
        "id": "txt-1",
        "text": "OLA",
        "boxes": {"safe_text_box": [10, 20, 120, 80]},
        "style": {"font_family": "Arial", "font_size": 24, "color": "#000000"}
      }]
    }"##;

    let request: traduzai_typesetter_v2::contract::RenderRequest =
        serde_json::from_str(json).unwrap();

    assert_eq!(request.page_index, 0);
    assert_eq!(request.blocks[0].id, "txt-1");
}
```

**Step 2: Run test to verify it fails**

```powershell
cd src-tauri\traduzai-typesetter-v2
cargo test parses_minimal_request
```

Expected: FAIL because crate/API does not exist.

**Step 3: Implement schema structs**

Use `serde`, `anyhow`, and strict validation helpers for bbox and color.

**Step 4: Add CLI**

CLI:

```powershell
traduzai-typesetter-v2.exe --request request.json --output-dir out
```

Output:

```text
out/rendered.png
out/sprites/<block-id>.png
out/response.json
```

Stdout:

```json
{"status":"ok","response_path":"out/response.json"}
```

**Step 5: Commit**

```powershell
git add src-tauri/traduzai-typesetter-v2 src-tauri/Cargo.toml src-tauri/Cargo.lock
git commit -m "feat: add typesetter v2 rust crate"
```

---

### Task 4: Port Koharu Renderer Core

**Files:**
- Create: `N:\TraduzAI\src-tauri\traduzai-typesetter-v2\src\font.rs`
- Create: `N:\TraduzAI\src-tauri\traduzai-typesetter-v2\src\shape.rs`
- Create: `N:\TraduzAI\src-tauri\traduzai-typesetter-v2\src\layout.rs`
- Create: `N:\TraduzAI\src-tauri\traduzai-typesetter-v2\src\raster.rs`
- Create: `N:\TraduzAI\src-tauri\traduzai-typesetter-v2\src\types.rs`
- Test: `N:\TraduzAI\src-tauri\traduzai-typesetter-v2\tests\render_simple.rs`

**Step 1: Use Koharu as reference**

Read and port/adapt only the renderer crate pieces:

```text
N:\TraduzAI\koharu\koharu-renderer\src\font.rs
N:\TraduzAI\koharu\koharu-renderer\src\shape.rs
N:\TraduzAI\koharu\koharu-renderer\src\layout.rs
N:\TraduzAI\koharu\koharu-renderer\src\renderer.rs
N:\TraduzAI\koharu\koharu-renderer\src\types.rs
N:\TraduzAI\koharu\koharu-renderer\src\text\latin.rs
N:\TraduzAI\koharu\koharu-renderer\src\text\script.rs
```

If code is copied, add attribution in:

```text
N:\TraduzAI\src-tauri\traduzai-typesetter-v2\NOTICE.md
```

**Step 2: Add dependencies**

Add:

```toml
fontdb = "0.23"
fontdue = "0.9"
tiny-skia = "0.12"
skrifa = "0.42"
harfrust = "0.6"
icu_properties = "2.2"
icu_segmenter = "2.2"
hypher = "0.1.7"
image = "0.25"
```

**Step 3: Write failing simple render test**

Render `"OLA MUNDO"` into a 320x240 transparent RGBA page and assert:

- image dimensions are correct
- alpha coverage is non-zero
- output PNG can be decoded

**Step 4: Implement simple horizontal text**

Only horizontal text, one block, one system font.

**Step 5: Run test**

```powershell
cd src-tauri\traduzai-typesetter-v2
cargo test render_simple
```

Expected: PASS.

**Step 6: Commit**

```powershell
git add src-tauri/traduzai-typesetter-v2 src-tauri/Cargo.toml src-tauri/Cargo.lock
git commit -m "feat: port koharu renderer core"
```

---

### Task 5: Implement Full Page Composition

**Files:**
- Create: `N:\TraduzAI\src-tauri\traduzai-typesetter-v2\src\page.rs`
- Modify: `N:\TraduzAI\src-tauri\traduzai-typesetter-v2\src\lib.rs`
- Test: `N:\TraduzAI\src-tauri\traduzai-typesetter-v2\tests\page_composition.rs`

**Step 1: Write failing page composition test**

Create a blank base image, render two blocks, assert:

- final output dimensions match base image
- each sprite exists
- final output contains text pixels in expected rough regions
- response includes both block IDs

**Step 2: Implement page renderer**

The page renderer must:

- load base/inpaint image
- render each text block into an RGBA sprite
- position sprite around the resolved layout box center
- composite sprites over base image
- save `rendered.png`
- save one sprite per block
- return response JSON

**Step 3: Run test**

```powershell
cd src-tauri\traduzai-typesetter-v2
cargo test page_composition
```

Expected: PASS.

**Step 4: Commit**

```powershell
git add src-tauri/traduzai-typesetter-v2
git commit -m "feat: render complete page with text sprites"
```

---

### Task 6: Port Font Selection And Local Font Map

**Files:**
- Modify: `N:\TraduzAI\src-tauri\traduzai-typesetter-v2\src\font.rs`
- Modify: `N:\TraduzAI\pipeline\typesetter_v2\contract.py`
- Read: `N:\TraduzAI\fonts\font-map.json`
- Test: `N:\TraduzAI\src-tauri\traduzai-typesetter-v2\tests\font_selection.rs`

**Step 1: Write failing tests**

Cover:

- exact PostScript name
- family name
- project font file under `N:\TraduzAI\fonts`
- missing font fallback
- accented PT-BR glyph coverage

**Step 2: Implement resolver**

Resolution order:

1. explicit style font family/name
2. font-map entry by content class
3. project local font files
4. system fonts via `fontdb`
5. final fallback font

**Step 3: Do not fetch remote fonts**

This migration must not introduce runtime network dependency for fonts.

**Step 4: Run test**

```powershell
cd src-tauri\traduzai-typesetter-v2
cargo test font_selection
```

Expected: PASS.

**Step 5: Commit**

```powershell
git add src-tauri/traduzai-typesetter-v2 pipeline/typesetter_v2/contract.py
git commit -m "feat: implement v2 font selection"
```

---

### Task 7: Port Text Direction, Bidi, And CJK/Latin Policy

**Files:**
- Modify: `N:\TraduzAI\src-tauri\traduzai-typesetter-v2\src\shape.rs`
- Modify: `N:\TraduzAI\src-tauri\traduzai-typesetter-v2\src\layout.rs`
- Create: `N:\TraduzAI\src-tauri\traduzai-typesetter-v2\src\script.rs`
- Test: `N:\TraduzAI\src-tauri\traduzai-typesetter-v2\tests\script_layout.rs`

**Step 1: Write failing tests**

Cover:

- PT-BR horizontal text
- mixed numbers/punctuation
- Arabic/bidi sanity
- CJK vertical hint preserved
- source direction beats bbox aspect when explicit

**Step 2: Port Koharu script policy**

Use Koharu's `text/script.rs` as reference.

**Step 3: Add response field**

Each block response includes:

```json
"rendered_direction": "horizontal" | "vertical"
```

**Step 4: Run test**

```powershell
cd src-tauri\traduzai-typesetter-v2
cargo test script_layout
```

Expected: PASS.

**Step 5: Commit**

```powershell
git add src-tauri/traduzai-typesetter-v2
git commit -m "feat: port v2 script and direction policy"
```

---

### Task 8: Implement Bubble Mask Aware Fitting

**Files:**
- Create: `N:\TraduzAI\src-tauri\traduzai-typesetter-v2\src\bubble.rs`
- Modify: `N:\TraduzAI\src-tauri\traduzai-typesetter-v2\src\page.rs`
- Modify: `N:\TraduzAI\pipeline\typesetter_v2\contract.py`
- Test: `N:\TraduzAI\src-tauri\traduzai-typesetter-v2\tests\bubble_fit.rs`

**Step 1: Write failing tests**

Create synthetic masks and assert:

- text grows from OCR bbox into bubble interior
- sprite does not collide outside bubble ID
- small bubbles choose smaller font
- missing mask falls back to bbox fitting

**Step 2: Port/adapt `BubbleIndex`**

Use Koharu's `text/latin.rs` as reference, but keep TraduzAi fields:

```text
balloon_bbox
safe_text_box
layout_bbox
render_bbox
connected_subregions
```

**Step 3: Add metrics**

Block response must include:

```json
"qa_metrics": {
  "fit": {
    "status": "ok",
    "target_bbox": [..],
    "used_bubble_mask": true,
    "font_size": 23
  }
}
```

**Step 4: Run test**

```powershell
cd src-tauri\traduzai-typesetter-v2
cargo test bubble_fit
```

Expected: PASS.

**Step 5: Commit**

```powershell
git add src-tauri/traduzai-typesetter-v2 pipeline/typesetter_v2/contract.py
git commit -m "feat: add v2 bubble-aware fitting"
```

---

### Task 9: Implement TraduzAi QA Adapter

**Files:**
- Create: `N:\TraduzAI\pipeline\typesetter_v2\qa_adapter.py`
- Test: `N:\TraduzAI\pipeline\tests\test_typesetter_v2_qa_adapter.py`

**Step 1: Write failing QA adapter tests**

```python
from typesetter_v2.qa_adapter import apply_v2_block_response


def test_apply_v2_block_response_sets_existing_project_fields():
    layer = {"id": "txt-1", "qa_flags": []}
    block = {
        "id": "txt-1",
        "render_bbox": [10, 20, 100, 80],
        "safe_text_box": [12, 22, 98, 78],
        "font_size": 22,
        "qa_flags": ["TEXT_OVERFLOW"],
        "qa_metrics": {"fit": {"status": "overflow"}},
        "debug": {"renderer": "v2_koharu"},
    }

    apply_v2_block_response(layer, block)

    assert layer["render_bbox"] == [10, 20, 100, 80]
    assert layer["safe_text_box"] == [12, 22, 98, 78]
    assert "TEXT_OVERFLOW" in layer["qa_flags"]
    assert layer["_render_debug"]["renderer"] == "v2_koharu"
```

**Step 2: Run test to verify it fails**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_typesetter_v2_qa_adapter.py -v
```

Expected: FAIL.

**Step 3: Implement adapter**

Map v2 output to current fields:

- `render_bbox`
- `safe_text_box`
- `font_size`
- `render_preview_path`
- `qa_flags`
- `qa_metrics`
- `_render_debug`
- `rendered_direction`

**Step 4: Run test**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_typesetter_v2_qa_adapter.py -v
```

Expected: PASS.

**Step 5: Commit**

```powershell
git add pipeline/typesetter_v2/qa_adapter.py pipeline/tests/test_typesetter_v2_qa_adapter.py
git commit -m "feat: map v2 renderer output to project QA fields"
```

---

### Task 10: Implement Python V2 Runner

**Files:**
- Create: `N:\TraduzAI\pipeline\typesetter_v2\runner.py`
- Modify: `N:\TraduzAI\pipeline\main.py`
- Test: `N:\TraduzAI\pipeline\tests\test_typesetter_v2_runner.py`
- Test: `N:\TraduzAI\pipeline\tests\test_main_emit.py`

**Step 1: Write failing runner tests**

Mock the Rust binary and assert:

- request JSON is written
- binary is called
- response JSON is parsed
- output is copied/moved to the expected rendered path
- QA adapter is applied to text layers

**Step 2: Implement runner**

Runner responsibilities:

- locate v2 executable
- create temp work dir
- serialize request
- pass base image/mask paths
- call Rust synchronously
- parse response
- update page/layers
- return rendered path

**Step 3: Wire `TRADUZAI_TYPESETTER`**

In `pipeline/main.py`:

```python
if os.environ.get("TRADUZAI_TYPESETTER") == "v2_koharu":
    run_typesetter_v2(...)
else:
    run current v1 path
```

Apply to:

- automatic final typeset
- `render_page_image`
- `_run_render_preview_page`
- editor retypeset path

**Step 4: Run focused tests**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_typesetter_v2_runner.py pipeline\tests\test_main_emit.py -k "render" -x
```

Expected: PASS.

**Step 5: Commit**

```powershell
git add pipeline/typesetter_v2 pipeline/main.py pipeline/tests/test_typesetter_v2_runner.py pipeline/tests/test_main_emit.py
git commit -m "feat: wire typesetter v2 runner behind flag"
```

---

### Task 11: Tauri/App Integration

**Files:**
- Modify: `N:\TraduzAI\src-tauri\src\commands\pipeline.rs`
- Modify: `N:\TraduzAI\src\lib\tauri.ts`
- Modify: `N:\TraduzAI\src\lib\stores\appStore.ts`
- Modify: `N:\TraduzAI\src\components\editor\toolbar\RenderStatusBadge.tsx`
- Test: `N:\TraduzAI\src-tauri\src\commands\pipeline.rs`
- Test: relevant frontend tests

**Step 1: Add setting**

Add internal setting:

```text
typesetterBackend: "v1_python" | "v2_koharu"
```

Do not expose it prominently in product UI yet. Debug/settings only.

**Step 2: Pass backend to sidecar environment**

Tauri commands must set:

```text
TRADUZAI_TYPESETTER=v2_koharu
```

when requested.

**Step 3: Preserve preview contract**

`render_preview_page` still returns the same output path and event shape.

**Step 4: Run tests**

```powershell
cargo test --manifest-path src-tauri/Cargo.toml
npm run test -- --run
```

Expected: PASS.

**Step 5: Commit**

```powershell
git add src-tauri/src/commands/pipeline.rs src/lib/tauri.ts src/lib/stores/appStore.ts src/components/editor/toolbar/RenderStatusBadge.tsx
git commit -m "feat: expose selectable typesetter backend"
```

---

### Task 12: V1 vs V2 Parity Harness

**Files:**
- Create: `N:\TraduzAI\pipeline\scripts\compare_typesetter_v1_v2.py`
- Test: `N:\TraduzAI\pipeline\tests\test_compare_typesetter_v1_v2.py`

**Step 1: Write failing script test**

Test that a fixture run creates:

```text
v1.png
v2.png
diff.png
metadata_diff.json
contact_sheet.png
report.json
```

**Step 2: Implement comparison**

Metrics:

- pixel diff
- alpha coverage
- OCR/render bbox difference
- safe box difference
- font size difference
- QA flag delta
- render time

**Step 3: Run against fixtures**

```powershell
pipeline\venv\Scripts\python.exe pipeline\scripts\compare_typesetter_v1_v2.py --fixture-dir pipeline\tests\fixtures\typesetter_v2_golden --out debug\typesetter-v2-parity
```

Expected: report and contact sheet exist.

**Step 4: Commit**

```powershell
git add pipeline/scripts/compare_typesetter_v1_v2.py pipeline/tests/test_compare_typesetter_v1_v2.py
git commit -m "test: add v1 v2 typesetter parity harness"
```

---

### Task 13: Real Chapter Validation Gate

**Files:**
- Create: `N:\TraduzAI\docs\plans\typesetter-v2-validation-report.md`

**Step 1: Select real chapters**

Use at least:

- one manga chapter with connected balloons
- one manhwa/strip chapter
- one chapter with signs/narration
- one editor manual-flow project

**Step 2: Run V1**

```powershell
$env:TRADUZAI_TYPESETTER='v1_python'
pipeline\venv\Scripts\python.exe pipeline\main.py config.json
Remove-Item Env:\TRADUZAI_TYPESETTER
```

**Step 3: Run V2**

```powershell
$env:TRADUZAI_TYPESETTER='v2_koharu'
pipeline\venv\Scripts\python.exe pipeline\main.py config.json
Remove-Item Env:\TRADUZAI_TYPESETTER
```

**Step 4: Inspect artifacts**

Inspect:

```text
project.json
qa_report.json
debug/e2e/09_typeset/render_plan_final.jsonl
rendered images
contact sheets
editor preview output
```

**Step 5: Pass criteria**

V2 cannot become default unless:

- no missing `render_bbox`
- no missing `safe_text_box`
- no increase in hard QA blockers
- no editor preview/retypeset contract regression
- no obvious unreadable text in contact sheets
- connected balloons remain semantically coherent
- render time is not worse by more than 20%

**Step 6: Commit report**

```powershell
git add docs/plans/typesetter-v2-validation-report.md
git commit -m "docs: record typesetter v2 validation evidence"
```

---

### Task 14: Flip Default To V2

**Files:**
- Modify: `N:\TraduzAI\pipeline\main.py`
- Modify: `N:\TraduzAI\src-tauri\src\commands\pipeline.rs`
- Modify: `N:\TraduzAI\src\lib\stores\appStore.ts`
- Modify: docs/status files as needed

**Step 1: Require completed validation**

Do not start this task unless Task 13 passed and the validation report says V2 is acceptable.

**Step 2: Change default**

Default:

```text
TRADUZAI_TYPESETTER=v2_koharu
```

Fallback:

```text
TRADUZAI_TYPESETTER=v1_python
```

**Step 3: Keep emergency fallback visible in logs**

On every render, record:

```json
"typesetter_backend": "v2_koharu"
```

**Step 4: Run full verification**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_typesetting_layout.py pipeline\tests\test_typesetting_renderer.py pipeline\tests\test_layout_analysis.py pipeline\tests\test_main_emit.py pipeline\tests\test_typesetter_v2_contract.py pipeline\tests\test_typesetter_v2_runner.py -x
cargo test --manifest-path src-tauri/Cargo.toml
npm run test -- --run
```

Expected: PASS.

**Step 5: Commit**

```powershell
git add pipeline/main.py src-tauri/src/commands/pipeline.rs src/lib/stores/appStore.ts docs
git commit -m "feat: make typesetter v2 default"
```

---

### Task 15: Retire V1 Python Typesetter

**Files:**
- Modify: `N:\TraduzAI\pipeline\typesetter\renderer.py`
- Modify: `N:\TraduzAI\pipeline\typesetter_v2\runner.py`
- Modify: tests and docs

**Step 1: Wait for stability window**

Do not remove V1 immediately after flipping default. Wait until:

- at least one real workflow cycle passes
- no frequent fallback or crash logs
- editor preview/retypeset is stable
- user explicitly approves removal

**Step 2: Move V1 to legacy module**

Rename or isolate:

```text
pipeline/typesetter_legacy/
```

Keep command-line fallback for one release:

```text
TRADUZAI_TYPESETTER=v1_python_legacy
```

**Step 3: Delete only dead code**

Do not delete shared QA/reporting utilities still used by V2.

**Step 4: Run full verification**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests -x
cargo test --manifest-path src-tauri/Cargo.toml
npm run test -- --run
```

Expected: PASS.

**Step 5: Commit**

```powershell
git add pipeline src-tauri src docs
git commit -m "refactor: retire legacy python typesetter"
```

---

## Recommended Execution Slice

For the first implementation pass, do only Tasks 1-5:

1. Freeze V1 fixtures.
2. Define V2 contract.
3. Create Rust crate and CLI.
4. Port simple Koharu renderer core.
5. Render a full page with sprites.

Stop there for visual review before touching editor defaults or removing Python behavior.

## Validation Commands

Run often:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_typesetter_v2_contract.py pipeline\tests\test_typesetter_v2_runner.py -x
pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_typesetting_layout.py pipeline\tests\test_typesetting_renderer.py pipeline\tests\test_layout_analysis.py -x
cargo test --manifest-path src-tauri/Cargo.toml
npm run test -- --run
```

Visual comparison:

```powershell
pipeline\venv\Scripts\python.exe pipeline\scripts\compare_typesetter_v1_v2.py --fixture-dir pipeline\tests\fixtures\typesetter_v2_golden --out debug\typesetter-v2-parity
```

## Rollback Plan

- Keep `TRADUZAI_TYPESETTER=v1_python` available until after Task 15.
- If V2 fails on preview only, disable it for `--render-preview-page` first.
- If V2 fails on connected balloons, keep v2 raster core but route connected balloon cases to V1 until fixed.
- If V2 corrupts `project.json`, block default flip and fix the QA adapter.
- Never delete V1 until the user explicitly approves after real chapter evidence.

## Open Decisions

- Copy Koharu code directly with attribution vs reimplement equivalent modules.
- Make `traduzai-typesetter-v2` a Tauri-linked Rust library vs sidecar executable. The plan starts as sidecar because it is easier to test and roll back.
- Whether v2 should own all connected-balloon splitting immediately or consume current Python connected-subregion metadata first.
- Whether `render_plan_final.jsonl` remains Python-authored or becomes Rust-authored.
- Whether font-map behavior should be ported as JSON config loaded by Rust or pre-resolved by Python.

## Execution Handoff

Plan complete and saved to `docs/plans/2026-05-27-koharu-full-typesetter-migration.md`.

Recommended approach: execute Tasks 1-5 in a dedicated branch/worktree, produce visual artifacts, then decide whether to continue with full QA/editor/default migration.

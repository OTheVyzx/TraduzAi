# Bubble Mask Inpaint Cleaner V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` for task execution in this session, or `superpowers:executing-plans` in a separate session. Follow tasks in order. Do not touch the main checkout except this plan file.

**Goal:** Add a controlled experimental mask/inpaint path that rebuilds inpaint masks from real BubbleMask-constrained glyph components, runs inpaint through manga-cleaner-style ROI pasteback, and compares AOT against the existing LaMa ONNX path without regressing SFX/art or UI/layout text.

**Architecture:** Detection/OCR still owns text candidates. Mask construction becomes an explicit opt-in engine: it requires a real BubbleMask or a validated derived white-balloon mask and never uses `balloon_bbox` as the source mask. Inpaint runs on a source crop padded to model-safe dimensions, then pastes back only masked pixels. Defaults remain unchanged until visual gates approve promotion.

**Tech Stack:** Python 3.12, OpenCV, NumPy, existing `pipeline/inpainter`, `pipeline/vision_stack`, `pipeline/strip`, `pipeline/tests`, and existing `pipeline/inpainter/lama_onnx.py`.

---

## Non-Negotiable Constraints

- Work in a dedicated worktree, not the dirty main checkout.
- Do not import or vendor code from external repositories. Reimplement only the ideas.
- Do not persist NumPy masks to `project.json`.
- Do not let `balloon_bbox`, `tipo`, `content_class`, `balloon_type`, `skip_processing`, or `preserve_original` decide this experimental mask path.
- Do not promote this path as default until visual regression passes on at least One Second chapter 1 and one second English chapter from the user's manga folder.
- Do not version large `DEBUGM/runs` outputs.

---

## Baseline And Worktree

Run from the main checkout:

```powershell
cd N:\TraduzAI
git status --short
git worktree add .worktrees\bubble-mask-cleaner-v2 -b codex/bubble-mask-cleaner-v2 HEAD
cd N:\TraduzAI\.worktrees\bubble-mask-cleaner-v2
git status --short
```

Expected:

- Main checkout may be dirty.
- New worktree starts from `HEAD`.
- All edits below happen under `N:\TraduzAI\.worktrees\bubble-mask-cleaner-v2`.

Reference-only note:

- The older experiment at `N:\TraduzAI\.worktrees\bubble-primary-test` may be inspected for ideas, but do not cherry-pick blindly. Re-apply through the tests in this plan.

---

## Current Real Contracts

These are the contracts the implementation must match:

- `pipeline/inpainter/mask_builder.py`
  - `build_inpaint_mask(block: dict, image_shape: tuple[int, ...], image_rgb: np.ndarray | None = None) -> np.ndarray | None`
  - mask evidence is written to `block["mask_evidence"]` by `consolidate_mask_evidence(...)`.
  - There is no `MaskEvidence` return object.
- `pipeline/vision_stack/runtime.py`
  - `_call_inpainter_in_roi(inpainter, image_np, mask, roi_bbox, use_roi, batch_size=4, debug=None, force_no_tiling=False) -> np.ndarray`
  - tests must pass `roi_bbox` and `use_roi`.
- `pipeline/strip/types.py`
  - `Balloon` currently has no `mask` field; adding mask propagation must modify this file.
- `pipeline/inpainter/lama_onnx.py`
  - LaMa ONNX already exists. The task is to expose a clean provider/route contract, not to add a brand-new backend.

---

## Canonical Experimental Flags

Use these names everywhere:

```text
TRADUZAI_TEXT_MASK_ENGINE=component_bubble_cleaner
TRADUZAI_INPAINT_PRIMARY_ENGINE=aot_manga_roi | lama_onnx
```

Compatibility aliases allowed only inside selection helpers:

```text
TRADUZAI_BUBBLE_PRIMARY_ENGINE=notanother
TRADUZAI_INPAINT_PRIMARY_ENGINE=manga_cleaner_roi_lama
```

All evidence emitted by the new text mask path must use canonical kind:

```text
component_bubble_cleaner
```

Do not emit new evidence as `primary_bubble_cleaner`.

---

## Task 1: Add Opt-In Contract Tests And Selection Helpers

**Files:**

- Modify: `pipeline/inpainter/mask_builder.py`
- Modify: `pipeline/vision_stack/runtime.py`
- Modify: `pipeline/tests/test_mask_builder.py`
- Modify: `pipeline/tests/test_vision_stack_runtime.py`

### Step 1: Write failing tests for real signatures

Add tests that call the actual APIs:

```python
def test_component_bubble_cleaner_requires_real_bubble_mask(monkeypatch):
    monkeypatch.setenv("TRADUZAI_TEXT_MASK_ENGINE", "component_bubble_cleaner")
    image = np.full((80, 120, 3), 255, dtype=np.uint8)
    block = {
        "bbox": [20, 20, 60, 40],
        "text": "HELLO",
        "balloon_bbox": [10, 10, 90, 50],
    }

    mask = build_inpaint_mask(block, image.shape, image)

    assert mask is None
    evidence = block["mask_evidence"]
    assert evidence["kind"] == "none"
    assert "component_bubble_cleaner_missing_bubble_mask" in evidence["fast_fill_reject_reasons"]
```

```python
def test_legacy_notanother_flag_maps_to_component_bubble_cleaner(monkeypatch):
    monkeypatch.delenv("TRADUZAI_TEXT_MASK_ENGINE", raising=False)
    monkeypatch.setenv("TRADUZAI_BUBBLE_PRIMARY_ENGINE", "notanother")

    assert _selected_text_mask_engine() == "component_bubble_cleaner"
```

```python
def test_selected_inpaint_engine_reads_primary_env(monkeypatch):
    monkeypatch.setenv("TRADUZAI_INPAINT_PRIMARY_ENGINE", "lama_onnx")

    assert _selected_inpaint_engine() == "lama_onnx"
```

### Step 2: Run and confirm failure

```powershell
cd N:\TraduzAI\.worktrees\bubble-mask-cleaner-v2\pipeline
python -m pytest tests\test_mask_builder.py::test_component_bubble_cleaner_requires_real_bubble_mask tests\test_mask_builder.py::test_legacy_notanother_flag_maps_to_component_bubble_cleaner tests\test_vision_stack_runtime.py::test_selected_inpaint_engine_reads_primary_env -q
```

Expected: fail because helpers/mode are not implemented yet.

### Step 3: Implement minimal helpers

In `mask_builder.py`:

```python
TEXT_MASK_ENGINE_ENV = "TRADUZAI_TEXT_MASK_ENGINE"
LEGACY_BUBBLE_PRIMARY_ENGINE_ENV = "TRADUZAI_BUBBLE_PRIMARY_ENGINE"
COMPONENT_BUBBLE_CLEANER_MODES = {"component_bubble_cleaner", "notanother", "notanotherbubblecleaner", "notanother_bubble_cleaner"}


def _selected_text_mask_engine() -> str:
    value = os.getenv(TEXT_MASK_ENGINE_ENV, "").strip().lower()
    if value in COMPONENT_BUBBLE_CLEANER_MODES:
        return "component_bubble_cleaner"
    legacy = os.getenv(LEGACY_BUBBLE_PRIMARY_ENGINE_ENV, "").strip().lower()
    if legacy in COMPONENT_BUBBLE_CLEANER_MODES:
        return "component_bubble_cleaner"
    return value
```

In `runtime.py`:

```python
INPAINT_PRIMARY_ENGINE_ENV = "TRADUZAI_INPAINT_PRIMARY_ENGINE"


def _selected_inpaint_engine() -> str:
    return os.getenv(INPAINT_PRIMARY_ENGINE_ENV, "").strip().lower()
```

### Step 4: Run focused tests

```powershell
python -m pytest tests\test_mask_builder.py tests\test_vision_stack_runtime.py -q -k "component_bubble_cleaner or selected_inpaint_engine"
```

Expected: contract tests pass.

### Step 5: Commit in the worktree

```powershell
git add pipeline\inpainter\mask_builder.py pipeline\vision_stack\runtime.py pipeline\tests\test_mask_builder.py pipeline\tests\test_vision_stack_runtime.py
git commit -m "test: add component bubble cleaner contracts"
```

---

## Task 2: Preserve Real BubbleMask Through Strip Flow

**Files:**

- Modify: `pipeline/strip/types.py`
- Modify: `pipeline/strip/detect_balloons.py`
- Modify: `pipeline/strip/process_bands.py`
- Modify: `pipeline/project_writer.py`
- Modify: `pipeline/tests/test_strip_process_bands.py`
- Modify: `pipeline/tests/test_project_writer.py`

### Step 1: Write failing tests for mask propagation

Tests must prove three things:

- detector block masks survive into `Balloon.mask`;
- band/page conversion attaches `bubble_mask` to `_vision_blocks`;
- runtime raster masks are stripped before JSON persistence.

Use tests shaped like:

```python
def test_detect_strip_balloons_preserves_detector_mask(monkeypatch):
    mask = np.zeros((40, 60), dtype=np.uint8)
    mask[8:28, 12:42] = 255

    class FakeDetector:
        def detect(self, image):
            return [{"xyxy": [10, 10, 70, 50], "confidence": 0.9, "mask": mask}]

    balloons = detect_strip_balloons(np.full((100, 120, 3), 255, dtype=np.uint8), FakeDetector())

    assert balloons
    assert isinstance(balloons[0].mask, np.ndarray)
    assert int(np.count_nonzero(balloons[0].mask)) > 0
```

```python
def test_project_writer_strips_runtime_raster_masks():
    project = {
        "paginas": [{
            "text_layers": [{"text": "A", "bubble_mask": np.ones((4, 4), dtype=np.uint8)}],
            "_vision_blocks": [{"text": "A", "bubble_mask": np.ones((4, 4), dtype=np.uint8)}],
            "_bubble_regions": [{"bubble_mask": np.ones((4, 4), dtype=np.uint8)}],
        }]
    }

    cleaned = neutralize_project_compatibility_metadata(project)

    page = cleaned["paginas"][0]
    assert "bubble_mask" not in page["text_layers"][0]
    assert "bubble_mask" not in page["_vision_blocks"][0]
    assert "bubble_mask" not in page["_bubble_regions"][0]
```

### Step 2: Run and confirm failure

```powershell
python -m pytest tests\test_strip_process_bands.py tests\test_project_writer.py -q -k "bubble_mask or detector_mask"
```

Expected: fail because `Balloon.mask` and JSON stripping are incomplete.

### Step 3: Implement propagation

- Add `mask: Optional[np.ndarray] = None` to `Balloon`.
- In `detect_balloons.py`, add a helper that accepts detector masks under `mask`, `bubble_mask`, `balloon_mask`, or `segmentation_mask`.
- Store mask as `uint8` binary with non-zero pixels equal to 255.
- In `process_bands.py`, when matching text to a balloon, attach:

```python
block["bubble_mask"] = balloon.mask
block["bubble_mask_bbox"] = list(balloon.strip_bbox)
```

- If detector mask is absent, derive a visual white-balloon mask only when the region is visibly white/light and connected. Mark it with:

```python
block["bubble_mask_source"] = "derived_white_balloon"
```

- Do not derive masks on dark art/SFX panels.

### Step 4: Strip raster masks from persisted JSON

In `project_writer.py`, remove these keys from `text_layers`, `textos`, `texts`, `_vision_blocks`, `_bubble_regions`, `bubble_regions`, and nested `metadata` dicts:

```python
RUNTIME_RASTER_KEYS = {
    "bubble_mask",
    "bubbleMask",
    "balloon_mask",
    "balloonMask",
    "segmentation_mask",
    "mask",
}
```

### Step 5: Run focused tests

```powershell
python -m pytest tests\test_strip_process_bands.py tests\test_project_writer.py -q -k "bubble_mask or detector_mask"
```

Expected: pass.

### Step 6: Commit

```powershell
git add pipeline\strip\types.py pipeline\strip\detect_balloons.py pipeline\strip\process_bands.py pipeline\project_writer.py pipeline\tests\test_strip_process_bands.py pipeline\tests\test_project_writer.py
git commit -m "feat: preserve runtime bubble masks without serializing them"
```

---

## Task 3: Implement Component Bubble Cleaner Mask

**Files:**

- Modify: `pipeline/inpainter/mask_builder.py`
- Modify: `pipeline/tests/test_mask_builder.py`

### Step 1: Write failing tests for real BubbleMask extraction

Cover:

- full-page binary BubbleMask;
- crop BubbleMask with `bubble_mask_bbox`;
- numeric label mask with numeric `bubble_id`;
- ambiguous full-page multi-component mask with string `bubble_id` rejected;
- no fallback to `balloon_bbox`.

Required test shape:

```python
def test_component_bubble_cleaner_keeps_only_glyph_components_inside_real_bubble(monkeypatch):
    monkeypatch.setenv("TRADUZAI_TEXT_MASK_ENGINE", "component_bubble_cleaner")
    image = np.full((120, 180, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (55, 45), (70, 55), (0, 0, 0), -1)
    cv2.rectangle(image, (130, 90), (150, 105), (0, 0, 0), -1)
    bubble_mask = np.zeros((120, 180), dtype=np.uint8)
    cv2.ellipse(bubble_mask, (70, 55), (45, 25), 0, 0, 360, 255, -1)
    block = {
        "bbox": [45, 35, 105, 75],
        "text": "HELLO",
        "bubble_mask": bubble_mask,
    }

    mask = build_inpaint_mask(block, image.shape, image)

    assert isinstance(mask, np.ndarray)
    assert mask[50, 60] > 0
    assert mask[98, 140] == 0
    assert block["mask_evidence"]["kind"] == "component_bubble_cleaner"
    assert block["mask_evidence"]["fast_fill_allowed"] is True
```

### Step 2: Write failing tests for border preservation

```python
def test_component_bubble_cleaner_does_not_touch_bubble_outline(monkeypatch):
    monkeypatch.setenv("TRADUZAI_TEXT_MASK_ENGINE", "component_bubble_cleaner")
    image = np.full((90, 140, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (35, 42), (100, 45), (0, 0, 0), -1)
    bubble_mask = np.zeros((90, 140), dtype=np.uint8)
    cv2.ellipse(bubble_mask, (70, 45), (55, 30), 0, 0, 360, 255, -1)
    block = {"bbox": [25, 28, 115, 62], "text": "LONG", "bubble_mask": bubble_mask}

    mask = build_inpaint_mask(block, image.shape, image)

    outline = cv2.Canny(bubble_mask, 50, 150)
    assert int(np.count_nonzero((mask > 0) & (outline > 0))) == 0
```

### Step 3: Implement helpers

Add helpers in `mask_builder.py`:

- `_real_bubble_mask_from_block(block, image_shape) -> np.ndarray | None`
  - Accept full-page mask.
  - Accept crop mask only when `bubble_mask_bbox` or compatible bbox is present.
  - Clip label masks by numeric `bubble_id`.
  - Reject ambiguous multi-component masks for string/non-numeric `bubble_id`.
  - Return binary full-page `uint8` mask.
- `_component_bubble_support_mask(block, image_shape) -> np.ndarray`
  - Use `line_polygons`, `text_pixel_bbox`, then `bbox`, in that priority.
  - Expand support slightly, but do not include the whole balloon.
- `_threshold_glyphs_inside_bubble(image_rgb, bubble_mask, support_mask) -> np.ndarray`
  - Use dark text threshold from pixels inside safe bubble.
  - Include colored glyphs only if they contrast with local white/light bubble fill.
  - Do not include SFX/art outside support mask.
- `_component_mask_inside_bubble(glyphs, safe_bubble, support_mask) -> tuple[np.ndarray, dict]`
  - Connected components.
  - Reject small components.
  - Reject components whose centroid is outside safe bubble.
  - Reject components with low overlap against safe bubble or support mask.
  - Morph close only with tiny kernel.
  - Clip to eroded safe bubble to preserve outline.

Wire `build_inpaint_mask(...)`:

```python
if _selected_text_mask_engine() == "component_bubble_cleaner":
    return _build_component_bubble_cleaner_mask(block, image_shape, image_rgb)
```

Failure evidence:

```text
component_bubble_cleaner_missing_image
component_bubble_cleaner_image_shape_mismatch
component_bubble_cleaner_missing_bubble_mask
component_bubble_cleaner_no_components
```

### Step 4: Run focused tests

```powershell
python -m pytest tests\test_mask_builder.py -q -k "component_bubble_cleaner"
```

Expected: pass.

### Step 5: Commit

```powershell
git add pipeline\inpainter\mask_builder.py pipeline\tests\test_mask_builder.py
git commit -m "feat: add bubble-constrained component text masks"
```

---

## Task 4: Add Component Debug Evidence

**Files:**

- Modify: `pipeline/inpainter/mask_builder.py`
- Modify: `pipeline/vision_stack/runtime.py`
- Modify: `pipeline/tests/test_mask_builder.py`

### Step 1: Write failing debug metadata test

```python
def test_component_bubble_cleaner_records_component_debug(monkeypatch):
    monkeypatch.setenv("TRADUZAI_TEXT_MASK_ENGINE", "component_bubble_cleaner")
    image = np.full((80, 120, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (30, 30), (45, 40), (0, 0, 0), -1)
    bubble_mask = np.zeros((80, 120), dtype=np.uint8)
    cv2.rectangle(bubble_mask, (20, 20), (70, 55), 255, -1)
    block = {"bbox": [20, 20, 70, 55], "text": "OK", "bubble_mask": bubble_mask}

    mask = build_inpaint_mask(block, image.shape, image)

    assert isinstance(mask, np.ndarray)
    debug = block["mask_evidence"]["debug"]
    assert debug["component_total"] >= 1
    assert debug["component_accepted"] >= 1
    assert "component_rejected_small" in debug
    assert "component_rejected_outside_bubble" in debug
    assert "component_rejected_low_overlap" in debug
```

### Step 2: Implement debug counters

Store counters under:

```python
block["mask_evidence"]["debug"]
```

Allowed debug keys:

```text
component_total
component_accepted
component_rejected_small
component_rejected_outside_bubble
component_rejected_low_overlap
safe_bubble_pixels
support_mask_pixels
raw_glyph_pixels
final_mask_pixels
```

If `DebugRunRecorder` is available in runtime, emit optional images:

```text
component_bubble_cleaner_threshold.png
component_bubble_cleaner_bubble_mask.png
component_bubble_cleaner_support.png
component_bubble_cleaner_accepted.png
```

Do not require these images in unit tests.

### Step 3: Run tests

```powershell
python -m pytest tests\test_mask_builder.py -q -k "component_bubble_cleaner"
```

Expected: pass.

### Step 4: Commit

```powershell
git add pipeline\inpainter\mask_builder.py pipeline\vision_stack\runtime.py pipeline\tests\test_mask_builder.py
git commit -m "feat: record component mask debug evidence"
```

---

## Task 5: Move Manga-Cleaner-Style ROI Geometry Into Region Strategy

**Files:**

- Modify: `pipeline/inpainter/region_strategy.py`
- Modify: `pipeline/tests/test_inpaint_region_strategy.py`

### Step 1: Write failing ROI tests

```python
def test_manga_cleaner_roi_from_mask_tracks_source_crop_and_padded_size():
    mask = np.zeros((101, 157), dtype=np.uint8)
    mask[25:64, 41:83] = 255

    roi = manga_cleaner_roi_from_mask(mask, padding=12, multiple=8)

    assert roi.x1 <= 41
    assert roi.y1 <= 25
    assert roi.x2 > 83
    assert roi.y2 > 64
    assert roi.source_width == roi.x2 - roi.x1
    assert roi.source_height == roi.y2 - roi.y1
    assert roi.padded_width % 8 == 0
    assert roi.padded_height % 8 == 0
```

```python
def test_pasteback_masked_pixels_changes_only_masked_crop_pixels():
    base = np.zeros((64, 64, 3), dtype=np.uint8)
    crop_output = np.full((20, 20, 3), 200, dtype=np.uint8)
    crop_mask = np.zeros((20, 20), dtype=np.uint8)
    crop_mask[5:10, 5:10] = 255

    result = pasteback_masked_pixels(base, crop_output, crop_mask, [10, 10, 30, 30])

    assert np.all(result[16, 16] == 200)
    assert np.all(result[12, 12] == 0)
    assert np.all(result[0, 0] == 0)
```

### Step 2: Implement region helpers

In `region_strategy.py`, add:

```python
@dataclass(frozen=True)
class MangaCleanerROI:
    x1: int
    y1: int
    x2: int
    y2: int
    source_width: int
    source_height: int
    padded_width: int
    padded_height: int
```

Add functions:

```python
def manga_cleaner_roi_from_mask(mask: np.ndarray, padding: int = 16, multiple: int = 8) -> MangaCleanerROI
def reflect_pad_crop_to_multiple(crop_image: np.ndarray, crop_mask: np.ndarray, multiple: int = 8) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]
def pasteback_masked_pixels(base: np.ndarray, crop_output: np.ndarray, crop_mask: np.ndarray, roi_bbox: list[int]) -> np.ndarray
```

Rules:

- ROI source crop must stay inside the source image.
- Padded size may exceed source crop and is handled by reflection padding.
- Pasteback must change only `crop_mask > 0` pixels.

### Step 3: Run tests

```powershell
python -m pytest tests\test_inpaint_region_strategy.py -q -k "manga_cleaner or pasteback"
```

Expected: pass.

### Step 4: Commit

```powershell
git add pipeline\inpainter\region_strategy.py pipeline\tests\test_inpaint_region_strategy.py
git commit -m "feat: add manga-cleaner style inpaint roi geometry"
```

---

## Task 6: Route ROI Inpaint Through Region Strategy Without Changing Defaults

**Files:**

- Modify: `pipeline/vision_stack/runtime.py`
- Modify: `pipeline/tests/test_vision_stack_runtime.py`

### Step 1: Write failing runtime tests

Use the real `_call_inpainter_in_roi` signature:

```python
def test_aot_manga_roi_snaps_reflect_pads_and_pastes_mask_only(monkeypatch):
    monkeypatch.setenv("TRADUZAI_INPAINT_PRIMARY_ENGINE", "aot_manga_roi")
    image = np.zeros((65, 67, 3), dtype=np.uint8)
    mask = np.zeros((65, 67), dtype=np.uint8)
    mask[20:30, 20:30] = 255
    calls = []

    class FakeInpainter:
        def inpaint(self, crop, crop_mask, *args, **kwargs):
            calls.append((crop.shape, crop_mask.shape))
            out = crop.copy()
            out[crop_mask > 0] = 200
            return out

    result = _call_inpainter_in_roi(FakeInpainter(), image, mask, [15, 15, 35, 35], True)

    assert calls
    assert calls[0][0][0] % 8 == 0
    assert calls[0][0][1] % 8 == 0
    assert np.all(result[25, 25] == 200)
    assert np.all(result[0, 0] == 0)
```

### Step 2: Implement route

In `_call_inpainter_in_roi(...)`:

- If `use_roi` is false, keep current behavior.
- If selected engine is `aot_manga_roi`, use region strategy ROI helpers with the existing `inpainter`.
- Keep current non-experimental ROI path unchanged when env is unset.
- Do not alpha-blend outside mask in experimental mode.

### Step 3: Run tests

```powershell
python -m pytest tests\test_vision_stack_runtime.py -q -k "manga_roi or inpaint_roi"
```

Expected: pass.

### Step 4: Commit

```powershell
git add pipeline\vision_stack\runtime.py pipeline\tests\test_vision_stack_runtime.py
git commit -m "feat: route experimental inpaint through safe roi pasteback"
```

---

## Task 7: Adapt Existing LaMa ONNX Into The Same ROI Contract

**Files:**

- Modify: `pipeline/inpainter/lama_onnx.py`
- Modify: `pipeline/inpainter/__init__.py`
- Modify: `pipeline/vision_stack/runtime.py`
- Modify: `pipeline/tests/test_lama_onnx.py`
- Modify: `pipeline/tests/test_vision_stack_runtime.py`

### Step 1: Write provider-selection tests

```python
def test_select_lama_onnx_providers_prefers_cuda_when_available(monkeypatch):
    fake_ort = types.SimpleNamespace(
        get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    providers = select_lama_onnx_providers("auto")

    assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
```

```python
def test_select_lama_onnx_providers_cpu_mode():
    assert select_lama_onnx_providers("cpu") == ["CPUExecutionProvider"]
```

### Step 2: Implement provider helper on top of existing code

Add to `lama_onnx.py`:

```python
def select_lama_onnx_providers(mode: str = "auto") -> list[str]:
    mode = (mode or "auto").strip().lower()
    if mode == "cpu":
        return ["CPUExecutionProvider"]
    prepare_windows_onnxruntime_gpu_runtime()
    import onnxruntime as ort
    available = set(ort.get_available_providers())
    providers: list[str] = []
    if mode in {"auto", "cuda"} and "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers
```

Do not replace existing `get_lama_session(...)`; have it use this helper only where appropriate.

### Step 3: Runtime route

When `TRADUZAI_INPAINT_PRIMARY_ENGINE=lama_onnx`, route ROI crop through the existing LaMa ONNX session/function and the same region strategy pasteback.

Test with a fake session/function; do not download the model in unit tests.

### Step 4: Run tests

```powershell
python -m pytest tests\test_lama_onnx.py tests\test_vision_stack_runtime.py -q -k "lama_onnx"
```

Expected: pass without network/model download.

### Step 5: Commit

```powershell
git add pipeline\inpainter\lama_onnx.py pipeline\inpainter\__init__.py pipeline\vision_stack\runtime.py pipeline\tests\test_lama_onnx.py pipeline\tests\test_vision_stack_runtime.py
git commit -m "feat: expose lama onnx safe roi mode"
```

---

## Task 8: Protect SFX, Art, UI, And Tiny Text Cases

**Files:**

- Modify: `pipeline/inpainter/mask_builder.py`
- Modify: `pipeline/qa/export_gate.py` only if existing QA needs one new flag
- Modify: `pipeline/tests/test_mask_builder.py`
- Modify: `pipeline/tests/test_export_gate.py` only if QA flag is added

### Step 1: Add focused tests

Tests must cover:

- Korean/CJK SFX outside BubbleMask is not masked.
- UI/form text inside rectangular panels can be masked only when it has UI/layout evidence or a real mask, never by page-wide OCR alone.
- Component mask does not create `mask_density_high` as a blocking reason.
- Component mask does not mark `mask_outside_balloon` when using validated `bubble_mask_source="derived_white_balloon"`.

### Step 2: Implement only minimal gates

Rules:

- If no real/derived BubbleMask exists, return no mask with explicit reject reason.
- If the candidate support area is much larger than glyph evidence, reject instead of filling background.
- If OCR text language/script does not match work source language and is outside BubbleMask, preserve it as SFX/art.
- Confidence is metadata only; do not drop text solely due to low confidence.

### Step 3: Run tests

```powershell
python -m pytest tests\test_mask_builder.py tests\test_export_gate.py -q -k "component_bubble_cleaner or sfx or mask_density"
```

Expected: pass.

### Step 4: Commit

```powershell
git add pipeline\inpainter\mask_builder.py pipeline\qa\export_gate.py pipeline\tests\test_mask_builder.py pipeline\tests\test_export_gate.py
git commit -m "fix: guard component masks against sfx and art damage"
```

---

## Task 9: Focused Unit Regression

Run:

```powershell
cd N:\TraduzAI\.worktrees\bubble-mask-cleaner-v2\pipeline
python -m pytest tests\test_mask_builder.py tests\test_inpaint_region_strategy.py tests\test_lama_onnx.py tests\test_vision_stack_runtime.py tests\test_strip_process_bands.py tests\test_project_writer.py -q
```

Expected: all selected tests pass.

Also run the default-behavior guard:

```powershell
Remove-Item Env:\TRADUZAI_TEXT_MASK_ENGINE -ErrorAction SilentlyContinue
Remove-Item Env:\TRADUZAI_BUBBLE_PRIMARY_ENGINE -ErrorAction SilentlyContinue
Remove-Item Env:\TRADUZAI_INPAINT_PRIMARY_ENGINE -ErrorAction SilentlyContinue
python -m pytest tests\test_mask_builder.py tests\test_vision_stack_runtime.py -q
```

Expected: existing default behavior still passes.

Commit only if test fixtures/manifests changed:

```powershell
git status --short
```

---

## Task 10: Visual Comparison On One Chapter

Use unique output directories. Do not overwrite previous runs.

### AOT ROI run

```powershell
cd N:\TraduzAI\.worktrees\bubble-mask-cleaner-v2\pipeline
$env:TRADUZAI_TEXT_MASK_ENGINE='component_bubble_cleaner'
$env:TRADUZAI_INPAINT_PRIMARY_ENGINE='aot_manga_roi'
python main.py --input "D:\Mihon pra pc\downloads\mangas\Manhwatop (EN)\1 Second\Chapter 1.cbz" --work "1 Second" --target pt-BR --mode real --output "N:\TraduzAI\DEBUGM\runs\one_second_ch1_component_aot_roi_v2" --debug --export-mode debug
```

### LaMa ONNX ROI run

```powershell
$env:TRADUZAI_TEXT_MASK_ENGINE='component_bubble_cleaner'
$env:TRADUZAI_INPAINT_PRIMARY_ENGINE='lama_onnx'
python main.py --input "D:\Mihon pra pc\downloads\mangas\Manhwatop (EN)\1 Second\Chapter 1.cbz" --work "1 Second" --target pt-BR --mode real --output "N:\TraduzAI\DEBUGM\runs\one_second_ch1_component_lama_onnx_v2" --debug --export-mode debug
```

Collect:

```text
export_gate
completion_status
critical_issue_count
text_residual_after_inpaint
weak_text_residual_after_inpaint
mask_outside_balloon
fit_below_minimum_legible
render_on_art_suspected
average_inpaint_ms
mask_evidence kind counts
```

Inspect known pages:

- `LET'S GO!!`
- `Search` UI panel
- `Hosu 24 years old Unemployed`
- tiny text in balloons
- CJK/Korean SFX that must be preserved
- connected/double balloon with missing upper text

Expected:

- No new SFX/art damage.
- No page-wide UI damage.
- Text residuals decrease versus current default or older experimental run.
- Export remains BLOCK when visual issues remain.

---

## Task 11: Second Chapter Smoke

Pick one additional English chapter from:

```text
D:\Mihon pra pc\downloads\mangas
```

Criteria:

- folder name contains `(EN)`;
- chapter has normal balloons plus SFX/art;
- do not run every chapter at once.

Run only the better engine from Task 10 first. If it regresses, stop and diagnose before trying another chapter.

Expected:

- No Korean/CJK SFX inpaint unless inside a confirmed BubbleMask and selected as translatable text.
- No tiny Portuguese text caused by layout fallback.
- No false success when export gate is BLOCK.

---

## Task 12: Promotion Decision

Promote only if all are true:

- Unit suites in Task 9 pass.
- One Second visual run improves residuals without increasing art/SFX damage.
- Second chapter smoke does not show new destructive inpaint.
- `mask_evidence.kind == "component_bubble_cleaner"` appears on real balloon cases.
- `mask_evidence.kind == "none"` count is explained and not caused by lost BubbleMask propagation.
- `project.json` saves successfully with no NumPy serialization errors.
- Default behavior without env vars remains unchanged.

If promoted, do it in a separate commit:

```powershell
git add pipeline
git commit -m "feat: promote safe component bubble mask inpaint path"
```

If not promoted, keep it experimental and report:

- changed files;
- tests passed/failed;
- visual issue counts;
- exact pages still failing;
- whether AOT ROI or LaMa ONNX was better.

---

## Agent Split

- **Agent Mask:** Tasks 1, 3, 4, 8. Owns `mask_builder.py` and mask evidence.
- **Agent Strip Contract:** Task 2. Owns `strip/types.py`, `strip/detect_balloons.py`, `strip/process_bands.py`, and `project_writer.py`.
- **Agent Inpaint:** Tasks 5, 6, 7. Owns ROI geometry, runtime route, and LaMa ONNX adapter.
- **Agent Validation:** Tasks 9, 10, 11, 12. Owns test runs, visual comparison, and promotion recommendation.

Each agent works in the same dedicated worktree branch, but only one agent edits a given file group at a time. Agents must report `git diff --stat` and focused test output before handoff.

---

## Conflict Avoidance Checklist

- Before each task:

```powershell
git status --short
```

- Do not edit files outside the task's file list.
- If a needed file is already modified by another agent, stop and coordinate instead of overwriting.
- Do not run `git reset --hard`, `git checkout --`, or destructive cleanup.
- Do not add `DEBUGM/runs`, downloaded models, caches, or screenshots to git.
- Keep experimental env vars local to the shell used for visual runs.


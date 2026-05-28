# Koharu OCR Mask Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the automatic visual pipeline follow the Koharu OCR/mask/inpaint contract exactly: no low-confidence drops, no class/type/skip decisions, title-only cover/logo/art rules, glyph-only masks bounded by real BubbleMask IDs, and Koharu-style fast fill.

**Architecture:** Keep OCR, masking, inpaint, typesetting, and QA as separate steps, but remove old decision inputs from their automatic control flow. OCR always retains text and repairs joined/truncated text before review. Inpaint uses a single visual contract: text/glyph mask first, BubbleMask ID as the only real containment boundary, AOT as the default fallback, and fast fill only for low-variance pixels inside the same BubbleMask ID.

**Tech Stack:** Python 3.12 pipeline, pytest, OpenCV/numpy masks, existing Koharu Rust reference files under `koharu/koharu-ml/src/inpainting/`, React/TypeScript/Rust only for compatibility checks.

---

## File Structure

**Reference, do not modify:**
- `koharu/koharu-ml/src/inpainting/mask.rs`: exact rule for AOT/Lama mask expansion: grow detected glyph pixels only; do not fill text block or bubble background; use segmented bubble IDs as hard constraints.
- `koharu/koharu-ml/src/inpainting/balloon.rs`: exact rule for fast fill: use BubbleMask IDs, sample only same-bubble unmasked background, fill only masked pixels in that ID.

**Python files to modify:**
- `pipeline/vision_stack/engine_presets.py`: enforce segmenter and bubble segmenter for every preset.
- `pipeline/vision_stack/runtime.py`: remove low-confidence drops, cover/logo/noise automatic routing, type/class/skip decisions, bbox-based cleanup paths, and wire title-gated cover/logo logic.
- `pipeline/ocr/postprocess.py`: replace low-confidence and cover-logo filters with retention/recovery behavior.
- `pipeline/ocr/ocr_normalizer.py`: complete `ocr_truncated_or_joined` repair contract.
- `pipeline/ocr/contextual_reviewer.py`: stop turning low confidence, logo, noise, scanlator credit into skip/preserve decisions.
- `pipeline/ocr/text_router.py`: keep only title-gated title/logo match when a user title was provided; otherwise return normal text.
- `pipeline/inpainter/mask_builder.py`: replace current broad mask heuristics with Koharu glyph-only expansion bounded by BubbleMask ID.
- `pipeline/inpainter/__init__.py`: replace current fast fill and bbox cleanup logic with Koharu-style BubbleMask ID fill and AOT fallback.
- `pipeline/typesetter/renderer.py`: stop using `tipo`, `content_class`, `balloon_type`, `skip_processing`, `preserve_original` to decide layout/render policy; keep `rotation_deg`.
- `pipeline/main.py`: stop normalizing old class/skip fields into automatic decisions and stop letting old flags affect export routing.
- `pipeline/qa/export_gate.py`: export blocks only on real visual failure after repair attempts.
- `pipeline/qa/translation_qa.py`: remove severity/blocking role for removed filters.
- `pipeline/project_writer.py`: write compatibility fields only as neutral metadata, not decision inputs.

**Tests to modify or create:**
- `pipeline/tests/test_engine_presets.py`
- `pipeline/tests/test_ocr_postprocess.py`
- `pipeline/tests/test_ocr_normalizer.py`
- `pipeline/tests/test_ocr_retention.py`
- `pipeline/tests/test_mask_builder.py`
- `pipeline/tests/test_inpaint_mask_geometry.py`
- `pipeline/tests/test_vision_stack_runtime.py`
- `pipeline/tests/test_export_gate.py`
- `pipeline/tests/test_typesetting_renderer.py`
- `pipeline/tests/test_final_geometry_contract.py`

---

### Task 1: Baseline And Contract Tests

**Files:**
- Modify: `pipeline/tests/test_engine_presets.py`
- Modify: `pipeline/tests/test_ocr_postprocess.py`
- Modify: `pipeline/tests/test_ocr_normalizer.py`
- Modify: `pipeline/tests/test_ocr_retention.py`
- Modify: `pipeline/tests/test_mask_builder.py`
- Modify: `pipeline/tests/test_inpaint_mask_geometry.py`
- Modify: `pipeline/tests/test_vision_stack_runtime.py`
- Modify: `pipeline/tests/test_export_gate.py`

- [ ] **Step 1: Confirm dirty tree before work**

Run:

```powershell
git status --short
```

Expected: dirty tree is allowed. Do not reset or revert unrelated files.

- [ ] **Step 2: Add preset contract test**

Append to `pipeline/tests/test_engine_presets.py`:

```python
def test_all_presets_keep_segmenter_and_bubble_segmenter_enabled():
    from vision_stack.engine_presets import list_engine_presets, engine_steps_for_preset

    for preset in list_engine_presets():
        assert preset.segmenter == "comic-text-detector-seg"
        assert preset.bubble_segmenter == "speech-bubble-segmentation"
        assert preset.segmenter not in {"", "default", "disabled"}
        assert preset.bubble_segmenter not in {"", "default", "disabled"}
        steps = engine_steps_for_preset(preset)
        assert "comic-text-detector-seg" in steps
        assert "speech-bubble-segmentation" in steps
        assert "aot-inpainting" in steps
```

- [ ] **Step 3: Add low-confidence retention test**

Append to `pipeline/tests/test_ocr_retention.py`:

```python
def test_low_confidence_never_sets_skip_or_preserve():
    from ocr.postprocess import normalize_ocr_record_for_pipeline

    record = normalize_ocr_record_for_pipeline(
        {
            "text": "LET'S GO!!",
            "confidence": 0.12,
            "bbox": [10, 10, 130, 42],
            "qa_flags": ["low_confidence_visual_noise"],
            "skip_processing": True,
            "preserve_original": True,
            "content_class": "noise",
        },
        work_title="",
        work_title_aliases=[],
        work_title_user_provided=False,
    )

    assert record["text"] == "LET'S GO!!"
    assert record.get("skip_processing") is not True
    assert record.get("preserve_original") is not True
    assert "low_confidence_visual_noise" not in record.get("qa_flags", [])
    assert record.get("content_class") in (None, "", "text")
```

- [ ] **Step 4: Add joined/truncated repair-before-review test**

Append to `pipeline/tests/test_ocr_normalizer.py`:

```python
def test_joined_ocr_is_repaired_before_review_flag_survives():
    from ocr.ocr_normalizer import repair_ocr_truncated_or_joined

    repaired = repair_ocr_truncated_or_joined(
        {
            "text": "What!Then,why did we come to the cafe,what are you hiding?",
            "bbox": [24, 18, 300, 120],
            "qa_flags": ["ocr_truncated_or_joined"],
            "line_polygons": [
                [[24, 18], [290, 18], [290, 50], [24, 50]],
                [[24, 54], [280, 54], [280, 88], [24, 88]],
            ],
        }
    )

    assert repaired["text"] == "What! Then, why did we come to the cafe, what are you hiding?"
    assert repaired.get("ocr_repair_status") == "repaired"
    assert "ocr_truncated_or_joined" not in repaired.get("qa_flags", [])
    assert repaired.get("route_action") not in {"review_required", "preserve_original"}
```

- [ ] **Step 5: Add title-gated cover/logo test**

Append to `pipeline/tests/test_ocr_postprocess.py`:

```python
def test_cover_logo_rules_are_disabled_without_user_title():
    from ocr.text_router import route_text_record

    routed = route_text_record(
        {
            "text": "REGIONAL FIREMAN RECRUITMENT TEST",
            "bbox": [10, 10, 320, 80],
            "confidence": 0.91,
        },
        work_title="",
        work_title_aliases=[],
        work_title_user_provided=False,
    )

    assert routed.get("route_action") in (None, "", "translate_inpaint_render")
    assert routed.get("content_class") in (None, "", "text")
    assert routed.get("skip_processing") is not True
    assert routed.get("preserve_original") is not True
```

- [ ] **Step 6: Add Koharu mask contract test**

Append to `pipeline/tests/test_mask_builder.py`:

```python
def test_koharu_text_mask_ignores_balloon_bbox_and_class_fields():
    import numpy as np
    from inpainter.mask_builder import build_inpaint_mask

    image = np.full((80, 140, 3), 255, dtype=np.uint8)
    block = {
        "text": "SIM, NAO FUNCIONA",
        "bbox": [0, 0, 140, 80],
        "balloon_bbox": [0, 0, 140, 80],
        "content_class": "noise",
        "tipo": "sfx",
        "balloon_type": "white",
        "skip_processing": True,
        "preserve_original": True,
        "line_polygons": [
            [[42, 26], [98, 26], [98, 36], [42, 36]],
            [[42, 42], [104, 42], [104, 52], [42, 52]],
        ],
        "bubble_id": "b1",
        "bubble_mask": None,
    }

    mask = build_inpaint_mask(block, image.shape, image_rgb=image)

    assert mask is not None
    assert int(np.count_nonzero(mask)) > 0
    assert int(np.count_nonzero(mask[:, :8])) == 0
    assert int(np.count_nonzero(mask[:, 132:])) == 0
    assert "mask_density_high" not in block.get("qa_flags", [])
```

- [ ] **Step 7: Add Koharu fast-fill contract test**

Append to `pipeline/tests/test_inpaint_mask_geometry.py`:

```python
def test_fast_fill_only_changes_masked_pixels_inside_same_bubble_id():
    import numpy as np
    from inpainter import apply_koharu_bubble_fast_fill

    image = np.full((64, 96, 3), 240, dtype=np.uint8)
    image[28:36, 32:64] = 10
    mask = np.zeros((64, 96), dtype=np.uint8)
    mask[28:36, 32:64] = 255
    bubble_mask = np.zeros((64, 96), dtype=np.uint8)
    bubble_mask[12:52, 16:80] = 3

    result, remaining, metadata = apply_koharu_bubble_fast_fill(image, mask, bubble_mask)

    assert metadata["filled_pixels"] == int(np.count_nonzero(mask))
    assert int(np.count_nonzero(remaining)) == 0
    assert np.array_equal(result[0:12, :, :], image[0:12, :, :])
    assert np.array_equal(result[:, 0:16, :], image[:, 0:16, :])
    assert np.all(result[28:36, 32:64] == 240)
```

- [ ] **Step 8: Verify tests fail before implementation**

Run:

```powershell
python -m pytest pipeline/tests/test_engine_presets.py pipeline/tests/test_ocr_retention.py pipeline/tests/test_ocr_normalizer.py pipeline/tests/test_ocr_postprocess.py pipeline/tests/test_mask_builder.py pipeline/tests/test_inpaint_mask_geometry.py -q
```

Expected: fail because the old functions/behavior still exist.

---

### Task 2: Enforce Segmenter And Bubble Segmenter Everywhere

**Files:**
- Modify: `pipeline/vision_stack/engine_presets.py`
- Modify: `pipeline/vision_stack/runtime.py`
- Test: `pipeline/tests/test_engine_presets.py`
- Test: `pipeline/tests/test_vision_stack_runtime.py`

- [ ] **Step 1: Update preset constants**

In `pipeline/vision_stack/engine_presets.py`, ensure the preset constructor values use:

```python
COMIC_TEXT_DETECTOR_SEGMENTER = "comic-text-detector-seg"
SPEECH_BUBBLE_SEGMENTER = "speech-bubble-segmentation"
DEFAULT_INPAINTER = "aot-inpainting"
```

Every `EnginePreset(...)` must set:

```python
segmenter=COMIC_TEXT_DETECTOR_SEGMENTER,
bubble_segmenter=SPEECH_BUBBLE_SEGMENTER,
inpainter=DEFAULT_INPAINTER,
```

- [ ] **Step 2: Make step expansion non-optional**

In `engine_steps_for_preset`, keep detector and OCR defaults if required, but never filter out the segmenter, bubble segmenter, or AOT:

```python
def engine_steps_for_preset(preset: EnginePreset) -> list[str]:
    steps = [
        preset.detector,
        preset.segmenter or COMIC_TEXT_DETECTOR_SEGMENTER,
        preset.bubble_segmenter or SPEECH_BUBBLE_SEGMENTER,
        preset.ocr,
        preset.inpainter or DEFAULT_INPAINTER,
        preset.renderer,
    ]
    return [
        step
        for step in steps
        if step and step != "disabled" and not (
            step == "default" and step not in {COMIC_TEXT_DETECTOR_SEGMENTER, SPEECH_BUBBLE_SEGMENTER, DEFAULT_INPAINTER}
        )
    ]
```

- [ ] **Step 3: Normalize runtime preset fallback**

In `pipeline/vision_stack/runtime.py`, when resolving page preset dictionaries, override missing/disabled fields:

```python
def _force_koharu_visual_engines(preset: dict) -> dict:
    normalized = dict(preset or {})
    normalized["segmenter"] = "comic-text-detector-seg"
    normalized["bubble_segmenter"] = "speech-bubble-segmentation"
    normalized["inpainter"] = "aot-inpainting"
    return normalized
```

Call this inside the existing page preset resolver before runtime decisions use it.

- [ ] **Step 4: Run preset tests**

Run:

```powershell
python -m pytest pipeline/tests/test_engine_presets.py pipeline/tests/test_vision_stack_runtime.py -q
```

Expected: pass for preset/engine metadata tests.

---

### Task 3: Remove Low-Confidence Filtering Completely

**Files:**
- Modify: `pipeline/ocr/postprocess.py`
- Modify: `pipeline/ocr/contextual_reviewer.py`
- Modify: `pipeline/vision_stack/runtime.py`
- Modify: `pipeline/qa/export_gate.py`
- Modify: `pipeline/qa/translation_qa.py`
- Test: `pipeline/tests/test_ocr_postprocess.py`
- Test: `pipeline/tests/test_ocr_retention.py`
- Test: `pipeline/tests/test_export_gate.py`

- [ ] **Step 1: Remove or neutralize `is_low_confidence_visual_noise`**

In `pipeline/ocr/postprocess.py`, replace the function body with a compatibility-only false result:

```python
def is_low_confidence_visual_noise(*_args, **_kwargs) -> bool:
    """Compatibility shim. Low confidence is metadata only, never a drop/skip rule."""
    return False
```

- [ ] **Step 2: Strip the flag during normalization**

Add or update this helper in `pipeline/ocr/postprocess.py`:

```python
REMOVED_OCR_DECISION_FLAGS = {
    "low_confidence_visual_noise",
}


def strip_removed_ocr_decision_flags(record: dict) -> dict:
    cleaned = dict(record)
    flags = [
        str(flag)
        for flag in cleaned.get("qa_flags", [])
        if str(flag) not in REMOVED_OCR_DECISION_FLAGS
    ]
    cleaned["qa_flags"] = flags
    if cleaned.get("route_reason") in REMOVED_OCR_DECISION_FLAGS:
        cleaned.pop("route_reason", None)
    return cleaned
```

- [ ] **Step 3: Remove skip/preserve from low confidence paths**

In every low-confidence branch in `pipeline/vision_stack/runtime.py` and `pipeline/ocr/contextual_reviewer.py`, replace skip/preserve assignment with:

```python
text["confidence"] = float(text.get("confidence") or 0.0)
text.pop("skip_processing", None)
text.pop("preserve_original", None)
text["route_action"] = "translate_inpaint_render"
```

- [ ] **Step 4: Remove QA severity for low-confidence drop**

In `pipeline/qa/translation_qa.py` and `pipeline/qa/export_gate.py`, remove `low_confidence_visual_noise` from blocking/review severity maps. If existing fixtures still carry it, ignore it:

```python
IGNORED_QA_FLAGS = {
    "low_confidence_visual_noise",
}
```

Use:

```python
flags = [flag for flag in flags if flag not in IGNORED_QA_FLAGS]
```

- [ ] **Step 5: Run OCR retention/export tests**

Run:

```powershell
python -m pytest pipeline/tests/test_ocr_postprocess.py pipeline/tests/test_ocr_retention.py pipeline/tests/test_export_gate.py -q
```

Expected: low-confidence records are retained and never block export by that flag.

---

### Task 4: Complete Mandatory `ocr_truncated_or_joined` Repair

**Files:**
- Modify: `pipeline/ocr/ocr_normalizer.py`
- Modify: `pipeline/ocr/postprocess.py`
- Modify: `pipeline/vision_stack/runtime.py`
- Modify: `pipeline/qa/export_gate.py`
- Test: `pipeline/tests/test_ocr_normalizer.py`
- Test: `pipeline/tests/test_ocr_retention.py`
- Test: `pipeline/tests/test_export_gate.py`

- [ ] **Step 1: Implement deterministic text repair**

In `pipeline/ocr/ocr_normalizer.py`, add:

```python
import re

JOINED_PUNCT_REPAIRS = [
    (re.compile(r"([!?.,])(?=[A-Za-z])"), r"\1 "),
    (re.compile(r"(?<=[a-z])(?=[A-Z][a-z])"), r" "),
    (re.compile(r"\s+"), " "),
]


def repair_joined_ocr_text(text: str) -> str:
    repaired = str(text or "").strip()
    for pattern, replacement in JOINED_PUNCT_REPAIRS:
        repaired = pattern.sub(replacement, repaired)
    repaired = repaired.replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
    return repaired.strip()
```

- [ ] **Step 2: Implement record-level repair**

Add:

```python
def repair_ocr_truncated_or_joined(record: dict) -> dict:
    repaired = dict(record)
    flags = [str(flag) for flag in repaired.get("qa_flags", [])]
    if "ocr_truncated_or_joined" not in flags:
        return repaired

    before = str(repaired.get("text") or repaired.get("original") or "")
    after = repair_joined_ocr_text(before)
    if after and after != before:
        repaired["text"] = after
        repaired["original"] = after
        repaired["qa_flags"] = [flag for flag in flags if flag != "ocr_truncated_or_joined"]
        repaired["ocr_repair_status"] = "repaired"
        repaired["ocr_repair_action"] = "joined_punctuation_spacing"
        repaired["route_action"] = "translate_inpaint_render"
        repaired.pop("skip_processing", None)
        repaired.pop("preserve_original", None)
        return repaired

    repaired["ocr_repair_status"] = "repair_failed"
    repaired["qa_flags"] = flags
    return repaired
```

- [ ] **Step 3: Call repair before routing/review**

In `pipeline/vision_stack/runtime.py`, immediately after OCR records are built and before route/review decisions:

```python
from ocr.ocr_normalizer import repair_ocr_truncated_or_joined

texts = [repair_ocr_truncated_or_joined(text) for text in texts]
```

Use the local import style already present in the file if import cycles require fallback.

- [ ] **Step 4: Export gate blocks only failed repair**

In `pipeline/qa/export_gate.py`, treat `ocr_truncated_or_joined` as blocking/review only when repair failed:

```python
if flag == "ocr_truncated_or_joined" and layer.get("ocr_repair_status") != "repair_failed":
    continue
```

- [ ] **Step 5: Run OCR repair tests**

Run:

```powershell
python -m pytest pipeline/tests/test_ocr_normalizer.py pipeline/tests/test_ocr_retention.py pipeline/tests/test_export_gate.py -q
```

Expected: joined text is repaired; export gate only reviews failed repair.

---

### Task 5: Title-Gated Cover/Logo/Art Rules Only

**Files:**
- Modify: `pipeline/ocr/text_router.py`
- Modify: `pipeline/ocr/postprocess.py`
- Modify: `pipeline/vision_stack/runtime.py`
- Modify: `pipeline/ocr/contextual_reviewer.py`
- Test: `pipeline/tests/test_ocr_postprocess.py`
- Test: `pipeline/tests/test_ocr_normalizer.py`
- Test: `pipeline/tests/test_vision_stack_runtime.py`

- [ ] **Step 1: Add title alias matching only**

In `pipeline/ocr/text_router.py`, implement or replace the route function with:

```python
import re


def _title_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _title_alias_keys(work_title: str, aliases: list[str] | tuple[str, ...]) -> set[str]:
    keys = {_title_key(work_title)}
    keys.update(_title_key(alias) for alias in aliases or [])
    return {key for key in keys if key}


def route_text_record(
    record: dict,
    *,
    work_title: str = "",
    work_title_aliases: list[str] | tuple[str, ...] | None = None,
    work_title_user_provided: bool = False,
) -> dict:
    routed = dict(record)
    routed.pop("skip_processing", None)
    routed.pop("preserve_original", None)
    routed["route_action"] = "translate_inpaint_render"
    routed["content_class"] = "text"

    if not work_title_user_provided:
        return routed

    keys = _title_alias_keys(work_title, work_title_aliases or [])
    text_key = _title_key(routed.get("text") or routed.get("original") or "")
    if text_key and text_key in keys:
        routed["route_action"] = "title_text"
        routed["content_class"] = "title"
    return routed
```

- [ ] **Step 2: Remove old global cover/logo functions from decisions**

In `pipeline/ocr/postprocess.py`, keep compatibility shims returning false unless a title is passed through `route_text_record`:

```python
def is_cover_title_logo(*_args, **_kwargs) -> bool:
    """Compatibility shim. Cover/logo rules are title-gated in text_router."""
    return False
```

- [ ] **Step 3: Remove class-based scanlator/noise/logo skips**

In `pipeline/vision_stack/runtime.py` and `pipeline/ocr/contextual_reviewer.py`, replace code that sets:

```python
content_class = "logo"
content_class = "noise"
content_class = "scanlator_credit"
skip_processing = True
preserve_original = True
```

with:

```python
text = route_text_record(
    text,
    work_title=work_title,
    work_title_aliases=work_title_aliases or [],
    work_title_user_provided=work_title_user_provided,
)
```

- [ ] **Step 4: Run title gate tests**

Run:

```powershell
python -m pytest pipeline/tests/test_ocr_postprocess.py pipeline/tests/test_ocr_normalizer.py pipeline/tests/test_vision_stack_runtime.py -q
```

Expected: without user title, cover/logo/art text is treated as normal text; with matching user title, only title text is routed as title.

---

### Task 6: Remove Type/Class/Skip/Preserve From Automatic Decisions

**Files:**
- Modify: `pipeline/main.py`
- Modify: `pipeline/project_writer.py`
- Modify: `pipeline/typesetter/renderer.py`
- Modify: `pipeline/strip/run.py`
- Modify: `pipeline/strip/process_bands.py`
- Modify: `pipeline/qa/visual_text_leak.py`
- Modify: `src/lib/appStore.ts` only if TypeScript types require compatibility
- Test: `pipeline/tests/test_typesetting_renderer.py`
- Test: `pipeline/tests/test_final_geometry_contract.py`
- Test: `pipeline/tests/test_strip_run.py`

- [ ] **Step 1: Define neutral compatibility sanitizer**

Add in `pipeline/main.py` or an existing shared normalization area:

```python
REMOVED_AUTOMATIC_DECISION_FIELDS = {
    "tipo",
    "content_class",
    "balloon_type",
    "skip_processing",
    "preserve_original",
}


def neutralize_removed_decision_fields(layer: dict) -> dict:
    normalized = dict(layer)
    normalized["tipo"] = "text"
    normalized["content_class"] = "text"
    normalized["balloon_type"] = ""
    normalized["skip_processing"] = False
    normalized["preserve_original"] = False
    normalized["route_action"] = normalized.get("route_action") or "translate_inpaint_render"
    return normalized
```

- [ ] **Step 2: Use the sanitizer only for compatibility output**

Before writing `project.json`, call:

```python
layer = neutralize_removed_decision_fields(layer)
```

Do not use the resulting fields to branch; branch only on visual state such as missing render, residual, overflow, or repair failure.

- [ ] **Step 3: Remove renderer decisions by old fields**

In `pipeline/typesetter/renderer.py`, replace branches such as:

```python
if tipo == "narracao":
    ...
if balloon_type == "white":
    ...
if text_data.get("skip_processing"):
    ...
```

with geometry/rotation based decisions:

```python
rotation_deg = float(text_data.get("rotation_deg") or 0.0)
layout_bbox = _layout_bbox(text_data.get("safe_text_box")) or _layout_bbox(text_data.get("bbox"))
```

The renderer may read `rotation_deg`, `line_polygons`, `bubble_id`, `bubble_mask_bbox`, `safe_text_box`, and `render_bbox`; it must not skip because of removed fields.

- [ ] **Step 4: Remove strip skip-copy behavior**

In `pipeline/strip/run.py` and `pipeline/strip/process_bands.py`, replace automatic skip copies:

```python
if text.get("skip_processing") or text.get("preserve_original"):
    ...
```

with:

```python
if False:
    pass
```

Then remove the dead block in the same edit. The strip path must process text normally unless a real visual operation has no text mask or no BubbleMask and is routed to AOT.

- [ ] **Step 5: Run type/strip/render tests**

Run:

```powershell
python -m pytest pipeline/tests/test_typesetting_renderer.py pipeline/tests/test_final_geometry_contract.py pipeline/tests/test_strip_run.py -q
```

Expected: old fields may exist in output as neutral compatibility metadata, but no automatic branch depends on them.

---

### Task 7: Replace Mask Builder With Koharu Glyph-Only BubbleMask Contract

**Files:**
- Modify: `pipeline/inpainter/mask_builder.py`
- Test: `pipeline/tests/test_mask_builder.py`
- Test: `pipeline/tests/test_inpaint_mask_geometry.py`

- [ ] **Step 1: Remove mask density flag as decision**

Delete or neutralize:

```python
MASK_DENSITY_HIGH = 0.12
```

Replace any append of `mask_density_high` with no-op:

```python
def _record_mask_density_high(*_args, **_kwargs) -> None:
    return None
```

- [ ] **Step 2: Add glyph mask builder**

In `pipeline/inpainter/mask_builder.py`, add:

```python
def build_glyph_text_mask(block: dict, image_shape: tuple[int, ...]) -> np.ndarray | None:
    height, width = _image_hw(image_shape)
    mask = np.zeros((height, width), dtype=np.uint8)
    polygons = _text_geometry_polygons(block, width, height)
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.int32)
        if points.shape[0] >= 3:
            cv2.fillPoly(mask, [points], 255)
    text_bbox = _normalize_bbox(block.get("text_pixel_bbox"), width, height)
    if text_bbox:
        x1, y1, x2, y2 = text_bbox
        mask[y1:y2, x1:x2] = 255
    return mask if np.any(mask) else None
```

- [ ] **Step 3: Add BubbleMask ID clipping**

Add:

```python
def clip_mask_to_bubble_id(text_mask: np.ndarray, bubble_mask: np.ndarray | None, bubble_id: object) -> np.ndarray:
    if not isinstance(bubble_mask, np.ndarray) or bubble_mask.shape[:2] != text_mask.shape[:2]:
        return text_mask
    try:
        numeric_id = int(bubble_id)
    except Exception:
        numeric_id = None
    if numeric_id is None or numeric_id <= 0:
        return text_mask
    clipped = np.where((text_mask > 0) & (bubble_mask == numeric_id), 255, 0).astype(np.uint8)
    return clipped if np.any(clipped) else text_mask
```

- [ ] **Step 4: Replace `build_inpaint_mask` decision body**

The automatic path must start with glyph mask and never return `None` because of old fields:

```python
def build_inpaint_mask(
    block: dict,
    image_shape: tuple[int, ...],
    image_rgb: np.ndarray | None = None,
) -> np.ndarray | None:
    del image_rgb
    text_mask = build_glyph_text_mask(block, image_shape)
    if text_mask is None:
        return None

    bubble_mask = block.get("bubble_mask")
    bubble_id = block.get("bubble_id") or block.get("bubbleId")
    text_mask = clip_mask_to_bubble_id(text_mask, bubble_mask, bubble_id)

    radius = _koharu_glyph_dilate_radius(block)
    if radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (radius * 2 + 1, radius * 2 + 1))
        expanded = cv2.dilate(text_mask, kernel, iterations=1)
        expanded = clip_mask_to_bubble_id(expanded, bubble_mask, bubble_id)
        return expanded.astype(np.uint8)
    return text_mask.astype(np.uint8)
```

- [ ] **Step 5: Add Koharu-style radius helper**

Add:

```python
def _koharu_glyph_dilate_radius(block: dict) -> int:
    font_size = block.get("detected_font_size_px") or block.get("font_size")
    try:
        numeric = float(font_size)
    except Exception:
        numeric = 18.0
    return int(max(2, min(8, round(numeric * 0.16))))
```

- [ ] **Step 6: Remove fallback to `balloon_bbox`**

Delete automatic use of:

```python
block.get("balloon_bbox")
block.get("content_class")
block.get("tipo")
block.get("balloon_type")
block.get("skip_processing")
block.get("preserve_original")
```

from the mask decision path. These names may remain in debug serialization only.

- [ ] **Step 7: Run mask tests**

Run:

```powershell
python -m pytest pipeline/tests/test_mask_builder.py pipeline/tests/test_inpaint_mask_geometry.py -q
```

Expected: masks are glyph-only, do not fill balloon background, and ignore removed fields.

---

### Task 8: Replace Fast Fill With Koharu Bubble ID Fill

**Files:**
- Modify: `pipeline/inpainter/__init__.py`
- Modify: `pipeline/vision_stack/runtime.py`
- Test: `pipeline/tests/test_inpaint_mask_geometry.py`
- Test: `pipeline/tests/test_vision_stack_runtime.py`

- [ ] **Step 1: Add exact Koharu-style fill function**

In `pipeline/inpainter/__init__.py`, add:

```python
def apply_koharu_bubble_fast_fill(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    bubble_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if not isinstance(bubble_mask, np.ndarray) or bubble_mask.shape[:2] != mask.shape[:2]:
        return image_rgb.copy(), mask.copy(), {"filled_pixels": 0, "reason": "missing_bubble_mask"}

    result = image_rgb.copy()
    remaining = np.where(mask > 0, 255, 0).astype(np.uint8)
    filled_pixels = 0
    bubble_ids = sorted(int(v) for v in np.unique(bubble_mask[(remaining > 0) & (bubble_mask > 0)]))

    for bubble_id in bubble_ids:
        inside = bubble_mask == bubble_id
        target = (remaining > 0) & inside
        background = inside & (remaining == 0)
        if not np.any(target) or not np.any(background):
            continue

        samples = image_rgb[background]
        median = np.median(samples, axis=0)
        std = np.std(samples, axis=0)
        if float(np.max(std)) >= 10.0:
            continue

        result[target] = np.asarray(median, dtype=np.uint8)
        remaining[target] = 0
        filled_pixels += int(np.count_nonzero(target))

    return result, remaining, {"filled_pixels": filled_pixels, "bubble_ids": bubble_ids}
```

- [ ] **Step 2: Remove old solid/bbox fast fill decision**

In `pipeline/inpainter/__init__.py`, remove or bypass functions that use:

```python
_solid_fast_fill_override_allowed
_authorized_fast_fill_mask
_fast_fill_union_mask_from_bboxes
balloon_bbox
balloon_type
content_class
tipo
skip_processing
preserve_original
mask_density_high
```

The automatic fast-fill path must call only `apply_koharu_bubble_fast_fill`.

- [ ] **Step 3: Route remaining mask to AOT**

After `apply_koharu_bubble_fast_fill`, call the real inpainter only if `remaining` still has pixels:

```python
fast_image, remaining_mask, fast_meta = apply_koharu_bubble_fast_fill(original_rgb, raw_mask, bubble_mask)
if int(np.count_nonzero(remaining_mask)) > 0:
    output = aot_inpaint(fast_image, remaining_mask)
else:
    output = fast_image
```

Use the existing AOT call wrapper name in `pipeline/inpainter/__init__.py`.

- [ ] **Step 4: Runtime passes real BubbleMask**

In `pipeline/vision_stack/runtime.py`, pass the page/strip BubbleMask array into the inpainter. Do not synthesize it from `balloon_bbox`. If the real BubbleMask is missing, fast fill returns `missing_bubble_mask`, and the full glyph mask goes to AOT.

- [ ] **Step 5: Run inpaint tests**

Run:

```powershell
python -m pytest pipeline/tests/test_inpaint_mask_geometry.py pipeline/tests/test_vision_stack_runtime.py -q
```

Expected: fast fill changes only masked pixels inside the same BubbleMask ID; no bbox cleanup path runs.

---

### Task 9: QA And Export Gate Only Visual Failures

**Files:**
- Modify: `pipeline/qa/export_gate.py`
- Modify: `pipeline/qa/translation_qa.py`
- Modify: `pipeline/qa/visual_text_leak.py`
- Modify: `pipeline/main.py`
- Test: `pipeline/tests/test_export_gate.py`
- Test: `pipeline/tests/test_translation_qa.py`
- Test: `pipeline/tests/test_final_geometry_contract.py`

- [ ] **Step 1: Define ignored legacy flags**

In `pipeline/qa/export_gate.py`:

```python
IGNORED_LEGACY_FLAGS = {
    "low_confidence_visual_noise",
    "cover_title_logo",
    "mask_density_high",
}
```

- [ ] **Step 2: Ignore legacy flags during gate aggregation**

When iterating layer flags:

```python
if flag in IGNORED_LEGACY_FLAGS:
    continue
if flag == "ocr_truncated_or_joined" and layer.get("ocr_repair_status") != "repair_failed":
    continue
```

- [ ] **Step 3: Remove skip/preserve leak exemptions**

In `pipeline/qa/visual_text_leak.py`, remove exemptions:

```python
layer.get("skip_processing")
layer.get("preserve_original")
layer.get("tipo") == "sfx"
```

Visual leak QA must inspect visible original text regardless of old metadata.

- [ ] **Step 4: Keep only real visual blockers**

Export gate should block only:

```python
REAL_VISUAL_BLOCKERS = {
    "render_missing",
    "missing_render_bbox",
    "TEXT_OVERFLOW",
    "TEXT_CLIPPED",
    "render_outside_balloon",
    "weak_text_residual_after_inpaint",
    "inpaint_residual_confirmed",
    "ocr_truncated_or_joined",
}
```

For `ocr_truncated_or_joined`, block only when `ocr_repair_status == "repair_failed"`.

- [ ] **Step 5: Run QA tests**

Run:

```powershell
python -m pytest pipeline/tests/test_export_gate.py pipeline/tests/test_translation_qa.py pipeline/tests/test_final_geometry_contract.py -q
```

Expected: export gate ignores removed filters and blocks only real visual failures.

---

### Task 10: Manual/Editor Compatibility Check

**Files:**
- Modify only if required: `src/lib/tauri.ts`
- Modify only if required: `src/lib/stores/appStore.ts`
- Modify only if required: `src/components/editor/PropertyEditor.tsx`
- Test: `src/lib/__tests__/pipelineCompletion.test.ts`
- Test: `src/lib/stores/__tests__/editorBitmapTools.test.ts`

- [ ] **Step 1: Keep old fields compatible but neutral**

If TypeScript requires `tipo`, keep it as a compatibility value only:

```ts
tipo: "text" as TextEntry["tipo"]
```

Do not add UI behavior that depends on `tipo`, `content_class`, `skip_processing`, or `preserve_original`.

- [ ] **Step 2: Ensure frontend does not approve blocked visual runs**

Run:

```powershell
npx tsc --noEmit
```

Expected: TypeScript passes.

- [ ] **Step 3: Run frontend tests if available**

Run:

```powershell
npm test -- --run src/lib/__tests__/pipelineCompletion.test.ts src/lib/stores/__tests__/editorBitmapTools.test.ts
```

Expected: tests pass or command reports the repo's current test runner limitation. Record the exact result.

---

### Task 11: Final Static Sweep

**Files:**
- Inspect only unless failures are found.

- [ ] **Step 1: Search for removed decision flags**

Run:

```powershell
rg -n "low_confidence_visual_noise|cover_title_logo|mask_density_high" pipeline src src-tauri fonts -g "*.py" -g "*.ts" -g "*.tsx" -g "*.rs" -g "*.json"
```

Expected: no automatic decision path remains. Allowed hits are compatibility shims, tests asserting removal, or debug report labels that are explicitly ignored.

- [ ] **Step 2: Search for old decision fields**

Run:

```powershell
rg -n "skip_processing|preserve_original|content_class|balloon_type|\\btipo\\b" pipeline src src-tauri fonts -g "*.py" -g "*.ts" -g "*.tsx" -g "*.rs" -g "*.json"
```

Expected: remaining hits are neutral compatibility serialization, tests, or UI display. No branch may use these fields to skip OCR, translation, inpaint, render, QA, or export.

- [ ] **Step 3: Search for bbox-based fast fill/mask**

Run:

```powershell
rg -n "balloon_bbox|_authorized_fast_fill_mask|_solid_fast_fill_override_allowed|_fast_fill_union_mask_from_bboxes" pipeline/inpainter pipeline/vision_stack -g "*.py"
```

Expected: `balloon_bbox` may remain for layout/debug compatibility, but not in mask expansion, fast fill authorization, or cleanup decisions.

---

### Task 12: Final Test And Visual Validation

**Files:**
- No source edits unless validation fails.

- [ ] **Step 1: Run focused Python suite**

Run:

```powershell
python -m pytest pipeline/tests/test_engine_presets.py pipeline/tests/test_ocr_postprocess.py pipeline/tests/test_ocr_normalizer.py pipeline/tests/test_ocr_retention.py pipeline/tests/test_mask_builder.py pipeline/tests/test_inpaint_mask_geometry.py pipeline/tests/test_vision_stack_runtime.py pipeline/tests/test_export_gate.py pipeline/tests/test_typesetting_renderer.py pipeline/tests/test_final_geometry_contract.py -q
```

Expected: all pass.

- [ ] **Step 2: Run TypeScript check**

Run:

```powershell
npx tsc --noEmit
```

Expected: pass.

- [ ] **Step 3: Run Rust tests**

Run:

```powershell
Push-Location src-tauri
cargo test
Pop-Location
```

Expected: pass.

- [ ] **Step 4: Run One Second chapters with debug**

Use the existing command pattern from the current repo/debug workflow for One Second cap 1 and cap 2. The run must use the automatic pipeline, debug enabled, and the new preset contract.

Expected visual checkpoints:
- Page 22: `SIM, NAO FUNCIONA` does not lose balloon border.
- Page 23: neighboring bubbles are not merged by old connected-balloon/type heuristics.
- Page 37: text remains inside the bubble and does not get shrunk because of safe box degeneration.
- Page 42: leftover English glyphs are masked by text/glyph evidence, not by broad bbox.
- Page 45: no `fit_below_minimum_legible` critical caused by old layout class/type routing.
- Page 53: white rectangular text area uses glyph mask/AOT and does not rely on `balloon_bbox`.

- [ ] **Step 5: Report result**

Final report must include:
- files changed.
- tests run and exact pass/fail.
- debug run folder path.
- export gate result.
- visual status for pages 22, 23, 37, 42, 45, 53.
- any remaining old-field hits from `rg` and why each is compatibility-only.

---

## Execution Order By Agents

1. **Agent OCR:** Tasks 3, 4, 5.
2. **Agent Mask/Inpaint:** Tasks 7, 8.
3. **Agent Schema/Typesetting:** Task 6.
4. **Agent QA/Runtime/UI:** Tasks 2, 9, 10, 11, 12.

Agents must not commit, reset, revert, delete debug output, or stage files unless the user explicitly asks. Each agent must preserve unrelated dirty-tree changes and report only files it touched.

## Self-Review

- Spec coverage: every user requirement is mapped to at least one task.
- No alternate approach: the plan uses Koharu `mask.rs` and `balloon.rs` as the implementation reference for mask and fast fill.
- No old automatic decisions: `low_confidence_visual_noise`, `cover_title_logo`, `mask_density_high`, `content_class`, `tipo`, `balloon_type`, `skip_processing`, and `preserve_original` are either removed from control flow or neutral compatibility metadata.
- Repair before review: `ocr_truncated_or_joined` is repaired first and only blocks after `ocr_repair_status == "repair_failed"`.
- Validation: focused Python, TS, Rust, static `rg`, and visual chapter reruns are all required before claiming completion.

# CJK OCR-Guided Text Mask Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an OCR-guided CJK text-mask pipeline that removes manga/manhwa/manhua text more precisely, including SFX, while reducing mask invasion into art.

**Architecture:** Keep the current Manga-Text-Segmentation-2025 + AOT path as the baseline. Add optional evidence providers for OCR geometry, Hi-SAM stroke masks, CRAFT character heatmaps, and oar-ocr word boxes, then fuse them into a confidence-scored final mask. Every new backend must be lazy-loaded, optional, and safe to disable so the current manga/manhwa presets keep working.

**Tech Stack:** Python 3.12, OpenCV, NumPy, PyTorch/ONNX where available, current `pipeline/vision_stack` runtime, optional external model backends: Hi-SAM, CRAFT, oar-ocr.

---

## Current Baseline

The current CJK mask flow is:

```text
Detector/OCR -> _vision_blocks/texts
Manga-Text-Segmentation-2025 -> pixel-level text mask
CJK orphan recovery + dark-core absorption -> SFX cleanup
AOT inpainting -> mask-only cleanup
```

Relevant files:

- `pipeline/vision_stack/engine_presets.py`
- `pipeline/vision_stack/runtime.py`
- `pipeline/vision_stack/cjk_segmentation_mask.py`
- `pipeline/vision_stack/manga_text_segmenter.py`
- `pipeline/vision_stack/aot_inpainter.py`
- `pipeline/tests/test_cjk_segmentation_mask.py`
- `pipeline/tests/test_vision_stack_runtime.py`
- `tools/cjk_engine_preset_poc.py`

Do not remove the current CJK path. This plan adds a refinement layer above it.

---

## Non-Goals

- Do not use MaskTextSpotterV3 in production; its license is non-commercial.
- Do not make `comic-text-detector` a required runtime dependency; keep it optional due GPL-3.0.
- Do not require Hi-SAM, CRAFT, or oar-ocr for default startup.
- Do not run full-page Hi-SAM on tall webtoon/manhwa pages; use ROIs/crops only.
- Do not replace PaddleOCR-VL as the main OCR in this phase.

---

## Phase 1: Lock the Baseline and Diagnostics

**Files:**

- Modify: `tools/cjk_engine_preset_poc.py`
- Create: `pipeline/tests/test_cjk_mask_diagnostics.py`
- Test data: use synthetic images in tests, not protected manga pages.

**Step 1: Add a diagnostic summary schema test**

Create `pipeline/tests/test_cjk_mask_diagnostics.py` with a small synthetic page and assert the diagnostic output can represent:

```python
def test_cjk_mask_diagnostic_summary_has_required_fields():
    summary = {
        "text_mask_pixels": 120,
        "inpaint_mask_pixels": 180,
        "changed_outside_input_mask_pixels": 0,
        "coverage": {
            "ocr_dark_inside_mask": 80,
            "ocr_dark_outside_mask": 2,
        },
    }
    assert summary["changed_outside_input_mask_pixels"] == 0
    assert "coverage" in summary
```

**Step 2: Extend `tools/cjk_engine_preset_poc.py`**

Add optional per-target coverage metrics:

```text
orig_dark_inside_mask
orig_dark_outside_mask
dark_pixels_final
dark_outside_mask_final
mask_ratio
```

**Step 3: Run focused tests**

Run:

```powershell
$env:PYTHONPATH='pipeline'; pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_cjk_mask_diagnostics.py pipeline\tests\test_cjk_segmentation_mask.py -q
```

Expected: all selected tests pass.

**Step 4: Generate baseline artifacts manually**

Run the manga preset against `C:\Users\PICHAU\Downloads\japonesmang.png` and save to:

```text
.codex-tmp/japonesmang_ocr_guided_baseline/
```

Expected artifacts:

```text
05_text_mask.png
06_inpaint_input_mask.png
07_overlay.png
inpaint/japonesmang.png
summary.json
crops/*.png
```

---

## Phase 2: Add Evidence Data Contracts

**Files:**

- Create: `pipeline/vision_stack/text_mask_evidence.py`
- Test: `pipeline/tests/test_text_mask_evidence.py`

**Step 1: Write failing tests**

Create tests for normalized OCR/text evidence:

```python
from vision_stack.text_mask_evidence import TextEvidence, normalize_text_evidence


def test_normalize_text_evidence_keeps_bbox_and_text():
    page = {
        "texts": [{"text": "ガシャーン", "bbox": [10, 20, 80, 50], "confidence": 0.91}],
        "_vision_blocks": [{"bbox": [8, 18, 84, 54]}],
    }

    evidence = normalize_text_evidence(page, width=100, height=100)

    assert evidence[0].text == "ガシャーン"
    assert evidence[0].bbox == [10, 20, 80, 50]
    assert evidence[0].source == "ocr"
```

**Step 2: Implement minimal dataclasses**

`pipeline/vision_stack/text_mask_evidence.py` should define:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TextEvidence:
    bbox: list[int]
    text: str = ""
    confidence: float = 0.0
    source: str = "unknown"
    line_polygons: list[list[list[float]]] = field(default_factory=list)
    word_boxes: list[list[int]] = field(default_factory=list)
    char_boxes: list[list[int]] = field(default_factory=list)
    preserve_original: bool = False


def normalize_text_evidence(page_result: dict[str, Any], width: int, height: int) -> list[TextEvidence]:
    ...
```

Use existing bbox normalization style from `pipeline/vision_stack/cjk_segmentation_mask.py`.

**Step 3: Add coverage helper tests**

Test a function:

```python
def measure_mask_coverage(mask, image_rgb, evidence) -> dict:
    ...
```

Expected metrics:

```text
mask_pixels
dark_pixels
dark_inside_mask
dark_outside_mask
coverage_ratio
```

**Step 4: Run tests**

Run:

```powershell
$env:PYTHONPATH='pipeline'; pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_text_mask_evidence.py -q
```

Expected: pass.

---

## Phase 3: Add Optional oar-ocr Geometry Adapter

**Files:**

- Create: `pipeline/vision_stack/oar_ocr_adapter.py`
- Test: `pipeline/tests/test_oar_ocr_adapter.py`
- Modify later: `pipeline/vision_stack/runtime.py`

**Design:**

Use oar-ocr only as an optional geometry provider. Do not require it for runtime. Start with JSON file/subprocess adapter so we can test without compiling Rust inside the Python tests.

Supported inputs:

```text
TRADUZAI_OAR_OCR_BIN
TRADUZAI_OAR_OCR_JSON
```

Supported output:

```python
[
    {
        "text": "...",
        "bbox": [x1, y1, x2, y2],
        "word_boxes": [[x1, y1, x2, y2]],
        "char_boxes": [[x1, y1, x2, y2]],
        "confidence": 0.0,
        "source": "oar-ocr",
    }
]
```

**Step 1: Write parser tests**

Test JSON parsing for both snake_case and camelCase:

```python
def test_parse_oar_ocr_regions_with_word_boxes():
    payload = {
        "text_regions": [
            {
                "text": "ガシャーン",
                "bbox": [10, 20, 80, 50],
                "wordBoxes": [[10, 20, 40, 50], [42, 20, 80, 50]],
            }
        ]
    }

    regions = parse_oar_ocr_payload(payload, width=100, height=100)

    assert regions[0]["word_boxes"] == [[10, 20, 40, 50], [42, 20, 80, 50]]
```

**Step 2: Implement no-op safe loader**

If `TRADUZAI_OAR_OCR_BIN` and `TRADUZAI_OAR_OCR_JSON` are unset, return `[]`.

**Step 3: Add subprocess guard**

If a binary is configured, use timeout and never crash the pipeline:

```text
timeout: 60s
on failure: log warning, return []
```

**Step 4: Run tests**

Run:

```powershell
$env:PYTHONPATH='pipeline'; pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_oar_ocr_adapter.py -q
```

Expected: pass without oar-ocr installed.

---

## Phase 4: Add Optional Hi-SAM Stroke Refinement Backend

**Files:**

- Create: `pipeline/vision_stack/hi_sam_refiner.py`
- Modify: `pipeline/requirements.txt` only if a dependency is already acceptable locally; otherwise document optional install.
- Test: `pipeline/tests/test_hi_sam_refiner.py`

**Design:**

Hi-SAM is a refinement backend, not a full-page stage.

Inputs:

```python
image_rgb
roi_bbox
seed_mask
text_evidence
```

Output:

```python
TextMaskRefinement(mask=np.ndarray, source="hi-sam", confidence=...)
```

Rules:

- Run only inside ROI.
- Do not run on empty pages.
- Do not run if model is unavailable.
- Use OCR/detector boxes as prompts or ROI seeds.
- Use current Manga-Text-Segmentation mask as fallback.

**Step 1: Write unavailable-backend tests**

```python
def test_hi_sam_refiner_returns_none_when_disabled(monkeypatch):
    monkeypatch.setenv("TRADUZAI_HISAM_TEXT_REFINE", "0")
    refiner = HiSamTextRefiner()
    assert refiner.refine(image_rgb, roi_bbox, seed_mask, evidence=[]) is None
```

**Step 2: Write fake-model refinement test**

Mock the model so the test does not download Hi-SAM:

```python
def test_hi_sam_refiner_remaps_crop_mask_to_page(monkeypatch):
    ...
    assert result.mask[35, 45] == 255
    assert result.mask[5, 5] == 0
```

**Step 3: Implement lazy model resolver**

Environment variables:

```text
TRADUZAI_HISAM_TEXT_REFINE=1
TRADUZAI_HISAM_MODEL_PATH=
TRADUZAI_HISAM_DEVICE=cuda|cpu|auto
```

**Step 4: Implement ROI prompt generation**

Initial prompt strategy:

```text
1. If word_boxes/char_boxes exist, use center points.
2. Else if OCR bbox exists, use center of dark connected components inside bbox.
3. Else use seed_mask component centers.
```

**Step 5: Run tests**

Run:

```powershell
$env:PYTHONPATH='pipeline'; pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_hi_sam_refiner.py -q
```

Expected: pass without real Hi-SAM model.

---

## Phase 5: Add Optional CRAFT Coverage Validator

**Files:**

- Create: `pipeline/vision_stack/craft_text_validator.py`
- Test: `pipeline/tests/test_craft_text_validator.py`

**Design:**

CRAFT is a validator/coverage helper. It should not own the final mask.

Outputs:

```python
{
    "char_heatmap": np.ndarray,
    "affinity_heatmap": np.ndarray | None,
    "candidate_boxes": [[x1, y1, x2, y2]],
}
```

Rules:

- If CRAFT is unavailable, return no evidence.
- Accept candidate text regions only near existing OCR/detector/segmenter evidence.
- Use it to flag under-covered characters and to reject far isolated components.

**Step 1: Write fake heatmap test**

```python
def test_craft_validator_reports_undercovered_character_heatmap():
    mask = np.zeros((80, 120), dtype=np.uint8)
    heatmap = np.zeros((80, 120), dtype=np.float32)
    heatmap[30:40, 50:60] = 0.95

    result = measure_craft_coverage(mask, heatmap)

    assert result["undercovered_pixels"] > 0
```

**Step 2: Implement safe backend**

Environment variables:

```text
TRADUZAI_CRAFT_VALIDATE=1
TRADUZAI_CRAFT_MODEL_PATH=
```

**Step 3: Run tests**

Run:

```powershell
$env:PYTHONPATH='pipeline'; pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_craft_text_validator.py -q
```

Expected: pass without real CRAFT model.

---

## Phase 6: Build the OCR-Guided Mask Fusion Layer

**Files:**

- Create: `pipeline/vision_stack/cjk_mask_fusion.py`
- Modify: `pipeline/vision_stack/cjk_segmentation_mask.py`
- Test: `pipeline/tests/test_cjk_mask_fusion.py`

**Confidence tiers:**

```text
HIGH:
  OCR/detector bbox + segmenter overlap + enough dark coverage

MEDIUM:
  detector bbox + segmenter overlap
  or OCR bbox + local dark-core absorption

SFX:
  Manga-Text-Segmentation component connected/near an accepted text component
  plus dark-core absorption

LOW:
  segmenter-only component far from OCR/detector/SFX chain
```

Final rule:

```text
final_mask = HIGH + MEDIUM + SFX
reject LOW unless explicitly enabled for diagnostics
```

**Step 1: Write fusion tests**

Test that OCR-confirmed text is reinforced:

```python
def test_fusion_reinforces_undercovered_ocr_text():
    ...
    assert final_mask[ocr_dark_pixel] == 255
```

Test that far art is rejected:

```python
def test_fusion_rejects_far_segmenter_only_component():
    ...
    assert final_mask[far_art_pixel] == 0
```

Test SFX chain:

```python
def test_fusion_accepts_sfx_chain_near_seed_text():
    ...
    assert final_mask[sfx_pixel] == 255
```

**Step 2: Implement fusion function**

Public function:

```python
def fuse_cjk_text_mask(
    image_rgb: np.ndarray,
    base_mask: np.ndarray,
    evidence: list[TextEvidence],
    *,
    hi_sam_mask: np.ndarray | None = None,
    craft_heatmap: np.ndarray | None = None,
    allow_orphan_sfx: bool = True,
) -> np.ndarray:
    ...
```

**Step 3: Integrate into `build_manga_segmentation_mask`**

Keep existing output when no optional evidence exists.

**Step 4: Run tests**

Run:

```powershell
$env:PYTHONPATH='pipeline'; pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_cjk_mask_fusion.py pipeline\tests\test_cjk_segmentation_mask.py -q
```

Expected: pass.

---

## Phase 7: Integrate with Runtime and Presets

**Files:**

- Modify: `pipeline/vision_stack/runtime.py`
- Modify: `pipeline/vision_stack/engine_presets.py`
- Modify: `pipeline/tests/test_engine_presets.py`
- Modify: `pipeline/tests/test_vision_stack_runtime.py`

**Design:**

Add refinement flags without forcing all models:

```python
EnginePreset(
    id="manga",
    segmenter="manga-text-segmentation-2025",
    ocr="paddle-ocr-vl-1.5",
    inpainter="aot-inpainting",
    mask_strategy="ocr_guided_segmentation",
)
```

Keep aliases:

```text
segmentation_assisted -> current baseline
ocr_guided_segmentation -> new fusion path
roi_segmentation_assisted -> manhwa current baseline
ocr_guided_roi_segmentation -> manhwa/manhua new fusion path
```

**Step 1: Add preset tests**

Add assertions:

```python
def test_manga_can_use_ocr_guided_mask_strategy():
    preset = resolve_engine_preset({"engine_preset_id": "manga_ocr_guided"})
    assert preset.mask_strategy == "ocr_guided_segmentation"
```

**Step 2: Add runtime tests**

Mock optional providers and assert they are called only for the new strategy.

**Step 3: Add environment override**

Support:

```text
TRADUZAI_CJK_MASK_MODE=baseline|ocr_guided
```

Default should remain `baseline` until visual validation is accepted.

**Step 4: Run focused runtime tests**

Run:

```powershell
$env:PYTHONPATH='pipeline'; pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_engine_presets.py pipeline\tests\test_vision_stack_runtime.py -k "engine_preset or cjk or text_segmenter or mask_strategy" -q
```

Expected: pass selected tests.

---

## Phase 8: Visual Validation Matrix

**Files:**

- Modify: `tools/cjk_engine_preset_poc.py`
- Create: `docs/plans/cjk-ocr-guided-validation-notes.md` or append results to this plan after execution.

**Pages to validate:**

```text
1. Japanese manga black/white with SFX
2. Japanese manga with vertical dialogue
3. Korean manhwa tall page
4. Chinese manhua page
5. Dense speedline page
6. White balloon-only page
7. Page with character hair/line-art near text
```

**Metrics:**

```text
changed_outside_input_mask_pixels == 0
OCR text dark coverage >= baseline
SFX residual lower than baseline
manual visual pass: no obvious hair/face/weapon deletion
runtime cost per page recorded
```

**Step 1: Run baseline vs ocr-guided**

Example:

```powershell
$env:PYTHONPATH='pipeline'
$env:TRADUZAI_CJK_MASK_MODE='baseline'
pipeline\venv\Scripts\python.exe tools\cjk_engine_preset_poc.py --image C:\Users\PICHAU\Downloads\japonesmang.png --preset manga --out .codex-tmp\cjk_baseline

$env:TRADUZAI_CJK_MASK_MODE='ocr_guided'
pipeline\venv\Scripts\python.exe tools\cjk_engine_preset_poc.py --image C:\Users\PICHAU\Downloads\japonesmang.png --preset manga --out .codex-tmp\cjk_ocr_guided
```

**Step 2: Compare artifacts**

Inspect:

```text
07_overlay.png
inpaint/*.png
crops/*.png
summary.json
```

**Step 3: Decide default**

Only make `ocr_guided` default if it improves at least:

```text
Japanese manga SFX cleanup
no clear regression on white balloons
no clear regression on manhwa tall page
```

---

## Phase 9: Performance and Fallbacks

**Files:**

- Modify: `pipeline/vision_stack/runtime.py`
- Modify: `pipeline/vision_stack/cjk_mask_fusion.py`
- Test: `pipeline/tests/test_cjk_mask_fusion.py`

**Rules:**

- Hi-SAM runs only on ROIs with poor coverage or ambiguous SFX.
- CRAFT runs only if enabled and only on ROIs.
- oar-ocr runs at most once per page.
- If any optional backend fails, log warning and use baseline mask.

**Step 1: Add timeout tests**

Mock provider failure:

```python
def test_ocr_guided_mask_falls_back_when_hisam_fails():
    ...
    assert np.array_equal(result, baseline_mask)
```

**Step 2: Add timing fields**

Add to page result:

```text
_cjk_mask_mode
_cjk_mask_refiners_used
_cjk_mask_refine_ms
_cjk_mask_fallback_reason
```

**Step 3: Run tests**

Run:

```powershell
$env:PYTHONPATH='pipeline'; pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_cjk_mask_fusion.py pipeline\tests\test_vision_stack_runtime.py -k "cjk or mask" -q
```

Expected: pass selected tests.

---

## Phase 10: Rollout Plan

**Stage 1: Hidden experimental mode**

Enable only by env:

```text
TRADUZAI_CJK_MASK_MODE=ocr_guided
TRADUZAI_HISAM_TEXT_REFINE=1
TRADUZAI_CRAFT_VALIDATE=1
TRADUZAI_OAR_OCR_BIN=...
```

**Stage 2: Preset option in internal engine selector**

Expose:

```text
Manga - Precise Text Mask (experimental)
Manhwa/Manhua - Precise Text Mask (experimental)
```

**Stage 3: Default candidate**

Make it default only after visual matrix passes.

**Rollback:**

Set:

```text
TRADUZAI_CJK_MASK_MODE=baseline
```

Expected: previous Manga-Text-Segmentation-2025 + AOT path is used.

---

## License Notes

Allowed/low-risk candidates:

```text
Hi-SAM: Apache-2.0
CRAFT: MIT
oar-ocr: Apache-2.0
Manga-Text-Segmentation-2025: verify model card before distribution
```

Avoid as production dependency:

```text
MaskTextSpotterV3: CC BY-NC 4.0
comic-text-detector: GPL-3.0 unless kept optional/internal and not distributed
optlab: do not copy code unless license is clarified
```

---

## Final Verification Bundle

Run:

```powershell
pipeline\venv\Scripts\python.exe -m py_compile pipeline\vision_stack\text_mask_evidence.py pipeline\vision_stack\cjk_mask_fusion.py pipeline\vision_stack\hi_sam_refiner.py pipeline\vision_stack\craft_text_validator.py pipeline\vision_stack\oar_ocr_adapter.py pipeline\vision_stack\cjk_segmentation_mask.py pipeline\vision_stack\runtime.py
```

Run:

```powershell
$env:PYTHONPATH='pipeline'; pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_text_mask_evidence.py pipeline\tests\test_oar_ocr_adapter.py pipeline\tests\test_hi_sam_refiner.py pipeline\tests\test_craft_text_validator.py pipeline\tests\test_cjk_mask_fusion.py pipeline\tests\test_cjk_segmentation_mask.py -q
```

Run focused runtime tests:

```powershell
$env:PYTHONPATH='pipeline'; pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_vision_stack_runtime.py -k "cjk or text_segmenter or mask_strategy or aot" -q
```

Manual visual acceptance:

```text
.codex-tmp/cjk_baseline/
.codex-tmp/cjk_ocr_guided/
```

Pass criteria:

```text
1. No text regression in balloons.
2. SFX residual lower than current baseline.
3. No obvious deletion of face/hair/weapon line-art.
4. AOT still changes only inside the input mask.
5. Optional backend failure falls back to baseline.
```

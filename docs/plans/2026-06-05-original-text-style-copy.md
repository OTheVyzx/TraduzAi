# Original Text Style Copy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Detect and carry the original text style into automatic text layers, including font, fill color, outline/stroke, shadow/glow evidence, size, and rotation, without breaking the current conservative default for normal speech bubbles.

**Architecture:** Keep the current Python typesetter as the rendering authority. Add a lightweight style extraction layer before automatic style normalization, store its evidence in each `text_layer`, and change the auto style policy so detected style can opt out of the all-ComicNeue/no-effects default only when confidence gates pass. Use Koharu/YuzuMarker as reference behavior, but do not copy GPL code unless licensing is explicitly accepted.

**Tech Stack:** Python 3.12, NumPy, OpenCV, matplotlib/PIL renderer, unittest/pytest, React/Konva style preview tests.

---

### Task 1: Lock the Current Auto Style Contract

**Files:**
- Modify: `pipeline/tests/test_typesetting_style_policy.py`
- Reference: `pipeline/typesetter/style_policy.py`

**Step 1: Write the current-behavior tests**

Add tests that make the current default explicit:

```python
def test_auto_style_keeps_conservative_default_without_detected_style():
    from typesetter.style_policy import normalize_auto_typesetting_style

    style = normalize_auto_typesetting_style({}, (255, 255, 255))

    assert style["fonte"] == "ComicNeue-Bold.ttf"
    assert style["cor"] == "#000000"
    assert style["contorno"] == ""
    assert style["contorno_px"] == 0
    assert style["sombra"] is False
    assert style["glow"] is False
```

**Step 2: Run test to verify baseline**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; python -m pytest pipeline\tests\test_typesetting_style_policy.py -q
```

Expected: PASS before feature work.

**Step 3: Commit**

```powershell
git add pipeline\tests\test_typesetting_style_policy.py
git commit -m "test: lock automatic typesetting style defaults"
```

---

### Task 2: Add a Style Evidence Schema

**Files:**
- Create: `pipeline/typesetter/style_extractor.py`
- Create: `pipeline/tests/test_style_extractor.py`
- Modify: `pipeline/typesetter/__init__.py` only if imports are needed

**Step 1: Write the failing tests**

Create tests for a small data contract, not ML:

```python
import numpy as np

from typesetter.style_extractor import TextStyleEvidence, extract_text_style_evidence


def test_extracts_black_fill_from_dark_text_on_white_crop():
    crop = np.full((80, 160, 3), 255, dtype=np.uint8)
    crop[25:55, 40:120] = 0

    evidence = extract_text_style_evidence(crop)

    assert isinstance(evidence, TextStyleEvidence)
    assert evidence.text_color == "#000000"
    assert evidence.text_color_confidence >= 0.75
    assert evidence.source == "pixel_analysis"


def test_returns_low_confidence_for_empty_or_flat_crop():
    crop = np.full((80, 160, 3), 255, dtype=np.uint8)

    evidence = extract_text_style_evidence(crop)

    assert evidence.text_color_confidence < 0.5
    assert evidence.stroke_width_px == 0
```

**Step 2: Run test to verify it fails**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; python -m pytest pipeline\tests\test_style_extractor.py -q
```

Expected: FAIL with missing module/function.

**Step 3: Implement the minimal schema and color extraction**

In `pipeline/typesetter/style_extractor.py`, add:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class TextStyleEvidence:
    source: str
    text_color: str
    text_color_confidence: float
    stroke_color: str
    stroke_width_px: int
    stroke_confidence: float
    shadow: bool
    shadow_confidence: float
    glow: bool
    glow_confidence: float
    font_name: str
    font_confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def extract_text_style_evidence(crop_rgb: np.ndarray) -> TextStyleEvidence:
    if crop_rgb is None or crop_rgb.size == 0 or crop_rgb.ndim < 3:
        return _empty()

    gray = cv2.cvtColor(crop_rgb[:, :, :3], cv2.COLOR_RGB2GRAY)
    contrast = float(np.percentile(gray, 95) - np.percentile(gray, 5))
    if contrast < 20.0:
        return _empty()

    dark_mask = gray <= np.percentile(gray, 20)
    if int(np.count_nonzero(dark_mask)) < 8:
        return _empty()

    pixels = crop_rgb[:, :, :3][dark_mask]
    rgb = tuple(int(round(float(v))) for v in np.median(pixels, axis=0))
    confidence = min(1.0, max(0.0, contrast / 96.0))
    return TextStyleEvidence(
        source="pixel_analysis",
        text_color=_hex(rgb),
        text_color_confidence=confidence,
        stroke_color="",
        stroke_width_px=0,
        stroke_confidence=0.0,
        shadow=False,
        shadow_confidence=0.0,
        glow=False,
        glow_confidence=0.0,
        font_name="",
        font_confidence=0.0,
    )


def _empty() -> TextStyleEvidence:
    return TextStyleEvidence(
        source="none",
        text_color="",
        text_color_confidence=0.0,
        stroke_color="",
        stroke_width_px=0,
        stroke_confidence=0.0,
        shadow=False,
        shadow_confidence=0.0,
        glow=False,
        glow_confidence=0.0,
        font_name="",
        font_confidence=0.0,
    )
```

**Step 4: Run test to verify it passes**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; python -m pytest pipeline\tests\test_style_extractor.py -q
```

Expected: PASS.

**Step 5: Commit**

```powershell
git add pipeline\typesetter\style_extractor.py pipeline\tests\test_style_extractor.py
git commit -m "feat: add text style evidence extractor"
```

---

### Task 3: Estimate Stroke/Outline From Source Crop

**Files:**
- Modify: `pipeline/typesetter/style_extractor.py`
- Modify: `pipeline/tests/test_style_extractor.py`

**Step 1: Write the failing tests**

Add synthetic outline tests:

```python
def test_detects_light_text_with_dark_outline():
    crop = np.full((100, 180, 3), 255, dtype=np.uint8)
    crop[30:70, 45:135] = 0
    crop[36:64, 55:125] = 255

    evidence = extract_text_style_evidence(crop)

    assert evidence.text_color == "#FFFFFF"
    assert evidence.stroke_color == "#000000"
    assert evidence.stroke_width_px >= 2
    assert evidence.stroke_confidence >= 0.5
```

**Step 2: Run test to verify it fails**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; python -m pytest pipeline\tests\test_style_extractor.py -q
```

Expected: FAIL because stroke is not detected.

**Step 3: Implement minimal outline detection**

Use foreground clustering and edge bands:
- compute dark and light candidate masks
- choose text fill as the cluster with more interior pixels
- choose stroke as the high-contrast neighboring cluster around the fill mask
- estimate width using distance transform or dilation radius

Keep this conservative:
- only set `stroke_width_px` when contrast between fill/stroke is at least 80 luminance points
- clamp width to `1..8`
- leave stroke empty for ambiguous crops

**Step 4: Run test to verify it passes**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; python -m pytest pipeline\tests\test_style_extractor.py -q
```

Expected: PASS.

**Step 5: Commit**

```powershell
git add pipeline\typesetter\style_extractor.py pipeline\tests\test_style_extractor.py
git commit -m "feat: infer source text outline evidence"
```

---

### Task 4: Connect FontDetector as Optional Evidence

**Files:**
- Modify: `pipeline/typesetter/style_extractor.py`
- Modify: `pipeline/typesetter/font_detector.py` only if a wrapper hook is needed
- Modify: `pipeline/tests/test_style_extractor.py`
- Reference: `pipeline/tests/test_font_detector.py`

**Step 1: Write the failing test**

Do not load the real model in the unit test. Inject a fake detector:

```python
class FakeFontDetector:
    def detect(self, crop, allow_default=True):
        return "KOMIKAX_.ttf"


def test_uses_optional_font_detector_when_available():
    crop = np.full((80, 160, 3), 255, dtype=np.uint8)
    crop[25:55, 40:120] = 0

    evidence = extract_text_style_evidence(crop, font_detector=FakeFontDetector())

    assert evidence.font_name == "KOMIKAX_.ttf"
    assert evidence.font_confidence >= 0.5
```

**Step 2: Run test to verify it fails**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; python -m pytest pipeline\tests\test_style_extractor.py -q
```

Expected: FAIL because the extractor does not accept `font_detector`.

**Step 3: Implement optional detector parameter**

Add `font_detector=None` to `extract_text_style_evidence(...)`.

Rules:
- if detector raises, keep `font_name=""`
- do not force model loading in normal unit tests
- use `allow_default=False` for SFX/impact later, but default to `allow_default=True` for dialogue

**Step 4: Run focused tests**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; python -m pytest pipeline\tests\test_style_extractor.py pipeline\tests\test_font_detector.py -q
```

Expected: PASS.

**Step 5: Commit**

```powershell
git add pipeline\typesetter\style_extractor.py pipeline\tests\test_style_extractor.py
git commit -m "feat: attach optional font evidence to style extraction"
```

---

### Task 5: Preserve Detected Style Through Auto Normalization

**Files:**
- Modify: `pipeline/typesetter/style_policy.py`
- Modify: `pipeline/tests/test_typesetting_style_policy.py`

**Step 1: Write the failing tests**

Add tests for detected style opt-in:

```python
def test_auto_style_preserves_confident_detected_source_style():
    from typesetter.style_policy import normalize_auto_typesetting_style

    style = normalize_auto_typesetting_style(
        {
            "fonte": "KOMIKAX_.ttf",
            "cor": "#FFFFFF",
            "contorno": "#000000",
            "contorno_px": 3,
            "style_origin": "source_detected",
            "style_confidence": 0.82,
        },
        (240, 240, 240),
    )

    assert style["fonte"] == "KOMIKAX_.ttf"
    assert style["cor"] == "#FFFFFF"
    assert style["contorno"] == "#000000"
    assert style["contorno_px"] == 3


def test_auto_style_reverts_low_confidence_detected_style_to_conservative_default():
    from typesetter.style_policy import normalize_auto_typesetting_style

    style = normalize_auto_typesetting_style(
        {
            "fonte": "KOMIKAX_.ttf",
            "cor": "#FFFFFF",
            "contorno": "#000000",
            "contorno_px": 3,
            "style_origin": "source_detected",
            "style_confidence": 0.3,
        },
        (240, 240, 240),
    )

    assert style["fonte"] == "ComicNeue-Bold.ttf"
    assert style["contorno"] == ""
    assert style["contorno_px"] == 0
```

**Step 2: Run test to verify it fails**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; python -m pytest pipeline\tests\test_typesetting_style_policy.py -q
```

Expected: FAIL because all auto style is currently normalized.

**Step 3: Implement confidence-gated preservation**

Add policy:
- `style_origin == "source_detected"` and `style_confidence >= 0.70` preserves source style
- preserve only known-safe fields: `fonte`, `cor`, `contorno`, `contorno_px`, `glow`, `glow_cor`, `glow_px`, `sombra`, `sombra_cor`, `sombra_offset`, `rotacao`
- still enforce valid defaults for missing fields
- keep `force_black_text=True` stronger than source detection for plain white speech balloons unless the layer is `tipo == "sfx"` or `layout_profile != "white_balloon"`

**Step 4: Run tests**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; python -m pytest pipeline\tests\test_typesetting_style_policy.py -q
```

Expected: PASS.

**Step 5: Commit**

```powershell
git add pipeline\typesetter\style_policy.py pipeline\tests\test_typesetting_style_policy.py
git commit -m "feat: preserve confident source-detected text styles"
```

---

### Task 6: Attach Style Evidence When Building Text Layers

**Files:**
- Modify: `pipeline/main.py:4815-4920`
- Modify: `pipeline/tests/test_main_emit.py` or create `pipeline/tests/test_main_style_evidence.py`

**Step 1: Write the failing unit test**

Test `_normalize_text_layer_for_renderer(...)` or the layer-builder helper that receives OCR text. Use a fake image/crop helper if direct crop access is not available.

Expected layer fields:

```python
assert layer["style_origin"] == "source_detected"
assert layer["style_evidence"]["source"] == "pixel_analysis"
assert layer["estilo"]["contorno_px"] == 3
```

**Step 2: Run test to verify it fails**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; python -m pytest pipeline\tests\test_main_emit.py -k style -q
```

Expected: FAIL because no style evidence is attached.

**Step 3: Implement minimal integration**

In the text layer creation path around `pipeline/main.py:4829`:
- crop the source text region from the original page image when available
- call `extract_text_style_evidence(crop_rgb)`
- translate evidence into `estilo` patch
- set `style_origin="source_detected"` only when confidence gates pass
- always store `style_evidence` for debug, even if not applied

Mapping:
- `text_color` -> `estilo["cor"]`
- `stroke_color` -> `estilo["contorno"]`
- `stroke_width_px` -> `estilo["contorno_px"]`
- `font_name` -> `estilo["fonte"]`
- `shadow/glow` remain disabled until Task 7

**Step 4: Run focused tests**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; python -m pytest pipeline\tests\test_main_emit.py -k style -q
```

Expected: PASS.

**Step 5: Commit**

```powershell
git add pipeline\main.py pipeline\tests\test_main_emit.py
git commit -m "feat: attach source style evidence to automatic text layers"
```

---

### Task 7: Add Shadow/Glow Evidence Conservatively

**Files:**
- Modify: `pipeline/typesetter/style_extractor.py`
- Modify: `pipeline/tests/test_style_extractor.py`
- Modify: `pipeline/typesetter/style_policy.py`

**Step 1: Write the failing synthetic tests**

Use synthetic crops:
- black text with offset gray duplicate behind it -> shadow
- white/yellow text with blurred bright halo -> glow

Assertions:

```python
assert evidence.shadow is True
assert evidence.shadow_confidence >= 0.6
```

```python
assert evidence.glow is True
assert evidence.glow_confidence >= 0.6
```

**Step 2: Run test to verify it fails**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; python -m pytest pipeline\tests\test_style_extractor.py -q
```

Expected: FAIL.

**Step 3: Implement conservative detection**

Rules:
- shadow requires a coherent lower/right or directional offset region with lower contrast than text
- glow requires a low-frequency halo around text mask with high contrast against background
- never enable both if confidence is ambiguous
- clamp `sombra_offset` to `[-12, 12]`
- clamp `glow_px` to `[1, 8]`

**Step 4: Run tests**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; python -m pytest pipeline\tests\test_style_extractor.py pipeline\tests\test_typesetting_style_policy.py -q
```

Expected: PASS.

**Step 5: Commit**

```powershell
git add pipeline\typesetter\style_extractor.py pipeline\typesetter\style_policy.py pipeline\tests\test_style_extractor.py
git commit -m "feat: infer conservative shadow and glow style evidence"
```

---

### Task 8: Verify Renderer Uses Detected Effects Without Regressions

**Files:**
- Modify: `pipeline/tests/test_typesetting_renderer.py`
- Reference: `pipeline/typesetter/renderer.py:9560-9631`

**Step 1: Write or extend renderer test**

Add a small render test with a layer style:

```python
style = {
    "fonte": "ComicNeue-Bold.ttf",
    "cor": "#FFFFFF",
    "contorno": "#000000",
    "contorno_px": 3,
    "sombra": True,
    "sombra_cor": "#333333",
    "sombra_offset": [3, 4],
    "glow": False,
    "glow_cor": "",
    "glow_px": 0,
}
```

Assert rendered pixels include:
- main text color
- outline-like dark pixels around text
- offset shadow pixels
- non-empty `render_bbox`

**Step 2: Run test to verify baseline**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; python -m pytest pipeline\tests\test_typesetting_renderer.py -k style -q
```

Expected: PASS if renderer already supports it, FAIL only if tests expose a real renderer gap.

**Step 3: Patch renderer only if necessary**

If the test fails:
- fix the smallest issue in `pipeline/typesetter/renderer.py`
- do not refactor layout fitting
- preserve `render_bbox`, `safe_text_box`, and QA flags

**Step 4: Run focused renderer suite**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; python -m pytest pipeline\tests\test_typesetting_renderer.py pipeline\tests\test_typeset_render_plan_debug.py -q
```

Expected: PASS.

**Step 5: Commit**

```powershell
git add pipeline\typesetter\renderer.py pipeline\tests\test_typesetting_renderer.py
git commit -m "test: verify rendered source-detected text effects"
```

---

### Task 9: Preserve Style Through Project/UI Hydration

**Files:**
- Modify: `src/lib/tauri.ts` if hydration drops new evidence
- Modify: `src/lib/editorTextStylePolicy.ts`
- Modify: `src/lib/__tests__/tauriHydration.test.ts`
- Modify: `src/lib/__tests__/editorTextStylePolicy.test.ts`

**Step 1: Write failing tests**

Ensure source-detected style is not treated as legacy default:

```ts
it("preserves source-detected outline style during hydration", () => {
  const style = canonicalizeTextStyle(
    {
      fonte: "KOMIKAX_.ttf",
      cor: "#FFFFFF",
      contorno: "#000000",
      contorno_px: 3,
      glow: false,
      sombra: true,
      sombra_cor: "#333333",
      sombra_offset: [3, 4],
      style_origin: "source_detected",
    },
    { mode: "hydrate" },
  );

  expect(style.contorno).toBe("#000000");
  expect(style.contorno_px).toBe(3);
  expect(style.sombra).toBe(true);
});
```

**Step 2: Run test to verify it fails or passes**

Run:

```powershell
npm test -- --run src/lib/__tests__/editorTextStylePolicy.test.ts src/lib/__tests__/tauriHydration.test.ts
```

Expected: FAIL if hydration currently strips the new style.

**Step 3: Implement hydration preservation**

Rules:
- preserve `style_origin`, `style_evidence`, and known style effect fields
- keep old legacy normalization only for old white-outline defaults without source detection
- do not expose `style_evidence` as an editable UI field yet

**Step 4: Run tests**

Run:

```powershell
npm test -- --run src/lib/__tests__/editorTextStylePolicy.test.ts src/lib/__tests__/tauriHydration.test.ts
```

Expected: PASS.

**Step 5: Commit**

```powershell
git add src\lib\editorTextStylePolicy.ts src\lib\__tests__\editorTextStylePolicy.test.ts src\lib\__tests__\tauriHydration.test.ts
git commit -m "feat: preserve source-detected text styles in editor hydration"
```

---

### Task 10: Add Debug Artifacts for Review

**Files:**
- Modify: `pipeline/main.py`
- Modify: `pipeline/debug_tools/recorder.py` only if a new helper is needed
- Modify: `pipeline/tools/render_vision_debug_sheet.py` optional
- Test: `pipeline/tests/test_render_plan_trace_integrity.py`

**Step 1: Write failing artifact test**

Assert debug output includes:
- `style_evidence`
- `style_origin`
- final applied style
- confidence values

Prefer JSONL near existing `09_typeset/render_plan_raw.jsonl` or `09_typeset/render_plan_final.jsonl`.

**Step 2: Run test to verify it fails**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; python -m pytest pipeline\tests\test_render_plan_trace_integrity.py -q
```

Expected: FAIL for missing fields.

**Step 3: Implement debug fields**

Add fields without breaking existing readers:

```json
{
  "style_origin": "source_detected",
  "style_confidence": 0.82,
  "style_evidence": {
    "source": "pixel_analysis",
    "text_color": "#FFFFFF",
    "stroke_color": "#000000",
    "stroke_width_px": 3
  }
}
```

**Step 4: Run tests**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; python -m pytest pipeline\tests\test_render_plan_trace_integrity.py -q
```

Expected: PASS.

**Step 5: Commit**

```powershell
git add pipeline\main.py pipeline\tests\test_render_plan_trace_integrity.py
git commit -m "feat: record source text style evidence in render debug"
```

---

### Task 11: Real Chapter Validation Gate

**Files:**
- Modify: `context.md` after validation
- No code changes unless a defect is found

**Step 1: Run focused Python tests**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; python -m pytest pipeline\tests\test_style_extractor.py pipeline\tests\test_typesetting_style_policy.py pipeline\tests\test_font_detector.py pipeline\tests\test_typesetting_renderer.py -q
```

Expected: PASS.

**Step 2: Run integration regression**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; python -m pytest pipeline\tests\test_main_emit.py pipeline\tests\test_render_plan_trace_integrity.py -q
```

Expected: PASS.

**Step 3: Run one real image/page through the pipeline**

Use an existing local config or create a temporary single-page config from a known fixture. Run:

```powershell
cd pipeline
python main.py config.json
```

Expected:
- output page renders
- `project.json` text layers include `style_evidence`
- normal speech bubbles still default to readable ComicNeue/no-effect unless evidence is strong
- SFX or styled text can receive font/outline/color style

**Step 4: Inspect debug artifacts**

Check:
- `debug/.../09_typeset/render_plan_raw.jsonl`
- `debug/.../09_typeset/render_plan_final.jsonl`
- final rendered image crop

Expected:
- no geometry regression
- no unreadable white-on-white or black-on-black text
- style confidence explains every applied effect

**Step 5: Document validation**

Append a short note to `context.md`:
- command run
- page/chapter used
- tests passed
- known limitations

**Step 6: Commit**

```powershell
git add context.md
git commit -m "docs: record source style copy validation"
```

---

### Task 12: Optional Koharu/YuzuMarker Follow-Up

**Files:**
- Create or modify later only after Task 11 passes:
  - `pipeline/typesetter/koharu_style_adapter.py`
  - `pipeline/tests/test_koharu_style_adapter.py`
  - `docs/THIRD_PARTY_NOTICES.md`

**Step 1: Decide licensing boundary**

Do not copy Koharu code into TraduzAI unless the GPL boundary is explicitly accepted. If not accepted, use it as reference only.

**Step 2: Build adapter tests first**

Adapter contract:

```python
def test_maps_koharu_font_prediction_to_traduzai_style():
    prediction = {
        "text_color": [255, 255, 255],
        "stroke_color": [0, 0, 0],
        "stroke_width_px": 3.0,
        "font_size_px": 42.0,
    }

    style = map_koharu_prediction_to_traduzai_style(prediction)

    assert style["cor"] == "#FFFFFF"
    assert style["contorno"] == "#000000"
    assert style["contorno_px"] == 3
```

**Step 3: Implement adapter only if needed**

Use this after the simple extractor proves useful but insufficient on real pages.

**Step 4: Commit**

```powershell
git add pipeline\typesetter\koharu_style_adapter.py pipeline\tests\test_koharu_style_adapter.py docs\THIRD_PARTY_NOTICES.md
git commit -m "feat: add optional Koharu style prediction adapter"
```

---

## Rollback Strategy

- Feature is gated by `style_origin == "source_detected"` plus confidence.
- If visual quality drops, revert only the integration in `pipeline/main.py` and leave `style_extractor.py` tests/helpers intact.
- Keep `normalize_auto_typesetting_style()` conservative for all low-confidence or missing-evidence layers.
- Do not replace `pipeline/typesetter/renderer.py` in this feature unless Task 8 exposes a real rendering bug.

## Definition of Done

- Python tests pass for style extraction, style policy, renderer effects, and layer integration.
- Frontend hydration preserves source-detected styles.
- At least one real-page validation shows style evidence in debug artifacts and visually acceptable output.
- Normal dialogue bubbles remain readable and do not get noisy effects by default.
- No GPL code copied from Koharu unless licensing decision is documented.

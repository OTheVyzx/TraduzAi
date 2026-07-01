# Curved Text Rendering Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Detect curved text from source crops and render translated text along a simple arc, matching the curved examples in `C:/Users/PICHAU/Downloads/testar detector de fonte.png`.

**Architecture:** Add curved-text evidence as an additive style field, then promote it through `project.json` style metadata only when confidence is high. Implement a conservative renderer path for arc text using existing local FreeType/PIL-safe rendering primitives, with a fallback to current straight rendering when the arc is weak or unsupported.

**Tech Stack:** Python 3.12, OpenCV, NumPy, existing `pipeline/typesetter/renderer.py`, existing `pipeline/typesetter/style_extractor.py`, React/Konva editor preview follow-up.

---

### Task 1: Add Curvature Evidence Schema

**Files:**
- Modify: `pipeline/typesetter/style_extractor.py`
- Test: `pipeline/tests/test_style_extractor.py`

**Step 1: Write the failing tests**

Add tests near the existing style evidence serialization tests:

```python
def test_text_style_evidence_serializes_curvature_fields():
    evidence = TextStyleEvidence(
        source="pixel_analysis",
        text_color="#000000",
        text_color_confidence=0.9,
        stroke_color="",
        stroke_width_px=0,
        stroke_confidence=0.0,
        shadow=False,
        shadow_confidence=0.0,
        glow=False,
        glow_confidence=0.0,
        font_name="ComicNeue-Bold.ttf",
        font_confidence=1.0,
        curved=True,
        curve_direction="arc_up",
        curve_amount=0.42,
        curve_confidence=0.88,
    )

    data = evidence.to_dict()

    assert data["curved"] is True
    assert data["curve_direction"] == "arc_up"
    assert data["curve_amount"] == 0.42
    assert data["curve_confidence"] == 0.88
```

**Step 2: Run the test to verify it fails**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'
python -m pytest pipeline\tests\test_style_extractor.py::test_text_style_evidence_serializes_curvature_fields -q
```

Expected: FAIL because `TextStyleEvidence` does not yet accept curvature fields.

**Step 3: Implement minimal schema**

Add optional fields to `TextStyleEvidence`:

```python
curved: bool = False
curve_direction: str = ""
curve_amount: float = 0.0
curve_confidence: float = 0.0
```

**Step 4: Run test to verify it passes**

Run the same pytest command.

Expected: PASS.

---

### Task 2: Detect Simple Arc Curvature From Text Masks

**Files:**
- Modify: `pipeline/typesetter/style_extractor.py`
- Test: `pipeline/tests/test_style_extractor.py`

**Step 1: Write failing tests**

Create synthetic masks similar to the bottom curved examples:

```python
def test_detects_arc_up_text_baseline():
    crop = np.full((160, 360, 3), [138, 0, 247], dtype=np.uint8)
    for x in range(50, 310, 8):
        t = (x - 180) / 130.0
        y = int(92 - 28 * (1.0 - t * t))
        crop[y : y + 18, x : x + 6] = 0

    evidence = extract_text_style_evidence(crop)

    assert evidence.curved is True
    assert evidence.curve_direction == "arc_up"
    assert evidence.curve_confidence >= 0.65
    assert evidence.curve_amount > 0.15
```

```python
def test_straight_text_does_not_enable_curvature():
    crop = np.full((120, 320, 3), 255, dtype=np.uint8)
    crop[52:72, 40:280] = 0

    evidence = extract_text_style_evidence(crop)

    assert evidence.curved is False
    assert evidence.curve_confidence == 0.0
```

**Step 2: Run tests to verify failure**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'
python -m pytest pipeline\tests\test_style_extractor.py -k "arc_up_text_baseline or straight_text_does_not_enable_curvature" -q
```

Expected: FAIL because no detection exists.

**Step 3: Implement curvature detector**

Add `_detect_curve_details(rgb_crop, gray, evidence)`:

- Build a foreground mask from existing evidence using `_foreground_mask_for_evidence`.
- Split the foreground into x-bins, e.g. 12 bins.
- For each bin, compute median y of foreground pixels.
- Fit both a straight line and a quadratic using `np.polyfit`.
- Compute improvement: `line_rmse - quad_rmse`.
- Compute normalized curve amount: `abs(quadratic_y_center - mean(edge_y)) / crop_height`.
- Direction:
  - `arc_up` when center y is above edges.
  - `arc_down` when center y is below edges.
- Require:
  - at least 6 valid bins,
  - curve amount >= `0.10`,
  - quadratic improves RMSE by at least `3.0 px`,
  - text bbox width >= `80 px`.

Return:

```python
{"curved": True, "direction": "arc_up", "amount": 0.32, "confidence": 0.78}
```

**Step 4: Wire detector in `_with_effect_evidence`**

In `_with_effect_evidence`, call `_detect_curve_details` and add the values via `replace(...)`.

**Step 5: Run tests**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'
python -m pytest pipeline\tests\test_style_extractor.py -q
```

Expected: PASS.

---

### Task 3: Promote Curvature Through Pipeline Style

**Files:**
- Modify: `pipeline/main.py`
- Test: `pipeline/tests/test_main_emit.py`

**Step 1: Write failing test**

Add a focused test near the existing `style_evidence` tests:

```python
def test_style_from_evidence_promotes_curved_text_metadata():
    base = {"fonte": "ComicNeue-Bold.ttf", "cor": "#000000"}
    evidence = {
        "source": "pixel_analysis",
        "text_color": "#000000",
        "text_color_confidence": 0.9,
        "curved": True,
        "curve_direction": "arc_up",
        "curve_amount": 0.35,
        "curve_confidence": 0.82,
    }

    style, origin, confidence, source = main._style_from_evidence(base, evidence)

    assert origin == "source_detected"
    assert style["curva"] is True
    assert style["curva_direcao"] == "arc_up"
    assert style["curva_intensidade"] == 0.35
```

**Step 2: Run test to verify failure**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'
python -m pytest pipeline\tests\test_main_emit.py -k curved_text_metadata -q
```

Expected: FAIL because `_style_from_evidence` ignores curvature.

**Step 3: Update confidence aggregation**

In `_style_evidence_confidence`, include `curve_confidence` when `curved is True`.

**Step 4: Map evidence to style**

In `_style_from_evidence`, add:

```python
if evidence.get("curved") is True and float(evidence.get("curve_confidence") or 0.0) >= SOURCE_STYLE_CONFIDENCE_THRESHOLD:
    style["curva"] = True
    style["curva_direcao"] = evidence.get("curve_direction") or "arc_up"
    style["curva_intensidade"] = float(evidence.get("curve_amount") or 0.0)
```

**Step 5: Run test**

Expected: PASS.

---

### Task 4: Add Renderer Contract Defaults

**Files:**
- Modify: `pipeline/typesetter/renderer.py`
- Test: `pipeline/tests/test_typesetting_renderer.py`

**Step 1: Write failing test for canonical style**

Add a small test that calls the renderer style canonicalization path used by existing tests, or inspect render plan output if direct import is unstable:

```python
def test_render_plan_preserves_curve_style_fields():
    text_data = {
        "text": "OLA TUDO BEM",
        "translated": "OLA TUDO BEM",
        "bbox": [20, 20, 280, 120],
        "tipo": "sfx",
        "estilo": {
            "fonte": "KOMIKAX_.ttf",
            "cor": "#000000",
            "curva": True,
            "curva_direcao": "arc_up",
            "curva_intensidade": 0.35,
        },
    }
    plan = renderer._build_render_plan_for_test(text_data)
    assert plan["curva"] is True
    assert plan["curva_direcao"] == "arc_up"
```

If no helper exists, add a test around the smallest public renderer function already used in `test_typesetting_renderer.py`.

**Step 2: Run test to verify failure**

Expected: FAIL because render plan does not carry curve fields.

**Step 3: Add defaults**

In `_canonical_render_style`, set:

```python
style.setdefault("curva", False)
style.setdefault("curva_direcao", "")
style.setdefault("curva_intensidade", 0.0)
```

In render plan construction, carry:

```python
"curva": bool(estilo.get("curva", False)),
"curva_direcao": str(estilo.get("curva_direcao", "")),
"curva_intensidade": float(estilo.get("curva_intensidade", 0.0) or 0.0),
```

**Step 4: Run test**

Expected: PASS.

---

### Task 5: Implement Arc Text Rendering Path

**Files:**
- Modify: `pipeline/typesetter/renderer.py`
- Test: `pipeline/tests/test_typesetting_renderer.py`

**Step 1: Write failing visual/behavioral test**

Add a renderer test that renders a short line with `curva=True` and asserts:

- output is non-empty,
- text alpha bbox is taller than straight text,
- top/bottom y positions vary across x bands.

Pseudo-test:

```python
def test_curved_text_render_produces_arc_baseline(tmp_path):
    image = np.full((180, 360, 3), 255, dtype=np.uint8)
    layer = {
        "text": "OLA TUDO BEM",
        "translated": "OLA TUDO BEM",
        "bbox": [30, 40, 330, 140],
        "tipo": "sfx",
        "estilo": {
            "fonte": "KOMIKAX_.ttf",
            "cor": "#000000",
            "curva": True,
            "curva_direcao": "arc_up",
            "curva_intensidade": 0.35,
        },
    }

    rendered = renderer.render_text_layers(image, [layer])

    assert np.any(rendered != image)
    assert _ink_y_median_by_x_band(rendered) shows center above edges
```

**Step 2: Run test to verify failure**

Expected: FAIL because curved renderer does not exist.

**Step 3: Implement minimal arc renderer**

Add a dedicated helper near the existing render path:

```python
def _render_arc_text_layer(image_np, plan, text, font, positions):
    ...
```

Minimal implementation:

- Only support single-line text initially.
- Render each character/glyph or small chunks to transparent layers using existing font draw helpers.
- Compute cumulative x positions along text width.
- For each glyph center x, compute normalized `t` in `[-1, 1]`.
- Arc offset:
  - `arc_up`: `y_offset = -curve_px * (1 - t*t)`
  - `arc_down`: `y_offset = curve_px * (1 - t*t)`
- Local tangent angle:
  - approximate derivative and rotate each glyph by angle.
- Composite glyph layers onto image.
- Respect existing fill/stroke/glow/shadow as much as possible; if too risky, first version supports fill + outline and leaves glow/shadow to a follow-up.

**Step 4: Gate use conservatively**

Use arc rendering only when:

```python
plan["curva"] is True
abs(plan["curva_intensidade"]) >= 0.10
len(lines) == 1
rotation_deg == 0
```

Fallback to current renderer otherwise.

**Step 5: Run tests**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'
python -m pytest pipeline\tests\test_typesetting_renderer.py -k curved -q
```

Expected: PASS.

---

### Task 6: Add Visual Audit For Curved Text

**Files:**
- Modify: `pipeline/debug_tools/style_audit_report.py`
- Test: manual command on existing run and synthetic image

**Step 1: Add curved fields to report cards**

In `_make_card`, append:

```python
if rec.get("curved"):
    effects.append(f"curve {rec.get('curve_direction')}:{float(rec.get('curve_amount') or 0):.2f}")
```

**Step 2: Add curved contact sheet**

Add:

```python
curved = [rec for rec in records if rec.get("curved")]
"07_curved.jpg": curved[:45],
```

**Step 3: Add summary count**

Increment:

```python
if rec.get("curved"):
    counts["curved"] += 1
```

**Step 4: Run report**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'
python pipeline\debug_tools\style_audit_report.py --run 'N:\TraduzAI\DEBUGM\runs\2026-05-23_mihon_matrix20_v33\18_regressed_items_ch3'
```

Expected:

- `style_audit_summary.json` includes `curved`.
- `07_curved.jpg` exists.

---

### Task 7: Validate Against `testar detector de fonte.png`

**Files:**
- No production file changes unless bugs are found.
- Optional test fixture later: `pipeline/tests/fixtures/style_detector/curved_text_sample.png`

**Step 1: Run manual crop script**

Use the bottom curved boxes from the user image:

```python
boxes = {
    "curved_left": [40, 825, 460, 930],
    "curved_bottom_right": [580, 955, 950, 1070],
    "rotated_not_curved": [630, 655, 980, 810],
}
```

Run extractor and print `curved`, `curve_direction`, `curve_amount`, `curve_confidence`.

**Step 2: Expected result**

- `curved_left`: `curved=True`, likely `arc_up`.
- `curved_bottom_right`: `curved=True`, likely `arc_down` or `arc_up` depending crop orientation.
- `rotated_not_curved`: should be `curved=False` or low confidence.

**Step 3: Tune thresholds**

Only tune thresholds if:

- straight rotated text is falsely marked curved, or
- obvious curved text is missed.

Do not tune to one crop if it regresses synthetic tests.

---

### Task 8: Editor Contract Follow-Up

**Files:**
- Modify: `src/lib/stores/appStore.ts`
- Modify: `src/lib/tauri.ts`
- Modify: `src/lib/editorTextStylePolicy.ts`
- Modify: `src/components/editor/toolbar/TypesettingBar.tsx`
- Test: relevant frontend tests under `src/lib/__tests__/`

**Step 1: Add optional style fields**

Extend `TextStyle`/`estilo` typing with:

```ts
curva?: boolean;
curva_direcao?: "arc_up" | "arc_down" | "";
curva_intensidade?: number;
```

**Step 2: Hydration**

Ensure `hydrateTextLayer` preserves these fields from `project.json`.

**Step 3: Policy**

Ensure canonicalization does not strip source-detected curve fields.

**Step 4: UI**

Add a small disabled/readonly indicator first: `Curva detectada`.

Do not add manual curve editing until renderer output is verified.

---

### Task 9: Final Validation

**Files:**
- All touched files.

**Step 1: Run focused Python tests**

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'
python -m pytest pipeline\tests\test_style_extractor.py pipeline\tests\test_main_emit.py pipeline\tests\test_typesetting_renderer.py -k "style or curved or curve or render_plan" -q
```

**Step 2: Run frontend tests if Task 8 was implemented**

```powershell
npm test -- --run src/lib/__tests__/tauriHydration.test.ts src/lib/__tests__/editorTextStylePolicy.test.ts
```

**Step 3: Generate visual audit**

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'
python pipeline\debug_tools\style_audit_report.py --run 'N:\TraduzAI\DEBUGM\runs\2026-05-23_mihon_matrix20_v33\18_regressed_items_ch3'
```

**Step 4: Inspect**

Open:

```text
N:\TraduzAI\DEBUGM\runs\2026-05-23_mihon_matrix20_v33\18_regressed_items_ch3\debug\codex_style_audit\visual_report\index.html
```

Expected:

- curved report sheet exists,
- obvious curved texts are detected,
- straight rotated texts are not over-detected,
- renderer does not clip arc text.

---

## Implementation Notes

- Keep this additive. Existing `rotacao` must continue working.
- Do not add neural models for this first pass.
- Do not attempt arbitrary Bézier text. Start with one-line circular/parabolic arc.
- Do not apply curve rendering to normal white balloon text by default.
- Promote curvature only with confidence >= `SOURCE_STYLE_CONFIDENCE_THRESHOLD`.
- Fallback must always be current straight renderer.


# Bubble Shape Refiner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refine derived BubbleMask shapes so oval/rectangular balloon bodies are cleaned without cutting valid tails, points, and irregular speech-balloon edges.

**Architecture:** Add a small geometry refiner that receives a binary bubble mask component and returns a refined mask plus diagnostics. The refiner classifies components as rectangular, simple oval, or irregular/extended; it removes connected white noise only when it falls outside the accepted body/extension model. Existing derivation in `process_bands.py` and `mask_builder.py` calls the refiner after white-component extraction and before persisting `bubble_mask_source`.

**Tech Stack:** Python 3.12, OpenCV, NumPy, pytest.

---

### Task 1: Geometry Refiner Unit Tests

**Files:**
- Create: `pipeline/tests/test_bubble_shape_refiner.py`
- Create: `pipeline/vision_stack/bubble_shape_refiner.py`

- [ ] **Step 1: Write failing tests**

Create tests for:
- oval balloon with a thick connected white block above it: block must be removed and oval body kept;
- irregular/pointed balloon with thin tail/points: tail and points must be preserved;
- rectangular balloon with a side protrusion: protrusion must be removed while rectangle remains;
- simple oval must remain close to the original mask.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; python -m pytest pipeline\tests\test_bubble_shape_refiner.py -q
```

Expected: tests fail because `vision_stack.bubble_shape_refiner` does not exist or behavior is not implemented.

- [ ] **Step 3: Implement minimal refiner**

Implement:
- `refine_bubble_shape_mask(mask, prefer_shape=None) -> ShapeRefineResult`
- `ShapeRefineResult.mask`
- `ShapeRefineResult.shape_kind`
- `ShapeRefineResult.removed_pixels`
- `ShapeRefineResult.added_pixels`
- `ShapeRefineResult.accepted`

Keep logic conservative:
- rectangular: robust inner rectangle from row/column coverage;
- oval simple: ellipse fit only when ellipse compatibility is high;
- irregular: keep original contour and remove only disconnected or thick off-body blobs.

- [ ] **Step 4: Run tests and verify GREEN**

Run the same pytest command. Expected: all tests pass.

### Task 2: Integrate With Derived BubbleMask

**Files:**
- Modify: `pipeline/strip/process_bands.py`
- Modify: `pipeline/inpainter/mask_builder.py`
- Test: `pipeline/tests/test_strip_process_bands.py`
- Test: `pipeline/tests/test_mask_builder.py`

- [ ] **Step 1: Write focused failing tests**

Add tests that prove:
- a derived oval mask with connected top noise becomes `derived_white_crop` without the noise;
- a pointed/irregular balloon does not get forced into an ellipse;
- a rectangular balloon keeps rectangular body but drops side protrusion.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; python -m pytest pipeline\tests\test_strip_process_bands.py pipeline\tests\test_mask_builder.py -k "shape_refiner or connected_noise or pointed" -q
```

- [ ] **Step 3: Apply refiner after white component extraction**

In both derivation paths, call `refine_bubble_shape_mask(local_mask)` after `_fill_binary_holes(local_mask)` and before `_classify_derived_bubble_mask(...)`. Persist diagnostics under `qa_metrics.derived_shape_refiner` where available.

- [ ] **Step 4: Run focused and regression tests**

Run:

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'; python -m pytest pipeline\tests\test_bubble_shape_refiner.py pipeline\tests\test_strip_process_bands.py pipeline\tests\test_mask_builder.py pipeline\tests\test_mask_chain_debug.py -q
```

### Task 3: Visual Validation

**Files:**
- No code edits.
- Output only under `DEBUGM/runs/`.

- [ ] **Step 1: Rerun One Second chapter 1 with debug**

Run:

```powershell
$out='N:\TraduzAI\DEBUGM\runs\one_second_ch1_shape_refiner_20260603'
if (Test-Path $out) { Remove-Item -LiteralPath $out -Recurse -Force }
$env:PYTHONPATH='N:\TraduzAI\pipeline'
python N:\TraduzAI\pipeline\main.py --input 'D:\Mihon pra pc\downloads\mangas\Manhwatop (EN)\1 Second\Chapter 1.cbz' --work '1 Second' --target pt-BR --mode real --output $out --debug --export-mode debug
```

- [ ] **Step 2: Inspect target masks**

Inspect:
- `page_005_band_007`
- `page_006_band_008`
- `page_007_band_013`
- `page_008_band_014`

Expected:
- top pointed balloon keeps points/tail;
- lower oval drops connected white block;
- rectangular card drops protrusion but stays rectangular;
- masks do not regress to bbox fallback.

# CJK Engine Presets and Mask Routing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add CJK engine presets for Manga and Manhwa/Manhua with separate detector/mask strategies: Manga uses the Anime Text YOLO stack, while Manhwa/Manhua uses Comic Text & Bubble Detection plus crop/ROI segmentation.

**Architecture:** Project presets select an `engine_preset_id`, and the Python runtime resolves that id into per-stage engines and a mask strategy. Manga uses Anime Text YOLO (N), YuzuMarker, Manga-Text-Segmentation-2025, Speech Bubble Segmentation, PaddleOCR-VL, and AOT. Manhwa/Manhua uses Comic Text & Bubble Detection as the primary detector, PaddleOCR-VL as OCR, AOT as inpaint, and runs Manga-Text-Segmentation-2025 plus Speech Bubble Segmentation only on detector-derived crops/ROIs for mask refinement.

**Tech Stack:** React/TypeScript project presets, Tauri pipeline config, Python 3.12 runtime routing, OpenCV/NumPy mask builders, existing `pipeline/vision_stack/runtime.py`, `pipeline/vision_stack/detector.py`, `pipeline/typesetter/font_detector.py`, `pipeline/vision_stack/ocr.py`, and `pipeline/inpainter/`.

## Execution Status

- Completed: project preset metadata, frontend setup chip, Tauri/Rust/Python config propagation, Python preset resolver, runtime detect/OCR routing, Koharu HTTP/worker preset telemetry, CJK mask strategy helpers, Manhwa/Manhua crop/ROI mask validation, page-level `vision_engine` metadata in `project.json`, the visual validation CLI, and source-native AOT inpainting selection/runtime.
- AOT behavior: CJK presets now select `aot-inpainting` for inpaint. The Python runtime loads the local `mayocream/aot-inpainting` `config.json` + `model.safetensors` bundle when `TRADUZAI_AOT_INPAINT=1` is set. If AOT is selected but disabled or unavailable, the pipeline raises a clear error instead of silently falling back to LaMA/Flux.
- Manga-Text-Segmentation-2025 behavior: both CJK presets now select `manga-text-segmentation-2025` as the fine text segmenter. The runtime loads `a-b-c-x-y-z/Manga-Text-Segmentation-2025` from local Hugging Face models, runs it by crop/ROI, remaps the local mask back to the page, rejects unsafe broad masks, and falls back to text geometry when the model/dependencies are unavailable.
- Preserve behavior: CJK masks now skip blocks whose matched OCR text is marked `skip_processing=True` or `preserve_original=True`, so preserved SFX are not inpainted via detector-only `_vision_blocks`.
- Known validation note: the full `test_vision_stack_runtime.py` suite still depends on local fixture images `009__001.jpg`, `010__001.jpg`, and `012__001.jpg`; focused tests that do not depend on those images pass.

---

## Decision

### Preset 1: Manga

Use `engine_preset_id = "manga"` for Japanese manga pages, especially black-and-white pages and dense SFX pages.

```text
Detector: Anime Text YOLO (N)
Font Detector: YuzuMarker Font Detection
Segmenter: Manga-Text-Segmentation-2025
Bubble Segmenter: Speech Bubble Segmentation
OCR: PaddleOCR-VL
Inpainter: AOT Inpainting
Mask strategy: segmentation-assisted glyph/text mask, clipped by bubble only when safe
```

### Preset 2: Manhwa/Manhua

Use `engine_preset_id = "manhwa_manhua"` for Korean/Chinese vertical or color pages.

```text
Detector: Comic Text & Bubble Detection
Font Detector: default/none unless later testing proves YuzuMarker helps
Segmenter: Manga-Text-Segmentation-2025, crop/ROI only
Bubble Segmenter: Speech Bubble Segmentation, crop/ROI only
OCR: PaddleOCR-VL
Inpainter: AOT Inpainting
Mask strategy: crop/ROI segmentation refined by PaddleOCR-VL text geometry
```

Important: Manhwa/Manhua segmentation must not run on the whole tall page by default. The earlier bad result happened on a full page taller than 7000 px. For this preset, segmentation is part of the pipeline, but only after `Comic Text & Bubble Detection` defines smaller crops/ROIs. Local crop masks are remapped back to full-page coordinates.

### Explicitly Not Doing

- No Flux path.
- No API/server phase in this plan.
- No full-balloon erase mask.
- No broad bbox mask unless every safer text-pixel fallback failed.
- No production dependency on launching Koharu as the pipeline. Koharu engine ids can be used as reference names, but the target implementation is inside TraduzAI source.

---

## Implementation Flags

- `TRADUZAI_ENGINE_PRESET=manga|manhwa_manhua|auto`: force or auto-select the CJK engine preset.
- `TRADUZAI_CJK_PRESET_DEBUG=1`: write preset routing and mask artifacts.
- `TRADUZAI_PADDLEOCR_VL=1`: enable PaddleOCR-VL route.
- `TRADUZAI_AOT_INPAINT=1`: enable AOT inpainting route.
- `TRADUZAI_MANGA_TEXT_SEG_MAX_SIDE=1536`: max side used when running Manga-Text-Segmentation-2025 on a crop/ROI.
- `TRADUZAI_MANGA_TEXT_SEG_THRESHOLD=0.5`: binarization threshold for Manga-Text-Segmentation-2025.
- `TRADUZAI_MANHWA_ROI_SEGMENTATION=1`: legacy alias for the Manhwa/Manhua crop/ROI segmentation path.

Legacy flags such as `TRADUZAI_MANGA_MASK_STACK`, `TRADUZAI_PADDLE_TEXT_MASK`, and `TRADUZAI_COMIC_PADDLE_MASK` should be treated as migration aliases only if tests need backward compatibility.

---

### Task 1: Add Engine Preset Data to Project Presets

**Files:**

- Modify: `N:\TraduzAI\src\lib\projectPresets.ts`
- Modify: `N:\TraduzAI\src\lib\__tests__\projectPresets.test.ts`
- Modify: `N:\TraduzAI\src\pages\Setup.tsx`

**Step 1: Write failing TypeScript tests**

Add expectations:

```ts
expect(getProjectPreset("manga_bw").settings.engine_preset_id).toBe("manga");
expect(getProjectPreset("manhwa_webtoon_color").settings.engine_preset_id).toBe("manhwa_manhua");
expect(getProjectPreset("manhua_color").settings.engine_preset_id).toBe("manhwa_manhua");
```

Expected: fail before the field exists.

**Step 2: Extend `ProjectPresetSettings`**

Add:

```ts
engine_preset_id: "manga" | "manhwa_manhua" | "default";
```

Assign:

- `manga_bw` -> `manga`
- `manhwa_webtoon_color` -> `manhwa_manhua`
- `manhua_color` -> `manhwa_manhua`
- Other presets -> `default` unless they are clearly manga/manhwa-specific.

**Step 3: Show the engine preset in Setup**

In the preset summary chips in `Setup.tsx`, add a compact chip:

```text
Motores: Manga
Motores: Manhwa/Manhua
Motores: Padrao
```

Keep UI text in Portuguese. Do not show the full engine list in the setup card; keep the setup surface compact.

**Step 4: Verify**

```powershell
npx vitest run src/lib/__tests__/projectPresets.test.ts
npx tsc --noEmit
```

Expected: pass.

---

### Task 2: Carry Engine Preset Through Pipeline Config

**Files:**

- Modify: `N:\TraduzAI\src\lib\stores\appStore.ts`
- Modify: `N:\TraduzAI\src\lib\tauri.ts`
- Modify: `N:\TraduzAI\src-tauri\src\commands\pipeline.rs`
- Modify: `N:\TraduzAI\pipeline\main.py`
- Test: existing TypeScript/Rust/Python config tests, plus focused new tests if config shape tests exist.

**Step 1: Add failing config assertion**

Add a test or extend an existing pipeline-config test so a project using `manga_bw` sends:

```json
{
  "engine_preset_id": "manga"
}
```

And a project using `manhwa_webtoon_color` sends:

```json
{
  "engine_preset_id": "manhwa_manhua"
}
```

**Step 2: Propagate without breaking old projects**

Rules:

- If `project.preset.settings.engine_preset_id` exists, use it.
- Else infer from `project.preset.id`.
- Else infer from `idioma_origem` and content preset:
  - `ja` -> `manga`
  - `ko`, `zh`, `zh-CN`, `zh-TW` -> `manhwa_manhua`
  - otherwise `default`
- Preserve old `qualidade`, `runtime_profile`, and OCR config behavior.

**Step 3: Verify**

```powershell
npx tsc --noEmit
cd src-tauri; cargo check
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_runtime_profiles.py -q
```

Expected: pass or document unrelated pre-existing failures.

---

### Task 3: Add Python Engine Preset Resolver

**Files:**

- Create: `N:\TraduzAI\pipeline\vision_stack\engine_presets.py`
- Test: `N:\TraduzAI\pipeline\tests\test_engine_presets.py`

**Step 1: Write failing resolver tests**

Expected public API:

```python
resolve_engine_preset(config: dict | None = None, *, idioma_origem: str = "") -> EnginePreset
```

Expected manga stages:

```python
{
    "id": "manga",
    "detector": "anime-text-yolo-n",
    "font_detector": "yuzumarker-font-detection",
    "segmenter": "manga-text-segmentation-2025",
    "bubble_segmenter": "speech-bubble-segmentation",
    "ocr": "paddle-ocr-vl-1.5",
    "inpainter": "aot-inpainting",
    "mask_strategy": "segmentation_assisted",
}
```

Expected Manhwa/Manhua stages:

```python
{
    "id": "manhwa_manhua",
    "detector": "comic-text-bubble-detector",
    "font_detector": "default",
    "segmenter": "manga-text-segmentation-2025",
    "bubble_segmenter": "speech-bubble-segmentation",
    "ocr": "paddle-ocr-vl-1.5",
    "inpainter": "aot-inpainting",
    "mask_strategy": "roi_segmentation_assisted",
}
```

Expected default stages:

```python
{
    "id": "default",
    "detector": "default",
    "font_detector": "default",
    "segmenter": "disabled",
    "bubble_segmenter": "disabled",
    "ocr": "default",
    "inpainter": "default",
    "mask_strategy": "default",
}
```

**Step 2: Implement immutable preset objects**

Use a small dataclass:

```python
@dataclass(frozen=True)
class EnginePreset:
    id: str
    content_family: str
    detector: str
    font_detector: str
    segmenter: str
    bubble_segmenter: str
    ocr: str
    inpainter: str
    mask_strategy: str
```

`content_family` should be `"manga"`, `"manhwa_manhua"`, or `"default"`.

**Step 3: Verify**

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_engine_presets.py -q
```

Expected: pass.

---

### Task 4: Route Detect/OCR by Preset

**Files:**

- Modify: `N:\TraduzAI\pipeline\vision_stack\runtime.py`
- Modify: `N:\TraduzAI\pipeline\vision_stack\detector.py`
- Test: `N:\TraduzAI\pipeline\tests\test_vision_stack_runtime.py`
- Test: `N:\TraduzAI\pipeline\tests\test_vision_stack_detector.py`

**Step 1: Write routing tests**

Tests:

- Manga preset requests:
  - `anime-text-yolo-n`
  - `yuzumarker-font-detection`
  - `manga-text-segmentation-2025`
  - `speech-bubble-segmentation`
  - `paddle-ocr-vl-1.5`
  - `aot-inpainting`
- Manhwa/manhua preset requests:
  - `comic-text-bubble-detector`
  - `manga-text-segmentation-2025`
  - `speech-bubble-segmentation`
  - `paddle-ocr-vl-1.5`
  - `aot-inpainting`
- Manhwa/manhua segmentation stages are crop/ROI scoped.
- `default` keeps current behavior.

**Step 2: Replace fixed CJK stage list with preset-aware stage selection**

Current code has `_KOHARU_CJK_OCR_STEPS`. Keep a compatibility wrapper if needed, but introduce:

```python
def _engine_steps_for_preset(preset: EnginePreset) -> list[str]:
    ...
```

For source-native execution, each stage should call a TraduzAI adapter. If a source-native adapter is missing, the code must return a clear unsupported-engine error for that stage instead of silently switching to a different model.

**Step 3: Preserve preset telemetry**

Write preset id, content family, mask strategy, and stage list into debug/telemetry:

```json
{
  "engine_preset_id": "manhwa_manhua",
  "content_family": "manhwa_manhua",
  "mask_strategy": "roi_segmentation_assisted",
  "engine_steps": [...]
}
```

**Step 4: Verify**

```powershell
pipeline/venv/Scripts/python.exe -m pytest `
  pipeline/tests/test_engine_presets.py `
  pipeline/tests/test_vision_stack_runtime.py `
  pipeline/tests/test_vision_stack_detector.py `
  -q
```

Expected: pass.

---

### Task 5: Implement Mask Strategy per Preset

**Files:**

- Create or modify: `N:\TraduzAI\pipeline\vision_stack\cjk_segmentation_mask.py`
- Modify: `N:\TraduzAI\pipeline\vision_stack\runtime.py`
- Modify: `N:\TraduzAI\pipeline\inpainter\mask_builder.py`
- Test: `N:\TraduzAI\pipeline\tests\test_cjk_segmentation_mask.py`
- Test: `N:\TraduzAI\pipeline\tests\test_inpaint_mask_geometry.py`

**Step 1: Write Manga mask tests**

Expected behavior:

- Uses Anime Text YOLO boxes as primary text boxes.
- Uses Manga-Text-Segmentation-2025 as glyph/text probability source.
- Uses Speech Bubble Segmentation only as clip/region context.
- Uses YuzuMarker output as font/style metadata, not as OCR/layout.
- Uses PaddleOCR-VL text geometry to preserve OCR payload and refine mask guards when available.
- Final inpaint mask is expanded from glyph/text pixels, not broad boxes.

**Step 2: Write Manhwa/Manhua ROI mask tests**

Expected behavior:

- Uses Comic Text & Bubble Detection for primary text/bubble regions.
- Builds detector-derived crops/ROIs around those regions with padding.
- Runs Manga-Text-Segmentation-2025 only inside crops/ROIs.
- Runs Speech Bubble Segmentation only inside crops/ROIs.
- Remaps local crop masks back to full-page coordinates.
- Uses PaddleOCR-VL text geometry to refine/validate the mask.
- Rejects crop masks that are empty, too rectangular, too sparse, too broad, or worse than OCR text-pixel fallback.
- Never runs full-page segmentation on very tall Manhwa/Manhua pages by default.
- Never fills full bubble by default.

**Step 3: Implement strategy functions**

```python
build_manga_segmentation_mask(
    image_rgb,
    blocks,
    segmentation_mask,
    bubble_regions=None,
    ocr_texts=None,
) -> np.ndarray

build_manhwa_manhua_roi_segmentation_mask(
    image_rgb,
    detector_blocks,
    ocr_texts,
    *,
    segmenter,
    bubble_segmenter=None,
) -> np.ndarray
```

Selection rule:

```python
if preset.mask_strategy == "segmentation_assisted":
    mask = build_manga_segmentation_mask(...)
elif preset.mask_strategy == "roi_segmentation_assisted":
    mask = build_manhwa_manhua_roi_segmentation_mask(...)
```

**Step 4: Add crop/ROI rules for Manhwa/Manhua**

Rules:

- Create crops from merged detector text/bubble regions, not from the full 7000+ px page.
- Pad each crop enough to keep balloon borders and text halos, default `max(32, 0.20 * max(width, height))`.
- Clamp crop size to a model-friendly window; split again if height remains too large.
- Convert local segmentation and bubble masks back to full-page coordinates by adding crop origin.
- For SFX/outside-balloon text, do not clip to bubble segmentation unless overlap is clearly dominant.
- If all crop segmentation candidates are rejected, fall back to OCR/detector text-pixel extraction instead of full bbox fill.

**Step 5: Verify**

```powershell
pipeline/venv/Scripts/python.exe -m pytest `
  pipeline/tests/test_cjk_segmentation_mask.py `
  pipeline/tests/test_inpaint_mask_geometry.py `
  pipeline/tests/test_vision_stack_runtime.py `
  -q
```

Expected: pass.

---

### Task 6: Add AOT Inpainting as the CJK Preset Inpainter

**Files:**

- Create or modify: `N:\TraduzAI\pipeline\vision_stack\aot_inpainter.py`
- Modify: `N:\TraduzAI\pipeline\vision_stack\inpainter.py`
- Modify: `N:\TraduzAI\pipeline\vision_stack\runtime.py`
- Test: `N:\TraduzAI\pipeline\tests\test_vision_stack_inpainter.py`
- Test: `N:\TraduzAI\pipeline\tests\test_vision_stack_runtime.py`

**Step 1: Write AOT selection tests**

Expected:

- `manga` preset selects `aot-inpainting`.
- `manhwa_manhua` preset selects `aot-inpainting`.
- If AOT weights/runtime are unavailable, error/debug output is explicit.

**Step 2: Implement AOT adapter**

Rules:

- Keep AOT behind `TRADUZAI_AOT_INPAINT=1` until verified.
- Do not silently use Flux.
- Do not silently use LaMA unless an explicit fallback flag is enabled.
- The crop/ROI passed to AOT must come from the final nonzero mask.

**Step 3: Verify**

```powershell
pipeline/venv/Scripts/python.exe -m pytest `
  pipeline/tests/test_vision_stack_inpainter.py `
  pipeline/tests/test_vision_stack_runtime.py `
  -q
```

Expected: pass or clear unsupported-engine failure if AOT model/runtime is not present yet.

---

### Task 7: Visual Validation CLI

**Files:**

- Create: `N:\TraduzAI\tools\cjk_engine_preset_poc.py`

**Step 1: CLI contract**

```powershell
pipeline/venv/Scripts/python.exe tools/cjk_engine_preset_poc.py `
  --image "path\page.jpg" `
  --preset manga `
  --out "debug\cjk_preset_manga"
```

```powershell
pipeline/venv/Scripts/python.exe tools/cjk_engine_preset_poc.py `
  --image "path\page.jpg" `
  --preset manhwa_manhua `
  --out "debug\cjk_preset_manhwa"
```

**Step 2: Artifacts**

Write:

- `01_detector_boxes.png`
- `02_crop_roi_windows.png`
- `03_bubble_regions.png`
- `04_segmentation_or_text_geometry.png`
- `05_text_mask.png`
- `06_inpaint_input_mask.png`
- `07_overlay.png`
- `page_result.json`
- `engine_preset.json`

**Step 3: Acceptance**

Manga:

- Anime Text YOLO boxes match visible text/SFX.
- PaddleOCR-VL text geometry is preserved.
- Full-page segmentation-assisted text mask follows glyphs and SFX.
- Bubble segmentation clips only when it helps, not as erase fill.

Manhwa/Manhua:

- Comic Text & Bubble Detection boxes match the good detector result.
- Crop/ROI windows cover the detected text/bubble regions without running segmentation over the whole tall page.
- Crop segmentation produces useful text masks after remap.
- PaddleOCR-VL text geometry is preserved.
- Bubble segmentation clips only when it helps, not as erase fill.

Both:

- AOT receives a mask-derived ROI.
- No Flux path is used.
- Disabling the preset flag returns to current behavior.

---

## Final Validation Bundle

```powershell
npx vitest run src/lib/__tests__/projectPresets.test.ts
npx tsc --noEmit
cd src-tauri; cargo check
cd ..
pipeline/venv/Scripts/python.exe -m pytest `
  pipeline/tests/test_engine_presets.py `
  pipeline/tests/test_cjk_segmentation_mask.py `
  pipeline/tests/test_inpaint_mask_geometry.py `
  pipeline/tests/test_vision_stack_detector.py `
  pipeline/tests/test_vision_stack_ocr.py `
  pipeline/tests/test_vision_stack_inpainter.py `
  pipeline/tests/test_vision_stack_runtime.py `
  pipeline/tests/test_primary_ocr_routing.py `
  -q
```

## Stop Conditions

Stop and report instead of pushing forward if:

- Anime Text YOLO (N) cannot be loaded source-natively without copying GPL implementation code.
- Comic Text & Bubble Detection cannot produce stable Manhwa/Manhua crop regions.
- Manga-Text-Segmentation-2025 cannot be used source-natively for masks.
- Speech Bubble Segmentation cannot be used as a clip/context signal without damaging SFX.
- PaddleOCR-VL cannot be matched reliably to detector boxes.
- AOT runtime/weights are unavailable and no explicit fallback policy has been approved.
- The new mask removes important panel line art more often than current fallback.

## Future Phase Not Included Here

The API/server worker phase stays deferred. If the Manhwa/Manhua ROI segmentation path performs poorly, create a follow-up plan for per-family thresholds or OCR/detector text-pixel fallback.

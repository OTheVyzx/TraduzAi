# Koharu CJK Quality Gate Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fazer o TraduzAi usar o Koharu de forma adaptativa para CJK: manter o caminho rapido por pagina/ROI quando ele e confiavel, mas cair automaticamente para deteccao full-page do Koharu quando houver texto coreano residual, OCR ruim, SFX/fala mal classificado, inpaint incompleto ou typesetting ilegivel.

**Architecture:** O `fast-page` e o `vision-worker` persistente continuam sendo o caminho principal. O pipeline passa a registrar qual rota visual foi usada por pagina/bloco (`roi_ocr_only`, `worker_page_detect`, `koharu_http_page`, `legacy`) e roda um QA deterministico com acoes, nao apenas avisos. Quando o QA reprova CJK, o pipeline reprocessa somente a pagina ou banda afetada usando deteccao full-page do worker Koharu, preserva tudo em `project.json`, e so entao renderiza/exporta.

**Tech Stack:** Python pipeline (`pipeline/main.py`, `pipeline/strip/run.py`, `pipeline/vision_stack/runtime.py`), Rust `vision-worker`, Koharu ML models, JSONL sidecar protocol, `project.json`, pytest, scripts de benchmark visual, React/Tauri editor para revisao final.

---

## Current Root Cause

Hoje o resultado ruim nao vem de "Koharu nao consegue". O Koharu consegue quando roda a pagina inteira.

O problema atual e que a rodada CJK rapida usa:

```text
TraduzAi detecta bandas/baloes
-> recorta ROI
-> chama vision-worker em mode=ocrOnly
-> Koharu pula o detector
-> OCR roda nas caixas que o TraduzAi ja escolheu
-> TraduzAi filtra textos
-> TraduzAi traduz/inpaint/typeset
```

Isso e diferente do fluxo do Koharu:

```text
Comic Text & Bubble Detector
-> segmentation/masks
-> PaddleOCR-VL
-> text nodes preservados
-> translate/render/inpaint sobre a cena
```

Evidencia da rodada `DDDDDDDDDDDDDDDDDDDD/traduzido3`:

- `batch_mode = roi`
- `roi_job_count = 120`
- `ocr_only_job_count = 120`
- `text_count = 57`
- `filtered_text_count = 36`
- `empty_precomputed_band_count = 63`
- erros visuais ainda tinham coreano residual, SFX tratado como fala, OCR "The image is too blurry..." traduzido e inpaint incompleto.

## Design Rules

1. O fast path continua existindo, mas nao pode ser o caminho unico para CJK.
2. `ocrOnly` so pode ser usado quando a caixa conhecida e confiavel.
3. Se o QA detectar risco CJK, rerodar a pagina com detector do Koharu.
4. "The image is too blurry to recognize any text content." e falha de OCR, nao texto traduzivel.
5. Hangul residual em fala/balao branco e falha automatica.
6. SFX coreano deve ser preservado quando for SFX real, nao quando for fala curta dentro de balao.
7. `project.json` precisa preservar fonte OCR, rota usada, textos filtrados e tentativas de rerun.
8. O editor continua usando o renderer/typesetter do TraduzAi; Koharu Renderer fica fora do default ate validacao separada.
9. Qualquer ganho de velocidade precisa ser validado com imagem e metadados, nao so tempo total.

---

## Engine Selector Contract

The setup screen can show these engines, but the runtime must record whether each one was actually used. This avoids the current ambiguity where the UI says "Koharu/PaddleOCR-VL", while the fast route actually sends cropped known boxes through `ocrOnly` and skips detector/segmentation.

| UI engine | Runtime contract | Required telemetry |
| --- | --- | --- |
| Detector: `PP-DocLayout V3` | Used only when page/band structural detection runs. It is not proof that speech text was detected. | `detector_engine`, `detector_route`, `page_or_band` |
| Segmentador: `Comic Text Detector (Segmentation)` | Must run in `worker_page_detect`/`koharu_http_page`. It is skipped in `roi_ocr_only`. | `segmentation_engine`, `segmentation_ran`, `text_region_count` |
| Segmentador de baloes: `Speech Bubble Segmentation` | Must run before classifying balloon text vs SFX on full-page fallback. | `bubble_engine`, `bubble_count`, `balloon_text_count`, `sfx_candidate_count` |
| OCR: `PaddleOCR-VL` | Can run either on known ROI or on Koharu full-page detections. The route matters more than the OCR name. | `ocr_engine`, `ocr_route`, `ocr_only`, `confidence`, `raw_text` |
| Tradutor: `LLM` / Google | Must not translate OCR failure phrases or mojibake. CJK default remains Google unless a separate local LLM validation is enabled. | `translator_engine`, `skipped_ocr_failure_count`, `translation_source` |
| Inpainter: `Lama Manga` | Must rerun when QA finds residual source text in dialogue regions. It must not erase SFX marked as preserved. | `inpaint_engine`, `mask_source`, `rerun_count`, `residual_text_count` |
| Renderizador: `Koharu Renderer` | Not default in this plan. TraduzAi renderer remains owner until parity is proven. | `renderer_engine`, `preview_renderer`, `export_renderer` |

Acceptance for this contract:

- every page in `project.json` has a machine-readable route history;
- `roi_ocr_only` never pretends that `Comic Text & Bubble Detector` ran;
- the UI can show "Rapido" vs "Koharu pagina" from telemetry, not from assumptions;
- QA decisions can be traced back to the route that created the text.

---

## Phase 0: Baseline and Reproduction Lock

**Files:**
- Create: `pipeline/scripts/compare_koharu_cjk_routes.py`
- Create: `pipeline/tests/test_koharu_cjk_route_summary.py`
- Read/reference: `N:/TraduzAI/TraduzAi/DDDDDDDDDDDDDDDDDDDD/traduzido3/project.json`
- Read/reference: `N:/TraduzAI/TraduzAi/DDDDDDDDDDDDDDDDDDDD/traduzido3/qa_report.json`
- Read/reference: `pipeline/vision_stack/runtime.py`
- Read/reference: `pipeline/strip/run.py`
- Read/reference: `vision-worker/src/main.rs`

**Step 1: Write summary parser test**

Create `pipeline/tests/test_koharu_cjk_route_summary.py`.

Test data should include a small fake `project.json` summary with:

```python
def test_route_summary_counts_roi_and_filtered_texts(tmp_path):
    project = {
        "paginas": [
            {
                "page_profile": {
                    "strip_perf_summary": {
                        "koharu_cjk_precompute": {
                            "batch_mode": "roi",
                            "roi_job_count": 120,
                            "text_count": 57,
                            "filtered_text_count": 36,
                            "worker_batch": {
                                "persistent": True,
                                "ocr_only_job_count": 120,
                            },
                        }
                    }
                }
            }
        ]
    }
    summary = summarize_koharu_routes(project)
    assert summary["batch_mode"] == "roi"
    assert summary["ocr_only_job_count"] == 120
    assert summary["filtered_text_count"] == 36
```

**Step 2: Implement route summary script**

In `pipeline/scripts/compare_koharu_cjk_routes.py`, add:

```python
def summarize_koharu_routes(project: dict) -> dict:
    ...

def detect_cjk_quality_risks(project: dict, qa_report: dict | None = None) -> list[dict]:
    ...
```

Initial risk codes:

- `roi_ocr_only_all_jobs`
- `high_filtered_text_count`
- `ocr_failure_phrase`
- `hangul_residual`
- `empty_translation_with_source`
- `sfx_preserved_inside_dialogue_candidate`
- `low_confidence_cjk_source`

**Step 3: Add CLI output**

Run:

```powershell
python pipeline\scripts\compare_koharu_cjk_routes.py `
  --project "N:\TraduzAI\TraduzAi\DDDDDDDDDDDDDDDDDDDD\traduzido3\project.json" `
  --qa "N:\TraduzAI\TraduzAi\DDDDDDDDDDDDDDDDDDDD\traduzido3\qa_report.json" `
  --out "N:\TraduzAI\TraduzAi\debug\koharu_cjk_route_baseline"
```

Expected outputs:

- `summary.json`
- `risks.json`
- `pages_to_inspect.json`

**Step 4: Run tests**

Run:

```powershell
python -m pytest pipeline\tests\test_koharu_cjk_route_summary.py -q
```

Expected: PASS.

---

## Phase 1: Make Koharu Route Selection Explicit

**Files:**
- Create: `pipeline/vision_stack/koharu_routes.py`
- Modify: `pipeline/vision_stack/runtime.py`
- Modify: `pipeline/strip/run.py`
- Test: `pipeline/tests/test_koharu_routes.py`
- Test: `pipeline/tests/test_vision_stack_runtime.py`
- Test: `pipeline/tests/test_strip_run.py`

**Step 1: Add route decision dataclass**

Create `pipeline/vision_stack/koharu_routes.py`.

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class KoharuRouteDecision:
    mode: str
    reason: str
    use_known_bboxes: bool = False
    force_detector: bool = False
    max_new_tokens: int = 128
    metadata: dict = field(default_factory=dict)
```

Allowed modes:

- `roi_ocr_only`
- `roi_detect`
- `page_detect`
- `http_page_pipeline`
- `legacy`

**Step 2: Write route tests**

Cases:

- trusted ROI with known boxes -> `roi_ocr_only`
- CJK page with prior OCR failure -> `page_detect`
- high filtered count -> `page_detect`
- missing worker path but Koharu exe available -> `http_page_pipeline`
- non-CJK -> preserve current route

Run:

```powershell
python -m pytest pipeline\tests\test_koharu_routes.py -q
```

Expected: FAIL until module exists.

**Step 3: Implement route selection**

Add:

```python
def choose_koharu_route(
    *,
    idioma_origem: str,
    has_known_bboxes: bool,
    risk_codes: list[str] | None = None,
    page_retry_count: int = 0,
    worker_available: bool = False,
    http_available: bool = False,
) -> KoharuRouteDecision:
    ...
```

Default policy:

- CJK + no risk + known boxes -> `roi_ocr_only`
- CJK + risk -> `page_detect`
- CJK + repeated risk + no worker -> `http_page_pipeline`
- non-CJK -> existing path

**Step 4: Wire runtime payload**

In `pipeline/vision_stack/runtime.py`, update `_build_koharu_worker_batch_request_payload()` so jobs can pass:

```python
job["koharu_route"] = {
    "mode": "page_detect",
    "reason": "hangul_residual",
    "use_known_bboxes": False,
}
```

Behavior:

- `roi_ocr_only` -> keep current `mode = "ocrOnly"` and `knownTextBBoxes`
- `roi_detect` -> use `mode = "region"` without `knownTextBBoxes`
- `page_detect` -> use `mode = "page"` without `knownTextBBoxes`

**Step 5: Persist telemetry**

Each result should include:

```json
"_koharu_route": {
  "mode": "page_detect",
  "reason": "hangul_residual",
  "used_known_bboxes": false,
  "worker": "persistent"
}
```

**Step 6: Run focused tests**

Run:

```powershell
python -m pytest pipeline\tests\test_koharu_routes.py pipeline\tests\test_vision_stack_runtime.py::TestKoharuWorker -q
python -m pytest pipeline\tests\test_strip_run.py -q
```

Expected: PASS.

---

## Phase 2: CJK QA With Actions

**Files:**
- Create: `pipeline/qa/page_quality.py`
- Create: `pipeline/tests/test_page_quality_cjk.py`
- Modify: `pipeline/main.py`
- Modify: `pipeline/strip/run.py`
- Modify: `pipeline/ocr/postprocess.py`

**Step 1: Write tests for OCR failure phrases**

In `pipeline/tests/test_page_quality_cjk.py`:

```python
def test_blurry_ocr_sentence_is_failure_not_translation():
    page = {
        "numero": 26,
        "textos": [
            {
                "original": "The image is too blurry to recognize any text content.",
                "translated": "A imagem esta muito desfocada para reconhecer qualquer conteudo de texto.",
                "tipo": "fala",
                "skip_processing": False,
            }
        ],
    }
    result = score_page_quality(page, source_language="korean")
    assert result["status"] == "fail"
    assert result["actions"][0]["action"] == "rerun_ocr_detect"
```

**Step 2: Write tests for Hangul residual**

```python
def test_hangul_in_dialogue_render_is_fail():
    page = {
        "numero": 37,
        "textos": [
            {"translated": "하하하.", "tipo": "fala", "skip_processing": False}
        ],
    }
    result = score_page_quality(page, source_language="korean")
    assert result["status"] == "fail"
    assert any(a["action"] == "rerun_ocr_detect" for a in result["actions"])
```

**Step 3: Write tests for SFX preservation**

Cases:

- Korean SFX outside speech balloon can be preserved.
- Korean text inside white balloon with sentence punctuation should be treated as dialogue candidate.
- pure punctuation or sound effect should not force translation.

**Step 4: Implement `score_page_quality()`**

Return:

```python
{
    "status": "pass" | "review" | "fail",
    "score": 0.0,
    "findings": [
        {
            "code": "ocr_failure_phrase",
            "severity": "critical",
            "page": 26,
            "block_id": "...",
            "message": "OCR returned a failure sentence",
        }
    ],
    "actions": [
        {
            "action": "rerun_ocr_detect",
            "scope": "page",
            "reason": "ocr_failure_phrase",
        }
    ],
}
```

**Step 5: Add OCR text helpers**

In `pipeline/ocr/postprocess.py`, add helpers:

```python
def is_ocr_failure_phrase(text: str) -> bool: ...
def has_hangul(text: str) -> bool: ...
def looks_like_mojibake_korean(text: str) -> bool: ...
def repair_utf8_mojibake_if_safe(text: str) -> str: ...
```

Do not auto-repair broad text without tests.

**Step 6: Store QA in `project.json`**

In `pipeline/main.py` and strip page builders, add:

```json
"qa": {
  "status": "fail",
  "score": 0.41,
  "findings": [],
  "actions": []
}
```

Keep old `qa_flags` for compatibility.

**Step 7: Run focused tests**

Run:

```powershell
python -m pytest pipeline\tests\test_page_quality_cjk.py -q
python -m pytest pipeline\tests\test_ocr_reviewer.py pipeline\tests\test_contextual_reviewer.py -q
```

Expected: PASS.

---

## Phase 3: Automatic CJK Page Fallback

**Files:**
- Modify: `pipeline/strip/run.py`
- Modify: `pipeline/vision_stack/runtime.py`
- Modify: `pipeline/main.py`
- Test: `pipeline/tests/test_strip_run.py`
- Test: `pipeline/tests/test_vision_stack_runtime.py`
- Test: `pipeline/tests/test_main_emit.py`

**Step 1: Add fallback test for ROI failure**

In `pipeline/tests/test_strip_run.py`, simulate:

- first ROI result has `The image is too blurry...`
- QA returns `rerun_ocr_detect`
- second call uses `page_detect`
- final precomputed page uses page result, not ROI result

Expected:

```python
assert telemetry["koharu_cjk_page_rerun_count"] == 1
assert mapped[band_index]["_ocr_stats"]["koharu_cjk_mode"] == "page_detect_fallback"
```

**Step 2: Add fallback scheduler**

In `pipeline/strip/run.py`, after ROI precompute:

1. evaluate each mapped band/page with `score_page_quality()`;
2. group failed CJK bands by source page;
3. rerun only failed source pages using `page_detect`;
4. map page-level results back into bands by bbox intersection;
5. replace only failed band results unless page-level result is empty.

**Step 3: Avoid infinite reruns**

Add limits:

```text
TRADUZAI_CJK_QA_RERUN=1
TRADUZAI_CJK_QA_RERUN_MAX=1
```

Persist:

```json
"_cjk_rerun": {
  "attempted": true,
  "route": "page_detect",
  "reason": "ocr_failure_phrase",
  "recovered_text_count": 2
}
```

**Step 4: Preserve filtered text audit**

When `_filter_koharu_cjk_page_result()` drops texts, keep an audit list:

```json
"_koharu_filtered_texts": [
  {
    "text": "...",
    "bbox": [x1, y1, x2, y2],
    "reason": "sfx_noise"
  }
]
```

Do not include this in final render, but preserve it in debug/profile fields.

**Step 5: Make selective filter less destructive**

Policy:

- Pure SFX outside balloon -> can skip.
- Hangul inside white/connected/top narration context -> keep unless pure punctuation/SFX.
- Short Korean text with punctuation -> keep for translation.
- OCR confidence `0.0` from worker should not alone force skip.

**Step 6: Run tests**

Run:

```powershell
python -m pytest pipeline\tests\test_strip_run.py -q
python -m pytest pipeline\tests\test_vision_stack_runtime.py -q
```

Expected: PASS.

---

## Phase 4: Worker Parity With Koharu Page Pipeline

**Files:**
- Modify: `vision-worker/src/main.rs`
- Modify: `pipeline/vision_stack/runtime.py`
- Test: `vision-worker` Rust tests
- Test: `pipeline/tests/test_vision_stack_runtime.py`

**Step 1: Confirm worker modes**

Existing modes:

- `page`: uses `ComicTextBubbleDetector` + `PaddleOCR-VL`
- `region`: crops region then detects
- `ocrOnly`: skips detector and OCRs known boxes

Keep all three.

**Step 2: Add explicit route metadata to worker response**

In `VisionResponse`, add optional:

```rust
route: Option<String>,
detector_used: Option<String>,
ocr_only: bool,
```

Use serde defaults so old consumers do not break.

**Step 3: Add fallback request test**

In `vision-worker/src/main.rs` tests:

- request with `mode = "page"` and `known_text_bboxes` should still detect page, not use `ocrOnly`;
- request with `mode = "ocrOnly"` should skip detector;
- response should report route.

**Step 4: Build and test**

Run:

```powershell
cd vision-worker
cargo test
cargo build
```

Expected: PASS and `target/debug/traduzai-vision.exe` updated.

**Step 5: Python runtime test**

In `pipeline/tests/test_vision_stack_runtime.py`, assert route payload:

```python
assert payload["mode"] == "page"
assert "knownTextBBoxes" not in payload
```

for fallback route.

---

## Phase 5: Inpaint QA and Recovery

**Files:**
- Create: `pipeline/qa/image_quality.py`
- Modify: `pipeline/vision_stack/runtime.py`
- Modify: `pipeline/inpainter/`
- Test: `pipeline/tests/test_image_quality.py`
- Test: `pipeline/tests/test_vision_stack_inpainter.py`
- Test: `pipeline/tests/test_mask_builder.py`

**Step 1: Add residual text tests**

Test cases:

- dark Hangul/Latin remnants inside white balloon after inpaint -> fail
- rectangular white patch over connected balloon border -> review/fail
- SFX skip region should not be inpainted
- balloon border should not be erased by over-expanded mask

**Step 2: Implement cheap image QA**

Add:

```python
def score_inpaint_quality(original: np.ndarray, cleaned: np.ndarray, page: dict) -> dict:
    ...
```

Rules:

- strong dark strokes remain inside cleaned text bbox -> `residual_text`
- large uniform rectangle overlapping non-rectangular balloon -> `white_patch_artifact`
- mask overlaps balloon outline too much -> `border_damage_risk`

**Step 3: Rerun inpaint only when needed**

Actions:

- `rerun_inpaint_with_expanded_mask`
- `rerun_inpaint_with_balloon_fill`
- `manual_review`

Do not blur artifacts as first solution.

**Step 4: Persist inpaint route**

Add per page/block:

```json
"_inpaint_quality": {
  "status": "pass",
  "findings": [],
  "rerun_count": 0
}
```

**Step 5: Run tests**

Run:

```powershell
python -m pytest pipeline\tests\test_image_quality.py pipeline\tests\test_vision_stack_inpainter.py pipeline\tests\test_mask_builder.py -q
```

Expected: PASS.

---

## Phase 6: Typesetting QA and Connected Balloons

**Files:**
- Modify: `pipeline/typesetter/renderer.py`
- Modify: `pipeline/layout/balloon_layout.py`
- Create: `pipeline/tests/test_typesetting_quality_gate.py`
- Test: `pipeline/tests/test_typesetting_layout.py`
- Test: `pipeline/tests/test_typesetting_renderer.py`

**Step 1: Add connected balloon tests**

Cases:

- one detected text covering two connected lobes should split into subregions when original text has two semantic chunks;
- two speech bubbles connected but separate texts should not become one rectangle;
- text must not cross balloon border;
- font must not shrink below readable floor unless `tipo=sfx` or explicit tiny text.

**Step 2: Add render quality score**

Expose:

```python
def score_typeset_block(block: dict, rendered_bbox: list[int], balloon_bbox: list[int]) -> dict:
    ...
```

Findings:

- `text_overflow`
- `font_too_small`
- `line_spacing_inconsistent`
- `connected_balloon_unbalanced`
- `translation_too_long_for_balloon`

**Step 3: Prefer reflow before shrink**

Fit order:

1. normalize text;
2. choose line breaks;
3. adjust bbox within safe balloon area;
4. shrink down to readable floor;
5. mark review instead of silently making unreadable text.

**Step 4: Keep renderer deterministic**

No LLM/VLM writes pixels. Optional model guidance can only supply JSON hints in a later phase.

**Step 5: Run tests**

Run:

```powershell
python -m pytest pipeline\tests\test_typesetting_quality_gate.py pipeline\tests\test_typesetting_layout.py pipeline\tests\test_typesetting_renderer.py -q
```

Expected: PASS.

---

## Phase 7: Partial Project and Page Events

**Files:**
- Create: `pipeline/project_stream.py`
- Modify: `pipeline/project_writer.py`
- Modify: `pipeline/fast_page_server.py`
- Modify: `worker/runner.py`
- Test: `pipeline/tests/test_project_stream.py`
- Test: `pipeline/tests/test_fast_page_server.py`
- Test: `worker/tests/test_runner_artifacts.py`

**Step 1: Preserve real schema**

Do not create a simplified parallel schema. Partial project pages must preserve fields consumed by the editor:

- `numero`
- `original_path`
- `translated_path`
- `image_path`
- `textos`
- `text_layers`
- `page_profile`
- `qa`
- `qa_flags`
- `_vision_backend`
- `_koharu_route`

**Step 2: Add atomic partial writer**

Use `project_writer.write_project_json_atomic()` when possible.

Functions:

```python
def load_partial_project(work_dir: Path, base_project: dict | None = None) -> dict: ...
def upsert_partial_page(project: dict, page: dict) -> dict: ...
def write_partial_project(work_dir: Path, project: dict) -> Path: ...
def finalize_partial_project(work_dir: Path, project: dict) -> Path: ...
```

**Step 3: Emit page event with QA**

`page_completed` should include:

```json
{
  "type": "page_completed",
  "protocol": "fast-page.v1",
  "current_page": 26,
  "artifact_kind": "translated_image",
  "artifact_path": "translated/026.jpg",
  "partial_project_path": "project.partial.json",
  "qa_status": "review",
  "qa_score": 0.72,
  "qa_findings_count": 2,
  "rerun_count": 1
}
```

**Step 4: Worker uploads partial project**

In `worker/runner.py`, upload:

- `translated_image`
- `project_partial`
- final `project_json`

**Step 5: Run tests**

Run:

```powershell
python -m pytest pipeline\tests\test_project_stream.py pipeline\tests\test_fast_page_server.py worker\tests\test_runner_artifacts.py -q
```

Expected: PASS.

---

## Phase 7.5: Preview and Editor Render Contract

**Files:**
- Modify: `src/pages/Preview.tsx`
- Modify: `src/pages/previewImage.ts`
- Modify: `src/lib/konvaExportRenderer.ts`
- Modify: `src/lib/stores/editorStore.ts`
- Modify: `src/components/editor/stage/EditorStage.tsx`
- Test: `src/lib/__tests__/previewImage.test.ts`
- Test: `src/lib/stores/__tests__/editorRenderPreviewCache.test.ts`
- Test: `e2e/editor-rebuild.spec.ts`

**Problem:**

The preview can look wrong while the editor looks correct because they are not always rendering the same artifact:

- the editor uses live editable text layers, current style hydration, Konva placement and inpaint/base image state;
- preview/export can use a cached raster or an older render path;
- if the pipeline changed a layer, preview freshness can become stale even when the editor view is correct;
- forcing "Salvar+Render" fixes it because it rebuilds the raster from the editor state.

This plan should remove that manual step for pipeline outputs.

**Step 1: Define one authoritative render input**

Create a shared normalized page render payload:

```ts
type RenderSource = {
  pageId: string
  baseImage: string
  cleanImage?: string
  layers: TextLayer[]
  rendererVersion: string
  sourceRevision: string
}
```

Both preview and editor export must derive from this payload.

**Step 2: Add freshness hash**

Hash the fields that affect final pixels:

- page image path and modified time/hash;
- clean image path and modified time/hash;
- layer ids, text, bbox, font, size, color, stroke, shadow, rotation, alignment;
- renderer version.

Store:

```json
{
  "render_cache_key": "...",
  "render_cache_path": "...",
  "render_cache_status": "fresh"
}
```

**Step 3: Auto-render stale preview before export**

If preview is stale, do not block export with a modal first. The export flow should:

1. detect stale pages;
2. render them with the same renderer used by editor/export;
3. update `project.json`;
4. continue export if all pages pass;
5. show a blocking dialog only if auto-render fails.

**Step 4: Keep editor direct manipulation unchanged**

Do not move the live editor onto a separate preview-only renderer. If Konva is used for final render, use it as a shared export renderer behind the same text-fit contract, not as a second visual truth.

**Step 5: Tests**

Add tests for:

- preview cache becomes stale after text/layer/style change;
- export auto-renders stale pages instead of requiring manual "Salvar+Render";
- editor and preview use the same text fit result for connected balloons;
- CJK page with rerun fallback opens in editor without losing route metadata.

**Run:**

```powershell
npm run check
npx playwright test e2e/editor-rebuild.spec.ts --grep "preview|editor|export" --timeout=90000
```

Expected: PASS.

---

## Phase 8: UI Review Queue

**Files:**
- Modify: `src/pages/Processing.tsx`
- Modify: `src/lib/stores/appStore.ts`
- Modify: `src/lib/tauri.ts`
- Test: `src/lib/**/__tests__/**/*.test.ts`
- Test: `e2e/editor-rebuild.spec.ts`

**Step 1: Add store tests**

Test:

- page event updates page status;
- QA review page appears under review filter;
- clicking a page opens editor/preview with correct page selected;
- completed job still opens final project path.

**Step 2: Add Processing UI**

Show dense operational controls:

- page number;
- thumbnail;
- status: `Processando`, `Pronta`, `Revisar`, `Falhou`;
- badges: `OCR`, `Inpaint`, `Texto`, `SFX`;
- route label: `Rapido`, `Koharu pagina`, `Manual`;
- button to open editor at that page.

No marketing copy.

**Step 3: Add filters**

- `Todas`
- `Prontas`
- `Revisar`
- `Falharam`

**Step 4: Run frontend tests**

Run:

```powershell
npm run check
npx playwright test e2e/editor-rebuild.spec.ts --grep "processing|preview|editor" --timeout=90000
```

Expected: PASS.

---

## Phase 9: Benchmark Harness and Acceptance Gate

**Files:**
- Create: `pipeline/scripts/benchmark_koharu_cjk_quality_gate.py`
- Modify: `.gitignore`
- Test: `pipeline/tests/test_koharu_cjk_quality_benchmark.py`

**Step 1: Add benchmark script**

Input examples:

```powershell
python pipeline\scripts\benchmark_koharu_cjk_quality_gate.py `
  --source "N:\TraduzAI\TraduzAi\exemplos\exemploko\환생천마" `
  --out "N:\TraduzAI\TraduzAi\debug\koharu_cjk_quality_gate"
```

**Step 2: Record required metrics**

Write `summary.json` with:

- `time_to_first_page_sec`
- `full_chapter_sec`
- `page_count`
- `fast_roi_pages`
- `page_detect_reruns`
- `ocr_failure_phrase_count`
- `hangul_residual_count`
- `sfx_preserved_count`
- `filtered_text_count`
- `qa_pass`
- `qa_review`
- `qa_fail`
- `inpaint_rerun_count`
- `typeset_review_count`

**Step 3: Create visual sample sheet**

Output:

- `samples_before_after.html`
- `samples_contact_sheet.jpg`
- selected pages: known problematic pages plus random pass pages

**Step 4: Add non-flaky parser tests**

Test only summary parsing and page selection. Do not run full chapters in CI.

**Step 5: Manual acceptance run**

Run on:

- `N:\TraduzAI\TraduzAi\DDDDDDDDDDDDDDDDDDDD\traduzido3` as baseline reference
- `N:\TraduzAI\TraduzAi\exemplos\exemploko\환생천마`
- the English/PT-BR references when comparing meaning and visual placement

Acceptance:

- no `The image is too blurry...` appears in final translated text;
- no Hangul remains in normal dialogue balloons;
- SFX stays when it is real SFX;
- `filtered_text_count` is audited;
- full chapter time remains close to fast-page baseline unless reruns are triggered;
- rerun pages are visible in the summary.

---

## Phase 10: Safe Defaults and Rollback Flags

**Files:**
- Modify: `.env.example`
- Modify: `pipeline/runtime_profiles.py`
- Modify: `worker/config.py`
- Test: `pipeline/tests/test_runtime_profiles.py`
- Test: `worker/tests/test_worker_config.py`

**Flags:**

```text
TRADUZAI_KOHARU_CJK_STRIP_ROI=1
TRADUZAI_KOHARU_WORKER_OCR_ONLY=adaptive
TRADUZAI_CJK_QA=1
TRADUZAI_CJK_QA_RERUN=1
TRADUZAI_CJK_QA_RERUN_MAX=1
TRADUZAI_CJK_PAGE_DETECT_FALLBACK=1
TRADUZAI_CJK_AUDIT_FILTERED_TEXTS=1
TRADUZAI_INPAINT_QA=1
TRADUZAI_TYPESET_QA=1
TRADUZAI_LAYOUT_GUIDANCE=0
```

**Default policy:**

- ROI stays on for speed.
- `ocrOnly` becomes adaptive, not unconditional.
- CJK QA and CJK page fallback default on.
- Inpaint/typeset QA default on once tests pass.
- VLM/layout guidance default off.
- Legacy behavior remains available:

```text
TRADUZAI_CJK_QA=0
TRADUZAI_KOHARU_WORKER_OCR_ONLY=1
TRADUZAI_CJK_PAGE_DETECT_FALLBACK=0
```

**Run:**

```powershell
python -m pytest pipeline\tests\test_runtime_profiles.py worker\tests\test_worker_config.py -q
```

Expected: PASS.

---

## Definition of Done

The rollout is done when all are true:

- Focused Python tests pass:

```powershell
python -m pytest `
  pipeline\tests\test_koharu_routes.py `
  pipeline\tests\test_page_quality_cjk.py `
  pipeline\tests\test_strip_run.py `
  pipeline\tests\test_vision_stack_runtime.py `
  pipeline\tests\test_vision_stack_inpainter.py `
  pipeline\tests\test_typesetting_renderer.py `
  -q
```

- `vision-worker` tests/build pass:

```powershell
cd vision-worker
cargo test
cargo build
```

- Frontend checks pass after UI phase:

```powershell
npm run check
npx playwright test e2e/editor-rebuild.spec.ts --grep "processing|preview|editor" --timeout=90000
```

- Manual CJK benchmark produces:
  - `summary.json`
  - visual sample sheet
  - final `project.json`
  - final translated images
  - route telemetry and rerun counts

- Visual acceptance on known bad pages:
  - no untranslated Hangul in dialogue;
  - no OCR failure sentence translated;
  - SFX preserved only when it is actually SFX;
  - no large white rectangle over connected balloons;
  - text does not overflow balloons;
  - preview/editor consistency remains intact.

## Recommended Execution Order

1. Phase 0: baseline and route summary.
2. Phase 1: explicit Koharu route selection.
3. Phase 2: CJK QA with actions.
4. Phase 3: automatic CJK page fallback.
5. Phase 4: worker parity/metadata.
6. Phase 5: inpaint QA.
7. Phase 6: typesetting QA.
8. Phase 7: partial project/page events.
9. Phase 7.5: preview/editor render contract.
10. Phase 8: UI review queue.
11. Phase 9: benchmark harness.
12. Phase 10: defaults and rollback flags.

## First Implementation Slice

Start with Phases 0-3 only.

That gives the highest quality gain with the least product churn:

- we know exactly where `ocrOnly` fails;
- we can keep fast ROI for pages that pass;
- bad CJK pages rerun with Koharu page detector;
- failures are visible in `project.json`;
- UI/export work waits until the pipeline stops producing obvious CJK errors.

Do not start Phase 8 UI before Phase 3 is validated on the Korean chapter.

Phase 7.5 can be implemented before Phase 8 if the app is blocking export with "Preview final desatualizado". It should remain a render/cache fix, not a rewrite of the editor.

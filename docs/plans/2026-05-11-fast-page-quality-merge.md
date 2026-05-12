# Fast Page Quality Merge Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Mesclar a velocidade do fluxo `fast-page` com a qualidade, rastreabilidade e edicao do TraduzAi, entregando paginas prontas rapidamente e reprocessando automaticamente apenas o que precisa melhorar.

**Architecture:** O worker quente continua sendo o caminho principal para processamento e eventos por pagina. A pipeline passa a salvar estado parcial por pagina, executar um QA visual/textual leve, gerar candidatos de render quando necessario e escolher automaticamente o melhor resultado antes de exportar o capitulo. O editor vira a camada final de revisao focada, nao o gargalo do fluxo.

**Tech Stack:** Python pipeline/worker, JSONL sidecar protocol, `project.json` schema v2, FastAPI worker API, React/Tauri editor, PIL/OpenCV para metricas visuais, pytest/Playwright para validacao.

---

## Current Evidence

- Teste `full-113-fast-page`: `60` paginas em `228.014s`, com `60` eventos `page_completed` e uma sessao `fast-page`.
- Amostras mostram boa velocidade e boa preservacao geral, mas ainda ha casos para QA:
  - texto pequeno demais em alguns baloes;
  - alguns textos coreanos/SFX devem permanecer sem traducao quando sao SFX ou creditos;
  - precisamos distinguir "pagina boa" de "pagina precisa revisar" automaticamente.
- O gargalo antigo vinha de fluxo serial e cold start. O caminho correto agora e manter processo quente e melhorar decisao/qualidade por pagina.

## Design Principles

1. Mostrar a pagina assim que estiver pronta.
2. Nunca bloquear o capitulo inteiro por uma pagina ruim.
3. Salvar tudo em `project.json` para permitir edicao humana.
4. Reprocessar seletivamente so o que falhou no QA.
5. Escolher por pagina/balao entre render rapido e render de qualidade.
6. Manter flags de escape para voltar ao comportamento antigo.

## Target Flow

```text
CBZ/upload
  -> extract pages
  -> fast-page quente
  -> page_completed + artifacts por pagina
  -> project.json parcial
  -> QA por pagina
  -> render selector
  -> reprocess only bad pages
  -> UI mostra filtros de revisao
  -> export CBZ final
```

---

## Task 1: Freeze the Current Fast-Page Contract

**Files:**
- Modify: `pipeline/fast_page_server.py`
- Modify: `worker/fast_page.py`
- Modify: `worker/runner.py`
- Test: `pipeline/tests/test_fast_page_server.py`
- Test: `worker/tests/test_fast_page_client.py`
- Test: `worker/tests/test_runner_artifacts.py`

**Step 1: Write contract tests for page and chapter inputs**

Add tests asserting:
- image input `040.jpg` emits `source_page_number=40`;
- archive input `113화.cbz` emits pages `1..N`, not `113`;
- JSON written by `FastPageProcessClient` is ASCII escaped;
- same process is reused while stdin/stdout pipes are open.

**Step 2: Run tests to verify current contract**

Run:

```powershell
python -m pytest pipeline\tests\test_fast_page_server.py worker\tests\test_fast_page_client.py worker\tests\test_runner_artifacts.py -q
```

Expected:

```text
PASS
```

**Step 3: Add protocol version to fast-page events**

In `pipeline/fast_page_server.py`, include:

```python
"protocol": "fast-page.v1"
```

on `page_completed`, `ready`, `complete`, `bye`, and `error` events.

**Step 4: Preserve backward compatibility**

Update worker consumers to ignore unknown fields and not require `protocol`.

**Step 5: Run focused tests**

Run:

```powershell
python -m pytest pipeline\tests\test_fast_page_server.py worker\tests -q
```

Expected:

```text
PASS
```

---

## Task 2: Persist Partial Project State Per Page

**Files:**
- Create: `pipeline/project_stream.py`
- Modify: `pipeline/fast_page_server.py`
- Modify: `worker/runner.py`
- Test: `pipeline/tests/test_project_stream.py`
- Test: `pipeline/tests/test_fast_page_server.py`

**Step 1: Write failing test for partial page materialization**

Create `pipeline/tests/test_project_stream.py`.

Test behavior:
- given `work_dir`, page number, original path, translated path, and text layers;
- writes/updates `project.partial.json`;
- preserves existing pages;
- records status per page.

Expected minimal shape:

```json
{
  "versao": "2.0",
  "app": "traduzai",
  "processing": {
    "status": "running",
    "pages_completed": 1
  },
  "paginas": [
    {
      "numero": 1,
      "status": "ready",
      "original_path": "originals/001.jpg",
      "translated_path": "translated/001.jpg",
      "text_layers": []
    }
  ]
}
```

**Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest pipeline\tests\test_project_stream.py -q
```

Expected:

```text
FAIL: module pipeline.project_stream not found
```

**Step 3: Implement `pipeline/project_stream.py`**

Add functions:

```python
def load_partial_project(work_dir: Path, base_config: dict) -> dict: ...
def upsert_page_result(project: dict, page: dict) -> dict: ...
def write_partial_project(work_dir: Path, project: dict) -> Path: ...
def mark_project_complete(work_dir: Path, project: dict) -> Path: ...
```

Use atomic writes via existing `project_writer.write_project_json_atomic` if available.

**Step 4: Emit partial project artifact**

In `pipeline/fast_page_server.py`, after each translated artifact is discovered:
- update `project.partial.json`;
- emit `page_completed` with:

```json
{
  "partial_project_path": "project.partial.json",
  "page_status": "ready"
}
```

**Step 5: Worker uploads partial project**

In `worker/runner.py`, when event contains `partial_project_path`, upload it as:

```text
kind = project_partial
```

**Step 6: Run tests**

Run:

```powershell
python -m pytest pipeline\tests\test_project_stream.py pipeline\tests\test_fast_page_server.py worker\tests\test_runner_artifacts.py -q
```

Expected:

```text
PASS
```

---

## Task 3: Add Page QA Scoring

**Files:**
- Create: `pipeline/qa/page_quality.py`
- Modify: `pipeline/fast_page_server.py`
- Modify: `pipeline/main.py`
- Test: `pipeline/tests/test_page_quality.py`

**Step 1: Write failing tests for QA findings**

Create tests for:
- text too small;
- text bbox outside balloon/image;
- empty translated text;
- Korean source still present in translated text where not SFX/credit;
- very small rendered text relative to balloon;
- page with no text but no issue if it is cover/action-only.

Example expected finding:

```json
{
  "code": "text_too_small",
  "severity": "warning",
  "page": 40,
  "block_id": "text-2",
  "message": "Texto pequeno demais"
}
```

**Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest pipeline\tests\test_page_quality.py -q
```

Expected:

```text
FAIL: page_quality module not found
```

**Step 3: Implement deterministic QA**

Implement:

```python
def score_page_quality(page: dict, image_size: tuple[int, int]) -> dict:
    return {
        "status": "pass" | "review" | "fail",
        "score": 0.0,
        "findings": [...]
    }
```

Initial rules:
- font size below threshold -> `warning`;
- bbox overflow -> `critical`;
- empty translation with source text -> `critical`;
- untranslated Hangul in normal dialogue -> `warning`;
- over-shrunk text area -> `warning`;
- missing rendered file -> `critical`.

**Step 4: Store QA per page**

Add to page shape:

```json
"qa": {
  "status": "review",
  "score": 0.72,
  "findings": []
}
```

**Step 5: Emit QA in page events**

In `page_completed` event include:

```json
"qa_status": "pass",
"qa_score": 0.91,
"qa_findings_count": 0
```

**Step 6: Run tests**

Run:

```powershell
python -m pytest pipeline\tests\test_page_quality.py pipeline\tests\test_fast_page_server.py -q
```

Expected:

```text
PASS
```

---

## Task 4: Build Render Candidate Selection

**Files:**
- Create: `pipeline/typesetter/render_selector.py`
- Modify: `pipeline/typesetter/renderer.py`
- Modify: `pipeline/main.py`
- Test: `pipeline/tests/test_render_selector.py`
- Test: `pipeline/tests/test_typesetting_renderer.py`

**Step 1: Write failing tests for candidate choice**

Test cases:
- candidate A has smaller font and candidate B fits better -> choose B;
- candidate A and B both pass -> choose faster/current;
- candidate with overflow is rejected;
- candidate with missing text is rejected;
- selector records reasons.

Expected:

```python
decision = select_render_candidate([fast, quality])
assert decision.selected == "quality"
assert "text_too_small" in decision.reasons
```

**Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest pipeline\tests\test_render_selector.py -q
```

Expected:

```text
FAIL: render_selector module not found
```

**Step 3: Implement candidate model**

Add dataclasses:

```python
@dataclass
class RenderCandidate:
    name: str
    image_path: Path
    page: dict
    quality: dict
    elapsed_ms: int

@dataclass
class RenderDecision:
    selected: str
    image_path: Path
    score: float
    reasons: list[str]
```

**Step 4: Generate two candidates only when needed**

Default:
- render fast once;
- if QA passes, keep it;
- if QA is `review` or `fail`, render quality candidate.

Quality candidate should use stricter typesetting:
- larger minimum font;
- stronger line wrapping;
- avoid over-compressing text;
- prefer multi-line readable text over tiny single block.

**Step 5: Persist candidate metadata**

In page:

```json
"render_decision": {
  "selected": "fast",
  "candidates": [
    {"name": "fast", "score": 0.86},
    {"name": "quality", "score": 0.92}
  ],
  "reasons": []
}
```

**Step 6: Run tests**

Run:

```powershell
python -m pytest pipeline\tests\test_render_selector.py pipeline\tests\test_typesetting_renderer.py -q
```

Expected:

```text
PASS
```

---

## Task 5: Improve Typesetting Quality Defaults

**Files:**
- Modify: `pipeline/typesetter/renderer.py`
- Modify: `pipeline/typesetter/style_policy.py`
- Modify: `pipeline/layout/balloon_layout.py`
- Test: `pipeline/tests/test_typesetting_fit_qa.py`
- Test: `pipeline/tests/test_typesetting_renderer.py`
- Test: `pipeline/tests/test_typesetting_style_policy.py`

**Step 1: Add tests for readable font floor**

Test:
- dialogue balloons should not shrink below readable threshold unless explicitly marked as tiny/SFX;
- long text should wrap before shrinking too far;
- narrow balloons may use more lines.

**Step 2: Run failing tests**

Run:

```powershell
python -m pytest pipeline\tests\test_typesetting_fit_qa.py -q
```

Expected:

```text
FAIL on new readable font floor assertions
```

**Step 3: Implement stricter fit strategy**

In `renderer.py`, prefer this order:

1. normalize text;
2. choose line breaks;
3. fit with font size floor;
4. if impossible, mark QA `needs_review`;
5. do not silently make unreadably tiny text.

**Step 4: Keep renderer deterministic**

Do not let VLM or LLM directly render pixels. VLM may only produce JSON guidance:

```json
{
  "preferred_bbox": [x, y, w, h],
  "line_break_hint": ["line 1", "line 2"],
  "avoid_regions": []
}
```

**Step 5: Run focused tests**

Run:

```powershell
python -m pytest pipeline\tests\test_typesetting_fit_qa.py pipeline\tests\test_typesetting_renderer.py pipeline\tests\test_typesetting_style_policy.py -q
```

Expected:

```text
PASS
```

---

## Task 6: Add Optional VLM Guidance Layer

**Files:**
- Modify: `pipeline/layout/balloon_layout.py`
- Modify: `pipeline/main.py`
- Test: `pipeline/tests/test_layout_analysis.py`
- Test: `pipeline/tests/test_typesetting_layout.py`

**Step 1: Write tests for VLM JSON guidance merge**

Test:
- VLM guidance can adjust text bbox within safe bounds;
- invalid JSON is ignored;
- guidance outside image is rejected;
- deterministic layout remains fallback.

**Step 2: Enable model family gate**

Extend inline-image whitelist to include `qwen3-vl` when configured.

**Step 3: Add config flag**

Use:

```json
"layout_guidance": {
  "enabled": false,
  "provider": "ollama",
  "model": "qwen3-vl:8b"
}
```

Default remains off until QA proves it helps.

**Step 4: Persist decisions**

Write in page:

```json
"layout_guidance": {
  "used": true,
  "model": "qwen3-vl:8b",
  "accepted": true
}
```

**Step 5: Run tests**

Run:

```powershell
python -m pytest pipeline\tests\test_layout_analysis.py pipeline\tests\test_typesetting_layout.py -q
```

Expected:

```text
PASS
```

---

## Task 7: Worker/API Support for Live Page Readiness

**Files:**
- Modify: `worker/runner.py`
- Modify: `worker/__main__.py`
- Modify: `server/app.py`
- Test: `worker/tests/test_runner_artifacts.py`
- Test: `worker/tests/test_worker_run_once.py`
- Test: `server/tests/test_worker_api.py`

**Step 1: Add server test for page artifact event**

In `server/tests/test_worker_api.py`, assert:
- worker uploads `translated_image`;
- worker posts page event with QA fields;
- job detail exposes page-level artifacts/status.

**Step 2: Run failing test**

Run:

```powershell
python -m pytest server\tests\test_worker_api.py::test_worker_page_completed_updates_page_status -q
```

Expected:

```text
FAIL: endpoint/status field missing
```

**Step 3: Store page status in job metadata**

In `server/app.py`, persist:

```json
"pages": {
  "40": {
    "status": "ready",
    "qa_status": "pass",
    "artifact_id": "..."
  }
}
```

**Step 4: Expose page status in API**

Add to job detail response:

```json
"page_status": [...]
```

**Step 5: Run tests**

Run:

```powershell
python -m pytest server\tests\test_worker_api.py worker\tests -q
```

Expected:

```text
PASS
```

---

## Task 8: UI Review Queue

**Files:**
- Modify: `src/pages/Processing.tsx`
- Modify: `src/lib/stores/appStore.ts`
- Modify: `src/lib/tauri.ts`
- Test: `src/lib/**/__tests__/**/*.test.ts`
- Test: `e2e/editor-rebuild.spec.ts`

**Step 1: Add UI state tests**

Test:
- page ready event updates page card;
- QA review page appears in review filter;
- clicking ready page opens preview/editor path;
- completed job still opens final export.

**Step 2: Add Processing page UI**

Show:
- grid/list of pages;
- status: `Processando`, `Pronta`, `Revisar`, `Falhou`;
- QA badges: `Texto pequeno`, `OCR duvidoso`, `Balão vazio`;
- latest translated thumbnail.

No marketing copy. Keep interface operational and dense.

**Step 3: Add review filters**

Filters:
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

Expected:

```text
PASS
```

---

## Task 9: Export Final CBZ from Best Pages

**Files:**
- Create: `pipeline/exporter/chapter_export.py`
- Modify: `server/app.py`
- Modify: `worker/runner.py`
- Test: `pipeline/tests/test_chapter_export.py`
- Test: `server/tests/test_worker_api.py`

**Step 1: Write export tests**

Test:
- exports selected rendered pages in numeric order;
- skips temp candidates;
- includes final chosen image;
- preserves extension normalization.

**Step 2: Implement exporter**

Add:

```python
def export_translated_cbz(project_json_path: Path, output_path: Path) -> Path:
    ...
```

**Step 3: Call exporter on complete**

When job completes:
- export `translated.cbz`;
- upload artifact kind `translated_cbz`;
- keep individual page images.

**Step 4: Run tests**

Run:

```powershell
python -m pytest pipeline\tests\test_chapter_export.py server\tests\test_worker_api.py -q
```

Expected:

```text
PASS
```

---

## Task 10: Benchmark and Visual Regression Harness

**Files:**
- Create: `pipeline/scripts/benchmark_fast_page_quality.py`
- Create: `pipeline/tests/test_fast_page_quality_benchmark.py`
- Modify: `.gitignore`

**Step 1: Build benchmark script**

Inputs:

```powershell
python pipeline\scripts\benchmark_fast_page_quality.py `
  --source "N:\TraduzAI\TraduzAi\exemplos\exemploko\환생천마\113화.cbz" `
  --out "data\fast-page-quality-benchmarks"
```

Outputs:
- `summary.json`;
- `samples.jpg`;
- `translated.cbz`;
- per-page timing;
- QA distribution.

**Step 2: Record metrics**

Required metrics:
- time to first page;
- full chapter time;
- pages/sec;
- `qa_pass`;
- `qa_review`;
- `qa_fail`;
- pages rerendered;
- render selector choices.

**Step 3: Add non-flaky test**

Unit-test the summary parser and sample selection. Do not run full chapter in CI.

**Step 4: Run benchmark manually**

Run on the same 113 chapter and compare against:
- current full-chapter result: `60` pages, `228.014s`;
- previous measured serial baseline where available.

---

## Task 11: Rollout Flags and Safe Defaults

**Files:**
- Modify: `.env.example`
- Modify: `worker/config.py`
- Modify: `pipeline/runtime_profiles.py`
- Test: `worker/tests/test_worker_config.py`
- Test: `pipeline/tests/test_runtime_profiles.py`

**Flags:**

```text
TRADUZAI_FAST_PAGE_SERVER=1
TRADUZAI_PAGE_QA=1
TRADUZAI_RENDER_SELECTOR=1
TRADUZAI_QUALITY_RERENDER=1
TRADUZAI_LAYOUT_GUIDANCE=0
```

**Policy:**
- `fast-page` default on.
- deterministic QA default on.
- render selector default on after tests pass.
- VLM guidance default off until manually validated.
- legacy runner remains available with `TRADUZAI_FAST_PAGE_SERVER=0`.

**Run:**

```powershell
python -m pytest worker\tests\test_worker_config.py pipeline\tests\test_runtime_profiles.py -q
```

Expected:

```text
PASS
```

---

## Definition of Done

The rollout is complete when all are true:

- `python -m pytest pipeline\tests\test_fast_page_server.py worker\tests server\tests\test_worker_api.py -q` passes.
- `npm run check` passes.
- A full chapter run creates:
  - partial page events;
  - final `project.json`;
  - `translated.cbz`;
  - QA summary;
  - sample sheet.
- Time to first useful page is under `45s` on current local machine.
- Full `113화.cbz` stays near or below the measured `228s` unless quality rerender is intentionally enabled.
- At least `80%` of pages auto-pass QA on the 113 chapter.
- Pages marked `review` open directly in the editor with the correct page selected.

## Recommended Execution Order

1. Task 1: freeze fast-page contract.
2. Task 2: partial project state.
3. Task 3: deterministic QA.
4. Task 5: improve text fit quality.
5. Task 4: render selector.
6. Task 7: worker/API page status.
7. Task 8: UI review queue.
8. Task 9: final CBZ export.
9. Task 10: benchmark harness.
10. Task 11: rollout flags.

## Risks

- QA thresholds can become too aggressive and send too many pages to review.
- Quality rerender can erase the speed gain if it runs on every page.
- Unicode paths on Windows must stay ASCII-escaped over JSONL.
- VLM guidance can improve placement but must never directly override deterministic renderer safety checks.
- Existing dirty/untracked worker/server directories should be handled carefully when committing.

## First Implementation Slice

Start with Tasks 1-3 only. That gives:
- stable hot runner contract;
- partial `project.json`;
- page QA status.

This is the smallest useful product increment because it lets the UI show pages immediately and identify which ones need review before we change renderer behavior.

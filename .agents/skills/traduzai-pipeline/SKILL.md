---
name: traduzai-pipeline
description: Use when TraduzAI finishes but QA blocks the preview, reports success with critical issues, exits differently in strict mode, loses completion state between Python, Rust and React, or diverges between individual and batch completion.
---

# TraduzAI Pipeline

## Owners e fluxo automático

Owners: `pipeline/main.py`, `pipeline/strip/run.py`, `pipeline/strip/process_bands.py`, `pipeline/qa/export_gate.py`, `pipeline/project_writer.py`, `src-tauri/src/commands/pipeline.rs`, `src/lib/tauri.ts`, `src/lib/pipelineCompletion.ts`, `src/lib/stores/appStore.ts` e `src/pages/Processing.tsx`.

Rust inicia o sidecar Python; `main.py` → `run_chapter` → `process_band` → stages → `project.json`/QA/export gate. Rust lê os artefatos e emite `pipeline-complete` para store/tela React.

## Contrato de conclusão

- `success` é sucesso **técnico** do processo/IPC; não aprova a saída.
- `completion_status`: `approved | blocked | overridden | error`. Nunca use `success` como valor.
- `output_review_state`: `approved | blocked_preview | overridden`, persistido no projeto.
- `qa.export_gate.status`: `PASS | REVIEW | BLOCK | OVERRIDDEN`; é a autoridade visual junto de issues/contagens.

Em execução normal, `BLOCK` salva preview, emite `complete`/exit 0 e o Rust publica `success: true`, `completion_status: blocked`. Com `strict` ou `export_mode: strict`, somente `BLOCK` emite `error`/exit 2. `REVIEW` sai do Rust como `completion_status: approved`, preservando `needs_review` e contagens; não confunda com ausência de revisão.

O Rust resolve o resumo nesta ordem: `project.json` (`qa.export_gate`) → `qa_report.json` → fallback `PASS`. Erro do sidecar gera `success: false` e `completion_status: error`.

## Triagem e divergência conhecida

Individual: `pipelineCompletion.ts` deriva `done_blocked`; `ChapterCompletionScreen` mostra “Preview bloqueado”, issues e mantém revisão/editor.

Batch: `BatchCompletionScreen` mostra ícones verdes; `openBatchChapter` força `status: done` e perde QA/completion. **Important — lacuna de teste:** Vitest cobre só derivação individual; Playwright cobre só navegação do batch, não capítulo bloqueado.

## Artefatos e primeira divergência

- Raiz: `project.json`, `qa_report.json`/`.md`, `decision_trace.jsonl`, `performance_timing.json` e imagens.
- `debug/e2e/00_run`: config, ambiente, argumentos e timing.
- `debug/e2e/11_qa_export_gate`: `export_gate.json`, `qa_export_gate_consistency.json`, `qa_flag_propagation_audit.json`, `final_rerender_visual_qa.json`/`.jsonl`; `strict_exit_audit.json` existe apenas no BLOCK strict.

Compare primeiro `project.json` com `qa_report.json`; depois payload Rust, evento, store e tela. Não conclua pelo último log isolado.

## Testes focados

```powershell
Push-Location pipeline
try { python -m pytest tests/test_main_emit.py::MainEmitTests::test_runner_cli_strict_returns_nonzero_when_mock_has_critical_flag tests/test_main_emit.py::MainEmitTests::test_runner_cli_mock_critical_persists_blocked_preview_without_path_migration -q } finally { Pop-Location }
cargo test --manifest-path src-tauri/Cargo.toml commands::pipeline::tests::pipeline_complete_payload
npx vitest run src/lib/__tests__/pipelineCompletion.test.ts
npx playwright test e2e/editor-rebuild.spec.ts -g "processing final de lote mostra capitulos e volta do preview"
```

## Fronteiras e checklist

Use `traduzai-detect`, `traduzai-ocr`, `traduzai-inpaint` ou `traduzai-typesetting` para a primeira etapa divergente; esta skill possui orquestração, persistência, QA/export gate e entrega ao app. Use `mangatl-dev` para coordenação geral.

- Registre primeira divergência, `trace_id`, gate e modo strict/normal.
- Valide saída visual, contrato Python→Rust→frontend, individual e batch.
- Preserve checkout sujo; rode nodeids, auditor, UTF-8 e `git diff --check`.

## Última verificação

- Data: 2026-07-24
- Runtime-base: `c0801348`

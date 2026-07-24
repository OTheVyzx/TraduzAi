---
name: traduzai-translation
description: Use when TraduzAI sends the wrong OCR text to translation, preserves untranslated source, mistranslates names or glossary terms, loses placeholders, skips the wrong route_action, repeats fragments, or falls through after Google health failures.
---

# TraduzAI Translation

## Owners, contrato e fluxo

Owners: `pipeline/translator/translate.py`, `pipeline/translator/term_protection.py`, `pipeline/strip/process_bands.py`, `pipeline/ocr/ocr_normalizer.py`, `pipeline/ocr/contextual_reviewer.py`, `pipeline/ocr/text_router.py` (gate), `pipeline/main.py` e `pipeline/qa/export_gate.py`.

Band OCR/review → `_finalize_ocr_page_before_translation` → `_run_translate_stage` → `translate_pages` → normalização/merge → Google ou passthrough → `_merge_translated_page_metadata` → project/QA.

Contrato: preserve `texts`, IDs/trace/metadados; produza `original`, `translated`, `source_text_sent_to_translator`, reparos e `qa_flags`.

## Seleção, gate e merge

`_source_text_before_normalization`: reparo especial de lóbulo; senão `raw_ocr` → `original` → `text`. `_source_text_for_translation` usa `normalized_text_final` alterado apenas com confiança ≥ 0,7; senão `text`, depois raw. Atenção: se upstream já sobrescreveu `text` com o normalizado, o fallback pode contornar o gate baixo.

`route_action` é o gate autoritativo via `route_action_requires_translation`; campos legacy de skip não substituem essa decisão. Antes do envio, `normalize_ocr_record` e `merge_same_balloon_fragments_before_translation` fazem merge/dedupe e preservam `source_text_ids`/`source_trace_ids`; o wrapper mescla a resposta de volta ao snapshot OCR.

## Backend e proteção

- Google: health probe, cooldown, cache e três tentativas; lote cai por item se o split falhar. Chunks paralelos opt-in deduplicam em ordem.
- Sem Google saudável, `_resolve_translation_backend` retorna `passthrough`; não há fallback automático para Ollama.
- Glossário é ativo em normalização, memória, proteção/restore de placeholders e locks pós-processamento. `translation_context` rico é distinto: hoje não chega às requisições Google automáticas.
- Proteja termos antes do backend; restaure placeholders, aplique entity/name locks e pós-processamento; placeholder perdido gera `unrestored_placeholder`.

O backend Ollama e o semantic review por LLM existem como entrypoints, mas `translate_pages` fixa o status Ollama como indisponível e `semantic_review_requested = False`. Não prometa execução/fallback. O pós-processamento determinístico `_review_translation_grammar_semantics` continua separado e ativo.

## Primeira divergência e artefatos

| Evidência | Investigue |
|---|---|
| `03_ocr/ocr_raw_blocks.jsonl` errado | OCR/detect |
| `04_text_normalization_router/*` muda indevidamente | review/normalizer/merge |
| `07_translation/translation_inputs.jsonl` ausente/errado | source precedence ou `route_action` |
| input correto, `translation_outputs.jsonl` errado | Google, proteção, restore ou pós-processo |
| `decision_trace.jsonl` diverge | policy/reason em `record_decision` |
| `07` correto, `project.json`/QA errado | merge, persistência ou export gate |

Use também `translation_debug_summary.json`, `glossary_application.jsonl` e `translation_fallbacks.jsonl`.

## Testes e REDs pendentes

```powershell
Push-Location pipeline
try {
  python -m pytest tests/test_translate_context.py::TranslateContextTests::test_should_skip_translation_item_ignores_legacy_skip_fields tests/test_translate_context.py::TranslateContextTests::test_translate_pages_does_not_call_ollama_when_google_health_fails tests/test_translate_context.py::TranslateContextTests::test_resolve_translation_backend_does_not_fallback_to_ollama tests/test_translate_context.py::TranslateContextTests::test_translate_pages_locks_glossary_terms_inside_sentence -q
  python -m pytest tests/test_normalized_text_propagates_to_translation.py::test_translator_uses_confident_normalized_text_final tests/test_normalized_text_propagates_to_translation.py::test_same_balloon_fragments_are_repaired_before_translation tests/test_translate_context.py::TranslateContextTests::test_translate_pages_flags_dropped_placeholder_as_unrestored -q
} finally { Pop-Location }
```

REDs pendentes: impedir bypass quando `text` já contém normalizado de baixa confiança; definir se contexto rico alcançará Google; `test_translate_pages_writes_complete_translation_debug_artifacts` espera 2 inputs por skip legacy, mas o runtime autoritativo gera 4. Header isolado não prova integração.

## Fronteiras e checklist

OCR possui texto/rota; translation possui seleção, backend, termos e saída; pipeline possui merge/QA; typesetting apenas renderiza. Registre primeira divergência, texto efetivamente enviado, rota, backend/fallback e trace; valide nodeids, artefatos, visual downstream, UTF-8, auditor, diff e checkout sujo.

## Última verificação

- Data: 2026-07-24
- Runtime-base: `c0801348`

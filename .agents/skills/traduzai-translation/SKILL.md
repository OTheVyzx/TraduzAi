---
name: traduzai-translation
description: Use when TraduzAI applies a translation route incorrectly, sends the wrong accepted text to the backend, loses translation outputs during merge, preserves untranslated text, mistranslates protected terms, loses placeholders, or falls through after Google health failures.
---

# TraduzAI Translation

## Owners, contrato e fluxo

Owners: `pipeline/translator/translate.py`, `pipeline/translator/term_protection.py` e o trecho de tradução de `pipeline/strip/process_bands.py`: precedência do texto aceito, consumo/aplicação de `route_action`, backend, outputs e `_merge_translated_page_metadata`.

OCR é owner de normalização, review, `text_router` e criação de `route_action`; rota criada errada pertence a `traduzai-ocr`. Pipeline é owner de `main.py`, orquestração, QA e export. São fronteiras upstream/downstream, não owners desta skill.

Band OCR/review → `_finalize_ocr_page_before_translation` → `_run_translate_stage` → `translate_pages` → normalização/merge → Google ou passthrough → `_merge_translated_page_metadata` → project/QA.

Contrato: preserve `texts`, IDs/trace/metadados; produza `original`, `translated`, `source_text_sent_to_translator`, reparos e `qa_flags`.

## Seleção, gate e merge

`_source_text_before_normalization`: reparo especial de lóbulo; senão `raw_ocr` → `original` → `text`. `_source_text_for_translation` usa `normalized_text_final` alterado apenas com confiança ≥ 0,7; senão `text`, depois raw. Atenção: se upstream já sobrescreveu `text` com o normalizado, o fallback pode contornar o gate baixo.

`route_action` é o gate autoritativo; skips legacy não o substituem. `normalize_ocr_record` e `merge_same_balloon_fragments_before_translation` fazem merge/dedupe. `_merge_translated_page_metadata` percorre só `translated_texts`: OCR sem output correspondente pode desaparecer. Exija paridade por `trace_id` entre inputs → outputs → project.

## Backend e proteção

- Google: health probe, cooldown, cache e três tentativas; lote cai por item se o split falhar. Chunks paralelos opt-in deduplicam em ordem.
- Sem Google saudável, `_resolve_translation_backend` retorna `passthrough`; não há fallback automático para Ollama.
- Glossário é ativo em normalização, memória, proteção/restore de placeholders e locks pós-processamento. `translation_context` rico é distinto: hoje não chega às requisições Google automáticas.
- Proteja termos antes do backend; restaure placeholders, aplique entity/name locks e pós-processamento; placeholder perdido gera `unrestored_placeholder`.

Ollama/semantic review LLM são entrypoints inativos: `translate_pages` fixa Ollama indisponível e `semantic_review_requested = False`. Não prometa fallback. `_review_translation_grammar_semantics` é pós-processo determinístico ativo.

## Primeira divergência e artefatos

| Evidência | Investigue |
|---|---|
| `03_ocr/ocr_raw_blocks.jsonl` errado | `traduzai-ocr`/`traduzai-detect` |
| `04_text_normalization_router/*` ou `route_action` criada errada | `traduzai-ocr` |
| rota correta, mas `07_translation/translation_inputs.jsonl` ausente/errado | aplicação do gate ou source precedence desta skill |
| input correto, `translation_outputs.jsonl` errado | Google, proteção, restore ou pós-processo |
| `decision_trace.jsonl` diverge | policy/reason em `record_decision` |
| output correto, merge em `project.json` perde tradução | merge desta skill |
| merge correto, QA/export diverge | `traduzai-pipeline` |

Use também `07_translation/translation_debug_summary.json`, `07_translation/glossary_application.jsonl` e `07_translation/translation_fallbacks.jsonl`.

## Testes e REDs pendentes

```powershell
Push-Location pipeline
try {
  python -m pytest tests/test_translate_context.py::TranslateContextTests::test_should_skip_translation_item_ignores_legacy_skip_fields tests/test_translate_context.py::TranslateContextTests::test_translate_pages_does_not_call_ollama_when_google_health_fails tests/test_translate_context.py::TranslateContextTests::test_resolve_translation_backend_does_not_fallback_to_ollama tests/test_translate_context.py::TranslateContextTests::test_translate_pages_locks_glossary_terms_inside_sentence -q
  python -m pytest tests/test_normalized_text_propagates_to_translation.py::test_translator_uses_confident_normalized_text_final tests/test_normalized_text_propagates_to_translation.py::test_same_balloon_fragments_are_repaired_before_translation tests/test_translate_context.py::TranslateContextTests::test_translate_pages_flags_dropped_placeholder_as_unrestored -q
} finally { Pop-Location }
```

REDs pendentes: preservar/sinalizar OCR sem output e testar paridade por trace; impedir bypass de normalizado baixo; definir se contexto rico alcançará Google; o teste de debug espera 2 skips legacy, mas o runtime gera 4. Header isolado não prova integração.

## Fronteiras e checklist

OCR possui normalização/review/router e cria a rota; translation consome o gate e possui text-precedence, backend, outputs, termos e merge; pipeline possui main/orquestração/QA/export. Registre primeira divergência, texto enviado, rota aplicada, backend e trace; valide nodeids, artefatos, UTF-8, auditor, diff e checkout sujo.

## Última verificação

- Data: 2026-07-24
- Runtime-base: `c0801348`

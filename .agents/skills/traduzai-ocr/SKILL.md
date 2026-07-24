---
name: traduzai-ocr
description: Use when TraduzAI OCR text is wrong, disappears after recognition, is normalized unexpectedly, or diverges in review, language, SFX, watermark, scanlation-credit, route_action, or route_reason behavior.
---

# TraduzAI OCR

## Escopo e owner

Esta skill é dona do reconhecimento, limpeza/normalização, review, classificação e contrato de roteamento OCR. Detect é dona da geometria; tradução apenas consome o texto e aplica o gate; QA/export é downstream. Em falhas multietapa, use também `mangatl-dev`.

Entrypoints: `pipeline/vision_stack/runtime.py`, `pipeline/ocr/` e `pipeline/strip/process_bands.py`. `pipeline/translator/translate.py` é consumer do contrato.

## Fluxo ativo por bandas

`process_band` → `run_ocr_stage` / `build_page_result` → `_record_ocr_raw_blocks` → `_run_review_layout_stage` / `contextual_review_page` → `_finalize_ocr_page_before_translation` / normalizers → `translate_pages` → `normalize_ocr_record` novamente → `_should_skip_translation_item` / gate de `route_action`.

## Quick reference: primeira divergência

| Primeira evidência divergente | Owner provável |
|---|---|
| Ausente/incorreta antes de `03_ocr/ocr_raw_blocks.jsonl` | backend OCR, cleanup ou `build_page_result` |
| Correta em `03_ocr`, alterada em `04_text_normalization_router/*` | reviewer, normalizer ou `text_router` |
| Correta em `04`, ausente de `07_translation/translation_inputs.jsonl` | `route_action`/`route_reason`; gate de tradução |
| Presente em inputs, incorreta em `translation_outputs.jsonl`/`translation_debug_summary.json` | tradução/backend; use `traduzai-translation` |
| Correta em `07`, divergente em `project.json` ou `export_gate` | serialização/QA downstream; use `mangatl-dev` |

`03_ocr` prova aceitação no momento do OCR, antes da normalização e do gate; não contém todas as tentativas. `decision_trace.jsonl` localiza drops e decisões. `04` explica reparos; `07` prova o que cruzou o gate.

## Contrato e invariantes

- Preserve o shape: `bbox`, `text`, `raw_ocr`, `normalized_ocr`, `normalized_text_final`, `original` quando usado, `confidence`, `tipo`/`type`, `route_action`, `route_reason`, `needs_review`, `qa_flags` e `skip_processing` legado.
- `route_action` é autoritativo. `skip_processing` é legado/derivado e pode divergir; `review_required` não traduz mesmo com `skip_processing: false`.
- Decida a causa por `route_reason` e traces: `scanlation_credit_suppressed`, fragmento de arte, scene text, texto truncado/joined. Não use apenas confidence/skip.
- A tradução reexecuta `normalize_ocr_record` antes do gate. Watermark, não-inglês, SFX e créditos podem mudar tradução, inpaint e render.
- Não assuma `review_reason` como campo top-level; procure `route_reason`, `qa_flags` e metadata/trace de normalização.

## Diagnóstico antes de patch

Siga um `text_id`/`trace_id` pelos artefatos, encontre a primeira divergência e compare texto, rota, motivo e flags. Só então altere threshold, classifier, reviewer ou normalizer. Tradução não cria a semântica OCR.

## Testes focados

```powershell
Push-Location pipeline
try {
  python -m pytest tests/test_vision_stack_ocr.py::VisionStackOCRTests::test_normalize_paddleocr_language_handles_regions_and_common_languages -q
  python -m pytest tests/test_ocr_normalizer.py -k "scanlation_credit_overrides_translate_route_action or short_consonant_art_fragment_routes_to_review_without_skip or standard_scene_text_routes_to_review_before_inpaint or record_persists_raw_normalized_and_reason" -q
  python -m pytest tests/test_contextual_reviewer.py::ContextualReviewerTests::test_uses_page_lexicon_to_fix_low_confidence_name tests/test_text_normalization_debug.py::test_contextual_review_applies_normalization_trace_before_translation -q
} finally { Pop-Location }
```

Mapa adicional: `test_vision_stack_ocr.py` cobre backend, idioma e mapping; `test_ocr_reviewer.py`, escolha primary/fallback; testes `translation`/`debug`, o consumer e seus artefatos. Selecione nodeid/`-k`, não o arquivo inteiro.

Se um teste de debug de tradução esperar skip baseado apenas em `skip_processing`, trate primeiro como possível expectativa legada do teste, não como falha confirmada do produto.

## Última verificação

- Data: 2026-07-24
- Runtime-base: `c0801348`

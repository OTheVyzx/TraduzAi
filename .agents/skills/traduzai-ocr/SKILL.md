---
name: traduzai-ocr
description: Use when TraduzAI OCR text is wrong, disappears, changes during normalization, or diverges in review, language, SFX, watermark, credits, route_action, or route_reason.
---

# TraduzAI OCR

## Owner e fronteiras

Owner de reconhecimento, normalização, review, classificação e roteamento: `pipeline/vision_stack/runtime.py`, `pipeline/vision_stack/ocr.py`, `pipeline/ocr/detector.py` e `pipeline/ocr/`. `process_bands.py` orquestra; `translate.py` consome.

Detect fornece geometria; tradução aplica o gate, não cria semântica OCR; QA/export é downstream. Em problema multietapa do pipeline automático, use `traduzai-pipeline` junto da especialista afetada; use `mangatl-dev` para coordenação geral.

## Fluxo ativo por bandas

`process_band` → `_run_band_ocr_stage` → `run_ocr_stage` / `build_page_result` → `_record_ocr_raw_blocks` → `_run_review_layout_stage` / `contextual_review_page` → `_finalize_ocr_page_before_translation` / normalizers → `_run_translate_stage` → `translate_pages` → `normalize_ocr_record` novamente → `_should_skip_translation_item`.

Entrada: imagem/crop, `_vision_blocks` com `bbox`, idioma/preset e contexto. Saída: página com `texts` e `_vision_blocks` alinhados. Preserve `bbox`, `text`, `raw_ocr`, `normalized_ocr`, `normalized_text_final`, `original`, `confidence`, `tipo`/`type`, `route_action`, `route_reason`, `needs_review`, `qa_flags` e `skip_processing` legado.

## Backend, idioma e texto enviado

`run_ocr` usa o stack ativo e falha fechado, sem legacy/EasyOCR. MangaOCR indisponível cai para PaddleOCR. OCR regional com preset tenta o stack e pode cair para crop Paddle; EasyOCR está desativado. `normalize_paddleocr_language` resolve aliases/regiões (`en-US`, `pt-BR`, `zh-CN`, `zh-TW`, `ja`, `ko`); confira o mapping EasyOCR antes de mudar legado.

Após nova `normalize_ocr_record`, `_source_text_for_translation` escolhe: raw para `leading_dark_lobe_duplicate_fragment_removed`; senão `normalized_text_final` alterado com confiança ≥0,7; senão `text`; por último raw, cuja precedência é `raw_ocr` → `original` → `text`.

## Quick reference: primeira divergência

| Primeira evidência divergente | Owner provável |
|---|---|
| Ausente/incorreta antes de `03_ocr/ocr_raw_blocks.jsonl` | backend OCR, cleanup ou `build_page_result` |
| Correta em `03_ocr`, alterada em `04_text_normalization_router/*` | reviewer, normalizer ou `text_router` |
| Correta em `04`, ausente de `07_translation/translation_inputs.jsonl` | `route_action`/`route_reason`; gate de tradução |
| Presente em inputs, incorreta em `translation_outputs.jsonl`/`translation_debug_summary.json` | tradução/backend; use `traduzai-translation` |
| Correta em `07`, divergente em `project.json`/`export_gate` | serialização/QA; use `traduzai-pipeline` + especialista |

`03_ocr` prova aceitação antes da normalização/gate, não todas as tentativas. `decision_trace.jsonl` localiza drops; `04` explica reparos; `07` prova o que cruzou o gate.

## Invariantes e diagnóstico

- `route_action` é autoritativo. `skip_processing` é legado/derivado e pode divergir; `review_required` não traduz mesmo com `skip_processing: false`.
- Decida por `route_reason`/traces (`scanlation_credit_suppressed`, art fragment, scene text, truncated/joined), não apenas confidence/skip. `review_reason` não é contrato top-level.
- Preserve shape/alinhamento. Watermark, não-inglês, SFX e crédito alteram tradução, inpaint e render.

Siga `text_id`/`trace_id` até a primeira divergência; só então altere threshold, classifier, reviewer ou normalizer.

## Testes focados

```powershell
Push-Location pipeline
try {
  python -m pytest tests/test_vision_stack_ocr.py::VisionStackOCRTests::test_normalize_paddleocr_language_handles_regions_and_common_languages -q
  python -m pytest tests/test_ocr_normalizer.py -k "scanlation_credit_overrides_translate_route_action or short_consonant_art_fragment_routes_to_review_without_skip or standard_scene_text_routes_to_review_before_inpaint or record_persists_raw_normalized_and_reason" -q
  python -m pytest tests/test_contextual_reviewer.py::ContextualReviewerTests::test_uses_page_lexicon_to_fix_low_confidence_name tests/test_text_normalization_debug.py::test_contextual_review_applies_normalization_trace_before_translation -q
} finally { Pop-Location }
```

`test_ocr_reviewer.py` cobre primary/fallback; testes `translation`/`debug`, o consumer. Se expectativa usar só `skip_processing`, trate-a primeiro como possível legado, não falha confirmada.

## Checklist de encerramento

- Primeiro artefato divergente e owner registrados.
- Entrada/saída, aliases, shape, `route_action` e consumidores vizinhos preservados.
- Nodeids focados, auditor, UTF-8 e `git diff --check` validados.

## Última verificação

- Data: 2026-07-24
- Runtime-base: `c0801348`

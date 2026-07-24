---
name: traduzai-typesetting
description: Use when TraduzAI renders translated text with overflow, tiny or unreadable fonts, wrong wrapping, unsafe placement, broken connected balloons, incorrect source-style copying, or divergent typesetting render-plan artifacts.
---

# TraduzAI Typesetting

## Owner, contrato e fluxo ativo

Owners: `pipeline/typesetter/renderer.py`, `pipeline/typesetter/style_policy.py`, `pipeline/layout/balloon_layout.py` e `fonts/font-map.json`; `pipeline/strip/process_bands.py` orquestra.

`pipeline/main.py` → `run_chapter` → `process_band` → `_run_review_layout_stage` / `enrich_page_layout` → evidência de estilo → `_run_typeset_stage` → `render_band_image` → `build_render_blocks` → `render_text_block` / sub-regiões conectadas → FT2Font → render plan.

Entrada: banda RGB limpa e página traduzida/revisada com `texts`, `_vision_blocks`, bboxes, safe boxes, perfil, estilo e metadados conectados. Saída: banda renderizada, geometria copiada aos contratos da página e registros do plano de render.

## Invariantes

- Preserve execução serial: `strip/run.py` cria `typeset_stage_lock`, `process_band` o consome e o batch sem overlap permanece serial. O cache FT2Font reduz recriação, mas **não garante segurança**; não paralelize o render.
- Style-copy exige `style_origin: source_detected` e confiança ≥ `SOURCE_STYLE_CONFIDENCE_THRESHOLD` (0,70). Abaixo disso, `normalize_auto_typesetting_style` aplica fonte/cor/efeitos automáticos conservadores.
- Balões conectados usam ordem, pesos/capacidade e safe boxes por lóbulo. Preserve texto completo e geometria pai/filhos; não divida semanticamente só para caber.
- Ajuste wrapping, espaçamento e tamanho dentro da capacidade. Overflow ou fonte abaixo da legibilidade mínima deve gerar fallback/skip/QA, não texto invisível nem invasão de arte.

## Primeira divergência

| Evidência | Investigue |
|---|---|
| `05_layout_geometry/layout_blocks.jsonl` incorreto | `balloon_layout`, capacidade e safe boxes |
| layout correto; `09_typeset/render_plan_raw.jsonl` incorreto | construção de blocos/estilo |
| raw correto; `render_plan_candidates.jsonl` ou `render_plan_skipped.jsonl` diverge | fit, overflow, conectados e gate |
| candidates corretos; `render_plan_final.jsonl` incorreto | dedupe, coordenadas e seleção final |
| `09_typeset/balloon_bbox_missing_audit.jsonl` registra item | propagação de `balloon_bbox`, geometria agregada e trace |
| plano correto; imagem ruim | FT2Font, rasterização, contraste e inspeção visual |

Preview técnico, teste verde ou render plan válido **não provam sucesso visual**. Inspecione a página renderizada no tamanho real e o export/QA.

`pipeline/debug_tools/style_audit_report.py` é auditor offline de style-copy; não participa do runtime.

## Fronteiras

Detect/OCR fornecem geometria e texto; inpaint fornece pixels limpos; typesetting decide layout, fit, estilo e rasterização. Copyback, QA e export são downstream. Em falha multietapa, combine esta skill com `traduzai-pipeline`; use `mangatl-dev` para coordenação.

## Testes focados

```powershell
Push-Location pipeline
try {
  python -m pytest tests/test_typesetting_style_policy.py::test_auto_style_reverts_low_confidence_detected_style_to_conservative_default tests/test_typesetting_renderer.py::TypesettingRendererTests::test_render_text_block_uses_connected_balloon_subregions tests/test_typesetting_renderer.py::TypesettingRendererTests::test_split_aggregate_drops_fit_flag_when_children_finish_ok -q
  python -m pytest tests/test_typesetting_renderer.py::TypesettingRendererTests::test_render_text_block_wraps_long_text_to_multiple_rows tests/test_typesetting_renderer.py::TypesettingRendererTests::test_render_text_block_rejects_underfit_white_run_safe_area tests/test_typesetting_renderer.py::TypesettingRendererTests::test_render_plan_records_candidates_and_skipped_with_trace_metadata -q
} finally { Pop-Location }
```

Lacuna conhecida: não há nodeid que afirme diretamente a propagação agregada de `below_minimum_legible` e `fit_below_minimum_legible` dos filhos conectados ao pai. Os dois testes verdes mais próximos são `test_render_text_block_uses_connected_balloon_subregions` e `test_split_aggregate_drops_fit_flag_when_children_finish_ok`; não os trate como cobertura positiva dessa integração.

## Checklist de encerramento

- Primeira divergência e owner registrados; raw/candidates/skipped/final correlacionados por trace.
- Safe boxes, conectados, wrapping, overflow, tamanho mínimo e contraste inspecionados visualmente no tamanho real.
- Nodeids, auditor, UTF-8 e `git diff --check` validados; checkout sujo preservado e mudanças preexistentes reportadas.

## Última verificação

- Data: 2026-07-24
- Runtime-base: `c0801348`

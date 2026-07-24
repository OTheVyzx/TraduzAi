---
name: traduzai-detect
description: Use when text or balloon boxes are missing, duplicated, oversized, clipped, shifted between band and page coordinates, or disappear from _vision_blocks before OCR.
---

# TraduzAI Detect

## Escopo e owner

Esta skill é dona da criação, cobertura e geometria dos candidatos de texto/balão e de sua transformação até o `_vision_blocks` inicial. OCR decide quais pares sobrevivem; inpaint transforma os blocos aceitos em máscara. Use `traduzai-ocr` ou `traduzai-inpaint` para esses estágios. Em falhas multietapa, use também `traduzai-pipeline`.

O fluxo ativo é:

`pipeline/main.py` → `pipeline/strip/run.py::run_chapter` → `pipeline/strip/detect_balloons.py::detect_strip_balloons` → `pipeline/strip/bands.py::group_balloons_into_bands` → `pipeline/strip/bands.py::attach_band_slices` → `pipeline/strip/process_bands.py::_band_to_page_dict` / `_run_band_ocr_stage` → `pipeline/vision_stack/runtime.py::run_ocr_stage` / `build_page_result`.

`pipeline/vision_stack/detector.py` fornece o backend alcançado por `_get_detector`; `runtime.py` consome caixas e monta resultados. Nenhum deles, isoladamente, é o owner do fluxo por bandas.

## Quick reference: onde a caixa sumiu?

| Evidência | Localize a perda em |
|---|---|
| Ausente em `_strip_debug` e `02_strip_detect/bands_manifest.json` | construção do strip, detecção ou agrupamento em bandas |
| Ausente em `02_strip_detect/detect_candidates.jsonl` | o modelo não emitiu ou o candidato foi eliminado antes da escrita; o arquivo contém somente aceitos pós-NMS/filtros |
| Presente em candidatos, sem match em `candidate_text_matching.jsonl` | OCR/pareamento da banda; confira `decision_trace.jsonl` |
| Aceita no trace e presente em `03_ocr/ocr_raw_blocks.jsonl` | finalização, remapeamento banda→página ou copyback |
| Presente na página, ausente em `08_inpaint/inpaint_blocks.jsonl` | fronteira detect→inpaint; carregue `traduzai-inpaint` |

`ocr_raw_blocks.jsonl` registra textos/blocos aceitos e prontos para layout; não é dump bruto de todas as tentativas. Confira ainda `10_copyback_reassemble/copyback_decisions.jsonl` e os artefatos de copyback.

## Contratos e invariantes

- Detecção produz candidatos antes de OCR. Não diagnostique caixa faltante apenas pelo resultado final.
- `strip_bbox` usa coordenadas do strip; `_band_to_page_dict` subtrai `band.y_top`; a reatribuição final converte para coordenadas de página.
- Shape mínimo verificável: dict com `bbox: [x1, y1, x2, y2]`, `x2 > x1`, `y2 > y1`, limites válidos no espaço declarado e `confidence` numérica.
- `_vision_blocks` muda de significado: começa em `band.balloons`; `build_page_result` usa `zip(blocks, texts)` e filtros, portanto termina apenas com pares sobreviventes.
- Preserve o fast path sem texto: resultado vazio/sinalizado não autoriza inventar candidatos.
- Caixa perdida ou larga contamina OCR, layout, copyback e inpaint.

## Diagnóstico antes de patch

1. Reproduza com debug e siga um bbox pelos artefatos na ordem acima.
2. Confirme o espaço de coordenadas e o primeiro estágio onde diverge.
3. Leia caller e consumer do estágio; só então ajuste threshold, NMS ou filtros.

Não comece por `_run_detect_ocr_on_image`, não trate `detect_candidates` como saída crua do modelo e não corrija detecção em OCR/inpaint.

## Testes focados

```powershell
rg -n "test_detect_strip_balloons_remaps_to_strip_coords|test_band_to_page_dict_remaps_balloon_coords_to_local|test_candidate_text_matching_rejects_edge_overlap" pipeline/tests
python -m pytest pipeline/tests/test_strip_detect.py::DetectStripBalloonsTests::test_detect_strip_balloons_remaps_to_strip_coords pipeline/tests/test_strip_process_bands.py::BandToPageDictTests::test_band_to_page_dict_remaps_balloon_coords_to_local -q
python -m pytest pipeline/tests/test_strip_ocr_debug_artifacts.py -k "candidate_text_matching_rejects_edge_overlap or process_band_writes_ocr_raw_blocks" -q
```

Mapa adicional, não execução automática: `test_vision_stack_runtime.py` cobre sobrevivência/filtros de `_vision_blocks`; `test_strip_inpaint_complete.py` cobre a fronteira com inpaint.

## Última verificação

- Data: 2026-07-24
- Runtime-base: `c0801348`

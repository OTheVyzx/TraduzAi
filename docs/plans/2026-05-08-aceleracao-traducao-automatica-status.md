# Aceleracao Da Traducao Automatica Status

Plano base: `docs/plans/2026-05-08-aceleracao-traducao-automatica.md`

## Estado Atual

Status: Fase 0 concluida, Gate 1 aprovado, Fase 1 shadow implementada e avaliada, Gate 2 reprovado por ganho insuficiente. Smart Skip real foi implementado atras de flag e medido em capitulo completo, mas reprovou como default por ganho real irrelevante e aumento de RAM/VRAM. Macro OCR mapping e Gate 4 por artefato aprovados. Macro OCR full-page shadow real reprovou por diferenca textual/fallback alto. Macro OCR por janelas agrupadas tem uma configuracao conservadora aprovada no runner externo e ja foi acoplado ao pipeline principal em modo shadow. A classificacao de risco do Macro OCR foi refinada, mas o modo real continua bloqueado. Gate 6 estrutural, Gate 7 visual amostrado, Gate 8 de recursos e Gate 9 de decisao consolidada foram adicionados.

Nao ativar Smart Skip real por default.
Nao avancar para Macro OCR real ainda.
Nao ativar Scheduler Overlap por default: passou os gates estruturais depois de locks, mas reprovou como Performance.

## Entregas Implementadas

### Fase 0: Baseline Automatizado

Arquivos criados:

- `pipeline/tools/analyze_pipeline_run.py`
- `pipeline/tools/__init__.py`
- `pipeline/tests/test_analyze_pipeline_run.py`

Evidencia:

- `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_analyze_pipeline_run.py pipeline\tests\test_performance_gate.py -q`
- Resultado: `5 passed in 0.17s`

Artefato gerado:

- `debug/performance_baselines/traduzido2.json`

Baseline do `D:\TraduzAi\AAAAAAA\traduzido2`:

- tempo total: `130.4s`
- paginas: `27`
- textos: `114`
- bandas: `154`
- OCR: `52.706s`
- inpaint: `52.768s`
- traducao: `0.6751s`
- typeset: `4.9963s`

### Gate 1: Baseline Visual Bottleneck

Arquivos criados:

- `pipeline/tools/run_performance_gate.py`
- `pipeline/tests/test_performance_gate.py`

Evidencia real:

- comando: `pipeline\venv\Scripts\python.exe pipeline\tools\run_performance_gate.py D:\TraduzAi\AAAAAAA\traduzido2 --out D:\TraduzAi\debug\performance_gates\traduzido2_baseline`
- resultado: `PASS`
- OCR + inpaint: `94.12%` do tempo medido por estagios

Artefato:

- `debug/performance_gates/traduzido2_baseline/summary.json`

### Fase 1: Smart Skip Shadow

Arquivos criados:

- `pipeline/strip/smart_skip.py`
- `pipeline/tests/test_strip_smart_skip.py`

Arquivos modificados:

- `pipeline/strip/process_bands.py`
- `pipeline/strip/run.py`
- `pipeline/tools/analyze_pipeline_run.py`
- `pipeline/tests/test_strip_process_bands.py`
- `pipeline/tests/test_strip_run.py`

Comportamento implementado:

- flag `TRADUZAI_SMART_SKIP_SHADOW=1`
- classifica textos candidatos sem alterar `skip_processing`
- adiciona auditoria `_smart_skip_shadow`
- agrega contadores em `strip_perf_summary`
- o analisador passa a ler `smart_skip_shadow_candidate_count`

Evidencia de testes:

- comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_strip_smart_skip.py pipeline\tests\test_strip_process_bands.py pipeline\tests\test_strip_run.py pipeline\tests\test_analyze_pipeline_run.py pipeline\tests\test_performance_gate.py -q`
- resultado: `50 passed in 23.01s`

Rodada real shadow:

- config: `debug/performance_gates/smart_skip_shadow_20260508_1458/config.json`
- output: `debug/performance_gates/smart_skip_shadow_20260508_1458/work`
- status pipeline: `complete`

### Gate 2: Smart Skip Shadow

Arquivos criados:

- `pipeline/tools/run_smart_skip_shadow_gate.py`
- `pipeline/tests/test_smart_skip_shadow_gate.py`

Evidencia de testes:

- comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_smart_skip_shadow_gate.py -q`
- resultado: `3 passed in 0.16s`

Evidencia real:

- comando: `pipeline\venv\Scripts\python.exe pipeline\tools\run_smart_skip_shadow_gate.py D:\TraduzAi\debug\performance_gates\smart_skip_shadow_20260508_1458\work --baseline D:\TraduzAi\AAAAAAA\traduzido2 --out D:\TraduzAi\debug\performance_gates\smart_skip_shadow_20260508_1458\gate2`
- resultado: `FAIL`

Motivo:

- economia estimada: `14.4453s`
- minimo exigido no plano: `16.75s`
- candidatos: `4`
- candidatos inseguros: `0`
- comparacao estrutural: passou, contagens batem com baseline

Artefato:

- `debug/performance_gates/smart_skip_shadow_20260508_1458/gate2/summary.json`

## Decisao

Resultado: `voltar para sombra`

Smart Skip shadow esta funcionando e parece seguro, mas ainda nao justifica ativar Smart Skip real como etapa isolada porque nao bateu o ganho minimo definido no plano.

Proxima acao recomendada:

1. Nao ativar `TRADUZAI_SMART_SKIP=1` ainda.
2. Manter a instrumentacao shadow.
3. Seguir para Macro OCR shadow, porque OCR ainda consome dezenas de segundos e Smart Skip sozinho ficou abaixo da meta.

## Riscos Remanescentes

- A heuristica atual e conservadora; ela evita falsos positivos, mas deixa economia na mesa.
- Relaxar a heuristica para incluir textos decorativos/logos da pagina 1 pode aumentar ganho, mas tambem aumenta risco de pular texto real.
- O Gate 2 usa economia estimada por banda; se uma banda tiver candidato e texto real misturados, a economia real pode ser menor que a estimada.

## Macro OCR: Pre-Gate Por Artefato

### Fase 3 Parcial: Mapping/Estimativa

Arquivos criados:

- `pipeline/ocr/macro_ocr.py`
- `pipeline/tests/test_macro_ocr_mapping.py`
- `pipeline/tools/run_macro_ocr_shadow_gate.py`
- `pipeline/tests/test_macro_ocr_shadow_gate.py`

Comportamento implementado:

- coleta janelas de banda a partir de `strip_perf_summary`
- infere pagina de origem por intersecao com `page_profile.y_in_strip_top/y_in_strip_bottom`
- mapeia linhas/page-local para bandas por centro/overlap vertical
- marca fallback para linha que cruza borda de banda
- estima economia por reduzir chamadas OCR de bandas para janelas/paginas

Evidencia de testes:

- comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_macro_ocr_mapping.py pipeline\tests\test_macro_ocr_shadow_gate.py -q`
- resultado: `7 passed in 0.15s`

Evidencia real no baseline `traduzido2`:

- comando: `pipeline\venv\Scripts\python.exe pipeline\tools\run_macro_ocr_shadow_gate.py D:\TraduzAi\AAAAAAA\traduzido2 --out D:\TraduzAi\debug\performance_gates\macro_ocr_shadow_traduzido2_artifact`
- resultado: `PASS`
- OCR atual: `52.706s`
- chamadas OCR atuais estimadas: `154`
- janelas macro estimadas: `21`
- economia estimada: `45.5188s`
- textos mapeados: `113/114`
- missing text rate: `0.0088`
- fallback rate: `0.0`

Evidencia real no output Smart Skip shadow:

- comando: `pipeline\venv\Scripts\python.exe pipeline\tools\run_macro_ocr_shadow_gate.py D:\TraduzAi\debug\performance_gates\smart_skip_shadow_20260508_1458\work --out D:\TraduzAi\debug\performance_gates\macro_ocr_shadow_smart_skip_shadow_artifact`
- resultado: `PASS`
- OCR atual: `45.132s`
- economia estimada: `38.9776s`
- textos mapeados: `113/114`
- missing text rate: `0.0088`
- fallback rate: `0.0`

Limitacao:

- Este Gate 4 ainda e artifact-level. Ele valida risco de remapeamento e economia provavel, mas ainda nao executa PaddleOCR macro real no mesmo capitulo.

Proxima acao recomendada:

1. Nao ativar `TRADUZAI_MACRO_OCR=1` ainda.
2. Manter o runner externo de Macro OCR real como diagnostico de sombra.
3. Usar a configuracao conservadora de janelas agrupadas como candidata para o proximo acoplamento shadow no pipeline principal.
4. So depois disso registrar `macro_ocr_shadow` no `project.json` e reexecutar Gate 4 completo.

## Macro OCR: Shadow Real Amostrado

Arquivos criados:

- `pipeline/tools/run_macro_ocr_actual_shadow.py`
- `pipeline/tests/test_macro_ocr_actual_shadow.py`

Comportamento implementado:

- executa `OCREngine.recognize_blocks_from_page()` contra paginas de um output existente
- permite OCR puro por pagina com `--crop-fallback-max 0`
- permite medir fallback limitado com `--crop-fallback-max N`
- gera `summary.json` com `missing_text_rate`, `different_text_rate`, `fallback_rate`, samples de diferenca e estatisticas internas do OCR

Evidencia de testes:

- comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_strip_smart_skip.py pipeline\tests\test_strip_process_bands.py pipeline\tests\test_strip_run.py pipeline\tests\test_analyze_pipeline_run.py pipeline\tests\test_performance_gate.py pipeline\tests\test_smart_skip_shadow_gate.py pipeline\tests\test_macro_ocr_mapping.py pipeline\tests\test_macro_ocr_shadow_gate.py pipeline\tests\test_macro_ocr_actual_shadow.py -q`
- resultado: `67 passed in 17.60s`

Evidencia real, pagina 1 sem fallback:

- comando: `pipeline\venv\Scripts\python.exe pipeline\tools\run_macro_ocr_actual_shadow.py D:\TraduzAi\AAAAAAA\traduzido2 --out D:\TraduzAi\debug\performance_gates\macro_ocr_actual_shadow_sample --max-pages 1`
- resultado: `FAIL`
- missing text rate: `55.56%`
- different text rate: `33.33%`
- full-page mapped: `4/9`

Evidencia real, paginas 1-5 com fallback maximo por bloco:

- comando: `pipeline\venv\Scripts\python.exe pipeline\tools\run_macro_ocr_actual_shadow.py D:\TraduzAi\AAAAAAA\traduzido2 --out D:\TraduzAi\debug\performance_gates\macro_ocr_actual_shadow_sample5_fallback --max-pages 5 --crop-fallback-max 9`
- resultado: `FAIL`
- missing text rate: `0.0%`
- different text rate: `37.21%`
- fallback rate: `74.42%`
- crop fallback attempts: `32/43`

Decisao:

- resultado: `voltar para sombra`
- motivo: OCR full-page ainda falha em paginas reais e o fallback recupera texto, mas usa crop em blocos demais, o que reduz ou elimina o ganho esperado de OCR
- recomendacao de default: `off`

## Macro OCR: Janelas Agrupadas Em Sombra

Comportamento adicional implementado:

- `pipeline/tools/run_macro_ocr_actual_shadow.py` agora aceita `--window-mode band-groups`
- agrupa blocos proximos em janelas maiores que um crop individual e menores que uma pagina inteira
- traduz bboxes para coordenadas do crop antes de chamar `recognize_blocks_from_page()`
- mede `macro_window_count`, `window_reduction_rate`, `missing_text_rate`, `different_text_rate` e `fallback_rate`
- permite bloquear configuracoes sem ganho com `--min-window-reduction-rate`

Evidencia de testes:

- comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_strip_smart_skip.py pipeline\tests\test_strip_process_bands.py pipeline\tests\test_strip_run.py pipeline\tests\test_analyze_pipeline_run.py pipeline\tests\test_performance_gate.py pipeline\tests\test_smart_skip_shadow_gate.py pipeline\tests\test_macro_ocr_mapping.py pipeline\tests\test_macro_ocr_shadow_gate.py pipeline\tests\test_macro_ocr_actual_shadow.py -q`
- resultado: `69 passed in 17.56s`

Tentativas reais:

1. Janelas leves:
   - comando: `pipeline\venv\Scripts\python.exe pipeline\tools\run_macro_ocr_actual_shadow.py D:\TraduzAi\AAAAAAA\traduzido2 --out D:\TraduzAi\debug\performance_gates\macro_ocr_actual_shadow_windows_sample5 --max-pages 5 --window-mode band-groups --window-max-blocks 3 --window-merge-gap 220 --window-padding 64 --min-window-reduction-rate 0.25`
   - resultado: `FAIL`
   - motivo: qualidade passou, mas `window_reduction_rate` foi so `9.30%`

2. Janelas agressivas:
   - comando: `pipeline\venv\Scripts\python.exe pipeline\tools\run_macro_ocr_actual_shadow.py D:\TraduzAi\AAAAAAA\traduzido2 --out D:\TraduzAi\debug\performance_gates\macro_ocr_actual_shadow_windows_full_aggressive --window-mode band-groups --window-max-blocks 4 --window-merge-gap 1000 --window-padding 96 --min-window-reduction-rate 0.25`
   - resultado: `FAIL`
   - `window_reduction_rate`: `38.60%`
   - `different_text_rate`: `27.19%`
   - motivo: ganho bom, mas qualidade passou do limite de `25%`

3. Janelas intermediarias:
   - comando: `pipeline\venv\Scripts\python.exe pipeline\tools\run_macro_ocr_actual_shadow.py D:\TraduzAi\AAAAAAA\traduzido2 --out D:\TraduzAi\debug\performance_gates\macro_ocr_actual_shadow_windows_full_mid --window-mode band-groups --window-max-blocks 3 --window-merge-gap 1000 --window-padding 96 --min-window-reduction-rate 0.25`
   - resultado: `FAIL`
   - `window_reduction_rate`: `35.96%`
   - `different_text_rate`: `25.44%`
   - motivo: ficou a uma linha do limite de qualidade

4. Janelas conservadoras:
   - comando: `pipeline\venv\Scripts\python.exe pipeline\tools\run_macro_ocr_actual_shadow.py D:\TraduzAi\AAAAAAA\traduzido2 --out D:\TraduzAi\debug\performance_gates\macro_ocr_actual_shadow_windows_full_conservative --window-mode band-groups --window-max-blocks 2 --window-merge-gap 1000 --window-padding 96 --min-window-reduction-rate 0.25`
   - resultado: `PASS`
   - paginas processadas: `21`
   - blocos/textos: `114`
   - janelas macro: `81`
   - `window_reduction_rate`: `28.95%`
   - `missing_text_rate`: `0.0%`
   - `different_text_rate`: `23.68%`
   - `fallback_rate`: `0.0%`
   - runtime do runner: `41.0837s`

Decisao:

- resultado: `shadow aprovado`
- motivo: existe uma configuracao candidata que passa nos gates do runner externo e agora tambem foi executada dentro do pipeline real em modo shadow
- recomendacao de default: `shadow`, nao `performance`
- proxima acao: manter `TRADUZAI_MACRO_OCR_SHADOW=1` para telemetria em capitulos reais; nao substituir o OCR atual antes de reduzir a diferenca textual e provar ganho real end-to-end

## Macro OCR: Shadow Acoplado Ao Pipeline Real

Arquivos modificados:

- `pipeline/ocr/macro_ocr.py`
- `pipeline/strip/run.py`
- `pipeline/tools/run_macro_ocr_actual_shadow.py`
- `pipeline/tools/compare_pipeline_outputs.py`
- `pipeline/tools/export_visual_review_sheet.py`
- `pipeline/tools/measure_resource_profile.py`
- `pipeline/tools/run_translation_batch_gate.py`
- `pipeline/tests/test_macro_ocr_actual_shadow.py`
- `pipeline/tests/test_compare_pipeline_outputs.py`
- `pipeline/tests/test_export_visual_review_sheet.py`
- `pipeline/tests/test_resource_profile.py`
- `pipeline/tests/test_translation_batch_gate.py`
- `pipeline/tests/test_strip_run.py`

Comportamento implementado:

- `TRADUZAI_MACRO_OCR_SHADOW=1` executa Macro OCR por janelas agrupadas depois do OCR atual
- OCR atual continua sendo a fonte da verdade; textos, imagens finais e blocos de inpaint nao sao substituidos
- auditoria `macro_ocr_shadow` e gravada em `page_profile` do `project.json`
- configuracao padrao de sombra: `window_max_blocks=2`, `window_merge_gap=1000`, `window_padding=96`
- se o runtime do strip nao expor `_get_ocr_engine`, o shadow usa fallback controlado para `vision_stack.runtime._get_ocr_engine("max")`

Evidencia de testes:

- comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_strip_run.py::RunChapterSmokeTests::test_run_chapter_attaches_macro_ocr_shadow_when_enabled -q`
- resultado: `1 passed`
- comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_strip_run.py::RunChapterSmokeTests::test_macro_ocr_shadow_uses_vision_stack_engine_when_runtime_has_no_engine -q`
- resultado: `1 passed`
- comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_strip_smart_skip.py pipeline\tests\test_strip_process_bands.py pipeline\tests\test_strip_run.py pipeline\tests\test_analyze_pipeline_run.py pipeline\tests\test_performance_gate.py pipeline\tests\test_smart_skip_shadow_gate.py pipeline\tests\test_macro_ocr_mapping.py pipeline\tests\test_macro_ocr_shadow_gate.py pipeline\tests\test_macro_ocr_actual_shadow.py -q`
- resultado: `71 passed in 17.00s`
- comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_strip_smart_skip.py pipeline\tests\test_strip_process_bands.py pipeline\tests\test_strip_run.py pipeline\tests\test_analyze_pipeline_run.py pipeline\tests\test_performance_gate.py pipeline\tests\test_smart_skip_shadow_gate.py pipeline\tests\test_macro_ocr_mapping.py pipeline\tests\test_macro_ocr_shadow_gate.py pipeline\tests\test_macro_ocr_actual_shadow.py pipeline\tests\test_compare_pipeline_outputs.py -q`
- resultado: `74 passed in 17.88s`
- comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_strip_smart_skip.py pipeline\tests\test_strip_process_bands.py pipeline\tests\test_strip_run.py pipeline\tests\test_analyze_pipeline_run.py pipeline\tests\test_performance_gate.py pipeline\tests\test_smart_skip_shadow_gate.py pipeline\tests\test_macro_ocr_mapping.py pipeline\tests\test_macro_ocr_shadow_gate.py pipeline\tests\test_macro_ocr_actual_shadow.py pipeline\tests\test_compare_pipeline_outputs.py pipeline\tests\test_export_visual_review_sheet.py -q`
- resultado: `78 passed in 16.75s`
- comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_strip_smart_skip.py pipeline\tests\test_strip_process_bands.py pipeline\tests\test_strip_run.py pipeline\tests\test_analyze_pipeline_run.py pipeline\tests\test_performance_gate.py pipeline\tests\test_smart_skip_shadow_gate.py pipeline\tests\test_macro_ocr_mapping.py pipeline\tests\test_macro_ocr_shadow_gate.py pipeline\tests\test_macro_ocr_actual_shadow.py pipeline\tests\test_compare_pipeline_outputs.py pipeline\tests\test_export_visual_review_sheet.py pipeline\tests\test_resource_profile.py -q`
- resultado: `91 passed in 20.86s`

Primeira rodada real:

- config: `debug/performance_gates/macro_ocr_pipeline_shadow_20260508_1715/config.json`
- output: `debug/performance_gates/macro_ocr_pipeline_shadow_20260508_1715/work`
- status pipeline: `complete`
- resultado do shadow: `BLOCK`
- motivo: `runtime has no _get_ocr_engine`
- acao tomada: adicionado fallback controlado para obter o OCR engine pelo runtime global

Segunda rodada real:

- config: `debug/performance_gates/macro_ocr_pipeline_shadow_20260508_1742/config.json`
- output: `debug/performance_gates/macro_ocr_pipeline_shadow_20260508_1742/work`
- status pipeline: `complete`
- resultado do shadow: `PASS`
- paginas processadas pelo shadow: `21`
- blocos/textos: `114`
- janelas macro: `81`
- `window_reduction_rate`: `28.95%`
- `missing_text_rate`: `0.0%`
- `different_text_rate`: `21.93%`
- `fallback_rate`: `0.0%`
- runtime do shadow dentro do pipeline: `41.6877s`

Comparacao estrutural contra `D:\TraduzAi\AAAAAAA\traduzido2`:

- paginas baseline/candidato: `27/27`
- textos baseline/candidato: `114/114`
- blocos de inpaint baseline/candidato: `114/114`
- `macro_ocr_shadow` presente no `project.json`: `true`

Gates sobre a rodada real:

- comando: `pipeline\venv\Scripts\python.exe pipeline\tools\run_performance_gate.py D:\TraduzAi\debug\performance_gates\macro_ocr_pipeline_shadow_20260508_1742\work --out D:\TraduzAi\debug\performance_gates\macro_ocr_pipeline_shadow_20260508_1742\performance_gate`
- resultado: `PASS`
- OCR + inpaint: `94.12%` do tempo medido por estagios
- comando: `pipeline\venv\Scripts\python.exe pipeline\tools\run_macro_ocr_shadow_gate.py D:\TraduzAi\debug\performance_gates\macro_ocr_pipeline_shadow_20260508_1742\work --out D:\TraduzAi\debug\performance_gates\macro_ocr_pipeline_shadow_20260508_1742\artifact_gate`
- resultado: `PASS`
- economia OCR estimada pelo gate de artefato: `40.8077s`
- comando: `pipeline\venv\Scripts\python.exe pipeline\tools\compare_pipeline_outputs.py D:\TraduzAi\AAAAAAA\traduzido2 D:\TraduzAi\debug\performance_gates\macro_ocr_pipeline_shadow_20260508_1742\work --out D:\TraduzAi\debug\performance_gates\macro_ocr_pipeline_shadow_20260508_1742\output_compare`
- resultado: `PASS`
- paginas baseline/candidato: `27/27`
- textos baseline/candidato: `114/114`
- regioes traduzidas baseline/candidato: `114/114`
- blocos de inpaint baseline/candidato: `114/114`
- diferencas de dimensao das imagens finais: `0`
- comando: `pipeline\venv\Scripts\python.exe pipeline\tools\export_visual_review_sheet.py --baseline D:\TraduzAi\AAAAAAA\traduzido2 --candidate D:\TraduzAi\debug\performance_gates\macro_ocr_pipeline_shadow_20260508_1742\work --out D:\TraduzAi\debug\performance_gates\macro_ocr_pipeline_shadow_20260508_1742\visual_review_sheet.html --max-pages 10 --max-crops-per-page 6`
- resultado: `PASS`
- paginas selecionadas por risco: `8, 27, 14, 9, 10, 6, 1, 21, 12, 7`
- maior diferenca de pixels amostrada: `2.4953%` na pagina `1`
- assets gerados: `108`
- prancha revisada: `debug/performance_gates/macro_ocr_pipeline_shadow_20260508_1742/visual_review_contact_sheet.jpg`

Decisao:

- resultado: `shadow aprovado, real bloqueado`
- motivo: a integracao shadow nao altera o contrato de saida e passa nos gates, mas ainda adiciona custo extra quando ligada e tem `different_text_rate` relevante demais para substituir o OCR atual automaticamente
- recomendacao de default: `off` para usuarios finais; `shadow` para rodadas de medicao; `performance` ainda bloqueado
- proxima acao: criar um gate de comparacao textual/visual por pagina e reduzir diferenca textual antes de implementar `TRADUZAI_MACRO_OCR=1`

## Gate 6: Comparacao Estrutural De Outputs

Arquivos criados:

- `pipeline/tools/compare_pipeline_outputs.py`
- `pipeline/tests/test_compare_pipeline_outputs.py`

Comportamento implementado:

- compara `project.json` entre baseline e candidato
- valida contagem de paginas, textos, regioes traduzidas e blocos de inpaint
- valida existencia e dimensoes das imagens finais em `translated/` ou `images/`
- permite metadados extras de sombra sem falhar o contrato principal
- falha se texto some sem skip auditado
- bloqueia quando `project.json` ou imagens finais estao ausentes

Evidencia de TDD:

- primeiro teste rodado antes do tool existir:
  - comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_compare_pipeline_outputs.py -q`
  - resultado: `ModuleNotFoundError: No module named 'pipeline.tools.compare_pipeline_outputs'`
- apos implementacao:
  - comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_compare_pipeline_outputs.py -q`
  - resultado: `3 passed in 0.22s`

Evidencia real:

- comando: `pipeline\venv\Scripts\python.exe pipeline\tools\compare_pipeline_outputs.py D:\TraduzAi\AAAAAAA\traduzido2 D:\TraduzAi\debug\performance_gates\macro_ocr_pipeline_shadow_20260508_1742\work --out D:\TraduzAi\debug\performance_gates\macro_ocr_pipeline_shadow_20260508_1742\output_compare`
- resultado: `PASS`
- paginas: `27/27`
- textos: `114/114`
- regioes traduzidas: `114/114`
- blocos de inpaint: `114/114`
- diferencas de dimensao de imagem final: `0`
- artefato: `debug/performance_gates/macro_ocr_pipeline_shadow_20260508_1742/output_compare/summary.json`

Decisao:

- resultado: `aprovar gate estrutural`
- motivo: a rodada Macro OCR shadow preservou o contrato estrutural e visual dimensional do baseline
- limitacao: este gate ainda nao faz diff pixel-a-pixel nem revisao visual de crops
- proxima acao: implementar visual review sheet amostrada para paginas com maior `different_text_rate`

## Gate 7: Comparacao Visual Amostrada

Arquivos criados:

- `pipeline/tools/export_visual_review_sheet.py`
- `pipeline/tests/test_export_visual_review_sheet.py`

Comportamento implementado:

- seleciona paginas por risco do `macro_ocr_shadow.page_reports`
- copia imagens baseline/candidato para `assets/`
- gera HTML lado a lado com metricas de OCR shadow
- gera crops lado a lado por regiao de texto
- calcula `pixel_diff_rate` por pagina selecionada
- bloqueia quando `project.json` obrigatorio esta ausente

Evidencia de TDD:

- primeiro teste rodado antes do tool existir:
  - comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_export_visual_review_sheet.py -q`
  - resultado: `ModuleNotFoundError: No module named 'pipeline.tools.export_visual_review_sheet'`
- teste de `pixel_diff_rate` antes da metrica existir:
  - comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_export_visual_review_sheet.py::test_visual_review_sheet_reports_pixel_difference_rate -q`
  - resultado: `KeyError: 'page_reports'`
- teste de crops antes do contador existir:
  - comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_export_visual_review_sheet.py::test_visual_review_sheet_exports_text_region_crops -q`
  - resultado: `KeyError: 'crop_count'`
- apos implementacao:
  - comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_export_visual_review_sheet.py -q`
  - resultado: `4 passed in 0.26s`

Evidencia real:

- comando: `pipeline\venv\Scripts\python.exe pipeline\tools\export_visual_review_sheet.py --baseline D:\TraduzAi\AAAAAAA\traduzido2 --candidate D:\TraduzAi\debug\performance_gates\macro_ocr_pipeline_shadow_20260508_1742\work --out D:\TraduzAi\debug\performance_gates\macro_ocr_pipeline_shadow_20260508_1742\visual_review_sheet.html --max-pages 10 --max-crops-per-page 6`
- resultado: `PASS`
- paginas selecionadas: `8, 27, 14, 9, 10, 6, 1, 21, 12, 7`
- assets gerados: `108`
- maior `pixel_diff_rate`: `0.024953`
- artefatos:
  - `debug/performance_gates/macro_ocr_pipeline_shadow_20260508_1742/visual_review_sheet.html`
  - `debug/performance_gates/macro_ocr_pipeline_shadow_20260508_1742/summary.json`
  - `debug/performance_gates/macro_ocr_pipeline_shadow_20260508_1742/visual_review_contact_sheet.jpg`

Revisao visual amostrada:

- paginas revisadas na prancha: `1, 7, 9, 27`
- nenhum texto faltando, duplicado ou inpaint quebrado foi identificado visualmente na amostra
- existem variacoes pequenas de texto/render entre execucoes, entao o resultado nao deve ser tratado como equivalencia pixel-perfect

Decisao:

- resultado: `aprovar gate visual amostrado`
- motivo: a folha foi gerada, as paginas de maior risco estao auditaveis e a amostra revisada nao mostrou perda obvia de texto
- limitacao: ainda falta revisao visual completa ou diff por crop antes de liberar Macro OCR real
- proxima acao: atacar divergencia textual do Macro OCR por janelas antes de `TRADUZAI_MACRO_OCR=1`

## Gate 8: Recursos Do Sistema

Arquivos criados:

- `pipeline/tools/measure_resource_profile.py`
- `pipeline/tests/test_resource_profile.py`

Comportamento implementado:

- executa um comando filho e mede tempo total
- mede pico de RSS do processo e filhos via `psutil`
- mede CPU media por delta de tempo de CPU do processo e filhos
- tenta medir VRAM total usada via `nvidia-smi`, quando disponivel
- grava `command_stdout.log`, `command_stderr.log` e `resources.json`
- usa timeout interno para matar a arvore de processos e gravar `BLOCK`
- evita deadlock de stdout redirecionando saida para arquivo

Evidencia de TDD:

- primeiro teste rodado antes do tool existir:
  - comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_resource_profile.py -q`
  - resultado: `ModuleNotFoundError: No module named 'pipeline.tools.measure_resource_profile'`
- teste de timeout antes do parametro existir:
  - comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_resource_profile.py::test_resource_profile_blocks_and_kills_command_when_timeout_expires -q`
  - resultado: `TypeError: measure_resource_profile() got an unexpected keyword argument 'timeout_seconds'`
- teste de stdout verboso antes da correcao:
  - comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_resource_profile.py::test_resource_profile_does_not_deadlock_on_chatty_stdout -q`
  - resultado: `BLOCK` por timeout
- teste de CPU antes da correcao:
  - comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_resource_profile.py::test_resource_profile_reports_nonzero_cpu_for_busy_command -q`
  - resultado: `avg_cpu_percent == 0.0`
- apos implementacao:
  - comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_resource_profile.py -q`
  - resultado: `6 passed in 1.67s`

Evidencia controlada:

- comando: `pipeline\venv\Scripts\python.exe pipeline\tools\measure_resource_profile.py --out D:\TraduzAi\debug\performance_gates\resource_profile_controlled --sample-interval 0.05 --timeout-seconds 10 -- pipeline\venv\Scripts\python.exe -c "import time; data='x'*5000000; time.sleep(0.4); print(len(data))"`
- resultado: `PASS`
- tempo: `0.6719s`
- pico RSS: `19.121 MB`
- VRAM total aproximada: `1215 MB`

Evidencia real no pipeline atual:

- config: `debug/performance_gates/resource_profile_pipeline_20260508_1845/config.json`
- comando: `pipeline\venv\Scripts\python.exe pipeline\tools\measure_resource_profile.py --out D:\TraduzAi\debug\performance_gates\resource_profile_pipeline_20260508_1845\resources --sample-interval 2.0 --timeout-seconds 420 -- pipeline\venv\Scripts\python.exe pipeline\main.py D:\TraduzAi\debug\performance_gates\resource_profile_pipeline_20260508_1845\config.json`
- resultado: `PASS`
- tempo total medido: `130.4294s`
- pico RSS: `6341.109 MB`
- CPU media: `94.162%`
- VRAM total aproximada: `3017 MB`
- amostras: `64`
- timeout: `false`
- output completo: `project.json` presente e `27` imagens finais
- performance gate do output: `PASS`, OCR + inpaint `94.21%`
- compare output vs `traduzido2`: `PASS`

Artefatos:

- `debug/performance_gates/resource_profile_pipeline_20260508_1845/resources/resources.json`
- `debug/performance_gates/resource_profile_pipeline_20260508_1845/resources/command_stdout.log`
- `debug/performance_gates/resource_profile_pipeline_20260508_1845/resources/command_stderr.log`
- `debug/performance_gates/resource_profile_pipeline_20260508_1845/performance_gate/summary.json`
- `debug/performance_gates/resource_profile_pipeline_20260508_1845/output_compare/summary.json`

Decisao:

- resultado: `aprovar gate de recursos`
- motivo: o gate mede uma execucao real completa sem travar stdout e preserva o contrato do output
- leitura: o gargalo continua visual; pico de RAM medido ficou em `~6.34 GB`, VRAM total em `~3.02 GB`, CPU media perto de um core ocupado
- limitacao: VRAM e aproximada pelo total do `nvidia-smi`, nao isolada por PID
- proxima acao: usar este gate para comparar futuros modos `performance` e `eco`

## Gate 9: Decisao Final Do Agente

Artefato criado:

- `debug/performance_gates/2026-05-08-aceleracao-traducao-automatica-current/decision.md`

Resultado:

- status: `voltar para sombra`
- baseline: `130.4s`
- candidato atual medido pelo Gate 8: `130.4294s`
- economia real aprovada para default: `0s`
- Smart Skip real: bloqueado
- Macro OCR real: bloqueado
- Macro OCR shadow: aprovado para diagnostico
- Gate 6 estrutural: `PASS`
- Gate 7 visual amostrado: `PASS`
- Gate 8 recursos: `PASS`

Riscos consolidados:

- Smart Skip isolado ficou abaixo da economia minima.
- Macro OCR full-page falhou por texto ausente/fallback alto.
- Macro OCR por janelas ainda tem divergencia textual alta para virar fonte da verdade.
- Ainda nao existe perfil `performance` nem `eco` implementado.

Recomendacao de default:

- usuarios finais: `off`
- diagnostico: `shadow`
- performance: `bloqueado`
- eco: `a implementar`

## Macro OCR: Classificacao De Divergencia Textual

Arquivos modificados:

- `pipeline/ocr/macro_ocr.py`
- `pipeline/tools/run_macro_ocr_actual_shadow.py`
- `pipeline/strip/run.py`
- `pipeline/tests/test_macro_ocr_mapping.py`
- `pipeline/tests/test_macro_ocr_actual_shadow.py`

Comportamento implementado:

- classifica diferencas entre OCR atual e Macro OCR como `exact`, `line_marker_artifact` ou `material`
- adiciona `material_different_count`, `material_different_text_rate` e `line_marker_artifact_count`
- propaga os novos campos no runner externo e no shadow acoplado ao pipeline
- nao altera OCR, traducao, inpaint, typeset nem `project.json` de producao fora do relatorio shadow

Evidencia de TDD:

- teste antes do classificador existir:
  - comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_macro_ocr_mapping.py::test_classify_ocr_text_difference_separates_line_marker_artifacts_from_material_changes pipeline\tests\test_macro_ocr_mapping.py::test_compare_aligned_macro_ocr_texts_reports_material_difference_rate -q`
  - resultado: `ImportError: cannot import name 'classify_ocr_text_difference'`
- teste antes do runner agregar os campos:
  - comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_macro_ocr_actual_shadow.py::test_actual_macro_ocr_shadow_fails_when_macro_output_changes_too_much_text -q`
  - resultado: `KeyError: 'material_different_count'`
- apos implementacao:
  - comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_macro_ocr_mapping.py pipeline\tests\test_macro_ocr_actual_shadow.py pipeline\tests\test_strip_run.py -q`
  - resultado: `32 passed in 3.99s`
- suite impactada:
  - comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_strip_smart_skip.py pipeline\tests\test_strip_process_bands.py pipeline\tests\test_strip_run.py pipeline\tests\test_analyze_pipeline_run.py pipeline\tests\test_performance_gate.py pipeline\tests\test_smart_skip_shadow_gate.py pipeline\tests\test_macro_ocr_mapping.py pipeline\tests\test_macro_ocr_shadow_gate.py pipeline\tests\test_macro_ocr_actual_shadow.py pipeline\tests\test_compare_pipeline_outputs.py pipeline\tests\test_export_visual_review_sheet.py pipeline\tests\test_resource_profile.py -q`
  - resultado: `86 passed in 19.67s`

Evidencia real:

- comando: `pipeline\venv\Scripts\python.exe pipeline\tools\run_macro_ocr_actual_shadow.py D:\TraduzAi\AAAAAAA\traduzido2 --out D:\TraduzAi\debug\performance_gates\macro_ocr_actual_shadow_windows_full_conservative_classified --window-mode band-groups --window-max-blocks 2 --window-merge-gap 1000 --window-padding 96 --min-window-reduction-rate 0.25`
- resultado: `PASS`
- runtime: `41.0141s`
- textos: `114`
- janelas macro: `81`
- `different_count`: `27`
- `material_different_count`: `25`
- `line_marker_artifact_count`: `2`
- `different_text_rate`: `23.68%`
- `material_different_text_rate`: `21.93%`

Decisao:

- resultado: `bloqueio confirmado`
- motivo: a maior parte das divergencias do Macro OCR conservador ainda e material, nao apenas ruido superficial de pontuacao ou marcador numerico
- recomendacao: manter Macro OCR real bloqueado e atacar divergencias/fallback por bloco antes de qualquer default `performance`

## Macro OCR: Gate De Fallback Efetivo

Arquivos modificados:

- `pipeline/ocr/macro_ocr.py`
- `pipeline/tools/run_macro_ocr_actual_shadow.py`
- `pipeline/strip/run.py`
- `pipeline/tests/test_macro_ocr_mapping.py`
- `pipeline/tests/test_macro_ocr_actual_shadow.py`

Comportamento implementado:

- estima o custo de fallback real por bloco para divergencias materiais
- calcula `fallback_adjusted_ocr_call_count`
- calcula `fallback_adjusted_window_reduction_rate`
- adiciona `--min-fallback-adjusted-reduction-rate` no runner externo
- propaga a metrica no shadow acoplado ao pipeline para auditoria futura

Evidencia de TDD:

- teste antes do helper existir:
  - comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_macro_ocr_mapping.py::test_estimate_macro_ocr_fallback_cost_counts_material_differences_as_block_fallbacks -q`
  - resultado: `ImportError: cannot import name 'estimate_macro_ocr_fallback_cost'`
- teste antes do runner expor os campos:
  - comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_macro_ocr_actual_shadow.py::test_actual_macro_ocr_shadow_fails_when_macro_output_changes_too_much_text -q`
  - resultado: `KeyError: 'fallback_adjusted_ocr_call_count'`
- teste antes do limite existir:
  - comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_macro_ocr_actual_shadow.py::test_actual_macro_ocr_shadow_fails_when_fallback_adjusted_reduction_is_too_low -q`
  - resultado: `TypeError: evaluate_actual_macro_ocr_shadow() got an unexpected keyword argument 'min_fallback_adjusted_reduction_rate'`
- apos implementacao:
  - comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_macro_ocr_actual_shadow.py pipeline\tests\test_macro_ocr_mapping.py -q`
  - resultado: `17 passed in 8.14s`
- suite impactada:
  - comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_strip_smart_skip.py pipeline\tests\test_strip_process_bands.py pipeline\tests\test_strip_run.py pipeline\tests\test_analyze_pipeline_run.py pipeline\tests\test_performance_gate.py pipeline\tests\test_smart_skip_shadow_gate.py pipeline\tests\test_macro_ocr_mapping.py pipeline\tests\test_macro_ocr_shadow_gate.py pipeline\tests\test_macro_ocr_actual_shadow.py pipeline\tests\test_compare_pipeline_outputs.py pipeline\tests\test_export_visual_review_sheet.py pipeline\tests\test_resource_profile.py -q`
  - resultado: `88 passed in 22.25s`

Evidencia real:

- comando: `pipeline\venv\Scripts\python.exe pipeline\tools\run_macro_ocr_actual_shadow.py D:\TraduzAi\AAAAAAA\traduzido2 --out D:\TraduzAi\debug\performance_gates\macro_ocr_actual_shadow_windows_full_conservative_fallback_gate --window-mode band-groups --window-max-blocks 2 --window-merge-gap 1000 --window-padding 96 --min-window-reduction-rate 0.25 --min-fallback-adjusted-reduction-rate 0.25`
- resultado: `FAIL`
- motivo: `fallback-adjusted window reduction rate 7.02% is below 25.00%`
- textos: `114`
- janelas macro: `81`
- diferencas materiais: `25`
- chamadas OCR efetivas com fallback: `106`
- reducao bruta de chamadas: `28.95%`
- reducao ajustada por fallback: `7.02%`
- artefato: `debug/performance_gates/macro_ocr_actual_shadow_windows_full_conservative_fallback_gate/summary.json`
- decisao: `debug/performance_gates/macro_ocr_actual_shadow_windows_full_conservative_fallback_gate/decision.md`

Decisao:

- resultado: `Fase 4 bloqueada`
- motivo: com fallback por bloco para proteger qualidade, a economia efetiva cai para `7.02%`, abaixo do limite minimo usado no gate
- recomendacao: nao implementar `TRADUZAI_MACRO_OCR=1` ainda; reduzir divergencia material primeiro

## Fase 5: Pre-Gate De Traducao Em Lote

Arquivos criados:

- `pipeline/tools/run_translation_batch_gate.py`
- `pipeline/tests/test_translation_batch_gate.py`

Comportamento implementado:

- avalia se a etapa `translate` e material o suficiente para justificar traducao em lote agora
- usa metricas existentes de `strip_perf_summary`
- falha por baixo impacto quando traducao fica abaixo de `5s` e abaixo de `5%` dos tempos por estagio
- nao altera tradutor, cache, Google-only, output ou `project.json`

Evidencia de TDD:

- primeiro teste rodado antes do tool existir:
  - comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_translation_batch_gate.py -q`
  - resultado: `ModuleNotFoundError: No module named 'pipeline.tools.run_translation_batch_gate'`
- apos implementacao:
  - comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_translation_batch_gate.py -q`
  - resultado: `3 passed in 0.12s`
- suite impactada:
  - comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_strip_smart_skip.py pipeline\tests\test_strip_process_bands.py pipeline\tests\test_strip_run.py pipeline\tests\test_analyze_pipeline_run.py pipeline\tests\test_performance_gate.py pipeline\tests\test_smart_skip_shadow_gate.py pipeline\tests\test_macro_ocr_mapping.py pipeline\tests\test_macro_ocr_shadow_gate.py pipeline\tests\test_macro_ocr_actual_shadow.py pipeline\tests\test_compare_pipeline_outputs.py pipeline\tests\test_export_visual_review_sheet.py pipeline\tests\test_resource_profile.py pipeline\tests\test_translation_batch_gate.py -q`
  - resultado: `91 passed in 20.86s`

Evidencia real:

- comando: `pipeline\venv\Scripts\python.exe pipeline\tools\run_translation_batch_gate.py D:\TraduzAi\AAAAAAA\traduzido2 --out D:\TraduzAi\debug\performance_gates\traduzido2_translation_batch_gate`
- resultado: `FAIL`
- motivo: `translation stage is below batching threshold (0.68s, 0.60%)`
- traducao: `0.6751s`
- textos: `114`
- comando: `pipeline\venv\Scripts\python.exe pipeline\tools\run_translation_batch_gate.py D:\TraduzAi\debug\performance_gates\resource_profile_pipeline_20260508_1845\work --out D:\TraduzAi\debug\performance_gates\resource_profile_pipeline_20260508_1845\translation_batch_gate`
- resultado: `FAIL`
- motivo: `translation stage is below batching threshold (0.67s, 0.64%)`
- artefato: `debug/performance_gates/traduzido2_translation_batch_gate/summary.json`
- decisao: `debug/performance_gates/traduzido2_translation_batch_gate/decision.md`

Decisao:

- resultado: `adiar Fase 5`
- motivo: traducao textual nao e gargalo no capitulo de referencia; melhoria perfeita economizaria menos de `1s`
- recomendacao: manter Google-only atual e reabrir traducao em lote so quando um capitulo mostrar traducao acima de `5s` ou `5%`

## Gate 10: Contrato De Importacao Do Project.json

Arquivos criados:

- `pipeline/tools/run_project_import_gate.py`
- `pipeline/tests/test_project_import_gate.py`

Artefatos criados:

- `debug/performance_gates/traduzido2_project_import_gate/summary.json`
- `debug/performance_gates/macro_ocr_pipeline_shadow_20260508_1742/project_import_gate/summary.json`
- `debug/performance_gates/resource_profile_pipeline_20260508_1845/project_import_gate/summary.json`
- `debug/performance_gates/project_import_gate_20260508/decision.md`

Comportamento implementado:

- valida se o output contem `project.json` carregavel com `paginas`
- valida consistencia de `estatisticas.total_paginas` e `estatisticas.total_textos`
- valida paths de `image_layers` e aliases legados `arquivo_original` / `arquivo_traduzido`
- decodifica imagens referenciadas pelo editor
- exige dimensao igual a base para `inpaint` e `rendered`
- aceita `mask`, `brush` e `recovery` como placeholder `1x1` ou imagem full-size
- valida bbox hidratavel para cada camada de texto
- grava `summary.json` e retorna codigo nao-zero quando o contrato falha

Evidencia de TDD:

- primeiro teste rodado antes do tool existir:
  - comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_project_import_gate.py -q`
  - resultado: `ModuleNotFoundError: No module named 'pipeline.tools.run_project_import_gate'`
- teste de placeholder antes da correcao:
  - comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_project_import_gate.py::test_project_import_gate_allows_placeholder_editing_layers -q`
  - resultado: `AssertionError: assert 'FAIL' == 'PASS'`
- apos implementacao e correcao:
  - comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_project_import_gate.py -q`
  - resultado: `4 passed in 0.20s`
- testes de contrato proximos:
  - comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_project_import_gate.py pipeline\tests\test_compare_pipeline_outputs.py -q`
  - resultado: `6 passed in 0.22s`
  - comando: `..\pipeline\venv\Scripts\python.exe -m pytest tests\test_project_writer.py -q` a partir de `D:\TraduzAi\pipeline`
  - resultado: `6 passed in 0.08s`
- suite impactada atualizada:
  - comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_strip_smart_skip.py pipeline\tests\test_strip_process_bands.py pipeline\tests\test_strip_run.py pipeline\tests\test_analyze_pipeline_run.py pipeline\tests\test_performance_gate.py pipeline\tests\test_smart_skip_shadow_gate.py pipeline\tests\test_macro_ocr_mapping.py pipeline\tests\test_macro_ocr_shadow_gate.py pipeline\tests\test_macro_ocr_actual_shadow.py pipeline\tests\test_compare_pipeline_outputs.py pipeline\tests\test_export_visual_review_sheet.py pipeline\tests\test_resource_profile.py pipeline\tests\test_translation_batch_gate.py pipeline\tests\test_project_import_gate.py -q`
  - resultado: `95 passed in 32.62s`
  - comando: `..\pipeline\venv\Scripts\python.exe -m pytest tests\test_project_writer.py -q` a partir de `D:\TraduzAi\pipeline`
  - resultado: `6 passed in 0.14s`

Evidencia real:

- baseline `D:\TraduzAi\AAAAAAA\traduzido2`: `PASS`
- Macro OCR shadow `D:\TraduzAi\debug\performance_gates\macro_ocr_pipeline_shadow_20260508_1742\work`: `PASS`
- resource profile `D:\TraduzAi\debug\performance_gates\resource_profile_pipeline_20260508_1845\work`: `PASS`
- cada run validou `27` paginas, `114` text layers, `162` imagens decodificadas, `0` imagens ausentes, `0` bboxes invalidas e `0` divergencias de dimensao em `base/inpaint/rendered`

Decisao:

- resultado: `aprovar gate de importacao`
- motivo: os outputs atuais, incluindo Macro OCR shadow e resource profile, preservam o contrato de reabertura por artefato
- warning aceito: varias camadas de texto ainda chegam sem `id`, mas o importador Rust gera IDs durante a migracao
- limitacao: este gate nao substitui E2E visual/interativo do editor

## Fase 7: Contrato Inicial De Perfis Performance/Eco

Arquivos criados:

- `pipeline/runtime_profiles.py`
- `pipeline/tests/test_runtime_profiles.py`

Arquivos modificados:

- `pipeline/main.py`
- `pipeline/tests/test_main_strip_config.py`

Artefatos criados:

- `debug/performance_gates/runtime_profile_gate_20260508/summary.json`
- `debug/performance_gates/runtime_profile_gate_20260508/decision.md`

Comportamento implementado:

- adiciona resolucao pura de `runtime_profile` com valores `balanced`, `performance` e `eco`
- aceita pedido direto por `runtime_profile` ou via objeto `preset.runtime_profile`
- default `balanced` preserva comportamento atual
- `performance` fica declarado, mas bloqueia `TRADUZAI_SMART_SKIP` e `TRADUZAI_MACRO_OCR` para default ate os gates passarem
- `eco` desativa warmup visual opcional e prewarm do inpainter quando solicitado explicitamente
- `eco` aplica defaults conservadores de threads (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`) sem sobrescrever env ja definido pelo usuario
- `main.py` grava `runtime_profile_decision` no config em memoria e persiste `runtime_profile` no `project.json`

Evidencia de TDD:

- primeiro teste rodado antes do modulo existir:
  - comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_runtime_profiles.py -q`
  - resultado: `ModuleNotFoundError: No module named 'pipeline.runtime_profiles'`
- teste de integracao antes da funcao existir:
  - comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_main_strip_config.py::StripTargetPagesConfigTests::test_runtime_profile_config_records_eco_decision_and_applies_env_defaults -q`
  - resultado: `ImportError: cannot import name '_apply_runtime_profile_config'`
- apos implementacao:
  - comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_runtime_profiles.py -q`
  - resultado: `6 passed in 0.18s`
  - comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_main_strip_config.py -q`
  - resultado: `10 passed in 0.46s`
- suite impactada atualizada:
  - comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_strip_smart_skip.py pipeline\tests\test_strip_process_bands.py pipeline\tests\test_strip_run.py pipeline\tests\test_main_strip_config.py pipeline\tests\test_analyze_pipeline_run.py pipeline\tests\test_performance_gate.py pipeline\tests\test_smart_skip_shadow_gate.py pipeline\tests\test_macro_ocr_mapping.py pipeline\tests\test_macro_ocr_shadow_gate.py pipeline\tests\test_macro_ocr_actual_shadow.py pipeline\tests\test_compare_pipeline_outputs.py pipeline\tests\test_export_visual_review_sheet.py pipeline\tests\test_resource_profile.py pipeline\tests\test_translation_batch_gate.py pipeline\tests\test_project_import_gate.py pipeline\tests\test_runtime_profiles.py -q`
  - resultado: `111 passed in 23.31s`
  - comando: `..\pipeline\venv\Scripts\python.exe -m pytest tests\test_project_writer.py -q` a partir de `D:\TraduzAi\pipeline`
  - resultado: `6 passed in 0.11s`

Evidencia do gate:

- comando: `pipeline\venv\Scripts\python.exe pipeline\runtime_profiles.py --out D:\TraduzAi\debug\performance_gates\runtime_profile_gate_20260508`
- resultado: `PASS`
- motivo: Performance documenta aceleradores bloqueados; Eco tem contrato executavel de menor consumo

Decisao:

- resultado: `aprovar contrato inicial de perfis`
- default atual: `balanced`
- perfil Performance: `bloqueado para default`
- perfil Eco: `executavel`, mas ainda precisa benchmark real com `measure_resource_profile.py`

## Fase 7: Benchmark Real Do Perfil Eco

Artefatos criados:

- `debug/performance_gates/eco_profile_pipeline_20260508/config.json`
- `debug/performance_gates/eco_profile_pipeline_20260508/resources/resources.json`
- `debug/performance_gates/eco_profile_pipeline_20260508/performance_gate/summary.json`
- `debug/performance_gates/eco_profile_pipeline_20260508/output_compare/summary.json`
- `debug/performance_gates/eco_profile_pipeline_20260508/project_import_gate/summary.json`

Config aplicada:

- `runtime_profile`: `eco`
- `visual_stack_warmup`: `false`
- `strip_inpainter_prewarm`: `false`
- `semantic_review`: `false`
- `cpu_thread_limit`: `2`

Evidencia real:

- comando: `pipeline\venv\Scripts\python.exe pipeline\tools\measure_resource_profile.py --out D:\TraduzAi\debug\performance_gates\eco_profile_pipeline_20260508\resources --sample-interval 2.0 --timeout-seconds 420 -- pipeline\venv\Scripts\python.exe pipeline\main.py D:\TraduzAi\debug\performance_gates\eco_profile_pipeline_20260508\config.json`
- resultado: `PASS`
- tempo total medido: `143.0826s`
- pico RSS: `6827.332 MB`
- CPU media: `93.479%`
- VRAM total aproximada: `4615 MB`
- performance gate do output: `PASS`, OCR + inpaint `94.34%`
- compare output vs `traduzido2`: `PASS`
- project import gate: `PASS`

Comparacao contra `resource_profile_pipeline_20260508_1845`:

- tempo: `130.4294s` -> `143.0826s` (`+12.6532s`)
- pico RSS: `6341.109 MB` -> `6827.332 MB` (`+486.223 MB`)
- CPU media: `94.162%` -> `93.479%` (`-0.683 pp`)
- VRAM total aproximada: `3017 MB` -> `4615 MB` (`+1598 MB`)

Decisao:

- resultado: `Eco reprovado como melhoria real nesta rodada`
- motivo: a saida preserva qualidade estrutural/importacao, mas ficou mais lenta e nao reduziu pico medido de RAM/VRAM
- recomendacao: manter Eco fora da UI/default; se for exposto no futuro, rotular como experimental ate haver reducao real de pico
- limitacao: VRAM continua sendo medida como total aproximado do `nvidia-smi`, nao por PID

## Fase 6: Contrato De Scheduler DAG Com Worker GPU Unico

Arquivos criados:

- `pipeline/strip/scheduler.py`
- `pipeline/tests/test_strip_scheduler.py`

Artefatos criados:

- `debug/performance_gates/strip_scheduler_gate_20260508/summary.json`
- `debug/performance_gates/strip_scheduler_gate_20260508/decision.md`

Comportamento implementado:

- modela o capitulo como DAG puro de tarefas `concat`, `detect`, `ocr`, `translate_batch`, `inpaint`, `typeset` e `reassemble`
- limita tarefas GPU a `max_gpu_parallel=1`
- permite `typeset` CPU caminhar depois do `inpaint` da mesma banda, sem sobrepor dois trabalhos GPU
- grava `summary.json` do contrato para auditoria
- mantem o scheduler como contrato/plano; `run_chapter` ainda nao usa este DAG

Evidencia de TDD:

- primeiro teste antes do modulo existir:
  - comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_strip_scheduler.py -q`
  - resultado: `ModuleNotFoundError: No module named 'pipeline.strip.scheduler'`
- teste de ordenacao antes da correcao numerica:
  - comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_strip_scheduler.py::test_strip_scheduler_orders_band_tasks_by_numeric_band_index -q`
  - resultado: `AssertionError`, com `ocr:10` antes de `ocr:2`
- apos implementacao/correcao:
  - comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_strip_scheduler.py -q -p no:cacheprovider --basetemp=C:\tmp\pytest-traduzai-scheduler`
  - resultado: `5 passed in 0.14s`

Evidencia do gate:

- comando: `pipeline\venv\Scripts\python.exe pipeline\strip\scheduler.py --out D:\TraduzAi\debug\performance_gates\strip_scheduler_gate_20260508 --band-count 154 --page-count 27`
- resultado: `PASS`
- tarefas totais: `466`
- tarefas CPU/GPU: `158/308`
- `max_cpu_parallel`: `2`
- `max_gpu_parallel`: `1`
- stages: `concat=1`, `detect=1`, `ocr=154`, `translate_batch=1`, `inpaint=154`, `typeset=154`, `reassemble=1`

Decisao:

- resultado: `aprovar contrato, nao ativar runtime`
- motivo: o contrato e topologico, serializa GPU para seguranca e agora ordena bandas numericamente
- limitacao: nao ha speedup real nesta fase porque o scheduler ainda nao substitui a execucao sequencial de `run_chapter`
- risco para ativacao futura: preservar `band_history`, glossario/contexto, eventos Tauri, ordem de logs e todos os gates de qualidade

## Fase 6b: Scheduler DAG Shadow Gate

Arquivos modificados:

- `pipeline/tools/run_scheduler_shadow_gate.py`
- `pipeline/tests/test_strip_scheduler_shadow_gate.py`

Comportamento implementado:

- cria um gate de sombra para o scheduler DAG
- deriva `band_count` e `page_count` do `strip_perf_summary` de um output real
- monta o plano com `build_strip_scheduler_plan`
- exige topologia valida e `max_gpu_parallel=1`
- roda `compare_pipeline_outputs` entre baseline e candidato
- retorna `PASS` somente se o DAG for valido e o candidato continuar estruturalmente equivalente
- retorna `FAIL` quando o candidato altera textos, inpaint ou regioes traduzidas
- retorna `BLOCK` quando o output nao tem `strip_perf_summary`

Evidencia de TDD/testes:

- red: `ModuleNotFoundError: No module named 'pipeline.tools.run_scheduler_shadow_gate'`
- green focado: `3 passed in 0.87s`
- suite scheduler: `8 passed in 0.33s`
- suite de impacto final: `11 passed in 0.29s`

Gates reais:

- artefato PASS: `debug/performance_gates/strip_scheduler_shadow_gate_20260508/decision.md`
- comando PASS: `pipeline\venv\Scripts\python.exe pipeline\tools\run_scheduler_shadow_gate.py D:\TraduzAi\debug\performance_gates\resource_profile_pipeline_20260508_1845\work D:\TraduzAi\debug\performance_gates\resource_profile_pipeline_20260508_1845\work --out D:\TraduzAi\debug\performance_gates\strip_scheduler_shadow_gate_20260508`
- resultado PASS: `task_count=466`, CPU/GPU `158/308`, `max_gpu_parallel=1`, `output_compare=PASS`
- controle negativo: `debug/performance_gates/strip_scheduler_shadow_gate_macro_real_fail_20260508/decision.md`
- resultado negativo: `FAIL`, porque o candidato Macro OCR real falha o output compare

Decisao:

- resultado: `aprovado como shadow gate, runtime DAG ainda nao ativo`
- motivo: agora existe um gate reutilizavel que valida o DAG e bloqueia candidato com regressao estrutural
- limitacao: ainda nao ha speedup real; o executor experimental foi criado na Fase 6c, mas ainda executa `process_band` em ordem sequencial

## Fase 6c: Scheduler Executor Experimental

Arquivos modificados:

- `pipeline/strip/run.py`
- `pipeline/tests/test_strip_run.py`

Comportamento implementado:

- adiciona flag experimental `TRADUZAI_STRIP_SCHEDULER_EXECUTOR=1`
- default continua desligado
- quando ligada, `run_chapter` cria um plano com `build_strip_scheduler_plan`
- o processamento de bandas continua sequencial para preservar `band_history`, glossario mutavel e objetos compartilhados de OCR/inpaint/typeset
- `strip_perf_summary.scheduler_executor` registra modo, tarefas, limites CPU/GPU, validacao do plano e bandas processadas

Evidencia de TDD/testes:

- red: `1 failed`, porque `strip_perf_summary` nao tinha `scheduler_executor`
- green focado: `1 passed in 0.34s`
- `test_strip_run.py`: `27 passed in 2.49s`
- `test_strip_scheduler.py` + `test_strip_scheduler_shadow_gate.py`: `8 passed in 0.20s`
- suite impactada: `45 passed in 2.95s`

Rodadas reais:

- artefato executor: `debug/performance_gates/strip_scheduler_executor_pipeline_20260508/decision.md`
- flag: `TRADUZAI_STRIP_SCHEDULER_EXECUTOR=1`
- elapsed executor: `115.3691s`
- peak RSS executor: `8560.73 MB`
- peak VRAM executor: `2608 MB`
- output compare: `PASS`, com `114` textos/regioes contra `114`
- project import: `PASS`
- scheduler shadow gate: `PASS`, task_count `466`, CPU/GPU `158/308`, `max_gpu_parallel=1`
- visual review todas as paginas: `PASS`, pixel diff `0.0`
- artefato baseline fresco: `debug/performance_gates/resource_profile_pipeline_current_default_20260508/decision.md`
- baseline fresco sem flag: `119.6386s`, RSS `8537.824 MB`, VRAM aproximada `2602 MB`

Decisao:

- resultado: `aprovado como telemetria/executor sequencial seguro, nao aprovado como speedup`
- motivo: o output candidato e equivalente e passa o shadow gate, mas a implementacao ainda e sequencial; o delta observado de `4.2695s` pode ser cache/variacao e RAM continua alta
- recomendacao: manter `TRADUZAI_STRIP_SCHEDULER_EXECUTOR=0` por default; usar a flag apenas para diagnostico/gate de futuros candidatos

## Fase 6d: Process Band Stage Contract

Arquivos modificados:

- `pipeline/strip/scheduler.py`
- `pipeline/tests/test_strip_scheduler.py`

Artefato criado:

- `debug/performance_gates/process_band_stage_contract_20260508/decision.md`
- `debug/performance_gates/process_band_stage_contract_20260508/summary.json`

Comportamento implementado:

- adiciona `evaluate_process_band_stage_contract_gate`
- documenta os estagios logicos atuais de `process_band`: `ocr`, `review_layout`, `translate`, `inpaint`, `typeset`, `copy_back`
- declara recursos por estagio: CPU `3`, GPU `2`, network `1`
- bloqueia paralelismo real enquanto existirem `shared_gpu_models` e `ordered_band_context`

Evidencia de TDD/testes:

- red: `1 failed`, porque `pipeline.strip.scheduler` nao tinha `evaluate_process_band_stage_contract_gate`
- green focado: `1 passed in 0.11s`
- suite scheduler: `9 passed in 0.21s`
- atualizacao apos contratos visuais: red `1 failed`, porque o gate ainda reportava `mutates_band_state`; green `1 passed in 0.15s`
- atualizacao apos contrato de commit final: red `1 failed`, porque `final_band_commit` ainda aparecia como bloqueio; green `1 passed in 0.14s`
- atualizacao apos contrato de contexto ordenado: red `1 failed`, porque `band_history` e `running_glossary` ainda apareciam separados; green `1 passed in 0.13s`

Decisao:

- resultado: `aprovado como contrato de prontidao, bloqueado para paralelismo`
- motivo: agora o plano tem um gate explicito que mostra por que `process_band` ainda e monolitico e quais dependencias precisam virar contratos imutaveis
- recomendacao: nao ativar paralelismo real; proximo passo e separar a posse dos modelos GPU antes de novo executor

## Fase 6e: OCR Stage Output Contract

Arquivos modificados:

- `pipeline/strip/process_bands.py`
- `pipeline/tests/test_strip_process_bands.py`

Artefatos criados:

- `debug/performance_gates/process_band_ocr_stage_output_20260508/decision.md`
- `debug/performance_gates/process_band_ocr_stage_output_20260508/summary.json`

Comportamento implementado:

- adiciona `BandStageOutput`
- adiciona `_run_band_ocr_stage`
- o estagio OCR de `process_band` agora retorna snapshot via `to_page_dict()`
- os updates de perf do OCR precomputado ficam isolados em `perf_updates`
- o fluxo restante continua sequencial e sem paralelismo real

Evidencia de TDD/testes:

- red: `1 failed`, porque `strip.process_bands` nao tinha `_run_band_ocr_stage`
- green focado: `1 passed in 0.46s`
- arquivo completo `test_strip_process_bands.py`: `22 passed in 17.14s`
- suite impactada: `58 passed in 16.67s`

Decisao:

- resultado: `PASS_WITH_LIMITATION`
- motivo: OCR agora tem o primeiro contrato real de saida de estagio, mas `process_band` ainda muta `Band` nos estagios seguintes
- recomendacao: extrair `review_layout` e `translate` para resultados de estagio antes de qualquer sobreposicao CPU

## Fase 6f: Review/Translate Stage Output Contract

Arquivos modificados:

- `pipeline/strip/process_bands.py`
- `pipeline/tests/test_strip_process_bands.py`

Artefatos criados:

- `debug/performance_gates/process_band_review_translate_stage_output_20260508/decision.md`
- `debug/performance_gates/process_band_review_translate_stage_output_20260508/summary.json`

Comportamento implementado:

- adiciona `_run_review_layout_stage`
- adiciona `_run_translate_stage`
- `review_layout` retorna `BandStageOutput` com snapshot da pagina revisada/enriquecida
- `translate` retorna `BandStageOutput` com metadados OCR mesclados ao payload traduzido
- `process_band` continua sequencial e consome os snapshots antes dos proximos estagios

Evidencia de TDD/testes:

- red review_layout: `1 failed`, porque `strip.process_bands` nao tinha `_run_review_layout_stage`
- red translate: `1 failed`, porque `strip.process_bands` nao tinha `_run_translate_stage`
- green focado: `2 passed in 0.33s`
- arquivo completo `test_strip_process_bands.py`: `24 passed in 16.99s`
- suite impactada: `60 passed in 17.98s`

Decisao:

- resultado: `PASS_WITH_LIMITATION`
- motivo: `review_layout` e `translate` agora tem contratos de saida, mas ainda dependem de `band_history`, glossario/contexto mutavel e fluxo sequencial
- recomendacao: extrair `inpaint`, `typeset` e `copy_back` antes de reavaliar qualquer executor paralelo

## Fase 6g: Visual Stage Output Contract

Arquivos modificados:

- `pipeline/strip/process_bands.py`
- `pipeline/strip/scheduler.py`
- `pipeline/tests/test_strip_process_bands.py`
- `pipeline/tests/test_strip_scheduler.py`

Artefatos criados:

- `debug/performance_gates/process_band_visual_stage_output_20260508/decision.md`
- `debug/performance_gates/process_band_visual_stage_output_20260508/summary.json`

Comportamento implementado:

- adiciona `BandImageStageOutput`
- adiciona `_run_inpaint_stage`
- adiciona `_run_typeset_stage`
- adiciona `_run_copy_back_stage`
- `process_band` passa a consumir snapshots de imagem nos estagios visuais
- `cleaned_slice` e `rendered_slice` so sao atribuidos na `Band` no commit final
- o gate `process_band_stage_contract` troca o bloqueio antigo `mutates_band_state` por `final_band_commit`

Evidencia de TDD/testes:

- red visual: `2 failed`, porque `strip.process_bands` nao tinha `_run_inpaint_stage` nem `_run_typeset_stage`
- green visual focado: `2 passed in 0.26s`
- arquivo completo `test_strip_process_bands.py`: `26 passed in 26.27s`
- suite impactada: `62 passed in 26.84s`
- red gate: `1 failed`, porque o gate ainda reportava `mutates_band_state`
- green gate focado: `1 passed in 0.15s`

Decisao:

- resultado: `PASS_WITH_LIMITATION`
- motivo: todos os estagios de `process_band` agora tem objetos de saida, mas o commit final da `Band`, `band_history`, `running_glossary` e GPU compartilhada ainda bloqueiam paralelismo real
- recomendacao: criar contrato explicito de commit final da banda antes de qualquer executor paralelo

## Fase 6h: Final Band Commit Contract

Arquivos modificados:

- `pipeline/strip/process_bands.py`
- `pipeline/strip/scheduler.py`
- `pipeline/tests/test_strip_process_bands.py`
- `pipeline/tests/test_strip_scheduler.py`

Artefatos criados:

- `debug/performance_gates/process_band_final_commit_contract_20260508/decision.md`
- `debug/performance_gates/process_band_final_commit_contract_20260508/summary.json`

Comportamento implementado:

- adiciona `_commit_band_outputs`
- centraliza a escrita final de `cleaned_slice`, `rendered_slice` e `ocr_result`
- copia imagens e OCR antes de gravar na `Band`
- substitui commits diretos nos retornos de `process_band`
- atualiza `process_band_stage_contract` para remover `final_band_commit` dos bloqueios

Evidencia de TDD/testes:

- red commit: `1 failed`, porque `strip.process_bands` nao tinha `_commit_band_outputs`
- green commit: `1 passed in 0.27s`
- red gate: `1 failed`, porque `final_band_commit` ainda aparecia como bloqueio
- green gate: `1 passed in 0.14s`
- suite impactada: `63 passed in 24.01s`

Decisao:

- resultado: `PASS_WITH_LIMITATION`
- motivo: a mutacao final agora tem contrato explicito, mas o gate segue `BLOCK` por `shared_gpu_models`, `band_history` e `running_glossary`
- recomendacao: isolar contexto ordenado de banda e atualizacoes de glossario antes de qualquer executor paralelo

## Fase 6i: Ordered Band Context Contract

Arquivos modificados:

- `pipeline/strip/run.py`
- `pipeline/strip/scheduler.py`
- `pipeline/tests/test_strip_run.py`
- `pipeline/tests/test_strip_scheduler.py`

Artefatos criados:

- `debug/performance_gates/ordered_band_context_contract_20260508/decision.md`
- `debug/performance_gates/ordered_band_context_contract_20260508/summary.json`

Comportamento implementado:

- adiciona `OrderedBandContextSnapshot`
- adiciona `_build_ordered_band_context_snapshot`
- adiciona `_merge_ordered_band_context_after_commit`
- `run_chapter` passa snapshot copiado de `band_history` e `glossario` para `process_band`
- atualizacoes de glossario e historico passam por merge explicito apos o commit da banda
- o gate consolida `band_history` e `running_glossary` em `ordered_band_context`

Evidencia de TDD/testes:

- red context: `2 failed`, porque `strip.run` nao tinha os helpers de contexto ordenado
- green context: `2 passed in 0.28s`
- arquivo completo `test_strip_run.py`: `29 passed in 3.60s`
- red gate: `1 failed`, porque `band_history` e `running_glossary` ainda apareciam separados
- green gate: `1 passed in 0.13s`
- suite impactada: `65 passed in 22.50s`

Decisao:

- resultado: `PASS_WITH_LIMITATION`
- motivo: contexto ordenado agora tem contrato, mas ele ainda serializa review/layout e propagacao de glossario
- recomendacao: definir contrato de posse GPU unico para OCR/inpaint antes de qualquer executor paralelo

## Fase 6j: GPU Ownership Contract

Arquivos modificados:

- `pipeline/strip/scheduler.py`
- `pipeline/tests/test_strip_scheduler.py`

Artefatos criados/atualizados:

- `debug/performance_gates/process_band_gpu_ownership_contract_20260508/decision.md`
- `debug/performance_gates/process_band_gpu_ownership_contract_20260508/summary.json`
- `debug/performance_gates/process_band_stage_contract_20260508/decision.md`
- `debug/performance_gates/process_band_stage_contract_20260508/summary.json`

Comportamento implementado:

- adiciona `ProcessBandGpuOwnershipContract`
- adiciona `build_process_band_gpu_ownership_contract`
- adiciona `evaluate_process_band_gpu_ownership_gate`
- define `strip_single_gpu_lane` com `max_concurrent=1`
- declara `ocr` e `inpaint` como estagios GPU nessa fila unica
- proibe sobreposicao entre OCR e inpaint no contrato
- remove `shared_gpu_models` dos bloqueios atuais do `process_band_stage_contract`
- o gate de prontidao segue `BLOCK` apenas por `ordered_band_context`

Evidencia de TDD/testes:

- red: `2 failed`, porque o gate de posse GPU nao existia e `shared_gpu_models` ainda aparecia como bloqueio
- green focado: `2 passed in 0.11s`
- arquivo `test_strip_scheduler.py`: `7 passed in 0.10s`
- suite impactada de contratos: `66 passed in 16.71s`

Decisao:

- resultado: `PASS_WITH_LIMITATION`
- motivo: a posse da GPU agora esta contratada por fila unica, mas o runtime continua sequencial e nao ha aceleracao
- recomendacao: isolar/particionar `ordered_band_context` antes de testar qualquer sobreposicao real

## Fase 6k: Ordered Context Release

Arquivos modificados:

- `pipeline/strip/process_bands.py`
- `pipeline/strip/run.py`
- `pipeline/strip/scheduler.py`
- `pipeline/tests/test_strip_process_bands.py`
- `pipeline/tests/test_strip_run.py`
- `pipeline/tests/test_strip_scheduler.py`

Artefatos criados/atualizados:

- `debug/performance_gates/process_band_ordered_context_release_20260508/decision.md`
- `debug/performance_gates/process_band_ordered_context_release_20260508/summary.json`
- `debug/performance_gates/process_band_stage_contract_20260508/decision.md`
- `debug/performance_gates/process_band_stage_contract_20260508/summary.json`

Comportamento implementado:

- adiciona `ordered_context_after_translate_callback` em `process_band`
- `process_band` chama o callback depois de `translate` e antes de `inpaint`
- o callback recebe snapshot copiado da pagina traduzida
- `run_chapter` usa o callback para mesclar history/glossario antes dos estagios visuais terminarem
- `run_chapter` mantem fallback de merge apos `process_band` para bandas sem traducao ou retornos antecipados
- adiciona `ProcessBandOrderedContextReleaseContract`
- adiciona `evaluate_process_band_ordered_context_release_gate`
- o `process_band_stage_contract` passa para `PASS`, sem bloqueios de prontidao restantes

Evidencia de TDD/testes:

- red scheduler: `2 failed`, porque o gate de release nao existia e o stage contract ainda estava `BLOCK`
- red process_band: `1 failed`, porque `process_band` nao aceitava `ordered_context_after_translate_callback`
- red run_chapter: `1 failed`, porque a adicao de glossario do callback nao chegava na proxima banda
- green scheduler focado: `2 passed in 0.13s`
- green process_band focado: `1 passed in 0.33s`
- green run_chapter focado: `1 passed in 0.33s`
- suite impactada de contratos: `69 passed in 17.17s`

Decisao:

- resultado: `PASS_WITH_LIMITATION`
- motivo: o contexto ordenado agora tem release apos `translate`, mas o runtime continua sequencial e nao ha speedup real
- recomendacao naquele ponto: criar executor experimental de sobreposicao atras de flag; esse executor foi implementado e avaliado na Fase 6l

## Fase 6l: Scheduler Overlap Executor

Arquivos modificados:

- `pipeline/strip/process_bands.py`
- `pipeline/strip/run.py`
- `pipeline/tests/test_strip_process_bands.py`
- `pipeline/tests/test_strip_run.py`

Artefatos criados/atualizados:

- `debug/performance_gates/strip_scheduler_overlap_pipeline_20260508/resources/resources.json`
- `debug/performance_gates/strip_scheduler_overlap_typeset_lock_pipeline_20260508/resources/resources.json`
- `debug/performance_gates/strip_scheduler_overlap_typeset_lock_pipeline_20260508/performance_gate/summary.json`
- `debug/performance_gates/strip_scheduler_overlap_typeset_lock_pipeline_20260508/output_compare/summary.json`
- `debug/performance_gates/strip_scheduler_overlap_typeset_lock_pipeline_20260508/project_import_gate/summary.json`
- `debug/performance_gates/strip_scheduler_overlap_typeset_lock_pipeline_20260508/scheduler_shadow_gate/summary.json`
- `debug/performance_gates/strip_scheduler_overlap_typeset_lock_pipeline_20260508/visual_review_all_pages.html`
- `debug/performance_gates/strip_scheduler_overlap_typeset_lock_pipeline_20260508/decision.md`

Comportamento implementado:

- env experimental `TRADUZAI_STRIP_SCHEDULER_EXECUTOR=overlap`
- modo reportado em telemetria: `overlap_context_release`
- `run_chapter` usa `ThreadPoolExecutor(max_workers=2)`
- a proxima banda pode iniciar depois que a banda anterior libera history/glossario apos `translate`
- `gpu_stage_lock` serializa `ocr` e `inpaint`
- `typeset_stage_lock` serializa `typeset` para evitar concorrencia em FreeType/matplotlib
- o modo antigo `TRADUZAI_STRIP_SCHEDULER_EXECUTOR=1` segue como `sequential_safe`

Evidencia de TDD/testes:

- red GPU lock: `1 failed`, porque `process_band()` nao aceitava `gpu_stage_lock`
- red overlap executor: `1 failed`, porque a banda seguinte nao iniciava antes da anterior terminar
- green GPU lock: `1 passed in 0.40s`
- green overlap executor: `1 passed in 0.50s`
- compatibilidade `TRADUZAI_STRIP_SCHEDULER_EXECUTOR=1`: `1 passed in 0.49s`
- red typeset lock: `1 failed`, porque `process_band()` nao aceitava `typeset_stage_lock`
- green locks GPU/typeset: `2 passed in 0.39s`
- suite impactada final: `72 passed in 18.02s`

Rodada real sem lock de typeset:

- resource profile: `FAIL`
- exit code: `3221225477`
- elapsed: `55.9598s`
- pico RSS: `4230.742 MB`
- pico VRAM aproximada: `2164 MB`
- causa observada: fatal access violation em `pipeline/typesetter/renderer.py`, dentro do caminho de renderizacao/typeset

Rodada real estabilizada:

- resource profile: `PASS`
- elapsed: `128.178s`
- baseline controlado: `130.4294s`
- baseline fresco sem flag: `119.6386s`
- ganho contra baseline controlado: `2.2514s`
- delta contra baseline fresco: `+8.5394s`
- pico RSS: `7770.172 MB` contra `6341.109 MB` do baseline controlado
- pico VRAM aproximada: `2843 MB` contra `3017 MB` do baseline controlado
- output compare: `PASS`, `27/27` paginas, `114/114` textos, `114/114` regioes traduzidas, `114/114` blocos de inpaint
- project import: `PASS`, `27` paginas, `114` text layers, `162` imagens verificadas, `0` imagens ausentes
- scheduler shadow gate: `PASS`, `466` tarefas, CPU/GPU `158/308`, `max_gpu_parallel=1`
- visual review todas as paginas: `PASS`, `pixel_diff_rate=0.0`, `different_text_rate=0.0`, `missing_text_rate=0.0`, `fallback_rate=0.0`

Decisao:

- resultado: `PASS_ESTRUTURAL_FAIL_PERF`
- motivo: o executor overlap passa output/import/visual/shadow depois dos locks, mas o ganho e pequeno contra o baseline controlado, piora contra o baseline fresco, aumenta RAM e a tentativa inicial crashou sem serializar typeset
- recomendacao: manter `TRADUZAI_STRIP_SCHEDULER_EXECUTOR=overlap` desligado por default e usar apenas como experimento/gate

## Gate: Editor Abre Project.json Gerado Real

Arquivos modificados:

- `e2e/editor-rebuild.spec.ts`
- `src/lib/e2e/fixtureProject.ts`

Artefato criado:

- `debug/performance_gates/real_project_editor_open_20260508/decision.md`

Comportamento implementado:

- teste Playwright `@real-project` injeta o `project.json` gerado em `debug/performance_gates/strip_scheduler_overlap_typeset_lock_pipeline_20260508/work/project.json`
- o harness E2E aceita `window.__TRADUZAI_E2E_PROJECT__`
- o projeto real e normalizado para o editor React, gerando IDs quando ausentes e normalizando `translated`/`traduzido`, bbox e estilo
- as imagens locais reais sao substituidas por bitmaps fixture E2E; a presenca dos assets reais continua coberta pelo `project_import_gate`

Evidencia de TDD/testes:

- red: `npx playwright test e2e/editor-rebuild.spec.ts --grep "@real-project"` falhou porque o editor abriu a fixture padrao e nao mostrou `Grand Finale`
- green focado: `npx playwright test e2e/editor-rebuild.spec.ts --grep "@real-project"` passou, `1 passed`
- `npm run check`: `PASS`
- `npx playwright test e2e/editor-rebuild.spec.ts --grep "@smoke|@manual-flow|@real-project"`: `3 passed in 29.4s`

Validacao do projeto real:

- obra: `Grand Finale`
- paginas: `27`
- primeira pagina: `9` text layers
- editor mostra contador `1/27`
- estado interno do editor tem `9` layers
- todas as layers hidratadas tem `id`
- texto traduzido real aparece nas layers

Decisao:

- resultado: `PASS_WITH_LIMITATION`
- motivo: o editor React abre e hidrata um `project.json` gerado real, mas os bitmaps no browser ainda sao fixtures por limitacao do E2E Vite fora do Tauri
- recomendacao: considerar o item do DoD coberto em conjunto com `project_import_gate`

## Experimento: Combined Fast Paths

Artefatos criados:

- `debug/performance_gates/combined_fast_paths_default_20260509/config.json`
- `debug/performance_gates/combined_fast_paths_default_20260509/resources/resources.json`
- `debug/performance_gates/combined_fast_paths_candidate_20260509/config.json`
- `debug/performance_gates/combined_fast_paths_candidate_20260509/resources/resources.json`
- `debug/performance_gates/combined_fast_paths_candidate_20260509/performance_gate/summary.json`
- `debug/performance_gates/combined_fast_paths_candidate_20260509/output_compare/summary.json`
- `debug/performance_gates/combined_fast_paths_candidate_20260509/project_import_gate/summary.json`
- `debug/performance_gates/combined_fast_paths_candidate_20260509/visual_review_all_pages.html`
- `debug/performance_gates/combined_fast_paths_candidate_20260509/decision.md`

Flags testadas:

- `TRADUZAI_STRIP_FAST_WHITE_NARRATION=1`
- `TRADUZAI_SMART_SKIP=1`

Resultado pareado:

- baseline fresco: `181.9548s`
- candidato: `167.195s`
- economia pareada: `14.7598s`
- meta do plano: `<=113.65s`
- pico RSS baseline/candidato: `8546.637 MB` / `8545.18 MB`
- pico VRAM aproximada baseline/candidato: `3070 MB` / `3104 MB`
- OCR: `65.1973s` -> `64.7246s`
- inpaint: `72.526s` -> `58.1191s`
- blocos restantes para LaMA: `45` -> `36`
- fast white fills: `21` -> `35`
- Smart Skip aplicado: `4`

Gates:

- resource profile baseline: `PASS`
- resource profile candidato: `PASS`
- performance gate candidato: `PASS`
- output compare: `PASS`, `27/27` paginas, `114/114` textos, `114/114` regioes traduzidas, `audited_skip_count=4`
- project import: `PASS`, `27` paginas, `114` text layers, `162` imagens verificadas, `0` ausentes
- visual review todas as paginas: `PASS`, maior `pixel_diff_rate=0.037944`, `different_text_rate=0.0`, `missing_text_rate=0.0`

Decisao:

- resultado: `PASS_ESTRUTURAL_FAIL_DEFAULT`
- motivo: a combinacao reduz inpaint e passa qualidade/importacao, mas fica muito acima da meta absoluta, depende de baseline pareado anormalmente lento e combina flags que individualmente nao foram aprovadas como default
- recomendacao: manter desligado por default; usar apenas como experimento controlado se quiser comparar em mais capitulos

## Validacao Final Atualizada

Suite impactada:

- comando: `pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_strip_smart_skip.py pipeline\tests\test_strip_process_bands.py pipeline\tests\test_strip_run.py pipeline\tests\test_main_emit.py pipeline\tests\test_main_strip_config.py pipeline\tests\test_analyze_pipeline_run.py pipeline\tests\test_performance_gate.py pipeline\tests\test_smart_skip_shadow_gate.py pipeline\tests\test_macro_ocr_mapping.py pipeline\tests\test_macro_ocr_shadow_gate.py pipeline\tests\test_macro_ocr_actual_shadow.py pipeline\tests\test_compare_pipeline_outputs.py pipeline\tests\test_export_visual_review_sheet.py pipeline\tests\test_resource_profile.py pipeline\tests\test_translation_batch_gate.py pipeline\tests\test_project_import_gate.py pipeline\tests\test_runtime_profiles.py pipeline\tests\test_strip_scheduler.py -q -p no:cacheprovider --basetemp=C:\tmp\pytest-traduzai-smart-real-impact`
- resultado: `145 passed in 20.36s`

Project writer:

- comando: `..\pipeline\venv\Scripts\python.exe -m pytest tests\test_project_writer.py -q -p no:cacheprovider --basetemp=C:\tmp\pytest-traduzai-project-writer-smart-real` a partir de `D:\TraduzAi\pipeline`
- resultado: `6 passed in 0.07s`

Editor E2E parcial:

- comando: `npx playwright test e2e/editor-rebuild.spec.ts --grep "@smoke|@manual-flow"`
- resultado: `2 passed in 26.2s`
- cobertura: prova que o editor abre e o fluxo manual chega ao editor com fixture E2E; nao prova abertura do `traduzido2` real
- comando: `npx playwright test e2e/editor-rebuild.spec.ts --grep "@smoke|@manual-flow|@real-project"`
- resultado: `3 passed in 29.4s`
- cobertura adicional: prova abertura/hidratacao de `project.json` real gerado em `debug/performance_gates/strip_scheduler_overlap_typeset_lock_pipeline_20260508/work/project.json`

Higiene:

- comando: `git diff --check`
- resultado: exit code `0`, apenas warnings de normalizacao CRLF
- suite de impacto scheduler shadow final: `11 passed in 0.29s`
- suite impactada scheduler executor: `45 passed in 2.95s`
- suite scheduler apos stage contract: `9 passed in 0.21s`
- gpu ownership focado: `2 passed in 0.11s`
- arquivo `test_strip_scheduler.py` apos gpu ownership: `7 passed in 0.10s`
- suite impactada de contratos apos gpu ownership: `66 passed in 16.71s`
- ordered context release focado: scheduler `2 passed in 0.13s`; process_band `1 passed in 0.33s`; run_chapter `1 passed in 0.33s`
- suite impactada de contratos apos ordered context release: `69 passed in 17.17s`
- scheduler overlap focado: GPU lock `1 passed in 0.40s`; overlap executor `1 passed in 0.50s`; compatibilidade sequencial `1 passed in 0.49s`; locks GPU/typeset `2 passed in 0.39s`
- suite impactada apos scheduler overlap: `72 passed in 18.02s`
- arquivo `test_strip_process_bands.py` apos OCR stage output: `22 passed in 17.14s`
- suite impactada apos OCR stage output: `58 passed in 16.67s`
- arquivo `test_strip_process_bands.py` apos review/translate stage output: `24 passed in 16.99s`
- suite impactada apos review/translate stage output: `60 passed in 17.98s`
- arquivo `test_strip_process_bands.py` apos visual stage output: `26 passed in 26.27s`
- suite impactada apos visual stage output: `62 passed in 26.84s`
- gate de prontidao apos visual stage output: `1 passed in 0.15s`
- commit final focado: `1 passed in 0.27s`
- gate de prontidao apos commit final: `1 passed in 0.14s`
- suite impactada apos commit final: `63 passed in 24.01s`
- contexto ordenado focado: `2 passed in 0.28s`
- arquivo `test_strip_run.py` apos contexto ordenado: `29 passed in 3.60s`
- gate de prontidao apos contexto ordenado: `1 passed in 0.13s`
- suite impactada apos contexto ordenado: `65 passed in 22.50s`
- processos Python apos os testes: apenas `python -m server` e `python -m worker`; nenhum `pytest` remanescente

## Experimento: Fast White Narration

Artefatos criados:

- `debug/performance_gates/fast_white_narration_pipeline_20260508/config.json`
- `debug/performance_gates/fast_white_narration_pipeline_20260508/resources/resources.json`
- `debug/performance_gates/fast_white_narration_pipeline_20260508/performance_gate/summary.json`
- `debug/performance_gates/fast_white_narration_pipeline_20260508/output_compare/summary.json`
- `debug/performance_gates/fast_white_narration_pipeline_20260508/project_import_gate/summary.json`
- `debug/performance_gates/fast_white_narration_pipeline_20260508/visual_review_sheet.html`
- `debug/performance_gates/fast_white_narration_pipeline_20260508/visual_review_all_pages.html`
- `debug/performance_gates/fast_white_narration_pipeline_20260508/decision.md`

Config aplicada:

- `TRADUZAI_STRIP_FAST_WHITE_NARRATION=1`

Evidencia real:

- comando: `pipeline\venv\Scripts\python.exe pipeline\tools\measure_resource_profile.py --out D:\TraduzAi\debug\performance_gates\fast_white_narration_pipeline_20260508\resources --sample-interval 2.0 --timeout-seconds 420 -- pipeline\venv\Scripts\python.exe pipeline\main.py D:\TraduzAi\debug\performance_gates\fast_white_narration_pipeline_20260508\config.json`
- resultado: `PASS`
- tempo total medido: `125.859s`
- baseline de recursos: `130.4294s`
- economia real: `4.5704s`
- pico RSS: `8143.293 MB` contra `6341.109 MB`
- VRAM total aproximada: `4575 MB` contra `3017 MB`
- inpaint por estagios: `45.6371s` contra `49.7094s`
- bandas com LaMA: `37` contra `43`
- blocos restantes para LaMA: `39` contra `45`

Gates:

- performance gate: `PASS`
- output compare vs `traduzido2`: `PASS`
- project import gate: `PASS`
- visual review amostrado: `PASS`
- visual review todas as paginas: `PASS`
- maior `pixel_diff_rate` amostrado: `2.4953%` na pagina `1`
- diferenca/falta textual estrutural: `0.0%`

Decisao:

- resultado: `reprovar como default`
- motivo: existe ganho real pequeno, mas a RAM e a VRAM medidas aumentaram muito e a economia fica longe da meta
- recomendacao: manter `TRADUZAI_STRIP_FAST_WHITE_NARRATION=0` por default; considerar apenas como diagnostico/performance experimental em outros capitulos

## Fase 2: Smart Skip Real Conservador

Arquivos modificados:

- `pipeline/strip/process_bands.py`
- `pipeline/strip/run.py`
- `pipeline/main.py`
- `pipeline/tests/test_strip_process_bands.py`
- `pipeline/tests/test_strip_run.py`
- `pipeline/tests/test_main_emit.py`

Comportamento implementado:

- flag real `TRADUZAI_SMART_SKIP=1`
- aplica skip apenas quando todos os textos da banda sao candidatos seguros e `not_safe_count == 0`
- grava `skip_reason="smart_skip"` e `smart_skip_decision` por texto
- agrega `smart_skip_real_candidate_count`, `smart_skip_real_not_safe_count`, `smart_skip_real_applied_band_count` e categorias em `strip_perf_summary`
- preserva a auditoria no `project.json`, no renderer normalizado e no alias legado `textos`

Evidencia de TDD:

- teste vermelho inicial para aplicacao real: `ImportError: cannot import name '_apply_smart_skip_real'`
- apos implementacao: `2 passed in 0.20s`
- teste vermelho de resumo: `KeyError: smart_skip_real_candidate_count`
- apos agregacao: `1 passed in 0.25s`
- teste vermelho de emissao do `project.json`: `KeyError: 'skip_reason'`
- apos preservar campos no layer: `1 passed in 0.42s`
- suite focada: `51 passed in 17.16s`

Artefatos criados:

- `debug/performance_gates/smart_skip_real_pipeline_20260508_isolated/config.json`
- `debug/performance_gates/smart_skip_real_pipeline_20260508_isolated/resources/resources.json`
- `debug/performance_gates/smart_skip_real_pipeline_20260508_isolated/performance_gate/summary.json`
- `debug/performance_gates/smart_skip_real_pipeline_20260508_isolated/output_compare/summary.json`
- `debug/performance_gates/smart_skip_real_pipeline_20260508_isolated/project_import_gate/summary.json`
- `debug/performance_gates/smart_skip_real_pipeline_20260508_isolated/visual_review_all_pages.html`
- `debug/performance_gates/smart_skip_real_pipeline_20260508_isolated/decision.md`

Evidencia real:

- comando: `TRADUZAI_SMART_SKIP=1 pipeline\venv\Scripts\python.exe pipeline\tools\measure_resource_profile.py --out D:\TraduzAi\debug\performance_gates\smart_skip_real_pipeline_20260508_isolated\resources --sample-interval 2.0 --timeout-seconds 420 -- pipeline\venv\Scripts\python.exe pipeline\main.py D:\TraduzAi\debug\performance_gates\smart_skip_real_pipeline_20260508_isolated\config.json`
- resultado: `PASS`
- tempo baseline de recursos: `130.4294s`
- tempo candidato: `130.1651s`
- economia real: `0.2643s`
- pico RSS: `6341.109 MB` -> `7485.855 MB`
- VRAM total aproximada: `3017 MB` -> `4575 MB`
- bandas aplicadas: `4`
- candidatos seguros: `4`
- candidatos inseguros: `110`
- blocos restantes para inpaint: `45` -> `42`

Gates:

- performance gate: `PASS`
- output compare: `PASS`
- project import gate: `PASS`
- visual review todas as paginas: `PASS`
- maior `pixel_diff_rate`: `3.7944%` na pagina `1`
- diferenca/falta textual estrutural: `0.0%`
- `audited_skip_count`: `4`

Rodada invalida preservada como risco:

- pasta: `debug/performance_gates/smart_skip_real_pipeline_20260508`
- motivo: foi executada com fast paths de inpaint desligados, entao nao compara contra o baseline
- tempo: `156.5476s`
- blocos restantes para inpaint: `101`

Decisao:

- resultado: `reprovar como default`
- motivo: qualidade e importacao passam, mas a economia real fica dentro de ruido de medicao e RAM/VRAM sobem
- recomendacao: manter `TRADUZAI_SMART_SKIP=0` por default; permitir apenas experimento controlado por flag

## Macro OCR: Classificacao De Risco De Divergencias

Arquivos modificados:

- `pipeline/ocr/macro_ocr.py`
- `pipeline/tools/run_macro_ocr_actual_shadow.py`
- `pipeline/tests/test_macro_ocr_actual_shadow.py`

Comportamento implementado:

- separa divergencias em `line_marker_artifact`, `minor_ocr_variation`, `numeric_token_change` e `material`
- adiciona `fallback_required_count`, `acceptable_variation_count` e `fallback_required_text_rate`
- custo ajustado por fallback passa a usar `fallback_required_count`, mantendo `numeric_token_change` como risco conservador
- variacoes pequenas de OCR ficam auditadas, mas nao viram fallback obrigatorio

Evidencia de TDD:

- red: `2 failed` porque variacao pequena ainda era `material` e o resumo nao tinha `minor_ocr_variation_count`
- green focado: `2 passed in 0.22s`
- arquivo completo: `10 passed in 0.30s`
- suite impactada: `40 passed in 2.76s`

Artefatos criados:

- `debug/performance_gates/macro_ocr_actual_shadow_windows_full_conservative_risk_gate_20260508/summary.json`
- `debug/performance_gates/macro_ocr_actual_shadow_windows_full_aggressive_risk_gate_20260508/summary.json`
- `debug/performance_gates/macro_ocr_actual_shadow_windows_full_max5_risk_gate_20260508/summary.json`
- `debug/performance_gates/macro_ocr_risk_classification_20260508/decision.md`

Resultados:

- conservador: `FAIL`, reducao ajustada `14.91%`
- agressivo: `FAIL`, reducao ajustada `24.56%` e `different_text_rate=27.19%`
- max_blocks=5: `FAIL`, reducao ajustada `25.44%`, mas `different_text_rate=28.07%`

Decisao:

- resultado: `melhor diagnostico, modo real ainda bloqueado`
- motivo: existe uma configuracao que cruza reducao ajustada, mas ela depende de aceitar diferencas pequenas demais para liberar sem gate visual/textual adicional
- proxima acao: separar mudancas numericas seguras de codigos/alvos perigosos e tentar reduzir `different_text_rate` abaixo de `25%` sem perder reducao ajustada

## Macro OCR: Classificacao Numerica De Divergencias

Arquivos modificados:

- `pipeline/ocr/macro_ocr.py`
- `pipeline/tools/run_macro_ocr_actual_shadow.py`
- `pipeline/tests/test_macro_ocr_actual_shadow.py`

Comportamento implementado:

- separa variacoes numericas auditaveis em `numeric_confusable_variation` e `episode_marker_variation`
- adiciona `line_marker_minor_variation` para casos onde a janela macro remove marcador de linha e aplica pequena variacao OCR
- mantem alteracoes perigosas em `numeric_token_change`
- o custo ajustado por fallback continua usando apenas `material` + `numeric_token_change`
- variacoes auditaveis nao entram como fallback obrigatorio, mas ainda contam em `different_text_rate`

Evidencia de TDD:

- red: testes novos falharam porque variacoes numericas seguras ainda eram `numeric_token_change` e o resumo nao tinha os novos contadores
- green focado: `3 passed in 0.28s`
- arquivo completo apos ajuste: `13 passed in 0.50s`
- limpeza final: `13 passed in 0.84s`
- suite impactada: `43 passed in 4.47s`

Artefatos criados:

- `debug/performance_gates/macro_ocr_actual_shadow_windows_full_conservative_numeric_gate_20260508/summary.json`
- `debug/performance_gates/macro_ocr_actual_shadow_windows_full_aggressive_numeric_gate_20260508/summary.json`
- `debug/performance_gates/macro_ocr_actual_shadow_windows_full_max5_numeric_gate_20260508/summary.json`
- `debug/performance_gates/macro_ocr_numeric_classification_20260508/decision.md`

Resultados:

- conservador: `FAIL`, reducao ajustada `20.18%`, fallback obrigatorio `8.77%`, diferenca total `23.68%`
- agressivo: `FAIL`, reducao ajustada `29.82%`, fallback obrigatorio `8.77%`, diferenca total `27.19%`
- max_blocks=5: `FAIL`, reducao ajustada `30.70%`, fallback obrigatorio `8.77%`, diferenca total `28.07%`

Decisao:

- resultado: `diagnostico melhor, modo real ainda bloqueado`
- motivo: agressivo/max5 ja economizam chamadas OCR depois do fallback estimado, mas ainda excedem o teto global de `different_text_rate=25%`
- recomendacao: manter `TRADUZAI_MACRO_OCR=0`; usar `TRADUZAI_MACRO_OCR_SHADOW=1` apenas para diagnostico
- proxima acao: criar fallback real por janela/bloco para os casos `fallback_required_count` e/ou definir um gate visual-textual especifico para aceitar variacoes auditaveis com revisao humana antes de qualquer default

## Macro OCR: Gate Fallback-Resolved

Arquivos modificados:

- `pipeline/ocr/macro_ocr.py`
- `pipeline/tools/run_macro_ocr_actual_shadow.py`
- `pipeline/tests/test_macro_ocr_actual_shadow.py`

Comportamento implementado:

- adiciona metricas `fallback_resolved_different_count` e `fallback_resolved_different_text_rate`
- adiciona flag opt-in `--gate-on-fallback-resolved-text`
- quando a flag esta ativa, o gate textual usa a diferenca restante depois de assumir fallback para os casos `material` e `numeric_token_change`
- quando a flag esta desligada, o gate antigo continua usando `different_text_rate`

Evidencia de TDD:

- red: `2 failed` porque os campos `fallback_resolved_*` e o parametro `gate_on_fallback_resolved_text` nao existiam
- green focado: `2 passed in 0.38s`
- arquivo completo: `15 passed in 0.70s`
- suite impactada: `45 passed in 6.31s`
- `git diff --check`: exit code `0`, apenas warnings de CRLF existentes
- processos Python apos os testes: apenas `python -m server` e `python -m worker`

Artefatos criados:

- `debug/performance_gates/macro_ocr_actual_shadow_windows_full_aggressive_fallback_resolved_gate_20260508/summary.json`
- `debug/performance_gates/macro_ocr_actual_shadow_windows_full_max5_fallback_resolved_gate_20260508/summary.json`
- `debug/performance_gates/macro_ocr_fallback_resolved_gate_20260508/decision.md`

Resultados:

- agressivo: `PASS`, reducao ajustada `29.82%`, taxa textual usada pelo gate `18.42%`, fallback obrigatorio `8.77%`
- max_blocks=5: `PASS`, reducao ajustada `30.70%`, taxa textual usada pelo gate `19.30%`, fallback obrigatorio `8.77%`

Decisao:

- resultado: `candidato aprovado para proximo prototipo, runtime real ainda bloqueado`
- motivo: o gate prova que Macro OCR pode valer com fallback seletivo, mas ainda e simulacao sobre baseline antigo
- recomendacao: nao ativar `TRADUZAI_MACRO_OCR=1` por default; o proximo passo deve ser fallback real atras de flag e rodada end-to-end com recursos/output/visual/importacao

## Macro OCR: Paridade Do Shadow Acoplado Ao Pipeline

Arquivos modificados:

- `pipeline/strip/run.py`
- `pipeline/tests/test_strip_run.py`

Comportamento implementado:

- o `macro_ocr_shadow` gravado no `project.json` agora reporta `fallback_resolved_different_count`, `fallback_resolved_different_text_rate`, `fallback_required_count`, `fallback_required_text_rate` e as categorias numericas do runner externo
- `fallback_adjusted_ocr_call_count` no shadow acoplado passa a usar `fallback_required_count`, nao apenas `material_different_count`
- `TRADUZAI_MACRO_OCR_GATE_FALLBACK_RESOLVED=1` muda somente o julgamento do shadow para usar `fallback_resolved_different_text_rate`
- sem essa env, o default continua conservador e usa `different_text_rate`

Evidencia de testes:

- red: `test_macro_ocr_shadow_reports_fallback_resolved_metrics` falhou com `KeyError: 'numeric_token_change_count'`
- green focado: `1 passed in 3.82s`
- cobertura da flag: `2 passed in 5.55s`
- suite impactada: `47 passed in 8.35s`

Artefato criado:

- `debug/performance_gates/macro_ocr_pipeline_shadow_parity_20260508/decision.md`

Decisao:

- resultado: `telemetria aprovada, runtime real ainda bloqueado`
- motivo: futuras rodadas end-to-end agora carregam no `project.json` a mesma leitura fallback-resolved do runner externo
- limitacao: isto nao reduz tempo ainda; o proximo passo continua sendo prototipo real de fallback seletivo atras de flag

## Macro OCR: Ponte Para OCR Precomputado Por Banda

Arquivos modificados:

- `pipeline/strip/process_bands.py`
- `pipeline/tests/test_strip_process_bands.py`

Comportamento implementado:

- `process_band` agora aceita `precomputed_ocr_page` como contrato opt-in
- quando esse payload e passado, a banda pula `runtime.run_ocr_stage`
- `_prepare_precomputed_ocr_page` copia `texts`, `_vision_blocks` e `_ocr_stats` para evitar mutacao direta de payload compartilhado
- metadados band-local (`numero`, `width`, `height`, `_band_y_top`, `_band_index`, `_source_page_number`) sao completados a partir da banda
- a telemetria registra `ocr_precomputed_page=True` e `ocr_runtime_skipped=True`
- o fluxo downstream continua igual: review/layout, traducao, inpaint, typeset e copy-back

Evidencia de TDD:

- red focado: `1 failed`, porque `process_band()` ainda nao aceitava `precomputed_ocr_page`
- green focado: `1 passed, 20 deselected in 0.55s`
- arquivo completo: `21 passed in 28.01s`
- suite impactada: `68 passed in 28.95s`

Artefato criado:

- `debug/performance_gates/macro_ocr_precomputed_band_bridge_20260508/decision.md`

Decisao:

- resultado: `ponte aprovada, runtime real ainda bloqueado`
- motivo: esta etapa permite injetar OCR macro/fallback por banda sem mexer no restante do pipeline, mas ainda nao cria o payload real nem reduz tempo por default
- proxima acao: alimentar `precomputed_ocr_page` em `run_chapter` atras de flag experimental, com Macro OCR por janelas e fallback seletivo real para os casos obrigatorios

## Macro OCR Real: Rodada Experimental Inicial

Arquivos modificados:

- `pipeline/strip/run.py`
- `pipeline/strip/process_bands.py`
- `pipeline/tests/test_strip_run.py`
- `pipeline/tests/test_strip_process_bands.py`

Comportamento implementado:

- `TRADUZAI_MACRO_OCR=1` cria OCR macro por pagina original e injeta `precomputed_ocr_page` nas bandas com texto aceito
- o default continua desligado
- o resumo de bandas agrega `ocr_precomputed_page`, `ocr_runtime_skipped`, `ocr_macro_ocr_real`, `ocr_macro_window_count`, `ocr_macro_ocr_block_count` e `ocr_macro_ocr_empty_record_count`

Evidencia de TDD/testes:

- red `run_chapter`: `1 failed`, porque Macro OCR real ainda nao chamava `recognize_macro_ocr_windows`
- green focado `run_chapter`: `1 passed, 20 deselected in 7.07s`
- red telemetria `process_band`: `1 failed`, por ausencia de `ocr_macro_ocr_real`
- green telemetria: `1 passed, 20 deselected in 0.45s`
- red resumo: `1 failed`, por ausencia de contadores Macro OCR real no resumo
- green resumo: `1 passed, 21 deselected in 0.42s`
- arquivos completos: `test_strip_process_bands.py` com `21 passed in 22.70s`; `test_strip_run.py` com `22 passed in 3.88s`
- suite impactada: `70 passed in 24.05s`; reexecucao final `70 passed in 32.86s`

Rodada real:

- artefato: `debug/performance_gates/macro_ocr_real_pipeline_20260508/decision.md`
- elapsed: `276.4666s`
- peak RSS: `7980.797 MB`
- peak VRAM: `7745 MB`
- exit code: `0`
- performance gate: `PASS`, mas apenas confirma gargalo visual remanescente (`OCR+inpaint=62.80%` dos estagios medidos), nao aprova aceleracao
- output compare: `FAIL`, porque o candidato tem `131` textos/regioes contra `114` do baseline
- project import: `PASS`, com `27` paginas, `131` text layers, `0` imagens ausentes e `0` bboxes invalidas
- visual review todas as paginas: prancha gerada em `debug/performance_gates/macro_ocr_real_pipeline_20260508/visual_review_all_pages.html`
- pagina afetada estruturalmente: pagina `27`, com `+17` textos extras de creditos/scanlation
- bandas com OCR precomputado/runtime pulado: `113/154`
- janelas macro: `94`
- OCR por estagios: `10.1937s` contra `48.228s` do baseline controlado
- traducao por estagios: `44.7333s` contra `0.6683s`
- inpaint por estagios: `78.8274s` contra `49.7094s`
- blocos restantes para inpaint: `75` contra `45`

Decisao:

- resultado: `reprovado como aceleracao e reprovado estruturalmente`
- motivo: a primeira rodada real ficou muito mais lenta que o baseline controlado de `130.4294s`, subiu memoria/VRAM e adicionou camadas de creditos/scanlation na pagina final
- recomendacao: manter `TRADUZAI_MACRO_OCR=0`; se for retomado, primeiro reutilizar filtros de scanlation/cover editorial no precompute e contabilizar tempo de precompute no `strip_perf_summary`

## Macro OCR Real: Filtros De Precompute E Telemetria

Arquivos modificados:

- `pipeline/strip/run.py`
- `pipeline/tests/test_strip_run.py`

Comportamento implementado:

- o precompute Macro OCR consulta os mesmos filtros de `scanlation_credit` e `cover_editorial` usados pelo OCR sequencial antes de chamar `recognize_macro_ocr_windows`
- paginas filtradas deixam de receber OCR precomputado e caem no caminho sequencial antigo
- `strip_perf_summary.macro_ocr_precompute` registra tempo, paginas avaliadas, paginas puladas, motivos, bandas precomputadas, janelas e blocos Macro OCR
- `strip_perf_summary.durations_sec.macro_ocr_precompute` passa a contabilizar o custo do precompute no resumo por estagio

Evidencia de TDD/testes:

- red scanlation: `1 failed`, porque `recognize_macro_ocr_windows` ainda era chamado quando o filtro de scanlation retornava `True`
- green scanlation: `1 passed in 2.83s`
- red cover editorial: `1 failed`, porque o filtro de capa/editorial ainda nao era consultado
- green filtros: `2 passed in 2.30s`
- red telemetria: `1 failed`, porque `strip_perf_summary` nao tinha `macro_ocr_precompute`
- green telemetria: `1 passed in 2.64s`
- arquivo completo `test_strip_run.py`: `25 passed in 2.37s`
- suite impactada: `101 passed in 19.19s`

Rodada real:

- artefato: `debug/performance_gates/macro_ocr_real_precompute_filters_telemetry_20260508/decision.md`
- elapsed: `153.1377s`
- baseline controlado: `130.4294s`
- peak RSS: `8881.914 MB`
- peak VRAM aproximada: `2601 MB`
- output compare: `PASS`, com `114` textos/regioes contra `114`
- project import: `PASS`, com `27` paginas, `114` text layers e `0` imagens ausentes
- visual review todas as paginas: `PASS`, prancha gerada em `debug/performance_gates/macro_ocr_real_precompute_filters_telemetry_20260508/visual_review_all_pages.html`
- performance gate: `PASS`, mas apenas confirma gargalo visual remanescente, nao aprova aceleracao
- `macro_ocr_precompute`: `41.4386s`
- OCR por bandas: `23.2621s`
- inpaint por bandas: `55.9444s`
- paginas puladas no precompute: `5`, todas por `scanlation_credit`
- bandas com OCR precomputado/runtime pulado: `70/154`
- blocos restantes para inpaint: `57`

Decisao:

- resultado: `aprovado estruturalmente, reprovado como aceleracao/default`
- motivo: os filtros removem a regressao de creditos/scanlation e o output volta a bater com o baseline, mas o precompute custa `41.4386s`, o tempo total fica `22.7083s` acima do baseline e o pico de RAM sobe para `8881.914 MB`
- recomendacao: manter `TRADUZAI_MACRO_OCR=0`; a proxima tentativa deve reduzir/paralelizar com seguranca o precompute ou mover o foco para scheduler executor experimental

## Macro OCR Real: Thresholds De Precompute

Arquivos modificados:

- `pipeline/strip/run.py`
- `pipeline/tests/test_strip_run.py`

Comportamento implementado:

- env experimental `TRADUZAI_MACRO_OCR_PRECOMPUTE_MIN_BLOCKS`
- default `1`, preservando o comportamento anterior
- paginas com menos blocos que o threshold pulam o precompute e caem no OCR sequencial
- telemetria registra `min_blocks` e `skip_reasons.below_min_blocks`

Evidencia de TDD/testes:

- red threshold: `1 failed`, porque `recognize_macro_ocr_windows` ainda era chamado com apenas 1 bloco e threshold `2`
- green threshold: `1 passed in 2.52s`
- arquivo completo `test_strip_run.py`: `26 passed in 2.64s`
- suite impactada: `102 passed in 19.11s`

Rodadas reais:

- artefato: `debug/performance_gates/macro_ocr_precompute_thresholds_20260508/decision.md`
- baseline controlado: `130.4294s`, RSS `6341.109 MB`, VRAM aproximada `3017 MB`
- sem threshold seletivo: `153.1377s`, precompute `41.4386s`, output/import/visual `PASS`
- `min_blocks=3`: `140.1823s`, RSS `7460.695 MB`, VRAM aproximada `2595 MB`, precompute `34.1823s`, output/import/visual `PASS`
- `min_blocks=5`: `142.5177s`, RSS `6698.273 MB`, VRAM aproximada `2596 MB`, precompute `34.5492s`, output/import/visual `PASS`

Decisao:

- resultado: `reprovado como aceleracao`
- motivo: `min_blocks=3` foi o melhor perfil, mas ainda ficou `9.7529s` mais lento que o baseline e aumentou RAM; `min_blocks=5` reduziu RAM, mas piorou tempo
- recomendacao: manter `TRADUZAI_MACRO_OCR=0` e nao promover threshold seletivo para Performance

## Atualizacao Final: Fast Fill Default

Artefato consolidado:

- `debug/performance_gates/fast_fill_default_20260509/decision.md`

Mudancas finais:

- `TRADUZAI_STRIP_FAST_WHITE_NARRATION` agora fica ligado por default.
- `TRADUZAI_STRIP_FAST_METADATA_FILL` foi implementado e fica ligado por default.
- `TRADUZAI_SMART_SKIP` continua desligado por default.
- `TRADUZAI_STRIP_SCHEDULER_EXECUTOR=overlap` continua desligado por default.

Evidencia adicionada:

- red defaults: `2 failed`, porque narracao branca e metadata fill ainda nao eram default
- green defaults: `2 passed in 3.87s`
- suite impactada final: `87 passed in 27.27s`
- default novo: resource profile `PASS`, elapsed `139.7025s`, peak RSS `7328.574 MB`, peak VRAM `2894 MB`
- performance gate: `PASS`, OCR+inpaint `93.02%`
- output compare: `PASS`, `27/27` paginas, `114/114` textos, `114/114` regioes traduzidas
- project import: `PASS`, `27` paginas, `114` text layers, `162` imagens verificadas
- visual review todas as paginas: `PASS`, maior `pixel_diff_rate=0.017514`, `different_text_rate=0.0`, `missing_text_rate=0.0`

Comparacao:

- baseline pareado antigo: `181.9548s`
- default novo: `139.7025s`
- ganho pareado: `42.2523s`
- melhor baseline historico controlado: `130.4294s`
- meta do plano: `<=113.65s`

Veredito adicional:

Rodada encerrada como ganho parcial validado. O default ficou mais rapido e passou estrutura/importacao/visual, mas a meta numerica de `<=113.65s` nao foi atingida. O gargalo restante continua sendo OCR/inpaint: com o default novo, esses estagios somam `99.6763s` e respondem por `93.02%` do tempo medido por estagios.

## Decisao De Goal

Meta encerrada por decisao do usuario com as etapas implementadas testadas, gateadas e avaliadas. Nao continuar repetindo Macro OCR real, Smart Skip real, bundle antigo ou scheduler overlap sem uma mudanca nova de arquitetura/hardware, porque todos ja foram medidos nesta rodada.

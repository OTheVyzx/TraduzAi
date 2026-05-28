# Plano final de correcao e melhoria - pipeline automatico visual

Data: 2026-05-19
Escopo: pipeline automatico EN -> PT-BR com foco em resultado visual real, rastreabilidade E2E e export gate.

## Entradas analisadas

- `C:/Users/PICHAU/Downloads/latest_visual_pipeline_remediation_plan.md`
- `N:/TraduzAI/docs/plans/2026-05-19-automatic-pipeline-visual-fix.md`
- `N:/TraduzAI/DEBUGM/runs/2026-05-19_chapter1_e2e_debug_codex_2026-05-18_215312.zip`
- Plano consolidado gerado: `N:/TraduzAI/docs/plans/2026-05-19-visual-pipeline-merged-execution-plan.md`

## Resultado dos 4 ciclos executados

Run final de referencia:

- `N:/TraduzAI/DEBUGM/runs/2026-05-19_cycle4_gatefix_visual_pipeline_2026-05-19_010552`
- Variante: `C_fast_fill`
- Pipeline exit code: `0`
- Analyzer exit code: `0`
- Export gate final: `BLOCK`
- Issues finais: `16`
- Criticas finais: `4`
- Reviews finais: `12`

Comparacao principal contra o ZIP inicial na variante `C_fast_fill`:

- Antes: `31` issues QA, `13` criticas visuais, `render_plan_final` divergente do `project.json`.
- Depois: `16` issues QA, `4` criticas, `render_plan_final` sincronizado a partir do `project.json` final.

## Atualizacao da continuacao - Fase 1 executada

Run de continuacao:

- `N:/TraduzAI/DEBUGM/runs/2026-05-19_phase1_mask_p0_cfast_2026-05-19_115342`
- Variante: `C_fast_fill`
- Pipeline exit code: `0`
- Analyzer exit code: `0`
- Export gate: `PASS`
- Issues finais: `12`
- Criticas finais: `0`
- Reviews finais: `12`

Correcao aplicada:

- `pipeline/debug_tools/masks.py` agora exige pixels absolutos fora do balao **e** proporcao minima (`outside_balloon_ratio >= 0.18`) antes de emitir `mask_outside_balloon_critical`.
- Vazamento pequeno causado por dilatacao normal da mascara continua rastreavel como `mask_outside_balloon`, mas nao bloqueia exportacao como P0.
- Teste regressivo adicionado em `pipeline/tests/test_mask_chain_debug.py` cobrindo o caso de muitos pixels absolutos fora do balao, mas baixa proporcao.

Validacoes executadas:

- `N:/TraduzAI/pipeline/venv/Scripts/python.exe -m pytest tests/test_mask_chain_debug.py tests/test_qa_flag_propagation_v2.py -q` -> `11 passed`.
- `N:/TraduzAI/pipeline/venv/Scripts/python.exe -m pytest tests/test_export_gate.py tests/test_mask_builder.py -q` -> `17 passed`.
- `tools/analyze_e2e_debug.py <run_root> --write-report --strict-debug-audit` -> exit code `0`.
- `git diff --check` no escopo tocado -> sem erros.

Artefatos de evidencia:

- `N:/TraduzAI/DEBUGM/runs/2026-05-19_phase1_mask_p0_cfast_2026-05-19_115342/debug_report.md`
- `N:/TraduzAI/DEBUGM/runs/2026-05-19_phase1_mask_p0_cfast_2026-05-19_115342/C_fast_fill/debug/e2e/11_qa_export_gate/export_gate.json`
- `N:/TraduzAI/DEBUGM/runs/2026-05-19_phase1_mask_p0_cfast_2026-05-19_115342/C_fast_fill/debug/e2e/12_contact_sheets/problem_bands.jpg`
- `N:/TraduzAI/DEBUGM/runs/2026-05-19_phase1_mask_p0_cfast_2026-05-19_115342/C_fast_fill/debug/e2e/12_contact_sheets/translated_comparison.jpg`

## Atualizacao da continuacao - Fase 2 executada

Run final da Fase 2:

- `N:/TraduzAI/DEBUGM/runs/2026-05-19_phase2_geometry_cfast_final_2026-05-19_123822`
- Variante: `C_fast_fill`
- Pipeline: evento `complete`
- Analyzer strict: exit code `0`
- Debug report: `PASS`
- Issues no debug report: `8`
- Export blockers: `0`
- Criticas finais: `0`
- Warnings finais: `8` (`bbox_overreach=3`, `mask_density_high=5`)
- `mask_outside_balloon`: `0` no export gate
- `balloon_bbox_missing`: `0` no audit strict

Correcoes aplicadas:

- `pipeline/debug_tools/masks.py` agora usa o `band_id` canonico vindo de `trace_id`/texto antes de cair no indice do `ocr_page`, eliminando a pasta de mascara com `band+1`.
- `pipeline/debug_tools/masks.py` alinha `mask_outside_balloon` ao limiar proporcional de review (`outside_balloon_ratio >= 0.08`), removendo vazamentos minimos de dilatacao do export gate.
- `pipeline/typesetter/renderer.py` so audita `balloon_bbox_missing` depois de `build_render_blocks`, ou seja, apenas para texto realmente renderizavel.
- `pipeline/strip/process_bands.py` preenche `balloon_bbox` defensivo antes dos early returns de `skip_processing` e `unchanged_translation`.
- `pipeline/main.py` deixou de propagar `bbox_overreach` a partir de `mask_decision` band-level; esse flag agora deve vir do `render_plan`/texto individual.
- `pipeline/inpainter/mask_builder.py` anexa metricas de `bbox_overreach` (`bbox`, `text_geometry_bbox`, ratio e se a bbox ampla dirigiu a mascara) para os warnings restantes.

Validacoes executadas:

- `N:/TraduzAI/pipeline/venv/Scripts/python.exe -m pytest tests/test_mask_chain_debug.py tests/test_mask_builder.py tests/test_typeset_render_plan_debug.py tests/test_strip_process_bands.py::ProcessBandTests::test_process_band_skips_repaint_when_all_texts_are_skip_processing tests/test_strip_process_bands.py::ProcessBandTests::test_process_band_skips_repaint_when_translation_marks_all_texts_skip_processing tests/test_strip_process_bands.py::ProcessBandTests::test_process_band_skips_repaint_when_all_translations_are_unchanged tests/test_strip_balloon_bbox_propagation.py::RenderBandImageGuardTests -q` -> `25 passed`.
- `N:/TraduzAI/pipeline/venv/Scripts/python.exe -m pytest tests/test_qa_flag_propagation_v2.py tests/test_mask_chain_debug.py tests/test_mask_builder.py tests/test_typeset_render_plan_debug.py tests/test_export_gate.py tests/test_export_gate_debug_consistency.py -q` -> `35 passed`.
- `tools/analyze_e2e_debug.py N:/TraduzAI/DEBUGM/runs/2026-05-19_phase2_geometry_cfast_final_2026-05-19_123822/C_fast_fill --write-report --strict-debug-audit` -> exit code `0`.
- `git diff --check` no escopo tocado -> sem erros.

Warnings restantes justificados:

- `bbox_overreach` restante: `ocr_001@page_002_band_002`, `ocr_005@page_002_band_018`, `ocr_002@page_004_band_066`. Todos agora aparecem com bbox ampla, bbox de geometria real e ratio em `render_plan_final.jsonl`.
- `mask_density_high` restante: 5 textos em 4 regioes. Eles indicam mascara/grupo denso e devem ser tratados na Fase 2.1/Fase 3 com politica de texto nao-balao, top narration e grupos conectados, nao como blocker do export.

## Atualizacao da continuacao - Fase 2.1 executada

Run final da Fase 2.1:

- `N:/TraduzAI/DEBUGM/runs/2026-05-19_phase21_warning_cleanup_cfast_2026-05-19_132335`
- Variante: `C_fast_fill`
- Pipeline: evento `complete`
- Analyzer strict: exit code `0`
- Debug report: `PASS`
- Issues no debug report: `2`
- Export blockers: `0`
- Criticas finais: `0`
- Warnings finais: `2` (`mask_density_high=2`)
- `bbox_overreach`: `0` no export gate
- `mask_outside_balloon`: `0` no export gate
- `balloon_bbox_missing`: `0` no audit strict; o run atual nao emitiu `balloon_bbox_missing_audit.jsonl` porque nao houve missing auditavel

Analise com agentes:

- Agente visual: confirmou que os 3 `bbox_overreach` eram ruido de diagnostico, porque `render_bbox` e `safe_text_box` estavam contidos e a bbox ampla nao dirigia a mascara.
- Agente de mascara: separou `mask_density_high` em falso positivo limpo (`top_narration`, painel escuro e densidade borderline) versus bug visual real em grupo conectado.
- Agente de geometria: recomendou manter metricas de overreach no `render_plan_final.jsonl`, mas so promover flag quando `broad_bbox_drives_mask=true`.

Correcoes aplicadas:

- `pipeline/inpainter/mask_builder.py` continua gravando `qa_metrics.bbox_overreach`, mas so emite `bbox_overreach`/`bbox_overreach_critical` se a bbox ampla realmente dirige a mascara.
- `pipeline/main.py` filtra `bbox_overreach` de `render_plan` quando `broad_bbox_drives_mask=false`, mantendo a metrica para rastreabilidade.
- `pipeline/debug_tools/masks.py` ganhou gate seletivo de `mask_density_high`: preserva caso forte com grupo conectado/fonte ampla e remove falso positivo limpo em line polygons confiaveis.
- Testes regressivos foram adicionados para narracao limpa, densidade borderline de dialogo, grupo conectado denso e render plan com overreach apenas metrico.

Validacoes executadas:

- `N:/TraduzAI/pipeline/venv/Scripts/python.exe -m pytest tests/test_mask_chain_debug.py tests/test_mask_builder.py tests/test_qa_flag_propagation_v2.py -q` -> `30 passed`.
- `N:/TraduzAI/pipeline/venv/Scripts/python.exe -m pytest tests/test_mask_chain_debug.py tests/test_mask_builder.py tests/test_qa_flag_propagation_v2.py tests/test_typeset_render_plan_debug.py tests/test_export_gate.py tests/test_export_gate_debug_consistency.py tests/test_strip_process_bands.py::ProcessBandTests::test_process_band_skips_repaint_when_all_texts_are_skip_processing tests/test_strip_process_bands.py::ProcessBandTests::test_process_band_skips_repaint_when_translation_marks_all_texts_skip_processing tests/test_strip_process_bands.py::ProcessBandTests::test_process_band_skips_repaint_when_all_translations_are_unchanged tests/test_strip_balloon_bbox_propagation.py::RenderBandImageGuardTests -q` -> `44 passed`.
- `tools/analyze_e2e_debug.py N:/TraduzAI/DEBUGM/runs/2026-05-19_phase21_warning_cleanup_cfast_2026-05-19_132335/C_fast_fill --write-report --strict-debug-audit` -> exit code `0`.
- `git diff --check` no escopo tocado -> sem erros, apenas avisos CRLF existentes em arquivos Python.

Warnings restantes justificados:

- `ocr_001@page_003_band_042`: `mask_density_high`, texto `EI, VAMOS! ESTOU MORRENDO DE FOME`.
- `ocr_003@page_003_band_042`: `mask_density_high`, texto `QUEM ESTA PAGANDO HOJE?`.
- Ambos pertencem ao mesmo grupo conectado. `mask_decision.json` mostra `mask_source=line_polygons`, `outside_balloon_pixels=0`, `mask_density_in_band=0.328357`, `source_glyph_area_ratio=1.761899`, `expanded_raw_ratio=1.0`. Ou seja: nao ha vazamento nem overreach operacional, mas a mascara ainda esta densa demais para o grupo e deve ser o proximo alvo visual.

Artefatos de evidencia da Fase 2.1:

- `N:/TraduzAI/DEBUGM/runs/2026-05-19_phase21_warning_cleanup_cfast_2026-05-19_132335/C_fast_fill/debug/e2e/13_report/debug_report.json`
- `N:/TraduzAI/DEBUGM/runs/2026-05-19_phase21_warning_cleanup_cfast_2026-05-19_132335/C_fast_fill/debug/e2e/11_qa_export_gate/qa_issues.jsonl`
- `N:/TraduzAI/DEBUGM/runs/2026-05-19_phase21_warning_cleanup_cfast_2026-05-19_132335/C_fast_fill/debug/e2e/06_mask_segmentation/page_003_band_042/mask_decision.json`
- `N:/TraduzAI/DEBUGM/runs/2026-05-19_phase21_warning_cleanup_cfast_2026-05-19_132335/C_fast_fill/debug/e2e/06_mask_segmentation/page_003_band_042/10_mask_overlay.jpg`
- `N:/TraduzAI/DEBUGM/runs/2026-05-19_phase21_warning_cleanup_cfast_2026-05-19_132335/C_fast_fill/debug/e2e/12_contact_sheets/problem_bands.jpg`
- `N:/TraduzAI/DEBUGM/runs/2026-05-19_phase21_warning_cleanup_cfast_2026-05-19_132335/C_fast_fill/debug/e2e/12_contact_sheets/translated_comparison.jpg`

## Atualizacao da continuacao - Fase 2.2 guard de rastreabilidade

Motivo:

- Revisao final com agente encontrou um risco de contrato: se `mask_decision.json` emitisse uma flag critica, mas a identidade nao casasse com nenhuma layer final, a flag podia ficar apenas em `qa_flag_propagation_audit.json` e nao bloquear o export gate.

Correcao aplicada:

- `pipeline/main.py` agora guarda `qa.flag_propagation_audit` dentro do `project.json`.
- `pipeline/main.py` aumenta o `qa.summary` com `qa_flag_not_propagated` quando houver flags de debug nao propagadas.
- `pipeline/qa/export_gate.py` cria um blocker `p0_traceability_blocker` quando `qa.flag_propagation_audit.missing_in_project` nao esta vazio.
- `pipeline/debug_tools/masks.py` passou a calcular `outside_balloon_pixels` sobre a mascara efetiva (`final_mask`) quando existe `final_mask`/`protection_mask`, evitando warning em pixels que a protecao final nao aplicaria.

Validacoes executadas:

- `N:/TraduzAI/pipeline/venv/Scripts/python.exe -m pytest tests/test_qa_flag_propagation_v2.py tests/test_export_gate.py tests/test_export_gate_debug_consistency.py tests/test_mask_chain_debug.py -q` -> `27 passed`.
- `N:/TraduzAI/pipeline/venv/Scripts/python.exe -m pytest tests/test_mask_chain_debug.py tests/test_mask_builder.py tests/test_qa_flag_propagation_v2.py tests/test_typeset_render_plan_debug.py tests/test_export_gate.py tests/test_export_gate_debug_consistency.py tests/test_strip_process_bands.py::ProcessBandTests::test_process_band_skips_repaint_when_all_texts_are_skip_processing tests/test_strip_process_bands.py::ProcessBandTests::test_process_band_skips_repaint_when_translation_marks_all_texts_skip_processing tests/test_strip_process_bands.py::ProcessBandTests::test_process_band_skips_repaint_when_all_translations_are_unchanged tests/test_strip_balloon_bbox_propagation.py::RenderBandImageGuardTests -q` -> `47 passed`.
- `tools/analyze_e2e_debug.py N:/TraduzAI/DEBUGM/runs/2026-05-19_phase21_warning_cleanup_cfast_2026-05-19_132335/C_fast_fill --write-report --strict-debug-audit` -> exit code `0`.

## Atualizacao da continuacao - Fase 2.3 typeset/inpaint visual fechado

Run final validado:

- `N:/TraduzAI/DEBUGM/runs/2026-05-19_phase33_trace_candidates_cfast_2026-05-19_162253`
- Variante: `C_fast_fill`
- Pipeline exit code: `0`
- Analyzer strict: exit code `0`
- Export gate: `PASS`
- Issues no debug report: `0`
- Criticas finais: `0`
- Reviews finais: `0`
- `render_plan_project_mismatch_count`: `0`
- `render_plan_project_field_mismatch_count`: `0`
- `render_on_art_count`: `0`
- `bbox_overreach_count`: `0`
- `visual_blocker_count`: `0`
- `debug_errors_count`: `0`
- `mixed_coordinate_space_count`: `0`

Correcoes aplicadas nesta etapa:

- O split visual do balao `page_003_band_042` deixou de usar a linha fina original como caixa final quando a traducao e longa. O bloco `ocr_001@page_003_band_042` agora expande a capacidade de `[343,567,538,591]` para `[304,545,554,607]`, usa fonte `25`, quebra em `ESTOU MORRENDO` / `DE FOME`, e fica sem `qa_flags`.
- O cleanup final de pagina agora rejeita rerender que reintroduz pixels escuros dentro da geometria de texto, usando todos os candidatos de cleanup, nao apenas os textos que passaram pelo filtro `_white_cleanup_texts`.
- `render_plan_candidates.jsonl`, `render_plan_skipped.jsonl` e `balloon_bbox_missing_audit.jsonl` passaram a preservar `coordinate_space` real (`band` ou `page`) em vez de marcar tudo como `band`. No phase33: `render_plan_candidates.jsonl` tem `546` entradas `page` e `370` `band`; `render_plan_skipped.jsonl` tem `226` entradas `page` e `59` `band`.

Validacoes executadas:

- `N:/TraduzAI/pipeline/venv/Scripts/python.exe -m pytest tests/test_typesetting_renderer.py::TypesettingRendererTests::test_render_plan_candidates_and_skipped_preserve_page_coordinate_space tests/test_typesetting_renderer.py::TypesettingRendererTests::test_render_plan_records_candidates_and_skipped_with_trace_metadata tests/test_typesetting_renderer.py::TypesettingRendererTests::test_visual_lobe_split_long_translation_expands_tiny_line_capacity -q` -> `3 passed`.
- `N:/TraduzAI/pipeline/venv/Scripts/python.exe -m pytest tests/test_typesetting_renderer.py tests/test_typeset_render_plan_debug.py tests/test_render_plan_trace_integrity.py tests/test_e2e_debug_report_consistency.py tests/test_analyze_e2e_debug.py tests/test_strip_run.py::RunChapterSmokeTests::test_page_final_cleanup_rejects_text_residual_regression tests/test_strip_run.py::RunChapterSmokeTests::test_page_final_near_text_cleanup_can_be_enabled tests/test_mask_chain_debug.py tests/test_mask_builder.py -q` -> `105 passed`.
- `tools/analyze_e2e_debug.py N:/TraduzAI/DEBUGM/runs/2026-05-19_phase33_trace_candidates_cfast_2026-05-19_162253/C_fast_fill --write-report --strict-debug-audit` -> exit code `0`.

Evidencia visual:

- `N:/TraduzAI/DEBUGM/runs/2026-05-19_phase33_trace_candidates_cfast_2026-05-19_162253/C_fast_fill/page003_band042_bottom_bubble_zoom_phase33_translated.jpg`
- `N:/TraduzAI/DEBUGM/runs/2026-05-19_phase33_trace_candidates_cfast_2026-05-19_162253/C_fast_fill/debug/e2e/12_contact_sheets/problem_bands.jpg`
- `N:/TraduzAI/DEBUGM/runs/2026-05-19_phase33_trace_candidates_cfast_2026-05-19_162253/C_fast_fill/debug/e2e/12_contact_sheets/translated_comparison.jpg`

Leitura visual do crop final: o balao inferior da pagina 3 esta em PT-BR, sem `I'M STARVING` residual, sem texto espremido e sem render sobre arte.

## Melhorias ja aplicadas

1. Rastreabilidade canonica
   - `render_plan_final.jsonl` passou a ser reescrito a partir do `project.json` final no debug/export gate.
   - O debug registra `render_plan_final_sync.json` para provar a origem e a contagem de linhas.
   - Teste regressivo cobre o caso em que o render plan antigo divergia do projeto final.

2. Politica de OCR/router
   - Fala real em balao branco na abertura deixou de ser descartada como `noise`.
   - OCR suspeito em arte/rosto com palavra colada e baixa confianca passou a ser descartado.
   - O caso de falso texto sobre o rosto/personagem foi removido do output visual.

3. Geometria e QA de mascara
   - `bbox_overreach_critical` foi rebaixado para warning quando existe geometria de linhas confiavel e a bbox ampla nao dirige a mascara.
   - `bbox_overreach` deixou de virar warning de export quando a bbox ampla e apenas envelope diagnostico e nao dirige mascara/render.
   - `mask_outside_balloon_critical` passou a exigir evidencia mais forte quando ha line polygons confiaveis.
   - `mask_density_high` agora diferencia mascara limpa/densa por tipo de texto de grupo conectado com fonte ampla.
   - Flags de mascara sincronizadas foram separadas para evitar propagacao cega em parte do fluxo de debug.

4. Inpaint em paineis escuros
   - Foi adicionado preenchimento local para texto claro antigo em painel escuro.
   - `text_residual_after_inpaint` deixou de bloquear quando o preenchimento escuro remove o fantasma visual.
   - O painel da pagina 6 agora fica visualmente limpo no recorte de referencia.

5. Falso positivo de typesetting
   - Texto curto preservado, como `HM`, nao gera mais `render_on_art_suspected` quando original e traduzido normalizados sao equivalentes.

6. Typeset visual de balao conectado
   - Texto longo em split visual de balao branco agora pode expandir para a capacidade local da bolha quando o anchor original e uma linha fina.
   - O debug registra a razao `visual_lobe_long_text`, a caixa de capacidade e a quebra de linhas usada.

7. Cleanup final sem regressao de residuo
   - O rerender de cleanup de pagina e rejeitado quando aumenta pixels escuros dentro da geometria de texto.
   - Isso evita reintroduzir texto original depois de um inpaint local que ja estava limpo.

## Evidencia visual final

Artefatos principais do ciclo 4, da Fase 2.1 e do phase33:

- `N:/TraduzAI/DEBUGM/runs/2026-05-19_cycle4_gatefix_visual_pipeline_2026-05-19_010552/C_fast_fill/page006_title_area.jpg`
- `N:/TraduzAI/DEBUGM/runs/2026-05-19_cycle4_gatefix_visual_pipeline_2026-05-19_010552/C_fast_fill/page003_balloon_area.jpg`
- `N:/TraduzAI/DEBUGM/runs/2026-05-19_cycle4_gatefix_visual_pipeline_2026-05-19_010552/C_fast_fill/page005_hm_area.jpg`
- `N:/TraduzAI/DEBUGM/runs/2026-05-19_cycle4_gatefix_visual_pipeline_2026-05-19_010552/C_fast_fill/debug/e2e/12_contact_sheets/problem_bands.jpg`
- `N:/TraduzAI/DEBUGM/runs/2026-05-19_cycle4_gatefix_visual_pipeline_2026-05-19_010552/C_fast_fill/debug/e2e/12_contact_sheets/translated_comparison.jpg`
- `N:/TraduzAI/DEBUGM/runs/2026-05-19_phase21_warning_cleanup_cfast_2026-05-19_132335/C_fast_fill/debug/e2e/12_contact_sheets/problem_bands.jpg`
- `N:/TraduzAI/DEBUGM/runs/2026-05-19_phase21_warning_cleanup_cfast_2026-05-19_132335/C_fast_fill/debug/e2e/12_contact_sheets/translated_comparison.jpg`
- `N:/TraduzAI/DEBUGM/runs/2026-05-19_phase33_trace_candidates_cfast_2026-05-19_162253/C_fast_fill/page003_band042_bottom_bubble_zoom_phase33_translated.jpg`
- `N:/TraduzAI/DEBUGM/runs/2026-05-19_phase33_trace_candidates_cfast_2026-05-19_162253/C_fast_fill/debug/e2e/12_contact_sheets/problem_bands.jpg`
- `N:/TraduzAI/DEBUGM/runs/2026-05-19_phase33_trace_candidates_cfast_2026-05-19_162253/C_fast_fill/debug/e2e/12_contact_sheets/translated_comparison.jpg`

Leitura visual:

- O painel escuro da pagina 6 esta limpo o suficiente para nao justificar `text_residual_after_inpaint`.
- O balao da pagina 3 voltou a preservar fala real, sem renderizar texto em cima do rosto.
- O balao inferior da pagina 3 nao mostra mais ingles residual nem texto espremido.
- O `HM` curto permanece como SFX preservado, sem falso blocker de texto sobre arte.

## Bloqueadores restantes

### P0 - Autoridade final das flags de mascara

Status: resolvido na continuacao da Fase 1. O gate `C_fast_fill` agora esta `PASS` com `0` criticas. O bloco abaixo fica como historico da causa que foi fechada.

O gate bloqueava por 4 `mask_outside_balloon_critical`:

- `ocr_002@page_002_band_008`: `POR FAVOR, PELO BEM DA CRIANCA.`
- `ocr_001@page_006_band_113`: `A sincronizacao foi concluida.`
- `ocr_001@page_006_band_115`: `O anfitriao recebeu o titulo,`
- `ocr_001@page_006_band_117`: `Classe de nivel atual: espirito flutuante`

Problema confirmado: o debug de mascara tratava `outside_balloon_pixels > 50` como critico absoluto. Nos casos reais, o vazamento era proporcionalmente pequeno e vinha da dilatacao normal da mascara expandida.

Correcao executada:

1. Alinhar o debug de mascara ao criterio proporcional usado pelo pipeline real.
2. Manter `mask_outside_balloon` como warning quando ha vazamento pequeno.
3. Bloquear como `mask_outside_balloon_critical` apenas quando houver pixels absolutos e proporcao fora do balao suficientes.
4. Cobrir a diferenca com teste regressivo.

### P1 - Invariante de `balloon_bbox` e caixas seguras

Status: resolvido para a variante `C_fast_fill` no phase33. `bbox_overreach`, `mask_outside_balloon`, `balloon_bbox_missing`, `mask_density_high`, `render_on_art` e blockers visuais estao em `0` no export/debug atual. O antigo grupo conectado `page_003_band_042` foi fechado por combinacao de mascara por geometria, expansao controlada de typeset e guard contra regressao no cleanup final.

Correcao recomendada:

1. Antes do typesetting, validar que todo texto renderizavel tem `balloon_bbox`, `safe_text_box`, `source_bbox` e `render_bbox` em coordenadas canonicas.
2. Quando o detector nao tiver balao confiavel, marcar explicitamente `non_balloon_text` ou `needs_human_review`, em vez de deixar o renderer inferir.
3. Para baloes conectados, usar subregioes por linha/poligono como limite de mascara, nao a bbox ampla do grupo inteiro.

### P2 - OCR fragmentado e texto colado

O patch atual removeu o caso de falso OCR sobre arte, mas ainda ha texto com fragmentos ruins ou traducao literal estranha em alguns baloes.

Correcao recomendada:

1. Criar uma etapa `ocr_text_repair` antes da traducao com regras rastreaveis:
   - juntar linhas do mesmo balao por proximidade e ordem de leitura;
   - descartar fragmentos curtos sem contexto;
   - manter fala real quando houver pontuacao/frase e geometria de balao.
2. Emitir `ocr_repair_decisions.jsonl` com `trace_id`, texto original, texto reparado, motivo e confianca.
3. Adicionar crops antes/depois para os casos de fala real em balao branco e falso texto sobre arte.

### P3 - Politica visual por tipo de texto

O pipeline ainda mistura casos de fala, sistema, titulo, SFX e documento em heuristicas parecidas.

Correcao recomendada:

1. Formalizar tipos: `dialogue`, `narration`, `sign`, `document`, `system`, `sfx`, `cover`, `watermark`, `noise`.
2. Dar a cada tipo uma politica propria de OCR, traducao, inpaint, style e gate.
3. Para `system/title` em painel escuro, usar style e inpaint diferentes de balao branco.

### P4 - Matriz completa A/B/C/D como gate final

Os ciclos 2 a 4 e as Fases 2.1 a 2.3 foram focados em `C_fast_fill` para iterar rapido sobre o melhor resultado visual. Antes de encerrar a correcao como pronta para todos os modos, a matriz A/B/C/D precisa rodar novamente.

Correcao recomendada:

1. Rodar A/B/C/D no mesmo capitulo usando o codigo do phase33.
2. Exigir:
   - `render_plan_project_mismatch_count == 0`
   - `render_plan_project_field_mismatch_count == 0`
   - `render_on_art_suspected == 0`
   - `text_residual_after_inpaint == 0`
   - `bbox_overreach_critical == 0`
   - `mask_outside_balloon_critical == 0`, salvo override manual com crop anexado.
3. Gerar `problem_bands.jpg` e `translated_comparison.jpg` para cada variante.

## Proximo plano de execucao

### Fase 1 - Fechar o P0 de mascara

Status: concluida.

Arquivos provaveis:

- `pipeline/main.py`
- `pipeline/inpainter/mask_builder.py`
- `pipeline/inpainter/__init__.py`
- `pipeline/qa/export_gate.py`
- testes em `pipeline/tests/test_qa_flag_propagation_v2.py` e `pipeline/tests/test_export_gate.py`

Entrega:

- Nenhum `mask_outside_balloon_critical` pode chegar ao gate sem decisao final de mascara correspondente.
- O ultimo run `C_fast_fill` deve cair de `4` criticas para `0` criticas ou para warnings justificados.

### Fase 2 - Fixar contrato de geometria

Status: executada na continuacao e refinada ate a Fase 2.3. O contrato de rastreabilidade e propagacao foi corrigido, e o run `C_fast_fill` chegou a `0` issues no phase33. `bbox_overreach` fica como metrica quando nao dirige mascara/render; candidatos e skips de render agora preservam `coordinate_space` real para auditoria band/page.

Arquivos provaveis:

- `pipeline/layout/`
- `pipeline/typesetter/renderer.py`
- `pipeline/inpainter/mask_builder.py`

Entrega:

- Concluido: warnings de `bbox_overreach` e `mask_density_high` foram eliminados no run `C_fast_fill` de referencia.
- Pendente: repetir a matriz A/B/C/D e congelar o baseline visual final.

### Fase 3 - Reparar OCR antes da traducao

Status: proxima fase recomendada, junto com refinamento de mascara para grupo conectado.

Arquivos provaveis:

- `pipeline/vision_stack/runtime.py`
- `pipeline/ocr/`
- `pipeline/translator/`

Entrega:

- Criar `ocr_repair_decisions.jsonl`.
- Reduzir fragmentos como palavra inicial solta, texto colado e frase sem contexto.
- Preservar fala real em balao branco, mesmo em pagina de abertura.

### Fase 4 - Rodar matriz final e congelar baseline visual

Entrega:

- Rodar A/B/C/D no capitulo de referencia.
- Criar pacote de debug final com contact sheets.
- Atualizar o plano com o run vencedor e os criterios de aceite cumpridos.

## Criterio de pronto

O pipeline automatico pode ser considerado pronto para esse capitulo quando:

1. `C_fast_fill` tiver export gate `PASS` ou `BLOCK` apenas por override manual explicitamente justificado.
2. A matriz A/B/C/D passar na auditoria estrita de rastreabilidade.
3. Os contact sheets nao mostrarem:
   - ingles residual em fala traduzida;
   - portugues renderizado sobre rosto/arte;
   - texto antigo fantasma apos inpaint;
   - caixa de texto grande invadindo arte por erro de layout.
4. Cada blocker ou warning relevante tiver `trace_id`, `page_id`, `band_id`, bbox canonica e crop verificavel.

## Atualizacao - auditoria dos prints do usuario e phase39

Data: 2026-05-19

Run final desta rodada:

- `N:/TraduzAI/DEBUGM/runs/2026-05-19_phase39_user_screenshot_gate_cfast/C_fast_fill`
- Pipeline: evento `complete`
- Analyzer strict: exit code `0`
- Export gate: `PASS`
- `needs_review`: `false`
- Criticas finais: `0`
- Visual blockers: `0`
- `render_outside_count`: `0`
- `render_on_art_count`: `0`
- `bbox_overreach_count`: `0`
- `inpaint_trace_id_missing_count`: `0`
- `translation_debug_entry_count`: `80`

Problemas confirmados a partir dos prints:

- Texto em ingles preservado por skip policy (`WHAT?`, `I LIVE...`, `AJUMMA...`) era falha real de classificacao/skip.
- Texto separado em duas colunas no retangulo do "Won Bin" era falso positivo de `connected_balloon`.
- `ANCE` vinha de merge/normalizacao OCR antes da traducao.
- O painel azul/sistema tinha residuo claro de inpaint que o QA antigo nao detectava.
- O balao branco cortado na borda usava largura retangular cheia, gerando wrap largo demais para a area visivel.

Correcoes aplicadas nesta rodada:

- `pipeline/vision_stack/runtime.py`: skip policy ficou mais conservadora para nao preservar falas inglesas reais.
- `pipeline/typesetter/renderer.py`: veto de split conectado quando um unico texto cruza a costura entre subregioes; fallback de area segura para balao branco cortado na borda.
- `pipeline/ocr/text_normalizer.py`: reparos de OCR colado/truncado antes da traducao, incluindo `REAL-LIFEINSURANCE`, `TOSHOWYOUR` e prefixo `ANCE, TE YOU`.
- `pipeline/inpainter/mask_builder.py`: mascara de painel texturizado/tight anchor nao e mais recortada como se fosse balao falso.
- `pipeline/qa/inpaint_residual.py` e `pipeline/inpainter/__init__.py`: detector de residuo claro em painel nao branco, retry adaptativo de inpaint e gate de baixo residuo apos retry.

Evidencias visuais criadas:

- `N:/TraduzAI/DEBUGM/runs/2026-05-19_phase39_user_screenshot_gate_cfast/C_fast_fill/translated_pages_contact_sheet.jpg`
- `N:/TraduzAI/DEBUGM/runs/2026-05-19_phase39_user_screenshot_gate_cfast/C_fast_fill/user_print_check_won_bin_phase38_vs_phase39.jpg`
- `N:/TraduzAI/DEBUGM/runs/2026-05-19_phase39_user_screenshot_gate_cfast/C_fast_fill/user_print_check_system_title_phase38_vs_phase39.jpg`
- `N:/TraduzAI/DEBUGM/runs/2026-05-19_phase39_user_screenshot_gate_cfast/C_fast_fill/user_print_check_insurance_phase38_vs_phase39.jpg`

Validacoes executadas:

- `N:/TraduzAI/pipeline/venv/Scripts/python.exe -m pytest tests/test_typesetting_layout.py tests/test_mask_builder.py tests/test_joined_word_normalization_v2.py tests/test_normalized_text_propagates_to_translation.py tests/test_typesetting_renderer.py::TypesettingRendererTests::test_overbroad_white_balloon_keeps_translation_inside_text_anchor tests/test_typesetting_renderer.py::TypesettingRendererTests::test_edge_clipped_white_balloon_uses_visible_safe_width_for_wrap tests/test_inpaint_debug_residual.py tests/test_vision_stack_runtime.py::VisionStackRuntimeTests::test_textured_light_residual_cleanup_removes_white_ghost_text -q`
- Resultado: `105 passed, 1 skipped, 2 subtests passed`.
- `tools/analyze_e2e_debug.py <phase39>/C_fast_fill --write-report --strict-debug-audit`
- Resultado: analyzer exit code `0`, strict audit `all_passed=true`.

Melhoria recomendada para a proxima rodada:

- Implementar QA OCR residual opcional antes do typeset: recortar a regiao da mascara, comparar OCR pre-inpaint com OCR pos-inpaint, e acionar expansao/retry quando tokens fonte longos ainda forem reconhecidos. Isso deve ficar no inpaint/debug, nao no export gate, porque o gate deve ler evidencia final e nao tentar reparar tarde demais.
- Usar essa checagem apenas como segunda linha, pois OCR pode falhar em texto claro/fraco. A metrica visual de residuo deve continuar existindo.
- Repetir matriz A/B/C/D depois de estabilizar essa checagem para garantir que a correcao nao ficou especifica ao `C_fast_fill`.

## Atualizacao - rodada dos 5 prints adicionais do usuario

Data: 2026-05-19

Run final desta rodada:

- `N:/TraduzAI/DEBUGM/runs/2026-05-19_phase45_user5_final_visual_cfast/C_fast_fill`
- Pipeline: evento `complete`
- Export gate/debug report: `PASS`
- Issues finais: `0`
- Criticas finais: `0`

Problemas corrigidos a partir dos novos prints:

- Texto faltando no balao direito da cena do dinheiro: o OCR existia, mas `cover_opening` classificava texto curto em balao branco como `cover_logo_or_art_ocr`; a regra agora preserva fala/narracao curta em balao branco quando ha geometria/confiança suficiente.
- Balao duplo conectado: o layout agora aceita split conectado para regiao visualmente branca mesmo quando o classificador marcou `textured`, evitando um bloco unico no meio.
- Balao branco "POR QUE VOCE NAO VEM": o fast white fill agora protege traco escuro fora da caixa de texto e o typeset nao ancora narracao branca no bbox antigo.
- Caixa "BEM...": o detector de residuo usa gate absoluto de pixels, pegando restos pequenos mas visiveis e acionando reparo real quando necessario.
- Painel texturizado de sistema: `dark_panel_fill` deixou de atuar em `textured` generico; o gate de residuo distingue texto claro removido em fundo escuro de residuo real, evitando falso P0 sem voltar ao preenchimento solido.

Validacoes executadas:

- `N:/TraduzAI/pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_vision_stack_inpainter.py pipeline/tests/test_inpaint_debug_residual.py pipeline/tests/test_runtime_profiles.py pipeline/tests/test_vision_stack_runtime.py::VisionStackRuntimeTests::test_cover_opening_keeps_short_white_balloon_line_with_geometry pipeline/tests/test_vision_stack_runtime.py::VisionStackRuntimeTests::test_apply_white_balloon_fill_preserves_outline_outside_text_bbox pipeline/tests/test_layout_analysis.py::GeometricFallbackSubregionsTests::test_geometric_fallback_splits_visually_white_balloon_marked_textured pipeline/tests/test_typesetting_renderer.py::TypesettingRendererTests::test_plan_text_layout_does_not_lock_long_white_narration_to_low_anchor pipeline/tests/test_typesetting_layout.py -q`
- Resultado: `123 passed, 1 skipped, 2 subtests passed`.
- Recortes finais: `N:/TraduzAI/DEBUGM/runs/2026-05-19_phase45_user5_final_visual_cfast/C_fast_fill/debug/codex_focus_2026_05_19_user_5_final`

Recomendacao sobre OCR na mascara:

- Faz sentido como QA seletiva e diagnostico de mascara, mas nao como juiz unico do export gate.
- Fluxo recomendado: primeiro medir coverage geometrico entre `expanded_mask` e `text_pixel_bbox`/`line_polygons`; se coverage for baixo, se houver `text_residual_after_inpaint`, ou se o debug estiver em modo estrito, rodar OCR regional no crop da mascara.
- Pre-inpaint: se o OCR regional nao encontra tokens do OCR inicial dentro da mascara, a mascara esta subcobrindo o texto e deve crescer localmente antes do inpaint.
- Pos-inpaint: se tokens fonte longos continuam aparecendo dentro da regiao limpa, acionar retry com mascara expandida e registrar `post_inpaint_ocr_residual.json`.
- Limites: rodar so em debug/strict ou top issues, limitar regioes por pagina, ignorar `skip_processing`/SFX/watermark, e nunca bloquear export apenas por OCR curto ou de baixa confianca.

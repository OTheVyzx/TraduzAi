# Plano consolidado de execucao - pipeline visual automatico

Data: 2026-05-19
Escopo: pipeline automatico EN -> PT-BR com evidencia visual real por capitulo.

## Fontes analisadas

- `C:/Users/PICHAU/Downloads/latest_visual_pipeline_remediation_plan.md`
- `N:/TraduzAI/docs/plans/2026-05-19-automatic-pipeline-visual-fix.md`
- `N:/TraduzAI/DEBUGM/runs/2026-05-19_chapter1_e2e_debug_codex_2026-05-18_215312.zip`

## Veredito inicial

O melhor resultado visual do ZIP e a referencia atual para iteracao e `C_fast_fill`, mas ele ainda bloqueia exportacao. O debug mostrou que o problema nao e apenas qualidade de inpaint: a rastreabilidade entre `render_plan_final.jsonl`, `project.json`, QA e imagem exportada ainda diverge, entao o debug pode apontar para caixas diferentes das que a UI/export usa.

Antes de otimizar visualmente, o pipeline precisa garantir que todo blocker visual tenha o mesmo `trace_id`, `page_id`, `band_id`, `balloon_bbox`, `safe_text_box` e `render_bbox` em:

- `debug/e2e/09_typeset/render_plan_final.jsonl`
- `project.json`
- `debug/e2e/11_qa_export_gate/qa_issues.jsonl`
- `debug/e2e/11_qa_export_gate/visual_blockers.jsonl`

## Ordem consolidada

### P0 - Rastreabilidade canonica

1. Fazer `render_plan_final.jsonl` representar o `project.json` final, em coordenadas de pagina.
2. Preservar `render_plan_raw.jsonl` como evidencia do typesetter em coordenadas de banda/tira.
3. Regerar o relatorio de debug depois da sincronizacao, para que `render_plan_final_matches_project` possa passar em todos os cenarios A/B/C/D.
4. Adicionar teste de regressao com um `render_plan_final` antigo errado e um `project.json` final correto.

### P1 - Politica visual de OCR/router

1. Corrigir falas reais classificadas como `noise`, especialmente texto em balao com geometria plausivel e texto em ingles legivel.
2. Bloquear falso OCR sobre rosto/arte quando nao houver balao ou quando a area visual nao suportar texto renderizado.
3. Separar politicas de `dialogue`, `narration`, `sign`, `document`, `system`, `sfx`, `cover`, `watermark` e `noise`.

### P2 - Geometria e typesetting

1. Reduzir `bbox_overreach_critical` quando a caixa do texto invade arte/personagem.
2. Garantir que `safe_text_box` fique dentro de `balloon_bbox` ou subregiao valida.
3. Preservar baloes conectados sem colapsar tudo para uma caixa gigante.
4. Impedir texto renderizado em rosto/arte quando o render plan falhar no gate visual.

### P3 - Mascara e inpaint

1. Manter `C_fast_fill` como referencia de velocidade/visual atual.
2. Adicionar fallback automatico quando `text_residual_after_inpaint` persistir.
3. Reduzir mascara fora do balao e densidade excessiva sem voltar a deixar ingles visivel.

### P4 - Validacao E2E real

1. Rodar matriz A/B/C/D depois de cada ciclo de correcao.
2. Gerar contact sheets com original + variantes e crops dos blockers.
3. Comparar metricas: `export_gate_status`, `qa_issue_count`, `visual_blocker_count`, `render_on_art_count`, `bbox_overreach_count`, `text_residual_after_inpaint`, `render_plan_project_mismatch_count`.

## Criterios de saida

- `render_plan_project_mismatch_count == 0`.
- `render_plan_project_field_mismatch_count == 0`.
- `render_plan_trace_page_mismatch_count == 0`.
- `render_plan_trace_band_mismatch_count == 0`.
- Todo blocker visual tem link para artefato, pagina, trace e crop.
- O melhor run visual deve reduzir blockers contra `C_fast_fill` do ZIP, nao apenas passar teste sintetico.

## Estado do ciclo 1

Diagnostico do ZIP:

- `A_baseline_debug`: 76 issues QA, 27 blockers visuais, 1 mismatch render plan/projeto.
- `B_skip_inpaint`: 2 issues QA, 2 blockers visuais, 55 mismatches render plan/projeto.
- `C_fast_fill`: 31 issues QA, 13 blockers visuais, 1 mismatch render plan/projeto.
- `D_strict_export_gate`: exit code 2, 76 issues QA, 27 blockers visuais, 1 mismatch render plan/projeto.

Acao imediata do ciclo 1: corrigir a fonte canonica de `render_plan_final.jsonl` e revalidar antes de atacar as falhas visuais de OCR/router/inpaint.

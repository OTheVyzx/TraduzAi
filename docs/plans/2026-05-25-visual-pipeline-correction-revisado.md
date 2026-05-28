# Plano Sênior Revisado — Correção da Visual Pipeline

> **Data:** 2026-05-25
> **Status:** Aprovado para execução em fases (R1 → R4)
> **Substitui:** `plano_senior_visual_pipeline_correction.md` (Downloads, externo ao repo)
> **Branch sugerida:** `Troca_de_motores` (atual) ou nova `fix/visual-pipeline-truthful-output`
> **Autor sênior:** revisão e reconciliação com o estado real do código em `N:\TraduzAI`

---

> **Revisão Codex 2026-05-25:** este documento foi ajustado para execução segura no repo atual. A revisão troca afirmações absolutas por hipóteses verificáveis, evita migração prematura da árvore `translated/`, alinha flags novas com a taxonomia existente, e mantém os módulos já existentes (`term_protection.py`, `runtime_profiles.py`, `debug_tools/recorder.py`, `qa/export_gate.py`) como pontos de extensão preferenciais.

## 0. Como ler este documento

Este plano é o resultado de uma **revisão crítica do plano original**, com validação ponto-a-ponto contra o código existente. Toda afirmação `"já existe"` está apoiada em arquivo + linha. Toda afirmação `"gap real"` foi verificada por `Grep` em `pipeline/`, `src-tauri/` e `src/`.

A leitura é por releases, em ordem. **R1 é bloqueante para tudo**: o resto só faz sentido depois que a UI passar a reconhecer um run bloqueado como bloqueado.

Convenção: `[arquivo:linha]` em código existente, `(novo)` em código a criar.

---

## 1. Resumo executivo

O plano original diagnostica corretamente o problema-raiz: **quebra de contrato entre estágios**. Mas a sua proposta de solução duplica boa parte do que já está implementado:

- Propõe schema `v2` quando o atual é `v12` ([pipeline/schema/project_schema_v12.py](../../pipeline/schema/project_schema_v12.py)).
- Propõe `TextBlockV2`/`RoutedBlockV2`/`MaskPlanV2` quando os campos já existem distribuídos em `qa_flags`, `qa_metrics`, `render_bbox`, `safe_text_box`, `balloon_bbox`, `text_id`, `trace_id`, `band_id`, `page_id`, `text_instance_id`, `coordinate_space`.
- Propõe taxonomia `BLOCK/REVIEW/INFO` quando já existe `critical/high/medium/low` com 70+ flags em [pipeline/qa/translation_qa.py:8-73](../../pipeline/qa/translation_qa.py).
- Propõe **nomes de flags inéditos** (`unresolved_english_dialogue`, `safe_text_box_degenerate`, `raw_mask_pixels_zero_for_dialogue_fast_fill`) que colidem com nomes existentes (`untranslated_english`, `safe_text_box_recomputed`, `fast_fill_insufficient_coverage`).

**O gap prioritário de integração comprovado** é a integração Rust/UI: o backend já bloqueia (`export_gate.status == "BLOCK"`), mas o Rust/UI ainda tratam o run como fluxo normal de conclusão. Isso precisa vir primeiro porque impede falso sucesso e torna as próximas correções auditáveis.

Os demais gaps continuam críticos para qualidade visual, mas são **consolidações e endurecimentos** de estruturas que já existem fragmentadas: `safe_text_box`/fit, evidência de máscara, fast fill, OCR retention, name-lock, balões conectados e texto rotacionado. Eles não devem virar uma segunda arquitetura paralela.

Este plano revisado **preserva o diagnóstico** do original, **mantém os princípios de arquitetura**, e **reorganiza a execução** em torno de quatro releases que estendem o sistema atual em vez de duplicá-lo. R1 torna o output honesto; R2-R4 corrigem as causas visuais.

---

## 2. Veredito sênior por dimensão

| Dimensão do plano original | Avaliação | Comentário |
|---|---|---|
| Diagnóstico do problema | Verde | "Quebra de contrato entre estágios" é o frame correto |
| Princípios arquiteturais (contratos > heurísticas, QA é fonte de verdade, etc.) | Verde | Preservados integralmente |
| Ordem de implementação (Task -1, 0, 10+13, 2, 7, 4-5-6...) | Amarelo | Boa intuição, mas mistura "criar do zero" com "estender". Reordenado para R1 (gap real) → R2-R4 (extensões) |
| Contratos versionados (Task -1, schema v2) | Vermelho | Schema v12 já existe; criar v2 é regressão |
| Severidade BLOCK/REVIEW/INFO | Vermelho | Reusar `critical/high/medium/low` existente |
| Export gate + UI integration (Task 10+13) | Verde | Gap real, prioridade absoluta |
| Roteamento de conteúdo (Task 2) | Amarelo | `content_class` já existe; falta `route_action` consolidado |
| Safe text box e font fit (Task 7) | Amarelo | `safe_text_box` existe; falta `fit_attempts` rastreável |
| Máscara/fast-fill/residual (Tasks 4-5-6) | Amarelo | Existe parcialmente; falta objeto único `mask_evidence` |
| OCR retention (Task 1) | Amarelo | Reusar `low_ocr_confidence`; adicionar regra de retenção |
| Name-lock (Task 3) | Amarelo | Reusar `unrestored_placeholder`; adicionar pré-passe |
| Balões conectados (Task 8) | Amarelo | `connected_balloon_id`/`lobe_id` já existem; falta política de fallback |
| Texto rotacionado (Task 9) | Amarelo | Gap real; precisa de `text_angle_degrees`/`rotated_polygon` + política |
| Debug traceability (Task 11) | Verde | Já implementado via `pipeline/debug_tools/recorder.py` + estrutura `debug/e2e/00..11` |
| Matriz de capítulos (Task 12) | Verde | Lista boa; concretizar via manifest JSON |
| Feature flags de rollout | Amarelo | Plano não diz onde vivem; reusar `pipeline/runtime_profiles.py` |
| Métricas de performance | Verde | Já existe `_PipelineTiming` em main.py; falta persistir no `qa` do projeto |
| Definition of Done | Amarelo | Critérios bons; alinhar com flags reais e testes existentes |
| Estratégia de commits | Verde | Granularidade adequada |

---

## 3. Análise vs código real (15 itens, com evidência)

### Conflitos do tipo "já existe"

| # | Plano original afirma | Realidade verificada (com path) | Severidade |
|---|---|---|---|
| 1 | `schema_version: 2` deve ser criado | Schema atual é v12 — [pipeline/schema/project_schema_v12.py](../../pipeline/schema/project_schema_v12.py), [pipeline/schema/migrate_project.py](../../pipeline/schema/migrate_project.py), [pipeline/tests/test_project_schema_v12.py](../../pipeline/tests/test_project_schema_v12.py) | **BLOCK** |
| 2 | `TextBlockV2`/`RoutedBlockV2`/`MaskPlanV2`/`RenderPlanV2`/`QAIssueV2`/`ArtifactRefV2` são novos contratos | Campos já existem em `text_layers[].{qa_flags, qa_metrics, render_bbox, safe_text_box, balloon_bbox, source_bbox, text_id, trace_id, band_id, page_id, text_instance_id, coordinate_space}` — ver `_iter_project_text_layers()` em [pipeline/main.py:436](../../pipeline/main.py) e `_project_render_plan_row()` em [main.py:1053](../../pipeline/main.py) | **BLOCK** |
| 3 | `content_class` é novidade da Task 2 | Já em 20+ arquivos: [pipeline/vision_stack/runtime.py](../../pipeline/vision_stack/runtime.py), [pipeline/typesetter/renderer.py](../../pipeline/typesetter/renderer.py), [pipeline/tests/test_content_classifier.py](../../pipeline/tests/test_content_classifier.py) | **REVIEW** |
| 4 | `connected_balloon_id`/`lobe_id` precisam ser criados (Task 8) | Já existem: [pipeline/strip/process_bands.py](../../pipeline/strip/process_bands.py), [pipeline/layout/balloon_layout.py](../../pipeline/layout/balloon_layout.py), [pipeline/tests/regression/test_connected_text_cut.py](../../pipeline/tests/regression/test_connected_text_cut.py) | **REVIEW** |
| 5 | `mask_evidence_score`, `raw_mask_pixels`, `fast_fill_allowed` são novos | Parcial: [pipeline/vision_stack/cjk_mask_fusion.py](../../pipeline/vision_stack/cjk_mask_fusion.py), [pipeline/tests/test_text_mask_evidence.py](../../pipeline/tests/test_text_mask_evidence.py), [pipeline/vision_stack/oar_ocr_adapter.py](../../pipeline/vision_stack/oar_ocr_adapter.py). Falta consolidar em objeto único | **REVIEW** |
| 6 | Severidade BLOCK/REVIEW/INFO é nova | Existe `FLAG_SEVERITY` (critical/high/medium/low) com 70+ flags em [pipeline/qa/translation_qa.py:8](../../pipeline/qa/translation_qa.py) | **BLOCK** |
| 7 | Nomes de flags: `unresolved_english_dialogue`, `safe_text_box_degenerate`, `raw_mask_pixels_zero_for_dialogue_fast_fill`, `connected_lobe_untranslated` | **Nenhum** existe; nomes atuais são `untranslated_english`, `safe_text_box_recomputed`, `fast_fill_insufficient_coverage`, `balloon_bbox_collapsed_to_text`. Ver mapeamento na §5 | **BLOCK** |
| 8 | Criar `test_visual_regression_manifest.py` em pasta nova | [pipeline/tests/regression/](../../pipeline/tests/regression/) já existe com vários testes (`test_connected_text_cut.py`, `test_text_clipping_page001.py`) | **REVIEW** |
| 9 | "Test:" `test_final_geometry_contract.py`, `test_export_gate.py`, `test_export_gate_debug_consistency.py` parecem novos | **Os três já existem** em [pipeline/tests/](../../pipeline/tests/). Verbo correto é "Modify" | **INFO** |
| 10 | `export_gate=BLOCK` ainda aparece como sucesso na UI | **Backend já bloqueia** ([pipeline/qa/export_gate.py:38-58](../../pipeline/qa/export_gate.py) retorna PASS/BLOCK/OVERRIDDEN). **Gap de integração está no Rust/UI**: o evento de conclusão precisa propagar `completion_status` e a UI precisa persistir/mostrar o estado bloqueado. `Grep` em `src/` por `export_gate\|completed_blocked\|approved_output` → **0 arquivos** | **GAP REAL — CRÍTICO** |
| 11 | `translated/approved/` vs `translated/blocked_preview/` | Hoje só existe `translated/{NNN}.{ext}` único. Editor lê esse caminho diretamente. Separação é nova e exige migração | **REVIEW** |
| 12 | `PipelineFinalStatus` (run) tem 4 estados | `Project.status` em [src/lib/stores/appStore.ts:165](../../src/lib/stores/appStore.ts) é `"idle"\|"setup"\|"processing"\|"done"\|"error"` (estado **persistente**, não evento). Confunde run-event com project-state | **REVIEW** |
| 13 | "Artifact refs mínimos desde o início" é novo | Já existe via [pipeline/debug_tools/recorder.py](../../pipeline/debug_tools/recorder.py), `linked_artifacts` no `qa_flag_propagation_audit`, estrutura `debug/e2e/00..11/` | **INFO** |
| 14 | OCR-on-mask deve usar cache `sha256(...)` | [pipeline/editor_vision_cache.py](../../pipeline/editor_vision_cache.py) já existe; plano não cita | **REVIEW** |
| 15 | Feature flags `visual_pipeline_contract_v2`, `connected_balloon_v2`, `rotated_text_v2` | Sem sistema de feature flags central. Reusar `runtime_profiles.py` (ver Apêndice A) | **REVIEW** |

### Resumo numérico

- **Itens BLOCK (devem ser revertidos no plano):** 4
- **Itens REVIEW (ajustar antes de executar):** 9
- **Itens INFO (corrigir verbo "Create"→"Modify"):** 2
- **Gap de integração prioritário comprovado:** item #10, Rust/UI.
- **Gaps visuais comprovados nos runs:** caixas de typeset degeneradas, fast fill sem evidência de glifo, OCR drop/truncation, name-lock ausente, e lobe/connected-balloon inconsistente.

---

## 4. Princípios mantidos do plano original

Preservar literalmente:

1. **Contratos antes de heurísticas** — todo estágio consome e produz objetos versionados.
2. **QA é fonte de verdade, não stdout** — um evento `pipeline-complete` não significa sucesso visual.
3. **Bloquear é melhor que entregar errado** — diálogo inglês em balão, residual após inpaint, render ilegível bloqueiam export normal.
4. **Debug deve explicar cada decisão** — cada QA issue aponta para artefatos visuais e dados.
5. **Rollout precisa ser controlado** — mudanças sensíveis ativadas por flag em `runtime_profiles.py`.

---

## 5. Mapeamento autoritativo de flags

Toda flag nova **deve** entrar em `FLAG_SEVERITY` ([pipeline/qa/translation_qa.py:8](../../pipeline/qa/translation_qa.py)) com severidade declarada. Sem taxonomia paralela.

| Plano original (renomeie para) | Atual ou novo (severidade) | Observação |
|---|---|---|
| `unresolved_english_dialogue` | **Reusar `untranslated_english`** (high). Para diálogo dentro de balão, **promover a `critical`** condicional via lookup do `content_class`. | Promoção condicional vai na função `severity_for_flag()` ou via override no `evaluate_export_gate()`. |
| `safe_text_box_degenerate` | **Reusar `safe_text_box_recomputed`** (high) + nova `fit_below_minimum_legible` (critical). | Dois conceitos distintos: "tive que recomputar" vs "não cabe minimamente legível". |
| `raw_mask_pixels_zero_for_dialogue_fast_fill` | Nova `fast_fill_no_glyph_evidence` (critical). | Adicionar a `FLAG_SEVERITY`. |
| `connected_lobe_untranslated` | Reusar `untranslated_english` com escopo `lobe_id` no contexto do issue. | A export_gate já loga `text_instance_id`. |
| `render_outside_balloon` | **Já existe** (critical). | OK. |
| `mask_outside_balloon` / `mask_outside_balloon_critical` | **Já existem** (high / critical). | OK. |
| `weak_text_residual_after_inpaint` / `text_residual_after_inpaint` | **Já existem** (high / critical). | OK. |
| `fast_fill_insufficient_coverage` / `fast_fill_unverified_residual` | **Já existem** (critical). | OK. |
| `translated_layer_missing_render_bbox` | Nova `missing_render_bbox` (critical). | Só dispara para layers `route_action == translate_*` e `skip_processing == false`. |
| `TEXT_CLIPPED` / `TEXT_OVERFLOW` | **Já existem** (high). | OK. |
| `rotated_text_policy_uncertain` | Nova `rotated_text_policy_unmet` (high). | Disparada quando ângulo cai fora dos thresholds da policy. |
| `lobe_assignment_low_confidence` | Nova `lobe_assignment_low_confidence` (high). | Bloqueia render se acompanhada de tradução não-trivial. |
| `name_lock_placeholder_lost` | **Reusar `unrestored_placeholder`** (critical). | OK. |
| `ocr_truncated_or_joined` | Nova `ocr_truncated_or_joined` (high). | Não bloqueia export, mas marca `route_action: review_required`. |
| `scanlation_credit_preserved` / `logo_preserved` / `title_card_preserved` | Não viram flags; viram `route_action: preserve` + `preserve_reason`. | Métrica, não issue. |

**Total de flags novas a adicionar a `FLAG_SEVERITY`:** 6
- `fast_fill_no_glyph_evidence` → critical
- `fit_below_minimum_legible` → critical
- `missing_render_bbox` → critical
- `rotated_text_policy_unmet` → high
- `lobe_assignment_low_confidence` → high
- `ocr_truncated_or_joined` → high

---

## 6. Plano por Releases

### Visão de releases

```
R1 (semana 1)  : Truthful Output      — Rust+UI reconhecem BLOCK
R2 (semanas 2-3): Visual Hardening    — mask_evidence + fit_attempts + fast_fill gate
R3 (semana 4)  : Semantic Quality     — route_action + name-lock + OCR retention
R4 (semana 5)  : Edge Cases           — lobes + rotação + matriz completa
```

Cada release tem:
- Dependências explícitas
- Arquivos exatos (com paths que existem hoje, marcados `(novo)` quando não)
- Aceite verificável com pytest
- Critério de rollback

---

### Release 1 — Truthful Output (bloqueante)

> **Por que primeiro:** é o gap que transforma erro visual em falso sucesso. O pipeline pode gerar imagens e emitir conclusão enquanto `export_gate.status == "BLOCK"` em `qa.export_gate` do `project.json`. R2-R4 corrigem qualidade visual; R1 garante que o app e o usuário vejam o bloqueio enquanto essas correções ainda estão sendo trabalhadas.

**Pré-requisito:** nenhum.

#### R1.1 — Rust emite `export_gate` para UI

**Objetivo:** Após `pipeline-complete`, o Rust lê `project.json` do `work_dir`, extrai `qa.summary` e `qa.export_gate`, e os anexa ao evento. Se o `project.json` ainda não tiver o bloco `qa` completo, usar `qa_report.json` como fallback de leitura, sem mudar o contrato emitido para a UI.

**Arquivos:**
- [src-tauri/src/commands/pipeline.rs:524-540](../../src-tauri/src/commands/pipeline.rs) (modificar `tokio::spawn` que emite `pipeline-complete`)
- [src-tauri/src/commands/pipeline.rs:644-657](../../src-tauri/src/commands/pipeline.rs) (mesmo, para `run_pipeline_with_fast_worker`)
- `src-tauri/src/commands/pipeline.rs` (adicionar função `read_export_gate_summary(work_dir) -> ExportGateSummary`)
- `src/lib/e2e/tauriMock.ts` (atualizar mock do evento para testes/dev sem Tauri)

**Contrato emitido (novo payload de `pipeline-complete`):**

```jsonc
{
  "success": true,                    // mantém compat — false só em error técnico
  "job_id": "uuid",
  "output_path": "...",
  "completion_status": "approved",    // "approved" | "blocked" | "overridden" | "error"
  "export_gate": {
    "status": "PASS",                 // "PASS" | "BLOCK" | "OVERRIDDEN"
    "critical_issue_count": 0,
    "critical_flag_count": 0,
    "review_issue_count": 0,
    "needs_review": false
  },
  "blocking_flags": [],               // top-N flags críticas, deduplicadas
  "review_flags": []                  // top-N flags de severidade high
}
```

**Decisão de design:** **manter `success: true`** mesmo quando BLOCK. `success: false` continua reservado para erro técnico (sidecar crashou, IO falhou). A UI decide bloqueio com `completion_status`. Isso evita quebrar listeners atuais.

**Aceite:**
- Run de fixture com `--mock-critical` ([main.py:311](../../pipeline/main.py)) emite `completion_status: "blocked"` e `blocking_flags: ["visual_text_leak"]`.
- Run normal emite `completion_status: "approved"`.
- `cargo test --package traduzai pipeline::tests::pipeline_complete_payload` passa (novo teste).
- Não-regressão: testes Rust existentes seguem passando.

**Rollback:** reverter os 2 hunks no `pipeline.rs`. Frontend ignora campos novos via `serde_json::Value::get(...)` defensivo, então não quebra ao reverter.

#### R1.2 — Frontend reconhece bloqueio

**Objetivo:** Project pode estar em estado `done_blocked` ou `needs_review`. Editor mostra banner persistente; Home mostra badge.

**Arquivos:**
- [src/lib/stores/appStore.ts:165](../../src/lib/stores/appStore.ts) — estender `Project.status` para incluir `"done_blocked"` e `"needs_review"`.
- [src/lib/tauri.ts](../../src/lib/tauri.ts) — tipar evento:

  ```typescript
  export interface PipelineCompleteEvent {
    success: boolean;
    job_id: string;
    output_path: string;
    completion_status: "approved" | "blocked" | "overridden" | "error";
    export_gate: {
      status: "PASS" | "BLOCK" | "OVERRIDDEN";
      critical_issue_count: number;
      critical_flag_count: number;
      review_issue_count: number;
      needs_review: boolean;
    };
    blocking_flags: string[];
    review_flags: string[];
  }
  ```

- `src/components/PipelineBlockedBanner.tsx` (novo) — banner com lista expandível de flags + link para abrir editor na primeira issue crítica.
- [src/pages/Processing.tsx](../../src/pages/Processing.tsx) — não transformar `completion_status: "blocked"` em `status: "done"` normal; persistir `done_blocked` ou `needs_review`.
- [src/pages/Preview.tsx](../../src/pages/Preview.tsx) — render condicional do banner se `project.status === "done_blocked"`.
- [src/pages/Home.tsx](../../src/pages/Home.tsx) — badge nos cards de capítulo: "Bloqueado (N issues)" vs "Aprovado".
- [src/lib/e2e/tauriMock.ts](../../src/lib/e2e/tauriMock.ts) — mockar `completion_status` para evitar regressão em dev/test.

**Aceite:**
- Run blocked: `Processing` finaliza como bloqueado, não como aprovado.
- Run blocked: usuário vê banner em `Preview` com contagem de issues e botão "Ver detalhes".
- Run blocked: card no `Home` mostra badge vermelho "N issues críticas".
- Run aprovado: nenhum banner, badge verde "Aprovado".

**Rollback:** banner novo é componente isolado; remover import em `Preview.tsx`/`Home.tsx`. Estados extras em `Project.status` ficam inertes.

#### R1.3 — Manifestar preview bloqueado sem migrar a árvore `translated/`

**Objetivo:** Evitar falsa aprovação sem fazer uma migração grande de paths no mesmo release. Em R1, o pipeline continua gravando no layout atual, mas o `project.json` e o evento Tauri deixam explícito que a saída é `blocked_preview`, não saída aprovada.

**Decisão:** Não criar `translated/approved/` e `translated/blocked_preview/` em R1. Essa separação física é útil, mas mexe no editor, preview, imports antigos e reabertura de projetos; deve ser uma migração posterior, depois que a UI já respeitar `completion_status`.

**Arquivos:**
- [pipeline/main.py](../../pipeline/main.py) — persistir `qa.export_gate` e `output_review_state: "approved" | "blocked_preview" | "overridden"` no `project.json`.
- [pipeline/project_writer.py](../../pipeline/project_writer.py) — não mudar o path físico; apenas garantir que `arquivo_traduzido` continue apontando para o arquivo realmente escrito.
- [src-tauri/src/commands/project.rs](../../src-tauri/src/commands/project.rs) e [src-tauri/src/commands/pipeline.rs](../../src-tauri/src/commands/pipeline.rs) — ler `output_review_state` quando disponível.
- [src/pages/Preview.tsx](../../src/pages/Preview.tsx) e [src/pages/Processing.tsx](../../src/pages/Processing.tsx) — exibir que o preview está bloqueado.

**Aceite:**
- Test: run com `--mock-critical` mantém imagens no path legado, mas grava `output_review_state: "blocked_preview"` e `qa.export_gate.status: "BLOCK"`.
- Test: abrir projeto antigo sem `output_review_state` continua funcionando.
- UI não mostra botão/estado de aprovado quando `output_review_state == "blocked_preview"`.

**Migração futura opcional:** só depois da matriz R4 passar, criar uma task separada para `translated/{approved,blocked_preview}/`, com fixtures de projetos antigos e validação de editor/preview. Não bloquear R1 por isso.

**Rollback:** remover a leitura/gravação de `output_review_state`; paths físicos continuam no formato antigo.

#### R1.4 — Manifestos de regressão visual

**Objetivo:** Bloquear regressão das classes de issue conhecidas, sem versionar imagens grandes.

**Arquivos:**
- `pipeline/tests/regression/manifests/one_second_ch2.json` (novo)
- `pipeline/tests/regression/manifests/articuno_ch61.json` (novo)
- `pipeline/tests/regression/manifests/chapter_39.json` (novo)
- `pipeline/tests/regression/manifests/god_of_death_ch2.json` (novo)
- `pipeline/tests/regression/manifests/one_second_ch1.json` (novo)
- `pipeline/tests/regression/manifests/chapter_1.json` (novo)
- `pipeline/tests/regression/test_visual_regression_manifest.py` (novo)
- `docs/debug/e2e_pipeline_debug_guide.md` (modificar — listar capítulos)

**Formato do manifesto:**

```jsonc
{
  "manifest_version": 1,
  "run_id": "one_second_ch2",
  "run_path": "N:/TraduzAI/DEBUGM/runs/.../one_second_ch2",
  "schema_version": 12,
  "current_issue_classes": [
    "TEXT_CLIPPED",
    "TEXT_OVERFLOW",
    "render_outside_balloon",
    "untranslated_english"
  ],
  "target_issue_classes_after_fix": [
    "none_or_waived"
  ],
  "pages_of_interest": [3, 7, 12],
  "qa_report_sha256_at_record_time": "...",
  "sample_artifacts": [
    {
      "page": 7,
      "issue_class": "TEXT_CLIPPED",
      "crop_path": "debug/e2e/05_layout_geometry/page_007/band_012.png"
    }
  ],
  "recorded_at": "2026-05-25",
  "recorded_by": "vinicius"
}
```

**Comportamento do teste:**

1. Para cada manifesto, validar que `run_path` existe e contém `project.json` + `debug/e2e/`.
2. Carregar `qa` do `project.json`.
3. Para cada `current_issue_classes`, exigir que a classe seja observável no run gravado. Não exigir flags novas que ainda não existem, como `fit_below_minimum_legible`, antes da task que as implementa.
4. Para `pages_of_interest`, exigir que existam artefatos correspondentes em `debug/e2e/`.
5. Validar que `sample_artifacts[].crop_path` existem.

**Aceite:**
- `pytest pipeline/tests/regression/test_visual_regression_manifest.py -v` passa.
- Excluir qualquer classe esperada do `qa_report` faz o teste falhar com mensagem precisa: `"Expected issue class 'TEXT_CLIPPED' missing from run one_second_ch2"`.
- Se um manifesto de run local referencia caminho que não existe, teste é **skipado com warning** em ambiente de desenvolvimento.
- Em CI, só rodar manifestos marcados como `ci_fixture: true` e com fixtures pequenas versionadas; não deixar CI depender de `N:/TraduzAI/DEBUGM/runs/...`.

**Rollback:** apagar pasta `manifests/` e o teste novo.

#### R1.5 — Links de evidência por QA issue

**Objetivo:** Cada issue crítica precisa apontar para os artefatos que explicam a falha. A estrutura `debug/e2e/00..11` já existe, mas o problema recorrente é o vínculo incompleto entre `qa_report.json`, `decision_trace.jsonl`, crop, máscara, inpaint decision, render plan e página traduzida.

**Arquivos:**
- [pipeline/debug_tools/recorder.py](../../pipeline/debug_tools/recorder.py) — expor helper para registrar artifact refs por `trace_id`.
- [pipeline/debug_tools/report.py](../../pipeline/debug_tools/report.py) — preencher `artifact_links` por issue.
- [pipeline/qa/export_gate.py](../../pipeline/qa/export_gate.py) — incluir `trace_id`, `page_id`, `band_id`, `text_instance_id`, `artifact_links` nos blockers.
- [pipeline/tools/export_visual_review_sheet.py](../../pipeline/tools/export_visual_review_sheet.py) — gerar contact sheets por classe de issue.
- [pipeline/tests/test_debug_report.py](../../pipeline/tests/test_debug_report.py) (estender)
- [pipeline/tests/test_export_gate_debug_consistency.py](../../pipeline/tests/test_export_gate_debug_consistency.py) (estender)

**Aceite:**
- Uma issue `render_outside_balloon` tem links para `translated`, `render_plan_final`, `layout_geometry` e crop de contact sheet.
- Uma issue `weak_text_residual_after_inpaint` tem links para `mask_overlay`, `inpaint_decision.json` e crop pós-inpaint.
- `qa_report.json` não pode ter blocker com `artifact_links: []`.

**Rollback:** manter os campos extras como opcionais; remover geração de contact sheet por issue sem alterar gate.

---

### Release 2 — Visual hardening (semanas 2-3)

**Pré-requisito:** R1 fechado. Sem R1, R2 corrige problemas que a UI continua reportando como sucesso.

#### R2.1 — Consolidar `mask_evidence` como objeto único

**Objetivo:** Cada region passa a expor um único `mask_evidence: {...}` consolidando o que hoje está disperso entre `cjk_mask_fusion`, `oar_ocr_adapter`, e helpers de `mask_builder`.

**Arquivos:**
- [pipeline/inpainter/mask_builder.py](../../pipeline/inpainter/mask_builder.py)
- [pipeline/inpainter/__init__.py](../../pipeline/inpainter/__init__.py) (router)
- [pipeline/vision_stack/cjk_mask_fusion.py](../../pipeline/vision_stack/cjk_mask_fusion.py)
- [pipeline/vision_stack/oar_ocr_adapter.py](../../pipeline/vision_stack/oar_ocr_adapter.py)
- [pipeline/qa/translation_qa.py](../../pipeline/qa/translation_qa.py) (adicionar `fast_fill_no_glyph_evidence: "critical"` a `FLAG_SEVERITY`)
- [pipeline/tests/test_text_mask_evidence.py](../../pipeline/tests/test_text_mask_evidence.py) (estender)

**Contrato:**

```python
# Vive em region["mask_evidence"]
{
  "kind": "ocr_pixels" | "glyph_segmentation" | "cjk_segmentation"
          | "clipped_line_polygon" | "verified_rect_sign" | "none",
  "raw_mask_pixels": int,
  "expanded_mask_pixels": int,
  "evidence_score": float,           # 0.0 - 1.0
  "fast_fill_allowed": bool,
  "fast_fill_reject_reasons": [str]  # nomes estáveis: "raw_mask_pixels_zero",
                                     #                 "high_local_variance",
                                     #                 "color_diverges_across_lobes",
                                     #                 "transparency_detected",
                                     #                 "coverage_too_low",
                                     #                 "post_inpaint_ocr_still_reads_source"
}
```

**Política consolidada (uma regra, em `mask_builder.py`):**

Fast fill **só é permitido** se:
1. `kind` ∈ {`ocr_pixels`, `glyph_segmentation`, `cjk_segmentation`, `verified_rect_sign`}, **E**
2. `raw_mask_pixels > 0` (para `dialogue`/`narration`), **E**
3. variância local da amostra de cor está abaixo do threshold, **E**
4. cor difere < threshold entre lobos do balão.

Senão: `fast_fill_allowed = false`, motivos preenchidos, e o roteador escolhe inpaint real (LaMA).

**Aceite:**
- Test regression: band de `content_class: dialogue` com `raw_mask_pixels=0` produz `mask_evidence.fast_fill_allowed=false`, flag `fast_fill_no_glyph_evidence`, e gate bloqueia.
- Test regression: band com evidência válida não dispara flag e usa fast fill.
- Não-regressão de [pipeline/tests/test_mask_builder.py](../../pipeline/tests/test_mask_builder.py) e [pipeline/tests/test_inpaint_mask_geometry.py](../../pipeline/tests/test_inpaint_mask_geometry.py).

**Rollback:** flag em `FLAG_SEVERITY` permanece (não bloqueia nada); reverter a função `_decide_fast_fill()` para o comportamento anterior.

#### R2.2 — `fit_attempts` rastreável no typeset

**Objetivo:** Persistir histórico de tentativas de fit; eliminar texto minúsculo silencioso.

**Arquivos:**
- [pipeline/typesetter/renderer.py](../../pipeline/typesetter/renderer.py)
- [pipeline/layout/balloon_layout.py](../../pipeline/layout/balloon_layout.py)
- [pipeline/layout/simple_text_geometry.py](../../pipeline/layout/simple_text_geometry.py)
- [pipeline/qa/translation_qa.py](../../pipeline/qa/translation_qa.py) (adicionar `fit_below_minimum_legible: "critical"`, `missing_render_bbox: "critical"`)
- [pipeline/tests/test_typesetting_layout.py](../../pipeline/tests/test_typesetting_layout.py) (estender)
- [pipeline/tests/test_typesetting_renderer.py](../../pipeline/tests/test_typesetting_renderer.py) (estender)

**Contrato:**

```python
# Em region["fit_attempts"]
[
  {"font_px": 22, "lines": 1, "status": "overflow"},
  {"font_px": 20, "lines": 2, "status": "ok"}
]

# Em region["fit_status"]
"ok" | "below_minimum_legible"
```

**Política:**

1. Tentar mais quebras de linha **antes** de reduzir fonte.
2. Tentar largura disponível total do balão.
3. Preservar espaçamento legível (line-height ≥ 1.10).
4. `min_font_px = max(12, page_width * 0.012)`.
5. Se passar de `min_font_px` ainda overflow → `fit_status: "below_minimum_legible"` + flag `fit_below_minimum_legible` + `route_action` permanece, mas gate bloqueia export.

**Restrição técnica do renderer:**
- Não introduzir paralelismo. FT2Font não é thread-safe.
- Não usar PIL/TextPath para medição. Manter `len(text) * size * 0.55` (estimativa) e `FT2Font` final.

**Aceite:**
- Layer com texto que não cabe gera `fit_attempts` com tentativas + `fit_status: "below_minimum_legible"` + flag crítica.
- Export é bloqueado.
- Layer normal grava `fit_attempts` com 1-2 entries e `fit_status: "ok"`.
- Não-regressão dos testes de typesetter.

**Rollback:** desativar persistência de `fit_attempts` (manter cálculo, sem gravar) e remover flag de `FLAG_SEVERITY`.

#### R2.3 — Gate único de fast fill

**Objetivo:** Não permitir fast fill sem `mask_evidence.fast_fill_allowed=true`.

**Arquivos:**
- [pipeline/inpainter/__init__.py](../../pipeline/inpainter/__init__.py) (router)
- [pipeline/strip/process_bands.py](../../pipeline/strip/process_bands.py) (auditar `_strip_used_fast_white_fill`, `_strip_used_fast_local_fill`)
- `pipeline/tests/regression/test_fast_fill_evidence_gate.py` (novo)

**Trabalho:** Após R2.1, basta consumir `mask_evidence.fast_fill_allowed`. Onde hoje a decisão é local, virar consulta a esse campo.

**Aceite:** Test regression valida fixtures de balão branco simples, balão preto/glow com interior sólido, balão colorido sólido e balão texturizado/translúcido. Fast fill só dispara quando a amostra interna é sólida e há `mask_evidence.fast_fill_allowed=true`; balões texturizados/translúcidos caem para inpaint real ou review.

---

### Release 3 — Semantic Quality (semana 4)

**Pré-requisito:** R1 fechado (UI já distingue blocked). R2.1 ajuda mas não bloqueia.

#### R3.1 — `route_action` como fonte de verdade

**Objetivo:** Consolidar `skip_processing` + `is_watermark` + `is_non_english` + classificação de tipo em **uma decisão única** por bloco.

**Arquivos:**
- [pipeline/ocr/postprocess.py](../../pipeline/ocr/postprocess.py)
- [pipeline/vision_stack/runtime.py](../../pipeline/vision_stack/runtime.py)
- [pipeline/main.py](../../pipeline/main.py) (chamada do roteador)
- [pipeline/tests/test_content_classifier.py](../../pipeline/tests/test_content_classifier.py) (estender)
- [pipeline/tests/test_special_content_router_v2.py](../../pipeline/tests/test_special_content_router_v2.py) (estender)
- `pipeline/tests/test_content_routing.py` (novo apenas se os testes existentes ficarem grandes demais)

**Contrato:**

```python
# Em region["route_action"]
"translate_inpaint_render"   # caso normal
| "translate_render_only"    # signage detectada ou texto onde inpaint apaga arte
| "inpaint_only"             # watermark/email/READ AT...
| "preserve"                 # SFX Hangul, logo, title card, kanji preservado
| "review_required"          # OCR truncado, low confidence em diálogo
| "skip"                     # outros: ruído, vazio
```

**Regra:** `skip_processing` é **derivado** de `route_action == "skip"` por compatibilidade, não removido.

**Persistir:**

```python
{
  "route_action": "translate_inpaint_render",
  "route_reason": "dialogue_balloon_with_english_text",  # texto curto, auditável
  "content_class": "dialogue"  # já existe
}
```

**Aceite:**
- Bloco `is_watermark=True` → `route_action: "inpaint_only"`, `route_reason: "watermark_detected"`.
- Bloco Hangul SFX → `route_action: "preserve"`, `route_reason: "korean_sfx_preserved_by_default"`.
- Bloco diálogo EN curto → `route_action: "translate_inpaint_render"`.
- Não-regressão: `is_watermark`, `is_non_english`, `skip_processing` continuam expostos e consistentes com `route_action`.

#### R3.2 — Name-lock antes da tradução

**Objetivo:** Preservar nomes próprios e termos de tratamento via placeholders robustos.

**Arquivos:**
- [pipeline/translator/translate.py](../../pipeline/translator/translate.py)
- [pipeline/translator/term_protection.py](../../pipeline/translator/term_protection.py) — ponto preferencial; criar `name_lock.py` só se o módulo ficar grande demais.
- [pipeline/context/entity_detector.py](../../pipeline/context/entity_detector.py)
- `pipeline/tests/test_name_lock.py` (novo, ou estender testes de `term_protection` se já existirem)

**Algoritmo:**

```text
1. Antes do Google: substituir entidades protegidas por tokens ASCII TZN.
   - Entidades vêm do work_context (personagens + aliases) + glossário do projeto.
   - Token format: __TZN_NAME_0__, __TZN_NAME_1__.
2. Após Google: restaurar pares (token → nome original).
3. Validar:
   - placeholder_count_in == placeholder_count_out (senão → flag `unrestored_placeholder`, critical)
   - protected_terms_in ⊆ restored_terms_out
```

**Cuidados explícitos:**
- **Não** detectar entidade só por capitalização (evita `ONE`, `HOSPITAL`, `READ`, `THE`, `I`).
- Denylist em código (não em config) para esses 5+ ASCII uppercase.
- Usar `work_context.characters + aliases` como fonte primária.

**Aceite:**
- Test: `Wonho disse "estou indo"` traduz sem virar "maravilhoso".
- Test: `Hosu...?` preserva pontuação e nome.
- Test: `ONE!` (count, não nome) não é placeholder.

#### R3.3 — OCR retention

**Objetivo:** Não descartar diálogo curto válido em balão.

**Arquivos:**
- [pipeline/ocr/postprocess.py](../../pipeline/ocr/postprocess.py)
- [pipeline/ocr/ocr_normalizer.py](../../pipeline/ocr/ocr_normalizer.py)
- [pipeline/qa/translation_qa.py](../../pipeline/qa/translation_qa.py) (adicionar `ocr_truncated_or_joined: "high"`)
- `pipeline/tests/test_ocr_retention.py` (novo)

**Regra de retenção:**

Não descartar (mesmo com `confidence < 0.55`) quando:
- O bloco está **dentro de balão** (tem `balloon_bbox`), **E**
- A região é speech/narration box, **E**
- O texto é curto mas alfabetizado/pontuado como diálogo, **E**
- Não há duplicata melhor na mesma band.

**Flag de truncated/joined:**
- Texto contém quebras tipo `Why?!What's`, `WEDO`, `ittous`, `lyingil` → `ocr_truncated_or_joined: "high"` + `route_action: "review_required"`.

**Aceite:**
- Test: `What happened?` (curto, low-conf, em balão) é retido com `route_action: "translate_inpaint_render"`.
- Test: `WEDO` flagado como joined.

---

### Release 4 — Edge Cases (semana 5)

**Pré-requisito:** R1-R3 fechados.

#### R4.1 — Política de balão conectado (fallback)

**Objetivo:** Já existe estrutura; falta política explícita para baixa confiança.

**Arquivos:**
- [pipeline/strip/process_bands.py](../../pipeline/strip/process_bands.py)
- [pipeline/layout/balloon_layout.py](../../pipeline/layout/balloon_layout.py)
- [pipeline/qa/translation_qa.py](../../pipeline/qa/translation_qa.py) (adicionar `lobe_assignment_low_confidence: "high"`)
- [pipeline/tests/regression/test_connected_text_cut.py](../../pipeline/tests/regression/test_connected_text_cut.py) (estender)

**Política:**
- Adicionar `lobe_assignment_confidence: float` ao region.
- Se `< 0.6` e bloco tem tradução não-trivial: `lobe_assignment_low_confidence` (high) → gate decide.
- Se confiança alta: render normal.

#### R4.2 — Texto rotacionado

**Objetivo:** Geometria de ângulo + política consistente.

**Arquivos:**
- [pipeline/vision_stack/ocr.py](../../pipeline/vision_stack/ocr.py) — persistir `text_angle_degrees`, `text_orientation`, `rotated_polygon` quando OCR expõe ângulo.
- [pipeline/vision_stack/detector.py](../../pipeline/vision_stack/detector.py) — só complementar ângulo quando o backend de detector trouxer polygon/rotated bbox confiável.
- [pipeline/ocr/postprocess.py](../../pipeline/ocr/postprocess.py) — normalizar metadados de ângulo sem perder `trace_id`.
- [pipeline/inpainter/mask_builder.py](../../pipeline/inpainter/mask_builder.py) — usar polígono rotacionado, não axis-aligned, quando `|angle| > 5`.
- [pipeline/typesetter/renderer.py](../../pipeline/typesetter/renderer.py) — render no mesmo ângulo para `dialogue` + `signage`.
- [pipeline/qa/translation_qa.py](../../pipeline/qa/translation_qa.py) (adicionar `rotated_text_policy_unmet: "high"`)
- `pipeline/tests/test_rotated_text.py` (novo)

**Policy (em `runtime_profiles.py`):**

```python
ROTATED_TEXT_POLICY = {
  "dialogue":     {"render_same_angle_if_abs_angle_in": (5, 35), "block_if_vertical": True},
  "signage":      {"render_same_angle": True},
  "sfx_preserve": {"inpaint": False, "render": False},
  "title_card":   {"preserve_by_default": True},
}
```

#### R4.3 — Validação na matriz completa

**Objetivo:** Rodar R1.4 com todos os capítulos preenchidos e cruzar resultados.

**Trabalho:**
- Atualizar todos 6 manifestos com flags esperadas pós-R3.
- Adicionar relatório `docs/debug/visual_regression_report.md` gerado pelo teste com tabela: capítulo × páginas × flags antes/depois × status final.

---

## 7. Definition of Done (global, por critério)

A implementação está completa quando, **em ordem**:

1. ☐ **R1.1**: Rust emite `completion_status: "blocked"` quando `qa.export_gate.status == "BLOCK"` no `project.json`.
2. ☐ **R1.2**: UI mostra banner em `Preview` e badge em `Home` para `done_blocked`.
3. ☐ **R1.3**: Run bloqueado grava `output_review_state: "blocked_preview"` sem migrar a árvore `translated/`; projetos antigos abrem sem erro.
4. ☐ **R1.4**: Matriz de manifestos roda localmente/CI e bloqueia regressão das classes de issue registradas.
5. ☐ **R1.5**: Todo blocker em `qa_report.json` tem `artifact_links` não vazios e rastreáveis por `trace_id`.
6. ☐ **R2.1**: Toda region tem `mask_evidence` consolidado; `fast_fill_allowed` é a única chave da decisão de fast fill.
7. ☐ **R2.2**: Todo layer não-skip com `route_action == translate_*` tem `render_bbox`, `safe_text_box`, `fit_attempts`, `fit_status`, `mask_evidence`, `qa_flags`.
8. ☐ **R2.3**: Fast fill em diálogo sem evidência de glifo bloqueia (`fast_fill_no_glyph_evidence`).
9. ☐ **R3.1**: `route_action` está populado em 100% dos blocks; `skip_processing` é derivado.
10. ☐ **R3.2**: Nomes próprios protegidos não são traduzidos semanticamente; `placeholder_count_in == placeholder_count_out` em 100% dos batches.
11. ☐ **R3.3**: Diálogo curto válido em balão é retido com `route_action: "translate_inpaint_render"` ou `"review_required"`.
12. ☐ **R4.1**: Balões conectados com baixa confiança bloqueiam render.
13. ☐ **R4.2**: Texto rotacionado segue policy; gate bloqueia violações.
14. ☐ **R4.3**: Matriz completa: 6 capítulos com manifestos atualizados e relatório gerado.
15. ☐ **Taxonomia**: Zero nomes de flags paralelos. Toda flag nova está em `FLAG_SEVERITY`.
16. ☐ **Performance**: Tempo total ≤ baseline × 1.15 em runs de regressão (ver §10).

---

## 8. Risk Register

| ID | Risco | Impacto | Probabilidade | Mitigação |
|----|---|---|---|---|
| RR-1 | Uma migração física para `translated/{approved,blocked_preview}` quebra editor/projetos antigos | Alto | Média | Não fazer essa migração em R1; usar `output_review_state` no `project.json` e manter paths atuais até uma task dedicada pós-R4 |
| RR-2 | Flags novas em `FLAG_SEVERITY` disparam massa de bloqueios em capítulos já aprovados | Alto | Alta | Introduzir a coleta por feature flag/telemetria primeiro; promover severidade efetiva só depois da matriz validar impacto. Quando promovida, a severidade declarada deve bater com §5 |
| RR-3 | `route_action` introduz divergência com `skip_processing` legado | Médio | Média | `skip_processing` continua sendo escrito; teste de invariante: `skip_processing == (route_action == "skip")` |
| RR-4 | `fit_attempts` aumenta tamanho de `project.json` significativamente | Baixo | Média | Limitar a últimas 4 tentativas; descartar atributos detalhados em export final |
| RR-5 | `name_lock` placeholders sobrevivem à tradução errada e quebram visualmente | Alto | Baixa | Validar `placeholder_count_in == out`; se falhar, flag `unrestored_placeholder` (critical) → gate bloqueia |
| RR-6 | Tempo de OCR residual estoura budget | Médio | Alta | OCR residual só roda quando `route_action in {translate_*, review_required}` E `mask_evidence.evidence_score < threshold`. Cache por `sha256(page_id+crop_bbox+mask_hash+inpaint_strategy)` (já existe via `editor_vision_cache.py`) |
| RR-7 | UI banner R1.2 é ignorado pelo usuário em workflow rápido | Médio | Média | Em `Home`, o card bloqueado é vermelho persistente; export aprovado é apenas botão "Exportar" — bloqueado mostra "Resolver issues" |
| RR-8 | Manifestos R1.4 referenciam runs locais que não existem em CI | Médio | Alta | Teste skipa (com warning) quando `run_path` não existe; CI roda só manifestos que apontam para fixtures versionadas |
| RR-9 | Plano R4.2 (rotated text) introduz instabilidade no detector | Alto | Média | Atrás de feature flag em `runtime_profiles.py`; default off por 1 release; validar via fixtures rotacionadas antes de habilitar |

---

## 9. Apêndice A — Feature flags (estratégia consolidada)

**Decisão:** Não criar sistema novo. Reusar [pipeline/runtime_profiles.py](../../pipeline/runtime_profiles.py).

```python
# Em pipeline/runtime_profiles.py
VISUAL_PIPELINE_FLAGS = {
  "strict_export_gate": True,            # default on, ligado em R1
  "safe_fast_fill": True,                # default on, ligado em R2.3
  "ocr_residual_check": "targeted",      # off|targeted|always
  "name_lock": True,                     # ligado em R3.2
  "connected_balloon_v2": False,         # ligado em R4.1
  "rotated_text_v2": False,              # ligado em R4.2
}

# Pode ser override via env var: TRADUZAI_FLAG_<NOME>=true|false
# Ou via config.json: { "flags": { ... } }
```

Cada flag é lida no início da pipeline e propagada via `runtime_profile_decision` (já existe). Toda decisão de gate consulta `runtime_profile_decision["flags"]["<flag>"]`.

Se `runtime_profile_decision` ainda não expuser `flags` nesse formato no ponto de uso, implementar o menor adaptador em `runtime_profiles.py` primeiro. Não espalhar `os.getenv(...)` diretamente pelos módulos de OCR/inpaint/typeset.

---

## 10. Apêndice B — Métricas de performance

**Persistir em `project.json` → `qa.timing`:**

```jsonc
{
  "timing": {
    "total_sec": 287.4,
    "instrumented_sec": 271.8,
    "unattributed_sec": 15.6,
    "durations_sec": {
      "ocr": 84.2,
      "routing": 0.8,
      "translation": 22.4,
      "mask_build": 12.6,
      "inpaint": 124.0,
      "residual_ocr": 5.0,
      "typeset": 6.2,
      "qa": 1.4,
      "export_gate": 0.2
    },
    "budget_check": {
      "baseline_total_sec": 250.0,
      "current_total_sec": 287.4,
      "ratio": 1.15,
      "within_budget": true
    }
  }
}
```

**Já existe**: `_PipelineTiming` em [pipeline/main.py:37](../../pipeline/main.py).

**Trabalho restante (em R1.4):** copiar `timing` snapshot para `qa.timing` no `project.json` final, com `baseline_total_sec` armazenado por capítulo no manifesto.

**Budget:** runs de regressão devem respeitar `current / baseline ≤ 1.15`. Acima disso, teste falha com aviso (não-bloqueante por padrão; flag `--strict-perf` torna bloqueante).

---

## 11. Apêndice C — Waivers (preservado do plano original)

Waivers permitem que fixtures de debug rodem mesmo com flags críticas conhecidas.

**Formato (em `pipeline/tests/regression/manifests/{run}.json`):**

```jsonc
{
  "...": "...",
  "waivers": [
    {
      "waiver_id": "debug_2026_05_25_001",
      "flag": "fit_below_minimum_legible",
      "scope": "debug_only",
      "reason": "fixture intentionally validates blocked export",
      "expires": "2026-06-15",
      "approved_by": "vinicius"
    }
  ]
}
```

**Comportamento:**
- Teste de manifesto **ignora** flags com waiver ativo.
- Após `expires`, waiver vence e teste passa a falhar com mensagem clara: `"Waiver debug_2026_05_25_001 expired 2026-06-15; review fixture or extend waiver"`.

---

## 12. Apêndice D — Matriz de capítulos para validação final

| Capítulo | Páginas críticas | Flags esperadas (pós-R3) | Status alvo |
|---|---|---|---|
| Articuno ch61 | TBD via run inicial | TBD | PASS / REVIEW |
| Chapter 39 | TBD | TBD | PASS / REVIEW |
| God of Death ch2 | TBD | TBD | PASS / REVIEW |
| One Second ch1 | TBD | TBD | PASS / REVIEW |
| One Second ch2 | 2, 3, 4, 5, 6, 7 | `TEXT_CLIPPED`, `TEXT_OVERFLOW`, `render_outside_balloon`, `untranslated_english`, `fast_fill_insufficient_coverage` | Pós-correção: PASS ou REVIEW com waiver explícito; BLOCK não é aceitável como saída normal |
| Chapter 1 | TBD | TBD | PASS / REVIEW |

Preencher TBDs em **R4.3** após rodar cada capítulo uma vez pós-R3.3.

---

## 13. Apêndice E — Estratégia de commits

Granularidade: 1 commit por task, mensagens em inglês conforme estilo do repo (`feat:`/`fix:`/`refactor:`/`test:`).

```text
# R1
feat(rust): emit export_gate status in pipeline-complete event
feat(ui): handle done_blocked and needs_review project status
feat(pipeline): persist blocked_preview review state without path migration
test(regression): add visual regression manifest framework
feat(debug): link qa blockers to visual artifacts

# R2
refactor(mask): consolidate mask_evidence into single object
feat(typeset): persist fit_attempts trace and fit_status
fix(inpaint): require glyph evidence for fast fill

# R3
refactor(ocr): introduce route_action as single source of truth
feat(translate): name-lock with TZN placeholder validation
fix(ocr): retain short valid dialogue inside balloons

# R4
fix(layout): block render on low-confidence lobe assignment
feat(detect): persist rotated text geometry and policy
test(regression): validate full chapter matrix
```

---

## 14. Self-review desta revisão

Antes de declarar este plano "100%", os pontos abaixo foram conscientemente verificados:

| Checagem | Status | Evidência |
|---|---|---|
| Schema atual confirmado | OK | [pipeline/schema/project_schema_v12.py](../../pipeline/schema/project_schema_v12.py) existe |
| Export gate atual confirmado | OK | [pipeline/qa/export_gate.py:38-58](../../pipeline/qa/export_gate.py) retorna PASS/BLOCK/OVERRIDDEN |
| Severidades atuais confirmadas | OK | `FLAG_SEVERITY` em [pipeline/qa/translation_qa.py:8](../../pipeline/qa/translation_qa.py) tem 70+ flags |
| Gap Rust/UI confirmado | OK | `Grep` em `src/` por `export_gate` retornou 0 arquivos; [pipeline.rs:524-540](../../src-tauri/src/commands/pipeline.rs) emite só `success` |
| `content_class` confirmado existente | OK | 20+ arquivos |
| `connected_balloon_id`/`lobe_id` confirmados existentes | OK | 20+ arquivos |
| `safe_text_box` confirmado existente | OK | 21 arquivos |
| `mask_evidence` parcial | OK | `cjk_mask_fusion.py`, `oar_ocr_adapter.py`, `test_text_mask_evidence.py` |
| Mapping de flags substitui todos os nomes propostos | OK | §5, 6 nomes mapeados |
| Tests citados existem | OK (com fixes) | §3 item #9 — `test_final_geometry_contract.py`, `test_export_gate.py`, `test_export_gate_debug_consistency.py` já existem |
| Restrição técnica do renderer (FT2Font, no thread-safe) respeitada em R2.2 | OK | Citado explicitamente |
| Migração para projetos antigos endereçada em R1.3 | OK | §6 R1.3 estratégia de migração |
| Feature flags têm casa concreta | OK | Apêndice A → `runtime_profiles.py` |
| Performance budget concreto | OK | Apêndice B, com fórmula |
| Waivers preservados do plano original | OK | Apêndice C |
| Matriz de capítulos preservada | OK | Apêndice D |
| Riscos catalogados | OK | §8, 9 riscos com mitigação |
| Definition of Done verificável (não-vago) | OK | §7, 15 critérios |
| Ordem dos releases tem pré-requisitos explícitos | OK | Cada release começa com "Pré-requisito" |
| Rollback plan por task | OK | Cada task tem seção "Rollback" |
| Não há contradição interna entre R1.3 e leitura do editor | OK | R1.3 prevê leitura via `arquivo_traduzido` do project.json (caminho absoluto) — não montado |
| Compatibilidade com legado (`skip_processing`) | OK | R3.1 mantém como campo derivado |

**Gaps remanescentes (assumidos):**
- Caminhos absolutos dos runs em §12 estão como `TBD` — só podem ser preenchidos após R3 rodar nos 6 capítulos.
- `baseline_total_sec` por capítulo (Apêndice B) precisa ser registrado pós-R1 com pipeline atual antes das mudanças de R2-R4.

Esses dois TBDs são esperados; ambos têm momento explícito de preenchimento (R4.3 e R1.4 respectivamente).

---

## 15. Veredito final

- **Diagnóstico do plano original:** correto e bem articulado.
- **Implementação proposta no original:** sobrestima o gap real em ~60% (duplicaria estruturas existentes).
- **Este plano revisado:** executa **R1 primeiro** (gap de integração que cria falso sucesso) e trata R2-R4 como **extensões consolidadoras** que atacam as causas visuais comprovadas.
- **Risco se o original for executado literalmente:** taxonomia paralela de flags, schema duplicado, e regressão silenciosa dos testes atuais (`test_qa_flag_propagation_v2`, `test_export_gate`, `test_final_geometry_contract`).
- **Aprovado para execução** nesta forma revisada, em ordem R1 → R2 → R3 → R4.

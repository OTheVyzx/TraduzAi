# Ownership de arquivos — INTEGRACAO_V1

| Frente | Ownership de escrita inicial |
|---|---|
| Integração | schema/adapter canônico em `pipeline/integration_v1/**`, `pipeline/consumer_fast/**`, persistência/jobs/export backend, `BOOTSTRAP.json` e adapters compartilhados |
| Vision | `AnalysisRecord`, providers OCR/análise, geometria/máscaras propostas e adapters Vision designados |
| Renderer | plano tipográfico, preferências, raster/receitas e `pipeline/typesetter/**` após handoff explícito |
| Studio | UI React/TS, stores e adapter Tauri designado; não publica um segundo projeto nem rasteriza |

Arquivos herdados de `pipeline/ownership/**` permanecem sob integração, exceto `render_geometry.py`, cujo conteúdo inicial veio do overlay Vision e só muda por handoff/versionamento. `pipeline/typesetter/**` está congelado no bootstrap e passa a Renderer após criação do seu worktree. Um escritor por arquivo continua obrigatório.

O contrato Studio é `traduzai.studio-ipc.v1`: jobs endereçados por `job_id`, eventos monotônicos por revisão/sequência, escrita por `expected_revision` + `idempotency_key`, retypeset com recibo de receita e exportação final separada da exportação diagnóstica.

O campo `owner` do registry identifica quem produz e evolui semanticamente cada payload; não concede escrita no arquivo canônico. Integração é o único escritor do schema publicado. Vision produz `AnalysisRecord`; Renderer produz `LayoutRequest/LayoutPlan/Recipe`; Studio produz `ReviewDecision` e seu adapter Tauri/UI. Mudança de envelope segue proposta versionada e publicação pela Integração.

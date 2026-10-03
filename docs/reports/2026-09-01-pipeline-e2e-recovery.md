# Recuperação E2E total do pipeline — diário de execução

## Estado inicial

- Data local: 2026-09-01.
- Branch: `Troca_de_motores`.
- HEAD: `57caa2e88b493ae8bd32d2181fd04d6a3dc06cfe`.
- Python: 3.12.10, `N:\TraduzAI\pipeline\venv\Scripts\python.exe`.
- Checkout preexistente extremamente sujo: 51.090 entradas `D`, 96 `M` e
  129 não rastreadas no snapshot inicial. Nenhuma dessas alterações foi
  revertida, limpa, movida ou adicionada ao índice.
- Fontes CBZ permanecem somente leitura.
- Referência histórica congelada:
  `docs/reports/2026-08-31-benchmark-owner-enforce-failures.md`.
- Resumo histórico SHA-256:
  `612359D224A656E703EDBB1DDD9CEF1267E0479E641D9408C2179F0DC0A2ED99`.

## Baseline automatizado anterior à nova implementação

Comando executado dentro de `pipeline` sobre coverage/recovery, builder,
enforce, tradução, page pipeline, execução, repair/geometria, QA/export,
typesetting e emissão principal:

- 928 testes passaram;
- 53 falharam;
- 6 subtests passaram;
- duração: 110,90 s.

As 53 falhas estão todas em `tests/test_typesetting_renderer.py` e já existiam
antes da implementação do novo timing. Elas abrangem capacidade/âncora de
balões escuros, connected lobes, safe boxes, tamanho de fonte e alguns
contratos de render. Não foram relaxadas nem alteradas nesta fase.

A primeira tentativa do baseline, disparada da raiz do repositório, não chegou
a coletar testes por `ModuleNotFoundError: ownership`; ela foi repetida do
diretório correto e não é contada como baseline funcional.

## Primeira divergência do timing

O recorder `_PipelineTiming` era criado dentro de `_run_pipeline`, enquanto
`_run_pipeline_runner_cli` chamava o pipeline sem `try/finally`. O sidecar
`performance_timing.json` só era escrito após projeto, QA, export gate e
cleanup. Qualquer exceção anterior escapava sem artefato final.

## Correção de timing — RED/GREEN

Foi adicionado `tests/test_pipeline_performance_timing.py`, cobrindo:

1. falha produz sidecar final atômico;
2. snapshot parcial durante execução;
3. `finalize()` idempotente;
4. timers aninhados sem dupla contagem;
5. runner real finaliza timing e relança a exceção;
6. falha do profiler não mascara a exceção original;
7. relógio falso mantém total/spans coerentes;
8. `KeyboardInterrupt` vira `interrupted`;
9. sidecar truncado preexistente é substituído atomicamente;
10. stage bem-sucedido termina como `completed`.

O conjunto foi ampliado para cobrir tambÃ©m bridge do strip, contagem de cargas
de modelo e `SystemExit(2)` emitido depois de um export gate `BLOCK`. Esse exit
continua sendo relanÃ§ado ao chamador, mas o timing fica `completed`, pois a run
jÃ¡ alcanÃ§ou conclusÃ£o tÃ©cnica.

Resultado focado atual (timing + page pipeline + traduÃ§Ã£o + enforce):
**123 passed in 4,02 s**.

O recorder agora atravessa `run_chapter`, `run_page_owner_pipeline` e
`execute_owner_page_graph`. Os spans downstream distinguem
`owner_geometry`, `owner_mask_creation`, `owner_inpaint`,
`owner_typesetting` e `owner_final_render`, em vez de esconder todo o custo
em `owner_execution`.

## Gate real de falha proposital

Workdir:
`N:\TraduzAI\.codex-tmp\pipeline_e2e_recovery_20260901\timing_failure_probe_02`.

Uma execução CLI com input inexistente terminou com:

- exit code: 1;
- timing: presente e JSON válido;
- status: `failed`;
- total: 3,3903 s;
- failed stage: `extract_source`;
- exception: `SystemExit(1)`;
- `load_config`, `apply_runtime_profile` e `load_corpus`: `completed`;
- `extract_source`: `failed`.

Isso fecha o gate de lifecycle externo em falha. A taxonomia detalhada,
contadores, páginas e modelos ainda precisa ser integrada antes de declarar a
Fase 1 completa.

## Estado E2E

- Grand Finale 24, run real 02: terminou em 647,6799 s com
  `TranslationValidationExhausted`.
- Timing final: presente, `status=failed`, `failed_stage=translation`.
- Contadores no terminal: 22 componentes de coverage, 29 observações OCR,
  15 owners, 15 chamadas de tradução e 15 falhas.
- Maiores spans: `strip_run_chapter=592,8663 s`,
  `page_ocr_coverage=244,6696 s`,
  `source_component_discovery=163,9045 s`,
  `owner_control_plane=141,6214 s` e `translation=18,6261 s`.
- Não houve `project.json` nem export gate nessa run; ela continua sendo
  `PIPELINE_ERROR`, não conclusão técnica.
- Foi criado o sidecar incremental `translation_attempts.jsonl`, sem texto
  integral, com owner, hashes, provider/model, latência e motivo do validador.
- Tradução agora isola `TranslationValidationExhausted` por owner: bindings
  válidos permanecem autoritativos; owner sem binding preserva os pixels,
  recebe `translation_unresolved`, não recebe action mask nem commit e bloqueia
  o QA. Indisponibilidade global de providers continua fatal.
- Suíte focada após essa mudança: **125 passed in 4,97 s**.
- E2E-1 Grand Finale 24: run 03 em andamento.
- Chegou ao export gate: não comprovado.
- Status geral atual: `NO_GO` até existir um gate real com os artefatos
  obrigatórios no mesmo workdir.

## Continuação das runs Grand Finale 24

As runs seguintes foram mantidas em workdirs independentes e não substituem
nem apagam evidência anterior:

| Run | Resultado terminal | Classificação |
|---|---|---|
| 03 | percorreu as 18 páginas, preservou 5 owners sem tradução e falhou na primeira promoção por residual de idioma-fonte | `PIPELINE_ERROR`; revelou que a tradução já estava isolada por owner |
| 04 | residual foi convertido em preservação fail-closed; falhou na cardinalidade de verdicts visuais | `PIPELINE_ERROR`; invariante downstream ainda supunha que todo binding produzia substituição |
| 05 | passou a primeira cardinalidade; falhou na construção do `TerminalPixelProof` | `PIPELINE_ERROR`; materializações/verdicts ainda eram comparados a bindings preservados |
| 06 | passou a construção do proof; falhou na promoção do `PageExecutionResult` | `PIPELINE_ERROR`; segunda validação repetia a mesma suposição de cardinalidade |
| 07 | passou as cardinalidades; falhou no hash-chain dos bindings do proof | `PIPELINE_ERROR`; validação comparava o proof filtrado ao journal completo |
| 08 | chegou à página 17/18 e foi encerrada externamente antes da finalização | `INTERRUPTED/INCONCLUSIVE`; só há timing parcial, portanto não entra na média nem conta como falha funcional |
| 09 | não entrou no pipeline porque o launcher Windows separou um caminho com espaços | erro de invocação, não benchmark |
| 10 | processou 18 páginas; falhou antes do gate por 193 pixels residuais em um owner | `PIPELINE_ERROR`; residual físico não detectado pelo OCR terminal |
| 11 | revogou o owner residual e processou 18 páginas; falhou ao validar aliases dos stages visuais | `PIPELINE_ERROR`; `typeset` ainda referia o candidato anterior à recomposição |
| 12 | relançada após reconstruir cleanup/typeset a partir da autoridade restante | em andamento |

Cada invariante das runs 04–07 foi reproduzida por teste unitário antes do
ajuste. O contrato atual distingue o journal completo de tradução de três
subconjuntos físicos: bindings que preservam pixels, bindings com commit
autoritativo e owners realmente substituídos que exigem materialização e
verdict. Owners rejeitados não recebem commit, não entram como substituição e
continuam visíveis na evidência de QA como `BLOCK`.

A suíte focada acumulada após essas correções tem **154 testes passando em
5,57 s**. A run 08 registrou, antes da interrupção, 17 páginas, 292 componentes
de coverage, 300 observações OCR, 116 owners, 116 chamadas de tradução e 5
falhas isoladas de tradução. Como não produziu `performance_timing.json` final,
`project.json` nem export gate, seus 2.079,2168 s parciais não são tempo E2E.

A run 10 terminou em 4.086,4106 s e a run 11 em 3.138,1001 s; ambas são
tempo até falha. Na run 11, o residual físico da run 10 já foi convertido em
preservação local, e a execução avançou até a validação dos stages canônicos.
O cleanup agora é recomposto somente com commits ainda autoritativos e os
pixels de typeset são relidos do candidato persistido. Suíte focada atual:
**156 passed in 4,58 s**.

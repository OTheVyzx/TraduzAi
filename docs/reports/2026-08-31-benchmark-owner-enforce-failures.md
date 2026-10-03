# Benchmark final de 10 capítulos — owner graph `enforce`

## Conclusão executiva

O benchmark final executou dez capítulos diferentes de nove obras, totalizando
207 páginas de entrada. Nenhuma execução alcançou `project.json`,
`performance_timing.json` ou `qa.export_gate`. Portanto:

- **tempo médio observado até término/falha:** 587,347 s (**9m47s**);
- **tempo médio do processo completo:** **não mensurável nesta amostra**;
- **aprovação pelo export gate:** 0/10;
- **resultado da execução do plano:** **NO_GO end-to-end**.

Um `exit_code=0` do harness apenas significa que o harness terminou as dez
tentativas. Cada processo do pipeline retornou `exit_code=1`.

## Escopo e método

- Fonte somente leitura:
  `D:\Mihon pra pc\downloads\mangas\Manga Ball (EN)`.
- Artefatos:
  `N:\TraduzAI\.codex-tmp\owner_enforce_timing_20260831\benchmark_final_v3`.
- Resumo canônico: `benchmark_summary.json` dentro da raiz acima.
- Configuração: `mode=auto`, `export_mode=with_warnings`,
  `owner_graph_mode=enforce`, `style_copy_mode=shadow`, `debug=false`.
- Execução serial, usando a mesma máquina/GPU e um diretório novo por capítulo.
- `wall_sec` foi medido externamente do início ao término do processo.
- O P90 usa interpolação linear no índice `(n - 1) * 0,90`.
- Runs diagnósticas e calibrações anteriores foram excluídas das estatísticas.

## Tempos

| Métrica | Segundos | Tempo aproximado |
| --- | ---: | ---: |
| Soma das dez tentativas | 5.873,472 | 1h37m53s |
| Média até término/falha | 587,347 | 9m47s |
| Mediana | 543,985 | 9m04s |
| Mínimo | 242,393 | 4m02s |
| Máximo | 1.218,398 | 20m18s |
| P90 interpolado | 737,486 | 12m17s |

Não há valores de `pipeline_total_sec`, overhead, páginas/minuto completo nem
estatísticas separadas de `PASS`, `REVIEW` e `BLOCK`, porque nenhuma run
produziu o artefato de timing ou chegou ao gate. Dividir o tempo até falha pelo
número total de páginas criaria uma taxa enganosa e não é reportado como
throughput.

## Resultado e causa por run

| # | Obra / capítulo | Páginas | `wall_sec` | Falha terminal | Por que terminou com erro |
| ---: | --- | ---: | ---: | --- | --- |
| 1 | Grand Finale 11 | 18 | 505,120 | `CoverageInvariantError`, `region_p007_009_c9cee96c73` | Um componente marcado como material não recebeu observação OCR após as tentativas de recuperação. Outros componentes primários/áreas de crédito também ficaram pendentes. A causa imediata é comprovada; a classe visual exata do primeiro componente não foi persistida em `debug=false`. |
| 2 | Grand Finale 24 | 18 | 582,850 | `TranslationValidationExhausted` | O capítulo atravessou o antigo bloqueio de cobertura e chegou à tradução, mas todas as tentativas aceitas pelo tradutor foram rejeitadas pelo contrato de validação. O resumo terminal não serializa o motivo individual de cada tentativa, então não é possível afirmar qual regra sem nova instrumentação. |
| 3 | I Was Immediately Mistaken for a Monster Genius Actor 79 | 18 | 1.218,398 | `TranslationValidationExhausted` | A run atravessou as antigas falhas de lifecycle, cardinalidade, hash e máscara; depois esgotou as tentativas de tradução. Assim como na run 2, o motivo de cada rejeição não foi publicado pelo runner final. |
| 4 | Martial Wild West 113 | 17 | 684,051 | `CoverageInvariantError`, `region_p002_006_46160844c5` | Várias regiões pequenas vistas apenas pelo detector primário foram consideradas materiais, mas o OCR não as confirmou. O padrão indica baixa precisão/classificação insuficiente na fronteira detector→cobertura. |
| 5 | Myriad Realms Gatekeeper 130 | 14 | 242,393 | `CoverageInvariantError`, `region_p001_011_24060bb4f3` | O primeiro blocker é uma região estreita detectada apenas pelo primário e sem OCR; casos semelhantes apareceram nas páginas 3 e 12. A política conservadora não tinha corroboradores suficientes para ignorá-los. |
| 6 | Nebula's Civilization 124 | 26 | 624,642 | `CoverageInvariantError`, `region_p001_008_cd5e049887` | O blocker terminal mede 12×8 px e foi visto apenas pelo primário. A mesma run também encontrou uma faixa larga no rodapé e um fragmento próximo a `SURASCANS.COM`, mostrando mistura de ruído minúsculo e arte/crédito sem política suficiente. |
| 7 | Primal Hunter 112 | 18 | 501,568 | `CoverageInvariantError`, `region_p002_005_24198fbe14` | Uma região larga de rodapé, vinda do detector negativo, permaneceu material sem OCR. A classificação de faixa/rodapé não encerrou o componente como não traduzível. |
| 8 | Reincarnated Murim Lord 113 | 40 | 477,421 | `CoverageInvariantError`, `region_p003_004_9b1dc344ef` | O capítulo revelou vários componentes estreitos vistos apenas pelo primário e uma região grande próxima a `VORTEXSCANS.COM`. É um padrão sistêmico de detector/política, não um único OCR ausente. |
| 9 | Rise of the Devourer 40 | 20 | 588,787 | `CoverageInvariantError: coverage has pending recovery requests` | O builder foi chamado com ao menos um request de recuperação ainda não terminal. O `decision_trace.jsonl` comprova recuperação de texto rotacionado na página 12 e OCR posterior, mas `debug=false` não publicou o ID/estado do request que ficou pendente. A causa imediata é lifecycle incompleto da recuperação; o produtor exato precisa ser instrumentado. |
| 10 | The Ten Thousand Clans Invasion: Guardian of the Rear 16 | 18 | 448,242 | `CoverageInvariantError`, `region_p001_009_1ca3f76fa5` | Duas regiões grandes corroboradas por `glyph_scan` e detector primário aparecem em material promocional/scanlation; uma terceira região primária surge perto de `DIGHT-SCADS.COM`. A política atual cobre crédito de borda estreito, mas não esse layout promocional maior. |

### Contagem por classe terminal

- `PIPELINE_ERROR`: 10
  - componente material sem OCR: 7;
  - validação de tradução esgotada: 2;
  - request de recuperação pendente: 1.
- `PASS`: 0
- `REVIEW`: 0
- `BLOCK` pelo export gate: 0
- `INTERRUPTED`: 0

`PIPELINE_ERROR` não equivale a `BLOCK`: `BLOCK` exigiria que a execução
técnica chegasse ao gate e fosse recusada por ele.

## Correções aplicadas antes do benchmark final

1. Cobertura e blockers ganharam publicadores canônicos em modo debug antes do
   builder, e grafos inválidos ganharam um artefato de `enforce_failure`.
2. Rejeições de geometria/máscara no modo `enforce` deixaram de mutar o grafo
   autoritativo para estados legacy/review; agora geram QA derivado invisível,
   sem autoridade de escrita.
3. A materialização/commit passou a excluir somente registros explicitamente
   rejeitados e fail-closed, preservando a cardinalidade e os hashes dos owners
   autoritativos.
4. Os hashes de binding da tradução passaram a existir antes dos estágios
   visuais.
5. `UnsafeOwnerMaskError` e esgotamento da repair ladder passaram a rejeitar o
   owner de forma derivada em vez de derrubar o capítulo.
6. Foram adicionadas políticas conservadoras para arte de crédito de
   scanlation e SFX pequeno corroborado. Elas corrigiram o caso diagnosticado
   de Grand Finale 24, mas não cobrem todos os layouts residuais.

Evidência funcional da correção: Grand Finale 24 não repetiu o blocker antigo
`region_p001_006_bcf9ad589c`; Monster Genius Actor 79 não repetiu
`owner_state_legacy_in_enforce`, `review_terminal_forbidden_in_enforce`,
cardinalidade/hash incompatível ou crash de máscara. Ambos avançaram até a
tradução.

## Validação automatizada

- Suíte final focada: **251 passed in 9,42 s**.
- `git diff --check` nos arquivos de implementação/teste: sem erro; apenas
  avisos locais de futura conversão LF→CRLF.
- As fontes CBZ não foram modificadas.
- Nenhum commit, merge, push, stash, reset ou limpeza foi realizado.

## Lacunas e correções seguintes

1. Persistir em toda run, não apenas em `debug=true`, um resumo mínimo dos
   blockers de cobertura e dos requests de recuperação pendentes.
2. Serializar cada tentativa de tradução com `owner_id`, validador, razão de
   rejeição e hash do payload; hoje `TranslationValidationExhausted` perde a
   causa específica no terminal.
3. Separar políticas testadas para: ruído primário minúsculo, faixa de rodapé,
   página/arte promocional de scanlation e região grande próxima a créditos.
   Não usar uma regra genérica que possa suprimir texto real.
4. Reexecutar primeiro os nodeids e os capítulos que representam cada classe;
   só depois repetir o benchmark de dez capítulos.
5. Só calcular tempo completo quando houver `project.json`,
   `performance_timing.json` e `qa.export_gate` consistentes.

## Runs excluídas da média final

As runs diagnósticas de Grand Finale 24 e as quatro tentativas de Monster
Genius Actor 79, além dos benchmarks de calibração `benchmark_final` e
`benchmark_final_v2`, foram usadas para revelar e corrigir falhas sucessivas.
Elas não entram na média porque executaram revisões diferentes do código ou
foram interrompidas deliberadamente durante a calibração. O conjunto v3 foi
rodado sem alteração de código entre as dez amostras.

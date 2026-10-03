# Regressão de performance do pipeline — execução corrente

## Regra de medição

Tempo completo só é reportado quando o mesmo workdir contém
`performance_timing.json`, `project.json` e evidência do export gate. Falhas
anteriores ao gate são tempo até falha; interrupções com apenas
`performance_timing.partial.json` ficam fora das médias.

## Baseline histórico

O benchmark owner-enforce de 10 capítulos de 2026-08-31 processou 207 páginas,
mas nenhuma das 10 runs chegou ao export gate. Portanto, a média histórica de
587,347 s é **tempo até falha**, não duração média do processo completo.

## Medições da recuperação

| Caso | Estado | Wall/total registrado | Uso correto |
|---|---:|---:|---|
| probe de input inexistente | falha em `extract_source` | 3,3903 s | valida lifecycle do timing, não performance real |
| Grand Finale 24 run 02 | falha em tradução | 647,6799 s | tempo até falha |
| Grand Finale 24 run 03 | falha na promoção terminal | 3.045,3339 s | tempo até falha após processar as páginas |
| Grand Finale 24 run 08 | interrupção externa | 2.079,2168 s parcial | excluído de médias |
| Grand Finale 24 run 10 | falha em residual físico terminal | 4.086,4106 s | tempo até falha após 18 páginas |
| Grand Finale 24 run 11 | falha no alias de stage visual | 3.138,1001 s | tempo até falha após 18 páginas |

Ainda não existe amostra válida para média E2E completa. O documento será
fechado depois que a execução real produzir projeto e export gate e depois do
benchmark diversificado final.

## Sinais preliminares

Na run 08, os spans acumulados antes da interrupção apontaram custo dominante
em execução por owner e render final. Os 2.051 s não atribuídos não podem ser
interpretados literalmente como trabalho desconhecido porque há spans pais e
filhos sobrepostos e scopes ainda abertos no snapshot parcial. A análise final
usará apenas sidecars terminados e separará wall time de soma de spans.

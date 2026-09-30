# INTEGRACAO_V1 — inventário reproduzível de I0

## Base observada

- Seed Git comum: `57caa2e88b493ae8bd32d2181fd04d6a3dc06cfe`.
- Branch de integração: `codex/traduzai-v1-integration`.
- O checkout raiz `Troca_de_motores` tinha 51.090 remoções, 121 modificações e 155 entradas não rastreadas no primeiro inventário; ele foi preservado.
- O antigo `optimized-e2e-integration` tinha 53.766 alterações rastreadas e não continha os módulos `pipeline` no disco.
- O R007 operacional permanece somente leitura no worktree de qualidade. `project.json`: `27f45d968a7b5b8213ae46d25afd5901380f35f3d789dcc56dd040940266f329`; pacote: `35a853fa9e9fee02219fc0d3b8a27f068004cf38037bccc32deb8a4b8f6cfe47`.

## Runtime materializado

O launcher R007 misturava imports não rastreados de três checkouts. O closure mínimo foi copiado por arquivo com SHA-256 validado para `pipeline/consumer_fast/runtime`, enquanto os módulos canônicos continuam em `pipeline`. `BASE_SOURCES.json` registra os hashes publicados. Os caminhos absolutos para outros worktrees foram removidos; fontes continuam uma dependência local read-only fornecida explicitamente por argumento.

Não foram promovidos capítulos, ZIPs, PNG/WebP/CBZ, fontes binárias, pesos, respostas focais, traduções aceitas, caches OCR, pickles ou outputs publicados.

## Evidência inicial

- Baseline limpo do seed: 101 testes passaram e dois testes de publicação falharam; portanto o seed sozinho não é a base operacional.
- Closure R007 materializado: 31 testes focais passaram.
- Smoke de imports: 22 módulos do projeto resolveram exclusivamente dentro do novo worktree.
- Contratos I0: RED observado por ausência de `integration_v1`, seguido de testes verdes para revisão, evento, jobs, export fail-closed e paths locais.
- Contratos + closure + rerender sintético real: 42 testes passaram. Todos os nove exemplos publicados atravessam o adapter estrito; o controle rasteriza a mesma receita duas vezes e compara pixels por SHA-256.
- Owner/repair/publicação: 70 testes focais passaram após materializar a dependência `canonical_component_container_ids` que o `coverage.py` carregado exigia.
- O gate integrado exige `qa.export_gate.allowed=true`, `status=PASS`, zero issues/flags/blockers, `verified=true`, `completion_status=approved` e `output_review_state=approved`.

## Falhas herdadas observadas fora do gate I0

A suíte ampla do renderer não é verde na raiz nem no overlay operacional R007. Dez falhas iniciais da integração foram reproduzidas na origem pertinente (raiz ou `pipeline_overlay`), incluindo merge de lobo, roteamento de falso OCR, curva compartilhada e âncoras dark-connected. Elas não foram mascaradas nem usadas como PASS. Uma regressão distinta de cap de altura de linha, presente na raiz mas ausente do conjunto aprovado do overlay, foi corrigida neste worktree e seu teste focal passou. Os gates I0 usam as suítes focais publicadas; Renderer continua dono da convergência visual ampla.

Essas evidências destravam desenvolvimento paralelo, mas não aprovam o CH57 nem substituem I1–I6.

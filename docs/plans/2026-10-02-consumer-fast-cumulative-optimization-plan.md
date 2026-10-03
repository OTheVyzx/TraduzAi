# Plano cumulativo de otimização do Consumer Fast

**Data da análise:** 2026-10-02  
**Escopo desta entrega:** planejamento e prompt de implementação. Nenhum código, branch, processo ou artefato de execução foi alterado ou iniciado.

## Resumo executivo

As otimizações podem coexistir se forem conectadas por um scheduler page-native com admission control, identidade e estado isolados por página/owner, pools compartilhados e limites de memória por recurso. A solução não deve reativar o executor legado `overlap_context_release`: hoje `run_chapter` calcula `overlap_executor=False` em `owner_graph_mode="enforce"`. O novo scheduler deve agendar operações comprovadamente independentes sem retirar `enforce`, sem afrouxar contratos de owner e sem transformar a ordem de conclusão física em ordem de commit ou publicação.

O objetivo é cumulativo: preservar ConsumerFast-HY, qualidade, as correções independentes que estão em andamento e os experimentos existentes; adicionar detecção, OCR, preparação, cache, limpeza, render e scheduling como capacidades independentes. Quando duas capacidades precisam do mesmo recurso, o scheduler deve enfileirar/admitir trabalho com limites explícitos. Não deve desligar silenciosamente uma otimização anterior nem duplicar o modelo CUDA ou multiplicar pools por página.

Os alvos de **menos de 10 s por estágio** e **menos de 20 s por página** são metas para medições controladas, não promessas. Os números históricos de aproximadamente 20% em quatro páginas e de 620 s em 45 páginas são referências limitadas e não representam uma medição aprovada do `main` atual.

### Estado que condiciona a implementação

- O checkout observado é `main`, HEAD `f71e213c7`, com alterações locais extensas e alterações em arquivos do Consumer Fast, `run.py`, coverage, renderer e outros módulos. O trabalho ativo `01a0fdd6-427d-7756-9153-cfb65c8700c8` está corrigindo owner coverage/GOTWHAT da parte composta 1–8. A implementação cumulativa deve começar sobre o estado final desse trabalho, sem editar, mover, limpar, descartar, substituir ou duplicar suas mudanças.
- A parte 1 em avaliação é a imagem composta real de **690 × 18.610**, baseada nas originais 1–8; o resultado citado reportou **91,14% de sobreposição GOTWHAT/falha** e a correção está em andamento. Isso não prova que o problema seja “seis workers” ou cortes adaptativos. A investigação deve manter fixos os grupos, os workers e as entradas enquanto valida a correção, salvo evidência nova e uma autorização específica.
- O caminho direto atual do Consumer Fast chama `run_chapter` com `owner_graph_mode="enforce"`. O executor legado de overlap entre bandas é excluído nesse modo. Essa restrição de arquitetura precisa ser contornada com agendamento compatível com page-native ownership, não removida por flag.
- Cache de visão, Mayo, fastfill, OCR crop-first e OCR-region aparecem em código experimental/local e têm flags default OFF. A existência do código ou da flag não prova que tenha sido chamado numa run. O renderer e o código do executor têm mudanças locais pendentes; suas métricas precisam ser identificadas pela revisão e contadores da configuração concreta.
- As referências históricas são: multipágina, quatro páginas, wall 791,139 → 633,117 s (−19,974%) com RSS 12,19 → 15,76 GiB e RAM livre mínima 5,08 → 2,12 GiB; Consumer Fast-HY, quatro páginas, 261,607 s; Consumer Fast, 45 páginas, 620,244 s com QA terminal OFF e `verified=false`; E2E experimental do capítulo 24, 3.407,74 s, com 701,68 s de render. Nenhuma delas é um benchmark da implementação cumulativa atual.

## Inventário para integração

Os estados abaixo são estados observados no checkout/artefatos, não aprovação para ligar defaults. Na implementação, cada feature precisa de quatro estados separados: **presente no código**, **elegível/configurada**, **chamada com trabalho real** e **resultado aceito/equivalente**.

| Capacidade | Estado observado e pontos de entrada | Tratamento no plano |
|---|---|---|
| ConsumerFast HY local | Runner valida endpoint de loopback; `physical_executor` associa `hy_mt2_gguf_local` e o `provider_adapter` usa tentativas por owner. Há prova histórica HY de quatro páginas, não prova do HEAD atual. | Preservar HY como backend selecionado no perfil aprovado; não substituir por Google/Ollama. Medir lifecycle, tentativas, retries, validação e custo por owner. Só considerar batch local se o endpoint e os contratos demonstrarem suporte real; batching Google não é dependência. |
| Mayo e deduplicação de candidatos | `strip.experimental_mayo` requer `TRADUZAI_EXPERIMENTAL_MAYO=1` (default `0`); `detect_balloons` consulta esse módulo. Mayo propõe geometria candidata e deduplica tiles; modelo/dispositivo e chamadas precisam ser registrados. | Manter desligado até A/B controlado de recall/precisão e tempo. Separar dedupe de tiles, NMS de caixas e dedupe OCR por identidade/trace. Nunca remover texto/owner por igualdade textual sem regra explícita e teste de cobertura. |
| Owner fastfill | `inpainter.experimental_fastfill` requer `TRADUZAI_EXPERIMENTAL_OWNER_FASTFILL=1` (default `0`); seletor restrito a interiores bem amostrados. | Manter limpeza AOT/qualidade como fallback. Contar elegíveis, selecionados, pixels, razões de recusa e saídas; integrar só casos aprovados por máscara/owner e equivalência visual. |
| OCR crop-first | `TRADUZAI_OCR_CROP_FIRST` default `0`; ramificações em `vision_stack/ocr.py` e `ownership/coverage.py`. | Tratar como escolha de geometria/entrada OCR, não como pool. Validar associação de cada crop ao page hash, bbox e owner/trace; comparar cobertura e identidade. |
| OCR por região e ROI agrupada | `ownership.experimental_ocr_region` usa `TRADUZAI_EXPERIMENTAL_OCR_REGION=0`; `vision_stack/grouped_crop.py` planeja grupos limitados, mas presença no preflight não prova conexão no caminho físico. | Integrar depois do baseline de coverage. Uma chamada agrupada deve produzir tentativa identificável por região, ordem recuperável e prova de que não houve região omitida ou repetida. Recusar grupos fora do orçamento de pixels/ROI e cair para fluxo original. |
| Cache pre-OCR | `consumer_fast.vision_cache` é consultado/publicado pelo executor local, mas `TRADUZAI_EXPERIMENTAL_VISION_CACHE` default `0`; o arquivo está local/untracked. `vision_runtime` valida origem, hash/config visual, arrays e versão. | Feature independente e idempotente, útil em reprocessamento do mesmo material/configuração. Cache miss/invalidez sempre executa as fases reais. Proibir reaproveitamento de translation, inpaint ou render como resultado funcional da run. Contabilizar hit por página/estágio e razão de invalidação. |
| Cache de fonte, contornos, máscaras e ROI de render | `renderer.py` contém caches FT2Font/máscara/run; alterações locais recentes incluem otimização de renderer/contornos. A prova citada no chat ainda tinha validação em outra página pendente. | Manter render seguro e serial por enquanto. Promover por teste de output/recipe e medidas cold/warm. Chaves devem incluir hashes de fonte, tamanho, texto, transformação, estilo e efeitos relevantes. Não partilhar objetos FT2Font mutáveis em threads sem isolamento demonstrado. |
| Compactação de mapas/buffers | `page_record`/stage cache externalizam estrutura e arrays; o experimento multipágina também liberou arrays após o último consumidor. | Reter dados até cobertura, inpaint, render, QA e persistência registrarem término/posse. Compactar mapas imutáveis com contrato e hash; medir bytes reais retidos/liberados e impedir cópia integral por cada worker. Não inferir economia linear a partir de um page root. |
| Pools persistentes, startup e discovery/owner workers | E2E histórico tinha adapters live de startup/sweep/render; outras propostas (discovery CPU, owner OCR workers, workers persistentes de preparação e cleanup) eram protótipos ou “aprovados, não conectados”. | Unificar por capacidade: no máximo um pool residente por domínio/recurso e lifecycle explícito. Nenhum pool dentro de cada página/owner e nenhum pool aninhado. Instrumentar pool criado/pronto/fechado, PID, tarefas, fila, erros e reutilização real. |
| Paralelismo entre páginas | Experimento histórico tinha três páginas residentes e commits/publicação ordenados. É um scheduler experimental separado do Consumer Fast atual. | Projetar novo coordenador page-native; não reativar `overlap_context_release` em enforce. Admissão deve limitar páginas in-flight, arrays, OCR/cleanup/render e dispositivos; o resultado é republicado em ordem canônica de página mesmo com conclusão fora de ordem. |
| strict GPU e prewarm | `TRADUZAI_REQUIRE_GPU_INPAINT` é um caminho estrito de teste que impede init CUDA concorrente com detecção e valida dispositivo CUDA; prewarm tem flag própria. | Um único inicializador CUDA por processo/modelo até instrumentação provar segurança diferente. Compartilhar handle/residência em vez de recarregar modelo. Em erro de device, interromper ou seguir fallback apenas pela política aprovada e registrar qual ocorreu; nunca declarar strict GPU atendido com fallback CPU. |
| `REVIEW`, `BLOCK` e `PASS` | Executor atual calcula gate; runner trata blockers/estado de revisão. Skills e código mantêm revisão distinta de aprovação. | REVIEW/warnings permanecem visíveis e pendentes. `PASS` exige ausência dos blockers/warnings definidos pelo gate. `BLOCK` continua preview técnico bloqueado. Não converter `complete`, `success` ou publicação em aprovação. |

### Flags que precisam de recibo de execução

Resolver e registrar no início da run: todas as flags `TRADUZAI_*` consumidas por cada feature, valor resolvido, origem do valor (default/config/ambiente), versão/hash dos módulos, modelo e pesos, dispositivo real e condições do caller. Defaults observados importantes: `TRADUZAI_OCR_CROP_FIRST=0`; `TRADUZAI_EXPERIMENTAL_OCR_REGION=0`; `TRADUZAI_EXPERIMENTAL_VISION_CACHE=0`; `TRADUZAI_EXPERIMENTAL_MAYO=0`; `TRADUZAI_EXPERIMENTAL_OWNER_FASTFILL=0`. `TRADUZAI_EXPERIMENTAL_MAYO_DEVICE` usa CPU como default.

`TRADUZAI_STRIP_SCHEDULER_EXECUTOR=overlap_context_release` não prova overlap no Consumer Fast: `run.py` força `overlap_executor=False` em owner enforce. `TRADUZAI_REQUIRE_GPU_INPAINT` evita init CUDA em paralelo com detecção no caminho estrito; deve ser respeitado até teste de segurança específico. Um contador de “flag true” é apenas configuração; a run também deve provar `eligible > 0`, `submitted > 0`, `started > 0`, `completed > 0` e outputs aceitos por feature.

## Arquitetura cumulativa proposta

### Grafo de trabalho e contratos

1. **Fonte imutável:** extrair/validar páginas, gravar hash dos bytes e da identidade da página; IDs canônicos de run, execução, página, grupo, owner, tentativa e trace são atribuídos antes de agendar trabalho.
2. **Análise visual:** detector, Mayo opcional, dedupe/NMS e coverage OCR. Pode haver paralelismo CPU ou lotes de regiões quando cada request carrega página/hash/bbox/region_id; o coletor restaura a ordem e rejeita duplicatas ou lacunas.
3. **Grafo de owners page-native:** cada página tem contexto/arrays próprios. Construção de planos e captura de estilo só podem ser executadas em paralelo para entradas imutáveis e owners independentes; mutations, ownership transitions e receipts voltam ao coordenador com identidade validada.
4. **Tradução:** HY local continua por owner e bounded. A leitura do histórico/glossário recebe snapshot explícito; mudanças de contexto são aplicadas pelo coordenador na ordem de página/owner existente. Não paralelizar retries/commits dependentes. Google batch não é usado para preencher um caminho HY local.
5. **Limpeza e render:** cleanup concorre apenas com admission de memória/dispositivo, com uma fila global limitada; fallbacks e fontes de máscara permanecem auditáveis. Render consome seus buffers depois dos pré-requisitos; mantém o lock/serialidade do typesetter enquanto segurança concorrente não for provada. O scheduler pode avançar análise CPU de outra página enquanto um estágio serial utiliza GPU/render, sem executar dois renderizadores ou inicializadores CUDA acidentalmente.
6. **Review, export e publicação:** revisão/QA consume artifacts finais e receitas; não antecipa PASS. Workers retornam resultados ao owner/coordenador; publicação preserva ordem de página, IDs e hashes, independentemente da ordem física de término.

O novo scheduler controla a concorrência ao redor do engine canônico por tarefa, em vez de importar outro pipeline inteiro. Chamadas canônicas continuam chamando o owner correto. O scheduler recebe requests imutáveis, futures e receipts; o coordenador é o único que aplica mudanças de contexto, commits de owner, ledger e publicação. Cada page root/array tem dono, contagem de referências e ponto de liberação após o último consumidor. Ao atingir orçamento, a fila aguarda ou executa com capacidade menor já autorizada; não altera qualidade, não cria outro modelo e não salta etapas.

### Recursos e limites

- Pool único e residente por classe necessária (I/O, discovery CPU, OCR/preparação CPU, cleanup, render, QA); configuração global, não multiplicada por página. Sem pools aninhados. Provar reutilização através de PID e ciclo `CREATE/READY/CLOSE`, não pelo nome do pool.
- CUDA inicializada em seção crítica única por processo; uma instância residente de cada modelo. Admission controla separadamente contadores de consumers, bytes de VRAM observados/reservados e reserva explícita para outros processos. Começar com uma faixa GPU ativa e aumentá-la só com evidência da memória e sem OOM; 8 GB é capacidade total, não orçamento livre do app.
- Limite explícito de pages in-flight e de memória CPU por página/owner; backpressure por filas. Medir RSS da árvore, memória disponível, arrays/nbytes retidos, GPU/VRAM globais e por processo quando disponível. Coleta/compactação libera somente valores sem consumidor pendente.
- Estado de cada feature: `requested`, `admitted`, `started`, `completed`, `used_count`, `cache_hit/miss`, `fallback_reason`, `resource_wait`, `output_receipt`. Feature requested que não foi admitida ou chamada deve aparecer como tal; nenhuma desativação silenciosa.
- Metadados não determinísticos de wall/data/caminhos são diagnósticos separados dos contratos visuais. Não reescrever hashes/digests só para passar gates; documentar igualdade funcional e divergência de envelopes separadamente.

## Milestones, provas e critérios de decisão

### M0 — Proteger a correção ativa e congelar baseline

- Não iniciar implementação enquanto a correção da parte 1 estiver ativa nos mesmos arquivos. Registrar o ID da tarefa, worktree/branch, `HEAD`, `git status`, diffs antes/depois e hashes dos arquivos alterados. Esperar essa correção terminar ou obter worktree isolado da versão consolidada pelo usuário; não fazer stash/reset/clean/rebase/cherry-pick automático.
- Preservar a falha reportada da parte 1 e seu input real (com hashes), medidas **690 × 18.610**, originais 1–8, identificação dos seis grupos/partes que já existe e configuração real. Resolver a correção GOTWHAT sob as mesmas condições. Não atribuir o 91,14% a workers/cortes sem evidência. Não criar novos cortes adaptativos nem declarar “seis workers” como prova.
- Criar o baseline reproduzível com commit e dirty diff hash, manifest de módulos aprovados, hashes de inputs/modelos/config/deps, ambiente/driver/dispositivo, valores resolvidos de flags e contadores existentes. Guardar cópia imutável de métricas/receipts; congelar fonte/config durante cada comparação.

**Gate M0:** falha da parte 1 resolvida pelas regras de cobertura e identidade existentes, sem perder correções atuais; seis partes e mapeamento de originais definidos pelo artefato real. Se não passar, parar otimização de performance e relatar o bloqueio sem elevar concorrência.

### M1 — Mapa executável de rotas e testes atuais

- Traçar `chapter_runner → physical_executor → run_chapter → coverage/detect/OCR → owner page pipeline → HY provider → inpaint → renderer → QA/export → receipt/publication` para o HEAD consolidado e o diff local aceito.
- Para cada capacidade do inventário, apontar callable, caller, condição/flag, default, evidência do contador em runtime, módulos/hash no BASE_SOURCES e testes existentes. Classificar: integrado e chamado; integrado mas default OFF; chamado sem prova de trabalho; harness/probe; histórico fora do main; ausente.
- Descobrir os testes reais com a busca do repo. Não copiar testes/harness do worktree histórico. Não ampliar `BASE_SOURCES.json` nem outro manifest para incluir módulos não aprovados sem autorização explícita do usuário.

**Gate M1:** cada otimização que entrará na run tem dono, rota até Consumer Fast e teste/receipt definido. Itens sem rota permanecem fora; o plano deve nomeá-los, não fingir integração.

### M2 — Contratos e testes isolados por feature

Implementar em pequenas mudanças reversíveis, uma capacidade por vez, com flag independente e default preservado até aprovação:

1. IDs, dedupe, crop/ROI agrupada, Mayo e cache pre-OCR: cobertura exata por página/região, hash de pixels/config, no drop/duplicate, invalidação de cache para bytes/modelo/versão/flags diferentes.
2. ConsumerFast-HY local: request bounded, timeout e tentativas por owner; ordenação/contexto, retries, placeholders, idioma e binding intactos. Batch local somente como experimento separado, sem adaptar requisito Google.
3. Fastfill: mask safety e eligibility; refusals caem no engine de qualidade; pixel diff limitado e persistência de reason/counters.
4. Fontes/contornos/compact maps: cache key completa e consistência cold/warm; comparar masks, recipes e flatten; liberação só após referência de consumidor chegar a zero.
5. Scheduler e pools: exceção/cancelamento propaga; receipts chegam exatamente uma vez; recurso sempre liberado; controle de capacidade global; teste de uma página, várias páginas, buffers imutáveis e finalização fora de ordem.
6. CUDA/strictGPU e REVIEW: prova de uma inicialização/residência, dispositivo efetivo e admission; falha fecha com política correta. REVIEW conserva warnings e `needs_review`; somente critérios existentes do gate permitem PASS.

**Gate M2:** todos os testes focais e de contrato relevantes passam; baseline de page 1 é igual em pixels/owners/coverage/recipes quando a feature não deve mudar semântica; cache inválido executa fluxo real; nenhum worker/modelo vaza.

### M3 — Matriz A/B isolada e combinada

- Usar a mesma versão congelada, mesmas entradas reais e mesmos recursos; comparar A=baseline atual e B=uma feature. Rodar cold e warm em colunas separadas, com runs alternadas (ABBA/BAAB quando o custo permitir) e repetição suficiente para estimar ruído; conservar todas as tentativas, falhas e recibos.
- Fazer A/B pelo menos de: cache on/off; crop-first on/off; região agrupada on/off; Mayo/dedupe on/off; fastfill/fallback on/off; renderer cache/contorno on/off; compactação on/off; resident pools on/off; scheduler page-native janela 1/2/limitada; GPU consumidor 1/2 apenas se aprovado pelo budget. HY local continua presente nas comparações de Consumer Fast.
- Fazer combinações pareadas para features que compartilham entradas/recursos: (cache × Mayo/crop-first/region), (ROI × OCR concurrency), (pool preparation × page scheduler), (compactação × workers), (fastfill × cleanup admission), (renderer caches × serial renderer), (prewarm × strictGPU/CUDA init). Depois testar a combinação cumulativa elegível, sem explosão fatorial.
- Cada célula registra `requested/admitted/started/completed/used`, flags resolvidas e callsites; se a feature não executou trabalho, a célula não conta como B. Qualquer feature anterior que regredir além do ruído estimado reprova a combinação até causa observada e explicada por carga/contador. Não desligar feature para esconder regressão; testar causa e interação isoladamente.

**Gate M3:** combinação candidata mantém cobertura/outputs/QA e não regressa otimizações anteriores sem explicação reproduzível. Comparação de métricas usa soma exclusiva correta/interval union; tempos de workers concorrentes não são somados ao wall.

### M4 — Integração do scheduler cumulativo na parte composta e seis partes

- Integrar o scheduler como coordenador do fluxo ConsumerFast page-native, por interfaces pequenas. Não tirar `owner_graph_mode="enforce"`; não tornar o overlap legado um requisito; não importar ou sobrescrever diretórios inteiros do worktree histórico/pipeline legacy.
- Manter exatamente os seis grupos e a ligação aos originais definida pelos manifests existentes. O teste atual da parte 1 (composição real, 690 × 18.610, originais 1–8) é um checkpoint de correção, não evidência de concorrência. Reexecutar primeiro essa parte após correção active, sem alterar número de workers/cortes; integrar os outros grupos ao mesmo executor compartilhado apenas após aceite daquele checkpoint.
- Para seis partes: um coordenador e pools compartilhados; não executar seis runners completos em paralelo nem copiar imagem/modelo para cada parte. Admission usa pixels/bytes e dispositivos, não só contagem de tasks. Cada output registra `part_id`, `page_id`, `owner_id`, source hash e sequence; publicação recompõe ordem canônica e verifica cada original 1–8 exatamente uma vez.
- Partes podem terminar fora de ordem se e somente se suas dependências/context snapshots são independentes e commit/ledger/export estão serializados pela ordem contractual. Se o contexto HY depender do resultado anterior, snapshot e commit permanecem ordenados; a análise CPU seguinte pode sobrepor sem usar mutações futuras.

**Gate M4:** seis partes cobertas uma vez, sem lacuna/duplicata; mesma autoridade de owner, recipes e revisão; memória fica limitada pelo máximo in-flight admitido, sem crescimento linear em número de partes.

### M5 — Benchmark de capítulo completo e decisão de promoção

- Só medir capítulo completo quando M0–M4 passaram e o manifest contém o input aprovado; congelar source/config/model hashes e impedir execução concorrente pesada. Separar cold start, warm pages, QA/review, publicação e verificação pós-run.
- Aceitação de desempenho compara medianas/intervalos A/B no mesmo input e ambiente, reporta wall externo e exclusivo, union de intervalos, filas/IPC/startup/waits, CPU, RSS/RAM disponível, GPU/VRAM, throughput e contadores de pool/cache. Registrar também variação da composição de carga externa.
- Verificar por source hash, page/owner/trace coverage, OCR/translation bindings, export gate, render recipes e pixel hashes; comparar assets/pixels byte ou pixel-identicamente quando o contrato exige. Artefatos técnicos não aprovam conteúdo por si.
- Se houver warnings sem blockers, o resultado pode ser tecnicamente completo em `REVIEW/awaiting_review`; não atribuir `approved/PASS`. Se existir blocker, `BLOCK` continua explícito. Rodar E2E não converte warnings em aprovação.
- Metas `<10 s/estágio` e `<20 s/página` são relatadas como meta atingida/não atingida por intervalo; não compensar estágio lento com qualidade menor. Historical +20%, 620 s/45p QA OFF e 56m47s do capítulo 24 são apenas referências de contexto.

**Promover** só se qualidade/contracts passam, counters provam chamada, wall melhora de forma reproduzível, recursos ficam dentro dos limites acordados e a matriz não mostra regressão inexplicada. **Parar/rollback apenas a feature experimental culpada** se outputs divergirem, coverage cair, review/blocker for apagado, cache reutilizar fonte/config incorreta, worker/modelo vazar, memória ficar sem headroom, CUDA duplicar/inicializar concorrentemente sem aprovação ou a performance anterior regredir sem causa comprovada. Não reverter correções independentes nem a correção GOTWHAT para fazer rollback de uma feature de velocidade.

## Manifesto e artefatos a produzir na futura implementação

- `consumer_fast_optimization_manifest.json`: schema/version, allowlist explícita de módulos e hashes, feature ID, estado, caller/flag, defaults, custo CPU/RAM/GPU, dependências, testes, rollback knob e contador de ativação. Atualizar o manifest apenas para paths que já foram aprovados. Uma necessidade de adicionar path fora do allowlist exige pedir autorização explícita antes de editar o manifest.
- `baseline.json` e `run_environment.json`: commit e hashes dirty dos arquivos aprovados, source/page/group hashes, modelo/tokenizer/dicionário/fontes/deps, config, flags resolvidas, CPU/GPU/CUDA, VRAM livre/amostrada e pool config.
- `optimization_activation.jsonl`: uma linha por feature/run com requested/admitted/start/finish/used-count/fallback e hash do output/receipt.
- `consumer_metrics.json`/`consumer_resources.json`: wall/exclusive, overlap/union, queues, IPC, stage calls, memory/resource profile e lifecycle.
- `equivalence.json`: cobertura original 1–8 e seis partes, page/owner/trace cardinality, bytes/dtypes/strides, page outputs, masks, recipes, pixel/asset hashes, gate e diferenças explicadas.
- `AB_MATRIX.md` + resultados JSON; `FINAL_REPORT.md`: ganhos e regressões por estágio, reproduções, status `REVIEW/BLOCK/PASS`, limitações e decisão de promoção/rollback. Preservar relatórios antigos e não recalcular seus digests.

---

# Prompt integral pronto para Codex

```text
Tarefa: integrar cumulativamente as otimizações históricas aprovadas e as otimizações atuais do Consumer Fast HY local, sem substituir, desativar silenciosamente ou regredir as outras. Nesta etapa de trabalho você está autorizado a implementar somente depois de proteger o estado ativo descrito abaixo; siga os gates e pare para pedir autorização somente se precisar expandir um manifest para incluir path/módulo que não esteja no allowlist aprovado. Não copie worktree/pipeline legacy inteiro, não transforme esse trabalho em cherry-pick de outra branch e não desfaça correções independentes.

ANTES DE CODIFICAR

1. Leia N:\TraduzAI\AGENTS.md e as skills locais `mangatl-dev`, `traduzai-pipeline`, `traduzai-detect`, `traduzai-ocr`, `traduzai-inpaint`, `traduzai-translation` e `traduzai-typesetting`. Obedeça os owners desses módulos.
2. O checkout principal está sujo. A tarefa ativa `01a0fdd6-427d-7756-9153-cfb65c8700c8` corrige owner coverage/GOTWHAT na parte composta 1–8. Não edite os arquivos/diffs dessa tarefa, não compartilhe o mesmo worktree para integração concorrente, não faça stash/reset/clean/rebase e não duplique a parte que ela está preparando. Espere essa correção fechar ou use um worktree de integração criado a partir do estado consolidado explicitamente pelo usuário, preservando todo diff aprovado.
3. Freeze de entrada: salve HEAD, status e hash dos diffs aprovados; identifique hashes de input, config, fontes/modelos/dependências, CPU/GPU/driver/CUDA e flags efetivas. Preserve logs e failures anteriores. Não altere/calcule novamente digests históricos para passar gate.
4. O reporte ativo da parte 1 é a imagem composta real de 690x18610 a partir das originais 1–8, com falha de 91,14% GOTWHAT/overlap e correção em curso. Reproduza/valide a correção com entrada, partição, worker count, flags e modelo fixos. Esse número não prova causa por “seis workers” nem valida cortes adaptativos. Não ajuste workers ou partições no mesmo passo que corrige cobertura.
5. Primeiro produza o inventário de capacidades/call graph e manifest allowlist, com evidência por código e receipts. Se faltar evidência, marque não comprovado. Se precisar adicionar ao ConsumerFast BASE_SOURCES ou a outro manifest um path não aprovado, pare antes de editar esse manifest, liste o path, por que é necessário e impacto, e peça autorização explícita. Não busque/compre/instale modelos e não faça upload de artefatos.

OBJETIVO ARQUITETURAL

Implemente uma camada de scheduling/admission page-native compatível com `owner_graph_mode="enforce"`. Hoje `pipeline/strip/run.py` desliga `overlap_executor` nesse modo; não reative nem contorne esse guard, não retire enforce e não use `overlap_context_release` como compatibilidade. O scheduler novo coordena tarefas por dependência, mantendo IDs, owner graph, autoridade de escrita, context snapshot, translation binding, receipts e publicação íntegros.

Fluxo obrigatório: origem/hash imutável → detecção/Mayo/dedupe → coverage/OCR por página/região → owner graph page-native → tradução HY local por owner com contexto/ordem → cleanup/inpaint → typesetting/render → REVIEW/QA → persistência/manifest/publicação ordenada. Uma página pode ter seus resultados físicos prontos fora de ordem; commits de contexto, owner ledger e publicação retornam à ordem canônica e aos IDs originais. Nunca perder ou duplicar owner/page/region/translation attempt. Buffers enviados a workers são imutáveis ou têm ownership exclusivo, cópia contabilizada e lifetime explícito.

Sobreponha somente trabalho cuja independência seja comprovada: análises CPU e preparação de requests podem avançar enquanto uma fase serial usa outro recurso; páginas podem ficar in-flight em janela limitada se seus snapshots são isolados. Tradução/context updates, mutações de grafo/owner, commit, arrays sem isolamento e publicação não podem concorrer em estado compartilhado sem contrato de ordenação. Não paralelize renderer/FT2Font mutável até segurança demonstrada; reutilização de cache não torna rasterização thread-safe.

Use pools residentes globais únicos por capacidade, compartilhados por todas as páginas/partes; nunca multiplique pools/modelos por página, owner ou etapa aninhada. Preserve uma única instância residente de cada modelo. Serializar inicialização CUDA e evitar concorrência com detector/inpaint até haver prova específica. Admission control precisa limitar páginas in-flight, bytes de arrays/RSS, workers CPU, OCR e consumers CUDA/VRAM; no hardware observado a GPU tem 8 GB totais, que não equivalem a 8 GB livres. Com pressão, esperar com contador ou reduzir para um perfil menor previamente testado; nunca baixar qualidade, omitir stage ou fingir que feature foi executada.

INVENTÁRIO CUMULATIVO

Preserve ConsumerFast-HY local (`consumer_fast/chapter_runner.py`, `physical_executor.py`, `provider_adapter.py`, `translator/hy_mt2_local.py`) como backend local selecionado e bounded. Não imponha Google batching quando HY local estiver em uso. Um batch do endpoint local é apenas hipótese: requer prova de capacidade, identidade, ordem, retry/validator e A/B independente; não substitui uma chamada que os contratos exigem por owner.

Investigue e integre, cada qual atrás de capacidade/flag própria, sem ligar defaults existentes às cegas:
- inicialização e reutilização de worker/modelo já comprovadas em probes;
- detector CPU/auxiliary overlap e Mayo candidatos/dedupe, com NMS e coverage conservadas;
- OCR crop-first, owner OCR e ROI agrupada com page/pixel hash, bboxes, region IDs e completude;
- CPU discovery e preparação owner/style/font com cache de chave completa;
- Fastfill owner em casos seguros, AOT/quality como fallback explícito;
- cache pre-OCR exato por source hash, modelo, versão e configuração visual;
- cache/contorno/ROI/font glyphs do renderer; manter render seguro/serial até teste de paralelismo;
- compact maps/arrays somente após último consumer, sem retenção/copias por worker;
- pools residentes globais de OCR/discovery/preparation/cleanup/render com admission, sem aninhamento;
- scheduler entre páginas/partes independente do executor legacy;
- strict GPU, lifecycle, receipts e gate REVIEW/strict sem falsa aprovação.

Os defaults observados `TRADUZAI_EXPERIMENTAL_MAYO`, `TRADUZAI_EXPERIMENTAL_OWNER_FASTFILL`, `TRADUZAI_EXPERIMENTAL_VISION_CACHE`, `TRADUZAI_EXPERIMENTAL_OCR_REGION` e `TRADUZAI_OCR_CROP_FIRST` são OFF. Preserve defaults até gate de promoção. `TRADUZAI_STRIP_SCHEDULER_EXECUTOR=overlap_context_release` não ativa overlap no Consumer Fast enforce. `TRADUZAI_REQUIRE_GPU_INPAINT` é caminho estrito/teste para serializar init CUDA com detect; não declarar GPU estrita se ocorreu CPU fallback. Flags configuradas e trabalho efetivamente chamado são fatos distintos: registrar por feature requested/admitted/started/completed/used_count, páginas/owners/requests, cache hit/miss, output receipt e fallback reason. Se flag ON e chamada zero, o relatório precisa dizer por que e a célula experimental não é considerada executada.

PLANO/MILESTONES DE IMPLEMENTAÇÃO

M0 — Proteção e baseline. Não tocar o worktree ativo de GOTWHAT. Depois da correção consolidada, capturar HEAD/diff/input/config/model hashes e flags. Revalidar a parte 1 composta 690x18610/originais1–8 sem alterar workers ou cortes. Aceitação: coverage passa pelo contrato existente, input original preservado, IDs sem duplicação/lacuna. Se falhar, parar scheduling/performance e reportar causa.

M1 — Call graph/inventário. Para cada feature, identificar callable → caller → guard/flag/default → contador → output. Classificar: integrado+chamado; integrado, default OFF; flag ativa sem trabalho; harness/probe; histórico externo; ausente. Conferir hashes e BASE_SOURCES. Não importar arquivos de worktree histórico nem converter estado “approved_not_connected” em runtime ativo sem adapter e teste.

M2 — Interfaces/testes unitários e contratuais, um feature slice por vez. Descobrir nodeids reais existentes; adicionar testes para identity/order/exception/lifecycle/admission, coverage sem gaps/dupes, cache invalidation (source, model, visual config, version, arrays/pixel hash), crop/region bounds e ordem, owner binding/context, fastfill safe mask/fallback, fontes/máscaras/recipes iguais, compactação apenas após lifetime, CUDA init serial/device assert e REVIEW sem promoção silenciosa. Comparar page outputs, masks, recipes, arrays (shape/dtype/strides/writeability) e bytes/pixels conforme o contrato. Cache miss ou falha experimental executa o caminho canônico real. Rodar somente testes focais; não iniciar capítulo E2E antes dos gates.

M3 — Matriz de medição. A/B isolado por feature: cache; crop-first; grouped ROI; Mayo/dedupe; fastfill; renderer contour/font/cache; compact maps; resident pool; overlap do scheduler page-native com janela=1 e janela limitada; GPU consumers somente nos valores autorizados pelo budget. HY local permanece selecionado. Faça cold e warm separados, runs em ordem intercalada ABBA/BAAB quando possível e repetições para estimar ruído. Depois use pares onde há interação: cache×crop/Mayo/ROI; ROI×OCR pool; preparation pool×scheduler; compactação×workers; fastfill×cleanup admission; render cache×renderer serial; prewarm×strictGPU. Por fim uma combinação cumulativa das células elegíveis. Preserve os resultados negativos e contadores; feature que não processou trabalho não prova ganho. Use métricas exclusive/interval union; não some worker wall concorrente ao wall do capítulo.

M4 — Integração das seis partes. Use a partição e mapeamento originais existentes, sem adaptively recortar. A parte1 corrigida é gate de entrada; as seis partes compartilham um coordenador/pools e cache/config manifest. Não lançar seis Consumer Fast independentes em paralelo. Limitar in-flight por bytes/capacidade medida; identidade garante originais1–8 exatamente uma vez. Manter contexto HY e mutations ordenados; permitir completion física fora de ordem apenas com merge/ledger/publicação final ordenados. Aplicar compactação após receipts do último consumidor. Aceitação: cobertura/exatamente-uma-vez, hashes de fonte, owner/trace, recipes, output e estado REVIEW/gate íntegros; RSS não cresce linearmente com partes e VRAM não duplica modelo.

M5 — E2E completo e relatório, somente após M0–M4. Congelar manifest, modelo, flags, input e aparelho; impedir outra workload pesada. Medir cold startup, warm stage/page, wall externo, exclusive spans/union, queue/wait/IPC, chamadas/throughput, CPU, RSS/available RAM, arrays retidos, device/GPU/VRAM global e por processo se houver, render/translation/coverage/QA/publication. Verificar source/page/owner lineage, original pixels, OCR/translation receipts, render recipes, hashes dos outputs, blockers/warnings e gate. Saída real com warnings permanece REVIEW/awaiting_review; BLOCK permanece blocked; apenas critérios existentes e ausência de issues permitem PASS. “Completo” ou exit 0 não basta.

Métricas alvo são <10s/estágio e <20s/página, alvos medidos, não condições para relaxar qualidade. Referências históricas (quatro páginas −19,974% com RSS alto; quatro páginas HY 261,607s; 45 páginas 620,244s com QA OFF/verified=false) não são baseline atual nem aprovação.

GATES DE CONTINUAR/PARAR E ROLLBACK

Continuar feature a feature se testes contratuais passam, counters provam trabalho, outputs/coverage esperados permanecem iguais ou diferença autorizada está registrada, recursos respeitam limites e a melhoria supera o ruído A/B. Combinação só avança se interações estiverem medidas.

Parar aquela feature e registrar reprodução se houver owner/original perdido ou duplicado, divergência de identidade/hash, alteração de pixels/recipe fora da política, gate/review suprimido, cache stale, exceção/worker não fechado, alocação fora de orçamento, RAM crescente linear, CUDA duplicada/init insegura, OOM ou regressão de uma otimização anterior sem carga/contador explicativo. Rollback altera somente o switch/adapter da feature causadora; nunca reverte automaticamente fixes independentes ou a correção GOTWHAT. Não falsifique PASS editando hashes, removendo evidências, warnings ou publicação.

Manifest expansion: qualquer módulo novo necessário que não esteja no allowlist aprovado exige autorização explícita do usuário antes de editar `BASE_SOURCES.json` ou outro manifest. Não copie pipeline/worktree inteiro, não substitua ConsumerFast-HY por um harness, e não apresente protótipo ou feature habilitada sem contador como integrada.

ARTEFATOS FINAIS

Produzir manifest versionado/allowlist, baseline e environment receipts, activation JSONL por feature, métricas/resources JSON, matriz A/B, equivalence/coverage/recipe/pixel audit e relatório final com status de cada feature, ganhos/regressões, REVIEW/BLOCK/PASS, recursos, bloqueios e decisão de promoção. Preservar o diff de GOTWHAT e todas as tentativas/relatórios históricos. Não alterar digests antigos. Nenhum upload para Library.

Na resposta ao usuário, resumir por capacidade: integrada e chamada; ligada mas sem trabalho; protótipo; fora do escopo/aguardando aprovação. Citar evidência e configuração real. Não afirmar benchmark/aprovação sem os artefatos acima.
```

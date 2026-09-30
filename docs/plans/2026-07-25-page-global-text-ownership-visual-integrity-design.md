# Ownership textual page-global e integridade visual sistêmica

Data: 2026-07-25
Status: aprovado para planejamento de implementação

## Objetivo

Impedir, por contrato arquitetural, que as classes de erro observadas no pipeline visual reapareçam em outros bands, páginas, capítulos ou obras.

Os casos visuais conhecidos serão usados somente como regressões e sentinelas. Nenhuma regra de produção poderá depender de `band_id`, número de página, capítulo, obra, texto específico ou coordenada fixa de um caso observado.

## Problema estrutural

O pipeline atual detecta bandas e executa OCR por band antes de possuir uma identidade semântica global da página. A arbitragem posterior consegue remover alguns prefixos ou duplicatas, mas ocorre tarde demais para garantir que:

- todo texto-fonte foi descoberto;
- todas as linhas do mesmo corpo pertencem à mesma unidade semântica;
- somente um band pode limpar e renderizar essa unidade;
- a tradução recebeu o corpo completo;
- a máscara cobre todo o texto e somente o texto do owner;
- overlaps não restauram pixels ou apagam um typeset já correto;
- o QA avalia os pixels realmente exportados.

O design anterior preservava OCR band-first, adicionava arbitragem page-space depois e mantinha fora de escopo a cobertura textual/tradução. Isso melhora casos conhecidos, mas não estabelece invariantes suficientes para generalizar.

## Decisão arquitetural

Adotar um **control plane textual page-global**, mantendo bands exclusivamente como tiles de execução.

O owner nasce em coordenadas de página antes de tradução, inpaint ou typeset. OCRs de página, crop, linha, negativo, card e band tornam-se observações do owner. Um band pode carregar uma projeção local do owner, mas não pode criar uma segunda identidade, repartir o body nem decidir autonomamente o conteúdo traduzido.

Foram rejeitadas duas alternativas:

1. **Adicionar mais heurísticas ao fluxo band-first**: mudança menor, porém mantém a causa da recorrência e aumenta regras conflitantes.
2. **Remover bands completamente**: simplifica ownership, mas amplia custo de memória, latência e risco sem necessidade. Bands continuam úteis como tiles, desde que não sejam unidades semânticas.

## Invariantes globais

### Cobertura da fonte

1. Todo componente visual com evidência textual deve ter exatamente um destes estados:
   - associado a um owner traduzível;
   - preservado explicitamente como SFX/nome/crédito;
   - suprimido explicitamente como ruído, com evidência e motivo auditável;
   - `review_required`, bloqueando export.
2. Componente textual sem owner ou decisão explícita é `source_text_unowned` crítico.
3. `no_matching_project_layer` nunca é suficiente para liberar export quando há evidência visual de texto.

### Identidade e atomicidade

1. Um owner é identificado por `owner_id` estável em page-space.
2. Um componente-fonte pertence a no máximo um owner ativo.
3. Um owner traduzível possui exatamente um payload semântico de source e um payload traduzido.
4. Um balão comum possui um único owner/body.
5. Cards podem possuir owners por papel (`title`, `note`, `body`, `footer`), mas um parágrafo/body não pode ser dividido por linha, OCR ou band.
6. Quebras visuais de linha pertencem somente ao typesetter.

### Mutação visual

1. Somente o owner de execução pode produzir máscara, inpaint ou typeset.
2. Pixels fora da `action_mask` page-space do owner devem permanecer iguais ao original, salvo mutação pertencente a outro owner válido.
3. `action_mask ∩ protected_art_mask = 0` é obrigatório.
4. Inpaint acionável exige engine, máscara e pixels alterados autoritativos. Telemetria contraditória bloqueia.
5. Um owner não pode ser renderizado se o texto-fonte correspondente não foi limpo com segurança.

### Composição e export

1. A composição sempre começa da página original canônica em memória ou cache lossless.
2. Artefatos sob `debug/` nunca são fontes de produção.
3. Ordem de escrita: original → deltas de inpaint pertencentes ao owner → glyphs traduzidos.
4. Contexto inalterado de um tile nunca pode sobrescrever pixels pertencentes a outro owner.
5. O QA final roda depois do último writer e sobre os mesmos pixels exportados.

## Modelo de dados

### `SourceTextComponent`

Representa evidência visual de texto, independentemente de OCR:

```json
{
  "component_id": "src_p001_0007",
  "page_id": "page_001",
  "bbox_page": [120, 840, 510, 940],
  "polygon_page": [[120, 840], [510, 840], [510, 940], [120, 940]],
  "detectors": ["comic_text_detector", "glyph_component_scan"],
  "script_evidence": {"latin": 0.93},
  "coverage_state": "owned"
}
```

IDs são derivados da página e da ordenação espacial determinística dos componentes, nunca do texto reconhecido, obra ou band.

### `TextObservation`

Representa uma leitura candidata:

```json
{
  "observation_id": "obs_p001_0012",
  "provider": "paddle_full_page",
  "component_ids": ["src_p001_0007", "src_p001_0008"],
  "text": "SOURCE BODY...",
  "confidence": 0.91,
  "bbox_page": [120, 840, 510, 1020],
  "coverage_score": 1.0,
  "language_score": 0.98,
  "tile_projection": "tile_p001_0009"
}
```

`tile_projection` é apenas proveniência. A observação continua page-space.

### `TextOwner`

```json
{
  "owner_id": "own_p001_0004",
  "page_id": "page_001",
  "component_ids": ["src_p001_0007", "src_p001_0008"],
  "observation_ids": ["obs_p001_0012", "obs_p001_0013"],
  "selected_observation_ids": ["obs_p001_0012"],
  "semantic_role": "dialogue_body",
  "source_payload": "SOURCE BODY...",
  "translated_payload": "CORPO TRADUZIDO...",
  "state": "translated",
  "route_action": "translate_inpaint_render",
  "execution_tile_id": "tile_p001_0009"
}
```

### `OwnerProjection`

Adapta um owner a um tile sem alterar identidade ou payload:

```json
{
  "owner_id": "own_p001_0004",
  "tile_id": "tile_p001_0009",
  "role": "executor",
  "bbox_page": [120, 840, 510, 1020],
  "bbox_tile": [120, 80, 510, 260],
  "coordinate_transform": {"dx": 0, "dy": -760}
}
```

Projeções adicionais podem ser `context_only`; elas nunca produzem mutações.

### `OwnerGraph`

O grafo agrega components, observations, owners, projections, conflitos e resultados dos invariantes. Ele é serializável, versionado e armazenado no projeto para retomada e auditoria.

## Construção do grafo

### Descoberta page-global

Executar detecção normal e uma passagem independente de recall textual por página. A segunda passagem procura componentes text-like sem depender dos layers já aceitos. Componentes são unidos por geometria, contorno de balão/card e conectividade visual.

Essa etapa não traduz nem decide estilo. Ela responde apenas: onde há texto e quais componentes pertencem à mesma unidade visual.

### Agrupamento semântico

- Balão comum: todos os componentes internos formam um body owner.
- Balão conectado: lobes podem ser owners distintos somente quando a geometria e a ordem de leitura sustentarem falas distintas.
- Card/painel: componentes são agrupados por papel semântico; linhas adjacentes do mesmo parágrafo permanecem no mesmo body owner.
- Texto tabular: rows são preservadas estruturalmente, com números e labels associados, sem converter cada linha visual em tradução independente.
- SFX/créditos: recebem política explícita, não desaparecem do mapa de cobertura.

### Arbitration de OCR

Observações são pontuadas por:

1. cobertura dos `component_ids`;
2. consistência geométrica;
3. adequação ao idioma-fonte;
4. confiança OCR;
5. coerência lexical/contextual;
6. preservação de números, pontuação e nomes.

A seleção é monotônica em cobertura: uma observação nova não pode substituir outra se reduzir componentes cobertos, salvo decisão explícita de review. Observações rejeitadas continuam persistidas com motivo.

O payload do owner é montado antes da tradução. O tradutor nunca recebe fragments de linha independentes pertencentes ao mesmo body.

## Corte obrigatório entre controle e execução

`run_chapter` passa a operar em duas fases, sem intercalá-las:

1. **Control plane:** atribuir IDs de região antes de formar bands, coletar observações de todos os tiles sem mutar pixels, converter toda geometria para page-space, construir e validar o `OwnerGraph` e persistir o snapshot autoritativo.
2. **Execution plane:** traduzir uma vez por owner, selecionar um único tile executor, produzir mutações identificadas, compor em page-space e executar os contratos finais.

Nenhum band pode traduzir, inpaintar ou renderizar antes de o grafo da página estar válido. Merges band-scoped e reparos semânticos tardios ficam disponíveis somente para projetos legados sem grafo; quando `OwnerGraph` existe, esses caminhos não podem alterar identidade, payload nem decisão de rota.

## Bands como tiles de execução

Depois de construir e validar o grafo:

1. cada owner recebe exatamente um `execution_tile_id`, escolhido por cobertura e distância das bordas;
2. tiles sobrepostos recebem projeções `context_only`;
3. owner cortado pode solicitar uma expansão page-bounded ou execução page-space, sem criar novo owner;
4. OCR, tradução e payload não são repetidos durante expansão;
5. resultados locais voltam ao compositor acompanhados de `owner_id` e máscaras page-space.

O reconciliador cross-band atual vira uma camada de compatibilidade temporária em shadow mode e é removido quando o grafo assumir enforcement.

## Estado e tratamento de erros

Estados válidos:

```text
discovered
  -> ocr_ready
  -> translated
  -> mask_ready
  -> inpainted
  -> laid_out
  -> rendered
  -> verified
```

Qualquer falha leva a `review_required` com `failed_stage`, `reason`, evidência e artifact links. Não há rollback silencioso para texto-fonte nem render parcial.

Regras:

- sem cobertura completa: não traduzir parcialmente;
- sem máscara segura: não inpaintar nem renderizar;
- sem fit legível: manter owner bloqueado, sem restaurar inglês silenciosamente;
- conflito de owners: bloquear antes de mutação;
- telemetria pure contraditória: bloquear;
- falha do QA independente: bloquear export.

## Máscara e inpaint por owner

A máscara positiva nasce da união de glyph/line evidence dos componentes do owner. A máscara negativa protege:

- molduras e linhas longas;
- personagens, ícones e componentes saturados;
- SFX preservado;
- owners vizinhos;
- arte fora do interior confiável do balão/card.

`protected_art_mask` participa da decisão operacional e da validação do compositor; não pode existir apenas como overlay ou telemetria de debug. A `action_mask` autoritativa é persistida e referenciada pelo `owner_id`, para que rollback, copyback e QA nunca caiam silenciosamente para um bbox aproximado.

O inpaint retorna `OwnerMutation`:

```json
{
  "owner_id": "own_p001_0004",
  "action_mask_path": "...",
  "changed_mask_path": "...",
  "engine": "aot_or_lama",
  "mask_pixels": 4210,
  "changed_pixels": 3988,
  "changed_outside_owner_pixels": 0,
  "residual_score": 0.0
}
```

O compositor rejeita mutations sem identidade, com mudança fora da máscara ou sobre arte protegida.

## Layout e typeset

O typesetter recebe um owner completo e seu safe polygon:

- `translated_payload` é refluído como unidade;
- tamanho possui limites inferior e superior derivados da evidência-fonte e do container;
- centro é calculado pelo safe polygon, não pelo bbox OCR;
- cards usam hierarquia por papel semântico;
- o renderer devolve glyph mask e bbox page-space para o compositor e o QA.

Layout e renderer podem dividir um payload em sub-regiões visuais, mas a concatenação normalizada dos chunks deve reconstruir exatamente o payload do owner. Eles não podem fundir bodies, suprimir textos, trocar route, criar tradução ou inferir identidade com helpers de máscara.

Fragmentação de style-copy não pode criar owners. O perfil visual é um atributo separado do owner e não participa da identidade semântica.

## Compositor page-space

O compositor mantém um canvas original RGB e um mapa de propriedade por pixel.

Para cada owner:

1. validar `OwnerMutation` e ausência de conflito;
2. aplicar somente pixels do inpaint sob a action mask;
3. aplicar glyphs traduzidos por último;
4. registrar pixels escritos, conflitos e hash do resultado.

Overlaps de tiles deixam de ser resolvidos pela ordem de paste. A precedência é semântica e ligada ao owner. JPEGs e artefatos de debug não participam da composição.

## QA final independente

O QA roda depois do último writer e combina seis contratos.

Qualquer consistency guard que ainda altere pixels deve ser executado antes desse QA ou convertido em auditor read-only. Não pode existir writer, rerender, copyback ou restauração depois da captura dos pixels auditados.

O observer recebe os bytes finais já persistidos e executa detector/OCR fresh sem reutilizar bboxes aceitos, crops de debug ou flags dos próprios layers. O relatório guarda o SHA-256 desses bytes; relatório ausente, cobertura de páginas incompleta, observer indisponível ou hash divergente fecham o gate em `BLOCK`.

### `source_coverage_contract`

- todo `SourceTextComponent` tem estado final explícito;
- nenhum texto-fonte fica sem owner;
- preserved/suppressed possui política e evidência.

### `owner_graph_contract`

- zero owners: `translatable_owner_missing`;
- múltiplos owners com mesmo payload: `duplicate_visible_owner`;
- múltiplos owners com payload divergente: `split_owner_payload`;
- body com mais de um payload: `semantic_body_split`.

### `route_state_contract`

- somente owners `rendered` podem aparecer nos pixels finais;
- `review_required` não entra no plano de render;
- tradução sem render seguro é blocker explícito.

### `pixel_ownership_contract`

- diferença original/final fora das máscaras possuídas deve ser zero dentro da tolerância de codec;
- nenhum owner perde glyphs após composição;
- nenhum tile restaura contexto sobre outro owner;
- seams, flattening e dano a protected art bloqueiam.

### `final_language_contract`

Executar OCR/detector independente sobre os crops finais e comparar n-grams com o source do owner. Nomes, SFX e créditos só são permitidos por política explícita. Inglês-fonte residual ou payload parcial bloqueia export.

### `qa_integrity_contract`

- toda identidade usa `owner_id` e `trace_id` composto;
- todo row carrega `offenders[]`;
- contagens de summary e gate devem coincidir;
- flags não propagadas, joins cross-band ou auditorias circulares bloqueiam o próprio QA.

## Cópia de estilo separada

Este design não tenta resolver a fidelidade de fonte, stroke, glow ou gradiente. Ele apenas garante que style-copy não altere ownership nem fragmente conteúdo.

Depois que os contratos funcionais estiverem verdes, um design separado poderá definir:

- token visual-base por owner;
- variações por papel semântico de card;
- confiança mínima para preservar variação intra-frase;
- fallback uniforme quando a evidência visual for fraca.

## Telemetria e artefatos

Novos artefatos versionados:

```text
02_strip_detect/page_owner_components.jsonl
03_ocr/page_owner_observations.jsonl
04_text_normalization_router/page_owner_graph.json
06_mask_segmentation/owner_masks/<owner_id>/...
09_typeset/owner_render_plan.jsonl
10_copyback_reassemble/owner_composition.jsonl
11_qa_export_gate/owner_invariant_report.json
11_qa_export_gate/final_pixel_ocr.jsonl
```

Cada registro inclui `run_id`, `page_id`, `owner_id`, coordinate space e schema version. Debug continua observável, mas não autoritativo para produção.

## Migração

### Fase 1 — Shadow graph

- construir o grafo sem alterar pixels;
- comparar owners do grafo com os layers atuais;
- bloquear apenas corrupção do próprio grafo;
- coletar diferenças por categoria, não por obra/band.

### Fase 2 — Owner-enforced semantic path

- tradução recebe somente payloads de owner;
- masks e renderer consomem `OwnerProjection`;
- reconciliadores/merges legados permanecem apenas para comparação;
- composição continua atual até o owner composer estar validado.

### Fase 3 — Owner compositor e QA enforcement

- owner composer passa a ser a única origem dos pixels finais;
- QA independente passa a bloquear;
- reconciliadores e merges band-first deixam o caminho ativo.

### Fase 4 — Remoção de compatibilidade

- remover aliases locais ambíguos e joins por `text_id` simples;
- remover autoridade de artifacts debug;
- manter leitura de projetos antigos por adapter de schema, sem reintroduzir decisões band-first.

## Estratégia de testes generalizáveis

### Unitários e property tests determinísticos

Gerar geometrias com seeds fixos:

- mesmo owner observado em 1–4 tiles sobrepostos;
- fragments prefixo/sufixo e OCRs divergentes;
- textos repetidos em geometrias distintas;
- body com número variável de linhas;
- card com title/body/footer;
- ícone ou personagem tangenciando o texto;
- masks que tentam escapar do owner;
- ordem aleatória de tiles e observations.

Propriedades obrigatórias:

- resultado independente da ordem de bands/observations;
- exatamente um executor por owner;
- nenhum body split;
- nenhum pixel escrito fora do owner;
- mesmas entradas produzem grafo e composição determinísticos.

### Integração sintética

Fixtures devem representar categorias visuais, não capítulos específicos:

- balão branco, translúcido e burst;
- painel escuro e card colorido;
- texto sobre arte;
- painel tabular;
- owners cruzando overlap de tiles;
- texto-fonte deliberadamente omitido do OCR principal para testar recall independente.

### Regressão cruzada

Executar uma matriz de obras diferentes e exigir os mesmos invariantes. Os casos visuais já auditados entram como sentinelas de regressão, sem lógica especial de produção.

## Critérios de aceitação

- zero `SourceTextComponent` traduzível sem owner ou review blocker;
- zero owner duplicado ou body semanticamente dividido;
- uma tradução e um render por owner;
- zero `review_required` no plano final;
- zero mudança fora das máscaras possuídas;
- zero protected-art damage;
- zero inglês-fonte residual não permitido pelo contrato independente;
- zero mismatch entre glyphs pós-typeset e pixels exportados;
- summary, rows e export gate consistentes;
- resultado determinístico sob permutação de bands e observations;
- matriz cruzada aprovada sem regras por obra, capítulo, página ou band.

## Fora do escopo

- treinamento de novos modelos de OCR/detecção;
- troca do provedor de tradução;
- remoção completa do processamento por bands;
- fidelidade avançada de style-copy;
- correções condicionais para os casos auditados.

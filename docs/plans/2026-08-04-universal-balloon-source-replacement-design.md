# Universal Balloon Source Replacement Design

**Data:** 2026-08-04
**Status:** aprovado
**Escopo inicial:** remover todo texto-fonte em inglês de balões e materializar o payload PT-BR correspondente, independentemente de obra, capítulo, página ou band.

## Objetivo

O pipeline automático deve terminar cada região textual traduzível com duas provas simultâneas:

1. o texto-fonte em inglês foi removido dos pixels finais;
2. o payload PT-BR vinculado à mesma unidade semântica foi renderizado nos pixels finais.

Falhas de detecção, OCR, associação, container, máscara, inpaint, layout, render ou QA não podem encerrar o processamento preservando o inglês. Elas devem acionar uma estratégia automática de recuperação mais forte. A última estratégia funcional é reconstruir o interior do container e renderizar o PT-BR com um estilo-base seguro.

Não será criada lógica condicionada a obra, capítulo, página, band, frase ou coordenada de um caso conhecido. Os casos reais servem apenas como fixtures e sentinelas.

## Evidência do problema atual

O owner graph existente é conceitualmente correto, mas não é a única autoridade operacional:

- componentes visuais são descobertos em page-space, enquanto o OCR de owner ainda depende dos bands existentes;
- uma página com componentes e sem band recebe `observations=[]`;
- observações full-page corretas podem ficar sem componente e não produzir owner;
- o singleton de OCR mantém `_last_full_page_line_records` em estado mutável separado do retorno da inferência;
- aliases derivados da mesma inferência Paddle podem ser contados como providers independentes;
- execução page-space volta a ser armazenada em bands para depois ser recomposta por página;
- rollback atômico protege a arte, mas preserva o inglês e encerra o owner sem uma recuperação coordenada;
- o QA final reencontra parte do inglês somente depois da persistência e apenas produz issues/gate;
- paths `legacy`, `shadow` e `enforce`, além de reparos tardios, convivem na mesma orquestração.

O tamanho dos módulos amplifica o acoplamento: `run_chapter()` ocupa aproximadamente 987 linhas, `process_band()` aproximadamente 522 linhas, e os módulos centrais de strip/runtime acumulam dezenas de milhares de linhas. O defeito, porém, não é a contagem isolada; é a existência de side channels, modelos paralelos e joins implícitos por bbox, posição ou índice.

## Alternativas consideradas

### 1. Acrescentar retries ao fluxo atual

Adicionar resets, locks, novos thresholds e chamadas extras nos caminhos atuais teria menor custo imediato. Foi rejeitado como solução principal porque manteria a divisão de autoridade e aumentaria o número de interações implícitas.

### 2. Transação owner-first por página com recuperação progressiva — escolhida

Uma estrutura imutável acompanha a identidade do texto desde a página original até os pixels finais. Os serviços atuais de detecção, OCR, tradução, inpaint e typeset tornam-se adaptadores. Falhas retornam pedidos de reparo tipados, e um controlador de recuperação escala automaticamente até concluir.

Essa alternativa preserva os motores existentes, elimina o telefone sem fio e permite migração TDD incremental.

### 3. Reconstruir todos os interiores de balões desde o início

É o caminho mais simples para garantir remoção do inglês, mas degradaria desnecessariamente balões translúcidos, gradientes, cards e arte sob texto. Será usado apenas como fallback final para o owner que esgotou estratégias precisas.

## Autoridade única

O caminho automático `enforce` passa a possuir uma única autoridade de produção:

```text
CanonicalPage
  -> PageCoverageLedger
  -> OCRInvocationResult[]
  -> OwnerGraph
  -> TranslationBinding
  -> OwnerReplacementTransaction
  -> PageComposition
  -> FinalReplacementVerdict
```

Cada etapa recebe um snapshot e devolve outro. Nenhuma decisão semântica é lida de atributos `last_*`, arquivos de debug, ordem de listas ou estado mutável deixado no motor.

Bands continuam disponíveis como projeções/crops de desempenho. Elas não são condição para descobrir texto, criar owner, traduzir, executar um owner ou compor a página.

## Identidade e hashes

Todo registro funcional carrega:

- `run_id`;
- `page_id`;
- `page_source_sha256`;
- `component_id`, quando aplicável;
- `observation_id`, quando aplicável;
- `owner_id`, quando aplicável;
- `invocation_id` para a inferência real;
- `provider_family` para agrupar aliases derivados da mesma inferência;
- `payload_sha256` para source e target;
- `attempt_id` e `strategy` para recuperação.

Um registro com `page_id` ou `page_source_sha256` diferente da transação é rejeitado antes de entrar no ledger. Colisões de ID com payloads divergentes são erro de integridade; nunca vale a regra silenciosa “primeiro vence”.

## Modelo de dados

### `OCRInvocationResult`

Resultado imutável e atômico de uma chamada OCR:

```python
@dataclass(frozen=True)
class OCRInvocationResult:
    run_id: str
    page_id: str
    page_source_sha256: str
    invocation_id: str
    provider_family: str
    variant_id: str
    blocks: tuple[OCRBlock, ...]
    observations: tuple[TextObservation, ...]
    full_page_lines: tuple[TextObservation, ...]
    attempts: tuple[OCRAttempt, ...]
    diagnostics: OCRDiagnostics
```

O modelo Paddle pode continuar cacheado no singleton. Resultados de requisição não podem permanecer no singleton. `_last_full_page_line_records`, estatísticas `last_*` e snapshots semelhantes deixam de ser APIs de produção.

### `PageCoverageLedger`

Mantém todo componente material até uma conclusão explícita:

```python
@dataclass(frozen=True)
class CoverageEntry:
    component_id: str
    page_id: str
    container_id: str | None
    bbox_page: BBox
    polygon_page: Polygon
    materiality: str
    ocr_attempt_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    owner_id: str | None
    semantic_role: str | None
    state: str
```

Estados permitidos:

```text
discovered
  -> challenged
  -> observed
  -> owned
  -> target_ready
  -> cleaned
  -> rendered
  -> verified
```

Estados de reparo são transitórios: `needs_ocr_repair`, `needs_owner_repair`, `needs_cleanup_repair` e `needs_target_rerender`. Eles não são resultados finais entregáveis.

Terminais funcionais:

- `verified`: texto-fonte removido e PT-BR materializado;
- `explicit_non_dialogue_preserve`: somente nome, SFX, crédito ou marca fora de diálogo, com política auditável.

Um componente material dentro de balão não pode terminar como `suppress`, `review_required` ou preservação genérica.

### `TranslationBinding`

Vincula semanticamente a tradução:

```python
@dataclass(frozen=True)
class TranslationBinding:
    page_id: str
    owner_id: str
    source_payload: str
    source_payload_sha256: str
    target_payload: str
    target_payload_sha256: str
    target_locale: str
    backend_attempts: tuple[str, ...]
    validation_status: str
```

O tradutor recebe um owner completo e devolve exatamente o mesmo `owner_id` e hash fonte. Fallback por índice ou posição é proibido.

### `OwnerReplacementTransaction`

Registra cleanup e render como uma única transação:

```python
@dataclass(frozen=True)
class OwnerReplacementTransaction:
    page_id: str
    owner_id: str
    attempt_id: str
    strategy: str
    source_mask_sha256: str
    action_mask_sha256: str
    changed_mask_sha256: str
    target_glyph_mask_sha256: str
    target_payload_sha256: str
    source_removed: bool
    target_materialized: bool
```

Não existe commit contendo somente inpaint ou somente glyphs. A composição aceita a transação apenas quando os dois lados pertencem ao mesmo owner e à mesma cadeia de hashes.

## Fluxo canônico

### 1. Página original canônica

Cada página é carregada lossless e recebe `page_source_sha256`. Toda recuperação recomeça desses pixels, nunca do preview, JPEG, crop debug ou resultado parcialmente alterado.

### 2. Cobertura page-global

O pipeline reúne antes do owner graph:

- detector de texto/balões;
- glyph component scan;
- detector negativo/escuro;
- UI/card/container evidence;
- OCR full-page realmente independente;
- OCRs ancorados por componente;
- observações recuperadas de crops.

OCR full-page roda mesmo quando `blocks=[]`. Todo componente material recebe pelo menos uma tentativa OCR. `no_ocr_evidence_non_text` só pode ser produzido depois de tentativas negativas explícitas e evidência visual de não texto.

Uma linha OCR material sem componente inicia associação de recuperação. O sistema tenta, em ordem:

1. interseção de polígonos;
2. interseção/containment de bbox;
3. proximidade e alinhamento dentro do mesmo container;
4. criação de componente ancorado em glyph evidence;
5. novo crop OCR localizado.

A observação não pode simplesmente desaparecer como `unassociated_observation`.

### 3. Container antes do owner

Componentes OCR-confirmados tentam obter container por:

1. detector primário;
2. white-balloon scan;
3. dark/negative scan;
4. UI/card scan;
5. contorno local ancorado no glyph;
6. safe region sintética limitada à vizinhança textual.

Essa recuperação ocorre antes da resolução final do owner. Um `semantic_container_missing` nunca encerra o conteúdo; ele escala para o próximo método.

### 4. Independência real de evidência

Consenso conta `invocation_id`/`provider_family`, não a string `provider`. `paddle_full_page`, `paddle_full_page_raw_line`, `visual_card_full_page_raw` e qualquer adapter do mesmo passe representam um voto.

Duplicar wrappers, reordenar bands ou reordenar observações não pode alterar o payload escolhido. Evidência de outra página ou hash é descartada antes do ranking.

### 5. Owner e corpo atômico

Um balão de diálogo possui um owner/body. Linhas visuais, bands, tentativas OCR e fragmentos não criam traduções separadas. Cards podem ter roles distintos, mas cada body permanece atômico.

O `OwnerGraph` é resolvido somente depois de fechar a cobertura da página. A tradução começa somente quando todos os componentes materiais de diálogo possuem owner e payload fonte.

### 6. Tradução PT-BR

O target passa por validação linguística antes de render:

- uma frase inglesa inalterada não é PT-BR válido;
- tokens ingleses materiais exigem nova tentativa;
- números e placeholders devem ser semanticamente equivalentes;
- nomes próprios só podem permanecer iguais se estiverem no registro de entidades;
- palavras curtas dentro de balão não são presumidas SFX;
- locale configurado é `pt-BR`.

A escada de tradução usa backend primário, retry contextual e fallback local. O resultado mais forte que preserva o source completo é vinculado ao owner. Não há fallback que simplesmente devolve o inglês como tradução válida.

### 7. Escada de substituição visual

Cada owner começa pela estratégia de maior fidelidade e escala automaticamente:

#### R0 — `precise_glyph_inpaint`

- união das máscaras de glyph/line de toda evidência fonte material;
- proteção de bordas, personagens, ícones e owners vizinhos;
- inpaint configurado;
- render PT-BR no safe polygon.

#### R1 — `expanded_source_support`

- amplia somente a evidência positiva da fonte;
- inclui polígonos corroborados omitidos;
- troca a variante de inpaint;
- recompõe a partir da página original.

#### R2 — `text_region_rebuild`

- reconstrói a região textual completa dentro do container;
- usa contexto ao redor do texto e respeita a borda do balão;
- renderiza novamente o mesmo target.

#### R3 — `container_interior_rebuild`

- último recurso obrigatório;
- reconstrói todo o interior seguro do balão/card;
- para fundo uniforme, usa estimação robusta do interior;
- para fundo translúcido/complexo, usa inpaint page-space com proteção da borda;
- se ainda houver suporte-fonte residual, aplica preenchimento interior determinístico limitado ao container;
- renderiza PT-BR com estilo-base seguro.

R3 prioriza a remoção do inglês e a presença do PT-BR sobre fidelidade decorativa. Não pode apagar pixels fora do container nem bordas protegidas.

### 8. Render funcional independente do style-copy

O contrato funcional usa uma fonte-base vetorial/configurada e antialiasing normal. Style-copy só é chamado depois que owner, source, target, mask e safe region estão estáveis.

Style-copy pode fornecer fonte, cor, gradiente, contorno ou sombra. Ele não pode:

- alterar payload fonte ou target;
- criar/remover/splitar owners;
- alterar action mask ou container;
- impedir o fallback funcional;
- mudar a condição `verified`.

Se style-copy falhar, o target é renderizado com o estilo-base; o inglês jamais é preservado por causa de estilo.

## QA como controlador de reparo

O QA final deixa de ser somente um produtor de bloqueio. Ele devolve `RepairRequest`s deduplicados:

```text
english_source_visible
source_region_without_owner
target_payload_missing
cleanup_incomplete
target_glyphs_missing
mixed_language_overlay
```

Roteamento:

- inglês associado a owner: reabre esse owner;
- inglês sem owner: cria componente/owner de recuperação na página original;
- PT-BR ausente: reaplica o glyph patch;
- cleanup incompleto: avança R0 → R1 → R2 → R3;
- entrada original PT+EN: limpa a união das camadas textuais e materializa um único target PT-BR;
- issue duplicada de detector/component challenge: uma única recuperação.

O primeiro ciclo usa estratégias precisas. O segundo ciclo força a estratégia imediatamente superior. Depois de esgotar precisão, R3 é obrigatório. O QA roda novamente sobre os bytes persistidos após cada reparo.

## Verificação terminal sem falso bloqueio

O sistema não depende apenas de OCR final, que pode ser inconclusivo. A prova terminal combina:

1. cobertura determinística do source support pela action/changed mask;
2. inexistência de pixels-fonte preservados dentro do suporte conhecido;
3. glyph mask do target aplicado na composição final;
4. hash do target e owner coerente em todas as etapas;
5. OCR final linguístico como observador independente;
6. desafio full-page para texto alfabético material sem owner.

Se o OCR final não conseguir ler o PT-BR, mas a cadeia de pixels comprovar remoção da fonte e materialização do glyph patch, isso não gera loop infinito. Se ele reconhecer inglês fonte, a recuperação continua.

Não existe conclusão funcional `BLOCK` por resíduo de conteúdo. Existe `repair_pending` até o owner ficar `verified`. Erros fatais de infraestrutura são falhas de processo, não uma aprovação de página com inglês.

## Políticas especiais

### Nomes próprios

Permanecem vinculados ao owner. Resultado idêntico só é permitido quando todos os tokens não traduzidos são entidades explícitas. Um nome não autoriza preservar o restante da frase inglesa.

### Números

Podem permanecer iguais quando a equivalência numérica é validada. Texto inglês ao redor do número continua traduzível.

### SFX

Preservação exige `semantic_role=sfx` e evidência visual/geométrica. Texto curto dentro de balão usa política de diálogo por padrão.

### Créditos, URLs e marcas

Só podem ser preservados por policy explícita e fora de containers traduzíveis. A policy carrega bbox, evidência e motivo.

### Entrada previamente contaminada

Uma página que já contém PT sobre inglês recebe `mixed_language_rebuild`. O sistema usa a página recebida como fonte canônica, detecta toda a união textual do container, remove ambas as camadas e renderiza uma única tradução PT-BR autoritativa. Não sobrepõe uma terceira camada.

### Página já em PT-BR

Um container integralmente PT-BR recebe `already_target_language` e pode ser preservado, desde que não haja tokens-fonte ingleses materiais.

## Simplificação do código

Será criado um coordenador page-first pequeno. Responsabilidades propostas:

```text
pipeline/ownership/ocr_contract.py       resultado OCR request-scoped
pipeline/ownership/coverage.py           ledger e recuperação de cobertura
pipeline/ownership/lifecycle.py          estados e invariantes
pipeline/ownership/execution.py          transação cleanup + render
pipeline/ownership/repair.py             escada R0-R3
pipeline/qa/language_residual.py          classificação/deduplicação linguística
pipeline/strip/page_pipeline.py           orquestração canônica por página
```

`run.py` agenda páginas, agrega telemetria e persiste resultados. `process_bands.py` fornece adapters temporários para crops/motores, sem decidir semântica. O caminho `enforce` não chama fallback legado; `legacy` e `shadow` ficam disponíveis somente para auditoria offline durante migração.

Reparos tardios que reidratam, renormalizam ou remontam payloads deixam de operar sobre projetos owner-verified. A remoção será incremental, somente após testes de caracterização e matriz visual verde.

## Observabilidade

Artefatos canônicos:

```text
01_input_extract/page_transactions.jsonl
02_strip_detect/page_coverage_ledger.jsonl
03_ocr/ocr_invocations.jsonl
03_ocr/page_owner_observations.jsonl
04_text_normalization_router/page_owner_graph.json
07_translation/translation_bindings.jsonl
08_inpaint/owner_replacement_attempts.jsonl
09_typeset/owner_target_materialization.jsonl
10_copyback_reassemble/page_composition.jsonl
11_qa_export_gate/repair_requests.jsonl
11_qa_export_gate/final_replacement_verdicts.jsonl
```

Cada linha aponta para os hashes do anterior. Artefatos de debug nunca são lidos como input funcional.

## Migração

### Fase 1 — caracterização e isolamento OCR

- reproduzir vazamento página 9 → 10 → 11;
- introduzir `OCRInvocationResult`;
- eliminar side channels `last_*` da produção;
- rejeitar identidade/hash cruzado.

### Fase 2 — cobertura page-global obrigatória

- criar ledger;
- OCR ancorado para componente sem band;
- materializar observação full-page sem componente;
- recuperar container antes do owner.

### Fase 3 — lifecycle e tradução vinculada

- estados canônicos;
- validação PT-BR;
- remover joins por índice;
- owner/body atômico.

### Fase 4 — transação e escada R0-R3

- commit cleanup + target;
- retry sempre a partir do original;
- container rebuild obrigatório quando precisão falhar;
- commits diretamente por página.

### Fase 5 — QA reparador

- classificador linguístico contextual;
- issues deduplicadas;
- retorno automático ao lifecycle;
- verificação por pixels + OCR.

### Fase 6 — remoção de autoridade legada

- `enforce` usa somente page pipeline;
- legacy/shadow fora da execução automática;
- remover reconstruções tardias de payload no owner-verified;
- style-copy testado como consumidor não autoritativo.

## Estratégia de testes

### Unitários

- duas chamadas OCR sequenciais e concorrentes nunca trocam linhas;
- aliases de uma inferência contam como um voto;
- observação de hash/página diferente é rejeitada;
- componente sem band recebe OCR;
- componente com zero tentativas não pode ser suprimido;
- observação sem componente é recuperada;
- body não é repartido;
- inglês inalterado não é tradução PT-BR válida;
- nome/número/SFX exigem policy explícita;
- cada estratégia R0-R3 preserva o container boundary;
- target e cleanup fazem commit atômico.

### Property tests

- resultado invariável à ordem de páginas, bands, aliases e observações;
- duplicar wrappers não altera seleção;
- exatamente um terminal por componente material;
- nenhum write fora do owner/container;
- recovery sempre avança de estratégia e nunca volta a pixels parcialmente mutados.

### Integração sintética

- balão sem detector/band;
- OCR full-page sem componente;
- dark, white, burst e translucent balloons;
- card/tabela;
- texto tocando borda/personagem;
- rollback de residual seguido de sucesso;
- forced container rebuild;
- entrada PT+EN sobreposta;
- inglês residual detectado somente no QA final;
- target glyph ausente após composição.

### Regressão real e cross-work

O capítulo 39 de Mitch Items é a regressão principal, com páginas 10, 11, 19, 21, 27, 28, 30, 34, 36 e 39. Páginas 30, 34 e 36 representam a categoria genérica `mixed_language_overlay`.

A matriz inclui pelo menos outras três obras, com cards, balões translúcidos, painéis escuros, texto sobre arte, burst e cross-tile. Nenhuma fixture autoriza regra por obra.

Executar duas passagens:

1. style-copy desativado, validando conteúdo puro;
2. style-copy ativado, comprovando que estilo não altera o contrato funcional.

### Validação visual

Inspecionar em escala nativa:

- original;
- coverage/OCR;
- inpaint;
- typeset;
- composição;
- bytes finais persistidos.

O manifest de inspeção conta artefatos únicos por hash. Uma imagem multi-tag pode ter várias categorias, mas uma única inspeção e uma única nota factual; o runner não pode multiplicá-la como provas independentes. Notas ou vereditos não são preenchidos automaticamente.

## Critérios de aceitação

- `english_dialogue_residual_count == 0`;
- `translatable_components_without_owner == 0`;
- `material_components_without_ocr_attempt == 0`;
- `owners_without_valid_pt_br == 0`;
- `owners_without_atomic_cleanup_render == 0`;
- `owners_without_target_materialization == 0`;
- `material_components_without_terminal_lifecycle == 0`;
- nenhum owner de diálogo termina preservado, suprimido ou em review;
- nenhum resultado OCR cru cruza página/hash;
- aliases correlacionados contam como uma inferência;
- nenhuma recuperação lê pixels parcialmente modificados;
- nenhuma mutação sai do container/protected boundary;
- final persisted bytes correspondem aos hashes auditados;
- capítulo 39 passa nas dez páginas sentinela;
- matriz cross-work e holdouts passam sem regras específicas;
- inspeção visual nativa retorna GO real;
- style-copy ligado ou desligado produz o mesmo conteúdo e a mesma cobertura.

## Fora do escopo inicial

- aprimorar fidelidade de fonte, gradiente, contorno, sombra ou glow;
- substituir o provedor principal de tradução;
- treinamento de novos modelos;
- alterar o Studio/editor;
- regras especiais para frases, páginas ou obras conhecidas.

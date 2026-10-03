# Owner Integrity V4.2 — Contract Lock

- Data da inspeção: 2026-08-11
- Checkout: `Troca_de_motores`
- HEAD: `57caa2e88b493ae8bd32d2181fd04d6a3dc06cfe`
- Escopo desta etapa: contrato e diagnóstico, sem alteração funcional

## 1. Resultado executivo

O contrato arquitetural necessário para implementar owner resolution, `cleanup_only`, REVIEW canônico, prova física terminal e publicação verificável está fechado neste documento. As correções abaixo substituem as ordens temporais ou nomes abstratos que ainda divergiam do checkout no texto V4.2.

A implementação funcional, entretanto, **não está autorizada a começar**. O corpus privado reprodutível exigido pelo próprio plano não está disponível nem congelado. Um diretório histórico do capítulo 39 e runs anteriores não substituem o contrato do corpus, as máscaras independentes nem a separação de holdout.

```text
CONTRACT_LOCK_STATUS:
NO_GO_MISSING_PRIVATE_CORPUS_CONTRACT
```

O anexo V4.2 usa uma vez o token alternativo `NO_GO_MISSING_REPRODUCIBLE_CORPUS`, mas seu contrato de saída e bloco final exigem `NO_GO_MISSING_PRIVATE_CORPUS_CONTRACT`. Este relatório adota o token final como único valor canônico; leitores não devem aceitar os dois como writers equivalentes.

Este NO-GO não invalida as decisões técnicas abaixo. Ele impede que essas decisões sejam implementadas e calibradas usando um baseline incompleto, mutável ou contaminado pelo mesmo código que será avaliado.

## 2. Limite desta entrega

Foi autorizado criar somente este relatório. Nesta etapa:

- nenhum arquivo funcional foi editado;
- nenhum teste funcional foi alterado;
- nenhuma configuração, fixture ou saída histórica foi promovida a baseline;
- nenhum commit foi criado;
- não foram usados `reset`, `checkout`, `stash`, `clean` nem worktree alternativo;
- as mudanças locais preexistentes continuam sendo propriedade do usuário.

O relatório corrige e torna executáveis as decisões do V4.2, mas não afirma que os novos tipos, codecs, gates ou providers já existem no runtime.

## 3. Estado factual do checkout

### 3.1 Git e alterações locais

Na inspeção:

- branch: `Troca_de_motores`;
- HEAD: `57caa2e88b493ae8bd32d2181fd04d6a3dc06cfe`;
- índice Git: nenhuma entrada staged;
- `git status --porcelain=v1 --untracked-files=normal`: 51.304 entradas;
- 90 entradas modificadas, 51.090 deletadas e 124 caminhos não rastreados na visualização normal;
- a expansão de arquivos não rastreados é muito maior, portanto nenhuma contagem resumida deve ser usada como autorização para limpeza.

O diff funcional central já existente abrange 14 arquivos e soma 1.625 inserções e 133 remoções, incluindo:

- `pipeline/main.py`;
- `pipeline/ownership/{container_evidence,coverage,discovery,evidence,execution,model,owner_builder,publication,reconcile,repair}.py`;
- `pipeline/qa/{export_gate,final_pixel_observer}.py`;
- `pipeline/strip/run.py`.

Esses arquivos estavam dirty antes deste relatório e não foram tocados por esta etapa.

### 3.2 Pré-requisitos encontrados e ausentes

| Requisito | Estado atual | Decisão |
|---|---:|---|
| `TRADUZAI_PRIVATE_FIXTURE_ROOT` | ausente | bloqueia corpus privado |
| `TRADUZAI_HOLDOUT_ROOT` | ausente | bloqueia holdout isolado |
| `pipeline/tests/fixtures/private_corpus_manifest_v1.json` | ausente | bloqueia seleção e replay canônicos |
| manifest de inputs do holdout | ausente | bloqueia avaliação cega |
| expectativas privadas do holdout | ausentes | bloqueia acceptance |
| máscaras independentes congeladas | ausentes | bloqueia avaliação de cleanup/inpaint |
| `TranslationReplayProvider` | não implementado | fase funcional futura |
| `TranslationReplayLedgerV1` | não implementado | fase funcional futura |
| flags de replay offline de tradução | ausentes | fase funcional futura |
| `N:\TraduzAI\temporario1\mch39` | presente | evidência histórica, não corpus contratado |

Uma run histórica `off6` contém 42 páginas e um manifest cujos hashes de imagens originais decodificadas coincidem com o source manifest daquela run. Isso é útil como evidência de diagnóstico, mas o manifest não possui splits, papéis de calibração/holdout, refs de máscaras independentes, partições por obra nem todas as identidades exigidas aqui. Os source files originais também não estão autocontidos nessa pasta; apenas reencodes em `originals/page_###` foram preservados.

Essa run tampouco é replay offline: seus 61 `TranslationAttempt` físicos registram 21 chamadas de provider e 40 cache hits. Os backends incluem Google e Ollama, e o cache local é somente um mapa chave→texto sem ledger de request/resposta/modelo/provenance. Um cache miss ainda pode chamar Google vivo. Logo, `off6` não satisfaz nem o corpus nem o replay V4.2.

Existe uma fixture sintética `english_owner_recovery` com entradas marcadas como calibration/holdout, mas inputs e expectations ficam abertos no mesmo manifest, os paths privados associados não estão configurados e não há máscaras/partições independentes. Ela continua útil para testes unitários, não para holdout cego.

## 4. Taxonomia real de persistência e versionamento

A lista original de cinco classes não representava tipos que já possuem schema/hash canônico, mas não têm decoder completo. Fica adotada a seguinte taxonomia exata:

- `EVC`: codec versionado completo, com escrita e leitura/reabertura verificável;
- `EVH`: registro versionado ou hash-linked real, sem codec completo do objeto;
- `EUC`: forma canônica/hash real sem `schema_version` explícito; considerada legacy implícita V1;
- `ESN`: estrutura serializada como membro de outro contrato, sem identidade canônica própria;
- `EHR`: objeto apenas de runtime, sem codec/identidade do objeto;
- `NEW`: contrato ainda ausente; sua primeira versão será V1.

Não serão inventados “bytes V1” para tipos que nunca foram persistidos. Golden migration só é exigida onde bytes ou projeções legacy realmente existem.

### 4.1 Tipos existentes — matriz de lock

| Tipo real | Dono atual | Classe | Versão real | Próxima escrita |
|---|---|---:|---:|---|
| `SourceTextComponent` | `ownership/model.py` | ESN | dentro de OwnerGraph V1/V2 | OwnerGraph V3 |
| `TextObservation` | `ownership/model.py` | ESN | dentro de OwnerGraph V1/V2 | OwnerGraph V3 |
| `ComponentDisposition` | `ownership/model.py` | ESN | dentro de OwnerGraph V1/V2 | OwnerGraph V3; permanece a fonte de verdade |
| `TextOwner` | `ownership/model.py` | ESN | dentro de OwnerGraph V1/V2 | OwnerGraph V3 |
| `OwnerProjection` | `ownership/model.py` | ESN | dentro de OwnerGraph V1/V2 | OwnerGraph V3 |
| `OwnerGraph` | `ownership/model.py:1519` | EVC | V1 legacy, V2 atual | V3 |
| `OwnerGraphSnapshot` | `strip/page_pipeline.py` | EVC | acompanha V1/V2 do graph | V3 |
| `PageCoverageResult` | `ownership/coverage.py` | EVC | V1 | V2 se receber refs pré-tradução |
| `OwnerTranslationRequest` | `ownership/translation.py:415` | EUC | V1 implícita | V2 explícita |
| `TranslationAttempt` | `ownership/translation.py:482` | EUC | V1 implícita | V2 explícita |
| `TargetLanguageVerdict` | `translator/language_policy.py:206` | EUC | V1 implícita | V2 explícita |
| `TranslationBinding` | `ownership/translation.py:666` | EUC | V1 implícita | V2 explícita e autorizada |
| `OwnerPageTranslationResult` | `ownership/translation.py:767` | EUC | V1 implícita | V2 explícita ou substituição por resolution result |
| `OwnerTextExecutionAuthority` | `ownership/delivery.py:66` | EVC | V1 | V2 discriminada |
| `OwnerTextDeliveryContract` | `ownership/delivery.py` | EVH | hash-linked, sem schema agregado | V1 explícita ou evolução conjunta da authority |
| `GlyphRunObservation` / `RenderedGlyphSpanEvidence` | `ownership/delivery.py` | ESN/EVH | embutidos | records explícitos se cruzarem fronteira persistida |
| `OwnerMutation` | `ownership/model.py:329` | EHR | ausente | receipt V1, sem legado fictício |
| `OwnerGlyphPatch` | `ownership/model.py:1127` | EVH | hash payload V1 | V2 explícita ou record V2 |
| `OwnerStyleRasterContract` | `ownership/model.py` | EVC | V1/V2 reais | V3 somente se necessário |
| `OwnerExecutionCommit` | `ownership/model.py:1374` | EUC | V1 implícita/projeção persistida | V2 completo |
| `OwnerReplacementTransaction` | `ownership/execution.py:251` | EHR | ausente | permanece runtime ou ganha receipt V1 |
| `OwnerReplacementOutcome` | `ownership/execution.py:104` | EHR | ausente | outcome canônico V1 discriminado |
| `OwnerTargetMaterialization` | `ownership/model.py:2833` | EVC | V1 | V2 |
| `PageCompositionResult` | `ownership/model.py:1243` | ESN/EHR | serializado, não canônico | não confundir com snapshot |
| `PageCompositionSnapshot` | `ownership/execution.py:430` | EVC | V1 | V2 discriminado por outcome |
| `FinalReplacementVerdict` | `ownership/model.py:2906` | EVC | V1 | V2 |
| `LanguageResidualIssue` | `ownership/model.py:2853` | EVC | V1 | V2 quando a prova mudar |
| `FinalQAProbe` | `ownership/model.py:2882` | EVC | V1 | V2 |
| `OwnerRepairRequest` / `RepairAttempt` | `ownership/model.py` | EVC | V1 | V2 se integrarem resolution/review |
| `TerminalPixelProof` | `ownership/execution.py:618` | EUC | V1 implícita persistida | V2 explícita e discriminada |
| `PageExecutionResult` | `strip/page_pipeline.py:293` | EVC | V1 | V2 |
| `PageExecutionEvidenceSnapshot` | `ownership/execution.py:1077` | EVC | V1 | V2 |
| `PageExecutionEvidenceRef` | `ownership/execution.py:1144` | EVC | V1 | V2 |
| `PageCandidateTransaction` | `ownership/execution.py:1222` | EHR | ausente | manter runtime; receipt V1 se necessário |
| `ArtifactGenerationMarker` | `ownership/execution.py` | EVC | V1 | V2 se a authority de publicação mudar |
| `PersistedAssetRef` / `PersistedRGBImageArtifactRef` | `ownership/execution.py` | EVH | hash-linked | formalizar V1 antes de ampliar |
| `ChapterSourceManifest` | `ownership/chapter_contract.py` | EVC | V1 | V2 |
| `VerifiedPageProjectInput` | `ownership/chapter_contract.py` | ESN | dentro de VerifiedProjectInputs V1 | dentro de V2 |
| `VerifiedProjectInputs` | `ownership/chapter_contract.py:335` | EVC | V1 | V2 |
| `ChapterAssetManifest` | `ownership/chapter_contract.py:552` | EVC | V1 | V2 |
| `ExportPageEntry` | `ownership/chapter_contract.py` | EUC | V1 implícita/embutida | V2 explícita ou pelo manifest V2 |
| `ExportManifest` | `ownership/chapter_contract.py:694` | EVC | V1, somente PASS | V2 |
| `PublicationReceipt` | `ownership/chapter_contract.py:747` | EVC | V1 | V2 |
| `VerifiedChapterBundle` | `ownership/chapter_contract.py:786` | EVH | payload/hash V1 sem decoder completo | V2 ou bundle receipt V1 |
| `PublicationTransaction` | `ownership/publication.py` | EHR | ausente | manter runtime; receipt é autoridade persistida |

O uso atual de `FINAL_OWNER_RECORD_SCHEMA_VERSION = 1` por vários tipos não pode continuar para V2. Cada record evoluído passa a possuir sua própria constante e dispatch de decoder; um tipo não muda a versão dos demais por efeito colateral.

### 4.2 Novos contratos e primeiras escritas explícitas

Os seguintes nomes não existem no checkout e são contratos novos, não migrações:

| Grupo | Novos contratos V1 |
|---|---|
| identidade | `OwnerVisualIdentityV1`, `OwnerIdentityReconciliationV1`, `OwnerAuthorityEvidenceV1` |
| artefatos pré-tradução | `PageSpaceArtifactRefV1`, `OwnerPretranslationEvidenceBundleV1`, `SourceContaminationEvidenceV1` |
| elegibilidade/recovery | `OwnerTranslationEligibilityV1`, `RecoveredSourceCandidateV1`, `RecoveredTargetCandidateV1` |
| tradução | `TranslationBindingCandidateV1`, `TranslationTransformationReceiptV1`, `TranslationAuthorityChainV1`, `TranslationTrustVerdictV1` |
| resolução | `OwnerSemanticResolutionV1`, `OwnerResolutionDecisionV1`, `OwnerPageResolutionResultV1` |
| target/execução | `SealedOwnerTargetV1`, `ReplaceExecutionPayloadV1`, `CleanupOnlyExecutionPayloadV1`, `OwnerOperationIntentV1`, `OwnerCleanupTransactionV1`, `CleanupExecutionCommitV1` |
| terminal | `OwnerReplacementOutcomeRecordV1`, `CleanupOutcomeV1`, `OwnerTerminalOutcomeV1`, `OwnerPreservationProofV1`, `OwnerExclusionProofV1`, `FinalCleanupVerdictV1` |
| review/conflitos | `OwnerReviewRecordRevisionV1`, `SharedMutationRegionResolutionV1` |
| auditoria | `OwnerPartitionAuditV1`, `RuntimeOwnerIntegrityAuditV1`, `ExternalPublicationVerificationV1` |
| gate/publicação | `CanonicalExportGateV2`, `ExportOverrideProofV1`, `PublicationReviewIndexV1` |
| replay/corpus | `TranslationReplayLedgerV1`, `PrivateCorpusManifestV1`, `HoldoutInputsManifestV1` |

`DetectedComponentDispositionV1` não será uma segunda fonte de verdade. Se for mantido como nome de API, será apenas uma projeção auditada de `ComponentDisposition` do OwnerGraph V3 e deverá carregar o hash do graph e provar equivalência byte a byte da decisão projetada.

Todos os tipos realmente novos começam em V1. `CanonicalExportGateV2` é a exceção nominal porque é o codec V2 explícito do protocolo export-gate já persistido de forma não canônica; não implica a existência inventada de bytes `CanonicalExportGateV1`.

## 5. Regras canônicas comuns

### 5.1 Dois domínios de identidade

Todo novo record que precise ser comparado entre runs separa:

1. `semantic_sha256`: conteúdo estável, sem `run_id`, `execution_id`, relógio, path absoluto ou versão da policy;
2. `authority_sha256`: envelope da execução, contendo `semantic_sha256`, `run_id`, `execution_id`, graph authority, policy snapshot e refs dos artefatos efetivamente usados.

IDs de execução continuam nos hashes de autoridade. Eles não serão removidos dos contratos que evitam cross-run substitution. A comparação reprodutível usa o hash semântico; autorização e publicação usam o hash de autoridade.

### 5.2 Canonicalização e hash de texto

Para V2:

- JSON canônico usa chaves ordenadas, números inteiros onde possível e UTF-8;
- `target_text_utf8_sha256` é SHA-256 dos **bytes UTF-8 exatos** do target persistido após todas as transformações determinísticas;
- não há trim, case fold, normalização Unicode, troca de newline ou colapso de whitespace durante o hash;
- qualquer normalização é uma transformação anterior, nomeada e registrada no `TranslationTransformationReceiptV1`;
- V1 continua sendo lido e verificado com seus bytes e regras originais.

Campos de relógio de parede podem existir como metadata não identitária, nunca dentro de `semantic_sha256`.

### 5.3 Identidade visual do owner

`OwnerVisualIdentityV1` contém no mínimo:

- `source_asset_decoded_pixel_sha256`;
- dimensões da página;
- geometria em page space, normalizada e ordenada;
- chaves visuais dos componentes e observations que formam o owner;
- hashes das regiões visuais usadas na reconciliação;
- `owner_semantic_key` derivada apenas desse conteúdo.

`policy_version`, `run_id`, `execution_id` e owner IDs efêmeros ficam fora dessa chave. A policy pertence à decisão/authority.

Se `OwnerAuthorityEvidenceV1` for armazenada junto ao graph, `owner_graph_content_sha256` é calculado sobre uma projeção explicitamente sem refs de authority/evidence. Assim o graph nunca depende do próprio hash.

## 6. Evidência pré-tradução em page space

O ponto obrigatório de integração é logo após `OwnerGraph.require_valid()` e antes de `OwnerTranslationRequest.from_graph()` em `run_page_owner_pipeline` (`pipeline/strip/page_pipeline.py:1764`). A captura tardia hoje existente em `execute_owner_page_graph` não pode autorizar uma tradução que já ocorreu.

### 6.1 `PageSpaceArtifactRefV1`

Cada imagem ou máscara reabrível contém:

- `artifact_generation_id`;
- path relativo dentro do artifact store, nunca path absoluto na identidade;
- `file_sha256`;
- `decoded_pixel_sha256` quando raster;
- `width`, `height`, `channels`, `dtype`;
- `coordinate_space = page_xy`;
- significado da máscara e owner/component refs;
- codec e parâmetros de decodificação.

Reabertura recalcula hash de arquivo, pixels decodificados, shape e dtype. Divergência é erro, não warning.

### 6.2 `OwnerPretranslationEvidenceBundleV1`

O bundle sela:

- identidade da página e hash semântico do graph;
- snapshot RGB original em page space;
- universo fechado de components, observations e owners;
- refs das máscaras de glyph, container, balloon e exclusão;
- OCR bruto e candidatos recuperados, com provenance;
- projeção exata de `ComponentDisposition`;
- hashes semântico e de autoridade do bundle.

Todo request, recovery, contamination evidence e resolution referencia o bundle. Observação criada depois do fechamento não pode ser usada na mesma execução sem criar uma nova revisão do bundle.

## 7. Contaminação, elegibilidade e recovery

`SourceContaminationEvidenceV1` é calculada antes de qualquer request ao provider e contém o conjunto completo de evidências, a classificação, o policy hash e seu próprio evidence hash.

Antes da tradução existe obrigatoriamente `OwnerTranslationEligibilityV1`:

| Situação | Provider remoto | Caminho permitido |
|---|---:|---|
| source comprovadamente limpo | permitido | request normal |
| source suspeito ou contaminado | proibido | recovery comprovado ou cleanup-only |
| recovered source limpo | permitido | novo request ligado ao candidate hash |
| recovered target PT-BR confiável | não necessário | target candidate, após trust |
| preserve/exclude | proibido | no-op provado |

`RecoveredSourceCandidateV1` contém texto de origem limpo e pode originar um request novo. `RecoveredTargetCandidateV1` contém texto alvo e só pode virar target após validação de idioma, authority e trust. Um nunca substitui semanticamente o outro.

Passthrough, repaint OCR e texto lido da imagem já modificada nunca são automaticamente `trusted`.

## 8. Cadeia de tradução e target selado

### 8.1 Ordem acíclica

```text
OwnerTranslationRequest V2
  -> TranslationAttempt V2 / resposta bruta
  -> TranslationTransformationReceipt V1
  -> TranslationBindingCandidate V1
  -> TranslationAuthorityChain V1
  -> TranslationTrustVerdict V1 (recalculado, não caller-asserted)
  -> OwnerSemanticResolution V1
  -> OwnerResolutionDecision V1
  -> TranslationBinding V2 autorizado
  -> SealedOwnerTarget V1
  -> OwnerTextExecutionAuthority V2
```

O binding **não** referencia `SealedOwnerTarget`. O target selado nasce depois e referencia o binding. Isso elimina o ciclo binding ↔ target.

### 8.2 Transformações do target

`TranslationTransformationReceiptV1` liga a resposta bruta ao target final. Ele registra, em ordem:

- hash de entrada;
- ID e versão de cada transformação;
- hashes before/after de placeholder restoration, correção de mojibake, pós-processamento, glossary/entity locks, name repair e normalizações;
- hash exato do target final.

Sem esse receipt, uma resposta correta do provider não prova o texto que chegou ao renderer.

### 8.3 Authority e trust

`TranslationAuthorityChainV1` contém:

- hash semântico e de autoridade do request;
- `target_text_utf8_sha256`;
- lista ordenada de todos os attempt hashes ou chain hash equivalente;
- transformation receipt hash;
- tipo de authority e referências obrigatórias por tipo;
- nenhuma referência a artefato futuro.

A matriz de authority é fechada:

- provider: terminal accepted attempt obrigatório;
- translation memory: entry ref + entry bytes hash obrigatórios;
- project approved: approval record + project revision/hash obrigatórios;
- human approved: identity/approval record + target hash obrigatórios;
- recovered target: recovered-target evidence + language verdict obrigatórios;
- passthrough/repaint: proibidos como authority de replace.

`TranslationTrustVerdictV1` é derivado por verificador que reabre as refs. O caller não pode definir `trust=trusted` por conta própria.

### 8.4 Resolução semântica

`OwnerSemanticResolutionV1` sela:

- owner visual/authority identities;
- todos os candidates em ordem determinística;
- eligibility, rejeições e motivos;
- evidence/policy hashes;
- candidate selecionado, ou ausência comprovada;
- target final hash quando replace;
- resolution hash.

`cleanup_only` só é permitido quando o auditor consegue demonstrar que o conjunto de candidates elegíveis e trusted é vazio. Se houver um target PT-BR trusted, a única resolução válida é replace.

`OwnerResolutionDecisionV1` é a decisão pré-execução. `OwnerTerminalOutcomeV1` é o resultado pós-composição. Eles não serão combinados em um único objeto, pois isso criaria dependência temporal da authority em prova ainda inexistente.

## 9. Replay offline reprodutível

`TranslationReplayProvider` substitui **somente** o acesso remoto do translator. Discovery, OCR, coverage, reconciliation, graph, inpaint, layout, typeset, composição, QA, gate e publicação continuam executando pelo pipeline real.

O replay atual de owner artifacts em `pipeline/strip/run.py` não atende ao contrato: ele pula discovery/OCR/coverage/translation/inpaint e ainda depende de style-copy. Não será reutilizado como prova V4.2.

`TranslationReplayLedgerV1` é indexado por `translation_request_semantic_sha256`, nunca apenas pelo request hash atual que inclui IDs de execução. Cada hit também verifica o request authority hash da run corrente. O record contém resposta bruta, attempts, transformation expectations, provider identity e hashes.

Regras:

- `--translation-replay-ledger <path>` habilita o provider;
- `--offline-translation-replay` proíbe rede;
- miss no ledger é erro determinístico;
- `provider_called = false` e `replay_hit = true` são estados V2 explícitos;
- flags aparecem no parser e no help;
- não há fallback silencioso ao provider real.

## 10. Modelo de owner, partições e operação

### 10.1 Três conceitos separados

- `TextOwner.disposition`: continua representando ownership (`owned|review`) no graph;
- `resolution_outcome`: decisão semântica (`replace|cleanup_only|preserve`);
- `operation_mode`: execução autorizada (`replace|cleanup_only|preserve|exclude`).

Não serão renomeadas dispositions atuais para `replace`, pois isso quebraria as invariantes entre `ComponentDisposition` e `TextOwner`.

### 10.2 Duas partições exatas

O auditor valida duas populações diferentes:

1. cada `TextOwner` do graph pertence a exatamente um terminal outcome;
2. cada `SourceTextComponent` pertence a exatamente uma `ComponentDisposition` (`owned`, `preserve` ou `suppress/exclude`, conforme o codec do graph).

Componentes preservados ou excluídos não ganham owner artificial. O `OwnerPartitionAuditV1` contém as duas partições e um crosswalk; nenhuma entrada pode faltar ou aparecer duas vezes.

### 10.3 `OwnerTextExecutionAuthorityV2`

A authority é discriminada por `operation_mode`:

- replace → `ReplaceExecutionPayloadV1`, com sealed target e binding autorizados;
- cleanup-only → `CleanupOnlyExecutionPayloadV1`, com resolution, contamination evidence, review rev0 e mask plan;
- preserve → decision + policy proof, sem target/mutação;
- exclude → component policy proof, sem target/mutação.

Preserve/exclude nunca entram no compositor como glyph patch. Sua prova terminal confirma ausência de mutação autorizada e igualdade dos pixels relevantes.

## 11. DAGs de execução corrigidos

### 11.1 Replace

```text
evidence bundle
  -> contamination / eligibility
  -> request / attempts / transformations
  -> trust / semantic resolution / decision(replace)
  -> authorized binding
  -> sealed target
  -> OwnerTextExecutionAuthorityV2(replace)
  -> OwnerOperationIntentV1
  -> OwnerMutation + OwnerGlyphPatch
  -> OwnerReplacementTransaction.build(...)
  -> transaction.commit()
  -> OwnerExecutionCommit + OwnerTargetMaterialization + OwnerReplacementOutcome
  -> PageCompositionSnapshotV2 + candidato final persistido
  -> fresh physical QA
  -> FinalReplacementVerdictV2
  -> OwnerTerminalOutcomeV1
  -> TerminalPixelProofV2
```

O runtime `OwnerReplacementTransaction` é construído depois que mutation e glyph patch existem; sua chamada `commit()` produz commit, materialization e outcome. Ele nunca aparece depois do outcome. Um intent canônico separado é o que autoriza a mutação antes dela ocorrer.

### 11.2 Cleanup-only

```text
decision(cleanup_only)
  -> OwnerReviewRecordRevisionV1 rev0, open, sem verdict
  -> OwnerTextExecutionAuthorityV2(cleanup_only)
  -> OwnerOperationIntentV1
  -> OwnerMutation, sem glyph patch
  -> OwnerCleanupTransactionV1.commit()
  -> CleanupExecutionCommitV1 + CleanupOutcomeV1
  -> PageCompositionSnapshotV2 + candidato final persistido
  -> fresh physical OCR/QA
  -> FinalCleanupVerdictV1
  -> OwnerReviewRecordRevisionV1 rev1, referenciando verdict
  -> OwnerTerminalOutcomeV1
  -> TerminalPixelProofV2
```

Authority rev0 → commit → verdict → review rev1 é acíclico. A transação não referencia resultado futuro, e o verdict só nasce depois dos pixels compostos e persistidos.

### 11.3 Preserve e exclude

```text
decision(preserve|exclude)
  -> OwnerTextExecutionAuthorityV2(no-op), quando houver owner
  -> zero mutations / zero glyph patches
  -> candidato final persistido
  -> OwnerPreservationProofV1 ou OwnerExclusionProofV1
  -> terminal outcome / page partition proof
```

### 11.4 Regiões de mutação compartilhadas

Máscaras de owners diferentes que se sobrepõem exigem `SharedMutationRegionResolutionV1` antes de composição. O record contém owners participantes, região page-space, policy aplicada, ordem/união autorizada e hash. Overlap sem resolução é BLOCK; não se escolhe silenciosamente o último writer.

### 11.5 Revisões canônicas

`OwnerReviewRecordRevisionV1` contém `owner_semantic_key`, revision monotônica, hash da revisão anterior, estado, reason code, evidence refs, resolution hash e os refs permitidos pela fase:

- rev0: `open`, sem commit e sem verdict; autoriza somente o caminho cleanup-only já resolvido;
- rev1: referencia rev0, cleanup authority, commit e final cleanup verdict;
- revisão humana posterior: referencia a revisão anterior e um approval/rejection record verificável.

O hash de rev0 pode entrar na cleanup authority. O hash de rev1 não pode voltar para a authority ou o commit. Dessa forma não existe ciclo review ↔ authority ↔ verdict.

## 12. Prova física terminal

`TerminalPixelProofV2` é discriminada por terminal outcome e contém exatamente uma prova por TextOwner.

Para replace, exige:

- binding autorizado e target selado;
- cleanup materializado;
- glyph patch e commit coerentes;
- target glyph fisicamente observado;
- ausência de source-language residual dentro e ao redor da região autorizada;
- hashes dos pixels finais persistidos.

Para cleanup-only, exige:

- nenhum binding/glyph/target;
- mutation e commit apenas de limpeza;
- source glyph removido segundo máscara independente e OCR físico fresco;
- ausência de texto inglês residual com threshold contratado;
- nenhuma alteração fora da união autorizada;
- review rev1 ligada ao final cleanup verdict.

Para preserve/exclude, exige zero mutações e prova de pixels preservados na região controlada.

Uma página só fecha quando:

- todos os TextOwners estão em exatamente um terminal outcome;
- todos os SourceTextComponents estão em exatamente uma disposition;
- todos os conflicts de overlap têm resolução;
- não existe owner unresolved invisível ao resultado;
- os refs físicos podem ser reabertos e rehashados.

## 13. Auditoria independente, gate e estados

### 13.1 Ordem do capítulo

```text
PageExecutionEvidenceSnapshot reaberto
  -> VerifiedProjectInputsV2
  -> RuntimeOwnerIntegrityAuditV1
  -> CanonicalExportGateV2
  -> PublicationReviewIndexV1 / ExportOverrideProofV1, quando aplicável
  -> ChapterAssetManifestV2 + ExportManifestV2
  -> PublicationReceiptV2
  -> commit atômico da publicação
  -> ExternalPublicationVerificationV1
```

O auditor runtime reabre evidence e recomputa partições, hashes e provas. Ele compartilha apenas codecs e primitivas de hash com o produtor; não importa decisões prontas do pipeline. O auditor externo roda sobre a publicação final e referencia o receipt. O terminal proof nunca referencia o auditor, e o publication receipt nunca referencia uma verificação externa futura.

`RuntimeOwnerIntegrityAuditV1` contém os hashes de todos os page evidence reabertos, as duas partições exatas, terminal outcome hashes, conflicts resolvidos, issues ordenadas, status e audit hash. `PublicationReviewIndexV1` contém refs físicas reabríveis de todos os review records abertos/fechados e prova cardinalidade exata. `ExternalPublicationVerificationV1` contém receipt ref, tree hash recomputado, hashes de página/export, issues independentes, verifier version e verdict; ele é produzido somente depois do commit físico da publicação.

### 13.2 Gate canônico

`CanonicalExportGateV2` possui hash próprio e estados:

- `PASS`;
- `REVIEW`;
- `BLOCK`;
- `OVERRIDDEN`.

Regras:

- PASS publica normalmente;
- REVIEW pode publicar, mas exige `PublicationReviewIndexV1` completo e `output_review_state=needs_review`;
- OVERRIDDEN pode publicar, mas exige `ExportOverrideProofV1` verificável;
- BLOCK não recebe publication receipt final; pode manter preview/candidate bloqueado;
- REVIEW nunca é convertido em approved/clean.

Exit codes permanecem compatíveis:

- PASS/REVIEW/OVERRIDDEN: 0;
- BLOCK em modo normal: 0 com status bloqueado;
- BLOCK em strict: 2;
- falha de execução/contrato: não zero.

### 13.3 Migração Python/Rust/TypeScript

O protocolo V2 será escrito atomicamente nas três camadas:

- `completion_status = approved|review|blocked_preview|overridden|error`;
- `output_review_state = clean|needs_review`;
- `needs_review` permanece sinal explícito derivado e consistente.

Leitores legacy aceitam os valores atuais e derivam `needs_review` também de `qa.export_gate.needs_review`, `project.needs_review`, issue counts, review flags e histórico. `approved` legacy só vira `clean` quando nenhum desses sinais existe. Isso evita falso clean em projetos antigos.

### 13.4 Cadeia física de publicação V2

`VerifiedProjectInputsV2`, `ChapterAssetManifestV2`, `ExportManifestV2`, `PublicationReceiptV2` e `VerifiedChapterBundleV2` evoluem juntos. O receipt inclui hashes de:

- source manifest e asset manifest;
- page execution evidence e terminal proofs;
- runtime owner audit;
- canonical export gate;
- review index ou override proof, quando exigido;
- árvore física publicada.

`reopen_verified_publication` reconstrói e compara bytes de toda a cadeia. Um reader V1 permanece intacto; nenhuma leitura V1 injeta defaults que mudem seus bytes.

## 14. Corpus privado e holdout exigidos

### 14.1 `PrivateCorpusManifestV1`

O manifest canônico deve congelar, por obra/capítulo/página:

- corpus ID e revision;
- papel: `train_fixture`, `calibration`, `regression` ou `holdout`;
- path relativo sob a raiz privada;
- file SHA-256 e decoded-pixel SHA-256;
- width, height, channels e codec;
- page/source IDs semânticos;
- expected component/owner partitions;
- refs e hashes das máscaras independentes;
- ledger de tradução correspondente;
- policy/model/font/config hashes;
- licença/origem e regra de acesso, fora de hashes quando sensível;
- tree hash do corpus.

O corpus mínimo contratado pelo plano precisa localizar explicitamente:

- Mitch Items capítulo 39;
- God of Death capítulos 1, 2 e 3;
- pelo menos duas obras adicionais com layout/arte diferentes;
- páginas contendo balões simples, texto colorido, contorno, gradiente, texto grande, texto parcial e overlaps.

### 14.2 Máscaras independentes

Máscaras de expectativa não podem ser geradas pelo mesmo código/versão que está sendo avaliado. Cada ref registra autor, método, revisão, file/pixel hash e coordinate space. Alteração de máscara cria nova revision e invalida resultados anteriores.

### 14.3 Holdout cego

- inputs e expectativas ficam em manifests separados;
- tuning não pode ler expectativas do holdout;
- nenhum threshold é alterado após a primeira abertura das expectativas;
- resultados são append-only e ligados ao commit/config/model hashes;
- páginas usadas para corrigir um erro saem do holdout e entram em regression na revisão seguinte.

## 15. Estratégia de testes e fases

### 15.1 TDD por fronteira contratual

A futura implementação seguirá RED → GREEN → refactor, nesta ordem:

1. codecs/golden readers V1 e writers V2/V3;
2. identities e regra dos dois hashes;
3. refs físicos e reabertura do pretranslation bundle;
4. contamination + eligibility antes do provider;
5. request/attempt/transformation/trust chain;
6. semantic resolution e partições exatas;
7. replace, cleanup-only e no-op authorities;
8. transactions/outcomes discriminados;
9. composição e terminal proofs por outcome;
10. auditor runtime independente e gate REVIEW;
11. publicação/reabertura V1/V2;
12. replay offline completo;
13. corpus regression e holdout cego;
14. integração Python/Rust/TypeScript.

Cada teste RED precisa falhar pelo motivo contratado, não por import ausente ou fixture inválida.

### 15.2 Ordem operacional correta

```text
contract codecs
  -> shadow instrumentation
  -> tuning apenas em train/calibration
  -> thresholds congelados
  -> regression conhecida
  -> holdout cego
  -> enforcement
  -> publicação integrada
```

Não se ativa enforcement antes da calibração e do freeze.

### 15.3 Protocolo de performance

O baseline registra dispositivo CPU/GPU, modelos, versões, cache state, warmup, número de repetições e mediana/p95. Overhead é medido contra o mesmo corpus e ledger offline. Provider vivo nunca participa do benchmark reprodutível.

## 16. Desenvolvimento seguro no checkout dirty

Quando os pré-requisitos forem fornecidos:

- capturar branch, HEAD, status e blobs/diffs de cada arquivo alvo antes da primeira edição;
- manter ledger de mudanças preexistentes por arquivo;
- editar somente hunks necessários;
- nunca usar reset, checkout, stash, clean ou staging amplo por path;
- separar “development gate” dos novos testes do “integration gate” contra todas as mudanças locais;
- revisar `git diff --cached` se houver commit e provar que nenhum hunk preexistente foi apropriado;
- não criar commit enquanto a fronteira entre mudanças do usuário e da implementação não puder ser provada.

## 17. Gates para iniciar implementação

| Gate | Estado | Evidência exigida |
|---|---:|---|
| decisões de identidade/hash | fechado neste relatório | seções 4–5 |
| DAG acíclico e lifecycles | fechado neste relatório | seções 8–13 |
| matriz real de codecs | fechado neste relatório | seção 4 |
| root privado | NO-GO | env/path existente e congelado |
| corpus manifest | NO-GO | `PrivateCorpusManifestV1` válido |
| máscaras independentes | NO-GO | refs reabríveis e hashes |
| separação holdout | NO-GO | input/expectation manifests e roles |
| ledger offline | não implementado | fase TDD posterior ao corpus lock |
| inventário de transições de testes | ausente | captura canônica antes do primeiro RED |
| runtime/environment manifest | ausente | Python/deps/modelos/fontes/configs/seeds/device congelados |
| auditor independente V4.2 | não implementado | fase funcional posterior ao lock |
| ledger three-way do checkout dirty | ausente | capturar antes da primeira edição funcional |
| runtime funcional V4.2 | não iniciado por escopo | execução integral do plano após GO |

## 18. Requisitos faltantes e próxima ação

Além do bloqueio do corpus, o checkout ainda não contém os scaffolds de implementação previstos: replay provider/ledger, scripts de inventário de transições, environment bundle imutável, masks independentes, auditor runtime V4.2 ou ledger three-way de arquivos alvo. Eles devem ser produzidos nas fases TDD apropriadas, mas não convertem material histórico em corpus autorizado.

Até os quatro requisitos de corpus abaixo existirem, resultados históricos continuam úteis para diagnóstico visual, mas não podem ser usados para declarar regressão resolvida, calibrar thresholds finais, autorizar enforcement ou publicar uma conclusão de qualidade universal.

```text
CONTRACT_LOCK_STATUS:
NO_GO_MISSING_PRIVATE_CORPUS_CONTRACT

MISSING_REQUIREMENTS:
- TRADUZAI_PRIVATE_FIXTURE_ROOT
- private corpus manifests
- independent masks
- holdout role separation

NEXT_REQUIRED_ACTION:
obter e congelar esses artefatos antes de qualquer implementação funcional.
```

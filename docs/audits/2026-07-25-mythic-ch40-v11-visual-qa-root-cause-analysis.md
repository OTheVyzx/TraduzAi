# Auditoria visual e de QA — Mythic Items ch. 40 V11

Data: 2026-07-25  
Run auditado: `.codex-tmp/mythic_ch40_pure_inpaint_validation_20260725_v11`  
Run ID: `2026-07-25T14-42-18Z_regressed_genius_creates_mythic_items_12e1e7`

## Veredito

O V11 é **NO-GO**. O export gate está corretamente em `BLOCK`, mas o QA não descreve de forma confiável os defeitos visuais reais. Ele bloqueia principalmente por geometria e tamanho, enquanto deixa passar como `PASS` ou `WARN` texto inglês intacto, texto-fonte sem owner, inpaint que danifica a arte, resíduos, duplicação de owner e perda de texto na recomposição.

A tradução em si não é o gargalo dominante: foram 183 inputs e 183 outputs, todos processados pelo Google, sem fallback. A maioria dos vazamentos de inglês nasce antes ou depois do tradutor: recall incompleto de OCR, ownership fragmentado, máscara insegura, `review_required` ainda renderizado ou não bloqueado, e copyback/reassembly.

A captura `THE ARENA WILL BEGIN` foi anexada duas vezes. Ela é tratada como um único caso para manter os 48 casos únicos pedidos.

## Categorias de causa

- **COR** — contrato RGB/BGR inconsistente nos boundaries de leitura e gravação.
- **OCR** — texto-fonte ausente, parcial ou substituído por candidato pior.
- **OWN** — um corpo lógico repartido entre linhas/bands ou reivindicado por mais de um owner.
- **ROUTE** — divergência entre `route_action`, plano final e pixels exportados.
- **INP** — máscara insegura, inpaint não executado, resíduo ou dano à arte.
- **LAY** — escala sem teto, centro calculado pelo bbox errado ou texto fora do safe polygon.
- **CMP** — overlap/copyback sobrescreve texto já renderizado ou repõe inglês.
- **QA** — falso `PASS/WARN`, identidade errada ou auditoria circular.
- **STYLE** — fonte, stroke, glow e hierarquia visual. Esta categoria fica separada das correções funcionais abaixo.

## Regra obrigatória para o corpo do texto

O corpo não pode ser repartido semanticamente no OCR nem no typeset.

1. Cada balão comum gera um `owner_id`, um payload OCR, uma chamada de tradução e um bloco de typeset.
2. Linhas e fragments são evidências do owner; nunca viram traduções ou layers independentes.
3. Cards podem ter papéis semânticos (`title`, `note`, `body`, `footer`), mas cada parágrafo/body continua atômico.
4. Quebra de linha visual pertence exclusivamente ao typesetter.
5. Bands são janelas de execução, não unidades de ownership.

## Auditoria imagem por imagem

| Img | Band(s) | Causa raiz | Falha do QA | Solução completa |
|---:|---|---|---|---|
| 1 | `008–009` | **COR/CMP**: canais R/B inconsistentes em arrays do strip e writers; crops de band contaminavam a página. | Não mede alteração cromática fora da máscara de texto. | Contrato RGB canônico, conversão apenas nos boundaries OpenCV e teste vermelho/azul. Correção aplicada e validada no piloto V2. |
| 2 | `011` | **LAY/INP**: o bbox retangular da mancha radial é tratado como área útil; o texto invade o mascote e o contorno. | Detecta drift/underfill, mas não containment real por pixels. | Safe polygon pelo distance transform do interior, subtraindo SFX, raios e foreground; validar ink contra máscara, não bbox. |
| 3 | `020` | **LAY**: fonte 42 é aceita porque cabe, sem teto relativo ao texto-fonte. O burst também é classificado como `item_card`. | `PASS` apesar do texto desproporcional. | Exigir frame retangular fechado para card; aplicar teto por altura mediana dos glifos originais e ocupação visual. |
| 4 | `020` | **INP**: a máscara expandida cobre 1.177 de 5.507 pixels saturados do mascote azul e altera 1.068 deles. | `PASS`; não mede dano a foreground. | Máscara positiva por glyph union e máscara negativa para componentes cromáticos/ilustração; interseção com arte protegida deve ser zero. |
| 5 | `028` | **ROUTE/INP**: `HURRY, HURRY!` foi traduzido para `DEPRESSA, DEPRESSA!`, mas a máscara insegura pulou o inpaint (`changed_pixels=0`, residual `0.584738`) e não houve render. | Só acusa `render_bbox_missing`; não acusa inglês intacto. | Retry de máscara por evidência de glyph; se continuar insegura, bloquear com `source_text_not_cleaned`. |
| 6 | `033` | **OCR/OWN/INP**: card repartido em título/note/stat/body/footer independentes; coordenadas page/strip se misturam; AOT preserva fantasmas. | Enxerga sintomas de bbox/residual, não a fragmentação causal. | Owner único do card, rows estruturadas no mesmo espaço de coordenadas e rerender a partir do original + inpaint final. |
| 7 | `040–041` | **OWN/INP**: body dividido entre fragments, `item_card_joint_layout_failed`, limpeza com resíduos e engine real não comprovado. | `040` chega a `PASS`; `041` só acusa tiny/underfill/missing. | Full-card OCR, body atômico, máscara por linhas pertencentes ao owner e contrato pure verificável. |
| 8 | `042–043` | **ROUTE/CMP**: título foi traduzido e limpo, mas falha de fit/rollback restaura o inglês; body fica separado. | Tiny/underfill, sem indicar rollback do texto-fonte. | Retry de layout sem restaurar silenciosamente o inglês; qualquer owner traduzido precisa renderizar ou bloquear. |
| 9 | `049` | **LAY**: burst escuro recebe a mesma escala sem teto da img. 3. | `PASS`. | Teto de escala relativo à fonte e rejeição de burst como card retangular. |
| 10 | `053` | **CMP/OWN**: `09_typeset` contém a frase correta; outro overlap apaga o miolo depois de `NÃO!` em `10_copyback`. | Acusa geometria, não perda pós-typeset. | Compositor por ownership com prioridade por pixel `typeset > inpaint > original`; teste de sobrevivência dos glifos. |
| 11 | `061–063` | **OWN/INP**: um painel lógico vira três owners sobrepostos; algumas rows ficam review e resíduos permanecem (`0.218289`/`0.338476`). | Muitos sintomas locais, sem detectar um único card repartido. | Owner page-space global, rows completas e tradução/render atômicos do painel. |
| 12 | `067` | **OCR/QA**: OCR captura apenas `ACHIEVED.`; o título `ACHIEVEMENT ...` nunca vira trace. | `PASS`, pois só avalia o trace existente. | Detector de recall independente e reOCR full-card quando houver componentes de texto sem owner. |
| 13 | `068–069` | **OWN/LAY**: um título vira três payloads, inclusive `IPGRADED`; cada fragmento recebe layout próprio. | Todos os fragments passam. | Uma string/owner semântico; lines apenas como evidência. STYLE será tratado depois por token de papel. |
| 14 | `074` | **OCR/ROUTE**: card inclinado gera OCR severamente corrompido, termina em review e não renderiza. | Detecta clipping, mas não cobertura textual incompleta. | Retificação de perspectiva, OCR multi-candidato e arbitration por cobertura, idioma e coerência. |
| 15 | `081` | **OCR/QA**: `LET'S HEAD THERE RIGHT AWAY!` não foi detectado. | Apenas `WARN no_matching_project_layer`. | Recall independente do metadata; texto latino sem owner deve ser crítico. |
| 16 | `086` | **LAY**: centro deriva do bbox estreito do OCR, não do interior do painel dourado. | `PASS`. | Detectar frame/interior do card e limitar drift a 4–6% da largura ou 8–16 px. |
| 17 | `090` | **OCR/ROUTE**: OCR vira `SURVIVAL MODE IODE`, termina review e não renderiza. | Só acusa tiny. | Full-card OCR, validação léxica e paridade obrigatória tradução → render. |
| 18 | `094–095` | **OCR/OWN**: reOCR mistura SFX `DING` com `PART`, renderiza `DING PARTE` e perde `PARTICIPANTS`. | Detecta center/missing, mas permite o candidato errado. | Segregar SFX por componente/posição e impedir que um candidato parcial substitua o owner do card. |
| 19 | `097` | **OCR/QA**: só `ONCE ALL PLAYERS HAVE` é capturado; linhas inicial/final ficam em inglês. | `PASS`. | OCR do painel completo e validação de cobertura de todos os componentes-fonte. |
| 20 | `111` | **OCR/QA**: só o miolo do card é capturado; `KIM SIMUN` e `PROMOTION MATCH` ficam em inglês. | `PASS`. | Owner full-panel e teste de sentença completa. |
| 21 | `114` | **ROUTE/INP**: OCR/tradução completos, mas máscara insegura leva a review e nenhum render. | Acusa center drift, não o inglês preservado. | Reconstruir máscara por glyph; persistindo risco, blocker dedicado. |
| 22 | `118–121` | **OWN/OCR/INP**: primeiro card é repartido/triplicado; fragmento duplicado em `119`; card inferior não recebe OCR. | Topo falha por clipping; área inferior recebe apenas `WARN`. | Owner cross-band, dedupe antes da tradução e recall independente para o segundo card. |
| 23 | `122–123` | **OWN/CMP**: mesmo balão, source e tradução são renderizados duas vezes por overlap. | Só `tiny_text`. | Dedupe global por `owner_id`, IoU e similaridade textual; um único `render_band_id`. |
| 24 | `128–129` | **INP/QA**: limpeza sem engine real deixa barras marrons/smears. | Ambos `PASS`. | Detector de seam/patch por luma, chroma e gradiente contra o ring; retry de inpaint. |
| 25 | `136` | **INP/ROUTE**: layer review ainda renderiza sobre máscara incompleta e deixa retângulo/resíduo. | `PASS` final apesar de warning de máscara. | Proibir `review_required` no plano final e medir residual nos pixels exportados. |
| 26 | `140` | **ROUTE/INP**: tradução existe, mas review sem render deixa `PITCH-BLACK` em inglês. | `render_bbox_missing`, sem flag de inglês intacto. | Retry seguro ou `untranslated_source_text` crítico. |
| 27 | `141` | **INP/ROUTE**: PT é sobreposto a limpeza incompleta/patch preto. | `PASS`. | Glyph union, residual OCR/stroke após inpaint e gate final autoritativo. |
| 28 | `143` | **OCR/QA**: só `VISION SKILL` recebe owner; `UGH... I WISH...` permanece em inglês. | `PASS`. | OCR do balão inteiro e completeness por componentes; body indivisível. |
| 29 | `144` | **INP/ROUTE**: PT em review é renderizado sobre texto-fonte não removido. | `PASS`. | Limpeza validada antes do render e proibição de review no plano final. |
| 30 | `155` | **INP/LAY**: máscara não cobre todo o inglês; PT sobrepõe o original. | Tiny + residual fraco. | Pass independente de residual dentro do owner e retry adaptativo. |
| 31 | `156` | **OCR/QA**: `WAIT...!` é descartado como `cjk_visual_misread_in_english_source`, embora seja latino. | Band `PASS` porque outro trace existe nela. | Nunca descartar candidato com alta razão latina por regra CJK; QA por componente-fonte. |
| 32 | `160` | **OCR/ROUTE**: `TH-THERE...` vira `Th There.`, perde hesitação, traduz errado e não renderiza. | Missing render. | Arbitration e normalização que preservem pontuação/hesitação. |
| 33 | `161` | **INP/LAY**: tradução correta, mas bbox/máscara deixam resíduo. | `render_bbox_outside_safe_or_balloon`. | Glyph union e safe polygon do balão. |
| 34 | `162` | **OCR/QA**: `SOMETHING JUST MOVED OVER THERE...` não tem trace. | Apenas `WARN no_matching_project_layer`. | Recall independente e blocker para source sem owner. |
| 35 | `163` | **OCR**: candidato full-crop pior (`SEB AN A`) substitui um primário mais completo. | Só acusa ausência de render. | Arbitration monotônica: candidato novo não substitui outro se reduzir cobertura/qualidade; guardar alternates. |
| 36 | `164` | **ROUTE/INP**: tradução correta termina review, não renderiza e o inglês permanece. | Missing render/mask critical. | Retry de máscara ou blocker visual dedicado. |
| 37 | `171` | **OCR/QA**: `WH-WHAT IS THAT?!` não foi detectado. | Apenas `WARN`. | Recall independente do projeto. |
| 38 | `172` | **INP/LAY**: PT é renderizado sobre inglês residual e ainda clipa. | Detecta clipping/overflow, não exige limpeza prévia. | Inpaint validado antes do typeset e fit estrito dentro do safe polygon. |
| 39 | `185` | **OCR/QA**: `WAIT... YOU'RE...?!` não possui layer. | Apenas `WARN`. | Recall final independente e `source_text_unowned` crítico. |
| 40 | `195/198/199` | **ROUTE/INP**: os três chats foram traduzidos, terminam review e não renderizam. | `195` chega a `PASS`; `198/199` só missing render. | Audit de idioma na imagem final e paridade output → owner → render. |
| 41 | `200–201` | **OWN/CMP**: mesmo corpo vira `VAMOS VER COMO ESTOU...` e `ESTOU FAZENDO...` em owners sobrepostos. | Só `200` recebe tiny. | Owner cross-band único, tradução única e reflow único. |
| 42 | `206` | **OCR/INP/QA**: ranking perde números/estrutura; máscara de 96.485 px destrói painel e bordas. | `PASS`. | OCR tabular/row-aware, números preservados, proteção de linhas/moldura e completeness. |
| 43 | `208–209` | **OWN/INP**: sentença completa + sufixo recebem owners distintos; segundo inpaint achata a textura. | `208 PASS`; `209` só tiny/flattened. | Dedupe global, máscara única e compositor por owner. |
| 44 | `212` | **OCR/QA**: `BEAST KING CHOI JIN-SOO!` não foi detectado. | Apenas `WARN`. | Recall independente; traduzir o título e preservar somente o nome próprio. |
| 45 | `213–215` | **OCR/QA**: só `FORMER UFC` é capturado; `HEAVYWEIGHT FIGHTER` e o parágrafo ficam em inglês. | `213` tiny; `214/215` apenas `WARN`. | Owner do balão completo e verificação de cobertura. |
| 46 | `217` | **OCR/QA**: parágrafo `BEASTIFICATION` inteiro não foi detectado. | Apenas `WARN`. | Recall independente e full-bubble OCR. |
| 47 | `219` | **OWN/OCR/INP**: body repartido, `Derformance` vira `deformidade`, review é parcialmente renderizado sobre inglês; inpaint pulado, residual `0.542106`. | Tiny/missing, não acusa source intacto. | Payload único, arbitration contextual, retry de máscara e blocker se não limpar. |
| 48 | `222–223` | **OCR/QA**: só `LOOKS LIKE IT'S` vira `PARECE QUE É`; `TIME TO GO ALL IN.` permanece em inglês. | Só tiny/`WARN`, sem leak de idioma. | Completeness por componentes e tradução atômica do balão inteiro. |

## Falhas do QA V11

### Contagens e inconsistência interna

- Export gate: `BLOCK`, 223 issues, 105 bloqueantes, 102 critical e 121 warning.
- QA visual final: 227 rows — 43 `PASS`, 101 `WARN`, 83 `FAIL`.
- `qa_export_gate_consistency.json` registra `consistent=false`:
  - summary: 76 flags críticas / 62 issues críticas;
  - gate: 164 flags críticas / 102 issues críticas.
- Nove flags não foram propagadas.
- O summary é calculado antes de blockers visuais/traceability adicionais e não é recomputado.

### QA não lê os pixels finais de forma independente

- `_qa_translated_final_crops_against_layers` valida principalmente geometria e metadata.
- `qa.visual_text_leak` existe, mas não participa do pipeline final e sua lista fixa não cobre os exemplos deste capítulo.
- O residual final é reaproveitado de um score do layer; não é medido novamente no crop exportado.
- `translated_crop_matches_final_band` é circular: compara a página com o crop que acabou de ser aplicado a ela.
- O QA roda antes de `final_translated_page_consistency_guard`, que ainda pode mudar os pixels.

### Texto inglês ou sem owner não bloqueia

- 101 rows recebem somente `WARN no_matching_project_layer`.
- Não há flags finais dedicadas `untranslated_source_text` e `source_text_unowned`.
- 30 layers estão `visible:false`; 17 foram suprimidos por render inseguro. O export gate exclui todo layer invisível, inclusive texto traduzível que ficou visualmente em inglês.
- Exemplo: `band195` mantém chat em inglês, recebe `PASS` e zero issues de gate.

### Ownership e identidade estão quebrados no QA

- 89 de 223 issues foram hidratadas com `text_instance_id` de outro band.
- A chave usada é o `text_id` local (`ocr_001`, `ocr_002`), que colide entre bands; a identidade precisa ser `trace_id`/`text_instance_id` composto.
- `band118` possui owners visíveis conflitantes sobre os mesmos source traces e payloads divergentes; não há flag de split/duplicate owner.
- O row schema agrega flags ao primeiro matched layer, escondendo o offender real.

### O contrato pure não é auditado

- Há 128 decisões de inpaint.
- Apenas quatro possuem simultaneamente `real_inpaint_engine` preenchido e `real_inpaint_mask_pixels > 0`.
- 107 declaram `used_real_inpaint=true`; em 103 delas isso contradiz engine/mask telemetry.
- Há 21 `real_inpaint_skipped_unsafe_mask`, todos com zero pixels alterados; o QA final não emite essa flag nem `source_text_not_cleaned`.
- As 22 ocorrências de `fast_fill_no_glyph_evidence` são stale: todas têm `used_fast_solid_fill=false` e `fast_fill_mask_pixels=0`.
- `mask_outside_balloon_critical` aparece 12 vezes, mas dez são rebaixadas a warning.

### Mismatch final não chega ao gate

- `final_band_crops_refresh.json` registra 70 `clean_band_final_mismatch` em 219 fontes limpas.
- Esses mismatches não são anexados aos rows/owners e não bloqueiam.
- A auditoria seguinte informa 226/226 comparações aprovadas porque compara artefatos já promovidos para a mesma página.

## Correção do QA proposta

1. Rodar detector/OCR final independente, cacheado por crop, depois do último writer.
2. Comparar n-grams do OCR final com o source de cada trace, excluindo apenas nomes, SFX e créditos explicitamente preservados.
3. Emitir `visual_text_leak` e `untranslated_english` críticos com evidência do OCR final.
4. Transformar `no_matching_project_layer + texto inglês` em `FAIL`; apenas crop realmente sem texto pode permanecer `WARN`.
5. Emitir `translatable_layer_suppressed` para layer traduzível invisível sem owner visível válido.
6. Validar o grafo source trace → owner:
   - zero owner: `translatable_owner_missing`;
   - mais de um owner com mesmo payload: `duplicate_visible_owner`;
   - mais de um owner com payload diferente: `split_owner_payload`.
7. Alterar o row schema para `offenders[]`, com flags, métricas e evidências por trace; gerar um issue por offender.
8. Usar `trace_id` composto em todas as junções e afirmar que page/band coincidem.
9. Propagar `clean_band_final_mismatch` por ownership mask e torná-lo bloqueante.
10. Tornar `consistent=false`, flag sem propagação e telemetry pure contraditória blockers de integridade do próprio QA.

## Cópia de estilo — frente separada

STYLE não explica a maior parte dos vazamentos de inglês, corpos incompletos ou inpaints ruins. Ela deve ser tratada somente depois de ownership, recall, máscara e composição estarem corretos.

Problemas observados:

- fontes/fallbacks diferentes dentro do mesmo balão, sobretudo imgs. 11 e 13;
- escala de `title/body/footer` inconsistente nos cards das imgs. 6–8;
- cor, stroke e glow escolhidos por fragmento, não por owner;
- bursts classificados como card e recebendo política visual errada.

Contrato futuro:

- um token de estilo-base por owner;
- balão comum com estilo uniforme;
- card varia apenas por papel semântico;
- variação intra-frase somente com evidência visual confiável, nunca por fragmentação do OCR.

## Ordem de implementação TDD

1. **COR — concluído nesta auditoria**
   - fixture assimétrica vermelho/azul em concat, output writer e writers de `images/originals`;
   - piloto real de uma página e inspeção visual.
2. **Ownership atômico**
   - um owner para `122/123`, `200/201`, `208/209`;
   - body de cards e balões não pode virar payloads de linha.
3. **Recall/arbitration OCR**
   - inglês sem layer vira crítico;
   - candidato novo não pode reduzir cobertura do owner.
4. **Máscara/inpaint pure**
   - burst não é card;
   - proteção pixel-exata do mascote azul;
   - unsafe skip gera retry ou blocker, nunca inglês silencioso.
5. **Layout/composição**
   - teto de escala;
   - centro pelo safe polygon;
   - sobrevivência do typeset após overlaps.
6. **QA final independente**
   - OCR de pixels finais, offenders, grafo de owners, identidade composta e consistência do gate.
7. **STYLE**
   - somente após os sentinelas funcionais estarem verdes.

Sentinelas visuais prioritários para a próxima execução: `008–009`, `020`, `028`, `033`, `040–043`, `053`, `061–063`, `118–123`, `128–129`, `136`, `140–144`, `155–156`, `160–164`, `171–172`, `185`, `195/198/199`, `200–201`, `206`, `208–209`, `212–215`, `217`, `219`, `222–223`.

## Validação da correção de cor

Piloto: `.codex-tmp/rgb_contract_page1_pilot_output_v2`  
Tempo: 108,1 s  
Resultado do pipeline: exit code 0

Comparação contra a imagem-fonte; menor MAE normal que MAE com R/B trocados significa contrato correto:

| Artefato | MAE normal | MAE R/B trocado | Resultado |
|---|---:|---:|---|
| `originals/001.jpg` | 0,1084 | 22,4025 | PASS |
| `images/001.jpg` | 5,8051 | 26,9945 | PASS |
| `translated/001.jpg` | 5,1597 | 26,6040 | PASS |
| `debug_inpaint/.../00_band_original.jpg` | 0,7459 | 36,4895 | PASS |
| `09_typeset/.../post_typeset.jpg` | 5,8662 | 41,5983 | PASS |
| `10_copyback_reassemble/final_bands/...009.jpg` | 5,9456 | 41,6040 | PASS |

A inspeção visual do `translated/001.jpg` não mostra mais as faixas amarelo/ciano com canais trocados da img. 1.

### Resultado dos testes

- Testes TDD específicos do contrato de cor: **3 passed**.
- Piloto real page 1: **exit 0**, inspeção visual aprovada e todos os seis boundaries acima em `PASS` cromático.
- Suíte ampliada (`test_main_emit.py`, `test_strip_concat.py`, `test_strip_run.py`): **290 passed, 16 failed**.

As 16 falhas ampliadas estão fora dos hunks de cor e cobrem contratos já divergentes no checkout sujo: style/SFX, route action, safe-area/mask repair, stale fit flags, multisource ownership, preview e a geometria page-space já conhecida. Portanto, a suíte global atual não está verde e essas falhas não foram ocultadas como sucesso da correção cromática.

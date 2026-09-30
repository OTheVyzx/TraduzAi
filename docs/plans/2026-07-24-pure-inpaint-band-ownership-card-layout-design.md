# Correção de inpaint puro, ownership entre bandas e layout de cards

**Data:** 2026-07-24  
**Status:** aprovado para planejamento de implementação  
**Escopo inicial:** pipeline automático do TraduzAI; capítulo 40 de *Regressed Genius Creates Mythic Items* como reprodução principal

## Objetivo

Eliminar os preenchimentos sólidos que criam faixas e regiões chapadas, impedir que textos de bandas sobrepostas sejam limpos ou renderizados duas vezes, preservar o original quando limpeza e render não puderem completar juntos e produzir um layout legível e hierárquico para cards de item.

O modo de qualidade do pipeline automático passa a usar inpaint puro como comportamento padrão. AOT ou LaMA removem os glifos dentro da máscara; Telea permanece apenas como fallback local. Preenchimentos diretos de cor não podem substituir o inpaint nesse modo.

## Evidência da reprodução

O run analisado é:

```text
N:\TraduzAI\.codex-tmp\mythic_ch40_card_resolved4_full_20260724
run_id: 2026-07-24T02-43-27Z_regressed_genius_creates_mythic_items_ea1037
```

Casos representativos da página 2:

- bands 25 e 26: duas leituras quase idênticas de `DID IT GO WELL?` ocupam a mesma região de página;
- band 28: o texto original é apagado, mas o texto traduzido não é renderizado;
- band 32: fundo escuro recebe região retangular chapada;
- band 40: fragmento OCR em `y=0` não pertence ao candidato detectado da banda;
- bands 41 e 43: títulos de cards aparecem exatamente na borda superior da banda e terminam renderizados com 6 px;
- bands 41, 42, 43 e 45: rows de cards/painéis recebem faixas sólidas;
- bands 35 e 39: referências positivas que não podem regredir.

O QA principal registrou 35 warnings e zero blockers na página 2. O QA visual final registrou `tiny_text`, `render_bbox_missing`, `dark_text_underfilled`, `translated_crop_mismatch_final_band` e outros failures, mas o export gate não os transformou em bloqueios.

## Primeiras divergências

### Ownership entre bandas

As bandas possuem margens intencionalmente sobrepostas. O reconciliador atual cobre somente o caso em que uma leitura curta é prefixo de outra leitura significativamente maior. Ele não cobre duplicatas completas com pequenas variações de caixa/pontuação nem fragmentos sem suporte do detector.

Ownership deve ser decidido em coordenadas de página antes de tradução, inpaint e typesetting. A decisão deve considerar conjuntamente:

- sobreposição geométrica;
- similaridade normalizada do texto;
- cobertura pelo candidato detectado;
- distância até as bordas da banda;
- confiança OCR;
- completude da leitura;
- perfil especial de card visual.

O registro perdedor não é somente ocultado no renderer: ele não pode chegar à tradução nem ao inpaint.

### Recorte adaptativo de cards

A margem padrão de 160 px é calculada antes do recall visual do card. Nos casos observados, o título começa exatamente no primeiro pixel disponível. Aumentar a margem global criaria mais sobreposição e mais custo.

Quando OCR confiável ou `visual_card_ocr_recall` tocar a borda superior/inferior, a banda deve ser expandida somente naquela direção, limitada à página de origem, e o OCR deve ser repetido uma vez. O `band_id` permanece estável e o manifest registra o recorte anterior, o novo recorte e a razão da expansão.

### Inpaint puro

O pipeline atual possui múltiplos caminhos que atribuem uma única cor diretamente à máscara. Alguns são protegidos por flags `TRADUZAI_STRIP_FAST_*`; outros, como o guard final de cards e o cleanup geométrico da página, são executados independentemente dessas flags.

Uma política única deve controlar todos esses caminhos. Em `pure`:

- fast white, solid, local, metadata e dark fills ficam desativados;
- `text_contract_direct_fill` não altera pixels;
- `visual_item_card_forced_contract_cleanup` não altera pixels;
- o cleanup de página não aplica cor sólida;
- AOT/LaMA recebe a action mask final;
- Telea só pode operar dentro da mesma action mask;
- pixels fora da action mask são restaurados pelo clamp;
- máscara insegura ou engine indisponível preserva o original e gera review/block.

### Atomicidade limpeza-render

Limpar o texto original e depois suprimir a tradução gera retângulos vazios. Para cada trace, o resultado deve ser atômico:

1. limpeza e render concluídos; ou
2. pixels originais restaurados e camada marcada para revisão.

Se o renderer não produzir bbox válido, ultrapassar a área segura ou cair abaixo do mínimo legível, o copyback restaura a área original daquele trace usando a máscara de ação registrada.

### Layout de card

Cada card possui um pai visual e filhos semânticos independentes. O texto não deve ser concatenado apenas para caber, mas o layout deve ser resolvido como conjunto.

Os filhos recebem papéis geométricos:

- `title`: primeiro grupo, próximo ao topo e com destaque;
- `note`: nota/grade imediatamente abaixo do ícone ou título;
- `body`: descrição principal, que pode ocupar várias linhas;
- `footer`: texto curto próximo ao limite inferior.

O solver preserva ordem, centro e proporção visual do original, distribui espaço vertical sem sobreposição e utiliza tamanho consistente por papel. Nenhum resultado abaixo do mínimo legível é renderizado.

### QA e export gate

O QA visual pós-rerender já roda antes de `evaluate_export_gate`, mas seus rows não são consumidos pelo gate. Eles devem virar issues ligadas aos `trace_ids`.

Flags críticas:

- `render_bbox_missing`;
- `render_bbox_outside_crop`;
- `render_bbox_outside_safe_or_balloon`;
- `translated_crop_mismatch_final_band`;
- `clipped_text_flag`;
- `dark_original_residual`;
- `inpaint_texture_flattened`;
- `cross_band_render_collision`;
- `pure_inpaint_unresolved`;
- `fit_below_minimum_legible` não resolvido.

`tiny_text` e `dark_text_underfilled` começam como crítica quando cruzarem os thresholds atuais de fail. A calibração deve ser feita contra os casos positivos antes de ampliar o gate a toda a suíte visual.

## Fluxo proposto

```text
detectar bandas
  -> OCR por banda
  -> recorte adaptativo de edge/card e retry único
  -> arbitragem page-space de ownership
  -> tradução apenas dos owners
  -> construção da action mask
  -> AOT/LaMA ou Telea restrito
  -> layout conjunto por card/balão
  -> render
  -> rollback por trace se limpeza-render não for atômica
  -> copyback/reassemble
  -> QA visual final
  -> export gate
```

## Contratos e telemetria

Cada layer renderizável deve manter:

- `trace_id`, `band_id` e coordenadas declaradas;
- `cross_band_owner_trace_id` e razão da escolha, quando aplicável;
- `card_panel_id`, `card_panel_role` e índice do filho;
- `inpaint_policy` e engine efetivamente usada;
- raw, expanded/action e effective limit masks;
- `font_size_seed`, `font_size_final` e `minimum_legible_font_px`;
- `fit_status` e `fit_attempts` autoritativos;
- decisão de rollback/preserve quando não houver render seguro.

Todo layer `translate_inpaint_render` visível deve possuir uma decisão em `08_inpaint` e uma decisão final em `09_typeset`. Ausência de qualquer uma é issue crítica de completude.

## Artefatos de debug

Os artefatos `debug_inpaint` usam PIL enquanto o recorder E2E usa OpenCV. Isso troca os canais vermelho e azul no primeiro conjunto. O debug deve adotar uma convenção única e registrar `color_space` e `run_id` no metadata para impedir comparações com artefatos ambíguos ou antigos.

## Tratamento de erros

- Recorte adaptativo só pode ocorrer uma vez por banda; novo toque de borda vira review.
- Ownership incerto preserva os registros e bloqueia ação destrutiva; não escolhe por texto isoladamente.
- Falha de AOT/LaMA permite Telea local apenas com máscara segura.
- Falha de todas as engines preserva a imagem original.
- Falha de layout após limpeza restaura os pixels originais daquele trace.
- Falha de escrita de artefato não altera a saída, mas gera issue de observabilidade.

## Validação

### Unitária e de contrato

- duplicata completa em bandas adjacentes;
- fragmento de borda sem candidato correspondente;
- prefixo curto existente continua reconciliado;
- recorte adaptativo inclui título sem ultrapassar a página;
- pure mode não executa nenhum fill direto;
- máscara insegura preserva a fonte;
- card mantém todos os filhos e hierarquia;
- 6 px não é normalizado para fit `ok`;
- QA visual crítico bloqueia o gate;
- debug BGR/RGB corresponde ao artefato E2E.

### Visual

Validar bands 25, 26, 28, 32, 40, 41, 42, 43 e 45, mantendo 35 e 39 como controles positivos. A validação final deve incluir pelo menos um balão branco, um translúcido, texto sobre arte, painel escuro e card colorido de outro capítulo.

## Critérios de aceitação

- zero pixels atribuídos por fast/solid fill em `pure`;
- zero `visual_item_card_forced_contract_cleanup` aplicado;
- nenhum layer visível abaixo do mínimo legível;
- nenhuma duplicação page-space entre bandas;
- nenhuma limpeza destrutiva sem render correspondente;
- alterações fora da action mask iguais ao original;
- todos os failures visuais críticos presentes no export gate;
- cores equivalentes entre `debug_inpaint`, E2E e imagem final;
- casos positivos sem regressão visual.

## Fora do escopo

- reescrever todo o pipeline para processamento exclusivamente page-first;
- alterar tradução ou conteúdo textual além do necessário para ownership;
- modificar o Studio;
- paralelizar o renderer;
- remover o pipeline de bandas.

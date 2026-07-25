# Validação — pure inpaint, ownership de bandas e layout de cards

Data final da execução: 25/07/2026
Plano: `docs/plans/2026-07-24-pure-inpaint-band-ownership-card-layout-implementation.md`

## Veredito

As Tasks 1–13 foram implementadas e exercitadas. O contrato técnico de `inpaint_policy=pure` foi atendido no run integral V11: todas as 128 decisões usam a policy pura, sem fast fill, preenchimento sólido de painel escuro, cleanup forçado de card, alteração fora do limite efetivo, colisão cross-band ou layer renderizável sem decisão de inpaint.

A aprovação visual permanece **BLOCK**. O gate está coerente com os artefatos: ainda existem residual original, texto pequeno/incompleto e composição visual insuficiente em cards. O término técnico do pipeline não é aprovação para exportação.

## Run integral autoritativo

- Run ID: `2026-07-25T14-42-18Z_regressed_genius_creates_mythic_items_12e1e7`
- Config: `N:\TraduzAI\.codex-tmp\mythic_ch40_pure_inpaint_validation_20260725_v11_config.json`
- Workdir: `N:\TraduzAI\.codex-tmp\mythic_ch40_pure_inpaint_validation_20260725_v11`
- Projeto: `N:\TraduzAI\.codex-tmp\mythic_ch40_pure_inpaint_validation_20260725_v11\project.json`
- Auditoria: `N:\TraduzAI\.codex-tmp\mythic_ch40_pure_inpaint_validation_20260725_v11\pure_inpaint_contract_audit.json`
- Página 002: `N:\TraduzAI\.codex-tmp\mythic_ch40_pure_inpaint_validation_20260725_v11\translated\002.jpg`
- Sheet alvo: `N:\TraduzAI\.codex-tmp\mythic_ch40_pure_inpaint_validation_20260725_v11\validation_sheets\target_bands_before_after_final.jpg`
- QA visual: `N:\TraduzAI\.codex-tmp\mythic_ch40_pure_inpaint_validation_20260725_v11\debug\e2e\11_qa_export_gate\final_rerender_visual_qa.json`
- Export gate: `N:\TraduzAI\.codex-tmp\mythic_ch40_pure_inpaint_validation_20260725_v11\debug\e2e\11_qa_export_gate\export_gate.json`

O run persistiu 10 páginas, 157 layers de texto, 227 bandas finais, 128 decisões e 128 metadados de inpaint.

## Contratos automáticos

| Contrato | Resultado |
|---|---:|
| Decisões com policy `pure` | 128/128 |
| `fast_fill_mask_pixels > 0` | 0 |
| `dark_panel_fill_count > 0` | 0 |
| Cleanup forçado de card | 0 |
| Layers visíveis com fit inválido | 0 |
| Layers renderizáveis sem decisão | 0/112 |
| Candidatos a colisão cross-band | 0 |
| Crops comparados à página traduzida | 226 |
| Falhas de consistência crop/página | 0 |
| Pixels alterados fora do limite nos casos medidos | 0 |

Um fragmento de layout (`#fragment_2`) foi corretamente auditado pelo `trace_id` base herdado, cuja decisão está registrada na mesma banda. As bandas marcadas por achatamento de textura foram `page_002_band_025`, `page_002_band_045`, `page_005_band_108` e `page_009_band_209`.

## Inspeção visual das bandas-alvo

Foram inspecionadas as bandas 25, 26, 28, 32, 40, 41, 42, 43 e 45, além dos controles 35 e 39.

| Critério | Resultado | Evidência |
|---|---|---|
| `CORREU BEM?` aparece uma vez | PASS | Página 002 e sheet V11 |
| Fragmento `IMLAK...` ausente | PASS | Página 002 e banda 40 |
| Cards sem faixas sólidas | PASS | Metadata e inspeção visual |
| Cor/arte fora da máscara preservada | PASS | Card dourado preservado e zero alteração fora do limite |
| Título principal recuperado | PASS | `ELIXIR DO PODER SELVAGEM` legível na banda 40 |
| Nenhum retângulo vazio | PASS | Página e sheet V11 |
| Todos os textos de card limpos e legíveis | FAIL | Residual e texto pequeno persistem nas bandas 41–43 |

A quarentena de fragmentos cross-band removeu o falso `IMLAK/Simlak` sem remover `TAMBÉM, CERTO?`. A composição por ownership de pixels também impediu que um crop superior reintroduzisse conteúdo de contexto sobre o card inferior.

## QA visual e export gate

QA pós-rerender em 227 bandas:

- PASS: 43
- WARN: 101
- FAIL: 83

Export gate final:

- Status: `BLOCK`
- Issues: 223
- Bloqueantes: 105
- Críticas: 102
- Review: 121

O bloqueio é esperado e necessário: `weak_text_residual_after_inpaint` permanece recorrente e a inspeção confirma falhas reais de card.

## Regressão cruzada

- Raiz: `N:\TraduzAI\.codex-tmp\pure_inpaint_cross_category_20260725_v4`
- Resumo: `N:\TraduzAI\.codex-tmp\pure_inpaint_cross_category_20260725_v4\summary.json`

| Categoria | Caso | Resultado |
|---|---|---:|
| Balão branco simples | `page_001_band_007` | PASS |
| Balão translúcido | `page_001_band_009` | PASS |
| Texto sobre arte | corpus `mythic_ch39_p006_candidate_crop.png` | PASS |
| Painel escuro | `page_001_band_000` | PASS |
| Card colorido | `page_001_band_001` | FAIL |
| Referência `case16_final_validation_20260722` | SHA-256 e visual | PASS |
| Referência `case41_typeset_validation_20260722` | SHA-256 e visual | PASS |

O painel escuro passou com 25.624 pixels alterados, sem residual ou alteração fora do limite. O card colorido alterou 53.651 pixels sem sair do limite, mas falhou por residual original visível; portanto não foi promovido artificialmente a PASS.

## Validação automatizada

- Contrato principal de emit/gate/rerender/metadata: 28 testes focados aprovados.
- Recall de título colorido, normalização `EUXIR`, composição de overlap e quarentena cross-band: testes TDD verdes.
- Runtime/card/layout/inpaint: baterias focadas verdes.
- Regressão cruzada e referências positivas: nodeids focados verdes.
- Auditoria local de skills: 8 verificações aprovadas.
- `git diff --check`: executado no fechamento.

A suíte owner ampla conserva falhas legadas fora do recorte e dois testes antigos de fast dark fill incompatíveis com o novo default puro. Esses resultados não foram ocultados nem “corrigidos” afrouxando o contrato.

## Conclusão

O plano está implementado: pure inpaint é o default, ownership/copyback evita colisões, fragments inseguros são colocados em quarentena, cards usam layout conjunto e o gate impede exportar resultados visualmente ruins. O resultado do capítulo 40 é **implementação concluída com validação visual bloqueada**, com evidência reproduzível nos caminhos acima.

# Validação — pure inpaint, ownership de bandas e layout de cards

Data da execução: 24–25/07/2026
Plano executado: `docs/plans/2026-07-24-pure-inpaint-band-ownership-card-layout-implementation.md`

## Veredito

O contrato técnico de `inpaint_policy=pure` foi atendido no run integral do capítulo 40: não houve fast fill, preenchimento sólido de painel escuro, cleanup forçado de card, colisão cross-band, alteração fora da máscara nem layer renderizável sem decisão de inpaint. Os crops finais também são coerentes com as páginas traduzidas.

A aprovação visual do capítulo permanece **BLOCK**. O gate está coerente com a inspeção: cards e painéis escuros ainda contêm texto original residual sob traduções pequenas ou incompletas. Portanto, o run não deve ser tratado como visualmente aprovado apesar de a execução técnica ter terminado com sucesso.

## Run integral do capítulo 40

- Run ID: `2026-07-25T01-13-01Z_regressed_genius_creates_mythic_items_6d5d06`
- Config: `N:\TraduzAI\.codex-tmp\mythic_ch40_pure_inpaint_validation_20260724_v2_config.json`
- Workdir: `N:\TraduzAI\.codex-tmp\mythic_ch40_pure_inpaint_validation_20260724_v2`
- Projeto: `N:\TraduzAI\.codex-tmp\mythic_ch40_pure_inpaint_validation_20260724_v2\project.json`
- Auditoria: `N:\TraduzAI\.codex-tmp\mythic_ch40_pure_inpaint_validation_20260724_v2\pure_inpaint_contract_audit.json`
- Página traduzida de referência: `N:\TraduzAI\.codex-tmp\mythic_ch40_pure_inpaint_validation_20260724_v2\translated\002.jpg`
- Sheet das bandas-alvo: `N:\TraduzAI\.codex-tmp\mythic_ch40_pure_inpaint_validation_20260724_v2\validation_sheets\target_bands_before_after_final.jpg`
- QA visual final: `N:\TraduzAI\.codex-tmp\mythic_ch40_pure_inpaint_validation_20260724_v2\debug\e2e\11_qa_export_gate\final_rerender_visual_qa.json`
- Export gate: `N:\TraduzAI\.codex-tmp\mythic_ch40_pure_inpaint_validation_20260724_v2\debug\e2e\11_qa_export_gate\export_gate.json`

O run processou 10 páginas, 174 textos e 227 bandas finais. Foram persistidas 126 decisões de inpaint e 126 registros de metadata de inpaint.

## Auditoria automática dos contratos

| Contrato | Resultado |
|---|---:|
| Decisões com policy `pure` | 126/126 |
| `fast_fill_mask_pixels > 0` | 0 |
| `dark_panel_fill_count > 0` | 0 |
| Layers com cleanup forçado de card | 0 |
| Layers visíveis com fit inválido | 0 |
| Layers renderizáveis sem decisão de inpaint | 0/106 |
| Candidatos a colisão cross-band | 0 |
| Bandas comparadas com página traduzida | 226 |
| Falhas de consistência crop/página | 0 |
| Alterações fora da máscara nos casos medidos | 0 |

A detecção de textura foi reclassificada usando contexto externo à máscara. Quatro bandas continuam corretamente marcadas como `inpaint_texture_flattened`: `page_002_band_025`, `page_002_band_045`, `page_005_band_108` e `page_009_band_209`. A marca antiga de `page_004_band_094` foi removida porque a métrica final autoritativa registra `flattened=false`.

## Inspeção visual das bandas 25, 26, 28, 32, 40, 41, 42, 43 e 45

| Critério | Resultado | Evidência |
|---|---|---|
| `CORREU BEM?` aparece uma vez | PASS | Conferido na página final |
| `IMLAK...` não aparece | PASS | Conferido na página final |
| Nenhum retângulo vazio | PASS | Sheet final e página 002 |
| Cards sem faixas sólidas | PASS | Nenhum fill sólido em metadata e inspeção do sheet |
| Arte fora da máscara preservada | PASS | Zero pixels alterados fora do limite nos casos auditados |
| Título legível e próximo do tamanho original | FAIL | Cards 40–43 mantêm overlays pequenos/inseguros |
| Todos os textos do card presentes e limpos | FAIL | Texto inglês residual permanece sob parte das traduções |

As bandas-controle 35 e 39 não apresentaram regressão de propriedade/copyback. A falha dominante nos cards não é uma faixa sólida: é inpaint ausente ou incompleto combinado com typesetting pequeno, razão pela qual o gate bloqueia corretamente a exportação.

## QA visual e export gate

QA pós-rerender para 227 bandas:

- PASS: 45
- WARN: 104
- FAIL: 78

Export gate final:

- Status: `BLOCK`
- Issues: 224
- Issues bloqueantes: 99
- Issues críticas: 95
- Issues de review: 129

Flags mais frequentes no gate: `safe_text_box_recomputed` (121), `weak_text_residual_after_inpaint` (67), `tiny_text` (50), `mask_outside_balloon` (33), `fast_fill_no_glyph_evidence` (32), `render_bbox_missing` (25), `mask_outside_balloon_critical` (22), `dark_text_underfilled` (14), `dark_text_tiny` (12), `render_bbox_outside_crop` (10) e `inpaint_texture_flattened` (4).

## Regressão cruzada por categoria

Raiz dos artefatos: `N:\TraduzAI\.codex-tmp\pure_inpaint_cross_category_20260724_v2`
Resumo: `N:\TraduzAI\.codex-tmp\pure_inpaint_cross_category_20260724_v2\summary.json`

| Categoria | Caso | Resultado | Observação |
|---|---|---|---|
| Balão branco simples | `page_001_band_007` | PASS | 25.609 pixels alterados, zero fora da máscara, sem residual/flatten |
| Balão translúcido | `page_001_band_009` | PASS | 49.517 pixels alterados, zero fora da máscara, sem residual/flatten |
| Texto sobre arte | corpus `mythic_ch39_p006_candidate_crop.png` | PASS | Referência preservada e teste focado verde |
| Painel escuro | `page_001_band_000` | FAIL | Zero pixels de inpaint e residual original visível |
| Card colorido | `page_001_band_001` | FAIL | Zero pixels de inpaint e residual original visível |
| Referência positiva | `case16_final_validation_20260722` | PASS | Integridade por SHA-256 e comparação visual sem regressão |
| Referência positiva | `case41_typeset_validation_20260722` | PASS | Integridade por SHA-256 e comparação visual sem regressão |

Comparações visuais:

- `N:\TraduzAI\.codex-tmp\pure_inpaint_cross_category_20260724_v2\simple_white_balloon\comparison.jpg`
- `N:\TraduzAI\.codex-tmp\pure_inpaint_cross_category_20260724_v2\translucent_balloon\comparison.jpg`
- `N:\TraduzAI\.codex-tmp\pure_inpaint_cross_category_20260724_v2\dark_panel\comparison.jpg`
- `N:\TraduzAI\.codex-tmp\pure_inpaint_cross_category_20260724_v2\colored_card\comparison.jpg`
- `N:\TraduzAI\.codex-tmp\pure_inpaint_cross_category_20260724_v2\case16_final_validation_20260722\comparison.jpg`
- `N:\TraduzAI\.codex-tmp\pure_inpaint_cross_category_20260724_v2\case41_typeset_validation_20260722\comparison.jpg`

## Validação automatizada executada

- Contrato principal de emit/gate/rerender/consistência: 28 testes focados aprovados.
- Regressões finais de rollback, visibilidade e sincronização da classificação de textura: 9 testes aprovados.
- Regressão cruzada white/translucent/text-on-art/dark/card e referências positivas: 9 testes focados aprovados.
- Auditoria das skills locais: 8 verificações aprovadas.
- `git diff --check`: sem erros; apenas avisos esperados de conversão LF/CRLF.

A suíte owner ampla ainda possui falhas legadas fora do recorte desta implementação. Elas não foram mascaradas nem corrigidas por alteração massiva de defaults de testes. O fechamento deste plano se baseia nos nodeids focados, no run integral novo, nos artefatos finais e no gate visual efetivo.

## Conclusão

As Tasks 1–13 do plano foram implementadas e exercitadas. O sistema agora preserva ownership de banda, aplica pure inpaint como default, impede fills sólidos e oculta layers inseguros após rollback atômico. O estado final do produto para este capítulo é, corretamente, **implementação concluída com validação visual bloqueada**: o próximo trabalho deve atacar residual original e capacidade de layout em cards/painéis escuros, sem reintroduzir os atalhos de preenchimento proibidos.

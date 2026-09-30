# Validacao Style Copy V2 — remediacao do NO-GO

Data da execucao: 2026-08-01 (America/Sao_Paulo)

Plano: `docs/plans/2026-07-31-style-copy-v2-no-go-remediation-implementation.md`

Git HEAD validado: `1bb48d53dd5aa847a640f2d7168059b83a63e44c`
Scoped diff SHA-256 gravado nos manifests: `087f7cdb916f918f5fd3561f0186104f08f1ab5994b326cbd04e930300ca50ce`

## Resultado

| Dimensao | Veredicto | Evidencia principal |
|---|---|---|
| Style | **NO-GO** | Benchmark `BLOCK`; top-1 0,40 < 0,90, top-3 0 < 0,98 e metricas obrigatorias ausentes. |
| Functional | **NO-GO** | As tres execucoes da matriz terminaram `BLOCK`; 8/9 owners inspecionados mantiveram ingles. |
| Inspection | **NO-GO** | 9/9 contact sheets autenticados e inspecionados em escala nativa falharam. |
| Overall | **NO-GO** | Overall exige Style GO + Functional GO + Inspection GO. |

A remediacao tornou contratos, composicao de gates e auditoria fail-closed reproduziveis, mas nao tornou o output visual aceitavel. O export gate bloqueou os resultados em vez de publicar pixels inconsistentes.

## Provenance e comandos

Root externo imutavel desta rodada:

`N:\TraduzAI_style_runs\style_copy_v2_remediation_holdout_20260801_001546`

Runtime: Python 3.12.10 via `pipeline\venv\Scripts\python.exe`; benchmark seed `1729`; matrix seed `0`; configs efetivos com `style_copy_mode=enforce`.

Comandos executados:

```powershell
pipeline/venv/Scripts/python.exe pipeline/debug_tools/run_style_benchmark_v2.py --spec pipeline/tests/fixtures/style_benchmark_v2/benchmark_spec.json --level all --output-root N:\TraduzAI_style_runs\style_copy_v2_remediation_holdout_20260801_001546\benchmark --run-id style-v2-enforce --seed 1729 --mode enforce
# exit 2: Style BLOCK medido

pipeline/venv/Scripts/python.exe pipeline/tools/validate_owner_visual_matrix.py --manifest pipeline/tests/fixtures/style_copy_corpus/matrix.json --output-root N:\TraduzAI_style_runs\style_copy_v2_remediation_holdout_20260801_001546\matrix --report N:\TraduzAI_style_runs\style_copy_v2_remediation_holdout_20260801_001546\matrix-report.md --inspection-template N:\TraduzAI_style_runs\style_copy_v2_remediation_holdout_20260801_001546\matrix\style-inspection-template.json
# exit 2: Functional/Style BLOCK e inspection PENDING

pipeline/venv/Scripts/python.exe pipeline/debug_tools/style_runtime_probe.py --output N:\TraduzAI_style_runs\style_copy_v2_remediation_holdout_20260801_001546\runtime-probe.json <tres project.json>
# exit 2: runtime BLOCK

pipeline/venv/Scripts/python.exe pipeline/tools/validate_owner_visual_matrix.py --manifest pipeline/tests/fixtures/style_copy_corpus/matrix.json --output-root N:\TraduzAI_style_runs\style_copy_v2_remediation_holdout_20260801_001546\matrix --report N:\TraduzAI_style_runs\style_copy_v2_remediation_holdout_20260801_001546\matrix-report-inspected.md --inspection-template N:\TraduzAI_style_runs\style_copy_v2_remediation_holdout_20260801_001546\matrix\style-inspection-template-validated.json --validate-only --inspection-manifest N:\TraduzAI_style_runs\style_copy_v2_remediation_holdout_20260801_001546\matrix\style-inspection-v2.json
# exit 2: Inspection NO-GO autenticado
```

Hashes centrais:

| Artefato | SHA-256 |
|---|---|
| benchmark spec | `9fced8255c9bbbb28dba36b3c447407ccdd6053d553946ecacba6d32f2d01fdc` |
| matrix manifest | `e937b5ee666806228aa5c84af04ec3ba15d0a869aafb821d6102c496c8a65456` |
| benchmark tool | `8ec5f6e24073a8100c572d304e3332088aef0a62affcebc2013d2d92cf2ffd0c` |
| matrix tool | `39d96a3ac3195c1f3b4b7b3cf4824e471bd4462afc47ad2f20d40e1d34ca82cb` |
| benchmark summary | `18f7d8dbb7fdc5c9149a950827924501f766f31b89cb5b79c8343d7f958f3145` |
| runtime probe | `03ffec23466f1679c6507440e005b688bfd36589019e01517c4237cfdb57ce3f` |
| inspection manifest | `24709d8f548702ac2ab1b32c731270c4282e316364891b1c0d5aecce43095ba1` |
| inspected matrix report | `bb89ebc74a557bef19ac2270953e60cab73f61c6a3e019be9c7c8516e9c49a83` |

Os tres `run_manifest.json` registram os hashes de input, config original/efetivo, ferramenta, stdout, stderr, projeto e imagens finais. Seus `runner_evidence_sha256` sao, respectivamente, `f1be5d390242358b7941c1a862861a93c39be7d48b0fc73ad759280909f001c4`, `644e419c3fcbf5c874122ac55235e4be49e3cf17bcac4f619c84d12f7d301662` e `4c24f8168f8216b1a4f8b345d013f7ba087ff96b9ba2e6a7a30188379b032b31`.

## Suites e diferencial

Executadas no venv correto antes do holdout:

- Style: 466 passed, 53 failed, 6 subtests; as 53 falhas ja existiam no baseline da Task 9; novas falhas: 0.
- Adjacente funcional: 731 passed, 15 failed; as 15 falhas ja existiam no baseline amplo; novas falhas: 0.
- Invariancia: 44 passed, 30 deselected.
- Suite ampla: 3640 passed, 249 failed, 33 skipped, 59 subtests; baseline 249 failed, depois 249 failed, novas 0, resolvidas 0.

Isso comprova ausencia de regressao diferencial nova, nao qualidade visual suficiente.

## Runtime antes e depois

| Contador | Baseline congelado | Holdout novo |
|---|---:|---:|
| projetos | 3 | 3 |
| rendered owners | 43 | 2 |
| visual profiles | 43 | 2 |
| profile coverage | 100% | 100% |
| raster contracts | 0 | 2 |
| raster-contract coverage | 0% | 100% |
| atributos aplicados | 0 | 4 |
| atributos abstidos | 559 | 30 |

O contrato raster deixou de estar ausente nos owners renderizados, mas o numero de owners efetivamente renderizados caiu para 2 e o probe bloqueou `style_colored_cards_calibration` por `no_rendered_owners`. Abstencoes: `no_text_evidence=17`, `attribute_confidence_below_threshold=5`, `functional_layout_owned=3`, e uma ocorrencia de cada uma das demais razoes documentadas no JSON do probe.

## Benchmark e thresholds

| Metrica | Exigido | Medido | Estado |
|---|---:|---:|---|
| font top-1 | >= 0,90 | 0,40 (8/20) | BLOCK |
| font top-3 | >= 0,98 | 0,00 (0/20) | BLOCK |
| menor categoria | >= 0,85 | ausente | BLOCK |
| speech GO rate | >= 0,95 | ausente | BLOCK |
| owner GO rate | >= 0,90 | ausente | BLOCK |
| fill Delta E mediana | <= 8 | ausente | BLOCK |
| fill Delta E p95 | <= 12 | ausente | BLOCK |
| safe containment | 1,00 | ausente | BLOCK |
| mismatch catastrofico | 0 | ausente | BLOCK |

Metricas auxiliares: fill classification precision 0,55 (11/20), stroke precision 0, shadow precision 0, round-trip 1,00 (10/10) e hard-negative abstention V2 shadow 1,00 (6/6). Metric Ausente bloqueia por contrato; nao e convertido em zero silencioso nem em passe.

## Matriz e inspecao owner-scoped

Cada categoria possui um target exato e denominador 1; todos os nove foram resolvidos por `page_id`, `owner_id`, `component_ids` e crop SHA-256. Nao houve page clamping.

| Categoria | Owner exato | Dimensao sheet | SHA-256 | Functional | Style | Observacao visual |
|---|---|---:|---|---|---|---|
| colored_card | `page_001/owner_p001_024cf5ddf5ea` | 2292x195 | `e2b3eed194407deea11a158f4ff68231b84488a8ed040e119a1310917350dd95` | NO-GO | NO-GO | Ingles intacto; evidencia vazia; nenhum PT-BR renderizado. |
| primary_ocr_omission | `page_001/owner_p001_0f39ce3adba0` | 2238x158 | `4f81a6ffc6c81c8606931d49846ffb12d23d1965ffbcbff8a66b94b944364053` | NO-GO | NO-GO | Omissao persiste; texto e contorno ingles permanecem. |
| table_ranking | `page_001/owner_p001_06c2c46ee27f` | 1572x172 | `0496ae150bfd0157e73a61a26454a97dcd1478855530c8c153588692502aec04` | NO-GO | NO-GO | Tabela ainda em ingles; fidelidade aparente e somente original preservado. |
| translucent_balloon | `page_001/owner_p001_0dc05817313b` | 2436x229 | `5df98c587a17f5c84c80a2302b5d6023ded58699cf400156d824d7b03894896e` | NO-GO | NO-GO | Ingles integral e nenhum raster PT-BR. |
| burst | `page_002/owner_p002_7227e479da0d` | 1002x123 | `44664b457fcc6fcfd7265df61ce28e3fbd96b197497685da37fb40ff5ba38a92` | NO-GO | NO-GO | Ingles permanece e corpo termina no meio das linhas. |
| cross_tile_owner | `page_002/owner_p002_d9afb32ef935` | 1254x163 | `87362b2c6dc9f5ef9e4d329dc78a7e50e91499222817df80356a818fb7952853` | NO-GO | NO-GO | Unico PT-BR, cortado no tile e com fonte/peso/escala incompatíveis. |
| white_balloon | `page_002/owner_p002_43752a8f8c08` | 1362x155 | `29a5c01a40d2ceb807d98e3fa4a959eac75a080b2d5ad4774786ccf325ec3e75` | NO-GO | NO-GO | Ingles permanece; corpo cortado pelo recorte. |
| dark_panel | `page_002/owner_p002_5a6976dbbf65` | 1902x243 | `d26ef1cea3c801b2de494a6d7079c34c93d98399ceeb87496257448ade81cde8` | NO-GO | NO-GO | Ingles permanece; requested/raster/final repetem a fonte. |
| text_over_art | `page_003/owner_p003_98e72af0685a` | 1986x154 | `be443e13dd0de2830aeb8e31ab128f753aa49d305645105063cba683aa6b073a` | NO-GO | NO-GO | Ingles permanece; safe-region laranja divide as linhas. |

Contact sheets: `N:\TraduzAI_style_runs\style_copy_v2_remediation_holdout_20260801_001546\matrix\contact_sheets`. A inspecao comparou `source | masks/evidence | requested | raster | final | safe-region` em escala nativa.

## Gates e bloqueios restantes

As tres entradas ficaram `Gate=BLOCK` e `Status=BLOCK`. Contratos agrupados:

- todas: `export_gate_blocked`, `final_language_contract`, `qa_integrity_failure`, `route_state_contract`, `unclassified_gate_issue`;
- cards e dark panels: `style_fidelity_gate_blocked`, `style_fidelity_high_confidence_mismatch`;
- dark panels: `source_coverage_contract`;
- runtime geral: `no_rendered_owners`.

Os logs de commit atomico mostram rollbacks por `render_contract_invalid: render was not completed` e `render_contract_invalid: style raster contract applied decision mismatch`. O fail-closed esta correto; a geracao do raster aplicavel e a cobertura funcional ainda nao estao.

Nenhuma regra especifica de pagina, obra ou capitulo foi adicionada. A remediacao usa contratos, thresholds, targets e gates sistemicos; o NO-GO observado tambem e reportado sistemicamente.

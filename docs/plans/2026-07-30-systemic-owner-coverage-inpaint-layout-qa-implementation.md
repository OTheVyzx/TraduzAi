# Systemic Owner Coverage, Inpaint, Layout, and QA Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminar sistemicamente texto-fonte residual, payload truncado ou duplicado, body repartido, máscara/inpaint incompletos, costura entre bands, dano à arte, layout desproporcional e falso `PASS` do QA em qualquer obra, capítulo, página ou band.

**Architecture:** O `OwnerGraph` page-global continua como autoridade semântica, mas passa a conservar e classificar todo o conjunto de evidências OCR. Reconciliation, máscara, inpaint, layout e composição executam uma transação atômica por owner em coordenadas de página. Um observador OCR final realmente independente confronta os bytes persistidos com toda a evidência-fonte, e o export gate falha fechado quando cobertura, residual, legibilidade, locale ou integridade do QA não forem comprovados.

**Tech Stack:** Python 3.12, dataclasses/typing, NumPy, OpenCV, PIL, PaddleOCR/Koharu adapters, pytest, Hypothesis/property tests, JSON/JSONL versionado e SHA-256.

---

## Regras obrigatórias de execução

- Use as skills `mangatl-dev`, `traduzai-ocr`, `traduzai-inpaint`, `traduzai-typesetting` e `traduzai-pipeline` conforme o owner de cada tarefa.
- Trabalhe no checkout atual e preserve integralmente o dirty worktree. Não use `reset`, `checkout`, `stash`, `clean`, `git add -A` nem crie worktree.
- Antes de cada patch, rode `git status --short -- <arquivos>` e `git diff -- <arquivos>`. Hunks preexistentes pertencem ao usuário.
- Aplique TDD estrito: teste RED pela razão esperada, implementação sistêmica mínima, teste GREEN, regressão próxima e commit restrito.
- Não aceite RED provocado apenas por erro de sintaxe, import incorreto ou fixture ausente depois da tarefa que cria o módulo.
- Faça staging interativo (`git add -p --`) para todo arquivo preexistente e `git add --` apenas para arquivos novos criados integralmente pela task. Antes do commit, rode `git diff --cached --check`, `git diff --cached -- <paths>` e `git diff --cached --name-only`; nenhum hunk preexistente do usuário pode entrar.
- Checkpoints são automáticos. Continue enquanto estiverem verdes; não pause para pedir confirmação.
- Nenhuma regra de produção pode depender de título, obra, capítulo, página, coordenada literal, texto literal ou `band_id`.
- Casos reais podem aparecer somente em fixtures, manifests e relatórios que não são importados pelo runtime.
- Bodies não podem ser abreviados, resumidos, truncados, duplicados ou semanticamente repartidos para caber.
- Bands são apenas projeção/scheduling; não podem arbitrar payload, máscara, fundo, render ou QA.
- Cópia de estilo está fora deste plano. Fonte, peso, cor, stroke, glow, sombra e gradiente não entram no veredito funcional.
- Use um `output-root` novo para cada execução visual. Nunca reutilize o v10 como prova de correção.
- Não declare sucesso por `exit 0`, `success=true`, teste unitário isolado ou relatório vazio. O resultado final exige gate não bloqueado, hashes atuais e inspeção nativa aprovada.

## Baseline e escopo

O design aprovado está no commit `976141bd` e em `docs/plans/2026-07-30-systemic-owner-coverage-inpaint-layout-qa-design.md`.

O checkout conhecido no planejamento é `Troca_de_motores`, muito à frente do remoto e com milhares de exclusões locais preexistentes sob `DEBUGM/runs`. Os arquivos de produção relevantes estavam sem diff; três documentos de plano não relacionados já existiam como untracked. Preserve tudo isso.

O v10 em `.codex-tmp/page_owner_systemic_validation_v10_20260730` é baseline **NO-GO**, apesar de `docs/reports/2026-07-25-page-owner-systemic-validation.md` dizer `GO`. Evidência mínima que os novos testes precisam congelar:

- observações fortes contêm `TOTAL PURCHASE AMOUNT 200MILLION`, mas o owner seleciona apenas `TOTAL PURCHASE AMOUNT:`;
- a máscara cobre somente o cabeçalho e deixa `200 MILLION` visível;
- a página registra 25 blocos detectados, zero OCR records, `observation_complete=true` e gate `PASS`;
- owners de `Sim.`, `Vamos...` e `Grand Finale` renderizam com 16 px contra ink fonte de 36, 40 e 66 px;
- o QA de underfill legado não roda no caminho owner verificado.

## Checkpoints contínuos

1. **Checkpoint A — evidência e owner:** Tasks 1–3. O payload completo e a ambiguidade devem ser decididos antes de qualquer mutação visual.
2. **Checkpoint B — cleanup atômico:** Tasks 4–7. Toda fonte autorizada deve ser limpa uma vez, sem costura nem arte danificada, e o conteúdo deve estar em PT-BR.
3. **Checkpoint C — layout proporcional:** Tasks 8–11. Todo glyph patch deve conter prova mensurável de legibilidade e contenção.
4. **Checkpoint D — QA independente:** Tasks 12–15. QA inconclusivo, fonte residual ou contrato ausente bloqueiam o export.
5. **Checkpoint E — prova cross-work:** Tasks 16–17. Suites, matriz nova, inspeção nativa e relatório honesto.

### Task 1: Fixar o baseline funcional e ampliar o corpus genérico

**Files:**

- Modify: `pipeline/tests/fixtures/systemic_visual_contracts/owner_cases.json`
- Modify: `pipeline/tests/test_systemic_owner_contract_manifest.py`
- Create: `pipeline/tests/fixtures/owner_visual_matrix/functional_matrix.json`
- Create: `pipeline/tests/fixtures/owner_visual_matrix/inputs.json`
- Create: `pipeline/tests/fixtures/owner_visual_matrix/configs/colored_cards.json`
- Create: `pipeline/tests/fixtures/owner_visual_matrix/configs/cross_tile_owners.json`
- Create: `pipeline/tests/fixtures/owner_visual_matrix/configs/dark_panels.json`
- Read only: `.codex-tmp/page_owner_systemic_validation_v10_20260730/**`

**Step 1: Inspecionar o checkout e registrar baseline focado**

```powershell
git status --short --branch
git diff -- pipeline docs/plans
New-Item -ItemType Directory -Force .codex-tmp | Out-Null
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_systemic_owner_contract_manifest.py `
  pipeline/tests/test_owner_reconcile.py `
  pipeline/tests/test_owner_mask.py `
  pipeline/tests/test_owner_layout.py `
  pipeline/tests/test_final_pixel_qa.py `
  pipeline/tests/test_final_pixel_export_gate.py `
  -q | Tee-Object .codex-tmp\owner_functional_baseline_20260730.txt
```

Expected: registrar os nodeids atuais. Não corrigir falhas históricas fora do plano.

**Step 2: Escrever os testes RED do manifest**

Acrescente categorias genéricas ao conjunto obrigatório:

```python
REQUIRED_CATEGORIES |= {
    "tight_truncation_vs_complete_numeric_body",
    "cross_band_background_continuity",
    "proportional_layout_underfill",
    "incomplete_final_observation",
}


def test_complete_numeric_case_contains_competing_full_and_truncated_evidence():
    case = next(
        item for item in _load_manifest()["cases"]
        if item["category"] == "tight_truncation_vs_complete_numeric_body"
    )
    assert {item["coverage_kind"] for item in case["observations"]} == {
        "complete", "truncated"
    }
    assert case["expected"]["selected_coverage_kind"] == "complete"
    assert case["expected"]["production_exception_count"] == 0
```

Exija também um manifest de matriz versionado com `entries[]`, split `calibration|holdout`, `entry_id`, `work_id`, `chapter_id`, `config_path`, `input_key`, hashes esperados e categorias. Cada `config_path` precisa existir dentro da fixture versionada. `inputs.json` mapeia cada `input_key` para variável de ambiente/resolver, tipo de input e SHA-256 esperado; não contém path absoluto nem output temporário. Os três configs guardam somente opções canônicas do pipeline e referenciam `input_key`.

**Step 3: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_systemic_owner_contract_manifest.py -q
```

Expected: FAIL porque as quatro categorias e o manifest versionado ainda não existem.

**Step 4: Adicionar somente as fixtures genéricas**

Inclua casos sem nomes de obras, textos de personagens ou coordenadas do v10. O caso numérico deve possuir uma observação curta de confiança maior/IoU mais apertado e pelo menos duas observações completas corroboradas. O caso de costura usa fundo com gradiente contínuo atravessando duas projeções. O caso de QA registra blocos detectados e OCR vazio.

Não implemente runtime nesta task. Todos os testes adicionados aqui devem ficar GREEN no mesmo commit.

**Step 5: Confirmar GREEN**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_systemic_owner_contract_manifest.py -q
```

Expected: PASS, inclusive splits calibration/holdout, existência/hash dos três configs, cobertura de todos os `input_key` e ausência de paths `.codex-tmp`/absolutos.

**Step 6: Commit**

```powershell
git add -p -- pipeline/tests/fixtures/systemic_visual_contracts/owner_cases.json pipeline/tests/test_systemic_owner_contract_manifest.py
git add -- pipeline/tests/fixtures/owner_visual_matrix/functional_matrix.json pipeline/tests/fixtures/owner_visual_matrix/inputs.json pipeline/tests/fixtures/owner_visual_matrix/configs/colored_cards.json pipeline/tests/fixtures/owner_visual_matrix/configs/cross_tile_owners.json pipeline/tests/fixtures/owner_visual_matrix/configs/dark_panels.json
git diff --cached --check
git diff --cached -- pipeline/tests/fixtures/systemic_visual_contracts/owner_cases.json pipeline/tests/test_systemic_owner_contract_manifest.py pipeline/tests/fixtures/owner_visual_matrix
git commit -m "test: characterize systemic owner visual failures"
```

### Task 2: Criar comparação e ledger de evidências compartilhados

**Files:**

- Create: `pipeline/ownership/evidence.py`
- Create: `pipeline/tests/test_owner_evidence.py`
- Modify: `pipeline/ownership/artifacts.py`
- Test: `pipeline/tests/test_owner_debug_artifacts.py`

**Step 1: Escrever testes RED para normalização e dominância**

```python
from ownership.evidence import (
    build_source_evidence_ledger,
    normalize_evidence_tokens,
    safely_dominates,
)


def test_letter_digit_boundaries_normalize_200million():
    assert normalize_evidence_tokens("200MILLION") == ("200", "million")


def test_complete_observation_safely_dominates_coherent_truncation():
    assert safely_dominates(
        complete=_observation("TOTAL PURCHASE AMOUNT 200MILLION", (10, 10, 90, 70)),
        truncated=_observation("TOTAL PURCHASE AMOUNT", (10, 10, 90, 35)),
        same_region=True,
        corroboration_count=2,
    )


def test_long_adjacent_contamination_never_dominates():
    assert not safely_dominates(
        complete=_observation("TOTAL PURCHASE AMOUNT PLAYER NAME", (10, 10, 180, 70)),
        truncated=_observation("TOTAL PURCHASE AMOUNT", (10, 10, 90, 35)),
        same_region=False,
        corroboration_count=1,
    )
```

Também exija que cada ledger row possua `observation_id`, `owner_id`, `component_coverage`, `observation_precision`, `material`, `disposition`, `reason` e `dominant_observation_id`.

**Step 2: Caracterizar a primeira divergência real**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_owner_evidence.py -q
```

Expected: FAIL com `ModuleNotFoundError: ownership.evidence`.

**Step 3: Implementar API pura mínima**

Crie um dataclass imutável sem alterar a identidade do `OwnerGraph`:

```python
@dataclass(frozen=True)
class SourceEvidenceRecord:
    observation_id: str
    owner_id: str | None
    component_ids: tuple[str, ...]
    normalized_tokens: tuple[str, ...]
    component_coverage: float
    observation_precision: float
    material: bool
    disposition: str
    reason: str | None = None
    dominant_observation_id: str | None = None
```

Implemente `normalize_evidence_tokens()`, métricas geométricas, `safely_dominates()` e `build_source_evidence_ledger()`. O ledger é derivado do grafo e publicado para debug; não entra no hash semântico do owner.

**Step 4: Publicar o ledger nos artefatos**

Em `pipeline/ownership/artifacts.py`, escreva `source_evidence_ledger.jsonl` no estágio de normalização/router, incluindo versão e hashes de entrada. Nenhum estágio lê o arquivo de debug de volta.

**Step 5: Confirmar GREEN e regressão de artefatos**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_owner_evidence.py `
  pipeline/tests/test_owner_debug_artifacts.py -q
```

Expected: PASS.

**Step 6: Commit**

```powershell
git add -- pipeline/ownership/evidence.py pipeline/tests/test_owner_evidence.py
git add -p -- pipeline/ownership/artifacts.py pipeline/tests/test_owner_debug_artifacts.py
git diff --cached --check
git commit -m "feat: add auditable source evidence ledger"
```

### Task 3: Fazer evidência completa vencer truncamento seguro

**Files:**

- Modify: `pipeline/ownership/reconcile.py:131-385`
- Read/verify only: `pipeline/ownership/ocr_adapter.py:339-427`
- Modify: `pipeline/ownership/translation.py:28-180`
- Modify: `pipeline/strip/run.py`
- Test: `pipeline/tests/test_owner_reconcile.py`
- Test: `pipeline/tests/test_owner_reconcile_properties.py`
- Test: `pipeline/tests/test_owner_enforcement.py`
- Test: `pipeline/tests/test_owner_translation.py`
- Test: `pipeline/tests/test_strip_owner_control_plane.py`

**Step 1: Escrever testes RED de seleção**

```python
def test_complete_consensus_reading_beats_tighter_truncated_reading(): ...
def test_complete_reading_selection_is_permutation_invariant(): ...
def test_incompatible_non_dominated_readings_fail_to_review(): ...
def test_numeric_footer_and_header_reach_translator_as_one_payload(): ...
def test_legacy_rejection_reason_remains_evidence_not_destructive_filter(): ...
def test_repeated_band_observations_never_create_duplicate_owner(): ...
def test_owner_payload_is_unchanged_between_reconcile_and_translation(): ...
```

O primeiro teste deve criar uma região inicialmente subrecortada, uma leitura curta com confiança/IoU maiores e duas leituras completas coerentes contendo `200MILLION`.

**Step 2: Confirmar RED pela seleção truncada atual**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_owner_reconcile.py::test_complete_consensus_reading_beats_tighter_truncated_reading `
  pipeline/tests/test_owner_reconcile_properties.py::test_complete_reading_selection_is_permutation_invariant `
  pipeline/tests/test_owner_enforcement.py::test_numeric_footer_and_header_reach_translator_as_one_payload `
  pipeline/tests/test_strip_owner_control_plane.py::test_repeated_band_observations_never_create_duplicate_owner `
  pipeline/tests/test_owner_translation.py::test_owner_payload_is_unchanged_between_reconcile_and_translation -q
```

Expected: o owner atual seleciona somente a observação curta ou muda com a ordem.

**Step 3: Substituir ranking geométrico insuficiente**

Em `_observation_support_rank()` e `_select_observations()`:

- rankeie cobertura semântica/component coverage antes de IoU;
- use observation precision como veto a contaminação, não como motivo para truncar;
- use `safely_dominates()` para marcar truncados coerentes;
- exija corroboração ou geometria suficiente para a leitura completa;
- marque leituras máximas incompatíveis como `ambiguous_reading` e owner `review_required`;
- preserve todas as observações no grafo com motivo explícito.

**Step 4: Expandir geometria antes da máscara**

Faça `_expand_owned_components_to_selected_ink()` consumir todos os polígonos das observações completas selecionadas. A expansão deve ser determinística e não atravessar componentes/roles alheios.

Em `_resolve_page_owner_graphs_once()` e `owners_to_translation_page()`, reafirme que IDs de band/tile são apenas provenance: um owner reconciliado gera exatamente uma entrada no tradutor, e `merge_owner_translations()` devolve o mesmo payload ao mesmo owner. Fragmentos podem permanecer como evidência, mas não podem criar chamadas ou renders adicionais.

**Step 5: Confirmar GREEN e properties**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_owner_reconcile.py `
  pipeline/tests/test_owner_reconcile_properties.py `
  pipeline/tests/test_owner_enforcement.py `
  pipeline/tests/test_owner_translation.py `
  pipeline/tests/test_strip_owner_control_plane.py -q
```

Expected: PASS, inclusive para permutações de tiles e observações.

**Step 6: Commit**

```powershell
git add -p -- pipeline/ownership/reconcile.py pipeline/ownership/translation.py pipeline/strip/run.py pipeline/tests/test_owner_reconcile.py pipeline/tests/test_owner_reconcile_properties.py pipeline/tests/test_owner_enforcement.py pipeline/tests/test_owner_translation.py pipeline/tests/test_strip_owner_control_plane.py
git diff --cached --check
git commit -m "fix: prefer corroborated complete owner evidence"
```

### Task 4: Exigir cobertura integral da action mask

**Files:**

- Modify: `pipeline/inpainter/owner_mask.py:622-1545`
- Modify: `pipeline/strip/process_bands.py:9724-10450`
- Test: `pipeline/tests/test_owner_mask.py`
- Test: `pipeline/tests/test_strip_owner_control_plane.py`
- Test: `pipeline/tests/test_owner_atomic_execution.py`

**Step 1: Escrever testes RED**

```python
def test_selected_multiline_observation_authorizes_every_line_polygon(): ...
def test_missing_second_line_mask_revokes_entire_owner(): ...
def test_unselected_material_complete_evidence_is_audited_for_cleanup(): ...
def test_mask_plan_never_falls_back_to_overbroad_owner_bbox(): ...
def test_expected_line_identity_missing_before_planner_still_revokes_owner(): ...
```

O fixture deve ter duas linhas separadas; remova deliberadamente pixels da segunda linha da máscara esperada.

**Step 2: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_owner_mask.py::test_selected_multiline_observation_authorizes_every_line_polygon `
  pipeline/tests/test_owner_atomic_execution.py::test_missing_second_line_mask_revokes_entire_owner -q
```

Expected: a implementação atual aceita a máscara parcial.

**Step 3: Implementar o contrato de cobertura**

Antes de rasterizar, derive o conjunto obrigatório de identidades `(observation_id, line_index)` de todas as observações selecionadas. `_owner_component_glyph_raster()` deve emitir uma evidência separada por identidade; não associe todas as linhas a `selected[0]`. Passe ao planner tanto `expected_line_ids` quanto as evidências efetivamente geradas.

Faça `build_owner_mask_plan()` validar igualdade entre IDs esperados e IDs cobertos após aplicar proteção e devolver, além das máscaras, um resumo imutável:

```python
{
    "selected_observation_ids": [...],
    "expected_line_ids": [["observation_full", 0], ["observation_full", 1]],
    "covered_line_ids": [["observation_full", 0], ["observation_full", 1]],
    "expected_line_polygon_count": 2,
    "covered_line_polygon_count": 2,
    "uncovered_source_ink_pixels": 0,
    "coverage_complete": True,
}
```

Use glyph/polygon evidence owner-scoped. Ausência de geração ou cobertura em qualquer identidade esperada marca o owner `review_required`. Não use bbox amplo para forçar sucesso.

**Step 4: Integrar no executor atômico**

`execute_owner_page_graph()` e `apply_atomic_owner_execution()` só chamam `execute_owner_inpaint()` quando `coverage_complete=True` e o hash do geometry/mask plan corresponde ao owner atual.

**Step 5: Confirmar GREEN**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_owner_mask.py `
  pipeline/tests/test_strip_owner_control_plane.py `
  pipeline/tests/test_owner_atomic_execution.py -q
```

Expected: PASS.

**Step 6: Commit**

```powershell
git add -p -- pipeline/inpainter/owner_mask.py pipeline/strip/process_bands.py pipeline/tests/test_owner_mask.py pipeline/tests/test_strip_owner_control_plane.py pipeline/tests/test_owner_atomic_execution.py
git diff --cached --check
git commit -m "fix: require complete owner mask coverage"
```

### Task 5: Tornar residual e proteção parte obrigatória da transação

**Files:**

- Modify: `pipeline/inpainter/owner_mask.py:1472-1625`
- Modify: `pipeline/ownership/model.py:251-317`
- Modify: `pipeline/strip/process_bands.py:8682-8865`
- Reuse: `pipeline/qa/inpaint_residual.py`
- Test: `pipeline/tests/test_owner_mask.py`
- Test: `pipeline/tests/test_owner_atomic_execution.py`
- Test: `pipeline/tests/test_owner_compositor_properties.py`
- Test: `pipeline/tests/test_owner_enforcement.py`

**Step 1: Escrever testes RED**

```python
def test_owner_commit_rejects_unverified_residual_score(): ...
def test_owner_commit_rejects_residual_above_profile_threshold(): ...
def test_adjacent_protected_icon_survives_complete_multiline_cleanup(): ...
def test_connected_line_art_crossing_support_is_never_authorized(): ...
def test_changed_pixels_remain_subset_of_action_mask(): ...
def test_residual_threshold_and_evidence_are_part_of_mutation_hash_chain(): ...
```

**Step 2: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_owner_atomic_execution.py::test_owner_commit_rejects_unverified_residual_score `
  pipeline/tests/test_owner_mask.py::test_adjacent_protected_icon_survives_complete_multiline_cleanup -q
```

Expected: `residual_score=None` ainda pode chegar ao commit ou a proteção negativa não é materializada.

**Step 3: Implementar verificação obrigatória**

Crie `_owner_protected_evidence()` em `process_bands.py`. Ele materializa proteção a partir de glyphs/componentes de outros owners, disposições `preserve`, evidência explícita de ícones/bordas e foreground conectado que atravessa o suporte OCR. O helper devolve máscara, provenance e confidence; separação inconclusiva revoga o owner.

- Execute residual detection no crop owner-scoped após inpaint.
- Adicione campos canônicos a `OwnerMutation`: `residual_score`, `residual_verified`, `residual_threshold`, `residual_method`, `residual_evidence_sha256` e `residual_flags`.
- Inclua threshold/método/evidence hash na cadeia de hashes da mutação; o commit não depende de configuração externa mutável.
- Trate `None`, NaN, artifact hash incorreto e score acima do perfil como falha.
- Se a separação não for confiável, devolva revisão sem mutar pixels.

**Step 4: Endurecer o commit atômico**

Antes de compor, exija:

```text
changed_outside_owner_pixels == 0
protected_art_changed_pixels == 0
residual_verified == true
residual_score <= threshold
residual_evidence_sha256 matches current owner mutation
component_geometry_verified == true
```

**Step 5: Confirmar GREEN e properties**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_owner_mask.py `
  pipeline/tests/test_owner_atomic_execution.py `
  pipeline/tests/test_owner_compositor_properties.py `
  pipeline/tests/test_owner_enforcement.py -q
```

Expected: PASS.

**Step 6: Commit**

```powershell
git add -p -- pipeline/inpainter/owner_mask.py pipeline/ownership/model.py pipeline/strip/process_bands.py pipeline/tests/test_owner_mask.py pipeline/tests/test_owner_atomic_execution.py pipeline/tests/test_owner_compositor_properties.py pipeline/tests/test_owner_enforcement.py
git diff --cached --check
git commit -m "fix: verify owner cleanup residuals and protected art"
```

### Task 6: Impedir costura de cor e execução duplicada entre bands

**Files:**

- Modify: `pipeline/strip/process_bands.py:8682-8865,9953-10450`
- Modify: `pipeline/inpainter/owner_mask.py`
- Modify: `pipeline/compositor/owner_compositor.py`
- Test: `pipeline/tests/test_strip_owner_composition_integration.py`
- Test: `pipeline/tests/test_owner_compositor_properties.py`
- Test: `pipeline/tests/test_owner_atomic_execution.py`

**Step 1: Escrever testes RED de invariância por partição**

```python
def test_cross_band_gradient_owner_inpaints_once_in_page_space(): ...
def test_owner_result_is_identical_for_one_two_or_four_tile_partitions(): ...
def test_projection_never_estimates_independent_background_color(): ...
def test_one_owner_has_one_cleanup_and_one_render_write(): ...
```

O fixture deve usar fundo com gradiente/ruído contínuo e um owner atravessando a fronteira de duas bands.

**Step 2: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_strip_owner_composition_integration.py::test_cross_band_gradient_owner_inpaints_once_in_page_space `
  pipeline/tests/test_owner_compositor_properties.py::test_owner_result_is_identical_for_one_two_or_four_tile_partitions -q
```

Expected: compare os bytes em `08_inpaint`, `09_typeset` e `10_copyback_reassemble` para localizar o primeiro estágio divergente. O owner enforce atual já pretende executar page-space: o teste pode nascer GREEN. Nesse caso, mantenha a regressão e não altere produção sem RED.

**Step 3: Corrigir somente o primeiro estágio divergente, se houver RED**

- Construa ação/proteção uma vez no canvas da página.
- Execute inpaint/background normalization uma vez por owner.
- Projete o resultado pronto para tiles apenas quando necessário para scheduling/debug.
- Proíba qualquer fill/color estimation no caminho projection.
- Faça o compositor rejeitar `write_counts[owner_id] != 1` por fase.

Se a caracterização provar que inpaint/compositor já são invariantes, faça commit apenas dos testes de regressão e registre no commit que nenhuma mudança produtiva foi necessária.

**Step 4: Confirmar igualdade de pixels e hashes**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_strip_owner_composition_integration.py `
  pipeline/tests/test_owner_compositor_properties.py `
  pipeline/tests/test_owner_atomic_execution.py -q
```

Expected: PASS e SHA-256 final idêntico para 1/2/4 partições.

**Step 5: Commit**

```powershell
git add -p -- pipeline/strip/process_bands.py pipeline/inpainter/owner_mask.py pipeline/compositor/owner_compositor.py pipeline/tests/test_strip_owner_composition_integration.py pipeline/tests/test_owner_compositor_properties.py pipeline/tests/test_owner_atomic_execution.py
git diff --cached --check
git commit -m "fix: keep owner cleanup atomic across band projections"
```

### Task 7: Adicionar conformidade determinística de locale PT-BR

**Files:**

- Create: `pipeline/translator/locale_policy.py`
- Create: `pipeline/tests/test_translation_locale_policy.py`
- Modify: `pipeline/translator/translate.py`
- Modify: `pipeline/qa/translation_qa.py`
- Modify: `pipeline/ownership/translation.py`
- Test: `pipeline/tests/test_owner_translation.py`
- Test: `pipeline/tests/test_translate_context.py`

**Step 1: Escrever testes RED**

```python
def test_pt_br_rejects_pt_pt_mil_milhoes_scale(): ...
def test_pt_br_accepts_bilhoes_with_decimal_comma(): ...
def test_locale_policy_preserves_numeric_meaning(): ...
def test_translate_pages_preserves_pt_br_after_backend_language_normalization(): ...
def test_deterministic_mismatch_blocks_without_silent_rewrite(): ...
def test_locale_validator_is_not_imported_by_style_modules(): ...
```

**Step 2: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_translation_locale_policy.py -q
```

Expected: FAIL porque `translator.locale_policy` não existe.

**Step 3: Implementar regras configuráveis e conservadoras**

Crie uma API sem correção automática:

```python
def validate_target_locale(
    *, source_text: str, target_text: str, target_locale: str
) -> LocaleValidationResult:
    ...
```

As regras devem separar equivalência numérica de escolha lexical. Apenas mismatch determinístico de alta confiança é crítico; caso ambíguo vira review. Não codifique nomes de obras ou textos completos.

**Step 4: Preservar locale separado do código do backend**

Em `translate.py`, mantenha `target_locale="pt-BR"` mesmo quando o backend Google/Ollama precisar do código `pt`. Propague o locale por `translate_pages`, `_translate_with_google`, `_translate_google_single_page`, `_translate_with_ollama` e `translate_single_block`. Valide após `_postprocess` e antes de devolver o payload ao owner.

`owners_to_translation_page()` e `translation_qa` devem anexar issues e impedir `translation_ready` quando houver blocker. O texto não deve ser reescrito silenciosamente.

**Step 5: Confirmar GREEN**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_translation_locale_policy.py `
  pipeline/tests/test_translate_context.py `
  pipeline/tests/test_owner_translation.py `
  pipeline/tests/test_translation_qa.py -q
```

Expected: PASS.

**Step 6: Commit**

```powershell
git add -- pipeline/translator/locale_policy.py pipeline/tests/test_translation_locale_policy.py
git add -p -- pipeline/translator/translate.py pipeline/qa/translation_qa.py pipeline/ownership/translation.py pipeline/tests/test_translate_context.py pipeline/tests/test_owner_translation.py pipeline/tests/test_translation_qa.py
git diff --cached --check
git commit -m "fix: enforce pt br translation locale contract"
```

### Task 8: Criar o contrato puro de qualidade proporcional

**Files:**

- Create: `pipeline/typesetter/owner_render_quality.py`
- Create: `pipeline/tests/test_owner_render_quality.py`

**Step 1: Escrever os testes RED de caracterização**

```python
def test_source_scale_rejects_16px_render_for_36px_source_ink(): ...
def test_source_scale_rejects_16px_render_for_66px_source_ink(): ...
def test_trusted_container_rejects_eight_percent_height_occupancy(): ...
def test_proportional_render_accepts_source_scale_between_075_and_135(): ...
def test_missing_owner_render_quality_metrics_fail_closed(): ...
def test_render_quality_canonical_round_trip_is_immutable(): ...
def test_multiline_quality_uses_median_core_ink_height_per_line(): ...
def test_render_with_pixels_outside_curved_safe_polygon_is_rejected(): ...
```

Use valores neutros equivalentes aos casos observados: fonte 28/36/40/66 px, render 16/16/16/16 px e ocupação 0,06–0,15.

**Step 2: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_owner_render_quality.py -q
```

Expected: FAIL por módulo ausente.

**Step 3: Implementar a API pura mínima**

```python
@dataclass(frozen=True)
class OwnerRenderQuality:
    schema_version: int
    status: str
    font_size_final: int
    minimum_legible_font_px: int
    source_ink_height_px: float | None
    render_ink_height_px: int
    source_x_height_px: float | None
    render_x_height_px: float
    source_scale_ratio: float | None
    x_height_ratio: float | None
    rendered_line_core_heights_px: tuple[int, ...]
    safe_height_occupancy: float
    safe_area_occupancy: float
    wrapped_line_count: int
    containment_status: str
    outside_safe_pixels: int
    page_width: int
    page_height: int
    reasons: tuple[str, ...]


def evaluate_owner_render_quality(
    *,
    render_bbox: list[int],
    safe_bbox: list[int],
    safe_mask: np.ndarray,
    glyph_core_mask: np.ndarray,
    glyph_pixels: int,
    font_size_final: int,
    minimum_legible_font_px: int,
    source_ink_heights_px: tuple[int, ...],
    source_x_heights_px: tuple[float, ...],
    source_evidence_confidence: float,
    page_width: int,
    page_height: int,
    translated_text: str,
    layout_profile: str,
    trusted_container: bool,
) -> OwnerRenderQuality:
    ...
```

Implemente serialização canônica `to_dict()`/`from_dict()` com tuples/reasons imutáveis. Meça por linha a altura do core ink e use a mediana, não apenas o bbox agregado. Calcule `outside_safe_pixels` contra a safe mask/polygon real; bbox dentro de balloon curvo não é prova de contenção. Regras mínimas:

- fonte abaixo do mínimo: `below_minimum`;
- fonte confiável e razão `< 0.75`: `under_source_scale`;
- razão `> 1.35`: `over_source_scale`;
- sem fonte confiável, container verificado abaixo da ocupação por perfil/comprimento: `underfilled`;
- qualquer core glyph fora da safe polygon: `outside_safe_region`;
- métricas ausentes/inválidas: `invalid`;
- nenhum cálculo altera texto, quebra semântica ou estilo.

**Step 4: Confirmar GREEN**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_owner_render_quality.py -q
```

Expected: PASS.

**Step 5: Commit**

```powershell
git add -- pipeline/typesetter/owner_render_quality.py pipeline/tests/test_owner_render_quality.py
git diff --cached --check
git commit -m "feat: add proportional owner render quality contract"
```

### Task 9: Derivar escala fonte da geometria OCR selecionada

**Files:**

- Modify: `pipeline/strip/process_bands.py:9780-9952`
- Modify: `pipeline/layout/balloon_layout.py:47-949`
- Test: `pipeline/tests/test_owner_enforcement.py`
- Test: `pipeline/tests/test_balloon_layout_shared_regions.py`
- Test: `pipeline/tests/test_owner_layout.py`

**Step 1: Escrever testes RED**

```python
def test_owner_layout_region_derives_source_ink_height_from_selected_polygon(): ...
def test_owner_layout_ignores_overbroad_container_height_as_source_scale(): ...
def test_connected_owner_preserves_source_scale_evidence_per_region(): ...
def test_owner_layout_source_evidence_is_permutation_stable(): ...
```

**Step 2: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_owner_enforcement.py::test_owner_layout_region_derives_source_ink_height_from_selected_polygon `
  pipeline/tests/test_balloon_layout_shared_regions.py::test_connected_owner_preserves_source_scale_evidence_per_region `
  pipeline/tests/test_owner_layout.py::test_owner_layout_source_evidence_is_permutation_stable -q
```

Expected: os campos de escala fonte não existem.

**Step 3: Implementar derivação robusta**

Em `_owner_layout_regions()`:

- use somente polígonos não degenerados de observações selecionadas pertencentes ao componente;
- derive alturas de linha/core x-height e medianas robustas;
- propague `source_ink_heights_px`, `source_x_heights_px`, medianas correspondentes, `source_scale_evidence_confidence` e IDs usados;
- preserve evidência por lóbulo conectado;
- não infira fonte, peso, stroke ou efeito.

Atualize os campos visuais permitidos em `balloon_layout.py` e valide inteiros positivos.

**Step 4: Confirmar GREEN**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_owner_enforcement.py `
  pipeline/tests/test_balloon_layout_shared_regions.py `
  pipeline/tests/test_owner_layout.py -q
```

Expected: PASS.

**Step 5: Commit**

```powershell
git add -p -- pipeline/strip/process_bands.py pipeline/layout/balloon_layout.py pipeline/tests/test_owner_enforcement.py pipeline/tests/test_balloon_layout_shared_regions.py pipeline/tests/test_owner_layout.py
git diff --cached --check
git commit -m "feat: derive owner source scale from OCR ink geometry"
```

### Task 10: Fazer o renderer buscar um fit proporcional

**Files:**

- Modify: `pipeline/typesetter/renderer.py:11706-11808,13561-13725,14823-15350`
- Test: `pipeline/tests/test_owner_layout.py`
- Test: `pipeline/tests/test_typesetting_renderer.py`

**Step 1: Escrever testes RED**

```python
def test_owner_short_body_grows_beyond_default_24px_to_preserve_source_scale(): ...
def test_owner_renderer_rejects_underfilled_fit_inside_safe_box(): ...
def test_owner_renderer_does_not_grow_above_source_scale_ceiling(): ...
def test_connected_owner_requires_common_proportional_font(): ...
def test_impossible_proportional_owner_fit_rolls_back_to_review(): ...
def test_owner_body_is_never_split_or_truncated_to_fit(): ...
```

**Step 2: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_owner_layout.py::test_owner_short_body_grows_beyond_default_24px_to_preserve_source_scale `
  pipeline/tests/test_typesetting_renderer.py::TypesettingRendererTests::test_owner_renderer_rejects_underfilled_fit_inside_safe_box -q
```

Expected: o teto implícito de 24 px ou a ausência de underfill mantém resultado inadequado como `ok`.

**Step 3: Implementar busca proporcional**

- Remova o teto implícito de 24 px quando não houver bounds explícitos.
- Derive intervalo pelo container, página e mínimo legível, com máximo global conservador de 96 px.
- Renderize candidatos temporários e meça o bbox real do glyph mask.
- Use `evaluate_owner_render_quality()` para aceitar somente candidatos `ok`.
- Priorize razão próxima de 1,0 e contenção integral.
- Em conectados, exija fonte comum e contrato `ok` para todos os lobos.
- Se nenhum candidato servir, retorne `below_proportional_legibility`, `render_completed=False` e revisão atômica.

**Step 4: Confirmar GREEN e regressão próxima**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_owner_layout.py `
  pipeline/tests/test_typesetting_renderer.py -q
```

Expected: PASS; nenhum corpo alterado para obter fit.

**Step 5: Commit**

```powershell
git add -p -- pipeline/typesetter/renderer.py pipeline/tests/test_owner_layout.py pipeline/tests/test_typesetting_renderer.py
git diff --cached --check
git commit -m "fix: enforce proportional layout for page owners"
```

### Task 11: Propagar o contrato de render até os artefatos finais

**Files:**

- Modify: `pipeline/ownership/model.py:319-354`
- Modify: `pipeline/typesetter/renderer.py`
- Modify: `pipeline/strip/process_bands.py:9953-10450`
- Modify: `pipeline/ownership/artifacts.py`
- Modify: `pipeline/main.py:7240-7395`
- Modify if schema requires: `pipeline/schema/project_schema_v12.py`
- Modify if schema requires: `pipeline/schema/project_schema_v12.json`
- Test: `pipeline/tests/test_owner_compositor.py`
- Test: `pipeline/tests/test_strip_owner_composition_integration.py`
- Test: `pipeline/tests/test_project_writer.py`
- Test: `pipeline/tests/test_render_plan_trace_integrity.py`
- Test: `pipeline/tests/test_main_emit.py`

**Step 1: Escrever testes RED**

```python
def test_owner_glyph_patch_requires_render_quality_contract(): ...
def test_atomic_owner_commit_rejects_missing_render_quality_contract(): ...
def test_owner_render_quality_round_trips_through_project_writer(): ...
def test_render_plan_final_preserves_owner_render_quality_contract(): ...
def test_main_debug_artifact_preserves_owner_render_quality_metrics(): ...
```

**Step 2: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_owner_compositor.py::test_owner_glyph_patch_requires_render_quality_contract `
  pipeline/tests/test_render_plan_trace_integrity.py::test_render_plan_final_preserves_owner_render_quality_contract -q
```

Expected: `OwnerGlyphPatch` não possui o contrato.

**Step 3: Ampliar o modelo e round-trip**

Adicione a `OwnerGlyphPatch` usando o tipo frozen criado na Task 8:

```python
render_quality_contract: OwnerRenderQuality
```

Serialize somente na fronteira de artefatos via `to_dict()`. Owner renderizado sem contrato `status=ok` não pode ter `render_completed=True`.

Propague integralmente para `text_layers`, `render_layout_contract`, `project.json`, `09_typeset/render_plan_final.jsonl` e artefatos owner. Não use `fit_status=ok` como substituto.

**Step 4: Confirmar GREEN**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_owner_compositor.py `
  pipeline/tests/test_strip_owner_composition_integration.py `
  pipeline/tests/test_project_writer.py `
  pipeline/tests/test_render_plan_trace_integrity.py `
  pipeline/tests/test_main_emit.py -q
```

Expected: PASS.

**Step 5: Commit somente arquivos realmente alterados**

```powershell
git add -p -- pipeline/ownership/model.py pipeline/typesetter/renderer.py pipeline/strip/process_bands.py pipeline/ownership/artifacts.py pipeline/main.py pipeline/tests/test_owner_compositor.py pipeline/tests/test_strip_owner_composition_integration.py pipeline/tests/test_project_writer.py pipeline/tests/test_render_plan_trace_integrity.py pipeline/tests/test_main_emit.py
git add -p -- pipeline/schema/project_schema_v12.py pipeline/schema/project_schema_v12.json
git diff --cached --check
git commit -m "feat: propagate owner render quality contract"
```

Se os schemas não mudarem, não os inclua no staging.

### Task 12: Tornar o observador OCR final independente e fail-closed

**Files:**

- Modify: `pipeline/vision_stack/runtime.py:14982-15350`
- Modify: `pipeline/qa/final_pixel_observer.py:15-150`
- Modify: `pipeline/main.py:9099-9205,14226-14312`
- Test: `pipeline/tests/test_vision_stack_runtime.py`
- Test: `pipeline/tests/test_final_pixel_observer.py`
- Test: `pipeline/tests/test_main_emit.py`

**Step 1: Escrever testes RED**

```python
def test_final_observer_consumes_raw_records_before_semantic_routing(): ...
def test_detector_blocks_with_zero_usable_ocr_are_incomplete(): ...
def test_zero_ocr_zero_detector_is_incomplete_when_material_components_exist(): ...
def test_empty_page_without_material_components_may_be_complete(): ...
def test_final_probe_uses_real_page_identity(): ...
def test_final_probe_records_reason_for_every_material_block(): ...
def test_main_does_not_mark_empty_probe_complete(): ...
def test_main_strip_runtime_exposes_final_probe_bridge(): ...
```

**Step 2: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_vision_stack_runtime.py::VisionStackRuntimeTests::test_final_observer_consumes_raw_records_before_semantic_routing `
  pipeline/tests/test_final_pixel_observer.py::test_detector_blocks_with_zero_usable_ocr_are_incomplete `
  pipeline/tests/test_main_emit.py::MainEmitTests::test_main_does_not_mark_empty_probe_complete -q
```

Expected: filtros podem zerar OCR e `observation_complete` é gravado como `True` incondicionalmente.

**Step 3: Implementar o modo raw final-pixel**

Crie uma API explícita no runtime:

```python
def run_final_pixel_ocr_probe(
    image_rgb: np.ndarray,
    *,
    detected_blocks: Sequence[dict[str, Any]],
    source_challenges: Sequence[dict[str, Any]],
    page_id: str,
    page_number: int,
    source_language: str,
) -> FinalPixelProbeResult:
    ...
```

Reutilize os snapshots brutos de observação do OCR, antes do routing semântico. Não passe por early return de scanlation, SFX, cover ou heurística sem tentar OCR. Remova a flag inerte `_final_pixel_fresh_observation` quando todos os callers usarem a API explícita. Registre tentativa e motivo por bloco/challenge e preserve page ID/número reais.

Adicione `StripRuntime.run_final_pixel_ocr_probe()` na classe bridge local de `main.py`; ela delega à API do runtime com a mesma configuração de engine. `DetectorOcrFinalPixelObserver` chama esse método real depois da detecção, em vez de `run_ocr_stage()`.

Amplie `FinalPixelObservation` com `detected_block_count`, `ocr_record_count`, `ocr_attempts`, `expected_source_challenge_count`, `completed_source_challenge_count`, `coverage_complete` e `coverage_failures`.

`_observe_verified_owner_final_pages()` passa todos os componentes materiais como source challenges e usa o estado real do observer; não fixa mais `observation_complete=True`. Completude exige `expected_source_challenge_count == completed_source_challenge_count` e artefatos OCR válidos. `0 detector/0 OCR` só é completo quando a página não possui owner/componente material.

**Step 4: Confirmar GREEN**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_vision_stack_runtime.py `
  pipeline/tests/test_final_pixel_observer.py `
  pipeline/tests/test_main_emit.py -q
```

Expected: PASS; detector não vazio/OCR vazio sem justificativa resulta em observação incompleta.

**Step 5: Commit**

```powershell
git add -p -- pipeline/vision_stack/runtime.py pipeline/qa/final_pixel_observer.py pipeline/main.py pipeline/tests/test_vision_stack_runtime.py pipeline/tests/test_final_pixel_observer.py pipeline/tests/test_main_emit.py
git diff --cached --check
git commit -m "fix: make final pixel OCR observation fail closed"
```

### Task 13: Desafiar o final contra toda a evidência-fonte

**Files:**

- Modify: `pipeline/qa/final_pixel_observer.py`
- Modify: `pipeline/qa/final_pixel_qa.py:78-405`
- Modify: `pipeline/main.py:14226-14312`
- Test: `pipeline/tests/test_final_pixel_observer.py`
- Test: `pipeline/tests/test_final_pixel_qa.py`
- Test: `pipeline/tests/test_final_pixel_qa_integration.py`

**Step 1: Escrever testes RED**

```python
def test_component_challenge_runs_anchored_final_crop_ocr(): ...
def test_unselected_high_confidence_source_suffix_blocks_incomplete_payload(): ...
def test_source_observation_polygon_outside_cleanup_map_blocks(): ...
def test_anchored_final_ocr_blocks_200_million_residual(): ...
def test_letter_digit_boundary_normalization_detects_200million(): ...
def test_explicit_preserve_policy_excludes_source_challenge(): ...
```

**Step 2: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_final_pixel_qa.py::test_unselected_high_confidence_source_suffix_blocks_incomplete_payload `
  pipeline/tests/test_final_pixel_qa.py::test_anchored_final_ocr_blocks_200_million_residual -q
```

Expected: o QA compara somente com `owner.source_payload` truncado.

**Step 3: Construir desafios owner/component**

Passe ao observer, para cada componente, owner, bbox/polígonos, todos os source candidates, payload selecionado e preserve policy. Execute crops nativo/2x e variantes RGB/cinza/invertida somente quando necessário. Deduplicate por token normalizado e IoU.

**Step 4: Fortalecer os contratos**

- Observação forte material ausente do payload: `source_payload_incomplete`.
- Polígono fonte fora do cleanup owner map: `source_evidence_outside_cleanup`.
- OCR final que casa com qualquer evidência-fonte: `source_payload_visible`.
- Preserve somente com `policy:*` explícita.
- O caso numérico deve bloquear mesmo se o OCR global retornar zero e apenas o crop ancorado detectar o residual.

**Step 5: Confirmar GREEN**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_final_pixel_observer.py `
  pipeline/tests/test_final_pixel_qa.py `
  pipeline/tests/test_final_pixel_qa_integration.py -q
```

Expected: PASS.

**Step 6: Commit**

```powershell
git add -p -- pipeline/qa/final_pixel_observer.py pipeline/qa/final_pixel_qa.py pipeline/main.py pipeline/tests/test_final_pixel_observer.py pipeline/tests/test_final_pixel_qa.py pipeline/tests/test_final_pixel_qa_integration.py
git diff --cached --check
git commit -m "fix: block source residuals from complete evidence"
```

### Task 14: Fazer o export gate respeitar todos os novos contratos

**Files:**

- Modify: `pipeline/qa/final_pixel_qa.py`
- Modify: `pipeline/qa/export_gate.py:136-1505`
- Modify: `pipeline/qa/translation_qa.py`
- Modify: `pipeline/main.py:8112-8525`
- Test: `pipeline/tests/test_final_pixel_export_gate.py`
- Test: `pipeline/tests/test_export_gate_debug_consistency.py`
- Test: `pipeline/tests/test_main_emit.py`

**Step 1: Escrever testes RED**

```python
def test_export_gate_blocks_missing_owner_render_quality_contract(): ...
def test_export_gate_blocks_under_source_scale_owner(): ...
def test_export_gate_blocks_detected_blocks_without_ocr(): ...
def test_export_gate_blocks_source_payload_incomplete(): ...
def test_export_gate_blocks_unverified_owner_residual(): ...
def test_export_gate_blocks_protected_art_contract_violation(): ...
def test_export_gate_blocks_core_pixels_outside_safe_polygon(): ...
def test_gate_summary_and_debug_counts_match_new_contract_issues(): ...
```

**Step 2: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_final_pixel_export_gate.py::test_export_gate_blocks_detected_blocks_without_ocr `
  pipeline/tests/test_final_pixel_export_gate.py::test_export_gate_blocks_missing_owner_render_quality_contract `
  pipeline/tests/test_export_gate_debug_consistency.py::test_gate_summary_and_debug_counts_match_new_contract_issues -q
```

Expected: o gate atual confia em `observation_complete` e `fit_status` insuficientes.

**Step 3: Publicar contratos críticos**

Adicione `layout_legibility_contract`, `residual_cleanup_contract`, `protected_art_contract` e `qa_integrity_contract` à lista final. Bloqueie:

- contrato ausente/inválido;
- `under_source_scale`, `underfilled`, `below_proportional_legibility`;
- x-height/line-core underfill e `outside_safe_pixels > 0`;
- residual ausente/acima do limite;
- qualquer overlap action/protection ou pixel protegido alterado;
- observação final incompleta;
- source payload incompleto ou fonte fora do cleanup;
- locale crítico.

`_qa_translated_final_crops_against_layers()` não pode simplesmente gerar zero rows no owner mode e ainda sugerir cobertura visual. O QA page-space novo substitui a autoridade band-local e publica métricas por owner.

**Step 4: Ligar issues aos artefatos**

Cada issue aponta para graph/ledger, mask/cleanup, render plan, final OCR e hash do persistido. Contagens do relatório, gate e debug devem ser idênticas.

**Step 5: Confirmar GREEN**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_final_pixel_export_gate.py `
  pipeline/tests/test_export_gate_debug_consistency.py `
  pipeline/tests/test_main_emit.py -q
```

Expected: PASS.

**Step 6: Commit**

```powershell
git add -p -- pipeline/qa/final_pixel_qa.py pipeline/qa/export_gate.py pipeline/qa/translation_qa.py pipeline/main.py pipeline/tests/test_final_pixel_export_gate.py pipeline/tests/test_export_gate_debug_consistency.py pipeline/tests/test_main_emit.py
git diff --cached --check
git commit -m "fix: enforce functional owner contracts at export"
```

### Task 15: Endurecer a matriz e o relatório de inspeção

**Files:**

- Modify: `pipeline/tools/validate_owner_visual_matrix.py`
- Modify: `pipeline/tests/test_owner_visual_matrix_tool.py`

**Step 1: Escrever testes RED**

```python
def test_matrix_blocks_missing_owner_render_quality_contract(): ...
def test_matrix_blocks_under_source_scale_owner(): ...
def test_matrix_blocks_incomplete_final_ocr_coverage(): ...
def test_matrix_blocks_unverified_residual(): ...
def test_matrix_groups_failures_by_contract(): ...
def test_report_lists_only_artifacts_actually_inspected(): ...
def test_matrix_requires_calibration_and_holdout_entries(): ...
def test_matrix_resolves_versioned_config_and_input_key_with_matching_hash(): ...
def test_matrix_rejects_missing_or_hash_mismatched_input(): ...
def test_inspection_manifest_requires_artifact_hash_scale_and_verdict(): ...
def test_report_rejects_unverified_inspection_claims(): ...
```

**Step 2: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_owner_visual_matrix_tool.py -q
```

Expected: a matriz atual aceita gates falsamente verdes e pode superdeclarar inspeção.

**Step 3: Implementar validação explícita**

Para cada owner renderizado, exija quality contract `ok`, source scale quando confiável, residual verificado, final report completo, nenhum desafio fonte inconclusivo, contratos não bloqueados e hash atual. Exija splits calibration e holdout por obra/capítulo.

Faça o tool resolver `config_path` somente dentro da fixture e `input_key` por `inputs.json`/variável de ambiente. Antes do runner, valide existência e SHA-256 do input; ausência ou mismatch é blocker explícito. Nenhum config pode depender de arquivo `.codex-tmp` não versionado.

Adicione suporte a `--inspection-template` e `--inspection-manifest`. O template lista todo artefato/segmento exigido; o manifest preenchido contém path relativo, SHA-256, escala (`native`), timestamp, categoria, owner/segmento, veredito e nota. O relatório separa “gerado”, “checado automaticamente” e “inspecionado visualmente” e rejeita qualquer claim sem hash correspondente.

**Step 4: Confirmar GREEN**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_owner_visual_matrix_tool.py -q
```

Expected: PASS.

**Step 5: Commit**

```powershell
git add -p -- pipeline/tools/validate_owner_visual_matrix.py pipeline/tests/test_owner_visual_matrix_tool.py
git diff --cached --check
git commit -m "test: enforce owner contracts in visual matrix"
```

### Task 16: Rodar regressões focadas, properties e suite ampla

**Files:**

- No production edits expected.
- Update tests only if a failure is demonstrably caused by a stale expectation replaced by this plan’s approved contract.

**Step 1: Rodar a suite funcional focada**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_owner_evidence.py `
  pipeline/tests/test_owner_reconcile.py `
  pipeline/tests/test_owner_reconcile_properties.py `
  pipeline/tests/test_owner_mask.py `
  pipeline/tests/test_owner_atomic_execution.py `
  pipeline/tests/test_owner_compositor.py `
  pipeline/tests/test_owner_compositor_properties.py `
  pipeline/tests/test_strip_owner_composition_integration.py `
  pipeline/tests/test_translation_locale_policy.py `
  pipeline/tests/test_translate_context.py `
  pipeline/tests/test_owner_render_quality.py `
  pipeline/tests/test_owner_enforcement.py `
  pipeline/tests/test_owner_layout.py `
  pipeline/tests/test_typesetting_renderer.py `
  pipeline/tests/test_final_pixel_observer.py `
  pipeline/tests/test_final_pixel_qa.py `
  pipeline/tests/test_final_pixel_qa_integration.py `
  pipeline/tests/test_final_pixel_export_gate.py `
  pipeline/tests/test_export_gate_debug_consistency.py `
  pipeline/tests/test_owner_visual_matrix_tool.py `
  -q
if ($LASTEXITCODE -ne 0) { throw "Focused functional suite failed" }
```

Expected: zero failures.

**Step 2: Rodar a suite ampla**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests -q --tb=short
if ($LASTEXITCODE -ne 0) { throw "Broad pipeline suite failed or diverged from baseline" }
```

Se existirem falhas históricas no baseline da Task 1, compare nodeid por nodeid. Nenhuma falha nova é aceitável.

**Step 3: Auditar o diff**

```powershell
git diff --check
git status --short
```

Expected: apenas mudanças do plano e mudanças locais preexistentes. Não crie commit vazio para resultados de teste.

### Task 17: Executar matriz nova, inspecionar pixels nativos e publicar veredito

**Files:**

- Create: `docs/reports/2026-07-30-page-owner-functional-validation.md`
- Create: `docs/reports/evidence/2026-07-30-page-owner-functional-inspection.json`
- Runtime only: `.codex-tmp/page_owner_functional_validation_v11_<timestamp>/**`

**Step 1: Executar a matriz em diretório novo**

```powershell
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$matrixRoot = Join-Path $PWD ".codex-tmp\page_owner_functional_validation_v11_$stamp"
$reportPath = Join-Path $PWD "docs\reports\2026-07-30-page-owner-functional-validation.md"
$inspectionTemplate = Join-Path $matrixRoot "inspection-template.json"
$env:TRADUZAI_MATRIX_MYTHIC_SOURCE = (Resolve-Path '.codex-tmp\mythic_ch40_pages_1_2_source_20260724').Path
$env:TRADUZAI_MATRIX_ONE_SECOND_SOURCE = (Resolve-Path '.codex-tmp\page_owner_systemic_sources_20260729\one_second').Path
$env:TRADUZAI_MATRIX_GRAND_FINALE_SOURCE = (Resolve-Path '.codex-tmp\page_owner_systemic_sources_20260729\grand_finale').Path

pipeline\venv\Scripts\python.exe pipeline/tools/validate_owner_visual_matrix.py `
  --manifest pipeline/tests/fixtures/owner_visual_matrix/functional_matrix.json `
  --output-root $matrixRoot `
  --report $reportPath `
  --inspection-template $inspectionTemplate

if ($LASTEXITCODE -ne 0) { throw "Functional matrix blocked" }
```

Expected: calibration e holdout em pelo menos três obras/oito páginas, hashes atuais, zero contrato funcional bloqueado, gates não bloqueados e template enumerando todos os artefatos que precisam de inspeção.

**Step 2: Fixar o checklist e seus hashes antes da inspeção**

```powershell
Get-FileHash -Algorithm SHA256 $inspectionTemplate
```

Expected: template fresco ligado aos mesmos bytes do run. Não marque inspeção automaticamente.

**Step 3: Inspecionar visualmente em escala nativa**

Inspecione originals, `08_inpaint`, `09_typeset`, `10_copyback_reassemble` e `11_qa_export_gate` lado a lado. Verifique todos os segmentos listados no relatório, com foco explícito em:

- card numérico completo, sem fonte residual;
- nenhuma costura/cor diferente na fronteira de bands;
- corpo atômico e sem duplicação;
- `Sim.`, `Vamos...`, títulos e corpos proporcionais;
- nenhum texto fora da safe region;
- nenhuma arte, personagem, ícone ou borda danificada;
- nenhum residual após inpaint;
- nenhum inglês não permitido;
- locale PT-BR;
- style copy fora do veredito.

Se qualquer item falhar, registre `NO-GO`, contrato, owner e artefato. Não corrija manualmente uma página e não reduza a matriz.

**Step 4: Criar e validar o manifest de inspeção**

Use o template como checklist, mas crie `docs/reports/evidence/2026-07-30-page-owner-functional-inspection.json` com `apply_patch`. Para cada item visto, registre SHA-256, `scale="native"`, timestamp, categoria, owner/segmento, `GO|NO-GO` e nota. Não copie um veredito automático.

```powershell
$inspectionManifest = Join-Path $PWD "docs\reports\evidence\2026-07-30-page-owner-functional-inspection.json"
pipeline\venv\Scripts\python.exe pipeline/tools/validate_owner_visual_matrix.py `
  --manifest pipeline/tests/fixtures/owner_visual_matrix/functional_matrix.json `
  --output-root $matrixRoot `
  --report $reportPath `
  --inspection-manifest $inspectionManifest `
  --validate-only

if ($LASTEXITCODE -ne 0) { throw "Persisted bytes or visual inspection manifest failed" }
```

Expected: determinismo, mesmos hashes, cobertura integral do checklist e nenhum claim visual sem evidência.

**Step 5: Validar o relatório e commit somente com evidência real**

```powershell
git diff --check -- docs/reports/2026-07-30-page-owner-functional-validation.md docs/reports/evidence/2026-07-30-page-owner-functional-inspection.json
git add -- docs/reports/2026-07-30-page-owner-functional-validation.md docs/reports/evidence/2026-07-30-page-owner-functional-inspection.json
git diff --cached --check
git commit -m "docs: record owner functional visual validation"
```

**Step 6: Auditoria final**

```powershell
git status --short --branch
git log --oneline -20
git diff --check
```

Expected: mudanças locais preexistentes preservadas; nenhum `DEBUGM`, `.codex-tmp` ou arquivo alheio adicionado.

## Definição final de pronto

- O corpus genérico reproduz todos os grupos de erro sem exceções de produção.
- O payload completo chega à tradução e a evidência truncada não domina por IoU.
- Todo ink fonte selecionado está na action mask.
- Todo cleanup possui residual verificado e não toca arte protegida.
- Resultado é invariável à partição em bands.
- Bodies permanecem atômicos.
- Todo owner renderizado possui contrato proporcional `ok` e fica dentro da safe region.
- O observador final é raw, fresco, completo e hash-correto.
- QA compara contra toda evidência-fonte e bloqueia residual/incompletude.
- Export gate e relatório concordam exatamente.
- Target locale PT-BR não contém mismatch determinístico.
- Suites focadas e amplas não introduzem regressão.
- Matriz nova cross-work e inspeção nativa retornam GO real.

# Page-Global Text Ownership and Visual Integrity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Impedir sistemicamente texto-fonte residual, body repartido, render duplicado ou deslocado, inpaint incompleto/destrutivo e corrupção por overlap em qualquer band, página, capítulo ou obra.

**Architecture:** O pipeline passa a ter um control plane textual page-global anterior à tradução. Todos os OCRs viram evidências imutáveis; um `OwnerGraph` resolve cobertura, identidade, body, rota e executor uma única vez. Bands permanecem tiles de coleta/execução. Máscara, inpaint, layout e typeset consomem owners autoritativos; um compositor page-space é a única autoridade de pixels; QA independente observa os bytes persistidos e fecha o export gate por hash.

**Tech Stack:** Python 3.12, dataclasses/typing, NumPy, OpenCV, PIL, PaddleOCR/Koharu adapters, pytest, JSON/JSONL versionado e SHA-256.

---

## Regras obrigatórias de execução

- Use as skills `mangatl-dev`, `traduzai-detect`, `traduzai-ocr`, `traduzai-inpaint`, `traduzai-typesetting` e `traduzai-pipeline` conforme o owner de cada tarefa.
- Trabalhe no checkout atual e preserve integralmente o dirty worktree. Não use `reset`, `checkout`, `stash`, `clean` nem crie worktree.
- Antes de cada patch, rode `git status --short -- <arquivos>` e `git diff -- <arquivos>`; trate qualquer hunk preexistente como pertencente ao usuário.
- Aplique TDD estrito: escreva um teste, confirme RED pela razão esperada, implemente o mínimo sistêmico e confirme GREEN.
- Não aceite RED causado por erro de sintaxe, import acidental ou fixture ausente depois da tarefa que cria o módulo.
- Faça commits apenas dos arquivos listados na tarefa. Antes de cada commit, confirme `git diff --cached --name-only`.
- Checkpoints são verificações automáticas; não pause para pedir confirmação se os critérios estiverem verdes.
- Nenhuma decisão de produção pode depender de obra, capítulo, número de página, texto literal, coordenada fixa ou `band_id`. `tile_id`/`band_id` são somente proveniência e scheduling.
- Casos reais podem existir apenas em manifests/reports de regressão que nenhum módulo de produção importa.
- Cópia de estilo permanece separada. Esta implementação só testa que style evidence não altera owner, body, rota, tradução ou máscara.
- Artefatos em `debug/` são derivados read-only e nunca entram como fonte de composição, QA ou rerender de produção.
- Não declare sucesso por `exit 0`. O run final precisa ter `export_gate=PASS`, hashes coerentes, zero blocker e inspeção visual aprovada.

## Baseline e escopo do checkout

O design aprovado está no commit `b32659e9`. No momento do planejamento já existem alterações locais preexistentes em `pipeline/main.py`, `pipeline/strip/run.py`, arquivos de contrato de cor e artefatos de auditoria. Preserve-as e faça staging por path/hunk intencional.

O último baseline amplo observado neste checkout foi `290 passed, 16 failed`; as 16 falhas pertencem a contratos locais já divergentes e não podem ser apagadas ou mascaradas por este plano. Na Task 1 registre novamente o baseline focado e, no encerramento, compare nodeid por nodeid.

## Checkpoints contínuos

1. **Checkpoint A — control plane:** Tasks 1–7; nenhum efeito visual pode começar antes do grafo.
2. **Checkpoint B — owner execution:** Tasks 8–12; tradução, máscara, rollback e layout são 1:1 por owner.
3. **Checkpoint C — pixel authority:** Tasks 13–16; compositor único, QA fresh e gate fail-closed.
4. **Checkpoint D — enforcement:** Tasks 17–19; compatibilidade legada isolada, properties verdes e validação visual cross-work.

### Task 1: Fixar o corpus sistêmico e o baseline sem casos de produção

**Files:**
- Create: `pipeline/tests/fixtures/systemic_visual_contracts/owner_cases.json`
- Create: `pipeline/tests/test_systemic_owner_contract_manifest.py`
- Read: `pipeline/tests/test_strip_ocr_debug_artifacts.py`
- Read: `pipeline/tests/test_vision_stack_runtime.py`
- Read: `pipeline/tests/test_strip_process_bands.py`
- Read: `pipeline/tests/test_typesetting_renderer.py`
- Read: `pipeline/tests/test_export_gate.py`

**Step 1: Inspecionar estado e capturar o baseline focado**

```powershell
git status --short -- pipeline/tests
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_strip_ocr_debug_artifacts.py `
  pipeline/tests/test_vision_stack_runtime.py `
  pipeline/tests/test_strip_process_bands.py `
  pipeline/tests/test_typesetting_renderer.py `
  pipeline/tests/test_export_gate.py `
  -q | Tee-Object .codex-tmp\page_owner_baseline_20260725.txt
```

Expected: registrar todos os nodeids atuais; não corrigir falhas fora deste plano nesta etapa.

**Step 2: Escrever primeiro o teste do manifest**

O teste deve carregar o JSON ainda inexistente e exigir:

```python
REQUIRED_CATEGORIES = {
    "overlapping_tiles_single_body",
    "card_semantic_roles",
    "connected_layout_regions",
    "repeated_text_distinct_geometry",
    "protected_art_contact",
    "primary_ocr_omission_recall",
}

assert all("work" not in case and "chapter" not in case for case in cases)
assert all(case["expected"]["production_exception_count"] == 0 for case in cases)
```

**Step 3: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_systemic_owner_contract_manifest.py -q
```

Expected: FAIL com `FileNotFoundError` para `owner_cases.json`.

**Step 4: Criar fixtures geométricas genéricas**

Cada caso deve conter `page_size`, `source_components`, `observations`, `tiles`, `protected_art` e `expected`. Use apenas IDs abstratos (`component_a`, `tile_left`) e coordenadas relativas à fixture. Inclua permutações de 1, 2 e 4 tiles.

**Step 5: Confirmar GREEN e commit**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_systemic_owner_contract_manifest.py -q
git add -- pipeline/tests/fixtures/systemic_visual_contracts/owner_cases.json pipeline/tests/test_systemic_owner_contract_manifest.py
git diff --cached --name-only
git commit -m "test: define systemic owner contract corpus"
```

Expected: PASS; o commit contém somente os dois arquivos novos.

### Task 2: Implementar o modelo e os invariantes do OwnerGraph

**Files:**
- Create: `pipeline/ownership/__init__.py`
- Create: `pipeline/ownership/model.py`
- Create: `pipeline/tests/test_owner_graph.py`

**Step 1: Escrever testes RED do modelo**

Cobrir:

```python
def test_every_component_has_exactly_one_final_disposition(): ...
def test_component_cannot_belong_to_two_active_owners(): ...
def test_translatable_owner_has_one_source_and_translation_payload(): ...
def test_body_owner_cannot_expose_multiple_semantic_payloads(): ...
def test_owner_has_exactly_one_executor_projection(): ...
def test_review_required_owner_cannot_enter_render_plan(): ...
def test_graph_serialization_is_deterministic(): ...
```

**Step 2: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_owner_graph.py -q
```

Expected: FAIL em collection com `ModuleNotFoundError: No module named 'ownership'`.

**Step 3: Implementar os tipos autoritativos**

`model.py` deve definir dataclasses frozen quando a entidade for evidência:

```python
@dataclass(frozen=True)
class SourceTextComponent:
    component_id: str
    page_id: str
    bbox_page: tuple[int, int, int, int]
    polygon_page: tuple[tuple[int, int], ...]
    detector_sources: tuple[str, ...]

@dataclass(frozen=True)
class TextObservation:
    observation_id: str
    page_id: str
    component_ids: tuple[str, ...]
    text: str
    confidence: float
    provider: str
    bbox_page: tuple[int, int, int, int]
    tile_provenance: tuple[str, ...] = ()

@dataclass
class TextOwner:
    owner_id: str
    page_id: str
    component_ids: list[str]
    observation_ids: list[str]
    semantic_role: str
    source_payload: str
    translated_payload: str | None
    disposition: str
    state: str
    execution_tile_id: str | None

@dataclass
class OwnerGraph:
    schema_version: int
    page_id: str
    components: list[SourceTextComponent]
    observations: list[TextObservation]
    owners: list[TextOwner]
    projections: list[OwnerProjection]
    violations: list[OwnerViolation]
```

Adicionar `OwnerGraph.validate()`, `require_valid()`, `to_dict()` e `from_dict()`. O validator deve produzir códigos estáveis e `offenders[]` com `owner_id`/`component_id`.

**Step 4: Confirmar GREEN**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_owner_graph.py -q
```

Expected: PASS.

**Step 5: Commit**

```powershell
git add -- pipeline/ownership/__init__.py pipeline/ownership/model.py pipeline/tests/test_owner_graph.py
git diff --cached --name-only
git commit -m "feat: define page owner graph contracts"
```

### Task 3: Tornar IDs e transformações page-space determinísticos

**Files:**
- Create: `pipeline/ownership/coordinates.py`
- Modify: `pipeline/ownership/model.py`
- Modify: `pipeline/strip/types.py`
- Modify: `pipeline/strip/detect_balloons.py`
- Modify: `pipeline/strip/bands.py`
- Create: `pipeline/tests/test_owner_coordinates.py`
- Modify: `pipeline/tests/test_strip_detect.py`
- Modify: `pipeline/tests/test_strip_bands.py`

**Step 1: Ler os diffs preexistentes**

```powershell
git status --short -- pipeline/strip/types.py pipeline/strip/detect_balloons.py pipeline/strip/bands.py
git diff -- pipeline/strip/types.py pipeline/strip/detect_balloons.py pipeline/strip/bands.py
```

**Step 2: Escrever testes RED**

Exigir:

```python
def test_component_ids_are_stable_under_detector_order(): ...
def test_observation_ids_do_not_depend_on_recognized_text(): ...
def test_tile_projection_roundtrips_bbox_and_polygon(): ...
def test_region_ids_exist_before_grouping_into_bands(): ...
def test_changing_band_boundaries_does_not_change_region_ids(): ...
```

**Step 3: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_owner_coordinates.py `
  pipeline/tests/test_strip_detect.py `
  pipeline/tests/test_strip_bands.py `
  -k "owner or region_id or projection" -q
```

Expected: FAIL porque IDs page-global e transforms ainda não existem.

**Step 4: Implementar coordenadas e IDs**

Criar funções puras:

```python
def stable_spatial_id(prefix: str, page_id: str, bbox: BBoxTuple, ordinal: int) -> str: ...
def bbox_tile_to_page(bbox: BBoxTuple, *, dx: int, dy: int) -> BBoxTuple: ...
def polygon_tile_to_page(points: Sequence[Point], *, dx: int, dy: int) -> tuple[Point, ...]: ...
def assign_component_ids(page_id: str, components: Sequence[ComponentSeed]) -> list[str]: ...
```

Ordene por `(y1, x1, y2, x2, detector_source)` antes de atribuir ordinal. Não inclua texto, obra, capítulo ou band no hash/ID. Adicione `region_id` a `Balloon.metadata` antes de `group_balloons_into_bands`; `Band` recebe somente `tile_id` e transform/proveniência.

**Step 5: Confirmar GREEN e commit**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_owner_coordinates.py `
  pipeline/tests/test_strip_detect.py `
  pipeline/tests/test_strip_bands.py `
  -k "owner or region_id or projection" -q
git add -- pipeline/ownership/coordinates.py pipeline/ownership/model.py pipeline/strip/types.py pipeline/strip/detect_balloons.py pipeline/strip/bands.py pipeline/tests/test_owner_coordinates.py pipeline/tests/test_strip_detect.py pipeline/tests/test_strip_bands.py
git diff --cached --name-only
git commit -m "feat: assign stable page-space text identities"
```

### Task 4: Resolver owners semânticos e arbitrar OCR por cobertura monotônica

**Files:**
- Create: `pipeline/ownership/reconcile.py`
- Create: `pipeline/tests/test_owner_reconcile.py`
- Create: `pipeline/tests/test_owner_reconcile_properties.py`
- Read: `pipeline/vision_stack/runtime.py:_merge_ocr_clusters`
- Read: `pipeline/ocr/ocr_normalizer.py:merge_same_balloon_fragments_before_translation`
- Read: `pipeline/inpainter/mask_builder.py:build_mask_regions`

**Step 1: Escrever testes RED do resolver puro**

Cobrir todos os casos do manifest e estas propriedades:

```python
def test_one_body_seen_in_multiple_tiles_has_one_owner(): ...
def test_partial_observation_never_replaces_full_coverage(): ...
def test_identical_text_in_distant_geometry_has_two_owners(): ...
def test_card_body_lines_stay_atomic_while_roles_stay_distinct(): ...
def test_connected_layout_lobes_do_not_force_multiple_translations(): ...
def test_every_evidence_is_owned_preserved_suppressed_or_review(): ...
def test_graph_is_invariant_to_observation_and_tile_permutation(): ...
def test_graph_is_invariant_to_one_two_or_four_tile_partition(): ...
```

Use seeds fixos e `itertools.permutations`; não adicione Hypothesis se ela ainda não estiver no projeto.

**Step 2: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_owner_reconcile.py `
  pipeline/tests/test_owner_reconcile_properties.py `
  -q
```

Expected: FAIL por ausência de `build_page_owner_graph`.

**Step 3: Implementar o resolver**

API autoritativa:

```python
def build_page_owner_graph(
    *,
    page_id: str,
    components: Sequence[SourceTextComponent],
    observations: Sequence[TextObservation],
    semantic_regions: Sequence[SemanticRegion],
) -> OwnerGraph: ...
```

Regras:

- construir conectividade por componentes/contorno/safe region, nunca por `band_id`;
- body/parágrafo é atômico; newline é evidência visual, não payload separado;
- card permite roles distintos, mas não owner por linha do mesmo body;
- escolha de OCR maximiza primeiro cobertura de componentes, depois consistência geométrica, idioma, confiança e coerência;
- candidato novo não substitui um completo por um parcial;
- decisões ambíguas viram `review_required`, nunca fallback silencioso;
- `build_mask_regions` pode fornecer raster/adjacência como feature, mas não decidir identidade.

**Step 4: Confirmar GREEN, executar teste anti-hardcode e commit**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_owner_graph.py `
  pipeline/tests/test_owner_reconcile.py `
  pipeline/tests/test_owner_reconcile_properties.py `
  pipeline/tests/test_systemic_owner_contract_manifest.py `
  -q
rg -n "obra|chapter|cap[ií]tulo|band_id|page_[0-9]+|if .*tile_id" pipeline/ownership/reconcile.py
```

Expected: testes PASS; `rg` não encontra branch semântico proibido. Proveniência pode ser copiada/serializada, nunca usada no score.

```powershell
git add -- pipeline/ownership/reconcile.py pipeline/tests/test_owner_reconcile.py pipeline/tests/test_owner_reconcile_properties.py
git diff --cached --name-only
git commit -m "feat: reconcile page text owners monotonically"
```

### Task 5: Descobrir componentes-fonte independentemente do OCR aceito

**Files:**
- Create: `pipeline/ownership/discovery.py`
- Modify: `pipeline/vision_stack/runtime.py`
- Modify: `pipeline/strip/run.py`
- Create: `pipeline/tests/test_source_component_discovery.py`
- Modify: `pipeline/tests/test_vision_stack_runtime.py`
- Modify: `pipeline/tests/test_strip_detect.py`

**Step 1: Escrever testes RED de recall independente**

```python
def test_discovers_textlike_component_omitted_by_primary_ocr(): ...
def test_discovery_does_not_depend_on_accepted_text_layers(): ...
def test_component_geometry_is_page_space_before_ocr(): ...
def test_detector_and_glyph_scan_candidates_are_deduplicated_without_text(): ...
def test_every_discovered_component_needs_owner_or_explicit_disposition(): ...
def test_unresolved_textlike_component_blocks_instead_of_disappearing(): ...
```

Use fixtures sintéticas com glyph strokes em balão, card e painel escuro. Inclua um componente que o mock de OCR deliberadamente omite.

**Step 2: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_source_component_discovery.py `
  pipeline/tests/test_vision_stack_runtime.py `
  pipeline/tests/test_strip_detect.py `
  -k "source_component or textlike or primary_ocr_omission" -q
```

Expected: FAIL por ausência de `ownership.discovery` e porque cobertura ainda nasce dos layers aceitos.

**Step 3: Implementar a passagem page-global de recall**

API pura:

```python
def discover_source_text_components(
    page_rgb: np.ndarray,
    *,
    page_id: str,
    detector_regions: Sequence[DetectorRegion],
    glyph_candidates: Sequence[GlyphCandidate] = (),
) -> list[SourceTextComponent]: ...
```

Combine regiões do detector normal com uma varredura independente de connected components/stroke/contraste já disponível no runtime. Una candidatos por geometria e contorno, sem usar texto reconhecido, tradução ou layers aceitos. Transforme tudo para page-space, atribua IDs estáveis e preserve detector/evidência de script.

A etapa só responde “onde há evidência visual de texto”; ela não escolhe OCR, owner, rota ou style. Componente ambíguo permanece no manifest e depois vira `review_required` se o resolver não puder dispô-lo.

**Step 4: Integrar antes da formação do grafo e confirmar GREEN**

Em `run_chapter`, execute discovery sobre cada página original canônica antes de `build_page_owner_graph`. Bands podem fornecer crops adicionais, mas não remover componentes globais.

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_source_component_discovery.py `
  pipeline/tests/test_vision_stack_runtime.py `
  pipeline/tests/test_strip_detect.py `
  -k "source_component or textlike or primary_ocr_omission" -q
```

Expected: PASS.

**Step 5: Commit**

```powershell
git add -- pipeline/ownership/discovery.py pipeline/vision_stack/runtime.py pipeline/strip/run.py pipeline/tests/test_source_component_discovery.py pipeline/tests/test_vision_stack_runtime.py pipeline/tests/test_strip_detect.py
git diff --cached --name-only
git commit -m "feat: discover source text independently of OCR"
```

### Task 6: Converter todos os caminhos OCR em observações não destrutivas

**Files:**
- Create: `pipeline/ownership/ocr_adapter.py`
- Modify: `pipeline/vision_stack/ocr.py`
- Modify: `pipeline/vision_stack/runtime.py`
- Modify: `pipeline/strip/process_bands.py`
- Modify: `pipeline/tests/test_vision_stack_ocr.py`
- Modify: `pipeline/tests/test_vision_stack_runtime.py`
- Modify: `pipeline/tests/test_strip_ocr_debug_artifacts.py`

**Step 1: Inspecionar diffs e os merges atuais**

```powershell
git status --short -- pipeline/vision_stack/ocr.py pipeline/vision_stack/runtime.py pipeline/strip/process_bands.py
git diff -- pipeline/vision_stack/ocr.py pipeline/vision_stack/runtime.py pipeline/strip/process_bands.py
rg -n "def (recognize_blocks_from_page|run_ocr_stage|build_page_result|_merge_ocr_clusters|_integrate_recovery_page|_merge_candidate_crop_recovery_into_ocr_page|fuse_negative_dark_bubble_candidates)" pipeline/vision_stack pipeline/strip/process_bands.py
```

**Step 2: Escrever testes RED**

Adicionar testes que exijam:

```python
def test_full_page_crop_negative_rotated_and_recovery_become_observations(): ...
def test_adapter_preserves_rejected_candidates_with_reason(): ...
def test_partial_and_full_readings_survive_until_owner_reconcile(): ...
def test_ocr_adapter_maps_every_polygon_to_page_space(): ...
def test_ocr_adapter_does_not_assign_semantic_owner_from_band(): ...
```

Atualize os antigos testes de cross-band para afirmar o grafo resultante, não uma supressão por prefixo.

**Step 3: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_vision_stack_ocr.py `
  pipeline/tests/test_vision_stack_runtime.py `
  pipeline/tests/test_strip_ocr_debug_artifacts.py `
  -k "observation or owner or full_text_for_translation or repeated_text" -q
```

Expected: FAIL porque retries/backends ainda sobrescrevem ou eliminam candidatos.

**Step 4: Implementar o adapter**

`ocr_adapter.py` expõe:

```python
def record_to_observation(record: dict, projection: TileProjection) -> TextObservation: ...
def collect_page_observations(records_by_provider: Mapping[str, Sequence[dict]], projection: TileProjection) -> list[TextObservation]: ...
def attach_observation_manifest(page: dict, observations: Sequence[TextObservation]) -> dict: ...
```

Em `run_ocr_stage` e `build_page_result`, capture outputs de full-page, crop, adaptive, rotated, raw, negative e recovery antes de qualquer merge/drop. Mantenha `texts` legado em shadow mode, mas `owner_observations` é append-only e autoritativo para o novo resolver.

Não remova ainda `_merge_ocr_clusters` nem os merges locais; marque seus resultados como `legacy_selected` apenas para comparação.

**Step 5: Confirmar GREEN e commit**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_vision_stack_ocr.py `
  pipeline/tests/test_vision_stack_runtime.py `
  pipeline/tests/test_strip_ocr_debug_artifacts.py `
  -k "observation or owner or full_text_for_translation or repeated_text" -q
git add -- pipeline/ownership/ocr_adapter.py pipeline/vision_stack/ocr.py pipeline/vision_stack/runtime.py pipeline/strip/process_bands.py pipeline/tests/test_vision_stack_ocr.py pipeline/tests/test_vision_stack_runtime.py pipeline/tests/test_strip_ocr_debug_artifacts.py
git diff --cached --name-only
git commit -m "feat: retain page-space OCR observations"
```

### Task 7: Introduzir a barreira page-global antes da primeira tradução

**Files:**
- Modify: `pipeline/strip/process_bands.py`
- Modify: `pipeline/strip/run.py`
- Modify: `pipeline/strip/types.py`
- Create: `pipeline/tests/test_strip_owner_control_plane.py`
- Modify: `pipeline/tests/test_strip_run.py`
- Modify: `pipeline/tests/test_strip_process_bands.py`

**Step 1: Inspecionar hunks preexistentes em arquivos de alto conflito**

```powershell
git status --short -- pipeline/strip/process_bands.py pipeline/strip/run.py pipeline/strip/types.py
git diff -- pipeline/strip/process_bands.py pipeline/strip/run.py pipeline/strip/types.py
```

Preserve especialmente as mudanças locais de contrato RGB já presentes em `run.py`.

**Step 2: Escrever testes RED de ordem de execução**

```python
def test_run_chapter_collects_all_tile_evidence_before_first_translation(): ...
def test_run_chapter_resolves_each_page_once_before_owner_execution(): ...
def test_executor_assignment_prefers_coverage_and_edge_distance(): ...
def test_context_only_projection_never_calls_translate_inpaint_or_typeset(): ...
def test_band_permutation_produces_same_owner_manifest(): ...
```

Use spies com um event log esperado:

```python
assert events == [
    "collect:tile_a",
    "collect:tile_b",
    "resolve:page_a",
    "translate:owner_a",
    "execute:owner_a",
]
```

**Step 3: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_strip_owner_control_plane.py `
  pipeline/tests/test_strip_run.py `
  pipeline/tests/test_strip_process_bands.py `
  -k "control_plane or owner_manifest or context_only or before_first_translation" -q
```

Expected: FAIL; hoje `process_band` traduz durante a coleta.

**Step 4: Separar coleta e execução**

Extrair de `process_band`:

```python
def collect_band_evidence(..., *, tile_projection: TileProjection) -> BandEvidenceResult: ...
def execute_owner_tile(..., *, graph: OwnerGraph, projection: OwnerProjection) -> OwnerExecutionResult: ...
```

Reorganizar `run_chapter`:

```text
build strip -> detect regions -> group tiles
-> collect evidence from every tile
-> map observations to source pages
-> build and validate one OwnerGraph per page
-> translate owners
-> execute owner projections
-> compose pages
```

Adicionar `owner_graph_mode` com valores `shadow|enforce|legacy`; use `shadow` durante esta tarefa. Mesmo em shadow, construa o grafo antes de chamar o caminho legado e grave divergências por categoria.

O reconciliador `_reconcile_overlapping_band_ocr_fragments_before_translation` não pode mais decidir a semântica no caminho owner; preserve-o temporariamente apenas em `legacy`.

**Step 5: Confirmar GREEN e Checkpoint A**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_owner_graph.py `
  pipeline/tests/test_owner_reconcile.py `
  pipeline/tests/test_owner_reconcile_properties.py `
  pipeline/tests/test_strip_owner_control_plane.py `
  pipeline/tests/test_strip_run.py `
  pipeline/tests/test_strip_process_bands.py `
  -k "owner or control_plane or context_only or before_first_translation" -q
```

Expected: PASS; spies provam que nenhuma mutação acontece antes do grafo.

**Step 6: Commit**

```powershell
git add -- pipeline/strip/process_bands.py pipeline/strip/run.py pipeline/strip/types.py pipeline/tests/test_strip_owner_control_plane.py pipeline/tests/test_strip_run.py pipeline/tests/test_strip_process_bands.py
git diff --cached --name-only
git commit -m "feat: resolve page owners before band execution"
```

### Task 8: Fazer tradução 1:1 por owner e bloquear resultados incompletos

**Files:**
- Create: `pipeline/ownership/translation.py`
- Modify: `pipeline/translator/translate.py`
- Modify: `pipeline/strip/process_bands.py`
- Modify: `pipeline/tests/test_normalized_text_propagates_to_translation.py`
- Create: `pipeline/tests/test_owner_translation.py`

**Step 1: Escrever testes RED**

```python
def test_translator_receives_one_complete_payload_per_owner(): ...
def test_body_lines_never_reach_translator_as_separate_requests(): ...
def test_translation_response_is_joined_by_owner_id_not_list_position(): ...
def test_missing_owner_translation_becomes_review_required(): ...
def test_duplicate_or_unknown_owner_translation_blocks(): ...
def test_same_owner_observed_in_many_tiles_is_translated_once(): ...
```

**Step 2: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_owner_translation.py `
  pipeline/tests/test_normalized_text_propagates_to_translation.py `
  -q
```

Expected: FAIL; o tradutor ainda recebe páginas band-scoped e faz merge posicional.

**Step 3: Implementar o boundary autoritativo**

```python
def owners_to_translation_page(graph: OwnerGraph) -> dict: ...
def merge_owner_translations(graph: OwnerGraph, translated_page: dict) -> OwnerGraph: ...
```

Cada record enviado contém `owner_id`, `text=source_payload`, `semantic_role` e contexto permitido. O retorno precisa conter exatamente o mesmo conjunto de `owner_id`. Falta, duplicata ou ID desconhecido gera violation crítica e `review_required`; não faça fallback por índice, `text_id` ou `band_id`.

Alterar `_run_translate_stage` para receber owners já validados. Connected layout regions continuam um único request.

**Step 4: Confirmar GREEN e commit**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_owner_translation.py `
  pipeline/tests/test_normalized_text_propagates_to_translation.py `
  pipeline/tests/test_strip_owner_control_plane.py `
  -q
git add -- pipeline/ownership/translation.py pipeline/translator/translate.py pipeline/strip/process_bands.py pipeline/tests/test_owner_translation.py pipeline/tests/test_normalized_text_propagates_to_translation.py
git diff --cached --name-only
git commit -m "feat: translate complete text owners once"
```

### Task 9: Persistir OwnerGraph e migrar projetos legados sem inventar confiança

**Files:**
- Create: `pipeline/ownership/project.py`
- Modify: `pipeline/main.py`
- Modify: `pipeline/project_writer.py`
- Modify: `pipeline/schema/project_schema_v12.py`
- Modify: `pipeline/schema/project_schema_v12.json`
- Modify: `pipeline/schema/migrate_project.py`
- Modify: `docs/project-schema.md`
- Modify: `pipeline/tests/test_project_schema_v12.py`
- Modify: `pipeline/tests/test_project_migration.py`
- Modify: `pipeline/tests/test_project_writer.py`
- Modify: `pipeline/tests/test_main_emit.py`

**Step 1: Inspecionar diffs locais**

```powershell
git status --short -- pipeline/main.py pipeline/project_writer.py pipeline/schema docs/project-schema.md
git diff -- pipeline/main.py pipeline/project_writer.py pipeline/schema docs/project-schema.md
```

Não remova nem misture o fix local de cor em `pipeline/main.py`.

**Step 2: Escrever testes RED**

Exigir:

```python
def test_project_roundtrip_preserves_owner_graph_and_ids(): ...
def test_text_layers_reference_owner_and_component_ids(): ...
def test_project_writer_rejects_layer_owner_not_in_graph(): ...
def test_legacy_project_migrates_as_legacy_unverified(): ...
def test_legacy_migration_never_synthesizes_verified_ownership(): ...
```

**Step 3: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_project_schema_v12.py `
  pipeline/tests/test_project_migration.py `
  pipeline/tests/test_project_writer.py `
  pipeline/tests/test_main_emit.py `
  -k "owner_graph or owner_id or legacy_unverified" -q
```

Expected: FAIL porque o schema/projeto não contém o grafo.

**Step 4: Implementar persistência e adapter legado**

Adicionar ao projeto:

```json
{
  "owner_graph_schema_version": 1,
  "owner_graph_status": "verified",
  "page_owner_graphs": [],
  "owner_invariant_summary": {}
}
```

`build_text_layer` deve persistir `owner_id`, `component_ids`, `observation_ids`, `semantic_role`, `route_action`, `action_mask_ref` e `layout_region_ids`. `build_project_json` inclui os graphs serializados.

Projetos antigos recebem `owner_graph_status="legacy_unverified"` e continuam editáveis, mas o automático não pode tratá-los como graph verificado sem reprocessar a página.

**Step 5: Confirmar GREEN e commit**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_project_schema_v12.py `
  pipeline/tests/test_project_migration.py `
  pipeline/tests/test_project_writer.py `
  pipeline/tests/test_main_emit.py `
  -k "owner_graph or owner_id or legacy_unverified" -q
git add -- pipeline/ownership/project.py pipeline/main.py pipeline/project_writer.py pipeline/schema/project_schema_v12.py pipeline/schema/project_schema_v12.json pipeline/schema/migrate_project.py docs/project-schema.md pipeline/tests/test_project_schema_v12.py pipeline/tests/test_project_migration.py pipeline/tests/test_project_writer.py pipeline/tests/test_main_emit.py
git diff --cached --name-only
git commit -m "feat: persist verified page owner graphs"
```

### Task 10: Produzir action masks e protected-art masks operacionais por owner

**Files:**
- Create: `pipeline/inpainter/owner_mask.py`
- Modify: `pipeline/inpainter/mask_builder.py`
- Modify: `pipeline/inpainter/__init__.py`
- Modify: `pipeline/ownership/model.py`
- Create: `pipeline/tests/test_owner_mask.py`
- Modify: `pipeline/tests/test_mask_builder.py`
- Modify: `pipeline/tests/test_inpaint_mask_geometry.py`
- Modify: `pipeline/tests/test_vision_stack_inpainter.py`

**Step 1: Inspecionar diffs e cadeia atual**

```powershell
git status --short -- pipeline/inpainter pipeline/ownership/model.py
git diff -- pipeline/inpainter pipeline/ownership/model.py
rg -n "def (build_inpaint_mask|build_mask_regions|inpaint_band_image|_record_mask_chain_debug)" pipeline/inpainter
```

**Step 2: Escrever testes RED**

```python
def test_owner_action_mask_is_union_of_owned_glyph_and_line_evidence(): ...
def test_mask_builder_cannot_merge_two_pre_resolved_owners(): ...
def test_action_mask_and_protected_art_mask_are_disjoint(): ...
def test_owner_mask_is_persisted_and_addressable_by_owner_id(): ...
def test_inpaint_changed_pixels_are_subset_of_action_mask(): ...
def test_protected_art_is_unchanged_even_when_touching_text(): ...
def test_missing_safe_mask_moves_owner_to_review_without_bbox_fallback(): ...
```

**Step 3: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_owner_mask.py `
  pipeline/tests/test_mask_builder.py `
  pipeline/tests/test_inpaint_mask_geometry.py `
  pipeline/tests/test_vision_stack_inpainter.py `
  -k "owner or protected_art or action_mask" -q
```

Expected: FAIL; `protection_mask` atual é diagnóstico e rollback ainda pode cair para bbox.

**Step 4: Implementar o contrato operacional**

```python
@dataclass(frozen=True)
class OwnerMaskPlan:
    owner_id: str
    page_id: str
    action_mask_ref: str
    action_mask: np.ndarray
    protected_art_mask: np.ndarray
    evidence_ids: tuple[str, ...]

def build_owner_mask_plan(original_rgb: np.ndarray, owner: TextOwner, evidence: Sequence[...]) -> OwnerMaskPlan: ...
```

`build_mask_regions` recebe owners já resolvidos e só rasteriza. `inpaint_band_image` retorna `OwnerMutation` com `owner_id`, `action_mask_ref`, `changed_mask`, engine, contagens e hashes. Remova fallback silencioso de action mask para bbox quando owner mode estiver ativo.

**Step 5: Confirmar GREEN e commit**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_owner_mask.py `
  pipeline/tests/test_mask_builder.py `
  pipeline/tests/test_inpaint_mask_geometry.py `
  pipeline/tests/test_vision_stack_inpainter.py `
  -k "owner or protected_art or action_mask" -q
git add -- pipeline/inpainter/owner_mask.py pipeline/inpainter/mask_builder.py pipeline/inpainter/__init__.py pipeline/ownership/model.py pipeline/tests/test_owner_mask.py pipeline/tests/test_mask_builder.py pipeline/tests/test_inpaint_mask_geometry.py pipeline/tests/test_vision_stack_inpainter.py
git diff --cached --name-only
git commit -m "feat: enforce owner-scoped inpaint masks"
```

### Task 11: Tornar rollback e copyback atômicos por owner

**Files:**
- Modify: `pipeline/strip/process_bands.py`
- Modify: `pipeline/ownership/model.py`
- Create: `pipeline/tests/test_owner_atomic_execution.py`
- Modify: `pipeline/tests/test_strip_process_bands.py`

**Step 1: Ler o diff relevante**

```powershell
git status --short -- pipeline/strip/process_bands.py pipeline/ownership/model.py pipeline/tests/test_strip_process_bands.py
git diff -- pipeline/strip/process_bands.py pipeline/ownership/model.py pipeline/tests/test_strip_process_bands.py
rg -n "def (_apply_atomic_inpaint_render_rollback|_run_copy_back_stage|_apply_copy_back_outside_balloons)" pipeline/strip/process_bands.py
```

**Step 2: Escrever testes RED**

```python
def test_failed_render_rolls_back_only_its_owner_action_mask(): ...
def test_rollback_never_uses_balloon_bbox_when_action_mask_exists(): ...
def test_neighbor_owner_pixels_survive_rollback(): ...
def test_context_only_projection_produces_no_owner_mutation(): ...
def test_copyback_cannot_restore_source_over_another_owner_glyph(): ...
def test_inpaint_without_successful_render_is_not_committed(): ...
```

**Step 3: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_owner_atomic_execution.py `
  pipeline/tests/test_strip_process_bands.py `
  -k "rollback or copyback or owner_action_mask or neighbor_owner" -q
```

Expected: FAIL; o caminho atual restaura por bbox/contexto de band.

**Step 4: Implementar execução atômica**

Substituir decisões implícitas por:

```python
def apply_atomic_owner_execution(
    original_rgb: np.ndarray,
    mutation: OwnerMutation,
    glyph_patch: OwnerGlyphPatch | None,
) -> OwnerExecutionCommit: ...
```

O commit só é válido quando limpeza e render pertencem ao mesmo `owner_id`, seus hashes batem e nenhum pixel fora da action mask foi alterado. Em falha, restaure somente a máscara do owner e marque `review_required`; não restaure o crop inteiro.

**Step 5: Confirmar GREEN e commit**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_owner_atomic_execution.py `
  pipeline/tests/test_strip_process_bands.py `
  -k "rollback or copyback or owner_action_mask or neighbor_owner" -q
git add -- pipeline/strip/process_bands.py pipeline/ownership/model.py pipeline/tests/test_owner_atomic_execution.py pipeline/tests/test_strip_process_bands.py
git diff --cached --name-only
git commit -m "fix: isolate rollback and copyback by owner"
```

### Task 12: Fazer layout e renderer consumirem owners sem recriar semântica

**Files:**
- Modify: `pipeline/layout/balloon_layout.py`
- Modify: `pipeline/typesetter/renderer.py`
- Modify: `pipeline/typesetter/style_policy.py`
- Create: `pipeline/tests/test_owner_layout.py`
- Modify: `pipeline/tests/test_balloon_layout_shared_regions.py`
- Modify: `pipeline/tests/test_typesetting_renderer.py`
- Modify: `pipeline/tests/test_typesetting_style_policy.py`

**Step 1: Inspecionar merges/splits atuais e diffs**

```powershell
git status --short -- pipeline/layout/balloon_layout.py pipeline/typesetter
git diff -- pipeline/layout/balloon_layout.py pipeline/typesetter
rg -n "def (enrich_page_layout|build_render_blocks|_split_text_for_connected_balloons|render_band_image)|_merge_adjacent|_split_single_ocr|_drop_low_quality" pipeline/layout/balloon_layout.py pipeline/typesetter/renderer.py
```

**Step 2: Escrever testes RED**

```python
def test_renderer_receives_one_complete_payload_per_owner(): ...
def test_connected_owner_may_use_many_layout_regions_but_one_translation(): ...
def test_normalized_chunks_reconstruct_exact_owner_payload(): ...
def test_renderer_never_merges_suppresses_or_reroutes_owners(): ...
def test_same_body_lines_share_font_size_and_safe_polygon(): ...
def test_font_size_stays_within_source_and_container_bounds(): ...
def test_text_is_centered_by_safe_polygon_not_ocr_bbox(): ...
def test_style_evidence_permutation_does_not_change_owner_body_route_or_mask(): ...
```

Inclua o caso metamórfico de remover/adicionar/mudar confiança de style evidence; somente o perfil visual pode variar.

**Step 3: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_owner_layout.py `
  pipeline/tests/test_balloon_layout_shared_regions.py `
  pipeline/tests/test_typesetting_renderer.py `
  pipeline/tests/test_typesetting_style_policy.py `
  -k "owner or connected or semantic or safe_polygon or style_evidence" -q
```

Expected: FAIL; `build_render_blocks` e `enrich_page_layout` ainda inferem/fundem bodies.

**Step 4: Remover autoridade semântica do layout/renderer**

`enrich_page_layout` recebe `OwnerGraph`/`layout_regions` e só calcula safe areas. No owner mode:

- não chamar merges semânticos baseados em `build_mask_regions`;
- não concatenar traduções no renderer;
- não criar/suprimir owner a partir de lobe, bbox ou style;
- `_split_text_for_connected_balloons` só produz chunks visuais determinísticos;
- `build_render_blocks` faz dedupe explícito por `owner_id` e rejeita duplicata divergente;
- `render_band_image` retorna `OwnerGlyphPatch` com glyph mask/bbox page-space.

Mantenha helpers antigos somente no adapter `legacy_unverified` até a Task 18.

**Step 5: Confirmar GREEN e Checkpoint B**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_owner_translation.py `
  pipeline/tests/test_owner_mask.py `
  pipeline/tests/test_owner_atomic_execution.py `
  pipeline/tests/test_owner_layout.py `
  pipeline/tests/test_balloon_layout_shared_regions.py `
  pipeline/tests/test_typesetting_renderer.py `
  pipeline/tests/test_typesetting_style_policy.py `
  -k "owner or connected or rollback or action_mask or style_evidence" -q
```

Expected: PASS; um body gera uma tradução e um render owner-scoped, independentemente das sub-regiões.

**Step 6: Commit**

```powershell
git add -- pipeline/layout/balloon_layout.py pipeline/typesetter/renderer.py pipeline/typesetter/style_policy.py pipeline/tests/test_owner_layout.py pipeline/tests/test_balloon_layout_shared_regions.py pipeline/tests/test_typesetting_renderer.py pipeline/tests/test_typesetting_style_policy.py
git diff --cached --name-only
git commit -m "feat: render resolved owners without semantic splits"
```

### Task 13: Implementar o compositor page-space como função pura

**Files:**
- Create: `pipeline/compositor/__init__.py`
- Create: `pipeline/compositor/owner_compositor.py`
- Modify: `pipeline/ownership/model.py`
- Create: `pipeline/tests/test_owner_compositor.py`
- Create: `pipeline/tests/test_owner_compositor_properties.py`

**Step 1: Escrever testes RED do compositor**

```python
def test_changed_pixels_must_be_subset_of_owner_action_mask(): ...
def test_action_mask_cannot_touch_protected_art(): ...
def test_context_only_tile_cannot_write_pixels(): ...
def test_conflicting_owner_mutations_fail_closed(): ...
def test_identical_duplicate_patch_for_same_owner_is_applied_once(): ...
def test_divergent_duplicate_patch_for_same_owner_blocks(): ...
def test_all_inpaints_precede_all_glyph_patches(): ...
def test_composition_is_invariant_to_tile_partition_and_input_order(): ...
def test_diff_from_original_is_subset_of_owned_masks(): ...
def test_color_space_mismatch_is_rejected(): ...
def test_composition_hash_is_deterministic(): ...
```

**Step 2: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_owner_compositor.py `
  pipeline/tests/test_owner_compositor_properties.py `
  -q
```

Expected: FAIL com ausência de `compositor.owner_compositor`.

**Step 3: Implementar a única autoridade de pixels**

```python
def compose_page(
    original_rgb: np.ndarray,
    owner_mutations: Sequence[OwnerMutation],
    owner_glyph_patches: Sequence[OwnerGlyphPatch],
    protected_art_mask: np.ndarray,
) -> PageCompositionResult: ...
```

`PageCompositionResult` contém `final_rgb`, `cleanup_owner_map`, `glyph_owner_map`, `conflicts`, `write_counts` e `sha256`. Regras:

1. validar todas as mutations antes de escrever;
2. aplicar somente deltas sob `action_mask`;
3. rejeitar overlap entre owners sem política semântica explícita;
4. aplicar todos os glyphs depois de todos os inpaints;
5. aceitar somente RGB uint8 canônico e arrays com shape da página;
6. ordenar por `owner_id` apenas para determinismo, nunca por tile/band.

**Step 4: Confirmar GREEN e commit**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_owner_compositor.py `
  pipeline/tests/test_owner_compositor_properties.py `
  -q
git add -- pipeline/compositor/__init__.py pipeline/compositor/owner_compositor.py pipeline/ownership/model.py pipeline/tests/test_owner_compositor.py pipeline/tests/test_owner_compositor_properties.py
git diff --cached --name-only
git commit -m "feat: compose final pages by pixel ownership"
```

### Task 14: Integrar o compositor e retirar autoridade da ordem dos bands

**Files:**
- Modify: `pipeline/strip/run.py`
- Modify: `pipeline/strip/reassemble.py`
- Modify: `pipeline/strip/process_bands.py`
- Create: `pipeline/tests/test_strip_owner_composition_integration.py`
- Modify: `pipeline/tests/test_strip_run.py`
- Modify: `pipeline/tests/test_strip_reassemble.py`

**Step 1: Inspecionar diff e o paste atual**

```powershell
git status --short -- pipeline/strip/run.py pipeline/strip/reassemble.py pipeline/strip/process_bands.py
git diff -- pipeline/strip/run.py pipeline/strip/reassemble.py pipeline/strip/process_bands.py
rg -n "def (_paste_band_attr_into_image|assemble_output_pages|run_chapter)" pipeline/strip
```

**Step 2: Escrever testes RED de integração**

```python
def test_run_chapter_uses_owner_compositor_as_only_final_pixel_authority(): ...
def test_run_chapter_output_is_identical_after_band_permutation(): ...
def test_changed_context_from_overlapping_tile_never_restores_source(): ...
def test_production_composition_succeeds_without_debug_directory(): ...
def test_debug_crops_are_derived_from_final_page_only(): ...
def test_reassemble_defines_framing_but_not_pixel_precedence(): ...
```

**Step 3: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_strip_owner_composition_integration.py `
  pipeline/tests/test_strip_run.py `
  pipeline/tests/test_strip_reassemble.py `
  -k "owner_compositor or permutation or debug_directory or pixel_precedence" -q
```

Expected: FAIL; `_paste_band_attr_into_image` ainda é last-write-wins por band.

**Step 4: Integrar em `run_chapter`**

- Colete `OwnerMutation`/`OwnerGlyphPatch` dos executores.
- Converta-os para page-space e chame `compose_page` uma vez por página.
- Use `assemble_output_pages` somente para limites/framing.
- Faça cleanup/rerender adicional produzir novas mutations antes da composição; não altere o canvas depois.
- `final_band_crops.jsonl` passa a ser derivado do resultado final read-only.
- Mantenha `_paste_band_attr_into_image` apenas dentro do adapter legado; owner mode não pode chamá-lo.
- Não leia `post_copyback`, `rendered_band`, `debug_inpaint` ou JPEG de debug como input.

**Step 5: Confirmar GREEN e commit**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_owner_compositor.py `
  pipeline/tests/test_owner_compositor_properties.py `
  pipeline/tests/test_strip_owner_composition_integration.py `
  pipeline/tests/test_strip_run.py `
  pipeline/tests/test_strip_reassemble.py `
  -k "owner or compositor or permutation or reassemble" -q
git add -- pipeline/strip/run.py pipeline/strip/reassemble.py pipeline/strip/process_bands.py pipeline/tests/test_strip_owner_composition_integration.py pipeline/tests/test_strip_run.py pipeline/tests/test_strip_reassemble.py
git diff --cached --name-only
git commit -m "feat: make owner compositor the final pixel authority"
```

### Task 15: Observar e auditar independentemente os bytes finais persistidos

**Files:**
- Create: `pipeline/qa/final_pixel_observer.py`
- Create: `pipeline/qa/final_pixel_qa.py`
- Modify: `pipeline/qa/visual_text_leak.py`
- Create: `pipeline/tests/test_final_pixel_observer.py`
- Create: `pipeline/tests/test_final_pixel_qa.py`
- Create: `pipeline/tests/test_final_pixel_qa_integration.py`
- Modify: `pipeline/tests/test_visual_text_leak.py`

**Step 1: Escrever testes RED do observer independente**

```python
def test_observer_receives_persisted_file_without_project_boxes(): ...
def test_default_observer_runs_fresh_detector_then_ocr_on_persisted_pixels(): ...
def test_observer_hashes_exact_bytes_it_reads(): ...
def test_source_payload_visible_in_final_pixels_blocks_with_empty_metadata_flags(): ...
def test_independently_detected_text_without_owner_blocks(): ...
def test_missing_owner_glyphs_block_even_when_render_state_is_complete(): ...
def test_owner_glyphs_survive_overlap_composition(): ...
def test_pixel_change_outside_owned_masks_blocks(): ...
def test_protected_art_damage_blocks(): ...
def test_preserved_name_sfx_or_credit_requires_explicit_policy(): ...
def test_source_language_detection_uses_owner_ngrams_not_fixed_phrase_list(): ...
```

**Step 2: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_final_pixel_observer.py `
  pipeline/tests/test_final_pixel_qa.py `
  pipeline/tests/test_final_pixel_qa_integration.py `
  pipeline/tests/test_visual_text_leak.py `
  -q
```

Expected: FAIL; os módulos ainda não existem e `visual_text_leak` usa padrões fixos.

**Step 3: Implementar observer e contratos**

Interface injetável:

```python
class FinalPixelObserver(Protocol):
    def observe(self, image_path: Path, *, source_language: str) -> FinalPixelObservation: ...

class DetectorOcrFinalPixelObserver:
    def __init__(self, *, detector, runtime): ...
    def observe(self, image_path: Path, *, source_language: str) -> FinalPixelObservation: ...

def evaluate_final_pixels(
    *,
    image_path: Path,
    graph: OwnerGraph,
    composition: PageCompositionResult,
    observer: FinalPixelObserver,
) -> FinalPixelQaReport: ...
```

O observer concreto abre o arquivo final persistido, calcula SHA-256, chama o detector sobre a página inteira e entrega somente os blocos recém-detectados ao OCR runtime. Ele não recebe project bboxes, final crops, `render_completed`, QA flags nem layers aceitos. Somente depois da observação o QA associa geometrias ao graph. O automático deve construir `DetectorOcrFinalPixelObserver`; a interface injetável existe para testes e providers futuros, não para permitir um mock no runtime de produção.

O report precisa cobrir:

- `source_coverage_contract`;
- `owner_graph_contract`;
- `route_state_contract`;
- `pixel_ownership_contract`;
- `final_language_contract` por n-grams do source real;
- `qa_integrity_contract`.

Cada issue contém `issue_id`, `page_id`, `owner_id`, `component_ids`, `severity`, `reason` e `offenders[]`.

**Step 4: Confirmar GREEN e commit**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_final_pixel_observer.py `
  pipeline/tests/test_final_pixel_qa.py `
  pipeline/tests/test_final_pixel_qa_integration.py `
  pipeline/tests/test_visual_text_leak.py `
  -q
git add -- pipeline/qa/final_pixel_observer.py pipeline/qa/final_pixel_qa.py pipeline/qa/visual_text_leak.py pipeline/tests/test_final_pixel_observer.py pipeline/tests/test_final_pixel_qa.py pipeline/tests/test_final_pixel_qa_integration.py pipeline/tests/test_visual_text_leak.py
git diff --cached --name-only
git commit -m "feat: audit persisted pages with independent OCR"
```

### Task 16: Fechar o export gate por cobertura, hash e inexistência de writer tardio

**Files:**
- Modify: `pipeline/main.py`
- Modify: `pipeline/qa/export_gate.py`
- Modify: `pipeline/project_writer.py`
- Create: `pipeline/tests/test_final_pixel_export_gate.py`
- Modify: `pipeline/tests/test_export_gate.py`
- Modify: `pipeline/tests/test_main_emit.py`
- Modify: `pipeline/tests/test_export_gate_debug_consistency.py`

**Step 1: Inspecionar a ordem viva e diffs locais**

```powershell
git status --short -- pipeline/main.py pipeline/qa/export_gate.py pipeline/project_writer.py
git diff -- pipeline/main.py pipeline/qa/export_gate.py pipeline/project_writer.py
rg -n "_run_post_rerender_final_visual_contract|final_translated_page_consistency_guard|evaluate_export_gate|render_page_image|_rerender_final_project_images_from_metadata" pipeline/main.py
```

**Step 2: Escrever testes RED**

```python
def test_automatic_export_requires_final_pixel_report_for_every_page(): ...
def test_missing_or_unavailable_observer_blocks_export(): ...
def test_missing_or_stale_artifact_hash_blocks_export(): ...
def test_post_qa_image_mutation_invalidates_report(): ...
def test_no_image_writer_runs_after_final_pixel_qa(): ...
def test_export_gate_identity_uses_owner_id_without_band_fallback(): ...
def test_summary_rows_offenders_and_gate_counts_are_identical(): ...
def test_render_failure_propagates_instead_of_counting_success(): ...
```

**Step 3: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_final_pixel_export_gate.py `
  pipeline/tests/test_export_gate.py `
  pipeline/tests/test_main_emit.py `
  pipeline/tests/test_export_gate_debug_consistency.py `
  -k "final_pixel or stale_artifact or post_qa or owner_id or render_failure" -q
```

Expected: FAIL; QA ausente pode passar e `final_translated_page_consistency_guard` ainda escreve depois do QA.

**Step 4: Reordenar e fechar o fluxo**

Ordem autoritativa:

```text
owner compose in memory
-> persist final image losslessly/canonical JPEG once
-> read persisted bytes and run FinalPixelObserver
-> verify persisted SHA-256 equals QA SHA-256
-> evaluate_export_gate
-> write project/report metadata only
```

Regras de implementação:

- mover qualquer consistency guard que altere pixels para antes da persistência ou torná-lo read-only;
- nenhum rerender, copyback, refresh ou restauração depois do QA;
- `_refresh_debug_final_band_crops_from_translated` não pode regravar `translated`;
- `_rerender_strip_reassembled_crops_from_metadata` é proibido em owner mode;
- `render_page_image` propaga falhas para o caller;
- relatório ausente, incompleto, observer indisponível, hash divergente ou página não coberta produz blocker crítico;
- `visible:false` não esconde source component sem disposição final.

**Step 5: Confirmar GREEN e Checkpoint C**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_owner_compositor.py `
  pipeline/tests/test_strip_owner_composition_integration.py `
  pipeline/tests/test_final_pixel_observer.py `
  pipeline/tests/test_final_pixel_qa.py `
  pipeline/tests/test_final_pixel_qa_integration.py `
  pipeline/tests/test_final_pixel_export_gate.py `
  pipeline/tests/test_export_gate.py `
  pipeline/tests/test_main_emit.py `
  pipeline/tests/test_export_gate_debug_consistency.py `
  -k "owner or final_pixel or export_gate or post_qa or hash" -q
```

Expected: PASS; um spy confirma zero writers após o observer.

**Step 6: Commit**

```powershell
git add -- pipeline/main.py pipeline/qa/export_gate.py pipeline/project_writer.py pipeline/tests/test_final_pixel_export_gate.py pipeline/tests/test_export_gate.py pipeline/tests/test_main_emit.py pipeline/tests/test_export_gate_debug_consistency.py
git diff --cached --name-only
git commit -m "fix: gate export on fresh final pixel evidence"
```

### Task 17: Publicar telemetria owner-first e garantir integridade do próprio QA

**Files:**
- Create: `pipeline/ownership/artifacts.py`
- Modify: `pipeline/debug_tools/recorder.py`
- Modify: `pipeline/strip/run.py`
- Modify: `pipeline/main.py`
- Modify: `pipeline/qa/export_gate.py`
- Create: `pipeline/tests/test_owner_debug_artifacts.py`
- Modify: `pipeline/tests/test_strip_ocr_debug_artifacts.py`
- Modify: `pipeline/tests/test_typeset_render_plan_debug.py`
- Modify: `pipeline/tests/test_render_plan_trace_integrity.py`
- Modify: `pipeline/tests/test_qa_flag_propagation_v2.py`

**Step 1: Escrever testes RED de artifacts/integridade**

Exigir a presença e o schema de:

```text
02_strip_detect/page_owner_components.jsonl
03_ocr/page_owner_observations.jsonl
04_text_normalization_router/page_owner_graph.json
06_mask_segmentation/owner_masks/<owner_id>/...
09_typeset/owner_render_plan.jsonl
10_copyback_reassemble/owner_composition.jsonl
11_qa_export_gate/owner_invariant_report.json
11_qa_export_gate/final_pixel_ocr.jsonl
11_qa_export_gate/owner_pixel_checks.jsonl
```

Testes:

```python
def test_every_artifact_row_has_schema_run_page_owner_and_coordinate_space(): ...
def test_render_plan_never_contains_review_required_owner(): ...
def test_all_qa_rows_have_composite_trace_and_offenders(): ...
def test_summary_row_and_gate_counts_match_exactly(): ...
def test_debug_artifacts_are_not_read_by_production_composer(): ...
```

**Step 2: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_owner_debug_artifacts.py `
  pipeline/tests/test_strip_ocr_debug_artifacts.py `
  pipeline/tests/test_typeset_render_plan_debug.py `
  pipeline/tests/test_render_plan_trace_integrity.py `
  pipeline/tests/test_qa_flag_propagation_v2.py `
  -k "owner or artifact or offenders or integrity" -q
```

Expected: FAIL por ausência dos novos artifacts/identidades.

**Step 3: Implementar writers derivados**

`artifacts.py` serializa somente snapshots recebidos do control plane/compositor/QA. Cada row inclui `schema_version`, `run_id`, `page_id`, `owner_id`, `trace_id`, `coordinate_space`, hashes e `offenders`. Não adicione readers desses caminhos ao pipeline automático.

`_write_debug_export_gate_artifacts` deve validar summary/rows/gate antes de persistir. Qualquer mismatch cria `qa_integrity_failure` crítico.

**Step 4: Confirmar GREEN e commit**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_owner_debug_artifacts.py `
  pipeline/tests/test_strip_ocr_debug_artifacts.py `
  pipeline/tests/test_typeset_render_plan_debug.py `
  pipeline/tests/test_render_plan_trace_integrity.py `
  pipeline/tests/test_qa_flag_propagation_v2.py `
  -k "owner or artifact or offenders or integrity" -q
git add -- pipeline/ownership/artifacts.py pipeline/debug_tools/recorder.py pipeline/strip/run.py pipeline/main.py pipeline/qa/export_gate.py pipeline/tests/test_owner_debug_artifacts.py pipeline/tests/test_strip_ocr_debug_artifacts.py pipeline/tests/test_typeset_render_plan_debug.py pipeline/tests/test_render_plan_trace_integrity.py pipeline/tests/test_qa_flag_propagation_v2.py
git diff --cached --name-only
git commit -m "feat: trace owner invariants across pipeline artifacts"
```

### Task 18: Ativar enforcement e isolar todos os reparos band-first como legado

**Files:**
- Create: `pipeline/ownership/legacy_adapter.py`
- Modify: `pipeline/strip/run.py`
- Modify: `pipeline/strip/process_bands.py`
- Modify: `pipeline/vision_stack/runtime.py`
- Modify: `pipeline/ocr/ocr_normalizer.py`
- Modify: `pipeline/layout/balloon_layout.py`
- Modify: `pipeline/typesetter/renderer.py`
- Modify: `pipeline/main.py`
- Create: `pipeline/tests/test_owner_enforcement.py`
- Modify: `pipeline/tests/test_project_migration.py`
- Modify: `pipeline/tests/test_main_emit.py`

**Step 1: Escrever testes RED de enforcement**

```python
def test_new_automatic_run_defaults_to_owner_enforce(): ...
def test_verified_graph_never_calls_cross_band_reconcile_or_late_layer_merge(): ...
def test_verified_graph_never_calls_debug_crop_rerender(): ...
def test_legacy_unverified_project_uses_explicit_adapter_only(): ...
def test_legacy_adapter_cannot_mark_graph_verified(): ...
def test_owner_mode_does_not_call_mask_or_renderer_semantic_merge_helpers(): ...
def test_semantic_modules_do_not_branch_on_work_chapter_page_number_or_band_id(): ...
```

O último teste deve usar AST sobre `pipeline/ownership`, `pipeline/compositor` e `pipeline/qa/final_pixel_*`, permitindo `tile_provenance` somente em serialização/log.

**Step 2: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_owner_enforcement.py `
  pipeline/tests/test_project_migration.py `
  pipeline/tests/test_main_emit.py `
  -k "enforce or legacy_adapter or late_layer_merge or semantic_modules" -q
```

Expected: FAIL enquanto shadow/merges tardios ainda forem ativos.

**Step 3: Ativar o caminho sistêmico**

- default do automático: `owner_graph_mode="enforce"`;
- `legacy` somente para projeto explicitamente `legacy_unverified`;
- mover `_reconcile_overlapping_band_ocr_fragments_before_translation`, `merge_same_balloon_fragments_before_translation`, `_merge_same_balloon_fragment_layers`, `_repair_project_split_lobe_text_payloads`, `_rehome_cross_page_band_layers` e rerenders baseados em debug para `legacy_adapter` ou guard explícito;
- no caminho verificado, esses helpers viram assertions/comparadores read-only;
- remover fallback de join por `band_id`/posição quando `owner_id` é obrigatório;
- manter leitura de projetos antigos sem reintroduzir autoridade band-first no automático novo.

**Step 4: Confirmar GREEN e commit**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_owner_enforcement.py `
  pipeline/tests/test_project_migration.py `
  pipeline/tests/test_owner_reconcile_properties.py `
  pipeline/tests/test_strip_owner_control_plane.py `
  pipeline/tests/test_strip_owner_composition_integration.py `
  pipeline/tests/test_main_emit.py `
  -k "owner or enforce or legacy or permutation" -q
git add -- pipeline/ownership/legacy_adapter.py pipeline/strip/run.py pipeline/strip/process_bands.py pipeline/vision_stack/runtime.py pipeline/ocr/ocr_normalizer.py pipeline/layout/balloon_layout.py pipeline/typesetter/renderer.py pipeline/main.py pipeline/tests/test_owner_enforcement.py pipeline/tests/test_project_migration.py pipeline/tests/test_main_emit.py
git diff --cached --name-only
git commit -m "refactor: enforce page owners over band semantics"
```

### Task 19: Executar regressão completa, matriz cross-work e validação visual

**Files:**
- Create: `pipeline/tools/validate_owner_visual_matrix.py`
- Create: `pipeline/tests/test_owner_visual_matrix_tool.py`
- Create: `docs/reports/2026-07-25-page-owner-systemic-validation.md`
- Generate only: `.codex-tmp/page_owner_systemic_validation_20260725/`
- Read: `docs/audits/2026-07-25-mythic-ch40-v11-visual-qa-root-cause-analysis.md`

**Step 1: Escrever teste RED do validador de matriz**

```python
def test_matrix_requires_multiple_works_and_visual_categories(): ...
def test_matrix_rejects_reused_output_directories(): ...
def test_matrix_requires_fresh_hash_and_non_blocked_export_gate(): ...
def test_matrix_report_groups_failures_by_contract_not_case_id(): ...
```

O tool recebe um manifest local de configs e nunca é importado pelo pipeline de produção.

**Step 2: Confirmar RED, implementar e confirmar GREEN**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_owner_visual_matrix_tool.py -q
```

Expected RED: módulo ausente.

Implementar CLI:

```powershell
pipeline\venv\Scripts\python.exe pipeline/tools/validate_owner_visual_matrix.py `
  --manifest .codex-tmp/page_owner_systemic_validation_20260725/matrix.json `
  --output-root .codex-tmp/page_owner_systemic_validation_20260725 `
  --report docs/reports/2026-07-25-page-owner-systemic-validation.md
```

O manifest local deve selecionar pelo menos três obras/capítulos distintos e cobrir: balão branco, translúcido, burst, painel escuro, card colorido, texto sobre arte, tabela/ranking, owner cruzando tiles e primary-OCR omission. Cada entry usa `work_dir` novo. O report exibe categoria, nunca cria regra por obra/caso.

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_owner_visual_matrix_tool.py -q
```

Expected GREEN: PASS.

**Step 3: Rodar a suíte sistêmica integrada**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_systemic_owner_contract_manifest.py `
  pipeline/tests/test_owner_graph.py `
  pipeline/tests/test_owner_coordinates.py `
  pipeline/tests/test_owner_reconcile.py `
  pipeline/tests/test_owner_reconcile_properties.py `
  pipeline/tests/test_owner_translation.py `
  pipeline/tests/test_strip_owner_control_plane.py `
  pipeline/tests/test_owner_mask.py `
  pipeline/tests/test_owner_atomic_execution.py `
  pipeline/tests/test_owner_layout.py `
  pipeline/tests/test_owner_compositor.py `
  pipeline/tests/test_owner_compositor_properties.py `
  pipeline/tests/test_strip_owner_composition_integration.py `
  pipeline/tests/test_final_pixel_observer.py `
  pipeline/tests/test_final_pixel_qa.py `
  pipeline/tests/test_final_pixel_qa_integration.py `
  pipeline/tests/test_final_pixel_export_gate.py `
  pipeline/tests/test_owner_debug_artifacts.py `
  pipeline/tests/test_owner_enforcement.py `
  pipeline/tests/test_owner_visual_matrix_tool.py `
  -q
```

Expected: PASS.

**Step 4: Rodar owners preexistentes e comparar com o baseline**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_vision_stack_ocr.py `
  pipeline/tests/test_vision_stack_runtime.py `
  pipeline/tests/test_strip_run.py `
  pipeline/tests/test_strip_process_bands.py `
  pipeline/tests/test_mask_builder.py `
  pipeline/tests/test_inpaint_mask_geometry.py `
  pipeline/tests/test_vision_stack_inpainter.py `
  pipeline/tests/test_balloon_layout_shared_regions.py `
  pipeline/tests/test_typesetting_renderer.py `
  pipeline/tests/test_typesetting_style_policy.py `
  pipeline/tests/test_project_schema_v12.py `
  pipeline/tests/test_project_migration.py `
  pipeline/tests/test_project_writer.py `
  pipeline/tests/test_main_emit.py `
  pipeline/tests/test_export_gate.py `
  pipeline/tests/test_export_gate_debug_consistency.py `
  pipeline/tests/test_qa_flag_propagation_v2.py `
  pipeline/tests/test_visual_text_leak.py `
  -q | Tee-Object .codex-tmp\page_owner_final_focused_20260725.txt
```

Expected: nenhum nodeid anteriormente verde regride; falhas preexistentes só permanecem se não forem ownership/cobertura/composição/QA.

**Step 5: Rodar a matriz E2E em outputs novos**

```powershell
$env:PYTHONPATH='N:\TraduzAI\pipeline'
$env:TRADUZAI_SKIP_LOCAL_VENV_REEXEC='1'
$env:TRADUZAI_PAGE_OWNER_GRAPH_MODE='enforce'
pipeline\venv\Scripts\python.exe pipeline/tools/validate_owner_visual_matrix.py `
  --manifest .codex-tmp/page_owner_systemic_validation_20260725/matrix.json `
  --output-root .codex-tmp/page_owner_systemic_validation_20260725 `
  --report docs/reports/2026-07-25-page-owner-systemic-validation.md
```

Expected para cada entry:

```text
pipeline technical status: complete
owner graph: valid
source_text_unowned: 0
semantic_body_split: 0
duplicate_visible_owner: 0
review_required in render plan: 0
changed_outside_owner_pixels: 0
protected_art_damage: 0
unallowed_source_language_residual: 0
post_qa_writes: 0
final artifact hash: matches QA
export_gate: PASS
```

Se qualquer entry bloquear, mantenha o BLOCK, classifique pelo contrato sistêmico e volte à primeira task que possui o defeito. Não adicione exceção ao caso.

**Step 6: Validar visualmente os outputs**

O tool deve gerar contact sheets `before | inpaint | final | masks | owner map` por categoria. Abra cada sheet e confirme:

- nenhum texto-fonte residual ou PT-BR pela metade;
- nenhum body repartido, duplicado ou com linhas concorrentes;
- texto dentro do safe polygon, centrado e com tamanho legível/proporcional;
- cards mantêm title/body/footer completos e hierarquia coerente;
- inpaint cobre todos os glyphs-fonte sem resíduos;
- personagem, ícone, borda e textura protegidos permanecem íntegros;
- overlaps não criam banding/cor diferente nem restauram source;
- nomes/SFX preservados aparecem apenas quando a policy explícita permite;
- style-copy não entra no GO/NO-GO desta entrega, salvo quando rompe legibilidade funcional.

Anexe caminhos absolutos das sheets e imagens finais ao report. Um resultado tecnicamente completo, mas visualmente ruim, permanece NO-GO.

**Step 7: Rodar validações finais de repositório**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/Test-TraduzAISkills.ps1
git diff --check
git status --short
```

Expected: skills PASS; zero novo erro de whitespace; todas as alterações locais preexistentes continuam presentes.

**Step 8: Commit do harness e relatório**

Não adicione `.codex-tmp`.

```powershell
git add -- pipeline/tools/validate_owner_visual_matrix.py pipeline/tests/test_owner_visual_matrix_tool.py docs/reports/2026-07-25-page-owner-systemic-validation.md
git diff --cached --name-only
git commit -m "test: validate page owners across visual categories"
```

## Critérios de encerramento

Antes de declarar o plano completo, todos devem ser verdadeiros:

1. owner nasce page-global antes da primeira tradução;
2. todos os componentes-fonte têm owner ou disposição explícita;
3. cada body possui um payload, uma tradução e no máximo um render;
4. partição/permutação de bands não altera graph, requests ou pixels finais;
5. action masks são owner-scoped e disjuntas de protected art;
6. layout/renderer não criam, fundem, suprimem nem reroteiam owners;
7. compositor page-space é a única autoridade de pixels;
8. debug artifacts não são inputs de produção;
9. QA observa bytes finais fresh e cobre todas as páginas;
10. nenhum writer roda depois do QA;
11. hash do QA coincide com o arquivo exportado;
12. export gate está em PASS sem override;
13. matriz cross-work está verde e visualmente aprovada;
14. nenhuma regra por obra, capítulo, página, texto ou band foi introduzida;
15. mudanças locais preexistentes permanecem intactas e são listadas no handoff.

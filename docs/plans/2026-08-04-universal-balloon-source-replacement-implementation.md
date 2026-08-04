# Universal Balloon Source Replacement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Garantir que todo texto-fonte em inglês dentro de balões seja removido e que o payload PT-BR correspondente seja materializado, usando recuperação automática progressiva em vez de concluir com inglês ou bloquear por defeito de conteúdo.

**Architecture:** Introduzir um coordenador owner-first por página com resultados OCR request-scoped, ledger de cobertura, tradução vinculada por IDs/hashes, transação atômica cleanup+render e controlador de reparo R0-R3. Bands permanecem somente como projeções de desempenho; o QA final solicita reparo e a reconstrução integral do interior do container é o fallback funcional obrigatório.

**Tech Stack:** Python 3.12, dataclasses/type hints, NumPy, OpenCV, PaddleOCR, pipeline owner graph existente, pytest com matrizes determinísticas, artefatos JSON/JSONL e validação visual lossless.

---

## Regras obrigatórias de execução

- Use `@executing-plans`, `@test-driven-development`, `@traduzai-pipeline`, `@traduzai-detect`, `@traduzai-ocr`, `@traduzai-translation`, `@traduzai-inpaint`, `@traduzai-typesetting` e `@verification-before-completion` nas fronteiras correspondentes.
- Trabalhe no checkout existente `N:\TraduzAI`; não crie worktree limpo.
- Preserve integralmente mudanças locais. Não use `reset`, `checkout`, `restore`, `stash`, `clean` ou equivalentes.
- Antes de cada Task, execute `git status --short --branch` e `git diff -- <arquivos da task>`.
- Arquivos centrais e testes já possuem mudanças locais. Edite por hunks pequenos; nunca substitua um arquivo inteiro.
- `pipeline/tests/test_owner_source_replacement_fail_closed.py` já existe como arquivo local não rastreado. Trate-o como trabalho do usuário e migre suas expectativas somente na Task indicada.
- Não adicione `DEBUGM/**`, `.codex-tmp/**`, outputs, modelos ou imagens de execução ao Git.
- Cada alteração de comportamento começa com teste RED, confirma o motivo da falha, implementa o mínimo e termina GREEN.
- Nenhuma regra de produção pode mencionar Mitch Items, capítulo 39, números de página, frases, paths ou coordenadas da regressão.
- Checkpoints são automáticos: continue enquanto estiverem verdes; pare apenas diante de falha real que não possa ser resolvida dentro da Task.
- O baseline funcional roda com `style_copy_mode=off`; style-copy não participa do veredito de conteúdo. `shadow` é diagnóstico opcional e `enforce` é apenas a run de compatibilidade separada.
- Não enfraqueça máscara, proteção de arte, residual ou atomicidade para evitar um bloqueio. Falha precisa escalar a estratégia R0 → R1 → R2 → R3.
- O gate continua sendo defesa; o controlador deve consumir e corrigir issues de conteúdo antes do gate final. Não transforme erro crítico em warning para obter `PASS`.
- Commits devem conter somente hunks da Task. Antes de cada commit: `git diff --cached --check` e `git diff --cached -- <arquivos>`.
- No início de cada Task, execute `git diff --cached --quiet`; se o índice não estiver vazio, não use restore/reset para limpá-lo e não faça commit misto. Registre a lista e preserve-a até obter uma fronteira segura. No fluxo normal deste plano, o índice deve estar vazio após cada commit.
- Antes de `git commit`, compare `git diff --cached --name-only` com a whitelist `Files` da Task e falhe diante de qualquer path extra. Depois do commit, confirme novamente `git diff --cached --quiet`. Isso protege mudanças staged do usuário; `git commit` simples só é permitido após essa prova.
- Cada Step “Confirmar RED” deve executar primeiro os nodeids novos isoladamente e comprovar a assertion/comportamento pretendido, não falha acidental de fixture/dependência. Quando a Task cria um módulo, a primeira collection failure é apenas preflight estrutural: crie imediatamente o esqueleto mínimo da API lançando `NotImplementedError`, colete o teste e observe o RED comportamental antes da implementação GREEN. A lista de arquivos do Step é a regressão GREEN; quando um bloco RED abaixo mostrar arquivos inteiros, selecione explicitamente os nodeids definidos no Step 1 antes de implementar.

## Baseline e evidência fixa

- Design aprovado: `docs/plans/2026-08-04-universal-balloon-source-replacement-design.md`.
- Commit do design: `6d5863d9`.
- Branch inspecionada: `Troca_de_motores`.
- HEAD antes deste plano: `6d5863d9`.
- Run diagnóstica principal: `.codex-tmp/mch39_full_directional_gradient_20260803_retry2/out`.
- Fonte real principal: `temporario1/mch39`.
- Páginas com inglês material confirmado: 10, 11, 19, 21, 27, 28, 30, 34, 36 e 39.
- Categorias genéricas: `cross_page_ocr_leak`, `no_band_component`, `unassociated_full_page_observation`, `semantic_container_missing`, `atomic_cleanup_rollback`, `mixed_language_overlay`, `final_target_missing` e `qa_duplicate_false_positive`.

## Invariantes de aceitação

```text
english_dialogue_residual_count == 0
translatable_components_without_owner == 0
material_components_without_ocr_attempt == 0
owners_without_valid_pt_br == 0
owners_without_atomic_cleanup_render == 0
owners_without_target_materialization == 0
material_components_without_terminal_lifecycle == 0
```

Além das contagens:

- um componente material dentro de balão termina somente `final_verified` no caminho novo (`verified` permanece alias legado de leitura);
- nomes/SFX/créditos preservados possuem policy explícita e não autorizam frase inglesa;
- nenhuma inferência cruza `run_id`, `page_id`, `page_source_sha256` ou pixels: o boundary mais baixo calcula o hash do ndarray RGB real imediatamente antes de **cada** chamada física ao provider;
- aliases correlacionados contam como uma inferência;
- toda recuperação começa dos pixels originais lossless;
- cleanup e glyph patch PT-BR fazem commit na mesma transação;
- R3 não altera pixels fora do interior seguro do container;
- `final_file_sha256` auditado possui o mesmo hash dos bytes PNG exportados; `page_output_pixel_sha256` identifica separadamente os pixels RGB decodificados.

Semântica única de identidade: `page_source_sha256` é SHA-256 canônico de `width + height + mode=RGB + decoded RGB bytes` da página original, independente do formato/metadados do arquivo. `source_file_sha256` registra os bytes externos apenas para auditoria. OCR, ledger, owner, replay e reparo preservam sempre `page_source_sha256`. Cada invocação lógica registra `root_input_pixel_sha256` dos pixels integrais que originaram a análise; cada tentativa física registra separadamente `input_pixel_sha256` do ndarray RGB **efetivamente entregue** ao provider, após crop/gray/inversão/escala. No OCR inicial page-global, `root_input_pixel_sha256 == page_source_sha256`; `OCRAttempt.input_pixel_sha256 == root_input_pixel_sha256` somente no attempt full-page nativo sem transformação. Downscale/resize, crop, gray, inversão, 2x, rotação e deskew possuem hash físico próprio ligado por parent + spec canônico. No observer do candidate final, `root_input_pixel_sha256 == page_output_pixel_sha256`, enquanto hashes de tentativas transformadas podem diferir.

## Checkpoints contínuos

1. **Checkpoint A — identidade OCR:** Tasks 1–3; nenhuma página ou inferência pode contaminar outra.
2. **Checkpoint B — cobertura e owner:** Tasks 4–7; nenhum texto material depende da existência de band.
3. **Checkpoint C — target PT-BR:** Task 8; nenhum diálogo inglês é aceito como tradução.
4. **Checkpoint D — transação e reparo:** Tasks 9–13; rollback sempre escala até R3 e termina com cleanup+target.
5. **Checkpoint E — QA reparador:** Tasks 14–16; QA reabre a página e não duplica/fabrica evidência.
6. **Checkpoint F — simplificação e prova:** Tasks 17–20; `enforce` possui um caminho autoritativo e a matriz real retorna GO.

### Task 1: Fixar o corpus universal e caracterizar as falhas reais

**Files:**

- Create: `pipeline/tests/fixtures/english_owner_recovery/manifest.json`
- Create: `pipeline/tests/fixtures/english_owner_recovery/recipes.json`
- Create: `pipeline/tests/test_universal_source_replacement_manifest.py`
- Read only: `.codex-tmp/mch39_full_directional_gradient_20260803_retry2/out/debug/e2e/**`
- Read only: `temporario1/mch39/**`

**Step 1: Registrar o estado e os diffs que precisam ser preservados**

Run:

```powershell
git status --short --branch
git diff -- pipeline/vision_stack/ocr.py pipeline/vision_stack/runtime.py pipeline/ownership pipeline/strip/run.py pipeline/strip/process_bands.py pipeline/qa pipeline/tests
git status --short -- pipeline/tests/test_owner_source_replacement_fail_closed.py
```

Expected: checkout muito sujo; nenhum arquivo é revertido ou limpo.

**Step 2: Rodar o baseline focado antes de adicionar comportamento**

Run:

```powershell
Push-Location pipeline
try {
  $baselineOutput = & .\venv\Scripts\python.exe -m pytest `
    tests/test_vision_stack_ocr.py `
    tests/test_vision_stack_runtime.py `
    tests/test_source_component_discovery.py `
    tests/test_owner_evidence.py `
    tests/test_owner_reconcile.py `
    tests/test_owner_reconcile_properties.py `
    tests/test_owner_graph.py `
    tests/test_project_migration.py `
    tests/test_project_schema_v12.py `
    tests/test_project_writer.py `
    tests/test_strip_owner_control_plane.py `
    tests/test_owner_translation.py `
    tests/test_owner_atomic_execution.py `
    tests/test_final_pixel_qa.py `
    tests/test_final_pixel_observer.py `
    tests/test_final_pixel_export_gate.py `
    tests/test_owner_visual_matrix_tool.py `
    -q 2>&1
  $baselineExit = $LASTEXITCODE
  $baselineOutput | Tee-Object ..\.codex-tmp\universal_source_replacement_baseline_20260804.txt
  Write-Output ("baseline_exit=" + $baselineExit)
} finally { Pop-Location }
```

Expected: registrar nodeids, exit code e falhas preexistentes sem alterar expectativas para escondê-las. Um baseline já vermelho é caracterizado aqui e deve estar verde no Checkpoint final; ele não deve ser confundido com o RED específico da Task seguinte.

**Step 3: Escrever o teste RED do manifest genérico**

```python
REQUIRED_CATEGORIES = {
    "cross_page_ocr_leak",
    "no_band_component",
    "unassociated_full_page_observation",
    "semantic_container_missing",
    "atomic_cleanup_rollback",
    "mixed_language_overlay",
    "final_target_missing",
    "qa_duplicate_false_positive",
    "burst_container",
    "card_table_multirole",
    "text_over_protected_art",
    "cross_tile_owner",
}


def test_manifest_covers_every_universal_failure_category():
    manifest = _load_manifest()
    recipes = _load_recipes()
    assert {case["category"] for case in manifest["cases"]} >= REQUIRED_CATEGORIES
    assert {case["recipe_id"] for case in manifest["cases"]} <= set(recipes)
    assert all("work_title" not in case for case in manifest["cases"])
    assert all("page_number_rule" not in case for case in manifest["cases"])


def test_real_regression_uses_hashed_external_inputs_not_production_rules():
    manifest = _load_manifest()
    real = manifest["real_regressions"]["mitch_items_ch39"]
    assert real["input_key"] == "TRADUZAI_MATRIX_MITCH39_SOURCE"
    assert set(real["sentinel_pages"]) == {10, 11, 19, 21, 27, 28, 30, 34, 36, 39}
    assert all(case["expected"]["english_dialogue_residual_count"] == 0 for case in manifest["cases"])
```

**Step 4: Confirmar RED**

Run:

```powershell
Push-Location pipeline
try { .\venv\Scripts\python.exe -m pytest tests/test_universal_source_replacement_manifest.py -q } finally { Pop-Location }
```

Expected: FAIL porque o manifest ainda não existe.

**Step 5: Criar somente o manifest e as receitas declarativas**

O manifest deve:

- usar nomes/categorias genéricos para casos sintéticos;
- registrar a regressão real por `input_key`, hashes e páginas sentinela;
- exigir style-copy off/on como duas variantes da mesma entrada;
- exigir `target_present=true`, não apenas `source_absent=true`;
- separar `calibration` e `holdout`;
- não conter path absoluto nem `.codex-tmp` como dependência versionada.

`recipes.json` descreve os casos sintéticos como primitivas genéricas (canvas, container, glyph layers, idioma e defeito induzido), sem screenshot da obra. Nesta Task os testes validam somente schema/cobertura; o primeiro RED comportamental que executa uma receita entra junto do coordenador na Task 9.

**Step 6: Confirmar GREEN**

Run:

```powershell
Push-Location pipeline
try { .\venv\Scripts\python.exe -m pytest tests/test_universal_source_replacement_manifest.py -q } finally { Pop-Location }
```

Expected: PASS.

**Step 7: Commit**

```powershell
git add -- pipeline/tests/fixtures/english_owner_recovery/manifest.json pipeline/tests/fixtures/english_owner_recovery/recipes.json pipeline/tests/test_universal_source_replacement_manifest.py
git diff --cached --check
git commit -m "test: characterize universal source replacement failures"
```

### Task 2: Tornar o resultado OCR atômico e request-scoped

**Files:**

- Create: `pipeline/ownership/ocr_contract.py`
- Modify: `pipeline/vision_stack/ocr.py:590-724,1735-2020`
- Modify: `pipeline/vision_stack/runtime.py:2226-2245,13732-13827,15480-15582`
- Test: `pipeline/tests/test_vision_stack_ocr.py`
- Test: `pipeline/tests/test_vision_stack_runtime.py`

**Step 1: Inspecionar os diffs existentes**

```powershell
git diff -- pipeline/vision_stack/ocr.py pipeline/vision_stack/runtime.py pipeline/tests/test_vision_stack_ocr.py pipeline/tests/test_vision_stack_runtime.py
```

**Step 2: Escrever testes RED de isolamento sequencial e concorrente**

Adicione testes com fake Paddle e `threading.Barrier`:

Os testes de concorrência/imutabilidade entram em `PaddleBlockMappingTests` (unittest, portanto recebem `self`); o teste de consumo entra em `VisionStackRuntimeTests` e usa `unittest.mock.patch`/injeção de helper, não fixture pytest em método unittest.

```python
def test_full_page_line_records_are_request_scoped_across_parallel_pages(self):
    engine = _interleaving_fake_engine()
    with ThreadPoolExecutor(max_workers=2) as pool:
        page_a = pool.submit(engine.recognize_page_with_evidence, _page("A"), [], request=_request("page_001"))
        page_b = pool.submit(engine.recognize_page_with_evidence, _page("B"), [], request=_request("page_002"))
    assert {line.text for line in page_a.result().full_page_lines} == {"PAGE A"}
    assert {line.text for line in page_b.result().full_page_lines} == {"PAGE B"}
    assert page_a.result().page_id == "page_001"
    assert page_b.result().page_id == "page_002"


def test_runtime_never_reads_last_full_page_side_channel(self):
    engine = _fake_atomic_ocr_result("page_010", "CURRENT PAGE")
    engine._last_full_page_line_records = [{"text": "STALE PAGE"}]
    result = _run_runtime_ocr(engine, page_id="page_010")
    assert "CURRENT PAGE" in _texts(result)
    assert "STALE PAGE" not in _texts(result)


def test_ocr_invocation_result_is_deeply_immutable(self):
    result = _fake_atomic_ocr_result("page_001", "TEXT")
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.blocks[0].text = "MUTATED"
    with pytest.raises(TypeError):
        result.blocks[0].extras["nested"] = "MUTATED"
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.diagnostics.provider = "other"
    with pytest.raises(TypeError):
        result.diagnostics.extras["nested"] = "MUTATED"


def test_every_ocr_record_links_exact_logical_request_and_physical_attempt(self):
    request = _request(
        run_id="run-a",
        page_id="page_010",
        page_source_sha256="sha-page-10",
        root_input_pixel_sha256="sha-root-page-10",
        invocation_id="ocr-10-primary",
    )
    result = _fake_atomic_ocr_result(request=request, text="CURRENT PAGE")
    attempts = {attempt.attempt_id: attempt for attempt in result.attempts}
    for record in (*result.observations, *result.full_page_lines):
        assert record.request_identity == request.identity
        attempt = attempts[record.attempt_id]
        assert record.attempt_identity == attempt.identity
        assert record.root_input_pixel_sha256 == request.root_input_pixel_sha256
        assert record.input_pixel_sha256 == attempt.input_pixel_sha256


def test_ocr_result_rejects_record_with_mismatched_request_identity(self):
    request = _request(run_id="run-a", page_id="page_010")
    stale = _ocr_record(run_id="run-previous", page_id="page_010")
    with pytest.raises(OCRRequestIdentityError):
        OCRInvocationResult.build(request=request, observations=(stale,))


def test_provider_boundary_rejects_declared_hash_b_when_pixels_a_are_passed(self):
    physical_kinds = (
        "full_page",
        "anchored_crop",
        "native",
        "gray",
        "inverted",
        "scale_2x",
        "final_observer",
        "external_auditor",
    )
    for physical_kind in physical_kinds:
        with self.subTest(physical_kind=physical_kind):
            provider = MagicMock()
            with self.assertRaises(OCRInputPixelIdentityError):
                execute_hash_bound_provider_attempt(
                    request=_request(root_input_pixel_sha256=canonical_page_sha256(_pixels_a())),
                    root_input_rgb=_pixels_a(),
                    actual_input_rgb=_pixels_a(),
                    expected_input_pixel_sha256=canonical_page_sha256(_pixels_b()),
                    variant_id=physical_kind,
                    transform_spec=_identity_transform_spec(),
                    provider=provider,
                )
            provider.assert_not_called()


def test_provider_boundary_records_hash_of_actual_array_on_success(self):
    provider = MagicMock(return_value=_raw_ocr("TEXT"))
    attempt, records = execute_hash_bound_provider_attempt(
        request=_request(root_input_pixel_sha256=canonical_page_sha256(_root_pixels())),
        root_input_rgb=_root_pixels(),
        actual_input_rgb=_transformed_pixels(),
        expected_input_pixel_sha256=canonical_page_sha256(_transformed_pixels()),
        variant_id="scale_2x",
        transform_spec=_scale_2x_transform_spec(_root_pixels(), _transformed_pixels()),
        provider=provider,
    )
    assert attempt.input_pixel_sha256 == canonical_page_sha256(_transformed_pixels())
    assert all(record.attempt_id == attempt.attempt_id for record in records)
    assert all(record.input_pixel_sha256 == attempt.input_pixel_sha256 for record in records)
    assert attempt.provider_called is True
    assert attempt.cache_hit is False
    provider.assert_called_once()


def test_parameterized_rotation_and_deskew_specs_replay_exact_provider_pixels(self):
    for kind in ("rotate_affine", "deskew_affine"):
        with self.subTest(kind=kind):
            root, spec, expected_rgb = _parameterized_affine_case(kind, angle_degrees=7.125)
            provider = MagicMock(return_value=_raw_ocr("TEXT"))
            attempt, _ = execute_hash_bound_provider_attempt(
                request=_request(root_input_pixel_sha256=canonical_page_sha256(root)),
                root_input_rgb=root,
                actual_input_rgb=expected_rgb,
                expected_input_pixel_sha256=canonical_page_sha256(expected_rgb),
                variant_id=kind,
                transform_spec=spec,
                provider=provider,
            )
            assert np.array_equal(spec.replay(root), provider.call_args.args[0])
            assert attempt.transform_spec.canonical_json_bytes == spec.canonical_json_bytes
            assert attempt.transform_spec.sha256 == sha256_bytes(spec.canonical_json_bytes)


def test_cache_or_dot_heuristic_attempt_is_not_fresh_physical_ocr(self):
    attempt = _cached_attempt(provider_called=False, cache_hit=True)
    assert attempt.qualifies_as_fresh_physical_inference is False


def test_production_provider_calls_exist_only_inside_hash_bound_boundary(self):
    assert find_direct_ocr_provider_call_sites(_production_ocr_modules()) == {
        "pipeline/vision_stack/ocr.py:execute_hash_bound_provider_attempt"
    }
```

Também exija `run_id`, `page_id`, `page_source_sha256`, `root_input_pixel_sha256`, `invocation_id`, `provider_family`, blocks, observations, full-page lines, tentativas físicas e diagnostics no mesmo retorno. Cada record referencia exatamente um `attempt_id`; `variant_id` e o hash do input físico pertencem à tentativa, não à request lógica.

**Step 3: Confirmar RED**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest `
    tests/test_vision_stack_ocr.py::PaddleBlockMappingTests::test_full_page_line_records_are_request_scoped_across_parallel_pages `
    tests/test_vision_stack_runtime.py::VisionStackRuntimeTests::test_runtime_never_reads_last_full_page_side_channel `
    tests/test_vision_stack_ocr.py::PaddleBlockMappingTests::test_ocr_invocation_result_is_deeply_immutable `
    tests/test_vision_stack_ocr.py::PaddleBlockMappingTests::test_every_ocr_record_links_exact_logical_request_and_physical_attempt `
    tests/test_vision_stack_ocr.py::PaddleBlockMappingTests::test_ocr_result_rejects_record_with_mismatched_request_identity `
    tests/test_vision_stack_ocr.py::PaddleBlockMappingTests::test_provider_boundary_rejects_declared_hash_b_when_pixels_a_are_passed `
    tests/test_vision_stack_ocr.py::PaddleBlockMappingTests::test_provider_boundary_records_hash_of_actual_array_on_success `
    tests/test_vision_stack_ocr.py::PaddleBlockMappingTests::test_parameterized_rotation_and_deskew_specs_replay_exact_provider_pixels `
    tests/test_vision_stack_ocr.py::PaddleBlockMappingTests::test_cache_or_dot_heuristic_attempt_is_not_fresh_physical_ocr `
    tests/test_vision_stack_ocr.py::PaddleBlockMappingTests::test_production_provider_calls_exist_only_inside_hash_bound_boundary `
    -q
} finally { Pop-Location }
```

Expected: FAIL porque full-page lines ainda vivem em `_last_full_page_line_records`.

**Step 4: Implementar os contratos imutáveis**

Em `ocr_contract.py`:

```python
@dataclass(frozen=True)
class OCRRequest:
    run_id: str
    page_id: str
    page_source_sha256: str
    root_input_pixel_sha256: str
    invocation_id: str
    provider_family: str


@dataclass(frozen=True)
class OCRTransformOperation:
    kind: Literal["identity", "crop", "resize", "grayscale_to_rgb", "invert", "rotate_affine", "deskew_affine"]
    bbox_page: BBox | None = None
    output_size: tuple[int, int] | None = None
    interpolation: Literal["nearest", "linear", "cubic", "area", "lanczos4"] | None = None
    border_mode: Literal["constant", "replicate", "reflect", "reflect101"] | None = None
    border_value_rgb: tuple[int, int, int] | None = None
    affine_matrix_fixed_1e6: tuple[int, int, int, int, int, int] | None = None
    algorithm_id: str = ""


@dataclass(frozen=True)
class OCRTransformSpec:
    schema_version: int
    operations: tuple[OCRTransformOperation, ...]
    canonical_json_bytes: bytes
    sha256: str

    @classmethod
    def build(cls, operations: tuple[OCRTransformOperation, ...]) -> "OCRTransformSpec": ...
    def replay(self, root_rgb: NDArray[np.uint8]) -> NDArray[np.uint8]: ...


@dataclass(frozen=True)
class OCRAttempt:
    attempt_id: str
    run_id: str
    page_id: str
    page_source_sha256: str
    root_input_pixel_sha256: str
    invocation_id: str
    provider_family: str
    variant_id: str
    input_pixel_sha256: str
    parent_input_pixel_sha256: str
    input_bbox_page: BBox | None
    input_kind: str
    transform_spec: OCRTransformSpec
    input_width: int
    input_height: int
    input_mode: Literal["RGB"]
    provider_called: bool
    cache_hit: bool

    @property
    def identity(self) -> tuple[str, str, str, str, str, str, str, str, str]: ...

    @property
    def qualifies_as_fresh_physical_inference(self) -> bool: ...


@dataclass(frozen=True)
class OCRObservationRecord:
    observation_id: str
    attempt_id: str
    run_id: str
    page_id: str
    page_source_sha256: str
    root_input_pixel_sha256: str
    input_pixel_sha256: str
    invocation_id: str
    provider_family: str
    variant_id: str
    payload_sha256: str
    text: str
    confidence: float
    bbox_page: BBox
    polygon_page: tuple[Point, ...]
    source: str

    @property
    def request_identity(self) -> tuple[str, str, str, str, str, str]: ...

    @property
    def attempt_identity(self) -> tuple[str, str, str, str, str, str, str, str, str]: ...


@dataclass(frozen=True)
class OCRInvocationResult:
    request: OCRRequest
    blocks: tuple[OCRBlock, ...]
    observations: tuple[OCRObservationRecord, ...]
    full_page_lines: tuple[OCRObservationRecord, ...]
    attempts: tuple[OCRAttempt, ...]
    attempt_chain_sha256: str
    diagnostics: OCRDiagnostics

    @property
    def run_id(self) -> str: return self.request.run_id
    @property
    def page_id(self) -> str: return self.request.page_id
    @property
    def page_source_sha256(self) -> str: return self.request.page_source_sha256
    @property
    def root_input_pixel_sha256(self) -> str: return self.request.root_input_pixel_sha256
    @property
    def invocation_id(self) -> str: return self.request.invocation_id
    @property
    def provider_family(self) -> str: return self.request.provider_family
```

Use records `frozen=True` também para `OCRBlock`, `OCRAttempt`, `OCRTransformOperation`, `OCRTransformSpec` e `OCRDiagnostics`; quaisquer extras JSON devem ser congelados recursivamente, sem dict/list mutável aninhado. `OCRObservationRecord` é o envelope OCR request-scoped desta Task, deliberadamente independente do modelo de ownership. `OCRInvocationResult.build()` valida que toda tentativa repete a identidade lógica da request, que todo record/linha referencia exatamente um `attempt_id`, que sua identidade/hash físico coincide com a tentativa referenciada e que `payload_sha256` corresponde ao texto normalizado. `OCRTransformSpec.build()` serializa operações em JSON canônico, recalcula `sha256` e rejeita combinação incompleta; rotação/deskew persistem matriz afim em inteiros fixos de 1e-6, algoritmo, interpolação, borda e output size, permitindo replay bit-exato sem depender de float JSON. `attempt_chain_sha256` é SHA-256 do JSON canônico da tupla na ordem real de execução, cobrindo todos os campos de identidade, parent/input, `transform_spec.canonical_json_bytes`, bbox, shape/mode e flags provider/cache; não ordenar por texto ou confidence.

Implemente `execute_hash_bound_provider_attempt(request, root_input_rgb, actual_input_rgb, transform_spec, ...)` no boundary mais baixo compartilhado por Paddle/runtime. Ele valida o hash do root contra a request, recebe o ndarray já cropado/transformado **e o `OCRTransformSpec` completo**, normaliza o array para RGB C-contiguous, calcula ali `canonical_page_sha256(width + height + mode + bytes)`, exige `transform_spec.replay(root_input_rgb)` bit-idêntico ao array físico e compara qualquer hash esperado **antes** de chamar o provider; mismatch lança `OCRInputPixelIdentityError` e o provider permanece sem chamadas. Somente esse boundary pode construir `OCRAttempt(provider_called=True, cache_hit=False)` e os records derivados. Full-page, crop ancorado, nativa, cinza, invertida, 2x, rotação/deskew, observer terminal e auditor externo são obrigados a passar por ele. Cache/dot heuristic podem registrar attempts com `provider_called=False`, mas nunca satisfazem prova de OCR fresco. `parent_input_pixel_sha256` liga cada transformação à entrada anterior; `OCRTransformSpec.operations` contém a cadeia completa root → input físico, e o parent identifica o intermediário imediato. `root_input_pixel_sha256` identifica a página/candidate integral que originou a invocação; `OCRAttempt.input_pixel_sha256` identifica o array físico, portanto transformações podem ter hashes distintos sem virar votos independentes. Um teste AST proíbe chamadas diretas a `.ocr(...)`, processor/generate ou provider equivalente fora desse adapter permitido. A Task 3 faz a conversão 1:1 para `TextObservation`, sem regenerar identidade.

**Step 5: Introduzir `recognize_page_with_evidence()`**

O método deve:

- resetar somente estado local da chamada;
- executar detecção/recognition;
- encaminhar **cada** chamada física ao provider por `execute_hash_bound_provider_attempt()`, inclusive crops e variantes internas;
- montar `OCRInvocationResult` antes de retornar;
- não publicar resultados em atributos do engine;
- funcionar com `blocks=[]` quando solicitado em modo full-page;
- manter apenas modelo/cache como estado compartilhado.

Adapte os dois callers de runtime para consumir o resultado diretamente. Pare de chamar `getattr(ocr, "_last_full_page_line_records", ...)`.

**Step 6: Confirmar ausência de consumidores do side channel**

```powershell
$sideChannels = & rg -n "_last_full_page_line_records|_last_recognize_blocks_stats|_last_batch_cache_stats|_snapshot_ocr_engine_observations|get_last_observation_records" pipeline/vision_stack pipeline/strip pipeline/ownership
$sideChannelExit = $LASTEXITCODE
$sideChannels
if ($sideChannelExit -gt 1) { throw "rg side-channel scan failed: $sideChannelExit" }
```

Expected: nenhum consumidor de produção; durante migração, atributos podem existir somente em adapter explicitamente legado e coberto por teste, nunca no `enforce`.

**Step 7: Confirmar GREEN**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest tests/test_vision_stack_ocr.py tests/test_vision_stack_runtime.py -q
} finally { Pop-Location }
```

Expected: PASS.

**Step 8: Commit**

```powershell
git add -- pipeline/ownership/ocr_contract.py
git add -p -- pipeline/vision_stack/ocr.py pipeline/vision_stack/runtime.py pipeline/tests/test_vision_stack_ocr.py pipeline/tests/test_vision_stack_runtime.py
git diff --cached --check
git commit -m "fix: isolate OCR evidence per page request"
```

### Task 3: Registrar origem real e impedir consenso artificial

**Files:**

- Modify: `pipeline/ownership/model.py:172-260`
- Modify: `pipeline/ownership/coordinates.py:83-145`
- Modify: `pipeline/ownership/ocr_adapter.py:339-550`
- Modify: `pipeline/ownership/evidence.py`
- Modify: `pipeline/ownership/reconcile.py:274-520`
- Modify: `pipeline/ownership/project.py`
- Modify: `pipeline/schema/project_schema_v12.py`
- Modify: `pipeline/strip/process_bands.py:2375-2555,9930-9980`
- Modify: `pipeline/strip/run.py:5550-5700`
- Test: `pipeline/tests/test_owner_evidence.py`
- Test: `pipeline/tests/test_owner_reconcile.py`
- Test: `pipeline/tests/test_owner_reconcile_properties.py`
- Test: `pipeline/tests/test_strip_owner_control_plane.py`
- Modify constructor call site: `pipeline/tests/test_balloon_layout_shared_regions.py`
- Modify constructor call site: `pipeline/tests/test_final_pixel_qa.py`
- Modify constructor call site: `pipeline/tests/test_main_emit.py`
- Modify constructor/read call site: `pipeline/tests/test_ocr_adapter.py`
- Modify constructor call site: `pipeline/tests/test_owner_atomic_execution.py`
- Modify constructor call site: `pipeline/tests/test_owner_compositor.py`
- Modify constructor call site: `pipeline/tests/test_owner_enforcement.py`
- Modify constructor call site: `pipeline/tests/test_owner_graph.py`
- Modify constructor call site: `pipeline/tests/test_owner_layout.py`
- Modify constructor/read call site: `pipeline/tests/test_owner_mask.py`
- Modify constructor call site: `pipeline/tests/test_owner_render_geometry.py`
- Modify constructor call site: `pipeline/tests/test_owner_style_capture.py`
- Modify constructor call site: `pipeline/tests/test_owner_style_profile.py`
- Modify constructor call site: `pipeline/tests/test_owner_translation.py`
- Modify constructor call site: `pipeline/tests/test_strip_owner_composition_integration.py`
- Modify constructor call site: `pipeline/tests/test_typesetting_renderer.py`
- Test schema/persistence: `pipeline/tests/test_project_migration.py`
- Test schema/persistence: `pipeline/tests/test_project_schema_v12.py`
- Test schema/persistence: `pipeline/tests/test_project_writer.py`

**Step 1: Inspecionar e preservar as alterações locais de reconciliação**

```powershell
git diff -- pipeline/ownership/model.py pipeline/ownership/coordinates.py pipeline/ownership/ocr_adapter.py pipeline/ownership/evidence.py pipeline/ownership/reconcile.py pipeline/strip/process_bands.py pipeline/strip/run.py
```

**Step 2: Escrever os testes RED da estrutura real da página 10**

```python
def test_wrappers_from_one_ocr_invocation_count_as_one_consensus_vote():
    observations = [
        _obs("STALE PAGE", provider="paddle_full_page", invocation_id="infer-9"),
        _obs("STALE PAGE", provider="paddle_full_page_raw_line", invocation_id="infer-9"),
        _obs("STALE PAGE", provider="visual_card_full_page_raw", invocation_id="infer-9"),
        _obs("CURRENT ENGLISH BODY", provider="negative_detect_ocr", invocation_id="infer-10-neg"),
    ]
    owner = _resolve(observations)
    assert owner.source_payload == "CURRENT ENGLISH BODY"


def test_previous_page_hash_is_rejected_before_consensus():
    current = _page_context("page_010", sha="sha-10")
    stale = _obs("STALE PAGE", page_id="page_009", page_sha="sha-9")
    with pytest.raises(OwnerEvidenceIdentityError):
        reconcile_page(current, [stale])


def test_previous_run_id_is_rejected_before_consensus():
    current = _page_context("page_010", run_id="run-current", sha="sha-10")
    stale = _obs("STALE RUN", run_id="run-previous", page_id="page_010", page_sha="sha-10")
    with pytest.raises(OwnerEvidenceIdentityError):
        reconcile_page(current, [stale])


def test_new_owner_graph_roundtrip_preserves_ocr_identity_and_hashes():
    graph = _graph_with_new_observation_identity()
    loaded = OwnerGraph.from_dict(graph.to_dict(), enforce=True)
    assert loaded.schema_version == OWNER_GRAPH_SCHEMA_VERSION == 2
    assert loaded.verification_status == "verified"
    assert loaded.run_id == graph.run_id
    assert loaded.page_source_sha256 == graph.page_source_sha256
    assert loaded.observations[0].run_id == graph.observations[0].run_id
    assert loaded.observations[0].invocation_id == graph.observations[0].invocation_id
    assert loaded.observations[0].ocr_attempt_id == graph.observations[0].ocr_attempt_id
    assert loaded.observations[0].page_source_sha256 == graph.observations[0].page_source_sha256
    assert loaded.observations[0].root_input_pixel_sha256 == graph.observations[0].root_input_pixel_sha256
    assert loaded.observations[0].input_pixel_sha256 == graph.observations[0].input_pixel_sha256
    assert loaded.observations[0].payload_sha256 == graph.observations[0].payload_sha256


def test_legacy_graph_without_identity_is_readable_only_as_legacy_unverified():
    legacy = OwnerGraph.from_dict(_legacy_graph_payload_without_invocation_or_page_hash(), enforce=False)
    assert legacy.schema_version == OWNER_GRAPH_LEGACY_SCHEMA_VERSION == 1
    assert legacy.verification_status == "legacy_unverified"
    with pytest.raises(OwnerGraphValidationError):
        OwnerGraph.from_dict(_legacy_graph_payload_without_invocation_or_page_hash(), enforce=True)


def test_current_schema_without_explicit_verification_status_is_not_enforceable():
    payload = _current_graph_payload()
    payload.pop("verification_status")
    with pytest.raises(OwnerGraphValidationError):
        OwnerGraph.from_dict(payload, enforce=True)


def test_serialized_graph_boundary_propagates_enforce_mode():
    legacy = _legacy_graph_payload_without_invocation_or_page_hash()
    assert validate_serialized_owner_graph(legacy, enforce=False).verification_status == "legacy_unverified"
    with pytest.raises(OwnerGraphValidationError):
        validate_serialized_owner_graph(legacy, enforce=True)


def test_verified_graph_rejects_observation_from_other_run_or_source_page_pixels():
    graph = _graph_with_new_observation_identity(run_id="run-a", page_source_sha256="sha-a")
    graph.observations[0] = dataclasses.replace(graph.observations[0], run_id="run-b")
    with pytest.raises(OwnerGraphValidationError):
        graph.validate_or_raise(mode="enforce")
    graph = _graph_with_new_observation_identity(run_id="run-a", page_source_sha256="sha-a")
    graph.observations[0] = dataclasses.replace(graph.observations[0], page_source_sha256="sha-b")
    with pytest.raises(OwnerGraphValidationError):
        graph.validate_or_raise(mode="enforce")


def test_legacy_positional_observation_remains_loadable_but_not_enforceable():
    legacy = TextObservation("obs", "page_001", (), "TEXT", 0.9, "fixture", (0, 0, 20, 10))
    assert not legacy.identity_complete
    with pytest.raises(OwnerGraphValidationError):
        _graph(observations=[legacy]).validate_or_raise(mode="enforce")


def test_new_observation_identity_fields_are_keyword_only_and_enforceable():
    observation = _observation_with_complete_identity()
    assert observation.identity_complete
    assert not _graph(observations=[observation]).validate(mode="enforce")


def test_provider_alias_permutation_never_changes_selected_payload():
    for observations in itertools.permutations(_correlated_observations()):
        assert _resolve(list(observations)).source_payload == "CURRENT ENGLISH BODY"
```

Adicione também colisão: mesmo `observation_id` com texto/hash diferente deve falhar em todos os append/merge paths.

**Step 3: Confirmar RED**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest `
    tests/test_owner_evidence.py `
    tests/test_owner_reconcile.py `
    tests/test_owner_reconcile_properties.py `
    tests/test_owner_graph.py `
    tests/test_project_migration.py `
    tests/test_project_schema_v12.py `
    tests/test_project_writer.py `
    tests/test_strip_owner_control_plane.py `
    -q
} finally { Pop-Location }
```

Expected: o consenso atual conta aliases como providers independentes ou aceita identidade incompleta.

**Step 4: Ampliar `TextObservation` sem quebrar construtores legados**

Campos obrigatórios:

```python
run_id: str
invocation_id: str
ocr_attempt_id: str
provider_family: str
page_source_sha256: str
root_input_pixel_sha256: str
input_pixel_sha256: str
payload_sha256: str
```

Acrescente esses campos **ao final** do dataclass, após um marcador `KW_ONLY`, com defaults vazios somente para desserialização/fixtures legadas. Isso preserva temporariamente os construtores posicionais atuais. `identity_complete` exige os oito campos e o boundary `enforce` rejeita vazio; legacy/shadow pode carregar o registro apenas como `legacy_unverified`.

Todos os aliases derivados do mesmo `OCRInvocationResult` preservam exatamente o `invocation_id` e `provider_family` da `OCRRequest`; observações de cada array físico preservam também o `ocr_attempt_id`, `root_input_pixel_sha256` e `input_pixel_sha256` do `OCRAttempt` real. Tentativas nativa/cinza/invertida/2x da mesma invocação continuam uma única origem de consenso. A identidade da página vem da request, nunca de parsing do filename. O adapter valida igualdade — não cria um segundo ID de inferência.

Defina em `ownership/model.py` a autoridade única `OWNER_GRAPH_SCHEMA_VERSION = 2` e `OWNER_GRAPH_LEGACY_SCHEMA_VERSION = 1`; `ownership/project.py` importa e pode reexportar esses símbolos para compatibilidade, mas não mantém outro literal. Faça o bump explícito do schema do `OwnerGraph`; acrescente ao final, como keywords, `run_id: str = ""`, `page_source_sha256: str = ""` e `verification_status: Literal["verified", "legacy_unverified"] = "verified"`, e atualize `to_dict()/from_dict(*, enforce=False)`. O schema 2 exige os três campos explícitos, identidade completa e igualdade de run/page/source com todas as observações para ser `verified`. Payload v1 é desserializado com identidade vazia e `verification_status="legacy_unverified"` somente em legacy/shadow; não fabrique IDs nem permita entrada em `enforce`. Versões desconhecidas falham em todos os modos. `validate_serialized_owner_graph(..., enforce=...)`, project writer/migration e schema v12 devem propagar o modo e a constante. O adapter de `run_chapter(legacy_project_status=...)` deve mapear para esse campo, sem manter dois status divergentes.

**Step 4a: Migrar todos os construtores diretos no mesmo commit**

```powershell
$observationSites = & rg -n "TextObservation\(" pipeline --glob '*.py'
$observationSiteExit = $LASTEXITCODE
$observationSites
if ($observationSiteExit -gt 1) { throw "TextObservation call-site scan failed: $observationSiteExit" }
```

Migre cada call site de produção e cada fixture que entra em `enforce` para keywords explícitas `run_id`, `invocation_id`, `ocr_attempt_id`, `provider_family`, `page_source_sha256`, `root_input_pixel_sha256`, `input_pixel_sha256`, `payload_sha256`. No snapshot do plano são 30 construções em 20 arquivos; rerode a busca após editar e classifique qualquer constructor ainda vazio como fixture `legacy_unverified` explícita. Não dependa da ordem posicional dos novos campos. Merge, `to_dict()/from_dict()` e todos os boundaries rejeitam `run_id`, root hash, attempt id ou input pixel hash divergente antes do consenso.

Converta cada `OCRObservationRecord` da Task 2 em `TextObservation` preservando `run_id`, `page_id`, `page_source_sha256`, `root_input_pixel_sha256`, `input_pixel_sha256`, `invocation_id`, `ocr_attempt_id` e `provider_family` sem alteração, mapeando apenas `variant_id` para o campo legado já existente `provider_variant`. Em seguida, migre **todas** as construções diretas de `OwnerGraph` encontradas por `rg -n "OwnerGraph\(" pipeline --glob '*.py'`: produção e testes devem usar `schema_version=OWNER_GRAPH_SCHEMA_VERSION`, `run_id` e `page_source_sha256`, nunca o literal `1` nem identidade vazia no modo novo. No snapshot atual são 20 ocorrências em 18 arquivos (o arquivo de produção é `ownership/reconcile.py`; os demais estão explicitados na lista desta Task). Releia também todos os boundaries `OwnerGraph.from_dict`/`validate_serialized_owner_graph`, inclusive `ownership/project.py`, `strip/process_bands.py` e os testes de project migration/writer/schema.

Depois da migração, rode uma checagem AST para impedir literal legado em qualquer construtor `OwnerGraph`:

```powershell
@'
import ast
from pathlib import Path

bad = []
for path in Path("pipeline").rglob("*.py"):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
        if name != "OwnerGraph":
            continue
        keywords = {kw.arg for kw in node.keywords}
        if not {"schema_version", "run_id", "page_source_sha256"} <= keywords:
            bad.append(f"{path}:{node.lineno}: missing canonical graph identity keyword")
        for kw in node.keywords:
            if kw.arg == "schema_version" and isinstance(kw.value, ast.Constant) and kw.value.value == 1:
                bad.append(f"{path}:{node.lineno}: legacy schema literal")
if bad:
    raise SystemExit("legacy OwnerGraph schema literals remain:\n" + "\n".join(bad))
'@ | python -
```

**Step 5: Substituir provider-count por inference-count**

Em `_strong_full_reading_consensus()`:

```python
def independent_origins(group: Sequence[TextObservation]) -> frozenset[str]:
    return frozenset(observation.invocation_id for observation in group)
```

Use `independent_origins` para share, runner-up e corroboration. Provider permanece telemetria, não voto.

**Step 6: Unificar colisões de observação**

Extraia um único `merge_observation_strict(existing, incoming)`. Todo append/merge usa essa função. Ela aceita a mesma observação com provenance adicional, mas rejeita identidade, payload ou page hash incompatível.

**Step 7: Confirmar GREEN e Checkpoint A**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest `
    tests/test_vision_stack_ocr.py `
    tests/test_vision_stack_runtime.py `
    tests/test_owner_evidence.py `
    tests/test_owner_reconcile.py `
    tests/test_owner_reconcile_properties.py `
    tests/test_strip_owner_control_plane.py `
    tests/test_balloon_layout_shared_regions.py `
    tests/test_final_pixel_qa.py `
    tests/test_main_emit.py `
    tests/test_ocr_adapter.py `
    tests/test_owner_atomic_execution.py `
    tests/test_owner_compositor.py `
    tests/test_owner_enforcement.py `
    tests/test_owner_graph.py `
    tests/test_owner_layout.py `
    tests/test_owner_mask.py `
    tests/test_owner_render_geometry.py `
    tests/test_owner_style_capture.py `
    tests/test_owner_style_profile.py `
    tests/test_owner_translation.py `
    tests/test_strip_owner_composition_integration.py `
    tests/test_typesetting_renderer.py `
    tests/test_project_migration.py `
    tests/test_project_schema_v12.py `
    tests/test_project_writer.py `
    -q
} finally { Pop-Location }
```

Expected: PASS; duplicar aliases ou reordenar páginas não altera owner/payload.

**Step 8: Commit**

```powershell
git add -p -- pipeline/ownership/model.py pipeline/ownership/coordinates.py pipeline/ownership/ocr_adapter.py pipeline/ownership/evidence.py pipeline/ownership/reconcile.py pipeline/ownership/project.py pipeline/schema/project_schema_v12.py pipeline/strip/process_bands.py pipeline/strip/run.py pipeline/tests/test_owner_evidence.py pipeline/tests/test_owner_reconcile.py pipeline/tests/test_owner_reconcile_properties.py pipeline/tests/test_strip_owner_control_plane.py pipeline/tests/test_balloon_layout_shared_regions.py pipeline/tests/test_final_pixel_qa.py pipeline/tests/test_main_emit.py pipeline/tests/test_ocr_adapter.py pipeline/tests/test_owner_atomic_execution.py pipeline/tests/test_owner_compositor.py pipeline/tests/test_owner_enforcement.py pipeline/tests/test_owner_graph.py pipeline/tests/test_owner_layout.py pipeline/tests/test_owner_mask.py pipeline/tests/test_owner_render_geometry.py pipeline/tests/test_owner_style_capture.py pipeline/tests/test_owner_style_profile.py pipeline/tests/test_owner_translation.py pipeline/tests/test_strip_owner_composition_integration.py pipeline/tests/test_typesetting_renderer.py pipeline/tests/test_project_migration.py pipeline/tests/test_project_schema_v12.py pipeline/tests/test_project_writer.py
git diff --cached --check
git commit -m "fix: count independent OCR evidence origins"
```

### Task 4: Criar o ledger de cobertura e lifecycle terminal

**Files:**

- Create: `pipeline/ownership/coverage.py`
- Create: `pipeline/ownership/lifecycle.py`
- Modify: `pipeline/ownership/model.py`
- Modify: `pipeline/ownership/__init__.py`
- Create: `pipeline/tests/test_owner_coverage_rescue.py`
- Create: `pipeline/tests/test_owner_lifecycle.py`
- Test: `pipeline/tests/test_owner_graph.py`

**Step 1: Escrever testes RED dos invariantes**

```python
def test_every_material_component_has_exactly_one_terminal_lifecycle():
    ledger = _ledger_with_verified_dialogue_and_preserved_credit()
    ledger.require_complete()


def test_dialogue_component_cannot_finish_suppressed_or_review_required():
    ledger = _dialogue_ledger(state="suppress")
    with pytest.raises(CoverageInvariantError):
        ledger.require_complete()


def test_component_without_ocr_attempt_cannot_be_classified_non_text():
    entry = _entry(state="explicit_non_text", ocr_attempt_ids=())
    with pytest.raises(CoverageInvariantError):
        _ledger(entries=(entry,)).require_complete()


def test_ledger_rejects_expected_component_missing_from_entries():
    ledger = _ledger(
        expected_component_ids=("component-a", "component-b"),
        entries=(_entry(component_id="component-a"),),
    )
    with pytest.raises(CoverageInvariantError, match="component-b"):
        ledger.require_complete()


def test_owner_identity_survives_every_lifecycle_transition():
    lifecycle = _owner_lifecycle()
    final = lifecycle.advance("final_verified", evidence=_verified_evidence())
    assert final.identity == lifecycle.identity


def test_owner_graph_accepts_canonical_repair_states_but_enforce_rejects_review_terminal():
    for state in ("target_ready", "execution_attempt", "repair_pending", "cleaned", "final_verified"):
        assert not _graph_with_owner_state(state).validate()
    assert "review_terminal_forbidden_in_enforce" in _violation_codes(
        _graph_with_owner_state("review_required", mode="enforce").validate()
    )
```

**Step 2: Confirmar RED**

```powershell
Push-Location pipeline
try { .\venv\Scripts\python.exe -m pytest tests/test_owner_coverage_rescue.py tests/test_owner_lifecycle.py tests/test_owner_graph.py::test_owner_graph_accepts_canonical_repair_states_but_enforce_rejects_review_terminal -q } finally { Pop-Location }
```

Expected: FAIL porque os módulos ainda não existem.

**Step 3: Implementar dataclasses imutáveis**

Em `coverage.py`:

```python
@dataclass(frozen=True)
class CoverageEntry:
    component_id: str
    page_id: str
    page_source_sha256: str
    bbox_page: BBox
    polygon_page: Polygon
    materiality: Literal["material", "non_text"]
    container_id: str | None = None
    ocr_attempt_ids: tuple[str, ...] = ()
    observation_ids: tuple[str, ...] = ()
    owner_id: str | None = None
    semantic_role: str | None = None
    protection_conflict: bool = False
    protection_evidence_ids: tuple[str, ...] = ()
    state: CoverageState = "discovered"


@dataclass(frozen=True)
class PageCoverageLedger:
    run_id: str
    page_id: str
    page_source_sha256: str
    expected_component_ids: tuple[str, ...]
    expected_observation_ids: tuple[str, ...]
    entries: tuple[CoverageEntry, ...]
```

Em `lifecycle.py`, codifique transições permitidas e proíba alterar identidade/hashes. Estados de reparo não são terminais. Migre `OWNER_STATES`, `POST_TRANSLATION_STATES`, `MASK_REQUIRED_STATES`, `EXECUTOR_REQUIRED_STATES` e validações relacionadas em `model.py` para aceitar os estados canônicos (`challenged`, `observed`, `owned`, `target_ready`, `execution_attempt`, `repair_pending`, `cleaned`, `rendered`, `final_verified`, `explicit_non_dialogue_preserve`). Preserve aliases antigos somente para desserializar `legacy/shadow`; `enforce` não pode terminalizar em `review_required`.

**Step 4: Implementar validação de completude**

`require_complete()` exige:

- exatamente um entry por componente;
- o conjunto de `entries.component_id` é exatamente o inventário imutável `expected_component_ids` capturado na descoberta, portanto remover componente também falha;
- toda observação esperada após coverage aparece exatamente uma vez nos `observation_ids` dos entries ou em disposição explícita auditável;
- todo material possui tentativa OCR;
- todo diálogo possui owner;
- diálogo termina `final_verified`;
- preserve exige policy não-diálogo explícita;
- nenhuma identidade cruza página/hash.

**Step 5: Confirmar GREEN**

```powershell
Push-Location pipeline
try { .\venv\Scripts\python.exe -m pytest tests/test_owner_coverage_rescue.py tests/test_owner_lifecycle.py tests/test_owner_graph.py -q } finally { Pop-Location }
```

Expected: PASS.

**Step 6: Commit**

```powershell
git add -- pipeline/ownership/coverage.py pipeline/ownership/lifecycle.py pipeline/tests/test_owner_coverage_rescue.py pipeline/tests/test_owner_lifecycle.py
git add -p -- pipeline/ownership/model.py pipeline/ownership/__init__.py pipeline/tests/test_owner_graph.py
git diff --cached --check
git commit -m "feat: add page coverage and owner lifecycle ledger"
```

### Task 5: Executar OCR de cobertura em toda página, mesmo sem bands

**Files:**

- Modify: `pipeline/vision_stack/ocr.py:850-930,1735-2020`
- Modify: `pipeline/vision_stack/runtime.py:15067-15610`
- Modify: `pipeline/ownership/coverage.py`
- Modify: `pipeline/ownership/discovery.py:774-1156`
- Modify: `pipeline/strip/run.py:2643-2713,6077-6237`
- Test: `pipeline/tests/test_vision_stack_ocr.py`
- Test: `pipeline/tests/test_source_component_discovery.py`
- Test: `pipeline/tests/test_owner_coverage_rescue.py`
- Test: `pipeline/tests/test_strip_owner_control_plane.py`

**Step 1: Escrever testes RED de páginas sem balloon/band**

```python
def test_full_page_ocr_runs_with_empty_detector_blocks(self):
    engine = _fake_paddle_full_page(lines=[_line("VISIBLE ENGLISH", bbox=(10, 10, 90, 30))])
    result = engine.recognize_page_with_evidence(
        _page_with_text(),
        [],
        request=_request("page_001"),
        force_full_page=True,
    )
    assert _texts(result.full_page_lines) == ["VISIBLE ENGLISH"]


def test_full_page_attempt_hashes_exact_array_passed_to_provider(self):
    provider = _capturing_provider()
    page = _asymmetric_rgb_page()
    result = _run_full_page_ocr(page, provider=provider, force_downscale=True)
    attempt = _only_provider_attempt(result)
    assert attempt.root_input_pixel_sha256 == canonical_page_sha256(page)
    assert attempt.input_pixel_sha256 == canonical_page_sha256(provider.received_rgb[0])
    assert attempt.input_pixel_sha256 != attempt.root_input_pixel_sha256
    assert attempt.input_width == provider.received_rgb[0].shape[1]
    assert attempt.input_height == provider.received_rgb[0].shape[0]


def test_anchored_crop_attempt_hashes_crop_pixels_not_page_pixels(self):
    page = _asymmetric_rgb_page()
    result, captured_crop = _run_anchored_crop_with_capture(page, bbox=(7, 11, 83, 41))
    attempt = _attempt(result, variant_id="anchored_crop")
    assert attempt.root_input_pixel_sha256 == canonical_page_sha256(page)
    assert attempt.input_pixel_sha256 == canonical_page_sha256(captured_crop)
    assert attempt.input_pixel_sha256 != attempt.root_input_pixel_sha256
    assert attempt.input_bbox_page == (7, 11, 83, 41)


def test_each_retry_variant_hashes_exact_provider_pixels(self):
    for variant in ("native", "gray", "inverted", "scale_2x"):
        with self.subTest(variant=variant):
            result, captured_rgb = _force_variant_and_capture(_asymmetric_rgb_page(), variant)
            attempt = _attempt(result, variant_id=variant)
            assert attempt.input_pixel_sha256 == canonical_page_sha256(captured_rgb)
            assert (attempt.input_width, attempt.input_height) == (captured_rgb.shape[1], captured_rgb.shape[0])
            assert attempt.transform_spec.sha256 == sha256_bytes(attempt.transform_spec.canonical_json_bytes)
            assert np.array_equal(attempt.transform_spec.replay(_asymmetric_rgb_page()), captured_rgb)


def test_component_without_band_gets_anchored_ocr_before_owner_resolution():
    page = _page_with_glyph_component_and_no_balloon()
    ledger, observations = complete_page_coverage(page, components=page.components, band_evidence=[])
    assert ledger.entries[0].ocr_attempt_ids
    assert observations[0].component_ids == (page.components[0].component_id,)


def test_source_discovery_keeps_material_glyph_component_outside_all_bands():
    components = discover_source_components(_page_with_glyphs_outside_all_bands(), bands=[])
    assert len(components) == 1
    assert components[0].materiality == "material"


def test_page_without_bands_never_builds_empty_observation_placeholder():
    result = _run_owner_control_plane_for_no_band_page()
    assert result.graph.observations
    assert result.graph.owners
```

**Step 2: Confirmar RED**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest `
    tests/test_vision_stack_ocr.py::PaddleBlockMappingTests::test_full_page_ocr_runs_with_empty_detector_blocks `
    tests/test_vision_stack_ocr.py::PaddleBlockMappingTests::test_full_page_attempt_hashes_exact_array_passed_to_provider `
    tests/test_vision_stack_ocr.py::PaddleBlockMappingTests::test_anchored_crop_attempt_hashes_crop_pixels_not_page_pixels `
    tests/test_vision_stack_ocr.py::PaddleBlockMappingTests::test_each_retry_variant_hashes_exact_provider_pixels `
    tests/test_source_component_discovery.py::test_source_discovery_keeps_material_glyph_component_outside_all_bands `
    tests/test_owner_coverage_rescue.py::test_component_without_band_gets_anchored_ocr_before_owner_resolution `
    tests/test_strip_owner_control_plane.py::test_page_without_bands_never_builds_empty_observation_placeholder `
    -q
} finally { Pop-Location }
```

Expected: early return com blocks vazios ou placeholder `observations=[]`.

**Step 3: Implementar `complete_page_coverage()`**

A função recebe a página original, componentes descobertos e evidência opcional de bands. Ela:

1. executa uma inferência full-page independente pelo boundary hash-bound da Task 2;
2. anexa observações por componente;
3. executa crops ancorados para componentes materiais ainda sem observação, preservando root hash, bbox e hash do crop físico;
4. usa variantes nativa, cinza, invertida e 2x somente quando necessário; cada ndarray efetivamente recebido pelo provider ganha `OCRAttempt` próprio, com parent/transform/hash/shape verificados;
5. registra toda tentativa no ledger; cache/heurística sem provider fica explícito e não conta como inferência física fresca;
6. devolve ledger e observações, sem mutar pixels.

**Step 4: Integrar antes do owner graph**

Em `run_chapter()`, faça coverage por página imediatamente após `_discover_source_components_for_strip()`. Bands podem acrescentar observações depois, mas não controlam a existência da página no OCR.

Remova o placeholder vazio em `_resolve_owner_page`. O resolver recebe sempre um `PageCoverageLedger` e a coleção page-global, mesmo com zero bands.

**Step 5: Confirmar GREEN**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest `
    tests/test_vision_stack_ocr.py `
    tests/test_source_component_discovery.py `
    tests/test_owner_coverage_rescue.py `
    tests/test_strip_owner_control_plane.py `
    -q
} finally { Pop-Location }
```

Expected: PASS; todo componente material possui tentativa OCR independentemente de bands.

**Step 6: Commit**

```powershell
git add -p -- pipeline/vision_stack/ocr.py pipeline/vision_stack/runtime.py pipeline/ownership/coverage.py pipeline/ownership/discovery.py pipeline/strip/run.py pipeline/tests/test_vision_stack_ocr.py pipeline/tests/test_source_component_discovery.py pipeline/tests/test_owner_coverage_rescue.py pipeline/tests/test_strip_owner_control_plane.py
git diff --cached --check
git commit -m "feat: complete OCR coverage before band execution"
```

### Task 6: Recuperar observações sem componente e containers antes do owner graph

**Files:**

- Modify: `pipeline/ownership/coverage.py`
- Modify: `pipeline/ownership/container_evidence.py`
- Modify: `pipeline/ownership/discovery.py`
- Modify: `pipeline/strip/run.py`
- Test: `pipeline/tests/test_owner_coverage_rescue.py`
- Test: `pipeline/tests/test_owner_container_evidence.py`
- Test: `pipeline/tests/test_strip_owner_control_plane.py`

**Step 1: Escrever testes RED de associação e container**

```python
def test_unique_full_page_observation_without_component_materializes_component():
    page = _page_with_visible_text()
    observation = _observation("READ ME", bbox=(40, 50, 160, 90), component_ids=())
    result = recover_unassociated_observations(page, components=(), observations=(observation,))
    assert len(result.components) == 1
    assert result.observations[0].component_ids == (result.components[0].component_id,)
    assert result.decisions[0].reason == "materialized_from_full_page_observation"


def test_ambiguous_full_page_observation_requests_anchored_recovery():
    result = recover_unassociated_observations(
        _ambiguous_page(),
        components=_two_close_components(),
        observations=(_observation("WAIT", component_ids=()),),
    )
    assert result.pending_recovery
    assert not result.suppressed_observation_ids


def test_ocr_confirmed_component_recovers_container_before_graph_build():
    recovered = complete_container_coverage(
        _page(),
        _ledger_with_observed_component_without_container(),
    )
    assert recovered.entries[0].container_id
    assert recovered.entries[0].state == "observed"


def test_unassociated_glyph_support_over_conservative_protection_still_materializes_component():
    result = recover_unassociated_observations(
        _page_with_conservative_protection_overlap(),
        components=(),
        observations=(_observation("VISIBLE ENGLISH", component_ids=(), glyph_support=True),),
    )
    assert len(result.components) == 1
    entry = result.ledger.entry(result.components[0].component_id)
    assert entry.container_id
    assert entry.protection_conflict
    assert entry.state == "observed"
    assert not result.pending_recovery
```

**Step 2: Confirmar RED**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest `
    tests/test_owner_coverage_rescue.py `
    tests/test_owner_container_evidence.py `
    tests/test_strip_owner_control_plane.py `
    -q
} finally { Pop-Location }
```

Expected: observação continua `unassociated_observation` ou componente confirmado continua `semantic_container_missing`.

**Step 3: Implementar recuperação determinística de associação**

Adicionar `recover_unassociated_observations()` com esta ordem, sempre em page-space:

1. interseção única forte com componente existente;
2. associação por distância/overlap dentro de container único;
3. crop OCR ancorado quando houver ambiguidade entre componentes existentes;
4. após orçamento finito de associação, materialização obrigatória de novo componente quando bbox/polígono OCR possui suporte de glyph positivo, mesmo se sobrepuser proteção conservadora;
5. marcar `protection_conflict=true` e guardar os IDs de evidência quando houver essa sobreposição, para que execution/R3 trate os pixels confirmados como texto sem desproteger arte externa;
6. `coverage_recovery_pending` existe apenas entre tentativas; não é terminal. Se glyph support material persistir sem associação, o passo 4 fecha coverage de forma monotônica.

Não descartar observação. IDs do componente materializado devem derivar de `page_source_sha256 + polygon_page`, nunca do texto reconhecido.

**Step 4: Antecipar recuperação de container**

Extrair o núcleo reutilizável de `recover_full_page_visual_container()` e chamar `complete_container_coverage()` antes de `build_owner_page_graph()`. Ordem:

1. container primário detectado;
2. scan claro/escuro;
3. scan de card/UI;
4. busca local ancorada no suporte de glyph;
5. container conservador derivado da região textual, marcado para R2/R3 caso nenhuma borda seja segura.

Um container conservador é executável e auditável; não deve virar `review_required` por defeito de conteúdo. Se nenhuma borda visual for recuperável, derive um container final support-local do union bbox/polígono de glyph confirmado + margem determinística, hard-clipped aos bounds válidos da página. A colisão com proteção conservadora viaja no ledger; ela nunca impede owner/binding nem deixa o fluxo preso antes do R3.

**Step 5: Confirmar GREEN**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest `
    tests/test_owner_coverage_rescue.py `
    tests/test_owner_container_evidence.py `
    tests/test_source_component_discovery.py `
    tests/test_strip_owner_control_plane.py `
    -q
} finally { Pop-Location }
```

Expected: PASS; toda observação material possui componente e todo componente textual possui container executável antes do graph.

**Step 6: Commit**

```powershell
git add -p -- pipeline/ownership/coverage.py pipeline/ownership/container_evidence.py pipeline/ownership/discovery.py pipeline/strip/run.py pipeline/tests/test_owner_coverage_rescue.py pipeline/tests/test_owner_container_evidence.py pipeline/tests/test_strip_owner_control_plane.py
git diff --cached --check
git commit -m "feat: recover page observations and containers before ownership"
```

### Task 7: Construir o owner graph exclusivamente do ledger completo

**Files:**

- Modify: `pipeline/ownership/reconcile.py`
- Modify: `pipeline/ownership/model.py`
- Modify: `pipeline/ownership/coverage.py`
- Modify: `pipeline/strip/run.py`
- Modify: `pipeline/strip/process_bands.py`
- Test: `pipeline/tests/test_owner_reconcile.py`
- Test: `pipeline/tests/test_owner_reconcile_properties.py`
- Test: `pipeline/tests/test_owner_coverage_rescue.py`
- Test: `pipeline/tests/test_strip_owner_control_plane.py`

**Step 1: Escrever testes RED de disposição total**

```python
def test_zero_ocr_attempts_can_never_produce_no_ocr_non_text_suppression():
    ledger = _ledger(component=_material_component(), ocr_attempt_ids=())
    with pytest.raises(CoverageInvariantError, match="ocr attempt"):
        build_owner_page_graph_from_ledger(ledger, observations=())


def test_negative_ocr_can_preserve_only_after_explicit_non_text_evidence():
    ledger = _ledger_with_negative_ocr_and_visual_non_text_evidence()
    graph = build_owner_page_graph_from_ledger(ledger, observations=())
    disposition = graph.component_dispositions[0]
    assert disposition.decision == "preserve"
    assert disposition.policy_id == "explicit_visual_non_text"


def test_fragments_in_one_balloon_produce_one_owner_with_one_complete_body():
    graph = build_owner_page_graph_from_ledger(_fragmented_same_balloon_ledger(), _observations())
    assert len(graph.owners) == 1
    assert graph.owners[0].component_ids == _all_fragment_component_ids()
    assert graph.owners[0].source_payload == _complete_source_body()


def test_adjacent_containers_remain_separate_owners():
    graph = build_owner_page_graph_from_ledger(_two_adjacent_balloon_ledger(), _observations())
    assert len(graph.owners) == 2
    assert all(owner.source_payload == _expected_body(owner.owner_id) for owner in graph.owners)


def test_short_english_word_inside_balloon_defaults_to_dialogue_not_sfx():
    graph = build_owner_page_graph_from_ledger(_balloon_with_short_word("WAIT"), _observations())
    assert graph.owners[0].semantic_role == "dialogue"
    assert graph.component_dispositions[0].decision == "owned"


def test_credit_url_or_mark_preserve_requires_outside_dialogue_and_auditable_policy():
    with pytest.raises(CoverageInvariantError):
        build_owner_page_graph_from_ledger(_credit_policy_inside_dialogue(), _observations())
    disposition = build_owner_page_graph_from_ledger(
        _credit_outside_dialogue_with_policy_bbox_reason_and_evidence(),
        _observations(),
    ).component_dispositions[0]
    assert disposition.policy_id == "explicit_credit_outside_translatable_container"
    assert disposition.policy_bbox_page
    assert disposition.policy_evidence_ids


def test_component_disposition_policy_roundtrip_preserves_audit_fields():
    graph = build_owner_page_graph_from_ledger(
        _credit_outside_dialogue_with_policy_bbox_reason_and_evidence(),
        _observations(),
    )
    loaded = OwnerGraph.from_dict(graph.to_dict(), enforce=True)
    assert loaded.component_dispositions[0] == graph.component_dispositions[0]


@pytest.mark.parametrize(("ledger", "observations"), _deterministic_complete_ledger_cases())
def test_graph_never_loses_a_material_component(ledger, observations):
    graph = build_owner_page_graph_from_ledger(ledger, observations)
    disposed = {item.component_id for item in graph.component_dispositions}
    assert disposed == set(ledger.expected_component_ids)
```

**Step 2: Confirmar RED**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest `
    tests/test_owner_reconcile.py `
    tests/test_owner_reconcile_properties.py `
    tests/test_owner_coverage_rescue.py `
    tests/test_strip_owner_control_plane.py `
    -q
} finally { Pop-Location }
```

Expected: o ramo `no_ocr_evidence_non_text` ainda aceita ausência de tentativa e o resolver ainda aceita coleção vazia criada por band.

**Step 3: Criar a única entrada de reconciliação de produção**

Adicionar `build_owner_page_graph_from_ledger(ledger, observations)`. Ela deve:

- validar `ledger.require_ready_for_ownership()`;
- consumir somente observações com a mesma identidade de página/hash;
- produzir exatamente uma disposição por componente;
- agrupar fragmentos/linhas do mesmo corpo semântico e criar exatamente um owner/payload por balão, narração ou corpo de card traduzível; um owner pode possuir vários `component_ids`;
- ordenar fragmentos em page-space e consolidar o corpo completo antes de OCR voting/tradução/typeset; nunca traduzir ou renderizar uma linha isolada quando ela pertence ao mesmo container;
- permitir preserve somente por policy explícita de SFX, crédito, URL, marca ou não-texto;
- devolver itens não resolvidos como pedidos de coverage recovery, nunca como terminal `review`.

Estenda `ComponentDisposition` sem criar um modelo paralelo: `policy_id`, `policy_bbox_page`, `policy_evidence_ids` e `policy_reason` opcionais. SFX exige evidência semântica/geométrica; palavra curta dentro de balão continua diálogo. Crédito, URL e marca só podem ser preservados fora de container traduzível e com bbox, motivo e evidências preenchidos.

**Step 4: Remover os bypasses de produção**

No caminho `enforce`:

- eliminar o placeholder `observations=[]` em torno de `run.py:6221`;
- impedir o ramo de supressão sem tentativa em `reconcile.py:1071`;
- transformar `unassociated_observation` e `semantic_container_missing` em retorno ao coverage controller;
- impedir que `process_bands.py` construa um segundo owner graph.

Manter helpers legados somente para `shadow`/fixtures antigas; marque-os com comentário de depreciação e teste que `enforce` não os chama.

**Step 5: Confirmar GREEN**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest `
    tests/test_owner_reconcile.py `
    tests/test_owner_reconcile_properties.py `
    tests/test_owner_evidence.py `
    tests/test_owner_coverage_rescue.py `
    tests/test_strip_owner_control_plane.py `
    -q
} finally { Pop-Location }
```

Expected: PASS; `enforce` só reconcilia ledger completo e não existe sucesso silencioso sem owner.

**Step 6: Checkpoint B — cobertura e ownership**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest `
    tests/test_vision_stack_ocr.py `
    tests/test_vision_stack_runtime.py `
    tests/test_source_component_discovery.py `
    tests/test_owner_evidence.py `
    tests/test_owner_reconcile.py `
    tests/test_owner_reconcile_properties.py `
    tests/test_owner_coverage_rescue.py `
    tests/test_owner_lifecycle.py `
    tests/test_strip_owner_control_plane.py `
    -q
} finally { Pop-Location }
```

Aceite somente se todos passarem e uma fixture sem bands produzir `translatable_components_without_owner == 0`.

**Step 7: Commit**

```powershell
git add -p -- pipeline/ownership/reconcile.py pipeline/ownership/model.py pipeline/ownership/coverage.py pipeline/strip/run.py pipeline/strip/process_bands.py pipeline/tests/test_owner_reconcile.py pipeline/tests/test_owner_reconcile_properties.py pipeline/tests/test_owner_coverage_rescue.py pipeline/tests/test_strip_owner_control_plane.py
git diff --cached --check
git commit -m "fix: resolve ownership from complete page coverage"
```

### Task 8: Validar PT-BR e vincular tradução ao mesmo owner

**Files:**

- Create: `pipeline/translator/language_policy.py`
- Modify: `pipeline/translator/locale_policy.py`
- Modify: `pipeline/translator/translate.py`
- Modify: `pipeline/ownership/translation.py`
- Modify: `pipeline/ownership/lifecycle.py`
- Create: `pipeline/tests/test_translation_language_policy.py`
- Test: `pipeline/tests/test_translation_locale_policy.py`
- Test: `pipeline/tests/test_owner_translation.py`

**Step 1: Escrever testes RED de idioma e binding**

```python
def test_unchanged_english_dialogue_is_not_valid_pt_br():
    verdict = validate_target_language(
        source="LET'S GO RIGHT AWAY!",
        target="LET'S GO RIGHT AWAY!",
        role="dialogue",
        explicit_entities=(),
    )
    assert not verdict.accepted
    assert verdict.reason == "unchanged_source_dialogue"


def test_mostly_english_target_requests_next_backend():
    verdict = validate_target_language(
        source="THE ARENA WILL BEGIN",
        target="THE ARENA WILL BEGIN assim que todos entrarem",
        role="dialogue",
    )
    assert verdict.retryable


def test_pt_br_with_explicit_proper_name_and_equivalent_number_is_valid():
    verdict = validate_target_language(
        source="KIM SIMUN HAS 10 KILLS",
        target="KIM SIMUN TEM 10 ABATES",
        role="card",
        explicit_entities=("KIM SIMUN",),
    )
    assert verdict.accepted


@pytest.mark.parametrize(
    ("source", "target", "reason"),
    [
        ("YOU HAVE 10 KILLS", "VOCÊ TEM 100 ABATES", "numeric_mismatch"),
        ("PLAYER {name} HAS {count}", "O JOGADOR {name} TEM", "placeholder_mismatch"),
        ("THE ARENA WILL BEGIN", "", "empty_target"),
    ],
)
def test_semantically_incomplete_targets_are_rejected_before_binding(source, target, reason):
    verdict = validate_target_language(source=source, target=target, role="dialogue")
    assert not verdict.accepted
    assert verdict.reason == reason


def test_translation_binding_preserves_owner_and_hash_chain():
    binding = bind_translation(_owner_request(), _valid_response())
    assert binding.owner_id == _owner_request().owner_id
    assert binding.source_payload_sha256 == _owner_request().source_payload_sha256
    assert binding.target_payload_sha256 == sha256_text(binding.target_text)
    assert binding.target_locale == "pt-BR"
    assert binding.translation_binding_sha256 == canonical_json_sha256(binding.canonical_payload())


@pytest.mark.parametrize("route_action", sorted(TRANSLATION_ROUTE_ACTIONS))
def test_every_declared_translation_route_receives_exactly_one_binding(route_action):
    owner = _owner_request(route_action=route_action)
    bindings = translate_owner_page((_owner(owner),), _valid_backend())
    assert [binding.owner_id for binding in bindings] == [owner.owner_id]


def test_invalid_english_targets_advance_backends_until_valid_local_ptbr_binding():
    binding, attempts = translate_owner(
        _owner_request(),
        backends=_backends_returning_unchanged_then_mixed_then_valid_ptbr(),
    )
    assert [attempt.language_verdict.accepted for attempt in attempts] == [False, False, True]
    assert binding.target_locale == "pt-BR"
    assert binding.language_verdict.accepted


def test_operational_backend_that_always_returns_invalid_target_exhausts_budget_without_output():
    with pytest.raises(TranslationValidationExhausted):
        translate_owner(_owner_request(), backends=(_always_english_backend(),), max_attempts_per_backend=2)
    assert not _owner_execution_was_called()


def test_already_ptbr_container_is_preserved_only_when_no_source_english_exists():
    verdict = validate_target_language(
        source="VAMOS ENTRAR AGORA!",
        target="VAMOS ENTRAR AGORA!",
        role="dialogue",
        page_language_evidence=_fresh_full_page_evidence(
            coverage_complete=True,
            source_only_tokens=(),
        ),
    )
    assert verdict.accepted
    assert verdict.policy_id == "already_target_language"


def test_already_ptbr_preserve_is_rejected_without_fresh_complete_page_evidence():
    verdict = validate_target_language(
        source="VAMOS ENTRAR AGORA!",
        target="VAMOS ENTRAR AGORA!",
        role="dialogue",
        page_language_evidence=None,
    )
    assert not verdict.accepted
```

**Step 2: Confirmar RED**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest `
    tests/test_translation_language_policy.py `
    tests/test_translation_locale_policy.py `
    tests/test_owner_translation.py `
    -q
} finally { Pop-Location }
```

Expected: frase inglesa inalterada é aceita pela validação atual ou o novo módulo não existe.

**Step 3: Implementar `TargetLanguageVerdict`**

Combinar sinais locais auditáveis, sem API remota de imagem:

- razão de tokens exclusivos do source ainda presentes;
- léxico funcional EN/PT-BR;
- stopwords e padrões morfológicos;
- detector existente de locale;
- igualdade normalizada source/target;
- entidades e números explicitamente vinculados;
- role semântico.

Não rejeitar automaticamente nomes ou termos compartilhados. Dialogue/card/narration inglês ou majoritariamente inglês é inválido. SFX/crédito só pode permanecer com `policy_id` explícita. Um container integralmente PT-BR pode usar `already_target_language` sem repaint, mas somente após o OCR page-global provar ausência de tokens-fonte ingleses materiais naquela região.

**Step 4: Implementar retries de tradução limitados**

Para resposta inválida:

1. retry do backend primário com prompt de locale reforçado;
2. retry contextual usando o contexto da página/obra;
3. backend local configurado;
4. se respostas semanticamente inválidas persistirem, usar o fallback local final com prompt/contexto reforçado dentro de orçamento determinístico por backend/variante;
5. se o orçamento terminar com respostas inválidas, lançar `TranslationValidationExhausted`; se todos os backends estiverem indisponíveis, lançar `TranslationInfrastructureError`; ambos abortam sem export e nunca renderizam o source como target.

Cada tentativa recebe ID, backend e verdict. Não reutilizar texto de outro owner nem fallback posicional.

Corrija `_translation_owners()` para selecionar por `TRANSLATION_ROUTE_ACTIONS` importado do modelo, em vez de comparar somente `translate_inpaint_render`. `translate_sfx_inpaint_render` e `translate_render_only` também exigem exatamente um binding; qualquer nova route adicionada à constante entra automaticamente no teste parametrizado.

**Step 5: Introduzir `TranslationBinding` imutável**

Campos mínimos:

```python
@dataclass(frozen=True)
class TranslationBinding:
    run_id: str
    page_id: str
    page_source_sha256: str
    owner_id: str
    component_ids: tuple[str, ...]
    source_payload_sha256: str
    target_payload_sha256: str
    source_text: str
    target_text: str
    target_locale: Literal["pt-BR"]
    language_verdict: TargetLanguageVerdict
    attempt_ids: tuple[str, ...]
    translation_binding_sha256: str
```

`translation_binding_sha256` é calculado do JSON canônico de todos os demais campos, inclusive source/target hashes, locale, verdict e attempts; nunca é fornecido pelo backend. Cardinalidade: exatamente um binding aceito por owner traduzível. `target_locale` deve ser exatamente `pt-BR` e o teste de binding deve assertá-lo; português genérico ou locale ausente não fecha o contrato. Resposta desconhecida, duplicada ou sem owner continua erro atômico.

**Step 6: Confirmar GREEN**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest `
    tests/test_translation_language_policy.py `
    tests/test_translation_locale_policy.py `
    tests/test_owner_translation.py `
    tests/test_normalized_text_propagates_to_translation.py `
    -q
} finally { Pop-Location }
```

Expected: PASS; diálogo inglês nunca cria binding aceito nem avança o lifecycle de `owned` para `target_ready`.

**Step 7: Checkpoint C — cobertura + PT-BR**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest `
    tests/test_owner_coverage_rescue.py `
    tests/test_owner_lifecycle.py `
    tests/test_owner_translation.py `
    tests/test_translation_language_policy.py `
    tests/test_translation_locale_policy.py `
    -q
} finally { Pop-Location }
```

Aceite somente se todo owner traduzível possuir `TranslationBinding` PT-BR válido e nenhuma exceção implícita aceitar inglês.

**Step 8: Commit**

```powershell
git add -- pipeline/translator/language_policy.py pipeline/tests/test_translation_language_policy.py
git add -p -- pipeline/translator/locale_policy.py pipeline/translator/translate.py pipeline/ownership/translation.py pipeline/ownership/lifecycle.py pipeline/tests/test_translation_locale_policy.py pipeline/tests/test_owner_translation.py
git diff --cached --check
git commit -m "feat: bind validated pt-BR translations to owners"
```

### Task 9: Criar o coordenador page-first canônico

**Files:**

- Create: `pipeline/strip/page_pipeline.py`
- Create: `pipeline/tests/test_owner_lifecycle_e2e.py`
- Modify: `pipeline/strip/types.py`
- Modify: `pipeline/strip/run.py`
- Modify: `pipeline/strip/process_bands.py`
- Modify: `pipeline/main.py`
- Modify: `pipeline/ownership/lifecycle.py`
- Modify: `pipeline/ownership/translation.py`
- Create: `pipeline/tests/test_page_owner_pipeline.py`
- Test: `pipeline/tests/test_strip_owner_control_plane.py`
- Test: `pipeline/tests/test_strip_owner_composition_integration.py`
- Test: `pipeline/tests/test_main_emit.py`
- Test: `pipeline/tests/test_project_writer.py`

**Step 1: Escrever testes RED do fluxo page-first**

```python
def test_enforce_mode_processes_page_without_creating_band_placeholder():
    result = run_page_owner_pipeline(_page_with_text_and_no_bands(), _services())
    assert result.page_id == "page_001"
    assert result.coverage.entries
    assert result.owner_graph.read().owners


def test_no_band_dialogue_recipe_completes_basic_page_first_lifecycle():
    result = run_fixture_case("no_band_dialogue")
    assert result.coverage.entries
    assert result.owner_graph.read().owners[0].source_payload
    assert result.translations[0].target_locale == "pt-BR"
    assert result.status == "candidate_ready"
    assert result.final_page is None


def test_band_is_only_crop_provenance_not_owner_authority():
    page = _same_page_with_different_band_partitioning()
    a = run_page_owner_pipeline(page[0], _services())
    b = run_page_owner_pipeline(page[1], _services())
    assert _semantic_result(a) == _semantic_result(b)


def test_stale_mutable_band_payloads_are_not_imported_or_mutated_in_enforce():
    bands = _bands_with_stale_ocr_cleaned_and_rendered_slices()
    before = copy.deepcopy(bands)
    request = PagePipelineRequest.from_legacy_bands(_page_snapshot(), bands)
    result = run_page_owner_pipeline(request, _services())
    assert _semantic_result(result) == _expected_from_page_evidence_only()
    assert bands == before


def test_page_execution_result_carries_direct_page_commits():
    result = run_page_owner_pipeline(_page(), _services())
    assert result.page_commits
    assert all(commit.page_id == result.page_id for commit in result.page_commits)
    assert not hasattr(result, "commits_by_band")


def test_page_result_rejects_commit_or_binding_from_another_run_page_or_source_hash():
    with pytest.raises(PagePipelineIdentityError):
        PageExecutionResult.build(
            request=_page_request(run_id="run-a", page_id="page_001", source_sha="sha-a"),
            translations=(_binding(run_id="run-b"),),
            page_commits=(_commit(page_id="page_002"),),
        )


def test_page_result_rejects_owner_graph_from_another_run_or_source_hash():
    request = _page_request(run_id="run-a", page_id="page_001", source_sha="sha-a")
    with pytest.raises(PagePipelineIdentityError):
        PageExecutionResult.build(
            request=request,
            owner_graph=_owner_graph(run_id="run-b", page_source_sha256="sha-a"),
            **_page_result_parts_without_graph(),
        )
    with pytest.raises(PagePipelineIdentityError):
        PageExecutionResult.build(
            request=request,
            owner_graph=_owner_graph(run_id="run-a", page_source_sha256="sha-other"),
            **_page_result_parts_without_graph(),
        )


def test_final_verified_status_requires_hash_linked_final_page_and_terminal_proof():
    with pytest.raises(PagePipelineStateError):
        PageExecutionResult.build(status="final_verified", terminal_proof=None, **_page_result_parts())
    with pytest.raises(PagePipelineIdentityError):
        PageExecutionResult.build(
            status="final_verified",
            final_page=_final_page(pixel_sha="sha-final"),
            terminal_proof=_proof(final_page_pixel_sha256="sha-other"),
            **_page_result_parts(),
        )


def test_candidate_result_cannot_enter_project_export_adapter():
    page = adapt_page_execution_result_to_output_page(_candidate_page_result())
    with pytest.raises(PageNotTerminalError):
        main._project_inputs_from_output_pages([page])


def test_final_page_snapshot_cannot_be_mutated_after_terminal_verification():
    result = _verified_page_result()
    pixels = result.final_page.read_only_rgb()
    with pytest.raises(ValueError):
        pixels[0, 0] = 0
    assert result.final_page.page_output_pixel_sha256 == canonical_page_sha256(_expected_final_pixels())


def test_mutating_source_owner_graph_after_result_build_does_not_change_authority_snapshot():
    graph = _owner_graph()
    result = PageExecutionResult.build(owner_graph=graph, **_page_result_parts())
    graph.owners.clear()
    assert result.owner_graph.read().owners
    assert result.owner_graph.sha256 == sha256_bytes(result.owner_graph.canonical_json_bytes)


def test_adapter_cannot_mutate_canonical_original_snapshot_between_attempts():
    request = _page_request_from_pixels(_original_pixels())
    first = request.original_page.mutable_attempt_copy()
    first[:] = 0
    second = request.original_page.mutable_attempt_copy()
    assert np.array_equal(second, _original_pixels())
    assert request.original_page.page_source_sha256 == canonical_page_sha256(_original_pixels())


def test_page_identity_ignores_lossless_file_metadata_but_file_audit_hash_does_not():
    a, b = _two_png_encodings_with_same_decoded_rgb_and_different_metadata()
    snap_a = OriginalPageSnapshot.from_file(a)
    snap_b = OriginalPageSnapshot.from_file(b)
    assert snap_a.page_source_sha256 == snap_b.page_source_sha256
    assert snap_a.source_file_sha256 != snap_b.source_file_sha256


def test_coverage_pending_is_closed_before_graph_translation_or_execution():
    services = _services_with_coverage_recovery_on_second_attempt()
    result = run_page_owner_pipeline(_page(), services)
    assert services.call_order.index("coverage_ready") < services.call_order.index("owner_graph")
    assert result.coverage.require_ready_for_ownership() is None


def test_translation_infrastructure_failure_aborts_without_final_or_export():
    with pytest.raises(TranslationInfrastructureError):
        run_page_owner_pipeline(_page(), _all_translation_backends_unavailable())
    assert not _output_dir().joinpath("final.png").exists()


def test_run_chapter_keeps_output_page_contract_with_authoritative_page_result():
    output_pages = _run_minimal_chapter(
        owner_graph_mode="enforce",
        services=_verified_terminal_fixture_services(),
    )
    assert all(isinstance(page, OutputPage) for page in output_pages)
    assert all(page.owner_page_result is not None for page in output_pages)
    project_inputs = main._project_inputs_from_output_pages(output_pages)
    assert isinstance(project_inputs, VerifiedProjectInputs)
    assert tuple(Path(item.final_artifact.path) for item in project_inputs.pages) == tuple(
        Path(page.owner_page_result.final_page.artifact_ref.path) for page in output_pages
    )
    assert tuple(item.page_result_sha256 for item in project_inputs.pages) == tuple(
        page.owner_page_result.result_sha256 for page in output_pages
    )
    assert all(item.ocr_result.read()["_owner_graph_mode"] == "enforce" for item in project_inputs.pages)
    assert project_inputs.sha256 == sha256_bytes(project_inputs.canonical_json_bytes)


def test_output_page_compatibility_views_cannot_override_authoritative_result():
    page = adapt_page_execution_result_to_output_page(_verified_page_result())
    page.path = _original_english_page_path()
    page.image[:] = _original_english_pixels()
    page.inpainted_image[:] = _original_english_pixels()
    page.y_top, page.y_bottom = 999, 1000
    page.ocr_result["texts"] = [{"translated": "STALE"}]
    page.ocr_result.pop("_owner_graph_mode", None)
    page.text_layers["texts"] = [{"translated": "STALE"}]
    page.owner_graph = None
    page.owner_composition = None
    page.page_surface_geometry = _stale_geometry()
    inputs = main._project_inputs_from_output_pages([page])
    assert _target_payloads(inputs) == _authoritative_target_payloads(page.owner_page_result)
    assert tuple(Path(item.final_artifact.path) for item in inputs.pages) == (
        Path(page.owner_page_result.final_page.artifact_ref.path),
    )
    assert tuple(item.page_geometry.read() for item in inputs.pages) == (
        _authoritative_page_bounds(page.owner_page_result),
    )


def test_real_run_pipeline_dispatches_verified_bundle_once_and_ignores_every_mutable_view(self):
    with tempfile.TemporaryDirectory() as raw_dir:
        tmp_path = Path(raw_dir)
        page = adapt_page_execution_result_to_output_page(_verified_page_result(tmp_path))
        output_pages = [page]
        _corrupt_every_output_page_view_with_original_english(page)
        config_path = _write_verified_pipeline_config(tmp_path)
        run_chapter = MagicMock(return_value=output_pages)
        adapter = MagicMock(wraps=main._project_inputs_from_output_pages)
        wrap_up = MagicMock(wraps=main._wrap_up_verified_owner_pages)
        build_project = MagicMock(wraps=main.build_project_json)
        with _patched_real_verified_dispatch(
            run_chapter=run_chapter,
            adapter=adapter,
            wrap_up=wrap_up,
            build_project=build_project,
            forbid_legacy_hydrate_normalize_rerender_writers=True,
        ) as forbidden:
            main._run_pipeline(str(config_path))

        run_chapter.assert_called_once()
        adapter.assert_called_once_with(output_pages)
        wrap_up.assert_called_once()
        self.assertTrue(all(spy.call_count == 0 for spy in forbidden))
        self.assertEqual(build_project.call_count, 1)
        self.assertIsNone(build_project.call_args.kwargs.get("output_pages"))
        verified_inputs = wrap_up.call_args.args[0]
        self.assertIsInstance(verified_inputs, VerifiedProjectInputs)
        self.assertFalse(_contains_instance(wrap_up.call_args, OutputPage))
        self.assertFalse(_contains_instance(build_project.call_args, OutputPage))
        self.assertTrue(_contains_instance(build_project.call_args, VerifiedProjectInputs))
        self.assertEqual(verified_inputs.sha256, sha256_bytes(verified_inputs.canonical_json_bytes))
        public_root = _configured_output_root(config_path)
        manifest = load_json(public_root / "export_manifest.json")
        translated = public_root / manifest["pages"][0]["translated_path"]
        project = load_json(public_root / "project.json")
        self.assertEqual(translated.read_bytes(), page.owner_page_result.final_page.lossless_png_bytes)
        self.assertEqual(canonical_file_sha256(translated), page.owner_page_result.final_page.final_file_sha256)
        self.assertEqual(canonical_page_sha256(load_rgb(translated)), page.owner_page_result.final_page.page_output_pixel_sha256)
        self.assertEqual(_project_target_payloads(project), _authoritative_target_payloads(page.owner_page_result))
        self.assertEqual(_project_geometry(project), page.owner_page_result.page_geometry.read())
        self.assertEqual(manifest["page_result_sha256s"], [page.owner_page_result.result_sha256])
        self.assertFalse(_source_english_visible(load_rgb(translated)))


def test_real_run_pipeline_second_page_failure_publishes_no_chapter_outputs(self):
    for failure_mode in ("candidate_returned", "second_page_raises"):
        with self.subTest(failure_mode=failure_mode), tempfile.TemporaryDirectory() as raw_dir:
            tmp_path = Path(raw_dir)
            config_path = _write_verified_pipeline_config(tmp_path)
            public_root = _configured_output_root(config_path)
            run_chapter = _chapter_with_first_page_verified_then_second_page_failure(tmp_path, failure_mode)
            with patch.object(strip_run, "run_chapter", run_chapter):
                with self.assertRaises((PageNotTerminalError, TranslationInfrastructureError)):
                    main._run_pipeline(str(config_path))
            self.assertFalse((public_root / "translated").exists())
            self.assertFalse((public_root / "project.json").exists())
            self.assertFalse((public_root / "export_manifest.json").exists())
            self.assertFalse(_chapter_publication_staging(public_root).exists())


def test_chapter_publication_rolls_back_every_mid_commit_failure(self):
    for existing_publication in (False, True):
        for fault_phase in ("after_translated_promote", "after_project_promote", "after_manifest_promote"):
            with self.subTest(existing=existing_publication, phase=fault_phase), tempfile.TemporaryDirectory() as raw_dir:
                tmp_path = Path(raw_dir)
                config_path = _write_verified_pipeline_config(tmp_path)
                public_root = _configured_output_root(config_path)
                before = _seed_or_snapshot_publication(public_root, existing=existing_publication)
                page = adapt_page_execution_result_to_output_page(_verified_page_result(tmp_path))
                with patch.object(strip_run, "run_chapter", return_value=[page]), patch.object(
                    main.PublicationTransaction,
                    "_fault_checkpoint",
                    side_effect=_raise_on_publication_phase(fault_phase),
                ):
                    with self.assertRaises(InjectedPublicationFailure):
                        main._run_pipeline(str(config_path))
                self.assertEqual(_snapshot_publication(public_root), before)
                self.assertFalse(_chapter_publication_staging(public_root).exists())
                self.assertFalse(_chapter_publication_backup(public_root).exists())
                self.assertFalse(_chapter_publication_journal(public_root).exists())


def test_recover_chapter_publication_restores_interrupted_journal(self):
    for existing_publication in (False, True):
        for interrupted_phase in (
            "backups_prepared",
            "translated_promoted",
            "project_promoted",
            "manifest_promoted",
        ):
            with self.subTest(existing=existing_publication, phase=interrupted_phase), tempfile.TemporaryDirectory() as raw_dir:
                public_root = Path(raw_dir)
                before = _seed_or_snapshot_publication(public_root, existing=existing_publication)
                _simulate_process_death_with_publication_journal(public_root, interrupted_phase)
                main.recover_chapter_publication(public_root)
                self.assertEqual(_snapshot_publication(public_root), before)
                self.assertFalse(_chapter_publication_staging(public_root).exists())
                self.assertFalse(_chapter_publication_backup(public_root).exists())
                self.assertFalse(_chapter_publication_journal(public_root).exists())


def test_project_adapter_rejects_tampered_persisted_final_artifact():
    page = adapt_page_execution_result_to_output_page(_verified_page_result())
    _overwrite_file(page.owner_page_result.final_page.artifact_ref.path, _original_english_png_bytes())
    with pytest.raises(PageArtifactIntegrityError):
        main._project_inputs_from_output_pages([page])
```

Os três testes que chamam `_run_pipeline` entram em `MainEmitTests` e usam `unittest` (`self`, `TemporaryDirectory`, `patch.object`), não fixtures pytest. `_patched_real_verified_dispatch()` deve patchar `strip_run.run_chapter`, `_project_inputs_from_output_pages`, `_wrap_up_verified_owner_pages` e os helpers legados nos módulos onde `_run_pipeline` realmente os resolve. O config temporário percorre o parser/config loader real; a falha RED deve ser dispatch/publicação, nunca `TypeError` de assinatura nem `AttributeError` de import.

**Step 2: Confirmar RED**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest `
    tests/test_page_owner_pipeline.py `
    tests/test_owner_lifecycle_e2e.py::test_no_band_dialogue_recipe_completes_basic_page_first_lifecycle `
    tests/test_strip_owner_control_plane.py `
    tests/test_strip_owner_composition_integration.py `
    tests/test_main_emit.py `
    tests/test_project_writer.py `
    -q
} finally { Pop-Location }
```

Expected: não existe coordenador/adapter único, `_run_pipeline` ainda alcança views mutáveis e cria `translated/` cedo; fault injection deixa publicação parcial ou sem restauração transacional.

**Step 3: Implementar contratos do coordenador**

```python
@dataclass(frozen=True)
class BandProjection:
    band_id: str
    bbox_page: BBox
    source_crop_sha256: str
    provenance_ids: tuple[str, ...]


@dataclass(frozen=True)
class OriginalPageSnapshot:
    lossless_png_bytes: bytes
    source_file_sha256: str
    page_source_sha256: str
    width: int
    height: int
    mode: Literal["RGB"]

    def read_only_rgb(self) -> NDArray[np.uint8]: ...
    def mutable_attempt_copy(self) -> NDArray[np.uint8]: ...


@dataclass(frozen=True)
class PagePipelineRequest:
    run_id: str
    page_id: str
    page_source_sha256: str
    original_page: OriginalPageSnapshot
    band_projections: tuple[BandProjection, ...]


PageExecutionStatus = Literal["candidate_ready", "final_verified"]


@dataclass(frozen=True)
class PagePixelSnapshot:
    lossless_png_bytes: bytes
    pixel_sha256: str
    width: int
    height: int
    mode: Literal["RGB"]

    def read_only_rgb(self) -> NDArray[np.uint8]: ...
    def mutable_attempt_copy(self) -> NDArray[np.uint8]: ...


@dataclass(frozen=True)
class PageGeometrySnapshot:
    canonical_json_bytes: bytes
    sha256: str
    y_top: int
    y_bottom: int
    width: int
    height: int

    def read(self) -> dict[str, JSONValue]: ...  # fresh decoded copy


@dataclass(frozen=True)
class FrozenJSONSnapshot:
    canonical_json_bytes: bytes
    sha256: str

    def read(self) -> dict[str, JSONValue]: ...  # fresh decoded copy


@dataclass(frozen=True)
class OwnerGraphSnapshot:
    canonical_json_bytes: bytes
    sha256: str
    schema_version: int
    verification_status: Literal["verified"]
    run_id: str
    page_id: str
    page_source_sha256: str

    def read(self) -> OwnerGraph: ...  # validated fresh copy


@dataclass(frozen=True)
class PageCompositionSnapshot:
    canonical_json_bytes: bytes
    sha256: str
    run_id: str
    page_id: str
    page_source_sha256: str
    cleanup_base_sha256: str
    final_pixel_sha256: str
    commit_ids: tuple[str, ...]
    translation_binding_sha256s: tuple[str, ...]
    source_payload_sha256s: tuple[str, ...]
    target_payload_sha256s: tuple[str, ...]
    target_glyph_patch_sha256s: tuple[str, ...]
    target_materialization_sha256s: tuple[str, ...]

    def read(self) -> PageCompositionResult: ...


@dataclass(frozen=True)
class PersistedImageArtifactRef:
    path: str
    file_sha256: str
    pixel_sha256: str
    width: int
    height: int
    mode: Literal["RGB"]

    def load_verified(self) -> NDArray[np.uint8]: ...


@dataclass(frozen=True)
class FinalPageSnapshot:
    lossless_png_bytes: bytes
    page_output_pixel_sha256: str
    final_file_sha256: str
    artifact_ref: PersistedImageArtifactRef
    width: int
    height: int
    mode: Literal["RGB"]

    def read_only_rgb(self) -> NDArray[np.uint8]: ...


@dataclass(frozen=True)
class TerminalPixelProof:
    run_id: str
    page_id: str
    page_source_sha256: str
    final_page_pixel_sha256: str
    cleanup_base_sha256: str
    composition_sha256: str
    translation_binding_sha256s: tuple[str, ...]
    source_payload_sha256s: tuple[str, ...]
    target_payload_sha256s: tuple[str, ...]
    target_glyph_patch_sha256s: tuple[str, ...]
    target_materialization_sha256s: tuple[str, ...]
    fresh_ocr_invocation_id: str
    fresh_ocr_root_input_pixel_sha256: str
    fresh_ocr_attempt_ids: tuple[str, ...]
    fresh_ocr_attempt_chain_sha256: str
    fresh_ocr_payload_sha256: str
    source_support_removed: bool
    target_glyph_patch_applied: bool
    fresh_source_ocr_absent: bool
    coverage_complete: bool
    unowned_material_text_absent: bool
    proof_sha256: str


@dataclass(frozen=True)
class VerifiedPageProjectInput:
    run_id: str
    page_id: str
    page_source_sha256: str
    page_result_sha256: str
    final_artifact: PersistedImageArtifactRef
    final_page_pixel_sha256: str
    page_geometry: PageGeometrySnapshot
    ocr_result: FrozenJSONSnapshot
    text_layers: FrozenJSONSnapshot
    owner_graph: OwnerGraphSnapshot
    page_composition: PageCompositionSnapshot
    terminal_proof_sha256: str


@dataclass(frozen=True)
class VerifiedProjectInputs:
    run_id: str
    pages: tuple[VerifiedPageProjectInput, ...]
    canonical_json_bytes: bytes
    sha256: str

    @classmethod
    def build(cls, pages: tuple[VerifiedPageProjectInput, ...]) -> "VerifiedProjectInputs": ...


@dataclass(frozen=True)
class VerifiedChapterBundle:
    run_id: str
    verified_inputs_sha256: str
    translated_artifacts: tuple[PersistedImageArtifactRef, ...]
    project_snapshot: FrozenJSONSnapshot
    export_manifest_snapshot: FrozenJSONSnapshot
    bundle_sha256: str


@dataclass
class PublicationTransaction:
    run_root: Path
    transaction_id: str
    staging_root: Path
    backup_root: Path
    journal_path: Path

    def stage(self, bundle: VerifiedChapterBundle) -> None: ...
    def commit(self, bundle: VerifiedChapterBundle) -> None: ...
    def rollback(self) -> None: ...
    def _fault_checkpoint(self, phase: str) -> None: ...  # no-op em produção; seam de teste


@dataclass(frozen=True)
class PageExecutionResult:
    run_id: str
    page_id: str
    page_source_sha256: str
    result_sha256: str
    status: PageExecutionStatus
    original_page: OriginalPageSnapshot
    cleanup_base: PagePixelSnapshot
    page_geometry: PageGeometrySnapshot
    ocr_result_view: FrozenJSONSnapshot
    text_layers_view: FrozenJSONSnapshot
    ocr_requests: tuple[OCRRequest, ...]
    ocr_invocations: tuple[OCRInvocationResult, ...]
    coverage: PageCoverageLedger
    owner_graph: OwnerGraphSnapshot
    translations: tuple[TranslationBinding, ...]
    page_commits: tuple[OwnerExecutionCommit, ...]
    page_composition: PageCompositionSnapshot
    final_page: FinalPageSnapshot | None
    terminal_proof: TerminalPixelProof | None = None

    @property
    def lifecycle(self) -> PageCoverageLedger: return self.coverage
```

`PagePixelSnapshot`, `PageGeometrySnapshot`, `FrozenJSONSnapshot`, `OwnerGraphSnapshot`, `PageCompositionSnapshot`, `PersistedImageArtifactRef`, `FinalPageSnapshot`, `TerminalPixelProof`, `VerifiedPageProjectInput`, `VerifiedProjectInputs` e `VerifiedChapterBundle` são contratos reais desta Task, não pseudotipos. Os construtores recalculam seus hashes; `read()` sempre desserializa uma cópia nova; arrays retornados por `read_only_rgb()` têm `writeable=False`. `VerifiedProjectInputs` não contém `OutputPage`, list ou dict mutável: apenas tuples, snapshots frozen e artifact refs verificados, e seu `sha256` cobre o JSON canônico de todas as páginas na ordem. `VerifiedChapterBundle` repete esse hash e cobre snapshots dos outputs antes da publicação. O lifecycle não é uma cópia concorrente: `result.lifecycle` é uma propriedade read-only sobre o mesmo `PageCoverageLedger` frozen que contém os estados. `OwnerGraphSnapshot.build()` extrai `run_id/page_id/page_source_sha256` de todas as observações, exige unanimidade com request+ledger e `verification_status="verified"`; graph vazio usa explicitamente a identidade do request e ainda é validado pelo ledger. `TerminalPixelProof` nasce aqui apenas como envelope serializável/hash-linked para permitir testar o boundary e o adapter; somente fixtures podem construí-lo diretamente. A Task 15 implementa o builder de produção a partir do PNG persistido e do OCR fresco.

Nesta Task, o coordenador chama: discovery → OCR/coverage → association/container recovery → graph → translation → adapters atuais de execution/final QA. Sem uma prova terminal injetada em teste, o resultado intermediário é `candidate_ready`, mantém `final_page=None` e é recusado pelo adapter de projeto/export. As Tasks 10–15 substituem os adapters pela transação, acrescentam `repair_history` ao `PageExecutionResult` e produzem o primeiro `final_verified` de produção sem quebrar o contrato page-first intermediário.

Nesta Task, `PageExecutionResult.build()` valida o que já existe: toda request/invocation OCR, ledger/lifecycle, graph, binding, commit e composição repete `run_id/page_id/page_source_sha256`; cada invocation valida sua request raiz e a cadeia de attempts físicos; `page_composition.commit_ids` e os conjuntos ordenados de binding/source/target hashes coincidem com as coleções do resultado. `target_glyph_patch_sha256s` e `target_materialization_sha256s` permanecem tuplas vazias no resultado candidato intermediário; a Task 10 cria esses records e elimina essa tolerância para owner traduzível. `final_verified` de fixture exige simultaneamente `final_page` e `terminal_proof`, identidades iguais, `terminal_proof.composition_sha256 == page_composition.sha256` e `terminal_proof.final_page_pixel_sha256 == terminal_proof.fresh_ocr_root_input_pixel_sha256 == final_page.page_output_pixel_sha256 == final_page.artifact_ref.pixel_sha256`; `fresh_ocr_attempt_chain_sha256` cobre os attempts ordenados e nenhum record pode apontar para attempt ausente. `candidate_ready` exige prova/final ausentes. Estado transitório ou erro operacional não é serializado como resultado exportável. O `result_sha256` cobre status e hashes de todos os snapshots/artefatos/provas.

`from_legacy_bands()` copia somente geometria/hash/proveniência para `BandProjection` frozen. Nunca importe `ocr_result`, `cleaned_slice`, `rendered_slice` ou decisões de um `Band`; nunca escreva de volta no objeto mutável. Adapters recebem `OriginalPageSnapshot.mutable_attempt_copy()`, de modo que uma mutação OpenCV não contamina o original canônico nem o retry seguinte.

Antes do graph, execute um loop de fechamento de coverage. Cada `coverage_recovery_pending` deve consumir a próxima tentativa determinística (full-page, associação, crop ancorado, variante visual, container conservador) e produzir um snapshot novo; graph/translation/execution ficam proibidos enquanto `require_ready_for_ownership()` falhar. Não repita uma combinação `component_id + attempt kind + input hash`. Esgotadas tentativas visuais com glyph support material positivo, materialize obrigatoriamente componente + container support-local e marque `protection_conflict`; pending não pode virar terminal de conteúdo. Exaustão por falha real de engine é erro operacional sem export, não preservação de conteúdo.

Antes da execução, feche todos os bindings. Resposta inválida avança backend/prompt/contexto; indisponibilidade total levanta `TranslationInfrastructureError` e não cria `final_page`. `translation_recovery_pending` é estado transitório consumido pelo coordenador, nunca retorno final.

**Step 4: Integrar de forma incremental**

- `run_chapter()` continua retornando exatamente `list[OutputPage]`. Ele mantém `candidate_ready` somente em memória, persiste apenas `final_verified` e usa `adapt_page_execution_result_to_output_page()` para ligar o snapshot autoritativo ao contrato existente.
- acrescente `owner_page_result: PageExecutionResult | None` a `OutputPage` por forward reference/`TYPE_CHECKING`; `y_top/y_bottom/image/path/original_image/inpainted_image/owner_graph/ocr_result/text_layers/page_surface_geometry` permanecem views de compatibilidade derivadas, nunca fontes de autoridade;
- `PageExecutionResult` carrega snapshots frozen/hash-linked do request original, cleanup base, geometria, owner graph, OCR compatibility view, target text layers e final. `FinalPageSnapshot.artifact_ref` aponta para o PNG persistido e prova hash de arquivo **e** pixels decodificados;
- crie em `main.py` `_project_inputs_from_output_pages() -> VerifiedProjectInputs`: quando `owner_page_result` existe, exige `status == "final_verified"`, reabre `artifact_ref.load_verified()`, extrai bounds/metadata/text layers dos snapshots e ignora todas as views mutáveis de `OutputPage`. Estado não terminal lança `PageNotTerminalError`; arquivo ausente ou qualquer hash divergente lança `PageArtifactIntegrityError`; sem page result, mantém o comportamento legacy até a retirada final da Task 17. O adapter é o **único** boundary que aceita `OutputPage` no branch verificado e é chamado exatamente uma vez;
- a cadeia normativa é `list[OutputPage] -> VerifiedProjectInputs -> _wrap_up_verified_owner_pages(inputs) -> VerifiedChapterBundle -> PublicationTransaction`. Extraia do corpo **real** de `_run_pipeline(config_path: str)` um único boundary `_run_verified_strip_chapter(...)`: ele chama `run_chapter()`, passa seu retorno uma vez ao adapter e nunca mais retém `OutputPage`. Para criar um seam real sem mudar a assinatura pública, mova o import local de `run_chapter` para `import strip.run as strip_run` no módulo; `_run_verified_strip_chapter(..., run_chapter_fn=None)` resolve `strip_run.run_chapter` quando `None`. Os REDs criam config JSON temporário válido, patcham `strip_run.run_chapter` e chamam `main._run_pipeline(str(config_path))`, sem kwargs inexistentes. Espione `adapter.assert_called_once_with(output_pages)` e prove recursivamente que args de wrap-up/builders não contêm `OutputPage`;
- `_wrap_up_verified_owner_pages(inputs: VerifiedProjectInputs) -> VerifiedChapterBundle` preflighta **todas** as páginas, reabre/verifica artifacts e monta translated/project/manifest sob `<run>/.publication-staging/<transaction_id>/`. `build_project_json` recebe somente `VerifiedProjectInputs`; nenhum dict/list derivado de `OutputPage` atravessa o boundary. Substitua em `main.py:9410+` as leituras de `p.path/p.ocr_result/p.text_layers/inpainted_image/image/original_image`, em `_owner_pages_have_final_pixel_authority()` a inferência por `_owner_graph_mode/owner_graph/owner_composition`, e em `build_project_json(..., output_pages=...)` a geometria lida de `page_surface_geometry`. Imagens traduzidas vêm de `FinalPageSnapshot.artifact_ref`; OCR/text layers/geometry vêm dos snapshots; authority vem de status + proof + hashes;
- implemente `PublicationTransaction` com alvos resolvidos exatos `translated/`, `project.json` e `export_manifest.json`, journal frozen/hash-linked e backup por target. Só após bundle completo/hash-verificado, mova targets preexistentes para `.publication-backup/<transaction_id>/`, promova translated, project e manifest (manifest por último) e faça fsync dos parents. Qualquer exceção em qualquer fase remove apenas targets promovidos por essa transação, restaura backups e apaga staging/journal/backup; `recover_chapter_publication()` executa a mesma restauração após interrupção. Fault injection e recovery pós-morte cobrem run nova e publicação anterior, desde `backups_prepared` até cada promote: o resultado deve ser os três outputs anteriores byte-idênticos, ou os três inexistentes numa run nova. Não execute `translated_dir.mkdir()` antecipadamente no modo `enforce`; isso permanece apenas para legacy/manual;
- depois do adapter, o branch `enforce` não entrega `OutputPage` a normalizador, hydrate, rerender, QA ou writer tardio. Faça um `rg` de todos os consumidores (`p.path`, `p.ocr_result`, `p.text_layers`, `page_surface_geometry` e helpers em torno de `main.py:9410-9593,9976,15550-15615`) e migre-os para `VerifiedProjectInputs`/snapshots ou torne-os exclusivos do legacy/shadow. O teste real corrompe deliberadamente todas as views, inclusive `_owner_graph_mode`, `owner_graph` e `owner_composition`;
- `process_band()` permanece adapter de crop/OCR/inpaint/render, sem decidir owner ou terminalidade.
- composição recebe `page_commits` diretamente; remova do caminho `enforce` a anexação em bands de `run.py:6312` e a reextração em `run.py:6599`.
- mantenha o caminho antigo apenas em `shadow` enquanto a compatibilidade é testada.

**Step 5: Confirmar GREEN**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest `
    tests/test_page_owner_pipeline.py `
    tests/test_owner_lifecycle_e2e.py::test_no_band_dialogue_recipe_completes_basic_page_first_lifecycle `
    tests/test_strip_owner_control_plane.py `
    tests/test_strip_owner_composition_integration.py `
    tests/test_owner_lifecycle.py `
    tests/test_main_emit.py `
    tests/test_project_writer.py `
    -q
} finally { Pop-Location }
```

Expected: PASS; mudar o particionamento de bands não muda owners/traduções/commits, o adapter é único e falha multipágina ou mid-commit não deixa publicação parcial.

**Step 6: Commit**

```powershell
git add -- pipeline/strip/page_pipeline.py pipeline/tests/test_page_owner_pipeline.py pipeline/tests/test_owner_lifecycle_e2e.py
git add -p -- pipeline/strip/types.py pipeline/strip/run.py pipeline/strip/process_bands.py pipeline/main.py pipeline/ownership/lifecycle.py pipeline/ownership/translation.py pipeline/tests/test_strip_owner_control_plane.py pipeline/tests/test_strip_owner_composition_integration.py pipeline/tests/test_main_emit.py pipeline/tests/test_project_writer.py
git diff --cached --check
git commit -m "refactor: coordinate owner processing in page space"
```

### Task 10: Tornar cleanup e render uma transação atômica recuperável

**Files:**

- Create: `pipeline/ownership/execution.py`
- Modify: `pipeline/ownership/model.py`
- Modify: `pipeline/inpainter/owner_mask.py`
- Modify: `pipeline/strip/process_bands.py`
- Modify: `pipeline/strip/page_pipeline.py`
- Modify: `pipeline/ownership/lifecycle.py`
- Test: `pipeline/tests/test_owner_atomic_execution.py`
- Test: `pipeline/tests/test_strip_owner_composition_integration.py`
- Modify existing untracked test: `pipeline/tests/test_owner_source_replacement_fail_closed.py`

**Step 1: Inspecionar e preservar o teste local antes da edição**

```powershell
git status --short -- pipeline/tests/test_owner_source_replacement_fail_closed.py
$localTestDiff = & git diff --no-index -- NUL pipeline/tests/test_owner_source_replacement_fail_closed.py
$localTestDiffExit = $LASTEXITCODE
$localTestDiff
if ($localTestDiffExit -gt 1) { throw "could not inspect untracked test: $localTestDiffExit" }
```

Não substitua o arquivo. Altere somente as expectativas que hoje concluem `review/source preserved`, mantendo todos os cenários já escritos pelo usuário.

**Step 2: Escrever/migrar testes RED da transação**

```python
def test_cleanup_and_pt_br_render_commit_as_one_hash_chain():
    tx = _transaction()
    committed = tx.commit(_valid_cleanup(), _valid_ptbr_glyph_patch())
    assert committed.cleanup.source_before_sha256 == tx.source_page_sha256
    assert committed.render.base_sha256 == committed.cleanup.result_sha256
    assert committed.final_pixel_sha256 == canonical_page_sha256(committed.final_page)
    assert committed.translation_binding_sha256 == tx.translation.translation_binding_sha256
    assert committed.source_payload_sha256 == tx.translation.source_payload_sha256
    assert committed.target_payload_sha256 == tx.translation.target_payload_sha256
    assert committed.target_glyph_patch_sha256 == canonical_glyph_patch_sha256(committed.render)


@pytest.mark.parametrize(
    "field",
    [
        "translation_binding_sha256",
        "source_payload_sha256",
        "target_payload_sha256",
        "target_glyph_patch_sha256",
    ],
)
def test_atomic_commit_rejects_stale_binding_or_target_hash(field):
    with pytest.raises(AtomicReplacementError):
        _transaction().commit(_valid_cleanup(), _valid_ptbr_glyph_patch(), overrides={field: "stale"})


def test_candidate_page_result_requires_exact_binding_commit_materialization_composition_chain():
    result = _candidate_result_with_one_owner()
    assert result.translations[0].translation_binding_sha256 == result.page_commits[0].translation_binding_sha256
    assert result.page_commits[0].target_payload_sha256 == result.owner_target_materializations[0].target_payload_sha256
    assert result.page_commits[0].target_glyph_patch_sha256 == result.owner_target_materializations[0].target_glyph_patch_sha256
    assert result.page_composition.target_materialization_sha256s == (
        result.owner_target_materializations[0].materialization_sha256,
    )
    assert _layer_target_hash_map(result.text_layers_view.read()) == {
        result.translations[0].owner_id: result.translations[0].target_payload_sha256
    }
    with pytest.raises(PagePipelineIdentityError):
        PageExecutionResult.build_from(
            result,
            owner_target_materializations=(
                dataclasses.replace(result.owner_target_materializations[0], target_payload_sha256="stale"),
            ),
        )
    with pytest.raises(PagePipelineIdentityError):
        PageExecutionResult.build_from(result, text_layers_view=_stale_english_text_layers_snapshot())


def test_mask_failure_requests_repair_instead_of_finishing_with_source():
    result = execute_owner_replacement(_owner(), _mask_failure_services())
    assert result.status == "repair_pending"
    assert result.repair_request.reason == "mask_failure"
    assert result.final_page is None


def test_residual_rollback_retries_from_original_pixels():
    first, second = _execute_with_first_residual_then_success()
    assert first.input_sha256 == second.input_sha256 == _original_page_sha256()
    assert second.status == "committed"


def test_cleanup_without_target_or_target_without_cleanup_cannot_commit():
    tx = _transaction()
    with pytest.raises(AtomicReplacementError):
        tx.commit(_valid_cleanup(), None)
    with pytest.raises(AtomicReplacementError):
        tx.commit(None, _valid_ptbr_glyph_patch())
```

**Step 3: Confirmar RED**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest `
    tests/test_owner_atomic_execution.py `
    tests/test_owner_source_replacement_fail_closed.py `
    tests/test_strip_owner_composition_integration.py `
    -q
} finally { Pop-Location }
```

Expected: máscara/inpaint/residual ainda termina em review ou rollback que preserva source.

**Step 4: Implementar `OwnerReplacementTransaction`**

O objeto imutável deve manter:

- identidade run/page/owner/component;
- hash dos pixels originais;
- `TranslationBinding` aceito;
- máscara source, máscara protegida e máscara de cleanup;
- hash do cleanup;
- patch e máscara dos glyphs PT-BR;
- hash do resultado final;
- evidências de verificação.

`commit()` só produz o `OwnerExecutionCommit` já existente quando cleanup e target pertencem à mesma tentativa e à mesma cadeia de hashes. Não introduza um segundo tipo de commit; adapte o modelo existente para carregar e validar `run_id`, `page_id`, `page_source_sha256`, `translation_binding_sha256`, `source_payload_sha256`, `target_payload_sha256` e `target_glyph_patch_sha256`. Os quatro últimos vêm do `TranslationBinding`/patch real e qualquer mismatch aborta antes da composição.

Defina também `OwnerTargetMaterialization` frozen com `materialization_id`, identidade run/page/source/owner, `translation_binding_sha256`, `source_payload_sha256`, `target_payload_sha256`, `target_glyph_patch_sha256`, `glyph_mask_sha256`, `base_pixel_sha256`, `result_pixel_sha256` e `materialization_sha256` canônico. Exatamente uma materialização final corresponde a cada binding. Estenda `PageExecutionResult` com `owner_target_materializations: tuple[OwnerTargetMaterialization, ...]` e exija correspondência 1:1 de owner e hashes com `translations`/`page_commits`. `PageCompositionSnapshot` agrega os mesmos hashes ordenados; não os recalcula de texto solto. A partir deste commit, owner traduzível com tuple vazia de glyph/materialization ou qualquer cardinalidade/hash divergente é inválido inclusive em `candidate_ready`; fica removida a tolerância estrutural temporária da Task 9.

Valide também `text_layers_view`: exatamente uma layer final por binding, ligada por `owner_id`, `translation_binding_sha256` e `target_payload_sha256`; o texto da layer deve hashear para o target e nunca pode vir de source/índice. Essa snapshot é a única fonte de text layers do projeto.

`committed` é exclusivamente o outcome retornado por `OwnerReplacementTransaction`, não um estado do `CoverageEntry`/`OwnerLifecycle`. O commit atômico aplica internamente as transições canônicas `cleaned` e `rendered`; somente o QA terminal da Task 15 pode avançar para `final_verified`.

Defina também `OwnerRepairRequest` em `execution.py` como contrato mínimo (`request_id`, `issue_id: str | None`, `run_id`, `page_id`, `owner_id`, `original_page_sha256`, `failed_stage`, `reason`, `evidence_ids`, `next_strategy`). `request_id` deriva da identidade + stage/reason/evidence; quando o pedido nasce do QA, `issue_id` é obrigatório. Estenda `PageExecutionResult` com `repair_requests: tuple[OwnerRepairRequest, ...]`. A Task 11 o consome no ladder, sem depender antecipadamente do módulo `repair.py`.

**Step 5: Substituir terminais de review por pedidos de reparo**

No caminho `enforce`, converter falhas de máscara, geometria, inpaint, resíduo e render em `OwnerRepairRequest`. O snapshot original continua intacto para a próxima estratégia, mas nunca é usado como página final aprovada.

O lifecycle permitido passa a ser:

```text
owned -> target_ready -> execution_attempt -> cleaned -> rendered
target_ready -> execution_attempt -> repair_pending -> execution_attempt
rendered -> repair_pending -> execution_attempt
rendered -> final_verified  # somente após TerminalPixelProof
```

Uma falha operacional irrecuperável pode encerrar o processo sem exportar, mas um defeito de conteúdo jamais deve produzir uma saída terminal contendo o inglês.

**Step 6: Confirmar GREEN**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest `
    tests/test_owner_atomic_execution.py `
    tests/test_owner_source_replacement_fail_closed.py `
    tests/test_strip_owner_composition_integration.py `
    tests/test_owner_lifecycle.py `
    -q
} finally { Pop-Location }
```

Expected: PASS; rollback protege os pixels originais internamente e sempre agenda uma nova estratégia.

**Step 7: Commit**

```powershell
git add -- pipeline/ownership/execution.py
git add -p -- pipeline/ownership/model.py pipeline/inpainter/owner_mask.py pipeline/strip/process_bands.py pipeline/strip/page_pipeline.py pipeline/ownership/lifecycle.py pipeline/tests/test_owner_atomic_execution.py pipeline/tests/test_owner_source_replacement_fail_closed.py pipeline/tests/test_strip_owner_composition_integration.py
git diff --cached --check
git commit -m "feat: make owner replacement atomic and recoverable"
```

### Task 11: Implementar R0 preciso e R1 de suporte expandido

**Files:**

- Create: `pipeline/ownership/repair.py`
- Modify: `pipeline/inpainter/owner_mask.py`
- Modify: `pipeline/qa/inpaint_residual.py`
- Modify: `pipeline/ownership/execution.py`
- Modify: `pipeline/strip/page_pipeline.py`
- Create: `pipeline/tests/test_owner_repair.py`
- Test: `pipeline/tests/test_owner_atomic_execution.py`
- Test: `pipeline/tests/test_owner_mask.py`
- Test: `pipeline/tests/test_inpaint_debug_residual.py`

**Step 1: Escrever testes RED da política progressiva**

```python
def test_r0_uses_precise_source_support_and_preserves_container_border():
    attempt = build_repair_attempt(_fixture(), strategy="R0")
    assert _covers_all_source_glyphs(attempt.cleanup_mask)
    assert not _overlaps(attempt.cleanup_mask, _container_border_mask())


def test_r1_expands_only_from_positive_source_support():
    r0 = build_repair_attempt(_fixture_with_halo(), strategy="R0")
    r1 = build_repair_attempt(_fixture_with_halo(), strategy="R1", previous=r0)
    assert _area(r1.cleanup_mask) > _area(r0.cleanup_mask)
    assert _within_container(r1.cleanup_mask)
    assert _preserves_protected_art(r1.cleanup_mask)


def test_r0_residual_automatically_advances_to_r1_from_original_page():
    result = run_repair_ladder(_owner_case(first_strategy_leaves_residual=True), max_strategy="R1")
    assert [a.strategy for a in result.attempts] == ["R0", "R1"]
    assert len({a.input_sha256 for a in result.attempts}) == 1
    assert result.status == "committed"


def test_r1_failure_does_not_return_source_page_as_final():
    result = run_repair_ladder(_case_failing_through_r1(), max_strategy="R1")
    assert result.status == "repair_pending"
    assert result.next_strategy == "R2"
    assert result.final_page is None


def test_repair_attempt_consumes_exact_request_and_originating_qa_issue():
    request = _repair_request(request_id="request-1", issue_id="issue-1")
    attempt = build_repair_attempt(_fixture(), strategy="R0", request=request)
    assert attempt.request_id == "request-1"
    assert attempt.issue_id == "issue-1"
    assert request.request_id in attempt.consumed_request_ids
```

**Step 2: Confirmar RED**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest `
    tests/test_owner_repair.py `
    tests/test_owner_atomic_execution.py `
    tests/test_owner_mask.py `
    tests/test_inpaint_debug_residual.py `
    -q
} finally { Pop-Location }
```

Expected: não existe ladder canônica e falhas continuam terminando em review/rollback.

**Step 3: Implementar contratos de reparo**

```python
class RepairStrategy(str, Enum):
    R0_PRECISE_GLYPH = "R0"
    R1_EXPANDED_SUPPORT = "R1"
    R2_TEXT_REGION_REBUILD = "R2"
    R3_CONTAINER_INTERIOR_REBUILD = "R3"


@dataclass(frozen=True)
class RepairAttempt:
    run_id: str
    page_id: str
    page_source_sha256: str
    attempt_id: str
    request_id: str
    issue_id: str | None
    consumed_request_ids: tuple[str, ...]
    translation_binding_sha256: str
    strategy: RepairStrategy
    variant: str
    owner_id: str
    input_sha256: str
    cleanup_mask_sha256: str
    protected_mask_sha256: str
    outcome: str
    evidence_ids: tuple[str, ...]
```

Toda tentativa nasce de um `OwnerRepairRequest`; `request_id` é obrigatório e aparece exatamente uma vez em `consumed_request_ids`. `issue_id` deve coincidir quando o request veio do QA. Retries encadeados podem consumir requests adicionais, mas nenhum request pode desaparecer nem ser consumido por duas tentativas finais concorrentes. Nesta Task, estenda `PageExecutionResult` com `repair_history: tuple[RepairAttempt, ...]`. O controller escolhe somente a próxima estratégia, nunca coordenadas ou exceções específicas da obra.

**Step 4: Implementar R0**

- união dos glyph supports de todas as observações correlacionadas do source;
- anti-alias/halo medido a partir das bordas reais;
- clipping pelo interior do container;
- subtração de borda e arte protegida;
- inpaint a partir dos pixels originais;
- render do binding PT-BR e verificação atômica.

**Step 5: Implementar R1**

- dilatação anisotrópica baseada na altura/espessura dos glyphs, não em constante da fixture;
- inclusão de resíduos positivos encontrados pelo probe;
- união entre linhas pertencentes ao mesmo owner sem preencher o container inteiro;
- mesma proteção de borda/arte;
- reinício nos pixels originais.

**Step 6: Confirmar GREEN**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest `
    tests/test_owner_repair.py `
    tests/test_owner_atomic_execution.py `
    tests/test_owner_mask.py `
    tests/test_inpaint_debug_residual.py `
    -q
} finally { Pop-Location }
```

Expected: PASS; R0 e R1 são determinísticos, auditáveis e nunca aprovam rollback do source.

**Step 7: Commit**

```powershell
git add -- pipeline/ownership/repair.py pipeline/tests/test_owner_repair.py
git add -p -- pipeline/inpainter/owner_mask.py pipeline/qa/inpaint_residual.py pipeline/ownership/execution.py pipeline/strip/page_pipeline.py pipeline/tests/test_owner_atomic_execution.py pipeline/tests/test_owner_mask.py pipeline/tests/test_inpaint_debug_residual.py
git diff --cached --check
git commit -m "feat: add precise and expanded owner repair strategies"
```

### Task 12: Implementar R2 reconstruindo a região textual completa

**Files:**

- Modify: `pipeline/ownership/repair.py`
- Modify: `pipeline/inpainter/owner_mask.py`
- Modify: `pipeline/qa/inpaint_residual.py`
- Modify: `pipeline/typesetter/renderer.py`
- Modify: `pipeline/ownership/execution.py`
- Test: `pipeline/tests/test_owner_repair.py`
- Test: `pipeline/tests/test_owner_atomic_execution.py`
- Test: `pipeline/tests/test_typesetting_renderer.py`
- Test: `pipeline/tests/test_strip_owner_composition_integration.py`

**Step 1: Escrever testes RED de R2**

```python
def test_r2_rebuilds_union_of_source_lines_not_individual_fragments():
    attempt = build_repair_attempt(_multi_line_fragmented_source(), strategy="R2")
    assert _covers(attempt.cleanup_mask, _union_of_source_line_regions())
    assert _connected_by_line_group(attempt.cleanup_mask)


def test_r2_removes_mixed_en_pt_overlay_before_single_ptbr_render():
    result = run_repair_ladder(_mixed_language_overlay_case(), start_strategy="R2", max_strategy="R2")
    assert result.status == "committed"
    assert _visible_text_layers(result.final_page) == [result.translation.target_text]


def test_r2_renders_one_unsplit_target_body_for_one_owner():
    result = run_repair_ladder(_fragmented_ocr_one_balloon(), start_strategy="R2", max_strategy="R2")
    assert result.commit.render.owner_id == _owner_id()
    assert result.commit.render.text_layer_count == 1
    assert result.commit.render.target_text == _full_ptbr_payload()


def test_r2_preserves_border_and_art_outside_text_region():
    before, after, masks = _run_r2_on_translucent_balloon()
    assert np.array_equal(before[masks.protected], after[masks.protected])
    assert _border_similarity(before, after, masks.border) >= 0.99
```

**Step 2: Confirmar RED**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest `
    tests/test_owner_repair.py `
    tests/test_owner_atomic_execution.py `
    tests/test_typesetting_renderer.py `
    tests/test_strip_owner_composition_integration.py `
    -q
} finally { Pop-Location }
```

Expected: não existe R2 e fragments ainda podem gerar várias camadas ou cleanup parcial.

**Step 3: Implementar `R2_TEXT_REGION_REBUILD`**

Calcular uma região textual única por owner usando a união de:

- glyph supports e bboxes OCR associados;
- fragmentos same-balloon já consolidados no payload;
- halos/resíduos confirmados;
- sobreposições EN+PT da entrada contaminada.

A região pode incluir espaços internos entre glyphs/linhas, mas deve permanecer dentro do interior seguro do container. Reconstruir o fundo dessa região a partir de contexto do próprio container; não copiar conteúdo de outro band/página.

**Step 4: Renderizar o corpo como unidade semântica**

O typesetter recebe um único `TranslationBinding` e uma única caixa utilizável. Quebras de linha são layout do payload completo, não novos owners/camadas. O renderer vetorial/FT2Font existente deve manter anti-alias e nunca rasterizar uma fonte de preview em baixa resolução.

Style-copy continua fora do veredito: use fonte base configurada quando o style copier estiver em `off`, `shadow` ou sem confiança.

**Step 5: Confirmar GREEN**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest `
    tests/test_owner_repair.py `
    tests/test_owner_atomic_execution.py `
    tests/test_typesetting_renderer.py `
    tests/test_strip_owner_composition_integration.py `
    -q
} finally { Pop-Location }
```

Expected: PASS; overlay misto vira exatamente um corpo PT-BR e nenhuma linha é executada isoladamente.

**Step 6: Commit**

```powershell
git add -p -- pipeline/ownership/repair.py pipeline/inpainter/owner_mask.py pipeline/qa/inpaint_residual.py pipeline/typesetter/renderer.py pipeline/ownership/execution.py pipeline/tests/test_owner_repair.py pipeline/tests/test_owner_atomic_execution.py pipeline/tests/test_typesetting_renderer.py pipeline/tests/test_strip_owner_composition_integration.py
git diff --cached --check
git commit -m "feat: rebuild complete owner text regions"
```

### Task 13: Implementar R3 obrigatório para reconstrução do interior do container

**Files:**

- Modify: `pipeline/ownership/repair.py`
- Modify: `pipeline/ownership/execution.py`
- Modify: `pipeline/inpainter/owner_mask.py`
- Modify: `pipeline/qa/inpaint_residual.py`
- Modify: `pipeline/strip/page_pipeline.py`
- Test: `pipeline/tests/test_owner_repair.py`
- Test: `pipeline/tests/test_owner_atomic_execution.py`
- Test: `pipeline/tests/test_owner_lifecycle.py`
- Test: `pipeline/tests/test_strip_owner_composition_integration.py`

**Step 1: Escrever testes RED do fallback terminal**

```python
@pytest.mark.parametrize("failure", ["missing_mask", "inpaint_residual", "mixed_overlay", "target_collision"])
def test_content_failures_reach_r3_and_finish_with_ptbr(failure):
    result = run_repair_ladder(_hard_case(failure))
    assert [a.strategy for a in result.attempts][-1] == RepairStrategy.R3_CONTAINER_INTERIOR_REBUILD
    assert result.status == "committed"
    assert _source_english_visible(result.final_page) is False
    assert _target_ptbr_visible(result.final_page) is True


def test_r3_reconstructs_interior_but_preserves_container_border():
    before, result, masks = _run_r3(_translucent_container_case())
    assert np.array_equal(before[masks.border], result.final_page[masks.border])
    assert np.array_equal(before[masks.protected_art], result.final_page[masks.protected_art])
    assert _interior_has_no_source_glyph_support(result.final_page, masks.interior)


def test_r3_is_not_skipped_by_art_protection_conflict_inside_source_support():
    result = run_repair_ladder(_source_text_overlaps_conservative_protection())
    assert result.status == "committed"
    assert result.attempts[-1].strategy == RepairStrategy.R3_CONTAINER_INTERIOR_REBUILD


def test_r3_residual_forces_deterministic_safe_interior_fill():
    result = run_repair_ladder(_case_where_contextual_r3_leaves_source_support())
    assert result.status == "committed"
    assert result.attempts[-1].strategy == RepairStrategy.R3_CONTAINER_INTERIOR_REBUILD
    assert result.attempts[-1].variant == "deterministic_interior_fill"
    assert _source_english_visible(result.final_page) is False
    assert _target_ptbr_visible(result.final_page) is True


def test_r3_preserves_genuine_protected_art_outside_confirmed_source_support():
    before, result, masks = _run_r3(_container_with_icon_and_source_text())
    assert np.array_equal(before[masks.genuine_protected_art], result.final_page[masks.genuine_protected_art])
    assert _source_english_visible(result.final_page) is False


def test_r3_difficult_protected_geometry_uses_hard_clipped_support_local_rebuild():
    result = run_repair_ladder(_unassociated_missing_container_text_over_protected_art())
    assert result.status == "committed"
    assert result.attempts[-1].variant == "deterministic_support_local_fill"
    assert _source_english_visible(result.final_page) is False
    assert _target_ptbr_visible(result.final_page) is True


def test_content_repair_never_finishes_as_review_or_source_preserved():
    result = run_page_owner_pipeline(_hard_page(), _services())
    assert all(entry.state not in {"review", "source_preserved"} for entry in result.coverage.entries)
```

**Step 2: Confirmar RED**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest `
    tests/test_owner_repair.py `
    tests/test_owner_atomic_execution.py `
    tests/test_owner_lifecycle.py `
    tests/test_strip_owner_composition_integration.py `
    -q
} finally { Pop-Location }
```

Expected: ladder termina em review/rollback antes de uma reconstrução integral.

**Step 3: Implementar `R3_CONTAINER_INTERIOR_REBUILD`**

R3 é obrigatório após R0–R2 e recebe:

- máscara do interior obtida por flood-fill/contorno sem a borda;
- união completa de texto-fonte e overlays presentes;
- amostras robustas do fundo do próprio container;
- protected art revisada para distinguir arte real de pixels classificados por engano como texto/proteção;
- `TranslationBinding` integral.

Reconstruir o interior por modelo local compatível com o container:

- fundo aproximadamente uniforme: preenchimento robusto com preservação de ruído/alpha;
- gradiente: ajuste robusto de plano/campo suave pelos pixels livres de glyph;
- textura/translucidez: inpaint multiescala condicionado às bordas e contexto do interior;
- card/UI: preservar moldura, ícones e separadores detectados, reconstruindo apenas a área textual interna.

Depois, renderizar uma única camada PT-BR. O algoritmo escolhe a classe pela evidência visual; não contém regra por obra, página, cor ou coordenada.

Se o R3 contextual ainda deixar suporte-fonte confirmado, a mesma estratégia executa a variante `deterministic_interior_fill`: estima a cor/campo-base pelos pixels livres de glyph, preenche todo o interior seguro (não somente a máscara OCR) e reaplica o target com estilo-base. Essa variante sacrifica detalhe decorativo interno antes de aceitar inglês, mas preserva borda e pixels externos. Pixels de glyph confirmados não podem ser tratados como arte protegida apenas porque há arte subjacente; já ícones/personagens/separadores genuínos fora do suporte-fonte permanecem hard-protected. Se a linha residual estiver fora do interior calculado, o controller amplia/corrige o container ou materializa outro owner. Esgotado o orçamento de geometria visual, use `deterministic_support_local_fill`: union do suporte glyph confirmado + margem determinística, hard-clipped à página e ao menor container conservador executável, reconstrução local e reaplicação do target. Nunca repita indefinidamente a mesma máscara.

**Step 4: Definir a política terminal sem inglês**

- R0/R1/R2 falharam: executar R3 automaticamente.
- R3 encontra proteção conservadora sobre glyph source confirmado: recalcular interior/proteção e executar; encontra arte genuína fora do suporte que seria alterada: refinar a reconstrução preservando-a. Uma imagem difícil, container ausente ou conflito visual não pode lançar erro terminal: após tentativas limitadas, executar `deterministic_support_local_fill` na região confirmada. `UnsafeProtectedArtGeometryError` fica restrito a contrato estrutural inválido (shape/dtype/bounds impossíveis, hash/bytes corrompidos), nunca à ambiguidade de conteúdo ou à ausência de uma borda ideal.
- R3 contextual ainda tem source: executar `deterministic_interior_fill` e provar cobertura integral do interior; uma nova linha fora dele reabre coverage/container de forma monotônica.
- Falha técnica transitória: retry da mesma página a partir dos pixels originais.
- Erro operacional irrecuperável (arquivo ilegível, memória esgotada, modelo ausente): abortar sem exportar imagem; nunca marcar o conteúdo como aprovado ou publicar source.
- Não existe terminal de conteúdo `review_required`, `source_preserved` ou `with_warnings` para diálogo traduzível.

**Step 5: Confirmar GREEN**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest `
    tests/test_owner_repair.py `
    tests/test_owner_atomic_execution.py `
    tests/test_owner_lifecycle.py `
    tests/test_strip_owner_composition_integration.py `
    -q
} finally { Pop-Location }
```

Expected: PASS; toda falha de conteúdo sintética termina com source removido e PT-BR materializado.

**Step 6: Checkpoint D — transação e ladder R0–R3**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest `
    tests/test_owner_atomic_execution.py `
    tests/test_owner_source_replacement_fail_closed.py `
    tests/test_owner_repair.py `
    tests/test_owner_mask.py `
    tests/test_inpaint_debug_residual.py `
    tests/test_typesetting_renderer.py `
    tests/test_strip_owner_composition_integration.py `
    -q
} finally { Pop-Location }
```

Aceite somente se a matriz de falhas chegar a `committed` sem inglês e os testes de proteção de borda/arte permanecerem verdes.

**Step 7: Commit**

```powershell
git add -p -- pipeline/ownership/repair.py pipeline/ownership/execution.py pipeline/inpainter/owner_mask.py pipeline/qa/inpaint_residual.py pipeline/strip/page_pipeline.py pipeline/tests/test_owner_repair.py pipeline/tests/test_owner_atomic_execution.py pipeline/tests/test_owner_lifecycle.py pipeline/tests/test_strip_owner_composition_integration.py
git diff --cached --check
git commit -m "feat: guarantee container interior recovery as final fallback"
```

### Task 14: Tornar o QA final consciente de idioma, owner e região

**Files:**

- Create: `pipeline/qa/language_residual.py`
- Modify: `pipeline/qa/final_pixel_qa.py`
- Modify: `pipeline/qa/final_pixel_observer.py`
- Modify: `pipeline/vision_stack/runtime.py`
- Create: `pipeline/tests/test_language_residual.py`
- Test: `pipeline/tests/test_final_pixel_qa.py`
- Test: `pipeline/tests/test_final_pixel_observer.py`

**Step 1: Escrever testes RED do classificador final**

```python
def test_visible_english_dialogue_is_source_residual_by_language_and_binding():
    issues = classify_language_residual(
        observed="THE ARENA WILL BEGIN",
        binding=_binding(source="THE ARENA WILL BEGIN", target="A ARENA COMEÇARÁ"),
        region=_dialogue_region(),
    )
    assert [issue.kind for issue in issues] == ["source_language_visible"]
    assert issues[0].repair_required


def test_shared_tokens_proper_names_and_numbers_are_not_false_residuals():
    issues = classify_language_residual(
        observed="KIM SIMUN TEM 10 ABATES",
        binding=_binding(
            source="KIM SIMUN HAS 10 KILLS",
            target="KIM SIMUN TEM 10 ABATES",
            entities=("KIM SIMUN",),
        ),
        region=_card_region(),
    )
    assert issues == ()


def test_text_inside_translatable_container_without_owner_is_coverage_issue():
    issues = classify_unowned_text(_ocr_line("WAIT..."), _dialogue_container())
    assert [issue.kind for issue in issues] == ["independently_detected_text_without_owner"]
    assert issues[0].repair_required


def test_material_english_line_without_component_or_container_is_coverage_issue():
    issues = classify_unowned_text(
        _full_page_ocr_line("SOMETHING MOVED", glyph_support=True),
        container=None,
    )
    assert [issue.kind for issue in issues] == ["independently_detected_text_without_owner"]
    assert issues[0].component_id is None
    assert issues[0].container_id is None


def test_alias_probes_from_same_recognition_run_emit_one_issue():
    issues = deduplicate_language_issues(_same_line_from_three_wrappers())
    assert len(issues) == 1


def test_independent_invocations_for_same_region_and_binding_emit_one_repair_issue():
    issues = deduplicate_language_issues(_same_owner_line_from_two_invocations())
    assert len(issues) == 1
    assert set(issues[0].invocation_ids) == {"ocr-final", "detector-challenge"}


def test_incomplete_changed_mask_emits_cleanup_repair_even_if_ocr_is_inconclusive():
    issue = verify_replacement_pixels(_commit_with_uncovered_source_support())
    assert issue.kind == "cleanup_incomplete"
    assert issue.repair_required


def test_missing_target_glyph_mask_emits_rerender_request():
    issue = verify_replacement_pixels(_commit_without_materialized_target_patch())
    assert issue.kind == "target_glyphs_missing"
    assert issue.repair_required


def test_final_observer_rejects_root_hash_from_other_persisted_candidate(tmp_path):
    candidate_a = _persist_candidate(tmp_path, "a.png", _candidate_a_pixels())
    observer = _observer_returning_request_root(canonical_page_sha256(_candidate_b_pixels()))
    with pytest.raises(TerminalVerificationIdentityError):
        observe_persisted_candidate(candidate_a, observer)


def test_final_observer_records_exact_native_gray_inverted_and_2x_inputs(tmp_path):
    candidate = _persist_candidate(tmp_path, "candidate.png", _asymmetric_rgb_page())
    probe, captured = _observe_with_all_variants_and_capture(candidate)
    attempts = {attempt.variant_id: attempt for attempt in probe.ocr_invocation.attempts}
    for variant in ("native", "gray", "inverted", "scale_2x"):
        assert attempts[variant].input_pixel_sha256 == canonical_page_sha256(captured[variant])
        assert attempts[variant].provider_called is True


def test_final_observer_requires_uncached_full_page_physical_attempt_with_empty_blocks(tmp_path):
    candidate = _persist_candidate(tmp_path)
    probe = observe_persisted_candidate(candidate, _provider_with_empty_detector_blocks())
    full_page = _attempt(probe.ocr_invocation, variant_id="full_page")
    assert full_page.provider_called is True
    assert full_page.cache_hit is False
    assert probe.ocr_invocation.request.root_input_pixel_sha256 == candidate.pixel_sha256
    assert probe.coverage_complete is True
```

**Step 2: Confirmar RED**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest `
    tests/test_language_residual.py `
    tests/test_final_pixel_qa.py `
    tests/test_final_pixel_observer.py `
    -q
} finally { Pop-Location }
```

Expected: QA atual depende de overlap bruto de tokens e pode duplicar detector/component probe.

**Step 3: Implementar `LanguageResidualIssue`**

Campos mínimos:

```python
@dataclass(frozen=True)
class LanguageResidualIssue:
    issue_id: str
    run_id: str
    page_id: str
    page_source_sha256: str
    page_output_pixel_sha256: str
    owner_id: str | None
    component_id: str | None
    container_id: str | None
    invocation_ids: tuple[str, ...]
    kind: Literal[
        "source_language_visible",
        "independently_detected_text_without_owner",
        "target_payload_missing",
        "cleanup_incomplete",
        "target_glyphs_missing",
        "mixed_language_overlay",
    ]
    observed_text: str
    source_only_tokens: tuple[str, ...]
    repair_required: bool
```

Classificar pelo `TranslationBinding`, role e policies explícitas. A identidade funcional da issue é página/hash final + owner ou região canônica + kind + hash do source binding. `invocation_id`s, providers e textos observados são evidências acumuladas em tuplas, não parte da chave; duas invocações independentes sobre o mesmo defeito geram um único reparo.

**Step 4: Executar probe realmente page-global**

O observador final deve chamar a API OCR request-scoped full-page mesmo quando detector/challenges estiverem vazios. Em seguida:

- construir `OCRRequest.root_input_pixel_sha256` a partir dos pixels RGB do candidate lossless reaberto, nunca copiar `page_source_sha256` por conveniência;
- exigir pelo menos um `OCRAttempt` full-page `provider_called=true`, `cache_hit=false`, e encaminhar nativa/cinza/invertida/2x pelo mesmo boundary hash-bound; mismatch root/físico é erro de infraestrutura, nunca resultado OCR negativo;
- associar linhas a containers/owners pelo ledger final;
- criar issue para texto alfabético material dentro de container traduzível sem owner;
- criar a mesma issue para linha full-page material sem component **e** sem container; o glyph support positivo basta para reabrir coverage, e ausência de container nunca filtra a linha;
- comparar source-only tokens com target esperado;
- verificar ausência do target sem exigir leitura OCR perfeita de cada glyph;
- emitir `cleanup_incomplete` quando action/changed mask não cobrir todo o suporte source conhecido;
- emitir `target_glyphs_missing` quando o patch/máscara target não estiver materializado na cadeia final;
- registrar cobertura, inclusive regiões negativas.

**Step 5: Confirmar GREEN**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest `
    tests/test_language_residual.py `
    tests/test_final_pixel_qa.py `
    tests/test_final_pixel_observer.py `
    tests/test_vision_stack_runtime.py `
    -q
} finally { Pop-Location }
```

Expected: PASS; nomes/números compartilhados não geram falso positivo e inglês real gera uma issue canônica.

**Step 6: Commit**

```powershell
git add -- pipeline/qa/language_residual.py pipeline/tests/test_language_residual.py
git add -p -- pipeline/qa/final_pixel_qa.py pipeline/qa/final_pixel_observer.py pipeline/vision_stack/runtime.py pipeline/tests/test_final_pixel_qa.py pipeline/tests/test_final_pixel_observer.py
git diff --cached --check
git commit -m "feat: detect final source language by owner binding"
```

### Task 15: Fazer o QA reabrir a página e dirigir o reparo até prova terminal

**Files:**

- Modify: `pipeline/ownership/repair.py`
- Modify: `pipeline/ownership/execution.py`
- Modify: `pipeline/ownership/lifecycle.py`
- Modify: `pipeline/strip/page_pipeline.py`
- Modify: `pipeline/qa/final_pixel_qa.py`
- Modify: `pipeline/qa/final_pixel_observer.py`
- Re-run only — do not edit/stage: `pipeline/main.py` (boundary de publicação definido na Task 9)
- Create: `pipeline/tests/test_owner_repair_controller.py`
- Modify: `pipeline/tests/test_owner_lifecycle_e2e.py`
- Test: `pipeline/tests/test_final_pixel_qa.py`
- Re-run only — do not edit/stage: `pipeline/tests/test_final_pixel_observer.py`
- Re-run only — do not edit/stage: `pipeline/tests/test_main_emit.py`

**Step 1: Escrever testes RED do loop de reparo**

```python
def test_final_source_residual_reopens_same_owner_from_original_page():
    result = run_page_owner_pipeline(_page_with_first_pass_residual(), _services())
    assert result.repair_history
    assert result.repair_history[-1].owner_id == _owner_id()
    assert {a.input_sha256 for a in result.repair_history} == {_source_page_sha256()}
    assert result.lifecycle.entry(_component_id()).state == "final_verified"


def test_final_unowned_text_reopens_coverage_and_creates_owner():
    result = run_page_owner_pipeline(_page_where_final_probe_finds_missed_text(), _services())
    assert result.coverage.entry_for_bbox(_missed_bbox()).owner_id
    assert _final_text(result) == _expected_ptbr()


def test_final_english_without_component_or_container_materializes_both_then_ptbr():
    result = run_page_owner_pipeline(_page_with_final_uncontained_english_line(), _services())
    recovered = result.coverage.entry_for_bbox(_missed_bbox())
    assert recovered.component_id
    assert recovered.container_id
    assert recovered.owner_id
    assert recovered.state == "final_verified"
    assert _source_english_visible(result.final_page) is False


def test_unassociated_missing_container_text_over_protected_art_finishes_final_verified():
    result = run_page_owner_pipeline(
        _page_with_unassociated_missing_container_text_over_protected_art(),
        _services(),
    )
    recovered = result.coverage.entry_for_bbox(_protected_text_bbox())
    assert recovered.protection_conflict
    assert recovered.component_id and recovered.container_id and recovered.owner_id
    assert recovered.state == "final_verified"
    assert result.status == "final_verified"
    assert _source_english_visible(result.final_page) is False
    assert _target_ptbr_visible(result.final_page) is True


def test_missing_target_reuses_binding_and_rerenders_without_retranslation():
    result = run_page_owner_pipeline(_page_with_missing_target_first_pass(), _services())
    assert len(result.translations) == 1
    assert len(result.translations[0].attempt_ids) == 1
    assert result.repair_history
    assert {attempt.translation_binding_sha256 for attempt in result.repair_history} == {
        result.translations[0].translation_binding_sha256
    }
    assert result.lifecycle.entry(_component_id()).state == "final_verified"


def test_terminal_proof_uses_pixels_and_fresh_ocr_without_infinite_retry():
    result = run_page_owner_pipeline(_page_with_ocr_target_false_negative(), _services())
    assert result.terminal_proof.source_support_removed
    assert result.terminal_proof.target_glyph_patch_applied
    assert result.terminal_proof.fresh_source_ocr_absent
    assert result.status == "final_verified"


@pytest.mark.parametrize("observer_state", ["unavailable", "coverage_incomplete", "identity_mismatch"])
def test_terminal_proof_rejects_unavailable_incomplete_or_stale_fresh_ocr(observer_state):
    with pytest.raises(TerminalVerificationInfrastructureError):
        build_terminal_pixel_proof(_commit(), _fresh_ocr(state=observer_state))


def test_terminal_proof_rejects_fresh_ocr_from_different_candidate_pixels(tmp_path):
    candidate_a = _persist_candidate(tmp_path, name="a.png", pixels=_candidate_a_pixels())
    candidate_b = _persist_candidate(tmp_path, name="b.png", pixels=_candidate_b_pixels())
    fresh_ocr_b = _fresh_ocr(
        run_id=candidate_a.run_id,
        page_id=candidate_a.page_id,
        page_source_sha256=candidate_a.page_source_sha256,
        root_input_pixel_sha256=candidate_b.pixel_sha256,
        source_lines=(),
    )
    with pytest.raises(TerminalVerificationIdentityError):
        TerminalPixelProof.build_from_persisted_candidate(candidate_a, fresh_ocr_b)


@pytest.mark.parametrize("tamper", ["attempt_input_hash", "parent_hash", "transform_spec_payload", "transform_spec_hash"])
def test_terminal_proof_rejects_tampered_physical_attempt_chain(tamper, tmp_path):
    candidate = _persist_candidate(tmp_path)
    fresh_ocr = _fresh_ocr_for(
        candidate,
        physical_variants=("full_page", "gray", "inverted", "scale_2x", "rotate_affine", "deskew_affine"),
    )
    with pytest.raises(TerminalVerificationIdentityError):
        TerminalPixelProof.build_from_persisted_candidate(candidate, _tamper_attempt(fresh_ocr, tamper))


def test_terminal_proof_recomputes_variant_hashes_from_redecoded_candidate(tmp_path):
    candidate = _persist_candidate(tmp_path)
    proof = TerminalPixelProof.build_from_persisted_candidate(
        candidate,
        _fresh_ocr_for(
            candidate,
            physical_variants=("full_page", "gray", "inverted", "scale_2x", "rotate_affine", "deskew_affine"),
        ),
    )
    assert proof.fresh_ocr_root_input_pixel_sha256 == candidate.pixel_sha256
    assert proof.fresh_ocr_attempt_chain_sha256 == _recomputed_attempt_chain_from(candidate)


@pytest.mark.parametrize("invalid_probe", ["cache_only", "missing_full_page", "provider_not_called"])
def test_terminal_proof_rejects_nonphysical_or_incomplete_fresh_probe(invalid_probe, tmp_path):
    candidate = _persist_candidate(tmp_path)
    with pytest.raises(TerminalVerificationInfrastructureError):
        TerminalPixelProof.build_from_persisted_candidate(candidate, _fresh_ocr(invalid_probe))


@pytest.mark.parametrize(
    "field",
    [
        "translation_binding_sha256s",
        "source_payload_sha256s",
        "target_payload_sha256s",
        "target_glyph_patch_sha256s",
        "target_materialization_sha256s",
    ],
)
def test_terminal_proof_rejects_stale_translation_or_materialization_chain(field, tmp_path):
    candidate = _persist_candidate(tmp_path)
    composition = _verified_page_composition()
    tampered = dataclasses.replace(composition, **{field: ("stale",)})
    with pytest.raises(TerminalVerificationIdentityError):
        TerminalPixelProof.build_from_persisted_candidate(
            candidate,
            _fresh_ocr_for(candidate),
            page_composition=tampered,
            bindings=_bindings(),
            materializations=_materializations(),
        )


def test_each_repair_cycle_reopens_persisted_lossless_candidate_before_fresh_qa(tmp_path):
    result = run_page_owner_pipeline(
        _page_requiring_two_repairs(),
        _services_recording_probe_inputs(),
        candidate_dir=tmp_path,
    )
    assert result.status == "final_verified"
    assert all(probe.input_origin == "redecoded_persisted_candidate" for probe in result.qa_probes)
    assert all(probe.root_input_pixel_sha256 == probe.candidate_pixel_sha256 for probe in result.qa_probes)
    assert all(probe.candidate_file_sha256 == _persisted_candidate_file_hash_for(probe) for probe in result.qa_probes)
    assert all(probe.fresh_ocr_attempt_chain_sha256 for probe in result.qa_probes)


def test_only_production_builder_can_promote_candidate_to_final_verified(tmp_path):
    candidate = _persist_candidate(tmp_path)
    result = verify_and_promote_candidate(candidate, _fresh_terminal_services())
    assert result.status == "final_verified"
    assert result.final_page is not None
    assert result.terminal_proof is not None
    assert result.terminal_proof.final_page_pixel_sha256 == result.final_page.page_output_pixel_sha256
    assert result.qa_probes
    assert result.final_replacement_verdicts


def test_transient_or_incoherent_result_is_never_persisted_or_exported(tmp_path):
    for result in (_candidate_page_result(), _result_with_tampered_terminal_hash()):
        with pytest.raises((PageNotTerminalError, PagePipelineIdentityError)):
            persist_page_execution_result(result, tmp_path)
        assert not tmp_path.joinpath("pages", "page_001", "final.png").exists()
        assert not tmp_path.joinpath("pages", "page_001", "execution_result.json").exists()
        assert not tmp_path.joinpath("project.json").exists()
        assert not tmp_path.joinpath("export_manifest.json").exists()
        assert not tmp_path.joinpath("translated").exists()
        assert not _page_staging_dir(tmp_path, "page_001").exists()


def test_crash_recovery_removes_only_orphan_candidate_staging_and_never_publishes_it(tmp_path):
    orphan = _create_orphan_candidate(tmp_path, run_id="run-current", page_id="page_001")
    recover_owner_staging(tmp_path, run_id="run-current")
    assert not orphan.exists()
    assert not tmp_path.joinpath("pages", "page_001").exists()
    assert not tmp_path.joinpath("translated").exists()
```

**Step 2: Confirmar RED**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest `
    tests/test_owner_repair_controller.py `
    tests/test_owner_lifecycle_e2e.py `
    tests/test_final_pixel_qa.py `
    tests/test_final_pixel_observer.py `
    tests/test_main_emit.py::MainEmitTests::test_real_run_pipeline_second_page_failure_publishes_no_chapter_outputs `
    -q
} finally { Pop-Location }
```

Expected: QA apenas relata/bloqueia e não retorna a página ao lifecycle.

**Step 3: Implementar `PageRepairController`**

Para cada issue canônica:

- `source_language_visible`: reabrir o owner e avançar a estratégia;
- `target_payload_missing`: reutilizar o binding e rerenderizar; se cleanup também falhou, avançar reparo completo;
- `cleanup_incomplete`: avançar obrigatoriamente R0 → R1 → R2 → R3 usando os pixels originais;
- `target_glyphs_missing`: reaplicar o glyph patch do binding; se a base/hash divergir, refazer a transação completa;
- `independently_detected_text_without_owner`: reabrir coverage, materializar componente, criar owner, traduzir e executar;
- `mixed_language_overlay`: iniciar no mínimo em R2;
- após o R3 contextual, se o probe fresco ainda localizar suporte source material, executar a variante determinística de preenchimento integral do interior seguro e render base; se a detecção estiver fora do interior, reabrir coverage/container. Esgotado o orçamento monotônico de expansão, materializar o container support-local hard-clipped e executar `deterministic_support_local_fill`, sem repetir o mesmo estado nem abortar por dificuldade visual.

O controller processa todos os issues de uma página em uma transação de página, para que reparar um owner não restaure outro.

Defina `FinalQAProbe` frozen com `probe_id`, identidade run/page/source, `input_origin: Literal["redecoded_persisted_candidate"]`, `candidate_file_sha256` (bytes PNG), `candidate_pixel_sha256` (RGB decodificado), `root_input_pixel_sha256` (root efetivamente entregue à operação), `ocr_invocation_id`, `fresh_ocr_attempt_ids`, `fresh_ocr_attempt_chain_sha256`, `ocr_payload_sha256`, `observer_available`, `coverage_complete` e `issue_ids`. File hash e pixel hash nunca são comparados entre si; exige-se `root_input_pixel_sha256 == candidate_pixel_sha256`, e cada tentativa física é validada pela cadeia da única invocation referenciada. Defina `FinalReplacementVerdict` frozen por owner com `verdict_id`, identidade, `translation_binding_sha256`, `source_payload_sha256`, `target_payload_sha256`, `target_glyph_patch_sha256`, `target_materialization_sha256`, `source_support_removed`, `target_materialized` e `status: Literal["final_verified"]`. Estenda `PageExecutionResult` com `qa_probes: tuple[FinalQAProbe, ...]` e `final_replacement_verdicts: tuple[FinalReplacementVerdict, ...]`; a promoção terminal exige exatamente um verdict por binding e ao menos um probe fresco ligado ao candidate final.

**Step 4: Implementar o builder de produção de `TerminalPixelProof` e a promoção terminal**

Exigir conjuntamente:

1. máscara source esperada coberta pela ação de cleanup/rebuild;
2. diferença material nos pixels dessa máscara contra o original;
3. patch de glyphs PT-BR aplicado dentro da região de render;
4. cadeia de hashes cleanup → render → página final coerente;
5. conjuntos ordenados de `translation_binding_sha256`, source/target payload hashes, glyph-patch hashes e materialization hashes idênticos em bindings → commits → `PageCompositionSnapshot` → verdicts → proof;
6. OCR fresco com `observer_available=true`, `coverage_complete=true`, nova `invocation_id`, cache desabilitado, tentativa full-page física obrigatória, a mesma identidade run/page/source e `OCRRequest.root_input_pixel_sha256` exatamente igual ao hash RGB do `candidate.png` redecodificado; cada `OCRAttempt` é recalculado/ligado ao transform e aos pixels físicos, sem source-only tokens materiais;
7. ledger sem texto material não possuído.

O target OCR pode ser incompleto se 1–4 provarem materialização e 5 for negativo; isso evita loop por fonte estilizada que o OCR não lê. OCR source positivo sempre reabre o reparo.

Implemente `TerminalPixelProof.build_from_persisted_candidate(...)` como a única fábrica usada em produção; a construção direta permanece restrita a fixtures. Ela reabre o candidate, recalcula `fresh_ocr_root_input_pixel_sha256`, `fresh_ocr_attempt_chain_sha256`, `fresh_ocr_payload_sha256` e `proof_sha256`, valida a identidade completa e exige `fresh_ocr.request.root_input_pixel_sha256 == candidate.pixel_sha256 == final_page_pixel_sha256`. Para cada attempt, reexecuta deterministicamente bbox/transform sobre o candidate redecodificado e compara parent/input/transform hashes, shape/mode e `provider_called=true`; exige ao menos full-page físico e rejeita prova baseada só em cache/heurística. Também compara, sem rederivar de texto solto, todos os hashes ordenados de binding/source/target/glyph/materialization contra commits, `PageCompositionSnapshot` e verdicts, além de ligar `cleanup_base_sha256` e `composition_sha256` aos artefatos reais. Depois, `PageExecutionResult.promote_final(...)` cria atomicamente o par coerente `status="final_verified"` + `final_page` + `terminal_proof`; nunca altere um resultado candidato no lugar.

Após cada composição candidata, grave PNG lossless em `<run>/.staging/<run_id>/<page_id>/<attempt_id>/candidate.tmp.png`, faça rename atômico para `candidate.png` **dentro desse staging não publicável**, reabra/decodifique os bytes persistidos, recalcule o hash e rode o observer/`TerminalPixelProof` sobre essa decodificação — nunca apenas sobre o ndarray em memória. Só depois de `final_verified`, promova atomicamente os bytes hash-verificados para `pages/<page_id>/final.png` e então persista `execution_result.json`. Essa promoção page-local **não** publica o capítulo: somente o boundary real da Task 9 pode criar `translated/`, `project.json` e `export_manifest.json`, após preflight de todas as páginas. `candidate_ready`, `repair_pending`, prova ausente/incoerente, observer indisponível, coverage incompleta ou identidade stale são estados não exportáveis. Em falha, remova somente o attempt staging resolvido/validado dentro da raiz da run; na inicialização, `recover_owner_staging()` remove órfãos daquela run sem tocar runs anteriores nem diretórios publicáveis. Erros operacionais abortam sem export, não contam como OCR negativo e não preservam conteúdo source como saída.

**Step 5: Confirmar GREEN**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest `
    tests/test_owner_repair_controller.py `
    tests/test_owner_lifecycle_e2e.py `
    tests/test_owner_repair.py `
    tests/test_final_pixel_qa.py `
    tests/test_final_pixel_observer.py `
    tests/test_main_emit.py::MainEmitTests::test_real_run_pipeline_second_page_failure_publishes_no_chapter_outputs `
    -q
} finally { Pop-Location }
```

Expected: PASS; QA corrige em vez de apenas bloquear e somente `final_verified` sai do coordenador.

**Step 6: Commit**

```powershell
git add -- pipeline/tests/test_owner_repair_controller.py pipeline/tests/test_owner_lifecycle_e2e.py
git add -p -- pipeline/ownership/repair.py pipeline/ownership/execution.py pipeline/ownership/lifecycle.py pipeline/strip/page_pipeline.py pipeline/qa/final_pixel_qa.py pipeline/qa/final_pixel_observer.py pipeline/tests/test_final_pixel_qa.py
git diff --cached --check
git commit -m "feat: repair final language residuals before export"
```

### Task 16: Persistir artefatos completos e impedir evidência visual duplicada

**Files:**

- Modify: `pipeline/main.py`
- Modify: `pipeline/qa/export_gate.py`
- Modify: `pipeline/strip/page_pipeline.py`
- Modify: `pipeline/tools/validate_owner_visual_matrix.py`
- Modify: `pipeline/tests/test_final_pixel_export_gate.py`
- Modify: `pipeline/tests/test_owner_visual_matrix_tool.py`
- Create: `pipeline/tests/test_page_owner_artifacts.py`

**Step 1: Escrever testes RED de persistência/gate**

```python
def test_page_artifacts_share_run_page_and_hash_chain():
    artifacts = _write_page_artifacts(_verified_result())
    assert artifacts.coverage.page_id == artifacts.execution.page_id == artifacts.final_qa.page_id
    assert artifacts.execution.final_pixel_sha256 == artifacts.final_qa.page_output_pixel_sha256
    assert artifacts.final_page.final_file_sha256 == sha256_bytes(artifacts.final_page.lossless_png_bytes)
    assert artifacts.execution.cleanup_base_sha256 == canonical_page_sha256(
        decode_lossless(artifacts.cleanup_base_png_bytes)
    )


def test_every_repair_request_target_materialization_and_commit_is_accounted_for():
    artifacts = _write_page_artifacts(_verified_result_with_repair())
    assert _request_ids(artifacts.repair_requests) == _consumed_request_ids(artifacts.repair_attempts)
    assert _issue_request_pairs(artifacts.repair_requests) == _issue_attempt_pairs(artifacts.repair_attempts)
    assert _binding_hash_map(artifacts.translation_bindings) == _binding_hash_map(
        artifacts.owner_target_materialization
    )
    assert _source_hash_map(artifacts.translation_bindings) == _source_hash_map(
        artifacts.owner_target_materialization
    ) == _source_hash_map(artifacts.execution) == _source_hash_map(artifacts.page_composition)
    assert _target_hash_map(artifacts.translation_bindings) == _target_hash_map(
        artifacts.owner_target_materialization
    ) == _target_hash_map(artifacts.execution) == _target_hash_map(artifacts.page_composition)
    assert _glyph_hash_map(artifacts.execution) == _glyph_hash_map(
        artifacts.owner_target_materialization
    )
    assert _commit_ids(artifacts.execution) == _commit_ids(artifacts.page_composition)
    assert _owner_ids(artifacts.final_replacement_verdicts) == _translatable_owner_ids(artifacts.coverage)


def test_same_owner_with_stale_target_binding_or_materialization_hash_is_rejected():
    result = _verified_result_with_same_owner_but_stale_target_hash()
    with pytest.raises(PageArtifactIntegrityError):
        _write_page_artifacts(result)


def test_artifact_writer_serializes_only_evidence_owned_by_page_execution_result():
    result = _verified_result()
    artifacts = _write_page_artifacts(result)
    assert artifacts.ocr_requests == result.ocr_requests
    assert artifacts.ocr_invocations == result.ocr_invocations
    assert artifacts.repair_requests == result.repair_requests
    assert artifacts.repair_attempts == result.repair_history
    assert artifacts.owner_target_materialization == result.owner_target_materializations
    assert artifacts.page_composition.sha256 == result.page_composition.sha256
    assert artifacts.final_qa_probes == result.qa_probes
    assert artifacts.final_replacement_verdicts == result.final_replacement_verdicts


def test_artifacts_preserve_root_and_every_physical_ocr_attempt_hash():
    result = _verified_result_with_full_page_crop_and_variants()
    artifacts = _write_page_artifacts(result)
    for invocation in artifacts.ocr_invocations:
        assert invocation.request.root_input_pixel_sha256
        assert invocation.attempts
        assert all(attempt.input_pixel_sha256 for attempt in invocation.attempts)
        assert all(attempt.transform_spec.canonical_json_bytes for attempt in invocation.attempts)
        assert all(
            attempt.transform_spec.sha256 == sha256_bytes(attempt.transform_spec.canonical_json_bytes)
            for attempt in invocation.attempts
        )
        assert _records_reference_exact_attempts(invocation)
    assert artifacts.terminal_proof.fresh_ocr_attempt_chain_sha256 == _attempt_chain(
        _terminal_invocation(artifacts)
    )


def test_every_ledger_and_graph_observation_resolves_to_one_atomic_ocr_invocation():
    artifacts = _write_page_artifacts(_verified_result())
    observations = _index_by_id(artifacts.page_owner_observations)
    for observation_id in _all_observation_ids(artifacts.coverage, artifacts.owner_graph):
        matches = observations[observation_id]
        assert len(matches) == 1
        assert _identity(matches[0]) == _page_identity(artifacts.coverage)
        assert matches[0].invocation_id in _invocation_ids(artifacts.ocr_invocations)


def test_export_gate_accepts_only_final_verified_page_results():
    assert evaluate_export_gate(_chapter_with_all_pages_verified())["status"] == "PASS"
    blocked = evaluate_export_gate(_chapter_with_repair_pending_page())
    assert blocked["status"] == "BLOCK"
    assert "page_lifecycle_incomplete" in _reasons(blocked)


def test_visual_matrix_canonicalizes_identical_pixels_into_one_review_item_with_all_categories():
    report = validate_matrix(_manifest_reusing_same_output_for_three_cases())
    assert len(report.review_items) == 1
    assert set(report.review_items[0].categories) == {"category_a", "category_b", "category_c"}


def test_visual_matrix_rejects_same_hash_claimed_as_distinct_output_artifacts():
    report = validate_matrix(_manifest_claiming_distinct_outputs_with_one_reused_file())
    assert report.status == "INVALID_EVIDENCE"
    assert report.conflicting_output_claims


def test_inspection_verdict_cannot_be_autofilled_from_template_note():
    with pytest.raises(InspectionEvidenceError):
        load_inspection(_templated_repeated_verdicts_without_review_metadata())
```

**Step 2: Confirmar RED**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest `
    tests/test_page_owner_artifacts.py `
    tests/test_final_pixel_export_gate.py `
    tests/test_owner_visual_matrix_tool.py `
    -q
} finally { Pop-Location }
```

Expected: artefatos podem divergir e a matriz aceita o mesmo hash como casos visuais diferentes.

**Step 3: Definir artefatos canônicos por página**

Nesta altura `PageExecutionResult` é o journal frozen completo: `ocr_requests`, `ocr_invocations`, `coverage`/lifecycle, `owner_graph`, `translations`, `repair_requests`, `repair_history`, `page_commits`, `owner_target_materializations`, `page_composition`, `qa_probes`, `final_replacement_verdicts`, `final_page` e `terminal_proof`. `_write_page_artifacts(result)` aceita somente esse objeto `final_verified`; não recebe side inputs, não consulta debug global/bands e não reconstrói evidência ausente. Campo obrigatório ausente ou cadeia divergente é `PageArtifactIntegrityError` antes de qualquer publicação.

Persistir em diretório da run, sem incluir outputs no Git:

```text
page_###/
  coverage_ledger.json
  ocr_requests.jsonl
  ocr_invocations.jsonl
  page_owner_observations.jsonl
  owner_graph.json
  translation_bindings.json
  repair_requests.jsonl
  repair_attempts.jsonl
  owner_target_materialization.jsonl
  execution_result.json
  cleanup_base.png
  page_composition.json
  final_qa_probes.jsonl
  final_pixel_proof.json
  final_replacement_verdicts.jsonl
  final.png
```

Todos carregam `run_id`, `page_id`, `page_source_sha256`, versão do schema e hashes de entrada/saída. OCR requests persistem `root_input_pixel_sha256`; invocations persistem cada `OCRAttempt` físico com attempt/parent/input hashes, `OCRTransformSpec.canonical_json_bytes` + hash, bbox, shape/mode, `provider_called` e `cache_hit`; observations referenciam o attempt exato. Persistir apenas o hash do transform sem seu spec reproduzível é inválido. No probe terminal, o root coincide com os pixels de `final.png` e `fresh_ocr_attempt_chain_sha256` cobre a sequência física ordenada. `repair_requests` liga cada `issue_id/request_id` ao attempt que a consumiu; `owner_target_materialization` prova exatamente um glyph patch final por binding e repete os hashes exatos de binding/target/glyph; `cleanup_base.png` é a composição page-space de todos os cleanups verificados antes dos glyph patches, com `cleanup_base_sha256` no execution result; `page_composition` é o `PageCompositionSnapshot` autoritativo e liga todos os commits/hashes à página; `final_replacement_verdicts` fecha cada owner com os mesmos hashes. Escrita deve ser temporária + rename atômico para não deixar JSON parcial. A base limpa é artefato interno de replay, nunca export final isolado.

**Step 4: Integrar com o gate**

O export gate não decide reparo. Ele valida que todas as páginas recebidas já estão `final_verified` e que:

```text
english_dialogue_residual_count == 0
translatable_components_without_owner == 0
material_components_without_ocr_attempt == 0
owners_without_valid_pt_br == 0
owners_without_atomic_cleanup_render == 0
owners_without_target_materialization == 0
material_components_without_terminal_lifecycle == 0
final_pixel_ocr.coverage_complete == true
fresh_ocr_root_hash_matches_final == true
unverified_physical_ocr_attempts == 0
fresh_ocr_cache_hits == 0
```

Se o coordenador ainda estiver `repair_pending`, não gerar export intermediário `with_warnings`; continue o controller. Erro operacional aborta a run sem publicar capítulo parcial.

**Step 5: Corrigir integridade da matriz visual**

- calcular SHA-256 dos arquivos realmente inspecionados;
- canonicalizar o mesmo artefato/hash referenciado por várias categorias em um único item de inspeção com todas as categorias preservadas;
- rejeitar hash repetido somente quando o manifest/relatório alega que são outputs distintos ou duplica verdicts/notas para inflar cobertura;
- exigir `reviewed_at`, `reviewer`, `source_path`, `output_path`, dimensões e hashes;
- não gerar `GO` nem notas de inspeção automaticamente;
- distinguir teste funcional automático de veredito visual humano/agentivo.

**Step 6: Confirmar GREEN**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest `
    tests/test_page_owner_artifacts.py `
    tests/test_final_pixel_export_gate.py `
    tests/test_owner_visual_matrix_tool.py `
    -q
} finally { Pop-Location }
```

Expected: PASS; hashes e identidades atravessam os artefatos e a matriz não mascara duplicatas.

**Step 7: Checkpoint E — contrato de saída**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest `
    tests/test_page_owner_artifacts.py `
    tests/test_final_pixel_qa.py `
    tests/test_final_pixel_observer.py `
    tests/test_final_pixel_export_gate.py `
    tests/test_owner_visual_matrix_tool.py `
    -q
} finally { Pop-Location }
```

Aceite somente se o gate receber páginas verificadas, não houver hashes visuais reutilizados e nenhum verdict for preenchido automaticamente.

**Step 8: Commit**

```powershell
git add -- pipeline/tests/test_page_owner_artifacts.py
git add -p -- pipeline/main.py pipeline/qa/export_gate.py pipeline/strip/page_pipeline.py pipeline/tools/validate_owner_visual_matrix.py pipeline/tests/test_final_pixel_export_gate.py pipeline/tests/test_owner_visual_matrix_tool.py
git diff --cached --check
git commit -m "feat: persist verifiable page replacement artifacts"
```

### Task 17: Tornar `page_pipeline.py` o único caminho `enforce`

**Files:**

- Modify: `pipeline/main.py`
- Modify: `pipeline/strip/run.py`
- Modify: `pipeline/strip/process_bands.py`
- Modify: `pipeline/strip/page_pipeline.py`
- Modify: `pipeline/ownership/reconcile.py`
- Modify: `pipeline/typesetter/owner_style.py`
- Modify: `pipeline/qa/style_fidelity.py`
- Test: `pipeline/tests/test_main_emit.py`
- Test: `pipeline/tests/test_strip_owner_control_plane.py`
- Test: `pipeline/tests/test_page_owner_pipeline.py`
- Test: `pipeline/tests/test_strip_owner_composition_integration.py`
- Test: `pipeline/tests/test_owner_style_profile.py`
- Test: `pipeline/tests/test_style_fidelity_qa.py`

**Step 1: Escrever testes RED de autoridade única**

```python
def test_enforce_mode_calls_only_page_owner_pipeline(monkeypatch):
    legacy = MagicMock(wraps=strip_run._resolve_owner_graph_from_evidence)
    monkeypatch.setattr(strip_run, "_resolve_owner_graph_from_evidence", legacy)
    output_pages = _run_minimal_chapter(owner_graph_mode="enforce")
    assert output_pages
    assert all(page.owner_page_result is not None for page in output_pages)
    assert all(page.owner_page_result.status == "final_verified" for page in output_pages)
    legacy.assert_not_called()


def test_verified_owner_result_skips_late_payload_hydration_and_normalization(monkeypatch):
    hydrate = MagicMock(wraps=main._hydrate_project_render_metadata_from_debug_candidates)
    normalize = MagicMock(wraps=main._normalize_final_project_page_space_layers)
    repairs = MagicMock(wraps=main._apply_owner_mode_project_repairs)
    monkeypatch.setattr(main, "_hydrate_project_render_metadata_from_debug_candidates", hydrate)
    monkeypatch.setattr(main, "_normalize_final_project_page_space_layers", normalize)
    monkeypatch.setattr(main, "_apply_owner_mode_project_repairs", repairs)
    _finalize(_verified_page_result())
    hydrate.assert_not_called()
    normalize.assert_not_called()
    repairs.assert_not_called()


def test_shadow_mode_may_compare_but_cannot_mutate_enforce_result():
    enforce = _run_page(style_copy_mode="shadow", owner_graph_mode="shadow")
    assert enforce.final_pixel_sha256 == enforce.owner_result_pixel_sha256
    assert enforce.shadow_diagnostics


def test_style_copy_off_does_not_call_copier_or_style_audit(monkeypatch):
    profiles = MagicMock(side_effect=AssertionError("style profile builder must be disabled"))
    audit = MagicMock(side_effect=AssertionError("style audit must be disabled"))
    monkeypatch.setattr(owner_style, "build_owner_visual_profiles", profiles)
    monkeypatch.setattr(style_fidelity, "audit_style_fidelity", audit)
    result = _run_page(style_copy_mode="off")
    assert result.status == "final_verified"
    profiles.assert_not_called()
    audit.assert_not_called()


def test_style_enforce_replay_uses_hash_verified_owner_and_binding_artifacts(monkeypatch):
    discovery_ocr = MagicMock(side_effect=AssertionError("discovery OCR must not rerun during style replay"))
    final_observer_ocr = MagicMock(return_value=_fresh_final_ocr_result())
    translator = MagicMock(side_effect=AssertionError("translation must not rerun during style replay"))
    result = _run_page(
        style_copy_mode="enforce",
        replay_owner_artifacts=_verified_off_run_artifacts(),
        discovery_ocr=discovery_ocr,
        final_observer_ocr=final_observer_ocr,
        translator=translator,
    )
    assert result.owner_graph_sha256 == _off_owner_graph_sha256()
    assert result.translation_bindings_sha256 == _off_bindings_sha256()
    assert result.cleanup_base_sha256 == _off_cleanup_base_sha256()
    discovery_ocr.assert_not_called()
    translator.assert_not_called()
    final_observer_ocr.assert_called()
    assert result.terminal_proof.fresh_ocr_root_input_pixel_sha256 == result.final_page.page_output_pixel_sha256
    assert result.terminal_proof.fresh_ocr_attempt_chain_sha256
```

**Step 2: Confirmar RED**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest `
    tests/test_main_emit.py `
    tests/test_strip_owner_control_plane.py `
    tests/test_page_owner_pipeline.py `
    tests/test_strip_owner_composition_integration.py `
    tests/test_owner_style_profile.py `
    tests/test_style_fidelity_qa.py `
    -q
} finally { Pop-Location }
```

Expected: `enforce` ainda atravessa caminhos paralelos e reparos tardios de payload.

**Step 3: Roteamento final**

- `owner_graph_mode=enforce` chama exclusivamente `run_page_owner_pipeline()`.
- `run.py` mantém scheduling, leitura e persistência, sem redecidir semântica.
- `process_bands.py` mantém adapters de baixa camada e compatibilidade de teste.
- `legacy`/`shadow` não alteram pixels, owners, translations ou artefatos canônicos de uma run `enforce`.
- remover do caminho verificado as hidratações/normalizações tardias em torno de `main.py:9602`, `9657` e `9674`.
- suportar explicitamente `style_copy_mode=off|shadow|render|enforce` nos validators/configs reais: `off` não chama `build_owner_visual_profiles()` nem `audit_style_fidelity()` e usa renderer base; `shadow` só coleta diagnóstico sem mutar; `render/enforce` podem aplicar estilo, sempre com fallback base para o contrato de conteúdo. Cobrir o no-op audit/gate de `off` em `test_style_fidelity_qa.py`.
- ampliar o runner CLI com `--chapter`, `--owner-graph-mode`, `--style-copy-mode` e `--replay-owner-artifacts`; propagar os valores ao `runner_config.json` e cobrir parsing/defaults em `test_main_emit.py`, para que as runs reais da Task 20 sejam reproduzíveis sem editar configs antigas.
- em replay, validar source tree hash, schema/run/page/source hashes, cardinalidade do owner graph/bindings e `cleanup_base_sha256` da run `off`; pular discovery, OCR de coverage/source, tradução e inpaint, mas rerodar glyph render/style, composição e o OCR observer terminal fresco sobre os novos pixels. Isso isola o efeito do style copier de nondeterminismo upstream sem reutilizar prova terminal de outra imagem.

**Step 4: Adicionar assertions de fronteira**

Falhar imediatamente se um caminho legado tentar:

- trocar `owner_id`, source/target payload ou page hash;
- anexar commits page-space a bands;
- produzir camada de texto por índice;
- normalizar novamente um payload já vinculado;
- modificar a imagem depois de `TerminalPixelProof`.

**Step 5: Confirmar GREEN**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest `
    tests/test_main_emit.py `
    tests/test_strip_owner_control_plane.py `
    tests/test_page_owner_pipeline.py `
    tests/test_strip_owner_composition_integration.py `
    tests/test_owner_lifecycle_e2e.py `
    tests/test_owner_style_profile.py `
    tests/test_style_fidelity_qa.py `
    -q
} finally { Pop-Location }
```

Expected: PASS; `enforce` possui exatamente um coordenador e nenhuma pós-etapa modifica pixels verificados.

**Step 6: Commit**

```powershell
git add -p -- pipeline/main.py pipeline/strip/run.py pipeline/strip/process_bands.py pipeline/strip/page_pipeline.py pipeline/ownership/reconcile.py pipeline/typesetter/owner_style.py pipeline/qa/style_fidelity.py pipeline/tests/test_main_emit.py pipeline/tests/test_strip_owner_control_plane.py pipeline/tests/test_page_owner_pipeline.py pipeline/tests/test_strip_owner_composition_integration.py pipeline/tests/test_owner_style_profile.py pipeline/tests/test_style_fidelity_qa.py
git diff --cached --check
git commit -m "refactor: make page owner pipeline authoritative"
```

### Task 18: Criar corpus sintético, auditor de capítulo e matriz multiobra

**Files:**

- Modify: `pipeline/tests/fixtures/english_owner_recovery/manifest.json`
- Modify: `pipeline/tests/fixtures/english_owner_recovery/recipes.json`
- Modify: `pipeline/tests/test_owner_lifecycle_e2e.py`
- Create: `pipeline/tools/audit_owner_chapter_output.py`
- Create: `pipeline/tests/test_audit_owner_chapter_output.py`
- Create: `pipeline/tools/init_owner_validation_run.py`
- Create: `pipeline/tests/test_init_owner_validation_run.py`
- Modify: `pipeline/tools/export_visual_review_sheet.py`
- Modify: `pipeline/tests/test_export_visual_review_sheet.py`
- Modify: `pipeline/tests/fixtures/owner_visual_matrix/inputs.json`
- Modify: `pipeline/tests/fixtures/owner_visual_matrix/functional_matrix.json`
- Create: `pipeline/tests/fixtures/owner_visual_matrix/configs/mitch_items_ch39.json`
- Modify: `pipeline/tests/test_owner_visual_matrix_tool.py`

**Step 1: Escrever testes RED do corpus sistêmico**

```python
@pytest.mark.parametrize(
    "case_id",
    [
        "no_band_dialogue",
        "unassociated_full_page_line",
        "missing_container",
        "fragmented_multiline_body",
        "mixed_en_pt_overlay",
        "translucent_gradient_balloon",
        "dark_balloon_white_text",
        "source_residual_after_r0",
        "burst_balloon",
        "card_table_multirole",
        "text_touching_art_or_character",
        "cross_tile_owner",
    ],
)
def test_synthetic_case_finishes_with_only_bound_ptbr(case_id):
    result = run_fixture_case(case_id)
    metrics = compute_page_acceptance_metrics(result)
    assert result.status == "final_verified"
    assert metrics.english_dialogue_residual_count == 0
    assert metrics.translatable_components_without_owner == 0
    assert metrics.material_components_without_ocr_attempt == 0
    assert metrics.owners_without_valid_pt_br == 0
    assert metrics.owners_without_atomic_cleanup_render == 0
    assert metrics.owners_without_target_materialization == 0
    assert metrics.material_components_without_terminal_lifecycle == 0


def test_chapter_auditor_rejects_final_page_with_source_english():
    run = _run_with_one_visible_source_line()
    audit = audit_chapter(run, external_ocr_runner=_valid_external_runner_for(run))
    assert audit.status == "NO_GO"
    assert audit.english_dialogue_residual_count == 1


def test_chapter_auditor_rejects_fresh_ocr_root_hash_not_matching_redecoded_final():
    run = _run_with_terminal_ocr_root_from_other_pixels()
    audit = audit_chapter(run, external_ocr_runner=_valid_external_runner_for(run))
    assert audit.status == "NO_GO"
    assert "fresh_ocr_input_pixel_hash_mismatch" in audit.identity_mismatches


def test_chapter_auditor_rejects_tampered_physical_attempt_hash():
    run = _run_with_tampered_terminal_physical_attempt()
    audit = audit_chapter(run, external_ocr_runner=_valid_external_runner_for(run))
    assert audit.status == "NO_GO"
    assert audit.unverified_physical_ocr_attempts == 1


def test_chapter_auditor_accepts_hash_linked_full_page_crop_and_variant_probe():
    run = _verified_run_with_full_page_crop_and_variants()
    external = _external_invocation_for_redecoded_final(run, invocation_id="audit-fresh")
    audit = audit_chapter(run, external_ocr_runner=MagicMock(return_value=external))
    assert audit.external_auditor_fresh_ocr_complete is True
    assert audit.auditor_input_pixel_sha256 == _redecoded_final_pixel_sha256(run)
    assert audit.auditor_unverified_physical_ocr_attempts == 0
    assert audit.auditor_cache_hits == 0
    assert audit.auditor_physical_inference_count >= 1


def test_external_auditor_rejects_root_b_even_when_stored_terminal_proof_for_a_is_valid(tmp_path):
    run = _fully_valid_stored_run_for_candidate_a(tmp_path)
    external = _external_invocation(root_pixels=_candidate_b_pixels(), invocation_id="audit-new-b")
    runner = MagicMock(return_value=external)
    audit = audit_chapter(run, external_ocr_runner=runner)
    runner.assert_called_once()
    assert audit.status == "NO_GO"
    assert audit.external_auditor_fresh_ocr_complete is False
    assert "auditor_root_hash_mismatch" in audit.identity_mismatches


def test_external_auditor_uses_new_uncached_invocation_and_reports_its_exact_metrics(tmp_path):
    run = _fully_valid_stored_run_for_candidate_a(tmp_path, terminal_invocation_id="terminal-old")
    external = _external_invocation_for_redecoded_final(
        run,
        invocation_id="audit-new",
        variants=("full_page", "gray", "inverted", "scale_2x"),
        cache_hits=0,
    )
    runner = MagicMock(return_value=external)
    audit = audit_chapter(run, external_ocr_runner=runner)
    runner.assert_called_once_with(_persisted_final_path(run), force_new_process=True, disable_cache=True)
    assert audit.external_auditor_fresh_ocr_complete is True
    assert audit.auditor_invocation_id == external.request.invocation_id == "audit-new"
    assert audit.auditor_invocation_id != run.terminal_proof.fresh_ocr_invocation_id
    assert audit.auditor_input_pixel_sha256 == external.request.root_input_pixel_sha256
    assert audit.auditor_physical_attempt_chain_sha256 == external.attempt_chain_sha256
    assert audit.auditor_physical_inference_count == sum(a.provider_called for a in external.attempts)
    assert audit.auditor_cache_hits == 0


@pytest.mark.parametrize("external_failure", ["runner_missing", "provider_unavailable", "no_physical_full_page"])
def test_external_auditor_never_falls_back_to_stored_terminal_proof(external_failure, tmp_path):
    run = _fully_valid_stored_run_for_candidate_a(tmp_path)
    audit = audit_chapter(run, external_ocr_runner=_failing_external_runner(external_failure))
    assert audit.status == "NO_GO"
    assert audit.external_auditor_fresh_ocr_complete is False
    assert audit.auditor_invocation_id is None
    assert "external_auditor_unavailable_or_incomplete" in audit.identity_mismatches


def test_chapter_auditor_emits_real_unique_source_output_pairs():
    audit = audit_chapter(_verified_three_page_run(), write_review_pairs=True)
    assert len(audit.review_pairs) == 3
    assert len({pair.output_pixel_sha256 for pair in audit.review_pairs}) == 3
    assert all(pair.source_path.exists() and pair.output_path.exists() for pair in audit.review_pairs)


@pytest.mark.parametrize("changed", ["owner_graph", "translation_bindings", "cleanup_base"])
def test_style_comparison_rejects_changed_pre_render_content_hash(changed):
    audit = compare_content_replay(_off_run(), _style_replay_with_changed_hash(changed))
    assert audit.status == "NO_GO"
    assert changed in audit.identity_mismatches


def test_review_tool_never_autofills_visual_go():
    result = export_source_candidate_review(_source_dir(), _candidate_run(), _review_dir())
    assert result["visual_verdict"] is None
    assert all(item["reviewed"] is False for item in result["items"])


def test_validation_context_uses_new_root_and_is_atomically_reloadable(tmp_path):
    first = init_validation_context(tmp_path / "validation", tmp_path / "current.json")
    second = init_validation_context(tmp_path / "validation", tmp_path / "current.json")
    assert first.validation_root != second.validation_root
    assert load_validation_context(tmp_path / "current.json") == second
    assert not (tmp_path / "current.json.tmp").exists()
```

**Step 2: Confirmar RED**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest `
    tests/test_owner_lifecycle_e2e.py `
    tests/test_audit_owner_chapter_output.py `
    tests/test_init_owner_validation_run.py `
    tests/test_export_visual_review_sheet.py `
    tests/test_owner_visual_matrix_tool.py `
    -q
} finally { Pop-Location }
```

Expected: fixtures/auditor ainda não existem e a matriz não conhece a regressão principal.

**Step 3: Criar fixtures por receita, não por exceção visual**

`recipes.json` descreve dimensões, container, background, glyph layers, idioma e falha simulada. O teste gera as imagens em diretório temporário. Não versionar screenshots da obra nem codificar frases/coordenadas reais em produção.

O corpus completo — não cada caso isolado — deve cobrir por matriz:

- cor clara/escura/gradiente;
- texto sólido, contornado e com anti-alias;
- uma/múltiplas linhas;
- container presente/ausente/translúcido;
- band ausente/cross-band;
- burst, card/tabela com múltiplos roles, cross-tile e texto tocando arte/personagem protegidos;
- source EN, target PT-BR e input misto.

O teste do manifest calcula a cobertura dessa matriz e falha se qualquer dimensão não tiver ao menos um caso. Cada receita E2E exige todas as sete métricas globais zeradas, não apenas ausência de inglês/owner/tradução.

**Step 4: Implementar contexto reentrante e auditor read-only de output**

`init_owner_validation_run.py` cria uma raiz nova com sufixo monotônico sem apagar raízes anteriores e grava atomicamente `.codex-tmp/universal-source-replacement-current.json`. O JSON contém `validation_run_id`, `validation_root`, source path/hash, `created_at` e paths derivados `off_out`, `off_audit`, `off_review`, `style_out`, `style_audit`, `style_review`, `matrix_out`. Todo bloco da Task 20 recarrega e valida esse arquivo; não depende de variável PowerShell ou env de outro processo.

Defina no auditor `PageAcceptanceMetrics` frozen com exatamente os sete contadores globais deste plano, e `compute_page_acceptance_metrics(result: PageExecutionResult) -> PageAcceptanceMetrics`. Essa é a única origem de `metrics` dos testes/relatórios; `PageExecutionResult` não ganha campo derivado nem segunda autoridade. O builder lê ledger, graph, bindings, commits, materializations, verdicts e proof imutáveis e é reutilizado na agregação do capítulo.

CLI:

```text
python pipeline/tools/init_owner_validation_run.py \
  --base .codex-tmp/universal-source-replacement-20260804 \
  --source N:/TraduzAI/temporario1/mch39 \
  --context .codex-tmp/universal-source-replacement-current.json
```

O auditor usa o contexto apenas quando paths explícitos não forem fornecidos.

CLI:

```text
python pipeline/tools/audit_owner_chapter_output.py \
  --source <source_dir> \
  --run <work_dir> \
  --report <audit.json> \
  --review-dir <review_pairs_dir> \
  --source-lang en \
  --target-lang pt-BR \
  [--compare-content-run <off_run>] \
  --require-final-verified
```

O auditor:

- lê apenas artefatos canônicos e imagens finais;
- confere cardinalidade página-fonte/página-final;
- revalida hashes/ledger/bindings/provas/gate;
- executa OCR fresco em processo novo pelo mesmo `execute_hash_bound_provider_attempt()` da Task 2, nunca fabricando records no auditor. O único seam testável é `external_ocr_runner(path, force_new_process=True, disable_cache=True)`; ausência/falha/invocation sem full-page físico produz NO_GO e jamais reutiliza `TerminalPixelProof` armazenado;
- reabre cada final, exige root hash idêntico aos pixels redecodificados, exige `auditor_invocation_id != terminal_proof.fresh_ocr_invocation_id`, recompõe crops/transforms a partir do `OCRTransformSpec` persistido pela invocation **externa** e valida a cadeia física. Publica `external_auditor_fresh_ocr_complete`, `auditor_invocation_id`, `auditor_input_pixel_sha256`, `auditor_physical_attempt_chain_sha256`, `auditor_physical_inference_count`, `auditor_unverified_physical_ocr_attempts` e `auditor_cache_hits`; todos derivam do retorno do runner externo, nunca dos attempts terminais persistidos;
- quando `--compare-content-run` estiver presente, exige igualdade de source tree, owner graph, translation bindings e cleanup base antes de auditar o render de estilo;
- agrega as métricas de aceitação;
- produz pares PNG source/final com resolução nativa ou crop lossless;
- para sentinelas e casos solicitados, exporta um pacote hash-linked de seis estágios em escala nativa: `original`, `coverage_ocr_overlay`, `inpaint`, `typeset`, `page_composition` e `persisted_final`; overlays são derivados e marcados, enquanto os estágios de pixels apontam para artefatos canônicos reais;
- nunca altera a run nem preenche verdict visual.

O relatório possui `external_audit_gate_status: PASS|NO_GO`, separado do export gate runtime. Ele só é `PASS` quando todas as páginas têm a invocation externa independente completa e todas as métricas externas acima são válidas; prova terminal armazenada, subprocesso ausente ou cache hit força `NO_GO`.

**Step 5: Adicionar Mitch Items cap. 39 à matriz externa**

Em `inputs.json`, mapear `mitch_items_ch39` para `TRADUZAI_MATRIX_MITCH39_SOURCE`. A entrada do manifest usa categorias genéricas e páginas de regressão 10, 11, 19, 21, 27, 28, 30, 34, 36 e 39. Páginas 30/34/36 recebem categoria de fixture `mixed_source_overlay`; isso é expectativa de teste, não regra de produção.

Config da matriz:

```json
{
  "idioma_origem": "en",
  "idioma_destino": "pt-BR",
  "capitulo": 39,
  "owner_graph_mode": "enforce",
  "style_copy_mode": "off",
  "export_mode": "strict",
  "debug": true
}
```

Recalcular e registrar os SHA-256 do config e da entrada pelo mecanismo explícito do validador; o teste deve falhar se o arquivo mudar sem atualização do hash.

**Step 6: Confirmar GREEN**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest `
    tests/test_owner_lifecycle_e2e.py `
    tests/test_audit_owner_chapter_output.py `
    tests/test_init_owner_validation_run.py `
    tests/test_export_visual_review_sheet.py `
    tests/test_owner_visual_matrix_tool.py `
    -q
} finally { Pop-Location }
```

Expected: PASS; corpus cobre todas as classes sem depender de obra/página e auditor produz evidência real não preenchida.

**Step 7: Commit**

```powershell
git add -- pipeline/tests/fixtures/english_owner_recovery/manifest.json pipeline/tests/fixtures/english_owner_recovery/recipes.json pipeline/tools/audit_owner_chapter_output.py pipeline/tests/test_audit_owner_chapter_output.py pipeline/tools/init_owner_validation_run.py pipeline/tests/test_init_owner_validation_run.py pipeline/tests/fixtures/owner_visual_matrix/configs/mitch_items_ch39.json
git add -p -- pipeline/tests/test_owner_lifecycle_e2e.py pipeline/tools/export_visual_review_sheet.py pipeline/tests/test_export_visual_review_sheet.py pipeline/tests/fixtures/owner_visual_matrix/inputs.json pipeline/tests/fixtures/owner_visual_matrix/functional_matrix.json pipeline/tests/test_owner_visual_matrix_tool.py
git diff --cached --check
git commit -m "test: add universal source replacement validation corpus"
```

### Task 19: Executar suítes focadas e regressão completa

**Files:**

- Modify only if a test exposes a regression: files owned by Tasks 2–18
- Test: all `pipeline/tests/`

**Step 1: Verificar sintaxe/imports**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m compileall `
    ownership qa strip translator typesetter vision_stack tools
} finally { Pop-Location }
```

Expected: exit 0, sem erro de import/sintaxe.

**Step 2: Rodar a suíte owner/coverage/translation**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest `
    tests/test_vision_stack_ocr.py `
    tests/test_vision_stack_runtime.py `
    tests/test_owner_evidence.py `
    tests/test_owner_reconcile.py `
    tests/test_owner_reconcile_properties.py `
    tests/test_owner_coverage_rescue.py `
    tests/test_owner_lifecycle.py `
    tests/test_owner_translation.py `
    tests/test_translation_language_policy.py `
    tests/test_translation_locale_policy.py `
    -q
} finally { Pop-Location }
```

Expected: PASS.

**Step 3: Rodar a suíte execução/QA/E2E**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest `
    tests/test_page_owner_pipeline.py `
    tests/test_owner_atomic_execution.py `
    tests/test_owner_source_replacement_fail_closed.py `
    tests/test_owner_repair.py `
    tests/test_owner_repair_controller.py `
    tests/test_strip_owner_composition_integration.py `
    tests/test_language_residual.py `
    tests/test_final_pixel_qa.py `
    tests/test_final_pixel_observer.py `
    tests/test_final_pixel_export_gate.py `
    tests/test_owner_lifecycle_e2e.py `
    tests/test_page_owner_artifacts.py `
    tests/test_audit_owner_chapter_output.py `
    tests/test_owner_visual_matrix_tool.py `
    -q
} finally { Pop-Location }
```

Expected: PASS.

**Step 4: Rodar toda a suíte Python**

```powershell
Push-Location pipeline
try { .\venv\Scripts\python.exe -m pytest tests -q } finally { Pop-Location }
```

Expected: PASS. Não aceite “só os testes novos passam”.

**Step 5: Corrigir regressões uma por vez com TDD**

Para cada falha:

1. confirmar que ela não é mudança local alheia;
2. executar somente o teste falho;
3. registrar causa no checkpoint;
4. aplicar hunk mínimo no arquivo owner da Task;
5. rerodar teste isolado, suíte de área e suíte completa;
6. não alterar expectativa válida apenas para obter verde.

**Step 6: Commit de integração somente se necessário**

```powershell
git add -p -- <somente arquivos realmente corrigidos nesta Task>
git diff --cached --check
git commit -m "fix: close universal owner replacement regressions"
```

Se nenhuma correção foi necessária, não criar commit vazio.

### Task 20: Rodar capítulo 39 completo, holdouts e validar visualmente

**Files:**

- Create: `docs/reports/2026-08-04-universal-balloon-source-replacement-validation.md`
- Create: `docs/reports/evidence/2026-08-04-universal-balloon-source-replacement-validation.json`
- Do not track: `.codex-tmp/universal-source-replacement-*/**`

**Step 1: Preflight sem tocar nas mudanças locais**

```powershell
git status --short --branch
$env:TRADUZAI_REQUIRE_GPU = '1'
git diff --check 6d5863d9..HEAD -- pipeline docs
.\pipeline\venv\Scripts\python.exe pipeline\main.py --hardware-info
.\pipeline\venv\Scripts\python.exe -c "import paddle; print({'compiled_with_cuda': paddle.device.is_compiled_with_cuda(), 'device': paddle.device.get_device()})"
```

Expected: build CUDA disponível; a run deve registrar OCR/inpaint em GPU nos artefatos de hardware. Se o ambiente estiver operacionalmente sem GPU, corrija o runtime/ambiente antes da run; não troque silenciosamente para CPU e não altere o algoritmo para contornar o problema.

**Step 2: Criar raiz nova de evidência e rodar baseline autoritativo**

```powershell
.\pipeline\venv\Scripts\python.exe pipeline\tools\init_owner_validation_run.py `
  --base '.codex-tmp\universal-source-replacement-20260804' `
  --source 'N:\TraduzAI\temporario1\mch39' `
  --context '.codex-tmp\universal-source-replacement-current.json'
if ($LASTEXITCODE -ne 0) { throw "validation context init failed: $LASTEXITCODE" }
$validationContext = Get-Content -LiteralPath '.codex-tmp\universal-source-replacement-current.json' -Raw | ConvertFrom-Json
$offOut = [string]$validationContext.off_out
$env:TRADUZAI_REQUIRE_GPU = '1'

.\pipeline\venv\Scripts\python.exe pipeline\main.py `
  --input 'N:\TraduzAI\temporario1\mch39' `
  --work 'Mitch Items' `
  --chapter 39 `
  --source-lang en `
  --target pt-BR `
  --mode real `
  --output $offOut `
  --debug `
  --strict `
  --export-mode strict `
  --owner-graph-mode enforce `
  --style-copy-mode off
if ($LASTEXITCODE -ne 0) { throw "mch39 style-off run failed: $LASTEXITCODE" }
```

Não reutilizar nem apagar `.codex-tmp/mch39_full_directional_gradient_20260803_retry2/out`; ela é evidência diagnóstica anterior.

**Step 3: Auditar a run off em processo novo**

```powershell
$validationContext = Get-Content -LiteralPath '.codex-tmp\universal-source-replacement-current.json' -Raw | ConvertFrom-Json
$offOut = [string]$validationContext.off_out
$offAudit = [string]$validationContext.off_audit
$offReview = [string]$validationContext.off_review
$env:TRADUZAI_REQUIRE_GPU = '1'
.\pipeline\venv\Scripts\python.exe pipeline\tools\audit_owner_chapter_output.py `
  --source 'N:\TraduzAI\temporario1\mch39' `
  --run $offOut `
  --report $offAudit `
  --review-dir $offReview `
  --source-lang en `
  --target-lang pt-BR `
  --require-final-verified
if ($LASTEXITCODE -ne 0) { throw "mch39 off audit failed: $LASTEXITCODE" }
```

Verifique no JSON:

```text
page_count == 42
english_dialogue_residual_count == 0
translatable_components_without_owner == 0
material_components_without_ocr_attempt == 0
owners_without_valid_pt_br == 0
owners_without_atomic_cleanup_render == 0
owners_without_target_materialization == 0
material_components_without_terminal_lifecycle == 0
final_pixel_ocr_coverage_complete == true
fresh_ocr_root_hash_matches_final == true
unverified_physical_ocr_attempts == 0
fresh_ocr_cache_hits == 0
external_auditor_fresh_ocr_complete == true
auditor_invocation_is_independent == true
auditor_root_hash_matches_final == true
auditor_unverified_physical_ocr_attempts == 0
auditor_cache_hits == 0
external_audit_gate_status == PASS
export_gate_status == PASS
```

**Step 4: Inspecionar visualmente os 42 pares reais**

Recarregue `.codex-tmp/universal-source-replacement-current.json` e use seu campo `off_review`; não dependa da variável de um bloco anterior. Abra com `view_image` cada PNG desse diretório. Avalie em resolução original:

- o inglês foi realmente removido, não apenas coberto por PT-BR;
- o PT-BR correspondente está presente uma vez;
- corpo completo não foi repartido entre owners/camadas;
- texto fica dentro e centralizado no container;
- borda, personagens, ícones e arte protegida não foram apagados;
- não há resíduo, duplicação, texto apagado pela metade ou copyback com band de cor diferente.

Registre verdict por página manualmente no evidence JSON; não use notas auto preenchidas. Para as regressões 10, 11, 19, 21, 27, 28, 30, 34, 36 e 39, abra também crops lossless source/final e documente o owner/estratégia que fechou cada caso.

Nessas dez sentinelas, abra ainda os seis estágios reais gerados pelo auditor — original, coverage/OCR overlay, inpaint, typeset, composição e bytes finais redecodificados — em escala nativa. Registre hash/verdict por estágio para localizar reaparecimento do inglês, perda do PT-BR, copyback incorreto ou dano de arte. Faça a mesma inspeção de seis estágios para qualquer página inicialmente marcada NO-GO antes de corrigir e rerodar.

Se qualquer página for NO-GO, volte à Task dona da causa, escreva um teste genérico RED, corrija, rode as suítes e repita o capítulo inteiro em uma nova raiz. Não edite pixels da saída.

**Step 5: Rodar compatibilidade com style-copy ativo**

```powershell
$validationContext = Get-Content -LiteralPath '.codex-tmp\universal-source-replacement-current.json' -Raw | ConvertFrom-Json
$offOut = [string]$validationContext.off_out
$styleOut = [string]$validationContext.style_out
$styleAudit = [string]$validationContext.style_audit
$styleReview = [string]$validationContext.style_review
$env:TRADUZAI_REQUIRE_GPU = '1'
.\pipeline\venv\Scripts\python.exe pipeline\main.py `
  --input 'N:\TraduzAI\temporario1\mch39' `
  --work 'Mitch Items' `
  --chapter 39 `
  --source-lang en `
  --target pt-BR `
  --mode real `
  --output $styleOut `
  --debug `
  --strict `
  --export-mode strict `
  --owner-graph-mode enforce `
  --style-copy-mode enforce `
  --replay-owner-artifacts $offOut
if ($LASTEXITCODE -ne 0) { throw "mch39 style-enforce run failed: $LASTEXITCODE" }

.\pipeline\venv\Scripts\python.exe pipeline\tools\audit_owner_chapter_output.py `
  --source 'N:\TraduzAI\temporario1\mch39' `
  --run $styleOut `
  --report $styleAudit `
  --review-dir $styleReview `
  --source-lang en `
  --target-lang pt-BR `
  --compare-content-run $offOut `
  --require-final-verified
if ($LASTEXITCODE -ne 0) { throw "mch39 style-enforce audit failed: $LASTEXITCODE" }
```

O replay reutiliza o cleanup/base inpainted persistido da run `off` e reroda somente glyph render/style, composição e QA. Não reroda discovery/OCR de coverage, tradução nem inpaint; executa obrigatoriamente um OCR observer terminal novo, cujo `root_input_pixel_sha256` coincide com os pixels estilizados reabertos e cuja cadeia de attempts físicos é recalculada no boundary hash-bound. O auditor deve exigir `fresh_ocr_root_hash_matches_final=true`, zero attempts não verificados, zero cache hits e igualdade de `source_tree_sha256`, `owner_graph_sha256`, `translation_bindings_sha256` e `cleanup_base_sha256` antes de comparar estilo. O style copier pode alterar aparência, mas não coverage, owners, binding, cleanup ou a condição zero-inglês. Se style-copy não tiver confiança, o fallback obrigatório é o renderer base; não preservar source.

**Step 6: Rodar matriz multiobra**

```powershell
$validationContext = Get-Content -LiteralPath '.codex-tmp\universal-source-replacement-current.json' -Raw | ConvertFrom-Json
$validationRoot = [string]$validationContext.validation_root
$matrixOut = [string]$validationContext.matrix_out
$env:TRADUZAI_REQUIRE_GPU = '1'
$env:TRADUZAI_MATRIX_MITCH39_SOURCE = 'N:\TraduzAI\temporario1\mch39'
$env:TRADUZAI_MATRIX_MYTHIC_SOURCE = 'N:\TraduzAI\.codex-tmp\mythic_ch40_pages_1_2_source_20260724'
$env:TRADUZAI_MATRIX_ONE_SECOND_SOURCE = 'N:\TraduzAI\.codex-tmp\page_owner_systemic_sources_20260729\one_second'
$env:TRADUZAI_MATRIX_GRAND_FINALE_SOURCE = 'N:\TraduzAI\.codex-tmp\page_owner_systemic_sources_20260729\grand_finale'
.\pipeline\venv\Scripts\python.exe pipeline\tools\validate_owner_visual_matrix.py `
  --manifest pipeline\tests\fixtures\owner_visual_matrix\functional_matrix.json `
  --output-root $matrixOut `
  --report (Join-Path $validationRoot 'multiwork-matrix.md') `
  --inspection-template (Join-Path $validationRoot 'multiwork-inspection.json') `
  --run-id universal-source-replacement-v1
if ($LASTEXITCODE -ne 0) { throw "multiwork matrix failed: $LASTEXITCODE" }
```

Inspecione os outputs reais de Regressed Genius, 1 Second e Grand Finale pelo mesmo procedimento source/final. Exija as mesmas métricas de conteúdo; não aceite GO apenas porque o comando retornou exit 0.

**Step 7: Produzir relatório honesto e reproduzível**

O Markdown e o JSON devem incluir:

- branch, HEAD inicial/final e commits das Tasks;
- comandos, duração, hardware/device e paths das runs;
- hashes de source/output/audits;
- contagem por estado do ledger, estratégia R0–R3 e tipo de issue;
- métricas exatas da run end-to-end `off`, do replay de render `enforce` e dos três holdouts;
- tabela visual das 42 páginas e crops das dez regressões;
- distinção entre validação automática e inspeção visual;
- qualquer falha restante como NO-GO, sem maquiar por warning.

Copie apenas os JSONs resumidos necessários para `docs/reports/evidence/`; não copie páginas, crops ou outputs para o Git.

**Step 8: Checkpoint F — Definition of Done**

Aceite o plano como concluído somente quando todos forem verdadeiros:

```text
[ ] Suíte Python completa passa.
[ ] A run end-to-end `off` e o replay integral de render `enforce` têm 42/42 páginas final_verified.
[ ] Zero inglês material permanece em containers traduzíveis.
[ ] Todo texto PT-BR vinculado foi renderizado exatamente uma vez.
[ ] Todo componente material possui ao menos uma tentativa OCR registrada.
[ ] Nenhum owner está sem materialização target comprovada.
[ ] Nenhum componente traduzível está sem owner/tradução/commit.
[ ] OCR terminal de cada output prova root hash dos pixels finais, cadeia física íntegra e zero cache hit.
[ ] Auditor em processo novo prova invocation distinta, root dos finals reabertos, cadeia física íntegra e zero cache; prova terminal armazenada não satisfaz este item.
[ ] R3 fecha todos os casos de conteúdo que R0-R2 não fecharam.
[ ] Style-copy off e enforce preservam o mesmo contrato de conteúdo; shadow, se executado, não muta pixels.
[ ] Holdouts das três outras obras têm métricas zeradas e inspeção GO.
[ ] Todos os artefatos de inspeção têm paths e hashes únicos/reais.
[ ] Export gate final é PASS sem override e sem with_warnings.
[ ] Relatório visual é baseado em inspeção própria, não em template.
```

Se um item estiver falso, o trabalho continua na Task responsável; não declarar conclusão parcial como sucesso.

**Step 9: Commit do relatório, sem outputs**

```powershell
git status --short --branch
git add -- docs/reports/2026-08-04-universal-balloon-source-replacement-validation.md docs/reports/evidence/2026-08-04-universal-balloon-source-replacement-validation.json
git diff --cached --check
git diff --cached -- docs/reports/2026-08-04-universal-balloon-source-replacement-validation.md docs/reports/evidence/2026-08-04-universal-balloon-source-replacement-validation.json
git commit -m "docs: validate universal balloon source replacement"
Remove-Item Env:\TRADUZAI_REQUIRE_GPU -ErrorAction SilentlyContinue
```

## Ordem dos checkpoints

```text
A  OCR request-scoped e identidade de evidência
B  cobertura page-global e owner graph total
C  binding PT-BR válido por owner
D  transação atômica e reparo R0-R3
E  QA reparador, artefatos e gate verificáveis
F  run E2E cap. 39 + replay style + matriz multiobra + inspeção visual
```

Cada checkpoint acumula todos os anteriores. Uma falha encontrada depois deve produzir um teste genérico na Task dona e repetir todos os checkpoints subsequentes.

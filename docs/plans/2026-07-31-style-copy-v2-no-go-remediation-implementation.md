# Style Copy V2 NO-GO Remediation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Corrigir sistemicamente o Style Copy V2 para que evidência visual real de cada owner controle os pixels efetivamente commitados, seja persistida por um contrato raster imutável e seja validada por QA/gates owner-scoped e fail-closed em obras, capítulos e categorias não vistas.

**Architecture:** O pipeline cria um sidecar visual antes de tradução/inpaint, deriva elegibilidade das observações selecionadas, extrai evidência por máscaras e resolve fonte pela silhueta real. Decisões são independentes por atributo e totalmente materializadas por um backend com capabilities explícitas. O renderer entrega pixels e um contrato raster dentro de `OwnerGlyphPatch`; o commit atômico valida identidade, hashes, geometria, atributos e pixels antes de persistir. QA usa o owner graph como denominador, separa o subgate funcional do subgate de estilo e a matriz usa targets exatos, thresholds executáveis e inspeção nativa obrigatória.

**Tech Stack:** Python 3.12, dataclasses, typing, NumPy, OpenCV, Pillow, matplotlib FT2Font, fontTools já instalado, pytest, JSON/JSONL, SHA-256, Delta E 2000, IoU/contorno, PowerShell e Git com staging restrito.

---

## Fonte de verdade e estado inicial

- Design aprovado: `docs/plans/2026-07-31-style-copy-v2-no-go-remediation-design.md`, commit `84fd85a0`.
- Implementação Style Copy V2 anterior: commits `6d32540d` e `3b58f112`.
- Evidência real NO-GO: `.codex-tmp/style_copy_v2_validation_20260731_195900`.
- Baseline observado: 43/43 owners com perfil V2 em `fallback`, zero atributo aplicado, 559 abstenções, zero raster contract.
- 42/43 owners têm observação selecionada com confiança OCR >= 0.70: a confiança é perdida na integração.
- O gate atual reportou `PASS` apesar de zero contratos raster; esse resultado é inválido.
- O checkout inicial está em `Troca_de_motores` e contém milhares de mudanças locais alheias. Elas pertencem ao usuário.
- `pipeline/strip/process_bands.py`, `pipeline/tests/test_owner_enforcement.py`, `pipeline/tools/validate_owner_visual_matrix.py` e `pipeline/tests/test_owner_visual_matrix_tool.py` já têm mudanças locais relevantes. Preservá-las por hunk.

## Regras obrigatórias de execução

- Use `executing-plans` para conduzir o plano; em cada task use `test-driven-development` e, ao encontrar comportamento inesperado, `systematic-debugging`.
- Use `traduzai-typesetting` nas Tasks 2-9 e `traduzai-pipeline` nas Tasks 5-14.
- Use `verification-before-completion` antes de declarar o resultado final.
- Execute no checkout atual. Não use `reset`, `checkout`, `restore`, `stash`, `clean`, rebase destrutivo nem worktree nova.
- Antes de editar cada conjunto de arquivos, rode `git status --short -- <paths>` e leia `git diff -- <paths>` e `git diff --cached -- <paths>`.
- Faça toda edição manual com `apply_patch`.
- Siga TDD estrito: escrever um teste focal, executar e observar a falha esperada, implementar o mínimo, executar GREEN, rodar regressões adjacentes, auditar o diff e só então commitar.
- Não aceite um teste RED que falhe por import acidental, fixture quebrada ou erro de sintaxe quando a intenção era provar comportamento. Corrija o teste até ele falhar pela razão contratual esperada.
- Para arquivos já modificados, use `git add -p -- <arquivo>` e revise cada hunk. Para novos arquivos integrais, use `git add -- <arquivo>`. Nunca use `git add -A`, `git add .` ou commit por diretório amplo.
- Antes de cada commit: `git diff --cached --check`, `git diff --cached --stat`, `git diff --cached --name-only` e leitura integral do diff staged.
- Nenhuma regra pode depender de obra, capítulo, número de página, `band_id`, texto literal, coordenada fixa, cor específica de uma página ou nome de arquivo de um caso.
- `OwnerStyleCapture` e `OwnerVisualProfileV2` são sidecars. Nunca entram no hash semântico, payload de tradução ou identidade de `TextOwner`.
- Confiança não é transferível entre atributos. Confiança ausente, não finita ou sem provenance significa abstention.
- SFX só é elegível após promoção explícita no owner graph; candidate visual bruto não autoriza estilo.
- Não baixe fontes. Use apenas catálogo local versionado e com provenance/licença.
- Contrato ausente, incompleto, adulterado ou incompatível nunca é convertido em `{}` ou fallback silencioso.
- Erros de inglês residual, OCR, tradução, inpaint ou layout permanecem no gate funcional e não alteram score de estilo.
- Use raízes novas sob `.codex-tmp` para toda execução. Nunca sobrescreva a evidência NO-GO existente.
- Não declare GO com base apenas em testes, exit code ou auditoria automática; inspeção visual owner-scoped em escala nativa é obrigatória.

## Comando padrão e preflight por task

Use o interpretador fixado:

```powershell
$py = (Resolve-Path "pipeline\venv\Scripts\python.exe").Path
& $py --version
if ($LASTEXITCODE -ne 0) { throw "Python runtime unavailable" }
```

Antes de cada task:

```powershell
git status --short -- <paths-da-task>
git diff -- <paths-da-task>
git diff --cached -- <paths-da-task>
```

Se um arquivo contiver hunks do usuário na mesma região, adapte o patch sem apagá-los e stage somente os hunks desta task. Só pare se for impossível separar semanticamente as duas intenções; não reverta a mudança local.

## Checkpoints sequenciais

1. **R0 — baseline executável:** Tasks 0-1. Estado e conjunto de falhas congelados; probe reproduz o falso PASS como NO-GO.
2. **R1 — pixels vinculados:** Tasks 2-5. Contrato raster tipado atravessa single/multi-region, commit atômico e `project.json`.
3. **R2 — evidência real aplicada:** Tasks 6-9. Elegibilidade, máscaras, extractor, matcher, decisão e rasterização formam uma cadeia completa.
4. **R3 — QA honesto:** Tasks 10-12. Auditor, composição de gates, benchmark e matriz falham fechados e usam owners exatos.
5. **R4 — validação real:** Tasks 13-14. Suites, diferencial amplo, holdout novo e inspeção visual determinam Style/Functional/Overall GO ou NO-GO.

### Task 0: Preflight imutável e baseline diferencial

**Files:** None tracked.

**Step 1: Registrar branch, HEAD e mudanças sem tocar nelas**

```powershell
$baselineRoot = ".codex-tmp\style_copy_v2_remediation_baseline_20260731"
New-Item -ItemType Directory -Force $baselineRoot | Out-Null
git branch --show-current | Set-Content "$baselineRoot\branch.txt" -Encoding utf8
git rev-parse HEAD | Set-Content "$baselineRoot\head.txt" -Encoding ascii
git status --short --branch | Set-Content "$baselineRoot\status.txt" -Encoding utf8
git diff --stat | Set-Content "$baselineRoot\diff-stat.txt" -Encoding utf8
git diff --check | Set-Content "$baselineRoot\diff-check.txt" -Encoding utf8
git diff --cached --check | Set-Content "$baselineRoot\cached-diff-check.txt" -Encoding utf8
```

Expected: os artefatos são criados fora do índice. Falhas de `diff --check` preexistentes são registradas, não corrigidas por esta task.

**Step 2: Congelar a suite ampla atual para comparação, sem exigir verde global**

```powershell
$py = (Resolve-Path "pipeline\venv\Scripts\python.exe").Path
& $py -m pytest pipeline/tests -q `
  --junitxml="$baselineRoot\pytest-before.xml" `
  2>&1 | Tee-Object "$baselineRoot\pytest-before.log"
$pytestBeforeExit = $LASTEXITCODE
$pytestBeforeExit | Set-Content "$baselineRoot\pytest-before.exit.txt" -Encoding ascii
```

Expected: preservar o conjunto real de failures/errors/skips. O snapshot conhecido era 3543 passed, 33 skipped e 249 failed, mas o XML recém-gerado é a fonte de verdade desta execução. Não tente “limpar” failures alheias.

**Step 3: Validar o artefato NO-GO existente sem modificá-lo**

```powershell
$noGoRoot = Resolve-Path ".codex-tmp\style_copy_v2_validation_20260731_195900"
Get-ChildItem $noGoRoot -Recurse -Filter project.json |
  Sort-Object FullName |
  Select-Object -ExpandProperty FullName |
  Set-Content "$baselineRoot\no-go-projects.txt" -Encoding utf8
```

Expected: três projetos reais listados. Se o root não existir, a task continua usando os reports versionados e fixtures, mas registra explicitamente que a reprodução live ficou pendente.

**Step 4: Checkpoint**

Não há commit. Registre no diário de execução o caminho de `$baselineRoot`, HEAD, exit code e contagens do JUnit.

### Task 1: Criar um probe executável que torne o NO-GO impossível de esconder

**Files:**

- Create: `pipeline/debug_tools/style_runtime_probe.py`
- Create: `pipeline/tests/test_style_runtime_probe.py`
- Create: `pipeline/tests/fixtures/style_copy_v2_remediation/no_go_summary.json`

**Step 1: Escrever o teste RED**

```python
def test_probe_blocks_rendered_owner_without_applied_style_or_raster_contract():
    project = project_with_one_rendered_owner(
        profile_status="fallback",
        applied_attributes={},
        raster_contract=None,
    )

    report = probe_style_runtime(project)

    assert report["rendered_owner_count"] == 1
    assert report["raster_contract_coverage"] == 0.0
    assert report["status"] == "BLOCK"
    assert "missing_raster_contract" in report["contracts"]


def test_probe_does_not_claim_category_coverage_without_matching_owner():
    report = probe_style_runtime(
        project_with_one_rendered_owner(categories=["speech"]),
        required_categories={"sfx": 1},
    )

    assert report["category_metrics"]["sfx"]["owner_count"] == 0
    assert report["status"] == "BLOCK"
```

The fixture JSON records only portable counts and reasons from the previous run, never absolute `.codex-tmp` paths.

**Step 2: Confirmar RED**

```powershell
$py = (Resolve-Path "pipeline\venv\Scripts\python.exe").Path
& $py -m pytest pipeline/tests/test_style_runtime_probe.py -q
if ($LASTEXITCODE -eq 0) { throw "Expected RED: runtime probe does not exist yet" }
```

Expected: failure because `debug_tools.style_runtime_probe` is not implemented.

**Step 3: Implementar o probe mínimo**

Implement `probe_style_runtime(project, required_categories=None)` and a CLI that accepts one or more `project.json` paths. Derive the denominator from renderable owner-graph owners and report:

- rendered/profile/applied/fallback/review counts;
- applied and abstained attribute counts/reasons;
- raster-contract count and coverage;
- missing/mismatched owner IDs;
- owner-scoped category counts;
- `PASS` only when coverage and required categories are complete.

The probe is diagnostic and must not mutate projects.

**Step 4: Confirmar GREEN e reproduzir o baseline**

```powershell
$py = (Resolve-Path "pipeline\venv\Scripts\python.exe").Path
& $py -m pytest pipeline/tests/test_style_runtime_probe.py -q
if ($LASTEXITCODE -ne 0) { throw "Runtime probe tests failed" }

$projects = Get-Content ".codex-tmp\style_copy_v2_remediation_baseline_20260731\no-go-projects.txt"
if ($projects) {
  & $py pipeline/debug_tools/style_runtime_probe.py `
    --output ".codex-tmp\style_copy_v2_remediation_baseline_20260731\runtime-probe.json" `
    $projects
  if ($LASTEXITCODE -ne 2) { throw "Existing NO-GO must exit 2" }
}
```

Expected real summary: 43 rendered, 43 fallback, zero applied attributes, 559 abstentions, zero raster contracts, `BLOCK`.

**Step 5: Commit restrito**

```powershell
git add -- `
  pipeline/debug_tools/style_runtime_probe.py `
  pipeline/tests/test_style_runtime_probe.py `
  pipeline/tests/fixtures/style_copy_v2_remediation/no_go_summary.json
git diff --cached --check
git diff --cached --name-only
git commit -m "test(style): freeze runtime no-go contract"
```

**Checkpoint R0:** O probe deve reproduzir o run anterior como `BLOCK` com 43 fallbacks, zero atributos aplicados e zero contratos raster. O XML amplo e o status do checkout permanecem preservados para comparação final.

### Task 2: Definir e validar o contrato raster tipado

**Files:**

- Modify: `pipeline/ownership/model.py`
- Modify: `pipeline/typesetter/style_contract.py`
- Create: `pipeline/tests/style_v2_fixtures.py`
- Modify: `pipeline/tests/test_style_contract.py`

**Step 1: Escrever os testes RED de contrato e imutabilidade**

```python
def test_owner_style_raster_contract_is_canonical_hash_bound_and_immutable():
    contract = valid_owner_style_raster_contract(
        owner_id="owner_a",
        visual_profile_sha256="1" * 64,
        execution_component_geometry_sha256="2" * 64,
    )

    normalized = validate_owner_style_raster_contract(contract)

    assert normalized["contract_sha256"] == owner_style_raster_contract_sha256(normalized)
    with pytest.raises(TypeError):
        contract.applied_attributes["fill"] = "#ffffff"
```

Add explicit validation tests for missing schema, wrong owner/page, malformed hashes, non-canonical attributes, duplicate segment IDs, and self-hash tampering.

**Step 2: Confirmar RED**

```powershell
$py = (Resolve-Path "pipeline\venv\Scripts\python.exe").Path
& $py -m pytest `
  pipeline/tests/test_style_contract.py `
  -k "raster_contract" -q
if ($LASTEXITCODE -eq 0) { throw "Expected RED: raster contract is not modeled" }
```

Expected: missing contract type/serializer/validator.

**Step 3: Implementar o contrato mínimo**

Add a frozen `OwnerStyleRasterContract` and canonical serializer/hash/validator in `ownership/model.py`, where `OwnerGlyphPatch` can depend on it without an ownership-to-typesetter import cycle. Required fields:

- schema, page, owner, profile/source/mask hashes;
- `profile_component_geometry_sha256` and `execution_component_geometry_sha256` as separate fields;
- status, backend/version/capabilities;
- requested/applied/abstained attributes;
- core/effect envelopes, metrics, segments;
- before/patch/after hashes and contract self-hash.

Never compare the two geometry hashes to each other. Validate each against its own source later. Freeze nested mappings/sequences as well as arrays. Create `style_v2_fixtures.py` as the single test builder for a valid contract. It must create hashes from actual fixture arrays; no all-zero placeholder is accepted by validators.

Do not change `OwnerGlyphPatch` in this task. The transport field and every constructor change land atomically with the real renderer in Task 3, so this commit remains green and production cannot temporarily construct an invalid patch.

**Step 4: Confirmar GREEN e regressão de modelos**

```powershell
$py = (Resolve-Path "pipeline\venv\Scripts\python.exe").Path
& $py -m pytest `
  pipeline/tests/test_style_contract.py -q
if ($LASTEXITCODE -ne 0) { throw "Raster contract model regression" }
```

**Step 5: Commit por hunk**

```powershell
git add -p -- pipeline/ownership/model.py pipeline/typesetter/style_contract.py
git add -- pipeline/tests/style_v2_fixtures.py
git add -p -- pipeline/tests/test_style_contract.py
git diff --cached --check
git diff --cached --name-only
git commit -m "feat(style): define immutable owner raster contract"
```

### Task 3: Fazer o renderer emitir contrato vinculado aos pixels reais

**Files:**

- Modify: `pipeline/typesetter/renderer.py`
- Modify: `pipeline/ownership/model.py`
- Modify: `pipeline/tests/test_typesetting_renderer.py`
- Modify: `pipeline/tests/test_owner_enforcement.py`
- Modify: `pipeline/tests/test_owner_compositor.py`
- Modify: `pipeline/tests/test_strip_process_bands.py`
- Modify: `pipeline/tests/test_strip_owner_composition_integration.py`
- Modify: `pipeline/tests/test_owner_atomic_execution.py`

**Step 1: Escrever testes RED single-region**

```python
def test_owner_renderer_returns_profile_bound_raster_contract_with_pixels():
    patch = render_owner_with_v2_profile(profile=applied_profile(fill="#f4f4f4"))

    contract = patch.style_raster_contract.to_dict()
    assert contract["owner_id"] == patch.owner_id
    assert contract["visual_profile_sha256"] == PROFILE_SHA
    assert contract["execution_component_geometry_sha256"] == patch.component_geometry_sha256
    assert contract["applied_attributes"]["fill"] == "#f4f4f4"
    assert contract["rendered_patch_sha256"] == hash_masked_pixels(
        patch.result_rgb, patch.glyph_mask
    )
    assert contract["rendered_after_sha256"] == patch.after_sha256


def test_render_contract_is_returned_not_only_written_to_copied_text_block():
    patch, original_layout_record = render_without_observing_internal_copy()
    assert patch.style_raster_contract is not None
    assert "style_v2_raster_contract" not in original_layout_record


def test_owner_glyph_patch_requires_style_raster_contract():
    with pytest.raises(TypeError, match="style raster contract"):
        valid_owner_glyph_patch(style_raster_contract=None)
```

Also assert fallback/review contracts contain no falsely applied attributes and every omission has a reason.

**Step 2: Confirmar RED**

```powershell
$py = (Resolve-Path "pipeline\venv\Scripts\python.exe").Path
& $py -m pytest `
  pipeline/tests/test_typesetting_renderer.py `
  pipeline/tests/test_owner_compositor.py `
  -k "profile_bound_raster_contract or returned_not_only or glyph_patch_requires" -q
if ($LASTEXITCODE -eq 0) { throw "Expected RED: renderer loses contract at copy boundary" }
```

Expected: the internal text dict may contain metadata, but `OwnerGlyphPatch` does not receive the real contract.

**Step 3: Implementar o retorno estruturado**

Refactor `_render_v2_owner_text_layer()` so its output includes canonical applied/abstained attributes and pixel envelopes rather than relying on a side effect on `text_data`. Let `_render_owner_band_image()` build the final contract only after the real patch, mask, safe polygon, backend, quality metrics, and before/after hashes exist. Pass that exact immutable object into `OwnerGlyphPatch`.

The contract's `visual_profile_sha256` is validated against the attached profile. `profile_component_geometry_sha256` comes from the profile. `execution_component_geometry_sha256` comes from `_owner_component_geometry_sha256()` and equals the patch execution hash.

Add the required `style_raster_contract` field to `OwnerGlyphPatch` in the same patch. Update every constructor through `style_v2_fixtures.py`:

- the production constructor in `typesetter/renderer.py`;
- the direct fake renderer in `test_owner_enforcement.py`;
- `test_owner_compositor.py`;
- `test_strip_process_bands.py`;
- `test_strip_owner_composition_integration.py`;
- both dynamic factories in `test_owner_atomic_execution.py`.

Every fixture contract must bind to its real owner, arrays, and execution geometry.

**Step 4: Confirmar GREEN e goldens adjacentes**

```powershell
$py = (Resolve-Path "pipeline\venv\Scripts\python.exe").Path
& $py -m pytest `
  pipeline/tests/test_typesetting_renderer.py `
  pipeline/tests/test_renderer_backend_parity.py `
  pipeline/tests/test_owner_enforcement.py `
  pipeline/tests/test_owner_compositor.py `
  pipeline/tests/test_owner_atomic_execution.py `
  pipeline/tests/test_strip_owner_composition_integration.py `
  -k "style or raster or owner or backend" -q
if ($LASTEXITCODE -ne 0) { throw "Renderer contract tests failed" }
```

**Step 5: Commit restrito**

```powershell
git add -p -- `
  pipeline/typesetter/renderer.py `
  pipeline/ownership/model.py `
  pipeline/tests/test_typesetting_renderer.py `
  pipeline/tests/test_owner_enforcement.py `
  pipeline/tests/test_owner_compositor.py `
  pipeline/tests/test_strip_process_bands.py `
  pipeline/tests/test_strip_owner_composition_integration.py `
  pipeline/tests/test_owner_atomic_execution.py
git diff --cached --check
git commit -m "feat(style): bind raster contract to owner pixels"
```

### Task 4: Agregar contratos multi-region e fechar o bypass de texto curvo

**Files:**

- Modify: `pipeline/typesetter/renderer.py`
- Modify: `pipeline/tests/test_typesetting_renderer.py`
- Modify: `pipeline/tests/test_style_contract.py`

**Step 1: Escrever testes RED**

```python
def test_split_owner_aggregates_ordered_child_raster_contracts():
    patch = render_two_region_owner()
    contract = patch.style_raster_contract.to_dict()

    assert [row["segment_id"] for row in contract["segments"]] == ["region_0", "region_1"]
    assert all(row["rendered_patch_sha256"] for row in contract["segments"])
    assert contract["rendered_patch_sha256"] == hash_parent_patch(patch)


def test_split_owner_fails_closed_when_one_child_has_no_contract():
    with pytest.raises(ValueError, match="child raster contract"):
        aggregate_split_render_blocks([valid_child(), child_without_contract()])


def test_curved_owner_cannot_claim_v2_applied_without_supported_contract():
    patch = render_curved_owner_with_v2_profile(curve_capability=False)
    assert patch.style_raster_contract.status == "review_required"
    assert not patch.style_raster_contract.applied_attributes
```

Add permutation tests: changing child input order produces the same canonical ordered segment list and parent hash.

**Step 2: Confirmar RED**

```powershell
$py = (Resolve-Path "pipeline\venv\Scripts\python.exe").Path
& $py -m pytest pipeline/tests/test_typesetting_renderer.py `
  pipeline/tests/test_style_contract.py `
  -k "split_owner or child_raster or curved_owner" -q
if ($LASTEXITCODE -eq 0) { throw "Expected RED: aggregate/curve paths lose V2 contract" }
```

**Step 3: Implementar agregação e rota explícita**

Update `_copy_render_debug_fields()` and `_aggregate_split_render_blocks()` to carry child result contracts into deterministic parent segments. Reject missing, duplicate, cross-owner, cross-profile, overlapping, or hash-invalid segments.

Route curved rendering through capability negotiation. If the selected backend cannot produce a complete curve contract, return an explicit non-applied `review_required` result; do not execute the old early-return as a successful V2 render.

**Step 4: Confirmar GREEN**

```powershell
$py = (Resolve-Path "pipeline\venv\Scripts\python.exe").Path
& $py -m pytest `
  pipeline/tests/test_typesetting_renderer.py `
  pipeline/tests/test_style_contract.py `
  pipeline/tests/test_renderer_backend_parity.py -q
if ($LASTEXITCODE -ne 0) { throw "Multi-region/curve contract regression" }
```

**Step 5: Commit**

```powershell
git add -p -- `
  pipeline/typesetter/renderer.py `
  pipeline/tests/test_typesetting_renderer.py `
  pipeline/tests/test_style_contract.py
git diff --cached --check
git commit -m "fix(style): preserve contracts across render routes"
```

### Task 5: Validar o contrato no commit atômico e persistir somente pixels commitados

**Files:**

- Modify: `pipeline/strip/process_bands.py`
- Modify: `pipeline/ownership/model.py`
- Modify: `pipeline/tests/test_owner_atomic_execution.py`
- Modify: `pipeline/tests/test_owner_enforcement.py`
- Modify: `pipeline/tests/test_strip_process_bands.py`
- Modify: `pipeline/tests/test_project_writer.py`
- Modify: `pipeline/tests/test_strip_owner_composition_integration.py`

**Step 1: Escrever testes RED de rollback e persistência**

```python
@pytest.mark.parametrize("tamper", [
    "owner_id", "visual_profile_sha256", "execution_geometry",
    "patch_sha256", "after_sha256", "contract_sha256",
])
def test_atomic_commit_rolls_back_tampered_style_contract(tamper):
    original, mutation, patch = valid_atomic_chain()
    patch = tamper_style_contract(patch, tamper)

    commit = apply_atomic_owner_execution(original, mutation, patch)

    assert commit.committed is False
    assert commit.review_required is True
    assert np.array_equal(commit.result_rgb, original)
    assert commit.reason.startswith("render_contract_invalid:")


def test_successful_commit_persists_exact_raster_contract_to_final_record():
    execution = execute_single_owner_with_valid_style_contract()
    layer = execution.records[0]

    assert layer["style_v2_raster_contract"] == (
        execution.commits[0].glyph_patch.style_raster_contract.to_dict()
    )


def test_project_writer_round_trips_style_raster_contract(tmp_path):
    path = tmp_path / "project.json"
    write_project_json_atomic(path, project_with_valid_style_contract())
    assert read_layer(path)["style_v2_raster_contract"]["contract_sha256"] == CONTRACT_SHA
```

Include a test proving an uncommitted/review owner never persists an applied contract.

**Step 2: Confirmar RED**

```powershell
$py = (Resolve-Path "pipeline\venv\Scripts\python.exe").Path
& $py -m pytest `
  pipeline/tests/test_owner_atomic_execution.py `
  pipeline/tests/test_strip_process_bands.py `
  pipeline/tests/test_project_writer.py `
  -k "style_contract or raster_contract" -q
if ($LASTEXITCODE -eq 0) { throw "Expected RED: commit/persistence does not validate raster contract" }
```

**Step 3: Implementar validação no ponto de autoridade**

Extend `apply_atomic_owner_execution()` after the existing glyph patch validation. Validate contract schema/self-hash, owner/page, `visual_profile_sha256` supplied by the execution context, profile geometry, execution geometry, decision parity, patch hash, candidate final hash, and envelopes. If changing the function signature is necessary, pass the validated profile/expected profile hash explicitly; never look it up through ambient globals.

In `execute_owner_page_graph()`, construct `rendered_record` from the committed patch contract, not from the renderer's copied input block. Attach it only in the `commit.committed` branch. Preserve the exact rollback reason in the review record.

The existing changes in `process_bands.py` and `test_owner_enforcement.py` must remain. Stage only contract-specific hunks.

**Step 4: Confirmar GREEN e integração**

```powershell
$py = (Resolve-Path "pipeline\venv\Scripts\python.exe").Path
& $py -m pytest `
  pipeline/tests/test_owner_atomic_execution.py `
  pipeline/tests/test_owner_enforcement.py `
  pipeline/tests/test_strip_process_bands.py `
  pipeline/tests/test_strip_owner_composition_integration.py `
  pipeline/tests/test_project_writer.py -q
if ($LASTEXITCODE -ne 0) { throw "Atomic raster persistence regression" }
```

**Step 5: Commit por hunk**

```powershell
git add -p -- `
  pipeline/strip/process_bands.py `
  pipeline/ownership/model.py `
  pipeline/tests/test_owner_atomic_execution.py `
  pipeline/tests/test_owner_enforcement.py `
  pipeline/tests/test_strip_process_bands.py `
  pipeline/tests/test_project_writer.py `
  pipeline/tests/test_strip_owner_composition_integration.py
git diff --cached --check
git commit -m "fix(style): enforce raster contract at atomic commit"
```

**Checkpoint R1:** Run the Task 2-5 suites together. A rendered owner without a valid raster contract must now roll back; a successful owner must round-trip the exact contract.

### Task 6: Criar a elegibilidade owner-scoped antes da tradução

**Files:**

- Create: `pipeline/typesetter/style_capture.py`
- Create: `pipeline/tests/test_owner_style_capture.py`
- Modify: `pipeline/strip/process_bands.py`
- Modify: `pipeline/strip/run.py`
- Modify: `pipeline/ownership/translation.py`
- Modify: `pipeline/tests/test_owner_translation.py`
- Modify: `pipeline/tests/test_owner_style_profile.py`
- Modify: `pipeline/tests/test_strip_owner_control_plane.py`

**Step 1: Escrever testes RED de origem da confiança**

```python
def test_primary_owner_capture_derives_confidence_from_selected_observations():
    graph = graph_with_selected_confidences(0.91, 0.84)
    capture = build_owner_style_capture(graph, owner_id="owner_a", source_rgb=SOURCE)

    assert capture.candidate_kind == "primary_ocr"
    assert capture.candidate_confidence == pytest.approx(0.91)
    assert capture.candidate_confidence_provenance == ("obs_a", "obs_b")


def test_promoted_sfx_requires_explicit_promotion_provenance():
    assert build_capture(raw_sfx_candidate()).eligible is False
    promoted = build_capture(promoted_sfx_candidate(confidence=0.88))
    assert promoted.eligible is True
    assert promoted.promotion_provenance


def test_translation_payload_remains_free_of_visual_capture_fields():
    payload = owners_to_translation_page(graph_with_style_capture())
    forbidden = {
        "owner_style_capture", "style_evidence_v2", "candidate_confidence",
        "glyph_mask_sha256", "font_match_evidence",
    }
    assert forbidden.isdisjoint(payload)
    assert all(forbidden.isdisjoint(row) for row in payload["texts"])
```

Add order/permutation tests and missing/non-finite confidence tests.

**Step 2: Confirmar RED**

```powershell
$py = (Resolve-Path "pipeline\venv\Scripts\python.exe").Path
& $py -m pytest `
  pipeline/tests/test_owner_style_capture.py `
  pipeline/tests/test_owner_translation.py `
  -q
if ($LASTEXITCODE -eq 0) { throw "Expected RED: owner style capture is not connected" }
```

**Step 3: Implementar o sidecar e a fronteira limpa**

Create a frozen `OwnerStyleCapture` with canonical serialization/hash. Derive primary OCR confidence deterministically from selected observations and keep all observation IDs. Carry SFX promotion provenance from graph construction in `strip/run.py` into the sidecar. Never reconstruct style eligibility from translated records.

In `execute_owner_page_graph()`, build captures before calling the translator and retain them in a local `captures_by_owner` mapping. After translation, reattach by stable owner ID. Leave `ownership/translation.py` intentionally clean and strengthen its forbidden-field tests.

**Step 4: Confirmar GREEN e semantic invariance**

```powershell
$py = (Resolve-Path "pipeline\venv\Scripts\python.exe").Path
& $py -m pytest `
  pipeline/tests/test_owner_style_capture.py `
  pipeline/tests/test_owner_translation.py `
  pipeline/tests/test_owner_style_profile.py `
  pipeline/tests/test_strip_owner_control_plane.py `
  -q
if ($LASTEXITCODE -ne 0) { throw "Style capture/translation boundary regression" }
```

**Step 5: Commit**

```powershell
git add -- pipeline/typesetter/style_capture.py pipeline/tests/test_owner_style_capture.py
git add -p -- `
  pipeline/strip/process_bands.py `
  pipeline/strip/run.py `
  pipeline/ownership/translation.py `
  pipeline/tests/test_owner_translation.py `
  pipeline/tests/test_owner_style_profile.py `
  pipeline/tests/test_strip_owner_control_plane.py
git diff --cached --check
git commit -m "feat(style): preserve owner eligibility outside translation"
```

### Task 7: Conectar o extractor V2 a pixels e máscaras pre-inpaint reais

**Files:**

- Modify: `pipeline/typesetter/style_capture.py`
- Modify: `pipeline/typesetter/style_extractor.py`
- Modify: `pipeline/typesetter/owner_style.py`
- Modify: `pipeline/strip/process_bands.py`
- Modify: `pipeline/tests/test_style_extractor.py`
- Modify: `pipeline/tests/test_style_mask_evidence.py`
- Modify: `pipeline/tests/test_owner_style_capture.py`
- Modify: `pipeline/tests/test_owner_style_profile.py`
- Modify: `pipeline/tests/test_owner_enforcement.py`

**Step 1: Escrever testes RED de máscara e chamada runtime**

```python
def test_capture_extracts_fill_and_stroke_from_authoritative_owner_masks():
    capture = capture_colored_text_fixture()
    evidence = capture.style_evidence_v2

    assert evidence.attributes["fill"].status == "measured"
    assert evidence.attributes["fill"].provenance == ("owner_glyph_core",)
    assert evidence.attributes["stroke_color"].status == "measured"


def test_context_mask_excludes_foreign_owner_and_card_art():
    masks = build_style_capture_masks(graph_with_neighboring_owner_and_art())
    assert not np.any(masks.context & masks.foreign_owner)
    assert not np.any(masks.context & masks.glyph_core)


def test_execute_owner_page_graph_calls_v2_extractor_before_inpaint(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "typesetter.style_capture.extract_text_style_evidence_v2",
        lambda *args, **kwargs: calls.append(kwargs) or measured_evidence(),
    )
    execute_owner_fixture()
    assert calls and calls[0]["source_phase"] == "pre_inpaint"
```

Add negative fixtures for plain white balloon, colored art next to white text, empty masks, too-small support, and foreign-owner overlap.

**Step 2: Confirmar RED**

```powershell
$py = (Resolve-Path "pipeline\venv\Scripts\python.exe").Path
& $py -m pytest `
  pipeline/tests/test_style_extractor.py `
  pipeline/tests/test_style_mask_evidence.py `
  pipeline/tests/test_owner_style_capture.py `
  -k "authoritative or context_mask or pre_inpaint" -q
if ($LASTEXITCODE -eq 0) { throw "Expected RED: runtime does not use V2 masked extractor" }
```

**Step 3: Implementar máscaras e integração**

Build page-shaped canonical masks from selected owner glyph/support polygons. Expand context/effect envelopes relative to x-height and clip them to the owner/component bounds. Subtract glyph/stroke masks, all foreign owner glyph masks, protected art, and invalid support. Hash every mask.

Call `extract_text_style_evidence_v2()` during capture, before mutation. Replace the legacy empty `_style_evidence()` route for captured owners. Keep a legacy adapter only for explicit non-V2/not-applicable paths; it cannot claim eligibility or PASS in enforce mode.

**Step 4: Confirmar GREEN e ausência de contaminação**

```powershell
$py = (Resolve-Path "pipeline\venv\Scripts\python.exe").Path
& $py -m pytest `
  pipeline/tests/test_style_extractor.py `
  pipeline/tests/test_style_mask_layers.py `
  pipeline/tests/test_style_mask_evidence.py `
  pipeline/tests/test_owner_style_capture.py `
  pipeline/tests/test_owner_style_profile.py `
  pipeline/tests/test_owner_enforcement.py `
  -q
if ($LASTEXITCODE -ne 0) { throw "Masked V2 extraction regression" }
```

**Step 5: Commit por hunk**

```powershell
git add -p -- `
  pipeline/typesetter/style_capture.py `
  pipeline/typesetter/style_extractor.py `
  pipeline/typesetter/owner_style.py `
  pipeline/strip/process_bands.py `
  pipeline/tests/test_style_extractor.py `
  pipeline/tests/test_style_mask_evidence.py `
  pipeline/tests/test_owner_style_capture.py `
  pipeline/tests/test_owner_style_profile.py `
  pipeline/tests/test_owner_enforcement.py
git diff --cached --check
git commit -m "feat(style): extract owner evidence before inpaint"
```

### Task 8: Conectar o font matcher e completar atributos geométricos

**Files:**

- Modify: `pipeline/typesetter/style_contract.py`
- Modify: `pipeline/typesetter/style_extractor.py`
- Modify: `pipeline/typesetter/font_matcher.py`
- Modify: `pipeline/typesetter/style_capture.py`
- Modify: `pipeline/typesetter/owner_style.py`
- Modify: `pipeline/tests/test_style_contract.py`
- Modify: `pipeline/tests/test_style_extractor.py`
- Modify: `pipeline/tests/test_font_matcher.py`
- Modify: `pipeline/tests/test_owner_style_profile.py`

**Step 1: Escrever testes RED**

```python
def test_v2_contract_exposes_tracking_slant_width_and_vertical_scale():
    assert {
        "tracking_xh", "slant_tangent", "width_scale", "scale_y"
    }.issubset(STYLE_V2_ATTRIBUTE_NAMES)


def test_owner_capture_matches_font_using_frozen_source_text_and_glyph_mask():
    capture = build_capture(source_text="DING", glyph_mask=DING_MASK)
    result = capture.font_match_evidence
    assert result["top_k"]
    assert result["source_text_sha256"] == sha256_text("DING")
    assert result["glyph_mask_sha256"] == hash_array(DING_MASK)


def test_ambiguous_font_match_abstains_instead_of_using_role_prior_as_confidence():
    result = matcher_with_scores(0.81, 0.80).match(...)
    assert result.selected_font is None
    assert result.abstention_reason == "insufficient_margin"
```

Add cache-key/catalog-version and permutation tests. Add extractor tests proving tracking/slant are stored as typed measured evidence, not provenance-only strings.

**Step 2: Confirmar RED**

```powershell
$py = (Resolve-Path "pipeline\venv\Scripts\python.exe").Path
& $py -m pytest `
  pipeline/tests/test_style_contract.py `
  pipeline/tests/test_style_extractor.py `
  pipeline/tests/test_font_matcher.py `
  -k "tracking or slant or vertical_scale or frozen_source or ambiguous" -q
if ($LASTEXITCODE -eq 0) { throw "Expected RED: matcher/geometry evidence incomplete" }
```

**Step 3: Implementar matcher runtime e schema completo**

Add the missing canonical attribute names and serialization. Convert existing tracking/slant measurements into `StyleAttributeEvidenceV2`. Normalize width/scale/rotation before silhouette comparison. Invoke `FontShapeMatcher` from the capture path with the frozen source text and glyph mask. Record catalog version, top-k, margin, normalization, cache key, and abstention reason.

The matcher may use semantic role only as a bounded prior. It may not download or discover unversioned fonts during a run.

**Step 4: Confirmar GREEN**

```powershell
$py = (Resolve-Path "pipeline\venv\Scripts\python.exe").Path
& $py -m pytest `
  pipeline/tests/test_style_contract.py `
  pipeline/tests/test_style_extractor.py `
  pipeline/tests/test_font_matcher.py `
  pipeline/tests/test_owner_style_profile.py -q
if ($LASTEXITCODE -ne 0) { throw "Font matcher/geometry evidence regression" }
```

**Step 5: Commit**

```powershell
git add -p -- `
  pipeline/typesetter/style_contract.py `
  pipeline/typesetter/style_extractor.py `
  pipeline/typesetter/font_matcher.py `
  pipeline/typesetter/style_capture.py `
  pipeline/typesetter/owner_style.py `
  pipeline/tests/test_style_contract.py `
  pipeline/tests/test_style_extractor.py `
  pipeline/tests/test_font_matcher.py `
  pipeline/tests/test_owner_style_profile.py
git diff --cached --check
git commit -m "feat(style): match fonts and measure geometry per owner"
```

### Task 9: Materializar e rasterizar todos os atributos aprovados sem silent drop

**Files:**

- Modify: `pipeline/typesetter/style_policy.py`
- Modify: `pipeline/typesetter/owner_style.py`
- Modify: `pipeline/typesetter/backend_contract.py`
- Modify: `pipeline/typesetter/renderer.py`
- Modify: `pipeline/sfx/renderer.py`
- Modify: `pipeline/tests/test_style_policy_v2.py`
- Modify: `pipeline/tests/test_owner_style_profile.py`
- Modify: `pipeline/tests/test_typesetting_backend_contract.py`
- Modify: `pipeline/tests/test_typesetting_renderer.py`
- Modify: `pipeline/tests/test_sfx_renderer.py`
- Modify: `pipeline/tests/test_renderer_backend_parity.py`

**Step 1: Escrever testes RED de decisão total**

```python
def test_candidate_gate_allows_independent_per_attribute_decisions():
    decision = decide_style_copy_v2(
        eligible_candidate(confidence=0.91),
        evidence(fill=high(), glow=low()),
    )
    assert "fill" in decision["applied_attributes"]
    assert decision["abstained_attributes"]["glow"]["reason"] == "low_confidence"


def test_materialization_maps_every_applied_canonical_attribute():
    decision = decision_applying_all_supported_attributes()
    style = materialize_style(candidate(), decision)
    assert set(decision["applied_attributes"]) == set(style["canonical_applied_attributes"])


def test_backend_cannot_silently_drop_requested_tracking_or_slant():
    with pytest.raises(UnsupportedStyleCapability):
        choose_backend(requested={"tracking_xh", "slant_tangent"}, backends=[limited_backend()])


def test_sfx_uses_selected_font_instead_of_hershey_override():
    raster = render_sfx(profile=profile(font_name="LeagueGothic..."))
    assert raster.contract.applied_attributes["font_name"].startswith("LeagueGothic")
    assert raster.contract.backend != "opencv_hershey_override"
```

Add raster tests for width, scale_y, tracking, slant, rotation, fill/gradient, multi-stroke, shadow, glow, and effect envelope.

**Step 2: Confirmar RED**

```powershell
$py = (Resolve-Path "pipeline\venv\Scripts\python.exe").Path
& $py -m pytest `
  pipeline/tests/test_style_policy_v2.py `
  pipeline/tests/test_owner_style_profile.py `
  pipeline/tests/test_typesetting_backend_contract.py `
  pipeline/tests/test_typesetting_renderer.py `
  pipeline/tests/test_sfx_renderer.py `
  -k "independent or every_applied or silently_drop or selected_font" -q
if ($LASTEXITCODE -eq 0) { throw "Expected RED: decision/materialization can drop attributes" }
```

**Step 3: Implementar policy, materialization e capability handshake**

Move the global candidate eligibility check ahead of, but separate from, per-attribute gates. Expand `_materialize_style()` to every canonical field. Return a canonical applied-field manifest alongside renderer-specific aliases.

Make backends advertise exact supported fields and transforms. Select a capable backend before drawing; no post-hoc deletion. Remove/bypass the Latin SFX Hershey override when V2 supplies a font. When no backend can honor a high-confidence requested field, produce an explicit blocking/review contract.

**Step 4: Confirmar GREEN e parity**

```powershell
$py = (Resolve-Path "pipeline\venv\Scripts\python.exe").Path
& $py -m pytest `
  pipeline/tests/test_style_policy_v2.py `
  pipeline/tests/test_owner_style_profile.py `
  pipeline/tests/test_typesetting_backend_contract.py `
  pipeline/tests/test_typesetting_renderer.py `
  pipeline/tests/test_sfx_renderer.py `
  pipeline/tests/test_renderer_backend_parity.py `
  pipeline/tests/test_style_groups.py -q
if ($LASTEXITCODE -ne 0) { throw "Style materialization/raster regression" }
```

**Step 5: Commit**

```powershell
git add -p -- `
  pipeline/typesetter/style_policy.py `
  pipeline/typesetter/owner_style.py `
  pipeline/typesetter/backend_contract.py `
  pipeline/typesetter/renderer.py `
  pipeline/sfx/renderer.py `
  pipeline/tests/test_style_policy_v2.py `
  pipeline/tests/test_owner_style_profile.py `
  pipeline/tests/test_typesetting_backend_contract.py `
  pipeline/tests/test_typesetting_renderer.py `
  pipeline/tests/test_sfx_renderer.py `
  pipeline/tests/test_renderer_backend_parity.py
git diff --cached --check
git commit -m "feat(style): apply approved attributes without silent fallback"
```

**Checkpoint R2:** Re-run Tasks 6-9 plus `test_owner_translation.py`. A high-confidence colored-card fixture must apply at least one measured attribute and persist a matching raster contract; a plain balloon negative must not acquire false effects; translation and semantic hashes remain invariant.

### Task 10: Reescrever o auditor de estilo como owner-scoped, canonical e fail-closed

**Files:**

- Create: `pipeline/qa/style_fidelity_policy.py`
- Modify: `pipeline/qa/style_fidelity.py`
- Modify: `pipeline/tests/test_style_fidelity_qa.py`
- Modify: `pipeline/tests/test_style_copy_score.py`

**Step 1: Escrever matriz RED do auditor**

```python
def test_enforce_blocks_renderable_owner_without_raster_contract():
    report = audit_style_fidelity(project_missing_contract(), RUN, mode="enforce")
    assert report["gate"]["status"] == "BLOCK"
    assert report["coverage"]["contract"] == 0.0


@pytest.mark.parametrize("tamper", [
    "owner", "profile_hash", "evidence_hash", "execution_geometry",
    "patch_hash", "final_hash", "contract_hash",
])
def test_enforce_blocks_identity_or_hash_mismatch(tamper):
    report = audit_style_fidelity(tampered_project(tamper), RUN, mode="enforce")
    assert report["gate"]["status"] == "BLOCK"


def test_audit_compares_canonical_decision_to_canonical_raster():
    report = audit_style_fidelity(project_with_fill_mismatch(), RUN, mode="enforce")
    owner = report["owners"][0]
    assert owner["attributes"]["fill"]["status"] == "mismatch"


def test_low_confidence_abstention_is_explicit_but_not_individual_p0():
    report = audit_style_fidelity(project_with_valid_low_confidence_fallback(), RUN, mode="enforce")
    assert report["owners"][0]["attributes"]["glow"]["status"] == "abstained"
    assert not report["owners"][0]["catastrophic_mismatches"]


def test_empty_or_zero_denominator_report_never_passes_enforce():
    assert audit_style_fidelity(empty_project(), RUN, mode="enforce")["gate"]["status"] == "BLOCK"
```

Also test complete coverage, Python backend equivalence, segment contracts, category metrics, containment, and missing required metrics.

**Step 2: Confirmar RED**

```powershell
$py = (Resolve-Path "pipeline\venv\Scripts\python.exe").Path
& $py -m pytest pipeline/tests/test_style_fidelity_qa.py -q
if ($LASTEXITCODE -eq 0) { throw "Expected RED: current style audit is false-open" }
```

**Step 3: Implementar auditor e policy única**

Derive eligible/rendered owners from `page_owner_graphs`. Validate profiles with `validate_owner_visual_profile()` and raster contracts with the new validator. Compare `style_application_decision_v2.applied_attributes` to canonical raster attributes; never compare legacy `cor`/`fonte` aliases.

Recompute source-to-final metrics from bound source/final crops where feasible; do not trust only self-reported renderer numbers. Produce owner, attribute, category, coverage, fallback-reason, and catastrophic-mismatch metrics. Centralize thresholds and reason-to-severity mapping in `style_fidelity_policy.py`.

In `enforce`, missing profile/contract/metric, invalid hash, incomplete denominator, or required category with zero owners blocks. In `shadow`, preserve the same findings but do not alter export eligibility.

**Step 4: Confirmar GREEN**

```powershell
$py = (Resolve-Path "pipeline\venv\Scripts\python.exe").Path
& $py -m pytest `
  pipeline/tests/test_style_fidelity_qa.py `
  pipeline/tests/test_style_copy_score.py `
  pipeline/tests/test_style_audit_report.py -q
if ($LASTEXITCODE -ne 0) { throw "Owner-scoped style audit regression" }
```

**Step 5: Commit**

```powershell
git add -- pipeline/qa/style_fidelity_policy.py
git add -p -- `
  pipeline/qa/style_fidelity.py `
  pipeline/tests/test_style_fidelity_qa.py `
  pipeline/tests/test_style_copy_score.py
git diff --cached --check
git commit -m "fix(qa): make style fidelity owner scoped and fail closed"
```

### Task 11: Compor gates de forma normalizada e manter idioma separado de estilo

**Files:**

- Create: `pipeline/qa/gate_composition.py`
- Modify: `pipeline/qa/export_gate.py`
- Modify: `pipeline/qa/final_pixel_qa.py`
- Modify: `pipeline/main.py`
- Modify: `pipeline/tests/test_export_gate.py`
- Modify: `pipeline/tests/test_main_emit.py`
- Modify: `pipeline/tests/test_final_pixel_qa.py`
- Modify: `pipeline/tests/test_final_pixel_export_gate.py`
- Modify: `pipeline/tests/test_style_fidelity_qa.py`

**Step 1: Escrever testes RED de composição**

```python
@pytest.mark.parametrize(
    ("functional", "style", "expected"),
    [("PASS", "PASS", "PASS"), ("PASS", "BLOCK", "BLOCK"),
     ("BLOCK", "PASS", "BLOCK"), ("BLOCK", "BLOCK", "BLOCK")],
)
def test_export_gate_is_conjunction_of_normalized_subgates(functional, style, expected):
    gate = compose_export_gate(functional_gate(functional), style_gate(style))
    assert gate["status"] == expected
    assert gate["allowed"] is (expected == "PASS")


def test_style_audit_exception_becomes_qa_integrity_failure():
    result = run_main_with_style_audit_raising(mode="enforce")
    assert result.project["qa"]["export_gate"]["status"] == "BLOCK"
    assert "qa_integrity_failure" in issue_codes(result.project)


def test_english_residual_blocks_functional_but_not_style_subgate():
    qa = audit_project(source_english_visible_style_matching())
    assert qa["functional_export_gate"]["status"] == "BLOCK"
    assert qa["style_fidelity"]["gate"]["status"] == "PASS"


def test_style_mismatch_blocks_style_but_not_functional_subgate():
    qa = audit_project(ptbr_complete_style_mismatch())
    assert qa["functional_export_gate"]["status"] == "PASS"
    assert qa["style_fidelity"]["gate"]["status"] == "BLOCK"
```

Add tests for recomputed counts, override applied once, strict exit 2, normal blocked preview exit 0, and named entities not misclassified as generic English residual.

**Step 2: Confirmar RED**

```powershell
$py = (Resolve-Path "pipeline\venv\Scripts\python.exe").Path
& $py -m pytest `
  pipeline/tests/test_export_gate.py `
  pipeline/tests/test_main_emit.py `
  pipeline/tests/test_final_pixel_qa.py `
  pipeline/tests/test_final_pixel_export_gate.py `
  -k "subgate or integrity or english_residual or style_mismatch" -q
if ($LASTEXITCODE -eq 0) { throw "Expected RED: gate merge is inconsistent/fail-open" }
```

**Step 3: Implementar composição única**

Persist functional and style gates separately, then call one pure `compose_export_gate()` after both audits. Recompute `allowed`, issues, severity counts, block counts, and subgate summary. Apply any authorized override once and only after normalized composition.

In `main.py`, an exception or absent style report under `style_copy_mode=enforce` becomes `qa_integrity_failure`. Do not preserve prior `allowed=True` or stale counts. Keep final-language challenge pairing in functional QA; do not add generic language detection to style QA.

**Step 4: Confirmar GREEN e modos de saída**

```powershell
$py = (Resolve-Path "pipeline\venv\Scripts\python.exe").Path
& $py -m pytest `
  pipeline/tests/test_export_gate.py `
  pipeline/tests/test_export_gate_debug_consistency.py `
  pipeline/tests/test_main_emit.py `
  pipeline/tests/test_final_pixel_qa.py `
  pipeline/tests/test_final_pixel_export_gate.py `
  pipeline/tests/test_style_fidelity_qa.py -q
if ($LASTEXITCODE -ne 0) { throw "Gate composition regression" }
```

**Step 5: Commit**

```powershell
git add -- pipeline/qa/gate_composition.py
git add -p -- `
  pipeline/qa/export_gate.py `
  pipeline/qa/final_pixel_qa.py `
  pipeline/main.py `
  pipeline/tests/test_export_gate.py `
  pipeline/tests/test_main_emit.py `
  pipeline/tests/test_final_pixel_qa.py `
  pipeline/tests/test_final_pixel_export_gate.py `
  pipeline/tests/test_style_fidelity_qa.py
git diff --cached --check
git commit -m "fix(qa): compose functional and style gates fail closed"
```

### Task 12: Tornar thresholds, matriz owner-targeted e inspeção reproduzíveis

**Files:**

- Modify: `pipeline/debug_tools/style_benchmark_report.py`
- Modify: `pipeline/debug_tools/run_style_benchmark_v2.py`
- Modify: `pipeline/tools/validate_owner_visual_matrix.py`
- Create: `pipeline/tools/build_style_owner_target_manifest.py`
- Modify: `pipeline/tests/test_style_benchmark_v2_score.py`
- Modify: `pipeline/tests/test_owner_visual_matrix_tool.py`
- Create: `pipeline/tests/test_style_owner_target_manifest.py`
- Modify: `pipeline/tests/fixtures/style_benchmark_v2/benchmark_spec.json`
- Modify: `pipeline/tests/fixtures/style_copy_corpus/matrix.json`
- Create: `pipeline/tests/fixtures/style_copy_corpus/configs/colored_cards.style.json`
- Create: `pipeline/tests/fixtures/style_copy_corpus/configs/cross_tile_owners.style.json`
- Create: `pipeline/tests/fixtures/style_copy_corpus/configs/dark_panels.style.json`

**Step 1: Escrever testes RED de thresholds e targets exatos**

```python
def test_benchmark_enforce_returns_two_when_threshold_fails(tmp_path):
    result = run_benchmark_fixture(tmp_path, font_top1=0.50, mode="enforce")
    assert result.returncode == 2
    assert result.report["status"] == "BLOCK"


def test_matrix_rejects_out_of_range_page_instead_of_clamping(tmp_path):
    with pytest.raises(MatrixContractError, match="page target not found"):
        selected_owner_target(entry(page_id="page_055"), project_with_three_pages())


def test_matrix_rejects_missing_owner_or_zero_owner_category(tmp_path):
    result = validate_matrix(
        matrix_target(owner_id="missing", category="sfx"),
        project_without_sfx_owner(),
    )
    assert result["status"] == "BLOCK"
    assert "owner_target_not_found" in result["contracts"]
    assert result["category_metrics"]["sfx"]["owner_count"] == 0


def test_style_matrix_forces_enforce_without_mutating_functional_config(tmp_path):
    effective = build_effective_style_config(shared_config(), tmp_path)
    assert effective["style_copy_mode"] == "enforce"
    assert shared_config()["style_copy_mode"] != "enforce"


def test_validate_only_preserves_and_verifies_runner_manifest(tmp_path):
    first = run_matrix(tmp_path)
    second = validate_only(tmp_path)
    assert second["runner_evidence"] == first["runner_evidence"]
```

Add tests for inspection schema v2, real dimensions/hash, separate functional/style/overall verdicts, missing inspection=`PENDING`, missing metric/category denominator=`BLOCK`, and run-manifest hash tampering.

**Step 2: Confirmar RED**

```powershell
$py = (Resolve-Path "pipeline\venv\Scripts\python.exe").Path
& $py -m pytest `
  pipeline/tests/test_style_benchmark_v2_score.py `
  pipeline/tests/test_owner_visual_matrix_tool.py `
  pipeline/tests/test_style_owner_target_manifest.py -q
if ($LASTEXITCODE -eq 0) { throw "Expected RED: thresholds/targets/inspection are not enforceable" }
```

**Step 3: Implementar scorer e schema v2**

Consume `validation_thresholds` from `benchmark_spec.json`; absent metrics, zero denominator, absent required category, or threshold breach blocks and returns 2 in enforce mode.

Migrate the matrix from page-level `categories/category_pages` to exact targets containing `page_id`, `owner_id`, `component_ids`, category, split, source crop/hash, and minimum counts. Remove every clamp/default-to-last-page behavior. A helper tool may discover deterministic candidate owners from frozen graph artifacts, but the committed manifest is curated and exact; discovery is never accepted as inspection.

Create style-only config overlays that set `style_copy_mode=enforce`, inspection required, and style categories. Do not alter the shared functional owner-matrix configs.

Persist `run_manifest.json` with Git HEAD, scoped diff hash, manifest/input/config/tool hashes, effective config, Python/runtime lock, seed, exact command, timestamps, return codes, stdout/stderr hashes, project hash, and final-artifact hashes. Validate-only verifies and reuses it.

Inspection schema v2 is owner-scoped and shows `source | masks/evidence | requested | raster | final | safe-region`. Require native dimensions and independent functional/style notes. Overall is derived; incomplete inspection is `PENDING`.

The preexisting local removal of `under_source_scale` and its paired test must be preserved. Integrate new matrix logic around it and stage only remediation hunks.

**Step 4: Confirmar GREEN**

```powershell
$py = (Resolve-Path "pipeline\venv\Scripts\python.exe").Path
& $py -m pytest `
  pipeline/tests/test_style_benchmark_v2_score.py `
  pipeline/tests/test_style_benchmark_v2_generator.py `
  pipeline/tests/test_style_copy_benchmark.py `
  pipeline/tests/test_owner_visual_matrix_tool.py `
  pipeline/tests/test_style_owner_target_manifest.py -q
if ($LASTEXITCODE -ne 0) { throw "Benchmark/matrix contract regression" }
```

**Step 5: Commit por hunk**

```powershell
git add -p -- `
  pipeline/debug_tools/style_benchmark_report.py `
  pipeline/debug_tools/run_style_benchmark_v2.py `
  pipeline/tools/validate_owner_visual_matrix.py `
  pipeline/tests/test_style_benchmark_v2_score.py `
  pipeline/tests/test_owner_visual_matrix_tool.py `
  pipeline/tests/fixtures/style_benchmark_v2/benchmark_spec.json `
  pipeline/tests/fixtures/style_copy_corpus/matrix.json
git add -- `
  pipeline/tools/build_style_owner_target_manifest.py `
  pipeline/tests/test_style_owner_target_manifest.py `
  pipeline/tests/fixtures/style_copy_corpus/configs/colored_cards.style.json `
  pipeline/tests/fixtures/style_copy_corpus/configs/cross_tile_owners.style.json `
  pipeline/tests/fixtures/style_copy_corpus/configs/dark_panels.style.json
git diff --cached --check
git diff --cached --name-only
git commit -m "fix(style): enforce owner targeted holdout matrix"
```

**Checkpoint R3:** Use synthetic projects to prove all false-open paths now BLOCK. Validate that functional English failure and style mismatch produce independent subgate evidence. Validate zero-owner SFX/dark categories cannot pass.

### Task 13: Executar suites focadas, invariância funcional e diferencial amplo

**Files:**

- Modify only if a new failure is proven to be caused by Tasks 1-12; return to the owning task and add its regression test first.
- Create untracked evidence under a new `.codex-tmp/style_copy_v2_remediation_tests_<timestamp>` root.

**Step 1: Rodar suite Style V2 completa**

```powershell
$py = (Resolve-Path "pipeline\venv\Scripts\python.exe").Path
$testRoot = ".codex-tmp\style_copy_v2_remediation_tests_$(Get-Date -Format yyyyMMdd_HHmmss)"
New-Item -ItemType Directory -Force $testRoot | Out-Null

& $py -m pytest `
  pipeline/tests/test_style_contract.py `
  pipeline/tests/test_style_extractor.py `
  pipeline/tests/test_style_mask_layers.py `
  pipeline/tests/test_style_mask_evidence.py `
  pipeline/tests/test_font_matcher.py `
  pipeline/tests/test_style_policy_v2.py `
  pipeline/tests/test_owner_style_capture.py `
  pipeline/tests/test_owner_style_profile.py `
  pipeline/tests/test_style_groups.py `
  pipeline/tests/test_typesetting_backend_contract.py `
  pipeline/tests/test_typesetting_renderer.py `
  pipeline/tests/test_renderer_backend_parity.py `
  pipeline/tests/test_sfx_renderer.py `
  pipeline/tests/test_style_fidelity_qa.py `
  pipeline/tests/test_style_copy_score.py `
  pipeline/tests/test_style_audit_report.py `
  pipeline/tests/test_style_benchmark_v2_score.py `
  pipeline/tests/test_style_runtime_probe.py `
  pipeline/tests/test_style_owner_target_manifest.py `
  --junitxml="$testRoot\style.xml" -q `
  2>&1 | Tee-Object "$testRoot\style.log"
if ($LASTEXITCODE -ne 0) { throw "Style V2 focused suite failed" }
```

**Step 2: Rodar contratos owner/funcionais adjacentes**

```powershell
& $py -m pytest `
  pipeline/tests/test_owner_translation.py `
  pipeline/tests/test_owner_atomic_execution.py `
  pipeline/tests/test_owner_compositor.py `
  pipeline/tests/test_owner_compositor_properties.py `
  pipeline/tests/test_owner_enforcement.py `
  pipeline/tests/test_strip_process_bands.py `
  pipeline/tests/test_strip_owner_control_plane.py `
  pipeline/tests/test_strip_owner_composition_integration.py `
  pipeline/tests/test_project_writer.py `
  pipeline/tests/test_final_pixel_qa.py `
  pipeline/tests/test_final_pixel_export_gate.py `
  pipeline/tests/test_export_gate.py `
  pipeline/tests/test_export_gate_debug_consistency.py `
  pipeline/tests/test_main_emit.py `
  pipeline/tests/test_owner_visual_matrix_tool.py `
  --junitxml="$testRoot\functional-adjacent.xml" -q `
  2>&1 | Tee-Object "$testRoot\functional-adjacent.log"
if ($LASTEXITCODE -ne 0) { throw "Adjacent functional suite failed" }
```

**Step 3: Provar invariância semântica/máscara**

Run or add explicit property tests that compare the same fixture with style `off/shadow/enforce` and assert equality of owner semantic serialization, source/translated payload, component/route counts, action/protection masks, inpaint mutation hashes, functional QA findings, and layout safe region. Only final glyph pixels and style metadata may differ.

```powershell
& $py -m pytest `
  pipeline/tests/test_owner_style_profile.py `
  pipeline/tests/test_owner_style_capture.py `
  pipeline/tests/test_owner_translation.py `
  pipeline/tests/test_owner_atomic_execution.py `
  -k "invariant or semantic or translation or mask" -q
if ($LASTEXITCODE -ne 0) { throw "Style changed a forbidden functional contract" }
```

**Step 4: Rodar suite ampla e comparar o conjunto de falhas**

```powershell
& $py -m pytest pipeline/tests -q `
  --junitxml="$testRoot\pytest-after.xml" `
  2>&1 | Tee-Object "$testRoot\pytest-after.log"
$pytestAfterExit = $LASTEXITCODE
$pytestAfterExit | Set-Content "$testRoot\pytest-after.exit.txt" -Encoding ascii

[xml]$before = Get-Content ".codex-tmp\style_copy_v2_remediation_baseline_20260731\pytest-before.xml"
[xml]$after = Get-Content "$testRoot\pytest-after.xml"
$beforeFailures = @($before.testsuites.testsuite.testcase | Where-Object { $_.failure -or $_.error } | ForEach-Object { "$($_.classname)::$($_.name)" } | Sort-Object -Unique)
$afterFailures = @($after.testsuites.testsuite.testcase | Where-Object { $_.failure -or $_.error } | ForEach-Object { "$($_.classname)::$($_.name)" } | Sort-Object -Unique)
$newFailures = @(Compare-Object $beforeFailures $afterFailures | Where-Object SideIndicator -eq '=>' | Select-Object -ExpandProperty InputObject)
$newFailures | Set-Content "$testRoot\new-failures.txt" -Encoding utf8
if ($newFailures.Count -gt 0) { throw "Broad suite introduced new failures" }
```

Expected: zero failures novos. Falhas preexistentes podem desaparecer, mas não são requisito deste plano. Se o XML tiver estrutura agregada diferente, normalize todos os `testcase` descendants antes de comparar; não compare apenas contagens.

**Step 5: Auditar o diff e commits**

```powershell
git diff --check
git diff --cached --check
git status --short -- `
  pipeline/ownership `
  pipeline/typesetter `
  pipeline/sfx `
  pipeline/qa `
  pipeline/debug_tools `
  pipeline/tools `
  pipeline/tests
git log --oneline --decorate -14
```

Expected: nenhuma mudança da verificação ficou unstaged por acidente. Não commitar evidência `.codex-tmp`.

### Task 14: Executar novo holdout, inspecionar visualmente e publicar verdicts honestos

**Files:**

- Create: `docs/reports/2026-07-31-style-copy-v2-no-go-remediation-validation.md`
- Create: `docs/reports/evidence/2026-07-31-style-copy-v2-remediation-summary.json`
- Modify: `pipeline/tests/fixtures/style_copy_corpus/matrix.json` only if exact deterministic owner targets discovered in Task 12 were not yet pinned; do so test-first and in a separate scoped commit before the final run.

**Step 1: Criar root novo e rodar benchmark enforce**

```powershell
$py = (Resolve-Path "pipeline\venv\Scripts\python.exe").Path
$runRoot = ".codex-tmp\style_copy_v2_remediation_holdout_$(Get-Date -Format yyyyMMdd_HHmmss)"
New-Item -ItemType Directory -Force $runRoot | Out-Null

& $py pipeline/debug_tools/run_style_benchmark_v2.py `
  --spec pipeline/tests/fixtures/style_benchmark_v2/benchmark_spec.json `
  --output-root "$runRoot\benchmark" `
  --mode enforce
$benchmarkExit = $LASTEXITCODE
if ($benchmarkExit -notin 0,2) { throw "Benchmark infrastructure failed" }
```

Exit 2 means a measured Style NO-GO, not a tool crash. Preserve its report.

**Step 2: Rodar matriz owner-targeted em configs style-enforce**

```powershell
& $py pipeline/tools/validate_owner_visual_matrix.py `
  --manifest pipeline/tests/fixtures/style_copy_corpus/matrix.json `
  --output-root "$runRoot\matrix" `
  --run `
  --require-inspection
$matrixExit = $LASTEXITCODE
if ($matrixExit -notin 0,2) { throw "Style matrix infrastructure failed" }
```

Expected: no page clamping; each target resolves exact page/owner/components; run manifest and owner contact sheets are created. Style mode must be `enforce` in every effective config.

**Step 3: Executar probe e validar contratos**

```powershell
$projects = Get-ChildItem "$runRoot\matrix" -Recurse -Filter project.json | Select-Object -ExpandProperty FullName
& $py pipeline/debug_tools/style_runtime_probe.py `
  --output "$runRoot\runtime-probe.json" `
  $projects
$probeExit = $LASTEXITCODE
if ($probeExit -notin 0,2) { throw "Runtime probe infrastructure failed" }
```

Required for Style GO: 100% profile and raster-contract coverage, at least one appropriately applied attribute in every eligible high-confidence owner, no silent fallback, no hash/binding mismatch, and all required category denominators nonzero.

**Step 4: Inspecionar cada owner em escala nativa**

Open every required sheet and compare all six panels. For each owner, record separate verdicts:

- evidence masks isolate the source glyph and do not absorb card art/neighboring owners;
- requested attributes reflect the source;
- raster contract matches the drawn patch;
- final pixels preserve font class, fill, stroke/effects, rotation/geometry, hierarchy, and containment;
- no new false glow/shadow/stroke;
- functional note separately records any English, OCR, inpaint, duplication, overflow, split-body, or proportionality issue.

Fill inspection schema v2 with `style_verdict`, `functional_verdict`, notes, reviewer timestamp, native dimensions, and hashes. Re-run validate-only:

```powershell
& $py pipeline/tools/validate_owner_visual_matrix.py `
  --manifest pipeline/tests/fixtures/style_copy_corpus/matrix.json `
  --output-root "$runRoot\matrix" `
  --validate-only `
  --require-inspection
$validateExit = $LASTEXITCODE
if ($validateExit -notin 0,2) { throw "Validated evidence is inconsistent" }
```

**Step 5: Calcular verdicts separados**

Style GO requires every design acceptance criterion, including:

- raster coverage/binding 100%;
- font top-1 >= 0.90, top-3 >= 0.98, each category >= 0.85;
- fill Delta E median <= 8 and p95 <= 12;
- stroke Delta E <= 10 and thickness error <= 20% x-height or 1 px;
- glow/shadow Delta E <= 12 and radius/offset error <= 25% or 2 px;
- rotation <= 3 degrees; width/slant/tracking/occupancy error <= 15%;
- permitted x-height ratio 0.85-1.15;
- core containment 100%; false decorative effects <= 2%;
- owner visual GO >= 90%, speech >= 95%, no category < 85%;
- no zero-owner required category and no catastrophic high-confidence mismatch;
- complete native inspection approval.

Functional GO comes only from functional subgates. Overall GO requires Style GO + Functional GO + Inspection GO. A frozen style corpus may be Style GO while the live chapter is Functional/Overall NO-GO.

**Step 6: Publicar relatório mesmo quando o resultado for NO-GO**

The Markdown report must include:

- Git HEAD and scoped diff hash;
- exact commands, roots, return codes, runtime/config/input/tool hashes;
- before/after runtime counts;
- per-category metrics and sample counts;
- contract coverage and abstention reasons;
- benchmark thresholds and measured values;
- native inspection table by exact owner;
- separate Style, Functional, Inspection, and Overall verdicts;
- remaining blockers with direct artifact paths;
- explicit statement that no page-specific rule was added.

The JSON summary contains the same machine-readable verdicts and provenance. Do not copy bulky generated images into Git; link to their run-manifest paths and hashes.

**Step 7: Commit relatório e evidência resumida**

```powershell
git add -- `
  docs/reports/2026-07-31-style-copy-v2-no-go-remediation-validation.md `
  docs/reports/evidence/2026-07-31-style-copy-v2-remediation-summary.json
git diff --cached --check
git diff --cached --name-only
git commit -m "test(style): validate remediation on owner holdout"
```

If exact matrix targets required a late tracked update, commit that separately before the final run:

```powershell
git add -p -- pipeline/tests/fixtures/style_copy_corpus/matrix.json
git commit -m "test(style): pin exact holdout owner targets"
```

**Step 8: Verificação final antes da conclusão**

```powershell
git status --short --branch
git diff --check
git diff --cached --check
git log --oneline --decorate -16
git show --stat --oneline HEAD
```

Confirm that every commit contains only its scoped files and that preexisting local changes remain present. Never mark the plan complete while any mandatory test, holdout artifact, inspection row, hash, category, or verdict is missing.

## Critério de parada e resposta a falhas

- A RED inesperada triggers `systematic-debugging`: reproduce narrowly, trace the owning contract, form one hypothesis, add a regression test, then patch the earliest authoritative layer.
- Do not weaken thresholds, reinterpret missing evidence as fallback, remove an owner/category from the denominator, or change a target to make the matrix pass.
- Do not fix a visual mismatch by matching its page/text/color. Fix the extraction, decision, materialization, capability, transport, or QA contract that generalizes.
- If a style metric fails while contracts are valid, remain NO-GO and improve the owning generic algorithm in a follow-up TDD task before re-running into a fresh root.
- If functional errors remain while frozen style targets pass, report Style GO and Functional/Overall NO-GO separately.
- Completion means the full chain is implemented and verified, not merely that the code paths exist.

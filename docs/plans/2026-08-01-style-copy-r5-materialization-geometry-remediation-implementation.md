# Style Copy R5 Materialization, Geometry, and Acceptance Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fazer o pipeline automático materializar decisões de estilo em pixels observáveis, preservar o corpo semântico completo em coordenadas corretas e produzir uma aceitação Style/Functional/Inspection realmente mensurável e fail-closed.

**Architecture:** O perfil aprovado permanece imutável e gera um plano de materialização separado. Layout, fonte e raster registram observações próprias, canônicas e vinculadas por hash; o commit atômico compara intenção resolvida com observação real. Geometria de owner em página lógica fica separada de cleanup e de framing de saída. Benchmark sintético, QA raster e inspeção owner-targeted produzem métricas distintas, combinadas por um agregador final autenticado.

**Tech Stack:** Python 3.12, dataclasses/typing, MappingProxyType, NumPy, OpenCV, matplotlib FT2Font/TextPath, fontTools, pytest, JSON/JSONL, SHA-256, CIEDE2000, PowerShell e Git com staging por hunk.

---

## Fonte de verdade

- Design aprovado e refinado: `docs/plans/2026-08-01-style-copy-r5-materialization-geometry-remediation-design.md`.
- Relatório R4: `docs/reports/2026-07-31-style-copy-v2-no-go-remediation-validation.md`.
- Resumo R4: `docs/reports/evidence/2026-07-31-style-copy-v2-remediation-summary.json`.
- Holdout R4 somente leitura: `N:\TraduzAI_style_runs\style_copy_v2_remediation_holdout_20260801_001546`.
- Matriz exata de nove owners: `pipeline/tests/fixtures/style_copy_corpus/matrix.json`.
- Python obrigatório: `pipeline/venv/Scripts/python.exe`.
- Branch/check-out atual e todas as mudanças locais são autoridade. Não recriar o trabalho em checkout limpo.

## Diagnóstico que o plano deve preservar

1. `resolve_contextual_style_groups()` pode remover efeitos do estilo materializado sem manter decisão e raster coerentes.
2. Owners V2 podem passar novamente pela normalização automática e perder valores verificados.
3. `_render_v2_owner_text_layer()` exige do rasterizer atributos pertencentes a layout/fonte; os não resolvidos zeram o RGBA e deixam `render_completed=False`.
4. O runtime ecoa alguns valores solicitados, especialmente fonte/peso/largura, em vez de observar o arquivo e os eixos efetivamente usados.
5. `apply_atomic_owner_execution()` compara representações cruas e rejeita valores materialmente equivalentes.
6. O owner cross-tile `owner_p002_d9afb32ef935` contém o PT-BR completo na página final; o crop visual foi deslocado porque bbox de página lógica foi aplicado diretamente ao frame de 800 px. Página lógica: 690 px; origem do conteúdo no frame: `x=55`.
7. `source_replacement_bbox` pode ser promovido a container de layout, apesar de cleanup e capacidade de texto serem domínios diferentes.
8. O benchmark atual mede o extractor legado na imagem-fonte. Ele não executa renderer, owner QA ou inspeção, mas o scorer exige métricas que somente esses outros produtores podem provar.

## Regras obrigatórias de execução

- Use `@executing-plans`, `@test-driven-development`, `@traduzai-typesetting` e `@traduzai-pipeline` durante a implementação.
- Execute as tasks na ordem. Pare apenas quando um checkpoint estiver genuinamente bloqueado por evidência externa ausente.
- Para cada mudança de produção: teste RED, confirmar a falha esperada, implementação mínima, GREEN, refactor mantendo verde, commit restrito.
- Nunca use `git reset`, `git checkout`, `git restore`, `git stash`, `git clean` ou novo worktree.
- Antes de editar um arquivo existente, execute `git diff -- <arquivo>` e preserve cada hunk local.
- Arquivos já modificados que exigem staging por patch restrito ao índice: `pipeline/inpainter/owner_mask.py`, `pipeline/ownership/artifacts.py`, `pipeline/ownership/evidence.py`, `pipeline/ownership/reconcile.py`, `pipeline/ownership/translation.py`, `pipeline/qa/export_gate.py`, `pipeline/qa/final_pixel_qa.py`, `pipeline/qa/inpaint_residual.py`, `pipeline/strip/process_bands.py`, `pipeline/strip/run.py`, `pipeline/tests/test_final_pixel_export_gate.py`, `pipeline/tests/test_final_pixel_qa.py`, `pipeline/tests/test_inpaint_debug_residual.py`, `pipeline/tests/test_owner_debug_artifacts.py`, `pipeline/tests/test_owner_enforcement.py`, `pipeline/tests/test_owner_evidence.py`, `pipeline/tests/test_owner_mask.py`, `pipeline/tests/test_owner_reconcile.py`, `pipeline/tests/test_owner_translation.py`, `pipeline/tests/test_owner_visual_matrix_tool.py`, `pipeline/tests/test_translation_locale_policy.py`, `pipeline/tools/validate_owner_visual_matrix.py`, `pipeline/translator/locale_policy.py`, `pipeline/translator/translate.py`, além de qualquer arquivo que apareça dirty no preflight.
- Nunca stage por diretório. Antes de cada commit, execute `git diff --cached --check` e `git diff --cached --name-status`.
- `.codex-tmp`, caches, imagens, contact sheets e holdouts ficam fora do Git.
- Não enfraquecer gate, threshold, denominador, target ou hash para obter GO.
- Não adicionar regra por página, obra, capítulo, owner ID ou frase específica.
- Não considerar `exit 0`, `success=true` ou suite sem novas regressões como aprovação visual. `export_gate=BLOCK` e inspeção nativa `NO-GO` prevalecem.
- Se um teste novo passar antes da implementação, ele não reproduz a lacuna: reescreva o teste até observar o RED correto.

## Checkpoints

1. **R5-A — baseline:** Task 0.
2. **R5-B — materialização:** Tasks 1–8. Perfis problemáticos geram raster não vazio, observação completa e comparação canônica sem request echo.
3. **R5-C — geometria:** Tasks 9–11. Página lógica, frame e owner geometry round-trip sem deslocamento; cleanup nunca define container.
4. **R5-D — métricas:** Tasks 12–16. Cada produtor emite somente métricas que pode provar; aceitação final fica completa e fail-closed.
5. **R5-E — regressão sistêmica:** Tasks 17–18. Suites focadas/amplas sem nova falha e sentinela exata resolvida.
6. **R5-F — visual:** Task 19. Holdout externo novo, inspeção nativa 9/9 e relatório honesto.

Regra TDD para todas as tasks: um RED válido é `pytest` exit code **1** por assertion comportamental. Exit 2/3/4/5, collection error, import error, syntax error ou zero testes não contam. Para módulo novo, use `importlib.util.find_spec()` no primeiro teste ou crie um skeleton importável mínimo e confirme novamente o RED comportamental antes de escrever a implementação.

Protocolo obrigatório para arquivo já dirty: nunca use whole-file `git add --` nem dependa de `git add -p` em shell não interativo. Para cada task, crie com `apply_patch` um patch mínimo sob `$r5Tmp` contendo somente os hunks R5 dos paths dirty, defina `$r5ScopedPatch`, execute `git apply --cached --check --unidiff-zero $r5ScopedPatch` e depois `git apply --cached --unidiff-zero $r5ScopedPatch`, verificando exit 0 após ambos. Inspecione `git diff --cached -- <arquivo>` e `git diff -- <arquivo>`; o cached diff deve conter apenas R5 e o working diff deve preservar os hunks preexistentes, salvo linha sobreposta explicitamente registrada. Todo comando nativo de staging, `git diff --cached --check` e `git commit` deve ter seu `$LASTEXITCODE` testado imediatamente; nenhum bloco pode continuar após falha ou índice vazio/inesperado.

Ordem de contratos: o materialization plan/observation não contém `OwnerRenderGeometry`, evitando dependência circular nas Tasks 3–8. Na Task 10, o envelope externo `OwnerGlyphPatch` passa a autenticar conjuntamente plan/raster/delivery e `OwnerRenderGeometry`; qualquer mudança de layout causada pela geometria independente força novo `render_layout_contract` e novo plan antes do raster. O checkpoint sistêmico final só é válido depois das Tasks 10–12.

### Task 0: Congelar checkout, diffs e baseline diferencial

**Files:** Nenhum arquivo versionado.

**Step 1: Inspecionar o estado real**

Run:

```powershell
git status --short --branch
git rev-parse HEAD
git diff --stat
git diff -- pipeline/inpainter/owner_mask.py pipeline/ownership/artifacts.py pipeline/ownership/evidence.py pipeline/ownership/reconcile.py pipeline/ownership/translation.py pipeline/qa/export_gate.py pipeline/qa/final_pixel_qa.py pipeline/qa/inpaint_residual.py pipeline/strip/process_bands.py pipeline/strip/run.py pipeline/tests/test_final_pixel_export_gate.py pipeline/tests/test_final_pixel_qa.py pipeline/tests/test_inpaint_debug_residual.py pipeline/tests/test_owner_debug_artifacts.py pipeline/tests/test_owner_enforcement.py pipeline/tests/test_owner_evidence.py pipeline/tests/test_owner_mask.py pipeline/tests/test_owner_reconcile.py pipeline/tests/test_owner_translation.py pipeline/tests/test_owner_visual_matrix_tool.py pipeline/tests/test_translation_locale_policy.py pipeline/tools/validate_owner_visual_matrix.py pipeline/translator/locale_policy.py pipeline/translator/translate.py
git diff --check
```

Expected: branch `Troca_de_motores`, checkout sujo preservado e nenhum arquivo alterado pelo comando. Registre no log do executor o HEAD e os arquivos dirty; não corrija whitespace preexistente fora do escopo.

**Step 2: Criar somente o diretório de evidência temporária**

Run:

```powershell
$r5PlanAnchor = git log -1 --format=%H -- docs/plans/2026-08-01-style-copy-r5-materialization-geometry-remediation-implementation.md
if (-not $r5PlanAnchor) { throw 'R5 implementation plan must be committed before execution' }
$r5Tmp = Join-Path (Resolve-Path '.') ('.codex-tmp/style-copy-r5-' + $r5PlanAnchor.Substring(0, 12))
New-Item -ItemType Directory -Force -Path $r5Tmp | Out-Null
git diff --binary --output="$r5Tmp\preexisting-relevant.patch" -- pipeline/inpainter/owner_mask.py pipeline/ownership/artifacts.py pipeline/ownership/evidence.py pipeline/ownership/reconcile.py pipeline/ownership/translation.py pipeline/qa/export_gate.py pipeline/qa/final_pixel_qa.py pipeline/qa/inpaint_residual.py pipeline/strip/process_bands.py pipeline/strip/run.py pipeline/tests/test_final_pixel_export_gate.py pipeline/tests/test_final_pixel_qa.py pipeline/tests/test_inpaint_debug_residual.py pipeline/tests/test_owner_debug_artifacts.py pipeline/tests/test_owner_enforcement.py pipeline/tests/test_owner_evidence.py pipeline/tests/test_owner_mask.py pipeline/tests/test_owner_reconcile.py pipeline/tests/test_owner_translation.py pipeline/tests/test_owner_visual_matrix_tool.py pipeline/tests/test_translation_locale_policy.py pipeline/tools/validate_owner_visual_matrix.py pipeline/translator/locale_policy.py pipeline/translator/translate.py
if ($LASTEXITCODE -ne 0) { throw 'Could not snapshot preexisting relevant diff' }
$preexistingPatchIdLine = Get-Content -LiteralPath "$r5Tmp\preexisting-relevant.patch" -Raw | git patch-id --stable
if ($LASTEXITCODE -ne 0 -or -not $preexistingPatchIdLine) { throw 'Could not fingerprint preexisting relevant diff' }
Write-Output "preexisting_patch_id=$($preexistingPatchIdLine.Split()[0])"
$r5Tmp
```

Expected: diretório não versionado sob `.codex-tmp` e patch-base binário dos arquivos compartilhados. Não adicionar `.codex-tmp` ao `.gitignore` nem ao Git. Qualquer linha sobreposta consumida por R5 deve ser anotada no log do executor e conferida contra esse patch.

**Step 3: Congelar suites focadas atuais**

Run:

```powershell
$r5PlanAnchor = git log -1 --format=%H -- docs/plans/2026-08-01-style-copy-r5-materialization-geometry-remediation-implementation.md
$r5Tmp = Join-Path (Resolve-Path '.') ('.codex-tmp/style-copy-r5-' + $r5PlanAnchor.Substring(0, 12))
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_style_groups.py pipeline/tests/test_glyph_rasterizer.py pipeline/tests/test_typesetting_renderer.py pipeline/tests/test_style_contract.py pipeline/tests/test_owner_atomic_execution.py pipeline/tests/test_owner_enforcement.py pipeline/tests/test_owner_mask.py pipeline/tests/test_owner_compositor.py pipeline/tests/test_owner_layout.py pipeline/tests/test_strip_owner_composition_integration.py pipeline/tests/test_style_benchmark_v2_generator.py pipeline/tests/test_style_benchmark_v2_score.py pipeline/tests/test_style_fidelity_qa.py pipeline/tests/test_owner_visual_matrix_tool.py -q --junitxml "$r5Tmp\focused-baseline.xml"
$focusedBaselineExit = $LASTEXITCODE
if ($focusedBaselineExit -notin @(0, 1)) { throw "Focused baseline ended operationally with exit $focusedBaselineExit" }
[xml]$focusedBaselineXml = Get-Content -LiteralPath "$r5Tmp\focused-baseline.xml" -Raw
$focusedBaselineCases = @($focusedBaselineXml.SelectNodes('//testcase'))
if ($focusedBaselineCases.Count -eq 0) { throw 'Focused baseline JUnit is missing or has zero collected testcases' }
Write-Output "focused_baseline_exit=$focusedBaselineExit testcases=$($focusedBaselineCases.Count)"
```

Expected: o XML é criado mesmo se houver falhas preexistentes. Preserve a lista de nodeids; não use contagem isolada como diferencial.

**Step 4: Congelar baseline amplo**

Run:

```powershell
$r5PlanAnchor = git log -1 --format=%H -- docs/plans/2026-08-01-style-copy-r5-materialization-geometry-remediation-implementation.md
$r5Tmp = Join-Path (Resolve-Path '.') ('.codex-tmp/style-copy-r5-' + $r5PlanAnchor.Substring(0, 12))
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests -q --junitxml "$r5Tmp\full-baseline.xml"
$fullBaselineExit = $LASTEXITCODE
if ($fullBaselineExit -notin @(0, 1)) { throw "Full baseline ended operationally with exit $fullBaselineExit" }
[xml]$fullBaselineXml = Get-Content -LiteralPath "$r5Tmp\full-baseline.xml" -Raw
$fullBaselineCases = @($fullBaselineXml.SelectNodes('//testcase'))
if ($fullBaselineCases.Count -eq 0) { throw 'Full baseline JUnit is missing or has zero collected testcases' }
Write-Output "full_baseline_exit=$fullBaselineExit testcases=$($fullBaselineCases.Count)"
```

Expected: o snapshot recém-gerado é a autoridade. O R4 tinha 3640 passed, 249 failed e 33 skipped, mas não presuma que o checkout atual tenha números idênticos.

**Step 5: Confirmar que o preflight não alterou o índice**

Run:

```powershell
$preflightStaged = @(git diff --cached --name-only)
if ($LASTEXITCODE -ne 0) { throw 'Could not inspect preflight index' }
if ($preflightStaged.Count -ne 0) { throw "Preflight index is not empty: $($preflightStaged -join ', ')" }
git status --short -- .codex-tmp
```

Expected: nenhum arquivo novo staged; `.codex-tmp` permanece fora do índice.

### Task 1: Definir ownership de atributos e canonicalização tipada

**Files:**

- Create: `pipeline/typesetter/style_materialization.py`
- Create: `pipeline/tests/test_style_materialization.py`
- Modify: `pipeline/typesetter/style_contract.py` em `STYLE_V2_ATTRIBUTE_NAMES` para registrar `multistroke` como atributo raster versionado.
- Modify: `pipeline/tests/test_style_contract.py`
- Modify: `pipeline/qa/style_fidelity.py` somente depois de os testes do novo módulo ficarem verdes, para importar a implementação única de CIEDE2000.

**Step 1: Escrever os testes RED do registry**

Adicione testes que expressem a API desejada sem causar erro de collection quando o módulo ainda não existir:

```python
def test_materialization_module_and_domain_registry_are_complete():
    import importlib
    import importlib.util

    spec = importlib.util.find_spec("typesetter.style_materialization")
    assert spec is not None
    module = importlib.import_module("typesetter.style_materialization")
    assert set(module.ATTRIBUTE_DOMAIN) == STYLE_V2_ATTRIBUTE_NAME_SET
    assert set(module.attributes_for_domain("layout")) == {
        "font_size_px", "alignment", "container", "tracking_xh", "curve"
    }
    assert set(module.attributes_for_domain("font")) == {
        "font_name", "font_weight", "font_width"
    }
    assert set(module.attributes_for_domain("raster")) == (
        STYLE_V2_ATTRIBUTE_NAME_SET
        - set(module.attributes_for_domain("layout"))
        - set(module.attributes_for_domain("font"))
    )
```

Também cubra:

```python
def test_canonicalization_normalizes_equivalent_values():
    assert canonicalize_style_attribute("fill", "fff") == "#FFFFFF"
    assert canonicalize_style_attribute("rotation_deg", -0.0) == 0.0
    assert canonicalize_style_attribute(
        "stroke", {"width_xh": 0.1, "color": "#fff"}
    ) == {"color": "#FFFFFF", "width_xh": 0.1}


def test_material_difference_is_not_normalized_away():
    result = compare_style_attribute("fill", "#FFFFFF", "#F0A000")
    assert result.matches is False
    assert result.delta_e_2000 is not None and result.delta_e_2000 > 12


def test_unknown_or_duplicate_domain_is_rejected():
    with pytest.raises(ValueError, match="domain registry"):
        validate_attribute_domain_registry({"fill": "raster"})
```

**Step 2: Confirmar RED**

Run:

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_style_materialization.py -q
if ($LASTEXITCODE -ne 1) { throw "Expected behavioral RED exit 1, got ${LASTEXITCODE}: materialization registry does not exist yet" }
```

Expected: assertion failure porque o módulo/registry ainda não existe; não aceitar erro de sintaxe ou fixture.

**Step 3: Implementar o núcleo canônico mínimo**

Crie:

```python
MaterializationDomain = Literal["layout", "font", "raster"]

ATTRIBUTE_DOMAIN: Final[dict[str, MaterializationDomain]] = {
    "font_size_px": "layout",
    "alignment": "layout",
    "container": "layout",
    "tracking_xh": "layout",
    "curve": "layout",
    "font_name": "font",
    "font_weight": "font",
    "font_width": "font",
    "slant_tangent": "raster",
    "width_scale": "raster",
    "scale_y": "raster",
    "fill": "raster",
    "stroke": "raster",
    "multistroke": "raster",
    "shadow": "raster",
    "glow": "raster",
    "gradient": "raster",
    "rotation_deg": "raster",
}

@dataclass(frozen=True)
class AttributeComparison:
    name: str
    domain: MaterializationDomain
    expected: Any
    observed: Any
    matches: bool
    tolerance: Mapping[str, Any]
    delta_e_2000: float | None = None
    reason: str = ""
```

Implemente canonicalização determinística para cores sRGB, números finitos, aliases de alignment/weight/width, efeitos estruturados, gradiente, curve e container. `tracking_xh` e `curve` pertencem a layout porque alteram avanços/poses finais por glyph; `slant_tangent` pertence a raster porque o backend atual o aplica como shear do bitmap. Registre `multistroke` em `STYLE_V2_ATTRIBUTE_NAMES`: `stroke` representa exatamente uma camada e `multistroke` duas ou mais camadas em paint-order canônica. Um multistroke de uma camada canonicaliza para `stroke`; input que aprove simultaneamente os dois é rejeitado ou resolvido explicitamente pela relação `superseded_by`, nunca por preferência silenciosa. Mova `delta_e_2000()` de `qa/style_fidelity.py` para esse módulo sem mudar o resultado numérico; deixe re-export/import compatível em QA.

Não canonicalize fonte por nome nesta task; `font_name` deve exigir o resolver tipado da Task 2.

**Step 4: Confirmar GREEN e regressão QA**

Run:

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_style_materialization.py pipeline/tests/test_style_contract.py pipeline/tests/test_style_fidelity_qa.py -q
if ($LASTEXITCODE -ne 0) { throw 'Task 1 GREEN failed' }
```

Expected: todos os testes passam e `delta_e_2000("#FFFFFF", "#000000") > 90` continua válido.

**Step 5: Commit restrito**

Run:

```powershell
git add -- pipeline/typesetter/style_materialization.py pipeline/typesetter/style_contract.py pipeline/tests/test_style_materialization.py pipeline/tests/test_style_contract.py
if ($LASTEXITCODE -ne 0) { throw 'Task 1 primary staging failed' }
git add -- pipeline/qa/style_fidelity.py
if ($LASTEXITCODE -ne 0) { throw 'Task 1 QA staging failed' }
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Task 1 staged diff check failed' }
git diff --cached --name-status
if ($LASTEXITCODE -ne 0) { throw 'Task 1 staged path inspection failed' }
git commit -m "feat(style): define canonical materialization domains"
if ($LASTEXITCODE -ne 0) { throw 'Task 1 commit failed' }
```

Expected: apenas os cinco arquivos/hunks enumerados na task.

### Task 2: Observar a identidade real da fonte selecionada

**Files:**

- Create: `pipeline/typesetter/font_identity.py`
- Create: `pipeline/tests/test_font_identity.py`
- Modify: `pipeline/typesetter/style_materialization.py`
- Modify: `pipeline/requirements.txt` para declarar `fonttools` diretamente; não depender da instalação transitiva do matplotlib.
- Test: `pipeline/tests/test_font_matcher.py`
- Create: `pipeline/tests/test_sidecar_dependency_contract.py`

**Step 1: Escrever os testes RED**

Comece com `find_spec("typesetter.font_identity")`/`import_module()` dentro do teste; se criar skeleton importável, rerode até obter exit 1 por comportamento ausente antes da implementação.

```python
def test_resolved_font_identity_is_bound_to_file_and_face_metadata():
    identity = resolve_font_identity(FONTS_DIR / "ComicNeue-Bold.ttf")
    assert identity.filename == "ComicNeue-Bold.ttf"
    assert len(identity.file_sha256) == 64
    assert identity.family
    assert identity.weight_class >= 700


def test_font_intent_and_runtime_path_compare_by_resolved_identity():
    catalog = load_font_identity_catalog(FONTS_DIR, FONTS_DIR / "font-map.json")
    intent = canonicalize_font_intent("comicneue-bold.ttf", catalog)
    observed = observe_resolved_font(FONTS_DIR / "ComicNeue-Bold.ttf")
    assert compare_resolved_font(intent, observed).matches is True


def test_requested_weight_is_not_accepted_when_resolved_face_is_regular():
    catalog = load_font_identity_catalog(FONTS_DIR, FONTS_DIR / "font-map.json")
    expected = canonicalize_font_intent(
        {"font_name": "ComicNeue-Bold.ttf", "font_weight": 700}, catalog
    )
    observed = observe_resolved_font(FONTS_DIR / "ComicNeue-Regular.ttf")
    assert compare_resolved_font(expected, observed).matches is False


def test_font_run_records_character_level_fallback_instead_of_claiming_primary_face():
    run = resolve_font_run(primary_missing_ptbr_glyph(), text="AÇÃO", catalog=fallback_catalog())
    assert run.primary_identity.file_sha256
    assert any(span.fallback_identity is not None for span in run.spans)
    assert compare_font_run(approved_primary_only(), run).matches is False


def test_pre_raster_font_run_plan_can_explicitly_authorize_ptbr_fallback():
    target = resolve_font_run_plan(
        intent=primary_missing_ptbr_glyph(), text="AÇÃO", catalog=fallback_catalog(),
    )
    assert target.resolution_kind == "derived"
    assert target.reason == "glyph_coverage_fallback"
    observed = render_and_observe_font_run(target)
    assert compare_font_run(target, observed).matches is True


def test_variable_font_observes_configured_axis_not_only_fvar_metadata():
    run = render_variable_font_run(requested_axes={"wdth": 85.0})
    assert run.configured_axes == (("wdth", 85.0),)
    assert axis_definition(run.primary_identity, "wdth").default_value != 85.0
    assert run.instance_file_sha256 != run.primary_identity.file_sha256
    assert glyph_advance(run, "M") != glyph_advance(render_variable_font_run(requested_axes={"wdth": 100.0}), "M")


def test_fonttools_is_directly_pinned_for_sidecar_runtime():
    requirement = direct_requirement("fonttools", Path("pipeline/requirements.txt"))
    assert requirement == "fonttools==4.62.1"
```

Inclua teste de ordem invariável do catálogo e de variável `wdth` para `LeagueGothic-Regular-VariableFont_wdth.ttf`.

**Step 2: Confirmar RED**

Run:

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_font_identity.py -q
if ($LASTEXITCODE -ne 1) { throw "Expected behavioral RED exit 1, got ${LASTEXITCODE}: resolved font identity is missing" }
```

**Step 3: Implementar identidade e catálogo**

Use leitura lazy de `fontTools.ttLib.TTFont` e fixe uma constraint compatível com a versão validada no venv em `pipeline/requirements.txt`. O contrato mínimo é:

```python
@dataclass(frozen=True)
class FontAxisDefinition:
    tag: str
    minimum_value: float
    default_value: float
    maximum_value: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class ResolvedFontIdentity:
    filename: str
    file_sha256: str
    family: str
    subfamily: str
    postscript_name: str
    weight_class: int
    width_class: int
    variation_axes: tuple[FontAxisDefinition, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "file_sha256": self.file_sha256,
            "family": self.family,
            "subfamily": self.subfamily,
            "postscript_name": self.postscript_name,
            "weight_class": self.weight_class,
            "width_class": self.width_class,
            "variation_axes": [item.to_dict() for item in self.variation_axes],
        }
```

Normalize nomes somente como aliases auxiliares. A igualdade material exige o mesmo SHA-256/face ou uma regra explícita de fonte variável sobre o mesmo arquivo; não aceite apenas basename/family. Reuse os mesmos assets e metadados licenciados de `font_matcher.load_font_catalog()`.

Defina `FontRunPlan`, `FontRunObservation` e resolvers puros: identidade primária, spans/glyph IDs, codepoints, identidade de cada fallback, eixos realmente configurados, SHA da instância renderizada e hashes. Antes do raster, verifique cobertura do payload PT-BR completo. Fallback necessário pode virar target `derived:glyph_coverage_fallback` somente quando identidade/estilo e spans estão explicitamente no plan; fallback não planejado é mismatch/review. A instrumentação do `_render_text_with_fallback()` real pertence à Task 7, quando o renderer e todos os observers são migrados juntos.

Como `matplotlib.ft2font.FT2Font` do venv não configura variation axes, materialize fonte variável com `fontTools.varLib.instancer.instantiateVariableFont`: gere instância estática determinística/cacheada por `(source_file_sha256, canonical_axes)`, renderize essa instância em TextPath/FT2Font e observe source identity + configured axes + instance file SHA/face metadata. O cache root é explícito e fica no output/cache não versionado, com escrita atômica e revalidação do SHA; nunca sobrescreva o font asset original. Prove por pixels/advance que `wdth=85` difere do default 100; falha de instanciação/corrupção vira review, nunca request echo. Fixe `fonttools==4.62.1`, versão validada no venv, no requirements direto.

**Step 4: Confirmar GREEN**

Run:

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_font_identity.py pipeline/tests/test_font_matcher.py pipeline/tests/test_sidecar_dependency_contract.py -q
if ($LASTEXITCODE -ne 0) { throw 'Task 2 GREEN failed' }
```

**Step 5: Commit**

```powershell
git add -- pipeline/typesetter/font_identity.py pipeline/typesetter/style_materialization.py pipeline/requirements.txt pipeline/tests/test_font_identity.py pipeline/tests/test_font_matcher.py pipeline/tests/test_sidecar_dependency_contract.py
if ($LASTEXITCODE -ne 0) { throw 'Task 2 staging failed' }
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Task 2 staged diff check failed' }
git diff --cached --name-status
if ($LASTEXITCODE -ne 0) { throw 'Task 2 staged path inspection failed' }
git commit -m "feat(style): bind materialization to resolved font identity"
if ($LASTEXITCODE -ne 0) { throw 'Task 2 commit failed' }
```

### Task 3: Criar plano e observação de materialização imutáveis

**Files:**

- Modify: `pipeline/typesetter/style_materialization.py`
- Modify: `pipeline/tests/test_style_materialization.py`
- Test: `pipeline/tests/test_style_contract.py`

**Step 1: Escrever os testes RED de contrato**

```python
def test_materialization_plan_resolves_every_approved_attribute_once():
    intent = build_resolved_style_intent(
        owner_id="owner_a",
        page_id="page_001",
        visual_profile_sha256="a" * 64,
        decision_sha256="b" * 64,
        group_resolution_sha256="c" * 64,
        approved={"fill": "#fff", "font_name": "ComicNeue-Bold.ttf"},
        approved_abstentions={"glow": "low_confidence"},
    )
    plan = build_materialization_plan(
        intent=intent,
        render_layout_contract_sha256="d" * 64,
        targets={"fill": "#FFFFFF", "font_name": resolved_font_dict()},
        resolution_kinds={"fill": "exact", "font_name": "exact", "glow": "abstained"},
    )
    assert set(plan.attribute_plans) == {"fill", "font_name", "glow"}
    assert plan.attribute_plans["fill"].domain == "raster"
    assert plan.attribute_plans["glow"].resolution_kind == "abstained"
    assert len(plan.plan_sha256) == 64


def test_fit_adjustment_is_preserved_as_intent_but_compared_to_resolved_plan():
    intent = intent_with_attributes({"font_size_px": 48})
    plan = build_materialization_plan(
        intent=intent,
        render_layout_contract_sha256="d" * 64,
        targets={"font_size_px": 36},
        resolution_kinds={"font_size_px": "policy_adjusted"},
        resolution_reasons={"font_size_px": "fit_to_verified_container"},
    )
    assert plan.attribute_plans["font_size_px"].intent_value == 48
    assert plan.attribute_plans["font_size_px"].target_value == 36
    assert compare_materialization(plan, observed_font_size(36)).status == "match"


def test_gradient_supersedes_unobservable_solid_fill_explicitly():
    plan = plan_for_gradient_with_source_fill()
    assert plan.attribute_plans["gradient"].resolution_kind == "exact"
    assert plan.attribute_plans["fill"].resolution_kind == "superseded"
    assert plan.attribute_plans["fill"].superseded_by == "gradient"
    assert compare_materialization(plan, observed_gradient_only()).status == "match"


def test_xheight_units_resolve_before_raster_and_compare_in_execution_units():
    plan = plan_with_final_xheight(20, stroke_width_xh=.10, glow_radius_xh=.15)
    assert plan.attribute_plans["stroke"].target_value["width_px"] == 2
    assert plan.attribute_plans["glow"].target_value["radius_px"] == 3
    assert compare_materialization(plan, observed_effect_pixels(stroke=2, glow=3)).status == "match"


def test_observation_cannot_claim_requested_value_without_domain_evidence():
    with pytest.raises(ValueError, match="evidence"):
        build_materialization_observation(
            plan=plan_with_fill(),
            domain_observations={"raster": {"fill": {"value": "#FFFFFF"}}},
            render_completed=True,
        )


def test_comparison_reports_exact_attribute_domain_and_reason():
    comparison = compare_materialization(plan_with_fill("#FFFFFF"), observed_fill("#000000"))
    assert comparison.status == "mismatch"
    assert comparison.mismatches[0]["attribute"] == "fill"
    assert comparison.mismatches[0]["domain"] == "raster"
    assert comparison.mismatches[0]["reason"] == "canonical_value_mismatch"


def test_plan_and_observation_are_hash_bound_and_immutable():
    with pytest.raises(TypeError):
        observation.attributes["fill"] = {}
    assert validate_materialization_observation(observation)["observation_sha256"] == observation.observation_sha256
```

Também cubra atributo desconhecido, domínio duplicado, `unavailable`, `abstained`, `render_completed=False`, hash adulterado e permutação de mapas.

**Step 2: Confirmar RED**

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_style_materialization.py -q
if ($LASTEXITCODE -ne 1) { throw "Expected behavioral RED exit 1, got ${LASTEXITCODE}: immutable materialization contracts are missing" }
```

**Step 3: Implementar os contratos mínimos**

Use dataclasses frozen e JSON canônico:

```python
AttributeStatus = Literal["materialized", "abstained", "mismatch", "unavailable"]
ResolutionKind = Literal[
    "exact", "policy_adjusted", "derived", "superseded", "abstained", "review_required"
]

@dataclass(frozen=True)
class OwnerStyleResolvedIntent:
    schema_version: int
    owner_id: str
    page_id: str
    visual_profile_sha256: str
    decision_sha256: str
    group_resolution_sha256: str
    approved_attributes: Mapping[str, Any]
    approved_abstentions: Mapping[str, str]
    attribute_provenance: Mapping[str, Any]
    intent_sha256: str

@dataclass(frozen=True)
class MaterializationAttributePlan:
    name: str
    domain: MaterializationDomain
    intent_value: Any
    target_value: Any
    resolution_kind: ResolutionKind
    reason: str
    superseded_by: str
    evidence_ids: tuple[str, ...]

@dataclass(frozen=True)
class OwnerStyleMaterializationPlan:
    schema_version: int
    owner_id: str
    page_id: str
    visual_profile_sha256: str
    intent_sha256: str
    render_layout_contract_sha256: str
    rendered_x_height_px: float
    unit_resolution_sha256: str
    attribute_plans: Mapping[str, MaterializationAttributePlan]
    plan_sha256: str

@dataclass(frozen=True)
class OwnerStyleMaterializationObservation:
    schema_version: int
    owner_id: str
    page_id: str
    visual_profile_sha256: str
    plan_sha256: str
    attributes: Mapping[str, Mapping[str, Any]]
    render_completed: bool
    observation_sha256: str
```

Cada intent/plano/observação deve aplicar deep-freeze aos mappings internos antes de calcular JSON/hash canônico; `frozen=True` isolado não basta. Cada observação materializada exige `domain`, `canonical_value`, `evidence_kind`, `evidence_sha256` e métricas. O builder recusa valores sem evidência e nunca copia automaticamente `intent_value`/`target_value`. O plano imutável só pode ser selado depois que fit, resolução de fonte e capabilities produzirem valores executáveis, mas obrigatoriamente antes de rasterizar; a comparação usa `target_value`, enquanto `intent_value` preserva a intenção e qualquer ajuste exige `resolution_kind` e razão canônica.

Valores `*_xh` permanecem unit-bearing no intent. O plano registra `rendered_x_height_px`, converte determinísticamente cada largura/radius/offset para px antes do raster e guarda o hash da resolução de unidades. A observação registra px reais e o ratio xh recomputado. Nunca compare diretamente `width_xh` com `width_px`, nem recalcule target a partir do raster observado.

Relações entre atributos são parte do plano: gradient que torna o solid fill não observável marca fill como `superseded_by=gradient`; multistroke de duas ou mais camadas supersede stroke; um único stroke permanece `stroke`. A comparação exige ausência de observação independente para o atributo superseded e evidência real para o atributo vencedor. Nunca copie fill solicitado para fingir que gradient o materializou.

**Step 4: Confirmar GREEN**

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_style_materialization.py pipeline/tests/test_style_contract.py -q
if ($LASTEXITCODE -ne 0) { throw 'Task 3 GREEN failed' }
```

**Step 5: Commit**

```powershell
git add -- pipeline/typesetter/style_materialization.py pipeline/tests/test_style_materialization.py pipeline/tests/test_style_contract.py
if ($LASTEXITCODE -ne 0) { throw 'Task 3 staging failed' }
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Task 3 staged diff check failed' }
git diff --cached --name-status
if ($LASTEXITCODE -ne 0) { throw 'Task 3 staged path inspection failed' }
git commit -m "feat(style): define immutable materialization evidence"
if ($LASTEXITCODE -ne 0) { throw 'Task 3 commit failed' }
```

### Task 4: Introduzir o contrato raster v2 em paralelo, sem quebrar produtores v1

**Files:**

- Modify: `pipeline/ownership/model.py` em `OwnerStyleRasterContract`, constantes/hash/validator e `OwnerGlyphPatch`.
- Modify: `pipeline/tests/style_v2_fixtures.py`
- Modify: `pipeline/tests/test_style_contract.py`
- Modify: `pipeline/tests/test_owner_atomic_execution.py`
- Modify: `pipeline/tests/test_style_fidelity_qa.py`

**Step 1: Escrever testes RED do schema v2**

```python
def test_owner_raster_contract_v2_binds_plan_observation_and_comparison():
    contract = valid_owner_style_raster_contract_v2()
    payload = validate_owner_style_raster_contract(contract)
    assert payload["schema_version"] == 2
    assert payload["style_intent_sha256"] == payload["materialization_plan"]["intent_sha256"]
    assert payload["materialization_plan_sha256"] == payload["materialization_plan"]["plan_sha256"]
    assert payload["materialization_observation_sha256"] == payload["materialization_observation"]["observation_sha256"]
    assert payload["materialization_comparison"]["status"] == "match"


@pytest.mark.parametrize("field", [
    "materialization_plan", "materialization_observation", "materialization_comparison"
])
def test_owner_raster_contract_v2_rejects_tampered_materialization(field):
    payload = valid_owner_style_raster_contract_v2().to_dict()
    tamper_nested_payload(payload[field])
    payload["contract_sha256"] = owner_style_raster_contract_sha256(payload)
    with pytest.raises(ValueError, match="materialization"):
        validate_owner_style_raster_contract(payload)
```

Adicione teste garantindo que os campos legados `requested_attributes`, `applied_attributes` e `abstained_attributes` são projeções determinísticas dos novos contratos e não podem divergir.

**Step 2: Confirmar RED**

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_style_contract.py pipeline/tests/test_owner_atomic_execution.py -k "materialization or raster_contract_v2" -q
if ($LASTEXITCODE -ne 1) { throw "Expected behavioral RED exit 1, got ${LASTEXITCODE}: raster contract schema v2 is missing" }
```

**Step 3: Implementar schema v2 paralelo e projeções legadas determinísticas**

Adicione ao contrato:

```python
materialization_plan_sha256: str
materialization_observation_sha256: str
style_intent_sha256: str
materialization_plan: Mapping[str, Any]
materialization_observation: Mapping[str, Any]
materialization_comparison: Mapping[str, Any]
backend_selection_reason: str
```

Implemente `OwnerStyleRasterContractV2` e seu validator ao lado do tipo v1; nesta task não altere ainda o alias/produtor default do renderer. `OwnerGlyphPatch` aceita a união tipada somente durante a migração, e o switch de owner-enforce ocorre na Task 7 junto com todos os call sites.

Atualize self-hash, deep-freeze/thaw, campos obrigatórios e validação. Separe `render_status` (`completed|failed`) de `materialization_status` (`match|mismatch|review_required`). As projeções legadas são derivadas, nunca autoritativas:

- `requested_attributes`: nomes/valores do resolved intent;
- `applied_attributes`: somente atributos target que realmente deram match, e vazio se o status global não for match;
- `abstained_attributes`: abstentions aprovadas, superseded e mismatch/unavailable com razões canônicas;
- a observação diagnóstica completa permanece em `materialization_observation` mesmo quando há raster não vazio e mismatch.

O validator recalcula essas projeções e exige igualdade. Não implemente fallback de schema v1 no caminho `enforce` depois do switch da Task 7; fixtures antigas devem migrar explicitamente.

**Step 4: Confirmar GREEN amplo do modelo**

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_style_contract.py pipeline/tests/test_owner_atomic_execution.py pipeline/tests/test_style_fidelity_qa.py pipeline/tests/test_owner_layout.py -q
if ($LASTEXITCODE -ne 0) { throw 'Task 4 GREEN failed' }
```

**Step 5: Commit**

```powershell
git add -- pipeline/ownership/model.py pipeline/tests/style_v2_fixtures.py pipeline/tests/test_style_contract.py pipeline/tests/test_owner_atomic_execution.py pipeline/tests/test_style_fidelity_qa.py
if ($LASTEXITCODE -ne 0) { throw 'Task 4 staging failed' }
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Task 4 staged diff check failed' }
git diff --cached --name-status
if ($LASTEXITCODE -ne 0) { throw 'Task 4 staged path inspection failed' }
git commit -m "feat(style): version owner raster materialization contract"
if ($LASTEXITCODE -ne 0) { throw 'Task 4 commit failed' }
```

### Task 5: Tornar resolução contextual coerente sem apagar decisão owner-local

**Files:**

- Modify: `pipeline/typesetter/style_groups.py`
- Modify: `pipeline/typesetter/owner_style.py`
- Modify: `pipeline/tests/test_style_groups.py`
- Modify: `pipeline/tests/test_owner_style_profile.py`

**Step 1: Escrever os testes RED que reproduzem os cards**

```python
def test_group_resolution_preserves_compatible_owner_local_effects():
    profile = applied_profile(
        owner_id="card",
        applied={
            "fill": "#FFFFFF",
            "stroke": {"color": "#161616", "width_px": 2},
            "glow": {"color": "#FFD34D", "width_px": 3},
        },
    )
    resolved = resolve_contextual_style_groups({"card": profile})["card"]
    intent = resolved["style_resolved_intent_v1"]
    assert set(intent["approved_attributes"]) >= {"fill", "stroke", "glow"}
    assert resolved["style_application_decision_v2"] == profile["style_application_decision_v2"]
    assert "style_materialization_plan_v1" not in resolved


def test_group_inheritance_never_replaces_high_confidence_owner_decision():
    resolved = resolve_contextual_style_groups(
        {
            "a": applied_profile("a", font="ComicNeue-Bold.ttf", confidence=.99),
            "b": applied_profile("b", font="KOMIKAX_.ttf", confidence=.98),
        }
    )
    assert resolved["a"]["style_resolved_intent_v1"]["approved_attributes"]["font_name"] == "ComicNeue-Bold.ttf"
    assert resolved["b"]["style_resolved_intent_v1"]["approved_attributes"]["font_name"] == "KOMIKAX_.ttf"


def test_group_resolution_records_explicit_conflict_instead_of_silent_drop():
    resolved = resolve_contextual_style_groups(conflicting_group())
    for profile in resolved.values():
        assert profile["style_group_resolution_v3"]["conflicts"]
        assert not silent_removed_applied_attributes(profile)
```

**Step 2: Confirmar RED**

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_style_groups.py pipeline/tests/test_owner_style_profile.py -k "preserves_compatible or never_replaces or explicit_conflict" -q
if ($LASTEXITCODE -ne 1) { throw "Expected behavioral RED exit 1, got ${LASTEXITCODE}: contextual group resolution still drifts" }
```

Expected: a implementação atual escolhe uma classe de efeito, remove as demais chaves e/ou altera a decisão.

**Step 3: Implementar resolução derivada**

- Mantenha `style_application_decision_v2` intacta.
- Gere `style_group_resolution_v3` com donor, conflito, atributos herdados e razões.
- Construa `style_resolved_intent_v1` pelo contrato da Task 3, ligado aos hashes da decisão e da resolução de grupo; não sele `style_materialization_plan_v1` antes do fit/geometry.
- Só herde um atributo quando o owner alvo o abstiver e a política permitir compartilhamento.
- Não trate stroke, glow, shadow e gradient como mutuamente exclusivos quando o owner tem evidência compatível para mais de um.
- Conflitos já determináveis sem layout viram abstention/review no intent; ajustes dependentes de capability/safe geometry pertencem ao plano pós-fit da Task 7.
- Preserve `visual_profile_v2` e seu hash como evidência aprovada imutável. `style_group_resolution_v3` e `style_resolved_intent_v1` são contratos irmãos hash-bound e nunca são inseridos dentro do profile; o profile/hash aprovado não muda.

**Step 4: Confirmar GREEN**

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_style_groups.py pipeline/tests/test_owner_style_profile.py pipeline/tests/test_style_policy_v2.py -q
if ($LASTEXITCODE -ne 0) { throw 'Task 5 GREEN failed' }
```

**Step 5: Commit**

```powershell
git add -- pipeline/typesetter/style_groups.py pipeline/typesetter/owner_style.py pipeline/tests/test_style_groups.py pipeline/tests/test_owner_style_profile.py
if ($LASTEXITCODE -ne 0) { throw 'Task 5 staging failed' }
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Task 5 staged diff check failed' }
git diff --cached --name-status
if ($LASTEXITCODE -ne 0) { throw 'Task 5 staged path inspection failed' }
git commit -m "fix(style): keep group decisions and materialization coherent"
if ($LASTEXITCODE -ne 0) { throw 'Task 5 commit failed' }
```

### Task 6: Fazer o glyph compositor reportar apenas fatos raster observados

**Files:**

- Modify: `pipeline/typesetter/glyph_rasterizer.py`
- Modify: `pipeline/typesetter/renderer.py` em `_render_v2_owner_text_layer()`.
- Modify: `pipeline/tests/test_glyph_rasterizer.py`
- Modify: `pipeline/tests/test_typesetting_renderer.py`

**Step 1: Escrever os testes RED**

```python
def test_raster_observation_contains_actual_clamped_transforms_and_effects():
    result = rasterize_v2_glyph_layers(
        glyph_mask(), safe_mask(),
        {
            "fill": "fff",
            "slant_tangent": .25,
            "stroke": {"color": "000", "width_px": 2},
            "glow": {"color": "#ffcc00", "width_px": 3},
        },
        rendered_x_height_px=20,
    )
    assert result.observed_attributes["slant_tangent"] == 0.25
    assert result.observed_attributes["fill"] == "#FFFFFF"
    assert result.observed_attributes["stroke"]["width_px"] == 2
    assert result.observed_attributes["glow"]["width_px"] == 3
    assert result.attribute_evidence_sha256["fill"]


def test_glyph_compositor_never_claims_font_or_layout_attributes():
    result = rasterize_v2_glyph_layers(
        glyph_mask(), safe_mask(), {"fill": "#fff"}, rendered_x_height_px=20
    )
    assert not ({"font_name", "font_weight", "font_width", "font_size_px", "alignment", "container", "tracking_xh"} & set(result.observed_attributes))


def test_owner_fill_stroke_and_glow_produces_nonempty_raster():
    block = verified_v2_owner_block(applied={"fill": "#FFFFFF", "stroke": stroke(), "glow": glow()})
    result = render_owner_block(block)
    assert result.status == "applied"
    assert np.count_nonzero(result.rgba[:, :, 3]) > 0
    assert block["fit_status"] != "style_attribute_not_materialized"


def test_multistroke_observation_preserves_layer_order_and_actual_widths():
    result = rasterize_v2_glyph_layers(
        glyph_mask(), safe_mask(),
        {"multistroke": [stroke_px("#000", 4), stroke_px("#FFF", 2)]},
        rendered_x_height_px=20,
    )
    assert [layer["width_px"] for layer in result.observed_attributes["multistroke"]] == [4, 2]
    assert [layer["color"] for layer in result.observed_attributes["multistroke"]] == ["#000000", "#FFFFFF"]


def test_out_of_range_raster_intent_cannot_become_match_by_clamping():
    plan = resolve_plan(intent={"slant_tangent": 9.0})
    assert plan.attribute_plans["slant_tangent"].resolution_kind == "review_required"
    diagnostic = rasterize_diagnostic(plan, clamped_slant=.75)
    assert compare_materialization(plan, diagnostic.observation).status != "match"


def test_enforce_rasterizer_rejects_unresolved_xheight_units():
    with pytest.raises(ValueError, match="unresolved execution units"):
        rasterize_v2_glyph_layers(glyph_mask(), safe_mask(), {"stroke": {"width_xh": .1}}, enforce=True)


def test_fill_observation_uses_composited_pixels_when_backend_output_differs_from_target():
    backend = backend_returning_fill_pixels("#00FF00", falsely_declared_fill="#FF0000")
    result = rasterize_v2_glyph_layers(
        glyph_mask(), safe_mask(), {"fill": "#FF0000"}, rendered_x_height_px=20, backend=backend
    )
    assert result.observed_attributes["fill"] == "#00FF00"
    assert result.attribute_evidence["fill"]["evidence_kind"] == "layer_pixels_and_mask"
    assert compare_attribute("fill", "#FF0000", result.observed_attributes["fill"]).matches is False


def test_missing_effect_layer_is_unavailable_even_when_glow_was_requested():
    backend = backend_omitting_effect_layer_but_echoing_requested_metadata("glow")
    result = rasterize_v2_glyph_layers(
        glyph_mask(), safe_mask(), {"fill": "#FFFFFF", "glow": glow()}, rendered_x_height_px=20, backend=backend
    )
    assert "glow" not in result.observed_attributes
    assert result.unavailable_attributes["glow"] == "missing_observable_effect_layer"


def test_stroke_width_observation_is_measured_from_layer_masks_not_request_metadata():
    backend = backend_returning_stroke_mask(actual_width_px=5, falsely_declared_width_px=2)
    result = rasterize_v2_glyph_layers(
        glyph_mask(), safe_mask(), {"stroke": stroke_px("#000", 2)}, rendered_x_height_px=20, backend=backend
    )
    assert result.observed_attributes["stroke"]["width_px"] == pytest.approx(5, abs=0.5)
    assert compare_attribute("stroke", stroke_px("#000", 2), result.observed_attributes["stroke"]).matches is False
```

Adicione casos para shadow, gradient, rotation, slant, width, scale e efeito fora da safe-region. `curve` e `tracking_xh` não pertencem a este observer: são provados pelo glyph-run/layout da Task 7. O teste de abstention por envelope deve continuar zerando somente o atributo/owner conforme a política, nunca inventar observação.

**Step 2: Confirmar RED**

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_glyph_rasterizer.py pipeline/tests/test_typesetting_renderer.py -k "observed_attributes or never_claims_font or fill_stroke_and_glow or multistroke_observation or out_of_range or unresolved_xheight or composited_pixels or missing_effect_layer or measured_from_layer_masks" -q
if ($LASTEXITCODE -ne 1) { throw "Expected behavioral RED exit 1, got ${LASTEXITCODE}: rasterizer still mixes domains or blocks valid raster" }
```

**Step 3: Implementar observação raster real**

Renomeie semanticamente `GlyphRasterResult.applied_attributes` para `observed_attributes` e adicione evidência por atributo. Se for necessária compatibilidade interna temporária, exponha uma property read-only; não mantenha duas fontes mutáveis.

O rasterizer recebe somente `target_value` em unidades de execução já resolvidas pelo plano e registra fatos pós-render:

- transforms efetivamente aplicados;
- cores sRGB realmente pintadas;
- larguras/radius/offsets em px efetivamente usados e ratio xh diagnóstico recomputado com o `rendered_x_height_px` do plano;
- offsets/radius reais;
- endpoints/direção de gradient;
- envelopes e contagens reais.

Defina uma interface interna de resultado de backend com RGBA/layers/masks, mas trate metadata declarada pelo backend apenas como diagnóstico. O observer mede cores nos pixels restritos às máscaras de layer, deriva largura/offset/radius pela geometria core/effect e inclui hashes dos bytes de pixels+masks. Testes adversariais injetam metadata ecoada que contradiz pixels, layer ausente e stroke mask com largura diferente; qualquer implementação que copie `target_value` ou metadata não autenticada deve falhar.

Quando gradient é o vencedor, observe gradient pelos pixels/camadas efetivamente compostos e não ecoe fill. Quando multistroke vence, observe `multistroke` com paint order, cores e larguras pós-conversão; `stroke` fica ausente conforme a relação `superseded_by` do plano.

Em enforce, rejeite qualquer `*_xh` que tenha chegado ao rasterizer sem resolução. Clamp extremo ou efeito que só no raster excedeu a safe-region não pode reescrever o plano/abstention retroativamente: emita observation `mismatch|unavailable` e mantenha o owner em review. Somente envelope/capability provado antes do raster pode gerar `abstained|review_required` no plano selado.

Remova de `_render_v2_owner_text_layer()`:

- cópia de `font_weight`/`font_width` da decisão;
- cópia de fill solicitado para esconder gradient;
- cálculo global de `unresolved` contra todos os atributos da decisão.

Esse método só pode bloquear por falha raster real: core vazio, core/effect fora da safe-region, capability raster ausente ou composição vazia.

**Step 4: Confirmar GREEN e parity**

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_glyph_rasterizer.py pipeline/tests/test_typesetting_renderer.py pipeline/tests/test_renderer_backend_parity.py pipeline/tests/test_sfx_renderer.py -q
if ($LASTEXITCODE -ne 0) { throw 'Task 6 GREEN failed' }
```

**Step 5: Commit**

```powershell
git add -- pipeline/typesetter/glyph_rasterizer.py pipeline/typesetter/renderer.py pipeline/tests/test_glyph_rasterizer.py pipeline/tests/test_typesetting_renderer.py
if ($LASTEXITCODE -ne 0) { throw 'Task 6 staging failed' }
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Task 6 staged diff check failed' }
git diff --cached --name-status
if ($LASTEXITCODE -ne 0) { throw 'Task 6 staged path inspection failed' }
git commit -m "fix(style): report actual raster materialization"
if ($LASTEXITCODE -ne 0) { throw 'Task 6 commit failed' }
```

### Task 7: Observar layout e fonte e impedir renormalização do perfil V2

**Files:**

- Modify: `pipeline/typesetter/renderer.py` em `_apply_auto_style_policy_if_needed()`, `_render_owner_text_block()`, `_build_owner_render_blocks()`, `_owner_style_contract_attributes()` e `_build_owner_style_raster_contract()`.
- Modify: `pipeline/typesetter/backend_contract.py`
- Modify: `pipeline/ownership/model.py` para tornar v2 obrigatório no owner-enforce após migrar todos os produtores.
- Modify: `pipeline/tests/test_typesetting_renderer.py`
- Modify: `pipeline/tests/test_typesetting_backend_contract.py`
- Modify: `pipeline/tests/test_owner_layout.py`
- Modify: `pipeline/tests/test_owner_compositor.py`
- Modify: `pipeline/tests/test_strip_owner_composition_integration.py`
- Modify: `pipeline/tests/test_strip_process_bands.py`

**Step 1: Escrever os testes RED**

```python
def test_verified_v2_owner_style_is_not_renormalized_as_auto():
    block = verified_v2_owner_block(style_origin=None, fill="#F8F8F8")
    profile_before = deepcopy(block["visual_profile_v2"])
    intent_before = deepcopy(block["style_resolved_intent_v1"])
    patch = render_owner_block(block)
    assert block["visual_profile_v2"] == profile_before
    assert block["style_resolved_intent_v1"] == intent_before
    assert patch.style_raster_contract.materialization_plan["intent_sha256"] == intent_before["intent_sha256"]
    assert block["style_origin"] == "owner_style_v2"


def test_layout_observation_uses_final_fit_not_requested_values():
    patch = render_owner_with_fit(requested_font_size=48, actual_font_size=36)
    observed = patch.style_raster_contract.materialization_observation
    assert observed["attributes"]["font_size_px"]["canonical_value"] == 36
    assert observed["attributes"]["alignment"]["evidence_kind"] == "final_glyph_positions"


def test_alignment_is_derived_from_final_positions_not_echoed_from_request():
    patch = render_owner_with_positions(requested_alignment="center", final_center_offset_x=24)
    comparison = patch.style_raster_contract.materialization_comparison
    assert comparison["status"] == "mismatch"
    assert comparison["mismatches"][0]["attribute"] == "alignment"


def test_font_observation_uses_resolved_file_instead_of_request_echo():
    patch = render_owner_with_font(
        requested="ComicNeue-Bold.ttf",
        resolved="ComicNeue-Regular.ttf",
    )
    result = patch.style_raster_contract.materialization_comparison
    assert result["status"] == "mismatch"
    assert result["mismatches"][0]["attribute"] in {"font_name", "font_weight"}


def test_complete_observation_merges_disjoint_domains():
    patch = render_verified_owner_with_font_layout_and_glow()
    observed = patch.style_raster_contract.materialization_observation["attributes"]
    assert {row["domain"] for row in observed.values()} == {"layout", "font", "raster"}
    assert all(row["evidence_sha256"] for row in observed.values() if row["status"] == "materialized")


def test_tracking_and_curve_are_observed_from_final_glyph_run():
    patch = render_curved_owner(tracking_xh=.15, curve=verified_arc())
    observed = patch.style_raster_contract.materialization_observation["attributes"]
    assert observed["tracking_xh"]["evidence_kind"] == "glyph_run"
    assert observed["curve"]["evidence_kind"] == "glyph_run"
    assert observed["curve"]["metrics"]["glyph_poses_sha256"]


def test_style_v2_enforce_never_selects_backend_without_observation_contract():
    patch = render_verified_owner(rust_renderer_enabled=True, enforce=True)
    contract = patch.style_raster_contract
    assert contract.backend == "python_v2"
    assert contract.backend_selection_reason == "rust_missing_materialization_observation_v2"
    assert "materialization_observation_v2" in contract.capabilities
```

**Step 2: Confirmar RED**

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_typesetting_renderer.py pipeline/tests/test_typesetting_backend_contract.py pipeline/tests/test_owner_layout.py -k "renormalized_as_auto or final_fit or request_echo or disjoint_domains or final_glyph_run or observation_contract" -q
if ($LASTEXITCODE -ne 1) { throw "Expected behavioral RED exit 1, got ${LASTEXITCODE}: layout/font observers are missing" }
```

**Step 3: Implementar binders por domínio**

- Marque blocos com `visual_profile_v2` e `style_resolved_intent_v1` válidos/hash-bound como `style_origin=owner_style_v2` antes da policy automática; o plano ainda não existe nesse ponto.
- A policy automática só preenche atributos sem intent aprovado; nunca sobrescreve `target_value` de plano selado.
- O observer de layout lê `font_size_final`, linhas/posições, safe polygon, container source/hash e occupancy após o fit. Alignment é derivado de margens esquerda/direita, center offset e bboxes finais com tolerância versionada; nunca ecoe a string solicitada.
- Defina um `RenderedGlyphRun` imutável com codepoint/glyph id, font-span, origem, advance, bbox e pose/ângulo final. O observer de layout deriva `tracking_xh` e `curve` desse run; não os infere por connected components nem por uma máscara union.
- Remova `tracking_xh` de `_render_v2_owner_text_layer.raster_style` e da transformação por connected components. Aplique tracking aos advances antes de wrap/fit e inclua-o na medição de largura; migre a pose do arc renderer para o mesmo `RenderedGlyphRun`, garantindo que tracking/curve não sejam aplicados duas vezes.
- O observer de fonte recebe o `FontRunObservation` completo da Task 2, incluindo fallbacks e eixos realmente configurados; nunca usa apenas a string solicitada ou `SafeTextPathFont.font_path`.
- A resolução pré-raster sela um `FontRunPlan` para o payload final. Fallback por cobertura PT-BR é permitido apenas como `derived:glyph_coverage_fallback`, com style-distance/provenance e spans exatos; qualquer face adicional observada diverge.
- Migre `_render_text_with_fallback()` para retornar bitmap + `FontRunObservation` (ou um result dataclass equivalente) e atualize todos os callers; a observação vem das faces realmente usadas por span, não de uma pré-resolução desconectada do raster.
- O observer raster recebe exclusivamente `GlyphRasterResult` da Task 6.
- Execute a ordem anti-echo sem atalhos: `resolved intent -> fit/layout/font/capability resolution -> seal plan/hash -> rasterize(plan) -> observe -> compare`. Depois do fit e da resolução de fonte/capability, sele uma única vez o `OwnerStyleMaterializationPlan`: intenção em `intent_value`, valor executável em `target_value`, `resolution_kind` e razão explícita para todo ajuste/relação. Ligue-o a `style_resolved_intent_v1`, ao hash do `render_layout_contract` final escolhido, ao x-height final e à conversão de unidades. Agregue os três observers em `OwnerStyleMaterializationObservation`; só então compute comparação e contrato raster v2. O builder do plano não recebe `GlyphRasterResult` nem qualquer observação pós-render.
- Persista resolved intent completo e plan SHA no owner execution record antes do raster. O patch carrega o contrato/hash produzido, mas nunca se torna a autoridade de seu próprio expected; `apply_atomic_owner_execution()` recebe os valores selados do record.
- Permita `policy_adjusted` apenas por whitelist determinística: `font_size_px` pode ajustar dentro de ratio `[0.70, 1.10]` e acima do legibility floor; `container` pode ser `derived` somente de `OwnerRenderGeometry`; alignment, tracking e curve permanecem exact ou review. Fontes, fill e efeitos não podem adotar o valor observado pós-hoc. QA publica separadamente `intent_to_target_deviation` e `target_to_observation_match`, para que 48->36 possa materializar corretamente sem esconder perda excessiva de estilo.
- `render_completed` depende de pixels, fit, render quality e observação completa. Um mismatch material produz contrato completo `review_required`; não apaga o raster diagnóstico.

Atualize capabilities de backend para refletir quem executa cada atributo, sem confundir capability com evidência observada. Adicione capability separada `materialization_observation_v2`: em owner Style V2 `enforce`, backend sem essa capability nunca é selecionado. Enquanto Rust retornar apenas bool/raster, selecione `python_v2` com razão auditável; se nenhum backend observável estiver disponível, bloqueie em vez de executar fallback silencioso.

Depois que o produtor Python e todos os constructors diretos estiverem migrados, faça o switch do owner-enforce para `OwnerStyleRasterContractV2`. Modo legado pode ler v1 com audit explícito; owner-enforce rejeita v1. Rode também os testes de compositor/strip para provar que não ficou constructor v1 oculto.

**Step 4: Confirmar GREEN**

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_typesetting_renderer.py pipeline/tests/test_typesetting_backend_contract.py pipeline/tests/test_owner_layout.py pipeline/tests/test_owner_render_quality.py pipeline/tests/test_owner_compositor.py pipeline/tests/test_strip_owner_composition_integration.py pipeline/tests/test_strip_process_bands.py -q
if ($LASTEXITCODE -ne 0) { throw 'Task 7 GREEN failed' }
```

**Step 5: Checkpoint R5-B parcial em perfil realista**

Run:

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_style_groups.py pipeline/tests/test_glyph_rasterizer.py pipeline/tests/test_typesetting_renderer.py pipeline/tests/test_owner_layout.py -k "card or glow or materialization or complete_payload" -q
if ($LASTEXITCODE -ne 0) { throw 'R5-B partial checkpoint failed' }
```

Expected: perfil card com fill+stroke/glow produz tinta, contrato completo e nenhuma renormalização.

**Step 6: Commit**

```powershell
git add -- pipeline/typesetter/renderer.py pipeline/typesetter/backend_contract.py pipeline/ownership/model.py pipeline/tests/test_typesetting_renderer.py pipeline/tests/test_typesetting_backend_contract.py pipeline/tests/test_owner_layout.py pipeline/tests/test_owner_compositor.py pipeline/tests/test_strip_owner_composition_integration.py pipeline/tests/test_strip_process_bands.py
if ($LASTEXITCODE -ne 0) { throw 'Task 7 staging failed' }
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Task 7 staged diff check failed' }
git diff --cached --name-status
if ($LASTEXITCODE -ne 0) { throw 'Task 7 staged path inspection failed' }
git commit -m "fix(style): observe layout and font materialization"
if ($LASTEXITCODE -ne 0) { throw 'Task 7 commit failed' }
```

### Task 8: Comparar observação canônica no commit atômico

**Files:**

- Modify: `pipeline/strip/process_bands.py` somente em `apply_atomic_owner_execution()` e chamada do commit.
- Modify: `pipeline/tests/test_owner_atomic_execution.py`
- Modify: `pipeline/tests/test_owner_enforcement.py`

**Step 1: Inspecionar o diff local antes do RED**

```powershell
git diff -- pipeline/strip/process_bands.py
git diff -- pipeline/tests/test_owner_atomic_execution.py pipeline/tests/test_owner_enforcement.py
```

Expected: preservar integralmente os hunks locais de máscara, source replacement, safe geometry e rollback diagnostics.

**Step 2: Escrever os testes RED**

```python
def test_atomic_commit_accepts_canonically_equivalent_observation():
    sealed = prepare_owner_execution(intent_fill="#fff")
    patch = render_patch_v2(sealed, observed_fill="#FFFFFF")
    commit = apply_atomic_owner_execution(
        original(), valid_mutation(), patch,
        expected_style_intent=sealed.style_resolved_intent,
        expected_materialization_plan_sha256=sealed.materialization_plan_sha256,
    )
    assert commit.committed is True


def test_atomic_commit_does_not_compare_functional_layout_to_style_decision():
    sealed = prepare_owner_execution(intent_font_size=48, target_font_size=36)
    patch = render_patch_v2(sealed, observed_font_size=36)
    commit = apply_atomic_owner_execution(
        original(), valid_mutation(), patch,
        expected_style_intent=sealed.style_resolved_intent,
        expected_materialization_plan_sha256=sealed.materialization_plan_sha256,
    )
    assert commit.committed is True


def test_atomic_commit_rolls_back_true_material_divergence_with_precise_reason():
    sealed = prepare_owner_execution(intent_fill="#FFFFFF")
    patch = render_patch_v2(sealed, observed_fill="#FF0000")
    commit = apply_atomic_owner_execution(
        original(), valid_mutation(), patch,
        expected_style_intent=sealed.style_resolved_intent,
        expected_materialization_plan_sha256=sealed.materialization_plan_sha256,
    )
    assert commit.committed is False
    assert commit.reason == "render_contract_invalid:materialization_mismatch:raster:fill:canonical_value_mismatch"


def test_atomic_commit_rejects_self_consistent_rehashed_patch_that_diverges_from_sealed_plan():
    sealed = prepare_owner_execution(intent_fill="#FFFFFF")
    patch = tamper_plan_observation_comparison_and_rehash(render_patch_v2(sealed), fill="#FF0000")
    commit = apply_atomic_owner_execution(
        original(), valid_mutation(), patch,
        expected_style_intent=sealed.style_resolved_intent,
        expected_materialization_plan_sha256=sealed.materialization_plan_sha256,
    )
    assert commit.committed is False
    assert commit.reason == "render_contract_invalid:sealed_materialization_plan_mismatch"
```

Cubra fonte resolvida errada, evidência ausente, hash adulterado, status unavailable, plano/profile divergente e comparação forjada.

**Step 3: Confirmar RED**

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_owner_atomic_execution.py pipeline/tests/test_owner_enforcement.py -k "canonically_equivalent or functional_layout or material_divergence or self_consistent_rehashed" -q
if ($LASTEXITCODE -ne 1) { throw "Expected behavioral RED exit 1, got ${LASTEXITCODE}: atomic commit still uses raw equality" }
```

**Step 4: Substituir igualdade crua por recomputação canônica**

No commit:

1. valide schema/hash do plano e observação;
2. confirme os bindings owner/page/profile/pixel já disponíveis; a Task 10 acrescentará o binding de `OwnerRenderGeometry` no envelope externo sem criar dependência circular no contrato de estilo;
3. use como autoridade independente o resolved intent e o plan SHA selados no owner execution record antes do raster; valide que o plano carregado pelo patch tem esse SHA e liga exatamente intent/profile/render-layout hashes;
4. recompute `compare_materialization(plan, observation)`; não confie no comparison serializado;
5. aceite somente `match` completo para atributos resolved;
6. preserve abstentions aprovadas com razões canônicas;
7. em mismatch, retorne razão `materialization_mismatch:<domain>:<attribute>:<reason>`;
8. mantenha rollback inteiro de cleanup+raster.

Remova o loop de `decision_applied[name] != value`. Não relaxe checks de hash, render, fit, quality, envelopes ou protected art.

**Step 5: Confirmar GREEN e checkpoint R5-B**

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_style_materialization.py pipeline/tests/test_style_contract.py pipeline/tests/test_style_groups.py pipeline/tests/test_glyph_rasterizer.py pipeline/tests/test_typesetting_renderer.py pipeline/tests/test_owner_atomic_execution.py pipeline/tests/test_owner_enforcement.py pipeline/tests/test_owner_layout.py -q
if ($LASTEXITCODE -ne 0) { throw 'R5-B checkpoint failed' }
```

Expected: nenhum `render was not completed` causado por domínio errado; equivalências passam; divergências materiais continuam rollback.

**Step 6: Stage por hunk e commit**

Crie com `apply_patch` `$r5Tmp\task08-dirty-index.patch` contendo somente os hunks R5 de `pipeline/strip/process_bands.py` e `pipeline/tests/test_owner_enforcement.py`.

```powershell
$r5PlanAnchor = git log -1 --format=%H -- docs/plans/2026-08-01-style-copy-r5-materialization-geometry-remediation-implementation.md
$r5Tmp = Join-Path (Resolve-Path '.') ('.codex-tmp/style-copy-r5-' + $r5PlanAnchor.Substring(0, 12))
$r5ScopedPatch = Join-Path $r5Tmp 'task08-dirty-index.patch'
if (-not (Test-Path -LiteralPath $r5ScopedPatch)) { throw 'Create the Task 8 scoped index patch with apply_patch first' }
git apply --cached --check --unidiff-zero $r5ScopedPatch
if ($LASTEXITCODE -ne 0) { throw 'Task 8 scoped index patch does not apply cleanly' }
git apply --cached --unidiff-zero $r5ScopedPatch
if ($LASTEXITCODE -ne 0) { throw 'Task 8 scoped index patch failed' }
git add -- pipeline/tests/test_owner_atomic_execution.py
if ($LASTEXITCODE -ne 0) { throw 'Task 8 clean-path staging failed' }
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Task 8 staged diff check failed' }
$expectedTask08 = @('pipeline/strip/process_bands.py','pipeline/tests/test_owner_enforcement.py','pipeline/tests/test_owner_atomic_execution.py') | Sort-Object
$stagedTask08 = @(git diff --cached --name-only) | Sort-Object
if ($LASTEXITCODE -ne 0 -or (Compare-Object $expectedTask08 $stagedTask08)) { throw 'Task 8 staged path set is incomplete or contaminated' }
git diff --cached -- pipeline/strip/process_bands.py
git commit -m "fix(style): validate canonical materialization at owner commit"
if ($LASTEXITCODE -ne 0) { throw 'Task 8 commit failed' }
```

Expected: somente hunks de validação/integração desta task; demais mudanças locais em `process_bands.py` continuam unstaged.

### Task 9: Definir o contrato PageSurfaceGeometry e transformações únicas

**Files:**

- Create: `pipeline/strip/page_surface_geometry.py`
- Create: `pipeline/tests/test_page_surface_geometry.py`
- Modify: `pipeline/strip/types.py` em `OutputPage`.
- Modify: `pipeline/ownership/model.py` em `PageCompositionResult`.

**Step 1: Escrever os testes RED de round-trip**

Proteja o import do módulo novo com `find_spec()` ou skeleton importável e confirme exit 1 comportamental, nunca collection exit 2.

```python
def test_logical_bbox_round_trips_through_centered_frame():
    geometry = PageSurfaceGeometry.build(
        logical_width=690, logical_height=1600,
        frame_width=800, frame_height=1600,
        content_origin_xy=(55, 0),
    )
    logical = (109, 985, 318, 1104)
    assert geometry.logical_bbox_to_frame(logical) == (164, 985, 373, 1104)
    assert geometry.frame_bbox_to_logical((164, 985, 373, 1104)) == logical


def test_polygon_and_mask_use_same_transform_without_resampling():
    geometry = centered_geometry()
    assert geometry.logical_polygon_to_frame(((0, 0), (689, 1599))) == ((55, 0), (744, 1599))
    framed = geometry.logical_mask_to_frame(one_pixel_logical_mask(x=109, y=985))
    assert framed[985, 164] == 255
    assert np.count_nonzero(framed) == 1


@pytest.mark.parametrize("shape", [(1600, 690), (1600, 690, 1), (1600, 690, 3), (1600, 690, 4)])
def test_array_transform_preserves_trailing_dimensions_dtype_and_round_trip(shape):
    source = logical_array(shape, dtype=np.uint16)
    framed = centered_geometry().logical_array_to_frame(source, fill_value=0)
    assert framed.shape == (1600, 800, *shape[2:])
    assert framed.dtype == source.dtype
    assert np.array_equal(centered_geometry().frame_array_to_logical(framed), source)


def test_string_owner_map_round_trips_without_resize():
    owner_map = np.full((1600, 690), "", dtype=object)
    owner_map[985, 109] = "owner_a"
    framed = centered_geometry().logical_array_to_frame(owner_map, fill_value="")
    assert framed[985, 164] == "owner_a"


def test_page_surface_geometry_rejects_content_outside_frame_or_tampered_hash():
    with pytest.raises(ValueError):
        PageSurfaceGeometry.build(
            logical_width=900, logical_height=100,
            frame_width=800, frame_height=100,
            content_origin_xy=(0, 0),
        )
```

Cubra identidade sem frame (`690→690`, origin 0), offsets ímpares, máscara 2D/1/3/4 canais, owner map string, serialização e permutação. Use uma convenção única: bboxes inteiros são half-open `[x1,y1,x2,y2)`; polígonos usam coordenadas de borda no domínio contínuo `0 <= x <= width`, `0 <= y <= height` e uma regra única de rasterização por centro do pixel. Teste borda direita/inferior e não trate `x2/y2` como pixel inclusivo.

**Step 2: Confirmar RED**

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_page_surface_geometry.py -q
if ($LASTEXITCODE -ne 1) { throw "Expected behavioral RED exit 1, got ${LASTEXITCODE}: PageSurfaceGeometry is missing" }
```

**Step 3: Implementar o contrato**

```python
@dataclass(frozen=True)
class PageSurfaceGeometry:
    schema_version: int
    logical_space: Literal["logical_page"]
    artifact_space: Literal["framed_page"]
    logical_width: int
    logical_height: int
    frame_width: int
    frame_height: int
    content_origin_xy: tuple[int, int]
    content_bbox_frame: tuple[int, int, int, int]
    geometry_sha256: str

    def logical_bbox_to_frame(self, bbox: BBox) -> BBox:
        x, y = self.content_origin_xy
        x1, y1, x2, y2 = validate_logical_bbox(bbox, self.logical_size)
        return x1 + x, y1 + y, x2 + x, y2 + y

    def frame_bbox_to_logical(self, bbox: BBox) -> BBox:
        x, y = self.content_origin_xy
        x1, y1, x2, y2 = validate_frame_bbox(bbox, self.frame_size)
        return validate_logical_bbox((x1 - x, y1 - y, x2 - x, y2 - y), self.logical_size)

    def logical_polygon_to_frame(self, polygon: Sequence[Point]) -> tuple[Point, ...]:
        x, y = self.content_origin_xy
        return tuple((px + x, py + y) for px, py in validate_logical_polygon(polygon, self.logical_size))

    def logical_array_to_frame(self, source: np.ndarray, *, fill_value: Any = 0) -> np.ndarray:
        source = validate_logical_array(source, width=self.logical_width, height=self.logical_height)
        output_shape = (self.frame_height, self.frame_width, *source.shape[2:])
        output = np.full(output_shape, fill_value, dtype=source.dtype)
        x, y = self.content_origin_xy
        output[y:y + self.logical_height, x:x + self.logical_width] = source
        return output

    def frame_array_to_logical(self, source: np.ndarray) -> np.ndarray:
        source = validate_frame_array(source, width=self.frame_width, height=self.frame_height)
        x, y = self.content_origin_xy
        return source[y:y + self.logical_height, x:x + self.logical_width].copy()
```

Use uma única assinatura keyword-only: `PageSurfaceGeometry.build(*, logical_width, logical_height, frame_width, frame_height, content_origin_xy)`. Não crie aliases `logical_size`, `frame_size`, `content_origin_x` ou `content_origin_y`. Centralize validação, JSON canônico e transformações. `content_bbox_frame` é sempre derivado de origem+tamanho e validado no deserialize; proíba soma manual de offset e qualquer resize/resampling fora desse módulo.

Adicione `page_surface_geometry: PageSurfaceGeometry | None = None` a `OutputPage` e `page_surface_geometry_sha256: str | None = None` a `PageCompositionResult` como ponte de migração restrita às Tasks 9–10, para que constructors/callers legados ainda enumerados na Task 11 não quebrem no commit intermediário. O modo não-enforce pode carregar ausência/alias `page` somente com audit explícito; owner-enforce rejeita ambos. Na Task 11, depois de todos os produtores/callers migrarem, o runtime e os serializers distinguem dois estados: resultado interno do compositor com arrays `logical_page`, e novo resultado publicado com arrays `framed_page` mais geometry/hash obrigatórios. Mutations/glyph patches permanecem sempre em `logical_page`.

**Step 4: Confirmar GREEN**

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_page_surface_geometry.py pipeline/tests/test_owner_compositor.py pipeline/tests/test_strip_owner_composition_integration.py -q
if ($LASTEXITCODE -ne 0) { throw 'Task 9 GREEN failed' }
```

**Step 5: Commit**

```powershell
git add -- pipeline/strip/page_surface_geometry.py pipeline/strip/types.py pipeline/ownership/model.py pipeline/tests/test_page_surface_geometry.py
if ($LASTEXITCODE -ne 0) { throw 'Task 9 staging failed' }
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Task 9 staged diff check failed' }
git diff --cached --name-status
if ($LASTEXITCODE -ne 0) { throw 'Task 9 staged path inspection failed' }
git commit -m "feat(geometry): define logical page surface transforms"
if ($LASTEXITCODE -ne 0) { throw 'Task 9 commit failed' }
```

### Task 10: Criar OwnerRenderGeometry independente de cleanup

**Files:**

- Create: `pipeline/ownership/render_geometry.py`
- Create: `pipeline/tests/test_owner_render_geometry.py`
- Modify: `pipeline/ownership/model.py` em `OwnerMutation` e `OwnerGlyphPatch`.
- Modify: `pipeline/inpainter/owner_mask.py` em `OWNER_MASK_COORDINATE_SPACE`, plano/manifest e construção de `OwnerMutation`.
- Modify: `pipeline/compositor/owner_compositor.py` nos validators/envelopes owner-enforce.
- Modify: `pipeline/strip/process_bands.py` em `_owner_layout_regions()` e `execute_owner_page_graph()`.
- Modify: `pipeline/layout/balloon_layout.py` em `_enrich_owner_page_layout()`.
- Modify: `pipeline/typesetter/renderer.py` em `_render_owner_band_image()` e contrato raster.
- Modify: `pipeline/tests/test_owner_enforcement.py`
- Modify: `pipeline/tests/test_owner_layout.py`
- Modify: `pipeline/tests/test_balloon_layout_shared_regions.py`
- Modify: `pipeline/tests/test_owner_mask.py`
- Modify: `pipeline/tests/test_owner_compositor.py`

**Step 1: Inspecionar os hunks locais compartilhados**

```powershell
git diff -- pipeline/strip/process_bands.py pipeline/inpainter/owner_mask.py pipeline/ownership/reconcile.py pipeline/ownership/translation.py
```

Expected: preserve exatamente o patch-base registrado na Task 0; no snapshot de autoria, `process_bands.py` tinha 425 inserções/55 remoções relevantes, mas a autoridade é o diff capturado no início da execução. Esta task adiciona um contrato derivado e não reescreve reconciliação. O fallback local sobreposto `layout_container_bbox = source_replacement_bbox` é removido por hunk R5 explicitamente auditado, sem perder as demais linhas do mesmo hunk.

**Step 2: Escrever testes RED puros do contrato**

Proteja o import de `ownership.render_geometry` com `find_spec()` ou skeleton importável e confirme exit 1 comportamental antes de implementar.

```python
def test_owner_render_geometry_contains_complete_page_space_owner_evidence():
    geometry = build_owner_render_geometry(
        cross_tile_graph(), "owner_a",
        page_width=690, page_height=1600,
        container_evidence=balloon_polygon(),
    )
    assert geometry.logical_space == "logical_page"
    assert geometry.component_ids == ("component_a", "component_b")
    assert all(component.geometry_sha256 for component in geometry.components)
    assert all(observation.polygon_page for observation in geometry.selected_observations)
    assert all(projection.projection_sha256 for projection in geometry.projections)
    assert geometry.semantic_body_bbox_page == union_of_components()
    assert geometry.layout_container_source == "balloon_inner_polygon"
    assert geometry.source_replacement_bbox_page != geometry.layout_container_bbox_page
    assert len(geometry.geometry_sha256) == 64


def test_source_replacement_never_becomes_layout_container_without_evidence():
    geometry = build_owner_render_geometry(dialogue_without_container(), "owner_a", page_width=100, page_height=100)
    assert geometry.layout_container_bbox_page is None
    assert geometry.status == "review_required"
    assert geometry.reason == "missing_independent_dialogue_container"


def test_freeform_sfx_may_use_typed_component_union_not_cleanup_footprint():
    geometry = build_owner_render_geometry(freeform_sfx_graph(), "sfx", page_width=100, page_height=100)
    assert geometry.layout_container_source == "freeform_component_union"
    assert geometry.layout_container_bbox_page == geometry.semantic_body_bbox_page
```

Adicione testes de hashes/polígonos por componente e observação, connected subregions tipadas, offsets/roles por projection, independent container evidence IDs/confidence, `component_geometry_sha256`, foreign protection hash, hash adulterado e invariância à ordem/tile executor. Prove que duas geometrias com mesmos IDs/union bbox mas polígonos diferentes produzem hashes diferentes.

**Step 3: Confirmar RED**

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_owner_render_geometry.py -q
if ($LASTEXITCODE -ne 1) { throw "Expected behavioral RED exit 1, got ${LASTEXITCODE}: OwnerRenderGeometry is missing" }
```

**Step 4: Implementar o contrato puro**

```python
@dataclass(frozen=True)
class OwnerComponentGeometry:
    component_id: str
    bbox_page: BBox
    polygon_page: tuple[Point, ...]
    geometry_sha256: str

@dataclass(frozen=True)
class OwnerObservationGeometry:
    observation_id: str
    source_sha256: str
    bbox_page: BBox
    polygon_page: tuple[Point, ...]
    confidence: float
    observation_geometry_sha256: str

@dataclass(frozen=True)
class OwnerProjectionGeometry:
    projection_id: str
    tile_id: str
    role: str
    tile_bbox_page: BBox
    component_bbox_page: BBox
    tile_to_page_offset_xy: tuple[int, int]
    projection_sha256: str

@dataclass(frozen=True)
class ConnectedLayoutSubregion:
    subregion_id: str
    order: int
    component_ids: tuple[str, ...]
    bbox_page: BBox
    polygon_page: tuple[Point, ...]
    capacity_px2: int
    evidence_source: str
    evidence_ids: tuple[str, ...]
    subregion_sha256: str

@dataclass(frozen=True)
class OwnerRenderGeometry:
    schema_version: int
    owner_id: str
    page_id: str
    logical_space: Literal["logical_page"]
    page_width: int
    page_height: int
    component_ids: tuple[str, ...]
    components: tuple[OwnerComponentGeometry, ...]
    component_geometry_sha256: str
    selected_observations: tuple[OwnerObservationGeometry, ...]
    projections: tuple[OwnerProjectionGeometry, ...]
    semantic_body_bbox_page: BBox
    source_replacement_bbox_page: BBox
    layout_container_bbox_page: BBox | None
    layout_container_polygon_page: tuple[Point, ...] | None
    layout_container_source: str
    safe_polygons_page: tuple[tuple[Point, ...], ...]
    connected_subregions: tuple[ConnectedLayoutSubregion, ...]
    container_evidence_ids: tuple[str, ...]
    container_evidence_confidence: float
    protected_art_mask_sha256: str
    status: Literal["ready", "review_required"]
    reason: str
    geometry_sha256: str
```

Todos os subcontratos são deep-frozen, ordenados canonicamente e incluídos no hash; não use `tuple[Mapping, ...]`. Dimensões usam campos nomeados `page_width/page_height`, nunca tupla ambígua WH/HW.

Política sistêmica:

- dialogue/speech/thought exige balloon/layout evidence independente;
- card/UI aceita container visual verificado;
- freeform/SFX aceita component/observation union explicitamente classificada;
- `source_replacement_bbox_page` nunca participa da escolha do layout container;
- tile projection é provenance, nunca autoridade geométrica.

**Step 5: Escrever RED de integração antes de editar produção compartilhada**

```python
def test_source_replacement_bbox_never_becomes_owner_layout_container_without_container_evidence():
    record = execute_owner_fixture(dialogue_without_container())
    assert record["owner_render_geometry"]["source_replacement_bbox_page"]
    assert record["owner_render_geometry"]["layout_container_bbox_page"] is None
    assert record["route_action"] == "review_required"


def test_cross_tile_owner_layout_is_invariant_to_executor_tile_projection():
    first = build_layout_for_owner(executor_tile="tile_001")
    second = build_layout_for_owner(executor_tile="tile_002")
    assert first["owner_render_geometry_sha256"] == second["owner_render_geometry_sha256"]
    assert first["translated_payload"] == second["translated_payload"]


def test_single_region_owner_reconstructs_exact_payload_without_truncation():
    block = render_single_region_owner("CORPO COMPLETO SEM CORTE")
    assert " ".join(block["rendered_lines"]) == "CORPO COMPLETO SEM CORTE"
    assert block["render_completed"] is True


def test_mutation_and_glyph_patch_bind_same_owner_render_geometry_hash():
    mutation, glyph_patch = execute_owner_chain(verified_owner_fixture())
    assert mutation.owner_render_geometry_sha256 == glyph_patch.owner_render_geometry_sha256
    assert glyph_patch.owner_render_geometry.geometry_sha256 == mutation.owner_render_geometry_sha256


def test_owner_mask_plan_and_mutation_use_logical_page_and_bind_render_geometry():
    plan, mutation = build_owner_mask_and_mutation(verified_owner_fixture())
    assert plan.coordinate_space == mutation.coordinate_space == "logical_page"
    assert plan.owner_render_geometry_sha256 == mutation.owner_render_geometry_sha256


def test_owner_compositor_enforce_accepts_logical_page_and_rejects_legacy_page_alias():
    assert compose_owner_enforce(logical_page_owner_chain()).status == "committed"
    with pytest.raises(ValueError, match="legacy coordinate space"):
        compose_owner_enforce(logical_page_owner_chain(coordinate_space="page"))
```

Run:

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_owner_enforcement.py pipeline/tests/test_owner_layout.py pipeline/tests/test_balloon_layout_shared_regions.py pipeline/tests/test_owner_mask.py pipeline/tests/test_owner_compositor.py -k "source_replacement or executor_tile_projection or exact_payload or render_geometry_hash or logical_page or legacy_page_alias" -q
if ($LASTEXITCODE -ne 1) { throw "Expected behavioral RED exit 1, got ${LASTEXITCODE}: geometry is not propagated yet" }
```

**Step 6: Integrar antes do inpaint**

- Construa `OwnerRenderGeometry` uma vez, antes de mask/inpaint.
- Migre `OwnerMutation`, `OwnerGlyphPatch` e records v2 para `logical_space="logical_page"`. O alias legado `page` só pode ser canonicalizado em leitores non-enforce com evento de migração; não entra em hashes novos.
- Migre também `owner_mask.py`: constante/schema/manifest/deserialize e todo constructor de `OwnerMutation` usam `logical_page` e carregam `owner_render_geometry_sha256`; manifest legado só migra fora de enforce.
- Migre `owner_compositor.py`: validators de mutation/glyph exigem `logical_page`, recomputam `owner_render_geometry_sha256` além de `component_geometry_sha256` e nunca autorizam composição por um alias antigo.
- Nesta task, preserve temporariamente a semântica legada de uma única `glyph_mask`; a migração atômica para `glyph_core_mask`/`paint_mask` e containment por subconjunto ocorre na Task 12, junto com todos os constructors e testes, para não deixar um commit intermediário híbrido.
- Anexe o contrato ao record/layout payload por cópia imutável serializada.
- Use `source_replacement_bbox_page` somente para cleanup authorization.
- Use `layout_container_*`/safe polygons somente para layout.
- Vincule o mesmo `geometry_sha256` em `OwnerMutation`, `OwnerGlyphPatch`, render record e atomic commit. O contrato raster permanece focado em materialização; o envelope `OwnerGlyphPatch` autentica conjuntamente contrato raster e geometria, evitando dependência circular.
- Em geometry `review_required`, não execute inpaint; produza owner review auditável e bloqueie export.
- Remova o fallback `layout_container_bbox = source_replacement_bbox`.

**Step 7: Confirmar GREEN**

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_owner_render_geometry.py pipeline/tests/test_owner_enforcement.py pipeline/tests/test_owner_layout.py pipeline/tests/test_balloon_layout_shared_regions.py pipeline/tests/test_owner_atomic_execution.py pipeline/tests/test_owner_mask.py pipeline/tests/test_owner_compositor.py -q
if ($LASTEXITCODE -ne 0) { throw 'Task 10 GREEN failed' }
```

**Step 8: Stage por hunk e commit**

Crie com `apply_patch` `$r5Tmp\task10-dirty-index.patch` contendo somente os hunks R5 de `pipeline/strip/process_bands.py`, `pipeline/inpainter/owner_mask.py`, `pipeline/tests/test_owner_mask.py` e `pipeline/tests/test_owner_enforcement.py`.

```powershell
$r5PlanAnchor = git log -1 --format=%H -- docs/plans/2026-08-01-style-copy-r5-materialization-geometry-remediation-implementation.md
$r5Tmp = Join-Path (Resolve-Path '.') ('.codex-tmp/style-copy-r5-' + $r5PlanAnchor.Substring(0, 12))
$r5ScopedPatch = Join-Path $r5Tmp 'task10-dirty-index.patch'
if (-not (Test-Path -LiteralPath $r5ScopedPatch)) { throw 'Create the Task 10 scoped index patch with apply_patch first' }
git apply --cached --check --unidiff-zero $r5ScopedPatch
if ($LASTEXITCODE -ne 0) { throw 'Task 10 scoped index patch does not apply cleanly' }
git apply --cached --unidiff-zero $r5ScopedPatch
if ($LASTEXITCODE -ne 0) { throw 'Task 10 scoped index patch failed' }
git add -- pipeline/ownership/render_geometry.py pipeline/ownership/model.py pipeline/compositor/owner_compositor.py pipeline/layout/balloon_layout.py pipeline/typesetter/renderer.py pipeline/tests/test_owner_render_geometry.py pipeline/tests/test_owner_layout.py pipeline/tests/test_balloon_layout_shared_regions.py pipeline/tests/test_owner_compositor.py
if ($LASTEXITCODE -ne 0) { throw 'Task 10 clean-path staging failed' }
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Task 10 staged diff check failed' }
$expectedTask10 = @('pipeline/ownership/render_geometry.py','pipeline/ownership/model.py','pipeline/inpainter/owner_mask.py','pipeline/compositor/owner_compositor.py','pipeline/layout/balloon_layout.py','pipeline/typesetter/renderer.py','pipeline/tests/test_owner_render_geometry.py','pipeline/tests/test_owner_layout.py','pipeline/tests/test_balloon_layout_shared_regions.py','pipeline/tests/test_owner_mask.py','pipeline/tests/test_owner_compositor.py','pipeline/strip/process_bands.py','pipeline/tests/test_owner_enforcement.py') | Sort-Object
$stagedTask10 = @(git diff --cached --name-only) | Sort-Object
if ($LASTEXITCODE -ne 0 -or (Compare-Object $expectedTask10 $stagedTask10)) { throw 'Task 10 staged path set is incomplete or contaminated' }
git diff --cached -- pipeline/strip/process_bands.py pipeline/inpainter/owner_mask.py pipeline/compositor/owner_compositor.py
git commit -m "fix(owner): bind independent page space render geometry"
if ($LASTEXITCODE -ne 0) { throw 'Task 10 commit failed' }
```

### Task 11: Propagar frame geometry por runtime, QA e inspeção autenticada

**Files:**

- Modify: `pipeline/strip/run.py` em nova `_page_surface_geometry()`, `_compose_owner_output_pages()`, `_bind_owner_final_page_images()` e `_write_reassemble_manifest_debug()`; preserve `_source_page_geometry()` e seus callers em strip/logical crop.
- Modify: `pipeline/strip/types.py`
- Modify: `pipeline/ownership/model.py` para os estados lógico interno e framed publicado de `PageCompositionResult`.
- Modify: `pipeline/main.py` em `_run_final_pixel_gate_sequence()` e `build_project_json()`.
- Modify: `pipeline/qa/final_pixel_observer.py`
- Modify: `pipeline/vision_stack/runtime.py` em `run_final_pixel_ocr_probe()`.
- Modify: `pipeline/qa/final_pixel_qa.py`
- Modify: `pipeline/ownership/artifacts.py`
- Modify: `pipeline/tools/build_style_owner_target_manifest.py`
- Modify: `pipeline/tools/validate_owner_visual_matrix.py` em `_owner_map_panel()` e `_write_contact_sheets()`.
- Modify: `pipeline/tests/test_strip_owner_composition_integration.py`
- Modify: `pipeline/tests/test_owner_compositor.py`
- Modify: `pipeline/tests/test_main_emit.py`
- Modify: `pipeline/tests/test_final_pixel_observer.py`
- Modify: `pipeline/tests/test_vision_stack_runtime.py`
- Modify: `pipeline/tests/test_final_pixel_qa.py`
- Modify: `pipeline/tests/test_owner_visual_matrix_tool.py`
- Modify: `pipeline/tests/test_style_owner_target_manifest.py`
- Create: `pipeline/tests/fixtures/page_surface_geometry/narrow_project.json`
- Create: `pipeline/tests/fixtures/page_surface_geometry/target_matrix.json`
- Modify: `pipeline/tests/fixtures/style_copy_corpus/matrix.json`

**Step 1: Escrever RED de composição narrow-page**

Crie primeiro somente o diretório versionável ausente; os arquivos continuam sendo escritos com `apply_patch`:

```powershell
New-Item -ItemType Directory -Force -Path pipeline/tests/fixtures/page_surface_geometry | Out-Null
```

```python
def test_narrow_page_owner_composition_publishes_logical_to_framed_geometry():
    result = compose_narrow_page(logical_width=690, frame_width=800, x_offset=55)
    geometry = result.output_pages[1].page_surface_geometry
    assert geometry.logical_width == 690
    assert geometry.frame_width == 800
    assert geometry.content_origin_xy == (55, 0)
    assert result.compositions["page_002"].page_surface_geometry_sha256 == geometry.geometry_sha256


def test_cross_tile_owner_glyph_mask_round_trips_through_frame_without_crop():
    result = compose_owner_at_logical_bbox((109, 985, 318, 1104), x_offset=55)
    support = owner_support_bbox(result.glyph_owner_map, owner_id=result.owner_id)
    assert support == result.geometry.logical_bbox_to_frame(result.logical_glyph_support_bbox)
    assert bbox_is_inside(support, result.geometry.logical_bbox_to_frame(result.logical_safe_bbox))
    assert logical_crop_from_frame(result.framed_page, result.geometry, (109, 985, 318, 1104)).shape[:2] == (119, 209)


def test_owner_chain_uses_canonical_spaces_end_to_end_and_rejects_legacy_alias_in_enforce():
    result = compose_owner_at_logical_bbox((109, 985, 318, 1104), x_offset=55, enforce=True)
    assert result.mutation.coordinate_space == "logical_page"
    assert result.glyph_patch.coordinate_space == "logical_page"
    assert result.owner_render_geometry.logical_space == "logical_page"
    assert result.composition.coordinate_space == "framed_page"
    assert result.composition.page_surface_geometry_sha256 == result.geometry.geometry_sha256
    with pytest.raises(ValueError, match="legacy coordinate space"):
        enforce_deserialize(result.mutation.to_dict() | {"coordinate_space": "page"})


def test_internal_compositor_result_stays_logical_and_published_result_is_framed():
    internal, published = compose_narrow_owner_with_both_stages(logical_width=690, frame_width=800)
    assert internal.coordinate_space == "logical_page"
    assert internal.final_rgb.shape[:2] == (1600, 690)
    assert published.coordinate_space == "framed_page"
    assert published.final_rgb.shape[:2] == (1600, 800)
    assert published.page_surface_geometry_sha256 == published.page_surface_geometry.geometry_sha256


def test_published_framed_composition_deserialize_requires_valid_geometry_hash_and_shapes():
    payload = published_framed_composition_dict()
    assert PageCompositionResult.from_dict(payload, enforce=True).coordinate_space == "framed_page"
    with pytest.raises(ValueError, match="page surface geometry"):
        PageCompositionResult.from_dict(payload | {"page_surface_geometry": None}, enforce=True)
    with pytest.raises(ValueError, match="geometry hash"):
        PageCompositionResult.from_dict(payload | {"page_surface_geometry_sha256": "f" * 64}, enforce=True)
```

Run:

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_strip_owner_composition_integration.py pipeline/tests/test_owner_compositor.py -k "logical_to_framed or round_trips_through_frame or canonical_spaces or legacy_alias or internal_compositor or published_framed_composition" -q
if ($LASTEXITCODE -ne 1) { throw "Expected behavioral RED exit 1, got ${LASTEXITCODE}: reassembly does not publish page transform" }
```

**Step 2: Propagar o contrato na composição**

- Derive `PageSurfaceGeometry` de `source_page_widths`, `strip.width` e `page_x_offsets`.
- Preserve `_source_page_geometry()` com sua semântica atual de recorte strip->página lógica; crie `_page_surface_geometry()` para a relação logical->framed e migre apenas callers enumerados/testados.
- Anexe a mesma geometria a final/original/clean `OutputPage`.
- `compositor.owner_compositor.compose_page()` continua produzindo um `PageCompositionResult` interno em `logical_page`, com `final_rgb`, cleanup/glyph owner maps exatamente no shape lógico. Ele não recebe rótulo `framed_page` por metadata.
- `_compose_owner_output_pages()` aplica `PageSurfaceGeometry` uma única vez e cria um **novo** `PageCompositionResult` publicado em `OwnerChapterComposition.compositions`, agora `coordinate_space="framed_page"`, com `final_rgb`, cleanup/glyph owner maps exatamente `(frame_height, frame_width, ...)` e geometry/hash obrigatórios. Não mutar o resultado lógico in-place.
- Em `ownership/model.py`, deserialize non-enforce aceita a ponte opcional somente para o resultado interno `logical_page`; qualquer `framed_page` publicado exige `page_surface_geometry`, hash recalculado igual e shapes compatíveis. Owner-enforce rejeita framed sem geometry e logical com geometry incoerente.
- Inclua contrato/hash como campo de página de primeira classe em `PageCompositionResult`, `project.json`, registros de `ownership/artifacts.py` e `10_copyback_reassemble/reassemble_manifest.json`. Não o esconda apenas em `page_profile`/`ocr_result`.
- Corrija `page_profile.logical_width/logical_height` e `frame_width/frame_height`; não publique largura 800 como largura lógica 690.
- Migre `render_layout_contract.coordinate_space` para `logical_page`; o hash referenciado pelo plano de materialização deve rejeitar `page` em owner-enforce.
- `_bind_owner_final_page_images()` exige os três hashes iguais.
- Preserve a imagem framed existente; não desfaça letterbox nem mova execução owner para strip space.

**Step 3: Escrever RED do final-pixel runtime no espaço correto**

```python
def test_final_pixel_observer_crops_framed_image_with_transformed_bbox_and_returns_logical_evidence():
    report = observe_final_pixel_owner(
        framed_page=centered_frame_800(),
        owner_bbox_logical=(109, 985, 318, 1104),
        page_surface_geometry=centered_geometry(),
    )
    assert report["challenge_bbox_logical"] == [109, 985, 318, 1104]
    assert report["artifact_bbox_frame"] == [164, 985, 373, 1104]
    assert report["ocr_crop_shape"] == [119, 209]
    assert report["observation_space"] == "logical_page"


def test_final_pixel_qa_projects_logical_owner_polygon_to_framed_owner_maps_once():
    report = audit_final_pixel_owner(
        composition=framed_composition_800(),
        owner_geometry=logical_owner_geometry_690(),
        page_surface_geometry=centered_geometry(),
    )
    assert report["geometry_projection_count"] == 1
    assert report["owner_support_bbox_frame"] == expected_owner_support_bbox_frame()


def test_project_json_publishes_unambiguous_logical_and_frame_dimensions():
    page = build_narrow_page_project_json()
    assert page["logical_width"] == 690 and page["frame_width"] == 800
    assert page["page_surface_geometry"]["content_bbox_frame"] == [55, 0, 745, 1600]
    assert page["page_surface_geometry_sha256"] == page["page_surface_geometry"]["geometry_sha256"]
```

Run:

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_final_pixel_observer.py pipeline/tests/test_vision_stack_runtime.py pipeline/tests/test_final_pixel_qa.py pipeline/tests/test_main_emit.py -k "transformed_bbox or projects_logical or logical_and_frame_dimensions" -q
if ($LASTEXITCODE -ne 1) { throw "Expected behavioral RED exit 1, got ${LASTEXITCODE}: final-pixel runtime still mixes logical and framed spaces" }
```

**Step 4: Transformar o runtime/QA sem offset manual**

- `main.py` constrói o challenge com bbox lógico, geometry hash e bbox framed derivado; nunca passa bbox lógico diretamente para crop de imagem framed.
- `final_pixel_observer.py`/`vision_stack/runtime.py` recortam somente `artifact_bbox_frame`, registram shape/hash do crop e transformam observações frame/crop de volta para `logical_page` antes de comparar.
- `qa/final_pixel_qa.py` transforma polígonos/masks lógicos para owner maps framed uma única vez via `PageSurfaceGeometry`; valida shapes e bloqueia geometry ausente/divergente.
- `build_project_json()` publica campos lógicos/framed e o contrato completo por página; `ownership/artifacts.py` inclui ambos geometry hashes no registro de composição.
- Qualquer bbox fora de `content_bbox_frame`, transformação dupla, shape errado ou alias `page` em enforce produz BLOCK com razão específica.

**Step 5: Escrever RED das ferramentas**

```python
def test_contact_sheet_transforms_logical_owner_crop_into_framed_artifact_space(tmp_path):
    sheet = write_owner_sheet(
        framed_page(width=800, origin_x=55),
        target_bbox=[109, 985, 318, 1104],
        coordinate_space="logical_page",
    )
    assert sheet_metadata(sheet)["artifact_bbox_frame"] == [164, 985, 373, 1104]


def test_owner_map_panel_transforms_logical_polygons_into_frame_space(tmp_path):
    panel, metadata = write_owner_map_panel(
        tmp_path,
        polygon=((109, 985), (318, 985), (318, 1104), (109, 1104)),
        geometry=centered_page_geometry(origin_x=55),
    )
    assert panel.is_file()
    assert metadata["polygon_frame"][0] == [164, 985]
    assert metadata["polygon_frame"][2] == [373, 1104]


def test_contact_sheet_blocks_when_framed_page_geometry_is_missing(tmp_path):
    result = validate_entry_result(targeting_framed_artifact_without_geometry(), tmp_path)
    assert "missing_page_surface_geometry" in result["contracts"]
    assert result["status"] == "BLOCK"


@pytest.mark.parametrize("problem", [
    "missing_artifact", "dimension_mismatch", "duplicate_page_binding",
    "missing_geometry", "double_transform", "bbox_outside_content",
])
def test_contact_sheet_enforce_never_uses_placeholder_resize_or_page_fallback(tmp_path, problem):
    result = validate_entry_result(broken_contact_sheet_fixture(problem), tmp_path, enforce=True)
    assert result["status"] == "BLOCK"


def test_contact_sheet_six_panels_are_bound_to_real_artifact_hashes(tmp_path):
    sheet = write_authenticated_owner_sheet(real_owner_artifacts(), tmp_path, enforce=True)
    assert set(sheet.metadata["panels"]) == {
        "source", "masks_evidence", "requested_resolved",
        "observed_raster", "final", "safe_geometry",
    }
    assert all(panel["source_sha256"] for panel in sheet.metadata["panels"].values())
    assert sheet.metadata["panels"]["requested_resolved"]["plan_sha256"]
    assert sheet.metadata["panels"]["observed_raster"]["observation_sha256"]
```

Run:

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_owner_visual_matrix_tool.py -k "logical_owner_crop or transforms_logical_polygons or framed_page_geometry or placeholder_resize or six_panels" -q
if ($LASTEXITCODE -ne 1) { throw "Expected behavioral RED exit 1, got ${LASTEXITCODE}: inspection still uses logical bbox on frame" }
```

**Step 6: Corrigir tools e schema target**

- Toda source/final bbox declara `coordinate_space`.
- `logical_page` exige `page_surface_geometry_sha256` e transformação pelo contrato do projeto/run.
- `_owner_map_panel()` e `_write_contact_sheets()` transformam bbox/polygon uma vez.
- Persistir no metadata: bbox lógico, bbox no frame, geometry hash, crop dimensions e SHA-256.
- Em enforce, a ausência/divergência bloqueia; não aplicar `x_offset` default, presumir page 1, criar placeholder nem redimensionar painel para esconder shape incompatível.
- Os seis painéis são fontes reais: source crop autenticado; masks/evidence; intent+plan; raster+observation; final framed crop; safe geometry. Cada painel carrega artifact path, SHA-256, dimensões e hashes de contrato. `requested` nunca é cópia do source e `observed raster` nunca é intermediário sem `observation_sha256`.
- Atualize `build_style_owner_target_manifest.py` com modo `--matrix --entry-id --verify-only` para provar owner/page/components/source hash/page-surface binding contra um `project.json`; nunca editar bbox/hash manualmente.

**Step 7: Migrar e verificar o schema target sem alterar a evidência source**

Migre `matrix.json` para schema v3 com `source_crop.coordinate_space="logical_page"` e `expected_artifact_space="framed_page"`. Preserve integralmente page/owner/components/category, bbox e **SHA do source crop**; o hash do crop final framed é evidência de run e não entra como substituto do source hash.

Prove a CLI nova em fixture determinística:

```powershell
$r5PlanAnchor = git log -1 --format=%H -- docs/plans/2026-08-01-style-copy-r5-materialization-geometry-remediation-implementation.md
$r5Tmp = Join-Path (Resolve-Path '.') ('.codex-tmp/style-copy-r5-' + $r5PlanAnchor.Substring(0, 12))
pipeline/venv/Scripts/python.exe pipeline/tools/build_style_owner_target_manifest.py --project pipeline/tests/fixtures/page_surface_geometry/narrow_project.json --matrix pipeline/tests/fixtures/page_surface_geometry/target_matrix.json --entry-id narrow_page --verify-only --output "$r5Tmp\narrow-target-verification.json"
if ($LASTEXITCODE -ne 0) { throw 'Narrow target contract verification failed' }
```

Para `owner_p002_d9afb32ef935`, o diff deve manter source bbox `[109,985,318,1104]` e source SHA existentes; a evidência de run deve derivar frame bbox `[164,985,373,1104]` e gerar `final_artifact_crop_sha256` separado. Os comandos exatos para os três projetos reais aparecem na Task 19, depois de eles existirem.

**Step 8: Confirmar GREEN e checkpoint R5-C**

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_page_surface_geometry.py pipeline/tests/test_owner_render_geometry.py pipeline/tests/test_strip_owner_composition_integration.py pipeline/tests/test_owner_compositor.py pipeline/tests/test_main_emit.py pipeline/tests/test_final_pixel_observer.py pipeline/tests/test_vision_stack_runtime.py pipeline/tests/test_final_pixel_qa.py pipeline/tests/test_owner_visual_matrix_tool.py pipeline/tests/test_style_owner_target_manifest.py pipeline/tests/test_owner_layout.py pipeline/tests/test_owner_enforcement.py -q
if ($LASTEXITCODE -ne 0) { throw 'R5-C checkpoint failed' }
```

Também rode:

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_owner_layout.py -k "renderer_receives_one_complete_payload or connected_owner_may_use_many_layout_regions or body_is_never_split_or_truncated or normalized_chunks_reconstruct_exact_owner_payload" -q
if ($LASTEXITCODE -ne 0) { throw 'Semantic body regression' }
```

**Step 9: Stage por hunk e commit**

Crie com `apply_patch` `$r5Tmp\task11-dirty-index.patch` contendo somente os hunks R5 de `strip/run.py`, `ownership/artifacts.py`, `qa/final_pixel_qa.py`, seus dois testes dirty e o validator/test dirty. Não inclua nenhum hunk do snapshot preexistente.

```powershell
$r5PlanAnchor = git log -1 --format=%H -- docs/plans/2026-08-01-style-copy-r5-materialization-geometry-remediation-implementation.md
$r5Tmp = Join-Path (Resolve-Path '.') ('.codex-tmp/style-copy-r5-' + $r5PlanAnchor.Substring(0, 12))
$r5ScopedPatch = Join-Path $r5Tmp 'task11-dirty-index.patch'
if (-not (Test-Path -LiteralPath $r5ScopedPatch)) { throw 'Create the Task 11 scoped index patch with apply_patch first' }
git apply --cached --check --unidiff-zero $r5ScopedPatch
if ($LASTEXITCODE -ne 0) { throw 'Task 11 scoped index patch does not apply cleanly' }
git apply --cached --unidiff-zero $r5ScopedPatch
if ($LASTEXITCODE -ne 0) { throw 'Task 11 scoped index patch failed' }
git add -- pipeline/strip/types.py pipeline/ownership/model.py pipeline/main.py pipeline/qa/final_pixel_observer.py pipeline/vision_stack/runtime.py pipeline/tools/build_style_owner_target_manifest.py pipeline/tests/test_strip_owner_composition_integration.py pipeline/tests/test_owner_compositor.py pipeline/tests/test_main_emit.py pipeline/tests/test_final_pixel_observer.py pipeline/tests/test_vision_stack_runtime.py pipeline/tests/test_style_owner_target_manifest.py pipeline/tests/fixtures/page_surface_geometry/narrow_project.json pipeline/tests/fixtures/page_surface_geometry/target_matrix.json pipeline/tests/fixtures/style_copy_corpus/matrix.json
if ($LASTEXITCODE -ne 0) { throw 'Task 11 clean-path staging failed' }
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Task 11 staged diff check failed' }
$expectedTask11 = @('pipeline/strip/run.py','pipeline/tools/validate_owner_visual_matrix.py','pipeline/tests/test_owner_visual_matrix_tool.py','pipeline/qa/final_pixel_qa.py','pipeline/tests/test_final_pixel_qa.py','pipeline/ownership/artifacts.py','pipeline/strip/types.py','pipeline/ownership/model.py','pipeline/main.py','pipeline/qa/final_pixel_observer.py','pipeline/vision_stack/runtime.py','pipeline/tools/build_style_owner_target_manifest.py','pipeline/tests/test_strip_owner_composition_integration.py','pipeline/tests/test_owner_compositor.py','pipeline/tests/test_main_emit.py','pipeline/tests/test_final_pixel_observer.py','pipeline/tests/test_vision_stack_runtime.py','pipeline/tests/test_style_owner_target_manifest.py','pipeline/tests/fixtures/page_surface_geometry/narrow_project.json','pipeline/tests/fixtures/page_surface_geometry/target_matrix.json','pipeline/tests/fixtures/style_copy_corpus/matrix.json') | Sort-Object
$stagedTask11 = @(git diff --cached --name-only) | Sort-Object
if ($LASTEXITCODE -ne 0 -or (Compare-Object $expectedTask11 $stagedTask11)) { throw 'Task 11 staged path set is incomplete or contaminated' }
git diff --cached -- pipeline/strip/run.py pipeline/main.py pipeline/qa/final_pixel_observer.py pipeline/vision_stack/runtime.py pipeline/qa/final_pixel_qa.py pipeline/ownership/artifacts.py pipeline/tools/validate_owner_visual_matrix.py pipeline/tests/test_owner_visual_matrix_tool.py
git commit -m "fix(qa): transform logical owner evidence into framed pages"
if ($LASTEXITCODE -ne 0) { throw 'Task 11 commit failed' }
```

Expected: mudanças locais anteriores continuam preservadas e apenas hunks R5 são commitados; o cached diff contém todos os produtores/consumidores enumerados acima.

### Task 12: Vincular o corpo traduzido completo aos pixels commitados

**Files:**

- Create: `pipeline/ownership/delivery.py`
- Create: `pipeline/tests/test_owner_delivery_contract.py`
- Modify: `pipeline/ownership/model.py` em `OwnerGlyphPatch`.
- Modify: `pipeline/compositor/owner_compositor.py` para validar core/paint/delivery sem igualdade legada `actual_changed == glyph_mask`.
- Modify: `pipeline/typesetter/renderer.py` em `_render_owner_band_image()`.
- Modify: `pipeline/strip/process_bands.py` em `apply_atomic_owner_execution()` e record final.
- Modify: `pipeline/qa/final_pixel_qa.py`
- Modify: `pipeline/tests/test_owner_atomic_execution.py`
- Modify: `pipeline/tests/test_final_pixel_qa.py`
- Modify: `pipeline/tests/test_owner_compositor.py`
- Modify: `pipeline/tests/test_owner_enforcement.py`
- Modify: `pipeline/tests/test_strip_owner_composition_integration.py`
- Modify: `pipeline/tests/test_strip_process_bands.py`
- Test: `pipeline/tests/test_owner_translation.py`
- Test: `pipeline/tests/test_owner_layout.py`

**Step 1: Escrever os testes RED do delivery contract**

Proteja o import de `ownership.delivery` com `find_spec()` ou skeleton importável e confirme exit 1 comportamental antes de implementar.

```python
def test_delivery_contract_binds_one_complete_owner_payload_to_render_input():
    authority = seal_owner_text_execution_authority(
        owner_id="owner_a", page_id="page_001",
        source_payload="WHAT ARE THEY SAYING?",
        translated_payload="O QUE ELES ESTÃO DIZENDO?",
        normalized_chunks=["O QUE ELES ESTÃO DIZENDO?"],
    )
    contract = build_owner_text_delivery_contract(
        execution_authority=authority,
        layout_payload="O QUE ELES ESTÃO DIZENDO?",
        rendered_lines=["O QUE ELES", "ESTÃO DIZENDO?"],
        rendered_glyph_runs=glyph_runs_for("O QUE ELES\nESTÃO DIZENDO?"),
        glyph_core_mask=combined_glyph_core_mask(),
        rendered_patch_sha256="a" * 64,
    )
    assert contract.status == "delivered"
    assert contract.translated_payload_sha256 == contract.layout_payload_sha256
    assert contract.translated_payload_sha256 == contract.rendered_payload_sha256


@pytest.mark.parametrize("rendered", [
    ["O QUE ELES"],
    ["O QUE ELES", "ESTÃO"],
    ["O QUE ELES", "ESTÃO DIZENDO?", "ESTÃO DIZENDO?"],
])
def test_partial_or_duplicated_render_body_is_not_delivered(rendered):
    contract = build_contract(translated="O QUE ELES ESTÃO DIZENDO?", rendered_lines=rendered)
    assert contract.status == "review_required"
    assert contract.reason in {"render_payload_incomplete", "render_payload_duplicated"}


def test_atomic_commit_rejects_glyph_patch_with_divergent_delivery_hash():
    patch = valid_patch_with_delivery("CORPO COMPLETO")
    object.__setattr__(patch.delivery_contract, "rendered_payload_sha256", "f" * 64)
    commit = apply_atomic_owner_execution(original(), valid_mutation(), patch)
    assert commit.committed is False
    assert commit.reason == "render_contract_invalid:text_delivery_contract_hash_mismatch"


def test_atomic_commit_rejects_internally_rehashed_truncated_delivery_against_sealed_authority():
    authority = seal_authority(translated_payload="CORPO COMPLETO NÃO PODE SER CORTADO")
    mutation = valid_mutation(text_execution_authority=authority)
    patch = valid_patch_with_delivery(authority)
    truncated = rebuild_delivery_and_patch_with_valid_self_hashes(
        patch, rendered_payload="CORPO COMPLETO", glyph_spans=spans_for("CORPO COMPLETO")
    )
    commit = apply_atomic_owner_execution(original(), mutation, truncated)
    assert commit.committed is False
    assert commit.reason == "render_contract_invalid:translated_execution_authority_mismatch"


def test_atomic_commit_recomputes_each_span_mask_before_union():
    authority = seal_authority(translated_payload="CORPO COMPLETO")
    patch = valid_patch_with_delivery(authority)
    tampered = replace_span_mask_and_rehash_outer_patch(patch, span_index=0, replacement=np.zeros_like(patch.glyph_span_core_masks[0]))
    commit = apply_atomic_owner_execution(original(), valid_mutation(text_execution_authority=authority), tampered)
    assert commit.committed is False
    assert commit.reason == "render_contract_invalid:glyph_span_core_mask_hash_mismatch"


def test_rollback_with_english_pixels_can_never_pass_functional_gate():
    report = audit_review_owner_with_source_pixels_and_no_glyph_commit()
    assert report.contracts["route_state_contract"] == "BLOCK"
    assert report.contracts["final_language_contract"] == "BLOCK"


@pytest.mark.parametrize("problem", [
    "correct_text_empty_run", "duplicated_span_pixels", "unbound_extra_mask", "missing_span",
])
def test_delivery_requires_exact_once_glyph_evidence_not_only_matching_strings(problem):
    contract = build_contract_with_broken_glyph_runs(problem)
    assert contract.status == "review_required"
    assert contract.reason in {
        "glyph_run_empty", "glyph_span_duplicated", "glyph_mask_unbound", "glyph_span_gap",
    }


def test_delivery_uses_core_mask_while_paint_mask_includes_glow_and_stroke():
    patch = render_patch_with_fill_stroke_glow()
    assert np.count_nonzero(patch.paint_mask) > np.count_nonzero(patch.glyph_core_mask)
    assert mask_subset(patch.glyph_core_mask, patch.paint_mask)
    assert np.array_equal(union_span_core_masks(patch.glyph_span_core_masks), patch.glyph_core_mask)
    assert mask_subset(changed_pixel_mask(patch), patch.paint_mask)


def test_compositor_accepts_changed_subset_of_paint_but_never_outside_paint():
    patch = render_patch_with_sparse_antialias_and_glow()
    assert compose_owner_enforce(patch).status == "committed"
    assert mask_subset(changed_pixel_mask(patch), patch.paint_mask)
    with pytest.raises(ValueError, match="changed pixels outside paint mask"):
        compose_owner_enforce(patch_with_changed_pixel_outside_paint(patch))
```

Adicione um caso Unicode/PT-BR com acentos, um corpo connected/cross-tile e um nome próprio. A canonicalização pode normalizar Unicode NFC e espaços entre linhas; não pode remover palavra, número ou pontuação material.

**Step 2: Confirmar RED**

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_owner_delivery_contract.py pipeline/tests/test_owner_atomic_execution.py pipeline/tests/test_owner_compositor.py pipeline/tests/test_final_pixel_qa.py -k "delivery or partial_or_duplicated or english_pixels or sealed_authority or span_mask or changed_subset" -q
if ($LASTEXITCODE -ne 1) { throw "Expected behavioral RED exit 1, got ${LASTEXITCODE}: translated delivery is not hash-bound" }
```

**Step 3: Implementar contrato e binding**

```python
@dataclass(frozen=True)
class OwnerTextExecutionAuthority:
    schema_version: int
    owner_id: str
    page_id: str
    source_payload_sha256: str
    translated_payload_sha256: str
    normalized_chunks_sha256: str
    authority_sha256: str

@dataclass(frozen=True)
class RenderedGlyphSpanEvidence:
    span_index: int
    text_start: int
    text_end: int
    normalized_text_sha256: str
    non_whitespace_glyph_count: int
    glyph_coverage_ratio: float
    font_run_sha256: str
    glyph_run_sha256: str
    core_mask_sha256: str
    core_mask_bbox_logical: BBox

@dataclass(frozen=True)
class OwnerTextDeliveryContract:
    schema_version: int
    owner_id: str
    page_id: str
    execution_authority_sha256: str
    source_payload_sha256: str
    translated_payload_sha256: str
    layout_payload_sha256: str
    rendered_payload_sha256: str
    rendered_patch_sha256: str
    rendered_line_count: int
    glyph_spans: tuple[RenderedGlyphSpanEvidence, ...]
    glyph_spans_sha256: str
    glyph_core_mask_sha256: str
    status: Literal["delivered", "review_required"]
    reason: str
    contract_sha256: str
```

- O renderer reconstrói `rendered_lines` na ordem final e prova igualdade semântica com o payload completo.
- Antes de chamar layout/renderer, sele `OwnerTextExecutionAuthority` a partir do payload source, da única resposta PT-BR e dos chunks normalizados; anexe seu hash ao execution record e a `OwnerMutation`. O renderer recebe essa autoridade, mas não pode criá-la nem reescrevê-la.
- Cada span liga índices **na autoridade traduzida**, text hash, non-whitespace glyph count/coverage, FontRun/GlyphRun hashes e core-mask hash/bbox. Spans cobrem o payload autorizado em ordem, exatamente uma vez, sem lacuna/overlap/duplicação.
- Adicione a `OwnerGlyphPatch` máscaras distintas `glyph_core_mask` e `paint_mask` (fill+stroke+glow/shadow/effects), `glyph_span_core_masks` e `glyph_span_runs` ordenados, com uma máscara e um `GlyphRunObservation` verificáveis por span. O commit recalcula mask/run hashes, bbox, codepoints/clusters observados e a união exata das core masks; a sequência reconstruída dos runs deve ser exatamente a fatia autorizada. A união deve igualar `glyph_core_mask`; run vazio, core extra ou glyph não vinculado bloqueia mesmo se o delivery/patch tiver sido internamente rehashado.
- Migre todos os constructors diretos. `glyph_mask` pode existir temporariamente apenas como property read-only de `paint_mask` para leitores non-enforce; owner-enforce exige `glyph_core_mask` real e não aceita alias como prova de delivery.
- Exija `glyph_core_mask ⊆ paint_mask`, todo pixel alterado pelo patch dentro de `paint_mask`, e os envelopes/safe checks sobre a máscara apropriada. Não reutilize a antiga change mask como se fosse core glyph.
- `OwnerGlyphPatch` carrega/autentica conjuntamente execution-authority hash, delivery contract, style raster contract, máscaras por span e os dois mask hashes; não crie dependência funcional dentro do contrato de estilo.
- Migre `owner_compositor.py` para exigir `glyph_core_mask ⊆ paint_mask` e `actual_changed ⊆ paint_mask`, não igualdade. Use paint para autorização/conflito de pixels e core para coverage/delivery; recompute os hashes e a união por span no commit.
- O commit atômico recomputa hashes, compara o payload/span coverage contra a autoridade independente e rejeita corpo parcial/duplicado mesmo quando todos os hashes controlados pelo patch são coerentes entre si.
- O record final serializa `owner_text_execution_authority` antes do render e `owner_text_delivery_contract` somente quando commitado; ambos mantêm hashes independentes.
- Final-pixel QA exige delivery válido para owner `rendered`, glyph owner map não vazio e ausência de source residual. Um owner em review com inglês preservado bloqueia explicitamente `route_state_contract` e `final_language_contract`.
- Tradução continua uma requisição/resposta por owner; não altere `ownership/translation.py` a menos que um RED demonstre quebra. Se precisar, stage somente o hunk R5.
- `pipeline/tests/test_owner_translation.py` já está dirty e permanece read-only/unstaged nesta task; somente um RED R5 novo que exija mudar o teste autoriza patch indexado separado e atualização explícita do expected staged set.

**Step 4: Confirmar GREEN e invariância semântica**

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_owner_delivery_contract.py pipeline/tests/test_owner_translation.py pipeline/tests/test_owner_layout.py pipeline/tests/test_owner_atomic_execution.py pipeline/tests/test_owner_compositor.py pipeline/tests/test_owner_enforcement.py pipeline/tests/test_strip_owner_composition_integration.py pipeline/tests/test_strip_process_bands.py pipeline/tests/test_final_pixel_qa.py -q
if ($LASTEXITCODE -ne 0) { throw 'Task 12 GREEN failed' }
```

Rode também:

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_owner_translation.py -k "one_complete_payload or body_lines_never or many_tiles_is_translated_once" -q
if ($LASTEXITCODE -ne 0) { throw 'Owner translation invariance failed' }
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_owner_layout.py -k "complete_payload or never_split_or_truncated or reconstruct_exact" -q
if ($LASTEXITCODE -ne 0) { throw 'Owner layout invariance failed' }
```

**Step 5: Stage por hunk e commit**

Crie com `apply_patch` `$r5Tmp\task12-dirty-index.patch` contendo somente os hunks R5 de `process_bands.py`, `final_pixel_qa.py`, `test_final_pixel_qa.py` e `test_owner_enforcement.py`.

```powershell
$r5PlanAnchor = git log -1 --format=%H -- docs/plans/2026-08-01-style-copy-r5-materialization-geometry-remediation-implementation.md
$r5Tmp = Join-Path (Resolve-Path '.') ('.codex-tmp/style-copy-r5-' + $r5PlanAnchor.Substring(0, 12))
$r5ScopedPatch = Join-Path $r5Tmp 'task12-dirty-index.patch'
if (-not (Test-Path -LiteralPath $r5ScopedPatch)) { throw 'Create the Task 12 scoped index patch with apply_patch first' }
git apply --cached --check --unidiff-zero $r5ScopedPatch
if ($LASTEXITCODE -ne 0) { throw 'Task 12 scoped index patch does not apply cleanly' }
git apply --cached --unidiff-zero $r5ScopedPatch
if ($LASTEXITCODE -ne 0) { throw 'Task 12 scoped index patch failed' }
git add -- pipeline/ownership/delivery.py pipeline/ownership/model.py pipeline/compositor/owner_compositor.py pipeline/typesetter/renderer.py pipeline/tests/test_owner_delivery_contract.py pipeline/tests/test_owner_atomic_execution.py pipeline/tests/test_owner_compositor.py pipeline/tests/test_strip_owner_composition_integration.py pipeline/tests/test_strip_process_bands.py
if ($LASTEXITCODE -ne 0) { throw 'Task 12 clean-path staging failed' }
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Task 12 staged diff check failed' }
$expectedTask12 = @('pipeline/ownership/delivery.py','pipeline/ownership/model.py','pipeline/compositor/owner_compositor.py','pipeline/typesetter/renderer.py','pipeline/tests/test_owner_delivery_contract.py','pipeline/tests/test_owner_atomic_execution.py','pipeline/tests/test_owner_compositor.py','pipeline/tests/test_strip_owner_composition_integration.py','pipeline/tests/test_strip_process_bands.py','pipeline/strip/process_bands.py','pipeline/qa/final_pixel_qa.py','pipeline/tests/test_final_pixel_qa.py','pipeline/tests/test_owner_enforcement.py') | Sort-Object
$stagedTask12 = @(git diff --cached --name-only) | Sort-Object
if ($LASTEXITCODE -ne 0 -or (Compare-Object $expectedTask12 $stagedTask12)) { throw 'Task 12 staged path set is incomplete or contaminated' }
git commit -m "fix(owner): bind complete translated body to committed glyphs"
if ($LASTEXITCODE -ne 0) { throw 'Task 12 commit failed' }
```

### Task 13: Tornar o benchmark sintético mask-backed e realmente V2

**Files:**

- Modify: `pipeline/tests/fixtures/style_benchmark_v2/benchmark_spec.json`
- Modify: `pipeline/debug_tools/generate_style_benchmark_v2.py`
- Modify: `pipeline/debug_tools/run_style_benchmark_v2.py`
- Modify: `pipeline/debug_tools/style_benchmark_report.py`
- Modify: `pipeline/tests/test_style_benchmark_v2_generator.py`
- Modify: `pipeline/tests/test_style_benchmark_v2_score.py`
- Modify: `pipeline/tests/test_font_matcher.py`

**Step 1: Escrever RED do spec e artefatos ground-truth**

```python
def test_benchmark_cases_have_owner_category_role_work_and_split():
    spec = load_benchmark_spec(FIXTURE_SPEC)
    for case in spec["cases"]:
        assert case["owner_id"]
        assert case["category"]
        assert case["semantic_role"]
        assert case["work_id"] and case["chapter_id"]
        assert case["split"] in {"calibration", "holdout"}


def test_generator_emits_core_effect_safe_masks_and_expected_contract(tmp_path):
    run = generate_smoke(tmp_path)
    manifest = load_manifest(run)
    for case in manifest["cases"]:
        for key in ("glyph_core_mask_a", "glyph_core_mask_b", "effect_mask_a", "effect_mask_b", "safe_mask", "expected_materialization"):
            assert (run / case[key]).is_file()


def test_spec_rejects_owner_reused_across_calibration_and_holdout(tmp_path):
    spec = benchmark_spec_with_duplicate_owner_across_splits()
    with pytest.raises(ValueError, match="owner.*split"):
        load_benchmark_spec(write_spec(tmp_path, spec))


def test_spec_rejects_required_category_without_holdout_owner(tmp_path):
    spec = benchmark_spec_with_required_category_only_in_calibration()
    with pytest.raises(ValueError, match="holdout.*category"):
        load_benchmark_spec(write_spec(tmp_path, spec))
```

**Step 2: Confirmar RED**

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_style_benchmark_v2_generator.py -k "owner_category_role or emits_core or reused_across or required_category" -q
if ($LASTEXITCODE -ne 1) { throw "Expected behavioral RED exit 1, got ${LASTEXITCODE}: benchmark lacks ground truth contracts" }
```

**Step 3: Implementar geração determinística**

- Versione o spec para schema 3.
- Adicione identidade/split/category/semantic role a cada caso.
- `_render_style_image()` passa a retornar imagem e masks exatas da própria composição; não redetecte ground truth.
- Grave PNG lossless para core/effect/safe e JSON esperado com fonte/hash, cores, efeitos, transforms e geometry.
- Valide que todo path fica dentro do run isolado e que child crash nunca publica run parcial.
- Preserve dois textos distintos por estilo para round-trip.

**Step 4: Escrever RED do runner V2 e top-k real**

```python
def test_authoritative_score_uses_mask_backed_v2_and_real_font_top_k(tmp_path):
    summary = run_all(tmp_path)
    assert summary["authoritative_score"] == "v2_mask_backed"
    assert summary["score"]["font_top1"]["evaluated"] > 0
    assert summary["score"]["font_top3"]["evaluated"] == summary["score"]["font_top1"]["evaluated"]
    assert summary["score"]["font_top3"]["hits"] >= summary["score"]["font_top1"]["hits"]


def test_synthetic_score_emits_fill_delta_e_with_nonzero_denominator(tmp_path):
    metrics = run_all(tmp_path)["score"]["fill_delta_e_2000"]
    assert metrics["count"] > 0
    assert metrics["median"] is not None
    assert metrics["p95"] is not None


def test_legacy_detector_score_is_diagnostic_not_authoritative(tmp_path):
    summary = run_all(tmp_path)
    assert "legacy_diagnostic_score" in summary
    assert summary["validation"]["inputs"] == ["v2_mask_backed"]
```

**Step 5: Confirmar RED**

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_style_benchmark_v2_score.py -k "mask_backed or fill_delta_e or legacy_detector" -q
if ($LASTEXITCODE -ne 1) { throw "Expected behavioral RED exit 1, got ${LASTEXITCODE}: benchmark still scores legacy source extraction" }
```

**Step 6: Implementar medição V2**

- Carregue core/effect/safe masks do manifest.
- Use `FontShapeMatcher` com frozen source text e core mask; publique top-k real, não o tuple vazio do v1.
- Use extractor V2 mask-backed para fill/stroke/effect.
- Calcule CIEDE2000, largura/offset/radius, rotation, width/slant/tracking/occupancy e x-height com denominadores explícitos.
- `score_benchmark()` retorna seções sintéticas somente: font top-1/top-3, atributos, color/geometry errors, hard-negative, round-trip e false effects.
- Remova de `evaluate_validation_thresholds()` a exigência de owner/speech/category/inspection; essas métricas pertencem ao agregador final da Task 16.
- Mantenha `--mode enforce` retornando 2 quando qualquer threshold sintético falhar ou faltar.

**Step 7: Confirmar GREEN**

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_style_benchmark_v2_generator.py pipeline/tests/test_style_benchmark_v2_score.py pipeline/tests/test_font_matcher.py -q
if ($LASTEXITCODE -ne 0) { throw 'Task 13 GREEN failed' }
```

**Step 8: Rodar sentinel externo `all`, sem exigir ainda o gate owner final**

```powershell
$benchAttempt = (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + (git rev-parse --short=12 HEAD)
$benchRoot = 'N:\TraduzAI_style_runs\style_copy_r5_benchmark_all_' + $benchAttempt
$benchRunId = 'r5-all-' + $benchAttempt
if (Test-Path -LiteralPath $benchRoot) { throw "Benchmark root already exists: $benchRoot" }
pipeline/venv/Scripts/python.exe pipeline/debug_tools/run_style_benchmark_v2.py --spec pipeline/tests/fixtures/style_benchmark_v2/benchmark_spec.json --level all --output-root $benchRoot --run-id $benchRunId --seed 1729 --mode enforce
$benchExit = $LASTEXITCODE
$benchSummaryPath = Join-Path (Join-Path $benchRoot $benchRunId) 'style_benchmark_summary.json'
if ($benchExit -ne 0) { throw "R5 benchmark all failed with exit $benchExit; create a RED and fix before commit" }
if (-not (Test-Path -LiteralPath $benchSummaryPath)) { throw 'R5 benchmark summary was not produced' }
$benchSummary = Get-Content -LiteralPath $benchSummaryPath -Raw | ConvertFrom-Json
if ($benchSummary.schema_version -lt 3 -or $benchSummary.run_id -ne $benchRunId -or $benchSummary.level -ne 'all') { throw 'R5 benchmark summary identity/schema mismatch' }
if ($benchSummary.authoritative_score -ne 'v2_mask_backed' -or -not $benchSummary.provenance -or $benchSummary.validation.status -ne 'PASS') { throw 'R5 benchmark summary lacks authoritative provenance or PASS' }
Write-Output "benchmark_all_exit=$benchExit summary=$benchSummaryPath"
```

Expected: o level `all` cobre hard-negative, round-trip, gradient/effects e todos os denominadores aplicáveis; nenhum denominador obrigatório é zero. Se threshold ainda falhar, a task não falsifica PASS; diagnostique matcher/measurement com RED adicional antes de seguir. Levels menores podem omitir métricas declaradas `not_applicable` pelo spec, mas jamais publicar zero como PASS para métrica requerida naquele level.

**Step 9: Commit**

```powershell
git add -- pipeline/tests/fixtures/style_benchmark_v2/benchmark_spec.json pipeline/debug_tools/generate_style_benchmark_v2.py pipeline/debug_tools/run_style_benchmark_v2.py pipeline/debug_tools/style_benchmark_report.py pipeline/tests/test_style_benchmark_v2_generator.py pipeline/tests/test_style_benchmark_v2_score.py pipeline/tests/test_font_matcher.py
if ($LASTEXITCODE -ne 0) { throw 'Task 13 staging failed' }
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Task 13 staged diff check failed' }
git diff --cached --name-status
if ($LASTEXITCODE -ne 0) { throw 'Task 13 staged path inspection failed' }
git commit -m "test(style): make benchmark v2 mask backed"
if ($LASTEXITCODE -ne 0) { throw 'Task 13 commit failed' }
```

### Task 14: Agregar containment e mismatch catastrófico no QA owner-scoped

**Files:**

- Modify: `pipeline/qa/style_fidelity.py`
- Modify: `pipeline/qa/style_fidelity_policy.py`
- Modify: `pipeline/tests/test_style_fidelity_qa.py`
- Modify: `pipeline/tests/test_style_contract.py`
- Modify: `pipeline/debug_tools/style_runtime_probe.py`
- Modify: `pipeline/tests/test_style_runtime_probe.py`

**Step 1: Escrever testes RED de métricas owner**

```python
def test_owner_qa_emits_safe_containment_with_explicit_denominator(tmp_path):
    report = audit_style_fidelity(project_with_two_valid_owners(), tmp_path, mode="enforce")
    assert report["metrics"]["safe_containment"] == {
        "evaluated": 2, "contained": 2, "rate": 1.0
    }


def test_missing_outside_safe_measurement_blocks_instead_of_assuming_zero(tmp_path):
    project = project_with_two_valid_owners()
    del first_contract(project)["render_metrics"]["core_pixels_outside_safe"]
    rehash(first_contract(project))
    report = audit_style_fidelity(project, tmp_path, mode="enforce")
    assert report["gate"]["status"] == "BLOCK"
    assert any(row["code"] == "required_metric_missing" for row in report["findings"])


def test_catastrophic_count_requires_high_confidence_and_true_mismatch(tmp_path):
    report = audit_style_fidelity(project_with_one_high_confidence_font_mismatch(), tmp_path, mode="enforce")
    assert report["metrics"]["catastrophic_mismatches"] == {
        "evaluated": 1, "count": 1
    }


def test_style_eligible_owner_remains_in_denominator_after_execution_rollback(tmp_path):
    project = project_with_eligible_owner_rolled_back_to_review()
    report = audit_style_fidelity(project, tmp_path, mode="enforce")
    assert report["summary"]["eligible_owner_count"] == 1
    assert report["summary"]["rendered_owner_count"] == 0
    assert report["gate"]["status"] == "BLOCK"
    assert report["owners"][0]["findings"][0]["code"] == "owner_materialization_not_committed"


def test_runtime_probe_uses_exact_matrix_target_keys_not_all_renderable_owners(tmp_path):
    report = probe_projects(
        projects_with_extra_renderable_owners(tmp_path),
        matrix=exact_nine_target_matrix(tmp_path),
    )
    assert report["target_count"] == 9
    assert set(report["target_keys"]) == exact_nine_entry_page_owner_keys()


@pytest.mark.parametrize("problem", ["target_missing", "target_duplicate", "entry_project_missing"])
def test_runtime_probe_blocks_incomplete_or_ambiguous_matrix_resolution(tmp_path, problem):
    with pytest.raises(RuntimeProbeError):
        probe_projects(*runtime_probe_fixture_with_problem(tmp_path, problem))
```

Inclua effect containment, fill/stroke/effect Delta E count/median/p95, geometry errors, zero denominator e canonical equivalence.

**Step 2: Confirmar RED**

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_style_fidelity_qa.py pipeline/tests/test_style_runtime_probe.py -k "safe_containment or outside_safe_measurement or catastrophic_count or execution_rollback or exact_matrix_target_keys or ambiguous_matrix_resolution" -q
if ($LASTEXITCODE -ne 1) { throw "Expected behavioral RED exit 1, got ${LASTEXITCODE}: owner QA does not aggregate required metrics" }
```

**Step 3: Implementar métricas com provenance**

- Derive o denominador dos `OwnerStyleCapture.eligible` congelados antes da execução, preservados também no record de rollback; não use apenas owners cujo estado final já é `rendered`.
- Consuma materialization observation/comparison v2; remova `_same_value()` cru.
- Exija `core_pixels_outside_safe` e `effect_pixels_outside_safe` medidos.
- Agregue `{evaluated, numerator, rate}`; zero denominador é finding.
- Conte mismatch catastrófico somente quando atributo está no policy set, decisão foi aplicada, confidence >= threshold e comparação canônica falhou.
- Agregue Delta E/geometry com count/median/p95 e fonte de evidência (`decision_to_observation` ou source/final mask-backed).
- Agregue `intent_to_target_deviation`: contagem por `resolution_kind`, font-size ratio/minimum legibility, relações superseded autorizadas e zero policy adjustment fora da whitelist.
- Normalize a chave externa para `category_metrics` ou migre todos os consumidores para `categories`; escolha uma só, versão o schema e teste o contrato. Não mantenha aliases divergentes.
- Atualize runtime probe para exigir `--matrix <matrix.json>`, resolver exatamente `(entry_id,page_id,owner_id)` nos projects e contar materialization plan/observation/delivery/geometry somente desse conjunto. Owner extra não entra; target ausente/duplicado ou entry sem project bloqueia. Publique target keys e matrix SHA-256. A ligação ao acceptance bundle fica na Task 16, depois que o schema do bundle existir.

**Step 4: Confirmar GREEN**

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_style_fidelity_qa.py pipeline/tests/test_style_contract.py pipeline/tests/test_style_runtime_probe.py -q
if ($LASTEXITCODE -ne 0) { throw 'Task 14 GREEN failed' }
```

**Step 5: Commit**

```powershell
git add -- pipeline/qa/style_fidelity.py pipeline/qa/style_fidelity_policy.py pipeline/tests/test_style_fidelity_qa.py pipeline/tests/test_style_contract.py pipeline/debug_tools/style_runtime_probe.py pipeline/tests/test_style_runtime_probe.py
if ($LASTEXITCODE -ne 0) { throw 'Task 14 staging failed' }
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Task 14 staged diff check failed' }
git diff --cached --name-status
if ($LASTEXITCODE -ne 0) { throw 'Task 14 staged path inspection failed' }
git commit -m "fix(qa): aggregate owner materialization fidelity metrics"
if ($LASTEXITCODE -ne 0) { throw 'Task 14 commit failed' }
```

### Task 15: Produzir GO rates somente da inspeção holdout autenticada

**Files:**

- Modify: `pipeline/tools/validate_owner_visual_matrix.py`
- Modify: `pipeline/tests/test_owner_visual_matrix_tool.py`
- Modify: `pipeline/tests/fixtures/style_copy_corpus/matrix.json`

**Step 1: Escrever testes RED do summary**

```python
def test_inspection_summary_uses_exact_holdout_targets_not_filenames(tmp_path):
    summary = build_style_holdout_summary(
        matrix=matrix_with_calibration_and_holdout(),
        inspection=complete_inspection_rows(),
    )
    assert summary["owners"]["evaluated"] == holdout_target_count()
    assert summary["calibration_owner_count"] > 0
    assert summary["calibration_owner_count"] not in summary["owners"].values()


def test_inspection_summary_emits_owner_speech_and_category_go_rates(tmp_path):
    summary = build_style_holdout_summary(matrix=matrix(), inspection=complete_go_rows())
    assert summary["owners"]["go_rate"] == 1.0
    assert summary["speech"]["evaluated"] > 0
    assert summary["speech"]["go_rate"] == 1.0
    assert set(summary["required_holdout_categories"]) == {
        "white_balloon", "burst", "cross_tile_owner", "dark_panel", "text_over_art",
    }
    assert all(summary["categories"][name]["evaluated"] > 0 for name in summary["required_holdout_categories"])


def test_matrix_aggregates_bound_owner_qa_reports(tmp_path):
    summary = build_owner_qa_summary(
        matrix=matrix(),
        entry_reports=passing_entry_style_reports(),
    )
    assert summary["eligible_owner_count"] == sum_expected_eligible_owners()
    assert summary["materialization_observation_coverage"] == 1.0
    assert summary["safe_containment"]["evaluated"] > 0
    assert summary["source_reports_sha256"]


@pytest.mark.parametrize("problem", ["pending", "missing", "duplicate", "hash_mismatch", "zero_category"])
def test_inspection_summary_blocks_incomplete_or_unauthenticated_denominator(tmp_path, problem):
    summary = build_style_holdout_summary(
        matrix=matrix(),
        inspection=inspection_with_problem(problem),
    )
    assert summary["status"] == "BLOCK"
    assert summary["findings"]
```

O manifest passa a declarar `semantic_role`/`is_speech` por target, não inferir speech pelo nome `white_balloon`.

**Step 2: Confirmar RED**

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_owner_visual_matrix_tool.py -k "inspection_summary or aggregates_bound_owner_qa" -q
if ($LASTEXITCODE -ne 1) { throw "Expected behavioral RED exit 1, got ${LASTEXITCODE}: inspection GO rates are not produced" }
```

**Step 3: Implementar summary autenticado**

- Vincule linha por `(entry_id, page_id, owner_id, category, split, source_sha256, final_sha256, page_surface_geometry_sha256)`.
- Use somente `split=holdout` nos rates de aceitação; calibration permanece diagnóstico.
- Escreva `style_holdout_summary.json` dentro do output root da matriz.
- Escreva sempre `matrix_execution_summary.json` com producer run/bundle IDs, status Functional/Style/Inspection, findings tipados e a prova de que um exit 2 inicial decorre somente de inspeção `PENDING`; falha runtime/target/gate nunca se mistura com pending.
- Leia exclusivamente `project.json -> qa.style_fidelity` de cada entry autenticada, exija o schema/hash emitido pela Task 14 e agregue em `style_owner_qa_summary.json`, preservando entry/project/owner/page/profile/contract hashes e rejeitando owner duplicado ou report/run divergente. `source_reports_sha256` é o hash canônico dos três reports embutidos, não de filenames.
- Produza `owners`, `speech`, `categories`, `inspection_coverage`, hashes de manifest/inspection e status `PASS|BLOCK`.
- Declare explicitamente `required_holdout_categories = [white_balloon, burst, cross_tile_owner, dark_panel, text_over_art]`. Categorias somente calibration (`colored_card`, `table_ranking`, `translucent_balloon`, `primary_ocr_omission`) ficam diagnósticas e não criam denominador holdout impossível.
- Dedupe de pixels/contact sheet não remove target do denominador.
- `PENDING`, row ausente, target duplicado, hash divergente, categoria required vazia ou Functional/Style diferente de `GO` impedem PASS.

**Step 4: Confirmar GREEN**

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_owner_visual_matrix_tool.py -q
if ($LASTEXITCODE -ne 0) { throw 'Task 15 GREEN failed' }
```

**Step 5: Stage por hunk e commit**

Crie com `apply_patch` `$r5Tmp\task15-dirty-index.patch` contendo somente os hunks R5 do validator e de seu teste já dirty.

```powershell
$r5PlanAnchor = git log -1 --format=%H -- docs/plans/2026-08-01-style-copy-r5-materialization-geometry-remediation-implementation.md
$r5Tmp = Join-Path (Resolve-Path '.') ('.codex-tmp/style-copy-r5-' + $r5PlanAnchor.Substring(0, 12))
$r5ScopedPatch = Join-Path $r5Tmp 'task15-dirty-index.patch'
if (-not (Test-Path -LiteralPath $r5ScopedPatch)) { throw 'Create the Task 15 scoped index patch with apply_patch first' }
git apply --cached --check --unidiff-zero $r5ScopedPatch
if ($LASTEXITCODE -ne 0) { throw 'Task 15 scoped index patch does not apply cleanly' }
git apply --cached --unidiff-zero $r5ScopedPatch
if ($LASTEXITCODE -ne 0) { throw 'Task 15 scoped index patch failed' }
git add -- pipeline/tests/fixtures/style_copy_corpus/matrix.json
if ($LASTEXITCODE -ne 0) { throw 'Task 15 clean-path staging failed' }
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Task 15 staged diff check failed' }
$expectedTask15 = @('pipeline/tools/validate_owner_visual_matrix.py','pipeline/tests/test_owner_visual_matrix_tool.py','pipeline/tests/fixtures/style_copy_corpus/matrix.json') | Sort-Object
$stagedTask15 = @(git diff --cached --name-only) | Sort-Object
if ($LASTEXITCODE -ne 0 -or (Compare-Object $expectedTask15 $stagedTask15)) { throw 'Task 15 staged path set is incomplete or contaminated' }
git commit -m "fix(qa): summarize exact holdout inspection rates"
if ($LASTEXITCODE -ne 0) { throw 'Task 15 commit failed' }
```

### Task 16: Compor uma única aceitação R5 fail-closed

**Files:**

- Create: `pipeline/debug_tools/style_copy_r5_acceptance.py`
- Create: `pipeline/qa/execution_source_guard.py`
- Create: `pipeline/tests/test_style_copy_r5_acceptance.py`
- Create: `pipeline/tests/test_execution_source_guard.py`
- Create: `pipeline/tests/fixtures/style_copy_r5/acceptance_sources.json`
- Modify: `pipeline/main.py` para publicar o ledger do subprocesso quando o acceptance bundle estiver ativo.
- Modify: `pipeline/tests/fixtures/style_benchmark_v2/benchmark_spec.json` para thresholds agrupados e versionados.
- Modify: `pipeline/debug_tools/run_style_benchmark_v2.py` somente se necessário para emitir provenance compatível.
- Modify: `pipeline/debug_tools/style_runtime_probe.py` para ligar a seleção `--matrix` ao bundle já definido.
- Modify: `pipeline/tools/validate_owner_visual_matrix.py` para receber/publicar o mesmo acceptance bundle.
- Modify: `pipeline/tests/test_style_benchmark_v2_score.py`
- Modify: `pipeline/tests/test_owner_visual_matrix_tool.py`
- Modify: `pipeline/tests/test_main_emit.py`
- Modify: `pipeline/tests/test_style_runtime_probe.py`

**Step 1: Escrever os testes RED do agregador**

Proteja o import do novo CLI com `find_spec()` ou skeleton importável e confirme exit 1 comportamental antes de implementar.

```powershell
New-Item -ItemType Directory -Force -Path pipeline/tests/fixtures/style_copy_r5 | Out-Null
```

```python
def test_acceptance_passes_only_with_complete_bound_evidence(tmp_path):
    bundle = passing_acceptance_bundle(tmp_path)
    result = evaluate_acceptance(
        acceptance_bundle=bundle,
        synthetic=passing_synthetic_summary(),
        owner_qa=passing_owner_qa_summary(),
        inspection=passing_holdout_summary(),
        matrix_execution=passing_matrix_execution_summary(),
        thresholds=thresholds(),
    )
    assert result["status"] == "PASS"
    assert result["verdicts"] == {
        "style": "GO", "functional": "GO", "inspection": "GO", "overall": "GO"
    }


def test_distinct_producer_run_ids_pass_when_they_share_one_authenticated_bundle(tmp_path):
    inputs = passing_acceptance_inputs(
        benchmark_run_id="benchmark-001",
        matrix_run_ids=["matrix-a", "matrix-b", "matrix-c"],
        acceptance_bundle_id="bundle-123",
    )
    assert evaluate_acceptance(**inputs, thresholds=thresholds())["status"] == "PASS"


def test_init_run_creates_fresh_root_bundle_and_active_manifest(tmp_path):
    first = init_run(valid_init_args(tmp_path))
    second = init_run(valid_init_args(tmp_path))
    assert first.run_root != second.run_root
    assert Path(first.acceptance_bundle_path).is_file()
    assert load_active_manifest(first.active_manifest_path)["acceptance_bundle_id"] == first.acceptance_bundle_id


@pytest.mark.parametrize("problem", ["missing_required_source", "duplicate_source", "outside_repo"])
def test_init_run_rejects_incomplete_or_unsafe_source_manifest(tmp_path, problem):
    with pytest.raises(AcceptanceBundleError):
        init_run(init_args_with_source_manifest_problem(tmp_path, problem))


def test_init_run_hashes_current_dirty_bytes_for_every_execution_source(tmp_path):
    repo = build_minimal_acceptance_repo(tmp_path)
    dirty_source = repo / "pipeline/translator/translate.py"
    dirty_source.write_text(dirty_source.read_text(encoding="utf-8") + "\nDIRTY_SENTINEL = True\n", encoding="utf-8")
    bundle = init_run(valid_init_args(tmp_path, repo_root=repo)).acceptance_bundle
    assert bundle["execution_sources"]["pipeline/translator/translate.py"]["dirty"] is True
    assert bundle["execution_sources"]["pipeline/translator/translate.py"]["content_sha256"] == sha256_file(dirty_source)


def test_execution_guard_blocks_loaded_repo_module_outside_authenticated_closure(tmp_path):
    bundle = passing_acceptance_bundle(tmp_path)
    load_repo_module_not_declared_by_bundle(tmp_path)
    with pytest.raises(ExecutionSourceError, match="unmanifested_loaded_source"):
        finalize_execution_source_ledger(bundle)


@pytest.mark.parametrize("problem", ["missing_child_ledger", "loaded_hash_mismatch", "source_changed_after_init"])
def test_execution_guard_blocks_incomplete_or_stale_runtime_closure(tmp_path, problem):
    repo, bundle, ledgers = prepared_guard_fixture(tmp_path)
    inject_execution_source_problem(repo, bundle, ledgers, problem)
    with pytest.raises(ExecutionSourceError):
        finalize_acceptance_execution_ledgers(bundle, ledgers, required_children={"benchmark", "matrix", "pipeline"})


def test_init_run_blocks_dirty_execution_source_outside_manifest(tmp_path):
    repo = build_minimal_acceptance_repo(tmp_path)
    dirty_source = repo / "pipeline/inpainter/owner_mask.py"
    dirty_source.write_text(dirty_source.read_text(encoding="utf-8") + "\nDIRTY_SENTINEL = True\n", encoding="utf-8")
    manifest = valid_source_manifest_without(repo, "pipeline/inpainter/owner_mask.py")
    with pytest.raises(AcceptanceBundleError, match="dirty_execution_source_outside_manifest"):
        init_run(valid_init_args(tmp_path, repo_root=repo, source_manifest=manifest))


def test_verify_inputs_recomputes_tree_hash_and_blocks_substitution(tmp_path, monkeypatch):
    inputs = exact_input_manifest(tmp_path)
    monkeypatch.setenv("TRADUZAI_MATRIX_MYTHIC_SOURCE", str(inputs.exact_source))
    assert verify_inputs(inputs.path)["status"] == "PASS"
    monkeypatch.setenv("TRADUZAI_MATRIX_MYTHIC_SOURCE", str(inputs.substitute_source))
    assert verify_inputs(inputs.path)["status"] == "BLOCK"


def test_each_producer_publishes_its_own_run_id_under_the_same_bundle(tmp_path):
    bundle = passing_acceptance_bundle(tmp_path)
    benchmark = run_benchmark_with_bundle(bundle, run_id="benchmark-001")
    matrix = run_matrix_with_bundle(bundle, run_id="matrix-001")
    assert benchmark["producer_run_id"] != matrix["producer_run_id"]
    assert benchmark["acceptance_bundle_id"] == matrix["acceptance_bundle_id"] == bundle["acceptance_bundle_id"]
    assert benchmark["revision_sha256"] == matrix["revision_sha256"] == bundle["revision_sha256"]


def test_runtime_probe_requires_matrix_hash_bound_by_acceptance_bundle(tmp_path):
    bundle = passing_acceptance_bundle(tmp_path, matrix=exact_nine_target_matrix(tmp_path))
    report = run_runtime_probe(projects(), matrix=bundle.matrix_path, acceptance_bundle=bundle.path)
    assert report["target_count"] == 9
    assert report["matrix_sha256"] == bundle["matrix_sha256"]
    with pytest.raises(RuntimeProbeError, match="matrix.*bundle"):
        run_runtime_probe(projects(), matrix=different_matrix(tmp_path), acceptance_bundle=bundle.path)


@pytest.mark.parametrize("problem", [
    "missing_metric", "zero_denominator", "schema_mismatch", "bundle_mismatch",
    "revision_mismatch", "source_manifest_mismatch", "hash_mismatch",
    "threshold_breach", "pending_inspection", "missing_matrix_execution",
    "export_gate_hash_mismatch", "functional_block"
])
def test_acceptance_blocks_every_incomplete_or_divergent_input(tmp_path, problem):
    inputs = passing_acceptance_inputs()
    inject_acceptance_problem(inputs, problem)
    result = evaluate_acceptance(**inputs, thresholds=thresholds())
    assert result["status"] == "BLOCK"
    assert result["verdicts"]["overall"] == "NO-GO"
    assert result["findings"]


def test_acceptance_blocks_human_go_when_any_authenticated_export_gate_is_blocked(tmp_path):
    inputs = passing_acceptance_inputs(inspection=all_human_go_summary())
    inputs["matrix_execution"] = matrix_execution_with_entry_gate(
        entry_id="style_cross_tile_holdout", export_gate_status="BLOCK", bound_hash=True
    )
    result = evaluate_acceptance(**inputs, thresholds=thresholds())
    assert result["status"] == "BLOCK"
    assert result["verdicts"]["functional"] == "NO-GO"
    assert "entry_export_gate_block" in finding_codes(result)


def test_acceptance_cli_returns_two_on_block_and_writes_summary(tmp_path):
    paths = write_blocked_acceptance_inputs(tmp_path)
    output = tmp_path / "acceptance.json"
    exit_code = main([*paths.cli_args(), "--output", str(output), "--mode", "enforce"])
    assert exit_code == 2
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "BLOCK"
```

**Step 2: Confirmar RED**

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_style_copy_r5_acceptance.py pipeline/tests/test_execution_source_guard.py pipeline/tests/test_style_benchmark_v2_score.py pipeline/tests/test_owner_visual_matrix_tool.py -k "acceptance or producer or bundle or execution_source" -q
if ($LASTEXITCODE -ne 1) { throw "Expected behavioral RED exit 1, got ${LASTEXITCODE}: final acceptance aggregator is missing" }
```

**Step 3: Implementar o agregador**

CLI obrigatória:

```text
style_copy_r5_acceptance.py init-run
  --repo-root <checkout>
  --runs-root <external-runs-root>
  --run-prefix style_copy_r5_holdout
  --source-manifest <acceptance_sources.json>
  --benchmark-spec <benchmark_spec.json>
  --matrix <matrix.json>
  --matrix-inputs <inputs.json>
  --seed 1729
  --active-manifest <.codex-tmp/style-copy-r5-active-run.json>

style_copy_r5_acceptance.py validate-active
  --active-manifest <.codex-tmp/style-copy-r5-active-run.json>

style_copy_r5_acceptance.py verify-inputs
  --matrix-inputs <inputs.json>

style_copy_r5_acceptance.py evaluate
  --acceptance-bundle <acceptance_bundle.json>
  --benchmark-summary <style_benchmark_summary.json>
  --owner-qa-summary <style_owner_qa_summary.json>
  --inspection-summary <style_holdout_summary.json>
  --matrix-execution-summary <matrix_execution_summary.json>
  --thresholds <benchmark_spec.json>
  --output <style_copy_r5_acceptance.json>
  --mode enforce
```

`verify-inputs` resolve somente as env vars declaradas no `inputs.json`, recalcula `sha256-tree-v1` com a mesma implementação usada pela matriz e retorna 2 em path ausente, substituído ou hash divergente. `init-run` chama essa validação novamente antes de criar qualquer root. Depois, cria atomicamente um root externo nunca reutilizado, `acceptance_bundle.json` e um active manifest JSON self-hashed sob `.codex-tmp` com attempt ID, run root, bundle path/hash/id. `validate-active` recalcula manifest/bundle hashes, exige paths dentro dos roots declarados e rejeita adulteração. O bundle deriva de Git HEAD, hash do diff restrito à closure proprietária, hashes dos **bytes atuais** de cada source/asset expandido, spec, matrix, matrix inputs, hashes dos três inputs resolvidos e seed. Arquivo tracked dirty não é representado só por HEAD: seu `content_sha256`, `git_blob_sha256` quando houver e `dirty=true` entram no bundle. Benchmark, matrix e filhos de pipeline recebem `--acceptance-bundle`/`TRADUZAI_ACCEPTANCE_BUNDLE`, mantêm `producer_run_id` próprios e publicam o mesmo bundle ID + revision/config hashes.

`acceptance_sources.json` usa paths repo-relative, sem globs, e contém obrigatoriamente:

```text
pipeline/main.py
pipeline/requirements.txt
fonts/font-map.json
pipeline/typesetter/style_contract.py
pipeline/typesetter/style_materialization.py
pipeline/typesetter/font_identity.py
pipeline/typesetter/style_groups.py
pipeline/typesetter/owner_style.py
pipeline/typesetter/glyph_rasterizer.py
pipeline/typesetter/renderer.py
pipeline/typesetter/backend_contract.py
pipeline/typesetter/style_capture.py
pipeline/typesetter/style_extractor.py
pipeline/typesetter/style_policy.py
pipeline/typesetter/font_matcher.py
pipeline/ownership/model.py
pipeline/ownership/reconcile.py
pipeline/ownership/translation.py
pipeline/ownership/evidence.py
pipeline/ownership/render_geometry.py
pipeline/ownership/delivery.py
pipeline/ownership/artifacts.py
pipeline/layout/balloon_layout.py
pipeline/compositor/owner_compositor.py
pipeline/inpainter/owner_mask.py
pipeline/translator/translate.py
pipeline/translator/locale_policy.py
pipeline/strip/page_surface_geometry.py
pipeline/strip/types.py
pipeline/strip/process_bands.py
pipeline/strip/run.py
pipeline/qa/style_fidelity.py
pipeline/qa/style_fidelity_policy.py
pipeline/qa/execution_source_guard.py
pipeline/qa/export_gate.py
pipeline/qa/final_pixel_observer.py
pipeline/qa/final_pixel_qa.py
pipeline/qa/inpaint_residual.py
pipeline/vision_stack/runtime.py
pipeline/tools/build_style_owner_target_manifest.py
pipeline/tools/validate_owner_visual_matrix.py
pipeline/debug_tools/generate_style_benchmark_v2.py
pipeline/debug_tools/run_style_benchmark_v2.py
pipeline/debug_tools/style_benchmark_report.py
pipeline/debug_tools/style_copy_score.py
pipeline/debug_tools/style_runtime_probe.py
pipeline/debug_tools/style_copy_r5_acceptance.py
pipeline/debug_tools/style_copy_benchmark_runtime.lock.json
pipeline/tests/fixtures/style_benchmark_v2/benchmark_spec.json
pipeline/tests/fixtures/style_copy_corpus/matrix.json
pipeline/tests/fixtures/owner_visual_matrix/inputs.json
```

Além dos arquivos mínimos acima, o manifest declara `closure_roots=["pipeline"]` com inclusão de todo `.py` repo-local e assets runtime explicitamente referenciados. A allowlist de exclusão é fechada e versionada exatamente como `pipeline/tests/**`, `pipeline/venv/**`, `pipeline/.pytest_cache/**`, `pipeline/.tmp/**`, `pipeline/dummy-debug/**` e `**/__pycache__/**`; não existe exclusão genérica por nome `debug`, `legacy`, `scratch` ou `studio`. Código sob `pipeline/tools`, `pipeline/debug_tools`, OCR, detector, translator, inpainter, strip, QA e demais subpacotes entra no snapshot. `init-run` expande a closure deterministicamente, registra a lista ordenada e bloqueia arquivo dirty sob ela que não esteja no conjunto expandido. O validator possui o mesmo required set/versionamento e rejeita omissão, duplicata, path inexistente/outside-repo ou exclusão não permitida. Spec/matrix/inputs ainda recebem hashes semânticos próprios além do source-manifest hash.

`execution_source_guard.py` registra, no encerramento de cada produtor e subprocesso, todo `module.__file__` repo-local realmente carregado, path resolvido e SHA-256. O ledger deve ser subconjunto do snapshot autenticado e os hashes devem coincidir; módulo carregado fora da closure, source alterado durante o run, ledger ausente de filho ou dirty closure file omitido produz BLOCK. Assim a lista explícita é auditável e o censo runtime fecha imports lazy/dinâmicos. O bundle também inclui Python executable/version, versões de NumPy/OpenCV/matplotlib/fonttools, `font-map.json`, o runtime lock e um `font_catalog_sha256` dos arquivos realmente resolvidos pelo mapa; isso impede comparar runs com motores/fontes diferentes sob o mesmo ID.

Nesta task, adicione `--acceptance-bundle` ao runtime probe já matrix-targeted da Task 14. O probe valida que o hash da mesma `--matrix` é o hash selado no bundle, publica bundle/revision/config hashes e mantém a resolução exata dos nove target keys; o bundle autentica a seleção, mas nunca substitui `--matrix` nem faz o probe contar todos os owners.

`validate_owner_visual_matrix.py` publica em `matrix_execution_summary.json`, por entrada, `project_json` path/hash, `export_gate.json` path/hash/status, route/final-language/inpaint-residual gate status, pipeline child ledger hash e o acceptance-bundle ID. Ausência, duplicata, path fora do run root, hash divergente ou qualquer gate não-PASS mantém `functional_status=BLOCK` mesmo se toda inspeção humana disser GO.

`evaluate` recebe esse summary como fonte funcional obrigatória e valida schema, bundle, source/tool/input/artifact hashes e provenance. Run IDs dos produtores devem ser distintos/locais; não os compare por igualdade entre benchmark e matriz. Mistura de bundle/revision/config, child ledger ou export-gate binding bloqueia. Aplique:

- font top-1 >= 0.90; top-3 >= 0.98;
- fill Delta E median <= 8; p95 <= 12;
- stroke Delta E <= 10; thickness error <= `max(1 px, 20% x-height)`;
- glow/shadow Delta E <= 12; radius/offset error <= `max(2 px, 25%)`;
- rotation error <= 3 graus;
- width/slant/tracking/occupancy error <= 15%;
- final/source x-height entre 0.85 e 1.15 quando container permite;
- `policy_adjusted` fora da whitelist = 0; font-size intent/target ratio entre 0.70 e 1.10 e acima do legibility floor, com denominador explícito;
- safe/effect containment = 1.0; false decorative effects <= 0.02;
- catastrophic high-confidence mismatch = 0;
- owner GO >= 0.90; speech GO >= 0.95; cada categoria required >= 0.85;
- inspection coverage = 1.0 e todos os targets requeridos com denominador não zero;
- Functional gate PASS para cada entrada.

O agregador não recalcula imagem nem inventa métrica; apenas valida, compõe e explica. `--mode enforce` retorna 2 em BLOCK e 0 somente em PASS.

**Step 4: Confirmar GREEN e checkpoint R5-D**

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_style_benchmark_v2_generator.py pipeline/tests/test_style_benchmark_v2_score.py pipeline/tests/test_style_fidelity_qa.py pipeline/tests/test_style_runtime_probe.py pipeline/tests/test_owner_visual_matrix_tool.py pipeline/tests/test_style_copy_r5_acceptance.py pipeline/tests/test_execution_source_guard.py pipeline/tests/test_main_emit.py -q
if ($LASTEXITCODE -ne 0) { throw 'R5-D checkpoint failed' }
```

**Step 5: Commit**

Crie com `apply_patch` `$r5Tmp\task16-dirty-index.patch` contendo somente os hunks R5 do validator e de seu teste já dirty.

```powershell
$r5PlanAnchor = git log -1 --format=%H -- docs/plans/2026-08-01-style-copy-r5-materialization-geometry-remediation-implementation.md
$r5Tmp = Join-Path (Resolve-Path '.') ('.codex-tmp/style-copy-r5-' + $r5PlanAnchor.Substring(0, 12))
$r5ScopedPatch = Join-Path $r5Tmp 'task16-dirty-index.patch'
if (-not (Test-Path -LiteralPath $r5ScopedPatch)) { throw 'Create the Task 16 scoped index patch with apply_patch first' }
git apply --cached --check --unidiff-zero $r5ScopedPatch
if ($LASTEXITCODE -ne 0) { throw 'Task 16 scoped index patch does not apply cleanly' }
git apply --cached --unidiff-zero $r5ScopedPatch
if ($LASTEXITCODE -ne 0) { throw 'Task 16 scoped index patch failed' }
git add -- pipeline/debug_tools/style_copy_r5_acceptance.py pipeline/qa/execution_source_guard.py pipeline/main.py pipeline/debug_tools/style_runtime_probe.py pipeline/tests/test_style_copy_r5_acceptance.py pipeline/tests/test_execution_source_guard.py pipeline/tests/test_main_emit.py pipeline/tests/test_style_runtime_probe.py pipeline/tests/fixtures/style_copy_r5/acceptance_sources.json pipeline/tests/fixtures/style_benchmark_v2/benchmark_spec.json pipeline/debug_tools/run_style_benchmark_v2.py pipeline/tests/test_style_benchmark_v2_score.py
if ($LASTEXITCODE -ne 0) { throw 'Task 16 clean-path staging failed' }
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Task 16 staged diff check failed' }
$expectedTask16 = @('pipeline/debug_tools/style_copy_r5_acceptance.py','pipeline/qa/execution_source_guard.py','pipeline/main.py','pipeline/debug_tools/style_runtime_probe.py','pipeline/tests/test_style_copy_r5_acceptance.py','pipeline/tests/test_execution_source_guard.py','pipeline/tests/test_main_emit.py','pipeline/tests/test_style_runtime_probe.py','pipeline/tests/fixtures/style_copy_r5/acceptance_sources.json','pipeline/tests/fixtures/style_benchmark_v2/benchmark_spec.json','pipeline/debug_tools/run_style_benchmark_v2.py','pipeline/tests/test_style_benchmark_v2_score.py','pipeline/tools/validate_owner_visual_matrix.py','pipeline/tests/test_owner_visual_matrix_tool.py') | Sort-Object
$stagedTask16 = @(git diff --cached --name-only) | Sort-Object
if ($LASTEXITCODE -ne 0 -or (Compare-Object $expectedTask16 $stagedTask16)) { throw 'Task 16 staged path set is incomplete or contaminated' }
git commit -m "feat(style): compose authenticated r5 acceptance evidence"
if ($LASTEXITCODE -ne 0) { throw 'Task 16 commit failed' }
```

### Task 17: Congelar uma sentinela sistêmica para as três classes de falha e os nove owners

**Files:**

- Create: `pipeline/tests/fixtures/style_copy_r5/no_go_cases.json`
- Create: `pipeline/tests/test_style_copy_r5_systemic.py`
- Modify: `pipeline/tests/test_style_owner_target_manifest.py`
- Test: `pipeline/tests/fixtures/style_copy_corpus/matrix.json`
- Test: `docs/reports/evidence/2026-07-31-style-copy-v2-remediation-summary.json`

**Step 1: Escrever os testes RED sistêmicos**

Congele somente identidades, decisões mínimas e razões do R4; não copie imagens externas para Git.

```python
def test_r5_card_profile_materializes_nonempty_complete_raster():
    case = load_r5_case("card_fill_stroke_glow")
    patch = execute_synthetic_owner_case(case)
    assert patch.render_completed is True
    assert np.count_nonzero(patch.glyph_core_mask) > 0
    assert patch.style_raster_contract.materialization_comparison["status"] == "match"


def test_r5_fill_equivalence_commits_but_true_difference_rolls_back():
    assert execute_case("canonical_fill_alias").committed is True
    mismatch = execute_case("material_fill_difference")
    assert mismatch.committed is False
    assert ":raster:fill:" in mismatch.reason


def test_r5_cross_frame_case_preserves_complete_body_and_correct_crop():
    case = load_r5_case("cross_tile_framed_output")
    result = execute_cross_frame_case(case)
    assert result.delivery.status == "delivered"
    assert result.frame_bbox == (164, 985, 373, 1104)
    assert result.final_text == "É POR ISSO QUE VOCÊ DEVERIA TER IDO ANTES"


@pytest.mark.parametrize("case_id", ["white_balloon", "burst"])
def test_r5_speech_containers_keep_all_glyphs_inside_verified_safe_geometry(case_id):
    result = execute_synthetic_owner_case(load_r5_case(case_id))
    assert result.delivery.status == "delivered"
    assert result.core_pixels_outside_safe == 0


def test_r5_dark_panel_observes_real_font_and_fill_without_request_echo():
    patch = execute_synthetic_owner_case(load_r5_case("dark_panel"))
    assert patch.style_raster_contract.materialization_comparison["status"] == "match"
    assert patch.style_raster_contract.materialization_observation["attributes"]["fill"]["evidence_kind"] == "glyph_raster"


def test_r5_text_over_art_preserves_protected_art_and_complete_body():
    result = execute_synthetic_owner_case(load_r5_case("text_over_art"))
    assert result.delivery.status == "delivered"
    assert result.protected_art_changed_pixels == 0


def test_exact_nine_owner_targets_and_source_hashes_are_unchanged():
    matrix = load_style_matrix()
    expected = {
        ("style_colored_cards_calibration", "page_001", "owner_p001_024cf5ddf5ea"): ([246, 10391, 628, 10542], "2a2f55262ed94479e8ee1819a8eb5e932e7a1a5af5b56b5a6f984da0ba6b2404"),
        ("style_colored_cards_calibration", "page_001", "owner_p001_06c2c46ee27f"): ([340, 4052, 602, 4180], "8f17bc734c4946c6eea066e001a30e7900cb1fa54e4d0971f2bc4cd786ede261"),
        ("style_colored_cards_calibration", "page_001", "owner_p001_0dc05817313b"): ([198, 6057, 604, 6242], "4f145ec82428d6deb3c6b427baa450411e0af75774bdde0c3bdf26523c0b40dc"),
        ("style_colored_cards_calibration", "page_001", "owner_p001_0f39ce3adba0"): ([169, 11133, 542, 11247], "00e896e3eabc1003c611cfd8fb5ac847534c1847c10a8bf99f5506229bdf4f23"),
        ("style_cross_tile_holdout", "page_002", "owner_p002_43752a8f8c08"): ([391, 484, 618, 595], "2c2147610a183eeb8e7500ce5d44c5aba4d0f43fc7f09682646345eb89b2ee10"),
        ("style_cross_tile_holdout", "page_002", "owner_p002_7227e479da0d"): ([76, 54, 243, 133], "23aa55bd1525fab675a524e9856f5beea464a0eea07c87ab77508180c7048dbe"),
        ("style_cross_tile_holdout", "page_002", "owner_p002_d9afb32ef935"): ([109, 985, 318, 1104], "e4c60408223dfef9a4f0236b41b07c603b983420e9ca5a3715a030ccb1532c31"),
        ("style_dark_panels_holdout", "page_002", "owner_p002_5a6976dbbf65"): ([107, 1915, 424, 2114], "3596b34910f6cb295478af3daa875224b742e8ebc250735e4c7db86e6b0aeeab"),
        ("style_dark_panels_holdout", "page_003", "owner_p003_98e72af0685a"): ([183, 572, 514, 682], "054fe2c77ba95ef21ccc6c4f9d90975f2293938e10649419304a19b1efcfc5c3"),
    }
    assert exact_target_source_map(matrix) == expected
    assert all(target["source_crop"]["coordinate_space"] == "logical_page" for target in all_targets(matrix))
```

Essa autoridade literal congela conjuntamente `entry_id/page_id/owner_id`, bbox lógico half-open e SHA-256 do source crop. Não derive o expected da própria matriz. Os nove IDs são:

```text
page_001/owner_p001_024cf5ddf5ea
page_001/owner_p001_06c2c46ee27f
page_001/owner_p001_0dc05817313b
page_001/owner_p001_0f39ce3adba0
page_002/owner_p002_43752a8f8c08
page_002/owner_p002_7227e479da0d
page_002/owner_p002_d9afb32ef935
page_002/owner_p002_5a6976dbbf65
page_003/owner_p003_98e72af0685a
```

**Step 2: Confirmar RED**

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_style_copy_r5_systemic.py pipeline/tests/test_style_owner_target_manifest.py -q
if ($LASTEXITCODE -ne 1) { throw "Expected behavioral RED exit 1, got ${LASTEXITCODE}: systemic R5 sentinel is not frozen" }
```

O primeiro RED pode ser fixture ausente; crie então o JSON mínimo e rode novamente até a falha ser comportamental no caminho real, não apenas file-not-found.

**Step 3: Criar fixture mínima e adaptar helpers reais**

O fixture deve conter:

- schema/version e origem R4;
- categorias sistêmicas, nunca lógica de owner ID;
- perfis/applied decisions mínimos para card, fill alias/difference, cross-frame, white balloon, burst, dark panel e text-over-art;
- dimensões/offset do frame;
- payload source/PT-BR completo;
- razões esperadas.

Os helpers de teste chamam os builders/renderer/compositor reais. Não mockar o comparator, geometry transform, rasterizer ou atomic commit.

**Step 4: Confirmar GREEN e checkpoint R5-E parcial**

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_style_copy_r5_systemic.py pipeline/tests/test_style_owner_target_manifest.py pipeline/tests/test_style_materialization.py pipeline/tests/test_owner_render_geometry.py pipeline/tests/test_owner_delivery_contract.py -q
if ($LASTEXITCODE -ne 0) { throw 'R5 systemic sentinel failed' }
```

**Step 5: Verificar ausência de exceções por obra/página**

Run:

```powershell
Push-Location pipeline
try {
  rg -n "024cf5ddf5ea|06c2c46ee27f|0dc05817313b|0f39ce3adba0|43752a8f8c08|7227e479da0d|d9afb32ef935|5a6976dbbf65|98e72af0685a|regressed_genius|one_second|grand_finale" . --glob '!tests/**' --glob '!**/fixtures/**' --glob '!tools/**' --glob '!debug_tools/**'
  $rgExit = $LASTEXITCODE
  if ($rgExit -eq 0) { throw 'Owner/work-specific exception found in production code' }
  if ($rgExit -gt 1) { throw "rg failed with exit $rgExit" }
} finally {
  Pop-Location
}
```

O bloco entra explicitamente em `pipeline`; os globs são relativos a esse root e não dependem do diretório do shell chamador.

Expected: zero hits em produção. IDs podem existir somente em fixtures/tools de validação exata.

**Step 6: Commit**

```powershell
git add -- pipeline/tests/fixtures/style_copy_r5/no_go_cases.json pipeline/tests/test_style_copy_r5_systemic.py pipeline/tests/test_style_owner_target_manifest.py
if ($LASTEXITCODE -ne 0) { throw 'Task 17 staging failed' }
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Task 17 staged diff check failed' }
git diff --cached --name-status
if ($LASTEXITCODE -ne 0) { throw 'Task 17 staged path inspection failed' }
git commit -m "test(style): freeze r5 systemic owner regressions"
if ($LASTEXITCODE -ne 0) { throw 'Task 17 commit failed' }
```

### Task 18: Rodar suites focadas, invariância e diferencial amplo

**Files:**

- Create: `pipeline/tools/compare_pytest_junit.py`
- Create: `pipeline/tests/test_compare_pytest_junit.py`
- Outros arquivos somente se um RED novo estritamente necessário demonstrar regressão R5.

**Step 1: Escrever RED do comparador diferencial de JUnit**

Proteja o import do helper novo com `find_spec()` ou skeleton importável e confirme exit 1 comportamental antes de implementar.

```python
def test_junit_comparator_reports_only_new_failure_error_nodeids(tmp_path):
    baseline = write_junit(tmp_path / "baseline.xml", failures=[("a.Test", "test_old[x]")])
    candidate = write_junit(tmp_path / "candidate.xml", failures=[
        ("a.Test", "test_old[x]"), ("b.Test", "test_new[y]"),
    ])
    result = compare_junit(baseline, candidate)
    assert result["new_failures"] == [{"classname": "b.Test", "name": "test_new[y]"}]
    assert result["resolved_failures"] == []


def test_junit_comparator_rejects_malformed_or_empty_suite(tmp_path):
    with pytest.raises(JunitComparisonError):
        compare_junit(write_text(tmp_path / "bad.xml", "<bad>"), valid_junit(tmp_path))


def test_junit_comparator_blocks_baseline_nodeid_missing_from_candidate(tmp_path):
    baseline = write_junit(tmp_path / "baseline.xml", passed=[("a.Test", "test_kept"), ("a.Test", "test_must_not_disappear")])
    candidate = write_junit(tmp_path / "candidate.xml", passed=[("a.Test", "test_kept")])
    result = compare_junit(baseline, candidate)
    assert result["missing_nodeids"] == [{"classname": "a.Test", "name": "test_must_not_disappear"}]
    assert result["status"] == "BLOCK"
```

**Step 2: Confirmar RED e implementar o helper mínimo**

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_compare_pytest_junit.py -q
if ($LASTEXITCODE -ne 1) { throw "Expected behavioral RED exit 1, got ${LASTEXITCODE}: JUnit comparator is missing" }
```

Implemente parser seguro que usa `(classname,name)` completo — incluindo parametrização/subtests no name —, distingue pass/skip/failure/error, publica JSON canônico e retorna 2 com `--fail-on-new` quando aparece qualquer failure/error novo **ou qualquer nodeid do baseline desaparece da coleta candidate**. Novos nodeids são permitidos; nodeid renomeado/removido exige primeiro uma allowlist versionada, específica e justificada no plano — esta execução R5 não possui allowlist. XML ausente, vazio, malformado ou com zero testcases é erro operacional, nunca PASS.

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_compare_pytest_junit.py -q
if ($LASTEXITCODE -ne 0) { throw 'JUnit comparator GREEN failed' }
git add -- pipeline/tools/compare_pytest_junit.py pipeline/tests/test_compare_pytest_junit.py
if ($LASTEXITCODE -ne 0) { throw 'Task 18 comparator staging failed' }
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Task 18 comparator staged diff check failed' }
git commit -m "test(qa): compare pytest junit differentially"
if ($LASTEXITCODE -ne 0) { throw 'Task 18 comparator commit failed' }
```

**Step 3: Suite completa de materialização/style**

Run:

```powershell
$r5PlanAnchor = git log -1 --format=%H -- docs/plans/2026-08-01-style-copy-r5-materialization-geometry-remediation-implementation.md
$r5Tmp = Join-Path (Resolve-Path '.') ('.codex-tmp/style-copy-r5-' + $r5PlanAnchor.Substring(0, 12))
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_style_materialization.py pipeline/tests/test_font_identity.py pipeline/tests/test_style_contract.py pipeline/tests/test_style_groups.py pipeline/tests/test_glyph_rasterizer.py pipeline/tests/test_typesetting_renderer.py pipeline/tests/test_typesetting_backend_contract.py pipeline/tests/test_owner_style_profile.py pipeline/tests/test_style_fidelity_qa.py pipeline/tests/test_style_runtime_probe.py pipeline/tests/test_style_benchmark_v2_generator.py pipeline/tests/test_style_benchmark_v2_score.py pipeline/tests/test_style_copy_r5_acceptance.py pipeline/tests/test_style_copy_r5_systemic.py -q --junitxml "$r5Tmp\style-final.xml"
if ($LASTEXITCODE -ne 0) { throw 'R5 style suite failed' }
```

**Step 4: Suite funcional/owner/geometry adjacente**

```powershell
$r5PlanAnchor = git log -1 --format=%H -- docs/plans/2026-08-01-style-copy-r5-materialization-geometry-remediation-implementation.md
$r5Tmp = Join-Path (Resolve-Path '.') ('.codex-tmp/style-copy-r5-' + $r5PlanAnchor.Substring(0, 12))
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_page_surface_geometry.py pipeline/tests/test_owner_render_geometry.py pipeline/tests/test_owner_delivery_contract.py pipeline/tests/test_owner_reconcile.py pipeline/tests/test_owner_reconcile_properties.py pipeline/tests/test_owner_translation.py pipeline/tests/test_owner_layout.py pipeline/tests/test_owner_enforcement.py pipeline/tests/test_owner_mask.py pipeline/tests/test_owner_atomic_execution.py pipeline/tests/test_owner_compositor.py pipeline/tests/test_owner_compositor_properties.py pipeline/tests/test_strip_owner_composition_integration.py pipeline/tests/test_balloon_layout_shared_regions.py pipeline/tests/test_final_pixel_qa.py pipeline/tests/test_final_pixel_export_gate.py pipeline/tests/test_owner_visual_matrix_tool.py -q --junitxml "$r5Tmp\functional-adjacent-final.xml"
if ($LASTEXITCODE -ne 0) { throw 'R5 functional adjacent suite failed' }
```

**Step 5: Provar invariância de conteúdo e máscaras**

Run:

```powershell
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests/test_owner_translation.py pipeline/tests/test_owner_layout.py pipeline/tests/test_style_policy_v2.py pipeline/tests/test_style_mask_layers.py pipeline/tests/test_style_mask_evidence.py -k "complete_payload or unchanged or permutation or one_complete or never_split or mask" -q
if ($LASTEXITCODE -ne 0) { throw 'Semantic or mask invariance failed' }
```

Expected: conteúdo source/translated, cardinalidade owner, action/protection masks e inpaint não mudam por style. Somente pixels de glyph e metadados Style R5 podem variar.

**Step 6: Rodar suite ampla e comparar nodeids**

```powershell
$r5PlanAnchor = git log -1 --format=%H -- docs/plans/2026-08-01-style-copy-r5-materialization-geometry-remediation-implementation.md
$r5Tmp = Join-Path (Resolve-Path '.') ('.codex-tmp/style-copy-r5-' + $r5PlanAnchor.Substring(0, 12))
pipeline/venv/Scripts/python.exe -m pytest pipeline/tests -q --junitxml "$r5Tmp\full-final.xml"
$fullFinalExit = $LASTEXITCODE
Write-Output "full_final_exit=$fullFinalExit"
if ($fullFinalExit -notin @(0, 1)) { throw "Full suite ended operationally with exit $fullFinalExit" }
pipeline/venv/Scripts/python.exe pipeline/tools/compare_pytest_junit.py --baseline "$r5Tmp\full-baseline.xml" --candidate "$r5Tmp\full-final.xml" --output "$r5Tmp\full-junit-diff.json" --fail-on-new
if ($LASTEXITCODE -ne 0) { throw 'New pytest failure/error nodeids detected; inspect full-junit-diff.json' }
```

O helper compara por `(classname,name)`, incluindo parametrização/subtests. Critério: zero failures/errors novos e zero nodeids baseline ausentes. Falhas preexistentes podem ser resolvidas, mas o testcase correspondente precisa continuar coletado (pass/skip/failure/error) para não ser confundido com remoção silenciosa.

Se houver nova falha ou perda de inventário, escreva um RED mínimo no módulo proprietário, corrija e repita Steps 3–6, incluindo novamente a suite ampla e o comparador. Não masque failure com deselect/xfail/threshold.

**Step 7: Auditar diff, índice e preservação local**

```powershell
git status --short --branch
$r5PlanAnchor = git log -1 --format=%H -- docs/plans/2026-08-01-style-copy-r5-materialization-geometry-remediation-implementation.md
$r5Tmp = Join-Path (Resolve-Path '.') ('.codex-tmp/style-copy-r5-' + $r5PlanAnchor.Substring(0, 12))
git diff --check "$r5PlanAnchor..HEAD"
if ($LASTEXITCODE -ne 0) { throw 'R5 committed diff check failed' }
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Unexpected staged diff check failure at final audit' }
$unexpectedStaged = @(git diff --cached --name-only)
if ($LASTEXITCODE -ne 0 -or $unexpectedStaged.Count -ne 0) { throw 'Index is not empty before visual holdout/report work' }
$dirtyRelevant = @('pipeline/inpainter/owner_mask.py','pipeline/ownership/artifacts.py','pipeline/ownership/evidence.py','pipeline/ownership/reconcile.py','pipeline/ownership/translation.py','pipeline/qa/export_gate.py','pipeline/qa/final_pixel_qa.py','pipeline/qa/inpaint_residual.py','pipeline/strip/process_bands.py','pipeline/strip/run.py','pipeline/tests/test_final_pixel_export_gate.py','pipeline/tests/test_final_pixel_qa.py','pipeline/tests/test_inpaint_debug_residual.py','pipeline/tests/test_owner_debug_artifacts.py','pipeline/tests/test_owner_enforcement.py','pipeline/tests/test_owner_evidence.py','pipeline/tests/test_owner_mask.py','pipeline/tests/test_owner_reconcile.py','pipeline/tests/test_owner_translation.py','pipeline/tests/test_owner_visual_matrix_tool.py','pipeline/tests/test_translation_locale_policy.py','pipeline/tools/validate_owner_visual_matrix.py','pipeline/translator/locale_policy.py','pipeline/translator/translate.py')
git diff --binary --output="$r5Tmp\preexisting-residual.patch" -- $dirtyRelevant
if ($LASTEXITCODE -ne 0) { throw 'Could not snapshot residual preexisting diff' }
$basePatchId = (Get-Content -LiteralPath "$r5Tmp\preexisting-relevant.patch" -Raw | git patch-id --stable).Split()[0]
if ($LASTEXITCODE -ne 0 -or -not $basePatchId) { throw 'Could not fingerprint baseline local diff' }
$residualPatchId = (Get-Content -LiteralPath "$r5Tmp\preexisting-residual.patch" -Raw | git patch-id --stable).Split()[0]
if ($LASTEXITCODE -ne 0 -or -not $residualPatchId) { throw 'Could not fingerprint residual local diff' }
if ($basePatchId -ne $residualPatchId) { throw 'Preexisting local diff changed; inspect overlap log and do not reset or discard anything' }
git log --oneline --decorate -20
```

O patch-id stable torna executável a prova de preservação dos paths compartilhados. Se uma linha sobreposta realmente exigida pelo R5 impedir igualdade, pare: registre o path/hunk e prove por diff de três vias que o conteúdo preexistente sobreviveu antes de atualizar a autoridade com `apply_patch`; nunca normalize, descarte ou absorva silenciosamente o hunk. Expected: commits R5 pequenos e verificáveis; mudanças preexistentes não incluídas continuam presentes; nenhuma evidência temporária staged. Whitespace preexistente fora do range R5 não vira obrigação de correção.

### Task 19: Executar holdout novo, inspecionar 9/9 e publicar veredicto

**Files:**

- Create after successful or honest final run: `docs/reports/2026-08-01-style-copy-r5-materialization-validation.md`
- Create after successful or honest final run: `docs/reports/evidence/2026-08-01-style-copy-r5-materialization-summary.json`

**Step 1: Verificar inputs exatos sem substituição**

Run:

```powershell
$env:TRADUZAI_MATRIX_MYTHIC_SOURCE = (Resolve-Path -LiteralPath '.codex-tmp/mythic_ch40_pages_1_2_source_20260724').Path
$env:TRADUZAI_MATRIX_ONE_SECOND_SOURCE = (Resolve-Path -LiteralPath '.codex-tmp/page_owner_systemic_sources_20260729/one_second').Path
$env:TRADUZAI_MATRIX_GRAND_FINALE_SOURCE = (Resolve-Path -LiteralPath '.codex-tmp/page_owner_systemic_sources_20260729/grand_finale').Path
pipeline/venv/Scripts/python.exe pipeline/debug_tools/style_copy_r5_acceptance.py verify-inputs --matrix-inputs pipeline/tests/fixtures/owner_visual_matrix/inputs.json
if ($LASTEXITCODE -ne 0) { throw 'Exact matrix input tree hashes do not match inputs.json' }
```

Expected: os três inputs curados existentes recebem bootstrap apenas no processo atual e batem exatamente em `78242a961ff07fd1120b6da3156a015bc509a077b06fcc395b24d326071e06d3`, `bf8591406a1ed1639d3121178bb85c43383963098aa57b83e1d257e88225b40e` e `fd8ce2fec0e3453f7fae994ac97245ce716bcac2d34ca2e5ce8d1fbf699338af`. Não persistir as env vars globalmente nem substituir capítulo/página por piloto.

**Step 2: Criar root externo, campaign bundle e active manifest atomicamente**

```powershell
$env:TRADUZAI_MATRIX_MYTHIC_SOURCE = (Resolve-Path -LiteralPath '.codex-tmp/mythic_ch40_pages_1_2_source_20260724').Path
$env:TRADUZAI_MATRIX_ONE_SECOND_SOURCE = (Resolve-Path -LiteralPath '.codex-tmp/page_owner_systemic_sources_20260729/one_second').Path
$env:TRADUZAI_MATRIX_GRAND_FINALE_SOURCE = (Resolve-Path -LiteralPath '.codex-tmp/page_owner_systemic_sources_20260729/grand_finale').Path
$r5ActiveManifest = Join-Path (Resolve-Path '.') '.codex-tmp/style-copy-r5-active-run.json'
pipeline/venv/Scripts/python.exe pipeline/debug_tools/style_copy_r5_acceptance.py init-run --repo-root . --runs-root N:\TraduzAI_style_runs --run-prefix style_copy_r5_holdout --source-manifest pipeline/tests/fixtures/style_copy_r5/acceptance_sources.json --benchmark-spec pipeline/tests/fixtures/style_benchmark_v2/benchmark_spec.json --matrix pipeline/tests/fixtures/style_copy_corpus/matrix.json --matrix-inputs pipeline/tests/fixtures/owner_visual_matrix/inputs.json --seed 1729 --active-manifest $r5ActiveManifest
if ($LASTEXITCODE -ne 0) { throw 'Could not initialize fresh R5 run/bundle' }
pipeline/venv/Scripts/python.exe pipeline/debug_tools/style_copy_r5_acceptance.py validate-active --active-manifest $r5ActiveManifest
if ($LASTEXITCODE -ne 0) { throw 'Active R5 run/bundle validation failed' }
$r5Active = Get-Content -LiteralPath $r5ActiveManifest -Raw | ConvertFrom-Json
if (-not (Test-Path -LiteralPath $r5Active.run_root)) { throw 'R5 active run root was not created' }
if (-not (Test-Path -LiteralPath $r5Active.acceptance_bundle_path)) { throw 'R5 acceptance bundle was not created' }
$r5Active | ConvertTo-Json -Depth 8
```

Nunca reutilize root anterior. Cada retry visual ganha timestamp novo.

**Step 3: Rodar benchmark all em enforce**

```powershell
$env:TRADUZAI_MATRIX_MYTHIC_SOURCE = (Resolve-Path -LiteralPath '.codex-tmp/mythic_ch40_pages_1_2_source_20260724').Path
$env:TRADUZAI_MATRIX_ONE_SECOND_SOURCE = (Resolve-Path -LiteralPath '.codex-tmp/page_owner_systemic_sources_20260729/one_second').Path
$env:TRADUZAI_MATRIX_GRAND_FINALE_SOURCE = (Resolve-Path -LiteralPath '.codex-tmp/page_owner_systemic_sources_20260729/grand_finale').Path
$r5ActiveManifest = Join-Path (Resolve-Path '.') '.codex-tmp/style-copy-r5-active-run.json'
$r5Active = Get-Content -LiteralPath $r5ActiveManifest -Raw | ConvertFrom-Json
$r5RunRoot = [string]$r5Active.run_root
$benchmarkRunId = 'benchmark-' + [string]$r5Active.attempt_id
pipeline/venv/Scripts/python.exe pipeline/debug_tools/run_style_benchmark_v2.py --spec pipeline/tests/fixtures/style_benchmark_v2/benchmark_spec.json --level all --output-root "$r5RunRoot\benchmark" --run-id $benchmarkRunId --seed 1729 --mode enforce --acceptance-bundle $r5Active.acceptance_bundle_path
$benchmarkExit = $LASTEXITCODE
Write-Output "benchmark_exit=$benchmarkExit"
if ($benchmarkExit -ne 0) { throw "R5 benchmark failed with exit $benchmarkExit; return to Task 13" }
```

Expected para GO: exit 0, todos os denominadores não zero e thresholds sintéticos aprovados. Exit 2 é diagnóstico legítimo e exige ciclo RED na Task 13; não prossiga para declarar GO.

**Step 4: Rodar matriz owner-targeted**

```powershell
$env:TRADUZAI_MATRIX_MYTHIC_SOURCE = (Resolve-Path -LiteralPath '.codex-tmp/mythic_ch40_pages_1_2_source_20260724').Path
$env:TRADUZAI_MATRIX_ONE_SECOND_SOURCE = (Resolve-Path -LiteralPath '.codex-tmp/page_owner_systemic_sources_20260729/one_second').Path
$env:TRADUZAI_MATRIX_GRAND_FINALE_SOURCE = (Resolve-Path -LiteralPath '.codex-tmp/page_owner_systemic_sources_20260729/grand_finale').Path
$r5ActiveManifest = Join-Path (Resolve-Path '.') '.codex-tmp/style-copy-r5-active-run.json'
$r5Active = Get-Content -LiteralPath $r5ActiveManifest -Raw | ConvertFrom-Json
$r5RunRoot = [string]$r5Active.run_root
$matrixRunId = 'matrix-' + [string]$r5Active.attempt_id
pipeline/venv/Scripts/python.exe pipeline/tools/validate_owner_visual_matrix.py --manifest pipeline/tests/fixtures/style_copy_corpus/matrix.json --output-root "$r5RunRoot\matrix" --report "$r5RunRoot\matrix-report.md" --inspection-template "$r5RunRoot\matrix\style-inspection-template.json" --run-id $matrixRunId --acceptance-bundle $r5Active.acceptance_bundle_path
$matrixInitialExit = $LASTEXITCODE
Write-Output "matrix_initial_exit=$matrixInitialExit"
if ($matrixInitialExit -notin @(0, 2)) { throw "Matrix ended operationally with exit $matrixInitialExit" }
$matrixExecution = Get-Content -LiteralPath "$r5RunRoot\matrix\matrix_execution_summary.json" -Raw | ConvertFrom-Json
if ($matrixExecution.functional_status -ne 'GO' -or $matrixExecution.style_status -ne 'GO') { throw 'Matrix runtime/style contracts are not GO before inspection' }
if ($matrixInitialExit -eq 2 -and ($matrixExecution.inspection_status -ne 'PENDING' -or @($matrixExecution.findings | Where-Object code -ne 'inspection_pending').Count -ne 0)) { throw 'Matrix exit 2 contains failures beyond pending inspection' }
```

Expected: cada target resolve exatamente page/owner/components/source hash/page-surface geometry. A execução inicial pode retornar 2 apenas porque inspeção ainda está `PENDING`; falhas de runtime/gate/target devem ser corrigidas antes da inspeção.

**Step 5: Validar contratos runtime antes de olhar contact sheets**

Descubra os três `project.json` sob o root exato e execute:

```powershell
$r5ActiveManifest = Join-Path (Resolve-Path '.') '.codex-tmp/style-copy-r5-active-run.json'
$r5Active = Get-Content -LiteralPath $r5ActiveManifest -Raw | ConvertFrom-Json
$r5RunRoot = [string]$r5Active.run_root
$targetChecks = @(
  @{ entry='style_colored_cards_calibration'; project="$r5RunRoot\matrix\style_colored_cards_calibration\project.json" },
  @{ entry='style_cross_tile_holdout'; project="$r5RunRoot\matrix\style_cross_tile_holdout\project.json" },
  @{ entry='style_dark_panels_holdout'; project="$r5RunRoot\matrix\style_dark_panels_holdout\project.json" }
)
foreach ($check in $targetChecks) {
  if (-not (Test-Path -LiteralPath $check.project)) { throw "Missing exact project: $($check.project)" }
  $targetReport = "$r5RunRoot\matrix\target-contract-$($check.entry).json"
  pipeline/venv/Scripts/python.exe pipeline/tools/build_style_owner_target_manifest.py --project $check.project --matrix pipeline/tests/fixtures/style_copy_corpus/matrix.json --entry-id $check.entry --verify-only --output $targetReport
  if ($LASTEXITCODE -ne 0) { throw "Target contract verification failed: $($check.entry)" }
}
$projects = @($targetChecks | ForEach-Object { $_.project })
pipeline/venv/Scripts/python.exe pipeline/debug_tools/style_runtime_probe.py --output "$r5RunRoot\runtime-probe.json" --matrix pipeline/tests/fixtures/style_copy_corpus/matrix.json --acceptance-bundle $r5Active.acceptance_bundle_path $projects
$probeExit = $LASTEXITCODE
Write-Output "runtime_probe_exit=$probeExit"
if ($probeExit -ne 0) { throw "Runtime probe failed with exit $probeExit" }
$probe = Get-Content -LiteralPath "$r5RunRoot\runtime-probe.json" -Raw | ConvertFrom-Json
if ($probe.target_count -ne 9 -or $probe.acceptance_bundle_id -ne $r5Active.acceptance_bundle_id) { throw 'Runtime probe coverage/bundle mismatch' }
```

Exija antes da inspeção:

- 9/9 targets com owner exato e source hashes intactos;
- 9/9 owners `rendered`, delivery `delivered`, `glyph_core_mask` e `paint_mask` não vazias/consistentes;
- 100% visual profile, materialization plan, observation, raster contract, render geometry e page-surface geometry coverage;
- zero `owner_execution_rollback`;
- zero mismatch/fallback silencioso;
- zero Functional contract bloqueado;
- todos os `export_gate.json` não bloqueados.

Se qualquer item falhar, classifique por contrato, crie RED na task proprietária e gere um root novo após o fix.

**Step 6: Inspecionar visualmente cada contact sheet em escala nativa**

Primeiro prove cobertura exata pelo manifest, sem descoberta por filename:

```powershell
$r5ActiveManifest = Join-Path (Resolve-Path '.') '.codex-tmp/style-copy-r5-active-run.json'
$r5Active = Get-Content -LiteralPath $r5ActiveManifest -Raw | ConvertFrom-Json
$r5RunRoot = [string]$r5Active.run_root
$inspectionTemplatePath = "$r5RunRoot\matrix\style-inspection-template.json"
$inspectionTemplate = Get-Content -LiteralPath $inspectionTemplatePath -Raw | ConvertFrom-Json
$expectedPairs = @(
  'style_colored_cards_calibration/page_001/owner_p001_024cf5ddf5ea',
  'style_colored_cards_calibration/page_001/owner_p001_06c2c46ee27f',
  'style_colored_cards_calibration/page_001/owner_p001_0dc05817313b',
  'style_colored_cards_calibration/page_001/owner_p001_0f39ce3adba0',
  'style_cross_tile_holdout/page_002/owner_p002_43752a8f8c08',
  'style_cross_tile_holdout/page_002/owner_p002_7227e479da0d',
  'style_cross_tile_holdout/page_002/owner_p002_d9afb32ef935',
  'style_dark_panels_holdout/page_002/owner_p002_5a6976dbbf65',
  'style_dark_panels_holdout/page_003/owner_p003_98e72af0685a'
) | Sort-Object
$artifacts = @($inspectionTemplate.artifacts)
$actualPairs = @($artifacts | ForEach-Object { "$($_.entry_id)/$($_.page_id)/$($_.owner_id)" }) | Sort-Object
if ($artifacts.Count -ne 9 -or (Compare-Object $expectedPairs $actualPairs)) { throw 'Inspection manifest does not cover the exact nine owners once' }
foreach ($artifact in $artifacts) {
  $path = Join-Path "$r5RunRoot\matrix" $artifact.artifact_path
  if (-not (Test-Path -LiteralPath $path)) { throw "Missing inspection sheet: $path" }
  if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $artifact.sha256) { throw "Inspection sheet hash mismatch: $path" }
  Write-Output $path
}
```

Use `view_image` com detalhe original para cada uma das nove sheets. Inspecione os seis painéis autenticados:

```text
source | masks/evidence | requested/resolved | observed raster | final | safe geometry
```

Para cada owner, registre separadamente:

- **Functional:** source inglês removido; PT-BR completo; sem duplicação; sem corpo repartido/truncado; posição/centro/proporção legíveis; inpaint sem resíduo e sem dano de arte.
- **Style:** classe de fonte, peso/largura/slant, fill, stroke/glow/shadow/gradient, rotação, hierarquia e ocupação compatíveis; sem efeito decorativo falso.
- **Inspection:** dimensões nativas, bbox lógico e frame bbox corretos, geometry hash e SHA-256 autenticados.

Confirme especificamente:

- cards: nenhum inglês, corpo proporcional e efeitos coerentes entre cards relacionados;
- burst/white balloon: texto inteiro dentro do container e sem clip de frame;
- `owner_p002_d9afb32ef935`: bbox lógico `[109,985,318,1104]` vira frame bbox `[164,985,373,1104]`, payload inteiro e fonte compatível;
- dark/text-over-art: fonte/fill observados, safe geometry não cruza arte e nenhum inglês residual.

Não marque GO olhando thumbnail. Zoom nativo é obrigatório.

**Step 7: Completar e validar o inspection manifest**

Crie uma cópia de trabalho sem sobrescrever o template autenticado:

```powershell
$r5ActiveManifest = Join-Path (Resolve-Path '.') '.codex-tmp/style-copy-r5-active-run.json'
$r5Active = Get-Content -LiteralPath $r5ActiveManifest -Raw | ConvertFrom-Json
$r5RunRoot = [string]$r5Active.run_root
Copy-Item -LiteralPath "$r5RunRoot\matrix\style-inspection-template.json" -Destination "$r5RunRoot\matrix\style-inspection-v2.json" -ErrorAction Stop
```

Preencha `style-inspection-v2.json` com `apply_patch`, em todas as nove linhas, usando `functional_verdict`, `functional_note`, `style_verdict`, `style_note`, `inspection_verdict` e hashes/dimensões reais. Use apenas `GO` ou `NO-GO`; não deixe `PENDING`.

Depois execute:

```powershell
$env:TRADUZAI_MATRIX_MYTHIC_SOURCE = (Resolve-Path -LiteralPath '.codex-tmp/mythic_ch40_pages_1_2_source_20260724').Path
$env:TRADUZAI_MATRIX_ONE_SECOND_SOURCE = (Resolve-Path -LiteralPath '.codex-tmp/page_owner_systemic_sources_20260729/one_second').Path
$env:TRADUZAI_MATRIX_GRAND_FINALE_SOURCE = (Resolve-Path -LiteralPath '.codex-tmp/page_owner_systemic_sources_20260729/grand_finale').Path
$r5ActiveManifest = Join-Path (Resolve-Path '.') '.codex-tmp/style-copy-r5-active-run.json'
$r5Active = Get-Content -LiteralPath $r5ActiveManifest -Raw | ConvertFrom-Json
$r5RunRoot = [string]$r5Active.run_root
$matrixRunId = 'matrix-' + [string]$r5Active.attempt_id
pipeline/venv/Scripts/python.exe pipeline/tools/validate_owner_visual_matrix.py --manifest pipeline/tests/fixtures/style_copy_corpus/matrix.json --output-root "$r5RunRoot\matrix" --report "$r5RunRoot\matrix-report-inspected.md" --validate-only --inspection-manifest "$r5RunRoot\matrix\style-inspection-v2.json" --run-id $matrixRunId --acceptance-bundle $r5Active.acceptance_bundle_path
$matrixFinalExit = $LASTEXITCODE
Write-Output "matrix_final_exit=$matrixFinalExit"
if ($matrixFinalExit -ne 0) { throw "Inspected matrix failed with exit $matrixFinalExit" }
$holdoutSummary = Get-Content -LiteralPath "$r5RunRoot\matrix\style_holdout_summary.json" -Raw | ConvertFrom-Json
if ($holdoutSummary.status -ne 'PASS' -or $holdoutSummary.inspection_coverage -ne 1.0) { throw 'Inspected holdout summary is not complete PASS' }
```

Expected para GO: exit 0, 9/9 Functional GO, 9/9 Style GO, 9/9 Inspection GO. Os rates de aceitação usam somente os cinco holdouts; a ferramenta mantém um gate adicional mais forte exigindo também os quatro calibration targets inspecionados em GO.

**Step 8: Rodar o agregador final**

```powershell
$r5ActiveManifest = Join-Path (Resolve-Path '.') '.codex-tmp/style-copy-r5-active-run.json'
$r5Active = Get-Content -LiteralPath $r5ActiveManifest -Raw | ConvertFrom-Json
$r5RunRoot = [string]$r5Active.run_root
$benchmarkRunId = 'benchmark-' + [string]$r5Active.attempt_id
pipeline/venv/Scripts/python.exe pipeline/debug_tools/style_copy_r5_acceptance.py evaluate --acceptance-bundle $r5Active.acceptance_bundle_path --benchmark-summary "$r5RunRoot\benchmark\$benchmarkRunId\style_benchmark_summary.json" --owner-qa-summary "$r5RunRoot\matrix\style_owner_qa_summary.json" --inspection-summary "$r5RunRoot\matrix\style_holdout_summary.json" --matrix-execution-summary "$r5RunRoot\matrix\matrix_execution_summary.json" --thresholds pipeline/tests/fixtures/style_benchmark_v2/benchmark_spec.json --output "$r5RunRoot\style-copy-r5-acceptance.json" --mode enforce
$acceptanceExit = $LASTEXITCODE
Write-Output "acceptance_exit=$acceptanceExit"
if ($acceptanceExit -ne 0) { throw "Final R5 acceptance failed with exit $acceptanceExit" }
$acceptance = Get-Content -LiteralPath "$r5RunRoot\style-copy-r5-acceptance.json" -Raw | ConvertFrom-Json
if ($acceptance.status -ne 'PASS' -or $acceptance.verdicts.overall -ne 'GO') { throw 'Final R5 acceptance JSON is not PASS/GO' }
```

Expected para conclusão: exit 0 e `status=PASS`; verdicts Style/Functional/Inspection/Overall todos `GO`.

**Step 9: Loop obrigatório se houver NO-GO**

Não publique um patch individual por owner. Classifique a falha:

- decisão/grupo/normalização/raster/comparison → Tasks 1–8;
- corpo parcial, duplicação ou inglês → Task 12;
- container/cleanup/owner geometry → Task 10;
- crop/overlay/frame → Tasks 9 e 11;
- métrica sintética → Task 13;
- QA owner → Task 14;
- denominador/inspeção → Tasks 15–16.

Para cada falha: criar RED sistêmico sem ID específico, confirmar RED, corrigir, rodar checkpoint correspondente e repetir **Steps 2–8** para criar root e acceptance bundle novos; o bundle antigo não autentica o código corrigido. Reinspecione todas as sheets afetadas. Continue até GO ou até existir bloqueio externo real e demonstrável; três retries idênticos sem possibilidade de progresso são o mínimo para reportar bloqueio.

**Step 10: Publicar relatório e resumo honestos**

O Markdown deve conter:

- branch/HEAD e SHA-256 do diff R5;
- root externo, runtime, seeds, inputs/configs/tool hashes;
- suites e diferencial por nodeid;
- métricas/thresholds/denominadores;
- tabela dos nove owners com source/final/sheet hashes e três verdicts;
- contratos/gates, rollbacks, English residuals e fallback counts;
- correção logical-page→frame comprovada;
- declaração de zero regra específica;
- blockers restantes, se houver.

O JSON contém os mesmos dados em schema versionado. Imagens permanecem no root externo e são referenciadas por path+SHA-256.

Crie/edite os dois arquivos versionados com `apply_patch`; não copie summaries externos diretamente para o repositório sem selecionar e normalizar os campos do schema publicado.

**Step 11: Validar e commitar somente relatório/resumo**

```powershell
git diff --check -- docs/reports/2026-08-01-style-copy-r5-materialization-validation.md docs/reports/evidence/2026-08-01-style-copy-r5-materialization-summary.json
if ($LASTEXITCODE -ne 0) { throw 'R5 report working diff check failed' }
git add -- docs/reports/2026-08-01-style-copy-r5-materialization-validation.md docs/reports/evidence/2026-08-01-style-copy-r5-materialization-summary.json
if ($LASTEXITCODE -ne 0) { throw 'R5 report staging failed' }
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'R5 report staged diff check failed' }
$expectedReportPaths = @('docs/reports/2026-08-01-style-copy-r5-materialization-validation.md','docs/reports/evidence/2026-08-01-style-copy-r5-materialization-summary.json') | Sort-Object
$stagedReportPaths = @(git diff --cached --name-only) | Sort-Object
if ($LASTEXITCODE -ne 0 -or (Compare-Object $expectedReportPaths $stagedReportPaths)) { throw 'R5 report staged path set is incomplete or contaminated' }
git commit -m "test(style): validate r5 materialization on owner holdout"
if ($LASTEXITCODE -ne 0) { throw 'R5 report commit failed' }
```

**Step 12: Verificação antes de conclusão**

Use `@verification-before-completion` e confirme novamente:

```powershell
git status --short --branch
if ($LASTEXITCODE -ne 0) { throw 'Could not inspect final worktree status' }
$r5PlanAnchor = git log -1 --format=%H -- docs/plans/2026-08-01-style-copy-r5-materialization-geometry-remediation-implementation.md
if ($LASTEXITCODE -ne 0 -or -not $r5PlanAnchor) { throw 'Could not resolve committed R5 plan anchor' }
git diff --check "$r5PlanAnchor..HEAD"
if ($LASTEXITCODE -ne 0) { throw 'Final committed R5 diff check failed' }
$finalStaged = @(git diff --cached --name-only)
if ($LASTEXITCODE -ne 0 -or $finalStaged.Count -ne 0) { throw 'Final index is not empty' }
git log --oneline --decorate -25
if ($LASTEXITCODE -ne 0) { throw 'Could not inspect final R5 commit sequence' }
```

Somente declare o plano concluído quando:

- todos os testes/checkpoints obrigatórios foram executados no venv correto;
- zero nova falha ampla;
- benchmark/owner QA/inspection completos e autenticados;
- 9/9 owners entregam PT-BR completo e limpo;
- todos os gates relevantes passam;
- aceitação final retorna 0/PASS;
- inspeção nativa foi realmente realizada;
- mudanças locais preexistentes permanecem preservadas.

## Critério de parada

- Um teste RED que falha por motivo diferente do esperado deve ser corrigido antes da produção.
- Falha visual vira novo RED sistêmico e retorna à task proprietária; não vira exceção de página/owner.
- Missing metric/denominator/hash/inspection sempre bloqueia.
- Não reclassifique inglês preservado por rollback como sucesso funcional.
- Não reclassifique crop de inspeção incorreto como falha de renderer; corrija o coordinate contract e reinspecione.
- Não marque execução completa apenas porque o fail-closed funcionou. O resultado exigido é pixel visual aceitável e gate final GO.

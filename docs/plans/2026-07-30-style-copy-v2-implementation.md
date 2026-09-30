# Style Copy V2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Construir um copiador de estilo owner-scoped, auditável e confidence-gated que preserve fonte, proporção, peso, fill, stroke, glow, sombra, gradiente, rotação, slant, tracking e escala de SFX sem alterar conteúdo, ownership, máscara, inpaint ou layout funcional.

**Architecture:** A fonte visual é capturada antes do inpaint em um sidecar imutável por `owner_id`. O Style Engine V2 produz evidência e decisão independentes por atributo, resolve fonte pela forma dos glifos, aplica somente campos confiáveis após o layout funcional e rasteriza texto/SFX por um capability contract comum. QA de estilo pareia fonte, perfil e pixels finais por owner e só pode rodar depois do gate funcional.

**Tech Stack:** Python 3.12, dataclasses/typing, NumPy, OpenCV, PIL, matplotlib FT2Font, fontTools quando já disponível, pytest, JSON/JSONL, SHA-256 e métricas Delta E 2000/IoU/contorno.

---

## Regras obrigatórias de execução

- Use as skills `mangatl-dev`, `traduzai-typesetting`, `traduzai-ocr` e `traduzai-pipeline` conforme a tarefa.
- Execute este plano no checkout atual. Não use `reset`, `checkout`, `stash`, `clean`, `git add -A` nem crie worktree.
- Preserve todas as mudanças locais existentes. Antes de cada patch, inspecione `git status --short -- <paths>` e `git diff -- <paths>`.
- Aplique TDD estrito: teste RED, implementação mínima, GREEN, regressão próxima, diff auditado e commit restrito.
- Use `git add -p --` para todo arquivo preexistente e `git add --` apenas para arquivos novos criados integralmente pela task. Sempre revise `git diff --cached -- <paths>` antes do commit.
- Não faça `git merge codex/style-engine-s4` e não faça cherry-pick do range S4. As linhagens divergiram e isso importaria alterações alheias.
- Reutilize os commits S4 por patches de paths específicos, sempre testes primeiro. Adapte arquivos conflitantes manualmente com `apply_patch`.
- Inclua trailer `Adapted-from: <sha>` nos commits que transplantarem trabalho S4.
- Nenhuma regra pode depender de obra, capítulo, página, texto literal, coordenada fixa ou `band_id`.
- `OwnerVisualProfileV2` é sidecar. Nunca o adicione à identidade/hash semântico de `TextOwner`.
- Estilo não pode alterar OCR, tradução, payload, owner count, rota, mask plan, inpaint, safe region ou resultado do gate funcional.
- Confiança ausente nunca autoriza cópia. Cada atributo usa apenas sua própria confiança.
- Candidato SFX bruto/review-only não ativa scan nem aplicação; exige promoção e confiança.
- Não baixe fontes no runtime. Somente fontes locais com licença/proveniência auditável entram no catálogo.
- Goldens de estilo congelam texto correto. `DING -> VING`, inglês residual ou tradução parcial continuam sendo erros funcionais.
- Não habilite aplicação runtime antes de o plano funcional estar estável e com gate não bloqueado. Tasks 1–6 podem rodar em shadow mode antes disso; Tasks 7–16 exigem o checkpoint funcional final.
- Use diretórios novos sob `.codex-tmp` para matrizes e auditorias. O runner S4 de benchmark é a exceção: ele exige um `output-root` externo ao repositório, sob o temp do sistema. Nunca modifique baseline tracked durante medição.
- Inspeção visual em tamanho nativo é obrigatória; score automático isolado não conclui GO.

## Baseline S4 e estratégia de transplante

O design aprovado está no commit `976141bd` e em `docs/plans/2026-07-30-style-copy-v2-design.md`.

Branches/worktrees auditados no planejamento:

- `codex/style-engine-benchmark-v2`, HEAD `227926e9`, limpo;
- `codex/style-engine-s4`, HEAD `63d5013c`, limpo;
- S4 contém benchmark-v2 mais máscaras/evidência mascarada;
- testes observados: 28 PASS em benchmark-v2 e 5 PASS nas duas tasks S4 finais.

Commits-fonte imutáveis:

1. `305c5959154924e44bdcf325d1614cf9c65fceb1` — baseline reproduzível;
2. `51cf57bed65c2ec93abfa8f77ab082f291c22fd0` — benchmark sintético V2;
3. `321f1ee531fb1f4560c989617c2ccb70c7d174d9` — `StyleEvidenceV2` shadow;
4. `227926e9a480de1589670562b8ce7631af3d331d` — ranking calibrado de fontes;
5. `73e134f1480cd5f8999966d2cd9e715c7ed57cad` — máscaras tipográficas;
6. `63d5013ca7381b9fab3358091724868ec5bc8462` — evidência de fill/stroke mascarada.

Os commits não são ancestors nem patch-equivalentes do branch atual. Os patches `305c5959`, `321f1ee5` e `227926e9` conflitam conceitualmente com melhorias atuais; preserve `OWNER_STYLE_FORBIDDEN_FIELDS`, `AUTO_VISUAL_STYLE_FIELDS`, sanitização/deep-copy, instrumentação e APIs page-owner atuais.

## Checkpoints contínuos

1. **Checkpoint S1 — shadow baseline:** Tasks 0–6. Contratos S4 reproduzidos no branch atual sem alterar pixels runtime.
2. **Checkpoint S2 — owner evidence:** Tasks 7–10. Elegibilidade por campo, sidecar, extração real e matcher tipográfico.
3. **Checkpoint S3 — render:** Tasks 11–13. Consistência de grupo, rasterizador único e capability handshake.
4. **Checkpoint S4 — audit e rollout:** Tasks 14–16. QA por owner, shadow/canary, holdout e inspeção visual.

### Task 0: Preflight imutável e verificação dos commits-fonte

**Files:** None.

**Step 1: Auditar checkout e paths de estilo**

```powershell
git status --short --branch
git diff --check
git diff --cached --check

$stylePaths = @(
  "fonts/font-map.json",
  "pipeline/typesetter",
  "pipeline/sfx",
  "pipeline/debug_tools",
  "pipeline/main.py",
  "pipeline/strip/process_bands.py",
  "pipeline/ownership/translation.py",
  "pipeline/qa",
  "pipeline/tools/validate_owner_visual_matrix.py",
  "pipeline/tests/test_style*",
  "pipeline/tests/test_font*",
  "pipeline/tests/test_sfx*",
  "pipeline/tests/test_owner*",
  "pipeline/tests/test_strip*"
)

git status --short -- $stylePaths
git diff --stat -- $stylePaths
git diff --cached --stat -- $stylePaths
```

Expected: inventariar alterações tracked/untracked sem abortar por diretório amplo. Antes de cada task, repita `git status --short -- <paths da task>` e leia os diffs file/hunk-scoped. Só pare se um mesmo hunk não puder ser preservado e adaptado com segurança; não reverta o usuário.

**Step 2: Fixar os objetos git**

```powershell
$sourceCommits = @(
  "305c5959154924e44bdcf325d1614cf9c65fceb1",
  "51cf57bed65c2ec93abfa8f77ab082f291c22fd0",
  "321f1ee531fb1f4560c989617c2ccb70c7d174d9",
  "227926e9a480de1589670562b8ce7631af3d331d",
  "73e134f1480cd5f8999966d2cd9e715c7ed57cad",
  "63d5013ca7381b9fab3358091724868ec5bc8462"
)
$sourceCommits | ForEach-Object { git cat-file -e "$_^{commit}" }
if ($LASTEXITCODE -ne 0) { throw "Missing immutable S4 source commit" }
```

Expected: os seis commits existem localmente. Não use o nome móvel do branch durante o transplante.

**Step 3: Registrar baseline atual sem commit**

```powershell
New-Item -ItemType Directory -Force .codex-tmp | Out-Null
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_style_copy_benchmark.py `
  pipeline/tests/test_style_copy_score.py `
  pipeline/tests/test_style_audit_report.py `
  pipeline/tests/test_style_extractor.py `
  pipeline/tests/test_font_detector.py `
  pipeline/tests/test_sfx_style.py `
  pipeline/tests/test_sfx_renderer.py `
  -q | Tee-Object .codex-tmp\style_copy_v2_baseline_20260730.txt
```

Expected: baseline registrado; nenhuma edição/commit nesta task.

### Task 1: Transplantar o baseline reproduzível de `305c5959`

**Files:**

- Create: `pipeline/debug_tools/run_style_copy_baseline.py`
- Create: `pipeline/debug_tools/style_copy_benchmark_runtime.lock.json`
- Create: `pipeline/tests/test_style_copy_baseline_runner.py`
- Modify: `pipeline/debug_tools/run_style_copy_regression.py`
- Modify: `pipeline/debug_tools/style_copy_score.py`
- Modify: `pipeline/tests/test_style_copy_benchmark.py`
- Modify: `pipeline/tests/test_style_copy_score.py`

**Step 1: Aplicar somente os testes do commit-fonte**

```powershell
git show --format= --binary 305c5959154924e44bdcf325d1614cf9c65fceb1 -- `
  pipeline/tests/test_style_copy_baseline_runner.py `
  pipeline/tests/test_style_copy_benchmark.py `
  pipeline/tests/test_style_copy_score.py
```

Expected: inspecionar o patch sem alterar o checkout. Reproduza somente os hunks de teste com `apply_patch`, adaptando imports ao branch atual e sem tocar produção.

**Step 2: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_style_copy_baseline_runner.py `
  pipeline/tests/test_style_copy_benchmark.py `
  pipeline/tests/test_style_copy_score.py -q
```

Expected: FAIL por runner/lock/contrato reproduzível ausente.

**Step 3: Adaptar implementação sem substituir arquivos atuais**

Use `git show <sha>:<path>` apenas como referência e `apply_patch` para portar:

- baseline não sobrescrevível;
- hashes dos artefatos;
- runtime lock;
- output isolado fora da fixture;
- CLI reproduzível;
- score sem regenerar o atlas tracked.

Preserve toda lógica atual adicional em runner, score e auditor.

**Step 4: Confirmar GREEN**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_style_copy_baseline_runner.py `
  pipeline/tests/test_style_copy_benchmark.py `
  pipeline/tests/test_style_copy_score.py -q
```

Expected: PASS.

**Step 5: Commit adaptado**

```powershell
git add -- pipeline/debug_tools/run_style_copy_baseline.py pipeline/debug_tools/style_copy_benchmark_runtime.lock.json pipeline/tests/test_style_copy_baseline_runner.py
git add -p -- pipeline/debug_tools/run_style_copy_regression.py pipeline/debug_tools/style_copy_score.py pipeline/tests/test_style_copy_benchmark.py pipeline/tests/test_style_copy_score.py
git diff --cached --check
git commit -m "test(style): port reproducible style benchmark baseline" -m "Adapted-from: 305c5959154924e44bdcf325d1614cf9c65fceb1"
```

### Task 2: Transplantar o benchmark sintético V2 de `51cf57be`

**Files:**

- Create: `pipeline/debug_tools/generate_style_benchmark_v2.py`
- Create: `pipeline/debug_tools/run_style_benchmark_v2.py`
- Create: `pipeline/debug_tools/style_benchmark_report.py`
- Create: `pipeline/tests/fixtures/style_benchmark_v2/benchmark_spec.json`
- Create: `pipeline/tests/test_style_benchmark_v2_generator.py`
- Create: `pipeline/tests/test_style_benchmark_v2_score.py`

**Step 1: Aplicar fixture e testes primeiro**

```powershell
git show --format= --binary 51cf57bed65c2ec93abfa8f77ab082f291c22fd0 -- `
  pipeline/tests/fixtures/style_benchmark_v2/benchmark_spec.json `
  pipeline/tests/test_style_benchmark_v2_generator.py `
  pipeline/tests/test_style_benchmark_v2_score.py
```

Reproduza esses arquivos/hunks com `apply_patch`.

**Step 2: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_style_benchmark_v2_generator.py `
  pipeline/tests/test_style_benchmark_v2_score.py -q
```

Expected: FAIL por módulos de geração/runner/report ausentes.

**Step 3: Aplicar somente os novos módulos**

```powershell
git show --format= --binary 51cf57bed65c2ec93abfa8f77ab082f291c22fd0 -- `
  pipeline/debug_tools/generate_style_benchmark_v2.py `
  pipeline/debug_tools/run_style_benchmark_v2.py `
  pipeline/debug_tools/style_benchmark_report.py
```

Reproduza somente esses módulos com `apply_patch`; não aplique o commit inteiro.

**Step 4: Confirmar GREEN e isolamento**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_style_benchmark_v2_generator.py `
  pipeline/tests/test_style_benchmark_v2_score.py -q
```

Expected: PASS; o atlas sintético é bootstrap, não meta final de qualidade.

**Step 5: Commit**

```powershell
git add -- pipeline/debug_tools/generate_style_benchmark_v2.py pipeline/debug_tools/run_style_benchmark_v2.py pipeline/debug_tools/style_benchmark_report.py pipeline/tests/fixtures/style_benchmark_v2/benchmark_spec.json pipeline/tests/test_style_benchmark_v2_generator.py pipeline/tests/test_style_benchmark_v2_score.py
git diff --cached --check
git commit -m "test(style): port isolated synthetic style benchmark v2" -m "Adapted-from: 51cf57bed65c2ec93abfa8f77ab082f291c22fd0"
```

### Task 3: Adaptar o contrato shadow `StyleEvidenceV2`

**Files:**

- Create: `pipeline/typesetter/style_contract.py`
- Create: `pipeline/tests/test_style_contract.py`
- Create: `pipeline/tests/test_style_policy_v2.py`
- Modify: `pipeline/typesetter/style_policy.py`
- Modify: `pipeline/debug_tools/run_style_benchmark_v2.py`
- Modify: `pipeline/debug_tools/style_benchmark_report.py`
- Modify: `pipeline/debug_tools/style_audit_report.py`
- Modify: `pipeline/tests/test_style_audit_report.py`
- Modify: `pipeline/tests/test_style_benchmark_v2_score.py`

**Step 1: Aplicar testes do commit-fonte**

```powershell
git show --format= --binary 321f1ee531fb1f4560c989617c2ccb70c7d174d9 -- `
  pipeline/tests/test_style_contract.py `
  pipeline/tests/test_style_policy_v2.py `
  pipeline/tests/test_style_audit_report.py `
  pipeline/tests/test_style_benchmark_v2_score.py
```

Reproduza os hunks de teste com `apply_patch`, preservando expectativas atuais não relacionadas.

**Step 2: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_style_contract.py `
  pipeline/tests/test_style_policy_v2.py `
  pipeline/tests/test_style_audit_report.py `
  pipeline/tests/test_style_benchmark_v2_score.py -q
```

Expected: FAIL por `StyleEvidenceV2`/policy shadow ausentes.

**Step 3: Implementar o contrato tipado**

Porte/adapte:

```python
@dataclass(frozen=True)
class StyleAttributeEvidenceV2:
    value: Any
    confidence: float
    top_k: tuple[Any, ...] = ()
    margin: float = 0.0
    abstention_reason: str = ""


@dataclass(frozen=True)
class StyleEvidenceV2:
    source: str
    text_present: bool
    attributes: dict[str, StyleAttributeEvidenceV2]
    schema_version: int = 2
```

Porte o contrato S4 exatamente nesta task para manter os testes-fonte compatíveis: `source`, `text_present`, `attributes` e `schema_version=2`. Adicione `style_evidence_v2_from_v1()` e `style_evidence_v2_shadow_policy()`, mantendo `apply_to_renderer=False`. Provenance, hashes e campos opcionais entram de modo retrocompatível somente na Task 7, com migração testada.

Em `style_policy.py`, preserve integralmente `OWNER_STYLE_FORBIDDEN_FIELDS`, `AUTO_VISUAL_STYLE_FIELDS`, `source_style_copy_allowed()` e a sanitização/deep-copy atuais.

**Step 4: Adaptar benchmark/auditor e confirmar GREEN**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_style_contract.py `
  pipeline/tests/test_style_policy_v2.py `
  pipeline/tests/test_style_audit_report.py `
  pipeline/tests/test_style_benchmark_v2_score.py -q
```

Expected: PASS e nenhuma mudança de pixel runtime.

**Step 5: Commit**

```powershell
git add -- pipeline/typesetter/style_contract.py pipeline/tests/test_style_contract.py pipeline/tests/test_style_policy_v2.py
git add -p -- pipeline/typesetter/style_policy.py pipeline/debug_tools/run_style_benchmark_v2.py pipeline/debug_tools/style_benchmark_report.py pipeline/debug_tools/style_audit_report.py pipeline/tests/test_style_audit_report.py pipeline/tests/test_style_benchmark_v2_score.py
git diff --cached --check
git commit -m "feat(style): port shadow StyleEvidence v2 contract" -m "Adapted-from: 321f1ee531fb1f4560c989617c2ccb70c7d174d9"
```

### Task 4: Adaptar o ranking calibrado de fontes

**Files:**

- Create: `pipeline/tests/test_font_detector_benchmark.py`
- Modify: `pipeline/typesetter/font_detector.py`
- Modify: `pipeline/typesetter/style_contract.py`
- Modify: `pipeline/tests/test_style_contract.py`

**Step 1: Aplicar os testes do commit-fonte**

```powershell
git show --format= --binary 227926e9a480de1589670562b8ce7631af3d331d -- `
  pipeline/tests/test_font_detector_benchmark.py `
  pipeline/tests/test_style_contract.py
```

Reproduza os testes com `apply_patch`.

**Step 2: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_font_detector_benchmark.py `
  pipeline/tests/test_font_detector.py `
  pipeline/tests/test_style_contract.py -q
```

Expected: FAIL por `detect_with_evidence()`/ranking ausente.

**Step 3: Incorporar ranking sem substituir o detector atual**

Adicione `FINGERPRINT_SAMPLES`, normalização de região, fingerprints multi-sample, `_rank_candidates()` e `detect_with_evidence()`. O retorno deve incluir top-k, score, margin, classificação `exact|family|unknown` e abstention reason.

Mantenha `detect_with_score()` e toda instrumentação atual.

**Step 4: Confirmar GREEN**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_font_detector_benchmark.py `
  pipeline/tests/test_font_detector.py `
  pipeline/tests/test_style_contract.py -q
```

Expected: PASS.

**Step 5: Commit**

```powershell
git add -- pipeline/tests/test_font_detector_benchmark.py
git add -p -- pipeline/typesetter/font_detector.py pipeline/typesetter/style_contract.py pipeline/tests/test_style_contract.py
git diff --cached --check
git commit -m "feat(style): port calibrated font ranking in shadow mode" -m "Adapted-from: 227926e9a480de1589670562b8ce7631af3d331d"
```

### Task 5: Transplantar a decomposição tipográfica mascarada

**Files:**

- Create: `pipeline/typesetter/style_masks.py`
- Create: `pipeline/tests/test_style_mask_layers.py`

**Step 1: Aplicar o teste primeiro**

```powershell
git show --format= --binary 73e134f1480cd5f8999966d2cd9e715c7ed57cad -- pipeline/tests/test_style_mask_layers.py
```

Crie o teste com `apply_patch`.

**Step 2: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_style_mask_layers.py -q
```

Expected: FAIL por `typesetter.style_masks` ausente.

**Step 3: Aplicar o módulo e confirmar GREEN**

```powershell
git show --format= --binary 73e134f1480cd5f8999966d2cd9e715c7ed57cad -- pipeline/typesetter/style_masks.py
```

Crie o módulo com `apply_patch` e então rode:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_style_mask_layers.py -q
```

Expected: `3 passed`.

**Step 4: Commit**

```powershell
git add -- pipeline/typesetter/style_masks.py pipeline/tests/test_style_mask_layers.py
git diff --cached --check
git commit -m "feat(style): port shadow typographic mask decomposition" -m "Adapted-from: 73e134f1480cd5f8999966d2cd9e715c7ed57cad"
```

### Task 6: Transplantar evidência mascarada de fill e stroke

**Files:**

- Create: `pipeline/typesetter/style_mask_evidence.py`
- Create: `pipeline/tests/test_style_mask_evidence.py`

**Step 1: Aplicar o teste primeiro e confirmar RED**

```powershell
git show --format= --binary 63d5013ca7381b9fab3358091724868ec5bc8462 -- pipeline/tests/test_style_mask_evidence.py
```

Crie o teste com `apply_patch` e então rode:

```powershell
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_style_mask_evidence.py -q
```

Expected: FAIL por `typesetter.style_mask_evidence` ausente.

**Step 2: Aplicar o módulo e confirmar GREEN**

```powershell
git show --format= --binary 63d5013ca7381b9fab3358091724868ec5bc8462 -- pipeline/typesetter/style_mask_evidence.py
```

Crie o módulo com `apply_patch` e então rode:

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_style_mask_layers.py `
  pipeline/tests/test_style_mask_evidence.py -q
```

Expected: `5 passed` no conjunto das Tasks 5–6.

**Step 3: Commit**

```powershell
git add -- pipeline/typesetter/style_mask_evidence.py pipeline/tests/test_style_mask_evidence.py
git diff --cached --check
git commit -m "feat(style): port mask backed fill and stroke evidence" -m "Adapted-from: 63d5013ca7381b9fab3358091724868ec5bc8462"
```

### Task 7: Transformar shadow evidence em decisão por campo

**Precondition:** checkpoint funcional final fresco e verificável. Antes de editar:

```powershell
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$functionalRoot = Join-Path $PWD ".codex-tmp\style_v2_functional_preflight_$stamp"
$functionalReport = Join-Path $functionalRoot "functional-preflight.md"
$env:TRADUZAI_MATRIX_MYTHIC_SOURCE = (Resolve-Path '.codex-tmp\mythic_ch40_pages_1_2_source_20260724').Path
$env:TRADUZAI_MATRIX_ONE_SECOND_SOURCE = (Resolve-Path '.codex-tmp\page_owner_systemic_sources_20260729\one_second').Path
$env:TRADUZAI_MATRIX_GRAND_FINALE_SOURCE = (Resolve-Path '.codex-tmp\page_owner_systemic_sources_20260729\grand_finale').Path
pipeline\venv\Scripts\python.exe pipeline/tools/validate_owner_visual_matrix.py `
  --manifest pipeline/tests/fixtures/owner_visual_matrix/functional_matrix.json `
  --output-root $functionalRoot `
  --report $functionalReport
if ($LASTEXITCODE -ne 0) { throw "Functional gate is not ready for Style Copy runtime" }
```

Expected: owner graph verificado, todos os `export_gate.json` não bloqueados e relatório funcional do HEAD atual sem blocker. Se falhar, mantenha Tasks 1–6 em shadow e não avance.

**Files:**

- Modify: `pipeline/typesetter/style_contract.py`
- Modify: `pipeline/typesetter/style_policy.py`
- Modify: `pipeline/main.py:9953-10350`
- Modify: `pipeline/debug_tools/style_audit_report.py`
- Test: `pipeline/tests/test_style_contract.py`
- Test: `pipeline/tests/test_style_policy_v2.py`
- Test: `pipeline/tests/test_main_emit.py`
- Test: `pipeline/tests/test_style_audit_report.py`

**Step 1: Escrever testes RED**

```python
def test_missing_candidate_confidence_never_applies_style(): ...
def test_high_glow_confidence_cannot_authorize_low_confidence_fill(): ...
def test_raw_sfx_candidate_never_triggers_v2_style_scan(): ...
def test_review_required_candidate_abstains(): ...
def test_each_attribute_requires_its_own_confidence(): ...
def test_low_confidence_evidence_is_preserved_but_not_applied(): ...
def test_shadow_v2_contract_migrates_without_losing_source_or_text_present(): ...
```

**Step 2: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_style_policy_v2.py `
  pipeline/tests/test_style_contract.py `
  pipeline/tests/test_main_emit.py -k "style and (confidence or evidence or sfx)" -q
```

Expected: a maior confiança global pode autorizar atributos fracos ou status explícito não existe.

**Step 3: Implementar decisão tipada**

Adicione:

```python
StyleCopyStatus = Literal["applied", "fallback", "not_applicable", "review_required"]

@dataclass(frozen=True)
class StyleApplicationDecisionV2:
    status: StyleCopyStatus
    applied_attributes: Mapping[str, Any]
    abstained_attributes: Mapping[str, str]
    evidence_sha256: str
```

Estenda `StyleEvidenceV2` retrocompativelmente com `source_sha256` e provenance opcional por atributo, sempre com defaults que preservem `source`, `text_present` e serialização dos testes S4.

Implemente `style_candidate_copy_allowed()`, `evaluate_style_attribute()` e `decide_style_copy_v2()`. Mova a política real para `style_policy.py`; `main.py` mantém wrappers temporários que apenas delegam.

Preserve os gates compartilhados: OCR principal `0.70`; SFX promovido `0.66`. Não duplique os números no auditor.

**Step 4: Confirmar GREEN e espelhamento do auditor**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_style_policy_v2.py `
  pipeline/tests/test_style_contract.py `
  pipeline/tests/test_main_emit.py -k "style or sfx" -q
pipeline\venv\Scripts\python.exe -m pytest pipeline/tests/test_style_audit_report.py -q
```

Expected: PASS.

**Step 5: Commit**

```powershell
git add -p -- pipeline/typesetter/style_contract.py pipeline/typesetter/style_policy.py pipeline/main.py pipeline/debug_tools/style_audit_report.py pipeline/tests/test_style_contract.py pipeline/tests/test_style_policy_v2.py pipeline/tests/test_main_emit.py pipeline/tests/test_style_audit_report.py
git diff --cached --check
git commit -m "feat(style): gate style copy per candidate and attribute"
```

### Task 8: Criar o sidecar visual page-owner antes do inpaint

**Files:**

- Create: `pipeline/typesetter/owner_style.py`
- Create: `pipeline/tests/test_owner_style_profile.py`
- Modify: `pipeline/strip/process_bands.py:9953-10450`
- Modify: `pipeline/typesetter/renderer.py:6519-6685`
- Test: `pipeline/tests/test_owner_layout.py`
- Test: `pipeline/tests/test_strip_process_bands.py`
- Test: `pipeline/tests/test_owner_translation.py`

**Step 1: Escrever testes RED**

```python
def test_every_renderable_owner_receives_explicit_style_status(): ...
def test_owner_profile_is_captured_before_inpaint_mutates_pixels(): ...
def test_owner_profile_round_trips_to_renderer(): ...
def test_style_profile_cannot_change_owner_semantic_signature(): ...
def test_divergent_profiles_for_same_owner_are_rejected(): ...
def test_tile_permutation_does_not_change_owner_profile(): ...
def test_translation_page_never_serializes_visual_profile_fields(): ...
```

**Step 2: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_owner_style_profile.py `
  pipeline/tests/test_owner_layout.py -k "style or visual_profile" -q
```

Expected: owners atuais chegam ao renderer sem perfil/status V2.

**Step 3: Implementar sidecar imutável**

Crie `build_owner_visual_profiles()`, `build_owner_visual_profile()`, `owner_visual_profile_sha256()` e `attach_owner_visual_profile()`.

Em `execute_owner_page_graph()`:

1. capture perfis de todos os owners a partir dos pixels-fonte, componentes, observações selecionadas e glyph masks;
2. faça isso antes do primeiro inpaint;
3. mantenha `TextOwner` exclusivamente semântico;
4. anexe `visual_profile_v2`, hash e status ao record/runtime sidecar;
5. aplique o perfil somente depois de resolver layout.

`_owner_visual_profile()` aceita somente o contrato V2 normalizado; não reinterpreta `style_evidence` cru.

Inclua `visual_profile_v2`, `visual_profile_sha256` e status em `_owner_record_contract()` antes da deduplicação de records. Valide o hash antes de escolher o record; perfis divergentes para o mesmo owner bloqueiam em vez de “primeiro vence”. `owners_to_translation_page()` permanece intocado e o teste negativo garante que o sidecar nunca invade a serialização semântica.

**Step 4: Confirmar invariância e GREEN**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_owner_style_profile.py `
  pipeline/tests/test_owner_layout.py -k "style or visual_profile" `
  pipeline/tests/test_strip_process_bands.py -k "owner and style" `
  pipeline/tests/test_owner_translation.py -k "visual_profile" -q
```

Expected: PASS; owner/payload/route/mask hashes idênticos com ou sem sidecar.

**Step 5: Commit**

```powershell
git add -- pipeline/typesetter/owner_style.py pipeline/tests/test_owner_style_profile.py
git add -p -- pipeline/strip/process_bands.py pipeline/typesetter/renderer.py pipeline/tests/test_owner_layout.py pipeline/tests/test_strip_process_bands.py pipeline/tests/test_owner_translation.py
git diff --cached --check
git commit -m "feat(style): propagate owner visual profiles from source pixels"
```

### Task 9: Construir corpus owner-paired e extração mascarada real

**Files:**

- Create: `pipeline/tests/fixtures/style_copy_corpus/manifest.json`
- Create: `pipeline/tests/test_style_copy_corpus.py`
- Modify: `pipeline/typesetter/style_masks.py`
- Modify: `pipeline/typesetter/style_mask_evidence.py`
- Modify: `pipeline/typesetter/style_extractor.py:44-620`
- Test: `pipeline/tests/test_style_mask_layers.py`
- Test: `pipeline/tests/test_style_mask_evidence.py`
- Test: `pipeline/tests/test_style_extractor.py`

**Step 1: Criar manifest e testes RED antes dos assets derivados**

O manifest registra por caso: `case_id`, `split=calibration|holdout`, categoria, work/chapter somente como provenance de fixture, `owner_id`, source image, glyph mask, frozen source text, semantic role e expectativas por campo.

```python
def test_corpus_separates_calibration_and_holdout_by_work(): ...
def test_every_case_pairs_source_mask_and_owner_id(): ...
def test_colored_background_does_not_contaminate_fill(): ...
def test_plain_balloon_abstains_from_false_stroke_and_glow(): ...
def test_card_crop_preserves_condensed_white_evidence(): ...
def test_burst_crop_preserves_colored_effect_evidence(): ...
def test_metrics_are_normalized_by_source_x_height(): ...
```

**Step 2: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_style_copy_corpus.py `
  pipeline/tests/test_style_extractor.py `
  pipeline/tests/test_style_mask_evidence.py -q
```

Expected: corpus/API V2 ausentes.

**Step 3: Gerar fixtures reproduzíveis e implementar extração**

Use scripts/artefatos locais para gerar crops e máscaras; não edite PNGs manualmente. Implemente:

```python
def extract_text_style_evidence_v2(
    image_rgb: np.ndarray,
    glyph_mask: np.ndarray,
    context_mask: np.ndarray,
    *,
    owner_id: str,
    semantic_role: str,
) -> StyleEvidenceV2:
    ...
```

Meça fill, stroke/cor/espessura, glow, sombra, gradiente, rotação, slant, largura, peso, tracking, caixa, x-height e ocupação. Normalize dimensões pelo x-height. Gere abstention por campo quando máscara/evidência for insuficiente.

**Step 4: Confirmar GREEN e negativos**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_style_copy_corpus.py `
  pipeline/tests/test_style_mask_layers.py `
  pipeline/tests/test_style_mask_evidence.py `
  pipeline/tests/test_style_extractor.py -q
```

Expected: PASS e nenhum falso efeito no hard-negative simples.

**Step 5: Commit**

```powershell
git add -- pipeline/tests/fixtures/style_copy_corpus pipeline/tests/test_style_copy_corpus.py
git add -p -- pipeline/typesetter/style_masks.py pipeline/typesetter/style_mask_evidence.py pipeline/typesetter/style_extractor.py pipeline/tests/test_style_mask_layers.py pipeline/tests/test_style_mask_evidence.py pipeline/tests/test_style_extractor.py
git diff --cached --check
git commit -m "feat(style): extract mask backed source typography"
```

### Task 10: Implementar matcher tipográfico pela forma renderizada

**Files:**

- Create: `pipeline/typesetter/font_matcher.py`
- Create: `pipeline/tests/test_font_matcher.py`
- Modify: `pipeline/typesetter/font_detector.py`
- Modify: `fonts/font-map.json`
- Test: `pipeline/tests/test_font_detector_benchmark.py`

**Step 1: Escrever testes RED**

```python
def test_source_text_rerender_ranks_exact_font_first(): ...
def test_matcher_abstains_when_top_two_margin_is_ambiguous(): ...
def test_matcher_uses_frozen_source_text_not_translation(): ...
def test_matcher_handles_condensed_and_expanded_width(): ...
def test_font_catalog_contains_only_existing_licensed_files(): ...
def test_matcher_cache_key_includes_catalog_and_profile_hashes(): ...
```

**Step 2: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_font_matcher.py `
  pipeline/tests/test_font_detector_benchmark.py `
  pipeline/tests/test_font_detector.py -q
```

Expected: matcher real ausente.

**Step 3: Implementar shortlist + comparação de máscara**

O detector produz shortlist. Para cada fonte local elegível:

- renderize o texto-fonte congelado;
- normalize escala, rotação, slant e largura;
- calcule silhouette IoU e Chamfer/contorno normalizado;
- combine com prior de semantic role;
- exija score absoluto e margin sobre top-2;
- abstain quando ambíguo;
- cache por catálogo, texto e profile hash.

Amplie `font-map.json` somente com arquivos existentes/licenciados e tags de role/context. Não baixe nada.

**Step 4: Confirmar GREEN**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_font_matcher.py `
  pipeline/tests/test_font_detector_benchmark.py `
  pipeline/tests/test_font_detector.py -q
```

Expected: PASS.

**Step 5: Commit**

```powershell
git add -- pipeline/typesetter/font_matcher.py pipeline/tests/test_font_matcher.py
git add -p -- pipeline/typesetter/font_detector.py fonts/font-map.json pipeline/tests/test_font_detector_benchmark.py
git diff --cached --check
git commit -m "feat(style): match source fonts by rendered glyph shape"
```

### Task 11: Resolver consistência por grupo visual

**Files:**

- Create: `pipeline/typesetter/style_groups.py`
- Create: `pipeline/tests/test_style_groups.py`
- Modify: `pipeline/typesetter/owner_style.py`
- Modify: `pipeline/strip/process_bands.py`
- Test: `pipeline/tests/test_owner_style_profile.py`
- Test: `pipeline/tests/test_owner_layout.py`

**Step 1: Escrever testes RED**

```python
def test_card_rows_share_font_and_effect_class(): ...
def test_same_balloon_cannot_mix_unrelated_styles(): ...
def test_group_resolution_preserves_owner_count_and_payloads(): ...
def test_low_confidence_member_inherits_only_group_safe_fields(): ...
def test_title_body_and_value_keep_role_hierarchy(): ...
```

**Step 2: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_style_groups.py `
  pipeline/tests/test_owner_style_profile.py `
  pipeline/tests/test_owner_layout.py -k "group or style" -q
```

Expected: não existe resolução de grupo V2.

**Step 3: Implementar resolução conservadora**

Agrupe linhas/roles do mesmo card, fragmentos visuais do mesmo balloon e owners do mesmo burst/panel antes do loop de execução. Compartilhe família e classe de efeito quando evidência concordar; preserve tamanho/peso/hierarquia por role. Nunca funda payloads/owners.

**Step 4: Confirmar GREEN e invariância semântica**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_style_groups.py `
  pipeline/tests/test_owner_style_profile.py `
  pipeline/tests/test_owner_layout.py -k "group or style" -q
```

Expected: PASS; owner count e payload hashes inalterados.

**Step 5: Commit**

```powershell
git add -- pipeline/typesetter/style_groups.py pipeline/tests/test_style_groups.py
git add -p -- pipeline/typesetter/owner_style.py pipeline/strip/process_bands.py pipeline/tests/test_owner_style_profile.py pipeline/tests/test_owner_layout.py
git diff --cached --check
git commit -m "feat(style): resolve consistent contextual style groups"
```

### Task 12: Criar rasterizador único para texto e SFX

**Files:**

- Create: `pipeline/typesetter/glyph_rasterizer.py`
- Create: `pipeline/tests/test_glyph_rasterizer.py`
- Modify: `pipeline/typesetter/renderer.py`
- Modify: `pipeline/sfx/style.py:15-150`
- Modify: `pipeline/sfx/renderer.py:49-220`
- Test: `pipeline/tests/test_typesetting_renderer.py`
- Test: `pipeline/tests/test_sfx_style.py`
- Test: `pipeline/tests/test_sfx_renderer.py`

**Step 1: Escrever testes RED**

```python
def test_renderer_applies_v2_tracking_slant_and_width(): ...
def test_effect_pixels_scale_with_x_height(): ...
def test_latin_sfx_uses_selected_project_font_not_hershey(): ...
def test_sfx_applies_scale_x_and_scale_y(): ...
def test_renderer_applies_multistroke_shadow_glow_and_gradient_layers(): ...
def test_renderer_never_draws_outside_owner_safe_polygon(): ...
def test_style_effect_envelope_never_clips_core_or_high_confidence_effects(): ...
def test_unsafe_effect_abstains_without_reflow_or_payload_change(): ...
```

**Step 2: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_glyph_rasterizer.py `
  pipeline/tests/test_sfx_style.py `
  pipeline/tests/test_sfx_renderer.py `
  pipeline/tests/test_typesetting_renderer.py -k "style or sfx or owner" -q
```

Expected: tracking/slant/scale/effects reais ausentes ou Hershey substitui a fonte.

**Step 3: Implementar composição em camadas**

O rasterizador recebe glyph mask e decisões V2 e aplica, nesta ordem, transform de width/slant/tracking, stroke externo/múltiplo, shadow, glow, fill/gradient e composição dentro da safe mask. Antes de compor, calcule `glyph_core_envelope` e `effect_envelope` pós-transformação contra a mesma safe polygon do contrato funcional.

Exija zero core clipping e zero clipping para todo efeito que a decisão pretende aplicar. Quando um atributo ampliar o envelope além da safe region, abstenha/fallback somente esse atributo e revalide; não faça reflow, não mude font size funcional, payload, line breaks ou layout. Se nem o core couber, devolva review ao contrato funcional. Texto e SFX usam a mesma API; remova o override `FONT_HERSHEY_TRIPLEX` para SFX latino.

**Step 4: Confirmar GREEN e goldens**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_glyph_rasterizer.py `
  pipeline/tests/test_sfx_style.py `
  pipeline/tests/test_sfx_renderer.py `
  pipeline/tests/test_typesetting_renderer.py -k "style or sfx or owner" -q
```

Expected: PASS e nenhuma escrita fora da safe polygon.

**Step 5: Commit**

```powershell
git add -- pipeline/typesetter/glyph_rasterizer.py pipeline/tests/test_glyph_rasterizer.py
git add -p -- pipeline/typesetter/renderer.py pipeline/sfx/style.py pipeline/sfx/renderer.py pipeline/tests/test_typesetting_renderer.py pipeline/tests/test_sfx_style.py pipeline/tests/test_sfx_renderer.py
git diff --cached --check
git commit -m "feat(style): rasterize v2 typography for text and sfx"
```

### Task 13: Impedir perda silenciosa de efeitos entre backends

**Files:**

- Modify: `pipeline/typesetter/backend_contract.py`
- Modify: `pipeline/typesetter/rust_backend.py`
- Modify: `pipeline/typesetter/renderer.py`
- Test: `pipeline/tests/test_typesetting_backend_contract.py`
- Test: `pipeline/tests/test_renderer_backend_parity.py`
- Test: `pipeline/tests/test_typesetting_rust_backend.py`

**Step 1: Escrever testes RED**

```python
def test_backend_declares_v2_style_capabilities(): ...
def test_profile_with_unsupported_glow_falls_back_to_python(): ...
def test_backend_selection_never_silently_drops_effect(): ...
def test_supported_solid_profile_keeps_python_rust_parity(): ...
```

**Step 2: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_typesetting_backend_contract.py `
  pipeline/tests/test_renderer_backend_parity.py `
  pipeline/tests/test_typesetting_rust_backend.py -q
```

Expected: backend atual não negocia todos os efeitos V2.

**Step 3: Implementar capability handshake**

Defina capabilities explícitas por backend. Antes de renderizar, compare atributos aplicados do perfil com as capabilities. Se faltar um recurso, selecione Python e registre motivo; se nenhum backend puder satisfazer, marque review. Nunca retire o campo silenciosamente.

**Step 4: Confirmar GREEN**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_typesetting_backend_contract.py `
  pipeline/tests/test_renderer_backend_parity.py `
  pipeline/tests/test_typesetting_rust_backend.py -q
```

Expected: PASS.

**Step 5: Commit**

```powershell
git add -p -- pipeline/typesetter/backend_contract.py pipeline/typesetter/rust_backend.py pipeline/typesetter/renderer.py pipeline/tests/test_typesetting_backend_contract.py pipeline/tests/test_renderer_backend_parity.py pipeline/tests/test_typesetting_rust_backend.py
git diff --cached --check
git commit -m "fix(style): preserve v2 effects across renderer backends"
```

### Task 14: Criar auditoria source-to-final por owner

**Files:**

- Create: `pipeline/qa/style_fidelity.py`
- Create: `pipeline/tests/test_style_fidelity_qa.py`
- Modify: `pipeline/debug_tools/style_audit_report.py`
- Modify: `pipeline/debug_tools/style_copy_score.py`
- Modify: `pipeline/debug_tools/run_style_copy_regression.py`
- Modify: `pipeline/main.py`
- Test: `pipeline/tests/test_style_audit_report.py`
- Test: `pipeline/tests/test_style_copy_score.py`

**Step 1: Escrever testes RED**

```python
def test_png_original_is_resolved_from_project_record(): ...
def test_audit_pairs_source_and_final_by_owner_id(): ...
def test_unrelated_record_on_same_page_cannot_satisfy_expectation(): ...
def test_fallback_owner_is_not_high_confidence_failure(): ...
def test_high_confidence_style_mismatch_blocks_enforce_mode(): ...
def test_style_gate_cannot_override_blocked_functional_gate(): ...
```

**Step 2: Confirmar RED**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_style_fidelity_qa.py `
  pipeline/tests/test_style_audit_report.py `
  pipeline/tests/test_style_copy_score.py -q
```

Expected: audit atual não pareia toda a cadeia por owner/hash.

**Step 3: Implementar pareamento e métricas**

Pareie source path real (PNG/JPG), glyph mask, `StyleEvidenceV2`, decisão, render plan e final pixels por `owner_id` e hashes. Reporte por campo `not_scanned`, `fallback`, `applied`, `mismatch` ou `not_applicable`.

Calcule font rank, Delta E 2000, stroke thickness, glow/shadow radius/offset, rotation, width/slant, x-height e safe containment. Gere sheet `source | masks/evidence | expected | final`.

**Step 4: Implementar modos de rollout**

- `shadow`: relatório, nenhuma mudança de pixel/gate;
- `render`: aplica perfil, mismatch ainda não bloqueia;
- `enforce`: bloqueia apenas mismatch de alta confiança/catastrófico.

Gate funcional roda primeiro e não pode ser substituído por score de estilo.

**Step 5: Confirmar GREEN**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_style_fidelity_qa.py `
  pipeline/tests/test_style_audit_report.py `
  pipeline/tests/test_style_copy_score.py -q
```

Expected: PASS.

**Step 6: Commit**

```powershell
git add -- pipeline/qa/style_fidelity.py pipeline/tests/test_style_fidelity_qa.py
git add -p -- pipeline/debug_tools/style_audit_report.py pipeline/debug_tools/style_copy_score.py pipeline/debug_tools/run_style_copy_regression.py pipeline/main.py pipeline/tests/test_style_audit_report.py pipeline/tests/test_style_copy_score.py
git diff --cached --check
git commit -m "feat(qa): audit source to final style fidelity by owner"
```

### Task 15: Rodar suites, benchmark e canary sem habilitar gate prematuramente

**Files:** No production edits expected.

**Step 1: Rodar suite shadow/S4**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_style_copy_baseline_runner.py `
  pipeline/tests/test_style_benchmark_v2_generator.py `
  pipeline/tests/test_style_benchmark_v2_score.py `
  pipeline/tests/test_style_contract.py `
  pipeline/tests/test_style_policy_v2.py `
  pipeline/tests/test_font_detector_benchmark.py `
  pipeline/tests/test_style_mask_layers.py `
  pipeline/tests/test_style_mask_evidence.py -q
if ($LASTEXITCODE -ne 0) { throw "S4 shadow suite failed" }
```

Expected: zero failures e nenhuma mudança de pixel runtime no modo shadow.

**Step 2: Rodar suite Style Copy V2 completa**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_style_copy_corpus.py `
  pipeline/tests/test_owner_style_profile.py `
  pipeline/tests/test_style_extractor.py `
  pipeline/tests/test_font_matcher.py `
  pipeline/tests/test_style_groups.py `
  pipeline/tests/test_glyph_rasterizer.py `
  pipeline/tests/test_sfx_style.py `
  pipeline/tests/test_sfx_renderer.py `
  pipeline/tests/test_style_fidelity_qa.py `
  pipeline/tests/test_style_audit_report.py `
  pipeline/tests/test_style_copy_score.py -q
if ($LASTEXITCODE -ne 0) { throw "Style Copy V2 suite failed" }
```

Expected: zero failures.

**Step 3: Rodar invariância funcional e suite ampla**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_owner_graph.py `
  pipeline/tests/test_owner_reconcile_properties.py `
  pipeline/tests/test_owner_mask.py `
  pipeline/tests/test_owner_atomic_execution.py `
  pipeline/tests/test_owner_layout.py `
  pipeline/tests/test_final_pixel_export_gate.py -q
if ($LASTEXITCODE -ne 0) { throw "Style changed functional contracts" }

pipeline\venv\Scripts\python.exe -m pytest pipeline/tests -q --tb=short
if ($LASTEXITCODE -ne 0) { throw "Broad pipeline suite failed or diverged from baseline" }
```

Expected: zero nova regressão funcional.

**Step 4: Executar benchmark em diretório novo**

```powershell
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$benchOutput = Join-Path ([System.IO.Path]::GetTempPath()) "traduzai-style-benchmarks"
$runId = "style_copy_v2_$stamp"
pipeline\venv\Scripts\python.exe pipeline/debug_tools/run_style_benchmark_v2.py `
  --output-root $benchOutput `
  --run-id $runId `
  --seed 1729 `
  --level all
if ($LASTEXITCODE -ne 0) { throw "Style benchmark failed" }
```

Expected: baseline tracked permanece imutável; output externo ao repositório contém hashes, seed/run ID e split calibration/holdout.

### Task 16: Executar matriz holdout, inspecionar e publicar veredito

**Files:**

- Modify: `pipeline/tools/validate_owner_visual_matrix.py`
- Modify: `pipeline/tests/test_owner_visual_matrix_tool.py`
- Modify: `pipeline/tests/fixtures/style_benchmark_v2/benchmark_spec.json`
- Create: `pipeline/tests/fixtures/style_copy_corpus/matrix.json`
- Create: `docs/reports/2026-07-30-style-copy-v2-validation.md`
- Create: `docs/reports/evidence/2026-07-30-style-copy-v2-inspection.json`

**Step 1: Escrever testes RED da matriz**

```python
def test_style_matrix_requires_owner_paired_source_and_final(): ...
def test_style_matrix_manifest_uses_validator_entries_schema(): ...
def test_style_matrix_separates_calibration_and_holdout_works(): ...
def test_style_matrix_reports_explicit_fallbacks(): ...
def test_style_matrix_cannot_pass_when_functional_gate_is_blocked(): ...
def test_style_matrix_enforces_category_thresholds(): ...
```

**Step 2: Confirmar RED e implementar categoria `style_fidelity`**

```powershell
pipeline\venv\Scripts\python.exe -m pytest `
  pipeline/tests/test_owner_visual_matrix_tool.py `
  pipeline/tests/test_style_copy_corpus.py `
  pipeline/tests/test_style_fidelity_qa.py -q
```

Crie `style_copy_corpus/matrix.json` no schema real do validator, com `entries[]`, `entry_id`, `work_id`, `chapter_id`, `config_path`, `input_key`, `work_dir` de output e `categories`. Reuse o resolver/hash versionado de `pipeline/tests/fixtures/owner_visual_matrix/inputs.json`; nenhum source path absoluto entra no manifest. O `manifest.json` da Task 9 continua sendo o corpus owner/crop e nunca é passado diretamente ao validator.

Expected após implementação: PASS.

**Step 3: Executar matriz holdout nova**

```powershell
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$styleRoot = Join-Path $PWD ".codex-tmp\style_copy_v2_validation_$stamp"
$styleReport = Join-Path $PWD "docs\reports\2026-07-30-style-copy-v2-validation.md"
$inspectionTemplate = Join-Path $styleRoot "style-inspection-template.json"
$env:TRADUZAI_MATRIX_MYTHIC_SOURCE = (Resolve-Path '.codex-tmp\mythic_ch40_pages_1_2_source_20260724').Path
$env:TRADUZAI_MATRIX_ONE_SECOND_SOURCE = (Resolve-Path '.codex-tmp\page_owner_systemic_sources_20260729\one_second').Path
$env:TRADUZAI_MATRIX_GRAND_FINALE_SOURCE = (Resolve-Path '.codex-tmp\page_owner_systemic_sources_20260729\grand_finale').Path

pipeline\venv\Scripts\python.exe pipeline/tools/validate_owner_visual_matrix.py `
  --manifest pipeline/tests/fixtures/style_copy_corpus/matrix.json `
  --output-root $styleRoot `
  --report $styleReport `
  --inspection-template $inspectionTemplate

if ($LASTEXITCODE -ne 0) { throw "Style holdout matrix blocked" }
```

Expected: categorias speech, colored card, dark panel, burst, SFX, text-over-art, group e hard-negative em obras holdout.

**Step 4: Verificar critérios automáticos**

- 100% dos owners renderizáveis com status explícito;
- zero `style_origin=legacy` silencioso;
- zero mudança de payload/owner/route/mask causada pelo estilo;
- font top-1 >= 90%, top-3 >= 98%, nenhuma categoria abaixo de 85%;
- fill Delta E 2000 mediana <= 8 e p95 <= 12;
- stroke Delta E 2000 <= 10 e espessura <= 20% do x-height ou 1 px;
- glow/shadow Delta E 2000 <= 12 e raio/offset <= 25% ou 2 px;
- rotação <= 3 graus;
- width/slant/occupancy <= 15%;
- x-height final/source entre 0.85 e 1.15 quando o container permitir;
- safe containment 100%;
- falsos efeitos em balloons simples <= 2%;
- GO por owner >= 90%, speech >= 95%, nenhuma categoria abaixo de 85%;
- zero mismatch catastrófico de alta confiança.

**Step 5: Inspecionar visualmente em escala nativa**

Para cada owner selecionado, compare source, glyph/core/stroke/context masks, evidência/decisões, inpaint, final e safe region. Confirme especialmente:

- `DING` mantém tipografia/efeito expressivos com texto funcional correto;
- cards mantêm família condensada, fill/stroke/glow e hierarquia coerentes;
- bursts preservam inclinação, cor e efeito;
- balloons comuns não ganham glow/stroke falso;
- um mesmo balloon/card não mistura estilos incompatíveis;
- nenhuma melhoria de estilo encobre inglês residual, tradução parcial, duplicação ou inpaint ruim.

Se qualquer critério falhar, publique `NO-GO` com owner/categoria/artefato; não ajuste uma página individualmente.

**Step 6: Registrar e validar a inspeção visual**

Crie `docs/reports/evidence/2026-07-30-style-copy-v2-inspection.json` com `apply_patch`, preenchendo para cada item do template: path, SHA-256, `scale="native"`, owner/categoria, timestamp, `GO|NO-GO` e nota. Depois valide os mesmos bytes:

```powershell
$styleInspection = Join-Path $PWD "docs\reports\evidence\2026-07-30-style-copy-v2-inspection.json"
pipeline\venv\Scripts\python.exe pipeline/tools/validate_owner_visual_matrix.py `
  --manifest pipeline/tests/fixtures/style_copy_corpus/matrix.json `
  --output-root $styleRoot `
  --report $styleReport `
  --inspection-manifest $styleInspection `
  --validate-only
if ($LASTEXITCODE -ne 0) { throw "Style pixels or inspection evidence failed validation" }
```

**Step 7: Commit do tooling e relatório somente após validação honesta**

```powershell
git add -- pipeline/tests/fixtures/style_copy_corpus/matrix.json docs/reports/2026-07-30-style-copy-v2-validation.md docs/reports/evidence/2026-07-30-style-copy-v2-inspection.json
git add -p -- pipeline/tools/validate_owner_visual_matrix.py pipeline/tests/test_owner_visual_matrix_tool.py pipeline/tests/fixtures/style_benchmark_v2/benchmark_spec.json
git diff --cached --check
git commit -m "test(style): validate style copy v2 across held out works"
```

**Step 8: Auditoria final**

```powershell
git status --short --branch
git log --oneline -25
git diff --check
```

Expected: mudanças locais preexistentes preservadas; nenhum output de `.codex-tmp` ou `DEBUGM` staged.

## Definição final de pronto

- Os seis contratos S4 foram reaproveitados com provenance e testes RED/GREEN.
- Todo owner renderizável possui perfil/status V2 explícito.
- Evidência e decisão são independentes por atributo.
- Confiança ausente ou SFX não promovido nunca aplica estilo.
- Perfil é capturado antes do inpaint e aplicado somente após layout funcional.
- Sidecar não altera nenhuma assinatura semântica ou máscara.
- Extração usa máscaras e abstém diante de contaminação.
- Matcher usa forma do texto-fonte e catálogo local licenciado.
- Grupos mantêm coerência sem fundir owners/payloads.
- Texto e SFX usam o mesmo rasterizador/capabilities.
- Nenhum backend descarta efeitos silenciosamente.
- QA pareia source-to-final por owner/hash e respeita gate funcional.
- Suites, benchmark, holdout e inspeção nativa satisfazem os limites aprovados.

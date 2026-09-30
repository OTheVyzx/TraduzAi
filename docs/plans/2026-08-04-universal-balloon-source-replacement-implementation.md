# Universal Balloon Source Replacement Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task.

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
- `pipeline/tests/test_owner_source_replacement_fail_closed.py` já existe como arquivo local não rastreado. Trate-o como trabalho do usuário: execute read-only e traduza requisitos para o novo teste plan-owned da Task 10, sem editar/stagear o original.
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

### Protocolo executável de worktree sujo e staging sem TTY

Nenhum `git add -p` é permitido durante a execução. Antes da Task 1, use `apply_patch` para criar o helper privado `.codex-tmp/universal-source-replacement-plan-guard/TaskOwnedGit.ps1` com o conteúdo abaixo; ele nunca entra no Git. `Initialize` cria `dirty_overlap_ledger.json` imutável e o primeiro checkpoint hash-linked numa raiz monotônica, cobrindo status, conteúdo e diffs de todo path já sujo que cruza qualquer whitelist, e executa um repositório-probe privado que prova tanto a profundidade `-p2` quanto a rejeição de overwrite de um path local protegido. Somente labels `Create`, `Modify` e `Test` entram na whitelist por Task; `Read only`/`Re-run only` entram apenas na proteção global do ledger, e `Do not track` é sempre metadado excluído de ambos. `Start` exige igualdade exata com o último checkpoint e copia o estado **atual** dos paths da Task para a árvore `a/` antes de editar. `Stage` prova que todos os dirty paths fora da whitelist atual continuam idênticos, copia o pós-edição para `b/`, gera com `git diff --no-index --binary a b` um patch que contém somente o delta da Task e o aplica ao índice vazio com `git apply --cached -p2`; o duplo prefixo real é `a/a/<repo-path>`/`b/b/<repo-path>`. Se o autoteste falhar, um path protegido mudar ou o delta depender de contexto sujo ausente em `HEAD`, o apply falha fechado. `VerifyAfterCommit` repete a prova, impede absorver dirty state da própria Task, verifica árvore/parent e só então avança o checkpoint append-only. Não apague/reutilize snapshots antigos.

```powershell
param(
  [Parameter(Mandatory = $true)]
  [ValidateSet('Initialize', 'Start', 'Stage', 'VerifyAfterCommit')]
  [string]$Action,
  [int]$Task = 0,
  [string]$PlanPath = 'docs/plans/2026-08-04-universal-balloon-source-replacement-implementation.md'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
$repoRoot = (& git rev-parse --show-toplevel).Trim()
if ($LASTEXITCODE -ne 0) { throw 'not inside the expected Git checkout' }
$resolvedPlan = (Resolve-Path -LiteralPath (Join-Path $repoRoot $PlanPath)).Path
$guardBase = Join-Path $repoRoot '.codex-tmp\universal-source-replacement-plan-guard'
$currentContextPath = Join-Path $guardBase 'current.json'

function Get-TextSha256([string]$Value) {
  $bytes = [Text.Encoding]::UTF8.GetBytes($Value)
  $sha = [Security.Cryptography.SHA256]::Create()
  try {
    $digest = $sha.ComputeHash($bytes)
  } finally {
    $sha.Dispose()
  }
  return ([BitConverter]::ToString($digest) -replace '-', '').ToLowerInvariant()
}

function Save-AtomicJson([object]$Value, [string]$Path) {
  $parent = Split-Path -Parent $Path
  New-Item -ItemType Directory -Force -Path $parent | Out-Null
  $temporary = "$Path.tmp.$([guid]::NewGuid().ToString('N'))"
  [IO.File]::WriteAllText($temporary, ($Value | ConvertTo-Json -Depth 32), $utf8NoBom)
  if (Test-Path -LiteralPath $Path -PathType Leaf) {
    $backup = "$Path.replaced.$([guid]::NewGuid().ToString('N'))"
    [IO.File]::Replace($temporary, $Path, $backup)
  } else {
    [IO.File]::Move($temporary, $Path)
  }
}

function Get-TaskPaths([int]$TaskNumber, [switch]$AllTasks) {
  $text = Get-Content -LiteralPath $resolvedPlan -Raw -Encoding utf8
  $blocks = @(
    if ($AllTasks) {
      [regex]::Matches($text, '(?ms)^### Task \d+:.*?(?=^### Task \d+:|\z)')
    } else {
      [regex]::Matches($text, "(?ms)^### Task ${TaskNumber}:.*?(?=^### Task \d+:|\z)")
    }
  )
  if ($blocks.Count -eq 0) { throw "Task $TaskNumber not found in plan" }
  $paths = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
  foreach ($block in $blocks) {
    $files = $block.Value
    $step = $files.IndexOf('**Step 1:')
    if ($step -ge 0) { $files = $files.Substring(0, $step) }
    foreach ($match in [regex]::Matches($files, '(?m)^- (?<label>[^:]+): `(?<path>[^`]+)`')) {
      $label = $match.Groups['label'].Value
      $taskOwned = $label.StartsWith('Create') -or $label.StartsWith('Modify') -or $label.StartsWith('Test')
      $globalReadOnly = $label.StartsWith('Read only') -or $label.StartsWith('Re-run only')
      if ($AllTasks) {
        if (-not ($taskOwned -or $globalReadOnly)) { continue }
      } elseif (-not $taskOwned) {
        continue
      }
      $relative = $match.Groups['path'].Value -replace ':\d.*$', ''
      [void]$paths.Add($relative.Replace('\', '/'))
    }
  }
  return @($paths | Sort-Object)
}

function Get-DirtyPathState([string]$Relative, [string]$RepositoryRoot = $repoRoot) {
  $absolute = Join-Path $RepositoryRoot $Relative
  $unstaged = (& git -C $RepositoryRoot diff --binary -- $Relative | Out-String)
  $staged = (& git -C $RepositoryRoot diff --cached --binary -- $Relative | Out-String)
  $unstagedPatchId = (& git -C $RepositoryRoot diff --binary -- $Relative | & git patch-id --stable | Out-String).Trim()
  return [ordered]@{
    path = $Relative
    status = @(& git -C $RepositoryRoot status --short -- $Relative)
    worktree_file_sha256 = if (Test-Path -LiteralPath $absolute -PathType Leaf) {
      (Get-FileHash -Algorithm SHA256 -LiteralPath $absolute).Hash.ToLowerInvariant()
    } else { $null }
    unstaged_diff_sha256 = Get-TextSha256 $unstaged
    unstaged_patch_id = if ($unstagedPatchId) { ($unstagedPatchId -split '\s+')[0].ToLowerInvariant() } else { $null }
    staged_diff_sha256 = Get-TextSha256 $staged
  }
}

function Assert-DirtyStatesExplainable(
  [object[]]$ExpectedStates,
  [string[]]$ExcludedPaths = @(),
  [string]$RepositoryRoot = $repoRoot
) {
  foreach ($expected in $ExpectedStates) {
    $relative = [string]$expected.path
    if ($relative -in $ExcludedPaths) { continue }
    $current = Get-DirtyPathState -Relative $relative -RepositoryRoot $RepositoryRoot
    if ((@($current.status) -join "`0") -ne (@($expected.status) -join "`0")) {
      throw "dirty status changed outside current Task ownership: $relative"
    }
    foreach ($field in @('worktree_file_sha256', 'unstaged_diff_sha256', 'unstaged_patch_id', 'staged_diff_sha256')) {
      if ([string]$current[$field] -ne [string]$expected.$field) {
        throw "dirty state field $field changed outside current Task ownership: $relative"
      }
    }
  }
}

function Copy-ExistingPaths([string[]]$Paths, [string]$DestinationRoot) {
  New-Item -ItemType Directory -Force -Path $DestinationRoot | Out-Null
  foreach ($relative in $Paths) {
    $source = Join-Path $repoRoot $relative
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { continue }
    $destination = Join-Path $DestinationRoot $relative
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination
  }
}

function Assert-PatchDepthSelfTest([string]$RunRoot) {
  $probeRoot = Join-Path $RunRoot 'patch-depth-self-test'
  $probeRepo = Join-Path $probeRoot 'repo'
  $aFile = Join-Path $probeRoot 'a\owned\probe.txt'
  $bFile = Join-Path $probeRoot 'b\owned\probe.txt'
  New-Item -ItemType Directory -Path (Split-Path -Parent $aFile) | Out-Null
  New-Item -ItemType Directory -Path (Split-Path -Parent $bFile) | Out-Null
  New-Item -ItemType Directory -Path $probeRepo | Out-Null
  [IO.File]::WriteAllText($aFile, "before`n", $utf8NoBom)
  [IO.File]::WriteAllText($bFile, "after`n", $utf8NoBom)
  & git init --quiet -- $probeRepo
  if ($LASTEXITCODE -ne 0) { throw 'failed to initialize private patch-depth probe' }
  $baselineBlob = (& git -C $probeRepo hash-object -w -- $aFile).Trim()
  if ($LASTEXITCODE -ne 0) { throw 'failed to hash patch-depth baseline' }
  & git -C $probeRepo update-index --add --cacheinfo "100644,$baselineBlob,owned/probe.txt"
  if ($LASTEXITCODE -ne 0) { throw 'failed to seed patch-depth probe index' }
  $probePatch = Join-Path $probeRoot 'probe.patch'
  $probeError = Join-Path $probeRoot 'probe.stderr.txt'
  $gitExe = (Get-Command git).Source
  $process = Start-Process -FilePath $gitExe -ArgumentList @('diff', '--no-index', '--binary', '--', 'a', 'b') `
    -WorkingDirectory $probeRoot -RedirectStandardOutput $probePatch -RedirectStandardError $probeError `
    -WindowStyle Hidden -Wait -PassThru
  if ($process.ExitCode -ne 1) { throw "patch-depth diff did not produce one delta: $($process.ExitCode)" }
  & git -C $probeRepo apply --cached -p2 --check -- $probePatch
  if ($LASTEXITCODE -ne 0) { throw 'git apply -p2 self-check failed before apply' }
  & git -C $probeRepo apply --cached -p2 -- $probePatch
  if ($LASTEXITCODE -ne 0) { throw 'git apply -p2 self-check failed during apply' }
  $probePaths = @(& git -C $probeRepo ls-files)
  if ($probePaths.Count -ne 1 -or $probePaths[0] -ne 'owned/probe.txt') {
    throw "patch-depth self-check resolved the wrong index path: $($probePaths -join ', ')"
  }
  $expectedBlob = (& git -C $probeRepo hash-object -- $bFile).Trim()
  $stagedEntry = (& git -C $probeRepo ls-files --stage -- 'owned/probe.txt').Trim()
  $actualBlob = ($stagedEntry -split '\s+')[1]
  if ($actualBlob -ne $expectedBlob) { throw 'patch-depth self-check staged bytes differ from snapshot b' }
  $protectedPath = Join-Path $probeRepo 'protected\user.txt'
  New-Item -ItemType Directory -Path (Split-Path -Parent $protectedPath) | Out-Null
  [IO.File]::WriteAllText($protectedPath, "user-local`n", $utf8NoBom)
  $protectedState = Get-DirtyPathState -Relative 'protected/user.txt' -RepositoryRoot $probeRepo
  [IO.File]::WriteAllText($protectedPath, "overwritten`n", $utf8NoBom)
  $ledgerBlockedOverwrite = $false
  try {
    Assert-DirtyStatesExplainable -ExpectedStates @($protectedState) -RepositoryRoot $probeRepo
  } catch {
    $ledgerBlockedOverwrite = $true
  }
  if (-not $ledgerBlockedOverwrite) { throw 'dirty-ledger self-check accepted an overwritten untracked path' }
  Save-AtomicJson ([ordered]@{
    patch_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $probePatch).Hash.ToLowerInvariant()
    relative_path = $probePaths[0]
    staged_blob = $actualBlob
    dirty_ledger_overwrite_blocked = $ledgerBlockedOverwrite
    status = 'PASS'
  }) (Join-Path $probeRoot 'self-test-receipt.json')
}

function Read-CurrentContext {
  if (-not (Test-Path -LiteralPath $currentContextPath -PathType Leaf)) {
    throw 'plan guard is not initialized'
  }
  return Get-Content -LiteralPath $currentContextPath -Raw -Encoding utf8 | ConvertFrom-Json
}

if ($Action -eq 'Initialize') {
  & git diff --cached --quiet
  if ($LASTEXITCODE -ne 0) { throw 'index must be empty before guard initialization' }
  $guardedPlanPaths = @(Get-TaskPaths -TaskNumber 0 -AllTasks)
  $planText = Get-Content -LiteralPath $resolvedPlan -Raw -Encoding utf8
  $doNotTrackPaths = @(
    foreach ($match in [regex]::Matches($planText, '(?m)^- Do not track: `(?<path>[^`]+)`')) {
      ($match.Groups['path'].Value -replace ':\d.*$', '').Replace('\', '/')
    }
  )
  if (@($guardedPlanPaths | Where-Object { $_ -in $doNotTrackPaths }).Count -ne 0) {
    throw 'private Do not track evidence paths must never enter the dirty ledger or a Task whitelist'
  }
  New-Item -ItemType Directory -Force -Path $guardBase | Out-Null
  $runId = "guard-$((Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssfffffffZ'))-$([guid]::NewGuid().ToString('N'))"
  $runRoot = Join-Path $guardBase $runId
  New-Item -ItemType Directory -Path $runRoot | Out-Null
  Assert-PatchDepthSelfTest $runRoot
  & git diff --cached --quiet
  if ($LASTEXITCODE -ne 0) { throw 'patch-depth self-check changed the real repository index' }
  $entries = @()
  foreach ($relative in $guardedPlanPaths) {
    $state = Get-DirtyPathState -Relative $relative
    if (@($state.status).Count -eq 0) { continue }
    $entries += $state
  }
  $ledger = [ordered]@{
    schema_version = 1
    run_id = $runId
    branch = (& git branch --show-current).Trim()
    head = (& git rev-parse HEAD).Trim()
    plan_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $resolvedPlan).Hash.ToLowerInvariant()
    captured_at = (Get-Date).ToUniversalTime().ToString('o')
    dirty_entries = $entries
  }
  $ledgerPath = Join-Path $runRoot 'dirty_overlap_ledger.json'
  Save-AtomicJson $ledger $ledgerPath
  $ledgerSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $ledgerPath).Hash.ToLowerInvariant()
  $checkpointPath = Join-Path $runRoot 'dirty-checkpoint-000.json'
  Save-AtomicJson ([ordered]@{
    schema_version = 1
    run_id = $runId
    sequence = 0
    parent_checkpoint_sha256 = $null
    dirty_overlap_ledger_sha256 = $ledgerSha256
    states = $entries
  }) $checkpointPath
  $checkpointSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $checkpointPath).Hash.ToLowerInvariant()
  Save-AtomicJson ([ordered]@{
    run_id = $runId
    run_root = $runRoot
    dirty_overlap_ledger_sha256 = $ledgerSha256
    dirty_checkpoint_path = $checkpointPath
    dirty_checkpoint_sha256 = $checkpointSha256
  }) $currentContextPath
  return
}

$context = Read-CurrentContext
$runRoot = [string]$context.run_root
$expectedRunRoot = Join-Path $guardBase ([string]$context.run_id)
if ([IO.Path]::GetFullPath($runRoot) -ne [IO.Path]::GetFullPath($expectedRunRoot)) {
  throw 'guard context points outside its immutable run directory'
}
$ledgerPath = Join-Path $runRoot 'dirty_overlap_ledger.json'
if (-not (Test-Path -LiteralPath $ledgerPath -PathType Leaf)) { throw 'dirty overlap ledger is missing' }
$ledger = Get-Content -LiteralPath $ledgerPath -Raw -Encoding utf8 | ConvertFrom-Json
if ([string]$ledger.run_id -ne [string]$context.run_id) { throw 'guard context and ledger run IDs differ' }
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $ledgerPath).Hash.ToLowerInvariant() -ne [string]$context.dirty_overlap_ledger_sha256) {
  throw 'dirty overlap ledger changed after initialization'
}
if ([string]$ledger.branch -ne (& git branch --show-current).Trim()) { throw 'guard branch changed during execution' }
$currentPlanSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $resolvedPlan).Hash.ToLowerInvariant()
if ([string]$ledger.plan_sha256 -ne $currentPlanSha256) {
  throw 'implementation plan changed after guard initialization'
}
$checkpointPath = [string]$context.dirty_checkpoint_path
$runRootFull = [IO.Path]::GetFullPath($runRoot).TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)
$checkpointFull = [IO.Path]::GetFullPath($checkpointPath)
$runPrefix = $runRootFull + [IO.Path]::DirectorySeparatorChar
if (-not $checkpointFull.StartsWith($runPrefix, [StringComparison]::OrdinalIgnoreCase)) {
  throw 'dirty checkpoint points outside the immutable guard run'
}
if (-not (Test-Path -LiteralPath $checkpointPath -PathType Leaf)) { throw 'dirty checkpoint is missing' }
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $checkpointPath).Hash.ToLowerInvariant() -ne [string]$context.dirty_checkpoint_sha256) {
  throw 'dirty checkpoint changed after it became current'
}
$checkpoint = Get-Content -LiteralPath $checkpointPath -Raw -Encoding utf8 | ConvertFrom-Json
if ([string]$checkpoint.run_id -ne [string]$context.run_id) { throw 'dirty checkpoint run ID differs from guard context' }
if ([string]$checkpoint.dirty_overlap_ledger_sha256 -ne [string]$context.dirty_overlap_ledger_sha256) {
  throw 'dirty checkpoint is not bound to the immutable overlap ledger'
}
if ($Task -lt 1) { throw 'Task must be positive for this action' }
$taskPaths = @(Get-TaskPaths -TaskNumber $Task)
$taskPointer = Join-Path $runRoot ("task-{0:D2}-current.json" -f $Task)

if ($Action -eq 'Start') {
  & git diff --cached --quiet
  if ($LASTEXITCODE -ne 0) { throw 'index must be empty at Task start' }
  Assert-DirtyStatesExplainable -ExpectedStates @($checkpoint.states)
  $attemptId = "attempt-$((Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssfffffffZ'))-$([guid]::NewGuid().ToString('N'))"
  $attemptRoot = Join-Path $runRoot (("task-{0:D2}" -f $Task) + "-$attemptId")
  New-Item -ItemType Directory -Path $attemptRoot | Out-Null
  Copy-ExistingPaths $taskPaths (Join-Path $attemptRoot 'a')
  $baseline = foreach ($relative in $taskPaths) {
    $absolute = Join-Path $repoRoot $relative
    $state = Get-DirtyPathState -Relative $relative
    $state['exists'] = Test-Path -LiteralPath $absolute -PathType Leaf
    $state
  }
  Save-AtomicJson ([ordered]@{
    task = $Task
    attempt_root = $attemptRoot
    paths = $taskPaths
    baseline = $baseline
    dirty_checkpoint_path = $checkpointPath
    dirty_checkpoint_sha256 = [string]$context.dirty_checkpoint_sha256
  }) $taskPointer
  return
}

$taskContext = Get-Content -LiteralPath $taskPointer -Raw -Encoding utf8 | ConvertFrom-Json
$attemptRoot = [string]$taskContext.attempt_root
if ([string]$taskContext.dirty_checkpoint_sha256 -ne [string]$context.dirty_checkpoint_sha256) {
  throw 'a different dirty-state checkpoint became current after Task start'
}

if ($Action -eq 'Stage') {
  & git diff --cached --quiet
  if ($LASTEXITCODE -ne 0) { throw 'index must be empty before plan-owned staging' }
  Assert-DirtyStatesExplainable -ExpectedStates @($checkpoint.states) -ExcludedPaths $taskPaths
  $bRoot = Join-Path $attemptRoot 'b'
  if (Test-Path -LiteralPath $bRoot) { throw 'refusing to reuse an existing Task post-edit snapshot' }
  Copy-ExistingPaths $taskPaths $bRoot
  $patchPath = Join-Path $attemptRoot 'task-owned.patch'
  $errorPath = Join-Path $attemptRoot 'git-diff.stderr.txt'
  $gitExe = (Get-Command git).Source
  $process = Start-Process -FilePath $gitExe -ArgumentList @('diff', '--no-index', '--binary', '--', 'a', 'b') `
    -WorkingDirectory $attemptRoot -RedirectStandardOutput $patchPath -RedirectStandardError $errorPath `
    -WindowStyle Hidden -Wait -PassThru
  if ($process.ExitCode -notin @(0, 1)) { throw "git diff --no-index failed: $($process.ExitCode)" }
  if (-not (Test-Path -LiteralPath $patchPath -PathType Leaf) -or (Get-Item -LiteralPath $patchPath).Length -eq 0) {
    throw 'Task produced no plan-owned patch'
  }
  & git apply --cached -p2 --check -- $patchPath
  if ($LASTEXITCODE -ne 0) { throw 'owned patch does not apply cleanly to the empty index; do not stage dirty baseline hunks' }
  & git apply --cached -p2 -- $patchPath
  if ($LASTEXITCODE -ne 0) { throw 'failed to apply owned patch to index' }
  $cachedPaths = @(& git diff --cached --name-only)
  $extras = @($cachedPaths | Where-Object { $_ -notin $taskPaths })
  if ($extras.Count -gt 0) { throw "cached path outside Task whitelist: $($extras -join ', ')" }
  & git diff --cached --check
  if ($LASTEXITCODE -ne 0) { throw 'cached diff failed whitespace validation' }
  Assert-DirtyStatesExplainable -ExpectedStates @($checkpoint.states) -ExcludedPaths $taskPaths
  $baseHead = (& git rev-parse HEAD).Trim()
  $stagedTree = (& git write-tree).Trim()
  if ($LASTEXITCODE -ne 0) { throw 'failed to materialize the exact staged tree' }
  Save-AtomicJson ([ordered]@{
    task = $Task
    base_head = $baseHead
    staged_tree = $stagedTree
    patch_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $patchPath).Hash.ToLowerInvariant()
    cached_paths = $cachedPaths
  }) (Join-Path $attemptRoot 'staged-receipt.json')
  return
}

& git diff --cached --quiet
if ($LASTEXITCODE -ne 0) { throw 'index is not empty after Task commit' }
$commitPaths = @(& git diff-tree --no-commit-id --name-only -r HEAD)
$commitExtras = @($commitPaths | Where-Object { $_ -notin $taskPaths })
if ($commitExtras.Count -gt 0) { throw "commit contains path outside Task whitelist: $($commitExtras -join ', ')" }
if (-not (Test-Path -LiteralPath (Join-Path $attemptRoot 'staged-receipt.json'))) {
  throw 'missing plan-owned staging receipt'
}
$stagedReceipt = Get-Content -LiteralPath (Join-Path $attemptRoot 'staged-receipt.json') -Raw -Encoding utf8 | ConvertFrom-Json
$commitParent = (& git rev-parse 'HEAD^').Trim()
if ($commitParent -ne [string]$stagedReceipt.base_head) { throw 'HEAD is not the direct commit of the staged Task baseline' }
$commitTree = (& git rev-parse 'HEAD^{tree}').Trim()
if ($commitTree -ne [string]$stagedReceipt.staged_tree) { throw 'committed tree differs from the exact staged Task tree' }
Assert-DirtyStatesExplainable -ExpectedStates @($checkpoint.states) -ExcludedPaths $taskPaths
foreach ($baselineEntry in @($taskContext.baseline)) {
  if (@($baselineEntry.status).Count -eq 0) { continue }
  $currentEntry = Get-DirtyPathState -Relative ([string]$baselineEntry.path)
  if ((@($currentEntry.status) -join "`0") -ne (@($baselineEntry.status) -join "`0")) {
    throw "pre-existing dirty status was changed or absorbed by the Task commit: $($baselineEntry.path)"
  }
  $wasUntracked = @($baselineEntry.status | Where-Object { ([string]$_).StartsWith('??') }).Count -gt 0
  if ($wasUntracked) {
    if ([string]$currentEntry.worktree_file_sha256 -ne [string]$baselineEntry.worktree_file_sha256) {
      throw "pre-existing untracked bytes changed during Task commit: $($baselineEntry.path)"
    }
  } elseif ([string]$currentEntry.unstaged_patch_id -ne [string]$baselineEntry.unstaged_patch_id) {
    throw "pre-existing tracked dirty patch changed during Task commit: $($baselineEntry.path)"
  }
}
$nextStates = foreach ($entry in @($ledger.dirty_entries)) {
  Get-DirtyPathState -Relative ([string]$entry.path)
}
$nextCheckpointPath = Join-Path $runRoot ("dirty-checkpoint-task-{0:D2}-$([guid]::NewGuid().ToString('N')).json" -f $Task)
Save-AtomicJson ([ordered]@{
  schema_version = 1
  run_id = [string]$context.run_id
  sequence = [int]$checkpoint.sequence + 1
  task = $Task
  parent_checkpoint_sha256 = [string]$context.dirty_checkpoint_sha256
  dirty_overlap_ledger_sha256 = [string]$context.dirty_overlap_ledger_sha256
  states = @($nextStates)
}) $nextCheckpointPath
$nextCheckpointSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $nextCheckpointPath).Hash.ToLowerInvariant()
Save-AtomicJson ([ordered]@{
  task = $Task
  commit = (& git rev-parse HEAD).Trim()
  verified_at = (Get-Date).ToUniversalTime().ToString('o')
  commit_paths = $commitPaths
  previous_dirty_checkpoint_sha256 = [string]$context.dirty_checkpoint_sha256
  next_dirty_checkpoint_sha256 = $nextCheckpointSha256
}) (Join-Path $attemptRoot 'commit-verification.json')
Save-AtomicJson ([ordered]@{
  run_id = [string]$context.run_id
  run_root = $runRoot
  dirty_overlap_ledger_sha256 = [string]$context.dirty_overlap_ledger_sha256
  dirty_checkpoint_path = $nextCheckpointPath
  dirty_checkpoint_sha256 = $nextCheckpointSha256
}) $currentContextPath
```

Bootstrap e uso obrigatório:

```powershell
$guard = '.codex-tmp\universal-source-replacement-plan-guard\TaskOwnedGit.ps1'
& $guard -Action Initialize
```

Antes de **cada** Task N, execute `& $guard -Action Start -Task N` depois de confirmar o índice vazio e antes do primeiro edit. No commit, use somente `& $guard -Action Stage -Task N`, inspecione `git diff --cached`, faça o commit e execute `& $guard -Action VerifyAfterCommit -Task N`. O helper é o único escritor do índice; todos os `git add` interativos ou por arquivo ficam substituídos por esse protocolo. Se `Stage` falhar, não use `git add`, não altere o baseline e não force o patch.

## Revisão vinculante de executabilidade — 2026-08-04

Esta revisão foi feita contra o runtime e o worktree reais. Os contratos abaixo são normativos e substituem qualquer exemplo ou frase posterior que os contradiga. Ao executar uma Task, ajuste seus testes, `Files`, comandos de staging e Definition of Done para cumprir esta seção; não mantenha duas autoridades por compatibilidade acidental.

### Fronteira segura no worktree existente

- Registre no início da execução um `dirty_overlap_ledger.json` privado e imutável em `.codex-tmp/`, contendo `git diff --binary` hash, `git patch-id --stable`, status e hash de conteúdo de todo path já alterado que esteja na whitelist global das Tasks. Uma cadeia append-only de `dirty-checkpoint-*.json` ligada ao ledger registra o estado explicável após cada commit. `Start`, `Stage` e `VerifyAfterCommit` comparam exatamente status/conteúdo/diffs de toda entrada fora da whitelist da Task atual; para paths tracked dirty pertencentes à Task, o patch `a → b` precisa aplicar limpo sobre `HEAD` e o patch-id residual deve permanecer idêntico depois do commit; untracked/read-only preserva status e hash de bytes exatos. Nunca versionar ledger/checkpoints.
- `pipeline/ownership/reconcile.py`, inclusive `_strong_full_reading_consensus()`, já contém adição local não commitada. Trate essa implementação como **read-only**. A Task 3 cria `pipeline/ownership/consensus_v2.py`, `pipeline/ownership/owner_builder.py` e `pipeline/tests/test_owner_consensus_v2.py`; o caminho `enforce` passa a usá-los na Task 17. Não stage nem commit a adição local de `reconcile.py`.
- `pipeline/tests/test_owner_source_replacement_fail_closed.py` é arquivo não rastreado do usuário e o plano não toma posse dele. O guard o registra como baseline existente; um patch de Task não pode convertê-lo silenciosamente em arquivo novo do índice. Execute-o como regressão read-only; migre requisitos necessários para um novo `pipeline/tests/test_owner_source_replacement_enforce.py`, criado e versionado pela Task 10.
- Quando um teste de integração revelar correção em arquivo já sujo, a correção volta à Task proprietária e usa a whitelist concreta daquela Task. A Task 19 não possui staging genérico, não cria commit guarda-chuva e não contém placeholders executáveis.
- Antes de cada commit, capture `git diff --cached --name-only`; o conjunto deve ser subconjunto exato da whitelist explícita da Task. Se um hunk novo for inseparável de trabalho local do usuário, implemente a fronteira nova em módulo plan-owned e adapte somente um call site separável. Nunca use commit misto como atalho.

### Identidades e contratos canônicos

- `run_id` é a identidade imutável da **linhagem de conteúdo**. A run `off` e um replay de render compartilham `run_id`. Cada processo/raiz física recebe `execution_id` distinto; o replay acrescenta `replay_of_execution_id`. Relatórios e artefatos persistem os três campos quando aplicável.
- Records físicos mutáveis por execução — cleanup/render commit, materialization, repair request/attempt, page composition, stage/artifact refs, QA probes/verdicts, terminal proof e publication — exigem o `execution_id` atual. OCR/coverage/translation/graph de conteúdo persistem seu `origin_execution_id`; só podem ser reutilizados quando `origin_execution_id == replay_of_execution_id` e o replay valida toda a cadeia. Nunca reetiquete artifact antigo com o execution novo. No replay, reabra e valide a ref da run `off`, materialize os mesmos bytes em uma ref privada da execução nova e grave `origin_execution_id` + `source_artifact_ref_sha256`; a ref física publicada sempre pertence ao `execution_id` atual.
- `TextObservation` já possui `attempt_id`; esse é o único ID da tentativa OCR física. Não crie `ocr_attempt_id`. Um alias legado pode ser aceito apenas na desserialização `legacy/shadow`; se ambos existirem ou divergirem, `enforce` rejeita.
- A API única do graph passa a ser `validate(mode: Literal["legacy", "shadow", "enforce"]) -> tuple[OwnerViolation, ...]` e `require_valid(mode=...) -> None`, reutilizando o `OwnerViolation` real. Não criar/renomear um segundo tipo nem `validate_or_raise()` paralelo.
- `OCRBlock` e `OCRDiagnostics` do contrato request-scoped são snapshots frozen definidos explicitamente em `ownership/ocr_contract.py`; não congele nem exponha diretamente objetos mutáveis internos de `vision_stack`.
- Hashes canônicos (`canonical_page_sha256`, bytes, texto e JSON canônico) vivem em `pipeline/ownership/hash_contract.py`, usado por OCR, coverage, graph, execução, publicação e auditor. Nenhuma camada recalcula a mesma identidade com algoritmo próprio.
- `ChapterSourceManifest` é criado imediatamente após extrair a entrada e antes de qualquer processamento. Ele contém, em ordem: `page_id`, path relativo, `source_file_sha256`, `page_source_sha256`, dimensões/modo; também `source_page_count`, `source_tree_sha256`, `run_id` e `execution_id`. Paths absolutos não entram no JSON canônico. O builder valida SHA-256 como 64 hex e o tree hash contra os arquivos resolvidos pela variável de ambiente no momento da run.
- Artifact refs nunca serializam path absoluto: usam `artifact_store_id`, `generation_id`, `relative_path` POSIX e hashes. Na execução privada, existe exatamente uma raiz comum `<run>/.owner-executions/<execution_id>/`, um único `artifact_generation.json` e um único par `artifact_store_id/generation_id` para todas as páginas; `page_generation_id` identifica apenas a promoção atômica page-local sob essa raiz e nunca substitui o `generation_id` comum. Todo load recebe a raiz explícita correta, valida o marker comum, resolve com contenção e rejeita traversal/symlink/collision casefold **antes de qualquer open do target**. O journal canônico persiste somente refs/metadados/hashes; bytes PNG/ndarray nunca entram inline ou em base64.

### OCR, coverage e tradução sem perda de evidência

- Toda chamada física ao provider passa pelo adapter hash-bound da Task 2. A migração não se limita a “dois callers”: faça scan de **todos** os consumidores de `_last_full_page_line_records` e side channels equivalentes. No modo `enforce`, o allowlist desses símbolos deve ficar vazio; somente `legacy/shadow` explicitamente isolado pode lê-los.
- Tasks 5 e 6 retornam snapshots imutáveis `PageCoverageResult`, nunca tuplas parciais. O contrato contém `ledger_history` append-only (`ledger` é somente a cabeça derivada), `components`, `observations`, `ocr_requests`, `ocr_invocations`, o histórico append-only completo `recovery_requests`/`recovery_decisions` e `pending_request_ids`; cada passo de recuperação retorna uma nova instância preservando toda evidência anterior. `pending_requests` é somente uma property derivada desses três campos, nunca uma coleção concorrente serializada. Uma decisão `failed` cria obrigatoriamente uma request sucessora hash-linked para `next_strategy`; `exhausted` aborta a página e nunca pode desaparecer de um snapshot declarado ready.
- A descoberta usa a API real `discover_source_text_components(page_rgb, *, page_id, detector_regions, glyph_candidates=())`. `materiality` pertence à `CoverageEntry`, não a `SourceTextComponent`.
- A Task 8 define `TranslationAttempt` frozen com `attempt_id`, identidade run/page/source/owner, backend/variant/model/metadata, source/request/response hashes, provider-called/cache/status/error e verdict de idioma. O adapter estrito usa `translate_one_owner_attempt()`, o mesmo boundary real que o entrypoint público compatível `translate_pages()` passa a usar internamente, com um owner por chamada e backend forçado por tentativa; passthrough do source é sempre rejeitado, repetido dentro do budget e encerrado como falha operacional tipada sem export.
- Todo ID em `TranslationBinding.attempt_ids` resolve exatamente uma vez em `PageExecutionResult.translation_attempts`; tentativa órfã, binding órfão ou cardinalidade divergente é erro estrutural.

### Cardinalidade e autoridade completa do capítulo

- `VerifiedProjectInputs` carrega o `ChapterSourceManifest` original e exige correspondência bijetiva e ordenada com `PageExecutionResult`: página ausente, duplicada, extra, reordenada ou com hash divergente falha antes de staging/publicação. O adapter recebe uma única `private_execution_root`, exige que todas as `PageExecutionEvidenceRef` apontem ao mesmo marker/store/generation comum e que cada `page_id` resolva exatamente um `page_generation_id`/`current.json`; geração comum misturada, page-generation/pointer duplicado ou ref de outra raiz falha antes de construir `VerifiedProjectInputs`.
- `VerifiedPageProjectInput` inclui refs hash-verificadas para original/source, cleanup base, estágios canônicos e final, além de graph, binding, commits, geometria, texto e prova terminal. Não deriva original ou layer de `OutputPage` mutável.
- `ChapterAssetManifest` enumera **todo** payload relativo referenciado pelo projeto e toda evidência necessária ao auditor: `originals/**`, `images/**`, `layers/mask/**`, `layers/brush/**`, `layers/recovery/**`, `translated/**`, `evidence/verified_project_inputs.json`, `evidence/pages/<page_id>/**`, `project.json` e qualquer novo asset do schema. Cada item tem path relativo, file hash, pixel hash/modo quando aplicável, tamanho e owner/page provenance; evidência sem referência no projeto usa `project_reference=None` e `evidence_reference` explícita. Arquivos de controle não entram nesse tree hash para evitar auto-hash circular.
- Persista `chapter_source_manifest.json` e `chapter_asset_manifest.json` canônicos como controles detached. `evidence/verified_project_inputs.json` integra o asset tree. `export_manifest.json` referencia source manifest, verified inputs, asset manifest e hashes finais. `publication_receipt.json` referencia `artifact_generation.json`, source manifest, asset manifest, verified inputs, export manifest, project, evidence-source de replay e payload tree; o receipt não hasheia a si próprio. Essa DAG elimina ciclos e permite auditoria read-only.
- `VerifiedChapterBundle` cobre `ChapterSourceManifest`, `VerifiedProjectInputs`, `ChapterAssetManifest`, os quatro controles detached, tree hash da geração e todos os artefatos. Antes do commit, reabra o `project.json`, o evidence tree e os controles, resolva cada path somente dentro da raiz staged e prove existência/hash. Assets não referenciados ou referências fora da geração são rejeitados.
- No branch `enforce`, `_run_pipeline()` chama `recover_chapter_publication(work_dir)` **antes** de criar/escrever qualquer path público. Toda construção ocorre sob `.publication-staging/<transaction_id>/`; não escrever antecipadamente em `images/`, `originals/`, `translated/`, `layers/**`, `project.json` ou manifest.

### Publicação transacional e recuperação após crash

- `PublicationTransaction` publica a geração completa descrita por `ChapterAssetManifest` — inclusive `artifact_generation.json` — mais `chapter_source_manifest.json`, `chapter_asset_manifest.json`, `export_manifest.json` e `publication_receipt.json`, não apenas três alvos. Um `PublicationLock` cross-process exclusivo cobre recovery + staging + commit por raiz resolvida; leitores verificados usam lock compartilhado e só aceitam a geração cujo receipt, promovido por último, foi reaberto e validado. O teste compara a árvore pública inteira e todos os hashes antes/depois de cada fault point e prova que dois writers nunca intercalam WAL/renames.
- O WAL é criado e fsynced antes do primeiro rename. Para cada target, registra pre-state/path/hash e eventos `backup_intent`, `backup_done`, `promote_intent`, `promote_done`, todos fsynced antes/depois da operação. Inclua fault injection antes e depois de **cada** backup e promote.
- Recovery é idempotente em qualquer prefixo do journal. Se remover/restaurar falhar, conserva WAL, staging e backup; uma segunda chamada deve concluir. Só apague WAL/backup depois de reabrir e revalidar por hash a árvore restaurada ou promovida.
- A escrita de uma página também é atômica: `PageCandidateTransaction` grava uma geração completa de artefatos da página em diretório temporário dentro da raiz privada comum e promove o diretório/ponteiro em uma única transação recuperável. Cada página possui `page_generation_id` próprio, mas todas compartilham o marker e o `generation_id` da execução. Nunca publique `final.png` e `execution_result.json` em renames independentes.

### Composição multi-owner e reparo limitado

- Uma tentativa de reparo substitui somente o commit do owner afetado em um conjunto imutável e recompõe a página desde o original usando **todos** os cleanups finais e **todos** os glyph patches PT-BR finais. O resultado nunca parte do candidate parcial anterior.
- RED obrigatório com owners A/B: A já válido, B escala até R3; o final conserva A exatamente uma vez, materializa B exatamente uma vez e não contém nenhum source. Falha de B não persiste final, commit ou artefato parcial.
- `RepairBudgetPolicy` frozen/hash-bound fixa limites por estratégia/variante, expansões geométricas e retries operacionais; toda tentativa possui fingerprint `request + strategy + geometry + input hash`. Fingerprint repetido é rejeitado. Falha transitória persistente termina em `RepairInfrastructureExhausted` e zero export, nunca loop ou preservação do inglês.

### Prova terminal material, não apenas mudança de pixels

- Prova de remoção recompõe o suporte source completo e mede residual/ink pós-cleanup. “Algum pixel mudou” não basta. Residual ambíguo escala até R3; não pode virar `final_verified`.
- Prova do target exige glyph mask/delta não vazio, alpha material, contraste mínimo contra `cleanup_base`, contenção na safe region, ausência de clipping e igualdade do delta com a composição persistida.
- REDs adversariais obrigatórios: inglês ainda legível após alteração mínima; target zero-alpha; target com zero contraste; target cortado/fora da safe region; todos devem ser rejeitados.
- OCR target incompleto só pode ser tolerado quando os checks materiais, hashes e cadeia física forem válidos **e** o observer não contiver source-only tokens materiais. OCR indisponível/incompleto para a prova de source continua impedindo promoção.
- Métricas de residual do gate runtime usam exclusivamente o probe terminal identificado por `TerminalPixelProof.final_qa_probe_id` e os `issue_ids` hash-linked desse probe. Issues de probes anteriores permanecem no journal como histórico de reparo, mas não contam como residual terminal depois que um probe novo prova a correção; nunca escolha “o último” por ordem da tuple ou timestamp.

### Artefatos, replay de estilo e auditoria visual

- Persistir refs lossless/hash-verificadas para `original`, `cleanup/inpaint`, `target/typeset`, `page_composition` e `persisted_final`; `coverage_ocr_overlay` é derivado e rotulado. Dentro da mesma execução, se dois estágios canônicos tiverem pixels idênticos, ambos apontam para a mesma ref com `alias_reason`; entre executions, replay materializa uma ref nova ligada à fonte. O auditor não fabrica/reconstrói estágio ausente a partir de debug.
- O replay obrigatório deste plano usa `style_copy_mode=render`: style-copy fica ativo, mas o gate de fidelidade de estilo não decide a correção de conteúdo. Uma auditoria opcional `style_copy_mode=enforce` pode reportar NO-GO de estilo separadamente; ela só bloqueia este plano se violar conteúdo, hashes, cardinalidade ou zero-inglês.
- O produtor de config da matriz deve preservar `style_copy_mode=off` solicitado e nunca forçar `enforce`. Todos os configs funcionais declaram explicitamente `owner_graph_mode=enforce`, `style_copy_mode=off` e `input_key`; seus hashes são atualizados por teste.
- A matriz final contém quatro entries: Mitch Items cap. 39 como regressão principal, mais três obras externas — uma `calibration` e dois `holdout`. Não chame as três externas de “três holdouts”. Cada entry executa auditor independente, gera JSON hash-linked e exige as sete métricas, cardinalidade, identities e gates; exit code isolado não é evidência suficiente.
- O schema do auditor contém um `ExternalPageAudit` por página, em bijeção com `ChapterSourceManifest`, e inclui/testa `auditor_invocation_is_independent`, `auditor_root_hash_matches_final`, `auditor_execution_id`, `auditor_process_nonce` e `external_audit_gate_status`. A independência deriva de identidade emitida pelo processo filho, não da intenção `force_new_process=True` do caller.
- Inspeção manual usa `view_image(detail="original")`. Existem tabelas separadas para a run `off` e o replay ativo `render`; abrir todos os 42 pares nas duas runs e os seis estágios das sentinelas. Registre verdicts humanos ligados a paths/hashes. Imagens legitimamente idênticas podem compartilhar pixel hash; unicidade é exigida para a tupla `page_id + source hash + artifact path + lineage`, não para os pixels globais.

### REDs transversais obrigatórios

Além dos nodeids de cada Task, mantenha testes parametrizados que adulteram `source_tree_sha256`, schema, `run_id`, `execution_id`, page/source identity, coverage/OCR/translation/repair chain, owner/binding/cardinality, artifact ref/source-ref, path traversal e página ausente/extra/duplicada/reordenada. Todo caso deve falhar antes de renderer ou staging público. A auditoria externa repete esses negativos sobre artefatos persistidos, sem consultar objetos vivos da run.

## Baseline e evidência fixa

- Design aprovado: `docs/plans/2026-08-04-universal-balloon-source-replacement-design.md`.
- Commit do design: `6d5863d9`.
- Branch inspecionada: `Troca_de_motores`.
- HEAD na criação original do plano: `6d5863d9`.
- HEAD inspecionado antes desta revisão de executabilidade: `0558a9dd966b87cf1255de05befde8eb9a2ea826`.
- Snapshot original preservado para auditoria: 4.241 linhas, SHA-256 `c0719227d364c23af9ee27eb2a9de81b98c43d4f33657d49fcc6118a8a5b537d`.
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
- Create: `pipeline/ownership/hash_contract.py`
- Create: `pipeline/tests/test_hash_contract.py`
- Read only: `.codex-tmp/mch39_full_directional_gradient_20260803_retry2/out/debug/e2e/**`
- Read only: `temporario1/mch39/**`

**Step 1: Registrar o estado e os diffs que precisam ser preservados**

Run:

```powershell
$guard = '.codex-tmp\universal-source-replacement-plan-guard\TaskOwnedGit.ps1'
& $guard -Action Start -Task 1
git status --short --branch
git diff -- pipeline/vision_stack/ocr.py pipeline/vision_stack/runtime.py pipeline/ownership pipeline/strip/run.py pipeline/strip/process_bands.py pipeline/qa pipeline/tests
git status --short -- pipeline/tests/test_owner_source_replacement_fail_closed.py
$guardContext = Get-Content -LiteralPath '.codex-tmp\universal-source-replacement-plan-guard\current.json' -Raw | ConvertFrom-Json
if (-not (Test-Path -LiteralPath (Join-Path $guardContext.run_root 'dirty_overlap_ledger.json') -PathType Leaf)) {
  throw 'dirty_overlap_ledger.json was not created before Task 1'
}
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
    assert re.fullmatch(r"[0-9a-f]{64}", real["source_tree_sha256"])
    assert "source_path" not in real
    assert all(case["expected"]["english_dialogue_residual_count"] == 0 for case in manifest["cases"])


def test_manifest_input_tree_hash_recomputation_is_portable(tmp_path, monkeypatch):
    source = _write_small_external_input(tmp_path)
    monkeypatch.setenv("TRADUZAI_TEST_SOURCE", str(source))
    expected = canonical_source_tree_sha256(_ordered_images(source), source)
    assert recompute_manifest_input_tree_sha256("TRADUZAI_TEST_SOURCE") == expected


def test_canonical_source_tree_hash_binds_order_relative_paths_file_and_pixel_hashes(tmp_path):
    root = _write_nested_source_tree(tmp_path)
    ordered = _ordered_images(root)
    first = canonical_source_tree_sha256(ordered, root)
    assert first == canonical_source_tree_sha256(ordered, root)
    assert first != canonical_source_tree_sha256(tuple(reversed(ordered)), root)
    _mutate_one_source_pixel(ordered[0])
    assert first != canonical_source_tree_sha256(ordered, root)


@pytest.mark.integration
def test_real_regression_source_tree_hash_matches_runtime_input(monkeypatch):
    source_value = os.environ.get("TRADUZAI_MATRIX_MITCH39_SOURCE")
    if not source_value:
        pytest.skip("TRADUZAI_MATRIX_MITCH39_SOURCE is required for the real corpus integration")
    source = Path(source_value).resolve(strict=True)
    monkeypatch.setenv("TRADUZAI_MATRIX_MITCH39_SOURCE", str(source))
    real = _load_manifest()["real_regressions"]["mitch_items_ch39"]
    assert recompute_manifest_input_tree_sha256(real["input_key"]) == real["source_tree_sha256"]
```

**Step 4: Confirmar RED**

Run:

```powershell
Push-Location pipeline
try {
  $redOutput = & .\venv\Scripts\python.exe -m pytest tests/test_hash_contract.py tests/test_universal_source_replacement_manifest.py -q 2>&1
  $redExit = $LASTEXITCODE
  $redText = $redOutput | Out-String
  $redOutput
  if ($redExit -ne 1) { throw "Task 1 expected pytest exit 1 for a behavioral RED, got $redExit" }
  if ($redText -match '(?im)^ERROR(?:\s|:)|ERROR collecting|errors during collection|^E\s+fixture .* not found|UsageError|ModuleNotFoundError|ImportError') {
    throw 'Task 1 RED is structural rather than behavioral; create/fix the API skeleton and rerun the same nodeids'
  }
  if ($redText -notmatch '(?m)^FAILED\s+\S+::') { throw 'Task 1 RED did not report a targeted FAILED nodeid' }
} finally { Pop-Location }
```

Expected: FAIL porque o manifest e a primitiva canônica de source-tree hash ainda não existem.

**Step 5: Criar a primitiva canônica de hash, o manifest e as receitas declarativas**

O manifest deve:

- usar nomes/categorias genéricos para casos sintéticos;
- registrar a regressão real por `input_key`, hashes e páginas sentinela;
- exigir `style_copy_mode=off` e `style_copy_mode=render` como duas variantes da mesma entrada;
- exigir `target_present=true`, não apenas `source_absent=true`;
- separar `calibration` e `holdout`;
- não conter path absoluto nem `.codex-tmp` como dependência versionada.

Em `ownership/hash_contract.py`, defina a autoridade recursiva única `JSONValue` (`null|bool|int|float|string|list[JSONValue]|dict[str, JSONValue]`) e implemente já nesta Task `sha256_bytes`, `sha256_file`, `sha256_text`, `canonical_json_sha256`, `canonical_page_sha256` e `canonical_source_tree_sha256(ordered_paths, root)`. Todos os módulos importam esse alias; snapshots deep-freezeiam coleções ao persistir. O tree hash usa JSON canônico ordenado contendo path relativo POSIX, file hash, pixel hash RGB, dimensões e modo; rejeita path absoluto/fora da raiz/duplicado/casefold collision. `recompute_manifest_input_tree_sha256(input_key)` é helper do teste que resolve a env e chama essa única primitiva, não outro algoritmo. Tasks 2 e 9 apenas ampliam/reutilizam esse módulo.

`recipes.json` descreve os casos sintéticos como primitivas genéricas (canvas, container, glyph layers, idioma e defeito induzido), sem screenshot da obra. Nesta Task os testes validam somente schema/cobertura; o primeiro RED comportamental que executa uma receita entra junto do coordenador na Task 9.

**Step 6: Confirmar GREEN**

Run:

```powershell
Push-Location pipeline
try { .\venv\Scripts\python.exe -m pytest tests/test_hash_contract.py tests/test_universal_source_replacement_manifest.py -q } finally { Pop-Location }
```

Expected: PASS.

**Step 7: Commit**

```powershell
$guard = '.codex-tmp\universal-source-replacement-plan-guard\TaskOwnedGit.ps1'
& $guard -Action Stage -Task 1
git diff --cached --check
git commit -m "test: characterize universal source replacement failures"
& $guard -Action VerifyAfterCommit -Task 1
```

### Task 2: Tornar o resultado OCR atômico e request-scoped

**Files:**

- Modify: `pipeline/ownership/hash_contract.py`
- Create: `pipeline/ownership/ocr_contract.py`
- Modify: `pipeline/vision_stack/ocr.py:590-724,1735-2020`
- Modify: `pipeline/vision_stack/runtime.py:2226-2245,13732-13827,15480-15582`
- Test: `pipeline/tests/test_vision_stack_ocr.py`
- Test: `pipeline/tests/test_vision_stack_runtime.py`
- Test: `pipeline/tests/test_hash_contract.py`
- Create: `pipeline/tests/test_ocr_side_channel_guard.py`

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
        page_source_sha256=_sha256("page-10"),
        root_input_pixel_sha256=_sha256("root-page-10"),
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


def test_ocr_result_rejects_record_from_other_execution_under_same_content_run(self):
    request = _request(run_id="run-a", origin_execution_id="exec-a", page_id="page_010")
    stale = _ocr_record(run_id="run-a", origin_execution_id="exec-b", page_id="page_010")
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
  $redOutput = & .\venv\Scripts\python.exe -m pytest `
    tests/test_vision_stack_ocr.py::PaddleBlockMappingTests::test_full_page_line_records_are_request_scoped_across_parallel_pages `
    tests/test_vision_stack_runtime.py::VisionStackRuntimeTests::test_runtime_never_reads_last_full_page_side_channel `
    tests/test_vision_stack_ocr.py::PaddleBlockMappingTests::test_ocr_invocation_result_is_deeply_immutable `
    tests/test_vision_stack_ocr.py::PaddleBlockMappingTests::test_every_ocr_record_links_exact_logical_request_and_physical_attempt `
    tests/test_vision_stack_ocr.py::PaddleBlockMappingTests::test_ocr_result_rejects_record_with_mismatched_request_identity `
    tests/test_vision_stack_ocr.py::PaddleBlockMappingTests::test_ocr_result_rejects_record_from_other_execution_under_same_content_run `
    tests/test_vision_stack_ocr.py::PaddleBlockMappingTests::test_provider_boundary_rejects_declared_hash_b_when_pixels_a_are_passed `
    tests/test_vision_stack_ocr.py::PaddleBlockMappingTests::test_provider_boundary_records_hash_of_actual_array_on_success `
    tests/test_vision_stack_ocr.py::PaddleBlockMappingTests::test_parameterized_rotation_and_deskew_specs_replay_exact_provider_pixels `
    tests/test_vision_stack_ocr.py::PaddleBlockMappingTests::test_cache_or_dot_heuristic_attempt_is_not_fresh_physical_ocr `
    tests/test_vision_stack_ocr.py::PaddleBlockMappingTests::test_production_provider_calls_exist_only_inside_hash_bound_boundary `
    tests/test_ocr_side_channel_guard.py `
    -q 2>&1
  $redExit = $LASTEXITCODE
  $redText = $redOutput | Out-String
  $redOutput
  if ($redExit -ne 1) { throw "Task 2 expected pytest exit 1 for a behavioral RED, got $redExit" }
  if ($redText -match '(?im)^ERROR(?:\s|:)|ERROR collecting|errors during collection|^E\s+fixture .* not found|UsageError|ModuleNotFoundError|ImportError') {
    throw 'Task 2 RED is structural rather than behavioral; create/fix the API skeleton and rerun the same nodeids'
  }
  if ($redText -notmatch '(?m)^FAILED\s+\S+::') { throw 'Task 2 RED did not report a targeted FAILED nodeid' }
} finally { Pop-Location }
```

Expected: FAIL porque full-page lines ainda vivem em `_last_full_page_line_records`.

**Step 4: Implementar os contratos imutáveis**

Em `ocr_contract.py`:

```python
@dataclass(frozen=True)
class OCRRequest:
    run_id: str
    origin_execution_id: str
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
    origin_execution_id: str
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
    def identity(self) -> tuple[str, str, str, str, str, str, str, str, str, str]: ...

    @property
    def qualifies_as_fresh_physical_inference(self) -> bool: ...


@dataclass(frozen=True)
class OCRObservationRecord:
    observation_id: str
    attempt_id: str
    run_id: str
    origin_execution_id: str
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
    def request_identity(self) -> tuple[str, str, str, str, str, str, str]: ...

    @property
    def attempt_identity(self) -> tuple[str, str, str, str, str, str, str, str, str, str]: ...


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
    def origin_execution_id(self) -> str: return self.request.origin_execution_id
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

Defina explicitamente snapshots `OCRBlock` e `OCRDiagnostics` em `ownership/ocr_contract.py`; eles copiam apenas campos serializáveis do provider/runtime e não expõem objetos mutáveis de `vision_stack`. Use records `frozen=True` também para `OCRAttempt`, `OCRTransformOperation` e `OCRTransformSpec`; quaisquer extras JSON devem ser congelados recursivamente, sem dict/list mutável aninhado. `OCRObservationRecord` é o envelope OCR request-scoped desta Task, deliberadamente independente do modelo de ownership. `OCRInvocationResult.build()` valida que toda tentativa/record repete `origin_execution_id` e a identidade lógica da request, que todo record/linha referencia exatamente um `attempt_id`, que sua identidade/hash físico coincide com a tentativa referenciada e que `payload_sha256` corresponde ao texto normalizado. `OCRTransformSpec.build()` serializa operações em JSON canônico, recalcula `sha256` e rejeita combinação incompleta; rotação/deskew persistem matriz afim em inteiros fixos de 1e-6, algoritmo, interpolação, borda e output size, permitindo replay bit-exato sem depender de float JSON. `attempt_chain_sha256` é SHA-256 do JSON canônico da tupla na ordem real de execução, cobrindo todos os campos de identidade, parent/input, `transform_spec.canonical_json_bytes`, bbox, shape/mode e flags provider/cache; não ordenar por texto ou confidence. Funções de hash vêm exclusivamente de `ownership/hash_contract.py`.

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

Adapte **todos** os callers de produção para consumir o resultado diretamente. Faça `rg` sobre `_last_full_page_line_records`, atributos `_last_*`, caches de linhas e wrappers que copiam records; não confie numa contagem fixada no plano. `test_ocr_side_channel_guard.py` faz inspeção AST/lexical e exige allowlist vazia no branch `enforce`. Pare de chamar `getattr(ocr, "_last_full_page_line_records", ...)`; qualquer uso restante deve estar isolado e testado como `legacy/shadow` read-only.

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
  .\venv\Scripts\python.exe -m pytest tests/test_hash_contract.py tests/test_vision_stack_ocr.py tests/test_vision_stack_runtime.py tests/test_ocr_side_channel_guard.py -q
} finally { Pop-Location }
```

Expected: PASS.

**Step 8: Commit**

```powershell
$guard = '.codex-tmp\universal-source-replacement-plan-guard\TaskOwnedGit.ps1'
& $guard -Action Stage -Task 2
git diff --cached --check
git commit -m "fix: isolate OCR evidence per page request"
& $guard -Action VerifyAfterCommit -Task 2
```

### Task 3: Registrar origem real e impedir consenso artificial

**Files:**

- Modify: `pipeline/ownership/model.py:172-260`
- Modify: `pipeline/ownership/coordinates.py:83-145`
- Modify: `pipeline/ownership/ocr_adapter.py:339-550`
- Modify: `pipeline/ownership/evidence.py`
- Read only: `pipeline/ownership/reconcile.py:274-520`
- Create: `pipeline/ownership/consensus_v2.py`
- Create: `pipeline/ownership/owner_builder.py`
- Modify: `pipeline/ownership/project.py`
- Modify: `pipeline/schema/project_schema_v12.py`
- Modify: `pipeline/strip/process_bands.py:2375-2555,9930-9980`
- Modify: `pipeline/strip/run.py:5550-5700`
- Test: `pipeline/tests/test_owner_evidence.py`
- Test: `pipeline/tests/test_owner_reconcile.py`
- Test: `pipeline/tests/test_owner_reconcile_properties.py`
- Create: `pipeline/tests/test_owner_consensus_v2.py`
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
    current = _page_context("page_010", sha=_sha256("page-10"))
    stale = _obs("STALE PAGE", page_id="page_009", page_sha=_sha256("page-9"))
    with pytest.raises(OwnerEvidenceIdentityError):
        validate_and_group_owner_observations(current, [stale])


def test_previous_run_id_is_rejected_before_consensus():
    current = _page_context("page_010", run_id="run-current", sha=_sha256("page-10"))
    stale = _obs("STALE RUN", run_id="run-previous", page_id="page_010", page_sha=_sha256("page-10"))
    with pytest.raises(OwnerEvidenceIdentityError):
        validate_and_group_owner_observations(current, [stale])


def test_new_owner_graph_roundtrip_preserves_ocr_identity_and_hashes():
    graph = _graph_with_new_observation_identity()
    loaded = OwnerGraph.from_dict(graph.to_dict(), enforce=True)
    assert loaded.schema_version == OWNER_GRAPH_SCHEMA_VERSION == 2
    assert loaded.verification_status == "verified"
    assert loaded.run_id == graph.run_id
    assert loaded.page_source_sha256 == graph.page_source_sha256
    assert loaded.observations[0].run_id == graph.observations[0].run_id
    assert loaded.observations[0].origin_execution_id == graph.observations[0].origin_execution_id
    assert loaded.observations[0].invocation_id == graph.observations[0].invocation_id
    assert loaded.observations[0].attempt_id == graph.observations[0].attempt_id
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
    graph = _graph_with_new_observation_identity(run_id="run-a", page_source_sha256=_sha256("page-a"))
    graph = dataclasses.replace(
        graph,
        observations=(dataclasses.replace(graph.observations[0], run_id="run-b"),),
    )
    with pytest.raises(OwnerGraphValidationError):
        graph.require_valid(mode="enforce")
    graph = _graph_with_new_observation_identity(run_id="run-a", page_source_sha256=_sha256("page-a"))
    graph = dataclasses.replace(
        graph,
        observations=(dataclasses.replace(graph.observations[0], page_source_sha256=_sha256("page-b")),),
    )
    with pytest.raises(OwnerGraphValidationError):
        graph.require_valid(mode="enforce")


def test_verified_graph_rejects_observation_from_other_origin_execution():
    graph = _graph_with_new_observation_identity(origin_execution_id="exec-off")
    graph = dataclasses.replace(
        graph,
        observations=(dataclasses.replace(graph.observations[0], origin_execution_id="exec-other"),),
    )
    with pytest.raises(OwnerGraphValidationError):
        graph.require_valid(mode="enforce")


def test_legacy_positional_observation_remains_loadable_but_not_enforceable():
    legacy = TextObservation("obs", "page_001", (), "TEXT", 0.9, "fixture", (0, 0, 20, 10))
    assert not legacy.identity_complete
    with pytest.raises(OwnerGraphValidationError):
        _graph(observations=[legacy]).require_valid(mode="enforce")


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
  $redOutput = & .\venv\Scripts\python.exe -m pytest `
    tests/test_owner_evidence.py `
    tests/test_owner_reconcile.py `
    tests/test_owner_reconcile_properties.py `
    tests/test_owner_consensus_v2.py `
    tests/test_owner_graph.py `
    tests/test_project_migration.py `
    tests/test_project_schema_v12.py `
    tests/test_project_writer.py `
    tests/test_strip_owner_control_plane.py `
    -q 2>&1
  $redExit = $LASTEXITCODE
  $redText = $redOutput | Out-String
  $redOutput
  if ($redExit -ne 1) { throw "Task 3 expected pytest exit 1 for a behavioral RED, got $redExit" }
  if ($redText -match '(?im)^ERROR(?:\s|:)|ERROR collecting|errors during collection|^E\s+fixture .* not found|UsageError|ModuleNotFoundError|ImportError') {
    throw 'Task 3 RED is structural rather than behavioral; create/fix the API skeleton and rerun the same nodeids'
  }
  if ($redText -notmatch '(?m)^FAILED\s+\S+::') { throw 'Task 3 RED did not report a targeted FAILED nodeid' }
} finally { Pop-Location }
```

Expected: o consenso atual conta aliases como providers independentes ou aceita identidade incompleta.

**Step 4: Ampliar `TextObservation` sem quebrar construtores legados**

Campos obrigatórios:

```python
run_id: str
origin_execution_id: str
invocation_id: str
attempt_id: str
provider_family: str
page_source_sha256: str
root_input_pixel_sha256: str
input_pixel_sha256: str
payload_sha256: str
```

Acrescente esses campos **ao final** do dataclass, após um marcador `KW_ONLY`, com defaults vazios somente para desserialização/fixtures legadas. Isso preserva temporariamente os construtores posicionais atuais. `identity_complete` exige todos os dez campos e o boundary `enforce` rejeita vazio; legacy/shadow pode carregar o registro apenas como `legacy_unverified`.

Todos os aliases derivados do mesmo `OCRInvocationResult` preservam exatamente o `invocation_id` e `provider_family` da `OCRRequest`; observações de cada array físico preservam também o `attempt_id` já existente em `TextObservation`, `root_input_pixel_sha256` e `input_pixel_sha256` do `OCRAttempt` real. Tentativas nativa/cinza/invertida/2x da mesma invocação continuam uma única origem de consenso. A identidade da página vem da request, nunca de parsing do filename. O adapter valida igualdade — não cria um segundo ID de tentativa. Um payload legado com `ocr_attempt_id` pode ser lido apenas em `legacy/shadow`; se trouxer também `attempt_id` divergente, falha.

Defina em `ownership/model.py` a autoridade única `OWNER_GRAPH_SCHEMA_VERSION = 2` e `OWNER_GRAPH_LEGACY_SCHEMA_VERSION = 1`; `ownership/project.py` importa e pode reexportar esses símbolos para compatibilidade, mas não mantém outro literal. Faça o bump explícito do schema do `OwnerGraph`; acrescente ao final, como keywords, `run_id: str = ""`, `origin_execution_id: str = ""`, `page_source_sha256: str = ""` e `verification_status: Literal["verified", "legacy_unverified"] = "verified"`, e atualize `to_dict()/from_dict(*, enforce=False)`. O schema 2 exige os quatro campos explícitos, identidade completa e igualdade de run/origin-execution/page/source com todas as observações para ser `verified`. Payload v1 é desserializado com identidade vazia e `verification_status="legacy_unverified"` somente em legacy/shadow; não fabrique IDs nem permita entrada em `enforce`. Versões desconhecidas falham em todos os modos. A API canônica é `validate(mode=...) -> tuple[OwnerViolation, ...]` e `require_valid(mode=...) -> None`, reutilizando o tipo existente; remova exemplos de uma terceira API de validação. `validate_serialized_owner_graph(..., enforce=...)`, project writer/migration e schema v12 devem propagar o modo e a constante. O adapter de `run_chapter(legacy_project_status=...)` deve mapear para esse campo, sem manter dois status divergentes.

**Step 4a: Migrar todos os construtores diretos no mesmo commit**

```powershell
$observationSites = & rg -n "TextObservation\(" pipeline --glob '*.py'
$observationSiteExit = $LASTEXITCODE
$observationSites
if ($observationSiteExit -gt 1) { throw "TextObservation call-site scan failed: $observationSiteExit" }
```

Migre cada call site de produção e cada fixture que entra em `enforce` para keywords explícitas `run_id`, `origin_execution_id`, `invocation_id`, `attempt_id`, `provider_family`, `page_source_sha256`, `root_input_pixel_sha256`, `input_pixel_sha256`, `payload_sha256`. Não confie na contagem do snapshot: rerode a busca após editar e classifique qualquer constructor ainda vazio como fixture `legacy_unverified` explícita. Não dependa da ordem posicional dos novos campos. Merge, `to_dict()/from_dict()` e todos os boundaries rejeitam run/origin-execution, root hash, attempt id ou input pixel hash divergente antes do consenso.

Converta cada `OCRObservationRecord` da Task 2 em `TextObservation` preservando `run_id`, `origin_execution_id`, `page_id`, `page_source_sha256`, `root_input_pixel_sha256`, `input_pixel_sha256`, `invocation_id`, `attempt_id` e `provider_family` sem alteração, mapeando apenas `variant_id` para o campo legado já existente `provider_variant`. Em seguida, migre **todas** as construções diretas de `OwnerGraph` encontradas por `rg -n "OwnerGraph\(" pipeline --glob '*.py'`: produção e testes devem usar `schema_version=OWNER_GRAPH_SCHEMA_VERSION`, `run_id`, `origin_execution_id` e `page_source_sha256`, nunca o literal `1` nem identidade vazia no modo novo. Não modifique a função local preexistente em `ownership/reconcile.py`: novos builders/consenso vivem nos módulos plan-owned e são conectados ao `enforce` na Task 17. Releia também todos os boundaries `OwnerGraph.from_dict`/`validate_serialized_owner_graph`, inclusive `ownership/project.py`, `strip/process_bands.py` e os testes de project migration/writer/schema.

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
        if not {"schema_version", "run_id", "origin_execution_id", "page_source_sha256"} <= keywords:
            bad.append(f"{path}:{node.lineno}: missing canonical graph identity keyword")
        for kw in node.keywords:
            if kw.arg == "schema_version" and isinstance(kw.value, ast.Constant) and kw.value.value == 1:
                bad.append(f"{path}:{node.lineno}: legacy schema literal")
if bad:
    raise SystemExit("legacy OwnerGraph schema literals remain:\n" + "\n".join(bad))
'@ | python -
```

**Step 5: Substituir provider-count por inference-count**

Em `ownership/consensus_v2.py`, sem editar `_strong_full_reading_consensus()` local preexistente:

```python
def independent_origins(group: Sequence[TextObservation]) -> frozenset[str]:
    return frozenset(observation.invocation_id for observation in group)
```

Use `independent_origins` para share, runner-up e corroboration. Provider permanece telemetria, não voto.

Em `owner_builder.py`, defina contratos reais:

```python
@dataclass(frozen=True)
class OwnerPageEvidenceContext:
    run_id: str
    origin_execution_id: str
    page_id: str
    page_source_sha256: str


@dataclass(frozen=True)
class OwnerObservationGroup:
    group_id: str
    run_id: str
    origin_execution_id: str
    page_id: str
    page_source_sha256: str
    component_ids: tuple[str, ...]
    container_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    independent_invocation_ids: tuple[str, ...]
    source_payload: str
    source_payload_sha256: str
    consensus_sha256: str
```

`owner_builder.py` já nasce funcional nesta Task, não como arquivo vazio: exponha `validate_and_group_owner_observations(page_context: OwnerPageEvidenceContext, observations: Sequence[TextObservation]) -> tuple[OwnerObservationGroup, ...]`. Ela rejeita context mutável/incompleto, valida `run_id/origin_execution_id/page_id/page_source_sha256`, aplica `merge_observation_strict`, agrupa por suporte geométrico/componente e chama somente o consenso de `consensus_v2.py`. IDs/tuples têm ordem canônica; payload/hash/consensus são recalculados e roundtrip/tamper são cobertos em `test_owner_consensus_v2.py`. A Task 7 adiciona `build_owner_page_graph_from_coverage()` sobre esse núcleo, sem redefinir validação/agrupamento.

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
    tests/test_owner_consensus_v2.py `
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
$guard = '.codex-tmp\universal-source-replacement-plan-guard\TaskOwnedGit.ps1'
& $guard -Action Stage -Task 3
git diff --cached --check
git commit -m "fix: count independent OCR evidence origins"
& $guard -Action VerifyAfterCommit -Task 3
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
    entry = _entry(state="explicit_non_dialogue_preserve", ocr_attempt_ids=())
    with pytest.raises(CoverageInvariantError):
        _ledger(entries=(entry,)).require_complete()


def test_ledger_rejects_expected_component_missing_from_entries():
    ledger = _ledger(
        component_inventory=_discovery_inventory("component-a", "component-b"),
        entries=(_entry(component_id="component-a"),),
    )
    with pytest.raises(CoverageInvariantError, match="component-b"):
        ledger.require_complete()


def test_component_inventory_allows_only_recovery_bound_append():
    first = _ledger_with_discovery_inventory("component-a")
    decision = _successful_materialization_decision(
        request=_unassociated_region_request("observation-b", _polygon_b()),
        materialized_component_id="component-b",
    )
    second = first.extend_component_inventory(
        _materialized_inventory_entry("component-b", decision=decision)
    )
    assert second.inventory_version == first.inventory_version + 1
    assert second.parent_ledger_sha256 == first.sha256
    assert second.expected_component_ids == ("component-a", "component-b")


@pytest.mark.parametrize("tamper", ["remove", "replace", "reorder", "unbound_append"])
def test_component_inventory_rejects_non_monotonic_or_unbound_change(tamper):
    first, second = _valid_inventory_extension()
    with pytest.raises(CoverageInvariantError):
        validate_inventory_successor(first, _tamper_inventory(second, tamper))


def test_owner_identity_survives_every_lifecycle_transition():
    lifecycle = _owner_lifecycle()
    final = lifecycle.advance("final_verified", evidence=_verified_evidence())
    assert final.identity == lifecycle.identity


def test_owner_graph_accepts_canonical_repair_states_but_enforce_rejects_review_terminal():
    for state in ("target_ready", "execution_attempt", "repair_pending", "cleaned", "final_verified"):
        assert not _graph_with_owner_state(state).validate(mode="enforce")
    assert "review_terminal_forbidden_in_enforce" in _violation_codes(
        _graph_with_owner_state("review_required", mode="enforce").validate(mode="enforce")
    )
```

**Step 2: Confirmar RED**

```powershell
Push-Location pipeline
try {
  $redOutput = & .\venv\Scripts\python.exe -m pytest tests/test_owner_coverage_rescue.py tests/test_owner_lifecycle.py tests/test_owner_graph.py::test_owner_graph_accepts_canonical_repair_states_but_enforce_rejects_review_terminal -q 2>&1
  $redExit = $LASTEXITCODE
  $redText = $redOutput | Out-String
  $redOutput
  if ($redExit -ne 1) { throw "Task 4 expected pytest exit 1 for a behavioral RED, got $redExit" }
  if ($redText -match '(?im)^ERROR(?:\s|:)|ERROR collecting|errors during collection|^E\s+fixture .* not found|UsageError|ModuleNotFoundError|ImportError') {
    throw 'Task 4 RED is structural rather than behavioral; create/fix the API skeleton and rerun the same nodeids'
  }
  if ($redText -notmatch '(?m)^FAILED\s+\S+::') { throw 'Task 4 RED did not report a targeted FAILED nodeid' }
} finally { Pop-Location }
```

Expected: FAIL porque os módulos ainda não existem.

**Step 3: Implementar dataclasses imutáveis**

Em `coverage.py`:

```python
CoverageState = Literal[
    "discovered",
    "challenged",
    "observed",
    "owned",
    "target_ready",
    "execution_attempt",
    "repair_pending",
    "cleaned",
    "rendered",
    "final_verified",
    "explicit_non_dialogue_preserve",
]


@dataclass(frozen=True)
class CoverageRecoveryRequest:
    request_id: str
    run_id: str
    origin_execution_id: str
    page_id: str
    page_source_sha256: str
    component_id: str | None
    observation_ids: tuple[str, ...]
    anchor_polygon_page: Polygon | None
    anchor_sha256: str
    attempt_kind: str
    transform_spec_sha256: str
    geometry_sha256: str
    input_pixel_sha256: str
    attempt_fingerprint: str
    reason: str
    evidence_ids: tuple[str, ...]
    parent_decision_id: str | None
    parent_decision_sha256: str | None
    next_strategy: str | None
    request_sha256: str


@dataclass(frozen=True)
class CoverageRecoveryDecision:
    decision_id: str
    request_id: str
    run_id: str
    origin_execution_id: str
    page_id: str
    page_source_sha256: str
    request_sha256: str
    component_id: str | None
    observation_ids: tuple[str, ...]
    anchor_polygon_page: Polygon | None
    anchor_sha256: str
    materialized_component_id: str | None
    attempt_kind: str
    transform_spec_sha256: str
    geometry_sha256: str
    input_pixel_sha256: str
    attempt_fingerprint: str
    attempt_id: str | None
    status: Literal["scheduled", "succeeded", "failed", "exhausted"]
    reason: str
    evidence_ids: tuple[str, ...]
    next_strategy: str | None
    decision_sha256: str


@dataclass(frozen=True)
class CoverageComponentInventoryEntry:
    component_id: str
    origin: Literal["discovery", "recovery_materialization"]
    introduced_by_decision_id: str | None
    anchor_sha256: str
    ordinal: int
    inventory_entry_sha256: str


@dataclass(frozen=True)
class CoverageEntry:
    component_id: str
    run_id: str
    origin_execution_id: str
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
    origin_execution_id: str
    page_id: str
    page_source_sha256: str
    inventory_version: int
    parent_ledger_sha256: str | None
    component_inventory: tuple[CoverageComponentInventoryEntry, ...]
    expected_observation_ids: tuple[str, ...]
    entries: tuple[CoverageEntry, ...]
    canonical_json_bytes: bytes
    sha256: str

    @property
    def expected_component_ids(self) -> tuple[str, ...]:
        return tuple(item.component_id for item in self.component_inventory)
```

Em `lifecycle.py`, codifique transições permitidas e proíba alterar identidade/hashes. Estados de reparo não são terminais. `CoverageRecoveryRequest/Decision` recalculam seus hashes, rejeitam identidade divergente e são os únicos tipos usados por Tasks 5, 6 e 9 — sem dict paralelo. Cada request/decision possui exatamente uma âncora: ou `component_id` não nulo, ou `component_id is None` com `observation_ids` não vazio e `anchor_polygon_page` page-space não nulo; misturar, omitir ou perder essa âncora é inválido. `anchor_sha256` é o hash do JSON canônico da alternativa escolhida: `component_id`, ou `observation_ids` ordenados + polígono page-space normalizado. A decision repete e valida `request_sha256`, âncora, `attempt_kind`, `transform_spec_sha256`, `geometry_sha256`, `input_pixel_sha256` e `attempt_fingerprint` da request; não pode resolver outra região que coincidentemente use os mesmos pixels. A request inicial tem `parent_decision_id/sha256=None`. Toda request sucessora repete identidade e âncora da request pai, aponta ao ID **e** hash da decisão `failed` anterior e usa `attempt_kind == parent.next_strategy`; request órfã, dois filhos para a mesma decisão, salto de estratégia ou hash de pai divergente é inválido. `succeeded` exige `next_strategy=None`; `failed` exige `next_strategy` não nulo e exatamente um filho posterior; `exhausted` exige `next_strategy=None`, encerra a execução com `CoverageRecoveryExhausted` e jamais autoriza owner graph/export. Assim a recuperação `unassociated_full_page_observation` é representável antes de existir componente e a cadeia preserva a mesma âncora. Migre `OWNER_STATES`, `POST_TRANSLATION_STATES`, `MASK_REQUIRED_STATES`, `EXECUTOR_REQUIRED_STATES` e validações relacionadas em `model.py` para derivar da única `CoverageState` e aceitar os estados canônicos; preserve aliases antigos somente para desserializar `legacy/shadow`. `enforce` não pode terminalizar em `review_required`.

O inventário é imutável **por snapshot**, porém versionado e append-only entre snapshots. A versão inicial contém, em ordem, todos os componentes de discovery. Um componente novo só pode ser anexado por uma `CoverageRecoveryDecision(status="succeeded")` que esteja ligada a uma request sem componente, preserve sua âncora/source input e declare o `materialized_component_id`; esse campo é nulo em decisões que não materializam componente. A nova ledger repete integralmente o prefixo anterior, incrementa `inventory_version` em um e liga `parent_ledger_sha256`. Cada `materialized_component_id` aparece exatamente uma vez no inventário e cada inventory entry de recovery resolve exatamente uma decision. Remover, substituir, reordenar ou anexar sem essa decision é inválido. IDs continuam derivados de `page_source_sha256 + polygon_page`, nunca do OCR.

**Step 4: Implementar validação de completude**

`require_complete()` exige:

- exatamente um entry por componente;
- o conjunto de `entries.component_id` é exatamente o inventário versionado `expected_component_ids`; o prefixo de discovery e cada extensão materializada são provados pela cadeia `parent_ledger_sha256`, portanto remover, substituir ou reordenar componente também falha;
- toda observação esperada após coverage aparece exatamente uma vez nos `observation_ids` dos entries ou em disposição explícita auditável;
- todo material possui tentativa OCR;
- todo diálogo possui owner;
- diálogo termina `final_verified`;
- preserve exige policy não-diálogo explícita;
- nenhuma identidade cruza run/origin-execution/página/hash; requests, invocations, observations e entries devem repetir a identidade do ledger.

**Step 5: Confirmar GREEN**

```powershell
Push-Location pipeline
try { .\venv\Scripts\python.exe -m pytest tests/test_owner_coverage_rescue.py tests/test_owner_lifecycle.py tests/test_owner_graph.py -q } finally { Pop-Location }
```

Expected: PASS.

**Step 6: Commit**

```powershell
$guard = '.codex-tmp\universal-source-replacement-plan-guard\TaskOwnedGit.ps1'
& $guard -Action Stage -Task 4
git diff --cached --check
git commit -m "feat: add page coverage and owner lifecycle ledger"
& $guard -Action VerifyAfterCommit -Task 4
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
    coverage = complete_page_coverage(
        page,
        run_id="run-a",
        origin_execution_id="exec-a",
        page_id="page_001",
        page_source_sha256=canonical_page_sha256(page),
        components=page.components,
        band_evidence=[],
    )
    assert coverage.ledger.entries[0].ocr_attempt_ids
    assert coverage.observations[0].component_ids == (page.components[0].component_id,)
    assert coverage.ocr_requests
    assert coverage.ocr_invocations
    assert {a.attempt_id for i in coverage.ocr_invocations for a in i.attempts} >= set(
        coverage.ledger.entries[0].ocr_attempt_ids
    )


def test_source_discovery_keeps_material_glyph_component_outside_all_bands():
    page = _page_with_glyphs_outside_all_bands()
    components = discover_source_text_components(
        page,
        page_id="page_001",
        detector_regions=[],
    )
    coverage = PageCoverageResult.initialize(
        run_id="run-a",
        origin_execution_id="exec-a",
        page_id="page_001",
        page_source_sha256=canonical_page_sha256(page),
        components=components,
    )
    assert len(coverage.components) == 1
    assert coverage.ledger.entries[0].materiality == "material"


def test_coverage_recovery_snapshot_never_drops_request_or_invocation_evidence():
    page = _page()
    first = complete_page_coverage(
        page,
        run_id="run-a",
        origin_execution_id="exec-a",
        page_id="page_001",
        page_source_sha256=canonical_page_sha256(page),
        components=_components(),
        band_evidence=[],
    )
    request, invocation, observations = _fresh_recovery_evidence_for(first)
    recovered = PageCoverageResult.build_from(
        first,
        observations=first.observations + observations,
        ocr_requests=first.ocr_requests + (request,),
        ocr_invocations=first.ocr_invocations + (invocation,),
    )
    assert {request.invocation_id for request in first.ocr_requests} <= {
        request.invocation_id for request in recovered.ocr_requests
    }
    assert {(item.invocation_id, item.attempt_chain_sha256) for item in first.ocr_invocations} <= {
        (item.invocation_id, item.attempt_chain_sha256) for item in recovered.ocr_invocations
    }
    assert recovered.sha256 != first.sha256


def test_coverage_recovery_history_keeps_resolved_requests_and_derives_pending_ids():
    first = _coverage_with_recovery_request("request-a")
    resolved = PageCoverageResult.build_from(
        first,
        recovery_requests=first.recovery_requests,
        recovery_decisions=first.recovery_decisions + (_successful_decision_for(first.recovery_requests[0]),),
    )
    assert resolved.recovery_requests == first.recovery_requests
    assert resolved.pending_request_ids == ()
    assert resolved.pending_requests == ()
    assert resolved.recovery_decisions[-1].request_sha256 == first.recovery_requests[0].request_sha256


def test_coverage_ledger_history_is_reopenable_and_hash_chained_after_restart():
    first = _coverage_with_discovery_ledger()
    second = _coverage_with_materialized_component_successor(first)
    reopened = PageCoverageResult.from_canonical_json_bytes(second.canonical_json_bytes)
    assert tuple(item.sha256 for item in reopened.ledger_history) == tuple(
        item.sha256 for item in second.ledger_history
    )
    assert reopened.ledger_history[0].parent_ledger_sha256 is None
    assert reopened.ledger_history[1].parent_ledger_sha256 == reopened.ledger_history[0].sha256
    reopened.require_ready_for_ownership()


def test_coverage_rejects_missing_ledger_predecessor_even_when_head_and_outer_hashes_are_recomputed():
    result = _coverage_with_three_ledger_versions()
    tampered = _drop_middle_ledger_and_recompute_head_and_outer_hashes(result)
    with pytest.raises(CoverageInvariantError, match="parent_ledger_sha256"):
        PageCoverageResult.from_canonical_json_bytes(tampered.canonical_json_bytes)


@pytest.mark.parametrize(
    "tamper",
    ["drop_observation", "drop_ocr_request", "drop_ocr_invocation", "drop_recovery_request", "reorder", "inject_stale"],
)
def test_coverage_build_from_rejects_dropped_reordered_or_stale_history(tamper):
    first = _coverage_with_two_monotonic_updates()
    with pytest.raises((CoverageIdentityError, CoverageInvariantError)):
        PageCoverageResult.build_from(first, **_tampered_successor_parts(first, tamper))


def test_coverage_readiness_requires_every_recovery_request_to_have_consistent_closure():
    request = _anchored_recovery_request("request-a")
    for result in (
        _coverage(recovery_requests=(request,), recovery_decisions=(), pending_request_ids=()),
        _coverage(recovery_requests=(request,), recovery_decisions=(_decision_for_other_request(),), pending_request_ids=()),
        _coverage(recovery_requests=(request,), recovery_decisions=(), pending_request_ids=(request.request_id,)),
    ):
        with pytest.raises(CoverageInvariantError):
            result.require_ready_for_ownership()


def test_failed_recovery_is_ready_only_after_one_hash_linked_successor_succeeds():
    first = _anchored_recovery_request("request-a")
    failed = _failed_decision(first, next_strategy="anchored_gray")
    successor = _successor_request(
        first,
        parent_decision=failed,
        request_id="request-b",
        attempt_kind="anchored_gray",
    )
    succeeded = _successful_decision_for(successor)
    result = _coverage_with_closed_recovery_chain(
        requests=(first, successor),
        decisions=(failed, succeeded),
    )
    assert successor.parent_decision_id == failed.decision_id
    assert successor.parent_decision_sha256 == failed.decision_sha256
    assert result.pending_request_ids == ()
    result.require_ready_for_ownership()


@pytest.mark.parametrize(
    "tamper",
    ["missing_child", "orphan_parent", "wrong_parent_hash", "wrong_strategy", "duplicate_child"],
)
def test_coverage_rejects_broken_recovery_successor_chain(tamper):
    result = _tampered_failed_recovery_chain(tamper)
    with pytest.raises(CoverageInvariantError):
        result.require_ready_for_ownership()


def test_exhausted_recovery_aborts_and_never_becomes_ready_even_with_empty_pending_view():
    result = _coverage_with_exhausted_recovery_and_recomputed_pending_ids()
    assert result.pending_request_ids == ()
    with pytest.raises(CoverageRecoveryExhausted):
        result.require_ready_for_ownership()


@pytest.mark.parametrize("boundary", ["build_from", "canonical_reopen"])
def test_coverage_rejects_duplicate_attempt_fingerprint_with_distinct_request_ids(boundary):
    first = _coverage_with_recovery_request("request-a")
    duplicate = _clone_recovery_request_with_new_ids_and_recomputed_hashes(
        first.recovery_requests[0],
        request_id="request-b",
        keep_attempt_fingerprint=True,
    )
    if boundary == "build_from":
        with pytest.raises(CoverageInvariantError, match="attempt_fingerprint"):
            PageCoverageResult.build_from(
                first,
                recovery_requests=first.recovery_requests + (duplicate,),
            )
        return
    tampered = _append_request_and_recompute_complete_canonical_coverage(first, duplicate)
    with pytest.raises(CoverageInvariantError, match="attempt_fingerprint"):
        PageCoverageResult.from_canonical_json_bytes(tampered).require_ready_for_ownership()


def test_coverage_rejects_evidence_from_other_origin_execution():
    coverage = _coverage(origin_execution_id="exec-off")
    stale = dataclasses.replace(coverage.observations[0], origin_execution_id="exec-other")
    with pytest.raises(CoverageIdentityError):
        PageCoverageResult.build_from(coverage, observations=(stale,))


@pytest.mark.parametrize("invalid_anchor", ["none", "both", "observation_without_polygon", "polygon_without_observation"])
def test_coverage_recovery_request_requires_exactly_one_component_or_observation_anchor(invalid_anchor):
    with pytest.raises(CoverageIdentityError):
        CoverageRecoveryRequest.build(**_invalid_recovery_anchor_parts(invalid_anchor))


def test_page_without_bands_never_builds_empty_observation_placeholder():
    result = _run_owner_control_plane_for_no_band_page()
    assert result.graph.observations
    assert result.graph.owners
```

**Step 2: Confirmar RED**

```powershell
Push-Location pipeline
try {
  $redOutput = & .\venv\Scripts\python.exe -m pytest `
    tests/test_vision_stack_ocr.py::PaddleBlockMappingTests::test_full_page_ocr_runs_with_empty_detector_blocks `
    tests/test_vision_stack_ocr.py::PaddleBlockMappingTests::test_full_page_attempt_hashes_exact_array_passed_to_provider `
    tests/test_vision_stack_ocr.py::PaddleBlockMappingTests::test_anchored_crop_attempt_hashes_crop_pixels_not_page_pixels `
    tests/test_vision_stack_ocr.py::PaddleBlockMappingTests::test_each_retry_variant_hashes_exact_provider_pixels `
    tests/test_source_component_discovery.py::test_source_discovery_keeps_material_glyph_component_outside_all_bands `
    tests/test_owner_coverage_rescue.py::test_component_without_band_gets_anchored_ocr_before_owner_resolution `
    tests/test_owner_coverage_rescue.py::test_coverage_recovery_snapshot_never_drops_request_or_invocation_evidence `
    tests/test_owner_coverage_rescue.py::test_coverage_recovery_history_keeps_resolved_requests_and_derives_pending_ids `
    tests/test_owner_coverage_rescue.py::test_coverage_ledger_history_is_reopenable_and_hash_chained_after_restart `
    tests/test_owner_coverage_rescue.py::test_coverage_rejects_missing_ledger_predecessor_even_when_head_and_outer_hashes_are_recomputed `
    tests/test_owner_coverage_rescue.py::test_coverage_build_from_rejects_dropped_reordered_or_stale_history `
    tests/test_owner_coverage_rescue.py::test_coverage_readiness_requires_every_recovery_request_to_have_consistent_closure `
    tests/test_owner_coverage_rescue.py::test_failed_recovery_is_ready_only_after_one_hash_linked_successor_succeeds `
    tests/test_owner_coverage_rescue.py::test_coverage_rejects_broken_recovery_successor_chain `
    tests/test_owner_coverage_rescue.py::test_exhausted_recovery_aborts_and_never_becomes_ready_even_with_empty_pending_view `
    tests/test_owner_coverage_rescue.py::test_coverage_rejects_duplicate_attempt_fingerprint_with_distinct_request_ids `
    tests/test_owner_coverage_rescue.py::test_coverage_rejects_evidence_from_other_origin_execution `
    tests/test_owner_coverage_rescue.py::test_coverage_recovery_request_requires_exactly_one_component_or_observation_anchor `
    tests/test_strip_owner_control_plane.py::test_page_without_bands_never_builds_empty_observation_placeholder `
    -q 2>&1
  $redExit = $LASTEXITCODE
  $redText = $redOutput | Out-String
  $redOutput
  if ($redExit -ne 1) { throw "Task 5 expected pytest exit 1 for a behavioral RED, got $redExit" }
  if ($redText -match '(?im)^ERROR(?:\s|:)|ERROR collecting|errors during collection|^E\s+fixture .* not found|UsageError|ModuleNotFoundError|ImportError') {
    throw 'Task 5 RED is structural rather than behavioral; create/fix the API skeleton and rerun the same nodeids'
  }
  if ($redText -notmatch '(?m)^FAILED\s+\S+::') { throw 'Task 5 RED did not report a targeted FAILED nodeid' }
} finally { Pop-Location }
```

Expected: early return com blocks vazios ou placeholder `observations=[]`.

**Step 3: Implementar `complete_page_coverage()`**

A autoridade retornada é:

```python
def complete_page_coverage(
    page_rgb: NDArray[np.uint8],
    *,
    run_id: str,
    origin_execution_id: str,
    page_id: str,
    page_source_sha256: str,
    components: tuple[SourceTextComponent, ...],
    band_evidence: Sequence[BandEvidence],
) -> "PageCoverageResult": ...


@dataclass(frozen=True)
class PageCoverageResult:
    run_id: str
    origin_execution_id: str
    page_id: str
    page_source_sha256: str
    ledger_history: tuple[PageCoverageLedger, ...]
    components: tuple[SourceTextComponent, ...]
    observations: tuple[TextObservation, ...]
    ocr_requests: tuple[OCRRequest, ...]
    ocr_invocations: tuple[OCRInvocationResult, ...]
    recovery_requests: tuple[CoverageRecoveryRequest, ...]
    recovery_decisions: tuple[CoverageRecoveryDecision, ...]
    pending_request_ids: tuple[str, ...]
    canonical_json_bytes: bytes
    sha256: str

    @classmethod
    def initialize(
        cls,
        *,
        run_id: str,
        origin_execution_id: str,
        page_id: str,
        page_source_sha256: str,
        components: tuple[SourceTextComponent, ...],
    ) -> "PageCoverageResult": ...

    @property
    def ledger(self) -> PageCoverageLedger: return self.ledger_history[-1]
    @property
    def entries(self) -> tuple[CoverageEntry, ...]: return self.ledger.entries
    @property
    def pending_requests(self) -> tuple[CoverageRecoveryRequest, ...]: ...  # derivada por pending_request_ids
    def entry(self, component_id: str) -> CoverageEntry: return self.ledger.entry(component_id)
    def entry_for_bbox(self, bbox: BBox) -> CoverageEntry: return self.ledger.entry_for_bbox(bbox)
    def require_ready_for_ownership(self) -> None: ...  # valida o snapshot inteiro, então a ledger
```

A função recebe a página original, identidade `run_id/origin_execution_id/page_id/page_source_sha256`, componentes descobertos e evidência opcional de bands. O builder inclui essa identidade no JSON/hash e rejeita qualquer request, invocation, observation, ledger, recovery request ou recovery decision divergente. `ledger_history` contém a ledger inicial e toda versão sucessora; cada ledger recalcula `canonical_json_bytes/sha256`, a primeira tem `parent_ledger_sha256=None` e cada posterior aponta exatamente ao hash do item anterior. A property `ledger` é somente a cabeça derivada. `build_from(previous, ...)` aceita somente sucessores monotônicos: cada tuple append-only mantém o prefixo byte/hash idêntico; mudança de ledger anexa uma única versão, nunca substitui a cabeça; request/decision não pode ser removida, substituída, reordenada nem injetada com identidade stale. `attempt_fingerprint` é globalmente único dentro do `PageCoverageResult`: duas requests com IDs/hashes externos diferentes mas o mesmo fingerprint são rejeitadas tanto no `build_from()` quanto no decoder/readiness, impedindo reexecução cíclica da mesma tentativa. Decisions de uma request também são append-only: zero ou uma `scheduled` mantém o request pending; exatamente uma decision terminal `succeeded|failed|exhausted` fecha aquela tentativa e deve vir depois de qualquer `scheduled`; segunda terminal, transição reversa ou decision sem request é inválida. `pending_request_ids` é exatamente a ordem das requests ainda sem decision terminal, mas isso não basta para readiness: uma `failed` só fecha sua obrigação quando existe exatamente uma request filha posterior ligada por `parent_decision_id/sha256`, mesma âncora e `attempt_kind == next_strategy`, e toda cadeia termina em `succeeded`; `exhausted` lança `CoverageRecoveryExhausted` mesmo que a visão pending esteja vazia. `require_ready_for_ownership()` valida o `PageCoverageResult` inteiro: reabre e verifica toda a hash-chain da ledger; não há pending IDs; toda decision resolve uma request preservada e repete request/anchor/attempt/input; cada cadeia de requests possui fechamento bem-formado e sucesso final; não há request/decision/invocation/observation órfã; os conjuntos ordenados de observations e componentes coincidem com a cabeça ledger/inventory; só então delega os invariantes terminais à ledger. Ela:

1. executa uma inferência full-page independente pelo boundary hash-bound da Task 2;
2. anexa observações por componente;
3. executa crops ancorados para componentes materiais ainda sem observação, preservando root hash, bbox e hash do crop físico;
4. usa variantes nativa, cinza, invertida e 2x somente quando necessário; cada ndarray efetivamente recebido pelo provider ganha `OCRAttempt` próprio, com parent/transform/hash/shape verificados;
5. registra toda tentativa no ledger; cache/heurística sem provider fica explícito e não conta como inferência física fresca;
6. devolve um `PageCoverageResult` frozen com ledger, components, observations, OCR requests/invocations, histórico completo de recovery requests/decisions e pending IDs derivados, sem mutar pixels nem perder evidência recebida.

**Step 4: Integrar antes do owner graph**

Em `run_chapter()`, faça coverage por página imediatamente após `discover_source_text_components(...)`. Bands podem acrescentar observações depois, mas não controlam a existência da página no OCR. Cada acréscimo produz um novo `PageCoverageResult` e preserva requests/invocations já registrados.

Remova o placeholder vazio em `_resolve_owner_page`. O resolver recebe sempre um único `PageCoverageResult` page-global, mesmo com zero bands; ledger, observações, requests e invocations não atravessam parâmetros paralelos.

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
  if ($LASTEXITCODE -ne 0) { throw "Task 5 GREEN failed: $LASTEXITCODE" }
} finally { Pop-Location }
```

Expected: PASS; todo componente material possui tentativa OCR independentemente de bands.

**Step 6: Commit**

```powershell
$guard = '.codex-tmp\universal-source-replacement-plan-guard\TaskOwnedGit.ps1'
& $guard -Action Stage -Task 5
git diff --cached --check
git commit -m "feat: complete OCR coverage before band execution"
& $guard -Action VerifyAfterCommit -Task 5
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
    initial = _coverage_result(components=(), observations=(observation,))
    result = recover_unassociated_observations(page, initial)
    assert len(result.components) == 1
    assert result.observations[0].component_ids == (result.components[0].component_id,)
    assert result.recovery_decisions[0].reason == "materialized_from_full_page_observation"
    assert result.ledger.expected_component_ids == (result.components[0].component_id,)
    assert result.ledger.component_inventory[0].introduced_by_decision_id == result.recovery_decisions[0].decision_id
    assert result.ledger.parent_ledger_sha256 == initial.ledger.sha256
    assert result.ocr_requests == initial.ocr_requests
    assert result.ocr_invocations == initial.ocr_invocations


def test_ambiguous_full_page_observation_requests_anchored_recovery():
    result = recover_unassociated_observations(
        _ambiguous_page(),
        _coverage_result(
            components=_two_close_components(),
            observations=(_observation("WAIT", component_ids=()),),
        ),
    )
    assert result.pending_requests
    request = result.pending_requests[0]
    assert request.component_id is None
    assert request.observation_ids == (_ambiguous_observation_id(),)
    assert request.anchor_polygon_page == _ambiguous_observation_polygon()
    assert not _suppressed_observation_ids(result)


def test_ocr_confirmed_component_recovers_container_before_graph_build():
    recovered = complete_container_coverage(
        _page(),
        _coverage_result(ledger=_ledger_with_observed_component_without_container()),
    )
    assert recovered.entries[0].container_id
    assert recovered.entries[0].state == "observed"


def test_unassociated_glyph_support_over_conservative_protection_still_materializes_component():
    result = recover_unassociated_observations(
        _page_with_conservative_protection_overlap(),
        _coverage_result(
            components=(),
            observations=(_observation("VISIBLE ENGLISH", component_ids=(), glyph_support=True),),
        ),
    )
    assert len(result.components) == 1
    entry = result.ledger.entry(result.components[0].component_id)
    assert entry.container_id
    assert entry.protection_conflict
    assert entry.state == "observed"
    assert not result.pending_requests


def test_two_unassociated_regions_on_same_page_have_distinct_recovery_fingerprints_and_both_close():
    initial = _coverage_result(
        components=(),
        observations=(
            _observation("FIRST", component_ids=(), bbox=(10, 20, 90, 50)),
            _observation("SECOND", component_ids=(), bbox=(210, 220, 330, 260)),
        ),
    )
    pending = recover_unassociated_observations(_page(), initial)
    assert len(pending.pending_requests) == 2
    assert len({request.anchor_sha256 for request in pending.pending_requests}) == 2
    assert len({request.attempt_fingerprint for request in pending.pending_requests}) == 2
    closed = _resolve_all_coverage_requests(_page(), pending)
    assert closed.pending_request_ids == ()
    assert len(closed.recovery_requests) == 2
    assert len(closed.recovery_decisions) >= 2
    assert len(closed.components) == 2
    closed.require_ready_for_ownership()
```

**Step 2: Confirmar RED**

```powershell
Push-Location pipeline
try {
  $redOutput = & .\venv\Scripts\python.exe -m pytest `
    tests/test_owner_coverage_rescue.py `
    tests/test_owner_container_evidence.py `
    tests/test_strip_owner_control_plane.py `
    -q 2>&1
  $redExit = $LASTEXITCODE
  $redText = $redOutput | Out-String
  $redOutput
  if ($redExit -ne 1) { throw "Task 6 expected pytest exit 1 for a behavioral RED, got $redExit" }
  if ($redText -match '(?im)^ERROR(?:\s|:)|ERROR collecting|errors during collection|^E\s+fixture .* not found|UsageError|ModuleNotFoundError|ImportError') {
    throw 'Task 6 RED is structural rather than behavioral; create/fix the API skeleton and rerun the same nodeids'
  }
  if ($redText -notmatch '(?m)^FAILED\s+\S+::') { throw 'Task 6 RED did not report a targeted FAILED nodeid' }
} finally { Pop-Location }
```

Expected: observação continua `unassociated_observation` ou componente confirmado continua `semantic_container_missing`.

**Step 3: Implementar recuperação determinística de associação**

Adicionar `recover_unassociated_observations(page, coverage: PageCoverageResult) -> PageCoverageResult` com esta ordem, sempre em page-space. Cada passagem cria snapshot novo, mantém OCR requests/invocations/observations e recovery requests/decisions anteriores como prefixos append-only e acrescenta somente decisões, requests ou evidência realmente produzidos:

1. interseção única forte com componente existente;
2. associação por distância/overlap dentro de container único;
3. crop OCR ancorado quando houver ambiguidade entre componentes existentes;
4. após orçamento finito de associação, materialização obrigatória de novo componente quando bbox/polígono OCR possui suporte de glyph positivo, mesmo se sobrepuser proteção conservadora;
5. marcar `protection_conflict=true` e guardar os IDs de evidência quando houver essa sobreposição, para que execution/R3 trate os pixels confirmados como texto sem desproteger arte externa;
6. `coverage_recovery_pending` existe apenas entre tentativas; não é terminal. Se glyph support material persistir sem associação, o passo 4 fecha coverage de forma monotônica.

Não descartar observação. IDs do componente materializado devem derivar de `page_source_sha256 + polygon_page`, nunca do texto reconhecido. A decision bem-sucedida é persistida antes da extensão do inventário; a nova ledger anexa uma única `CoverageComponentInventoryEntry` ligada por `introduced_by_decision_id`, incrementa a versão e referencia o hash da ledger anterior. O fingerprint da tentativa é `sha256(anchor_sha256 + attempt_kind + transform_spec_sha256 + geometry_sha256 + input_pixel_sha256)`: usa `component_id` quando existe ou os IDs/polígono da observação quando ainda não existe, evitando colisão entre duas regiões sem componente na mesma full page.

**Step 4: Antecipar recuperação de container**

Extrair o núcleo reutilizável de `recover_full_page_visual_container()` e chamar `complete_container_coverage(page, coverage: PageCoverageResult) -> PageCoverageResult` antes de `build_owner_page_graph_from_coverage()`. Ordem:

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
  if ($LASTEXITCODE -ne 0) { throw "Task 6 GREEN failed: $LASTEXITCODE" }
} finally { Pop-Location }
```

Expected: PASS; toda observação material possui componente e todo componente textual possui container executável antes do graph.

**Step 6: Commit**

```powershell
$guard = '.codex-tmp\universal-source-replacement-plan-guard\TaskOwnedGit.ps1'
& $guard -Action Stage -Task 6
git diff --cached --check
git commit -m "feat: recover page observations and containers before ownership"
& $guard -Action VerifyAfterCommit -Task 6
```

### Task 7: Construir o owner graph exclusivamente do ledger completo

**Files:**

- Modify: `pipeline/ownership/owner_builder.py`
- Modify: `pipeline/ownership/consensus_v2.py`
- Read only: `pipeline/ownership/reconcile.py`
- Modify: `pipeline/ownership/model.py`
- Modify: `pipeline/ownership/coverage.py`
- Modify: `pipeline/strip/run.py`
- Modify: `pipeline/strip/process_bands.py`
- Re-run only — do not edit/stage: `pipeline/tests/test_owner_reconcile.py`
- Re-run only — do not edit/stage: `pipeline/tests/test_owner_reconcile_properties.py`
- Test: `pipeline/tests/test_owner_coverage_rescue.py`
- Test: `pipeline/tests/test_strip_owner_control_plane.py`
- Create: `pipeline/tests/test_owner_builder_v2.py`

**Step 1: Escrever testes RED de disposição total**

```python
def test_zero_ocr_attempts_can_never_produce_no_ocr_non_text_suppression():
    ledger = _ledger(component=_material_component(), ocr_attempt_ids=())
    with pytest.raises(CoverageInvariantError, match="ocr attempt"):
        build_owner_page_graph_from_coverage(_coverage_result(ledger=ledger, observations=()))


def test_negative_ocr_can_preserve_only_after_explicit_non_dialogue_evidence():
    ledger = _ledger_with_negative_ocr_and_visual_non_text_evidence()
    graph = build_owner_page_graph_from_coverage(_coverage_result(ledger=ledger, observations=()))
    disposition = graph.component_dispositions[0]
    assert disposition.decision == "preserve"
    assert disposition.policy_id == "explicit_visual_non_text"


def test_fragments_in_one_balloon_produce_one_owner_with_one_complete_body():
    graph = build_owner_page_graph_from_coverage(
        _coverage_result(ledger=_fragmented_same_balloon_ledger(), observations=_observations())
    )
    assert len(graph.owners) == 1
    assert graph.owners[0].component_ids == _all_fragment_component_ids()
    assert graph.owners[0].source_payload == _complete_source_body()


def test_adjacent_containers_remain_separate_owners():
    graph = build_owner_page_graph_from_coverage(
        _coverage_result(ledger=_two_adjacent_balloon_ledger(), observations=_observations())
    )
    assert len(graph.owners) == 2
    assert all(owner.source_payload == _expected_body(owner.owner_id) for owner in graph.owners)


def test_short_english_word_inside_balloon_defaults_to_dialogue_not_sfx():
    graph = build_owner_page_graph_from_coverage(
        _coverage_result(ledger=_balloon_with_short_word("WAIT"), observations=_observations())
    )
    assert graph.owners[0].semantic_role == "dialogue"
    assert graph.component_dispositions[0].decision == "owned"


def test_credit_url_or_mark_preserve_requires_outside_dialogue_and_auditable_policy():
    with pytest.raises(CoverageInvariantError):
        build_owner_page_graph_from_coverage(
            _coverage_result(ledger=_credit_policy_inside_dialogue(), observations=_observations())
        )
    disposition = build_owner_page_graph_from_coverage(
        _coverage_result(
            ledger=_credit_outside_dialogue_with_policy_bbox_reason_and_evidence(),
            observations=_observations(),
        )
    ).component_dispositions[0]
    assert disposition.policy_id == "explicit_credit_outside_translatable_container"
    assert disposition.policy_bbox_page
    assert disposition.policy_evidence_ids


def test_component_disposition_policy_roundtrip_preserves_audit_fields():
    graph = build_owner_page_graph_from_coverage(
        _coverage_result(
            ledger=_credit_outside_dialogue_with_policy_bbox_reason_and_evidence(),
            observations=_observations(),
        )
    )
    loaded = OwnerGraph.from_dict(graph.to_dict(), enforce=True)
    assert loaded.component_dispositions[0] == graph.component_dispositions[0]


@pytest.mark.parametrize(("ledger", "observations"), _deterministic_complete_ledger_cases())
def test_graph_never_loses_a_material_component(ledger, observations):
    graph = build_owner_page_graph_from_coverage(
        _coverage_result(ledger=ledger, observations=observations)
    )
    disposed = {item.component_id for item in graph.component_dispositions}
    assert disposed == set(ledger.expected_component_ids)
```

**Step 2: Confirmar RED**

```powershell
Push-Location pipeline
try {
  $redOutput = & .\venv\Scripts\python.exe -m pytest `
    tests/test_owner_reconcile.py `
    tests/test_owner_reconcile_properties.py `
    tests/test_owner_builder_v2.py `
    tests/test_owner_coverage_rescue.py `
    tests/test_strip_owner_control_plane.py `
    -q 2>&1
  $redExit = $LASTEXITCODE
  $redText = $redOutput | Out-String
  $redOutput
  if ($redExit -ne 1) { throw "Task 7 expected pytest exit 1 for a behavioral RED, got $redExit" }
  if ($redText -match '(?im)^ERROR(?:\s|:)|ERROR collecting|errors during collection|^E\s+fixture .* not found|UsageError|ModuleNotFoundError|ImportError') {
    throw 'Task 7 RED is structural rather than behavioral; create/fix the API skeleton and rerun the same nodeids'
  }
  if ($redText -notmatch '(?m)^FAILED\s+\S+::') { throw 'Task 7 RED did not report a targeted FAILED nodeid' }
} finally { Pop-Location }
```

Expected: o ramo `no_ocr_evidence_non_text` ainda aceita ausência de tentativa e o resolver ainda aceita coleção vazia criada por band.

**Step 3: Criar a única entrada de reconciliação de produção**

Adicionar `build_owner_page_graph_from_coverage(coverage: PageCoverageResult)` em `owner_builder.py`. Ela deve:

- validar `coverage.require_ready_for_ownership()` sobre o snapshot completo, nunca apenas a ledger;
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
- não importar nem chamar o ramo legado de supressão sem tentativa em `reconcile.py:1071`;
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
    tests/test_owner_builder_v2.py `
    tests/test_owner_evidence.py `
    tests/test_owner_coverage_rescue.py `
    tests/test_strip_owner_control_plane.py `
    -q
  if ($LASTEXITCODE -ne 0) { throw "Task 7 GREEN failed: $LASTEXITCODE" }
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
    tests/test_owner_builder_v2.py `
    tests/test_owner_coverage_rescue.py `
    tests/test_owner_lifecycle.py `
    tests/test_strip_owner_control_plane.py `
    -q
  if ($LASTEXITCODE -ne 0) { throw "Checkpoint B failed: $LASTEXITCODE" }
} finally { Pop-Location }
```

Aceite somente se todos passarem e uma fixture sem bands produzir `translatable_components_without_owner == 0`.

**Step 7: Commit**

```powershell
$guard = '.codex-tmp\universal-source-replacement-plan-guard\TaskOwnedGit.ps1'
& $guard -Action Stage -Task 7
git diff --cached --check
git commit -m "fix: resolve ownership from complete page coverage"
& $guard -Action VerifyAfterCommit -Task 7
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
- Test: `pipeline/tests/test_translate_context.py`

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
    result = translate_owner_page((_owner(owner),), _valid_backend())
    assert [binding.owner_id for binding in result.bindings] == [owner.owner_id]


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


def test_every_binding_attempt_id_resolves_to_one_persistable_translation_attempt():
    binding, attempts = translate_owner(_owner_request(), backends=(_valid_backend(),))
    assert set(binding.attempt_ids) == {attempt.attempt_id for attempt in attempts}
    assert len({attempt.attempt_id for attempt in attempts}) == len(attempts)
    assert all(attempt.provider_called or attempt.cache_hit for attempt in attempts)


def test_bulk_translator_adapter_never_cross_assigns_owner_payloads():
    backend = _capturing_translate_pages_backend()
    result = translate_owner_page((_owner("owner-a"), _owner("owner-b")), backend)
    assert [call.owner_ids for call in backend.calls] == [("owner-a",), ("owner-b",)]
    assert [binding.owner_id for binding in result.bindings] == ["owner-a", "owner-b"]
    assert {attempt.owner_id for attempt in result.attempts} == {"owner-a", "owner-b"}


def test_public_translate_pages_signature_remains_compatible():
    parameters = inspect.signature(translate_module.translate_pages).parameters
    assert list(parameters) == [
        "ocr_results", "obra", "context", "glossario", "idioma_destino", "idioma_origem",
        "qualidade", "ollama_host", "ollama_model", "progress_callback", "models_dir",
        "translation_context",
    ]


def test_google_and_ollama_low_level_calls_are_owned_only_by_attempt_boundary():
    offenders = find_direct_calls_outside_function(
        Path("translator/translate.py"),
        called_names={"_translate_with_google", "_translate_with_ollama"},
        allowed_function="translate_one_owner_attempt",
    )
    assert offenders == []


def test_google_invalid_target_advances_to_forced_ollama_attempt_with_real_metadata(monkeypatch):
    google = _capturing_google_low_level(target="UNCHANGED SOURCE")
    ollama = _capturing_ollama_low_level(target="TRADUÇÃO VÁLIDA")
    monkeypatch.setattr(translate_module, "_translate_with_google", google)
    monkeypatch.setattr(translate_module, "_translate_with_ollama", ollama)
    result = translate_owner_page((_owner("owner-a"),), attempt_fn=translate_module.translate_one_owner_attempt)
    assert [attempt.backend for attempt in result.attempts] == ["google", "ollama"]
    assert [attempt.status for attempt in result.attempts] == ["rejected", "accepted"]
    assert [attempt.provider_called for attempt in result.attempts] == [True, True]
    assert all(attempt.provider_metadata_sha256 for attempt in result.attempts)
    assert result.bindings[0].owner_id == "owner-a"
    assert result.bindings[0].attempt_ids == tuple(attempt.attempt_id for attempt in result.attempts)
    google.assert_called_once()
    ollama.assert_called_once()
    assert google.received_context == ollama.received_context == _context()


def test_forced_ollama_attempt_never_hides_google_repair_inside_same_attempt(monkeypatch):
    ollama = _capturing_ollama_low_level(target="TRADUÇÃO LOCAL")
    monkeypatch.setattr(translate_module, "_translate_with_ollama", ollama)
    result = translate_module.translate_one_owner_attempt(
        _legacy_ocr_result_for("owner-a"),
        _obra(),
        _context(),
        _glossario(),
        **_attempt_kwargs(control=TranslationAttemptControl("ollama", "local", True)),
    )
    assert result.backend == "ollama"
    assert ollama.call_args.kwargs["repair_translator"] is None


@pytest.mark.parametrize("cache_kind", ["memory", "persistent"])
def test_disable_cache_bypasses_populated_cache_and_records_physical_provider_call(cache_kind):
    provider, cache = _provider_with_populated_cache(cache_kind)
    result = translate_one_owner_attempt(
        _legacy_ocr_result_for("owner-a"),
        _obra(),
        _context(),
        _glossario(),
        **_attempt_kwargs(control=TranslationAttemptControl("google", "primary", True)),
    )
    assert result.provider_called is True
    assert result.cache_hit is False
    provider.assert_called_once()


def test_enabled_cache_reports_hit_and_does_not_fake_provider_call():
    provider, cache = _provider_with_populated_cache("persistent")
    result = translate_one_owner_attempt(
        _legacy_ocr_result_for("owner-a"),
        _obra(),
        _context(),
        _glossario(),
        **_attempt_kwargs(control=TranslationAttemptControl("google", "primary", False)),
    )
    assert result.cache_hit is True
    assert result.provider_called is False
    provider.assert_not_called()


def test_provider_metadata_hash_is_derived_from_canonical_metadata_bytes():
    result = _real_google_attempt_result()
    assert result.provider_metadata_sha256 == sha256_bytes(result.provider_metadata_json_bytes)


def test_real_attempt_boundary_unavailable_records_attempts_and_aborts_before_execution(monkeypatch):
    monkeypatch.setattr(translate_module, "_translate_with_google", _raise_google_unavailable)
    monkeypatch.setattr(translate_module, "_translate_with_ollama", _raise_ollama_unavailable)
    with pytest.raises(TranslationInfrastructureError) as exc:
        translate_owner_page((_owner("owner-a"),), attempt_fn=translate_module.translate_one_owner_attempt)
    assert exc.value.attempts
    assert all(a.status == "operational_error" and not a.cache_hit for a in exc.value.attempts)
    assert not _owner_execution_was_called()
    assert not _public_export_exists()


def test_real_attempt_boundary_source_passthrough_exhausts_validation_without_binding(monkeypatch):
    monkeypatch.setattr(translate_module, "_translate_with_google", _return_source_text)
    monkeypatch.setattr(translate_module, "_translate_with_ollama", _return_source_text)
    with pytest.raises(TranslationValidationExhausted) as exc:
        translate_owner_page((_owner("owner-a"),), attempt_fn=translate_module.translate_one_owner_attempt)
    assert exc.value.attempts
    assert all(a.status == "rejected" for a in exc.value.attempts)
    assert not _owner_execution_was_called()


def test_translation_result_rejects_attempt_or_binding_from_other_origin_execution():
    result = _translation_result(origin_execution_id="exec-off")
    stale = dataclasses.replace(result.bindings[0], origin_execution_id="exec-other")
    with pytest.raises(TranslationIdentityError):
        OwnerPageTranslationResult.build(result.attempts, (stale,))


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
  $redOutput = & .\venv\Scripts\python.exe -m pytest `
    tests/test_translation_language_policy.py `
    tests/test_translation_locale_policy.py `
    tests/test_owner_translation.py `
    tests/test_translate_context.py `
    -q 2>&1
  $redExit = $LASTEXITCODE
  $redText = $redOutput | Out-String
  $redOutput
  if ($redExit -ne 1) { throw "Task 8 expected pytest exit 1 for a behavioral RED, got $redExit" }
  if ($redText -match '(?im)^ERROR(?:\s|:)|ERROR collecting|errors during collection|^E\s+fixture .* not found|UsageError|ModuleNotFoundError|ImportError') {
    throw 'Task 8 RED is structural rather than behavioral; create/fix the API skeleton and rerun the same nodeids'
  }
  if ($redText -notmatch '(?m)^FAILED\s+\S+::') { throw 'Task 8 RED did not report a targeted FAILED nodeid' }
} finally { Pop-Location }
```

Expected: frase inglesa inalterada é aceita pela validação atual ou o novo módulo não existe.

**Step 3: Implementar `TargetLanguageVerdict`**

```python
@dataclass(frozen=True)
class TargetLanguageVerdict:
    accepted: bool
    retryable: bool
    reason: str
    policy_id: str
    target_locale: Literal["pt-BR"]
    normalized_source_sha256: str
    normalized_target_sha256: str
    source_only_tokens: tuple[str, ...]
    english_token_ratio_ppm: int
    ptbr_token_ratio_ppm: int
    entities_equivalent: bool
    numbers_equivalent: bool
    placeholders_equivalent: bool
    canonical_json_bytes: bytes
    verdict_sha256: str
```

O builder normaliza/ordena sinais, usa inteiros ppm em vez de float instável, recalcula hashes e rejeita `accepted=True` quando qualquer check obrigatório falha. Attempts/bindings persistem o snapshot inteiro, não apenas o booleano/reason.

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

Crie em `translator/translate.py` o boundary real e controlável, reutilizado pelo entrypoint público:

```python
@dataclass(frozen=True)
class TranslationAttemptControl:
    backend: Literal["google", "ollama"]
    variant: str
    disable_cache: bool
    provider_model: str | None = None


@dataclass(frozen=True)
class FrozenTranslationOCRResult:
    owner_id: str
    canonical_json_bytes: bytes
    sha256: str

    def read(self) -> dict[str, JSONValue]: ...


@dataclass(frozen=True)
class TranslationProviderAttemptResult:
    translated_items: tuple[FrozenTranslationOCRResult, ...]
    backend: Literal["google", "ollama"]
    variant: str
    provider_model: str | None
    provider_called: bool
    cache_hit: bool
    provider_metadata_json_bytes: bytes
    provider_metadata_sha256: str


def translate_one_owner_attempt(
    ocr_result: dict[str, JSONValue],
    obra: str,
    context: dict[str, JSONValue],
    glossario: dict[str, str],
    *,
    idioma_destino: str,
    idioma_origem: str,
    qualidade: str,
    ollama_host: str,
    ollama_model: str,
    models_dir: str,
    translation_context: dict[str, JSONValue] | None,
    control: TranslationAttemptControl,
) -> TranslationProviderAttemptResult: ...
```

`translate_one_owner_attempt()` recebe exatamente um owner, força o backend/variant indicado e chama os produtores reais `_translate_with_google` ou `_translate_with_ollama`; não repassa pela seleção global baseada apenas no health-check. Ao forçar Ollama, chama `_translate_with_ollama(..., repair_translator=None)`: qualquer reparo Google vira outro attempt explícito, nunca resposta híbrida rotulada como Ollama. `disable_cache=True` bypassa leitura/escrita do cache para aquele attempt. O retorno deep-freezeia os itens e deriva metadata canônica do backend/model/request id/runtime/cache reais; indisponibilidade lança erro tipado com metadata parcial. `_resolve_translation_backend()` passa a escolher Ollama quando Google falhar e o serviço/modelo local estiver disponível. `translate_pages(...)` mantém exatamente a assinatura/retorno públicos atuais, mas delega sua execução selecionada a esse boundary e descongela uma cópia apenas no retorno legado. O adapter owner cria a sequência determinística de `TranslationAttemptControl` (Google primário/contextual, depois Ollama configurado/contextual), chama o boundary diretamente e reatacha a resposta por owner/hash, nunca por posição de lote. Assim um Google saudável que devolva inglês é rejeitado e a tentativa seguinte realmente seleciona Ollama, em vez de chamar Google novamente. Passthrough não é backend aceito pelo boundary owner.

Migre `test_translate_context.py`: expectativas que hoje exigem “Ollama nunca chamado + passthrough source” permanecem somente para modo legacy explicitamente nomeado. Para o boundary owner/enforce, Google indisponível deve selecionar Ollama quando disponível; ambos indisponíveis devem produzir erro operacional tipado, jamais source como target. Um guard AST impede qualquer chamada direta a `_translate_with_google/_translate_with_ollama` fora de `translate_one_owner_attempt()`.

Cada chamada materializa um `TranslationAttempt` frozen. Não reutilizar texto de outro owner nem fallback posicional.

Corrija `_translation_owners()` para selecionar por `TRANSLATION_ROUTE_ACTIONS` importado do modelo, em vez de comparar somente `translate_inpaint_render`. `translate_sfx_inpaint_render` e `translate_render_only` também exigem exatamente um binding; qualquer nova route adicionada à constante entra automaticamente no teste parametrizado.

**Step 5: Introduzir `TranslationBinding` imutável**

Campos mínimos:

```python
@dataclass(frozen=True)
class TranslationAttempt:
    attempt_id: str
    run_id: str
    origin_execution_id: str
    page_id: str
    page_source_sha256: str
    owner_id: str
    backend: str
    variant: str
    provider_model: str | None
    provider_metadata_json_bytes: bytes
    provider_metadata_sha256: str
    source_payload_sha256: str
    request_sha256: str
    response_sha256: str | None
    provider_called: bool
    cache_hit: bool
    status: Literal["accepted", "rejected", "operational_error"]
    error_code: str | None
    language_verdict: TargetLanguageVerdict | None
    attempt_sha256: str


@dataclass(frozen=True)
class TranslationBinding:
    run_id: str
    origin_execution_id: str
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


@dataclass(frozen=True)
class OwnerPageTranslationResult:
    run_id: str
    origin_execution_id: str
    page_id: str
    page_source_sha256: str
    attempts: tuple[TranslationAttempt, ...]
    bindings: tuple[TranslationBinding, ...]
    canonical_json_bytes: bytes
    sha256: str
```

`attempt_sha256`, `translation_binding_sha256` e `OwnerPageTranslationResult.sha256` são calculados de JSON canônico; nunca são fornecidos pelo backend. `TranslationAttempt` registra inclusive respostas rejeitadas e falhas operacionais, sem guardar segredo/credential, copia `backend/variant/provider_model/provider_metadata_json_bytes/provider_metadata_sha256/provider_called/cache_hit` do `TranslationProviderAttemptResult` real e recalcula o hash dos bytes sanitizados ao construir/desserializar. Ao serializar o attempt dentro de outro JSON, decodifique esses bytes canônicos para o campo objeto `provider_metadata` e recalcule os mesmos bytes/hash na leitura; nunca use base64 nem bytes Python dentro do payload externo. `translate_owner_page()` retorna somente `OwnerPageTranslationResult`, preservando attempts e bindings juntos. Cardinalidade: exatamente um binding aceito por owner traduzível, e todos os IDs de sua cadeia resolvem para attempts da mesma `run_id/origin_execution_id/page_id/page_source_sha256/owner_id`. `target_locale` deve ser exatamente `pt-BR` e o teste de binding deve assertá-lo; português genérico ou locale ausente não fecha o contrato. Resposta desconhecida, duplicada ou sem owner continua erro atômico. O entrypoint compatível `translate_pages(ocr_results, obra, context, glossario, idioma_destino="pt-BR", idioma_origem="en", qualidade=..., ollama_host=..., ollama_model=..., progress_callback=None, models_dir=..., translation_context=...)` preserva assinatura e usa internamente o mesmo boundary, enquanto o adapter owner chama o boundary por controle explícito. A Task 9 copia `attempts` para `PageExecutionResult.translation_attempts`; a Task 16 persiste `translation_attempts.jsonl` com metadata reabrível.

**Step 6: Confirmar GREEN**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest `
    tests/test_translation_language_policy.py `
    tests/test_translation_locale_policy.py `
    tests/test_owner_translation.py `
    tests/test_translate_context.py `
    tests/test_normalized_text_propagates_to_translation.py `
    -q
  if ($LASTEXITCODE -ne 0) { throw "Task 8 GREEN failed: $LASTEXITCODE" }
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
    tests/test_translate_context.py `
    -q
} finally { Pop-Location }
```

Aceite somente se todo owner traduzível possuir `TranslationBinding` PT-BR válido e nenhuma exceção implícita aceitar inglês.

**Step 8: Commit**

```powershell
$guard = '.codex-tmp\universal-source-replacement-plan-guard\TaskOwnedGit.ps1'
& $guard -Action Stage -Task 8
git diff --cached --check
git commit -m "feat: bind validated pt-BR translations to owners"
& $guard -Action VerifyAfterCommit -Task 8
```

### Task 9: Criar o coordenador page-first canônico

**Files:**

- Create: `pipeline/ownership/chapter_contract.py`
- Create: `pipeline/ownership/publication.py`
- Create: `pipeline/ownership/execution.py` (shell transacional de geração/pointer; Task 10 amplia a composição)
- Create: `pipeline/strip/page_pipeline.py`
- Create: `pipeline/tests/test_owner_lifecycle_e2e.py`
- Create: `pipeline/tests/test_chapter_source_manifest.py`
- Create: `pipeline/tests/test_chapter_publication.py`
- Modify: `pipeline/ownership/model.py`
- Modify: `pipeline/strip/types.py`
- Modify: `pipeline/strip/run.py`
- Modify: `pipeline/strip/process_bands.py`
- Modify: `pipeline/main.py`
- Modify: `pipeline/ownership/lifecycle.py`
- Modify: `pipeline/ownership/translation.py`
- Modify: `pipeline/extractor/extractor.py`
- Create: `pipeline/tests/test_page_owner_pipeline.py`
- Test: `pipeline/tests/test_extractor.py`
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
            request=_page_request(run_id="run-a", page_id="page_001", source_sha=_sha256("page-a")),
            translations=(_binding(run_id="run-b"),),
            page_commits=(_commit(page_id="page_002"),),
        )


def test_page_result_rejects_owner_graph_from_another_run_or_source_hash():
    request = _page_request(run_id="run-a", page_id="page_001", source_sha=_sha256("page-a"))
    with pytest.raises(PagePipelineIdentityError):
        PageExecutionResult.build(
            request=request,
            owner_graph=_owner_graph(run_id="run-b", page_source_sha256=_sha256("page-a")),
            **_page_result_parts_without_graph(),
        )
    with pytest.raises(PagePipelineIdentityError):
        PageExecutionResult.build(
            request=request,
            owner_graph=_owner_graph(run_id="run-a", page_source_sha256=_sha256("page-other")),
            **_page_result_parts_without_graph(),
        )


def test_page_result_rejects_physical_records_from_same_run_but_other_execution():
    parts = _page_result_parts(run_id="content-a", execution_id="exec-current")
    with pytest.raises(PagePipelineIdentityError):
        PageExecutionResult.build(
            **parts,
            page_commits=(dataclasses.replace(parts["page_commits"][0], execution_id="exec-other"),),
        )
    with pytest.raises(PagePipelineIdentityError):
        PageExecutionResult.build(
            **parts,
            page_geometry=dataclasses.replace(parts["page_geometry"], execution_id="exec-other"),
        )


def test_content_replay_requires_origin_execution_equal_replay_parent():
    parts = _replay_page_result_parts(replay_of_execution_id="exec-off")
    stale_graph = dataclasses.replace(parts["owner_graph"], origin_execution_id="exec-unrelated")
    with pytest.raises(PagePipelineIdentityError):
        PageExecutionResult.build(**parts, owner_graph=stale_graph)


def test_final_verified_status_requires_hash_linked_final_page_and_terminal_proof():
    with pytest.raises(PagePipelineStateError):
        PageExecutionResult.build(status="final_verified", terminal_proof=None, **_page_result_parts())
    with pytest.raises(PagePipelineIdentityError):
        PageExecutionResult.build(
            status="final_verified",
            final_page=_final_page(pixel_sha=_sha256("final")),
            terminal_proof=_proof(final_page_pixel_sha256=_sha256("other")),
            **_page_result_parts(),
        )


def test_candidate_result_cannot_enter_project_export_adapter():
    page = adapt_page_execution_result_to_output_page(_candidate_page_result())
    with pytest.raises(PageNotTerminalError):
        main._project_inputs_from_output_pages(
            _source_manifest_for_pages(1), [page], private_execution_root=_candidate_private_root()
        )


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


def test_chapter_source_manifest_uses_relative_paths_real_hashes_and_stable_order(tmp_path):
    source = _write_three_source_pages(tmp_path / "source")
    image_files, extraction_root = extractor.extract(source, tmp_path / "work")
    manifest = ChapterSourceManifest.from_extracted_pages(
        image_files,
        extraction_root,
        run_id="content-lineage-1",
        execution_id="execution-1",
    )
    assert [entry.page_id for entry in manifest.pages] == ["page_001", "page_002", "page_003"]
    assert all(not Path(entry.relative_source_path).is_absolute() for entry in manifest.pages)
    assert all(re.fullmatch(r"[0-9a-f]{64}", entry.source_file_sha256) for entry in manifest.pages)
    assert all(re.fullmatch(r"[0-9a-f]{64}", entry.page_source_sha256) for entry in manifest.pages)
    manifest_paths = tuple(extraction_root / entry.relative_source_path for entry in manifest.pages)
    assert manifest.source_tree_sha256 == canonical_source_tree_sha256(manifest_paths, extraction_root)
    assert manifest.source_page_count == 3


@pytest.mark.parametrize("input_kind", ["directory", "archive", "single_image"])
def test_source_manifest_preserves_exact_extractor_result_for_every_supported_input(tmp_path, input_kind):
    extracted = _extract_test_input(tmp_path, input_kind, ordered_names=("001.png", "002.png", "010.png"))
    manifest = ChapterSourceManifest.from_extracted_pages(
        extracted.image_files,
        extracted.extraction_root,
        **_run_identity(),
    )
    assert [entry.relative_source_path for entry in manifest.pages] == extracted.relative_paths
    assert manifest.source_page_count == len(extracted.image_files)


@pytest.mark.parametrize("input_kind", ["directory", "archive"])
def test_extractor_preserves_nested_duplicate_basenames_without_overwrite(tmp_path, input_kind):
    source = _input_with_nested_pages(tmp_path, input_kind, ("chapter-a/001.png", "chapter-b/001.png"))
    image_files, extraction_root = extractor.extract(source, tmp_path / "work")
    assert [path.relative_to(extraction_root).as_posix() for path in image_files] == [
        "chapter-a/001.png",
        "chapter-b/001.png",
    ]
    assert len({sha256_file(path) for path in image_files}) == 2
    manifest = ChapterSourceManifest.from_extracted_pages(image_files, extraction_root, **_run_identity())
    assert manifest.source_page_count == 2


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
    source_manifest = _source_manifest_for_pages(1)
    output_pages = _run_minimal_chapter(
        owner_graph_mode="enforce",
        services=_verified_terminal_fixture_services(),
    )
    assert all(isinstance(page, OutputPage) for page in output_pages)
    assert all(page.owner_page_evidence_ref is not None for page in output_pages)
    assert all(page.owner_page_result is not None for page in output_pages)
    generation_root = _private_root_for(output_pages[0].owner_page_evidence_ref)
    authoritative_results = tuple(page.owner_page_evidence_ref.read_verified(generation_root) for page in output_pages)
    project_inputs = main._project_inputs_from_output_pages(
        source_manifest, output_pages, private_execution_root=generation_root
    )
    assert isinstance(project_inputs, VerifiedProjectInputs)
    assert project_inputs.source_manifest_sha256 == source_manifest.sha256
    assert tuple(
        (item.final_artifact.artifact_store_id, item.final_artifact.generation_id, item.final_artifact.relative_path)
        for item in project_inputs.pages
    ) == tuple(
        (
            result.final_page.artifact_ref.artifact_store_id,
            result.final_page.artifact_ref.generation_id,
            result.final_page.artifact_ref.relative_path,
        )
        for result in authoritative_results
    )
    assert tuple(item.page_result_sha256 for item in project_inputs.pages) == tuple(
        result.result_sha256 for result in authoritative_results
    )
    assert all(item.ocr_result.read()["_owner_graph_mode"] == "enforce" for item in project_inputs.pages)
    assert project_inputs.sha256 == sha256_bytes(project_inputs.canonical_json_bytes)


def test_multi_page_private_execution_uses_one_marker_and_distinct_atomic_page_generations(tmp_path):
    private_execution_root, output_pages = _persist_three_verified_output_pages_in_one_execution(tmp_path)
    refs = tuple(page.owner_page_evidence_ref for page in output_pages)
    assert len({ref.artifact_store_id for ref in refs}) == 1
    assert len({ref.generation_id for ref in refs}) == 1
    assert len({ref.page_generation_id for ref in refs}) == 3
    assert _artifact_generation_markers(private_execution_root) == (
        private_execution_root / "artifact_generation.json",
    )
    results = tuple(ref.read_verified(private_execution_root) for ref in refs)
    assert all(
        asset.generation_id == refs[0].generation_id
        for result in results
        for asset in result.project_asset_refs
    )
    inputs = main._project_inputs_from_output_pages(
        _source_manifest_for_pages(3),
        output_pages,
        private_execution_root=private_execution_root,
    )
    assert inputs.artifact_store_id == refs[0].artifact_store_id
    assert inputs.generation_id == refs[0].generation_id
    assert tuple(page.page_id for page in inputs.pages) == tuple(page.page_id for page in results)


@pytest.mark.parametrize(
    "tamper",
    ["mixed_common_generation", "duplicate_page_generation_id", "swapped_current_pointer"],
)
def test_multi_page_project_adapter_rejects_mixed_generation_or_page_pointer_identity(tmp_path, tamper):
    private_execution_root, output_pages = _persist_three_verified_output_pages_in_one_execution(tmp_path)
    _tamper_multi_page_private_evidence_and_recompute_outer_hashes(
        private_execution_root,
        output_pages,
        tamper,
    )
    with pytest.raises(PageArtifactIntegrityError):
        main._project_inputs_from_output_pages(
            _source_manifest_for_pages(3),
            output_pages,
            private_execution_root=private_execution_root,
        )


def test_three_page_private_generations_publish_and_reopen_portably_end_to_end(tmp_path):
    private_execution_root, source_manifest, requests = _three_page_private_execution_requests(tmp_path)
    marker = _load_generation_marker(private_execution_root)
    output_pages = []
    private_semantic_hashes = {}
    for ordinal, request in enumerate(requests, 1):
        result = _verified_result_for_request(request, private_execution_root)
        ref = PageCandidateTransaction(
            private_execution_root=private_execution_root,
            run_id=request.run_id,
            execution_id=request.execution_id,
            page_id=request.page_id,
            artifact_store_id=marker.artifact_store_id,
            generation_id=marker.generation_id,
            page_generation_id=f"page-generation-{ordinal}",
            transaction_id=f"page-transaction-{ordinal}",
        ).commit_verified_generation(result)
        private_semantic_hashes[request.page_id] = result.page_content_semantic_sha256
        output_pages.append(adapt_page_execution_result_to_output_page(result, evidence_ref=ref))

    source_inputs = main._project_inputs_from_output_pages(
        source_manifest,
        output_pages,
        private_execution_root=private_execution_root,
    )
    bundle = main._wrap_up_verified_owner_pages(
        source_inputs,
        source_private_execution_root=private_execution_root,
    )
    public_root = tmp_path / "published"
    publication = PublicationTransaction.for_bundle(public_root, bundle)
    publication.stage(bundle)
    publication.commit(bundle)

    moved_root = tmp_path / "published-moved"
    shutil.move(public_root, moved_root)
    shutil.rmtree(private_execution_root)
    reopened = reopen_verified_publication(moved_root)
    assert tuple(page.page_id for page in reopened.verified_inputs.pages) == tuple(
        page.page_id for page in source_manifest.pages
    )
    assert len(reopened.verified_inputs.pages) == 3
    assert _artifact_generation_markers(moved_root) == (moved_root / "artifact_generation.json",)
    assert reopened.verified_inputs.artifact_store_id == reopened.receipt.artifact_store_id
    assert reopened.verified_inputs.generation_id == reopened.receipt.generation_id
    assert len({page.final_artifact.relative_path for page in reopened.verified_inputs.pages}) == 3
    for page, export_page in zip(reopened.verified_inputs.pages, reopened.export_manifest.pages, strict=True):
        result = page.page_execution_evidence.read_verified(moved_root)
        assert result.page_content_semantic_sha256 == private_semantic_hashes[page.page_id]
        assert export_page.page_result_sha256 == result.result_sha256 == page.page_result_sha256
        assert export_page.page_execution_evidence_sha256 == page.page_execution_evidence.sha256
        assert export_page.final_pixel_sha256 == result.final_page.page_output_pixel_sha256
        assert export_page.translated_path == result.final_page.artifact_ref.relative_path
        assert all(ref.artifact_store_id == reopened.receipt.artifact_store_id for ref in result.project_asset_refs)
        assert all(ref.generation_id == reopened.receipt.generation_id for ref in result.project_asset_refs)


def test_verified_project_input_preserves_complete_page_journal_after_output_page_is_discarded(tmp_path):
    generation_root = _execution_generation_root(tmp_path)
    result = _verified_page_result(generation_root)
    expected = result.to_canonical_dict()
    evidence_ref = _persist_page_result_and_current_pointer(result, generation_root)
    page = adapt_page_execution_result_to_output_page(result, evidence_ref=evidence_ref)
    inputs = main._project_inputs_from_output_pages(
        _source_manifest_for_pages(1), [page], private_execution_root=generation_root
    )
    del page, result
    reopened = inputs.pages[0].page_execution_evidence.read_verified(generation_root)
    assert reopened.to_canonical_dict() == expected
    assert reopened.result_sha256 == inputs.pages[0].page_result_sha256
    bundle = main._wrap_up_verified_owner_pages(inputs, source_private_execution_root=generation_root)
    assert _published_evidence_records(bundle, "page_001") == _expected_evidence_records_from(expected)


def test_project_adapter_uses_persisted_pointer_when_live_result_is_deleted_or_corrupted(tmp_path):
    generation_root = _execution_generation_root(tmp_path)
    result = _verified_page_result(generation_root)
    expected_hash = result.result_sha256
    evidence_ref = _persist_page_result_and_current_pointer(result, generation_root)
    for live_view in (None, _corrupted_live_result_with_original_english(result)):
        page = adapt_page_execution_result_to_output_page(result, evidence_ref=evidence_ref)
        page.owner_page_result = live_view
        inputs = main._project_inputs_from_output_pages(
            _source_manifest_for_pages(1), [page], private_execution_root=generation_root
        )
        assert inputs.pages[0].page_result_sha256 == expected_hash
        assert not _source_english_visible(inputs.pages[0].final_artifact.load_verified(generation_root))


@pytest.mark.parametrize(
    "tamper",
    [
        "missing_pointer",
        "pointer_hash",
        "pointer_traversal",
        "pointer_symlink_escape",
        "snapshot_file",
        "snapshot_hash",
        "snapshot_symlink_escape",
        "nfc_casefold_collision",
    ],
)
def test_project_adapter_rejects_missing_or_tampered_page_evidence_pointer_before_using_live_result(tmp_path, tamper):
    generation_root = _execution_generation_root(tmp_path)
    result = _verified_page_result(generation_root)
    evidence_ref = _persist_page_result_and_current_pointer(result, generation_root)
    page = adapt_page_execution_result_to_output_page(result, evidence_ref=evidence_ref)
    _tamper_page_evidence_pointer_or_snapshot(generation_root, evidence_ref, tamper)
    with pytest.raises(PageArtifactIntegrityError):
        main._project_inputs_from_output_pages(
            _source_manifest_for_pages(1), [page], private_execution_root=generation_root
        )


@pytest.mark.parametrize(
    ("tamper", "section"),
    [
        ("missing", "coverage"), ("missing", "ocr_requests"), ("missing", "ocr_invocations"),
        ("missing", "owner_graph"), ("missing", "translation_attempts"), ("missing", "translations"),
        ("missing", "repair_requests"), ("missing", "repair_history"), ("missing", "page_commits"),
        ("missing", "owner_target_materializations"), ("missing", "page_composition"),
        ("missing", "final_qa_ocr_requests"), ("missing", "final_qa_ocr_invocations"),
        ("missing", "language_residual_issues"), ("missing", "qa_probes"),
        ("missing", "final_replacement_verdicts"), ("missing", "final_page"), ("missing", "terminal_proof"),
        ("subhash", "repair_budget_policy"), ("subhash", "replacement_verification_policy"),
        ("extra", "unknown_top_level"), ("subhash", "coverage"), ("identity", "owner_graph"),
    ],
)
def test_project_adapter_rejects_incomplete_or_tampered_page_execution_evidence(tamper, section):
    result, generation_root, evidence_ref = _persisted_verified_page_result()
    page = adapt_page_execution_result_to_output_page(result, evidence_ref=evidence_ref)
    _tamper_persisted_page_evidence_and_recompute_outer_hashes(
        generation_root, evidence_ref, tamper=tamper, section=section
    )
    with pytest.raises(PageArtifactIntegrityError):
        main._project_inputs_from_output_pages(
            _source_manifest_for_pages(1), [page], private_execution_root=generation_root
        )


def test_page_execution_evidence_json_contains_refs_and_hashes_never_raster_bytes(tmp_path):
    small_root, large_root = _two_generation_roots_with_same_records_and_different_page_sizes(tmp_path)
    small = PageExecutionEvidenceSnapshot.build(_verified_page_result(small_root))
    large = PageExecutionEvidenceSnapshot.build(_verified_large_page_result(large_root))
    for snapshot in (small, large):
        payload = json.loads(snapshot.canonical_json_bytes)
        assert "lossless_png_bytes" not in _recursive_keys(payload)
        assert b"\x89PNG" not in snapshot.canonical_json_bytes
        assert not _contains_base64_or_absolute_path(payload)
        assert _all_artifact_refs_are_relative_and_root_bound(payload)
    assert abs(len(large.canonical_json_bytes) - len(small.canonical_json_bytes)) < 512


@pytest.mark.parametrize(
    ("tamper", "section"),
    [("missing", "coverage"), ("extra", "unknown_top_level"), ("subhash", "owner_graph"), ("identity", "translation_attempts")],
)
def test_page_execution_evidence_read_verified_rejects_schema_subhash_or_identity_tamper(tmp_path, tamper, section):
    generation_root = _execution_generation_root(tmp_path)
    evidence = PageExecutionEvidenceSnapshot.build(_verified_page_result(generation_root))
    corrupted = _tamper_snapshot_canonical_payload_and_recompute_outer_sha(evidence, tamper, section)
    with pytest.raises(PageArtifactIntegrityError):
        corrupted.read_verified(generation_root)


def test_candidate_evidence_reserves_final_schema_sections_without_future_top_level_extension(tmp_path):
    generation_root = _execution_generation_root(tmp_path)
    snapshot = PageExecutionEvidenceSnapshot.build(_candidate_page_result(generation_root))
    payload = json.loads(snapshot.canonical_json_bytes)
    assert set(_reserved_task_10_to_17_sections()) <= set(payload)
    assert payload["repair_requests"] == payload["repair_history"] == []
    assert payload["owner_target_materializations"] == payload["qa_probes"] == []
    assert payload["final_qa_ocr_requests"] == payload["final_qa_ocr_invocations"] == []
    assert payload["language_residual_issues"] == []
    assert payload["final_replacement_verdicts"] == []
    assert payload["replay_source_page_evidence_sha256"] is None


@pytest.mark.parametrize(
    "bad_path",
    ["C:/outside/page.png", "/outside/page.png", "../escape.png", "images/../escape.png", "images\\page.png"],
)
def test_artifact_ref_rejects_absolute_traversal_or_noncanonical_separator(tmp_path, bad_path):
    with pytest.raises(ArtifactPathError):
        _persisted_ref(relative_path=bad_path, generation_root=tmp_path)


def test_artifact_generation_rejects_casefold_collision(tmp_path):
    with pytest.raises(ArtifactPathCollisionError):
        _build_generation_refs(tmp_path, relative_paths=("images/Page.png", "images/page.png"))


def test_artifact_generation_rejects_nfc_casefold_collision(tmp_path):
    with pytest.raises(ArtifactPathCollisionError):
        _build_generation_refs(
            tmp_path,
            relative_paths=("images/Café.png", "images/CAFE\u0301.PNG"),
        )


def test_artifact_ref_rejects_symlink_junction_or_reparse_escape_before_open(tmp_path):
    generation_root, outside, relative_escape = _generation_with_supported_platform_escape_link(tmp_path)
    sentinel = outside / "must-not-be-read.png"
    with _spy_file_opens_under(outside) as outside_opens:
        with pytest.raises(ArtifactPathError):
            _persisted_ref(relative_path=relative_escape, generation_root=generation_root).load_verified_bytes(
                generation_root
            )
    assert outside_opens == ()
    assert sentinel.read_bytes() == _outside_sentinel_bytes()


def test_output_page_compatibility_views_cannot_override_authoritative_result():
    source_manifest = _source_manifest_for_pages(1)
    result, generation_root, evidence_ref = _persisted_verified_page_result()
    page = adapt_page_execution_result_to_output_page(result, evidence_ref=evidence_ref)
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
    inputs = main._project_inputs_from_output_pages(
        source_manifest, [page], private_execution_root=generation_root
    )
    authoritative = evidence_ref.read_verified(generation_root)
    assert _target_payloads(inputs) == _authoritative_target_payloads(authoritative)
    assert tuple(item.final_artifact.relative_path for item in inputs.pages) == (
        authoritative.final_page.artifact_ref.relative_path,
    )
    assert tuple(item.page_geometry.read() for item in inputs.pages) == (
        _authoritative_page_bounds(authoritative),
    )


@pytest.mark.parametrize("tamper", ["missing", "duplicate", "extra", "reordered", "source_hash"])
def test_project_adapter_rejects_page_set_not_bijective_with_source_manifest(tamper):
    manifest = _source_manifest_for_pages(3)
    pages = _tampered_output_pages_for_manifest(manifest, tamper)
    with pytest.raises(ChapterCardinalityError):
        main._project_inputs_from_output_pages(
            manifest, pages, private_execution_root=_private_execution_root_for_pages(pages)
        )


def test_source_manifest_is_built_before_any_public_output_path_is_created(tmp_path):
    calls = []
    with _spy_recovery_extract_and_public_mkdir(calls):
        main._run_pipeline(str(_write_verified_pipeline_config(tmp_path)))
    assert calls.index("recover_publication") < calls.index("extract_source_manifest")
    assert calls.index("extract_source_manifest") < calls.index("first_public_path_write")


def test_real_run_pipeline_dispatches_verified_bundle_once_and_ignores_every_mutable_view(self):
    with tempfile.TemporaryDirectory() as raw_dir:
        tmp_path = Path(raw_dir)
        private_result, private_execution_root, evidence_ref = _persisted_verified_page_result(tmp_path)
        page = adapt_page_execution_result_to_output_page(private_result, evidence_ref=evidence_ref)
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
        source_manifest = _manifest_passed_to_run_chapter(run_chapter)
        adapter.assert_called_once_with(
            source_manifest, output_pages, private_execution_root=private_execution_root
        )
        wrap_up.assert_called_once()
        self.assertEqual(
            wrap_up.call_args.kwargs["source_private_execution_root"],
            private_execution_root,
        )
        self.assertTrue(all(spy.call_count == 0 for spy in forbidden))
        self.assertEqual(build_project.call_count, 1)
        self.assertIsNone(build_project.call_args.kwargs.get("output_pages"))
        source_verified_inputs = wrap_up.call_args.args[0]
        self.assertIsInstance(source_verified_inputs, VerifiedProjectInputs)
        self.assertFalse(_contains_instance(wrap_up.call_args, OutputPage))
        self.assertFalse(_contains_instance(build_project.call_args, OutputPage))
        self.assertTrue(_contains_instance(build_project.call_args, VerifiedProjectInputs))
        self.assertEqual(source_verified_inputs.sha256, sha256_bytes(source_verified_inputs.canonical_json_bytes))
        public_root = _configured_output_root(config_path)
        published = reopen_verified_publication(public_root)
        published_inputs = published.verified_inputs
        published_page = published_inputs.pages[0]
        published_result = published_page.page_execution_evidence.read_verified(public_root)
        manifest = load_json(public_root / "export_manifest.json")
        export_page = manifest["pages"][0]
        translated = public_root / export_page["translated_path"]
        project = load_json(public_root / "project.json")
        self.assertEqual(translated.read_bytes(), published_result.final_page.lossless_png_bytes)
        self.assertEqual(sha256_file(translated), published_result.final_page.final_file_sha256)
        self.assertEqual(canonical_page_sha256(load_rgb(translated)), published_result.final_page.page_output_pixel_sha256)
        self.assertEqual(_project_target_payloads(project), _authoritative_target_payloads(published_result))
        self.assertEqual(_project_geometry(project), published_result.page_geometry.read())
        self.assertEqual(export_page["page_result_sha256"], published_result.result_sha256)
        self.assertEqual(export_page["page_execution_evidence_sha256"], published_page.page_execution_evidence.sha256)
        self.assertEqual(export_page["page_content_semantic_sha256"], private_result.page_content_semantic_sha256)
        self.assertEqual(published_result.page_content_semantic_sha256, private_result.page_content_semantic_sha256)
        self.assertNotEqual(published_result.result_sha256, private_result.result_sha256)
        self.assertEqual(manifest["source_manifest_sha256"], source_manifest.sha256)
        self.assertTrue(_all_project_asset_paths_resolve_and_match_hashes(public_root, project, manifest))
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
            self.assertEqual(_snapshot_publication_tree(public_root), {})
            self.assertFalse(_chapter_publication_staging(public_root).exists())


def test_chapter_publication_rolls_back_whole_asset_tree_at_every_wal_checkpoint(self):
    for existing_publication in (False, True):
        for fault_phase in _every_before_and_after_backup_and_promote_checkpoint():
            with self.subTest(existing=existing_publication, phase=fault_phase), tempfile.TemporaryDirectory() as raw_dir:
                tmp_path = Path(raw_dir)
                config_path = _write_verified_pipeline_config(tmp_path)
                public_root = _configured_output_root(config_path)
                before = _seed_or_snapshot_complete_publication_tree(public_root, existing=existing_publication)
                page = _persisted_verified_output_page(tmp_path)
                with patch.object(strip_run, "run_chapter", return_value=[page]), patch.object(
                    main.PublicationTransaction,
                    "_fault_checkpoint",
                    side_effect=_raise_on_publication_phase(fault_phase),
                ):
                    with self.assertRaises(InjectedPublicationFailure):
                        main._run_pipeline(str(config_path))
                self.assertEqual(_snapshot_complete_publication_tree(public_root), before)
                self.assertFalse(_chapter_publication_staging(public_root).exists())
                self.assertFalse(_chapter_publication_backup(public_root).exists())
                self.assertFalse(_chapter_publication_journal(public_root).exists())


def test_recover_chapter_publication_restores_interrupted_journal(self):
    for existing_publication in (False, True):
        for interrupted_phase in _every_wal_prefix_for_every_asset_target():
            with self.subTest(existing=existing_publication, phase=interrupted_phase), tempfile.TemporaryDirectory() as raw_dir:
                public_root = Path(raw_dir)
                before = _seed_or_snapshot_complete_publication_tree(public_root, existing=existing_publication)
                _simulate_process_death_with_publication_journal(public_root, interrupted_phase)
                main.recover_chapter_publication(public_root)
                self.assertEqual(_snapshot_complete_publication_tree(public_root), before)
                self.assertFalse(_chapter_publication_staging(public_root).exists())
                self.assertFalse(_chapter_publication_backup(public_root).exists())
                self.assertFalse(_chapter_publication_journal(public_root).exists())


def test_restore_failure_keeps_wal_and_backup_then_second_recovery_is_idempotent(tmp_path):
    public_root = tmp_path / "out"
    before = _seed_or_snapshot_complete_publication_tree(public_root, existing=True)
    _simulate_process_death_with_publication_journal(public_root, "after_first_backup_done")
    with _fail_first_restore_rename():
        with pytest.raises(PublicationRecoveryError):
            recover_chapter_publication(public_root)
    assert _chapter_publication_journal(public_root).exists()
    assert _chapter_publication_backup(public_root).exists()
    recover_chapter_publication(public_root)
    assert _snapshot_complete_publication_tree(public_root) == before
    assert not _chapter_publication_journal(public_root).exists()


def test_project_adapter_rejects_tampered_persisted_final_artifact():
    result, generation_root, evidence_ref = _persisted_verified_page_result()
    page = adapt_page_execution_result_to_output_page(result, evidence_ref=evidence_ref)
    _overwrite_file(
        generation_root / result.final_page.artifact_ref.relative_path,
        _original_english_png_bytes(),
    )
    with pytest.raises(PageArtifactIntegrityError):
        main._project_inputs_from_output_pages(
            _source_manifest_for_pages(1), [page], private_execution_root=generation_root
        )


@pytest.mark.parametrize("stage", ["original", "cleanup_base", "inpaint", "typeset", "page_composition"])
def test_project_adapter_rejects_missing_or_tampered_persisted_stage_ref(stage):
    result, generation_root, evidence_ref = _persisted_verified_page_result(stage_tamper=stage)
    page = adapt_page_execution_result_to_output_page(result, evidence_ref=evidence_ref)
    with pytest.raises(PageArtifactIntegrityError):
        main._project_inputs_from_output_pages(
            _source_manifest_for_pages(1), [page], private_execution_root=generation_root
        )


@pytest.mark.parametrize(
    "tamper",
    ["source_manifest", "asset_manifest", "project", "export_manifest", "publication_receipt"],
)
def test_bundle_reopen_rejects_tampered_detached_manifest_or_receipt(tmp_path, tamper):
    published = _publish_verified_complete_bundle(tmp_path)
    _tamper_control_or_payload(published, tamper)
    with pytest.raises(PublicationIntegrityError):
        reopen_verified_publication(published)


def test_publication_receipt_hash_dag_has_no_self_reference(tmp_path):
    published = _publish_verified_complete_bundle(tmp_path)
    receipt = _load_receipt(published)
    controls = {
        "chapter_source_manifest.json",
        "chapter_asset_manifest.json",
        "export_manifest.json",
        "publication_receipt.json",
    }
    assert controls.isdisjoint(_load_asset_manifest(published).relative_paths)
    assert receipt.receipt_sha256 == canonical_json_sha256(receipt.payload_without_receipt_sha256())
    assert receipt.source_manifest_file_sha256 == sha256_file(published / "chapter_source_manifest.json")
    assert receipt.asset_manifest_file_sha256 == sha256_file(published / "chapter_asset_manifest.json")
    assert receipt.verified_inputs_file_sha256 == sha256_file(published / "evidence/verified_project_inputs.json")
    assert receipt.export_manifest_file_sha256 == sha256_file(published / "export_manifest.json")
    assert receipt.generation_marker_file_sha256 == sha256_file(published / "artifact_generation.json")


@pytest.mark.parametrize(
    "control",
    ["chapter_source_manifest.json", "chapter_asset_manifest.json", "export_manifest.json", "publication_receipt.json"],
)
def test_unknown_control_schema_version_is_rejected_before_reader_accepts_publication(tmp_path, control):
    published = _publish_verified_complete_bundle(tmp_path)
    _replace_schema_version_and_rehash_nothing(published / control, 999)
    with pytest.raises(PublicationIntegrityError, match="schema_version"):
        reopen_verified_publication(published)


def test_export_manifest_identity_mismatch_is_rejected_even_after_attacker_rehashes_controls(tmp_path):
    published = _publish_verified_complete_bundle(tmp_path)
    _tamper_export_identity_and_recompute_export_and_receipt_hashes(published, execution_id="exec-other")
    with pytest.raises(PublicationIntegrityError, match="execution_id"):
        reopen_verified_publication(published)


@pytest.mark.parametrize(
    "tamper",
    [
        "missing_page_entry",
        "extra_page_entry",
        "duplicate_page_entry",
        "reordered_page_entry",
        "page_id",
        "page_source_sha256",
        "page_result_sha256",
        "page_execution_evidence_sha256",
        "page_content_semantic_sha256",
        "final_pixel_sha256",
        "translated_path_other_page",
        "translated_path_traversal",
        "entry_sha256",
    ],
)
def test_reopen_rejects_rehashed_export_page_entry_not_equal_to_published_authorities(tmp_path, tamper):
    published = _publish_verified_complete_bundle(tmp_path, page_count=3)
    _tamper_export_page_entries_and_recompute_entry_export_and_receipt_hashes(
        published,
        tamper,
    )
    with _spy_final_asset_exposure() as exposed:
        with pytest.raises(PublicationIntegrityError):
            reopen_verified_publication(published)
    assert exposed == ()


@pytest.mark.parametrize(
    "tamper",
    ["missing_asset", "extra_asset", "path_traversal", "project_reference", "evidence_reference", "decoded_pixel_hash"],
)
def test_asset_manifest_rejects_tree_or_reference_divergence_even_when_manifest_is_rehashed(tmp_path, tamper):
    published = _publish_verified_complete_bundle(tmp_path)
    _tamper_asset_tree_and_recompute_manifest_and_receipt_hashes(published, tamper)
    with pytest.raises(PublicationIntegrityError):
        reopen_verified_publication(published)


def test_published_generation_is_portable_and_independent_from_private_execution_root(tmp_path):
    private_root, published = _publish_verified_bundle_from_private_execution_root(tmp_path)
    private_refs_by_page_and_role = _all_private_refs_by_page_and_semantic_role(private_root)
    moved = tmp_path / "moved-publication"
    shutil.move(published, moved)
    assert not published.exists()
    shutil.rmtree(private_root)
    reopened = reopen_verified_publication(moved)
    for page in reopened.verified_inputs.pages:
        result = page.page_execution_evidence.read_verified(moved)
        assert result.final_page.artifact_ref.load_verified(moved).shape[:2] == (result.final_page.height, result.final_page.width)
        published_refs = _all_published_refs_by_semantic_role(page, result)
        private_refs = private_refs_by_page_and_role[page.page_id]
        assert set(published_refs) == set(private_refs)
        for role, published_ref in published_refs.items():
            private_ref = private_refs[role]
            assert published_ref.artifact_store_id != private_ref.artifact_store_id
            assert published_ref.generation_id == reopened.receipt.generation_id
            assert published_ref.source_artifact_ref_sha256 == private_ref.artifact_ref_sha256
            assert published_ref.load_verified_bytes(moved) == private_ref.verified_bytes_before_delete
    assert reopened.receipt.receipt_sha256 == _load_receipt(moved).receipt_sha256


def test_successful_publication_removes_old_only_assets_via_wal_remove_entries(tmp_path):
    root = _seed_complete_publication_with_old_only_asset(tmp_path, "layers/recovery/obsolete.png")
    tx = PublicationTransaction.for_bundle(root, _new_bundle_without_obsolete_asset())
    tx.stage_and_commit()
    assert not (root / "layers/recovery/obsolete.png").exists()
    assert "layers/recovery/obsolete.png" not in reopen_verified_publication(root).asset_manifest.relative_paths
    assert _wal_operations_seen(tx, "layers/recovery/obsolete.png") == ("remove",)


@pytest.mark.parametrize("checkpoint", _every_old_only_remove_backup_and_finalize_checkpoint())
def test_old_only_asset_fault_recovery_yields_whole_old_or_whole_new_tree(tmp_path, checkpoint):
    root = _seed_complete_publication_with_old_only_asset(tmp_path, "layers/recovery/obsolete.png")
    old_tree = _snapshot_complete_publication_tree(root)
    new_tree = _expected_complete_new_tree_without_obsolete_asset(root)
    _interrupt_publication_replacing_tree(root, checkpoint=checkpoint)
    recover_chapter_publication(root)
    assert _snapshot_complete_publication_tree(root) in (old_tree, new_tree)
    reopen_verified_publication(root)


@pytest.mark.parametrize(
    "tamper",
    [
        "schema",
        "wal_hash",
        "run_id",
        "execution_id",
        "transaction_id",
        "resolved_run_root",
        "target_operation",
        "target_pre_state_hash",
        "target_not_in_allowlist",
        "duplicate_target",
        "traversal_target",
        "symlink_target",
        "nfc_casefold_target_collision",
    ],
)
def test_recovery_rejects_untrusted_wal_before_any_filesystem_mutation(tmp_path, tamper):
    root, outside, journal = _publication_with_tampered_wal(tmp_path, tamper)
    sentinel_before = _snapshot_tree(outside)
    publication_before = _snapshot_tree(root, exclude=(journal,))
    with pytest.raises(PublicationRecoveryError):
        recover_chapter_publication(root)
    assert _snapshot_tree(outside) == sentinel_before
    assert _snapshot_tree(root, exclude=(journal,)) == publication_before
    assert journal.exists()


def test_replay_receipt_binds_every_parent_page_evidence_snapshot_in_manifest_order(tmp_path):
    off = _publish_verified_complete_bundle(tmp_path / "off")
    replay = _publish_verified_replay_from_receipt(off, tmp_path / "render")
    reopened = reopen_verified_publication(replay)
    expected = tuple(page.page_execution_evidence.sha256 for page in reopen_verified_publication(off).verified_inputs.pages)
    assert reopened.receipt.replay_source_page_evidence_sha256s == expected
    assert reopened.export_manifest.replay_source_page_evidence_sha256s == expected
    assert tuple(
        page.page_execution_evidence.replay_source_page_evidence_sha256 for page in reopened.verified_inputs.pages
    ) == expected


def test_receipt_is_promoted_last_and_is_the_only_reader_commit_marker(tmp_path):
    tx = _staged_complete_publication(tmp_path)
    phases = _capture_promote_order(tx)
    tx.commit(tx.bundle)
    assert phases[-1] == "promote:publication_receipt.json"
    reopened = reopen_verified_publication(tx.run_root)
    assert reopened.receipt.receipt_sha256 == tx.bundle.publication_receipt.receipt_sha256
    assert reopened.publication_receipt_snapshot.sha256 == sha256_file(tx.run_root / "publication_receipt.json")


def test_second_live_writer_cannot_interleave_wal_staging_or_renames(tmp_path):
    root = tmp_path / "out"
    with PublicationLock(root.resolve(), mode="exclusive_writer"):
        second = PublicationTransaction.for_bundle(root, _other_verified_bundle())
        with pytest.raises(PublicationLockedError):
            second.stage_and_commit()
        assert not second.journal_path.exists()
        assert not second.staging_root.exists()


@pytest.mark.parametrize("checkpoint", _every_publication_commit_checkpoint())
def test_verified_reader_sees_old_receipt_or_waits_for_new_complete_receipt_never_mixed(tmp_path, checkpoint):
    root = _seed_complete_publication(tmp_path)
    old_receipt = reopen_verified_publication(root).receipt.receipt_sha256
    writer = _pause_new_publication_at(root, checkpoint)
    reader = _start_verified_reader(root)
    if checkpoint != "after_receipt_promote":
        assert reader.is_waiting_for_shared_lock()
    writer.finish()
    observed = reader.result()
    assert observed.receipt.receipt_sha256 in {old_receipt, writer.new_receipt_sha256}
    assert observed.asset_manifest.asset_tree_sha256 == recompute_asset_tree_sha256(root, observed.asset_manifest)


def test_asset_manifest_covers_page_evidence_not_referenced_by_project(tmp_path):
    published = _publish_verified_complete_bundle(tmp_path)
    manifest = _load_asset_manifest(published)
    evidence_entries = [entry for entry in manifest.entries if entry.relative_path.startswith("evidence/pages/")]
    assert evidence_entries
    assert all(entry.project_reference is None and entry.evidence_reference for entry in evidence_entries)
    assert _evidence_page_ids(evidence_entries) == _source_manifest_page_ids(published)
```

Os testes que chamam `_run_pipeline` ficam em `MainEmitTests` e usam `unittest` (`self`, `TemporaryDirectory`, `patch.object`), não fixtures pytest. Testes puros de manifest/cardinalidade ficam em `test_chapter_source_manifest.py`; WAL/fault injection/recovery ficam em `test_chapter_publication.py`. `_patched_real_verified_dispatch()` deve patchar `strip_run.run_chapter`, `_project_inputs_from_output_pages`, `_wrap_up_verified_owner_pages` e os helpers legados nos módulos onde `_run_pipeline` realmente os resolve. O config temporário percorre o parser/config loader real; a falha RED deve ser dispatch/publicação, nunca `TypeError` de assinatura nem `AttributeError` de import.

**Step 2: Confirmar RED**

```powershell
Push-Location pipeline
try {
  $redOutput = & .\venv\Scripts\python.exe -m pytest `
    tests/test_page_owner_pipeline.py::test_multi_page_private_execution_uses_one_marker_and_distinct_atomic_page_generations `
    tests/test_page_owner_pipeline.py::test_multi_page_project_adapter_rejects_mixed_generation_or_page_pointer_identity `
    tests/test_page_owner_pipeline.py::test_artifact_ref_rejects_symlink_junction_or_reparse_escape_before_open `
    tests/test_chapter_publication.py::test_three_page_private_generations_publish_and_reopen_portably_end_to_end `
    tests/test_chapter_publication.py::test_reopen_rejects_rehashed_export_page_entry_not_equal_to_published_authorities `
    tests/test_chapter_publication.py::test_published_generation_is_portable_and_independent_from_private_execution_root `
    -q 2>&1
  $redExit = $LASTEXITCODE
  $redText = $redOutput | Out-String
  $redOutput
  if ($redExit -ne 1) { throw "Task 9 expected pytest exit 1 for a behavioral RED, got $redExit" }
  if ($redText -match '(?im)^ERROR(?:\s|:)|ERROR collecting|errors during collection|^E\s+fixture .* not found|UsageError|ModuleNotFoundError|ImportError') {
    throw 'Task 9 RED is structural rather than behavioral; create/fix the API skeleton and rerun the same nodeids'
  }
  if ($redText -notmatch '(?m)^FAILED\s+\S+::') { throw 'Task 9 RED did not report a targeted FAILED nodeid' }
  .\venv\Scripts\python.exe -m pytest `
    tests/test_page_owner_pipeline.py `
    tests/test_extractor.py `
    tests/test_chapter_source_manifest.py `
    tests/test_chapter_publication.py `
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
class SourcePageManifestEntry:
    page_id: str
    relative_source_path: str
    source_file_sha256: str
    page_source_sha256: str
    width: int
    height: int
    mode: Literal["RGB"]


@dataclass(frozen=True)
class ChapterSourceManifest:
    schema_version: int
    run_id: str                 # linhagem de conteúdo
    execution_id: str           # processo/raiz física
    replay_of_execution_id: str | None
    pages: tuple[SourcePageManifestEntry, ...]
    source_page_count: int
    source_tree_sha256: str
    canonical_json_bytes: bytes
    sha256: str


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
    artifact_ref: "PersistedRGBImageArtifactRef"
    width: int
    height: int
    mode: Literal["RGB"]

    def read_only_rgb(self) -> NDArray[np.uint8]: ...
    def mutable_attempt_copy(self) -> NDArray[np.uint8]: ...


@dataclass(frozen=True)
class PagePipelineRequest:
    run_id: str
    execution_id: str
    replay_of_execution_id: str | None
    page_id: str
    page_source_sha256: str
    original_page: OriginalPageSnapshot
    band_projections: tuple[BandProjection, ...]


PageExecutionStatus = Literal["candidate_ready", "final_verified"]


@dataclass(frozen=True)
class PagePixelSnapshot:
    lossless_png_bytes: bytes
    pixel_sha256: str
    artifact_ref: "PersistedRGBImageArtifactRef"
    width: int
    height: int
    mode: Literal["RGB"]

    def read_only_rgb(self) -> NDArray[np.uint8]: ...
    def mutable_attempt_copy(self) -> NDArray[np.uint8]: ...


@dataclass(frozen=True)
class PageGeometrySnapshot:
    run_id: str
    execution_id: str
    page_id: str
    page_source_sha256: str
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
    origin_execution_id: str
    page_id: str
    page_source_sha256: str

    def read(self) -> OwnerGraph: ...  # validated fresh copy


@dataclass(frozen=True)
class PageCompositionSnapshot:
    canonical_json_bytes: bytes
    sha256: str
    run_id: str
    execution_id: str
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
class ArtifactGenerationMarker:
    schema_version: int
    artifact_store_id: str
    generation_id: str
    run_id: str
    execution_id: str
    replay_of_execution_id: str | None
    canonical_json_bytes: bytes
    sha256: str


@dataclass(frozen=True)
class PersistedAssetRef:
    run_id: str
    execution_id: str
    origin_execution_id: str
    source_artifact_ref_sha256: str | None
    page_id: str | None
    artifact_store_id: str
    generation_id: str
    relative_path: str
    file_sha256: str
    size_bytes: int
    media_type: str
    decoded_mode: str | None
    decoded_pixel_sha256: str | None
    width: int | None
    height: int | None
    artifact_ref_sha256: str

    def load_verified_bytes(self, generation_root: Path) -> bytes: ...


@dataclass(frozen=True)
class PersistedRGBImageArtifactRef:
    run_id: str
    execution_id: str
    origin_execution_id: str
    source_artifact_ref_sha256: str | None
    page_id: str
    artifact_store_id: str
    generation_id: str
    relative_path: str
    file_sha256: str
    pixel_sha256: str
    width: int
    height: int
    mode: Literal["RGB"]
    artifact_ref_sha256: str

    def load_verified(self, generation_root: Path) -> NDArray[np.uint8]: ...


@dataclass(frozen=True)
class PageVisualStageArtifacts:
    inpaint: PersistedRGBImageArtifactRef
    typeset: PersistedRGBImageArtifactRef
    page_composition: PersistedRGBImageArtifactRef
    canonical_json_bytes: bytes
    sha256: str


@dataclass(frozen=True)
class FinalPageSnapshot:
    lossless_png_bytes: bytes
    page_output_pixel_sha256: str
    final_file_sha256: str
    artifact_ref: PersistedRGBImageArtifactRef
    width: int
    height: int
    mode: Literal["RGB"]

    def read_only_rgb(self) -> NDArray[np.uint8]: ...


@dataclass(frozen=True)
class TerminalPixelProof:
    run_id: str
    execution_id: str
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
    final_replacement_verdict_sha256s: tuple[str, ...]
    repair_budget_policy_sha256: str
    replacement_verification_policy_sha256: str
    final_qa_probe_id: str
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
class PageExecutionEvidenceSnapshot:
    schema_version: int
    run_id: str
    execution_id: str
    replay_of_execution_id: str | None
    replay_source_page_evidence_sha256: str | None
    page_id: str
    page_source_sha256: str
    page_result_sha256: str
    page_content_semantic_sha256: str
    artifact_store_id: str
    generation_id: str
    canonical_json_bytes: bytes
    sha256: str

    def read_verified(self, generation_root: Path) -> "PageExecutionResult": ...


@dataclass(frozen=True)
class PageExecutionEvidenceRef:
    schema_version: int
    run_id: str
    execution_id: str
    replay_of_execution_id: str | None
    page_id: str
    page_source_sha256: str
    artifact_store_id: str
    generation_id: str  # geração comum da execução privada, igual em todas as páginas
    page_generation_id: str  # geração atômica específica desta página, citada por current.json
    current_pointer_relative_path: str
    current_pointer_file_sha256: str
    page_execution_evidence_relative_path: str
    page_execution_evidence_file_sha256: str
    page_execution_evidence_sha256: str
    page_result_sha256: str
    pointer_sha256: str
    canonical_json_bytes: bytes
    sha256: str

    def read_verified(self, private_execution_root: Path) -> "PageExecutionResult": ...


@dataclass(frozen=True)
class VerifiedPageProjectInput:
    run_id: str
    execution_id: str
    replay_of_execution_id: str | None
    page_id: str
    page_source_sha256: str
    page_result_sha256: str
    page_content_semantic_sha256: str
    artifact_store_id: str
    generation_id: str
    page_execution_evidence_relative_path: str
    page_execution_evidence: PageExecutionEvidenceSnapshot
    original_artifact: PersistedRGBImageArtifactRef
    cleanup_base_artifact: PersistedRGBImageArtifactRef
    inpaint_artifact: PersistedRGBImageArtifactRef
    target_typeset_artifact: PersistedRGBImageArtifactRef
    page_composition_artifact: PersistedRGBImageArtifactRef
    final_artifact: PersistedRGBImageArtifactRef
    project_asset_refs: tuple[PersistedAssetRef, ...]  # RGB/RGBA/L e demais assets desta página
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
    execution_id: str
    replay_of_execution_id: str | None
    source_manifest: ChapterSourceManifest
    source_manifest_sha256: str
    artifact_store_id: str
    generation_id: str
    pages: tuple[VerifiedPageProjectInput, ...]
    canonical_json_bytes: bytes
    sha256: str

    @classmethod
    def build(
        cls,
        source_manifest: ChapterSourceManifest,
        pages: tuple[VerifiedPageProjectInput, ...],
    ) -> "VerifiedProjectInputs": ...

    @classmethod
    def read_verified(cls, generation_root: Path) -> "VerifiedProjectInputs": ...


@dataclass(frozen=True)
class ChapterAssetEntry:
    relative_path: str
    kind: str  # role versionado, não enum fechado; novos assets do schema são permitidos
    project_reference: str | None
    evidence_reference: str | None
    file_sha256: str
    media_type: str
    decoded_mode: str | None
    decoded_pixel_sha256: str | None
    width: int | None
    height: int | None
    size_bytes: int
    page_id: str | None
    owner_ids: tuple[str, ...]


@dataclass(frozen=True)
class ChapterAssetManifest:
    schema_version: int
    run_id: str
    execution_id: str
    replay_of_execution_id: str | None
    artifact_store_id: str
    generation_id: str
    entries: tuple[ChapterAssetEntry, ...]
    asset_tree_sha256: str
    canonical_json_bytes: bytes
    sha256: str


@dataclass(frozen=True)
class ExportPageEntry:
    page_id: str
    page_source_sha256: str
    page_result_sha256: str
    page_execution_evidence_sha256: str
    page_content_semantic_sha256: str
    final_pixel_sha256: str
    translated_path: str
    entry_sha256: str


@dataclass(frozen=True)
class ExportManifest:
    schema_version: int
    run_id: str
    execution_id: str
    replay_of_execution_id: str | None
    artifact_store_id: str
    generation_id: str
    source_manifest_sha256: str
    verified_inputs_sha256: str
    asset_manifest_sha256: str
    replay_source_page_evidence_sha256s: tuple[str, ...]
    pages: tuple[ExportPageEntry, ...]
    export_gate_status: Literal["PASS"]
    canonical_json_bytes: bytes
    sha256: str


@dataclass(frozen=True)
class PublicationReceipt:
    schema_version: int
    run_id: str
    execution_id: str
    replay_of_execution_id: str | None
    artifact_store_id: str
    generation_id: str
    replay_source_page_evidence_sha256s: tuple[str, ...]
    generation_marker_file_sha256: str
    source_manifest_file_sha256: str
    source_manifest_sha256: str
    asset_manifest_file_sha256: str
    asset_manifest_sha256: str
    verified_inputs_file_sha256: str
    verified_inputs_sha256: str
    project_file_sha256: str
    export_manifest_file_sha256: str
    export_manifest_sha256: str
    payload_tree_sha256: str
    canonical_json_bytes: bytes
    receipt_sha256: str  # hash dos campos acima; nunca inclui os próprios bytes do receipt


@dataclass(frozen=True)
class VerifiedChapterBundle:
    schema_version: int
    run_id: str
    execution_id: str
    replay_of_execution_id: str | None
    artifact_store_id: str
    generation_id: str
    replay_source_page_evidence_sha256s: tuple[str, ...]
    source_manifest_sha256: str
    verified_inputs_sha256: str
    asset_manifest: ChapterAssetManifest
    source_manifest_snapshot: FrozenJSONSnapshot
    asset_manifest_snapshot: FrozenJSONSnapshot
    verified_inputs_snapshot: FrozenJSONSnapshot
    project_snapshot: FrozenJSONSnapshot
    export_manifest: ExportManifest
    publication_receipt: PublicationReceipt
    publication_receipt_snapshot: FrozenJSONSnapshot
    generation_tree_sha256: str
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


@dataclass
class PublicationLock:
    resolved_run_root: Path
    mode: Literal["shared_reader", "exclusive_writer"]

    def __enter__(self) -> "PublicationLock": ...
    def __exit__(self, *exc_info: object) -> None: ...


@dataclass
class PageCandidateTransaction:
    private_execution_root: Path
    run_id: str
    execution_id: str
    page_id: str
    artifact_store_id: str
    generation_id: str  # geração comum da execução privada
    page_generation_id: str
    transaction_id: str

    def commit_verified_generation(
        self,
        result: "PageExecutionResult",
    ) -> PageExecutionEvidenceRef: ...
    def recover(self) -> None: ...
    def _fault_checkpoint(self, phase: str) -> None: ...


# Os seis records abaixo têm schema/decoder final declarado nesta Task.
# Tasks 10–15 implementam seus builders/produtores e apertam invariantes; não redefinem os tipos.
@dataclass(frozen=True)
class OwnerRepairRequest:
    request_id: str
    issue_id: str | None
    run_id: str
    execution_id: str
    page_id: str
    page_source_sha256: str
    owner_id: str
    original_page_sha256: str
    failed_stage: str
    reason: str
    evidence_ids: tuple[str, ...]
    next_strategy: str
    request_sha256: str


@dataclass(frozen=True)
class RepairAttempt:
    run_id: str
    execution_id: str
    page_id: str
    page_source_sha256: str
    attempt_id: str
    request_id: str
    issue_id: str | None
    consumed_request_ids: tuple[str, ...]
    translation_binding_sha256: str
    repair_budget_policy_sha256: str
    strategy: str
    variant: str
    owner_id: str
    input_sha256: str
    cleanup_mask_sha256: str
    protected_mask_sha256: str
    geometry_sha256: str
    attempt_fingerprint: str
    outcome: str
    evidence_ids: tuple[str, ...]
    attempt_sha256: str


@dataclass(frozen=True)
class OwnerTargetMaterialization:
    materialization_id: str
    run_id: str
    execution_id: str
    page_id: str
    page_source_sha256: str
    owner_id: str
    translation_binding_sha256: str
    source_payload_sha256: str
    target_payload_sha256: str
    target_glyph_patch_sha256: str
    glyph_mask_sha256: str
    base_pixel_sha256: str
    result_pixel_sha256: str
    materialization_sha256: str


@dataclass(frozen=True)
class LanguageResidualIssue:
    issue_id: str
    run_id: str
    execution_id: str
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
    observed_text_sha256: str
    source_binding_sha256: str | None
    region_sha256: str
    source_only_tokens: tuple[str, ...]
    repair_required: bool
    issue_sha256: str


@dataclass(frozen=True)
class FinalQAProbe:
    probe_id: str
    run_id: str
    execution_id: str
    page_id: str
    page_source_sha256: str
    input_origin: Literal["redecoded_persisted_candidate"]
    candidate_file_sha256: str
    candidate_pixel_sha256: str
    root_input_pixel_sha256: str
    ocr_invocation_id: str
    fresh_ocr_attempt_ids: tuple[str, ...]
    fresh_ocr_attempt_chain_sha256: str
    ocr_payload_sha256: str
    observer_available: bool
    coverage_complete: bool
    issue_ids: tuple[str, ...]
    probe_sha256: str


@dataclass(frozen=True)
class FinalReplacementVerdict:
    verdict_id: str
    run_id: str
    execution_id: str
    page_id: str
    page_source_sha256: str
    owner_id: str
    translation_binding_sha256: str
    source_payload_sha256: str
    target_payload_sha256: str
    target_glyph_patch_sha256: str
    target_materialization_sha256: str
    source_support_sha256: str
    post_cleanup_residual_mask_sha256: str
    source_residual_material_pixel_count: int
    target_delta_mask_sha256: str
    target_alpha_material_pixel_count: int
    target_contrast_score: float
    target_clipped_pixel_count: int
    target_within_safe_region: bool
    source_support_removed: bool
    target_materialized: bool
    replacement_verification_policy_sha256: str
    status: Literal["final_verified"]
    verdict_sha256: str


@dataclass(frozen=True)
class PageExecutionResult:
    run_id: str
    execution_id: str
    replay_of_execution_id: str | None
    page_id: str
    page_source_sha256: str
    result_sha256: str
    page_content_semantic_sha256: str
    status: PageExecutionStatus
    original_page: OriginalPageSnapshot
    cleanup_base: PagePixelSnapshot
    visual_stage_artifacts: PageVisualStageArtifacts
    project_asset_refs: tuple[PersistedAssetRef, ...]
    page_geometry: PageGeometrySnapshot
    ocr_result_view: FrozenJSONSnapshot
    text_layers_view: FrozenJSONSnapshot
    coverage: PageCoverageResult
    owner_graph: OwnerGraphSnapshot
    translation_attempts: tuple[TranslationAttempt, ...]
    translations: tuple[TranslationBinding, ...]
    repair_requests: tuple["OwnerRepairRequest", ...]
    repair_history: tuple["RepairAttempt", ...]
    repair_budget_policy_sha256: str | None
    page_commits: tuple[OwnerExecutionCommit, ...]
    owner_target_materializations: tuple["OwnerTargetMaterialization", ...]
    page_composition: PageCompositionSnapshot
    final_qa_ocr_requests: tuple[OCRRequest, ...]
    final_qa_ocr_invocations: tuple[OCRInvocationResult, ...]
    language_residual_issues: tuple["LanguageResidualIssue", ...]
    qa_probes: tuple["FinalQAProbe", ...]
    final_replacement_verdicts: tuple["FinalReplacementVerdict", ...]
    replacement_verification_policy_sha256: str | None
    replay_source_page_evidence_sha256: str | None
    final_page: FinalPageSnapshot | None
    terminal_proof: TerminalPixelProof | None = None

    @property
    def lifecycle(self) -> PageCoverageLedger: return self.coverage.ledger
    @property
    def ocr_requests(self) -> tuple[OCRRequest, ...]: return self.coverage.ocr_requests
    @property
    def ocr_invocations(self) -> tuple[OCRInvocationResult, ...]: return self.coverage.ocr_invocations
```

`ChapterSourceManifest`, `PagePixelSnapshot`, `PageGeometrySnapshot`, `FrozenJSONSnapshot`, `OwnerGraphSnapshot`, `PageCompositionSnapshot`, `ArtifactGenerationMarker`, `PersistedAssetRef`, `PersistedRGBImageArtifactRef`, `PageVisualStageArtifacts`, `FinalPageSnapshot`, `TerminalPixelProof`, `PageExecutionEvidenceSnapshot`, `PageExecutionEvidenceRef`, `VerifiedPageProjectInput`, `VerifiedProjectInputs`, `ChapterAssetManifest`, `ExportPageEntry`, `ExportManifest`, `VerifiedChapterBundle` e os seis records finais de repair/QA são contratos reais desta Task, não pseudotipos. Os construtores recalculam seus hashes; `read()` sempre desserializa uma cópia nova; arrays retornados por `read_only_rgb()` têm `writeable=False`. Stages normativos usam `PersistedRGBImageArtifactRef`; masks/brush/recovery e qualquer RGBA/L/binário usam `PersistedAssetRef`, que verifica bytes e opcionalmente pixels decodificados sem forçar `mode=RGB`. Os records `OwnerRepairRequest`, `RepairAttempt`, `OwnerTargetMaterialization`, `LanguageResidualIssue`, `FinalQAProbe` e `FinalReplacementVerdict` têm schema/version/decoder concretos em `ownership/model.py` desde esta Task; Tasks 10–15 implementam builders e produtores sobre esses mesmos tipos, sem forward-reference fictícia, redefinição ou bypass de fixture.

Implemente já nesta Task o shell de persistência de `PageCandidateTransaction.commit_verified_generation()`. O coordenador cria uma única raiz por execução privada, `<run>/.owner-executions/<execution_id>/`, com um único `artifact_generation.json`; `artifact_store_id/generation_id` desse marker são comuns a todas as páginas. A transação aceita somente um `PageExecutionResult(status="final_verified")` estruturalmente válido, grava assets/snapshot sob um diretório temporário da página, reabre tudo contra a raiz comum, promove para `.page-generations/<page_id>/<page_generation_id>/` e troca `.page-current/<page_id>/current.json` atomicamente, retornando `PageExecutionEvidenceRef`. O pointer contém o `page_generation_id`, mas todas as artifact refs continuam relativas à raiz comum e repetem o único `generation_id` do marker. A transação nunca cria marker por página. Os fixtures terminais da Task 9 chamam essa mesma API de produção; não escrevem snapshot/pointer à mão. A Task 10 acrescenta `from_original()/replace_owner_commit()/compose()` ao mesmo tipo, e a Task 15 passa a produzir o `final_verified` real e completa fault recovery — sem criar uma segunda transação page-local.

O fixture `final_verified` usado nos REDs desta Task instancia, pelos constructors/decoders reais, todas as seções finais já declaradas: request/attempt/policy, commit/materialization/composition, request+invocation OCR final, issue/probe, verdict/policy, final e proof mutuamente ligados. Não existe flag `skip_validation`, result parcial marcado terminal nem writer alternativo. Somente o **produtor visual real** dessas seções chega nas Tasks 10–15; o schema, os invariantes do builder e o boundary persistido já são finais aqui.

Toda artifact ref canônica contém somente `artifact_store_id + generation_id + relative_path` POSIX e hashes; não contém path absoluto, URI ou raiz serializada. A raiz física é um argumento confiável e explícito de `load_verified*(generation_root)`, cujo marker versionado deve repetir store/generation/run/execution. O resolver normaliza e exige contenção após `resolve()`, rejeita `..`, `.`, drive/UNC, barra invertida, symlink/reparse-point que escape e colisão casefold/NFC antes de abrir. O `generation_id` é estável durante staging → promoção/movimento e não deriva do nome temporário da pasta; copiar/mover a geração junto de seu marker conserva validade. Refs frescas têm `origin_execution_id == execution_id` e `source_artifact_ref_sha256=None`; replay ou publicação reabrem a ref fonte contra sua raiz, verificam bytes/pixels e copiam para a geração destino, criando uma ref destino nova com o mesmo conteúdo, IDs físicos do destino e `source_artifact_ref_sha256` da ref fonte. Nunca se reetiqueta a ref antiga.

`OriginalPageSnapshot.artifact_ref`, `PagePixelSnapshot.artifact_ref`, `PageVisualStageArtifacts`, `PageExecutionResult.project_asset_refs` e `FinalPageSnapshot.artifact_ref` são a autoridade persistida. `PageExecutionResult.project_asset_refs` inclui cada mask/brush/recovery/image/translated asset produzido para a página. A primeira ponte persistida é `OutputPage.owner_page_evidence_ref: PageExecutionEvidenceRef | None`, produzida pela transação page-local. Seu `read_verified(private_execution_root)` valida primeiro o único marker da execução e a igualdade de `artifact_store_id/generation_id`; resolve `.page-current/<page_id>/current.json` com as mesmas regras de path relativo/contenção/symlink/NFC-casefold das artifact refs, valida `current_pointer_file_sha256`; dentro do pointer, `pointer_sha256` cobre o payload sem o próprio campo e o `page_generation_id` seleciona o subdiretório imutável; então confere o `PageExecutionEvidenceRef.sha256` e só depois abre `page_execution_evidence.json`. Todas as refs internas são resolvidas contra `private_execution_root`, não contra o subdiretório da página. Em `enforce`, `owner_page_result` é mera view de compatibilidade; pode ser apagado ou corrompido sem alterar o adapter, e nunca substitui pointer/snapshot ausente ou inválido. Fixtures `final_verified` persistem geração + pointer reais pelo mesmo helper de produção; não existe fixture-only authority bypass.

`VerifiedPageProjectInput.project_asset_refs` copia e revalida esse conjunto, contém também equivalentes genéricos das refs explícitas de original/stages/final e rejeita `relative_path` duplicado divergente. O adapter recebe `private_execution_root` explicitamente para verificar a única geração privada comum e produz `source_verified_inputs`. Antes do build, exige que todas as páginas repitam o mesmo `artifact_store_id/generation_id` do único marker, que cada `PageExecutionEvidenceRef.page_generation_id` seja distinto e pertença ao `page_id` do seu `current.json`, e que nenhum pointer/evidence path seja compartilhado ou trocado entre páginas. O wrap-up recebe `source_private_execution_root`, copia cada ref já verificada para a geração staged e reconstrói, por rebinding hash-linked, cada result/snapshot; então chama novamente `VerifiedProjectInputs.build()` para produzir `published_verified_inputs` com os IDs/refs da geração destino. Somente o hash deste segundo objeto entra no bundle/export e somente ele chega a `build_project_json`. Depois disso nenhuma ref publicada depende da raiz privada. `ExportPageEntry.page_result_sha256` e `page_execution_evidence_sha256` são extraídos dos results **publicados e rebindados**, nunca dos results privados de `OutputPage`. `page_content_semantic_sha256` exclui `execution_id`, store/generation/path e hashes de refs físicas; ele deve permanecer igual entre privado e publicado, enquanto `result_sha256` normalmente muda com o rebinding. `VerifiedProjectInputs` não contém `OutputPage`, list, dict mutável nem raiz absoluta serializada: apenas tuples, snapshots frozen, IDs da geração e artifact refs. No JSON, cada página registra `page_execution_evidence_relative_path + sha256` e os summaries; não embute novamente os bytes do snapshot. `read_verified(generation_root)` carrega cada arquivo por esse path relativo, valida hash/identidade e chama o reader da página. Seu builder exige bijeção ordenada com o source manifest e um único store/generation comum antes de calcular `sha256`. `VerifiedChapterBundle` repete os hashes/IDs publicados, cobre a geração completa e valida todos os paths relativos do projeto e da evidência contra `ChapterAssetManifest`.

`PageExecutionEvidenceSnapshot.build(result)` serializa e deep-freezeia a projeção canônica completa do journal: coverage/ledger/inventory/components/observations/recovery requests/decisions, OCR de coverage, owner graph, translation attempts/bindings, repair requests/history/policies, commits/materializations, geometry/text layers/composition, artifact refs, requests/invocations OCR do QA final, language issues, QA probes/verdicts, final e `TerminalPixelProof`. `PageExecutionResult.to_canonical_dict()` devolve essa mesma projeção metadata-only, nunca os buffers runtime. O payload JSON contém metadados, IDs, refs e hashes, **nunca** `lossless_png_bytes`, ndarray, PNG/base64 ou path absoluto; os bytes raster continuam uma única vez nos assets lossless. `read_verified(generation_root)` valida marker/contenção, reabre todas as refs, reconstrói os snapshots raster em memória, recalcula cada subhash e executa `PageExecutionResult.build()`. Campo ausente/extra, versão desconhecida, identidade/ref/subhash divergente ou asset fora da raiz falha. `VerifiedPageProjectInput` carrega esse snapshot obrigatório, e todos os seus campos-resumo devem coincidir com ele. Após o adapter, `_wrap_up_verified_owner_pages()` e a Task 16 leem exclusivamente `page_execution_evidence`; não recebem `OutputPage`, o result vivo nem debug side channel.

O schema top-level final de `PageExecutionEvidenceSnapshot` já nasce nesta Task com todas as seções de Tasks 10–15: `repair_requests`, `repair_history`, `repair_budget_policy_sha256`, `owner_target_materializations`, `final_qa_ocr_requests`, `final_qa_ocr_invocations`, `language_residual_issues`, `qa_probes`, `final_replacement_verdicts`, `replacement_verification_policy_sha256`, `final_page`, `terminal_proof` e `replay_source_page_evidence_sha256`. Em `candidate_ready`, tuples ainda não aplicáveis são vazias e policies/final/proof/replay-source são nulos; os record types usam schema/version próprios e são populados pelas Tasks posteriores sem mudar as chaves do envelope. Qualquer nova chave top-level futura exige bump de `PAGE_EXECUTION_EVIDENCE_SCHEMA_VERSION`, decoder explícito da versão anterior e RED de migração; nunca aceite silently campo desconhecido nem omita nova evidência.

O lifecycle não é uma cópia concorrente: `result.lifecycle` é uma propriedade read-only sobre `result.coverage.ledger`. `OwnerGraphSnapshot.build()` extrai `run_id/origin_execution_id/page_id/page_source_sha256` de todas as observações, exige unanimidade com request+coverage e `verification_status="verified"`; graph vazio usa explicitamente a identidade do request e ainda é validado pelo ledger. `PageGeometrySnapshot` repete a identidade física da página e rejeita swap mesmo quando dimensões/bounds coincidem. Todo ID de `TranslationBinding.attempt_ids` resolve exatamente uma vez em `translation_attempts`. `TerminalPixelProof` nasce aqui apenas como envelope serializável/hash-linked para permitir testar o boundary e o adapter; somente fixtures podem construí-lo diretamente. A Task 15 implementa o builder de produção a partir do PNG persistido e do OCR fresco.

`ChapterAssetEntry.kind` é um role versionado e não uma `Literal` fechada; valide formato/tamanho e persista o valor no hash, permitindo que novos assets do schema entrem sem bypass. `project_reference` identifica a referência exata do `project.json`; `evidence_reference` identifica o record canônico de `evidence/pages/<page_id>/**`, e exatamente um deles pode ser preenchido quando o asset não é compartilhado. `media_type/decoded_mode/decoded_pixel_sha256/width/height` copiam a ref verificada e fazem parte do hash. A completude é derivada da união da enumeração real de paths do projeto staged com o evidence tree exigido pela Task 16, não de uma lista estática de kinds. `ChapterAssetManifest` cobre payloads, evidências e `project.json`, mas exclui deliberadamente `chapter_source_manifest.json`, `chapter_asset_manifest.json`, `export_manifest.json` e `publication_receipt.json`; `PublicationReceipt` fecha essa DAG sem self-hash.

`replay_source_page_evidence_sha256s` é vazio numa execução fresca e, em replay, tem cardinalidade/ordem idênticas às páginas do source manifest, sem duplicatas; seus valores devem coincidir 1:1 no result/snapshot de cada página, no export manifest, receipt e bundle. O reader valida essa cadeia antes de expor qualquer asset.

Defina constantes únicas `ARTIFACT_GENERATION_MARKER_SCHEMA_VERSION`, `PAGE_EXECUTION_EVIDENCE_SCHEMA_VERSION`, `PAGE_EXECUTION_EVIDENCE_REF_SCHEMA_VERSION`, `CHAPTER_SOURCE_MANIFEST_SCHEMA_VERSION`, `CHAPTER_ASSET_MANIFEST_SCHEMA_VERSION`, `EXPORT_MANIFEST_SCHEMA_VERSION`, `PUBLICATION_RECEIPT_SCHEMA_VERSION` e `VERIFIED_CHAPTER_BUNDLE_SCHEMA_VERSION`. Os readers rejeitam versão ausente/desconhecida antes de confiar em hashes. `run_id/execution_id/replay_of_execution_id` devem coincidir entre source manifest, verified inputs, asset manifest, export manifest, receipt e bundle; `artifact_store_id/generation_id` devem coincidir entre marker, refs publicadas, verified inputs, asset manifest, export manifest, receipt e bundle. Em replay, o receipt prova explicitamente a execução pai e os hashes dos evidence snapshots fonte.

Nesta Task, adapte também o `OwnerExecutionCommit` existente para incluir a identidade mínima `commit_id`, `run_id`, `execution_id`, `page_id` e `page_source_sha256`, calculada/validada no produtor real. Isso torna os REDs de identidade desta própria Task implementáveis e impede reutilizar commit físico de outro replay. A Task 10 acrescenta binding/source/target/glyph hashes ao mesmo record; não criar outro tipo de commit.

Nesta Task, o coordenador chama: discovery → OCR/coverage → association/container recovery → graph → translation → adapters atuais de execution/final QA. Sem uma prova terminal injetada em teste, o resultado intermediário é `candidate_ready`, mantém `final_page=None` e é recusado pelo adapter de projeto/export. As Tasks 10–15 substituem os adapters pela transação e populam as seções finais já reservadas no `PageExecutionResult`/evidence schema; elas não acrescentam chaves top-level silenciosamente. Assim produzem o primeiro `final_verified` de produção sem quebrar o contrato page-first intermediário.

Nesta Task, `PageExecutionResult.build()` valida o que já existe: toda request/invocation OCR de source vem de `coverage` (as propriedades são views, não coleções concorrentes), e coverage/lifecycle, graph e attempts/bindings de tradução repetem a linhagem `run/origin-execution/page/source`; commit, geometria, composição, stage refs, repair/materialization/QA e proof repetem também o `execution_id` atual. Em replay, records de conteúdo carregados preservam `origin_execution_id == replay_of_execution_id`, enquanto envelope, refs copiadas e novos render/QA/proof usam o `execution_id` atual; qualquer reetiquetagem ou cruzamento falha. Cada invocation valida sua request raiz e a cadeia de attempts físicos; `page_composition.commit_ids` e os conjuntos ordenados de binding/source/target hashes coincidem com as coleções do resultado. `target_glyph_patch_sha256s` e `target_materialization_sha256s` permanecem tuplas vazias no resultado candidato intermediário; a Task 10 cria esses records e elimina essa tolerância para owner traduzível.

Para `final_verified`, cada `FinalQAProbe.ocr_invocation_id` resolve exatamente uma invocation em `final_qa_ocr_invocations`, que por sua vez resolve exatamente uma request em `final_qa_ocr_requests`; attempt IDs, root/input/chain/payload hashes coincidem. Cada `LanguageResidualIssue.invocation_ids` resolve apenas invocations finais persistidas, cada `issue_id` citado por probe/request de repair resolve exatamente uma issue e não há issue repair-required não consumida. `terminal_proof.final_qa_probe_id` resolve o probe final e `fresh_ocr_invocation_id` é exatamente a invocation desse probe. O estado exige simultaneamente `final_page` e `terminal_proof`, identidades iguais, `terminal_proof.composition_sha256 == page_composition.sha256` e `terminal_proof.final_page_pixel_sha256 == terminal_proof.fresh_ocr_root_input_pixel_sha256 == final_page.page_output_pixel_sha256 == final_page.artifact_ref.pixel_sha256`; `fresh_ocr_attempt_chain_sha256` cobre os attempts ordenados, nenhum record aponta para attempt ausente e os hashes de verdict/policy do proof coincidem com os records persistidos. `candidate_ready` exige todas as coleções finais vazias e prova/final ausentes. Estado transitório ou erro operacional não é serializado como resultado exportável. O `result_sha256` cobre status, schema, execution/ref física, `replay_source_page_evidence_sha256` e hashes de todos os snapshots/artefatos/provas; `page_content_semantic_sha256` cobre conteúdo, geometry sem path, coverage/graph/bindings/commits/QA/verdicts e pixels, excluindo apenas IDs/paths/hashes de rebinding físico.

A relação request/invocation/probe acima é obrigatoriamente **bidirecional** e de cardinalidade um: os IDs de `final_qa_ocr_requests` e `final_qa_ocr_invocations` são o mesmo conjunto e `Counter(probe.ocr_invocation_id) == Counter({invocation_id: 1 ...})`. Request/invocation sem probe ou invocation compartilhada por dois probes é erro estrutural, mesmo que todos os hashes externos sejam recalculados.

`from_legacy_bands()` copia somente geometria/hash/proveniência para `BandProjection` frozen. Nunca importe `ocr_result`, `cleaned_slice`, `rendered_slice` ou decisões de um `Band`; nunca escreva de volta no objeto mutável. Adapters recebem `OriginalPageSnapshot.mutable_attempt_copy()`, de modo que uma mutação OpenCV não contamina o original canônico nem o retry seguinte.

Antes do graph, execute um loop de fechamento de coverage. Cada `coverage_recovery_pending` deve consumir a próxima tentativa determinística (full-page, associação, crop ancorado, variante visual, container conservador) e produzir um snapshot novo; graph/translation/execution ficam proibidos enquanto `require_ready_for_ownership()` falhar. Não repita `coverage_attempt_fingerprint = sha256(anchor_sha256 + attempt_kind + transform_spec_sha256 + geometry_sha256 + input_pixel_sha256)`: duas observações sem componente na mesma página têm âncoras distintas e podem progredir independentemente. Esgotadas tentativas visuais com glyph support material positivo, materialize obrigatoriamente componente + container support-local, estenda o inventário por decision hash-bound e marque `protection_conflict`; pending não pode virar terminal de conteúdo. Exaustão por falha real de engine é erro operacional sem export, não preservação de conteúdo.

Antes da execução, feche todos os bindings. Resposta inválida avança backend/prompt/contexto; indisponibilidade total levanta `TranslationInfrastructureError` e não cria `final_page`. `translation_recovery_pending` é estado transitório consumido pelo coordenador, nunca retorno final.

Na reabertura, `ExportManifest.pages` precisa ser uma bijeção **ordenada** com `ChapterSourceManifest.pages` e `published_verified_inputs.pages`. Para cada ordinal, `page_id`, source/result/evidence/semantic/final hashes devem coincidir com as autoridades publicadas; `translated_path` é exatamente `published_result.final_page.artifact_ref.relative_path`, resolve para a mesma entry do `ChapterAssetManifest` e não pode apontar a outra página ou traversal. Essa validação ocorre antes de abrir/expor qualquer asset final e não confia em `entry_sha256`, export hash ou receipt rehashados pelo mesmo payload adulterado.

**Step 4: Integrar de forma incremental**

- `run_chapter()` continua retornando exatamente `list[OutputPage]`. Ele mantém `candidate_ready` somente em memória; para cada `final_verified`, a transação page-local persiste snapshot + `current.json` e usa `adapt_page_execution_result_to_output_page(result, evidence_ref=...)` para ligar a ponte autoritativa ao contrato existente.
- acrescente `owner_page_evidence_ref: PageExecutionEvidenceRef | None` e `owner_page_result: PageExecutionResult | None` a `OutputPage` por forward reference/`TYPE_CHECKING`; somente a primeira é autoridade em `enforce`. `owner_page_result`, `y_top/y_bottom/image/path/original_image/inpainted_image/owner_graph/ocr_result/text_layers/page_surface_geometry` permanecem views de compatibilidade derivadas;
- `PageExecutionResult` carrega snapshots frozen/hash-linked do request original, cleanup base, geometria, owner graph, OCR compatibility view, target text layers e final. `FinalPageSnapshot.artifact_ref` aponta para o PNG persistido e prova hash de arquivo **e** pixels decodificados;
- crie em `main.py` `_project_inputs_from_output_pages(source_manifest, output_pages, *, private_execution_root: Path) -> VerifiedProjectInputs`: exige bijeção ordenada com `ChapterSourceManifest`; primeiro valida o único marker da raiz privada comum e então, para cada página `enforce`, exige `owner_page_evidence_ref`, chama `ref.read_verified(private_execution_root)`, exige `status == "final_verified"`, reabre todas as artifact refs com essa mesma raiz e extrai bounds/metadata/text layers dos snapshots. Antes do build, rejeita store/generation comum misturado, `page_generation_id` duplicado, pointer trocado entre páginas ou qualquer ref que não pertença à mesma execution/root. Ignora `owner_page_result` e todas as views mutáveis de `OutputPage`, mesmo quando presentes; ref/pointer/snapshot ausente ou tamper nunca cai de volta para o objeto vivo. Estado não terminal lança `PageNotTerminalError`; página ausente/extra/duplicada/reordenada lança `ChapterCardinalityError`; arquivo ausente, escape de path ou qualquer hash divergente lança `PageArtifactIntegrityError`. Sem evidence ref, mantenha o comportamento legacy somente fora de `enforce` até a retirada final da Task 17. O adapter é o **único** boundary que aceita `OutputPage` no branch verificado e é chamado exatamente uma vez;
- a cadeia normativa é `ChapterSourceManifest + list[OutputPage] + private_execution_root comum -> VerifiedProjectInputs -> _wrap_up_verified_owner_pages(inputs, source_private_execution_root=...) -> VerifiedChapterBundle -> PublicationTransaction`. No início de `_run_pipeline(config_path: str)`, antes de qualquer `mkdir` público, adquira `PublicationLock(..., "exclusive_writer")` e chame `recover_chapter_publication(work_dir)` sob o mesmo lock. `extractor.extract()` passa a preservar paths relativos aninhados para pasta/CBZ e a retornar `image_files` na ordem canônica sem achatar basenames; rejeita traversal/collision casefold antes de escrever. Depois da extração, construa o source manifest uma vez com `ChapterSourceManifest.from_extracted_pages(image_files, extraction_root, ...)` — nunca rescaneie a pasta — e passe a mesma instância/hash a `run_chapter`, ao adapter, ao bundle e ao gate. Extraia um único boundary `_run_verified_strip_chapter(...)`: ele chama `run_chapter()`, passa manifest+retorno+raiz privada comum uma vez ao adapter e nunca mais retém `OutputPage`. Para criar um seam real sem mudar a assinatura pública, mova o import local de `run_chapter` para `import strip.run as strip_run`; `_run_verified_strip_chapter(..., run_chapter_fn=None)` resolve `strip_run.run_chapter` quando `None`. Os REDs criam config JSON temporário válido, patcham `strip_run.run_chapter` e chamam `main._run_pipeline(str(config_path))`, sem kwargs inexistentes. Espione `adapter.assert_called_once_with(source_manifest, output_pages, private_execution_root=private_execution_root)` e prove recursivamente que args de wrap-up/builders não contêm `OutputPage`;
- `_wrap_up_verified_owner_pages(inputs: VerifiedProjectInputs, *, source_private_execution_root: Path) -> VerifiedChapterBundle` trata `inputs` como `source_verified_inputs`, preflighta **todas** as páginas e reabre/verifica o evidence snapshot completo e cada artifact ref contra a raiz privada comum. Então monta a geração completa sob `<run>/.publication-staging/<transaction_id>/`: `artifact_generation.json`, `originals/**`, `images/**`, `layers/mask/**`, `layers/brush/**`, `layers/recovery/**`, `translated/**`, `evidence/verified_project_inputs.json`, `evidence/pages/<page_id>/**`, `project.json`, `chapter_source_manifest.json`, `chapter_asset_manifest.json`, `export_manifest.json`, `publication_receipt.json` e qualquer path adicional referenciado pelo schema. Cada cópia gera ref destino relativa/hash-linked e um `PageExecutionEvidenceSnapshot` reconstruído que resolve somente contra a geração staged/publicada; com esses snapshots, o wrap-up constrói `published_verified_inputs` novo, persiste exatamente seus bytes canônicos em `evidence/verified_project_inputs.json` e não conserva path/ID físico da raiz privada. Os quatro controles são detached do asset tree; `artifact_generation.json` e `evidence/verified_project_inputs.json` integram o asset tree, e `publication_receipt.json` é calculado/promovido por último. `build_project_json` recebe somente `published_verified_inputs`; nenhum dict/list derivado de `OutputPage` atravessa o boundary. Substitua em `main.py:9410+` as leituras de `p.path/p.ocr_result/p.text_layers/inpainted_image/image/original_image`, em `_owner_pages_have_final_pixel_authority()` a inferência por `_owner_graph_mode/owner_graph/owner_composition`, e em `build_project_json(..., output_pages=...)` a geometria lida de `page_surface_geometry`. Imagens e layers vêm dos snapshots/refs; authority vem de status + proof + hashes. Reabra o projeto, `verified_project_inputs`, os snapshots por página e os quatro controles staged usando apenas a raiz staged; depois da promoção, apague/mova a raiz privada em teste e reabra a geração pelo receipt, provando que todo path permanece contido e coincide com `ChapterAssetManifest`/receipt;
- implemente `PublicationLock`/`PublicationTransaction` em `ownership/publication.py` sobre a geração completa. O lock é cross-process por raiz resolvida (`LockFileEx` via `ctypes` stdlib no Windows, com shared/exclusive reais; `fcntl.flock` nos demais), exclusivo para writer/recovery e compartilhado para `reopen_verified_publication`; o segundo writer falha antes de criar WAL/staging ou aguarda de forma determinística, nunca intercala. Antes de qualquer mutação, compare a árvore verificada antiga com o manifest novo: paths antigos ausentes no novo viram operações WAL `remove`, com o mesmo backup/restore dos replaces, para não deixar assets órfãos. Grave/fsync o WAL com schema/version, hash canônico, run/execution/transaction, raiz resolvida, pre-state/hash e operação de cada target. O reader de recovery valida tudo isso, normaliza cada target, exige contenção real após resolve e rejeita traversal, symlink/junction/reparse escape, collision NFC/casefold ou target fora da allowlist **antes do primeiro rename/remove**. Para cada target, grave e fsync `backup_intent`, faça rename+fsync, grave `backup_done`; depois, quando aplicável, `promote_intent`, rename+fsync e `promote_done`, sempre deixando `publication_receipt.json` por último. Fault injection cobre before/after de replace e remove. Qualquer exceção remove somente alvos promovidos por esta transação e restaura backups; se a restauração falhar, conserve WAL/staging/backup e lance `PublicationRecoveryError`. `recover_chapter_publication()` é idempotente para qualquer prefixo e só remove journal/backup depois de reabrir e validar árvore + receipt. Leitores sem receipt válido não aceitam a publicação. Não execute `images_dir.mkdir()`, `originals_dir.mkdir()`, `translated_dir.mkdir()` nem `layers/**` antecipadamente em `enforce`; isso permanece apenas para legacy/manual;
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
    tests/test_extractor.py `
    tests/test_chapter_source_manifest.py `
    tests/test_chapter_publication.py `
    tests/test_owner_lifecycle_e2e.py::test_no_band_dialogue_recipe_completes_basic_page_first_lifecycle `
    tests/test_strip_owner_control_plane.py `
    tests/test_strip_owner_composition_integration.py `
    tests/test_owner_lifecycle.py `
    tests/test_main_emit.py `
    tests/test_project_writer.py `
    -q
} finally { Pop-Location }
```

Expected: PASS; mudar o particionamento de bands não muda owners/traduções/commits, o adapter é único, a cardinalidade coincide com o source manifest e nenhuma falha multipágina/WAL deixa a árvore pública parcial.

**Step 6: Commit**

```powershell
$guard = '.codex-tmp\universal-source-replacement-plan-guard\TaskOwnedGit.ps1'
& $guard -Action Stage -Task 9
git diff --cached --check
git commit -m "refactor: coordinate owner processing in page space"
& $guard -Action VerifyAfterCommit -Task 9
```

### Task 10: Tornar cleanup e render uma transação atômica recuperável

**Files:**

- Modify: `pipeline/ownership/execution.py`
- Modify: `pipeline/ownership/model.py`
- Modify: `pipeline/inpainter/owner_mask.py`
- Modify: `pipeline/strip/process_bands.py`
- Modify: `pipeline/strip/page_pipeline.py`
- Modify: `pipeline/ownership/lifecycle.py`
- Test: `pipeline/tests/test_owner_atomic_execution.py`
- Test: `pipeline/tests/test_strip_owner_composition_integration.py`
- Create: `pipeline/tests/test_owner_source_replacement_enforce.py`
- Read only regression: `pipeline/tests/test_owner_source_replacement_fail_closed.py`

**Step 1: Inspecionar e preservar o teste local antes da edição**

```powershell
git status --short -- pipeline/tests/test_owner_source_replacement_fail_closed.py
$localTestDiff = & git diff --no-index -- NUL pipeline/tests/test_owner_source_replacement_fail_closed.py
$localTestDiffExit = $LASTEXITCODE
$localTestDiff
if ($localTestDiffExit -gt 1) { throw "could not inspect untracked test: $localTestDiffExit" }
```

Não substitua, edite, stage ou commit esse arquivo. Execute seus cenários como regressão local; traduza as expectativas normativas necessárias para o novo `test_owner_source_replacement_enforce.py`, sem copiar alterações proprietárias do usuário.

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
        _transaction().commit(_valid_cleanup(), _valid_ptbr_glyph_patch(), overrides={field: _sha256("stale")})


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
                dataclasses.replace(result.owner_target_materializations[0], target_payload_sha256=_sha256("stale")),
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


def test_repairing_owner_b_recomposes_from_original_and_preserves_owner_a_once():
    original = _two_owner_original_page()
    tx = PageCandidateTransaction.from_original(original, commits=(_valid_owner_a_commit(),))
    candidate = tx.replace_owner_commit(_owner_b_r3_commit()).compose()
    assert _target_occurrences(candidate, "owner-a") == 1
    assert _target_occurrences(candidate, "owner-b") == 1
    assert not _source_support_visible(candidate, "owner-a")
    assert not _source_support_visible(candidate, "owner-b")
    assert candidate.base_pixel_sha256 == canonical_page_sha256(original)


def test_failed_owner_b_repair_persists_no_partial_page_generation(tmp_path):
    tx = PageCandidateTransaction.from_original(
        _two_owner_original_page(),
        commits=(_valid_owner_a_commit(),),
        artifact_root=tmp_path,
    )
    with pytest.raises(AtomicReplacementError):
        tx.replace_owner_commit(_invalid_owner_b_commit()).compose()
    assert not list(tmp_path.glob("page-current/**/final.png"))
    assert _published_commit_ids(tmp_path) == ()
```

**Step 3: Confirmar RED**

```powershell
Push-Location pipeline
try {
  $redOutput = & .\venv\Scripts\python.exe -m pytest `
    tests/test_owner_atomic_execution.py `
    tests/test_owner_source_replacement_enforce.py `
    tests/test_owner_source_replacement_fail_closed.py `
    tests/test_strip_owner_composition_integration.py `
    -q 2>&1
  $redExit = $LASTEXITCODE
  $redText = $redOutput | Out-String
  $redOutput
  if ($redExit -ne 1) { throw "Task 10 expected pytest exit 1 for a behavioral RED, got $redExit" }
  if ($redText -match '(?im)^ERROR(?:\s|:)|ERROR collecting|errors during collection|^E\s+fixture .* not found|UsageError|ModuleNotFoundError|ImportError') {
    throw 'Task 10 RED is structural rather than behavioral; create/fix the API skeleton and rerun the same nodeids'
  }
  if ($redText -notmatch '(?m)^FAILED\s+\S+::') { throw 'Task 10 RED did not report a targeted FAILED nodeid' }
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

`commit()` só completa o `OwnerExecutionCommit` adaptado na Task 9 quando cleanup e target pertencem à mesma tentativa e à mesma cadeia de hashes. Não introduza um segundo tipo de commit; acrescente e valide `translation_binding_sha256`, `source_payload_sha256`, `target_payload_sha256` e `target_glyph_patch_sha256`. Eles vêm do `TranslationBinding`/patch real e qualquer mismatch aborta antes da composição.

Implemente o builder/produtor do `OwnerTargetMaterialization` frozen já declarado na Task 9, sem redefinir seu schema. Exatamente uma materialização final corresponde a cada binding. Passe a popular o campo já reservado `PageExecutionResult.owner_target_materializations`, exigindo correspondência 1:1 de owner e hashes com `translations`/`page_commits`; o schema top-level do evidence não muda. `PageCompositionSnapshot` agrega os mesmos hashes ordenados e o execution atual; não os recalcula de texto solto. A partir deste commit, owner traduzível com tuple vazia de glyph/materialization ou qualquer cardinalidade/hash divergente é inválido inclusive em `candidate_ready`; fica removida a tolerância estrutural temporária da Task 9.

Valide também `text_layers_view`: exatamente uma layer final por binding, ligada por `owner_id`, `translation_binding_sha256` e `target_payload_sha256`; o texto da layer deve hashear para o target e nunca pode vir de source/índice. Essa snapshot é a única fonte de text layers do projeto.

`committed` é exclusivamente o outcome retornado por `OwnerReplacementTransaction`, não um estado do `CoverageEntry`/`OwnerLifecycle`. O commit atômico aplica internamente as transições canônicas `cleaned` e `rendered`; somente o QA terminal da Task 15 pode avançar para `final_verified`.

Implemente em `execution.py` o builder/produtor do `OwnerRepairRequest` já declarado na Task 9. `request_id` deriva da identidade `run/execution/page/source/owner` + stage/reason/evidence, `request_sha256` cobre o record inteiro e, quando o pedido nasce do QA, `issue_id` é obrigatório. Passe a popular o campo já reservado `PageExecutionResult.repair_requests`. A Task 11 o consome no ladder, sem redefinir o tipo em `repair.py` nem alterar o envelope.

Amplie o `PageCandidateTransaction` criado na Task 9 para ser também o compositor page-level imutável. `replace_owner_commit()` substitui somente o commit do owner solicitado; `compose()` sempre reabre `OriginalPageSnapshot`, aplica na ordem canônica todos os cleanups finais e depois todos os glyph patches finais. Ela nunca usa o candidate parcial como base. `commit_verified_generation()` continua sendo a única promoção privada de diretório/snapshot/ponteiro; falha não publica commit, final ou JSON parcial. A Task 15 usa e conclui essa mesma transação durante R0–R3. A Task 16 não a reabre: ela deriva o evidence tree já rebindado dentro do staging da `PublicationTransaction` de capítulo.

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
    tests/test_owner_source_replacement_enforce.py `
    tests/test_owner_source_replacement_fail_closed.py `
    tests/test_strip_owner_composition_integration.py `
    tests/test_owner_lifecycle.py `
    -q
} finally { Pop-Location }
```

Expected: PASS; rollback protege os pixels originais internamente e sempre agenda uma nova estratégia.

**Step 7: Commit**

```powershell
$guard = '.codex-tmp\universal-source-replacement-plan-guard\TaskOwnedGit.ps1'
& $guard -Action Stage -Task 10
git diff --cached --check
git commit -m "feat: make owner replacement atomic and recoverable"
& $guard -Action VerifyAfterCommit -Task 10
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
    attempt = build_repair_attempt(
        _fixture(), request=_initial_replacement_request(), policy=RepairBudgetPolicy.default(), strategy="R0"
    )
    assert _covers_all_source_glyphs(attempt.cleanup_mask)
    assert not _overlaps(attempt.cleanup_mask, _container_border_mask())


def test_r0_starts_from_one_canonical_initial_repair_request_consumed_exactly_once():
    result = run_repair_ladder(_owner_case(), max_strategy="R0")
    initial = result.repair_requests[0]
    assert initial.issue_id is None
    assert initial.reason == "initial_atomic_replacement"
    assert initial.failed_stage == "pre_execution"
    assert result.attempts[0].request_id == initial.request_id
    assert result.attempts[0].consumed_request_ids == (initial.request_id,)
    assert _count_request_consumptions(result, initial.request_id) == 1


def test_r1_expands_only_from_positive_source_support():
    policy = RepairBudgetPolicy.default()
    r0 = build_repair_attempt(
        _fixture_with_halo(), request=_initial_replacement_request(), policy=policy, strategy="R0"
    )
    r1 = build_repair_attempt(
        _fixture_with_halo(), request=_residual_request_after(r0), policy=policy, strategy="R1", previous=r0
    )
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
    attempt = build_repair_attempt(
        _fixture(), strategy="R0", request=request, policy=RepairBudgetPolicy.default()
    )
    assert attempt.request_id == "request-1"
    assert attempt.issue_id == "issue-1"
    assert request.request_id in attempt.consumed_request_ids


def test_repair_budget_rejects_duplicate_attempt_fingerprint():
    controller = RepairController(policy=RepairBudgetPolicy.default())
    attempt = _attempt(strategy="R1", variant="expanded", geometry=_geometry(), input_sha=_source_sha())
    controller.record(attempt)
    with pytest.raises(DuplicateRepairAttemptError):
        controller.record(attempt)


def test_persistent_transient_failure_exhausts_exact_budget_without_export():
    policy = RepairBudgetPolicy.default()
    with pytest.raises(RepairInfrastructureExhausted) as exc:
        run_repair_ladder(_always_transient_failure_case(), policy=policy)
    assert exc.value.attempt_count == policy.max_transient_retries
    assert not _public_export_exists()


def test_repair_history_and_page_result_reject_policy_hash_swap():
    policy = RepairBudgetPolicy.default()
    result = run_repair_ladder(_owner_case(), policy=policy)
    assert result.repair_budget_policy_sha256 == policy.policy_sha256
    assert all(attempt.repair_budget_policy_sha256 == policy.policy_sha256 for attempt in result.attempts)
    with pytest.raises(RepairPolicyIdentityError):
        PageExecutionResult.build_from(
            result.page_result,
            repair_budget_policy_sha256=_sha256("other-policy"),
        )
```

**Step 2: Confirmar RED**

```powershell
Push-Location pipeline
try {
  $redOutput = & .\venv\Scripts\python.exe -m pytest `
    tests/test_owner_repair.py `
    tests/test_owner_atomic_execution.py `
    tests/test_owner_mask.py `
    tests/test_inpaint_debug_residual.py `
    -q 2>&1
  $redExit = $LASTEXITCODE
  $redText = $redOutput | Out-String
  $redOutput
  if ($redExit -ne 1) { throw "Task 11 expected pytest exit 1 for a behavioral RED, got $redExit" }
  if ($redText -match '(?im)^ERROR(?:\s|:)|ERROR collecting|errors during collection|^E\s+fixture .* not found|UsageError|ModuleNotFoundError|ImportError') {
    throw 'Task 11 RED is structural rather than behavioral; create/fix the API skeleton and rerun the same nodeids'
  }
  if ($redText -notmatch '(?m)^FAILED\s+\S+::') { throw 'Task 11 RED did not report a targeted FAILED nodeid' }
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
class RepairBudgetPolicy:
    max_attempts_by_variant: tuple[tuple[str, int], ...] = (
        ("R0:precise", 1),
        ("R1:expanded", 2),
        ("R2:text_region", 2),
        ("R3:contextual", 2),
        ("R3:deterministic_interior_fill", 1),
        ("R3:deterministic_support_local_fill", 1),
    )
    max_geometry_expansions: int = 3
    max_transient_retries: int = 2
    transient_backoff_ms: tuple[int, ...] = (250, 500)
    policy_sha256: str = ""  # calculado do JSON canônico dos demais campos


def build_repair_attempt(
    case: OwnerRepairCase,
    *,
    request: OwnerRepairRequest,
    policy: RepairBudgetPolicy,
    strategy: RepairStrategy,
    variant: str = "default",
    previous: RepairAttempt | None = None,
) -> RepairAttempt: ...  # record final/schema já declarado na Task 9
```

Toda tentativa nasce de um `OwnerRepairRequest`; antes de R0 o coordenador cria uma request canônica `failed_stage="pre_execution"`, `reason="initial_atomic_replacement"`, `issue_id=None`, ligada ao binding/source original. Assim R0 não é uma exceção sem origem. `request_id` é obrigatório e aparece exatamente uma vez em `consumed_request_ids`; `issue_id` deve coincidir quando o request veio do QA. `attempt_fingerprint` é o hash canônico de request, strategy, variant, geometry e input; nunca execute duas vezes o mesmo fingerprint. Retries encadeados podem consumir requests adicionais, mas nenhum request pode desaparecer nem ser consumido por duas tentativas finais concorrentes. Nesta Task, implemente o builder/produtor do `RepairAttempt` já declarado na Task 9 e passe a popular os campos já reservados `PageExecutionResult.repair_history` e `repair_budget_policy_sha256`; toda tentativa repete esse hash, o result/proof o persistem e replay/config stale é rejeitado. O controller escolhe somente a próxima estratégia dentro de `RepairBudgetPolicy`, nunca coordenadas/exceções específicas da obra. Testes injetam scheduler sem sleep; produção usa apenas os backoffs explícitos do policy. Budget operacional esgotado lança `RepairInfrastructureExhausted` e não publica, enquanto dificuldade visual válida continua avançando até o fill determinístico de R3.

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
$guard = '.codex-tmp\universal-source-replacement-plan-guard\TaskOwnedGit.ps1'
& $guard -Action Stage -Task 11
git diff --cached --check
git commit -m "feat: add precise and expanded owner repair strategies"
& $guard -Action VerifyAfterCommit -Task 11
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
    attempt = build_repair_attempt(
        _multi_line_fragmented_source(),
        request=_r2_repair_request(),
        policy=RepairBudgetPolicy.default(),
        strategy="R2",
    )
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
  $redOutput = & .\venv\Scripts\python.exe -m pytest `
    tests/test_owner_repair.py `
    tests/test_owner_atomic_execution.py `
    tests/test_typesetting_renderer.py `
    tests/test_strip_owner_composition_integration.py `
    -q 2>&1
  $redExit = $LASTEXITCODE
  $redText = $redOutput | Out-String
  $redOutput
  if ($redExit -ne 1) { throw "Task 12 expected pytest exit 1 for a behavioral RED, got $redExit" }
  if ($redText -match '(?im)^ERROR(?:\s|:)|ERROR collecting|errors during collection|^E\s+fixture .* not found|UsageError|ModuleNotFoundError|ImportError') {
    throw 'Task 12 RED is structural rather than behavioral; create/fix the API skeleton and rerun the same nodeids'
  }
  if ($redText -notmatch '(?m)^FAILED\s+\S+::') { throw 'Task 12 RED did not report a targeted FAILED nodeid' }
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
$guard = '.codex-tmp\universal-source-replacement-plan-guard\TaskOwnedGit.ps1'
& $guard -Action Stage -Task 12
git diff --cached --check
git commit -m "feat: rebuild complete owner text regions"
& $guard -Action VerifyAfterCommit -Task 12
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
  $redOutput = & .\venv\Scripts\python.exe -m pytest `
    tests/test_owner_repair.py `
    tests/test_owner_atomic_execution.py `
    tests/test_owner_lifecycle.py `
    tests/test_strip_owner_composition_integration.py `
    -q 2>&1
  $redExit = $LASTEXITCODE
  $redText = $redOutput | Out-String
  $redOutput
  if ($redExit -ne 1) { throw "Task 13 expected pytest exit 1 for a behavioral RED, got $redExit" }
  if ($redText -match '(?im)^ERROR(?:\s|:)|ERROR collecting|errors during collection|^E\s+fixture .* not found|UsageError|ModuleNotFoundError|ImportError') {
    throw 'Task 13 RED is structural rather than behavioral; create/fix the API skeleton and rerun the same nodeids'
  }
  if ($redText -notmatch '(?m)^FAILED\s+\S+::') { throw 'Task 13 RED did not report a targeted FAILED nodeid' }
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

Se o R3 contextual ainda deixar suporte-fonte confirmado, a mesma estratégia executa a variante `deterministic_interior_fill`: estima a cor/campo-base pelos pixels livres de glyph, preenche todo o interior seguro (não somente a máscara OCR) e reaplica o target com estilo-base. Essa variante sacrifica detalhe decorativo interno antes de aceitar inglês, mas preserva borda e pixels externos. Pixels de glyph confirmados não podem ser tratados como arte protegida apenas porque há arte subjacente; já ícones/personagens/separadores genuínos fora do suporte-fonte permanecem hard-protected. Se a linha residual estiver fora do interior calculado, o controller amplia/corrige o container ou materializa outro owner. Ao atingir `RepairBudgetPolicy.max_geometry_expansions`, use uma única tentativa `deterministic_support_local_fill`: union do suporte glyph confirmado + margem determinística, hard-clipped à página e ao menor container conservador executável, reconstrução local e reaplicação do target. Fingerprint repetido falha antes de executar; não existe loop implícito.

**Step 4: Definir a política terminal sem inglês**

- R0/R1/R2 falharam: executar R3 automaticamente.
- R3 encontra proteção conservadora sobre glyph source confirmado: recalcular interior/proteção e executar; encontra arte genuína fora do suporte que seria alterada: refinar a reconstrução preservando-a. Uma imagem difícil, container ausente ou conflito visual não pode lançar erro terminal: após tentativas limitadas, executar `deterministic_support_local_fill` na região confirmada. `UnsafeProtectedArtGeometryError` fica restrito a contrato estrutural inválido (shape/dtype/bounds impossíveis, hash/bytes corrompidos), nunca à ambiguidade de conteúdo ou à ausência de uma borda ideal.
- R3 contextual ainda tem source: executar `deterministic_interior_fill` e provar cobertura integral do interior; uma nova linha fora dele reabre coverage/container de forma monotônica.
- Falha técnica transitória: retry da mesma página a partir dos pixels originais até `RepairBudgetPolicy.max_transient_retries`; persistência lança `RepairInfrastructureExhausted` e zero export.
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
$guard = '.codex-tmp\universal-source-replacement-plan-guard\TaskOwnedGit.ps1'
& $guard -Action Stage -Task 13
git diff --cached --check
git commit -m "feat: guarantee container interior recovery as final fallback"
& $guard -Action VerifyAfterCommit -Task 13
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
  $redOutput = & .\venv\Scripts\python.exe -m pytest `
    tests/test_language_residual.py `
    tests/test_final_pixel_qa.py `
    tests/test_final_pixel_observer.py `
    -q 2>&1
  $redExit = $LASTEXITCODE
  $redText = $redOutput | Out-String
  $redOutput
  if ($redExit -ne 1) { throw "Task 14 expected pytest exit 1 for a behavioral RED, got $redExit" }
  if ($redText -match '(?im)^ERROR(?:\s|:)|ERROR collecting|errors during collection|^E\s+fixture .* not found|UsageError|ModuleNotFoundError|ImportError') {
    throw 'Task 14 RED is structural rather than behavioral; create/fix the API skeleton and rerun the same nodeids'
  }
  if ($redText -notmatch '(?m)^FAILED\s+\S+::') { throw 'Task 14 RED did not report a targeted FAILED nodeid' }
} finally { Pop-Location }
```

Expected: QA atual depende de overlap bruto de tokens e pode duplicar detector/component probe.

**Step 3: Implementar o builder de `LanguageResidualIssue`**

Use o record final e o decoder já implementados em `ownership/model.py` na Task 9; esta Task implementa em `qa/language_residual.py` somente o factory `build_language_residual_issue(...)` e o classificador, sem editar/redefinir campos, schema ou decoder. O factory chama o constructor validado do record. `execution_id` é obrigatório e atual, `observed_text_sha256`, `region_sha256`, `source_binding_sha256` e `issue_sha256` são recalculados de conteúdo canônico. Classificar pelo `TranslationBinding`, role e policies explícitas. A identidade funcional da issue é run/execution/página/hash final + owner ou região canônica + kind + hash do source binding. `invocation_id`s, providers e textos observados são evidências acumuladas em tuplas, não parte da chave; duas invocações independentes sobre o mesmo defeito geram um único reparo. Todo `invocation_id` resolve uma `final_qa_ocr_invocation` persistida da mesma execução/página/pixels; invocation de coverage/source herdada não satisfaz o QA terminal.

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
$guard = '.codex-tmp\universal-source-replacement-plan-guard\TaskOwnedGit.ps1'
& $guard -Action Stage -Task 14
git diff --cached --check
git commit -m "feat: detect final source language by owner binding"
& $guard -Action VerifyAfterCommit -Task 14
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


def test_final_qa_journal_closes_request_invocation_probe_issue_and_terminal_proof():
    result = run_page_owner_pipeline(_page_with_first_pass_residual(), _services())
    requests = {item.invocation_id: item for item in result.final_qa_ocr_requests}
    invocations = {item.request.invocation_id: item for item in result.final_qa_ocr_invocations}
    issues = {item.issue_id: item for item in result.language_residual_issues}
    assert requests.keys() == invocations.keys()
    probe_invocation_counts = Counter(probe.ocr_invocation_id for probe in result.qa_probes)
    assert probe_invocation_counts == Counter({invocation_id: 1 for invocation_id in invocations})
    for probe in result.qa_probes:
        invocation = invocations[probe.ocr_invocation_id]
        assert probe.root_input_pixel_sha256 == invocation.request.root_input_pixel_sha256
        assert probe.fresh_ocr_attempt_ids == tuple(item.attempt_id for item in invocation.attempts)
        assert probe.fresh_ocr_attempt_chain_sha256 == invocation.attempt_chain_sha256
        assert set(probe.issue_ids) <= issues.keys()
    final_probe = result.qa_probes[-1]
    assert result.terminal_proof.final_qa_probe_id == final_probe.probe_id
    assert result.terminal_proof.fresh_ocr_invocation_id == final_probe.ocr_invocation_id


def test_qa_repair_request_resolves_one_persisted_language_issue_from_same_execution():
    result = run_page_owner_pipeline(_page_with_first_pass_residual(), _services())
    issues = {item.issue_id: item for item in result.language_residual_issues}
    qa_requests = [item for item in result.repair_requests if item.issue_id is not None]
    assert qa_requests
    for request in qa_requests:
        issue = issues[request.issue_id]
        assert issue.execution_id == request.execution_id == result.execution_id
        assert issue.page_output_pixel_sha256


@pytest.mark.parametrize(
    "tamper",
    [
        "missing_request",
        "missing_invocation",
        "extra_request",
        "extra_invocation",
        "duplicate_probe_invocation",
        "wrong_attempt_ids",
        "wrong_issue",
        "wrong_probe",
    ],
)
def test_final_result_rejects_broken_final_qa_journal_even_when_outer_hashes_are_recomputed(tamper):
    with pytest.raises(PagePipelineIdentityError):
        PageExecutionResult.build(**_tampered_final_qa_result_parts(tamper, recompute_outer_hashes=True))


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
    tampered = dataclasses.replace(composition, **{field: (_sha256("stale"),)})
    with pytest.raises(TerminalVerificationIdentityError):
        TerminalPixelProof.build_from_persisted_candidate(
            candidate,
            _fresh_ocr_for(candidate),
            page_composition=tampered,
            bindings=_bindings(),
            materializations=_materializations(),
        )


@pytest.mark.parametrize("tamper", ["verdict_hash", "repair_policy_hash", "verification_policy_hash"])
def test_terminal_proof_rejects_tampered_verdict_or_policy_chain(tamper, tmp_path):
    candidate = _persist_candidate(tmp_path)
    kwargs = _valid_verdict_and_policy_chain()
    kwargs = _tamper_verdict_or_policy(kwargs, tamper, replacement=_sha256("stale"))
    with pytest.raises(TerminalVerificationIdentityError):
        TerminalPixelProof.build_from_persisted_candidate(
            candidate,
            _fresh_ocr_for(candidate),
            **kwargs,
        )


def test_terminal_proof_rejects_changed_pixels_when_source_is_still_legible(tmp_path):
    candidate = _persist_candidate(tmp_path, pixels=_source_slightly_blurred_but_legible())
    with pytest.raises(SourceResidualStillMaterialError):
        TerminalPixelProof.build_from_persisted_candidate(
            candidate,
            _fresh_ocr_for(candidate, source_lines=()),
            source_support=_complete_source_support(),
        )


@pytest.mark.parametrize("target_defect", ["zero_alpha", "zero_contrast", "clipped", "outside_safe_region"])
def test_terminal_proof_rejects_nonmaterial_or_misplaced_target_patch(target_defect, tmp_path):
    candidate, materialization = _candidate_with_target_defect(tmp_path, target_defect)
    with pytest.raises(TargetMaterializationError):
        TerminalPixelProof.build_from_persisted_candidate(
            candidate,
            _fresh_ocr_for(candidate),
            materializations=(materialization,),
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


@pytest.mark.parametrize("phase", ["after_generation_rename", "before_pointer_replace", "after_pointer_replace"])
def test_page_generation_crash_never_exposes_final_without_matching_result(tmp_path, phase):
    before = _snapshot_current_page_generation(tmp_path)
    with pytest.raises(InjectedPageCommitFailure):
        _commit_page_generation_with_fault(tmp_path, phase)
    recover_page_candidate_transaction(tmp_path, page_id="page_001")
    current = _load_current_page_generation_if_any(tmp_path)
    assert current is None or _final_execution_evidence_and_pointer_match(current)
    assert current is None or current.evidence_ref.read_verified(tmp_path).status == "final_verified"
    assert current is None or current in (before, _expected_new_page_generation(tmp_path))


@pytest.mark.parametrize(
    "tamper",
    [
        "current_pointer_symlink_escape",
        "page_generation_traversal",
        "staging_symlink_escape",
        "stale_page_id",
        "stale_marker_generation",
        "stale_page_generation_id",
        "nfc_casefold_target_collision",
    ],
)
def test_page_candidate_recovery_rejects_untrusted_paths_or_identity_before_mutation(tmp_path, tamper):
    private_root, outside, journal = _private_execution_with_tampered_page_recovery(tmp_path, tamper)
    outside_before = _snapshot_tree(outside)
    private_before = _snapshot_tree(private_root, exclude=(journal,))
    with _spy_file_opens_under(outside) as outside_opens:
        with pytest.raises(PageCandidateRecoveryError):
            recover_page_candidate_transaction(private_root, page_id="page_001")
    assert outside_opens == ()
    assert _snapshot_tree(outside) == outside_before
    assert _snapshot_tree(private_root, exclude=(journal,)) == private_before
    assert journal.exists()
```

**Step 2: Confirmar RED**

```powershell
Push-Location pipeline
try {
  $redOutput = & .\venv\Scripts\python.exe -m pytest `
    tests/test_owner_lifecycle_e2e.py::test_final_qa_journal_closes_request_invocation_probe_issue_and_terminal_proof `
    tests/test_owner_lifecycle_e2e.py::test_final_result_rejects_broken_final_qa_journal_even_when_outer_hashes_are_recomputed `
    tests/test_owner_repair_controller.py::test_page_candidate_recovery_rejects_untrusted_paths_or_identity_before_mutation `
    -q 2>&1
  $redExit = $LASTEXITCODE
  $redText = $redOutput | Out-String
  $redOutput
  if ($redExit -ne 1) { throw "Task 15 expected pytest exit 1 for a behavioral RED, got $redExit" }
  if ($redText -match '(?im)^ERROR(?:\s|:)|ERROR collecting|errors during collection|^E\s+fixture .* not found|UsageError|ModuleNotFoundError|ImportError') {
    throw 'Task 15 RED is structural rather than behavioral; create/fix the API skeleton and rerun the same nodeids'
  }
  if ($redText -notmatch '(?m)^FAILED\s+\S+::') { throw 'Task 15 RED did not report a targeted FAILED nodeid' }
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

O controller processa todos os issues de uma página via `PageCandidateTransaction`: cada owner reparado substitui um commit no conjunto imutável e a página é recomposta do original com todos os cleanups e targets finais. Assim, reparar um owner não restaura nem duplica outro. Reexecute aqui o RED multi-owner da Task 10 com B chegando a R3.

Implemente os builders/produtores de `FinalQAProbe` e `FinalReplacementVerdict` usando os records finais já declarados na Task 9, sem redefinir seus schemas. File hash e pixel hash nunca são comparados entre si; exige-se `root_input_pixel_sha256 == candidate_pixel_sha256`, e cada tentativa física é validada pela cadeia da única invocation referenciada. Passe a popular `PageExecutionResult.final_qa_ocr_requests`, `final_qa_ocr_invocations`, `language_residual_issues`, `qa_probes`, `final_replacement_verdicts` e `replacement_verification_policy_sha256`. Cada probe resolve uma única invocation final e sua request; cada issue resolve somente invocations dessa coleção; cada repair request com `issue_id` resolve uma única issue da mesma execução. A promoção terminal exige exatamente um verdict por binding e ao menos um probe fresco ligado ao candidate final, sem mudar o schema top-level do evidence.

**Step 4: Implementar o builder de produção de `TerminalPixelProof` e a promoção terminal**

Exigir conjuntamente:

1. suporte source completo recomposto a partir das evidências e coberto pela ação de cleanup/rebuild;
2. residual/ink pós-cleanup recomputado sobre todo esse suporte e abaixo do limiar material; uma diferença arbitrária de pixels não satisfaz;
3. patch de glyphs PT-BR com delta/máscara não vazios, alpha material, contraste contra `cleanup_base`, zero clipping e contenção na safe region;
4. cadeia de hashes cleanup → render → página final coerente;
5. conjuntos ordenados de `translation_binding_sha256`, source/target payload hashes, glyph-patch hashes, materialization hashes e `verdict_sha256` idênticos em bindings → commits → `PageCompositionSnapshot` → verdicts → proof, com `repair_budget_policy_sha256` e `replacement_verification_policy_sha256` explícitos;
6. OCR fresco com `observer_available=true`, `coverage_complete=true`, nova `invocation_id`, cache desabilitado, tentativa full-page física obrigatória, a mesma identidade run/execution/page/source e `OCRRequest.root_input_pixel_sha256` exatamente igual ao hash RGB do `candidate.png` redecodificado; request, invocation, attempts e issues são anexados ao journal final antes do probe; cada `OCRAttempt` é recalculado/ligado ao transform e aos pixels físicos, sem source-only tokens materiais;
7. ledger sem texto material não possuído.

O target OCR pode ser incompleto somente se 1–5 e 7 forem válidos, a invocation física do item 6 estiver completa e o item 6 não contiver source-only tokens materiais. Isso evita loop por fonte estilizada que o OCR não lê sem transformar target invisível em sucesso. OCR source positivo ou residual visual ambíguo sempre reabre o reparo e avança até R3.

Implemente `TerminalPixelProof.build_from_persisted_candidate(...)` como a única fábrica usada em produção; a construção direta permanece restrita a fixtures. Ela recebe explicitamente o `FinalQAProbe` final e as coleções finais persistíveis, reabre o candidate, recalcula `fresh_ocr_root_input_pixel_sha256`, `fresh_ocr_attempt_chain_sha256`, `fresh_ocr_payload_sha256` e `proof_sha256`, valida a identidade completa e exige `final_qa_probe_id == probe.probe_id`, `fresh_ocr_invocation_id == probe.ocr_invocation_id` e `fresh_ocr.request.root_input_pixel_sha256 == candidate.pixel_sha256 == final_page_pixel_sha256`. Para cada attempt, reexecuta deterministicamente bbox/transform sobre o candidate redecodificado e compara parent/input/transform hashes, shape/mode e `provider_called=true`; exige ao menos full-page físico e rejeita prova baseada só em cache/heurística. Também compara, sem rederivar de texto solto, todos os hashes ordenados de binding/source/target/glyph/materialization/verdict contra commits, `PageCompositionSnapshot` e verdicts, liga `cleanup_base_sha256`/`composition_sha256` aos artefatos reais e exige os hashes exatos das políticas de budget e verificação. Depois, `PageExecutionResult.promote_final(...)` cria atomicamente o par coerente `status="final_verified"` + journal QA + `final_page` + `terminal_proof`; nunca altere um resultado candidato no lugar.

Após cada composição candidata, grave PNG lossless em `<run>/.staging/<execution_id>/<page_id>/<attempt_id>/candidate.tmp.png`, faça rename atômico para `candidate.png` **dentro desse staging não publicável**, reabra/decodifique os bytes persistidos, recalcule o hash e rode o observer/`TerminalPixelProof` sobre essa decodificação — nunca apenas sobre o ndarray em memória. A raiz privada comum `<run>/.owner-executions/<execution_id>/` e seu único `artifact_generation.json` são criados/validados uma vez pelo coordenador antes das páginas. Só depois de `final_verified`, `PageCandidateTransaction` grava `final.png`, `execution_result.json`, `page_execution_evidence.json` e refs de estágios dentro de um diretório temporário da página, com paths relativos à raiz comum; fsynca, renomeia o diretório para `.page-generations/<page_id>/<page_generation_id>/` e substitui atomicamente `.page-current/<page_id>/current.json`. A mesma operação retorna `PageExecutionEvidenceRef`, contendo IDs comuns da geração, `page_generation_id` e hashes do pointer/snapshot, para `OutputPage.owner_page_evidence_ref`; nenhum caller fabrica essa ponte em memória. O evidence snapshot contém somente refs relativas à raiz privada comum e hashes; não duplica PNG inline. O leitor ignora page-generations sem pointer; recovery remove órfãos ou completa um pointer já journaled. Nunca faça renames independentes de final, evidence e pointer. Essa promoção page-local **não** publica o capítulo: somente o boundary real da Task 9 publica a geração completa do projeto. `candidate_ready`, `repair_pending`, prova ausente/incoerente, observer indisponível, coverage incompleta ou identidade stale são estados não exportáveis. Em falha, remova somente o attempt staging resolvido/validado dentro da raiz da run; na inicialização, `recover_owner_staging()` e `recover_page_candidate_transaction()` tratam órfãos daquela execution sem tocar runs anteriores nem diretórios públicos. Erros operacionais abortam sem export, não contam como OCR negativo e não preservam conteúdo source como saída.

`recover_page_candidate_transaction()` trata seu journal como input hostil: antes de qualquer open/rename/remove, valida schema/hash, marker/store/generation comum, run/execution/page/transaction, `page_generation_id`, paths allowlisted de staging/generation/current, contenção real e unicidade NFC/casefold. Symlink/junction/reparse, traversal, pointer/page identity stale ou target fora da raiz comum preserva journal e toda árvore, não abre o alvo externo e lança `PageCandidateRecoveryError`.

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
$guard = '.codex-tmp\universal-source-replacement-plan-guard\TaskOwnedGit.ps1'
& $guard -Action Stage -Task 15
git diff --cached --check
git commit -m "feat: repair final language residuals before export"
& $guard -Action VerifyAfterCommit -Task 15
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
    evidence, generation_root = _verified_evidence()
    artifacts = _write_page_artifacts(evidence, generation_root=generation_root)
    assert artifacts.coverage.page_id == artifacts.execution.page_id == artifacts.final_qa.page_id
    assert artifacts.execution.final_pixel_sha256 == artifacts.final_qa.page_output_pixel_sha256
    assert artifacts.final_page.final_file_sha256 == sha256_bytes(artifacts.final_page.lossless_png_bytes)
    assert artifacts.execution.cleanup_base_sha256 == canonical_page_sha256(
        decode_lossless(artifacts.cleanup_base_png_bytes)
    )


def test_every_repair_request_target_materialization_and_commit_is_accounted_for():
    evidence, generation_root = _verified_evidence(with_repair=True)
    artifacts = _write_page_artifacts(evidence, generation_root=generation_root)
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


def test_coverage_and_final_qa_histories_are_persisted_without_reconstruction():
    evidence, generation_root = _verified_evidence(with_repair=True)
    artifacts = _write_page_artifacts(evidence, generation_root=generation_root)
    assert artifacts.coverage_ledgers == evidence.read_verified(generation_root).coverage.ledger_history
    assert artifacts.coverage_recovery_requests == evidence.read_verified(generation_root).coverage.recovery_requests
    assert artifacts.coverage_recovery_decisions == evidence.read_verified(generation_root).coverage.recovery_decisions
    assert artifacts.coverage_pending_request_ids == evidence.read_verified(generation_root).coverage.pending_request_ids
    assert artifacts.final_qa_ocr_requests == evidence.read_verified(generation_root).final_qa_ocr_requests
    assert artifacts.final_qa_ocr_invocations == evidence.read_verified(generation_root).final_qa_ocr_invocations
    assert artifacts.language_residual_issues == evidence.read_verified(generation_root).language_residual_issues
    assert _terminal_invocation(artifacts).request.invocation_id == artifacts.terminal_proof.fresh_ocr_invocation_id


def test_page_artifact_manifest_excludes_itself_and_chapter_manifest_covers_generation_marker():
    evidence, generation_root = _verified_evidence()
    artifacts = _write_page_artifacts(evidence, generation_root=generation_root)
    assert "evidence/pages/page_001/artifact_manifest.json" not in artifacts.artifact_manifest.relative_paths
    chapter_manifest = _build_chapter_asset_manifest(generation_root)
    assert "artifact_generation.json" in chapter_manifest.relative_paths
    assert "evidence/pages/page_001/artifact_manifest.json" in chapter_manifest.relative_paths


def test_same_owner_with_stale_target_binding_or_materialization_hash_is_rejected():
    evidence, generation_root = _verified_evidence(tamper="stale_target_hash")
    with pytest.raises(PageArtifactIntegrityError):
        _write_page_artifacts(evidence, generation_root=generation_root)


def test_artifact_writer_serializes_only_evidence_reopened_from_page_execution_snapshot():
    result = _verified_result()
    generation_root = _generation_root_for(result)
    expected = result.to_canonical_dict()
    evidence = PageExecutionEvidenceSnapshot.build(result)
    del result
    artifacts = _write_page_artifacts(evidence, generation_root=generation_root)
    assert _canonical_evidence_records(artifacts) == _expected_evidence_records_from(expected)
    assert artifacts.execution.page_result_sha256 == evidence.page_result_sha256


def test_all_canonical_visual_stages_resolve_to_lossless_hash_verified_artifacts():
    evidence, generation_root = _verified_evidence()
    artifacts = _write_page_artifacts(evidence, generation_root=generation_root)
    stages = artifacts.visual_stages
    assert set(stages) == {"original", "inpaint", "typeset", "page_composition", "persisted_final"}
    for name, stage in stages.items():
        assert stage.artifact_ref.load_verified(generation_root).shape[:2] == _page_shape()
        assert stage.pixel_sha256 == stage.artifact_ref.pixel_sha256
        if stage.alias_of is not None:
            assert stage.alias_reason
            assert stage.pixel_sha256 == stages[stage.alias_of].pixel_sha256


def test_artifact_writer_rejects_missing_stage_instead_of_reconstructing_from_debug():
    evidence, generation_root = _verified_evidence(tamper="missing_canonical_inpaint_stage")
    with pytest.raises(PageArtifactIntegrityError):
        _write_page_artifacts(evidence, generation_root=generation_root)


def test_artifacts_preserve_root_and_every_physical_ocr_attempt_hash():
    result = _verified_result_with_full_page_crop_and_variants()
    generation_root = _generation_root_for(result)
    evidence = PageExecutionEvidenceSnapshot.build(result)
    artifacts = _write_page_artifacts(evidence, generation_root=generation_root)
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
    evidence, generation_root = _verified_evidence()
    artifacts = _write_page_artifacts(evidence, generation_root=generation_root)
    observations = _index_by_id(artifacts.page_owner_observations)
    for observation_id in _all_observation_ids(artifacts.coverage, artifacts.owner_graph):
        matches = observations[observation_id]
        assert len(matches) == 1
        assert _identity(matches[0]) == _page_identity(artifacts.coverage)
        assert matches[0].invocation_id in _invocation_ids(artifacts.ocr_invocations)


def test_replay_artifacts_keep_current_publication_execution_and_parent_content_origin():
    evidence, generation_root = _verified_replay_evidence(parent_execution_id="exec-off", execution_id="exec-render")
    artifacts = _write_page_artifacts(evidence, generation_root=generation_root)
    assert artifacts.execution.publication_execution_id == "exec-render"
    assert artifacts.execution.replay_source_page_evidence_sha256
    for record in (*artifacts.ocr_requests, *artifacts.ocr_invocations, *artifacts.translation_attempts):
        assert record.origin_execution_id == "exec-off"
    for path in _json_and_jsonl_paths(artifacts):
        payloads = _read_records(path)
        assert all(item["publication_execution_id"] == "exec-render" for item in payloads)
        assert all(item.get("origin_execution_id", "exec-render") in {"exec-off", "exec-render"} for item in payloads)
    assert _no_inherited_record_was_retagged(artifacts)


def test_export_gate_accepts_only_final_verified_page_results():
    assert evaluate_export_gate(_chapter_with_all_pages_verified())["status"] == "PASS"
    blocked = evaluate_export_gate(_chapter_with_repair_pending_page())
    assert blocked["status"] == "BLOCK"
    assert "page_lifecycle_incomplete" in _reasons(blocked)


def test_export_gate_rejects_page_set_not_bijective_with_source_manifest():
    chapter = _chapter_with_all_pages_verified()
    for tamper in ("missing", "duplicate", "extra", "reordered"):
        blocked = evaluate_export_gate(_tamper_chapter_pages(chapter, tamper))
        assert blocked["status"] == "BLOCK"
        assert "source_manifest_page_set_mismatch" in _reasons(blocked)


def test_export_gate_counts_only_issues_reachable_from_exact_terminal_probe():
    chapter = _chapter_repaired_after_historical_source_issue()
    page = chapter.pages[0]
    assert page.language_residual_issues
    terminal_probe = _probe_by_id(page, page.terminal_proof.final_qa_probe_id)
    assert terminal_probe.issue_ids == ()
    gate = evaluate_export_gate(chapter)
    assert gate["english_dialogue_residual_count"] == 0
    assert gate["status"] == "PASS"


def test_export_gate_blocks_source_issue_reachable_from_exact_terminal_probe():
    chapter = _chapter_with_terminal_probe_source_issue()
    gate = evaluate_export_gate(chapter)
    assert gate["english_dialogue_residual_count"] == 1
    assert gate["status"] == "BLOCK"


def test_export_gate_rejects_missing_or_ambiguous_terminal_probe_link_instead_of_inferring_latest():
    for tamper in ("missing_probe", "duplicate_probe_id", "proof_points_to_historical_probe"):
        blocked = evaluate_export_gate(_chapter_with_tampered_terminal_probe_link(tamper))
        assert blocked["status"] == "BLOCK"
        assert "terminal_probe_integrity_error" in _reasons(blocked)


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
  $redOutput = & .\venv\Scripts\python.exe -m pytest `
    tests/test_final_pixel_export_gate.py::test_export_gate_counts_only_issues_reachable_from_exact_terminal_probe `
    tests/test_final_pixel_export_gate.py::test_export_gate_blocks_source_issue_reachable_from_exact_terminal_probe `
    tests/test_final_pixel_export_gate.py::test_export_gate_rejects_missing_or_ambiguous_terminal_probe_link_instead_of_inferring_latest `
    -q 2>&1
  $redExit = $LASTEXITCODE
  $redText = $redOutput | Out-String
  $redOutput
  if ($redExit -ne 1) { throw "Task 16 expected pytest exit 1 for a behavioral RED, got $redExit" }
  if ($redText -match '(?im)^ERROR(?:\s|:)|ERROR collecting|errors during collection|^E\s+fixture .* not found|UsageError|ModuleNotFoundError|ImportError') {
    throw 'Task 16 RED is structural rather than behavioral; create/fix the API skeleton and rerun the same nodeids'
  }
  if ($redText -notmatch '(?m)^FAILED\s+\S+::') { throw 'Task 16 RED did not report a targeted FAILED nodeid' }
  .\venv\Scripts\python.exe -m pytest `
    tests/test_page_owner_artifacts.py `
    tests/test_final_pixel_export_gate.py `
    tests/test_owner_visual_matrix_tool.py `
    -q
} finally { Pop-Location }
```

Expected: artefatos podem divergir e a matriz aceita o mesmo hash como casos visuais diferentes.

**Step 3: Definir artefatos canônicos por página**

Nesta altura `PageExecutionEvidenceSnapshot` é a autoridade persistível do journal frozen completo. `_write_page_artifacts(evidence: PageExecutionEvidenceSnapshot, *, generation_root: Path)` aceita somente esse snapshot já rebindado para a geração **staged do capítulo**, chama `evidence.read_verified(generation_root)` antes de qualquer escrita e deriva cada arquivo do retorno reconstruído. Não recebe `PageExecutionResult` vivo, `OutputPage`, coleções paralelas ou debug global/bands; não reconstrói evidência ausente. Campo obrigatório/extra, ref fora da raiz, cadeia divergente ou arquivo do evidence tree que não tenha origem 1:1 numa seção/ref do snapshot é `PageArtifactIntegrityError` antes de qualquer publicação. Esta função apenas escreve dentro da geração já corrente de `PublicationTransaction`; ela não chama, promove nem substitui `PageCandidateTransaction/current.json`. A transação page-local da Task 15 já terminou na raiz privada antes do adapter; a transação de capítulo da Task 9 controla staging/promote público. Não há terceiro rename ou writer por arquivo.

Persistir no evidence tree da geração staged/publicada, sem incluir outputs no Git:

```text
evidence/pages/page_###/
  artifact_manifest.json
  page_execution_evidence.json
  coverage_result.json
  coverage_ledgers.jsonl
  coverage_recovery_requests.jsonl
  coverage_recovery_decisions.jsonl
  coverage_pending_request_ids.json
  ocr_requests.jsonl
  ocr_invocations.jsonl
  page_owner_observations.jsonl
  owner_graph.json
  translation_attempts.jsonl
  translation_bindings.json
  repair_requests.jsonl
  repair_attempts.jsonl
  owner_target_materialization.jsonl
  execution_result.json
  original.ref.json
  cleanup_base.png
  inpaint.png
  target_typeset.png
  page_composition.json
  page_composition.png
  final_qa_ocr_requests.jsonl
  final_qa_ocr_invocations.jsonl
  language_residual_issues.jsonl
  final_qa_probes.jsonl
  final_pixel_proof.json
  final_replacement_verdicts.jsonl
  persisted_final.ref.json
```

Cada arquivo possui envelope com `run_id`, `publication_execution_id` (o `execution_id` da geração atual), `page_id`, `page_source_sha256`, versão do schema e hashes de entrada/saída. `page_execution_evidence.json` é exatamente `PageExecutionEvidenceSnapshot.canonical_json_bytes` e a raiz dos demais records; cada arquivo derivado deve ser reproduzível 1:1 a partir dele e, exceto o próprio `artifact_manifest.json`, ter seu hash registrado nesse manifest. Records internos preservam ainda seu `origin_execution_id`: em execução fresca ele coincide com a atual; em replay, coverage/OCR/source/translation/owner records continuam apontando à execução `off`, enquanto refs copiadas, render, composição, QA e envelope usam a execução nova. É proibido reetiquetar records herdados. `coverage_result.json` serializa a cabeça e os hashes/IDs dos históricos; `coverage_ledgers.jsonl` preserva a ledger inicial e todos os sucessores em ordem, permitindo reabrir `parent_ledger_sha256`; `coverage_recovery_requests.jsonl` preserva todas as requests, concluídas ou pending, `coverage_recovery_decisions.jsonl` preserva todas as decisões e `coverage_pending_request_ids.json` contém apenas a visão derivada. OCR requests/invocations de coverage são persistidos em `ocr_*.jsonl` e devem coincidir byte/hash com os IDs citados pela coverage. OCR requests persistem `root_input_pixel_sha256`; invocations persistem cada `OCRAttempt` físico com attempt/parent/input hashes, `OCRTransformSpec.canonical_json_bytes` + hash, bbox, shape/mode, `provider_called` e `cache_hit`; observations referenciam o attempt exato. Persistir apenas o hash do transform sem seu spec reproduzível é inválido.

`final_qa_ocr_requests.jsonl`, `final_qa_ocr_invocations.jsonl` e `language_residual_issues.jsonl` são coleções distintas, pertencentes à execução atual, e fecham `final_qa_probes.jsonl`; nunca são reconstruídas a partir do OCR source. `translation_attempts.jsonl` fecha cada ID citado pelos bindings. No probe terminal, o root coincide com os pixels resolvidos por `persisted_final.ref.json` e `fresh_ocr_attempt_chain_sha256` cobre a sequência física ordenada da invocation final referenciada. `repair_requests` liga cada `issue_id/request_id` ao issue persistido e ao attempt que a consumiu; `owner_target_materialization` prova exatamente um glyph patch final por binding e repete os hashes exatos de binding/target/glyph; `cleanup_base.png` é a composição page-space de todos os cleanups verificados antes dos glyph patches, com `cleanup_base_sha256` no execution result; `page_composition` é o `PageCompositionSnapshot` autoritativo e liga todos os commits/hashes à página; `final_replacement_verdicts` fecha cada owner com os mesmos hashes e policies. Cada path do evidence tree entra em `ChapterAssetManifest` com `project_reference=None` + `evidence_reference`; `persisted_final.ref.json` contém somente store/generation/relative-path + hashes e resolve a ref única do PNG final publicado contra a raiz da geração, evitando duplicar/fabricar pixels.

`artifact_manifest.json` aponta para os artefatos lossless canônicos `original`, `inpaint`, `typeset`, `page_composition` e `persisted_final`, mas exclui deliberadamente seu próprio path/hash; quem fecha seu arquivo é `ChapterAssetManifest`, evitando self-reference. `artifact_generation.json` também é obrigatório no `ChapterAssetManifest`. `original.ref.json` liga ao item correspondente do `ChapterSourceManifest`. Quando arquitetura page-first tornar `typeset`, `page_composition` e `final` pixel-identical, use a mesma artifact ref e declare `alias_of` + `alias_reason`; não duplique bytes nem fabrique um estágio. `coverage_ocr_overlay` não integra a cadeia de pixels: o auditor o deriva, marca `derived=true` e liga seu hash aos inputs canônicos. A base limpa é artefato interno de replay, nunca export final isolado.

**Step 4: Integrar com o gate**

O export gate não decide reparo. Ele recebe o `ChapterSourceManifest` e o bundle, valida primeiro a bijeção/cardinalidade ordenada e somente então valida que todas as páginas estão `final_verified`. Para cada página, resolve **por ID exato** `terminal_proof.final_qa_probe_id` em exatamente um `FinalQAProbe`, valida que sua invocation/root/attempt chain é a mesma do proof e calcula residuals somente dos `LanguageResidualIssue.issue_id` citados por esse probe. Issues históricas de probes reparados continuam persistidas para auditoria, mas não entram nas métricas terminais; probe ausente/duplicado, ID histórico apontado pelo proof ou fallback por posição/ordem bloqueia por integridade. Depois exige que:

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

Aceite somente se o gate receber páginas verificadas, não houver claims distintos ou linhagem ambígua reutilizando o mesmo arquivo e nenhum verdict for preenchido automaticamente. Estágios legitimamente pixel-idênticos continuam permitidos quando `alias_of/alias_reason` e a provenance da ref forem exatos.

**Step 8: Commit**

```powershell
$guard = '.codex-tmp\universal-source-replacement-plan-guard\TaskOwnedGit.ps1'
& $guard -Action Stage -Task 16
git diff --cached --check
git commit -m "feat: persist verifiable page replacement artifacts"
& $guard -Action VerifyAfterCommit -Task 16
```

### Task 17: Tornar `page_pipeline.py` o único caminho `enforce`

**Files:**

- Modify: `pipeline/main.py`
- Modify: `pipeline/strip/run.py`
- Modify: `pipeline/strip/process_bands.py`
- Modify: `pipeline/strip/page_pipeline.py`
- Modify: `pipeline/ownership/consensus_v2.py`
- Modify: `pipeline/ownership/owner_builder.py`
- Read only: `pipeline/ownership/reconcile.py`
- Modify: `pipeline/typesetter/owner_style.py`
- Modify: `pipeline/qa/style_fidelity.py`
- Test: `pipeline/tests/test_main_emit.py`
- Test: `pipeline/tests/test_strip_owner_control_plane.py`
- Test: `pipeline/tests/test_page_owner_pipeline.py`
- Test: `pipeline/tests/test_strip_owner_composition_integration.py`
- Test: `pipeline/tests/test_owner_style_profile.py`
- Test: `pipeline/tests/test_style_fidelity_qa.py`
- Create: `pipeline/tests/test_owner_content_replay.py`

**Step 1: Escrever testes RED de autoridade única**

```python
def test_enforce_mode_calls_only_page_owner_pipeline(monkeypatch):
    legacy = MagicMock(wraps=strip_run._resolve_owner_graph_from_evidence)
    monkeypatch.setattr(strip_run, "_resolve_owner_graph_from_evidence", legacy)
    output_pages = _run_minimal_chapter(owner_graph_mode="enforce")
    assert output_pages
    assert all(page.owner_page_evidence_ref is not None for page in output_pages)
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


def test_style_render_replay_uses_hash_verified_owner_and_binding_artifacts(monkeypatch):
    off = _publish_verified_off_run()
    forbidden_upstream = {
        name: MagicMock(side_effect=AssertionError(f"{name} must not rerun during style replay"))
        for name in (
            "component_discovery",
            "coverage_source_ocr",
            "association_recovery",
            "container_recovery",
            "owner_graph_build",
            "translation",
            "inpaint",
        )
    }
    final_observer_ocr = MagicMock(return_value=_fresh_final_ocr_result())
    result = _run_page(
        style_copy_mode="render",
        replay_owner_artifacts=off.publication_root,
        services=_services_with_overrides(**forbidden_upstream),
        final_observer_ocr=final_observer_ocr,
    )
    assert result.replay_source_page_evidence_sha256 == off.page_execution_evidence.sha256
    assert result.owner_graph_sha256 == _off_owner_graph_sha256()
    assert result.translation_bindings_sha256 == _off_bindings_sha256()
    assert result.cleanup_base_sha256 == _off_cleanup_base_sha256()
    assert result.run_id == _off_content_run_id()
    assert result.execution_id != _off_execution_id()
    assert result.replay_of_execution_id == _off_execution_id()
    cleanup_ref = result.cleanup_base.artifact_ref
    assert cleanup_ref.execution_id == result.execution_id
    assert cleanup_ref.origin_execution_id == result.replay_of_execution_id
    assert cleanup_ref.source_artifact_ref_sha256 == _off_cleanup_artifact_ref_sha256()
    assert cleanup_ref.load_verified(_generation_root_for(result)).tobytes() == off.cleanup_ref.load_verified(
        off.publication_root
    ).tobytes()
    for spy in forbidden_upstream.values():
        spy.assert_not_called()
    final_observer_ocr.assert_called()
    assert result.terminal_proof.fresh_ocr_root_input_pixel_sha256 == result.final_page.page_output_pixel_sha256
    assert result.terminal_proof.fresh_ocr_attempt_chain_sha256


def test_owner_cli_defaults_do_not_hardcode_chapter_or_override_loaded_modes(tmp_path):
    loaded_replay = tmp_path / "loaded-parent-publication"
    args = main.parse_cli_args(["--input", "source", "--output", str(tmp_path / "out")])
    assert args.chapter is None
    assert args.owner_graph_mode is None
    assert args.style_copy_mode is None
    assert args.replay_owner_artifacts is None
    runner = main.resolve_runner_config_from_cli(
        args,
        loaded_config=_loaded_config(
            chapter=7,
            owner_graph_mode="shadow",
            style_copy_mode="off",
            replay_owner_artifacts=str(loaded_replay),
        ),
    )
    assert (runner.chapter, runner.owner_graph_mode, runner.style_copy_mode) == (7, "shadow", "off")
    assert Path(runner.replay_owner_artifacts) == loaded_replay.resolve()


def _task20_runner_projection(runner_config):
    replay = runner_config["replay_owner_artifacts"]
    return {
        "source_path": Path(runner_config["source_path"]).resolve(),
        "obra": runner_config["obra"],
        "work_title_user_provided": runner_config["work_title_user_provided"],
        "chapter": runner_config["chapter"],
        "idioma_origem": runner_config["idioma_origem"],
        "idioma_destino": runner_config["idioma_destino"],
        "mode": runner_config["mode"],
        "work_dir": Path(runner_config["work_dir"]).resolve(),
        "debug": runner_config["debug"],
        "strict": runner_config["strict"],
        "export_mode": runner_config["export_mode"],
        "owner_graph_mode": runner_config["owner_graph_mode"],
        "style_copy_mode": runner_config["style_copy_mode"],
        "replay_owner_artifacts": None if replay is None else Path(replay).resolve(),
    }


def test_owner_cli_off_arguments_reach_runner_config_exactly(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "off"
    runner_config = _run_cli_until_runner_config(
        [
            "--input", str(source),
            "--work", "Mitch Items",
            "--chapter", "39",
            "--source-lang", "en",
            "--target", "pt-BR",
            "--mode", "real",
            "--output", str(output),
            "--debug", "--strict",
            "--export-mode", "strict",
            "--owner-graph-mode", "enforce",
            "--style-copy-mode", "off",
        ],
        stop_after_runner_config=True,
    )
    assert _task20_runner_projection(runner_config) == {
        "source_path": source.resolve(),
        "obra": "Mitch Items",
        "work_title_user_provided": True,
        "chapter": 39,
        "idioma_origem": "en",
        "idioma_destino": "pt-BR",
        "mode": "real",
        "work_dir": output.resolve(),
        "debug": True,
        "strict": True,
        "export_mode": "strict",
        "owner_graph_mode": "enforce",
        "style_copy_mode": "off",
        "replay_owner_artifacts": None,
    }


def test_owner_cli_render_replay_arguments_reach_runner_config_exactly(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "render"
    replay_root = tmp_path / "off-publication"
    runner_config = _run_cli_until_runner_config(
        [
            "--input", str(source),
            "--work", "Mitch Items",
            "--chapter", "39",
            "--source-lang", "en",
            "--target", "pt-BR",
            "--mode", "real",
            "--output", str(output),
            "--debug", "--strict",
            "--export-mode", "strict",
            "--owner-graph-mode", "enforce",
            "--style-copy-mode", "render",
            "--replay-owner-artifacts", str(replay_root),
        ],
        stop_after_runner_config=True,
    )
    assert _task20_runner_projection(runner_config) == {
        "source_path": source.resolve(),
        "obra": "Mitch Items",
        "work_title_user_provided": True,
        "chapter": 39,
        "idioma_origem": "en",
        "idioma_destino": "pt-BR",
        "mode": "real",
        "work_dir": output.resolve(),
        "debug": True,
        "strict": True,
        "export_mode": "strict",
        "owner_graph_mode": "enforce",
        "style_copy_mode": "render",
        "replay_owner_artifacts": replay_root.resolve(),
    }


@pytest.mark.parametrize(
    "tamper",
    [
        "source_tree_hash",
        "schema",
        "run_id",
        "execution_id",
        "origin_execution_id",
        "page_id",
        "page_source_hash",
        "page_evidence_hash",
        "coverage_hash",
        "coverage_ledger_history",
        "coverage_ledger_parent_hash",
        "coverage_ocr_request_hash",
        "coverage_ocr_invocation_hash",
        "coverage_ocr_attempt_hash",
        "coverage_component_inventory",
        "coverage_recovery_request",
        "coverage_recovery_decision",
        "coverage_pending_request_ids",
        "translation_attempt_hash",
        "translation_attempt_chain",
        "source_payload_hash",
        "target_payload_hash",
        "missing_owner",
        "extra_owner",
        "missing_binding",
        "extra_binding",
        "duplicate_binding",
        "reordered_binding",
        "swapped_binding",
        "repair_request_chain",
        "repair_attempt_chain",
        "repair_budget_policy",
        "owner_target_materializations",
        "page_commits",
        "page_composition",
        "cleanup_base_hash",
        "stage_ref_original",
        "stage_ref_cleanup_base",
        "stage_ref_inpaint",
        "stage_ref_typeset",
        "stage_ref_page_composition",
        "stage_ref_final",
        "final_qa_ocr_requests",
        "final_qa_ocr_invocations",
        "language_residual_issues",
        "qa_probes",
        "final_replacement_verdicts",
        "replacement_verification_policy",
        "final_page",
        "terminal_proof",
        "source_artifact_ref_sha256",
        "missing_page",
        "extra_page",
        "duplicate_page",
        "reordered_page",
    ],
)
def test_style_render_replay_rejects_tampered_content_lineage_before_render(tamper):
    publication = _tamper_replay_publication_and_recompute_outer_hashes(_publish_verified_off_run(), tamper)
    with _forbid_style_renderer_and_publication_staging() as effects:
        with pytest.raises(ContentReplayIntegrityError):
            _run_page(style_copy_mode="render", replay_owner_artifacts=publication.publication_root)
    effects.renderer.assert_not_called()
    effects.staging.assert_no_writes()
```

**Step 2: Confirmar RED**

```powershell
Push-Location pipeline
try {
  $redOutput = & .\venv\Scripts\python.exe -m pytest `
    tests/test_main_emit.py::test_owner_cli_defaults_do_not_hardcode_chapter_or_override_loaded_modes `
    tests/test_main_emit.py::test_owner_cli_off_arguments_reach_runner_config_exactly `
    tests/test_main_emit.py::test_owner_cli_render_replay_arguments_reach_runner_config_exactly `
    -q 2>&1
  $redExit = $LASTEXITCODE
  $redText = $redOutput | Out-String
  $redOutput
  if ($redExit -ne 1) { throw "Task 17 expected pytest exit 1 for a behavioral RED, got $redExit" }
  if ($redText -match '(?im)^ERROR(?:\s|:)|ERROR collecting|errors during collection|^E\s+fixture .* not found|UsageError|ModuleNotFoundError|ImportError') {
    throw 'Task 17 RED is structural rather than behavioral; create/fix the CLI API skeleton and rerun the same nodeids'
  }
  if ($redText -notmatch '(?m)^FAILED\s+\S+::') { throw 'Task 17 RED did not report a targeted FAILED nodeid' }
  .\venv\Scripts\python.exe -m pytest `
    tests/test_main_emit.py `
    tests/test_strip_owner_control_plane.py `
    tests/test_page_owner_pipeline.py `
    tests/test_strip_owner_composition_integration.py `
    tests/test_owner_style_profile.py `
    tests/test_style_fidelity_qa.py `
    tests/test_owner_content_replay.py `
    -q
} finally { Pop-Location }
```

Expected: `enforce` ainda atravessa caminhos paralelos e reparos tardios de payload.

**Step 3: Roteamento final**

- `owner_graph_mode=enforce` chama exclusivamente `run_page_owner_pipeline()`, que usa `owner_builder.py` + `consensus_v2.py`; não chama o resolver local preexistente de `reconcile.py`.
- `run.py` mantém scheduling, leitura e persistência, sem redecidir semântica.
- `process_bands.py` mantém adapters de baixa camada e compatibilidade de teste.
- `legacy`/`shadow` não alteram pixels, owners, translations ou artefatos canônicos de uma run `enforce`.
- remover do caminho verificado as hidratações/normalizações tardias em torno de `main.py:9602`, `9657` e `9674`.
- suportar explicitamente `style_copy_mode=off|shadow|render|enforce` nos validators/configs reais: `off` não chama `build_owner_visual_profiles()` nem `audit_style_fidelity()` e usa renderer base; `shadow` só coleta diagnóstico sem mutar; `render` aplica estilo com fallback base e mantém o gate de conteúdo autoritativo; `enforce` acrescenta gate de fidelidade de estilo e é auditoria opcional separada. Cobrir o no-op audit/gate de `off` em `test_style_fidelity_qa.py`.
- ampliar `parse_cli_args(argv)` e o runner CLI com `--chapter`, `--owner-graph-mode`, `--style-copy-mode` e `--replay-owner-artifacts`; o default de cada override é `None`, de modo que a ausência do flag preserva o valor do config carregado e nunca mantém capítulo hardcoded. `resolve_runner_config_from_cli()` aplica overrides tipados (`chapter` inteiro, modes em enums, replay root resolvida), persiste os valores efetivos em `runner_config.json` e é o mesmo caminho usado por `main(argv)`. Os REDs exercitam defaults e os argv **exatos** das runs `off` e `render` da Task 20, incluindo propagação do replay path.
- em replay, receber a raiz de uma publicação `off`, adquirir reader lock e reabri-la exclusivamente pelo `publication_receipt.json`; preservar `run_id` como linhagem de conteúdo, gerar novo `execution_id` e registrar `replay_of_execution_id`. Para cada página, chamar `PageExecutionEvidenceSnapshot.read_verified(off_publication_root)` e validar o snapshot **inteiro** antes de selecionar qualquer campo: source tree, schemas/IDs, coverage+OCR requests/invocations/attempt chains, owner graph, translation attempts/bindings/source+target payloads, repair requests/history/policies, materializations/commits/composition, stage refs e cardinalidade/ordem. Persistir `replay_source_page_evidence_sha256` no `PageExecutionResult`, no evidence snapshot e no receipt novo. Só depois copiar cada cleanup/ref necessária para a raiz privada do execution novo, verificando bytes/pixels e ligando a nova ref a `source_artifact_ref_sha256`; records herdados mantêm `origin_execution_id` pai e nunca são reetiquetados. Pular component discovery, OCR de coverage/source, association/container recovery, owner build, tradução e inpaint; rerodar apenas glyph render/style, composição e o OCR observer terminal fresco sobre os novos pixels. Isso impede uma publicação Frankenstein e isola o style copier sem reutilizar prova terminal ou ref física de outra execução. Missing/extra/reordered/swapped record ou qualquer mismatch interno deve falhar antes de renderer/staging.

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
    tests/test_owner_content_replay.py `
    -q
} finally { Pop-Location }
```

Expected: PASS; `enforce` possui exatamente um coordenador e nenhuma pós-etapa modifica pixels verificados.

**Step 6: Commit**

```powershell
$guard = '.codex-tmp\universal-source-replacement-plan-guard\TaskOwnedGit.ps1'
& $guard -Action Stage -Task 17
git diff --cached --check
git commit -m "refactor: make page owner pipeline authoritative"
& $guard -Action VerifyAfterCommit -Task 17
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
- Create: `pipeline/tools/validate_visual_review_evidence.py`
- Create: `pipeline/tests/test_validate_visual_review_evidence.py`
- Modify: `pipeline/tools/export_visual_review_sheet.py`
- Modify: `pipeline/tests/test_export_visual_review_sheet.py`
- Modify: `pipeline/tools/build_style_owner_target_manifest.py`
- Create: `pipeline/tests/test_build_style_owner_target_manifest.py`
- Modify: `pipeline/tools/validate_owner_visual_matrix.py`
- Modify: `pipeline/tests/fixtures/owner_visual_matrix/inputs.json`
- Modify: `pipeline/tests/fixtures/owner_visual_matrix/functional_matrix.json`
- Modify: `pipeline/tests/fixtures/owner_visual_matrix/configs/colored_cards.json`
- Modify: `pipeline/tests/fixtures/owner_visual_matrix/configs/cross_tile_owners.json`
- Modify: `pipeline/tests/fixtures/owner_visual_matrix/configs/dark_panels.json`
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


def test_external_auditor_rejects_english_seen_on_final_even_when_stored_proof_claims_clean(tmp_path):
    run = _fully_valid_stored_run_for_candidate_a(tmp_path, stored_english_residual_count=0)
    external = _external_invocation_for_redecoded_final(
        run,
        invocation_id="audit-sees-english",
        lines=("THE ARENA WILL BEGIN",),
    )
    audit = audit_chapter(run, external_ocr_runner=_child_runner_returning(external))
    assert audit.external_audit_gate_status == "NO_GO"
    assert audit.external_english_dialogue_residual_count == 1
    assert audit.external_page_audits[0].source_only_residuals


def test_external_auditor_rejects_material_english_omitted_from_stored_component_owner_and_binding_graph(tmp_path):
    run = _fully_valid_stored_run_whose_graph_omits_visible_english_region(tmp_path)
    external = _external_invocation_for_redecoded_final(
        run,
        invocation_id="audit-finds-unowned-english",
        lines=("FAILURE WILL RESULT IN PENALTIES",),
        line_regions=(_omitted_translatable_container_region(),),
    )
    audit = audit_chapter(run, external_ocr_runner=_child_runner_returning(external))
    assert audit.external_audit_gate_status == "NO_GO"
    assert audit.external_english_dialogue_residual_count == 1
    assert audit.unowned_external_source_residuals == 1
    assert audit.external_page_audits[0].unowned_source_residuals


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


def test_external_auditor_emits_one_independent_record_for_every_source_page(tmp_path):
    run = _fully_valid_stored_three_page_run(tmp_path)
    runner = _capturing_child_process_runner_for_every_final()
    audit = audit_chapter(run, external_ocr_runner=runner)
    assert [item.page_id for item in audit.external_page_audits] == _source_manifest_page_ids(run)
    assert len(audit.external_page_audits) == 3
    assert runner.paths == [_persisted_final_path(run, page_id) for page_id in _source_manifest_page_ids(run)]
    assert len({item.auditor_invocation_id for item in audit.external_page_audits}) == 3
    assert len({item.auditor_execution_id for item in audit.external_page_audits}) == 3
    assert len({item.auditor_process_nonce for item in audit.external_page_audits}) == 3
    assert all(item.auditor_invocation_id != item.terminal_invocation_id for item in audit.external_page_audits)
    assert all(item.auditor_execution_id != run.execution_id for item in audit.external_page_audits)
    assert all(item.auditor_root_hash_matches_final for item in audit.external_page_audits)


@pytest.mark.parametrize(
    "tamper",
    [
        "missing_page",
        "duplicate_page",
        "wrong_root",
        "duplicate_invocation",
        "duplicate_auditor_execution_id",
        "auditor_execution_equals_run_execution",
        "duplicate_process_nonce",
    ],
)
def test_external_page_audit_bijection_rejects_any_single_page_gap_or_reuse(tmp_path, tamper):
    run = _fully_valid_stored_three_page_run(tmp_path)
    audit = audit_chapter(run, external_ocr_runner=_tampered_three_page_external_runner(tamper))
    assert audit.external_audit_gate_status == "NO_GO"


def test_runner_claiming_new_process_without_child_emitted_identity_is_rejected(tmp_path):
    run = _fully_valid_stored_run_for_candidate_a(tmp_path)
    runner = _runner_ignoring_force_new_process_and_reusing_parent_identity(run)
    audit = audit_chapter(run, external_ocr_runner=runner)
    assert audit.external_audit_gate_status == "NO_GO"
    assert "auditor_process_not_independent" in audit.identity_mismatches


@pytest.mark.parametrize(
    "tamper",
    [
        "source_tree_sha256",
        "schema_version",
        "run_id",
        "execution_id",
        "page_id",
        "page_source_sha256",
        "page_evidence_sha256",
        "coverage_hash",
        "coverage_ledger_history",
        "coverage_ledger_parent_hash",
        "ocr_request_hash",
        "ocr_invocation_hash",
        "ocr_attempt_chain",
        "coverage_component_inventory",
        "coverage_recovery_requests",
        "coverage_recovery_decisions",
        "coverage_pending_request_ids",
        "translation_attempt_hash",
        "translation_binding_hash",
        "repair_requests",
        "repair_chain_hash",
        "repair_budget_policy",
        "owner_target_materializations",
        "page_commits",
        "page_composition",
        "stage_ref_original",
        "stage_ref_cleanup_base",
        "stage_ref_inpaint",
        "stage_ref_typeset",
        "stage_ref_page_composition",
        "stage_ref_final",
        "final_qa_ocr_requests",
        "final_qa_ocr_invocations",
        "language_residual_issues",
        "qa_probes",
        "final_replacement_verdicts",
        "replacement_verification_policy",
        "final_page",
        "terminal_proof",
        "artifact_relative_path",
        "source_artifact_ref_sha256",
        "missing_owner",
        "extra_owner",
        "missing_binding",
        "extra_binding",
        "missing_page",
        "extra_page",
        "duplicate_page",
        "reordered_page",
    ],
)
def test_external_auditor_rejects_persisted_identity_or_cardinality_tamper_before_ocr_or_staging(tmp_path, tamper):
    run = _tamper_persisted_verified_tree_and_recompute_all_outer_hashes(
        _fully_valid_stored_three_page_run(tmp_path), tamper
    )
    runner = MagicMock()
    with _forbid_public_or_review_staging() as staging:
        audit = audit_chapter(run, external_ocr_runner=runner)
    assert audit.external_audit_gate_status == "NO_GO"
    runner.assert_not_called()
    staging.assert_no_writes()


@pytest.mark.parametrize("external_failure", ["runner_missing", "provider_unavailable", "no_physical_full_page"])
def test_external_auditor_never_falls_back_to_stored_terminal_proof(external_failure, tmp_path):
    run = _fully_valid_stored_run_for_candidate_a(tmp_path)
    audit = audit_chapter(run, external_ocr_runner=_failing_external_runner(external_failure))
    assert audit.status == "NO_GO"
    assert audit.external_auditor_fresh_ocr_complete is False
    assert audit.auditor_invocation_id is None
    assert "external_auditor_unavailable_or_incomplete" in audit.identity_mismatches


def test_chapter_auditor_emits_real_page_bound_source_output_pairs():
    run = _verified_three_page_run()
    audit = audit_chapter(run, external_ocr_runner=_valid_external_runner_for(run), write_review_pairs=True)
    assert len(audit.review_pairs) == 3
    assert len({(pair.page_id, pair.source_pixel_sha256, pair.output_path, pair.lineage_sha256) for pair in audit.review_pairs}) == 3
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


def test_external_audit_report_serializes_independence_root_and_gate_fields():
    report = audit_chapter(_verified_run(), external_ocr_runner=_valid_external_runner())
    payload = report.to_dict()
    assert payload["auditor_invocation_is_independent"] is True
    assert payload["auditor_root_hash_matches_final"] is True
    assert payload["external_audit_gate_status"] == "PASS"


def test_auditor_rejects_missing_canonical_stage_and_never_rebuilds_it_from_debug():
    run = _verified_run_missing_stage("typeset")
    audit = audit_chapter(run, external_ocr_runner=_valid_external_runner())
    assert audit.external_audit_gate_status == "NO_GO"
    assert "canonical_visual_stage_missing:typeset" in audit.identity_mismatches


def test_functional_matrix_builder_preserves_explicit_style_copy_off():
    effective = build_effective_style_config(
        _functional_matrix_config(style_copy_mode="off"),
        required_categories=[],
    )
    assert effective["owner_graph_mode"] == "enforce"
    assert effective["style_copy_mode"] == "off"


def test_visual_matrix_requires_independent_external_audit_for_every_entry(tmp_path):
    report = validate_matrix(
        _four_entry_manifest_with_mitch_regression(),
        audit_dir=tmp_path / "audits",
        require_external_audit=True,
        external_audit_records=_four_valid_persisted_external_audit_records(tmp_path),
    )
    assert report.status == "PASS"
    assert {item.entry_id for item in report.external_audits} == _four_entry_ids()
    assert all(item.auditor_invocation_is_independent for item in report.external_audits)
    assert all(item.external_audit_gate_status == "PASS" for item in report.external_audits)


def test_validation_context_uses_new_root_and_is_atomically_reloadable(tmp_path):
    source = _write_validation_source(tmp_path / "source")
    base = tmp_path / "validation"
    context_path = tmp_path / "current.json"
    first = init_validation_context(base, source, context_path)
    second = init_validation_context(base, source, context_path)
    assert first.validation_root != second.validation_root
    assert second.source_path == source.resolve()
    assert second.source_tree_sha256 == canonical_source_tree_sha256(_ordered_images(source), source)
    assert load_validation_context(context_path) == second
    assert not (tmp_path / "current.json.tmp").exists()


def test_visual_review_completion_requires_exact_off_render_matrix_sets_and_bound_sentinels(tmp_path):
    evidence = _review_evidence(
        off_pages=range(1, 43),
        render_pages=range(1, 43),
        matrix_pages=_all_matrix_entry_pages(),
        sentinel_stages=_six_stages_for_ten_sentinels_in_both_runs(),
    )
    result = validate_visual_review_evidence(evidence, expected=_expected_review_manifest())
    assert result.status == "PASS"


@pytest.mark.parametrize(
    "tamper",
    ["missing", "duplicate", "stale_path", "stale_hash", "wrong_execution", "missing_stage", "autofilled_verdict"],
)
def test_visual_review_completion_rejects_incomplete_or_unbound_evidence(tmp_path, tamper):
    evidence = _tamper_review_evidence(_complete_review_evidence(tmp_path), tamper)
    with pytest.raises(VisualReviewEvidenceError):
        validate_visual_review_evidence(evidence, expected=_expected_review_manifest())


def test_init_owner_validation_cli_forwards_exact_paths_and_returns_nonzero_on_failure(tmp_path, monkeypatch):
    base = tmp_path / "validation"
    source = tmp_path / "source"
    context = tmp_path / "current.json"
    initialize = MagicMock(return_value=_validation_context(base, source, context))
    monkeypatch.setattr(init_owner_validation_run, "init_validation_context", initialize)
    assert init_owner_validation_run.main(
        ["--base", str(base), "--source", str(source), "--context", str(context)]
    ) == 0
    initialize.assert_called_once_with(base.resolve(), source.resolve(), context.resolve())
    initialize.side_effect = ValidationContextError("cannot initialize")
    assert init_owner_validation_run.main(
        ["--base", str(base), "--source", str(source), "--context", str(context)]
    ) != 0


@pytest.mark.parametrize(("gate_status", "expected_exit"), [("PASS", 0), ("NO_GO", 1)])
def test_audit_owner_chapter_output_cli_forwards_task20_arguments_and_exit_code(
    tmp_path, monkeypatch, gate_status, expected_exit
):
    audit = MagicMock(return_value=_audit_report(external_audit_gate_status=gate_status))
    monkeypatch.setattr(audit_owner_chapter_output, "audit_chapter_from_paths", audit)
    source, run = tmp_path / "source", tmp_path / "run"
    report, review = tmp_path / "audit.json", tmp_path / "review"
    exit_code = audit_owner_chapter_output.main(
        [
            "--source", str(source),
            "--run", str(run),
            "--report", str(report),
            "--review-dir", str(review),
            "--source-lang", "en",
            "--target-lang", "pt-BR",
            "--require-final-verified",
        ]
    )
    assert exit_code == expected_exit
    audit.assert_called_once_with(
        source=source.resolve(),
        run=run.resolve(),
        report=report.resolve(),
        review_dir=review.resolve(),
        source_lang="en",
        target_lang="pt-BR",
        compare_content_run=None,
        require_final_verified=True,
    )


def test_audit_owner_chapter_output_cli_returns_nonzero_when_auditor_raises(tmp_path, monkeypatch):
    audit = MagicMock(side_effect=RuntimeError("invalid publication"))
    monkeypatch.setattr(audit_owner_chapter_output, "audit_chapter_from_paths", audit)
    assert audit_owner_chapter_output.main(
        [
            "--source", str(tmp_path / "source"),
            "--run", str(tmp_path / "run"),
            "--report", str(tmp_path / "audit.json"),
            "--review-dir", str(tmp_path / "review"),
            "--source-lang", "en",
            "--target-lang", "pt-BR",
            "--require-final-verified",
        ]
    ) != 0


def test_audit_owner_chapter_output_cli_forwards_render_compare_content_run(tmp_path, monkeypatch):
    audit = MagicMock(return_value=_audit_report(external_audit_gate_status="PASS"))
    monkeypatch.setattr(audit_owner_chapter_output, "audit_chapter_from_paths", audit)
    source = tmp_path / "source"
    render_run = tmp_path / "render-run"
    off_run = tmp_path / "off-run"
    report = tmp_path / "render-audit.json"
    review = tmp_path / "render-review"
    assert audit_owner_chapter_output.main(
        [
            "--source", str(source),
            "--run", str(render_run),
            "--report", str(report),
            "--review-dir", str(review),
            "--source-lang", "en",
            "--target-lang", "pt-BR",
            "--compare-content-run", str(off_run),
            "--require-final-verified",
        ]
    ) == 0
    audit.assert_called_once_with(
        source=source.resolve(),
        run=render_run.resolve(),
        report=report.resolve(),
        review_dir=review.resolve(),
        source_lang="en",
        target_lang="pt-BR",
        compare_content_run=off_run.resolve(),
        require_final_verified=True,
    )


@pytest.mark.parametrize(("matrix_status", "expected_exit"), [("PASS", 0), ("NO_GO", 1)])
def test_validate_owner_visual_matrix_cli_forwards_external_audit_flags_and_exit_code(
    tmp_path, monkeypatch, matrix_status, expected_exit
):
    validate = MagicMock(return_value=_matrix_report(status=matrix_status))
    monkeypatch.setattr(validate_owner_visual_matrix, "validate_matrix_cli_request", validate)
    manifest = tmp_path / "functional_matrix.json"
    output = tmp_path / "matrix-out"
    report = tmp_path / "matrix.md"
    inspection = tmp_path / "inspection.json"
    audit_dir = tmp_path / "audits"
    exit_code = validate_owner_visual_matrix.main(
        [
            "--manifest", str(manifest),
            "--output-root", str(output),
            "--report", str(report),
            "--inspection-template", str(inspection),
            "--audit-dir", str(audit_dir),
            "--require-external-audit",
            "--run-id", "universal-source-replacement-v1",
        ]
    )
    assert exit_code == expected_exit
    validate.assert_called_once_with(
        manifest=manifest.resolve(),
        output_root=output.resolve(),
        report=report.resolve(),
        inspection_template=inspection.resolve(),
        audit_dir=audit_dir.resolve(),
        require_external_audit=True,
        run_id="universal-source-replacement-v1",
    )


def test_validate_owner_visual_matrix_cli_returns_nonzero_when_validation_raises(tmp_path, monkeypatch):
    validate = MagicMock(side_effect=RuntimeError("external audit missing"))
    monkeypatch.setattr(validate_owner_visual_matrix, "validate_matrix_cli_request", validate)
    assert validate_owner_visual_matrix.main(
        [
            "--manifest", str(tmp_path / "functional_matrix.json"),
            "--output-root", str(tmp_path / "matrix-out"),
            "--report", str(tmp_path / "matrix.md"),
            "--inspection-template", str(tmp_path / "inspection.json"),
            "--audit-dir", str(tmp_path / "audits"),
            "--require-external-audit",
            "--run-id", "universal-source-replacement-v1",
        ]
    ) != 0


@pytest.mark.parametrize(
    ("script", "argv"),
    [
        (
            "tools/init_owner_validation_run.py",
            lambda root: [
                "--base", str(root / "validation"),
                "--source", str(root / "missing-source"),
                "--context", str(root / "current.json"),
            ],
        ),
        (
            "tools/audit_owner_chapter_output.py",
            lambda root: [
                "--source", str(root / "missing-source"),
                "--run", str(root / "missing-run"),
                "--report", str(root / "audit.json"),
                "--review-dir", str(root / "review"),
                "--source-lang", "en",
                "--target-lang", "pt-BR",
                "--require-final-verified",
            ],
        ),
        (
            "tools/validate_owner_visual_matrix.py",
            lambda root: [
                "--manifest", str(root / "missing-matrix.json"),
                "--output-root", str(root / "matrix-out"),
                "--report", str(root / "matrix.md"),
                "--inspection-template", str(root / "inspection.json"),
                "--audit-dir", str(root / "audits"),
                "--require-external-audit",
                "--run-id", "cli-exit-contract",
            ],
        ),
    ],
)
def test_cli_entrypoints_propagate_main_failure_to_real_process_exit(tmp_path, script, argv):
    completed = subprocess.run(
        [sys.executable, str(_pipeline_root() / script), *argv(tmp_path)],
        cwd=_pipeline_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "Traceback" not in completed.stderr


@pytest.mark.parametrize(
    "script",
    [
        "tools/init_owner_validation_run.py",
        "tools/audit_owner_chapter_output.py",
        "tools/validate_owner_visual_matrix.py",
    ],
)
def test_cli_entrypoint_raises_system_exit_from_main(script):
    tree = ast.parse((_pipeline_root() / script).read_text(encoding="utf-8"))
    assert _main_guard_calls_raise_system_exit_main(tree) is True


def test_external_chapter_auditor_launches_one_real_child_process_per_page(tmp_path):
    publication = _write_two_page_verified_publication(tmp_path)
    child_command = _write_external_ocr_child_command(tmp_path)
    report = audit_chapter_from_paths(
        source=publication.source,
        run=publication.run,
        report=tmp_path / "audit.json",
        review_dir=tmp_path / "review",
        source_lang="en",
        target_lang="pt-BR",
        compare_content_run=None,
        require_final_verified=True,
        external_ocr_command=child_command,
    )
    assert len(report.external_page_audits) == 2
    assert len({item.auditor_pid for item in report.external_page_audits}) == 2
    assert len({item.auditor_execution_id for item in report.external_page_audits}) == 2
    assert len({item.auditor_process_nonce for item in report.external_page_audits}) == 2


def test_visual_matrix_launches_auditor_cli_in_distinct_real_process_per_entry(tmp_path):
    manifest = _two_entry_process_matrix(tmp_path)
    invocation_log = tmp_path / "auditor-processes.jsonl"
    auditor_command = _write_valid_fake_auditor_cli(tmp_path, invocation_log=invocation_log)
    report = validate_matrix(
        manifest,
        audit_dir=tmp_path / "audits",
        require_external_audit=True,
        external_auditor_command=auditor_command,
    )
    invocations = _read_jsonl(invocation_log)
    assert report.status == "PASS"
    assert {item["entry_id"] for item in invocations} == _two_entry_ids()
    assert len({item["pid"] for item in invocations}) == len(invocations) == 2
```

**Step 2: Confirmar RED**

```powershell
Push-Location pipeline
try {
  $redOutput = & .\venv\Scripts\python.exe -m pytest `
    tests/test_init_owner_validation_run.py::test_init_owner_validation_cli_forwards_exact_paths_and_returns_nonzero_on_failure `
    tests/test_audit_owner_chapter_output.py::test_audit_owner_chapter_output_cli_forwards_task20_arguments_and_exit_code `
    tests/test_audit_owner_chapter_output.py::test_audit_owner_chapter_output_cli_returns_nonzero_when_auditor_raises `
    tests/test_audit_owner_chapter_output.py::test_audit_owner_chapter_output_cli_forwards_render_compare_content_run `
    tests/test_init_owner_validation_run.py::test_cli_entrypoints_propagate_main_failure_to_real_process_exit `
    tests/test_init_owner_validation_run.py::test_cli_entrypoint_raises_system_exit_from_main `
    tests/test_audit_owner_chapter_output.py::test_external_chapter_auditor_launches_one_real_child_process_per_page `
    tests/test_owner_visual_matrix_tool.py::test_validate_owner_visual_matrix_cli_forwards_external_audit_flags_and_exit_code `
    tests/test_owner_visual_matrix_tool.py::test_validate_owner_visual_matrix_cli_returns_nonzero_when_validation_raises `
    tests/test_owner_visual_matrix_tool.py::test_visual_matrix_launches_auditor_cli_in_distinct_real_process_per_entry `
    -q 2>&1
  $redExit = $LASTEXITCODE
  $redText = $redOutput | Out-String
  $redOutput
  if ($redExit -ne 1) { throw "Task 18 expected pytest exit 1 for a behavioral RED, got $redExit" }
  if ($redText -match '(?im)^ERROR(?:\s|:)|ERROR collecting|errors during collection|^E\s+fixture .* not found|UsageError|ModuleNotFoundError|ImportError') {
    throw 'Task 18 RED is structural rather than behavioral; create/fix each main(argv) skeleton and rerun the same nodeids'
  }
  if ($redText -notmatch '(?m)^FAILED\s+\S+::') { throw 'Task 18 RED did not report a targeted FAILED nodeid' }
  .\venv\Scripts\python.exe -m pytest `
    tests/test_owner_lifecycle_e2e.py `
    tests/test_audit_owner_chapter_output.py `
    tests/test_init_owner_validation_run.py `
    tests/test_validate_visual_review_evidence.py `
    tests/test_export_visual_review_sheet.py `
    tests/test_build_style_owner_target_manifest.py `
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

Defina no auditor `PageAcceptanceMetrics` frozen com exatamente os sete contadores globais deste plano, e `compute_page_acceptance_metrics(result: PageExecutionResult) -> PageAcceptanceMetrics`. Essa é a única origem de `metrics` dos testes/relatórios; `PageExecutionResult` não ganha campo derivado nem segunda autoridade. O builder lê ledger, graph, bindings, commits, materializations, verdicts e proof imutáveis e é reutilizado na agregação do capítulo. Para `english_dialogue_residual_count`, resolve exatamente `result.terminal_proof.final_qa_probe_id` e conta somente as issues citadas por esse probe terminal; issues históricas reparadas permanecem auditáveis, sem contaminar a métrica atual.

Defina também `ExternalPageAudit` frozen com `page_id`, `run_id`, `execution_id`, `page_source_sha256`, final path/file/pixel hash, `terminal_invocation_id`, `auditor_invocation_id`, `auditor_execution_id`, `auditor_process_nonce`, attempt chain/inference/cache, root hash, observações OCR externas frozen, `source_only_residuals`, `unowned_source_residuals`, `english_dialogue_residual_count`, `language_verdict`, status e `audit_sha256`. `auditor_execution_id` e um nonce criptograficamente aleatório são gerados e emitidos pelo processo filho junto do resultado; PID/start time podem ser diagnósticos, mas não substituem essas identidades. Classifique a invocation externa contra os source/target payloads e roles hash-bound do owner graph/bindings persistidos; um token/frase source material encontrado nos pixels finais conta mesmo quando `TerminalPixelProof` armazenado diz limpo. O chapter report carrega `external_page_audits: tuple[ExternalPageAudit, ...]` em bijeção ordenada com as páginas do `ChapterSourceManifest`; campos agregados, inclusive `external_english_dialogue_residual_count` e `unowned_external_source_residuals`, são calculados somente dessa tuple. O gate exige `auditor_invocation_id`, `auditor_execution_id` e `auditor_process_nonce` únicos por página, além de `auditor_execution_id != run.execution_id`; qualquer página ausente/duplicada, inglês externo, source externo sem owner ou identidade de child reutilizada força `NO_GO`.

Implemente `validate_visual_review_evidence.py` como validador read-only do evidence JSON manual. Ele recebe os manifests/audits finais e exige conjuntos exatos: 42 páginas `off`, 42 páginas `render` e todas as páginas de cada uma das quatro entries da matriz; rejeita item ausente/extra/duplicado, path inexistente, file/pixel hash stale, `execution_id`/lineage divergente e verdict sem `reviewer`/`reviewed_at`. Para cada uma das dez sentinelas em `off` e `render`, exige exatamente os seis stages hash-bound. Nenhum template/autofill pode satisfazer `visual_verdict`; somente `GO|NO_GO` registrado após inspeção conta.

CLI:

```text
python pipeline/tools/init_owner_validation_run.py \
  --base .codex-tmp/universal-source-replacement-20260804 \
  --source N:/TraduzAI/temporario1/mch39 \
  --context .codex-tmp/universal-source-replacement-current.json
```

O auditor usa o contexto apenas quando paths explícitos não forem fornecidos. `init_owner_validation_run.py`, `audit_owner_chapter_output.py` e `validate_owner_visual_matrix.py` expõem `main(argv: Sequence[str] | None = None) -> int`; parsing é separado do núcleo para teste. Cada módulo termina exatamente com `if __name__ == "__main__": raise SystemExit(main())`, nunca apenas `main()`. Os argv usados nas Tasks 18/20 atravessam esse mesmo entrypoint. `0` significa contexto persistido ou relatório/gate `PASS`; erro de parsing, I/O, auditor `NO_GO` ou matriz `NO_GO` retorna não zero. Os REDs chamam `main(argv)` com os flags exatos, cobrem `off` e `render --compare-content-run`, verificam forwarding/exit code e também executam os três scripts por `subprocess.run`; um erro controlado precisa terminar o processo com código não zero e sem traceback não tratado.

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

- adquire reader lock, reabre a publicação exclusivamente pelo receipt e chama `read_verified(run_root)` para cada `PageExecutionEvidenceSnapshot` antes de OCR; lê apenas esses artefatos canônicos e imagens finais, nunca objetos vivos ou debug;
- confere cardinalidade página-fonte/página-final;
- revalida hashes/ledger/bindings/provas/gate;
- executa OCR fresco em um processo filho novo **para cada final** pelo mesmo `execute_hash_bound_provider_attempt()` da Task 2, nunca fabricando records no auditor. `ExternalOCRProcessRunner` usa `subprocess.Popen`/`run` para cada página; o seam de teste injeta somente `external_ocr_command`, nunca um callable in-process, e o child fake ainda é um processo real. A implementação recebe do child `auditor_execution_id` + `auditor_process_nonce` e persiste PID/start time diagnósticos. Ausência/falha/invocation sem full-page físico, `auditor_execution_id == run.execution_id`, PID/execution/nonce reutilizado ou qualquer tentativa de fallback in-process produz NO_GO e jamais reutiliza `TerminalPixelProof` armazenado;
- reabre cada final, exige root hash idêntico aos pixels redecodificados, exige `auditor_invocation_id != terminal_proof.fresh_ocr_invocation_id` e identidades de processo distintas, recompõe crops/transforms a partir do `OCRTransformSpec` persistido pela invocation **externa** e valida a cadeia física. Publica um `ExternalPageAudit` por página com invocation/execution/nonce, input/root hashes, attempt-chain/inference/cache, observações e verdict de idioma. Classifica primeiro toda linha externa material pela linguagem e região visível, sem exigir que graph/component/owner/binding persistido já a conheça; linha source provável em container traduzível ou região material omitida vira `unowned_source_residual`, NO-GO e incrementa `unowned_external_source_residuals`. Só depois cruza linhas que possuem autoridade armazenada com source/target payloads e roles. Todos os achados derivam do retorno do child externo, nunca dos attempts/verdicts terminais persistidos. O aggregate só passa após bijeção com todo o source manifest, `external_english_dialogue_residual_count == 0` e `unowned_external_source_residuals == 0`;
- quando `--compare-content-run` estiver presente, exige igualdade de source tree, owner graph, translation bindings e cleanup base antes de auditar o render de estilo;
- agrega as métricas de aceitação;
- produz pares PNG source/final com resolução nativa ou crop lossless;
- para sentinelas e casos solicitados, exporta um pacote hash-linked de seis estágios em escala nativa: `original`, `coverage_ocr_overlay`, `inpaint`, `typeset`, `page_composition` e `persisted_final`; overlay é derivado e marcado, enquanto cada estágio de pixels resolve uma artifact ref canônica da Task 16. Alias pixel-idêntico exige `alias_of`/`alias_reason`; estágio ausente é NO-GO e nunca é reconstruído de debug;
- nunca altera a run nem preenche verdict visual.

O relatório possui `external_audit_gate_status: PASS|NO_GO`, separado do export gate runtime. Ele só é `PASS` quando `external_page_audits` é uma bijeção exata com o source manifest, todas as páginas têm invocation + child execution + process nonce externos independentes/completos e as somas de inglês material e inglês externo sem owner são zero; prova terminal armazenada, subprocesso ausente, nonce/invocation/execution repetida, cache hit ou source residual externo força `NO_GO`.

Amplie `validate_owner_visual_matrix.py` com `--audit-dir` e `--require-external-audit`. Para cada entry concluída, execute o auditor em processo novo sobre source+run reais, grave `<audit-dir>/<entry_id>.json`, valide schema/hash, as sete métricas, cardinalidade e os três campos externos obrigatórios. Um relatório ausente, stale ou NO-GO torna a entry e a matriz NO-GO antes de criar template de inspeção.

**Step 5: Adicionar Mitch Items cap. 39 à matriz externa**

Em `inputs.json`, mapear `mitch_items_ch39` para `TRADUZAI_MATRIX_MITCH39_SOURCE`. A entrada do manifest usa categorias genéricas e páginas de regressão 10, 11, 19, 21, 27, 28, 30, 34, 36 e 39. Páginas 30/34/36 recebem categoria de fixture `mixed_source_overlay`; isso é expectativa de teste, não regra de produção.

Config da matriz:

```json
{
  "schema_version": 1,
  "input_key": "mitch_items_ch39",
  "idioma_origem": "en",
  "idioma_destino": "pt-BR",
  "capitulo": 39,
  "owner_graph_mode": "enforce",
  "style_copy_mode": "off",
  "export_mode": "strict",
  "debug": true
}
```

Recalcular e registrar os SHA-256 do config e da entrada pelo mecanismo explícito do validador; o teste deve falhar se o arquivo mudar sem atualização do hash. Atualize também os três configs existentes para declarar `owner_graph_mode="enforce"` e `style_copy_mode="off"`. Corrija `build_style_owner_target_manifest.py` para preservar um modo explicitamente solicitado; ele não pode forçar `enforce` sobre a matriz funcional.

**Step 6: Confirmar GREEN**

```powershell
Push-Location pipeline
try {
  .\venv\Scripts\python.exe -m pytest `
    tests/test_owner_lifecycle_e2e.py `
    tests/test_audit_owner_chapter_output.py `
    tests/test_init_owner_validation_run.py `
    tests/test_validate_visual_review_evidence.py `
    tests/test_export_visual_review_sheet.py `
    tests/test_build_style_owner_target_manifest.py `
    tests/test_owner_visual_matrix_tool.py `
    -q
} finally { Pop-Location }
```

Expected: PASS; corpus cobre todas as classes sem depender de obra/página e auditor produz evidência real não preenchida.

**Step 7: Commit**

```powershell
$guard = '.codex-tmp\universal-source-replacement-plan-guard\TaskOwnedGit.ps1'
& $guard -Action Stage -Task 18
git diff --cached --check
git commit -m "test: add universal source replacement validation corpus"
& $guard -Action VerifyAfterCommit -Task 18
```

### Task 19: Executar suítes focadas e regressão completa

**Files:**

- Re-run only: files owned by Tasks 2–18
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
4. voltar à Task proprietária, acrescentar o RED ao arquivo de teste/whitelist daquela Task e criar ali o commit de correção;
5. retornar à Task 19 e rerodar teste isolado, suíte de área e suíte completa;
6. não alterar expectativa válida apenas para obter verde nem criar commit de integração genérico.

**Step 6: Confirmar fronteira de integração**

```powershell
git diff --cached --quiet
if ($LASTEXITCODE -ne 0) { throw "Task 19 must not own staged changes; return them to the owning Task" }
git status --short --branch
```

A Task 19 nunca cria commit. Cada correção fica no commit/whitelist da Task que possui o comportamento, seguida pela repetição integral desta Task.

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
$env:TRADUZAI_MATRIX_MITCH39_SOURCE = (Resolve-Path -LiteralPath 'N:\TraduzAI\temporario1\mch39').Path
.\pipeline\venv\Scripts\python.exe -m pytest pipeline\tests\test_universal_source_replacement_manifest.py::test_real_regression_source_tree_hash_matches_runtime_input -q
if ($LASTEXITCODE -ne 0) { throw "mch39 source manifest integration mismatch: $LASTEXITCODE" }
.\pipeline\venv\Scripts\python.exe pipeline\tools\init_owner_validation_run.py `
  --base '.codex-tmp\universal-source-replacement-20260804' `
  --source $env:TRADUZAI_MATRIX_MITCH39_SOURCE `
  --context '.codex-tmp\universal-source-replacement-current.json'
if ($LASTEXITCODE -ne 0) { throw "validation context init failed: $LASTEXITCODE" }
$validationContext = Get-Content -LiteralPath '.codex-tmp\universal-source-replacement-current.json' -Raw | ConvertFrom-Json
$offOut = [string]$validationContext.off_out
$env:TRADUZAI_REQUIRE_GPU = '1'

.\pipeline\venv\Scripts\python.exe pipeline\main.py `
  --input $env:TRADUZAI_MATRIX_MITCH39_SOURCE `
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
$sourcePath = [string]$validationContext.source_path
$env:TRADUZAI_REQUIRE_GPU = '1'
.\pipeline\venv\Scripts\python.exe pipeline\tools\audit_owner_chapter_output.py `
  --source $sourcePath `
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
source_manifest_page_count == 42
source_manifest_matches_final_page_set == true
all_project_asset_paths_resolve == true
all_project_asset_hashes_match == true
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
external_page_audits_count == 42
external_page_audit_page_ids == source_manifest_page_ids
external_page_audit_invocation_ids_are_unique == true
external_page_audit_execution_ids_are_unique_and_external == true
external_page_audit_process_nonces_are_unique == true
external_english_dialogue_residual_count == 0
unowned_external_source_residuals == 0
auditor_invocation_is_independent == true
auditor_root_hash_matches_final == true
auditor_unverified_physical_ocr_attempts == 0
auditor_cache_hits == 0
external_audit_gate_status == PASS
export_gate_status == PASS
```

**Step 4: Inspecionar visualmente os 42 pares reais**

Recarregue `.codex-tmp/universal-source-replacement-current.json` e use seu campo `off_review`; não dependa da variável de um bloco anterior. O auditor já deve ter criado `<off_review>/visual-review-evidence.json` com page/path/hash/execution/stages e `visual_verdict=null`; preencha somente os campos humanos após abrir as imagens. Abra com `view_image(detail="original")` cada PNG desse diretório. Avalie em resolução original:

- o inglês foi realmente removido, não apenas coberto por PT-BR;
- o PT-BR correspondente está presente uma vez;
- corpo completo não foi repartido entre owners/camadas;
- texto fica dentro e centralizado no container;
- borda, personagens, ícones e arte protegida não foram apagados;
- não há resíduo, duplicação, texto apagado pela metade ou copyback com band de cor diferente.

Registre verdict por página manualmente no evidence JSON; não use notas auto preenchidas. Para as regressões 10, 11, 19, 21, 27, 28, 30, 34, 36 e 39, abra também crops lossless source/final e documente o owner/estratégia que fechou cada caso.

Nessas dez sentinelas, abra ainda os seis estágios reais gerados pelo auditor — original, coverage/OCR overlay, inpaint, typeset, composição e bytes finais redecodificados — em escala nativa. Registre hash/verdict por estágio para localizar reaparecimento do inglês, perda do PT-BR, copyback incorreto ou dano de arte. Faça a mesma inspeção de seis estágios para qualquer página inicialmente marcada NO-GO antes de corrigir e rerodar.

Se qualquer página for NO-GO, volte à Task dona da causa, escreva um teste genérico RED, corrija, rode as suítes e repita o capítulo inteiro em uma nova raiz. Não edite pixels da saída.

**Step 5: Rodar compatibilidade de conteúdo com style-copy ativo**

```powershell
$validationContext = Get-Content -LiteralPath '.codex-tmp\universal-source-replacement-current.json' -Raw | ConvertFrom-Json
$offOut = [string]$validationContext.off_out
$styleOut = [string]$validationContext.style_out
$styleAudit = [string]$validationContext.style_audit
$styleReview = [string]$validationContext.style_review
$sourcePath = [string]$validationContext.source_path
$env:TRADUZAI_REQUIRE_GPU = '1'
.\pipeline\venv\Scripts\python.exe pipeline\main.py `
  --input $sourcePath `
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
  --style-copy-mode render `
  --replay-owner-artifacts $offOut
if ($LASTEXITCODE -ne 0) { throw "mch39 style-render run failed: $LASTEXITCODE" }

.\pipeline\venv\Scripts\python.exe pipeline\tools\audit_owner_chapter_output.py `
  --source $sourcePath `
  --run $styleOut `
  --report $styleAudit `
  --review-dir $styleReview `
  --source-lang en `
  --target-lang pt-BR `
  --compare-content-run $offOut `
  --require-final-verified
if ($LASTEXITCODE -ne 0) { throw "mch39 style-render audit failed: $LASTEXITCODE" }
```

O replay reutiliza o cleanup/base inpainted persistido da run `off` e reroda somente glyph render/style, composição e QA. Preserva o `run_id` de conteúdo, cria outro `execution_id` e registra `replay_of_execution_id`. Não reroda discovery/OCR de coverage, tradução nem inpaint; executa obrigatoriamente um OCR observer terminal novo, cujo `root_input_pixel_sha256` coincide com os pixels estilizados reabertos e cuja cadeia de attempts físicos é recalculada no boundary hash-bound. O auditor deve exigir `fresh_ocr_root_hash_matches_final=true`, zero attempts não verificados, zero cache hits e igualdade de `source_tree_sha256`, `owner_graph_sha256`, `translation_bindings_sha256` e `cleanup_base_sha256` antes de comparar estilo. O style copier pode alterar aparência, mas não coverage, owners, binding, cleanup ou a condição zero-inglês. Se style-copy não tiver confiança, o fallback obrigatório é o renderer base; não preservar source.

Abra com `view_image(detail="original")` **todos os 42 pares** de `$styleReview` e preencha `<style_review>/visual-review-evidence.json`, separado da run `off` e ligado a `execution_id`, paths e hashes. Aplique os mesmos critérios de inglês removido, PT-BR presente uma vez, corpo íntegro, layout e arte preservada. Nas dez sentinelas, abra novamente os seis estágios. A run `render` é obrigatória e bloqueia por conteúdo/hash/cardinalidade; uma run adicional `style_copy_mode=enforce` é opcional e seu NO-GO puramente estilístico pertence ao plano separado do copiador. Se essa run opcional alterar conteúdo ou reintroduzir inglês, então ela bloqueia por violação deste contrato.

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
  --audit-dir (Join-Path $validationRoot 'multiwork-audits') `
  --require-external-audit `
  --run-id universal-source-replacement-v1
if ($LASTEXITCODE -ne 0) { throw "multiwork matrix failed: $LASTEXITCODE" }

$auditDir = Join-Path $validationRoot 'multiwork-audits'
$matrixAudits = @(Get-ChildItem -LiteralPath $auditDir -Filter '*.json' -File)
if ($matrixAudits.Count -ne 4) { throw "expected 4 independent matrix audits, got $($matrixAudits.Count)" }
$zeroMetrics = @(
  'english_dialogue_residual_count',
  'translatable_components_without_owner',
  'material_components_without_ocr_attempt',
  'owners_without_valid_pt_br',
  'owners_without_atomic_cleanup_render',
  'owners_without_target_materialization',
  'material_components_without_terminal_lifecycle',
  'external_english_dialogue_residual_count',
  'unowned_external_source_residuals'
)
foreach ($auditFile in $matrixAudits) {
  $audit = Get-Content -LiteralPath $auditFile.FullName -Raw | ConvertFrom-Json
  $pageAudits = @($audit.external_page_audits)
  $expectedPageIds = @($audit.source_manifest_page_ids)
  if ($pageAudits.Count -ne [int]$audit.source_manifest_page_count) { throw "$($auditFile.Name): external page audit count mismatch" }
  $actualPageIds = @($pageAudits | ForEach-Object { [string]$_.page_id })
  if (($actualPageIds | ConvertTo-Json -Compress) -ne ($expectedPageIds | ConvertTo-Json -Compress)) { throw "$($auditFile.Name): external page audit bijection/order mismatch" }
  if (@($pageAudits.auditor_invocation_id | Sort-Object -Unique).Count -ne $pageAudits.Count) { throw "$($auditFile.Name): reused auditor invocation" }
  if (@($pageAudits.auditor_execution_id | Sort-Object -Unique).Count -ne $pageAudits.Count) { throw "$($auditFile.Name): reused auditor execution" }
  if (@($pageAudits | Where-Object { $_.auditor_execution_id -eq $audit.execution_id }).Count -gt 0) { throw "$($auditFile.Name): auditor reused runtime execution" }
  if (@($pageAudits.auditor_process_nonce | Sort-Object -Unique).Count -ne $pageAudits.Count) { throw "$($auditFile.Name): reused auditor process nonce" }
  foreach ($metric in $zeroMetrics) {
    if ([int]$audit.$metric -ne 0) { throw "$($auditFile.Name): $metric=$($audit.$metric)" }
  }
  if (-not $audit.auditor_invocation_is_independent) { throw "$($auditFile.Name): auditor is not independent" }
  if (-not $audit.auditor_root_hash_matches_final) { throw "$($auditFile.Name): auditor root mismatch" }
  if ([int]$audit.auditor_unverified_physical_ocr_attempts -ne 0) { throw "$($auditFile.Name): unverified OCR attempts" }
  if ([int]$audit.auditor_cache_hits -ne 0) { throw "$($auditFile.Name): cached external OCR" }
  if ($audit.external_audit_gate_status -ne 'PASS' -or $audit.export_gate_status -ne 'PASS') {
    throw "$($auditFile.Name): gate is not PASS"
  }
}
```

O runner da matriz deve lançar `audit_owner_chapter_output.py` como um **subprocesso real distinto por entry**, nunca importar/chamar seu `main()` no mesmo processo; o seam de teste injeta apenas o comando executável e comprova PIDs distintos. Dentro de cada entry, o auditor cria ainda um child OCR novo por página e grava um JSON hash-linked em `multiwork-audits/<entry_id>.json`. Depois do comando, reabra cada JSON e exija programaticamente as sete métricas internas, `external_english_dialogue_residual_count` e `unowned_external_source_residuals` zerados, `external_page_audits` em bijeção ordenada com `ChapterSourceManifest`, invocation/process nonce únicos, root correto por final, zero cache/unverified attempts e ambos os gates `PASS`.

Inspecione com `view_image(detail="original")` todos os outputs reais de Regressed Genius, 1 Second e Grand Finale pelo mesmo procedimento source/final e registre verdict por página/entry. A matriz contém uma entrada `calibration` e dois `holdout`; exija as mesmas métricas de conteúdo para as três. Não aceite GO apenas porque o comando retornou exit 0.

Depois de preencher manualmente todos os verdicts, valide a completude antes de escrever o relatório:

```powershell
$validationContext = Get-Content -LiteralPath '.codex-tmp\universal-source-replacement-current.json' -Raw | ConvertFrom-Json
$validationRoot = [string]$validationContext.validation_root
$offEvidence = Join-Path ([string]$validationContext.off_review) 'visual-review-evidence.json'
$renderEvidence = Join-Path ([string]$validationContext.style_review) 'visual-review-evidence.json'
$matrixEvidence = Join-Path $validationRoot 'multiwork-inspection.json'
$visualValidation = Join-Path $validationRoot 'visual-review-validation.json'
.\pipeline\venv\Scripts\python.exe pipeline\tools\validate_visual_review_evidence.py `
  --off-audit ([string]$validationContext.off_audit) `
  --render-audit ([string]$validationContext.style_audit) `
  --matrix-manifest 'pipeline\tests\fixtures\owner_visual_matrix\functional_matrix.json' `
  --matrix-audit-dir (Join-Path $validationRoot 'multiwork-audits') `
  --off-evidence $offEvidence `
  --render-evidence $renderEvidence `
  --matrix-evidence $matrixEvidence `
  --required-sentinel-pages '10,11,19,21,27,28,30,34,36,39' `
  --report $visualValidation `
  --require-complete
if ($LASTEXITCODE -ne 0) { throw "visual review evidence is incomplete or stale: $LASTEXITCODE" }
```

Expected: PASS com conjuntos exatos, nenhuma duplicata/path/hash/lineage stale, 42 verdicts `off`, 42 `render`, todas as páginas das quatro entries e seis stages por sentinela nas duas runs. Qualquer ausência impede o relatório/commit e volta à inspeção, não é convertida em warning.

**Step 7: Produzir relatório honesto e reproduzível**

O Markdown e o JSON devem incluir:

- branch, HEAD inicial/final e commits das Tasks;
- comandos, duração, hardware/device e paths das runs;
- hashes de source/output/audits;
- contagem por estado do ledger, estratégia R0–R3 e tipo de issue;
- métricas exatas da run end-to-end `off`, do replay ativo `render` e das três obras externas (uma calibration e dois holdouts);
- tabelas visuais separadas das 42 páginas `off` e das 42 páginas `render`, crops/estágios das dez regressões e todas as páginas externas;
- path/hash/status do `visual-review-validation.json` que prova a cardinalidade e a linhagem dos verdicts manuais;
- distinção entre validação automática e inspeção visual;
- qualquer falha restante como NO-GO, sem maquiar por warning.

Copie apenas os JSONs resumidos necessários para `docs/reports/evidence/`; não copie páginas, crops ou outputs para o Git.

**Step 8: Checkpoint F — Definition of Done**

Aceite o plano como concluído somente quando todos forem verdadeiros:

```text
[ ] Suíte Python completa passa.
[ ] A run end-to-end `off` e o replay integral de style ativo `render` têm 42/42 páginas final_verified.
[ ] Zero inglês material permanece em containers traduzíveis.
[ ] Todo texto PT-BR vinculado foi renderizado exatamente uma vez.
[ ] Todo componente material possui ao menos uma tentativa OCR registrada.
[ ] Nenhum owner está sem materialização target comprovada.
[ ] Nenhum componente traduzível está sem owner/tradução/commit.
[ ] OCR terminal de cada output prova root hash dos pixels finais, cadeia física íntegra e zero cache hit.
[ ] Auditor em processo novo prova invocation distinta, root dos finals reabertos, cadeia física íntegra e zero cache; prova terminal armazenada não satisfaz este item.
[ ] R3 fecha todos os casos de conteúdo que R0-R2 não fecharam.
[ ] Style-copy off e render preservam o mesmo contrato de conteúdo; shadow, se executado, não muta pixels; enforce opcional não bloqueia por fidelidade isolada.
[ ] As três outras obras — uma calibration e dois holdouts — têm auditoria externa, métricas zeradas e inspeção GO.
[ ] Todo artefato de inspeção possui path real e tupla page/source/path/lineage não ambígua; pixels legitimamente idênticos podem compartilhar pixel hash.
[ ] Export gate final é PASS sem override e sem with_warnings.
[ ] Relatório visual é baseado em inspeção própria, não em template.
[ ] `validate_visual_review_evidence.py --require-complete` passa para 42 off + 42 render + todas as quatro entries e seis stages das sentinelas.
```

Se um item estiver falso, o trabalho continua na Task responsável; não declarar conclusão parcial como sucesso.

**Step 9: Commit do relatório, sem outputs**

```powershell
git status --short --branch
$guard = '.codex-tmp\universal-source-replacement-plan-guard\TaskOwnedGit.ps1'
& $guard -Action Stage -Task 20
git diff --cached --check
git diff --cached -- docs/reports/2026-08-04-universal-balloon-source-replacement-validation.md docs/reports/evidence/2026-08-04-universal-balloon-source-replacement-validation.json
git commit -m "docs: validate universal balloon source replacement"
& $guard -Action VerifyAfterCommit -Task 20
Remove-Item Env:\TRADUZAI_REQUIRE_GPU -ErrorAction SilentlyContinue
Get-ChildItem Env:TRADUZAI_MATRIX_* -ErrorAction SilentlyContinue | Remove-Item -ErrorAction SilentlyContinue
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

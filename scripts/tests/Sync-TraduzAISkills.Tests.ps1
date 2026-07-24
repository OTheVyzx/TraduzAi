[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$skillNames = @(
    'mangatl-dev'
    'traduzai-detect'
    'traduzai-ocr'
    'traduzai-inpaint'
    'traduzai-typesetting'
    'traduzai-pipeline'
    'traduzai-translation'
    'traduzai-studio'
)

function Assert-True {
    param([bool]$Condition, [string]$Message)

    if (-not $Condition) {
        throw "ASSERT: $Message"
    }
}

function Assert-BytesEqual {
    param([byte[]]$Expected, [byte[]]$Actual, [string]$Message)

    Assert-True ($Expected.Length -eq $Actual.Length) "$Message (tamanho)"
    for ($index = 0; $index -lt $Expected.Length; $index++) {
        if ($Expected[$index] -ne $Actual[$index]) {
            throw "ASSERT: $Message (byte $index)"
        }
    }
}

function Assert-SafeTestRoot {
    param([string]$Path)

    $tempRoot = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath()).TrimEnd('\', '/')
    $fullPath = [System.IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
    $prefix = $tempRoot + [System.IO.Path]::DirectorySeparatorChar
    Assert-True ($fullPath.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) 'raiz de teste fora do TEMP'
    Assert-True ([System.IO.Path]::GetFileName($fullPath).StartsWith('traduzai-skills-sync-test-', [System.StringComparison]::Ordinal)) 'nome inseguro para cleanup'
    return $fullPath
}

$scriptUnderTest = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\Sync-TraduzAISkills.ps1'))
$testRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("traduzai-skills-sync-test-{0}" -f [guid]::NewGuid().ToString('N'))
$testRoot = Assert-SafeTestRoot $testRoot
$sourceRoot = Join-Path $testRoot 'source'
$destinationRoot = Join-Path $testRoot 'destination'
$utf8 = New-Object System.Text.UTF8Encoding($false)
$fixtureBytes = @{}
$shellExe = (Get-Process -Id $PID).Path

function Invoke-Sync {
    param(
        [switch]$Check,
        [string]$Source = $sourceRoot,
        [string]$Destination = $destinationRoot,
        [string]$ScriptPath = $scriptUnderTest,
        [switch]$OmitSourceRoot
    )

    $arguments = @(
        '-NoProfile'
        '-ExecutionPolicy', 'Bypass'
        '-File', $ScriptPath
    )
    if (-not $OmitSourceRoot) {
        $arguments += @('-SourceRoot', $Source)
    }
    $arguments += @('-DestinationRoot', $Destination)
    if ($Check) {
        $arguments += '-Check'
    }
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $output = @(& $shellExe @arguments 2>&1)
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    foreach ($line in $output) {
        [Console]::Out.WriteLine($line.ToString())
    }
    return [int]$exitCode
}

try {
    [void](New-Item -ItemType Directory -Path $sourceRoot, $destinationRoot -Force)

    foreach ($skillName in $skillNames) {
        $sourceSkill = Join-Path $sourceRoot $skillName
        [void](New-Item -ItemType Directory -Path $sourceSkill -Force)
        $bytes = $utf8.GetBytes("---`nname: $skillName`ndescription: fixture-$skillName`n---`n`n# $skillName`n")
        $fixtureBytes[$skillName] = $bytes
        [System.IO.File]::WriteAllBytes((Join-Path $sourceSkill 'SKILL.md'), $bytes)
        [System.IO.File]::WriteAllText((Join-Path $sourceSkill 'source-only.txt'), 'nao copiar', $utf8)
    }

    $extraSkill = Join-Path $destinationRoot 'skill-local-nao-gerenciada'
    [void](New-Item -ItemType Directory -Path $extraSkill -Force)
    $extraBytes = $utf8.GetBytes('preservar skill local')
    [System.IO.File]::WriteAllBytes((Join-Path $extraSkill 'SKILL.md'), $extraBytes)

    $exitCode = Invoke-Sync
    Assert-True ($exitCode -eq 0) "sync inicial falhou com exit code $exitCode"

    foreach ($skillName in $skillNames) {
        $destinationSkill = Join-Path (Join-Path $destinationRoot $skillName) 'SKILL.md'
        Assert-True (Test-Path -LiteralPath $destinationSkill -PathType Leaf) "destino ausente: $skillName"
        Assert-BytesEqual $fixtureBytes[$skillName] ([System.IO.File]::ReadAllBytes($destinationSkill)) "bytes divergentes: $skillName"
        Assert-True (-not (Test-Path -LiteralPath (Join-Path (Split-Path $destinationSkill) 'source-only.txt'))) "arquivo extra copiado: $skillName"
    }
    Assert-BytesEqual $extraBytes ([System.IO.File]::ReadAllBytes((Join-Path $extraSkill 'SKILL.md'))) 'nona skill foi alterada'

    $beforeCheck = Get-ChildItem -LiteralPath $destinationRoot -File -Recurse | ForEach-Object {
        [pscustomobject]@{
            Path = $_.FullName
            Bytes = [System.IO.File]::ReadAllBytes($_.FullName)
            LastWriteTimeUtc = $_.LastWriteTimeUtc
        }
    }
    $exitCode = Invoke-Sync -Check
    Assert-True ($exitCode -eq 0) "check alinhado falhou com exit code $exitCode"
    $afterCheck = @(Get-ChildItem -LiteralPath $destinationRoot -File -Recurse)
    Assert-True ($afterCheck.Count -eq $beforeCheck.Count) 'check criou ou removeu arquivo'
    foreach ($snapshot in $beforeCheck) {
        $current = Get-Item -LiteralPath $snapshot.Path
        Assert-BytesEqual $snapshot.Bytes ([System.IO.File]::ReadAllBytes($snapshot.Path)) "check escreveu em $($snapshot.Path)"
        Assert-True ($current.LastWriteTimeUtc -eq $snapshot.LastWriteTimeUtc) "check tocou timestamp de $($snapshot.Path)"
    }

    $absentDestination = Join-Path $testRoot 'check-destination-ausente'
    $exitCode = Invoke-Sync -Check -Destination $absentDestination
    Assert-True ($exitCode -ne 0) 'check aceitou destino inteiramente ausente'
    Assert-True (-not (Test-Path -LiteralPath $absentDestination)) 'check criou DestinationRoot ausente'

    $defaultRepoRoot = Join-Path $testRoot 'default-source-repo'
    $defaultScriptsRoot = Join-Path $defaultRepoRoot 'scripts'
    $defaultSourceRoot = Join-Path (Join-Path $defaultRepoRoot '.agents') 'skills'
    $defaultDestination = Join-Path $testRoot 'default-source-destination'
    [void](New-Item -ItemType Directory -Path $defaultScriptsRoot, $defaultSourceRoot -Force)
    $copiedSync = Join-Path $defaultScriptsRoot 'Sync-TraduzAISkills.ps1'
    Copy-Item -LiteralPath $scriptUnderTest -Destination $copiedSync
    foreach ($skillName in $skillNames) {
        $defaultSkillRoot = Join-Path $defaultSourceRoot $skillName
        [void](New-Item -ItemType Directory -Path $defaultSkillRoot -Force)
        [System.IO.File]::WriteAllBytes((Join-Path $defaultSkillRoot 'SKILL.md'), $fixtureBytes[$skillName])
    }
    $exitCode = Invoke-Sync -ScriptPath $copiedSync -Destination $defaultDestination -OmitSourceRoot
    Assert-True ($exitCode -eq 0) "sync sem SourceRoot falhou com exit code $exitCode"
    foreach ($skillName in $skillNames) {
        $defaultDestinationFile = Join-Path (Join-Path $defaultDestination $skillName) 'SKILL.md'
        Assert-BytesEqual $fixtureBytes[$skillName] ([System.IO.File]::ReadAllBytes($defaultDestinationFile)) "SourceRoot padrao incorreto: $skillName"
    }

    $divergentPath = Join-Path (Join-Path $destinationRoot 'traduzai-ocr') 'SKILL.md'
    $divergentBytes = $utf8.GetBytes('destino divergente')
    [System.IO.File]::WriteAllBytes($divergentPath, $divergentBytes)
    $divergentTime = (Get-Item -LiteralPath $divergentPath).LastWriteTimeUtc
    $exitCode = Invoke-Sync -Check
    Assert-True ($exitCode -ne 0) 'check nao detectou divergencia'
    Assert-BytesEqual $divergentBytes ([System.IO.File]::ReadAllBytes($divergentPath)) 'check corrigiu divergencia indevidamente'
    Assert-True ((Get-Item -LiteralPath $divergentPath).LastWriteTimeUtc -eq $divergentTime) 'check tocou arquivo divergente'

    [System.IO.File]::WriteAllBytes($divergentPath, $fixtureBytes['traduzai-ocr'])
    $missingDestination = Join-Path (Join-Path $destinationRoot 'traduzai-translation') 'SKILL.md'
    Remove-Item -LiteralPath $missingDestination
    $exitCode = Invoke-Sync -Check
    Assert-True ($exitCode -ne 0) 'check nao detectou destino ausente'
    Assert-True (-not (Test-Path -LiteralPath $missingDestination)) 'check criou destino ausente'
    [System.IO.File]::WriteAllBytes($missingDestination, $fixtureBytes['traduzai-translation'])

    $missingSource = Join-Path (Join-Path $sourceRoot 'traduzai-studio') 'SKILL.md'
    Remove-Item -LiteralPath $missingSource
    $protectedPath = Join-Path (Join-Path $destinationRoot 'mangatl-dev') 'SKILL.md'
    $protectedBytes = [System.IO.File]::ReadAllBytes($protectedPath)
    $exitCode = Invoke-Sync
    Assert-True ($exitCode -ne 0) 'sync aceitou origem incompleta'
    Assert-BytesEqual $protectedBytes ([System.IO.File]::ReadAllBytes($protectedPath)) 'sync escreveu antes de validar todas as origens'

    Write-Output '[OK] Sync-TraduzAISkills'
}
finally {
    if (Test-Path -LiteralPath $testRoot) {
        $safeRoot = Assert-SafeTestRoot $testRoot
        Remove-Item -LiteralPath $safeRoot -Recurse -Force
    }
}

[CmdletBinding()]
param(
    [string]$SourceRoot,

    [string]$DestinationRoot = 'C:\Users\PICHAU\.agents\skills',

    [switch]$Check
)

$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($SourceRoot)) {
    $SourceRoot = Join-Path (Join-Path $PSScriptRoot '..') '.agents\skills'
}

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

function Get-Sha256 {
    param([byte[]]$Bytes)

    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        return ([System.BitConverter]::ToString($sha256.ComputeHash($Bytes))).Replace('-', '')
    }
    finally {
        $sha256.Dispose()
    }
}

$resolvedSourceRoot = (Resolve-Path -LiteralPath $SourceRoot -ErrorAction Stop).Path
$resolvedDestinationRoot = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($DestinationRoot)
$sources = New-Object System.Collections.Generic.List[object]
$sourceErrors = New-Object System.Collections.Generic.List[string]

# Preflight completo: nenhuma pasta ou arquivo de destino existe antes deste loop terminar.
foreach ($skillName in $skillNames) {
    $sourcePath = Join-Path (Join-Path $resolvedSourceRoot $skillName) 'SKILL.md'
    if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
        $sourceErrors.Add("origem ausente: $sourcePath")
        continue
    }

    try {
        $bytes = [System.IO.File]::ReadAllBytes($sourcePath)
        $sources.Add([pscustomobject]@{
            Name = $skillName
            Path = $sourcePath
            Bytes = $bytes
            Hash = Get-Sha256 $bytes
        })
    }
    catch {
        $sourceErrors.Add("origem ilegivel: $sourcePath - $($_.Exception.Message)")
    }
}

if ($sourceErrors.Count -gt 0) {
    throw "Falha na validacao das origens; nada foi escrito.`n$($sourceErrors -join "`n")"
}

if ($Check) {
    $divergences = New-Object System.Collections.Generic.List[string]
    foreach ($source in $sources) {
        $destinationPath = Join-Path (Join-Path $resolvedDestinationRoot $source.Name) 'SKILL.md'
        if (-not (Test-Path -LiteralPath $destinationPath -PathType Leaf)) {
            $divergences.Add("ausente: $destinationPath")
            continue
        }

        $destinationHash = Get-Sha256 ([System.IO.File]::ReadAllBytes($destinationPath))
        if ($destinationHash -cne $source.Hash) {
            $divergences.Add("SHA256 divergente: $destinationPath")
        }
    }

    if ($divergences.Count -gt 0) {
        foreach ($divergence in $divergences) {
            Write-Output "[DIVERGENTE] $divergence"
        }
        exit 1
    }

    Write-Output '[OK] oito skills alinhados; nenhuma escrita realizada'
    exit 0
}

[void](New-Item -ItemType Directory -Path $resolvedDestinationRoot -Force)
foreach ($source in $sources) {
    $destinationDirectory = Join-Path $resolvedDestinationRoot $source.Name
    $destinationPath = Join-Path $destinationDirectory 'SKILL.md'
    [void](New-Item -ItemType Directory -Path $destinationDirectory -Force)

    $destinationHash = $null
    if (Test-Path -LiteralPath $destinationPath -PathType Leaf) {
        $destinationHash = Get-Sha256 ([System.IO.File]::ReadAllBytes($destinationPath))
    }
    if ($destinationHash -cne $source.Hash) {
        Copy-Item -LiteralPath $source.Path -Destination $destinationPath -Force
    }

    $copiedHash = Get-Sha256 ([System.IO.File]::ReadAllBytes($destinationPath))
    if ($copiedHash -cne $source.Hash) {
        throw "Falha de integridade SHA256 apos copia: $destinationPath"
    }
    Write-Output "[OK] $($source.Name) $($source.Hash)"
}

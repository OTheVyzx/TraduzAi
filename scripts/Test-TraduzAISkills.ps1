[CmdletBinding()]
param(
    [string]$SkillsRoot
)

if ([string]::IsNullOrWhiteSpace($SkillsRoot)) {
    $SkillsRoot = Join-Path (Split-Path -Parent $PSScriptRoot) '.agents\skills'
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

$utf8 = New-Object System.Text.UTF8Encoding($false, $true)
$ultimaVerificacaoHeading = '## ' + [char]0x00DA + 'ltima verifica' + [char]0x00E7 + [char]0x00E3 + 'o'
$mojibakeSequences = @(
    [string][char]0x00C3
    [string][char]0x00C2
    ([string][char]0x00E2 + [char]0x20AC)
    [string][char]0xFFFD
)
$hasErrors = $false

foreach ($skillName in $skillNames) {
    $skillFile = Join-Path (Join-Path $SkillsRoot $skillName) 'SKILL.md'
    $errors = New-Object System.Collections.Generic.List[string]
    $content = $null

    if (-not (Test-Path -LiteralPath $skillFile -PathType Leaf)) {
        $errors.Add('SKILL.md nao existe')
    }
    else {
        try {
            $bytes = [System.IO.File]::ReadAllBytes($skillFile)
            $content = $utf8.GetString($bytes)
        }
        catch [System.Text.DecoderFallbackException] {
            $errors.Add('UTF-8 invalido')
        }
        catch {
            $errors.Add("falha ao ler SKILL.md: $($_.Exception.Message)")
        }
    }

    if ($null -ne $content) {
        $frontmatterMatch = [regex]::Match($content, '(?s)\A---\r?\n(?<frontmatter>.*?)\r?\n---(?:\r?\n|\z)')

        if (-not $frontmatterMatch.Success) {
            $errors.Add('frontmatter ausente ou invalido')
        }
        else {
            $frontmatter = $frontmatterMatch.Groups['frontmatter'].Value
            $nameMatch = [regex]::Match($frontmatter, '(?m)^\s*name\s*:\s*(?<value>.*?)\s*$')
            $declaredName = if ($nameMatch.Success) {
                $nameMatch.Groups['value'].Value.Trim().Trim('"').Trim("'")
            }
            else {
                $null
            }

            if ($declaredName -cne $skillName) {
                $errors.Add("frontmatter name deve ser '$skillName'")
            }

            $descriptionMatch = [regex]::Match($frontmatter, '(?m)^\s*description\s*:\s*(?<value>.*?)\s*$')
            if (-not $descriptionMatch.Success -or [string]::IsNullOrWhiteSpace($descriptionMatch.Groups['value'].Value)) {
                $errors.Add('frontmatter description ausente')
            }
        }

        $headingPattern = '(?m)^' + [regex]::Escape($ultimaVerificacaoHeading) + '\s*$'
        if (-not [regex]::IsMatch($content, $headingPattern)) {
            $errors.Add('secao Ultima verificacao ausente')
        }

        foreach ($sequence in $mojibakeSequences) {
            if ($content.Contains($sequence)) {
                $errors.Add('texto contem mojibake')
                break
            }
        }
    }

    if ($errors.Count -eq 0) {
        Write-Output "[OK] $skillName"
    }
    else {
        $hasErrors = $true
        Write-Output "[ERRO] $skillName - $($errors -join '; ')"
    }
}

if ($hasErrors) {
    exit 1
}

exit 0

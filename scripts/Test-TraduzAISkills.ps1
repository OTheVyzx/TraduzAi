[CmdletBinding()]
param(
    [string]$SkillsRoot
)

if ([string]::IsNullOrWhiteSpace($SkillsRoot)) {
    $SkillsRoot = Join-Path (Split-Path -Parent $PSScriptRoot) '.agents\skills'
}

function ConvertFrom-FrontmatterScalar {
    param(
        [string]$RawValue
    )

    $trimmed = $RawValue.Trim()

    if ($trimmed.StartsWith('"')) {
        $quotedValue = [regex]::Match($trimmed, '^"(?<value>(?:\\.|[^"])*)"[ \t]*(?:#.*)?$')
        if (-not $quotedValue.Success) {
            return $null
        }

        return $quotedValue.Groups['value'].Value
    }

    if ($trimmed.StartsWith([string][char]0x0027)) {
        $quotedValue = [regex]::Match($trimmed, '^\x27(?<value>(?:\x27\x27|[^\x27])*)\x27[ \t]*(?:#.*)?$')
        if (-not $quotedValue.Success) {
            return $null
        }

        return $quotedValue.Groups['value'].Value.Replace("''", [string][char]0x0027)
    }

    if ($trimmed.StartsWith('#')) {
        return ''
    }

    return [regex]::Replace($trimmed, '[ \t]+#.*$', '').Trim()
}

$continuationByteArtifacts = @(
    0x0192
    0x0152
    0x0153
    0x0160
    0x0161
    0x0178
    0x017D
    0x017E
    0x02C6
    0x02DC
    0x2013
    0x2014
    0x2018
    0x2019
    0x201A
    0x201C
    0x201D
    0x201E
    0x2020
    0x2021
    0x2022
    0x2026
    0x2030
    0x2039
    0x203A
    0x20AC
    0x2122
)

function Test-ContainsMojibake {
    param(
        [string]$Text
    )

    for ($index = 0; $index -lt $Text.Length; $index++) {
        $currentCodePoint = [int]$Text[$index]

        if ($currentCodePoint -eq 0xFFFD) {
            return $true
        }

        if ($index + 1 -ge $Text.Length) {
            continue
        }

        $nextCodePoint = [int]$Text[$index + 1]
        if ($currentCodePoint -eq 0x00E2 -and $nextCodePoint -eq 0x20AC) {
            return $true
        }

        if ($currentCodePoint -ne 0x00C3 -and $currentCodePoint -ne 0x00C2) {
            continue
        }

        $isContinuationByteArtifact =
            ($nextCodePoint -ge 0x0080 -and $nextCodePoint -le 0x00BF) -or
            ($continuationByteArtifacts -contains $nextCodePoint)

        if ($isContinuationByteArtifact) {
            return $true
        }
    }

    return $false
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
            $nameMatches = [regex]::Matches($frontmatter, '(?m)^name[ \t]*:[ \t]*(?<value>[^\r\n]*)\r?$')
            if ($nameMatches.Count -eq 0) {
                $errors.Add("frontmatter name deve ser '$skillName'")
            }
            elseif ($nameMatches.Count -gt 1) {
                $errors.Add('frontmatter name duplicado')
            }
            else {
                $declaredName = ConvertFrom-FrontmatterScalar $nameMatches[0].Groups['value'].Value
                if ($declaredName -cne $skillName) {
                    $errors.Add("frontmatter name deve ser '$skillName'")
                }
            }

            $descriptionMatches = [regex]::Matches($frontmatter, '(?m)^description[ \t]*:[ \t]*(?<value>[^\r\n]*)\r?$')
            if ($descriptionMatches.Count -eq 0) {
                $errors.Add('frontmatter description ausente')
            }
            elseif ($descriptionMatches.Count -gt 1) {
                $errors.Add('frontmatter description duplicada')
            }
            else {
                $description = ConvertFrom-FrontmatterScalar $descriptionMatches[0].Groups['value'].Value
                if ([string]::IsNullOrWhiteSpace($description)) {
                    $errors.Add('frontmatter description ausente')
                }
            }
        }

        $headingPattern = '(?m)^' + [regex]::Escape($ultimaVerificacaoHeading) + '\s*$'
        if (-not [regex]::IsMatch($content, $headingPattern)) {
            $errors.Add('secao Ultima verificacao ausente')
        }

        if (Test-ContainsMojibake $content) {
            $errors.Add('texto contem mojibake')
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

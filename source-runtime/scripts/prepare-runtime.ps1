[CmdletBinding()]
param(
    [string]$ToolsRoot = (Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) '.source-runtime-tools')
)

$ErrorActionPreference = 'Stop'
$runtimeRoot = (Resolve-Path (Split-Path -Parent $PSScriptRoot)).Path
$workspaceRoot = (Resolve-Path (Split-Path -Parent $runtimeRoot)).Path
$jdkHome = Join-Path $ToolsRoot 'jdk-21.0.12.1+1'
$gradle = Join-Path $ToolsRoot 'gradle-8.14.3\bin\gradle.bat'
if (-not (Test-Path -LiteralPath (Join-Path $jdkHome 'bin\jlink.exe'))) { throw 'JDK privado não encontrado; execute bootstrap-windows.ps1' }
if (-not (Test-Path -LiteralPath $gradle)) { throw 'Gradle fixado não encontrado; execute bootstrap-windows.ps1' }

$env:JAVA_HOME = $jdkHome
$env:TRADUZAI_SOURCE_RUNTIME_TOOLS = (Resolve-Path -LiteralPath $ToolsRoot).Path
$env:GRADLE_OPTS = '-Xmx4g -Dfile.encoding=UTF-8 -Dkotlin.compiler.execution.strategy=in-process'
& $gradle -p $runtimeRoot clean :runtime-broker:installDist --no-daemon --no-configuration-cache
if ($LASTEXITCODE -ne 0) { throw 'Build do broker falhou' }

$packageRoot = Join-Path $runtimeRoot 'build\package\source-runtime'
$expectedPackageRoot = Join-Path $runtimeRoot 'build\package'
$resolvedParent = [IO.Path]::GetFullPath((Split-Path -Parent $packageRoot))
if ($resolvedParent -ne [IO.Path]::GetFullPath($expectedPackageRoot)) { throw 'Destino de pacote inválido' }
if (Test-Path -LiteralPath $packageRoot) { Remove-Item -LiteralPath $packageRoot -Recurse -Force }
New-Item -ItemType Directory -Force -Path $packageRoot | Out-Null

$jre = Join-Path $packageRoot 'jre'
$modules = 'java.base,java.desktop,java.instrument,java.logging,java.management,java.naming,java.net.http,java.prefs,java.security.jgss,java.sql,java.xml,jdk.crypto.ec,jdk.unsupported,jdk.zipfs'
& (Join-Path $jdkHome 'bin\jlink.exe') --module-path (Join-Path $jdkHome 'jmods') --add-modules $modules --strip-debug --no-header-files --no-man-pages --compress=zip-6 --output $jre
if ($LASTEXITCODE -ne 0) { throw 'jlink falhou' }

$brokerSource = Join-Path $runtimeRoot 'runtime-broker\build\install\runtime-broker'
$brokerTarget = Join-Path $packageRoot 'broker'
Copy-Item -LiteralPath $brokerSource -Destination $brokerTarget -Recurse
Copy-Item -LiteralPath (Join-Path $runtimeRoot 'UPSTREAM.lock.json') -Destination $packageRoot
Copy-Item -LiteralPath (Join-Path $runtimeRoot 'THIRD_PARTY_NOTICES.md') -Destination $packageRoot

$components = Get-ChildItem -LiteralPath (Join-Path $brokerTarget 'lib') -File | Sort-Object Name | ForEach-Object {
    [ordered]@{
        type = 'library'
        name = $_.Name
        hashes = @([ordered]@{ alg = 'SHA-256'; content = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant() })
    }
}
$sbom = [ordered]@{
    bomFormat = 'CycloneDX'
    specVersion = '1.5'
    version = 1
    metadata = [ordered]@{ component = [ordered]@{ type = 'application'; name = 'traduzai-source-runtime'; version = '0.1.0' } }
    components = @($components)
}
$sbom | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $packageRoot 'sbom.cdx.json') -Encoding utf8

Write-Output "SOURCE_RUNTIME_PACKAGE=$packageRoot"

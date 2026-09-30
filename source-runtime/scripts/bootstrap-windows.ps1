[CmdletBinding()]
param(
    [string]$Destination = (Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) '.source-runtime-tools')
)

$ErrorActionPreference = 'Stop'
$jdkName = 'OpenJDK21U-jdk_x64_windows_hotspot_21.0.12.1_1.zip'
$jdkUrl = 'https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.12.1%2B1/OpenJDK21U-jdk_x64_windows_hotspot_21.0.12.1_1.zip'
$jdkSha = 'f9d6e191ab098c0d416e7d588a24420a8621cd2f4720dab2459b8b7b2d2d8b4e'
$gradleName = 'gradle-8.14.3-bin.zip'
$gradleUrl = 'https://services.gradle.org/distributions/gradle-8.14.3-bin.zip'
$gradleSha = 'bd71102213493060956ec229d946beee57158dbd89d0e62b91bca0fa2c5f3531'
$suwayomiRepository = 'https://github.com/Suwayomi/Suwayomi-Server.git'
$suwayomiCommit = 'ac3dd314dba275fdadc4a3208002a9fa42135bf2'

New-Item -ItemType Directory -Force -Path $Destination | Out-Null

function Get-VerifiedArchive([string]$Name, [string]$Url, [string]$Sha256) {
    $archive = Join-Path $Destination $Name
    if (-not (Test-Path -LiteralPath $archive)) {
        curl.exe -L --fail --retry 3 --output $archive $Url
        if ($LASTEXITCODE -ne 0) { throw "Falha ao baixar $Name" }
    }
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash.ToLowerInvariant()
    if ($actual -ne $Sha256) { throw "SHA-256 divergente para $Name`: $actual" }
    return $archive
}

$jdkArchive = Get-VerifiedArchive $jdkName $jdkUrl $jdkSha
$gradleArchive = Get-VerifiedArchive $gradleName $gradleUrl $gradleSha

$jdkHome = Join-Path $Destination 'jdk-21.0.12.1+1'
if (-not (Test-Path -LiteralPath $jdkHome)) {
    Expand-Archive -LiteralPath $jdkArchive -DestinationPath $Destination
}
$gradleHome = Join-Path $Destination 'gradle-8.14.3'
if (-not (Test-Path -LiteralPath $gradleHome)) {
    Expand-Archive -LiteralPath $gradleArchive -DestinationPath $Destination
}

$suwayomiRoot = Join-Path $Destination 'suwayomi-server'
if (-not (Test-Path -LiteralPath (Join-Path $suwayomiRoot '.git'))) {
    git clone --filter=blob:none --no-checkout $suwayomiRepository $suwayomiRoot
    if ($LASTEXITCODE -ne 0) { throw 'Falha ao clonar o snapshot Suwayomi' }
    git -C $suwayomiRoot fetch origin $suwayomiCommit --depth 1
    if ($LASTEXITCODE -ne 0) { throw 'Falha ao obter o commit fixado do Suwayomi' }
    git -C $suwayomiRoot checkout --detach $suwayomiCommit
    if ($LASTEXITCODE -ne 0) { throw 'Falha ao fixar o commit do Suwayomi' }
}

$suwayomiHead = (git -C $suwayomiRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $suwayomiHead -ne $suwayomiCommit) {
    throw "Checkout Suwayomi divergente: esperado $suwayomiCommit, encontrado $suwayomiHead"
}
$suwayomiDirty = @(git -C $suwayomiRoot status --porcelain)
if ($LASTEXITCODE -ne 0) { throw 'Falha ao verificar o checkout Suwayomi' }
if ($suwayomiDirty.Count -gt 0) { throw 'Checkout Suwayomi possui alterações locais; o bootstrap não irá sobrescrevê-las' }

$suwayomiLib = Join-Path $suwayomiRoot 'server\build\install\server\lib'
if (-not (Test-Path -LiteralPath (Join-Path $suwayomiLib 'server-1.0.jar'))) {
    $previousJavaHome = $env:JAVA_HOME
    $previousGradleOpts = $env:GRADLE_OPTS
    try {
        $env:JAVA_HOME = $jdkHome
        $env:GRADLE_OPTS = '-Xmx6g -Dfile.encoding=UTF-8 -Dkotlin.compiler.execution.strategy=in-process'
        & (Join-Path $suwayomiRoot 'gradlew.bat') :server:installDist --no-daemon --no-configuration-cache
        if ($LASTEXITCODE -ne 0) { throw 'Build do snapshot Suwayomi/AndroidCompat falhou' }
    } finally {
        $env:JAVA_HOME = $previousJavaHome
        $env:GRADLE_OPTS = $previousGradleOpts
    }
}
if (-not (Test-Path -LiteralPath (Join-Path $suwayomiLib 'AndroidCompat-1.0.jar'))) {
    throw 'AndroidCompat fixado não foi produzido pelo build Suwayomi'
}

[pscustomobject]@{
    JavaHome = $jdkHome
    GradleHome = $gradleHome
    SuwayomiCommit = $suwayomiHead
    SuwayomiCompatibilityLibs = $suwayomiLib
}

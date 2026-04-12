# ============================================================
# MangáTL — Setup Completo no Windows
# Execute este script no PowerShell como Administrador
# ============================================================

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " MangaTL - Setup Windows" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ─── 1. Verificar o que ja tem instalado ─────────────────────

Write-Host "[1/7] Verificando ferramentas instaladas..." -ForegroundColor Yellow

$hasNode = Get-Command node -ErrorAction SilentlyContinue
$hasRust = Get-Command rustc -ErrorAction SilentlyContinue
$hasCargo = Get-Command cargo -ErrorAction SilentlyContinue
$hasPython = Get-Command python -ErrorAction SilentlyContinue
$hasGit = Get-Command git -ErrorAction SilentlyContinue
$hasWinget = Get-Command winget -ErrorAction SilentlyContinue

if ($hasNode) { Write-Host "  Node.js: $(node -v)" -ForegroundColor Green }
else { Write-Host "  Node.js: NAO ENCONTRADO" -ForegroundColor Red }

if ($hasRust) { Write-Host "  Rust: $(rustc --version)" -ForegroundColor Green }
else { Write-Host "  Rust: NAO ENCONTRADO" -ForegroundColor Red }

if ($hasPython) { Write-Host "  Python: $(python --version)" -ForegroundColor Green }
else { Write-Host "  Python: NAO ENCONTRADO" -ForegroundColor Red }

if ($hasGit) { Write-Host "  Git: $(git --version)" -ForegroundColor Green }
else { Write-Host "  Git: NAO ENCONTRADO" -ForegroundColor Red }

Write-Host ""

# ─── 2. Instalar o que falta via winget ──────────────────────

Write-Host "[2/7] Instalando dependencias que faltam..." -ForegroundColor Yellow

if (-not $hasNode) {
    Write-Host "  Instalando Node.js 22 LTS..." -ForegroundColor Cyan
    winget install OpenJS.NodeJS.LTS --accept-source-agreements --accept-package-agreements
}

if (-not $hasRust) {
    Write-Host "  Instalando Rust via rustup..." -ForegroundColor Cyan
    winget install Rustlang.Rustup --accept-source-agreements --accept-package-agreements
    Write-Host "  >> IMPORTANTE: Feche e reabra o terminal apos instalar o Rust!" -ForegroundColor Red
}

if (-not $hasPython) {
    Write-Host "  Instalando Python 3.12..." -ForegroundColor Cyan
    winget install Python.Python.3.12 --accept-source-agreements --accept-package-agreements
}

if (-not $hasGit) {
    Write-Host "  Instalando Git..." -ForegroundColor Cyan
    winget install Git.Git --accept-source-agreements --accept-package-agreements
}

# Visual Studio Build Tools (necessario para compilar Rust no Windows)
Write-Host "  Verificando Visual Studio Build Tools..." -ForegroundColor Cyan
$hasVS = Test-Path "C:\Program Files (x86)\Microsoft Visual Studio\*\BuildTools" -ErrorAction SilentlyContinue
if (-not $hasVS) {
    $hasVS2 = Test-Path "C:\Program Files\Microsoft Visual Studio\*\*\VC" -ErrorAction SilentlyContinue
    if (-not $hasVS2) {
        Write-Host "  Instalando Visual Studio Build Tools (C++ workload)..." -ForegroundColor Cyan
        winget install Microsoft.VisualStudio.2022.BuildTools --accept-source-agreements --accept-package-agreements
        Write-Host "  >> Apos instalar, abra o Visual Studio Installer e adicione:" -ForegroundColor Red
        Write-Host "     'Desktop development with C++'" -ForegroundColor Red
    }
}

# WebView2 (necessario para Tauri no Windows)
Write-Host "  Verificando WebView2 Runtime..." -ForegroundColor Cyan
$webview2 = Get-ItemProperty "HKLM:\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" -ErrorAction SilentlyContinue
if (-not $webview2) {
    Write-Host "  Instalando WebView2 Runtime..." -ForegroundColor Cyan
    winget install Microsoft.EdgeWebView2Runtime --accept-source-agreements --accept-package-agreements
}

Write-Host ""
Write-Host "[OK] Dependencias do sistema verificadas." -ForegroundColor Green
Write-Host ""

# ─── 3. Atualizar PATH ──────────────────────────────────────

Write-Host "[3/7] Atualizando PATH..." -ForegroundColor Yellow
$env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

# ─── 4. Instalar Tauri CLI ──────────────────────────────────

Write-Host "[4/7] Instalando Tauri CLI..." -ForegroundColor Yellow
if (Get-Command cargo -ErrorAction SilentlyContinue) {
    cargo install tauri-cli
} else {
    Write-Host "  >> Rust nao encontrado no PATH. Feche e reabra o terminal, depois rode:" -ForegroundColor Red
    Write-Host "     cargo install tauri-cli" -ForegroundColor Red
}

Write-Host ""

# ─── 5. Setup do projeto ────────────────────────────────────

Write-Host "[5/7] Instalando dependencias do frontend (npm)..." -ForegroundColor Yellow
if (Test-Path "package.json") {
    npm install
} else {
    Write-Host "  >> Execute este script dentro da pasta do projeto mangatl/" -ForegroundColor Red
}

Write-Host ""

# ─── 6. Setup Python venv ───────────────────────────────────

Write-Host "[6/7] Configurando Python virtual environment..." -ForegroundColor Yellow
if (Test-Path "pipeline/requirements.txt") {
    Push-Location pipeline
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    pip install --upgrade pip
    pip install -r requirements.txt
    deactivate
    Pop-Location
    Write-Host "  Python venv criado em pipeline/venv/" -ForegroundColor Green
} else {
    Write-Host "  >> pasta pipeline/ nao encontrada" -ForegroundColor Red
}

Write-Host ""

# ─── 7. Verificacao final ───────────────────────────────────

Write-Host "[7/7] Verificacao final..." -ForegroundColor Yellow
Write-Host ""

$checks = @(
    @{ Name = "Node.js"; Cmd = "node -v" },
    @{ Name = "npm"; Cmd = "npm -v" },
    @{ Name = "Rust"; Cmd = "rustc --version" },
    @{ Name = "Cargo"; Cmd = "cargo --version" },
    @{ Name = "Python"; Cmd = "python --version" },
    @{ Name = "Git"; Cmd = "git --version" }
)

$allGood = $true
foreach ($check in $checks) {
    try {
        $result = Invoke-Expression $check.Cmd 2>&1
        Write-Host "  [OK] $($check.Name): $result" -ForegroundColor Green
    } catch {
        Write-Host "  [X]  $($check.Name): NAO ENCONTRADO" -ForegroundColor Red
        $allGood = $false
    }
}

Write-Host ""
if ($allGood) {
    Write-Host "========================================" -ForegroundColor Green
    Write-Host " Tudo pronto! Proximo passo:" -ForegroundColor Green
    Write-Host "   claude" -ForegroundColor White
    Write-Host " (abre o Claude Code nesta pasta)" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
} else {
    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host " Algumas ferramentas faltam." -ForegroundColor Yellow
    Write-Host " Feche o terminal, reabra, e rode" -ForegroundColor Yellow
    Write-Host " este script novamente." -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Yellow
}

Write-Host "== Arquivos rastreados que parecem lixo/dev-only ==" -ForegroundColor Cyan
git ls-files | Select-String -Pattern "target|venv|debug_runs|debug_pipeline_test|exemplos|pk/|dek/|testes|pipeline/teste|pipeline/models|\.log$"

Write-Host "`n== Arquivos grandes rastreados (>10MB) ==" -ForegroundColor Cyan
git ls-files | ForEach-Object {
  if (Test-Path $_) {
    $size = (Get-Item $_).Length
    if ($size -gt 10MB) {
      "{0:N2} MB`t{1}" -f ($size / 1MB), $_
    }
  }
}

Write-Host "`n== Referências antigas de branding ==" -ForegroundColor Cyan
git grep -n "MangáTL\|MangaTL\|mangatl\|MANGATL" -- . ':!docs/plans/*' ':!context.md' 2>$null

Write-Host "`n== Referências ao Lab ==" -ForegroundColor Cyan
git grep -n "Lab\|/lab\|openLabWindow\|startLab\|getLabState\|approveLab\|rejectLab" -- src src-tauri pipeline docs README.md 2>$null

Write-Host "`n== Possíveis arquivos temporários rastreados ==" -ForegroundColor Cyan
git ls-files | Select-String -Pattern "\.tmp$|\.bak$|\.old$|\.log$|debug|backup"

Write-Host "`n== Modelos/binários grandes rastreados ==" -ForegroundColor Cyan
git ls-files | Select-String -Pattern "\.onnx$|\.safetensors$|\.pt$|\.pth$|\.ckpt$"

Write-Host "`nAuditoria concluída." -ForegroundColor Green

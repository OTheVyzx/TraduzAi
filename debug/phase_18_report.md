# Fase 18 - Export Validator e relatorios

Data: 2026-05-01

## Resultado
- Fase 18 concluida.
- `export_project` agora gera pacote de qualidade para `zip_full`.
- ZIP completo inclui `qa_report.md`, `qa_report.json`, `issues.csv`, `glossary_used.json`, `ocr_corrections.json`, `export_manifest.json`, `structured_log.jsonl` e arquivos em `layers/masks/*` quando existem.
- Manifest inclui `run_id`, `created_at`, `status` e SHA-256 dos arquivos exportados.
- Modo `clean` bloqueia flags `critical/high`.
- Modo `with_warnings` permite `high/medium/low`, mas bloqueia `critical`.
- Modo `debug` permite critical e marca `status=blocked_debug_export`.

## Arquivos alterados
- `src-tauri/Cargo.toml`
- `src-tauri/Cargo.lock`
- `src-tauri/src/commands/project.rs`
- `src/lib/tauri.ts`

## Testes e comandos
- `cargo test export_ --lib` passou com 6 testes.
- `cargo check` passou.
- `npm run build` passou.

## Falhas e correcoes
- RED esperado: testes falharam porque `ExportConfig` ainda nao tinha `export_mode`.
- Falha corrigida: o ZIP era criado antes de validar bloqueios; a criacao do arquivo foi movida para depois da validacao para evitar artefato parcial.

## Proximo ponto
Avancar para a Fase 19: Memoria local.

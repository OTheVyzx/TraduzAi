# Fase 5 - Glossary Manager

Data: 2026-05-01

## Resultado
- Fase 5 concluida no nucleo e integrada ao fluxo atual do Setup.
- Criado `Glossary Manager` em Rust com persistencia em `works/<work_id>/glossary.json`.
- Implementadas funcoes: load, save, upsert, remove, find_exact, find_alias, find_fuzzy, extract_hits, validate_translation, create_candidate e export_used_glossary.
- Normalizacao usa Unicode NFKD, case-insensitive quando aplicavel e comparacao accent-insensitive.
- `forbidden` e `protect=true` geram flags de validacao.
- Editor simples de glossario no Setup salva/remove entradas estruturadas quando ha obra ativa.

## Arquivos alterados
- `src-tauri/Cargo.toml`
- `src-tauri/Cargo.lock`
- `src-tauri/src/glossary.rs`
- `src-tauri/src/commands/glossary.rs`
- `src-tauri/src/commands/mod.rs`
- `src-tauri/src/lib.rs`
- `src/lib/tauri.ts`
- `src/pages/Setup.tsx`
- `e2e/editor-rebuild.spec.ts`

## Testes e comandos
- `cargo test glossary --lib` passou com 5 testes filtrados.
- `cargo check` passou.
- `npm run build` passou.
- `npx playwright test --grep "@phase5"` passou com 1 teste.
- `npx playwright test --grep "@smoke|@phase4|@phase5"` passou com 3 testes.

## Falhas e correcoes
- `cargo check` inicialmente acusou warnings de dead code para funcoes obrigatorias ainda nao chamadas pelo fluxo completo. Mantive as funcoes publicas e marquei o modulo com `#![allow(dead_code)]`, porque elas sao parte do contrato da fase e serao chamadas nas fases seguintes.

## Observacoes
- A UI atual cobre adicionar/remover termo; edicao detalhada de campos avancados como aliases, forbidden, tipo e protect ainda usa o contrato backend e pode ser expandida sem mudar o arquivo.
- O tradutor atual ainda recebe o mapa simples `glossario`; a fase entrega o gerenciador estruturado e sincroniza termos revisados para esse mapa no Setup.

## Proximo ponto
Avancar para a Fase 6: Reading Order e Region Grouping.

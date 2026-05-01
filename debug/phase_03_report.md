# Fase 3 - Storage seguro por ambiente

Data: 2026-05-01

## Resultado
- Fase 3 concluida.
- Criado `StorageService` em Rust para resolver paths de dev e producao Tauri.
- Dev usa `data/`, `debug/` e `fixtures/` dentro do repositorio.
- Producao usa `appDataDir()/TraduzAI`.
- Paths centrais expostos: `works`, `memory`, `logs`, `exports`, `models`, `projects`, `settings`, `debug`, `fixtures`.
- Removidos hardcodes `D:\traduzai_data` e `D:/traduzai_data` de `src-tauri/src`.

## Arquivos alterados
- `src-tauri/src/storage.rs`
- `src-tauri/src/lib.rs`
- `src-tauri/src/commands/storage.rs`
- `src-tauri/src/commands/mod.rs`
- `src-tauri/src/commands/settings.rs`
- `src-tauri/src/commands/credits.rs`
- `src-tauri/src/commands/pipeline.rs`
- `src-tauri/src/commands/lab.rs`
- `src/lib/tauri.ts`

## Integracao
- Boot do Tauri cria e valida os diretorios centrais antes do warmup.
- `settings.json` e `credits.json` passaram a usar o storage central.
- Pipeline passou a criar work dirs em `works/<job_id>` e usar `models` do storage.
- Lab passou a resolver sua area persistente sob `memory/lab`.
- Adicionado comando Tauri `get_storage_paths` e binding TypeScript `getStoragePaths()`.

## Testes e comandos
- `cargo test storage --lib` passou com 6 testes.
- `cargo test credits --lib` passou com 4 testes.
- `cargo test settings --lib` passou com 3 testes.
- `cargo check` passou.
- `Select-String -Path src-tauri\src\**\*.rs -Pattern 'traduzai_data|D:\\|D:/'` nao encontrou ocorrencias.
- `npm run build` passou.

## Falhas e correcoes
- Primeiro teste RED falhou porque `StorageService` e `StorageMode` ainda nao existiam. Corrigido com a implementacao do servico.
- Um comando de teste combinado `cargo test credits settings --lib` falhou por sintaxe do Cargo. Corrigido rodando `cargo test credits --lib` e `cargo test settings --lib` separadamente.

## Observacoes
- Playwright nao foi rodado nesta fase porque a alteracao nao mudou tela ou fluxo visual.
- O worktree ja tinha alteracoes fora desta fase; nada foi revertido.

## Proximo ponto
Avancar para a Fase 4: Work Context Profile.

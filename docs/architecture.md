# Arquitetura

## Visao geral

TraduzAI combina frontend React, backend Tauri/Rust e pipeline Python.

```text
React/TypeScript
-> Tauri invoke
-> Rust commands
-> Python sidecar
-> JSON lines/events
-> Zustand/UI
```

## Frontend

- `src/pages`: Home, Setup, Processing, Preview, Editor, Settings e Lab.
- `src/lib/tauri.ts`: bindings IPC.
- `src/lib/stores`: estado global e editor.
- `src/components/editor`: camadas, propriedades e stage.

## Backend

- `src-tauri/src/commands`: comandos expostos ao frontend.
- `src-tauri/src/glossary.rs`: persistencia e validacao de glossario.
- `src-tauri/src/internet_context.rs`: ponte de contexto online.

## Sidecar

O Rust spawna o Python e consome JSON lines de progresso, logs e resultados. O pipeline escreve artefatos de projeto e export em disco local.

## Dados

Dados de projeto devem ficar centralizados em caminhos resolvidos pelo app, sem espalhar paths hardcoded pela UI.

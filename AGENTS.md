# AGENTS.md — Instruções para o Codex

## Sobre o Projeto
TraduzAi é um app desktop Tauri v2 (React + TypeScript frontend, Rust backend, Python sidecar) para tradução automática de mangá/manhwa/manhua EN→PT-BR.

## Stack
- **Frontend:** React 19 + TypeScript + Tailwind CSS + Zustand + Framer Motion
- **Backend:** Rust (Tauri v2 core)
- **Pipeline IA:** Python 3.12 (PaddleOCR, Google Translate + Ollama fallback, OpenCV inpainting, matplotlib FT2Font typesetting)
- **Comunicação:** Tauri IPC (invoke) + Python sidecar via stdout JSON lines

## Arquitetura chave
- O frontend (src/) comunica com o Rust (src-tauri/) via `invoke()`
- O Rust spawna o Python sidecar (pipeline/) como processo filho
- O Python emite JSON lines no stdout que o Rust lê e repassa como eventos Tauri
- Créditos são gerenciados localmente com verificação server-side futura
- Free tier: 2 capítulos/semana (~40 páginas), reseta toda segunda-feira

## Padrões de código
- Frontend: componentes funcionais React, hooks, Zustand para state
- Rust: async/await com tokio, serde para serialização
- Python: type hints, docstrings, lazy imports para startup rápido
- Todas as mensagens de UI em português brasileiro
- Tema escuro (dark mode) é o padrão e único tema

## Caminhos importantes
- `src/lib/tauri.ts` — Todas as bindings Tauri (invoke calls)
- `src/lib/stores/appStore.ts` — State global (projeto, pipeline, créditos)
- `src-tauri/src/commands/` — Commands Rust expostos ao frontend
- `pipeline/main.py` — Entry point do pipeline Python
- `pipeline/translator/translate.py` — Tradução via Google Translate (primário) + Ollama local (fallback)
- `fonts/font-map.json` — Mapeamento tipo de texto → fonte

## Como testar
```bash
npm run tauri dev          # Dev mode completo
cd pipeline && python main.py config.json  # Pipeline isolado
```

## Decisões de design
- Processamento 100% local (tradução via Google Translate público ou Ollama local; contexto via AniList)
- Nenhuma imagem é enviada a servidores — apenas texto extraído
- Sem dependência de APIs pagas no runtime
- Formato project.json aberto e reimportável
- Sem funcionalidade de compartilhamento (segurança legal)
- 1 crédito = 1 página (modelo simples e transparente)

## Comandos rápidos do usuário
- Quando o usuário digitar `cntbk`, isso significa:
  - atualizar o `context.md`
  - criar um novo backup versionado do projeto
  - excluir o backup versionado anterior

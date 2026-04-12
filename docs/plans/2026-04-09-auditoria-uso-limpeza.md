# 2026-04-09 — Auditoria de uso + candidatos de limpeza (TraduzAi)

Este documento inventaria **o que está sendo usado hoje** no projeto e lista **candidatos de remoção/arquivamento** para quando vocês decidirem fazer a limpeza.

> Nota de naming: o README e manifests principais já estão como **TraduzAi**, mas ainda existem referências a **MangáTL / mangatl** em partes do código/config (UI, Tauri identifier, pipeline, nomes internos). A seção **“Checklist de rename (TraduzAi)”** lista os pontos.

## Resumo

- **Status:** rename para **TraduzAi** fechado + limpeza de deps/crates aplicada (mantendo compat com legado).
- **Candidatos claros de remoção (quando forem limpar):**
  - Frontend: `framer-motion`, `clsx`, `@tauri-apps/plugin-shell`, `@tauri-apps/plugin-dialog`
  - Rust: `anyhow`, `tempfile` e possivelmente `tauri-plugin-shell` (confirmar antes)
- **Peso em disco hoje está dominado por artifacts e dados de dev** (`src-tauri/target/`, `vision-worker/target/`, `pipeline/venv/`, `exemplos/`, `pk/`, `debug_*`).

## Dependências — Frontend (React/Tauri)

Fonte: `package.json`.

### Usadas (encontradas via imports em `src/`)

- `react`, `react-dom`, `react-router-dom`
- `zustand`
- `lucide-react`
- `@tauri-apps/api`
- `@tauri-apps/plugin-fs` (há imports diretos de `readFile`)

### Não encontrei uso em `src/` (candidatas a remoção quando decidirem limpar)

- `framer-motion` (nenhum import encontrado em `src/`)
- `clsx` (nenhum import encontrado em `src/`)
- `@tauri-apps/plugin-shell` (nenhum import encontrado em `src/`)
- `@tauri-apps/plugin-dialog` (nenhum import encontrado em `src/`)

Observação: hoje o app abre dialogs via `invoke()` → Rust (`src/lib/tauri.ts` chama `open_source_dialog`, `open_project_dialog`, etc.), então **não precisa** do plugin de dialog do lado JS.

## Dependências — Rust (`src-tauri/`)

Fonte: `src-tauri/Cargo.toml` e uso por leitura de código em `src-tauri/src`.

### Usadas (há referências no código)

- `tauri`, `tauri-plugin-dialog`, `tauri-plugin-fs`
- `serde`, `serde_json`
- `tokio`
- `uuid`
- `reqwest`
- `zip`, `walkdir`
- `once_cell`
- `regex`
- `html-escape`

### Não encontrei uso em `src-tauri/src` (candidatas a remoção quando decidirem limpar)

- `anyhow`
- `tempfile`

### Possível remoção (precisa confirmar antes)

- `tauri-plugin-shell`
  - Está inicializado em `src-tauri/src/lib.rs`, mas não encontrei chamadas às APIs do plugin no backend.
  - Se for removido: também remover `.plugin(tauri_plugin_shell::init())` do builder.

## Dependências — Python (`pipeline/requirements.txt`)

O `pipeline/` usa um conjunto “grande”, mas a maioria aparece de fato em algum módulo do pipeline (OCR, detector, inpainting, typesetter e utilitários).

Mapeamento alto nível (heurístico; não é “garantia formal”):

- OCR/detector:
  - `easyocr`, `paddleocr`, `opencv-python-headless`, `numpy`, `Pillow`
  - `torch` (usado no stack visual / detector e em testes)
  - `ultralytics` (backend do detector)
  - `transformers` (caminho “manga-ocr” opcional no OCR)
  - `torchvision`, `safetensors` (FontDetector)
  - `sentencepiece` (normalmente indireto do `transformers`; não aparece importado direto)
- Inpainting:
  - `simple-lama-inpainting` (vision stack inpainter)
  - `onnxruntime`, `huggingface_hub` (LaMA ONNX + download de artefatos)
- Tradução:
  - `deep-translator` (GoogleTranslator)

Se a meta futura for reduzir deps do Python, a decisão grande é: **manter ou remover o caminho “manga-ocr”** (depende do quanto vocês realmente usam `transformers` no OCR).

## Pastas — o que é “dev-only” vs. “runtime”

### Maiores pastas (ordem aproximada)

- `src-tauri/target/` e `vision-worker/target/`: **artifacts de build (dev-only)** — podem ser apagados e reconstroem.
- `pipeline/venv/`: **ambiente local (dev-only)** — não entra em release/backup.
- `exemplos/` (~5.7GB): **dataset/exemplos (dev-only)**.
- `pk/` (~5.0GB): **cache/snapshots locais** (huggingface/runtime). Em geral **dev-only**, mas pode ser “cache de modelos” útil em máquina de dev.
- `debug_runs/` (~420MB), `debug_pipeline_test/` (~97MB): **saídas e fixtures de debug (dev-only)**.
- `pipeline/models/` (~866MB): modelos locais do pipeline (depende do fluxo: pode ser “runtime/dev-cache”).
- `pipeline/teste/` (~276MB): scripts/experimentos (dev-only).
- `desempacotar .cbz/` (~65MB): ferramenta auxiliar (dev-only).
- `tradutor_mmm/` (~0.7MB), `dek/`, `bug/`: parecem utilitários/rascunhos (dev-only).

### Política sugerida (simples)

- **Nunca entra em release/backup:**
  - `node_modules/`, `dist/`, `pipeline/venv/`, `src-tauri/target/`, `vision-worker/target/`, `debug_runs/`
- **Dev-only (pode manter no repo, mas fora do bundle):**
  - `exemplos/`, `debug_pipeline_test/`, `testes/`, `pipeline/teste/`, `desempacotar .cbz/`, `tradutor_mmm/`, `dek/`, `bug/`
- **Runtime / necessário para build do app:**
  - `src/`, `src-tauri/src/`, `pipeline/` (código), `fonts/`, `docs/`

## Checklist de limpeza (quando vocês forem executar)

### Frontend

1. Remover deps do `package.json`:
   - `framer-motion`, `clsx`, `@tauri-apps/plugin-shell`, `@tauri-apps/plugin-dialog`
2. Rodar `npm install` para atualizar `package-lock.json`
3. Verificar com `npm run build`

### Rust

1. Remover crates do `src-tauri/Cargo.toml`:
   - `anyhow`, `tempfile`
2. (Se confirmado) remover `tauri-plugin-shell` + plugin init
3. Rodar `cd src-tauri && cargo check`

### Limpeza de artifacts (local)

- Apagar (quando precisar liberar disco) e re-buildar depois:
  - `src-tauri/target/`
  - `vision-worker/target/`
  - `pipeline/venv/` (recria via `pip install -r requirements.txt`)

## Checklist de rename (TraduzAi)

O rename pode ser apenas “branding” (UI/README) **ou** um rename completo (identifiers, paths e variáveis).

Status aplicado (rename completo com compat):

- Branding: UI/README/manifests em **TraduzAi**
- Identifier Tauri: `com.traduzai.app`
- Compat com legado (para não quebrar installs/dados antigos):
  - Cache HuggingFace: procura `com.traduzai.app` e `com.mangatl.app`
  - Modelos locais: prioriza `~/.traduzai`, mas usa `~/.mangatl` se existir e o novo ainda não
  - Flags env: aceita `TRADUZAI_*` e `MANGATL_*`
  - Vision-worker: procura `traduzai-vision(.exe)` e cai para `mangatl-vision(.exe)` se necessário
  - Ollama: recomendado `traduzai-translator` (legado: `mangatl-translator`)

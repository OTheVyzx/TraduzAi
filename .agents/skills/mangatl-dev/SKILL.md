---
name: mangatl-dev
description: Use when work in TraduzAI crosses React, Tauri/Rust, Python, project.json, the automatic strip pipeline, or the separate Studio; also when an entrypoint or owner is unclear, IPC or JSON-lines fields disappear, export_gate diverges, or Studio/lasso behavior is mistaken for automatic-pipeline behavior.
---

# TraduzAI: mapa mestre

## Visão geral

Skill arquitetural e roteadora: encontre o caminho executado e entregue o comportamento à especialista dona, sem replicar algoritmos especialistas.

## Fluxos autoritativos

- **App principal:** UI React/stores, incluindo `src/lib/stores/appStore.ts` → `src/lib/tauri.ts` (`invoke`/`listen`) → registro em `src-tauri/src/lib.rs` → owner em `src-tauri/src/commands/` → sidecar `pipeline/main.py`. Python emite JSON lines; Rust publica eventos Tauri consumidos pelo React.
- **Pipeline automático:** `pipeline/main.py` prepara execução e importa `run_chapter` de `pipeline/strip/run.py`, owner da orquestração strip/bandas atual.
- **Studio separado:** `studio/src/App.tsx` → `studio/src/editor/StudioSharedEditor.tsx` → `src/pages/Editor.tsx`. O backend/adapter está em `studio/src/backend/editorBackend.ts`, `editorBackendCompat.ts` e `studio/src/shims/currentEditorBackend.ts`; `studio/src-tauri/src/main.rs` reutiliza `src-tauri/src/commands/studio_lite.rs`/`pipeline/studio_lite`. Não é o pipeline automático.

## Referência rápida de roteamento

| Sintoma ou arquivo inicial | Skill obrigatória |
|---|---|
| caixas, balões, máscaras de detecção, classificação visual | `traduzai-detect` |
| texto reconhecido, ordem de leitura, confiança OCR, `review_reason` | `traduzai-ocr` |
| limpeza, máscara, reinpaint, recovery, artefato visual | `traduzai-inpaint` |
| fonte, estilo, layout, fit, renderização, texto cortado | `traduzai-typesetting` |
| `pipeline/main.py`, `pipeline/strip/run.py`, bandas, progresso, `export_gate`, performance | `traduzai-pipeline` |
| Google/Ollama, glossário, contexto, idioma, texto traduzido | `traduzai-translation` |
| `studio/`, lasso, editor compartilhado, adapter, Studio Lite, placeholder | `traduzai-studio` |

Se um contrato cruza owners, carregue todas as especialistas envolvidas.

## Contratos globais

- **IPC TS/Rust:** alinhe nome do `invoke`, argumentos camelCase/serde, retorno, registro, mocks e consumidores.
- **JSON lines Rust/Python:** stdout é protocolo. Para campo/evento novo, rastreie `emit` Python, parser/emissão Rust e listener/store/UI React, incluindo erro, lote e fallback.
- **`project.json`:** rastreie produtor, allow-lists/schema, aliases e consumidores. Campo novo exige migração/default, hidratação, persistência e round-trip. Confira `text_layers`/`textos`, `style`/`estilo`, `translated`/`traduzido` e caminhos relativos/absolutos.

## Regras de trabalho

1. Antes de editar, identifique entrypoint, caller e owner com `rg`; classifique implementado, fallback, stub/placeholder ou ausente.
2. Diagnóstico/análise não autoriza patch. Preserve worktree sujo e alterações alheias.
3. Valide a camada e interfaces vizinhas. Helper/teste verde não prova resultado visual completo; confira o artefato/runtime quando a aceitação for visual.

## Validação

Localize testes com `rg --files | rg "(test|spec)"`; não invente nomes. Conforme a camada:

```powershell
npm run check
npm test -- <arquivo-ou-filtro-existente>
cargo test --manifest-path src-tauri/Cargo.toml <filtro-existente>
pipeline/venv/Scripts/python.exe -m pytest <arquivo-ou-nodeid-existente>
npm --prefix studio run build
npm --prefix studio test -- <arquivo-ou-filtro-existente>
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/Test-TraduzAISkills.ps1
git diff --check
```

Se o venv não existir, use o Python configurado. Rode build/E2E ou inspeção visual quando o escopo tocar integração/UI.

## Última verificação

Verificado em 2026-07-24 contra o runtime no commit-base `c0801348` (HEAD documental `37930d14`).

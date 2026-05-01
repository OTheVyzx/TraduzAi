# Repo Inventory

Inventario criado para a Fase 0 do plano `TRADUZAI_CORRECAO_TOTAL_V12_SENIOR.md`.
Data da inspecao: 2026-04-30.
Branch atual: `fixpipeline`.

## Stack detectada
- Frontend: React 19, TypeScript 5.7, Vite 6, Tailwind CSS, Zustand, React Router 7, Konva/React-Konva, lucide-react.
- Backend Tauri: Tauri v2 em Rust edition 2021, commands em `src-tauri/src/commands`, plugins `dialog` e `fs`, async via Tokio.
- Pipeline Python: `pipeline/main.py` como entry point do sidecar, Python local via `pipeline/venv/Scripts/python.exe` quando disponivel, comunicacao por JSON lines no stdout.
- OCR atual: `pipeline/vision_stack/ocr.py` e modulos `pipeline/ocr/*`, com PaddleOCR/EasyOCR/fallbacks e pos-processamento em `pipeline/ocr/postprocess.py`.
- Traducao atual: `pipeline/translator/translate.py`, com Google Translate/deep-translator, memoria/glossario/corpus e fallback local quando configurado.
- Inpaint atual: stack visual em `pipeline/vision_stack/inpainter.py`, implementacoes em `pipeline/inpainter/classical.py`, `pipeline/inpainter/lama_onnx.py` e `pipeline/inpainter/mask_builder.py`.
- Typesetting atual: `pipeline/typesetter/*` e layout em `pipeline/layout/balloon_layout.py`; fontes por `fonts/font-map.json`; renderizacao e preview tambem passam por helpers de `pipeline/main.py`.
- Lab/agentes: `lab/runner.py`, `lab/planner.py`, critics/coders e commands Tauri em `src-tauri/src/commands/lab.rs`.

## Scripts reais
- `npm run dev`: executa `node scripts/run-vite.mjs`.
- `npm run check`: executa `tsc --noEmit`.
- `npm run test`: executa `vitest run`.
- `npm run e2e`: executa `playwright test`.
- `npm run e2e:headed`: executa `playwright test --headed`.
- `npm run build`: executa `tsc && node scripts/run-vite.mjs build`.
- `npm run preview`: executa `node scripts/run-vite.mjs preview`.
- `npm run tauri`: executa `node scripts/run-tauri.mjs`; para dev completo o comando real e `npm run tauri -- dev`.
- Testes existentes: Vitest em `src/lib/**/*.test.ts`, `src/pages/*.test.ts`; Playwright em `e2e/editor-rebuild.spec.ts`; pytest em `pipeline/tests`; testes Rust via `cargo test`/`cargo check` em `src-tauri`.
- Playwright config: `playwright.config.ts`, baseURL `http://127.0.0.1:1420`, webServer `npm run dev -- --host 127.0.0.1`, `VITE_E2E=1`.
- Vite config: porta fixa `1420`, `strictPort: true`, ignora `src-tauri`, `vision-worker`, `pipeline`, `debug_runs`, `.venv` e outros diretorios pesados.

## Entradas/saidas atuais
- Input principal do pipeline: arquivo de config JSON passado para `pipeline/main.py`.
- CLI auxiliares do pipeline: `--list-supported-languages`, `--warmup-visual`, `--retypeset`, `--render-preview-page`, `--process-block`, `--detect-page` e comandos manuais semelhantes.
- Output principal: diretorio de trabalho com imagens, `project.json`, `pipeline.log`, `decision_trace.jsonl` e `qa_report.json` quando `utils.decision_log` esta configurado.
- Exemplos atuais de `project.json`: `.test_output_ch82/project.json`, `.test_output_ch82_fix/project.json`, `DPE/traduzido*/project.json`, `NOV/traduzido*/project.json`, `outapp/traduzido*/project.json` e fixture `e2e/fixtures/project-basic.json`.
- Formato atual do editor: `versao`/`app`/`obra`/`capitulo`/`idioma_origem`/`idioma_destino`/`paginas[]`; pagina com `arquivo_original`, `arquivo_traduzido`, `image_layers`, `text_layers` e alias legado `textos`.
- Schema Rust atual: `src-tauri/src/commands/project_schema.rs` normaliza para `PROJECT_VERSION_V2 = "2.0"` e sincroniza aliases legados.
- Logs atuais: `pipeline.log`, `decision_trace.jsonl`, `qa_report.json`, logs soltos em `logs/`, `debug_runs/`, `DEBUGR*` e arquivos de saida de testes.

## Pontos de integracao
- Onde carregar contexto: `pipeline/corpus/runtime.py`, `pipeline/corpus/parallel_dataset.py`, `pipeline/models/corpus/*`, `src/lib/tauri.ts` nos tipos `EnrichedWorkContext` e commands relacionados.
- Onde aplicar glossario: `pipeline/translator/translate.py`, `pipeline/strip/run.py` e metadados propagados em `pipeline/main.py` para `glossary_hits`.
- Onde normalizar OCR: `pipeline/ocr/postprocess.py`, `pipeline/ocr_legacy/postprocess.py`, `pipeline/vision_stack/ocr.py` e testes em `pipeline/tests/test_vision_stack_ocr.py`.
- Onde salvar QA: `pipeline/utils/decision_log.py` gera `decision_trace.jsonl` e `qa_report.json`; `pipeline/main.py` finaliza o trace; UI de processamento/preview le informacoes via `src/lib/tauri.ts` e stores.
- Onde exportar ZIP: backend Rust em `src-tauri/src/commands/project.rs` e modulo `src-tauri/src/export`; export PSD em `src-tauri/src/export/psd`.
- Onde expor Tauri IPC: `src-tauri/src/commands/*`, registro em `src-tauri/src/lib.rs`, bindings frontend em `src/lib/tauri.ts`.
- Onde manter estado global: `src/lib/stores/appStore.ts`, `src/lib/stores/editorStore.ts`, `src/lib/stores/labStore.ts`.
- Onde inserir `data-testid`: paginas `src/pages/*.tsx`, componentes de editor `src/components/editor/*`, UI de QA/processamento em `src/pages/Processing.tsx`/`Preview.tsx` e Lab quando aplicavel.

## Test harness atual
- Playwright ja instalado e configurado.
- Fixture E2E atual: `e2e/fixtures/project-basic.json` com imagens deterministicas.
- Teste E2E atual cobre editor Konva, layers editaveis, visibilidade, lock, criacao de bloco, brush e mascara.
- Vitest configurado por `vitest.config.ts`; memoria previa do projeto indica que rodar Vitest sem config ampla pode coletar arquivos gerados indevidos.
- Pytest esta disponivel dentro de `pipeline/venv/Scripts/pytest.exe`.

## Riscos
- Worktree esta suja e extensa; qualquer fase deve preservar mudancas existentes do usuario e fazer patches pequenos.
- O plano v12 e amplo demais para ser aplicado de uma vez sem gates; deve ser executado por fases com verificacao e status de retomada.
- `rg` falhou com `Acesso negado` neste ambiente; usar `Get-ChildItem` e `Select-String` como fallback.
- Ja existe um schema Rust v2, nao v12; migracao precisa ser incremental para nao quebrar editor, pipeline e fixtures atuais.
- Ja existem Playwright e fixtures; Fase 1 deve adaptar scripts/test IDs ao que existe, nao reinstalar dependencias sem necessidade.
- Existem muitos `project.json` grandes em outputs reais; testes e buscas precisam excluir `pipeline/venv`, `node_modules`, `src-tauri/target` e outputs pesados quando possivel.
- Commands manuais do editor e aliases legados (`textos`, `arquivo_original`, `arquivo_traduzido`) ainda parecem importantes para compatibilidade.
- O pipeline ja gera `decision_trace.jsonl` e `qa_report.json`; novas exigencias de structured logging devem reaproveitar esse contrato em vez de duplicar logs.

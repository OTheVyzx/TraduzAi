# Editor Desktop Para Site Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** colocar no site a mesma superficie do editor desktop do TraduzAi, com a mesma UI e as mesmas ferramentas, preservando o desktop.

**Architecture:** o editor visual passa a depender de um contrato `EditorBackendApi`. O desktop continua usando Tauri; o site configura um backend HTTP/FastAPI antes de montar a mesma UI do desktop. A rota web deixa de usar o editor simplificado em `site/src/App.tsx`.

**Tech Stack:** React 19, Vite, Tailwind, Zustand, Konva/react-konva, FastAPI, pipeline Python existente.

---

## Current Code Facts

- Repo ativo: `N:\TraduzAI\TraduzAi`.
- Branch atual: `feat/editor-brush-mask-typesetting`, com worktree suja. Nao usar reset/revert.
- UI desktop: `src/pages/Editor.tsx`.
- Canvas/interacao: `src/components/editor/stage/EditorStage.tsx` e `src/components/editor/stage/useEditorStageController.ts`.
- Estado editor: `src/lib/stores/editorStore.ts`.
- Dependencia Tauri atual: `editorStore.ts` chama `import("../tauri")`.
- Editor web simplificado: `site/src/App.tsx`, funcao `WebEditor`.
- API HTTP existente: `site/src/editor/editorApi.ts`, `site/src/projectApi.ts`, `server/projects/editor_api.py`, `server/projects/api.py`.
- Gap real no backend: `server/projects/editor_api.py::run_page_action` ainda nao chama a pipeline; `server/projects/api.py::render_preview` ainda copia imagem em vez de renderizar preview fiel.

## Critical Risk Review

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Importar a UI desktop no site carrega Tauri por acidente | Build web quebrado ou chunk web tentando usar API Tauri | Primeiro criar `EditorBackendApi`; site configura backend HTTP antes de montar o editor. |
| Duplicar React/Zustand entre root e site | Hooks quebram, store duplicada, erro de runtime | `site/vite.config.ts` deve usar `resolve.dedupe` para React, React DOM, Zustand, Konva e react-konva. |
| Paths de imagem do `project.json` sao relativos ao workspace | Canvas/thumbnail nao carregam no browser | Normalizar projeto web para URLs `/api/projects/:id/assets/...`; denormalizar antes de salvar `project.json`. |
| Tailwind do site nao enxerga classes do editor desktop | UI abre sem estilos | `site/tailwind.config.js` deve incluir `../src/components/editor`, `../src/pages`, `../src/lib`. |
| Rota web atual envolve `Shell` | Editor aparece dentro do dashboard, diferente da screenshot | Criar `ProtectedFullScreen` e usar na rota `/projects/:id/editor`. |
| Backend HTTP salva schema parcial | Desktop/site divergem e render quebra | Criar helper web de normalizacao e, na fase de servidor, helper de schema para texto/camadas. |
| Pipeline action web simula sucesso | Usuario clica OCR/Inpaint e nada real acontece | Primeira fatia pode montar UI, mas o plano exige fase backend real antes de declarar paridade completa. |
| Worktree suja tem alteracoes de usuario | Perda de trabalho | Editar arquivos com escopo fechado e registrar tudo no status. |

## Implementation Strategy

Fazer em tres cortes testaveis:

1. **Corte A - UI compartilhada no site:** abstrair backend, configurar site para importar o editor desktop e montar rota fullscreen. Esse corte ja muda o site para a mesma UI.
2. **Corte B - Paridade HTTP basica:** texto, visibilidade, bitmap, mascara/lasso, save full e render preview por HTTP sem depender de Tauri.
3. **Corte C - Pipeline real no servidor:** preview fiel e acoes detect/OCR/translate/inpaint/process-block chamando `pipeline/main.py`.

Nao declarar "todas as funcionalidades completas" antes do Corte C.

## File Ownership

- `src/lib/editorBackend.ts`: contrato e registry do backend ativo.
- `src/lib/editorBackends/tauriEditorBackend.ts`: adaptador desktop para comandos Tauri.
- `src/lib/stores/editorStore.ts`: troca chamadas Tauri diretas pelo backend ativo.
- `src/pages/Editor.tsx`: aceita callbacks opcionais para uso no site.
- `site/src/editor/webProjectAdapter.ts`: normaliza/denormaliza projeto para browser.
- `site/src/editor/httpEditorBackend.ts`: backend HTTP que implementa `EditorBackendApi`.
- `site/src/editor/WebEditorRoute.tsx`: bootstrap da rota web com a UI desktop.
- `site/src/App.tsx`: troca rota do editor simplificado para `WebEditorRoute`.
- `site/vite.config.ts`: alias, proxy, fs allow e dedupe.
- `site/tailwind.config.js`: tokens e content do editor desktop.
- `site/package.json` e `site/package-lock.json`: dependencias Konva/Zustand.
- `site/src/projectApi.ts` e `site/src/editor/editorApi.ts`: endpoints HTTP faltantes.
- `server/projects/editor_api.py`: endpoints editor reais.
- `server/projects/api.py`: preview fiel.
- `docs/plans/2026-05-10-editor-desktop-para-site-status.md`: status incremental.

## Corte A - UI Compartilhada no Site

### Task A1: Backend registry

**Files:**
- Create: `src/lib/editorBackend.ts`
- Create: `src/lib/editorBackends/tauriEditorBackend.ts`
- Modify: `src/lib/stores/editorStore.ts`

- [ ] Criar `EditorBackendApi` com a mesma forma dos comandos que a store ja consome.
- [ ] Criar `configureEditorBackend(backend | null)`.
- [ ] Criar `getEditorBackend()`, com fallback lazy para Tauri.
- [ ] Trocar o helper local `getTauriEditorApi()` da store para retornar `getEditorBackend()`.
- [ ] Remover import Tauri direto de `clearMask`.
- [ ] Trocar thumbnail de camada para `loadImageSource`, evitando `@tauri-apps/plugin-fs` direto na store.

**Verification:**
- `npm run check`

### Task A2: Site Vite/Tailwind/deps

**Files:**
- Modify: `site/package.json`
- Modify: `site/package-lock.json`
- Modify: `site/vite.config.ts`
- Modify: `site/tsconfig.json`
- Modify: `site/tailwind.config.js`

- [ ] Adicionar `konva`, `react-konva`, `zustand`.
- [ ] Configurar proxy `/api -> http://127.0.0.1:8787`.
- [ ] Permitir import de `../src` no Vite.
- [ ] Usar `resolve.dedupe`.
- [ ] Copiar tokens Tailwind do desktop para o site.
- [ ] Incluir os arquivos do editor desktop em `content`.

**Verification:**
- `cd site; npm run build`

### Task A3: Projeto web normalizado

**Files:**
- Create: `site/src/editor/webProjectAdapter.ts`
- Modify: `site/src/projectApi.ts`
- Modify: `site/src/editor/editorApi.ts`

- [ ] Criar prefixo `web-project:<projectId>` para preencher `project.output_path`.
- [ ] Converter paths relativos de assets para `/api/projects/:id/assets/...`.
- [ ] Converter URLs `/api/projects/:id/assets/...` de volta para paths relativos antes de salvar.
- [ ] Garantir defaults de `Project`, `PageData`, `TextEntry` e `image_layers`.
- [ ] Expor `projectApi.saveProject`.
- [ ] Expor endpoints editor que faltam para backend HTTP.

**Verification:**
- teste de build do site.

### Task A4: Montar UI desktop na rota web

**Files:**
- Modify: `src/pages/Editor.tsx`
- Create: `site/src/editor/WebEditorRoute.tsx`
- Modify: `site/src/App.tsx`

- [ ] `Editor` aceita `onBack` opcional.
- [ ] Criar `ProtectedFullScreen`.
- [ ] Criar `WebEditorRoute` que configura `httpEditorBackend`, hidrata `useAppStore`, reseta `useEditorStore` e renderiza `<Editor />`.
- [ ] Trocar rota `/projects/:id/editor` para `ProtectedFullScreen + WebEditorRoute`.
- [ ] Manter `WebEditor` antigo sem uso ate o build passar; remover em limpeza posterior.

**Verification:**
- `cd site; npm run build`
- iniciar `npm run saas:server` e `npm run saas:web`.
- abrir rota do editor e validar que o shell/dashboard nao envolve o editor.

## Corte B - Paridade HTTP Basica

### Task B1: Schema HTTP de texto/camadas

**Files:**
- Modify: `server/projects/editor_api.py`
- Optionally create: `server/projects/editor_schema.py`

- [ ] `create_text_layer` deve gerar camada completa com `bbox`, `layout_bbox`, `balloon_bbox`, `estilo`, `visible`, `locked`, `order`.
- [ ] `patch_text_layer` deve espelhar `translated/traduzido` e `style/estilo`.
- [ ] `delete_text_layer` deve manter `textos` e `text_layers` sincronizados.
- [ ] `set_visibility` deve aceitar pagina/tipo/camada e atualizar o `project.json`.

### Task B2: Bitmap/mask HTTP

**Files:**
- Modify: `server/projects/workspace.py`
- Modify: `server/projects/editor_api.py`

- [ ] `write_png_layer` deve aceitar replace/add/subtract quando `png_data` existir.
- [ ] `update_brush`, `update_mask`, `update_recovery` devem atualizar `image_layers` da pagina.
- [ ] Retornar `changed_assets` e asset URL consistente.

**Verification:**
- testes unitarios de API com projeto fixture.

## Corte C - Pipeline Real no Servidor

### Task C1: Runner da pipeline

**Files:**
- Create: `server/projects/pipeline_runner.py`
- Modify: `server/projects/editor_api.py`
- Modify: `server/projects/api.py`

- [ ] Implementar chamada controlada a `pipeline/main.py`.
- [ ] Suportar `--render-preview-page`.
- [ ] Suportar `--detect-page`, `--ocr-page`, `--translate-page`, `--reinpaint-page`.
- [ ] Suportar `--process-block`.
- [ ] Suportar `--region-bbox` e `--external-mask`.
- [ ] Recarregar `project.json` depois da pipeline.

### Task C2: Concorrencia e erros

**Files:**
- Modify: `server/projects/editor_api.py`
- Modify: `server/projects/api.py`

- [ ] Adicionar lock por `project_id/page_index`.
- [ ] Retornar erro 409 quando pagina estiver ocupada.
- [ ] Retornar stderr/stdout resumido quando pipeline falhar.

**Verification:**
- rodar uma acao real no site: Detectar, OCR, Traduzir, Inpaint e Preview.

## Final Verification Gates

Antes de declarar pronto:

- [ ] `npm run check`
- [ ] `cd site; npm run build`
- [ ] testes focados de servidor/editor API
- [ ] browser em `/projects/:id/editor`
- [ ] screenshot desktop-like comparada com a referencia
- [ ] uma acao real da pipeline funcionando pelo site
- [ ] status atualizado com comandos, resultados e gaps

## Status During Execution

Atualizar `docs/plans/2026-05-10-editor-desktop-para-site-status.md` apos cada corte com:

- arquivos alterados
- comandos executados
- resultado dos testes
- risco remanescente
- proximo passo concreto

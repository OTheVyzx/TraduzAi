# Editor Desktop Para Site - Status

## 2026-05-10

### Escopo em execucao

- Corte A do plano `docs/plans/2026-05-10-editor-desktop-para-site.md`.
- Objetivo imediato: montar a UI do editor desktop no site com backend HTTP configuravel, sem quebrar o desktop.

### Estado inicial

- Repo: `N:\TraduzAI\TraduzAi`.
- Branch: `feat/editor-brush-mask-typesetting`.
- Worktree ja estava suja antes desta implementacao. Alteracoes geradas/runtime e mudancas de usuario devem ser preservadas.

### Arquivos planejados neste corte

- `src/lib/editorBackend.ts`
- `src/lib/editorBackends/tauriEditorBackend.ts`
- `src/lib/stores/editorStore.ts`
- `src/pages/Editor.tsx`
- `site/src/editor/webProjectAdapter.ts`
- `site/src/editor/httpEditorBackend.ts`
- `site/src/editor/WebEditorRoute.tsx`
- `site/src/App.tsx`
- `site/src/projectApi.ts`
- `site/src/editor/editorApi.ts`
- `site/vite.config.ts`
- `site/tsconfig.json`
- `site/tailwind.config.js`
- `site/package.json`
- `site/package-lock.json`

### Implementado

- Plano revisado com matriz de riscos em `docs/plans/2026-05-10-editor-desktop-para-site.md`.
- `EditorBackendApi` e registry configuravel criados.
- Store do editor passou a usar backend ativo em vez de importar Tauri diretamente.
- Backend Tauri mantido como fallback default do desktop.
- Backend HTTP criado para o site.
- Adaptador web de projeto criado para normalizar assets relativos como URLs `/api/projects/:id/assets/...`.
- Rota `WebEditorRoute` criada usando a UI desktop `Editor`.
- `Editor` aceita `onBack` opcional para funcionar dentro do site.
- Rota `/projects/:id/editor` agora usa fullscreen sem `Shell`.
- Site configurado com Konva, react-konva, Zustand, Tailwind tokens do desktop, proxy `/api` e dedupe de React/Zustand/Konva.
- Fontes do editor copiadas para `site/public/fonts`.
- API web ajustada para default same-origin, evitando CORS fora da porta 5174.
- Servidor recebeu runner de pipeline para preview e acoes por pagina.
- Endpoints basicos de editor no servidor melhorados para schema de texto/camadas, visibilidade e bitmap.

### Verificacao executada

- `npm run check` passou.
- `cd site; npm run build` passou.
- `python -m compileall server\projects` passou.
- API local respondeu em `http://127.0.0.1:8787/api/health`.
- Site local existente respondeu em `http://127.0.0.1:5174`.
- Site de verificacao iniciado em `http://127.0.0.1:5175` com Vite recarregando Tailwind config.
- Projeto fixture real importado: `d32fd148-2ced-4ba1-8ee0-1fab4a288b76`.
- Playwright abriu `http://127.0.0.1:5175/projects/d32fd148-2ced-4ba1-8ee0-1fab4a288b76/editor`.
- Screenshot salva em `test-results/site-editor-real-5175.png`.
- Smoke de canvas: tecla `T` + drag criou nova caixa de texto e a UI mudou para `2/2`.

### Riscos remanescentes

- O botao Salvar precisa de um teste dedicado sem ambiguidade de seletor; uma tentativa manual via Playwright nao confirmou persistencia no `project.json`.
- A pipeline real agora e chamada pelo servidor, mas detect/OCR/translate/inpaint ainda precisam de smoke dedicado por acao em fixture controlado.
- O build do site ainda inclui chunks lazy de Tauri porque o fallback desktop fica disponivel; isso nao quebrou runtime, mas pode ser otimizado depois com split/alias especifico.
- O bundle principal do site ficou acima de 500 kB; e um aviso de performance, nao erro funcional.

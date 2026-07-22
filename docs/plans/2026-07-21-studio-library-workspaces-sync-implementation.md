# Studio Library, Workspaces, and Work Tracking Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transformar o TraduzAI Studio em um aplicativo mestre de organização, tradução manual e edição de capítulos, com biblioteca `obra -> capítulos`, áreas de trabalho alternáveis e acompanhamento opcional por AniList/MangaDex.

**Architecture:** O Studio continua sendo um aplicativo Tauri independente. O React mantém estado de catálogo, documento e área de trabalho em stores separadas; Rust persiste o catálogo, prepara importações locais e acessa provedores externos. O `project.json` permanece como documento editável e compatível; `studio-library.json` guarda apenas organização, referências e cache de metadados.

**Tech Stack:** React 19, TypeScript, Zustand, Vitest, Tailwind CSS, Tauri v2, Rust, serde, reqwest, zip 2, image 0.25.

---

## Regras de execução

- Trabalhar apenas em `studio/`, nos pontos compartilhados do editor explicitamente citados e na documentação deste plano.
- Não modificar o pipeline automático em `pipeline/` nem incorporar FLUX nesta entrega.
- Antes de cada tarefa, executar `git status --short` e preservar todas as alterações preexistentes.
- Usar TDD: teste falhando, implementação mínima, teste passando e refatoração.
- Não fazer chamadas AniList/MangaDex diretamente do React; os provedores pertencem ao backend Rust do Studio.
- Não baixar páginas de capítulos. Dados externos são somente identidade, status e disponibilidade.
- Não versionar arquivos de catálogo do usuário, projetos importados ou imagens de teste produzidas em runtime.
- Em cada commit, adicionar somente os arquivos listados na tarefa.

## Resultado por marco

1. **Marco A — Biblioteca utilizável:** obras e capítulos persistentes, abertura de projetos existentes e recuperação de caminhos ausentes.
2. **Marco B — Produção manual:** criação de capítulo por pasta/ZIP/CBZ e alternância `Tradução | Edição` sobre o mesmo documento.
3. **Marco C — Acompanhamento:** vínculo opcional com AniList/MangaDex, cache offline, atualização e alertas.
4. **Marco D — Robustez:** verificação completa, documentação, acessibilidade e empacotamento.

## Task 1: Criar o domínio puro do catálogo

**Files:**

- Create: `studio/src/library/libraryModel.ts`
- Create: `studio/src/library/__tests__/libraryModel.test.ts`

**Step 1: Write the failing test**

Cobrir criação, migração, ordenação natural de capítulos, deduplicação por caminho normalizado e cálculo de progresso:

```ts
import { describe, expect, it } from "vitest";
import {
  createEmptyLibrary,
  normalizeLibrary,
  sortChapterEntries,
  upsertChapter,
} from "../libraryModel";

describe("libraryModel", () => {
  it("migra um documento vazio para a versão atual", () => {
    expect(normalizeLibrary({}).schemaVersion).toBe(1);
  });

  it("ordena capítulos numericamente e mantém especiais no fim", () => {
    const values = ["10", "2", "2.5", "Extra"].map((label) => ({ label }));
    expect(sortChapterEntries(values).map((item) => item.label)).toEqual(["2", "2.5", "10", "Extra"]);
  });

  it("não duplica o mesmo project.json", () => {
    const initial = createEmptyLibrary();
    const once = upsertChapter(initial, "work-1", { id: "a", label: "1", projectPath: "C:/obra/project.json" });
    const twice = upsertChapter(once, "work-1", { id: "b", label: "1", projectPath: "C:\\obra\\project.json" });
    expect(twice.works[0].chapters).toHaveLength(1);
  });
});
```

**Step 2: Run test to verify it fails**

Run: `npm --prefix studio test -- src/library/__tests__/libraryModel.test.ts`

Expected: FAIL porque `libraryModel.ts` ainda não existe.

**Step 3: Write minimal implementation**

Definir o contrato inicial:

```ts
export type PublicationStatus =
  | "releasing"
  | "hiatus"
  | "completed"
  | "cancelled"
  | "not_yet_released"
  | "unknown";

export type ChapterWorkflowStatus = "pending" | "translating" | "editing" | "review" | "completed";

export interface LibraryChapter {
  id: string;
  label: string;
  title?: string;
  projectPath: string;
  coverPath?: string | null;
  pageCount?: number;
  completedPages?: number;
  workflowStatus?: ChapterWorkflowStatus;
  lastOpenedAt?: string | null;
}

export interface ExternalWorkLink {
  anilistId?: number;
  mangaDexId?: string;
  canonicalUrl?: string;
  manualStatusOverride?: PublicationStatus | null;
}

export interface LibraryWork {
  id: string;
  title: string;
  aliases: string[];
  coverPath?: string | null;
  publicationStatus: PublicationStatus;
  external: ExternalWorkLink;
  chapters: LibraryChapter[];
}

export interface StudioLibrary {
  schemaVersion: 1;
  selectedWorkId: string | null;
  works: LibraryWork[];
  preferences: { chapterView: "grid" | "list"; thumbnailSize: number };
}
```

Implementar funções imutáveis, normalização defensiva e comparação natural. Caminho deve ser comparado sem diferenciar barras ou caixa no Windows, mas preservado como o usuário o escolheu.

**Step 4: Run test to verify it passes**

Run: `npm --prefix studio test -- src/library/__tests__/libraryModel.test.ts`

Expected: PASS.

**Step 5: Commit**

```powershell
git add studio/src/library/libraryModel.ts studio/src/library/__tests__/libraryModel.test.ts
git commit -m "feat(studio): add library domain model"
```

## Task 2: Persistir `studio-library.json` de forma atômica no Rust

**Files:**

- Create: `studio/src-tauri/src/library.rs`
- Modify: `studio/src-tauri/src/main.rs`

**Step 1: Write the failing Rust tests**

Em `library.rs`, testar arquivo inexistente, round trip e recuperação pelo `.bak` quando o JSON principal estiver inválido:

```rust
#[test]
fn loads_backup_when_primary_is_corrupt() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("studio-library.json");
    std::fs::write(&path, "{").unwrap();
    std::fs::write(path.with_extension("json.bak"), r#"{"schemaVersion":1,"works":[]}"#).unwrap();

    let loaded = load_library_from_path(&path).unwrap();
    assert!(loaded.recovered_from_backup);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --manifest-path studio/src-tauri/Cargo.toml library::tests`

Expected: FAIL porque o módulo ainda não existe.

**Step 3: Implement persistence and Tauri commands**

Criar funções internas testáveis:

```rust
pub(crate) fn load_library_from_path(path: &Path) -> Result<LibraryLoadResult, String>;
pub(crate) fn save_library_to_path(path: &Path, document: &Value) -> Result<(), String>;

#[tauri::command]
pub(crate) fn studio_load_library(app: tauri::AppHandle) -> Result<LibraryLoadResult, String>;

#[tauri::command]
pub(crate) fn studio_save_library(app: tauri::AppHandle, document: Value) -> Result<(), String>;
```

Usar `app.path().app_data_dir()`, criar o diretório se necessário e escrever `tmp -> fsync -> rename`, preservando o último documento válido em `.bak`. Registrar os dois comandos no `invoke_handler!` de `main.rs`.

**Step 4: Run tests**

Run: `cargo test --manifest-path studio/src-tauri/Cargo.toml library::tests`

Expected: PASS para criação, round trip e fallback.

**Step 5: Commit**

```powershell
git add studio/src-tauri/src/library.rs studio/src-tauri/src/main.rs
git commit -m "feat(studio): persist project library"
```

## Task 3: Ligar o catálogo ao frontend por backend e store

**Files:**

- Create: `studio/src/library/libraryBackend.ts`
- Create: `studio/src/store/libraryStore.ts`
- Create: `studio/src/store/__tests__/libraryStore.test.ts`

**Step 1: Write the failing store tests**

Usar um backend falso injetável para confirmar carregamento, seleção e gravação serializada:

```ts
it("persiste a obra selecionada", async () => {
  const backend = createFakeLibraryBackend(createEmptyLibrary());
  const store = createLibraryStore(backend);
  await store.getState().load();
  await store.getState().addWork({ id: "w1", title: "Obra", aliases: [] });
  await store.getState().selectWork("w1");
  expect(backend.lastSaved?.selectedWorkId).toBe("w1");
});
```

**Step 2: Run test to verify it fails**

Run: `npm --prefix studio test -- src/store/__tests__/libraryStore.test.ts`

Expected: FAIL por módulos ausentes.

**Step 3: Implement backend and store**

O backend deve expor somente:

```ts
export interface LibraryBackend {
  load(): Promise<StudioLibrary>;
  save(document: StudioLibrary): Promise<void>;
}
```

No runtime Tauri, usar `invoke("studio_load_library")` e `invoke("studio_save_library")`. Em testes/browser sem Tauri, usar memória; não reutilizar `localStorage` como fonte autoritativa. A store deve manter `status`, `error`, `document`, ações imutáveis e uma fila de salvamento para impedir que uma gravação antiga sobreponha uma nova.

**Step 4: Run tests**

Run: `npm --prefix studio test -- src/store/__tests__/libraryStore.test.ts`

Expected: PASS.

**Step 5: Commit**

```powershell
git add studio/src/library/libraryBackend.ts studio/src/store/libraryStore.ts studio/src/store/__tests__/libraryStore.test.ts
git commit -m "feat(studio): add library frontend store"
```

## Task 4: Substituir a home plana pela biblioteca de obras e capítulos

**Files:**

- Create: `studio/src/library/StudioLibraryHome.tsx`
- Create: `studio/src/library/WorkLibrarySidebar.tsx`
- Create: `studio/src/library/ChapterBrowser.tsx`
- Create: `studio/src/library/LibraryToolbar.tsx`
- Modify: `studio/src/App.tsx`
- Modify: `studio/src/index.css`
- Modify: `studio/src/__tests__/StudioHome.test.ts`

**Step 1: Replace the old home expectations with failing behavior tests**

Cobrir os elementos essenciais da referência:

```ts
expect(screen.getByRole("heading", { name: "Obras" })).toBeInTheDocument();
expect(screen.getByRole("button", { name: "Adicionar obra" })).toBeInTheDocument();
expect(screen.getByRole("heading", { name: "Capítulos" })).toBeInTheDocument();
expect(screen.getByRole("button", { name: "Adicionar capítulo" })).toBeDisabled();
```

Adicionar teste de seleção de obra, grade/lista, tamanho de miniatura, busca e abertura de card. Manter o teste de recuperação/abertura do projeto atual.

**Step 2: Run test to verify it fails**

Run: `npm --prefix studio test -- src/__tests__/StudioHome.test.ts`

Expected: FAIL porque a home ainda exibe recentes.

**Step 3: Implement the home shell**

- Barra esquerda fixa para obras com busca, status e contador local/remoto.
- Área principal com toolbar, cards/lista de capítulos e estado vazio acionável.
- Rodapé com `Importar projeto`, `Adicionar capítulo` e `Abrir`.
- `Adicionar obra` cria uma entrada manual mínima; edição detalhada vem na próxima tarefa.
- Ao abrir um `project.json` pela ação existente, registrar/atualizar obra e capítulo no catálogo e então chamar `projectStore.openProject`.
- Migrar os itens de `RECENTS_KEY` uma única vez para uma obra `Projetos importados`, sem apagar a chave até a gravação do catálogo ser confirmada.
- Reusar tokens, ícones Lucide e superfícies já existentes no Studio; não copiar assets do DaVinci.

**Step 4: Run focused tests and build**

Run: `npm --prefix studio test -- src/__tests__/StudioHome.test.ts`

Expected: PASS.

Run: `npm --prefix studio run build`

Expected: typecheck e build PASS; o aviso atual de chunk grande não bloqueia este marco.

**Step 5: Commit**

```powershell
git add studio/src/library/StudioLibraryHome.tsx studio/src/library/WorkLibrarySidebar.tsx studio/src/library/ChapterBrowser.tsx studio/src/library/LibraryToolbar.tsx studio/src/App.tsx studio/src/index.css studio/src/__tests__/StudioHome.test.ts
git commit -m "feat(studio): add work and chapter library home"
```

## Task 5: Implementar criação, edição e relink de obras/capítulos

**Files:**

- Create: `studio/src/library/WorkDialog.tsx`
- Create: `studio/src/library/AttachProjectDialog.tsx`
- Create: `studio/src/library/__tests__/libraryDialogs.test.tsx`
- Modify: `studio/src/library/StudioLibraryHome.tsx`
- Modify: `studio/src/store/libraryStore.ts`
- Modify: `studio/src/store/__tests__/libraryStore.test.ts`
- Modify: `studio/src/backend/projectDialog.ts`

**Step 1: Write failing interaction tests**

Testar validação de título, renomear sem perder capítulos, anexar `project.json`, detectar duplicata, remover somente a referência e relinkar caminho ausente.

**Step 2: Run tests**

Run: `npm --prefix studio test -- src/library/__tests__/libraryDialogs.test.tsx src/store/__tests__/libraryStore.test.ts`

Expected: FAIL.

**Step 3: Implement dialogs and actions**

- `WorkDialog`: título, aliases, capa local opcional, status manual e campos externos inicialmente vazios.
- `AttachProjectDialog`: abre o seletor JSON existente, lê o projeto pelo backend atual, pré-preenche obra/capítulo e pede confirmação.
- Remover obra/capítulo deve declarar claramente que só remove do catálogo, nunca apaga arquivos.
- Se `projectPath` não existir, o card fica em estado `Caminho ausente` com ação `Relocalizar`.
- Caminhos não podem ser editados como texto livre; usar seletor do sistema.

**Step 4: Run focused tests**

Run: `npm --prefix studio test -- src/library/__tests__/libraryDialogs.test.tsx src/store/__tests__/libraryStore.test.ts`

Expected: PASS.

**Step 5: Commit**

```powershell
git add studio/src/library/WorkDialog.tsx studio/src/library/AttachProjectDialog.tsx studio/src/library/__tests__/libraryDialogs.test.tsx studio/src/library/StudioLibraryHome.tsx studio/src/store/libraryStore.ts studio/src/store/__tests__/libraryStore.test.ts studio/src/backend/projectDialog.ts
git commit -m "feat(studio): manage works and attached chapters"
```

## Task 6: Preparar importação segura de pasta, ZIP e CBZ

**Files:**

- Modify: `studio/src-tauri/Cargo.toml`
- Create: `studio/src-tauri/src/chapter_import.rs`
- Modify: `studio/src-tauri/src/main.rs`

**Step 1: Add failing Rust tests**

Cobrir:

- ordenação natural `1, 2, 10`;
- extensões PNG/JPEG/WebP, ignorando arquivos não-imagem;
- rejeição de `../` e caminhos absolutos em ZIP/CBZ;
- limite de quantidade, tamanho individual e tamanho total descompactado;
- limpeza do staging após falha;
- leitura de largura/altura sem decodificar a imagem inteira quando suportado.

**Step 2: Run tests to verify failure**

Run: `cargo test --manifest-path studio/src-tauri/Cargo.toml chapter_import::tests`

Expected: FAIL porque o módulo não existe.

**Step 3: Add dependencies and implementation**

Adicionar as mesmas versões já usadas no Tauri principal:

```toml
zip = "2"
image = { version = "0.25", default-features = false, features = ["png", "jpeg", "webp"] }
```

Expor contratos:

```rust
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ImportedPage {
    pub number: u32,
    pub relative_path: String,
    pub width: u32,
    pub height: u32,
}

#[tauri::command]
pub(crate) async fn studio_prepare_manual_chapter(
    source_path: String,
    project_json_path: String,
) -> Result<Vec<ImportedPage>, String>;
```

Copiar/extrair para `<diretório-do-project>/original/` via diretório temporário irmão e promover apenas após validar tudo. Nunca seguir symlink da origem para fora da pasta escolhida. Se já houver destino, interromper sem sobrescrever silenciosamente.

**Step 4: Run Rust tests and checks**

Run: `cargo test --manifest-path studio/src-tauri/Cargo.toml chapter_import::tests`

Expected: PASS.

Run: `cargo check --manifest-path studio/src-tauri/Cargo.toml`

Expected: PASS.

**Step 5: Commit**

```powershell
git add studio/src-tauri/Cargo.toml studio/src-tauri/Cargo.lock studio/src-tauri/src/chapter_import.rs studio/src-tauri/src/main.rs
git commit -m "feat(studio): prepare safe manual chapter imports"
```

## Task 7: Criar o `project.json` manual a partir das imagens

**Files:**

- Create: `studio/src/project/createManualProject.ts`
- Create: `studio/src/project/__tests__/createManualProject.test.ts`
- Create: `studio/src/library/CreateChapterDialog.tsx`
- Modify: `studio/src/backend/projectDialog.ts`
- Modify: `studio/src/library/StudioLibraryHome.tsx`
- Modify: `studio/src/store/libraryStore.ts`

**Step 1: Write the failing project factory test**

```ts
it("cria páginas compatíveis com base raster e cena normalizada", () => {
  const project = createManualProject({
    workTitle: "Obra",
    chapterLabel: "12",
    sourceLanguage: "en",
    targetLanguage: "pt-BR",
    pages: [{ number: 1, relativePath: "original/001.webp", width: 800, height: 1200 }],
  });
  expect(project.paginas[0].image_layers.base?.path).toBe("original/001.webp");
  expect(project.paginas[0].text_layers).toEqual([]);
  expect(project.paginas[0].studio_scene.roots.length).toBeGreaterThan(0);
});
```

**Step 2: Run tests to verify failure**

Run: `npm --prefix studio test -- src/project/__tests__/createManualProject.test.ts`

Expected: FAIL.

**Step 3: Implement the factory and dialog flow**

Fluxo do diálogo:

1. escolher pasta/ZIP/CBZ;
2. informar rótulo/título e idiomas;
3. escolher onde salvar `project.json`;
4. invocar `studio_prepare_manual_chapter`;
5. criar o documento pela factory e normalizá-lo pelo adaptador existente;
6. salvar pelo backend atual;
7. cadastrar o capítulo e abri-lo.

Em caso de falha depois da preparação, oferecer `Tentar novamente` e manter diagnóstico suficiente para remover o staging; não criar card apontando para projeto incompleto.

**Step 4: Run focused tests and build**

Run: `npm --prefix studio test -- src/project/__tests__/createManualProject.test.ts src/__tests__/StudioHome.test.ts`

Expected: PASS.

Run: `npm --prefix studio run build`

Expected: PASS.

**Step 5: Commit**

```powershell
git add studio/src/project/createManualProject.ts studio/src/project/__tests__/createManualProject.test.ts studio/src/library/CreateChapterDialog.tsx studio/src/backend/projectDialog.ts studio/src/library/StudioLibraryHome.tsx studio/src/store/libraryStore.ts
git commit -m "feat(studio): create manual chapters from images"
```

## Task 8: Criar o shell de áreas de trabalho sem recarregar o capítulo

**Files:**

- Create: `studio/src/editor/studioWorkspace.ts`
- Create: `studio/src/editor/StudioWorkspaceShell.tsx`
- Create: `studio/src/editor/__tests__/StudioWorkspaceShell.test.tsx`
- Modify: `studio/src/editor/StudioSharedEditor.tsx`
- Modify: `studio/src/App.tsx`
- Modify: `studio/src/store/projectStore.ts`
- Modify: `studio/src/store/__tests__/projectStore.test.ts`
- Modify: `src/pages/Editor.tsx`

**Step 1: Write failing workspace and lifecycle tests**

Cobrir:

- padrão `Edição` para projeto já traduzido;
- padrão `Tradução` para capítulo manual vazio;
- alternância mantém `currentPageNumber` e seleção;
- botão voltar fecha o editor e retorna à mesma obra/capítulo;
- alterações sujas exigem confirmação antes de fechar.

**Step 2: Run focused tests**

Run: `npm --prefix studio test -- src/editor/__tests__/StudioWorkspaceShell.test.tsx src/store/__tests__/projectStore.test.ts`

Expected: FAIL.

**Step 3: Implement the shell and a narrow shared-editor slot**

Definir:

```ts
export type StudioWorkspace = "translation" | "editing";
```

`StudioWorkspaceShell` é dono apenas da escolha de workspace e navegação. O documento continua na mesma `projectStore`. Adicionar a `EditorProps` um slot opcional e neutro:

```ts
workspaceSwitcher?: React.ReactNode;
```

Renderizá-lo no canto superior direito antes das ações existentes. Não adicionar regras de biblioteca ao editor compartilhado. Implementar `closeProject()` explícito na store e conectar o `onBack` hoje vazio em `StudioSharedEditor.tsx`.

**Step 4: Run tests and build both frontends affected by the shared file**

Run: `npm --prefix studio test -- src/editor/__tests__/StudioWorkspaceShell.test.tsx src/store/__tests__/projectStore.test.ts`

Expected: PASS.

Run: `npm --prefix studio run build`

Expected: PASS.

Run: `npm run build`

Expected: PASS no app principal, pois `src/pages/Editor.tsx` é compartilhado.

**Step 5: Commit**

```powershell
git add studio/src/editor/studioWorkspace.ts studio/src/editor/StudioWorkspaceShell.tsx studio/src/editor/__tests__/StudioWorkspaceShell.test.tsx studio/src/editor/StudioSharedEditor.tsx studio/src/App.tsx studio/src/store/projectStore.ts studio/src/store/__tests__/projectStore.test.ts src/pages/Editor.tsx
git commit -m "feat(studio): add translation and editing workspaces"
```

## Task 9: Estender o documento com estado de tradução manual compatível

**Files:**

- Modify: `studio/src/project/studioProject.ts`
- Modify: `studio/schemas/studio_project.schema.json`
- Modify: `studio/src/project/adapters.ts`
- Modify: `studio/src/project/__tests__/adapters.test.ts`
- Create: `studio/src/translation/translationQueue.ts`
- Create: `studio/src/translation/__tests__/translationQueue.test.ts`

**Step 1: Write failing compatibility tests**

Casos:

- projeto antigo sem estado importa como `pending` se a tradução estiver vazia e `translated` se preenchida;
- exportação preserva aliases atuais `translated`/`traduzido`;
- notas e status sobrevivem ao round trip;
- progresso do capítulo usa blocos, mas página vazia não gera divisão por zero.

**Step 2: Run tests to verify failure**

Run: `npm --prefix studio test -- src/project/__tests__/adapters.test.ts src/translation/__tests__/translationQueue.test.ts`

Expected: FAIL.

**Step 3: Implement additive fields**

```ts
export type TranslationStatus = "pending" | "translated" | "review" | "approved";

export interface StudioTextLayer {
  // campos existentes...
  translation_status?: TranslationStatus;
  translation_notes?: string;
}
```

Adicionar somente propriedades opcionais no schema. `translationQueue.ts` deve derivar filas/filtros/progresso sem mutar o projeto. A fonte de verdade continua sendo cada `StudioTextLayer`, não um banco paralelo.

**Step 4: Run tests**

Run: `npm --prefix studio test -- src/project/__tests__/adapters.test.ts src/translation/__tests__/translationQueue.test.ts`

Expected: PASS.

**Step 5: Commit**

```powershell
git add studio/src/project/studioProject.ts studio/schemas/studio_project.schema.json studio/src/project/adapters.ts studio/src/project/__tests__/adapters.test.ts studio/src/translation/translationQueue.ts studio/src/translation/__tests__/translationQueue.test.ts
git commit -m "feat(studio): track manual translation status"
```

## Task 10: Implementar a área de trabalho Tradução

**Files:**

- Create: `studio/src/translation/StudioTranslationWorkspace.tsx`
- Create: `studio/src/translation/TranslationQueuePanel.tsx`
- Create: `studio/src/translation/TranslationInspector.tsx`
- Create: `studio/src/translation/GlossaryPanel.tsx`
- Create: `studio/src/translation/__tests__/StudioTranslationWorkspace.test.tsx`
- Modify: `studio/src/editor/StudioWorkspaceShell.tsx`
- Modify: `studio/src/editor/StudioSharedEditor.tsx`
- Modify: `src/pages/Editor.tsx`
- Modify: `src/components/editor/editorMode.ts`

**Step 1: Write failing user-flow tests**

Testar o fluxo principal:

1. selecionar página pendente;
2. selecionar um bloco no canvas;
3. editar a tradução sem alterar o original;
4. marcar como `Revisão`;
5. ir automaticamente ao próximo bloco pendente;
6. alternar `Original | Limpa | Traduzida`;
7. voltar para `Edição` mantendo o bloco e a página.

Adicionar testes de teclado: `Ctrl+Enter` confirma e avança; `Alt+Up/Down` muda o bloco; foco em textarea não dispara ferramenta do canvas.

**Step 2: Run tests to verify failure**

Run: `npm --prefix studio test -- src/translation/__tests__/StudioTranslationWorkspace.test.tsx`

Expected: FAIL.

**Step 3: Implement by composing the existing editor**

- Reusar `Editor`, `EditorStage`, `PageThumbnails` e os métodos atuais da `projectStore`.
- Usar o slot `layersPanel` para o painel direito de tradução; não duplicar canvas nem store.
- Painel direito: original somente leitura, tradução editável, tipo, notas, status, glossário/contexto local.
- Fila/páginas: filtros `Todos`, `Pendentes`, `Revisão`, `Aprovados`, progresso e indicadores por página.
- Barra de visualização: `Original`, `Limpa`, `Traduzida` usando camadas já disponíveis; desabilitar `Limpa` com explicação quando não houver camada de inpaint.
- Em workspace `translation`, limitar a barra principal a seleção/criação de texto, zoom, pan e visualização. O workspace `editing` conserva todas as ferramentas.
- Glossário v1 é local ao projeto em `work_context.glossary`; não adicionar tradução automática.

**Step 4: Run tests and visual smoke**

Run: `npm --prefix studio test -- src/translation/__tests__/StudioTranslationWorkspace.test.tsx`

Expected: PASS.

Run: `npm --prefix studio run build`

Expected: PASS.

Abrir `npm --prefix studio run tauri:dev` e verificar manualmente em 1440x900: seletor no canto superior direito, nenhum corte nos painéis, textarea utilizável, troca de página e retorno à biblioteca.

**Step 5: Commit**

```powershell
git add studio/src/translation/StudioTranslationWorkspace.tsx studio/src/translation/TranslationQueuePanel.tsx studio/src/translation/TranslationInspector.tsx studio/src/translation/GlossaryPanel.tsx studio/src/translation/__tests__/StudioTranslationWorkspace.test.tsx studio/src/editor/StudioWorkspaceShell.tsx studio/src/editor/StudioSharedEditor.tsx src/pages/Editor.tsx src/components/editor/editorMode.ts
git commit -m "feat(studio): add manual translation workspace"
```

## Task 11: Criar contratos de acompanhamento e integrar AniList

**Files:**

- Create: `studio/src/tracking/workTracking.ts`
- Create: `studio/src/tracking/__tests__/workTracking.test.ts`
- Create: `studio/src-tauri/src/work_tracking.rs`
- Create: `studio/src-tauri/src/fixtures/anilist_media.json`
- Modify: `studio/src-tauri/src/main.rs`

**Step 1: Write failing mapping and fixture tests**

Frontend deve mapear `RELEASING`, `HIATUS`, `FINISHED`, `CANCELLED`, `NOT_YET_RELEASED` para os valores internos. Rust deve testar parse de fixture e distinguir erro de rede, limite e resposta inválida.

**Step 2: Run tests to verify failure**

Run: `npm --prefix studio test -- src/tracking/__tests__/workTracking.test.ts`

Expected: FAIL.

Run: `cargo test --manifest-path studio/src-tauri/Cargo.toml work_tracking::tests`

Expected: FAIL.

**Step 3: Implement provider contract and AniList adapter**

Contrato retornado pelo Rust:

```rust
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct WorkTrackingSnapshot {
    pub provider: String,
    pub provider_id: String,
    pub title: String,
    pub status: String,
    pub remote_chapter_count: Option<f64>,
    pub cover_url: Option<String>,
    pub site_url: Option<String>,
    pub fetched_at: String,
}
```

Comandos:

```rust
studio_search_tracking_works(query, provider)
studio_sync_tracking_work(anilist_id, manga_dex_id)
```

Para AniList, usar GraphQL oficial com `Media(type: MANGA)` e solicitar somente identidade, títulos, status, capítulos, `updatedAt`, capa e `siteUrl`. Definir timeout, User-Agent do Studio e mensagens PT-BR no boundary Tauri. Testes não acessam a rede.

**Step 4: Run focused tests**

Run: `npm --prefix studio test -- src/tracking/__tests__/workTracking.test.ts`

Expected: PASS.

Run: `cargo test --manifest-path studio/src-tauri/Cargo.toml work_tracking::tests`

Expected: PASS.

**Step 5: Commit**

```powershell
git add studio/src/tracking/workTracking.ts studio/src/tracking/__tests__/workTracking.test.ts studio/src-tauri/src/work_tracking.rs studio/src-tauri/src/fixtures/anilist_media.json studio/src-tauri/src/main.rs
git commit -m "feat(studio): add AniList work tracking"
```

## Task 12: Integrar MangaDex, comparação local/remota e Atualizações

**Files:**

- Create: `studio/src-tauri/src/fixtures/mangadex_feed.json`
- Modify: `studio/src-tauri/src/work_tracking.rs`
- Modify: `studio/src/tracking/workTracking.ts`
- Modify: `studio/src/tracking/__tests__/workTracking.test.ts`
- Create: `studio/src/tracking/LinkWorkDialog.tsx`
- Create: `studio/src/tracking/UpdatesView.tsx`
- Create: `studio/src/tracking/__tests__/UpdatesView.test.tsx`
- Modify: `studio/src/library/StudioLibraryHome.tsx`
- Modify: `studio/src/library/WorkDialog.tsx`
- Modify: `studio/src/store/libraryStore.ts`

**Step 1: Write failing comparison and UI tests**

Casos obrigatórios:

- capítulos decimais e especiais;
- idiomas diferentes no feed;
- duplicatas de grupos distintos;
- `remote latest > local latest` gera atualização;
- status manual prevalece visualmente, mas conflito é exibido;
- snapshot expirado continua visível como `Desatualizado` offline.

**Step 2: Run tests to verify failure**

Run: `npm --prefix studio test -- src/tracking/__tests__/workTracking.test.ts src/tracking/__tests__/UpdatesView.test.tsx`

Expected: FAIL.

**Step 3: Implement MangaDex and updates UI**

- Buscar manga por título/ID e feed de capítulos pela API oficial MangaDex.
- Normalizar status MangaDex para o enum interno.
- Filtrar idioma configurado, deduplicar o mesmo rótulo e manter data mais recente.
- Persistir no catálogo somente snapshot normalizado, IDs externos, `fetchedAt`, `expiresAt`, `lastError` e preferências; não persistir resposta bruta completa.
- TTL padrão: 6 horas para status/contagem e 30 minutos para a tela Atualizações quando o usuário pede atualização explícita.
- Exponential backoff limitado para `429`/erros transitórios; respeitar `Retry-After`; botão `Atualizar agora` nunca cria loop.
- `LinkWorkDialog` pesquisa, mostra título/capa/status/fontes e exige confirmação da identidade.
- `UpdatesView` lista somente metadados e abre a obra local. Não expor ação de download.

**Step 4: Run frontend and Rust tests**

Run: `npm --prefix studio test -- src/tracking/__tests__/workTracking.test.ts src/tracking/__tests__/UpdatesView.test.tsx`

Expected: PASS.

Run: `cargo test --manifest-path studio/src-tauri/Cargo.toml work_tracking::tests`

Expected: PASS.

**Step 5: Commit**

```powershell
git add studio/src-tauri/src/fixtures/mangadex_feed.json studio/src-tauri/src/work_tracking.rs studio/src/tracking/workTracking.ts studio/src/tracking/__tests__/workTracking.test.ts studio/src/tracking/LinkWorkDialog.tsx studio/src/tracking/UpdatesView.tsx studio/src/tracking/__tests__/UpdatesView.test.tsx studio/src/library/StudioLibraryHome.tsx studio/src/library/WorkDialog.tsx studio/src/store/libraryStore.ts
git commit -m "feat(studio): track MangaDex chapter updates"
```

## Task 13: Endurecer recuperação, acessibilidade e operação offline

**Files:**

- Create: `studio/src/library/LibraryRecoveryBanner.tsx`
- Create: `studio/src/library/__tests__/LibraryRecoveryBanner.test.tsx`
- Modify: `studio/src/library/StudioLibraryHome.tsx`
- Modify: `studio/src/library/ChapterBrowser.tsx`
- Modify: `studio/src/store/libraryStore.ts`
- Modify: `studio/src/store/projectStore.ts`
- Modify: `studio/src-tauri/src/library.rs`
- Modify: `studio/src-tauri/src/work_tracking.rs`

**Step 1: Write failing recovery tests**

Testar biblioteca recuperada do backup, salvamento falho sem perda do estado em memória, caminho ausente, provider offline, cache vencido e navegação completa por teclado.

**Step 2: Run tests to verify failure**

Run: `npm --prefix studio test -- src/library/__tests__/LibraryRecoveryBanner.test.tsx src/__tests__/StudioHome.test.ts`

Expected: FAIL.

**Step 3: Implement recovery states**

- Banner persistente quando `.bak` for usado, com ação `Salvar cópia recuperada`.
- Erro de gravação mantém a versão em memória e bloqueia fechamento silencioso.
- Cards com caminho ausente permanecem pesquisáveis e relinkáveis.
- Cache externo vencido mostra dados antigos com hora da última atualização e erro atual.
- Todas as ações têm nome acessível, foco visível e ordem lógica; seleção da grade usa setas sem capturar teclas dentro de campos.
- Confirmar contraste das badges de status no tema escuro.

**Step 4: Run full focused suite**

Run: `npm --prefix studio test -- src/library src/tracking src/editor src/translation src/store`

Expected: PASS.

Run: `cargo test --manifest-path studio/src-tauri/Cargo.toml`

Expected: PASS.

**Step 5: Commit**

```powershell
git add studio/src/library/LibraryRecoveryBanner.tsx studio/src/library/__tests__/LibraryRecoveryBanner.test.tsx studio/src/library/StudioLibraryHome.tsx studio/src/library/ChapterBrowser.tsx studio/src/store/libraryStore.ts studio/src/store/projectStore.ts studio/src-tauri/src/library.rs studio/src-tauri/src/work_tracking.rs
git commit -m "fix(studio): harden library recovery and offline states"
```

## Task 14: Verificação integral, documentação e empacotamento

**Files:**

- Modify: `studio/README.md`
- Modify: `docs/plans/2026-07-21-studio-library-workspaces-sync-design.md` only if implementation decisions changed
- Create: `docs/verification/2026-07-21-studio-library-workspaces-sync.md`

**Step 1: Run the complete automated verification**

Run: `npm --prefix studio test`

Expected: todos os testes Studio PASS.

Run: `npm --prefix studio run build`

Expected: TypeScript e Vite PASS.

Run: `cargo fmt --manifest-path studio/src-tauri/Cargo.toml -- --check`

Expected: PASS.

Run: `cargo test --manifest-path studio/src-tauri/Cargo.toml`

Expected: PASS.

Run: `cargo clippy --manifest-path studio/src-tauri/Cargo.toml --all-targets -- -D warnings`

Expected: PASS.

Run: `cargo check --manifest-path studio/src-tauri/Cargo.toml`

Expected: PASS.

Run: `npm run build`

Expected: app principal continua compilando após os slots compartilhados.

**Step 2: Execute the acceptance matrix in Tauri dev**

Run: `npm --prefix studio run tauri:dev`

Validar e registrar evidência para:

1. biblioteca vazia;
2. criar obra manual;
3. anexar projeto Central existente;
4. criar capítulo por pasta;
5. criar capítulo por ZIP e CBZ;
6. alternar Tradução/Edição sem perder seleção;
7. salvar, fechar e reabrir;
8. relink de capítulo movido;
9. link AniList e MangaDex;
10. atualização online e abertura offline pelo cache;
11. conflito de status manual/provedor;
12. confirmar que não existe download de imagens nem chamada ao pipeline.

**Step 3: Verify packaged application**

Run: `npm --prefix studio run tauri:build`

Expected: pacote desktop gerado e iniciado sem depender do servidor Vite.

**Step 4: Document the actual behavior**

Atualizar `studio/README.md` com:

- localização lógica do catálogo no app data;
- diferença entre biblioteca e `project.json`;
- formatos de importação;
- áreas de trabalho;
- provedores, cache e limites;
- privacidade e comportamento offline;
- recuperação/relink;
- FLUX explicitamente adiado.

Em `docs/verification/...`, registrar comandos, resultados, SO, viewport, fixtures e limitações ainda abertas. Não afirmar paridade com Photoshop: declarar o escopo específico de manga/manhwa alcançado.

**Step 5: Commit verification artifacts**

```powershell
git add studio/README.md docs/plans/2026-07-21-studio-library-workspaces-sync-design.md docs/verification/2026-07-21-studio-library-workspaces-sync.md
git commit -m "docs(studio): document library and workspaces verification"
```

## Critérios de aceite finais

- A home mostra obras à esquerda e capítulos da obra selecionada na área principal.
- É possível adicionar uma obra, anexar um `project.json` existente e criar capítulo por pasta/ZIP/CBZ.
- O catálogo persiste independentemente dos projetos e se recupera de escrita interrompida.
- Um capítulo aberto alterna `Tradução | Edição` sem recarregar nem duplicar o documento.
- Tradução manual permite criar/selecionar blocos, editar tradução, notas e status, filtrar pendências e avançar por teclado.
- Projetos antigos continuam abrindo e exportando com os aliases atuais.
- AniList/MangaDex são opcionais, funcionam por Rust, têm cache offline e não baixam páginas.
- Status manual não é sobrescrito silenciosamente; conflitos ficam visíveis.
- Caminhos ausentes podem ser relinkados sem apagar entradas ou dados.
- Suite Studio, build Studio, testes/checks Rust e build do app principal passam.
- Nenhuma modificação foi feita no pipeline automático e FLUX segue fora deste marco.

## Fora deste plano

- FLUX/ControlNet inpainting e qualquer gerador por prompt.
- Tradução automática, OCR automático ou execução do pipeline Central dentro do Studio.
- PSD completo, plugins Photoshop ou paridade genérica com todas as funções do Photoshop.
- Download automático de capítulos, scraping de leitores ou armazenamento de conteúdo remoto.
- Colaboração, nuvem, contas e compartilhamento.

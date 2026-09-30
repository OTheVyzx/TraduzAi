import { lazy, Suspense, useEffect, useRef, useState } from "react";
import { useStore } from "zustand";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { save as saveDialog } from "@tauri-apps/plugin-dialog";
import { StudioLibraryHome } from "./library/StudioLibraryHome";
import { StudioHomeDashboard } from "./home/StudioHomeDashboard";
import { StudioAppShell, type StudioShellView } from "./shell/StudioAppShell";
import { StudioSettingsView } from "./settings/StudioSettingsView";
import { ReaderView, type ReaderTranslationRequest } from "./reader/ReaderView";
import { tauriSourceRuntimeClient } from "./reader/sourceRuntimeClient";
import { createManualChapterFromImages } from "./backend/projectDialog";
import { createDefaultLibraryBackend } from "./library/libraryBackend";
import {
  findChapterByProjectPath,
  findChapterForProjectRegistration,
  findWorkForProjectRegistration,
} from "./library/projectRegistration";
import type { StudioProject } from "./project/studioProject";
import { createLibraryStore } from "./store/libraryStore";
import { useStudioProjectStore } from "./store/projectStore";

const StudioWorkspaceShell = lazy(async () => {
  const mod = await import("./editor/StudioWorkspaceShell");
  return { default: mod.StudioWorkspaceShell };
});

const libraryStore = createLibraryStore(createDefaultLibraryBackend());

function stableId(prefix: string, value: string): string {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return `${prefix}-${(hash >>> 0).toString(36)}`;
}

function projectWorkTitle(project: StudioProject, projectPath: string): string {
  if (project.obra?.trim()) return project.obra.trim();
  const normalized = projectPath.replace(/\\/g, "/").replace(/\/project\.json$/i, "");
  return normalized.split("/").filter(Boolean).at(-2) ?? "Obra sem título";
}

function projectChapterLabel(project: StudioProject, projectPath: string): string {
  if (project.capitulo !== undefined && String(project.capitulo).trim()) {
    return String(project.capitulo).trim();
  }
  const normalized = projectPath.replace(/\\/g, "/").replace(/\/project\.json$/i, "");
  return normalized.split("/").filter(Boolean).at(-1) ?? "1";
}

function catalogCoverPath(projectPath: string, coverPath?: string | null): string | null {
  if (!coverPath) return null;
  if (/^(data|blob|asset|file):/i.test(coverPath) || /^https?:\/\//i.test(coverPath) || /^[A-Za-z]:[\\/]/.test(coverPath)) {
    return coverPath;
  }
  const normalizedProjectPath = projectPath.replace(/\\/g, "/");
  const projectDirectory = normalizedProjectPath.toLocaleLowerCase("en-US").endsWith(".json")
    ? normalizedProjectPath.slice(0, normalizedProjectPath.lastIndexOf("/"))
    : normalizedProjectPath;
  return `${projectDirectory}/${coverPath.replace(/\\/g, "/")}`;
}

export function App() {
  const project = useStudioProjectStore((state) => state.project);
  const projectPath = useStudioProjectStore((state) => state.projectPath);
  const projectError = useStudioProjectStore((state) => state.error);
  const loadProject = useStudioProjectStore((state) => state.loadProject);
  const openProjectFromDialog = useStudioProjectStore((state) => state.openProjectFromDialog);
  const recoverySnapshot = useStudioProjectStore((state) => state.recoverySnapshot);
  const restoreRecovery = useStudioProjectStore((state) => state.restoreRecovery);
  const dismissRecovery = useStudioProjectStore((state) => state.dismissRecovery);
  const closeProject = useStudioProjectStore((state) => state.closeProject);
  const projectHasUnsavedChanges = useStudioProjectStore((state) => state.hasUnsavedChanges);
  const library = useStore(libraryStore, (state) => state);
  const registeredProjects = useRef(new Set<string>());
  const [lastOpenedChapterPath, setLastOpenedChapterPath] = useState<string | null>(null);
  const [activeView, setActiveView] = useState<StudioShellView>("home");
  const [pendingNewWorkRequest, setPendingNewWorkRequest] = useState(false);
  const [notification, setNotification] = useState<string | null>(null);

  useEffect(() => {
    void libraryStore.getState().load();
  }, []);

  useEffect(() => {
    const hasUnsavedChanges = library.hasUnsavedChanges || projectHasUnsavedChanges;
    if (!hasUnsavedChanges) return;
    const closeWarning = library.hasUnsavedChanges
      ? "Há alterações da biblioteca que ainda não foram gravadas. Fechar o Studio e descartar esta sessão?"
      : "Há alterações não salvas neste capítulo. Fechar o Studio e descartá-las?";
    const warnBeforeClose = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = "";
    };
    const tauriRuntime = "__TAURI_INTERNALS__" in window || "__TAURI__" in window;
    if (!tauriRuntime) window.addEventListener("beforeunload", warnBeforeClose);
    let disposed = false;
    let unlistenCloseRequest: (() => void) | undefined;
    if (tauriRuntime) {
      void getCurrentWindow().onCloseRequested((event) => {
        if (!window.confirm(closeWarning)) event.preventDefault();
      }).then((unlisten) => {
        if (disposed) unlisten();
        else unlistenCloseRequest = unlisten;
      }).catch((error) => {
        console.warn("Não foi possível proteger o fechamento da janela do Studio:", error);
      });
    }
    return () => {
      disposed = true;
      if (!tauriRuntime) window.removeEventListener("beforeunload", warnBeforeClose);
      unlistenCloseRequest?.();
    };
  }, [library.hasUnsavedChanges, projectHasUnsavedChanges]);

  useEffect(() => {
    if (project) return;
    const configuredProjectPath = import.meta.env.VITE_STUDIO_PROJECT_PATH?.trim();
    if (configuredProjectPath) void loadProject(configuredProjectPath);
  }, [loadProject, project]);

  useEffect(() => {
    if (!project || !projectPath || projectPath.startsWith("memory://") || library.status !== "ready") return;
    setLastOpenedChapterPath(projectPath);
    if (registeredProjects.current.has(projectPath)) return;
    registeredProjects.current.add(projectPath);

    const register = async () => {
      const title = projectWorkTitle(project, projectPath);
      const chapterLabel = projectChapterLabel(project, projectPath);
      const existingWork = findWorkForProjectRegistration(library.document.works, title, projectPath);
      const existingChapter = findChapterForProjectRegistration(existingWork, projectPath, chapterLabel);
      const workId = existingWork?.id ?? stableId("work", title.toLocaleLowerCase("pt-BR"));
      if (!existingWork) {
        await libraryStore.getState().addWork({
          id: workId,
          title,
          aliases: [],
          publicationStatus: "unknown",
        });
      }
      await libraryStore.getState().upsertChapter(workId, {
        id: existingChapter?.id ?? stableId("chapter", projectPath.toLocaleLowerCase("en-US")),
        label: chapterLabel,
        projectPath,
        coverPath: catalogCoverPath(projectPath, project.paginas[0]?.arquivo_original),
        pageCount: project.paginas.length,
        completedPages: 0,
        workflowStatus: "editing",
        lastOpenedAt: new Date().toISOString(),
      });
      await libraryStore.getState().selectWork(workId);
    };

    void register();
  }, [library.document.works, library.status, project, projectPath]);

  if (project && projectPath) {
    return (
      <Suspense fallback={(
        <StudioAppShell
          activeView="library"
          onNavigate={() => undefined}
          notification={notification}
          onDismissNotification={() => setNotification(null)}
        >
          <StudioBoot message="Carregando editor..." />
        </StudioAppShell>
      )}>
        <StudioWorkspaceShell
          project={project}
          projectPath={projectPath}
          onProjectPromoted={async (previousPath, promotedPath) => {
            const linked = findChapterByProjectPath(
              libraryStore.getState().document.works,
              previousPath,
            );
            if (linked) {
              await libraryStore.getState().relinkChapter(
                linked.work.id,
                linked.chapter.id,
                promotedPath,
              );
              const chapterLabel = projectChapterLabel(project, promotedPath);
              const refreshedWork = libraryStore.getState().document.works.find(
                (work) => work.id === linked.work.id,
              );
              const duplicates = refreshedWork?.chapters.filter(
                (chapter) => chapter.id !== linked.chapter.id && chapter.label === chapterLabel,
              ) ?? [];
              for (const duplicate of duplicates) {
                await libraryStore.getState().removeChapter(linked.work.id, duplicate.id);
              }
            }
          }}
          onBack={() => {
            if (projectHasUnsavedChanges && !window.confirm("Há alterações não salvas neste capítulo. Descartar e voltar para a biblioteca?")) return;
            closeProject(true);
          }}
        />
      </Suspense>
    );
  }

  const openLibraryWork = (workId: string) => {
    void library.selectWork(workId);
    setActiveView("library");
  };

  const translateReaderChapter = async ({ manga, chapter, download }: ReaderTranslationRequest) => {
    const projectJsonPath = await saveDialog({
      title: `Salvar ${manga.title} — ${chapter.name}`,
      defaultPath: "project.json",
      filters: [{ name: "Projeto TraduzAI", extensions: ["json"] }],
    });
    if (!projectJsonPath) return;
    const prepared = await tauriSourceRuntimeClient.translateReaderChapter({ jobId: download.jobId, projectJsonPath });
    const chapterLabel = chapter.chapterNumber !== undefined && chapter.chapterNumber >= 0
      ? String(chapter.chapterNumber)
      : chapter.name;
    const result = await createManualChapterFromImages({
      workTitle: manga.title,
      chapterLabel,
      chapterTitle: chapter.name,
      sourceLanguage: "auto",
      targetLanguage: "pt-BR",
      sourcePath: projectJsonPath,
      projectJsonPath,
    }, undefined, prepared.preparedPages);
    const currentLibrary = libraryStore.getState().document;
    const existingWork = currentLibrary.works.find((work) => work.title.toLocaleLowerCase("pt-BR") === manga.title.toLocaleLowerCase("pt-BR"));
    const workId = existingWork?.id ?? stableId("work", manga.title.toLocaleLowerCase("pt-BR"));
    if (!existingWork) {
      await libraryStore.getState().addWork({
        id: workId,
        title: manga.title,
        aliases: [],
        coverPath: manga.thumbnailUrl ?? null,
        external: {
          contentOrigin: {
            kind: "mihon-extension",
            runtimeRecordId: manga.id,
            extensionPackage: manga.extensionPackage,
            sourceId: manga.sourceId,
            sourceName: manga.sourceName,
          },
        },
      });
    }
    await libraryStore.getState().upsertChapter(workId, {
      id: stableId("chapter", projectJsonPath.toLocaleLowerCase("en-US")),
      label: chapterLabel,
      title: chapter.name,
      projectPath: projectJsonPath,
      coverPath: catalogCoverPath(projectJsonPath, result.project.paginas[0]?.arquivo_original),
      pageCount: result.project.paginas.length,
      completedPages: 0,
      workflowStatus: "pending",
      lastOpenedAt: new Date().toISOString(),
    });
    await libraryStore.getState().selectWork(workId);
    setLastOpenedChapterPath(projectJsonPath);
    await loadProject(projectJsonPath);
  };

  const libraryView = <StudioLibraryHome
      document={library.document}
      status={library.status}
      error={library.error ?? projectError}
      libraryError={library.error}
      recoveryAvailable={Boolean(recoverySnapshot)}
      libraryRecoveredFromBackup={library.recoveredFromBackup}
      hasUnsavedLibraryChanges={library.hasUnsavedChanges}
      onRecover={() => void restoreRecovery()}
      onDismissRecovery={() => void dismissRecovery()}
      onSaveRecoveredCopy={() => library.saveRecoveredCopy()}
      onSaveWork={async (input) => {
        await library.addWork(input);
        await library.selectWork(input.id);
      }}
      onRemoveWork={(workId) => library.removeWork(workId)}
      onAttachChapter={(workId, draft) => library.upsertChapter(workId, {
        id: stableId("chapter", draft.projectPath.toLocaleLowerCase("en-US")),
        label: draft.chapterLabel,
        projectPath: draft.projectPath,
        coverPath: draft.coverPath,
        pageCount: draft.pageCount,
        completedPages: 0,
        workflowStatus: "editing",
        lastOpenedAt: null,
      })}
      onCreateManualChapter={async (workId, input, preparedPages) => {
        const result = await createManualChapterFromImages(input, undefined, preparedPages);
        await library.upsertChapter(workId, {
          id: stableId("chapter", input.projectJsonPath.toLocaleLowerCase("en-US")),
          label: input.chapterLabel,
          title: input.chapterTitle,
          projectPath: input.projectJsonPath,
          coverPath: catalogCoverPath(input.projectJsonPath, result.project.paginas[0]?.arquivo_original),
          pageCount: result.project.paginas.length,
          completedPages: 0,
          workflowStatus: "pending",
          lastOpenedAt: new Date().toISOString(),
        });
        await library.selectWork(workId);
        setLastOpenedChapterPath(input.projectJsonPath);
        await loadProject(input.projectJsonPath);
      }}
      onRemoveChapter={(workId, chapterId) => library.removeChapter(workId, chapterId)}
      onRelinkChapter={(workId, chapterId, path) => library.relinkChapter(workId, chapterId, path)}
      onImportProject={() => void openProjectFromDialog()}
      onSelectWork={(workId) => void library.selectWork(workId)}
      onOpenChapter={(path) => {
        setLastOpenedChapterPath(path);
        void loadProject(path);
      }}
      initialSelectedChapterPath={lastOpenedChapterPath}
      onSetChapterView={(view) => void library.setChapterView(view)}
      onSetThumbnailSize={(size) => void library.setThumbnailSize(size)}
      onSetTrackingLanguage={(language) => void library.setTrackingLanguage(language)}
      openWorkDialogRequest={pendingNewWorkRequest}
      onConsumeWorkDialogRequest={() => setPendingNewWorkRequest(false)}
    />;

  return (
    <StudioAppShell activeView={activeView} onNavigate={setActiveView} notification={notification} onDismissNotification={() => setNotification(null)}>
      {activeView === "home" && <StudioHomeDashboard
        document={library.document}
        onOpenWork={openLibraryWork}
        onCreateWork={() => { setPendingNewWorkRequest(true); setActiveView("library"); }}
        onImportProject={() => void openProjectFromDialog()}
        onOpenSettings={() => setActiveView("settings")}
      />}
      {activeView === "library" && libraryView}
      {activeView === "reader" && <ReaderView onTranslateChapter={translateReaderChapter} onNotification={setNotification} />}
      {activeView === "settings" && <StudioSettingsView
        preferences={library.document.preferences}
        saving={library.status === "saving"}
        onSave={async (draft) => {
          await library.setChapterView(draft.defaultChapterView);
          await library.setThumbnailSize(draft.thumbnailSize);
          await library.setTrackingLanguage(draft.trackingLanguage);
        }}
      />}
    </StudioAppShell>
  );
}

function StudioBoot({ message, error }: { message: string; error?: string | null }) {
  return (
    <main className="studio-boot studio-boot-backdrop" role="status" aria-live="polite">
      <section className="studio-boot-panel">
        <span className="studio-boot-spinner" aria-hidden="true" />
        <p className="eyebrow">TraduzAI Studio</p>
        <h1>Preparando editor</h1>
        <p>{message}</p>
        {error && <p className="error">{error}</p>}
      </section>
    </main>
  );
}

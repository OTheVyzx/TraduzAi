import { lazy, Suspense, useEffect, useRef, useState } from "react";
import { useStore } from "zustand";
import { StudioLibraryHome } from "./library/StudioLibraryHome";
import { createDefaultLibraryBackend } from "./library/libraryBackend";
import type { StudioProject } from "./project/studioProject";
import { createLibraryStore } from "./store/libraryStore";
import { useStudioProjectStore } from "./store/projectStore";

const StudioSharedEditor = lazy(async () => {
  const mod = await import("./editor/StudioSharedEditor");
  return { default: mod.StudioSharedEditor };
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

export function App() {
  const project = useStudioProjectStore((state) => state.project);
  const projectPath = useStudioProjectStore((state) => state.projectPath);
  const projectError = useStudioProjectStore((state) => state.error);
  const loadProject = useStudioProjectStore((state) => state.loadProject);
  const openProjectFromDialog = useStudioProjectStore((state) => state.openProjectFromDialog);
  const recoverySnapshot = useStudioProjectStore((state) => state.recoverySnapshot);
  const restoreRecovery = useStudioProjectStore((state) => state.restoreRecovery);
  const dismissRecovery = useStudioProjectStore((state) => state.dismissRecovery);
  const library = useStore(libraryStore, (state) => state);
  const registeredProjects = useRef(new Set<string>());
  const [addingWork, setAddingWork] = useState(false);

  useEffect(() => {
    void libraryStore.getState().load();
  }, []);

  useEffect(() => {
    if (project) return;
    const configuredProjectPath = import.meta.env.VITE_STUDIO_PROJECT_PATH?.trim();
    if (configuredProjectPath) void loadProject(configuredProjectPath);
  }, [loadProject, project]);

  useEffect(() => {
    if (!project || !projectPath || projectPath.startsWith("memory://") || library.status !== "ready") return;
    if (registeredProjects.current.has(projectPath)) return;
    registeredProjects.current.add(projectPath);

    const register = async () => {
      const title = projectWorkTitle(project, projectPath);
      const existingWork = library.document.works.find(
        (work) => work.title.localeCompare(title, "pt-BR", { sensitivity: "base" }) === 0,
      );
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
        id: stableId("chapter", projectPath.toLocaleLowerCase("en-US")),
        label: projectChapterLabel(project, projectPath),
        projectPath,
        coverPath: project.paginas[0]?.arquivo_original ?? null,
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
      <Suspense fallback={<StudioBoot message="Carregando editor..." />}>
        <StudioSharedEditor project={project} projectPath={projectPath} />
      </Suspense>
    );
  }

  const handleAddWork = async () => {
    if (addingWork) return;
    setAddingWork(true);
    try {
      const sequence = library.document.works.length + 1;
      const id = `work-${Date.now().toString(36)}`;
      await library.addWork({ id, title: `Nova obra ${sequence}`, aliases: [] });
      await library.selectWork(id);
    } finally {
      setAddingWork(false);
    }
  };

  return (
    <StudioLibraryHome
      document={library.document}
      status={library.status}
      error={library.error ?? projectError}
      recoveryAvailable={Boolean(recoverySnapshot)}
      onRecover={() => void restoreRecovery()}
      onDismissRecovery={() => void dismissRecovery()}
      onAddWork={() => void handleAddWork()}
      onAddChapter={() => void openProjectFromDialog()}
      onImportProject={() => void openProjectFromDialog()}
      onSelectWork={(workId) => void library.selectWork(workId)}
      onOpenChapter={(path) => void loadProject(path)}
      onSetChapterView={(view) => void library.setChapterView(view)}
      onSetThumbnailSize={(size) => void library.setThumbnailSize(size)}
    />
  );
}

function StudioBoot({ message, error }: { message: string; error?: string | null }) {
  return (
    <main className="studio-boot">
      <section className="studio-boot-panel">
        <p className="eyebrow">TraduzAI Studio</p>
        <h1>Preparando editor</h1>
        <p>{message}</p>
        {error && <p className="error">{error}</p>}
      </section>
    </main>
  );
}

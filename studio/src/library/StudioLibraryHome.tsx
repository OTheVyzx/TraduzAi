import { useEffect, useMemo, useRef, useState, type CSSProperties, type KeyboardEvent, type PointerEvent } from "react";
import { AlertTriangle, LoaderCircle } from "lucide-react";
import {
  openCoverImageDialog,
  openManualChapterArchiveDialog,
  openManualChapterFolderDialog,
  openProjectForAttachment,
  projectPathExists,
  saveProjectDialog,
  type ManualChapterCreationInput,
  type PreparedManualPage,
} from "../backend/projectDialog";
import type { AddLibraryWorkInput, LibraryStoreStatus } from "../store/libraryStore";
import { LinkWorkDialog } from "../tracking/LinkWorkDialog";
import { UpdatesView } from "../tracking/UpdatesView";
import { createTrackingCache, TRACKING_CACHE_TTL_MS, type WorkTrackingSnapshot } from "../tracking/workTracking";
import { AttachProjectDialog, type ProjectAttachmentDraft } from "./AttachProjectDialog";
import type { StudioLibrary } from "./libraryModel";
import { ChapterBrowser } from "./ChapterBrowser";
import { CreateChapterDialog } from "./CreateChapterDialog";
import { LibraryToolbar } from "./LibraryToolbar";
import { LibraryRecoveryBanner } from "./LibraryRecoveryBanner";
import { WorkDialog } from "./WorkDialog";
import { WorkLibrarySidebar } from "./WorkLibrarySidebar";
import { WorkInspector } from "./WorkInspector";
import { selectSelectedWorkChapterMetrics } from "./librarySelectors";
import {
  DEFAULT_LIBRARY_PANES,
  LIBRARY_PANE_STORAGE_KEY,
  parseLibraryPaneLayout,
  resizeLibraryPane,
  type LibraryPaneLayout,
  type LibraryPaneSide,
} from "./libraryPaneLayout";

export function StudioLibraryHome({
  document,
  status,
  error,
  libraryError = null,
  recoveryAvailable = false,
  libraryRecoveredFromBackup = false,
  hasUnsavedLibraryChanges = false,
  onRecover,
  onDismissRecovery,
  onSaveRecoveredCopy,
  onSaveWork,
  onRemoveWork,
  onAttachChapter,
  onCreateManualChapter,
  onRemoveChapter,
  onRelinkChapter,
  onImportProject,
  onSelectWork,
  onOpenChapter,
  onSetChapterView,
  onSetThumbnailSize,
  onSetTrackingLanguage,
  initialSelectedChapterPath = null,
  openWorkDialogRequest = false,
  onConsumeWorkDialogRequest,
}: {
  document: StudioLibrary;
  status: LibraryStoreStatus;
  error?: string | null;
  libraryError?: string | null;
  recoveryAvailable?: boolean;
  libraryRecoveredFromBackup?: boolean;
  hasUnsavedLibraryChanges?: boolean;
  onRecover?: () => void;
  onDismissRecovery?: () => void;
  onSaveRecoveredCopy?: () => void | Promise<void>;
  onSaveWork: (input: AddLibraryWorkInput) => void | Promise<void>;
  onRemoveWork: (workId: string) => void | Promise<void>;
  onAttachChapter: (workId: string, draft: ProjectAttachmentDraft) => void | Promise<void>;
  onCreateManualChapter: (
    workId: string,
    input: ManualChapterCreationInput,
    preparedPages?: PreparedManualPage[] | null,
  ) => Promise<void>;
  onRemoveChapter: (workId: string, chapterId: string) => void | Promise<void>;
  onRelinkChapter: (workId: string, chapterId: string, projectPath: string) => void | Promise<void>;
  onImportProject: () => void;
  onSelectWork: (workId: string) => void;
  onOpenChapter: (projectPath: string) => void;
  onSetChapterView: (view: "grid" | "list") => void;
  onSetThumbnailSize: (size: number) => void;
  onSetTrackingLanguage?: (language: string) => void;
  initialSelectedChapterPath?: string | null;
  openWorkDialogRequest?: boolean;
  onConsumeWorkDialogRequest?: () => void;
}) {
  const [workQuery, setWorkQuery] = useState("");
  const [chapterQuery, setChapterQuery] = useState("");
  const selectedWork = useMemo(
    () => document.works.find((work) => work.id === document.selectedWorkId) ?? null,
    [document.selectedWorkId, document.works],
  );
  const chapterMetrics = useMemo(() => selectSelectedWorkChapterMetrics(document), [document]);
  const initialChapterId = selectedWork?.chapters.find((chapter) => chapter.projectPath === initialSelectedChapterPath)?.id ?? null;
  const [selectedChapterId, setSelectedChapterId] = useState<string | null>(initialChapterId);
  const [workDialogOpen, setWorkDialogOpen] = useState(false);
  const [editingWorkId, setEditingWorkId] = useState<string | null>(null);
  const [attachDialogOpen, setAttachDialogOpen] = useState(false);
  const [createChapterDialogOpen, setCreateChapterDialogOpen] = useState(false);
  const [linkWorkId, setLinkWorkId] = useState<string | null>(null);
  const [updatesOpen, setUpdatesOpen] = useState(false);
  const libraryLayoutRef = useRef<HTMLDivElement>(null);
  const activeResizeCleanupRef = useRef<(() => void) | null>(null);
  const [paneLayout, setPaneLayout] = useState<LibraryPaneLayout>(() => {
    if (typeof window === "undefined") return DEFAULT_LIBRARY_PANES;
    return parseLibraryPaneLayout(window.localStorage.getItem(LIBRARY_PANE_STORAGE_KEY), window.innerWidth);
  });
  const [missingProjectPaths, setMissingProjectPaths] = useState<Set<string>>(new Set());
  const linkingWork = document.works.find((work) => work.id === linkWorkId) ?? null;

  useEffect(() => () => activeResizeCleanupRef.current?.(), []);

  const persistPaneLayout = (next: LibraryPaneLayout) => {
    if (typeof window !== "undefined") window.localStorage.setItem(LIBRARY_PANE_STORAGE_KEY, JSON.stringify(next));
  };

  const updatePaneLayout = (side: LibraryPaneSide, delta: number) => {
    const availableWidth = libraryLayoutRef.current?.clientWidth ?? window.innerWidth;
    setPaneLayout((current) => {
      const next = resizeLibraryPane(current, side, delta, availableWidth);
      persistPaneLayout(next);
      return next;
    });
  };

  const beginPaneResize = (side: LibraryPaneSide, event: PointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    activeResizeCleanupRef.current?.();
    const startX = event.clientX;
    const startLayout = paneLayout;
    const availableWidth = libraryLayoutRef.current?.clientWidth ?? window.innerWidth;
    let latest = startLayout;
    const move = (moveEvent: globalThis.PointerEvent) => {
      latest = resizeLibraryPane(startLayout, side, moveEvent.clientX - startX, availableWidth);
      setPaneLayout(latest);
    };
    const cleanup = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", finish);
      window.removeEventListener("pointercancel", finish);
      globalThis.document.body.classList.remove("studio-library-pane-resizing");
      activeResizeCleanupRef.current = null;
    };
    const finish = () => {
      persistPaneLayout(latest);
      cleanup();
    };
    globalThis.document.body.classList.add("studio-library-pane-resizing");
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", finish);
    window.addEventListener("pointercancel", finish);
    activeResizeCleanupRef.current = cleanup;
  };

  const handlePaneDividerKey = (side: LibraryPaneSide, event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "Home") {
      event.preventDefault();
      setPaneLayout((current) => {
        const next = { ...current, [side]: DEFAULT_LIBRARY_PANES[side] };
        persistPaneLayout(next);
        return next;
      });
      return;
    }
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    updatePaneLayout(side, event.key === "ArrowRight" ? 12 : -12);
  };

  const resetPaneDivider = (side: LibraryPaneSide) => {
    setPaneLayout((current) => {
      const next = { ...current, [side]: DEFAULT_LIBRARY_PANES[side] };
      persistPaneLayout(next);
      return next;
    });
  };
  useEffect(() => {
    setSelectedChapterId(
      selectedWork?.chapters.find((chapter) => chapter.projectPath === initialSelectedChapterPath)?.id ?? null,
    );
    setChapterQuery("");
  }, [initialSelectedChapterPath, selectedWork?.id]);

  useEffect(() => {
    if (!openWorkDialogRequest) return;
    setEditingWorkId(null);
    setWorkDialogOpen(true);
    onConsumeWorkDialogRequest?.();
  }, [onConsumeWorkDialogRequest, openWorkDialogRequest]);

  const projectPathSignature = useMemo(
    () => document.works.flatMap((work) => work.chapters.map((chapter) => chapter.projectPath)).join("\u0000"),
    [document.works],
  );

  useEffect(() => {
    let cancelled = false;
    const paths = projectPathSignature ? projectPathSignature.split("\u0000") : [];
    void Promise.all(paths.map(async (path) => {
      try {
        return { path, exists: await projectPathExists(path) };
      } catch {
        return { path, exists: true };
      }
    })).then((results) => {
      if (!cancelled) setMissingProjectPaths(new Set(results.filter((result) => !result.exists).map((result) => result.path)));
    });
    return () => { cancelled = true; };
  }, [projectPathSignature]);

  const chooseAttachment = async (): Promise<ProjectAttachmentDraft | null> => {
    const selected = await openProjectForAttachment();
    if (!selected) return null;
    const { project, projectPath } = selected;
    const normalized = projectPath.replace(/\\/g, "/").replace(/\/project\.json$/i, "");
    const pathParts = normalized.split("/").filter(Boolean);
    return {
      projectPath,
      workTitle: project.obra?.trim() || pathParts.at(-2) || selectedWork?.title || "Obra sem título",
      chapterLabel: project.capitulo === undefined || !String(project.capitulo).trim()
        ? pathParts.at(-1) || "1"
        : String(project.capitulo).trim(),
      pageCount: project.paginas.length,
      coverPath: project.paginas[0]?.arquivo_original ?? null,
    };
  };

  const relinkChapter = async (chapterId: string) => {
    if (!selectedWork) return;
    const selected = await openProjectForAttachment();
    if (!selected) return;
    await onRelinkChapter(selectedWork.id, chapterId, selected.projectPath);
  };

  const removeChapterReference = async (chapterId: string) => {
    if (!selectedWork) return;
    const confirmed = window.confirm("Remover este capítulo somente da biblioteca? Nenhum arquivo será apagado do disco.");
    if (!confirmed) return;
    await onRemoveChapter(selectedWork.id, chapterId);
    setSelectedChapterId(null);
  };

  const linkTrackingSource = async (snapshot: WorkTrackingSnapshot) => {
    if (!linkingWork) return;
    const previousSnapshots = linkingWork.external.tracking?.snapshots ?? [];
    const snapshots = [
      ...previousSnapshots.filter((candidate) => candidate.provider !== snapshot.provider),
      snapshot,
    ];
    await onSaveWork({
      id: linkingWork.id,
      title: linkingWork.title,
      aliases: linkingWork.aliases,
      coverPath: linkingWork.coverPath,
      publicationStatus: linkingWork.external.manualStatusOverride ?? snapshot.status,
      external: {
        ...linkingWork.external,
        ...(snapshot.provider === "anilist" ? { anilistId: Number(snapshot.providerId) } : {}),
        ...(snapshot.provider === "mangadex" ? { mangaDexId: snapshot.providerId } : {}),
        ...(snapshot.siteUrl ? { canonicalUrl: snapshot.siteUrl } : {}),
        tracking: createTrackingCache(snapshots, new Date(), TRACKING_CACHE_TTL_MS),
      },
    });
  };

  return (
    <main className="studio-home">
      <div
        ref={libraryLayoutRef}
        className="studio-library-layout"
        style={{
          "--studio-library-left-pane": `${paneLayout.left}px`,
          "--studio-library-right-pane": `${paneLayout.right}px`,
        } as CSSProperties}
      >
        <WorkLibrarySidebar
          works={document.works}
          selectedWorkId={document.selectedWorkId}
          query={workQuery}
          onQueryChange={setWorkQuery}
          onSelectWork={onSelectWork}
          onAddWork={() => {
            setEditingWorkId(null);
            setWorkDialogOpen(true);
          }}
        />

        <div
          className="studio-library-pane-divider studio-library-pane-divider-left"
          role="separator"
          aria-label="Redimensionar painel de obras"
          aria-orientation="vertical"
          aria-valuemin={220}
          aria-valuemax={480}
          aria-valuenow={paneLayout.left}
          tabIndex={0}
          onPointerDown={(event) => beginPaneResize("left", event)}
          onKeyDown={(event) => handlePaneDividerKey("left", event)}
          onDoubleClick={() => resetPaneDivider("left")}
        />

        <section className="studio-library-main">
          {chapterMetrics && (
            <div className="studio-library-chapter-metrics" aria-label="Métricas dos capítulos">
              <article><span className="studio-library-chapter-metric-line"><strong>{chapterMetrics.totalChapters}</strong><span>Capítulos totais</span></span></article>
              <article><span className="studio-library-chapter-metric-line"><strong>{chapterMetrics.translatedChapters}</strong><span>Traduzidos</span></span></article>
              <article><span className="studio-library-chapter-metric-line"><strong>{chapterMetrics.editingChapters}</strong><span>Em edição</span></span></article>
              <article><span className="studio-library-chapter-metric-line"><strong>{chapterMetrics.reviewChapters}</strong><span>Em revisão</span></span></article>
            </div>
          )}
          <LibraryToolbar
            title={selectedWork?.title ?? "Nenhuma obra selecionada"}
            chapterCount={selectedWork?.chapters.length ?? 0}
            query={chapterQuery}
            view={document.preferences.chapterView}
            thumbnailSize={document.preferences.thumbnailSize}
            onQueryChange={setChapterQuery}
            onSetView={onSetChapterView}
            onSetThumbnailSize={onSetThumbnailSize}
            onOpenUpdates={() => setUpdatesOpen(true)}
            onEditWork={selectedWork ? () => {
              setEditingWorkId(selectedWork.id);
              setWorkDialogOpen(true);
            } : undefined}
          />

          <LibraryRecoveryBanner
            recoveredFromBackup={libraryRecoveredFromBackup}
            hasUnsavedChanges={hasUnsavedLibraryChanges}
            error={libraryError}
            saving={status === "saving"}
            onSaveRecoveredCopy={() => onSaveRecoveredCopy?.()}
          />

          {recoveryAvailable && (
            <div className="studio-library-recovery" role="status">
              <AlertTriangle size={16} />
              <span><strong>Sessão recuperável encontrada.</strong> O último autosave pode ser restaurado.</span>
              <button type="button" onClick={onRecover}>Recuperar</button>
              <button type="button" onClick={onDismissRecovery}>Ignorar</button>
            </div>
          )}

          {(status === "loading" || status === "idle") && document.works.length === 0 ? (
            <div className="studio-library-loading"><LoaderCircle size={22} /> Carregando biblioteca…</div>
          ) : (
            <ChapterBrowser
              work={selectedWork}
              query={chapterQuery}
              view={document.preferences.chapterView}
              thumbnailSize={document.preferences.thumbnailSize}
              selectedChapterId={selectedChapterId}
              onSelectChapter={setSelectedChapterId}
              onOpenChapter={onOpenChapter}
              onImportProject={onImportProject}
              onAddChapter={() => setCreateChapterDialogOpen(true)}
              missingProjectPaths={missingProjectPaths}
              onRelinkChapter={(chapterId) => void relinkChapter(chapterId)}
              onRemoveChapter={(chapterId) => void removeChapterReference(chapterId)}
            />
          )}

          {error && <p className="studio-home-error">{error}</p>}
        </section>
        <div
          className="studio-library-pane-divider studio-library-pane-divider-right"
          role="separator"
          aria-label="Redimensionar painel de detalhes"
          aria-orientation="vertical"
          aria-valuemin={240}
          aria-valuemax={520}
          aria-valuenow={paneLayout.right}
          tabIndex={0}
          onPointerDown={(event) => beginPaneResize("right", event)}
          onKeyDown={(event) => handlePaneDividerKey("right", event)}
          onDoubleClick={() => resetPaneDivider("right")}
        />
        <WorkInspector work={selectedWork} onEditWork={selectedWork ? () => { setEditingWorkId(selectedWork.id); setWorkDialogOpen(true); } : undefined} />
      </div>

      <WorkDialog
        open={workDialogOpen}
        work={editingWorkId ? document.works.find((work) => work.id === editingWorkId) ?? null : null}
        onClose={() => setWorkDialogOpen(false)}
        onSave={onSaveWork}
        onRemove={onRemoveWork}
        onChooseCover={openCoverImageDialog}
        onLinkTracking={editingWorkId ? () => {
          setWorkDialogOpen(false);
          setLinkWorkId(editingWorkId);
        } : undefined}
      />

      <LinkWorkDialog
        open={Boolean(linkingWork)}
        work={linkingWork}
        onClose={() => setLinkWorkId(null)}
        onConfirm={linkTrackingSource}
      />

      <UpdatesView
        open={updatesOpen}
        works={document.works}
        trackingLanguage={document.preferences.trackingLanguage}
        onClose={() => setUpdatesOpen(false)}
        onOpenWork={(workId) => {
          onSelectWork(workId);
          setUpdatesOpen(false);
        }}
        onPersistWork={onSaveWork}
        onSetTrackingLanguage={onSetTrackingLanguage}
      />

      {selectedWork && (
        <>
          <CreateChapterDialog
            open={createChapterDialogOpen}
            work={selectedWork}
            onChooseFolder={openManualChapterFolderDialog}
            onChooseArchive={openManualChapterArchiveDialog}
            onChooseDestination={saveProjectDialog}
            onAttachExisting={() => {
              setCreateChapterDialogOpen(false);
              setAttachDialogOpen(true);
            }}
            onClose={() => setCreateChapterDialogOpen(false)}
            onCreate={(input, preparedPages) => onCreateManualChapter(selectedWork.id, input, preparedPages)}
          />
          <AttachProjectDialog
            open={attachDialogOpen}
            work={selectedWork}
            onChooseProject={chooseAttachment}
            onClose={() => setAttachDialogOpen(false)}
            onConfirm={(draft) => onAttachChapter(selectedWork.id, draft)}
          />
        </>
      )}
    </main>
  );
}

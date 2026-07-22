import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, Bell, LoaderCircle } from "lucide-react";
import traduzaiLogoUrl from "../../../traduzaistudiologo.svg";
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
}) {
  const [workQuery, setWorkQuery] = useState("");
  const [chapterQuery, setChapterQuery] = useState("");
  const selectedWork = useMemo(
    () => document.works.find((work) => work.id === document.selectedWorkId) ?? null,
    [document.selectedWorkId, document.works],
  );
  const initialChapterId = selectedWork?.chapters.find((chapter) => chapter.projectPath === initialSelectedChapterPath)?.id ?? null;
  const [selectedChapterId, setSelectedChapterId] = useState<string | null>(initialChapterId);
  const [workDialogOpen, setWorkDialogOpen] = useState(false);
  const [editingWorkId, setEditingWorkId] = useState<string | null>(null);
  const [attachDialogOpen, setAttachDialogOpen] = useState(false);
  const [createChapterDialogOpen, setCreateChapterDialogOpen] = useState(false);
  const [linkWorkId, setLinkWorkId] = useState<string | null>(null);
  const [updatesOpen, setUpdatesOpen] = useState(false);
  const [missingProjectPaths, setMissingProjectPaths] = useState<Set<string>>(new Set());
  const linkingWork = document.works.find((work) => work.id === linkWorkId) ?? null;
  useEffect(() => {
    setSelectedChapterId(
      selectedWork?.chapters.find((chapter) => chapter.projectPath === initialSelectedChapterPath)?.id ?? null,
    );
    setChapterQuery("");
  }, [initialSelectedChapterPath, selectedWork?.id]);

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
      <div className="studio-library-topbar">
        <img src={traduzaiLogoUrl} alt="TraduzAI Studio" />
        <span>Biblioteca local</span>
        {status === "saving" && <small>Salvando catálogo…</small>}
      </div>

      <div className="studio-library-layout">
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

        <section className="studio-library-main">
          <LibraryToolbar
            title={selectedWork?.title ?? "Nenhuma obra selecionada"}
            chapterCount={selectedWork?.chapters.length ?? 0}
            query={chapterQuery}
            view={document.preferences.chapterView}
            thumbnailSize={document.preferences.thumbnailSize}
            onQueryChange={setChapterQuery}
            onSetView={onSetChapterView}
            onSetThumbnailSize={onSetThumbnailSize}
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

          <div className="flex justify-end border-b border-zinc-800 px-5 py-2">
            <button type="button" className="inline-flex items-center gap-2 rounded border border-zinc-700 px-3 py-1.5 text-sm text-zinc-300 hover:bg-zinc-800" onClick={() => setUpdatesOpen(true)}>
              <Bell size={14} /> Atualizações
            </button>
          </div>

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

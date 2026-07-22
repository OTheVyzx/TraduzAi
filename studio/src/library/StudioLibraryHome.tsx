import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, LoaderCircle } from "lucide-react";
import traduzaiLogoUrl from "../../../traduzaistudiologo.svg";
import {
  openCoverImageDialog,
  openProjectForAttachment,
  projectPathExists,
} from "../backend/projectDialog";
import type { AddLibraryWorkInput, LibraryStoreStatus } from "../store/libraryStore";
import { AttachProjectDialog, type ProjectAttachmentDraft } from "./AttachProjectDialog";
import type { StudioLibrary } from "./libraryModel";
import { ChapterBrowser } from "./ChapterBrowser";
import { LibraryToolbar } from "./LibraryToolbar";
import { WorkDialog } from "./WorkDialog";
import { WorkLibrarySidebar } from "./WorkLibrarySidebar";

export function StudioLibraryHome({
  document,
  status,
  error,
  recoveryAvailable = false,
  onRecover,
  onDismissRecovery,
  onSaveWork,
  onRemoveWork,
  onAttachChapter,
  onRemoveChapter,
  onRelinkChapter,
  onImportProject,
  onSelectWork,
  onOpenChapter,
  onSetChapterView,
  onSetThumbnailSize,
}: {
  document: StudioLibrary;
  status: LibraryStoreStatus;
  error?: string | null;
  recoveryAvailable?: boolean;
  onRecover?: () => void;
  onDismissRecovery?: () => void;
  onSaveWork: (input: AddLibraryWorkInput) => void | Promise<void>;
  onRemoveWork: (workId: string) => void | Promise<void>;
  onAttachChapter: (workId: string, draft: ProjectAttachmentDraft) => void | Promise<void>;
  onRemoveChapter: (workId: string, chapterId: string) => void | Promise<void>;
  onRelinkChapter: (workId: string, chapterId: string, projectPath: string) => void | Promise<void>;
  onImportProject: () => void;
  onSelectWork: (workId: string) => void;
  onOpenChapter: (projectPath: string) => void;
  onSetChapterView: (view: "grid" | "list") => void;
  onSetThumbnailSize: (size: number) => void;
}) {
  const [workQuery, setWorkQuery] = useState("");
  const [chapterQuery, setChapterQuery] = useState("");
  const [selectedChapterId, setSelectedChapterId] = useState<string | null>(null);
  const [workDialogOpen, setWorkDialogOpen] = useState(false);
  const [editingWorkId, setEditingWorkId] = useState<string | null>(null);
  const [attachDialogOpen, setAttachDialogOpen] = useState(false);
  const [missingProjectPaths, setMissingProjectPaths] = useState<Set<string>>(new Set());
  const selectedWork = useMemo(
    () => document.works.find((work) => work.id === document.selectedWorkId) ?? null,
    [document.selectedWorkId, document.works],
  );

  useEffect(() => {
    setSelectedChapterId(null);
    setChapterQuery("");
  }, [selectedWork?.id]);

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
              onAddChapter={() => setAttachDialogOpen(true)}
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
      />

      {selectedWork && (
        <AttachProjectDialog
          open={attachDialogOpen}
          work={selectedWork}
          onChooseProject={chooseAttachment}
          onClose={() => setAttachDialogOpen(false)}
          onConfirm={(draft) => onAttachChapter(selectedWork.id, draft)}
        />
      )}
    </main>
  );
}

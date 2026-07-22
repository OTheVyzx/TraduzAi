import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, LoaderCircle } from "lucide-react";
import traduzaiLogoUrl from "../../../traduzaistudiologo.svg";
import type { LibraryStoreStatus } from "../store/libraryStore";
import type { StudioLibrary } from "./libraryModel";
import { ChapterBrowser } from "./ChapterBrowser";
import { LibraryToolbar } from "./LibraryToolbar";
import { WorkLibrarySidebar } from "./WorkLibrarySidebar";

export function StudioLibraryHome({
  document,
  status,
  error,
  recoveryAvailable = false,
  onRecover,
  onDismissRecovery,
  onAddWork,
  onAddChapter,
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
  onAddWork: () => void;
  onAddChapter: (workId: string) => void;
  onImportProject: () => void;
  onSelectWork: (workId: string) => void;
  onOpenChapter: (projectPath: string) => void;
  onSetChapterView: (view: "grid" | "list") => void;
  onSetThumbnailSize: (size: number) => void;
}) {
  const [workQuery, setWorkQuery] = useState("");
  const [chapterQuery, setChapterQuery] = useState("");
  const [selectedChapterId, setSelectedChapterId] = useState<string | null>(null);
  const selectedWork = useMemo(
    () => document.works.find((work) => work.id === document.selectedWorkId) ?? null,
    [document.selectedWorkId, document.works],
  );

  useEffect(() => {
    setSelectedChapterId(null);
    setChapterQuery("");
  }, [selectedWork?.id]);

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
          onAddWork={onAddWork}
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
              onAddChapter={() => selectedWork && onAddChapter(selectedWork.id)}
            />
          )}

          {error && <p className="studio-home-error">{error}</p>}
        </section>
      </div>
    </main>
  );
}

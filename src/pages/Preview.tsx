import {
  useEffect,
  useRef,
  useState,
  type MouseEvent as ReactMouseEvent,
  type WheelEvent as ReactWheelEvent,
} from "react";
import { useNavigate } from "react-router-dom";
import {
  AlertTriangle,
  ArrowLeft,
  ArrowRight,
  ChevronLeft,
  Check,
  Download,
  Edit3,
  Eye,
  EyeOff,
  FileText,
  LocateFixed,
  Minus,
  Plus,
  RotateCw,
  ShieldPlus,
} from "lucide-react";
import { readFile } from "@tauri-apps/plugin-fs";
import { useAppStore } from "../lib/stores/appStore";
import {
  buildQaReviewSummary,
  collectIgnoredQaActions,
  collectQaIssues,
  ignoreQaIssue,
  qaIssueGroup,
  type QaIssue,
} from "../lib/qaPanel";
import { getStaleRenderPreviewPages, useEditorStore } from "../lib/stores/editorStore";
import {
  exportPagePsd,
  exportProject,
  exportTextFile,
  openExportDialog,
  openLogSaveDialog,
} from "../lib/tauri";
import {
  getDraggedPreviewPan,
  getNextPreviewZoom,
  getPreviewWheelState,
  PREVIEW_ZOOM_DEFAULT,
  type PreviewPanOffset,
  type PreviewPanSession,
} from "./previewZoom";
import { getPreviewImageCandidates, getPreviewToggleLabel } from "./previewImage";
import { EXPORT_MODE_OPTIONS, exportBlockReason, exportModeForBackend, type ExportMode } from "../lib/exportModes";

export function Preview() {
  const navigate = useNavigate();
  const { project, updateProject } = useAppStore();
  const renderPreviewCacheByPageKey = useEditorStore((s) => s.renderPreviewCacheByPageKey);
  const [currentPage, setCurrentPage] = useState(0);
  const [showOriginal, setShowOriginal] = useState(false);
  const [exportFormat, setExportFormat] = useState<"zip_full" | "jpg_only" | "cbz" | "psd">("zip_full");
  const [exportMode, setExportMode] = useState<ExportMode>("clean");
  const [exporting, setExporting] = useState(false);
  const [showExportPanel, setShowExportPanel] = useState(false);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [zoom, setZoom] = useState(PREVIEW_ZOOM_DEFAULT);
  const [panOffset, setPanOffset] = useState<PreviewPanOffset>({ x: 0, y: 0 });
  const [panSession, setPanSession] = useState<PreviewPanSession | null>(null);
  const [isSpacePressed, setIsSpacePressed] = useState(false);
  const [ignoreIssueId, setIgnoreIssueId] = useState<string | null>(null);
  const [ignoreReason, setIgnoreReason] = useState("");
  const [ignoreError, setIgnoreError] = useState<string | null>(null);
  const [lastIgnoredReason, setLastIgnoredReason] = useState<string | null>(null);
  const [exportBlockMessage, setExportBlockMessage] = useState<string | null>(null);
  const prevBlobRef = useRef<string | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);

  const totalPages = project?.paginas.length || 0;
  const page = project?.paginas[currentPage] ?? null;
  const staleRenderPages = getStaleRenderPreviewPages(project, renderPreviewCacheByPageKey);
  const qaIssues = collectQaIssues(project);
  const ignoredQaActions = collectIgnoredQaActions(project);
  const qaReviewSummary = buildQaReviewSummary(project);
  const activeIgnoreIssue = qaIssues.find((issue) => issue.id === ignoreIssueId) ?? null;
  const qaGroups = Object.entries(qaReviewSummary.groups);

  useEffect(() => {
    if (!page) {
      if (prevBlobRef.current) {
        URL.revokeObjectURL(prevBlobRef.current);
        prevBlobRef.current = null;
      }
      setImageSrc(null);
      return;
    }

    let cancelled = false;
    const candidatePaths = getPreviewImageCandidates(page, showOriginal);

    const loadImage = (path: string) =>
      readFile(path).then((bytes) => {
        if (cancelled) return;
        const blob = new Blob([bytes], { type: "image/jpeg" });
        const url = URL.createObjectURL(blob);
        if (prevBlobRef.current) URL.revokeObjectURL(prevBlobRef.current);
        prevBlobRef.current = url;
        setImageSrc(url);
      });

    const loadFirstAvailable = async () => {
      for (const candidatePath of candidatePaths) {
        try {
          await loadImage(candidatePath);
          return;
        } catch {
          if (cancelled) return;
        }
      }

      if (!cancelled) {
        setImageSrc(null);
      }
    };

    void loadFirstAvailable();

    return () => {
      cancelled = true;
    };
  }, [page, showOriginal]);

  useEffect(() => {
    return () => {
      if (prevBlobRef.current) {
        URL.revokeObjectURL(prevBlobRef.current);
      }
    };
  }, []);

  useEffect(() => {
    setZoom(PREVIEW_ZOOM_DEFAULT);
    setPanOffset({ x: 0, y: 0 });
    setPanSession(null);
  }, [currentPage, showOriginal]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const active = document.activeElement;
      const isTyping =
        !!active &&
        (active.tagName === "INPUT" || active.tagName === "TEXTAREA" || active.tagName === "SELECT");

      if (event.key === " ") {
        if (isTyping) return;
        event.preventDefault();
        setIsSpacePressed(true);
        return;
      }

      if (isTyping) return;

      if (event.key === "=" || event.key === "+") {
        event.preventDefault();
        setZoom((currentZoom) => getNextPreviewZoom(currentZoom, "in"));
      }
      if (event.key === "-") {
        event.preventDefault();
        setZoom((currentZoom) => getNextPreviewZoom(currentZoom, "out"));
      }
      if (event.key === "0") {
        event.preventDefault();
        resetPreviewView();
      }
    };

    const handleKeyUp = (event: KeyboardEvent) => {
      if (event.key === " ") setIsSpacePressed(false);
    };

    const handleBlur = () => {
      setIsSpacePressed(false);
      setPanSession(null);
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    window.addEventListener("blur", handleBlur);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
      window.removeEventListener("blur", handleBlur);
    };
  }, []);

  useEffect(() => {
    if (!panSession) return;

    const handleMouseMove = (event: MouseEvent) => {
      event.preventDefault();
      setPanOffset(getDraggedPreviewPan(panSession, event));
    };

    const handleMouseUp = () => setPanSession(null);

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);

    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [panSession]);

  function resetPreviewView() {
    setZoom(PREVIEW_ZOOM_DEFAULT);
    setPanOffset({ x: 0, y: 0 });
    setPanSession(null);
  }

  function beginPan(clientX: number, clientY: number) {
    setPanSession({
      startX: clientX,
      startY: clientY,
      originX: panOffset.x,
      originY: panOffset.y,
    });
  }

  function handleViewportWheel(event: ReactWheelEvent<HTMLDivElement>) {
    event.preventDefault();

    const nextState = getPreviewWheelState({
      zoom,
      pan: panOffset,
      deltaX: event.deltaX,
      deltaY: event.deltaY,
      withZoomModifier: event.ctrlKey || event.metaKey,
    });

    setZoom(nextState.zoom);
    setPanOffset(nextState.pan);
  }

  function handleViewportMouseDown(event: ReactMouseEvent<HTMLDivElement>) {
    if (event.button === 1 || (event.button === 0 && isSpacePressed)) {
      event.preventDefault();
      beginPan(event.clientX, event.clientY);
    }
  }

  async function handleExport(options: { mode?: ExportMode } = {}) {
    if (!project) return;
    setExportBlockMessage(null);
    const activeMode = options.mode ?? exportMode;
    const blockReason = exportBlockReason(activeMode, qaReviewSummary);
    if (blockReason) {
      setExportBlockMessage(blockReason);
      return;
    }
    if (staleRenderPages.length > 0) {
      alert(
        `Preview final desatualizado nas paginas: ${staleRenderPages.join(", ")}. ` +
          "Abra o editor e use Salvar+Render para renderizar antes de exportar.",
      );
      return;
    }
    setExporting(true);

    try {
      const outputPath = await openExportDialog(exportFormat);
      if (!outputPath) return;

      if (exportFormat === "psd") {
        const projectPath = project.output_path ?? project.source_path;
        const baseOutputPath = outputPath.replace(/[/\\][^/\\]+$/, "");

        for (let i = 0; i < project.paginas.length; i++) {
          const currentProjectPage = project.paginas[i];
          const fileName =
            currentProjectPage.arquivo_original
              .split(/[/\\]/)
              .pop()
              ?.replace(/\.\w+$/, ".psd") || `pg-${currentProjectPage.numero}.psd`;
          const finalPath = `${baseOutputPath}/${fileName}`.replace(/\\/g, "/");

          await exportPagePsd({
            project_path: projectPath,
            page_index: i,
            output_path: finalPath,
          });
        }
      } else {
        await exportProject({
          project_path: project.output_path ?? project.source_path,
          format: exportFormat,
          output_path: outputPath,
          export_mode: exportModeForBackend(activeMode),
        });
      }

      alert("Exportacao concluida!");
    } catch (err) {
      console.error("Erro na exportacao:", err);
      alert(`Erro ao exportar: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setExporting(false);
    }
  }

  async function handleExportLog() {
    if (!project || !project.output_path) return;

    try {
      const logPath = `${project.output_path}/pipeline.log`.replace(/\\/g, "/");
      const contents = await readFile(logPath);
      const text = new TextDecoder().decode(contents);
      const savePath = await openLogSaveDialog(`log-${project.obra}-${project.capitulo}.log`);
      if (!savePath) return;

      await exportTextFile(savePath, text);
      alert("Log exportado com sucesso!");
    } catch (err) {
      console.error("Erro ao exportar log:", err);
      alert("O arquivo de log ainda nao foi gerado ou nao pode ser lido.");
    }
  }

  function goToQaIssue(issue: QaIssue) {
    setCurrentPage(issue.pageIndex);
  }

  function startIgnoreIssue(issue: QaIssue) {
    setIgnoreIssueId(issue.id);
    setIgnoreReason("");
    setIgnoreError(null);
  }

  function confirmIgnoreIssue() {
    if (!project || !ignoreIssueId) return;
    try {
      const updatedProject = ignoreQaIssue(project, ignoreIssueId, ignoreReason);
      updateProject({ paginas: updatedProject.paginas });
      setLastIgnoredReason(ignoreReason.trim());
      setIgnoreIssueId(null);
      setIgnoreReason("");
      setIgnoreError(null);
    } catch (err) {
      setIgnoreError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleExportQaReport() {
    if (!project) return;
    const report = [
      `# QA - ${project.obra}`,
      "",
      `Capitulo: ${project.capitulo}`,
      `Paginas: ${qaReviewSummary.totalPages}`,
      `Aprovadas: ${qaReviewSummary.approvedPages}`,
      `Com aviso: ${qaReviewSummary.warningPages}`,
      `Bloqueadas: ${qaReviewSummary.blockedPages}`,
      `Flags ativas: ${qaIssues.length}`,
      `Ignoradas: ${ignoredQaActions.length}`,
      "",
      "## Grupos",
      ...qaGroups.map(([group, count]) => `- ${group}: ${count}`),
      "",
      "## Flags",
      ...qaIssues.map(
        (issue) =>
          `- Pagina ${issue.pageNumber} / ${issue.regionId}: ${issue.label} (${issue.severity}) - ${issue.sourceText}`,
      ),
      "",
      "## Acoes do usuario",
      ...ignoredQaActions.map(
        (action) => `- ${action.flag_id}: ignorado em ${action.ignored_at ?? "-"} - ${action.ignored_reason ?? "-"}`,
      ),
      "",
    ].join("\n");

    const savePath = await openLogSaveDialog(`qa-${project.obra}-${project.capitulo}.md`);
    if (!savePath) return;
    await exportTextFile(savePath, report);
  }

  const viewportCursor = panSession ? "cursor-grabbing" : isSpacePressed ? "cursor-grab" : "cursor-default";

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-border bg-bg-secondary px-6 py-3">
        <div className="flex items-center gap-3">
          <button
            onClick={() => navigate("/")}
            title="Voltar para o inicio"
            className="p-1.5 text-text-secondary transition-smooth hover:text-text-primary"
          >
            <ChevronLeft size={18} />
          </button>
          <div>
            <p className="text-sm font-medium">{project?.obra}</p>
            <p className="text-xs text-text-secondary">
              Capitulo {project?.capitulo} - {totalPages} paginas
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {staleRenderPages.length > 0 && (
            <span
              className="rounded-full border border-status-warning/25 bg-status-warning/10 px-2.5 py-1 text-[11px] text-status-warning"
              title="Existem paginas com preview final desatualizado."
            >
              Preview final pendente
            </span>
          )}

          <button
            onClick={() => setShowOriginal(!showOriginal)}
            title={getPreviewToggleLabel(showOriginal)}
            className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs transition-smooth ${
              showOriginal
                ? "border border-status-warning/20 bg-status-warning/10 text-status-warning"
                : "border border-border bg-bg-tertiary text-text-secondary"
            }`}
          >
            {showOriginal ? <EyeOff size={14} /> : <Eye size={14} />}
            {getPreviewToggleLabel(showOriginal)}
          </button>

          <button
            onClick={() => navigate("/editor")}
            className="flex items-center gap-1.5 rounded-lg bg-brand px-3 py-1.5 text-xs font-medium text-white transition-smooth hover:bg-brand-600"
          >
            Abrir Editor
          </button>

          <button
            data-testid="export-panel-toggle"
            onClick={() => setShowExportPanel(!showExportPanel)}
            className="flex items-center gap-1.5 rounded-lg bg-brand/10 px-3 py-1.5 text-xs text-brand-300 transition-smooth hover:bg-brand/20"
          >
            <Download size={14} />
            Exportar
          </button>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        <div
          className={`relative flex-1 overflow-hidden bg-[radial-gradient(circle_at_top,_rgba(72,176,255,0.08),_transparent_38%),linear-gradient(180deg,_rgba(255,255,255,0.02),_transparent_28%)] ${viewportCursor}`}
          onWheel={handleViewportWheel}
          onMouseDown={handleViewportMouseDown}
        >
          {page && imageSrc ? (
            <div className="absolute right-4 top-4 z-10 flex items-center gap-1.5 rounded-xl border border-border bg-bg-secondary/90 px-2 py-2 shadow-lg backdrop-blur">
              <button
                onClick={() => setZoom((currentZoom) => getNextPreviewZoom(currentZoom, "out"))}
                className="rounded-xl bg-bg-tertiary p-1.5 text-text-secondary transition-smooth hover:text-text-primary"
                title="Diminuir zoom (-)"
              >
                <Minus size={14} />
              </button>
              <button
                onClick={() => setZoom((currentZoom) => getNextPreviewZoom(currentZoom, "in"))}
                className="rounded-xl bg-bg-tertiary p-1.5 text-text-secondary transition-smooth hover:text-text-primary"
                title="Aumentar zoom (+)"
              >
                <Plus size={14} />
              </button>
              <button
                onClick={resetPreviewView}
                className="rounded-xl bg-bg-tertiary px-2 py-1.5 text-[11px] text-text-secondary transition-smooth hover:text-text-primary"
                title="Resetar zoom e posicao (0)"
              >
                Ajustar
              </button>
              <button
                onClick={() => setZoom(2)}
                className="rounded-xl bg-bg-tertiary px-2 py-1.5 text-[11px] text-text-secondary transition-smooth hover:text-text-primary"
                title="Zoom 2x"
              >
                2x
              </button>
              <button
                onClick={() => setPanOffset({ x: 0, y: 0 })}
                className="rounded-xl bg-bg-tertiary p-1.5 text-text-secondary transition-smooth hover:text-text-primary"
                title="Centralizar (pan)"
              >
                <LocateFixed size={14} />
              </button>
              <span className="w-12 text-right font-mono text-[11px] text-text-muted">
                {Math.round(zoom * 100)}%
              </span>
            </div>
          ) : null}

          <div className="pointer-events-none absolute inset-x-0 bottom-3 z-20 flex justify-center px-4">
            <div className="rounded-full border border-border bg-black/45 px-3 py-1 text-[11px] text-text-secondary backdrop-blur">
              Ctrl+scroll: zoom • Scroll: mover • Space+drag: mover
            </div>
          </div>

          <div className="flex h-full items-center justify-center overflow-hidden px-6 py-4">
            <div
              className="will-change-transform"
              style={{
                transform: `translate(${panOffset.x}px, ${panOffset.y}px) scale(${zoom})`,
                transformOrigin: "center center",
                transition: panSession ? "none" : "transform 0.12s ease-out",
              }}
            >
              {page && imageSrc ? (
                <div className="relative inline-block">
                  <img
                    ref={imgRef}
                    src={imageSrc}
                    alt={`Pagina ${page.numero}`}
                    draggable={false}
                    className="max-h-[calc(100vh-180px)] w-auto select-none rounded-2xl border border-border object-contain shadow-[0_20px_60px_rgba(0,0,0,0.55)]"
                  />
                </div>
              ) : page ? (
                <p className="text-sm text-text-secondary">Carregando imagem...</p>
              ) : (
                <p className="text-text-secondary">Nenhuma pagina para exibir</p>
              )}
            </div>
          </div>
        </div>

        <aside data-testid="qa-panel" className="w-80 overflow-y-auto border-l border-border bg-bg-secondary p-5">
          <div className="mb-4 flex items-start justify-between gap-3">
            <div>
              <h3 className="text-sm font-medium">Relatorio do capitulo</h3>
              <p className="mt-1 text-xs text-text-secondary">Revisao profissional antes do export.</p>
            </div>
            <span
              data-testid="qa-issue-count"
              className={`rounded-full px-2 py-1 font-mono text-xs ${
                qaIssues.length > 0
                  ? "bg-status-warning/10 text-status-warning"
                  : "bg-status-success/10 text-status-success"
              }`}
            >
              {qaIssues.length}
            </span>
          </div>

          <div data-testid="qa-review-report" className="mb-4 rounded-xl border border-border bg-bg-tertiary/70 p-3">
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <p className="text-text-muted">Paginas</p>
                <p className="text-sm font-medium text-text-primary">{qaReviewSummary.totalPages}</p>
              </div>
              <div>
                <p className="text-text-muted">Aprovadas</p>
                <p className="text-sm font-medium text-status-success">{qaReviewSummary.approvedPages}</p>
              </div>
              <div>
                <p className="text-text-muted">Com aviso</p>
                <p className="text-sm font-medium text-status-warning">{qaReviewSummary.warningPages}</p>
              </div>
              <div>
                <p className="text-text-muted">Bloqueadas</p>
                <p data-testid="qa-blocked-pages" className="text-sm font-medium text-status-error">{qaReviewSummary.blockedPages}</p>
              </div>
            </div>
            <div className="mt-3 grid grid-cols-2 gap-2 border-t border-border pt-3 text-xs">
              <div>
                <p className="text-text-muted">Criticos</p>
                <p data-testid="qa-critical-count" className="text-sm font-medium text-status-error">{qaReviewSummary.criticalCount}</p>
              </div>
              <div>
                <p className="text-text-muted">Warnings</p>
                <p className="text-sm font-medium text-status-warning">{qaReviewSummary.warningCount}</p>
              </div>
            </div>
          </div>

          <div data-testid="qa-group-list" className="mb-4 space-y-1">
            {qaGroups.length === 0 ? (
              <div className="rounded-lg border border-border bg-bg-tertiary px-3 py-2 text-xs text-text-secondary">
                Sem grupos ativos.
              </div>
            ) : (
              qaGroups.map(([group, count]) => (
                <div key={group} className="flex items-center justify-between rounded-lg border border-border bg-bg-tertiary px-3 py-2 text-xs">
                  <span className="text-text-secondary">{group}</span>
                  <span className="font-mono text-text-primary">{count}</span>
                </div>
              ))
            )}
          </div>

          <div className="space-y-2">
            {qaIssues.length === 0 ? (
              <div className="rounded-lg border border-border bg-bg-tertiary p-3 text-xs text-text-secondary">
                Nenhuma flag ativa.
              </div>
            ) : (
              qaIssues.map((issue) => (
                <div key={issue.id} className="rounded-lg border border-border bg-bg-tertiary p-3">
                  <button
                    data-testid="qa-flag-item"
                    onClick={() => goToQaIssue(issue)}
                    className="flex w-full items-start gap-2 text-left"
                  >
                    <AlertTriangle
                      size={16}
                      className={
                        issue.severity === "critical" || issue.severity === "high"
                          ? "mt-0.5 text-status-error"
                          : "mt-0.5 text-status-warning"
                      }
                    />
                      <span className="min-w-0 flex-1">
                        <span className="block text-sm font-medium text-text-primary">{issue.label}</span>
                        <span className="mt-1 inline-flex rounded-md bg-bg-primary px-2 py-0.5 text-[10px] text-text-muted">
                          {qaIssueGroup(issue.flagId)}
                        </span>
                        <span className="mt-1 block text-xs text-text-secondary">
                          Pagina {issue.pageNumber} - regiao {issue.regionId}
                      </span>
                      <span className="mt-1 block truncate text-[11px] text-text-muted">{issue.sourceText}</span>
                    </span>
                  </button>

                  <div className="mt-3 grid grid-cols-2 gap-2">
                    <button
                      onClick={() => goToQaIssue(issue)}
                      className="flex items-center justify-center gap-1 rounded-md border border-border px-2 py-1.5 text-[11px] text-text-secondary transition-smooth hover:text-text-primary"
                    >
                      <LocateFixed size={12} />
                      Ir para pagina
                    </button>
                    <button
                      onClick={() => navigate("/editor")}
                      className="flex items-center justify-center gap-1 rounded-md border border-border px-2 py-1.5 text-[11px] text-text-secondary transition-smooth hover:text-text-primary"
                    >
                      <Edit3 size={12} />
                      Corrigir texto
                    </button>
                    <button
                      onClick={() => navigate("/setup")}
                      className="flex items-center justify-center gap-1 rounded-md border border-border px-2 py-1.5 text-[11px] text-text-secondary transition-smooth hover:text-text-primary"
                    >
                      <ShieldPlus size={12} />
                      Glossario
                    </button>
                    <button
                      onClick={() => alert("Regiao marcada para reprocessamento.")}
                      className="flex items-center justify-center gap-1 rounded-md border border-border px-2 py-1.5 text-[11px] text-text-secondary transition-smooth hover:text-text-primary"
                    >
                      <RotateCw size={12} />
                      Reprocessar
                    </button>
                    <button
                      onClick={() => alert("Mascara marcada para regeneracao.")}
                      className="flex items-center justify-center gap-1 rounded-md border border-border px-2 py-1.5 text-[11px] text-text-secondary transition-smooth hover:text-text-primary"
                    >
                      <RotateCw size={12} />
                      Mascara
                    </button>
                  </div>

                  {activeIgnoreIssue?.id === issue.id ? (
                    <div className="mt-3 space-y-2">
                      <textarea
                        data-testid="qa-ignore-reason"
                        value={ignoreReason}
                        onChange={(event) => {
                          setIgnoreReason(event.target.value);
                          setIgnoreError(null);
                        }}
                        placeholder="Motivo para ignorar"
                        className="min-h-[72px] w-full rounded-md border border-border bg-bg-primary px-2 py-2 text-xs text-text-primary outline-none focus:border-brand/50"
                      />
                      {ignoreError && <p className="text-xs text-status-error">{ignoreError}</p>}
                      <div className="flex gap-2">
                        <button
                          data-testid="qa-save-ignore"
                          onClick={confirmIgnoreIssue}
                          disabled={ignoreReason.trim().length === 0}
                          className="flex flex-1 items-center justify-center gap-1 rounded-md bg-brand px-2 py-1.5 text-[11px] font-medium text-white transition-smooth hover:bg-brand-600 disabled:opacity-40"
                        >
                          <Check size={12} />
                          Salvar motivo
                        </button>
                        <button
                          onClick={() => setIgnoreIssueId(null)}
                          className="rounded-md border border-border px-2 py-1.5 text-[11px] text-text-secondary transition-smooth hover:text-text-primary"
                        >
                          Cancelar
                        </button>
                      </div>
                    </div>
                  ) : (
                    <button
                      data-testid="qa-ignore-button"
                      onClick={() => startIgnoreIssue(issue)}
                      className="mt-3 w-full rounded-md border border-status-warning/25 bg-status-warning/10 px-2 py-1.5 text-[11px] font-medium text-status-warning transition-smooth hover:bg-status-warning/15"
                    >
                      Ignorar com motivo
                    </button>
                  )}
                </div>
              ))
            )}
          </div>

          <button
            onClick={handleExportQaReport}
            className="mt-4 flex w-full items-center justify-center gap-2 rounded-lg border border-border bg-bg-tertiary py-2 text-xs font-medium text-text-secondary transition-smooth hover:bg-white/[0.03] hover:text-text-primary"
          >
            <FileText size={14} />
            Exportar relatorio
          </button>

          {(lastIgnoredReason || ignoredQaActions.length > 0) && (
            <div className="mt-4 rounded-lg border border-status-success/20 bg-status-success/10 p-3 text-xs text-status-success">
              {lastIgnoredReason ?? ignoredQaActions[ignoredQaActions.length - 1]?.ignored_reason}
            </div>
          )}
        </aside>

        {showExportPanel && (
          <div className="w-72 space-y-4 border-l border-border bg-bg-secondary p-5">
            <h3 className="text-sm font-medium">Exportar projeto</h3>

            <div className="space-y-2">
              {(
                [
                  { value: "zip_full", label: "ZIP completo", desc: "Originais + traduzidas + project.json" },
                  { value: "jpg_only", label: "Somente traduzidas", desc: "Apenas as imagens traduzidas" },
                  { value: "cbz", label: "CBZ", desc: "Formato de leitor de manga" },
                  { value: "psd", label: "Photoshop (PSD)", desc: "Camadas separadas: Original, Inpaint, Texto" },
                ] as const
              ).map((option) => (
                <button
                  key={option.value}
                  onClick={() => setExportFormat(option.value)}
                  className={`w-full rounded-lg border p-3 text-left transition-smooth ${
                    exportFormat === option.value
                      ? "border-brand/25 bg-brand/5"
                      : "border-border hover:border-border"
                  }`}
                >
                  <p className="text-sm font-medium">{option.label}</p>
                  <p className="mt-0.5 text-xs text-text-secondary">{option.desc}</p>
                </button>
              ))}
            </div>

            <div data-testid="export-mode-options" className="space-y-2">
              {EXPORT_MODE_OPTIONS.map((option) => (
                <button
                  key={option.id}
                  data-testid={`export-mode-${option.id}`}
                  type="button"
                  onClick={() => setExportMode(option.id)}
                  className={`w-full rounded-lg border p-3 text-left transition-smooth ${
                    exportMode === option.id
                      ? "border-brand/25 bg-brand/5"
                      : "border-border hover:border-border"
                  }`}
                >
                  <p className="text-sm font-medium">{option.label}</p>
                  <p className="mt-0.5 text-xs text-text-secondary">{option.description}</p>
                </button>
              ))}
            </div>

            <div className="space-y-2 pt-2">
              {exportBlockMessage && (
                <div data-testid="export-block-message" className="rounded-lg border border-status-error/25 bg-status-error/10 px-3 py-2 text-xs text-status-error">
                  {exportBlockMessage}
                </div>
              )}
              <button
                data-testid="export-button"
                onClick={() => handleExport()}
                disabled={exporting}
                className="w-full rounded-lg bg-brand py-2.5 text-sm font-medium text-white transition-smooth hover:bg-brand-600 disabled:opacity-50"
              >
                {exporting ? "Exportando..." : "Exportar limpo"}
              </button>

              <button
                data-testid="export-with-warnings-button"
                onClick={() => handleExport({ mode: "with_warnings" })}
                disabled={exporting}
                className="w-full rounded-lg border border-status-warning/25 bg-status-warning/10 py-2 text-xs font-medium text-status-warning transition-smooth hover:bg-status-warning/15 disabled:opacity-50"
              >
                Exportar com avisos
              </button>

              <button
                data-testid="export-debug-button"
                onClick={() => handleExport({ mode: "debug" })}
                disabled={exporting}
                className="w-full rounded-lg border border-border bg-bg-tertiary py-2 text-xs font-medium text-text-secondary transition-smooth hover:text-text-primary disabled:opacity-50"
              >
                Exportar debug
              </button>

              <button
                data-testid="export-report-link"
                onClick={handleExportLog}
                className="flex w-full items-center justify-center gap-2 rounded-lg border border-border bg-bg-tertiary py-2 text-xs font-medium text-text-secondary transition-smooth hover:bg-white/[0.03] hover:text-text-primary"
              >
                <FileText size={14} />
                Exportar Log do Pipeline
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="flex items-center justify-center gap-4 border-t border-border bg-bg-secondary px-6 py-3">
        <button
          onClick={() => setCurrentPage(Math.max(0, currentPage - 1))}
          disabled={currentPage === 0}
          title="Pagina anterior"
          className="rounded-lg bg-bg-tertiary p-2 text-text-secondary transition-smooth hover:text-text-primary disabled:opacity-30"
        >
          <ArrowLeft size={16} />
        </button>

        <span data-testid="preview-page-counter" className="min-w-[80px] text-center font-mono text-sm text-text-secondary">
          {currentPage + 1} / {totalPages}
        </span>

        <button
          onClick={() => setCurrentPage(Math.min(totalPages - 1, currentPage + 1))}
          disabled={currentPage >= totalPages - 1}
          title="Proxima pagina"
          className="rounded-lg bg-bg-tertiary p-2 text-text-secondary transition-smooth hover:text-text-primary disabled:opacity-30"
        >
          <ArrowRight size={16} />
        </button>
      </div>
    </div>
  );
}

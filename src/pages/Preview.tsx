import {
  useEffect,
  useRef,
  useState,
  type MouseEvent as ReactMouseEvent,
  type WheelEvent as ReactWheelEvent,
} from "react";
import { useNavigate } from "react-router-dom";
import {
  ArrowLeft,
  ArrowRight,
  ChevronLeft,
  Download,
  Eye,
  EyeOff,
  FileText,
  LocateFixed,
  Minus,
  Plus,
} from "lucide-react";
import { readFile } from "@tauri-apps/plugin-fs";
import { useAppStore } from "../lib/stores/appStore";
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

export function Preview() {
  const navigate = useNavigate();
  const { project } = useAppStore();
  const [currentPage, setCurrentPage] = useState(0);
  const [showOriginal, setShowOriginal] = useState(false);
  const [exportFormat, setExportFormat] = useState<"zip_full" | "jpg_only" | "cbz" | "psd">("zip_full");
  const [exporting, setExporting] = useState(false);
  const [showExportPanel, setShowExportPanel] = useState(false);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [zoom, setZoom] = useState(PREVIEW_ZOOM_DEFAULT);
  const [panOffset, setPanOffset] = useState<PreviewPanOffset>({ x: 0, y: 0 });
  const [panSession, setPanSession] = useState<PreviewPanSession | null>(null);
  const [isSpacePressed, setIsSpacePressed] = useState(false);
  const prevBlobRef = useRef<string | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);

  const totalPages = project?.paginas.length || 0;
  const page = project?.paginas[currentPage] ?? null;

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

  async function handleExport() {
    if (!project) return;
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

            <div className="space-y-2 pt-2">
              <button
                onClick={handleExport}
                disabled={exporting}
                className="w-full rounded-lg bg-brand py-2.5 text-sm font-medium text-white transition-smooth hover:bg-brand-600 disabled:opacity-50"
              >
                {exporting ? "Exportando..." : "Salvar arquivo"}
              </button>

              <button
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

        <span className="min-w-[80px] text-center font-mono text-sm text-text-secondary">
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

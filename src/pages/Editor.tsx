import { useEffect, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import {
  ArrowLeft,
  ArrowRight,
  Brush,
  Check,
  ChevronLeft,
  Eraser,
  Eye,
  EyeOff,
  FileText,
  GripHorizontal,
  Image,
  Layers,
  LocateFixed,
  Minus,
  PenTool,
  Play,
  Plus,
  SquareDashedMousePointer,
  Undo2,
} from "lucide-react";
import { useAppStore } from "../lib/stores/appStore";
import { EditorCanvas } from "../components/editor/EditorCanvas";
import { LayersPanel } from "../components/editor/LayersPanel";
import { PageThumbnails } from "../components/editor/PageThumbnails";
import { useEditorStore, type EditorToolMode } from "../lib/stores/editorStore";

const VIEW_MODES = [
  { key: "original" as const, label: "Original", icon: Image, hotkey: "1" },
  { key: "inpainted" as const, label: "Limpa", icon: Eraser, hotkey: "2" },
  { key: "translated" as const, label: "Camadas", icon: FileText, hotkey: "3" },
];

const TOOL_MODES: { key: EditorToolMode; label: string; icon: typeof PenTool; hotkey: string }[] = [
  { key: "select", label: "Selecionar", icon: PenTool, hotkey: "V" },
  { key: "block", label: "Novo bloco", icon: SquareDashedMousePointer, hotkey: "B" },
  { key: "brush", label: "Brush", icon: Brush, hotkey: "N" },
  { key: "repairBrush", label: "Máscara", icon: Brush, hotkey: "M" },
  { key: "eraser", label: "Borracha", icon: Eraser, hotkey: "E" },
];

export function Editor() {
  const navigate = useNavigate();
  const project = useAppStore((s) => s.project);
  const {
    currentPageIndex,
    currentPage,
    selectedLayerId,
    toolMode,
    viewMode,
    showOverlays,
    zoom,
    brushSize,
    pendingEdits,
    isRetypesetting,
    isReinpainting,
    loadCurrentPage,
    setCurrentPage,
    setToolMode,
    setViewMode,
    toggleOverlays,
    zoomIn,
    zoomOut,
    setZoom,
    resetViewport,
    setPan,
    commitEdits,
    discardEdits,
    deleteSelectedLayer,
    retypesetCurrentPage,
    reinpaintCurrentPage,
    setBrushSize,
  } = useEditorStore();

  const totalPages = project?.paginas.length ?? 0;
  const pendingCount = Object.keys(pendingEdits).length;

  useEffect(() => {
    if (!project || totalPages === 0) return;
    void loadCurrentPage();
  }, [loadCurrentPage, project, totalPages]);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      const active = document.activeElement;
      const isTyping =
        !!active &&
        (active.tagName === "INPUT" || active.tagName === "TEXTAREA" || active.tagName === "SELECT");

      if (!isTyping) {
        if (event.key === "1") setViewMode("original");
        if (event.key === "2") setViewMode("inpainted");
        if (event.key === "3") setViewMode("translated");
        if (event.key.toLowerCase() === "v") setToolMode("select");
        if (event.key.toLowerCase() === "b") setToolMode("block");
        if (event.key.toLowerCase() === "n") setToolMode("brush");
        if (event.key.toLowerCase() === "m") setToolMode("repairBrush");
        if (event.key.toLowerCase() === "e") setToolMode("eraser");
        if (event.key.toLowerCase() === "o") toggleOverlays();
        if (event.key === "=" || event.key === "+") {
          event.preventDefault();
          zoomIn();
        }
        if (event.key === "-") {
          event.preventDefault();
          zoomOut();
        }
        if (event.key === "0") {
          event.preventDefault();
          resetViewport();
        }
      }

      if (event.key === "ArrowLeft" && event.altKey && currentPageIndex > 0) {
        event.preventDefault();
        void setCurrentPage(currentPageIndex - 1);
      }
      if (event.key === "ArrowRight" && event.altKey && currentPageIndex < totalPages - 1) {
        event.preventDefault();
        void setCurrentPage(currentPageIndex + 1);
      }
      if ((event.key === "Delete" || event.key === "Backspace") && selectedLayerId && !isTyping) {
        event.preventDefault();
        void deleteSelectedLayer();
      }
      if (event.key === "s" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        void commitEdits();
      }
    };

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [
    commitEdits,
    currentPageIndex,
    deleteSelectedLayer,
    resetViewport,
    selectedLayerId,
    setCurrentPage,
    setToolMode,
    setViewMode,
    toggleOverlays,
    totalPages,
    zoomIn,
    zoomOut,
  ]);

  const currentPageSummary = useMemo(() => {
    if (!currentPage) return "Carregando página";
    return `${currentPage.text_layers.length} camada(s) de texto`;
  }, [currentPage]);

  if (!project || totalPages === 0) {
    return (
      <div className="flex h-screen items-center justify-center bg-bg-primary">
        <div className="space-y-3 text-center">
          <Layers size={32} className="mx-auto text-text-muted" />
          <p className="text-sm text-text-secondary">Nenhum projeto carregado</p>
          <button
            onClick={() => navigate("/")}
            className="text-sm text-accent-purple hover:underline"
          >
            Voltar ao início
          </button>
        </div>
      </div>
    );
  }

  const applyCurrentView = () => {
    if (viewMode === "inpainted") return reinpaintCurrentPage();
    return retypesetCurrentPage();
  };

  return (
    <div className="flex h-screen bg-bg-primary">
      <PageThumbnails />

      <div className="flex min-w-0 flex-1 flex-col border-r border-white/5">
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/5 bg-[linear-gradient(180deg,rgba(255,255,255,0.04),rgba(255,255,255,0.01))] px-4 py-3">
          <div className="flex min-w-0 items-center gap-3">
            <button
              onClick={() => navigate("/preview")}
              className="rounded p-1.5 text-text-secondary transition-smooth hover:text-text-primary"
              title="Voltar ao preview"
            >
              <ChevronLeft size={18} />
            </button>
            <div className="min-w-0">
              <p className="truncate text-sm font-medium">{project.obra}</p>
              <p className="truncate text-[11px] text-text-secondary">
                Cap. {project.capitulo} | Pag. {currentPageIndex + 1}/{totalPages} | {currentPageSummary}
              </p>
            </div>
            <div
              data-tauri-drag-region
              className="hidden h-6 w-16 shrink-0 cursor-grab items-center justify-center rounded-full border border-white/10 bg-bg-tertiary/70 text-text-muted md:flex"
              title="Arrastar janela"
            >
              <GripHorizontal size={14} />
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-1.5">
            {VIEW_MODES.map(({ key, label, icon: Icon, hotkey }) => (
              <button
                key={key}
                onClick={() => setViewMode(key)}
                className={`flex items-center gap-1 rounded-xl border px-2.5 py-1.5 text-xs transition-smooth ${
                  viewMode === key
                    ? "border-accent-purple/35 bg-accent-purple/15 text-accent-purple"
                    : "border-transparent text-text-secondary hover:text-text-primary"
                }`}
                title={`${label} (${hotkey})`}
              >
                <Icon size={13} />
                {label}
              </button>
            ))}

            <div className="mx-1 h-5 w-px bg-white/5" />

            {TOOL_MODES.map(({ key, label, icon: Icon, hotkey }) => (
              <button
                key={key}
                onClick={() => setToolMode(key)}
                className={`flex items-center gap-1 rounded-xl border px-2.5 py-1.5 text-xs transition-smooth ${
                  toolMode === key
                    ? "border-accent-cyan/35 bg-accent-cyan/12 text-accent-cyan"
                    : "border-transparent text-text-secondary hover:text-text-primary"
                }`}
                title={`${label} (${hotkey})`}
              >
                <Icon size={13} />
                {label}
              </button>
            ))}

            <div className="mx-1 h-5 w-px bg-white/5" />

            <button
              onClick={toggleOverlays}
              className={`flex items-center gap-1 rounded-xl border px-2.5 py-1.5 text-xs transition-smooth ${
                showOverlays
                  ? "border-accent-cyan/25 bg-accent-cyan/10 text-accent-cyan"
                  : "border-transparent text-text-muted"
              }`}
              title="Guias de seleção (O)"
            >
              {showOverlays ? <Eye size={13} /> : <EyeOff size={13} />}
              Guias
            </button>

            {(toolMode === "brush" || toolMode === "repairBrush" || toolMode === "eraser") && (
              <div className="flex items-center gap-2 rounded-xl border border-white/5 bg-bg-tertiary/50 px-2 py-1">
                <span className="text-[11px] text-text-muted">Pincel</span>
                <input
                  type="range"
                  min={4}
                  max={96}
                  value={brushSize}
                  onChange={(event) => setBrushSize(Number(event.target.value))}
                />
                <span className="w-8 text-right text-[11px] text-text-secondary">{brushSize}px</span>
              </div>
            )}
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              onClick={() => void commitEdits()}
              disabled={pendingCount === 0}
              className="flex items-center gap-1 rounded-xl border border-transparent px-2.5 py-1.5 text-xs text-status-success transition-smooth hover:bg-status-success/10 disabled:opacity-35"
              title="Salvar alterações (Ctrl+S)"
            >
              <Check size={13} />
              Salvar
            </button>
            <button
              onClick={discardEdits}
              disabled={pendingCount === 0}
              className="flex items-center gap-1 rounded-xl border border-transparent px-2.5 py-1.5 text-xs text-text-secondary transition-smooth hover:bg-bg-tertiary disabled:opacity-35"
              title="Descartar alterações pendentes"
            >
              <Undo2 size={13} />
              Descartar
            </button>
            {pendingCount > 0 && (
              <span className="rounded-full bg-accent-purple/12 px-2 py-1 text-[11px] text-accent-purple">
                {pendingCount} alteração(ões)
              </span>
            )}

            <div className="mx-1 h-5 w-px bg-white/5" />

            <button
              disabled={isRetypesetting || isReinpainting}
              onClick={() => void applyCurrentView()}
              className="flex items-center gap-1 rounded-xl bg-accent-purple px-2.5 py-1.5 text-xs text-white transition-smooth hover:bg-accent-purple/90 disabled:opacity-50"
              title={viewMode === "inpainted" ? "Refazer página limpa" : "Rerenderizar página"}
            >
              <Play size={13} className={isRetypesetting || isReinpainting ? "animate-pulse" : ""} />
              {viewMode === "inpainted"
                ? isReinpainting
                  ? "Reinpaint..."
                  : "Reinpaint"
                : isRetypesetting
                  ? "Rerender..."
                  : "Rerender"}
            </button>

            <div className="mx-1 h-5 w-px bg-white/5" />

            <button
              onClick={zoomOut}
              className="rounded-xl bg-bg-tertiary p-1.5 text-text-secondary transition-smooth hover:text-text-primary"
              title="Diminuir zoom (-)"
            >
              <Minus size={14} />
            </button>
            <button
              onClick={zoomIn}
              className="rounded-xl bg-bg-tertiary p-1.5 text-text-secondary transition-smooth hover:text-text-primary"
              title="Aumentar zoom (+)"
            >
              <Plus size={14} />
            </button>
            <button
              onClick={resetViewport}
              className="rounded-xl bg-bg-tertiary px-2 py-1.5 text-[11px] text-text-secondary transition-smooth hover:text-text-primary"
              title="Resetar zoom e posição (0)"
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
              onClick={() => setPan({ x: 0, y: 0 })}
              className="rounded-xl bg-bg-tertiary p-1.5 text-text-secondary transition-smooth hover:text-text-primary"
              title="Centralizar (pan)"
            >
              <LocateFixed size={14} />
            </button>
            <span className="w-12 text-right font-mono text-[11px] text-text-muted">
              {Math.round(zoom * 100)}%
            </span>

            <div className="mx-1 h-5 w-px bg-white/5" />

            <button
              onClick={() => void setCurrentPage(Math.max(0, currentPageIndex - 1))}
              disabled={currentPageIndex === 0}
              className="rounded-xl bg-bg-tertiary p-1.5 text-text-secondary transition-smooth hover:text-text-primary disabled:opacity-30"
              title="Página anterior (Alt+Left)"
            >
              <ArrowLeft size={14} />
            </button>
            <span className="min-w-[56px] text-center font-mono text-xs text-text-secondary">
              {currentPageIndex + 1}/{totalPages}
            </span>
            <button
              onClick={() => void setCurrentPage(Math.min(totalPages - 1, currentPageIndex + 1))}
              disabled={currentPageIndex >= totalPages - 1}
              className="rounded-xl bg-bg-tertiary p-1.5 text-text-secondary transition-smooth hover:text-text-primary disabled:opacity-30"
              title="Próxima página (Alt+Right)"
            >
              <ArrowRight size={14} />
            </button>
          </div>
        </div>

        <EditorCanvas />
      </div>

      <LayersPanel />
    </div>
  );
}

import { useEffect, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import {
  ChevronLeft,
  ArrowLeft,
  ArrowRight,
  Eye,
  EyeOff,
  Image,
  Eraser,
  FileText,
  Layers,
  Play,
  Minus,
  Plus,
  LocateFixed,
  Check,
  Undo2,
  GripHorizontal,
} from "lucide-react";
import { useAppStore } from "../lib/stores/appStore";
import { useEditorStore } from "../lib/stores/editorStore";
import { EditorCanvas } from "../components/editor/EditorCanvas";
import { LayersPanel } from "../components/editor/LayersPanel";
import { PageThumbnails } from "../components/editor/PageThumbnails";

const VIEW_MODES = [
  { key: "original" as const, label: "Original", icon: Image, hotkey: "1" },
  { key: "inpainted" as const, label: "Limpa", icon: Eraser, hotkey: "2" },
  { key: "translated" as const, label: "Traduzida", icon: FileText, hotkey: "3" },
];

export function Editor() {
  const navigate = useNavigate();
  const project = useAppStore((s) => s.project);
  const {
    currentPageIndex,
    viewMode,
    showOverlays,
    zoom,
    pendingEdits,
  } = useEditorStore();
  const setCurrentPage = useEditorStore((s) => s.setCurrentPage);
  const setViewMode = useEditorStore((s) => s.setViewMode);
  const toggleOverlays = useEditorStore((s) => s.toggleOverlays);
  const zoomIn = useEditorStore((s) => s.zoomIn);
  const zoomOut = useEditorStore((s) => s.zoomOut);
  const setZoom = useEditorStore((s) => s.setZoom);
  const resetViewport = useEditorStore((s) => s.resetViewport);
  const commitEdits = useEditorStore((s) => s.commitEdits);
  const discardEdits = useEditorStore((s) => s.discardEdits);
  const isRetypesetting = useEditorStore((s) => s.isRetypesetting);
  const isReinpainting = useEditorStore((s) => s.isReinpainting);
  const retypesetCurrentPage = useEditorStore((s) => s.retypesetCurrentPage);
  const reinpaintCurrentPage = useEditorStore((s) => s.reinpaintCurrentPage);

  const totalPages = project?.paginas.length ?? 0;
  const page = project?.paginas[currentPageIndex] ?? null;
  const pendingCount = Object.keys(pendingEdits).length;

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
        if (event.key === "o") toggleOverlays();
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

      if (event.key === "ArrowLeft" && event.altKey) {
        event.preventDefault();
        if (currentPageIndex > 0) setCurrentPage(currentPageIndex - 1);
      }
      if (event.key === "ArrowRight" && event.altKey) {
        event.preventDefault();
        if (currentPageIndex < totalPages - 1) setCurrentPage(currentPageIndex + 1);
      }
      if (event.key === "Escape") {
        useEditorStore.getState().selectLayer(null);
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
    resetViewport,
    setCurrentPage,
    setViewMode,
    toggleOverlays,
    totalPages,
    zoomIn,
    zoomOut,
  ]);

  const currentPageSummary = useMemo(() => {
    if (!page) return "Nenhuma pagina carregada";
    return `${page.textos.length} bloco(s) de texto`;
  }, [page]);

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
            Voltar ao inicio
          </button>
        </div>
      </div>
    );
  }

  const canApplyCurrentView = viewMode === "translated" || viewMode === "inpainted";
  const isApplyingCurrentView = isRetypesetting || isReinpainting;
  const applyCurrentView = () => {
    if (viewMode === "inpainted") {
      return reinpaintCurrentPage();
    }
    return retypesetCurrentPage();
  };
  const applyLabel =
    viewMode === "inpainted"
      ? isReinpainting
        ? "Refazendo limpa..."
        : "Refazer limpa"
      : isRetypesetting
        ? "Aplicando..."
        : "Aplicar";

  return (
    <div className="flex h-screen bg-bg-primary">
      <PageThumbnails />

      <div className="flex min-w-0 flex-1 flex-col border-r border-white/5">
        <div className="flex items-center justify-between gap-4 border-b border-white/5 bg-bg-secondary px-4 py-2.5">
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

          <div className="flex flex-wrap items-center justify-center gap-1.5">
            {VIEW_MODES.map(({ key, label, icon: Icon, hotkey }) => (
              <button
                key={key}
                onClick={() => setViewMode(key)}
                className={`flex items-center gap-1 rounded border px-2.5 py-1.5 text-xs transition-smooth ${
                  viewMode === key
                    ? "border-accent-purple/30 bg-accent-purple/15 text-accent-purple"
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
              className={`flex items-center gap-1 rounded border px-2.5 py-1.5 text-xs transition-smooth ${
                showOverlays
                  ? "border-accent-cyan/20 bg-accent-cyan/10 text-accent-cyan"
                  : "border-transparent text-text-muted"
              }`}
              title="Guias de selecao (O)"
            >
              {showOverlays ? <Eye size={13} /> : <EyeOff size={13} />}
              Guias
            </button>

            <div className="mx-1 h-5 w-px bg-white/5" />

            <button
              onClick={() => void commitEdits()}
              disabled={pendingCount === 0}
              className="flex items-center gap-1 rounded border border-transparent px-2.5 py-1.5 text-xs text-status-success transition-smooth hover:bg-status-success/10 disabled:opacity-40"
              title="Salvar edicoes (Ctrl+S)"
            >
              <Check size={13} />
              Salvar
            </button>
            <button
              onClick={discardEdits}
              disabled={pendingCount === 0}
              className="flex items-center gap-1 rounded border border-transparent px-2.5 py-1.5 text-xs text-text-secondary transition-smooth hover:bg-bg-tertiary disabled:opacity-40"
              title="Descartar edicoes pendentes"
            >
              <Undo2 size={13} />
              Descartar
            </button>
            {pendingCount > 0 && (
              <span className="rounded-full bg-accent-purple/12 px-2 py-1 text-[11px] text-accent-purple">
                {pendingCount} alteracao(oes)
              </span>
            )}

            <div className="mx-1 h-5 w-px bg-white/5" />

            <button
              disabled={isApplyingCurrentView || !canApplyCurrentView}
              onClick={applyCurrentView}
              className="flex items-center gap-1 rounded bg-accent-purple px-2.5 py-1.5 text-xs text-white transition-smooth hover:bg-accent-purple/90 disabled:opacity-50"
              title={viewMode === "inpainted" ? "Refazer pagina limpa" : "Atualizar imagem traduzida"}
            >
              <Play size={13} className={isApplyingCurrentView ? "animate-pulse" : ""} />
              {applyLabel}
            </button>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={zoomOut}
              className="rounded bg-bg-tertiary p-1.5 text-text-secondary transition-smooth hover:text-text-primary"
              title="Diminuir zoom (-)"
            >
              <Minus size={14} />
            </button>
            <button
              onClick={zoomIn}
              className="rounded bg-bg-tertiary p-1.5 text-text-secondary transition-smooth hover:text-text-primary"
              title="Aumentar zoom (+)"
            >
              <Plus size={14} />
            </button>
            <button
              onClick={resetViewport}
              className="rounded bg-bg-tertiary px-2 py-1.5 text-[11px] text-text-secondary transition-smooth hover:text-text-primary"
              title="Ajustar viewport (0)"
            >
              Ajustar
            </button>
            <button
              onClick={() => setZoom(2)}
              className="rounded bg-bg-tertiary px-2 py-1.5 text-[11px] text-text-secondary transition-smooth hover:text-text-primary"
              title="Zoom 2x"
            >
              2x
            </button>
            <button
              onClick={resetViewport}
              className="rounded bg-bg-tertiary p-1.5 text-text-secondary transition-smooth hover:text-text-primary"
              title="Centralizar"
            >
              <LocateFixed size={14} />
            </button>
            <span className="w-12 text-right font-mono text-[11px] text-text-muted">
              {Math.round(zoom * 100)}%
            </span>

            <div className="mx-1 h-5 w-px bg-white/5" />

            <button
              onClick={() => setCurrentPage(Math.max(0, currentPageIndex - 1))}
              disabled={currentPageIndex === 0}
              className="rounded bg-bg-tertiary p-1.5 text-text-secondary transition-smooth hover:text-text-primary disabled:opacity-30"
              title="Pagina anterior (Alt+Left)"
            >
              <ArrowLeft size={14} />
            </button>
            <span className="min-w-[56px] text-center font-mono text-xs text-text-secondary">
              {currentPageIndex + 1}/{totalPages}
            </span>
            <button
              onClick={() => setCurrentPage(Math.min(totalPages - 1, currentPageIndex + 1))}
              disabled={currentPageIndex >= totalPages - 1}
              className="rounded bg-bg-tertiary p-1.5 text-text-secondary transition-smooth hover:text-text-primary disabled:opacity-30"
              title="Proxima pagina (Alt+Right)"
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

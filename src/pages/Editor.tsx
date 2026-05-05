import { useEffect, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import {
  ArrowLeft,
  ArrowRight,
  Brush,
  ChevronLeft,
  Eraser,
  Eye,
  EyeOff,
  FileText,
  GripHorizontal,
  Image,
  Layers,
  PenTool,
  SquareDashedMousePointer,
  Undo2,
  ScanText,
  Languages,
  Loader2,
} from "lucide-react";
import { useAppStore } from "../lib/stores/appStore";
import { EditorStage } from "../components/editor/stage/EditorStage";
import { LayersPanel } from "../components/editor/LayersPanel";
import { PageThumbnails } from "../components/editor/PageThumbnails";
import { useEditorStore, type EditorToolMode } from "../lib/stores/editorStore";
import { ZoomControls } from "../components/editor/toolbar/ZoomControls";
import { AutoSaveIndicator } from "../components/editor/toolbar/AutoSaveIndicator";
import { preloadEditorFonts } from "../lib/fonts";

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

function isEditableTarget(target: EventTarget | null) {
  if (!(target instanceof HTMLElement)) return false;
  return (
    target.tagName === "INPUT" ||
    target.tagName === "TEXTAREA" ||
    target.tagName === "SELECT" ||
    target.isContentEditable ||
    !!target.closest("[contenteditable='true']")
  );
}

export function Editor() {
  const navigate = useNavigate();
  const project = useAppStore((s) => s.project);
  const pipeline = useAppStore((s) => s.pipeline);
  const {
    currentPageIndex,
    currentPage,
    selectedLayerId,
    toolMode,
    viewMode,
    showOverlays,
    brushSize,
    pendingEdits,
    pendingStructuralEdits,
    renderPreviewCacheByPageKey,
    isRetypesetting,
    isReinpainting,
    loadCurrentPage,
    setCurrentPage,
    setToolMode,
    setViewMode,
    toggleOverlays,
    zoomIn,
    zoomOut,
    resetViewport,
    commitEdits,
    discardEdits,
    undoEditor,
    redoEditor,
    deleteSelectedLayer,
    retypesetCurrentPage,
    renderPreviewPage,
    currentPageKey,
    setBrushSize,
    activePageAction,
    runMaskedAction,
    pageActionError,
    clearPageActionError,
    runAutoSave,
    flushAutoSave,
  } = useEditorStore();

  const totalPages = project?.paginas.length ?? 0;
  const projectId = project?.id ?? null;
  const pendingCount =
    Object.keys(pendingEdits).length +
    pendingStructuralEdits.created.length +
    Object.keys(pendingStructuralEdits.deleted).length +
    (pendingStructuralEdits.order ? 1 : 0);
  const pagePipelineBusy = isRetypesetting || isReinpainting;
  const pageKey = currentPageKey();
  // renderPreviewState é mantido em escopo para a Fase 6 (RenderStatusBadge),
  // mas o flag previewRendering antigo (do botão removido) saiu junto.
  void renderPreviewCacheByPageKey;
  void pageKey;

  useEffect(() => {
    if (!project || totalPages === 0) return;
    void loadCurrentPage();
  }, [loadCurrentPage, projectId, totalPages]);

  // Fase 2A: preload das fontes bundle antes do primeiro draw do Konva.
  // Sem isto, o canvas pode renderizar com fonte fallback do sistema enquanto
  // o navegador ainda baixa o TTF — o usuário via "Comic Neue" no select mas
  // o canvas mostrava Arial. Aguarda document.fonts.ready depois.
  useEffect(() => {
    void preloadEditorFonts().catch((err) => {
      console.warn("[fonts] preloadEditorFonts falhou:", err);
    });
  }, []);

  // Fase 3: auto-save híbrido — interval 3s + flush em saída do app.
  useEffect(() => {
    const intervalId = window.setInterval(() => {
      void runAutoSave();
    }, 3000);
    const onBeforeUnload = () => {
      // Best-effort: o browser não espera Promise, mas patches HTTP em curso
      // costumam concluir. Para garantia real, usar Tauri close-requested.
      void flushAutoSave().catch(() => {});
    };
    const onVisibility = () => {
      if (document.visibilityState === "hidden") {
        void flushAutoSave().catch(() => {});
      }
    };
    window.addEventListener("beforeunload", onBeforeUnload);
    window.addEventListener("pagehide", onBeforeUnload);
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      window.clearInterval(intervalId);
      window.removeEventListener("beforeunload", onBeforeUnload);
      window.removeEventListener("pagehide", onBeforeUnload);
      document.removeEventListener("visibilitychange", onVisibility);
      // Unmount também faz flush.
      void flushAutoSave().catch(() => {});
    };
  }, [runAutoSave, flushAutoSave]);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      const isTyping = isEditableTarget(document.activeElement);

      if (!isTyping) {
        if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
          event.preventDefault();
          if (event.shiftKey) redoEditor();
          else undoEditor();
          return;
        }
        if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "y") {
          event.preventDefault();
          redoEditor();
          return;
        }
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
      // Fase 1.2: atalhos de fallback para preview/render que perderam o botão
      if (
        (event.ctrlKey || event.metaKey) &&
        event.shiftKey &&
        event.key.toLowerCase() === "r"
      ) {
        event.preventDefault();
        void retypesetCurrentPage();
      }
      if (
        (event.ctrlKey || event.metaKey) &&
        event.shiftKey &&
        event.key.toLowerCase() === "p"
      ) {
        event.preventDefault();
        void renderPreviewPage(currentPageKey());
      }
    };

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [
    commitEdits,
    currentPageIndex,
    deleteSelectedLayer,
    redoEditor,
    resetViewport,
    selectedLayerId,
    setCurrentPage,
    setToolMode,
    setViewMode,
    toggleOverlays,
    totalPages,
    undoEditor,
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
            className="text-sm text-brand hover:underline"
          >
            Voltar ao início
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-bg-primary">
      <PageThumbnails />

      <div className="flex min-w-0 flex-1 flex-col">
        {/* ── Row 1: Header ── */}
        <div className="flex items-center gap-3 border-b border-border bg-bg-secondary/60 px-3 py-2">
          <button
            onClick={() => navigate("/preview")}
            className="rounded-lg p-1.5 text-text-muted transition-smooth hover:bg-white/[0.04] hover:text-text-primary"
            title="Voltar ao preview"
          >
            <ChevronLeft size={16} />
          </button>

          <div className="min-w-0 flex-1">
            <p className="truncate text-[13px] font-semibold tracking-tight text-text-primary">{project.obra}</p>
            <p className="truncate text-[11px] text-text-muted">
              Cap. {project.capitulo} · {currentPageSummary}
            </p>
          </div>

          <div
            data-tauri-drag-region
            className="hidden h-5 w-10 shrink-0 cursor-grab items-center justify-center rounded-full bg-white/[0.03] text-text-muted/50 md:flex"
            title="Arrastar janela"
          >
            <GripHorizontal size={12} />
          </div>

          {/* Page navigation */}
          <div className="flex items-center gap-1 rounded-lg border border-border bg-bg-tertiary/40 px-1 py-0.5">
            <button
              onClick={() => void setCurrentPage(Math.max(0, currentPageIndex - 1))}
              disabled={currentPageIndex === 0}
              className="rounded p-1 text-text-muted transition-smooth hover:text-text-primary disabled:opacity-25"
              title="Página anterior (Alt+←)"
            >
              <ArrowLeft size={12} />
            </button>
            <span className="min-w-[40px] text-center font-mono text-[11px] text-text-secondary">
              {currentPageIndex + 1}/{totalPages}
            </span>
            <button
              onClick={() => void setCurrentPage(Math.min(totalPages - 1, currentPageIndex + 1))}
              disabled={currentPageIndex >= totalPages - 1}
              className="rounded p-1 text-text-muted transition-smooth hover:text-text-primary disabled:opacity-25"
              title="Próxima página (Alt+→)"
            >
              <ArrowRight size={12} />
            </button>
          </div>

          {/* Auto-save indicator + descartar (Fase 3 substitui o botão Salvar). */}
          <div className="flex items-center gap-1">
            <AutoSaveIndicator />
            <button
              onClick={discardEdits}
              disabled={pendingCount === 0}
              className="rounded-lg p-1.5 text-text-muted transition-smooth hover:bg-white/[0.04] hover:text-text-primary disabled:opacity-30"
              title="Descartar alterações"
            >
              <Undo2 size={13} />
            </button>
          </div>
        </div>

        {/* ── Row 2: Tools ── */}
        <div className="flex items-center gap-2 border-b border-border bg-bg-primary px-3 py-1.5">
          {/* View modes — segmented control */}
          <div className="flex items-center rounded-lg border border-border bg-bg-tertiary/30 p-0.5">
            {VIEW_MODES.map(({ key, label, icon: Icon, hotkey }) => (
              <button
                key={key}
                data-testid={`editor-view-${key}`}
                onClick={() => setViewMode(key)}
                className={`flex items-center gap-1 rounded-md px-2.5 py-1 text-[11px] font-medium transition-smooth ${
                  viewMode === key
                    ? "bg-brand/15 text-brand shadow-sm"
                    : "text-text-muted hover:text-text-primary"
                }`}
                title={`${label} (${hotkey})`}
              >
                <Icon size={12} />
                {label}
              </button>
            ))}
          </div>

          <div className="h-4 w-px bg-border" />

          {/* Tool modes — segmented control */}
          <div className="flex items-center rounded-lg border border-border bg-bg-tertiary/30 p-0.5">
            {TOOL_MODES.map(({ key, label, icon: Icon, hotkey }) => (
              <button
                key={key}
                data-testid={`editor-tool-${key}`}
                onClick={() => setToolMode(key)}
                className={`flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-medium transition-smooth ${
                  toolMode === key
                    ? "bg-accent-cyan/12 text-accent-cyan shadow-sm"
                    : "text-text-muted hover:text-text-primary"
                }`}
                title={`${label} (${hotkey})`}
              >
                <Icon size={12} />
                <span className="hidden lg:inline">{label}</span>
              </button>
            ))}
          </div>

          {/* Overlays toggle */}
          <button
            onClick={toggleOverlays}
            className={`flex items-center gap-1 rounded-lg px-2 py-1 text-[11px] font-medium transition-smooth ${
              showOverlays
                ? "bg-accent-cyan/10 text-accent-cyan"
                : "text-text-muted hover:text-text-primary"
            }`}
            title="Guias de seleção (O)"
          >
            {showOverlays ? <Eye size={12} /> : <EyeOff size={12} />}
          </button>

          {/* Brush size — contextual */}
          {(toolMode === "brush" || toolMode === "repairBrush" || toolMode === "eraser") && (
            <div className="flex items-center gap-1.5 rounded-lg border border-border bg-bg-tertiary/40 px-2 py-1">
              <span className="text-[10px] text-text-muted">Pincel</span>
              <input
                type="range"
                min={4}
                max={96}
                value={brushSize}
                title="Tamanho do pincel"
                aria-label="Tamanho do pincel"
                onChange={(event) => setBrushSize(Number(event.target.value))}
                className="w-20"
              />
              <span className="w-7 text-right font-mono text-[10px] text-text-secondary">{brushSize}</span>
            </div>
          )}

          <div className="flex-1" />

          {/* Pipeline actions */}
          <div className="flex items-center gap-0.5 rounded-lg border border-border bg-bg-tertiary/30 p-0.5">
            {activePageAction !== null && (
              <span className="px-1.5 text-[10px] font-medium text-brand animate-pulse">
                {activePageAction}...
              </span>
            )}
            <button
              disabled={pagePipelineBusy || activePageAction !== null}
              onClick={() => void runMaskedAction("detect")}
              className="flex items-center gap-1 rounded-md px-2 py-1 text-[11px] text-text-muted transition-smooth hover:bg-white/[0.04] hover:text-text-primary disabled:opacity-40"
              title="Detectar balões"
            >
              <ScanText size={12} className={activePageAction === "detect" ? "animate-pulse" : ""} />
              <span className="hidden xl:inline">Detectar</span>
            </button>
            <button
              disabled={pagePipelineBusy || activePageAction !== null}
              onClick={() => void runMaskedAction("ocr")}
              className="flex items-center gap-1 rounded-md px-2 py-1 text-[11px] text-text-muted transition-smooth hover:bg-white/[0.04] hover:text-text-primary disabled:opacity-40"
              title="Executar OCR"
            >
              <FileText size={12} className={activePageAction === "ocr" ? "animate-pulse" : ""} />
              <span className="hidden xl:inline">OCR</span>
            </button>
            <button
              disabled={pagePipelineBusy || activePageAction !== null}
              onClick={() => void runMaskedAction("translate")}
              className="flex items-center gap-1 rounded-md px-2 py-1 text-[11px] text-text-muted transition-smooth hover:bg-white/[0.04] hover:text-text-primary disabled:opacity-40"
              title="Traduzir textos"
            >
              <Languages size={12} className={activePageAction === "translate" ? "animate-pulse" : ""} />
              <span className="hidden xl:inline">Traduzir</span>
            </button>
            <button
              disabled={pagePipelineBusy || activePageAction !== null}
              onClick={() => void runMaskedAction("inpaint")}
              className="flex items-center gap-1 rounded-md px-2 py-1 text-[11px] text-text-muted transition-smooth hover:bg-white/[0.04] hover:text-text-primary disabled:opacity-40"
              title="Limpar imagem (Inpaint)"
            >
              <Eraser size={12} className={activePageAction === "inpaint" ? "animate-pulse" : ""} />
              <span className="hidden xl:inline">Inpaint</span>
            </button>

            {/* Fase 1.2: Preview/Render removidos da UI principal.
                Auto-render da Fase 6 (debounce 1.5s) faz o trabalho automaticamente.
                Ctrl+Shift+R força render fiel; Ctrl+Shift+P força preview. */}
          </div>

          {/* Pipeline progress indicator */}
          {(isRetypesetting || isReinpainting) && !!pipeline && (
            <div className="flex items-center gap-1.5 rounded-lg bg-brand/8 px-2.5 py-1 border border-brand/15">
              <Loader2 size={11} className="text-brand animate-spin" />
              <span className="text-[10px] text-brand font-medium truncate max-w-[100px]">
                {pipeline.message || "Processando"}
              </span>
              <span className="text-[10px] text-brand/60 font-mono">
                {Math.round(pipeline.step_progress)}%
              </span>
            </div>
          )}

          {/* Zoom controls — Fase 1: movido do canto inferior direito do canvas
              para ficar próximo das ações pipeline e liberar área visual */}
          <div className="ml-auto">
            <ZoomControls />
          </div>
        </div>

        {/* Banner de erro de ação pipeline (Fase 0 - sem falhas silenciosas) */}
        {pageActionError && (
          <div className="flex items-start gap-2 border-b border-status-error/30 bg-status-error/10 px-3 py-2">
            <span className="mt-0.5 inline-flex h-4 w-4 flex-shrink-0 items-center justify-center rounded-full bg-status-error/20 text-[10px] font-bold text-status-error">
              !
            </span>
            <div className="min-w-0 flex-1">
              <p className="text-[11px] font-medium text-status-error">
                Falha em {pageActionError.action.toUpperCase()}
              </p>
              <p className="mt-0.5 max-h-24 overflow-y-auto whitespace-pre-wrap break-words text-[10px] leading-relaxed text-text-muted font-mono">
                {pageActionError.message}
              </p>
            </div>
            <button
              onClick={() => void runMaskedAction(pageActionError.action)}
              disabled={activePageAction !== null}
              className="rounded-md border border-border bg-bg-secondary px-2 py-0.5 text-[10px] text-text-primary hover:bg-bg-tertiary disabled:opacity-40"
              title="Tentar novamente"
            >
              Retry
            </button>
            <button
              onClick={clearPageActionError}
              className="rounded-md px-1.5 py-0.5 text-[10px] text-text-muted hover:bg-white/[0.04]"
              title="Fechar"
            >
              ×
            </button>
          </div>
        )}

        <EditorStage />
      </div>

      <LayersPanel />
    </div>
  );
}

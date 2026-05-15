import { useEffect, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import {
  ArrowLeft,
  ArrowRight,
  Check,
  ChevronLeft,
  Eraser,
  Eye,
  EyeOff,
  FileText,
  GripHorizontal,
  Image,
  Layers,
  Undo2,
  ScanText,
  Languages,
  Loader2,
  X,
} from "lucide-react";
import { useAppStore } from "../lib/stores/appStore";
import { EditorStage } from "../components/editor/stage/EditorStage";
import { LayersPanel } from "../components/editor/LayersPanel";
import { PageThumbnails } from "../components/editor/PageThumbnails";
import { getRenderPreviewStateForPage, useEditorStore } from "../lib/stores/editorStore";
import { ZoomControls } from "../components/editor/toolbar/ZoomControls";
import { AutoSaveIndicator } from "../components/editor/toolbar/AutoSaveIndicator";
import { TypesettingBar } from "../components/editor/toolbar/TypesettingBar";
import { ToolSidebar } from "../components/editor/toolbar/ToolSidebar";
import { BrushOptionsInline } from "../components/editor/toolbar/BrushOptionsPopover";
import { UndoRedoControls } from "../components/editor/toolbar/UndoRedoControls";
import { preloadEditorFonts } from "../lib/fonts";

const VIEW_MODES = [
  { key: "original" as const, label: "Original", icon: Image, hotkey: "1" },
  { key: "inpainted" as const, label: "Limpa", icon: Eraser, hotkey: "2" },
  { key: "translated" as const, label: "Camadas", icon: FileText, hotkey: "3" },
];


/** Controles contextuais da ferramenta Lasso (Fase 8). */
function MaskLassoControls() {
  const maskShape = useEditorStore((s) => s.maskShape);
  const maskOp = useEditorStore((s) => s.maskOp);
  const setMaskShape = useEditorStore((s) => s.setMaskShape);
  const setMaskOp = useEditorStore((s) => s.setMaskOp);
  const clearMask = useEditorStore((s) => s.clearMask);
  const activeLassoSelection = useEditorStore((s) => s.activeLassoSelection);
  const applyLassoSelectionToMask = useEditorStore((s) => s.applyLassoSelectionToMask);
  const setActiveLassoSelection = useEditorStore((s) => s.setActiveLassoSelection);

  return (
    <div className="flex items-center gap-2">
      {/* Shape toggle */}
      <div className="flex items-center rounded-lg border border-border bg-bg-tertiary/30 p-0.5">
        {(["freehand", "polygonal"] as const).map((shape) => (
          <button
            key={shape}
            onClick={() => setMaskShape(shape)}
            className={`rounded-md px-2 py-1 text-[10px] font-medium transition-smooth ${
              maskShape === shape ? "bg-accent-cyan/15 text-accent-cyan" : "text-text-muted hover:text-text-primary"
            }`}
          >
            {shape === "freehand" ? "Livre" : "Poligonal"}
          </button>
        ))}
      </div>

      {/* Op toggle */}
      <div className="flex items-center rounded-lg border border-border bg-bg-tertiary/30 p-0.5">
        {(["replace", "add", "subtract"] as const).map((op) => (
          <button
            key={op}
            onClick={() => setMaskOp(op)}
            className={`rounded-md px-2 py-1 text-[10px] font-medium transition-smooth ${
              maskOp === op ? "bg-accent-cyan/15 text-accent-cyan" : "text-text-muted hover:text-text-primary"
            }`}
            title={op === "replace" ? "Substituir (padrão)" : op === "add" ? "Adicionar (Shift)" : "Subtrair (Alt)"}
          >
            {op === "replace" ? "⊙" : op === "add" ? "+" : "−"}
          </button>
        ))}
      </div>

      {/* Clear mask */}
      <button
        onClick={() => void clearMask()}
        className="flex items-center gap-1 rounded-lg border border-border bg-bg-tertiary/30 px-2 py-1 text-[10px] text-text-muted transition-smooth hover:border-status-error/30 hover:text-status-error"
        title="Limpar máscara"
      >
        Limpar
      </button>

      {activeLassoSelection && (
        <div className="flex items-center rounded-lg border border-border bg-bg-tertiary/30 p-0.5">
          <button
            type="button"
            onClick={() => void applyLassoSelectionToMask()}
            className="flex items-center gap-1 rounded-md px-2 py-1 text-[10px] font-medium text-accent-cyan transition-smooth hover:bg-accent-cyan/10"
            title="Aplicar seleção à máscara"
          >
            <Check size={11} />
            Aplicar
          </button>
          <button
            type="button"
            onClick={() => setActiveLassoSelection(null)}
            className="flex h-6 w-6 items-center justify-center rounded-md text-text-muted transition-smooth hover:bg-white/[0.06] hover:text-text-primary"
            title="Cancelar seleção"
          >
            <X size={12} />
          </button>
        </div>
      )}
    </div>
  );
}

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

export interface EditorProps {
  onBack?: () => void;
  emptyBackLabel?: string;
}

export function Editor({ onBack, emptyBackLabel = "Voltar ao início" }: EditorProps = {}) {
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
    isHealingBrushApplying,
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
    renderPreviewPage,
    currentPageKey,
    setBrushSize,
    activePageAction,
    runMaskedAction,
    runMaskedActionFromLasso,
    activeLassoSelection,
    pageActionError,
    clearPageActionError,
    forceFidelityRender,
    eraserTarget,
    lastPaintedLayer,
    setEraserTarget,
  } = useEditorStore();

  const totalPages = project?.paginas.length ?? 0;
  const projectId = project?.id ?? null;
  const pendingCount =
    Object.keys(pendingEdits).length +
    pendingStructuralEdits.created.length +
    Object.keys(pendingStructuralEdits.deleted).length +
    (pendingStructuralEdits.order ? 1 : 0);
  const pagePipelineBusy = isRetypesetting || isReinpainting || isHealingBrushApplying;
  const pageKey = currentPageKey();
  const renderPreviewState = useMemo(
    () => getRenderPreviewStateForPage(pageKey, currentPage, renderPreviewCacheByPageKey),
    [currentPage, pageKey, renderPreviewCacheByPageKey],
  );
  const saveDisabled =
    !currentPage ||
    pagePipelineBusy ||
    (pendingCount === 0 && renderPreviewState.status === "fresh");
  const runSelectionAwareAction = (action: Parameters<typeof runMaskedAction>[0]) =>
    activeLassoSelection ? runMaskedActionFromLasso(action) : runMaskedAction(action);

  const saveAndRenderCurrentPage = async () => {
    const targetPageKey = currentPageKey();
    await commitEdits();
    await renderPreviewPage(targetPageKey);
  };
  const handleBack = () => {
    if (onBack) {
      onBack();
      return;
    }
    navigate("/preview");
  };

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

  // Auto-save desligado — usuário salva manualmente via botão ou Ctrl+S
  // (estilo Photoshop). markDirty continua funcionando para alimentar o
  // indicador "Alterações não salvas".

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
        if (event.key.toLowerCase() === "t") setToolMode("block");
        if (event.key.toLowerCase() === "b") setToolMode("brush");
        if (event.key.toLowerCase() === "r") setToolMode("repairBrush");
        if (event.key.toLowerCase() === "i") setToolMode("reinpaintBrush");
        if (event.key.toLowerCase() === "e") setToolMode("eraser");
        if (event.key.toLowerCase() === "l") setToolMode("mask");
        // Tab: cicla alvo da borracha quando eraser ativo
        if (event.key === "Tab" && toolMode === "eraser") {
          event.preventDefault();
          setEraserTarget(eraserTarget === "brush" || eraserTarget === null ? "mask" : "brush");
        }
        // Legacy aliases
        if (event.key.toLowerCase() === "n") setToolMode("brush");
        if (event.key.toLowerCase() === "m") setToolMode("mask");
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
        void saveAndRenderCurrentPage();
      }
      // Fase 1.2: atalhos de fallback para preview/render que perderam o botão
      if (
        (event.ctrlKey || event.metaKey) &&
        event.shiftKey &&
        event.key.toLowerCase() === "r"
      ) {
        event.preventDefault();
        void forceFidelityRender();
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
    currentPageIndex,
    currentPageKey,
    deleteSelectedLayer,
    eraserTarget,
    forceFidelityRender,
    redoEditor,
    renderPreviewPage,
    resetViewport,
    selectedLayerId,
    setCurrentPage,
    setEraserTarget,
    setToolMode,
    setViewMode,
    toggleOverlays,
    toolMode,
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
            onClick={onBack ?? (() => navigate("/"))}
            className="text-sm text-brand hover:underline"
          >
            {emptyBackLabel}
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
            onClick={handleBack}
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

          {/* Undo/Redo + indicador "Não salvo" + Salvar manual + descartar */}
          <div className="flex items-center gap-1.5">
            <UndoRedoControls />
            <AutoSaveIndicator />
            <button
              onClick={() => void saveAndRenderCurrentPage()}
              disabled={saveDisabled}
              className="flex items-center gap-1 rounded-lg border border-status-success/30 bg-status-success/10 px-2.5 py-1 text-[11px] font-medium text-status-success transition-smooth hover:bg-status-success/15 disabled:opacity-30"
              title="Salvar e renderizar preview final (Ctrl+S)"
            >
              <Check size={12} />
              Salvar
            </button>
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

        {/* ── Row 2: View + Pipeline + Zoom ── */}
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

          {/* Brush options — contextual quando ferramenta brush ativa (Fase 7: inclui color picker) */}
          {toolMode === "brush" && <BrushOptionsInline />}
          {/* Máscara Lasso options — Fase 8 */}
          {toolMode === "mask" && <MaskLassoControls />}
          {/* Brush size simples para repairBrush/reinpaintBrush/eraser */}
          {(toolMode === "repairBrush" || toolMode === "reinpaintBrush" || toolMode === "eraser") && (
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
          {/* Fase 9: Indicador de alvo da borracha */}
          {toolMode === "eraser" && (
            <div className="flex items-center rounded-lg border border-border bg-bg-tertiary/30 p-0.5">
              {(["brush", "mask"] as const).map((t) => {
                const active = eraserTarget === t || (eraserTarget === null && lastPaintedLayer === t);
                return (
                  <button
                    key={t}
                    onClick={() => setEraserTarget(active ? null : t)}
                    title={`Apagar: ${t === "brush" ? "Pintura" : "Máscara"} (Tab para ciclar)`}
                    className={`rounded-md px-2 py-1 text-[10px] font-medium transition-smooth ${
                      active ? "bg-accent-cyan/15 text-accent-cyan" : "text-text-muted hover:text-text-primary"
                    }`}
                  >
                    {t === "brush" ? "Pintura" : "Máscara"}
                  </button>
                );
              })}
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
              onClick={() => void runSelectionAwareAction("detect")}
              className="flex items-center gap-1 rounded-md px-2 py-1 text-[11px] text-text-muted transition-smooth hover:bg-white/[0.04] hover:text-text-primary disabled:opacity-40"
              title="Detectar balões"
            >
              <ScanText size={12} className={activePageAction === "detect" ? "animate-pulse" : ""} />
              <span className="hidden xl:inline">Detectar</span>
            </button>
            <button
              disabled={pagePipelineBusy || activePageAction !== null}
              onClick={() => void runSelectionAwareAction("ocr")}
              className="flex items-center gap-1 rounded-md px-2 py-1 text-[11px] text-text-muted transition-smooth hover:bg-white/[0.04] hover:text-text-primary disabled:opacity-40"
              title="Executar OCR"
            >
              <FileText size={12} className={activePageAction === "ocr" ? "animate-pulse" : ""} />
              <span className="hidden xl:inline">OCR</span>
            </button>
            <button
              disabled={pagePipelineBusy || activePageAction !== null}
              onClick={() => void runSelectionAwareAction("translate")}
              className="flex items-center gap-1 rounded-md px-2 py-1 text-[11px] text-text-muted transition-smooth hover:bg-white/[0.04] hover:text-text-primary disabled:opacity-40"
              title="Traduzir textos"
            >
              <Languages size={12} className={activePageAction === "translate" ? "animate-pulse" : ""} />
              <span className="hidden xl:inline">Traduzir</span>
            </button>
            <button
              disabled={pagePipelineBusy || activePageAction !== null}
              onClick={() => void runSelectionAwareAction("inpaint")}
              className="flex items-center gap-1 rounded-md px-2 py-1 text-[11px] text-text-muted transition-smooth hover:bg-white/[0.04] hover:text-text-primary disabled:opacity-40"
              title="Limpar imagem (Inpaint)"
            >
              <Eraser size={12} className={activePageAction === "inpaint" ? "animate-pulse" : ""} />
              <span className="hidden xl:inline">Inpaint</span>
            </button>
          </div>

          {/* Pipeline progress indicator */}
          {(isRetypesetting || isReinpainting || isHealingBrushApplying) && (
            <div className="flex items-center gap-1.5 rounded-lg bg-brand/8 px-2.5 py-1 border border-brand/15">
              <Loader2 size={11} className="text-brand animate-spin" />
              <span className="text-[10px] text-brand font-medium truncate max-w-[100px]">
                {isHealingBrushApplying ? "Corrigindo" : pipeline?.message || "Processando"}
              </span>
              <span className="text-[10px] text-brand/60 font-mono">
                {isHealingBrushApplying ? "" : `${Math.round(pipeline?.step_progress ?? 0)}%`}
              </span>
            </div>
          )}

          {/* Zoom controls (movido do canvas para cá na Fase 1) */}
          <ZoomControls />
        </div>

        {/* ── Row 3: TypesettingBar — só quando texto selecionado (Fase 4) ── */}
        <TypesettingBar />

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
              onClick={() => void runSelectionAwareAction(pageActionError.action)}
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

        {/* ── Canvas area: ToolSidebar + Stage ── */}
        <div className="flex min-h-0 flex-1">
          {/* Fase 4: ToolSidebar vertical substituindo o segmented control horizontal */}
          <ToolSidebar />
          <EditorStage />
        </div>
      </div>

      <LayersPanel />
    </div>
  );
}

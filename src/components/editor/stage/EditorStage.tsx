import { Layer, Line, Rect, Stage } from "react-konva";
import { getRenderPreviewStateForPage, useEditorStore } from "../../../lib/stores/editorStore";
import { EditorBitmapOverlay } from "./EditorBitmapOverlay";
import { EditorPaintCursor } from "./EditorPaintCursor";
import { EditorStageBackground } from "./EditorStageBackground";
import { EditorTextLayer } from "./EditorTextLayer";
import { EditorTransformer } from "./EditorTransformer";
import { FloatingTextEditor } from "./FloatingTextEditor";
import { MaskInProgressOverlay } from "./MaskInProgressOverlay";
import { useEditorStageController } from "./useEditorStageController";

function StageStatusBadge() {
  const currentPage = useEditorStore((state) => state.currentPage);
  const currentPageKey = useEditorStore((state) => state.currentPageKey());
  const cache = useEditorStore((state) => state.renderPreviewCacheByPageKey);
  const viewMode = useEditorStore((state) => state.viewMode);
  const state = getRenderPreviewStateForPage(currentPageKey, currentPage, cache);
  if (viewMode !== "translated") return null;

  const label =
    state.status === "rendering"
      ? "Renderizando preview"
      : state.status === "error"
        ? "Preview pendente"
        : state.status === "stale"
          ? "Preview desatualizado"
          : state.previewPath
            ? "Preview fiel"
            : null;
  if (!label) return null;

  const fresh = state.status === "fresh" && !!state.previewPath;
  return (
    <div className="pointer-events-none absolute left-4 top-4 z-20">
      <div
        data-testid="editor-preview-status"
        data-status={state.status}
        className={`rounded-lg border px-2.5 py-1 text-[10px] font-medium backdrop-blur-md ${
          fresh
            ? "border-status-success/20 bg-status-success/10 text-status-success"
            : "border-status-warning/20 bg-status-warning/10 text-status-warning"
        }`}
      >
        {label}
      </div>
    </div>
  );
}

export function EditorStage() {
  const controller = useEditorStageController();
  const brushColor = useEditorStore((s) => s.brushColor);
  const brushOpacity = useEditorStore((s) => s.brushOpacity);
  const imageLayers = useEditorStore((s) => s.currentPage?.image_layers ?? {});
  const {
    containerRef,
    containerSize,
    currentPage,
    toolMode,
    showOverlays,
    brushSize,
    panOffset,
    panSession,
    viewportCursor,
    baseImage,
    maskImage,
    brushImage,
    blockDraft,
    paintStroke,
    layers,
    selectedLayerId,
    selectedImageLayerKey,
    hoveredLayerId,
    selectedNodeName,
    stageScale,
    faithfulPreview,
    bitmapInspection,
    translatedEditing,
    selectLayer,
    selectImageLayer,
    hoverLayer,
    commitBbox,
    handleViewportMouseDown,
    handleStageMouseDown,
    handleStageMouseMove,
    handleStageMouseUp,
    handleStageMouseEnter,
    handleStageMouseLeave,
    cursorPoint,
    maskInProgress,
    maskShape,
  } = controller;

  const width = baseImage.size.width;
  const height = baseImage.size.height;

  const hintText =
    toolMode === "block"
      ? "Arraste para criar uma nova camada de texto"
      : toolMode === "brush" || toolMode === "repairBrush" || toolMode === "eraser"
        ? "Pintura ativa no Stage"
        : "Ctrl+scroll: zoom · Scroll: mover · Space+drag: mover";

  return (
    <div
      ref={containerRef}
      data-testid="editor-stage"
      data-base-kind={
        selectedImageLayerKey === "base" ||
        selectedImageLayerKey === "inpaint" ||
        selectedImageLayerKey === "rendered"
          ? selectedImageLayerKey
          : currentPage?.image_layers?.inpaint?.path
            ? "inpaint"
            : currentPage?.image_layers?.base?.path
              ? "base"
              : "original"
      }
      className="relative flex-1 overflow-hidden bg-[radial-gradient(circle_at_top,_rgba(108,92,231,0.06),_transparent_40%)]"
      style={{ cursor: viewportCursor }}
      onMouseDownCapture={handleViewportMouseDown}
    >
      {/* Bottom hint */}
      <div className="pointer-events-none absolute inset-x-0 bottom-4 z-20 flex justify-center px-4">
        <div className="rounded-lg border border-border bg-bg-secondary/80 px-3 py-1 text-[10px] text-text-muted backdrop-blur-md">
          {hintText}
        </div>
      </div>

      {/* Status badges */}
      <StageStatusBadge />
      {bitmapInspection && (
        <div className="pointer-events-none absolute right-4 top-4 z-20">
          <div className="rounded-lg border border-accent-cyan/20 bg-accent-cyan/10 px-2.5 py-1 text-[10px] font-medium text-accent-cyan backdrop-blur-md">
            Bitmap: {selectedImageLayerKey}
          </div>
        </div>
      )}

      {/* Fase 5: FloatingTextEditor — painel flutuante para edição rápida */}
      <FloatingTextEditor
        stageScale={stageScale}
        panOffset={panOffset}
        imageWidth={baseImage.size.width}
        imageHeight={baseImage.size.height}
        containerSize={containerSize}
      />

      {/* Hidden state for tests */}
      <div
        data-testid="editor-stage-state"
        data-layers={JSON.stringify(
          layers.map((layer) => ({
            id: layer.id,
            bbox: layer.layout_bbox ?? layer.bbox,
            visible: layer.visible ?? true,
            locked: layer.locked ?? false,
            text: layer.traduzido ?? layer.translated ?? "",
            color: layer.estilo?.cor,
          })),
        )}
        className="sr-only"
      />

      <div className="flex h-full items-center justify-center overflow-hidden px-6 py-4">
        {currentPage && baseImage.image && width > 0 && height > 0 ? (
          <div
            style={{
              transform: `translate(${panOffset.x}px, ${panOffset.y}px) scale(${stageScale})`,
              transformOrigin: "center center",
              transition: panSession ? "none" : "transform 0.12s ease-out",
            }}
            className="rounded-2xl border border-border shadow-[0_20px_60px_rgba(0,0,0,0.45)]"
          >
            <Stage
              width={width}
              height={height}
              onMouseDown={(event) => {
                handleStageMouseDown(event);
                if (event.cancelBubble) return;
                if (event.target === event.target.getStage()) {
                  selectLayer(null);
                  selectImageLayer(null);
                }
              }}
              onMouseMove={handleStageMouseMove}
              onMouseUp={handleStageMouseUp}
              onMouseEnter={handleStageMouseEnter}
              onMouseLeave={handleStageMouseLeave}
              onTap={(event) => {
                if (event.target === event.target.getStage()) {
                  selectLayer(null);
                  selectImageLayer(null);
                }
              }}
            >
              <Layer>
                <EditorStageBackground image={baseImage.image} width={width} height={height} />
                {maskImage.image && (
                  <EditorBitmapOverlay
                    image={maskImage.image}
                    width={width}
                    height={height}
                    color="#6C5CE7"
                    opacity={(imageLayers.mask?.opacity ?? 1) * 0.65}
                  />
                )}
                {brushImage.image && (
                  <EditorBitmapOverlay
                    image={brushImage.image}
                    width={width}
                    height={height}
                    color={brushColor}
                    opacity={(imageLayers.brush?.opacity ?? 1) * brushOpacity}
                  />
                )}
                <Rect width={width} height={height} fill="rgba(0,0,0,0)" />
              </Layer>

              <Layer>
                {blockDraft && (
                  <Rect
                    x={Math.min(blockDraft.start.x, blockDraft.current.x)}
                    y={Math.min(blockDraft.start.y, blockDraft.current.y)}
                    width={Math.abs(blockDraft.current.x - blockDraft.start.x)}
                    height={Math.abs(blockDraft.current.y - blockDraft.start.y)}
                    cornerRadius={12}
                    stroke="rgba(108, 92, 231, 0.95)"
                    strokeWidth={2}
                    dash={[8, 6]}
                    fill="rgba(108, 92, 231, 0.10)"
                    listening={false}
                  />
                )}
                {paintStroke.length > 0 && (
                  <Line
                    points={paintStroke.flatMap(([x, y]) => [x, y])}
                    stroke={
                      toolMode === "brush"
                        ? brushColor  // Fase 7: usa brushColor dinâmica
                        : toolMode === "eraser"
                          ? "rgba(255,255,255,0.72)"
                          : "rgba(108, 92, 231, 0.90)"
                    }
                    opacity={toolMode === "brush" ? brushOpacity : 0.85}
                    strokeWidth={Math.max(4, brushSize)}
                    lineCap="round"
                    lineJoin="round"
                    listening={false}
                  />
                )}
                {(toolMode === "brush" || toolMode === "repairBrush" || toolMode === "eraser") &&
                  cursorPoint && (
                    <EditorPaintCursor
                      x={cursorPoint.x}
                      y={cursorPoint.y}
                      radius={brushSize / 2}
                      toolMode={toolMode as "brush" | "repairBrush" | "eraser"}
                    />
                  )}
                {translatedEditing &&
                  layers.map((entry) => (
                    <EditorTextLayer
                      key={entry.id}
                      entry={entry}
                      selected={selectedLayerId === entry.id}
                      hovered={hoveredLayerId === entry.id}
                      showGuides={!faithfulPreview && showOverlays}
                      interactive={toolMode === "select"}
                      onSelect={() => selectLayer(entry.id)}
                      onHover={(isHovered) => hoverLayer(isHovered ? entry.id : null)}
                      onCommitBbox={(before, after) => commitBbox(entry, before, after)}
                    />
                  ))}
                {translatedEditing && <EditorTransformer selectedNodeName={selectedNodeName} />}
              </Layer>

              {/* Fase 8: Lasso em construção */}
              {maskInProgress && maskInProgress.points.length > 0 && (
                <MaskInProgressOverlay
                  points={maskInProgress.points}
                  shape={maskShape}
                />
              )}
            </Stage>
          </div>
        ) : currentPage ? (
          <p className="text-sm text-text-secondary">Carregando imagem...</p>
        ) : (
          <p className="text-text-secondary">Nenhuma pagina carregada</p>
        )}
      </div>
    </div>
  );
}

import { Layer, Line, Rect, Stage } from "react-konva";
import { getRenderPreviewStateForPage, useEditorStore } from "../../../lib/stores/editorStore";
import { EditorStageBackground } from "./EditorStageBackground";
import { EditorTextLayer } from "./EditorTextLayer";
import { EditorTransformer } from "./EditorTransformer";
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
      ? "Renderizando preview final"
      : state.status === "error"
        ? "Preview final pendente"
        : state.status === "stale"
          ? "Preview final desatualizado"
          : state.previewPath
            ? "Preview final fiel"
            : null;
  if (!label) return null;

  const fresh = state.status === "fresh" && !!state.previewPath;
  return (
    <div className="pointer-events-none absolute left-4 top-4 z-20">
      <div
        data-testid="editor-preview-status"
        data-status={state.status}
        className={`rounded-full border px-3 py-1 text-[11px] backdrop-blur ${
          fresh
            ? "border-status-success/25 bg-status-success/10 text-status-success"
            : "border-status-warning/25 bg-status-warning/10 text-status-warning"
        }`}
      >
        {label}
      </div>
    </div>
  );
}

export function EditorStage() {
  const controller = useEditorStageController();
  const {
    containerRef,
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
  } = controller;

  const width = baseImage.size.width;
  const height = baseImage.size.height;

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
      className="relative flex-1 overflow-hidden bg-[radial-gradient(circle_at_top,_rgba(72,176,255,0.08),_transparent_38%),linear-gradient(180deg,_rgba(255,255,255,0.02),_transparent_28%)]"
      style={{ cursor: viewportCursor }}
      onMouseDownCapture={handleViewportMouseDown}
    >
      <div className="pointer-events-none absolute inset-x-0 bottom-3 z-20 flex justify-center px-4">
        <div className="rounded-full border border-border bg-black/45 px-3 py-1 text-[11px] text-text-secondary backdrop-blur">
          {toolMode === "block"
            ? "Arraste para criar uma nova camada de texto"
            : toolMode === "brush" || toolMode === "repairBrush" || toolMode === "eraser"
              ? "Pintura ativa no Stage"
              : "Ctrl+scroll: zoom | scroll: mover | Space+drag: mover | Arraste blocos para reposicionar"}
        </div>
      </div>

      <StageStatusBadge />
      {bitmapInspection && (
        <div className="pointer-events-none absolute right-4 top-4 z-20">
          <div className="rounded-full border border-accent-cyan/25 bg-accent-cyan/10 px-3 py-1 text-[11px] text-accent-cyan backdrop-blur">
            Inspecionando bitmap: {selectedImageLayerKey}
          </div>
        </div>
      )}
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
            className="rounded-2xl border border-border shadow-[0_20px_60px_rgba(0,0,0,0.55)]"
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
                  <EditorStageBackground image={maskImage.image} width={width} height={height} />
                )}
                {brushImage.image && (
                  <EditorStageBackground image={brushImage.image} width={width} height={height} />
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
                    stroke="rgba(124, 92, 255, 0.95)"
                    strokeWidth={2}
                    dash={[8, 6]}
                    fill="rgba(124, 92, 255, 0.10)"
                    listening={false}
                  />
                )}
                {paintStroke.length > 0 && (
                  <Line
                    points={paintStroke.flatMap(([x, y]) => [x, y])}
                    stroke={
                      toolMode === "brush"
                        ? "rgba(72, 176, 255, 0.90)"
                        : toolMode === "eraser"
                          ? "rgba(255,255,255,0.72)"
                          : "rgba(124, 92, 255, 0.90)"
                    }
                    strokeWidth={Math.max(4, brushSize)}
                    lineCap="round"
                    lineJoin="round"
                    opacity={0.85}
                    listening={false}
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

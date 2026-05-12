import { lazy, Suspense, useState } from "react";
import { Image as KonvaImage, Layer, Line, Rect, Stage } from "react-konva";
import { useEditorStore } from "../../../lib/stores/editorStore";
import { EditorBitmapOverlay } from "./EditorBitmapOverlay";
import { EditorPaintCursor } from "./EditorPaintCursor";
import { EditorRotationHotspots } from "./EditorRotationHotspots";
import { EditorStageBackground } from "./EditorStageBackground";
import { EditorTextLayer } from "./EditorTextLayer";
import { EditorTransformer } from "./EditorTransformer";
import { MaskInProgressOverlay } from "./MaskInProgressOverlay";
import { useEditorStageController } from "./useEditorStageController";

const FloatingTextEditor = lazy(async () => {
  const mod = await import("./FloatingTextEditor");
  return { default: mod.FloatingTextEditor };
});

function isE2E() {
  const meta = import.meta as ImportMeta & { env?: Record<string, string | undefined> };
  if ((meta.env?.VITE_E2E ?? "") === "1") return true;
  return typeof navigator !== "undefined" && navigator.webdriver === true;
}

// StageStatusBadge removido — canvas Konva é WYSIWYG, não há "preview" separado.

export function EditorStage() {
  const e2e = isE2E();
  const controller = useEditorStageController();
  const [draftTextRotation, setDraftTextRotation] = useState<{ layerId: string; rotation: number } | null>(null);
  const brushColor = useEditorStore((s) => s.brushColor);
  const brushOpacity = useEditorStore((s) => s.brushOpacity);
  // Selectors estáveis (escalares) para evitar loop por nova referência {} a cada render
  const maskLayerOpacity = useEditorStore((s) => s.currentPage?.image_layers?.mask?.opacity ?? 1);
  const brushLayerOpacity = useEditorStore((s) => s.currentPage?.image_layers?.brush?.opacity ?? 1);
  const {
    containerRef,
    containerSize,
    currentPage,
    currentPageIndex,
    toolMode,
    showOverlays,
    brushSize,
    panOffset,
    panSession,
    viewportCursor,
    baseImage,
    maskImage,
    brushImage,
    recoveryPreviewPatches,
    reinpaintPreviewPatches,
    blockDraft,
    paintStroke,
    layers,
    selectedLayerId,
    hoveredLayerId,
    selectedNodeName,
    stageScale,
    faithfulPreview,
    translatedEditing,
    selectLayer,
    hoverLayer,
    commitTextLayerTransform,
    handleViewportMouseDown,
    handleViewportContextMenu,
    handleStageMouseDown,
    handleStageMouseMove,
    handleStageMouseUp,
    handleStageMouseEnter,
    handleStageMouseLeave,
    cursorViewportPoint,
    maskInProgress,
    maskShape,
  } = controller;

  const width = baseImage.size.width;
  const height = baseImage.size.height;
  const selectedLayer = selectedLayerId ? layers.find((layer) => layer.id === selectedLayerId) ?? null : null;

  const hintText =
    toolMode === "block"
      ? "Arraste para criar uma nova camada de texto"
      : toolMode === "brush" || toolMode === "repairBrush" || toolMode === "reinpaintBrush" || toolMode === "eraser"
        ? "Pintura ativa no Stage"
        : "Ctrl+scroll: zoom · Scroll: navegar paginas · Space+drag: mover";

  return (
    <div
      ref={containerRef}
      data-testid="editor-stage"
      data-base-kind={
        currentPage?.image_layers?.inpaint?.path
            ? "inpaint"
            : currentPage?.image_layers?.base?.path
              ? "base"
              : "original"
      }
      data-text-editing={translatedEditing ? "true" : "false"}
      className="relative flex-1 overflow-hidden bg-[radial-gradient(circle_at_top,_rgba(108,92,231,0.06),_transparent_40%)]"
      style={{ cursor: viewportCursor }}
      onMouseDownCapture={handleViewportMouseDown}
      onContextMenu={handleViewportContextMenu}
    >
      {(toolMode === "brush" || toolMode === "repairBrush" || toolMode === "reinpaintBrush" || toolMode === "eraser") &&
        cursorViewportPoint && (
          <EditorPaintCursor
            x={cursorViewportPoint.x}
            y={cursorViewportPoint.y}
            radius={(brushSize / 2) * stageScale}
            toolMode={toolMode as "brush" | "repairBrush" | "reinpaintBrush" | "eraser"}
          />
        )}

      {/* Bottom hint */}
      <div className="pointer-events-none absolute inset-x-0 bottom-4 z-20 flex justify-center px-4">
        <div className="rounded-lg border border-border bg-bg-secondary/80 px-3 py-1 text-[10px] text-text-muted backdrop-blur-md">
          {hintText}
        </div>
      </div>

      {/* Status badges (StageStatusBadge removido — canvas é WYSIWYG) */}
      {/* Fase 5: FloatingTextEditor — painel flutuante para edição rápida */}
      {selectedLayerId && (
        <Suspense fallback={null}>
          <FloatingTextEditor
            stageScale={stageScale}
            panOffset={panOffset}
            imageWidth={baseImage.size.width}
            imageHeight={baseImage.size.height}
            containerSize={containerSize}
          />
        </Suspense>
      )}

      {/* Hidden state for tests */}
      {e2e && (
        <div
          data-testid="editor-stage-state"
          data-page-index={String(currentPageIndex)}
          data-layers={JSON.stringify(
            layers.map((layer) => ({
              id: layer.id,
              bbox: layer.layout_bbox ?? layer.bbox,
              visible: layer.visible ?? true,
              locked: layer.locked ?? false,
              text: layer.traduzido ?? layer.translated ?? "",
              color: layer.estilo?.cor,
              rotation: layer.estilo?.rotacao ?? 0,
            })),
          )}
          className="sr-only"
        />
      )}

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
                }
              }}
              onMouseMove={handleStageMouseMove}
              onMouseUp={handleStageMouseUp}
              onMouseEnter={handleStageMouseEnter}
              onMouseLeave={handleStageMouseLeave}
              onTap={(event) => {
                if (event.target === event.target.getStage()) {
                  selectLayer(null);
                }
              }}
            >
              <Layer>
                <EditorStageBackground image={baseImage.image} width={width} height={height} />
                {recoveryPreviewPatches.map((patch) => (
                  <KonvaImage
                    key={patch.id}
                    image={patch.image}
                    x={patch.x}
                    y={patch.y}
                    width={patch.width}
                    height={patch.height}
                    listening={false}
                  />
                ))}
                {reinpaintPreviewPatches.map((patch) => (
                  <KonvaImage
                    key={patch.id}
                    image={patch.image}
                    x={patch.x}
                    y={patch.y}
                    width={patch.width}
                    height={patch.height}
                    listening={false}
                  />
                ))}
                {maskImage.image && (
                  <EditorBitmapOverlay
                    image={maskImage.image}
                    width={width}
                    height={height}
                    color="#6C5CE7"
                    opacity={maskLayerOpacity * 0.65}
                  />
                )}
                {brushImage.image && (
                  <KonvaImage
                    image={brushImage.image}
                    x={0}
                    y={0}
                    width={width}
                    height={height}
                    opacity={brushLayerOpacity}
                    listening={false}
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
                          : toolMode === "reinpaintBrush"
                            ? "rgba(34, 211, 238, 0.90)"
                            : "rgba(108, 92, 231, 0.90)"
                    }
                    opacity={toolMode === "brush" ? brushOpacity : 0.85}
                    strokeWidth={Math.max(4, brushSize)}
                    lineCap="round"
                    lineJoin="round"
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
                      draftRotation={draftTextRotation?.layerId === entry.id ? draftTextRotation.rotation : null}
                      onSelect={() => selectLayer(entry.id)}
                      onHover={(isHovered) => hoverLayer(isHovered ? entry.id : null)}
                      onCommitTransform={(before, after) => commitTextLayerTransform(entry, before, after)}
                    />
                  ))}
                {translatedEditing && <EditorTransformer selectedNodeName={selectedNodeName} />}
                {translatedEditing && toolMode === "select" && selectedLayer && (
                  <EditorRotationHotspots
                    entry={selectedLayer}
                    draftRotation={draftTextRotation?.layerId === selectedLayer.id ? draftTextRotation.rotation : null}
                    onDraftRotation={(rotation) =>
                      setDraftTextRotation(rotation === null ? null : { layerId: selectedLayer.id, rotation })
                    }
                    onCommitTransform={(before, after) => commitTextLayerTransform(selectedLayer, before, after)}
                  />
                )}
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

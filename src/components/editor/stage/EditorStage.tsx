import { Fragment, lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import { ArrowDown, ArrowUp } from "lucide-react";
import { Group, Image as KonvaImage, Layer, Line, Rect, Stage } from "react-konva";
import { loadImageSource, preloadImageSource } from "../../../lib/imageSource";
import type { PageData } from "../../../lib/stores/appStore";
import { useEditorStore, type EditorViewMode } from "../../../lib/stores/editorStore";
import { LassoContextMenu } from "../LassoContextMenu";
import { EditorBitmapOverlay } from "./EditorBitmapOverlay";
import { EditorPaintCursor } from "./EditorPaintCursor";
import { EditorRotationHotspots } from "./EditorRotationHotspots";
import { EditorSnapGuides } from "./EditorSnapGuides";
import { EditorStageBackground } from "./EditorStageBackground";
import { EditorTextLayer } from "./EditorTextLayer";
import { EditorTransformer } from "./EditorTransformer";
import { LassoSelectionOverlay } from "./LassoSelectionOverlay";
import { MaskInProgressOverlay } from "./MaskInProgressOverlay";
import { editingBaseImagePath, originalImagePath } from "./renderModeUtils";
import type { SnapGuide } from "./snapGuides";
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

const LASSO_CONTEXT_MENU_SIZE = { width: 190, height: 232 };

// StageStatusBadge removido — canvas Konva é WYSIWYG, não há "preview" separado.

function readerImagePathForPage(page: PageData, viewMode: EditorViewMode) {
  if (viewMode === "original") return originalImagePath(page);
  return editingBaseImagePath(page);
}

function EditorReaderStaticPage({
  page,
  pageIndex,
  viewMode,
  pageWidth,
  setPageRef,
}: {
  page: PageData;
  pageIndex: number;
  viewMode: EditorViewMode;
  pageWidth: number;
  setPageRef: (index: number, node: HTMLDivElement | null) => void;
}) {
  const [src, setSrc] = useState<string | null>(null);
  const [shouldLoad, setShouldLoad] = useState(false);
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    setShouldLoad(false);
    setSrc(null);
  }, [pageIndex]);

  useEffect(() => {
    if (shouldLoad) return;
    const node = containerRef.current;
    if (!node) return;
    if (typeof IntersectionObserver === "undefined") {
      setShouldLoad(true);
      return;
    }

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (!entry.isIntersecting) return;
        setShouldLoad(true);
        observer.disconnect();
      },
      { rootMargin: "1400px 0px" },
    );
    observer.observe(node);

    return () => observer.disconnect();
  }, [shouldLoad]);

  useEffect(() => {
    if (!shouldLoad) return;
    const path = readerImagePathForPage(page, viewMode);
    if (!path) {
      setSrc(null);
      return;
    }

    let cancelled = false;
    let revokeSource: (() => void) | null = null;
    setSrc(null);
    loadImageSource(path, "image/png", 0)
      .then((loaded) => {
        if (cancelled) {
          loaded.revoke?.();
          return;
        }
        revokeSource = loaded.revoke ?? null;
        setSrc(loaded.src);
      })
      .catch((error) => {
        console.error("Falha ao carregar pagina do leitor do editor:", error);
        if (!cancelled) setSrc(null);
      });

    return () => {
      cancelled = true;
      if (revokeSource) {
        const revoke = revokeSource;
        window.setTimeout(revoke, 100);
      }
    };
  }, [page, shouldLoad, viewMode]);

  return (
    <div
      ref={(node) => {
        containerRef.current = node;
        setPageRef(pageIndex, node);
      }}
      data-testid={`editor-reader-page-${pageIndex + 1}`}
      className="m-0 flex w-full min-w-0 justify-center p-0 leading-none"
    >
      {src ? (
        <img
          src={src}
          alt={`Pagina ${page.numero}`}
          draggable={false}
          loading="lazy"
          decoding="async"
          className="block h-auto max-w-none select-none"
          style={{ width: `${Math.max(1, Math.round(pageWidth))}px` }}
        />
      ) : (
        <div
          className="flex h-64 items-center justify-center text-sm text-text-secondary"
          style={{ width: `${Math.max(1, Math.round(pageWidth))}px` }}
        >
          Carregando imagem...
        </div>
      )}
    </div>
  );
}

function EditorReaderPageBarrier({
  upperPageIndex,
  currentPageIndex,
  onGoToPage,
  onPreloadPage,
}: {
  upperPageIndex: number;
  currentPageIndex: number;
  onGoToPage: (pageIndex: number) => void;
  onPreloadPage: (pageIndex: number) => void;
}) {
  const lowerPageIndex = upperPageIndex + 1;
  const buttonClass = (active: boolean) =>
    `flex h-8 w-8 items-center justify-center rounded-full border transition-smooth ${
      active
        ? "border-accent-purple/40 bg-accent-purple/15 text-accent-purple"
        : "border-border bg-bg-secondary text-text-secondary hover:border-accent-purple/50 hover:text-text-primary"
    }`;

  return (
    <div
      data-testid={`editor-reader-barrier-${upperPageIndex + 1}-${lowerPageIndex + 1}`}
      className="flex h-24 w-full items-center justify-center px-10"
    >
      <div className="flex w-full max-w-md items-center gap-3">
        <div className="h-px flex-1 bg-border/70" />
        <div className="flex items-center gap-2 rounded-full border border-border bg-bg-primary/80 p-1 shadow-[0_10px_30px_rgba(0,0,0,0.25)] backdrop-blur-md">
          <button
            type="button"
            data-testid={`editor-reader-go-up-${upperPageIndex + 1}`}
            title={`Editar pagina ${upperPageIndex + 1}`}
            disabled={currentPageIndex === upperPageIndex}
            onFocus={() => onPreloadPage(upperPageIndex)}
            onMouseEnter={() => onPreloadPage(upperPageIndex)}
            onClick={() => onGoToPage(upperPageIndex)}
            className={buttonClass(currentPageIndex === upperPageIndex)}
          >
            <ArrowUp size={14} />
          </button>
          <button
            type="button"
            data-testid={`editor-reader-go-down-${lowerPageIndex + 1}`}
            title={`Editar pagina ${lowerPageIndex + 1}`}
            disabled={currentPageIndex === lowerPageIndex}
            onFocus={() => onPreloadPage(lowerPageIndex)}
            onMouseEnter={() => onPreloadPage(lowerPageIndex)}
            onClick={() => onGoToPage(lowerPageIndex)}
            className={buttonClass(currentPageIndex === lowerPageIndex)}
          >
            <ArrowDown size={14} />
          </button>
        </div>
        <div className="h-px flex-1 bg-border/70" />
      </div>
    </div>
  );
}

export function EditorStage() {
  const e2e = isE2E();
  const controller = useEditorStageController();
  const [draftTextRotation, setDraftTextRotation] = useState<{ layerId: string; rotation: number } | null>(null);
  const [snapGuides, setSnapGuides] = useState<SnapGuide[]>([]);
  const brushColor = useEditorStore((s) => s.brushColor);
  const brushOpacity = useEditorStore((s) => s.brushOpacity);
  // Selectors estáveis (escalares) para evitar loop por nova referência {} a cada render
  const maskLayerOpacity = useEditorStore((s) => s.currentPage?.image_layers?.mask?.opacity ?? 1);
  const brushLayerOpacity = useEditorStore((s) => s.currentPage?.image_layers?.brush?.opacity ?? 1);
  const setCurrentPage = useEditorStore((s) => s.setCurrentPage);
  const {
    containerRef,
    containerSize,
    projectPages,
    currentPage,
    currentPageIndex,
    viewMode,
    toolMode,
    showOverlays,
    brushSize,
    zoom,
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
    activeLassoSelection,
    maskShape,
  } = controller;

  const width = baseImage.size.width;
  const height = baseImage.size.height;
  const selectedLayer = selectedLayerId ? layers.find((layer) => layer.id === selectedLayerId) ?? null : null;
  const readerPageRefs = useRef<Record<number, HTMLDivElement | null>>({});
  const readerScrollSyncFrameRef = useRef<number | null>(null);
  const readerScrollSyncTargetRef = useRef<number | null>(null);
  const readerScrollSuppressionTimeoutRef = useRef<number | null>(null);
  const previousReaderPageIndexRef = useRef(currentPageIndex);
  const suppressReaderScrollSyncRef = useRef(false);
  const readerPageWidth = Math.max(
    1,
    Math.round(width > 0 ? width * stageScale : Math.max(1, containerSize.width - 48) * zoom),
  );
  const lassoMenuPosition = useMemo(() => {
    if (!activeLassoSelection || activeLassoSelection.pageIndex !== currentPageIndex) return null;
    const [x, y, boxWidth, boxHeight] = activeLassoSelection.bbox;
    const pageWidth = Math.max(1, Math.round(width * stageScale));
    const pageHeight = Math.max(1, Math.round(height * stageScale));
    const centerX = panOffset.x + (x + boxWidth / 2) * stageScale;
    const left = Math.max(
      8,
      Math.min(pageWidth - LASSO_CONTEXT_MENU_SIZE.width - 8, centerX - LASSO_CONTEXT_MENU_SIZE.width / 2),
    );
    const below = panOffset.y + (y + boxHeight) * stageScale + 10;
    const above = panOffset.y + y * stageScale - LASSO_CONTEXT_MENU_SIZE.height - 10;
    const top =
      below + LASSO_CONTEXT_MENU_SIZE.height <= pageHeight - 8
        ? below
        : Math.max(8, Math.min(pageHeight - LASSO_CONTEXT_MENU_SIZE.height - 8, above));

    return { x: Math.round(left), y: Math.round(top) };
  }, [activeLassoSelection, currentPageIndex, height, panOffset.x, panOffset.y, stageScale, width]);

  useEffect(() => {
    setSnapGuides([]);
  }, [currentPageIndex, selectedLayerId, toolMode]);

  const hintText =
    toolMode === "block"
      ? "Arraste para criar uma nova camada de texto"
      : toolMode === "brush" || toolMode === "repairBrush" || toolMode === "reinpaintBrush" || toolMode === "eraser"
        ? "Pintura ativa no Stage"
        : "Ctrl+scroll: zoom · Scroll: navegar paginas · Space+drag: mover";

  const paintStrokePreview =
    paintStroke.length > 0 ? (
      <Line
        points={paintStroke.flatMap(([x, y]) => [x, y])}
        stroke={
          toolMode === "brush"
            ? brushColor
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
    ) : null;
  const clippedPaintStrokePreview =
    paintStrokePreview && activeLassoSelection?.pageIndex === currentPageIndex ? (
      <Group
        listening={false}
        clipFunc={(ctx) => {
          const points = activeLassoSelection.points;
          if (points.length < 3) return;
          ctx.beginPath();
          ctx.moveTo(points[0][0], points[0][1]);
          for (const [x, y] of points.slice(1)) {
            ctx.lineTo(x, y);
          }
          ctx.closePath();
        }}
      >
        {paintStrokePreview}
      </Group>
    ) : (
      paintStrokePreview
    );

  const setReaderPageRef = (index: number, node: HTMLDivElement | null) => {
    readerPageRefs.current[index] = node;
  };

  const releaseReaderScrollSuppressionSoon = (delayMs = 220) => {
    if (readerScrollSuppressionTimeoutRef.current !== null) {
      window.clearTimeout(readerScrollSuppressionTimeoutRef.current);
    }
    readerScrollSuppressionTimeoutRef.current = window.setTimeout(() => {
      suppressReaderScrollSyncRef.current = false;
      readerScrollSuppressionTimeoutRef.current = null;
    }, delayMs);
  };

  const scrollReaderPageIntoView = (pageIndex: number) => {
    const viewport = containerRef.current;
    const pageNode = readerPageRefs.current[pageIndex];
    if (!viewport || !pageNode) return;

    const viewportRect = viewport.getBoundingClientRect();
    const pageRect = pageNode.getBoundingClientRect();
    const centeredTop =
      viewport.scrollTop + pageRect.top - viewportRect.top - Math.max(0, (viewport.clientHeight - pageRect.height) / 2);
    const maxScrollTop = Math.max(0, viewport.scrollHeight - viewport.clientHeight);
    const nextScrollTop = Math.max(0, Math.min(maxScrollTop, centeredTop));
    const previousScrollBehavior = viewport.style.scrollBehavior;

    suppressReaderScrollSyncRef.current = true;
    viewport.style.scrollBehavior = "auto";
    viewport.scrollTo({ top: nextScrollTop, behavior: "auto" });
    window.requestAnimationFrame(() => {
      viewport.style.scrollBehavior = previousScrollBehavior;
      releaseReaderScrollSuppressionSoon();
    });
  };

  const preloadReaderPage = (pageIndex: number) => {
    const page = projectPages[pageIndex];
    const path = readerImagePathForPage(page, viewMode);
    if (!path) return;
    void preloadImageSource(path, "image/png", 0).catch(() => {});
  };

  useEffect(() => {
    const indexes = [
      currentPageIndex,
      currentPageIndex - 2,
      currentPageIndex - 1,
      currentPageIndex + 1,
      currentPageIndex + 2,
    ];
    for (const index of indexes) {
      if (index < 0 || index >= projectPages.length) continue;
      preloadReaderPage(index);
    }
  }, [currentPageIndex, projectPages, viewMode]);

  useEffect(() => {
    return () => {
      if (readerScrollSyncFrameRef.current !== null) {
        window.cancelAnimationFrame(readerScrollSyncFrameRef.current);
      }
      if (readerScrollSuppressionTimeoutRef.current !== null) {
        window.clearTimeout(readerScrollSuppressionTimeoutRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (previousReaderPageIndexRef.current === currentPageIndex) return;
    previousReaderPageIndexRef.current = currentPageIndex;

    window.requestAnimationFrame(() => {
      scrollReaderPageIntoView(currentPageIndex);
    });
  }, [currentPageIndex]);

  const syncReaderPageFromScroll = () => {
    if (suppressReaderScrollSyncRef.current || projectPages.length <= 1) return;
    const viewport = containerRef.current;
    if (!viewport) return;

    const viewportRect = viewport.getBoundingClientRect();
    const edgeThreshold = viewportRect.height * 0.35;
    const activePageNode = readerPageRefs.current[currentPageIndex];
    if (!activePageNode) return;

    const activePageRect = activePageNode.getBoundingClientRect();
    const nextPageIndex =
      activePageRect.bottom < viewportRect.top + edgeThreshold && currentPageIndex < projectPages.length - 1
        ? currentPageIndex + 1
        : activePageRect.top > viewportRect.bottom - edgeThreshold && currentPageIndex > 0
          ? currentPageIndex - 1
          : currentPageIndex;

    if (nextPageIndex === currentPageIndex || nextPageIndex === readerScrollSyncTargetRef.current) return;

    const preservedZoom = useEditorStore.getState().zoom;
    suppressReaderScrollSyncRef.current = true;
    readerScrollSyncTargetRef.current = nextPageIndex;
    const pageChange = setCurrentPage(nextPageIndex);
    useEditorStore.getState().setZoom(preservedZoom);
    void pageChange
      .then(() => {
        useEditorStore.getState().setZoom(preservedZoom);
      })
      .catch((error) => {
        console.error("Falha ao sincronizar pagina pelo scroll do leitor vertical:", error);
      })
      .finally(() => {
        if (readerScrollSyncTargetRef.current === nextPageIndex) {
          readerScrollSyncTargetRef.current = null;
        }
        releaseReaderScrollSuppressionSoon();
      });
  };

  const handleReaderScroll = () => {
    if (readerScrollSyncFrameRef.current !== null) return;
    readerScrollSyncFrameRef.current = window.requestAnimationFrame(() => {
      readerScrollSyncFrameRef.current = null;
      syncReaderPageFromScroll();
    });
  };

  const goToReaderPage = (pageIndex: number) => {
    if (pageIndex < 0 || pageIndex >= projectPages.length || pageIndex === currentPageIndex) return;
    const preservedZoom = useEditorStore.getState().zoom;
    const pageChange = setCurrentPage(pageIndex);
    useEditorStore.getState().setZoom(preservedZoom);
    void pageChange
      .then(() => {
        useEditorStore.getState().setZoom(preservedZoom);
      })
      .catch((error) => {
        console.error("Falha ao trocar pagina pelo leitor vertical do editor:", error);
      });
  };

  return (
    <div
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
          data-lasso-progress-points={String(maskInProgress?.points.length ?? 0)}
          data-lasso-selection={
            activeLassoSelection
              ? JSON.stringify({
                  pageIndex: activeLassoSelection.pageIndex,
                  points: activeLassoSelection.points,
                  bbox: activeLassoSelection.bbox,
                })
              : ""
          }
          data-layers={JSON.stringify(
            layers.map((layer) => ({
              id: layer.id,
              bbox: layer.layout_bbox ?? layer.bbox,
              visible: layer.visible ?? true,
              locked: layer.locked ?? false,
              text: layer.traduzido ?? layer.translated ?? "",
              color: layer.estilo?.cor,
              gradient: layer.estilo?.cor_gradiente ?? [],
              rotation: layer.estilo?.rotacao ?? 0,
            })),
          )}
          className="sr-only"
        />
      )}

      <div
        ref={containerRef}
        data-testid="editor-reader-viewport"
        className="h-full overflow-y-auto overflow-x-hidden scroll-smooth"
        onScroll={handleReaderScroll}
        onMouseDownCapture={handleViewportMouseDown}
        onContextMenu={handleViewportContextMenu}
      >
        <div className="flex min-h-full w-full flex-col items-center overflow-visible p-0">
        {projectPages.slice(0, currentPageIndex).map((page, index) => (
          <Fragment key={`reader-before-${index}`}>
            <EditorReaderStaticPage
              page={page}
              pageIndex={index}
              viewMode={viewMode}
              pageWidth={readerPageWidth}
              setPageRef={setReaderPageRef}
            />
            {index < projectPages.length - 1 && (
              <EditorReaderPageBarrier
                upperPageIndex={index}
                currentPageIndex={currentPageIndex}
                onGoToPage={goToReaderPage}
                onPreloadPage={preloadReaderPage}
              />
            )}
          </Fragment>
        ))}

        {currentPage && baseImage.image && width > 0 && height > 0 ? (
          <div
            ref={(node) => setReaderPageRef(currentPageIndex, node)}
            data-testid={`editor-reader-page-${currentPageIndex + 1}`}
            className="m-0 flex w-full min-w-0 justify-center p-0 leading-none"
            style={{ height: `${Math.max(1, Math.round(height * stageScale))}px` }}
          >
            <div
              className="relative shrink-0 overflow-visible leading-none"
              style={{
                width: `${Math.max(1, Math.round(width * stageScale))}px`,
                height: `${Math.max(1, Math.round(height * stageScale))}px`,
              }}
            >
              {lassoMenuPosition && (
                <LassoContextMenu
                  x={lassoMenuPosition.x}
                  y={lassoMenuPosition.y}
                  position="absolute"
                  onClose={() => undefined}
                />
              )}
              <div
                style={{
                  width,
                  height,
                  transform: `translate(${panOffset.x}px, ${panOffset.y}px) scale(${stageScale})`,
                  transformOrigin: "top left",
                  transition: panSession ? "none" : "transform 0.12s ease-out",
                }}
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
                {clippedPaintStrokePreview}
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
                      pageSize={{ width, height }}
                      snapLayers={layers}
                      onSelect={() => selectLayer(entry.id)}
                      onHover={(isHovered) => hoverLayer(isHovered ? entry.id : null)}
                      onCommitTransform={(before, after) => commitTextLayerTransform(entry, before, after)}
                      onSnapGuidesChange={setSnapGuides}
                    />
                  ))}
                <EditorSnapGuides guides={snapGuides} />
                {translatedEditing && toolMode === "select" && selectedLayer && (
                  <EditorTransformer
                    selectedNodeName={selectedNodeName}
                    pageSize={{ width, height }}
                    selectedLayerId={selectedLayer.id}
                    snapLayers={layers}
                    disabled={selectedLayer.locked === true}
                    onSnapGuidesChange={setSnapGuides}
                  />
                )}
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
              {activeLassoSelection && activeLassoSelection.pageIndex === currentPageIndex && (
                <LassoSelectionOverlay selection={activeLassoSelection} />
              )}
              {maskInProgress && maskInProgress.points.length > 0 && (
                <MaskInProgressOverlay
                  points={maskInProgress.points}
                  shape={maskShape}
                />
            )}
            </Stage>
              </div>
            </div>
          </div>
        ) : currentPage ? (
          <div
            ref={(node) => setReaderPageRef(currentPageIndex, node)}
            data-testid={`editor-reader-page-${currentPageIndex + 1}`}
            className="m-0 flex w-full min-w-0 justify-center p-0 leading-none"
          >
            <div
              className="flex h-64 items-center justify-center text-sm text-text-secondary"
              style={{ width: `${readerPageWidth}px` }}
            >
              Carregando imagem...
            </div>
          </div>
        ) : (
          <p className="text-text-secondary">Nenhuma pagina carregada</p>
        )}
        {currentPageIndex < projectPages.length - 1 && (
          <EditorReaderPageBarrier
            upperPageIndex={currentPageIndex}
            currentPageIndex={currentPageIndex}
            onGoToPage={goToReaderPage}
            onPreloadPage={preloadReaderPage}
          />
        )}
        {projectPages.slice(currentPageIndex + 1).map((page, offset) => {
          const pageIndex = currentPageIndex + offset + 1;
          return (
            <Fragment key={`reader-after-${pageIndex}`}>
              <EditorReaderStaticPage
                page={page}
                pageIndex={pageIndex}
                viewMode={viewMode}
                pageWidth={readerPageWidth}
                setPageRef={setReaderPageRef}
              />
              {pageIndex < projectPages.length - 1 && (
                <EditorReaderPageBarrier
                  upperPageIndex={pageIndex}
                  currentPageIndex={currentPageIndex}
                  onGoToPage={goToReaderPage}
                  onPreloadPage={preloadReaderPage}
                />
              )}
            </Fragment>
          );
        })}
      </div>
      </div>
    </div>
  );
}

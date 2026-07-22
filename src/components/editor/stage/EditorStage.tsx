import { forwardRef, Fragment, lazy, Suspense, useEffect, useImperativeHandle, useMemo, useRef, useState } from "react";
import { ArrowDown, ArrowUp } from "lucide-react";
import { Image as KonvaImage, Layer, Rect, Stage } from "react-konva";
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
import { strokePassesForHardness } from "./bitmapStrokePreview";
import { editingBaseImagePath, isStudioBitmapCompositeActive, originalImagePath } from "./renderModeUtils";
import type { SnapGuide } from "./snapGuides";
import { useEditorStageController } from "./useEditorStageController";
import type { EditorMode } from "../editorMode";
import type { EditorSceneVisualNode } from "./editorSceneVisual";

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

type PaintStrokeCanvasOverlayHandle = {
  begin: (point: [number, number]) => void;
  append: (point: [number, number]) => void;
  clear: () => void;
};

const PaintStrokeCanvasOverlay = forwardRef<PaintStrokeCanvasOverlayHandle, {
  width: number;
  height: number;
  toolMode: string;
  brushSize: number;
  brushColor: string;
  brushOpacity: number;
  brushHardness: number;
  clipPolygon?: [number, number][];
}>(function PaintStrokeCanvasOverlay({
  width,
  height,
  toolMode,
  brushSize,
  brushColor,
  brushOpacity,
  brushHardness,
  clipPolygon,
}, ref) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const lastPointRef = useRef<[number, number] | null>(null);

  const clear = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);
    lastPointRef.current = null;
  };

  const drawSegment = (from: [number, number], to: [number, number]) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.save();
    if (clipPolygon && clipPolygon.length >= 3) {
      ctx.beginPath();
      ctx.moveTo(clipPolygon[0][0], clipPolygon[0][1]);
      for (const [x, y] of clipPolygon.slice(1)) {
        ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.clip();
    }

    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle =
      toolMode === "brush"
        ? brushColor
        : toolMode === "eraser"
          ? "rgba(255,255,255,0.72)"
          : toolMode === "reinpaintBrush"
            ? "rgba(34, 211, 238, 0.90)"
            : "rgba(108, 92, 231, 0.90)";

    const opacity = toolMode === "brush" ? brushOpacity : 0.85;
    for (const pass of strokePassesForHardness({ brushSize: Math.max(4, brushSize), opacity, hardness: brushHardness })) {
      ctx.globalAlpha = pass.alpha;
      ctx.lineWidth = pass.width;
      ctx.beginPath();
      ctx.moveTo(from[0], from[1]);
      ctx.lineTo(to[0], to[1]);
      ctx.stroke();
    }
    ctx.restore();
  };

  useImperativeHandle(ref, () => ({
    begin(point) {
      clear();
      lastPointRef.current = point;
      drawSegment(point, [point[0] + 0.01, point[1] + 0.01]);
    },
    append(point) {
      const lastPoint = lastPointRef.current;
      if (!lastPoint) {
        lastPointRef.current = point;
        drawSegment(point, [point[0] + 0.01, point[1] + 0.01]);
        return;
      }
      drawSegment(lastPoint, point);
      lastPointRef.current = point;
    },
    clear,
  }), [brushColor, brushHardness, brushOpacity, brushSize, clipPolygon, height, toolMode, width]);

  useEffect(() => {
    clear();
  }, [height, width]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="pointer-events-none absolute inset-0 z-10"
      aria-hidden="true"
    />
  );
});

function ProcessRegionOverlayNode({
  cropPath,
  bbox,
  version,
}: {
  cropPath: string;
  bbox: [number, number, number, number];
  version: number;
}) {
  const [src, setSrc] = useState<string | null>(null);
  const [image, setImage] = useState<HTMLImageElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    let revokeSource: (() => void) | null = null;
    setSrc(null);
    setImage(null);
    loadImageSource(cropPath, "image/png", version)
      .then((loaded) => {
        if (cancelled) {
          loaded.revoke?.();
          return;
        }
        revokeSource = loaded.revoke ?? null;
        setSrc(loaded.src);
      })
      .catch((error) => {
        console.error("Falha ao carregar crop do processo:", error);
        if (!cancelled) setSrc(null);
      });

    return () => {
      cancelled = true;
      if (revokeSource) window.setTimeout(revokeSource, 100);
    };
  }, [cropPath, version]);

  useEffect(() => {
    if (!src) {
      setImage(null);
      return;
    }
    const nextImage = new Image();
    nextImage.onload = () => setImage(nextImage);
    nextImage.onerror = () => setImage(null);
    nextImage.src = src;
  }, [src]);

  const [x1, y1, x2, y2] = bbox;
  const overlayWidth = Math.max(1, x2 - x1);
  const overlayHeight = Math.max(1, y2 - y1);
  if (!image) return null;
  return (
    <KonvaImage
      image={image}
      x={x1}
      y={y1}
      width={overlayWidth}
      height={overlayHeight}
      listening={false}
    />
  );
}

function EditorSceneBitmapNode({
  node,
  width,
  height,
}: {
  node: Extract<EditorSceneVisualNode, { kind: "bitmap" }>;
  width: number;
  height: number;
}) {
  const [image, setImage] = useState<HTMLImageElement | null>(null);

  useEffect(() => {
    let disposed = false;
    const next = new Image();
    next.decoding = "async";
    next.onload = () => {
      if (!disposed) setImage(next);
    };
    next.onerror = () => {
      if (!disposed) setImage(null);
    };
    next.src = node.source;
    return () => {
      disposed = true;
    };
  }, [node.source]);

  if (!image) return null;
  const blendMode = node.blendMode === "normal" ? "source-over" : node.blendMode;
  return (
    <KonvaImage
      image={image}
      width={width}
      height={height}
      opacity={node.opacity}
      globalCompositeOperation={blendMode as "source-over" | "multiply" | "screen" | "overlay" | "darken" | "lighten" | "color-dodge" | "color-burn" | "hard-light" | "soft-light" | "difference" | "exclusion" | "hue" | "saturation" | "color" | "luminosity"}
      listening={false}
    />
  );
}

function EditorReaderStaticPage({
  page,
  pageIndex,
  viewMode,
  pageWidth,
  setPageRef,
  onSelectTextLayer,
}: {
  page: PageData;
  pageIndex: number;
  viewMode: EditorViewMode;
  pageWidth: number;
  setPageRef: (index: number, node: HTMLDivElement | null) => void;
  onSelectTextLayer: (pageIndex: number, layerId: string) => void;
}) {
  const [src, setSrc] = useState<string | null>(null);
  const [naturalSize, setNaturalSize] = useState<{ width: number; height: number } | null>(null);
  const [shouldLoad, setShouldLoad] = useState(false);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const textLayers = page.text_layers ?? page.textos ?? [];
  const displayedWidth = Math.max(1, Math.round(pageWidth));
  const scale = naturalSize?.width ? displayedWidth / naturalSize.width : 1;
  const displayedHeight = naturalSize?.height ? Math.max(1, Math.round(naturalSize.height * scale)) : null;

  useEffect(() => {
    setShouldLoad(false);
    setSrc(null);
    setNaturalSize(null);
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
        <div
          className="relative max-w-none"
          style={{
            width: `${displayedWidth}px`,
            height: displayedHeight ? `${displayedHeight}px` : undefined,
          }}
        >
          <img
            src={src}
            alt={`Pagina ${page.numero}`}
            draggable={false}
            loading="lazy"
            decoding="async"
            onLoad={(event) => {
              const image = event.currentTarget;
              setNaturalSize({
                width: image.naturalWidth || displayedWidth,
                height: image.naturalHeight || Math.max(1, image.clientHeight),
              });
            }}
            className="block h-auto max-w-none select-none"
            style={{ width: `${displayedWidth}px` }}
          />
          {naturalSize && textLayers.map((layer) => {
            if (layer.visible === false) return null;
            const bbox = layer.layout_bbox ?? layer.bbox;
            if (!Array.isArray(bbox) || bbox.length !== 4) return null;
            const [x1, y1, x2, y2] = bbox;
            const left = Math.max(0, Math.min(x1, x2) * scale);
            const top = Math.max(0, Math.min(y1, y2) * scale);
            const width = Math.max(8, Math.abs(x2 - x1) * scale);
            const height = Math.max(8, Math.abs(y2 - y1) * scale);
            const label = layer.traduzido || layer.translated || layer.original || "Texto";
            return (
              <button
                key={layer.id}
                type="button"
                title={`Abrir pagina ${pageIndex + 1} e selecionar: ${label}`}
                aria-label={`Abrir pagina ${pageIndex + 1} e selecionar texto`}
                onClick={(event) => {
                  event.stopPropagation();
                  onSelectTextLayer(pageIndex, layer.id);
                }}
                className="absolute rounded-md border border-transparent bg-transparent transition-smooth hover:border-brand/80 hover:bg-brand/10 focus-visible:border-brand/90 focus-visible:bg-brand/15"
                style={{ left, top, width, height }}
              />
            );
          })}
        </div>
      ) : (
        <div
          className="flex h-64 items-center justify-center text-sm text-text-secondary"
          style={{ width: `${displayedWidth}px` }}
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

export function EditorStage({
  mode = "traduzai",
  selectionTargetNodeId = null,
  bitmapCompositeSource = null,
  sceneVisualNodes = null,
  showFloatingTextEditor = true,
}: {
  mode?: EditorMode;
  selectionTargetNodeId?: string | null;
  bitmapCompositeSource?: string | null;
  sceneVisualNodes?: EditorSceneVisualNode[] | null;
  showFloatingTextEditor?: boolean;
}) {
  const e2e = isE2E();
  const controller = useEditorStageController({ mode, selectionTargetNodeId, bitmapCompositeSource });
  const [draftTextRotation, setDraftTextRotation] = useState<{ layerId: string; rotation: number } | null>(null);
  const [snapGuides, setSnapGuides] = useState<SnapGuide[]>([]);
  const brushColor = useEditorStore((s) => s.brushColor);
  const brushOpacity = useEditorStore((s) => s.brushOpacity);
  const brushHardness = useEditorStore((s) => s.brushHardness);
  // Selectors estáveis (escalares) para evitar loop por nova referência {} a cada render
  const maskLayerOpacity = useEditorStore((s) => s.currentPage?.image_layers?.mask?.opacity ?? 1);
  const brushLayerOpacity = useEditorStore((s) => s.currentPage?.image_layers?.brush?.opacity ?? 1);
  const processOverlayVersion = useEditorStore((s) =>
    Math.max(s.bitmapLayerVersions.inpaint ?? 0, s.bitmapLayerVersions.rendered ?? 0),
  );
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
    viewportCursor,
    baseImage,
    maskImage,
    brushImage,
    recoveryPreviewPatches,
    reinpaintPreviewPatches,
    blockDraft,
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
    setPaintPreviewOverlay,
    cursorViewportPoint,
    maskInProgress,
    activeLassoSelection,
    maskShape,
  } = controller;

  const width = baseImage.size.width;
  const height = baseImage.size.height;
  const selectedLayer = selectedLayerId ? layers.find((layer) => layer.id === selectedLayerId) ?? null : null;
  const textLayerById = useMemo(() => new Map(layers.map((layer) => [layer.id, layer])), [layers]);
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
    if (mode === "studio") return null;
    if (toolMode !== "mask") return null;
    if (!activeLassoSelection || activeLassoSelection.pageIndex !== currentPageIndex) return null;
    const [x1, y1, x2, y2] = activeLassoSelection.bbox;
    const pageWidth = Math.max(1, Math.round(width * stageScale));
    const pageHeight = Math.max(1, Math.round(height * stageScale));
    const centerX = panOffset.x + ((x1 + x2) / 2) * stageScale;
    const left = Math.max(
      8,
      Math.min(pageWidth - LASSO_CONTEXT_MENU_SIZE.width - 8, centerX - LASSO_CONTEXT_MENU_SIZE.width / 2),
    );
    const below = panOffset.y + y2 * stageScale + 10;
    const above = panOffset.y + y1 * stageScale - LASSO_CONTEXT_MENU_SIZE.height - 10;
    const top =
      below + LASSO_CONTEXT_MENU_SIZE.height <= pageHeight - 8
        ? below
        : Math.max(8, Math.min(pageHeight - LASSO_CONTEXT_MENU_SIZE.height - 8, above));

    return { x: Math.round(left), y: Math.round(top) };
  }, [activeLassoSelection, currentPageIndex, height, mode, panOffset.x, panOffset.y, stageScale, toolMode, width]);

  useEffect(() => {
    setSnapGuides([]);
  }, [currentPageIndex, selectedLayerId, toolMode]);

  const paintStrokeClipPolygon =
    activeLassoSelection?.pageIndex === currentPageIndex ? activeLassoSelection.points : undefined;
  const paintPreviewRef = useRef<PaintStrokeCanvasOverlayHandle | null>(null);
  const studioCompositeActive = isStudioBitmapCompositeActive(mode, viewMode, bitmapCompositeSource);
  const studioSceneVisualsActive = studioCompositeActive && Boolean(sceneVisualNodes?.length);

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

  const selectTextOnReaderPage = (pageIndex: number, layerId: string) => {
    const pageChange = setCurrentPage(pageIndex);
    readerScrollSyncTargetRef.current = pageIndex;
    void pageChange.then(() => {
      selectLayer(layerId);
      scrollReaderPageIntoView(pageIndex);
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

      {/* Status badges (StageStatusBadge removido — canvas é WYSIWYG) */}
      {/* Fase 5: FloatingTextEditor — painel flutuante para edição rápida */}
      {showFloatingTextEditor && selectedLayerId && (
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
                  feather: activeLassoSelection.feather ?? 0,
                  expansion: activeLassoSelection.expansion ?? 0,
                  targetNodeId: activeLassoSelection.targetNodeId ?? null,
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
              onSelectTextLayer={selectTextOnReaderPage}
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
                  position: "relative",
                  transform: `translate(${panOffset.x}px, ${panOffset.y}px) scale(${stageScale})`,
                  transformOrigin: "top left",
                  transition: "none",
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
              {studioSceneVisualsActive && sceneVisualNodes ? (
                <Layer>
                  {sceneVisualNodes.map((node) => {
                    if (node.kind === "bitmap") {
                      return <EditorSceneBitmapNode key={node.id} node={node} width={width} height={height} />;
                    }
                    const entry = textLayerById.get(node.textLayerId);
                    if (!entry || !translatedEditing) return null;
                    return (
                      <EditorTextLayer
                        key={node.id}
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
                    );
                  })}
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
              ) : (
                <>
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
                <EditorBitmapOverlay
                  brushImage={studioCompositeActive ? null : brushImage.image}
                  brushOpacity={brushLayerOpacity}
                  maskImage={studioCompositeActive ? null : maskImage.image}
                  maskOpacity={maskLayerOpacity * 0.65}
                  width={width}
                  height={height}
                />
                {(currentPage.process_overlays ?? [])
                  .filter((overlay) => overlay.visible !== false)
                  .map((overlay) => (
                    <ProcessRegionOverlayNode
                      key={overlay.id}
                      cropPath={overlay.crop_path}
                      bbox={overlay.bbox}
                      version={processOverlayVersion}
                    />
                  ))}
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
                </>
              )}
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
            {(toolMode === "brush" || toolMode === "repairBrush" || toolMode === "reinpaintBrush" || toolMode === "eraser") && (
              <PaintStrokeCanvasOverlay
                ref={(handle) => {
                  paintPreviewRef.current = handle;
                  setPaintPreviewOverlay(handle);
                }}
                width={width}
                height={height}
                toolMode={toolMode}
                brushSize={brushSize}
                brushColor={brushColor}
                brushOpacity={brushOpacity}
                brushHardness={brushHardness}
                clipPolygon={paintStrokeClipPolygon}
              />
            )}
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
                onSelectTextLayer={selectTextOnReaderPage}
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
      <div className="pointer-events-none absolute inset-x-0 bottom-4 flex justify-center">
        <div className="rounded-full border border-border-primary bg-bg-secondary/90 px-4 py-1 text-[11px] text-text-muted shadow-lg backdrop-blur">
          Ctrl+scroll: zoom - Scroll: navegar paginas - Space+drag: mover
        </div>
      </div>
    </div>
  );
}

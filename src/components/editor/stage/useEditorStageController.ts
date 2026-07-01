import { useEffect, useMemo, useRef, useState } from "react";
import type Konva from "konva";
import { bitmapCache } from "../../../lib/editorHistory";
import {
  bitmapTargetForEditorTool,
  pointFromStageClientRect,
  shouldAppendStrokePoint,
  strokeDirtyBbox,
} from "../../../lib/editorStroke";
import { loadImageSource } from "../../../lib/imageSource";
import { createLassoSelection, rasterizeLassoToPng } from "../../../lib/lassoSelection";
import {
  clampEditorZoom,
  getRenderPreviewStateForPage,
  useEditorStore,
  type TextTransformSnapshot,
} from "../../../lib/stores/editorStore";
import { useAppStore, type PageData, type TextEntry } from "../../../lib/stores/appStore";
import { LayeredBitmapCanvas } from "../../../editor-shared/bitmap/layeredBitmapCanvas";
import { createBitmapStrokePreviewOnCanvas, encodeDataUrl } from "./bitmapStrokePreview";
import {
  applyRecoveryStrokeToCanvas,
  createRecoveryStrokePreviewPatch,
  type RecoveryStrokePreviewPatch,
} from "./recoveryComposite";
import { createHealingBrushMaskPngDataUrl, paddedStrokeBBox } from "./healingBrushMask";
import { displayImagePathForMode, isFaithfulPreviewMode, originalImagePath } from "./renderModeUtils";
import { mergePendingTextEntry } from "./textLayerStyleUtils";
import { useEditorBitmapDrawing } from "./useEditorBitmapDrawing";

const EMPTY_PAGES: PageData[] = [];
const WHEEL_ZOOM_SPEED = 0.0012;

type PaintPreviewOverlayHandle = {
  begin: (point: [number, number]) => void;
  append: (point: [number, number]) => void;
  clear: () => void;
};

function intersectBbox(
  a: [number, number, number, number],
  b: [number, number, number, number],
  width: number,
  height: number,
): [number, number, number, number] | null {
  const x1 = Math.max(0, Math.floor(Math.max(a[0], b[0])));
  const y1 = Math.max(0, Math.floor(Math.max(a[1], b[1])));
  const x2 = Math.min(width, Math.ceil(Math.min(a[2], b[2])));
  const y2 = Math.min(height, Math.ceil(Math.min(a[3], b[3])));
  if (x2 <= x1 || y2 <= y1) return null;
  return [x1, y1, x2, y2];
}

function useObjectUrl(path: string | null | undefined, type = "image/png", version = 0) {
  const [src, setSrc] = useState<string | null>(null);

  useEffect(() => {
    if (!path) {
      setSrc(null);
      return;
    }

    let cancelled = false;
    let revokeSource: (() => void) | null = null;

    loadImageSource(path, type, version)
      .then((loaded) => {
        if (cancelled) {
          loaded.revoke?.();
          return;
        }
        revokeSource = loaded.revoke ?? null;
        setSrc(loaded.src);
      })
      .catch((error) => {
        console.error("Falha ao carregar imagem do editor:", error);
        if (!cancelled) setSrc(null);
      });

    return () => {
      cancelled = true;
      if (revokeSource) {
        const revoke = revokeSource;
        window.setTimeout(revoke, 100);
      }
    };
  }, [path, type, version]);

  return src;
}

function useImageElement(src: string | null) {
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    if (!src) {
      setImage(null);
      setSize({ width: 0, height: 0 });
      return;
    }
    const img = new Image();
    img.onload = () => {
      setImage(img);
      setSize({ width: img.naturalWidth, height: img.naturalHeight });
    };
    img.onerror = () => {
      setImage(null);
      setSize({ width: 0, height: 0 });
    };
    img.src = src;
  }, [src]);

  return { image, size };
}

function preventContextMenu(event: MouseEvent) {
  event.preventDefault();
}

export function useEditorStageController() {
  const containerRef = useRef<HTMLDivElement>(null);
  const projectPages = useAppStore((state) => state.project?.paginas ?? EMPTY_PAGES);
  const currentPage = useEditorStore((state) => state.currentPage);
  const currentPageIndex = useEditorStore((state) => state.currentPageIndex);
  const viewMode = useEditorStore((state) => state.viewMode);
  const toolMode = useEditorStore((state) => state.toolMode);
  const showOverlays = useEditorStore((state) => state.showOverlays);
  const brushSize = useEditorStore((state) => state.brushSize);
  const brushColor = useEditorStore((state) => state.brushColor);
  const brushOpacity = useEditorStore((state) => state.brushOpacity);
  const brushHardness = useEditorStore((state) => state.brushHardness);
  const zoom = useEditorStore((state) => state.zoom);
  const panOffset = useEditorStore((state) => state.panOffset);
  const lastRetypesetTime = useEditorStore((state) => state.lastRetypesetTime);
  const bitmapLayerVersions = useEditorStore((state) => state.bitmapLayerVersions);
  const selectedLayerId = useEditorStore((state) => state.selectedLayerId);
  const hoveredLayerId = useEditorStore((state) => state.hoveredLayerId);
  const pendingEdits = useEditorStore((state) => state.pendingEdits);
  const currentPageKey = useEditorStore((state) => state.currentPageKey());
  const renderPreviewCacheByPageKey = useEditorStore((state) => state.renderPreviewCacheByPageKey);

  const selectLayer = useEditorStore((state) => state.selectLayer);
  const hoverLayer = useEditorStore((state) => state.hoverLayer);
  const setPan = useEditorStore((state) => state.setPan);
  const setBrushSize = useEditorStore((state) => state.setBrushSize);
  const commitTextTransform = useEditorStore((state) => state.commitTextTransform);
  const executeEditorCommand = useEditorStore((state) => state.executeEditorCommand);
  const createTextLayer = useEditorStore((state) => state.createTextLayer);
  const applyBitmapStroke = useEditorStore((state) => state.applyBitmapStroke);
  const healPaintedRegion = useEditorStore((state) => state.healPaintedRegion);
  // Fase 8 — Lasso
  const maskShape = useEditorStore((state) => state.maskShape);
  const maskOp = useEditorStore((state) => state.maskOp);
  const maskInProgress = useEditorStore((state) => state.maskInProgress);
  const activeLassoSelection = useEditorStore((state) => state.activeLassoSelection);
  const setMaskInProgress = useEditorStore((state) => state.setMaskInProgress);
  const setActiveLassoSelection = useEditorStore((state) => state.setActiveLassoSelection);
  const runProcessRegionFromSelection = useEditorStore((state) => state.runProcessRegionFromSelection);
  const bumpBitmapLayerVersion = useEditorStore((state) => state.bumpBitmapLayerVersion);

  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
  const [blockDraft, setBlockDraft] = useState<{
    start: { x: number; y: number };
    current: { x: number; y: number };
  } | null>(null);
  const [isPaintStrokeActive, setIsPaintStrokeActive] = useState(false);
  const [recoveryPreviewPatches, setRecoveryPreviewPatches] = useState<RecoveryStrokePreviewPatch[]>([]);
  const [reinpaintPreviewPatches, setReinpaintPreviewPatches] = useState<RecoveryStrokePreviewPatch[]>([]);
  const [cursorPoint, setCursorPoint] = useState<{ x: number; y: number } | null>(null);
  const [cursorViewportPoint, setCursorViewportPoint] = useState<{ x: number; y: number } | null>(null);
  const [isSpacePressed, setIsSpacePressed] = useState(false);
  // Ref para garantir acesso ao finishPaintStroke mais recente sem criar stale closure
  const finishPaintStrokeRef = useRef<(() => Promise<void>) | null>(null);
  const activeRecoveryPreviewIdsRef = useRef<Set<string>>(new Set());
  const paintStrokeRef = useRef<[number, number][]>([]);
  const paintPreviewOverlayRef = useRef<PaintPreviewOverlayHandle | null>(null);
  const paintPreviewClearTimeoutRef = useRef<number | null>(null);
  const paintPreviewPendingCommitRef = useRef(false);
  const bitmapWorkingCanvasRef = useRef<LayeredBitmapCanvas | null>(null);
  const recoveryWorkingCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const sessionInpaintCacheCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [panSession, setPanSession] = useState<{
    startX: number;
    startY: number;
    originX: number;
    originY: number;
  } | null>(null);
  const [brushSizeDragSession, setBrushSizeDragSession] = useState<{
    startX: number;
    startSize: number;
  } | null>(null);

  useEffect(() => {
    const node = containerRef.current;
    if (!node) return;
    const observer = new ResizeObserver(([entry]) => {
      setContainerSize({
        width: entry.contentRect.width,
        height: entry.contentRect.height,
      });
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    setRecoveryPreviewPatches([]);
    setReinpaintPreviewPatches([]);
    bitmapWorkingCanvasRef.current = null;
    recoveryWorkingCanvasRef.current = null;
    sessionInpaintCacheCanvasRef.current = null;
  }, [currentPageKey]);

  useEffect(() => {
    recoveryWorkingCanvasRef.current = null;
  }, [bitmapLayerVersions.inpaint, currentPage?.image_layers?.inpaint?.path]);

  const cancelPendingPaintPreviewClear = () => {
    if (paintPreviewClearTimeoutRef.current !== null) {
      window.clearTimeout(paintPreviewClearTimeoutRef.current);
      paintPreviewClearTimeoutRef.current = null;
    }
    paintPreviewPendingCommitRef.current = false;
  };

  const clearPaintPreviewOverlay = () => {
    cancelPendingPaintPreviewClear();
    paintPreviewOverlayRef.current?.clear();
  };

  const schedulePaintPreviewClear = (delayMs = 900) => {
    if (paintPreviewClearTimeoutRef.current !== null) {
      window.clearTimeout(paintPreviewClearTimeoutRef.current);
    }
    paintPreviewPendingCommitRef.current = true;
    paintPreviewClearTimeoutRef.current = window.setTimeout(() => {
      paintPreviewClearTimeoutRef.current = null;
      paintPreviewPendingCommitRef.current = false;
      paintPreviewOverlayRef.current?.clear();
    }, delayMs);
  };

  useEffect(() => () => {
    if (paintPreviewClearTimeoutRef.current !== null) {
      window.clearTimeout(paintPreviewClearTimeoutRef.current);
      paintPreviewClearTimeoutRef.current = null;
    }
  }, []);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const active = document.activeElement;
      if (active && ["INPUT", "TEXTAREA", "SELECT"].includes(active.tagName)) return;

      if (event.key === " ") {
        event.preventDefault();
        setIsSpacePressed(true);
        return;
      }

      // Lasso keyboard shortcuts — lê estado fresco via getState para evitar stale closure
      const s = useEditorStore.getState();
      if (s.toolMode === "mask" || s.toolMode === "process") {
        if (event.key === "Escape") {
          event.preventDefault();
          s.setMaskInProgress(null);
          s.setActiveLassoSelection(null);
          return;
        }
        // Enter fecha o polígono (equivalente a clicar no primeiro ponto)
        if (event.key === "Enter" && s.maskShape === "polygonal") {
          event.preventDefault();
          window.dispatchEvent(new CustomEvent("lasso:commit-polygon"));
          return;
        }
      }
    };
    const handleKeyUp = (event: KeyboardEvent) => {
      if (event.key === " ") setIsSpacePressed(false);
    };
    const handleBlur = () => setIsSpacePressed(false);

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
      setPan({
        x: panSession.originX + (event.clientX - panSession.startX),
        y: panSession.originY + (event.clientY - panSession.startY),
      });
    };
    const handleMouseUp = () => setPanSession(null);
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [panSession, setPan]);

  useEffect(() => {
    if (!brushSizeDragSession) return;
    const handleMouseMove = (event: MouseEvent) => {
      event.preventDefault();
      const delta = Math.round((event.clientX - brushSizeDragSession.startX) / 2);
      setBrushSize(brushSizeDragSession.startSize + delta);
    };
    const handleMouseUp = () => setBrushSizeDragSession(null);
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    window.addEventListener("contextmenu", preventContextMenu);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
      window.removeEventListener("contextmenu", preventContextMenu);
    };
  }, [brushSizeDragSession, setBrushSize]);

  // Ref para commitLasso mais recente (evita stale closure no event listener)
  const commitLassoRef = useRef<((pts: Array<[number, number]>) => Promise<void>) | null>(null);

  useEffect(() => {
    const handleCommitPolygon = () => {
      const s = useEditorStore.getState();
      const pts = s.maskInProgress?.points ?? [];
      if (pts.length >= 3 && commitLassoRef.current) {
        void commitLassoRef.current(pts);
      }
    };
    window.addEventListener("lasso:commit-polygon", handleCommitPolygon);
    return () => window.removeEventListener("lasso:commit-polygon", handleCommitPolygon);
  }, []);

  const renderPreviewState = useMemo(
    () => getRenderPreviewStateForPage(currentPageKey, currentPage, renderPreviewCacheByPageKey),
    [currentPage, currentPageKey, renderPreviewCacheByPageKey],
  );
  const displayImagePath = useMemo(
    () => displayImagePathForMode(currentPage, viewMode, renderPreviewState),
    [currentPage, renderPreviewState, viewMode],
  );

  const originalImageSrc = useObjectUrl(originalImagePath(currentPage), "image/png", 0);
  const baseImageSrc = useObjectUrl(displayImagePath, "image/png", lastRetypesetTime);
  // Usar versão dedicada por camada para garantir re-load imediato após stroke
  const maskOverlaySrc = useObjectUrl(
    currentPage?.image_layers?.mask?.visible ? currentPage.image_layers.mask.path : null,
    "image/png",
    bitmapLayerVersions.mask ?? 0,
  );
  const brushOverlaySrc = useObjectUrl(
    currentPage?.image_layers?.brush?.visible ? currentPage.image_layers.brush.path : null,
    "image/png",
    bitmapLayerVersions.brush ?? 0,
  );
  const baseImage = useImageElement(baseImageSrc);
  const originalImage = useImageElement(originalImageSrc);
  const maskImage = useImageElement(maskOverlaySrc);
  const brushImage = useImageElement(brushOverlaySrc);

  useEffect(() => {
    const node = containerRef.current;
    if (!node) return;
    const handleWheel = (event: WheelEvent) => {
      if (!event.ctrlKey && !event.metaKey) return;

      event.preventDefault();
      const state = useEditorStore.getState();
      const viewportRect = node.getBoundingClientRect();
      const normalizedDeltaY =
        event.deltaMode === WheelEvent.DOM_DELTA_LINE
          ? event.deltaY * 16
          : event.deltaMode === WheelEvent.DOM_DELTA_PAGE
            ? event.deltaY * Math.max(1, viewportRect.height)
            : event.deltaY;
      const nextZoom = clampEditorZoom(state.zoom * Math.exp(-normalizedDeltaY * WHEEL_ZOOM_SPEED));
      if (nextZoom === state.zoom) return;

      const imageWidth = baseImage.size.width;
      const imageHeight = baseImage.size.height;
      if (!imageWidth || !imageHeight || !containerSize.width) {
        state.setZoom(nextZoom);
        return;
      }

      const viewportWidth = node.clientWidth || viewportRect.width;
      const fit = Math.min((containerSize.width - 48) / imageWidth, 1);
      const currentScale = Math.max(0.05, fit * state.zoom);
      const nextScale = Math.max(0.05, fit * nextZoom);
      const stageCanvas = node.querySelector("canvas");
      const stageRect = stageCanvas?.getBoundingClientRect();

      const pageBaseLeft = viewportRect.left + viewportWidth / 2 - (imageWidth * currentScale) / 2;
      const nextPageBaseLeft = viewportRect.left + viewportWidth / 2 - (imageWidth * nextScale) / 2;
      const currentStageLeft = pageBaseLeft + state.panOffset.x;
      const currentStageTop = stageRect?.top ?? viewportRect.top + state.panOffset.y;
      const pageBaseTop = currentStageTop - state.panOffset.y;

      const anchorImageX = (event.clientX - currentStageLeft) / currentScale;
      const anchorImageY = (event.clientY - currentStageTop) / currentScale;

      useEditorStore.setState({
        zoom: nextZoom,
        panOffset: {
          x: event.clientX - nextPageBaseLeft - anchorImageX * nextScale,
          y: event.clientY - pageBaseTop - anchorImageY * nextScale,
        },
      });
    };
    node.addEventListener("wheel", handleWheel, { passive: false, capture: true });
    return () => node.removeEventListener("wheel", handleWheel, true);
  }, [baseImage.size.height, baseImage.size.width, containerSize.width]);

  useEffect(() => {
    if (!paintPreviewPendingCommitRef.current || paintStrokeRef.current.length > 0) return;
    clearPaintPreviewOverlay();
  }, [baseImage.image, brushImage.image, maskImage.image]);

  const {
    enqueueBitmapPersist,
    enqueueRecoveryPersist,
    applyBitmapStroke: applyBitmapStrokePersist,
    healPaintedRegion: healPaintedRegionPersist,
  } = useEditorBitmapDrawing({
    pageKey: currentPageKey,
    pageIndex: currentPageIndex,
    width: baseImage.size.width,
    height: baseImage.size.height,
    applyBitmapStroke,
    healPaintedRegion,
  });

  const bitmapTargetForTool = (mode: typeof toolMode) => {
    const state = useEditorStore.getState();
    return bitmapTargetForEditorTool(mode, state.eraserTarget, state.lastPaintedLayer);
  };

  const getBitmapWorkingCanvas = (layerKey: "brush" | "mask") => {
    const existing = bitmapWorkingCanvasRef.current;
    if (existing?.size.width === baseImage.size.width && existing.size.height === baseImage.size.height) {
      const layerCanvas = existing.getLayerCanvas(layerKey);
      if (layerCanvas) return layerCanvas as HTMLCanvasElement;
    }

    const layeredCanvas = new LayeredBitmapCanvas({
      width: baseImage.size.width,
      height: baseImage.size.height,
      createCanvas: (width, height) => {
        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        return canvas;
      },
    });

    if (brushImage.image?.naturalWidth && brushImage.image?.naturalHeight) {
      layeredCanvas.drawImageToLayer("brush", brushImage.image);
    } else {
      layeredCanvas.ensureLayer("brush");
    }
    if (maskImage.image?.naturalWidth && maskImage.image?.naturalHeight) {
      layeredCanvas.drawImageToLayer("mask", maskImage.image);
    } else {
      layeredCanvas.ensureLayer("mask");
    }
    bitmapWorkingCanvasRef.current = layeredCanvas;
    return layeredCanvas.getLayerCanvas(layerKey) as HTMLCanvasElement;
  };

  const getRecoveryWorkingCanvas = () => {
    const existing = recoveryWorkingCanvasRef.current;
    if (existing?.width === baseImage.size.width && existing.height === baseImage.size.height) {
      return existing;
    }

    const canvas = document.createElement("canvas");
    canvas.width = baseImage.size.width;
    canvas.height = baseImage.size.height;
    const ctx = canvas.getContext("2d");
    if (ctx && baseImage.image?.naturalWidth && baseImage.image?.naturalHeight) {
      ctx.drawImage(baseImage.image, 0, 0, canvas.width, canvas.height);
    }
    recoveryWorkingCanvasRef.current = canvas;
    return canvas;
  };

  const rememberInpaintCacheSource = () => {
    if (sessionInpaintCacheCanvasRef.current || !baseImage.image?.naturalWidth || !baseImage.image.naturalHeight) {
      return sessionInpaintCacheCanvasRef.current;
    }
    const canvas = document.createElement("canvas");
    canvas.width = baseImage.size.width;
    canvas.height = baseImage.size.height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(baseImage.image, 0, 0, canvas.width, canvas.height);
    sessionInpaintCacheCanvasRef.current = canvas;
    return canvas;
  };

  // Fallback DOM-level: garante que cursorPoint atualize mesmo quando os
  // eventos Konva não disparam (ex: entrar via canto, HMR). Procura o
  // <canvas> do Konva Stage dentro do container e usa seu rect.
  useEffect(() => {
    if (toolMode !== "brush" && toolMode !== "repairBrush" && toolMode !== "reinpaintBrush" && toolMode !== "eraser") {
      setCursorPoint(null);
      setCursorViewportPoint(null);
      return;
    }
    const node = containerRef.current;
    if (!node) return;

    const onMove = (event: MouseEvent) => {
      const containerRect = node.getBoundingClientRect();
      const viewportX = event.clientX - containerRect.left;
      const viewportY = event.clientY - containerRect.top;
      if (
        viewportX < 0 ||
        viewportY < 0 ||
        viewportX > containerRect.width ||
        viewportY > containerRect.height
      ) {
        if (paintStrokeRef.current.length === 0) {
          setCursorPoint(null);
          setCursorViewportPoint(null);
        }
        return;
      }
      setCursorViewportPoint({ x: viewportX, y: viewportY });

      const stageCanvas = node.querySelector("canvas");
      if (!stageCanvas || !baseImage.size.width || !baseImage.size.height) return;
      const rect = stageCanvas.getBoundingClientRect();
      if (
        event.clientX < rect.left ||
        event.clientX > rect.left + rect.width ||
        event.clientY < rect.top ||
        event.clientY > rect.top + rect.height
      ) {
        if (paintStrokeRef.current.length === 0) setCursorPoint(null);
        return;
      }
      const point = pointFromStageClientRect({
        clientX: event.clientX,
        clientY: event.clientY,
        rect,
        imageWidth: baseImage.size.width,
        imageHeight: baseImage.size.height,
      });
      if (point) setCursorPoint(point);
    };

    window.addEventListener("mousemove", onMove);
    return () => window.removeEventListener("mousemove", onMove);
  }, [toolMode, baseImage.size.width, baseImage.size.height, isPaintStrokeActive]);

  const stageScale = useMemo(() => {
    if (!baseImage.size.width || !baseImage.size.height || !containerSize.width || !containerSize.height) return 1;
    const fit = Math.min((containerSize.width - 48) / baseImage.size.width, 1);
    return Math.max(0.05, fit * zoom);
  }, [baseImage.size.height, baseImage.size.width, containerSize.height, containerSize.width, zoom]);

  const layers = useMemo<TextEntry[]>(
    () => (currentPage?.text_layers ?? []).map((entry) => mergePendingTextEntry(entry, pendingEdits[entry.id])),
    [currentPage?.text_layers, pendingEdits],
  );

  const faithfulPreview = isFaithfulPreviewMode(viewMode, renderPreviewState);
  const translatedEditing = viewMode === "translated" && !faithfulPreview;
  const isLassoTool = toolMode === "mask" || toolMode === "process";
  const selectedNodeName = selectedLayerId
    ? `text-layer-${selectedLayerId.replace(/[^a-zA-Z0-9_-]/g, "_")}`
    : null;

  useEffect(() => {
    if (toolMode !== "select" && selectedLayerId) {
      selectLayer(null);
    }
  }, [selectedLayerId, selectLayer, toolMode]);

  const commitTextLayerTransform = (
    entry: TextEntry,
    before: TextTransformSnapshot,
    after: TextTransformSnapshot,
  ) => {
    commitTextTransform(entry.id, before, after);
  };

  const beginPan = (clientX: number, clientY: number) => {
    setPanSession({
      startX: clientX,
      startY: clientY,
      originX: panOffset.x,
      originY: panOffset.y,
    });
  };

  const pointFromStageEvent = (event: Konva.KonvaEventObject<MouseEvent>) => {
    const stage = event.target.getStage();
    const rect = stage?.container().getBoundingClientRect();
    if (!rect) return null;
    return pointFromStageClientRect({
      clientX: event.evt.clientX,
      clientY: event.evt.clientY,
      rect,
      imageWidth: baseImage.size.width,
      imageHeight: baseImage.size.height,
    });
  };

  const handleStageMouseDown = (event: Konva.KonvaEventObject<MouseEvent>) => {
    if (event.evt.button !== 0) return;
    if (toolMode === "block") {
      const point = pointFromStageEvent(event);
      if (!point) return;
      event.cancelBubble = true;
      selectLayer(null);
      setBlockDraft({ start: point, current: point });
      return;
    }

    if (toolMode === "brush" || toolMode === "repairBrush" || toolMode === "reinpaintBrush" || toolMode === "eraser") {
      const point = pointFromStageEvent(event);
      if (!point) return;
      event.cancelBubble = true;
      selectLayer(null);
      const startPoint: [number, number] = [point.x, point.y];
      cancelPendingPaintPreviewClear();
      paintStrokeRef.current = [startPoint];
      setIsPaintStrokeActive(true);
      paintPreviewOverlayRef.current?.begin(startPoint);
    }

    if (isLassoTool) {
      const point = pointFromStageEvent(event);
      if (!point) return;
      event.cancelBubble = true;
      selectLayer(null);
      if (activeLassoSelection?.pageKey === currentPageKey) return;

      if (maskShape === "freehand") {
        // Inicia traço freehand
        setMaskInProgress({ points: [[point.x, point.y]] });
      } else {
        // Poligonal: adicionar ponto ou fechar se perto do primeiro
        const existing = maskInProgress?.points ?? [];
        if (existing.length >= 3) {
          const [fx, fy] = existing[0];
          const dist = Math.hypot(point.x - fx, point.y - fy);
          if (dist < 12) {
            void commitLasso(existing);
            return;
          }
        }
        setMaskInProgress({ points: [...existing, [point.x, point.y]] });
      }
    }
  };

  const handleStageMouseMove = (event: Konva.KonvaEventObject<MouseEvent>) => {
    const containerRect = containerRef.current?.getBoundingClientRect();
    if (containerRect && (toolMode === "brush" || toolMode === "repairBrush" || toolMode === "reinpaintBrush" || toolMode === "eraser")) {
      setCursorViewportPoint({
        x: event.evt.clientX - containerRect.left,
        y: event.evt.clientY - containerRect.top,
      });
    }
    const point = pointFromStageEvent(event);
    if (!point) return;
    if (blockDraft) {
      setBlockDraft((draft) => (draft ? { ...draft, current: point } : null));
      return;
    }
    if (paintStrokeRef.current.length > 0) {
      const points = paintStrokeRef.current;
      const last = points[points.length - 1];
      if (last && shouldAppendStrokePoint(last, point)) {
        const nextPoint: [number, number] = [point.x, point.y];
        points.push(nextPoint);
        paintPreviewOverlayRef.current?.append(nextPoint);
      }
    }

    // Freehand lasso — decimação mínima de 2px para evitar pontos redundantes
    if (isLassoTool && maskShape === "freehand" && maskInProgress !== null && (event.evt.buttons & 1) === 1) {
      const last = maskInProgress.points[maskInProgress.points.length - 1];
      if (!last || Math.hypot(point.x - last[0], point.y - last[1]) >= 2) {
        setMaskInProgress({ points: [...maskInProgress.points, [point.x, point.y]] });
      }
    }
    // Atualizar posição do cursor circular em modos de pintura
    if (toolMode === "brush" || toolMode === "repairBrush" || toolMode === "reinpaintBrush" || toolMode === "eraser") {
      setCursorPoint(point);
    }
  };

  const handleStageMouseEnter = (event: Konva.KonvaEventObject<MouseEvent>) => {
    const containerRect = containerRef.current?.getBoundingClientRect();
    if (containerRect && (toolMode === "brush" || toolMode === "repairBrush" || toolMode === "reinpaintBrush" || toolMode === "eraser")) {
      setCursorViewportPoint({
        x: event.evt.clientX - containerRect.left,
        y: event.evt.clientY - containerRect.top,
      });
    }
    const point = pointFromStageEvent(event);
    if (point && (toolMode === "brush" || toolMode === "repairBrush" || toolMode === "reinpaintBrush" || toolMode === "eraser")) {
      setCursorPoint(point);
    }
  };

  const handleStageMouseLeave = () => {
    // Manter cursor visível enquanto está pintando (stroke ativo).
    // O ponteiro visual continua via cursorViewportPoint mesmo fora da pagina.
    if (paintStrokeRef.current.length === 0) setCursorPoint(null);
  };

  const finishBlockDraft = async () => {
    const draft = blockDraft;
    setBlockDraft(null);
    if (!draft) return;
    const x1 = Math.min(draft.start.x, draft.current.x);
    const y1 = Math.min(draft.start.y, draft.current.y);
    const x2 = Math.max(draft.start.x, draft.current.x);
    const y2 = Math.max(draft.start.y, draft.current.y);
    if (x2 - x1 < 12 || y2 - y1 < 12) return;
    await createTextLayer([x1, y1, x2, y2]);
  };

  const finishPaintStroke = async () => {
    const stroke = paintStrokeRef.current;
    const strokeToolMode = toolMode;
    const strokeBrushSize = brushSize;
    const strokeBrushColor = brushColor;
    const strokeBrushOpacity = brushOpacity;
    const strokeBrushHardness = brushHardness;
    paintStrokeRef.current = [];
    setIsPaintStrokeActive(false);
    schedulePaintPreviewClear();
    if (!baseImage.size.width || !baseImage.size.height || stroke.length === 0) return;
    const basicDirtyBBox = strokeDirtyBbox({
      stroke,
      brushSize: strokeBrushSize,
      width: baseImage.size.width,
      height: baseImage.size.height,
    });
    if (!basicDirtyBBox) return;
    const strokeDirtyBBox =
      strokeToolMode === "reinpaintBrush"
        ? (paddedStrokeBBox({
            stroke,
            brushSize: strokeBrushSize,
            width: baseImage.size.width,
            height: baseImage.size.height,
          }) ?? basicDirtyBBox)
        : basicDirtyBBox;
    const strokeSelection = activeLassoSelection?.pageKey === currentPageKey ? activeLassoSelection : null;
    const dirty_bbox = strokeSelection
      ? intersectBbox(strokeDirtyBBox, strokeSelection.bbox, baseImage.size.width, baseImage.size.height)
      : strokeDirtyBBox;
    if (!dirty_bbox) return;
    const clipPolygon = strokeSelection?.points;
    const clipMaskPng = strokeSelection
      ? rasterizeLassoToPng(strokeSelection.points, strokeSelection.width, strokeSelection.height)
      : undefined;
    if (strokeSelection && !clipMaskPng) return;
    const payload = {
      width: baseImage.size.width,
      height: baseImage.size.height,
      strokes: [stroke],
      dirty_bbox,
      clipMaskPng,
    };

    if (strokeToolMode === "repairBrush") {
      rememberInpaintCacheSource();
      const previewId = crypto.randomUUID();
      activeRecoveryPreviewIdsRef.current.add(previewId);
      let recoveryPngData: string | undefined;
      let recoveryBeforeDataUrl: string | undefined;
      if (baseImage.image && originalImage.image) {
        const recoveryCanvas = getRecoveryWorkingCanvas();
        recoveryBeforeDataUrl = recoveryCanvas.toDataURL("image/png");
        recoveryPngData = applyRecoveryStrokeToCanvas(
          recoveryCanvas,
          originalImage.image,
          stroke,
          strokeBrushSize,
          clipPolygon,
        ) ?? undefined;
      }

      if (recoveryBeforeDataUrl && recoveryPngData) {
        const commandId = `bitmap-${crypto.randomUUID()}`;
        bitmapCache.set(commandId, {
          pageKey: currentPageKey,
          commandId,
          before: encodeDataUrl(recoveryBeforeDataUrl),
          after: encodeDataUrl(recoveryPngData),
          byteLength: recoveryBeforeDataUrl.length + recoveryPngData.length,
        });
        executeEditorCommand({
          commandId,
          pageKey: currentPageKey,
          createdAt: Date.now(),
          type: "bitmap-stroke",
          layerKey: "inpaint",
          bbox: dirty_bbox,
        });
      } else if (originalImage.image) {
        void createRecoveryStrokePreviewPatch(originalImage.image, stroke, strokeBrushSize, dirty_bbox, clipPolygon)
          .then((patch) => {
            if (patch && activeRecoveryPreviewIdsRef.current.has(previewId)) {
              setRecoveryPreviewPatches((patches) => [...patches, { ...patch, id: previewId }]);
            }
          })
          .catch((error) => console.error("Erro ao criar preview local da recuperação:", error));
      }

      const persistRecoveryStroke = async (context: { pageKey: string; pageIndex: number }) => {
        try {
          await applyBitmapStrokePersist({
            pageKey: context.pageKey,
            pageIndex: context.pageIndex,
            ...payload,
            layerKey: "recovery",
            erase: false,
            brushSize: strokeBrushSize,
            color: strokeBrushColor,
            opacity: strokeBrushOpacity,
            hardness: strokeBrushHardness,
            pngData: recoveryPngData,
            optimisticPath: recoveryPngData,
          });
          activeRecoveryPreviewIdsRef.current.delete(previewId);
          window.setTimeout(() => {
            setRecoveryPreviewPatches((patches) => patches.filter((patch) => patch.id !== previewId));
          }, 500);
          const state = useEditorStore.getState();
          if (state.currentPageIndex === context.pageIndex && state.currentPageKey() === context.pageKey) {
            bumpBitmapLayerVersion("inpaint");
          }
        } catch (error) {
          activeRecoveryPreviewIdsRef.current.delete(previewId);
          setRecoveryPreviewPatches((patches) => patches.filter((patch) => patch.id !== previewId));
          console.error("Erro ao persistir pincel de recuperação:", error);
        }
      };

      void enqueueRecoveryPersist(persistRecoveryStroke);
      return;
    }

    if (strokeToolMode === "reinpaintBrush") {
      const maskPngData = createHealingBrushMaskPngDataUrl({
        width: baseImage.size.width,
        height: baseImage.size.height,
        stroke,
        brushSize: strokeBrushSize,
        dirtyBBox: dirty_bbox,
        clipPolygon,
      });
      if (!maskPngData) return;

      const persistHealingStroke = async (context: { pageKey: string; pageIndex: number }) => {
        try {
          await healPaintedRegionPersist({ pageKey: context.pageKey, pageIndex: context.pageIndex, bbox: dirty_bbox, maskPngData });
          setReinpaintPreviewPatches([]);
        } catch (error) {
          setReinpaintPreviewPatches([]);
          console.error("Erro ao aplicar pincel corretor:", error);
        }
      };

      void enqueueRecoveryPersist(persistHealingStroke);
      return;
    }

    const layerKey = bitmapTargetForTool(strokeToolMode);
    if (!layerKey) return;
    if (layerKey === "recovery" || layerKey === "reinpaint") return;
    const erase = strokeToolMode === "eraser";
    const workingCanvas = getBitmapWorkingCanvas(layerKey);
    const preview = createBitmapStrokePreviewOnCanvas(workingCanvas, {
      layerKey,
      stroke,
      brushSize: strokeBrushSize,
      color: strokeBrushColor,
      opacity: strokeBrushOpacity,
      hardness: strokeBrushHardness,
      erase,
      clipPolygon,
    });

    if (preview) {
      const commandId = `bitmap-${crypto.randomUUID()}`;
      bitmapCache.set(commandId, {
        pageKey: currentPageKey,
        commandId,
        before: encodeDataUrl(preview.beforeDataUrl),
        after: encodeDataUrl(preview.afterDataUrl),
        byteLength: preview.beforeDataUrl.length + preview.afterDataUrl.length,
      });
      executeEditorCommand({
        commandId,
        pageKey: currentPageKey,
        createdAt: Date.now(),
        type: "bitmap-stroke",
        layerKey: preview.layerKey,
        bbox: dirty_bbox,
      });
    }

    const persistBitmapStroke = async (context: { pageKey: string; pageIndex: number }) => {
      await applyBitmapStrokePersist({
        pageKey: context.pageKey,
        pageIndex: context.pageIndex,
        ...payload,
        layerKey,
        erase,
        brushSize: strokeBrushSize,
        color: strokeBrushColor,
        opacity: strokeBrushOpacity,
        hardness: strokeBrushHardness,
        optimisticPath: preview?.afterDataUrl,
      });
    };

    void enqueueBitmapPersist(layerKey, persistBitmapStroke).catch((error) => {
      console.error("Erro ao persistir pincel:", error);
    });
  };

  // ── Fase 8: Lasso commit ─────────────────────────────────────────────────
  const commitLasso = async (points: Array<[number, number]>) => {
    if (points.length < 3 || !baseImage.size.width || !baseImage.size.height) {
      setMaskInProgress(null);
      return;
    }
    const w = baseImage.size.width;
    const h = baseImage.size.height;
    const state = useEditorStore.getState();
    const selection = createLassoSelection({
      pageKey: state.currentPageKey(),
      pageIndex: state.currentPageIndex,
      points,
      width: w,
      height: h,
    });
    setMaskInProgress(null);
    setActiveLassoSelection(selection);
    if (state.toolMode === "process") {
      void runProcessRegionFromSelection(selection);
    }
  };

  const finishFreehandLasso = (points: Array<[number, number]>) => {
    if (points.length >= 3) {
      void commitLasso(points);
      return;
    }
    setMaskInProgress(null);
  };

  // Sincronizar ref para evitar stale closure no global mouseup handler
  finishPaintStrokeRef.current = finishPaintStroke;
  commitLassoRef.current = commitLasso;

  // FIX CRÍTICO: commit do stroke quando mouseup ocorre FORA do canvas Konva
  // Sem isso, soltar o mouse fora do <Stage> descarta o stroke silenciosamente.
  useEffect(() => {
    if (!isPaintStrokeActive) return;

    const handleGlobalMouseUp = (event: MouseEvent) => {
      // Não duplicar com handler do Konva: só age fora do container do stage
      if (containerRef.current?.contains(event.target as Node)) return;
      void finishPaintStrokeRef.current?.();
    };

    window.addEventListener("mouseup", handleGlobalMouseUp);
    return () => window.removeEventListener("mouseup", handleGlobalMouseUp);
  }, [isPaintStrokeActive]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!isLassoTool || maskShape !== "freehand" || maskInProgress === null) return;

    const handleGlobalMouseUp = (event: MouseEvent) => {
      if (containerRef.current?.contains(event.target as Node)) return;
      const points = useEditorStore.getState().maskInProgress?.points ?? [];
      finishFreehandLasso(points);
    };

    window.addEventListener("mouseup", handleGlobalMouseUp);
    return () => window.removeEventListener("mouseup", handleGlobalMouseUp);
  }, [isLassoTool, maskInProgress, maskShape]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleStageMouseUp = () => {
    if (blockDraft) {
      void finishBlockDraft();
      return;
    }
    if (paintStrokeRef.current.length > 0) {
      void finishPaintStroke();
    }
    // Freehand lasso commit ao soltar o mouse
    if (isLassoTool && maskShape === "freehand" && maskInProgress) {
      finishFreehandLasso(maskInProgress.points);
    }
  };

  const handleViewportMouseDown = (event: React.MouseEvent<HTMLDivElement>) => {
    if (event.altKey && event.button === 2) {
      event.preventDefault();
      event.stopPropagation();
      setBrushSizeDragSession({
        startX: event.clientX,
        startSize: useEditorStore.getState().brushSize,
      });
      return;
    }

    if (event.button === 1 || (event.button === 0 && isSpacePressed)) {
      event.preventDefault();
      event.stopPropagation();
      beginPan(event.clientX, event.clientY);
      return;
    }

  };

  const handleViewportContextMenu = (event: React.MouseEvent<HTMLDivElement>) => {
    if (event.altKey || brushSizeDragSession) {
      event.preventDefault();
      event.stopPropagation();
    }
  };

  const viewportCursor = panSession
    ? "grabbing"
    : brushSizeDragSession
      ? "ew-resize"
    : isSpacePressed
      ? "grab"
      : toolMode === "block" || isLassoTool
        ? "crosshair"
        : toolMode === "brush" || toolMode === "repairBrush" || toolMode === "reinpaintBrush" || toolMode === "eraser"
          ? "none"
          : "default";

  return {
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
    renderPreviewState,
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
    setPaintPreviewOverlay: (handle: PaintPreviewOverlayHandle | null) => {
      paintPreviewOverlayRef.current = handle;
    },
    cursorPoint,
    cursorViewportPoint,
    // Fase 8 — Lasso
    maskInProgress,
    activeLassoSelection,
    maskShape,
    maskOp,
  };
}

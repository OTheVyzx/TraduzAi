import { useEffect, useMemo, useRef, useState } from "react";
import type Konva from "konva";
import { readFile } from "@tauri-apps/plugin-fs";
import { getRenderPreviewStateForPage, useEditorStore } from "../../../lib/stores/editorStore";
import type { TextEntry } from "../../../lib/stores/appStore";
import { displayImagePathForMode, isBitmapInspectionLayer, isFaithfulPreviewMode } from "./renderModeUtils";
import { mergePendingTextEntry } from "./textLayerStyleUtils";

function isE2E() {
  const meta = import.meta as ImportMeta & { env?: Record<string, string | undefined> };
  return (meta.env?.VITE_E2E ?? "") === "1";
}

function useObjectUrl(path: string | null | undefined, type = "image/png", version = 0) {
  const [src, setSrc] = useState<string | null>(null);

  useEffect(() => {
    if (!path) {
      setSrc(null);
      return;
    }

    if (isE2E() || path.startsWith("data:") || path.startsWith("/") || /^https?:\/\//.test(path)) {
      setSrc(path);
      return;
    }

    let cancelled = false;
    let objectUrl: string | null = null;

    readFile(path)
      .then((bytes) => {
        if (cancelled) return;
        objectUrl = URL.createObjectURL(new Blob([bytes], { type }));
        setSrc(objectUrl);
      })
      .catch((error) => {
        console.error("Falha ao carregar imagem do editor:", error);
        if (!cancelled) setSrc(null);
      });

    return () => {
      cancelled = true;
      if (objectUrl) {
        const url = objectUrl;
        window.setTimeout(() => URL.revokeObjectURL(url), 100);
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

export function useEditorStageController() {
  const containerRef = useRef<HTMLDivElement>(null);
  const currentPage = useEditorStore((state) => state.currentPage);
  const viewMode = useEditorStore((state) => state.viewMode);
  const toolMode = useEditorStore((state) => state.toolMode);
  const showOverlays = useEditorStore((state) => state.showOverlays);
  const brushSize = useEditorStore((state) => state.brushSize);
  const zoom = useEditorStore((state) => state.zoom);
  const panOffset = useEditorStore((state) => state.panOffset);
  const lastRetypesetTime = useEditorStore((state) => state.lastRetypesetTime);
  const selectedLayerId = useEditorStore((state) => state.selectedLayerId);
  const selectedImageLayerKey = useEditorStore((state) => state.selectedImageLayerKey);
  const hoveredLayerId = useEditorStore((state) => state.hoveredLayerId);
  const pendingEdits = useEditorStore((state) => state.pendingEdits);
  const currentPageKey = useEditorStore((state) => state.currentPageKey());
  const renderPreviewCacheByPageKey = useEditorStore((state) => state.renderPreviewCacheByPageKey);

  const selectLayer = useEditorStore((state) => state.selectLayer);
  const selectImageLayer = useEditorStore((state) => state.selectImageLayer);
  const hoverLayer = useEditorStore((state) => state.hoverLayer);
  const setZoom = useEditorStore((state) => state.setZoom);
  const setPan = useEditorStore((state) => state.setPan);
  const setWorkingBbox = useEditorStore((state) => state.setWorkingBbox);
  const recordEditorCommand = useEditorStore((state) => state.recordEditorCommand);
  const createTextLayer = useEditorStore((state) => state.createTextLayer);
  const applyBitmapStroke = useEditorStore((state) => state.applyBitmapStroke);

  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
  const [blockDraft, setBlockDraft] = useState<{
    start: { x: number; y: number };
    current: { x: number; y: number };
  } | null>(null);
  const [paintStroke, setPaintStroke] = useState<[number, number][]>([]);
  const [isSpacePressed, setIsSpacePressed] = useState(false);
  const [panSession, setPanSession] = useState<{
    startX: number;
    startY: number;
    originX: number;
    originY: number;
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
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== " ") return;
      const active = document.activeElement;
      if (active && ["INPUT", "TEXTAREA", "SELECT"].includes(active.tagName)) return;
      event.preventDefault();
      setIsSpacePressed(true);
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
    const node = containerRef.current;
    if (!node) return;
    const handleWheel = (event: WheelEvent) => {
      event.preventDefault();
      if (event.ctrlKey || event.metaKey) {
        setZoom(zoom + (event.deltaY > 0 ? -0.12 : 0.12));
        return;
      }
      setPan({
        x: panOffset.x - event.deltaX * 0.65,
        y: panOffset.y - event.deltaY * 0.65,
      });
    };
    node.addEventListener("wheel", handleWheel, { passive: false });
    return () => node.removeEventListener("wheel", handleWheel);
  }, [panOffset.x, panOffset.y, setPan, setZoom, zoom]);

  const renderPreviewState = useMemo(
    () => getRenderPreviewStateForPage(currentPageKey, currentPage, renderPreviewCacheByPageKey),
    [currentPage, currentPageKey, renderPreviewCacheByPageKey],
  );
  const displayImagePath = useMemo(
    () => displayImagePathForMode(currentPage, viewMode, renderPreviewState, selectedImageLayerKey),
    [currentPage, renderPreviewState, selectedImageLayerKey, viewMode],
  );

  const baseImageSrc = useObjectUrl(displayImagePath, "image/png", lastRetypesetTime);
  const maskOverlaySrc = useObjectUrl(
    currentPage?.image_layers?.mask?.visible ? currentPage.image_layers.mask.path : null,
    "image/png",
    lastRetypesetTime,
  );
  const brushOverlaySrc = useObjectUrl(
    currentPage?.image_layers?.brush?.visible ? currentPage.image_layers.brush.path : null,
    "image/png",
    lastRetypesetTime,
  );

  const baseImage = useImageElement(baseImageSrc);
  const maskImage = useImageElement(maskOverlaySrc);
  const brushImage = useImageElement(brushOverlaySrc);

  const stageScale = useMemo(() => {
    if (!baseImage.size.width || !baseImage.size.height || !containerSize.width || !containerSize.height) return 1;
    const fit = Math.min(
      (containerSize.width - 48) / baseImage.size.width,
      (containerSize.height - 48) / baseImage.size.height,
      1,
    );
    return Math.max(0.05, fit * zoom);
  }, [baseImage.size.height, baseImage.size.width, containerSize.height, containerSize.width, zoom]);

  const layers = useMemo<TextEntry[]>(
    () => (currentPage?.text_layers ?? []).map((entry) => mergePendingTextEntry(entry, pendingEdits[entry.id])),
    [currentPage?.text_layers, pendingEdits],
  );

  const faithfulPreview = isFaithfulPreviewMode(viewMode, renderPreviewState);
  const bitmapInspection = isBitmapInspectionLayer(selectedImageLayerKey);
  const translatedEditing = viewMode === "translated" && !faithfulPreview && !bitmapInspection;
  const selectedNodeName = selectedLayerId
    ? `text-layer-${selectedLayerId.replace(/[^a-zA-Z0-9_-]/g, "_")}`
    : null;

  const commitBbox = (entry: TextEntry, before: TextEntry["bbox"], after: TextEntry["bbox"]) => {
    setWorkingBbox(currentPageKey, entry.id, after);
    recordEditorCommand({
      commandId: `edit-bbox-${crypto.randomUUID()}`,
      pageKey: currentPageKey,
      createdAt: Date.now(),
      type: "edit-bbox",
      layerId: entry.id,
      before,
      after,
    });
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
    if (!rect || !baseImage.size.width || !baseImage.size.height) return null;
    const x = ((event.evt.clientX - rect.left) / rect.width) * baseImage.size.width;
    const y = ((event.evt.clientY - rect.top) / rect.height) * baseImage.size.height;
    return {
      x: Math.max(0, Math.min(baseImage.size.width, Math.round(x))),
      y: Math.max(0, Math.min(baseImage.size.height, Math.round(y))),
    };
  };

  const handleStageMouseDown = (event: Konva.KonvaEventObject<MouseEvent>) => {
    if (event.evt.button !== 0) return;
    if (toolMode === "block") {
      const point = pointFromStageEvent(event);
      if (!point) return;
      event.cancelBubble = true;
      selectLayer(null);
      selectImageLayer(null);
      setBlockDraft({ start: point, current: point });
      return;
    }

    if (toolMode === "brush" || toolMode === "repairBrush" || toolMode === "eraser") {
      const point = pointFromStageEvent(event);
      if (!point) return;
      event.cancelBubble = true;
      selectLayer(null);
      selectImageLayer(toolMode === "brush" ? "brush" : "mask");
      setPaintStroke([[point.x, point.y]]);
    }
  };

  const handleStageMouseMove = (event: Konva.KonvaEventObject<MouseEvent>) => {
    const point = pointFromStageEvent(event);
    if (!point) return;
    if (blockDraft) {
      setBlockDraft((draft) => (draft ? { ...draft, current: point } : null));
      return;
    }
    if (paintStroke.length > 0) {
      setPaintStroke((points) => {
        const last = points[points.length - 1];
        if (last && last[0] === point.x && last[1] === point.y) return points;
        return [...points, [point.x, point.y]];
      });
    }
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
    const stroke = paintStroke;
    setPaintStroke([]);
    if (!baseImage.size.width || !baseImage.size.height || stroke.length === 0) return;
    await applyBitmapStroke({
      width: baseImage.size.width,
      height: baseImage.size.height,
      strokes: [stroke],
    });
  };

  const handleStageMouseUp = () => {
    if (blockDraft) {
      void finishBlockDraft();
      return;
    }
    if (paintStroke.length > 0) {
      void finishPaintStroke();
    }
  };

  const handleViewportMouseDown = (event: React.MouseEvent<HTMLDivElement>) => {
    if (event.button === 1 || (event.button === 0 && isSpacePressed)) {
      event.preventDefault();
      event.stopPropagation();
      beginPan(event.clientX, event.clientY);
      return;
    }

    if (event.target === event.currentTarget) {
      selectLayer(null);
      selectImageLayer(null);
    }
  };

  const viewportCursor = panSession
    ? "grabbing"
    : isSpacePressed
      ? "grab"
      : toolMode === "block"
        ? "crosshair"
        : toolMode === "brush" || toolMode === "repairBrush" || toolMode === "eraser"
          ? "none"
          : "default";

  return {
    containerRef,
    currentPage,
    viewMode,
    toolMode,
    showOverlays,
    brushSize,
    panOffset,
    panSession,
    viewportCursor,
    renderPreviewState,
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
  };
}

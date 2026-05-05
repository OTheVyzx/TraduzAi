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
  const bitmapLayerVersions = useEditorStore((state) => state.bitmapLayerVersions);
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
  // Fase 8 — Lasso
  const maskShape = useEditorStore((state) => state.maskShape);
  const maskOp = useEditorStore((state) => state.maskOp);
  const maskInProgress = useEditorStore((state) => state.maskInProgress);
  const setMaskInProgress = useEditorStore((state) => state.setMaskInProgress);
  const bumpBitmapLayerVersion = useEditorStore((state) => state.bumpBitmapLayerVersion);

  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
  const [blockDraft, setBlockDraft] = useState<{
    start: { x: number; y: number };
    current: { x: number; y: number };
  } | null>(null);
  const [paintStroke, setPaintStroke] = useState<[number, number][]>([]);
  const [cursorPoint, setCursorPoint] = useState<{ x: number; y: number } | null>(null);
  const [isSpacePressed, setIsSpacePressed] = useState(false);
  // Ref para garantir acesso ao finishPaintStroke mais recente sem criar stale closure
  const finishPaintStrokeRef = useRef<(() => Promise<void>) | null>(null);
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
      const active = document.activeElement;
      if (active && ["INPUT", "TEXTAREA", "SELECT"].includes(active.tagName)) return;

      if (event.key === " ") {
        event.preventDefault();
        setIsSpacePressed(true);
        return;
      }

      // Lasso keyboard shortcuts — lê estado fresco via getState para evitar stale closure
      const s = useEditorStore.getState();
      if (s.toolMode === "mask") {
        if (event.key === "Escape") {
          event.preventDefault();
          s.setMaskInProgress(null);
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

    if (toolMode === "mask") {
      const point = pointFromStageEvent(event);
      if (!point) return;
      event.cancelBubble = true;
      selectLayer(null);
      selectImageLayer("mask");

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

    // Freehand lasso — decimação mínima de 2px para evitar pontos redundantes
    if (toolMode === "mask" && maskShape === "freehand" && maskInProgress !== null) {
      const last = maskInProgress.points[maskInProgress.points.length - 1];
      if (!last || Math.hypot(point.x - last[0], point.y - last[1]) >= 2) {
        setMaskInProgress({ points: [...maskInProgress.points, [point.x, point.y]] });
      }
    }
    // Atualizar posição do cursor circular em modos de pintura
    if (toolMode === "brush" || toolMode === "repairBrush" || toolMode === "eraser") {
      setCursorPoint(point);
    }
  };

  const handleStageMouseEnter = (event: Konva.KonvaEventObject<MouseEvent>) => {
    const point = pointFromStageEvent(event);
    if (point && (toolMode === "brush" || toolMode === "repairBrush" || toolMode === "eraser")) {
      setCursorPoint(point);
    }
  };

  const handleStageMouseLeave = () => {
    // Manter cursor visível enquanto está pintando (stroke ativo)
    if (paintStroke.length === 0) setCursorPoint(null);
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

  // ── Fase 8: Lasso commit ─────────────────────────────────────────────────
  const commitLasso = async (points: Array<[number, number]>) => {
    const { project } = (await import("../../../lib/stores/appStore")).useAppStore.getState();
    const path = project ? (project.output_path ?? project.source_path) : null;
    if (points.length < 3 || !baseImage.size.width || !baseImage.size.height || !path) {
      setMaskInProgress(null);
      return;
    }
    const w = baseImage.size.width;
    const h = baseImage.size.height;
    const currentPageIndex = useEditorStore.getState().currentPageIndex;

    // Rasterizar polígono em offscreen canvas (espaço da imagem)
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) { setMaskInProgress(null); return; }

    // Se "subtract", apagar (preto no canal alpha); caso contrário, preencher (branco)
    const isSub = maskOp === "subtract";
    if (isSub) {
      ctx.globalCompositeOperation = "source-over";
    }
    ctx.fillStyle = isSub ? "#000000" : "#ffffff";
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i][0], points[i][1]);
    }
    ctx.closePath();
    ctx.fill();

    const pngData = canvas.toDataURL("image/png");
    setMaskInProgress(null);

    try {
      const { writeMaskFromPng } = await import("../../../lib/tauri");
      await writeMaskFromPng({
        project_path: path,
        page_index: currentPageIndex,
        png_data: pngData,
        layer_key: "mask",
        op: maskOp,
      });
      selectImageLayer("mask");
      bumpBitmapLayerVersion("mask");
    } catch (e) {
      console.error("[Lasso] Falha ao escrever máscara:", e);
    }
  };

  // Sincronizar ref para evitar stale closure no global mouseup handler
  finishPaintStrokeRef.current = finishPaintStroke;
  commitLassoRef.current = commitLasso;

  // FIX CRÍTICO: commit do stroke quando mouseup ocorre FORA do canvas Konva
  // Sem isso, soltar o mouse fora do <Stage> descarta o stroke silenciosamente.
  useEffect(() => {
    if (paintStroke.length === 0) return;

    const handleGlobalMouseUp = (event: MouseEvent) => {
      // Não duplicar com handler do Konva: só age fora do container do stage
      if (containerRef.current?.contains(event.target as Node)) return;
      void finishPaintStrokeRef.current?.();
    };

    window.addEventListener("mouseup", handleGlobalMouseUp);
    return () => window.removeEventListener("mouseup", handleGlobalMouseUp);
  }, [paintStroke.length]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleStageMouseUp = () => {
    if (blockDraft) {
      void finishBlockDraft();
      return;
    }
    if (paintStroke.length > 0) {
      void finishPaintStroke();
    }
    // Freehand lasso commit ao soltar o mouse
    if (toolMode === "mask" && maskShape === "freehand" && maskInProgress && maskInProgress.points.length >= 3) {
      void commitLasso(maskInProgress.points);
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
      : toolMode === "block" || toolMode === "mask"
        ? "crosshair"
        : toolMode === "brush" || toolMode === "repairBrush" || toolMode === "eraser"
          ? "none"
          : "default";

  return {
    containerRef,
    containerSize,
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
    handleStageMouseEnter,
    handleStageMouseLeave,
    cursorPoint,
    // Fase 8 — Lasso
    maskInProgress,
    maskShape,
    maskOp,
  };
}

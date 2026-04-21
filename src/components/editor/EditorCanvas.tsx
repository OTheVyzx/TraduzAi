/* eslint-disable */
import { useEffect, useMemo, useRef, useState } from "react";
import { readFile } from "@tauri-apps/plugin-fs";
import { useEditorStore } from "../../lib/stores/editorStore";
import { TextOverlay } from "./TextOverlay";

function useDynamicStyle<T extends HTMLElement>(styleObj: Record<string, string | number>, deps: any[]) {
  const ref = useRef<T>(null);
  useEffect(() => {
    if (ref.current) {
      for (const [key, value] of Object.entries(styleObj)) {
        if (value === undefined || value === null) {
          ref.current.style.removeProperty(key);
        } else {
          ref.current.style.setProperty(key, String(value));
        }
      }
    }
  }, deps);
  return ref;
}

function BrushCursor({ mousePos, brushSize, zoom }: { mousePos: { x: number, y: number }, brushSize: number, zoom: number }) {
  const ref = useDynamicStyle<HTMLDivElement>({
    "--brush-w": `${brushSize * zoom}px`,
    "--brush-h": `${brushSize * zoom}px`,
    "--brush-x": `${mousePos.x - (brushSize * zoom) / 2}px`,
    "--brush-y": `${mousePos.y - (brushSize * zoom) / 2}px`,
  }, [mousePos, brushSize, zoom]);
  return (
    <div
      ref={ref}
      className="pointer-events-none fixed z-50 rounded-full border-2 border-accent-purple/70 bg-accent-purple/10 dynamic-brush"
    />
  );
}

function TransformContainer({ panOffset, zoom, panSession, children }: { panOffset: { x: number, y: number }, zoom: number, panSession: boolean, children?: React.ReactNode }) {
  const ref = useDynamicStyle<HTMLDivElement>({
    "--canvas-transform": `translate(${panOffset.x}px, ${panOffset.y}px) scale(${zoom})`,
    "--canvas-origin": "center center",
    "--canvas-transition": panSession ? "none" : "transform 0.12s ease-out",
  }, [panOffset, zoom, panSession]);
  return (
    <div ref={ref} className="dynamic-canvas-transform">
      {children}
    </div>
  );
}

function SelectionHighlight({ bbox, scaleX, scaleY }: { bbox: [number, number, number, number], scaleX: number, scaleY: number }) {
  const ref = useDynamicStyle<HTMLDivElement>({
    "--left": `${bbox[0] * scaleX}px`,
    "--top": `${bbox[1] * scaleY}px`,
    "--width": `${(bbox[2] - bbox[0]) * scaleX}px`,
    "--height": `${(bbox[3] - bbox[1]) * scaleY}px`,
  }, [bbox, scaleX, scaleY]);
  return <div ref={ref} className="pointer-events-none absolute rounded-[18px] border border-dashed border-accent-cyan/55 bg-accent-cyan/8 dynamic-pos" />;
}

function DraftHighlight({ startX, startY, currentX, currentY, offsetRect }: { startX: number, startY: number, currentX: number, currentY: number, offsetRect: DOMRect }) {
  const ref = useDynamicStyle<HTMLDivElement>({
    "--left": `${Math.min(startX, currentX) - offsetRect.left}px`,
    "--top": `${Math.min(startY, currentY) - offsetRect.top}px`,
    "--width": `${Math.abs(currentX - startX)}px`,
    "--height": `${Math.abs(currentY - startY)}px`,
  }, [startX, startY, currentX, currentY, offsetRect]);
  return <div ref={ref} className="pointer-events-none absolute rounded-2xl border border-dashed border-accent-purple bg-accent-purple/10 dynamic-pos" />;
}

function useObjectUrl(path: string | null | undefined, type = "image/png", version = 0) {
  const [src, setSrc] = useState<string | null>(null);

  useEffect(() => {
    if (!path) {
      setSrc(null);
      return;
    }

    let cancelled = false;
    let newObjectUrl: string | null = null;

    readFile(path)
      .then((bytes) => {
        if (cancelled) return;
        newObjectUrl = URL.createObjectURL(new Blob([bytes], { type }));
        
        // We set the new source, but notice we don't clear the old one first.
        // This keeps the image "stable" until the new blob is ready.
        setSrc(prev => {
          // If we have a previous URL, we should probably revoke it,
          // but we can't do it here easily without a separate effect 
          // or a more complex state.
          // For now, the cleanup function will handle the current 'newObjectUrl'.
          return newObjectUrl;
        });
      })
      .catch((err) => {
        console.error("Failed to read file for object URL:", err);
        if (!cancelled) setSrc(null);
      });

    return () => {
      cancelled = true;
      if (newObjectUrl) {
        // Small delay to ensure the browser has actually swapped the image
        // before we zap the memory for the old one.
        const urlToRevoke = newObjectUrl;
        setTimeout(() => {
          URL.revokeObjectURL(urlToRevoke);
        }, 100);
      }
    };
  }, [path, type, version]);

  return src;
}

function LayerSprite({
  path,
  bbox,
  scaleX,
  scaleY,
  selected,
  version,
}: {
  path: string;
  bbox: [number, number, number, number];
  scaleX: number;
  scaleY: number;
  selected: boolean;
  version: number;
}) {
  const src = useObjectUrl(path, "image/png", version);
  const imgRef = useRef<HTMLImageElement>(null);
  
  useEffect(() => {
    if (imgRef.current) {
      const [x1, y1, x2, y2] = bbox;
      imgRef.current.style.setProperty("--left", `${x1 * scaleX}px`);
      imgRef.current.style.setProperty("--top", `${y1 * scaleY}px`);
      imgRef.current.style.setProperty("--width", `${Math.max(1, (x2 - x1) * scaleX)}px`);
      imgRef.current.style.setProperty("--height", `${Math.max(1, (y2 - y1) * scaleY)}px`);
    }
  }, [bbox, scaleX, scaleY]);

  if (!src) return null;

  return (
    <img
      ref={imgRef}
      src={src}
      alt=""
      draggable={false}
      className={`pointer-events-none absolute select-none dynamic-pos ${selected ? "drop-shadow-[0_0_18px_rgba(124,92,255,0.35)]" : ""}`}
    />
  );
}

export function EditorCanvas() {
  const {
    currentPage,
    viewMode,
    toolMode,
    showOverlays,
    zoom,
    panOffset,
    lastRetypesetTime,
    selectedLayerId,
    selectedImageLayerKey,
    brushSize,
  } = useEditorStore();
  const setZoom = useEditorStore((s) => s.setZoom);
  const setPan = useEditorStore((s) => s.setPan);
  const selectLayer = useEditorStore((s) => s.selectLayer);
  const selectImageLayer = useEditorStore((s) => s.selectImageLayer);
  const createTextLayer = useEditorStore((s) => s.createTextLayer);
  const applyBitmapStroke = useEditorStore((s) => s.applyBitmapStroke);

  const [naturalSize, setNaturalSize] = useState({ w: 0, h: 0 });
  const [displaySize, setDisplaySize] = useState({ w: 0, h: 0 });
  const [isSpacePressed, setIsSpacePressed] = useState(false);
  const [mousePos, setMousePos] = useState({ x: -999, y: -999 });
  const [panSession, setPanSession] = useState<{
    startX: number;
    startY: number;
    originX: number;
    originY: number;
  } | null>(null);
  const [blockDraft, setBlockDraft] = useState<{
    startX: number;
    startY: number;
    currentX: number;
    currentY: number;
  } | null>(null);
  const [paintStroke, setPaintStroke] = useState<[number, number][]>([]);

  const imgRef = useRef<HTMLImageElement>(null);

  const displayImagePath = useMemo(() => {
    if (!currentPage) return null;
    if (viewMode === "original") return currentPage.image_layers?.base?.path ?? currentPage.arquivo_original;
    if (viewMode === "inpainted") {
      return (
        currentPage.image_layers?.inpaint?.path ??
        currentPage.image_layers?.base?.path ??
        currentPage.arquivo_original
      );
    }
    return (
      currentPage.image_layers?.rendered?.path ??
      currentPage.image_layers?.inpaint?.path ??
      currentPage.image_layers?.base?.path ??
      currentPage.arquivo_original
    );
  }, [currentPage, viewMode]);

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

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === " ") {
        const active = document.activeElement;
        if (active && ["INPUT", "TEXTAREA", "SELECT"].includes(active.tagName)) return;
        event.preventDefault();
        setIsSpacePressed(true);
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
    if (!blockDraft) return;
    const handleMouseMove = (event: MouseEvent) => {
      const rect = imgRef.current?.getBoundingClientRect();
      if (!rect) return;
      setBlockDraft((draft) =>
        draft
          ? {
              ...draft,
              currentX: Math.max(rect.left, Math.min(rect.right, event.clientX)),
              currentY: Math.max(rect.top, Math.min(rect.bottom, event.clientY)),
            }
          : null,
      );
    };
    const handleMouseUp = async () => {
      const draft = blockDraft;
      setBlockDraft(null);
      if (!draft || !naturalSize.w || !displaySize.w) return;
      const x1 = Math.min(draft.startX, draft.currentX);
      const y1 = Math.min(draft.startY, draft.currentY);
      const x2 = Math.max(draft.startX, draft.currentX);
      const y2 = Math.max(draft.startY, draft.currentY);
      if (x2 - x1 < 12 || y2 - y1 < 12) return;
      const rect = imgRef.current?.getBoundingClientRect();
      if (!rect) return;
      const scaleX = naturalSize.w / rect.width;
      const scaleY = naturalSize.h / rect.height;
      await createTextLayer([
        Math.round((x1 - rect.left) * scaleX),
        Math.round((y1 - rect.top) * scaleY),
        Math.round((x2 - rect.left) * scaleX),
        Math.round((y2 - rect.top) * scaleY),
      ]);
    };
    window.addEventListener("mousemove", handleMouseMove);
    const onMouseUp = () => void handleMouseUp();
    window.addEventListener("mouseup", onMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
    };
  }, [blockDraft, createTextLayer, displaySize.w, naturalSize.h, naturalSize.w]);

  useEffect(() => {
    if (paintStroke.length === 0) return;
    const handleMouseMove = (event: MouseEvent) => {
      const rect = imgRef.current?.getBoundingClientRect();
      if (!rect || !naturalSize.w || !naturalSize.h) return;
      const x = Math.round(((event.clientX - rect.left) / rect.width) * naturalSize.w);
      const y = Math.round(((event.clientY - rect.top) / rect.height) * naturalSize.h);
      setPaintStroke((points) => [...points, [x, y]]);
    };
    const handleMouseUp = async () => {
      if (!naturalSize.w || !naturalSize.h || paintStroke.length === 0) {
        setPaintStroke([]);
        return;
      }
      const stroke = paintStroke;
      setPaintStroke([]);
      await applyBitmapStroke({
        width: naturalSize.w,
        height: naturalSize.h,
        strokes: [stroke],
      });
    };
    window.addEventListener("mousemove", handleMouseMove);
    const onMouseUp = () => void handleMouseUp();
    window.addEventListener("mouseup", onMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
    };
  }, [applyBitmapStroke, naturalSize.h, naturalSize.w, paintStroke]);

  useEffect(() => {
    const observer = new ResizeObserver(() => {
      const img = imgRef.current;
      if (img) {
        setDisplaySize({ w: img.clientWidth, h: img.clientHeight });
      }
    });
    if (imgRef.current) observer.observe(imgRef.current);
    return () => observer.disconnect();
  }, [baseImageSrc, lastRetypesetTime]);

  const handleImageLoad = () => {
    const img = imgRef.current;
    if (!img) return;
    setNaturalSize({ w: img.naturalWidth, h: img.naturalHeight });
    setDisplaySize({ w: img.clientWidth, h: img.clientHeight });
  };

  const handleWheel = (event: React.WheelEvent) => {
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

  const beginPan = (clientX: number, clientY: number) => {
    setPanSession({
      startX: clientX,
      startY: clientY,
      originX: panOffset.x,
      originY: panOffset.y,
    });
  };

  const handleViewportMouseDown = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!imgRef.current) return;
    const rect = imgRef.current.getBoundingClientRect();
    const insideImage =
      event.clientX >= rect.left &&
      event.clientX <= rect.right &&
      event.clientY >= rect.top &&
      event.clientY <= rect.bottom;

    if (event.button === 1 || (event.button === 0 && isSpacePressed)) {
      event.preventDefault();
      beginPan(event.clientX, event.clientY);
      return;
    }

    if (!insideImage) {
      selectLayer(null);
      selectImageLayer(null);
      return;
    }

    if (toolMode === "block") {
      event.preventDefault();
      setBlockDraft({
        startX: event.clientX,
        startY: event.clientY,
        currentX: event.clientX,
        currentY: event.clientY,
      });
      return;
    }

    if (toolMode === "brush" || toolMode === "repairBrush" || toolMode === "eraser") {
      event.preventDefault();
      const x = Math.round(((event.clientX - rect.left) / rect.width) * naturalSize.w);
      const y = Math.round(((event.clientY - rect.top) / rect.height) * naturalSize.h);
      setPaintStroke([[x, y]]);
      selectLayer(null);
      selectImageLayer(toolMode === "brush" ? "brush" : "mask");
      return;
    }

    if (event.target === event.currentTarget || (event.target instanceof HTMLElement && event.target.id === "canvas-viewport")) {
      selectLayer(null);
      selectImageLayer(null);
    }
  };

  const scaleX = naturalSize.w > 0 ? displaySize.w / naturalSize.w : 1;
  const scaleY = naturalSize.h > 0 ? displaySize.h / naturalSize.h : 1;
  const showSpritePreview = viewMode === "translated";
  const layers = currentPage?.text_layers ?? [];
  const selectedLayer = layers.find((layer) => layer.id === selectedLayerId) ?? null;

  const viewportRef = useDynamicStyle<HTMLDivElement>(
    {
      "--canvas-cursor": panSession
        ? "grabbing"
        : isSpacePressed
        ? "grab"
        : toolMode === "block"
        ? "crosshair"
        : toolMode === "brush" || toolMode === "repairBrush" || toolMode === "eraser"
        ? "none"
        : "default",
    },
    [panSession, isSpacePressed, toolMode]
  );

  return (
    <div
      ref={viewportRef}
      className="relative flex-1 overflow-hidden dynamic-cursor bg-[radial-gradient(circle_at_top,_rgba(72,176,255,0.08),_transparent_38%),linear-gradient(180deg,_rgba(255,255,255,0.02),_transparent_28%)]"
      onWheel={handleWheel}
      onMouseDown={handleViewportMouseDown}
      onMouseMove={(e) => {
        if (toolMode === "brush" || toolMode === "repairBrush" || toolMode === "eraser") {
          setMousePos({ x: e.clientX, y: e.clientY });
        }
      }}
      onMouseLeave={() => setMousePos({ x: -999, y: -999 })}
    >
      <div className="pointer-events-none absolute inset-x-0 bottom-3 z-20 flex justify-center px-4">
        <div className="rounded-full border border-white/10 bg-black/45 px-3 py-1 text-[11px] text-text-secondary backdrop-blur">
          {toolMode === "block"
            ? "Arraste para criar uma nova camada de texto"
            : toolMode === "brush" || toolMode === "repairBrush" || toolMode === "eraser"
              ? `Pintura ativa • pincel ${brushSize}px`
              : "Ctrl+scroll: zoom • Space+drag: mover • Arraste blocos para reposicionar"}
        </div>
      </div>

      {(toolMode === "brush" || toolMode === "repairBrush" || toolMode === "eraser") &&
        mousePos.x > 0 && (
          <BrushCursor mousePos={mousePos} brushSize={brushSize} zoom={zoom} />
        )}

      <div className="flex h-full items-center justify-center overflow-hidden px-6 py-4">
        <TransformContainer panOffset={panOffset} zoom={zoom} panSession={!!panSession}>
          {currentPage && baseImageSrc ? (
            <div className="relative inline-block">
              <img
                ref={imgRef}
                src={baseImageSrc}
                alt={`Página ${currentPage.numero}`}
                className="max-h-[calc(100vh-96px)] w-auto rounded-2xl border border-white/10 object-contain shadow-[0_20px_60px_rgba(0,0,0,0.55)]"
                onLoad={handleImageLoad}
                draggable={false}
              />

              {maskOverlaySrc && (
                <img
                  src={maskOverlaySrc}
                  alt=""
                  draggable={false}
                  className={`absolute inset-0 h-full w-full object-contain ${selectedImageLayerKey === "mask" ? "opacity-75" : "opacity-55"}`}
                />
              )}

              {brushOverlaySrc && (
                <img
                  src={brushOverlaySrc}
                  alt=""
                  draggable={false}
                  className="absolute inset-0 h-full w-full object-contain mix-blend-screen"
                  style={{ opacity: selectedImageLayerKey === "brush" ? 0.85 : 0.6 }}
                />
              )}

              {showSpritePreview &&
                layers
                  .filter((entry) => entry.visible !== false && entry.render_preview_path && entry.render_bbox)
                  .map((entry) => (
                    <LayerSprite
                      key={`${entry.id}-sprite`}
                      path={entry.render_preview_path!}
                      bbox={entry.render_bbox!}
                      scaleX={scaleX}
                      scaleY={scaleY}
                      selected={selectedLayerId === entry.id}
                      version={lastRetypesetTime}
                    />
                  ))}

              {layers.map((entry) => (
                <TextOverlay
                  key={entry.id}
                  entry={entry}
                  scaleX={scaleX}
                  scaleY={scaleY}
                  mode={showSpritePreview && entry.render_preview_path ? "guide" : showSpritePreview ? "text" : "guide"}
                  showGuides={showOverlays}
                />
              ))}

              {selectedLayer && showOverlays && (
                <SelectionHighlight
                  bbox={selectedLayer.layout_bbox ?? selectedLayer.bbox}
                  scaleX={scaleX}
                  scaleY={scaleY}
                />
              )}

              {blockDraft && imgRef.current && (
                <DraftHighlight
                  startX={blockDraft.startX}
                  startY={blockDraft.startY}
                  currentX={blockDraft.currentX}
                  currentY={blockDraft.currentY}
                  offsetRect={imgRef.current.getBoundingClientRect()}
                />
              )}
            </div>
          ) : currentPage ? (
            <p className="text-sm text-text-secondary">Carregando imagem...</p>
          ) : (
            <p className="text-text-secondary">Nenhuma página carregada</p>
          )}
        </TransformContainer>
      </div>
    </div>
  );
}

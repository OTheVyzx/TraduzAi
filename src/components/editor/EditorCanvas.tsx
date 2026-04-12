import { useEffect, useMemo, useRef, useState } from "react";
import { readFile } from "@tauri-apps/plugin-fs";
import { useAppStore } from "../../lib/stores/appStore";
import { useEditorStore } from "../../lib/stores/editorStore";
import { TextOverlay } from "./TextOverlay";

function getInpaintedImagePath(
  outputPath: string | undefined,
  originalPath: string | undefined,
) {
  if (!outputPath || !originalPath) return null;
  const filename = originalPath.split(/[\\/]/).pop();
  if (!filename) return null;
  return `${outputPath}/images/${filename}`.replace(/\\/g, "/");
}

export function EditorCanvas() {
  const project = useAppStore((s) => s.project);
  const {
    currentPageIndex,
    viewMode,
    showOverlays,
    zoom,
    panOffset,
    lastRetypesetTime,
  } = useEditorStore();
  const setZoom = useEditorStore((s) => s.setZoom);
  const setPan = useEditorStore((s) => s.setPan);
  const selectLayer = useEditorStore((s) => s.selectLayer);

  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [naturalSize, setNaturalSize] = useState({ w: 0, h: 0 });
  const [displaySize, setDisplaySize] = useState({ w: 0, h: 0 });
  const [isSpacePressed, setIsSpacePressed] = useState(false);
  const [panSession, setPanSession] = useState<{
    startX: number;
    startY: number;
    originX: number;
    originY: number;
  } | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  const page = project?.paginas[currentPageIndex];
  const textos = page?.textos ?? [];

  const currentImagePath = useMemo(() => {
    if (!project || project.paginas.length === 0) return null;
    const currentPage = project.paginas[currentPageIndex];
    if (!currentPage) return null;

    const inpaintedPath = getInpaintedImagePath(project.output_path, currentPage.arquivo_original);

    switch (viewMode) {
      case "translated":
        return inpaintedPath ?? currentPage.arquivo_traduzido ?? currentPage.arquivo_original ?? null;
      case "original":
        return currentPage.arquivo_original ?? null;
      case "inpainted":
        return inpaintedPath ?? currentPage.arquivo_original ?? null;
    }
  }, [project, currentPageIndex, viewMode]);

  useEffect(() => {
    let objectUrl: string | null = null;
    let cancelled = false;

    if (currentImagePath) {
      readFile(currentImagePath)
        .then((bytes) => {
          if (cancelled) return;
          const blob = new Blob([bytes]);
          objectUrl = URL.createObjectURL(blob);
          setImageSrc(objectUrl);
        })
        .catch(console.error);
    } else {
      setImageSrc(null);
    }

    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [currentImagePath, lastRetypesetTime]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== " ") return;
      const active = document.activeElement;
      if (
        active &&
        (active.tagName === "INPUT" || active.tagName === "TEXTAREA" || active.tagName === "SELECT")
      ) {
        return;
      }
      event.preventDefault();
      setIsSpacePressed(true);
    };

    const handleKeyUp = (event: KeyboardEvent) => {
      if (event.key === " ") {
        setIsSpacePressed(false);
      }
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

  const handleImageLoad = () => {
    const img = imgRef.current;
    if (!img) return;
    setNaturalSize({ w: img.naturalWidth, h: img.naturalHeight });
    setDisplaySize({ w: img.clientWidth, h: img.clientHeight });
  };

  useEffect(() => {
    const observer = new ResizeObserver(() => {
      const img = imgRef.current;
      if (img) setDisplaySize({ w: img.clientWidth, h: img.clientHeight });
    });
    if (imgRef.current) observer.observe(imgRef.current);
    return () => observer.disconnect();
  }, [imageSrc]);

  const handleWheel = (event: React.WheelEvent) => {
    event.preventDefault();

    if (event.ctrlKey || event.metaKey) {
      const delta = event.deltaY > 0 ? -0.1 : 0.1;
      setZoom(zoom + delta);
      return;
    }

    setPan({
      x: panOffset.x - event.deltaX * 0.6,
      y: panOffset.y - event.deltaY * 0.6,
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
    const target = event.target as HTMLElement;
    const clickedImage = imgRef.current != null && target === imgRef.current;
    const clickedStage = target === event.currentTarget;

    if (event.button === 1 || (event.button === 0 && isSpacePressed)) {
      event.preventDefault();
      beginPan(event.clientX, event.clientY);
      return;
    }

    if (event.button === 0 && (clickedStage || clickedImage)) {
      selectLayer(null);
    }
  };

  const scaleX = naturalSize.w > 0 ? displaySize.w / naturalSize.w : 1;
  const scaleY = naturalSize.h > 0 ? displaySize.h / naturalSize.h : 1;
  const cursor = panSession ? "grabbing" : isSpacePressed ? "grab" : "default";
  const showLiveText = viewMode === "translated";
  const shouldRenderTextLayer = showLiveText || showOverlays;

  return (
    <div
      ref={containerRef}
      className="relative flex-1 overflow-hidden bg-bg-primary"
      onWheel={handleWheel}
      onMouseDown={handleViewportMouseDown}
      style={{ cursor }}
    >
      <div className="pointer-events-none absolute inset-x-0 bottom-3 z-10 flex justify-center px-4">
        <div className="rounded-full border border-white/10 bg-black/45 px-3 py-1 text-[11px] text-text-secondary backdrop-blur">
          Ctrl+scroll: zoom | Space+drag ou botao do meio: mover | Na vista traduzida voce move o texto direto
        </div>
      </div>

      <div className="flex h-full items-center justify-center overflow-hidden px-6 py-4">
        <div
          style={{
            transform: `translate(${panOffset.x}px, ${panOffset.y}px) scale(${zoom})`,
            transformOrigin: "center center",
            transition: panSession ? "none" : "transform 0.1s ease-out",
          }}
        >
          {page && imageSrc ? (
            <div className="relative inline-block">
              <img
                ref={imgRef}
                src={imageSrc}
                alt={`Pagina ${page.numero}`}
                className="max-h-[calc(100vh-96px)] w-auto object-contain rounded shadow-2xl"
                onLoad={handleImageLoad}
                draggable={false}
              />
              {shouldRenderTextLayer &&
                textos.map((entry) => (
                  <TextOverlay
                    key={entry.id}
                    entry={entry}
                    scaleX={scaleX}
                    scaleY={scaleY}
                    mode={showLiveText ? "text" : "guide"}
                    showGuides={showOverlays}
                  />
                ))}
            </div>
          ) : page ? (
            <p className="text-sm text-text-secondary">Carregando imagem...</p>
          ) : (
            <p className="text-text-secondary">Nenhuma pagina para exibir</p>
          )}
        </div>
      </div>
    </div>
  );
}

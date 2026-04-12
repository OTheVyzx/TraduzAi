import { useState, useEffect } from "react";
import { useEditorStore } from "../../lib/stores/editorStore";
import type { TextEntry } from "../../lib/stores/appStore";

interface TextOverlayProps {
  entry: TextEntry;
  scaleX: number;
  scaleY: number;
  mode?: "guide" | "text";
  showGuides?: boolean;
}

export function TextOverlay({
  entry,
  scaleX,
  scaleY,
  mode = "guide",
  showGuides = true,
}: TextOverlayProps) {
  const { selectedLayerId, hoveredLayerId, pendingEdits, hiddenLayers, currentPageIndex } =
    useEditorStore();
  const selectLayer = useEditorStore((s) => s.selectLayer);
  const hoverLayer = useEditorStore((s) => s.hoverLayer);
  const updatePendingEdit = useEditorStore((s) => s.updatePendingEdit);

  const hidden = (hiddenLayers[currentPageIndex] ?? []).includes(entry.id);

  const [dragState, setDragState] = useState<{
    type: 'move' | 'n' | 's' | 'e' | 'w' | 'nw' | 'ne' | 'sw' | 'se';
    startX: number;
    startY: number;
    initialBbox: [number, number, number, number];
  } | null>(null);

  useEffect(() => {
    if (!dragState) return;

    const handleMouseMove = (e: MouseEvent) => {
      e.preventDefault();
      const dx = Math.round((e.clientX - dragState.startX) / scaleX);
      const dy = Math.round((e.clientY - dragState.startY) / scaleY);

      let [nx1, ny1, nx2, ny2] = dragState.initialBbox;

      if (dragState.type === 'move') {
        nx1 += dx; ny1 += dy; nx2 += dx; ny2 += dy;
      } else {
        if (dragState.type.includes('n')) ny1 = Math.min(ny1 + dy, ny2 - 10);
        if (dragState.type.includes('s')) ny2 = Math.max(ny2 + dy, ny1 + 10);
        if (dragState.type.includes('w')) nx1 = Math.min(nx1 + dx, nx2 - 10);
        if (dragState.type.includes('e')) nx2 = Math.max(nx2 + dx, nx1 + 10);
      }

      updatePendingEdit(entry.id, { bbox: [nx1, ny1, nx2, ny2] });
    };

    const handleMouseUp = () => setDragState(null);

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [dragState, scaleX, scaleY, entry.id, updatePendingEdit]);

  if (hidden) return null;

  const edit = pendingEdits[entry.id];
  const traduzido = edit?.traduzido ?? entry.traduzido;
  const estilo = edit?.estilo ? { ...entry.estilo, ...edit.estilo } : entry.estilo;
  const bbox = edit?.bbox ?? entry.bbox;
  const isTextMode = mode === "text";

  const isSelected = selectedLayerId === entry.id;
  const isHovered = hoveredLayerId === entry.id;

  const [x1, y1, x2, y2] = bbox;
  const left = x1 * scaleX;
  const top = y1 * scaleY;
  const width = (x2 - x1) * scaleX;
  const height = (y2 - y1) * scaleY;
  const fontSize = Math.max(8, estilo.tamanho * Math.min(scaleX, scaleY));

  const handlePointerDown = (type: 'move' | 'n' | 's' | 'e' | 'w' | 'nw' | 'ne' | 'sw' | 'se') => (e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    selectLayer(entry.id);
    setDragState({
      type,
      startX: e.clientX,
      startY: e.clientY,
      initialBbox: bbox as [number, number, number, number]
    });
  };

  const handleSize = 6;
  const handleStyle = {
    position: "absolute" as const,
    width: handleSize,
    height: handleSize,
    background: "#ffffff",
    border: "1px solid rgb(124, 92, 255)",
    zIndex: 10,
  };

  const showFrame = isSelected || isHovered || (!isTextMode && showGuides);
  const borderStyle = isSelected
    ? "2px solid rgb(124, 92, 255)"
    : isHovered
      ? "1px solid rgba(124, 92, 255, 0.5)"
      : !isTextMode && showGuides
        ? "1px dashed rgba(255, 255, 255, 0.15)"
        : "1px solid transparent";
  const backgroundStyle = isTextMode
    ? "transparent"
    : isSelected
      ? "rgba(124, 92, 255, 0.08)"
      : isHovered
        ? "rgba(124, 92, 255, 0.04)"
        : "transparent";

  return (
    <div
      className={`absolute select-none ${dragState ? "" : "transition-border transition-colors duration-150"}`}
      style={{
        left,
        top,
        width,
        height,
        border: showFrame ? borderStyle : "1px solid transparent",
        borderRadius: isTextMode ? 10 : 4,
        background: backgroundStyle,
        display: "flex",
        alignItems: "center",
        justifyContent:
          estilo.alinhamento === "left"
            ? "flex-start"
            : estilo.alinhamento === "right"
              ? "flex-end"
              : "center",
        padding: "2px 4px",
        overflow: "visible", // To let handles show outside
        cursor: dragState?.type === 'move' ? 'grabbing' : isSelected ? 'grab' : 'pointer',
      }}
      onMouseDown={handlePointerDown('move')}
      onClick={(e) => e.stopPropagation()}
      onDoubleClick={(e) => e.stopPropagation()}
      onMouseEnter={() => hoverLayer(entry.id)}
      onMouseLeave={() => hoverLayer(null)}
    >
      <div className="relative w-full h-full flex items-center justify-inherit overflow-hidden">
        <span
          style={{
            fontSize,
            color: estilo.cor || "#FFFFFF",
            fontWeight: estilo.bold ? "bold" : "normal",
            fontStyle: estilo.italico ? "italic" : "normal",
            textAlign: estilo.alinhamento || "center",
            lineHeight: 1.15,
            wordBreak: "break-word",
            WebkitTextStroke:
              estilo.contorno_px > 0
                ? `${Math.max(1, estilo.contorno_px * Math.min(scaleX, scaleY) * 0.5)}px ${estilo.contorno || "#000000"}`
                : undefined,
            textShadow: estilo.sombra
              ? `${estilo.sombra_offset?.[0] ?? 2}px ${estilo.sombra_offset?.[1] ?? 2}px 2px ${estilo.sombra_cor || "#000000"}`
              : undefined,
            width: "100%",
            filter: isTextMode && isSelected ? "drop-shadow(0 0 12px rgba(124, 92, 255, 0.18))" : undefined,
          }}
        >
          {traduzido}
        </span>
      </div>

      {/* Resize handles */}
      {isSelected && (
        <>
          <div onMouseDown={handlePointerDown('nw')} style={{ ...handleStyle, top: -3, left: -3, cursor: 'nwse-resize' }} />
          <div onMouseDown={handlePointerDown('n')} style={{ ...handleStyle, top: -3, left: '50%', transform: 'translateX(-50%)', cursor: 'ns-resize' }} />
          <div onMouseDown={handlePointerDown('ne')} style={{ ...handleStyle, top: -3, right: -3, cursor: 'nesw-resize' }} />
          <div onMouseDown={handlePointerDown('w')} style={{ ...handleStyle, top: '50%', left: -3, transform: 'translateY(-50%)', cursor: 'ew-resize' }} />
          <div onMouseDown={handlePointerDown('e')} style={{ ...handleStyle, top: '50%', right: -3, transform: 'translateY(-50%)', cursor: 'ew-resize' }} />
          <div onMouseDown={handlePointerDown('sw')} style={{ ...handleStyle, bottom: -3, left: -3, cursor: 'nesw-resize' }} />
          <div onMouseDown={handlePointerDown('s')} style={{ ...handleStyle, bottom: -3, left: '50%', transform: 'translateX(-50%)', cursor: 'ns-resize' }} />
          <div onMouseDown={handlePointerDown('se')} style={{ ...handleStyle, bottom: -3, right: -3, cursor: 'nwse-resize' }} />
        </>
      )}
    </div>
  );
}

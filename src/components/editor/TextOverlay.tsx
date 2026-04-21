import { useEffect, useState } from "react";
import { useEditorStore } from "../../lib/stores/editorStore";
import type { TextEntry } from "../../lib/stores/appStore";

interface TextOverlayProps {
  entry: TextEntry;
  scaleX: number;
  scaleY: number;
  mode?: "guide" | "text";
  showGuides?: boolean;
}

type DragHandle = "move" | "n" | "s" | "e" | "w" | "nw" | "ne" | "sw" | "se";

export function TextOverlay({
  entry,
  scaleX,
  scaleY,
  mode = "guide",
  showGuides = true,
}: TextOverlayProps) {
  const { selectedLayerId, hoveredLayerId, pendingEdits, toolMode } = useEditorStore();
  const selectLayer = useEditorStore((s) => s.selectLayer);
  const hoverLayer = useEditorStore((s) => s.hoverLayer);
  const updatePendingEdit = useEditorStore((s) => s.updatePendingEdit);

  const [dragState, setDragState] = useState<{
    type: DragHandle;
    startX: number;
    startY: number;
    initialBbox: [number, number, number, number];
  } | null>(null);

  useEffect(() => {
    if (!dragState) return;
    const handleMouseMove = (event: MouseEvent) => {
      event.preventDefault();
      const dx = Math.round((event.clientX - dragState.startX) / scaleX);
      const dy = Math.round((event.clientY - dragState.startY) / scaleY);
      let [x1, y1, x2, y2] = dragState.initialBbox;

      if (dragState.type === "move") {
        x1 += dx;
        y1 += dy;
        x2 += dx;
        y2 += dy;
      } else {
        if (dragState.type.includes("n")) y1 = Math.min(y1 + dy, y2 - 12);
        if (dragState.type.includes("s")) y2 = Math.max(y2 + dy, y1 + 12);
        if (dragState.type.includes("w")) x1 = Math.min(x1 + dx, x2 - 12);
        if (dragState.type.includes("e")) x2 = Math.max(x2 + dx, x1 + 12);
      }

      updatePendingEdit(entry.id, { bbox: [x1, y1, x2, y2] });
    };

    const handleMouseUp = () => setDragState(null);
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [dragState, entry.id, scaleX, scaleY, updatePendingEdit]);

  if (entry.visible === false) return null;

  const edit = pendingEdits[entry.id];
  const bbox = edit?.bbox ?? entry.layout_bbox ?? entry.bbox;
  const text = edit?.traduzido ?? edit?.translated ?? entry.traduzido ?? entry.translated ?? "";
  const style = edit?.estilo ? { ...entry.estilo, ...edit.estilo } : entry.estilo;
  const [x1, y1, x2, y2] = bbox;
  const left = x1 * scaleX;
  const top = y1 * scaleY;
  const width = Math.max(1, (x2 - x1) * scaleX);
  const height = Math.max(1, (y2 - y1) * scaleY);
  const fontSize = Math.max(8, style.tamanho * Math.min(scaleX, scaleY));
  const isSelected = selectedLayerId === entry.id;
  const isHovered = hoveredLayerId === entry.id;
  const showFrame = isSelected || isHovered || (showGuides && mode === "guide");
  const interactive = toolMode === "select";

  const handlePointerDown = (type: DragHandle) => (event: React.MouseEvent) => {
    if (!interactive) return;
    event.preventDefault();
    event.stopPropagation();
    selectLayer(entry.id);
    setDragState({
      type,
      startX: event.clientX,
      startY: event.clientY,
      initialBbox: bbox as [number, number, number, number],
    });
  };

  const handleStyle = {
    position: "absolute" as const,
    width: 7,
    height: 7,
    background: "#f8fbff",
    border: "1px solid rgba(124, 92, 255, 0.9)",
    boxShadow: "0 0 12px rgba(124, 92, 255, 0.35)",
    zIndex: 10,
  };

  return (
    <div
      className="absolute select-none"
      style={{
        left,
        top,
        width,
        height,
        border: showFrame
          ? isSelected
            ? "2px solid rgba(124, 92, 255, 0.95)"
            : isHovered
              ? "1px solid rgba(124, 92, 255, 0.55)"
              : "1px dashed rgba(255, 255, 255, 0.18)"
          : "1px solid transparent",
        borderRadius: 18,
        background:
          mode === "guide"
            ? isSelected
              ? "rgba(124, 92, 255, 0.08)"
              : isHovered
                ? "rgba(124, 92, 255, 0.05)"
                : "transparent"
            : "transparent",
        cursor: interactive ? (dragState ? "grabbing" : "grab") : "default",
        pointerEvents: "auto",
      }}
      onMouseDown={handlePointerDown("move")}
      onMouseEnter={() => hoverLayer(entry.id)}
      onMouseLeave={() => hoverLayer(null)}
      onClick={(event) => {
        event.stopPropagation();
        selectLayer(entry.id);
      }}
    >
      {mode === "text" && (
        <div className="flex h-full w-full items-center justify-center overflow-hidden px-2 text-center">
          <span
            style={{
              fontSize,
              color: style.cor || "#FFFFFF",
              fontWeight: style.bold ? "bold" : "normal",
              fontStyle: style.italico ? "italic" : "normal",
              textAlign: style.alinhamento || "center",
              lineHeight: 1.1,
              wordBreak: "break-word",
              width: "100%",
              WebkitTextStroke:
                style.contorno_px > 0
                  ? `${Math.max(1, style.contorno_px * Math.min(scaleX, scaleY) * 0.5)}px ${style.contorno || "#000000"}`
                  : undefined,
              textShadow: style.sombra
                ? `${style.sombra_offset?.[0] ?? 2}px ${style.sombra_offset?.[1] ?? 2}px 2px ${style.sombra_cor || "#000000"}`
                : undefined,
            }}
          >
            {text}
          </span>
        </div>
      )}

      {interactive && isSelected && (
        <>
          <div onMouseDown={handlePointerDown("nw")} style={{ ...handleStyle, top: -4, left: -4, cursor: "nwse-resize" }} />
          <div onMouseDown={handlePointerDown("n")} style={{ ...handleStyle, top: -4, left: "50%", transform: "translateX(-50%)", cursor: "ns-resize" }} />
          <div onMouseDown={handlePointerDown("ne")} style={{ ...handleStyle, top: -4, right: -4, cursor: "nesw-resize" }} />
          <div onMouseDown={handlePointerDown("w")} style={{ ...handleStyle, top: "50%", left: -4, transform: "translateY(-50%)", cursor: "ew-resize" }} />
          <div onMouseDown={handlePointerDown("e")} style={{ ...handleStyle, top: "50%", right: -4, transform: "translateY(-50%)", cursor: "ew-resize" }} />
          <div onMouseDown={handlePointerDown("sw")} style={{ ...handleStyle, bottom: -4, left: -4, cursor: "nesw-resize" }} />
          <div onMouseDown={handlePointerDown("s")} style={{ ...handleStyle, bottom: -4, left: "50%", transform: "translateX(-50%)", cursor: "ns-resize" }} />
          <div onMouseDown={handlePointerDown("se")} style={{ ...handleStyle, bottom: -4, right: -4, cursor: "nwse-resize" }} />
        </>
      )}
    </div>
  );
}

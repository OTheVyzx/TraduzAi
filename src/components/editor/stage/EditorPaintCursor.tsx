type Props = {
  x: number;
  y: number;
  /** Raio em pixels do viewport. */
  radius: number;
  toolMode: "brush" | "repairBrush" | "reinpaintBrush" | "eraser";
};

export function EditorPaintCursor({ x, y, radius, toolMode }: Props) {
  const cursorRadius = Math.max(4, radius);
  const strokeWidth = Math.max(3, Math.min(6, cursorRadius * 0.12));

  return (
    <div
      className="pointer-events-none absolute z-30 rounded-full"
      style={{
        left: x,
        top: y,
        width: cursorRadius * 2,
        height: cursorRadius * 2,
        transform: "translate(-50%, -50%)",
        border: `${strokeWidth}px solid #ffffff`,
        opacity: toolMode === "repairBrush" || toolMode === "reinpaintBrush" ? 0.96 : 1,
        mixBlendMode: "difference",
      }}
    />
  );
}

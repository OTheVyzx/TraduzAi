import { Circle } from "react-konva";

type Props = {
  x: number;
  y: number;
  /** Raio em pixels do canvas (brushSize / 2). */
  radius: number;
  toolMode: "brush" | "repairBrush" | "eraser";
};

/**
 * Cursor circular que acompanha o mouse em modos de pintura.
 * Cor varia por ferramenta: brush=azul, repairBrush=roxo, eraser=branco.
 */
export function EditorPaintCursor({ x, y, radius, toolMode }: Props) {
  const stroke =
    toolMode === "brush"
      ? "rgba(72, 176, 255, 0.9)"
      : toolMode === "eraser"
        ? "rgba(255, 255, 255, 0.75)"
        : "rgba(108, 92, 231, 0.9)";

  return (
    <Circle
      x={x}
      y={y}
      radius={Math.max(2, radius)}
      stroke={stroke}
      strokeWidth={1.5}
      fill="transparent"
      listening={false}
    />
  );
}

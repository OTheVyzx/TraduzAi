/**
 * Fase 8 — MaskInProgressOverlay
 *
 * Konva Layer que desenha o lasso em construção (freehand ou poligonal)
 * sobre o canvas. Coordenadas em espaço da imagem (mesmas do Stage).
 */
import { Circle, Layer, Line } from "react-konva";

interface Props {
  /** Pontos do lasso em espaço da imagem (px). */
  points: Array<[number, number]>;
  /** Modo freehand ou poligonal. */
  shape: "freehand" | "polygonal";
}

const LASSO_COLOR = "#6C5CE7";
const LASSO_DASH: number[] = [6, 4];

export function MaskInProgressOverlay({ points, shape }: Props) {
  if (points.length === 0) return null;

  // Flatten para o formato do Konva.Line: [x0, y0, x1, y1, ...]
  const flat = points.flatMap(([x, y]) => [x, y]);

  return (
    <Layer listening={false}>
      {/* Linha principal do lasso */}
      <Line
        points={flat}
        stroke={LASSO_COLOR}
        strokeWidth={1.5}
        dash={LASSO_DASH}
        closed={false}
        perfectDrawEnabled={false}
      />

      {/* Vértices visíveis somente no modo poligonal */}
      {shape === "polygonal" &&
        points.map(([x, y], i) => (
          <Circle
            key={i}
            x={x}
            y={y}
            radius={i === 0 ? 5 : 3}
            fill={i === 0 ? LASSO_COLOR : "#ffffff"}
            stroke={LASSO_COLOR}
            strokeWidth={1.5}
            perfectDrawEnabled={false}
          />
        ))}
    </Layer>
  );
}

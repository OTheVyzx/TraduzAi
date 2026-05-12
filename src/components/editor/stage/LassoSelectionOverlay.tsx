import { Layer, Line, Rect } from "react-konva";
import type { LassoSelection } from "../../../lib/lassoSelection";

export function LassoSelectionOverlay({ selection }: { selection: LassoSelection }) {
  return (
    <Layer listening={false}>
      <Line
        points={selection.points.flatMap(([x, y]) => [x, y])}
        closed
        stroke="rgba(255,255,255,0.95)"
        strokeWidth={1}
        dash={[5, 5]}
      />
      <Line
        points={selection.points.flatMap(([x, y]) => [x, y])}
        closed
        stroke="rgba(20,20,20,0.9)"
        strokeWidth={1}
        dash={[5, 5]}
        dashOffset={5}
      />
      <Rect
        x={selection.bbox[0]}
        y={selection.bbox[1]}
        width={selection.bbox[2] - selection.bbox[0]}
        height={selection.bbox[3] - selection.bbox[1]}
        stroke="rgba(108,92,231,0.45)"
        strokeWidth={1}
      />
    </Layer>
  );
}

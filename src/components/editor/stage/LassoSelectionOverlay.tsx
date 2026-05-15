import { useEffect, useState } from "react";
import { Layer, Line } from "react-konva";
import type { LassoSelection } from "../../../lib/lassoSelection";

export function LassoSelectionOverlay({ selection }: { selection: LassoSelection }) {
  const [dashOffset, setDashOffset] = useState(0);

  useEffect(() => {
    let frame = 0;
    let last = 0;
    const tick = (time: number) => {
      if (time - last > 80) {
        last = time;
        setDashOffset((value) => (value + 1) % 10);
      }
      frame = window.requestAnimationFrame(tick);
    };
    frame = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(frame);
  }, []);

  return (
    <Layer listening={false}>
      <Line
        points={selection.points.flatMap(([x, y]) => [x, y])}
        closed
        stroke="rgba(255,255,255,0.95)"
        strokeWidth={1}
        dash={[5, 5]}
        dashOffset={dashOffset}
      />
      <Line
        points={selection.points.flatMap(([x, y]) => [x, y])}
        closed
        stroke="rgba(20,20,20,0.9)"
        strokeWidth={1}
        dash={[5, 5]}
        dashOffset={dashOffset + 5}
      />
    </Layer>
  );
}

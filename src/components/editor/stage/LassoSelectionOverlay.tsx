import { useEffect, useState } from "react";
import { Group, Layer, Line } from "react-konva";
import type { LassoSelection } from "../../../lib/lassoSelection";

export function LassoSelectionOverlay({ selection }: { selection: LassoSelection }) {
  const [dashOffset, setDashOffset] = useState(0);
  const regions = selection.regions ?? [{ operation: "add" as const, points: selection.points }];

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
      {regions.map((region, index) => {
        const points = region.points.flatMap(([x, y]) => [x, y]);
        const subtracting = region.operation === "subtract";
        return (
          <Group key={`${region.operation}-${index}`}>
            <Line
              points={points}
              closed
              stroke={subtracting ? "rgba(251,146,60,0.98)" : "rgba(255,255,255,0.95)"}
              strokeWidth={1}
              dash={[5, 5]}
              dashOffset={dashOffset}
            />
            <Line
              points={points}
              closed
              stroke={subtracting ? "rgba(67,20,7,0.92)" : "rgba(20,20,20,0.9)"}
              strokeWidth={1}
              dash={[5, 5]}
              dashOffset={dashOffset + 5}
            />
          </Group>
        );
      })}
    </Layer>
  );
}

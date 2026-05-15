import { Fragment } from "react";
import { Line } from "react-konva";
import type { SnapGuide } from "./snapGuides";

export function EditorSnapGuides({ guides }: { guides: SnapGuide[] }) {
  if (guides.length === 0) return null;

  return (
    <Fragment>
      {guides.map((guide, index) => (
        <Line
          key={`${guide.orientation}-${guide.position}-${index}`}
          points={
            guide.orientation === "vertical"
              ? [guide.position, guide.start, guide.position, guide.end]
              : [guide.start, guide.position, guide.end, guide.position]
          }
          stroke={guide.source === "page" ? "rgba(0, 212, 255, 0.85)" : "rgba(255, 214, 102, 0.85)"}
          strokeWidth={1}
          dash={guide.source === "page" ? [7, 5] : [4, 4]}
          listening={false}
          perfectDrawEnabled={false}
        />
      ))}
    </Fragment>
  );
}

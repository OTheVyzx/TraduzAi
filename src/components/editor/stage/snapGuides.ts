import type { TextEntry } from "../../../lib/stores/appStore";
import { bboxToRect } from "./coordinateUtils";
import type { PageSize, TransformBox } from "./transformConstraints";

export type SnapGuide = {
  orientation: "vertical" | "horizontal";
  position: number;
  start: number;
  end: number;
  source: "page" | "layer";
};

type SnapTarget = {
  orientation: SnapGuide["orientation"];
  position: number;
  start: number;
  end: number;
  source: SnapGuide["source"];
};

export interface SnapGuideContext {
  pageSize: PageSize;
  layers: TextEntry[];
  excludeLayerId: string | null;
  threshold?: number;
}

export interface SnapGuideResult<T extends TransformBox> {
  rect: T;
  guides: SnapGuide[];
}

const DEFAULT_SNAP_THRESHOLD = 8;

function rectHandles(rect: TransformBox, orientation: SnapGuide["orientation"]) {
  if (orientation === "vertical") {
    return [
      { key: "start", value: rect.x },
      { key: "center", value: rect.x + rect.width / 2 },
      { key: "end", value: rect.x + rect.width },
    ] as const;
  }
  return [
    { key: "start", value: rect.y },
    { key: "center", value: rect.y + rect.height / 2 },
    { key: "end", value: rect.y + rect.height },
  ] as const;
}

function moveRect<T extends TransformBox>(rect: T, dx: number, dy: number): T {
  return {
    ...rect,
    x: rect.x + dx,
    y: rect.y + dy,
  };
}

function targetForPage(pageSize: PageSize): SnapTarget[] {
  return [
    { orientation: "vertical", position: 0, start: 0, end: pageSize.height, source: "page" },
    { orientation: "vertical", position: pageSize.width / 2, start: 0, end: pageSize.height, source: "page" },
    { orientation: "vertical", position: pageSize.width, start: 0, end: pageSize.height, source: "page" },
    { orientation: "horizontal", position: 0, start: 0, end: pageSize.width, source: "page" },
    { orientation: "horizontal", position: pageSize.height / 2, start: 0, end: pageSize.width, source: "page" },
    { orientation: "horizontal", position: pageSize.height, start: 0, end: pageSize.width, source: "page" },
  ];
}

function targetsForLayers(layers: TextEntry[], excludeLayerId: string | null, pageSize: PageSize): SnapTarget[] {
  const targets: SnapTarget[] = [];
  for (const layer of layers) {
    if (layer.id === excludeLayerId || layer.visible === false) continue;
    const rect = bboxToRect(layer.layout_bbox ?? layer.bbox);
    targets.push(
      { orientation: "vertical", position: rect.x, start: 0, end: pageSize.height, source: "layer" },
      { orientation: "vertical", position: rect.x + rect.width / 2, start: 0, end: pageSize.height, source: "layer" },
      { orientation: "vertical", position: rect.x + rect.width, start: 0, end: pageSize.height, source: "layer" },
      { orientation: "horizontal", position: rect.y, start: 0, end: pageSize.width, source: "layer" },
      { orientation: "horizontal", position: rect.y + rect.height / 2, start: 0, end: pageSize.width, source: "layer" },
      { orientation: "horizontal", position: rect.y + rect.height, start: 0, end: pageSize.width, source: "layer" },
    );
  }
  return targets;
}

function bestDelta(
  rect: TransformBox,
  targets: SnapTarget[],
  orientation: SnapGuide["orientation"],
  threshold: number,
) {
  let best: { delta: number; guide: SnapGuide; distance: number } | null = null;
  const handles = rectHandles(rect, orientation);
  for (const handle of handles) {
    for (const target of targets) {
      if (target.orientation !== orientation) continue;
      const delta = target.position - handle.value;
      const distance = Math.abs(delta);
      if (distance > threshold) continue;
      const preferTarget =
        !best ||
        distance < best.distance ||
        (distance === best.distance && best.guide.source === "page" && target.source === "layer");
      if (preferTarget) {
        best = {
          delta,
          distance,
          guide: {
            orientation,
            position: target.position,
            start: target.start,
            end: target.end,
            source: target.source,
          },
        };
      }
    }
  }
  return best;
}

export function snapRectToGuides<T extends TransformBox>(rect: T, context: SnapGuideContext): SnapGuideResult<T> {
  if (!context.pageSize.width || !context.pageSize.height) {
    return { rect, guides: [] };
  }

  const threshold = Math.max(0, context.threshold ?? DEFAULT_SNAP_THRESHOLD);
  const targets = [
    ...targetForPage(context.pageSize),
    ...targetsForLayers(context.layers, context.excludeLayerId, context.pageSize),
  ];
  const vertical = bestDelta(rect, targets, "vertical", threshold);
  const afterVertical = moveRect(rect, vertical?.delta ?? 0, 0);
  const horizontal = bestDelta(afterVertical, targets, "horizontal", threshold);
  const snapped = moveRect(afterVertical, 0, horizontal?.delta ?? 0);

  return {
    rect: snapped,
    guides: [vertical?.guide, horizontal?.guide].filter((guide): guide is SnapGuide => !!guide),
  };
}

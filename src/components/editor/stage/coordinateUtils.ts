import type { TextEntry } from "../../../lib/stores/appStore";

export type Bbox = TextEntry["bbox"];

export function normalizeBbox(bbox: Bbox): Bbox {
  const [x1, y1, x2, y2] = bbox;
  return [
    Math.round(Math.min(x1, x2)),
    Math.round(Math.min(y1, y2)),
    Math.round(Math.max(x1, x2)),
    Math.round(Math.max(y1, y2)),
  ];
}

export function bboxToRect(bbox: Bbox) {
  const [x1, y1, x2, y2] = normalizeBbox(bbox);
  return {
    x: x1,
    y: y1,
    width: Math.max(1, x2 - x1),
    height: Math.max(1, y2 - y1),
  };
}

export function rectToBbox(rect: { x: number; y: number; width: number; height: number }): Bbox {
  const x = Math.round(rect.x);
  const y = Math.round(rect.y);
  return normalizeBbox([x, y, x + Math.round(rect.width), y + Math.round(rect.height)]);
}

export function sameBbox(a: Bbox, b: Bbox) {
  return a.length === b.length && a.every((value, index) => Math.abs(value - b[index]) < 1);
}

export function clientPointToImagePoint(args: {
  clientX: number;
  clientY: number;
  stageRect: DOMRect;
  scale: number;
}) {
  return {
    x: Math.round((args.clientX - args.stageRect.left) / args.scale),
    y: Math.round((args.clientY - args.stageRect.top) / args.scale),
  };
}

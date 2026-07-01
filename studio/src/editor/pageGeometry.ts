import type { StudioPage } from "../project/studioProject";

export interface PageSize {
  width: number;
  height: number;
}

export function inferPageSize(page: StudioPage | null): PageSize {
  const boxes = page?.text_layers.map((layer) => layer.bbox) ?? [];
  const maxX = Math.max(900, ...boxes.map((bbox) => bbox[2]));
  const maxY = Math.max(1280, ...boxes.map((bbox) => bbox[3]));
  return { width: maxX, height: maxY };
}

export function bboxToPercentStyle(bbox: [number, number, number, number], pageSize: PageSize) {
  const [x1, y1, x2, y2] = bbox;
  const width = Math.max(1, x2 - x1);
  const height = Math.max(1, y2 - y1);
  return {
    left: `${(x1 / pageSize.width) * 100}%`,
    top: `${(y1 / pageSize.height) * 100}%`,
    width: `${(width / pageSize.width) * 100}%`,
    height: `${(height / pageSize.height) * 100}%`,
  };
}

export function readableLayerLabel(index: number) {
  return `Texto ${String(index + 1).padStart(2, "0")}`;
}

import type { EditorToolMode } from "./stores/editorStore";

export type ImagePoint = { x: number; y: number };
export type Bbox = [number, number, number, number];
export type BitmapTarget = "brush" | "mask" | "recovery" | "reinpaint";

export function pointFromStageClientRect(args: {
  clientX: number;
  clientY: number;
  rect: Pick<DOMRect, "left" | "top" | "width" | "height">;
  imageWidth: number;
  imageHeight: number;
}): ImagePoint | null {
  const { clientX, clientY, rect, imageWidth, imageHeight } = args;
  if (
    !Number.isFinite(clientX) ||
    !Number.isFinite(clientY) ||
    !Number.isFinite(rect.left) ||
    !Number.isFinite(rect.top) ||
    !Number.isFinite(rect.width) ||
    !Number.isFinite(rect.height) ||
    !Number.isFinite(imageWidth) ||
    !Number.isFinite(imageHeight) ||
    rect.width <= 0 ||
    rect.height <= 0 ||
    imageWidth <= 0 ||
    imageHeight <= 0
  ) {
    return null;
  }

  const x = ((clientX - rect.left) / rect.width) * imageWidth;
  const y = ((clientY - rect.top) / rect.height) * imageHeight;

  return {
    x: Math.max(0, Math.min(imageWidth, Math.round(x))),
    y: Math.max(0, Math.min(imageHeight, Math.round(y))),
  };
}

export function shouldAppendStrokePoint(last: [number, number] | undefined, point: ImagePoint): boolean {
  return !last || last[0] !== point.x || last[1] !== point.y;
}

export function strokeDirtyBbox(args: {
  stroke: [number, number][];
  brushSize: number;
  width: number;
  height: number;
}): Bbox | null {
  const { stroke, brushSize, width, height } = args;
  if (
    stroke.length === 0 ||
    !Number.isFinite(brushSize) ||
    !Number.isFinite(width) ||
    !Number.isFinite(height) ||
    width <= 0 ||
    height <= 0
  ) {
    return null;
  }

  const pad = Math.max(1, Math.ceil(brushSize / 2) + 2);
  const finitePoints = stroke.filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
  if (finitePoints.length === 0) return null;
  const xs = finitePoints.map(([x]) => x);
  const ys = finitePoints.map(([, y]) => y);

  return [
    Math.max(0, Math.floor(Math.min(...xs) - pad)),
    Math.max(0, Math.floor(Math.min(...ys) - pad)),
    Math.min(width, Math.ceil(Math.max(...xs) + pad)),
    Math.min(height, Math.ceil(Math.max(...ys) + pad)),
  ];
}

export function bitmapTargetForEditorTool(
  toolMode: EditorToolMode,
  eraserTarget: "brush" | "mask" | "recovery" | null,
  lastPaintedLayer: "brush" | "mask" | "recovery" = "brush",
): BitmapTarget | null {
  if (toolMode === "brush") return "brush";
  if (toolMode === "repairBrush") return "recovery";
  if (toolMode === "reinpaintBrush") return "reinpaint";
  if (toolMode === "mask") return "mask";
  if (toolMode === "eraser") {
    const target = eraserTarget ?? lastPaintedLayer;
    return target === "mask" ? "mask" : "brush";
  }
  return null;
}

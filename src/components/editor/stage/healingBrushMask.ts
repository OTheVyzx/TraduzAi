export type HealingBrushBBox = [number, number, number, number];

export function clampHealingBrushBBox(
  bbox: HealingBrushBBox,
  width: number,
  height: number,
): HealingBrushBBox | null {
  const x1 = Math.max(0, Math.floor(Math.min(bbox[0], bbox[2])));
  const y1 = Math.max(0, Math.floor(Math.min(bbox[1], bbox[3])));
  const x2 = Math.min(width, Math.ceil(Math.max(bbox[0], bbox[2])));
  const y2 = Math.min(height, Math.ceil(Math.max(bbox[1], bbox[3])));
  if (x2 <= x1 || y2 <= y1) return null;
  return [x1, y1, x2, y2];
}

export function paddedStrokeBBox(input: {
  stroke: [number, number][];
  brushSize: number;
  width: number;
  height: number;
}): HealingBrushBBox | null {
  const { stroke, brushSize, width, height } = input;
  if (stroke.length === 0 || width <= 0 || height <= 0) return null;
  const xs = stroke.map(([x]) => x);
  const ys = stroke.map(([, y]) => y);
  const pad = Math.max(32, Math.ceil(brushSize * 2));
  return clampHealingBrushBBox(
    [
      Math.min(...xs) - pad,
      Math.min(...ys) - pad,
      Math.max(...xs) + pad,
      Math.max(...ys) + pad,
    ],
    width,
    height,
  );
}

export function createHealingBrushMaskPngDataUrl(input: {
  width: number;
  height: number;
  stroke: [number, number][];
  brushSize: number;
  dirtyBBox: HealingBrushBBox;
  clipPolygon?: [number, number][];
}): string | null {
  const { width, height, stroke, brushSize, dirtyBBox, clipPolygon } = input;
  if (width <= 0 || height <= 0 || stroke.length === 0 || brushSize <= 0) return null;
  const bbox = clampHealingBrushBBox(dirtyBBox, width, height);
  if (!bbox) return null;
  const [x1, y1, x2, y2] = bbox;
  const maskWidth = Math.max(1, x2 - x1);
  const maskHeight = Math.max(1, y2 - y1);

  const canvas = document.createElement("canvas");
  canvas.width = maskWidth;
  canvas.height = maskHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, maskWidth, maskHeight);
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = Math.max(1, brushSize);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(stroke[0][0] - x1, stroke[0][1] - y1);
  for (const [x, y] of stroke.slice(1)) {
    ctx.lineTo(x - x1, y - y1);
  }
  if (stroke.length === 1) {
    const [x, y] = stroke[0];
    ctx.lineTo(x - x1 + 0.01, y - y1 + 0.01);
  }
  ctx.stroke();
  if (clipPolygon && clipPolygon.length >= 3) {
    ctx.globalCompositeOperation = "destination-in";
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.moveTo(clipPolygon[0][0] - x1, clipPolygon[0][1] - y1);
    for (const [x, y] of clipPolygon.slice(1)) {
      ctx.lineTo(x - x1, y - y1);
    }
    ctx.closePath();
    ctx.fill();
    ctx.globalCompositeOperation = "source-over";
  }

  return canvas.toDataURL("image/png");
}

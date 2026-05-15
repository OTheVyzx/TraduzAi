type BitmapPreviewLayerKey = "brush" | "mask" | "inpaint";

interface BitmapStrokePreviewConfig {
  layerKey: BitmapPreviewLayerKey;
  baseImage?: HTMLImageElement | null;
  originalImage?: HTMLImageElement | null;
  width: number;
  height: number;
  stroke: [number, number][];
  brushSize: number;
  color: string;
  opacity: number;
  hardness?: number;
  erase: boolean;
  clipPolygon?: [number, number][];
}

export interface BitmapStrokePreviewResult {
  layerKey: BitmapPreviewLayerKey;
  beforeDataUrl: string;
  afterDataUrl: string;
}

function clamp01(value: number) {
  if (!Number.isFinite(value)) return 1;
  return Math.min(1, Math.max(0, value));
}

export function strokePassesForHardness({
  brushSize,
  opacity,
  hardness = 1,
}: {
  brushSize: number;
  opacity: number;
  hardness?: number;
}) {
  const width = Math.max(1, Math.round(brushSize));
  const alpha = clamp01(opacity);
  const hard = clamp01(hardness);
  if (hard >= 0.95 || width <= 2) return [{ width, alpha }];

  const softness = 1 - hard;
  const outerWidth = Math.max(width + 1, Math.round(width * (1 + softness * 0.9)));
  const middleWidth = Math.max(width + 1, Math.round(width * (1 + softness * 0.45)));

  return [
    { width: outerWidth, alpha: Math.round(alpha * softness * 0.18 * 1000) / 1000 },
    { width: middleWidth, alpha: Math.round(alpha * softness * 0.32 * 1000) / 1000 },
    { width, alpha },
  ].filter((pass) => pass.alpha > 0.005);
}

function drawStrokePath(ctx: CanvasRenderingContext2D, stroke: [number, number][], brushSize: number) {
  if (stroke.length === 0) return;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.lineWidth = Math.max(1, brushSize);
  ctx.beginPath();
  ctx.moveTo(stroke[0][0], stroke[0][1]);
  for (const [x, y] of stroke.slice(1)) {
    ctx.lineTo(x, y);
  }
  if (stroke.length === 1) {
    ctx.lineTo(stroke[0][0] + 0.01, stroke[0][1] + 0.01);
  }
  ctx.stroke();
}

function drawStrokePasses(
  ctx: CanvasRenderingContext2D,
  stroke: [number, number][],
  brushSize: number,
  opacity: number,
  hardness?: number,
) {
  for (const pass of strokePassesForHardness({ brushSize, opacity, hardness })) {
    ctx.globalAlpha = pass.alpha;
    drawStrokePath(ctx, stroke, pass.width);
  }
}

function clipToPolygon(ctx: CanvasRenderingContext2D, polygon?: [number, number][]) {
  if (!polygon || polygon.length < 3) return;
  ctx.beginPath();
  ctx.moveTo(polygon[0][0], polygon[0][1]);
  for (const [x, y] of polygon.slice(1)) {
    ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.clip();
}

function canvasWithBase(width: number, height: number, image?: HTMLImageElement | null) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  if (image?.naturalWidth && image?.naturalHeight) {
    ctx.drawImage(image, 0, 0, width, height);
  }
  return { canvas, ctx };
}

export function createBitmapStrokePreview(config: BitmapStrokePreviewConfig): BitmapStrokePreviewResult | null {
  const { layerKey, width, height, stroke, brushSize, erase } = config;
  const originalImage = config.originalImage ?? null;
  const base = canvasWithBase(width, height, config.baseImage);
  if (!base) return null;
  const { canvas, ctx } = base;
  const beforeDataUrl = canvas.toDataURL("image/png");

  ctx.save();
  clipToPolygon(ctx, config.clipPolygon);
  if (layerKey === "inpaint") {
    if (!originalImage) {
      ctx.restore();
      return null;
    }
    drawStrokePasses(ctx, stroke, brushSize, 1, config.hardness);
    ctx.globalCompositeOperation = "source-in";
    ctx.drawImage(originalImage, 0, 0, width, height);
  } else if (erase) {
    ctx.globalCompositeOperation = "destination-out";
    ctx.strokeStyle = "rgba(0,0,0,1)";
    drawStrokePasses(ctx, stroke, brushSize, 1, config.hardness);
  } else if (layerKey === "mask") {
    ctx.strokeStyle = "#ffffff";
    drawStrokePasses(ctx, stroke, brushSize, 1, config.hardness);
  } else {
    ctx.strokeStyle = config.color || "#000000";
    drawStrokePasses(ctx, stroke, brushSize, config.opacity, config.hardness);
  }
  ctx.restore();

  return {
    layerKey,
    beforeDataUrl,
    afterDataUrl: canvas.toDataURL("image/png"),
  };
}

export function createBitmapStrokePreviewOnCanvas(
  canvas: HTMLCanvasElement,
  config: Omit<BitmapStrokePreviewConfig, "baseImage" | "originalImage" | "width" | "height">,
): BitmapStrokePreviewResult | null {
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  const beforeDataUrl = canvas.toDataURL("image/png");

  ctx.save();
  clipToPolygon(ctx, config.clipPolygon);
  if (config.erase) {
    ctx.globalCompositeOperation = "destination-out";
    ctx.strokeStyle = "rgba(0,0,0,1)";
    drawStrokePasses(ctx, config.stroke, config.brushSize, 1, config.hardness);
  } else if (config.layerKey === "mask") {
    ctx.globalCompositeOperation = "source-over";
    ctx.strokeStyle = "#ffffff";
    drawStrokePasses(ctx, config.stroke, config.brushSize, 1, config.hardness);
  } else {
    ctx.globalCompositeOperation = "source-over";
    ctx.strokeStyle = config.color || "#000000";
    drawStrokePasses(ctx, config.stroke, config.brushSize, config.opacity, config.hardness);
  }
  ctx.restore();

  return {
    layerKey: config.layerKey,
    beforeDataUrl,
    afterDataUrl: canvas.toDataURL("image/png"),
  };
}

export function encodeDataUrl(value: string) {
  return new TextEncoder().encode(value);
}

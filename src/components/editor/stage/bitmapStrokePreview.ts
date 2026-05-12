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
  erase: boolean;
}

export interface BitmapStrokePreviewResult {
  layerKey: BitmapPreviewLayerKey;
  beforeDataUrl: string;
  afterDataUrl: string;
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
  const base = canvasWithBase(width, height, config.baseImage);
  if (!base) return null;
  const { canvas, ctx } = base;
  const beforeDataUrl = canvas.toDataURL("image/png");

  if (layerKey === "inpaint") {
    if (!config.originalImage) return null;
    ctx.save();
    drawStrokePath(ctx, stroke, brushSize);
    ctx.globalCompositeOperation = "source-in";
    ctx.drawImage(config.originalImage, 0, 0, width, height);
    ctx.restore();
  } else if (erase) {
    ctx.save();
    ctx.globalCompositeOperation = "destination-out";
    ctx.strokeStyle = "rgba(0,0,0,1)";
    drawStrokePath(ctx, stroke, brushSize);
    ctx.restore();
  } else if (layerKey === "mask") {
    ctx.save();
    ctx.strokeStyle = "#ffffff";
    drawStrokePath(ctx, stroke, brushSize);
    ctx.restore();
  } else {
    ctx.save();
    ctx.globalAlpha = Math.max(0, Math.min(1, config.opacity));
    ctx.strokeStyle = config.color || "#000000";
    drawStrokePath(ctx, stroke, brushSize);
    ctx.restore();
  }

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

  if (config.erase) {
    ctx.save();
    ctx.globalCompositeOperation = "destination-out";
    ctx.strokeStyle = "rgba(0,0,0,1)";
    drawStrokePath(ctx, config.stroke, config.brushSize);
    ctx.restore();
  } else if (config.layerKey === "mask") {
    ctx.save();
    ctx.globalCompositeOperation = "source-over";
    ctx.strokeStyle = "#ffffff";
    drawStrokePath(ctx, config.stroke, config.brushSize);
    ctx.restore();
  } else {
    ctx.save();
    ctx.globalCompositeOperation = "source-over";
    ctx.globalAlpha = Math.max(0, Math.min(1, config.opacity));
    ctx.strokeStyle = config.color || "#000000";
    drawStrokePath(ctx, config.stroke, config.brushSize);
    ctx.restore();
  }

  return {
    layerKey: config.layerKey,
    beforeDataUrl,
    afterDataUrl: canvas.toDataURL("image/png"),
  };
}

export function encodeDataUrl(value: string) {
  return new TextEncoder().encode(value);
}

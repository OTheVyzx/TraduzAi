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
  clipMaskImage?: CanvasImageSource;
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
  if (stroke.length === 1) {
    ctx.lineTo(stroke[0][0] + 0.01, stroke[0][1] + 0.01);
  } else if (stroke.length === 2) {
    ctx.lineTo(stroke[1][0], stroke[1][1]);
  } else {
    for (let index = 1; index < stroke.length - 1; index += 1) {
      const current = stroke[index];
      const next = stroke[index + 1];
      ctx.quadraticCurveTo(current[0], current[1], (current[0] + next[0]) / 2, (current[1] + next[1]) / 2);
    }
    const last = stroke[stroke.length - 1];
    ctx.lineTo(last[0], last[1]);
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

  if (config.clipMaskImage) {
    const overlay = document.createElement("canvas");
    overlay.width = canvas.width;
    overlay.height = canvas.height;
    const overlayContext = overlay.getContext("2d");
    if (!overlayContext) return null;

    overlayContext.save();
    overlayContext.globalCompositeOperation = "source-over";
    if (config.erase || config.layerKey === "mask") {
      overlayContext.strokeStyle = "#ffffff";
      drawStrokePasses(overlayContext, config.stroke, config.brushSize, 1, config.hardness);
    } else {
      overlayContext.strokeStyle = config.color || "#000000";
      drawStrokePasses(overlayContext, config.stroke, config.brushSize, config.opacity, config.hardness);
    }
    overlayContext.globalAlpha = 1;
    overlayContext.globalCompositeOperation = "destination-in";
    overlayContext.drawImage(config.clipMaskImage, 0, 0, canvas.width, canvas.height);
    overlayContext.restore();

    ctx.save();
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = config.erase ? "destination-out" : "source-over";
    ctx.drawImage(overlay, 0, 0);
    ctx.restore();
    return {
      layerKey: config.layerKey,
      beforeDataUrl,
      afterDataUrl: canvas.toDataURL("image/png"),
    };
  }

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


export type StudioBitmapOperationCommit = {
  commandId: string;
  pageIndex?: number;
  layerKey: "brush" | "recovery" | "inpaint";
  name: string;
  pngData: string;
  baselineDataUrl: string | null;
  bbox: [number, number, number, number];
  blendMode: "normal" | "destination-out";
};

export interface BitmapStrokeOperationPatch {
  pngData: string;
  bbox: [number, number, number, number];
}

export function createMaskedImageOperationPatch(config: {
  image: CanvasImageSource;
  maskImage: CanvasImageSource;
  width: number;
  height: number;
  bbox: [number, number, number, number];
}): BitmapStrokeOperationPatch | null {
  const [rawX1, rawY1, rawX2, rawY2] = config.bbox;
  const x1 = Math.max(0, Math.floor(Math.min(rawX1, rawX2)));
  const y1 = Math.max(0, Math.floor(Math.min(rawY1, rawY2)));
  const x2 = Math.min(config.width, Math.ceil(Math.max(rawX1, rawX2)));
  const y2 = Math.min(config.height, Math.ceil(Math.max(rawY1, rawY2)));
  if (x2 <= x1 || y2 <= y1) return null;

  const canvas = document.createElement("canvas");
  canvas.width = x2 - x1;
  canvas.height = y2 - y1;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.drawImage(config.image, x1, y1, x2 - x1, y2 - y1, 0, 0, x2 - x1, y2 - y1);

  const maskCanvas = document.createElement("canvas");
  maskCanvas.width = x2 - x1;
  maskCanvas.height = y2 - y1;
  const maskContext = maskCanvas.getContext("2d");
  if (!maskContext) return null;
  maskContext.drawImage(config.maskImage, 0, 0, x2 - x1, y2 - y1);
  const maskPixels = maskContext.getImageData(0, 0, x2 - x1, y2 - y1);
  for (let index = 0; index < maskPixels.data.length; index += 4) {
    const alpha = Math.round((maskPixels.data[index] + maskPixels.data[index + 1] + maskPixels.data[index + 2]) / 3);
    maskPixels.data[index] = 255;
    maskPixels.data[index + 1] = 255;
    maskPixels.data[index + 2] = 255;
    maskPixels.data[index + 3] = alpha;
  }
  maskContext.putImageData(maskPixels, 0, 0);
  ctx.globalCompositeOperation = "destination-in";
  ctx.drawImage(maskCanvas, 0, 0);
  return { pngData: canvas.toDataURL("image/png"), bbox: [x1, y1, x2, y2] };
}

/* Encode only the modified rectangle rather than another copy of the whole page. */
export function createBitmapStrokeOperationPatch(config: {
  width: number;
  height: number;
  stroke: [number, number][];
  brushSize: number;
  color: string;
  opacity: number;
  hardness?: number;
  erase: boolean;
  bbox: [number, number, number, number];
  clipPolygon?: [number, number][];
  clipMaskImage?: CanvasImageSource;
}): BitmapStrokeOperationPatch | null {
  const [rawX1, rawY1, rawX2, rawY2] = config.bbox;
  const x1 = Math.max(0, Math.floor(Math.min(rawX1, rawX2)));
  const y1 = Math.max(0, Math.floor(Math.min(rawY1, rawY2)));
  const x2 = Math.min(config.width, Math.ceil(Math.max(rawX1, rawX2)));
  const y2 = Math.min(config.height, Math.ceil(Math.max(rawY1, rawY2)));
  if (config.stroke.length === 0 || x2 <= x1 || y2 <= y1) return null;

  const canvas = document.createElement("canvas");
  canvas.width = x2 - x1;
  canvas.height = y2 - y1;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  ctx.save();
  ctx.translate(-x1, -y1);
  clipToPolygon(ctx, config.clipPolygon);
  ctx.globalCompositeOperation = "source-over";
  ctx.strokeStyle = config.erase ? "#ffffff" : (config.color || "#000000");
  drawStrokePasses(
    ctx,
    config.stroke,
    config.brushSize,
    config.erase ? 1 : config.opacity,
    config.hardness,
  );
  ctx.restore();

  if (config.clipMaskImage) {
    ctx.save();
    ctx.globalCompositeOperation = "destination-in";
    ctx.drawImage(
      config.clipMaskImage,
      x1,
      y1,
      x2 - x1,
      y2 - y1,
      0,
      0,
      x2 - x1,
      y2 - y1,
    );
    ctx.restore();
  }

  return {
    pngData: canvas.toDataURL("image/png"),
    bbox: [x1, y1, x2, y2],
  };
}

export function encodeDataUrl(value: string) {
  return new TextEncoder().encode(value);
}

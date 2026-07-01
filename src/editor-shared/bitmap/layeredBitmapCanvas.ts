export type LayeredBitmapKey = "base" | "inpaint" | "brush" | "mask" | "rendered";

export interface Canvas2DLike {
  globalAlpha: number;
  globalCompositeOperation: string;
  lineCap: CanvasLineCap;
  lineJoin: CanvasLineJoin;
  lineWidth: number;
  strokeStyle: string | CanvasGradient | CanvasPattern;
  clearRect(x: number, y: number, width: number, height: number): void;
  drawImage(image: CanvasImageSource, dx: number, dy: number, width: number, height: number): void;
  beginPath(): void;
  moveTo(x: number, y: number): void;
  lineTo(x: number, y: number): void;
  stroke(): void;
  save(): void;
  restore(): void;
}

export interface CanvasLike {
  width: number;
  height: number;
  getContext(type: "2d"): Canvas2DLike | null;
  toDataURL(type?: string): string;
}

export interface LayeredBitmapLayer {
  key: LayeredBitmapKey;
  canvas: CanvasLike;
  visible: boolean;
  opacity: number;
  order: number;
}

export interface LayeredBitmapCanvasOptions {
  width: number;
  height: number;
  createCanvas: (width: number, height: number) => CanvasLike;
}

export interface DrawBitmapStrokeOptions {
  layerKey: LayeredBitmapKey;
  stroke: [number, number][];
  brushSize: number;
  color?: string;
  opacity?: number;
  hardness?: number;
  erase?: boolean;
}

export function bitmapStrokePasses(input: { brushSize: number; opacity?: number; hardness?: number }) {
  const width = Math.max(1, Math.round(input.brushSize));
  const opacity = clamp01(input.opacity ?? 1);
  const hardness = clamp01(input.hardness ?? 1);
  if (hardness >= 0.95 || width <= 2) return [{ width, alpha: opacity }];

  const softness = 1 - hardness;
  return [
    { width: Math.max(width + 1, Math.round(width * (1 + softness * 0.9))), alpha: roundAlpha(opacity * softness * 0.18) },
    { width: Math.max(width + 1, Math.round(width * (1 + softness * 0.45))), alpha: roundAlpha(opacity * softness * 0.32) },
    { width, alpha: opacity },
  ].filter((pass) => pass.alpha > 0.005);
}

function clamp01(value: number) {
  if (!Number.isFinite(value)) return 1;
  return Math.min(1, Math.max(0, value));
}

function roundAlpha(value: number) {
  return Math.round(value * 1000) / 1000;
}

function defaultOrder(key: LayeredBitmapKey) {
  switch (key) {
    case "base": return 0;
    case "inpaint": return 10;
    case "brush": return 20;
    case "mask": return 30;
    case "rendered": return 40;
  }
}

export class LayeredBitmapCanvas {
  private readonly width: number;
  private readonly height: number;
  private readonly createCanvas: (width: number, height: number) => CanvasLike;
  private readonly layers = new Map<LayeredBitmapKey, LayeredBitmapLayer>();

  constructor(options: LayeredBitmapCanvasOptions) {
    this.width = Math.max(1, Math.round(options.width));
    this.height = Math.max(1, Math.round(options.height));
    this.createCanvas = options.createCanvas;
  }

  get size() {
    return { width: this.width, height: this.height };
  }

  ensureLayer(key: LayeredBitmapKey, config: Partial<Pick<LayeredBitmapLayer, "visible" | "opacity" | "order">> = {}) {
    const existing = this.layers.get(key);
    if (existing) {
      existing.visible = config.visible ?? existing.visible;
      existing.opacity = config.opacity ?? existing.opacity;
      existing.order = config.order ?? existing.order;
      return existing;
    }
    const layer: LayeredBitmapLayer = {
      key,
      canvas: this.createCanvas(this.width, this.height),
      visible: config.visible ?? true,
      opacity: config.opacity ?? 1,
      order: config.order ?? defaultOrder(key),
    };
    this.layers.set(key, layer);
    return layer;
  }

  clearLayer(key: LayeredBitmapKey) {
    const layer = this.ensureLayer(key);
    const ctx = layer.canvas.getContext("2d");
    ctx?.clearRect(0, 0, this.width, this.height);
  }

  drawImageToLayer(key: LayeredBitmapKey, image: CanvasImageSource) {
    const layer = this.ensureLayer(key);
    const ctx = layer.canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, this.width, this.height);
    ctx.drawImage(image, 0, 0, this.width, this.height);
  }

  drawStroke(options: DrawBitmapStrokeOptions) {
    if (options.stroke.length === 0) return null;
    const layer = this.ensureLayer(options.layerKey);
    const ctx = layer.canvas.getContext("2d");
    if (!ctx) return null;

    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.globalCompositeOperation = options.erase ? "destination-out" : "source-over";
    ctx.strokeStyle = options.layerKey === "mask" ? "#ffffff" : options.color ?? "#000000";

    for (const pass of bitmapStrokePasses({
      brushSize: options.brushSize,
      opacity: options.erase ? 1 : options.opacity,
      hardness: options.hardness,
    })) {
      ctx.globalAlpha = pass.alpha;
      ctx.lineWidth = pass.width;
      drawStrokePath(ctx, options.stroke);
    }
    ctx.restore();
    return layer.canvas.toDataURL("image/png");
  }

  compositeVisibleLayers() {
    const output = this.createCanvas(this.width, this.height);
    const ctx = output.getContext("2d");
    if (!ctx) return output;
    for (const layer of this.orderedLayers().filter((item) => item.visible)) {
      ctx.save();
      ctx.globalAlpha = clamp01(layer.opacity);
      ctx.drawImage(layer.canvas as unknown as CanvasImageSource, 0, 0, this.width, this.height);
      ctx.restore();
    }
    return output;
  }

  exportLayerDataUrl(key: LayeredBitmapKey) {
    return this.layers.get(key)?.canvas.toDataURL("image/png") ?? null;
  }

  getLayerCanvas(key: LayeredBitmapKey) {
    return this.layers.get(key)?.canvas ?? null;
  }

  orderedLayers() {
    return [...this.layers.values()].sort((a, b) => a.order - b.order);
  }
}

function drawStrokePath(ctx: Canvas2DLike, stroke: [number, number][]) {
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

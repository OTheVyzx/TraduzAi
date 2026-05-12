export type InpaintCacheStrokePreviewPatch = {
  id: string;
  image: HTMLImageElement;
  x: number;
  y: number;
  width: number;
  height: number;
};

type PaintSource = HTMLImageElement | HTMLCanvasElement;

function sourceHasPixels(source: PaintSource | null | undefined) {
  if (!source) return false;
  if (source instanceof HTMLImageElement) {
    return Boolean(source.naturalWidth && source.naturalHeight);
  }
  return Boolean(source.width && source.height);
}

function drawInpaintBrushMask(
  ctx: CanvasRenderingContext2D,
  stroke: [number, number][],
  brushSize: number,
) {
  if (stroke.length === 0) return;
  ctx.strokeStyle = "#ffffff";
  ctx.fillStyle = "#ffffff";
  ctx.lineWidth = Math.max(1, brushSize);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  const [firstX, firstY] = stroke[0];
  ctx.moveTo(firstX, firstY);
  if (stroke.length === 1) {
    ctx.arc(firstX, firstY, Math.max(1, brushSize / 2), 0, Math.PI * 2);
    ctx.fill();
    return;
  }
  for (let i = 1; i < stroke.length; i++) {
    const [x, y] = stroke[i];
    ctx.lineTo(x, y);
  }
  ctx.stroke();
}

export function applyInpaintCacheStrokeToCanvas(
  canvas: HTMLCanvasElement,
  inpaintCache: PaintSource,
  stroke: [number, number][],
  brushSize: number,
): string | null {
  const width = canvas.width;
  const height = canvas.height;
  if (!width || !height || stroke.length === 0 || !sourceHasPixels(inpaintCache)) {
    return null;
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  const maskCanvas = document.createElement("canvas");
  maskCanvas.width = width;
  maskCanvas.height = height;
  const maskCtx = maskCanvas.getContext("2d");
  if (!maskCtx) return null;
  drawInpaintBrushMask(maskCtx, stroke, brushSize);

  const patchCanvas = document.createElement("canvas");
  patchCanvas.width = width;
  patchCanvas.height = height;
  const patchCtx = patchCanvas.getContext("2d");
  if (!patchCtx) return null;
  patchCtx.drawImage(inpaintCache, 0, 0, width, height);
  patchCtx.globalCompositeOperation = "destination-in";
  patchCtx.drawImage(maskCanvas, 0, 0);

  ctx.drawImage(patchCanvas, 0, 0);
  return canvas.toDataURL("image/png");
}

export function createInpaintCacheStrokePreviewPatch(
  inpaintCache: PaintSource,
  stroke: [number, number][],
  brushSize: number,
  dirtyBbox: [number, number, number, number],
): Promise<InpaintCacheStrokePreviewPatch | null> {
  const [x1, y1, x2, y2] = dirtyBbox;
  const width = Math.max(1, x2 - x1);
  const height = Math.max(1, y2 - y1);
  if (stroke.length === 0 || !sourceHasPixels(inpaintCache)) {
    return Promise.resolve(null);
  }

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return Promise.resolve(null);

  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(inpaintCache, x1, y1, width, height, 0, 0, width, height);

  const maskCanvas = document.createElement("canvas");
  maskCanvas.width = width;
  maskCanvas.height = height;
  const maskCtx = maskCanvas.getContext("2d");
  if (!maskCtx) return Promise.resolve(null);

  drawInpaintBrushMask(
    maskCtx,
    stroke.map(([x, y]) => [x - x1, y - y1] as [number, number]),
    brushSize,
  );

  ctx.globalCompositeOperation = "destination-in";
  ctx.drawImage(maskCanvas, 0, 0);
  ctx.globalCompositeOperation = "source-over";

  return new Promise((resolve) => {
    const image = new Image();
    image.onload = () => {
      resolve({
        id: crypto.randomUUID(),
        image,
        x: x1,
        y: y1,
        width,
        height,
      });
    };
    image.onerror = () => resolve(null);
    image.src = canvas.toDataURL("image/png");
  });
}

export function composeRecoveryImage(
  rendered: HTMLImageElement,
  original: HTMLImageElement,
  recoveryMask: HTMLImageElement,
): string | null {
  const width = rendered.naturalWidth || rendered.width;
  const height = rendered.naturalHeight || rendered.height;
  if (!width || !height) return null;

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  ctx.drawImage(rendered, 0, 0, width, height);

  const maskCanvas = document.createElement("canvas");
  maskCanvas.width = width;
  maskCanvas.height = height;
  const maskCtx = maskCanvas.getContext("2d");
  if (!maskCtx) return canvas.toDataURL("image/png");
  maskCtx.drawImage(recoveryMask, 0, 0, width, height);

  const originalCanvas = document.createElement("canvas");
  originalCanvas.width = width;
  originalCanvas.height = height;
  const originalCtx = originalCanvas.getContext("2d");
  if (!originalCtx) return canvas.toDataURL("image/png");
  originalCtx.drawImage(original, 0, 0, width, height);

  const output = ctx.getImageData(0, 0, width, height);
  const originalData = originalCtx.getImageData(0, 0, width, height).data;
  const maskData = maskCtx.getImageData(0, 0, width, height).data;
  for (let i = 0; i < output.data.length; i += 4) {
    if (maskData[i + 3] > 0 || maskData[i] > 0 || maskData[i + 1] > 0 || maskData[i + 2] > 0) {
      output.data[i] = originalData[i];
      output.data[i + 1] = originalData[i + 1];
      output.data[i + 2] = originalData[i + 2];
      output.data[i + 3] = originalData[i + 3];
    }
  }
  ctx.putImageData(output, 0, 0);
  return canvas.toDataURL("image/png");
}

export type RecoveryStrokePreviewPatch = {
  id: string;
  image: HTMLImageElement;
  x: number;
  y: number;
  width: number;
  height: number;
};

function drawRecoveryStrokeMask(
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

function drawPolygonMask(
  ctx: CanvasRenderingContext2D,
  polygon?: [number, number][],
) {
  if (!polygon || polygon.length < 3) return false;
  ctx.fillStyle = "#ffffff";
  ctx.beginPath();
  ctx.moveTo(polygon[0][0], polygon[0][1]);
  for (const [x, y] of polygon.slice(1)) {
    ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fill();
  return true;
}

function clipMaskToPolygon(
  ctx: CanvasRenderingContext2D,
  polygon?: [number, number][],
) {
  if (!polygon || polygon.length < 3) return;
  ctx.save();
  ctx.globalCompositeOperation = "destination-in";
  drawPolygonMask(ctx, polygon);
  ctx.restore();
}

export function applyRecoveryStrokeToCanvas(
  canvas: HTMLCanvasElement,
  original: HTMLImageElement,
  stroke: [number, number][],
  brushSize: number,
  clipPolygon?: [number, number][],
): string | null {
  const width = canvas.width;
  const height = canvas.height;
  if (!width || !height || stroke.length === 0 || !original.naturalWidth || !original.naturalHeight) {
    return null;
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  const maskCanvas = document.createElement("canvas");
  maskCanvas.width = width;
  maskCanvas.height = height;
  const maskCtx = maskCanvas.getContext("2d");
  if (!maskCtx) return null;
  drawRecoveryStrokeMask(maskCtx, stroke, brushSize);
  clipMaskToPolygon(maskCtx, clipPolygon);

  const patchCanvas = document.createElement("canvas");
  patchCanvas.width = width;
  patchCanvas.height = height;
  const patchCtx = patchCanvas.getContext("2d");
  if (!patchCtx) return null;
  patchCtx.drawImage(original, 0, 0, width, height);
  patchCtx.globalCompositeOperation = "destination-in";
  patchCtx.drawImage(maskCanvas, 0, 0);

  ctx.drawImage(patchCanvas, 0, 0);
  return canvas.toDataURL("image/png");
}

export function createRecoveryStrokePreviewPatch(
  original: HTMLImageElement,
  stroke: [number, number][],
  brushSize: number,
  dirtyBbox: [number, number, number, number],
  clipPolygon?: [number, number][],
): Promise<RecoveryStrokePreviewPatch | null> {
  const [x1, y1, x2, y2] = dirtyBbox;
  const width = Math.max(1, x2 - x1);
  const height = Math.max(1, y2 - y1);
  if (stroke.length === 0 || !original.naturalWidth || !original.naturalHeight) {
    return Promise.resolve(null);
  }

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return Promise.resolve(null);

  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(original, x1, y1, width, height, 0, 0, width, height);

  const maskCanvas = document.createElement("canvas");
  maskCanvas.width = width;
  maskCanvas.height = height;
  const maskCtx = maskCanvas.getContext("2d");
  if (!maskCtx) return Promise.resolve(null);

  drawRecoveryStrokeMask(
    maskCtx,
    stroke.map(([x, y]) => [x - x1, y - y1] as [number, number]),
    brushSize,
  );
  clipMaskToPolygon(
    maskCtx,
    clipPolygon?.map(([x, y]) => [x - x1, y - y1] as [number, number]),
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

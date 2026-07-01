import { useEffect, useState } from "react";
import { Image as KonvaImage } from "react-konva";
import { LayeredBitmapCanvas } from "../../../editor-shared/bitmap/layeredBitmapCanvas";

type Props = {
  brushImage: HTMLImageElement | null;
  brushOpacity?: number;
  maskImage: HTMLImageElement | null;
  maskOpacity?: number;
  width: number;
  height: number;
};

function isRenderableImage(image: HTMLImageElement | null, width: number, height: number) {
  if (!image || image.naturalWidth === 0 || image.naturalHeight === 0) return false;
  return !(image.naturalWidth <= 1 && image.naturalHeight <= 1 && (width > 1 || height > 1));
}

function warnDimensionMismatch(layer: "brush" | "mask", image: HTMLImageElement, width: number, height: number) {
  if (image.naturalWidth === width && image.naturalHeight === height) return;
  console.warn(
    `[EditorBitmapOverlay] Mismatch dimensoes ${layer}: ${image.naturalWidth}x${image.naturalHeight} vs ${width}x${height}`,
  );
}

function createTintedAlphaCanvas(image: HTMLImageElement, color: string, opacity: number) {
  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  ctx.drawImage(image, 0, 0);
  const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const r = parseInt(color.slice(1, 3), 16);
  const g = parseInt(color.slice(3, 5), 16);
  const b = parseInt(color.slice(5, 7), 16);
  const alphaScale = Math.min(1, Math.max(0, opacity));

  for (let i = 0; i < imgData.data.length; i += 4) {
    const luma = 0.299 * imgData.data[i] + 0.587 * imgData.data[i + 1] + 0.114 * imgData.data[i + 2];
    imgData.data[i] = r;
    imgData.data[i + 1] = g;
    imgData.data[i + 2] = b;
    imgData.data[i + 3] = Math.round(luma * alphaScale);
  }

  ctx.putImageData(imgData, 0, 0);
  return canvas;
}

export function EditorBitmapOverlay({
  brushImage,
  brushOpacity = 1,
  maskImage,
  maskOpacity = 0.65,
  width,
  height,
}: Props) {
  const [compositedCanvas, setCompositedCanvas] = useState<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const hasBrush = isRenderableImage(brushImage, width, height);
    const hasMask = isRenderableImage(maskImage, width, height);
    if (!hasBrush && !hasMask) {
      setCompositedCanvas(null);
      return;
    }

    const layered = new LayeredBitmapCanvas({
      width,
      height,
      createCanvas: (canvasWidth, canvasHeight) => {
        const canvas = document.createElement("canvas");
        canvas.width = canvasWidth;
        canvas.height = canvasHeight;
        return canvas;
      },
    });

    if (hasMask && maskImage) {
      warnDimensionMismatch("mask", maskImage, width, height);
      const tintedMask = createTintedAlphaCanvas(maskImage, "#6C5CE7", maskOpacity);
      if (tintedMask) layered.drawImageToLayer("mask", tintedMask);
      layered.ensureLayer("mask", { order: 20 });
    }

    if (hasBrush && brushImage) {
      warnDimensionMismatch("brush", brushImage, width, height);
      layered.drawImageToLayer("brush", brushImage);
      layered.ensureLayer("brush", { opacity: brushOpacity, order: 30 });
    }

    setCompositedCanvas(layered.compositeVisibleLayers() as HTMLCanvasElement);
  }, [brushImage, brushOpacity, height, maskImage, maskOpacity, width]);

  if (!compositedCanvas) return null;

  return (
    <KonvaImage
      image={compositedCanvas}
      x={0}
      y={0}
      width={width}
      height={height}
      listening={false}
    />
  );
}

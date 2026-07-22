import { createCanvas, loadImage } from "canvas";
import "../../../node_modules/konva/lib/canvas-backend.js";
import { initializeCanvas, readPsd } from "ag-psd";
import { describe, expect, it } from "vitest";
import type { StudioTextLayer } from "../project/studioProject";
import {
  psdTextSpecFromLayer,
  rasterizePsdTextLayer,
  writePsdRasterLayers,
} from "../export/psd";

initializeCanvas(createCanvas as unknown as (width: number, height: number) => HTMLCanvasElement);

function solid(width: number, height: number, rgba: [number, number, number, number]) {
  const pixels = new Uint8Array(width * height * 4);
  for (let index = 0; index < pixels.length; index += 4) pixels.set(rgba, index);
  return pixels;
}

function expectPixelParity(left: Uint8Array | Uint8ClampedArray, right: Uint8Array | Uint8ClampedArray) {
  expect(right.length).toBe(left.length);
  let maxDifference = 0;
  for (let index = 0; index < left.length; index += 1) {
    maxDifference = Math.max(maxDifference, Math.abs(left[index] - right[index]));
  }
  expect(maxDifference).toBeLessThanOrEqual(1);
}

describe("paridade de exportacao canvas/PNG/PSD", () => {
  it("mantem o texto visivel no composite e os pixels iguais nos tres formatos", async () => {
    const width = 96;
    const height = 48;
    const layer: StudioTextLayer = {
      id: "text-parity",
      kind: "text",
      original: "PARIDADE",
      translated: "PARIDADE",
      traduzido: "PARIDADE",
      bbox: [8, 6, 88, 42],
      style: { fonte: "Arial", tamanho: 20, cor: "#111111", alinhamento: "center" },
      estilo: { fonte: "Arial", tamanho: 20, cor: "#111111", alinhamento: "center" },
      visible: true,
      locked: false,
      order: 1,
    };
    const textSpec = psdTextSpecFromLayer(layer.translated, layer);
    const textPixels = await rasterizePsdTextLayer(layer.translated, layer, textSpec);
    expect(textPixels.some((value, index) => index % 4 === 3 && value > 0)).toBe(true);

    const layers = [
      { name: "Original", pixels: solid(width, height, [255, 255, 255, 255]) },
      {
        name: "Texto",
        pixels: textPixels,
        left: textSpec.x,
        top: textSpec.y,
        right: textSpec.x + textSpec.width,
        bottom: textSpec.y + textSpec.height,
        textSpec,
      },
    ];
    const canvas = createCanvas(width, height);
    const context = canvas.getContext("2d");
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, width, height);
    const textCanvas = createCanvas(textSpec.width, textSpec.height);
    const textContext = textCanvas.getContext("2d");
    const textImageData = textContext.createImageData(textSpec.width, textSpec.height);
    textImageData.data.set(textPixels);
    textContext.putImageData(textImageData, 0, 0);
    context.drawImage(textCanvas, textSpec.x, textSpec.y);
    const canvasPixels = context.getImageData(0, 0, width, height).data;

    const pngImage = await loadImage(canvas.toBuffer("image/png"));
    const pngCanvas = createCanvas(width, height);
    pngCanvas.getContext("2d").drawImage(pngImage, 0, 0);
    const pngPixels = pngCanvas.getContext("2d").getImageData(0, 0, width, height).data;

    const psd = readPsd(writePsdRasterLayers(width, height, layers), {
      skipLayerImageData: true,
      skipThumbnail: true,
    });
    const psdPixels = psd.imageData?.data ?? psd.canvas?.getContext("2d")?.getImageData(0, 0, width, height).data;
    expect(psdPixels).toBeDefined();

    expectPixelParity(canvasPixels, pngPixels);
    expectPixelParity(canvasPixels, Uint8Array.from(psdPixels!));
  });
});

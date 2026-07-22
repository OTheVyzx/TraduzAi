import { loadImageSource } from "../../../../src/lib/imageSource";
import { rasterizeLassoSelectionToCanvas } from "../../../../src/lib/lassoSelection";
import { isolateFluxVariantPixels } from "../../ai/fluxContract";
import {
  studioSelectionEffectiveBbox,
  type StudioSelection,
} from "../selection/selectionModel";

export interface PreparedFluxCrop {
  bbox: [number, number, number, number];
  width: number;
  height: number;
  pageWidth: number;
  pageHeight: number;
  sourcePngData: string;
  maskPngData: string;
  maskAlpha: Uint8Array;
}

export function fluxCropBbox(
  selection: StudioSelection,
  options: { contextPadding?: number; alignment?: number } = {},
): [number, number, number, number] {
  const padding = Math.max(0, Math.round(options.contextPadding ?? 64));
  const alignment = Math.max(1, Math.round(options.alignment ?? 16));
  const [x1, y1, x2, y2] = studioSelectionEffectiveBbox(selection);
  const left = Math.max(0, Math.floor((x1 - padding) / alignment) * alignment);
  const top = Math.max(0, Math.floor((y1 - padding) / alignment) * alignment);
  const right = Math.min(selection.width, Math.ceil((x2 + padding) / alignment) * alignment);
  const bottom = Math.min(selection.height, Math.ceil((y2 + padding) / alignment) * alignment);
  if (right <= left || bottom <= top) throw new Error("A seleção FLUX não possui área utilizável");
  if (right - left > 4096 || bottom - top > 4096) {
    throw new Error("A seleção FLUX excede o limite local de 4096 px por dimensão");
  }
  return [left, top, right, bottom];
}

async function loadHtmlImage(source: string) {
  const loaded = await loadImageSource(source, "image/png");
  return new Promise<{ image: HTMLImageElement; revoke?: () => void }>((resolve, reject) => {
    const image = new Image();
    image.decoding = "async";
    image.onload = () => resolve({ image, revoke: loaded.revoke });
    image.onerror = () => {
      loaded.revoke?.();
      reject(new Error("Não foi possível carregar a imagem para o FLUX"));
    };
    image.src = loaded.src;
  });
}

function canvas2d(width: number, height: number) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Canvas 2D indisponível para preparar o FLUX");
  return { canvas, context };
}

export async function prepareFluxCrop(
  sourcePath: string,
  selection: StudioSelection,
): Promise<PreparedFluxCrop> {
  const loaded = await loadHtmlImage(sourcePath);
  try {
    const pageWidth = loaded.image.naturalWidth;
    const pageHeight = loaded.image.naturalHeight;
    if (selection.width !== pageWidth || selection.height !== pageHeight) {
      throw new Error("A seleção e a camada-alvo possuem dimensões diferentes");
    }
    const bbox = fluxCropBbox(selection);
    const [left, top, right, bottom] = bbox;
    const width = right - left;
    const height = bottom - top;
    const source = canvas2d(width, height);
    source.context.drawImage(loaded.image, left, top, width, height, 0, 0, width, height);

    const selectionMask = rasterizeLassoSelectionToCanvas(selection);
    if (!selectionMask) throw new Error("Não foi possível rasterizar a seleção FLUX");
    const providerMask = canvas2d(width, height);
    providerMask.context.fillStyle = "#000000";
    providerMask.context.fillRect(0, 0, width, height);
    providerMask.context.drawImage(selectionMask, left, top, width, height, 0, 0, width, height);

    const alphaMask = canvas2d(width, height);
    alphaMask.context.clearRect(0, 0, width, height);
    alphaMask.context.drawImage(selectionMask, left, top, width, height, 0, 0, width, height);
    const rgba = alphaMask.context.getImageData(0, 0, width, height).data;
    const maskAlpha = new Uint8Array(width * height);
    for (let sourceIndex = 3, target = 0; sourceIndex < rgba.length; sourceIndex += 4, target += 1) {
      maskAlpha[target] = rgba[sourceIndex];
    }

    return {
      bbox,
      width,
      height,
      pageWidth,
      pageHeight,
      sourcePngData: source.canvas.toDataURL("image/png"),
      maskPngData: providerMask.canvas.toDataURL("image/png"),
      maskAlpha,
    };
  } finally {
    loaded.revoke?.();
  }
}

export async function materializeFluxVariantOverlay(
  prepared: PreparedFluxCrop,
  variantSource: string,
) {
  const loaded = await loadHtmlImage(variantSource);
  try {
    const crop = canvas2d(prepared.width, prepared.height);
    crop.context.drawImage(loaded.image, 0, 0, prepared.width, prepared.height);
    const image = crop.context.getImageData(0, 0, prepared.width, prepared.height);
    const isolated = isolateFluxVariantPixels(new Uint8Array(image.data), prepared.maskAlpha);
    image.data.set(isolated);
    crop.context.putImageData(image, 0, 0);

    const output = canvas2d(prepared.pageWidth, prepared.pageHeight);
    output.context.clearRect(0, 0, prepared.pageWidth, prepared.pageHeight);
    output.context.drawImage(crop.canvas, prepared.bbox[0], prepared.bbox[1]);
    return output.canvas.toDataURL("image/png");
  } finally {
    loaded.revoke?.();
  }
}

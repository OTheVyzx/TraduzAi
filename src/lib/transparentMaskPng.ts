const TRANSPARENT_PNG_FALLBACK =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4nGNgAAIAAAUAAXpeqz8AAAAASUVORK5CYII=";

function assertValidDimensions(width: number, height: number) {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new Error("Dimensões da página indisponíveis para limpar a máscara");
  }
}

function bytesToBase64(bytes: Uint8Array) {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

export async function transparentMaskPngDataUrl(width: number, height: number): Promise<string> {
  assertValidDimensions(width, height);
  if (typeof document !== "undefined") {
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    return canvas.toDataURL("image/png");
  }
  if (typeof OffscreenCanvas !== "undefined") {
    const canvas = new OffscreenCanvas(width, height);
    const blob = await canvas.convertToBlob({ type: "image/png" });
    return `data:image/png;base64,${bytesToBase64(new Uint8Array(await blob.arrayBuffer()))}`;
  }
  return TRANSPARENT_PNG_FALLBACK;
}

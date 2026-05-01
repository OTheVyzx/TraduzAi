import { writePsd, Psd } from 'ag-psd';
import { readFile } from '@tauri-apps/plugin-fs';

export async function createPsdFromLayers(
  originalPath: string,
  inpaintPath: string,
  textCanvas: HTMLCanvasElement,
  width: number,
  height: number
): Promise<Uint8Array> {
  const originalBytes = await readFile(originalPath);
  const inpaintBytes = await readFile(inpaintPath);

  // Helper to convert Uint8Array to HTMLImageElement
  const bytesToImage = (bytes: Uint8Array): Promise<HTMLImageElement> => {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => resolve(img);
      const blob = new Blob([bytes as any], { type: 'image/jpeg' });
      img.src = URL.createObjectURL(blob);
    });
  };

  const originalImg = await bytesToImage(originalBytes);
  const inpaintImg = await bytesToImage(inpaintBytes);

  // Create canvases for the background layers
  const originalCanvas = document.createElement('canvas');
  originalCanvas.width = width;
  originalCanvas.height = height;
  originalCanvas.getContext('2d')?.drawImage(originalImg, 0, 0);

  const inpaintCanvas = document.createElement('canvas');
  inpaintCanvas.width = width;
  inpaintCanvas.height = height;
  inpaintCanvas.getContext('2d')?.drawImage(inpaintImg, 0, 0);

  const psd: Psd = {
    width,
    height,
    children: [
      {
        name: 'Original',
        canvas: originalCanvas,
      },
      {
        name: 'Inpaint (Limpeza)',
        canvas: inpaintCanvas,
      },
      {
        name: 'Tradução',
        canvas: textCanvas,
      },
    ],
  };

  const psdBuffer = writePsd(psd);
  return new Uint8Array(psdBuffer);
}

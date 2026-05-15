import { useEffect, useState } from "react";
import { Image as KonvaImage } from "react-konva";

type Props = {
  image: HTMLImageElement | null;
  width: number;
  height: number;
  /** Cor hex para o overlay, ex: "#48B0FF" para brush, "#6C5CE7" para máscara */
  color: string;
  /** Opacidade do overlay (0–1). Default: 0.65 */
  opacity?: number;
};

/**
 * Converte imagem grayscale em overlay RGBA colorido.
 * A luminância de cada pixel vira o canal alpha; o canal RGB é substituído pela cor fornecida.
 * Renderiza como camada Konva sem interação (listening=false).
 */
export function EditorBitmapOverlay({ image, width, height, color, opacity = 0.65 }: Props) {
  const [coloredImage, setColoredImage] = useState<HTMLImageElement | null>(null);

  useEffect(() => {
    if (!image || image.naturalWidth === 0) {
      setColoredImage(null);
      return;
    }

    const isPlaceholder =
      image.naturalWidth <= 1 && image.naturalHeight <= 1 && (width > 1 || height > 1);
    if (isPlaceholder) {
      setColoredImage(null);
      return;
    }

    if (image.naturalWidth !== width || image.naturalHeight !== height) {
      console.warn(
        `[EditorBitmapOverlay] Mismatch dimensões: ${image.naturalWidth}x${image.naturalHeight} vs ${width}x${height}`,
      );
    }

    const canvas = document.createElement("canvas");
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(image, 0, 0);
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    const r = parseInt(color.slice(1, 3), 16);
    const g = parseInt(color.slice(3, 5), 16);
    const b = parseInt(color.slice(5, 7), 16);

    for (let i = 0; i < imgData.data.length; i += 4) {
      const luma =
        0.299 * imgData.data[i] + 0.587 * imgData.data[i + 1] + 0.114 * imgData.data[i + 2];
      imgData.data[i] = r;
      imgData.data[i + 1] = g;
      imgData.data[i + 2] = b;
      imgData.data[i + 3] = Math.round(luma * opacity);
    }

    ctx.putImageData(imgData, 0, 0);

    let cancelled = false;
    const coloredImg = new Image();
    coloredImg.onload = () => {
      if (!cancelled) setColoredImage(coloredImg);
    };
    coloredImg.src = canvas.toDataURL("image/png");

    return () => {
      cancelled = true;
    };
  }, [image, color, opacity, width, height]);

  if (!coloredImage) return null;

  return (
    <KonvaImage
      image={coloredImage}
      x={0}
      y={0}
      width={width}
      height={height}
      listening={false}
    />
  );
}

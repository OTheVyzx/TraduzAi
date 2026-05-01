import { Image as KonvaImage } from "react-konva";

export function EditorStageBackground({
  image,
  width,
  height,
}: {
  image: HTMLImageElement | null;
  width: number;
  height: number;
}) {
  if (!image) return null;
  return <KonvaImage image={image} width={width} height={height} listening={false} />;
}

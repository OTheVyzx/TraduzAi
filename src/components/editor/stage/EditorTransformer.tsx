import { useEffect, useRef } from "react";
import Konva from "konva";
import { Transformer } from "react-konva";

export function EditorTransformer({ selectedNodeName }: { selectedNodeName: string | null }) {
  const transformerRef = useRef<Konva.Transformer>(null);

  useEffect(() => {
    const transformer = transformerRef.current;
    if (!transformer) return;
    const stage = transformer.getStage();
    const node = selectedNodeName ? stage?.findOne(`.${selectedNodeName}`) : null;
    transformer.nodes(node ? [node] : []);
    transformer.getLayer()?.batchDraw();
  }, [selectedNodeName]);

  return (
    <Transformer
      ref={transformerRef}
      rotateEnabled
      rotateAnchorOffset={34}
      rotationSnaps={[-180, -90, -45, -30, -15, 0, 15, 30, 45, 90, 180]}
      rotationSnapTolerance={5}
      enabledAnchors={["top-left", "top-right", "bottom-left", "bottom-right", "middle-left", "middle-right", "top-center", "bottom-center"]}
      boundBoxFunc={(_, nextBox) => ({
        ...nextBox,
        width: Math.max(20, nextBox.width),
        height: Math.max(20, nextBox.height),
      })}
      borderStroke="rgba(108, 92, 231, 0.95)"
      borderStrokeWidth={2}
      anchorFill="#f8fbff"
      anchorStroke="rgba(108, 92, 231, 0.95)"
      anchorStrokeWidth={2}
      anchorSize={16}
      anchorCornerRadius={3}
    />
  );
}

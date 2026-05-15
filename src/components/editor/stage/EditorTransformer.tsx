import { useEffect, useRef } from "react";
import Konva from "konva";
import type { Box } from "konva/lib/shapes/Transformer";
import { Transformer } from "react-konva";
import type { TextEntry } from "../../../lib/stores/appStore";
import { snapRectToGuides, type SnapGuide } from "./snapGuides";
import { clampTextTransformBox, type PageSize } from "./transformConstraints";

export function EditorTransformer({
  selectedNodeName,
  pageSize,
  selectedLayerId,
  snapLayers = [],
  disabled = false,
  onSnapGuidesChange,
}: {
  selectedNodeName: string | null;
  pageSize: PageSize | null;
  selectedLayerId?: string | null;
  snapLayers?: TextEntry[];
  disabled?: boolean;
  onSnapGuidesChange?: (guides: SnapGuide[]) => void;
}) {
  const transformerRef = useRef<Konva.Transformer>(null);

  useEffect(() => {
    const transformer = transformerRef.current;
    if (!transformer) return;
    const stage = transformer.getStage();
    const node = selectedNodeName && !disabled ? stage?.findOne(`.${selectedNodeName}`) : null;
    transformer.nodes(node ? [node] : []);
    transformer.getLayer()?.batchDraw();
    if (!node) onSnapGuidesChange?.([]);
  }, [disabled, onSnapGuidesChange, selectedNodeName]);

  const boundTextBox = (_: Box, nextBox: Box) => {
    const clamped = clampTextTransformBox(nextBox, pageSize);
    if (!pageSize || disabled || !selectedLayerId) {
      onSnapGuidesChange?.([]);
      return clamped as Box;
    }
    const snapped = snapRectToGuides(clamped, {
      pageSize,
      layers: snapLayers,
      excludeLayerId: selectedLayerId,
    });
    onSnapGuidesChange?.(snapped.guides);
    return snapped.rect as Box;
  };

  return (
    <Transformer
      ref={transformerRef}
      visible={!disabled}
      flipEnabled={false}
      keepRatio={false}
      centeredScaling={false}
      ignoreStroke
      rotateEnabled
      rotateAnchorOffset={34}
      rotationSnaps={[-180, -90, -45, -30, -15, 0, 15, 30, 45, 90, 180]}
      rotationSnapTolerance={5}
      enabledAnchors={["top-left", "top-right", "bottom-left", "bottom-right", "middle-left", "middle-right", "top-center", "bottom-center"]}
      boundBoxFunc={boundTextBox}
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

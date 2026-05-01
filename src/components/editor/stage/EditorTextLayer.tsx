import { useEffect, useRef, useState } from "react";
import Konva from "konva";
import { Group, Rect, Text } from "react-konva";
import type { TextEntry } from "../../../lib/stores/appStore";
import { bboxToRect, rectToBbox, sameBbox } from "./coordinateUtils";
import {
  fontFamilyFromStyle,
  fontStyleFromStyle,
  styleForLayer,
  textForLayer,
} from "./textLayerStyleUtils";

export function EditorTextLayer({
  entry,
  selected,
  hovered,
  showGuides,
  interactive,
  onSelect,
  onHover,
  onCommitBbox,
}: {
  entry: TextEntry;
  selected: boolean;
  hovered: boolean;
  showGuides: boolean;
  interactive: boolean;
  onSelect: () => void;
  onHover: (hovered: boolean) => void;
  onCommitBbox: (before: TextEntry["bbox"], after: TextEntry["bbox"]) => void;
}) {
  const textRef = useRef<Konva.Text>(null);
  const [, bumpFontVersion] = useState(0);
  if (entry.visible === false) return null;

  const bbox = entry.layout_bbox ?? entry.bbox;
  const rect = bboxToRect(bbox);
  const style = styleForLayer(entry);
  const showFrame = selected || hovered || showGuides;
  const text = textForLayer(entry);
  const nodeName = `text-layer-${entry.id.replace(/[^a-zA-Z0-9_-]/g, "_")}`;
  const fontFamily = fontFamilyFromStyle(style);

  useEffect(() => {
    const fonts = document.fonts;
    if (!fonts) return;
    let cancelled = false;
    void fonts.load(`${Math.max(8, style.tamanho)}px "${fontFamily}"`).then(() => {
      if (cancelled) return;
      bumpFontVersion((version) => version + 1);
      textRef.current?.getLayer()?.batchDraw();
    });
    return () => {
      cancelled = true;
    };
  }, [fontFamily, style.tamanho]);

  const commitGroupBbox = (node: Konva.Group) => {
    const next = rectToBbox({
      x: node.x(),
      y: node.y(),
      width: rect.width * node.scaleX(),
      height: rect.height * node.scaleY(),
    });
    node.scaleX(1);
    node.scaleY(1);
    if (!sameBbox(bbox, next)) onCommitBbox(bbox, next);
  };

  return (
    <Group
      name={nodeName}
      x={rect.x}
      y={rect.y}
      width={rect.width}
      height={rect.height}
      draggable={interactive && !entry.locked}
      onClick={(event) => {
        event.cancelBubble = true;
        onSelect();
      }}
      onTap={(event) => {
        event.cancelBubble = true;
        onSelect();
      }}
      onMouseEnter={() => onHover(true)}
      onMouseLeave={() => onHover(false)}
      onDragStart={(event) => {
        event.cancelBubble = true;
        onSelect();
      }}
      onDragEnd={(event) => {
        event.cancelBubble = true;
        commitGroupBbox(event.target as Konva.Group);
      }}
      onTransformEnd={(event) => {
        event.cancelBubble = true;
        commitGroupBbox(event.target as Konva.Group);
      }}
    >
      <Rect
        width={rect.width}
        height={rect.height}
        cornerRadius={12}
        fill={selected ? "rgba(124, 92, 255, 0.08)" : hovered ? "rgba(124, 92, 255, 0.05)" : "rgba(0,0,0,0)"}
        stroke={
          showFrame
            ? selected
              ? "rgba(124, 92, 255, 0.95)"
              : hovered
                ? "rgba(124, 92, 255, 0.55)"
                : "rgba(255, 255, 255, 0.18)"
            : "rgba(0,0,0,0)"
        }
        dash={selected || hovered ? undefined : [6, 5]}
        strokeWidth={selected ? 2 : 1}
      />
      <Text
        ref={textRef}
        x={8}
        y={6}
        width={Math.max(1, rect.width - 16)}
        height={Math.max(1, rect.height - 12)}
        text={text}
        align={style.alinhamento}
        verticalAlign="middle"
        wrap="word"
        ellipsis={false}
        fontSize={Math.max(8, style.tamanho)}
        fontFamily={fontFamily}
        fontStyle={fontStyleFromStyle(style)}
        fill={style.cor || "#000000"}
        stroke={style.contorno || "#000000"}
        strokeWidth={Math.max(0, style.contorno_px || 0)}
        shadowEnabled={!!style.sombra}
        shadowColor={style.sombra_cor || "#000000"}
        shadowOffsetX={style.sombra_offset?.[0] ?? 0}
        shadowOffsetY={style.sombra_offset?.[1] ?? 0}
        shadowBlur={style.sombra ? 2 : 0}
        listening={false}
      />
    </Group>
  );
}

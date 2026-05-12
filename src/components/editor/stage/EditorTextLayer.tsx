import { useEffect, useMemo, useRef, useState } from "react";
import Konva from "konva";
import { Group, Rect, Text } from "react-konva";
import type { TextEntry } from "../../../lib/stores/appStore";
import type { TextTransformSnapshot } from "../../../lib/stores/editorStore";
import { bboxToRect, rectToBbox, sameBbox } from "./coordinateUtils";
import { EDITOR_TEXT_LINE_HEIGHT, fitEditorTextFontSize } from "./textFit";
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
  draftRotation,
  onSelect,
  onHover,
  onCommitTransform,
}: {
  entry: TextEntry;
  selected: boolean;
  hovered: boolean;
  showGuides: boolean;
  interactive: boolean;
  draftRotation?: number | null;
  onSelect: () => void;
  onHover: (hovered: boolean) => void;
  onCommitTransform: (before: TextTransformSnapshot, after: TextTransformSnapshot) => void;
}) {
  // IMPORTANTE: nenhum return condicional entre hooks — todos os hooks antes
  // do early-return de visibility, senão React quebra com "Rendered fewer
  // hooks than expected" e o canvas pisca.
  const textRef = useRef<Konva.Text>(null);
  const [fontVersion, bumpFontVersion] = useState(0);

  const bbox = entry.layout_bbox ?? entry.bbox;
  const rect = bboxToRect(bbox);
  const style = styleForLayer(entry);
  const showFrame = selected || hovered || showGuides;
  const text = textForLayer(entry);
  const rotation = normalizeRotationDegrees(style.rotacao);
  const displayedRotation = normalizeRotationDegrees(draftRotation ?? rotation);
  const nodeName = `text-layer-${entry.id.replace(/[^a-zA-Z0-9_-]/g, "_")}`;
  const fontFamily = fontFamilyFromStyle(style);
  const fontStyle = fontStyleFromStyle(style);
  const textBoxWidth = Math.max(1, rect.width - 16);
  const textBoxHeight = Math.max(1, rect.height - 12);
  const fontSize = useMemo(
    () =>
      fitEditorTextFontSize({
        text,
        fontFamily,
        fontStyle,
        maxFontSize: Math.max(8, style.tamanho),
        maxWidth: textBoxWidth,
        maxHeight: textBoxHeight,
      }),
    [fontFamily, fontStyle, fontVersion, style.tamanho, text, textBoxHeight, textBoxWidth],
  );
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

  // Early-return só DEPOIS de todos os hooks
  if (entry.visible === false) return null;

  const commitGroupTransform = (node: Konva.Group) => {
    const nextWidth = Math.max(1, rect.width * node.scaleX());
    const nextHeight = Math.max(1, rect.height * node.scaleY());
    const next = rectToBbox({
      x: node.x() - nextWidth / 2,
      y: node.y() - nextHeight / 2,
      width: nextWidth,
      height: nextHeight,
    });
    const nextRotation = normalizeRotationDegrees(node.rotation());
    node.scaleX(1);
    node.scaleY(1);
    const before = { bbox, rotacao: rotation };
    const after = { bbox: next, rotacao: nextRotation };
    if (!sameBbox(before.bbox, after.bbox) || Math.abs(before.rotacao - after.rotacao) >= 0.01) {
      onCommitTransform(before, after);
    }
  };

  return (
    <Group
      name={nodeName}
      x={rect.x + rect.width / 2}
      y={rect.y + rect.height / 2}
      offsetX={rect.width / 2}
      offsetY={rect.height / 2}
      width={rect.width}
      height={rect.height}
      rotation={displayedRotation}
      listening={interactive}
      draggable={interactive && !entry.locked}
      onClick={(event) => {
        if (!interactive) return;
        event.cancelBubble = true;
        onSelect();
      }}
      onTap={(event) => {
        if (!interactive) return;
        event.cancelBubble = true;
        onSelect();
      }}
      onMouseEnter={() => {
        if (interactive) onHover(true);
      }}
      onMouseLeave={() => {
        if (interactive) onHover(false);
      }}
      onDragStart={(event) => {
        if (!interactive) return;
        event.cancelBubble = true;
        onSelect();
      }}
      onDragEnd={(event) => {
        if (!interactive) return;
        event.cancelBubble = true;
        commitGroupTransform(event.target as Konva.Group);
      }}
      onTransformEnd={(event) => {
        if (!interactive) return;
        event.cancelBubble = true;
        commitGroupTransform(event.target as Konva.Group);
      }}
    >
      <Rect
        width={rect.width}
        height={rect.height}
        cornerRadius={12}
        fill={selected ? "rgba(108, 92, 231, 0.08)" : hovered ? "rgba(108, 92, 231, 0.05)" : "rgba(0,0,0,0)"}
        stroke={
          showFrame
            ? selected
              ? "rgba(108, 92, 231, 0.95)"
              : hovered
                ? "rgba(108, 92, 231, 0.55)"
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
        width={textBoxWidth}
        height={textBoxHeight}
        text={text}
        align={style.alinhamento}
        verticalAlign="middle"
        wrap="word"
        ellipsis={false}
        fontSize={fontSize}
        fontFamily={fontFamily}
        fontStyle={fontStyle}
        lineHeight={EDITOR_TEXT_LINE_HEIGHT}
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

function normalizeRotationDegrees(value: unknown) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  let normalized = numeric % 360;
  if (normalized > 180) normalized -= 360;
  if (normalized <= -180) normalized += 360;
  if (Math.abs(normalized) < 0.01) return 0;
  return Math.round(normalized * 100) / 100;
}

import Konva from "konva";
import { Text } from "react-konva";
import type { Ref } from "react";
import type { TextConfig } from "konva/lib/shapes/Text";
import type { TextLayerStyle } from "../../../lib/stores/appStore";
import {
  colorToRgba,
  fontStyleForResolvedTextStyle,
  resolveEditorTextStyle,
  type ResolvedEditorTextStyle,
  type ResolvedTextFill,
} from "../../../lib/editorTextStyleResolver";

export interface StyledKonvaTextConfig {
  x: number;
  y: number;
  width: number;
  height: number;
  text: string;
  align: TextLayerStyle["alinhamento"];
  fontSize: number;
  fontFamily: string;
  fontStyle: string;
  lineHeight: number;
  style: TextLayerStyle;
  listening?: boolean;
}

type StyledTextNodeConfig = TextConfig & { key: string };

function colorWithOpacity(color: string, opacity: number) {
  const [r, g, b, a] = colorToRgba(color, opacity);
  return `rgba(${r}, ${g}, ${b}, ${a / 255})`;
}

function commonTextConfig(config: StyledKonvaTextConfig, resolved: ResolvedEditorTextStyle): TextConfig {
  const professional = resolved.source === "studio";
  const fontSize = config.fontSize;
  return {
    x: config.x,
    y: config.y - (professional ? resolved.typography.baselineShift : 0),
    width: config.width,
    height: config.height,
    text: config.text,
    align: professional ? resolved.typography.align : config.align,
    verticalAlign: "middle",
    wrap: "word",
    ellipsis: false,
    fontSize,
    fontFamily: professional ? resolved.typography.fontFamily : config.fontFamily,
    fontStyle: professional ? fontStyleForResolvedTextStyle(resolved) : config.fontStyle,
    lineHeight: professional ? resolved.typography.lineHeight : config.lineHeight,
    letterSpacing: professional ? (fontSize * resolved.typography.tracking) / 1000 : 0,
    scaleX: professional ? resolved.typography.horizontalScale / 100 : 1,
    scaleY: professional ? resolved.typography.verticalScale / 100 : 1,
    listening: config.listening ?? false,
  };
}

function gradientPoints(width: number, height: number, angle: number) {
  const radians = (angle * Math.PI) / 180;
  const vectorX = Math.cos(radians);
  const vectorY = Math.sin(radians);
  const radius = Math.abs(vectorX) * width / 2 + Math.abs(vectorY) * height / 2;
  const centerX = width / 2;
  const centerY = height / 2;
  return {
    start: { x: centerX - vectorX * radius, y: centerY - vectorY * radius },
    end: { x: centerX + vectorX * radius, y: centerY + vectorY * radius },
  };
}

function fillConfig(fill: ResolvedTextFill, width: number, height: number): TextConfig {
  if (fill.type === "solid") {
    return {
      fill: fill.color,
      fillPriority: "color",
      opacity: fill.opacity,
    };
  }
  const points = gradientPoints(width, height, fill.angle);
  return {
    fill: fill.stops[0]?.color ?? "#000000",
    fillPriority: "linear-gradient",
    fillLinearGradientStartPoint: points.start,
    fillLinearGradientEndPoint: points.end,
    fillLinearGradientColorStops: fill.stops.flatMap((stop) => [
      stop.offset,
      colorWithOpacity(stop.color, stop.opacity),
    ]),
    opacity: fill.opacity,
  };
}

export function buildStyledKonvaTextNodeConfigs(config: StyledKonvaTextConfig): StyledTextNodeConfig[] {
  const resolved = resolveEditorTextStyle(config.style);
  const base = commonTextConfig(config, resolved);
  const nodes: StyledTextNodeConfig[] = [];

  resolved.effects.dropShadows.forEach((shadow, index) => {
    nodes.push({
      ...base,
      key: index === 0 ? "shadow" : `shadow-${index}`,
      x: Number(base.x ?? config.x) + shadow.offsetX,
      y: Number(base.y ?? config.y) + shadow.offsetY,
      fill: shadow.color,
      stroke: shadow.color,
      strokeWidth: 0,
      shadowEnabled: shadow.blur > 0,
      shadowColor: shadow.color,
      shadowBlur: shadow.blur,
      shadowOpacity: shadow.opacity,
      opacity: shadow.opacity,
    });
  });

  if (resolved.effects.outerGlow) {
    const glow = resolved.effects.outerGlow;
    nodes.push({
      ...base,
      key: "glow",
      fill: glow.color,
      stroke: glow.color,
      strokeWidth: Math.max(0, glow.spread * 2),
      shadowEnabled: true,
      shadowColor: glow.color,
      shadowBlur: Math.max(1, glow.blur),
      shadowOpacity: glow.opacity,
      opacity: glow.opacity,
    });
  }

  [...resolved.strokes].reverse().forEach((stroke, reverseIndex) => {
    const index = resolved.strokes.length - reverseIndex - 1;
    nodes.push({
      ...base,
      key: `stroke-${index}`,
      fillEnabled: false,
      stroke: stroke.color,
      strokeWidth: stroke.position === "outside" ? stroke.width * 2 : stroke.width,
      strokeEnabled: true,
      opacity: stroke.opacity,
      lineJoin: "round",
    });
  });

  if (resolved.fills.length === 0) {
    nodes.push({ ...base, key: "main", fillEnabled: false, strokeEnabled: false });
  } else {
    resolved.fills.forEach((fill, index) => {
      nodes.push({
        ...base,
        key: index === resolved.fills.length - 1 ? "main" : `fill-${index}`,
        ...fillConfig(fill, config.width, config.height),
        strokeEnabled: false,
      });
    });
  }

  return nodes;
}

export function addStyledKonvaTextNodes(group: Konva.Group, config: StyledKonvaTextConfig) {
  for (const { key: _key, ...nodeConfig } of buildStyledKonvaTextNodeConfigs(config)) {
    group.add(new Konva.Text(nodeConfig));
  }
}

export function createStyledKonvaTextGroup(config: StyledKonvaTextConfig) {
  const group = new Konva.Group();
  addStyledKonvaTextNodes(group, config);
  return group;
}

export function KonvaStyledText({
  textRef,
  ...config
}: StyledKonvaTextConfig & { textRef?: Ref<Konva.Text> }) {
  return (
    <>
      {buildStyledKonvaTextNodeConfigs(config).map(({ key, ...nodeConfig }) => (
        <Text key={key} ref={key === "main" ? textRef : undefined} {...nodeConfig} />
      ))}
    </>
  );
}

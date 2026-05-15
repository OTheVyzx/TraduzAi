import Konva from "konva";
import { Text } from "react-konva";
import type { Ref } from "react";
import type { TextConfig } from "konva/lib/shapes/Text";
import type { TextLayerStyle } from "../../../lib/stores/appStore";

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

type StyledTextNodeConfig = TextConfig & { key: "shadow" | "glow" | "main" };

function gradientStops(colors: string[] | undefined) {
  const normalized = (colors ?? []).map((color) => color.trim()).filter(Boolean);
  if (normalized.length < 2) return null;
  return normalized.flatMap((color, index) => [
    normalized.length === 1 ? 0 : index / (normalized.length - 1),
    color,
  ]);
}

function commonTextConfig(config: StyledKonvaTextConfig): TextConfig {
  return {
    x: config.x,
    y: config.y,
    width: config.width,
    height: config.height,
    text: config.text,
    align: config.align,
    verticalAlign: "middle",
    wrap: "word",
    ellipsis: false,
    fontSize: config.fontSize,
    fontFamily: config.fontFamily,
    fontStyle: config.fontStyle,
    lineHeight: config.lineHeight,
    listening: config.listening ?? false,
  };
}

function mainFillConfig(style: TextLayerStyle, height: number): TextConfig {
  const stops = gradientStops(style.cor_gradiente);
  if (!stops) {
    return {
      fill: style.cor || "#000000",
      fillPriority: "color",
    };
  }
  return {
    fill: style.cor || String(stops[1] ?? "#000000"),
    fillPriority: "linear-gradient",
    fillLinearGradientStartPoint: { x: 0, y: 0 },
    fillLinearGradientEndPoint: { x: 0, y: Math.max(1, height) },
    fillLinearGradientColorStops: stops,
  };
}

export function buildStyledKonvaTextNodeConfigs(config: StyledKonvaTextConfig): StyledTextNodeConfig[] {
  const style = config.style;
  const base = commonTextConfig(config);
  const strokeWidth = Math.max(0, style.contorno_px || 0);
  const nodes: StyledTextNodeConfig[] = [];

  if (style.sombra) {
    const [offsetX, offsetY] = style.sombra_offset ?? [0, 0];
    const color = style.sombra_cor || "#000000";
    nodes.push({
      ...base,
      key: "shadow",
      x: config.x + offsetX,
      y: config.y + offsetY,
      fill: color,
      stroke: color,
      strokeWidth,
      opacity: 0.9,
    });
  }

  if (style.glow && (style.glow_px ?? 0) > 0) {
    const color = style.glow_cor || "#ffffff";
    const glowWidth = Math.max(strokeWidth, Math.round((style.glow_px ?? 0) * 0.7));
    nodes.push({
      ...base,
      key: "glow",
      fill: color,
      stroke: color,
      strokeWidth: glowWidth,
      shadowEnabled: true,
      shadowColor: color,
      shadowBlur: Math.max(1, style.glow_px ?? 0),
      shadowOpacity: 0.85,
      opacity: 0.65,
    });
  }

  nodes.push({
    ...base,
    key: "main",
    ...mainFillConfig(style, config.height),
    stroke: style.contorno || undefined,
    strokeWidth,
  });

  return nodes;
}

export function addStyledKonvaTextNodes(group: Konva.Group, config: StyledKonvaTextConfig) {
  for (const { key: _key, ...nodeConfig } of buildStyledKonvaTextNodeConfigs(config)) {
    group.add(new Konva.Text(nodeConfig));
  }
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

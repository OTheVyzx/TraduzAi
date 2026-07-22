import { describe, expect, it } from "vitest";
import { readPsd } from "ag-psd";
import { buildStyledKonvaTextNodeConfigs } from "../../../../src/components/editor/stage/konvaTextStyleRenderer";
import { writePsdRasterLayers } from "../../export/psd";
import { STUDIO_TEXT_STYLE_VERSION } from "../styleModel";
import { resolveStudioTextStyle } from "../styleResolver";

const professionalStyle = {
  fonte: "Legacy.ttf",
  tamanho: 24,
  cor: "#000000",
  cor_gradiente: [],
  contorno: "",
  contorno_px: 0,
  glow: false,
  glow_cor: "",
  glow_px: 0,
  sombra: false,
  sombra_cor: "",
  sombra_offset: [0, 0] as [number, number],
  bold: false,
  italico: false,
  rotacao: 0,
  alinhamento: "center" as const,
  studio_style: {
    version: STUDIO_TEXT_STYLE_VERSION,
    typography: {
      fontFamily: "Comic Neue",
      fontSize: 36,
      fontWeight: 700,
      fontStyle: "italic" as const,
      align: "left" as const,
      lineHeight: 1.25,
      tracking: 60,
      horizontalScale: 105,
      verticalScale: 100,
      baselineShift: 1,
      vertical: false,
    },
    fills: [
      { type: "solid" as const, color: "#f5f5f5", opacity: 1 },
      {
        type: "linear-gradient" as const,
        angle: 30,
        opacity: 0.85,
        stops: [
          { offset: 0, color: "#ffcc00", opacity: 1 },
          { offset: 1, color: "#ff3300", opacity: 1 },
        ],
      },
    ],
    strokes: [
      { color: "#ffffff", width: 3, opacity: 1, position: "outside" as const },
      { color: "#111111", width: 9, opacity: 0.9, position: "outside" as const },
    ],
    effects: {
      dropShadows: [
        { color: "#000000", opacity: 0.7, blur: 5, offsetX: 4, offsetY: 6 },
      ],
      outerGlow: { color: "#33ccff", opacity: 0.65, blur: 8, spread: 1 },
    },
  },
};

function transparent(width: number, height: number) {
  return new Uint8Array(width * height * 4);
}

describe("paridade do motor de estilos", () => {
  it("usa o mesmo estilo resolvido para os passes Konva e os efeitos editáveis do PSD", () => {
    const resolved = resolveStudioTextStyle(professionalStyle);
    const nodes = buildStyledKonvaTextNodeConfigs({
      x: 0,
      y: 0,
      width: 160,
      height: 80,
      text: "ESTILO",
      align: "center",
      fontSize: 36,
      fontFamily: "Comic Neue",
      fontStyle: "bold italic",
      lineHeight: 1.25,
      style: professionalStyle,
    });

    expect(nodes.map((node) => node.key)).toEqual([
      "shadow",
      "glow",
      "stroke-1",
      "stroke-0",
      "fill-0",
      "main",
    ]);
    expect(nodes.filter((node) => node.key.startsWith("stroke-")).map((node) => node.strokeWidth)).toEqual([18, 6]);
    expect(nodes.at(-1)).toMatchObject({
      key: "main",
      fillPriority: "linear-gradient",
      opacity: 0.85,
    });

    const bytes = writePsdRasterLayers(200, 120, [
      {
        name: "Texto profissional",
        pixels: transparent(160, 80),
        left: 10,
        top: 10,
        right: 170,
        bottom: 90,
        textSpec: {
          text: "ESTILO",
          x: 10,
          y: 10,
          width: 160,
          height: 80,
          fontName: resolved.typography.fontFamily,
          fontSize: resolved.typography.fontSize,
          color: [245, 245, 245, 255],
          vertical: resolved.typography.vertical,
          justification: resolved.typography.align,
          resolvedStyle: resolved,
        },
      },
    ]);
    const psd = readPsd(bytes, {
      skipLayerImageData: true,
      skipCompositeImageData: true,
      skipThumbnail: true,
    });
    const layer = psd.children?.find((child) => child.name === "Texto profissional");

    expect(layer?.text?.style).toMatchObject({
      fontSize: 36,
      fauxBold: true,
      fauxItalic: true,
      tracking: 60,
      leading: 45,
    });
    expect(layer?.effects?.stroke?.map((stroke) => stroke.size?.value)).toEqual([3, 9]);
    expect(layer?.effects?.gradientOverlay).toHaveLength(1);
    expect(layer?.effects?.dropShadow).toHaveLength(1);
    expect(layer?.effects?.outerGlow?.size?.value).toBe(8);
  });
});

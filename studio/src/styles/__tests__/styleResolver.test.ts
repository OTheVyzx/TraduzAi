import { describe, expect, it } from "vitest";
import { STUDIO_TEXT_STYLE_VERSION } from "../styleModel";
import { resolveStudioTextStyle } from "../styleResolver";

describe("resolveStudioTextStyle", () => {
  it("converte o estilo legado sem perder gradiente, contorno e efeitos", () => {
    const resolved = resolveStudioTextStyle({
      fonte: "CC Wild Words.ttf",
      tamanho: 32,
      cor: "#112233",
      cor_gradiente: ["#ffcc00", "#ff3300"],
      contorno: "#220000",
      contorno_px: 4,
      bold: true,
      italico: false,
      alinhamento: "center",
      sombra: true,
      sombra_cor: "#000000",
      sombra_offset: [3, 5],
      glow: true,
      glow_cor: "#ffffff",
      glow_px: 6,
    });

    expect(resolved.source).toBe("legacy");
    expect(resolved.typography).toMatchObject({
      fontFamily: "CC Wild Words.ttf",
      fontSize: 32,
      fontWeight: 700,
      fontStyle: "normal",
      align: "center",
    });
    expect(resolved.fills).toEqual([
      {
        type: "linear-gradient",
        angle: 90,
        opacity: 1,
        stops: [
          { offset: 0, color: "#ffcc00", opacity: 1 },
          { offset: 1, color: "#ff3300", opacity: 1 },
        ],
      },
    ]);
    expect(resolved.strokes).toEqual([
      { color: "#220000", width: 4, opacity: 1, position: "center" },
    ]);
    expect(resolved.effects.dropShadows[0]).toMatchObject({
      color: "#000000",
      offsetX: 3,
      offsetY: 5,
    });
    expect(resolved.effects.outerGlow).toMatchObject({ color: "#ffffff", blur: 6 });
  });

  it("prioriza o contrato profissional e preserva fills e strokes múltiplos", () => {
    const resolved = resolveStudioTextStyle({
      fonte: "Legacy.ttf",
      tamanho: 18,
      cor: "#000000",
      studio_style: {
        version: STUDIO_TEXT_STYLE_VERSION,
        typography: {
          fontFamily: "Anime Ace",
          fontSize: 42,
          fontWeight: 600,
          fontStyle: "italic",
          align: "right",
          lineHeight: 1.35,
          tracking: 80,
          horizontalScale: 110,
          verticalScale: 95,
          baselineShift: 2,
          vertical: false,
        },
        fills: [
          { type: "solid", color: "#fefefe", opacity: 0.8 },
          {
            type: "linear-gradient",
            angle: 25,
            opacity: 0.7,
            stops: [
              { offset: 0, color: "#ff0000", opacity: 1 },
              { offset: 1, color: "#0000ff", opacity: 0.5 },
            ],
          },
        ],
        strokes: [
          { color: "#ffffff", width: 3, opacity: 1, position: "outside" },
          { color: "#000000", width: 8, opacity: 0.9, position: "outside" },
        ],
        effects: {
          dropShadows: [
            { color: "#101010", opacity: 0.75, blur: 4, offsetX: 5, offsetY: 7 },
          ],
          outerGlow: { color: "#33ccff", opacity: 0.6, blur: 9, spread: 2 },
        },
      },
    });

    expect(resolved.source).toBe("studio");
    expect(resolved.typography.fontFamily).toBe("Anime Ace");
    expect(resolved.typography.tracking).toBe(80);
    expect(resolved.fills).toHaveLength(2);
    expect(resolved.strokes.map((stroke) => stroke.width)).toEqual([3, 8]);
    expect(resolved.effects.outerGlow?.spread).toBe(2);
  });
});

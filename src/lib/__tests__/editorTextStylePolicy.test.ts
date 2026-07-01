import { describe, expect, it } from "vitest";
import { CANONICAL_FONT_FILE, canonicalizeTextStyle } from "../editorTextStylePolicy";

describe("editorTextStylePolicy", () => {
  it("uses Comic Neue Bold only when style fields are missing", () => {
    const style = canonicalizeTextStyle({
      cor: "#ff00ff",
      tamanho: 32,
    });

    expect(style).toMatchObject({
      fonte: CANONICAL_FONT_FILE,
      cor: "#ff00ff",
      tamanho: 32,
      bold: true,
      italico: false,
      sombra: false,
      sombra_cor: "",
      sombra_offset: [0, 0],
      glow: false,
      glow_cor: "",
      glow_px: 0,
      contorno: "",
      contorno_px: 0,
      curva: false,
      curva_direcao: "",
      curva_intensidade: 0,
    });
  });

  it("preserves explicit font and text style choices", () => {
    const style = canonicalizeTextStyle({
      fonte: "Newrotic.ttf",
      cor: "#ff00ff",
      tamanho: 32,
      italico: true,
      sombra: true,
      glow: true,
      contorno_px: 4,
      contorno: "#00ffff",
      glow_cor: "#ffffff",
      glow_px: 7,
      sombra_cor: "#111111",
      sombra_offset: [2, 3],
      bold: false,
    });

    expect(style).toMatchObject({
      fonte: "Newrotic.ttf",
      cor: "#ff00ff",
      tamanho: 32,
      bold: false,
      italico: true,
      sombra: true,
      glow: true,
      contorno: "#00ffff",
      contorno_px: 4,
      glow_cor: "#ffffff",
      glow_px: 7,
      sombra_cor: "#111111",
      sombra_offset: [2, 3],
    });
  });

  it("normalizes legacy default white outline style while hydrating", () => {
    const style = canonicalizeTextStyle(
      {
        fonte: "CCDaveGibbonsLower W00 Regular.ttf",
        tamanho: 28,
        cor: "#FFFFFF",
        contorno: "#000000",
        contorno_px: 2,
        glow: false,
        sombra: false,
        bold: false,
      },
      { mode: "hydrate" },
    );

    expect(style).toMatchObject({
      fonte: CANONICAL_FONT_FILE,
      cor: "#000000",
      bold: true,
      contorno: "",
      contorno_px: 0,
      glow: false,
      glow_px: 0,
      sombra: false,
      sombra_offset: [0, 0],
    });
  });

  it("preserves explicit outlined white text when it is not a legacy default", () => {
    const style = canonicalizeTextStyle(
      {
        fonte: "Newrotic.ttf",
        cor: "#FFFFFF",
        contorno: "#000000",
        contorno_px: 2,
        glow: false,
        sombra: false,
        bold: false,
      },
      { mode: "hydrate" },
    );

    expect(style).toMatchObject({
      fonte: "Newrotic.ttf",
      cor: "#FFFFFF",
      bold: false,
      contorno: "#000000",
      contorno_px: 2,
    });
  });

  it("preserves source-detected outline style during hydration", () => {
    const style = canonicalizeTextStyle(
      {
        fonte: "KOMIKAX_.ttf",
        cor: "#FFFFFF",
        contorno: "#000000",
        contorno_px: 3,
        glow: false,
      sombra: true,
      sombra_cor: "#333333",
      sombra_offset: [3, 4],
      curva: true,
      curva_direcao: "arc_up",
      curva_intensidade: 0.35,
      style_origin: "source_detected",
    },
      { mode: "hydrate" },
    );

    expect(style.contorno).toBe("#000000");
    expect(style.contorno_px).toBe(3);
    expect(style.sombra).toBe(true);
    expect(style.sombra_cor).toBe("#333333");
    expect(style.sombra_offset).toEqual([3, 4]);
    expect(style.curva).toBe(true);
    expect(style.curva_direcao).toBe("arc_up");
    expect(style.curva_intensidade).toBe(0.35);
  });

  it("does not normalize source-detected styles that resemble legacy defaults", () => {
    const style = canonicalizeTextStyle(
      {
        fonte: "ComicNeue-Bold.ttf",
        cor: "#FFFFFF",
        contorno: "#000000",
        contorno_px: 2,
        glow: false,
        sombra: false,
      },
      { mode: "hydrate", styleOrigin: "source_detected" },
    );

    expect(style).toMatchObject({
      fonte: "ComicNeue-Bold.ttf",
      cor: "#FFFFFF",
      contorno: "#000000",
      contorno_px: 2,
      glow: false,
      sombra: false,
    });
  });
});

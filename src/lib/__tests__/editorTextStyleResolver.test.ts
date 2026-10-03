import { describe, expect, it } from "vitest";
import { resolveEditorTextStyle } from "../editorTextStyleResolver";

describe("resolveEditorTextStyle legacy effect controls", () => {
  it("honors an explicit outline-off flag without discarding its saved width", () => {
    const resolved = resolveEditorTextStyle({ contorno: "#ffffff", contorno_px: 4, contorno_ativo: false });

    expect(resolved.strokes).toEqual([]);
  });

  it("uses the saved legacy gradient angle and shadow blur", () => {
    const resolved = resolveEditorTextStyle({
      cor_gradiente: ["#ff0000", "#0000ff"],
      cor_gradiente_angulo: -35,
      sombra: true,
      sombra_blur: 12,
      sombra_offset: [2, 3],
    });

    expect(resolved.fills[0]).toMatchObject({ type: "linear-gradient", angle: -35 });
    expect(resolved.effects.dropShadows[0]).toMatchObject({ blur: 12, offsetX: 2, offsetY: 3 });
  });

  it("lets toolbar controls override professional fills, strokes, and shadows", () => {
    const resolved = resolveEditorTextStyle({
      cor: "#00ff00",
      cor_gradiente: ["#ff0000", "#0000ff"],
      cor_gradiente_ativo: true,
      cor_gradiente_angulo: 25,
      contorno: "#ffffff",
      contorno_px: 3,
      contorno_ativo: false,
      sombra: true,
      sombra_blur: 16,
      studio_style: {
        version: "1.0",
        fills: [{ type: "linear-gradient", angle: 90, stops: [{ offset: 0, color: "#000000" }, { offset: 1, color: "#ffffff" }] }],
        strokes: [{ color: "#000000", width: 9 }],
        effects: { dropShadows: [{ color: "#000000", blur: 2 }] },
      },
    });

    expect(resolved.fills[0]).toMatchObject({ type: "linear-gradient", angle: 25 });
    expect(resolved.strokes).toEqual([]);
    expect(resolved.effects.dropShadows[0]).toMatchObject({ blur: 16 });
  });
});

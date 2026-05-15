import { describe, expect, it } from "vitest";
import {
  BUILTIN_TEXT_STYLE_PRESETS,
  cloneTextStylePresetPatch,
  createCustomTextStylePreset,
  mergeTextStylePresetLists,
  sanitizeTextStylePresetPatch,
  sanitizeTextStylePresets,
} from "../editorTextStylePresets";
import type { TextLayerStyle } from "../stores/appStore";

function makeStyle(overrides: Partial<TextLayerStyle> = {}): TextLayerStyle {
  return {
    fonte: "ComicNeue-Bold.ttf",
    tamanho: 28,
    cor: "#000000",
    cor_gradiente: [],
    contorno: "",
    contorno_px: 0,
    glow: false,
    glow_cor: "",
    glow_px: 0,
    sombra: false,
    sombra_cor: "",
    sombra_offset: [0, 0],
    bold: true,
    italico: false,
    rotacao: 0,
    alinhamento: "center",
    force_upper: false,
    ...overrides,
  };
}

describe("editor text style presets", () => {
  it("ships the initial visual presets", () => {
    expect(BUILTIN_TEXT_STYLE_PRESETS.map((preset) => preset.id)).toEqual([
      "town_gradient",
      "bang_comic",
      "whoosh_sfx",
      "clean_dialogue",
    ]);
    expect(BUILTIN_TEXT_STYLE_PRESETS.find((preset) => preset.id === "bang_comic")?.stylePatch).toMatchObject({
      fonte: "KOMIKAX_.ttf",
      cor: "#ffe900",
      contorno: "#000000",
      contorno_px: 4,
      sombra: true,
    });
  });

  it("sanitizes arbitrary preset payloads before saving or applying", () => {
    const patch = sanitizeTextStylePresetPatch({
      fonte: "KOMIKAX_.ttf",
      tamanho: 999,
      cor_gradiente: ["#fff", "", 42, "#000"],
      sombra_offset: [500, -500],
      alinhamento: "invalid",
      unknown: "ignored",
    });

    expect(patch).toEqual({
      fonte: "KOMIKAX_.ttf",
      tamanho: 240,
      cor_gradiente: ["#fff", "#000"],
      sombra_offset: [100, -100],
    });
  });

  it("creates custom presets from the selected layer style", () => {
    const now = new Date("2026-05-14T12:00:00.000Z");
    const preset = createCustomTextStylePreset(
      makeStyle({
        fonte: "Newrotic.ttf",
        cor: "#00d8ff",
        cor_gradiente: ["#f6ff8f", "#03d8ff"],
        sombra_offset: [3, 4],
      }),
      "Meu estilo",
      now,
    );

    expect(preset).toMatchObject({
      id: "custom_mp5fs000",
      name: "Meu estilo",
      kind: "custom",
      createdAt: "2026-05-14T12:00:00.000Z",
      stylePatch: expect.objectContaining({
        fonte: "Newrotic.ttf",
        cor: "#00d8ff",
        cor_gradiente: ["#f6ff8f", "#03d8ff"],
        sombra_offset: [3, 4],
      }),
    });
  });

  it("keeps builtins first and ignores custom presets that reuse builtin ids", () => {
    const custom = sanitizeTextStylePresets([
      {
        id: "bang_comic",
        name: "Nao deve sobrescrever",
        kind: "custom",
        stylePatch: { cor: "#ff0000" },
      },
      {
        id: "custom_valid",
        name: "Valido",
        kind: "custom",
        stylePatch: { cor: "#00ff00" },
      },
    ]);

    expect(mergeTextStylePresetLists(custom).map((preset) => preset.id)).toEqual([
      "town_gradient",
      "bang_comic",
      "whoosh_sfx",
      "clean_dialogue",
      "custom_valid",
    ]);
  });

  it("clones array fields before handing a patch to the store", () => {
    const source = { cor_gradiente: ["#fff", "#000"], sombra_offset: [1, 2] as [number, number] };
    const cloned = cloneTextStylePresetPatch(source);
    expect(cloned).toEqual(source);
    expect(cloned.cor_gradiente).not.toBe(source.cor_gradiente);
    expect(cloned.sombra_offset).not.toBe(source.sombra_offset);
  });
});

import { describe, expect, it, vi } from "vitest";
import { createCustomPreset, getProjectPreset, PROJECT_PRESETS } from "../projectPresets";

describe("projectPresets", () => {
  it("has the initial professional presets", () => {
    expect(PROJECT_PRESETS.map((preset) => preset.name)).toEqual([
      "Manhwa/Webtoon colorido",
      "Manga preto e branco",
      "Manhua colorido",
      "Baloes pequenos",
      "Scanlation clean",
      "Traducao natural BR",
      "Traducao mais literal",
      "SFX preservar",
      "SFX traduzir parcial",
    ]);
  });

  it("falls back to the recommended preset", () => {
    expect(getProjectPreset("missing").id).toBe("manhwa_webtoon_color");
  });

  it("creates custom presets from the current preset", () => {
    vi.spyOn(Date, "now").mockReturnValue(123);
    const custom = createCustomPreset(getProjectPreset("manga_bw"), "Meu preset");

    expect(custom).toMatchObject({
      id: "custom_123",
      name: "Meu preset",
      custom: true,
      settings: getProjectPreset("manga_bw").settings,
    });
  });
});

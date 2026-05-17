import { describe, expect, it, vi } from "vitest";
import { createCustomPreset, getProjectPreset, PROJECT_PRESETS, resolveEnginePresetId } from "../projectPresets";

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

  it("maps content presets to engine presets", () => {
    expect(getProjectPreset("manga_bw").settings.engine_preset_id).toBe("manga");
    expect(getProjectPreset("manhwa_webtoon_color").settings.engine_preset_id).toBe("manhwa_manhua");
    expect(getProjectPreset("manhua_color").settings.engine_preset_id).toBe("manhwa_manhua");
  });

  it("resolves engine preset ids from preset payloads and source language", () => {
    expect(resolveEnginePresetId(getProjectPreset("manga_bw"), "en")).toBe("manga");
    expect(resolveEnginePresetId({ id: "manhua_color" }, "en")).toBe("manhwa_manhua");
    expect(resolveEnginePresetId(null, "ja")).toBe("manga");
    expect(resolveEnginePresetId(null, "ko")).toBe("manhwa_manhua");
    expect(resolveEnginePresetId(null, "en")).toBe("default");
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

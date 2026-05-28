import { describe, expect, it } from "vitest";

import { buildPlainPageCommandArgs } from "../tauri";

describe("buildPlainPageCommandArgs", () => {
  it("keeps the editor engine preset on plain page actions", () => {
    expect(
      buildPlainPageCommandArgs({
        project_path: "N:/work/project.json",
        page_index: 2,
        idioma_origem: "ja",
        engine_preset_id: "default",
      }),
    ).toEqual({
      projectPath: "N:/work/project.json",
      pageIndex: 2,
      idiomaOrigem: "ja",
      idiomaDestino: undefined,
      enginePresetId: "default",
    });
  });
});

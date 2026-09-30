import { beforeEach, describe, expect, it, vi } from "vitest";
import { TauriStudioEditorBackend } from "../tauriBackend";

const { invoke } = vi.hoisted(() => ({ invoke: vi.fn() }));
vi.mock("@tauri-apps/api/core", () => ({ invoke }));

describe("TauriStudioEditorBackend", () => {
  beforeEach(() => vi.clearAllMocks());

  it("persists the normalized empty text aliases it returns", async () => {
    invoke.mockImplementation(async (command: string) => {
      if (command === "studio_load_project") return { versao: "2.0", paginas: [{ numero: 1, text_layers: [{ id: "a", bbox: [0, 0, 10, 10], original: "SOURCE", texto: "SOURCE", translated: "OLD", traduzido: "OLD" }] }] };
      if (command === "studio_save_project") return undefined;
      throw new Error(`Comando inesperado: ${command}`);
    });
    const backend = new TauriStudioEditorBackend();
    await expect(backend.patchEditorTextLayer({
      project_path: "C:/fixture/project.json", page_index: 0, layer_id: "a", patch: { original: "", translated: "" },
    })).resolves.toMatchObject({ original: "", texto: "", translated: "", traduzido: "" });
    expect(invoke.mock.calls.find(([command]) => command === "studio_save_project")?.[1]).toEqual(expect.objectContaining({
      config: expect.objectContaining({ project_json: expect.objectContaining({ paginas: [expect.objectContaining({ text_layers: [expect.objectContaining({ original: "", texto: "", translated: "", traduzido: "" })] })] }) }),
    }));
  });
});

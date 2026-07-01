import { describe, expect, it } from "vitest";
import { importStudioProject } from "../../project/adapters";
import { MemoryStudioEditorBackend } from "../memoryBackend";

function sampleProject() {
  return importStudioProject({
    versao: "1.0",
    paginas: [
      {
        numero: 1,
        arquivo_original: "original/001.png",
        textos: [{ id: "a", bbox: [0, 0, 10, 10], texto: "A", traduzido: "B" }],
      },
    ],
  }).project;
}

describe("MemoryStudioEditorBackend", () => {
  it("loads pages and patches text aliases", async () => {
    const backend = new MemoryStudioEditorBackend({ "memory://test": sampleProject() });

    const patched = await backend.patchEditorTextLayer({
      project_path: "memory://test",
      page_index: 0,
      layer_id: "a",
      patch: { translated: "C" },
    });
    const page = await backend.loadEditorPage({ project_path: "memory://test", page_index: 0 });

    expect(patched.traduzido).toBe("C");
    expect(page.page.text_layers[0].traduzido).toBe("C");
    expect(page.page.textos[0].translated).toBe("C");
  });

  it("creates and deletes text layers", async () => {
    const backend = new MemoryStudioEditorBackend({ "memory://test": sampleProject() });
    const created = await backend.createEditorTextLayer({
      project_path: "memory://test",
      page_index: 0,
      layout_bbox: [10, 20, 30, 40],
    });

    let page = await backend.loadEditorPage({ project_path: "memory://test", page_index: 0 });
    expect(page.page.text_layers).toHaveLength(2);

    await backend.deleteEditorTextLayer({ project_path: "memory://test", page_index: 0, layer_id: created.id });
    page = await backend.loadEditorPage({ project_path: "memory://test", page_index: 0 });
    expect(page.page.text_layers).toHaveLength(1);
  });

  it("updates bitmap layers and final rendered alias", async () => {
    const backend = new MemoryStudioEditorBackend({ "memory://test": sampleProject() });
    const pngData = "data:image/png;base64,abc";
    const path = await backend.updateBitmapLayer({
      project_path: "memory://test",
      page_index: 0,
      layer_key: "rendered",
      width: 100,
      height: 100,
      png_data: pngData,
    });

    const page = await backend.loadEditorPage({ project_path: "memory://test", page_index: 0 });
    expect(path).toBe(pngData);
    expect(page.page.arquivo_traduzido).toBe(pngData);
    expect(page.page.image_layers.rendered?.path).toBe(pngData);
  });
});

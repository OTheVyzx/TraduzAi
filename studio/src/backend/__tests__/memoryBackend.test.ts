import { describe, expect, it } from "vitest";
import { importStudioProject } from "../../project/adapters";
import { MemoryStudioEditorBackend } from "../memoryBackend";
import type { FluxGenerateConfig } from "../../ai/fluxContract";

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

  it("keeps projected scene nodes synchronized across text mutations", async () => {
    const backend = new MemoryStudioEditorBackend({ "memory://scene-sync": sampleProject() });
    const created = await backend.createEditorTextLayer({
      project_path: "memory://scene-sync",
      page_index: 0,
      layout_bbox: [10, 20, 30, 40],
    });

    await backend.setEditorLayerVisibility({
      project_path: "memory://scene-sync",
      page_index: 0,
      layer_kind: "text",
      layer_id: created.id,
      visible: false,
    });
    let page = await backend.loadEditorPage({ project_path: "memory://scene-sync", page_index: 0 });
    expect(page.page.studio_scene.nodes.find((node) => node.text_layer_id === created.id)).toMatchObject({
      kind: "text",
      visible: false,
    });

    await backend.deleteEditorTextLayer({
      project_path: "memory://scene-sync",
      page_index: 0,
      layer_id: created.id,
    });
    page = await backend.loadEditorPage({ project_path: "memory://scene-sync", page_index: 0 });
    expect(page.page.studio_scene.nodes.some((node) => node.text_layer_id === created.id)).toBe(false);
    expect(page.page.studio_scene.nodes.some((node) => node.text_layer_id === "a")).toBe(true);
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

  it("serializes concurrent scene and text mutations without losing either update", async () => {
    const backend = new MemoryStudioEditorBackend({ "memory://concurrent": sampleProject() });
    let releaseFirst!: () => void;
    let markStarted!: () => void;
    const firstStarted = new Promise<void>((resolve) => { markStarted = resolve; });
    const firstGate = new Promise<void>((resolve) => { releaseFirst = resolve; });

    const sceneMutation = backend.mutateProject({
      project_path: "memory://concurrent",
      mutate: async (project) => {
        markStarted();
        await firstGate;
        project.paginas[0].studio_scene.nodes[0].name = "Cena atualizada";
      },
    });
    await firstStarted;
    const textMutation = backend.patchEditorTextLayer({
      project_path: "memory://concurrent",
      page_index: 0,
      layer_id: "a",
      patch: { translated: "Texto concorrente" },
    });
    releaseFirst();
    await Promise.all([sceneMutation, textMutation]);

    const project = await backend.loadProject({ project_path: "memory://concurrent" });
    expect(project.paginas[0].studio_scene.nodes[0].name).toBe("Cena atualizada");
    expect(project.paginas[0].text_layers[0].translated).toBe("Texto concorrente");
  });

  it("stores a generated asset without overwriting compatibility bitmap layers", async () => {
    const backend = new MemoryStudioEditorBackend({ "memory://test": sampleProject() });
    const pngData = "data:image/png;base64,generated";

    const path = await backend.saveGeneratedAsset({
      project_path: "memory://test",
      page_index: 0,
      asset_id: "retouch-1",
      png_data: pngData,
    });

    const page = await backend.loadEditorPage({ project_path: "memory://test", page_index: 0 });
    expect(path).toBe(pngData);
    expect(page.page.image_layers.rendered?.path).not.toBe(pngData);
    expect(page.page.image_layers.inpaint?.path).not.toBe(pngData);
  });

  it("reports FLUX as local-only when the browser memory backend has no adapter", async () => {
    const backend = new MemoryStudioEditorBackend({ "memory://test": sampleProject() });
    const status = await backend.fluxProviderStatus();

    expect(status).toMatchObject({ status: "missing", provider: "local-adapter" });
    await expect(backend.generateFluxFill({ job_id: "job" } as FluxGenerateConfig))
      .rejects.toThrow("Tauri");
  });
});

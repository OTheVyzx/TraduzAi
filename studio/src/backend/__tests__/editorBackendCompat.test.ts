import { describe, expect, it, vi } from "vitest";
import { importStudioProject } from "../../project/adapters";
import { createLegacyEditorBackendAdapter } from "../editorBackendCompat";
import type { StudioEditorBackend } from "../editorBackend";
import { MemoryStudioEditorBackend } from "../memoryBackend";

function backendWithProject() {
  const project = importStudioProject({
    versao: "1.0",
    paginas: [
      {
        numero: 1,
        arquivo_original: "original/001.png",
        textos: [{ id: "a", bbox: [0, 0, 10, 10], texto: "A", traduzido: "B" }],
      },
    ],
  }).project;
  const backend = new MemoryStudioEditorBackend({ "memory://compat": project });
  return { backend, compat: createLegacyEditorBackendAdapter(backend) };
}

describe("createLegacyEditorBackendAdapter", () => {
  it("exposes current editor text methods", async () => {
    const { compat } = backendWithProject();

    const patched = await compat.patchEditorTextLayer({
      project_path: "memory://compat",
      page_index: 0,
      layer_id: "a",
      patch: { translated: "C" },
    });

    expect(patched.traduzido).toBe("C");
  });

  it("merges structural text saves over the latest scene instead of restoring a stale scene", async () => {
    const { backend, compat } = backendWithProject();
    const staleEditorProject = (await compat.loadEditorPage({ project_path: "memory://compat", page_index: 0 })).project!;
    staleEditorProject.paginas[0].text_layers[0].translated = "Estrutural";
    staleEditorProject.paginas[0].text_layers[0].traduzido = "Estrutural";
    staleEditorProject.paginas[0].textos = staleEditorProject.paginas[0].text_layers;
    await backend.mutateProject({
      project_path: "memory://compat",
      mutate: (latest) => {
        latest.paginas[0].studio_scene.nodes[0].name = "Cena mais recente";
      },
    });

    await compat.saveProjectJson({ project_path: "memory://compat", project_json: staleEditorProject });

    const saved = await backend.loadProject({ project_path: "memory://compat" });
    expect(saved.paginas[0].text_layers[0].translated).toBe("Estrutural");
    expect(saved.paginas[0].studio_scene.nodes[0].name).toBe("Cena mais recente");
  });

  it("applies only the structural delta and preserves a concurrent chapter style", async () => {
    const { backend, compat } = backendWithProject();
    const incoming = (await compat.loadEditorPage({ project_path: "memory://compat", page_index: 0 })).project!;
    incoming.paginas[0].text_layers.push({
      ...incoming.paginas[0].text_layers[0],
      id: "created-locally",
      translated: "Nova",
      traduzido: "Nova",
      order: 1,
    });
    incoming.paginas[0].textos = incoming.paginas[0].text_layers;
    await backend.patchEditorTextLayer({
      project_path: "memory://compat",
      page_index: 0,
      layer_id: "a",
      patch: { style: { fonte: "Wild" }, estilo: { fonte: "Wild" } },
    });

    await compat.saveProjectJson({ project_path: "memory://compat", project_json: incoming });

    const saved = await backend.loadProject({ project_path: "memory://compat" });
    expect(saved.paginas[0].text_layers.map((layer) => layer.id)).toEqual(["a", "created-locally"]);
    expect(saved.paginas[0].text_layers[0].style).toMatchObject({ fonte: "Wild" });
  });

  it("rejects a structural save that would overwrite a newer edit in the same field", async () => {
    const { backend, compat } = backendWithProject();
    const incoming = (await compat.loadEditorPage({ project_path: "memory://compat", page_index: 0 })).project!;
    incoming.paginas[0].text_layers[0].translated = "Edicao estrutural";
    incoming.paginas[0].text_layers[0].traduzido = "Edicao estrutural";
    await backend.patchEditorTextLayer({
      project_path: "memory://compat",
      page_index: 0,
      layer_id: "a",
      patch: { translated: "Edicao mais recente" },
    });

    await expect(compat.saveProjectJson({ project_path: "memory://compat", project_json: incoming }))
      .rejects.toThrow("mudou no campo");
    expect((await backend.loadProject({ project_path: "memory://compat" })).paginas[0].text_layers[0].translated)
      .toBe("Edicao mais recente");
  });

  it("maps brush/mask/recovery/reinpaint calls to Studio bitmap layers", async () => {
    const { backend, compat } = backendWithProject();
    const base = {
      project_path: "memory://compat",
      page_index: 0,
      width: 100,
      height: 100,
      brush_size: 20,
      strokes: [[[1, 1], [2, 2]]] as [number, number][][],
      png_data: "data:image/png;base64,abc",
    };

    await compat.updateBrushRegion(base);
    await compat.updateMaskRegion(base);
    await compat.updateRecoveryRegion(base);
    await compat.updateReinpaintRegion(base);

    const page = await backend.loadEditorPage({ project_path: "memory://compat", page_index: 0 });
    expect(page.page.image_layers.brush?.path).toBe(base.png_data);
    expect(page.page.image_layers.mask?.path).toBe(base.png_data);
    expect(page.page.image_layers.recovery?.path).toBeNull();
    expect(page.page.image_layers.recovery?.visible).toBe(false);
    expect(page.page.image_layers.inpaint?.path).toBe(base.png_data);
  });

  it("keeps healing mask out of the inpaint layer in Studio compat mode", async () => {
    const { backend, compat } = backendWithProject();

    const maskPath = await compat.writeHealingMask({
      project_path: "memory://compat",
      page_index: 0,
      png_data: "data:image/png;base64,mask",
      bbox: [1, 2, 30, 40],
    });
    const result = await compat.healInpaintRegion({
      project_path: "memory://compat",
      page_index: 0,
      bbox: [1, 2, 30, 40],
      mask_path: maskPath,
    });

    const page = await backend.loadEditorPage({ project_path: "memory://compat", page_index: 0 });
    expect(page.page.image_layers.mask?.path).toBe("data:image/png;base64,mask");
    expect(page.page.image_layers.inpaint?.path).toBe("original/001.png");
    expect(result.inpaint_path).toBe("original/001.png");
    expect(result.bbox).toEqual([1, 2, 30, 40]);
  });

  it("repairs a mask path leaked into inpaint when loading a page", async () => {
    const { backend, compat } = backendWithProject();
    await backend.updateBitmapLayer({
      project_path: "memory://compat",
      page_index: 0,
      layer_key: "mask",
      width: 100,
      height: 100,
      png_data: "data:image/png;base64,mask",
      dirty_bbox: null,
    });
    await backend.updateBitmapLayer({
      project_path: "memory://compat",
      page_index: 0,
      layer_key: "inpaint",
      width: 100,
      height: 100,
      png_data: "data:image/png;base64,mask",
      dirty_bbox: null,
    });

    const payload = await compat.loadEditorPage({ project_path: "memory://compat", page_index: 0 });

    expect(payload.page.image_layers.mask?.path).toBe("data:image/png;base64,mask");
    expect(payload.page.image_layers.inpaint?.path).toBe("original/001.png");
  });

  it("rejects stroke-only bitmap updates in Studio compat mode", async () => {
    const { compat } = backendWithProject();

    await expect(
      compat.updateBrushRegion({
        project_path: "memory://compat",
        page_index: 0,
        width: 100,
        height: 100,
        brush_size: 20,
        strokes: [[[1, 1], [2, 2]]],
      }),
    ).rejects.toThrow(/png_data/);
  });

  it("renders preview by updating rendered layer", async () => {
    const { backend, compat } = backendWithProject();
    const loaded = await compat.loadEditorPage({ project_path: "memory://compat", page_index: 0 });

    const result = await compat.renderPreviewPage({
      project_path: "memory://compat",
      page_index: 0,
      page: loaded.page,
      fingerprint: "abc",
    });

    const page = await backend.loadEditorPage({ project_path: "memory://compat", page_index: 0 });
    expect(result.renderer_backend).toBe("studio-local");
    expect(page.page.image_layers.rendered?.path).toBe("data:image/png;base64,");
  });

  it("uses Studio Lite detect when the backend provides it", async () => {
    const { backend } = backendWithProject();
    const liteBackend: StudioEditorBackend = Object.assign(backend, {
      studioLiteDetectPage: vi.fn(async () => ({
        detections: [{ bbox: [1, 2, 30, 40] as [number, number, number, number] }],
        message: "studio lite detect ok",
      })),
    });
    const compat = createLegacyEditorBackendAdapter(liteBackend);

    await expect(compat.detectPage({ project_path: "memory://compat", page_index: 0 })).resolves.toBe(
      "studio lite detect ok",
    );
    await expect(compat.detectBoxesPage({ project_path: "memory://compat", page_index: 0 })).resolves.toBe(
      "studio lite detect ok",
    );

    expect(liteBackend.studioLiteDetectPage).toHaveBeenCalledTimes(2);
    expect(liteBackend.studioLiteDetectPage).toHaveBeenNthCalledWith(1, {
      project_path: "memory://compat",
      page_index: 0,
      boxes_only: false,
    });
  });

  it("builds a Studio Lite mask from detect action detections when no mask path is returned", async () => {
    const { backend } = backendWithProject();
    const liteBackend: StudioEditorBackend = Object.assign(backend, {
      studioLiteDetectPage: vi.fn(async () => ({
        detections: [{ bbox: [1, 2, 30, 40] as [number, number, number, number] }],
        message: "studio lite detect ok",
      })),
      studioLiteBuildMask: vi.fn(async () => "mask/studio-lite.png"),
    });
    const compat = createLegacyEditorBackendAdapter(liteBackend);

    const result = await compat.runPageActionWithOptionalMask({
      project_path: "memory://compat",
      page_index: 0,
      action: "detect",
    });

    expect(liteBackend.studioLiteBuildMask).toHaveBeenCalledWith({
      project_path: "memory://compat",
      page_index: 0,
      detections: [{ bbox: [1, 2, 30, 40] }],
    });
    expect(result.changed_assets).toContain("mask");
  });

  it("keeps local detect stubs when Studio Lite detect is unavailable", async () => {
    const { compat } = backendWithProject();

    await expect(compat.detectPage({ project_path: "memory://compat", page_index: 0 })).resolves.toBe(
      "Deteccao ainda nao conectada no Studio local",
    );
    await expect(compat.detectBoxesPage({ project_path: "memory://compat", page_index: 0 })).resolves.toBe(
      "Deteccao de caixas ainda nao conectada no Studio local",
    );
  });

  it("uses Studio Lite inpaint for regional reinpaint when bbox or mask is provided", async () => {
    const { backend } = backendWithProject();
    const liteBackend: StudioEditorBackend = Object.assign(backend, {
      studioLiteInpaintRegion: vi.fn(async () => ({
        page_index: 0,
        inpaint_path: "inpaint/studio-lite.png",
        before_inpaint_path: "original/001.png",
        bbox: [1, 2, 30, 40] as [number, number, number, number],
      })),
    });
    const compat = createLegacyEditorBackendAdapter(liteBackend);

    await expect(
      compat.reinpaintPage({
        project_path: "memory://compat",
        page_index: 0,
        bbox: [1, 2, 30, 40],
        mask_path: "mask/studio-lite.png",
      }),
    ).resolves.toBe("inpaint/studio-lite.png");

    expect(liteBackend.studioLiteInpaintRegion).toHaveBeenCalledWith({
      project_path: "memory://compat",
      page_index: 0,
      bbox: [1, 2, 30, 40],
      mask_path: "mask/studio-lite.png",
    });
  });

  it("uses Studio Lite inpaint for optional-mask inpaint actions", async () => {
    const { backend } = backendWithProject();
    const liteBackend: StudioEditorBackend = Object.assign(backend, {
      studioLiteInpaintRegion: vi.fn(async () => ({
        page_index: 0,
        inpaint_path: "inpaint/studio-lite.png",
        bbox: [3, 4, 50, 60] as [number, number, number, number],
      })),
    });
    const compat = createLegacyEditorBackendAdapter(liteBackend);

    const result = await compat.runPageActionWithOptionalMask({
      project_path: "memory://compat",
      page_index: 0,
      action: "inpaint",
      bbox: [3, 4, 50, 60],
      mask_path: "mask/studio-lite.png",
    });

    expect(liteBackend.studioLiteInpaintRegion).toHaveBeenCalledWith({
      project_path: "memory://compat",
      page_index: 0,
      bbox: [3, 4, 50, 60],
      mask_path: "mask/studio-lite.png",
    });
    expect(result).toEqual({
      action: "inpaint",
      changed_assets: ["inpaint", "project_json"],
      message: "Inpaint Studio Lite aplicado",
    });
  });
});

import { beforeEach, describe, expect, it } from "vitest";
import { copyStyleFromLayer, createApplyStyleCommand } from "../../editor/batch/chapterCommands";
import { createRecoverySnapshot } from "../../autosave/recovery";
import { configureStudioEditorBackend, getStudioEditorBackend } from "../../backend/editorBackend";
import { MemoryStudioEditorBackend } from "../../backend/memoryBackend";
import { importStudioProject } from "../../project/adapters";
import { useStudioProjectStore } from "../projectStore";

describe("useStudioProjectStore", () => {
  beforeEach(() => {
    useStudioProjectStore.setState({
      project: null,
      projectPath: null,
      currentPageIndex: 0,
      lastImport: null,
      error: null,
      chapterHistory: [],
      chapterHistoryIndex: 0,
      isProjectSaving: false,
      recoverySnapshot: null,
    });
  });

  it("imports project json into the configured backend", async () => {
    await useStudioProjectStore.getState().importProjectJson(
      JSON.stringify({
        versao: "1.0",
        paginas: [{ numero: 1, textos: [{ id: "a", bbox: [0, 0, 1, 1], texto: "A", traduzido: "B" }] }],
      }),
      "memory://store-test",
    );

    const state = useStudioProjectStore.getState();
    expect(state.project?.paginas).toHaveLength(1);
    expect(state.lastImport?.kind).toBe("traduzai_v1");

    await state.loadProject("memory://store-test");
    expect(useStudioProjectStore.getState().project?.paginas[0].text_layers[0].translated).toBe("B");
  });

  it("closes the active project explicitly and clears chapter-only state", async () => {
    await useStudioProjectStore.getState().importProjectJson(
      JSON.stringify({ versao: "1.0", paginas: [{ numero: 1, textos: [] }] }),
      "memory://close-project",
    );

    useStudioProjectStore.getState().closeProject();

    expect(useStudioProjectStore.getState()).toMatchObject({
      project: null,
      projectPath: null,
      currentPageIndex: 0,
      lastImport: null,
      chapterHistory: [],
      chapterHistoryIndex: 0,
      recoverySnapshot: null,
    });
  });

  it("patches current text layers through the compatibility backend", async () => {
    await useStudioProjectStore.getState().importProjectJson(
      JSON.stringify({
        versao: "1.0",
        paginas: [{ numero: 1, textos: [{ id: "a", bbox: [0, 0, 1, 1], texto: "A", traduzido: "B" }] }],
      }),
      "memory://store-patch-test",
    );

    await useStudioProjectStore.getState().patchCurrentTextLayer("a", { translated: "C", traduzido: "C" });

    const layer = useStudioProjectStore.getState().project?.paginas[0].text_layers[0];
    expect(layer?.translated).toBe("C");
    expect(layer?.traduzido).toBe("C");
  });

  it("toggles text and image layer visibility through the compatibility backend", async () => {
    await useStudioProjectStore.getState().importProjectJson(
      JSON.stringify({
        versao: "1.0",
        paginas: [{ numero: 1, arquivo_original: "base.png", textos: [{ id: "a", bbox: [0, 0, 1, 1] }] }],
      }),
      "memory://store-visibility-test",
    );

    await useStudioProjectStore.getState().setCurrentTextLayerVisibility("a", false);
    await useStudioProjectStore.getState().setCurrentImageLayerVisibility("base", false);

    const page = useStudioProjectStore.getState().project?.paginas[0];
    expect(page?.text_layers[0].visible).toBe(false);
    expect(page?.image_layers.base?.visible).toBe(false);
  });

  it("persists chapter commands and supports transactional undo/redo", async () => {
    await useStudioProjectStore.getState().importProjectJson(
      JSON.stringify({
        versao: "1.0",
        paginas: [{
          numero: 1,
          textos: [
            { id: "source", bbox: [0, 0, 1, 1], traduzido: "A", estilo: { fonte: "Wild", tamanho: 32 } },
            { id: "target", bbox: [0, 2, 1, 3], traduzido: "B", estilo: { fonte: "Arial", tamanho: 16 } },
          ],
        }],
      }),
      "memory://store-command-test",
    );
    const project = useStudioProjectStore.getState().project!;
    const clipboard = copyStyleFromLayer(project, { pageIndex: 0, layerId: "source" });
    const command = createApplyStyleCommand(project, clipboard, [{ pageIndex: 0, layerId: "target" }]);

    expect(await useStudioProjectStore.getState().executeChapterCommand(command)).toBe(true);
    expect(useStudioProjectStore.getState().project?.paginas[0].text_layers[1].style).toMatchObject({ fonte: "Wild" });
    expect(useStudioProjectStore.getState().chapterHistoryIndex).toBe(1);

    await useStudioProjectStore.getState().patchCurrentTextLayer("target", {
      translated: "Edicao posterior",
      traduzido: "Edicao posterior",
    });

    expect(await useStudioProjectStore.getState().undoChapterCommand()).toBe(true);
    expect(useStudioProjectStore.getState().project?.paginas[0].text_layers[1].style).toMatchObject({ fonte: "Arial" });
    expect(useStudioProjectStore.getState().project?.paginas[0].text_layers[1].translated).toBe("Edicao posterior");

    expect(await useStudioProjectStore.getState().redoChapterCommand()).toBe(true);
    expect(useStudioProjectStore.getState().project?.paginas[0].text_layers[1].style).toMatchObject({ fonte: "Wild" });
    expect(useStudioProjectStore.getState().project?.paginas[0].text_layers[1].translated).toBe("Edicao posterior");
  });

  it("refuses chapter undo when the same style field changed afterwards", async () => {
    await useStudioProjectStore.getState().importProjectJson(
      JSON.stringify({
        versao: "1.0",
        paginas: [{
          numero: 1,
          textos: [
            { id: "source", bbox: [0, 0, 1, 1], traduzido: "A", estilo: { fonte: "Wild" } },
            { id: "target", bbox: [0, 2, 1, 3], traduzido: "B", estilo: { fonte: "Arial" } },
          ],
        }],
      }),
      "memory://store-command-conflict",
    );
    const project = useStudioProjectStore.getState().project!;
    const command = createApplyStyleCommand(
      project,
      copyStyleFromLayer(project, { pageIndex: 0, layerId: "source" }),
      [{ pageIndex: 0, layerId: "target" }],
    );
    expect(await useStudioProjectStore.getState().executeChapterCommand(command)).toBe(true);
    await useStudioProjectStore.getState().patchCurrentTextLayer("target", {
      style: { fonte: "Manual" },
      estilo: { fonte: "Manual" },
    });

    expect(await useStudioProjectStore.getState().undoChapterCommand()).toBe(false);
    expect(useStudioProjectStore.getState().project?.paginas[0].text_layers[1].style).toMatchObject({ fonte: "Manual" });
    expect(useStudioProjectStore.getState().error).toContain("mudou no campo");
  });

  it("offers a divergent recovery snapshot and restores it explicitly", async () => {
    await useStudioProjectStore.getState().importProjectJson(
      JSON.stringify({
        versao: "1.0",
        paginas: [{ numero: 1, textos: [{ id: "a", bbox: [0, 0, 1, 1], traduzido: "Disco" }] }],
      }),
      "memory://store-recovery-test",
    );
    const snapshotProject = structuredClone(useStudioProjectStore.getState().project!);
    snapshotProject.paginas[0].text_layers[0].translated = "Recuperado";
    snapshotProject.paginas[0].text_layers[0].traduzido = "Recuperado";
    snapshotProject.paginas[0].textos = snapshotProject.paginas[0].text_layers;
    await getStudioEditorBackend().saveRecoverySnapshot({
      project_path: "memory://store-recovery-test",
      snapshot: createRecoverySnapshot("memory://store-recovery-test", snapshotProject, 1234),
    });

    await useStudioProjectStore.getState().loadProject("memory://store-recovery-test");
    expect(useStudioProjectStore.getState().recoverySnapshot?.savedAt).toBe(1234);

    expect(await useStudioProjectStore.getState().restoreRecovery()).toBe(true);
    expect(useStudioProjectStore.getState().project?.paginas[0].text_layers[0].translated).toBe("Recuperado");
    expect(useStudioProjectStore.getState().recoverySnapshot).toBeNull();
  });

  it("opens a valid project even when recovery storage is unavailable", async () => {
    const originalBackend = getStudioEditorBackend();
    const project = importStudioProject({
      versao: "1.0",
      paginas: [{ numero: 1, textos: [{ id: "a", bbox: [0, 0, 1, 1], traduzido: "Aberto" }] }],
    }).project;
    const backend = new MemoryStudioEditorBackend({ "memory://recovery-unavailable": project });
    backend.loadRecoverySnapshot = async () => {
      throw new Error("pasta de recovery bloqueada");
    };
    configureStudioEditorBackend(backend);
    try {
      await useStudioProjectStore.getState().loadProject("memory://recovery-unavailable");
      expect(useStudioProjectStore.getState().project?.paginas[0].text_layers[0].translated).toBe("Aberto");
      expect(useStudioProjectStore.getState().error).toBeNull();
      expect(useStudioProjectStore.getState().recoverySnapshot).toBeNull();
    } finally {
      configureStudioEditorBackend(originalBackend);
    }
  });
});

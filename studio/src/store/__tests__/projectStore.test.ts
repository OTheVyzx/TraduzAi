import { beforeEach, describe, expect, it } from "vitest";
import {
  copyStyleFromLayer,
  createApplyStyleCommand,
  createReplaceTextCommand,
  previewChapterReplacements,
} from "../../editor/batch/chapterCommands";
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
      hasUnsavedChanges: false,
      recoverySnapshot: null,
    });
  });

  it("imports project json into the configured backend", async () => {
    const originalBackend = getStudioEditorBackend();
    const backend = new MemoryStudioEditorBackend();
    configureStudioEditorBackend(backend);
    try {
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
      expect((await backend.loadProject({ project_path: "memory://store-test" }))
        .paginas[0].text_layers[0].translated).toBe("B");

      await state.loadProject("memory://store-test");
      expect(useStudioProjectStore.getState().project?.paginas[0].text_layers[0].translated).toBe("B");
    } finally {
      configureStudioEditorBackend(originalBackend);
    }
  });

  it("ignores a project load that finishes after another project becomes active", async () => {
    const originalBackend = getStudioEditorBackend();
    const stalePath = "memory://stale-load";
    const staleProject = importStudioProject({
      versao: "1.0",
      paginas: [{ numero: 1, textos: [{ id: "stale", bbox: [0, 0, 1, 1], traduzido: "Antigo" }] }],
    }).project;
    const backend = new MemoryStudioEditorBackend({ [stalePath]: staleProject });
    let releaseLoad!: () => void;
    let markLoadStarted!: () => void;
    const loadStarted = new Promise<void>((resolve) => { markLoadStarted = resolve; });
    const loadGate = new Promise<void>((resolve) => { releaseLoad = resolve; });
    const loadProject = backend.loadProject.bind(backend);
    backend.loadProject = async (config) => {
      markLoadStarted();
      await loadGate;
      return loadProject(config);
    };
    configureStudioEditorBackend(backend);
    try {
      const staleLoad = useStudioProjectStore.getState().loadProject(stalePath);
      await loadStarted;
      await useStudioProjectStore.getState().importProjectJson(
        JSON.stringify({
          versao: "1.0",
          paginas: [{ numero: 1, textos: [{ id: "new", bbox: [0, 0, 1, 1], traduzido: "Atual" }] }],
        }),
        "memory://active-project",
      );

      releaseLoad();
      await staleLoad;

      expect(useStudioProjectStore.getState().projectPath).toBe("memory://active-project");
      expect(useStudioProjectStore.getState().project?.paginas[0].text_layers[0].translated).toBe("Atual");
    } finally {
      configureStudioEditorBackend(originalBackend);
    }
  });

  it("does not restore a saved project over a newer active project", async () => {
    const originalBackend = getStudioEditorBackend();
    const savedPath = "memory://save-finishes-late";
    const savedProject = importStudioProject({
      versao: "1.0",
      paginas: [{ numero: 1, textos: [{ id: "saved", bbox: [0, 0, 1, 1], traduzido: "Salvo" }] }],
    }).project;
    const backend = new MemoryStudioEditorBackend({ [savedPath]: savedProject });
    let releaseSave!: () => void;
    let markSaveStarted!: () => void;
    const saveStarted = new Promise<void>((resolve) => { markSaveStarted = resolve; });
    const saveGate = new Promise<void>((resolve) => { releaseSave = resolve; });
    const saveProjectJson = backend.saveProjectJson.bind(backend);
    backend.saveProjectJson = async (config) => {
      markSaveStarted();
      await saveGate;
      return saveProjectJson(config);
    };
    configureStudioEditorBackend(backend);
    try {
      await useStudioProjectStore.getState().loadProject(savedPath);
      const lateSave = useStudioProjectStore.getState().saveProject();
      await saveStarted;
      const activateNewProject = useStudioProjectStore.getState().importProjectJson(
        JSON.stringify({
          versao: "1.0",
          paginas: [{ numero: 1, textos: [{ id: "new", bbox: [0, 0, 1, 1], traduzido: "Projeto atual" }] }],
        }),
        "memory://new-active-project",
      );

      releaseSave();
      await Promise.all([lateSave, activateNewProject]);

      expect(useStudioProjectStore.getState().projectPath).toBe("memory://new-active-project");
      expect(useStudioProjectStore.getState().project?.paginas[0].text_layers[0].translated).toBe("Projeto atual");
    } finally {
      configureStudioEditorBackend(originalBackend);
    }
  });

  it("does not apply a late text mutation to a newer active project", async () => {
    const originalBackend = getStudioEditorBackend();
    const editedPath = "memory://edit-finishes-late";
    const editedProject = importStudioProject({
      versao: "1.0",
      paginas: [{ numero: 1, textos: [{ id: "edited", bbox: [0, 0, 1, 1], traduzido: "Antes" }] }],
    }).project;
    const backend = new MemoryStudioEditorBackend({ [editedPath]: editedProject });
    let releasePatch!: () => void;
    let markPatchPersisted!: () => void;
    const patchPersisted = new Promise<void>((resolve) => { markPatchPersisted = resolve; });
    const patchGate = new Promise<void>((resolve) => { releasePatch = resolve; });
    const patchEditorTextLayer = backend.patchEditorTextLayer.bind(backend);
    backend.patchEditorTextLayer = async (config) => {
      const result = await patchEditorTextLayer(config);
      markPatchPersisted();
      await patchGate;
      return result;
    };
    configureStudioEditorBackend(backend);
    try {
      await useStudioProjectStore.getState().loadProject(editedPath);
      const latePatch = useStudioProjectStore.getState().patchCurrentTextLayer("edited", {
        translated: "Depois",
        traduzido: "Depois",
      });
      await patchPersisted;
      await useStudioProjectStore.getState().importProjectJson(
        JSON.stringify({
          versao: "1.0",
          paginas: [{ numero: 1, textos: [{ id: "new", bbox: [0, 0, 1, 1], traduzido: "Projeto atual" }] }],
        }),
        "memory://active-after-edit",
      );

      releasePatch();
      await latePatch;

      expect(useStudioProjectStore.getState().projectPath).toBe("memory://active-after-edit");
      expect(useStudioProjectStore.getState().project?.paginas[0].text_layers[0].translated).toBe("Projeto atual");
    } finally {
      configureStudioEditorBackend(originalBackend);
    }
  });

  it("closes the active project explicitly and clears chapter-only state", async () => {
    await useStudioProjectStore.getState().importProjectJson(
      JSON.stringify({ versao: "1.0", paginas: [{ numero: 1, textos: [] }] }),
      "memory://close-project",
    );

    useStudioProjectStore.getState().closeProject(true);

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

  it("blocks a silent close while the in-memory project has unsaved changes", async () => {
    await useStudioProjectStore.getState().importProjectJson(
      JSON.stringify({ versao: "1.0", paginas: [{ numero: 1, textos: [] }] }),
      "memory://dirty-project",
    );

    expect(useStudioProjectStore.getState().hasUnsavedChanges).toBe(true);
    expect(useStudioProjectStore.getState().closeProject()).toBe(false);
    expect(useStudioProjectStore.getState().project).not.toBeNull();
    expect(useStudioProjectStore.getState().error).toContain("alterações não salvas");
    expect(useStudioProjectStore.getState().closeProject(true)).toBe(true);
    expect(useStudioProjectStore.getState().project).toBeNull();
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

  it("keeps a multipage chapter command recoverable when persistence fails", async () => {
    const originalBackend = getStudioEditorBackend();
    const projectPath = "memory://store-command-write-failure";
    const project = importStudioProject({
      versao: "2.0",
      paginas: [
        { numero: 1, textos: [{ id: "a", bbox: [0, 0, 1, 1], texto: "A", traduzido: "Antes" }] },
        { numero: 2, textos: [{ id: "b", bbox: [0, 0, 1, 1], texto: "B", traduzido: "Antes" }] },
      ],
    }).project;
    const backend = new MemoryStudioEditorBackend({ [projectPath]: project });
    configureStudioEditorBackend(backend);
    try {
      await useStudioProjectStore.getState().loadProject(projectPath);
      const before = structuredClone(useStudioProjectStore.getState().project!);
      const matches = previewChapterReplacements(before, {
        query: "Antes",
        replacement: "Depois",
        caseSensitive: false,
        wholeWord: true,
      });
      const command = createReplaceTextCommand(before, matches);
      backend.mutateProject = async () => {
        throw new Error("falha simulada ao persistir lote");
      };

      expect(await useStudioProjectStore.getState().executeChapterCommand(command)).toBe(false);
      const state = useStudioProjectStore.getState();
      expect(state.project).toEqual(before);
      expect(state.chapterHistory).toEqual([]);
      expect(state.chapterHistoryIndex).toBe(0);
      expect(state.hasUnsavedChanges).toBe(true);
      expect(state.error).toContain("falha simulada");
    } finally {
      configureStudioEditorBackend(originalBackend);
    }
  });

  it("does not apply a completed chapter command to a newer active project", async () => {
    const originalBackend = getStudioEditorBackend();
    const editedPath = "memory://command-finishes-late";
    const editedProject = importStudioProject({
      versao: "2.0",
      paginas: [{ numero: 1, textos: [{ id: "a", bbox: [0, 0, 1, 1], traduzido: "Antes" }] }],
    }).project;
    const backend = new MemoryStudioEditorBackend({ [editedPath]: editedProject });
    configureStudioEditorBackend(backend);
    try {
      await useStudioProjectStore.getState().loadProject(editedPath);
      const active = useStudioProjectStore.getState().project!;
      const command = createReplaceTextCommand(active, previewChapterReplacements(active, {
        query: "Antes",
        replacement: "Depois",
        caseSensitive: false,
        wholeWord: true,
      }));
      let releaseMutation!: () => void;
      let markMutationPersisted!: () => void;
      const mutationPersisted = new Promise<void>((resolve) => { markMutationPersisted = resolve; });
      const mutationGate = new Promise<void>((resolve) => { releaseMutation = resolve; });
      const mutateProject = backend.mutateProject.bind(backend);
      backend.mutateProject = async (config) => {
        const result = await mutateProject(config);
        markMutationPersisted();
        await mutationGate;
        return result;
      };

      const lateCommand = useStudioProjectStore.getState().executeChapterCommand(command);
      await mutationPersisted;
      await useStudioProjectStore.getState().importProjectJson(
        JSON.stringify({
          versao: "1.0",
          paginas: [{ numero: 1, textos: [{ id: "new", bbox: [0, 0, 1, 1], traduzido: "Projeto atual" }] }],
        }),
        "memory://active-after-command",
      );
      releaseMutation();
      await lateCommand;

      expect(useStudioProjectStore.getState().projectPath).toBe("memory://active-after-command");
      expect(useStudioProjectStore.getState().project?.paginas[0].text_layers[0].translated).toBe("Projeto atual");
      expect(useStudioProjectStore.getState().chapterHistory).toEqual([]);
    } finally {
      configureStudioEditorBackend(originalBackend);
    }
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

import { describe, expect, it, vi } from "vitest";
import { MemoryStudioEditorBackend } from "../../backend/memoryBackend";
import { importStudioProject } from "../../project/adapters";
import {
  createRecoverySnapshot,
  isRecoveryCandidate,
  parseStudioRecoverySnapshot,
  recoverStudioProject,
  runStudioAutosaveCycle,
} from "../recovery";

function projectFixture(text = "Texto salvo") {
  return importStudioProject({
    versao: "1.0",
    paginas: [{ numero: 1, textos: [{ id: "a", bbox: [0, 0, 10, 10], traduzido: text }] }],
  }).project;
}

describe("recuperacao automatica do Studio", () => {
  it("rejeita snapshots corrompidos antes de expor recuperacao na interface", () => {
    expect(parseStudioRecoverySnapshot(null)).toBeNull();
    expect(parseStudioRecoverySnapshot({ version: "1.0", savedAt: "ontem", project: {} })).toBeNull();
    const valid = createRecoverySnapshot("memory://valid", projectFixture(), 42);
    expect(parseStudioRecoverySnapshot(valid)).toEqual(valid);
    expect(parseStudioRecoverySnapshot(valid, "memory://outro-projeto")).toBeNull();
    expect(parseStudioRecoverySnapshot(valid, "memory://valid")).toEqual(valid);
  });

  it("recusa salvar um snapshot identificado como outro projeto", async () => {
    const backend = new MemoryStudioEditorBackend({ "memory://correto": projectFixture() });
    await expect(backend.saveRecoverySnapshot({
      project_path: "memory://correto",
      snapshot: createRecoverySnapshot("memory://incorreto", projectFixture("Outro"), 50),
    })).rejects.toThrow("outro projeto");
  });

  it("salva snapshots isolados e detecta apenas conteudo divergente", async () => {
    const backend = new MemoryStudioEditorBackend({ "memory://recovery": projectFixture() });
    const snapshot = createRecoverySnapshot("memory://recovery", projectFixture("Rascunho"), 1234);

    await backend.saveRecoverySnapshot({ project_path: "memory://recovery", snapshot });
    const loaded = await backend.loadRecoverySnapshot({ project_path: "memory://recovery" });

    expect(loaded).toEqual(snapshot);
    expect(loaded).not.toBe(snapshot);
    expect(isRecoveryCandidate(loaded, projectFixture())).toBe(true);
    expect(isRecoveryCandidate(loaded, projectFixture("Rascunho"))).toBe(false);
  });

  it("executa o autosave incremental antes de capturar o projeto persistido", async () => {
    const backend = new MemoryStudioEditorBackend({ "memory://cycle": projectFixture() });
    const runAutoSave = vi.fn(async () => {
      await backend.saveProjectJson({
        project_path: "memory://cycle",
        project_json: projectFixture("Edicao automatica"),
      });
    });

    const snapshot = await runStudioAutosaveCycle({
      backend,
      projectPath: "memory://cycle",
      dirty: true,
      runAutoSave,
      now: () => 5678,
    });

    expect(runAutoSave).toHaveBeenCalledOnce();
    expect(snapshot.project.paginas[0].text_layers[0].translated).toBe("Edicao automatica");
    expect(await backend.loadRecoverySnapshot({ project_path: "memory://cycle" })).toEqual(snapshot);
  });

  it("restaura o snapshot no project.json e limpa a sessao recuperada", async () => {
    const backend = new MemoryStudioEditorBackend({ "memory://restore": projectFixture() });
    await backend.saveRecoverySnapshot({
      project_path: "memory://restore",
      snapshot: createRecoverySnapshot("memory://restore", projectFixture("Recuperado"), 9999),
    });

    const restored = await recoverStudioProject(backend, "memory://restore");

    expect(restored?.paginas[0].text_layers[0].translated).toBe("Recuperado");
    expect((await backend.loadProject({ project_path: "memory://restore" })).paginas[0].text_layers[0].translated).toBe("Recuperado");
    expect(await backend.loadRecoverySnapshot({ project_path: "memory://restore" })).toBeNull();
  });
});

import { describe, expect, it } from "vitest";
import type { StudioLibrary } from "../../library/libraryModel";
import { createEmptyLibrary } from "../../library/libraryModel";
import {
  createDefaultLibraryBackend,
  createMemoryLibraryBackend,
  createTauriLibraryBackend,
  type LibraryBackend,
} from "../../library/libraryBackend";
import { createLibraryStore } from "../libraryStore";

function clone(document: StudioLibrary): StudioLibrary {
  return JSON.parse(JSON.stringify(document)) as StudioLibrary;
}

class FakeLibraryBackend implements LibraryBackend {
  document: StudioLibrary;
  saved: StudioLibrary[] = [];

  constructor(initial = createEmptyLibrary()) {
    this.document = clone(initial);
  }

  async load() {
    return clone(this.document);
  }

  async save(document: StudioLibrary) {
    this.document = clone(document);
    this.saved.push(clone(document));
  }
}

class GatedLibraryBackend extends FakeLibraryBackend {
  pending: Array<() => void> = [];
  activeWrites = 0;
  maxActiveWrites = 0;

  override async save(document: StudioLibrary) {
    this.activeWrites += 1;
    this.maxActiveWrites = Math.max(this.maxActiveWrites, this.activeWrites);
    this.saved.push(clone(document));
    await new Promise<void>((resolve) => {
      this.pending.push(() => {
        this.document = clone(document);
        this.activeWrites -= 1;
        resolve();
      });
    });
  }
}

async function nextTick() {
  await Promise.resolve();
  await Promise.resolve();
}

async function waitForSavedCount(backend: GatedLibraryBackend, count: number) {
  for (let attempt = 0; attempt < 20 && backend.saved.length < count; attempt += 1) {
    await nextTick();
  }
}

describe("libraryStore", () => {
  it("uses isolated memory storage outside Tauri", async () => {
    const initial = createEmptyLibrary();
    const backend = createMemoryLibraryBackend(initial);
    const loaded = await backend.load();
    loaded.works.push({
      id: "mutated",
      title: "Mutação externa",
      aliases: [],
      publicationStatus: "unknown",
      external: {},
      chapters: [],
    });

    expect((await backend.load()).works).toEqual([]);
    expect(createDefaultLibraryBackend(false)).toBeDefined();
  });

  it("uses the Studio Tauri commands and normalizes their payload", async () => {
    const calls: Array<{ command: string; args?: Record<string, unknown> }> = [];
    const invoke = async <T>(command: string, args?: Record<string, unknown>) => {
      calls.push({ command, args });
      if (command === "studio_load_library") {
        return { document: { works: [] }, recoveredFromBackup: false } as T;
      }
      return undefined as T;
    };
    const backend = createTauriLibraryBackend(invoke);
    const document = await backend.load();
    await backend.save(document);

    expect(document).toEqual(createEmptyLibrary());
    expect(calls).toEqual([
      { command: "studio_load_library", args: undefined },
      { command: "studio_save_library", args: { document } },
    ]);
  });

  it("loads, adds and persists the selected work", async () => {
    const backend = new FakeLibraryBackend();
    const store = createLibraryStore(backend);

    await store.getState().load();
    await store.getState().addWork({ id: "work-1", title: "Obra", aliases: [] });
    await store.getState().selectWork("work-1");

    expect(store.getState().status).toBe("ready");
    expect(backend.document.selectedWorkId).toBe("work-1");
    expect(backend.document.works[0]).toMatchObject({
      id: "work-1",
      title: "Obra",
      publicationStatus: "unknown",
    });
  });

  it("serializes writes so an older save cannot overwrite a newer one", async () => {
    const backend = new GatedLibraryBackend();
    const store = createLibraryStore(backend);
    await store.getState().load();

    const first = store.getState().addWork({ id: "work-1", title: "Primeira", aliases: [] });
    await waitForSavedCount(backend, 1);
    const second = store.getState().addWork({ id: "work-2", title: "Segunda", aliases: [] });
    await nextTick();

    expect(backend.saved).toHaveLength(1);
    expect(backend.maxActiveWrites).toBe(1);

    backend.pending.shift()?.();
    await waitForSavedCount(backend, 2);
    expect(backend.saved).toHaveLength(2);
    expect(backend.maxActiveWrites).toBe(1);

    backend.pending.shift()?.();
    await Promise.all([first, second]);
    expect(backend.document.works.map((work) => work.id)).toEqual(["work-1", "work-2"]);
  });

  it("keeps the in-memory document when persistence fails", async () => {
    const backend: LibraryBackend = {
      load: async () => createEmptyLibrary(),
      save: async () => {
        throw new Error("disco indisponível");
      },
    };
    const store = createLibraryStore(backend);
    await store.getState().load();

    await store.getState().addWork({ id: "work-1", title: "Não perder", aliases: [] });

    expect(store.getState().document.works[0].title).toBe("Não perder");
    expect(store.getState().status).toBe("error");
    expect(store.getState().error).toBe("disco indisponível");
  });

  it("persists chapter registration and home view preferences", async () => {
    const backend = new FakeLibraryBackend();
    const store = createLibraryStore(backend);
    await store.getState().load();
    await store.getState().addWork({ id: "work-1", title: "Obra", aliases: [] });

    await store.getState().upsertChapter("work-1", {
      id: "chapter-1",
      label: "1",
      projectPath: "N:/Obra/001/project.json",
    });
    await store.getState().addWork({ id: "work-1", title: "Obra renomeada", aliases: [] });
    await store.getState().setChapterView("list");
    await store.getState().setThumbnailSize(220);

    expect(backend.document.works[0].chapters[0].label).toBe("1");
    expect(backend.document.works[0].title).toBe("Obra renomeada");
    expect(backend.document.preferences).toEqual({ chapterView: "list", thumbnailSize: 220 });
  });

  it("renames and removes a work without touching its attached chapter files", async () => {
    const backend = new FakeLibraryBackend();
    const store = createLibraryStore(backend);
    await store.getState().load();
    await store.getState().addWork({ id: "work-1", title: "Nome antigo", aliases: [] });
    await store.getState().upsertChapter("work-1", {
      id: "chapter-1",
      label: "1",
      projectPath: "N:/Obra/001/project.json",
    });

    await store.getState().addWork({ id: "work-1", title: "Nome novo", aliases: ["Alias"] });
    expect(backend.document.works[0].chapters[0].projectPath).toBe("N:/Obra/001/project.json");

    await store.getState().removeWork("work-1");
    expect(backend.document.works).toEqual([]);
  });

  it("does not duplicate project paths and can relink or remove only a chapter reference", async () => {
    const backend = new FakeLibraryBackend();
    const store = createLibraryStore(backend);
    await store.getState().load();
    await store.getState().addWork({ id: "work-1", title: "Obra", aliases: [] });
    await store.getState().upsertChapter("work-1", {
      id: "chapter-1",
      label: "1",
      projectPath: "N:/Obra/001/project.json",
    });
    await store.getState().upsertChapter("work-1", {
      id: "chapter-duplicated",
      label: "1 atualizado",
      projectPath: "n:\\obra\\001\\project.json",
    });

    expect(backend.document.works[0].chapters).toHaveLength(1);
    await store.getState().relinkChapter("work-1", "chapter-1", "N:/Obra/movido/project.json");
    expect(backend.document.works[0].chapters[0].projectPath).toBe("N:/Obra/movido/project.json");

    await store.getState().removeChapter("work-1", "chapter-1");
    expect(backend.document.works[0].chapters).toEqual([]);
  });
});

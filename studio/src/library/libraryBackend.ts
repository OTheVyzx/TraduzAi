import { invoke as tauriInvoke } from "@tauri-apps/api/core";
import { createEmptyLibrary, normalizeLibrary, type StudioLibrary } from "./libraryModel";

export interface LibraryBackend {
  load(): Promise<StudioLibrary>;
  save(document: StudioLibrary): Promise<void>;
}

export type LibraryInvoke = <T>(command: string, args?: Record<string, unknown>) => Promise<T>;

interface LibraryLoadPayload {
  document: unknown;
  recoveredFromBackup?: boolean;
}

function cloneLibrary(document: StudioLibrary): StudioLibrary {
  return JSON.parse(JSON.stringify(document)) as StudioLibrary;
}

export function createMemoryLibraryBackend(initial = createEmptyLibrary()): LibraryBackend {
  let document = cloneLibrary(normalizeLibrary(initial));

  return {
    async load() {
      return cloneLibrary(document);
    },
    async save(nextDocument) {
      document = cloneLibrary(normalizeLibrary(nextDocument));
    },
  };
}

export function createTauriLibraryBackend(invoke: LibraryInvoke = tauriInvoke): LibraryBackend {
  return {
    async load() {
      const payload = await invoke<LibraryLoadPayload>("studio_load_library");
      return normalizeLibrary(payload.document);
    },
    async save(document) {
      await invoke<void>("studio_save_library", { document: normalizeLibrary(document) });
    },
  };
}

function isTauriRuntime() {
  return typeof window !== "undefined" && ("__TAURI_INTERNALS__" in window || "__TAURI__" in window);
}

export function createDefaultLibraryBackend(tauriRuntime = isTauriRuntime()): LibraryBackend {
  return tauriRuntime ? createTauriLibraryBackend() : createMemoryLibraryBackend();
}

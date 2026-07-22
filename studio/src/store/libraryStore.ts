import { createStore, type StoreApi } from "zustand/vanilla";
import type { LibraryBackend } from "../library/libraryBackend";
import {
  normalizeLibrary,
  type ExternalWorkLink,
  type LibraryWork,
  type PublicationStatus,
  type StudioLibrary,
} from "../library/libraryModel";

export type LibraryStoreStatus = "idle" | "loading" | "saving" | "ready" | "error";

export interface AddLibraryWorkInput {
  id: string;
  title: string;
  aliases: string[];
  coverPath?: string | null;
  publicationStatus?: PublicationStatus;
  external?: ExternalWorkLink;
}

export interface LibraryStoreState {
  status: LibraryStoreStatus;
  error: string | null;
  document: StudioLibrary;
  load(): Promise<void>;
  addWork(work: AddLibraryWorkInput): Promise<void>;
  selectWork(workId: string): Promise<void>;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function createLibraryStore(backend: LibraryBackend): StoreApi<LibraryStoreState> {
  let saveQueue: Promise<void> = Promise.resolve();
  let latestRevision = 0;

  return createStore<LibraryStoreState>((set, get) => {
    async function persist(document: StudioLibrary) {
      const normalized = normalizeLibrary(document);
      const revision = ++latestRevision;
      set({ document: normalized, status: "saving", error: null });

      const operation = saveQueue
        .catch(() => undefined)
        .then(() => backend.save(normalized));
      saveQueue = operation;

      try {
        await operation;
        if (revision === latestRevision) set({ status: "ready", error: null });
      } catch (error) {
        if (revision === latestRevision) set({ status: "error", error: errorMessage(error) });
      }
    }

    return {
      status: "idle",
      error: null,
      document: normalizeLibrary(null),

      load: async () => {
        set({ status: "loading", error: null });
        try {
          const document = normalizeLibrary(await backend.load());
          latestRevision += 1;
          set({ document, status: "ready", error: null });
        } catch (error) {
          set({ status: "error", error: errorMessage(error) });
        }
      },

      addWork: async (input) => {
        const current = get().document;
        const work: LibraryWork = {
          id: input.id,
          title: input.title,
          aliases: input.aliases,
          ...(input.coverPath === undefined ? {} : { coverPath: input.coverPath }),
          publicationStatus: input.publicationStatus ?? "unknown",
          external: input.external ?? {},
          chapters: [],
        };
        const existingIndex = current.works.findIndex((candidate) => candidate.id === work.id);
        const works = [...current.works];
        if (existingIndex >= 0) works[existingIndex] = { ...works[existingIndex], ...work };
        else works.push(work);

        await persist({
          ...current,
          selectedWorkId: current.selectedWorkId ?? work.id,
          works,
        });
      },

      selectWork: async (workId) => {
        const current = get().document;
        if (!current.works.some((work) => work.id === workId)) return;
        await persist({ ...current, selectedWorkId: workId });
      },
    };
  });
}

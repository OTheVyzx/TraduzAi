import { createStore, type StoreApi } from "zustand/vanilla";
import type { LibraryBackend } from "../library/libraryBackend";
import {
  normalizeLibrary,
  upsertChapter,
  type ExternalWorkLink,
  type LibraryChapter,
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
  removeWork(workId: string): Promise<void>;
  selectWork(workId: string): Promise<void>;
  upsertChapter(workId: string, chapter: LibraryChapter): Promise<void>;
  removeChapter(workId: string, chapterId: string): Promise<void>;
  relinkChapter(workId: string, chapterId: string, projectPath: string): Promise<void>;
  setChapterView(view: "grid" | "list"): Promise<void>;
  setThumbnailSize(size: number): Promise<void>;
  setTrackingLanguage(language: string): Promise<void>;
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
        if (existingIndex >= 0) {
          works[existingIndex] = {
            ...works[existingIndex],
            ...work,
            chapters: works[existingIndex].chapters,
          };
        }
        else works.push(work);

        await persist({
          ...current,
          selectedWorkId: current.selectedWorkId ?? work.id,
          works,
        });
      },

      removeWork: async (workId) => {
        const current = get().document;
        const works = current.works.filter((work) => work.id !== workId);
        if (works.length === current.works.length) return;
        await persist({
          ...current,
          works,
          selectedWorkId: current.selectedWorkId === workId
            ? works[0]?.id ?? null
            : current.selectedWorkId,
        });
      },

      selectWork: async (workId) => {
        const current = get().document;
        if (!current.works.some((work) => work.id === workId)) return;
        await persist({ ...current, selectedWorkId: workId });
      },

      upsertChapter: async (workId, chapter) => {
        await persist(upsertChapter(get().document, workId, chapter));
      },

      removeChapter: async (workId, chapterId) => {
        const current = get().document;
        const workIndex = current.works.findIndex((work) => work.id === workId);
        if (workIndex < 0) return;
        const chapters = current.works[workIndex].chapters.filter((chapter) => chapter.id !== chapterId);
        if (chapters.length === current.works[workIndex].chapters.length) return;
        const works = [...current.works];
        works[workIndex] = { ...works[workIndex], chapters };
        await persist({ ...current, works });
      },

      relinkChapter: async (workId, chapterId, projectPath) => {
        const current = get().document;
        const workIndex = current.works.findIndex((work) => work.id === workId);
        if (workIndex < 0 || !projectPath.trim()) return;
        const chapterIndex = current.works[workIndex].chapters.findIndex((chapter) => chapter.id === chapterId);
        if (chapterIndex < 0) return;
        const chapters = [...current.works[workIndex].chapters];
        chapters[chapterIndex] = { ...chapters[chapterIndex], projectPath: projectPath.trim() };
        const works = [...current.works];
        works[workIndex] = { ...works[workIndex], chapters };
        await persist({ ...current, works });
      },

      setChapterView: async (view) => {
        const current = get().document;
        await persist({
          ...current,
          preferences: { ...current.preferences, chapterView: view },
        });
      },

      setThumbnailSize: async (size) => {
        const current = get().document;
        await persist({
          ...current,
          preferences: { ...current.preferences, thumbnailSize: size },
        });
      },

      setTrackingLanguage: async (language) => {
        const current = get().document;
        const normalized = language.trim().toLocaleLowerCase("en-US") || "en";
        await persist({
          ...current,
          preferences: { ...current.preferences, trackingLanguage: normalized },
        });
      },
    };
  });
}

export interface ReaderRepository {
  id: string;
  url: string;
  fingerprint: string;
  trustedAt: string;
}

export interface ReaderExtension {
  packageName: string;
  name: string;
  versionCode: number;
  versionName: string;
  enabled: boolean;
  sourceIds: string[];
}

export interface ReaderChapter {
  id: string;
  url: string;
  name: string;
  read: boolean;
  bookmarked: boolean;
  lastPageRead: number;
  pageCount: number | null;
  chapterNumber?: number;
  scanlator?: string;
  dateUpload?: number;
  memo?: Record<string, unknown>;
}

export interface ReaderManga {
  id: string;
  extensionPackage: string;
  sourceId: string;
  sourceName: string;
  mangaUrl: string;
  title: string;
  thumbnailUrl?: string;
  artist?: string;
  author?: string;
  description?: string;
  genres?: string[];
  status?: number;
  chapters: ReaderChapter[];
  categoryIds: string[];
  favorite: boolean;
  autoUpdate: boolean;
  autoDownload: boolean;
}

export function normalizeGenres(value: string | undefined): string[] {
  if (!value) return [];
  return [...new Set(value.split(",").map((genre) => genre.trim()).filter(Boolean))];
}

export interface ReaderCategory {
  id: string;
  name: string;
  order: number;
}

export interface ReaderDocument {
  version: 1;
  repositories: ReaderRepository[];
  extensions: ReaderExtension[];
  manga: ReaderManga[];
  categories: ReaderCategory[];
  updatedAt: string;
}

export function createEmptyReaderDocument(): ReaderDocument {
  return {
    version: 1,
    repositories: [],
    extensions: [],
    manga: [],
    categories: [],
    updatedAt: new Date(0).toISOString(),
  };
}

export function parseReaderDocument(value: unknown): ReaderDocument {
  if (!value || typeof value !== "object") return createEmptyReaderDocument();
  const candidate = value as Partial<ReaderDocument>;
  if (candidate.version !== 1) return createEmptyReaderDocument();
  const manga = Array.isArray(candidate.manga) ? candidate.manga.filter(isReaderManga) : [];
  return {
    version: 1,
    repositories: Array.isArray(candidate.repositories) ? candidate.repositories : [],
    extensions: Array.isArray(candidate.extensions) ? candidate.extensions : [],
    categories: Array.isArray(candidate.categories) ? candidate.categories : [],
    manga,
    updatedAt: typeof candidate.updatedAt === "string" ? candidate.updatedAt : new Date(0).toISOString(),
  };
}

function isReaderManga(value: unknown): value is ReaderManga {
  if (!value || typeof value !== "object") return false;
  const manga = value as Partial<ReaderManga>;
  return typeof manga.id === "string"
    && typeof manga.extensionPackage === "string"
    && typeof manga.sourceId === "string"
    && typeof manga.mangaUrl === "string"
    && typeof manga.title === "string"
    && Array.isArray(manga.chapters);
}

function touched(document: ReaderDocument): ReaderDocument {
  return { ...document, updatedAt: new Date().toISOString() };
}

export function addReaderManga(document: ReaderDocument, manga: ReaderManga): ReaderDocument {
  const identity = `${manga.extensionPackage}\u0000${manga.sourceId}\u0000${manga.mangaUrl}`;
  const withoutDuplicate = document.manga.filter((item) =>
    `${item.extensionPackage}\u0000${item.sourceId}\u0000${item.mangaUrl}` !== identity,
  );
  return touched({ ...document, manga: [...withoutDuplicate, manga] });
}

export function markReaderProgress(
  document: ReaderDocument,
  mangaId: string,
  chapterId: string,
  lastPageRead: number,
): ReaderDocument {
  return touched({
    ...document,
    manga: document.manga.map((manga) => manga.id !== mangaId ? manga : {
      ...manga,
      chapters: manga.chapters.map((chapter) => chapter.id !== chapterId ? chapter : {
        ...chapter,
        lastPageRead: Math.max(0, Math.min(lastPageRead, chapter.pageCount ?? lastPageRead)),
        read: chapter.pageCount !== null && lastPageRead >= chapter.pageCount,
      }),
    }),
  });
}

export function toggleReaderBookmark(
  document: ReaderDocument,
  mangaId: string,
  chapterId: string,
): ReaderDocument {
  return touched({
    ...document,
    manga: document.manga.map((manga) => manga.id !== mangaId ? manga : {
      ...manga,
      chapters: manga.chapters.map((chapter) => chapter.id !== chapterId ? chapter : {
        ...chapter,
        bookmarked: !chapter.bookmarked,
      }),
    }),
  });
}

export const READER_STORAGE_KEY = "traduzai.studio.reader.v1";

export function loadReaderDocument(storage: Pick<Storage, "getItem"> | null): ReaderDocument {
  if (!storage) return createEmptyReaderDocument();
  try {
    const raw = storage.getItem(READER_STORAGE_KEY);
    return raw ? parseReaderDocument(JSON.parse(raw)) : createEmptyReaderDocument();
  } catch {
    return createEmptyReaderDocument();
  }
}

export function saveReaderDocument(storage: Pick<Storage, "setItem"> | null, document: ReaderDocument) {
  storage?.setItem(READER_STORAGE_KEY, JSON.stringify(document));
}

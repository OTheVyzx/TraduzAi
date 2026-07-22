export const STUDIO_LIBRARY_SCHEMA_VERSION = 1 as const;

export type PublicationStatus =
  | "releasing"
  | "hiatus"
  | "completed"
  | "cancelled"
  | "not_yet_released"
  | "unknown";

export type ChapterWorkflowStatus = "pending" | "translating" | "editing" | "review" | "completed";

export interface LibraryChapter {
  id: string;
  label: string;
  title?: string;
  projectPath: string;
  coverPath?: string | null;
  pageCount?: number;
  completedPages?: number;
  workflowStatus?: ChapterWorkflowStatus;
  lastOpenedAt?: string | null;
}

export interface ExternalWorkLink {
  anilistId?: number;
  mangaDexId?: string;
  canonicalUrl?: string;
  manualStatusOverride?: PublicationStatus | null;
  tracking?: WorkTrackingCache;
}

export interface ExternalTrackingSnapshot {
  provider: string;
  providerId: string;
  title: string;
  status: PublicationStatus;
  remoteChapterCount: number | null;
  latestChapter: string | null;
  coverUrl: string | null;
  siteUrl: string | null;
  fetchedAt: string;
}

export interface WorkTrackingCache {
  snapshots: ExternalTrackingSnapshot[];
  fetchedAt: string;
  expiresAt: string;
  lastError: string | null;
}

export interface LibraryWork {
  id: string;
  title: string;
  aliases: string[];
  coverPath?: string | null;
  publicationStatus: PublicationStatus;
  external: ExternalWorkLink;
  chapters: LibraryChapter[];
}

export interface StudioLibrary {
  schemaVersion: typeof STUDIO_LIBRARY_SCHEMA_VERSION;
  selectedWorkId: string | null;
  works: LibraryWork[];
  preferences: {
    chapterView: "grid" | "list";
    thumbnailSize: number;
    trackingLanguage: string;
  };
}

const PUBLICATION_STATUSES = new Set<PublicationStatus>([
  "releasing",
  "hiatus",
  "completed",
  "cancelled",
  "not_yet_released",
  "unknown",
]);

const WORKFLOW_STATUSES = new Set<ChapterWorkflowStatus>([
  "pending",
  "translating",
  "editing",
  "review",
  "completed",
]);

const DEFAULT_THUMBNAIL_SIZE = 176;
const MIN_THUMBNAIL_SIZE = 112;
const MAX_THUMBNAIL_SIZE = 240;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asTrimmedString(value: unknown, fallback = ""): string {
  return typeof value === "string" && value.trim() ? value.trim() : fallback;
}

function asOptionalString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function normalizePublicationStatus(value: unknown): PublicationStatus {
  return typeof value === "string" && PUBLICATION_STATUSES.has(value as PublicationStatus)
    ? (value as PublicationStatus)
    : "unknown";
}

function normalizeWorkflowStatus(value: unknown): ChapterWorkflowStatus | undefined {
  return typeof value === "string" && WORKFLOW_STATUSES.has(value as ChapterWorkflowStatus)
    ? (value as ChapterWorkflowStatus)
    : undefined;
}

function normalizeExternalLink(value: unknown): ExternalWorkLink {
  if (!isRecord(value)) return {};

  const anilistId = typeof value.anilistId === "number" && Number.isFinite(value.anilistId) ? value.anilistId : undefined;
  const mangaDexId = asOptionalString(value.mangaDexId);
  const canonicalUrl = asOptionalString(value.canonicalUrl);
  const override = value.manualStatusOverride;
  const manualStatusOverride =
    override === null
      ? null
      : typeof override === "string" && PUBLICATION_STATUSES.has(override as PublicationStatus)
        ? (override as PublicationStatus)
        : undefined;
  const tracking = normalizeTrackingCache(value.tracking);

  return {
    ...(anilistId === undefined ? {} : { anilistId }),
    ...(mangaDexId === undefined ? {} : { mangaDexId }),
    ...(canonicalUrl === undefined ? {} : { canonicalUrl }),
    ...(manualStatusOverride === undefined ? {} : { manualStatusOverride }),
    ...(tracking === undefined ? {} : { tracking }),
  };
}

function normalizeNullableString(value: unknown): string | null {
  return asOptionalString(value) ?? null;
}

function normalizeTrackingSnapshot(value: unknown): ExternalTrackingSnapshot | null {
  if (!isRecord(value)) return null;
  const provider = asOptionalString(value.provider);
  const providerId = asOptionalString(value.providerId);
  const title = asOptionalString(value.title);
  const fetchedAt = asOptionalString(value.fetchedAt);
  if (!provider || !providerId || !title || !fetchedAt) return null;

  return {
    provider,
    providerId,
    title,
    status: normalizePublicationStatus(value.status),
    remoteChapterCount: typeof value.remoteChapterCount === "number" && Number.isFinite(value.remoteChapterCount)
      ? Math.max(0, value.remoteChapterCount)
      : null,
    latestChapter: normalizeNullableString(value.latestChapter),
    coverUrl: normalizeNullableString(value.coverUrl),
    siteUrl: normalizeNullableString(value.siteUrl),
    fetchedAt,
  };
}

function normalizeTrackingCache(value: unknown): WorkTrackingCache | undefined {
  if (!isRecord(value)) return undefined;
  const fetchedAt = asOptionalString(value.fetchedAt);
  const expiresAt = asOptionalString(value.expiresAt);
  if (!fetchedAt || !expiresAt) return undefined;
  const snapshots = Array.isArray(value.snapshots)
    ? value.snapshots
        .map(normalizeTrackingSnapshot)
        .filter((snapshot): snapshot is ExternalTrackingSnapshot => snapshot !== null)
    : [];

  return {
    snapshots,
    fetchedAt,
    expiresAt,
    lastError: normalizeNullableString(value.lastError),
  };
}

function normalizeChapter(value: unknown, index: number): LibraryChapter | null {
  if (!isRecord(value)) return null;

  const label = asTrimmedString(value.label, String(index + 1));
  const projectPath = asTrimmedString(value.projectPath);
  if (!projectPath) return null;

  const pageCount = typeof value.pageCount === "number" && Number.isFinite(value.pageCount)
    ? Math.max(0, Math.trunc(value.pageCount))
    : undefined;
  const completedPages = typeof value.completedPages === "number" && Number.isFinite(value.completedPages)
    ? Math.max(0, Math.trunc(value.completedPages))
    : undefined;
  const workflowStatus = normalizeWorkflowStatus(value.workflowStatus);

  return {
    id: asTrimmedString(value.id, `chapter-${index + 1}`),
    label,
    projectPath,
    ...(asOptionalString(value.title) ? { title: asOptionalString(value.title) } : {}),
    ...(value.coverPath === null ? { coverPath: null } : asOptionalString(value.coverPath) ? { coverPath: asOptionalString(value.coverPath) } : {}),
    ...(pageCount === undefined ? {} : { pageCount }),
    ...(completedPages === undefined ? {} : { completedPages }),
    ...(workflowStatus === undefined ? {} : { workflowStatus }),
    ...(value.lastOpenedAt === null
      ? { lastOpenedAt: null }
      : asOptionalString(value.lastOpenedAt)
        ? { lastOpenedAt: asOptionalString(value.lastOpenedAt) }
        : {}),
  };
}

function normalizeWork(value: unknown, index: number): LibraryWork | null {
  if (!isRecord(value)) return null;

  const chapters = Array.isArray(value.chapters)
    ? value.chapters
        .map((chapter, chapterIndex) => normalizeChapter(chapter, chapterIndex))
        .filter((chapter): chapter is LibraryChapter => chapter !== null)
    : [];

  return {
    id: asTrimmedString(value.id, `work-${index + 1}`),
    title: asTrimmedString(value.title, "Obra sem título"),
    aliases: Array.isArray(value.aliases)
      ? value.aliases.map((alias) => asOptionalString(alias)).filter((alias): alias is string => Boolean(alias))
      : [],
    ...(value.coverPath === null ? { coverPath: null } : asOptionalString(value.coverPath) ? { coverPath: asOptionalString(value.coverPath) } : {}),
    publicationStatus: normalizePublicationStatus(value.publicationStatus),
    external: normalizeExternalLink(value.external),
    chapters: sortChapterEntries(chapters),
  };
}

export function createEmptyLibrary(): StudioLibrary {
  return {
    schemaVersion: STUDIO_LIBRARY_SCHEMA_VERSION,
    selectedWorkId: null,
    works: [],
    preferences: {
      chapterView: "grid",
      thumbnailSize: DEFAULT_THUMBNAIL_SIZE,
      trackingLanguage: "en",
    },
  };
}

export function normalizeLibrary(value: unknown): StudioLibrary {
  if (!isRecord(value)) return createEmptyLibrary();

  const works = Array.isArray(value.works)
    ? value.works.map((work, index) => normalizeWork(work, index)).filter((work): work is LibraryWork => work !== null)
    : [];
  const preferences = isRecord(value.preferences) ? value.preferences : {};
  const requestedSelection = asOptionalString(value.selectedWorkId);
  const selectedWorkId = requestedSelection && works.some((work) => work.id === requestedSelection)
    ? requestedSelection
    : works[0]?.id ?? null;
  const rawThumbnailSize = typeof preferences.thumbnailSize === "number" && Number.isFinite(preferences.thumbnailSize)
    ? Math.round(preferences.thumbnailSize)
    : DEFAULT_THUMBNAIL_SIZE;

  return {
    schemaVersion: STUDIO_LIBRARY_SCHEMA_VERSION,
    selectedWorkId,
    works,
    preferences: {
      chapterView: preferences.chapterView === "list" ? "list" : "grid",
      thumbnailSize: Math.min(MAX_THUMBNAIL_SIZE, Math.max(MIN_THUMBNAIL_SIZE, rawThumbnailSize)),
      trackingLanguage: asOptionalString(preferences.trackingLanguage) ?? "en",
    },
  };
}

function chapterNumber(label: string): number | null {
  const normalized = label.trim().replace(",", ".");
  if (!/^\d+(?:\.\d+)?$/.test(normalized)) return null;

  const parsed = Number.parseFloat(normalized);
  return Number.isFinite(parsed) ? parsed : null;
}

export function sortChapterEntries<T extends Pick<LibraryChapter, "label">>(entries: readonly T[]): T[] {
  return [...entries].sort((left, right) => {
    const leftNumber = chapterNumber(left.label);
    const rightNumber = chapterNumber(right.label);
    if (leftNumber !== null && rightNumber !== null) return leftNumber - rightNumber;
    if (leftNumber !== null) return -1;
    if (rightNumber !== null) return 1;
    return left.label.localeCompare(right.label, "pt-BR", { numeric: true, sensitivity: "base" });
  });
}

function comparablePath(path: string): string {
  return path.trim().replace(/\\/g, "/").replace(/\/+$/, "").toLocaleLowerCase("en-US");
}

export function upsertChapter(
  library: StudioLibrary,
  workId: string,
  chapter: LibraryChapter,
): StudioLibrary {
  const workIndex = library.works.findIndex((work) => work.id === workId);
  const targetWork: LibraryWork = workIndex >= 0
    ? library.works[workIndex]
    : {
        id: workId,
        title: "Obra sem título",
        aliases: [],
        publicationStatus: "unknown",
        external: {},
        chapters: [],
      };
  const existingIndex = targetWork.chapters.findIndex(
    (candidate) => comparablePath(candidate.projectPath) === comparablePath(chapter.projectPath),
  );
  const chapters = [...targetWork.chapters];
  if (existingIndex >= 0) {
    chapters[existingIndex] = { ...chapters[existingIndex], ...chapter, id: chapters[existingIndex].id };
  } else {
    chapters.push(chapter);
  }
  const updatedWork = { ...targetWork, chapters: sortChapterEntries(chapters) };
  const works = [...library.works];
  if (workIndex >= 0) works[workIndex] = updatedWork;
  else works.push(updatedWork);

  return {
    ...library,
    selectedWorkId: library.selectedWorkId ?? workId,
    works,
  };
}

export function chapterProgress(chapter: Pick<LibraryChapter, "pageCount" | "completedPages">): number {
  const pageCount = Math.max(0, chapter.pageCount ?? 0);
  if (pageCount === 0) return 0;
  const completedPages = Math.min(pageCount, Math.max(0, chapter.completedPages ?? 0));
  return Math.round((completedPages / pageCount) * 100);
}

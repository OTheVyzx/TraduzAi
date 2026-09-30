import { invoke } from "@tauri-apps/api/core";
import type { ReaderCategory, ReaderManga } from "./readerModel";

export interface SourceRuntimeStatus {
  configured: boolean;
  running: boolean;
  protocol: number;
  sandbox: "required" | string;
}

export interface SourceSearchResult {
  v: number;
  id: string;
  ok: boolean;
  result: {
    sourceId: string;
    sourceName: string;
    manga: Array<{
      sourceId: string;
      url: string;
      title: string;
      thumbnailUrl?: string;
      status: number;
      memo: Record<string, unknown>;
    }>;
  };
}

export type SourceCatalogResult = SourceSearchResult;

export type SourceFilter =
  | { path: number[]; type: "header"; name: string }
  | { path: number[]; type: "separator"; name: string }
  | { path: number[]; type: "select"; name: string; values: string[]; value: number }
  | { path: number[]; type: "text"; name: string; value: string }
  | { path: number[]; type: "checkbox"; name: string; value: boolean }
  | { path: number[]; type: "tristate"; name: string; value: number }
  | { path: number[]; type: "sort"; name: string; values: string[]; value: { index: number; ascending: boolean } | null }
  | { path: number[]; type: "group"; name: string; children: SourceFilter[] };

export interface SourceFilterChange { path: number[]; value: unknown }
export interface SourceFiltersResult { result: { filters: SourceFilter[] } }

export interface RuntimeManga {
  sourceId?: string;
  url: string;
  title: string;
  thumbnailUrl?: string;
  artist?: string;
  author?: string;
  status?: number;
  description?: string;
  genre?: string;
  initialized?: boolean;
  memo?: Record<string, unknown>;
}

export interface RuntimeChapter {
  url: string;
  name: string;
  chapterNumber?: number;
  scanlator?: string;
  dateUpload?: number;
  memo?: Record<string, unknown>;
}

export interface MangaDetailsResult { result: { manga: RuntimeManga } }
export interface ChaptersResult { result: { chapters: RuntimeChapter[] } }

export interface RepositoryPreviewResult {
  v: number;
  id: string;
  ok: boolean;
  result: {
    url: string;
    certificateFingerprint: string;
    indexSha256: string;
    signingKey?: string;
    extensions: Array<{
      name: string;
      packageName: string;
      versionCode: number;
      versionName: string;
      lang: string;
      nsfw?: number;
      iconUrl?: string;
      jarUrl?: string;
      apkUrl?: string;
      sources: Array<{ name: string; lang: string; id: string; baseUrl?: string }>;
    }>;
  };
}

export interface RepositoryListResult {
  v: number;
  result: { repositories: Array<{ id: string; url: string; certificateFingerprint: string; indexSha256: string; signingKey?: string }> };
}

export interface RepositoryMutationResult {
  v: number;
  id: string;
  ok: boolean;
  result: { id: string; url: string; certificateFingerprint: string; indexSha256: string };
}

export interface RepositoryExtension {
  repositoryId: string;
  name: string;
  packageName: string;
  versionCode: number;
  versionName: string;
  lang: string;
  nsfw?: number;
  iconUrl?: string;
  jarUrl?: string;
  apkUrl?: string;
  sha256?: string;
  sources: Array<{ id: string; name: string; lang: string; baseUrl?: string }>;
}

export interface InstalledExtension {
  packageName: string;
  name: string;
  versionCode: number;
  versionName: string;
  repositoryId: string;
  sha256: string;
  enabled: boolean;
  rollbackAvailable: boolean;
}

export interface ExtensionListResult {
  v: number;
  id: string;
  ok: boolean;
  result: { extensions: InstalledExtension[] };
}

export interface ReaderLibraryResult {
  result: { manga: ReaderManga[] };
}

export interface ReaderMangaMutationResult {
  result: { manga: ReaderManga };
}

export interface ReaderCategoriesResult {
  result: { categories: ReaderCategory[] };
}

export interface ReaderAutomationSettings {
  enabled: boolean;
  intervalHours: number;
  lastRunAt?: string;
  lastRunSummary?: { checked: number; updated: number; failed: number };
}

export interface ReaderAutomationResult {
  result: { automation: ReaderAutomationSettings };
}

export interface ReaderAutomationRunResult {
  result: { ranAt: string; summary: { checked: number; updated: number; failed: number; failures: Array<{ mangaId: string; message: string }> } };
}

export interface ReaderDownloadResult {
  result: {
    jobId: string;
    status: "queued" | "running" | "paused" | "retrying" | "completed" | "failed";
    pageCount: number;
    downloadedPages?: number;
    totalBytes?: number;
    attempt?: number;
    error?: string;
    pages: Array<{ id: string; number: number; mime: string; size: number; sha256: string; width: number; height: number }>;
  };
}

export type ReaderDownloadRecord = ReaderDownloadResult["result"] & { mangaId: string; chapterId: string };

export interface ReaderDownloadListResult {
  result: { downloads: ReaderDownloadRecord[] };
}

export interface ReaderHistoryEntry {
  mangaId: string;
  chapterId: string;
  lastPage: number;
  pageCount: number | null;
  read: boolean;
  updatedAt: string;
}

export interface ReaderHistoryResult {
  result: { history: ReaderHistoryEntry[] };
}

export interface RepositoryRefreshResult {
  v: number;
  id: string;
  ok: boolean;
  result: { extensions: RepositoryExtension[] };
}

export interface SourceRuntimeClient {
  status(): Promise<SourceRuntimeStatus>;
  start(): Promise<unknown>;
  shutdown(): Promise<boolean>;
  previewRepository(url: string): Promise<RepositoryPreviewResult>;
  addRepository(input: { url: string; certificateFingerprint: string; indexSha256: string; signingKey?: string }): Promise<RepositoryMutationResult>;
  listRepositories(): Promise<RepositoryListResult>;
  refreshRepositories(): Promise<RepositoryRefreshResult>;
  listExtensions(): Promise<ExtensionListResult>;
  installExtension(config: { repositoryId: string; packageName: string; versionCode: number }): Promise<unknown>;
  setExtensionEnabled(config: { packageName: string; enabled: boolean }): Promise<unknown>;
  uninstallExtension(packageName: string): Promise<unknown>;
  rollbackExtension(packageName: string): Promise<unknown>;
  removeRepository(id: string): Promise<unknown>;
  filters(config: { extensionPackage: string; sourceId: string }): Promise<SourceFiltersResult>;
  search(config: { extensionPackage: string; query: string; sourceId?: string; filters?: SourceFilterChange[] }): Promise<SourceSearchResult>;
  popular(config: { extensionPackage: string; sourceId: string; page?: number }): Promise<SourceCatalogResult>;
  latest(config: { extensionPackage: string; sourceId: string; page?: number }): Promise<SourceCatalogResult>;
  mangaDetails(config: { extensionPackage: string; sourceId: string; manga: RuntimeManga }): Promise<MangaDetailsResult>;
  chapters(config: { extensionPackage: string; sourceId: string; manga: RuntimeManga }): Promise<ChaptersResult>;
  pages(config: { extensionPackage: string; sourceId: string; chapter: RuntimeChapter }): Promise<unknown>;
  listReaderLibrary(): Promise<ReaderLibraryResult>;
  addReaderManga(manga: ReaderManga): Promise<ReaderMangaMutationResult>;
  updateReaderManga(manga: ReaderManga): Promise<ReaderMangaMutationResult>;
  removeReaderManga(recordId: string): Promise<unknown>;
  readerHistory(): Promise<ReaderHistoryResult>;
  setReaderProgress(config: { mangaId: string; chapterId: string; lastPage: number; pageCount?: number; read?: boolean }): Promise<unknown>;
  readerCategories(categories?: ReaderCategory[]): Promise<ReaderCategoriesResult>;
  readerAutomationGet(): Promise<ReaderAutomationResult>;
  readerAutomationSet(config: { enabled: boolean; intervalHours: number }): Promise<ReaderAutomationResult>;
  readerAutomationRunNow(): Promise<ReaderAutomationRunResult>;
  enqueueReaderDownload(config: { mangaId: string; chapterId: string; extensionPackage: string; sourceId: string; chapter: RuntimeChapter }): Promise<ReaderDownloadResult>;
  pauseReaderDownload(jobId: string): Promise<ReaderDownloadResult>;
  resumeReaderDownload(jobId: string): Promise<ReaderDownloadResult>;
  retryReaderDownload(jobId: string): Promise<ReaderDownloadResult>;
  removeReaderDownload(jobId: string): Promise<boolean>;
  listReaderDownloads(): Promise<ReaderDownloadListResult>;
  translateReaderChapter(config: { jobId: string; projectJsonPath: string }): Promise<{ preparedPages: Array<{ number: number; relativePath: string; width: number; height: number }> }>;
}

export const tauriSourceRuntimeClient: SourceRuntimeClient = {
  status: () => invoke("studio_source_runtime_status"),
  start: () => invoke("studio_source_runtime_start"),
  shutdown: () => invoke("studio_source_runtime_shutdown"),
  previewRepository: (url) => invoke("studio_source_repository_preview", { config: { url } }),
  addRepository: (config) => invoke("studio_source_repository_add", { config }),
  listRepositories: () => invoke("studio_source_repository_list"),
  refreshRepositories: () => invoke("studio_source_repository_refresh"),
  listExtensions: () => invoke("studio_source_extension_list"),
  installExtension: (config) => invoke("studio_source_extension_install", { config }),
  setExtensionEnabled: (config) => invoke("studio_source_extension_set_enabled", { config }),
  uninstallExtension: (packageName) => invoke("studio_source_extension_uninstall", { config: { packageName } }),
  rollbackExtension: (packageName) => invoke("studio_source_extension_rollback", { config: { packageName } }),
  removeRepository: (id) => invoke("studio_source_repository_remove", { config: { id } }),
  filters: (config) => invoke("studio_source_filters", { config }),
  search: (config) => invoke("studio_source_search", { config }),
  popular: (config) => invoke("studio_source_popular", { config }),
  latest: (config) => invoke("studio_source_latest", { config }),
  mangaDetails: (config) => invoke("studio_source_manga_details", { config }),
  chapters: (config) => invoke("studio_source_chapters", { config }),
  pages: (config) => invoke("studio_source_pages", { config }),
  listReaderLibrary: () => invoke("studio_reader_library_list"),
  addReaderManga: (manga) => invoke("studio_reader_library_add", { config: { manga } }),
  updateReaderManga: (manga) => invoke("studio_reader_library_update", { config: { manga } }),
  removeReaderManga: (recordId) => invoke("studio_reader_library_remove", { config: { recordId } }),
  readerHistory: () => invoke("studio_reader_history"),
  setReaderProgress: (config) => invoke("studio_reader_progress", { config }),
  readerCategories: (categories) => invoke("studio_reader_categories", { config: categories ? { categories } : null }),
  readerAutomationGet: () => invoke("studio_reader_automation_get"),
  readerAutomationSet: (config) => invoke("studio_reader_automation_set", { config }),
  readerAutomationRunNow: () => invoke("studio_reader_automation_run_now"),
  enqueueReaderDownload: (config) => invoke("studio_reader_download_enqueue", { config }),
  pauseReaderDownload: (recordId) => invoke("studio_reader_download_pause", { config: { recordId } }),
  resumeReaderDownload: (recordId) => invoke("studio_reader_download_resume", { config: { recordId } }),
  retryReaderDownload: (recordId) => invoke("studio_reader_download_retry", { config: { recordId } }),
  removeReaderDownload: (recordId) => invoke("studio_reader_download_remove", { config: { recordId } }),
  listReaderDownloads: () => invoke("studio_reader_download_list"),
  translateReaderChapter: (config) => invoke("studio_reader_translate_chapter", { config }),
};

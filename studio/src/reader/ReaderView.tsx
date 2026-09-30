import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowLeft, BookMarked, Clock3, Compass, Download, History, LibraryBig, RefreshCw, Server, ShieldCheck } from "lucide-react";
import { addReaderManga, createEmptyReaderDocument, loadReaderDocument, markReaderProgress, normalizeGenres, saveReaderDocument, toggleReaderBookmark, type ReaderDocument } from "./readerModel";
import { buildInstalledSources, filterCatalogByAllowedLanguages, filterExtensions, listLanguages, type InstalledSource } from "./sourceCatalogModel";
import { ExtensionsTab, NavigateTabs, SourceCatalogView, SourcesTab, type NavigateManga, type NavigateSourceMode, type NavigateTab } from "./ReaderNavigate";
import { sourceIdentity } from "./navigateModel";
import { tauriSourceRuntimeClient, type InstalledExtension, type ReaderAutomationSettings, type ReaderDownloadRecord, type ReaderHistoryEntry, type RepositoryExtension, type RepositoryPreviewResult, type RuntimeChapter, type RuntimeManga, type SourceFilter, type SourceFilterChange, type SourceRuntimeClient, type SourceRuntimeStatus } from "./sourceRuntimeClient";
import { ReaderMangaDetails } from "./ReaderMangaDetails";
import { ReaderLanguageDialog } from "./ReaderLanguageDialog";
import { parseAllowedLanguages, READER_LANGUAGES_STORAGE_KEY, saveAllowedLanguages, type AllowedLanguages } from "./readerPreferences";
import { ReaderLibraryGrid } from "./ReaderLibraryGrid";
import { ReaderCover } from "./ReaderCover";

type ReaderSection = "discover" | "library" | "updates" | "downloads" | "history";
type DiscoverTab = NavigateTab;
type SourceCatalogMode = NavigateSourceMode;
type DiscoveredManga = NavigateManga;

const CATALOG_PAGE_SIZE = 60;
const RECENT_SOURCE_STORAGE_KEY = "traduzai.studio.reader.recent-source.v1";
const SECTIONS: Array<{ id: ReaderSection; label: string; icon: typeof Compass }> = [
  { id: "discover", label: "Descobrir", icon: Compass },
  { id: "library", label: "Biblioteca do leitor", icon: LibraryBig },
  { id: "updates", label: "Atualizações", icon: RefreshCw },
  { id: "downloads", label: "Downloads", icon: Download },
  { id: "history", label: "Histórico", icon: History },
];

function hasTauriRuntime() {
  return typeof window !== "undefined" && ("__TAURI_INTERNALS__" in window || "__TAURI__" in window);
}

export interface ReaderTranslationRequest {
  manga: ReaderDocument["manga"][number];
  chapter: ReaderDocument["manga"][number]["chapters"][number];
  download: ReaderDownloadRecord;
}

export function ReaderView({ runtimeClient = tauriSourceRuntimeClient, onTranslateChapter, onNotification }: { runtimeClient?: SourceRuntimeClient; onTranslateChapter?: (request: ReaderTranslationRequest) => Promise<void>; onNotification?: (message: string) => void }) {
  const [section, setSection] = useState<ReaderSection>("discover");
  const [document, setDocument] = useState<ReaderDocument>(() => typeof window === "undefined" ? createEmptyReaderDocument() : loadReaderDocument(window.localStorage));
  const [runtime, setRuntime] = useState<SourceRuntimeStatus | null>(null);
  const [runtimeError, setRuntimeError] = useState<string | null>(null);
  const [repositoryPanel, setRepositoryPanel] = useState(false);
  const [repositoryUrl, setRepositoryUrl] = useState("");
  const [repositoryPreview, setRepositoryPreview] = useState<RepositoryPreviewResult["result"] | null>(null);
  const [repositoryBusy, setRepositoryBusy] = useState(false);
  const [catalogExtensions, setCatalogExtensions] = useState<RepositoryExtension[]>([]);
  const [installedExtensions, setInstalledExtensions] = useState<InstalledExtension[]>([]);
  const [discoverTab, setDiscoverTab] = useState<DiscoverTab>("sources");
  const [allowedLanguages, setAllowedLanguages] = useState<AllowedLanguages>(() => typeof window === "undefined" ? null : parseAllowedLanguages(window.localStorage.getItem(READER_LANGUAGES_STORAGE_KEY)));
  const [languageDialogOpen, setLanguageDialogOpen] = useState(false);
  const [catalogSearchOpen, setCatalogSearchOpen] = useState(false);
  const [catalogQuery, setCatalogQuery] = useState("");
  const [visibleExtensionCount, setVisibleExtensionCount] = useState(CATALOG_PAGE_SIZE);
  const [selectedSource, setSelectedSource] = useState<InstalledSource | null>(null);
  const [sourceMode, setSourceMode] = useState<SourceCatalogMode>("popular");
  const [recentSourceIdentity, setRecentSourceIdentity] = useState<string | null>(() => typeof window === "undefined" ? null : window.localStorage.getItem(RECENT_SOURCE_STORAGE_KEY));
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<DiscoveredManga[]>([]);
  const [resultHeading, setResultHeading] = useState("");
  const [searchBusy, setSearchBusy] = useState(false);
  const [sourceFilters, setSourceFilters] = useState<SourceFilter[]>([]);
  const [sourceFiltersBusy, setSourceFiltersBusy] = useState(false);
  const [sourceFiltersError, setSourceFiltersError] = useState<string | null>(null);
  const [activeSourceFilters, setActiveSourceFilters] = useState<SourceFilterChange[]>([]);
  const [selectedRemoteManga, setSelectedRemoteManga] = useState<DiscoveredManga | null>(null);
  const [detailManga, setDetailManga] = useState<RuntimeManga | null>(null);
  const [detailChapters, setDetailChapters] = useState<RuntimeChapter[]>([]);
  const [detailBusy, setDetailBusy] = useState(false);
  const [detailChaptersError, setDetailChaptersError] = useState<string | null>(null);
  const detailRequestRef = useRef(0);
  const [automation, setAutomation] = useState<ReaderAutomationSettings>({ enabled: false, intervalHours: 12 });
  const [automationBusy, setAutomationBusy] = useState(false);
  const [automationMessage, setAutomationMessage] = useState("");
  const [downloads, setDownloads] = useState<ReaderDownloadRecord[]>([]);
  const [downloadsLoaded, setDownloadsLoaded] = useState(() => !hasTauriRuntime());
  const [selectedMangaId, setSelectedMangaId] = useState<string | null>(null);
  const [readingDownload, setReadingDownload] = useState<ReaderDownloadRecord | null>(null);
  const [readerMode, setReaderMode] = useState<"vertical" | "ltr" | "rtl">("vertical");
  const [history, setHistory] = useState<ReaderHistoryEntry[]>([]);
  const [newCategoryName, setNewCategoryName] = useState("");

  const unread = useMemo(() => document.manga.reduce((total, manga) => total + manga.chapters.filter((chapter) => !chapter.read).length, 0), [document.manga]);
  const selectedLibraryManga = useMemo(() => document.manga.find((manga) => manga.id === selectedMangaId) ?? null, [document.manga, selectedMangaId]);
  const languageOptions = useMemo(() => listLanguages(catalogExtensions), [catalogExtensions]);
  const allowedCatalog = useMemo(() => filterCatalogByAllowedLanguages(catalogExtensions, allowedLanguages), [allowedLanguages, catalogExtensions]);
  const filteredExtensions = useMemo(() => filterExtensions(allowedCatalog, "*", catalogQuery), [allowedCatalog, catalogQuery]);
  const installedSources = useMemo(() => buildInstalledSources(allowedCatalog, installedExtensions, "*", catalogQuery), [allowedCatalog, installedExtensions, catalogQuery]);

  useEffect(() => {
    const storage = typeof window === "undefined" ? null : window.localStorage;
    saveReaderDocument(storage, document);
  }, [document]);

  useEffect(() => {
    if (!runtimeError || !onNotification) return;
    onNotification(runtimeError);
    setRuntimeError(null);
  }, [onNotification, runtimeError]);

  useEffect(() => {
    if (!hasTauriRuntime()) return;
    void runtimeClient.status().then(setRuntime).catch((error) => setRuntimeError(error instanceof Error ? error.message : String(error)));
  }, [runtimeClient]);

  useEffect(() => {
    if (!hasTauriRuntime() || !runtime?.configured) return;
    let cancelled = false;
    void (async () => {
      if (!runtime.running) await runtimeClient.start();
      return Promise.all([runtimeClient.listReaderLibrary(), runtimeClient.readerCategories(), runtimeClient.listRepositories(), runtimeClient.listExtensions(), runtimeClient.readerAutomationGet(), runtimeClient.listReaderDownloads(), runtimeClient.readerHistory()]);
    })()
      .then(async ([library, categories, repositories, extensions, automationResult, downloadResult, historyResult]) => {
        if (cancelled) return;
        setInstalledExtensions(extensions.result.extensions);
        setAutomation(automationResult.result.automation);
        setDownloads(downloadResult.result.downloads);
        setDownloadsLoaded(true);
        setHistory(historyResult.result.history);
        setDocument((current) => ({
          ...current,
          manga: library.result.manga,
          categories: categories.result.categories,
          repositories: repositories.result.repositories.map((repository) => ({
            id: repository.id,
            url: repository.url,
            fingerprint: repository.certificateFingerprint,
            trustedAt: current.repositories.find((item) => item.id === repository.id)?.trustedAt ?? new Date(0).toISOString(),
          })),
          updatedAt: new Date().toISOString(),
        }));
        if (repositories.result.repositories.length > 0) {
          const catalog = await runtimeClient.refreshRepositories();
          if (!cancelled) setCatalogExtensions(catalog.result.extensions);
        }
        if (!cancelled) setRuntime((current) => current ? { ...current, running: true } : current);
      })
      .catch((error) => {
        if (!cancelled) {
          setDownloadsLoaded(true);
          setRuntimeError(error instanceof Error ? error.message : String(error));
        }
      });
    return () => { cancelled = true; };
  }, [runtime?.configured, runtimeClient]);

  useEffect(() => {
    if (!hasTauriRuntime() || !downloads.some((download) => ["queued", "running", "retrying"].includes(download.status))) return;
    const timer = window.setInterval(() => {
      void runtimeClient.listReaderDownloads()
        .then((response) => setDownloads(response.result.downloads))
        .catch((error) => setRuntimeError(error instanceof Error ? error.message : String(error)));
    }, 750);
    return () => window.clearInterval(timer);
  }, [downloads, runtimeClient]);

  const saveAutomation = async (next: ReaderAutomationSettings) => {
    setAutomationBusy(true);
    setRuntimeError(null);
    try {
      const response = await runtimeClient.readerAutomationSet({ enabled: next.enabled, intervalHours: next.intervalHours });
      setAutomation(response.result.automation);
      setAutomationMessage("Preferências de atualização salvas.");
    } catch (error) {
      setRuntimeError(error instanceof Error ? error.message : String(error));
    } finally {
      setAutomationBusy(false);
    }
  };

  const runAutomationNow = async () => {
    setAutomationBusy(true);
    setRuntimeError(null);
    try {
      const response = await runtimeClient.readerAutomationRunNow();
      const { checked, updated, failed } = response.result.summary;
      setAutomation((current) => ({ ...current, lastRunAt: response.result.ranAt, lastRunSummary: response.result.summary }));
      setAutomationMessage(`${checked} obra(s) verificadas, ${updated} atualizada(s)${failed ? ` e ${failed} com falha` : ""}.`);
      const library = await runtimeClient.listReaderLibrary();
      setDocument((current) => ({ ...current, manga: library.result.manga, updatedAt: new Date().toISOString() }));
    } catch (error) {
      setRuntimeError(error instanceof Error ? error.message : String(error));
    } finally {
      setAutomationBusy(false);
    }
  };

  const setMangaAutoUpdate = async (mangaId: string, enabled: boolean) => {
    const manga = document.manga.find((item) => item.id === mangaId);
    if (!manga) return;
    const updated = { ...manga, autoUpdate: enabled };
    try {
      const saved = hasTauriRuntime() ? (await runtimeClient.updateReaderManga(updated)).result.manga : updated;
      setDocument((current) => ({ ...current, manga: current.manga.map((item) => item.id === mangaId ? saved : item), updatedAt: new Date().toISOString() }));
    } catch (error) {
      setRuntimeError(error instanceof Error ? error.message : String(error));
    }
  };

  const toggleChapterBookmark = async (mangaId: string, chapterId: string) => {
    const next = toggleReaderBookmark(document, mangaId, chapterId);
    const manga = next.manga.find((item) => item.id === mangaId);
    if (!manga) return;
    try {
      const saved = hasTauriRuntime() ? (await runtimeClient.updateReaderManga(manga)).result.manga : manga;
      setDocument((current) => ({ ...current, manga: current.manga.map((item) => item.id === mangaId ? saved : item), updatedAt: new Date().toISOString() }));
    } catch (error) {
      setRuntimeError(error instanceof Error ? error.message : String(error));
    }
  };

  const addCategory = async () => {
    const name = newCategoryName.trim();
    if (!name || document.categories.some((category) => category.name.toLocaleLowerCase("pt-BR") === name.toLocaleLowerCase("pt-BR"))) return;
    const categories = [...document.categories, { id: crypto.randomUUID(), name, order: document.categories.length }];
    try {
      const saved = hasTauriRuntime() ? (await runtimeClient.readerCategories(categories)).result.categories : categories;
      setDocument((current) => ({ ...current, categories: saved, updatedAt: new Date().toISOString() }));
      setNewCategoryName("");
    } catch (error) {
      setRuntimeError(error instanceof Error ? error.message : String(error));
    }
  };

  const toggleMangaCategory = async (mangaId: string, categoryId: string) => {
    const manga = document.manga.find((item) => item.id === mangaId);
    if (!manga) return;
    const updated = { ...manga, categoryIds: manga.categoryIds.includes(categoryId) ? manga.categoryIds.filter((id) => id !== categoryId) : [...manga.categoryIds, categoryId] };
    try {
      const saved = hasTauriRuntime() ? (await runtimeClient.updateReaderManga(updated)).result.manga : updated;
      setDocument((current) => ({ ...current, manga: current.manga.map((item) => item.id === mangaId ? saved : item), updatedAt: new Date().toISOString() }));
    } catch (error) {
      setRuntimeError(error instanceof Error ? error.message : String(error));
    }
  };

  const previewRepository = async () => {
    setRepositoryBusy(true);
    setRuntimeError(null);
    try {
      if (!runtime?.running) await runtimeClient.start();
      const preview = await runtimeClient.previewRepository(repositoryUrl.trim());
      setRepositoryPreview(preview.result);
      setRuntime((current) => current ? { ...current, running: true } : current);
    } catch (error) {
      setRepositoryPreview(null);
      setRuntimeError(error instanceof Error ? error.message : String(error));
    } finally {
      setRepositoryBusy(false);
    }
  };

  const trustRepository = async () => {
    if (!repositoryPreview) return;
    setRepositoryBusy(true);
    setRuntimeError(null);
    try {
      const saved = await runtimeClient.addRepository(repositoryPreview);
      setDocument((current) => ({
        ...current,
        repositories: [...current.repositories.filter((item) => item.url !== repositoryPreview.url), {
          id: saved.result.id,
          url: saved.result.url,
          fingerprint: saved.result.certificateFingerprint,
          trustedAt: new Date().toISOString(),
        }],
        updatedAt: new Date().toISOString(),
      }));
      setRepositoryPanel(false);
      setRepositoryPreview(null);
      setRepositoryUrl("");
      const [catalog, installed] = await Promise.all([runtimeClient.refreshRepositories(), runtimeClient.listExtensions()]);
      setCatalogExtensions(catalog.result.extensions);
      setInstalledExtensions(installed.result.extensions);
    } catch (error) {
      setRuntimeError(error instanceof Error ? error.message : String(error));
    } finally {
      setRepositoryBusy(false);
    }
  };

  const refreshCatalog = async () => {
    setRepositoryBusy(true);
    setRuntimeError(null);
    try {
      if (!runtime?.running) await runtimeClient.start();
      const [catalog, installed] = await Promise.all([runtimeClient.refreshRepositories(), runtimeClient.listExtensions()]);
      setCatalogExtensions(catalog.result.extensions);
      setInstalledExtensions(installed.result.extensions);
      setVisibleExtensionCount(CATALOG_PAGE_SIZE);
      setRuntime((current) => current ? { ...current, running: true } : current);
    } catch (error) {
      setRuntimeError(error instanceof Error ? error.message : String(error));
    } finally {
      setRepositoryBusy(false);
    }
  };

  const installExtension = async (extension: RepositoryExtension) => {
    setRepositoryBusy(true);
    setRuntimeError(null);
    try {
      await runtimeClient.installExtension({ repositoryId: extension.repositoryId, packageName: extension.packageName, versionCode: extension.versionCode });
      const installed = await runtimeClient.listExtensions();
      setInstalledExtensions(installed.result.extensions);
      setDocument((current) => ({
        ...current,
        extensions: [...current.extensions.filter((item) => item.packageName !== extension.packageName), {
          packageName: extension.packageName,
          name: extension.name,
          versionCode: extension.versionCode,
          versionName: extension.versionName,
          enabled: true,
          sourceIds: extension.sources.map((source) => source.id),
        }],
        updatedAt: new Date().toISOString(),
      }));
      setDiscoverTab("sources");
      setCatalogQuery("");
    } catch (error) {
      setRuntimeError(error instanceof Error ? error.message : String(error));
    } finally {
      setRepositoryBusy(false);
    }
  };

  const setExtensionEnabled = async (extension: InstalledExtension, enabled: boolean) => {
    setRepositoryBusy(true);
    setRuntimeError(null);
    try {
      await runtimeClient.setExtensionEnabled({ packageName: extension.packageName, enabled });
      setInstalledExtensions((current) => current.map((item) => item.packageName === extension.packageName ? { ...item, enabled } : item));
      setDocument((current) => ({
        ...current,
        extensions: current.extensions.map((item) => item.packageName === extension.packageName ? { ...item, enabled } : item),
        updatedAt: new Date().toISOString(),
      }));
    } catch (error) {
      setRuntimeError(error instanceof Error ? error.message : String(error));
    } finally {
      setRepositoryBusy(false);
    }
  };

  const uninstallExtension = async (extension: InstalledExtension) => {
    if (!window.confirm(`Desinstalar ${extension.name}? As obras do leitor serão preservadas, mas ficarão indisponíveis até a extensão ser instalada novamente.`)) return;
    setRepositoryBusy(true);
    setRuntimeError(null);
    try {
      await runtimeClient.uninstallExtension(extension.packageName);
      setInstalledExtensions((current) => current.filter((item) => item.packageName !== extension.packageName));
      setDocument((current) => ({
        ...current,
        extensions: current.extensions.filter((item) => item.packageName !== extension.packageName),
        updatedAt: new Date().toISOString(),
      }));
    } catch (error) {
      setRuntimeError(error instanceof Error ? error.message : String(error));
    } finally {
      setRepositoryBusy(false);
    }
  };

  const rollbackExtension = async (extension: InstalledExtension) => {
    if (!window.confirm(`Restaurar a versão anterior de ${extension.name}?`)) return;
    setRepositoryBusy(true);
    setRuntimeError(null);
    try {
      await runtimeClient.rollbackExtension(extension.packageName);
      const installed = await runtimeClient.listExtensions();
      setInstalledExtensions(installed.result.extensions);
    } catch (error) {
      setRuntimeError(error instanceof Error ? error.message : String(error));
    } finally {
      setRepositoryBusy(false);
    }
  };

  const loadSourceCatalog = async (source: InstalledSource, mode: SourceCatalogMode) => {
    setSelectedSource(source);
    setSourceMode(mode);
    setSearchBusy(true);
    setRuntimeError(null);
    try {
      const response = mode === "popular"
        ? await runtimeClient.popular({ extensionPackage: source.extensionPackage, sourceId: source.id, page: 1 })
        : await runtimeClient.latest({ extensionPackage: source.extensionPackage, sourceId: source.id, page: 1 });
      setSearchResults(response.result.manga.map((manga) => ({ ...manga, extensionPackage: source.extensionPackage, sourceName: response.result.sourceName })));
      setResultHeading(mode === "popular" ? "Obras populares" : "Últimas atualizações");
      const identity = sourceIdentity(source);
      setRecentSourceIdentity(identity);
      if (typeof window !== "undefined") window.localStorage.setItem(RECENT_SOURCE_STORAGE_KEY, identity);
    } catch (error) {
      setSearchResults([]);
      setRuntimeError(error instanceof Error ? error.message : String(error));
    } finally {
      setSearchBusy(false);
    }
  };

  const loadSourceFilters = async () => {
    if (!selectedSource || sourceFiltersBusy || sourceFilters.length) return;
    setSourceFiltersBusy(true);
    setSourceFiltersError(null);
    try {
      const response = await runtimeClient.filters({ extensionPackage: selectedSource.extensionPackage, sourceId: selectedSource.id });
      setSourceFilters(response.result.filters);
    } catch (error) {
      setSourceFiltersError(error instanceof Error ? error.message : String(error));
    } finally {
      setSourceFiltersBusy(false);
    }
  };

  const searchSelectedSource = async (filters: SourceFilterChange[] = activeSourceFilters) => {
    if (!selectedSource || (!searchQuery.trim() && filters.length === 0)) return;
    setSearchBusy(true);
    setRuntimeError(null);
    try {
      const response = await runtimeClient.search({ extensionPackage: selectedSource.extensionPackage, sourceId: selectedSource.id, query: searchQuery.trim(), filters });
      setSearchResults(response.result.manga.map((manga) => ({ ...manga, extensionPackage: selectedSource.extensionPackage, sourceName: response.result.sourceName })));
      setResultHeading(searchQuery.trim() ? `Resultados para “${searchQuery.trim()}”` : "Resultados filtrados");
    } catch (error) {
      setSearchResults([]);
      setRuntimeError(error instanceof Error ? error.message : String(error));
    } finally {
      setSearchBusy(false);
    }
  };

  const openDiscoveredManga = async (item: DiscoveredManga) => {
    const requestId = ++detailRequestRef.current;
    const sourceId = item.sourceId ?? selectedSource?.id ?? "";
    setSelectedRemoteManga(item);
    setDetailManga(item);
    setDetailChapters([]);
    setDetailChaptersError(null);
    setDetailBusy(true);
    setRuntimeError(null);
    try {
      const details = await runtimeClient.mangaDetails({ extensionPackage: item.extensionPackage, sourceId, manga: item });
      if (detailRequestRef.current !== requestId) return;
      const enriched = { ...item, ...details.result.manga, sourceId };
      setDetailManga(enriched);
      try {
        const chapters = await runtimeClient.chapters({ extensionPackage: item.extensionPackage, sourceId, manga: enriched });
        if (detailRequestRef.current !== requestId) return;
        setDetailChapters(chapters.result.chapters);
      } catch (error) {
        if (detailRequestRef.current === requestId) setDetailChaptersError(error instanceof Error ? error.message : String(error));
      }
    } catch (error) {
      if (detailRequestRef.current === requestId) setDetailChaptersError(error instanceof Error ? error.message : String(error));
    } finally {
      if (detailRequestRef.current === requestId) setDetailBusy(false);
    }
  };

  const addDiscoveredManga = async (item: DiscoveredManga, knownDetails?: RuntimeManga, knownChapters?: RuntimeChapter[]) => {
    setSearchBusy(true);
    setRuntimeError(null);
    try {
      const sourceId = item.sourceId ?? "";
      const details = knownDetails ?? (await runtimeClient.mangaDetails({ extensionPackage: item.extensionPackage, sourceId, manga: item })).result.manga;
      const chapters = knownChapters ?? (await runtimeClient.chapters({ extensionPackage: item.extensionPackage, sourceId, manga: details })).result.chapters;
      const discovered = {
        id: crypto.randomUUID(), extensionPackage: item.extensionPackage, sourceId, sourceName: item.sourceName, mangaUrl: item.url,
        title: details.title, thumbnailUrl: details.thumbnailUrl, author: details.author, artist: details.artist, description: details.description,
        genres: normalizeGenres(details.genre), status: details.status,
        chapters: chapters.map((chapter) => ({ ...chapter, id: crypto.randomUUID(), read: false, bookmarked: false, lastPageRead: 0, pageCount: null })),
        categoryIds: [], favorite: true, autoUpdate: false, autoDownload: false,
      };
      const saved = hasTauriRuntime() ? (await runtimeClient.addReaderManga(discovered)).result.manga : discovered;
      setDocument((current) => addReaderManga(current, saved));
    } catch (error) {
      setRuntimeError(error instanceof Error ? error.message : String(error));
    } finally {
      setSearchBusy(false);
    }
  };

  const removeLibraryManga = async (mangaId: string) => {
    const manga = document.manga.find((item) => item.id === mangaId);
    if (!manga || !window.confirm(`Remover ${manga.title} da biblioteca do leitor?`)) return false;
    try {
      if (hasTauriRuntime()) await runtimeClient.removeReaderManga(manga.id);
      setDocument((current) => ({ ...current, manga: current.manga.filter((item) => item.id !== manga.id), updatedAt: new Date().toISOString() }));
      if (selectedMangaId === manga.id) setSelectedMangaId(null);
      return true;
    } catch (error) {
      setRuntimeError(error instanceof Error ? error.message : String(error));
      return false;
    }
  };

  const refreshLibraryManga = async (mangaId: string) => {
    const manga = document.manga.find((item) => item.id === mangaId);
    if (!manga) return;
    setSearchBusy(true);
    setRuntimeError(null);
    try {
      const details = (await runtimeClient.mangaDetails({ extensionPackage: manga.extensionPackage, sourceId: manga.sourceId, manga: { url: manga.mangaUrl, title: manga.title, thumbnailUrl: manga.thumbnailUrl } })).result.manga;
      const remoteChapters = (await runtimeClient.chapters({ extensionPackage: manga.extensionPackage, sourceId: manga.sourceId, manga: details })).result.chapters;
      const existingByUrl = new Map(manga.chapters.map((chapter) => [chapter.url, chapter]));
      const refreshed = {
        ...manga,
        title: details.title,
        thumbnailUrl: details.thumbnailUrl,
        author: details.author,
        artist: details.artist,
        description: details.description,
        genres: normalizeGenres(details.genre),
        status: details.status,
        chapters: remoteChapters.map((chapter) => ({ ...chapter, ...(existingByUrl.get(chapter.url) ?? { id: crypto.randomUUID(), read: false, bookmarked: false, lastPageRead: 0, pageCount: null }) })),
      };
      const saved = hasTauriRuntime() ? (await runtimeClient.updateReaderManga(refreshed)).result.manga : refreshed;
      setDocument((current) => ({ ...current, manga: current.manga.map((item) => item.id === saved.id ? saved : item), updatedAt: new Date().toISOString() }));
    } catch (error) {
      setRuntimeError(error instanceof Error ? error.message : String(error));
    } finally {
      setSearchBusy(false);
    }
  };

  const toggleSelectedRemoteLibrary = async () => {
    if (!selectedRemoteManga || !detailManga) return;
    const existing = document.manga.find((manga) => manga.extensionPackage === selectedRemoteManga.extensionPackage && manga.sourceId === (selectedRemoteManga.sourceId ?? "") && manga.mangaUrl === selectedRemoteManga.url);
    if (existing) {
      setDetailBusy(true);
      try {
        await removeLibraryManga(existing.id);
      } finally {
        setDetailBusy(false);
      }
      return;
    }
    await addDiscoveredManga(selectedRemoteManga, detailManga, detailChapters);
  };

  const downloadChapter = async (mangaId: string, chapterId: string) => {
    const manga = document.manga.find((item) => item.id === mangaId);
    const chapter = manga?.chapters.find((item) => item.id === chapterId);
    if (!manga || !chapter) return;
    setSearchBusy(true);
    setRuntimeError(null);
    try {
      const response = await runtimeClient.enqueueReaderDownload({
        mangaId,
        chapterId,
        extensionPackage: manga.extensionPackage,
        sourceId: manga.sourceId,
        chapter: chapter as RuntimeChapter,
      });
      const record: ReaderDownloadRecord = { ...response.result, mangaId, chapterId };
      setDownloads((current) => [...current.filter((item) => item.chapterId !== chapterId), record]);
    } catch (error) {
      setRuntimeError(error instanceof Error ? error.message : String(error));
    } finally {
      setSearchBusy(false);
    }
  };

  const removeDownload = async (jobId: string) => {
    try {
      await runtimeClient.removeReaderDownload(jobId);
      setDownloads((current) => current.filter((item) => item.jobId !== jobId));
      if (readingDownload?.jobId === jobId) setReadingDownload(null);
    } catch (error) {
      setRuntimeError(error instanceof Error ? error.message : String(error));
    }
  };

  const updateDownloadAction = async (jobId: string, action: "pause" | "resume" | "retry") => {
    setRuntimeError(null);
    try {
      const response = action === "pause"
        ? await runtimeClient.pauseReaderDownload(jobId)
        : action === "retry"
          ? await runtimeClient.retryReaderDownload(jobId)
          : await runtimeClient.resumeReaderDownload(jobId);
      const record = response.result as ReaderDownloadRecord;
      setDownloads((current) => current.map((item) => item.jobId === jobId ? record : item));
    } catch (error) {
      setRuntimeError(error instanceof Error ? error.message : String(error));
    }
  };

  const translateChapter = async (request: ReaderTranslationRequest) => {
    if (!onTranslateChapter) return;
    setSearchBusy(true);
    setRuntimeError(null);
    try {
      await onTranslateChapter(request);
    } catch (error) {
      setRuntimeError(error instanceof Error ? error.message : String(error));
    } finally {
      setSearchBusy(false);
    }
  };

  const recordReadingProgress = (download: ReaderDownloadRecord, page: number) => {
    setDocument((current) => markReaderProgress(current, download.mangaId, download.chapterId, page));
    setHistory((current) => [{ mangaId: download.mangaId, chapterId: download.chapterId, lastPage: page, pageCount: download.pageCount, read: page >= download.pageCount, updatedAt: new Date().toISOString() }, ...current.filter((item) => item.mangaId !== download.mangaId || item.chapterId !== download.chapterId)]);
    void runtimeClient.setReaderProgress({
      mangaId: download.mangaId,
      chapterId: download.chapterId,
      lastPage: page,
      pageCount: download.pageCount,
      read: page >= download.pageCount,
    }).catch((error) => setRuntimeError(error instanceof Error ? error.message : String(error)));
  };

  useEffect(() => {
    if (!hasTauriRuntime() || !runtime?.configured || !automation.enabled) return;
    const intervalMs = automation.intervalHours * 60 * 60 * 1000;
    const lastRunMs = automation.lastRunAt ? Date.parse(automation.lastRunAt) : 0;
    let cancelled = false;
    const execute = async () => {
      if (cancelled) return;
      setAutomationBusy(true);
      try {
        const response = await runtimeClient.readerAutomationRunNow();
        if (cancelled) return;
        setAutomation((current) => ({ ...current, lastRunAt: response.result.ranAt, lastRunSummary: response.result.summary }));
        const library = await runtimeClient.listReaderLibrary();
        if (!cancelled) setDocument((current) => ({ ...current, manga: library.result.manga, updatedAt: new Date().toISOString() }));
      } catch (error) {
        if (!cancelled) setRuntimeError(error instanceof Error ? error.message : String(error));
      } finally {
        if (!cancelled) setAutomationBusy(false);
      }
    };
    if (!lastRunMs || Date.now() - lastRunMs >= intervalMs) void execute();
    const timer = window.setInterval(() => void execute(), intervalMs);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [automation.enabled, automation.intervalHours, automation.lastRunAt, runtime?.configured, runtimeClient]);

  return <section className="studio-reader-view">
    <aside className="studio-reader-sidebar">
      <div><p className="eyebrow">Fontes locais</p><h1>Leitor</h1></div>
      <nav aria-label="Áreas do leitor">{SECTIONS.map(({ id, label, icon: Icon }) => <button type="button" key={id} aria-current={section === id ? "page" : undefined} onClick={() => setSection(id)}><Icon size={18} /><span>{label}</span>{id === "updates" && unread > 0 && <b>{unread}</b>}</button>)}</nav>
      <div className="studio-reader-runtime"><ShieldCheck size={18} /><div><strong>Sandbox obrigatória</strong><small>{runtime?.running ? "Runtime ativo" : runtime?.configured ? "Pronto para iniciar" : "Runtime não empacotado"}</small></div></div>
    </aside>

    <main className="studio-reader-main">
      {runtimeError && !onNotification && <p className="studio-reader-error" role="alert">{runtimeError}</p>}
      {section === "discover" && selectedRemoteManga && detailManga
        ? <ReaderMangaDetails manga={detailManga} sourceName={selectedRemoteManga.sourceName} chapters={detailChapters} inLibrary={document.manga.some((manga) => manga.extensionPackage === selectedRemoteManga.extensionPackage && manga.sourceId === (selectedRemoteManga.sourceId ?? "") && manga.mangaUrl === selectedRemoteManga.url)} busy={detailBusy} chaptersLoading={detailBusy && detailChapters.length === 0 && !detailChaptersError} chaptersError={detailChaptersError} onBack={() => { detailRequestRef.current += 1; setSelectedRemoteManga(null); setDetailManga(null); setDetailChapters([]); setDetailChaptersError(null); }} onToggleLibrary={() => void toggleSelectedRemoteLibrary()} onRetryChapters={() => void openDiscoveredManga(selectedRemoteManga)} />
        : section === "discover" && <>
        <ReaderLanguageDialog open={languageDialogOpen} languages={languageOptions.filter((language) => language.code !== "*")} allowed={allowedLanguages} onCancel={() => setLanguageDialogOpen(false)} onApply={(allowed) => { setAllowedLanguages(allowed); setLanguageDialogOpen(false); setVisibleExtensionCount(CATALOG_PAGE_SIZE); if (typeof window !== "undefined") saveAllowedLanguages(window.localStorage, allowed); }} />
        {repositoryPanel && <RepositoryOverlay preview={repositoryPreview} url={repositoryUrl} busy={repositoryBusy} configured={Boolean(runtime?.configured)} onUrlChange={(url) => { setRepositoryUrl(url); setRepositoryPreview(null); }} onPreview={() => void previewRepository()} onTrust={() => void trustRepository()} onClose={() => { setRepositoryPanel(false); setRepositoryPreview(null); setRuntimeError(null); }} />}
        {!repositoryPanel && document.repositories.length === 0 && <section className="studio-reader-empty"><Server size={34} /><h2>Nenhum repositório configurado</h2><p>O TraduzAI não inclui catálogos nem extensões. Adicione manualmente a URL de um repositório e confira as fingerprints antes de confiar.</p><button type="button" disabled={!runtime?.configured} onClick={() => setRepositoryPanel(true)}>Adicionar repositório</button></section>}
        {!repositoryPanel && document.repositories.length > 0 && <>
          {!selectedSource && <>
            <NavigateTabs active={discoverTab} busy={repositoryBusy} searchOpen={catalogSearchOpen} query={catalogQuery} onChange={(tab) => { setDiscoverTab(tab); setSearchResults([]); setResultHeading(""); }} onRefresh={() => void refreshCatalog()} onToggleSearch={() => { if (catalogSearchOpen) setCatalogQuery(""); setCatalogSearchOpen((open) => !open); }} onQueryChange={(query) => { setCatalogQuery(query); setVisibleExtensionCount(CATALOG_PAGE_SIZE); }} onOpenRepositories={() => setRepositoryPanel(true)} onOpenLanguages={() => setLanguageDialogOpen(true)} />
            {discoverTab === "extensions" && <ExtensionsTab busy={repositoryBusy} catalogLoaded={catalogExtensions.length > 0} extensions={filteredExtensions} installed={installedExtensions} visibleCount={visibleExtensionCount} onInstall={installExtension} onSetEnabled={setExtensionEnabled} onRollback={rollbackExtension} onUninstall={uninstallExtension} onLoadMore={() => setVisibleExtensionCount((count) => count + CATALOG_PAGE_SIZE)} />}
            {discoverTab === "sources" && <SourcesTab catalogLoaded={catalogExtensions.length > 0} sources={installedSources} recentSourceIdentity={recentSourceIdentity} busy={searchBusy} onOpen={loadSourceCatalog} onShowExtensions={() => setDiscoverTab("extensions")} />}
          </>}
          {selectedSource && <SourceCatalogView source={selectedSource} mode={sourceMode} query={searchQuery} heading={resultHeading} busy={searchBusy} items={searchResults} filters={sourceFilters} filtersBusy={sourceFiltersBusy} filtersError={sourceFiltersError} onLoadFilters={() => void loadSourceFilters()} onApplyFilters={(filters) => { setActiveSourceFilters(filters); void searchSelectedSource(filters); }} onBack={() => { setSelectedSource(null); setSearchResults([]); setResultHeading(""); setSearchQuery(""); setSourceFilters([]); setActiveSourceFilters([]); setSourceFiltersError(null); }} onModeChange={(mode) => void loadSourceCatalog(selectedSource, mode)} onQueryChange={setSearchQuery} onSearch={() => void searchSelectedSource()} onOpen={(item) => void openDiscoveredManga(item)} onAdd={(item) => void addDiscoveredManga(item)} />}
        </>}
      </>}
      {section === "library" && !readingDownload && (selectedMangaId
        ? selectedLibraryManga && <SavedMangaDetails manga={selectedLibraryManga} downloads={downloads} busy={searchBusy} onBack={() => setSelectedMangaId(null)} onRefresh={() => void refreshLibraryManga(selectedLibraryManga.id)} onRemoveManga={() => void removeLibraryManga(selectedLibraryManga.id)} onDownload={downloadChapter} onRead={setReadingDownload} onTranslate={onTranslateChapter ? translateChapter : undefined} onToggleBookmark={toggleChapterBookmark} onPause={(jobId) => updateDownloadAction(jobId, "pause")} onResume={(jobId) => updateDownloadAction(jobId, "resume")} onRetry={(jobId) => updateDownloadAction(jobId, "retry")} onRemoveDownload={removeDownload} />
        : document.manga.length === 0 ? <ReaderEmpty icon={LibraryBig} title="Biblioteca do leitor vazia" text="Adicione uma obra descoberta por uma extensão. Projetos de tradução continuam na Biblioteca principal." /> : <ReaderLibraryGrid document={document} newCategoryName={newCategoryName} onNewCategoryName={setNewCategoryName} onAddCategory={addCategory} onToggleCategory={toggleMangaCategory} onOpen={setSelectedMangaId} onSetAutoUpdate={setMangaAutoUpdate} onRemove={(mangaId) => void removeLibraryManga(mangaId)} />)}
      {section === "library" && readingDownload && <OfflineReader download={readingDownload} mode={readerMode} onModeChange={setReaderMode} onProgress={(page) => recordReadingProgress(readingDownload, page)} onClose={() => setReadingDownload(null)} />}
      {section === "updates" && <AutomationPanel settings={automation} busy={automationBusy} message={automationMessage} eligible={document.manga.filter((manga) => manga.autoUpdate).length} onChange={saveAutomation} onRunNow={runAutomationNow} />}
      {section === "downloads" && <DownloadsPanel downloads={downloads} loaded={downloadsLoaded} document={document} onRead={(download) => { setSelectedMangaId(download.mangaId); setReadingDownload(download); setSection("library"); }} onPause={(jobId) => updateDownloadAction(jobId, "pause")} onResume={(jobId) => updateDownloadAction(jobId, "resume")} onRetry={(jobId) => updateDownloadAction(jobId, "retry")} onRemove={removeDownload} />}
      {section === "history" && <HistoryPanel history={history} document={document} downloads={downloads} onResume={(download) => { setSelectedMangaId(download.mangaId); setReadingDownload(download); setSection("library"); }} />}
    </main>
  </section>;
}

type RepositoryPanelProps = { preview: RepositoryPreviewResult["result"] | null; url: string; busy: boolean; configured: boolean; onUrlChange(url: string): void; onPreview(): void; onTrust(): void; onClose(): void };

export function RepositoryOverlay(props: RepositoryPanelProps) {
  return <div className="studio-reader-repository-backdrop studio-floating-layer" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) props.onClose(); }}><RepositoryPanel {...props} /></div>;
}

export function RepositoryPanel({ preview, url, busy, configured, onUrlChange, onPreview, onTrust, onClose }: RepositoryPanelProps) {
  return <section className="studio-reader-repository-panel" onKeyDown={(event) => { if (event.key === "Escape") onClose(); }}><header><button type="button" aria-label="Voltar ao catálogo" title="Voltar" onClick={onClose}><ArrowLeft size={19} /></button><h2>Adicionar repositório</h2></header><p>Informe a URL HTTPS do índice. Nada será salvo antes de você conferir e confiar nas fingerprints.</p><div><input aria-label="URL do repositório" type="url" value={url} onChange={(event) => onUrlChange(event.currentTarget.value)} placeholder="https://…/index.min.json" /><button type="button" disabled={!configured || !url.trim() || busy} onClick={onPreview}>{busy ? "Verificando…" : "Visualizar"}</button></div>{preview && <article className="studio-reader-fingerprint"><strong>Confirme antes de confiar</strong><dl><dt>Certificado TLS</dt><dd>{preview.certificateFingerprint}</dd><dt>Índice SHA-256</dt><dd>{preview.indexSha256}</dd><dt>Chave de assinatura</dt><dd>{preview.signingKey ?? "Não declarada (repositório legado)"}</dd><dt>Extensões encontradas</dt><dd>{preview.extensions.length}</dd></dl><button type="button" disabled={busy} onClick={onTrust}>Confiar e adicionar</button></article>}</section>;
}

function ReaderCollection({ document, newCategoryName, onNewCategoryName, onAddCategory, onToggleCategory, onOpen, onSetAutoUpdate }: { document: ReaderDocument; newCategoryName: string; onNewCategoryName(value: string): void; onAddCategory(): void; onToggleCategory(mangaId: string, categoryId: string): void; onOpen(mangaId: string): void; onSetAutoUpdate(mangaId: string, enabled: boolean): void }) {
  if (document.manga.length === 0) return <ReaderEmpty icon={LibraryBig} title="Biblioteca do leitor vazia" text="Adicione uma obra descoberta por uma extensão. Projetos de tradução continuam na Biblioteca principal." />;
  return <><header><div><p className="eyebrow">Obras salvas</p><h1>Biblioteca do leitor</h1></div></header><form className="studio-reader-category-form" onSubmit={(event) => { event.preventDefault(); onAddCategory(); }}><input aria-label="Nova categoria" value={newCategoryName} onChange={(event) => onNewCategoryName(event.currentTarget.value)} placeholder="Nova categoria…" /><button type="submit" disabled={!newCategoryName.trim()}>Adicionar</button></form><div className="studio-reader-grid">{document.manga.map((manga) => <article key={manga.id}><button className="studio-reader-work-open" type="button" onClick={() => onOpen(manga.id)}><h2>{manga.title}</h2><p>{manga.sourceName}</p><span>{manga.chapters.filter((chapter) => !chapter.read).length} não lidos</span></button>{document.categories.length > 0 && <div className="studio-reader-category-chips">{document.categories.map((category) => <button type="button" key={category.id} aria-pressed={manga.categoryIds.includes(category.id)} onClick={() => onToggleCategory(manga.id, category.id)}>{category.name}</button>)}</div>}<label className="studio-reader-auto-toggle"><input type="checkbox" checked={manga.autoUpdate} onChange={(event) => onSetAutoUpdate(manga.id, event.currentTarget.checked)} /> Atualizar automaticamente</label></article>)}</div></>;
}

function SavedMangaDetails({ manga, downloads, busy, onBack, onRefresh, onRemoveManga, onDownload, onRead, onTranslate, onToggleBookmark, onPause, onResume, onRetry, onRemoveDownload }: { manga: ReaderDocument["manga"][number]; downloads: ReaderDownloadRecord[]; busy: boolean; onBack(): void; onRefresh(): void; onRemoveManga(): void; onDownload(mangaId: string, chapterId: string): void; onRead(download: ReaderDownloadRecord): void; onTranslate?: (request: ReaderTranslationRequest) => void; onToggleBookmark(mangaId: string, chapterId: string): void; onPause(jobId: string): void; onResume(jobId: string): void; onRetry(jobId: string): void; onRemoveDownload(jobId: string): void }) {
  const runtimeManga: RuntimeManga = { url: manga.mangaUrl, title: manga.title, thumbnailUrl: manga.thumbnailUrl, author: manga.author, artist: manga.artist, description: manga.description, genre: manga.genres?.join(", "), status: manga.status };
  return <ReaderMangaDetails manga={runtimeManga} sourceName={manga.sourceName} chapters={manga.chapters} inLibrary busy={busy} chaptersLoading={false} chaptersError={null} onBack={onBack} onToggleLibrary={onRemoveManga} onRetryChapters={onRefresh} renderChapterAction={(runtimeChapter) => {
    const chapter = manga.chapters.find((item) => item.url === runtimeChapter.url);
    if (!chapter) return null;
    const download = downloads.find((item) => item.chapterId === chapter.id);
    return <div className="studio-reader-chapter-controls"><button className={chapter.bookmarked ? "studio-reader-bookmark active" : "studio-reader-bookmark"} type="button" aria-label={chapter.bookmarked ? `Remover bookmark de ${chapter.name}` : `Adicionar bookmark a ${chapter.name}`} title={chapter.bookmarked ? "Remover bookmark" : "Adicionar bookmark"} onClick={() => onToggleBookmark(manga.id, chapter.id)}><BookMarked size={16} /></button>{download ? <DownloadActions download={download} busy={busy} onRead={() => onRead(download)} onTranslate={download.status === "completed" && onTranslate ? () => onTranslate({ manga, chapter, download }) : undefined} onPause={onPause} onResume={onResume} onRetry={onRetry} onRemove={onRemoveDownload} /> : <button type="button" disabled={busy} onClick={() => onDownload(manga.id, chapter.id)}>{busy ? "Enfileirando…" : "Baixar"}</button>}</div>;
  }} />;
}

function OfflineReader({ download, mode, onModeChange, onProgress, onClose }: { download: ReaderDownloadRecord; mode: "vertical" | "ltr" | "rtl"; onModeChange(mode: "vertical" | "ltr" | "rtl"): void; onProgress(page: number): void; onClose(): void }) {
  const pagesRef = useRef<HTMLDivElement>(null);
  const onProgressRef = useRef(onProgress);
  onProgressRef.current = onProgress;
  useEffect(() => {
    const root = pagesRef.current;
    if (!root || typeof IntersectionObserver === "undefined") return;
    let greatest = 0;
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        // Webtoon pages can be tens of thousands of pixels tall, so they can
        // never reach a 55% intersection ratio in a desktop viewport.
        if (!entry.isIntersecting) return;
        const page = Number((entry.target as HTMLElement).dataset.page);
        if (Number.isFinite(page) && page > greatest) { greatest = page; onProgressRef.current(page); }
      });
    }, { threshold: [0] });
    root.querySelectorAll("img").forEach((image) => observer.observe(image));
    return () => observer.disconnect();
  }, [download.jobId, mode]);
  return <section className="studio-reader-offline"><header><button type="button" onClick={onClose}><ArrowLeft size={16} /> Capítulos</button><strong>{download.pageCount} páginas</strong><select aria-label="Modo de leitura" value={mode} onChange={(event) => onModeChange(event.currentTarget.value as typeof mode)}><option value="vertical">Vertical</option><option value="ltr">Paginado E → D</option><option value="rtl">Paginado D → E</option></select></header><div ref={pagesRef} className={`studio-reader-pages ${mode}`}>{download.pages.map((page) => <img key={page.id} data-page={page.number} src={readerPageUrl(page.id)} width={page.width} height={page.height} loading="lazy" alt={`Página ${page.number}`} />)}</div></section>;
}

function readerPageUrl(pageId: string) {
  const separator = pageId.indexOf(":");
  const jobId = pageId.slice(0, separator);
  const filename = pageId.slice(separator + 1);
  return `http://traduzai-reader.localhost/${encodeURIComponent(jobId)}/${encodeURIComponent(filename)}`;
}

function AutomationPanel({ settings, busy, message, eligible, onChange, onRunNow }: { settings: ReaderAutomationSettings; busy: boolean; message: string; eligible: number; onChange(settings: ReaderAutomationSettings): void; onRunNow(): void }) {
  return <><header><div><p className="eyebrow">Atualizações do leitor</p><h1>Automações</h1><p>Executadas somente enquanto o Studio estiver aberto ou minimizado.</p></div></header><section className="studio-reader-automation"><div><strong>Verificação periódica</strong><small>{eligible} obra(s) autorizaram atualização automática.</small></div><label><input type="checkbox" checked={settings.enabled} disabled={busy} onChange={(event) => onChange({ ...settings, enabled: event.currentTarget.checked })} /> Ativada</label><label>Intervalo<select aria-label="Intervalo de atualização" value={settings.intervalHours} disabled={busy} onChange={(event) => onChange({ ...settings, intervalHours: Number(event.currentTarget.value) })}><option value={6}>6 horas</option><option value={12}>12 horas</option><option value={24}>24 horas</option><option value={48}>48 horas</option></select></label><button type="button" disabled={busy || eligible === 0} onClick={onRunNow}>{busy ? "Verificando…" : "Verificar agora"}</button>{settings.lastRunAt && <small>Última execução: {new Date(settings.lastRunAt).toLocaleString("pt-BR")}</small>}{message && <p>{message}</p>}</section></>;
}

function ReaderEmpty({ icon: Icon, title, text }: { icon: typeof Compass; title: string; text: string }) {
  return <section className="studio-reader-empty"><Icon size={34} /><h2>{title}</h2><p>{text}</p></section>;
}

export function DownloadsPanel({ downloads, loaded = true, document, onRead, onPause, onResume, onRetry, onRemove }: { downloads: ReaderDownloadRecord[]; loaded?: boolean; document: ReaderDocument; onRead(download: ReaderDownloadRecord): void; onPause(jobId: string): void; onResume(jobId: string): void; onRetry(jobId: string): void; onRemove(jobId: string): void }) {
  if (!loaded) return <ReaderEmpty icon={RefreshCw} title="Carregando downloads" text="Recuperando a fila e os capítulos disponíveis offline." />;
  if (downloads.length === 0) return <ReaderEmpty icon={Download} title="Fila de downloads vazia" text="Downloads validados aparecem aqui e permanecem disponíveis offline." />;
  const completed = downloads.filter((download) => download.status === "completed").length;
  return <><header><div><p className="eyebrow">Armazenamento local</p><h1>Downloads</h1><p>{downloads.length} na fila · {completed} disponível(is) offline.</p></div></header><section className="studio-reader-chapters studio-reader-download-queue">{downloads.map((download) => { const manga = document.manga.find((item) => item.id === download.mangaId); const chapter = manga?.chapters.find((item) => item.id === download.chapterId); return <article className="reader-download-row" key={download.jobId}><span className="reader-download-cover"><ReaderCover src={manga?.thumbnailUrl} alt={manga ? `Capa de ${manga.title}` : "Capa indisponível"} /></span><div><strong>{manga?.title ?? "Obra indisponível"}</strong><small>{chapter?.name ?? "Capítulo"} · {downloadProgressLabel(download)}</small>{download.error && <span className="studio-reader-download-error">{download.error}</span>}<progress max={Math.max(download.pageCount, 1)} value={download.downloadedPages ?? download.pages.length} aria-label={`Progresso do download ${manga?.title ?? download.jobId}`} /></div><DownloadActions download={download} busy={false} onRead={() => onRead(download)} onPause={onPause} onResume={onResume} onRetry={onRetry} onRemove={onRemove} /></article>; })}</section></>;
}

export function DownloadActions({ download, busy, onRead, onTranslate, onPause, onResume, onRetry, onRemove }: { download: ReaderDownloadRecord; busy: boolean; onRead(): void; onTranslate?: () => void; onPause(jobId: string): void; onResume(jobId: string): void; onRetry(jobId: string): void; onRemove(jobId: string): void }) {
  const active = ["queued", "running", "retrying"].includes(download.status);
  return <div className="studio-reader-chapter-actions">{download.status === "completed" && <button type="button" onClick={onRead}>Ler</button>}{onTranslate && <button type="button" disabled={busy} onClick={onTranslate} title="Validar as páginas baixadas e criar um projeto editável">Preparar no Studio</button>}{active && <button type="button" disabled={busy} onClick={() => onPause(download.jobId)}>Pausar</button>}{download.status === "paused" && <button type="button" disabled={busy} onClick={() => onResume(download.jobId)}>Continuar</button>}{download.status === "failed" && <button type="button" disabled={busy} onClick={() => onRetry(download.jobId)}>Tentar novamente</button>}{!active && <button className="danger" type="button" disabled={busy} onClick={() => onRemove(download.jobId)}>Remover</button>}</div>;
}

function downloadProgressLabel(download: ReaderDownloadRecord) {
  const labels: Record<ReaderDownloadRecord["status"], string> = { queued: "Na fila", running: "Baixando", paused: "Pausado", retrying: `Tentativa ${download.attempt ?? 1}`, completed: "Concluído", failed: "Falhou" };
  const downloaded = download.downloadedPages ?? download.pages.length;
  return download.pageCount > 0 ? `${labels[download.status]} · ${downloaded} de ${download.pageCount} páginas` : `${labels[download.status]} · preparando páginas`;
}

export function HistoryPanel({ history, document, downloads, onResume }: { history: ReaderHistoryEntry[]; document: ReaderDocument; downloads: ReaderDownloadRecord[]; onResume(download: ReaderDownloadRecord): void }) {
  if (history.length === 0) return <ReaderEmpty icon={BookMarked} title="Histórico vazio" text="Quando você ler um capítulo, a página de retomada aparecerá aqui." />;
  const groups = new Map<string, ReaderHistoryEntry[]>();
  history.forEach((entry) => { const label = new Date(entry.updatedAt).toLocaleDateString("pt-BR", { timeZone: "UTC" }); groups.set(label, [...(groups.get(label) ?? []), entry]); });
  return <><header><div><p className="eyebrow">Retomar leitura</p><h1>Histórico</h1></div></header><section className="reader-history-groups">{[...groups].map(([date, entries]) => <section key={date}><h2>{date}</h2><div className="studio-reader-chapters">{entries.map((entry) => { const manga = document.manga.find((item) => item.id === entry.mangaId); const chapter = manga?.chapters.find((item) => item.id === entry.chapterId); const download = downloads.find((item) => item.chapterId === entry.chapterId); return <article className="reader-history-row" key={`${entry.mangaId}:${entry.chapterId}`}><span className="reader-history-cover"><ReaderCover src={manga?.thumbnailUrl} alt={manga ? `Capa de ${manga.title}` : "Capa indisponível"} /></span><div><strong>{manga?.title ?? "Obra indisponível"}</strong><small>{chapter?.name ?? "Capítulo"} · página {entry.lastPage}{entry.pageCount ? ` de ${entry.pageCount}` : ""}</small></div>{download && <button type="button" onClick={() => onResume(download)}>Continuar</button>}</article>; })}</div></section>)}</section></>;
}

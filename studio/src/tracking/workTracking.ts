import { invoke as tauriInvoke } from "@tauri-apps/api/core";
import type {
  ExternalTrackingSnapshot,
  PublicationStatus,
  WorkTrackingCache,
} from "../library/libraryModel";

export type WorkTrackingProvider = "anilist" | "mangadex";

export type WorkTrackingSnapshot = ExternalTrackingSnapshot;

export interface WorkTrackingIdentity {
  anilistId?: number | null;
  mangaDexId?: string | null;
  trackingLanguage?: string | null;
}

export type TrackingInvoke = <T>(command: string, args?: Record<string, unknown>) => Promise<T>;

const STATUS_MAP: Record<string, PublicationStatus> = {
  RELEASING: "releasing",
  HIATUS: "hiatus",
  FINISHED: "completed",
  CANCELLED: "cancelled",
  NOT_YET_RELEASED: "not_yet_released",
  ongoing: "releasing",
  releasing: "releasing",
  hiatus: "hiatus",
  completed: "completed",
  cancelled: "cancelled",
  not_yet_released: "not_yet_released",
  unknown: "unknown",
};

export const TRACKING_CACHE_TTL_MS = 6 * 60 * 60 * 1000;
export const UPDATES_REFRESH_TTL_MS = 30 * 60 * 1000;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function requiredString(value: unknown, field: string): string {
  if (typeof value === "string" && value.trim()) return value.trim();
  throw new Error(`Resposta de acompanhamento inválida: ${field} ausente.`);
}

function optionalString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

export function normalizeTrackingStatus(value: unknown): PublicationStatus {
  return typeof value === "string" ? STATUS_MAP[value] ?? "unknown" : "unknown";
}

export function normalizeTrackingSnapshot(value: unknown): WorkTrackingSnapshot {
  if (!isRecord(value)) throw new Error("Resposta de acompanhamento inválida.");
  const remoteChapterCount = typeof value.remoteChapterCount === "number" && Number.isFinite(value.remoteChapterCount)
    ? value.remoteChapterCount
    : null;

  return {
    provider: requiredString(value.provider, "provider"),
    providerId: requiredString(value.providerId, "providerId"),
    title: requiredString(value.title, "title"),
    status: normalizeTrackingStatus(value.status),
    remoteChapterCount,
    latestChapter: optionalString(value.latestChapter),
    coverUrl: optionalString(value.coverUrl),
    siteUrl: optionalString(value.siteUrl),
    fetchedAt: requiredString(value.fetchedAt, "fetchedAt"),
  };
}

function normalizedChapterLabel(value: string): string {
  return value
    .trim()
    .replace(/^(?:cap[ií]tulo|chapter|ch\.?)\s*/iu, "")
    .replace(",", ".")
    .replace(/\s+/g, " ")
    .toLocaleLowerCase("pt-BR");
}

function numericChapterLabel(value: string): number | null {
  const normalized = normalizedChapterLabel(value);
  if (!/^\d+(?:\.\d+)?$/.test(normalized)) return null;
  const parsed = Number.parseFloat(normalized);
  return Number.isFinite(parsed) ? parsed : null;
}

export function hasRemoteChapterUpdate(localLabels: readonly string[], latestChapter: string | null): boolean {
  if (!latestChapter?.trim()) return false;
  const remoteLabel = normalizedChapterLabel(latestChapter);
  const localNormalized = new Set(localLabels.map(normalizedChapterLabel));
  if (localNormalized.has(remoteLabel)) return false;

  const remoteNumber = numericChapterLabel(remoteLabel);
  if (remoteNumber === null) return true;
  const localNumbers = localLabels
    .map(numericChapterLabel)
    .filter((value): value is number => value !== null);
  return localNumbers.length === 0 || remoteNumber > Math.max(...localNumbers);
}

export interface ResolvedTrackingStatus {
  status: PublicationStatus;
  source: "manual" | "anilist" | "mangadex" | "local";
  hasConflict: boolean;
}

export function resolveTrackingStatus(
  localStatus: PublicationStatus,
  manualStatusOverride: PublicationStatus | null | undefined,
  snapshots: readonly WorkTrackingSnapshot[],
): ResolvedTrackingStatus {
  const usefulSnapshots = snapshots.filter((snapshot) => snapshot.status !== "unknown");
  const preferred = usefulSnapshots.find((snapshot) => snapshot.provider === "anilist") ?? usefulSnapshots[0];
  const status = manualStatusOverride ?? preferred?.status ?? localStatus;
  const source = manualStatusOverride
    ? "manual"
    : preferred?.provider === "anilist"
      ? "anilist"
      : preferred?.provider === "mangadex"
        ? "mangadex"
        : "local";
  const comparedStatuses = new Set([
    ...(manualStatusOverride ? [manualStatusOverride] : []),
    ...usefulSnapshots.map((snapshot) => snapshot.status),
  ]);

  return { status, source, hasConflict: comparedStatuses.size > 1 };
}

export function createTrackingCache(
  snapshots: readonly WorkTrackingSnapshot[],
  now = new Date(),
  ttlMs = TRACKING_CACHE_TTL_MS,
  lastError: string | null = null,
): WorkTrackingCache {
  const fetchedAt = now.toISOString();
  return {
    snapshots: snapshots.map((snapshot) => ({ ...snapshot })),
    fetchedAt,
    expiresAt: new Date(now.getTime() + Math.max(0, ttlMs)).toISOString(),
    lastError,
  };
}

export function isTrackingCacheStale(cache: WorkTrackingCache | undefined, now = new Date()): boolean {
  if (!cache) return true;
  const expiresAt = Date.parse(cache.expiresAt);
  return !Number.isFinite(expiresAt) || expiresAt <= now.getTime();
}

export function preserveTrackingCacheOnError(
  cache: WorkTrackingCache | undefined,
  error: string,
  now = new Date(),
): WorkTrackingCache {
  return cache
    ? { ...cache, snapshots: cache.snapshots.map((snapshot) => ({ ...snapshot })), lastError: error }
    : createTrackingCache([], now, 0, error);
}

function normalizeSnapshotList(value: unknown): WorkTrackingSnapshot[] {
  if (!Array.isArray(value)) throw new Error("Resposta de acompanhamento inválida: lista ausente.");
  return value.map(normalizeTrackingSnapshot);
}

export async function searchTrackingWorks(
  query: string,
  provider: WorkTrackingProvider = "anilist",
  invoke: TrackingInvoke = tauriInvoke,
): Promise<WorkTrackingSnapshot[]> {
  const normalizedQuery = query.trim();
  if (!normalizedQuery) throw new Error("Informe o nome da obra para pesquisar.");
  const snapshots = await invoke<unknown>("studio_search_tracking_works", {
    query: normalizedQuery,
    provider,
  });
  return normalizeSnapshotList(snapshots);
}

export async function syncTrackingWork(
  identity: WorkTrackingIdentity,
  invoke: TrackingInvoke = tauriInvoke,
): Promise<WorkTrackingSnapshot[]> {
  const anilistId = typeof identity.anilistId === "number" && Number.isFinite(identity.anilistId)
    ? Math.trunc(identity.anilistId)
    : null;
  const mangaDexId = optionalString(identity.mangaDexId);
  const trackingLanguage = optionalString(identity.trackingLanguage) ?? "en";
  if ((!anilistId || anilistId <= 0) && !mangaDexId) {
    throw new Error("Vincule a obra ao AniList ou MangaDex antes de atualizar.");
  }
  const snapshots = await invoke<unknown>("studio_sync_tracking_work", {
    anilistId,
    mangaDexId,
    trackingLanguage,
  });
  return normalizeSnapshotList(snapshots);
}

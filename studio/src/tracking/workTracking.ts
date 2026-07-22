import { invoke as tauriInvoke } from "@tauri-apps/api/core";
import type { PublicationStatus } from "../library/libraryModel";

export type WorkTrackingProvider = "anilist" | "mangadex";

export interface WorkTrackingSnapshot {
  provider: string;
  providerId: string;
  title: string;
  status: PublicationStatus;
  remoteChapterCount: number | null;
  coverUrl: string | null;
  siteUrl: string | null;
  fetchedAt: string;
}

export interface WorkTrackingIdentity {
  anilistId?: number | null;
  mangaDexId?: string | null;
}

export type TrackingInvoke = <T>(command: string, args?: Record<string, unknown>) => Promise<T>;

const STATUS_MAP: Record<string, PublicationStatus> = {
  RELEASING: "releasing",
  HIATUS: "hiatus",
  FINISHED: "completed",
  CANCELLED: "cancelled",
  NOT_YET_RELEASED: "not_yet_released",
  releasing: "releasing",
  hiatus: "hiatus",
  completed: "completed",
  cancelled: "cancelled",
  not_yet_released: "not_yet_released",
  unknown: "unknown",
};

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
    coverUrl: optionalString(value.coverUrl),
    siteUrl: optionalString(value.siteUrl),
    fetchedAt: requiredString(value.fetchedAt, "fetchedAt"),
  };
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
  if ((!anilistId || anilistId <= 0) && !mangaDexId) {
    throw new Error("Vincule a obra ao AniList ou MangaDex antes de atualizar.");
  }
  const snapshots = await invoke<unknown>("studio_sync_tracking_work", {
    anilistId,
    mangaDexId,
  });
  return normalizeSnapshotList(snapshots);
}

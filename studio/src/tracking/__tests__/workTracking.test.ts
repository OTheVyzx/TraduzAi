import { describe, expect, it } from "vitest";
import type { PublicationStatus } from "../../library/libraryModel";
import {
  createTrackingCache,
  hasRemoteChapterUpdate,
  isTrackingCacheStale,
  normalizeTrackingSnapshot,
  normalizeTrackingStatus,
  preserveTrackingCacheOnError,
  resolveTrackingStatus,
  searchTrackingWorks,
  syncTrackingWork,
  type TrackingInvoke,
} from "../workTracking";

const STATUS_CASES: Array<[string, PublicationStatus]> = [
  ["RELEASING", "releasing"],
  ["HIATUS", "hiatus"],
  ["FINISHED", "completed"],
  ["CANCELLED", "cancelled"],
  ["NOT_YET_RELEASED", "not_yet_released"],
  ["ongoing", "releasing"],
  ["hiatus", "hiatus"],
  ["completed", "completed"],
  ["cancelled", "cancelled"],
];

describe("workTracking", () => {
  it.each(STATUS_CASES)("maps AniList status %s to %s", (providerStatus, expected) => {
    expect(normalizeTrackingStatus(providerStatus)).toBe(expected);
  });

  it("keeps known internal statuses and falls back to unknown", () => {
    expect(normalizeTrackingStatus("completed")).toBe("completed");
    expect(normalizeTrackingStatus("DISCONTINUED")).toBe("unknown");
    expect(normalizeTrackingStatus(null)).toBe("unknown");
  });

  it("normalizes the Tauri snapshot without retaining a raw provider payload", () => {
    expect(normalizeTrackingSnapshot({
      provider: "anilist",
      providerId: "105398",
      title: "Solo Leveling",
      status: "FINISHED",
      remoteChapterCount: 200,
      latestChapter: "200",
      coverUrl: "https://s4.anilist.co/file/cover.jpg",
      siteUrl: "https://anilist.co/manga/105398",
      fetchedAt: "2026-07-22T12:00:00Z",
      ignoredRawField: { data: "must not leak" },
    })).toEqual({
      provider: "anilist",
      providerId: "105398",
      title: "Solo Leveling",
      status: "completed",
      remoteChapterCount: 200,
      latestChapter: "200",
      coverUrl: "https://s4.anilist.co/file/cover.jpg",
      siteUrl: "https://anilist.co/manga/105398",
      fetchedAt: "2026-07-22T12:00:00Z",
    });
  });

  it("routes search and sync through the Studio Tauri boundary", async () => {
    const calls: Array<{ command: string; args?: Record<string, unknown> }> = [];
    const invoke: TrackingInvoke = async <T>(command: string, args?: Record<string, unknown>) => {
      calls.push({ command, args });
      return [{
        provider: "anilist",
        providerId: "105398",
        title: "Solo Leveling",
        status: "RELEASING",
        remoteChapterCount: null,
        latestChapter: null,
        coverUrl: null,
        siteUrl: "https://anilist.co/manga/105398",
        fetchedAt: "2026-07-22T12:00:00Z",
      }] as T;
    };

    const search = await searchTrackingWorks("  Solo Leveling  ", "anilist", invoke);
    const sync = await syncTrackingWork({ anilistId: 105398 }, invoke);

    expect(search[0].status).toBe("releasing");
    expect(sync[0].providerId).toBe("105398");
    expect(calls).toEqual([
      {
        command: "studio_search_tracking_works",
        args: { query: "Solo Leveling", provider: "anilist" },
      },
      {
        command: "studio_sync_tracking_work",
        args: { anilistId: 105398, mangaDexId: null, trackingLanguage: "en" },
      },
    ]);
  });

  it("rejects empty searches and sync without an external identity", async () => {
    const invoke: TrackingInvoke = async <T>() => [] as T;

    await expect(searchTrackingWorks(" ", "anilist", invoke)).rejects.toThrow("Informe o nome da obra");
    await expect(syncTrackingWork({}, invoke)).rejects.toThrow("Vincule a obra");
  });

  it("detects decimal and special chapters without treating duplicate labels as updates", () => {
    expect(hasRemoteChapterUpdate(["10", "10.25"], "10.5")).toBe(true);
    expect(hasRemoteChapterUpdate(["10", "Extra"], "Extra")).toBe(false);
    expect(hasRemoteChapterUpdate(["10"], "Especial de verao")).toBe(true);
  });

  it("keeps a manual status visually authoritative while exposing provider conflict", () => {
    const result = resolveTrackingStatus("unknown", "hiatus", [
      normalizeTrackingSnapshot({
        provider: "anilist",
        providerId: "10",
        title: "Obra",
        status: "FINISHED",
        remoteChapterCount: 20,
        latestChapter: "20",
        coverUrl: null,
        siteUrl: null,
        fetchedAt: "2026-07-22T12:00:00Z",
      }),
      normalizeTrackingSnapshot({
        provider: "mangadex",
        providerId: "uuid",
        title: "Obra",
        status: "ongoing",
        remoteChapterCount: 20,
        latestChapter: "20",
        coverUrl: null,
        siteUrl: null,
        fetchedAt: "2026-07-22T12:00:00Z",
      }),
    ]);

    expect(result.status).toBe("hiatus");
    expect(result.source).toBe("manual");
    expect(result.hasConflict).toBe(true);
  });

  it("marks an expired normalized cache as stale while retaining its snapshots", () => {
    const snapshot = normalizeTrackingSnapshot({
      provider: "mangadex",
      providerId: "uuid",
      title: "Obra",
      status: "ongoing",
      remoteChapterCount: 11,
      latestChapter: "10.5",
      coverUrl: null,
      siteUrl: "https://mangadex.org/title/uuid",
      fetchedAt: "2026-07-22T12:00:00Z",
      raw: { discarded: true },
    });
    const cache = createTrackingCache([snapshot], new Date("2026-07-22T12:00:00Z"), 30 * 60 * 1000);

    expect(isTrackingCacheStale(cache, new Date("2026-07-22T12:31:00Z"))).toBe(true);
    expect(cache.snapshots).toEqual([snapshot]);
    expect(cache).not.toHaveProperty("raw");
  });

  it("preserves stale provider data and records the current offline error", () => {
    const snapshot = normalizeTrackingSnapshot({
      provider: "mangadex",
      providerId: "uuid",
      title: "Obra",
      status: "ongoing",
      remoteChapterCount: 11,
      latestChapter: "10.5",
      coverUrl: null,
      siteUrl: null,
      fetchedAt: "2026-07-22T10:00:00Z",
    });
    const cache = createTrackingCache([snapshot], new Date("2026-07-22T10:00:00Z"), 30 * 60 * 1000);

    const offline = preserveTrackingCacheOnError(cache, "Sem conexão", new Date("2026-07-22T12:00:00Z"));

    expect(offline.snapshots).toEqual([snapshot]);
    expect(offline.expiresAt).toBe(cache.expiresAt);
    expect(offline.lastError).toBe("Sem conexão");
    expect(isTrackingCacheStale(offline, new Date("2026-07-22T12:00:00Z"))).toBe(true);
  });
});

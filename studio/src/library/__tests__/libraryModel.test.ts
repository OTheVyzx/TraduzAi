import { describe, expect, it } from "vitest";
import {
  chapterProgress,
  createEmptyLibrary,
  normalizeLibrary,
  sortChapterEntries,
  upsertChapter,
} from "../libraryModel";

describe("libraryModel", () => {
  it("migrates an empty document to the current schema", () => {
    expect(normalizeLibrary({})).toEqual(createEmptyLibrary());
  });

  it("sorts numeric chapters before specials", () => {
    const values = ["10", "2", "2.5", "Extra"].map((label) => ({ label }));

    expect(sortChapterEntries(values).map((item) => item.label)).toEqual(["2", "2.5", "10", "Extra"]);
  });

  it("does not duplicate the same project path on Windows", () => {
    const initial = createEmptyLibrary();
    const once = upsertChapter(initial, "work-1", {
      id: "chapter-a",
      label: "1",
      projectPath: "C:/obra/project.json",
    });
    const twice = upsertChapter(once, "work-1", {
      id: "chapter-b",
      label: "1",
      projectPath: "c:\\obra\\project.json",
    });

    expect(twice.works[0].chapters).toHaveLength(1);
    expect(twice.works[0].chapters[0]).toMatchObject({
      id: "chapter-a",
      projectPath: "c:\\obra\\project.json",
    });
  });

  it("clamps chapter progress and handles an empty chapter", () => {
    expect(chapterProgress({ pageCount: 0, completedPages: 0 })).toBe(0);
    expect(chapterProgress({ pageCount: 20, completedPages: 5 })).toBe(25);
    expect(chapterProgress({ pageCount: 20, completedPages: 30 })).toBe(100);
  });

  it("normalizes malformed work and preference data defensively", () => {
    const document = normalizeLibrary({
      schemaVersion: 99,
      selectedWorkId: "missing",
      works: [{ id: "", title: "  Obra  ", chapters: "invalid" }],
      preferences: { chapterView: "rows", thumbnailSize: 999 },
    });

    expect(document.schemaVersion).toBe(1);
    expect(document.selectedWorkId).toBe(document.works[0].id);
    expect(document.works[0]).toMatchObject({
      title: "Obra",
      publicationStatus: "unknown",
      chapters: [],
    });
    expect(document.preferences).toEqual({ chapterView: "grid", thumbnailSize: 240, trackingLanguage: "en" });
  });

  it("persists only normalized tracking snapshots and the configured source language", () => {
    const document = normalizeLibrary({
      works: [{
        id: "work-1",
        title: "Obra",
        publicationStatus: "releasing",
        external: {
          mangaDexId: "uuid",
          tracking: {
            fetchedAt: "2026-07-22T12:00:00Z",
            expiresAt: "2026-07-22T12:30:00Z",
            lastError: null,
            rawResponse: { shouldDisappear: true },
            snapshots: [{
              provider: "mangadex",
              providerId: "uuid",
              title: "Obra",
              status: "ongoing",
              remoteChapterCount: 12,
              latestChapter: "10.5",
              coverUrl: null,
              siteUrl: "https://mangadex.org/title/uuid",
              fetchedAt: "2026-07-22T12:00:00Z",
              rawChapterFeed: [1, 2, 3],
            }],
          },
        },
        chapters: [],
      }],
      preferences: { trackingLanguage: "ko" },
    });

    expect(document.preferences.trackingLanguage).toBe("ko");
    expect(document.works[0].external.tracking?.snapshots[0]).toEqual({
      provider: "mangadex",
      providerId: "uuid",
      title: "Obra",
      status: "unknown",
      remoteChapterCount: 12,
      latestChapter: "10.5",
      coverUrl: null,
      siteUrl: "https://mangadex.org/title/uuid",
      fetchedAt: "2026-07-22T12:00:00Z",
    });
    expect(document.works[0].external.tracking).not.toHaveProperty("rawResponse");
  });
});

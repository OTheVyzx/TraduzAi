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
    expect(document.preferences).toEqual({ chapterView: "grid", thumbnailSize: 240 });
  });
});

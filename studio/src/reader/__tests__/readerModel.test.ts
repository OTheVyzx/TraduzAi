import { describe, expect, it } from "vitest";
import { addReaderManga, createEmptyReaderDocument, markReaderProgress, normalizeGenres, parseReaderDocument, toggleReaderBookmark } from "../readerModel";

describe("readerModel", () => {
  it("starts without bundled repositories or extensions", () => {
    const reader = createEmptyReaderDocument();
    expect(reader.repositories).toEqual([]);
    expect(reader.extensions).toEqual([]);
    expect(reader.manga).toEqual([]);
  });

  it("keeps Long source ids as strings and resumes page progress", () => {
    const initial = addReaderManga(createEmptyReaderDocument(), {
      id: "reader-1",
      extensionPackage: "ext.fixture",
      sourceId: "9223372036854775807",
      sourceName: "Fixture PT",
      mangaUrl: "/manga/luna",
      title: "Luna de Teste",
      chapters: [{ id: "chapter-1", url: "/chapter/1", name: "Capítulo 1", read: false, bookmarked: false, lastPageRead: 0, pageCount: 12 }],
      categoryIds: [],
      favorite: true,
      autoUpdate: false,
      autoDownload: false,
    });
    const updated = markReaderProgress(initial, "reader-1", "chapter-1", 7);
    const parsed = parseReaderDocument(JSON.parse(JSON.stringify(updated)));

    expect(parsed.manga[0].sourceId).toBe("9223372036854775807");
    expect(parsed.manga[0].chapters[0].lastPageRead).toBe(7);
    expect(parsed.manga[0].chapters[0].read).toBe(false);
  });

  it("toggles only the selected chapter bookmark", () => {
    const initial = addReaderManga(createEmptyReaderDocument(), {
      id: "reader-1",
      extensionPackage: "ext.fixture",
      sourceId: "1",
      sourceName: "Fixture PT",
      mangaUrl: "/manga/luna",
      title: "Luna de Teste",
      chapters: [
        { id: "chapter-1", url: "/chapter/1", name: "Capítulo 1", read: false, bookmarked: false, lastPageRead: 0, pageCount: 12 },
        { id: "chapter-2", url: "/chapter/2", name: "Capítulo 2", read: false, bookmarked: true, lastPageRead: 0, pageCount: 10 },
      ],
      categoryIds: [],
      favorite: true,
      autoUpdate: false,
      autoDownload: false,
    });

    const updated = toggleReaderBookmark(initial, "reader-1", "chapter-1");

    expect(updated.manga[0].chapters.map((chapter) => chapter.bookmarked)).toEqual([true, true]);
  });

  it("preserves optional Mihon metadata in existing version-one reader records", () => {
    const parsed = parseReaderDocument({
      ...createEmptyReaderDocument(),
      manga: [{
        id: "reader-details",
        extensionPackage: "ext.fixture",
        sourceId: "1",
        sourceName: "Fixture PT",
        mangaUrl: "/manga/details",
        title: "Obra detalhada",
        thumbnailUrl: "https://covers.invalid/details.webp",
        author: "Autora",
        artist: "Artista",
        description: "Sinopse persistida",
        genres: ["Ação", "Fantasia"],
        status: 1,
        chapters: [],
        categoryIds: [],
        favorite: true,
        autoUpdate: false,
        autoDownload: false,
      }],
    });

    expect(parsed.manga[0]).toMatchObject({
      author: "Autora",
      artist: "Artista",
      description: "Sinopse persistida",
      genres: ["Ação", "Fantasia"],
      status: 1,
    });
  });

  it("normalizes the comma-separated genre field returned by Mihon extensions", () => {
    expect(normalizeGenres(" Ação, Fantasia,  Ação ,, Drama ")).toEqual(["Ação", "Fantasia", "Drama"]);
    expect(normalizeGenres(undefined)).toEqual([]);
  });
});

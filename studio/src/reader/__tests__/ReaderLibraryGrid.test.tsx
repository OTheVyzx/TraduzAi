import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { ReaderLibraryGrid } from "../ReaderLibraryGrid";
import { createEmptyReaderDocument } from "../readerModel";

describe("ReaderLibraryGrid", () => {
  it("renders saved works as cover cards with unread count and compact actions", () => {
    const document = {
      ...createEmptyReaderDocument(),
      manga: [{
        id: "m1", extensionPackage: "ext.flix", sourceId: "20", sourceName: "MangaFlix", mangaUrl: "/solo", title: "Solo Max-Level Newbie",
        thumbnailUrl: "https://covers.invalid/solo.webp",
        chapters: [{ id: "c1", url: "/c1", name: "Capítulo 1", read: false, bookmarked: false, lastPageRead: 0, pageCount: null }],
        categoryIds: [], favorite: true, autoUpdate: false, autoDownload: false,
      }],
    };
    const html = renderToStaticMarkup(createElement(ReaderLibraryGrid, {
      document,
      newCategoryName: "",
      onNewCategoryName: () => undefined,
      onAddCategory: () => undefined,
      onToggleCategory: () => undefined,
      onOpen: () => undefined,
      onSetAutoUpdate: () => undefined,
      onRemove: () => undefined,
    }));

    expect(html).toContain('src="https://covers.invalid/solo.webp"');
    expect(html).toContain("Solo Max-Level Newbie");
    expect(html).toContain("1 não lido");
    expect(html).toContain('aria-label="Abrir Solo Max-Level Newbie"');
    expect(html).toContain('aria-label="Ações de Solo Max-Level Newbie"');
    expect(html).toContain('class="reader-media-cover reader-library-ambient"');
    expect(html).not.toContain("reader-library-card-gradient");
    expect(html).not.toContain("Atualizar automaticamente</label>");
  });
});

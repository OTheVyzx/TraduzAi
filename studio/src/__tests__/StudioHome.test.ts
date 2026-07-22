import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import {
  ChapterBrowser,
  nextChapterSelection,
  shouldHandleChapterArrowKey,
} from "../library/ChapterBrowser";
import { StudioLibraryHome } from "../library/StudioLibraryHome";
import { WorkLibrarySidebar } from "../library/WorkLibrarySidebar";
import { createEmptyLibrary, type LibraryWork } from "../library/libraryModel";

const work: LibraryWork = {
  id: "work-1",
  title: "A Espada do Norte",
  aliases: ["Northern Blade"],
  publicationStatus: "releasing",
  external: {},
  chapters: [
    {
      id: "chapter-2",
      label: "2",
      projectPath: "N:/Obras/Norte/002/project.json",
      pageCount: 24,
      completedPages: 6,
      workflowStatus: "editing",
    },
  ],
};

describe("StudioLibraryHome", () => {
  it("shows works and disables chapter creation while no work is selected", () => {
    const html = renderToStaticMarkup(createElement(StudioLibraryHome, {
      document: createEmptyLibrary(),
      status: "ready",
      onSaveWork: () => undefined,
      onRemoveWork: () => undefined,
      onAttachChapter: () => undefined,
      onCreateManualChapter: async () => undefined,
      onRemoveChapter: () => undefined,
      onRelinkChapter: () => undefined,
      onImportProject: () => undefined,
      onSelectWork: () => undefined,
      onOpenChapter: () => undefined,
      onSetChapterView: () => undefined,
      onSetThumbnailSize: () => undefined,
    }));

    expect(html).toContain("Obras");
    expect(html).toContain("Adicionar obra");
    expect(html).toContain("Capítulos");
    expect(html).toContain("Adicionar capítulo");
    expect(html).toContain("Importar projeto");
    expect(html).toContain("disabled");
    expect(html).not.toContain("Novo projeto");
  });

  it("renders the selected work and its chapter grid", () => {
    const document = {
      ...createEmptyLibrary(),
      selectedWorkId: work.id,
      works: [work],
      preferences: { chapterView: "grid" as const, thumbnailSize: 192, trackingLanguage: "en" },
    };
    const html = renderToStaticMarkup(createElement(StudioLibraryHome, {
      document,
      status: "ready",
      onSaveWork: () => undefined,
      onRemoveWork: () => undefined,
      onAttachChapter: () => undefined,
      onCreateManualChapter: async () => undefined,
      onRemoveChapter: () => undefined,
      onRelinkChapter: () => undefined,
      onImportProject: () => undefined,
      onSelectWork: () => undefined,
      onOpenChapter: () => undefined,
      onSetChapterView: () => undefined,
      onSetThumbnailSize: () => undefined,
      initialSelectedChapterPath: "N:/Obras/Norte/002/project.json",
    }));

    expect(html).toContain("A Espada do Norte");
    expect(html).toContain("Capítulo 2");
    expect(html).toContain("6 de 24 páginas");
    expect(html).toContain("--chapter-card-size:192px");
    expect(html).toMatch(/aria-label="Selecionar capítulo 2" aria-pressed="true"/);
  });

  it("filters the work sidebar and exposes selection state", () => {
    const html = renderToStaticMarkup(createElement(WorkLibrarySidebar, {
      works: [work, { ...work, id: "work-2", title: "Outra obra" }],
      selectedWorkId: work.id,
      query: "espada",
      onQueryChange: () => undefined,
      onSelectWork: () => undefined,
      onAddWork: () => undefined,
    }));

    expect(html).toContain("A Espada do Norte");
    expect(html).not.toContain("Outra obra");
    expect(html).toContain('aria-current="true"');
  });

  it("enables opening after a chapter is selected", () => {
    const html = renderToStaticMarkup(createElement(ChapterBrowser, {
      work,
      view: "list",
      thumbnailSize: 176,
      selectedChapterId: "chapter-2",
      onSelectChapter: () => undefined,
      onOpenChapter: () => undefined,
    }));

    expect(html).toContain('aria-pressed="true"');
    expect(html).toContain("Abrir");
    expect(html).toMatch(/class="studio-library-open"[^>]*><svg/);
    expect(html).not.toMatch(/class="studio-library-open"[^>]*disabled/);
  });

  it("keeps a missing chapter searchable by path and exposes relinking", () => {
    const html = renderToStaticMarkup(createElement(ChapterBrowser, {
      work,
      query: "obras/norte/002",
      view: "grid",
      thumbnailSize: 176,
      selectedChapterId: null,
      missingProjectPaths: new Set([work.chapters[0].projectPath]),
      onSelectChapter: () => undefined,
      onOpenChapter: () => undefined,
      onRelinkChapter: () => undefined,
    }));

    expect(html).toContain("Capítulo 2");
    expect(html).toContain("Caminho ausente");
    expect(html).toContain("Relocalizar");
  });

  it("moves chapter selection in every grid direction without capturing form fields", () => {
    const chapters = ["a", "b", "c", "d", "e"].map((id) => ({ id }));

    expect(nextChapterSelection(chapters, "a", "ArrowRight", 2)).toBe("b");
    expect(nextChapterSelection(chapters, "a", "ArrowDown", 2)).toBe("c");
    expect(nextChapterSelection(chapters, "d", "ArrowLeft", 2)).toBe("c");
    expect(nextChapterSelection(chapters, "e", "ArrowUp", 2)).toBe("c");
    expect(shouldHandleChapterArrowKey("ArrowRight", { tagName: "INPUT", isContentEditable: false })).toBe(false);
    expect(shouldHandleChapterArrowKey("ArrowRight", { tagName: "BUTTON", isContentEditable: false })).toBe(true);
  });
});

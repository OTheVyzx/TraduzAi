import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { WorkInspector } from "../WorkInspector";
import type { LibraryWork } from "../libraryModel";

const work: LibraryWork = {
  id: "gosu",
  title: "Gosu",
  aliases: [],
  description: "Um mestre retorna ao mundo marcial.",
  genres: ["Ação", "Artes Marciais"],
  sourceLanguage: "ko",
  targetLanguage: "pt-BR",
  publicationStatus: "releasing",
  coverPath: "memory://covers/gosu.jpg",
  external: {},
  chapters: [
    { id: "one", label: "1", projectPath: "memory://gosu/1", pageCount: 10, completedPages: 10, workflowStatus: "completed" },
    { id: "two", label: "2", projectPath: "memory://gosu/2", pageCount: 10, completedPages: 5, workflowStatus: "editing" },
  ],
};

describe("WorkInspector", () => {
  it("shows selected-work metadata and derives progress from chapters", () => {
    const html = renderToStaticMarkup(createElement(WorkInspector, { work, onEditWork: () => undefined }));

    expect(html).toContain("Gosu");
    expect(html).toContain("Ação");
    expect(html).toContain("Artes Marciais");
    expect(html).toContain("KO");
    expect(html).toContain("PT-BR");
    expect(html).toContain("75% concluído");
    expect(html).toContain("1 capítulo concluído");
    expect(html).toContain("Editar obra");
    expect(html).toContain("studio-work-inspector-cover-wrap");
    expect(html).toContain("studio-work-inspector-cover-ambient");
    expect(html).toContain('aria-hidden="true"');
    expect(html.match(/<img[^>]+src="memory:\/\/covers\/gosu\.jpg"/g)).toHaveLength(2);
    expect(html).not.toContain("58% concluído");
  });

  it("does not render ambient light when the work has no cover", () => {
    const html = renderToStaticMarkup(createElement(WorkInspector, {
      work: { ...work, coverPath: null },
      onEditWork: () => undefined,
    }));

    expect(html).toContain("studio-work-inspector-cover-wrap");
    expect(html).not.toContain("studio-work-inspector-cover-ambient");
  });
});

import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { CreateChapterDialog, validateCreateChapterDraft } from "../CreateChapterDialog";
import type { LibraryWork } from "../libraryModel";

const work: LibraryWork = {
  id: "work-1",
  title: "A Espada do Norte",
  aliases: [],
  publicationStatus: "releasing",
  external: {},
  chapters: [],
};

describe("CreateChapterDialog", () => {
  it("requires a source, chapter label and project destination", () => {
    expect(validateCreateChapterDraft({
      chapterLabel: "",
      sourceLanguage: "ko",
      targetLanguage: "pt-BR",
      sourcePath: null,
      projectJsonPath: null,
    })).toBe("Informe o capítulo.");
  });

  it("offers folder, ZIP/CBZ and existing-project routes without editable paths", () => {
    const html = renderToStaticMarkup(createElement(CreateChapterDialog, {
      open: true,
      work,
      onClose: () => undefined,
      onChooseFolder: async () => null,
      onChooseArchive: async () => null,
      onChooseDestination: async () => null,
      onAttachExisting: () => undefined,
      onCreate: async () => undefined,
    }));

    expect(html).toContain("Criar capítulo manual");
    expect(html).toContain("Pasta de imagens");
    expect(html).toContain("ZIP ou CBZ");
    expect(html).toContain("Anexar project.json existente");
    expect(html).not.toContain('name="sourcePath"');
    expect(html).not.toContain('name="projectJsonPath"');
    expect(html).toMatch(/type="submit"[^>]*disabled/);
  });
});

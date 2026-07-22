import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { AttachProjectDialog, type ProjectAttachmentDraft } from "../AttachProjectDialog";
import { WorkDialog, validateWorkDraft } from "../WorkDialog";
import type { LibraryWork } from "../libraryModel";

const work: LibraryWork = {
  id: "work-1",
  title: "A Espada do Norte",
  aliases: ["Northern Blade"],
  publicationStatus: "releasing",
  external: {},
  chapters: [{ id: "chapter-1", label: "1", projectPath: "N:/Norte/001/project.json" }],
};

const attachment: ProjectAttachmentDraft = {
  projectPath: "N:/Norte/001/project.json",
  workTitle: "A Espada do Norte",
  chapterLabel: "1",
  pageCount: 24,
  coverPath: "N:/Norte/001/001.png",
};

describe("WorkDialog", () => {
  it("rejects an empty title", () => {
    expect(validateWorkDraft({ title: "  ", aliases: "" })).toBe("Informe o título da obra.");
  });

  it("shows editable metadata and a non-destructive removal warning", () => {
    const html = renderToStaticMarkup(createElement(WorkDialog, {
      open: true,
      work,
      onClose: () => undefined,
      onSave: () => undefined,
      onRemove: () => undefined,
    }));

    expect(html).toContain('value="A Espada do Norte"');
    expect(html).toContain('value="Northern Blade"');
    expect(html).toContain("não apaga capítulos nem arquivos do disco");
    expect(html).toContain("Remover da biblioteca");
  });
});

describe("AttachProjectDialog", () => {
  it("requires explicit confirmation before updating a duplicate attachment", () => {
    const html = renderToStaticMarkup(createElement(AttachProjectDialog, {
      open: true,
      work,
      draft: attachment,
      onChooseProject: async () => null,
      onClose: () => undefined,
      onConfirm: () => undefined,
    }));

    expect(html).toContain("Este project.json já está anexado");
    expect(html).toContain("Confirmar atualização da referência");
    expect(html).toMatch(/type="submit"[^>]*disabled/);
    expect(html).not.toContain('name="projectPath"');
  });

  it("presents metadata read from the selected project", () => {
    const html = renderToStaticMarkup(createElement(AttachProjectDialog, {
      open: true,
      work: { ...work, chapters: [] },
      draft: attachment,
      onChooseProject: async () => null,
      onClose: () => undefined,
      onConfirm: () => undefined,
    }));

    expect(html).toContain("N:/Norte/001/project.json");
    expect(html).toContain('value="1"');
    expect(html).toContain("24 páginas");
  });
});

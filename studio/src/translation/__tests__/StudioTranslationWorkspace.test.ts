import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import {
  editorToolsForMode,
  editorViewLabelForMode,
  isEditorViewAvailable,
  resolveEditorCapabilities,
} from "../../../../src/components/editor/editorMode";
import { importStudioProject } from "../../project/adapters";
import type { StudioProject } from "../../project/studioProject";
import { GlossaryPanel } from "../GlossaryPanel";
import {
  StudioTranslationWorkspace,
  createTranslationPatch,
  findAdjacentTranslationTarget,
  findNextPendingTranslationTarget,
  translationUsesStudioComposite,
  translationTargetRequiresPageChange,
  translationShortcutFor,
} from "../StudioTranslationWorkspace";
import { TranslationInspector } from "../TranslationInspector";
import { TranslationQueuePanel } from "../TranslationQueuePanel";

function createProject(): StudioProject {
  return importStudioProject({
    versao: "2.0",
    obra: "A Espada do Norte",
    work_context: { glossary: { Murim: "mundo marcial" } },
    paginas: [
      {
        numero: 1,
        arquivo_original: "original/001.png",
        textos: [
          { id: "p1-a", bbox: [0, 0, 10, 10], texto: "FIRST", traduzido: "" },
          {
            id: "p1-b",
            bbox: [0, 10, 10, 20],
            texto: "SECOND",
            traduzido: "Segundo",
            translation_status: "review",
          },
        ],
      },
      {
        numero: 2,
        arquivo_original: "original/002.png",
        image_layers: { inpaint: { path: "inpaint/002.png" } },
        textos: [
          { id: "p2-a", bbox: [0, 0, 10, 10], texto: "THIRD", traduzido: "" },
          {
            id: "p2-b",
            bbox: [0, 10, 10, 20],
            texto: "FOURTH",
            traduzido: "Quarto",
            translation_status: "approved",
          },
        ],
      },
    ],
  }).project;
}

describe("StudioTranslationWorkspace", () => {
  it("keeps translation mode focused on text selection, creation, pan and zoom", () => {
    expect(editorToolsForMode("studio-translation")).toEqual(["select", "block"]);
    expect(resolveEditorCapabilities("studio-translation").showTypesettingControls).toBe(false);
    expect(editorViewLabelForMode("studio-translation", "translated")).toBe("Traduzida");
    expect(isEditorViewAvailable("studio-translation", "inpainted", false)).toBe(false);
    expect(isEditorViewAvailable("studio-translation", "inpainted", true)).toBe(true);
    expect(translationUsesStudioComposite("original")).toBe(false);
    expect(translationUsesStudioComposite("inpainted")).toBe(false);
    expect(translationUsesStudioComposite("translated")).toBe(true);
  });

  it("builds a translation patch without changing the original and advances to the next pending block", () => {
    const project = createProject();
    const patch = createTranslationPatch({
      translated: "Primeiro",
      type: "fala",
      notes: "Checar tratamento",
      status: "review",
    });
    const reviewedProject = structuredClone(project);
    Object.assign(reviewedProject.paginas[0].text_layers[0], patch);

    expect(patch).toEqual({
      translated: "Primeiro",
      traduzido: "Primeiro",
      tipo: "fala",
      translation_notes: "Checar tratamento",
      translation_status: "review",
    });
    expect(patch).not.toHaveProperty("original");
    expect(findNextPendingTranslationTarget(reviewedProject, 0, "p1-a")).toEqual({
      pageIndex: 1,
      layerId: "p2-a",
    });
  });

  it("navigates blocks with Alt arrows and reserves Ctrl+Enter for confirm-and-next", () => {
    const project = createProject();

    expect(findAdjacentTranslationTarget(project, 0, "p1-a", "next")).toEqual({
      pageIndex: 0,
      layerId: "p1-b",
    });
    expect(findAdjacentTranslationTarget(project, 0, "p1-a", "previous")).toEqual({
      pageIndex: 1,
      layerId: "p2-b",
    });
    expect(translationTargetRequiresPageChange(0, { pageIndex: 0, layerId: "p1-b" })).toBe(false);
    expect(translationTargetRequiresPageChange(0, { pageIndex: 1, layerId: "p2-a" })).toBe(true);
    expect(translationShortcutFor({ key: "Enter", ctrlKey: true })).toBe("confirm-next");
    expect(translationShortcutFor({ key: "ArrowDown", altKey: true })).toBe("next-block");
    expect(translationShortcutFor({ key: "ArrowUp", altKey: true })).toBe("previous-block");
    expect(translationShortcutFor({ key: "b", editableTarget: true })).toBeNull();
  });

  it("renders queue filters, block progress and page indicators", () => {
    const html = renderToStaticMarkup(createElement(TranslationQueuePanel, {
      project: createProject(),
      filter: "all",
      currentPageIndex: 0,
      selectedLayerId: "p1-a",
      onFilterChange: () => undefined,
      onSelectTarget: () => undefined,
    }));

    expect(html).toContain("Fila de tradução");
    expect(html).toContain("Todos");
    expect(html).toContain("Pendentes");
    expect(html).toContain("Revisão");
    expect(html).toContain("Aprovados");
    expect(html).toContain("50%");
    expect(html).toContain("Página 1");
    expect(html).toContain("Página 2");
    expect(html).toContain('aria-pressed="true"');
  });

  it("renders read-only source, editable translation, notes, status and local glossary", () => {
    const project = createProject();
    const layer = project.paginas[0].text_layers[0];
    const inspector = renderToStaticMarkup(createElement(TranslationInspector, {
      layer,
      onChange: () => undefined,
      onConfirmNext: () => undefined,
    }));
    const workspace = renderToStaticMarkup(createElement(StudioTranslationWorkspace, {
      project,
      layer,
      onChange: () => undefined,
      onConfirmNext: () => undefined,
      onNavigateBlock: () => undefined,
      onUpdateGlossary: () => undefined,
    }));
    const glossary = renderToStaticMarkup(createElement(GlossaryPanel, {
      glossary: { Murim: "mundo marcial" },
      onChange: () => undefined,
    }));

    expect(inspector).toContain("Original");
    expect(inspector).toContain('readOnly=""');
    expect(inspector).toContain("Tradução");
    expect(inspector).toContain("Notas");
    expect(inspector).toContain("Revisão");
    expect(inspector).toContain("Confirmar e próximo");
    expect(workspace).toContain("Tradução manual");
    expect(workspace).toContain("Glossário do projeto");
    expect(workspace).toContain('data-editor-preserve-text-selection="true"');
    expect(glossary).toContain("Murim");
    expect(glossary).toContain("mundo marcial");
  });
});

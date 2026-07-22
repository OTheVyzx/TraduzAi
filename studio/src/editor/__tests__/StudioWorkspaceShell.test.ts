import { createElement, type ReactNode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { importStudioProject } from "../../project/adapters";
import {
  StudioWorkspaceShell,
  StudioWorkspaceSwitcher,
} from "../StudioWorkspaceShell";
import {
  defaultStudioWorkspace,
  requestWorkspaceClose,
} from "../studioWorkspace";

let capturedWorkspace: string | null = null;

vi.mock("../StudioSharedEditor", () => ({
  StudioSharedEditor: ({ workspace, workspaceSwitcher }: { workspace: string; workspaceSwitcher: ReactNode }) => {
    capturedWorkspace = workspace;
    return createElement("section", { "data-workspace": workspace }, workspaceSwitcher);
  },
}));

function project(input: Record<string, unknown>) {
  return importStudioProject({
    versao: "2.0",
    studio_schema_version: "1.0",
    paginas: [{ numero: 1, arquivo_original: "original/1.png", textos: [] }],
    ...input,
  }).project;
}

describe("studioWorkspace", () => {
  beforeEach(() => {
    capturedWorkspace = null;
    localStorageMock.clear();
  });

  it("defaults translated projects to editing and empty manual projects to translation", () => {
    expect(defaultStudioWorkspace(project({ obra: "Traduzida" }))).toBe("editing");
    expect(defaultStudioWorkspace(project({
      obra: "Manual",
      work_context: { manual_chapter: true },
    }))).toBe("translation");
    expect(defaultStudioWorkspace(project({
      obra: "Manual traduzida",
      work_context: { manual_chapter: true },
      paginas: [{
        numero: 1,
        arquivo_original: "original/1.png",
        textos: [{
          id: "text-1",
          original: "Hello",
          traduzido: "Olá",
          bbox: [0, 0, 100, 50],
        }],
      }],
    }))).toBe("editing");
  });

  it("renders the upper-right workspace selector without replacing the editor", () => {
    const manual = project({ obra: "Manual", work_context: { manual_chapter: true } });
    const html = renderToStaticMarkup(createElement(StudioWorkspaceShell, {
      project: manual,
      projectPath: "N:/Manual/1/project.json",
      onBack: () => undefined,
      storage: localStorageMock,
      confirmDiscard: () => true,
    }));

    expect(capturedWorkspace).toBe("translation");
    expect(html).toContain("Tradução");
    expect(html).toContain("Edição");
    expect(html).toContain('aria-pressed="true"');
  });

  it("switches using the selector callback and confirms only dirty closes", () => {
    let selected = "translation";
    const switcher = StudioWorkspaceSwitcher({
      workspace: "translation",
      onChange: (workspace) => { selected = workspace; },
    });
    const editingButton = (switcher.props.children as Array<{ props: { onClick: () => void } }>)[1];
    editingButton.props.onClick();

    let confirmations = 0;
    expect(requestWorkspaceClose(false, () => { confirmations += 1; return false; })).toBe(true);
    expect(requestWorkspaceClose(true, () => { confirmations += 1; return false; })).toBe(false);
    expect(selected).toBe("editing");
    expect(confirmations).toBe(1);
  });
});

const localStorageMock = (() => {
  const values = new Map<string, string>();
  return {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => { values.set(key, value); },
    removeItem: (key: string) => { values.delete(key); },
    clear: () => values.clear(),
    key: (index: number) => [...values.keys()][index] ?? null,
    get length() { return values.size; },
  } satisfies Storage;
})();

import { beforeEach, describe, expect, it } from "vitest";
import type { EditorBackendApi } from "../../../../src/lib/editorBackend";
import { configureEditorBackend } from "../../../../src/lib/editorBackend";
import type { PageData, Project, TextEntry, TextLayerStyle } from "../../../../src/lib/stores/appStore";
import { useAppStore } from "../../../../src/lib/stores/appStore";
import { useEditorStore } from "../../../../src/lib/stores/editorStore";

const projectPath = "memory://translation-commit";

function textLayer(): TextEntry {
  return {
    id: "translation-layer",
    bbox: [0, 0, 100, 100],
    tipo: "fala",
    original: "Source",
    traduzido: "Destino",
    translated: "Destino",
    confianca_ocr: 1,
    estilo: {} as TextLayerStyle,
  };
}

function pageFixture(): PageData {
  const layer = textLayer();
  return {
    numero: 1,
    arquivo_original: "original.png",
    arquivo_traduzido: "translated.png",
    image_layers: {},
    text_layers: [layer],
    textos: [layer],
  };
}

function projectFixture(page: PageData): Project {
  return {
    id: "translation-project",
    obra: "Obra",
    capitulo: 1,
    idioma_origem: "en",
    idioma_destino: "pt-BR",
    qualidade: "normal",
    contexto: {} as Project["contexto"],
    paginas: [page],
    status: "done",
    source_path: projectPath,
    output_path: projectPath,
    totalPages: 1,
    mode: "manual",
  };
}

describe("persistência de metadados da tradução manual", () => {
  let patches: Array<Record<string, unknown>>;
  let persistedPage: PageData;

  beforeEach(() => {
    patches = [];
    persistedPage = pageFixture();
    configureEditorBackend({
      patchEditorTextLayer: async ({ patch }) => {
        patches.push(patch);
        const layer = { ...persistedPage.text_layers[0], ...patch } as TextEntry;
        persistedPage = { ...persistedPage, text_layers: [layer], textos: [layer] };
        return layer;
      },
      loadEditorPage: async () => ({ page_index: 0, page: persistedPage }),
    } as Partial<EditorBackendApi> as EditorBackendApi);
    useAppStore.setState({ project: projectFixture(persistedPage) });
    useEditorStore.setState({
      currentPageIndex: 0,
      currentPage: persistedPage,
      pendingEdits: {},
      pendingStructuralEdits: { created: [], deleted: {}, order: undefined },
    });
  });

  it.each(["commitEdits", "commitEditsPatchOnly"] as const)(
    "encaminha status e notas pelo caminho %s",
    async (commitMethod) => {
      useEditorStore.setState({
        pendingEdits: {
          "translation-layer": {
            translated: "Texto revisado",
            traduzido: "Texto revisado",
            translation_status: "review",
            translation_notes: "Manter o tratamento formal",
          } as Partial<TextEntry>,
        },
      });

      await useEditorStore.getState()[commitMethod]();

      expect(patches).toHaveLength(1);
      expect(patches[0]).toMatchObject({
        translated: "Texto revisado",
        translation_status: "review",
        translation_notes: "Manter o tratamento formal",
      });
    },
  );

  it("persists a manual region as pending analysis instead of pretending OCR confidence", async () => {
    let persistedProject = projectFixture(pageFixture());
    configureEditorBackend({
      saveProjectJson: async ({ project_json }) => {
        persistedProject = structuredClone(project_json as Project);
      },
      loadEditorPage: async ({ page_index }) => ({
        page_index,
        page: structuredClone(persistedProject.paginas[page_index]),
        project: structuredClone(persistedProject),
      }),
    } as Partial<EditorBackendApi> as EditorBackendApi);
    useAppStore.setState({ project: persistedProject });
    useEditorStore.setState({
      currentPageIndex: 0,
      currentPage: structuredClone(persistedProject.paginas[0]),
      pendingEdits: {},
      pendingStructuralEdits: { created: [], deleted: {}, order: undefined },
    });

    await useEditorStore.getState().createTextLayer([10, 20, 80, 90]);
    const created = useEditorStore.getState().currentPage?.text_layers.at(-1);
    expect(created).toMatchObject({
      source_bbox: [10, 20, 80, 90],
      layout_bbox: [10, 20, 80, 90],
      render_bbox: null,
      translation_status: "pending",
      confianca_ocr: 0,
      ocr_confidence: 0,
      qa_flags: ["manual_region_requires_analysis"],
      style_origin: "editor",
    });

    await useEditorStore.getState().commitEdits();
    expect(persistedProject.paginas[0].text_layers.at(-1)).toMatchObject({
      id: created?.id,
      qa_flags: ["manual_region_requires_analysis"],
      render_preview_path: null,
    });
    expect(useEditorStore.getState().pendingStructuralEdits.created).toEqual([]);
  });
});

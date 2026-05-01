import { beforeEach, describe, expect, it, vi } from "vitest";
import type { PageData, Project, TextEntry, TextLayerStyle } from "../appStore";
import { useAppStore } from "../appStore";
import { useEditorStore } from "../editorStore";

const invokeMock = vi.hoisted(() => vi.fn());

vi.mock("@tauri-apps/api/core", () => ({
  invoke: invokeMock,
}));

vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn(),
}));

function makeStyle(): TextLayerStyle {
  return {
    fonte: "ComicNeue-Bold.ttf",
    tamanho: 28,
    cor: "#ffffff",
    cor_gradiente: [],
    contorno: "#000000",
    contorno_px: 2,
    glow: false,
    glow_cor: "",
    glow_px: 0,
    sombra: false,
    sombra_cor: "",
    sombra_offset: [0, 0],
    bold: false,
    italico: false,
    rotacao: 0,
    alinhamento: "center",
  };
}

function makeLayer(overrides: Partial<TextEntry> = {}): TextEntry {
  const estilo = makeStyle();
  return {
    id: "layer-a",
    kind: "text",
    source_bbox: [0, 0, 100, 100],
    layout_bbox: [0, 0, 100, 100],
    render_bbox: null,
    bbox: [0, 0, 100, 100],
    tipo: "fala",
    original: "Hello",
    traduzido: "Ola",
    translated: "Ola",
    confianca_ocr: 1,
    ocr_confidence: 1,
    estilo,
    style: estilo,
    visible: true,
    locked: false,
    order: 0,
    ...overrides,
  };
}

function makePage(pageNumber: number, layers = [makeLayer()]): PageData {
  return {
    numero: pageNumber,
    arquivo_original: `originals/${String(pageNumber).padStart(3, "0")}.png`,
    arquivo_traduzido: `translated/${String(pageNumber).padStart(3, "0")}.png`,
    image_layers: {
      base: {
        key: "base",
        path: `originals/${String(pageNumber).padStart(3, "0")}.png`,
        visible: true,
        locked: true,
      },
      rendered: {
        key: "rendered",
        path: `translated/${String(pageNumber).padStart(3, "0")}.png`,
        visible: true,
        locked: true,
      },
    },
    inpaint_blocks: [],
    text_layers: layers,
    textos: layers,
  };
}

function makeProject(pages: PageData[]): Project {
  return {
    id: "project-a",
    obra: "Obra",
    capitulo: 1,
    idioma_origem: "en",
    idioma_destino: "pt-BR",
    qualidade: "normal",
    contexto: {
      sinopse: "",
      genero: [],
      personagens: [],
      glossario: {},
      aliases: [],
      termos: [],
      relacoes: [],
      faccoes: [],
      resumo_por_arco: [],
      memoria_lexical: {},
      fontes_usadas: [],
    },
    paginas: pages,
    status: "done",
    source_path: "D:/tmp/project",
    totalPages: pages.length,
    mode: "manual",
  };
}

beforeEach(() => {
  invokeMock.mockReset();
  const page = makePage(1);
  useAppStore.setState({ project: makeProject([page]) });
  useEditorStore.setState({
    currentPageIndex: 0,
    currentPage: page,
    pendingEdits: {},
    pendingStructuralEdits: { created: [], deleted: {}, order: undefined },
    historyByPageKey: {},
    renderPreviewCacheByPageKey: {},
  });
});

describe("editor render preview cache", () => {
  it("marks the current page preview stale when text, bbox, style or visibility changes", () => {
    const pageKey = useEditorStore.getState().currentPageKey();

    expect(useEditorStore.getState().getRenderPreviewState(pageKey).status).toBe("fresh");

    useEditorStore.getState().setWorkingTraduzido(pageKey, "layer-a", "Oi");
    expect(useEditorStore.getState().getRenderPreviewState(pageKey).status).toBe("stale");

    useEditorStore.getState().markRenderPreviewFresh(pageKey, "D:/tmp/project/translated/001.png");
    useEditorStore.getState().setWorkingBbox(pageKey, "layer-a", [10, 20, 120, 140]);
    expect(useEditorStore.getState().getRenderPreviewState(pageKey).status).toBe("stale");

    useEditorStore.getState().markRenderPreviewFresh(pageKey, "D:/tmp/project/translated/001.png");
    useEditorStore.getState().setWorkingEstiloPatch(pageKey, "layer-a", { tamanho: 34 }, ["tamanho"]);
    expect(useEditorStore.getState().getRenderPreviewState(pageKey).status).toBe("stale");

    useEditorStore.getState().markRenderPreviewFresh(pageKey, "D:/tmp/project/translated/001.png");
    useEditorStore.getState().setWorkingVisibility(pageKey, "layer-a", false);
    expect(useEditorStore.getState().getRenderPreviewState(pageKey).status).toBe("stale");
  });

  it("reports stale page numbers for export blocking", () => {
    const page = makePage(1);
    const secondPage = makePage(2, [makeLayer({ id: "layer-b" })]);
    useAppStore.setState({ project: makeProject([page, secondPage]) });
    useEditorStore.setState({
      currentPageIndex: 0,
      currentPage: page,
      pendingEdits: {},
      pendingStructuralEdits: { created: [], deleted: {}, order: undefined },
      renderPreviewCacheByPageKey: {},
    });

    expect(useEditorStore.getState().getStaleRenderPreviewPages()).toEqual([]);

    const firstPageKey = useEditorStore.getState().currentPageKey();
    useEditorStore.getState().setWorkingTraduzido(firstPageKey, "layer-a", "Oi");

    expect(useEditorStore.getState().getStaleRenderPreviewPages()).toEqual([1]);
  });

  it("renders a faithful preview without committing pending edits", async () => {
    invokeMock.mockResolvedValue("D:/tmp/project/render-cache/preview/001-preview.png");
    const pageKey = useEditorStore.getState().currentPageKey();

    useEditorStore.getState().setWorkingTraduzido(pageKey, "layer-a", "Oi");
    await useEditorStore.getState().renderPreviewPage(pageKey);

    expect(invokeMock).toHaveBeenCalledWith(
      "render_preview_page",
      expect.objectContaining({
        config: expect.objectContaining({
          project_path: "D:/tmp/project",
          page_index: 0,
          fingerprint: expect.any(String),
          page: expect.objectContaining({
            text_layers: [
              expect.objectContaining({
                id: "layer-a",
                traduzido: "Oi",
                translated: "Oi",
              }),
            ],
          }),
        }),
      }),
    );
    expect(useEditorStore.getState().pendingEdits["layer-a"]).toEqual({
      traduzido: "Oi",
      translated: "Oi",
    });
    expect(useEditorStore.getState().getRenderPreviewState(pageKey)).toMatchObject({
      status: "fresh",
      previewPath: "D:/tmp/project/render-cache/preview/001-preview.png",
    });
    expect(useEditorStore.getState().getStaleRenderPreviewPages()).toEqual([1]);
  });
});

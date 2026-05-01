import { beforeEach, describe, expect, it } from "vitest";
import type { PageData, TextEntry, TextLayerStyle } from "../appStore";
import { useAppStore } from "../appStore";
import { useEditorStore } from "../editorStore";

function makeLayer(overrides: Partial<TextEntry> = {}): TextEntry {
  const estilo: TextLayerStyle = {
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

function makePage(layers = [makeLayer()]): PageData {
  return {
    numero: 1,
    arquivo_original: "originals/001.png",
    arquivo_traduzido: "translated/001.png",
    image_layers: {},
    inpaint_blocks: [],
    text_layers: layers,
    textos: layers,
  };
}

beforeEach(() => {
  const page = makePage();
  useAppStore.setState({
    project: {
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
      paginas: [page],
      status: "done",
      source_path: "D:/tmp/project",
      totalPages: 1,
      mode: "manual",
    },
  });
  useEditorStore.setState({
    currentPageIndex: 0,
    currentPage: page,
    pendingEdits: {},
    pendingStructuralEdits: { created: [], deleted: {}, order: undefined },
    historyByPageKey: {},
  });
});

describe("editorStore history working state", () => {
  it("setWorkingEstiloPatch writes only touched keys and clears the pending edit when value returns to base", () => {
    const pageKey = useEditorStore.getState().currentPageKey();
    expect(pageKey).toBe("project-a:0:originals/001.png");

    useEditorStore.getState().setWorkingEstiloPatch(pageKey, "layer-a", {
      tamanho: 34,
      cor: "#ff0000",
    }, ["tamanho"]);

    expect(useEditorStore.getState().pendingEdits).toEqual({
      "layer-a": {
        estilo: {
          tamanho: 34,
        },
      },
    });

    useEditorStore.getState().setWorkingEstiloPatch(pageKey, "layer-a", { tamanho: 28 }, ["tamanho"]);

    expect(useEditorStore.getState().pendingEdits).toEqual({});
  });

  it("creates and removes a working layer through undo and redo", () => {
    const pageKey = useEditorStore.getState().currentPageKey();
    const layer = makeLayer({ id: "layer-b", order: 1, traduzido: "Novo", translated: "Novo" });

    const createResult = useEditorStore.getState().executeEditorCommand({
      commandId: "create-1",
      pageKey,
      createdAt: 1,
      type: "create-layer",
      layerId: "layer-b",
      layer,
      insertIndex: 1,
    });

    expect(createResult).toEqual({ ok: true });
    expect(useEditorStore.getState().currentPage?.text_layers.map((item) => item.id)).toEqual(["layer-a", "layer-b"]);
    expect(useEditorStore.getState().pendingStructuralEdits.created.map((item) => item.id)).toEqual(["layer-b"]);

    expect(useEditorStore.getState().undoEditor()).toEqual({ ok: true });
    expect(useEditorStore.getState().currentPage?.text_layers.map((item) => item.id)).toEqual(["layer-a"]);
    expect(useEditorStore.getState().pendingStructuralEdits.created).toEqual([]);

    expect(useEditorStore.getState().redoEditor()).toEqual({ ok: true });
    expect(useEditorStore.getState().currentPage?.text_layers.map((item) => item.id)).toEqual(["layer-a", "layer-b"]);
  });

  it("deletes and restores an existing working layer through undo", () => {
    const pageKey = useEditorStore.getState().currentPageKey();
    const layer = useEditorStore.getState().currentPage?.text_layers[0];
    expect(layer?.id).toBe("layer-a");

    const deleteResult = useEditorStore.getState().executeEditorCommand({
      commandId: "delete-1",
      pageKey,
      createdAt: 1,
      type: "delete-layer",
      layerId: "layer-a",
      layer: layer!,
      index: 0,
    });

    expect(deleteResult).toEqual({ ok: true });
    expect(useEditorStore.getState().currentPage?.text_layers).toEqual([]);
    expect(Object.keys(useEditorStore.getState().pendingStructuralEdits.deleted)).toEqual(["layer-a"]);
    expect(useEditorStore.getState().selectedLayerId).toBeNull();

    expect(useEditorStore.getState().undoEditor()).toEqual({ ok: true });
    expect(useEditorStore.getState().currentPage?.text_layers.map((item) => item.id)).toEqual(["layer-a"]);
    expect(useEditorStore.getState().pendingStructuralEdits.deleted).toEqual({});
  });

  it("records reorder and persistent visibility/lock changes in working state", () => {
    const layerA = makeLayer({ id: "layer-a", order: 0 });
    const layerB = makeLayer({ id: "layer-b", order: 1, traduzido: "B", translated: "B" });
    const page = makePage([layerA, layerB]);
    useAppStore.setState((state) => ({
      project: state.project ? { ...state.project, paginas: [page] } : null,
    }));
    useEditorStore.setState({ currentPage: page, pendingEdits: {}, pendingStructuralEdits: { created: [], deleted: {}, order: undefined } });

    const pageKey = useEditorStore.getState().currentPageKey();

    expect(
      useEditorStore.getState().executeEditorCommand({
        commandId: "reorder-1",
        pageKey,
        createdAt: 1,
        type: "reorder-layers",
        before: ["layer-a", "layer-b"],
        after: ["layer-b", "layer-a"],
      }),
    ).toEqual({ ok: true });
    expect(useEditorStore.getState().getOrderedLayerIds(pageKey)).toEqual(["layer-b", "layer-a"]);
    expect(useEditorStore.getState().pendingStructuralEdits.order).toEqual(["layer-b", "layer-a"]);

    expect(
      useEditorStore.getState().executeEditorCommand({
        commandId: "visible-1",
        pageKey,
        createdAt: 2,
        type: "toggle-visibility",
        layerId: "layer-b",
        before: true,
        after: false,
      }),
    ).toEqual({ ok: true });
    expect(useEditorStore.getState().getLayer(pageKey, "layer-b")?.visible).toBe(false);
    expect(useEditorStore.getState().pendingEdits["layer-b"]).toEqual({ visible: false });

    expect(
      useEditorStore.getState().executeEditorCommand({
        commandId: "lock-1",
        pageKey,
        createdAt: 3,
        type: "toggle-lock",
        layerId: "layer-b",
        before: false,
        after: true,
      }),
    ).toEqual({ ok: true });
    expect(useEditorStore.getState().getLayer(pageKey, "layer-b")?.locked).toBe(true);
    expect(useEditorStore.getState().pendingEdits["layer-b"]).toEqual({ visible: false, locked: true });

    expect(useEditorStore.getState().undoEditor()).toEqual({ ok: true });
    expect(useEditorStore.getState().getLayer(pageKey, "layer-b")?.locked).toBe(false);
  });

  it("uses history-backed working state for public create, delete and visibility actions", async () => {
    useEditorStore.getState().selectLayer("layer-a");

    await useEditorStore.getState().deleteSelectedLayer();
    expect(useEditorStore.getState().currentPage?.text_layers.map((item) => item.id)).toEqual([]);
    expect(Object.keys(useEditorStore.getState().pendingStructuralEdits.deleted)).toEqual(["layer-a"]);

    expect(useEditorStore.getState().undoEditor()).toEqual({ ok: true });
    expect(useEditorStore.getState().currentPage?.text_layers.map((item) => item.id)).toEqual(["layer-a"]);

    await useEditorStore.getState().toggleTextLayerVisibility("layer-a");
    expect(useEditorStore.getState().getLayer(useEditorStore.getState().currentPageKey(), "layer-a")?.visible).toBe(false);
    expect(useEditorStore.getState().pendingEdits["layer-a"]).toEqual({ visible: false });

    await useEditorStore.getState().createTextLayer([10, 20, 80, 120]);
    const ids = useEditorStore.getState().currentPage?.text_layers.map((item) => item.id) ?? [];
    expect(ids).toHaveLength(2);
    expect(useEditorStore.getState().pendingStructuralEdits.created.map((item) => item.id)).toHaveLength(1);
    expect(useEditorStore.getState().selectedLayerId).toBe(ids[1]);
  });
});

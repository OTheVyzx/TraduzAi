import { beforeEach, describe, expect, it } from "vitest";
import type { PageData, TextEntry, TextLayerStyle } from "../stores/appStore";
import { useAppStore } from "../stores/appStore";
import { useEditorStore } from "../stores/editorStore";
import { selectFieldEditValue } from "../useFieldEditSession";
import { selectTextEditValue } from "../useTextEditSession";

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
    sombra: true,
    sombra_cor: "#000000",
    sombra_offset: [2, 4],
    bold: false,
    italico: false,
    rotacao: 0,
    alinhamento: "center",
  };
}

function makeLayer(): TextEntry {
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
  };
}

function makePage(layer = makeLayer()): PageData {
  return {
    numero: 1,
    arquivo_original: "originals/001.png",
    arquivo_traduzido: "translated/001.png",
    image_layers: {},
    inpaint_blocks: [],
    text_layers: [layer],
    textos: [layer],
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

describe("editor session selectors", () => {
  it("return stable primitive and array references when visibility is the only pending edit", () => {
    const pageKey = useEditorStore.getState().currentPageKey();
    useEditorStore.getState().setWorkingVisibility(pageKey, "layer-a", false);
    const state = useEditorStore.getState();

    expect(selectTextEditValue(state, pageKey, "layer-a")).toBe("Ola");
    expect(selectTextEditValue(state, pageKey, "layer-a")).toBe("Ola");

    const firstOffset = selectFieldEditValue(state, pageKey, "layer-a", "sombra_offset");
    const secondOffset = selectFieldEditValue(state, pageKey, "layer-a", "sombra_offset");
    expect(firstOffset).toBe(secondOffset);
    expect(firstOffset).toEqual([2, 4]);
  });

  it("prefers pending text and style values without merging a layer object", () => {
    const pageKey = useEditorStore.getState().currentPageKey();
    useEditorStore.getState().setWorkingTraduzido(pageKey, "layer-a", "Oi");
    useEditorStore.getState().setWorkingEstiloPatch(pageKey, "layer-a", { tamanho: 34 }, ["tamanho"]);
    const state = useEditorStore.getState();

    expect(selectTextEditValue(state, pageKey, "layer-a")).toBe("Oi");
    expect(selectFieldEditValue(state, pageKey, "layer-a", "tamanho")).toBe(34);
    expect(selectFieldEditValue(state, pageKey, "layer-a", "cor")).toBe("#ffffff");
  });
});

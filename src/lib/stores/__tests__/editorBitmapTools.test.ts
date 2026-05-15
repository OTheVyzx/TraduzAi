import { beforeEach, describe, expect, it, vi } from "vitest";
import type { PageData, Project, TextEntry, TextLayerStyle } from "../appStore";
import { useAppStore } from "../appStore";
import { useEditorStore } from "../editorStore";

const {
  updateBrushRegion,
  updateMaskRegion,
  updateRecoveryRegion,
  updateReinpaintRegion,
  healInpaintRegion,
  patchEditorTextLayer,
  runPageActionWithOptionalMask,
  loadEditorPage,
} = vi.hoisted(() => ({
  updateBrushRegion: vi.fn(async () => "images/brush.png"),
  updateMaskRegion: vi.fn(async () => "images/mask.png"),
  updateRecoveryRegion: vi.fn(async () => "images/inpaint.png"),
  updateReinpaintRegion: vi.fn(async () => "images/reinpaint.png"),
  healInpaintRegion: vi.fn(async () => ({
    page_index: 0,
    inpaint_path: "images/001.png",
    before_inpaint_path: "editor_cache/healing_inpaint/page-0001/before.png",
    bbox: [10, 10, 40, 40] as [number, number, number, number],
  })),
  patchEditorTextLayer: vi.fn(async () => ({})),
  runPageActionWithOptionalMask: vi.fn(async () => ({
    action: "ocr",
    mode: "regional",
    bbox: [12, 14, 50, 40],
    changed_assets: ["project_json", "rendered"],
    changed_layers: [],
    message: "ok",
  })),
  loadEditorPage: vi.fn(async () => ({
    page_index: 0,
    total_pages: 1,
    page: {
      numero: 1,
      arquivo_original: "originals/001.png",
      arquivo_traduzido: "translated/001.png",
      image_layers: {
        base: { key: "base", path: "originals/001.png", visible: true, locked: true },
        inpaint: { key: "inpaint", path: "images/001.png", visible: true, locked: false },
        brush: { key: "brush", path: "images/brush.png", visible: true, locked: false },
        mask: { key: "mask", path: "images/mask.png", visible: true, locked: false },
        recovery: { key: "recovery", path: null, visible: false, locked: false },
      },
      inpaint_blocks: [],
      text_layers: [],
      textos: [],
    },
  })),
}));

vi.mock("../../editorBackend", () => ({
  getEditorBackend: vi.fn(async () => ({
    updateBrushRegion,
    updateMaskRegion,
    updateRecoveryRegion,
    updateReinpaintRegion,
    healInpaintRegion,
    patchEditorTextLayer,
    runPageActionWithOptionalMask,
    loadEditorPage,
  })),
}));

function makeLayer(): TextEntry {
  const estilo: TextLayerStyle = {
    fonte: "ComicNeue-Bold.ttf",
    tamanho: 28,
    cor: "#000000",
    cor_gradiente: [],
    contorno: "",
    contorno_px: 0,
    glow: false,
    glow_cor: "",
    glow_px: 0,
    sombra: false,
    sombra_cor: "",
    sombra_offset: [0, 0],
    bold: true,
    italico: false,
    rotacao: 0,
    alinhamento: "center",
  };

  return {
    id: "text-a",
    kind: "text",
    source_bbox: [0, 0, 80, 80],
    layout_bbox: [0, 0, 80, 80],
    render_bbox: null,
    bbox: [0, 0, 80, 80],
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

function makePage(): PageData {
  const layer = makeLayer();
  return {
    numero: 1,
    arquivo_original: "originals/001.png",
    arquivo_traduzido: "translated/001.png",
    image_layers: {
      base: { key: "base", path: "originals/001.png", visible: true, locked: true },
      inpaint: { key: "inpaint", path: "images/001.png", visible: true, locked: false },
      brush: { key: "brush", path: "images/brush.png", visible: true, locked: false },
      mask: { key: "mask", path: "images/mask.png", visible: true, locked: false },
      recovery: { key: "recovery", path: null, visible: false, locked: false },
    },
    inpaint_blocks: [],
    text_layers: [layer],
    textos: [layer],
  };
}

function makeProject(page: PageData): Project {
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
    paginas: [page],
    status: "done",
    source_path: "D:/tmp/project.json",
    output_path: "D:/tmp/project.json",
    totalPages: 1,
    mode: "manual",
  };
}

async function applyStroke() {
  await useEditorStore.getState().applyBitmapStroke({
    width: 100,
    height: 100,
    strokes: [[[10, 10], [20, 20]]],
  });
}

beforeEach(() => {
  vi.clearAllMocks();
  const page = makePage();
  useAppStore.setState({ project: makeProject(page) });
  useEditorStore.getState().resetEditor();
  useEditorStore.setState({
    currentPageIndex: 0,
    currentPage: page,
    selectedLayerId: "text-a",
    toolMode: "select",
    lastPaintedLayer: "brush",
    eraserTarget: null,
    pendingEdits: {},
    pendingStructuralEdits: { created: [], deleted: {}, order: undefined },
  });
});

describe("editor bitmap tools", () => {
  it("keeps patch-only saved text visible after pending edits are cleared", async () => {
    const pageKey = useEditorStore.getState().currentPageKey();

    useEditorStore.getState().setWorkingTraduzido(pageKey, "text-a", "Texto novo");
    expect(useEditorStore.getState().pendingEdits["text-a"]?.traduzido).toBe("Texto novo");

    await useEditorStore.getState().commitEditsPatchOnly();

    expect(patchEditorTextLayer).toHaveBeenCalledWith(
      expect.objectContaining({
        project_path: "D:/tmp/project.json",
        page_index: 0,
        layer_id: "text-a",
        patch: expect.objectContaining({ translated: "Texto novo" }),
      }),
    );
    expect(useEditorStore.getState().pendingEdits).toEqual({});
    expect(useEditorStore.getState().currentPage?.text_layers[0].traduzido).toBe("Texto novo");
    expect(useAppStore.getState().project?.paginas[0].text_layers[0].traduzido).toBe("Texto novo");
  });

  it("updates internal bitmap assets without exposing bitmap selection", async () => {
    useEditorStore.setState({ toolMode: "repairBrush" });
    await applyStroke();
    expect(updateRecoveryRegion).toHaveBeenCalledTimes(1);
    expect(useEditorStore.getState().currentPage?.image_layers?.inpaint?.path).toBe("images/inpaint.png");
    expect("selectedImageLayerKey" in useEditorStore.getState()).toBe(false);

    useEditorStore.setState({ toolMode: "brush" });
    await applyStroke();
    expect(updateBrushRegion).toHaveBeenCalledTimes(1);
    expect("selectedImageLayerKey" in useEditorStore.getState()).toBe(false);

    useEditorStore.setState({ toolMode: "mask" });
    await applyStroke();
    expect(updateMaskRegion).toHaveBeenCalledTimes(1);
    expect("selectedImageLayerKey" in useEditorStore.getState()).toBe(false);

    useEditorStore.setState({ toolMode: "reinpaintBrush" });
    await applyStroke();
    expect(updateReinpaintRegion).toHaveBeenCalledTimes(1);
    expect(updateRecoveryRegion).toHaveBeenCalledTimes(1);
    expect(useEditorStore.getState().currentPage?.image_layers?.inpaint?.path).toBe("images/reinpaint.png");
  });

  it("eraser uses explicit target or last painted layer", async () => {
    useEditorStore.setState({
      toolMode: "eraser",
      lastPaintedLayer: "brush",
      eraserTarget: null,
    });

    await applyStroke();

    expect(updateBrushRegion).toHaveBeenCalledTimes(1);
    expect(updateMaskRegion).not.toHaveBeenCalled();
    expect("selectedImageLayerKey" in useEditorStore.getState()).toBe(false);
  });

  it("eraser never writes recovery over the inpaint layer", async () => {
    useEditorStore.setState({
      toolMode: "eraser",
      lastPaintedLayer: "recovery",
      eraserTarget: null,
    });

    await applyStroke();

    expect(updateRecoveryRegion).not.toHaveBeenCalled();
    expect(updateBrushRegion).toHaveBeenCalledTimes(1);
    expect(useEditorStore.getState().currentPage?.image_layers?.inpaint?.path).toBe("images/001.png");

    vi.clearAllMocks();
    useEditorStore.setState({
      toolMode: "eraser",
      lastPaintedLayer: "mask",
      eraserTarget: "recovery",
    });

    await applyStroke();

    expect(updateRecoveryRegion).not.toHaveBeenCalled();
    expect(updateBrushRegion).toHaveBeenCalledTimes(1);
    expect(updateMaskRegion).not.toHaveBeenCalled();
  });

  it("persists the layer captured by the stroke instead of the current tool", async () => {
    useEditorStore.setState({ toolMode: "brush" });

    await useEditorStore.getState().applyBitmapStroke({
      width: 100,
      height: 100,
      strokes: [[[10, 10], [20, 20]]],
      layerKey: "recovery",
      erase: false,
      brushSize: 18,
      color: "#ff0000",
      opacity: 0.5,
      hardness: 1,
      clipMaskPng: "data:image/png;base64,clip",
    });

    expect(updateRecoveryRegion).toHaveBeenCalledWith(
      expect.objectContaining({ clip_mask_png: "data:image/png;base64,clip" }),
    );
    expect(updateBrushRegion).not.toHaveBeenCalled();
    expect(useEditorStore.getState().currentPage?.image_layers?.inpaint?.path).toBe("images/inpaint.png");
  });

  it("sends composed png data for recovery strokes", async () => {
    const pngData = "data:image/png;base64,recovered";

    await useEditorStore.getState().applyBitmapStroke({
      width: 100,
      height: 100,
      strokes: [[[10, 10], [20, 20]]],
      layerKey: "recovery",
      erase: false,
      brushSize: 18,
      pngData,
    });

    expect(updateRecoveryRegion).toHaveBeenCalledWith(expect.objectContaining({ png_data: pngData }));
    expect(useEditorStore.getState().currentPage?.image_layers?.inpaint?.path).toBe("images/inpaint.png");
  });

  it("reapplies cached inpaint through reinpaint region without using recovery", async () => {
    await useEditorStore.getState().applyBitmapStroke({
      width: 100,
      height: 100,
      strokes: [[[10, 10], [20, 20]]],
      layerKey: "reinpaint",
      erase: false,
      brushSize: 18,
      pngData: "data:image/png;base64,reinpainted",
    });

    expect(updateReinpaintRegion).toHaveBeenCalledWith(
      expect.objectContaining({ png_data: "data:image/png;base64,reinpainted" }),
    );
    expect(updateRecoveryRegion).not.toHaveBeenCalled();
    expect(useEditorStore.getState().currentPage?.image_layers?.inpaint?.path).toBe("images/reinpaint.png");
  });

  it("runs lasso page actions with the selected region bbox", async () => {
    useEditorStore.setState({
      activeLassoSelection: {
        pageKey: useEditorStore.getState().currentPageKey(),
        pageIndex: 0,
        points: [
          [12, 14],
          [62, 14],
          [62, 54],
          [12, 54],
        ],
        bbox: [12, 14, 50, 40],
        width: 100,
        height: 100,
      },
    });

    await useEditorStore.getState().runMaskedActionFromLasso("ocr");

    expect(runPageActionWithOptionalMask).toHaveBeenCalledWith({
      project_path: "D:/tmp/project.json",
      page_index: 0,
      action: "ocr",
      bbox: [12, 14, 50, 40],
    });
    expect(useEditorStore.getState().activeLassoSelection).toBeNull();
  });

  it("applies healing brush result with undoable inpaint paths", async () => {
    await useEditorStore.getState().healPaintedRegion({
      bbox: [10, 10, 40, 40],
      maskPath: "editor_cache/healing_masks/page-0001/mask.png",
    });

    expect(healInpaintRegion).toHaveBeenCalledWith({
      project_path: "D:/tmp/project.json",
      page_index: 0,
      bbox: [10, 10, 40, 40],
      mask_path: "editor_cache/healing_masks/page-0001/mask.png",
    });
    expect(useEditorStore.getState().currentPage?.image_layers?.inpaint?.path).toBe(
      "images/001.png",
    );
    expect(useEditorStore.getState().getRenderPreviewState(useEditorStore.getState().currentPageKey()).status).toBe("stale");
    expect(useEditorStore.getState().undoEditor()).toEqual({ ok: true });
    expect(useEditorStore.getState().currentPage?.image_layers?.inpaint?.path).toBe(
      "editor_cache/healing_inpaint/page-0001/before.png",
    );
  });

  it("does not let stale bitmap persistence overwrite a newer optimistic layer", async () => {
    const page = useEditorStore.getState().currentPage!;
    useEditorStore.setState({
      toolMode: "brush",
      currentPage: {
        ...page,
        image_layers: {
          ...page.image_layers,
          brush: {
            ...(page.image_layers?.brush ?? { key: "brush" as const, visible: true, locked: false }),
            key: "brush",
            path: "data:image/png;base64,newer",
          },
        },
      },
    });

    await useEditorStore.getState().applyBitmapStroke({
      width: 100,
      height: 100,
      strokes: [[[10, 10], [20, 20]]],
      layerKey: "brush",
      erase: false,
      optimisticPath: "data:image/png;base64,older",
    });

    expect(updateBrushRegion).toHaveBeenCalledTimes(1);
    expect(useEditorStore.getState().currentPage?.image_layers?.brush?.path).toBe("data:image/png;base64,newer");
  });
});

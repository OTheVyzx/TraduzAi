import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
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
  processBlock,
  retypesetPage,
  reinpaintPage,
  detectPage,
  ocrPage,
  translatePage,
  snapshotImageLayer,
  runProcessRegion,
  runPageActionWithOptionalMask,
  writeMaskFromPng,
  loadEditorPage,
  preloadEditorVisionPage,
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
  processBlock: vi.fn(async () => "translated/001.png"),
  retypesetPage: vi.fn(async () => "translated/001.png"),
  reinpaintPage: vi.fn(async () => "images/reinpaint-page.png"),
  detectPage: vi.fn(async () => "detected"),
  ocrPage: vi.fn(async () => "ocr"),
  translatePage: vi.fn(async () => "translate"),
  snapshotImageLayer: vi.fn(async ({ source_path }: { source_path?: string | null }) =>
    source_path ? `editor_cache/history/${source_path.replace(/[/:\\]/g, "_")}-${crypto.randomUUID()}.png` : null,
  ),
  runProcessRegion: vi.fn(async () => ({
    page_index: 0,
    overlay: {
      id: "process-1",
      page_index: 0,
      bbox: [12, 14, 62, 54] as [number, number, number, number],
      crop_path: "editor_cache/process_regions/page-0001/process-1.png",
      text_layer_ids: ["text-a"],
      visible: true,
      locked: false,
      order: 0,
    },
    changed_assets: ["project_json", "inpaint", "rendered"],
    changed_layers: ["text-a"],
    message: "ok",
  })),
  writeMaskFromPng: vi.fn(async () => "editor_cache/masks/lasso-area.png"),
  preloadEditorVisionPage: vi.fn(async () => "queued"),
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
    processBlock,
    retypesetPage,
    reinpaintPage,
    detectPage,
    ocrPage,
    translatePage,
    snapshotImageLayer,
    runProcessRegion,
    runPageActionWithOptionalMask,
    writeMaskFromPng,
    loadEditorPage,
    preloadEditorVisionPage,
  })),
}));

vi.mock("../../lassoSelection", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../lassoSelection")>();
  return {
    ...actual,
    rasterizeLassoToPng: vi.fn(() => "data:image/png;base64,lasso"),
  };
});

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

function makePage(pageNumber = 1): PageData {
  const layer = makeLayer();
  const padded = String(pageNumber).padStart(3, "0");
  return {
    numero: pageNumber,
    arquivo_original: `originals/${padded}.png`,
    arquivo_traduzido: `translated/${padded}.png`,
    image_layers: {
      base: { key: "base", path: `originals/${padded}.png`, visible: true, locked: true },
      inpaint: { key: "inpaint", path: `images/${padded}.png`, visible: true, locked: false },
      brush: { key: "brush", path: `images/brush-${padded}.png`, visible: true, locked: false },
      mask: { key: "mask", path: `images/mask-${padded}.png`, visible: true, locked: false },
      recovery: { key: "recovery", path: null, visible: false, locked: false },
    },
    inpaint_blocks: [],
    text_layers: [layer],
    textos: [layer],
  };
}

function makeBackendChangedPage(pageNumber = 1): PageData {
  const page = makePage(pageNumber);
  return {
    ...page,
    arquivo_traduzido: `translated/${String(pageNumber).padStart(3, "0")}-changed.png`,
    image_layers: {
      ...page.image_layers,
      inpaint: { key: "inpaint", path: `images/${String(pageNumber).padStart(3, "0")}-changed.png`, visible: true, locked: false },
      rendered: { key: "rendered", path: `translated/${String(pageNumber).padStart(3, "0")}-changed.png`, visible: true, locked: false },
    },
    text_layers: [
      {
        ...page.text_layers[0],
        traduzido: "Alterado",
        translated: "Alterado",
      },
    ],
    textos: [
      {
        ...page.text_layers[0],
        traduzido: "Alterado",
        translated: "Alterado",
      },
    ],
  };
}

function makeProject(page: PageData | PageData[]): Project {
  const paginas = Array.isArray(page) ? page : [page];
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
    paginas,
    status: "done",
    source_path: "D:/tmp/project.json",
    output_path: "D:/tmp/project.json",
    totalPages: paginas.length,
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

afterEach(() => {
  vi.useRealTimers();
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

  it("makes direct bitmap stroke persistence undoable when no optimistic command exists", async () => {
    const pageKey = useEditorStore.getState().currentPageKey();
    useEditorStore.setState({ toolMode: "brush" });

    await useEditorStore.getState().applyBitmapStroke({
      width: 100,
      height: 100,
      strokes: [[[10, 10], [20, 20]]],
      layerKey: "brush",
      erase: false,
    });

    expect(useEditorStore.getState().currentPage?.image_layers?.brush?.path).toBe("images/brush.png");
    expect(useEditorStore.getState().historyByPageKey[pageKey].commands).toEqual([
      expect.objectContaining({
        type: "bitmap-asset-replace",
        layerKey: "brush",
        beforePath: expect.stringContaining("editor_cache/history/images_brush-001.png"),
        afterPath: expect.stringContaining("editor_cache/history/images_brush.png"),
      }),
    ]);

    expect(useEditorStore.getState().undoEditor()).toEqual({ ok: true });
    expect(useEditorStore.getState().currentPage?.image_layers?.brush?.path).toContain(
      "editor_cache/history/images_brush-001.png",
    );

    expect(useEditorStore.getState().redoEditor()).toEqual({ ok: true });
    expect(useEditorStore.getState().currentPage?.image_layers?.brush?.path).toContain(
      "editor_cache/history/images_brush.png",
    );
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

  it("runs lasso page actions with the selected region mask and manga engines", async () => {
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

    expect(writeMaskFromPng).toHaveBeenCalledWith({
      project_path: "D:/tmp/project.json",
      page_index: 0,
      png_data: "data:image/png;base64,lasso",
      layer_key: "mask",
      op: "replace",
    });
    expect(runPageActionWithOptionalMask).toHaveBeenCalledWith({
      project_path: "D:/tmp/project.json",
      page_index: 0,
      action: "ocr",
      bbox: [12, 14, 50, 40],
      mask_path: "editor_cache/masks/lasso-area.png",
      engine_preset_id: "manga",
      idioma_origem: "en",
      idioma_destino: "pt-BR",
    });
    expect(useEditorStore.getState().activeLassoSelection).toBeNull();
  });

  it("makes clearing the mask undoable and redoable", async () => {
    const pageKey = useEditorStore.getState().currentPageKey();

    await useEditorStore.getState().clearMask();

    expect(writeMaskFromPng).toHaveBeenCalledWith({
      project_path: "D:/tmp/project.json",
      page_index: 0,
      png_data: expect.stringContaining("data:image/png"),
      layer_key: "mask",
      op: "replace",
    });
    expect(useEditorStore.getState().currentPage?.image_layers?.mask?.path).toBe("editor_cache/masks/lasso-area.png");
    expect(useEditorStore.getState().historyByPageKey[pageKey].commands).toEqual([
      expect.objectContaining({
        type: "bitmap-asset-replace",
        layerKey: "mask",
        beforePath: "images/mask-001.png",
        afterPath: "editor_cache/masks/lasso-area.png",
      }),
    ]);

    expect(useEditorStore.getState().undoEditor()).toEqual({ ok: true });
    expect(useEditorStore.getState().currentPage?.image_layers?.mask?.path).toBe("images/mask-001.png");

    expect(useEditorStore.getState().redoEditor()).toEqual({ ok: true });
    expect(useEditorStore.getState().currentPage?.image_layers?.mask?.path).toBe("editor_cache/masks/lasso-area.png");
  });

  it("makes applying a lasso selection to the mask undoable and redoable", async () => {
    const pageKey = useEditorStore.getState().currentPageKey();
    useEditorStore.setState({
      activeLassoSelection: {
        pageKey,
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

    await useEditorStore.getState().applyLassoSelectionToMask();

    expect(writeMaskFromPng).toHaveBeenCalledWith({
      project_path: "D:/tmp/project.json",
      page_index: 0,
      png_data: "data:image/png;base64,lasso",
      layer_key: "mask",
      op: "replace",
    });
    expect(useEditorStore.getState().activeLassoSelection).toBeNull();
    expect(useEditorStore.getState().currentPage?.image_layers?.mask?.path).toBe("editor_cache/masks/lasso-area.png");
    expect(useEditorStore.getState().historyByPageKey[pageKey].commands).toEqual([
      expect.objectContaining({
        type: "bitmap-asset-replace",
        layerKey: "mask",
        beforePath: "images/mask-001.png",
        afterPath: "editor_cache/masks/lasso-area.png",
      }),
    ]);

    expect(useEditorStore.getState().undoEditor()).toEqual({ ok: true });
    expect(useEditorStore.getState().currentPage?.image_layers?.mask?.path).toBe("images/mask-001.png");

    expect(useEditorStore.getState().redoEditor()).toEqual({ ok: true });
    expect(useEditorStore.getState().currentPage?.image_layers?.mask?.path).toBe("editor_cache/masks/lasso-area.png");
  });

  it("runs the automatic process tool with a lasso mask and stores the returned crop overlay", async () => {
    const selection = {
      pageKey: useEditorStore.getState().currentPageKey(),
      pageIndex: 0,
      points: [
        [12, 14],
        [62, 14],
        [62, 54],
        [12, 54],
      ] as Array<[number, number]>,
      bbox: [12, 14, 62, 54] as [number, number, number, number],
      width: 100,
      height: 100,
    };

    await useEditorStore.getState().runProcessRegionFromSelection(selection);

    expect(writeMaskFromPng).toHaveBeenCalledWith({
      project_path: "D:/tmp/project.json",
      page_index: 0,
      png_data: "data:image/png;base64,lasso",
      layer_key: "mask",
      op: "replace",
    });
    expect(runProcessRegion).toHaveBeenCalledWith({
      project_path: "D:/tmp/project.json",
      page_index: 0,
      bbox: [12, 14, 62, 54],
      mask_path: "editor_cache/masks/lasso-area.png",
      engine_preset_id: "manga",
      idioma_origem: "en",
      idioma_destino: "pt-BR",
    });
    expect(useEditorStore.getState().currentPage?.process_overlays?.[0]).toMatchObject({
      id: "process-1",
      crop_path: "editor_cache/process_regions/page-0001/process-1.png",
      text_layer_ids: ["text-a"],
      visible: true,
    });
    expect(useEditorStore.getState().selectedLayerId).toBe("text-a");
  });

  it("makes the automatic process result undoable and redoable", async () => {
    const selection = {
      pageKey: useEditorStore.getState().currentPageKey(),
      pageIndex: 0,
      points: [
        [12, 14],
        [62, 14],
        [62, 54],
        [12, 54],
      ] as Array<[number, number]>,
      bbox: [12, 14, 62, 54] as [number, number, number, number],
      width: 100,
      height: 100,
    };

    await useEditorStore.getState().runProcessRegionFromSelection(selection);
    expect(useEditorStore.getState().currentPage?.process_overlays?.map((overlay) => overlay.id)).toEqual(["process-1"]);

    expect(useEditorStore.getState().undoEditor()).toEqual({ ok: true });
    expect(useEditorStore.getState().currentPage?.process_overlays ?? []).toEqual([]);

    expect(useEditorStore.getState().redoEditor()).toEqual({ ok: true });
    expect(useEditorStore.getState().currentPage?.process_overlays?.map((overlay) => overlay.id)).toEqual(["process-1"]);
  });

  it("passes the selected source language to page OCR actions", async () => {
    const page = makePage();
    useAppStore.setState({ project: { ...makeProject(page), idioma_origem: "ko" } });
    useEditorStore.setState({ currentPage: page, currentPageIndex: 0 });

    await useEditorStore.getState().runMaskedAction("ocr");

    expect(runPageActionWithOptionalMask).toHaveBeenCalledWith(
      expect.objectContaining({
        project_path: "D:/tmp/project.json",
        page_index: 0,
        action: "ocr",
        idioma_origem: "ko",
      }),
    );
  });

  it("makes full-page masked actions undoable and redoable", async () => {
    const beforePage = makePage();
    const afterPage = makeBackendChangedPage();
    useAppStore.setState({ project: makeProject(beforePage) });
    useEditorStore.setState({ currentPage: beforePage, currentPageIndex: 0 });
    loadEditorPage.mockResolvedValueOnce({
      page_index: 0,
      total_pages: 1,
      page: afterPage as never,
    });

    await useEditorStore.getState().runMaskedAction("inpaint");

    expect(useEditorStore.getState().currentPage?.text_layers[0].traduzido).toBe("Alterado");
    expect(useEditorStore.getState().historyByPageKey[useEditorStore.getState().currentPageKey()].commands).toEqual([
      expect.objectContaining({ type: "page-snapshot", label: "Limpar fundo" }),
    ]);

    expect(useEditorStore.getState().undoEditor()).toEqual({ ok: true });
    expect(useEditorStore.getState().currentPage?.text_layers[0].traduzido).toBe("Ola");

    expect(useEditorStore.getState().redoEditor()).toEqual({ ok: true });
    expect(useEditorStore.getState().currentPage?.text_layers[0].traduzido).toBe("Alterado");
  });

  it("makes lasso page actions undoable and redoable", async () => {
    const beforePage = makePage();
    const afterPage = makeBackendChangedPage();
    useAppStore.setState({ project: makeProject(beforePage) });
    useEditorStore.setState({
      currentPage: beforePage,
      currentPageIndex: 0,
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
    loadEditorPage.mockResolvedValueOnce({
      page_index: 0,
      total_pages: 1,
      page: afterPage as never,
    });

    await useEditorStore.getState().runMaskedActionFromLasso("ocr");

    expect(useEditorStore.getState().currentPage?.text_layers[0].traduzido).toBe("Alterado");
    expect(useEditorStore.getState().undoEditor()).toEqual({ ok: true });
    expect(useEditorStore.getState().currentPage?.text_layers[0].traduzido).toBe("Ola");
    expect(useEditorStore.getState().redoEditor()).toEqual({ ok: true });
    expect(useEditorStore.getState().currentPage?.text_layers[0].traduzido).toBe("Alterado");
  });

  it("routes detector-only page action as a separate editor action", async () => {
    const page = makePage();
    useAppStore.setState({ project: makeProject(page) });
    useEditorStore.setState({ currentPage: page, currentPageIndex: 0 });

    await useEditorStore.getState().runMaskedAction("detect_boxes");

    expect(runPageActionWithOptionalMask).toHaveBeenCalledWith(
      expect.objectContaining({
        project_path: "D:/tmp/project.json",
        page_index: 0,
        action: "detect_boxes",
        idioma_origem: "en",
      }),
    );
  });

  it("passes the selected source language when reprocessing a text block OCR", async () => {
    const page = makePage();
    useAppStore.setState({ project: { ...makeProject(page), idioma_origem: "ko" } });
    useEditorStore.setState({ currentPage: page, currentPageIndex: 0, selectedLayerId: "text-a" });

    await useEditorStore.getState().reProcessBlock("ocr");

    expect(processBlock).toHaveBeenCalledWith(
      expect.objectContaining({
        project_path: "D:/tmp/project.json",
        page_index: 0,
        block_id: "text-a",
        mode: "ocr",
        idioma_origem: "ko",
      }),
    );
  });

  it("makes standalone page pipeline actions undoable and redoable", async () => {
    const beforePage = makePage();
    const afterPage = makeBackendChangedPage();
    useAppStore.setState({ project: makeProject(beforePage) });
    useEditorStore.setState({ currentPage: beforePage, currentPageIndex: 0, selectedLayerId: "text-a" });

    for (const run of [
      () => useEditorStore.getState().retypesetCurrentPage(),
      () => useEditorStore.getState().reinpaintCurrentPage(),
      () => useEditorStore.getState().detectInPage(),
      () => useEditorStore.getState().ocrInPage(),
      () => useEditorStore.getState().translateInPage(),
      () => useEditorStore.getState().reProcessBlock("ocr"),
      () => useEditorStore.getState().disconnectBlock(),
    ]) {
      useEditorStore.setState({ currentPage: beforePage, historyByPageKey: {}, selectedLayerId: "text-a" });
      useAppStore.setState({ project: makeProject(beforePage) });
      loadEditorPage.mockResolvedValueOnce({
        page_index: 0,
        total_pages: 1,
        page: afterPage as never,
      });

      await run();

      expect(useEditorStore.getState().currentPage?.text_layers[0].traduzido).toBe("Alterado");
      expect(useEditorStore.getState().undoEditor()).toEqual({ ok: true });
      expect(useEditorStore.getState().currentPage?.text_layers[0].traduzido).toBe("Ola");
      expect(useEditorStore.getState().redoEditor()).toEqual({ ok: true });
      expect(useEditorStore.getState().currentPage?.text_layers[0].traduzido).toBe("Alterado");
    }
  });

  it("requests editor vision preload after loading a page", async () => {
    vi.useFakeTimers();
    const page = makePage();
    useAppStore.setState({ project: makeProject(page) });
    useEditorStore.setState({ currentPage: null, currentPageIndex: 0 });

    await useEditorStore.getState().loadCurrentPage();
    expect(preloadEditorVisionPage).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(950);

    expect(preloadEditorVisionPage).toHaveBeenCalledWith(expect.objectContaining({
      project_path: "D:/tmp/project.json",
      page_index: 0,
      idioma_origem: "en",
      target: "detect_ocr",
    }));
  });

  it("requests OCR-layer preload when the loaded page already has text layers", async () => {
    vi.useFakeTimers();
    const page = makePage();
    loadEditorPage.mockResolvedValueOnce({
      page_index: 0,
      total_pages: 1,
      page: page as never,
    });
    useAppStore.setState({ project: makeProject(page) });
    useEditorStore.setState({ currentPage: null, currentPageIndex: 0 });

    await useEditorStore.getState().loadCurrentPage();
    await vi.advanceTimersByTimeAsync(950);

    expect(preloadEditorVisionPage).toHaveBeenCalledWith(expect.objectContaining({
      project_path: "D:/tmp/project.json",
      page_index: 0,
      idioma_origem: "en",
      target: "ocr_layers",
    }));
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

  it("applies page-scoped bitmap persistence to the captured page after navigation", async () => {
    let resolveBrush!: (path: string) => void;
    updateBrushRegion.mockImplementationOnce(
      () =>
        new Promise<string>((resolve) => {
          resolveBrush = resolve;
        }),
    );
    const pageA = makePage(1);
    const pageB = makePage(2);
    useAppStore.setState({ project: makeProject([pageA, pageB]) });
    useEditorStore.getState().resetEditor();
    useEditorStore.setState({
      currentPageIndex: 0,
      currentPage: pageA,
      toolMode: "brush",
      lastPaintedLayer: "brush",
      eraserTarget: null,
    });
    const pageAKey = useEditorStore.getState().currentPageKey();

    const strokePromise = useEditorStore.getState().applyBitmapStroke({
      pageKey: pageAKey,
      pageIndex: 0,
      width: 100,
      height: 100,
      strokes: [[[10, 10], [20, 20]]],
      layerKey: "brush",
      erase: false,
    });
    await vi.waitFor(() => expect(updateBrushRegion).toHaveBeenCalledTimes(1));

    useEditorStore.setState({ currentPageIndex: 1, currentPage: pageB });
    resolveBrush("images/page-a-brush-after-nav.png");
    await strokePromise;

    expect(updateBrushRegion).toHaveBeenCalledWith(expect.objectContaining({ page_index: 0 }));
    expect(useAppStore.getState().project?.paginas[0].image_layers?.brush?.path).toBe(
      "images/page-a-brush-after-nav.png",
    );
    expect(useAppStore.getState().project?.paginas[1].image_layers?.brush?.path).toBe("images/brush-002.png");
    expect(useEditorStore.getState().currentPageIndex).toBe(1);
    expect(useEditorStore.getState().currentPage?.image_layers?.brush?.path).toBe("images/brush-002.png");
  });

  it("applies page-scoped healing persistence to the captured page after navigation", async () => {
    let resolveHealing!: (result: {
      page_index: number;
      inpaint_path: string;
      before_inpaint_path: string;
      bbox: [number, number, number, number];
    }) => void;
    healInpaintRegion.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveHealing = resolve;
        }),
    );
    const pageA = makePage(1);
    const pageB = makePage(2);
    useAppStore.setState({ project: makeProject([pageA, pageB]) });
    useEditorStore.getState().resetEditor();
    useEditorStore.setState({
      currentPageIndex: 0,
      currentPage: pageA,
      pendingEdits: {},
      pendingStructuralEdits: { created: [], deleted: {}, order: undefined },
    });
    const pageAKey = useEditorStore.getState().currentPageKey();

    const healingPromise = useEditorStore.getState().healPaintedRegion({
      pageKey: pageAKey,
      pageIndex: 0,
      bbox: [10, 10, 40, 40],
      maskPath: "editor_cache/healing_masks/page-0001/mask.png",
    });
    await vi.waitFor(() => expect(healInpaintRegion).toHaveBeenCalledTimes(1));

    useEditorStore.setState({ currentPageIndex: 1, currentPage: pageB });
    resolveHealing({
      page_index: 0,
      inpaint_path: "images/page-a-healed-after-nav.png",
      before_inpaint_path: "images/001.png",
      bbox: [10, 10, 40, 40],
    });
    await healingPromise;

    expect(healInpaintRegion).toHaveBeenCalledWith(expect.objectContaining({ page_index: 0 }));
    expect(useAppStore.getState().project?.paginas[0].image_layers?.inpaint?.path).toBe(
      "images/page-a-healed-after-nav.png",
    );
    expect(useAppStore.getState().project?.paginas[1].image_layers?.inpaint?.path).toBe("images/002.png");
    expect(useEditorStore.getState().currentPageIndex).toBe(1);
    expect(useEditorStore.getState().currentPage?.image_layers?.inpaint?.path).toBe("images/002.png");
  });
});

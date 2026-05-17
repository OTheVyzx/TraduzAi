import { describe, expect, it } from "vitest";
import { buildEditorScene, searchTextLayers } from "../editorScene";
import type { PageData, TextEntry, TextLayerStyle } from "../stores/appStore";

function makeStyle(overrides: Partial<TextLayerStyle> = {}): TextLayerStyle {
  return {
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
    force_upper: false,
    ...overrides,
  };
}

function makeLayer(id: string, overrides: Partial<TextEntry> = {}): TextEntry {
  const estilo = makeStyle(overrides.estilo);
  return {
    id,
    kind: "text",
    style_origin: "editor",
    bbox: [10, 20, 110, 80],
    tipo: "fala",
    original: "",
    traduzido: "",
    translated: "",
    confianca_ocr: 1,
    ocr_confidence: 1,
    estilo,
    style: estilo,
    ...overrides,
  };
}

function makePage(layers: TextEntry[] = []): PageData {
  return {
    numero: 1,
    arquivo_original: "originals/001.png",
    arquivo_traduzido: "translated/001.png",
    image_layers: {},
    text_layers: layers,
    textos: layers,
  };
}

describe("buildEditorScene", () => {
  it("returns an empty scene while preserving selectedLayerId when page is null", () => {
    const scene = buildEditorScene({
      page: null,
      pendingEdits: {
        ignored: { traduzido: "Ignorado" },
      },
      selectedLayerId: "kept",
    });

    expect(scene).toEqual({
      page: null,
      textLayers: [],
      imageLayers: [],
      selectedLayerId: "kept",
      selectedTextLayer: null,
      textCount: 0,
      visibleTextCount: 0,
    });
  });

  it("merges pending edits, sorts text layers, resolves the selected layer, and does not mutate the source page", () => {
    const page = makePage([
      makeLayer("b", {
        order: 2,
        original: "Original B",
        traduzido: "Texto B",
        bbox: [20, 20, 80, 60],
        layout_bbox: [22, 24, 82, 64],
        estilo: makeStyle({ cor: "#111111", tamanho: 24 }),
      }),
      makeLayer("a", {
        order: 1,
        original: "Original A",
        translated: "Texto A",
        traduzido: "",
        bbox: [5, 5, 50, 50],
        estilo: makeStyle({ cor: "#222222", tamanho: 20 }),
      }),
      makeLayer("c", {
        order: 1,
        original: "Original C",
        traduzido: "Texto C",
        visible: false,
      }),
    ]);
    const before = structuredClone(page);

    const scene = buildEditorScene({
      page,
      pendingEdits: {
        b: {
          traduzido: "Texto B editado",
          bbox: [30, 30, 90, 70],
          visible: false,
          locked: true,
          estilo: { cor: "#ffffff" },
        },
      },
      selectedLayerId: "b",
    });

    expect(scene.textLayers.map((layer) => layer.id)).toEqual(["a", "c", "b"]);
    expect(scene.selectedTextLayer?.id).toBe("b");
    expect(scene.selectedLayerId).toBe("b");
    expect(scene.textCount).toBe(3);
    expect(scene.visibleTextCount).toBe(1);
    expect(scene.selectedTextLayer).toMatchObject({
      displayText: "Texto B editado",
      displayOriginal: "Original B",
      effectiveBbox: [30, 30, 90, 70],
      visible: false,
      locked: true,
      order: 2,
      estilo: {
        cor: "#ffffff",
        tamanho: 24,
      },
    });
    expect(page).toEqual(before);
  });

  it("sorts text layers by patched order after pending edits are merged", () => {
    const page = makePage([
      makeLayer("a", { order: 1 }),
      makeLayer("b", { order: 2 }),
      makeLayer("c", { order: 3 }),
    ]);

    const scene = buildEditorScene({
      page,
      pendingEdits: {
        c: { order: 0 },
      },
      selectedLayerId: null,
    });

    expect(scene.textLayers.map((layer) => `${layer.id}:${layer.order}`)).toEqual([
      "c:0",
      "a:1",
      "b:2",
    ]);
  });

  it("preserves a missing selectedLayerId while selectedTextLayer stays null", () => {
    const scene = buildEditorScene({
      page: makePage([makeLayer("present")]),
      pendingEdits: {},
      selectedLayerId: "missing",
    });

    expect(scene.selectedLayerId).toBe("missing");
    expect(scene.selectedTextLayer).toBeNull();
  });

  it("merges pending style alias into both estilo and style as a full style object", () => {
    const page = makePage([
      makeLayer("a", {
        estilo: makeStyle({ cor: "#111111", tamanho: 24 }),
      }),
    ]);

    const scene = buildEditorScene({
      page,
      pendingEdits: {
        a: {
          style: { cor: "#ff00ff" },
        },
      },
      selectedLayerId: "a",
    });

    expect(scene.selectedTextLayer?.estilo.cor).toBe("#ff00ff");
    expect(scene.selectedTextLayer?.style?.cor).toBe("#ff00ff");
    expect(scene.selectedTextLayer?.estilo.tamanho).toBe(24);
    expect(scene.selectedTextLayer?.style?.tamanho).toBe(24);
    expect(scene.selectedTextLayer?.estilo.fonte).toBe("ComicNeue-Bold.ttf");
    expect(scene.selectedTextLayer?.style?.fonte).toBe("ComicNeue-Bold.ttf");
  });

  it("syncs pending bbox into bbox, layout_bbox, balloon_bbox, and effectiveBbox", () => {
    const page = makePage([
      makeLayer("a", {
        bbox: [10, 20, 110, 80],
        layout_bbox: [15, 25, 115, 85],
        balloon_bbox: [5, 10, 120, 90],
      }),
    ]);

    const scene = buildEditorScene({
      page,
      pendingEdits: {
        a: {
          bbox: [30, 40, 130, 100],
        },
      },
      selectedLayerId: "a",
    });

    expect(scene.selectedTextLayer).toMatchObject({
      bbox: [30, 40, 130, 100],
      layout_bbox: [30, 40, 130, 100],
      balloon_bbox: [30, 40, 130, 100],
      effectiveBbox: [30, 40, 130, 100],
    });
  });

  it("uses translated text for display when traduzido is empty", () => {
    const scene = buildEditorScene({
      page: makePage([makeLayer("a", { traduzido: "", translated: "Texto A" })]),
      pendingEdits: {},
      selectedLayerId: "a",
    });

    expect(scene.selectedTextLayer?.displayText).toBe("Texto A");
  });

  it("derives text status from original, translated display text, and primary OCR confidence", () => {
    const scene = buildEditorScene({
      page: makePage([
        makeLayer("a", {
          original: "  Hello  ",
          traduzido: "  Ola  ",
          confianca_ocr: 0.876,
        }),
      ]),
      pendingEdits: {},
      selectedLayerId: "a",
    });

    expect(scene.selectedTextLayer).toMatchObject({
      hasOriginal: true,
      hasTranslation: true,
      confidencePercent: 88,
    });
  });

  it("uses OCR confidence alias for status when the primary field is absent", () => {
    const scene = buildEditorScene({
      page: makePage([
        makeLayer("a", {
          original: "   ",
          traduzido: "",
          translated: "   ",
          confianca_ocr: undefined,
          ocr_confidence: 0.42,
        }),
      ]),
      pendingEdits: {},
      selectedLayerId: "a",
    });

    expect(scene.selectedTextLayer).toMatchObject({
      hasOriginal: false,
      hasTranslation: false,
      confidencePercent: 42,
    });
  });

  it("clamps derived confidence status to the 0..100 range", () => {
    const scene = buildEditorScene({
      page: makePage([
        makeLayer("low", { confianca_ocr: -0.2 }),
        makeLayer("high", { confianca_ocr: 1.4 }),
      ]),
      pendingEdits: {},
      selectedLayerId: null,
    });

    expect(scene.textLayers.find((layer) => layer.id === "low")?.confidencePercent).toBe(0);
    expect(scene.textLayers.find((layer) => layer.id === "high")?.confidencePercent).toBe(100);
  });

  it("syncs translated-only pending edits into translated, traduzido, and displayText", () => {
    const scene = buildEditorScene({
      page: makePage([makeLayer("a", { traduzido: "Antigo", translated: "Old" })]),
      pendingEdits: {
        a: {
          translated: "Texto novo",
        },
      },
      selectedLayerId: "a",
    });

    expect(scene.selectedTextLayer).toMatchObject({
      traduzido: "Texto novo",
      translated: "Texto novo",
      displayText: "Texto novo",
    });
  });

  it("normalizes image layer roles in stable order with defaults and existing overrides", () => {
    const page = makePage();
    page.image_layers = {
      brush: { key: "brush", path: "layers/brush.png", visible: true, locked: false, opacity: 0.5, order: 9 },
      rendered: { key: "rendered", path: "layers/rendered.png", visible: false, locked: true },
      base: { key: "base", path: "layers/base.png", visible: false, locked: false, opacity: 0.8 },
    };

    const scene = buildEditorScene({ page, pendingEdits: {}, selectedLayerId: null });

    expect(scene.imageLayers.map((layer) => layer.key)).toEqual([
      "base",
      "mask",
      "inpaint",
      "brush",
      "recovery",
      "rendered",
    ]);
    expect(scene.imageLayers).toEqual([
      {
        key: "base",
        path: "layers/base.png",
        visible: false,
        locked: false,
        opacity: 0.8,
        order: 0,
        technical: false,
        hasContent: true,
      },
      {
        key: "mask",
        path: null,
        visible: false,
        locked: false,
        opacity: 1,
        order: 1,
        technical: true,
        hasContent: false,
      },
      {
        key: "inpaint",
        path: null,
        visible: false,
        locked: false,
        opacity: 1,
        order: 2,
        technical: false,
        hasContent: false,
      },
      {
        key: "brush",
        path: "layers/brush.png",
        visible: true,
        locked: false,
        opacity: 0.5,
        order: 9,
        technical: false,
        hasContent: true,
      },
      {
        key: "recovery",
        path: null,
        visible: false,
        locked: false,
        opacity: 1,
        order: 4,
        technical: false,
        hasContent: false,
      },
      {
        key: "rendered",
        path: "layers/rendered.png",
        visible: false,
        locked: true,
        opacity: 1,
        order: 5,
        technical: false,
        hasContent: true,
      },
    ]);
  });

  it("defaults rendered image layer to translated image with visible and locked enabled", () => {
    const page = makePage();
    page.image_layers = {};

    const scene = buildEditorScene({ page, pendingEdits: {}, selectedLayerId: null });
    const rendered = scene.imageLayers.find((layer) => layer.key === "rendered");

    expect(rendered).toMatchObject({
      key: "rendered",
      path: "translated/001.png",
      visible: true,
      locked: true,
      hasContent: true,
    });
  });
});

describe("searchTextLayers", () => {
  it("matches translated text, original text, and tipo", () => {
    const layers = buildEditorScene({
      page: makePage([
        makeLayer("speech", { traduzido: "Ola mundo", original: "Hello world", tipo: "fala" }),
        makeLayer("thought", { traduzido: "Silencio", original: "Quiet", tipo: "pensamento" }),
        makeLayer("sfx", { traduzido: "Bang", original: "Boom", tipo: "sfx" }),
      ]),
      pendingEdits: {},
      selectedLayerId: null,
    }).textLayers;

    expect(searchTextLayers(layers, " mundo ").map((layer) => layer.id)).toEqual(["speech"]);
    expect(searchTextLayers(layers, "quiet").map((layer) => layer.id)).toEqual(["thought"]);
    expect(searchTextLayers(layers, "SFX").map((layer) => layer.id)).toEqual(["sfx"]);
    expect(searchTextLayers(layers, "   ")).toBe(layers);
  });
});

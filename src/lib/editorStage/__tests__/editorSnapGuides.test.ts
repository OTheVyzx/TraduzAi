import { describe, expect, it } from "vitest";
import { snapRectToGuides } from "../../../components/editor/stage/snapGuides";
import type { TextEntry, TextLayerStyle } from "../../stores/appStore";

function makeStyle(): TextLayerStyle {
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
  };
}

function makeLayer(id: string, bbox: TextEntry["bbox"]): TextEntry {
  const estilo = makeStyle();
  return {
    id,
    kind: "text",
    bbox,
    layout_bbox: bbox,
    source_bbox: bbox,
    render_bbox: null,
    tipo: "fala",
    original: "",
    traduzido: "",
    translated: "",
    confianca_ocr: 1,
    ocr_confidence: 1,
    estilo,
    style: estilo,
    visible: true,
    locked: false,
    order: 0,
  };
}

describe("editor snap guides", () => {
  it("snaps a moving rect to the page center", () => {
    const result = snapRectToGuides(
      { x: 147, y: 20, width: 100, height: 80 },
      {
        pageSize: { width: 500, height: 700 },
        layers: [],
        excludeLayerId: "selected",
        threshold: 5,
      },
    );

    expect(result.rect.x).toBe(150);
    expect(result.guides).toContainEqual(
      expect.objectContaining({ orientation: "vertical", position: 250 }),
    );
  });

  it("snaps a moving rect edge to another text layer edge", () => {
    const other = makeLayer("other", [300, 40, 420, 140]);
    const result = snapRectToGuides(
      { x: 197, y: 200, width: 100, height: 80 },
      {
        pageSize: { width: 500, height: 700 },
        layers: [other],
        excludeLayerId: "selected",
        threshold: 5,
      },
    );

    expect(result.rect.x + result.rect.width).toBe(300);
    expect(result.guides).toContainEqual(
      expect.objectContaining({ orientation: "vertical", position: 300 }),
    );
  });
});

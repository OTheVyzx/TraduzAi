import { describe, expect, it } from "vitest";
import { hydratePageData, hydrateProjectJson } from "../tauri";

describe("hydratePageData text style", () => {
  it("normalizes legacy default style to black text without effects", () => {
    const page = hydratePageData(
      {
        numero: 1,
        arquivo_original: "originals/001.jpg",
        arquivo_traduzido: "translated/001.jpg",
        text_layers: [
          {
            id: "tl_001_001",
            bbox: [100, 200, 500, 700],
            tipo: "fala",
            original: "Ghislain",
            traduzido: "Ghislain",
            confianca_ocr: 0.91,
            estilo: {
              fonte: "CCDaveGibbonsLower W00 Regular.ttf",
              tamanho: 28,
              cor: "#FFFFFF",
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
              force_upper: false,
            },
          },
        ],
      },
      "D:/TraduzAi/outapp/traduzido15/project.json",
    );

    expect(page.text_layers[0].estilo).toMatchObject({
      fonte: "ComicNeue-Bold.ttf",
      cor: "#000000",
      contorno: "",
      contorno_px: 0,
      glow: false,
      glow_px: 0,
      sombra: false,
      sombra_offset: [0, 0],
      bold: true,
    });
    expect(page.text_layers[0].style_origin).toBe("legacy");
  });

  it("preserves explicit font and effects", () => {
    const page = hydratePageData(
      {
        numero: 1,
        arquivo_original: "originals/001.jpg",
        arquivo_traduzido: "translated/001.jpg",
        text_layers: [
          {
            id: "tl_001_001",
            bbox: [100, 200, 500, 700],
            tipo: "fala",
            original: "Texto",
            traduzido: "Texto",
            confianca_ocr: 0.91,
            style_origin: "editor",
            estilo: {
              fonte: "Newrotic.ttf",
              tamanho: 30,
              cor: "#FFFFFF",
              cor_gradiente: [],
              contorno: "#000000",
              contorno_px: 2,
              glow: true,
              glow_cor: "#ff00ff",
              glow_px: 3,
              sombra: true,
              sombra_cor: "#111111",
              sombra_offset: [2, 3],
              bold: false,
              italico: true,
              rotacao: 0,
              alinhamento: "center",
              force_upper: false,
            },
          },
        ],
      },
      "D:/TraduzAi/outapp/traduzido15/project.json",
    );

    expect(page.text_layers[0].estilo).toMatchObject({
      fonte: "Newrotic.ttf",
      cor: "#FFFFFF",
      contorno: "#000000",
      contorno_px: 2,
      glow: true,
      sombra: true,
      bold: false,
      italico: true,
    });
    expect(page.text_layers[0].style_origin).toBe("editor");
  });
});

describe("hydrateProjectJson", () => {
  it("usa a pasta do project.json quando source/output vierem vazios", () => {
    const project = hydrateProjectJson(
      {
        obra: "",
        capitulo: 1,
        source_path: "",
        output_path: "",
        _work_dir: "N:/TraduzAI/TraduzAi/data/works/abc",
        paginas: [
          {
            numero: 1,
            arquivo_original: "originals/001.jpg",
            arquivo_traduzido: "translated/001.jpg",
            text_layers: [],
            textos: [],
          },
        ],
      },
      "N:/TraduzAI/TraduzAi/data/works/abc/project.json",
    );

    expect(project.source_path).toBe("N:/TraduzAI/TraduzAi/data/works/abc");
    expect(project.output_path).toBe("N:/TraduzAI/TraduzAi/data/works/abc");
    expect(project._work_dir).toBe("N:/TraduzAI/TraduzAi/data/works/abc");
    expect(project.paginas?.[0]?.arquivo_traduzido).toBe(
      "N:/TraduzAI/TraduzAi/data/works/abc/translated/001.jpg",
    );
    expect(project.paginas?.[0]?.image_layers?.base?.path).toBe(
      "N:/TraduzAI/TraduzAi/data/works/abc/originals/001.jpg",
    );
  });
});

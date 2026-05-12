import test from "node:test";
import assert from "node:assert/strict";

import { buildPlainPageCommandArgs, hydratePageData } from "./tauri.ts";

test("buildPlainPageCommandArgs converte project_path/page_index para o contrato camelCase do Tauri", () => {
  assert.deepEqual(
    buildPlainPageCommandArgs({
      project_path: "D:/TraduzAi/outapp/traduzido15",
      page_index: 7,
    }),
    {
      projectPath: "D:/TraduzAi/outapp/traduzido15",
      pageIndex: 7,
    },
  );
});

test("hydratePageData usa render_bbox como bbox principal da UI quando existir", () => {
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
          traduzido: "Ghislain",
          confianca_ocr: 0.91,
          original: "Ghislain",
          translated: "Ghislain",
          layout_bbox: [100, 200, 500, 700],
          render_bbox: [140, 260, 420, 520],
          source_bbox: [90, 180, 530, 740],
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
          style: {
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

  assert.deepEqual(page.text_layers[0].bbox, [140, 260, 420, 520]);
  assert.deepEqual(page.text_layers[0].layout_bbox, [100, 200, 500, 700]);
  assert.deepEqual(page.text_layers[0].render_bbox, [140, 260, 420, 520]);
  assert.equal(page.text_layers[0].estilo.fonte, "ComicNeue-Bold.ttf");
  assert.equal(page.text_layers[0].estilo.cor, "#000000");
  assert.equal(page.text_layers[0].estilo.contorno, "");
  assert.equal(page.text_layers[0].estilo.contorno_px, 0);
  assert.equal(page.text_layers[0].estilo.glow, false);
  assert.equal(page.text_layers[0].estilo.sombra, false);
});

test("hydratePageData preserva fonte e efeitos explicitos", () => {
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

  assert.equal(page.text_layers[0].estilo.fonte, "Newrotic.ttf");
  assert.equal(page.text_layers[0].estilo.cor, "#FFFFFF");
  assert.equal(page.text_layers[0].estilo.contorno, "#000000");
  assert.equal(page.text_layers[0].estilo.contorno_px, 2);
  assert.equal(page.text_layers[0].estilo.glow, true);
  assert.equal(page.text_layers[0].estilo.sombra, true);
  assert.equal(page.text_layers[0].estilo.bold, false);
  assert.equal(page.text_layers[0].estilo.italico, true);
});

test("hydratePageData preserva rotacao manual separada do metadado OCR", () => {
  const page = hydratePageData(
    {
      numero: 1,
      arquivo_original: "originals/001.jpg",
      arquivo_traduzido: "translated/001.jpg",
      text_layers: [
        {
          id: "tl_rotated",
          bbox: [10, 20, 110, 120],
          tipo: "fala",
          original: "Texto",
          traduzido: "Texto",
          translated: "Texto",
          confianca_ocr: 0.91,
          ocr_confidence: 0.91,
          rotation_deg: 90,
          estilo: {
            fonte: "ComicNeue-Bold.ttf",
            tamanho: 28,
            cor: "#111111",
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
            rotacao: 15,
            alinhamento: "center",
            force_upper: false,
          },
        },
      ],
    },
    "D:/TraduzAi/outapp/traduzido15/project.json",
  );

  const layer = page.text_layers[0];
  assert.ok(layer);
  assert.ok(layer.style);
  assert.equal(layer.estilo.rotacao, 15);
  assert.equal(layer.style.rotacao, 15);
  assert.equal(layer.rotation_deg, 90);
});

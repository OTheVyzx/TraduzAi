import test from "node:test";
import assert from "node:assert/strict";

import type { PageData } from "../lib/stores/appStore";
import { getPreviewImageCandidates, getPreviewToggleLabel } from "./previewImage.ts";

function makePage(overrides: Partial<PageData> = {}): PageData {
  return {
    numero: 1,
    arquivo_original: "D:/TraduzAi/out/originals/001.jpg",
    arquivo_traduzido: "D:/TraduzAi/out/translated/001.jpg",
    image_layers: {
      base: {
        key: "base",
        path: "D:/TraduzAi/out/originals/001.jpg",
        visible: true,
        locked: true,
      },
      inpaint: {
        key: "inpaint",
        path: "D:/TraduzAi/out/images/001.jpg",
        visible: false,
        locked: true,
      },
      rendered: {
        key: "rendered",
        path: "D:/TraduzAi/out/translated/001.jpg",
        visible: true,
        locked: true,
      },
    },
    inpaint_blocks: [],
    text_layers: [],
    textos: [],
    ...overrides,
  };
}

test("getPreviewImageCandidates usa image_layers como fonte principal do preview", () => {
  const page = makePage({
    arquivo_original: "D:/legado/original-velho.jpg",
    arquivo_traduzido: "D:/legado/render-velho.jpg",
    image_layers: {
      base: { key: "base", path: "D:/novo/original.jpg", visible: true, locked: true },
      rendered: { key: "rendered", path: "D:/novo/render.jpg", visible: true, locked: true },
    },
  });

  assert.deepEqual(getPreviewImageCandidates(page, true), ["D:/novo/original.jpg"]);
  assert.deepEqual(getPreviewImageCandidates(page, false), ["D:/novo/render.jpg", "D:/novo/original.jpg"]);
});

test("getPreviewImageCandidates usa inpaint antes do original quando o render falhar", () => {
  const page = makePage({
    image_layers: {
      base: { key: "base", path: "D:\\TraduzAi\\out\\originals\\001.jpg", visible: true, locked: true },
      inpaint: { key: "inpaint", path: "D:\\TraduzAi\\out\\images\\001.jpg", visible: false, locked: true },
      rendered: { key: "rendered", path: null, visible: true, locked: true },
    },
    arquivo_traduzido: "",
  });

  assert.deepEqual(getPreviewImageCandidates(page, false), [
    "D:/TraduzAi/out/images/001.jpg",
    "D:/TraduzAi/out/originals/001.jpg",
  ]);
});

test("getPreviewToggleLabel descreve a ação do botão", () => {
  assert.equal(getPreviewToggleLabel(false), "Ver original");
  assert.equal(getPreviewToggleLabel(true), "Ver traduzido");
});

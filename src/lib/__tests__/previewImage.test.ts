import { describe, expect, it } from "vitest";
import type { PageData } from "../stores/appStore";
import { getPreviewImageCandidates } from "../../pages/previewImage";

function makePage(overrides: Partial<PageData> = {}): PageData {
  return {
    numero: 1,
    arquivo_original: "originals/001.jpg",
    arquivo_traduzido: "translated/001.jpg",
    image_layers: {
      base: { key: "base", path: "originals/001.jpg", visible: true, locked: true },
      rendered: { key: "rendered", path: "translated/001.jpg", visible: true, locked: true },
    },
    inpaint_blocks: [],
    text_layers: [],
    textos: [],
    ...overrides,
  };
}

describe("getPreviewImageCandidates", () => {
  it("resolves relative image paths against the project directory", () => {
    expect(getPreviewImageCandidates(makePage(), false, "N:/TraduzAI/TraduzAi/data/works/abc/project.json")).toEqual([
      "N:/TraduzAI/TraduzAi/data/works/abc/translated/001.jpg",
      "N:/TraduzAI/TraduzAi/data/works/abc/originals/001.jpg",
    ]);
  });

  it("preserves direct browser image sources", () => {
    expect(
      getPreviewImageCandidates(
        makePage({
          image_layers: {
            base: { key: "base", path: "/fixture/original.jpg", visible: true, locked: true },
            rendered: { key: "rendered", path: "data:image/png;base64,AAA", visible: true, locked: true },
          },
        }),
        false,
        "N:/TraduzAI/TraduzAi/data/works/abc",
      ),
    ).toEqual(["data:image/png;base64,AAA", "/fixture/original.jpg"]);
  });
});

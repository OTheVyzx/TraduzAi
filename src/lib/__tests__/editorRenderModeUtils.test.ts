import { describe, expect, it } from "vitest";
import type { PageData } from "../stores/appStore";
import type { RenderPreviewCacheEntry } from "../stores/editorStore";
import { displayImagePathForMode, isFaithfulPreviewMode } from "../../components/editor/stage/renderModeUtils";

const page = {
  numero: 1,
  arquivo_original: "originals/001.jpg",
  arquivo_traduzido: "translated/001.jpg",
  image_layers: {
    base: { key: "base", path: "originals/001.jpg", visible: true, locked: true },
    inpaint: { key: "inpaint", path: "images/001.jpg", visible: true, locked: true },
    rendered: { key: "rendered", path: "translated/001.jpg", visible: true, locked: true },
  },
  text_layers: [],
  textos: [],
} satisfies PageData;

const freshPreview = {
  status: "fresh",
  path: "translated/001.jpg",
  previewPath: "render-cache/preview/001-preview.jpg",
  rendererBackend: "python",
  fingerprint: "fresh",
  generatedAt: 1,
  error: null,
} satisfies RenderPreviewCacheEntry;

describe("editor render mode image selection", () => {
  it("keeps the translated editor on the editable base image even when a rendered preview is fresh", () => {
    expect(displayImagePathForMode(page, "translated", freshPreview)).toBe("images/001.jpg");
    expect(isFaithfulPreviewMode("translated", freshPreview)).toBe(false);
  });

  it("still uses the original image in original mode", () => {
    expect(displayImagePathForMode(page, "original", freshPreview)).toBe("originals/001.jpg");
  });
});

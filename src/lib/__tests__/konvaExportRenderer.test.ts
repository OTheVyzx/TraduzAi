import { afterEach, describe, expect, it, vi } from "vitest";
import { shouldUseKonvaPreviewRenderer } from "../konvaExportRenderer";

describe("Konva preview renderer flag", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("keeps faithful rendered images as the default preview path", () => {
    vi.stubEnv("VITE_TRADUZAI_KONVA_RENDER_PREVIEW", "");

    expect(shouldUseKonvaPreviewRenderer()).toBe(false);
  });

  it("allows the browser Konva preview renderer as an explicit opt-in", () => {
    vi.stubEnv("VITE_TRADUZAI_KONVA_RENDER_PREVIEW", "1");

    expect(shouldUseKonvaPreviewRenderer()).toBe(true);
  });
});

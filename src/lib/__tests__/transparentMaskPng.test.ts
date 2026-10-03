import { describe, expect, it } from "vitest";
import { transparentMaskPngDataUrl } from "../transparentMaskPng";

function dataUrlBytes(dataUrl: string) {
  const payload = dataUrl.split(",", 2)[1] ?? "";
  return Uint8Array.from(atob(payload), (character) => character.charCodeAt(0));
}

describe("transparentMaskPngDataUrl", () => {
  it("returns a non-empty PNG data URL", async () => {
    const pngData = await transparentMaskPngDataUrl(100, 200);
    const bytes = dataUrlBytes(pngData);
    expect(pngData).toMatch(/^data:image\/png;base64,/);
    expect(bytes.slice(0, 8)).toEqual(Uint8Array.from([137, 80, 78, 71, 13, 10, 26, 10]));
  });

  it("rejects unavailable page dimensions", async () => {
    await expect(transparentMaskPngDataUrl(0, 200)).rejects.toThrow("Dimensões da página indisponíveis");
  });
});

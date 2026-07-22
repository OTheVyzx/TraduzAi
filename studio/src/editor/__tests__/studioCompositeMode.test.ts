import { describe, expect, it } from "vitest";
import {
  isStudioBitmapCompositeActive,
  visibleBitmapOverlayPath,
} from "../../../../src/components/editor/stage/renderModeUtils";

describe("Studio bitmap composite mode", () => {
  it("uses the composite only on editable Studio views", () => {
    expect(isStudioBitmapCompositeActive("studio", "translated", "data:image/png;base64,x")).toBe(true);
    expect(isStudioBitmapCompositeActive("studio", "inpainted", "data:image/png;base64,x")).toBe(true);
    expect(isStudioBitmapCompositeActive("studio", "original", "data:image/png;base64,x")).toBe(false);
    expect(isStudioBitmapCompositeActive("traduzai", "translated", "data:image/png;base64,x")).toBe(false);
  });

  it("does not activate before the Studio composite is ready", () => {
    expect(isStudioBitmapCompositeActive("studio", "translated", null)).toBe(false);
  });

  it("keeps mask and brush sources loaded while the Studio composite is visible", () => {
    const page = {
      image_layers: {
        mask: { key: "mask", path: "mask.png", visible: true },
        brush: { key: "brush", path: "brush.png", visible: true },
      },
    } as never;

    expect(isStudioBitmapCompositeActive("studio", "translated", "data:image/png;base64,x")).toBe(true);
    expect(visibleBitmapOverlayPath(page, "mask")).toBe("mask.png");
    expect(visibleBitmapOverlayPath(page, "brush")).toBe("brush.png");
  });
});

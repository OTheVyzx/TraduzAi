import { createCanvas } from "canvas";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  createBitmapStrokePreviewOnCanvas,
  strokePassesForHardness,
} from "../../../components/editor/stage/bitmapStrokePreview";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("bitmap stroke preview", () => {
  it("uses a single pass for hard brush strokes", () => {
    expect(strokePassesForHardness({ brushSize: 20, opacity: 0.75, hardness: 1 })).toEqual([
      { width: 20, alpha: 0.75 },
    ]);
  });

  it("adds soft outer passes when brush hardness is reduced", () => {
    const passes = strokePassesForHardness({ brushSize: 20, opacity: 0.8, hardness: 0.25 });

    expect(passes.length).toBeGreaterThan(1);
    expect(passes[0].width).toBeGreaterThan(20);
    expect(passes[0].alpha).toBeLessThan(0.8);
    expect(passes[passes.length - 1]).toEqual({ width: 20, alpha: 0.8 });
  });

  it("clips the authoritative bitmap preview with the composite alpha mask", () => {
    vi.stubGlobal("document", {
      createElement: () => createCanvas(1, 1),
    });
    const working = createCanvas(5, 3);
    const mask = createCanvas(5, 3);
    const maskContext = mask.getContext("2d");
    maskContext.fillStyle = "#fff";
    maskContext.fillRect(2, 0, 1, 3);

    createBitmapStrokePreviewOnCanvas(working as unknown as HTMLCanvasElement, {
      layerKey: "brush",
      stroke: [[0, 1], [4, 1]],
      brushSize: 1,
      color: "#ff0000",
      opacity: 1,
      hardness: 1,
      erase: false,
      clipMaskImage: mask as unknown as CanvasImageSource,
    });

    const pixels = working.getContext("2d").getImageData(0, 0, 5, 3).data;
    expect(pixels[3]).toBe(0);
    expect(pixels[(1 * 5 + 2) * 4]).toBe(255);
    expect(pixels[(1 * 5 + 2) * 4 + 3]).toBeGreaterThan(0);
    expect(pixels[(1 * 5 + 4) * 4 + 3]).toBe(0);
  }, 15_000);
});

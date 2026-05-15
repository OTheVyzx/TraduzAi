import { describe, expect, it } from "vitest";
import { strokePassesForHardness } from "../../../components/editor/stage/bitmapStrokePreview";

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
});

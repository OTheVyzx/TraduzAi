import { describe, expect, it } from "vitest";
import {
  clampHealingBrushBBox,
  paddedStrokeBBox,
} from "../../components/editor/stage/healingBrushMask";

describe("healing brush mask geometry", () => {
  it("clamps bbox to image bounds", () => {
    expect(clampHealingBrushBBox([-10, 5, 120, 80], 100, 60)).toEqual([0, 5, 100, 60]);
  });

  it("pads stroke bbox with the healing context padding", () => {
    expect(
      paddedStrokeBBox({
        stroke: [[50, 50], [60, 70]],
        brushSize: 12,
        width: 200,
        height: 200,
      }),
    ).toEqual([18, 18, 92, 102]);
  });

  it("rejects empty stroke", () => {
    expect(paddedStrokeBBox({ stroke: [], brushSize: 12, width: 200, height: 200 })).toBeNull();
  });
});

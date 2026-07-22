import { describe, expect, it } from "vitest";
import {
  applySelectionAlphaModifiers,
  combineLassoSelections,
  createLassoSelection,
  lassoBoundingBox,
  lassoSelectionEffectiveBbox,
  lassoSelectionProcessingBbox,
  withLassoSelectionModifiers,
} from "../lassoSelection";

describe("lasso selection geometry", () => {
  it("builds an active selection with bbox and page identity", () => {
    const selection = createLassoSelection({
      pageKey: "page-1",
      pageIndex: 0,
      points: [
        [40.4, 20.2],
        [88.6, 24.9],
        [81.1, 61.8],
        [45.2, 55.1],
      ],
      width: 120,
      height: 90,
    });

    expect(selection).toEqual({
      pageKey: "page-1",
      pageIndex: 0,
      points: [
        [40, 20],
        [89, 25],
        [81, 62],
        [45, 55],
      ],
      bbox: [40, 20, 89, 62],
      width: 120,
      height: 90,
    });
  });

  it("keeps bbox inside image bounds", () => {
    expect(
      lassoBoundingBox([
        [-10, 5],
        [30, 110],
        [150, 20],
      ], 100, 80),
    ).toEqual([0, 5, 100, 80]);
  });

  it("combines additive and subtractive regions without losing the original polygon", () => {
    const base = createLassoSelection({
      pageKey: "page-1",
      pageIndex: 0,
      points: [[10, 10], [50, 10], [50, 50], [10, 50]],
      width: 100,
      height: 80,
    });
    const added = createLassoSelection({
      pageKey: "page-1",
      pageIndex: 0,
      points: [[60, 20], [90, 20], [90, 40], [60, 40]],
      width: 100,
      height: 80,
    });
    const subtracted = createLassoSelection({
      pageKey: "page-1",
      pageIndex: 0,
      points: [[20, 20], [30, 20], [30, 30], [20, 30]],
      width: 100,
      height: 80,
    });

    const combined = combineLassoSelections(
      combineLassoSelections(base, added, "add"),
      subtracted,
      "subtract",
    );

    expect(combined.regions).toEqual([
      { operation: "add", points: base.points },
      { operation: "add", points: added.points },
      { operation: "subtract", points: subtracted.points },
    ]);
    expect(combined.bbox).toEqual([10, 10, 90, 50]);
  });

  it("applies expansion, contraction and feather to selection bounds", () => {
    const selection = withLassoSelectionModifiers(
      createLassoSelection({
        pageKey: "page-1",
        pageIndex: 0,
        points: [[20, 20], [80, 20], [80, 60], [20, 60]],
        width: 100,
        height: 80,
      }),
      { expansion: -6, feather: 10, targetNodeId: "image:base" },
    );

    expect(selection).toMatchObject({
      expansion: -6,
      feather: 10,
      targetNodeId: "image:base",
    });
    expect(lassoSelectionEffectiveBbox(selection)).toEqual([16, 16, 84, 64]);
    expect(lassoSelectionProcessingBbox(selection)).toEqual([10, 10, 90, 70]);
  });

  it("expands, contracts and feathers a pure alpha mask", () => {
    const source = new Uint8ClampedArray([
      0, 0, 0, 0, 0,
      0, 0, 0, 0, 0,
      0, 0, 255, 0, 0,
      0, 0, 0, 0, 0,
      0, 0, 0, 0, 0,
    ]);

    const expanded = applySelectionAlphaModifiers(source, 5, 5, { expansion: 1, feather: 0 });
    expect(expanded[12]).toBe(255);
    expect(expanded[7]).toBe(255);
    expect(expanded[11]).toBe(255);

    const contracted = applySelectionAlphaModifiers(expanded, 5, 5, { expansion: -1, feather: 0 });
    expect(contracted[12]).toBe(255);
    expect(contracted[7]).toBe(0);

    const feathered = applySelectionAlphaModifiers(source, 5, 5, { expansion: 0, feather: 1 });
    expect(feathered[12]).toBeGreaterThan(0);
    expect(feathered[12]).toBeLessThan(255);
    expect(feathered[7]).toBeGreaterThan(0);
  });
});

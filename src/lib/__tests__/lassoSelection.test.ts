import { describe, expect, it } from "vitest";
import { createLassoSelection, lassoBoundingBox } from "../lassoSelection";

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
});

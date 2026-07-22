import { describe, expect, it } from "vitest";
import { adjustStudioSelection, createStudioSelection } from "../../selection/selectionModel";
import { fluxCropBbox } from "../fluxImagePreparation";

describe("FLUX image preparation", () => {
  it("adds local context, respects feather/expansion and aligns the crop", () => {
    const selection = adjustStudioSelection(createStudioSelection({
      id: "selection:crop",
      pageKey: "memory://crop::0",
      pageIndex: 0,
      points: [[100, 100], [140, 100], [140, 140], [100, 140]],
      width: 200,
      height: 200,
      targetNodeId: "image:inpaint",
    }), { feather: 10, expansion: 5 });

    expect(fluxCropBbox(selection, { contextPadding: 32, alignment: 16 })).toEqual([48, 48, 192, 192]);
  });

  it("never expands a crop beyond page boundaries", () => {
    const selection = createStudioSelection({
      id: "selection:edge",
      pageKey: "memory://crop::0",
      pageIndex: 0,
      points: [[2, 2], [10, 2], [10, 10], [2, 10]],
      width: 100,
      height: 80,
      targetNodeId: "image:inpaint",
    });

    expect(fluxCropBbox(selection, { contextPadding: 32, alignment: 16 })).toEqual([0, 0, 48, 48]);
  });
});

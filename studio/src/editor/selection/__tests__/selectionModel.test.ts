import { describe, expect, it } from "vitest";
import type { StudioScene } from "../../../project/studioProject";
import {
  adjustStudioSelection,
  attachStudioSelectionMask,
  combineStudioSelections,
  createStudioSelection,
  studioSelectionEffectiveBbox,
} from "../selectionModel";

function makeScene(): StudioScene {
  return {
    version: "1.0",
    roots: ["image:base"],
    nodes: [
      {
        id: "image:base",
        kind: "raster",
        name: "Arte base",
        visible: true,
        locked: false,
        opacity: 1,
        blend_mode: "normal",
        parent_id: null,
        order: 0,
        mask_ids: [],
        image_layer_key: "base",
        metadata: {},
      },
    ],
  };
}

describe("Studio selection model", () => {
  it("keeps professional modifiers and target layer in a serializable selection", () => {
    const selection = adjustStudioSelection(
      createStudioSelection({
        id: "selection:1",
        pageKey: "project::0",
        pageIndex: 0,
        points: [[10.2, 12.8], [80.1, 12], [80, 70], [10, 70]],
        width: 100,
        height: 90,
        targetNodeId: "image:base",
      }),
      { feather: 8, expansion: -3 },
    );

    expect(selection).toMatchObject({
      id: "selection:1",
      targetNodeId: "image:base",
      feather: 8,
      expansion: -3,
    });
    expect(selection.regions).toHaveLength(1);
    expect(studioSelectionEffectiveBbox(selection)).toEqual([5, 7, 85, 75]);
  });

  it("combines regions using add and subtract operations", () => {
    const base = createStudioSelection({
      id: "selection:base",
      pageKey: "project::0",
      pageIndex: 0,
      points: [[10, 10], [40, 10], [40, 40], [10, 40]],
      width: 100,
      height: 100,
      targetNodeId: "image:base",
    });
    const next = createStudioSelection({
      id: "selection:next",
      pageKey: "project::0",
      pageIndex: 0,
      points: [[50, 20], [70, 20], [70, 50], [50, 50]],
      width: 100,
      height: 100,
      targetNodeId: "image:base",
    });

    const combined = combineStudioSelections(base, next, "add");
    expect(combined.regions.map((region) => region.operation)).toEqual(["add", "add"]);
    expect(combined.bbox).toEqual([10, 10, 70, 50]);
  });

  it("attaches a non-destructive mask node to the target layer", () => {
    const scene = makeScene();
    const selection = adjustStudioSelection(
      createStudioSelection({
        id: "selection:1",
        pageKey: "project::0",
        pageIndex: 0,
        points: [[10, 10], [80, 10], [80, 70], [10, 70]],
        width: 100,
        height: 90,
        targetNodeId: "image:base",
      }),
      { feather: 6, expansion: 2 },
    );

    const next = attachStudioSelectionMask(scene, selection, {
      maskId: "mask:selection:1",
      name: "Máscara de retoque",
    });

    expect(scene.nodes).toHaveLength(1);
    expect(next.nodes.find((node) => node.id === "image:base")?.mask_ids).toEqual(["mask:selection:1"]);
    expect(next.nodes.find((node) => node.id === "mask:selection:1")).toMatchObject({
      kind: "mask",
      parent_id: "image:base",
      name: "Máscara de retoque",
      metadata: {
        mask_role: "layer",
        target_node_id: "image:base",
        selection: expect.objectContaining({ feather: 6, expansion: 2 }),
      },
    });
  });

  it("rejects a layer mask without a valid target", () => {
    const selection = createStudioSelection({
      id: "selection:1",
      pageKey: "project::0",
      pageIndex: 0,
      points: [[10, 10], [80, 10], [80, 70]],
      width: 100,
      height: 90,
    });

    expect(() => attachStudioSelectionMask(makeScene(), selection)).toThrow(/camada-alvo/i);
  });

  it("moves a reused mask id without leaving a stale link on the old target", () => {
    const scene = makeScene();
    scene.roots.push("image:ink");
    scene.nodes.push({
      ...scene.nodes[0],
      id: "image:ink",
      name: "Traço",
      order: 1,
      image_layer_key: "brush",
      mask_ids: [],
    });
    const baseSelection = createStudioSelection({
      id: "selection:1",
      pageKey: "project::0",
      pageIndex: 0,
      points: [[10, 10], [80, 10], [80, 70]],
      width: 100,
      height: 90,
      targetNodeId: "image:base",
    });
    const first = attachStudioSelectionMask(scene, baseSelection, { maskId: "mask:shared" });
    const moved = attachStudioSelectionMask(
      first,
      adjustStudioSelection(baseSelection, { targetNodeId: "image:ink" }),
      { maskId: "mask:shared" },
    );

    expect(moved.nodes.find((node) => node.id === "image:base")?.mask_ids).toEqual([]);
    expect(moved.nodes.find((node) => node.id === "image:ink")?.mask_ids).toEqual(["mask:shared"]);
  });
});

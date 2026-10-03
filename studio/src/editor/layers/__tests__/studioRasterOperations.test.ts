import { describe, expect, it } from "vitest";
import type { StudioScene } from "../../../project/studioProject";
import { appendStudioBitmapOperation } from "../studioRasterOperations";

function makeScene(layerKey: "brush" | "inpaint"): StudioScene {
  const category = layerKey === "brush" ? "painting" : "cleanup";
  return {
    version: "1.0",
    roots: ["image:base", `image:${layerKey}`],
    nodes: [
      {
        id: "image:base",
        kind: "raster",
        name: "Original",
        visible: true,
        locked: true,
        opacity: 1,
        blend_mode: "normal",
        parent_id: null,
        order: 0,
        mask_ids: [],
        image_layer_key: "base",
        metadata: {},
      },
      {
        id: `image:${layerKey}`,
        kind: "raster",
        name: layerKey === "brush" ? "Pintura" : "Limpeza",
        visible: true,
        locked: false,
        opacity: 1,
        blend_mode: "normal",
        parent_id: null,
        order: 1,
        mask_ids: [],
        image_layer_key: layerKey,
        metadata: { projected_from: "image_layers" },
      },
    ],
  };
}

function operation(
  layerKey: "brush" | "inpaint",
  commandId: string,
  operationPath: string,
  baselinePath: string | null,
  blendMode: "normal" | "destination-out" = "normal",
) {
  return {
    commandId,
    layerKey,
    name: blendMode === "destination-out" ? "Borracha" : layerKey === "brush" ? "Pincelada" : "Limpeza",
    pngData: `data:image/png;${commandId}`,
    baselineDataUrl: null,
    bbox: [10, 20, 30, 40] as [number, number, number, number],
    blendMode,
    baselinePath,
    operationPath,
  };
}

describe("Studio raster operation layers", () => {
  it("stores every brush stroke as a separate PNG under Pintura and keeps erases ordered", () => {
    const original = makeScene("brush");
    const first = appendStudioBitmapOperation(original, operation("brush", "brush-1", "assets/brush-1.png", "assets/brush-base.png"));
    const second = appendStudioBitmapOperation(first, operation("brush", "erase-1", "assets/erase-1.png", null, "destination-out"));

    const group = second.nodes.find((node) => node.id === "group:auto:painting");
    const children = second.nodes.filter((node) => node.parent_id === group?.id).sort((left, right) => left.order - right.order);
    expect(original.nodes.some((node) => node.id === "image:brush")).toBe(true);
    expect(second.roots).toEqual(["image:base", "group:auto:painting"]);
    expect(group?.metadata).toMatchObject({ auto_category_group: "painting", raster_operations_key: "brush" });
    expect(children.map((node) => node.metadata.image_path)).toEqual([
      "assets/brush-base.png",
      "assets/brush-1.png",
      "assets/erase-1.png",
    ]);
    expect(children[2]).toMatchObject({ blend_mode: "destination-out", metadata: { operation_kind: "erase" } });
  });

  it("places each masked cleanup result inside the Limpeza group", () => {
    const next = appendStudioBitmapOperation(
      makeScene("inpaint"),
      operation("inpaint", "cleanup-1", "assets/cleanup-1.png", null),
    );
    const group = next.nodes.find((node) => node.id === "group:auto:cleanup");
    const cleanup = next.nodes.find((node) => node.id === "generated:studio-stroke:cleanup-1");

    expect(next.roots).toEqual(["image:base", "group:auto:cleanup"]);
    expect(group?.metadata).toMatchObject({ auto_category_group: "cleanup", raster_operations_key: "inpaint" });
    expect(cleanup).toMatchObject({ kind: "generated", parent_id: group?.id, metadata: { image_path: "assets/cleanup-1.png" } });
  });
});

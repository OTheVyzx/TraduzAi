import { describe, expect, it } from "vitest";
import type { StudioScene } from "../../project/studioProject";
import { createStudioSelection } from "../../editor/selection/selectionModel";
import {
  acceptFluxVariant,
  activateFluxVariant,
  applyFluxGenerationToScene,
  assertFluxExecutionContext,
  createFluxGeneration,
  findFluxGenerationId,
  isolateFluxVariantPixels,
  rejectFluxGeneration,
} from "../fluxContract";

function sceneWithTarget(): StudioScene {
  return {
    version: "1.0",
    roots: ["image:base", "image:inpaint"],
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
        id: "image:inpaint",
        kind: "raster",
        name: "Limpeza",
        visible: true,
        locked: false,
        opacity: 1,
        blend_mode: "normal",
        parent_id: null,
        order: 1,
        mask_ids: [],
        image_layer_key: "inpaint",
        metadata: {},
      },
    ],
  };
}

function generation() {
  const selection = createStudioSelection({
    id: "selection:flux",
    pageKey: "memory://flux::0",
    pageIndex: 0,
    points: [[1, 1], [3, 1], [3, 3], [1, 3]],
    width: 4,
    height: 4,
    targetNodeId: "image:inpaint",
  });
  return createFluxGeneration({
    id: "flux-job",
    targetNodeId: "image:inpaint",
    selection,
    prompt: "reconstruir a textura do casaco",
    negativePrompt: "texto, assinatura",
    provider: "local-adapter",
    model: "flux-fill-local",
    cropBbox: [1, 1, 3, 3],
    seed: 100,
    variants: [
      { id: "variant-a", seed: 100, resultPath: "layers/generated/001/variant-a.png" },
      { id: "variant-b", seed: 101, resultPath: "layers/generated/001/variant-b.png" },
    ],
    createdAt: 123,
  });
}

describe("FLUX contract", () => {
  it("removes every possible contribution outside the selection mask", () => {
    const candidate = new Uint8Array([
      255, 0, 0, 255,
      0, 255, 0, 128,
    ]);
    const isolated = isolateFluxVariantPixels(candidate, new Uint8Array([0, 255]));

    expect(Array.from(isolated)).toEqual([
      255, 0, 0, 0,
      0, 255, 0, 128,
    ]);
    expect(Array.from(candidate)).toEqual([
      255, 0, 0, 255,
      0, 255, 0, 128,
    ]);
  });

  it("adds 2-4 generated variants with editable masks and complete provenance", () => {
    const next = applyFluxGenerationToScene(sceneWithTarget(), generation());
    const group = next.nodes.find((node) => node.id === "group:flux-job");
    const first = next.nodes.find((node) => node.id === "generated:flux-job:variant-a");
    const second = next.nodes.find((node) => node.id === "generated:flux-job:variant-b");

    expect(next.roots).toEqual(["image:base", "image:inpaint", "group:flux-job"]);
    expect(group).toMatchObject({ kind: "group", parent_id: null, order: 2 });
    expect(first).toMatchObject({
      kind: "generated",
      parent_id: "group:flux-job",
      visible: true,
      mask_ids: ["mask:flux-job:variant-a"],
      metadata: {
        generator: "flux-fill",
        generation_id: "flux-job",
        provider: "local-adapter",
        model: "flux-fill-local",
        seed: 100,
        prompt: "reconstruir a textura do casaco",
        source_node_id: "image:inpaint",
      },
    });
    expect(second).toMatchObject({ visible: false, mask_ids: ["mask:flux-job:variant-b"] });
    expect(next.nodes.find((node) => node.id === "mask:flux-job:variant-a")?.parent_id).toBe(first?.id);
  });

  it("previews, accepts and rejects variants through reversible scene transforms", () => {
    const generated = applyFluxGenerationToScene(sceneWithTarget(), generation());
    const activated = activateFluxVariant(generated, "flux-job", "variant-b");
    expect(activated.nodes.find((node) => node.id === "generated:flux-job:variant-a")?.visible).toBe(false);
    expect(activated.nodes.find((node) => node.id === "generated:flux-job:variant-b")?.visible).toBe(true);

    const accepted = acceptFluxVariant(activated, "flux-job", "variant-b");
    expect(accepted.nodes.some((node) => node.id === "group:flux-job")).toBe(false);
    expect(accepted.nodes.some((node) => node.id === "generated:flux-job:variant-a")).toBe(false);
    expect(accepted.nodes.find((node) => node.id === "generated:flux-job:variant-b")).toMatchObject({
      parent_id: null,
      visible: true,
      order: 2,
    });
    expect(accepted.roots).toEqual(["image:base", "image:inpaint", "generated:flux-job:variant-b"]);

    const rejected = rejectFluxGeneration(generated, "flux-job");
    expect(rejected).toEqual(sceneWithTarget());

    expect(findFluxGenerationId(generated, "generated:flux-job:variant-b")).toBe("flux-job");
    expect(findFluxGenerationId(generated, "image:inpaint")).toBe("flux-job");
  });

  it("cancels application when the page or scene changes during generation", () => {
    const scene = sceneWithTarget();
    const expected = { pageKey: "memory://flux::0", pageIndex: 0, scene };

    expect(() => assertFluxExecutionContext(expected, {
      pageKey: "memory://flux::0",
      pageIndex: 0,
      scene,
    })).not.toThrow();
    expect(() => assertFluxExecutionContext(expected, {
      pageKey: "memory://flux::1",
      pageIndex: 1,
      scene: sceneWithTarget(),
    })).toThrow("página mudou");
  });
});

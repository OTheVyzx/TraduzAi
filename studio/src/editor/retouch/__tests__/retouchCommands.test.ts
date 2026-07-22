import { describe, expect, it } from "vitest";
import type { StudioScene } from "../../../project/studioProject";
import { createStudioSelection } from "../../selection/selectionModel";
import {
  applyRetouchCommandToScene,
  assertRetouchExecutionContext,
  createRetouchCommand,
  renderRetouchCommandBitmap,
} from "../retouchCommands";

function makeScene(locked = false): StudioScene {
  return {
    version: "1.0",
    roots: ["image:base", "text:a"],
    nodes: [
      {
        id: "image:base",
        kind: "raster",
        name: "Arte base",
        visible: true,
        locked,
        opacity: 1,
        blend_mode: "normal",
        parent_id: null,
        order: 0,
        mask_ids: [],
        image_layer_key: "base",
        metadata: {},
      },
      {
        id: "text:a",
        kind: "text",
        name: "Texto",
        visible: true,
        locked: false,
        opacity: 1,
        blend_mode: "normal",
        parent_id: null,
        order: 1,
        mask_ids: [],
        text_layer_id: "a",
        metadata: {},
      },
    ],
  };
}

function makeSelection() {
  return createStudioSelection({
    id: "selection:1",
    pageKey: "project::0",
    pageIndex: 0,
    points: [[20, 20], [80, 20], [80, 70], [20, 70]],
    width: 100,
    height: 90,
    targetNodeId: "image:base",
  });
}

describe("Studio retouch commands", () => {
  it("requires an explicit source sample for clone and patch", () => {
    expect(() => createRetouchCommand({
      id: "retouch:clone",
      tool: "clone",
      targetNodeId: "image:base",
      selection: makeSelection(),
    })).toThrow(/amostra/i);

    expect(() => createRetouchCommand({
      id: "retouch:patch",
      tool: "patch",
      targetNodeId: "image:base",
      selection: makeSelection(),
    })).toThrow(/amostra/i);
  });

  it("supports automatic healing and clamps brush settings", () => {
    const command = createRetouchCommand({
      id: "retouch:healing",
      tool: "healing",
      targetNodeId: "image:base",
      selection: makeSelection(),
      settings: { opacity: 4, hardness: -1, spacing: 0 },
    });

    expect(command).toMatchObject({
      tool: "healing",
      sampling: { mode: "automatic" },
      settings: { opacity: 1, hardness: 0, spacing: 0.01 },
      status: "pending_render",
    });
  });

  it("records clone as a generated result layer with its own layer mask", () => {
    const original = makeScene();
    const command = createRetouchCommand({
      id: "retouch:clone",
      tool: "clone",
      targetNodeId: "image:base",
      selection: makeSelection(),
      sampling: {
        mode: "sampled",
        sourceNodeId: "image:base",
        sourceOffset: [-24, 12],
        aligned: true,
      },
      resultPath: "retouch/clone-1.png",
    });

    const next = applyRetouchCommandToScene(original, command, {
      outputNodeId: "generated:retouch:clone",
      maskNodeId: "mask:retouch:clone",
    });

    expect(original.nodes).toHaveLength(2);
    expect(next.roots).toEqual(["image:base", "generated:retouch:clone", "text:a"]);
    expect(next.nodes.find((node) => node.id === "image:base")?.mask_ids).toEqual([]);
    expect(next.nodes.find((node) => node.id === "generated:retouch:clone")).toMatchObject({
      kind: "generated",
      parent_id: null,
      mask_ids: ["mask:retouch:clone"],
      metadata: {
        generator: "retouch",
        image_path: "retouch/clone-1.png",
        retouch_command: expect.objectContaining({ tool: "clone", status: "ready" }),
      },
    });
    expect(next.nodes.find((node) => node.id === "mask:retouch:clone")).toMatchObject({
      kind: "mask",
      parent_id: "generated:retouch:clone",
    });
  });

  it("does not retouch a locked or non-raster target", () => {
    const command = createRetouchCommand({
      id: "retouch:healing",
      tool: "healing",
      targetNodeId: "image:base",
      selection: makeSelection(),
    });

    expect(() => applyRetouchCommandToScene(makeScene(true), command)).toThrow(/bloqueada/i);
    expect(() => applyRetouchCommandToScene(makeScene(), { ...command, targetNodeId: "text:a" })).toThrow(/raster/i);

    const cloneFromText = createRetouchCommand({
      id: "retouch:clone-text",
      tool: "clone",
      targetNodeId: "image:base",
      selection: makeSelection(),
      sampling: {
        mode: "sampled",
        sourceNodeId: "text:a",
        sourceOffset: [10, 0],
        aligned: true,
      },
    });
    expect(() => applyRetouchCommandToScene(makeScene(), cloneFromText)).toThrow(/amostra.*raster/i);
  });

  it("renders clone pixels from the sampled offset into a new bitmap", async () => {
    const operations: string[] = [];
    const canvas = {
      width: 100,
      height: 90,
      getContext: () => ({
        filter: "none",
        globalAlpha: 1,
        clearRect: () => operations.push("clear"),
        drawImage: (_image: unknown, x: number, y: number, width: number, height: number) => {
          operations.push(`draw:${x},${y},${width},${height}`);
        },
        save: () => operations.push("save"),
        restore: () => operations.push("restore"),
      }),
      toDataURL: () => "data:image/png;base64,clone",
    };
    const command = createRetouchCommand({
      id: "retouch:render-clone",
      tool: "clone",
      targetNodeId: "image:base",
      selection: makeSelection(),
      sampling: {
        mode: "sampled",
        sourceNodeId: "image:base",
        sourceOffset: [12, -8],
        aligned: true,
      },
    });

    const result = await renderRetouchCommandBitmap(command, {
      width: 100,
      height: 90,
      createCanvas: () => canvas,
      loadNodeImage: async (nodeId) => ({ image: { nodeId } }),
    });

    expect(result).toBe("data:image/png;base64,clone");
    expect(operations).toContain("draw:-12,8,100,90");
  });

  it("renders automatic healing from the target with a softened pass", async () => {
    const operations: string[] = [];
    const context = {
      filter: "none",
      globalAlpha: 1,
      clearRect: () => undefined,
      drawImage: () => operations.push(`draw:${context.filter}:alpha=${context.globalAlpha}`),
      save: () => undefined,
      restore: () => undefined,
    };
    const command = createRetouchCommand({
      id: "retouch:render-healing",
      tool: "healing",
      targetNodeId: "image:base",
      selection: makeSelection(),
      settings: { hardness: 0.25, opacity: 0.8 },
    });

    await renderRetouchCommandBitmap(command, {
      width: 100,
      height: 90,
      createCanvas: () => ({
        width: 100,
        height: 90,
        getContext: () => context,
        toDataURL: () => "data:image/png;base64,healing",
      }),
      loadNodeImage: async () => ({ image: {} }),
    });

    expect(operations.some((operation) => operation.includes("blur("))).toBe(true);
    expect(operations.some((operation) => operation.includes("alpha=0.8"))).toBe(true);
  });

  it("cancels delayed retouch work after a page or scene switch", () => {
    const scene = makeScene();
    const token = { pageKey: "project::0", pageIndex: 0, scene };

    expect(() => assertRetouchExecutionContext(token, {
      pageKey: "project::0",
      pageIndex: 0,
      scene,
    })).not.toThrow();
    expect(() => assertRetouchExecutionContext(token, {
      pageKey: "project::1",
      pageIndex: 1,
      scene: makeScene(),
    })).toThrow(/página mudou/i);
    expect(() => assertRetouchExecutionContext(token, {
      pageKey: "project::0",
      pageIndex: 0,
      scene: makeScene(),
    })).toThrow(/página mudou/i);
  });
});

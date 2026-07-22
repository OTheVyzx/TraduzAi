import { describe, expect, it, vi } from "vitest";
import type { StudioPage, StudioScene } from "../../../project/studioProject";
import {
  composeStudioSceneBitmap,
  composeStudioSceneLayerBitmaps,
  resolveStudioSceneRenderLayers,
  resolveStudioSceneVisualOrder,
} from "../studioSceneCompositor";

function makePage(): StudioPage {
  return {
    numero: 1,
    arquivo_original: "base.png",
    arquivo_traduzido: "rendered.png",
    image_layers: {
      base: { key: "base", path: "base.png", visible: true, locked: false },
      rendered: { key: "rendered", path: "rendered.png", visible: true, locked: false },
    },
    text_layers: [{
      id: "a",
      kind: "text",
      original: "A",
      translated: "B",
      traduzido: "B",
      bbox: [1, 1, 4, 4],
      style: {},
      estilo: {},
      visible: true,
      locked: false,
      order: 1,
    }],
    textos: [],
    studio_scene: makeScene(),
  };
}

function makeScene(): StudioScene {
  return {
    version: "1.0",
    roots: ["group:art", "text:a", "image:rendered", "generated:clone"],
    nodes: [
      {
        id: "group:art",
        kind: "group",
        name: "Arte",
        visible: true,
        locked: false,
        opacity: 0.5,
        blend_mode: "normal",
        parent_id: null,
        order: 0,
        mask_ids: [],
        metadata: {},
      },
      {
        id: "image:base",
        kind: "raster",
        name: "Original",
        visible: true,
        locked: false,
        opacity: 0.8,
        blend_mode: "multiply",
        parent_id: "group:art",
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
      {
        id: "image:rendered",
        kind: "raster",
        name: "Preview legado",
        visible: true,
        locked: false,
        opacity: 1,
        blend_mode: "normal",
        parent_id: null,
        order: 1,
        mask_ids: [],
        image_layer_key: "rendered",
        metadata: { projected_from: "image_layers" },
      },
      {
        id: "generated:clone",
        kind: "generated",
        name: "Clone",
        visible: true,
        locked: false,
        opacity: 0.75,
        blend_mode: "screen",
        parent_id: null,
        order: 2,
        mask_ids: ["mask:clone"],
        metadata: { image_path: "clone.png" },
      },
      {
        id: "mask:clone",
        kind: "mask",
        name: "Mascara do clone",
        visible: true,
        locked: false,
        opacity: 0.6,
        blend_mode: "normal",
        parent_id: "generated:clone",
        order: 0,
        mask_ids: [],
        metadata: {
          selection: {
            id: "selection:clone",
            pageKey: "project::0",
            pageIndex: 0,
            points: [[2, 2], [8, 2], [8, 8], [2, 8]],
            regions: [{ operation: "add", points: [[2, 2], [8, 2], [8, 8], [2, 8]] }],
            bbox: [2, 2, 8, 8],
            width: 10,
            height: 10,
            feather: 2,
            expansion: 1,
            targetNodeId: "generated:clone",
          },
        },
      },
    ],
  };
}

class FakeContext {
  globalAlpha = 1;
  globalCompositeOperation = "source-over";
  fillStyle = "#000000";
  readonly operations: string[] = [];

  clearRect() {}
  fillRect() {
    this.operations.push(`fill:${this.fillStyle}`);
  }
  drawImage(image: { id?: string }) {
    this.operations.push(`draw:${image.id ?? "canvas"}:alpha=${this.globalAlpha}:blend=${this.globalCompositeOperation}`);
  }
  save() {
    this.operations.push("save");
  }
  restore() {
    this.operations.push("restore");
  }
}

class FakeCanvas {
  readonly id: string;
  readonly context = new FakeContext();

  constructor(public width: number, public height: number, index: number) {
    this.id = `canvas-${index}`;
  }

  getContext() {
    return this.context;
  }

  toDataURL() {
    return `data:image/png;fake,${this.id}`;
  }
}

describe("Studio scene compositor", () => {
  it("resolves generated layers and masks in visual order with inherited opacity", () => {
    const layers = resolveStudioSceneRenderLayers(makePage(), makeScene());

    expect(layers.map((layer) => layer.nodeId)).toEqual(["image:base", "generated:clone"]);
    expect(layers[0]).toMatchObject({ sourcePath: "base.png", opacity: 0.4, blendMode: "multiply" });
    expect(layers[1]).toMatchObject({ sourcePath: "clone.png", opacity: 0.75, blendMode: "screen" });
    expect(layers[1].masks).toHaveLength(1);
    expect(layers[1].masks[0]).toMatchObject({ nodeId: "mask:clone", opacity: 0.6 });
    expect(layers[1].masks[0].selection).toMatchObject({ feather: 2, expansion: 1 });
  });

  it("keeps bitmap and text nodes interleaved in the professional scene order", () => {
    expect(resolveStudioSceneVisualOrder(makePage(), makeScene())).toEqual([
      { kind: "bitmap", nodeId: "image:base" },
      { kind: "text", nodeId: "text:a", textLayerId: "a" },
      { kind: "bitmap", nodeId: "generated:clone" },
    ]);
  });

  it("applies the layer mask before compositing opacity and blend mode", async () => {
    const canvases: FakeCanvas[] = [];
    const createCanvas = (width: number, height: number) => {
      const canvas = new FakeCanvas(width, height, canvases.length + 1);
      canvases.push(canvas);
      return canvas;
    };
    const loadImage = vi.fn(async (path: string) => ({
      image: { id: path },
      width: 10,
      height: 10,
    }));
    const rasterizeSelection = vi.fn(() => ({ id: "selection-mask" }));

    const output = await composeStudioSceneBitmap({
      page: makePage(),
      scene: makeScene(),
      createCanvas,
      loadImage,
      rasterizeSelection,
    });

    expect(loadImage).toHaveBeenCalledTimes(2);
    expect(rasterizeSelection).toHaveBeenCalledTimes(1);
    expect(canvases.some((canvas) => canvas.context.operations.includes(
      "draw:selection-mask:alpha=0.6:blend=destination-in",
    ))).toBe(true);
    expect((output as FakeCanvas).context.operations).toContain("draw:canvas-2:alpha=0.75:blend=screen");
  });

  it("renders one masked bitmap per raster scene node for stage interleaving", async () => {
    const canvases: FakeCanvas[] = [];
    const rendered = await composeStudioSceneLayerBitmaps({
      page: makePage(),
      scene: makeScene(),
      createCanvas: (width, height) => {
        const canvas = new FakeCanvas(width, height, canvases.length + 1);
        canvases.push(canvas);
        return canvas;
      },
      loadImage: async (path) => ({ image: { id: path }, width: 10, height: 10 }),
      rasterizeSelection: () => ({ id: "selection-mask" }),
    });

    expect(rendered.layers.map((layer) => layer.nodeId)).toEqual(["image:base", "generated:clone"]);
    expect(rendered.layers[1]).toMatchObject({ opacity: 0.75, blendMode: "screen" });
  });
});

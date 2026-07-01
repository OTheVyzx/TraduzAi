import { describe, expect, it } from "vitest";
import { LayeredBitmapCanvas, bitmapStrokePasses, type Canvas2DLike, type CanvasLike } from "../../../../src/editor-shared";

class FakeContext implements Canvas2DLike {
  globalAlpha = 1;
  globalCompositeOperation = "source-over";
  lineCap: CanvasLineCap = "butt";
  lineJoin: CanvasLineJoin = "miter";
  lineWidth = 1;
  strokeStyle: string | CanvasGradient | CanvasPattern = "#000000";
  readonly ops: string[] = [];

  clearRect(x: number, y: number, width: number, height: number): void {
    this.ops.push(`clear:${x},${y},${width},${height}`);
  }
  drawImage(_image: CanvasImageSource, dx: number, dy: number, width: number, height: number): void {
    this.ops.push(`draw:${dx},${dy},${width},${height}:alpha=${this.globalAlpha}`);
  }
  beginPath(): void {
    this.ops.push("begin");
  }
  moveTo(x: number, y: number): void {
    this.ops.push(`move:${x},${y}`);
  }
  lineTo(x: number, y: number): void {
    this.ops.push(`line:${x},${y}`);
  }
  stroke(): void {
    this.ops.push(`stroke:w=${this.lineWidth}:a=${this.globalAlpha}:op=${this.globalCompositeOperation}:style=${String(this.strokeStyle)}`);
  }
  save(): void {
    this.ops.push("save");
  }
  restore(): void {
    this.ops.push("restore");
  }
}

class FakeCanvas implements CanvasLike {
  readonly ctx = new FakeContext();

  constructor(public width: number, public height: number, private readonly id: number) {}

  getContext(type: "2d") {
    return type === "2d" ? this.ctx : null;
  }

  toDataURL(type = "image/png") {
    return `data:${type};fake,${this.id}`;
  }
}

function fakeCanvasFactory() {
  const canvases: FakeCanvas[] = [];
  return {
    canvases,
    createCanvas: (width: number, height: number) => {
      const canvas = new FakeCanvas(width, height, canvases.length + 1);
      canvases.push(canvas);
      return canvas;
    },
  };
}

describe("LayeredBitmapCanvas", () => {
  it("orders layers using the editor bitmap stack", () => {
    const factory = fakeCanvasFactory();
    const bitmap = new LayeredBitmapCanvas({ width: 100, height: 200, createCanvas: factory.createCanvas });

    bitmap.ensureLayer("mask");
    bitmap.ensureLayer("base");
    bitmap.ensureLayer("brush");

    expect(bitmap.orderedLayers().map((layer) => layer.key)).toEqual(["base", "brush", "mask"]);
  });

  it("draws soft brush strokes into the selected layer", () => {
    const factory = fakeCanvasFactory();
    const bitmap = new LayeredBitmapCanvas({ width: 100, height: 100, createCanvas: factory.createCanvas });

    const dataUrl = bitmap.drawStroke({
      layerKey: "brush",
      stroke: [[10, 10], [30, 30]],
      brushSize: 20,
      color: "#ff0000",
      opacity: 0.5,
      hardness: 0.5,
    });

    const ops = factory.canvases[0].ctx.ops;
    expect(dataUrl).toBe("data:image/png;fake,1");
    expect(ops.filter((op) => op.startsWith("stroke:"))).toHaveLength(3);
    expect(ops.some((op) => op.includes("style=#ff0000"))).toBe(true);
  });

  it("uses destination-out for erase strokes", () => {
    const factory = fakeCanvasFactory();
    const bitmap = new LayeredBitmapCanvas({ width: 100, height: 100, createCanvas: factory.createCanvas });

    bitmap.drawStroke({
      layerKey: "brush",
      stroke: [[10, 10]],
      brushSize: 8,
      erase: true,
    });

    expect(factory.canvases[0].ctx.ops.some((op) => op.includes("op=destination-out"))).toBe(true);
  });

  it("composites only visible layers with configured opacity", () => {
    const factory = fakeCanvasFactory();
    const bitmap = new LayeredBitmapCanvas({ width: 64, height: 64, createCanvas: factory.createCanvas });
    bitmap.ensureLayer("base");
    bitmap.ensureLayer("brush", { opacity: 0.25 });
    bitmap.ensureLayer("mask", { visible: false });

    const output = bitmap.compositeVisibleLayers() as FakeCanvas;

    expect(output.ctx.ops.filter((op) => op.startsWith("draw:"))).toEqual([
      "draw:0,0,64,64:alpha=1",
      "draw:0,0,64,64:alpha=0.25",
    ]);
  });

  it("keeps hard strokes to a single pass", () => {
    expect(bitmapStrokePasses({ brushSize: 10, opacity: 0.75, hardness: 1 })).toEqual([{ width: 10, alpha: 0.75 }]);
  });
});

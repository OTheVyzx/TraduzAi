import { describe, expect, it } from "vitest";
import {
  bitmapTargetForEditorTool,
  pointFromStageClientRect,
  shouldAppendStrokePoint,
  strokeDirtyBbox,
} from "../editorStroke";

describe("editor stroke helpers", () => {
  it("converts client coordinates into clamped rounded image points", () => {
    const rect = { left: 10, top: 20, width: 200, height: 100 };

    expect(
      pointFromStageClientRect({
        clientX: 60.4,
        clientY: 45.6,
        rect,
        imageWidth: 1000,
        imageHeight: 500,
      }),
    ).toEqual({ x: 252, y: 128 });

    expect(
      pointFromStageClientRect({
        clientX: -20,
        clientY: 200,
        rect,
        imageWidth: 1000,
        imageHeight: 500,
      }),
    ).toEqual({ x: 0, y: 500 });
  });

  it("returns null for invalid coordinate inputs", () => {
    const rect = { left: 0, top: 0, width: 0, height: 100 };

    expect(pointFromStageClientRect({ clientX: 1, clientY: 1, rect, imageWidth: 100, imageHeight: 100 })).toBeNull();
    expect(
      pointFromStageClientRect({
        clientX: 1,
        clientY: 1,
        rect: { ...rect, width: 100 },
        imageWidth: 0,
        imageHeight: 100,
      }),
    ).toBeNull();
  });

  it("returns null instead of NaN points for non-finite coordinate inputs", () => {
    const rect = { left: 0, top: 0, width: 100, height: 100 };

    expect(
      pointFromStageClientRect({
        clientX: Number.NaN,
        clientY: 1,
        rect,
        imageWidth: 100,
        imageHeight: 100,
      }),
    ).toBeNull();
    expect(
      pointFromStageClientRect({
        clientX: 1,
        clientY: Number.POSITIVE_INFINITY,
        rect,
        imageWidth: 100,
        imageHeight: 100,
      }),
    ).toBeNull();
    expect(
      pointFromStageClientRect({
        clientX: 1,
        clientY: 1,
        rect: { ...rect, left: Number.NaN },
        imageWidth: 100,
        imageHeight: 100,
      }),
    ).toBeNull();
    expect(
      pointFromStageClientRect({
        clientX: 1,
        clientY: 1,
        rect,
        imageWidth: Number.POSITIVE_INFINITY,
        imageHeight: 100,
      }),
    ).toBeNull();
  });

  it("filters duplicate stroke points", () => {
    expect(shouldAppendStrokePoint(undefined, { x: 8, y: 9 })).toBe(true);
    expect(shouldAppendStrokePoint([8, 9], { x: 8, y: 9 })).toBe(false);
    expect(shouldAppendStrokePoint([8, 9], { x: 9, y: 9 })).toBe(true);
  });

  it("computes dirty bbox with brush padding and image clamping", () => {
    expect(
      strokeDirtyBbox({
        stroke: [
          [2, 3],
          [98, 79],
        ],
        brushSize: 10,
        width: 100,
        height: 80,
      }),
    ).toEqual([0, 0, 100, 80]);

    expect(
      strokeDirtyBbox({
        stroke: [[50, 40]],
        brushSize: 1,
        width: 100,
        height: 80,
      }),
    ).toEqual([47, 37, 53, 43]);
  });

  it("returns null for empty strokes or invalid bitmap dimensions", () => {
    expect(strokeDirtyBbox({ stroke: [], brushSize: 10, width: 100, height: 100 })).toBeNull();
    expect(strokeDirtyBbox({ stroke: [[1, 1]], brushSize: 10, width: 0, height: 100 })).toBeNull();
    expect(strokeDirtyBbox({ stroke: [[1, 1]], brushSize: 10, width: 100, height: 0 })).toBeNull();
  });

  it("does not emit NaN bbox values for non-finite stroke inputs", () => {
    expect(
      strokeDirtyBbox({
        stroke: [
          [Number.NaN, 3],
          [50, 40],
          [Number.POSITIVE_INFINITY, 79],
        ],
        brushSize: 10,
        width: 100,
        height: 80,
      }),
    ).toEqual([43, 33, 57, 47]);

    expect(strokeDirtyBbox({ stroke: [[50, 40]], brushSize: Number.NaN, width: 100, height: 80 })).toBeNull();
    expect(strokeDirtyBbox({ stroke: [[Number.NaN, 40]], brushSize: 10, width: 100, height: 80 })).toBeNull();
    expect(strokeDirtyBbox({ stroke: [[50, 40]], brushSize: 10, width: Number.POSITIVE_INFINITY, height: 80 })).toBeNull();
  });

  it("maps editor tools to bitmap targets", () => {
    expect(bitmapTargetForEditorTool("brush", null, "mask")).toBe("brush");
    expect(bitmapTargetForEditorTool("repairBrush", null, "brush")).toBe("recovery");
    expect(bitmapTargetForEditorTool("reinpaintBrush", null, "brush")).toBe("reinpaint");
    expect(bitmapTargetForEditorTool("mask", null, "brush")).toBe("mask");
    expect(bitmapTargetForEditorTool("eraser", "mask", "brush")).toBe("mask");
    expect(bitmapTargetForEditorTool("eraser", "mask", "recovery")).toBe("mask");
    expect(bitmapTargetForEditorTool("eraser", "brush", "mask")).toBe("brush");
    expect(bitmapTargetForEditorTool("eraser", null, "mask")).toBe("mask");
    expect(bitmapTargetForEditorTool("eraser", null, "recovery")).toBe("brush");
    expect(bitmapTargetForEditorTool("eraser", "recovery", "mask")).toBe("brush");
    expect(bitmapTargetForEditorTool("eraser", null)).toBe("brush");
    expect(bitmapTargetForEditorTool("select", null, "brush")).toBeNull();
    expect(bitmapTargetForEditorTool("block", null, "brush")).toBeNull();
  });
});

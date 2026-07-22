import { describe, expect, it } from "vitest";
import {
  editorToolsForMode,
  layerProcessingActionsForMode,
} from "../../../../src/components/editor/editorMode";

describe("Studio editor surface", () => {
  it("removes automatic tools from the Studio toolbar", () => {
    const tools = editorToolsForMode("studio");

    expect(tools).not.toContain("process");
    expect(tools).not.toContain("repairBrush");
    expect(tools).not.toContain("reinpaintBrush");
    expect(tools).toContain("brush");
    expect(tools).toContain("mask");
  });

  it("removes OCR, translation and automatic cleaning from Studio layer rows", () => {
    const actions = layerProcessingActionsForMode("studio");

    expect(actions).toEqual([]);
    expect(layerProcessingActionsForMode("traduzai")).toEqual(["ocr", "translate", "inpaint"]);
  });
});

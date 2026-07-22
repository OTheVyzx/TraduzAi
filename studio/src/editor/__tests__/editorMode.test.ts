import { describe, expect, it } from "vitest";
import {
  isEditorToolVisible,
  resolveEditorCapabilities,
} from "../../../../src/components/editor/editorMode";

describe("Studio editor mode", () => {
  it("keeps automatic processing available in the TraduzAI editor", () => {
    const capabilities = resolveEditorCapabilities("traduzai");

    expect(capabilities.showPipelineActions).toBe(true);
    expect(capabilities.showSourceLanguage).toBe(true);
    expect(capabilities.showBlockProcessingActions).toBe(true);
    expect(isEditorToolVisible("traduzai", "process")).toBe(true);
  });

  it("exposes only editorial tools in Studio", () => {
    const capabilities = resolveEditorCapabilities("studio");

    expect(capabilities.showPipelineActions).toBe(false);
    expect(capabilities.showSourceLanguage).toBe(false);
    expect(capabilities.showBlockProcessingActions).toBe(false);
    expect(isEditorToolVisible("studio", "repairBrush")).toBe(false);
    expect(isEditorToolVisible("studio", "reinpaintBrush")).toBe(false);
    expect(isEditorToolVisible("studio", "process")).toBe(false);
    expect(isEditorToolVisible("studio", "select")).toBe(true);
    expect(isEditorToolVisible("studio", "block")).toBe(true);
    expect(isEditorToolVisible("studio", "brush")).toBe(true);
    expect(isEditorToolVisible("studio", "eraser")).toBe(true);
    expect(isEditorToolVisible("studio", "mask")).toBe(true);
  });
});


import { afterEach, describe, expect, it, vi } from "vitest";
import {
  buildReorderLayersCommand,
  buildToggleLockCommand,
  buildToggleVisibilityCommand,
  nextEditorCommandId,
} from "../editorOps";

const uuid = (suffix: string) => `00000000-0000-4000-8000-${suffix.padStart(12, "0")}` as ReturnType<Crypto["randomUUID"]>;

afterEach(() => {
  vi.restoreAllMocks();
});

describe("editorOps", () => {
  it("creates readable command ids with the provided prefix", () => {
    vi.spyOn(crypto, "randomUUID").mockReturnValue(uuid("1"));

    expect(nextEditorCommandId("visible")).toBe("visible-00000000-0000-4000-8000-000000000001");
  });

  it("builds toggle visibility commands with the expected payload", () => {
    vi.spyOn(crypto, "randomUUID").mockReturnValue(uuid("2"));
    vi.spyOn(Date, "now").mockReturnValue(123);

    expect(
      buildToggleVisibilityCommand({
        pageKey: "page-1",
        layerId: "layer-a",
        before: true,
        after: false,
      }),
    ).toEqual({
      commandId: "toggle-visibility-00000000-0000-4000-8000-000000000002",
      pageKey: "page-1",
      createdAt: 123,
      type: "toggle-visibility",
      layerId: "layer-a",
      before: true,
      after: false,
    });
  });

  it("builds toggle lock commands with the expected payload", () => {
    vi.spyOn(crypto, "randomUUID").mockReturnValue(uuid("3"));
    vi.spyOn(Date, "now").mockReturnValue(456);

    expect(
      buildToggleLockCommand({
        pageKey: "page-2",
        layerId: "layer-b",
        before: false,
        after: true,
      }),
    ).toEqual({
      commandId: "toggle-lock-00000000-0000-4000-8000-000000000003",
      pageKey: "page-2",
      createdAt: 456,
      type: "toggle-lock",
      layerId: "layer-b",
      before: false,
      after: true,
    });
  });

  it("defensively copies reorder command layer ids", () => {
    vi.spyOn(crypto, "randomUUID").mockReturnValue(uuid("4"));
    vi.spyOn(Date, "now").mockReturnValue(789);
    const before = ["layer-a", "layer-b"];
    const after = ["layer-b", "layer-a"];

    const command = buildReorderLayersCommand({
      pageKey: "page-3",
      before,
      after,
    });
    before.push("layer-c");
    after.pop();

    expect(command.type).toBe("reorder-layers");
    if (command.type !== "reorder-layers") {
      throw new Error("Expected a reorder-layers command");
    }
    expect(command).toEqual({
      commandId: "reorder-layers-00000000-0000-4000-8000-000000000004",
      pageKey: "page-3",
      createdAt: 789,
      type: "reorder-layers",
      before: ["layer-a", "layer-b"],
      after: ["layer-b", "layer-a"],
    });
    expect(command.before).not.toBe(before);
    expect(command.after).not.toBe(after);
  });
});

import { describe, expect, it } from "vitest";
import {
  DEFAULT_LIBRARY_PANES,
  parseLibraryPaneLayout,
  resizeLibraryPane,
} from "../libraryPaneLayout";

describe("library pane layout", () => {
  it("resizes either divider while preserving a usable center pane", () => {
    expect(resizeLibraryPane(DEFAULT_LIBRARY_PANES, "left", 80, 1200)).toEqual({ left: 372, right: 300 });
    expect(resizeLibraryPane(DEFAULT_LIBRARY_PANES, "right", 60, 1200)).toEqual({ left: 292, right: 240 });
    expect(resizeLibraryPane(DEFAULT_LIBRARY_PANES, "left", 900, 980)).toEqual({ left: 260, right: 300 });
  });

  it("clamps stored values and rejects malformed persistence", () => {
    expect(parseLibraryPaneLayout('{"left":999,"right":12}', 1400)).toEqual({ left: 480, right: 240 });
    expect(parseLibraryPaneLayout("not-json", 1400)).toEqual(DEFAULT_LIBRARY_PANES);
    expect(parseLibraryPaneLayout(null, 1400)).toEqual(DEFAULT_LIBRARY_PANES);
  });
});

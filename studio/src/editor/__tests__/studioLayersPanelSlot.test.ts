import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { EditorLayersPanelSlot } from "../../../../src/components/editor/EditorLayersPanelSlot";

describe("EditorLayersPanelSlot", () => {
  it("renders a Studio override without changing the default editor panel contract", () => {
    const html = renderToStaticMarkup(
      createElement(EditorLayersPanelSlot, {
        mode: "studio",
        panel: createElement("aside", { "data-testid": "studio-tree" }, "Árvore profissional"),
      }),
    );

    expect(html).toContain("studio-tree");
    expect(html).toContain("Árvore profissional");
  });
});

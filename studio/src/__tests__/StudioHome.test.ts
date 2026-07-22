import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { StudioHome } from "../App";

describe("StudioHome", () => {
  it("opens an existing TraduzAI project instead of starting a pipeline project", () => {
    const html = renderToStaticMarkup(createElement(StudioHome, {
      error: null,
      recents: [],
      onOpenProject: () => undefined,
      onOpenRecent: () => undefined,
    }));

    expect(html).toContain("Abrir projeto TraduzAI");
    expect(html).toContain("projeto já traduzido");
    expect(html).not.toContain("Novo projeto");
  });
});

import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { StudioScene } from "../../../project/studioProject";
import { useStudioSceneStore } from "../../../store/studioSceneStore";
import { StudioLayersTree } from "../StudioLayersTree";

const scene: StudioScene = {
  version: "1.0",
  roots: ["image:base", "text:a"],
  nodes: [
    {
      id: "image:base",
      kind: "raster",
      name: "Original",
      visible: true,
      locked: true,
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
      name: "Olá, mundo",
      visible: true,
      locked: false,
      opacity: 0.8,
      blend_mode: "multiply",
      parent_id: null,
      order: 1,
      mask_ids: [],
      text_layer_id: "a",
      metadata: {},
    },
  ],
};

describe("StudioLayersTree", () => {
  it("renders the professional layer tree and selected-node properties", () => {
    useStudioSceneStore.getState().hydrate("page:1", scene, async () => undefined);
    useStudioSceneStore.getState().selectNode("text:a");

    const html = renderToStaticMarkup(createElement(StudioLayersTree));

    expect(html).toContain("Camadas");
    expect(html).toContain("Original");
    expect(html).toContain("Olá, mundo");
    expect(html).toContain("Opacidade");
    expect(html).toContain("Modo de mesclagem");
    expect(html).toContain("Criar grupo com a seleção");
  });
});

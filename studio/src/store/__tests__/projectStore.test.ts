import { beforeEach, describe, expect, it } from "vitest";
import { useStudioProjectStore } from "../projectStore";

describe("useStudioProjectStore", () => {
  beforeEach(() => {
    useStudioProjectStore.setState({
      project: null,
      projectPath: null,
      currentPageIndex: 0,
      lastImport: null,
      error: null,
    });
  });

  it("imports project json into the configured backend", async () => {
    await useStudioProjectStore.getState().importProjectJson(
      JSON.stringify({
        versao: "1.0",
        paginas: [{ numero: 1, textos: [{ id: "a", bbox: [0, 0, 1, 1], texto: "A", traduzido: "B" }] }],
      }),
      "memory://store-test",
    );

    const state = useStudioProjectStore.getState();
    expect(state.project?.paginas).toHaveLength(1);
    expect(state.lastImport?.kind).toBe("traduzai_v1");

    await state.loadProject("memory://store-test");
    expect(useStudioProjectStore.getState().project?.paginas[0].text_layers[0].translated).toBe("B");
  });

  it("patches current text layers through the compatibility backend", async () => {
    await useStudioProjectStore.getState().importProjectJson(
      JSON.stringify({
        versao: "1.0",
        paginas: [{ numero: 1, textos: [{ id: "a", bbox: [0, 0, 1, 1], texto: "A", traduzido: "B" }] }],
      }),
      "memory://store-patch-test",
    );

    await useStudioProjectStore.getState().patchCurrentTextLayer("a", { translated: "C", traduzido: "C" });

    const layer = useStudioProjectStore.getState().project?.paginas[0].text_layers[0];
    expect(layer?.translated).toBe("C");
    expect(layer?.traduzido).toBe("C");
  });

  it("toggles text and image layer visibility through the compatibility backend", async () => {
    await useStudioProjectStore.getState().importProjectJson(
      JSON.stringify({
        versao: "1.0",
        paginas: [{ numero: 1, arquivo_original: "base.png", textos: [{ id: "a", bbox: [0, 0, 1, 1] }] }],
      }),
      "memory://store-visibility-test",
    );

    await useStudioProjectStore.getState().setCurrentTextLayerVisibility("a", false);
    await useStudioProjectStore.getState().setCurrentImageLayerVisibility("base", false);

    const page = useStudioProjectStore.getState().project?.paginas[0];
    expect(page?.text_layers[0].visible).toBe(false);
    expect(page?.image_layers.base?.visible).toBe(false);
  });
});

import { describe, expect, it } from "vitest";
import { importStudioProject } from "../../project/adapters";
import type { StudioScene } from "../../project/studioProject";
import {
  createStudioSceneStore,
  projectStudioSceneToPage,
} from "../studioSceneStore";

function makeScene(): StudioScene {
  return {
    version: "1.0",
    roots: ["image:base", "text:a", "text:b"],
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
        metadata: { projected_from: "image_layers" },
      },
      {
        id: "text:a",
        kind: "text",
        name: "Olá",
        visible: true,
        locked: false,
        opacity: 1,
        blend_mode: "normal",
        parent_id: null,
        order: 1,
        mask_ids: [],
        text_layer_id: "a",
        metadata: { projected_from: "text_layers" },
      },
      {
        id: "text:b",
        kind: "text",
        name: "Mundo",
        visible: true,
        locked: false,
        opacity: 1,
        blend_mode: "normal",
        parent_id: null,
        order: 2,
        mask_ids: [],
        text_layer_id: "b",
        metadata: { projected_from: "text_layers" },
      },
    ],
  };
}

describe("studioSceneStore", () => {
  it("commits node properties transactionally and supports undo and redo", async () => {
    const persisted: StudioScene[] = [];
    const store = createStudioSceneStore();
    store.getState().hydrate("page:1", makeScene(), async (scene) => {
      persisted.push(structuredClone(scene));
    });

    await store.getState().patchNode("text:a", {
      visible: false,
      locked: true,
      opacity: 0.35,
      blend_mode: "multiply",
    });

    expect(store.getState().scene?.nodes.find((node) => node.id === "text:a")).toMatchObject({
      visible: false,
      locked: true,
      opacity: 0.35,
      blend_mode: "multiply",
    });
    expect(store.getState().historyIndex).toBe(1);
    expect(persisted).toHaveLength(1);

    await store.getState().undo();
    expect(store.getState().scene?.nodes.find((node) => node.id === "text:a")).toMatchObject({
      visible: true,
      locked: false,
      opacity: 1,
      blend_mode: "normal",
    });

    await store.getState().redo();
    expect(store.getState().scene?.nodes.find((node) => node.id === "text:a")?.opacity).toBe(0.35);
    expect(persisted).toHaveLength(3);
  });

  it("commits an external non-destructive scene command through the same history", async () => {
    const store = createStudioSceneStore();
    store.getState().hydrate("page:1", makeScene(), async () => undefined);

    await store.getState().executeSceneCommand("Adicionar máscara", (scene) => ({
      ...scene,
      nodes: scene.nodes.map((node) => node.id === "image:base"
        ? { ...node, mask_ids: ["mask:one"] }
        : node),
    }));

    expect(store.getState().scene?.nodes.find((node) => node.id === "image:base")?.mask_ids).toEqual(["mask:one"]);
    expect(store.getState().history[0]?.label).toBe("Adicionar máscara");

    await store.getState().undo();
    expect(store.getState().scene?.nodes.find((node) => node.id === "image:base")?.mask_ids).toEqual([]);
  });

  it("groups selected siblings and reorders the group as one node", async () => {
    const store = createStudioSceneStore();
    store.getState().hydrate("page:1", makeScene(), async () => undefined);
    store.getState().selectNode("text:a");
    store.getState().selectNode("text:b", true);

    await store.getState().groupSelected("Lettering", "group:lettering");

    let scene = store.getState().scene!;
    expect(store.getState().selectedNodeIds).toEqual(["group:lettering"]);
    expect(scene.roots).toEqual(["image:base", "group:lettering"]);
    expect(scene.nodes.find((node) => node.id === "group:lettering")).toMatchObject({
      kind: "group",
      name: "Lettering",
      parent_id: null,
    });
    expect(scene.nodes.filter((node) => node.parent_id === "group:lettering").map((node) => node.id)).toEqual([
      "text:a",
      "text:b",
    ]);

    await store.getState().moveNodeBefore("group:lettering", "image:base");
    scene = store.getState().scene!;
    expect(scene.roots).toEqual(["group:lettering", "image:base"]);

    await store.getState().undo();
    expect(store.getState().scene?.roots).toEqual(["image:base", "group:lettering"]);
    await store.getState().undo();
    expect(store.getState().scene?.roots).toEqual(["image:base", "text:a", "text:b"]);
  });

  it("rolls back the complete command when persistence fails", async () => {
    const store = createStudioSceneStore();
    store.getState().hydrate("page:1", makeScene(), async () => {
      throw new Error("disco indisponível");
    });

    await expect(store.getState().patchNode("text:a", { opacity: 0.2 })).rejects.toThrow("disco indisponível");

    expect(store.getState().scene?.nodes.find((node) => node.id === "text:a")?.opacity).toBe(1);
    expect(store.getState().history).toHaveLength(0);
    expect(store.getState().error).toBe("disco indisponível");
  });

  it("does not overwrite a newly hydrated page when an older save finishes later", async () => {
    let releaseSave!: () => void;
    const pendingSave = new Promise<void>((resolve) => {
      releaseSave = resolve;
    });
    const store = createStudioSceneStore();
    store.getState().hydrate("page:1", makeScene(), async () => pendingSave);

    const oldCommand = store.getState().patchNode("text:a", { opacity: 0.2 });
    const pageTwo = makeScene();
    pageTwo.nodes = pageTwo.nodes.map((node) => node.id === "text:a" ? { ...node, name: "Página 2" } : node);
    store.getState().hydrate("page:2", pageTwo, async () => undefined);
    releaseSave();
    await oldCommand;

    expect(store.getState().pageKey).toBe("page:2");
    expect(store.getState().scene?.nodes.find((node) => node.id === "text:a")).toMatchObject({
      name: "Página 2",
      opacity: 1,
    });
    expect(store.getState().history).toHaveLength(0);
  });

  it.each(["undo", "redo"] as const)(
    "does not overwrite a newly hydrated page when %s finishes later",
    async (operation) => {
      let deferred = false;
      let releaseSave!: () => void;
      const pendingSave = new Promise<void>((resolve) => {
        releaseSave = resolve;
      });
      const store = createStudioSceneStore();
      store.getState().hydrate("page:1", makeScene(), async () => {
        if (deferred) await pendingSave;
      });
      await store.getState().patchNode("text:a", { opacity: 0.2 });
      if (operation === "redo") await store.getState().undo();
      deferred = true;

      const historyCommand = store.getState()[operation]();
      const pageTwo = makeScene();
      pageTwo.nodes = pageTwo.nodes.map((node) => node.id === "text:a" ? { ...node, name: "Página 2" } : node);
      store.getState().hydrate("page:2", pageTwo, async () => undefined);
      releaseSave();
      await historyCommand;

      expect(store.getState().pageKey).toBe("page:2");
      expect(store.getState().scene?.nodes.find((node) => node.id === "text:a")).toMatchObject({
        name: "Página 2",
        opacity: 1,
      });
      expect(store.getState().history).toHaveLength(0);
    },
  );

  it("rejects grouping nodes from different parents without a partial scene change", async () => {
    const store = createStudioSceneStore();
    const scene = makeScene();
    scene.nodes.push({
      id: "group:existing",
      kind: "group",
      name: "Grupo existente",
      visible: true,
      locked: false,
      opacity: 1,
      blend_mode: "normal",
      parent_id: null,
      order: 2,
      mask_ids: [],
      metadata: {},
    });
    scene.roots = ["image:base", "text:a", "group:existing"];
    scene.nodes = scene.nodes.map((node) =>
      node.id === "text:b" ? { ...node, parent_id: "group:existing", order: 0 } : node,
    );
    store.getState().hydrate("page:1", scene, async () => undefined);
    store.getState().selectNode("text:a");
    store.getState().selectNode("text:b", true);
    const before = structuredClone(store.getState().scene);

    await expect(store.getState().groupSelected("Inválido", "group:invalid")).rejects.toThrow(
      "mesmo grupo",
    );

    expect(store.getState().scene).toEqual(before);
    expect(store.getState().history).toHaveLength(0);
    expect(store.getState().error).toContain("mesmo grupo");
  });

  it("projects group state to compatibility layers without destroying intrinsic child state", () => {
    const page = importStudioProject({
      versao: "2.0",
      paginas: [
        {
          numero: 1,
          image_layers: { base: { path: "base.png" } },
          text_layers: [
            { id: "a", bbox: [0, 0, 100, 100], original: "A", translated: "Olá" },
            { id: "b", bbox: [0, 0, 100, 100], original: "B", translated: "Mundo" },
          ],
        },
      ],
    }).project.paginas[0];
    const scene = makeScene();
    scene.nodes.push({
      id: "group:lettering",
      kind: "group",
      name: "Lettering",
      visible: false,
      locked: true,
      opacity: 0.5,
      blend_mode: "normal",
      parent_id: null,
      order: 1,
      mask_ids: [],
      metadata: {},
    });
    scene.roots = ["image:base", "group:lettering"];
    scene.nodes = scene.nodes.map((node) =>
      node.id === "text:a" || node.id === "text:b" ? { ...node, parent_id: "group:lettering" } : node,
    );

    const projected = projectStudioSceneToPage(page, scene);

    expect(projected.text_layers[0]).toMatchObject({ visible: false, locked: true, opacity: 0.5 });
    expect(projected.studio_scene.nodes.find((node) => node.id === "text:a")).toMatchObject({
      visible: true,
      locked: false,
      opacity: 1,
      metadata: expect.objectContaining({ scene_owned: true }),
    });
    expect(projected.textos).toEqual(projected.text_layers);
  });
});

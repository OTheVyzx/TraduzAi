import { describe, expect, it } from "vitest";
import { MemoryStudioEditorBackend } from "../../../backend/memoryBackend";
import { importStudioProject } from "../../../project/adapters";
import { persistStudioScene } from "../studioScenePersistence";

describe("persistStudioScene", () => {
  it("loads the latest project, projects the scene and saves it as one transaction", async () => {
    const project = importStudioProject({
      versao: "2.0",
      paginas: [
        {
          numero: 1,
          image_layers: { base: { path: "base.png" } },
          text_layers: [{ id: "t1", bbox: [0, 0, 100, 100], original: "A", translated: "B" }],
        },
      ],
    }).project;
    const backend = new MemoryStudioEditorBackend({ "memory://scene-persist": project });
    const scene = structuredClone(project.paginas[0].studio_scene);
    scene.nodes = scene.nodes.map((node) =>
      node.text_layer_id === "t1" ? { ...node, visible: false, opacity: 0.45 } : node,
    );

    const savedPage = await persistStudioScene({
      backend,
      projectPath: "memory://scene-persist",
      pageIndex: 0,
      scene,
    });
    const reopened = await backend.loadProject({ project_path: "memory://scene-persist" });

    expect(savedPage.text_layers[0]).toMatchObject({ visible: false, opacity: 0.45 });
    expect(reopened.paginas[0].text_layers[0]).toMatchObject({ visible: false, opacity: 0.45 });
    expect(reopened.paginas[0].studio_scene.nodes.find((node) => node.text_layer_id === "t1")).toMatchObject({
      visible: false,
      opacity: 0.45,
      metadata: expect.objectContaining({ scene_owned: true }),
    });
  });
});

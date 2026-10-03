import type { StudioBitmapOperationCommit } from "../../../../src/components/editor/stage/bitmapStrokePreview";
import { getStudioEditorBackend } from "../../backend/editorBackend";
import type { StudioScene, StudioSceneNode } from "../../project/studioProject";
import { useStudioSceneStore } from "../../store/studioSceneStore";

type StoredStudioBitmapOperation = StudioBitmapOperationCommit & {
  baselinePath: string | null;
  operationPath: string;
};

function categoryFor(layerKey: StoredStudioBitmapOperation["layerKey"]) {
  if (layerKey === "brush") return "painting";
  if (layerKey === "inpaint") return "cleanup";
  return "recovery";
}

function operationGroup(scene: StudioScene, layerKey: string) {
  const category = layerKey === "brush" ? "painting" : layerKey === "inpaint" ? "cleanup" : "recovery";
  return scene.nodes.find((node) =>
    node.kind === "group" &&
    (node.metadata.raster_operations_key === layerKey || node.metadata.auto_category_group === category),
  );
}

export function appendStudioBitmapOperation(
  scene: StudioScene,
  operation: StoredStudioBitmapOperation,
): StudioScene {
  const category = categoryFor(operation.layerKey);
  const nodes = [...scene.nodes];
  const roots = [...scene.roots];
  let group = operationGroup(scene, operation.layerKey);
  if (!group) {
    let id = "group:auto:" + category;
    let suffix = 2;
    while (nodes.some((node) => node.id === id)) id = "group:auto:" + category + ":" + suffix++;
    group = {
      id,
      kind: "group",
      name: category === "painting" ? "Pintura" : category === "cleanup" ? "Limpeza" : "Recuperação",
      visible: true,
      locked: false,
      opacity: 1,
      blend_mode: "normal",
      parent_id: null,
      order: roots.length,
      mask_ids: [],
      metadata: { auto_category_group: category, scene_owned: true },
    };
    nodes.push(group);
    roots.push(group.id);
  }

  const alreadyIsolated = group.metadata.raster_operations_key === operation.layerKey;
  const projectedLayer = nodes.find((node) => node.image_layer_key === operation.layerKey);
  const filteredNodes = nodes.filter((node) => node !== projectedLayer);
  const baselineId = "generated:studio-" + operation.layerKey + "-baseline";
  if (!alreadyIsolated && operation.baselinePath) {
    for (let index = 0; index < filteredNodes.length; index += 1) {
      const node = filteredNodes[index];
      if (node.parent_id === group.id) filteredNodes[index] = { ...node, order: node.order + 1 };
    }
    filteredNodes.push({
      id: baselineId,
      kind: "generated",
      name: "Base anterior",
      visible: true,
      locked: false,
      opacity: 1,
      blend_mode: "normal",
      parent_id: group.id,
      order: 0,
      mask_ids: [],
      metadata: {
        scene_owned: true,
        generator: "studio-bitmap-baseline",
        image_path: operation.baselinePath,
      },
    });
  }

  const updatedGroup: StudioSceneNode = {
    ...group,
    metadata: {
      ...group.metadata,
      raster_operations_key: operation.layerKey,
      scene_owned: true,
    },
  };
  const updatedNodes = filteredNodes.map((node) => node.id === group!.id ? updatedGroup : node);
  const operationOrder = updatedNodes
    .filter((node) => node.parent_id === group!.id)
    .reduce((max, node) => Math.max(max, node.order), -1) + 1;
  updatedNodes.push({
    id: "generated:studio-stroke:" + operation.commandId,
    kind: "generated",
    name: operation.name,
    visible: true,
    locked: false,
    opacity: 1,
    blend_mode: operation.blendMode,
    parent_id: group.id,
    order: operationOrder,
    mask_ids: [],
    metadata: {
      scene_owned: true,
      generator: "studio-bitmap-operation",
      image_path: operation.operationPath,
      source_bbox: operation.bbox,
      editor_command_id: operation.commandId,
      operation_kind: operation.blendMode === "destination-out" ? "erase" : "paint",
    },
  });

  const rootsWithoutProjection = roots.filter((id) => id !== projectedLayer?.id);
  return { ...scene, roots: rootsWithoutProjection, nodes: updatedNodes };
}

export async function persistStudioBitmapOperation(input: {
  projectPath: string;
  pageIndex: number;
  operation: StudioBitmapOperationCommit;
}) {
  const { projectPath, pageIndex, operation } = input;
  const state = useStudioSceneStore.getState();
  if (state.pageKey !== `${projectPath}::${pageIndex}`) {
    throw new Error("A página mudou antes de a camada PNG ser salva");
  }
  if (!state.scene) throw new Error("The Studio scene has not loaded yet");
  const firstOperation = operationGroup(state.scene, operation.layerKey)?.metadata.raster_operations_key !== operation.layerKey;
  const backend = getStudioEditorBackend();
  const baselinePath = firstOperation && operation.baselineDataUrl
    ? await backend.saveGeneratedAsset({
        project_path: projectPath,
        page_index: pageIndex,
        asset_id: "studio-" + operation.layerKey + "-baseline-" + crypto.randomUUID(),
        png_data: operation.baselineDataUrl,
      })
    : null;
  const operationPath = await backend.saveGeneratedAsset({
    project_path: projectPath,
    page_index: pageIndex,
    asset_id: "studio-stroke-" + crypto.randomUUID(),
    png_data: operation.pngData,
  });
  const changed = await useStudioSceneStore.getState().executeSceneCommand(
    operation.name + " - PNG layer",
    (scene) => appendStudioBitmapOperation(scene, { ...operation, baselinePath, operationPath }),
    { editorCommandId: operation.commandId },
  );
  if (!changed) throw new Error("The PNG layer was not added to the Studio scene");
}

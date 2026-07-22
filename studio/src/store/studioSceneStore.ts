import { create } from "zustand";
import type {
  StudioPage,
  StudioScene,
  StudioSceneNode,
} from "../project/studioProject";

const MAX_SCENE_HISTORY = 100;

export type StudioScenePersist = (scene: StudioScene) => Promise<void>;
export type StudioSceneTransform = (scene: StudioScene) => StudioScene;

export interface StudioSceneHistoryEntry {
  id: string;
  label: string;
  before: StudioScene;
  after: StudioScene;
  createdAt: number;
}

export type StudioSceneNodePatch = Partial<
  Pick<StudioSceneNode, "visible" | "locked" | "opacity" | "blend_mode" | "name">
>;

export interface StudioSceneState {
  pageKey: string | null;
  scene: StudioScene | null;
  selectedNodeIds: string[];
  primaryNodeId: string | null;
  history: StudioSceneHistoryEntry[];
  historyIndex: number;
  isSaving: boolean;
  error: string | null;
  persist: StudioScenePersist | null;
  hydrate: (pageKey: string, scene: StudioScene, persist: StudioScenePersist) => void;
  selectNode: (nodeId: string, additive?: boolean) => void;
  executeSceneCommand: (label: string, transform: StudioSceneTransform) => Promise<boolean>;
  patchNode: (nodeId: string, patch: StudioSceneNodePatch) => Promise<boolean>;
  groupSelected: (name?: string, groupId?: string) => Promise<boolean>;
  moveNodeBefore: (nodeId: string, targetNodeId: string) => Promise<boolean>;
  undo: () => Promise<boolean>;
  redo: () => Promise<boolean>;
  clearError: () => void;
}

function cloneScene(scene: StudioScene): StudioScene {
  return JSON.parse(JSON.stringify(scene)) as StudioScene;
}

function sceneEquals(left: StudioScene, right: StudioScene) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function clampOpacity(opacity: number) {
  return Math.min(1, Math.max(0, opacity));
}

export function orderedSceneChildren(scene: StudioScene, parentId: string | null): StudioSceneNode[] {
  const children = scene.nodes.filter((node) => node.parent_id === parentId);
  if (parentId !== null) {
    return children.sort((left, right) => left.order - right.order || left.id.localeCompare(right.id));
  }

  const byId = new Map(children.map((node) => [node.id, node]));
  const ordered = scene.roots.map((id) => byId.get(id)).filter((node): node is StudioSceneNode => Boolean(node));
  const included = new Set(ordered.map((node) => node.id));
  const missing = children
    .filter((node) => !included.has(node.id))
    .sort((left, right) => left.order - right.order || left.id.localeCompare(right.id));
  return [...ordered, ...missing];
}

function applySiblingOrder(scene: StudioScene, parentId: string | null, orderedIds: string[]) {
  const orderById = new Map(orderedIds.map((id, order) => [id, order]));
  const nodes = scene.nodes.map((node) =>
    node.parent_id === parentId && orderById.has(node.id)
      ? { ...node, order: orderById.get(node.id)! }
      : node,
  );
  return {
    ...scene,
    nodes,
    roots: parentId === null ? [...orderedIds] : scene.roots,
  };
}

function patchSceneNode(scene: StudioScene, nodeId: string, patch: StudioSceneNodePatch) {
  const safePatch: StudioSceneNodePatch = {
    ...patch,
    ...(typeof patch.opacity === "number" ? { opacity: clampOpacity(patch.opacity) } : {}),
  };
  return {
    ...scene,
    nodes: scene.nodes.map((node) =>
      node.id === nodeId
        ? {
            ...node,
            ...safePatch,
            metadata: { ...node.metadata, scene_owned: true },
          }
        : node,
    ),
  };
}

function groupSceneNodes(scene: StudioScene, selectedIds: string[], name: string, groupId: string) {
  if (scene.nodes.some((node) => node.id === groupId)) {
    throw new Error(`Já existe uma camada com o id ${groupId}`);
  }
  const selectedSet = new Set(selectedIds);
  const selectedNodes = scene.nodes.filter((node) => selectedSet.has(node.id));
  if (selectedNodes.length === 0) return scene;
  const parentId = selectedNodes[0].parent_id;
  if (selectedNodes.some((node) => node.parent_id !== parentId)) {
    throw new Error("Selecione camadas do mesmo grupo para agrupá-las");
  }

  const siblings = orderedSceneChildren(scene, parentId);
  const orderedSelected = siblings.filter((node) => selectedSet.has(node.id));
  if (orderedSelected.length === 0) return scene;
  const insertionIndex = Math.min(...orderedSelected.map((node) => siblings.findIndex((item) => item.id === node.id)));
  const siblingIds = siblings.filter((node) => !selectedSet.has(node.id)).map((node) => node.id);
  siblingIds.splice(insertionIndex, 0, groupId);

  const group: StudioSceneNode = {
    id: groupId,
    kind: "group",
    name,
    visible: true,
    locked: false,
    opacity: 1,
    blend_mode: "normal",
    parent_id: parentId,
    order: insertionIndex,
    mask_ids: [],
    metadata: { scene_owned: true },
  };
  const groupedNodes = scene.nodes.map((node) => {
    const childOrder = orderedSelected.findIndex((item) => item.id === node.id);
    if (childOrder < 0) return node;
    return {
      ...node,
      parent_id: groupId,
      order: childOrder,
      metadata: { ...node.metadata, scene_owned: true },
    };
  });
  return applySiblingOrder({ ...scene, nodes: [...groupedNodes, group] }, parentId, siblingIds);
}

function moveSceneNodeBefore(scene: StudioScene, nodeId: string, targetNodeId: string) {
  if (nodeId === targetNodeId) return scene;
  const node = scene.nodes.find((item) => item.id === nodeId);
  const target = scene.nodes.find((item) => item.id === targetNodeId);
  if (!node || !target || node.parent_id !== target.parent_id) return scene;

  const siblings = orderedSceneChildren(scene, node.parent_id).map((item) => item.id);
  const withoutNode = siblings.filter((id) => id !== nodeId);
  const targetIndex = withoutNode.indexOf(targetNodeId);
  if (targetIndex < 0) return scene;
  withoutNode.splice(targetIndex, 0, nodeId);
  return applySiblingOrder(scene, node.parent_id, withoutNode);
}

function sceneNodesInVisualOrder(scene: StudioScene) {
  const ordered: StudioSceneNode[] = [];
  const visited = new Set<string>();
  const visit = (parentId: string | null) => {
    for (const node of orderedSceneChildren(scene, parentId)) {
      if (visited.has(node.id)) continue;
      visited.add(node.id);
      ordered.push(node);
      visit(node.id);
    }
  };
  visit(null);
  for (const node of scene.nodes) {
    if (!visited.has(node.id)) ordered.push(node);
  }
  return ordered;
}

function effectiveNodeProperties(node: StudioSceneNode, byId: Map<string, StudioSceneNode>) {
  let visible = node.visible;
  let locked = node.locked;
  let opacity = node.opacity;
  let parentId = node.parent_id;
  const visited = new Set<string>([node.id]);
  while (parentId) {
    if (visited.has(parentId)) break;
    visited.add(parentId);
    const parent = byId.get(parentId);
    if (!parent) break;
    visible = visible && parent.visible;
    locked = locked || parent.locked;
    opacity *= parent.opacity;
    parentId = parent.parent_id;
  }
  return { visible, locked, opacity: clampOpacity(opacity) };
}

export function projectStudioSceneToPage(page: StudioPage, scene: StudioScene): StudioPage {
  const ownedScene: StudioScene = {
    ...cloneScene(scene),
    nodes: scene.nodes.map((node) => ({
      ...node,
      metadata: { ...node.metadata, scene_owned: true },
    })),
  };
  const imageLayers = { ...page.image_layers };
  const textLayers = page.text_layers.map((layer) => ({ ...layer }));
  const textById = new Map(textLayers.map((layer, index) => [layer.id, index]));
  const byId = new Map(ownedScene.nodes.map((node) => [node.id, node]));

  sceneNodesInVisualOrder(ownedScene).forEach((node, visualOrder) => {
    const effective = effectiveNodeProperties(node, byId);
    if (node.image_layer_key && imageLayers[node.image_layer_key]) {
      imageLayers[node.image_layer_key] = {
        ...imageLayers[node.image_layer_key]!,
        visible: effective.visible,
        locked: effective.locked,
        opacity: effective.opacity,
        order: visualOrder,
        blend_mode: node.blend_mode,
      };
    }
    if (node.text_layer_id) {
      const textIndex = textById.get(node.text_layer_id);
      if (textIndex === undefined) return;
      textLayers[textIndex] = {
        ...textLayers[textIndex],
        visible: effective.visible,
        locked: effective.locked,
        opacity: effective.opacity,
        order: visualOrder,
        blend_mode: node.blend_mode,
      };
    }
  });

  return {
    ...page,
    image_layers: imageLayers,
    text_layers: textLayers,
    textos: textLayers,
    studio_scene: ownedScene,
  };
}

export function createStudioSceneStore() {
  return create<StudioSceneState>((set, get) => {
    const persistScene = async (scene: StudioScene) => {
      const persist = get().persist;
      if (persist) await persist(scene);
    };

    const commit = async (
      label: string,
      transform: (scene: StudioScene) => StudioScene,
    ): Promise<boolean> => {
      const state = get();
      if (!state.scene) return false;
      if (state.isSaving) throw new Error("Aguarde a operação de camada atual terminar");
      const before = cloneScene(state.scene);
      let after: StudioScene;
      try {
        after = transform(cloneScene(state.scene));
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        set({ error: message });
        throw error;
      }
      if (sceneEquals(before, after)) return false;

      set({ scene: after, isSaving: true, error: null });
      try {
        await persistScene(after);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        if (get().scene === after && get().pageKey === state.pageKey) {
          set({ scene: before, isSaving: false, error: message });
        }
        throw error;
      }

      if (get().scene !== after || get().pageKey !== state.pageKey) return true;

      let history = [
        ...state.history.slice(0, state.historyIndex),
        {
          id: crypto.randomUUID(),
          label,
          before,
          after: cloneScene(after),
          createdAt: Date.now(),
        },
      ];
      if (history.length > MAX_SCENE_HISTORY) history = history.slice(-MAX_SCENE_HISTORY);
      set({ scene: after, history, historyIndex: history.length, isSaving: false, error: null });
      return true;
    };

    const restoreHistoryScene = async (scene: StudioScene, historyIndex: number) => {
      const state = get();
      const current = state.scene;
      if (!current || get().isSaving) return false;
      const target = cloneScene(scene);
      set({ scene: target, isSaving: true, error: null });
      try {
        await persistScene(target);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        if (get().scene === target && get().pageKey === state.pageKey) {
          set({ scene: current, isSaving: false, error: message });
        }
        throw error;
      }
      if (get().scene !== target || get().pageKey !== state.pageKey) return true;
      set({ scene: target, historyIndex, isSaving: false, error: null });
      return true;
    };

    return {
      pageKey: null,
      scene: null,
      selectedNodeIds: [],
      primaryNodeId: null,
      history: [],
      historyIndex: 0,
      isSaving: false,
      error: null,
      persist: null,

      hydrate: (pageKey, scene, persist) => {
        set({
          pageKey,
          scene: cloneScene(scene),
          selectedNodeIds: [],
          primaryNodeId: null,
          history: [],
          historyIndex: 0,
          isSaving: false,
          error: null,
          persist,
        });
      },

      selectNode: (nodeId, additive = false) => {
        const scene = get().scene;
        if (!scene?.nodes.some((node) => node.id === nodeId)) return;
        if (!additive) {
          set({ selectedNodeIds: [nodeId], primaryNodeId: nodeId });
          return;
        }
        const selected = get().selectedNodeIds;
        if (selected.includes(nodeId)) {
          const next = selected.filter((id) => id !== nodeId);
          set({ selectedNodeIds: next, primaryNodeId: next.at(-1) ?? null });
          return;
        }
        set({ selectedNodeIds: [...selected, nodeId], primaryNodeId: nodeId });
      },

      executeSceneCommand: (label, transform) => commit(label, transform),

      patchNode: (nodeId, patch) => commit("Editar propriedades da camada", (scene) => patchSceneNode(scene, nodeId, patch)),

      groupSelected: async (name = "Novo grupo", groupId = `group:${crypto.randomUUID()}`) => {
        const selectedIds = [...get().selectedNodeIds];
        const changed = await commit("Agrupar camadas", (scene) => groupSceneNodes(scene, selectedIds, name, groupId));
        if (changed) set({ selectedNodeIds: [groupId], primaryNodeId: groupId });
        return changed;
      },

      moveNodeBefore: (nodeId, targetNodeId) =>
        commit("Reordenar camada", (scene) => moveSceneNodeBefore(scene, nodeId, targetNodeId)),

      undo: async () => {
        const { history, historyIndex } = get();
        if (historyIndex <= 0) return false;
        return restoreHistoryScene(history[historyIndex - 1].before, historyIndex - 1);
      },

      redo: async () => {
        const { history, historyIndex } = get();
        if (historyIndex >= history.length) return false;
        return restoreHistoryScene(history[historyIndex].after, historyIndex + 1);
      },

      clearError: () => set({ error: null }),
    };
  });
}

export const useStudioSceneStore = createStudioSceneStore();

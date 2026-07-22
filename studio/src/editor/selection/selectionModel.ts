import {
  combineLassoSelections,
  createLassoSelection,
  lassoSelectionEffectiveBbox,
  withLassoSelectionModifiers,
  type LassoSelection,
  type LassoSelectionRegion,
  type LassoSelectionRegionOperation,
} from "../../../../src/lib/lassoSelection";
import type { StudioScene, StudioSceneNode } from "../../project/studioProject";

export interface StudioSelection extends LassoSelection {
  id: string;
  regions: LassoSelectionRegion[];
  feather: number;
  expansion: number;
  targetNodeId: string | null;
}

export interface CreateStudioSelectionInput {
  id?: string;
  pageKey: string;
  pageIndex: number;
  points: Array<[number, number]>;
  width: number;
  height: number;
  targetNodeId?: string | null;
}

function cloneRegions(regions: LassoSelectionRegion[]) {
  return regions.map((region) => ({
    operation: region.operation,
    points: region.points.map(([x, y]) => [x, y] as [number, number]),
  }));
}

function asStudioSelection(selection: LassoSelection, fallbackId?: string): StudioSelection {
  return {
    ...selection,
    id: selection.id ?? fallbackId ?? `selection:${crypto.randomUUID()}`,
    regions: cloneRegions(selection.regions ?? [{ operation: "add", points: selection.points }]),
    feather: selection.feather ?? 0,
    expansion: selection.expansion ?? 0,
    targetNodeId: selection.targetNodeId ?? null,
  };
}

export function createStudioSelection(input: CreateStudioSelectionInput): StudioSelection {
  const selection = createLassoSelection({
    ...input,
    id: input.id ?? `selection:${crypto.randomUUID()}`,
    feather: 0,
    expansion: 0,
    targetNodeId: input.targetNodeId ?? null,
  });
  return asStudioSelection(selection, input.id);
}

export function studioSelectionFromLasso(
  selection: LassoSelection,
  targetNodeId: string | null = selection.targetNodeId ?? null,
): StudioSelection {
  return asStudioSelection(withLassoSelectionModifiers(selection, { targetNodeId }));
}

export function adjustStudioSelection(
  selection: StudioSelection,
  patch: { feather?: number; expansion?: number; targetNodeId?: string | null },
): StudioSelection {
  return asStudioSelection(withLassoSelectionModifiers(selection, patch), selection.id);
}

export function combineStudioSelections(
  current: StudioSelection | null,
  next: StudioSelection,
  operation: "replace" | LassoSelectionRegionOperation,
): StudioSelection {
  const combined = asStudioSelection(combineLassoSelections(current, next, operation), current?.id ?? next.id);
  return { ...combined, id: current && operation !== "replace" ? current.id : next.id };
}

export function studioSelectionEffectiveBbox(selection: StudioSelection) {
  return lassoSelectionEffectiveBbox(selection);
}

function cloneScene(scene: StudioScene): StudioScene {
  return JSON.parse(JSON.stringify(scene)) as StudioScene;
}

function maskNodeForSelection(
  selection: StudioSelection,
  target: StudioSceneNode,
  maskId: string,
  name: string,
): StudioSceneNode {
  const childOrder = target.mask_ids.length;
  return {
    id: maskId,
    kind: "mask",
    name,
    visible: true,
    locked: false,
    opacity: 1,
    blend_mode: "normal",
    parent_id: target.id,
    order: childOrder,
    mask_ids: [],
    metadata: {
      scene_owned: true,
      mask_role: "layer",
      target_node_id: target.id,
      selection: {
        ...selection,
        points: selection.points.map(([x, y]) => [x, y]),
        regions: cloneRegions(selection.regions),
      },
    },
  };
}

export function attachStudioSelectionMask(
  scene: StudioScene,
  selection: StudioSelection,
  options: { maskId?: string; name?: string } = {},
): StudioScene {
  if (!selection.targetNodeId) throw new Error("Selecione uma camada-alvo antes de criar a máscara");
  const target = scene.nodes.find((node) => node.id === selection.targetNodeId);
  if (!target) throw new Error(`Camada-alvo não encontrada: ${selection.targetNodeId}`);
  if (target.locked) throw new Error(`A camada-alvo está bloqueada: ${target.name}`);
  if (target.kind === "mask" || target.kind === "group") {
    throw new Error("A camada-alvo não aceita máscara de camada");
  }

  const next = cloneScene(scene);
  const nextTarget = next.nodes.find((node) => node.id === target.id)!;
  const maskId = options.maskId ?? `mask:${selection.id}:${crypto.randomUUID()}`;
  next.nodes = next.nodes.map((node) => node.id !== nextTarget.id && node.mask_ids.includes(maskId)
    ? { ...node, mask_ids: node.mask_ids.filter((id) => id !== maskId) }
    : node);
  nextTarget.mask_ids = nextTarget.mask_ids.filter((id) => id !== maskId);
  const mask = maskNodeForSelection(selection, nextTarget, maskId, options.name ?? "Máscara de camada");
  const existingIndex = next.nodes.findIndex((node) => node.id === maskId);
  if (existingIndex >= 0 && next.nodes[existingIndex].kind !== "mask") {
    throw new Error(`O id ${maskId} já pertence a outra camada`);
  }
  if (existingIndex >= 0) next.nodes[existingIndex] = mask;
  else next.nodes.push(mask);
  nextTarget.mask_ids = [...nextTarget.mask_ids, maskId];
  nextTarget.metadata = { ...nextTarget.metadata, scene_owned: true };
  return next;
}

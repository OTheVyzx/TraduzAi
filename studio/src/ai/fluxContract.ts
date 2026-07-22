import type { StudioScene, StudioSceneNode } from "../project/studioProject";
import {
  adjustStudioSelection,
  attachStudioSelectionMask,
  type StudioSelection,
} from "../editor/selection/selectionModel";

export const FLUX_ADAPTER_CONTRACT_VERSION = "1.0" as const;
export const DEFAULT_FLUX_MODEL = "black-forest-labs/FLUX.1-Fill-dev";

export interface FluxProviderStatus {
  status: "ready" | "configured" | "missing" | "error";
  provider: string;
  model?: string | null;
  message?: string | null;
}

export interface FluxGenerateConfig {
  contract_version: typeof FLUX_ADAPTER_CONTRACT_VERSION;
  job_id: string;
  prompt: string;
  negative_prompt: string;
  model: string;
  source_png_data: string;
  mask_png_data: string;
  width: number;
  height: number;
  variant_count: number;
  seed: number;
  steps: number;
  guidance_scale: number;
}

export interface FluxProviderVariant {
  id: string;
  seed: number;
  png_data?: string | null;
  path?: string | null;
}

export interface FluxGenerateResult {
  contract_version: typeof FLUX_ADAPTER_CONTRACT_VERSION;
  job_id: string;
  provider: string;
  model: string;
  variants: FluxProviderVariant[];
}

export interface FluxGenerationVariant {
  id: string;
  seed: number;
  resultPath: string;
}

export interface FluxGeneration {
  version: "1.0";
  id: string;
  targetNodeId: string;
  selection: StudioSelection;
  prompt: string;
  negativePrompt: string;
  provider: string;
  model: string;
  cropBbox: [number, number, number, number];
  seed: number;
  variants: FluxGenerationVariant[];
  createdAt: number;
}

export interface CreateFluxGenerationInput extends Omit<FluxGeneration, "version"> {}

export interface FluxExecutionContextToken {
  pageKey: string | null;
  pageIndex: number;
  scene: StudioScene;
}

export function assertFluxExecutionContext(
  expected: FluxExecutionContextToken,
  current: { pageKey: string | null; pageIndex: number; scene: StudioScene | null },
) {
  if (
    !expected.pageKey ||
    current.pageKey !== expected.pageKey ||
    current.pageIndex !== expected.pageIndex ||
    current.scene !== expected.scene
  ) {
    throw new Error("A página mudou durante a geração FLUX; a operação foi cancelada");
  }
}

function cloneScene(scene: StudioScene): StudioScene {
  return JSON.parse(JSON.stringify(scene)) as StudioScene;
}

function normalizeSiblingOrder(scene: StudioScene, parentId: string | null) {
  const siblings = scene.nodes
    .filter((node) => node.parent_id === parentId)
    .sort((left, right) => left.order - right.order || left.id.localeCompare(right.id));
  const orderById = new Map(siblings.map((node, order) => [node.id, order]));
  scene.nodes = scene.nodes.map((node) => (
    node.parent_id === parentId && orderById.has(node.id)
      ? { ...node, order: orderById.get(node.id)! }
      : node
  ));
  if (parentId === null) scene.roots = siblings.map((node) => node.id);
}

function insertSiblingAfter(scene: StudioScene, target: StudioSceneNode, output: StudioSceneNode) {
  const siblings = scene.nodes
    .filter((node) => node.parent_id === target.parent_id && node.id !== output.id)
    .sort((left, right) => left.order - right.order || left.id.localeCompare(right.id));
  const targetIndex = siblings.findIndex((node) => node.id === target.id);
  siblings.splice(targetIndex < 0 ? siblings.length : targetIndex + 1, 0, output);
  const orderById = new Map(siblings.map((node, order) => [node.id, order]));
  output.order = orderById.get(output.id) ?? target.order + 1;
  scene.nodes = scene.nodes.map((node) => (
    node.parent_id === target.parent_id && orderById.has(node.id)
      ? { ...node, order: orderById.get(node.id)! }
      : node
  ));
  scene.nodes.push(output);
  if (target.parent_id === null) scene.roots = siblings.map((node) => node.id);
}

function generationGroupId(generationId: string) {
  return `group:${generationId}`;
}

function generatedVariantNodeId(generationId: string, variantId: string) {
  return `generated:${generationId}:${variantId}`;
}

function variantMaskNodeId(generationId: string, variantId: string) {
  return `mask:${generationId}:${variantId}`;
}

function generationGroup(scene: StudioScene, generationId: string) {
  return scene.nodes.find((node) => (
    node.id === generationGroupId(generationId) &&
    node.kind === "group" &&
    node.metadata.generation_id === generationId
  ));
}

export function findFluxGenerationId(scene: StudioScene | null, primaryNodeId: string | null) {
  if (!scene || !primaryNodeId) return null;
  const primary = scene.nodes.find((node) => node.id === primaryNodeId);
  const directId = typeof primary?.metadata.generation_id === "string"
    ? primary.metadata.generation_id
    : null;
  if (directId && generationGroup(scene, directId)) return directId;

  const candidates = scene.nodes
    .filter((node) => (
      node.kind === "group" &&
      node.metadata.generator === "flux-fill" &&
      node.metadata.source_node_id === primaryNodeId &&
      typeof node.metadata.generation_id === "string"
    ))
    .sort((left, right) => {
      const leftGeneration = left.metadata.flux_generation as Partial<FluxGeneration> | undefined;
      const rightGeneration = right.metadata.flux_generation as Partial<FluxGeneration> | undefined;
      return (rightGeneration?.createdAt ?? 0) - (leftGeneration?.createdAt ?? 0);
    });
  return typeof candidates[0]?.metadata.generation_id === "string"
    ? candidates[0].metadata.generation_id
    : null;
}

function descendantIds(scene: StudioScene, parentId: string) {
  const ids = new Set<string>();
  const visit = (id: string) => {
    for (const child of scene.nodes.filter((node) => node.parent_id === id)) {
      if (ids.has(child.id)) continue;
      ids.add(child.id);
      visit(child.id);
    }
  };
  visit(parentId);
  return ids;
}

export function createFluxGeneration(input: CreateFluxGenerationInput): FluxGeneration {
  if (!input.id.trim()) throw new Error("A geração FLUX precisa de um identificador");
  if (!input.targetNodeId.trim()) throw new Error("Selecione uma camada-alvo para o FLUX");
  if (input.selection.targetNodeId && input.selection.targetNodeId !== input.targetNodeId) {
    throw new Error("A seleção e o FLUX precisam usar a mesma camada-alvo");
  }
  if (input.variants.length < 2 || input.variants.length > 4) {
    throw new Error("O FLUX precisa retornar entre 2 e 4 variantes");
  }
  const [left, top, right, bottom] = input.cropBbox.map((value) => Math.round(value));
  if (right <= left || bottom <= top) throw new Error("A área selecionada para o FLUX é inválida");
  const variantIds = new Set<string>();
  const variants = input.variants.map((variant) => {
    const id = variant.id.trim();
    const resultPath = variant.resultPath.trim();
    if (!id || variantIds.has(id)) throw new Error("As variantes FLUX precisam de ids únicos");
    if (!resultPath) throw new Error(`A variante ${id} ainda não possui imagem salva`);
    variantIds.add(id);
    return { id, seed: Math.trunc(variant.seed), resultPath };
  });
  return {
    version: "1.0",
    id: input.id.trim(),
    targetNodeId: input.targetNodeId,
    selection: adjustStudioSelection(input.selection, { targetNodeId: input.targetNodeId }),
    prompt: input.prompt.trim(),
    negativePrompt: input.negativePrompt.trim(),
    provider: input.provider.trim() || "local-adapter",
    model: input.model.trim() || DEFAULT_FLUX_MODEL,
    cropBbox: [left, top, right, bottom],
    seed: Math.trunc(input.seed),
    variants,
    createdAt: input.createdAt,
  };
}

export function isolateFluxVariantPixels(candidate: Uint8Array, mask: Uint8Array) {
  if (candidate.length % 4 !== 0 || mask.length !== candidate.length / 4) {
    throw new Error("A variante FLUX e a máscara possuem dimensões incompatíveis");
  }
  const isolated = new Uint8Array(candidate);
  for (let pixel = 0; pixel < mask.length; pixel += 1) {
    if (mask[pixel] === 0) isolated[pixel * 4 + 3] = 0;
  }
  return isolated;
}

export function applyFluxGenerationToScene(scene: StudioScene, generation: FluxGeneration) {
  const target = scene.nodes.find((node) => node.id === generation.targetNodeId);
  if (!target) throw new Error(`Camada-alvo não encontrada: ${generation.targetNodeId}`);
  if (target.locked) throw new Error(`A camada-alvo está bloqueada: ${target.name}`);
  if (target.kind !== "raster" && target.kind !== "generated") {
    throw new Error("O preenchimento FLUX exige uma camada raster ou gerada como alvo");
  }
  const groupId = generationGroupId(generation.id);
  const reservedIds = new Set([groupId]);
  for (const variant of generation.variants) {
    reservedIds.add(generatedVariantNodeId(generation.id, variant.id));
    reservedIds.add(variantMaskNodeId(generation.id, variant.id));
  }
  if (scene.nodes.some((node) => reservedIds.has(node.id))) {
    throw new Error("Já existem camadas para esta geração FLUX");
  }

  let next = cloneScene(scene);
  const nextTarget = next.nodes.find((node) => node.id === target.id)!;
  const group: StudioSceneNode = {
    id: groupId,
    kind: "group",
    name: generation.prompt ? `FLUX — ${generation.prompt.slice(0, 48)}` : "FLUX — preenchimento",
    visible: true,
    locked: false,
    opacity: 1,
    blend_mode: "normal",
    parent_id: target.parent_id,
    order: target.order + 1,
    mask_ids: [],
    metadata: {
      scene_owned: true,
      generator: "flux-fill",
      generation_id: generation.id,
      source_node_id: generation.targetNodeId,
      flux_generation: generation,
    },
  };
  insertSiblingAfter(next, nextTarget, group);

  const variantNodes = generation.variants.map((variant, index): StudioSceneNode => ({
    id: generatedVariantNodeId(generation.id, variant.id),
    kind: "generated",
    name: `Variante ${index + 1} — FLUX`,
    visible: index === 0,
    locked: false,
    opacity: 1,
    blend_mode: "normal",
    parent_id: group.id,
    order: index,
    mask_ids: [],
    metadata: {
      scene_owned: true,
      generator: "flux-fill",
      generation_id: generation.id,
      variant_id: variant.id,
      image_path: variant.resultPath,
      source_node_id: generation.targetNodeId,
      provider: generation.provider,
      model: generation.model,
      seed: variant.seed,
      prompt: generation.prompt,
      negative_prompt: generation.negativePrompt,
      crop_bbox: generation.cropBbox,
      selection_id: generation.selection.id,
      created_at: generation.createdAt,
    },
  }));
  next.nodes.push(...variantNodes);
  for (const [index, variant] of generation.variants.entries()) {
    const outputNodeId = generatedVariantNodeId(generation.id, variant.id);
    next = attachStudioSelectionMask(
      next,
      adjustStudioSelection(generation.selection, { targetNodeId: outputNodeId }),
      {
        maskId: variantMaskNodeId(generation.id, variant.id),
        name: `Máscara — Variante ${index + 1}`,
      },
    );
  }
  return next;
}

export function activateFluxVariant(scene: StudioScene, generationId: string, variantId: string) {
  const group = generationGroup(scene, generationId);
  if (!group) throw new Error(`Geração FLUX não encontrada: ${generationId}`);
  const targetId = generatedVariantNodeId(generationId, variantId);
  if (!scene.nodes.some((node) => node.id === targetId && node.parent_id === group.id)) {
    throw new Error(`Variante FLUX não encontrada: ${variantId}`);
  }
  return {
    ...cloneScene(scene),
    nodes: scene.nodes.map((node) => (
      node.parent_id === group.id && node.kind === "generated"
        ? { ...node, visible: node.id === targetId, metadata: { ...node.metadata, scene_owned: true } }
        : JSON.parse(JSON.stringify(node)) as StudioSceneNode
    )),
  };
}

export function acceptFluxVariant(scene: StudioScene, generationId: string, variantId: string) {
  const group = generationGroup(scene, generationId);
  if (!group) throw new Error(`Geração FLUX não encontrada: ${generationId}`);
  const chosenId = generatedVariantNodeId(generationId, variantId);
  const chosen = scene.nodes.find((node) => node.id === chosenId && node.parent_id === group.id);
  if (!chosen) throw new Error(`Variante FLUX não encontrada: ${variantId}`);

  const next = cloneScene(scene);
  const descendants = descendantIds(next, group.id);
  const chosenDescendants = descendantIds(next, chosenId);
  const removeIds = new Set([...descendants].filter((id) => id !== chosenId && !chosenDescendants.has(id)));
  removeIds.add(group.id);
  next.nodes = next.nodes
    .filter((node) => !removeIds.has(node.id))
    .map((node) => node.id === chosenId
      ? {
          ...node,
          parent_id: group.parent_id,
          order: group.order,
          visible: true,
          metadata: { ...node.metadata, generation_status: "accepted", scene_owned: true },
        }
      : node);
  if (group.parent_id === null) {
    next.roots = next.roots.map((id) => id === group.id ? chosenId : id).filter((id) => !removeIds.has(id));
  }
  normalizeSiblingOrder(next, group.parent_id);
  return next;
}

export function rejectFluxGeneration(scene: StudioScene, generationId: string) {
  const group = generationGroup(scene, generationId);
  if (!group) throw new Error(`Geração FLUX não encontrada: ${generationId}`);
  const next = cloneScene(scene);
  const removeIds = descendantIds(next, group.id);
  removeIds.add(group.id);
  next.nodes = next.nodes.filter((node) => !removeIds.has(node.id));
  next.roots = next.roots.filter((id) => !removeIds.has(id));
  normalizeSiblingOrder(next, group.parent_id);
  return next;
}

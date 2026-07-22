import type { StudioScene, StudioSceneNode } from "../../project/studioProject";
import {
  attachStudioSelectionMask,
  adjustStudioSelection,
  type StudioSelection,
} from "../selection/selectionModel";

export type RetouchTool = "clone" | "healing" | "patch";

export type RetouchSampling =
  | { mode: "automatic" }
  | {
      mode: "sampled";
      sourceNodeId: string;
      sourceOffset: [number, number];
      aligned: boolean;
    };

export interface RetouchSettings {
  opacity: number;
  hardness: number;
  spacing: number;
}

export interface RetouchCommand {
  version: "1.0";
  id: string;
  tool: RetouchTool;
  targetNodeId: string;
  selection: StudioSelection;
  sampling: RetouchSampling;
  settings: RetouchSettings;
  resultPath: string | null;
  status: "pending_render" | "ready";
  createdAt: number;
}

export interface CreateRetouchCommandInput {
  id?: string;
  tool: RetouchTool;
  targetNodeId: string;
  selection: StudioSelection;
  sampling?: RetouchSampling;
  settings?: Partial<RetouchSettings>;
  resultPath?: string | null;
  createdAt?: number;
}

interface RetouchRenderContextLike {
  filter: string;
  globalAlpha: number;
  clearRect(x: number, y: number, width: number, height: number): void;
  drawImage(image: unknown, x: number, y: number, width: number, height: number): void;
  save(): void;
  restore(): void;
}

interface RetouchRenderCanvasLike {
  width: number;
  height: number;
  getContext(type?: "2d"): RetouchRenderContextLike | null;
  toDataURL(type?: string): string;
}

export interface RetouchRenderOptions {
  width: number;
  height: number;
  createCanvas: (width: number, height: number) => RetouchRenderCanvasLike;
  loadNodeImage: (nodeId: string) => Promise<{ image: unknown }>;
}

export interface RetouchExecutionContextToken {
  pageKey: string | null;
  pageIndex: number;
  scene: StudioScene;
}

export function assertRetouchExecutionContext(
  expected: RetouchExecutionContextToken,
  current: { pageKey: string | null; pageIndex: number; scene: StudioScene | null },
) {
  if (
    !expected.pageKey ||
    current.pageKey !== expected.pageKey ||
    current.pageIndex !== expected.pageIndex ||
    current.scene !== expected.scene
  ) {
    throw new Error("A página mudou durante o retoque; a operação foi cancelada");
  }
}

const TOOL_NAMES: Record<RetouchTool, string> = {
  clone: "Clone",
  healing: "Correção",
  patch: "Remendo",
};

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function normalizeSampling(tool: RetouchTool, sampling?: RetouchSampling): RetouchSampling {
  if (!sampling) {
    if (tool === "healing") return { mode: "automatic" };
    throw new Error(`${TOOL_NAMES[tool]} requer uma amostra de origem`);
  }
  if ((tool === "clone" || tool === "patch") && sampling.mode !== "sampled") {
    throw new Error(`${TOOL_NAMES[tool]} requer uma amostra de origem`);
  }
  if (sampling.mode === "sampled" && !sampling.sourceNodeId.trim()) {
    throw new Error("A amostra de origem precisa apontar para uma camada");
  }
  return sampling.mode === "automatic"
    ? { mode: "automatic" }
    : {
        mode: "sampled",
        sourceNodeId: sampling.sourceNodeId,
        sourceOffset: [Math.round(sampling.sourceOffset[0]), Math.round(sampling.sourceOffset[1])],
        aligned: sampling.aligned,
      };
}

export function createRetouchCommand(input: CreateRetouchCommandInput): RetouchCommand {
  if (!input.targetNodeId.trim()) throw new Error("Selecione uma camada-alvo para o retoque");
  if (input.selection.targetNodeId && input.selection.targetNodeId !== input.targetNodeId) {
    throw new Error("A seleção e o retoque precisam usar a mesma camada-alvo");
  }
  const resultPath = input.resultPath?.trim() || null;
  return {
    version: "1.0",
    id: input.id ?? `retouch:${crypto.randomUUID()}`,
    tool: input.tool,
    targetNodeId: input.targetNodeId,
    selection: adjustStudioSelection(input.selection, { targetNodeId: input.targetNodeId }),
    sampling: normalizeSampling(input.tool, input.sampling),
    settings: {
      opacity: clamp(input.settings?.opacity ?? 1, 0, 1),
      hardness: clamp(input.settings?.hardness ?? 0.75, 0, 1),
      spacing: clamp(input.settings?.spacing ?? 0.15, 0.01, 1),
    },
    resultPath,
    status: resultPath ? "ready" : "pending_render",
    createdAt: input.createdAt ?? Date.now(),
  };
}

export async function renderRetouchCommandBitmap(command: RetouchCommand, options: RetouchRenderOptions) {
  const width = Math.max(1, Math.round(options.width));
  const height = Math.max(1, Math.round(options.height));
  const sourceNodeId = command.sampling.mode === "sampled"
    ? command.sampling.sourceNodeId
    : command.targetNodeId;
  const source = await options.loadNodeImage(sourceNodeId);
  const canvas = options.createCanvas(width, height);
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas 2D indisponivel para renderizar o retoque");
  ctx.clearRect(0, 0, width, height);
  ctx.save();
  ctx.globalAlpha = command.settings.opacity;
  if (command.tool === "healing") {
    const blurRadius = Math.max(1, Math.round((1 - command.settings.hardness) * 12));
    ctx.filter = `blur(${blurRadius}px)`;
    ctx.drawImage(source.image, 0, 0, width, height);
  } else {
    const sampling = command.sampling;
    if (sampling.mode !== "sampled") throw new Error(`${TOOL_NAMES[command.tool]} requer uma amostra de origem`);
    ctx.filter = command.tool === "patch" ? "blur(1px)" : "none";
    ctx.drawImage(
      source.image,
      -sampling.sourceOffset[0],
      -sampling.sourceOffset[1],
      width,
      height,
    );
  }
  ctx.restore();
  return canvas.toDataURL("image/png");
}

function cloneScene(scene: StudioScene): StudioScene {
  return JSON.parse(JSON.stringify(scene)) as StudioScene;
}

function insertSiblingAfter(scene: StudioScene, target: StudioSceneNode, output: StudioSceneNode) {
  const siblings = scene.nodes
    .filter((node) => node.parent_id === target.parent_id && node.id !== output.id)
    .sort((left, right) => left.order - right.order || left.id.localeCompare(right.id));
  const targetIndex = siblings.findIndex((node) => node.id === target.id);
  const ordered = [...siblings];
  ordered.splice(targetIndex < 0 ? ordered.length : targetIndex + 1, 0, output);
  const orderById = new Map(ordered.map((node, order) => [node.id, order]));
  scene.nodes = scene.nodes.map((node) => orderById.has(node.id) ? { ...node, order: orderById.get(node.id)! } : node);
  output.order = orderById.get(output.id) ?? target.order + 1;
  scene.nodes.push(output);
  if (target.parent_id === null) scene.roots = ordered.map((node) => node.id);
}

export function applyRetouchCommandToScene(
  scene: StudioScene,
  command: RetouchCommand,
  options: { outputNodeId?: string; maskNodeId?: string; name?: string } = {},
): StudioScene {
  const target = scene.nodes.find((node) => node.id === command.targetNodeId);
  if (!target) throw new Error(`Camada-alvo não encontrada: ${command.targetNodeId}`);
  if (target.locked) throw new Error(`A camada-alvo está bloqueada: ${target.name}`);
  if (target.kind !== "raster" && target.kind !== "generated" && target.kind !== "fill") {
    throw new Error("O retoque exige uma camada raster ou gerada como alvo");
  }
  const sampling = command.sampling;
  if (sampling.mode === "sampled") {
    const source = scene.nodes.find((node) => node.id === sampling.sourceNodeId);
    if (!source) throw new Error(`Camada de amostra não encontrada: ${sampling.sourceNodeId}`);
    if (source.kind !== "raster" && source.kind !== "generated" && source.kind !== "fill") {
      throw new Error("A amostra do retoque precisa vir de uma camada raster ou gerada");
    }
  }

  const next = cloneScene(scene);
  const nextTarget = next.nodes.find((node) => node.id === target.id)!;
  const outputNodeId = options.outputNodeId ?? `generated:${command.id}`;
  const maskNodeId = options.maskNodeId ?? `mask:${command.id}`;
  if (next.nodes.some((node) => node.id === outputNodeId || node.id === maskNodeId)) {
    throw new Error("Já existe uma camada para este comando de retoque");
  }
  const output: StudioSceneNode = {
    id: outputNodeId,
    kind: "generated",
    name: options.name ?? `${TOOL_NAMES[command.tool]} — ${target.name}`,
    visible: true,
    locked: false,
    opacity: 1,
    blend_mode: "normal",
    parent_id: target.parent_id,
    order: target.order + 1,
    mask_ids: [],
    metadata: {
      scene_owned: true,
      generator: "retouch",
      image_path: command.resultPath,
      source_node_id: target.id,
      retouch_command: command,
    },
  };
  insertSiblingAfter(next, nextTarget, output);
  const outputSelection = adjustStudioSelection(command.selection, { targetNodeId: outputNodeId });
  return attachStudioSelectionMask(next, outputSelection, {
    maskId: maskNodeId,
    name: `Máscara — ${TOOL_NAMES[command.tool]}`,
  });
}

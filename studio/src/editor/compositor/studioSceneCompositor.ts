import { loadImageSource } from "../../../../src/lib/imageSource";
import { rasterizeLassoSelectionToCanvas } from "../../../../src/lib/lassoSelection";
import type { ImageLayerKey, StudioPage, StudioScene, StudioSceneNode } from "../../project/studioProject";
import type { StudioSelection } from "../selection/selectionModel";

export interface ResolvedStudioSceneMask {
  nodeId: string;
  name: string;
  opacity: number;
  selection: StudioSelection;
}

export interface ResolvedStudioSceneRenderLayer {
  nodeId: string;
  name: string;
  kind: "raster" | "generated" | "fill";
  imageLayerKey?: ImageLayerKey;
  sourcePath: string | null;
  fillColor: string | null;
  visible: boolean;
  opacity: number;
  blendMode: string;
  masks: ResolvedStudioSceneMask[];
}

interface StudioCanvasContextLike {
  globalAlpha: number;
  globalCompositeOperation: string;
  fillStyle: string;
  clearRect(x: number, y: number, width: number, height: number): void;
  fillRect(x: number, y: number, width: number, height: number): void;
  drawImage(image: unknown, dx?: number, dy?: number, width?: number, height?: number): void;
  save(): void;
  restore(): void;
}

export interface StudioCanvasLike {
  width: number;
  height: number;
  getContext(type?: "2d"): StudioCanvasContextLike | null;
  toDataURL(type?: string): string;
}

export interface LoadedStudioImage {
  image: unknown;
  width: number;
  height: number;
  revoke?: () => void;
}

export interface ComposeStudioSceneBitmapOptions {
  page: StudioPage;
  scene: StudioScene;
  width?: number;
  height?: number;
  createCanvas?: (width: number, height: number) => StudioCanvasLike;
  loadImage?: (path: string) => Promise<LoadedStudioImage>;
  rasterizeSelection?: (selection: StudioSelection) => unknown;
  resolveSourcePath?: (path: string) => string;
}

export interface ResolveStudioSceneRenderLayerOptions {
  includeHidden?: boolean;
}

export type ResolvedStudioSceneVisualItem =
  | { kind: "bitmap"; nodeId: string }
  | { kind: "text"; nodeId: string; textLayerId: string };

export interface ComposedStudioSceneBitmapLayer {
  nodeId: string;
  canvas: StudioCanvasLike;
  opacity: number;
  blendMode: string;
}

export interface ComposedStudioSceneBitmapLayers {
  width: number;
  height: number;
  layers: ComposedStudioSceneBitmapLayer[];
}

function clamp01(value: number) {
  if (!Number.isFinite(value)) return 1;
  return Math.min(1, Math.max(0, value));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isStudioSelection(value: unknown): value is StudioSelection {
  if (!isRecord(value)) return false;
  return (
    typeof value.id === "string" &&
    typeof value.pageKey === "string" &&
    typeof value.pageIndex === "number" &&
    typeof value.width === "number" &&
    typeof value.height === "number" &&
    Array.isArray(value.points)
  );
}

function orderedChildren(scene: StudioScene, parentId: string | null) {
  const children = scene.nodes.filter((node) => node.parent_id === parentId);
  if (parentId !== null) {
    return children.sort((left, right) => left.order - right.order || left.id.localeCompare(right.id));
  }
  const byId = new Map(children.map((node) => [node.id, node]));
  const roots = scene.roots.map((id) => byId.get(id)).filter((node): node is StudioSceneNode => Boolean(node));
  const included = new Set(roots.map((node) => node.id));
  const missing = children
    .filter((node) => !included.has(node.id))
    .sort((left, right) => left.order - right.order || left.id.localeCompare(right.id));
  return [...roots, ...missing];
}

function nodesInVisualOrder(scene: StudioScene) {
  const ordered: StudioSceneNode[] = [];
  const visited = new Set<string>();
  const visit = (parentId: string | null) => {
    for (const node of orderedChildren(scene, parentId)) {
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

function effectiveNodeState(node: StudioSceneNode, byId: Map<string, StudioSceneNode>) {
  let visible = node.visible;
  let opacity = clamp01(node.opacity);
  let parentId = node.parent_id;
  const visited = new Set([node.id]);
  while (parentId) {
    if (visited.has(parentId)) break;
    visited.add(parentId);
    const parent = byId.get(parentId);
    if (!parent) break;
    visible = visible && parent.visible;
    opacity *= clamp01(parent.opacity);
    parentId = parent.parent_id;
  }
  return { visible, opacity: clamp01(opacity) };
}

function rasterSourcePath(page: StudioPage, node: StudioSceneNode) {
  if (!node.image_layer_key) return null;
  if (node.image_layer_key === "mask" || node.image_layer_key === "rendered") return null;
  const source = page.image_layers[node.image_layer_key]?.path;
  if (source) return source;
  if (node.image_layer_key === "base") return page.arquivo_original ?? null;
  return null;
}

function generatedSourcePath(node: StudioSceneNode) {
  const source = node.metadata.image_path;
  return typeof source === "string" && source.trim() ? source : null;
}

function fillColor(node: StudioSceneNode) {
  const source = node.metadata.color ?? node.metadata.fill_color;
  return typeof source === "string" && source.trim() ? source : null;
}

function masksForNode(node: StudioSceneNode, byId: Map<string, StudioSceneNode>): ResolvedStudioSceneMask[] {
  return node.mask_ids.flatMap((maskId) => {
    const mask = byId.get(maskId);
    const selection = mask?.metadata.selection;
    if (!mask || mask.kind !== "mask" || !mask.visible || !isStudioSelection(selection)) return [];
    return [{
      nodeId: mask.id,
      name: mask.name,
      opacity: clamp01(mask.opacity),
      selection,
    }];
  });
}

export function resolveStudioSceneMasksForNode(scene: StudioScene, nodeId: string) {
  const byId = new Map(scene.nodes.map((node) => [node.id, node]));
  const node = byId.get(nodeId);
  return node ? masksForNode(node, byId) : [];
}

export function resolveStudioSceneRenderLayers(
  page: StudioPage,
  scene: StudioScene = page.studio_scene,
  options: ResolveStudioSceneRenderLayerOptions = {},
): ResolvedStudioSceneRenderLayer[] {
  const byId = new Map(scene.nodes.map((node) => [node.id, node]));
  return nodesInVisualOrder(scene).flatMap((node) => {
    if (node.kind !== "raster" && node.kind !== "generated" && node.kind !== "fill") return [];
    const state = effectiveNodeState(node, byId);
    if (!state.visible && !options.includeHidden) return [];
    const sourcePath = node.kind === "raster" ? rasterSourcePath(page, node) : generatedSourcePath(node);
    const color = node.kind === "fill" ? fillColor(node) : null;
    if (!sourcePath && !color) return [];
    return [{
      nodeId: node.id,
      name: node.name,
      kind: node.kind,
      ...(node.image_layer_key ? { imageLayerKey: node.image_layer_key } : {}),
      sourcePath,
      fillColor: color,
      visible: state.visible,
      opacity: state.opacity,
      blendMode: node.blend_mode || "normal",
      masks: masksForNode(node, byId),
    }];
  });
}

export function resolveStudioSceneVisualOrder(
  page: StudioPage,
  scene: StudioScene = page.studio_scene,
  options: ResolveStudioSceneRenderLayerOptions = {},
): ResolvedStudioSceneVisualItem[] {
  const byId = new Map(scene.nodes.map((node) => [node.id, node]));
  const renderableIds = new Set(
    resolveStudioSceneRenderLayers(page, scene, options).map((layer) => layer.nodeId),
  );
  const textLayerIds = new Set(page.text_layers.map((layer) => layer.id));
  return nodesInVisualOrder(scene).flatMap<ResolvedStudioSceneVisualItem>((node) => {
    const state = effectiveNodeState(node, byId);
    if (!state.visible && !options.includeHidden) return [];
    if (renderableIds.has(node.id)) return [{ kind: "bitmap" as const, nodeId: node.id }];
    if (node.kind === "text" && node.text_layer_id && textLayerIds.has(node.text_layer_id)) {
      return [{ kind: "text" as const, nodeId: node.id, textLayerId: node.text_layer_id }];
    }
    return [];
  });
}

export function resolveStudioAssetPath(projectPath: string, assetPath: string) {
  if (/^(data|blob|asset|file):/i.test(assetPath) || /^https?:\/\//i.test(assetPath)) return assetPath;
  if (/^[A-Za-z]:[\\/]/.test(assetPath) || assetPath.startsWith("/")) return assetPath;
  if (projectPath.startsWith("memory://")) return assetPath;
  const normalizedProject = projectPath.replace(/\\/g, "/");
  const directory = normalizedProject.toLowerCase().endsWith(".json")
    ? normalizedProject.slice(0, normalizedProject.lastIndexOf("/"))
    : normalizedProject;
  return `${directory}/${assetPath.replace(/\\/g, "/")}`;
}

function canvasBlendMode(blendMode: string) {
  if (blendMode === "normal") return "source-over";
  const supported = new Set([
    "multiply",
    "screen",
    "overlay",
    "darken",
    "lighten",
    "color-dodge",
    "color-burn",
    "hard-light",
    "soft-light",
    "difference",
    "exclusion",
    "hue",
    "saturation",
    "color",
    "luminosity",
  ]);
  return supported.has(blendMode) ? blendMode : "source-over";
}

function defaultCreateCanvas(width: number, height: number) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas as unknown as StudioCanvasLike;
}

async function defaultLoadImage(path: string): Promise<LoadedStudioImage> {
  const loaded = await loadImageSource(path, "image/png");
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.decoding = "async";
    image.onload = () => resolve({
      image,
      width: image.naturalWidth,
      height: image.naturalHeight,
      revoke: loaded.revoke,
    });
    image.onerror = () => {
      loaded.revoke?.();
      reject(new Error(`Falha ao carregar camada do Studio: ${path}`));
    };
    image.src = loaded.src;
  });
}

function renderStudioBitmapLayer(
  layer: ResolvedStudioSceneRenderLayer,
  width: number,
  height: number,
  createCanvas: (width: number, height: number) => StudioCanvasLike,
  loadedImages: Map<string, LoadedStudioImage>,
  rasterizeSelection: (selection: StudioSelection) => unknown,
) {
  const layerCanvas = createCanvas(width, height);
  const layerContext = layerCanvas.getContext("2d");
  if (!layerContext) throw new Error(`Canvas 2D indisponivel para a camada ${layer.name}`);
  layerContext.clearRect(0, 0, width, height);
  if (layer.fillColor) {
    layerContext.fillStyle = layer.fillColor;
    layerContext.fillRect(0, 0, width, height);
  }
  const loaded = layer.sourcePath ? loadedImages.get(layer.sourcePath) : null;
  if (loaded) layerContext.drawImage(loaded.image, 0, 0, width, height);
  for (const mask of layer.masks) {
    const maskImage = rasterizeSelection(mask.selection);
    if (!maskImage) continue;
    layerContext.save();
    layerContext.globalAlpha = mask.opacity;
    layerContext.globalCompositeOperation = "destination-in";
    layerContext.drawImage(maskImage, 0, 0, width, height);
    layerContext.restore();
  }
  return layerCanvas;
}

export async function composeStudioSceneLayerBitmaps(
  options: ComposeStudioSceneBitmapOptions,
): Promise<ComposedStudioSceneBitmapLayers> {
  const createCanvas = options.createCanvas ?? defaultCreateCanvas;
  const loadImage = options.loadImage ?? defaultLoadImage;
  const rasterizeSelection = options.rasterizeSelection ?? rasterizeLassoSelectionToCanvas;
  const resolveSourcePath = options.resolveSourcePath ?? ((path: string) => path);
  const layers = resolveStudioSceneRenderLayers(options.page, options.scene);
  const loadedImages = new Map<string, LoadedStudioImage>();

  try {
    for (const layer of layers) {
      if (!layer.sourcePath || loadedImages.has(layer.sourcePath)) continue;
      loadedImages.set(layer.sourcePath, await loadImage(resolveSourcePath(layer.sourcePath)));
    }
    const firstImage = loadedImages.values().next().value as LoadedStudioImage | undefined;
    const width = Math.max(1, Math.round(options.width ?? firstImage?.width ?? 1));
    const height = Math.max(1, Math.round(options.height ?? firstImage?.height ?? 1));
    return {
      width,
      height,
      layers: layers.map((layer) => ({
        nodeId: layer.nodeId,
        canvas: renderStudioBitmapLayer(layer, width, height, createCanvas, loadedImages, rasterizeSelection),
        opacity: layer.opacity,
        blendMode: layer.blendMode,
      })),
    };
  } finally {
    for (const loaded of loadedImages.values()) loaded.revoke?.();
  }
}

export function compositeStudioSceneLayerBitmaps(
  rendered: ComposedStudioSceneBitmapLayers,
  createCanvas: (width: number, height: number) => StudioCanvasLike = defaultCreateCanvas,
) {
  const output = createCanvas(rendered.width, rendered.height);
  const outputContext = output.getContext("2d");
  if (!outputContext) throw new Error("Canvas 2D indisponivel para compor a cena do Studio");
  outputContext.clearRect(0, 0, rendered.width, rendered.height);
  for (const layer of rendered.layers) {
    outputContext.save();
    outputContext.globalAlpha = layer.opacity;
    outputContext.globalCompositeOperation = canvasBlendMode(layer.blendMode);
    outputContext.drawImage(layer.canvas, 0, 0, rendered.width, rendered.height);
    outputContext.restore();
  }
  return output;
}

export async function composeStudioSceneBitmap(options: ComposeStudioSceneBitmapOptions) {
  const rendered = await composeStudioSceneLayerBitmaps(options);
  return compositeStudioSceneLayerBitmaps(rendered, options.createCanvas ?? defaultCreateCanvas);
}

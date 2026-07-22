import {
  COMPAT_PROJECT_VERSION,
  STUDIO_SCENE_VERSION,
  STUDIO_SCHEMA_VERSION,
  type ImageLayerKey,
  type ProjectImportKind,
  type ProjectImportResult,
  type StudioImageLayer,
  type StudioPage,
  type StudioProject,
  type StudioScene,
  type StudioSceneNode,
  type StudioSceneNodeKind,
  type StudioTextLayer,
  type StudioTextStyle,
} from "./studioProject";

const IMAGE_LAYER_KEYS: ImageLayerKey[] = ["base", "mask", "inpaint", "brush", "recovery", "rendered"];
const STUDIO_SCENE_NODE_KINDS: StudioSceneNodeKind[] = [
  "raster",
  "text",
  "group",
  "mask",
  "generated",
  "adjustment",
  "fill",
];
const IMAGE_LAYER_NAMES: Record<ImageLayerKey, string> = {
  base: "Original",
  mask: "Máscara",
  inpaint: "Limpeza",
  brush: "Pintura",
  recovery: "Recuperação",
  rendered: "Resultado",
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asString(value: unknown, fallback = "") {
  return typeof value === "string" ? value : fallback;
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function clampOpacity(value: unknown, fallback = 1) {
  const opacity = asNumber(value) ?? fallback;
  return Math.min(1, Math.max(0, opacity));
}

function isImageLayerKey(value: unknown): value is ImageLayerKey {
  return typeof value === "string" && IMAGE_LAYER_KEYS.includes(value as ImageLayerKey);
}

function isStudioSceneNodeKind(value: unknown): value is StudioSceneNodeKind {
  return typeof value === "string" && STUDIO_SCENE_NODE_KINDS.includes(value as StudioSceneNodeKind);
}

function asStringArray(value: unknown) {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function asBBox(value: unknown): [number, number, number, number] {
  if (Array.isArray(value) && value.length >= 4) {
    const nums = value.slice(0, 4).map((item) => (typeof item === "number" ? item : Number(item)));
    if (nums.every((item) => Number.isFinite(item))) {
      return [nums[0], nums[1], nums[2], nums[3]];
    }
  }
  return [0, 0, 1, 1];
}

function textStyleFrom(value: unknown): StudioTextStyle {
  return isRecord(value) ? { ...value } : {};
}

function normalizeImageLayer(key: ImageLayerKey, value: unknown, fallbackPath?: string | null): StudioImageLayer {
  const layer = isRecord(value) ? value : {};
  const rawPath = typeof layer.path === "string" ? layer.path : undefined;
  return {
    ...layer,
    key,
    path: rawPath ?? fallbackPath ?? null,
    visible: layer.visible === undefined ? true : layer.visible !== false,
    locked: layer.locked === true,
    opacity: asNumber(layer.opacity) ?? undefined,
    order: asNumber(layer.order) ?? undefined,
    technical: layer.technical === true || undefined,
  };
}

function syncLayerAliases(layer: StudioTextLayer): StudioTextLayer {
  const translated = layer.translated || layer.traduzido || "";
  const style = Object.keys(layer.style ?? {}).length > 0 ? layer.style : layer.estilo ?? {};
  const original = layer.original || asString(layer.texto ?? layer.text, "");
  return {
    ...layer,
    kind: "text",
    original,
    texto: asString(layer.texto, original),
    translated,
    traduzido: translated,
    style,
    estilo: style,
  };
}

function normalizeTextLayer(value: unknown, index: number): StudioTextLayer {
  const layer = isRecord(value) ? value : {};
  const confidence = asNumber(layer.ocr_confidence) ?? asNumber(layer.confianca_ocr) ?? asNumber(layer.confidence);
  const style = textStyleFrom(layer.style ?? layer.estilo);
  return syncLayerAliases({
    ...layer,
    id: asString(layer.id, `text-${index + 1}`),
    kind: "text",
    original: asString(layer.original ?? layer.texto ?? layer.text, ""),
    translated: asString(layer.translated ?? layer.traduzido, ""),
    traduzido: asString(layer.traduzido ?? layer.translated, ""),
    bbox: asBBox(layer.render_bbox ?? layer.layout_bbox ?? layer.bbox ?? layer.source_bbox ?? layer.balloon_bbox),
    source_bbox: Array.isArray(layer.source_bbox) ? asBBox(layer.source_bbox) : undefined,
    layout_bbox: Array.isArray(layer.layout_bbox) ? asBBox(layer.layout_bbox) : undefined,
    render_bbox: Array.isArray(layer.render_bbox) ? asBBox(layer.render_bbox) : undefined,
    tipo: asString(layer.tipo, "fala"),
    style,
    estilo: style,
    visible: layer.visible === undefined ? true : layer.visible !== false,
    locked: layer.locked === true,
    order: asNumber(layer.order) ?? index,
    ocr_confidence: confidence,
    confianca_ocr: confidence,
    qa_flags: Array.isArray(layer.qa_flags) ? layer.qa_flags : undefined,
  });
}

function sceneTextName(layer: StudioTextLayer, index: number) {
  const name = (layer.translated || layer.original || `Texto ${index + 1}`).replace(/\s+/g, " ").trim();
  return name.length > 48 ? `${name.slice(0, 45)}...` : name;
}

function deriveSceneNodes(
  imageLayers: Partial<Record<ImageLayerKey, StudioImageLayer>>,
  textLayers: StudioTextLayer[],
): StudioSceneNode[] {
  const imageNodes = IMAGE_LAYER_KEYS.map((key, index) => ({ key, index, layer: imageLayers[key] }))
    .filter((entry): entry is { key: ImageLayerKey; index: number; layer: StudioImageLayer } => Boolean(entry.layer?.path))
    .sort((left, right) => {
      const leftOrder = left.layer.order ?? left.index;
      const rightOrder = right.layer.order ?? right.index;
      return leftOrder - rightOrder || left.index - right.index;
    })
    .map(({ key, layer }) => ({
      id: `image:${key}`,
      kind: key === "mask" ? ("mask" as const) : ("raster" as const),
      name: IMAGE_LAYER_NAMES[key],
      visible: layer.visible,
      locked: layer.locked,
      opacity: clampOpacity(layer.opacity),
      blend_mode: "normal",
      parent_id: null,
      order: 0,
      mask_ids: [],
      image_layer_key: key,
      metadata: {
        projected_from: "image_layers",
        ...(layer.technical ? { technical: true } : {}),
      },
    }));

  const textNodes = [...textLayers]
    .map((layer, index) => ({ layer, index }))
    .sort((left, right) => left.layer.order - right.layer.order || left.index - right.index)
    .map(({ layer, index }) => ({
      id: `text:${layer.id}`,
      kind: "text" as const,
      name: sceneTextName(layer, index),
      visible: layer.visible,
      locked: layer.locked,
      opacity: clampOpacity(layer.opacity),
      blend_mode: "normal",
      parent_id: null,
      order: 0,
      mask_ids: [],
      text_layer_id: layer.id,
      metadata: { projected_from: "text_layers" },
    }));

  return [...imageNodes, ...textNodes].map((node, order) => ({ ...node, order }));
}

function normalizeSceneNode(value: unknown, index: number): StudioSceneNode {
  if (!isRecord(value)) {
    throw new Error(`studio_scene.nodes[${index}] must be an object`);
  }
  const id = asString(value.id).trim();
  if (!id) {
    throw new Error(`studio_scene.nodes[${index}] must have an id`);
  }
  if (!isStudioSceneNodeKind(value.kind)) {
    throw new Error(`Unsupported studio_scene node kind: ${asString(value.kind, "missing")}`);
  }
  const parentId = typeof value.parent_id === "string" ? value.parent_id : typeof value.parentId === "string" ? value.parentId : null;
  const imageLayerKey = isImageLayerKey(value.image_layer_key) ? value.image_layer_key : undefined;
  const textLayerId = typeof value.text_layer_id === "string" ? value.text_layer_id : undefined;
  return {
    ...value,
    id,
    kind: value.kind,
    name: asString(value.name, id),
    visible: value.visible === undefined ? true : value.visible !== false,
    locked: value.locked === true,
    opacity: clampOpacity(value.opacity),
    blend_mode: asString(value.blend_mode ?? value.blendMode, "normal"),
    parent_id: parentId,
    order: asNumber(value.order) ?? index,
    mask_ids: asStringArray(value.mask_ids ?? value.maskIds),
    ...(imageLayerKey ? { image_layer_key: imageLayerKey } : {}),
    ...(textLayerId ? { text_layer_id: textLayerId } : {}),
    metadata: isRecord(value.metadata) ? { ...value.metadata } : {},
  };
}

function sameProjectionSource(left: StudioSceneNode, right: StudioSceneNode) {
  if (left.image_layer_key && right.image_layer_key) {
    return left.image_layer_key === right.image_layer_key;
  }
  if (left.text_layer_id && right.text_layer_id) {
    return left.text_layer_id === right.text_layer_id;
  }
  return left.id === right.id;
}

function normalizeStudioScene(
  value: unknown,
  imageLayers: Partial<Record<ImageLayerKey, StudioImageLayer>>,
  textLayers: StudioTextLayer[],
): StudioScene {
  const projectedNodes = deriveSceneNodes(imageLayers, textLayers);
  if (!isRecord(value)) {
    return {
      version: STUDIO_SCENE_VERSION,
      roots: projectedNodes.map((node) => node.id),
      nodes: projectedNodes,
      metadata: { projection_source: "traduzai_v2" },
    };
  }
  if (value.version !== undefined && value.version !== STUDIO_SCENE_VERSION) {
    throw new Error(`Unsupported studio_scene version: ${String(value.version)}`);
  }
  if (!Array.isArray(value.nodes)) {
    throw new Error("studio_scene.nodes must be an array");
  }

  const projectedImageKeys = new Set(
    projectedNodes.map((node) => node.image_layer_key).filter((key): key is ImageLayerKey => key !== undefined),
  );
  const projectedTextIds = new Set(
    projectedNodes.map((node) => node.text_layer_id).filter((id): id is string => id !== undefined),
  );
  const nodes = value.nodes.map(normalizeSceneNode).filter((node) => {
    if (node.metadata.projected_from === "image_layers" && node.image_layer_key) {
      return projectedImageKeys.has(node.image_layer_key);
    }
    if (node.metadata.projected_from === "text_layers" && node.text_layer_id) {
      return projectedTextIds.has(node.text_layer_id);
    }
    return true;
  });
  const nodeIds = new Set<string>();
  for (const node of nodes) {
    if (nodeIds.has(node.id)) {
      throw new Error(`Duplicate studio_scene node id: ${node.id}`);
    }
    nodeIds.add(node.id);
  }

  for (const projectedNode of projectedNodes) {
    const existingIndex = nodes.findIndex((node) => sameProjectionSource(node, projectedNode));
    if (existingIndex < 0) {
      nodes.push(projectedNode);
      nodeIds.add(projectedNode.id);
      continue;
    }
    const existingNode = nodes[existingIndex];
    if (existingNode.metadata.scene_owned === true) continue;
    nodes[existingIndex] = {
      ...existingNode,
      visible: projectedNode.visible,
      locked: projectedNode.locked,
      opacity: projectedNode.opacity,
    };
  }

  const roots: string[] = [];
  for (const id of asStringArray(value.roots)) {
    if (nodeIds.has(id) && !roots.includes(id)) roots.push(id);
  }
  for (const node of nodes) {
    if (node.parent_id === null && !roots.includes(node.id)) roots.push(node.id);
  }

  return {
    ...value,
    version: STUDIO_SCENE_VERSION,
    roots,
    nodes,
    metadata: isRecord(value.metadata) ? { ...value.metadata } : undefined,
  };
}

function normalizePage(value: unknown, index: number): StudioPage {
  const page = isRecord(value) ? value : {};
  const imageLayers = isRecord(page.image_layers) ? page.image_layers : {};
  const originalPath = asString(page.arquivo_original ?? page.original_path, null as never) || null;
  const translatedPath =
    asString(page.arquivo_traduzido ?? page.rendered_path ?? page.translated_path, null as never) || null;
  const normalizedLayers: Partial<Record<ImageLayerKey, StudioImageLayer>> = {};
  for (const key of IMAGE_LAYER_KEYS) {
    const fallback = key === "base" ? originalPath : key === "rendered" ? translatedPath : null;
    normalizedLayers[key] = normalizeImageLayer(key, imageLayers[key], fallback);
  }

  const rawTextLayers = Array.isArray(page.text_layers)
    ? page.text_layers
    : Array.isArray(page.textos)
      ? page.textos
      : [];
  const textLayers = rawTextLayers.map(normalizeTextLayer);
  const numero = asNumber(page.numero) ?? index + 1;
  return {
    ...page,
    numero,
    arquivo_original: originalPath ?? normalizedLayers.base?.path ?? null,
    arquivo_traduzido: translatedPath ?? normalizedLayers.rendered?.path ?? null,
    image_layers: normalizedLayers,
    text_layers: textLayers,
    textos: textLayers,
    studio_scene: normalizeStudioScene(page.studio_scene, normalizedLayers, textLayers),
  };
}

function normalizeBaseProject(value: unknown, kind: ProjectImportKind): ProjectImportResult {
  if (!isRecord(value)) {
    throw new Error("Project payload must be an object");
  }
  const pages = Array.isArray(value.paginas) ? value.paginas : [];
  const project: StudioProject = {
    ...value,
    app: "traduzai",
    versao: COMPAT_PROJECT_VERSION,
    studio_schema_version: STUDIO_SCHEMA_VERSION,
    source_project_metadata: {
      app: value.app,
      versao: value.versao,
      schema_version: value.schema_version,
    },
    paginas: pages.map(normalizePage),
  };
  return { kind, project, warnings: [] };
}

function normalizeV12Project(value: Record<string, unknown>): ProjectImportResult {
  const warnings: string[] = [];
  const legacy = isRecord(value.legacy) ? value.legacy : {};
  if (Array.isArray(legacy.paginas)) {
    const result = normalizeBaseProject({ ...value, paginas: legacy.paginas }, "v12_analysis_project");
    result.warnings.push("Imported v12 project from legacy.paginas");
    return result;
  }

  const rawPages = Array.isArray(value.pages) ? value.pages : [];
  const paginas = rawPages.map((rawPage, pageIndex) => {
    const page = isRecord(rawPage) ? rawPage : {};
    const regions = Array.isArray(page.regions) ? page.regions : [];
    return {
      numero: asNumber(page.number) ?? asNumber(page.numero) ?? pageIndex + 1,
      arquivo_original: asString(page.source_path ?? page.image_path, ""),
      arquivo_traduzido: asString(page.rendered_path ?? page.translated_path, ""),
      textos: regions.map((region, regionIndex) => {
        const item = isRecord(region) ? region : {};
        const translation = isRecord(item.translation) ? item.translation : {};
        return {
          ...item,
          id: asString(item.id, `p${pageIndex + 1}-r${regionIndex + 1}`),
          bbox: item.bbox,
          texto: item.raw_ocr ?? item.normalized_ocr ?? item.text,
          traduzido: translation.text ?? item.translated,
          tipo: item.tipo ?? item.kind,
          confidence: item.confidence,
          estilo: item.style ?? item.layout,
          v12_region: item,
        };
      }),
    };
  });
  warnings.push("Imported v12 project from pages[].regions; some pipeline metadata may be projected only");
  const result = normalizeBaseProject({ ...value, paginas }, "v12_analysis_project");
  result.warnings.push(...warnings);
  return result;
}

export function importStudioProject(value: unknown): ProjectImportResult {
  if (!isRecord(value)) {
    throw new Error("Project payload must be an object");
  }
  if (value.schema_version === "12.0" || value.schema_version === 12) {
    return normalizeV12Project(value);
  }
  if (value.studio_schema_version === STUDIO_SCHEMA_VERSION) {
    return normalizeBaseProject(value, "studio_project");
  }
  if (value.versao === "1.0") {
    return normalizeBaseProject(value, "traduzai_v1");
  }
  return normalizeBaseProject(value, "traduzai_v2");
}

export function toTraduzAiV2Compat(project: StudioProject): Record<string, unknown> {
  const paginas = project.paginas.map((page) => {
    const textLayers = page.text_layers.map(syncLayerAliases);
    const basePath = page.image_layers.base?.path ?? page.arquivo_original ?? asString(page.original_path, "") ?? null;
    const renderedPath =
      page.image_layers.rendered?.path ??
      page.arquivo_traduzido ??
      asString(page.rendered_path, "") ??
      asString(page.translated_path, "") ??
      null;
    const inpaintPath =
      page.image_layers.inpaint?.path ?? asString(page.inpaint_path, "") ?? asString(page.arquivo_final, "") ?? null;
    return {
      ...page,
      arquivo_original: basePath,
      arquivo_traduzido: renderedPath,
      arquivo_final: inpaintPath ?? page.arquivo_final ?? renderedPath,
      original_path: basePath,
      rendered_path: renderedPath,
      translated_path: renderedPath,
      inpaint_path: inpaintPath,
      text_layers: textLayers,
      textos: textLayers,
      studio_scene: normalizeStudioScene(page.studio_scene, page.image_layers, textLayers),
    };
  });
  return {
    ...project,
    app: "traduzai",
    versao: COMPAT_PROJECT_VERSION,
    paginas,
  };
}

export function finalImagePathForPage(page: Pick<StudioPage, "arquivo_traduzido" | "image_layers"> & Record<string, unknown>) {
  return (
    asString(page.arquivo_traduzido, "") ||
    asString(page.rendered_path, "") ||
    asString(page.translated_path, "") ||
    page.image_layers.rendered?.path ||
    page.image_layers.inpaint?.path ||
    page.image_layers.base?.path ||
    null
  );
}

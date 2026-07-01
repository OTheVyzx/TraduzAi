import {
  COMPAT_PROJECT_VERSION,
  STUDIO_SCHEMA_VERSION,
  type ImageLayerKey,
  type ProjectImportKind,
  type ProjectImportResult,
  type StudioImageLayer,
  type StudioPage,
  type StudioProject,
  type StudioTextLayer,
  type StudioTextStyle,
} from "./studioProject";

const IMAGE_LAYER_KEYS: ImageLayerKey[] = ["base", "mask", "inpaint", "brush", "recovery", "rendered"];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asString(value: unknown, fallback = "") {
  return typeof value === "string" ? value : fallback;
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
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
  return {
    ...page,
    numero: asNumber(page.numero) ?? index + 1,
    arquivo_original: originalPath ?? normalizedLayers.base?.path ?? null,
    arquivo_traduzido: translatedPath ?? normalizedLayers.rendered?.path ?? null,
    image_layers: normalizedLayers,
    text_layers: textLayers,
    textos: textLayers,
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

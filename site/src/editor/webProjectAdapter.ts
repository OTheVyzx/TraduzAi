import type { ImageLayerKey, PageData, Project, TextEntry, TextLayerStyle } from "../../../src/lib/stores/appStore";
import { DEFAULT_TEXT_STYLE, canonicalizeTextStyle } from "../../../src/lib/editorTextStylePolicy";
import { normalizeWebProjectQuality } from "../projectConfig";

export const WEB_PROJECT_PATH_PREFIX = "web-project:";

const IMAGE_LAYER_KEYS: ImageLayerKey[] = ["base", "mask", "inpaint", "brush", "recovery", "rendered"];

function isRecord(value: unknown): value is Record<string, any> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function encodeAssetPath(path: string) {
  return path
    .replace(/\\/g, "/")
    .split("/")
    .filter(Boolean)
    .map((segment) => encodeURIComponent(segment))
    .join("/");
}

function decodeAssetPath(path: string) {
  return path
    .split("/")
    .filter(Boolean)
    .map((segment) => decodeURIComponent(segment))
    .join("/");
}

export function webProjectPath(projectId: string) {
  return `${WEB_PROJECT_PATH_PREFIX}${projectId}`;
}

export function projectIdFromWebPath(projectPath: string) {
  if (!projectPath.startsWith(WEB_PROJECT_PATH_PREFIX)) {
    throw new Error("Projeto web invalido para backend HTTP");
  }
  return projectPath.slice(WEB_PROJECT_PATH_PREFIX.length);
}

export function assetPathToUrl(projectId: string, path?: string | null) {
  if (!path) return null;
  const normalized = path.replace(/\\/g, "/");
  if (/^(data|blob|file):/i.test(normalized) || /^https?:\/\//i.test(normalized)) return normalized;
  if (normalized.startsWith(`/api/projects/${projectId}/assets/`)) return normalized;
  if (normalized.startsWith("/api/")) return normalized;
  if (normalized.startsWith("/")) return normalized;
  return `/api/projects/${projectId}/assets/${encodeAssetPath(normalized)}`;
}

export function assetUrlToPath(projectId: string, path?: string | null) {
  if (!path) return path ?? null;
  const normalized = path.replace(/\\/g, "/");
  const marker = `/api/projects/${projectId}/assets/`;
  const markerIndex = normalized.indexOf(marker);
  if (markerIndex >= 0) {
    return decodeAssetPath(normalized.slice(markerIndex + marker.length));
  }
  return normalized;
}

function bboxFrom(value: unknown): [number, number, number, number] {
  if (Array.isArray(value) && value.length >= 4) {
    return [
      Number(value[0]) || 0,
      Number(value[1]) || 0,
      Number(value[2]) || 32,
      Number(value[3]) || 32,
    ];
  }
  return [0, 0, 32, 32];
}

function normalizeTextLayer(layer: unknown, index: number): TextEntry {
  const raw = isRecord(layer) ? layer : {};
  const bbox = bboxFrom(raw.render_bbox ?? raw.layout_bbox ?? raw.bbox ?? raw.source_bbox ?? raw.balloon_bbox);
  const style = canonicalizeTextStyle(
    { ...DEFAULT_TEXT_STYLE, ...(isRecord(raw.style) ? raw.style : {}), ...(isRecord(raw.estilo) ? raw.estilo : {}) },
    { mode: "hydrate" },
  ) as TextLayerStyle;
  return {
    ...raw,
    id: String(raw.id ?? `text-${index + 1}`),
    kind: "text",
    style_origin: raw.style_origin ?? "legacy",
    source_bbox: bboxFrom(raw.source_bbox ?? raw.bbox ?? bbox),
    layout_bbox: bboxFrom(raw.layout_bbox ?? bbox),
    render_bbox: Array.isArray(raw.render_bbox) ? bboxFrom(raw.render_bbox) : null,
    bbox,
    tipo: (raw.tipo ?? "fala") as TextEntry["tipo"],
    original: String(raw.original ?? ""),
    traduzido: String(raw.traduzido ?? raw.translated ?? ""),
    translated: String(raw.translated ?? raw.traduzido ?? ""),
    confianca_ocr: Number(raw.confianca_ocr ?? raw.ocr_confidence ?? 0),
    ocr_confidence: Number(raw.ocr_confidence ?? raw.confianca_ocr ?? 0),
    estilo: style,
    style,
    visible: raw.visible ?? true,
    locked: raw.locked ?? false,
    order: Number(raw.order ?? index),
    render_preview_path: typeof raw.render_preview_path === "string" ? raw.render_preview_path : null,
    detector: typeof raw.detector === "string" ? raw.detector : null,
    line_polygons: raw.line_polygons ?? null,
    source_direction: raw.source_direction ?? null,
    rendered_direction: raw.rendered_direction ?? null,
    source_language: raw.source_language ?? null,
    rotation_deg: Number(raw.rotation_deg ?? 0),
    detected_font_size_px: raw.detected_font_size_px ?? null,
    page_profile: raw.page_profile ?? null,
    block_profile: raw.block_profile ?? null,
    layout_profile: raw.layout_profile ?? raw.block_profile ?? null,
    balloon_bbox: bboxFrom(raw.balloon_bbox ?? bbox),
    balloon_subregions: Array.isArray(raw.balloon_subregions) ? raw.balloon_subregions : [],
    layout_group_size: Number(raw.layout_group_size ?? 1),
  };
}

function normalizeImageLayers(projectId: string, page: Record<string, any>) {
  const rawLayers = isRecord(page.image_layers) ? page.image_layers : {};
  const fallback: Partial<Record<ImageLayerKey, string | null | undefined>> = {
    base: page.arquivo_original ?? page.original_path,
    inpaint: page.inpaint_path,
    rendered: page.arquivo_traduzido ?? page.rendered_path ?? page.translated_path,
  };
  return Object.fromEntries(
    IMAGE_LAYER_KEYS.map((key) => {
      const rawLayer = isRecord(rawLayers[key]) ? rawLayers[key] : {};
      const path = assetPathToUrl(projectId, rawLayer.path ?? fallback[key] ?? null);
      return [
        key,
        {
          key,
          path,
          visible: rawLayer.visible ?? (key === "base" || key === "rendered"),
          locked: rawLayer.locked ?? (key === "base" || key === "rendered"),
          opacity: rawLayer.opacity,
          order: rawLayer.order,
          technical: rawLayer.technical,
        },
      ];
    }),
  ) as PageData["image_layers"];
}

export function normalizeWebPage(projectId: string, page: unknown, index: number): PageData {
  const raw = isRecord(page) ? page : {};
  const textLayers = ((Array.isArray(raw.text_layers) && raw.text_layers.length > 0 ? raw.text_layers : raw.textos) ?? [])
    .map((layer: unknown, layerIndex: number) => normalizeTextLayer(layer, layerIndex))
    .sort((a: TextEntry, b: TextEntry) => (a.order ?? 0) - (b.order ?? 0));
  const imageLayers = normalizeImageLayers(projectId, raw);
  return {
    ...raw,
    numero: Number(raw.numero ?? raw.index ?? index + 1),
    arquivo_original: imageLayers?.base?.path ?? assetPathToUrl(projectId, raw.arquivo_original ?? raw.original_path) ?? "",
    arquivo_traduzido:
      imageLayers?.rendered?.path ?? assetPathToUrl(projectId, raw.arquivo_traduzido ?? raw.rendered_path) ?? "",
    image_layers: imageLayers,
    inpaint_blocks: Array.isArray(raw.inpaint_blocks) ? raw.inpaint_blocks : [],
    text_layers: textLayers,
    textos: textLayers,
  };
}

export function normalizeWebProject(projectId: string, project: unknown): Project {
  const raw = isRecord(project) ? project : {};
  const pages = (Array.isArray(raw.paginas) ? raw.paginas : []).map((page, index) =>
    normalizeWebPage(projectId, page, index),
  );
  return {
    ...raw,
    id: String(raw.id ?? raw.job_id ?? projectId),
    obra: String(raw.obra ?? "Projeto sem nome"),
    capitulo: Number(raw.capitulo ?? 1),
    idioma_origem: String(raw.idioma_origem ?? raw.src_lang ?? "auto"),
    idioma_destino: String(raw.idioma_destino ?? raw.dst_lang ?? "pt-BR"),
    qualidade: normalizeWebProjectQuality(raw.pipeline_quality ?? raw.qualidade),
    contexto: isRecord(raw.contexto)
      ? raw.contexto as Project["contexto"]
      : {
          sinopse: "",
          genero: [],
          personagens: [],
          glossario: {},
          aliases: [],
          termos: [],
          relacoes: [],
          faccoes: [],
          resumo_por_arco: [],
          memoria_lexical: {},
          fontes_usadas: [],
        },
    paginas: pages,
    status: (raw.status ?? "done") as Project["status"],
    source_path: webProjectPath(projectId),
    output_path: webProjectPath(projectId),
    totalPages: Number(raw.totalPages ?? pages.length),
    mode: (raw.mode ?? "manual") as Project["mode"],
    _work_dir: webProjectPath(projectId),
  } as Project & { _work_dir: string };
}

function denormalizePage(projectId: string, page: PageData): PageData {
  const imageLayers = Object.fromEntries(
    Object.entries(page.image_layers ?? {}).map(([key, layer]) => [
      key,
      layer ? { ...layer, path: assetUrlToPath(projectId, layer.path) } : layer,
    ]),
  ) as PageData["image_layers"];
  const textLayers = page.text_layers.map((layer) => ({
    ...layer,
    style: layer.style ?? layer.estilo,
    estilo: layer.estilo ?? layer.style,
  }));
  return {
    ...page,
    arquivo_original: assetUrlToPath(projectId, page.arquivo_original) ?? "",
    arquivo_traduzido: assetUrlToPath(projectId, page.arquivo_traduzido) ?? "",
    image_layers: imageLayers,
    text_layers: textLayers,
    textos: textLayers,
  };
}

export function denormalizeWebProject(projectId: string, project: unknown) {
  const raw = isRecord(project) ? { ...project } : {};
  const pipelineQuality = normalizeWebProjectQuality(raw.pipeline_quality ?? raw.qualidade);
  raw.qualidade = pipelineQuality;
  raw.pipeline_quality = pipelineQuality;
  raw.paginas = (Array.isArray(raw.paginas) ? raw.paginas : []).map((page: PageData) =>
    denormalizePage(projectId, page),
  );
  if (raw.source_path === webProjectPath(projectId)) delete raw.source_path;
  if (raw.output_path === webProjectPath(projectId)) delete raw.output_path;
  if (raw._work_dir === webProjectPath(projectId)) delete raw._work_dir;
  return raw;
}

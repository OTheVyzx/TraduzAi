import { useAppStore, type PageData, type Project, type TextEntry, type TextLayerStyle } from "../stores/appStore";
import { useEditorStore } from "../stores/editorStore";
import originalImageUrl from "../../../e2e/fixtures/images/page-001-original.png";
import inpaintImageUrl from "../../../e2e/fixtures/images/page-001-inpaint.png";

declare global {
  interface Window {
    __TRADUZAI_E2E_PROJECT__?: unknown;
    __TRADUZAI_E2E_BATCH_SOURCES__?: unknown;
  }
}

function buildFixtureProject(): Project {
  const textLayer = {
    id: "fixture-text-1",
    kind: "text" as const,
    source_bbox: [118, 166, 302, 252] as [number, number, number, number],
    layout_bbox: [118, 166, 302, 252] as [number, number, number, number],
    render_bbox: null,
    bbox: [118, 166, 302, 252] as [number, number, number, number],
    tipo: "fala" as const,
    original: "BURNED",
    traduzido: "TEXTO LIMPO",
    translated: "TEXTO LIMPO",
    confianca_ocr: 0.98,
    ocr_confidence: 0.98,
    estilo: {
      fonte: "ComicNeue-Bold.ttf",
      tamanho: 28,
      cor: "#ffffff",
      cor_gradiente: [],
      contorno: "#111111",
      contorno_px: 2,
      glow: false,
      glow_cor: "",
      glow_px: 0,
      sombra: false,
      sombra_cor: "",
      sombra_offset: [0, 0] as [number, number],
      bold: true,
      italico: false,
      rotacao: 0,
      alinhamento: "center" as const,
      force_upper: false,
    },
    visible: true,
    locked: false,
    order: 0,
    render_preview_path: null,
    detector: null,
    line_polygons: null,
    source_direction: null,
    rendered_direction: null,
    source_language: null,
    rotation_deg: 0,
    detected_font_size_px: null,
    balloon_bbox: [118, 166, 302, 252] as [number, number, number, number],
    balloon_subregions: [],
    layout_group_size: 1,
    qa_flags: ["visual_text_leak"],
    qa_actions: [],
  };

  return {
    id: "fixture-project",
    obra: "Fixture E2E",
    capitulo: 1,
    idioma_origem: "en",
    idioma_destino: "pt-BR",
    qualidade: "normal",
    contexto: {
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
    paginas: [
      {
        numero: 1,
        arquivo_original: originalImageUrl,
        arquivo_traduzido: "/e2e-rendered-stale.png",
        image_layers: {
          base: { key: "base", path: originalImageUrl, visible: true, locked: true },
          inpaint: { key: "inpaint", path: inpaintImageUrl, visible: true, locked: true },
          rendered: { key: "rendered", path: "/e2e-rendered-stale.png", visible: true, locked: true },
        },
        inpaint_blocks: [],
        text_layers: [textLayer],
        textos: [textLayer],
      },
    ],
    status: "done",
    source_path: "e2e/project-basic.json",
    output_path: "e2e/project-basic.json",
    totalPages: 1,
    mode: "manual",
  };
}

let fixtureProject = buildFixtureProject();

const defaultContext = {
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
};

function toObject(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : {};
}

function tuple4(value: unknown, fallback: [number, number, number, number]): [number, number, number, number] {
  if (!Array.isArray(value) || value.length < 4) return fallback;
  const next = value.slice(0, 4).map((item) => Number(item));
  if (next.some((item) => !Number.isFinite(item))) return fallback;
  return next as [number, number, number, number];
}

function tuple2(value: unknown, fallback: [number, number]): [number, number] {
  if (!Array.isArray(value) || value.length < 2) return fallback;
  const next = value.slice(0, 2).map((item) => Number(item));
  if (next.some((item) => !Number.isFinite(item))) return fallback;
  return next as [number, number];
}

function normalizeE2EStyle(value: unknown): TextLayerStyle {
  const style = toObject(value);
  return {
    fonte: String(style.fonte ?? "ComicNeue-Bold.ttf"),
    tamanho: Number(style.tamanho ?? 28),
    cor: String(style.cor ?? "#ffffff"),
    cor_gradiente: Array.isArray(style.cor_gradiente) ? style.cor_gradiente.map(String) : [],
    contorno: String(style.contorno ?? "#111111"),
    contorno_px: Number(style.contorno_px ?? 2),
    glow: Boolean(style.glow),
    glow_cor: String(style.glow_cor ?? ""),
    glow_px: Number(style.glow_px ?? 0),
    sombra: Boolean(style.sombra),
    sombra_cor: String(style.sombra_cor ?? ""),
    sombra_offset: tuple2(style.sombra_offset, [0, 0]),
    bold: Boolean(style.bold ?? true),
    italico: Boolean(style.italico),
    rotacao: Number(style.rotacao ?? 0),
    alinhamento: ["left", "center", "right"].includes(String(style.alinhamento))
      ? (String(style.alinhamento) as TextLayerStyle["alinhamento"])
      : "center",
    force_upper: Boolean(style.force_upper),
  };
}

function normalizeE2ETextLayer(value: unknown, index: number): TextEntry {
  const layer = toObject(value);
  const bbox = tuple4(layer.render_bbox ?? layer.layout_bbox ?? layer.bbox ?? layer.source_bbox ?? layer.balloon_bbox, [0, 0, 32, 32]);
  const style = normalizeE2EStyle(layer.style ?? layer.estilo);
  const translated = String(layer.traduzido ?? layer.translated ?? layer.text ?? layer.original ?? "");
  const tipo = ["fala", "narracao", "sfx", "pensamento"].includes(String(layer.tipo))
    ? (String(layer.tipo) as TextEntry["tipo"])
    : "fala";

  return {
    ...(layer as Partial<TextEntry>),
    id: String(layer.id ?? `real-generated-text-${index + 1}`),
    kind: "text",
    style_origin: (layer.style_origin as TextEntry["style_origin"]) ?? "legacy",
    source_bbox: tuple4(layer.source_bbox, bbox),
    layout_bbox: tuple4(layer.layout_bbox, bbox),
    render_bbox: layer.render_bbox === null ? null : tuple4(layer.render_bbox, bbox),
    bbox,
    tipo,
    original: String(layer.original ?? layer.text ?? ""),
    traduzido: translated,
    translated,
    confianca_ocr: Number(layer.confianca_ocr ?? layer.ocr_confidence ?? layer.confidence ?? 0),
    ocr_confidence: Number(layer.ocr_confidence ?? layer.confianca_ocr ?? layer.confidence ?? 0),
    estilo: style,
    style,
    visible: layer.visible !== false,
    locked: Boolean(layer.locked),
    order: Number(layer.order ?? layer.reading_order ?? index),
    render_preview_path: typeof layer.render_preview_path === "string" ? layer.render_preview_path : null,
    detector: typeof layer.detector === "string" ? layer.detector : null,
    line_polygons: layer.line_polygons ?? null,
    source_direction: typeof layer.source_direction === "string" ? layer.source_direction : null,
    rendered_direction: typeof layer.rendered_direction === "string" ? layer.rendered_direction : null,
    source_language: typeof layer.source_language === "string" ? layer.source_language : null,
    rotation_deg: Number(layer.rotation_deg ?? 0),
    detected_font_size_px: typeof layer.detected_font_size_px === "number" ? layer.detected_font_size_px : null,
    balloon_bbox: tuple4(layer.balloon_bbox, bbox),
    balloon_subregions: Array.isArray(layer.balloon_subregions)
      ? layer.balloon_subregions.map((item) => tuple4(item, bbox))
      : [],
    layout_group_size: Number(layer.layout_group_size ?? 1),
    qa_flags: Array.isArray(layer.qa_flags) ? layer.qa_flags.map(String) : [],
    qa_actions: Array.isArray(layer.qa_actions) ? (layer.qa_actions as TextEntry["qa_actions"]) : [],
  };
}

function normalizeE2EWorkContext(value: unknown): Project["work_context"] {
  const raw = toObject(value);
  if (Object.keys(raw).length === 0) return null;
  const coverUrl = typeof raw.cover_url === "string" ? raw.cover_url.trim() : "";
  return {
    selected: Boolean(raw.selected ?? raw.title),
    work_id: String(raw.work_id ?? ""),
    title: String(raw.title ?? ""),
    context_loaded: Boolean(raw.context_loaded),
    glossary_loaded: Boolean(raw.glossary_loaded),
    glossary_entries_count: Number(raw.glossary_entries_count ?? 0),
    internet_context_loaded: Boolean(raw.internet_context_loaded),
    ...(coverUrl ? { cover_url: coverUrl } : {}),
    risk_level: ["high", "medium", "low"].includes(String(raw.risk_level))
      ? (String(raw.risk_level) as NonNullable<Project["work_context"]>["risk_level"])
      : "high",
    user_ignored_warning: Boolean(raw.user_ignored_warning),
  };
}

function normalizeInjectedProject(value: unknown): Project | null {
  const raw = toObject(value);
  const rawPages = Array.isArray(raw.paginas) ? raw.paginas : [];
  if (rawPages.length === 0) return null;

  const paginas: PageData[] = rawPages.map((pageValue, pageIndex) => {
    const page = toObject(pageValue);
    const rawImageLayers = toObject(page.image_layers);
    const baseLayer = toObject(rawImageLayers.base);
    const inpaintLayer = toObject(rawImageLayers.inpaint);
    const renderedLayer = toObject(rawImageLayers.rendered);
    const basePath = typeof page.arquivo_original === "string" && page.arquivo_original.length > 0
      ? page.arquivo_original
      : typeof baseLayer.path === "string" && baseLayer.path.length > 0
        ? baseLayer.path
        : originalImageUrl;
    const inpaintPath = typeof page.arquivo_traduzido === "string" && page.arquivo_traduzido.length > 0
      ? page.arquivo_traduzido
      : typeof inpaintLayer.path === "string" && inpaintLayer.path.length > 0
        ? inpaintLayer.path
        : inpaintImageUrl;
    const renderedPath = typeof renderedLayer.path === "string" && renderedLayer.path.length > 0
      ? renderedLayer.path
      : inpaintPath;
    const rawLayers = Array.isArray(page.text_layers) && page.text_layers.length > 0
      ? page.text_layers
      : Array.isArray(page.textos)
        ? page.textos
        : [];
    const textLayers = rawLayers.map((layer, layerIndex) => normalizeE2ETextLayer(layer, layerIndex));
    return {
      numero: Number(page.numero ?? pageIndex + 1),
      arquivo_original: basePath,
      arquivo_traduzido: inpaintPath,
      image_layers: {
        base: { key: "base", path: basePath, visible: true, locked: true },
        inpaint: { key: "inpaint", path: inpaintPath, visible: true, locked: true },
        rendered: { key: "rendered", path: renderedPath, visible: true, locked: true },
      },
      inpaint_blocks: Array.isArray(page.inpaint_blocks) ? (page.inpaint_blocks as PageData["inpaint_blocks"]) : [],
      text_layers: textLayers,
      textos: textLayers,
    } satisfies PageData;
  });

  return {
    id: String(raw.id ?? "real-generated-project-e2e"),
    obra: String(raw.obra ?? "Projeto E2E"),
    capitulo: Number(raw.capitulo ?? 1),
    idioma_origem: String(raw.idioma_origem ?? "en"),
    idioma_destino: String(raw.idioma_destino ?? "pt-BR"),
    qualidade: "normal",
    contexto: { ...defaultContext, ...toObject(raw.contexto) },
    work_context: normalizeE2EWorkContext(raw.work_context),
    paginas,
    status: "done",
    source_path: String(raw.source_path ?? "e2e/real-generated-project.json"),
    output_path: String(raw.output_path ?? raw.source_path ?? "e2e/real-generated-project.json"),
    totalPages: paginas.length,
    mode: raw.mode === "manual" ? "manual" : "auto",
  };
}

function activateE2EProject(project: Project) {
  fixtureProject = project;
  useAppStore.getState().setProject(fixtureProject);
  useEditorStore.getState().resetEditor();
  useEditorStore.setState({
    currentPageIndex: 0,
    currentPage: fixtureProject.paginas[0],
    viewMode: "translated",
    zoom: 1,
    panOffset: { x: 0, y: 0 },
  });
}

export function getE2EFixtureProject() {
  return fixtureProject;
}

export function setE2EFixtureProject(project: Project) {
  fixtureProject = project;
}

export function resetE2EFixtureProject() {
  activateE2EProject(buildFixtureProject());
}

export function installE2EFixtureProject() {
  const injected = typeof window !== "undefined"
    ? normalizeInjectedProject(window.__TRADUZAI_E2E_PROJECT__)
    : null;
  if (injected) {
    activateE2EProject(injected);
  } else {
    resetE2EFixtureProject();
  }
  const batchSources = Array.isArray(window.__TRADUZAI_E2E_BATCH_SOURCES__)
    ? window.__TRADUZAI_E2E_BATCH_SOURCES__.filter((item): item is string => typeof item === "string" && item.length > 0)
    : [];
  if (batchSources.length > 0) {
    useAppStore.getState().setBatchSources(batchSources);
  }
  window.localStorage.setItem("traduzai_e2e_loaded", "1");
}

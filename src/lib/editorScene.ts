import type { ImageLayer, ImageLayerKey, PageData, TextEntry, TextLayerStyle } from "./stores/appStore";

export type EditorTextLayerPatch = Partial<Omit<TextEntry, "estilo" | "style">> & {
  estilo?: Partial<TextLayerStyle>;
  style?: Partial<TextLayerStyle>;
};

export type NormalizedTextLayer = TextEntry & {
  displayText: string;
  displayOriginal: string;
  effectiveBbox: TextEntry["bbox"];
  hasOriginal: boolean;
  hasTranslation: boolean;
  confidencePercent: number;
  visible: boolean;
  locked: boolean;
  order: number;
};

export type NormalizedImageLayer = ImageLayer & {
  hasContent: boolean;
  opacity: number;
  order: number;
  technical: boolean;
};

export interface EditorScene {
  page: PageData | null;
  textLayers: NormalizedTextLayer[];
  imageLayers: NormalizedImageLayer[];
  selectedLayerId: string | null;
  selectedTextLayer: NormalizedTextLayer | null;
  textCount: number;
  visibleTextCount: number;
}

const IMAGE_LAYER_ROLES: ImageLayerKey[] = ["base", "mask", "inpaint", "brush", "recovery", "rendered"];

function mergePendingTextLayer(layer: TextEntry, patch: EditorTextLayerPatch | undefined): TextEntry {
  if (!patch) return { ...layer };
  const stylePatch = patch.estilo ?? patch.style;
  const estilo = stylePatch ? { ...layer.estilo, ...stylePatch } : layer.estilo;
  return {
    ...layer,
    ...patch,
    traduzido: patch.traduzido ?? patch.translated ?? layer.traduzido,
    translated: patch.translated ?? patch.traduzido ?? layer.translated,
    bbox: patch.bbox ?? layer.bbox,
    layout_bbox: patch.bbox ?? layer.layout_bbox,
    balloon_bbox: patch.bbox ?? layer.balloon_bbox,
    estilo,
    style: estilo,
  } as TextEntry;
}

function normalizeTextLayer(layer: TextEntry): NormalizedTextLayer {
  const order = layer.order ?? 0;
  const displayText = layer.traduzido?.trim() ? layer.traduzido : layer.translated?.trim() ? layer.translated : "";
  const displayOriginal = layer.original ?? "";
  const confidence = (layer.confianca_ocr ?? layer.ocr_confidence ?? 0) || 0;
  const confidencePercent = Math.min(100, Math.max(0, Math.round(Number(confidence) * 100)));
  return {
    ...layer,
    displayText,
    displayOriginal,
    effectiveBbox: layer.layout_bbox ?? layer.bbox,
    hasOriginal: displayOriginal.trim().length > 0,
    hasTranslation: displayText.trim().length > 0,
    confidencePercent: Number.isFinite(confidencePercent) ? confidencePercent : 0,
    visible: layer.visible !== false,
    locked: layer.locked === true,
    order,
  };
}

function compareTextLayers(a: TextEntry, b: TextEntry) {
  const orderDiff = (a.order ?? 0) - (b.order ?? 0);
  if (orderDiff !== 0) return orderDiff;
  return a.id.localeCompare(b.id);
}

function imageLayerDefaultPath(page: PageData, key: ImageLayerKey) {
  if (key === "base") return page.arquivo_original ?? null;
  if (key === "rendered") return page.image_layers?.rendered?.path ?? page.arquivo_traduzido ?? null;
  return null;
}

function normalizeImageLayer(page: PageData, key: ImageLayerKey, index: number): NormalizedImageLayer {
  const existing = page.image_layers?.[key];
  const path = existing?.path ?? imageLayerDefaultPath(page, key);
  const defaultEnabled = key === "base" || key === "rendered";
  return {
    key,
    path,
    visible: existing?.visible ?? defaultEnabled,
    locked: existing?.locked ?? defaultEnabled,
    opacity: existing?.opacity ?? 1,
    order: existing?.order ?? index,
    technical: existing?.technical ?? key === "mask",
    hasContent: Boolean(path),
  };
}

export function buildEditorScene(_input: {
  page?: PageData | null;
  pendingEdits?: Record<string, EditorTextLayerPatch>;
  selectedLayerId?: string | null;
}): EditorScene {
  const page = _input.page ?? null;
  const selectedLayerId = _input.selectedLayerId ?? null;

  if (!page) {
    return {
      page: null,
      textLayers: [],
      imageLayers: [],
      selectedLayerId,
      selectedTextLayer: null,
      textCount: 0,
      visibleTextCount: 0,
    };
  }

  const pendingEdits = _input.pendingEdits ?? {};
  const textLayers = [...(page.text_layers ?? [])]
    .map((layer) => mergePendingTextLayer(layer, pendingEdits[layer.id]))
    .sort(compareTextLayers)
    .map((layer) => normalizeTextLayer(layer));
  const selectedTextLayer = textLayers.find((layer) => layer.id === selectedLayerId) ?? null;
  const imageLayers = IMAGE_LAYER_ROLES.map((key, index) => normalizeImageLayer(page, key, index));

  return {
    page,
    textLayers,
    imageLayers,
    selectedLayerId,
    selectedTextLayer,
    textCount: textLayers.length,
    visibleTextCount: textLayers.filter((layer) => layer.visible).length,
  };
}

export function searchTextLayers(layers: NormalizedTextLayer[], query: string): NormalizedTextLayer[] {
  const normalizedQuery = query.trim().toLowerCase();
  if (!normalizedQuery) return layers;
  return layers.filter((layer) =>
    [layer.displayText, layer.displayOriginal, layer.tipo].some((value) =>
      value.toLowerCase().includes(normalizedQuery),
    ),
  );
}

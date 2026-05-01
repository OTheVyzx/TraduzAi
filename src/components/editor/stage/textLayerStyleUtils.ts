import type { TextEntry, TextLayerStyle } from "../../../lib/stores/appStore";

const DEFAULT_STYLE: TextLayerStyle = {
  fonte: "CCDaveGibbonsLower W00 Regular.ttf",
  tamanho: 28,
  cor: "#000000",
  cor_gradiente: [],
  contorno: "",
  contorno_px: 0,
  glow: false,
  glow_cor: "",
  glow_px: 0,
  sombra: false,
  sombra_cor: "",
  sombra_offset: [0, 0],
  bold: false,
  italico: false,
  rotacao: 0,
  alinhamento: "center",
  force_upper: false,
};

export function mergePendingTextEntry(
  entry: TextEntry,
  pending: Partial<TextEntry> | undefined,
): TextEntry {
  if (!pending) return entry;
  const estilo = pending.estilo ? { ...entry.estilo, ...pending.estilo } : entry.estilo;
  return {
    ...entry,
    ...pending,
    traduzido: pending.traduzido ?? pending.translated ?? entry.traduzido,
    translated: pending.translated ?? pending.traduzido ?? entry.translated,
    bbox: pending.bbox ?? entry.bbox,
    layout_bbox: pending.bbox ?? entry.layout_bbox,
    balloon_bbox: pending.bbox ?? entry.balloon_bbox,
    estilo,
    style: estilo,
  };
}

export function textForLayer(entry: TextEntry) {
  const text = entry.traduzido ?? entry.translated ?? "";
  return entry.estilo?.force_upper ? text.toUpperCase() : text;
}

export function styleForLayer(entry: TextEntry) {
  return { ...DEFAULT_STYLE, ...(entry.style ?? entry.estilo ?? {}) };
}

export function fontFamilyFromStyle(style: TextLayerStyle) {
  return (style.fonte || DEFAULT_STYLE.fonte).replace(/\.(ttf|otf)$/i, "");
}

export function fontStyleFromStyle(style: TextLayerStyle) {
  if (style.bold && style.italico) return "bold italic";
  if (style.bold) return "bold";
  if (style.italico) return "italic";
  return "normal";
}

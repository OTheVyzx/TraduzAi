import type { TextLayerStyle, TextLayerStyleOrigin } from "./stores/appStore";

export const CANONICAL_FONT_FILE = "ComicNeue-Bold.ttf";

export const DEFAULT_TEXT_STYLE: TextLayerStyle = {
  fonte: CANONICAL_FONT_FILE,
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
  bold: true,
  italico: false,
  rotacao: 0,
  curva: false,
  curva_direcao: "",
  curva_intensidade: 0,
  alinhamento: "center",
  force_upper: false,
};

type CanonicalizeMode = "default" | "hydrate" | "preserve-explicit";

interface CanonicalizeOptions {
  mode?: CanonicalizeMode;
  styleOrigin?: TextLayerStyleOrigin | null;
}

function normalizeHex(value: unknown) {
  return typeof value === "string" ? value.trim().toLowerCase() : "";
}

function isLegacyDefaultStyle(style: Partial<TextLayerStyle>) {
  const font = String(style.fonte ?? "").trim().toLowerCase();
  const defaultLikeFont =
    !font ||
    font === "comicneue-bold.ttf" ||
    font === "cccdavegibbonslower w00 regular.ttf" ||
    font === "ccdavegibbonslower w00 regular.ttf";
  return (
    defaultLikeFont &&
    normalizeHex(style.cor) === "#ffffff" &&
    normalizeHex(style.contorno) === "#000000" &&
    Number(style.contorno_px ?? 0) === 2 &&
    (style.glow ?? false) === false &&
    (style.sombra ?? false) === false
  );
}

function styleOriginOf(style: Partial<TextLayerStyle>, options: CanonicalizeOptions) {
  const rawStyle = style as Partial<TextLayerStyle> & { style_origin?: unknown };
  return typeof rawStyle.style_origin === "string" ? rawStyle.style_origin : options.styleOrigin;
}

export function canonicalizeTextStyle<T extends Partial<TextLayerStyle>>(
  style: T,
  options: CanonicalizeOptions = {},
): T & TextLayerStyle {
  const sourceDetected = styleOriginOf(style, options) === "source_detected";
  const shouldNormalizeLegacy = options.mode === "hydrate" && !sourceDetected && isLegacyDefaultStyle(style);
  const contour = shouldNormalizeLegacy ? "" : (style.contorno ?? DEFAULT_TEXT_STYLE.contorno);
  const contourPx = contour ? (style.contorno_px ?? DEFAULT_TEXT_STYLE.contorno_px) : 0;

  return {
    ...style,
    fonte: shouldNormalizeLegacy ? CANONICAL_FONT_FILE : (style.fonte ?? DEFAULT_TEXT_STYLE.fonte),
    tamanho: style.tamanho ?? DEFAULT_TEXT_STYLE.tamanho,
    cor: shouldNormalizeLegacy ? DEFAULT_TEXT_STYLE.cor : (style.cor ?? DEFAULT_TEXT_STYLE.cor),
    cor_gradiente: style.cor_gradiente ?? DEFAULT_TEXT_STYLE.cor_gradiente,
    contorno: contour,
    contorno_px: contourPx,
    glow: shouldNormalizeLegacy ? false : (style.glow ?? DEFAULT_TEXT_STYLE.glow),
    glow_cor: shouldNormalizeLegacy ? "" : (style.glow_cor ?? DEFAULT_TEXT_STYLE.glow_cor),
    glow_px: shouldNormalizeLegacy ? 0 : (style.glow_px ?? DEFAULT_TEXT_STYLE.glow_px),
    sombra: shouldNormalizeLegacy ? false : (style.sombra ?? DEFAULT_TEXT_STYLE.sombra),
    sombra_cor: shouldNormalizeLegacy ? "" : (style.sombra_cor ?? DEFAULT_TEXT_STYLE.sombra_cor),
    sombra_offset: shouldNormalizeLegacy
      ? DEFAULT_TEXT_STYLE.sombra_offset
      : (style.sombra_offset ?? DEFAULT_TEXT_STYLE.sombra_offset),
    bold: shouldNormalizeLegacy ? DEFAULT_TEXT_STYLE.bold : (style.bold ?? DEFAULT_TEXT_STYLE.bold),
    italico: style.italico ?? DEFAULT_TEXT_STYLE.italico,
    rotacao: style.rotacao ?? DEFAULT_TEXT_STYLE.rotacao,
    curva: style.curva ?? DEFAULT_TEXT_STYLE.curva,
    curva_direcao: style.curva_direcao ?? DEFAULT_TEXT_STYLE.curva_direcao,
    curva_intensidade: style.curva_intensidade ?? DEFAULT_TEXT_STYLE.curva_intensidade,
    alinhamento: style.alinhamento ?? DEFAULT_TEXT_STYLE.alinhamento,
    force_upper: style.force_upper ?? DEFAULT_TEXT_STYLE.force_upper,
  };
}

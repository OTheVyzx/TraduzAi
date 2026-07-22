export const STUDIO_TEXT_STYLE_VERSION = "1.0" as const;

export type ResolvedTextAlign = "left" | "center" | "right";
export type ResolvedStrokePosition = "inside" | "center" | "outside";

export interface StudioTextTypography {
  fontFamily: string;
  fontSize: number;
  fontWeight: number;
  fontStyle: "normal" | "italic";
  align: ResolvedTextAlign;
  lineHeight: number;
  tracking: number;
  horizontalScale: number;
  verticalScale: number;
  baselineShift: number;
  vertical: boolean;
}

export interface StudioSolidFill {
  type: "solid";
  color: string;
  opacity?: number;
  enabled?: boolean;
}

export interface StudioGradientStop {
  offset: number;
  color: string;
  opacity?: number;
}

export interface StudioLinearGradientFill {
  type: "linear-gradient";
  stops: StudioGradientStop[];
  angle?: number;
  opacity?: number;
  enabled?: boolean;
}

export type StudioTextFill = StudioSolidFill | StudioLinearGradientFill;

export interface StudioTextStroke {
  color: string;
  width: number;
  opacity?: number;
  position?: ResolvedStrokePosition;
  enabled?: boolean;
}

export interface StudioDropShadow {
  color: string;
  opacity?: number;
  blur?: number;
  offsetX?: number;
  offsetY?: number;
  enabled?: boolean;
}

export interface StudioOuterGlow {
  color: string;
  opacity?: number;
  blur?: number;
  spread?: number;
  enabled?: boolean;
}

export interface StudioTextEffects {
  dropShadows?: StudioDropShadow[];
  outerGlow?: StudioOuterGlow | null;
}

export interface StudioTextStyleDefinition {
  version: typeof STUDIO_TEXT_STYLE_VERSION;
  typography?: Partial<StudioTextTypography>;
  fills?: StudioTextFill[];
  strokes?: StudioTextStroke[];
  effects?: StudioTextEffects;
}

export interface ResolvedSolidFill {
  type: "solid";
  color: string;
  opacity: number;
}

export interface ResolvedLinearGradientFill {
  type: "linear-gradient";
  stops: Array<Required<StudioGradientStop>>;
  angle: number;
  opacity: number;
}

export type ResolvedTextFill = ResolvedSolidFill | ResolvedLinearGradientFill;

export interface ResolvedTextStroke {
  color: string;
  width: number;
  opacity: number;
  position: ResolvedStrokePosition;
}

export interface ResolvedDropShadow {
  color: string;
  opacity: number;
  blur: number;
  offsetX: number;
  offsetY: number;
}

export interface ResolvedOuterGlow {
  color: string;
  opacity: number;
  blur: number;
  spread: number;
}

export interface ResolvedEditorTextStyle {
  version: typeof STUDIO_TEXT_STYLE_VERSION;
  source: "legacy" | "studio";
  typography: StudioTextTypography;
  fills: ResolvedTextFill[];
  strokes: ResolvedTextStroke[];
  effects: {
    dropShadows: ResolvedDropShadow[];
    outerGlow: ResolvedOuterGlow | null;
  };
}

type UnknownRecord = Record<string, unknown>;

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function finiteNumber(value: unknown) {
  const number = typeof value === "number" ? value : typeof value === "string" && value.trim() ? Number(value) : Number.NaN;
  return Number.isFinite(number) ? number : null;
}

function boundedNumber(value: unknown, fallback: number, min: number, max: number) {
  const number = finiteNumber(value);
  return number === null ? fallback : Math.min(max, Math.max(min, number));
}

function stringValue(value: unknown) {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function firstString(...values: unknown[]) {
  for (const value of values) {
    const normalized = stringValue(value);
    if (normalized) return normalized;
  }
  return null;
}

function normalizeHexColor(value: unknown, fallback: string | null = null): string | null {
  const input = stringValue(value)?.toLowerCase();
  if (!input) return fallback;
  const short = input.match(/^#([0-9a-f]{3}|[0-9a-f]{4})$/i)?.[1];
  if (short) return `#${[...short].map((char) => `${char}${char}`).join("")}`;
  if (/^#[0-9a-f]{6}([0-9a-f]{2})?$/i.test(input)) return input;
  return fallback;
}

function normalizeAlign(value: unknown, fallback: ResolvedTextAlign): ResolvedTextAlign {
  return value === "left" || value === "right" || value === "center" ? value : fallback;
}

function normalizeWeight(value: unknown, fallback: number) {
  if (value === "bold") return 700;
  if (value === "normal") return 400;
  return Math.round(boundedNumber(value, fallback, 100, 900) / 100) * 100;
}

function normalizeFontStyle(value: unknown, fallback: "normal" | "italic") {
  return value === "italic" || value === "oblique" ? "italic" : value === "normal" ? "normal" : fallback;
}

function normalizeTypography(source: UnknownRecord, professional: UnknownRecord | null): StudioTextTypography {
  const typography = isRecord(professional?.typography) ? professional.typography : {};
  const legacyBold = source.bold === true ? 700 : 400;
  const legacyItalic = source.italico === true ? "italic" : "normal";
  return {
    fontFamily: firstString(typography.fontFamily, source.fontFamily, source.fonte) ?? "ArialMT",
    fontSize: boundedNumber(typography.fontSize ?? source.fontSize ?? source.tamanho, 24, 1, 2000),
    fontWeight: normalizeWeight(typography.fontWeight, legacyBold),
    fontStyle: normalizeFontStyle(typography.fontStyle, legacyItalic),
    align: normalizeAlign(typography.align ?? source.align ?? source.alinhamento, "center"),
    lineHeight: boundedNumber(typography.lineHeight ?? source.lineHeight, 1.15, 0.25, 10),
    tracking: boundedNumber(typography.tracking ?? source.tracking, 0, -1000, 1000),
    horizontalScale: boundedNumber(typography.horizontalScale ?? source.horizontalScale, 100, 1, 1000),
    verticalScale: boundedNumber(typography.verticalScale ?? source.verticalScale, 100, 1, 1000),
    baselineShift: boundedNumber(typography.baselineShift ?? source.baselineShift, 0, -2000, 2000),
    vertical: typography.vertical === true || source.vertical === true,
  };
}

function normalizeGradientStop(value: unknown): Required<StudioGradientStop> | null {
  if (!isRecord(value)) return null;
  const color = normalizeHexColor(value.color);
  if (!color) return null;
  return {
    offset: boundedNumber(value.offset, 0, 0, 1),
    color,
    opacity: boundedNumber(value.opacity, 1, 0, 1),
  };
}

function normalizeProfessionalFill(value: unknown): ResolvedTextFill | null {
  if (!isRecord(value) || value.enabled === false) return null;
  if (value.type === "solid") {
    const color = normalizeHexColor(value.color);
    if (!color) return null;
    return { type: "solid", color, opacity: boundedNumber(value.opacity, 1, 0, 1) };
  }
  if (value.type === "linear-gradient") {
    const stops = Array.isArray(value.stops)
      ? value.stops.map(normalizeGradientStop).filter((stop): stop is Required<StudioGradientStop> => stop !== null)
      : [];
    if (stops.length < 2) return null;
    return {
      type: "linear-gradient",
      angle: boundedNumber(value.angle, 90, -360, 360),
      opacity: boundedNumber(value.opacity, 1, 0, 1),
      stops: stops.sort((a, b) => a.offset - b.offset),
    };
  }
  return null;
}

function legacyFills(source: UnknownRecord): ResolvedTextFill[] {
  const colors = Array.isArray(source.cor_gradiente)
    ? source.cor_gradiente.map((color) => normalizeHexColor(color)).filter((color): color is string => color !== null)
    : [];
  if (colors.length >= 2) {
    return [{
      type: "linear-gradient",
      angle: 90,
      opacity: 1,
      stops: colors.map((color, index) => ({
        offset: index / (colors.length - 1),
        color,
        opacity: 1,
      })),
    }];
  }
  return [{
    type: "solid",
    color: normalizeHexColor(source.color ?? source.cor, "#000000") ?? "#000000",
    opacity: 1,
  }];
}

function normalizeFills(source: UnknownRecord, professional: UnknownRecord | null) {
  if (!professional || !Array.isArray(professional.fills)) return legacyFills(source);
  const fills = professional.fills
    .map(normalizeProfessionalFill)
    .filter((fill): fill is ResolvedTextFill => fill !== null)
    .slice(0, 16);
  return fills.length > 0 ? fills : legacyFills(source);
}

function normalizeProfessionalStroke(value: unknown): ResolvedTextStroke | null {
  if (!isRecord(value) || value.enabled === false) return null;
  const color = normalizeHexColor(value.color);
  const width = boundedNumber(value.width, 0, 0, 200);
  if (!color || width <= 0) return null;
  const position = value.position === "inside" || value.position === "center" || value.position === "outside"
    ? value.position
    : "outside";
  return {
    color,
    width,
    opacity: boundedNumber(value.opacity, 1, 0, 1),
    position,
  };
}

function normalizeStrokes(source: UnknownRecord, professional: UnknownRecord | null): ResolvedTextStroke[] {
  if (professional && Array.isArray(professional.strokes)) {
    return professional.strokes
      .map(normalizeProfessionalStroke)
      .filter((stroke): stroke is ResolvedTextStroke => stroke !== null)
      .slice(0, 16);
  }
  const width = boundedNumber(source.strokeWidth ?? source.contorno_px, 0, 0, 200);
  const color = normalizeHexColor(source.strokeColor ?? source.contorno);
  return width > 0 && color ? [{ color, width, opacity: 1, position: "center" }] : [];
}

function normalizeDropShadow(value: unknown): ResolvedDropShadow | null {
  if (!isRecord(value) || value.enabled === false) return null;
  const color = normalizeHexColor(value.color);
  if (!color) return null;
  return {
    color,
    opacity: boundedNumber(value.opacity, 0.75, 0, 1),
    blur: boundedNumber(value.blur, 0, 0, 200),
    offsetX: boundedNumber(value.offsetX, 0, -2000, 2000),
    offsetY: boundedNumber(value.offsetY, 0, -2000, 2000),
  };
}

function legacyDropShadows(source: UnknownRecord): ResolvedDropShadow[] {
  if (source.sombra !== true) return [];
  const offset = Array.isArray(source.sombra_offset) ? source.sombra_offset : [];
  return [{
    color: normalizeHexColor(source.sombra_cor, "#000000") ?? "#000000",
    opacity: 0.9,
    blur: boundedNumber(source.sombra_blur, 0, 0, 200),
    offsetX: boundedNumber(offset[0], 0, -2000, 2000),
    offsetY: boundedNumber(offset[1], 0, -2000, 2000),
  }];
}

function normalizeOuterGlow(value: unknown): ResolvedOuterGlow | null {
  if (!isRecord(value) || value.enabled === false) return null;
  const color = normalizeHexColor(value.color);
  if (!color) return null;
  return {
    color,
    opacity: boundedNumber(value.opacity, 0.75, 0, 1),
    blur: boundedNumber(value.blur, 0, 0, 200),
    spread: boundedNumber(value.spread, 0, 0, 200),
  };
}

function legacyOuterGlow(source: UnknownRecord): ResolvedOuterGlow | null {
  const blur = boundedNumber(source.glow_px, 0, 0, 200);
  if (source.glow !== true || blur <= 0) return null;
  return {
    color: normalizeHexColor(source.glow_cor, "#ffffff") ?? "#ffffff",
    opacity: 0.85,
    blur,
    spread: 0,
  };
}

function normalizeEffects(source: UnknownRecord, professional: UnknownRecord | null) {
  const effects = isRecord(professional?.effects) ? professional.effects : null;
  const dropShadows = effects && Array.isArray(effects.dropShadows)
    ? effects.dropShadows.map(normalizeDropShadow).filter((shadow): shadow is ResolvedDropShadow => shadow !== null).slice(0, 8)
    : legacyDropShadows(source);
  const outerGlow = effects && Object.prototype.hasOwnProperty.call(effects, "outerGlow")
    ? normalizeOuterGlow(effects.outerGlow)
    : legacyOuterGlow(source);
  return { dropShadows, outerGlow };
}

export function resolveEditorTextStyle(style: unknown, legacyStyle?: unknown): ResolvedEditorTextStyle {
  const fallback = isRecord(legacyStyle) ? legacyStyle : {};
  const primary = isRecord(style) ? style : {};
  const source: UnknownRecord = { ...fallback, ...primary };
  const professional = isRecord(source.studio_style) ? source.studio_style : null;
  return {
    version: STUDIO_TEXT_STYLE_VERSION,
    source: professional ? "studio" : "legacy",
    typography: normalizeTypography(source, professional),
    fills: normalizeFills(source, professional),
    strokes: normalizeStrokes(source, professional),
    effects: normalizeEffects(source, professional),
  };
}

export function fontStyleForResolvedTextStyle(style: ResolvedEditorTextStyle) {
  const values: string[] = [];
  if (style.typography.fontWeight >= 600) values.push("bold");
  if (style.typography.fontStyle === "italic") values.push("italic");
  return values.join(" ") || "normal";
}

export function colorToRgba(value: string, opacity = 1): [number, number, number, number] {
  const color = normalizeHexColor(value, "#000000") ?? "#000000";
  const hex = color.slice(1);
  const alpha = hex.length === 8 ? Number.parseInt(hex.slice(6, 8), 16) / 255 : 1;
  return [
    Number.parseInt(hex.slice(0, 2), 16),
    Number.parseInt(hex.slice(2, 4), 16),
    Number.parseInt(hex.slice(4, 6), 16),
    Math.round(255 * alpha * Math.min(1, Math.max(0, opacity))),
  ];
}

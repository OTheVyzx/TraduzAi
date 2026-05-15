import type { TextLayerStyle } from "./stores/appStore";

export type EditorTextStylePresetKind = "builtin" | "custom";
export type EditorTextStylePresetPatch = Partial<TextLayerStyle>;

export interface EditorTextStylePreset {
  id: string;
  name: string;
  description?: string;
  kind: EditorTextStylePresetKind;
  stylePatch: EditorTextStylePresetPatch;
  createdAt?: string;
  updatedAt?: string;
}

const CUSTOM_ID_PREFIX = "custom_";

const STYLE_KEYS = [
  "fonte",
  "tamanho",
  "cor",
  "cor_gradiente",
  "contorno",
  "contorno_px",
  "glow",
  "glow_cor",
  "glow_px",
  "sombra",
  "sombra_cor",
  "sombra_offset",
  "bold",
  "italico",
  "rotacao",
  "alinhamento",
  "force_upper",
] as const satisfies readonly (keyof TextLayerStyle)[];

export const BUILTIN_TEXT_STYLE_PRESETS: EditorTextStylePreset[] = [
  {
    id: "town_gradient",
    name: "Town",
    description: "Titulo grande com gradiente frio e contorno escuro.",
    kind: "builtin",
    stylePatch: {
      fonte: "Newrotic.ttf",
      tamanho: 52,
      cor: "#f6ff8f",
      cor_gradiente: ["#f6ff8f", "#03d8ff"],
      contorno: "#102a54",
      contorno_px: 3,
      glow: false,
      glow_cor: "",
      glow_px: 0,
      sombra: true,
      sombra_cor: "#111234",
      sombra_offset: [4, 5],
      bold: true,
      italico: false,
      rotacao: 0,
      alinhamento: "center",
      force_upper: false,
    },
  },
  {
    id: "bang_comic",
    name: "Bang Comic",
    description: "SFX amarelo/laranja com contorno preto e sombra pesada.",
    kind: "builtin",
    stylePatch: {
      fonte: "KOMIKAX_.ttf",
      tamanho: 44,
      cor: "#ffe900",
      cor_gradiente: ["#fff247", "#ff7a00"],
      contorno: "#000000",
      contorno_px: 4,
      glow: true,
      glow_cor: "#ffffff",
      glow_px: 2,
      sombra: true,
      sombra_cor: "#050505",
      sombra_offset: [5, 6],
      bold: true,
      italico: false,
      rotacao: 0,
      alinhamento: "center",
      force_upper: true,
    },
  },
  {
    id: "whoosh_sfx",
    name: "Whoosh",
    description: "SFX inclinado com amarelo forte, contorno preto e impacto.",
    kind: "builtin",
    stylePatch: {
      fonte: "KOMIKAX_.ttf",
      tamanho: 46,
      cor: "#ffe600",
      cor_gradiente: ["#fff56a", "#ffcf00"],
      contorno: "#000000",
      contorno_px: 4,
      glow: false,
      glow_cor: "",
      glow_px: 0,
      sombra: true,
      sombra_cor: "#111111",
      sombra_offset: [4, 5],
      bold: true,
      italico: true,
      rotacao: -6,
      alinhamento: "center",
      force_upper: true,
    },
  },
  {
    id: "clean_dialogue",
    name: "Fala limpa",
    description: "Texto limpo para fala comum.",
    kind: "builtin",
    stylePatch: {
      fonte: "ComicNeue-Bold.ttf",
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
      alinhamento: "center",
      force_upper: false,
    },
  },
];

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : null;
}

function boundedNumber(value: unknown, fallback: number, min: number, max: number) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return fallback;
  return Math.max(min, Math.min(max, Math.round(numeric)));
}

function optionalBoolean(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined;
}

function optionalString(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : "";
}

function normalizeGradient(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) return undefined;
  return value
    .filter((item): item is string => typeof item === "string")
    .map((item) => item.trim())
    .filter(Boolean)
    .slice(0, 4);
}

function normalizeOffset(value: unknown): [number, number] | undefined {
  if (!Array.isArray(value) || value.length < 2) return undefined;
  return [
    boundedNumber(value[0], 0, -100, 100),
    boundedNumber(value[1], 0, -100, 100),
  ];
}

function normalizeAlignment(value: unknown): TextLayerStyle["alinhamento"] | undefined {
  return value === "left" || value === "center" || value === "right" ? value : undefined;
}

export function sanitizeTextStylePresetPatch(value: unknown): EditorTextStylePresetPatch {
  const raw = asRecord(value);
  if (!raw) return {};

  const patch: EditorTextStylePresetPatch = {};
  for (const key of STYLE_KEYS) {
    if (!(key in raw)) continue;
    const item = raw[key];
    switch (key) {
      case "fonte":
      case "cor":
      case "contorno":
      case "glow_cor":
      case "sombra_cor": {
        const stringValue = optionalString(item);
        if (stringValue !== undefined) patch[key] = stringValue;
        break;
      }
      case "tamanho":
        patch.tamanho = boundedNumber(item, 28, 6, 240);
        break;
      case "contorno_px":
        patch.contorno_px = boundedNumber(item, 0, 0, 40);
        break;
      case "glow_px":
        patch.glow_px = boundedNumber(item, 0, 0, 80);
        break;
      case "rotacao":
        patch.rotacao = boundedNumber(item, 0, -180, 180);
        break;
      case "cor_gradiente": {
        const gradient = normalizeGradient(item);
        if (gradient) patch.cor_gradiente = gradient;
        break;
      }
      case "sombra_offset": {
        const offset = normalizeOffset(item);
        if (offset) patch.sombra_offset = offset;
        break;
      }
      case "alinhamento": {
        const alignment = normalizeAlignment(item);
        if (alignment) patch.alinhamento = alignment;
        break;
      }
      case "glow":
      case "sombra":
      case "bold":
      case "italico":
      case "force_upper": {
        const boolValue = optionalBoolean(item);
        if (boolValue !== undefined) patch[key] = boolValue;
        break;
      }
    }
  }
  return patch;
}

export function cloneTextStylePresetPatch(patch: EditorTextStylePresetPatch): EditorTextStylePresetPatch {
  return sanitizeTextStylePresetPatch(patch);
}

export function sanitizeTextStylePreset(value: unknown, fallbackKind: EditorTextStylePresetKind = "custom") {
  const raw = asRecord(value);
  if (!raw) return null;
  const id = typeof raw.id === "string" ? raw.id.trim() : "";
  const name = typeof raw.name === "string" ? raw.name.trim() : "";
  const stylePatch = sanitizeTextStylePresetPatch(raw.stylePatch);
  if (!id || !name || Object.keys(stylePatch).length === 0) return null;
  const kind = raw.kind === "builtin" || raw.kind === "custom" ? raw.kind : fallbackKind;
  return {
    id,
    name,
    description: typeof raw.description === "string" ? raw.description.trim() : undefined,
    kind,
    stylePatch,
    createdAt: typeof raw.createdAt === "string" ? raw.createdAt : undefined,
    updatedAt: typeof raw.updatedAt === "string" ? raw.updatedAt : undefined,
  } satisfies EditorTextStylePreset;
}

export function sanitizeTextStylePresets(value: unknown, fallbackKind: EditorTextStylePresetKind = "custom") {
  if (!Array.isArray(value)) return [];
  const presets: EditorTextStylePreset[] = [];
  const seen = new Set<string>();
  for (const item of value) {
    const preset = sanitizeTextStylePreset(item, fallbackKind);
    if (!preset || seen.has(preset.id)) continue;
    presets.push(preset);
    seen.add(preset.id);
  }
  return presets;
}

export function mergeTextStylePresetLists(customPresets: EditorTextStylePreset[]) {
  const builtinIds = new Set(BUILTIN_TEXT_STYLE_PRESETS.map((preset) => preset.id));
  const custom = sanitizeTextStylePresets(customPresets, "custom").filter(
    (preset) => preset.kind === "custom" && !builtinIds.has(preset.id),
  );
  return [...BUILTIN_TEXT_STYLE_PRESETS, ...custom];
}

export function createCustomTextStylePreset(
  style: Partial<TextLayerStyle>,
  name: string,
  now = new Date(),
): EditorTextStylePreset {
  const label = name.trim() || "Preset customizado";
  const iso = now.toISOString();
  return {
    id: `${CUSTOM_ID_PREFIX}${now.getTime().toString(36)}`,
    name: label,
    description: "Criado a partir do texto selecionado.",
    kind: "custom",
    stylePatch: sanitizeTextStylePresetPatch(style),
    createdAt: iso,
    updatedAt: iso,
  };
}

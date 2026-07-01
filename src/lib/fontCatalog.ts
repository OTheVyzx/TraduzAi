import { BUNDLE_FONTS, ensureEditorFontLoaded, registerImportedFont, registerRemoteFont, type FontEntry } from "./fonts";
import { GOOGLE_FONTS_CATALOG, type GoogleFontCatalogEntry } from "./googleFontsCatalog";
import type { CacheGoogleFontInput, GoogleFontSearchResult, SystemFontInfo } from "./tauri";

export type EditorFontCatalogSource = "bundle" | "google" | "system";

export interface EditorFontOption {
  label: string;
  value: string;
  cssFamily: string;
  source: EditorFontCatalogSource;
  groupLabel: string;
  variants: string[];
  variant?: string;
  downloadUrl?: string;
  localPath?: string;
  style?: string;
}

export interface EditorFontGroup {
  label: string;
  source: EditorFontCatalogSource;
  options: EditorFontOption[];
}

const GROUP_LABELS: Record<EditorFontCatalogSource, string> = {
  bundle: "Embutidas",
  google: "Google Fonts",
  system: "Sistema",
};

function filenameFromPath(path: string): string {
  return path.split("/").pop() ?? path;
}

function bundleFontFilename(entry: FontEntry): string {
  const path =
    entry.files.bold ??
    entry.files.regular ??
    entry.files.italic ??
    entry.files.boldItalic ??
    "";
  return filenameFromPath(path);
}

function bundleFontVariants(entry: FontEntry): string[] {
  const variants: string[] = [];
  if (entry.files.regular) variants.push("regular");
  if (entry.files.bold) variants.push("700");
  if (entry.files.italic) variants.push("italic");
  if (entry.files.boldItalic) variants.push("700italic");
  return variants;
}

function bundleFontOption(entry: FontEntry): EditorFontOption | null {
  const value = bundleFontFilename(entry);
  if (!value) return null;
  return {
    label: entry.cssFamily,
    value,
    cssFamily: entry.cssFamily,
    source: "bundle",
    groupLabel: GROUP_LABELS.bundle,
    variants: bundleFontVariants(entry),
  };
}

function googleFontOption(entry: GoogleFontCatalogEntry): EditorFontOption {
  const variant = entry.variants[0] ?? "regular";
  return {
    label: entry.label,
    value: entry.files[variant] ?? entry.filename,
    cssFamily: entry.cssFamily,
    source: "google",
    groupLabel: GROUP_LABELS.google,
    variants: [...entry.variants],
    variant,
    downloadUrl: entry.downloadUrls[variant],
  };
}

export function googleFontSearchResultToOption(result: GoogleFontSearchResult): EditorFontOption {
  return {
    label: result.family,
    value: result.filename,
    cssFamily: result.css_family || result.family,
    source: "google",
    groupLabel: GROUP_LABELS.google,
    variants: [result.variant || "regular"],
    variant: result.variant || "regular",
    downloadUrl: result.download_url,
  };
}

export function isSystemFontValue(value: string): boolean {
  return /^SystemFont__.+__.+\.(?:ttf|otf)$/i.test(value.trim());
}

export function systemFontInfoToOption(font: SystemFontInfo): EditorFontOption {
  return {
    label: font.family,
    value: font.filename,
    cssFamily: font.family,
    source: "system",
    groupLabel: GROUP_LABELS.system,
    variants: [font.weight || "400"],
    variant: font.weight || "400",
    localPath: font.path,
    style: font.style || "normal",
  };
}

export function buildEditorFontCatalog({
  bundleFonts = BUNDLE_FONTS,
  googleFonts = GOOGLE_FONTS_CATALOG,
}: {
  bundleFonts?: Record<string, FontEntry>;
  googleFonts?: readonly GoogleFontCatalogEntry[];
} = {}): EditorFontOption[] {
  const options: EditorFontOption[] = [];
  const seenFamilies = new Set<string>();
  const seenValues = new Set<string>();

  function addOption(option: EditorFontOption | null): void {
    if (!option) return;
    const familyKey = option.cssFamily.trim().toLowerCase();
    const valueKey = option.value.trim().toLowerCase();
    if (!familyKey || !valueKey || seenFamilies.has(familyKey) || seenValues.has(valueKey)) return;
    seenFamilies.add(familyKey);
    seenValues.add(valueKey);
    options.push(option);
  }

  for (const entry of Object.values(bundleFonts)) {
    addOption(bundleFontOption(entry));
  }
  for (const entry of googleFonts) {
    addOption(googleFontOption(entry));
  }

  return options;
}

export function listEditorFontGroups(): EditorFontGroup[] {
  const catalog = buildEditorFontCatalog();
  return (["bundle", "google"] as const)
    .map((source) => ({
      label: GROUP_LABELS[source],
      source,
      options: catalog.filter((option) => option.source === source),
    }))
    .filter((group) => group.options.length > 0);
}

function normalizeFontSearchQuery(query: string): string {
  return query
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase();
}

function optionMatchesSearch(option: EditorFontOption, normalizedQuery: string): boolean {
  const haystack = normalizeFontSearchQuery(`${option.label} ${option.cssFamily} ${option.value}`);
  const compactHaystack = haystack.replace(/\s+/g, "");
  const compactQuery = normalizedQuery.replace(/\s+/g, "");
  return haystack.includes(normalizedQuery) || (!!compactQuery && compactHaystack.includes(compactQuery));
}

export function searchEditorFontGroups(query: string): EditorFontGroup[] {
  const normalizedQuery = normalizeFontSearchQuery(query);
  if (!normalizedQuery) return listEditorFontGroups();

  const options = buildEditorFontCatalog().filter((option) => optionMatchesSearch(option, normalizedQuery));
  return (["bundle", "google"] as const)
    .map((source) => ({
      label: GROUP_LABELS[source],
      source,
      options: options.filter((option) => option.source === source),
    }))
    .filter((group) => group.options.length > 0);
}

export function findEditorFontOption(value: string): EditorFontOption | null {
  const catalogOption = buildEditorFontCatalog().find((option) => option.value === value);
  if (catalogOption) return catalogOption;
  if (isSystemFontValue(value)) {
    const label = value
      .replace(/^SystemFont__/i, "")
      .replace(/__[^.]+\.(?:ttf|otf)$/i, "")
      .replace(/_/g, " ");
    return {
      label,
      value,
      cssFamily: label,
      source: "system",
      groupLabel: GROUP_LABELS.system,
      variants: ["400"],
      variant: "400",
      style: "normal",
    };
  }
  return null;
}

function bytesToArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
}

export async function ensureEditorFontOptionReady(value: string | EditorFontOption): Promise<EditorFontOption | null> {
  const option = typeof value === "string" ? findEditorFontOption(value) : value;
  if (!option) return option;
  if (option.source === "system") {
    const { resolveSystemFont } = await import("./tauri");
    const font = await resolveSystemFont(option.value);
    if (!font) throw new Error(`Fonte do sistema nao encontrada: ${option.label}`);
    const weight = font.weight === "700" ? "700" : "400";
    const style = font.style === "italic" ? "italic" : "normal";
    try {
      const { readFile } = await import("@tauri-apps/plugin-fs");
      const bytes = await readFile(font.path);
      await registerImportedFont(font.family, bytesToArrayBuffer(bytes), weight, style);
    } catch (error) {
      console.warn("[fonts] falha ao registrar fonte do sistema por arquivo; usando fallback CSS:", error);
      await ensureEditorFontLoaded(font.family, 32, style);
    }
    return { ...option, localPath: font.path, cssFamily: font.family, style, variant: weight };
  }
  if (option.source !== "google") return option;
  if (!option.downloadUrl) {
    throw new Error(`Fonte Google sem URL de download: ${option.label}`);
  }

  const variant = option.variant ?? "regular";
  const input: CacheGoogleFontInput = {
    family: option.label,
    css_family: option.cssFamily,
    variant,
    url: option.downloadUrl,
    filename: option.value,
  };
  const { cacheGoogleFont } = await import("./tauri");
  const cached = await cacheGoogleFont(input);
  const weight = variant === "700" ? "700" : "400";

  try {
    const { readFile } = await import("@tauri-apps/plugin-fs");
    const bytes = await readFile(cached.path);
    await registerImportedFont(option.cssFamily, bytesToArrayBuffer(bytes), weight);
  } catch (error) {
    console.warn("[fonts] falha ao registrar fonte Google do cache; usando URL remota no preview:", error);
    await registerRemoteFont(option.cssFamily, option.downloadUrl, weight);
  }

  return option;
}

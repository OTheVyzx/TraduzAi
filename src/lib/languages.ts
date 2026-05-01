import type { SupportedLanguage } from "./tauri";

const LANGUAGE_ALIASES: Record<string, string> = {
  "pt-br": "pt",
  "pt_pt": "pt",
  "pt-pt": "pt",
  "en-gb": "en",
  "en-us": "en",
  zh: "zh-CN",
  "zh-cn": "zh-CN",
  "zh-hans": "zh-CN",
  "zh-tw": "zh-TW",
  "zh-hant": "zh-TW",
};

export const FALLBACK_SUPPORTED_LANGUAGES: SupportedLanguage[] = [
  { code: "en", label: "English", ocr_strategy: "dedicated" },
  { code: "pt", label: "Portuguese", ocr_strategy: "dedicated" },
  { code: "es", label: "Spanish", ocr_strategy: "dedicated" },
  { code: "de", label: "German", ocr_strategy: "dedicated" },
  { code: "ru", label: "Russian", ocr_strategy: "dedicated" },
  { code: "ja", label: "Japanese", ocr_strategy: "dedicated" },
  { code: "ko", label: "Korean", ocr_strategy: "dedicated" },
  { code: "zh-CN", label: "Chinese (simplified)", ocr_strategy: "dedicated" },
  { code: "zh-TW", label: "Chinese (traditional)", ocr_strategy: "dedicated" },
];

export function normalizeLanguageCodeForSelection(
  value: string | null | undefined,
  languages: SupportedLanguage[],
  fallbackCode: string
): string {
  const supported = new Set(languages.map((language) => language.code));
  const raw = (value || "").trim();
  if (!raw) return fallbackCode;
  if (supported.has(raw)) return raw;

  const normalized = raw.replaceAll("_", "-");
  const alias = LANGUAGE_ALIASES[normalized.toLowerCase()];
  if (alias && supported.has(alias)) return alias;

  const base = normalized.toLowerCase().split("-", 1)[0];
  const baseAlias = LANGUAGE_ALIASES[base];
  if (baseAlias && supported.has(baseAlias)) return baseAlias;
  if (supported.has(base)) return base;

  return fallbackCode;
}

export function getLanguageOptions(
  languages: SupportedLanguage[] | null | undefined
): SupportedLanguage[] {
  return languages && languages.length > 0 ? languages : FALLBACK_SUPPORTED_LANGUAGES;
}

export function formatSourceLanguageLabel(language: SupportedLanguage): string {
  return language.ocr_strategy === "best_effort"
    ? `${language.label} - OCR experimental`
    : language.label;
}

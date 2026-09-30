export const READER_LANGUAGES_STORAGE_KEY = "traduzai.studio.reader.languages.v1";

export type AllowedLanguages = string[] | null;

export function parseAllowedLanguages(raw: string | null): AllowedLanguages {
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return null;
    return [...new Set(parsed.filter((value): value is string => typeof value === "string" && Boolean(value.trim())).map((value) => value.trim()))];
  } catch {
    return null;
  }
}

export function isLanguageAllowed(allowed: AllowedLanguages, language: string): boolean {
  return allowed === null || allowed.includes(language);
}

export function toggleAllowedLanguage(
  current: AllowedLanguages,
  language: string,
  enabled: boolean,
  availableLanguages: string[],
): AllowedLanguages {
  if (language === "*") return enabled ? null : [];
  const selected = new Set(current === null ? availableLanguages : current);
  if (enabled) selected.add(language);
  else selected.delete(language);
  return availableLanguages.filter((code) => selected.has(code));
}

export function saveAllowedLanguages(storage: Pick<Storage, "setItem">, allowed: AllowedLanguages): void {
  storage.setItem(READER_LANGUAGES_STORAGE_KEY, JSON.stringify(allowed));
}

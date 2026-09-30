import type { RepositoryExtension } from "./sourceRuntimeClient";
import { isLanguageAllowed, type AllowedLanguages } from "./readerPreferences";

export interface InstalledExtensionRef {
  packageName: string;
  enabled: boolean;
}

export interface InstalledSource {
  extensionPackage: string;
  extensionName: string;
  id: string;
  name: string;
  lang: string;
  baseUrl?: string;
  iconUrl?: string;
  nsfw?: number;
}

const LANGUAGE_NAMES = new Intl.DisplayNames(["pt-BR"], { type: "language" });

export function languageLabel(code: string): string {
  if (code === "*") return "Todos os idiomas";
  if (code === "all") return "Multilíngue";
  if (code === "pt-BR") return "Português (Brasil)";
  if (code === "en") return "Inglês";
  if (code === "other") return "Outros";
  try {
    const localized = LANGUAGE_NAMES.of(code);
    return localized ? localized[0].toLocaleUpperCase("pt-BR") + localized.slice(1) : code;
  } catch {
    return code;
  }
}

export function listLanguages(catalog: RepositoryExtension[]) {
  const counts = new Map<string, number>();
  for (const extension of catalog) {
    for (const source of extension.sources) counts.set(source.lang, (counts.get(source.lang) ?? 0) + 1);
  }
  const rows = [...counts].sort(([left], [right]) => languageLabel(left).localeCompare(languageLabel(right), "pt-BR"));
  return [
    { code: "*", label: "Todos os idiomas", count: [...counts.values()].reduce((sum, count) => sum + count, 0) },
    ...rows.map(([code, count]) => ({ code, label: languageLabel(code), count })),
  ];
}

export function filterCatalogByAllowedLanguages(catalog: RepositoryExtension[], allowed: AllowedLanguages): RepositoryExtension[] {
  if (allowed === null) return catalog;
  return catalog
    .map((extension) => ({ ...extension, sources: extension.sources.filter((source) => isLanguageAllowed(allowed, source.lang)) }))
    .filter((extension) => extension.sources.length > 0);
}

export function filterExtensions(catalog: RepositoryExtension[], language: string, query: string) {
  const needle = query.trim().toLocaleLowerCase("pt-BR");
  return catalog.filter((extension) => {
    const matchesLanguage = language === "*" || extension.sources.some((source) => source.lang === language);
    const searchable = `${extension.name} ${extension.packageName} ${extension.sources.map((source) => source.name).join(" ")}`.toLocaleLowerCase("pt-BR");
    return matchesLanguage && (!needle || searchable.includes(needle));
  });
}

export function buildInstalledSources(
  catalog: RepositoryExtension[],
  installed: InstalledExtensionRef[],
  language: string,
  query: string,
): InstalledSource[] {
  const enabled = new Set(installed.filter((item) => item.enabled).map((item) => item.packageName));
  const needle = query.trim().toLocaleLowerCase("pt-BR");
  return catalog.flatMap((extension) => enabled.has(extension.packageName) ? extension.sources
    .filter((source) => language === "*" || source.lang === language)
    .filter((source) => !needle || `${source.name} ${extension.name}`.toLocaleLowerCase("pt-BR").includes(needle))
    .map((source) => ({
      extensionPackage: extension.packageName,
      extensionName: extension.name,
      ...(extension.iconUrl ? { iconUrl: extension.iconUrl } : {}),
      ...((extension.nsfw ?? 0) > 0 ? { nsfw: extension.nsfw } : {}),
      ...source,
    })) : []);
}

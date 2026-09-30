import { languageLabel, type InstalledSource } from "./sourceCatalogModel";
import type { InstalledExtension, RepositoryExtension } from "./sourceRuntimeClient";

export interface SourceGroup {
  id: string;
  label: string;
  sources: InstalledSource[];
}

export type ExtensionSectionId = "updates" | "installed" | "available" | "incompatible";

export interface ExtensionSectionItem {
  extension: RepositoryExtension;
  installed?: InstalledExtension;
}

export interface ExtensionSection {
  id: ExtensionSectionId;
  label: string;
  items: ExtensionSectionItem[];
}

export function sourceIdentity(source: Pick<InstalledSource, "extensionPackage" | "id">): string {
  return `${source.extensionPackage}::${source.id}`;
}

export function groupSources(sources: InstalledSource[], recentIdentity?: string | null): SourceGroup[] {
  const sorted = [...sources].sort((left, right) => left.name.localeCompare(right.name, "pt-BR"));
  const recent = recentIdentity ? sorted.find((source) => sourceIdentity(source) === recentIdentity) : undefined;
  const remaining = recent ? sorted.filter((source) => sourceIdentity(source) !== recentIdentity) : sorted;
  const byLanguage = new Map<string, InstalledSource[]>();
  for (const source of remaining) byLanguage.set(source.lang, [...(byLanguage.get(source.lang) ?? []), source]);
  const groups = [...byLanguage.entries()]
    .sort(([left], [right]) => languageLabel(left).localeCompare(languageLabel(right), "pt-BR"))
    .map(([id, groupedSources]) => ({ id, label: languageLabel(id), sources: groupedSources }));
  return recent ? [{ id: "recent", label: "Usado por último", sources: [recent] }, ...groups] : groups;
}

export function extensionSections(catalog: RepositoryExtension[], installed: InstalledExtension[]): ExtensionSection[] {
  const installedByPackage = new Map(installed.map((extension) => [extension.packageName, extension]));
  const sections: ExtensionSection[] = [
    { id: "updates", label: "Atualização pendente", items: [] },
    { id: "installed", label: "Instaladas", items: [] },
    { id: "available", label: "Disponíveis", items: [] },
    { id: "incompatible", label: "Incompatíveis", items: [] },
  ];
  const byId = new Map(sections.map((section) => [section.id, section]));
  for (const extension of [...catalog].sort((left, right) => left.name.localeCompare(right.name, "pt-BR"))) {
    const installedExtension = installedByPackage.get(extension.packageName);
    const sectionId: ExtensionSectionId = !extension.jarUrl
      ? "incompatible"
      : installedExtension && installedExtension.versionCode < extension.versionCode
        ? "updates"
        : installedExtension
          ? "installed"
          : "available";
    byId.get(sectionId)?.items.push({ extension, installed: installedExtension });
  }
  return sections.filter((section) => section.items.length > 0);
}

export function safeRemoteImageUrl(value?: string): string | undefined {
  if (!value) return undefined;
  try {
    const url = new URL(value);
    return url.protocol === "https:" || url.protocol === "http:" ? url.toString() : undefined;
  } catch {
    return undefined;
  }
}

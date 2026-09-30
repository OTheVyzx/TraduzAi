import { describe, expect, it } from "vitest";
import { extensionSections, groupSources, safeRemoteImageUrl, sourceIdentity } from "../navigateModel";
import type { InstalledExtension, RepositoryExtension } from "../sourceRuntimeClient";
import type { InstalledSource } from "../sourceCatalogModel";

const sources: InstalledSource[] = [
  { extensionPackage: "ext.flix", extensionName: "MangaFlix", id: "20", name: "MangaFlix (PT-BR)", lang: "pt-BR", iconUrl: "https://repo.invalid/flix.png", nsfw: 2 },
  { extensionPackage: "ext.reader", extensionName: "Reader", id: "10", name: "Reader EN", lang: "en" },
  { extensionPackage: "ext.reader", extensionName: "Reader", id: "11", name: "Reader PT", lang: "pt-BR" },
];

const catalog: RepositoryExtension[] = [
  { repositoryId: "r", name: "Atualizável", packageName: "ext.update", versionCode: 2, versionName: "2.0", lang: "pt-BR", jarUrl: "update.jar", sources: [] },
  { repositoryId: "r", name: "Instalada", packageName: "ext.installed", versionCode: 1, versionName: "1.0", lang: "en", jarUrl: "installed.jar", sources: [] },
  { repositoryId: "r", name: "Disponível", packageName: "ext.available", versionCode: 1, versionName: "1.0", lang: "pt-BR", jarUrl: "available.jar", sources: [] },
  { repositoryId: "r", name: "Somente APK", packageName: "ext.apk", versionCode: 1, versionName: "1.0", lang: "ja", apkUrl: "only.apk", sources: [] },
];

const installed: InstalledExtension[] = [
  { packageName: "ext.update", name: "Atualizável", versionCode: 1, versionName: "1.0", repositoryId: "r", sha256: "a", enabled: true, rollbackAvailable: false },
  { packageName: "ext.installed", name: "Instalada", versionCode: 1, versionName: "1.0", repositoryId: "r", sha256: "b", enabled: true, rollbackAvailable: false },
];

describe("navigateModel", () => {
  it("keeps the recent source once and groups the remainder by localized language", () => {
    const grouped = groupSources(sources, sourceIdentity(sources[0]));
    expect(grouped.map((group) => [group.id, group.label, group.sources.map((source) => source.name)])).toEqual([
      ["recent", "Usado por último", ["MangaFlix (PT-BR)"]],
      ["en", "Inglês", ["Reader EN"]],
      ["pt-BR", "Português (Brasil)", ["Reader PT"]],
    ]);
  });

  it("groups extension lifecycle states and marks APK-only packages incompatible", () => {
    const sections = extensionSections(catalog, installed);
    expect(sections.map((section) => [section.id, section.items.map((item) => item.extension.name)])).toEqual([
      ["updates", ["Atualizável"]],
      ["installed", ["Instalada"]],
      ["available", ["Disponível"]],
      ["incompatible", ["Somente APK"]],
    ]);
  });

  it("accepts only remote HTTP image URLs", () => {
    expect(safeRemoteImageUrl("https://repo.invalid/capa.webp")).toBe("https://repo.invalid/capa.webp");
    expect(safeRemoteImageUrl("javascript:alert(1)")).toBeUndefined();
    expect(safeRemoteImageUrl("file:///C:/segredo.png")).toBeUndefined();
  });
});

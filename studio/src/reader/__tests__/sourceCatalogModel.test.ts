import { describe, expect, it } from "vitest";
import { buildInstalledSources, filterCatalogByAllowedLanguages, filterExtensions, listLanguages } from "../sourceCatalogModel";
import type { RepositoryExtension } from "../sourceRuntimeClient";

const catalog: RepositoryExtension[] = [
  { repositoryId: "r", name: "Comikey", packageName: "ext.comikey", versionCode: 1, versionName: "1.0", lang: "all", jarUrl: "c.jar", sources: [
    { id: "1", name: "Comikey", lang: "en" },
    { id: "2", name: "Comikey Brasil", lang: "pt-BR" },
  ] },
  { repositoryId: "r", name: "Manga Livre", packageName: "ext.livre", versionCode: 2, versionName: "1.1", lang: "pt-BR", jarUrl: "m.jar", sources: [
    { id: "3", name: "Manga Livre", lang: "pt-BR" },
  ] },
];

describe("sourceCatalogModel", () => {
  it("lists languages from actual sources with localized labels", () => {
    expect(listLanguages(catalog)).toEqual([
      { code: "*", label: "Todos os idiomas", count: 3 },
      { code: "en", label: "Inglês", count: 1 },
      { code: "pt-BR", label: "Português (Brasil)", count: 2 },
    ]);
  });

  it("filters extensions by language and text", () => {
    expect(filterExtensions(catalog, "pt-BR", "livre").map((item) => item.name)).toEqual(["Manga Livre"]);
    expect(filterExtensions(catalog, "en", "comikey").map((item) => item.name)).toEqual(["Comikey"]);
  });

  it("exposes only sources from installed enabled extensions", () => {
    const sources = buildInstalledSources(catalog, [{ packageName: "ext.comikey", enabled: true }], "pt-BR", "");
    expect(sources).toEqual([{ extensionPackage: "ext.comikey", extensionName: "Comikey", id: "2", name: "Comikey Brasil", lang: "pt-BR" }]);
  });

  it("keeps multilingual extensions when any source uses an allowed language", () => {
    expect(filterCatalogByAllowedLanguages(catalog, ["pt-BR"]).map((item) => item.name)).toEqual(["Comikey", "Manga Livre"]);
    expect(filterCatalogByAllowedLanguages(catalog, ["en"]).map((item) => item.name)).toEqual(["Comikey"]);
    expect(filterCatalogByAllowedLanguages(catalog, null)).toEqual(catalog);
    expect(filterCatalogByAllowedLanguages(catalog, [])).toEqual([]);
  });
});

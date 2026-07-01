import { describe, expect, it } from "vitest";
import { BUNDLE_FONTS, resolveLegacyFontFamily } from "../fonts";
import {
  buildEditorFontCatalog,
  findEditorFontOption,
  googleFontSearchResultToOption,
  isSystemFontValue,
  listEditorFontGroups,
  searchEditorFontGroups,
  systemFontInfoToOption,
} from "../fontCatalog";
import { GOOGLE_FONTS_CATALOG } from "../googleFontsCatalog";

describe("editor font catalog", () => {
  it("includes bundled fonts using legacy filenames", () => {
    const catalog = buildEditorFontCatalog();

    expect(catalog).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          label: "Comic Neue",
          cssFamily: BUNDLE_FONTS.comicNeue.cssFamily,
          source: "bundle",
          value: "ComicNeue-Bold.ttf",
          groupLabel: "Embutidas",
        }),
      ]),
    );

    for (const bundledFont of Object.values(BUNDLE_FONTS)) {
      expect(catalog.some((option) => option.source === "bundle" && option.cssFamily === bundledFont.cssFamily)).toBe(
        true,
      );
    }
  });

  it("includes the local Google Fonts catalog with stable cache filenames", () => {
    expect(GOOGLE_FONTS_CATALOG.map((font) => font.label)).toEqual([
      "Bangers",
      "Comic Neue",
      "Patrick Hand",
      "M PLUS Rounded 1c",
      "Noto Sans",
      "Noto Serif",
      "Roboto Condensed",
    ]);

    const catalog = buildEditorFontCatalog();

    expect(catalog).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          label: "Bangers",
          cssFamily: "Bangers",
          source: "google",
          value: "GoogleFont__Bangers__regular.ttf",
          groupLabel: "Google Fonts",
          variants: ["regular"],
          variant: "regular",
          downloadUrl: "https://raw.githubusercontent.com/google/fonts/main/ofl/bangers/Bangers-Regular.ttf",
        }),
      ]),
    );
  });

  it("finds a Google font option by the stable cached filename", () => {
    expect(findEditorFontOption("GoogleFont__Bangers__regular.ttf")).toMatchObject({
      label: "Bangers",
      source: "google",
      value: "GoogleFont__Bangers__regular.ttf",
      downloadUrl: "https://raw.githubusercontent.com/google/fonts/main/ofl/bangers/Bangers-Regular.ttf",
    });
  });

  it("deduplicates bundle and Google families while keeping bundle entries first", () => {
    const catalog = buildEditorFontCatalog();
    const comicNeueOptions = catalog.filter((option) => option.cssFamily === "Comic Neue");

    expect(comicNeueOptions).toHaveLength(1);
    expect(comicNeueOptions[0]).toMatchObject({
      source: "bundle",
      value: "ComicNeue-Bold.ttf",
    });
  });

  it("returns grouped options for editor selects", () => {
    const groups = listEditorFontGroups();

    expect(groups.map((group) => group.label)).toEqual(["Embutidas", "Google Fonts"]);
    expect(groups[0].options.every((option) => option.source === "bundle")).toBe(true);
    expect(groups[1].options.every((option) => option.source === "google")).toBe(true);
  });

  it("filters Google Fonts by a search query while keeping grouped results", () => {
    const groups = searchEditorFontGroups("noto");

    expect(groups).toEqual([
      expect.objectContaining({
        label: "Google Fonts",
        source: "google",
        options: [
          expect.objectContaining({ label: "Noto Sans" }),
          expect.objectContaining({ label: "Noto Serif" }),
        ],
      }),
    ]);
  });

  it("normalizes font search queries across spaces and case", () => {
    const groups = searchEditorFontGroups("  rounded 1C ");

    expect(groups).toEqual([
      expect.objectContaining({
        label: "Google Fonts",
        options: [expect.objectContaining({ label: "M PLUS Rounded 1c" })],
      }),
    ]);
  });

  it("finds bundled CC Dave Gibbons with spaced and compact queries", () => {
    expect(searchEditorFontGroups("dave")).toEqual([
      expect.objectContaining({
        label: "Embutidas",
        source: "bundle",
        options: [expect.objectContaining({ label: "CC Dave Gibbons" })],
      }),
    ]);
    expect(searchEditorFontGroups("ccdave")).toEqual([
      expect.objectContaining({
        label: "Embutidas",
        options: [expect.objectContaining({ value: "CCDaveGibbonsLower W00 Regular.ttf" })],
      }),
    ]);
  });

  it("resolves Google cache filenames to cssFamily names", () => {
    expect(resolveLegacyFontFamily("GoogleFont__Bangers__regular.ttf")).toBe("Bangers");
    expect(resolveLegacyFontFamily("GoogleFont__M_PLUS_Rounded_1c__700.ttf")).toBe("M PLUS Rounded 1c");
  });

  it("converts remote Google Font search results into applicable editor options", () => {
    expect(
      googleFontSearchResultToOption({
        family: "Bebas Neue",
        css_family: "Bebas Neue",
        variant: "regular",
        filename: "GoogleFont__Bebas_Neue__regular.ttf",
        download_url: "https://raw.githubusercontent.com/google/fonts/main/ofl/bebasneue/BebasNeue-Regular.ttf",
        category: "Sans Serif",
      }),
    ).toMatchObject({
      label: "Bebas Neue",
      value: "GoogleFont__Bebas_Neue__regular.ttf",
      cssFamily: "Bebas Neue",
      source: "google",
      groupLabel: "Google Fonts",
      variant: "regular",
      downloadUrl: "https://raw.githubusercontent.com/google/fonts/main/ofl/bebasneue/BebasNeue-Regular.ttf",
    });
  });

  it("resolves dynamic Google Font cache filenames to cssFamily names", () => {
    expect(resolveLegacyFontFamily("GoogleFont__Bebas_Neue__regular.ttf")).toBe("Bebas Neue");
  });

  it("converts system font metadata into editor options", () => {
    expect(
      systemFontInfoToOption({
        family: "Arial",
        full_name: "Arial Regular",
        filename: "SystemFont__Arial__Regular.ttf",
        path: "C:/Windows/Fonts/arial.ttf",
        weight: "400",
        style: "normal",
        monospace: false,
      }),
    ).toMatchObject({
      label: "Arial",
      value: "SystemFont__Arial__Regular.ttf",
      cssFamily: "Arial",
      source: "system",
      groupLabel: "Sistema",
    });
  });

  it("detects persisted system font option values", () => {
    expect(isSystemFontValue("SystemFont__Arial__Regular.ttf")).toBe(true);
    expect(findEditorFontOption("SystemFont__Arial__Regular.ttf")).toMatchObject({
      label: "Arial",
      source: "system",
    });
  });
});

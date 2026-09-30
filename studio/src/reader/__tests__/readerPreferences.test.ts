import { describe, expect, it } from "vitest";
import { isLanguageAllowed, parseAllowedLanguages, toggleAllowedLanguage } from "../readerPreferences";

describe("reader language preferences", () => {
  it("treats an absent or malformed preference as all languages", () => {
    expect(parseAllowedLanguages(null)).toBeNull();
    expect(parseAllowedLanguages("not-json")).toBeNull();
    expect(parseAllowedLanguages("{}")) .toBeNull();
  });

  it("preserves a unique list of explicitly allowed language codes", () => {
    expect(parseAllowedLanguages('["pt-BR", "en", "pt-BR", ""]')).toEqual(["pt-BR", "en"]);
    expect(isLanguageAllowed(["pt-BR"], "pt-BR")).toBe(true);
    expect(isLanguageAllowed(["pt-BR"], "en")).toBe(false);
    expect(isLanguageAllowed(null, "ko")).toBe(true);
  });

  it("supports the Todos master option and individual switches", () => {
    const available = ["pt-BR", "en", "ko"];
    expect(toggleAllowedLanguage(["pt-BR"], "*", true, available)).toBeNull();
    expect(toggleAllowedLanguage(null, "en", false, available)).toEqual(["pt-BR", "ko"]);
    expect(toggleAllowedLanguage(["pt-BR"], "en", true, available)).toEqual(["pt-BR", "en"]);
    expect(toggleAllowedLanguage(["pt-BR", "en"], "pt-BR", false, available)).toEqual(["en"]);
  });
});

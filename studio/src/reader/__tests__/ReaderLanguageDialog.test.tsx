import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { ReaderLanguageDialog } from "../ReaderLanguageDialog";

const languages = [
  { code: "en", label: "Inglês", count: 12 },
  { code: "pt-BR", label: "Português (Brasil)", count: 4 },
  { code: "ko", label: "Coreano", count: 3 },
];

describe("ReaderLanguageDialog", () => {
  it("renders a Suwayomi-style allowed-language dialog with master and individual switches", () => {
    const html = renderToStaticMarkup(createElement(ReaderLanguageDialog, {
      open: true,
      languages,
      allowed: ["en", "pt-BR"],
      onCancel: () => undefined,
      onApply: () => undefined,
    }));

    expect(html).toContain('role="dialog"');
    expect(html).toContain("Idiomas permitidos");
    expect(html).toContain("Todos");
    expect(html).toContain("Inglês");
    expect(html).toContain("Português (Brasil)");
    expect(html).toContain("Cancelar");
    expect(html).toContain("Aplicar");
    expect((html.match(/checked=""/g) ?? []).length).toBe(2);
    expect((html.match(/aria-checked="true"/g) ?? []).length).toBe(2);
    expect((html.match(/aria-checked="false"/g) ?? []).length).toBe(2);
  });

  it("does not render when closed", () => {
    expect(renderToStaticMarkup(createElement(ReaderLanguageDialog, { open: false, languages, allowed: null, onCancel: () => undefined, onApply: () => undefined }))).toBe("");
  });
});

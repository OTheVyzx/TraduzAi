import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { StudioAppShell } from "../StudioAppShell";

describe("StudioAppShell", () => {
  it("renders one global navigation with the active view exposed to assistive technology", () => {
    const html = renderToStaticMarkup(createElement(StudioAppShell, {
      activeView: "library",
      onNavigate: () => undefined,
      children: createElement("section", null, "Conte\u00fado da biblioteca"),
    }));

    expect(html).toContain("TraduzAI Studio");
    expect(html).toContain("Home");
    expect(html).toContain("Tradutor");
    expect(html).toContain("Leitor");
    expect(html).toContain("Configura\u00e7\u00f5es");
    expect(html).toContain('aria-current="page"');
    expect(html).toContain("Conte\u00fado da biblioteca");
    expect(html).toContain('class="studio-global-footer"');
    expect(html.indexOf("studio-app-shell-content")).toBeLessThan(html.indexOf("studio-global-footer"));
    expect(html).not.toContain("Buscar obras, autores e g\u00eaneros");
    expect(html).not.toContain("Alternar visualiza\u00e7\u00e3o");
    expect(html).not.toContain("> Atualiza\u00e7\u00f5es</button>");
  });

  it("centers Tradutor, Home and Leitor while exposing settings as an icon-only action", () => {
    const html = renderToStaticMarkup(createElement(StudioAppShell, {
      activeView: "home",
      onNavigate: () => undefined,
      children: createElement("section", null, "Conte\u00fado"),
    }));

    const translatorIndex = html.indexOf(">Tradutor</button>");
    const homeIndex = html.indexOf(">Home</button>");
    const readerIndex = html.indexOf(">Leitor</button>");

    expect(html).toContain('class="studio-global-settings"');
    expect(html).toContain('aria-label="Configura\u00e7\u00f5es"');
    expect(html).toContain('title="Configura\u00e7\u00f5es"');
    expect(html).not.toContain(">Configura\u00e7\u00f5es</button>");
    expect(translatorIndex).toBeGreaterThan(-1);
    expect(translatorIndex).toBeLessThan(homeIndex);
    expect(homeIndex).toBeLessThan(readerIndex);
  });

  it("keeps reader failures inside an icon-only notification action in the top-right", () => {
    const html = renderToStaticMarkup(createElement(StudioAppShell, {
      activeView: "reader",
      onNavigate: () => undefined,
      notification: "SOURCE_TIMEOUT: o runtime excedeu o prazo",
      onDismissNotification: () => undefined,
      children: createElement("section", null, "Leitor"),
    }));

    expect(html).toContain('class="studio-global-notifications"');
    expect(html).toContain('aria-label="1 notificação"');
    expect(html).not.toContain("SOURCE_TIMEOUT: o runtime excedeu o prazo");
  });
});

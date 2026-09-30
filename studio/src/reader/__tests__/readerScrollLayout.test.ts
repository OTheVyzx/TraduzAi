import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const styles = readFileSync(new URL("../../styles.css", import.meta.url), "utf8");

function cssRule(selector: string): string {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return styles.match(new RegExp(`${escaped}\\s*\\{([^}]*)\\}`))?.[1] ?? "";
}

describe("Reader scroll layout", () => {
  it("keeps the Reader inside the viewport and gives the content pane its own vertical scrollbar", () => {
    expect(cssRule(".studio-app-shell")).toMatch(/display:\s*grid/);
    expect(cssRule(".studio-app-shell")).toMatch(/grid-template-rows:\s*var\(--studio-topbar-height\)\s+minmax\(0,\s*1fr\)\s+var\(--studio-bottom-nav-height\)/);
    expect(cssRule(".studio-reader-view")).toMatch(/height:\s*100%/);
    expect(cssRule(".studio-reader-view")).toMatch(/overflow:\s*hidden/);
    expect(cssRule(".studio-reader-main")).toMatch(/min-height:\s*0/);
    expect(cssRule(".studio-reader-main")).toMatch(/overflow-y:\s*auto/);
  });
});

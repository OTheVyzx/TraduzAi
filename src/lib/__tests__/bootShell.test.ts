import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

describe("static boot shell", () => {
  it("shows an immediate app shell instead of a blocking loading screen", () => {
    const html = readFileSync(new URL("../../../index.html", import.meta.url), "utf8");

    expect(html).toContain("data-html-boot-shell");
    expect(html).toContain("TraduzAi");
    expect(html).not.toContain("Carregando interface...");
  });
});

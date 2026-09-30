import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const shellStyles = readFileSync(new URL("../../styles.css", import.meta.url), "utf8");
const readerStyles = readFileSync(new URL("../readerNavigate.css", import.meta.url), "utf8");

describe("Reader TraduzAI brand theme", () => {
  it("defines one shared solid identity on the Studio shell", () => {
    expect(shellStyles).toContain("--studio-brand-violet: #7c5cff");
    expect(shellStyles).toContain("--studio-brand-cyan: #00d4ff");
    expect(shellStyles).toContain("--studio-surface-base: #111129");
    expect(shellStyles).toContain("--studio-text-base: #b1b0c0");
    expect(shellStyles).toContain("--studio-brand-glow:");
    expect(shellStyles).not.toMatch(/gradient\(/);
  });

  it("uses solid surface depth without tinting every content row", () => {
    expect(readerStyles).toContain("--reader-ambient-background:");
    expect(readerStyles).toContain("var(--studio-surface-base)");
    expect(readerStyles).toContain(".reader-manga-chapter-list article");
    expect(readerStyles).toContain("background: var(--reader-row-surface)");
    expect(readerStyles).not.toMatch(/gradient\(/);
  });

  it("applies solid accents and glow to active or primary Reader controls", () => {
    expect(readerStyles).toContain("background: var(--studio-brand-violet)");
    expect(readerStyles).toContain("box-shadow: var(--studio-brand-glow)");
    expect(readerStyles).toContain(".reader-cover-media:hover .reader-cover-open");
  });

  it("uses a seamless minimal footer and compact home metrics", () => {
    expect(shellStyles).toContain("grid-template-columns: 1fr auto 1fr");
    expect(shellStyles).toContain("grid-template-columns: repeat(3, 72px)");
    expect(shellStyles).toContain(".studio-global-settings");
    expect(shellStyles).toContain("grid-column: 3");
    expect(shellStyles).toContain("justify-self: end");
    expect(shellStyles).toContain("background: var(--studio-surface-base)");
    expect(shellStyles).toContain("min-height: 72px");
    expect(shellStyles).toContain("min-height: 100%");
  });

  it("integrates the footer with the canvas and uses restrained accessible motion", () => {
    expect(shellStyles).toMatch(/\.studio-global-footer\s*\{[^}]*border-top: 0;[^}]*background: var\(--studio-surface-base\);[^}]*box-shadow: 0 -14px 28px rgba\(17, 17, 41, 0\.9\);[^}]*backdrop-filter: none;/s);
    expect(shellStyles).toMatch(/\.studio-home-metric\s*\{[^}]*align-items: center;[^}]*justify-content: center;[^}]*animation: studio-home-metric-enter 180ms ease-out both;/s);
    expect(shellStyles).toMatch(/\.studio-home-metric-line\s*\{[^}]*align-items: center;/s);
    expect(shellStyles).toContain("@keyframes studio-home-metric-enter");
    expect(shellStyles).toMatch(/@media \(prefers-reduced-motion: reduce\)\s*\{[^}]*\.studio-home-metric/s);
  });
});

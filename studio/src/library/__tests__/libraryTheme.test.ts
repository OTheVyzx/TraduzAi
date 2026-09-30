import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const styles = readFileSync(new URL("../../styles.css", import.meta.url), "utf8");
const libraryHome = readFileSync(new URL("../StudioLibraryHome.tsx", import.meta.url), "utf8");
const toolbar = readFileSync(new URL("../LibraryToolbar.tsx", import.meta.url), "utf8");
const updatesView = readFileSync(new URL("../../tracking/UpdatesView.tsx", import.meta.url), "utf8");
const linkWorkDialog = readFileSync(new URL("../../tracking/LinkWorkDialog.tsx", import.meta.url), "utf8");

describe("library theme", () => {
  it("uses the solid Studio palette without gradients", () => {
    expect(styles).toContain("--studio-surface-base: #111129");
    expect(styles).toContain("--studio-text-base: #b1b0c0");
    expect(styles).not.toMatch(/gradient\(/);
  });

  it("removes the legacy local-library strip and centers compact chapter metrics", () => {
    expect(libraryHome).not.toContain("studio-library-topbar");
    expect(libraryHome).not.toContain("Biblioteca local");
    expect(libraryHome).toContain("studio-library-chapter-metric-line");
    expect(styles).toMatch(/\.studio-library-chapter-metrics article\s*\{[^}]*align-items:\s*center[^}]*justify-content:\s*center/s);
    expect(styles).toMatch(/\.studio-library-chapter-metric-line\s*\{[^}]*display:\s*inline-flex[^}]*align-items:\s*center[^}]*justify-content:\s*center/s);
  });

  it("unifies library controls on the base surface without dividing lines", () => {
    expect(libraryHome).not.toContain("studio-library-updates-row");
    expect(toolbar).toContain("studio-library-updates-trigger");
    expect(styles).toMatch(/\.studio-library-toolbar\s*\{[^}]*border(?:-bottom)?:\s*(?:0|none)[^}]*background:\s*var\(--studio-surface-base\)/s);
  });

  it("uses compact metrics and semantic resizable pane separators", () => {
    expect(styles).toMatch(/\.studio-library-chapter-metrics article\s*\{[^}]*min-height:\s*52px/s);
    expect(libraryHome).toContain('role="separator"');
    expect(libraryHome).toContain("studio-library-pane-divider-left");
    expect(libraryHome).toContain("studio-library-pane-divider-right");
    expect(styles).toMatch(/\.studio-library-layout\s*\{[^}]*var\(--studio-library-left-pane\)[^}]*var\(--studio-library-right-pane\)/s);
  });

  it("lets the shell own the viewport so footer actions never require document scrolling", () => {
    expect(styles).toMatch(/\.studio-home\s*\{[^}]*height:\s*100%[^}]*min-height:\s*0/s);
    expect(styles).toMatch(/\.studio-app-shell-content\s*\{[^}]*overflow:\s*hidden/s);
    expect(styles).not.toMatch(/\.studio-home\s*\{[^}]*height:\s*100dvh/s);
  });

  it("gives Updates and the bottom navigation one solid blurred identity", () => {
    expect(updatesView).toContain("studio-updates-backdrop");
    expect(updatesView).not.toContain("amber");
    expect(styles).toMatch(/\.studio-updates-backdrop\s*\{[^}]*background:\s*rgba\(17,\s*17,\s*41,[^}]*backdrop-filter:\s*blur\(/s);
    expect(styles).toMatch(/\.studio-updates-primary\s*\{[^}]*background:\s*var\(--studio-brand-violet\)/s);
    expect(styles).toMatch(/\.studio-global-footer\s*\{[^}]*background:\s*var\(--studio-surface-base\)/s);
  });

  it("keeps floating panels detached while the footer glass blurs behind sharp controls", () => {
    expect(styles).toMatch(/\.studio-floating-layer\s*\{[^}]*position:\s*fixed[^}]*inset:\s*0/s);
    expect(styles).toMatch(/\.studio-app-shell:has\(\.studio-floating-layer\) \.studio-global-footer\s*\{[^}]*backdrop-filter:\s*blur\(/s);
    expect(styles).toMatch(/\.studio-app-shell:has\(\.studio-floating-layer\) \.studio-global-nav[^{]*\{[^}]*position:\s*relative[^}]*z-index:/s);
    expect(styles).toMatch(/\.studio-updates-view\s*\{[^}]*height:\s*min\([^}]*margin:\s*clamp\(/s);
  });

  it("keeps every floating dialog on the violet solid palette", () => {
    expect(linkWorkDialog).toContain("studio-floating-layer");
    expect(linkWorkDialog).not.toMatch(/amber|yellow/);
  });

  it("joins the library to the global footer and uses only left-edge selection markers", () => {
    expect(styles).toMatch(/\.studio-global-footer\s*\{[^}]*border-top:\s*0[^}]*background:\s*var\(--studio-surface-base\)/s);
    expect(styles).toMatch(/\.studio-library-add-work,\s*\.studio-library-footer\s*\{[^}]*background:\s*var\(--studio-surface-base\)/s);
    expect(styles).toMatch(/\.studio-work-item\[aria-current="true"\]\s*\{[^}]*border-color:\s*transparent[^}]*box-shadow:\s*inset 3px 0 var\(--library-cyan\)/s);
    expect(styles).toMatch(/\.studio-global-nav button\[aria-current="page"\]\s*\{[^}]*border-color:\s*transparent[^}]*box-shadow:\s*inset 3px 0 var\(--studio-brand-cyan\)/s);
    expect(styles).toMatch(/\.studio-library-footer button\s*\{[^}]*background:\s*var\(--library-surface\)/s);
  });

  it("builds cinematic cover light from the cover itself and respects reduced motion", () => {
    expect(styles).toMatch(/\.studio-work-inspector-cover-ambient\s*\{[^}]*filter:\s*blur\([^)]*\)\s+saturate\([^)]*\)[^}]*animation:/s);
    expect(styles).toContain("@keyframes studio-cover-ambient-breathe");
    expect(styles).toMatch(/prefers-reduced-motion:\s*reduce[\s\S]*\.studio-work-inspector-cover-ambient[^{]*\{[^}]*animation:\s*none/s);
  });
});

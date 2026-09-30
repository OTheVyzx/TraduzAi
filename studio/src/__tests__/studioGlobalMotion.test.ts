import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const shellStyles = readFileSync(new URL("../styles.css", import.meta.url), "utf8");
const readerStyles = readFileSync(new URL("../reader/readerNavigate.css", import.meta.url), "utf8");

describe("Studio global motion system", () => {
  it("defines shared motion tokens and entrance primitives", () => {
    expect(shellStyles).toContain("--studio-motion-fast: 140ms");
    expect(shellStyles).toContain("--studio-motion-base: 180ms");
    expect(shellStyles).toContain("--studio-motion-view: 220ms");
    expect(shellStyles).toContain("--studio-motion-ease-standard: cubic-bezier(0.2, 0, 0, 1)");
    expect(shellStyles).toContain("@keyframes studio-view-enter");
    expect(shellStyles).toContain("@keyframes studio-dialog-enter");
    expect(shellStyles).toContain("@keyframes studio-backdrop-enter");
    expect(shellStyles).toContain("@keyframes studio-item-enter");
  });

  it("animates views, dialogs, controls, and repeated content without gradients", () => {
    expect(shellStyles).toMatch(/\.studio-app-shell-content > \*\s*\{[^}]*animation: studio-view-enter var\(--studio-motion-view\) var\(--studio-motion-ease-enter\) both;/s);
    expect(shellStyles).toMatch(/\.studio-dialog-backdrop\s*\{[^}]*animation: studio-backdrop-enter var\(--studio-motion-base\) var\(--studio-motion-ease-standard\) both;/s);
    expect(shellStyles).toMatch(/\.studio-dialog\s*\{[^}]*animation: studio-dialog-enter var\(--studio-motion-view\) var\(--studio-motion-ease-enter\) both;/s);
    expect(shellStyles).toContain("animation: studio-item-enter var(--studio-motion-view) var(--studio-motion-ease-enter) both");
    expect(shellStyles).toContain("transition-duration: var(--studio-motion-base)");
    expect(shellStyles).not.toMatch(/gradient\(/);
    expect(readerStyles).not.toMatch(/gradient\(/);
  });

  it("integrates Reader overlays and disables motion when requested", () => {
    expect(readerStyles).toMatch(/\.reader-language-backdrop\s*\{[^}]*animation: studio-backdrop-enter var\(--studio-motion-base\) var\(--studio-motion-ease-standard\) both;/s);
    expect(readerStyles).toMatch(/\.reader-language-dialog\s*\{[^}]*animation: studio-dialog-enter var\(--studio-motion-view\) var\(--studio-motion-ease-enter\) both;/s);
    expect(readerStyles).toContain("animation: studio-item-enter var(--studio-motion-view) var(--studio-motion-ease-enter) both");
    expect(shellStyles).toMatch(/@media \(prefers-reduced-motion: reduce\)[\s\S]*\.studio-app-shell \*[\s\S]*animation-duration: 0\.01ms !important;/);
    expect(readerStyles).toMatch(/@media \(prefers-reduced-motion: reduce\)[\s\S]*\.reader-language-backdrop[\s\S]*animation: none !important;/);
  });

  it("covers the editor chrome without animating the editing stage", () => {
    expect(shellStyles).toMatch(/\.studio-shared-editor\s*\{[^}]*animation: studio-view-enter var\(--studio-motion-view\) var\(--studio-motion-ease-enter\) both;/s);
    expect(shellStyles).toMatch(/\.studio-shared-editor :where\(button, input, select, textarea\)\s*\{[^}]*transition-duration: var\(--studio-motion-base\);/s);
    expect(shellStyles).not.toMatch(/\.studio-canvas-wrap\s*\{[^}]*animation:/s);
    expect(shellStyles).not.toMatch(/\.text-box\s*\{[^}]*animation:/s);
    expect(shellStyles).toMatch(/@media \(prefers-reduced-motion: reduce\)[\s\S]*\.studio-shared-editor \*[\s\S]*animation-duration: 0\.01ms !important;/);
  });
});

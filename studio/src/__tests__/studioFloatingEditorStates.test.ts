import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const app = readFileSync(new URL("../App.tsx", import.meta.url), "utf8");
const editor = readFileSync(new URL("../editor/StudioSharedEditor.tsx", import.meta.url), "utf8");
const styles = readFileSync(new URL("../styles.css", import.meta.url), "utf8");

describe("floating editor states", () => {
  it("keeps the editor loading panel inside the Studio shell", () => {
    expect(app).toContain("<Suspense fallback={(");
    expect(app).toContain("<StudioAppShell");
    expect(app).toContain('<StudioBoot message="Carregando editor..." />');
    expect(styles).toContain(".studio-boot-backdrop");
  });

  it("uses the shared floating identity for recovery without warning yellow", () => {
    expect(editor).toContain("studio-floating-layer studio-editor-recovery-backdrop");
    expect(editor).toContain("studio-editor-recovery-panel");
    expect(editor).not.toContain("bg-status-warning");
    expect(editor).not.toContain("border-status-warning");
  });

  it("identifies PSD as an editable diagnostic export, not the gated final result", () => {
    expect(editor).toContain("Exportar PSD editável");
    expect(editor).toContain("Não substitui a exportação final aprovada pelo gate");
    expect(editor).not.toContain("Salvar pagina atual em PSD");
  });
});

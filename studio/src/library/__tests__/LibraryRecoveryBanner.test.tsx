import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { LibraryRecoveryBanner } from "../LibraryRecoveryBanner";

describe("LibraryRecoveryBanner", () => {
  it("keeps backup recovery visible until the recovered copy is saved", () => {
    const html = renderToStaticMarkup(
      <LibraryRecoveryBanner
        recoveredFromBackup
        hasUnsavedChanges={false}
        error={null}
        saving={false}
        onSaveRecoveredCopy={() => undefined}
      />,
    );

    expect(html).toContain("Catálogo recuperado do backup");
    expect(html).toContain("Salvar cópia recuperada");
    expect(html).toContain('role="status"');
  });

  it("explains that a failed save remains in memory and offers an explicit retry", () => {
    const html = renderToStaticMarkup(
      <LibraryRecoveryBanner
        recoveredFromBackup={false}
        hasUnsavedChanges
        error="Disco indisponível"
        saving={false}
        onSaveRecoveredCopy={() => undefined}
      />,
    );

    expect(html).toContain("alterações continuam nesta sessão");
    expect(html).toContain("Disco indisponível");
    expect(html).toContain("Tentar salvar");
    expect(html).toContain('role="alert"');
  });
});

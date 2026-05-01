import { describe, expect, it } from "vitest";
import { exportBlockReason, exportModeForBackend } from "../exportModes";

const summary = {
  totalPages: 1,
  approvedPages: 0,
  warningPages: 0,
  blockedPages: 1,
  criticalCount: 1,
  warningCount: 0,
  groups: {},
};

describe("exportModes", () => {
  it("maps review package to backend warning export", () => {
    expect(exportModeForBackend("review_package")).toBe("with_warnings");
  });

  it("blocks clean export and allows debug", () => {
    expect(exportBlockReason("clean", summary)).toContain("Export limpo bloqueado");
    expect(exportBlockReason("debug", summary)).toBeNull();
  });
});

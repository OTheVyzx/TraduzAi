import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const source = readFileSync(
  fileURLToPath(new URL("../StudioSharedEditor.tsx", import.meta.url)),
  "utf8",
);

describe("StudioSharedEditor job integration", () => {
  it("keeps Consumer Fast controls visible in translation and editing workspaces", () => {
    expect(source).toContain('import { StudioJobControls } from "../jobs/StudioJobControls"');
    expect(source).toContain('import { StudioFinalExport } from "../export/StudioFinalExport"');
    expect(source).toContain("onPersisted={refreshProjectAfterBackendMutation}");
    expect(source).toContain("key={`export:${projectPath}:${translationProject.project_revision ?? \"unknown\"}`}");
    expect(source).toContain("headerActions={(");
  });

  it("binds the selected owner to the persisted Renderer A/B flow", () => {
    expect(source).toContain("<StudioTranslationWorkspace\n            project={translationProject}\n            projectPath={projectPath}");
    expect(source).toContain("createTauriStudioIpcClient().retypesetOwner");
    expect(source).toContain("error={translationCommitError}");
  });
});

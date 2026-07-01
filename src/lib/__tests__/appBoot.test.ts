import React from "react";
import { renderToString } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import App from "../../App";

const tauriMocks = vi.hoisted(() => {
  const pending = new Promise<never>(() => undefined);
  return {
    checkModels: vi.fn(() => pending),
    getCredits: vi.fn(() => pending),
    getSystemProfile: vi.fn(() => pending),
    onPipelineProgress: vi.fn(() => pending),
    warmupVisualStack: vi.fn(() => pending),
    openFiles: vi.fn(() => pending),
    openMultipleSources: vi.fn(() => pending),
    openProjectDialog: vi.fn(() => pending),
    loadProjectJson: vi.fn(() => pending),
    loadSettings: vi.fn(() => pending),
    validateImport: vi.fn(() => pending),
    openLabWindow: vi.fn(() => pending),
    restartApp: vi.fn(() => pending),
  };
});

vi.mock("react-router-dom", async () => {
  const ReactModule = await import("react");
  const actual = await vi.importActual<typeof import("react-router-dom")>("react-router-dom");

  return {
    ...actual,
    BrowserRouter: ({ children }: { children: React.ReactNode }) =>
      ReactModule.createElement(actual.MemoryRouter, { initialEntries: ["/"] }, children),
  };
});

vi.mock("../tauri", () => tauriMocks);

vi.mock("../appPreferences", () => ({
  applyAppPreferences: vi.fn(),
  getAppPreferences: vi.fn(() => ({ reduceMotion: false, compactMode: false })),
  watchSystemTheme: vi.fn(() => vi.fn()),
}));

vi.mock("../e2e/fixtureProject", () => ({
  installE2EFixtureProject: vi.fn(),
}));

describe("App boot", () => {
  it("renders the main shell at / without waiting for async startup calls", () => {
    const html = renderToString(React.createElement(App));

    expect(html).toContain('data-testid="app-shell"');
    expect(html).toContain('data-testid="home-content"');
    expect(html).not.toContain("Carregando tela...");
    expect(html).not.toContain("Inicializando...");
    expect(tauriMocks.checkModels).not.toHaveBeenCalled();
    expect(tauriMocks.getSystemProfile).not.toHaveBeenCalled();
    expect(tauriMocks.getCredits).not.toHaveBeenCalled();
  });
});

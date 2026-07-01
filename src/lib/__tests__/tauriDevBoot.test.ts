import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

describe("Tauri dev boot server", () => {
  it("uses the production preview server for the desktop dev window", () => {
    const packageJson = JSON.parse(readFileSync(new URL("../../../package.json", import.meta.url), "utf8")) as {
      scripts?: Record<string, string>;
    };
    const tauriConfig = JSON.parse(readFileSync(new URL("../../../src-tauri/tauri.conf.json", import.meta.url), "utf8")) as {
      build?: { devUrl?: string; beforeDevCommand?: string };
    };

    expect(packageJson.scripts?.["dev:tauri"]).toContain("preview");
    expect(tauriConfig.build?.devUrl).toBe("http://127.0.0.1:1420");
    expect(tauriConfig.build?.beforeDevCommand).toBe("npm run dev:tauri");
  });
});

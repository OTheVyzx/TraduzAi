import { afterEach, describe, expect, it, vi } from "vitest";
import { isE2E } from "../tauri";

describe("isE2E", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("does not enable E2E mocks just because the Tauri global is absent", () => {
    vi.stubGlobal("window", {});
    vi.stubGlobal("navigator", { webdriver: false });

    expect(isE2E()).toBe(false);
  });

  it("enables E2E mocks under an automated browser", () => {
    vi.stubGlobal("navigator", { webdriver: true });

    expect(isE2E()).toBe(true);
  });
});

import { describe, expect, it } from "vitest";
import { cacheBustImageSource, getViteDevImageSource } from "../imageSource";

describe("getViteDevImageSource", () => {
  it("maps Windows absolute files to Vite /@fs URLs on localhost", () => {
    expect(
      getViteDevImageSource("N:\\TraduzAI\\TraduzAi\\data\\works\\abc\\translated\\001.jpg", {
        protocol: "http:",
        hostname: "localhost",
      } as Location),
    ).toBe("/@fs/N:/TraduzAI/TraduzAi/data/works/abc/translated/001.jpg");
  });

  it("does not map local files outside a Vite dev origin", () => {
    expect(
      getViteDevImageSource("N:/TraduzAI/TraduzAi/data/works/abc/translated/001.jpg", {
        protocol: "tauri:",
        hostname: "localhost",
      } as Location),
    ).toBeNull();
  });

  it("adds a cache-busting version to Vite file URLs", () => {
    expect(
      getViteDevImageSource(
        "N:/TraduzAI/TraduzAi/data/works/abc/layers/brush/001.png",
        {
          protocol: "http:",
          hostname: "localhost",
        } as Location,
        123,
      ),
    ).toBe("/@fs/N:/TraduzAI/TraduzAi/data/works/abc/layers/brush/001.png?v=123");
  });
});

describe("cacheBustImageSource", () => {
  it("keeps data and blob URLs unchanged", () => {
    expect(cacheBustImageSource("data:image/png;base64,abc", 123)).toBe("data:image/png;base64,abc");
    expect(cacheBustImageSource("blob:http://localhost/abc", 123)).toBe("blob:http://localhost/abc");
  });

  it("uses an additional query parameter when the URL already has one", () => {
    expect(cacheBustImageSource("asset://localhost/image.png?old=1", 123)).toBe(
      "asset://localhost/image.png?old=1&v=123",
    );
  });
});

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  cacheBustImageSource,
  getViteDevImageSource,
  loadImageSource,
  localPathFromAssetSource,
  preloadImageSource,
} from "../imageSource";

const { convertFileSrcMock, readFileMock } = vi.hoisted(() => ({
  convertFileSrcMock: vi.fn((path: string) => `asset://localhost/${encodeURIComponent(path)}`),
  readFileMock: vi.fn(async () => new Uint8Array([1, 2, 3])),
}));

vi.mock("@tauri-apps/api/core", () => ({
  convertFileSrc: convertFileSrcMock,
}));

vi.mock("@tauri-apps/plugin-fs", () => ({
  readFile: readFileMock,
}));

beforeEach(() => {
  convertFileSrcMock.mockClear();
  readFileMock.mockClear();
  vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:local-image");
  vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
});

afterEach(() => {
  vi.restoreAllMocks();
});

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

  it("does not map local files on Tauri localhost asset origins", () => {
    expect(
      getViteDevImageSource("N:/TraduzAI/TraduzAi/data/works/abc/translated/001.jpg", {
        protocol: "http:",
        hostname: "tauri.localhost",
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

describe("localPathFromAssetSource", () => {
  it("decodes asset.localhost URLs back to local Windows paths", () => {
    expect(
      localPathFromAssetSource(
        "http://asset.localhost/N%3A%2FTraduzAI%2FTraduzAi%2Fdata%2Fworks%2Fabc%2Ftranslated%2F001.jpg",
      ),
    ).toBe("N:/TraduzAI/TraduzAi/data/works/abc/translated/001.jpg");
  });

  it("decodes asset protocol URLs back to local Windows paths", () => {
    expect(
      localPathFromAssetSource(
        "asset://localhost/N%3A%2FTraduzAI%2FTraduzAi%2Fdata%2Fworks%2Fabc%2Foriginals%2F001.jpg",
      ),
    ).toBe("N:/TraduzAI/TraduzAi/data/works/abc/originals/001.jpg");
  });

  it("ignores normal HTTP image URLs", () => {
    expect(localPathFromAssetSource("https://example.com/image.jpg")).toBeNull();
  });
});

describe("preloadImageSource", () => {
  it("reuses a preloaded direct image source for the next load", async () => {
    const src = "data:image/png;base64,abc";
    const preloaded = await preloadImageSource(src, "image/png");
    const loaded = await loadImageSource(src, "image/png");

    expect(preloaded.src).toBe(src);
    expect(loaded.src).toBe(src);
    expect(loaded.revoke).toBeUndefined();
  });

  it("uses a blob URL for Windows paths before protocol fallbacks", async () => {
    const loaded = await loadImageSource(
      "N:/TraduzAI/TraduzAi/data/works/abc/images/001.jpg",
      "image/png",
      456,
    );

    expect(readFileMock).toHaveBeenCalledWith("N:/TraduzAI/TraduzAi/data/works/abc/images/001.jpg");
    expect(convertFileSrcMock).not.toHaveBeenCalled();
    expect(loaded.src).toBe("blob:local-image");
    expect(loaded.revoke).toBeTypeOf("function");
  });

  it("falls back to the Tauri asset protocol when FS blob loading fails", async () => {
    readFileMock.mockRejectedValueOnce(new Error("fs denied"));

    const loaded = await loadImageSource(
      "N:/TraduzAI/TraduzAi/data/works/abc/images/001.jpg",
      "image/png",
      456,
    );

    expect(convertFileSrcMock).toHaveBeenCalledWith("N:/TraduzAI/TraduzAi/data/works/abc/images/001.jpg");
    expect(loaded.src).toBe(
      "asset://localhost/N%3A%2FTraduzAI%2FTraduzAi%2Fdata%2Fworks%2Fabc%2Fimages%2F001.jpg?v=456",
    );
  });
});

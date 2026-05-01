import { beforeEach, describe, expect, it, vi } from "vitest";
import type { PageData } from "../stores/appStore";

const invokeMock = vi.hoisted(() => vi.fn());

vi.mock("@tauri-apps/api/core", () => ({
  invoke: invokeMock,
}));

vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn(),
}));

describe("renderPreviewPage binding", () => {
  beforeEach(() => {
    invokeMock.mockReset();
  });

  it("invokes the non-destructive Tauri preview command with materialized page data", async () => {
    const { renderPreviewPage } = await import("../tauri");
    const page = {
      numero: 1,
      arquivo_original: "originals/001.png",
      arquivo_traduzido: "translated/001.png",
      image_layers: {},
      text_layers: [],
      textos: [],
    } satisfies PageData;
    invokeMock.mockResolvedValue("D:/tmp/project/render-cache/preview/001-preview.png");

    await expect(
      renderPreviewPage({
        project_path: "D:/tmp/project",
        page_index: 0,
        page,
        fingerprint: "abc123",
      }),
    ).resolves.toBe("D:/tmp/project/render-cache/preview/001-preview.png");

    expect(invokeMock).toHaveBeenCalledWith("render_preview_page", {
      config: {
        project_path: "D:/tmp/project",
        page_index: 0,
        page,
        fingerprint: "abc123",
      },
    });
  });
});

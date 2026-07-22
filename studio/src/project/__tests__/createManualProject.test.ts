import { describe, expect, it } from "vitest";
import { importStudioProject, toTraduzAiV2Compat } from "../adapters";
import { createManualProject } from "../createManualProject";
import {
  ManualChapterCreationError,
  createManualChapterFromImages,
  type PreparedManualPage,
} from "../../backend/projectDialog";

describe("createManualProject", () => {
  it("creates normalized pages with a base raster scene", () => {
    const project = createManualProject({
      workTitle: "Obra",
      chapterLabel: "12",
      chapterTitle: "O reencontro",
      sourceLanguage: "en",
      targetLanguage: "pt-BR",
      pages: [
        { number: 1, relativePath: "original/001.webp", width: 800, height: 1200 },
        { number: 2, relativePath: "original/002.png", width: 900, height: 1400 },
      ],
    });

    expect(project).toMatchObject({
      app: "traduzai",
      versao: "2.0",
      studio_schema_version: "1.0",
      obra: "Obra",
      capitulo: "12",
      chapter_title: "O reencontro",
      idioma_origem: "en",
      idioma_destino: "pt-BR",
    });
    expect(project.paginas).toHaveLength(2);
    expect(project.paginas[0].image_layers.base).toMatchObject({
      key: "base",
      path: "original/001.webp",
      visible: true,
      locked: true,
    });
    expect(project.paginas[0].arquivo_original).toBe("original/001.webp");
    expect(project.paginas[0].text_layers).toEqual([]);
    expect(project.paginas[0].textos).toEqual([]);
    expect(project.paginas[0].studio_scene.roots).toEqual(["image:base"]);
    expect(project.paginas[0].studio_scene.nodes[0]).toMatchObject({
      id: "image:base",
      kind: "raster",
      image_layer_key: "base",
    });
  });

  it("round-trips through the existing compatibility adapter", () => {
    const created = createManualProject({
      workTitle: "Obra",
      chapterLabel: "Especial",
      sourceLanguage: "ja",
      targetLanguage: "pt-BR",
      pages: [{ number: 1, relativePath: "original/page.jpg", width: 720, height: 1280 }],
    });

    const reopened = importStudioProject(toTraduzAiV2Compat(created)).project;

    expect(reopened.obra).toBe("Obra");
    expect(reopened.capitulo).toBe("Especial");
    expect(reopened.paginas[0].image_layers.base?.path).toBe("original/page.jpg");
    expect(reopened.paginas[0].studio_scene.roots).toContain("image:base");
  });

  it("rejects a manual chapter without valid pages", () => {
    expect(() => createManualProject({
      workTitle: "Obra",
      chapterLabel: "1",
      sourceLanguage: "en",
      targetLanguage: "pt-BR",
      pages: [],
    })).toThrow("página");
  });

  it("prepares, saves and returns a manual project through the existing backend contract", async () => {
    const calls: string[] = [];
    let savedProjectPath: string | null | undefined;
    const pages: PreparedManualPage[] = [
      { number: 1, relativePath: "original/001.png", width: 800, height: 1200 },
    ];
    const result = await createManualChapterFromImages({
      workTitle: "Obra",
      chapterLabel: "3",
      sourceLanguage: "ko",
      targetLanguage: "pt-BR",
      sourcePath: "N:/entrada/capitulo.cbz",
      projectJsonPath: "N:/biblioteca/obra/003/project.json",
    }, {
      prepare: async (sourcePath, projectJsonPath) => {
        calls.push(`prepare:${sourcePath}:${projectJsonPath}`);
        return pages;
      },
      save: async (projectJsonPath, project) => {
        calls.push(`save:${projectJsonPath}:${project.paginas.length}`);
        savedProjectPath = project.source_path;
      },
    });

    expect(calls).toEqual([
      "prepare:N:/entrada/capitulo.cbz:N:/biblioteca/obra/003/project.json",
      "save:N:/biblioteca/obra/003/project.json:1",
    ]);
    expect(result.project.obra).toBe("Obra");
    expect(savedProjectPath).toBe("N:/biblioteca/obra/003/project.json");
    expect(result.project.paginas[0].image_layers.base?.path).toBe("original/001.png");
    expect(result.preparedPages).toEqual(pages);
  });

  it("retains prepared pages for a retry when saving project.json fails", async () => {
    const pages: PreparedManualPage[] = [
      { number: 1, relativePath: "original/001.png", width: 800, height: 1200 },
    ];
    let prepareCalls = 0;
    const runtime = {
      prepare: async () => {
        prepareCalls += 1;
        return pages;
      },
      save: async () => {
        throw new Error("disco indisponível");
      },
    };

    const firstError = await createManualChapterFromImages({
      workTitle: "Obra",
      chapterLabel: "3",
      sourceLanguage: "ko",
      targetLanguage: "pt-BR",
      sourcePath: "N:/entrada",
      projectJsonPath: "N:/saida/project.json",
    }, runtime).catch((error: unknown) => error);

    expect(firstError).toBeInstanceOf(ManualChapterCreationError);
    expect((firstError as ManualChapterCreationError).preparedPages).toEqual(pages);

    await createManualChapterFromImages({
      workTitle: "Obra",
      chapterLabel: "3",
      sourceLanguage: "ko",
      targetLanguage: "pt-BR",
      sourcePath: "N:/entrada",
      projectJsonPath: "N:/saida/project.json",
    }, { ...runtime, save: async () => undefined }, pages);

    expect(prepareCalls).toBe(1);
  });
});

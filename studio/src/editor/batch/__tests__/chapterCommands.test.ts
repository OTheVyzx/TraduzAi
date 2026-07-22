import { describe, expect, it } from "vitest";
import { importStudioProject } from "../../../project/adapters";
import {
  buildChapterReviewQueue,
  copyStyleFromLayer,
  createApplyStyleCommand,
  createReplaceTextCommand,
  createResolveReviewCommand,
  previewChapterReplacements,
  restoreChapterCommand,
} from "../chapterCommands";

function projectFixture() {
  return importStudioProject({
    app: "traduzai",
    versao: "2.0",
    obra: "Capitulo QA",
    paginas: [
      {
        numero: 1,
        image_layers: {},
        text_layers: [
          {
            id: "p1-source",
            kind: "text",
            original: "Hello hero",
            translated: "Ola heroi",
            traduzido: "Ola heroi",
            bbox: [0, 0, 100, 40],
            style: {
              fonte: "CC Wild Words",
              tamanho: 30,
              studio_style: {
                version: "1.0",
                typography: { font_family: "CC Wild Words", font_size: 30 },
                fills: [{ type: "solid", color: "#111111", opacity: 1 }],
                strokes: [],
                effects: {},
              },
            },
            visible: true,
            locked: false,
            order: 0,
          },
          {
            id: "p1-target",
            kind: "text",
            original: "Hero returns",
            translated: "O heroi voltou",
            traduzido: "O heroi voltou",
            bbox: [0, 50, 100, 90],
            style: { fonte: "Arial", tamanho: 18 },
            visible: true,
            locked: false,
            order: 1,
            review_required: true,
            qa_flags: ["ocr_low_confidence"],
          },
        ],
      },
      {
        numero: 2,
        image_layers: {},
        text_layers: [
          {
            id: "p2-target",
            kind: "text",
            original: "The hero wins",
            translated: "O HEROI vence",
            traduzido: "O HEROI vence",
            bbox: [0, 0, 100, 40],
            style: { fonte: "Arial", tamanho: 16 },
            visible: true,
            locked: false,
            order: 0,
          },
          {
            id: "p2-empty",
            kind: "text",
            original: "Translate me",
            translated: "",
            traduzido: "",
            bbox: [0, 50, 100, 90],
            style: {},
            visible: true,
            locked: false,
            order: 1,
          },
        ],
      },
    ],
  }).project;
}

describe("comandos de produtividade do capitulo", () => {
  it("copia e aplica o estilo profissional sem compartilhar referencias mutaveis", () => {
    const project = projectFixture();
    const clipboard = copyStyleFromLayer(project, { pageIndex: 0, layerId: "p1-source" });
    const command = createApplyStyleCommand(project, clipboard, [
      { pageIndex: 0, layerId: "p1-target" },
      { pageIndex: 1, layerId: "p2-target" },
    ]);

    const first = command.after.paginas[0].text_layers[1];
    const second = command.after.paginas[1].text_layers[0];
    expect(first.style).toEqual(project.paginas[0].text_layers[0].style);
    expect(first.estilo).toEqual(first.style);
    expect(second.style).toEqual(first.style);
    expect(second.style).not.toBe(first.style);
    expect(first.translated).toBe("O heroi voltou");
    expect(project.paginas[0].text_layers[1].style).toMatchObject({ fonte: "Arial" });

    const restored = restoreChapterCommand(command, "undo");
    expect(restored.paginas[0].text_layers[1].style).toMatchObject({ fonte: "Arial" });
  });

  it("previsualiza e substitui texto no capitulo preservando translated/traduzido/textos", () => {
    const project = projectFixture();
    const matches = previewChapterReplacements(project, {
      query: "heroi",
      replacement: "protagonista",
      caseSensitive: false,
      wholeWord: true,
    });

    expect(matches.map((item) => [item.pageIndex, item.layerId, item.occurrences])).toEqual([
      [0, "p1-source", 1],
      [0, "p1-target", 1],
      [1, "p2-target", 1],
    ]);

    const command = createReplaceTextCommand(project, matches);
    const page2Layer = command.after.paginas[1].text_layers[0];
    expect(page2Layer.translated).toBe("O protagonista vence");
    expect(page2Layer.traduzido).toBe("O protagonista vence");
    expect(command.after.paginas[1].textos[0]).toEqual(page2Layer);
    expect(project.paginas[1].text_layers[0].translated).toBe("O HEROI vence");
  });

  it("ignora camadas bloqueadas em aplicar estilo e localizar/substituir", () => {
    const project = projectFixture();
    project.paginas[1].text_layers[0].locked = true;
    project.paginas[1].textos = project.paginas[1].text_layers;
    const clipboard = copyStyleFromLayer(project, { pageIndex: 0, layerId: "p1-source" });
    const styleCommand = createApplyStyleCommand(project, clipboard, [
      { pageIndex: 0, layerId: "p1-target" },
      { pageIndex: 1, layerId: "p2-target" },
    ]);

    expect(styleCommand.affectedLayers).toEqual([{ pageIndex: 0, layerId: "p1-target" }]);
    expect(styleCommand.after.paginas[1].text_layers[0].style).toMatchObject({ fonte: "Arial" });
    expect(previewChapterReplacements(project, {
      query: "heroi",
      replacement: "protagonista",
      caseSensitive: false,
      wholeWord: true,
    }).some((item) => item.layerId === "p2-target")).toBe(false);
  });

  it("monta a fila de revisao e resolve itens sem apagar a evidencia de QA", () => {
    const project = projectFixture();
    const queue = buildChapterReviewQueue(project);

    expect(queue.map((item) => item.id)).toEqual([
      "text:0:p1-target",
      "text:1:p2-empty",
    ]);
    expect(queue[0].reasons).toEqual(expect.arrayContaining(["Revisao solicitada", "ocr_low_confidence"]));
    expect(queue[1].reasons).toContain("Traducao vazia");

    const command = createResolveReviewCommand(project, [queue[0]], "2026-07-13T12:00:00.000Z");
    const resolved = command.after.paginas[0].text_layers[1];
    expect(resolved.qa_flags).toEqual(["ocr_low_confidence"]);
    expect(resolved.studio_review).toMatchObject({
      status: "resolved",
      resolved_at: "2026-07-13T12:00:00.000Z",
    });
    expect(buildChapterReviewQueue(command.after).map((item) => item.id)).toEqual(["text:1:p2-empty"]);

    command.after.paginas[0].text_layers[1].translated = "";
    command.after.paginas[0].text_layers[1].traduzido = "";
    expect(buildChapterReviewQueue(command.after).map((item) => item.id)).toEqual([
      "text:0:p1-target",
      "text:1:p2-empty",
    ]);
  });
});

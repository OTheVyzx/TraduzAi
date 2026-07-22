import { describe, expect, it } from "vitest";
import type { Project, TextLayerStyle } from "../../../../src/lib/stores/appStore";
import { reconcileStudioEditorPage } from "../sharedEditorProjectSync";

function projectFixture(): Project {
  return {
    id: "studio-sync",
    obra: "QA",
    capitulo: 1,
    idioma_origem: "en",
    idioma_destino: "pt-BR",
    qualidade: "normal",
    contexto: {} as Project["contexto"],
    paginas: [
      { numero: 1, arquivo_original: "1.png", arquivo_traduzido: "1.png", text_layers: [], textos: [], image_layers: {} },
      {
        numero: 2,
        arquivo_original: "2.png",
        arquivo_traduzido: "2.png",
        text_layers: [{
          id: "selected",
          bbox: [0, 0, 1, 1],
          original: "",
          traduzido: "",
          tipo: "fala",
          confianca_ocr: 1,
          estilo: {} as TextLayerStyle,
        }],
        textos: [],
        image_layers: {},
      },
    ],
    status: "done",
    source_path: "memory://sync",
    output_path: "memory://sync",
    totalPages: 2,
    mode: "manual",
  };
}

describe("reconcileStudioEditorPage", () => {
  it("preserva a pagina e a selecao quando o mesmo documento recebe um save interno", () => {
    const result = reconcileStudioEditorPage(projectFixture(), 1, "selected");
    expect(result.currentPageIndex).toBe(1);
    expect(result.currentPage?.numero).toBe(2);
    expect(result.selectedLayerId).toBe("selected");
  });

  it("remove apenas uma selecao que nao existe mais", () => {
    expect(reconcileStudioEditorPage(projectFixture(), 1, "removed").selectedLayerId).toBeNull();
  });

  it("preserva integralmente a pagina de trabalho enquanto existem mudancas locais", () => {
    const project = projectFixture();
    const workingPage = structuredClone(project.paginas[1]);
    workingPage.text_layers.push({
      ...workingPage.text_layers[0],
      id: "created-dirty",
    });
    const remoteProject = projectFixture();
    remoteProject.paginas[1].text_layers[0].traduzido = "Atualizacao remota";

    const result = reconcileStudioEditorPage(remoteProject, 1, "created-dirty", workingPage, true);

    expect(result.currentPage).toBe(workingPage);
    expect(result.currentPage?.text_layers.some((layer) => layer.id === "created-dirty")).toBe(true);
    expect(result.selectedLayerId).toBe("created-dirty");
  });
});

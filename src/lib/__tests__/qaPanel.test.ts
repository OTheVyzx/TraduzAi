import { describe, expect, it } from "vitest";
import {
  collectQaIssues,
  ignoreQaIssue,
  qaIssueLabel,
} from "../qaPanel";
import type { Project, TextEntry } from "../stores/appStore";

function buildTextLayer(overrides: Partial<TextEntry> = {}): TextEntry {
  return {
    id: "txt-1",
    bbox: [10, 20, 80, 90],
    tipo: "fala",
    original: "YOUNG MASTER?!",
    traduzido: "Jovem mestre?!",
    confianca_ocr: 0.9,
    estilo: {
      fonte: "ComicNeue-Bold.ttf",
      tamanho: 24,
      cor: "#ffffff",
      cor_gradiente: [],
      contorno: "#000000",
      contorno_px: 2,
      glow: false,
      glow_cor: "",
      glow_px: 0,
      sombra: false,
      sombra_cor: "",
      sombra_offset: [0, 0],
      bold: true,
      italico: false,
      rotacao: 0,
      alinhamento: "center",
    },
    ...overrides,
  };
}

function buildProject(textLayer: TextEntry): Project {
  return {
    id: "qa-project",
    obra: "QA Fixture",
    capitulo: 1,
    idioma_origem: "en",
    idioma_destino: "pt-BR",
    qualidade: "normal",
    contexto: {
      sinopse: "",
      genero: [],
      personagens: [],
      glossario: {},
      aliases: [],
      termos: [],
      relacoes: [],
      faccoes: [],
      resumo_por_arco: [],
      memoria_lexical: {},
      fontes_usadas: [],
    },
    paginas: [
      {
        numero: 1,
        arquivo_original: "original.png",
        arquivo_traduzido: "translated.png",
        text_layers: [textLayer],
        textos: [textLayer],
      },
    ],
    status: "done",
    source_path: "project.json",
    totalPages: 1,
    mode: "manual",
  };
}

describe("qaPanel", () => {
  it("collects active QA flags from text layers", () => {
    const project = buildProject(
      buildTextLayer({
        qa_flags: ["visual_text_leak", "glossary_violation"],
      }),
    );

    expect(collectQaIssues(project)).toMatchObject([
      {
        id: "0:txt-1:visual_text_leak",
        flagId: "visual_text_leak",
        pageIndex: 0,
        pageNumber: 1,
        regionId: "txt-1",
        severity: "critical",
      },
      {
        id: "0:txt-1:glossary_violation",
        flagId: "glossary_violation",
        pageIndex: 0,
        pageNumber: 1,
        regionId: "txt-1",
        severity: "high",
      },
    ]);
    expect(qaIssueLabel("visual_text_leak")).toBe("Ingles restante");
  });

  it("requires an ignore reason and hides ignored flags", () => {
    const project = buildProject(
      buildTextLayer({
        qa_flags: ["visual_text_leak"],
      }),
    );

    expect(() => ignoreQaIssue(project, "0:txt-1:visual_text_leak", "   ")).toThrow(
      "Informe o motivo para ignorar esta flag.",
    );

    const updated = ignoreQaIssue(project, "0:txt-1:visual_text_leak", "SFX preservado propositalmente");

    expect(collectQaIssues(updated)).toEqual([]);
    expect(updated.paginas[0].text_layers[0].qa_actions).toMatchObject([
      {
        flag_id: "visual_text_leak",
        status: "ignored",
        ignored_reason: "SFX preservado propositalmente",
      },
    ]);
    expect(updated.paginas[0].textos[0].qa_actions).toMatchObject(
      updated.paginas[0].text_layers[0].qa_actions ?? [],
    );
  });
});

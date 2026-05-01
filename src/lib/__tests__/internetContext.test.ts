import { describe, expect, it } from "vitest";
import {
  applyHighConfidenceInternetCandidates,
  countInternetContextKinds,
  sourceStatusLabel,
  type InternetContextResult,
} from "../internetContext";
import type { ProjectContext } from "../stores/appStore";

function emptyContext(): ProjectContext {
  return {
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
  };
}

const result: InternetContextResult = {
  title: "The Regressed Mercenary Has a Plan",
  synopsis: "A mercenary returns with a plan.",
  genres: ["Action", "Fantasy"],
  internet_context_loaded: true,
  context_quality: "partial",
  source_results: [
    { source: "anilist", status: "found", confidence: 0.95, title: "The Regressed Mercenary Has a Plan", url: "https://anilist.co/manga/fixture" },
    { source: "fandom", status: "found", confidence: 0.48, title: "Fan Wiki", url: "https://example.test/wiki" },
    { source: "myanimelist", status: "unavailable", confidence: 0, error: "api key ausente" },
  ],
  glossary_candidates: [
    { kind: "character", source: "Ghislain Perdium", target: "Ghislain Perdium", confidence: 0.95, sources: ["anilist"], status: "candidate", protect: true, aliases: [], forbidden: [], notes: "" },
    { kind: "place", source: "Cavald", target: "Cavald", confidence: 0.78, sources: ["fandom"], status: "candidate", protect: true, aliases: [], forbidden: [], notes: "" },
    { kind: "term", source: "mana technique", target: "Arte de mana", confidence: 1, sources: ["anilist"], status: "reviewed", protect: true, aliases: [], forbidden: [], notes: "" },
  ],
};

describe("internetContext", () => {
  it("labels source statuses in Portuguese", () => {
    expect(sourceStatusLabel("found")).toBe("encontrado");
    expect(sourceStatusLabel("unavailable")).toBe("indisponivel");
  });

  it("counts candidates by product categories", () => {
    expect(countInternetContextKinds(result)).toEqual({
      characters: 1,
      placesAndFactions: 1,
      loreTerms: 1,
      aliases: 0,
    });
  });

  it("applies high-confidence candidates without overwriting reviewed glossary", () => {
    const applied = applyHighConfidenceInternetCandidates(emptyContext(), result, {
      "mana technique": "Tecnica de mana revisada",
    });

    expect(applied.contexto.sinopse).toBe(result.synopsis);
    expect(applied.contexto.genero).toEqual(["Action", "Fantasy"]);
    expect(applied.contexto.personagens).toContain("Ghislain Perdium");
    expect(applied.contexto.glossario["Ghislain Perdium"]).toBe("Ghislain Perdium");
    expect(applied.contexto.glossario["Cavald"]).toBeUndefined();
    expect(applied.contexto.glossario["mana technique"]).toBe("Tecnica de mana revisada");
    expect(applied.appliedCount).toBe(1);
  });
});

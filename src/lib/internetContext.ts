import type { ContextSourceRef, ProjectContext } from "./stores/appStore";

export type InternetContextSourceStatus = "found" | "not_found" | "unavailable" | "error" | "cached";
export type InternetContextCandidateStatus = "candidate" | "auto" | "reviewed" | "rejected";

export interface InternetContextSourceResult {
  source: string;
  status: InternetContextSourceStatus;
  confidence: number;
  title?: string;
  synopsis?: string;
  genres?: string[];
  tags?: string[];
  url?: string;
  error?: string;
}

export interface InternetContextCandidate {
  kind: string;
  source: string;
  target: string;
  confidence: number;
  sources: string[];
  status: InternetContextCandidateStatus;
  protect: boolean;
  aliases: string[];
  forbidden: string[];
  notes: string;
}

export interface InternetContextResult {
  title: string;
  synopsis: string;
  genres: string[];
  source_results: InternetContextSourceResult[];
  glossary_candidates: InternetContextCandidate[];
  internet_context_loaded: boolean;
  context_quality: "empty" | "partial" | "reviewed";
}

export function sourceStatusLabel(status: InternetContextSourceStatus): string {
  switch (status) {
    case "found":
      return "encontrado";
    case "cached":
      return "cache";
    case "unavailable":
      return "indisponivel";
    case "error":
      return "erro";
    default:
      return "nao encontrado";
  }
}

export function countInternetContextKinds(result: InternetContextResult) {
  return result.glossary_candidates.reduce(
    (acc, candidate) => {
      if (candidate.kind === "character") acc.characters += 1;
      else if (candidate.kind === "place" || candidate.kind === "faction") acc.placesAndFactions += 1;
      else if (candidate.kind === "alias") acc.aliases += 1;
      else acc.loreTerms += 1;
      return acc;
    },
    { characters: 0, placesAndFactions: 0, loreTerms: 0, aliases: 0 },
  );
}

function unique(values: string[]) {
  return Array.from(new Set(values.filter(Boolean)));
}

function sourceRefs(result: InternetContextResult): ContextSourceRef[] {
  return result.source_results.map((source) => ({
    fonte: source.source,
    titulo: source.title || result.title,
    url: source.url || "",
    trecho: source.synopsis || source.error || sourceStatusLabel(source.status),
  }));
}

export function applyHighConfidenceInternetCandidates(
  context: ProjectContext,
  result: InternetContextResult,
  reviewedGlossary: Record<string, string> = {},
  threshold = 0.85,
): { contexto: ProjectContext; appliedCount: number } {
  const nextGlossary = { ...context.glossario, ...reviewedGlossary };
  const characters = [...context.personagens];
  const terms = [...context.termos];
  const factions = [...context.faccoes];
  const aliases = [...context.aliases];
  let appliedCount = 0;

  for (const candidate of result.glossary_candidates) {
    if (candidate.status === "rejected") continue;
    if (reviewedGlossary[candidate.source]) {
      nextGlossary[candidate.source] = reviewedGlossary[candidate.source];
      continue;
    }
    if (candidate.confidence < threshold && candidate.status !== "reviewed") continue;

    nextGlossary[candidate.source] = candidate.target;
    appliedCount += 1;
    if (candidate.kind === "character") characters.push(candidate.source);
    else if (candidate.kind === "faction") factions.push(candidate.source);
    else if (candidate.kind === "alias") aliases.push(candidate.source);
    else terms.push(candidate.source);
  }

  return {
    appliedCount,
    contexto: {
      ...context,
      sinopse: result.synopsis || context.sinopse,
      genero: unique([...context.genero, ...result.genres]),
      personagens: unique(characters),
      glossario: nextGlossary,
      aliases: unique(aliases),
      termos: unique(terms),
      faccoes: unique(factions),
      fontes_usadas: uniqueSourceRefs([...context.fontes_usadas, ...sourceRefs(result)]),
    },
  };
}

function uniqueSourceRefs(sources: ContextSourceRef[]) {
  const seen = new Set<string>();
  return sources.filter((source) => {
    const key = `${source.fonte}:${source.url}:${source.titulo}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

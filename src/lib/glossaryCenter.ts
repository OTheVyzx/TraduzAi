import type { InternetContextCandidate } from "./internetContext";
import type { GlossaryEntry } from "./tauri";

export type GlossaryTab = "reviewed" | "online" | "detected" | "rejected" | "conflicts";

export function glossaryEntryId(prefix: string, source: string) {
  const slug =
    source
      .trim()
      .toLocaleLowerCase("pt-BR")
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_|_$/g, "") || "termo";
  return `${prefix}_${slug}`;
}

export function candidateToGlossaryEntry(
  candidate: InternetContextCandidate,
  status: GlossaryEntry["status"] = "reviewed",
): GlossaryEntry {
  return {
    id: glossaryEntryId(candidate.kind || "term", candidate.source),
    source: candidate.source.trim(),
    target: candidate.target.trim() || candidate.source.trim(),
    type: candidate.kind || "generic_term",
    case_sensitive: false,
    protect: Boolean(candidate.protect),
    aliases: candidate.aliases ?? [],
    forbidden: candidate.forbidden ?? [],
    confidence: candidate.confidence,
    status,
    notes: candidate.notes ?? "",
    context_rule: "",
    sources: candidate.sources ?? [],
  };
}

export function manualTermToGlossaryEntry(source: string, target: string): GlossaryEntry {
  return {
    id: glossaryEntryId("term", source),
    source: source.trim(),
    target: target.trim(),
    type: "generic_term",
    case_sensitive: false,
    protect: false,
    aliases: [],
    forbidden: [],
    confidence: 1,
    status: "reviewed",
    notes: "",
    context_rule: "",
    sources: ["manual"],
  };
}

export function filterRejectedCandidates(
  candidates: InternetContextCandidate[],
  rejectedSources: string[],
  forceRefresh = false,
) {
  if (forceRefresh) return candidates;
  const rejected = new Set(rejectedSources.map((source) => source.toLocaleLowerCase("pt-BR")));
  return candidates.map((candidate) =>
    rejected.has(candidate.source.toLocaleLowerCase("pt-BR"))
      ? { ...candidate, status: "rejected" as const }
      : candidate,
  );
}

export function reviewedGlossaryForPipeline(entries: GlossaryEntry[]) {
  return Object.fromEntries(
    entries
      .filter((entry) => entry.status === "reviewed")
      .map((entry) => [entry.source, entry.target]),
  );
}

export function glossaryConflicts(
  glossary: Record<string, string>,
  candidates: InternetContextCandidate[],
) {
  return candidates.filter((candidate) => {
    const reviewed = glossary[candidate.source];
    return Boolean(reviewed && reviewed !== candidate.target);
  });
}

export function candidateNeedsReviewWarning(candidate: InternetContextCandidate) {
  return candidate.status === "candidate" || candidate.status === "auto";
}

export function forbiddenCriticalFlags(entry: Pick<GlossaryEntry, "id" | "forbidden">, translation: string) {
  const normalized = translation.toLocaleLowerCase("pt-BR");
  return entry.forbidden
    .filter((term) => term && normalized.includes(term.toLocaleLowerCase("pt-BR")))
    .map(() => `forbidden:${entry.id}`);
}

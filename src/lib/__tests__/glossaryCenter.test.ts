import { describe, expect, it } from "vitest";
import {
  candidateNeedsReviewWarning,
  candidateToGlossaryEntry,
  filterRejectedCandidates,
  forbiddenCriticalFlags,
  glossaryConflicts,
  reviewedGlossaryForPipeline,
} from "../glossaryCenter";
import type { InternetContextCandidate } from "../internetContext";

const candidate: InternetContextCandidate = {
  kind: "character",
  source: "Ghislain Perdium",
  target: "Ghislain Perdium",
  confidence: 0.92,
  sources: ["anilist"],
  status: "candidate",
  protect: true,
  aliases: ["Ghislain"],
  forbidden: ["Perdium Ghislain"],
  notes: "",
};

describe("glossaryCenter", () => {
  it("turns online candidates into reviewed glossary entries", () => {
    const entry = candidateToGlossaryEntry(candidate, "reviewed");

    expect(entry.status).toBe("reviewed");
    expect(entry.protect).toBe(true);
    expect(entry.sources).toEqual(["anilist"]);
  });

  it("keeps rejected candidates out of online suggestions without force refresh", () => {
    const filtered = filterRejectedCandidates([candidate], ["Ghislain Perdium"]);
    expect(filtered[0].status).toBe("rejected");

    const forced = filterRejectedCandidates([candidate], ["Ghislain Perdium"], true);
    expect(forced[0].status).toBe("candidate");
  });

  it("exports only reviewed terms to the pipeline glossary", () => {
    const reviewed = candidateToGlossaryEntry(candidate, "reviewed");
    const rejected = candidateToGlossaryEntry({ ...candidate, source: "Cavald" }, "rejected");

    expect(reviewedGlossaryForPipeline([reviewed, rejected])).toEqual({
      "Ghislain Perdium": "Ghislain Perdium",
    });
  });

  it("detects conflicts and forbidden critical flags", () => {
    expect(glossaryConflicts({ "Ghislain Perdium": "Ghislaine" }, [candidate])).toHaveLength(1);
    expect(candidateNeedsReviewWarning(candidate)).toBe(true);
    expect(forbiddenCriticalFlags(candidateToGlossaryEntry(candidate), "Perdium Ghislain")).toEqual([
      "forbidden:character_ghislain_perdium",
    ]);
  });
});

import { describe, expect, it } from "vitest";
import {
  emptyWorkContextSummary,
  riskLevel,
  setupWarningKind,
  shouldWarnWorkContext,
  summarizeWorkContext,
} from "../workContextProfile";

describe("workContextProfile", () => {
  it("marks empty context with no glossary as high risk", () => {
    const summary = summarizeWorkContext(
      { work_id: "solo-leveling", title: "Solo Leveling", context_quality: "empty" },
      0,
    );

    expect(summary.context_loaded).toBe(false);
    expect(summary.glossary_loaded).toBe(false);
    expect(summary.risk_level).toBe("high");
    expect(shouldWarnWorkContext(summary)).toBe(true);
  });

  it("reduces risk when context or glossary exists", () => {
    expect(riskLevel("partial", 0)).toBe("medium");
    expect(riskLevel("empty", 2)).toBe("medium");
    expect(riskLevel("reviewed", 2)).toBe("low");
  });

  it("preserves ignored warning state in project summary", () => {
    const summary = summarizeWorkContext(
      { work_id: "x", title: "X", context_quality: "partial" },
      0,
      true,
    );

    expect(summary.user_ignored_warning).toBe(true);
    expect(shouldWarnWorkContext(summary)).toBe(true);
  });

  it("separates missing work and empty glossary warnings", () => {
    expect(setupWarningKind(emptyWorkContextSummary(), "")).toBe("missing_work");

    const summary = summarizeWorkContext(
      { work_id: "fixture", title: "Fixture", context_quality: "empty" },
      0,
    );
    expect(setupWarningKind(summary, "Fixture")).toBe("empty_glossary");
  });

  it("keeps online context loading state in the summary", () => {
    const summary = summarizeWorkContext(
      {
        work_id: "fixture",
        title: "Fixture",
        context_quality: "partial",
        internet_context_loaded: true,
      },
      3,
    );

    expect(summary.internet_context_loaded).toBe(true);
    expect(summary.glossary_loaded).toBe(true);
    expect(setupWarningKind(summary, "Fixture")).toBeNull();
  });
});

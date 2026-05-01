import type { ProjectContext } from "./stores/appStore";
import type { WorkContextProfile, WorkContextSummary } from "./tauri";

export type SetupWorkContextWarning = "missing_work" | "empty_glossary";

export function glossaryEntriesCount(context: ProjectContext) {
  return Object.keys(context.glossario ?? {}).length;
}

export function riskLevel(contextQuality: string, glossaryCount: number): WorkContextSummary["risk_level"] {
  if (contextQuality === "reviewed" && glossaryCount > 0) return "low";
  if (contextQuality === "partial" || glossaryCount > 0) return "medium";
  return "high";
}

export function summarizeWorkContext(
  profile: Pick<WorkContextProfile, "work_id" | "title" | "context_quality"> & {
    internet_context_loaded?: boolean;
  },
  glossaryCount: number,
  userIgnoredWarning = false,
): WorkContextSummary {
  return {
    selected: true,
    work_id: profile.work_id,
    title: profile.title,
    context_loaded: profile.context_quality !== "empty",
    glossary_loaded: glossaryCount > 0,
    glossary_entries_count: glossaryCount,
    internet_context_loaded: profile.internet_context_loaded ?? false,
    risk_level: riskLevel(profile.context_quality, glossaryCount),
    user_ignored_warning: userIgnoredWarning,
  };
}

export function emptyWorkContextSummary(glossaryCount = 0, userIgnoredWarning = false): WorkContextSummary {
  return {
    selected: false,
    work_id: "",
    title: "",
    context_loaded: false,
    glossary_loaded: glossaryCount > 0,
    glossary_entries_count: glossaryCount,
    internet_context_loaded: false,
    risk_level: "high",
    user_ignored_warning: userIgnoredWarning,
  };
}

export function setupWarningKind(
  summary?: WorkContextSummary | null,
  requestedTitle = "",
): SetupWorkContextWarning | null {
  const hasTypedTitle = requestedTitle.trim().length > 0;
  const hasSelectedWork = Boolean(summary?.selected || summary?.title?.trim());
  if (!hasTypedTitle && !hasSelectedWork) return "missing_work";
  if ((hasTypedTitle || hasSelectedWork) && !summary?.glossary_loaded) return "empty_glossary";
  return null;
}

export function shouldWarnWorkContext(summary?: WorkContextSummary | null, requestedTitle = "") {
  return setupWarningKind(summary, requestedTitle) !== null;
}

export function contextQualityLabel(value?: string) {
  if (value === "reviewed") return "revisado";
  if (value === "partial") return "parcial";
  return "vazio";
}

export function riskLabel(value?: string) {
  if (value === "low") return "baixo";
  if (value === "medium") return "medio";
  return "alto";
}

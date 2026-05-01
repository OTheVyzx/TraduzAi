import type { ProjectContext } from "./stores/appStore";
import type { WorkContextProfile, WorkContextSummary } from "./tauri";

export function glossaryEntriesCount(context: ProjectContext) {
  return Object.keys(context.glossario ?? {}).length;
}

export function riskLevel(contextQuality: string, glossaryCount: number): WorkContextSummary["risk_level"] {
  if (contextQuality === "reviewed" && glossaryCount > 0) return "low";
  if (contextQuality === "partial" || glossaryCount > 0) return "medium";
  return "high";
}

export function summarizeWorkContext(
  profile: Pick<WorkContextProfile, "work_id" | "title" | "context_quality">,
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
    risk_level: riskLevel(profile.context_quality, glossaryCount),
    user_ignored_warning: userIgnoredWarning,
  };
}

export function shouldWarnWorkContext(summary?: WorkContextSummary | null) {
  if (!summary?.selected) return false;
  return !summary.context_loaded || !summary.glossary_loaded;
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

export type CompletionStatus = "approved" | "blocked" | "overridden" | "error";
export type ExportGateStatus = "PASS" | "BLOCK" | "OVERRIDDEN";
export type OutputReviewState = "approved" | "blocked_preview" | "overridden";
export type ReviewBadgeTone = "success" | "warning" | "error" | "neutral";
export type AppProjectStatus = "idle" | "setup" | "processing" | "done" | "done_blocked" | "needs_review" | "error";

export interface PipelineExportGateSummary {
  status: ExportGateStatus;
  critical_issue_count: number;
  critical_flag_count: number;
  review_issue_count: number;
  needs_review: boolean;
}

export interface PipelineCompleteEvent {
  success: boolean;
  job_id?: string;
  output_path: string;
  error?: string;
  completion_status?: CompletionStatus;
  export_gate?: PipelineExportGateSummary;
  blocking_flags?: string[];
  review_flags?: string[];
}

export interface PipelineBlockedBannerModel {
  criticalCount: number;
  reviewCount: number;
  blockingFlags: string[];
  reviewFlags: string[];
}

type ReviewSource = {
  status?: string | null;
  output_review_state?: string | null;
  completion_status?: string | null;
  export_gate?: Partial<PipelineExportGateSummary> | null;
  blocking_flags?: string[] | null;
  review_flags?: string[] | null;
  critical_issue_count?: number | null;
  review_issue_count?: number | null;
  qa?: unknown;
  needs_review?: boolean | null;
};

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : {};
}

function asStringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function asNumber(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function gateFromSource(source: ReviewSource | null | undefined): Record<string, unknown> {
  const directGate = asRecord(source?.export_gate);
  if (Object.keys(directGate).length > 0) return directGate;
  const qa = asRecord(source?.qa);
  return asRecord(qa.export_gate);
}

function qaSummaryFromSource(source: ReviewSource | null | undefined): Record<string, unknown> {
  return asRecord(asRecord(source?.qa).summary);
}

export function outputReviewStateForCompletion(status?: CompletionStatus | null): OutputReviewState {
  if (status === "blocked") return "blocked_preview";
  if (status === "overridden") return "overridden";
  return "approved";
}

export function deriveProjectStatusFromReviewState(source: ReviewSource | null | undefined): AppProjectStatus {
  const gate = gateFromSource(source);
  const summary = qaSummaryFromSource(source);
  const gateStatus = String(gate.status ?? "").toUpperCase();
  const completionStatus = String(source?.completion_status ?? "").toLowerCase();
  const reviewState = String(source?.output_review_state ?? "").toLowerCase();
  const reviewIssueCount = asNumber(source?.review_issue_count) || asNumber(gate.review_issue_count) || asNumber(summary.review_issue_count);

  if (completionStatus === "error") return "error";
  if (completionStatus === "blocked" || reviewState === "blocked_preview" || gateStatus === "BLOCK") {
    return "done_blocked";
  }
  if (Boolean(source?.needs_review) || Boolean(gate.needs_review) || reviewIssueCount > 0) {
    return "needs_review";
  }
  return "done";
}

export function deriveCompletionProjectStatus(
  event: PipelineCompleteEvent,
  projectJson?: ReviewSource | null,
): AppProjectStatus {
  if (!event.success || event.completion_status === "error") return "error";
  return deriveProjectStatusFromReviewState({
    ...projectJson,
    completion_status: event.completion_status,
    export_gate: event.export_gate ?? projectJson?.export_gate,
    blocking_flags: event.blocking_flags ?? projectJson?.blocking_flags,
    review_flags: event.review_flags ?? projectJson?.review_flags,
  });
}

export function isBlockedProjectReviewState(source: ReviewSource | null | undefined): boolean {
  return deriveProjectStatusFromReviewState(source) === "done_blocked";
}

function issueFlagsFromGate(gate: Record<string, unknown>, severity: "critical" | "review") {
  const issues = Array.isArray(gate.issues) ? gate.issues : [];
  const flags: string[] = [];
  for (const issue of issues) {
    const item = asRecord(issue);
    const issueSeverity = String(item.severity ?? "").toLowerCase();
    const issueType = String(item.type ?? "").toLowerCase();
    const matches =
      severity === "critical"
        ? issueSeverity === "critical" || issueType.startsWith("p0")
        : issueSeverity === "high" || issueType === "needs_review";
    if (!matches) continue;
    flags.push(...asStringArray(item.flags));
    if (typeof item.flag === "string") flags.push(item.flag);
    if (typeof item.flag_id === "string") flags.push(item.flag_id);
    if (flags.length === 0 && typeof item.type === "string") flags.push(item.type);
  }
  return Array.from(new Set(flags)).slice(0, 8);
}

export function buildPipelineBlockedBannerModel(project: ReviewSource | null): PipelineBlockedBannerModel | null {
  if (!isBlockedProjectReviewState(project)) return null;

  const gate = gateFromSource(project);
  const summary = qaSummaryFromSource(project);
  const blockingFlags =
    asStringArray(project?.blocking_flags).length > 0
      ? asStringArray(project?.blocking_flags)
      : issueFlagsFromGate(gate, "critical");
  const reviewFlags =
    asStringArray(project?.review_flags).length > 0
      ? asStringArray(project?.review_flags)
      : issueFlagsFromGate(gate, "review");
  const criticalCount =
    asNumber(project?.critical_issue_count) ||
    asNumber(gate.critical_issue_count) ||
    asNumber(summary.critical_issue_count) ||
    blockingFlags.length;
  const reviewCount =
    asNumber(project?.review_issue_count) ||
    asNumber(gate.review_issue_count) ||
    asNumber(summary.review_issue_count) ||
    reviewFlags.length;

  return {
    criticalCount,
    reviewCount,
    blockingFlags,
    reviewFlags,
  };
}

export function buildRecentProjectReviewBadge(
  project: {
    status: string;
    critical_issue_count?: number;
    review_issue_count?: number;
  },
): { label: string; tone: ReviewBadgeTone } {
  if (project.status === "done_blocked") {
    const count = Math.max(1, asNumber(project.critical_issue_count));
    return {
      label: count === 1 ? "Bloqueado (1 issue critica)" : `Bloqueado (${count} issues criticas)`,
      tone: "error",
    };
  }
  if (project.status === "needs_review") {
    const count = asNumber(project.review_issue_count);
    return {
      label: count > 0 ? `Revisar (${count} issues)` : "Revisar",
      tone: "warning",
    };
  }
  if (project.status === "done") {
    return { label: "Aprovado", tone: "success" };
  }
  return { label: "Em andamento", tone: "neutral" };
}

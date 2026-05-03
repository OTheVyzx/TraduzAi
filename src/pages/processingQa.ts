export interface ProcessingQaTopReason {
  reason: string;
  label: string;
  count: number;
}

export interface ProcessingQaSummary {
  totalIssues: number;
  totalDecisions: number;
  criticalCount: number;
  warningCount: number;
  flaggedPages: number[];
  topReasons: ProcessingQaTopReason[];
}

function normalizePageNumber(value: unknown): number | null {
  const page = Number(value);
  if (!Number.isFinite(page) || page <= 0) return null;
  return page;
}

function issueSeverity(issue: any): string {
  return String(issue?.severity ?? issue?.level ?? "").toLowerCase();
}

function issueReason(issue: any): string {
  return String(
    issue?.reason ??
      issue?.type ??
      issue?.flag_id ??
      issue?.flagId ??
      issue?.label ??
      "outro",
  );
}

function issueReasonLabel(reason: string): string {
  const labels: Record<string, string> = {
    text_clipped: "Texto cortado",
    text_overflow: "Texto estourando",
    ocr_low_confidence: "OCR baixo",
    inpaint_artifact: "Artefato de inpaint",
    mask_issue: "Problema de máscara",
    outro: "Outro",
  };

  return labels[reason.toLowerCase()] ?? reason.replace(/[_-]+/g, " ");
}

export function summarizeProcessingQaReport(raw: any): ProcessingQaSummary {
  const issues =
    Array.isArray(raw?.issues)
      ? raw.issues
      : Array.isArray(raw?.flags)
        ? raw.flags
        : Array.isArray(raw?.qa_flags)
          ? raw.qa_flags
          : [];

  const decisions =
    Array.isArray(raw?.decisions)
      ? raw.decisions
      : Array.isArray(raw?.actions)
        ? raw.actions
        : Array.isArray(raw?.qa_actions)
          ? raw.qa_actions
          : [];

  const pageSet = new Set<number>();

  for (const issue of issues) {
    const page = normalizePageNumber(
      issue?.page ??
        issue?.page_number ??
        issue?.pageNumber ??
        issue?.pagina ??
        (typeof issue?.page_index === "number" ? issue.page_index + 1 : null),
    );

    if (page !== null) {
      pageSet.add(page);
    }
  }

  const flaggedPages: number[] = Array.from(pageSet).sort((a: number, b: number) => a - b);

  const criticalCount = issues.filter((issue: any) =>
    ["critical", "high", "error"].includes(issueSeverity(issue)),
  ).length;

  const warningCount = issues.filter((issue: any) =>
    ["warning", "medium", "low"].includes(issueSeverity(issue)),
  ).length;

  const reasonCounts = new Map<string, number>();

  for (const issue of issues) {
    const reason = issueReason(issue);
    reasonCounts.set(reason, (reasonCounts.get(reason) ?? 0) + 1);
  }

  const topReasons = Array.from(reasonCounts.entries())
    .map(([reason, count]) => ({
      reason,
      label: issueReasonLabel(reason),
      count,
    }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 6);

  return {
    totalIssues: issues.length,
    totalDecisions: decisions.length,
    criticalCount,
    warningCount,
    flaggedPages,
    topReasons,
  };
}

export function formatFlaggedPages(pages: number[] | undefined | null): string {
  if (!pages || pages.length === 0) return "nenhuma";
  return pages.join(", ");
}

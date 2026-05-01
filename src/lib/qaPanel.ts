import type { Project, QaAction, TextEntry } from "./stores/appStore";

export type QaSeverity = "critical" | "high" | "medium" | "low";

export interface QaIssue {
  id: string;
  flagId: string;
  pageIndex: number;
  pageNumber: number;
  regionId: string;
  label: string;
  severity: QaSeverity;
  sourceText: string;
  translatedText: string;
}

export interface QaReviewSummary {
  totalPages: number;
  approvedPages: number;
  warningPages: number;
  blockedPages: number;
  criticalCount: number;
  warningCount: number;
  groups: Record<string, number>;
}

const QA_LABELS: Record<string, string> = {
  critical_error: "Erros criticos",
  glossary_violation: "Glossario violado",
  forbidden_term: "Glossario violado",
  missing_protected_term: "Glossario violado",
  visual_text_leak: "Ingles restante",
  page_not_processed: "Ingles restante",
  ocr_gibberish: "OCR suspeito",
  ocr_suspect: "OCR suspeito",
  typesetting_overflow: "Texto grande demais",
  text_too_large: "Texto grande demais",
  inpaint_suspicious: "Inpaint suspeito",
  invalid_mask: "Mascara ausente",
  missing_mask: "Mascara ausente",
};

const QA_SEVERITY: Record<string, QaSeverity> = {
  critical_error: "critical",
  visual_text_leak: "critical",
  page_not_processed: "critical",
  glossary_violation: "high",
  forbidden_term: "high",
  missing_protected_term: "high",
  invalid_mask: "high",
  missing_mask: "high",
  ocr_gibberish: "medium",
  ocr_suspect: "medium",
  typesetting_overflow: "medium",
  text_too_large: "medium",
  inpaint_suspicious: "medium",
};

export function qaIssueLabel(flagId: string): string {
  return QA_LABELS[flagId] ?? flagId.replace(/_/g, " ");
}

export function qaIssueSeverity(flagId: string): QaSeverity {
  return QA_SEVERITY[flagId] ?? "low";
}

export function qaIssueGroup(flagId: string): string {
  if (["critical_error"].includes(flagId)) return "Criticos";
  if (["glossary_violation", "forbidden_term", "missing_protected_term"].includes(flagId)) return "Glossario";
  if (["ocr_gibberish", "ocr_suspect"].includes(flagId)) return "OCR";
  if (["context_missing", "context_low_confidence"].includes(flagId)) return "Contexto";
  if (["inpaint_suspicious"].includes(flagId)) return "Inpaint";
  if (["typesetting_overflow", "text_too_large"].includes(flagId)) return "Typesetting";
  if (["invalid_mask", "missing_mask"].includes(flagId)) return "Mascaras";
  if (["visual_text_leak", "page_not_processed"].includes(flagId)) return "Ingles restante";
  return "Outros";
}

function flagIssueId(pageIndex: number, regionId: string, flagId: string): string {
  return `${pageIndex}:${regionId}:${flagId}`;
}

function isIgnored(layer: TextEntry, flagId: string): boolean {
  return (layer.qa_actions ?? []).some((action) => action.flag_id === flagId && action.status === "ignored");
}

export function collectQaIssues(project: Project | null): QaIssue[] {
  if (!project) return [];

  return project.paginas.flatMap((page, pageIndex) =>
    (page.text_layers ?? []).flatMap((layer) =>
      (layer.qa_flags ?? [])
        .filter((flagId) => !isIgnored(layer, flagId))
        .map((flagId) => ({
          id: flagIssueId(pageIndex, layer.id, flagId),
          flagId,
          pageIndex,
          pageNumber: page.numero,
          regionId: layer.id,
          label: qaIssueLabel(flagId),
          severity: qaIssueSeverity(flagId),
          sourceText: layer.original,
          translatedText: layer.traduzido || layer.translated || "",
        })),
    ),
  );
}

export function collectIgnoredQaActions(project: Project | null): QaAction[] {
  if (!project) return [];
  return project.paginas.flatMap((page) =>
    (page.text_layers ?? []).flatMap((layer) => (layer.qa_actions ?? []).filter((action) => action.status === "ignored")),
  );
}

export function buildQaReviewSummary(project: Project | null): QaReviewSummary {
  const issues = collectQaIssues(project);
  const totalPages = project?.paginas.length ?? 0;
  const blockedPageIndexes = new Set(
    issues
      .filter((issue) => issue.severity === "critical" || issue.severity === "high")
      .map((issue) => issue.pageIndex),
  );
  const warningPageIndexes = new Set(
    issues
      .filter((issue) => issue.severity === "medium" || issue.severity === "low")
      .map((issue) => issue.pageIndex)
      .filter((pageIndex) => !blockedPageIndexes.has(pageIndex)),
  );
  const groups = issues.reduce<Record<string, number>>((acc, issue) => {
    const group = qaIssueGroup(issue.flagId);
    acc[group] = (acc[group] ?? 0) + 1;
    return acc;
  }, {});
  const blockedPages = blockedPageIndexes.size;
  const warningPages = warningPageIndexes.size;

  return {
    totalPages,
    approvedPages: Math.max(0, totalPages - blockedPages - warningPages),
    warningPages,
    blockedPages,
    criticalCount: issues.filter((issue) => issue.severity === "critical" || issue.severity === "high").length,
    warningCount: issues.filter((issue) => issue.severity === "medium" || issue.severity === "low").length,
    groups,
  };
}

export function canExportClean(summary: QaReviewSummary) {
  return summary.criticalCount === 0 && summary.blockedPages === 0;
}

export function ignoreQaIssue(project: Project, issueId: string, reason: string, ignoredAt = new Date().toISOString()): Project {
  const trimmedReason = reason.trim();
  if (!trimmedReason) {
    throw new Error("Informe o motivo para ignorar esta flag.");
  }

  const [pageIndexRaw, regionId, flagId] = issueId.split(":");
  const pageIndex = Number(pageIndexRaw);
  if (!Number.isInteger(pageIndex) || !regionId || !flagId) {
    throw new Error("Flag de QA invalida.");
  }

  const action: QaAction = {
    flag_id: flagId,
    status: "ignored",
    ignored_reason: trimmedReason,
    ignored_at: ignoredAt,
  };

  const paginas = project.paginas.map((page, index) => {
    if (index !== pageIndex) return page;

    const updateLayer = (layer: TextEntry) => {
      if (layer.id !== regionId) return layer;
      const remainingActions = (layer.qa_actions ?? []).filter(
        (current) => !(current.flag_id === flagId && current.status === "ignored"),
      );
      return { ...layer, qa_actions: [...remainingActions, action] };
    };

    return {
      ...page,
      text_layers: (page.text_layers ?? []).map(updateLayer),
      textos: (page.textos ?? []).map(updateLayer),
    };
  });

  return { ...project, paginas };
}

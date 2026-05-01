import type { QaReviewSummary } from "./qaPanel";

export type ExportMode = "clean" | "with_warnings" | "debug" | "review_package";

export const EXPORT_MODE_OPTIONS: Array<{ id: ExportMode; label: string; description: string }> = [
  { id: "clean", label: "Clean", description: "Bloqueia criticos e avisos altos." },
  { id: "with_warnings", label: "With warnings", description: "Permite avisos, mas bloqueia criticos." },
  { id: "debug", label: "Debug", description: "Nao publicar. Inclui dados para investigacao." },
  { id: "review_package", label: "Review package", description: "Pacote com QA, issues, glossario e memoria." },
];

export function exportModeForBackend(mode: ExportMode): "clean" | "with_warnings" | "debug" {
  if (mode === "review_package") return "with_warnings";
  return mode;
}

export function exportBlockReason(mode: ExportMode, summary: QaReviewSummary): string | null {
  if (mode === "debug") return null;
  if (mode === "clean" && summary.criticalCount > 0) {
    return "Export limpo bloqueado: revise os criticos ou use Exportar debug.";
  }
  if (summary.blockedPages > 0) {
    return "Export bloqueado: ha paginas bloqueadas. Use Exportar debug para auditoria.";
  }
  return null;
}

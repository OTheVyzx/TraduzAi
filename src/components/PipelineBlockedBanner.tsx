import { AlertTriangle, Eye } from "lucide-react";
import type { PipelineBlockedBannerModel } from "../lib/pipelineCompletion";

export function PipelineBlockedBanner({
  model,
  onOpenDetails,
}: {
  model: PipelineBlockedBannerModel | null;
  onOpenDetails?: () => void;
}) {
  if (!model) return null;

  const flags = [...model.blockingFlags, ...model.reviewFlags].slice(0, 8);

  return (
    <section
      data-testid="pipeline-blocked-banner"
      className="border-b border-status-error/25 bg-status-error/10 px-6 py-3 text-status-error"
    >
      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div className="flex min-w-0 items-start gap-3">
          <AlertTriangle size={18} className="mt-0.5 shrink-0" />
          <div className="min-w-0">
            <p className="text-sm font-semibold">Preview bloqueado pelo QA visual</p>
            <p className="mt-0.5 text-xs text-status-error/80">
              {model.criticalCount} issue(s) critica(s)
              {model.reviewCount > 0 ? ` e ${model.reviewCount} item(ns) para revisar` : ""}.
            </p>
            {flags.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-1.5">
                {flags.map((flag) => (
                  <span
                    key={flag}
                    className="rounded-md border border-status-error/25 bg-bg-primary/70 px-2 py-0.5 text-[11px] text-status-error"
                  >
                    {flag}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
        {onOpenDetails && (
          <button
            type="button"
            onClick={onOpenDetails}
            className="inline-flex shrink-0 items-center justify-center gap-2 rounded-lg border border-status-error/30 bg-bg-secondary px-3 py-2 text-xs font-medium text-status-error transition-smooth hover:bg-bg-tertiary"
          >
            <Eye size={14} />
            Ver detalhes
          </button>
        )}
      </div>
    </section>
  );
}

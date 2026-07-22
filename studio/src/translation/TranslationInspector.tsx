import { CheckCircle2, Languages } from "lucide-react";
import type { StudioTextLayer, TranslationStatus } from "../project/studioProject";
import { resolveTranslationStatus } from "./translationQueue";

const TEXT_TYPES = [
  ["fala", "Fala"],
  ["narracao", "Narração"],
  ["pensamento", "Pensamento"],
  ["sfx", "SFX"],
] as const;

const STATUS_OPTIONS: Array<[TranslationStatus, string]> = [
  ["pending", "Pendente"],
  ["translated", "Traduzido"],
  ["review", "Revisão"],
  ["approved", "Aprovado"],
];

export function TranslationInspector({
  layer,
  onChange,
  onConfirmNext,
  isSaving = false,
}: {
  layer: StudioTextLayer | null;
  onChange: (patch: Partial<StudioTextLayer>) => void;
  onConfirmNext: () => void | Promise<void>;
  isSaving?: boolean;
}) {
  if (!layer) {
    return (
      <section className="flex min-h-52 flex-col items-center justify-center px-5 text-center">
        <Languages size={24} className="mb-3 text-text-muted/50" />
        <p className="text-xs font-medium text-text-secondary">Selecione um bloco de texto</p>
        <p className="mt-1 max-w-[220px] text-[10px] leading-4 text-text-muted">
          Use a fila à esquerda ou clique em um bloco no canvas.
        </p>
      </section>
    );
  }

  const status = resolveTranslationStatus(layer);
  return (
    <section className="space-y-3 px-3 py-3" aria-label="Inspetor de tradução">
      <label className="block">
        <span className="mb-1 block text-[10px] font-semibold uppercase tracking-[0.12em] text-text-muted">Original</span>
        <textarea
          readOnly
          value={layer.original}
          rows={3}
          className="w-full resize-none rounded-lg border border-border bg-black/20 px-2.5 py-2 text-[11px] leading-5 text-text-secondary outline-none"
        />
      </label>

      <label className="block">
        <span className="mb-1 block text-[10px] font-semibold uppercase tracking-[0.12em] text-accent-cyan">Tradução</span>
        <textarea
          value={layer.translated}
          rows={4}
          autoFocus
          onChange={(event) => onChange({ translated: event.target.value, traduzido: event.target.value })}
          className="w-full resize-y rounded-lg border border-accent-cyan/25 bg-bg-primary px-2.5 py-2 text-[12px] leading-5 text-text-primary outline-none transition focus:border-accent-cyan/60 focus:shadow-[0_0_0_2px_rgba(34,211,238,0.08)]"
        />
      </label>

      <div className="grid grid-cols-2 gap-2">
        <label>
          <span className="mb-1 block text-[10px] font-medium text-text-muted">Tipo</span>
          <select
            value={layer.tipo ?? "fala"}
            onChange={(event) => onChange({ tipo: event.target.value })}
            className="w-full rounded-md border border-border bg-bg-primary px-2 py-1.5 text-[10px] text-text-primary outline-none focus:border-accent-cyan/50"
          >
            {TEXT_TYPES.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select>
        </label>
        <label>
          <span className="mb-1 block text-[10px] font-medium text-text-muted">Status</span>
          <select
            value={status}
            onChange={(event) => onChange({ translation_status: event.target.value as TranslationStatus })}
            className="w-full rounded-md border border-border bg-bg-primary px-2 py-1.5 text-[10px] text-text-primary outline-none focus:border-accent-cyan/50"
          >
            {STATUS_OPTIONS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select>
        </label>
      </div>

      <label className="block">
        <span className="mb-1 block text-[10px] font-medium text-text-muted">Notas</span>
        <textarea
          value={layer.translation_notes ?? ""}
          rows={2}
          placeholder="Contexto, dúvida ou decisão editorial"
          onChange={(event) => onChange({ translation_notes: event.target.value })}
          className="w-full resize-y rounded-md border border-border bg-bg-primary px-2.5 py-2 text-[10px] leading-4 text-text-primary outline-none focus:border-accent-cyan/50"
        />
      </label>

      <button
        type="button"
        disabled={isSaving}
        onClick={() => void onConfirmNext()}
        className="flex w-full items-center justify-center gap-2 rounded-lg border border-status-success/30 bg-status-success/10 px-3 py-2 text-[11px] font-semibold text-status-success transition hover:bg-status-success/15 disabled:opacity-40"
        title="Confirmar e ir ao próximo pendente (Ctrl+Enter)"
      >
        <CheckCircle2 size={13} />
        Confirmar e próximo
        <kbd className="ml-auto rounded border border-white/10 bg-black/20 px-1 py-0.5 font-mono text-[8px] text-text-muted">Ctrl Enter</kbd>
      </button>
    </section>
  );
}

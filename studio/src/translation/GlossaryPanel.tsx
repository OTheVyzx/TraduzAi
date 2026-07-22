import { useState, type FormEvent } from "react";
import { BookOpen, Plus, Trash2 } from "lucide-react";

export function GlossaryPanel({
  glossary,
  onChange,
}: {
  glossary: Record<string, string>;
  onChange: (glossary: Record<string, string>) => void | Promise<void>;
}) {
  const [term, setTerm] = useState("");
  const [meaning, setMeaning] = useState("");
  const entries = Object.entries(glossary).sort(([left], [right]) => left.localeCompare(right, "pt-BR"));

  const addEntry = (event: FormEvent) => {
    event.preventDefault();
    const normalizedTerm = term.trim();
    const normalizedMeaning = meaning.trim();
    if (!normalizedTerm || !normalizedMeaning) return;
    void onChange({ ...glossary, [normalizedTerm]: normalizedMeaning });
    setTerm("");
    setMeaning("");
  };

  const removeEntry = (entryTerm: string) => {
    const next = { ...glossary };
    delete next[entryTerm];
    void onChange(next);
  };

  return (
    <section className="border-t border-border px-3 py-3" aria-labelledby="studio-glossary-title">
      <div className="mb-2 flex items-center gap-2">
        <BookOpen size={13} className="text-accent-cyan" />
        <h3 id="studio-glossary-title" className="text-[11px] font-semibold uppercase tracking-[0.12em] text-text-secondary">
          Glossário do projeto
        </h3>
        <span className="ml-auto rounded-full bg-white/[0.04] px-1.5 py-0.5 font-mono text-[9px] text-text-muted">
          {entries.length}
        </span>
      </div>

      <div className="max-h-32 space-y-1 overflow-y-auto pr-1">
        {entries.length === 0 ? (
          <p className="rounded-md border border-dashed border-border px-2 py-3 text-center text-[10px] text-text-muted">
            Nenhum termo cadastrado.
          </p>
        ) : entries.map(([entryTerm, entryMeaning]) => (
          <div key={entryTerm} className="group flex items-start gap-2 rounded-md border border-transparent bg-white/[0.025] px-2 py-1.5 hover:border-white/[0.06]">
            <div className="min-w-0 flex-1">
              <p className="truncate text-[10px] font-semibold text-text-primary">{entryTerm}</p>
              <p className="break-words text-[10px] leading-4 text-text-muted">{entryMeaning}</p>
            </div>
            <button
              type="button"
              onClick={() => removeEntry(entryTerm)}
              className="rounded p-1 text-text-muted opacity-0 transition hover:bg-status-error/10 hover:text-status-error group-hover:opacity-100"
              aria-label={`Remover ${entryTerm} do glossário`}
            >
              <Trash2 size={11} />
            </button>
          </div>
        ))}
      </div>

      <form className="mt-2 grid grid-cols-[0.8fr_1fr_auto] gap-1" onSubmit={addEntry}>
        <input
          value={term}
          onChange={(event) => setTerm(event.target.value)}
          placeholder="Termo"
          aria-label="Termo do glossário"
          className="min-w-0 rounded-md border border-border bg-bg-primary px-2 py-1.5 text-[10px] text-text-primary outline-none focus:border-accent-cyan/50"
        />
        <input
          value={meaning}
          onChange={(event) => setMeaning(event.target.value)}
          placeholder="Tradução/contexto"
          aria-label="Significado do termo"
          className="min-w-0 rounded-md border border-border bg-bg-primary px-2 py-1.5 text-[10px] text-text-primary outline-none focus:border-accent-cyan/50"
        />
        <button
          type="submit"
          disabled={!term.trim() || !meaning.trim()}
          className="rounded-md border border-accent-cyan/25 bg-accent-cyan/10 px-2 text-accent-cyan transition hover:bg-accent-cyan/15 disabled:opacity-30"
          aria-label="Adicionar termo ao glossário"
        >
          <Plus size={12} />
        </button>
      </form>
    </section>
  );
}

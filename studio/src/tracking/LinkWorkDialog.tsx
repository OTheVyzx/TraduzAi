import { useEffect, useState, type FormEvent } from "react";
import { CheckCircle2, Search, X } from "lucide-react";
import type { LibraryWork } from "../library/libraryModel";
import { searchTrackingWorks, type WorkTrackingProvider, type WorkTrackingSnapshot } from "./workTracking";

export function LinkWorkDialog({
  open,
  work,
  onClose,
  onConfirm,
}: {
  open: boolean;
  work: LibraryWork | null;
  onClose: () => void;
  onConfirm: (snapshot: WorkTrackingSnapshot) => void | Promise<void>;
}) {
  const [provider, setProvider] = useState<WorkTrackingProvider>("anilist");
  const [query, setQuery] = useState(work?.title ?? "");
  const [results, setResults] = useState<WorkTrackingSnapshot[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [identityConfirmed, setIdentityConfirmed] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    setQuery(work?.title ?? "");
    setResults([]);
    setSelectedId(null);
    setIdentityConfirmed(false);
    setError(null);
  }, [open, work?.id]);

  if (!open || !work) return null;
  const selected = results.find((result) => `${result.provider}:${result.providerId}` === selectedId) ?? null;

  const search = async (event: FormEvent) => {
    event.preventDefault();
    setLoading(true);
    setError(null);
    setResults([]);
    setSelectedId(null);
    setIdentityConfirmed(false);
    try {
      setResults(await searchTrackingWorks(query, provider));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/75 p-6" role="presentation">
      <section className="w-full max-w-3xl rounded-lg border border-zinc-700 bg-[#1b1c22] p-5 text-zinc-100" role="dialog" aria-modal="true" aria-labelledby="link-work-title">
        <header className="flex items-start justify-between">
          <div><small className="text-zinc-500">Acompanhamento opcional</small><h2 id="link-work-title" className="text-xl font-semibold">Vincular “{work.title}”</h2></div>
          <button type="button" aria-label="Fechar" className="rounded p-2 hover:bg-zinc-800" onClick={onClose}><X size={18} /></button>
        </header>

        <form className="mt-5 flex gap-2" onSubmit={(event) => void search(event)}>
          <select value={provider} onChange={(event) => setProvider(event.currentTarget.value as WorkTrackingProvider)} className="rounded border border-zinc-700 bg-zinc-900 px-3">
            <option value="anilist">AniList</option>
            <option value="mangadex">MangaDex</option>
          </select>
          <input aria-label="Título ou ID" value={query} onChange={(event) => setQuery(event.currentTarget.value)} placeholder="Título ou ID da obra" className="min-w-0 flex-1 rounded border border-zinc-700 bg-zinc-900 px-3 py-2" />
          <button type="submit" disabled={loading} className="inline-flex items-center gap-2 rounded bg-amber-500 px-4 font-semibold text-zinc-950"><Search size={15} />{loading ? "Buscando…" : "Buscar"}</button>
        </form>
        {error && <p className="mt-3 text-sm text-rose-300">{error}</p>}

        <div className="mt-4 grid max-h-80 gap-2 overflow-y-auto">
          {results.map((result) => {
            const id = `${result.provider}:${result.providerId}`;
            return (
              <button key={id} type="button" onClick={() => { setSelectedId(id); setIdentityConfirmed(false); }} className={`grid grid-cols-[48px_1fr_auto] items-center gap-3 rounded border p-3 text-left ${selectedId === id ? "border-amber-500 bg-amber-500/10" : "border-zinc-800 bg-zinc-900"}`}>
                <span className="h-16 overflow-hidden rounded bg-zinc-800">{result.coverUrl && <img className="h-full w-full object-cover" src={result.coverUrl} alt="" />}</span>
                <span><strong className="block">{result.title}</strong><small className="text-zinc-500">{result.provider === "anilist" ? "AniList" : "MangaDex"} · {result.status}</small></span>
                {selectedId === id && <CheckCircle2 className="text-amber-400" size={20} />}
              </button>
            );
          })}
        </div>

        {selected && (
          <label className="mt-5 flex items-start gap-3 rounded border border-zinc-700 bg-zinc-900 p-3 text-sm">
            <input type="checkbox" checked={identityConfirmed} onChange={(event) => setIdentityConfirmed(event.currentTarget.checked)} />
            <span>Confirmo que <strong>{selected.title}</strong> e <strong>{work.title}</strong> são a mesma obra.</span>
          </label>
        )}

        <footer className="mt-5 flex justify-end gap-2">
          <button type="button" className="rounded border border-zinc-700 px-4 py-2" onClick={onClose}>Cancelar</button>
          <button type="button" disabled={!selected || !identityConfirmed} className="rounded bg-amber-500 px-4 py-2 font-semibold text-zinc-950 disabled:opacity-40" onClick={() => selected && void Promise.resolve(onConfirm(selected)).then(onClose)}>Confirmar vínculo</button>
        </footer>
      </section>
    </div>
  );
}

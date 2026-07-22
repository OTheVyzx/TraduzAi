import { useMemo, useState } from "react";
import { AlertTriangle, ExternalLink, RefreshCw, X } from "lucide-react";
import type { LibraryWork, PublicationStatus } from "../library/libraryModel";
import type { AddLibraryWorkInput } from "../store/libraryStore";
import {
  createTrackingCache,
  hasRemoteChapterUpdate,
  isTrackingCacheStale,
  preserveTrackingCacheOnError,
  resolveTrackingStatus,
  syncTrackingWork,
  UPDATES_REFRESH_TTL_MS,
  type WorkTrackingSnapshot,
} from "./workTracking";

const STATUS_LABELS: Record<PublicationStatus, string> = {
  releasing: "Em publicação",
  hiatus: "Em hiato",
  completed: "Completa",
  cancelled: "Cancelada",
  not_yet_released: "Não iniciada",
  unknown: "Status desconhecido",
};

function formatTrackingTime(value: string): string {
  const timestamp = Date.parse(value);
  if (!Number.isFinite(timestamp)) return value;
  return new Intl.DateTimeFormat("pt-BR", {
    dateStyle: "short",
    timeStyle: "short",
  }).format(new Date(timestamp));
}

function workInput(work: LibraryWork, snapshots: WorkTrackingSnapshot[], lastError: string | null): AddLibraryWorkInput {
  const updatedProviders = new Set(snapshots.map((snapshot) => snapshot.provider));
  const mergedSnapshots = [
    ...(work.external.tracking?.snapshots ?? []).filter((snapshot) => !updatedProviders.has(snapshot.provider)),
    ...snapshots,
  ];
  const resolved = resolveTrackingStatus(
    work.publicationStatus,
    work.external.manualStatusOverride,
    mergedSnapshots,
  );
  return {
    id: work.id,
    title: work.title,
    aliases: work.aliases,
    coverPath: work.coverPath,
    publicationStatus: resolved.status,
    external: {
      ...work.external,
      tracking: createTrackingCache(mergedSnapshots, new Date(), UPDATES_REFRESH_TTL_MS, lastError),
    },
  };
}

export function UpdatesView({
  open,
  works,
  trackingLanguage,
  now = new Date(),
  onClose,
  onOpenWork,
  onPersistWork,
  onSetTrackingLanguage,
}: {
  open: boolean;
  works: LibraryWork[];
  trackingLanguage: string;
  now?: Date;
  onClose: () => void;
  onOpenWork: (workId: string) => void;
  onPersistWork: (work: AddLibraryWorkInput) => void | Promise<void>;
  onSetTrackingLanguage?: (language: string) => void | Promise<void>;
}) {
  const [refreshing, setRefreshing] = useState(false);
  const [refreshError, setRefreshError] = useState<string | null>(null);
  const trackedWorks = useMemo(
    () => works.filter((work) => work.external.anilistId || work.external.mangaDexId || work.external.tracking),
    [works],
  );

  if (!open) return null;

  const refresh = async () => {
    if (refreshing) return;
    setRefreshing(true);
    setRefreshError(null);
    let failures = 0;
    try {
      for (const work of trackedWorks) {
        try {
          const snapshots = await syncTrackingWork({
            anilistId: work.external.anilistId,
            mangaDexId: work.external.mangaDexId,
            trackingLanguage,
          });
          await onPersistWork(workInput(work, snapshots, null));
        } catch (error) {
          failures += 1;
          const message = error instanceof Error ? error.message : String(error);
          const cached = work.external.tracking;
          try {
            await onPersistWork({
              id: work.id,
              title: work.title,
              aliases: work.aliases,
              coverPath: work.coverPath,
              publicationStatus: work.publicationStatus,
              external: {
                ...work.external,
                tracking: preserveTrackingCacheOnError(cached, message),
              },
            });
          } catch {
            // A falha de persistência já conta como uma atualização malsucedida.
          }
        }
      }
    } finally {
      if (failures) setRefreshError(`${failures} obra(s) não puderam ser atualizadas. O cache local foi preservado.`);
      setRefreshing(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex bg-[#111217]/95 text-zinc-100" role="dialog" aria-modal="true" aria-labelledby="studio-updates-title">
      <section className="mx-auto flex h-full w-full max-w-6xl flex-col p-6">
        <header className="flex items-start justify-between border-b border-zinc-800 pb-5">
          <div>
            <small className="uppercase tracking-[0.18em] text-zinc-500">Biblioteca local</small>
            <h2 id="studio-updates-title" className="mt-1 text-2xl font-semibold">Atualizações</h2>
            <p className="mt-1 text-sm text-zinc-400">Somente metadados das fontes vinculadas. Nenhuma página é baixada.</p>
          </div>
          <button type="button" aria-label="Fechar atualizações" className="rounded p-2 hover:bg-zinc-800" onClick={onClose}><X size={19} /></button>
        </header>

        <div className="flex flex-wrap items-center gap-3 border-b border-zinc-800 py-4">
          <button
            type="button"
            disabled={refreshing || trackedWorks.length === 0}
            onClick={() => void refresh()}
            className="inline-flex items-center gap-2 rounded bg-amber-500 px-4 py-2 text-sm font-semibold text-zinc-950 disabled:opacity-50"
          >
            <RefreshCw size={15} className={refreshing ? "animate-spin" : ""} />
            {refreshing ? "Atualizando…" : "Atualizar agora"}
          </button>
          <label className="flex items-center gap-2 text-sm text-zinc-400">
            Idioma dos capítulos
            <select
              value={trackingLanguage}
              onChange={(event) => void onSetTrackingLanguage?.(event.currentTarget.value)}
              className="rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-zinc-100"
            >
              <option value="en">Inglês</option>
              <option value="ja">Japonês</option>
              <option value="ko">Coreano</option>
              <option value="zh">Chinês</option>
              <option value="pt-br">Português (Brasil)</option>
            </select>
          </label>
          <span className="ml-auto text-xs text-zinc-500">Atualização explícita válida por 30 minutos</span>
        </div>

        {refreshError && <p className="mt-4 flex items-center gap-2 rounded border border-amber-700/40 bg-amber-950/30 p-3 text-sm text-amber-200"><AlertTriangle size={15} />{refreshError}</p>}

        <div className="mt-5 grid gap-3 overflow-y-auto pb-6">
          {trackedWorks.length === 0 && (
            <div className="rounded border border-dashed border-zinc-700 p-10 text-center text-zinc-400">
              Vincule uma obra ao AniList ou MangaDex para acompanhar atualizações.
            </div>
          )}
          {trackedWorks.map((work) => {
            const cache = work.external.tracking;
            const snapshots = cache?.snapshots ?? [];
            const chapterSnapshot = snapshots.find((snapshot) => snapshot.provider === "mangadex" && snapshot.latestChapter);
            const status = resolveTrackingStatus(work.publicationStatus, work.external.manualStatusOverride, snapshots);
            const hasChapter = hasRemoteChapterUpdate(work.chapters.map((chapter) => chapter.label), chapterSnapshot?.latestChapter ?? null);
            const stale = isTrackingCacheStale(cache, now);
            return (
              <article key={work.id} className="grid grid-cols-[64px_1fr_auto] gap-4 rounded border border-zinc-800 bg-zinc-900/60 p-4">
                <div className="h-20 overflow-hidden rounded bg-zinc-800">
                  {(work.coverPath || snapshots[0]?.coverUrl) && <img className="h-full w-full object-cover" src={work.coverPath ?? snapshots[0]?.coverUrl ?? undefined} alt="" />}
                </div>
                <div>
                  <div className="flex flex-wrap items-center gap-2">
                    <h3 className="font-semibold">{work.title}</h3>
                    {stale && <span className="rounded bg-amber-950 px-2 py-0.5 text-xs text-amber-300">Desatualizado</span>}
                    {status.source === "manual" && <span className="rounded bg-sky-950 px-2 py-0.5 text-xs text-sky-300">Status manual</span>}
                    {status.hasConflict && <span className="rounded bg-rose-950 px-2 py-0.5 text-xs text-rose-300">Conflito</span>}
                  </div>
                  <p className="mt-1 text-sm text-zinc-400">{STATUS_LABELS[status.status]}</p>
                  {hasChapter && <p className="mt-2 text-sm text-emerald-300">Capítulo {chapterSnapshot?.latestChapter} disponível na fonte.</p>}
                  {!hasChapter && chapterSnapshot?.latestChapter && <p className="mt-2 text-sm text-zinc-500">Último capítulo remoto: {chapterSnapshot.latestChapter}</p>}
                  {cache?.lastError && <p className="mt-2 text-xs text-amber-300">Offline: {cache.lastError}</p>}
                  {cache && (
                    <p className="mt-1 text-xs text-zinc-500">
                      Última atualização: <time dateTime={cache.fetchedAt}>{formatTrackingTime(cache.fetchedAt)}</time>
                    </p>
                  )}
                  <div className="mt-2 flex gap-3 text-xs text-zinc-500">
                    {snapshots.map((snapshot) => <span key={`${snapshot.provider}:${snapshot.providerId}`}>{snapshot.provider === "anilist" ? "AniList" : "MangaDex"}</span>)}
                  </div>
                </div>
                <div className="flex flex-col items-end gap-2">
                  <button type="button" className="rounded border border-zinc-700 px-3 py-2 text-sm hover:bg-zinc-800" onClick={() => onOpenWork(work.id)}>Abrir obra</button>
                  {chapterSnapshot?.siteUrl && <a className="inline-flex items-center gap-1 text-xs text-zinc-400 hover:text-zinc-100" href={chapterSnapshot.siteUrl} target="_blank" rel="noreferrer">Ver fonte <ExternalLink size={12} /></a>}
                </div>
              </article>
            );
          })}
        </div>
      </section>
    </div>
  );
}

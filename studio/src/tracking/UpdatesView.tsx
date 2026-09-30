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
    <div className="studio-updates-backdrop studio-floating-layer" role="dialog" aria-modal="true" aria-labelledby="studio-updates-title">
      <section className="studio-updates-view">
        <header className="studio-updates-header">
          <div>
            <small>Biblioteca local</small>
            <h2 id="studio-updates-title">Atualizações</h2>
            <p>Somente metadados das fontes vinculadas. Nenhuma página é baixada.</p>
          </div>
          <button type="button" aria-label="Fechar atualizações" className="studio-updates-close" onClick={onClose}><X size={19} /></button>
        </header>

        <div className="studio-updates-controls">
          <button
            type="button"
            disabled={refreshing || trackedWorks.length === 0}
            onClick={() => void refresh()}
            className="studio-updates-primary"
          >
            <RefreshCw size={15} className={refreshing ? "animate-spin" : ""} />
            {refreshing ? "Atualizando…" : "Atualizar agora"}
          </button>
          <label className="studio-updates-language">
            Idioma dos capítulos
            <select
              value={trackingLanguage}
              onChange={(event) => void onSetTrackingLanguage?.(event.currentTarget.value)}
            >
              <option value="en">Inglês</option>
              <option value="ja">Japonês</option>
              <option value="ko">Coreano</option>
              <option value="zh">Chinês</option>
              <option value="pt-br">Português (Brasil)</option>
            </select>
          </label>
          <span className="studio-updates-ttl">Atualização explícita válida por 30 minutos</span>
        </div>

        {refreshError && <p className="studio-updates-warning"><AlertTriangle size={15} />{refreshError}</p>}

        <div className="studio-updates-list">
          {trackedWorks.length === 0 && (
            <div className="studio-updates-empty">
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
              <article key={work.id} className="studio-updates-card">
                <div className="studio-updates-cover">
                  {(work.coverPath || snapshots[0]?.coverUrl) && <img src={work.coverPath ?? snapshots[0]?.coverUrl ?? undefined} alt="" />}
                </div>
                <div className="studio-updates-body">
                  <div className="studio-updates-heading">
                    <h3>{work.title}</h3>
                    {stale && <span className="studio-updates-warning">Desatualizado</span>}
                    {status.source === "manual" && <span className="studio-updates-badge studio-updates-badge-manual">Status manual</span>}
                    {status.hasConflict && <span className="studio-updates-badge studio-updates-badge-conflict">Conflito</span>}
                  </div>
                  <p className="studio-updates-status">{STATUS_LABELS[status.status]}</p>
                  {hasChapter && <p className="studio-updates-available">Capítulo {chapterSnapshot?.latestChapter} disponível na fonte.</p>}
                  {!hasChapter && chapterSnapshot?.latestChapter && <p className="studio-updates-muted">Último capítulo remoto: {chapterSnapshot.latestChapter}</p>}
                  {cache?.lastError && <p className="studio-updates-error">Offline: {cache.lastError}</p>}
                  {cache && (
                    <p className="studio-updates-meta">
                      Última atualização: <time dateTime={cache.fetchedAt}>{formatTrackingTime(cache.fetchedAt)}</time>
                    </p>
                  )}
                  <div className="studio-updates-providers">
                    {snapshots.map((snapshot) => <span key={`${snapshot.provider}:${snapshot.providerId}`}>{snapshot.provider === "anilist" ? "AniList" : "MangaDex"}</span>)}
                  </div>
                </div>
                <div className="studio-updates-actions">
                  <button type="button" onClick={() => onOpenWork(work.id)}>Abrir obra</button>
                  {chapterSnapshot?.siteUrl && <a href={chapterSnapshot.siteUrl} target="_blank" rel="noreferrer">Ver fonte <ExternalLink size={12} /></a>}
                </div>
              </article>
            );
          })}
        </div>
      </section>
    </div>
  );
}

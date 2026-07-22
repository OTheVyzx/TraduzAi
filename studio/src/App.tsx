import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { ArrowRight, Clock3, FolderOpen } from "lucide-react";
import { useStudioProjectStore } from "./store/projectStore";
import traduzaiLogoUrl from "../../traduzaistudiologo.svg";

const StudioSharedEditor = lazy(async () => {
  const mod = await import("./editor/StudioSharedEditor");
  return { default: mod.StudioSharedEditor };
});

const RECENTS_KEY = "traduzai-studio-recents";
const MAX_RECENTS = 72;

interface RecentProject {
  path: string;
  name: string;
  updatedAt: number;
}

function readRecentProjects(): RecentProject[] {
  try {
    const raw = localStorage.getItem(RECENTS_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((item): item is RecentProject => (
        item &&
        typeof item.path === "string" &&
        typeof item.name === "string" &&
        typeof item.updatedAt === "number"
      ))
      .slice(0, MAX_RECENTS);
  } catch {
    return [];
  }
}

function writeRecentProjects(items: RecentProject[]) {
  localStorage.setItem(RECENTS_KEY, JSON.stringify(items.slice(0, MAX_RECENTS)));
}

function recentNameFromPath(path: string, projectTitle?: string | null) {
  if (projectTitle?.trim()) return projectTitle.trim();
  const normalized = path.replace(/\\/g, "/").replace(/\/project\.json$/i, "");
  const parts = normalized.split("/").filter(Boolean);
  return parts.at(-1) ?? path;
}

function relativeRecentAge(timestamp: number) {
  const diff = Date.now() - timestamp;
  const minute = 60 * 1000;
  const hour = 60 * minute;
  const day = 24 * hour;
  if (diff < hour) return `${Math.max(1, Math.round(diff / minute))}m`;
  if (diff < day) return `${Math.round(diff / hour)}h`;
  return `${Math.round(diff / day)}d`;
}

export function App() {
  const project = useStudioProjectStore((state) => state.project);
  const projectPath = useStudioProjectStore((state) => state.projectPath);
  const error = useStudioProjectStore((state) => state.error);
  const loadProject = useStudioProjectStore((state) => state.loadProject);
  const openProjectFromDialog = useStudioProjectStore((state) => state.openProjectFromDialog);
  const recoverySnapshot = useStudioProjectStore((state) => state.recoverySnapshot);
  const restoreRecovery = useStudioProjectStore((state) => state.restoreRecovery);
  const dismissRecovery = useStudioProjectStore((state) => state.dismissRecovery);
  const [recents, setRecents] = useState<RecentProject[]>(() => readRecentProjects());

  useEffect(() => {
    if (project) return;
    const configuredProjectPath = import.meta.env.VITE_STUDIO_PROJECT_PATH?.trim();
    if (configuredProjectPath) {
      void loadProject(configuredProjectPath);
    }
  }, [loadProject, project]);

  useEffect(() => {
    if (!project || !projectPath) return;
    if (projectPath.startsWith("memory://")) return;
    setRecents((current) => {
      const next = [
        {
          path: projectPath,
          name: recentNameFromPath(projectPath, project.obra),
          updatedAt: Date.now(),
        },
        ...current.filter((item) => item.path !== projectPath),
      ].slice(0, MAX_RECENTS);
      writeRecentProjects(next);
      return next;
    });
  }, [project, projectPath]);

  if (project && projectPath) {
    return (
      <Suspense fallback={<StudioBoot message="Carregando editor..." />}>
        <StudioSharedEditor project={project} projectPath={projectPath} />
      </Suspense>
    );
  }

  return (
    <StudioHome
      error={error}
      recents={recents}
      onOpenProject={() => void openProjectFromDialog().then(() => setRecents(readRecentProjects()))}
      onOpenRecent={(path) => void loadProject(path)}
      recoveryAvailable={Boolean(recoverySnapshot)}
      onRecover={() => void restoreRecovery()}
      onDismissRecovery={() => void dismissRecovery()}
    />
  );
}

function StudioBoot({ message, error }: { message: string; error?: string | null }) {
  return (
    <main className="studio-boot">
      <section className="studio-boot-panel">
        <p className="eyebrow">TraduzAI Studio</p>
        <h1>Preparando editor</h1>
        <p>{message}</p>
        {error && <p className="error">{error}</p>}
      </section>
    </main>
  );
}

export function StudioHome({
  error,
  recents,
  onOpenProject,
  onOpenRecent,
  recoveryAvailable = false,
  onRecover,
  onDismissRecovery,
}: {
  error?: string | null;
  recents: RecentProject[];
  onOpenProject: () => void;
  onOpenRecent: (path: string) => void;
  recoveryAvailable?: boolean;
  onRecover?: () => void;
  onDismissRecovery?: () => void;
}) {
  const visibleRecents = useMemo(() => recents.slice(0, 6), [recents]);
  return (
    <main className="studio-home">
      <section className="studio-home-shell">
        <div className="studio-home-brand">
          <img className="studio-home-wordmark" src={traduzaiLogoUrl} alt="TraduzAI Studio" />
          <p>Editor mestre de pós-tradução para mangá</p>
        </div>

        <div className="studio-home-actions">
          <button type="button" className="studio-home-primary" onClick={onOpenProject}>
            <span className="studio-home-primary-icon"><FolderOpen size={22} /></span>
            <span className="studio-home-primary-copy">
              <strong>Abrir projeto TraduzAI</strong>
              <small>Edite um projeto já traduzido pelo TraduzAI Central</small>
            </span>
            <ArrowRight size={18} className="studio-home-primary-arrow" />
          </button>
          {recoveryAvailable && (
            <div className="studio-home-recovery">
              <div>
                <strong>Sessão recuperável encontrada</strong>
                <small>O project.json falhou ou difere do último autosave.</small>
              </div>
              <button type="button" onClick={onRecover}>Recuperar</button>
              <button type="button" onClick={onDismissRecovery}>Ignorar</button>
            </div>
          )}
        </div>

        <div className="studio-home-recents">
          <div className="studio-home-recents-head">
            <span>Recentes</span>
            <span>{recents.length}</span>
          </div>
          <div className="studio-home-recents-list">
            {visibleRecents.length > 0 ? (
              visibleRecents.map((item) => (
                <button key={item.path} type="button" onClick={() => onOpenRecent(item.path)}>
                  <span className="studio-recent-main">
                    <strong>{item.name}</strong>
                    <small>{item.path}</small>
                  </span>
                  <span className="studio-recent-age">
                    <Clock3 size={13} />
                    {relativeRecentAge(item.updatedAt)}
                  </span>
                </button>
              ))
            ) : (
              <div className="studio-home-empty">Nenhum projeto recente.</div>
            )}
          </div>
        </div>

        {error && <p className="studio-home-error">{error}</p>}
      </section>
    </main>
  );
}

import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { ArrowRight, Clock3, FileInput, Plus } from "lucide-react";
import { useStudioProjectStore } from "./store/projectStore";
import traduzaiLogoUrl from "../../traduzaistudiologo.svg";

const StudioSharedEditor = lazy(async () => {
  const mod = await import("./editor/StudioSharedEditor");
  return { default: mod.StudioSharedEditor };
});

function samplePageDataUri() {
  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="900" height="1280" viewBox="0 0 900 1280">
      <rect width="900" height="1280" fill="#f8fafc"/>
      <path d="M120 210 C260 120 430 130 560 240 C700 360 720 560 600 700 C490 830 310 850 180 730 C40 600 20 340 120 210Z" fill="#e5e7eb" stroke="#0f172a" stroke-width="16"/>
      <ellipse cx="448" cy="430" rx="210" ry="130" fill="#ffffff" stroke="#111827" stroke-width="10"/>
      <path d="M310 595 C390 650 510 650 590 595" fill="none" stroke="#111827" stroke-width="14" stroke-linecap="round"/>
      <rect x="110" y="925" width="680" height="210" rx="18" fill="#ffffff" stroke="#111827" stroke-width="8"/>
    </svg>
  `.trim();
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

const sampleProject = JSON.stringify(
  {
    versao: "1.0",
    app: "TraduzAi",
    obra: "Exemplo Studio",
    paginas: [
      {
        numero: 1,
        arquivo_original: samplePageDataUri(),
        arquivo_traduzido: samplePageDataUri(),
        textos: [
          {
            id: "sample-text-1",
            bbox: [235, 350, 665, 505],
            texto: "HELLO",
            traduzido: "OLA",
            tipo: "fala",
            confidence: 0.98,
          },
          {
            id: "sample-text-2",
            bbox: [155, 955, 745, 1105],
            texto: "THIS IS A CLEANING NOTE",
            traduzido: "Texto de revisao da scan",
            tipo: "narracao",
          },
        ],
      },
    ],
  },
  null,
  2,
);

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
  const importProjectJson = useStudioProjectStore((state) => state.importProjectJson);
  const loadProject = useStudioProjectStore((state) => state.loadProject);
  const openProjectFromDialog = useStudioProjectStore((state) => state.openProjectFromDialog);
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
      onNewProject={() => void importProjectJson(sampleProject, "memory://novo-projeto")}
      onImportProject={() => void openProjectFromDialog().then(() => setRecents(readRecentProjects()))}
      onOpenRecent={(path) => void loadProject(path)}
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

function StudioHome({
  error,
  recents,
  onNewProject,
  onImportProject,
  onOpenRecent,
}: {
  error?: string | null;
  recents: RecentProject[];
  onNewProject: () => void;
  onImportProject: () => void;
  onOpenRecent: (path: string) => void;
}) {
  const visibleRecents = useMemo(() => recents.slice(0, 6), [recents]);
  return (
    <main className="studio-home">
      <section className="studio-home-shell">
        <div className="studio-home-brand">
          <img className="studio-home-wordmark" src={traduzaiLogoUrl} alt="TraduzAI Studio" />
          <p>Espaco de trabalho de traducao de manga</p>
        </div>

        <div className="studio-home-actions">
          <button type="button" className="studio-home-primary" onClick={onNewProject}>
            <span className="studio-home-primary-icon"><Plus size={22} /></span>
            <span className="studio-home-primary-copy">
              <strong>Novo projeto</strong>
              <small>Comece uma area de trabalho e importe paginas depois</small>
            </span>
            <ArrowRight size={18} className="studio-home-primary-arrow" />
          </button>

          <button type="button" className="studio-home-import" onClick={onImportProject}>
            <FileInput size={16} />
            Importar projeto
          </button>
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

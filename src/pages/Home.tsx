import { useEffect, useState, useRef } from "react";
import { useNavigate } from "react-router-dom";

function useDynamicStyle<T extends HTMLElement>(styleObj: Record<string, string | number>, deps: any[]) {
  const ref = useRef<T>(null);
  useEffect(() => {
    if (ref.current) {
      for (const [key, value] of Object.entries(styleObj)) {
        if (value === undefined || value === null) {
          ref.current.style.removeProperty(key);
        } else {
          ref.current.style.setProperty(key, String(value));
        }
      }
    }
  }, deps);
  return ref;
}

function ProgressBar({ progress }: { progress: number }) {
  const ref = useDynamicStyle<HTMLDivElement>({ "--progress": `${progress}%` }, [progress]);
  return (
    <div
      ref={ref}
      className="h-full bg-gradient-to-r from-brand to-accent-cyan rounded-pill transition-all duration-320 dynamic-progress"
    />
  );
}
import {
  Plus,
  FolderOpen,
  BookOpen,
  Clock,
  CheckCircle2,
  AlertCircle,
  X,
  Library,
  Sparkles,
  ArrowRight,
} from "lucide-react";
import { useAppStore } from "../lib/stores/appStore";
import {
  loadProjectJson,
  loadSettings,
  openFiles,
  openMultipleSources,
  openProjectDialog,
  validateImport,
} from "../lib/tauri";
import { Card, Badge } from "../components/ui";
import { ONBOARDING_FLOW } from "../lib/onboarding";

export function Home() {
  const navigate = useNavigate();
  const {
    recentProjects,
    freeRemaining,
    credits,
    setProject,
    removeRecentProject,
    setBatchSources,
  } = useAppStore();
  const free = freeRemaining();
  const [loading, setLoading] = useState(false);
  const [showOnboarding, setShowOnboarding] = useState(() => {
    if (typeof window === "undefined") return false;
    return window.localStorage.getItem("traduzai_onboarding_done") !== "1";
  });

  function skipOnboarding() {
    window.localStorage.setItem("traduzai_onboarding_done", "1");
    setShowOnboarding(false);
  }

  function buildProjectFromJson(
    path: string,
    raw: Awaited<ReturnType<typeof loadProjectJson>>
  ) {
    const outputDir = path.replace(/\\/g, "/");
    return {
      id: crypto.randomUUID(),
      obra: raw.obra || "",
      capitulo: raw.capitulo || 1,
      idioma_origem: raw.idioma_origem || "en",
      idioma_destino: raw.idioma_destino || "pt-BR",
      qualidade: "normal" as const,
      contexto: {
        sinopse: raw.contexto?.sinopse || "",
        genero: raw.contexto?.genero || [],
        personagens: raw.contexto?.personagens || [],
        glossario: raw.contexto?.glossario || {},
        aliases: raw.contexto?.aliases || [],
        termos: raw.contexto?.termos || [],
        relacoes: raw.contexto?.relacoes || [],
        faccoes: raw.contexto?.faccoes || [],
        resumo_por_arco: raw.contexto?.resumo_por_arco || [],
        memoria_lexical: raw.contexto?.memoria_lexical || {},
        fontes_usadas: raw.contexto?.fontes_usadas || [],
      },
      work_context: raw.work_context
        ? {
            selected: Boolean(raw.work_context.selected),
            work_id: String(raw.work_context.work_id ?? ""),
            title: String(raw.work_context.title ?? raw.obra ?? ""),
            context_loaded: Boolean(raw.work_context.context_loaded),
            glossary_loaded: Boolean(raw.work_context.glossary_loaded),
            glossary_entries_count: Number(raw.work_context.glossary_entries_count ?? 0),
            risk_level: (raw.work_context.risk_level as "high" | "medium" | "low") ?? "high",
            user_ignored_warning: Boolean(raw.work_context.user_ignored_warning),
          }
        : null,
      preset: raw.preset ?? null,
      paginas: raw.paginas ?? [],
      status: "done" as const,
      source_path: outputDir,
      output_path: outputDir,
      totalPages: raw.paginas?.length ?? 0,
      mode: "auto" as const,
    };
  }

  async function handleNewTranslation() {
    await startProject("auto");
  }

  async function handleManualTranslation() {
    await startProject("manual");
  }

  async function startProject(mode: "auto" | "manual") {
    setLoading(true);
    try {
      const path = await openFiles();
      if (!path) return;
      setBatchSources([]);
      const validation = await validateImport(path);
      if (!validation.valid) {
        alert(`Arquivo inválido: ${validation.error}`);
        return;
      }
      const settings = await loadSettings();
      setProject({
        id: crypto.randomUUID(),
        obra: "",
        capitulo: 1,
        idioma_origem: settings.idioma_origem || "en",
        idioma_destino: settings.idioma_destino || "pt-BR",
        qualidade: "normal",
        contexto: {
          sinopse: "",
          genero: [],
          personagens: [],
          glossario: {},
          aliases: [],
          termos: [],
          relacoes: [],
          faccoes: [],
          resumo_por_arco: [],
          memoria_lexical: {},
          fontes_usadas: [],
        },
        paginas: [],
        status: "setup",
        source_path: path,
        totalPages: validation.pages,
        mode,
      });
      navigate("/setup");
    } catch (err) {
      console.error("Erro ao abrir arquivo:", err);
    } finally {
      setLoading(false);
    }
  }

  async function handleBatchTranslation() {
    setLoading(true);
    try {
      const paths = await openMultipleSources();
      if (!paths || paths.length === 0) return;

      if (paths.length === 1) {
        setBatchSources([]);
        const validation = await validateImport(paths[0]);
        if (!validation.valid) {
          alert(`Arquivo inválido: ${validation.error}`);
          return;
        }
        const settings = await loadSettings();
        setProject({
          id: crypto.randomUUID(),
          mode: "auto" as const,
          obra: "",
          capitulo: 1,
          idioma_origem: settings.idioma_origem || "en",
          idioma_destino: settings.idioma_destino || "pt-BR",
          qualidade: "normal",
          contexto: {
            sinopse: "",
            genero: [],
            personagens: [],
            glossario: {},
            aliases: [],
            termos: [],
            relacoes: [],
            faccoes: [],
            resumo_por_arco: [],
            memoria_lexical: {},
            fontes_usadas: [],
          },
          paginas: [],
          status: "setup",
          source_path: paths[0],
          totalPages: validation.pages,
        });
        navigate("/setup");
        return;
      }

      setBatchSources(paths);
      const settings = await loadSettings();
      setProject({
        id: crypto.randomUUID(),
        mode: "auto" as const,
        obra: "",
        capitulo: 1,
        idioma_origem: settings.idioma_origem || "en",
        idioma_destino: settings.idioma_destino || "pt-BR",
        qualidade: "normal",
        contexto: {
          sinopse: "",
          genero: [],
          personagens: [],
          glossario: {},
          aliases: [],
          termos: [],
          relacoes: [],
          faccoes: [],
          resumo_por_arco: [],
          memoria_lexical: {},
          fontes_usadas: [],
        },
        paginas: [],
        status: "setup",
        source_path: paths[0],
        totalPages: 0,
      });
      navigate("/setup");
    } catch (err) {
      console.error("Erro ao abrir arquivos:", err);
    } finally {
      setLoading(false);
    }
  }

  async function handleOpenProject() {
    setLoading(true);
    try {
      const path = await openProjectDialog();
      if (!path) return;
      const raw = await loadProjectJson(path);
      const project = buildProjectFromJson(path, raw);
      if (project.paginas.length === 0) {
        alert("Projeto sem páginas traduzidas.");
        return;
      }
      setProject(project);
      navigate("/preview");
    } catch (err) {
      console.error("Erro ao abrir projeto:", err);
      alert("Não foi possível abrir o projeto. Selecione a pasta que contém o project.json.");
    } finally {
      setLoading(false);
    }
  }

  const quota = free + credits;
  const progressPercent = Math.min(100, (free / 40) * 100);

  return (
    <div className="relative min-h-full">
      {/* Ambient gradient — sutil e premium */}
      <div className="pointer-events-none absolute inset-x-0 top-0 h-[480px] bg-[radial-gradient(ellipse_at_top,_rgba(108,92,231,0.10),_transparent_55%)]" />
      <div className="pointer-events-none absolute inset-0 bg-noise" />

      <div className="relative px-10 py-10 max-w-5xl mx-auto animate-fade-in">
        {/* Header */}
        <header className="mb-10">
          <div className="flex items-center gap-2.5 mb-3">
            <Badge tone="brand" size="sm" icon={<Sparkles size={10} />}>
              BETA
            </Badge>
            <span className="text-2xs text-text-muted">v0.1 · desktop</span>
          </div>
          <h1 className="text-3xl font-bold text-text-primary tracking-tight">
            Bem-vindo ao{" "}
            <span className="bg-gradient-to-r from-brand-300 to-accent-cyan bg-clip-text text-transparent">
              TraduzAi
            </span>
          </h1>
          <p className="text-md text-text-secondary mt-2 max-w-lg">
            Traduza mangá, manhwa e manhua automaticamente com IA 100% local.
          </p>
          <button
            data-testid="open-onboarding"
            type="button"
            onClick={() => setShowOnboarding(true)}
            className="mt-4 rounded-lg border border-border bg-bg-secondary px-3 py-2 text-xs text-text-secondary transition-smooth hover:text-text-primary"
          >
            Ajuda
          </button>
        </header>

        {/* Quota banner */}
        <Card variant="highlight" padding="lg" className="mb-10">
          <div className="flex items-start justify-between gap-6">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1.5">
                <Sparkles size={14} className="text-brand-300" />
                <p className="text-sm font-semibold text-brand-200">
                  Plano gratuito
                </p>
              </div>
              <p className="text-sm text-text-secondary mb-3">
                {free > 0
                  ? `${free} páginas restantes esta semana · reseta toda segunda-feira`
                  : "Limite semanal atingido — compre créditos para continuar"}
              </p>
              <div className="h-1 bg-white/[0.04] rounded-pill overflow-hidden">
                <ProgressBar progress={progressPercent} />
              </div>
              {credits > 0 && (
                <p className="text-xs text-text-muted mt-3">
                  + {credits} créditos pagos disponíveis
                </p>
              )}
            </div>
            <div className="text-right shrink-0">
              <p className="text-4xl font-bold text-text-primary tabular leading-none tracking-tight">
                {quota}
              </p>
              <p className="text-2xs text-text-muted mt-1 uppercase tracking-wider">
                páginas
              </p>
            </div>
          </div>
        </Card>

        {/* Primary actions */}
        <section className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <button
            onClick={handleNewTranslation}
            disabled={loading}
            className="group relative w-full overflow-hidden rounded-xl p-6 text-left
              bg-gradient-to-br from-brand-500/90 to-brand-700/90
              hover:from-brand-400/90 hover:to-brand-600/90
              shadow-glow-brand/50 hover:shadow-glow-brand
              transition-all duration-240 ease-out-expo
              disabled:opacity-40 disabled:cursor-not-allowed
              focus-visible:ring-2 focus-visible:ring-brand-300 focus-visible:ring-offset-2 focus-visible:ring-offset-bg-primary"
          >
            <div className="absolute inset-0 bg-grid opacity-20" />
            <div className="relative flex items-center gap-5">
              <div className="w-12 h-12 rounded-xl bg-white/[0.06] backdrop-blur-sm flex items-center justify-center shrink-0
                group-hover:scale-105 group-hover:bg-white/[0.08] transition-all duration-240">
                <Plus size={24} className="text-white" strokeWidth={2.5} />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-lg font-semibold text-white">Novo capítulo</p>
                <p className="text-sm text-white/60 mt-0.5">
                  Automático: OCR + Tradução IA
                </p>
              </div>
              <ArrowRight
                size={20}
                className="text-white/40 group-hover:text-white/80 group-hover:translate-x-1 transition-all duration-240"
              />
            </div>
          </button>

          <button
            onClick={handleManualTranslation}
            disabled={loading}
            className="group relative w-full overflow-hidden rounded-xl p-6 text-left
              bg-bg-secondary border border-border hover:border-brand/25 hover:bg-brand/[0.03]
              shadow-card hover:shadow-card-hover
              transition-all duration-240 ease-out-expo
              disabled:opacity-40 disabled:cursor-not-allowed
              focus-visible:ring-2 focus-visible:ring-brand/30 focus-visible:ring-offset-2 focus-visible:ring-offset-bg-primary"
          >
            <div className="relative flex items-center gap-5">
              <div className="w-12 h-12 rounded-xl bg-accent-violet/8 flex items-center justify-center shrink-0
                group-hover:scale-105 group-hover:bg-accent-violet/12 transition-all duration-240 border border-accent-violet/15">
                <Plus size={24} className="text-accent-violet" strokeWidth={2.5} />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-lg font-semibold text-text-primary">Tradução Manual</p>
                <p className="text-sm text-text-secondary mt-0.5">
                  Apenas extração: Controle total
                </p>
              </div>
              <ArrowRight
                size={20}
                className="text-text-muted group-hover:text-text-secondary group-hover:translate-x-1 transition-all duration-240"
              />
            </div>
          </button>
        </section>

        {/* Secondary actions */}
        <section className="grid grid-cols-2 gap-4 mb-12">
          <SecondaryAction
            icon={Library}
            title="Tradução em lote"
            description="Vários capítulos em sequência"
            onClick={handleBatchTranslation}
            disabled={loading}
          />
          <SecondaryAction
            icon={FolderOpen}
            title="Abrir projeto"
            description="Continuar tradução existente"
            onClick={handleOpenProject}
            disabled={loading}
            accent="cyan"
          />
        </section>

        {/* Recent projects */}
        {recentProjects.length > 0 && (
          <section>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <Clock size={14} className="text-text-muted" />
                <h2 className="text-xs font-medium text-text-muted uppercase tracking-wider">
                  Projetos recentes
                </h2>
              </div>
              <span className="text-2xs text-text-muted tabular">
                {recentProjects.length}
              </span>
            </div>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
              {recentProjects.map((proj) => (
                <Card
                  key={proj.id}
                  variant="interactive"
                  padding="md"
                  className="group relative"
                >
                  <div className="flex items-start justify-between gap-2 mb-3">
                    <div className="flex items-center gap-2 min-w-0">
                      <BookOpen size={14} className="text-brand-300 shrink-0" />
                      <span className="text-2xs text-text-muted tabular uppercase tracking-wider">
                        Cap. {proj.capitulo}
                      </span>
                    </div>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        removeRecentProject(proj.id);
                      }}
                      className="shrink-0 w-7 h-7 rounded-lg text-text-muted opacity-0 group-hover:opacity-100
                        hover:text-text-primary hover:bg-white/[0.04] transition-all duration-200
                        flex items-center justify-center"
                      aria-label={`Remover ${proj.obra} dos recentes`}
                      title="Remover dos recentes"
                    >
                      <X size={13} />
                    </button>
                  </div>
                  <p className="text-sm font-medium text-text-primary truncate mb-2">
                    {proj.obra || "Sem nome"}
                  </p>
                  <div className="flex items-center gap-1.5">
                    {proj.status === "done" ? (
                      <CheckCircle2 size={12} className="text-status-success" />
                    ) : (
                      <AlertCircle size={12} className="text-status-warning" />
                    )}
                    <span className="text-xs text-text-secondary tabular">
                      {proj.pages} páginas
                    </span>
                  </div>
                </Card>
              ))}
            </div>
          </section>
        )}
      </div>

      {showOnboarding && (
        <div data-testid="onboarding-modal" className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4">
          <div className="w-full max-w-md rounded-xl border border-border bg-bg-secondary p-5 shadow-2xl">
            <div className="mb-4 flex items-start justify-between gap-3">
              <div>
                <h2 className="text-base font-semibold text-text-primary">Como o TraduzAI funciona</h2>
                <p className="mt-1 text-xs text-text-secondary">Fluxo recomendado para produzir um capitulo com controle.</p>
              </div>
              <button type="button" onClick={skipOnboarding} className="rounded-lg p-1 text-text-muted hover:text-text-primary">
                <X size={16} />
              </button>
            </div>
            <ol className="space-y-2">
              {ONBOARDING_FLOW.map((step, index) => (
                <li key={step} className="flex items-center gap-3 rounded-lg border border-border bg-bg-tertiary px-3 py-2 text-sm text-text-primary">
                  <span className="flex h-6 w-6 items-center justify-center rounded-full bg-brand/10 text-xs text-brand-300">{index + 1}</span>
                  {step}
                </li>
              ))}
            </ol>
            <button
              data-testid="skip-onboarding"
              type="button"
              onClick={skipOnboarding}
              className="mt-4 w-full rounded-lg bg-brand px-3 py-2 text-sm font-medium text-white transition-smooth hover:bg-brand-600"
            >
              Pular tutorial
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

interface SecondaryActionProps {
  icon: typeof Library;
  title: string;
  description: string;
  onClick: () => void;
  disabled?: boolean;
  accent?: "brand" | "cyan";
}

function SecondaryAction({
  icon: Icon,
  title,
  description,
  onClick,
  disabled,
  accent = "brand",
}: SecondaryActionProps) {
  const accentClasses =
    accent === "cyan"
      ? "group-hover:border-accent-cyan/25 group-hover:bg-accent-cyan/[0.03]"
      : "group-hover:border-brand/25 group-hover:bg-brand/[0.03]";
  const iconColor = accent === "cyan" ? "text-accent-cyan" : "text-brand-300";
  const iconBg =
    accent === "cyan"
      ? "bg-accent-cyan/8 group-hover:bg-accent-cyan/12"
      : "bg-brand/8 group-hover:bg-brand/12";

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`group flex items-center gap-4 p-5 rounded-xl text-left
        bg-bg-secondary border border-border shadow-card
        transition-all duration-200 ease-out-expo
        ${accentClasses}
        hover:shadow-card-hover
        disabled:opacity-35 disabled:cursor-not-allowed
        focus-visible:outline-none`}
    >
      <div
        className={`w-10 h-10 rounded-lg flex items-center justify-center shrink-0 transition-all duration-200 ${iconBg}`}
      >
        <Icon size={20} className={iconColor} strokeWidth={1.75} />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-text-primary">{title}</p>
        <p className="text-xs text-text-muted mt-0.5">{description}</p>
      </div>
      <ArrowRight
        size={16}
        className="text-text-muted group-hover:text-text-secondary group-hover:translate-x-0.5 transition-all duration-200"
      />
    </button>
  );
}

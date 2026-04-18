import { useNavigate } from "react-router-dom";
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
import { useState } from "react";
import { Card, Badge } from "../components/ui";

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
      paginas: (raw.paginas ?? []).map((page) => ({
        ...page,
        arquivo_original: `${outputDir}/${page.arquivo_original}`.replace(/\\/g, "/"),
        arquivo_traduzido: `${outputDir}/${page.arquivo_traduzido}`.replace(/\\/g, "/"),
        inpaint_blocks: page.inpaint_blocks ?? [],
      })),
      status: "done" as const,
      source_path: outputDir,
      output_path: outputDir,
      totalPages: raw.paginas?.length ?? 0,
    };
  }

  async function handleNewTranslation() {
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
      {/* Ambient gradient */}
      <div className="pointer-events-none absolute inset-x-0 top-0 h-[420px] bg-[radial-gradient(ellipse_at_top,_rgba(124,92,255,0.18),_transparent_55%)]" />

      <div className="relative px-10 py-10 max-w-5xl mx-auto animate-fade-in">
        {/* Header */}
        <header className="mb-10">
          <div className="flex items-center gap-2 mb-3">
            <Badge tone="brand" size="sm" icon={<Sparkles size={10} />}>
              BETA
            </Badge>
            <span className="text-xs text-text-muted">v0.1 · desktop</span>
          </div>
          <h1 className="text-3xl font-semibold text-text-primary tracking-tight">
            Bem-vindo ao TraduzAi
          </h1>
          <p className="text-md text-text-secondary mt-2">
            Traduza mangá, manhwa e manhua automaticamente com IA local.
          </p>
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
                  ? `${free} páginas restantes esta semana · 2 capítulos grátis, reseta toda segunda-feira`
                  : "Limite semanal atingido — compre créditos para continuar"}
              </p>
              <div className="h-1.5 bg-white/5 rounded-pill overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-brand to-accent-cyan rounded-pill transition-all duration-320"
                  style={{ width: `${progressPercent}%` }}
                />
              </div>
              {credits > 0 && (
                <p className="text-xs text-text-secondary mt-3">
                  + {credits} créditos pagos disponíveis
                </p>
              )}
            </div>
            <div className="text-right shrink-0">
              <p className="text-4xl font-semibold text-text-primary tabular leading-none">
                {quota}
              </p>
              <p className="text-xs text-text-muted mt-1 uppercase tracking-wider">
                páginas
              </p>
            </div>
          </div>
        </Card>

        {/* Primary action */}
        <section className="mb-8">
          <button
            onClick={handleNewTranslation}
            disabled={loading}
            className="group relative w-full overflow-hidden rounded-xl p-6 text-left
              bg-gradient-to-br from-brand-600 to-brand-700 hover:from-brand-500 hover:to-brand-600
              shadow-glow-brand transition-all duration-240 ease-out-expo
              disabled:opacity-50 disabled:cursor-not-allowed
              focus-visible:ring-2 focus-visible:ring-brand-300 focus-visible:ring-offset-2 focus-visible:ring-offset-bg-primary"
          >
            <div className="absolute inset-0 bg-grid opacity-40" />
            <div className="relative flex items-center gap-5">
              <div className="w-12 h-12 rounded-lg bg-white/15 backdrop-blur-sm flex items-center justify-center shrink-0
                group-hover:scale-105 transition-transform duration-240">
                <Plus size={24} className="text-white" strokeWidth={2.5} />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-lg font-semibold text-white">Novo capítulo</p>
                <p className="text-sm text-white/70 mt-0.5">
                  Importe um CBZ, ZIP ou pasta com imagens
                </p>
              </div>
              <ArrowRight
                size={20}
                className="text-white/70 group-hover:text-white group-hover:translate-x-1 transition-all duration-240"
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
                <Clock size={14} className="text-text-secondary" />
                <h2 className="text-sm font-medium text-text-secondary uppercase tracking-wider">
                  Projetos recentes
                </h2>
              </div>
              <span className="text-xs text-text-muted tabular">
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
                      className="shrink-0 w-7 h-7 rounded-md text-text-muted opacity-0 group-hover:opacity-100
                        hover:text-text-primary hover:bg-white/5 transition-all duration-180
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
      ? "group-hover:border-accent-cyan/40 group-hover:bg-accent-cyan/5"
      : "group-hover:border-brand/40 group-hover:bg-brand/5";
  const iconColor = accent === "cyan" ? "text-accent-cyan" : "text-brand-300";
  const iconBg =
    accent === "cyan"
      ? "bg-accent-cyan/10 group-hover:bg-accent-cyan/15"
      : "bg-brand/10 group-hover:bg-brand/15";

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`group flex items-center gap-4 p-5 rounded-lg text-left
        bg-bg-secondary border border-border transition-all duration-180 ease-out-expo
        ${accentClasses}
        disabled:opacity-40 disabled:cursor-not-allowed
        focus-visible:outline-none`}
    >
      <div
        className={`w-10 h-10 rounded-md flex items-center justify-center shrink-0 transition-colors duration-180 ${iconBg}`}
      >
        <Icon size={20} className={iconColor} />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-text-primary">{title}</p>
        <p className="text-xs text-text-secondary mt-0.5">{description}</p>
      </div>
      <ArrowRight
        size={16}
        className="text-text-muted group-hover:text-text-secondary group-hover:translate-x-0.5 transition-all duration-180"
      />
    </button>
  );
}

import { useNavigate } from "react-router-dom";
import { Plus, FolderOpen, BookOpen, Clock, CheckCircle2, AlertCircle, X } from "lucide-react";
import { useAppStore } from "../lib/stores/appStore";
import { loadProjectJson, loadSettings, openFiles, openProjectDialog, validateImport } from "../lib/tauri";
import { useState } from "react";

export function Home() {
  const navigate = useNavigate();
  const { recentProjects, freeRemaining, credits, setProject, removeRecentProject } = useAppStore();
  const free = freeRemaining();
  const [loading, setLoading] = useState(false);

  function buildProjectFromJson(path: string, raw: Awaited<ReturnType<typeof loadProjectJson>>) {
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
        idioma_origem: "en",
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

  return (
    <div className="p-8 max-w-4xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h2 className="text-2xl font-bold text-text-primary">Bem-vindo ao TraduzAi</h2>
        <p className="text-text-secondary mt-1">
          Traduza mangá, manhwa e manhua automaticamente com IA
        </p>
      </div>

      {/* Free tier info */}
      <div className="bg-accent-purple/5 border border-accent-purple/20 rounded-xl px-5 py-4 mb-8">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium text-accent-purple-light">
              Plano gratuito
            </p>
            <p className="text-xs text-text-secondary mt-0.5">
              {free > 0
                ? `${free} páginas restantes esta semana (2 capítulos/semana)`
                : "Limite semanal atingido — compre créditos para continuar"}
            </p>
          </div>
          <div className="text-right">
            <p className="text-lg font-bold text-accent-purple">{free}</p>
            <p className="text-xs text-text-secondary">páginas</p>
          </div>
        </div>
        {credits > 0 && (
          <p className="text-xs text-text-secondary mt-2 pt-2 border-t border-accent-purple/10">
            + {credits} créditos pagos disponíveis
          </p>
        )}
      </div>

      {/* Action buttons */}
      <div className="grid grid-cols-2 gap-4 mb-10">
        <button
          onClick={handleNewTranslation}
          disabled={loading}
          className="group flex flex-col items-center justify-center gap-3 p-8 rounded-xl
            bg-bg-secondary border border-white/5 hover:border-accent-purple/40
            hover:bg-accent-purple/5 transition-smooth disabled:opacity-50"
        >
          <div className="w-12 h-12 rounded-xl bg-accent-purple/10 flex items-center justify-center
            group-hover:bg-accent-purple/20 transition-smooth">
            <Plus size={24} className="text-accent-purple" />
          </div>
          <div className="text-center">
            <p className="font-medium text-text-primary">Nova Tradução</p>
            <p className="text-xs text-text-secondary mt-1">
              Importar .zip, .cbz, imagem ou pasta
            </p>
          </div>
        </button>

        <button
          onClick={handleOpenProject}
          disabled={loading}
          className="group flex flex-col items-center justify-center gap-3 p-8 rounded-xl
            bg-bg-secondary border border-white/5 hover:border-accent-cyan/40
            hover:bg-accent-cyan/5 transition-smooth disabled:opacity-50"
        >
          <div className="w-12 h-12 rounded-xl bg-accent-cyan/10 flex items-center justify-center
            group-hover:bg-accent-cyan/20 transition-smooth">
            <FolderOpen size={24} className="text-accent-cyan" />
          </div>
          <div className="text-center">
            <p className="font-medium text-text-primary">Abrir Projeto</p>
            <p className="text-xs text-text-secondary mt-1">
              Continuar tradução existente
            </p>
          </div>
        </button>
      </div>

      {/* Recent projects */}
      {recentProjects.length > 0 && (
        <div>
          <h3 className="text-sm font-medium text-text-secondary mb-3 flex items-center gap-2">
            <Clock size={14} />
            Projetos recentes
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {recentProjects.map((proj) => (
              <div
                key={proj.id}
                className="flex flex-col p-4 rounded-lg bg-bg-secondary border border-white/5
                  hover:border-white/10 hover:bg-bg-tertiary transition-smooth text-left"
              >
                <div className="flex items-start justify-between gap-2 mb-2">
                  <div className="flex items-center gap-2 min-w-0">
                    <BookOpen size={14} className="text-accent-purple shrink-0" />
                    <span className="text-xs text-text-secondary truncate">
                      Cap. {proj.capitulo}
                    </span>
                  </div>
                  <button
                    type="button"
                    onClick={() => removeRecentProject(proj.id)}
                    className="shrink-0 w-6 h-6 rounded-md border border-white/5 text-text-secondary
                      hover:text-text-primary hover:border-white/15 hover:bg-white/5 transition-smooth"
                    aria-label={`Remover ${proj.obra} dos recentes`}
                    title="Remover dos recentes"
                  >
                    <span className="flex items-center justify-center">
                      <X size={12} />
                    </span>
                  </button>
                </div>
                <p className="text-sm font-medium text-text-primary truncate">
                  {proj.obra}
                </p>
                <div className="flex items-center gap-1 mt-2">
                  {proj.status === "done" ? (
                    <CheckCircle2 size={12} className="text-status-success" />
                  ) : (
                    <AlertCircle size={12} className="text-status-warning" />
                  )}
                  <span className="text-xs text-text-secondary">
                    {proj.pages} páginas
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

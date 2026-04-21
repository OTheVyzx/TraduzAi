import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Search, Plus, X, Rocket, ArrowLeft, BookOpen, Sparkles, Globe, Gauge, Cpu } from "lucide-react";
import { LanguageSelectField } from "../components/ui";
import { useAppStore } from "../lib/stores/appStore";
import {
  enrichWorkContext,
  loadSupportedLanguages,
  searchWork,
  type WorkSearchCandidate,
} from "../lib/tauri";
import {
  getLanguageOptions,
  normalizeLanguageCodeForSelection,
} from "../lib/languages";
import {
  buildPipelineTimeEstimate,
  formatDuration,
  formatTierLabel,
} from "../lib/time-estimates";

const DEFAULT_QUALITY = "alta" as const;

export function Setup() {
  const navigate = useNavigate();
  const {
    project,
    updateProject,
    canTranslate,
    systemProfile,
    setSetupEstimate,
    batchSources,
    setBatchSources,
  } = useAppStore();

  const [obraSearch, setObraSearch] = useState(project?.obra || "");
  const [searching, setSearching] = useState(false);
  const [newTerm, setNewTerm] = useState({ key: "", value: "" });
  const [candidates, setCandidates] = useState<WorkSearchCandidate[]>([]);
  const [searchError, setSearchError] = useState("");
  const [loadingCandidateId, setLoadingCandidateId] = useState<string | null>(null);
  const [supportedLanguages, setSupportedLanguages] = useState(getLanguageOptions(null));

  const totalPages = project?.totalPages ?? 0;
  const estimate = buildPipelineTimeEstimate(systemProfile, totalPages, DEFAULT_QUALITY);

  useEffect(() => {
    let active = true;

    loadSupportedLanguages()
      .then((languages) => {
        if (!active) return;
        const options = getLanguageOptions(languages);
        setSupportedLanguages(options);
        if (!project) return;

        const idioma_origem = normalizeLanguageCodeForSelection(project.idioma_origem, options, "en");
        const idioma_destino = normalizeLanguageCodeForSelection(project.idioma_destino, options, "pt");
        if (idioma_origem !== project.idioma_origem || idioma_destino !== project.idioma_destino) {
          updateProject({ idioma_origem, idioma_destino });
        }
      })
      .catch(() => {
        if (!active) return;
        setSupportedLanguages(getLanguageOptions(null));
      });

    return () => {
      active = false;
    };
  }, [project?.id, updateProject]);

  async function handleSearchObra() {
    if (!obraSearch.trim()) return;
    setSearching(true);
    setSearchError("");
    try {
      const result = await searchWork(obraSearch);
      setCandidates(result.candidates);
      if (result.candidates.length === 0) {
        setSearchError("Nenhuma obra compativel encontrada em AniList, Webnovel ou Fandom.");
      }
    } catch (err) {
      console.error("Erro ao buscar obra:", err);
      setSearchError("Nao foi possivel buscar a obra agora.");
    } finally {
      setSearching(false);
    }
  }

  async function handleUseCandidate(candidate: WorkSearchCandidate) {
    setLoadingCandidateId(candidate.id);
    setSearchError("");
    try {
      const result = await enrichWorkContext(candidate);
      updateProject({
        obra: result.title,
        contexto: {
          sinopse: result.synopsis,
          genero: result.genres,
          personagens: result.characters,
          glossario: project?.contexto.glossario || {},
          aliases: result.aliases,
          termos: result.terms,
          relacoes: result.relationships,
          faccoes: result.factions,
          resumo_por_arco: result.arc_summaries,
          memoria_lexical: result.lexical_memory,
          fontes_usadas: result.sources_used.map((source) => ({
            fonte: source.source,
            titulo: source.title,
            url: source.url,
            trecho: source.snippet,
          })),
        },
      });
      setObraSearch(result.title);
      setCandidates([]);
    } catch (err) {
      console.error("Erro ao enriquecer contexto:", err);
      setSearchError("Nao foi possivel carregar o contexto dessa obra.");
    } finally {
      setLoadingCandidateId(null);
    }
  }

  function addGlossaryTerm() {
    if (!newTerm.key || !newTerm.value || !project) return;
    const updated = {
      ...project.contexto.glossario,
      [newTerm.key]: newTerm.value,
    };
    updateProject({
      contexto: { ...project.contexto, glossario: updated },
    });
    setNewTerm({ key: "", value: "" });
  }

  function removeGlossaryTerm(key: string) {
    if (!project) return;
    const { [key]: _, ...rest } = project.contexto.glossario;
    updateProject({
      contexto: { ...project.contexto, glossario: rest },
    });
  }

  function handleStart() {
    if (!project) return;
    if (batchSources.length > 0) {
      updateProject({ qualidade: DEFAULT_QUALITY, status: "processing" });
      navigate("/processing");
      return;
    }
    if (!canTranslate(totalPages)) {
      alert("Creditos insuficientes para traduzir este capitulo.");
      return;
    }
    setSetupEstimate(estimate);
    updateProject({ qualidade: DEFAULT_QUALITY, status: "processing" });
    navigate("/processing");
  }

  if (!project) {
    navigate("/");
    return null;
  }

  const hasEnoughCredits = canTranslate(totalPages);
  const sourceValue = normalizeLanguageCodeForSelection(project.idioma_origem, supportedLanguages, "en");
  const targetValue = normalizeLanguageCodeForSelection(project.idioma_destino, supportedLanguages, "pt");

  return (
    <div className="p-8 max-w-2xl mx-auto">
      <button
        onClick={() => navigate("/")}
        className="flex items-center gap-2 text-sm text-text-secondary hover:text-text-primary mb-6 transition-smooth"
      >
        <ArrowLeft size={16} />
        Voltar
      </button>

      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-bold">Configurar projeto</h2>
        {project.mode === "manual" && (
          <span className="px-2.5 py-1 rounded-full bg-accent-purple/10 border border-accent-purple/20 text-accent-purple text-[10px] font-bold uppercase tracking-widest flex items-center gap-1.5 animate-pulse">
            <span className="w-1.5 h-1.5 rounded-full bg-accent-purple" />
            Modo Manual
          </span>
        )}
      </div>

      <div className="mb-4">
        <label className="text-sm text-text-secondary mb-2 block">Nome da obra <span className="text-text-secondary/40">(opcional)</span></label>
        <div className="flex gap-2">
          <div className="flex-1 relative">
            <input
              type="text"
              value={obraSearch}
              onChange={(e) => setObraSearch(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearchObra()}
              placeholder="Ex: Solo Leveling, One Piece..."
              className="w-full px-4 py-2.5 bg-bg-secondary border border-white/10 rounded-lg
                text-text-primary placeholder:text-text-secondary/50 focus:border-accent-purple/50
                focus:outline-none transition-smooth"
            />
            <Search size={16} className="absolute right-3 top-3 text-text-secondary/50" />
          </div>
          <button
            onClick={handleSearchObra}
            disabled={searching}
            className="px-4 py-2.5 bg-accent-purple/10 text-accent-purple rounded-lg
              hover:bg-accent-purple/20 transition-smooth disabled:opacity-50 text-sm"
          >
            {searching ? "..." : "Buscar"}
          </button>
        </div>
      </div>

      {searchError && (
        <p className="text-xs text-status-warning mb-4">{searchError}</p>
      )}

      {candidates.length > 0 && (
        <div className="bg-bg-secondary border border-white/5 rounded-xl p-3 mb-4">
          <div className="flex items-center gap-2 mb-3 text-sm text-text-secondary">
            <Sparkles size={14} className="text-accent-purple" />
            Escolha a obra certa para montar o contexto com AniList, Webnovel e Fandom
          </div>
          <div className="space-y-2">
            {candidates.map((candidate) => (
              <button
                key={`${candidate.source}-${candidate.id}`}
                onClick={() => handleUseCandidate(candidate)}
                disabled={loadingCandidateId !== null}
                className="w-full text-left rounded-lg border border-white/5 bg-bg-tertiary hover:border-accent-purple/30
                  px-3 py-2.5 transition-smooth disabled:opacity-60"
              >
                <div className="flex items-start justify-between gap-3 mb-1">
                  <div>
                    <p className="text-sm font-medium text-text-primary">{candidate.title}</p>
                    <p className="text-[11px] uppercase tracking-wide text-text-secondary/70">
                      {candidate.source === "anilist"
                        ? "AniList"
                        : candidate.source === "webnovel"
                        ? "Webnovel"
                        : "Fandom"}
                    </p>
                  </div>
                  <span className="px-2 py-0.5 text-[11px] rounded bg-accent-purple/10 text-accent-purple">
                    {candidate.source}
                  </span>
                </div>
                {candidate.synopsis && (
                  <p className="text-xs text-text-secondary line-clamp-2 mb-1.5">
                    {candidate.synopsis}
                  </p>
                )}
                <div className="flex items-center justify-between text-[11px] text-text-secondary/70">
                  <span>Score {Math.round(candidate.score)}</span>
                  <span>{loadingCandidateId === candidate.id ? "Carregando..." : "Usar esta obra"}</span>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {project.contexto.sinopse && (
        <div className="bg-bg-secondary border border-white/5 rounded-xl p-4 mb-4">
          <div className="flex items-center gap-2 mb-2">
            <BookOpen size={14} className="text-accent-purple" />
            <span className="text-sm font-medium">{project.obra}</span>
          </div>
          <p className="text-xs text-text-secondary line-clamp-3 mb-2">
            {project.contexto.sinopse}
          </p>
          <div className="flex flex-wrap gap-1.5">
            {project.contexto.genero.map((g) => (
              <span
                key={g}
                className="px-2 py-0.5 text-xs bg-accent-purple/10 text-accent-purple rounded"
              >
                {g}
              </span>
            ))}
          </div>
          {project.contexto.personagens.length > 0 && (
            <p className="text-xs text-text-secondary mt-2">
              Personagens: {project.contexto.personagens.slice(0, 8).join(", ")}
            </p>
          )}
          {project.contexto.fontes_usadas.length > 0 && (
            <div className="mt-3 pt-3 border-t border-white/5">
              <p className="text-[11px] text-text-secondary/70 mb-1.5 flex items-center gap-1.5">
                <Globe size={12} />
                Fontes de contexto
              </p>
              <div className="flex flex-wrap gap-1.5">
                {project.contexto.fontes_usadas.slice(0, 6).map((fonte) => (
                  <span
                    key={`${fonte.fonte}-${fonte.url}`}
                    className="px-2 py-0.5 text-[11px] bg-white/5 text-text-secondary rounded"
                  >
                    {fonte.fonte}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      <div className="mb-4 rounded-2xl border border-white/5 bg-bg-secondary/70 p-4">
        <div className="mb-4 flex items-start gap-3">
          <div className="rounded-xl bg-accent-purple/10 p-2 text-accent-purple">
            <Globe size={16} />
          </div>
          <div>
            <p className="text-sm font-medium text-text-primary">Idiomas da traducao</p>
            <p className="text-xs text-text-secondary mt-1">
              Lista dinamica com todos os idiomas que o Google Translate expor no momento.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <LanguageSelectField
            label="Idioma de origem"
            value={sourceValue}
            languages={supportedLanguages}
            fallbackCode="en"
            sourceMode
            helperText="Use a busca para localizar rapido por nome ou codigo."
            onChange={(code) => updateProject({ idioma_origem: code })}
          />
          <LanguageSelectField
            label="Idioma de destino"
            value={targetValue}
            languages={supportedLanguages}
            fallbackCode="pt"
            helperText="Voce pode traduzir para qualquer idioma suportado pela API."
            onChange={(code) => updateProject({ idioma_destino: code })}
          />
        </div>
      </div>

      {batchSources.length > 1 ? (
        <div className="mb-6">
          <label className="text-sm text-text-secondary mb-2 block">
            Capitulos selecionados ({batchSources.length})
          </label>
          <div className="bg-bg-secondary border border-white/5 rounded-xl overflow-hidden">
            <div className="max-h-48 overflow-y-auto">
              {batchSources.map((path, index) => (
                <div key={path} className="flex items-center justify-between px-4 py-2.5 border-b border-white/5 last:border-0 group">
                  <div className="flex items-center gap-3 min-w-0">
                    <span className="text-[10px] font-mono text-text-secondary w-4">
                      {index + 1}
                    </span>
                    <span className="text-sm text-text-primary truncate">
                      {path.split(/[/\\]/).pop()}
                    </span>
                  </div>
                  <button
                    onClick={() => setBatchSources(batchSources.filter((p) => p !== path))}
                    title="Remover capítulo"
                    className="p-1 text-text-secondary/30 hover:text-status-error opacity-0 group-hover:opacity-100 transition-smooth"
                  >
                    <X size={14} />
                  </button>
                </div>
              ))}
            </div>
            <div className="px-4 py-2 bg-white/5 flex items-center justify-between">
              <span className="text-[11px] text-text-secondary italic">
                Capitulo inicial: {project.capitulo} (sera incrementado automaticamente)
              </span>
            </div>
          </div>
        </div>
      ) : (
        <div className="mb-4">
          <label className="text-sm text-text-secondary mb-2 block">Capitulo</label>
          <input
            type="number"
            value={project.capitulo}
            title="Número do capítulo"
            onChange={(e) => updateProject({ capitulo: parseInt(e.target.value) || 1 })}
            min={1}
            className="w-24 px-4 py-2.5 bg-bg-secondary border border-white/10 rounded-lg
              text-text-primary focus:border-accent-purple/50 focus:outline-none transition-smooth"
          />
        </div>
      )}

      <div className="mb-6">
        <label className="text-sm text-text-secondary mb-2 block">Tempo estimado</label>
        <div className="bg-bg-secondary border border-white/5 rounded-xl p-4">
          {estimate ? (
            <>
              <div className="flex items-start justify-between gap-3 mb-3">
                <div>
                  <p className="text-lg font-semibold text-text-primary">
                    ~{formatDuration(estimate.total_seconds)}
                  </p>
                  <p className="text-xs text-text-secondary mt-0.5">
                    Estimativa inicial baseada no hardware detectado neste PC.
                  </p>
                </div>
                <div className="px-2.5 py-1 rounded-lg bg-accent-cyan/10 text-accent-cyan text-xs">
                  {formatTierLabel(estimate.performance_tier)}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div className="rounded-lg bg-bg-tertiary/60 border border-white/5 px-3 py-2">
                  <p className="text-[11px] uppercase tracking-wide text-text-secondary/70">Ritmo base</p>
                  <p className="text-sm text-text-primary mt-0.5">~{estimate.seconds_per_page.toFixed(1)}s / pagina</p>
                </div>
                <div className="rounded-lg bg-bg-tertiary/60 border border-white/5 px-3 py-2">
                  <p className="text-[11px] uppercase tracking-wide text-text-secondary/70">Aquecimento</p>
                  <p className="text-sm text-text-primary mt-0.5">~{formatDuration(estimate.startup_seconds)}</p>
                </div>
              </div>

              <div className="mt-3 pt-3 border-t border-white/5 flex flex-col gap-1">
                <p className="text-xs text-text-secondary flex items-center gap-1.5">
                  <Cpu size={12} />
                  {estimate.hardware_summary}
                </p>
                <p className="text-xs text-text-secondary/70 flex items-center gap-1.5">
                  <Gauge size={12} />
                  ETA se ajusta dinamicamente durante o processamento.
                </p>
              </div>
            </>
          ) : (
            <div>
              <p className="text-sm text-text-primary">Detectando hardware do PC...</p>
              <p className="text-xs text-text-secondary mt-0.5">
                Assim que CPU, RAM e aceleracao local forem identificadas, a previsao aparece aqui.
              </p>
            </div>
          )}
        </div>
      </div>

      <div className="mb-6">
        <label className="text-sm text-text-secondary mb-2 block">
          Glossario (termos consistentes)
        </label>
        <div className="bg-bg-secondary border border-white/5 rounded-xl p-3">
          {Object.entries(project.contexto.glossario).map(([key, value]) => (
            <div key={key} className="flex items-center gap-2 py-1.5 border-b border-white/5 last:border-0">
              <span className="text-sm text-text-primary flex-1">{key}</span>
              <span className="text-xs text-text-secondary">=</span>
              <span className="text-sm text-accent-purple flex-1">{value}</span>
              <button
                onClick={() => removeGlossaryTerm(key)}
                title="Remover termo"
                className="p-1 text-text-secondary/40 hover:text-status-error transition-smooth"
              >
                <X size={14} />
              </button>
            </div>
          ))}

          <div className="flex items-center gap-2 pt-2">
            <input
              type="text"
              value={newTerm.key}
              onChange={(e) => setNewTerm({ ...newTerm, key: e.target.value })}
              placeholder="Termo origem"
              className="flex-1 px-2.5 py-1.5 bg-bg-tertiary border border-white/5 rounded text-sm
                text-text-primary placeholder:text-text-secondary/40 focus:outline-none focus:border-accent-purple/30"
            />
            <span className="text-xs text-text-secondary">=</span>
            <input
              type="text"
              value={newTerm.value}
              onChange={(e) => setNewTerm({ ...newTerm, value: e.target.value })}
              placeholder="Traducao"
              onKeyDown={(e) => e.key === "Enter" && addGlossaryTerm()}
              className="flex-1 px-2.5 py-1.5 bg-bg-tertiary border border-white/5 rounded text-sm
                text-text-primary placeholder:text-text-secondary/40 focus:outline-none focus:border-accent-purple/30"
            />
            <button
              onClick={addGlossaryTerm}
              title="Adicionar termo ao glossário"
              className="p-1.5 text-accent-purple hover:bg-accent-purple/10 rounded transition-smooth"
            >
              <Plus size={16} />
            </button>
          </div>
        </div>
      </div>

      <div className="border-t border-white/5 pt-5">
        <div className="flex items-center justify-between mb-3">
          <div className="text-sm text-text-secondary">
            <span className="text-text-primary font-medium">{totalPages}</span> paginas detectadas
          </div>
          <div className="text-sm">
            {hasEnoughCredits ? (
              <span className="text-status-success">Creditos suficientes</span>
            ) : (
              <span className="text-status-error">Creditos insuficientes</span>
            )}
          </div>
        </div>

        <button
          onClick={handleStart}
          disabled={!hasEnoughCredits}
          className="w-full py-3.5 bg-accent-purple hover:bg-accent-purple-dark text-white
            font-medium rounded-xl transition-smooth disabled:opacity-40 disabled:cursor-not-allowed
            flex items-center justify-center gap-2 text-base"
        >
          <Rocket size={20} />
          {project.mode === "manual" ? "Iniciar projeto manual" : "Traduzir"}
        </button>
      </div>
    </div>
  );
}

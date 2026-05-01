import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Search, Plus, X, Rocket, ArrowLeft, BookOpen, Sparkles, Globe, Gauge, Cpu, Star } from "lucide-react";
import { LanguageSelectField } from "../components/ui";
import { useAppStore } from "../lib/stores/appStore";
import { getFavoriteWorkSuggestions } from "../lib/favoriteWorks";
import {
  enrichWorkContext,
  loadOrCreateWorkContext,
  loadGlossary,
  loadSupportedLanguages,
  removeGlossaryEntry,
  searchWork,
  upsertGlossaryEntry,
  type WorkSearchCandidate,
  type WorkContextProfile,
} from "../lib/tauri";
import {
  contextQualityLabel,
  glossaryEntriesCount,
  riskLabel,
  shouldWarnWorkContext,
  summarizeWorkContext,
} from "../lib/workContextProfile";
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
    favoriteWorks,
    addFavoriteWork,
    removeFavoriteWork,
  } = useAppStore();

  const [obraSearch, setObraSearch] = useState(project?.obra || "");
  const [searching, setSearching] = useState(false);
  const [newTerm, setNewTerm] = useState({ key: "", value: "" });
  const [candidates, setCandidates] = useState<WorkSearchCandidate[]>([]);
  const [searchError, setSearchError] = useState("");
  const [loadingCandidateId, setLoadingCandidateId] = useState<string | null>(null);
  const [supportedLanguages, setSupportedLanguages] = useState(getLanguageOptions(null));
  const [showContextWarning, setShowContextWarning] = useState(false);

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

  async function handleSearchObra(searchOverride?: string) {
    const query = (searchOverride ?? obraSearch).trim();
    if (!query) return;
    setSearching(true);
    setSearchError("");
    try {
      const result = await searchWork(query);
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
      const glossaryCount = glossaryEntriesCount(project?.contexto ?? {
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
      });
      const workSummary = summarizeWorkContext(
        {
          work_id: result.work_id,
          title: result.title,
          context_quality: result.context_quality,
        },
        glossaryCount,
      );
      const savedGlossary = await loadGlossary(result.work_id);
      const savedGlossaryCount = savedGlossary.entries.length || workSummary.glossary_entries_count;
      const summaryWithGlossary = summarizeWorkContext(
        {
          work_id: result.work_id,
          title: result.title,
          context_quality: result.context_quality,
        },
        savedGlossaryCount,
      );
      const glossaryFromFile = Object.fromEntries(
        savedGlossary.entries
          .filter((entry) => entry.status === "reviewed")
          .map((entry) => [entry.source, entry.target]),
      );
      updateProject({
        obra: result.title,
        work_context: {
          ...summaryWithGlossary,
          glossary_loaded: savedGlossary.entries.length > 0 || workSummary.glossary_loaded,
          glossary_entries_count: savedGlossaryCount,
        },
        contexto: {
          sinopse: result.synopsis,
          genero: result.genres,
          personagens: result.characters,
          glossario: { ...(project?.contexto.glossario || {}), ...glossaryFromFile },
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
      addFavoriteWork(result.title);
      setCandidates([]);
      setShowContextWarning(shouldWarnWorkContext(workSummary));
    } catch (err) {
      console.error("Erro ao enriquecer contexto:", err);
      setSearchError("Nao foi possivel carregar o contexto dessa obra.");
    } finally {
      setLoadingCandidateId(null);
    }
  }

  async function handleUseFavorite(title: string) {
    setObraSearch(title);
    updateProject({ obra: title });
    await handleSearchObra(title);
  }

  function addGlossaryTerm() {
    if (!newTerm.key || !newTerm.value || !project) return;
    const updated = {
      ...project.contexto.glossario,
      [newTerm.key]: newTerm.value,
    };
    const count = Object.keys(updated).length;
    const activeWorkId = project.work_context?.work_id;
    if (activeWorkId) {
      void upsertGlossaryEntry(activeWorkId, {
        id: `term_${newTerm.key.trim().toLocaleLowerCase("pt-BR").replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "") || crypto.randomUUID()}`,
        source: newTerm.key.trim(),
        target: newTerm.value.trim(),
        type: "generic_term",
        case_sensitive: false,
        protect: false,
        aliases: [],
        forbidden: [],
        confidence: 1,
        status: "reviewed",
        notes: "",
        context_rule: "",
      }).catch((err) => {
        console.error("Erro ao salvar termo no glossario:", err);
      });
    }
    updateProject({
      contexto: { ...project.contexto, glossario: updated },
      work_context: project.work_context
        ? {
            ...project.work_context,
            glossary_loaded: count > 0,
            glossary_entries_count: count,
            risk_level: summarizeWorkContext(
              {
                work_id: project.work_context.work_id,
                title: project.work_context.title,
                context_quality: project.work_context.context_loaded ? "partial" : "empty",
              },
              count,
              project.work_context.user_ignored_warning,
            ).risk_level,
          }
        : project.work_context,
    });
    setNewTerm({ key: "", value: "" });
  }

  function removeGlossaryTerm(key: string) {
    if (!project) return;
    const { [key]: _, ...rest } = project.contexto.glossario;
    const count = Object.keys(rest).length;
    if (project.work_context?.work_id) {
      const entryId = `term_${key.trim().toLocaleLowerCase("pt-BR").replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "")}`;
      void removeGlossaryEntry(project.work_context.work_id, entryId).catch((err) => {
        console.error("Erro ao remover termo do glossario:", err);
      });
    }
    updateProject({
      contexto: { ...project.contexto, glossario: rest },
      work_context: project.work_context
        ? {
            ...project.work_context,
            glossary_loaded: count > 0,
            glossary_entries_count: count,
            risk_level: summarizeWorkContext(
              {
                work_id: project.work_context.work_id,
                title: project.work_context.title,
                context_quality: project.work_context.context_loaded ? "partial" : "empty",
              },
              count,
              project.work_context.user_ignored_warning,
            ).risk_level,
          }
        : project.work_context,
    });
  }

  async function ensureWorkContext(userIgnoredWarning = false) {
    if (!project) return;
    const title = obraSearch.trim() || project.obra.trim();
    if (!title) return project.work_context ?? null;
    if (project.work_context?.selected) {
      const count = glossaryEntriesCount(project.contexto);
      const nextSummary = {
        ...project.work_context,
        glossary_loaded: count > 0,
        glossary_entries_count: count,
        user_ignored_warning: project.work_context.user_ignored_warning || userIgnoredWarning,
      };
      updateProject({ work_context: nextSummary });
      return nextSummary;
    }

    const profile: WorkContextProfile = await loadOrCreateWorkContext({
      title,
      source_language: project.idioma_origem,
      target_language: project.idioma_destino,
      synopsis: project.contexto.sinopse,
      genre: project.contexto.genero,
      characters: project.contexto.personagens,
      terms: project.contexto.termos,
      factions: project.contexto.faccoes,
    });
    const summary = summarizeWorkContext(
      profile,
      glossaryEntriesCount(project.contexto),
      userIgnoredWarning,
    );
    updateProject({ obra: profile.title, work_context: summary });
    return summary;
  }

  async function handleStart(userIgnoredWarning = false) {
    if (!project) return;
    const summary = await ensureWorkContext(userIgnoredWarning);
    if (summary && shouldWarnWorkContext(summary) && !summary.user_ignored_warning) {
      setShowContextWarning(true);
      return;
    }
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
  const favoriteSuggestions = getFavoriteWorkSuggestions(favoriteWorks, obraSearch, 5);
  const normalizedCurrentWork = obraSearch.trim().toLocaleLowerCase("pt-BR");
  const isFavoriteWork = favoriteWorks.some((item) => item.toLocaleLowerCase("pt-BR") === normalizedCurrentWork);
  const contextSummary = project.work_context;

  return (
    <div className="p-8 max-w-2xl mx-auto animate-fade-in">
      <button
        onClick={() => navigate("/")}
        className="flex items-center gap-2 text-sm text-text-muted hover:text-text-primary mb-6 transition-smooth"
      >
        <ArrowLeft size={16} strokeWidth={1.75} />
        Voltar
      </button>

      <div className="flex items-center justify-between mb-8">
        <h2 className="text-xl font-bold tracking-tight text-text-primary">Configurar projeto</h2>
        {project.mode === "manual" && (
          <span className="px-2.5 py-1 rounded-pill bg-accent-violet/8 border border-accent-violet/15 text-accent-violet text-[10px] font-semibold uppercase tracking-widest flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-accent-violet animate-pulse" />
            Modo Manual
          </span>
        )}
      </div>

      {/* Obra search */}
      <div className="mb-5">
        <label className="text-sm text-text-secondary mb-2 block">Nome da obra <span className="text-text-muted">(opcional)</span></label>
        <div className="flex gap-2">
          <div className="flex-1 relative">
            <input
              data-testid="project-name-input"
              type="text"
              value={obraSearch}
              onChange={(e) => {
                setObraSearch(e.target.value);
                updateProject({ obra: e.target.value });
              }}
              onKeyDown={(e) => e.key === "Enter" && handleSearchObra()}
              placeholder="Ex: Solo Leveling, One Piece..."
              className="w-full px-4 py-2.5 bg-bg-tertiary border border-border rounded-xl
                text-text-primary placeholder:text-text-muted focus:border-brand/40
                focus:shadow-[0_0_0_3px_rgba(108,92,231,0.08)]
                focus:outline-none transition-smooth text-sm"
            />
            <Search size={15} className="absolute right-3 top-3 text-text-muted" />
          </div>
          <button
            type="button"
            onClick={() => {
              if (!obraSearch.trim()) return;
              if (isFavoriteWork) {
                removeFavoriteWork(obraSearch);
                return;
              }
              addFavoriteWork(obraSearch);
            }}
            disabled={!obraSearch.trim()}
            className={`px-3 py-2.5 rounded-xl border transition-smooth disabled:opacity-40 ${
              isFavoriteWork
                ? "border-accent-amber/25 bg-accent-amber/8 text-accent-amber"
                : "border-border bg-bg-tertiary text-text-muted hover:text-text-primary hover:border-border-strong"
            }`}
            title={isFavoriteWork ? "Remover das favoritas" : "Adicionar às favoritas"}
          >
            <Star size={16} className={isFavoriteWork ? "fill-current" : ""} strokeWidth={1.75} />
          </button>
          <button
            data-testid="project-search-button"
            onClick={() => {
              void handleSearchObra();
            }}
            disabled={searching}
            className="px-4 py-2.5 bg-brand/10 text-brand-300 rounded-xl
              hover:bg-brand/15 transition-smooth disabled:opacity-40 text-sm font-medium"
          >
            {searching ? "..." : "Buscar"}
          </button>
        </div>
        {(favoriteSuggestions.length > 0 || favoriteWorks.length > 0) && (
          <div className="mt-3 space-y-2">
            {favoriteSuggestions.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {favoriteSuggestions.map((title) => (
                  <button
                    key={`fav-${title}`}
                    type="button"
                    onClick={() => handleUseFavorite(title)}
                    className="px-3 py-1.5 rounded-pill border border-brand/15 bg-brand/5 text-xs text-text-primary hover:border-brand/30 transition-smooth"
                  >
                    {title}
                  </button>
                ))}
              </div>
            )}
            {favoriteWorks.length > 0 && (
              <p className="text-[11px] text-text-muted">
                Favoritas salvas: {favoriteWorks.slice(0, 6).join(", ")}
              </p>
            )}
          </div>
        )}
      </div>

      {searchError && (
        <p className="text-xs text-status-warning mb-4">{searchError}</p>
      )}

      {/* Candidates */}
      {candidates.length > 0 && (
        <div className="bg-bg-secondary border border-border rounded-xl p-3 mb-5 shadow-card">
          <div className="flex items-center gap-2 mb-3 text-sm text-text-secondary">
            <Sparkles size={14} className="text-brand-300" />
            Escolha a obra certa para montar o contexto
          </div>
          <div className="space-y-2">
            {candidates.map((candidate) => (
              <button
                data-testid="work-result-item"
                key={`${candidate.source}-${candidate.id}`}
                onClick={() => handleUseCandidate(candidate)}
                disabled={loadingCandidateId !== null}
                className="w-full text-left rounded-xl border border-border bg-bg-tertiary hover:border-brand/25
                  px-3 py-2.5 transition-smooth disabled:opacity-50"
              >
                <div className="flex items-start justify-between gap-3 mb-1">
                  <div>
                    <p className="text-sm font-medium text-text-primary">{candidate.title}</p>
                    <p className="text-[11px] uppercase tracking-wide text-text-muted">
                      {candidate.source === "anilist"
                        ? "AniList"
                        : candidate.source === "webnovel"
                        ? "Webnovel"
                        : "Fandom"}
                    </p>
                  </div>
                  <span className="px-2 py-0.5 text-[11px] rounded-md bg-brand/8 text-brand-300 font-medium">
                    {candidate.source}
                  </span>
                </div>
                {candidate.synopsis && (
                  <p className="text-xs text-text-secondary line-clamp-2 mb-1.5">
                    {candidate.synopsis}
                  </p>
                )}
                <div className="flex items-center justify-between text-[11px] text-text-muted">
                  <span>Score {Math.round(candidate.score)}</span>
                  <span className="text-brand-300">{loadingCandidateId === candidate.id ? "Carregando..." : "Usar esta obra"}</span>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {contextSummary?.selected && (
        <div
          data-testid="work-context-summary"
          className="bg-bg-secondary border border-border rounded-xl p-4 mb-5 shadow-card"
        >
          <div className="flex items-center gap-2 mb-3">
            <BookOpen size={14} className="text-brand-300" />
            <span className="text-sm font-medium text-text-primary">Obra: {contextSummary.title}</span>
          </div>
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
            <div className="rounded-lg border border-border bg-bg-tertiary/70 px-3 py-2">
              <p className="text-[11px] text-text-muted">Contexto</p>
              <p className="text-sm text-text-primary">{contextQualityLabel(contextSummary.context_loaded ? "partial" : "empty")}</p>
            </div>
            <div className="rounded-lg border border-border bg-bg-tertiary/70 px-3 py-2">
              <p className="text-[11px] text-text-muted">Glossario</p>
              <p className="text-sm text-text-primary">{contextSummary.glossary_entries_count} termos</p>
            </div>
            <div className="rounded-lg border border-border bg-bg-tertiary/70 px-3 py-2">
              <p className="text-[11px] text-text-muted">Risco</p>
              <p className="text-sm text-text-primary">{riskLabel(contextSummary.risk_level)}</p>
            </div>
          </div>
        </div>
      )}

      {/* Context card */}
      {project.contexto.sinopse && (
        <div
          data-testid="context-status-card"
          className="bg-bg-secondary border border-border rounded-xl p-4 mb-5 shadow-card"
        >
          <div className="flex items-center gap-2 mb-2">
            <BookOpen size={14} className="text-brand-300" />
            <span className="text-sm font-medium text-text-primary">{project.obra}</span>
          </div>
          <p className="text-xs text-text-secondary line-clamp-3 mb-2">
            {project.contexto.sinopse}
          </p>
          <div className="flex flex-wrap gap-1.5">
            {project.contexto.genero.map((g) => (
              <span
                key={g}
                className="px-2 py-0.5 text-xs bg-brand/8 text-brand-300 rounded-md"
              >
                {g}
              </span>
            ))}
          </div>
          {project.contexto.personagens.length > 0 && (
            <p className="text-xs text-text-muted mt-2">
              Personagens: {project.contexto.personagens.slice(0, 8).join(", ")}
            </p>
          )}
          {project.contexto.fontes_usadas.length > 0 && (
            <div className="mt-3 pt-3 border-t border-border">
              <p className="text-[11px] text-text-muted mb-1.5 flex items-center gap-1.5">
                <Globe size={12} />
                Fontes de contexto
              </p>
              <div className="flex flex-wrap gap-1.5">
                {project.contexto.fontes_usadas.slice(0, 6).map((fonte) => (
                  <span
                    key={`${fonte.fonte}-${fonte.url}`}
                    className="px-2 py-0.5 text-[11px] bg-white/[0.03] text-text-muted rounded-md"
                  >
                    {fonte.fonte}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Language selector */}
      <div className="mb-5 rounded-xl border border-border bg-bg-secondary/70 p-4 shadow-card">
        <div className="mb-4 flex items-start gap-3">
          <div className="rounded-xl bg-brand/8 p-2.5 text-brand-300">
            <Globe size={16} strokeWidth={1.75} />
          </div>
          <div>
            <p className="text-sm font-medium text-text-primary">Idiomas da traducao</p>
            <p className="text-xs text-text-muted mt-1">
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

      {/* Batch or chapter */}
      {batchSources.length > 1 ? (
        <div className="mb-6">
          <label className="text-sm text-text-secondary mb-2 block">
            Capitulos selecionados ({batchSources.length})
          </label>
          <div className="bg-bg-secondary border border-border rounded-xl overflow-hidden shadow-card">
            <div className="max-h-48 overflow-y-auto">
              {batchSources.map((path, index) => (
                <div key={path} className="flex items-center justify-between px-4 py-2.5 border-b border-border last:border-0 group">
                  <div className="flex items-center gap-3 min-w-0">
                    <span className="text-[10px] font-mono text-text-muted w-4 tabular">
                      {index + 1}
                    </span>
                    <span className="text-sm text-text-primary truncate">
                      {path.split(/[/\\]/).pop()}
                    </span>
                  </div>
                  <button
                    onClick={() => setBatchSources(batchSources.filter((p) => p !== path))}
                    title="Remover capítulo"
                    className="p-1 text-text-muted hover:text-status-error opacity-0 group-hover:opacity-100 transition-smooth"
                  >
                    <X size={14} />
                  </button>
                </div>
              ))}
            </div>
            <div className="px-4 py-2 bg-white/[0.02] flex items-center justify-between border-t border-border">
              <span className="text-[11px] text-text-muted italic">
                Capitulo inicial: {project.capitulo} (sera incrementado automaticamente)
              </span>
            </div>
          </div>
        </div>
      ) : (
        <div className="mb-5">
          <label className="text-sm text-text-secondary mb-2 block">Capitulo</label>
          <input
            type="number"
            value={project.capitulo}
            title="Número do capítulo"
            onChange={(e) => updateProject({ capitulo: parseInt(e.target.value) || 1 })}
            min={1}
            className="w-24 px-4 py-2.5 bg-bg-tertiary border border-border rounded-xl
              text-text-primary focus:border-brand/40 focus:shadow-[0_0_0_3px_rgba(108,92,231,0.08)]
              focus:outline-none transition-smooth text-sm tabular"
          />
        </div>
      )}

      {/* Time estimate */}
      <div className="mb-6">
        <label className="text-sm text-text-secondary mb-2 block">Tempo estimado</label>
        <div className="bg-bg-secondary border border-border rounded-xl p-4 shadow-card">
          {estimate ? (
            <>
              <div className="flex items-start justify-between gap-3 mb-3">
                <div>
                  <p className="text-lg font-bold text-text-primary tracking-tight">
                    ~{formatDuration(estimate.total_seconds)}
                  </p>
                  <p className="text-xs text-text-muted mt-0.5">
                    Estimativa inicial baseada no hardware detectado.
                  </p>
                </div>
                <div className="px-2.5 py-1 rounded-lg bg-accent-cyan/8 text-accent-cyan text-xs font-medium border border-accent-cyan/15">
                  {formatTierLabel(estimate.performance_tier)}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div className="rounded-xl bg-bg-tertiary/60 border border-border px-3 py-2">
                  <p className="text-[11px] uppercase tracking-wide text-text-muted">Ritmo base</p>
                  <p className="text-sm text-text-primary mt-0.5 tabular">~{estimate.seconds_per_page.toFixed(1)}s / pagina</p>
                </div>
                <div className="rounded-xl bg-bg-tertiary/60 border border-border px-3 py-2">
                  <p className="text-[11px] uppercase tracking-wide text-text-muted">Aquecimento</p>
                  <p className="text-sm text-text-primary mt-0.5 tabular">~{formatDuration(estimate.startup_seconds)}</p>
                </div>
              </div>

              <div className="mt-3 pt-3 border-t border-border flex flex-col gap-1">
                <p className="text-xs text-text-muted flex items-center gap-1.5">
                  <Cpu size={12} strokeWidth={1.75} />
                  {estimate.hardware_summary}
                </p>
                <p className="text-xs text-text-muted/70 flex items-center gap-1.5">
                  <Gauge size={12} strokeWidth={1.75} />
                  ETA se ajusta dinamicamente durante o processamento.
                </p>
              </div>
            </>
          ) : (
            <div>
              <p className="text-sm text-text-primary">Detectando hardware do PC...</p>
              <p className="text-xs text-text-muted mt-0.5">
                Assim que CPU, RAM e aceleracao local forem identificadas, a previsao aparece aqui.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Glossary */}
      <div className="mb-6">
        <label className="text-sm text-text-secondary mb-2 block">
          Glossario (termos consistentes)
        </label>
        <div data-testid="glossary-editor" className="bg-bg-secondary border border-border rounded-xl p-3 shadow-card">
          {Object.entries(project.contexto.glossario).map(([key, value]) => (
            <div key={key} className="flex items-center gap-2 py-1.5 border-b border-border last:border-0">
              <span className="text-sm text-text-primary flex-1">{key}</span>
              <span className="text-xs text-text-muted">=</span>
              <span className="text-sm text-brand-300 flex-1">{value}</span>
              <button
                onClick={() => removeGlossaryTerm(key)}
                title="Remover termo"
                className="p-1 text-text-muted hover:text-status-error transition-smooth"
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
              className="flex-1 px-2.5 py-1.5 bg-bg-tertiary border border-border rounded-lg text-sm
                text-text-primary placeholder:text-text-muted focus:outline-none focus:border-brand/30"
            />
            <span className="text-xs text-text-muted">=</span>
            <input
              type="text"
              value={newTerm.value}
              onChange={(e) => setNewTerm({ ...newTerm, value: e.target.value })}
              placeholder="Traducao"
              onKeyDown={(e) => e.key === "Enter" && addGlossaryTerm()}
              className="flex-1 px-2.5 py-1.5 bg-bg-tertiary border border-border rounded-lg text-sm
                text-text-primary placeholder:text-text-muted focus:outline-none focus:border-brand/30"
            />
            <button
              data-testid="glossary-add-entry-button"
              onClick={addGlossaryTerm}
              title="Adicionar termo ao glossário"
              className="p-1.5 text-brand-300 hover:bg-brand/8 rounded-lg transition-smooth"
            >
              <Plus size={16} />
            </button>
          </div>
        </div>
      </div>

      {/* Submit */}
      <div className="border-t border-border pt-5">
        <div className="flex items-center justify-between mb-3">
          <div className="text-sm text-text-secondary">
            <span className="text-text-primary font-medium tabular">{totalPages}</span> paginas detectadas
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
          onClick={() => {
            void handleStart();
          }}
          disabled={!hasEnoughCredits}
          className="w-full py-3.5 bg-gradient-to-b from-brand-400 to-brand text-white
            font-semibold rounded-xl transition-all duration-200 ease-out-expo
            hover:from-brand-300 hover:to-brand-500 hover:shadow-glow-brand
            disabled:opacity-35 disabled:cursor-not-allowed
            flex items-center justify-center gap-2 text-base"
        >
          <Rocket size={20} strokeWidth={2} />
          {project.mode === "manual" ? "Iniciar projeto manual" : "Traduzir"}
        </button>
      </div>

      {showContextWarning && (
        <div
          data-testid="work-context-warning-modal"
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4"
        >
          <div className="w-full max-w-md rounded-xl border border-border bg-bg-secondary p-5 shadow-2xl">
            <h3 className="text-base font-semibold text-text-primary mb-2">
              Esta obra esta sem glossario ativo.
            </h3>
            <p className="text-sm text-text-secondary mb-4">
              A traducao pode errar nomes, cargos, lugares, tecnicas e termos de lore.
            </p>
            <div className="grid grid-cols-1 gap-2">
              <button
                type="button"
                onClick={() => setSearchError("Geracao inicial de glossario entra na Fase 5.")}
                className="rounded-lg border border-border bg-bg-tertiary px-3 py-2 text-sm text-text-primary hover:border-brand/30 transition-smooth"
              >
                Gerar glossario inicial
              </button>
              <button
                type="button"
                onClick={() => setSearchError("Importacao de glossario entra na Fase 5.")}
                className="rounded-lg border border-border bg-bg-tertiary px-3 py-2 text-sm text-text-primary hover:border-brand/30 transition-smooth"
              >
                Importar glossario
              </button>
              <button
                type="button"
                onClick={() => setShowContextWarning(false)}
                className="rounded-lg border border-border bg-bg-tertiary px-3 py-2 text-sm text-text-primary hover:border-brand/30 transition-smooth"
              >
                Editar manualmente
              </button>
              <button
                data-testid="work-context-continue-without-context"
                type="button"
                onClick={() => {
                  setShowContextWarning(false);
                  void handleStart(true);
                }}
                className="rounded-lg bg-brand px-3 py-2 text-sm font-medium text-white hover:bg-brand-400 transition-smooth"
              >
                Continuar sem contexto
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

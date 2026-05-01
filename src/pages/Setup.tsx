import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Search, Plus, X, Rocket, ArrowLeft, BookOpen, Sparkles, Globe, Gauge, Cpu, Star } from "lucide-react";
import { LanguageSelectField } from "../components/ui";
import { useAppStore } from "../lib/stores/appStore";
import { getFavoriteWorkSuggestions } from "../lib/favoriteWorks";
import {
  enrichWorkContext,
  exportLocalMemory,
  importLocalMemory,
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
  applyHighConfidenceInternetCandidates,
  countInternetContextKinds,
  sourceStatusLabel,
  type InternetContextCandidate,
  type InternetContextResult,
} from "../lib/internetContext";
import {
  candidateNeedsReviewWarning,
  candidateToGlossaryEntry,
  filterRejectedCandidates,
  glossaryConflicts,
  manualTermToGlossaryEntry,
  type GlossaryTab,
} from "../lib/glossaryCenter";
import {
  contextQualityLabel,
  emptyWorkContextSummary,
  glossaryEntriesCount,
  riskLevel,
  riskLabel,
  setupWarningKind,
  shouldWarnWorkContext,
  summarizeWorkContext,
  type SetupWorkContextWarning,
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
import {
  createCustomPreset,
  getProjectPreset,
  PROJECT_PRESETS,
  type ProjectPreset,
} from "../lib/projectPresets";
import { buildWorkMemorySummary } from "../lib/workMemory";
import { buildOnboardingChecklist } from "../lib/onboarding";

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
  const [contextWarningKind, setContextWarningKind] = useState<SetupWorkContextWarning>("empty_glossary");
  const [internetContextResult, setInternetContextResult] = useState<InternetContextResult | null>(null);
  const [internetContextAppliedCount, setInternetContextAppliedCount] = useState<number | null>(null);
  const [glossaryTab, setGlossaryTab] = useState<GlossaryTab>("reviewed");
  const [rejectedGlossarySources, setRejectedGlossarySources] = useState<string[]>([]);
  const [customPresets, setCustomPresets] = useState<ProjectPreset[]>([]);
  const [customPresetName, setCustomPresetName] = useState("");
  const [memoryStatus, setMemoryStatus] = useState("");
  const obraInputRef = useRef<HTMLInputElement | null>(null);
  const glossaryEditorRef = useRef<HTMLDivElement | null>(null);

  const totalPages = project?.totalPages ?? 0;
  const initialPresetId =
    typeof (project?.preset as { id?: unknown } | null | undefined)?.id === "string"
      ? String((project?.preset as { id?: string }).id)
      : "manhwa_webtoon_color";
  const selectedPreset = [...PROJECT_PRESETS, ...customPresets].find((preset) => preset.id === initialPresetId)
    ?? getProjectPreset(initialPresetId);
  const estimate = buildPipelineTimeEstimate(systemProfile, totalPages, selectedPreset.quality);

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

  function applyProjectPreset(preset: ProjectPreset) {
    updateProject({
      preset,
      qualidade: preset.quality,
    });
  }

  function createPresetFromCurrent() {
    const custom = createCustomPreset(selectedPreset, customPresetName);
    setCustomPresets((current) => [...current, custom]);
    setCustomPresetName("");
    applyProjectPreset(custom);
  }

  async function handleExportMemory() {
    const payload = await exportLocalMemory();
    updateProject({
      contexto: {
        ...project!.contexto,
        internet_context: {
          ...(typeof project!.contexto.internet_context === "object" && project!.contexto.internet_context !== null
            ? project!.contexto.internet_context
            : {}),
          last_memory_export: payload,
        },
      },
    });
    setMemoryStatus("Memoria exportada.");
  }

  async function handleImportMemory() {
    const payload = {
      works: project?.work_context?.work_id ? [{ work_id: project.work_context.work_id, title: project.work_context.title }] : [],
      translation_memory: Object.entries(project?.contexto.memoria_lexical ?? {}).map(([source_text, target_text]) => ({
        source_text,
        target_text,
      })),
    };
    await importLocalMemory(payload);
    setMemoryStatus("Memoria importada.");
  }

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
      const internetResult: InternetContextResult = {
        title: result.title,
        synopsis: result.synopsis,
        genres: result.genres,
        internet_context_loaded: result.internet_context_loaded ?? true,
        context_quality: result.context_quality,
        source_results: result.source_results ?? result.sources_used.map((source) => ({
          source: source.source,
          status: "found" as const,
          confidence: 0.7,
          title: source.title,
          synopsis: source.snippet,
          url: source.url,
        })),
        glossary_candidates: result.glossary_candidates ?? [
          ...result.characters.map((name) => ({
            kind: "character",
            source: name,
            target: name,
            confidence: 0.88,
            sources: ["context"],
            status: "candidate" as const,
            protect: true,
            aliases: [],
            forbidden: [],
            notes: "",
          })),
          ...result.terms.map((term) => ({
            kind: "term",
            source: term,
            target: term,
            confidence: 0.72,
            sources: ["context"],
            status: "candidate" as const,
            protect: true,
            aliases: [],
            forbidden: [],
            notes: "",
          })),
        ],
      };
      const filteredInternetResult = {
        ...internetResult,
        glossary_candidates: filterRejectedCandidates(internetResult.glossary_candidates, rejectedGlossarySources),
      };
      setInternetContextResult(filteredInternetResult);
      setInternetContextAppliedCount(null);
      setGlossaryTab("online");
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
          internet_context_loaded: filteredInternetResult.internet_context_loaded,
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
          internet_context_loaded: internetResult.internet_context_loaded,
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
          internet_context_loaded: internetResult.internet_context_loaded,
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
          internet_context: filteredInternetResult,
        },
      });
      setObraSearch(result.title);
      addFavoriteWork(result.title);
      setCandidates([]);
      setShowContextWarning(false);
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
      void upsertGlossaryEntry(activeWorkId, manualTermToGlossaryEntry(newTerm.key, newTerm.value)).catch((err) => {
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

  function handleApplyHighConfidenceCandidates() {
    if (!project || !internetContextResult) return;
    const applied = applyHighConfidenceInternetCandidates(
      project.contexto,
      internetContextResult,
      project.contexto.glossario,
    );
    const glossaryCount = Object.keys(applied.contexto.glossario).length;
    updateProject({
      contexto: applied.contexto,
      work_context: project.work_context
        ? {
            ...project.work_context,
            context_loaded: true,
            internet_context_loaded: true,
            glossary_loaded: glossaryCount > 0,
            glossary_entries_count: glossaryCount,
            risk_level: glossaryCount > 0 ? "medium" : project.work_context.risk_level,
          }
        : project.work_context,
    });
    setInternetContextAppliedCount(applied.appliedCount);
  }

  function updateGlossarySummary(glossaryCount: number) {
    if (!project?.work_context) return project?.work_context;
    return {
      ...project.work_context,
      glossary_loaded: glossaryCount > 0,
      glossary_entries_count: glossaryCount,
      risk_level: glossaryCount > 0 ? "medium" as const : project.work_context.risk_level,
    };
  }

  function markInternetCandidate(source: string, status: InternetContextCandidate["status"]) {
    setInternetContextResult((current) =>
      current
        ? {
            ...current,
            glossary_candidates: current.glossary_candidates.map((candidate) =>
              candidate.source === source ? { ...candidate, status } : candidate,
            ),
          }
        : current,
    );
  }

  function confirmGlossaryCandidate(candidate: InternetContextCandidate, forceProtect = false) {
    if (!project) return;
    const entry = candidateToGlossaryEntry(
      {
        ...candidate,
        target: forceProtect ? candidate.source : candidate.target,
        protect: forceProtect || candidate.protect,
      },
      "reviewed",
    );
    const nextGlossary = {
      ...project.contexto.glossario,
      [entry.source]: entry.target,
    };
    const glossaryCount = Object.keys(nextGlossary).length;
    if (project.work_context?.work_id) {
      void upsertGlossaryEntry(project.work_context.work_id, entry).catch((err) => {
        console.error("Erro ao confirmar termo do glossario:", err);
      });
    }
    updateProject({
      contexto: { ...project.contexto, glossario: nextGlossary },
      work_context: updateGlossarySummary(glossaryCount),
    });
    markInternetCandidate(candidate.source, "reviewed");
    setGlossaryTab("reviewed");
  }

  function rejectGlossaryCandidate(candidate: InternetContextCandidate) {
    if (!project) return;
    const entry = candidateToGlossaryEntry(candidate, "rejected");
    const nextRejected = Array.from(new Set([...rejectedGlossarySources, candidate.source]));
    setRejectedGlossarySources(nextRejected);
    if (project.work_context?.work_id) {
      void upsertGlossaryEntry(project.work_context.work_id, entry).catch((err) => {
        console.error("Erro ao rejeitar termo do glossario:", err);
      });
    }
    updateProject({
      contexto: {
        ...project.contexto,
        internet_context: {
          ...(typeof project.contexto.internet_context === "object" && project.contexto.internet_context !== null
            ? project.contexto.internet_context
            : {}),
          rejected_glossary_candidates: nextRejected,
        },
      },
    });
    markInternetCandidate(candidate.source, "rejected");
    setGlossaryTab("rejected");
  }

  function editGlossaryCandidate(candidate: InternetContextCandidate) {
    setNewTerm({ key: candidate.source, value: candidate.target });
    setGlossaryTab("reviewed");
    glossaryEditorRef.current?.scrollIntoView({ block: "center", behavior: "smooth" });
  }

  function addForbiddenToGlossaryTerm(source: string, target: string) {
    if (!project?.work_context?.work_id) return;
    const entry = manualTermToGlossaryEntry(source, target);
    void upsertGlossaryEntry(project.work_context.work_id, {
      ...entry,
      forbidden: Array.from(new Set([target, ...entry.forbidden])).filter(Boolean),
      notes: "Forbidden adicionado pela revisao do glossario.",
    }).catch((err) => {
      console.error("Erro ao adicionar forbidden no glossario:", err);
    });
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
    const title = obraSearch.trim();
    if (!title) {
      const summary = emptyWorkContextSummary(glossaryEntriesCount(project.contexto), userIgnoredWarning);
      updateProject({ obra: "", work_context: summary });
      return summary;
    }
    if (project.work_context?.selected) {
      const count = glossaryEntriesCount(project.contexto);
      const recalculated = summarizeWorkContext(
        {
          work_id: project.work_context.work_id,
          title: project.work_context.title || title,
          context_quality: project.work_context.context_loaded ? "partial" : "empty",
          internet_context_loaded: project.work_context.internet_context_loaded ?? false,
        },
        count,
        project.work_context.user_ignored_warning || userIgnoredWarning,
      );
      const nextSummary = {
        ...project.work_context,
        ...recalculated,
        selected: true,
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
    const requestedTitle = obraSearch.trim();
    const summary = await ensureWorkContext(userIgnoredWarning);
    const warningKind = setupWarningKind(summary, requestedTitle);
    if (summary && shouldWarnWorkContext(summary, requestedTitle) && !summary.user_ignored_warning && warningKind) {
      setContextWarningKind(warningKind);
      setShowContextWarning(true);
      return;
    }
    if (batchSources.length > 0) {
      updateProject({ qualidade: selectedPreset.quality, preset: selectedPreset, status: "processing" });
      navigate("/processing");
      return;
    }
    if (!canTranslate(totalPages)) {
      alert("Creditos insuficientes para traduzir este capitulo.");
      return;
    }
    setSetupEstimate(estimate);
    updateProject({ qualidade: selectedPreset.quality, preset: selectedPreset, status: "processing" });
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
  const displayWorkTitle = contextSummary?.selected ? contextSummary.title : obraSearch.trim() || "Nenhuma obra";
  const displayGlossaryCount = contextSummary?.glossary_entries_count ?? glossaryEntriesCount(project.contexto);
  const displayContextQuality = contextSummary?.context_loaded ? "partial" : "empty";
  const displayRiskLevel = contextSummary?.risk_level ?? riskLevel(displayContextQuality, displayGlossaryCount);
  const memoryActive = Object.keys(project.contexto.memoria_lexical ?? {}).length > 0;
  const workMemorySummary = buildWorkMemorySummary(project);
  const onboardingChecklist = buildOnboardingChecklist(project);
  const glossaryCandidates = internetContextResult?.glossary_candidates ?? [];
  const onlineGlossaryCandidates = glossaryCandidates.filter(
    (candidate) => candidate.status !== "reviewed" && candidate.status !== "rejected",
  );
  const rejectedGlossaryCandidates = glossaryCandidates.filter((candidate) => candidate.status === "rejected");
  const detectedChapterTerms = project.contexto.termos.filter((term) => !project.contexto.glossario[term]);
  const conflictCandidates = glossaryConflicts(project.contexto.glossario, glossaryCandidates);
  const glossaryTabs: Array<{ id: GlossaryTab; label: string; count: number }> = [
    { id: "reviewed", label: "Revisados", count: Object.keys(project.contexto.glossario).length },
    { id: "online", label: "Candidatos online", count: onlineGlossaryCandidates.length },
    { id: "detected", label: "Detectados no capitulo", count: detectedChapterTerms.length },
    { id: "rejected", label: "Rejeitados", count: rejectedGlossaryCandidates.length },
    { id: "conflicts", label: "Conflitos", count: conflictCandidates.length },
  ];

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

      <div data-testid="setup-onboarding-checklist" className="mb-5 rounded-xl border border-border bg-bg-secondary p-4 shadow-card">
        <p className="mb-3 text-sm font-medium text-text-primary">Checklist do fluxo</p>
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          {onboardingChecklist.map((step) => (
            <div key={step.id} className="flex items-center gap-2 rounded-lg border border-border bg-bg-tertiary/70 px-3 py-2 text-xs">
              <span className={step.status === "done" ? "text-status-success" : step.status === "warning" ? "text-status-warning" : "text-text-muted"}>
                {step.status === "done" ? "OK" : step.status === "warning" ? "!" : "-"}
              </span>
              <span className="text-text-secondary">{step.label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Obra search */}
      <div className="mb-5">
        <label className="text-sm text-text-secondary mb-2 block">Nome da obra <span className="text-text-muted">(opcional)</span></label>
        <div className="flex gap-2">
          <div className="flex-1 relative">
            <input
              ref={obraInputRef}
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

      <div data-testid="internet-context-panel" className="mb-5 rounded-xl border border-border bg-bg-secondary p-4 shadow-card">
        <div className="mb-3 flex items-start justify-between gap-3">
          <div>
            <p className="text-sm font-medium text-text-primary">Contexto online</p>
            <p className="mt-1 text-xs text-text-muted">Busca fontes publicas sem enviar imagens ou capitulos.</p>
          </div>
          <button
            data-testid="internet-context-search"
            type="button"
            onClick={() => {
              void handleSearchObra();
            }}
            disabled={searching || !obraSearch.trim()}
            className="rounded-lg bg-brand/10 px-3 py-1.5 text-xs font-medium text-brand-300 transition-smooth hover:bg-brand/15 disabled:opacity-40"
          >
            Buscar contexto online
          </button>
        </div>
        <div className="grid grid-cols-2 gap-2 text-[11px] text-text-secondary sm:grid-cols-3">
          {["AniList", "MyAnimeList", "MangaUpdates", "Kitsu", "Shikimori", "Bangumi", "Wikipedia", "Wikidata", "Fandom"].map((source) => (
            <label key={source} className="flex items-center gap-2 rounded-lg border border-border bg-bg-tertiary/60 px-2 py-1.5">
              <input type="checkbox" checked readOnly className="accent-brand" />
              {source}
            </label>
          ))}
          <label className="flex items-center gap-2 rounded-lg border border-border bg-bg-tertiary/60 px-2 py-1.5 text-text-muted">
            <input type="checkbox" readOnly className="accent-brand" />
            Generic Web
          </label>
        </div>
      </div>

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

      <div
        data-testid="work-context-summary"
        className="bg-bg-secondary border border-border rounded-xl p-4 mb-5 shadow-card"
      >
        <div className="flex items-center gap-2 mb-3">
          <BookOpen size={14} className="text-brand-300" />
          <span className="text-sm font-medium text-text-primary">Estado do contexto da obra</span>
        </div>
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-5">
          <div className="rounded-lg border border-border bg-bg-tertiary/70 px-3 py-2">
            <p className="text-[11px] text-text-muted">Obra</p>
            <p className="truncate text-sm text-text-primary">{displayWorkTitle}</p>
          </div>
          <div className="rounded-lg border border-border bg-bg-tertiary/70 px-3 py-2">
            <p className="text-[11px] text-text-muted">Contexto</p>
            <p className="text-sm text-text-primary">{contextQualityLabel(displayContextQuality)}</p>
          </div>
          <div className="rounded-lg border border-border bg-bg-tertiary/70 px-3 py-2">
            <p className="text-[11px] text-text-muted">Glossario</p>
            <p className="text-sm text-text-primary">{displayGlossaryCount} termos</p>
          </div>
          <div className="rounded-lg border border-border bg-bg-tertiary/70 px-3 py-2">
            <p className="text-[11px] text-text-muted">Memoria da obra</p>
            <p className="text-sm text-text-primary">{memoryActive ? "ativa" : "inativa"}</p>
          </div>
          <div className="rounded-lg border border-border bg-bg-tertiary/70 px-3 py-2">
            <p className="text-[11px] text-text-muted">Risco</p>
            <p data-testid="work-context-risk" className="text-sm text-text-primary">{riskLabel(displayRiskLevel)}</p>
          </div>
        </div>
      </div>

      <div data-testid="project-preset-panel" className="mb-5 rounded-xl border border-border bg-bg-secondary p-4 shadow-card">
        <label className="mb-2 block text-sm font-medium text-text-primary" htmlFor="project-preset-select">
          Preset
        </label>
        <select
          id="project-preset-select"
          data-testid="project-preset-select"
          value={selectedPreset.id}
          onChange={(event) => {
            const preset = [...PROJECT_PRESETS, ...customPresets].find((item) => item.id === event.target.value)
              ?? getProjectPreset(event.target.value);
            applyProjectPreset(preset);
          }}
          className="w-full rounded-xl border border-border bg-bg-tertiary px-3 py-2 text-sm text-text-primary outline-none transition-smooth focus:border-brand/40"
        >
          {[...PROJECT_PRESETS, ...customPresets].map((preset) => (
            <option key={preset.id} value={preset.id}>
              {preset.name}
            </option>
          ))}
        </select>
        <p data-testid="project-preset-description" className="mt-2 text-xs text-text-secondary">
          {selectedPreset.description}
        </p>
        <div className="mt-3 grid grid-cols-2 gap-2 text-[11px] text-text-secondary sm:grid-cols-4">
          <span className="rounded-lg border border-border bg-bg-tertiary/70 px-2 py-1.5">OCR: {selectedPreset.settings.ocr_sensitivity}</span>
          <span className="rounded-lg border border-border bg-bg-tertiary/70 px-2 py-1.5">SFX: {selectedPreset.settings.sfx_mode}</span>
          <span className="rounded-lg border border-border bg-bg-tertiary/70 px-2 py-1.5">Inpaint: {selectedPreset.settings.inpaint_mode}</span>
          <span className="rounded-lg border border-border bg-bg-tertiary/70 px-2 py-1.5">QA: {selectedPreset.settings.qa_mode}</span>
        </div>
        <div className="mt-3 flex gap-2">
          <input
            data-testid="custom-preset-name"
            value={customPresetName}
            onChange={(event) => setCustomPresetName(event.target.value)}
            placeholder="Nome do preset customizado"
            className="min-w-0 flex-1 rounded-lg border border-border bg-bg-tertiary px-3 py-2 text-xs text-text-primary outline-none focus:border-brand/40"
          />
          <button
            data-testid="custom-preset-create"
            type="button"
            onClick={createPresetFromCurrent}
            className="rounded-lg bg-brand/10 px-3 py-2 text-xs font-medium text-brand-300 transition-smooth hover:bg-brand/15"
          >
            Criar preset
          </button>
        </div>
      </div>

      <div data-testid="work-memory-panel" className="mb-5 rounded-xl border border-border bg-bg-secondary p-4 shadow-card">
        <div className="mb-3 flex items-start justify-between gap-3">
          <div>
            <p className="text-sm font-medium text-text-primary">Memoria da obra carregada</p>
            <p className="mt-1 text-xs text-text-muted">Sugestoes podem ajudar, mas nao sobrescrevem termos revisados.</p>
          </div>
          <span className={`rounded-full px-2 py-1 text-[11px] ${memoryActive ? "bg-status-success/10 text-status-success" : "bg-bg-tertiary text-text-muted"}`}>
            {memoryActive ? "ativa" : "inativa"}
          </span>
        </div>
        <div className="grid grid-cols-2 gap-2 text-xs sm:grid-cols-3">
          <div className="rounded-lg border border-border bg-bg-tertiary/70 px-3 py-2">
            <p className="text-text-muted">Termos revisados</p>
            <p className="text-sm font-medium text-text-primary">{workMemorySummary.reviewed_terms}</p>
          </div>
          <div className="rounded-lg border border-border bg-bg-tertiary/70 px-3 py-2">
            <p className="text-text-muted">Personagens</p>
            <p className="text-sm font-medium text-text-primary">{workMemorySummary.characters}</p>
          </div>
          <div className="rounded-lg border border-border bg-bg-tertiary/70 px-3 py-2">
            <p className="text-text-muted">Correcoes OCR</p>
            <p className="text-sm font-medium text-text-primary">{workMemorySummary.ocr_corrections}</p>
          </div>
          <div className="rounded-lg border border-border bg-bg-tertiary/70 px-3 py-2">
            <p className="text-text-muted">Capitulos anteriores</p>
            <p className="text-sm font-medium text-text-primary">{workMemorySummary.previous_chapters}</p>
          </div>
          <div className="rounded-lg border border-border bg-bg-tertiary/70 px-3 py-2">
            <p className="text-text-muted">Traducoes anteriores</p>
            <p className="text-sm font-medium text-text-primary">{workMemorySummary.translation_memory}</p>
          </div>
          <div className="rounded-lg border border-border bg-bg-tertiary/70 px-3 py-2">
            <p className="text-text-muted">Decisoes de SFX</p>
            <p className="text-sm font-medium text-text-primary">{workMemorySummary.sfx_decisions}</p>
          </div>
        </div>
        <div className="mt-3 flex gap-2">
          <button data-testid="work-memory-export" type="button" onClick={() => void handleExportMemory()} className="rounded-lg border border-border bg-bg-tertiary px-3 py-2 text-xs text-text-secondary hover:text-text-primary">
            Exportar memoria
          </button>
          <button data-testid="work-memory-import" type="button" onClick={() => void handleImportMemory()} className="rounded-lg border border-border bg-bg-tertiary px-3 py-2 text-xs text-text-secondary hover:text-text-primary">
            Importar memoria
          </button>
          {memoryStatus && <span data-testid="work-memory-status" className="self-center text-xs text-status-success">{memoryStatus}</span>}
        </div>
      </div>

      {internetContextResult && (
        <div data-testid="internet-context-results" className="mb-5 rounded-xl border border-border bg-bg-secondary p-4 shadow-card">
          <div className="mb-3 flex items-center justify-between gap-3">
            <div>
              <p className="text-sm font-medium text-text-primary">Contexto encontrado</p>
              <p className="mt-1 text-xs text-text-muted">
                {countInternetContextKinds(internetContextResult).characters} personagens, {" "}
                {countInternetContextKinds(internetContextResult).placesAndFactions} lugares/faccoes, {" "}
                {countInternetContextKinds(internetContextResult).loreTerms} termos de lore
              </p>
            </div>
            <button
              data-testid="internet-context-apply"
              type="button"
              onClick={handleApplyHighConfidenceCandidates}
              className="rounded-lg bg-brand px-3 py-1.5 text-xs font-medium text-white transition-smooth hover:bg-brand-400"
            >
              Aplicar alta confianca
            </button>
          </div>
          <div className="mb-3 grid grid-cols-1 gap-2 sm:grid-cols-3">
            {internetContextResult.source_results.slice(0, 6).map((source) => (
              <div key={`${source.source}-${source.status}`} className="rounded-lg border border-border bg-bg-tertiary/70 px-3 py-2">
                <p className="text-[11px] text-text-muted">{source.source}</p>
                <p className="text-xs text-text-primary">{sourceStatusLabel(source.status)}</p>
              </div>
            ))}
          </div>
          <div className="flex flex-wrap gap-1.5">
            {internetContextResult.glossary_candidates.slice(0, 8).map((candidate) => (
              <span key={`${candidate.kind}-${candidate.source}`} className="rounded-md bg-brand/8 px-2 py-1 text-[11px] text-brand-300">
                {candidate.source}
              </span>
            ))}
          </div>
          {internetContextAppliedCount !== null && (
            <p data-testid="internet-context-applied" className="mt-3 text-xs text-status-success">
              {internetContextAppliedCount} candidatos aplicados ao glossario.
            </p>
          )}
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
        <div ref={glossaryEditorRef} data-testid="glossary-editor" className="bg-bg-secondary border border-border rounded-xl p-3 shadow-card">
          <div className="mb-3 flex flex-wrap gap-1.5">
            {glossaryTabs.map((tab) => (
              <button
                key={tab.id}
                data-testid={`glossary-tab-${tab.id}`}
                type="button"
                onClick={() => setGlossaryTab(tab.id)}
                className={`rounded-lg border px-2.5 py-1.5 text-[11px] transition-smooth ${
                  glossaryTab === tab.id
                    ? "border-brand/30 bg-brand/10 text-brand-300"
                    : "border-border bg-bg-tertiary text-text-secondary hover:text-text-primary"
                }`}
              >
                {tab.label} ({tab.count})
              </button>
            ))}
          </div>

          {glossaryTab === "reviewed" && (
            <div data-testid="glossary-reviewed-list">
              {Object.entries(project.contexto.glossario).map(([key, value]) => (
                <div data-testid="glossary-reviewed-row" key={key} className="flex items-center gap-2 py-1.5 border-b border-border last:border-0">
                  <span className="text-sm text-text-primary flex-1">{key}</span>
                  <span className="text-xs text-text-muted">=</span>
                  <span className="text-sm text-brand-300 flex-1">{value}</span>
                  <button
                    type="button"
                    onClick={() => addForbiddenToGlossaryTerm(key, value)}
                    className="rounded-md border border-border px-2 py-1 text-[11px] text-text-secondary hover:border-status-warning/40 hover:text-status-warning"
                  >
                    Adicionar forbidden
                  </button>
                  <button
                    onClick={() => removeGlossaryTerm(key)}
                    title="Remover termo"
                    className="p-1 text-text-muted hover:text-status-error transition-smooth"
                  >
                    <X size={14} />
                  </button>
                </div>
              ))}
              {Object.keys(project.contexto.glossario).length === 0 && (
                <p className="py-2 text-xs text-text-muted">Nenhum termo revisado ainda.</p>
              )}
            </div>
          )}

          {glossaryTab === "online" && (
            <div data-testid="glossary-online-list" className="space-y-2">
              {onlineGlossaryCandidates.map((candidate) => (
                <div data-testid="glossary-candidate-row" key={`${candidate.kind}-${candidate.source}`} className="rounded-lg border border-border bg-bg-tertiary/70 p-2">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="text-sm text-text-primary">{candidate.source}</p>
                      <p className="text-xs text-brand-300">{candidate.target}</p>
                    </div>
                    <span className="text-[11px] text-text-muted">{Math.round(candidate.confidence * 100)}%</span>
                  </div>
                  {candidateNeedsReviewWarning(candidate) && (
                    <p className="mt-1 text-[11px] text-status-warning">Candidato usado sem revisao gera warning.</p>
                  )}
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    <button data-testid="glossary-confirm-candidate" type="button" onClick={() => confirmGlossaryCandidate(candidate)} className="rounded-md bg-brand px-2 py-1 text-[11px] font-medium text-white">
                      Confirmar
                    </button>
                    <button type="button" onClick={() => editGlossaryCandidate(candidate)} className="rounded-md border border-border px-2 py-1 text-[11px] text-text-secondary hover:text-text-primary">
                      Editar
                    </button>
                    <button data-testid="glossary-reject-candidate" type="button" onClick={() => rejectGlossaryCandidate(candidate)} className="rounded-md border border-border px-2 py-1 text-[11px] text-text-secondary hover:text-status-error">
                      Rejeitar
                    </button>
                    <button type="button" onClick={handleApplyHighConfidenceCandidates} className="rounded-md border border-border px-2 py-1 text-[11px] text-text-secondary hover:text-brand-300">
                      Aplicar em todas as ocorrencias
                    </button>
                    <button type="button" onClick={() => addForbiddenToGlossaryTerm(candidate.source, candidate.target)} className="rounded-md border border-border px-2 py-1 text-[11px] text-text-secondary hover:text-status-warning">
                      Adicionar forbidden
                    </button>
                    <button data-testid="glossary-protect-candidate" type="button" onClick={() => confirmGlossaryCandidate(candidate, true)} className="rounded-md border border-border px-2 py-1 text-[11px] text-text-secondary hover:text-brand-300">
                      Transformar em nome protegido
                    </button>
                  </div>
                </div>
              ))}
              {onlineGlossaryCandidates.length === 0 && (
                <p className="py-2 text-xs text-text-muted">Nenhum candidato online pendente.</p>
              )}
            </div>
          )}

          {glossaryTab === "detected" && (
            <div data-testid="glossary-detected-list" className="space-y-1">
              {detectedChapterTerms.map((term) => (
                <div key={term} className="rounded-lg border border-border bg-bg-tertiary/70 px-3 py-2 text-sm text-text-primary">
                  {term}
                </div>
              ))}
              {detectedChapterTerms.length === 0 && (
                <p className="py-2 text-xs text-text-muted">Nenhum termo detectado no capitulo.</p>
              )}
            </div>
          )}

          {glossaryTab === "rejected" && (
            <div data-testid="glossary-rejected-list" className="space-y-1">
              {rejectedGlossaryCandidates.map((candidate) => (
                <div data-testid="glossary-rejected-row" key={`${candidate.kind}-${candidate.source}`} className="rounded-lg border border-border bg-bg-tertiary/70 px-3 py-2 text-sm text-text-secondary">
                  {candidate.source}
                </div>
              ))}
              {rejectedGlossaryCandidates.length === 0 && (
                <p className="py-2 text-xs text-text-muted">Nenhum termo rejeitado.</p>
              )}
            </div>
          )}

          {glossaryTab === "conflicts" && (
            <div data-testid="glossary-conflicts-list" className="space-y-1">
              {conflictCandidates.map((candidate) => (
                <div key={`${candidate.kind}-${candidate.source}`} className="rounded-lg border border-status-warning/25 bg-status-warning/5 px-3 py-2 text-sm text-text-primary">
                  {candidate.source}: {project.contexto.glossario[candidate.source]} / {candidate.target}
                </div>
              ))}
              {conflictCandidates.length === 0 && (
                <p className="py-2 text-xs text-text-muted">Nenhum conflito.</p>
              )}
            </div>
          )}

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
              {contextWarningKind === "missing_work"
                ? "Nenhuma obra selecionada."
                : "Obra selecionada, mas glossario vazio."}
            </h3>
            <p className="text-sm text-text-secondary mb-4">
              {contextWarningKind === "missing_work"
                ? "A traducao sera feita sem contexto. Isso pode causar erros em nomes, lugares e termos de lore."
                : "Busque contexto online ou adicione termos manualmente."}
            </p>
            <div className="grid grid-cols-1 gap-2">
              <button
                type="button"
                onClick={() => {
                  setShowContextWarning(false);
                  obraInputRef.current?.focus();
                }}
                className="rounded-lg border border-border bg-bg-tertiary px-3 py-2 text-sm text-text-primary hover:border-brand/30 transition-smooth"
              >
                Buscar obra
              </button>
              <button
                type="button"
                onClick={() => {
                  setShowContextWarning(false);
                  const title = obraSearch.trim();
                  if (!title) {
                    setSearchError("Informe o nome da obra para buscar contexto online.");
                    obraInputRef.current?.focus();
                    return;
                  }
                  void handleSearchObra(title);
                }}
                className="rounded-lg border border-border bg-bg-tertiary px-3 py-2 text-sm text-text-primary hover:border-brand/30 transition-smooth"
              >
                Buscar contexto online
              </button>
              <button
                type="button"
                onClick={() => {
                  setShowContextWarning(false);
                  glossaryEditorRef.current?.scrollIntoView({ block: "center", behavior: "smooth" });
                }}
                className="rounded-lg border border-border bg-bg-tertiary px-3 py-2 text-sm text-text-primary hover:border-brand/30 transition-smooth"
              >
                Revisar glossario
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

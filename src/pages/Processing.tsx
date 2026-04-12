import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  CheckCircle2,
  Loader2,
  Circle,
  XCircle,
  TimerReset,
  AlarmClock,
  Flag,
  PauseCircle,
  PlayCircle,
} from "lucide-react";
import { useAppStore, PipelineStep } from "../lib/stores/appStore";
import {
  onPipelineProgress,
  onPipelineComplete,
  cancelPipeline,
  pausePipeline,
  resumePipeline,
  startPipeline,
  loadProjectJson,
} from "../lib/tauri";
import type { PageData } from "../lib/stores/appStore";
import {
  blendRemainingSeconds,
  buildHardwareSummary,
  buildPipelineTimeEstimate,
  formatDuration,
  formatEtaClock,
} from "../lib/time-estimates";

const STEPS: { key: PipelineStep; label: string; description: string }[] = [
  { key: "extract", label: "Extracao", description: "Descompactando e validando arquivos" },
  { key: "ocr", label: "OCR", description: "Detectando texto nos baloes" },
  { key: "context", label: "Contexto", description: "Buscando sinopse e personagens" },
  { key: "translate", label: "Traducao", description: "Traduzindo com contexto local" },
  { key: "inpaint", label: "Inpainting", description: "Removendo texto original" },
  { key: "typeset", label: "Typesetting", description: "Aplicando texto traduzido" },
];

export function Processing() {
  const navigate = useNavigate();
  const {
    project,
    pipeline,
    setPipeline,
    updateProject,
    addRecentProject,
    setupEstimate,
    systemProfile,
  } = useAppStore();
  const startedRef = useRef(false);
  const [started, setStarted] = useState(false);
  const [startedAtMs, setStartedAtMs] = useState<number | null>(null);
  const [nowMs, setNowMs] = useState(() => Date.now());
  const [pauseState, setPauseState] = useState<"running" | "pausing" | "paused" | "resuming">("running");
  const [pausedDurationMs, setPausedDurationMs] = useState(0);
  const pauseStartedAtRef = useRef<number | null>(null);

  useEffect(() => {
    if (!startedAtMs) return;

    const timerId = window.setInterval(() => {
      setNowMs(Date.now());
    }, 1000);

    return () => window.clearInterval(timerId);
  }, [startedAtMs]);

  useEffect(() => {
    // Guard against StrictMode double-invoke: only start the pipeline once.
    if (startedRef.current) return;
    startedRef.current = true;

    let unlistenProgress: (() => void) | undefined;
    let unlistenComplete: (() => void) | undefined;

    // Limpa progresso anterior
    setPipeline(null);

    async function setup() {
      // 1. Registra listeners PRIMEIRO
      unlistenProgress = (await onPipelineProgress((progress) => {
        setPipeline(progress);
      })) as unknown as () => void;

      unlistenComplete = (await onPipelineComplete(async (result) => {
        if (result.success) {
          try {
            // Load project.json and build paginas with absolute paths.
            const raw = await loadProjectJson(result.output_path);
            const outputDir = result.output_path.replace(/\\/g, "/");
            const paginas: PageData[] = (raw.paginas ?? []).map((p) => ({
              numero: p.numero,
              arquivo_original: `${outputDir}/${p.arquivo_original}`.replace(/\\/g, "/"),
              arquivo_traduzido: `${outputDir}/${p.arquivo_traduzido}`.replace(/\\/g, "/"),
              inpaint_blocks: p.inpaint_blocks ?? [],
              textos: p.textos ?? [],
            }));
            updateProject({
              status: "done",
              paginas,
              output_path: outputDir,
              obra: raw.obra || project?.obra || "",
            });
            addRecentProject({
              id: project?.id || outputDir,
              obra: raw.obra || project?.obra || "Projeto sem nome",
              capitulo: raw.capitulo || project?.capitulo || 1,
              pages: paginas.length,
              date: new Date().toISOString(),
              status: "done",
            });
          } catch (e) {
            console.error("Erro ao carregar project.json:", e);
            updateProject({ status: "done" });
          }
          navigate("/preview");
        } else {
          updateProject({ status: "error" });
          alert(`Erro no processamento: ${result.error}`);
          navigate("/");
        }
      })) as unknown as () => void;

      // 2. So inicia o pipeline depois dos listeners prontos
      if (!project) return;
      try {
        const startedAt = Date.now();
        setStartedAtMs(startedAt);
        setNowMs(startedAt);

        await startPipeline({
          source_path: project.source_path,
          obra: project.obra,
          capitulo: project.capitulo,
          idioma_destino: project.idioma_destino,
          qualidade: project.qualidade,
          glossario: project.contexto.glossario,
          contexto: {
            sinopse: project.contexto.sinopse,
            genero: project.contexto.genero,
            personagens: project.contexto.personagens,
            aliases: project.contexto.aliases,
            termos: project.contexto.termos,
            relacoes: project.contexto.relacoes,
            faccoes: project.contexto.faccoes,
            resumo_por_arco: project.contexto.resumo_por_arco,
            memoria_lexical: project.contexto.memoria_lexical,
            fontes_usadas: project.contexto.fontes_usadas,
          },
        });
        setStarted(true);
      } catch (err) {
        setStartedAtMs(null);
        alert(`Erro ao iniciar pipeline: ${err}`);
        navigate("/");
      }
    }

    setup();
    return () => {
      unlistenProgress?.();
      unlistenComplete?.();
    };
  }, []);

  async function handleCancel() {
    if (confirm("Cancelar traducao em andamento?")) {
      await cancelPipeline();
      updateProject({ status: "idle" });
      navigate("/");
    }
  }

  async function handleTogglePause() {
    if (pauseState === "pausing" || pauseState === "resuming") return;

    try {
      if (pauseState === "paused") {
        setPauseState("resuming");
        await resumePipeline();
        const resumedAt = Date.now();
        const pausedAt = pauseStartedAtRef.current;
        if (pausedAt) {
          setPausedDurationMs((current) => current + (resumedAt - pausedAt));
        }
        pauseStartedAtRef.current = null;
        setNowMs(resumedAt);
        setPauseState("running");
        return;
      }

      setPauseState("pausing");
      await pausePipeline();
      pauseStartedAtRef.current = Date.now();
      setNowMs(pauseStartedAtRef.current);
      setPauseState("paused");
    } catch (err) {
      alert(`Erro ao alternar pausa: ${err}`);
      setPauseState(pauseStartedAtRef.current ? "paused" : "running");
    }
  }

  const currentStepIndex = pipeline
    ? STEPS.findIndex((s) => s.key === pipeline.step)
    : 0;

  const initialEstimate =
    setupEstimate ??
    buildPipelineTimeEstimate(
      systemProfile,
      project?.totalPages ?? 0,
      project?.qualidade ?? "normal"
    );
  const activePausedMs = pauseStartedAtRef.current ? Math.max(0, nowMs - pauseStartedAtRef.current) : 0;
  const elapsedSeconds = startedAtMs
    ? Math.max(0, Math.floor((nowMs - startedAtMs - pausedDurationMs - activePausedMs) / 1000))
    : 0;
  const remainingSeconds = pipeline
    ? blendRemainingSeconds({
        initialTotalSeconds: initialEstimate?.total_seconds ?? 0,
        elapsedSeconds,
        progressPercent: pipeline.overall_progress,
        liveEtaSeconds: pipeline.eta_seconds,
      })
    : initialEstimate
    ? Math.max(0, initialEstimate.total_seconds - elapsedSeconds)
    : 0;
  const finishAtLabel = remainingSeconds > 0 ? formatEtaClock(remainingSeconds) : "--:--";
  const hardwareSummary = buildHardwareSummary(systemProfile);
  const isPaused = pauseState === "paused" || pauseState === "pausing";
  const pauseButtonLabel =
    pauseState === "paused"
      ? "Continuar traducao"
      : pauseState === "pausing"
      ? "Pausando..."
      : pauseState === "resuming"
      ? "Continuando..."
      : "Pausar traducao";

  return (
    <div className="p-8 max-w-2xl mx-auto">
      <h2 className="text-xl font-bold mb-1">Traduzindo...</h2>
      <p className="text-sm text-text-secondary mb-8">
        {project?.obra} - Capitulo {project?.capitulo}
      </p>

      {/* Overall progress bar */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium">
            {isPaused
              ? "Pausado"
              : pipeline
              ? `${Math.round(pipeline.overall_progress)}%`
              : started
              ? "Aguardando pipeline..."
              : "Iniciando..."}
          </span>
          <span className="text-xs text-text-secondary">
            {remainingSeconds > 0
              ? `~${formatDuration(remainingSeconds)} restante`
              : initialEstimate
              ? `~${formatDuration(initialEstimate.total_seconds)} estimado`
              : "Calculando..."}
          </span>
        </div>
        <div className="h-2 bg-bg-tertiary rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-accent-purple to-accent-cyan rounded-full
              transition-all duration-500 ease-out"
            style={{ width: `${pipeline?.overall_progress || 0}%` }}
          />
        </div>
        {pipeline && (
          <p className="text-xs text-text-secondary mt-2">
            Pagina {pipeline.current_page}/{pipeline.total_pages}
            {isPaused ? " - processamento pausado em ponto seguro" : ""}
          </p>
        )}
      </div>

      {/* Timing cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4">
        <div className="rounded-xl border border-white/5 bg-bg-secondary px-4 py-3">
          <p className="text-[11px] uppercase tracking-wide text-text-secondary/70 flex items-center gap-1.5">
            <TimerReset size={12} />
            Decorrido
          </p>
          <p className="text-lg font-semibold text-text-primary mt-1">
            {formatDuration(elapsedSeconds)}
          </p>
        </div>

        <div className="rounded-xl border border-white/5 bg-bg-secondary px-4 py-3">
          <p className="text-[11px] uppercase tracking-wide text-text-secondary/70 flex items-center gap-1.5">
            <AlarmClock size={12} />
            Restante
          </p>
          <p className="text-lg font-semibold text-text-primary mt-1">
            {remainingSeconds > 0 ? formatDuration(remainingSeconds) : "--"}
          </p>
        </div>

        <div className="rounded-xl border border-white/5 bg-bg-secondary px-4 py-3">
          <p className="text-[11px] uppercase tracking-wide text-text-secondary/70 flex items-center gap-1.5">
            <Flag size={12} />
            Termino previsto
          </p>
          <p className="text-lg font-semibold text-text-primary mt-1">
            {finishAtLabel}
          </p>
        </div>
      </div>

      <div className="rounded-xl border border-white/5 bg-bg-secondary/60 px-4 py-3 mb-8">
        <p className="text-xs text-text-secondary">
          {initialEstimate
            ? `Base inicial: ~${formatDuration(initialEstimate.total_seconds)} para ${initialEstimate.total_pages} paginas.`
            : "Detectando o hardware para montar a previsao inicial."}
        </p>
        <p className="text-xs text-text-secondary/70 mt-1">
          {hardwareSummary}
        </p>
      </div>

      {/* Steps */}
      <div className="space-y-1 mb-8">
        {STEPS.map((step, i) => {
          const isCurrent = i === currentStepIndex;
          const isDone = i < currentStepIndex;

          return (
            <div
              key={step.key}
              className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-smooth
                ${isCurrent ? "bg-accent-purple/5 border border-accent-purple/20" : ""}
                ${isDone ? "opacity-60" : ""}
              `}
            >
              {/* Icon */}
              {isDone ? (
                <CheckCircle2 size={18} className="text-status-success flex-shrink-0" />
              ) : isCurrent ? (
                <Loader2 size={18} className="text-accent-purple animate-spin flex-shrink-0" />
              ) : (
                <Circle size={18} className="text-text-secondary/30 flex-shrink-0" />
              )}

              {/* Label */}
              <div className="flex-1 min-w-0">
                <p
                  className={`text-sm ${
                    isCurrent
                      ? "text-accent-purple font-medium"
                      : isDone
                      ? "text-text-secondary"
                      : "text-text-secondary/50"
                  }`}
                >
                  {step.label}
                </p>
                <p className="text-[11px] text-text-secondary/60 mt-0.5">
                  {isCurrent && pipeline ? pipeline.message : step.description}
                </p>
              </div>

              {/* Step progress */}
              {isCurrent && pipeline && (
                <span className="text-xs text-accent-purple font-mono">
                  {Math.round(pipeline.step_progress)}%
                </span>
              )}
            </div>
          );
        })}
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <button
          onClick={handleTogglePause}
          disabled={!started || pauseState === "pausing" || pauseState === "resuming"}
          className={`flex items-center gap-2 rounded-full border px-4 py-2 text-sm transition-smooth
            ${
              isPaused
                ? "border-accent-cyan/40 bg-accent-cyan/10 text-accent-cyan hover:bg-accent-cyan/15"
                : "border-accent-purple/35 bg-accent-purple/10 text-accent-purple hover:bg-accent-purple/15"
            }
            disabled:cursor-not-allowed disabled:opacity-50
          `}
        >
          {isPaused ? <PlayCircle size={16} /> : <PauseCircle size={16} />}
          {pauseButtonLabel}
        </button>

        <button
          onClick={handleCancel}
          className="flex items-center gap-2 text-sm text-text-secondary hover:text-status-error transition-smooth"
        >
          <XCircle size={16} />
          Cancelar traducao
        </button>
      </div>
    </div>
  );
}

import { useEffect, useRef, useState } from "react";
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
      className="h-full bg-gradient-to-r from-brand to-accent-cyan rounded-pill transition-all duration-500 ease-out-expo dynamic-progress"
    />
  );
}

function AnimContainer({ name, dur, delay, ease, fill, children, className }: { name: string, dur: string, delay?: string, ease?: string, fill?: string, children?: React.ReactNode, className?: string }) {
  const ref = useDynamicStyle<HTMLDivElement>({
    "--anim-name": name,
    "--anim-dur": dur,
    "--anim-delay": delay || "0s",
    "--anim-ease": ease || "ease",
    "--anim-fill": fill || "none",
  }, [name, dur, delay, ease, fill]);
  return <div ref={ref} className={className}>{children}</div>;
}
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
  Layers,
  Eye,
  Edit3,
  FileDown,
} from "lucide-react";
import { readFile } from "@tauri-apps/plugin-fs";
import { useAppStore, PipelineStep } from "../lib/stores/appStore";
import {
  onPipelineComplete,
  cancelPipeline,
  pausePipeline,
  resumePipeline,
  startPipeline,
  loadProjectJson,
  openLogSaveDialog,
  exportTextFile,
} from "../lib/tauri";
import type { PageData, PipelineLogEntry } from "../lib/stores/appStore";
import {
  blendRemainingSeconds,
  buildHardwareSummary,
  buildPipelineTimeEstimate,
  formatDuration,
  formatEtaClock,
} from "../lib/time-estimates";
import {
  formatFlaggedPages,
  summarizeProcessingQaReport,
  type ProcessingQaSummary,
} from "./processingQa";
import {
  countFlagLogs,
  hardwareUsageLabel,
  pagesPerMinute,
  PERCEIVED_PROCESSING_STEPS,
} from "../lib/processingMetrics";

interface CompletionData {
  obra: string;
  capitulo: number;
  pages: number;
  elapsedSeconds: number;
  firstPagePath: string | null;
  paginas: PageData[];
  qaSummary: ProcessingQaSummary | null;
}

function sanitizeForFilename(value: string): string {
  return value
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^A-Za-z0-9._-]+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "");
}

function buildLogFileName(obra?: string | null, capitulo?: number | null): string {
  const nowIso = new Date().toISOString().replace(/[:]/g, "-").replace(/\..+$/, "");
  const parts: string[] = ["traduzai"];
  if (obra) {
    const slug = sanitizeForFilename(obra);
    if (slug) parts.push(slug);
  }
  if (typeof capitulo === "number" && Number.isFinite(capitulo)) {
    parts.push(`cap${capitulo}`);
  }
  parts.push(nowIso);
  return `${parts.join("_")}.log`;
}

function pad2(value: number): string {
  return value.toString().padStart(2, "0");
}

function formatTimestamp(ms: number): string {
  const d = new Date(ms);
  return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())} ` +
    `${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`;
}

function formatPipelineLog(ctx: {
  obra: string | null;
  capitulo: number | null;
  startedAtMs: number | null;
  finishedAtMs: number | null;
  totalPages: number | null;
  hardware: string | null;
  entries: PipelineLogEntry[];
}): string {
  const lines: string[] = [];
  lines.push("=== TraduzAi — Log da tradução ===");
  lines.push(`Obra: ${ctx.obra ?? "(não informada)"}`);
  lines.push(`Capítulo: ${ctx.capitulo ?? "-"}`);
  lines.push(`Páginas estimadas: ${ctx.totalPages ?? "-"}`);
  lines.push(`Início: ${ctx.startedAtMs ? formatTimestamp(ctx.startedAtMs) : "-"}`);
  lines.push(`Término: ${ctx.finishedAtMs ? formatTimestamp(ctx.finishedAtMs) : "(em andamento)"}`);
  if (ctx.hardware) lines.push(`Hardware: ${ctx.hardware}`);
  lines.push(`Exportado em: ${formatTimestamp(Date.now())}`);
  lines.push(`Total de eventos: ${ctx.entries.length}`);
  lines.push("");
  lines.push("--- Eventos ---");
  for (const entry of ctx.entries) {
    const ts = formatTimestamp(entry.timestamp);
    const level = entry.level.toUpperCase().padEnd(8, " ");
    const step = entry.step ? `[${entry.step}]` : "";
    const pageInfo =
      entry.current_page != null && entry.total_pages != null
        ? ` p${entry.current_page}/${entry.total_pages}`
        : "";
    const overall = entry.overall_progress != null ? ` ${Math.round(entry.overall_progress)}%` : "";
    lines.push(`${ts} ${level}${step}${pageInfo}${overall} ${entry.message}`);
  }
  lines.push("");
  return lines.join("\n");
}

async function loadProcessingQaSummary(outputDir: string): Promise<ProcessingQaSummary | null> {
  try {
    const qaPath = `${outputDir.replace(/\\/g, "/")}/qa_report.json`;
    const bytes = await readFile(qaPath);
    const raw = JSON.parse(new TextDecoder().decode(bytes));
    return summarizeProcessingQaReport(raw);
  } catch {
    return null;
  }
}

const STEPS: { key: PipelineStep; label: string; description: string }[] = [
  { key: "extract", label: "Extração", description: "Descompactando e validando arquivos" },
  { key: "ocr", label: "OCR", description: "Detectando texto nos balões" },
  { key: "context", label: "Contexto", description: "Buscando sinopse e personagens" },
  { key: "translate", label: "Tradução", description: "Traduzindo com contexto local" },
  { key: "inpaint", label: "Inpainting", description: "Removendo texto original" },
  { key: "typeset", label: "Typesetting", description: "Aplicando texto traduzido" },
];

const MANUAL_STEPS: { key: PipelineStep; label: string; description: string }[] = [
  { key: "extract", label: "Preparando arquivos", description: "Copiando imagens e criando estrutura do projeto" },
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
    batchSources,
    setBatchSources,
    pipelineLog,
    appendPipelineLog,
  } = useAppStore();
  const startedRef = useRef(false);
  const [started, setStarted] = useState(false);
  const [startedAtMs, setStartedAtMs] = useState<number | null>(null);
  const [nowMs, setNowMs] = useState(() => Date.now());
  const [pauseState, setPauseState] = useState<"running" | "pausing" | "paused" | "resuming">("running");
  const [pausedDurationMs, setPausedDurationMs] = useState(0);
  const pauseStartedAtRef = useRef<number | null>(null);
  const [completionData, setCompletionData] = useState<CompletionData | null>(null);

  // Estados de Lote
  const [batchIndex, setBatchIndex] = useState(0);
  const [batchCompletedCount, setBatchCompletedCount] = useState(0);

  useEffect(() => {
    if (!startedAtMs) return;

    const timerId = window.setInterval(() => {
      setNowMs(Date.now());
    }, 1000);

    return () => window.clearInterval(timerId);
  }, [startedAtMs]);

  useEffect(() => {
    // Guard against StrictMode double-invoke
    if (startedRef.current) return;
    startedRef.current = true;

    let unlistenComplete: (() => void) | undefined;

    async function processChapter(index: number) {
      if (!project) return;
      const isBatch = batchSources.length > 0;
      const currentPath = isBatch ? batchSources[index] : project.source_path;
      const currentChapter = project.capitulo + (isBatch ? index : 0);

      try {
        setPipeline(null);
        appendPipelineLog({
          level: "info",
          message: `Iniciando capítulo ${currentChapter} — ${currentPath}`,
        });
        await startPipeline({
          source_path: currentPath,
          mode: project.mode,
          obra: project.obra,
          capitulo: currentChapter,
          idioma_origem: project.idioma_origem,
          idioma_destino: project.idioma_destino,
          qualidade: project.qualidade,
          glossario: project.contexto.glossario,
          work_context: project.work_context ?? null,
          preset: project.preset ?? null,
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
        updateProject({ status: "processing" });
        setStarted(true);
      } catch (err) {
        appendPipelineLog({
          level: "error",
          message: `Erro ao iniciar capítulo ${currentChapter}: ${err}`,
        });
        alert(`Erro ao iniciar capítulo ${currentChapter}: ${err}`);
        navigate("/");
      }
    }

    async function setup() {
      // 1. Registra listeners PRIMEIRO
      unlistenComplete = (await onPipelineComplete(async (result) => {
        appendPipelineLog({
          level: result.success ? "success" : "error",
          message: result.success
            ? `Capítulo concluído — saída em ${result.output_path}`
            : `Pipeline falhou: ${result.error ?? "erro desconhecido"}`,
        });
        if (result.success) {
          try {
            const raw = await loadProjectJson(result.output_path);
            const outputDir = result.output_path.replace(/\\/g, "/");
            const qaSummary = await loadProcessingQaSummary(outputDir);
            const paginas: PageData[] = raw.paginas ?? [];

            const chapterNum = raw.capitulo || project?.capitulo || 1;

            addRecentProject({
              id: crypto.randomUUID(),
              obra: raw.obra || project?.obra || "Projeto sem nome",
              capitulo: chapterNum,
              pages: paginas.length,
              date: new Date().toISOString(),
              status: "done",
            });

            const isBatch = batchSources.length > 0;
            const currentBatchIndex = indexRef.current; // Usamos ref para ler o valor atual

            if (isBatch && currentBatchIndex + 1 < batchSources.length) {
              // Avança para o próximo capítulo
              indexRef.current += 1;
              setBatchIndex(indexRef.current);
              setBatchCompletedCount((prev) => prev + 1);
              setTimeout(() => processChapter(indexRef.current), 1000);
            } else {
              // Finalizou tudo
              updateProject({
                status: "done",
                paginas,
                output_path: outputDir,
                obra: raw.obra || project?.obra || "",
                capitulo: chapterNum,
              });
              setBatchSources([]);

              if (project?.mode === "manual") {
                navigate("/editor");
                return;
              }

              const elapsed = startedAtMs
                ? Math.max(0, Math.floor((Date.now() - startedAtMs) / 1000))
                : 0;
              if (qaSummary) {
                appendPipelineLog({
                  level: "info",
                  message:
                    `QA detectou ${qaSummary.flaggedPages.length} página(s) sinalizada(s) ` +
                    `e ${qaSummary.totalDecisions} decisão(ões) registradas.`,
                });
              }
              setCompletionData({
                obra: raw.obra || project?.obra || "Projeto",
                capitulo: chapterNum,
                pages: paginas.length,
                elapsedSeconds: elapsed,
                firstPagePath: paginas[0]?.arquivo_traduzido ?? null,
                paginas,
                qaSummary,
              });
            }
          } catch (e) {
            console.error("Erro no lifecycle de conclusão:", e);
            updateProject({ status: "done" });
            navigate("/preview");
          }
        } else {
          alert(`Erro no processamento: ${result.error}`);
          updateProject({ status: "error" });
          navigate("/");
        }
      })) as unknown as () => void;

      // 2. Inicia o primeiro
      const startedAt = Date.now();
      setStartedAtMs(startedAt);
      setNowMs(startedAt);
      processChapter(0);
    }

    const indexRef = { current: 0 }; // Ref local para controle sequencial
    setup();

    return () => {
      unlistenComplete?.();
    };
  }, []);

  async function handleCancel() {
    if (confirm("Cancelar tradução em andamento?")) {
      appendPipelineLog({ level: "info", message: "Tradução cancelada pelo usuário." });
      await cancelPipeline();
      updateProject({ status: "idle" });
      navigate("/");
    }
  }

  async function handleExportLog() {
    try {
      const suggested = buildLogFileName(project?.obra, project?.capitulo);
      const target = await openLogSaveDialog(suggested);
      if (!target) return;
      const content = formatPipelineLog({
        obra: project?.obra ?? null,
        capitulo: project?.capitulo ?? null,
        startedAtMs,
        finishedAtMs: completionData ? startedAtMs ? startedAtMs + completionData.elapsedSeconds * 1000 : null : null,
        totalPages: project?.totalPages ?? null,
        hardware: hardwareSummary,
        entries: pipelineLog,
      });
      await exportTextFile(target, content);
    } catch (err) {
      alert(`Erro ao exportar log: ${err}`);
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

  const activeSteps = project?.mode === "manual" ? MANUAL_STEPS : STEPS;

  const currentStepIndex = pipeline
    ? activeSteps.findIndex((s) => s.key === pipeline.step)
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
  const ppm = pagesPerMinute(pipeline ?? null, elapsedSeconds);
  const flagLogCount = countFlagLogs(pipelineLog);
  const hardwareUsage = hardwareUsageLabel(systemProfile);
  const isPaused = pauseState === "paused" || pauseState === "pausing";
  const pauseButtonLabel =
    pauseState === "paused"
      ? "Continuar tradução"
      : pauseState === "pausing"
      ? "Pausando..."
      : pauseState === "resuming"
      ? "Continuando..."
      : "Pausar tradução";

  const isBatch = batchSources.length > 1;

  if (completionData) {
    return (
      <ChapterCompletionScreen
        data={completionData}
        onPreview={() => navigate("/preview")}
        onEditor={() => navigate("/editor")}
        onExportLog={handleExportLog}
        logCount={pipelineLog.length}
      />
    );
  }

  return (
    <div className="p-8 max-w-2xl mx-auto animate-fade-in">
      <div className="flex items-center justify-between mb-1">
        <h2 className="text-xl font-bold tracking-tight text-text-primary">
          {project?.mode === "manual" ? "Preparando projeto..." : "Traduzindo..."}
        </h2>
        {isBatch && (
          <div className="flex items-center gap-2 px-3 py-1 rounded-pill bg-brand/8 text-brand-300 text-xs font-medium border border-brand/15">
            <Layers size={14} />
            Lote: {batchIndex + 1} de {batchSources.length}
          </div>
        )}
      </div>
      <p className="text-sm text-text-muted mb-8">
        {project?.obra} - Capítulo {project?.capitulo ? project.capitulo + (isBatch ? batchIndex : 0) : ""}
      </p>

      {/* Batch progress summary */}
      {isBatch && (
        <div data-testid="page-status-grid" className="mb-8 grid grid-cols-2 gap-4">
            <div className="bg-bg-secondary border border-border rounded-xl p-3">
              <p className="text-[10px] text-text-secondary uppercase tracking-wider mb-1">Status do Lote</p>
              <p className="text-sm text-text-primary">
                {batchCompletedCount} concluídos
              </p>
            </div>
            <div className="bg-bg-secondary border border-border rounded-xl p-3">
              <p className="text-[10px] text-text-secondary uppercase tracking-wider mb-1">Atual</p>
              <p className="text-sm text-brand-300 truncate">
                {batchSources[batchIndex]?.split(/[/\\]/).pop()}
              </p>
            </div>
        </div>
      )}

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
        <div className="h-2 bg-bg-tertiary rounded-pill overflow-hidden">
          <ProgressBar progress={pipeline?.overall_progress || 0} />
        </div>
        {pipeline && (
          <p className="text-xs text-text-secondary mt-2">
            Página {pipeline.current_page}/{pipeline.total_pages}
            {isPaused ? " - processamento pausado em ponto seguro" : ""}
          </p>
        )}
      </div>

      {/* Timing cards */}
      {project?.mode !== "manual" && (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4">
          <div className="rounded-xl border border-border bg-bg-secondary px-4 py-3">
            <p className="text-[11px] uppercase tracking-wide text-text-muted flex items-center gap-1.5">
              <TimerReset size={12} />
              Decorrido
            </p>
            <p className="text-lg font-semibold text-text-primary mt-1">
              {formatDuration(elapsedSeconds)}
            </p>
          </div>

          <div className="rounded-xl border border-border bg-bg-secondary px-4 py-3">
            <p className="text-[11px] uppercase tracking-wide text-text-muted flex items-center gap-1.5">
              <AlarmClock size={12} />
              Restante
            </p>
            <p className="text-lg font-semibold text-text-primary mt-1">
              {remainingSeconds > 0 ? formatDuration(remainingSeconds) : "--"}
            </p>
          </div>

          <div className="rounded-xl border border-border bg-bg-secondary px-4 py-3">
            <p className="text-[11px] uppercase tracking-wide text-text-muted flex items-center gap-1.5">
              <Flag size={12} />
              Término previsto
            </p>
            <p className="text-lg font-semibold text-text-primary mt-1">
              {finishAtLabel}
            </p>
          </div>
        </div>
      )}

      {project?.mode !== "manual" && (
        <div className="rounded-xl border border-border bg-bg-secondary/60 px-4 py-3 mb-8">
          <p className="text-xs text-text-secondary">
            {initialEstimate
              ? `Base inicial: ~${formatDuration(initialEstimate.total_seconds)} para ${initialEstimate.total_pages} páginas.`
              : "Detectando o hardware para montar a previsão inicial."}
          </p>
          <p className="text-xs text-text-muted mt-1">
            {hardwareSummary}
          </p>
        </div>
      )}

      <div data-testid="processing-performance-panel" className="mb-8 rounded-xl border border-border bg-bg-secondary p-4">
        <div className="grid grid-cols-2 gap-3 text-xs sm:grid-cols-3">
          <div>
            <p className="text-text-muted">Pagina atual</p>
            <p className="mt-1 text-sm font-medium text-text-primary">{pipeline?.current_page ?? 0}</p>
          </div>
          <div>
            <p className="text-text-muted">Total de paginas</p>
            <p className="mt-1 text-sm font-medium text-text-primary">{pipeline?.total_pages ?? project?.totalPages ?? 0}</p>
          </div>
          <div>
            <p className="text-text-muted">Paginas/minuto</p>
            <p data-testid="processing-pages-per-minute" className="mt-1 text-sm font-medium text-text-primary">{ppm.toFixed(1)}</p>
          </div>
          <div>
            <p className="text-text-muted">Flags encontradas</p>
            <p className="mt-1 text-sm font-medium text-text-primary">{flagLogCount}</p>
          </div>
          <div className="sm:col-span-2">
            <p className="text-text-muted">Uso CPU/GPU</p>
            <p className="mt-1 truncate text-sm font-medium text-text-primary">{hardwareUsage}</p>
          </div>
        </div>
      </div>

      <div data-testid="processing-perceived-steps" className="mb-8 rounded-xl border border-border bg-bg-secondary p-4">
        <p className="mb-3 text-sm font-medium text-text-primary">Etapas detalhadas</p>
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          {PERCEIVED_PROCESSING_STEPS.map((step) => (
            <div key={step} className="rounded-lg border border-border bg-bg-tertiary/70 px-3 py-2 text-xs text-text-secondary">
              {step}
            </div>
          ))}
        </div>
      </div>

      {/* Steps */}
      <div className="space-y-1 mb-8">
        {activeSteps.map((step, i) => {
          const isCurrent = i === currentStepIndex || (project?.mode === "manual" && pipeline?.step !== "extract" && i === 0);
          const isDone = i < currentStepIndex && !(project?.mode === "manual" && i === 0);

          return (
            <div
              key={step.key}
              className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-smooth
                ${isCurrent ? "bg-brand/5 border border-brand/15" : ""}
                ${isDone ? "opacity-60" : ""}
              `}
            >
              {/* Icon */}
              {isDone ? (
                <CheckCircle2 size={18} className="text-status-success flex-shrink-0" />
              ) : isCurrent ? (
                <Loader2 size={18} className="text-brand-300 animate-spin flex-shrink-0" />
              ) : (
                <Circle size={18} className="text-text-secondary/30 flex-shrink-0" />
              )}

              {/* Label */}
              <div className="flex-1 min-w-0">
                <p
                  className={`text-sm ${
                    isCurrent
                      ? "text-brand-300 font-medium"
                      : isDone
                      ? "text-text-secondary"
                      : "text-text-secondary/50"
                  }`}
                >
                  {step.label}
                </p>
                <p className="text-[11px] text-text-muted mt-0.5">
                  {isCurrent && pipeline ? pipeline.message : step.description}
                </p>
              </div>

              {/* Step progress */}
              {isCurrent && pipeline && (
                <span className="text-xs text-brand-300 font-mono">
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
          className={`flex items-center gap-2 rounded-pill border px-4 py-2 text-sm transition-smooth
            ${
              isPaused
                ? "border-accent-cyan/25 bg-accent-cyan/8 text-accent-cyan hover:bg-accent-cyan/12"
                : "border-brand/25 bg-brand/8 text-brand-300 hover:bg-brand/12"
            }
            disabled:cursor-not-allowed disabled:opacity-50
          `}
        >
          {isPaused ? <PlayCircle size={16} /> : <PauseCircle size={16} />}
          {pauseButtonLabel}
        </button>

        <button
          onClick={handleExportLog}
          disabled={pipelineLog.length === 0}
          className="flex items-center gap-2 rounded-pill border border-border bg-bg-secondary px-4 py-2 text-sm text-text-secondary transition-smooth hover:border-border-strong hover:text-text-primary disabled:cursor-not-allowed disabled:opacity-50"
          title={pipelineLog.length === 0 ? "Nada registrado ainda" : `Exportar ${pipelineLog.length} eventos`}
        >
          <FileDown size={16} />
          Exportar log
        </button>

        <button
          onClick={handleCancel}
          className="flex items-center gap-2 text-sm text-text-secondary hover:text-status-error transition-smooth"
        >
          <XCircle size={16} />
          Cancelar tradução
        </button>
      </div>
    </div>
  );
}

function ChapterCompletionScreen({
  data,
  onPreview,
  onEditor,
  onExportLog,
  logCount,
}: {
  data: CompletionData;
  onPreview: () => void;
  onEditor: () => void;
  onExportLog: () => void;
  logCount: number;
}) {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const prevBlobRef = useRef<string | null>(null);

  useEffect(() => {
    const path = data.firstPagePath;
    if (!path) return;
    let cancelled = false;
    readFile(path)
      .then((bytes) => {
        if (cancelled) return;
        const blob = new Blob([bytes], { type: "image/jpeg" });
        const url = URL.createObjectURL(blob);
        if (prevBlobRef.current) URL.revokeObjectURL(prevBlobRef.current);
        prevBlobRef.current = url;
        setImageSrc(url);
      })
      .catch(() => {
        if (!cancelled) setImageSrc(null);
      });
    return () => {
      cancelled = true;
    };
  }, [data.firstPagePath]);

  useEffect(() => {
    return () => {
      if (prevBlobRef.current) URL.revokeObjectURL(prevBlobRef.current);
    };
  }, []);

  return (
    <div className="relative flex min-h-full flex-col items-center justify-center overflow-hidden p-8">
      <style>{`
        @keyframes successPop {
          from { opacity: 0; transform: scale(0.35); }
          to { opacity: 1; transform: scale(1); }
        }
        @keyframes pageZoomIn {
          from { opacity: 0; transform: scale(0.72) translateY(28px); }
          to { opacity: 1; transform: scale(1) translateY(0); }
        }
        @keyframes fadeSlideUp {
          from { opacity: 0; transform: translateY(14px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>

      {/* Ambient glow */}
      <div className="pointer-events-none absolute inset-x-0 top-0 h-72 bg-[radial-gradient(ellipse_at_top,_rgba(72,225,120,0.10),_transparent_55%)]" />

      <div className="relative w-full max-w-md">
        {/* Success icon */}
        <div className="mb-6 flex justify-center">
          <AnimContainer
            name="successPop"
            dur="0.55s"
            ease="cubic-bezier(0.34,1.56,0.64,1)"
            fill="forwards"
            className="flex h-16 w-16 items-center justify-center rounded-pill bg-status-success/15 shadow-[0_0_32px_rgba(72,225,120,0.18)] dynamic-animation"
          >
            <CheckCircle2 size={32} className="text-status-success" />
          </AnimContainer>
        </div>

        {/* Title */}
        <AnimContainer
          name="fadeSlideUp"
          dur="0.4s"
          delay="0.12s"
          ease="ease-out"
          fill="both"
          className="mb-6 text-center dynamic-animation"
        >
          <h2 className="text-2xl font-semibold text-text-primary">Capítulo concluído!</h2>
          <p className="mt-1 text-text-secondary">
            {data.obra} · Capítulo {data.capitulo}
          </p>
        </AnimContainer>

        {/* Stats */}
        <AnimContainer
          name="fadeSlideUp"
          dur="0.4s"
          delay="0.22s"
          ease="ease-out"
          fill="both"
          className="mb-5 grid grid-cols-2 gap-3 dynamic-animation"
        >
          <div data-testid="page-status-grid" className="rounded-xl border border-border bg-bg-secondary px-4 py-3 text-center">
            <p className="text-2xl font-semibold tabular text-text-primary">{data.pages}</p>
            <p className="mt-1 text-xs text-text-secondary">páginas traduzidas</p>
          </div>
          <div className="rounded-xl border border-border bg-bg-secondary px-4 py-3 text-center">
            <p className="text-2xl font-semibold tabular text-text-primary">{formatDuration(data.elapsedSeconds)}</p>
            <p className="mt-1 text-xs text-text-secondary">tempo total</p>
          </div>
        </AnimContainer>

        {data.qaSummary && (
          <AnimContainer
            name="fadeSlideUp"
            dur="0.4s"
            delay="0.28s"
            ease="ease-out"
            fill="both"
            className="mb-5 dynamic-animation"
          >
            <div
              data-testid="qa-panel"
              className="rounded-2xl border border-amber-400/15 bg-[linear-gradient(180deg,rgba(245,158,11,0.09),rgba(255,255,255,0.02))] px-4 py-4"
            >
              <div className="flex items-center gap-2 text-sm font-medium text-text-primary">
                <Flag size={15} className="text-amber-300" />
                RevisÃ£o sugerida
              </div>
              <p className="mt-2 text-xs text-text-secondary">
                {data.qaSummary.totalDecisions} decisÃ£o(Ãµes) registradas. PÃ¡ginas sinalizadas:{" "}
                {formatFlaggedPages(data.qaSummary.flaggedPages)}.
              </p>
              {data.qaSummary.topReasons.length > 0 && (
                <div className="mt-3 flex flex-wrap gap-2">
                  {data.qaSummary.topReasons.map((item) => (
                    <span
                      data-testid="qa-flag-item"
                      key={item.reason}
                      className="rounded-pill border border-border bg-bg-secondary/70 px-3 py-1 text-[11px] text-text-secondary"
                    >
                      {item.label}: {item.count}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </AnimContainer>
        )}

        {/* First page preview with zoom animation */}
        {imageSrc && (
          <AnimContainer
            name="pageZoomIn"
            dur="0.65s"
            delay="0.32s"
            ease="cubic-bezier(0.34,1.56,0.64,1)"
            fill="both"
            className="mb-5 overflow-hidden rounded-2xl border border-border shadow-[0_20px_50px_rgba(0,0,0,0.5)] dynamic-animation"
          >
            <img
              src={imageSrc}
              alt="Primeira página traduzida"
              className="max-h-64 w-full object-contain"
            />
          </AnimContainer>
        )}

        {/* Action buttons */}
        <AnimContainer
          name="fadeSlideUp"
          dur="0.4s"
          delay="0.42s"
          ease="ease-out"
          fill="both"
          className="flex gap-3 dynamic-animation"
        >
          <button
            onClick={onPreview}
            className="flex flex-1 items-center justify-center gap-2 rounded-xl bg-brand py-3 text-sm font-medium text-white transition-smooth hover:bg-brand/90"
          >
            <Eye size={16} />
            Ver Preview
          </button>
          <button
            onClick={onEditor}
            className="flex flex-1 items-center justify-center gap-2 rounded-xl border border-border bg-bg-secondary py-3 text-sm text-text-primary transition-smooth hover:border-border-strong"
          >
            <Edit3 size={16} />
            Abrir Editor
          </button>
        </AnimContainer>

        <AnimContainer
          name="fadeSlideUp"
          dur="0.4s"
          delay="0.52s"
          ease="ease-out"
          fill="both"
          className="mt-3 flex justify-center dynamic-animation"
        >
          <button
            data-testid="export-report-link"
            onClick={onExportLog}
            disabled={logCount === 0}
            className="flex items-center gap-2 rounded-xl border border-border bg-bg-secondary/70 px-4 py-2 text-xs text-text-secondary transition-smooth hover:border-border-strong hover:text-text-primary disabled:cursor-not-allowed disabled:opacity-50"
            title={logCount === 0 ? "Sem registros" : `Exportar ${logCount} eventos`}
          >
            <FileDown size={14} />
            Exportar log da tradução
          </button>
        </AnimContainer>
      </div>
    </div>
  );
}

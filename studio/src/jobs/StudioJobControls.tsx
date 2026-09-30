import { useEffect, useMemo, useState } from "react";
import { CirclePause, Play, RotateCcw, Square, Workflow } from "lucide-react";

import type { StudioProject } from "../project/studioProject";
import {
  applyProjectEvent,
  createStudioJobState,
  explainStudioIpcError,
  type ProjectEvent,
  type ProjectEventStatus,
  type StudioJobState,
} from "../backend/studioIpcV1";
import type { StudioIpcClient } from "../backend/studioIpcClient";
import {
  createTauriStudioIpcClient,
  subscribeToProjectEvents,
} from "../backend/studioIpcTauri";

export interface CanonicalStudioProjectIdentity {
  projectId: string;
  chapterId: string;
  projectPath: string;
  expectedRevision: number;
}

export function canonicalStudioProjectIdentity(
  project: StudioProject,
  projectPath: string,
): CanonicalStudioProjectIdentity | { error: string } {
  if (!project.id?.trim()) return { error: "Projeto sem ID canônico" };
  const chapterId = String(project.chapter_id ?? project.capitulo ?? "").trim();
  if (!chapterId) return { error: "Projeto sem ID de capítulo" };
  const revision = project.project_revision;
  if (!Number.isInteger(revision) || Number(revision) < 0) {
    return { error: "Projeto sem revisão canônica" };
  }
  return {
    projectId: project.id,
    chapterId,
    projectPath,
    expectedRevision: Number(revision),
  };
}

const STATUS_LABELS: Record<ProjectEventStatus, string> = {
  queued: "Na fila",
  running: "Em execução",
  pausing: "Pausando",
  paused: "Pausado",
  cancelling: "Cancelando",
  cancelled: "Cancelado",
  failed: "Falhou",
  blocked: "Revisão necessária",
  awaiting_review: "Aguardando revisão",
  completed: "Concluído",
};

const STAGE_LABELS: Record<string, string> = {
  queued: "Preparação",
  import: "Importação",
  analysis: "Análise",
  ocr: "OCR",
  translation: "Tradução",
  restoration: "Restauração",
  render: "Renderização",
  review: "Revisão",
  export: "Exportação",
  complete: "Finalização",
};

export type StudioJobAction = "pause" | "resume" | "cancel" | "retry";

function objectRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

export function promotedProjectPath(
  event: Pick<ProjectEvent, "reason_code" | "payload">,
): string | null {
  if (event.reason_code !== "physical_pipeline_completed") return null;
  const detail = objectRecord(event.payload.detail);
  const outputPath = detail?.output_project_path;
  return typeof outputPath === "string" && outputPath.trim() ? outputPath : null;
}

function newestPersistedJob(project: StudioProject, projectId: string): Record<string, unknown> | null {
  const integration = objectRecord(project.integration_v1);
  const jobs = objectRecord(integration?.jobs);
  if (!jobs) return null;
  return Object.values(jobs)
    .map(objectRecord)
    .filter((job): job is Record<string, unknown> => (
      job !== null
      && typeof job.job_id === "string"
      && job.job_id.trim().length > 0
      && job.project_id === projectId
    ))
    .sort((left, right) => {
      const revisionDelta = Number(right.project_revision ?? -1) - Number(left.project_revision ?? -1);
      if (revisionDelta !== 0) return revisionDelta;
      return Number(right.sequence ?? -1) - Number(left.sequence ?? -1);
    })[0] ?? null;
}

const PROJECT_EVENT_STATUSES = new Set<ProjectEventStatus>(Object.keys(STATUS_LABELS) as ProjectEventStatus[]);

export function persistedStudioJobState(
  project: StudioProject,
  identity: CanonicalStudioProjectIdentity,
): StudioJobState | null {
  const nested = newestPersistedJob(project, identity.projectId);
  const snapshot = nested ?? project;
  const jobId = typeof snapshot.job_id === "string" ? snapshot.job_id.trim() : "";
  if (!jobId) return null;
  const persistedStatus = nested?.status ?? project.job_status;
  const status = typeof persistedStatus === "string"
    && PROJECT_EVENT_STATUSES.has(persistedStatus as ProjectEventStatus)
    ? persistedStatus as ProjectEventStatus
    : "paused";
  const persistedExpectedRevision = nested?.expected_revision ?? project.job_expected_revision;
  const expectedRevision = Number.isInteger(persistedExpectedRevision)
    && Number(persistedExpectedRevision) >= 0
    ? Number(persistedExpectedRevision)
    : identity.expectedRevision;
  const persistedSequence = nested?.sequence ?? project.job_sequence;
  const lastSequence = Number.isInteger(persistedSequence) && Number(persistedSequence) >= 0
    ? Number(persistedSequence)
    : -1;
  const persistedPayload = objectRecord(nested?.payload ?? project.job_payload) ?? {};
  const units = objectRecord(nested?.units);
  const terminalUnitIds = units
    ? Object.entries(units)
      .filter(([, unit]) => objectRecord(unit)?.terminal === true)
      .map(([unitId]) => unitId)
    : [];
  const payload = Array.isArray(persistedPayload.terminal_unit_ids)
    ? persistedPayload
    : { ...persistedPayload, ...(terminalUnitIds.length > 0 ? { terminal_unit_ids: terminalUnitIds } : {}) };
  return {
    ...createStudioJobState({
      jobId,
      projectId: identity.projectId,
      expectedRevision,
    }),
    projectRevision: identity.expectedRevision,
    lastSequence,
    stage: typeof (nested?.stage ?? project.job_stage) === "string"
      ? String(nested?.stage ?? project.job_stage)
      : "queued",
    status,
    reasonCode: typeof (nested?.reason_code ?? project.job_reason_code) === "string"
      ? String(nested?.reason_code ?? project.job_reason_code)
      : "job_state_reopened",
    payload,
  };
}

export function newestStudioJobState(
  current: StudioJobState | null,
  persisted: StudioJobState | null,
): StudioJobState | null {
  if (!persisted) return current;
  if (!current || current.jobId !== persisted.jobId) return persisted;
  if (persisted.projectRevision > current.projectRevision) return persisted;
  if (persisted.projectRevision === current.projectRevision && persisted.lastSequence > current.lastSequence) {
    return persisted;
  }
  return current;
}

function actionButtons(status: ProjectEventStatus): StudioJobAction[] {
  if (status === "queued" || status === "running") return ["pause", "cancel"];
  if (status === "paused") return ["resume", "cancel"];
  if (
    status === "failed"
    || status === "cancelled"
    || status === "blocked"
    || status === "awaiting_review"
  ) return ["retry"];
  return [];
}

const ACTION_LABELS: Record<StudioJobAction, string> = {
  pause: "Pausar",
  resume: "Retomar",
  cancel: "Cancelar",
  retry: "Tentar novamente",
};

export function StudioJobStatusCard({
  state,
  busy,
  error,
  onAction,
}: {
  state: StudioJobState;
  busy: boolean;
  error: string | null;
  onAction: (action: StudioJobAction) => void;
}) {
  const actions = actionButtons(state.status);
  return (
    <section className="min-w-[220px] rounded-lg border border-border bg-bg-tertiary/60 px-2.5 py-1.5" aria-live="polite">
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <p className="truncate text-[10px] font-semibold text-text-primary">
            {STAGE_LABELS[state.stage] ?? state.stage}
          </p>
          <p className="truncate text-[9px] text-text-muted">
            {STATUS_LABELS[state.status]} · {state.reasonCode.replaceAll("_", " ")}
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-1">
          {actions.map((action) => (
            <button
              key={action}
              type="button"
              disabled={busy}
              onClick={() => onAction(action)}
              className="flex items-center gap-1 rounded-md border border-border px-1.5 py-1 text-[9px] text-text-secondary transition-smooth hover:bg-white/5 hover:text-text-primary disabled:opacity-40"
            >
              {action === "pause" && <CirclePause size={10} />}
              {action === "resume" && <Play size={10} />}
              {action === "cancel" && <Square size={10} />}
              {action === "retry" && <RotateCcw size={10} />}
              {ACTION_LABELS[action]}
            </button>
          ))}
        </div>
      </div>
      {error && <p role="alert" className="mt-1 text-[9px] text-status-error">{error}</p>}
    </section>
  );
}

function hasTauriRuntime() {
  return typeof window !== "undefined"
    && ("__TAURI_INTERNALS__" in window || "__TAURI__" in window);
}

function idempotencyKey(action: string, revision: number) {
  const nonce = typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return `${action}-r${revision}-${nonce}`;
}

export function StudioJobControls({
  project,
  projectPath,
  client = createTauriStudioIpcClient(),
  onPersisted,
}: {
  project: StudioProject;
  projectPath: string;
  client?: StudioIpcClient;
  onPersisted?: (projectPath?: string) => void | Promise<void>;
}) {
  const identity = useMemo(
    () => canonicalStudioProjectIdentity(project, projectPath),
    [project, projectPath],
  );
  const [state, setState] = useState<StudioJobState | null>(() => {
    if ("error" in identity) return null;
    return persistedStudioJobState(project, identity);
  });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const desktop = hasTauriRuntime();

  useEffect(() => {
    if ("error" in identity) return;
    const persisted = persistedStudioJobState(project, identity);
    setState((current) => newestStudioJobState(current, persisted));
  }, [identity, project]);

  useEffect(() => {
    if (!desktop || "error" in identity) return undefined;
    let disposed = false;
    let unsubscribe: (() => void) | undefined;
    void subscribeToProjectEvents((event) => {
      if (disposed || event.project_id !== identity.projectId) return;
      setState((current) => {
        if (!current || current.jobId !== event.job_id) return current;
        return applyProjectEvent(current, event);
      });
      const outputPath = promotedProjectPath(event);
      if (outputPath) {
        void Promise.resolve(onPersisted?.(outputPath)).catch((reloadError) => {
          if (!disposed) setError(explainStudioIpcError(reloadError));
        });
      }
    }, (invalidEvent) => setError(invalidEvent.message)).then((stop) => {
      if (disposed) stop();
      else unsubscribe = stop;
    }).catch((subscriptionError) => {
      if (!disposed) setError(explainStudioIpcError(subscriptionError));
    });
    return () => {
      disposed = true;
      unsubscribe?.();
    };
  }, [desktop, identity, onPersisted]);

  const start = async () => {
    if (!desktop) {
      setError("Processamento disponível apenas no aplicativo desktop");
      return;
    }
    if ("error" in identity) {
      setError(identity.error);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const result = await client.startConsumerFast({
        projectPath: identity.projectPath,
        chapterId: identity.chapterId,
        expectedRevision: identity.expectedRevision,
        idempotencyKey: idempotencyKey("start", identity.expectedRevision),
      });
      setState({
        ...createStudioJobState({
          jobId: result.job_id,
          projectId: identity.projectId,
          expectedRevision: identity.expectedRevision,
        }),
        projectRevision: result.project_revision,
      });
      await onPersisted?.();
    } catch (startError) {
      setError(explainStudioIpcError(startError));
    } finally {
      setBusy(false);
    }
  };

  const runAction = async (action: StudioJobAction) => {
    if (!state || busy) return;
    setBusy(true);
    setError(null);
    try {
      const input = {
        jobId: state.jobId,
        expectedRevision: state.projectRevision,
        idempotencyKey: idempotencyKey(action, state.projectRevision),
      };
      const result = action === "pause"
        ? await client.pauseConsumerFast(input)
        : action === "resume"
          ? await client.resumeConsumerFast(input)
          : action === "cancel"
            ? await client.cancelConsumerFast(input)
            : await client.retryConsumerFast({
              ...input,
              terminalUnitIds: Array.isArray(state.payload.terminal_unit_ids)
                ? state.payload.terminal_unit_ids.map(String)
                : [],
            });
      setState((current) => current && current.jobId === result.job_id ? {
        ...current,
        projectRevision: result.project_revision,
        status: result.status as ProjectEventStatus,
        stage: typeof result.stage === "string" ? result.stage : current.stage,
        reasonCode: typeof result.reason_code === "string" ? result.reason_code : current.reasonCode,
        lastSequence: Number.isInteger(result.sequence) ? Number(result.sequence) : current.lastSequence,
      } : current);
      await onPersisted?.();
    } catch (actionError) {
      setError(explainStudioIpcError(actionError));
    } finally {
      setBusy(false);
    }
  };

  if (state) {
    return <StudioJobStatusCard state={state} busy={busy} error={error} onAction={(action) => void runAction(action)} />;
  }

  const capabilityError = !desktop
    ? "Processamento disponível no app desktop"
    : "error" in identity
      ? identity.error
      : null;
  return (
    <div className="flex items-center gap-2">
      <button
        type="button"
        disabled={busy || Boolean(capabilityError)}
        onClick={() => void start()}
        className="flex items-center gap-1 rounded-lg border border-accent/35 bg-accent/10 px-2.5 py-1 text-[10px] font-medium text-accent transition-smooth hover:bg-accent/15 disabled:cursor-not-allowed disabled:opacity-40"
        title={capabilityError ?? "Executar o Consumer Fast neste capítulo"}
      >
        <Workflow size={11} />
        {busy ? "Iniciando…" : "Processar capítulo"}
      </button>
      {error && <span role="alert" className="max-w-56 truncate text-[9px] text-status-error">{error}</span>}
    </div>
  );
}

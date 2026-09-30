import type { ProjectEvent, ReviewDecision } from "./studioIpcV1";

export interface StudioIpcTransport {
  invoke<T>(command: string, args: Record<string, unknown>): Promise<T>;
}

export interface MutationIdentity {
  expectedRevision: number;
  idempotencyKey: string;
}

export interface JobControlInput extends MutationIdentity {
  jobId: string;
}

export interface JobStartResult {
  job_id: string;
  project_revision: number;
}

export interface JobControlResult extends JobStartResult {
  status: string;
  stage?: string;
  reason_code?: string;
  sequence?: number;
}

export interface ExportDecision {
  allowed: boolean;
  status: string;
  gate_allowed: boolean;
  verified: boolean;
  completion_status: string;
  output_review_state: string;
  blocker_count: number;
  critical_issue_count: number;
  critical_flag_count: number;
  review_issue_count: number;
  review_flag_count: number;
  issue_count: number;
}

export interface RendererPreferenceComparisonReadResult {
  owner_id: string;
  project_revision: number;
  state: string | null;
  comparison: unknown | null;
  artifact_base: string | null;
}

export function createStudioIpcClient(transport: StudioIpcTransport) {
  const controlJob = (command: string, input: JobControlInput) => (
    transport.invoke<JobControlResult>(command, {
      jobId: input.jobId,
      expectedRevision: input.expectedRevision,
      idempotencyKey: input.idempotencyKey,
    })
  );

  return {
    startConsumerFast(input: MutationIdentity & { projectPath: string; chapterId: string }) {
      return transport.invoke<JobStartResult>("start_consumer_fast", {
        projectPath: input.projectPath,
        chapterId: input.chapterId,
        expectedRevision: input.expectedRevision,
        idempotencyKey: input.idempotencyKey,
      });
    },

    cancelConsumerFast(input: JobControlInput) {
      return controlJob("cancel_consumer_fast", input);
    },

    pauseConsumerFast(input: JobControlInput) {
      return controlJob("pause_consumer_fast", input);
    },

    resumeConsumerFast(input: JobControlInput) {
      return controlJob("resume_consumer_fast", input);
    },

    retryConsumerFast(input: JobControlInput & { terminalUnitIds: string[] }) {
      return transport.invoke<JobControlResult>("retry_consumer_fast", {
        jobId: input.jobId,
        terminalUnitIds: [...input.terminalUnitIds],
        expectedRevision: input.expectedRevision,
        idempotencyKey: input.idempotencyKey,
      });
    },

    persistProjectEvent(input: MutationIdentity & { event: ProjectEvent }) {
      return transport.invoke<{ event_id: string; project_revision: number }>(
        "persist_project_event",
        {
          event: input.event,
          expectedRevision: input.expectedRevision,
          idempotencyKey: input.idempotencyKey,
        },
      );
    },

    retypesetOwner(input: MutationIdentity & {
      projectPath: string;
      ownerId: string;
      layoutRequest: Record<string, unknown>;
    }) {
      return transport.invoke<{
        project_revision: number;
        recipe_receipt: Record<string, unknown>;
        raster_artifact_ref: Record<string, unknown>;
      }>("retypeset_owner", {
        projectPath: input.projectPath,
        ownerId: input.ownerId,
        layoutRequest: input.layoutRequest,
        expectedRevision: input.expectedRevision,
        idempotencyKey: input.idempotencyKey,
      });
    },

    submitReviewDecision(input: MutationIdentity & {
      projectPath: string;
      decision: ReviewDecision;
    }) {
      if (
        input.decision.expected_revision !== input.expectedRevision
        || input.decision.idempotency_key !== input.idempotencyKey
      ) {
        return Promise.reject(new Error("A identidade da ReviewDecision diverge do comando de persistência"));
      }
      return transport.invoke<{
        decision_id: string;
        project_revision: number;
        owner_status: string;
      }>("submit_review_decision", {
        projectPath: input.projectPath,
        decision: input.decision,
        expectedRevision: input.expectedRevision,
        idempotencyKey: input.idempotencyKey,
      });
    },

    readRendererPreferenceComparison(input: {
      projectPath: string;
      ownerId: string;
      expectedRevision: number;
    }) {
      return transport.invoke<RendererPreferenceComparisonReadResult>(
        "read_renderer_preference_comparison",
        {
          projectPath: input.projectPath,
          ownerId: input.ownerId,
          expectedRevision: input.expectedRevision,
        },
      );
    },

    decideExport(input: { projectPath: string; expectedRevision: number }) {
      return transport.invoke<{ export_decision: ExportDecision }>("decide_export", {
        projectPath: input.projectPath,
        expectedRevision: input.expectedRevision,
      });
    },

    approveFinalReview(input: MutationIdentity & {
      projectPath: string;
      actorKind: "human";
      actorId: string;
    }) {
      return transport.invoke<{
        project_revision: number;
        decision: Record<string, unknown>;
        export_decision: ExportDecision;
      }>("approve_final_review", {
        projectPath: input.projectPath,
        expectedRevision: input.expectedRevision,
        idempotencyKey: input.idempotencyKey,
        actorKind: input.actorKind,
        actorId: input.actorId,
      });
    },

    exportFinal(input: MutationIdentity & { projectPath: string; destination: string }) {
      return transport.invoke<{
        project_revision: number;
        export_manifest: Record<string, unknown>;
        publication_receipt: Record<string, unknown>;
      }>("export_final", {
        projectPath: input.projectPath,
        destination: input.destination,
        expectedRevision: input.expectedRevision,
        idempotencyKey: input.idempotencyKey,
      });
    },

    exportDiagnostic(input: { projectPath: string; destination: string; expectedRevision: number }) {
      return transport.invoke<{ diagnostic_manifest: Record<string, unknown> }>(
        "export_diagnostic",
        {
          projectPath: input.projectPath,
          destination: input.destination,
          expectedRevision: input.expectedRevision,
        },
      );
    },
  };
}

export type StudioIpcClient = ReturnType<typeof createStudioIpcClient>;

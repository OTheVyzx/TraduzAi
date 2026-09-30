export type ProjectEventStatus =
  | "queued"
  | "running"
  | "pausing"
  | "paused"
  | "cancelling"
  | "cancelled"
  | "failed"
  | "blocked"
  | "awaiting_review"
  | "completed";

export interface ProjectEvent {
  job_id: string;
  project_id: string;
  expected_revision: number;
  project_revision: number;
  sequence: number;
  stage: string;
  status: ProjectEventStatus;
  reason_code: string;
  payload: Record<string, unknown>;
}

export interface StudioJobState {
  jobId: string;
  projectId: string;
  expectedRevision: number;
  projectRevision: number;
  lastSequence: number;
  stage: string;
  status: ProjectEventStatus;
  reasonCode: string;
  payload: Record<string, unknown>;
}

export interface CreateStudioJobStateInput {
  jobId: string;
  projectId: string;
  expectedRevision: number;
}

export function createStudioJobState(input: CreateStudioJobStateInput): StudioJobState {
  return {
    jobId: input.jobId,
    projectId: input.projectId,
    expectedRevision: input.expectedRevision,
    projectRevision: input.expectedRevision,
    lastSequence: -1,
    stage: "queued",
    status: "queued",
    reasonCode: "job_queued",
    payload: {},
  };
}

const TERMINAL_JOB_STATES = new Set<ProjectEventStatus>([
  "cancelled",
  "failed",
  "completed",
]);

export function applyProjectEvent(
  state: StudioJobState,
  event: ProjectEvent,
): StudioJobState {
  if (
    event.job_id !== state.jobId
    || event.project_id !== state.projectId
    || event.expected_revision !== state.expectedRevision
    || TERMINAL_JOB_STATES.has(state.status)
    || event.project_revision < state.projectRevision
    || (
      event.project_revision === state.projectRevision
      && event.sequence <= state.lastSequence
    )
  ) {
    return state;
  }

  return {
    ...state,
    projectRevision: event.project_revision,
    lastSequence: event.sequence,
    stage: event.stage,
    status: event.status,
    reasonCode: event.reason_code,
    payload: event.payload,
  };
}

export type ReviewActorKind = "human" | "model" | "system";

export type RendererPreferenceChoice = "A" | "B" | "equivalent" | "neither" | "unsure";

export interface RendererPreferenceResponsePayload {
  schema: "traduzai.renderer-preference-response.v1";
  comparison_sha256: string;
  choice: RendererPreferenceChoice;
  selected_candidate_id: string | null;
  displayed_candidate_ids: { A: string; B: string };
  candidate_recipe_sha256s: [string, string];
  candidate_output_sha256s: [string, string];
  target_sha256: string;
  randomization_sha256: string;
  actor_kind: ReviewActorKind;
  actor_id: string;
  recorded_at: string;
  training_eligible: boolean;
}

export interface ReviewDecision {
  project_id: string;
  owner_id: string;
  expected_revision: number;
  actor_kind: ReviewActorKind;
  actor_id: string;
  decision: string;
  reason_code: string;
  evidence_sha256s: string[];
  idempotency_key: string;
  preference_response?: RendererPreferenceResponsePayload;
}

export interface BuildReviewDecisionInput {
  projectId: string;
  ownerId: string;
  expectedRevision: number;
  actorKind: ReviewActorKind;
  actorId: string;
  decision: string;
  reasonCode: string;
  evidenceSha256s: string[];
  idempotencyKey: string;
  preferenceResponse?: RendererPreferenceResponsePayload;
}

function isSha256(value: string): boolean {
  return /^[0-9a-f]{64}$/.test(value);
}

function validatePreferenceResponse(
  response: RendererPreferenceResponsePayload,
  input: BuildReviewDecisionInput,
) {
  if (response.schema !== "traduzai.renderer-preference-response.v1") {
    throw new Error("Resposta de preferência do Renderer incompatível");
  }
  if (response.actor_kind !== input.actorKind || response.actor_id !== input.actorId) {
    throw new Error("Ator da preferência diverge da decisão de revisão");
  }
  if (response.training_eligible && response.actor_kind !== "human") {
    throw new Error("Somente preferência humana pode ser elegível para aprendizado");
  }
  const displayedIds = Object.values(response.displayed_candidate_ids);
  if (new Set(displayedIds).size !== 2 || displayedIds.some((id) => !id.startsWith("renderer-candidate:"))) {
    throw new Error("Candidatas exibidas na preferência são inválidas");
  }
  const expectedSelected = response.choice === "A" || response.choice === "B"
    ? response.displayed_candidate_ids[response.choice]
    : null;
  if (response.selected_candidate_id !== expectedSelected) {
    throw new Error("Candidata selecionada diverge da escolha exibida");
  }
  const boundEvidence = [
    response.comparison_sha256,
    response.target_sha256,
    response.randomization_sha256,
    ...response.candidate_recipe_sha256s,
    ...response.candidate_output_sha256s,
  ];
  if (boundEvidence.some((hash) => !isSha256(hash) || !input.evidenceSha256s.includes(hash))) {
    throw new Error("A decisão não vincula toda evidência da preferência do Renderer");
  }
  if (!response.recorded_at || Number.isNaN(Date.parse(response.recorded_at))) {
    throw new Error("Horário da preferência é inválido");
  }
  if (!/(?:Z|[+-]\d{2}:\d{2})$/.test(response.recorded_at)) {
    throw new Error("Horário da preferência exige fuso horário explícito");
  }
}

export function explainStudioIpcError(error: unknown) {
  const message = error instanceof Error ? error.message : String(error);
  if (message.includes("PROJECT_REVISION_CONFLICT")) {
    return "O projeto foi alterado por outra operação. Recarregue a revisão mais recente e tente novamente; suas edições locais foram preservadas.";
  }
  return message;
}

export function buildReviewDecision(input: BuildReviewDecisionInput): ReviewDecision {
  if (
    input.expectedRevision < 0
    || input.evidenceSha256s.length === 0
    || input.evidenceSha256s.some((hash) => !isSha256(hash))
  ) {
    throw new Error("A decisão de revisão exige revisão válida e evidência SHA-256");
  }

  if (input.preferenceResponse) validatePreferenceResponse(input.preferenceResponse, input);

  return {
    project_id: input.projectId,
    owner_id: input.ownerId,
    expected_revision: input.expectedRevision,
    actor_kind: input.actorKind,
    actor_id: input.actorId,
    decision: input.decision,
    reason_code: input.reasonCode,
    evidence_sha256s: [...input.evidenceSha256s],
    idempotency_key: input.idempotencyKey,
    ...(input.preferenceResponse ? { preference_response: input.preferenceResponse } : {}),
  };
}

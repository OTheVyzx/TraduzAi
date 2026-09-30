import { describe, expect, it } from "vitest";

import {
  applyProjectEvent,
  buildReviewDecision,
  createStudioJobState,
  explainStudioIpcError,
  type ProjectEvent,
} from "../studioIpcV1";

const event = (overrides: Partial<ProjectEvent> = {}): ProjectEvent => ({
  job_id: "job-001",
  project_id: "project-001",
  expected_revision: 4,
  project_revision: 4,
  sequence: 0,
  stage: "analysis",
  status: "running",
  reason_code: "provider_started",
  payload: {},
  ...overrides,
});

describe("traduzai.studio-ipc.v1", () => {
  it("ignores an event from another job or project", () => {
    const state = createStudioJobState({
      jobId: "job-001",
      projectId: "project-001",
      expectedRevision: 4,
    });

    expect(applyProjectEvent(state, event({ job_id: "job-old" }))).toBe(state);
    expect(applyProjectEvent(state, event({ project_id: "project-old" }))).toBe(state);
  });

  it("ignores stale revision and sequence without regressing visible progress", () => {
    const initial = createStudioJobState({
      jobId: "job-001",
      projectId: "project-001",
      expectedRevision: 4,
    });
    const advanced = applyProjectEvent(initial, event({
      project_revision: 5,
      sequence: 8,
      stage: "render",
      status: "running",
    }));

    expect(applyProjectEvent(advanced, event({
      project_revision: 5,
      sequence: 7,
      stage: "translation",
    }))).toBe(advanced);
    expect(applyProjectEvent(advanced, event({
      project_revision: 4,
      sequence: 99,
      stage: "analysis",
    }))).toBe(advanced);
  });

  it("keeps a cancelled job terminal when a late completion arrives", () => {
    const initial = createStudioJobState({
      jobId: "job-001",
      projectId: "project-001",
      expectedRevision: 4,
    });
    const cancelled = applyProjectEvent(initial, event({
      sequence: 3,
      status: "cancelled",
      reason_code: "cancelled_by_user",
    }));
    const lateCompletion = event({
      sequence: 4,
      stage: "complete",
      status: "completed",
      reason_code: "provider_completed",
    });

    expect(applyProjectEvent(cancelled, lateCompletion)).toBe(cancelled);
    expect(cancelled.status).toBe("cancelled");
  });

  it("builds a review decision with the canonical revision and idempotency fields", () => {
    expect(buildReviewDecision({
      projectId: "project-001",
      ownerId: "owner-007",
      expectedRevision: 12,
      actorKind: "human",
      actorId: "local-user",
      decision: "accept_candidate",
      reasonCode: "visual_review_passed",
      evidenceSha256s: ["a".repeat(64)],
      idempotencyKey: "review-owner-007-r12",
    })).toEqual({
      project_id: "project-001",
      owner_id: "owner-007",
      expected_revision: 12,
      actor_kind: "human",
      actor_id: "local-user",
      decision: "accept_candidate",
      reason_code: "visual_review_passed",
      evidence_sha256s: ["a".repeat(64)],
      idempotency_key: "review-owner-007-r12",
    });
  });

  it("identifies automated review as system test/synthetic, never as human", () => {
    expect(buildReviewDecision({
      projectId: "project-001",
      ownerId: "owner-007",
      expectedRevision: 12,
      actorKind: "system",
      actorId: "test/synthetic",
      decision: "accept_candidate",
      reasonCode: "synthetic_acceptance_test",
      evidenceSha256s: ["b".repeat(64)],
      idempotencyKey: "preference-owner-007-r12",
    }).actor_kind).toBe("system");
  });

  it("binds a renderer preference response to the Studio-owned review decision", () => {
    const hashes = "abcdef1".split("").map((value) => value.repeat(64));
    const preferenceResponse = {
      schema: "traduzai.renderer-preference-response.v1" as const,
      comparison_sha256: hashes[0],
      choice: "B" as const,
      selected_candidate_id: `renderer-candidate:${"2".repeat(32)}`,
      displayed_candidate_ids: {
        A: `renderer-candidate:${"1".repeat(32)}`,
        B: `renderer-candidate:${"2".repeat(32)}`,
      },
      candidate_recipe_sha256s: [hashes[3], hashes[4]] as [string, string],
      candidate_output_sha256s: [hashes[5], hashes[6]] as [string, string],
      target_sha256: hashes[1],
      randomization_sha256: hashes[2],
      actor_kind: "human" as const,
      actor_id: "local-user",
      recorded_at: "2026-09-27T14:30:00-03:00",
      training_eligible: false,
    };

    const decision = buildReviewDecision({
      projectId: "project-001",
      ownerId: "owner-007",
      expectedRevision: 12,
      actorKind: "human",
      actorId: "local-user",
      decision: "accept_candidate",
      reasonCode: "renderer_preference",
      evidenceSha256s: hashes,
      idempotencyKey: "preference-owner-007-r12",
      preferenceResponse,
    });

    expect(decision.preference_response).toEqual(preferenceResponse);
  });

  it("rejects a renderer preference timestamp without an explicit timezone", () => {
    const hashes = "abcdef1".split("").map((value) => value.repeat(64));
    expect(() => buildReviewDecision({
      projectId: "project-001",
      ownerId: "owner-007",
      expectedRevision: 12,
      actorKind: "human",
      actorId: "local-user",
      decision: "accept_candidate",
      reasonCode: "renderer_preference",
      evidenceSha256s: hashes,
      idempotencyKey: "preference-owner-007-r12",
      preferenceResponse: {
        schema: "traduzai.renderer-preference-response.v1",
        comparison_sha256: hashes[0],
        choice: "A",
        selected_candidate_id: `renderer-candidate:${"1".repeat(32)}`,
        displayed_candidate_ids: {
          A: `renderer-candidate:${"1".repeat(32)}`,
          B: `renderer-candidate:${"2".repeat(32)}`,
        },
        candidate_recipe_sha256s: [hashes[3], hashes[4]],
        candidate_output_sha256s: [hashes[5], hashes[6]],
        target_sha256: hashes[1],
        randomization_sha256: hashes[2],
        actor_kind: "human",
        actor_id: "local-user",
        recorded_at: "2026-09-27T14:30:00",
        training_eligible: false,
      },
    })).toThrow("fuso horário");
  });

  it("rejects a review decision without valid evidence", () => {
    expect(() => buildReviewDecision({
      projectId: "project-001",
      ownerId: "owner-007",
      expectedRevision: 12,
      actorKind: "human",
      actorId: "local-user",
      decision: "accept_candidate",
      reasonCode: "visual_review_passed",
      evidenceSha256s: [],
      idempotencyKey: "review-owner-007-r12",
    })).toThrow("evidência");
  });

  it("turns revision conflicts into an actionable message without discarding edits", () => {
    expect(explainStudioIpcError(new Error("PROJECT_REVISION_CONFLICT: expected 8, current 9")))
      .toContain("edições locais foram preservadas");
    expect(explainStudioIpcError("provider unavailable")).toBe("provider unavailable");
  });
});
